import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, PackedSequence
import random
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import json

# Load configuration
with open('train_model_config.json', 'r') as f:
    CONFIG = json.load(f)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SimpleLSTMPredictor(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        self.hidden_size = config['model']['hidden_size']
        self.num_layers = config['model']['num_layers']
        self.dropout = config['model']['dropout']
        
        # Input normalization
        self.layer_norm = nn.LayerNorm(input_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers,
                           batch_first=True, dropout=self.dropout if self.num_layers > 1 else 0,
                           bidirectional=True)
        
        # Output layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        )
    
    def forward(self, x, lengths):
        # Input normalization
        batch_size, seq_len, features = x.size()
        x = x.view(-1, features)
        x = self.layer_norm(x)
        x = x.view(batch_size, seq_len, features)
        
        # Pack sequence for LSTM
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Process through LSTM
        packed_out, _ = self.lstm(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        # Use the last output for each sequence
        idx = (lengths - 1).view(-1, 1).expand(-1, self.hidden_size * 2).unsqueeze(1)
        out = out.gather(1, idx).squeeze(1)
        
        # Final prediction
        return self.fc_layers(out)

class WormDataset(Dataset):
    def __init__(self, features, lifespans, lengths):
        self.features = features
        self.lifespans = lifespans
        self.lengths = lengths
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], float(self.lifespans[idx]), int(self.lengths[idx])

def collate_fn(batch):
    # Sort batch by sequence length (required for pack_padded_sequence)
    batch.sort(key=lambda x: x[2], reverse=True)
    
    # Separate features, lifespans, and lengths
    features = [torch.FloatTensor(x[0]) for x in batch]
    lifespans = torch.FloatTensor([x[1] for x in batch]).reshape(-1, 1)
    lengths = torch.LongTensor([x[2] for x in batch])
    
    # Pad sequences
    features = pad_sequence(features, batch_first=True)
    
    return features, lifespans, lengths

def load_data_from_directory(directory, max_frame=None):
    """
    Load preprocessed features from a directory.
    Args:
        directory: Directory containing the preprocessed feature files
        max_frame: If set, only use clusters up to this frame number
    """
    all_features = []
    lifespans = []
    lengths = []
    file_paths = []
    cluster_counts = []  # Store cluster counts for each file
    groups = []  # Store the group name
    
    files = [f for f in os.listdir(directory) if f.endswith('_features.npz')]
    for filename in tqdm(files, desc=f"Loading {os.path.basename(directory)}", leave=False):
        filepath = os.path.join(directory, filename)
        data = np.load(filepath, allow_pickle=True)
        total_clusters = len(data['features'])
        
        if max_frame is not None:
            # Load the original CSV to get frame numbers
            original_dir = directory.replace('Lifespan_calculated', 'Lifespan_cleaned')
            csv_path = os.path.join(original_dir, data['source_file'].item())
            df = pd.read_csv(csv_path)
            
            # Group by cluster and check max frame for each cluster
            cluster_max_frames = df.groupby('Cluster')['Frame'].max()
            valid_clusters = cluster_max_frames[cluster_max_frames <= max_frame].index
            
            # Filter features
            features = data['features']
            cluster_features = []
            for i, cluster_id in enumerate(range(len(features))):
                if cluster_id in valid_clusters:
                    cluster_features.append(features[i])
            
            if len(cluster_features) > 0:  # Only add if we have valid clusters
                cluster_features = np.array(cluster_features)
                all_features.append(cluster_features)
                lifespans.append(float(data['num_frames']))
                lengths.append(len(cluster_features))
                file_paths.append(filepath)
                cluster_counts.append((os.path.basename(filepath), total_clusters, len(cluster_features)))
                groups.append(os.path.basename(directory))  # Store the group name
        else:
            # Use all features
            all_features.append(data['features'])
            lifespans.append(float(data['num_frames']))
            lengths.append(int(len(data['features'])))
            file_paths.append(filepath)
            cluster_counts.append((os.path.basename(filepath), total_clusters, total_clusters))
            groups.append(os.path.basename(directory))  # Store the group name
    
    return all_features, lifespans, lengths, file_paths, cluster_counts, groups

def evaluate_model(model, val_loader, criterion, device, config):
    scale_factor = config['training']['scale_factor']
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    total_percent_error = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch_features, batch_lifespans, batch_lengths in val_loader:
            batch_features = batch_features.to(device)
            batch_lifespans = batch_lifespans.to(device)
            batch_lengths = batch_lengths.to(device)
            
            outputs = model(batch_features, batch_lengths)
            loss = criterion(outputs, batch_lifespans)
            total_loss += loss.item()
            
            # Scale back predictions and targets
            outputs_np = outputs.cpu().numpy() * scale_factor
            batch_lifespans_np = batch_lifespans.cpu().numpy() * scale_factor
            
            # Calculate percentage errors
            percent_errors = np.abs(outputs_np - batch_lifespans_np) / batch_lifespans_np * 100
            total_percent_error += np.sum(percent_errors)
            num_samples += len(outputs_np)
            
            all_predictions.extend(outputs_np)
            all_targets.extend(batch_lifespans_np)
    
    avg_percent_error = total_percent_error / num_samples
    return total_loss / len(val_loader), np.array(all_predictions), np.array(all_targets), avg_percent_error

def save_training_data(train_files, all_features, all_lifespans, all_lengths, 
                      val_files, val_features, val_lifespans, val_lengths,
                      feature_scaler):
    """Save training data and metadata for inspection."""
    
    # Save training data summary
    train_summary = []
    for file, features, lifespan, length in zip(train_files, all_features, all_lifespans, all_lengths):
        summary = {
            'file': file,
            'num_clusters': length,
            'lifespan': int(lifespan),
            'min_feature_values': features.min(axis=0).tolist(),
            'max_feature_values': features.max(axis=0).tolist(),
            'mean_feature_values': features.mean(axis=0).tolist()
        }
        train_summary.append(summary)
    
    # Save validation data summary
    val_summary = []
    for file, features, lifespan, length in zip(val_files, val_features, val_lifespans, val_lengths):
        summary = {
            'file': file,
            'num_clusters': length,
            'lifespan': int(lifespan),
            'min_feature_values': features.min(axis=0).tolist(),
            'max_feature_values': features.max(axis=0).tolist(),
            'mean_feature_values': features.mean(axis=0).tolist()
        }
        val_summary.append(summary)
    
    # Get feature names from first file
    data = np.load(train_files[0], allow_pickle=True)
    feature_names = data['feature_names'].tolist()
    
    # Save all metadata
    metadata = {
        'feature_names': feature_names,
        'num_training_samples': len(train_files),
        'num_validation_samples': len(val_files),
        'feature_scaler_mean': feature_scaler.mean_.tolist(),
        'feature_scaler_scale': feature_scaler.scale_.tolist(),
        'lifespan_scale_factor': 100000,
        'training_data': train_summary,
        'validation_data': val_summary
    }
    
    # Save to file
    with open('training_data_summary.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nTraining data summary saved to training_data_summary.json")

def plot_losses(train_losses, val_losses, train_errors, val_errors, save_path='training_metrics.png'):
    """Plot and save training and validation losses and percentage errors."""
    plt.figure(figsize=(15, 6))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    
    # Plot percentage errors
    plt.subplot(1, 2, 2)
    plt.plot(train_errors, label='Training Error')
    plt.plot(val_errors, label='Validation Error')
    plt.xlabel('Epoch')
    plt.ylabel('Average Absolute Percentage Error')
    plt.title('Training and Validation Percentage Errors')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_predictions(model, train_loader, val_loader, train_files, val_files, 
                    criterion, device, config):
    scale_factor = config['training']['scale_factor']
    """Save predictions for all samples to a CSV file."""
    model.eval()
    all_predictions = []
    
    # Process training data
    print("\nGenerating predictions for training set...")
    _, train_preds, train_targets, train_error = evaluate_model(model, train_loader, criterion, device, config)
    train_mape = 0
    for file, pred, target in zip(train_files, train_preds, train_targets):
        percent_error = float(abs(pred[0] - target[0]) / target[0] * 100)
        train_mape += percent_error
        all_predictions.append({
            'file': os.path.basename(file),
            'set': 'train',
            'predicted_lifespan': int(pred[0]),
            'actual_lifespan': int(target[0]),
            'absolute_error': int(abs(pred[0] - target[0])),
            'percent_error': percent_error
        })
    train_mape /= len(train_files)
    
    # Process validation data
    print("Generating predictions for validation set...")
    _, val_preds, val_targets, val_error = evaluate_model(model, val_loader, criterion, device, config)
    val_mape = 0
    for file, pred, target in zip(val_files, val_preds, val_targets):
        percent_error = float(abs(pred[0] - target[0]) / target[0] * 100)
        val_mape += percent_error
        all_predictions.append({
            'file': os.path.basename(file),
            'set': 'validation',
            'predicted_lifespan': int(pred[0]),
            'actual_lifespan': int(target[0]),
            'absolute_error': int(abs(pred[0] - target[0])),
            'percent_error': percent_error
        })
    val_mape /= len(val_files)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(all_predictions)
    
    # Add summary statistics
    summary_stats = pd.DataFrame([{
        'file': 'SUMMARY_TRAIN',
        'set': 'train',
        'predicted_lifespan': df[df['set'] == 'train']['predicted_lifespan'].mean(),
        'actual_lifespan': df[df['set'] == 'train']['actual_lifespan'].mean(),
        'absolute_error': df[df['set'] == 'train']['absolute_error'].mean(),
        'percent_error': train_mape
    }, {
        'file': 'SUMMARY_VALIDATION',
        'set': 'validation',
        'predicted_lifespan': df[df['set'] == 'validation']['predicted_lifespan'].mean(),
        'actual_lifespan': df[df['set'] == 'validation']['actual_lifespan'].mean(),
        'absolute_error': df[df['set'] == 'validation']['absolute_error'].mean(),
        'percent_error': val_mape
    }])
    
    df = pd.concat([df, summary_stats], ignore_index=True)
    
    # Save to CSV
    output_file = 'lifespan_predictions.csv'
    df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")
    
    # Print detailed summary statistics
    print("\nDetailed Summary Statistics:")
    print("\nTraining Set Statistics:")
    print(f"Number of samples: {len(train_files)}")
    print(f"Mean Absolute Error: {df[df['set'] == 'train']['absolute_error'].mean():.0f} frames")
    print(f"Mean Absolute Percentage Error (MAPE): {train_mape:.2f}%")
    print(f"Standard Deviation of Percentage Error: {df[df['set'] == 'train']['percent_error'].std():.2f}%")
    print(f"Min Percentage Error: {df[df['set'] == 'train']['percent_error'].min():.2f}%")
    print(f"Max Percentage Error: {df[df['set'] == 'train']['percent_error'].max():.2f}%")
    
    print("\nValidation Set Statistics:")
    print(f"Number of samples: {len(val_files)}")
    print(f"Mean Absolute Error: {df[df['set'] == 'validation']['absolute_error'].mean():.0f} frames")
    print(f"Mean Absolute Percentage Error (MAPE): {val_mape:.2f}%")
    print(f"Standard Deviation of Percentage Error: {df[df['set'] == 'validation']['percent_error'].std():.2f}%")
    print(f"Min Percentage Error: {df[df['set'] == 'validation']['percent_error'].min():.2f}%")
    print(f"Max Percentage Error: {df[df['set'] == 'validation']['percent_error'].max():.2f}%")

def train_fold(train_loader, val_loader, input_size, config, device, fold_num=None):
    """Train a single fold and return the best model and metrics."""
    model = SimpleLSTMPredictor(input_size, config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config['training']['scheduler']['factor'], 
        patience=config['training']['scheduler']['patience'], 
        verbose=True
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    train_errors = []
    val_errors = []
    
    fold_desc = f"Fold {fold_num}" if fold_num is not None else "Training"
    for epoch in tqdm(range(config['training']['num_epochs']), desc=f"{fold_desc} epochs"):
        # Training phase
        model.train()
        total_loss = 0
        for batch_features, batch_lifespans, batch_lengths in train_loader:
            batch_features = batch_features.to(device)
            batch_lifespans = batch_lifespans.to(device)
            batch_lengths = batch_lengths.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features, batch_lengths)
            loss = criterion(outputs, batch_lifespans)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['training']['gradient_clip'])
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss/len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluate training and validation errors
        _, _, _, train_error = evaluate_model(model, train_loader, criterion, device, config)
        val_loss, _, _, val_error = evaluate_model(model, val_loader, criterion, device, config)
        train_errors.append(train_error)
        val_losses.append(val_loss)
        val_errors.append(val_error)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        
        if (epoch + 1) % 50 == 0:  # Print status less frequently for k-fold
            tqdm.write(f'{fold_desc} - Epoch [{epoch+1}/{config["training"]["num_epochs"]}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
            tqdm.write(f'Train Error: {train_error:.1f}%, Val Error: {val_error:.1f}%')
    
    # Load best model state
    model.load_state_dict(best_model_state)
    
    # Save fold plots
    plot_fold_metrics(train_losses, val_losses, train_errors, val_errors, fold_num)
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_errors': train_errors,
        'val_errors': val_errors
    }

def plot_fold_metrics(train_losses, val_losses, train_errors, val_errors, fold_num, save_dir='plots'):
    """Plot and save training metrics for a single fold."""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 6))
    
    # Plot losses with log scale
    plt.subplot(1, 2, 1)
    plt.semilogy(train_losses, label='Training Loss')  # Use semilogy for log scale
    plt.semilogy(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title(f'Fold {fold_num} - Training and Validation Losses')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)  # Add grid for both major and minor ticks
    
    # Plot percentage errors (keep linear scale)
    plt.subplot(1, 2, 2)
    plt.plot(train_errors, label='Training Error')
    plt.plot(val_errors, label='Validation Error')
    plt.xlabel('Epoch')
    plt.ylabel('Average Absolute Percentage Error')
    plt.title(f'Fold {fold_num} - Training and Validation Errors')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'fold_{fold_num}_metrics.png'))
    plt.close()

def plot_all_folds_metrics(all_fold_metrics, save_dir='plots'):
    """Plot and save aggregate metrics across all folds."""
    plt.figure(figsize=(15, 6))
    
    # Plot losses with log scale
    plt.subplot(1, 2, 1)
    for fold, metrics in enumerate(all_fold_metrics):
        plt.semilogy(metrics['val_losses'], label=f'Fold {fold+1}')  # Use semilogy for log scale
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss (log scale)')
    plt.title('Validation Losses Across All Folds')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)  # Add grid for both major and minor ticks
    
    # Plot errors (keep linear scale)
    plt.subplot(1, 2, 2)
    for fold, metrics in enumerate(all_fold_metrics):
        plt.plot(metrics['val_errors'], label=f'Fold {fold+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation MAPE (%)')
    plt.title('Validation Errors Across All Folds')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_folds_metrics.png'))
    plt.close()

def k_fold_cross_validation(features, lifespans, lengths, files, groups, feature_scaler, config, input_size):
    """Perform stratified k-fold cross-validation and return detailed metrics."""
    k = config['training']['num_folds']
    scale_factor = config['training']['scale_factor']
    batch_size = config['training']['batch_size']
    # Group indices by treatment group
    group_indices = {}
    for i, group in enumerate(groups):
        if group not in group_indices:
            group_indices[group] = []
        group_indices[group].append(i)
    
    # Shuffle indices within each group
    for group in group_indices:
        random.shuffle(group_indices[group])
    
    # Calculate fold sizes for each group
    group_fold_sizes = {group: len(indices) // k for group, indices in group_indices.items()}
    
    # Store metrics for each fold
    fold_metrics_list = []  # Initialize the list to store fold metrics
    all_predictions = []
    
    print(f"\nStarting {k}-fold stratified cross-validation...")
    print("\nGroup distribution:")
    for group, indices in group_indices.items():
        print(f"{group}: {len(indices)} samples, {group_fold_sizes[group]} per fold")
    
    all_fold_metrics = []
    
    for fold in range(k):
        print(f"\nTraining Fold {fold + 1}/{k}")
        
        # Create validation indices for this fold from each group
        val_indices = []
        train_indices = []
        
        for group, indices in group_indices.items():
            fold_size = group_fold_sizes[group]
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < k - 1 else len(indices)
            
            # Get validation indices for this group
            group_val_indices = indices[start_idx:end_idx]
            val_indices.extend(group_val_indices)
            
            # Get training indices for this group
            group_train_indices = [idx for idx in indices if idx not in group_val_indices]
            train_indices.extend(group_train_indices)
        
        # Print fold distribution
        print("\nFold distribution:")
        val_groups = [groups[i] for i in val_indices]
        for group in set(groups):
            count = val_groups.count(group)
            print(f"{group} validation samples: {count}")
        
        # Prepare data for this fold
        train_features = [features[i] for i in train_indices]
        train_lifespans = [lifespans[i] for i in train_indices]
        train_lengths = [lengths[i] for i in train_indices]
        train_files = [files[i] for i in train_indices]
        
        val_features = [features[i] for i in val_indices]
        val_lifespans = [lifespans[i] for i in val_indices]
        val_lengths = [lengths[i] for i in val_indices]
        val_files = [files[i] for i in val_indices]
        
        # Scale features
        scaled_train_features = [feature_scaler.transform(f) for f in train_features]
        scaled_val_features = [feature_scaler.transform(f) for f in val_features]
        
        # Scale lifespans
        scale_factor = 100000
        scaled_train_lifespans = np.array(train_lifespans) / scale_factor
        scaled_val_lifespans = np.array(val_lifespans) / scale_factor
        
        # Create dataloaders
        train_dataset = WormDataset(scaled_train_features, scaled_train_lifespans, train_lengths)
        val_dataset = WormDataset(scaled_val_features, scaled_val_lifespans, val_lengths)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, collate_fn=collate_fn)
        
        # Train model for this fold
        model, fold_metrics = train_fold(
            train_loader, val_loader, input_size, config, 
            device, fold + 1
        )
        all_fold_metrics.append(fold_metrics)
        
        # Evaluate final performance
        val_loss, predictions, targets, val_error = evaluate_model(model, val_loader, nn.MSELoss(), device, config)
        
        # Calculate detailed metrics for this fold
        fold_pred_data = []
        for file, pred, target in zip(val_files, predictions, targets):
            abs_error = abs(pred[0] - target[0])
            pct_error = abs_error / target[0] * 100
            fold_pred_data.append({
                'file': os.path.basename(file),
                'fold': fold + 1,
                'predicted': int(pred[0]),
                'actual': int(target[0]),
                'abs_error': int(abs_error),
                'pct_error': float(pct_error)
            })
        
        # Store fold metrics
        fold_metrics = {
            'fold': fold + 1,
            'val_loss': val_loss,
            'mae': np.mean([d['abs_error'] for d in fold_pred_data]),
            'mape': np.mean([d['pct_error'] for d in fold_pred_data]),
            'std_ae': np.std([d['abs_error'] for d in fold_pred_data]),
            'std_pe': np.std([d['pct_error'] for d in fold_pred_data]),
            'num_samples': len(val_indices)
        }
        fold_metrics_list.append(fold_metrics)
        
        all_predictions.extend(fold_pred_data)
    
    # Calculate and print overall metrics
    print("\nCross-Validation Results:")
    print("\nPer-Fold Metrics:")
    print(f"{'Fold':^6} {'MAE (frames)':^15} {'MAPE (%)':^12} {'Std AE':^12} {'Std PE (%)':^12} {'Samples':^8}")
    print("-" * 70)
    
    for metrics in fold_metrics_list:
        print(f"{metrics['fold']:^6} {metrics['mae']:>13.0f} {metrics['mape']:>11.2f} "
              f"{metrics['std_ae']:>11.0f} {metrics['std_pe']:>11.2f} {metrics['num_samples']:^8}")
    
    # Calculate overall statistics
    overall_mae = np.mean([m['mae'] for m in fold_metrics_list])
    overall_mape = np.mean([m['mape'] for m in fold_metrics_list])
    std_mae = np.std([m['mae'] for m in fold_metrics_list])
    std_mape = np.std([m['mape'] for m in fold_metrics_list])
    
    print("\nOverall Cross-Validation Metrics:")
    print(f"Mean Absolute Error: {overall_mae:.0f} ± {std_mae:.0f} frames")
    print(f"Mean Absolute Percentage Error: {overall_mape:.2f}% ± {std_mape:.2f}%")
    
    # Save detailed predictions to CSV
    df = pd.DataFrame(all_predictions)
    df.to_csv('cross_validation_predictions.csv', index=False)
    print("\nDetailed predictions saved to cross_validation_predictions.csv")
    
    # Plot aggregate metrics
    plot_all_folds_metrics(all_fold_metrics)
    
    return fold_metrics_list, all_predictions

def main():
    # Set random seed for reproducibility
    random.seed(CONFIG['random_seed'])
    torch.manual_seed(CONFIG['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG['random_seed'])
    
    # Load data paths from config
    base_dir = CONFIG['data']['base_dir']
    subdirs = CONFIG['data']['subdirs']
    
    # Load all data
    all_features = []
    all_lifespans = []
    all_lengths = []
    all_files = []
    all_groups = []
    
    # Get maximum frame number from config
    max_frame = CONFIG['training']['max_frame']
    
    print("Loading data...")
    print(f"Using clusters up to frame {max_frame}" if max_frame is not None else "Using all clusters")
    
    all_cluster_counts = []
    for subdir in tqdm(subdirs, desc="Processing directories"):
        dir_path = os.path.join(base_dir, subdir)
        features, lifespans, lengths, file_paths, cluster_counts, groups = load_data_from_directory(dir_path, max_frame=max_frame)
        all_cluster_counts.extend(cluster_counts)
        
        all_features.extend(features)
        all_lifespans.extend(lifespans)
        all_lengths.extend(lengths)
        all_files.extend(file_paths)
        all_groups.extend(groups)
    
    # Print cluster counts
    print("\nCluster counts for each file:")
    print(f"{'File':<60} {'Total Clusters':>15} {'Used Clusters':>15} {'Percent Used':>15}")
    print("-" * 105)
    for filename, total, used in sorted(all_cluster_counts):
        percent = (used / total * 100) if total > 0 else 0
        print(f"{filename:<60} {total:>15} {used:>15} {percent:>14.1f}%")
    
    # Standardize features
    all_features_flat = np.vstack([f for f in all_features])
    feature_scaler = StandardScaler()
    feature_scaler.fit(all_features_flat)
    
    # Get input size from features
    input_size = all_features[0].shape[1]
    
    # Perform k-fold cross-validation
    k_fold_cross_validation(
        all_features, all_lifespans, all_lengths, all_files, all_groups,
        feature_scaler, CONFIG, input_size
    )

if __name__ == "__main__":
    main() 