import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import random
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import json

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, hidden_states, mask=None):
        attention_weights = self.attention(hidden_states)
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended = torch.sum(attention_weights * hidden_states, dim=1)
        return attended, attention_weights

class ResidualLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(ResidualLSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        
        # If input and output sizes don't match, use a linear projection
        self.projection = None
        if input_size != hidden_size * 2:
            self.projection = nn.Linear(input_size, hidden_size * 2)
    
    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        # Apply residual connection if shapes match
        if self.projection is not None:
            residual = self.projection(x)
        else:
            residual = x
            
        out = self.dropout(out)
        out = self.norm(out + residual)
        return out

class LSTMLifespanPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMLifespanPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input normalization and embedding
        self.batch_norm_input = nn.BatchNorm1d(input_size)
        self.input_embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Stack of residual LSTM blocks
        self.lstm_stack = nn.ModuleList([
            ResidualLSTMBlock(hidden_size if i == 0 else hidden_size * 2,
                             hidden_size, 2, dropout)
            for i in range(3)  # 3 blocks of stacked LSTMs
        ])
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            AttentionLayer(hidden_size * 2)
            for _ in range(4)  # 4 attention heads
        ])
        
        # Combine attention heads
        self.attention_combine = nn.Sequential(
            nn.Linear(hidden_size * 8, hidden_size * 4),
            nn.LayerNorm(hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Deep MLP for final prediction
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, x, lengths):
        # Input normalization
        batch_size, seq_len, features = x.size()
        x = x.view(-1, features)
        x = self.batch_norm_input(x)
        x = x.view(batch_size, seq_len, features)
        
        # Input embedding
        x = self.input_embedding(x)
        
        # Create attention mask based on lengths
        mask = torch.arange(seq_len).expand(batch_size, seq_len).to(x.device) < lengths.unsqueeze(1)
        
        # Process through LSTM stack
        for lstm_block in self.lstm_stack:
            x = lstm_block(x, lengths)
        
        # Multi-head attention
        attention_outputs = []
        for attention in self.attention_layers:
            attended, _ = attention(x, mask)
            attention_outputs.append(attended)
        
        # Combine attention heads
        combined = torch.cat(attention_outputs, dim=1)
        combined = self.attention_combine(combined)
        
        # Final prediction
        return self.fc_layers(combined)

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
        else:
            # Use all features
            all_features.append(data['features'])
            lifespans.append(float(data['num_frames']))
            lengths.append(int(len(data['features'])))
            file_paths.append(filepath)
            cluster_counts.append((os.path.basename(filepath), total_clusters, total_clusters))
    
    return all_features, lifespans, lengths, file_paths, cluster_counts

def evaluate_model(model, val_loader, criterion, device, scale_factor=100000):
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
                    criterion, device, scale_factor=100000):
    """Save predictions for all samples to a CSV file."""
    model.eval()
    all_predictions = []
    
    # Process training data
    print("\nGenerating predictions for training set...")
    _, train_preds, train_targets, train_error = evaluate_model(model, train_loader, criterion, device, scale_factor)
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
    _, val_preds, val_targets, val_error = evaluate_model(model, val_loader, criterion, device, scale_factor)
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

def main():
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Directories
    base_dir = 'data/Lifespan_calculated'
    subdirs = ['control', 'Terbinafin', 'controlTerbinafin', 'companyDrug']
    
    # Load all data and split into train/validation
    all_features = []
    all_lifespans = []
    all_lengths = []
    val_features = []
    val_lifespans = []
    val_lengths = []
    train_files = []
    val_files = []
    
    # Set maximum frame number for training (set to None to use all frames)
    max_frame = 10000  # Only use clusters up to frame 10000
    
    print("Loading data...")
    print(f"Using clusters up to frame {max_frame}" if max_frame is not None else "Using all clusters")
    
    all_cluster_counts = []
    for subdir in tqdm(subdirs, desc="Processing directories"):
        dir_path = os.path.join(base_dir, subdir)
        features, lifespans, lengths, file_paths, cluster_counts = load_data_from_directory(dir_path, max_frame=max_frame)
        all_cluster_counts.extend(cluster_counts)
        
        # Randomly select 3 files for validation
        if len(features) >= 3:  # Only if we have enough files after filtering
            val_indices = random.sample(range(len(features)), 3)
            
            for i in range(len(features)):
                if i in val_indices:
                    val_files.extend([file_paths[i]])
                    val_features.append(features[i])
                    val_lifespans.append(lifespans[i])
                    val_lengths.append(lengths[i])
                else:
                    train_files.extend([file_paths[i]])
                    all_features.append(features[i])
                    all_lifespans.append(lifespans[i])
                    all_lengths.append(lengths[i])
    
    # Print cluster counts
    print("\nCluster counts for each file:")
    print(f"{'File':<60} {'Total Clusters':>15} {'Used Clusters':>15} {'Percent Used':>15}")
    print("-" * 105)
    for filename, total, used in sorted(all_cluster_counts):
        percent = (used / total * 100) if total > 0 else 0
        print(f"{filename:<60} {total:>15} {used:>15} {percent:>14.1f}%")
    
    # Print summary
    total_clusters = sum(total for _, total, _ in all_cluster_counts)
    used_clusters = sum(used for _, _, used in all_cluster_counts)
    avg_percent = (used_clusters / total_clusters * 100) if total_clusters > 0 else 0
    print("\nSummary:")
    print(f"Total clusters across all files: {total_clusters}")
    print(f"Clusters used in training: {used_clusters}")
    print(f"Percentage of clusters used: {avg_percent:.1f}%")
    
    if len(val_files) < len(subdirs) * 3:
        print(f"\nWarning: Could only find {len(val_files)} files for validation after filtering")
        if len(val_files) == 0:
            raise ValueError("No validation files found after filtering. Try increasing max_frame or set it to None.")
    
    # Print validation files for reference
    print("\nValidation files:")
    for file in val_files:
        print(file)
    
    print("\nPreparing data for training...")
    
    # Standardize features
    all_features_flat = np.vstack([f for f in all_features])
    feature_scaler = StandardScaler()
    feature_scaler.fit(all_features_flat)
    
    # Scale features
    scaled_features = [feature_scaler.transform(f) for f in all_features]
    scaled_val_features = [feature_scaler.transform(f) for f in val_features]
    
    # Scale lifespans by dividing by 100k
    scale_factor = 100000
    scaled_lifespans = np.array(all_lifespans) / scale_factor
    scaled_val_lifespans = np.array(val_lifespans) / scale_factor
    
    # Save training data summary before training
    save_training_data(train_files, all_features, all_lifespans, all_lengths,
                      val_files, val_features, val_lifespans, val_lengths,
                      feature_scaler)
    
    # Create PyTorch datasets
    train_dataset = WormDataset(scaled_features, scaled_lifespans, all_lengths)
    val_dataset = WormDataset(scaled_val_features, scaled_val_lifespans, val_lengths)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, collate_fn=collate_fn)
    
    # Initialize model with increased complexity
    input_size = scaled_features[0].shape[1]  # number of features
    hidden_size = 512  # Increased hidden size
    num_layers = 8    # Increased number of layers
    model = LSTMLifespanPredictor(input_size, hidden_size, num_layers, dropout=0.3).to(device)
    
    # Training parameters
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-5)  # Reduced learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
    num_epochs = 300  # Increased epochs
    
    print("\nStarting training...")
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_errors = []
    val_errors = []
    
    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss/len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluate training error
        _, _, _, train_error = evaluate_model(model, train_loader, criterion, device)
        train_errors.append(train_error)
        
        # Validation phase
        val_loss, _, _, val_error = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_errors.append(val_error)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'feature_scaler': feature_scaler,
                'lifespan_scale_factor': 100000,
                'input_size': input_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_errors': train_errors,
                'val_errors': val_errors
            }, 'best_lifespan_predictor_model.pth')
        
        # Update plot and print status every N epochs
        if (epoch + 1) % 10 == 0:  # Plot every 10 epochs
            plot_losses(train_losses, val_losses, train_errors, val_errors)
            tqdm.write(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
            tqdm.write(f'Train Error: {train_error:.1f}%, Val Error: {val_error:.1f}%')
    
    print("\nTraining complete. Best model saved as best_lifespan_predictor_model.pth")
    print("Metrics plot saved as training_metrics.png")
    
    # Load best model for predictions
    checkpoint = torch.load('best_lifespan_predictor_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Save all predictions to CSV
    save_predictions(model, train_loader, val_loader, train_files, val_files, 
                    criterion, device, scale_factor)
    
    # Get final predictions on validation set
    _, predictions, targets, final_val_error = evaluate_model(model, val_loader, criterion, device, scale_factor)
    
    print("\nValidation Set Predictions:")
    print("File                                     Predicted    Actual    Difference    % Error")
    print("-" * 90)
    for i, (pred, target, file) in enumerate(zip(predictions, targets, val_files)):
        filename = os.path.basename(file)
        percent_error = abs(pred[0] - target[0]) / target[0] * 100
        print(f"{filename:<40} {pred[0]:>9.0f} {target[0]:>9.0f} {pred[0]-target[0]:>11.0f} {percent_error:>9.1f}%")

if __name__ == "__main__":
    main() 