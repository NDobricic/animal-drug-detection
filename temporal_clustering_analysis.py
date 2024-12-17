import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from visualization import plot_clustered_movement, plot_movement
import random
import os
import sys
from tqdm import tqdm

def temporal_dfs_clustering(session, distance_threshold=10, max_neighbors=20, time_threshold=1800):
    """
    Perform DFS-based clustering on worm movement paths, considering both spatial and temporal proximity.

    Args:
        session (pd.DataFrame): Session data with X, Y coordinates and Timestamp.
        distance_threshold (float): Distance threshold for spatial clustering.
        max_neighbors (int): Maximum number of subsequent points to consider.
        time_threshold (float): Time threshold (in seconds) for temporal vicinity.

    Returns:
        pd.Series: Cluster labels for each frame in the session.
    """
    visited = np.zeros(len(session), dtype=bool)
    clusters = -np.ones(len(session), dtype=int)  # Initialize all as unclustered
    cluster_id = 0

    # Initialize progress bar for the main clustering loop
    pbar = tqdm(total=len(session), desc="Clustering points", leave=False)
    
    for start_idx in range(len(session)):
        if not visited[start_idx]:
            # Start a new cluster
            stack = [start_idx]
            points_in_current_cluster = 0
            
            while stack:
                idx = stack.pop()
                if visited[idx]:
                    continue
                    
                visited[idx] = True
                clusters[idx] = cluster_id
                points_in_current_cluster += 1

                # Add the next `max_neighbors` rows to the stack if within both thresholds
                for neighbor_offset in range(1, max_neighbors + 1):
                    neighbor_idx = idx + neighbor_offset
                    if neighbor_idx < len(session) and not visited[neighbor_idx]:
                        # Check temporal threshold first (it's cheaper)
                        time_diff = abs(session.iloc[idx]['Timestamp'] - session.iloc[neighbor_idx]['Timestamp'])
                        if time_diff <= time_threshold:
                            # Then check spatial threshold
                            distance = np.sqrt(
                                (session.iloc[idx]['X'] - session.iloc[neighbor_idx]['X'])**2 +
                                (session.iloc[idx]['Y'] - session.iloc[neighbor_idx]['Y'])**2
                            )
                            if distance < distance_threshold:
                                stack.append(neighbor_idx)
            
            cluster_id += 1
            pbar.update(points_in_current_cluster)
    
    pbar.close()
    print(f"Found {cluster_id} clusters")
    return pd.Series(clusters, index=session.index, name="Cluster")

def process_treatment_group(input_path, output_path):
    """
    Process all files in a treatment group and save clustered data.

    Args:
        input_path (Path): Path to input treatment directory.
        output_path (Path): Path to output treatment directory.
    """
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all CSV files in the treatment directory
    csv_files = list(input_path.glob("*.csv"))
    print(f"Found {len(csv_files)} files to process")
    
    # Process each file
    for csv_file in tqdm(csv_files, desc="Processing files", leave=True):
        try:
            # Load data
            data = pd.read_csv(csv_file)
            print(f"\nProcessing {csv_file.name}")
            print(f"Initial shape: {data.shape}")
            
            # Remove rows where X or Y are NaN
            data_clean = data.dropna(subset=['X', 'Y'])
            print(f"Shape after removing NaN values: {data_clean.shape}")
            
            if len(data_clean) == 0:
                print(f"Warning: All rows were NaN in {csv_file.name}")
                continue
            
            # Perform temporal clustering
            clusters = temporal_dfs_clustering(data_clean)
            
            # Add cluster information to the data
            data_clean['Cluster'] = clusters
            
            # Save processed data
            output_file = output_path / csv_file.name
            data_clean.to_csv(output_file, index=False)
            print(f"Saved clustered data to: {output_file}")
            
        except Exception as e:
            print(f"Error processing file {csv_file.name}: {str(e)}")

def plot_sample_files(base_path, output_dir, samples_per_treatment=2):
    """
    Plot sample files from each treatment group.

    Args:
        base_path (Path): Path to the clustered data directory.
        output_dir (Path): Directory to save plots.
        samples_per_treatment (int): Number of files to sample from each treatment.
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each treatment group
    for treatment_dir in tqdm(list(base_path.iterdir()), desc="Processing treatment groups"):
        if not treatment_dir.is_dir():
            continue
            
        treatment = treatment_dir.name
        print(f"\nProcessing {treatment}")
        
        # Get all CSV files
        csv_files = list(treatment_dir.glob("*.csv"))
        if not csv_files:
            print(f"No files found in {treatment}")
            continue
            
        # Sample random files
        sampled_files = random.sample(csv_files, min(samples_per_treatment, len(csv_files)))
        
        # Plot each sampled file
        for csv_file in sampled_files:
            print(f"Plotting {csv_file.name}")
            
            try:
                # Load data
                data = pd.read_csv(csv_file)
                
                # Create figure
                fig, axes = plt.subplots(1, 2, figsize=(20, 8))
                fig.suptitle(f"{treatment} - {csv_file.stem}", fontsize=16)
                
                # Raw movement plot
                plot_movement(
                    axes[0], data, 
                    "Raw Movement"
                )
                
                # Clustered movement plot
                plot_clustered_movement(
                    axes[1], data, data['Cluster'], 
                    "Temporal Clustering"
                )
                
                plt.tight_layout()
                
                # Save plot
                output_path = output_dir / f"{treatment}_{csv_file.stem}_comparison.png"
                plt.savefig(
                    output_path,
                    dpi=300,
                    bbox_inches='tight'
                )
                plt.close()
                
            except Exception as e:
                print(f"Error plotting file {csv_file.name}: {str(e)}")
                plt.close()

def main():
    # Set up paths
    input_base = Path("data/Lifespan_processed")
    output_base = Path("data/Lifespan_clustered")
    plot_dir = Path("data/temporal_clustering_plots")
    
    print("Step 1: Processing files")
    print(f"Input directory: {input_base}")
    print(f"Output directory: {output_base}")
    
    # Process each treatment group
    for treatment_dir in tqdm(list(input_base.iterdir()), desc="Processing treatment groups"):
        if not treatment_dir.is_dir():
            continue
            
        treatment = treatment_dir.name
        print(f"\nProcessing {treatment} group...")
        
        # Set up input and output paths for this treatment
        input_path = input_base / treatment
        output_path = output_base / treatment
        
        # Process the treatment group
        process_treatment_group(input_path, output_path)
        print(f"Completed {treatment} group")
    
    print("\nStep 2: Plotting sample files")
    plot_sample_files(output_base, plot_dir)

if __name__ == "__main__":
    main() 