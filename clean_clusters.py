import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from visualization import plot_movement, plot_clustered_movement
import random

def clean_small_clusters(file_path, min_points):
    """
    Remove points belonging to clusters smaller than the threshold.

    Args:
        file_path (Path): Path to the clustered CSV file.
        min_points (int): Minimum number of points required to keep a cluster.

    Returns:
        pd.DataFrame: Cleaned DataFrame with small clusters removed.
    """
    # Load the data
    print(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)
    
    if 'Cluster' not in data.columns:
        raise ValueError("No 'Cluster' column found in the data")
    
    # Get cluster sizes
    print("Analyzing cluster sizes...")
    with tqdm(total=1, desc="Counting cluster sizes", leave=False) as pbar:
        cluster_sizes = data['Cluster'].value_counts()
        pbar.update(1)
    
    # Identify clusters to keep
    valid_clusters = cluster_sizes[cluster_sizes >= min_points].index
    print(f"\nCluster statistics:")
    print(f"Total clusters: {len(cluster_sizes)}")
    print(f"Clusters meeting threshold ({min_points} points): {len(valid_clusters)}")
    
    # Filter data with progress tracking
    print("\nFiltering data...")
    original_points = len(data)
    
    # Process each valid cluster
    filtered_chunks = []
    with tqdm(total=len(valid_clusters), desc="Processing valid clusters", leave=False) as pbar:
        for cluster_id in valid_clusters:
            cluster_data = data[data['Cluster'] == cluster_id].copy()
            filtered_chunks.append(cluster_data)
            pbar.update(1)
    
    # Combine filtered data and sort by Frame
    print("Combining filtered data and sorting...")
    with tqdm(total=1, desc="Combining and sorting", leave=False) as pbar:
        data_filtered = pd.concat(filtered_chunks, ignore_index=True)
        data_filtered = data_filtered.sort_values('Frame').reset_index(drop=True)
        pbar.update(1)
    
    remaining_points = len(data_filtered)
    
    print(f"\nPoints statistics:")
    print(f"Original points: {original_points:,}")
    print(f"Remaining points: {remaining_points:,}")
    print(f"Removed points: {original_points - remaining_points:,}")
    print(f"Percentage kept: {(remaining_points/original_points*100):.2f}%")
    
    return data_filtered

def plot_sample_files(base_path, samples_per_treatment=2):
    """
    Plot sample files from each treatment group.

    Args:
        base_path (Path): Path to the cleaned data directory.
        samples_per_treatment (int): Number of files to sample from each treatment.
    """
    # Create output directory for plots
    output_dir = Path("data/cleaned_movement_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each treatment group
    treatment_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    for treatment_dir in tqdm(treatment_dirs, desc="Plotting treatment groups", leave=True):
        treatment = treatment_dir.name
        print(f"\nPlotting {treatment} samples...")
        
        # Get all CSV files
        csv_files = list(treatment_dir.glob("*.csv"))
        if not csv_files:
            print(f"No files found in {treatment}")
            continue
            
        # Sample random files
        sampled_files = random.sample(csv_files, min(samples_per_treatment, len(csv_files)))
        
        # Plot each sampled file
        for csv_file in tqdm(sampled_files, desc=f"Plotting {treatment} files", leave=True):
            print(f"Plotting {csv_file.name}")
            
            try:
                # Load data
                data = pd.read_csv(csv_file)
                
                # Create figure
                fig, axes = plt.subplots(1, 2, figsize=(20, 8))
                fig.suptitle(f"{treatment} - {csv_file.stem} (Cleaned)", fontsize=16)
                
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
                output_path = output_dir / f"{treatment}_{csv_file.stem}_cleaned.png"
                plt.savefig(
                    output_path,
                    dpi=300,
                    bbox_inches='tight'
                )
                plt.close()
                
            except Exception as e:
                print(f"Error plotting file {csv_file.name}: {str(e)}")
                plt.close()

def process_directory(input_dir, output_dir, min_points):
    """
    Process all CSV files in a directory and its subdirectories.

    Args:
        input_dir (Path): Input directory containing clustered CSV files.
        output_dir (Path): Output directory for cleaned files.
        min_points (int): Minimum number of points required to keep a cluster.
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all treatment directories
    treatment_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    # Process each treatment group
    for treatment_dir in tqdm(treatment_dirs, desc="Processing treatment groups", leave=True):
        treatment = treatment_dir.name
        print(f"\nProcessing {treatment} group...")
        
        # Create output treatment directory
        treatment_output_dir = output_dir / treatment
        treatment_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all CSV files in the treatment directory
        csv_files = list(treatment_dir.glob("*.csv"))
        print(f"Found {len(csv_files)} files to process")
        
        # Process each file
        for csv_file in tqdm(csv_files, desc=f"Processing {treatment} files", leave=True):
            try:
                print(f"\nProcessing {csv_file.name}")
                
                # Clean clusters
                data_filtered = clean_small_clusters(csv_file, min_points)
                
                # Save cleaned data with progress tracking
                output_file = treatment_output_dir / csv_file.name
                print(f"Saving cleaned data to: {output_file}")
                with tqdm(total=1, desc="Saving file", leave=False) as pbar:
                    data_filtered.to_csv(output_file, index=False)
                    pbar.update(1)
                
            except Exception as e:
                print(f"Error processing file {csv_file.name}: {str(e)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Remove points belonging to small clusters for all files.')
    parser.add_argument('min_points', type=int, help='Minimum number of points required to keep a cluster')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up input and output directories
    input_base = Path("data/Lifespan_clustered")
    output_base = Path("data/Lifespan_cleaned")
    
    print(f"Input directory: {input_base}")
    print(f"Output directory: {output_base}")
    print(f"Minimum points threshold: {args.min_points}")
    
    # Process all files
    process_directory(input_base, output_base, args.min_points)
    
    # Plot samples from each treatment group
    print("\nGenerating sample plots...")
    plot_sample_files(output_base)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main() 