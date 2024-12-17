import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import json

def calculate_speed(df):
    """Calculate speed between consecutive frames."""
    # First row speed is 0
    speeds = [0]
    
    # Calculate speeds for remaining rows
    for i in range(1, len(df)):
        dx = df.iloc[i]['X'] - df.iloc[i-1]['X']
        dy = df.iloc[i]['Y'] - df.iloc[i-1]['Y']
        dt = df.iloc[i]['Timestamp'] - df.iloc[i-1]['Timestamp']
        
        # Calculate Euclidean distance
        distance = np.sqrt(dx**2 + dy**2)
        speed = distance / dt if dt != 0 else 0
        speeds.append(speed)
    
    return speeds

def extract_cluster_features(df):
    """Extract features for each cluster in the dataframe."""
    cluster_features = []
    
    for cluster_id in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster_id]
        
        features = {
            'cluster_id': cluster_id,
            'max_speed': cluster_data['speed'].max(),
            'min_speed': cluster_data['speed'].min(),
            'mean_speed': cluster_data['speed'].mean(),
            'std_speed': cluster_data['speed'].std(),
            'num_points': len(cluster_data),
            'duration': cluster_data['Timestamp'].max() - cluster_data['Timestamp'].min(),
            'total_distance': cluster_data['speed'].sum() * (cluster_data['Timestamp'].max() - cluster_data['Timestamp'].min()),
            'x_range': cluster_data['X'].max() - cluster_data['X'].min(),
            'y_range': cluster_data['Y'].max() - cluster_data['Y'].min(),
            'changed_pixels_mean': cluster_data['Changed Pixels'].mean(),
            'changed_pixels_std': cluster_data['Changed Pixels'].std()
        }
        
        cluster_features.append(features)
    
    return pd.DataFrame(cluster_features)

def process_directory(input_dir, output_dir):
    """Process all files in a directory and save features."""
    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    for filename in tqdm(files, desc=f"Processing {os.path.basename(input_dir)}", leave=False):
        input_filepath = os.path.join(input_dir, filename)
        output_filepath = os.path.join(output_dir, filename.replace('.csv', '_features.npz'))
        
        # Skip if already processed
        if os.path.exists(output_filepath):
            continue
        
        df = pd.read_csv(input_filepath)
        
        # Calculate speed
        df['speed'] = calculate_speed(df)
        
        # Extract features for each cluster
        cluster_features = extract_cluster_features(df)
        
        # Sort clusters chronologically by first timestamp
        cluster_features = cluster_features.sort_values('cluster_id')
        
        # Store features and metadata
        feature_values = cluster_features.drop('cluster_id', axis=1).values
        metadata = {
            'num_frames': len(df),
            'source_file': filename,
            'num_clusters': len(cluster_features),
            'feature_names': list(cluster_features.drop('cluster_id', axis=1).columns)
        }
        
        # Save features and metadata
        np.savez(output_filepath,
                features=feature_values,
                **metadata)

def main():
    # Directories
    base_input_dir = 'data/Lifespan_cleaned'
    base_output_dir = 'data/Lifespan_calculated'
    subdirs = ['control', 'Terbinafin', 'controlTerbinafin', 'companyDrug']
    
    # Create output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    print("Processing data...")
    for subdir in tqdm(subdirs, desc="Processing directories"):
        input_dir = os.path.join(base_input_dir, subdir)
        output_dir = os.path.join(base_output_dir, subdir)
        process_directory(input_dir, output_dir)
    
    print("\nFeature extraction complete. Data saved in data/Lifespan_calculated/")

if __name__ == "__main__":
    main() 