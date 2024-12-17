import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import json
from scipy.spatial import ConvexHull
from scipy.fft import rfft, rfftfreq
from scipy.stats import entropy

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

def classify_roaming_dwelling(df, slope=2.5, intercept=0):
    """Classifies each window as roaming or dwelling based on a threshold."""
    df['State'] = 'D'  # Default to dwelling
    df.loc[df['Speed (window)'] > (slope * df['Angular Velocity (window)'] + intercept), 'State'] = 'R'
    return df

def calculate_curvature(x, y):
    """Calculates curvature using a simple approximation."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(ddx * dy - dx * ddy) / (dx**2 + dy**2)**1.5
    return curvature

def calculate_autocorrelation(series, lag=1):
    """Calculates the autocorrelation of a series with a given lag."""
    if len(series) < lag + 1:
        return 0
    return np.corrcoef(series[:-lag], series[lag:])[0, 1]

def calculate_speed_and_angular_velocity(df):
    """Calculate speed and angular velocity for each frame."""
    # Calculate speed
    df['dx'] = df['X'].diff()
    df['dy'] = df['Y'].diff()
    df['dt'] = df['Timestamp'].diff()
    df['Speed'] = np.sqrt(df['dx']**2 + df['dy']**2) / df['dt']
    
    # Calculate angles and angular velocity (using every other frame)
    df['dx_2'] = df['X'].diff(2)
    df['dy_2'] = df['Y'].diff(2)
    df['angle_rad'] = np.arctan2(df['dy_2'], df['dx_2'])
    df['angle_deg'] = np.degrees(df['angle_rad'])
    df['Angular Velocity'] = df['angle_deg'].diff() / (df['Timestamp'].diff(2))
    
    # Calculate acceleration
    df['Acceleration'] = df['Speed'].diff() / df['dt']
    
    # Smooth the speed and angular velocity for roaming/dwelling classification
    df['Speed (window)'] = df['Speed'].rolling(5, center=True).mean()
    df['Angular Velocity (window)'] = df['Angular Velocity'].rolling(5, center=True).mean()
    
    # Classify roaming and dwelling states
    df = classify_roaming_dwelling(df)
    
    return df

def extract_cluster_features(cluster_data):
    """Extract features from a single cluster."""
    features = {}
    
    # Basic temporal features
    features['duration'] = cluster_data['Timestamp'].max() - cluster_data['Timestamp'].min()
    features['num_frames'] = len(cluster_data)
    
    # Speed statistics
    features['mean_speed'] = cluster_data['Speed'].mean()
    features['max_speed'] = cluster_data['Speed'].max()
    features['min_speed'] = cluster_data['Speed'].min()
    features['std_speed'] = cluster_data['Speed'].std()
    
    # Angular velocity statistics
    features['mean_angular_velocity'] = cluster_data['Angular Velocity'].mean()
    features['max_angular_velocity'] = cluster_data['Angular Velocity'].max()
    features['min_angular_velocity'] = cluster_data['Angular Velocity'].min()
    features['std_angular_velocity'] = cluster_data['Angular Velocity'].std()
    
    # Acceleration statistics
    features['max_acceleration'] = cluster_data['Acceleration'].max()
    features['min_acceleration'] = cluster_data['Acceleration'].min()
    features['mean_acceleration'] = cluster_data['Acceleration'].mean()
    features['std_acceleration'] = cluster_data['Acceleration'].std()
    
    # Path shape features
    curvature = calculate_curvature(cluster_data['X'].values, cluster_data['Y'].values)
    features['mean_curvature'] = np.nanmean(curvature)
    features['max_curvature'] = np.nanmax(curvature)
    
    # Convex hull area (if enough points)
    if len(cluster_data) >= 3:
        try:
            hull = ConvexHull(cluster_data[['X', 'Y']].values)
            features['convex_hull_area'] = hull.volume  # In 2D, volume is area
        except:
            features['convex_hull_area'] = 0
    else:
        features['convex_hull_area'] = 0
    
    # Radius of gyration
    centroid = cluster_data[['X', 'Y']].mean()
    distances = np.sqrt((cluster_data['X'] - centroid['X'])**2 + 
                       (cluster_data['Y'] - centroid['Y'])**2)
    features['radius_of_gyration'] = np.sqrt(np.sum(distances**2) / len(cluster_data))
    
    # Roaming/Dwelling features
    roaming_frames = cluster_data['State'] == 'R'
    dwelling_frames = cluster_data['State'] == 'D'
    
    features['fraction_roaming'] = roaming_frames.sum() / len(cluster_data)
    features['fraction_dwelling'] = dwelling_frames.sum() / len(cluster_data)
    
    # Calculate bout durations
    bouts = cluster_data['State'].ne(cluster_data['State'].shift()).cumsum()
    roaming_bouts = cluster_data[roaming_frames].groupby(bouts)['Timestamp'].agg(lambda x: x.max() - x.min())
    dwelling_bouts = cluster_data[dwelling_frames].groupby(bouts)['Timestamp'].agg(lambda x: x.max() - x.min())
    
    features['mean_roaming_bout_duration'] = roaming_bouts.mean() if len(roaming_bouts) > 0 else 0
    features['mean_dwelling_bout_duration'] = dwelling_bouts.mean() if len(dwelling_bouts) > 0 else 0
    
    features['roaming_frequency'] = roaming_frames.sum() / features['duration'] if features['duration'] > 0 else 0
    features['dwelling_frequency'] = dwelling_frames.sum() / features['duration'] if features['duration'] > 0 else 0
    
    # State transitions
    state_changes = cluster_data['State'].ne(cluster_data['State'].shift()).sum() - 1
    features['state_transitions'] = max(state_changes, 0)
    
    # Frequency domain features
    if len(cluster_data) > 1:
        yf = rfft(cluster_data['Speed'].values)
        xf = rfftfreq(len(cluster_data), 1 / 2)  # Assuming 2 seconds between frames
        dominant_frequency_index = np.argmax(np.abs(yf[1:])) + 1
        features['dominant_frequency'] = xf[dominant_frequency_index]
    else:
        features['dominant_frequency'] = 0
    
    # Entropy features
    speed_counts = np.histogram(cluster_data['Speed'], bins=10)[0]
    features['speed_entropy'] = entropy(speed_counts) if np.any(speed_counts > 0) else 0
    
    angular_velocity_counts = np.histogram(cluster_data['Angular Velocity'], bins=10)[0]
    features['angular_velocity_entropy'] = entropy(angular_velocity_counts) if np.any(angular_velocity_counts > 0) else 0
    
    # Autocorrelation features
    features['speed_autocorrelation_1'] = calculate_autocorrelation(cluster_data['Speed'], lag=1)
    features['speed_autocorrelation_5'] = calculate_autocorrelation(cluster_data['Speed'], lag=5)
    features['angular_velocity_autocorrelation_1'] = calculate_autocorrelation(cluster_data['Angular Velocity'], lag=1)
    features['angular_velocity_autocorrelation_5'] = calculate_autocorrelation(cluster_data['Angular Velocity'], lag=5)
    
    return features

def process_directory(input_dir, output_dir):
    """Process all CSV files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace('.csv', '_features.npz'))
            
            # Skip if output file already exists
            if os.path.exists(output_path):
                continue
            
            try:
                # Read and preprocess data
                df = pd.read_csv(input_path)
                
                # Calculate speed, angular velocity, and state features
                df = calculate_speed_and_angular_velocity(df)
                
                # Extract features for each cluster
                all_features = []
                feature_names = None
                
                for cluster_id in df['Cluster'].unique():
                    cluster_data = df[df['Cluster'] == cluster_id]
                    features = extract_cluster_features(cluster_data)
                    
                    if feature_names is None:
                        feature_names = list(features.keys())
                    
                    all_features.append([features[name] for name in feature_names])
                
                # Convert to numpy array
                features_array = np.array(all_features)
                
                # Save features
                np.savez(output_path,
                        features=features_array,
                        feature_names=feature_names,
                        num_frames=df['Frame'].max(),
                        source_file=filename)
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

def main():
    # Process each subdirectory
    base_dir = 'data/Lifespan_cleaned'
    output_base_dir = 'data/Lifespan_calculated'
    
    subdirs = ['control', 'Terbinafin', 'controlTerbinafin', 'companyDrug']
    
    for subdir in subdirs:
        print(f"\nProcessing {subdir}...")
        input_dir = os.path.join(base_dir, subdir)
        output_dir = os.path.join(output_base_dir, subdir)
        process_directory(input_dir, output_dir)

if __name__ == "__main__":
    main() 