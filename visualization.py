import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from tqdm import tqdm

def plot_movement(ax, session, label):
    """
    Plot the movement pattern of a worm session.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        session (pd.DataFrame): Session data.
        label (str): Label for the plot (e.g., "Animal 1, Session 1, Drugged").
    """
    with tqdm(total=2, desc="Plotting movement", leave=False) as pbar:
        scatter = ax.scatter(
            session['X'], session['Y'], c=session['Frame'], cmap='viridis', s=10
        )
        pbar.update(1)
        
        ax.plot(session['X'], session['Y'], color='gray', alpha=0.5, linewidth=0.5)
        ax.set_title(label)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        plt.colorbar(scatter, ax=ax, label="Frame Number")
        pbar.update(1)

def plot_absolute_diff(ax, session, label):
    """
    Plot the absolute coordinate differences between frames.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        session (pd.DataFrame): Session data.
        label (str): Label for the plot (e.g., "Animal 1, Session 1, Drugged").
    """
    with tqdm(total=1, desc="Plotting differences", leave=False) as pbar:
        ax.plot(
            session['Frame'], session['Delta_Distance'], marker='o', markersize=3, linestyle='-', alpha=0.8
        )
        ax.set_title(label)
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Absolute Distance Change")
        pbar.update(1)

def plot_clustered_movement(ax, session, clusters, label):
    """
    Plot the movement pattern of a worm session with clusters color-coded.
    Lines between points in the same cluster are gray; lines connecting different clusters are red.
    Colors are assigned sequentially while preserving original cluster IDs.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        session (pd.DataFrame): Session data.
        clusters (pd.Series): Cluster labels for each point.
        label (str): Label for the plot.
    """
    unique_clusters = sorted(clusters.unique())  # Sort to ensure consistent color assignment
    color_map = cm.get_cmap('tab10', len(unique_clusters))
    # Create mapping from cluster IDs to color indices
    color_indices = {cluster_id: idx for idx, cluster_id in enumerate(unique_clusters)}
    
    # Plot points with progress bar
    with tqdm(total=len(unique_clusters) + len(session) - 1, 
              desc="Plotting clustered movement", leave=False) as pbar:
        
        # Plot points for each cluster
        for cluster_id in unique_clusters:
            cluster_points = session[clusters == cluster_id]
            color_idx = color_indices[cluster_id]  # Get sequential color index while preserving ID
            ax.scatter(
                cluster_points['X'], cluster_points['Y'], 
                label=f"Cluster {cluster_id}", s=10, 
                c=np.array([color_map(color_idx)] * len(cluster_points)), alpha=0.6
            )
            pbar.update(1)

        # Draw connections with progress tracking
        for i in range(1, len(session)):
            if clusters.iloc[i] == clusters.iloc[i - 1]:  # Same cluster
                ax.plot(
                    [session.iloc[i - 1]['X'], session.iloc[i]['X']],
                    [session.iloc[i - 1]['Y'], session.iloc[i]['Y']],
                    color='red', linewidth=0.5
                )
            else:  # Different clusters
                ax.plot(
                    [session.iloc[i - 1]['X'], session.iloc[i]['X']],
                    [session.iloc[i - 1]['Y'], session.iloc[i]['Y']],
                    color='lightgray', linewidth=0.8
                )
            pbar.update(1)

    ax.set_title(label)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

def plot_clustered_movement_with_original_colors(ax, session, clusters, original_clusters, label):
    """
    Plot the movement pattern with clusters color-coded, preserving original colors.
    Lines between points in the same cluster are gray; lines connecting different clusters are red.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        session (pd.DataFrame): Filtered session data.
        clusters (pd.Series): Filtered cluster labels.
        original_clusters (pd.Series): Original cluster labels for color reference.
        label (str): Label for the plot.
    """
    unique_clusters = original_clusters.unique()
    color_map = cm.get_cmap('tab10', len(unique_clusters))

    # Calculate total operations for progress bar
    remaining_clusters = set(clusters.values)
    total_ops = sum(1 for c in unique_clusters if c in remaining_clusters) + len(session) - 1

    # Plot with progress tracking
    with tqdm(total=total_ops, desc="Plotting with original colors", leave=False) as pbar:
        # Plot points
        for cluster_id in unique_clusters:
            if cluster_id in clusters.values:  # Plot only remaining clusters
                cluster_points = session[clusters == cluster_id]
                ax.scatter(
                    cluster_points['X'], cluster_points['Y'], 
                    label=f"Cluster {cluster_id}", 
                    s=10, 
                    c=np.array([color_map(cluster_id)] * len(cluster_points)), 
                    alpha=0.6
                )
                pbar.update(1)

        # Draw connections
        for i in range(1, len(session)):
            if clusters.iloc[i] == clusters.iloc[i - 1]:  # Same cluster
                ax.plot(
                    [session.iloc[i - 1]['X'], session.iloc[i]['X']],
                    [session.iloc[i - 1]['Y'], session.iloc[i]['Y']],
                    color='red', linewidth=0.5
                )
            else:  # Different clusters
                ax.plot(
                    [session.iloc[i - 1]['X'], session.iloc[i]['X']],
                    [session.iloc[i - 1]['Y'], session.iloc[i]['Y']],
                    color='lightgray', linewidth=0.8
                )
            pbar.update(1)

    ax.set_title(label)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate") 