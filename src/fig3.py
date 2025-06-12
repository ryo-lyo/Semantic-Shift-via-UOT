import numpy as np
import pickle
from pathlib import Path
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import utils
import argparse


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Visualize OT and UOT matrices")
    parser.add_argument("--input_dir", type=str, default="embeddings", help="Input directory containing embedding files")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for figures")
    parser.add_argument("--reg_m", type=int, default=10, help="Regularization parameter for UOT")
    parser.add_argument("--tgt_word", type=str, default="record_nn", help="Target word to visualize transportation matrix")
    return parser.parse_args()


def load_embeddings(input_dir):
    """Load embedding data from pickle file"""
    embeddings_path = input_dir / "dwug_en_embeddings.pkl"
    with open(embeddings_path, "rb") as f:
        source_token2vecs, target_token2vecs = pickle.load(f)
    return source_token2vecs, target_token2vecs


def load_tsne_data(target_word):
    """Load t-SNE coordinate data"""
    tsne_path = f"embeddings/tsne/{target_word}_tsne.pkl"
    with open(tsne_path, "rb") as f:
        source_xy, target_xy = pickle.load(f)
    return source_xy, target_xy


def load_cluster_data(target_word):
    """Load cluster data from CSV file"""
    cluster_path = f"data/dwug_en/clusters/opt/{target_word}.csv"
    df = pd.read_table(cluster_path, quoting=csv.QUOTE_NONE)
    # Convert cluster -1 to 0
    df.loc[df["cluster"] == -1, "cluster"] = 0
    return df


def prepare_data_for_visualization(df, source_xy, target_xy, source_embeddings, target_embeddings):
    """Prepare data for visualization by sorting and calculating boundaries"""
    # Add coordinate data
    df["x"] = list(source_xy[:, 0]) + list(target_xy[:, 0])
    df["grouping"] = [1] * len(source_embeddings) + [2] * len(target_embeddings)
    
    # Calculate sorted indices for source and target
    source_indices, target_indices = calculate_sorted_indices(df)
    
    # Get cluster information
    source_cluster = df["cluster"][df["grouping"] == 1].values
    target_cluster = df["cluster"][df["grouping"] == 2].values
    
    # Calculate boundary positions for visualization
    source_boundaries = np.where(np.diff(source_cluster[source_indices]) != 0)[0] + 1
    target_boundaries = np.where(np.diff(target_cluster[target_indices]) != 0)[0] + 1
    
    return source_indices, target_indices, source_cluster, target_cluster, source_boundaries, target_boundaries


def calculate_sorted_indices(df):
    """Calculate sorted indices by cluster and x-coordinate"""
    source_indices = None
    target_indices = None
    
    for group_id in [1, 2]:
        cluster_values = df["cluster"][df["grouping"] == group_id].values
        x_values = df["x"][df["grouping"] == group_id].values
        
        # Sort by cluster first
        sorted_indices = np.argsort(cluster_values)
        sorted_clusters = cluster_values[sorted_indices]
        sorted_x = x_values[sorted_indices]
        
        # Sort by x-coordinate within each cluster
        final_indices = []
        for cluster_value in np.unique(sorted_clusters):
            cluster_mask = sorted_clusters == cluster_value
            cluster_indices = np.where(cluster_mask)[0]
            cluster_x = sorted_x[cluster_indices]
            
            # Sort by x-coordinate
            x_sort_indices = np.argsort(cluster_x)
            final_indices.append(sorted_indices[cluster_indices][x_sort_indices])
        
        final_indices = np.concatenate(final_indices)
        
        if group_id == 1:
            source_indices = final_indices
        else:
            target_indices = final_indices
    
    return source_indices, target_indices


def create_custom_colormap():
    """Create custom colormaps for visualization"""
    # Main colormap for transport matrices
    palette = np.array([
        [0, '#ffffff'],
        [0.0001, '#ffffff'],
        [0.01, '#ff0000']
    ])
    main_cmap = utils.pallet2cmap(palette)
    
    # Overlay colormap for cluster identity
    reds = plt.cm.Reds
    white_color = [1, 1, 1, 1]
    red_zero_color = reds(0)
    overlay_cmap = ListedColormap([white_color, red_zero_color])
    
    return main_cmap, overlay_cmap, red_zero_color


def plot_transport_matrices(T_ot, T_uot, source_indices, target_indices, 
                          source_cluster, target_cluster, source_boundaries, target_boundaries,
                          main_cmap, overlay_cmap, red_zero_color):
    """Plot OT and UOT transport matrices side by side"""
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
    # Create identity matrix for cluster overlay
    identity_matrix = np.array([
        [i == j for j in target_cluster[target_indices]] 
        for i in source_cluster[source_indices]
    ])
    
    # Plot OT matrix
    plot_single_matrix(
        axs[0], T_ot, source_indices, target_indices, 
        source_boundaries, target_boundaries, identity_matrix,
        main_cmap, overlay_cmap, red_zero_color, "OT"
    )
    
    # Plot UOT matrix
    im = plot_single_matrix(
        axs[1], T_uot, source_indices, target_indices, 
        source_boundaries, target_boundaries, identity_matrix,
        main_cmap, overlay_cmap, red_zero_color, "UOT"
    )
    
    # Add colorbar
    colorbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.76, pad=0.02)
    colorbar.set_label('Transported mass', fontsize=15, labelpad=25, loc='center', rotation=270)
    colorbar.ax.tick_params(labelsize=10)
    
    return fig


def plot_single_matrix(ax, transport_matrix, source_indices, target_indices,
                      source_boundaries, target_boundaries, identity_matrix,
                      main_cmap, overlay_cmap, red_zero_color, title):
    """Plot a single transport matrix with cluster boundaries and overlay"""
    # Display main transport matrix
    im = ax.imshow(
        transport_matrix[source_indices][:, target_indices], 
        cmap=main_cmap, vmin=0, vmax=0.01
    )
    
    # Set title and labels
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Target", fontsize=15)
    if title == "OT":
        ax.set_ylabel("Source", fontsize=15)
    
    # Draw cluster boundaries
    for boundary in target_boundaries:
        ax.axvline(boundary - 0.5, color='black', linewidth=0.1)
    for boundary in source_boundaries:
        ax.axhline(boundary - 0.5, color='black', linewidth=0.1)
    
    # Configure axes
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(bottom=False, left=False, right=False, top=False)
    
    # Add cluster identity overlay
    if 0 not in identity_matrix:
        ax.imshow(identity_matrix, cmap=ListedColormap([red_zero_color]), alpha=0.5)
    else:
        ax.imshow(identity_matrix, cmap=overlay_cmap, alpha=0.5)
    
    return im


def main():
    """Main processing function"""
    # Parse command line arguments
    args = parse_args()
    
    # Set up directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print(f"Target word: {args.tgt_word}")
    print("Loading data...")
    
    source_token2vecs, target_token2vecs = load_embeddings(input_dir)
    source_embeddings = source_token2vecs[args.tgt_word]
    target_embeddings = target_token2vecs[args.tgt_word]
    
    source_xy, target_xy = load_tsne_data(args.tgt_word)
    df = load_cluster_data(args.tgt_word)
    
    # Calculate OT and UOT matrices
    print("Computing OT and UOT matrices...")
    T_ot, T_uot = utils.calc_OT_and_UOT_matrix(
        source_embeddings, target_embeddings, 
        reg_m=args.reg_m
    )
    
    # Prepare visualization data
    print("Preparing visualization data...")
    (source_indices, target_indices, source_cluster, target_cluster, 
     source_boundaries, target_boundaries) = prepare_data_for_visualization(
        df, source_xy, target_xy, source_embeddings, target_embeddings
    )
    
    # Create colormaps
    main_cmap, overlay_cmap, red_zero_color = create_custom_colormap()
    
    # Create plots
    print("Creating plots...")
    fig = plot_transport_matrices(
        T_ot, T_uot, source_indices, target_indices,
        source_cluster, target_cluster, source_boundaries, target_boundaries,
        main_cmap, overlay_cmap, red_zero_color
    )
    
    # Save figure
    output_path = output_dir / f"fig3_{args.tgt_word}.png"
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()