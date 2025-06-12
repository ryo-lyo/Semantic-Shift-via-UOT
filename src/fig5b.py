import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
import pandas as pd
import csv
from pathlib import Path
import pickle
from sklearn.manifold import TSNE
import argparse
import utils
from fig1 import load_embeddings, validate_target_word, extract_word_vectors, compute_or_load_tsne, save_figure

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Visualize embeddings for DWUG English dataset")
    parser.add_argument("--input_dir", type=str, default="embeddings", help="Input directory containing embeddings")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for figures")
    parser.add_argument("--tgt_word", type=str, default="record_nn", help="Target word to visualize semantic change")
    return parser.parse_args()


def main():
    """Main processing function"""
    # Parse command line arguments
    args = parse_args()
    
    # Set up directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Processing target word: {args.tgt_word}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load embedding data
    print("Loading embeddings...")
    source_token2vecs, target_token2vecs = load_embeddings(input_dir)
    
    # Validate target word
    validate_target_word(args.tgt_word, target_token2vecs)
    
    # Extract vectors for the target word
    source_vecs, target_vecs = extract_word_vectors(
        source_token2vecs, target_token2vecs, args.tgt_word
    )
    
    print(f"Source vectors shape: {source_vecs.shape}")
    print(f"Target vectors shape: {target_vecs.shape}")
    
    
    # Compute or load t-SNE results
    source_vecs_2d, target_vecs_2d = compute_or_load_tsne(
        source_vecs, target_vecs, input_dir, args.tgt_word
    )
    
    # Load gold sense clusters
    df = pd.read_table(f"data/dwug_en/clusters/opt/{args.tgt_word}.csv", quoting=csv.QUOTE_NONE)
    clusters = df["cluster"].values.tolist()
    colors = ["lightblue", "orange", "red", "gray"]
    cluster_colors = [colors[i] if i in [0,1,2] else "gray" for i in clusters]
    labels = ["$\it [information]$", "$\it [achievement]$", "$\it [music]$", "others"]
    
    
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(source_vecs_2d[:, 0], source_vecs_2d[:, 1],
               c=cluster_colors[:len(source_vecs_2d)],
               s=100, edgecolors='k', linewidth=0.5)
    ax.scatter(target_vecs_2d[:, 0], target_vecs_2d[:, 1],
               c=cluster_colors[len(source_vecs_2d):],
               s=100, marker='s', edgecolors='k', linewidth=0.5)
    ax.scatter([], [], s=100, c="white", edgecolors="black", label=r"1810―1860", linewidths=0.5)
    ax.scatter([], [], s=100, c="white", edgecolors="black", label=r"1960―2010", marker="s", linewidths=0.5)
    
    for label, color in zip(labels, colors):
        ax.scatter([], [], s=100, c=color, label=label, edgecolors="black", linewidths=0.5)
    hans, labs = ax.get_legend_handles_labels()
    l1 = ax.legend(handles=hans[:2], labels=labs[:2], fontsize=14)
    l2 = ax.legend(handles=hans[2:], labels=labs[2:],
                   bbox_to_anchor=(0.5, 1.1), loc='center',
                   borderaxespad=0, fontsize=14,
                   handletextpad=0.01, 
                   ncol=4, columnspacing=0.4,)
    ax.add_artist(l1)

    # Save the figure
    output_path = output_dir / f"fig5b_{args.tgt_word}.png"
    save_figure(fig, output_path)
    
    print(f"Figure saved to: {output_path}")



if __name__ == "__main__":
    main()