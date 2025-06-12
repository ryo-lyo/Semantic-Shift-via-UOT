import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
import pandas as pd
from pathlib import Path
import pickle
from sklearn.manifold import TSNE
import argparse
import utils
from fig1 import load_embeddings, validate_target_word, extract_word_vectors, compute_or_load_tsne, create_visualization, save_figure

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Visualize embeddings for DWUG English dataset")
    parser.add_argument("--input_dir", type=str, default="embeddings", help="Input directory containing embeddings")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for figures")
    parser.add_argument("--tgt_word", type=str, default="record_nn", help="Target word to visualize semantic change")
    return parser.parse_args()

def calculate_ldr_scores(source_vecs, target_vecs):
    """Calculate Log Density Ratio (LDR) scores"""
    source_ldr, target_ldr = utils.calc_ldr(source_vecs, target_vecs)
    print("Calculating LDR scores...")
    return source_ldr, target_ldr


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
    
    # Calculate LDR scores
    source_ldr, target_ldr = calculate_ldr_scores(
        source_vecs, target_vecs
    )
    
    # Compute or load t-SNE results
    source_vecs_2d, target_vecs_2d = compute_or_load_tsne(
        source_vecs, target_vecs, input_dir, args.tgt_word
    )
    
    # Create visualization
    source_ldr = np.sign(source_ldr)*np.log10(1 + np.abs(source_ldr))
    target_ldr = np.sign(target_ldr)*np.log10(1 + np.abs(target_ldr))
    fig = create_visualization(source_vecs_2d, target_vecs_2d, source_ldr, target_ldr, ldr=True)
    
    # Save the figure
    output_path = output_dir / f"fig5a_{args.tgt_word}.png"
    save_figure(fig, output_path)
    
    print(f"Figure saved to: {output_path}")



if __name__ == "__main__":
    main()