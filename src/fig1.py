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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Visualize embeddings for DWUG English dataset")
    parser.add_argument("--input_dir", type=str, default="embeddings", help="Input directory containing embeddings")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for figures")
    parser.add_argument("--tgt_word", type=str, default="record_nn", help="Target word to visualize semantic change")
    parser.add_argument("--reg_m", type=int, default=100, help="Regularization parameter for UOT")
    return parser.parse_args()


def load_embeddings(input_dir):
    """Load embedding data from pickle file"""
    embeddings_path = input_dir / "dwug_en_embeddings.pkl"
    with open(embeddings_path, "rb") as f:
        source_token2vecs, target_token2vecs = pickle.load(f)
    return source_token2vecs, target_token2vecs


def validate_target_word(target_word, target_token2vecs):
    """Validate that the target word exists in the embeddings"""
    if target_word not in target_token2vecs:
        raise ValueError(f"Target word '{target_word}' not found in embeddings.")


def extract_word_vectors(source_token2vecs, target_token2vecs, target_word):
    """Extract vectors for the specified target word"""
    source_vecs = np.array(source_token2vecs[target_word])
    target_vecs = np.array(target_token2vecs[target_word])
    return source_vecs, target_vecs


def calculate_sus_scores(source_vecs, target_vecs, reg_m):
    """Calculate Semantic Usage Shift (SUS) scores"""
    print("Calculating SUS scores...")
    source_sus, target_sus = utils.calc_sus(
        source_vecs, target_vecs, 
        reg_m=reg_m
    )
    return source_sus, target_sus


def compute_or_load_tsne(source_vecs, target_vecs, input_dir, target_word):
    """Compute t-SNE dimensionality reduction or load from cache"""
    tsne_dir = input_dir / "tsne"
    tsne_dir.mkdir(exist_ok=True, parents=True)
    tsne_file = tsne_dir / f"{target_word}_tsne.pkl"
    
    if tsne_file.exists():
        print(f"Loading cached t-SNE results for '{target_word}'...")
        with open(tsne_file, "rb") as f:
            source_vecs_2d, target_vecs_2d = pickle.load(f)
    else:
        print(f"Computing t-SNE for '{target_word}' (this may take a while)...")
        source_vecs_2d, target_vecs_2d = compute_tsne(source_vecs, target_vecs)
        
        # Cache the results for future use
        with open(tsne_file, "wb") as f:
            pickle.dump((source_vecs_2d, target_vecs_2d), f)
        print(f"t-SNE results cached to {tsne_file}")
    
    return source_vecs_2d, target_vecs_2d


def compute_tsne(source_vecs, target_vecs, perplexity=30, random_state=42):
    """Compute t-SNE dimensionality reduction"""
    # Combine all vectors for consistent embedding
    all_vecs = np.vstack((source_vecs, target_vecs))
    
    # Apply t-SNE
    tsne = TSNE(perplexity=perplexity, n_components=2, random_state=random_state)
    all_vecs_2d = tsne.fit_transform(all_vecs)
    
    # Split back into source and target
    source_vecs_2d = all_vecs_2d[:len(source_vecs)]
    target_vecs_2d = all_vecs_2d[len(source_vecs):]
    
    return source_vecs_2d, target_vecs_2d


def create_visualization(source_vecs_2d, target_vecs_2d, source_sus, target_sus, ldr=False):
    """Create the scatter plot visualization"""
    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Use the utility function for scatter plotting
    utils.scatter_plot(source_vecs_2d, target_vecs_2d, source_sus, target_sus, fig, ax, ldr=ldr)
    
    return fig


def save_figure(fig, output_path):
    """Save the figure to the specified path"""
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {output_path}")


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
    
    # Calculate SUS scores
    source_sus, target_sus = calculate_sus_scores(
        source_vecs, target_vecs, args.reg_m
    )
    
    # Compute or load t-SNE results
    source_vecs_2d, target_vecs_2d = compute_or_load_tsne(
        source_vecs, target_vecs, input_dir, args.tgt_word
    )
    
    # Create visualization
    fig = create_visualization(source_vecs_2d, target_vecs_2d, source_sus, target_sus)
    
    # Save the figure
    output_path = output_dir / f"fig1_{args.tgt_word}.png"
    save_figure(fig, output_path)

if __name__ == "__main__":
    main()