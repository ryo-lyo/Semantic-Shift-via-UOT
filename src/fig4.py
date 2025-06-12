import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import matplotlib as mpl
from pathlib import Path
import pickle
import argparse
import utils


def parse_args():
    """Parse command line arguments for embedding visualization."""
    parser = argparse.ArgumentParser(description="Visualize embeddings for DWUG English dataset")
    parser.add_argument("--input_dir", type=str, default="embeddings", help="Input directory containing embeddings")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for figures")
    parser.add_argument("--tgt_word", type=str, default="record_nn", help="Target word to visualize semantic change")
    parser.add_argument("--reg_m", type=int, default=100, help="Regularization parameter for UOT")
    return parser.parse_args()


def load_embeddings(input_dir):
    """Load embeddings from pickle file.
    
    Args:
        input_dir (Path): Directory containing the embeddings file
        
    Returns:
        tuple: Source and target token-to-vectors mappings
    """
    embeddings_path = input_dir / "dwug_en_embeddings.pkl"
    with open(embeddings_path, "rb") as f:
        source_token2vecs, target_token2vecs = pickle.load(f)
    return source_token2vecs, target_token2vecs


def get_word_vectors(source_token2vecs, target_token2vecs, target_word):
    """Extract vectors for the target word from both time periods.
    
    Args:
        source_token2vecs (dict): Source period token-to-vectors mapping
        target_token2vecs (dict): Target period token-to-vectors mapping
        target_word (str): Word to extract vectors for
        
    Returns:
        tuple: Source and target vectors as numpy arrays
        
    Raises:
        ValueError: If target word not found in embeddings
    """
    if target_word not in target_token2vecs:
        raise ValueError(f"Target word '{target_word}' not found in embeddings.")
    
    source_vecs = np.array(source_token2vecs[target_word])
    target_vecs = np.array(target_token2vecs[target_word])
    
    return source_vecs, target_vecs


def setup_plot_style():
    """Configure matplotlib plot styling."""
    mpl.rcParams['hatch.linewidth'] = 0.1


def create_histogram_plot(source_sus, target_sus, target_word):
    """Create histogram plot comparing SUS distributions across time periods.
    
    Args:
        source_sus (np.array): SUS values for source time period
        target_sus (np.array): SUS values for target time period
        target_word (str): Target word being analyzed
        
    Returns:
        tuple: Figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
    
    # Define histogram parameters
    bins = np.linspace(min(source_sus), max(target_sus), 30)
    
    # Create histograms for both time periods
    ax.hist(
        source_sus, 
        bins=bins, 
        alpha=0.5, 
        label=r"1810―1860"
    )
    ax.hist(
        target_sus, 
        bins=bins, 
        alpha=0.5, 
        label=r"1960―2010",
        hatch='////', 
        edgecolor='black', 
        linewidth=0
    )
    
    return fig, ax


def configure_axes(ax):
    """Configure axis properties and styling.
    
    Args:
        ax: Matplotlib axis object
    """
    # Set y-axis to integer values with specific interval
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    
    # Set labels with custom font sizes
    ax.set_xlabel('Sense Usage Shift', fontdict={'size': 18})
    ax.set_ylabel('Frequency', fontdict={'size': 18})
    
    # Configure legend
    ax.legend(fontsize=15, loc='upper right', bbox_to_anchor=(1, 1.02))
    
    # Set tick label font sizes
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    
    # Set y-axis limits
    ax.set_ylim(0, 19.8)


def main():
    """Main function to orchestrate the visualization process."""
    # Parse command line arguments
    args = parse_args()
    
    # Setup directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load embeddings data
    source_token2vecs, target_token2vecs = load_embeddings(input_dir)
    
    # Extract vectors for target word
    source_vecs, target_vecs = get_word_vectors(
        source_token2vecs, target_token2vecs, args.tgt_word
    )
    
    # Calculate Sense Usage Shift (SUS) values
    source_sus, target_sus = utils.calc_sus(
        source_vecs, 
        target_vecs, 
        reg_m=args.reg_m
    )
    
    # Setup plot styling
    setup_plot_style()
    
    # Create histogram plot
    fig, ax = create_histogram_plot(source_sus, target_sus, args.tgt_word)
    
    # Configure axes properties
    configure_axes(ax)
    
    # Save figure
    output_path = output_dir / f"fig4_{args.tgt_word}.png"
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()