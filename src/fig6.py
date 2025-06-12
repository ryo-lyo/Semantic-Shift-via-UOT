import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import utils
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize tau values for DWUG English dataset")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for figures")
    return parser.parse_args()


def str2list(s):
    """Convert string representation of list to numpy array."""
    return np.array(eval(s))


def load_and_preprocess_data():
    """Load and preprocess the statistics data."""
    print("Loading statistics data...")
    df = pd.read_table("data/dwug_en/stats/opt/stats_groupings.csv", quoting=csv.QUOTE_NONE)
    df["cluster_prob_dist1"] = df["cluster_prob_dist1"].apply(str2list)
    df["cluster_prob_dist2"] = df["cluster_prob_dist2"].apply(str2list)
    return df


def calculate_tau_values(df):
    """Calculate tau values for each word and cluster."""
    print("Calculating tau values...")
    word_cluster2tau = defaultdict(dict)
    taus_unique = []
    
    for i, row in df.iterrows():
        # Calculate log ratio as tau values
        taus = np.log(row["cluster_prob_dist2"] / row["cluster_prob_dist1"])
        taus_unique.extend(taus)
        cluster2tau = {cluster: tau for cluster, tau in zip(range(len(taus)), taus)}
        word_cluster2tau[row["lemma"]] = cluster2tau
    
    return word_cluster2tau, taus_unique


def handle_infinite_values(word_cluster2tau, taus_unique):
    """Replace infinite values with finite extremes."""
    print("Handling infinite values...")
    finite_values = np.array(taus_unique)[~np.isinf(taus_unique)]
    max_finite = np.max(finite_values)
    min_finite = np.min(finite_values)
    
    for outer_key, subdict in word_cluster2tau.items():
        for inner_key, value in subdict.items():
            if value == np.inf:
                word_cluster2tau[outer_key][inner_key] = max_finite
            elif value == -np.inf:
                word_cluster2tau[outer_key][inner_key] = min_finite
    
    return word_cluster2tau


def load_cluster_assignments(word_cluster2tau):
    """Load cluster assignments for each word instance."""
    print("Loading cluster assignments...")
    word_instance2cluster = defaultdict(list)
    
    for lemma in word_cluster2tau.keys():
        df_cluster = pd.read_table(f"data/dwug_en/clusters/opt/{lemma}.csv")
        word_instance2cluster[lemma] = df_cluster["cluster"].tolist()
    
    return word_instance2cluster


def map_instances_to_tau(word_cluster2tau, word_instance2cluster):
    """Map word instances to their corresponding tau values."""
    print("Mapping instances to tau values...")
    word_instance2tau = defaultdict(list)
    
    for lemma, clusters in word_instance2cluster.items():
        # Filter out cluster -1 (noise/outliers)
        taus = [word_cluster2tau[lemma][cluster] 
                for cluster in clusters if cluster != -1]
        word_instance2tau[lemma] = taus
    
    return word_instance2tau


def load_embeddings():
    """Load pre-computed embeddings."""
    print("Loading embeddings...")
    with open("embeddings/dwug_en_embeddings.pkl", "rb") as f:
        source_token2vecs, target_token2vecs = pickle.load(f)
    return source_token2vecs, target_token2vecs


def calculate_method_tau_values(word_instance2tau, word_instance2cluster, 
                               source_token2vecs, target_token2vecs, method):
    """Calculate tau values using specified method (SUS or LDR)."""
    print(f"Calculating {method} tau values...")
    word_instance2tau_method = defaultdict(list)
    
    calc_func = utils.calc_sus if method == 'SUS' else utils.calc_ldr
    
    for lemma in word_instance2tau.keys():
        u, v = source_token2vecs[lemma], target_token2vecs[lemma]
        source_tau, target_tau = calc_func(u, v)
        taus = np.append(source_tau, target_tau)
        
        # Remove instances with cluster -1
        del_ids = [ids for ids, cluster in enumerate(word_instance2cluster[lemma]) 
                  if cluster == -1]
        taus = np.delete(taus, del_ids)
        word_instance2tau_method[lemma] = taus
    
    return word_instance2tau_method


def create_scatter_plot_with_averages(ax, x_vals, y_vals, color, method_name):
    """Create scatter plot with individual points and averaged points."""
    # Individual points
    ax.scatter(x_vals, y_vals, c=color, s=3)
    ax.scatter([], [], s=50, c=color, label=method_name)
    
    # Calculate and plot averages grouped by y-values
    df_temp = pd.DataFrame([x_vals, y_vals]).T
    grouped_means = df_temp.groupby(1).mean()
    ax.scatter(grouped_means.values, grouped_means.index,
              marker="^", label=f"Avg. {method_name}", c="black", s=50)


def create_visualization(tau_gold, tau_sus, tau_ldr, output_dir):
    """Create and save the visualization."""
    print("Creating visualization...")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    
    # Left subplot: SUS vs Gold
    create_scatter_plot_with_averages(axs[0], tau_sus, tau_gold, "limegreen", "SUS")
    axs[0].set_xlabel("SUS", fontsize=22)
    axs[0].set_ylabel(r"$\tau^*$", fontsize=22)
    
    # Right subplot: LDR vs Gold
    create_scatter_plot_with_averages(axs[1], tau_ldr, tau_gold, "limegreen", "LDR")
    axs[1].set_xlabel("LDR", fontsize=22)
    axs[1].set_ylabel(r"$\tau^*$")
    
    # Format both subplots
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.legend(fontsize=18)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    # Save figure
    output_path = f"{output_dir}/fig6.png"
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {output_path}")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Calculate tau values
    word_cluster2tau, taus_unique = calculate_tau_values(df)
    word_cluster2tau = handle_infinite_values(word_cluster2tau, taus_unique)
    
    # Load cluster assignments and map to tau values
    word_instance2cluster = load_cluster_assignments(word_cluster2tau)
    word_instance2tau = map_instances_to_tau(word_cluster2tau, word_instance2cluster)
    
    # Load embeddings
    source_token2vecs, target_token2vecs = load_embeddings()
    
    # Calculate method-specific tau values
    word_instance2tau_sus = calculate_method_tau_values(
        word_instance2tau, word_instance2cluster, 
        source_token2vecs, target_token2vecs, 'SUS'
    )
    
    word_instance2tau_ldr = calculate_method_tau_values(
        word_instance2tau, word_instance2cluster, 
        source_token2vecs, target_token2vecs, 'LDR'
    )
    
    # Concatenate all tau values for visualization
    tau_gold = np.concatenate(list(word_instance2tau.values()))
    tau_sus = np.concatenate(list(word_instance2tau_sus.values()))
    tau_ldr = np.concatenate(list(word_instance2tau_ldr.values()))
    
    # Create visualization
    create_visualization(tau_gold, tau_sus, tau_ldr, args.output_dir)
    
    print(f"Figure saved to: {output_path}")



if __name__ == "__main__":
    main()