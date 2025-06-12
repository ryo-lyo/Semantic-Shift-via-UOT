import numpy as np
from scipy.stats import spearmanr
import pickle
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import pandas as pd
import csv
from datetime import datetime
import logging
import itertools
import inspect
import metrics


def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('outputs/tau_evaluation.log'),
            logging.StreamHandler()
        ]
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate tau correlation between gold and methods")
    parser.add_argument("--output_dir", type=str, default="outputs", 
                       help="Output directory for results")
    parser.add_argument("--n_trials", type=int, default=100, 
                       help="Number of cross-validation trials")
    parser.add_argument("--train_ratio", type=float, default=0.8, 
                       help="Ratio of training data")
    parser.add_argument("--random_seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    return parser.parse_args()


def str2list(s):
    """Convert string representation of list to numpy array"""
    return np.array(eval(s))


def load_and_prepare_data():
    """Load and prepare all necessary data"""
    logging.info("Loading and preparing data...")
    
    # Load statistics data
    df = pd.read_table("data/dwug_en/stats/opt/stats_groupings.csv", quoting=csv.QUOTE_NONE)
    df["cluster_prob_dist1"] = df["cluster_prob_dist1"].apply(str2list)
    df["cluster_prob_dist2"] = df["cluster_prob_dist2"].apply(str2list)
    
    # Calculate gold tau values
    word_cluster2tau = defaultdict(dict)
    taus_unique = []
    
    for i, row in df.iterrows():
        taus = np.log(row["cluster_prob_dist2"] / row["cluster_prob_dist1"])
        taus_unique.extend(taus)
        cluster2tau = {cluster: tau for cluster, tau in zip(range(len(taus)), taus)}
        word_cluster2tau[row["lemma"]] = cluster2tau
    
    # Handle infinite values
    finite_values = np.array(taus_unique)[~np.isinf(taus_unique)]
    max_finite = np.max(finite_values)
    min_finite = np.min(finite_values)
    
    for outer_key, subdict in word_cluster2tau.items():
        for inner_key, value in subdict.items():
            if value == np.inf:
                word_cluster2tau[outer_key][inner_key] = max_finite
            elif value == -np.inf:
                word_cluster2tau[outer_key][inner_key] = min_finite
    
    # Load cluster assignments
    word_instance2cluster = defaultdict(list)
    for lemma in word_cluster2tau.keys():
        df_cluster = pd.read_table(f"data/dwug_en/clusters/opt/{lemma}.csv")
        word_instance2cluster[lemma] = df_cluster["cluster"].tolist()
    
    # Map instances to gold tau values
    word_instance2tau_gold = defaultdict(list)
    for lemma, clusters in word_instance2cluster.items():
        taus = [word_cluster2tau[lemma][cluster] 
                for cluster in clusters if cluster != -1]
        word_instance2tau_gold[lemma] = taus
    
    # Load embeddings
    with open("embeddings/dwug_en_embeddings.pkl", "rb") as f:
        source_token2vecs, target_token2vecs = pickle.load(f)
    
    logging.info(f"Loaded data for {len(word_instance2tau_gold)} words")
    
    return (word_instance2tau_gold, word_instance2cluster, 
            source_token2vecs, target_token2vecs)


def get_parameter_grid(method_name):
    """Get parameter grid for hyperparameter search based on method"""
    if method_name == "SUS":
        return [10, 20, 50, 100, 200, 500, 1000]
    elif method_name == "LDR":
        # LDR doesn't have hyperparameters in the original implementation
        return [None]
    elif method_name == "WiDiD":
        return [0.5, 0.6, 0.7, 0.8, 0.9]
    else:
        return [None]


def calculate_method_tau_values_with_params(lemmas, word_instance2cluster, 
                                          source_token2vecs, target_token2vecs, 
                                          method_name, param=None):
    """Calculate tau values using specified method with optional parameters"""
    word_instance2tau_method = defaultdict(list)
    
    if method_name == "SUS":
        calc_func = metrics.tau_SUS
    elif method_name == "LDR":
        calc_func = metrics.tau_LDR
    elif method_name == "WiDiD":
        calc_func = metrics.tau_WiDiD
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    # Get function signature to determine parameter handling
    try:
        signature = inspect.signature(calc_func)
        params = signature.parameters
    except:
        params = {}
    
    for lemma in lemmas:
        if lemma in source_token2vecs and lemma in target_token2vecs:
            u, v = source_token2vecs[lemma], target_token2vecs[lemma]
            
            # Call function with appropriate parameters
            if param is None:
                # No hyperparameters
                source_tau, target_tau = calc_func(u, v)
            elif 'reg_m' in params:
                # Only reg_m
                source_tau, target_tau = calc_func(u, v, reg_m=param)
            elif 'damping' in params:
                # Only damping
                source_tau, target_tau = calc_func(u, v, damping=param)
            else:
                # Default case
                source_tau, target_tau = calc_func(u, v)
            
            taus = np.append(source_tau, target_tau)
            
            # Remove instances with cluster -1
            del_ids = [ids for ids, cluster in enumerate(word_instance2cluster[lemma]) 
                      if cluster == -1]
            taus = np.delete(taus, del_ids)
            word_instance2tau_method[lemma] = taus
    
    return word_instance2tau_method


def precompute_all_method_results(method_name, all_lemmas, word_instance2cluster,
                                source_token2vecs, target_token2vecs):
    """Precompute tau values for all parameters and all lemmas"""
    logging.info(f"Precomputing results for method: {method_name}")
    
    param_grid = get_parameter_grid(method_name)
    param_to_results = {}
    
    for param in param_grid:
        logging.info(f"  Computing for parameter: {param}")
        param_to_results[param] = calculate_method_tau_values_with_params(
            all_lemmas, word_instance2cluster, 
            source_token2vecs, target_token2vecs, method_name, param
        )
    
    return param_to_results


def evaluate_correlation(tau_gold_flat, tau_method_flat, cluster_ids=None):
    """Calculate correlation metrics between gold and method tau values"""
    if len(tau_gold_flat) == 0 or len(tau_method_flat) == 0:
        return {'instance_level': 0.0, 'sense_level': 0.0}
    
    try:
        # Instance-Level correlation (original spearman)
        instance_spearman, _ = spearmanr(tau_gold_flat, tau_method_flat)
        instance_spearman = instance_spearman if not np.isnan(instance_spearman) else 0.0
        
        # Sense-Level correlation
        sense_spearman = 0.0
        if cluster_ids is not None and len(cluster_ids) > 0:
            try:
                sense_spearman, _ = spearmanr(pd.DataFrame([tau_gold_flat, tau_method_flat]).T.groupby(0).mean().values,
                                              pd.DataFrame([tau_gold_flat, tau_method_flat]).T.groupby(0).mean().index)
                sense_spearman = sense_spearman if not np.isnan(sense_spearman) else 0.0
            except Exception as e:
                logging.warning(f"Error calculating sense-level correlation: {e}")
                sense_spearman = 0.0
        
        return {
            'instance_level': instance_spearman, 
            'sense_level': sense_spearman
        }
    
    except Exception as e:
        logging.warning(f"Error calculating correlation: {e}")
        return {'instance_level': 0.0, 'sense_level': 0.0}


def evaluate_parameter_on_lemmas(param, param_to_results, lemmas, 
                                word_instance2tau_gold, word_instance2cluster, method_name=None):
    """Evaluate a specific parameter on given lemmas"""
    word_instance2tau_method = param_to_results[param]
    
    tau_gold_flat = []
    tau_method_flat = []
    cluster_ids = []
    
    for lemma in lemmas:
        if (lemma in word_instance2tau_gold and 
            lemma in word_instance2tau_method and
            len(word_instance2tau_gold[lemma]) > 0 and
            len(word_instance2tau_method[lemma]) > 0):
            
            gold_taus = word_instance2tau_gold[lemma]
            method_taus = word_instance2tau_method[lemma]
            
            # Get cluster information for sense-level evaluation
            lemma_clusters = [c for c in word_instance2cluster[lemma] if c != -1]
            
            # Ensure same length
            min_len = min(len(gold_taus), len(method_taus), len(lemma_clusters))
            tau_gold_flat.extend(gold_taus[:min_len])
            tau_method_flat.extend(method_taus[:min_len])
            cluster_ids.extend(lemma_clusters[:min_len])
    
    if len(tau_gold_flat) == 0 or len(tau_method_flat) == 0:
        return 0.0, {'instance_level': 0.0, 'sense_level': 0.0}
    
    # Calculate training correlation for hyperparameter selection
    if method_name == "WiDiD":
        max_val = max([val for val in tau_method_flat if val != np.inf], default=0.0)
        min_val = min([val for val in tau_method_flat if val != -np.inf], default=0.0)
        tau_method_flat = [max_val if val == np.inf else min_val if val == -np.inf else val 
                           for val in tau_method_flat]
    train_corr, _ = spearmanr(tau_gold_flat, tau_method_flat)
    train_corr = train_corr if not np.isnan(train_corr) else 0.0
    
    # Calculate full correlations for final evaluation
    correlations = evaluate_correlation(tau_gold_flat, tau_method_flat, cluster_ids)
    
    return train_corr, correlations


def cross_validate_method_efficient(method_name, word_instance2tau_gold, word_instance2cluster,
                                   source_token2vecs, target_token2vecs, n_trials=100, 
                                   train_ratio=0.8, random_seed=42):
    """Perform efficient cross-validation for tau correlation evaluation with precomputed results"""
    logging.info(f"Evaluating method: {method_name}")
    
    np.random.seed(random_seed)
    all_lemmas = list(word_instance2tau_gold.keys())
    
    # Precompute all method results for all parameters
    param_to_results = precompute_all_method_results(
        method_name, all_lemmas, word_instance2cluster,
        source_token2vecs, target_token2vecs
    )
    
    param_grid = list(param_to_results.keys())
    
    instance_scores = []
    sense_scores = []
    selected_params = []
    
    for trial in range(n_trials):
        if trial % 10 == 0:
            logging.info(f"  Trial {trial + 1}/{n_trials}")
            
        # Random split of words
        n_train = int(len(all_lemmas) * train_ratio)
        train_lemmas = np.random.choice(all_lemmas, n_train, replace=False).tolist()
        test_lemmas = [lemma for lemma in all_lemmas if lemma not in train_lemmas]
        
        if len(test_lemmas) == 0:
            continue
        
        # Hyperparameter optimization on training set
        best_param = None
        best_score = -np.inf
        
        if len(param_grid) > 1:  # Only optimize if there are hyperparameters
            for param in param_grid:
                train_corr, _ = evaluate_parameter_on_lemmas(
                    param, param_to_results, train_lemmas, 
                    word_instance2tau_gold, word_instance2cluster,
                    method_name
                )
                
                if train_corr > best_score:
                    best_score = train_corr
                    best_param = param
        else:
            best_param = param_grid[0]
        
        selected_params.append(best_param)
        
        # Evaluate on test set with best parameter
        _, correlations = evaluate_parameter_on_lemmas(
            best_param, param_to_results, test_lemmas, 
            word_instance2tau_gold, word_instance2cluster,
            method_name
        )
        
        instance_scores.append(correlations['instance_level'])
        sense_scores.append(correlations['sense_level'])
    
    # Calculate statistics
    mean_instance = np.mean(instance_scores)
    mean_sense = np.mean(sense_scores)
    
    # Find most frequent parameter
    param_counter = Counter(selected_params)
    most_common_param = param_counter.most_common(1)[0] if selected_params else (None, 0)
    
    return {
        'method': method_name,
        'mean_instance_level': mean_instance,
        'mean_sense_level': mean_sense,
        'n_trials': len(instance_scores),
        'most_common_param': most_common_param
    }


def format_parameter(param):
    """Format parameter for display"""
    if param is None:
        return "N/A"
    elif isinstance(param, tuple):
        return f"reg_m={param[0]}, theta={param[1]}"
    else:
        return str(param)


def save_results(results, output_dir):
    """Save results to table2.txt"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results_path = output_path / "table2.txt"
    
    with open(results_path, 'w') as f:
        # Write header
        f.write("=" * 90 + "\n")
        f.write("Tau Correlation Evaluation Results (with Efficient Hyperparameter Optimization)\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 90 + "\n\n")
        
        # Write summary table
        f.write("SUMMARY TABLE\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Method':<10} {'Instance-Level':<15} {'Sense-Level':<15} {'Trials':<8} {'Best Param':<30}\n")
        f.write("-" * 90 + "\n")
        
        for result in results:
            param_str = format_parameter(result['most_common_param'][0])
            f.write(f"{result['method']:<10} "
                   f"{result['mean_instance_level']:<15.2f} "
                   f"{result['mean_sense_level']:<15.2f} "
                   f"{result['n_trials']:<8} "
                   f"{param_str:<30}\n")
        
        f.write("-" * 90 + "\n\n")
        
        # Write detailed results
        f.write("DETAILED RESULTS\n")
        f.write("-" * 90 + "\n")
        
        for result in results:
            f.write(f"\nMethod: {result['method']}\n")
            f.write(f"  Instance-Level Spearman: {result['mean_instance_level']:.2f}\n")
            f.write(f"  Sense-Level Spearman:    {result['mean_sense_level']:.2f}\n")
            f.write(f"  Number of trials:        {result['n_trials']}\n")
            f.write(f"  Most Common Parameter:   {format_parameter(result['most_common_param'][0])} "
                   f"(selected {result['most_common_param'][1]} times)\n")
        
        f.write("\n" + "=" * 90 + "\n")
    
    logging.info(f"Results saved to {results_path}")


def main():
    """Main execution function"""
    setup_logging()
    args = parse_args()
    
    # Load and prepare data
    (word_instance2tau_gold, word_instance2cluster, 
     source_token2vecs, target_token2vecs) = load_and_prepare_data()
    
    # Define methods to evaluate
    methods = ["SUS", "LDR", "WiDiD"]
    
    results = []
    
    # Evaluate each method
    for method_name in methods:
        try:
            result = cross_validate_method_efficient(
                method_name, word_instance2tau_gold, word_instance2cluster,
                source_token2vecs, target_token2vecs, 
                n_trials=args.n_trials, train_ratio=args.train_ratio,
                random_seed=args.random_seed
            )
            results.append(result)
            
            logging.info(f"Method: {method_name}, "
                        f"Instance-Level: {result['mean_instance_level']:.2f}, "
                        f"Sense-Level: {result['mean_sense_level']:.2f}, "
                        f"Best Param: {format_parameter(result['most_common_param'][0])}")
            
        except Exception as e:
            logging.error(f"Error evaluating method {method_name}: {str(e)}")
            continue
    
    # Save results
    save_results(results, args.output_dir)
    
    # Print summary to console
    if results:
        print("\n" + "="*50)
        print("TAU CORRELATION EVALUATION SUMMARY")
        print("="*50)
        for result in results:
            print(f"{result['method']:<10}: Instance={result['mean_instance_level']:.2f}, "
                  f"Sense={result['mean_sense_level']:.2f}")
        print("="*50)
    else:
        logging.error("No methods were successfully evaluated")


if __name__ == "__main__":
    main()