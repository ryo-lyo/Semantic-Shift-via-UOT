import numpy as np
from scipy.stats import spearmanr, entropy
import ast
import itertools
import pickle
from pathlib import Path
from collections import defaultdict, Counter
import inspect
import argparse
import pandas as pd
import csv
from datetime import datetime
import logging

import utils
import metrics

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('outputs/evaluation.log'),
            logging.StreamHandler()
        ]
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate word-level semantic shift metrics")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for results")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of cross-validation trials")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def load_data():
    """Load and prepare data"""
    logging.info("Loading data...")
    
    # Load statistics
    df = pd.read_table("data/dwug_en/stats/opt/stats_groupings.csv", quoting=csv.QUOTE_NONE)
    id2word = dict(zip(df.index, df["lemma"]))
    word2id = dict(zip(df["lemma"], df.index))

    df['cluster_prob_dist1'] = df['cluster_prob_dist1'].apply(ast.literal_eval)
    df['cluster_prob_dist2'] = df['cluster_prob_dist2'].apply(ast.literal_eval)
    gold_entropy_diffs = (df['cluster_prob_dist2'].apply(lambda x: entropy(x)) - df['cluster_prob_dist1'].apply(lambda x: entropy(x))).to_list()
    word2gold_ent = dict(zip(df["lemma"], gold_entropy_diffs))
    
    # Load embeddings
    with open("embeddings/dwug_en_embeddings.pkl", "rb") as f:
        source_token2vecs, target_token2vecs = pickle.load(f)
    
    logging.info(f"Loaded {len(df)} words with embeddings")
    
    return df, id2word, word2id, word2gold_ent, source_token2vecs, target_token2vecs

def get_parameter_grid(metric_func):
    """Get parameter grid for hyperparameter search"""
    signature = inspect.signature(metric_func)
    params = signature.parameters
    
    if 'reg_m' in params and 'theta' in params:
        # Both reg_m and theta parameters
        regs = [10, 20, 50, 100, 200, 500, 1000]
        thetas = [0.4, 0.6, 0.8]
        return list(itertools.product(regs, thetas))
    elif 'reg_m' in params:
        # Only reg_m parameter
        return [10, 20, 50, 100, 200, 500, 1000]
    elif 'theta' in params:
        # Only theta parameter
        return [0.4, 0.6, 0.8]
    elif 'damping' in params:
        # Only damping parameter
        return [0.5, 0.6, 0.7, 0.8, 0.9]
    else:
        # No hyperparameters
        return [None]

def compute_scores_for_params(metric_func, source_token2vecs, target_token2vecs, param_grid):
    """Compute scores for all parameter combinations"""
    param_word_scores = defaultdict(dict)
    signature = inspect.signature(metric_func)
    params = signature.parameters
    
    for param in param_grid:
        word2score = {}
        for word in source_token2vecs.keys():
            u, v = source_token2vecs[word], target_token2vecs[word]
            
            if param is None:
                # No hyperparameters
                word2score[word] = metric_func(u, v)
            elif isinstance(param, tuple) and len(param) == 2:
                # Both reg_m and theta
                reg, theta = param
                word2score[word] = metric_func(u, v, reg_m=reg, theta=theta)
            elif 'reg_m' in params:
                # Only reg_m
                word2score[word] = metric_func(u, v, reg_m=param)
            elif 'theta' in params:
                # Only theta
                word2score[word] = metric_func(u, v, theta=param)
            elif 'damping' in params:
                # Only damping
                word2score[word] = metric_func(u, v, damping=param)
        
        param_word_scores[param] = word2score
    
    return param_word_scores

def cross_validate_method(method_name, metric_func, df, id2word, word2gold_ent, 
                         source_token2vecs, target_token2vecs, n_trials=100, 
                         train_ratio=0.8, random_seed=42):
    """Perform cross-validation for a single method"""
    logging.info(f"Evaluating method: {method_name}")
    
    # Get parameter grid and compute scores
    param_grid = get_parameter_grid(metric_func)
    param_word_scores = compute_scores_for_params(
        metric_func, source_token2vecs, target_token2vecs, param_grid
    )
    
    # Cross-validation
    np.random.seed(random_seed)
    performances = []
    selected_params = []
    
    for trial in range(n_trials):
        # Random split
        valid_ids = np.random.choice(
            range(len(df)), int(len(df) * train_ratio), replace=False
        )
        valid_gold_scores = [word2gold_ent[id2word[valid_id]] for valid_id in valid_ids]
        
        test_ids = [test_id for test_id in range(len(df)) if test_id not in valid_ids]
        test_gold_scores = [word2gold_ent[id2word[test_id]] for test_id in test_ids]
        
        # Find best parameter on validation set
        best_performance = -np.inf
        best_param = None
        
        for param in param_word_scores.keys():
            valid_pred_scores = [param_word_scores[param][id2word[valid_id]] for valid_id in valid_ids]
            performance = spearmanr(valid_gold_scores, valid_pred_scores)[0]
            
            if performance > best_performance:
                best_performance = performance
                best_param = param
        
        selected_params.append(best_param)
        
        # Evaluate on test set
        test_pred_scores = [param_word_scores[best_param][id2word[test_id]] for test_id in test_ids]
        test_performance = spearmanr(test_gold_scores, test_pred_scores)[0]
        performances.append(test_performance)
    
    # Calculate statistics
    mean_performance = np.mean(performances)
    
    # Find most frequent parameter
    param_counter = Counter(selected_params)
    most_common_param = param_counter.most_common(1)[0]
    
    return {
        'method': method_name,
        'mean_spearman': mean_performance,
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
    """Save results to table4.txt"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    table_path = output_path / "table4.txt"
    
    with open(table_path, 'w') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("Word-level Semantic Shift Evaluation Results\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write summary table
        f.write("SUMMARY TABLE\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Method':<15} {'Mean Spearman':<15} {'Most Common Param':<30}\n")
        f.write("-" * 60 + "\n")
        
        for result in results:
            param_str = format_parameter(result['most_common_param'][0])
            f.write(f"{result['method']:<15} {result['mean_spearman']:<15.2f} {param_str:<30}\n")
        
        f.write("-" * 60 + "\n\n")
        
        # Write detailed results
        f.write("DETAILED RESULTS\n")
        f.write("-" * 60 + "\n")
        
        for result in results:
            f.write(f"\nMethod: {result['method']}\n")
            f.write(f"  Mean Spearman Correlation: {result['mean_spearman']:.2f}\n")
            f.write(f"  Most Common Parameter: {format_parameter(result['most_common_param'][0])} "
                   f"(selected {result['most_common_param'][1]} times)\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    logging.info(f"Results saved to {table_path}")

def main():
    setup_logging()
    args = parse_args()
    
    # Load data
    df, id2word, word2id, word2gold_ent, source_token2vecs, target_token2vecs = load_data()
    
    # Define methods to evaluate
    methods = ["g_SUS", "g_vMF", "g_LDR", "g_WiDiD"]
    
    results = []
    
    # Evaluate each method
    for method_name in methods:
        try:
            metric_func = getattr(metrics, method_name)
            result = cross_validate_method(
                method_name, metric_func, df, id2word, word2gold_ent,
                source_token2vecs, target_token2vecs, 
                n_trials=args.n_trials, train_ratio=args.train_ratio,
                random_seed=args.random_seed
            )
            results.append(result)
            
            logging.info(f"Method: {method_name}, "
                        f"Spearman: {result['mean_spearman']:.2f}")
            
        except Exception as e:
            logging.error(f"Error evaluating method {method_name}: {str(e)}")
            continue
    
    # Save results
    if results:
        save_results(results, args.output_dir)
        
        # Print summary to console
        print("\n" + "="*40)
        print("EVALUATION SUMMARY")
        print("="*40)
        for result in results:
            print(f"{result['method']:<10}: {result['mean_spearman']:.2f}")
        print("="*40)
    
    else:
        logging.error("No methods were successfully evaluated")

if __name__ == "__main__":
    main()