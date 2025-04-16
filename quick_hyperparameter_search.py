#!/usr/bin/env python3
import torch
import json
import argparse
import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import from training module
from train_pruning_model import train_model

def run_focused_grid_search(train_csv, val_split=0.15, test_csv=None, model_name="bert-base-uncased",
                           epochs=8, base_output_dir="quick_hyperparameter_search", seed=42):
    """
    Run a focused grid search with carefully selected hyperparameter combinations.
    
    Args:
        train_csv: Path to training CSV
        val_split: Proportion of training data to use for validation
        test_csv: Path to test CSV (optional, only used for final evaluation)
        model_name: Name of the pre-trained model to use
        epochs: Number of training epochs
        base_output_dir: Directory to save trained models
        seed: Random seed for reproducibility
    """
    # Create base output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{base_output_dir}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define carefully selected hyperparameter combinations optimized for F1 score
    # These combinations are specifically designed to find a balance between 
    # precision and recall to maximize F1
    hyperparameter_sets = [
        # Baseline with higher learning rate
        {
            'lr': 5e-5,
            'lambda_urs': 0.5,
            'lambda_ws': 0.5,
            'pos_weight': 8.0,
            'dropout': 0.3,
            'focal_gamma': 2.0,
            'focal_alpha': 0.75,
            'weight_decay': 5e-5,
            'fp_penalty_weight': 0.2
        },
        # Higher positive weight to address class imbalance
        {
            'lr': 3e-5,
            'lambda_urs': 0.4,
            'lambda_ws': 0.6,
            'pos_weight': 12.0,  # Increased positive weight
            'dropout': 0.25,     # Less dropout
            'focal_gamma': 1.5,  # Less focus on hard examples
            'focal_alpha': 0.8,  # More weight on positive class
            'weight_decay': 1e-5,
            'fp_penalty_weight': 0.1  # Reduced FP penalty to improve recall
        },
        # More emphasis on weak supervision with higher WS weight
        {
            'lr': 4e-5,
            'lambda_urs': 0.3,
            'lambda_ws': 0.7,    # More weight on supervised component
            'pos_weight': 10.0,
            'dropout': 0.2,
            'focal_gamma': 2.0,
            'focal_alpha': 0.85,
            'weight_decay': 2e-5,
            'fp_penalty_weight': 0.15
        },
        # Balanced configuration with lower gamma
        {
            'lr': 3e-5,
            'lambda_urs': 0.5,
            'lambda_ws': 0.5,
            'pos_weight': 9.0,
            'dropout': 0.35,
            'focal_gamma': 1.0,  # Low gamma to be less harsh on easy examples
            'focal_alpha': 0.75,
            'weight_decay': 3e-5,
            'fp_penalty_weight': 0.25
        },
        # Configuration focused on recall
        {
            'lr': 5e-5,
            'lambda_urs': 0.4,
            'lambda_ws': 0.6,
            'pos_weight': 15.0,  # Very high positive weight
            'dropout': 0.2,
            'focal_gamma': 1.5,
            'focal_alpha': 0.9,  # Very high alpha for positive class focus
            'weight_decay': 1e-5,
            'fp_penalty_weight': 0.05  # Very low FP penalty
        },
    ]
    
    print(f"Running focused grid search with {len(hyperparameter_sets)} carefully selected parameter sets")
    
    # Initialize results storage
    results = []
    
    # Iterate through hyperparameter combinations
    for i, params in enumerate(tqdm(hyperparameter_sets, desc="Training models")):
        try:
            print(f"\n\n{'='*50}")
            print(f"Training model {i+1}/{len(hyperparameter_sets)} with parameters:")
            for k, v in params.items():
                print(f"  {k}: {v}")
            print(f"{'='*50}\n")
            
            # Create directory for this run
            run_dir = os.path.join(output_dir, f"run_{i+1}")
            os.makedirs(run_dir, exist_ok=True)
            
            # Save hyperparameters
            with open(os.path.join(run_dir, "hyperparameters.json"), 'w') as f:
                json.dump(params, f, indent=2)
            
            # Update the HP dictionary in train_pruning_model.py
            from train_pruning_model import HP
            for key, value in params.items():
                HP[key] = value
            
            # Run training
            model_output = train_model(
                train_csv=train_csv,
                val_split=val_split,
                test_csv=test_csv,
                model_name=model_name,
                epochs=epochs,
                output_dir=run_dir,
                seed=seed
            )
            
            # Extract metrics from history
            history = model_output['history']
            
            # Get best validation F1 score
            val_f1_scores = [metrics['f1'] for metrics in history['val_metrics']]
            best_val_epoch = np.argmax(val_f1_scores) if val_f1_scores else -1
            best_val_f1 = val_f1_scores[best_val_epoch] if best_val_epoch >= 0 else 0.0
            
            # Get corresponding metrics
            best_val_precision = history['val_metrics'][best_val_epoch]['precision'] if best_val_epoch >= 0 else 0.0
            best_val_recall = history['val_metrics'][best_val_epoch]['recall'] if best_val_epoch >= 0 else 0.0
            
            # Store results
            result = {
                'run_id': i+1,
                'params': params,
                'best_val_f1': best_val_f1,
                'best_val_precision': best_val_precision,
                'best_val_recall': best_val_recall,
                'best_val_epoch': best_val_epoch + 1 if best_val_epoch >= 0 else 0,
                'output_dir': run_dir
            }
            
            # Add test metrics if available
            if history['test_metrics']:
                result['test_f1'] = history['test_metrics'][0]['f1']
                result['test_precision'] = history['test_metrics'][0]['precision']
                result['test_recall'] = history['test_metrics'][0]['recall']
            
            results.append(result)
            
            # Save updated results after each run
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(output_dir, "focused_search_results.csv"), index=False)
            
            # Print current best
            if results:
                best_idx = results_df['best_val_f1'].idxmax()
                best_run = results_df.iloc[best_idx]
                print("\nCurrent best run:")
                print(f"Run {best_run['run_id']}: F1={best_run['best_val_f1']:.4f}, "
                      f"Precision={best_run['best_val_precision']:.4f}, "
                      f"Recall={best_run['best_val_recall']:.4f}")
                print(f"Parameters: {json.dumps(best_run['params'], indent=2)}")
            
        except Exception as e:
            print(f"Error in run {i+1}: {e}")
            # Continue with next parameter set
    
    # Final analysis and visualization
    if results:
        results_df = pd.DataFrame(results)
        results_path = os.path.join(output_dir, "focused_search_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\nAll results saved to {results_path}")
        
        # Find best parameters
        best_idx = results_df['best_val_f1'].idxmax()
        best_run = results_df.iloc[best_idx]
        
        print("\nBest hyperparameters:")
        print(json.dumps(best_run['params'], indent=2))
        
        print(f"\nBest validation metrics:")
        print(f"  F1: {best_run['best_val_f1']:.4f}")
        print(f"  Precision: {best_run['best_val_precision']:.4f}")
        print(f"  Recall: {best_run['best_val_recall']:.4f}")
        print(f"  Best epoch: {best_run['best_val_epoch']}")
        
        if 'test_f1' in best_run:
            print(f"\nTest metrics with best hyperparameters:")
            print(f"  F1: {best_run['test_f1']:.4f}")
            print(f"  Precision: {best_run['test_precision']:.4f}")
            print(f"  Recall: {best_run['test_recall']:.4f}")
        
        # Save best hyperparameters separately
        best_params_path = os.path.join(output_dir, "best_hyperparameters.json")
        with open(best_params_path, 'w') as f:
            json.dump(best_run['params'], f, indent=2)
        print(f"\nBest hyperparameters saved to {best_params_path}")
        
        # Copy best model to the main output directory
        best_model_src = os.path.join(best_run['output_dir'], "best_precision_model.pt")
        best_model_dst = os.path.join(output_dir, "best_model.pt")
        
        try:
            import shutil
            if os.path.exists(best_model_src):
                shutil.copyfile(best_model_src, best_model_dst)
                print(f"Best model copied to {best_model_dst}")
            else:
                print(f"Warning: Best model not found at {best_model_src}")
        except Exception as e:
            print(f"Error copying best model: {e}")
        
        # Visualize results
        create_result_visualizations(results_df, output_dir)
    
    return results

def create_result_visualizations(results_df, output_dir):
    """Create visualizations of hyperparameter search results."""
    print("\nGenerating result visualizations...")
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set up plot style
    sns.set_style("whitegrid")
    
    # 1. Bar chart comparing F1 scores across runs
    plt.figure(figsize=(12, 6))
    run_ids = results_df['run_id'].astype(str)
    
    # Sort by F1 score
    sorted_indices = results_df['best_val_f1'].argsort().values
    sorted_run_ids = run_ids.iloc[sorted_indices].values
    sorted_f1 = results_df['best_val_f1'].iloc[sorted_indices].values
    sorted_precision = results_df['best_val_precision'].iloc[sorted_indices].values
    sorted_recall = results_df['best_val_recall'].iloc[sorted_indices].values
    
    x = np.arange(len(run_ids))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width, sorted_precision, width, label='Precision')
    rects2 = ax.bar(x, sorted_f1, width, label='F1')
    rects3 = ax.bar(x + width, sorted_recall, width, label='Recall')
    
    ax.set_xlabel('Run ID')
    ax.set_ylabel('Score')
    ax.set_title('Validation Metrics by Run (Sorted by F1 Score)')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_run_ids)
    ax.legend()
    
    # Add value labels
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90, fontsize=8)
    
    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'metrics_comparison.png'))
    
    # 2. Scatter plot of precision vs recall with F1 as color
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(results_df['best_val_precision'], 
                         results_df['best_val_recall'], 
                         c=results_df['best_val_f1'],
                         s=100, cmap='viridis', 
                         edgecolors='black', linewidths=1)
    
    plt.colorbar(scatter, label='F1 Score')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision vs Recall (Color = F1 Score)')
    plt.grid(True, alpha=0.3)
    
    # Add run number annotations
    for i, row in results_df.iterrows():
        plt.annotate(f"Run {row['run_id']}", 
                    (row['best_val_precision'], row['best_val_recall']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'precision_recall_scatter.png'))
    
    print(f"Visualizations saved to {plots_dir}")
    
    # Return figure paths for reference
    return [
        os.path.join(plots_dir, 'metrics_comparison.png'),
        os.path.join(plots_dir, 'precision_recall_scatter.png')
    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run focused hyperparameter search for model training")
    parser.add_argument("--train", type=str, default="labeled_data/train_data.csv",
                        help="Path to training data CSV")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Proportion of training data to use for validation")
    parser.add_argument("--test", type=str, default="labeled_data/test_data.csv",
                        help="Path to test data CSV (only used for final evaluation)")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="Pretrained model name")
    parser.add_argument("--epochs", type=int, default=8,
                        help="Number of training epochs")
    parser.add_argument("--output", type=str, default="quick_hyperparameter_search",
                        help="Directory to save trained models")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    results = run_focused_grid_search(
        train_csv=args.train,
        val_split=args.val_split,
        test_csv=args.test,
        model_name=args.model,
        epochs=args.epochs,
        base_output_dir=args.output,
        seed=args.seed
    )