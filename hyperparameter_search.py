#!/usr/bin/env python3
import torch
import json
import argparse
import pandas as pd
import numpy as np
import os
import gc
from itertools import product
from sklearn.model_selection import ParameterGrid
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Import from training module
from train_pruning_model import train_model, HP

# Check for GPU availability and setup optimal configurations
def setup_gpu_optimizations():
    """Configure optimal GPU settings for faster training."""
    if torch.cuda.is_available():
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Get GPU info
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        
        print(f"\n{'='*50}")
        print(f"GPU Setup:")
        print(f"  Device: {gpu_name}")
        print(f"  Count: {gpu_count}")
        print(f"  Memory: {gpu_memory:.2f} GB")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  cuDNN enabled: {torch.backends.cudnn.enabled}")
        print(f"  Benchmark mode: {torch.backends.cudnn.benchmark}")
        print(f"{'='*50}\n")
        
        # Enable mixed precision if available
        use_amp = True if hasattr(torch.cuda, 'amp') else False
        
        return {
            "available": True,
            "device_count": gpu_count,
            "device_name": gpu_name,
            "memory": gpu_memory,
            "use_amp": use_amp
        }
    else:
        print("No GPU available. Running on CPU only.")
        return {"available": False}

def clean_gpu_memory():
    """Clean up GPU memory between runs."""
    if torch.cuda.is_available():
        # Safe GPU memory cleanup
        try:
            # Make sure all tensors are moved off GPU first
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"Warning: Could not clean GPU memory: {e}")
            # Try to recover from memory errors
            import sys
            sys.stdout.flush()
            sys.stderr.flush()

def get_optimal_batch_size(gpu_info):
    """Estimate optimal batch size based on GPU memory."""
    if not gpu_info["available"]:
        return HP['batch_size']  # Use default if no GPU
        
    # Calculate batch size based on GPU memory, but be very conservative
    # Regardless of memory size, keep batch size small to ensure training stability
    base_batch_size = HP['batch_size']
    
    # For mixed precision training we can use slightly larger batches
    if gpu_info.get("use_amp", False):
        return min(32, base_batch_size)  # Cap at 32 for stability with mixed precision
    else:
        return min(16, base_batch_size)  # Cap at 16 for stability with full precision

def get_reduced_parameter_grid():
    """Return a smaller parameter grid for faster experimentation."""
    return {
        'lr': [3e-5],  # Reduced to most common good value
        'lambda_urs': [0.5],  # Balanced value
        'lambda_ws': [0.5],    # Balanced value
        'pos_weight': [8.0],   # Median value
        'dropout': [0.3],      # Median value
        'focal_gamma': [2.0],  # Median value
        'focal_alpha': [0.75], # Median value
        'weight_decay': [5e-5] # Median value
    }

def train_with_params(params, train_csv, val_split, test_csv, model_name, epochs, output_dir, 
                      seed, run_id, gpu_info, early_stopping_f1=0.5, max_patience=2):
    """Train a model with given parameters and return results."""
    print(f"\n\n{'='*50}")
    print(f"Training model {run_id} with parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print(f"{'='*50}\n")
    
    # Create directory for this run's logs only (no model saving)
    run_dir = os.path.join(output_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save hyperparameters
    with open(os.path.join(run_dir, "hyperparameters.json"), 'w') as f:
        json.dump(params, f, indent=2)
    
    # Update the HP dictionary in train_pruning_model.py
    from train_pruning_model import HP as train_HP
    for key, value in params.items():
        train_HP[key] = value
    
    # Set conservative batch size for stability with mixed precision
    if gpu_info["available"]:
        batch_size = get_optimal_batch_size(gpu_info)
        train_HP['batch_size'] = batch_size
        print(f"Using conservative batch size: {batch_size}")
    
    use_amp = gpu_info.get("use_amp", False)
    
    try:
        # Clean memory before starting
        clean_gpu_memory()
        
        # Run training with optimized settings
        model_output = train_model(
            train_csv=train_csv,
            val_split=val_split,
            test_csv=test_csv,
            model_name=model_name,
            epochs=epochs,
            output_dir=run_dir,
            seed=seed,
            save_model=False,  # Don't save model for each run
            use_amp=use_amp,   # Use mixed precision if available
            early_stopping_f1=early_stopping_f1,  # Stop if F1 is below threshold
            max_patience=max_patience  # Reduce patience for faster failing
        )
        
        # Extract metrics from history
        history = model_output['history']
        
        # Get best validation F1 score
        val_f1_scores = [metrics['f1'] for metrics in history['val_metrics']]
        best_val_epoch = np.argmax(val_f1_scores) if val_f1_scores else -1
        best_val_f1_score = val_f1_scores[best_val_epoch] if best_val_epoch >= 0 else 0.0
        
        # Get corresponding metrics
        best_val_precision = history['val_metrics'][best_val_epoch]['precision'] if best_val_epoch >= 0 else 0.0
        best_val_recall = history['val_metrics'][best_val_epoch]['recall'] if best_val_epoch >= 0 else 0.0
        
        # Store results
        result = {
            'run_id': run_id,
            'params': params,
            'best_val_f1': best_val_f1_score,
            'best_val_precision': best_val_precision,
            'best_val_recall': best_val_recall,
            'best_val_epoch': best_val_epoch + 1 if best_val_epoch >= 0 else 0,
            'output_dir': run_dir,
            'completed': True,
            'error': None
        }
        
        # Add test metrics if available
        if history['test_metrics']:
            result['test_f1'] = history['test_metrics'][0]['f1']
            result['test_precision'] = history['test_metrics'][0]['precision']
            result['test_recall'] = history['test_metrics'][0]['recall']
        
        # Return model output for potential saving
        return result, model_output
    
    except Exception as e:
        print(f"Error in run {run_id}: {e}")
        result = {
            'run_id': run_id,
            'params': params,
            'completed': False,
            'error': str(e)
        }
        return result, None
    finally:
        # Clean GPU memory after run
        clean_gpu_memory()

def run_grid_search(train_csv, val_split=0.15, test_csv=None, model_name="bert-base-uncased",
                   epochs=10, base_output_dir="hyperparameter_search", seed=42, resume_from=None,
                   use_reduced_grid=False, parallel_runs=0, early_stopping_f1=0.5, use_amp=False):
    """
    Run grid search for hyperparameter optimization with GPU optimizations.
    
    Args:
        train_csv: Path to training CSV
        val_split: Proportion of training data to use for validation
        test_csv: Path to test CSV (optional, only used for final evaluation)
        model_name: Name of the pre-trained model to use
        epochs: Number of training epochs
        base_output_dir: Directory to save trained models
        seed: Random seed for reproducibility
        resume_from: Path to a previous grid search results file to resume from
        use_reduced_grid: Use a smaller parameter grid for faster search
        parallel_runs: Number of parallel runs (0 for sequential)
        early_stopping_f1: Minimum F1 score threshold for early stopping
        use_amp: Use mixed precision training for faster computation (DISABLED - causes issues)
    """
    # Setup GPU optimizations
    gpu_info = setup_gpu_optimizations()
    
    # FORCE DISABLE mixed precision due to compatibility issues
    use_amp = False
    gpu_info["use_amp"] = False
    print("\nMixed precision training disabled due to compatibility issues")
    
    # Handle resuming from previous run
    previous_results = []
    completed_params = set()
    
    if resume_from and os.path.exists(resume_from):
        try:
            print(f"Resuming grid search from {resume_from}")
            previous_df = pd.read_csv(resume_from)
            previous_results = previous_df.to_dict('records')
            
            # Extract previously completed parameter combinations
            for result in previous_results:
                if 'params' in result and isinstance(result['params'], dict):
                    param_tuple = tuple(sorted(result['params'].items()))
                    completed_params.add(param_tuple)
                elif 'params' in result and isinstance(result['params'], str):
                    # Handle case where params are stored as string
                    try:
                        param_dict = eval(result['params'].replace('true', 'True').replace('false', 'False'))
                        param_tuple = tuple(sorted(param_dict.items()))
                        completed_params.add(param_tuple)
                    except:
                        pass
            
            print(f"Found {len(previous_results)} previous results with {len(completed_params)} unique parameter combinations")
        except Exception as e:
            print(f"Error loading previous results: {e}")
            previous_results = []
            completed_params = set()
    
    # Create base output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{base_output_dir}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # If resuming, copy the previous results file to the new output directory
    if previous_results:
        import shutil
        shutil.copy(resume_from, os.path.join(output_dir, "previous_results.csv"))
    
    # Define hyperparameter grid
    if use_reduced_grid:
        param_grid = get_reduced_parameter_grid()
        print("\nUsing reduced parameter grid for faster search:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
    else:
        param_grid = {
            'lr': [1e-5, 3e-5, 5e-5],
            'lambda_urs': [0.3, 0.5, 0.7],
            'lambda_ws': [0.3, 0.5, 0.7],
            'pos_weight': [5.0, 8.0, 12.0],
            'dropout': [0.2, 0.3, 0.4],
            'focal_gamma': [1.5, 2.0, 2.5],
            'focal_alpha': [0.6, 0.75, 0.85],
            'weight_decay': [1e-5, 5e-5, 1e-4],
        }
    
    # Custom parameter constraints
    def is_valid_param_combo(params):
        # Ensure lambda_urs and lambda_ws sum to approximately 1.0
        if abs(params['lambda_urs'] + params['lambda_ws'] - 1.0) > 0.01:
            return False
        return True
    
    # Generate all combinations
    all_params = list(ParameterGrid(param_grid))
    valid_params = [p for p in all_params if is_valid_param_combo(p)]
    
    # Filter out already completed parameter combinations if resuming
    if completed_params:
        filtered_params = []
        for params in valid_params:
            param_tuple = tuple(sorted(params.items()))
            if param_tuple not in completed_params:
                filtered_params.append(params)
        
        print(f"Filtered {len(valid_params)} valid combinations to {len(filtered_params)} remaining combinations")
        valid_params = filtered_params
    
    print(f"Generated {len(valid_params)} valid parameter combinations to run")
    
    # Initialize results storage with previous results if resuming
    results = previous_results.copy() if previous_results else []
    
    # Track the best model performance
    best_val_f1 = 0.0
    best_run_id = None
    best_params = None
    best_model_path = None
    
    # Determine optimal parallel processing
    if parallel_runs <= 0 and gpu_info["available"] and gpu_info["device_count"] > 1:
        # Auto-detect parallel runs based on GPU count
        parallel_runs = gpu_info["device_count"]
        print(f"Auto-detected {parallel_runs} GPUs for parallel processing")
    elif parallel_runs > 0:
        print(f"Using {parallel_runs} parallel processes as specified")
    else:
        print("Running in sequential mode (no parallel processing)")
    
    # Process parameter combinations
    if parallel_runs > 1 and len(valid_params) > 1:
        # Parallel processing mode
        print(f"\nRunning hyperparameter search in parallel with {parallel_runs} processes")
        
        # Create pool with the specified number of processes
        pool = mp.Pool(processes=parallel_runs)
        
        # Partial function for parallel execution
        train_func = partial(
            train_with_params,
            train_csv=train_csv,
            val_split=val_split,
            test_csv=test_csv,
            model_name=model_name,
            epochs=epochs,
            output_dir=output_dir,
            seed=seed,
            gpu_info=gpu_info,
            early_stopping_f1=early_stopping_f1
        )
        
        # Generate run_ids
        start_run_id = len(results) + 1
        run_ids = list(range(start_run_id, start_run_id + len(valid_params)))
        
        # Run training in parallel
        parallel_results = []
        for i, (params, run_id) in enumerate(zip(valid_params, run_ids)):
            parallel_results.append(pool.apply_async(train_func, args=(params, run_id)))
        
        # Close the pool and wait for all processes to complete
        pool.close()
        
        # Process results as they complete
        for i, async_result in enumerate(tqdm(parallel_results, desc="Training models")):
            result, model_output = async_result.get()
            
            # Add to results and save
            results.append(result)
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(output_dir, "grid_search_results.csv"), index=False)
            
            # Check if this is the best model so far
            if result.get('completed', False) and result.get('best_val_f1', 0) > best_val_f1:
                best_val_f1 = result['best_val_f1']
                best_run_id = result['run_id']
                best_params = result['params']
                
                # If this is the best model, save it
                if model_output and 'model' in model_output:
                    # Save the best model directly to the main output directory
                    best_model_path = os.path.join(output_dir, "best_model.pt")
                    torch.save(model_output['model'].state_dict(), best_model_path)
                    print(f"New best model saved with F1={best_val_f1:.4f}")
                
                print("\nNew best run:")
                print(f"Run {best_run_id}: F1={best_val_f1:.4f}, "
                      f"Precision={result.get('best_val_precision', 0):.4f}, "
                      f"Recall={result.get('best_val_recall', 0):.4f}")
        
        pool.join()
        
    else:
        # Sequential processing
        for i, params in enumerate(tqdm(valid_params, desc="Training models")):
            # Calculate run_id
            run_id = len(results) + 1
            
            # Train with parameters
            result, model_output = train_with_params(
                params,
                train_csv,
                val_split,
                test_csv,
                model_name,
                epochs,
                output_dir,
                seed,
                run_id,
                gpu_info,
                early_stopping_f1=early_stopping_f1
            )
            
            results.append(result)
            
            # Save updated results after each run
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(output_dir, "grid_search_results.csv"), index=False)
            
            if result.get('completed', False):
                # Check if this is the best model so far
                if result.get('best_val_f1', 0) > best_val_f1:
                    best_val_f1 = result['best_val_f1']
                    best_run_id = run_id
                    best_params = params
                    
                    # If this is the best model, save it
                    if model_output and 'model' in model_output:
                        # Save the best model directly to the main output directory
                        best_model_path = os.path.join(output_dir, "best_model.pt")
                        torch.save(model_output['model'].state_dict(), best_model_path)
                        print(f"New best model saved with F1={best_val_f1:.4f}")
                
                # Print current best
                print("\nCurrent best run:")
                print(f"Run {best_run_id}: F1={best_val_f1:.4f}, "
                      f"Precision={result.get('best_val_precision', 0):.4f}, "
                      f"Recall={result.get('best_val_recall', 0):.4f}")
                print(f"Parameters: {best_params}")
    
    # Final analysis
    if results:
        # Filter only completed runs
        completed_results = [r for r in results if r.get('completed', False)]
        
        if completed_results:
            results_df = pd.DataFrame(completed_results)
            results_path = os.path.join(output_dir, "grid_search_results.csv")
            results_df.to_csv(results_path, index=False)
            print(f"\nAll results saved to {results_path}")
            
            # Find best parameters
            best_idx = results_df['best_val_f1'].idxmax()
            best_run = results_df.iloc[best_idx]
            
            print("\nBest hyperparameters:")
            for k, v in best_run['params'].items():
                print(f"  {k}: {v}")
            
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
            
            # The best model is already saved directly to the output directory,
            # so we don't need to copy it anymore
            if best_model_path and os.path.exists(best_model_path):
                print(f"Best model already saved to {best_model_path}")
            else:
                print(f"Warning: Best model was not saved during training")
            
            return completed_results
        else:
            print("No completed runs to analyze!")
            return results
    else:
        print("No results to analyze!")
        return []

def analyze_hyperparameter_importance(results_df):
    """
    Analyze which hyperparameters have the most impact on model performance.
    
    Args:
        results_df: DataFrame with grid search results
    """
    print("\nAnalyzing hyperparameter importance...")
    
    # Extract parameter columns
    param_columns = {}
    for i, row in results_df.iterrows():
        for param, value in row['params'].items():
            if param not in param_columns:
                param_columns[param] = []
            param_columns[param].append(value)
    
    # Create dataframe with parameters and F1 score
    analysis_df = pd.DataFrame({
        **{k: v for k, v in param_columns.items()},
        'f1_score': results_df['best_val_f1']
    })
    
    # Calculate correlation with F1 score
    correlations = {param: analysis_df[param].corr(analysis_df['f1_score']) 
                   for param in param_columns.keys()}
    
    # Print correlations
    print("\nParameter correlations with F1 score:")
    for param, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {param}: {corr:.4f}")
    
    # Group by each parameter and calculate mean F1
    param_impact = {}
    for param in param_columns.keys():
        grouped = analysis_df.groupby(param)['f1_score'].mean()
        param_impact[param] = grouped.max() - grouped.min()
    
    # Print parameter impact
    print("\nParameter impact on F1 score (max difference):")
    for param, impact in sorted(param_impact.items(), key=lambda x: x[1], reverse=True):
        print(f"  {param}: {impact:.4f}")
    
    # Generate plot of parameter impact
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        
        # Sort parameters by impact
        sorted_params = sorted(param_impact.items(), key=lambda x: x[1], reverse=True)
        
        # Plot bar chart
        plt.bar([p[0] for p in sorted_params], [p[1] for p in sorted_params])
        plt.xlabel("Hyperparameter")
        plt.ylabel("Impact on F1 Score")
        plt.title("Hyperparameter Impact Analysis")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig("hyperparameter_impact.png")
        print("\nHyperparameter impact plot saved to hyperparameter_impact.png")
    except Exception as e:
        print(f"Could not generate plot: {e}")

def optimize_hyperparameters(results_df):
    """
    Suggest optimized hyperparameters based on grid search results.
    
    Args:
        results_df: DataFrame with grid search results
    """
    print("\nOptimizing hyperparameters based on top performing runs...")
    
    # Take top 10% of runs or at least 3 runs
    top_n = max(3, int(len(results_df) * 0.1))
    top_runs = results_df.nlargest(top_n, 'best_val_f1')
    
    # Calculate weighted average for each parameter
    optimized_params = {}
    
    # Get parameter names
    param_names = list(top_runs.iloc[0]['params'].keys())
    
    for param in param_names:
        # Extract parameter values and weights (F1 scores)
        values = np.array([run['params'][param] for _, run in top_runs.iterrows()])
        weights = np.array([run['best_val_f1'] for _, run in top_runs.iterrows()])
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Calculate weighted average
        weighted_avg = np.sum(values * weights)
        
        # For discrete parameters, find closest valid value
        if param in ['batch_size'] or param.endswith('_size'):
            weighted_avg = int(round(weighted_avg))
        
        optimized_params[param] = weighted_avg
    
    print("\nOptimized hyperparameters (weighted average of top runs):")
    for param, value in optimized_params.items():
        print(f"  {param}: {value}")
    
    # Save optimized hyperparameters
    with open("optimized_hyperparameters.json", 'w') as f:
        json.dump(optimized_params, f, indent=2)
    
    print("\nOptimized hyperparameters saved to optimized_hyperparameters.json")
    
    return optimized_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter grid search for model training with GPU optimizations")
    parser.add_argument("--train", type=str, default="labeled_data/train_data.csv",
                        help="Path to training data CSV")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Proportion of training data to use for validation")
    parser.add_argument("--test", type=str, default="labeled_data/test_data.csv",
                        help="Path to test data CSV (only used for final evaluation)")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="Pretrained model name")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--output", type=str, default="hyperparameter_search",
                        help="Directory to save trained models")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--use_reduced_grid", action="store_true",
                        help="Use a reduced parameter grid for faster search")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a previous grid search results file to resume from")
    parser.add_argument("--parallel_runs", type=int, default=0,
                        help="Number of parallel training runs (0 for auto-detect)")
    parser.add_argument("--early_stopping_f1", type=float, default=0.0,
                        help="Early stopping threshold for F1 score (0.0 to disable)")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use mixed precision training for faster computation")
    
    args = parser.parse_args()
    
    # Start time
    start_time = datetime.now()
    print(f"Starting hyperparameter search at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run grid search
    results = run_grid_search(
        train_csv=args.train,
        val_split=args.val_split,
        test_csv=args.test,
        model_name=args.model,
        epochs=args.epochs,
        base_output_dir=args.output,
        seed=args.seed,
        resume_from=args.resume_from,
        use_reduced_grid=args.use_reduced_grid,
        parallel_runs=args.parallel_runs,
        early_stopping_f1=args.early_stopping_f1,
        use_amp=args.use_amp
    )
    
    # End time
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nHyperparameter search completed in {duration}")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Analyze results if available
    if results:
        results_df = pd.DataFrame([r for r in results if r.get('completed', False)])
        if not results_df.empty:
            analyze_hyperparameter_importance(results_df)
            optimize_hyperparameters(results_df)
        else:
            print("\nNo completed runs to analyze!")
    else:
        print("\nNo results to analyze!")