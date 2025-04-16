#!/usr/bin/env python3
import torch
import json
import argparse
import pandas as pd
import numpy as np
import jsonlines
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

# Import the necessary components
from pruning import EmbeddingModule
from train_pruning_model import TrainableCombinedModule, StableURSModule
from inference_with_trained_model import load_trained_model, compute_chunk_embeddings_batch, ensure_directory

def evaluate_on_test_query(query_file, model_path, chunks_json_path, test_queries=None,
                           threshold=0.5, top_n_tables=None, save_relevant_chunks=True):
    """
    Evaluate the trained model on a test query.
    
    Args:
        query_file: Path to CSV file with query information
        model_path: Path to trained model
        chunks_json_path: Path to JSON file with chunks
        test_queries: List of test query numbers to filter by
        threshold: Threshold for relevance
        top_n_tables: Number of top tables to process (None = all)
        save_relevant_chunks: Whether to save relevant chunks to JSON
        
    Returns:
        dict: Evaluation metrics and relevant chunks
    """
    # Check if this query is in the test set
    query_num = int(os.path.basename(query_file).split('_')[0].replace('query', ''))
    if test_queries and query_num not in test_queries:
        print(f"Skipping query {query_num} as it's not in the test set.")
        return None
    
    # Load query file
    try:
        df = pd.read_csv(query_file)
        query = df['query'].iloc[0]
        target_table = df['target table'].iloc[0]
        target_answer = df['target answer'].iloc[0]
    except Exception as e:
        print(f"Error reading query file {query_file}: {e}")
        return None
    
    print(f"Query {query_num}: {query}")
    print(f"Target table: {target_table}")
    print(f"Target answer: {target_answer}")
    
    # Select top tables if specified
    if top_n_tables:
        df = df.head(top_n_tables)
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_module, combined_module, tokenizer = load_trained_model(model_path, device)
    
    # Load chunks
    print(f"Loading chunks for query {query_num}...")
    table_to_chunks = {}
    with jsonlines.open(chunks_json_path, 'r') as reader:
        for chunk in reader:
            meta = chunk.get('metadata', {})
            table_id = meta.get('table_name', None)
            chunk_type = meta.get('chunk_type', None)
            if table_id and chunk_type == 'row':  # Keep only row chunks
                table_to_chunks.setdefault(table_id, []).append(chunk)
    
    # Process each table
    results = []
    target_table_found = False
    all_chunks_count = 0
    all_pruned_count = 0
    
    # Store all relevant chunks for each table
    relevant_chunks_by_table = {}
    
    # Process query once
    query_inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
    
    with torch.no_grad():
        query_embedding = embedding_module(**query_inputs)
    
    # Process target table (important for evaluation)
    target_table_metrics = {}
    target_chunks = table_to_chunks.get(target_table, [])
    
    if target_chunks:
        print(f"Processing target table {target_table} with {len(target_chunks)} chunks")
        target_table_found = True
        
        # Process chunks from target table
        chunk_embeddings = compute_chunk_embeddings_batch(target_chunks, embedding_module, tokenizer, device)
        
        with torch.no_grad():
            scores, _, _, eta_uns, eta_ws = combined_module.forward_train(chunk_embeddings, query_embedding)
            scores = scores.squeeze().cpu().numpy()
            
            # Ensure scores is always an array, even for a single chunk
            if np.isscalar(scores):
                scores = np.array([scores])
        
        # Find relevant chunks based on target answer
        relevant_indices = []
        for i, chunk in enumerate(target_chunks):
            if contains_answer(chunk, target_answer):
                relevant_indices.append(i)
        
        if relevant_indices:
            # Create binary labels for all chunks (1 for contains answer, 0 otherwise)
            true_labels = np.zeros(len(target_chunks))
            true_labels[relevant_indices] = 1
            
            # Calculate metrics
            pred_labels = (scores >= threshold).astype(int)
            
            # Calculate precision, recall, F1 safely (handling singleton arrays)
            try:
                # Make sure inputs are arrays, not scalars
                if np.isscalar(true_labels):
                    true_labels = np.array([true_labels])
                if np.isscalar(pred_labels):
                    pred_labels = np.array([pred_labels])
                    
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_labels, pred_labels, average='binary', zero_division=0)
                
                # Calculate accuracy
                accuracy = accuracy_score(true_labels, pred_labels)
                
                # Calculate AUC if there are both positive and negative examples
                if len(set(true_labels)) > 1:
                    auc = roc_auc_score(true_labels, scores)
                else:
                    auc = float('nan')
            except Exception as e:
                print(f"Warning: Error calculating metrics: {e}")
                # Fallback manual calculation for singleton case
                if len(target_chunks) == 1:
                    # Handle single chunk case manually
                    true_val = true_labels.item() if isinstance(true_labels, np.ndarray) else true_labels
                    pred_val = pred_labels.item() if isinstance(pred_labels, np.ndarray) else pred_labels
                    precision = 1.0 if (pred_val == 1 and true_val == 1) else 0.0
                    recall = 1.0 if (true_val == 1 and pred_val == 1) else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    accuracy = 1.0 if pred_val == true_val else 0.0
                    auc = float('nan')
                else:
                    # For other cases, set metrics to 0
                    precision, recall, f1, accuracy, auc = 0.0, 0.0, 0.0, 0.0, float('nan')
                    
            target_table_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'auc': auc,
                'relevant_chunks': len(relevant_indices),
                'total_chunks': len(target_chunks),
                'predicted_relevant': int(np.sum(pred_labels))
            }
            
            print(f"Target table metrics: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        else:
            print(f"No relevant chunks found in target table!")
    else:
        print(f"Target table {target_table} not found in chunks!")
    
    # Process all tables for this query
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing tables for query {query_num}"):
        table_id = row['top tables']
        is_target = (table_id == target_table)
        
        # Get chunks for this table
        chunks = table_to_chunks.get(table_id, [])
        if not chunks:
            continue
        
        all_chunks_count += len(chunks)
        
        # Calculate scores
        chunk_embeddings = compute_chunk_embeddings_batch(chunks, embedding_module, tokenizer, device)
        
        with torch.no_grad():
            scores, _, _, _, _ = combined_module.forward_train(chunk_embeddings, query_embedding)
            scores = scores.squeeze().cpu().numpy()
            
            # Ensure scores is always an array, even for a single chunk
            if np.isscalar(scores):
                scores = np.array([scores])
        
        # Count pruned chunks and get their indices
        pruned_indices = np.where(scores >= threshold)[0]
        pruned_chunks_count = len(pruned_indices)
        all_pruned_count += pruned_chunks_count
        
        # Store relevant chunks for this table
        if save_relevant_chunks and pruned_chunks_count > 0:
            relevant_chunks = []
            for i in range(len(pruned_indices)):
                idx = pruned_indices[i]
                chunk_copy = chunks[idx].copy()
                # Safely handle scalar or array scores
                try:
                    # Try to treat scores as an array
                    score_value = float(scores[idx])
                except (IndexError, TypeError):
                    # If that fails, it's probably a scalar
                    score_value = float(scores)
                    
                chunk_copy['score'] = score_value
                relevant_chunks.append(chunk_copy)
            
            if relevant_chunks:
                relevant_chunks_by_table[table_id] = sorted(
                    relevant_chunks, 
                    key=lambda x: x.get('score', 0), 
                    reverse=True
                )
        
        # Add table result
        results.append({
            'table_id': table_id,
            'is_target': is_target,
            'total_chunks': len(chunks),
            'pruned_chunks': pruned_chunks_count,
            'reduction_percentage': (1 - pruned_chunks_count/len(chunks)) * 100 if len(chunks) > 0 else 0
        })
    
    # Calculate overall reduction percentage
    overall_reduction = (1 - all_pruned_count/all_chunks_count) * 100 if all_chunks_count > 0 else 0
    
    # Return metrics and relevant chunks
    return {
        'query_num': query_num,
        'query': query,
        'target_table_found': target_table_found,
        'target_table': target_table,
        'target_answer': target_answer,
        'table_results': results,
        'overall_reduction': overall_reduction,
        'target_table_metrics': target_table_metrics,
        'relevant_chunks': relevant_chunks_by_table
    }

def contains_answer(chunk, answer, fuzzy_match=0.8):
    """
    Check if chunk contains the answer with improved detection.
    
    Args:
        chunk: The chunk to check
        answer: The expected answer string
        fuzzy_match: Threshold for fuzzy matching (0.0-1.0)
    
    Returns:
        bool: True if the chunk contains the answer, False otherwise
    """
    # Get text from chunk
    if isinstance(chunk['text'], list):
        chunk_text = " ".join([str(cell) for cell in chunk['text']]).lower()
    else:
        chunk_text = str(chunk['text']).lower()
    
    # Process the answer
    answer_lower = answer.lower()
    
    # First try exact matching
    if answer_lower in chunk_text:
        return True
    
    # Try matching individual tokens for multi-part answers
    if ';' in answer:
        answer_parts = [part.strip().lower() for part in answer.split(';')]
        for part in answer_parts:
            if part and part in chunk_text:
                return True
    
    # Try matching with only alphanumeric characters
    import re
    chunk_alphanum = re.sub(r'[^a-z0-9\s]', '', chunk_text)
    answer_alphanum = re.sub(r'[^a-z0-9\s]', '', answer_lower)
    
    if answer_alphanum in chunk_alphanum:
        return True
    
    # Try to match individual words for long answers
    answer_words = set(answer_lower.split())
    chunk_words = set(chunk_text.split())
    if len(answer_words) >= 3:  # Only for answers with multiple words
        common_words = answer_words.intersection(chunk_words)
        if len(common_words) / len(answer_words) >= fuzzy_match:
            return True
    
    return False

def load_test_queries(split_info_path):
    """Load test query numbers from split info file."""
    if not os.path.exists(split_info_path):
        print(f"Warning: Split info file {split_info_path} not found!")
        return None
    
    with open(split_info_path, 'r') as f:
        split_info = json.load(f)
    
    test_queries = split_info.get('test_queries', [])
    print(f"Loaded {len(test_queries)} test query numbers")
    return test_queries

def plot_evaluation_metrics(metrics_df, output_dir):
    """
    Create plots for precision, recall, and F1 scores from evaluation metrics.
    
    Args:
        metrics_df: DataFrame with precision, recall, and F1 metrics
        output_dir: Directory to save the generated plots
    """
    # Create plots subdirectory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"\nGenerating evaluation plots in {plots_dir}...")
    
    # Calculate average metrics
    avg_precision = metrics_df['precision'].mean()
    avg_recall = metrics_df['recall'].mean()
    avg_f1 = metrics_df['f1'].mean()
    avg_accuracy = metrics_df['accuracy'].mean()
    
    # Set plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 10))
    
    # 1. Create histogram plots of precision, recall, and F1
    plt.subplot(2, 2, 1)
    metrics_df[['precision', 'recall', 'f1']].hist(bins=10, alpha=0.7, ax=plt.gca())
    plt.title("Distribution of Precision, Recall, and F1 Scores")
    plt.xlabel("Score")
    plt.ylabel("Number of Queries")
    plt.grid(True, alpha=0.3)
    
    # 2. Create bar plot comparing average metrics
    plt.subplot(2, 2, 2)
    metrics = ['Precision', 'Recall', 'F1', 'Accuracy']
    values = [avg_precision, avg_recall, avg_f1, avg_accuracy]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    bars = plt.bar(metrics, values, color=colors)
    plt.title("Average Evaluation Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)  # Metrics are between 0 and 1
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')
    
    # 3. Create scatter plot of precision vs recall
    plt.subplot(2, 2, 3)
    plt.scatter(metrics_df['precision'], metrics_df['recall'], 
                alpha=0.6, c=metrics_df['f1'], cmap='viridis')
    plt.colorbar(label='F1 Score')
    plt.title("Precision vs Recall")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.grid(True, alpha=0.3)
    
    # 4. Create box plot of the metrics
    plt.subplot(2, 2, 4)
    sns.boxplot(data=metrics_df[['precision', 'recall', 'f1', 'accuracy']])
    plt.title("Box Plot of Evaluation Metrics")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the combined plot
    combined_plot_path = os.path.join(plots_dir, "evaluation_metrics_combined.png")
    plt.savefig(combined_plot_path)
    print(f"Combined plot saved to {combined_plot_path}")
    
    # Create individual plots for each metric over query numbers
    plt.figure(figsize=(12, 6))
    metrics_df.sort_values('query_num', inplace=True)  # Sort by query number for better visualization
    
    plt.plot(metrics_df['query_num'], metrics_df['precision'], 'r-', label='Precision')
    plt.plot(metrics_df['query_num'], metrics_df['recall'], 'b-', label='Recall')
    plt.plot(metrics_df['query_num'], metrics_df['f1'], 'g-', label='F1')
    
    plt.title("Precision, Recall and F1 Score by Query")
    plt.xlabel("Query Number")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the query-wise plot
    query_plot_path = os.path.join(plots_dir, "metrics_by_query.png")
    plt.savefig(query_plot_path)
    print(f"Query-wise plot saved to {query_plot_path}")
    
    # Create a heatmap for all queries and their metrics
    if len(metrics_df) <= 30:  # Only create heatmap if reasonable number of queries
        plt.figure(figsize=(14, 10))
        query_metrics = metrics_df.set_index('query_num')[['precision', 'recall', 'f1', 'accuracy']]
        sns.heatmap(query_metrics, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)
        plt.title("Metrics Heatmap by Query Number")
        
        # Save the heatmap
        heatmap_path = os.path.join(plots_dir, "metrics_heatmap.png")
        plt.savefig(heatmap_path)
        print(f"Heatmap saved to {heatmap_path}")
    
    print("All evaluation plots generated successfully!")

def evaluate_all_test_queries(query_dir, model_path, chunks_json_path, split_info_path, 
                             output_dir="test_evaluation", threshold=0.5, top_n_tables=None,
                             specific_queries=None):
    """
    Evaluate the model on all test queries.
    
    Args:
        query_dir: Directory containing query CSV files
        model_path: Path to trained model
        chunks_json_path: Path to JSON file with chunks
        split_info_path: Path to query split info file
        output_dir: Directory to save evaluation results
        threshold: Threshold for relevance
        top_n_tables: Number of top tables to process (None = all)
        specific_queries: List of specific query numbers to evaluate (None = all test queries)
    """
    # Load test query numbers
    test_queries = load_test_queries(split_info_path)
    if not test_queries:
        print("Error: No test queries found. Exiting.")
        return
    
    # Filter test queries if specific queries are provided
    if specific_queries:
        test_queries = [q for q in test_queries if q in specific_queries]
        print(f"Filtered to {len(test_queries)} specific test queries: {test_queries}")
    
    # Ensure output directory exists
    ensure_directory(output_dir)
    ensure_directory(os.path.join(output_dir, "relevant_chunks"))
    
    # Get query files
    query_files = [os.path.join(query_dir, f"query{num}_TopTables.csv") for num in test_queries]
    query_files = [f for f in query_files if os.path.exists(f)]
    
    print(f"Found {len(query_files)} test query files")
    
    # Evaluate each test query
    all_results = []
    target_metrics = []
    
    for query_file in tqdm(query_files, desc="Evaluating test queries"):
        print(f"\nProcessing {os.path.basename(query_file)}...")
        result = evaluate_on_test_query(
            query_file, model_path, chunks_json_path, test_queries, threshold, top_n_tables
        )
        
        if result:
            all_results.append(result)
            
            # Save relevant chunks for this query to a separate JSON file
            query_num = result['query_num']
            relevant_chunks_file = os.path.join(output_dir, "relevant_chunks", f"query{query_num}_relevant_chunks.json")
            with open(relevant_chunks_file, 'w') as f:
                json.dump(result['relevant_chunks'], f, indent=2)
            print(f"Saved relevant chunks to {relevant_chunks_file}")
            
            # Add target table metrics if available
            if result['target_table_metrics']:
                target_metrics.append({
                    'query_num': result['query_num'],
                    'query': result['query'],
                    **result['target_table_metrics']
                })
    
    # Save results
    if all_results:
        # Save details for each query
        details_path = os.path.join(output_dir, "test_queries_details.json")
        with open(details_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Detailed results saved to {details_path}")
        
        # Save summary of overall reduction
        summary_df = pd.DataFrame([{
            'query_num': r['query_num'],
            'query': r['query'],
            'target_table': r['target_table'],
            'target_found': r['target_table_found'],
            'overall_reduction': r['overall_reduction'],
            'target_table_reduction': next((t['reduction_percentage'] 
                                           for t in r['table_results'] 
                                           if t['is_target']), None)
        } for r in all_results])
        
        summary_path = os.path.join(output_dir, "test_queries_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")
        
        # Save target table metrics
        if target_metrics:
            metrics_df = pd.DataFrame(target_metrics)
            metrics_path = os.path.join(output_dir, "target_table_metrics.csv")
            metrics_df.to_csv(metrics_path, index=False)
            print(f"Target table metrics saved to {metrics_path}")
            
            # Print average metrics
            print("\nAverage Target Table Metrics:")
            print(f"Precision: {metrics_df['precision'].mean():.4f}")
            print(f"Recall: {metrics_df['recall'].mean():.4f}")
            print(f"F1: {metrics_df['f1'].mean():.4f}")
            print(f"Accuracy: {metrics_df['accuracy'].mean():.4f}")
            print(f"AUC: {metrics_df['auc'].mean():.4f}")
            
            # Generate evaluation plots from the metrics dataframe
            plot_evaluation_metrics(metrics_df, output_dir)
        
        # Print overall metrics
        print("\nOverall Metrics:")
        print(f"Average reduction: {summary_df['overall_reduction'].mean():.2f}%")
        target_reduction = summary_df['target_table_reduction'].dropna()
        if not target_reduction.empty:
            print(f"Average target table reduction: {target_reduction.mean():.2f}%")
        print(f"Target tables found: {summary_df['target_found'].sum()}/{len(summary_df)} ({summary_df['target_found'].mean()*100:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on test queries")
    parser.add_argument("--query_dir", type=str, default="top_150_queries",
                        help="Directory containing query CSV files")
    parser.add_argument("--model", type=str, default="trained_models/final_model.pt",
                        help="Path to trained model")
    parser.add_argument("--chunks", type=str, default="/home/mvyas7/TRIM-QA/chunks.json",
                        help="Path to chunks JSON file")
    parser.add_argument("--split_info", type=str, default="labeled_data/query_split_info.json",
                        help="Path to query split info file")
    parser.add_argument("--output", type=str, default="test_evaluation",
                        help="Directory to save evaluation results")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Relevance threshold")
    parser.add_argument("--top_tables", type=int, default=None,
                        help="Number of top tables to process per query")
    parser.add_argument("--specific_queries", type=int, nargs='+', default=None,
                        help="List of specific query numbers to evaluate")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip generating evaluation plots")
    
    args = parser.parse_args()
    
    evaluate_all_test_queries(
        args.query_dir, args.model, args.chunks, args.split_info,
        args.output, args.threshold, args.top_tables, args.specific_queries
    )