import torch
import json
import argparse
import pandas as pd
import numpy as np
import jsonlines
import os
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Import the necessary components
from pruning import EmbeddingModule
from train_pruning_model import TrainableCombinedModule, StableURSModule

def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_trained_model(model_path, device=None):
    """
    Load a trained pruning model.
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on
    
    Returns:
        tuple: (embedding_module, combined_module, tokenizer)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the saved model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get configuration
    config = checkpoint.get('config', {})
    model_name = config.get('model_name', 'bert-base-uncased')
    hidden_dim = config.get('hidden_dim', 768)
    lambda_urs = config.get('lambda_urs', 0.4)
    lambda_ws = config.get('lambda_ws', 0.6)
    
    # Initialize the tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    
    # Create modules
    embedding_module = EmbeddingModule(base_model).to(device)
    
    # Create a TrainableCombinedModule with just the hidden_dim parameter
    # The lambda_urs and lambda_ws values will be set from the HP dictionary in the class
    combined_module = TrainableCombinedModule(hidden_dim).to(device)
    
    # Update lambda values after instantiation if needed (not required if they're already set in HP)
    combined_module.lambda_urs = lambda_urs
    combined_module.lambda_ws = lambda_ws
    
    # Load state dictionaries
    embedding_module.load_state_dict(checkpoint['embedding_state_dict'])
    
    # Check if the model has precision enhancer components
    # This handles the case where the model was trained with a different architecture
    model_state_dict = checkpoint['combined_state_dict']
    missing_keys = []
    unexpected_keys = []
    
    # Attempt to load state dict with error handling for architecture differences
    try:
        combined_module.load_state_dict(model_state_dict)
    except RuntimeError as e:
        print(f"Warning: Architecture mismatch detected. Attempting flexible loading...")
        error_msg = str(e)
        
        # Extract missing keys from error message
        if "Missing key(s) in state_dict:" in error_msg:
            missing_part = error_msg.split("Missing key(s) in state_dict:")[1].split("\n")[0]
            missing_keys = [k.strip(' "') for k in missing_part.split(',')]
        
        # Load state dict with strict=False to ignore missing keys
        combined_module.load_state_dict(model_state_dict, strict=False)
        print(f"Model loaded with {len(missing_keys)} missing keys. This is expected for older model versions.")
    
    # Set to evaluation mode
    embedding_module.eval()
    combined_module.eval()
    
    return embedding_module, combined_module, tokenizer

def compute_chunk_embeddings_batch(chunks, embedding_model, tokenizer, device, batch_size=32):
    """Process chunks in batches for faster embedding computation"""
    all_embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_texts = []
        
        for chunk in batch_chunks:
            if isinstance(chunk['text'], list):
                chunk_text = " ".join([str(cell) for cell in chunk['text']])
            else:
                chunk_text = str(chunk['text'])
            batch_texts.append(chunk_text)
        
        # Process the entire batch at once
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Process entire batch in one forward pass
            embeddings = embedding_model(**inputs)
            all_embeddings.append(embeddings)
    
    return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.tensor([])

def run_inference(query, chunks, embedding_module, combined_module, tokenizer, device, threshold=0.5):
    """
    Run inference on chunks for a given query.
    
    Args:
        query: The query text
        chunks: List of chunks to evaluate
        embedding_module: The embedding module
        combined_module: The combined relevance module
        tokenizer: Tokenizer for text encoding
        device: Device to run inference on
        threshold: Threshold for relevance (default: 0.5)
    
    Returns:
        tuple: (pruned_chunks, scores)
    """
    # Process query
    query_inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
    
    with torch.no_grad():
        query_embedding = embedding_module(**query_inputs)
    
    # Process chunks in batches
    print(f"Computing embeddings for {len(chunks)} chunks...")
    chunk_embeddings = compute_chunk_embeddings_batch(chunks, embedding_module, tokenizer, device)
    
    # Compute relevance scores
    print("Computing relevance scores...")
    with torch.no_grad():
        scores, _, _, eta_uns, eta_ws = combined_module.forward_train(chunk_embeddings, query_embedding)
        scores = scores.squeeze().cpu().numpy()
    
    # Print score statistics
    print(f"Score stats - Min: {np.min(scores):.4f}, Max: {np.max(scores):.4f}, Avg: {np.mean(scores):.4f}")
    
    # Prune chunks based on threshold
    pruned_indices = np.where(scores >= threshold)[0]
    pruned_chunks = [chunks[i] for i in pruned_indices]
    pruned_scores = scores[pruned_indices]
    
    print(f"Kept {len(pruned_chunks)}/{len(chunks)} chunks ({len(pruned_chunks)/len(chunks)*100:.1f}%)")
    
    # Add scores to pruned chunks
    for chunk, score in zip(pruned_chunks, pruned_scores):
        chunk['score'] = float(score)
    
    return pruned_chunks, scores

def process_query_file(query_file, model_path, chunks_json_path, output_dir="inference_results", 
                       threshold=0.5, top_n_tables=None):
    """
    Process a query file with the trained model.
    
    Args:
        query_file: Path to CSV file with query information
        model_path: Path to trained model
        chunks_json_path: Path to JSON file with chunks
        output_dir: Directory to save results
        threshold: Threshold for relevance
        top_n_tables: Number of top tables to process (None = all)
    """
    # Ensure output directory exists
    ensure_directory(output_dir)
    
    # Load query file
    try:
        df = pd.read_csv(query_file)
        query = df['query'].iloc[0]
        target_table = df['target table'].iloc[0]
        target_answer = df['target answer'].iloc[0]
    except Exception as e:
        print(f"Error reading query file {query_file}: {e}")
        return
    
    print(f"Query: {query}")
    print(f"Target table: {target_table}")
    print(f"Target answer: {target_answer}")
    
    # Select top tables if specified
    if top_n_tables:
        df = df.head(top_n_tables)
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    embedding_module, combined_module, tokenizer = load_trained_model(model_path, device)
    
    # Prepare results directory
    query_id = os.path.basename(query_file).split('_')[0]
    query_dir = os.path.join(output_dir, query_id)
    ensure_directory(query_dir)
    
    # Load chunks
    print(f"Loading chunks from {chunks_json_path}...")
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
    for _, row in df.iterrows():
        table_id = row['top tables']
        print(f"\nProcessing table: {table_id}")
        
        # Get chunks for this table
        chunks = table_to_chunks.get(table_id, [])
        if not chunks:
            print(f"No chunks found for table {table_id}")
            continue
        
        print(f"Found {len(chunks)} chunks for table {table_id}")
        
        # Run inference
        pruned_chunks, scores = run_inference(
            query, chunks, embedding_module, combined_module, tokenizer, device, threshold
        )
        
        # Save results
        is_target = (table_id == target_table)
        result = {
            'table_id': table_id,
            'original_chunks': len(chunks),
            'pruned_chunks': len(pruned_chunks),
            'reduction': (1 - len(pruned_chunks)/len(chunks)) * 100,
            'is_target': is_target,
            'target_table': target_table,
            'query': query
        }
        results.append(result)
        
        # Save pruned chunks for this table
        output_path = os.path.join(query_dir, f"{table_id}_pruned.json")
        with open(output_path, 'w') as f:
            json.dump(pruned_chunks, f, indent=2)
        print(f"Pruned chunks saved to {output_path}")
    
    # Save summary
    summary_path = os.path.join(query_dir, "summary.csv")
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")
    
    return results

def batch_inference(query_dir, model_path, chunks_json_path, output_dir="inference_results",
                    threshold=0.5, top_n_tables=None, limit=None):
    """
    Process multiple query files in batch.
    
    Args:
        query_dir: Directory containing query CSV files
        model_path: Path to trained model
        chunks_json_path: Path to JSON file with chunks
        output_dir: Directory to save results
        threshold: Threshold for relevance
        top_n_tables: Number of top tables to process (None = all)
        limit: Maximum number of query files to process (None = all)
    """
    # Get query files
    query_files = [f for f in os.listdir(query_dir) if f.endswith('_TopTables.csv')]
    query_files.sort()  # Process in order
    
    if limit:
        query_files = query_files[:limit]
    
    print(f"Found {len(query_files)} query files")
    
    # Process each query file
    all_results = []
    for query_file in query_files:
        query_path = os.path.join(query_dir, query_file)
        print(f"\nProcessing {query_file}...")
        results = process_query_file(
            query_path, model_path, chunks_json_path, output_dir, threshold, top_n_tables
        )
        if results:
            all_results.extend(results)
    
    # Save overall summary
    if all_results:
        summary_path = os.path.join(output_dir, "overall_summary.csv")
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(summary_path, index=False)
        print(f"Overall summary saved to {summary_path}")
        
        # Calculate average metrics
        avg_reduction = summary_df['reduction'].mean()
        target_table_metrics = summary_df[summary_df['is_target']]['reduction'].mean()
        
        print("\nOverall Results Summary:")
        print(f"Average reduction: {avg_reduction:.2f}%")
        print(f"Average reduction for target tables: {target_table_metrics:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with trained pruning model")
    parser.add_argument("--query_dir", type=str, default="top_150_queries",
                        help="Directory containing query CSV files")
    parser.add_argument("--model", type=str, default="trained_models/final_model.pt",
                        help="Path to trained model")
    parser.add_argument("--chunks", type=str, default="chunks.json",
                        help="Path to chunks JSON file")
    parser.add_argument("--output", type=str, default="inference_results",
                        help="Directory to save results")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Relevance threshold")
    parser.add_argument("--top_tables", type=int, default=None,
                        help="Number of top tables to process per query")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of query files to process")
    parser.add_argument("--single_query", type=str, default=None,
                        help="Process a single query file instead of batch")
    
    args = parser.parse_args()
    
    if args.single_query:
        if os.path.exists(args.single_query):
            process_query_file(
                args.single_query, args.model, args.chunks, 
                args.output, args.threshold, args.top_tables
            )
        else:
            print(f"Error: Query file {args.single_query} does not exist.")
    else:
        batch_inference(
            args.query_dir, args.model, args.chunks,
            args.output, args.threshold, args.top_tables, args.limit
        )