import json
import pandas as pd
import argparse
import os
import jsonlines
import time
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

# Set start method for multiprocessing to avoid CUDA reinitialization errors
# This must be done at the module level before any Pool is created
if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn' instead of 'fork'
    mp.set_start_method('spawn', force=True)
    print("Set multiprocessing start method to 'spawn' for CUDA compatibility")

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration settings
USE_BATCHED_PROCESSING = True        # Process chunks in batches instead of one by one
ENABLE_CACHING = True                # Cache embeddings to avoid recomputation
GENERATE_FULL_REPORTS = True         # Generate detailed HTML reports
USE_EARLY_STOPPING = True            # Try to detect high-relevance chunks early
USE_MODEL_QUANTIZATION = False       # Use int8 quantization (faster but less precise)
USE_MIXED_PRECISION = True           # Use mixed precision (FP16) for faster computation
USE_DISTRIBUTED_TRAINING = False     # Use distributed training for multi-GPU setups
BATCH_SIZE = 128                     # Number of chunks to process at once (increased from 64)
MAX_CHUNKS_PER_TABLE = 5000          # Limit number of chunks per table (set to -1 for no limit)
# Initialize NUM_PROCESSES here at the global scope
NUM_PROCESSES = max(1, cpu_count() - 1)  # Number of processes for parallel processing

# Global cache for chunk embeddings
chunk_embedding_cache = {}

def setup_gpu_config():
    """Configure GPU settings for optimal performance."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        # Set memory optimizations for CUDA
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Enable TF32 for better performance on Ampere GPUs (A100, RTX 30xx, etc)
        if torch.cuda.get_device_capability()[0] >= 8:  # A100 and newer
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 precision enabled for faster computation on Ampere+ GPUs")
        
        # Set memory management
        torch.cuda.empty_cache()
        
        # Get detailed GPU information
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} GPU(s):")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {props.name} - {props.total_memory / 1e9:.2f} GB RAM")
            print(f"      CUDA Capability: {props.major}.{props.minor}")
            
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Active GPU: {torch.cuda.get_device_name()}")
        print(f"Current memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    else:
        print("No GPU available, using CPU")
    
    return device

# Function to setup supercomputer/multi-GPU configuration
def setup_supercomputer_config():
    """Configure settings for supercomputer or multi-GPU environments."""
    # Enable GPU if available
    if torch.cuda.is_available():
        # Set to use all available GPUs
        n_gpus = torch.cuda.device_count()
        device = torch.device("cuda")
        
        # Set GPU memory management
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Enable TF32 for better performance on Ampere GPUs (A100, RTX 30xx, etc)
        if torch.cuda.get_device_capability()[0] >= 8:  # A100 and newer
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 precision enabled for faster computation on Ampere+ GPUs")
        
        # Set optimal chunk size for GPU memory
        chunk_size = 32  # Adjust based on GPU memory
        
        if USE_DISTRIBUTED_TRAINING:
            # Enable distributed training
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
    else:
        device = torch.device("cpu")
        n_gpus = 0
        chunk_size = 16
    
    # Set number of CPU workers for data loading
    num_workers = os.cpu_count()
    
    # Set batch size based on available resources
    batch_size = BATCH_SIZE * max(1, n_gpus)  # Scale with number of GPUs
    
    return {
        'device': device,
        'n_gpus': n_gpus,
        'num_workers': num_workers,
        'batch_size': batch_size,
        'chunk_size': chunk_size
    }

def optimize_model_for_inference(model, device):
    """Optimize model for faster inference."""
    model.eval()  # Set to evaluation mode
    model = model.to(device)
    
    # Enable CUDA optimizations if available
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = True
    
    # Use mixed precision where applicable - NOTE: don't wrap the model in a function
    # Just return the model itself
    return model

# Load chunks from a JSON file
def load_chunks(file_path):
    """Load chunks from a JSON file."""
    chunks = []
    try:
        with open(file_path, 'r') as f:
            # Handle both JSON array and line-by-line JSON objects
            first_char = f.read(1)
            f.seek(0)
            
            if (first_char == '['):  # JSON array
                chunks = json.load(f)
            else:  # Line-by-line JSON objects
                chunks = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading chunks: {e}")
        
    return chunks

def chunks_to_dataframe(chunks, scores=None, threshold=0.5):
    """Convert chunks to a pandas DataFrame for better visualization."""
    data = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk['text']
        chunk_type = chunk['metadata'].get('chunk_type', 'unknown')
        chunk_id = chunk['metadata'].get('chunk_id', f'chunk_{i}')
        
        # Extract additional metadata if available
        col_id = chunk['metadata'].get('col_id', '')
        row_id = chunk['metadata'].get('row_id', '')
        table_name = chunk['metadata'].get('table_name', '')
        
        # Add score if provided
        score = scores[i] if scores is not None else chunk.get('score', 0.0)
        is_pruned = "Yes" if score < threshold else "No"
        
        data.append({
            'chunk_id': chunk_id,
            'chunk_type': chunk_type,
            'table_name': table_name,
            'col_id': col_id,
            'row_id': row_id,
            'text': chunk_text,
            'score': score,
            'is_pruned': is_pruned
        })
    
    return pd.DataFrame(data)

def extract_table_ids_from_test_jsonl(file_path='test.jsonl'):
    """Extract table IDs from test.jsonl file."""
    table_info = []
    table_ids = []
    
    try:
        with jsonlines.open(file_path, 'r') as reader:
            for item in reader:
                if 'table' in item and 'tableId' in item['table']:
                    table_id = item['table']['tableId']
                    title = item['table'].get('documentTitle', 'Unknown title')
                    
                    # Only add if not already in the list
                    if table_id not in [t['id'] for t in table_info]:
                        table_info.append({
                            'id': table_id,
                            'title': title,
                            'questions': [q['originalText'] for q in item.get('questions', [])]
                        })
                        table_ids.append(table_id)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return table_info, table_ids

def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

class SentenceTransformerPruner:
    def __init__(self, model_name='all-MiniLM-L6-v2', batch_size=BATCH_SIZE, device=None, config=None):
        self.batch_size = batch_size
        
        # Use config if provided, otherwise use device
        if config:
            self.device = config['device']
            self.batch_size = config['batch_size']
            self.n_gpus = config['n_gpus']
        else:
            self.device = device if device is not None else setup_gpu_config()
            self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            
        self.use_mixed_precision = USE_MIXED_PRECISION and self.device.type == 'cuda'
        if self.use_mixed_precision:
            print("Using mixed precision (FP16) for faster computation")
        
        print(f"Initializing SentenceTransformer model: {model_name}")
        try:
            # Initialize model with quantization if enabled
            if USE_MODEL_QUANTIZATION and self.device.type == 'cuda':
                print("Using int8 quantization for faster inference")
                self.model = SentenceTransformer(model_name, device=self.device, 
                                                 use_auth_token=False)
                self.model.half()  # Convert to FP16 for efficient quantization
            else:
                self.model = SentenceTransformer(model_name, device=self.device)
            
            # Enable parallel processing if multiple GPUs are available
            if self.device.type == "cuda" and self.n_gpus > 1:
                print(f"Using {self.n_gpus} GPUs with DataParallel!")
                self.model = torch.nn.DataParallel(self.model)
                
            # Set model to evaluation mode for inference
            self.model.eval()
            
            # Initialize mixed precision scaler if using mixed precision
            if self.use_mixed_precision:
                self.scaler = amp.GradScaler()
                
            # Apply additional optimizations
            self.model = optimize_model_for_inference(self.model, self.device)
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def get_embeddings(self, texts, show_progress=False, use_cache=ENABLE_CACHING):
        """Get embeddings for a list of texts using batched processing."""
        if not texts:
            return np.array([])
        
        # If caching is enabled, first check which texts are already cached
        if use_cache:
            uncached_indices = []
            cached_embeddings = []
            
            for i, text in enumerate(texts):
                # Use hash of text as cache key for efficiency
                cache_key = hash(text)
                if cache_key in chunk_embedding_cache:
                    cached_embeddings.append((i, chunk_embedding_cache[cache_key]))
                else:
                    uncached_indices.append(i)
                    
            if uncached_indices:
                # Only compute embeddings for uncached texts
                uncached_texts = [texts[i] for i in uncached_indices]
                
                # Process uncached texts
                uncached_embeddings = self._compute_embeddings(
                    uncached_texts, 
                    show_progress=show_progress
                )
                
                # Update cache with new embeddings
                for i, text in enumerate(uncached_texts):
                    cache_key = hash(text)
                    chunk_embedding_cache[cache_key] = uncached_embeddings[i]
                
                # Combine cached and newly computed embeddings
                all_embeddings = np.zeros((len(texts), uncached_embeddings.shape[1]))
                
                # First fill in the uncached embeddings
                for i, orig_idx in enumerate(uncached_indices):
                    all_embeddings[orig_idx] = uncached_embeddings[i]
                
                # Then fill in the cached embeddings
                for orig_idx, embedding in cached_embeddings:
                    all_embeddings[orig_idx] = embedding
                    
                return all_embeddings
            else:
                # All embeddings were in cache
                all_embeddings = np.zeros((len(texts), cached_embeddings[0][1].shape[0]))
                for orig_idx, embedding in cached_embeddings:
                    all_embeddings[orig_idx] = embedding
                return all_embeddings
        else:
            # No caching, compute all embeddings
            return self._compute_embeddings(texts, show_progress=show_progress)
    
    def _compute_embeddings(self, texts, show_progress=False):
        """Compute embeddings for texts without using cache."""
        embeddings = []
        iterator = range(0, len(texts), self.batch_size)
        
        # Use tqdm for progress bar if requested
        if show_progress:
            iterator = tqdm(iterator, desc="Computing embeddings", total=len(iterator))
        
        # Process in batches for memory efficiency
        for i in iterator:
            batch_texts = texts[i:i + self.batch_size]
            
            # Skip empty texts
            if not batch_texts:
                continue
            
            # Clear CUDA cache periodically to prevent memory fragmentation
            if i > 0 and i % (10 * self.batch_size) == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
            # Use mixed precision for faster computation if enabled
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    with torch.no_grad():  # Disable gradient calculation for inference
                        batch_embeddings = self.model.encode(
                            batch_texts,
                            convert_to_tensor=True,
                            show_progress_bar=False,
                            device=self.device
                        )
            else:
                with torch.no_grad():  # Disable gradient calculation for inference
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                        device=self.device
                    )
            
            # Move embeddings to CPU if they were on GPU
            if self.device.type == "cuda":
                batch_embeddings = batch_embeddings.cpu()
            
            # Convert to numpy and append to list
            batch_embeddings_np = batch_embeddings.numpy()
            embeddings.append(batch_embeddings_np)
            
            # Force release GPU memory
            del batch_embeddings
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        # Concatenate all batches
        if embeddings:
            result = np.vstack(embeddings)
            return result
        return np.array([])

    def compute_similarity_scores(self, chunk_texts, query, use_cache=ENABLE_CACHING):
        """Compute similarity scores between chunks and query."""
        # If no chunks, return empty array
        if not chunk_texts:
            return np.array([])
            
        # Get query embedding
        with torch.no_grad():
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    query_embedding = self.model.encode(
                        [query],
                        convert_to_tensor=True,
                        show_progress_bar=False,
                        device=self.device
                    )
            else:
                query_embedding = self.model.encode(
                    [query],
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=self.device
                )
                
            if self.device.type == "cuda":
                query_embedding = query_embedding.cpu()
            query_embedding = query_embedding.numpy()

        # Get chunk embeddings in batches
        chunk_embeddings = self.get_embeddings(chunk_texts, show_progress=True, use_cache=use_cache)
        
        # Compute cosine similarity using vectorized operations
        scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
        
        # If early stopping is enabled, we can sort and return only top chunks
        # This is mostly useful when processing very large tables
        if USE_EARLY_STOPPING and len(scores) > 100:
            # Only compute full cosine similarity if chunk shows promise
            promising_threshold = np.percentile(scores, 90)  # Top 10%
            high_potential_mask = scores >= promising_threshold
            
            if np.sum(high_potential_mask) > 0:
                print(f"Found {np.sum(high_potential_mask)} promising chunks (score >= {promising_threshold:.4f})")
        
        return scores
        
    def compute_answer_potential_scores(self, chunk_texts, query, expected_answer=None):
        """
        Compute a score that reflects how likely a chunk contains answer information.
        This creates a more generalized scoring that rates chunks higher if they potentially 
        contain answer-related information.
        
        Args:
            chunk_texts: List of chunk text strings
            query: Original user query
            expected_answer: Expected answer if available (for training/validation)
            
        Returns:
            Array of answer potential scores
        """
        if not chunk_texts:
            return np.array([])
            
        # Create synthetic "answer-seeking" queries to better judge answer potential
        answer_queries = [
            f"Find information about {query}",
            f"Which part answers {query}",
            f"Content that explains {query}"
        ]
        
        # If expected answer is provided, use it to create more targeted synthetic queries
        if expected_answer and isinstance(expected_answer, (list, tuple)) and expected_answer[0] != "Unknown":
            answer_text = str(expected_answer[0]) if isinstance(expected_answer, (list, tuple)) else str(expected_answer)
            # Create more specific queries based on expected answers
            answer_queries.extend([
                f"Content containing {answer_text}",
                f"Information similar to {answer_text}"
            ])
            
        # Compute similarity for each synthetic query and average them
        all_scores = []
        for answer_query in answer_queries:
            scores = self.compute_similarity_scores(chunk_texts, answer_query)
            all_scores.append(scores)
            
        # Average the scores from all synthetic queries using vectorized operations
        avg_scores = np.mean(all_scores, axis=0)
        return avg_scores

    def remove_redundant_chunks(self, chunks, scores, similarity_threshold=0.8):
        """Remove redundant chunks while preserving the most relevant ones."""
        if not chunks:
            return []
            
        # Sort chunks by score in descending order
        sorted_indices = np.argsort(-np.array(scores))
        
        selected_chunks = []
        selected_indices = []
        
        # Get embeddings for all chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        chunk_embeddings = self.get_embeddings(chunk_texts)
        
        # Compute pairwise similarity matrix
        sim_matrix = cosine_similarity(chunk_embeddings)
        
        # Track which chunks are covered/redundant
        covered = np.zeros(len(chunks), dtype=bool)
        
        # Process chunks in order of relevance
        for idx in sorted_indices:
            if not covered[idx]:
                selected_chunks.append((chunks[idx], scores[idx]))
                selected_indices.append(idx)
                
                # Mark similar chunks as covered using vectorized operations
                similar_indices = sim_matrix[idx] > similarity_threshold
                covered = covered | similar_indices
        
        print(f"Removed {len(chunks) - len(selected_chunks)} redundant chunks")
        return selected_chunks
    
    def prune_chunks(self, chunks, query, relevance_threshold=0.5, 
                     similarity_threshold=0.8, top_k=None, 
                     expected_answer=None, enhance_answer_potential=True):
        """
        Prune chunks based on query similarity and redundancy.
        
        Args:
            chunks: List of chunk dictionaries
            query: Question query for relevance scoring
            relevance_threshold: Minimum relevance score to keep a chunk
            similarity_threshold: Threshold for redundancy removal
            top_k: Maximum number of chunks to return
            expected_answer: Expected answer if available (for training/validation)
            enhance_answer_potential: Whether to enhance scores based on answer potential
            
        Returns:
            List of (chunk, score) tuples
        """
        if not chunks:
            return []
            
        # Filter chunks to keep row and column chunks
        filtered_chunks = [chunk for chunk in chunks 
                          if chunk['metadata'].get('chunk_type') in ['row', 'col']]
        
        if not filtered_chunks:
            print("No row or column chunks found in the input")
            return []
            
        print(f"Filtered {len(chunks)} total chunks to {len(filtered_chunks)} row/column chunks")
        
        # Extract text from chunks
        chunk_texts = [str(chunk['text']) for chunk in filtered_chunks]
        
        # Compute similarity scores
        print("Computing similarity scores with query...")
        base_scores = self.compute_similarity_scores(chunk_texts, query)
        
        # If enhance_answer_potential is True, compute additional scores based on answer potential
        if enhance_answer_potential:
            print("Computing answer potential scores...")
            answer_scores = self.compute_answer_potential_scores(chunk_texts, query, expected_answer)
            
            # Combine scores - give more weight to base scores (70%) and less to answer potential (30%)
            scores = base_scores * 0.7 + answer_scores * 0.3
            print("Using enhanced scoring that combines query relevance and answer potential")
        else:
            scores = base_scores
        
        # Print score statistics
        print(f"Score stats - Min: {min(scores):.4f}, Max: {max(scores):.4f}, Avg: {sum(scores)/len(scores):.4f}")
        
        # Adaptive threshold if no chunks meet the relevance threshold
        original_threshold = relevance_threshold
        if np.max(scores) < relevance_threshold:
            # Dynamically adjust threshold to include at least some chunks
            new_threshold = max(np.percentile(scores, 90), 0.3)  # Take top 10% or min 0.3
            print(f"No chunks met original threshold {relevance_threshold}. Adjusting to {new_threshold:.4f}")
            relevance_threshold = new_threshold
        
        # Filter by relevance threshold using vectorized operations
        mask = scores >= relevance_threshold
        relevant_indices = np.where(mask)[0]
        relevant_chunks = [filtered_chunks[i] for i in relevant_indices]
        relevant_scores = scores[relevant_indices]
        
        print(f"Kept {len(relevant_chunks)}/{len(filtered_chunks)} chunks after relevance filtering")
        
        # Remove redundant chunks if requested
        final_chunks = []
        if similarity_threshold < 1.0 and relevant_chunks:
            print(f"Removing redundant chunks with similarity threshold {similarity_threshold}...")
            selected_chunks = self.remove_redundant_chunks(relevant_chunks, relevant_scores, similarity_threshold)
            print(f"Kept {len(selected_chunks)}/{len(relevant_chunks)} chunks after redundancy removal")
            final_chunks = selected_chunks
        else:
            # Skip redundancy removal
            final_chunks = [(chunk, score) for chunk, score in zip(relevant_chunks, relevant_scores)]
        
        # Add scores to chunks directly
        for chunk, score in final_chunks:
            chunk['score'] = float(score)
        
        # Limit to top_k if specified
        if top_k is not None and final_chunks and len(final_chunks) > top_k:
            # Sort by score in descending order first
            sorted_chunks = sorted(final_chunks, key=lambda x: x[1], reverse=True)
            final_chunks = sorted_chunks[:top_k]
            print(f"Keeping top {len(final_chunks)} chunks")
        
        return final_chunks

# Use distributed data parallel for multi-GPU processing
def init_distributed():
    """Initialize distributed processing for multi-GPU setups."""
    if not torch.cuda.is_available() or not USE_DISTRIBUTED_TRAINING:
        return False
        
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    try:
        dist.init_process_group(backend='nccl', init_method='env://')
        print(f"Initialized process group: rank={rank}, world_size={world_size}")
        return True
    except Exception as e:
        print(f"Failed to initialize distributed process group: {e}")
        return False

# Utility function for parallel processing
def process_query_parallel(args):
    """
    Process a single query against its table chunks. For use with multiprocessing.
    
    Parameters:
    - args: tuple containing (query_item, chunks, pruner, relevance_threshold, similarity_threshold)
    
    Returns:
    - Dictionary with processing results
    """
    query_item, chunks, model_name, relevance_threshold, similarity_threshold = args
    
    # Create a new pruner instance for this process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Apply same supercomputer config as main process
    config = setup_supercomputer_config()
    
    pruner = SentenceTransformerPruner(model_name=model_name, device=device, config=config)
    
    return process_query(query_item, chunks, pruner, relevance_threshold, similarity_threshold)

def process_query(query_item, chunks, pruner, relevance_threshold=0.5, similarity_threshold=0.8):
    """
    Process a single query against its table chunks.
    
    Parameters:
    - query_item: dict containing 'question', 'table_id', and target information
    - chunks: list of chunks for the associated table
    - pruner: SentenceTransformerPruner instance
    - relevance_threshold: minimum relevance score to keep a chunk
    - similarity_threshold: threshold for redundancy removal
    
    Returns:
    - Dictionary with processing results
    """
    question = query_item['question']
    table_id = query_item['table_id']
    # Handle both 'answer' and 'target_answer' fields
    expected_answer = query_item.get('answer', query_item.get('target_answer', ['Unknown']))
    
    print(f"Processing question: {question}")
    print(f"Expected answer(s): {expected_answer}")
    
    # Filter chunks to keep only row and column chunks
    filtered_chunks = [chunk for chunk in chunks 
                      if chunk['metadata'].get('chunk_type') in ['row', 'col']]
    
    if not filtered_chunks:
        print(f"No row or column chunks found for table {table_id}")
        return None
        
    print(f"Filtered {len(chunks)} total chunks to {len(filtered_chunks)} row/column chunks")
    
    # Try early stopping if enabled
    if USE_EARLY_STOPPING and len(filtered_chunks) > 100:
        # Take a sample to test if we can stop early
        sample_size = min(50, len(filtered_chunks))
        sample_chunks = filtered_chunks[:sample_size]
        
        print(f"Checking for early stopping with {sample_size} sample chunks...")
        # Get pruned chunks with a higher threshold for early stopping
        early_sample_pruned = pruner.prune_chunks(
            sample_chunks, 
            question,
            relevance_threshold=0.8,  # Higher threshold for early stopping
            similarity_threshold=similarity_threshold,
            expected_answer=expected_answer
        )
        
        # If we found several high-quality chunks, we can stop early
        if len(early_sample_pruned) >= 5:
            print(f"Early stopping: Found {len(early_sample_pruned)} high-quality chunks")
            
            # Process just these high-quality chunks
            return {
                'question': question,
                'table_id': table_id,
                'target_table': query_item.get('target_table', 'Unknown'),
                'target_answer': expected_answer,
                'original_chunks': len(filtered_chunks),
                'pruned_chunks': [chunk for chunk, _ in early_sample_pruned],
                'pruned_chunks_count': len(early_sample_pruned),
                'reduction_percentage': (1 - len(early_sample_pruned)/len(filtered_chunks)) * 100,
                'processing_time': 0,  # We didn't time this specifically
                'early_stopped': True
            }
    
    start_time = time.time()
    
    # Prune chunks using sentence transformer
    pruned_chunks_with_scores = pruner.prune_chunks(
        filtered_chunks,
        question,
        relevance_threshold=relevance_threshold,
        similarity_threshold=similarity_threshold,
        expected_answer=expected_answer
    )
    
    if not pruned_chunks_with_scores:
        print(f"No chunks remained after pruning for table {table_id}")
        return {
            'question': question,
            'table_id': table_id,
            'target_table': query_item.get('target_table', 'Unknown'),
            'target_answer': expected_answer,
            'original_chunks': len(filtered_chunks),
            'pruned_chunks': [],
            'pruned_chunks_count': 0,
            'reduction_percentage': 100.0,
            'processing_time': time.time() - start_time
        }
    
    # Separate chunks and scores
    pruned_chunks = [chunk for chunk, _ in pruned_chunks_with_scores]
    scores = [score for _, score in pruned_chunks_with_scores]
    
    # Add scores to chunks
    for chunk, score in zip(pruned_chunks, scores):
        chunk['score'] = float(score)
    
    # Calculate statistics
    processing_time = time.time() - start_time
    reduction_pct = (1 - len(pruned_chunks)/len(filtered_chunks)) * 100
    
    print(f"Pruning completed in {processing_time:.2f} seconds")
    print(f"Kept {len(pruned_chunks)}/{len(filtered_chunks)} chunks ({reduction_pct:.1f}% reduction)")
    
    # Return results for summary
    return {
        'question': question,
        'table_id': table_id,
        'target_table': query_item.get('target_table', 'Unknown'),
        'target_answer': expected_answer,
        'original_chunks': len(filtered_chunks),
        'pruned_chunks': pruned_chunks,
        'pruned_chunks_count': len(pruned_chunks),
        'reduction_percentage': reduction_pct,
        'processing_time': processing_time,
        'scores': scores
    }

def merge_pruned_chunks(pruned_chunks):
    """
    Merge row and column chunks that were not pruned into a dataframe format.
    Returns a pandas DataFrame with merged chunks.
    """
    if not pruned_chunks:
        return pd.DataFrame()
        
    # Initialize dictionaries to store row and column chunks
    row_chunks = {}
    col_chunks = {}
    
    # Separate row and column chunks
    for chunk in pruned_chunks:
        chunk_type = chunk['metadata'].get('chunk_type', '')
        
        if chunk_type == 'row':
            row_id = chunk['metadata'].get('row_id', '')
            if row_id not in row_chunks:
                row_chunks[row_id] = []
            row_chunks[row_id].append(chunk)
        elif chunk_type == 'col':
            col_id = chunk['metadata'].get('col_id', '')
            if col_id not in col_chunks:
                col_chunks[col_id] = []
            col_chunks[col_id].append(chunk)

    # Create lists to store merged data
    merged_data = []
    
    # Merge row chunks
    for row_id, chunks in row_chunks.items():
        merged_text = " ".join([str(c['text']) for c in chunks])
        avg_score = sum([c.get('score', 0) for c in chunks]) / len(chunks)
        table_name = chunks[0]['metadata'].get('table_name', '')
        
        merged_data.append({
            'chunk_id': f'merged_row_{row_id}',
            'chunk_type': 'merged_row',
            'table_name': table_name,
            'row_id': row_id,
            'col_id': '',  # Empty for row chunks
            'text': merged_text,
            'score': avg_score,
            'original_chunks': len(chunks)
        })
    
    # Merge column chunks
    for col_id, chunks in col_chunks.items():
        merged_text = " ".join([str(c['text']) for c in chunks])
        avg_score = sum([c.get('score', 0) for c in chunks]) / len(chunks)
        table_name = chunks[0]['metadata'].get('table_name', '')
        
        merged_data.append({
            'chunk_id': f'merged_col_{col_id}',
            'chunk_type': 'merged_col',
            'table_name': table_name,
            'row_id': '',  # Empty for column chunks
            'col_id': col_id,
            'text': merged_text,
            'score': avg_score,
            'original_chunks': len(chunks)
        })
    
    # Create DataFrame from merged data
    df = pd.DataFrame(merged_data)
    
    # Sort by score in descending order
    if not df.empty:
        df = df.sort_values('score', ascending=False)
    
    return df

def save_pruning_results(query, table_id, original_chunks, pruned_chunks, relevance_threshold, 
                         target_table=None, target_answer=None, pruning_dir="pruning_results_st"):
    """Save pruning results with optimized report generation."""
    # Create directory structure
    base_dir = pruning_dir
    for subdir in ['original', 'pruned', 'merged', 'reports']:
        ensure_directory(os.path.join(base_dir, subdir))
    
    # Create safe filename components
    safe_table_id = table_id.replace('/', '_').replace(' ', '_')
    safe_query = "".join(c if c.isalnum() else "_" for c in query[:30])
    
    # Save original chunks
    original_path = os.path.join(base_dir, "original", f"original_{safe_table_id}_{safe_query}.json")
    with open(original_path, 'w') as f:
        json.dump(original_chunks, f, indent=2)
    
    # Save pruned chunks
    pruned_path = os.path.join(base_dir, "pruned", f"pruned_{safe_table_id}_{safe_query}.json")
    with open(pruned_path, 'w') as f:
        json.dump(pruned_chunks, f, indent=2)
    
    # Create merged version if there are pruned chunks
    merged_df = None
    if pruned_chunks:
        merged_df = merge_pruned_chunks(pruned_chunks)
    
    merged_path = os.path.join(base_dir, "merged", f"merged_{safe_table_id}_{safe_query}.json")
    with open(merged_path, 'w') as f:
        if merged_df is not None and not merged_df.empty:
            # Convert DataFrame to a list of dictionaries for JSON serialization
            merged_records = merged_df.to_dict('records')
            json.dump(merged_records, f, indent=2)
        else:
            # Save an empty list if no merged chunks
            json.dump([], f, indent=2)
    
    if GENERATE_FULL_REPORTS:
        # Generate HTML report
        report_path = os.path.join(base_dir, "reports", f"report_{safe_table_id}_{safe_query}.html")
        
        # Calculate statistics
        total_chunks = len(original_chunks)
        pruned_count = len(pruned_chunks)
        reduction_pct = ((total_chunks - pruned_count) / total_chunks * 100) if total_chunks > 0 else 0
        
        # Handle the case where there are no pruned chunks
        if pruned_chunks:
            # Get scores for visualization - only for chunks that were kept
            score_list = [chunk.get('score', 0.0) for chunk in pruned_chunks]
            # If original_chunks and pruned_chunks have different lengths, we need to create a list of scores
            # that matches the length of original_chunks
            if len(original_chunks) > len(pruned_chunks):
                # Map the pruned chunks' scores to their corresponding original chunks
                # For chunks that were pruned, use a score of 0
                full_scores = []
                pruned_chunk_ids = [chunk['metadata'].get('chunk_id', '') for chunk in pruned_chunks]
                
                for orig_chunk in original_chunks:
                    chunk_id = orig_chunk['metadata'].get('chunk_id', '')
                    try:
                        # Find the corresponding pruned chunk
                        idx = pruned_chunk_ids.index(chunk_id)
                        full_scores.append(pruned_chunks[idx].get('score', 0.0))
                    except ValueError:
                        # This chunk was pruned, so its score is 0
                        full_scores.append(0.0)
            else:
                full_scores = score_list
        else:
            # If no chunks were kept, use zeros for all scores
            full_scores = [0.0] * len(original_chunks)
        
        # Create dataframe with available information
        df = chunks_to_dataframe(original_chunks, scores=full_scores, threshold=relevance_threshold)
        
        # Generate basic HTML report
        html_content = f"""
        <html>
        <head>
            <title>Sentence Transformer Pruning Report - {safe_query}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .stats {{ margin: 20px 0; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .pruned {{ background-color: #f8d7da; }}
                .kept {{ background-color: #d4edda; }}
            </style>
        </head>
        <body>
            <h1>Sentence Transformer Pruning Report</h1>
            <div class="stats">
                <h2>Query Information</h2>
                <p><strong>Query:</strong> {query}</p>
                <p><strong>Table ID:</strong> {table_id}</p>
                <p><strong>Target Table:</strong> {target_table or "Not specified"}</p>
                <p><strong>Target Answer:</strong> {target_answer or "Not specified"}</p>
                <h2>Pruning Statistics</h2>
                <p>Original chunks: {total_chunks}</p>
                <p>Pruned chunks: {pruned_count}</p>
                <p>Reduction: {reduction_pct:.1f}%</p>
                <p>Relevance Threshold: {relevance_threshold}</p>
                {'' if pruned_count > 0 else '<p><strong>Note:</strong> No chunks exceeded the relevance threshold</p>'}
            </div>
            {df.to_html()}
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
    
    return {
        'original_path': os.path.join("original", f"original_{safe_table_id}_{safe_query}.json"),
        'pruned_path': os.path.join("pruned", f"pruned_{safe_table_id}_{safe_query}.json"),
        'merged_path': os.path.join("merged", f"merged_{safe_table_id}_{safe_query}.json"),
        'report_path': os.path.join("reports", f"report_{safe_table_id}_{safe_query}.html") if GENERATE_FULL_REPORTS else None
    }

def process_top_queries(query_range, top_n_tables, pruner, relevance_threshold=0.5, similarity_threshold=0.8):
    """Process queries from Top-150-Queries directory using sentence transformer with parallel processing"""
    try:
        # Create results directory
        base_dir = "pruning_results_st"
        ensure_directory(base_dir)
        
        # Create subdirectories
        for subdir in ['original', 'pruned', 'merged', 'reports']:
            ensure_directory(os.path.join(base_dir, subdir))
        
        # Load queries from the specified range
        queries_dir = "top_150_queries"
        
        # Get the specific query files based on query numbers
        selected_files = []
        for query_num in range(query_range[0], query_range[1] + 1):
            query_file = f"query{query_num}_TopTables.csv"
            if os.path.exists(os.path.join(queries_dir, query_file)):
                selected_files.append(query_file)
            else:
                print(f"Warning: Query file {query_file} not found")
        
        if not selected_files:
            print(f"No query files found for range {query_range[0]} to {query_range[1]}")
            return []
        
        print(f"\nProcessing queries {query_range[0]} to {query_range[1]}")
        print(f"Found {len(selected_files)} query files")
        print(f"Will process top {top_n_tables} tables for each query")
        
        all_results = []
        
        # Get the actual model name from the pruner
        # Extract model name if using a SentenceTransformer
        if hasattr(pruner, 'model'):
            if hasattr(pruner.model, 'modules'):
                # For DataParallel wrapped models
                model_components = pruner.model.module if isinstance(pruner.model, torch.nn.DataParallel) else pruner.model
                model_name = model_components._modules['0'].auto_model.config._name_or_path
            else:
                # Fallback to a default model name
                model_name = 'all-MiniLM-L6-v2'
        else:
            # Default model if we can't determine
            model_name = 'all-MiniLM-L6-v2'
        
        print(f"Using model: {model_name} for parallel processing")
        
        # Process each query file
        for query_file in selected_files:
            query_num = query_file.split('_')[0].replace('query', '')
            print(f"\n{'='*50}")
            print(f"Processing Query {query_num}...")
            
            # Read the query CSV file
            query_df = pd.read_csv(os.path.join(queries_dir, query_file))
            
            # Get the query text (same for all rows)
            query_text = query_df['query'].iloc[0]
            target_table = query_df['target table'].iloc[0]
            target_answer = query_df['target answer'].iloc[0]
            
            print(f"Query: {query_text}")
            print(f"Target table: {target_table}")
            print(f"Target answer: {target_answer}")
            
            # Take exactly top N tables
            top_tables = query_df.head(top_n_tables)
            print(f"Processing top {len(top_tables)} tables for this query")
            
            query_results = []
            
            # Check if we should use parallel processing
            if NUM_PROCESSES > 1 and len(top_tables) > 1:
                print(f"Using {NUM_PROCESSES} processes for parallel table processing")
                
                # Prepare arguments for parallel processing
                parallel_args = []
                
                for idx, row in top_tables.iterrows():
                    table_id = row['top tables']
                    print(f"Loading chunks for table: {table_id}")
                    
                    # Check if this is the target table
                    is_target = (table_id == target_table)
                    
                    # Load chunks for this table
                    chunks = []
                    with jsonlines.open('/home/mvyas7/TRIM-QA/chunks.json', 'r') as reader:
                        for chunk in reader:
                            if ('metadata' in chunk and 
                                'table_name' in chunk['metadata'] and 
                                chunk['metadata']['table_name'] == table_id):
                                chunks.append(chunk)
                    
                    if not chunks:
                        print(f"No chunks found for table {table_id}")
                        continue
                    
                    # Limit chunks if needed
                    if MAX_CHUNKS_PER_TABLE > 0 and len(chunks) > MAX_CHUNKS_PER_TABLE:
                        print(f"Limiting to {MAX_CHUNKS_PER_TABLE} chunks (from {len(chunks)})")
                        chunks = chunks[:MAX_CHUNKS_PER_TABLE]
                    
                    # Create query item
                    query_item = {
                        'question': query_text,
                        'table_id': table_id,
                        'target_table': target_table,
                        'target_answer': target_answer,
                        'is_target': is_target
                    }
                    
                    # Use the actual model name string instead of trying to get it from the model object
                    # Add to parallel arguments
                    parallel_args.append((query_item, chunks, model_name, relevance_threshold, similarity_threshold))
                
                # Process tables in parallel
                with Pool(processes=NUM_PROCESSES) as pool:
                    parallel_results = pool.map(process_query_parallel, parallel_args)
                
                # Filter out None results
                parallel_results = [r for r in parallel_results if r is not None]
                
                # Add is_target flag
                for i, result in enumerate(parallel_results):
                    result['is_target'] = parallel_args[i][0].get('is_target', False)
                
                # Save results for each table
                for result in parallel_results:
                    if result:
                        # Find the original chunks for this specific table
                        table_chunks = []
                        for arg in parallel_args:
                            if arg[0]['table_id'] == result['table_id']:
                                table_chunks = arg[1]
                                break
                        
                        # Save results with the correct chunks for this table
                        paths = save_pruning_results(
                            query=result['question'],
                            table_id=result['table_id'],
                            original_chunks=table_chunks,  # Now using the correct chunks for this table
                            pruned_chunks=result['pruned_chunks'],
                            relevance_threshold=relevance_threshold,
                            target_table=target_table,
                            target_answer=target_answer,
                            pruning_dir=base_dir
                        )
                        
                        # Add paths to result
                        result['paths'] = paths
                        
                        # Add to results
                        query_results.append(result)
                        all_results.append(result)
            else:
                # Process tables sequentially
                for idx, row in top_tables.iterrows():
                    table_id = row['top tables']
                    print(f"\n{'-'*40}")
                    print(f"Processing table {idx + 1}/{len(top_tables)}: {table_id}")
                    
                    # Check if this is the target table
                    is_target = (table_id == target_table)
                    if is_target:
                        print("*** This is the target table ***")
                    
                    # Load chunks for this table
                    chunks = []
                    with jsonlines.open('/home/mvyas7/TRIM-QA/chunks.json', 'r') as reader:
                        for chunk in reader:
                            if ('metadata' in chunk and 
                                'table_name' in chunk['metadata'] and 
                                chunk['metadata']['table_name'] == table_id):
                                chunks.append(chunk)
                    
                    if not chunks:
                        print(f"No chunks found for table {table_id}")
                        continue
                    
                    # Limit chunks if needed
                    if MAX_CHUNKS_PER_TABLE > 0 and len(chunks) > MAX_CHUNKS_PER_TABLE:
                        print(f"Limiting to {MAX_CHUNKS_PER_TABLE} chunks (from {len(chunks)})")
                        chunks = chunks[:MAX_CHUNKS_PER_TABLE]
                    
                    # Create query item
                    query_item = {
                        'question': query_text,
                        'table_id': table_id,
                        'target_table': target_table,
                        'target_answer': target_answer
                    }
                    
                    # Process the query
                    result = process_query(
                        query_item,
                        chunks,
                        pruner,
                        relevance_threshold=relevance_threshold,
                        similarity_threshold=similarity_threshold
                    )
                    
                    if not result:
                        continue
                    
                    # Save results
                    paths = save_pruning_results(
                        query=query_text,
                        table_id=table_id,
                        original_chunks=chunks,
                        pruned_chunks=result['pruned_chunks'],
                        relevance_threshold=relevance_threshold,
                        target_table=target_table,
                        target_answer=target_answer,
                        pruning_dir=base_dir
                    )
                    
                    # Add paths to result
                    result['paths'] = paths
                    result['is_target'] = is_target
                    
                    # Add to results
                    query_results.append(result)
                    all_results.append(result)
            
            # Create query-specific summary
            if query_results:
                query_summary_df = pd.DataFrame([{
                    'table_id': r['table_id'],
                    'is_target': r.get('is_target', False),
                    'original_chunks': r['original_chunks'],
                    'pruned_chunks': r['pruned_chunks_count'],
                    'reduction_percentage': r['reduction_percentage'],
                    'processing_time': r.get('processing_time', 0)
                } for r in query_results])
                
                # Sort by is_target (target table first) and then by number of pruned chunks
                query_summary_df = query_summary_df.sort_values(
                    by=['is_target', 'pruned_chunks'], 
                    ascending=[False, False]
                )
                
                # Save query-specific summary
                query_summary_path = os.path.join(base_dir, f"summary_query{query_num}.csv")
                query_summary_df.to_csv(query_summary_path, index=False)
                print(f"\nQuery {query_num} summary saved to {query_summary_path}")
                
                # Clear CUDA cache after each query to prevent memory leaks
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Create overall summary if we have results
        if all_results:
            # Main summary dataframe
            summary_df = pd.DataFrame([{
                'query_num': r.get('question', '').split()[0],
                'question': r.get('question', ''),
                'table_id': r.get('table_id', ''),
                'is_target': r.get('is_target', False),
                'target_table': r.get('target_table', ''),
                'original_chunks': r.get('original_chunks', 0),
                'pruned_chunks': r.get('pruned_chunks_count', 0),
                'reduction_percentage': r.get('reduction_percentage', 0),
                'processing_time': r.get('processing_time', 0)
            } for r in all_results])
            
            # Save overall summary
            summary_path = os.path.join(base_dir, f"summary_queries_{query_range[0]}-{query_range[1]}.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"\nOverall summary saved to {summary_path}")
            
            # Also save a summary of just the target tables
            target_df = summary_df[summary_df['is_target']]
            if not target_df.empty:
                target_path = os.path.join(base_dir, f"summary_target_tables_{query_range[0]}-{query_range[1]}.csv")
                target_df.to_csv(target_path, index=False)
                print(f"Target tables summary saved to {target_path}")
        
        return all_results
    
    except Exception as e:
        print(f"Error processing queries: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    print("\n=== TRIM-QA Sentence Transformer Pruning ===")
    print("=" * 45)
    
    # Setup GPU configuration using supercomputer config
    config = setup_supercomputer_config()
    device = config['device']
    
    # Update batch size from config
    global BATCH_SIZE
    BATCH_SIZE = config['batch_size']
    
    # Print detailed GPU configuration
    if device.type == "cuda":
        print(f"\nGPU Configuration:")
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {config['n_gpus']}")
        print(f"Device Capability: {torch.cuda.get_device_capability()}")
        print(f"Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Current Memory Usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Get query range from user
    while True:
        try:
            print("\nEnter query range (1-150, format: start end):")
            start, end = map(int, input("> ").split())
            if 1 <= start <= end <= 150:
                break
            print("Invalid range. Start must be <= end, and both must be between 1 and 150.")
        except ValueError:
            print("Please enter two numbers separated by space (e.g., '1 5')")
    
    # Get number of top tables to process
    while True:
        try:
            print("\nEnter number of top tables to process for each query:")
            top_n = int(input("> "))
            if top_n > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get relevance threshold
    while True:
        try:
            print("\nEnter relevance threshold (0.0-1.0, default: 0.5):")
            print("This determines how similar a chunk must be to the query to be kept.")
            threshold_input = input("> ").strip()
            if not threshold_input:  # Use default if empty
                relevance_threshold = 0.5
                break
            relevance_threshold = float(threshold_input)
            if 0 <= relevance_threshold <= 1:
                break
            print("Threshold must be between 0 and 1.")
        except ValueError:
            print("Please enter a valid number between 0 and 1.")
    
    # Get similarity threshold for redundancy removal
    while True:
        try:
            print("\nEnter similarity threshold for redundancy removal (0.0-1.0, default: 0.8):")
            print("Higher values mean chunks must be more similar to be considered redundant.")
            threshold_input = input("> ").strip()
            if not threshold_input:  # Use default if empty
                similarity_threshold = 0.8
                break
            similarity_threshold = float(threshold_input)
            if 0 <= similarity_threshold <= 1:
                break
            print("Threshold must be between 0 and 1.")
        except ValueError:
            print("Please enter a valid number between 0 and 1.")
    
    # Choose model
    print("\nSelect sentence transformer model:")
    print("1. all-MiniLM-L6-v2 (default, faster)")
    print("2. all-mpnet-base-v2 (more accurate but slower)")
    print("3. paraphrase-multilingual-MiniLM-L12-v2 (multilingual support)")
    while True:
        try:
            choice = input("\nEnter choice (1-3) or press Enter for default: ").strip()
            if not choice:  # Use default if empty
                model_name = 'all-MiniLM-L6-v2'
                break
            choice = int(choice)
            if choice == 1:
                model_name = 'all-MiniLM-L6-v2'
                break
            elif choice == 2:
                model_name = 'all-mpnet-base-v2'
                break
            elif choice == 3:
                model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
                break
            print("Please enter a number between 1 and 3.")
        except ValueError:
            print("Please enter a valid number between 1 and 3.")
    
    # Configure parallel processing
    global NUM_PROCESSES  # Now this is correct since NUM_PROCESSES is initialized at the module level
    print("\nConfigure parallel processing:")
    print(f"Auto-detected {NUM_PROCESSES} processes (cores-1)")
    while True:
        try:
            processes = input(f"Enter number of processes to use (1-{cpu_count()}, default: {NUM_PROCESSES}): ").strip()
            if not processes:  # Use default
                break
            processes = int(processes)
            if 1 <= processes <= cpu_count():
                NUM_PROCESSES = processes
                break
            print(f"Please enter a number between 1 and {cpu_count()}.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Choose whether to use mixed precision
    global USE_MIXED_PRECISION  # Also add global for this variable
    if torch.cuda.is_available():
        while True:
            try:
                print("\nUse mixed precision (FP16) for faster computation on GPU? (y/n, default: y)")
                choice = input("> ").strip().lower()
                if not choice or choice == 'y' or choice == 'yes':
                    USE_MIXED_PRECISION = True
                    break
                elif choice == 'n' or choice == 'no':
                    USE_MIXED_PRECISION = False
                    break
                print("Please enter 'y' or 'n'.")
            except ValueError:
                print("Please enter 'y' or 'n'.")
    
    # Choose whether to use distributed training
    global USE_DISTRIBUTED_TRAINING
    if torch.cuda.device_count() > 1:
        while True:
            try:
                print(f"\nUse distributed training across {torch.cuda.device_count()} GPUs? (y/n, default: n)")
                choice = input("> ").strip().lower()
                if choice == 'y' or choice == 'yes':
                    USE_DISTRIBUTED_TRAINING = True
                    break
                elif not choice or choice == 'n' or choice == 'no':
                    USE_DISTRIBUTED_TRAINING = False
                    break
                print("Please enter 'y' or 'n'.")
            except ValueError:
                print("Please enter 'y' or 'n'.")
    
    # Show selected configuration
    print("\n=== Configuration Summary ===")
    print(f"Query Range: {start}-{end}")
    print(f"Top Tables per Query: {top_n}")
    print(f"Relevance Threshold: {relevance_threshold}")
    print(f"Similarity Threshold: {similarity_threshold}")
    print(f"Model: {model_name}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Max Chunks Per Table: {MAX_CHUNKS_PER_TABLE}")
    print(f"Parallel Processes: {NUM_PROCESSES}")
    print(f"Mixed Precision: {'Enabled' if USE_MIXED_PRECISION else 'Disabled'}")
    print(f"Distributed Training: {'Enabled' if USE_DISTRIBUTED_TRAINING else 'Disabled'}")
    print(f"Device: {device}")
    
    # Confirm and proceed
    while True:
        confirm = input("\nProceed with pruning? (y/n): ").lower()
        if confirm in ['y', 'yes']:
            break
        elif confirm in ['n', 'no']:
            print("Pruning cancelled.")
            return
        else:
            print("Please enter 'y' or 'n'.")
    
    start_time = time.time()
    
    # Initialize distributed training if enabled
    if USE_DISTRIBUTED_TRAINING:
        init_distributed()
    
    # Create pruner instance with the supercomputer config
    print(f"\nInitializing sentence transformer with model: {model_name}")
    pruner = SentenceTransformerPruner(model_name=model_name, config=config)
    
    # Process queries
    results = process_top_queries(
        query_range=(start, end),
        top_n_tables=top_n,
        pruner=pruner,
        relevance_threshold=relevance_threshold,
        similarity_threshold=similarity_threshold
    )
    
    if results:
        print("\n===== Final Summary =====")
        print(f"Processed {len(results)} table-query pairs")
        avg_reduction = sum(r['reduction_percentage'] for r in results) / len(results)
        avg_pruned_count = sum(r['pruned_chunks_count'] for r in results) / len(results)
        avg_time = sum(r.get('processing_time', 0) for r in results) / len(results)
        
        print(f"Average chunk reduction: {avg_reduction:.1f}%")
        print(f"Average pruned chunks per table: {avg_pruned_count:.1f}")
        print(f"Average processing time per table: {avg_time:.2f} seconds")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\nGPU memory cleared")

if __name__ == "__main__":
    main()