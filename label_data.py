import csv
import json
import jsonlines
import os
import re
import pandas as pd
import string
import random
from sklearn.model_selection import train_test_split
from difflib import SequenceMatcher

def similar(a, b):
    """Calculate string similarity between two strings."""
    return SequenceMatcher(None, a, b).ratio()

def clean_text(text):
    """Clean text by removing punctuation and lowercasing."""
    if isinstance(text, str):
        # Remove punctuation and lowercase
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    return ""

def query_level_train_test_split(query_data, test_size=0.33, random_state=42):
    """
    Split data by query to ensure all chunks from the same query are in the same split.
    
    Args:
        query_data: Dictionary mapping query numbers to their labeled data
        test_size: Proportion of queries to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        tuple: (train_data, test_data) lists
    """
    # Get unique query numbers
    query_numbers = list(query_data.keys())
    
    # Sort the query numbers to ensure reproducibility
    query_numbers.sort()
    
    # Split at query level
    train_queries, test_queries = train_test_split(
        query_numbers, test_size=test_size, random_state=random_state
    )
    
    print(f"Training queries: {len(train_queries)}")
    print(f"Test queries: {len(test_queries)}")
    
    # Create train and test datasets based on query assignment
    train_data = []
    for q in train_queries:
        train_data.extend(query_data[q])
    
    test_data = []
    for q in test_queries:
        test_data.extend(query_data[q])
    
    return train_data, test_data

def label_data(query_csv_dir, chunks_json_path, output_dir="labeled_data", 
               test_size=0.33, fuzzy_match_threshold=0.8, 
               train_queries_list=None, test_queries_list=None):
    """
    Label data based on whether chunks contain the target answer.
    
    Args:
        query_csv_dir: Directory containing CSV files with queries and target answers
        chunks_json_path: Path to JSONL file with all table chunks
        output_dir: Directory to save labeled data
        test_size: Proportion of data to use for testing
        fuzzy_match_threshold: Threshold for fuzzy matching (0-1)
        train_queries_list: Optional list of query numbers for training
        test_queries_list: Optional list of query numbers for testing
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading chunks from {chunks_json_path}...")
    # Read all chunks from chunks.json and organize by table ID
    table_to_chunks = {}
    with jsonlines.open(chunks_json_path, 'r') as reader:
        for chunk in reader:
            meta = chunk.get('metadata', {})
            table_id = meta.get('table_name', None)
            chunk_type = meta.get('chunk_type', None)
            # Keep only "row" chunks, skip table-level or other chunks
            if table_id and chunk_type == 'row':
                table_to_chunks.setdefault(table_id, []).append(chunk)
    
    print(f"Loaded chunks for {len(table_to_chunks)} tables")
    
    # Process each query file in the directory
    query_files = [f for f in os.listdir(query_csv_dir) if f.endswith('_TopTables.csv')]
    print(f"Found {len(query_files)} query files to process")
    
    # Store all labeled data by query number
    query_to_labeled_data = {}
    
    # Track statistics
    stats = {
        'total_queries': 0,
        'total_chunks_processed': 0,
        'total_relevant_chunks': 0,
        'tables_with_no_chunks': 0,
        'tables_with_no_relevant_chunks': 0
    }
    
    # Extract query numbers for filtering
    if train_queries_list or test_queries_list:
        fixed_split = True
        all_query_numbers = set()
        if train_queries_list:
            all_query_numbers.update(train_queries_list)
        if test_queries_list:
            all_query_numbers.update(test_queries_list)
    else:
        fixed_split = False
        all_query_numbers = None
    
    for query_file in query_files:
        # Extract query number
        query_num_match = re.search(r'query(\d+)_', query_file)
        if not query_num_match:
            print(f"Warning: Could not extract query number from {query_file}, skipping.")
            continue
        
        query_num = int(query_num_match.group(1))
        
        # Skip if not in the specified query numbers (when fixed split is provided)
        if fixed_split and query_num not in all_query_numbers:
            print(f"Skipping query {query_num} as it's not in the specified query lists.")
            continue
        
        query_path = os.path.join(query_csv_dir, query_file)
        print(f"Processing {query_file} (Query {query_num})...")
        
        # Read query CSV
        try:
            df = pd.read_csv(query_path)
            stats['total_queries'] += 1
        except Exception as e:
            print(f"Error reading {query_path}: {e}")
            continue
        
        # Get query information
        query = df['query'].iloc[0]
        target_table = df['target table'].iloc[0]
        target_answer = df['target answer'].iloc[0]
        
        print(f"Query: {query}")
        print(f"Target table: {target_table}")
        print(f"Target answer: {target_answer}")
        
        # Clean the target answer for better matching
        clean_target_answer = clean_text(target_answer)
        
        # Get chunks for target table
        candidate_chunks = table_to_chunks.get(target_table, [])
        
        if not candidate_chunks:
            print(f"Warning: No chunks found for table {target_table}")
            stats['tables_with_no_chunks'] += 1
            continue
        
        print(f"Found {len(candidate_chunks)} chunks for table {target_table}")
        
        # Initialize storage for this query's labeled data
        query_to_labeled_data[query_num] = []
        relevant_count = 0
        
        # Label each chunk
        for chunk in candidate_chunks:
            stats['total_chunks_processed'] += 1
            
            chunk_text = chunk['text']
            # If chunk['text'] is a list of cells, join them into a single string
            if isinstance(chunk_text, list):
                chunk_text = " ".join([str(x) for x in chunk_text])
            
            clean_chunk_text = clean_text(chunk_text)
            
            # Determine if the chunk is relevant - try exact match first
            is_relevant = clean_target_answer in clean_chunk_text
            
            # If exact match fails, try fuzzy matching
            if not is_relevant and len(clean_target_answer) > 3:  # Only do fuzzy match for longer answers
                # Try fuzzy matching on substrings
                words = clean_chunk_text.split()
                for i in range(len(words)):
                    for j in range(i+1, min(i+10, len(words)+1)):  # Look at nearby word groups
                        chunk_segment = " ".join(words[i:j])
                        if len(chunk_segment) > 3:  # Only compare substantial segments
                            similarity = similar(clean_target_answer, chunk_segment)
                            if similarity >= fuzzy_match_threshold:
                                is_relevant = True
                                break
                    if is_relevant:
                        break
            
            # Set label (1 for relevant, 0 for not relevant)
            label = 1 if is_relevant else 0
            
            if label == 1:
                relevant_count += 1
                stats['total_relevant_chunks'] += 1
            
            # Add to labeled data for this query
            query_to_labeled_data[query_num].append({
                "query": query,
                "query_num": query_num,
                "table_id": target_table,
                "chunk_text": chunk_text,
                "target_answer": target_answer,
                "label": label
            })
        
        if relevant_count == 0:
            print(f"Warning: No relevant chunks found for table {target_table}")
            stats['tables_with_no_relevant_chunks'] += 1
        else:
            print(f"Found {relevant_count} relevant chunks out of {len(candidate_chunks)}")
    
    if not query_to_labeled_data:
        print("Error: No labeled data was generated!")
        return
    
    print(f"\nLabeling stats:")
    print(f"Total queries processed: {stats['total_queries']}")
    print(f"Total chunks processed: {stats['total_chunks_processed']}")
    print(f"Total relevant chunks found: {stats['total_relevant_chunks']}")
    print(f"Tables with no chunks: {stats['tables_with_no_chunks']}")
    print(f"Tables with no relevant chunks: {stats['tables_with_no_relevant_chunks']}")
    
    # Split data into training and test sets at the query level
    if fixed_split:
        print("\nUsing predefined train/test query lists")
        train_data = []
        test_data = []
        
        for query_num, labeled_examples in query_to_labeled_data.items():
            if train_queries_list and query_num in train_queries_list:
                train_data.extend(labeled_examples)
            elif test_queries_list and query_num in test_queries_list:
                test_data.extend(labeled_examples)
    else:
        print("\nPerforming automatic train/test split at query level")
        train_data, test_data = query_level_train_test_split(query_to_labeled_data, test_size=test_size, random_state=42)
    
    # Save training data
    train_path = os.path.join(output_dir, "train_data.csv")
    pd.DataFrame(train_data).to_csv(train_path, index=False)
    print(f"Training data saved to {train_path} ({len(train_data)} examples)")
    
    # Save test data
    test_path = os.path.join(output_dir, "test_data.csv")
    pd.DataFrame(test_data).to_csv(test_path, index=False)
    print(f"Test data saved to {test_path} ({len(test_data)} examples)")
    
    # Save full dataset
    all_data = train_data + test_data
    full_path = os.path.join(output_dir, "full_dataset.csv")
    pd.DataFrame(all_data).to_csv(full_path, index=False)
    print(f"Full dataset saved to {full_path} ({len(all_data)} examples)")
    
    # Save the query split information for future reference
    train_query_nums = sorted([q for q in query_to_labeled_data.keys() 
                              if any(item in train_data for item in query_to_labeled_data[q])])
    test_query_nums = sorted([q for q in query_to_labeled_data.keys() 
                             if any(item in test_data for item in query_to_labeled_data[q])])
    
    split_info = {
        'train_queries': train_query_nums,
        'test_queries': test_query_nums
    }
    
    split_info_path = os.path.join(output_dir, "query_split_info.json")
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"Query split info saved to {split_info_path}")
    print(f"Training queries: {train_query_nums}")
    print(f"Testing queries: {test_query_nums}")
    
    return train_path, test_path

if __name__ == "__main__":
    # For your case, you want 100 training queries and 50 testing queries
    # Let's create fixed lists of query numbers
    
    # Get all query numbers
    query_dir = "top_150_queries"
    all_query_files = [f for f in os.listdir(query_dir) if f.endswith('_TopTables.csv')]
    all_query_nums = []
    
    for query_file in all_query_files:
        match = re.search(r'query(\d+)_', query_file)
        if match:
            all_query_nums.append(int(match.group(1)))
    
    all_query_nums.sort()
    print(f"Found {len(all_query_nums)} total query numbers")
    
    # Randomly select 100 for training and 50 for testing
    random.seed(42)  # For reproducibility
    random.shuffle(all_query_nums)
    train_queries = sorted(all_query_nums[:100])
    test_queries = sorted(all_query_nums[100:150])
    
    print(f"Selected {len(train_queries)} queries for training")
    print(f"Selected {len(test_queries)} queries for testing")
    
    # Run the labeling with our fixed train/test split
    label_data(
        query_csv_dir="top_150_queries",
        chunks_json_path="/home/mvyas7/TRIM-QA/chunks.json",
        output_dir="labeled_data",
        train_queries_list=train_queries,
        test_queries_list=test_queries,
        fuzzy_match_threshold=0.8
    )