# TRIM-QA Pipeline Documentation

This document provides a comprehensive overview of the TRIM-QA system workflow, from data labeling to model optimization through hyperparameter tuning.

## Table of Contents
1. [Data Labeling Process](#1-data-labeling-process)
2. [Training the Model](#2-training-the-model)
3. [Evaluation Process](#3-evaluation-process)
4. [Hyperparameter Optimization](#4-hyperparameter-optimization)
5. [Key Parameters and Their Significance](#5-key-parameters-and-their-significance)
6. [Troubleshooting Common Issues](#6-troubleshooting-common-issues)

## 1. Data Labeling Process

The first step in building our TRIM-QA system was creating labeled training data for the model.

### Creating Labels

```bash
# Run the labeling script to create training data
python3 label_data.py --input_csv raw_data.csv --output_dir labeled_data
```

This script processes raw query-chunk pairs and creates labeled data with binary relevance judgments (0 for irrelevant, 1 for relevant). The labeled data is automatically split into training, validation, and test sets.

**Key files generated:**
- `labeled_data/train_data.csv` - Training data (70%)
- `labeled_data/test_data.csv` - Test data (30%)
- `labeled_data/query_split_info.json` - Information about the train/test query split

## 2. Training the Model

After labeling, we trained our precision-focused pruning model using the labeled data.

### Basic Training Run

```bash
# Run the basic training script
python3 train_pruning_model.py --train labeled_data/train_data.csv --test labeled_data/test_data.csv --output_dir trained_models/precision_focused
```

### Model Architecture

The model consists of several key components:
- **EmbeddingModule**: Extracts embeddings from text using a pre-trained language model
- **StableURSModule**: Unsupervised Relevance Scoring module with dropout for regularization
- **TrainableCombinedModule**: Combines multiple relevance signals with a precision enhancer

### Training Process

1. Input data is tokenized and split into training and validation sets
2. For each epoch:
   - Batches of query-chunk pairs are processed
   - Query and chunk embeddings are extracted
   - Forward pass through the combined module generates relevance scores
   - PrecisionFocalLoss is calculated
   - Backpropagation updates model weights
   - Validation metrics are calculated
3. Early stopping is applied based on validation precision
4. The best model is saved along with calibration information

### Initial Training Results

Our initial training run showed poor performance with an F1 score of around 0.15, indicating that the model was struggling with the class imbalance in our dataset.

```
Epoch 3/15 - Train Loss: 10.1924, Train Precision: 0.0000, Recall: 0.0000, F1: 0.0000
Evaluating on validation set...
Validation - Precision: 0.0000, Recall: 0.0000, F1: 0.0000, Accuracy: 0.7508
Validation precision did not improve. Patience: 3/3
Early stopping triggered after 3 epochs
```

## 3. Evaluation Process

After training, we evaluated the model on test queries to assess its performance.

### Basic Evaluation Command

```bash
# Evaluate on all test queries
python3 evaluate_on_test_queries.py --model trained_models/precision_focused/final_model.pt --chunks /home/mvyas7/TRIM-QA/chunks.json --split_info labeled_data/query_split_info.json --threshold 0.6
```

### Evaluation with Specific Queries

```bash
# Evaluate on specific test queries
python3 evaluate_on_test_queries.py --model trained_models/precision_focused/final_model.pt --chunks /home/mvyas7/TRIM-QA/chunks.json --split_info labeled_data/query_split_info.json --specific_queries 2 6 8 72 55 46 9 20 58 108 --threshold 0.6
```

### Evaluation Process Explained

1. The script loads the trained model and test queries
2. For each query:
   - Relevant tables are identified
   - Chunks from each table are processed and scored using the model
   - Chunks with scores above the threshold are kept
   - Precision, recall, and F1 scores are calculated
3. Evaluation metrics are saved to CSV files and visualized through plots

### Initial Evaluation Results

The initial model performance was poor, showing:
- Low precision (near 0)
- Low recall (near 0)
- F1 score of approximately 0.15
- The model was mostly predicting negative for all examples

## 4. Hyperparameter Optimization

To improve performance, we implemented hyperparameter optimization using two approaches: comprehensive grid search and focused optimization.

### Comprehensive Grid Search

```bash
# Run full grid search (takes longer but explores more combinations)
python3 hyperparameter_search.py --train labeled_data/train_data.csv --test labeled_data/test_data.csv --epochs 10 --output hyperparameter_search
```

This script systematically explores combinations of hyperparameters by:
1. Creating a grid of hyperparameter values
2. Training models with each valid combination
3. Evaluating performance on validation data
4. Tracking the best performing hyperparameter sets
5. Analyzing which parameters have the most impact on F1 score

### Quick Focused Search

```bash
# Run focused search with carefully selected parameter sets
python3 quick_hyperparameter_search.py --train labeled_data/train_data.csv --test labeled_data/test_data.csv --epochs 8 --output quick_hyperparameter_search
```

The focused search explores 5 carefully selected hyperparameter combinations specifically designed to address the class imbalance and improve F1 score:

1. Baseline with higher learning rate
2. Configuration with higher positive class weight
3. Version emphasizing weak supervision
4. Balanced configuration with lower gamma
5. Recall-focused configuration

### Evaluating with Optimized Hyperparameters

```bash
# Evaluate using the model trained with best hyperparameters
python3 evaluate_on_test_queries.py --model hyperparameter_search/20250415_154234/best_model.pt --chunks /home/mvyas7/TRIM-QA/chunks.json --split_info labeled_data/query_split_info.json --threshold 0.6
```

## 5. Key Parameters and Their Significance

The model's performance is highly dependent on several key hyperparameters:

### Learning Parameters

| Parameter | Description | Significance |
|-----------|-------------|-------------|
| `lr` | Learning rate | Controls how quickly model adapts to the problem; higher values (3e-5 to 5e-5) worked better than default 1e-5 |
| `weight_decay` | L2 regularization strength | Prevents overfitting; lower values (5e-5) worked better than default (1e-4) |
| `batch_size` | Samples per batch | Affects training stability and speed; 16 was optimal for our task |
| `epochs` | Training iterations | More epochs allow learning but risk overfitting; early stopping helps determine optimal count |

### Class Imbalance Handling

| Parameter | Description | Significance |
|-----------|-------------|-------------|
| `pos_weight` | Weight for positive class | Critical parameter due to class imbalance (only ~23% positive examples); increasing to 8.0-15.0 significantly improved recall |
| `focal_alpha` | Focal loss alpha | Controls weight of positive vs. negative examples; 0.75-0.85 worked best |
| `focal_gamma` | Focal loss focusing parameter | Controls focus on hard-to-classify examples; reducing from 2.0 to 1.5 helped with class imbalance |

### Model Architecture

| Parameter | Description | Significance |
|-----------|-------------|-------------|
| `lambda_urs` | Weight for unsupervised relevance scores | Controls contribution of unsupervised signals; balanced 50/50 split with lambda_ws works best |
| `lambda_ws` | Weight for weak supervision | Controls contribution of supervised signals; balanced 50/50 split with lambda_urs works best |
| `dropout` | Dropout rate | Controls regularization strength; 0.3 provided best balance |
| `fp_penalty_weight` | False positive penalty | Reducing from 0.3 to 0.2 or lower improved recall at minor precision cost |

## 6. Troubleshooting Common Issues

### Model Not Learning (F1 = 0)

The most common issue was the model not learning to identify any positive examples, resulting in zero precision, recall, and F1. This was addressed by:

1. **Increasing positive class weight**: Raising `pos_weight` from 5.0 to 8.0 or higher
2. **Reducing false positive penalty**: Lowering `fp_penalty_weight` from 0.3 to 0.2
3. **Using higher learning rate**: Increasing from 1e-5 to 3e-5 or 5e-5
4. **Balancing URS and WS weights**: Setting both `lambda_urs` and `lambda_ws` to 0.5

### TrainableCombinedModule Initialization Error

When running evaluation, you might encounter an error with the `TrainableCombinedModule` expecting 2 arguments but being called with 4. This happens because the module is defined in training code to accept just `hidden_dim`, but inference code tries to pass additional parameters.

**Solution**: Update the `load_trained_model` function in `inference_with_trained_model.py` to instantiate `TrainableCombinedModule` with just `hidden_dim` and set additional parameters afterward:

```python
# Create a TrainableCombinedModule with just the hidden_dim parameter
combined_module = TrainableCombinedModule(hidden_dim).to(device)

# Update lambda values after instantiation if needed
combined_module.lambda_urs = lambda_urs
combined_module.lambda_ws = lambda_ws
```

### Out of Memory Errors

When processing large tables with thousands of chunks, you might encounter GPU memory issues. These can be addressed by:

1. **Reducing batch size**: Lower `batch_size` from 128 to 64 or 32
2. **Limiting chunks per table**: Set `MAX_CHUNKS_PER_TABLE` to 1000-5000
3. **Using mixed precision**: Enable `USE_MIXED_PRECISION` flag
4. **Model quantization**: Enable `USE_MODEL_QUANTIZATION` for inference