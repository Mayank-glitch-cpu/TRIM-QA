import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

# Import necessary modules from pruning.py
from pruning import EmbeddingModule, URSModule, WeakSupervisionModule, CombinedModule

# Hyperparameters optimized for better F1 score
HP = {
    'lr': 3e-5,               # Increased from 1e-5 for faster learning
    'batch_size': 16,         # Keep same batch size
    'lambda_urs': 0.5,        # More balanced between URS and WS (was 0.3)
    'lambda_ws': 0.5,         # More balanced (was 0.7)
    'pos_weight': 8.0,        # Increased positive class weight (was 5.0)
    'dropout': 0.3,           # Slightly reduced dropout (was 0.4) 
    'focal_gamma': 2.0,       # Reduced gamma for less focus on hard examples
    'focal_alpha': 0.75,      # Adjusted alpha for class balance
    'weight_decay': 5e-5,     # Reduced weight decay (was 1e-4)
    'warmup_ratio': 0.2,      # Increased warmup period (was 0.1)
    'max_grad_norm': 1.0,     # Keep same gradient clipping
    'fp_penalty_weight': 0.2  # Reduced false positive penalty (was 0.3)
}

class PrecisionFocalLoss(nn.Module):
    """Enhanced loss function for better F1 score on imbalanced datasets"""
    def __init__(self, alpha=HP['focal_alpha'], gamma=HP['focal_gamma']):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = HP['pos_weight']  # Weight for positive class

    def forward(self, inputs, targets):
        # Apply positive class weighting to handle imbalance
        pos_weight = torch.tensor(self.pos_weight, device=inputs.device, dtype=inputs.dtype)
        
        # Avoid precision issues by ensuring inputs are properly bounded
        inputs = torch.clamp(inputs, min=1e-7, max=1-1e-7)
        
        # Convert pseudo-logits
        logits = torch.log(inputs / (1 - inputs))
        
        # Use standard binary cross entropy with pos_weight
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets, 
            pos_weight=pos_weight,
            reduction='none'
        )
        
        # Apply focal weighting
        pt = torch.exp(-loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        weighted_loss = focal_weight * loss
        
        # False positive penalty (reduced to improve recall)
        fp_mask = (inputs > 0.6) & (targets < 0.5)
        fp_penalty = torch.where(fp_mask, inputs**2, torch.zeros_like(inputs))
        
        return weighted_loss.mean() + HP['fp_penalty_weight'] * fp_penalty.mean()

class PrecisionEnhancer(nn.Module):
    """Neural module to identify potential false positives"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.fp_filter = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(HP['dropout']),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, h):
        return self.fp_filter(h)

class RelevanceDataset(Dataset):
    """Dataset for training the relevance model."""
    def __init__(self, csv_path, tokenizer, max_len=256):
        """
        Initialize the dataset from a CSV file containing labeled data.
        
        Args:
            csv_path: Path to CSV with columns [query, chunk_text, label]
            tokenizer: Tokenizer for encoding text
            max_len: Maximum sequence length for tokenization
        """
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} examples from {csv_path}")
        
        # Check for class imbalance
        n_positive = df['label'].sum()
        n_negative = len(df) - n_positive
        print(f"Class distribution: {n_positive} positive, {n_negative} negative examples")
        
        # If severe class imbalance, report it
        if n_positive / len(df) < 0.1 or n_positive / len(df) > 0.9:
            print(f"WARNING: Severe class imbalance detected ({n_positive/len(df):.2%} positive examples)")
        
        self.examples = df.to_dict('records')
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        item = self.examples[idx]
        query_text = item["query"]
        chunk_text = item["chunk_text"]
        label = float(item["label"])
        
        return query_text, chunk_text, label

def collate_fn(batch, tokenizer, device):
    """Prepare a batch of examples for the model."""
    queries = [b[0] for b in batch]
    chunks = [b[1] for b in batch]
    labels = torch.tensor([b[2] for b in batch], dtype=torch.float, device=device)
    
    # Tokenize queries and chunks
    query_inputs = tokenizer(queries, padding=True, truncation=True, max_length=128, return_tensors="pt")
    chunk_inputs = tokenizer(chunks, padding=True, truncation=True, max_length=384, return_tensors="pt")
    
    # Move to device
    query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
    chunk_inputs = {k: v.to(device) for k, v in chunk_inputs.items()}
    
    return query_inputs, chunk_inputs, labels

class StableURSModule(URSModule):
    """Modified URSModule with increased dropout for stability and regularization."""
    def __init__(self, hidden_dim):
        super().__init__(hidden_dim)
        # Increased dropout for regularization
        self.dropout = nn.Dropout(HP['dropout'])
        
        # Initialize weights with larger values to avoid vanishing gradients
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.5)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.5)
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=1.5)
        nn.init.xavier_uniform_(self.fc_sigma.weight, gain=0.5)
        
    def forward_train(self, h):
        x = F.relu(self.fc1(h))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # Fixed: Use x instead of h as input
        x = self.dropout(x)
        
        mu = self.fc_mu(x)
        sigma = F.softplus(self.fc_sigma(x))
        z = mu  # Deterministic during training
        
        # Make sure we don't start with all zeros (ensure some neurons fire)
        return torch.sigmoid(z + 0.1), mu, sigma

class TrainableCombinedModule(CombinedModule):
    """Enhanced CombinedModule with precision enhancer for false positive reduction."""
    def __init__(self, hidden_dim):
        super().__init__(hidden_dim)
        self.lambda_urs = HP['lambda_urs']
        self.lambda_ws = HP['lambda_ws']
        self.precision_enhancer = PrecisionEnhancer(hidden_dim)
        
        # Replace the default URS module with our stable version
        self.urs_module = StableURSModule(hidden_dim)
        
        # Initialize weights properly with bias to ensure some positive predictions
        nn.init.xavier_uniform_(self.ws_module.query_projection.weight, gain=1.5)
        nn.init.constant_(self.ws_module.query_projection.bias, 0.2)
        nn.init.xavier_uniform_(self.ws_module.chunk_projection.weight, gain=1.5)
        nn.init.constant_(self.ws_module.chunk_projection.bias, 0.2)
        
    def forward_train(self, chunk_emb, query_emb=None):
        eta_uns, mu, sigma = self.urs_module.forward_train(chunk_emb)
        eta_ws = self.ws_module(chunk_emb, query_emb)
        
        # Get false positive scores
        fp_score = self.precision_enhancer(chunk_emb)
        
        # Combine scores with FP penalty - with small bias to avoid all-zero predictions
        combined = self.lambda_urs*eta_uns + self.lambda_ws*eta_ws + 0.05
        final_score = combined * (1 - fp_score)
        
        return final_score, mu, sigma, eta_uns, eta_ws

def evaluate_model(model, dataloader, device):
    """Evaluate model on a dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for query_inputs, chunk_inputs, labels in dataloader:
            # Get embeddings
            query_emb = model.embedding_module(**query_inputs)
            chunk_emb = model.embedding_module(**chunk_inputs)  # Fixed: was using query_inputs
            
            # Get scores
            scores, _, _, _, _ = model.combined_module.forward_train(chunk_emb, query_emb)
            scores = scores.view(-1).cpu()
            
            # Store predictions and labels
            preds = (scores >= 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.numpy())
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Calculate AUC if there are both positive and negative examples
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_scores)
    else:
        auc = float('nan')
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'auc': auc
    }

class PruningModel(nn.Module):
    """Combined model for pruning."""
    def __init__(self, embedding_model, combined_model):
        super(PruningModel, self).__init__()
        self.embedding_module = embedding_model
        self.combined_module = combined_model
    
    def forward(self, query_inputs, chunk_inputs):
        """Forward pass through the full model."""
        query_emb = self.embedding_module(**query_inputs)
        chunk_emb = self.embedding_module(**chunk_inputs)
        scores, _, _, _, _ = self.combined_module.forward_train(chunk_emb, query_emb)
        return scores

def create_balanced_sampler(dataset):
    """Create sampler that oversamples borderline cases"""
    labels = [int(dataset.dataset.examples[dataset.indices[i]]['label']) for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    weights = 1. / class_counts[labels]
    return WeightedRandomSampler(weights, len(weights))

def get_validation_predictions(model, dataloader):
    """Get model predictions on validation set for calibration"""
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for query_inputs, chunk_inputs, labels in dataloader:
            query_emb = model.embedding_module(**query_inputs)
            chunk_emb = model.embedding_module(**chunk_inputs)  # Fixed: was using query_inputs
            scores, _, _, _, _ = model.combined_module.forward_train(chunk_emb, query_emb)
            
            all_scores.append(scores.view(-1))
            all_labels.append(labels)
    
    return torch.cat(all_scores), torch.cat(all_labels)

def train_model(train_csv, val_split=0.15, test_csv=None, model_name="bert-base-uncased", 
                epochs=15, output_dir="trained_models", seed=42, save_model=True,
                use_amp=False, early_stopping_f1=0.0, max_patience=3):
    """
    Train the pruning model using labeled data with precision-focused optimizations.
    
    Args:
        train_csv: Path to training CSV
        val_split: Proportion of training data to use for validation
        test_csv: Path to test CSV (optional, only used for final evaluation)
        model_name: Name of the pre-trained model to use
        epochs: Number of training epochs
        output_dir: Directory to save trained models
        seed: Random seed for reproducibility
        save_model: Whether to save models during training (default: True)
        use_amp: Whether to use mixed precision training (default: False)
        early_stopping_f1: Minimum F1 score threshold below which training stops (default: 0.0)
        max_patience: Number of epochs to wait before early stopping (default: 3)
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup automatic mixed precision if requested and available
    if use_amp and torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
        from torch.cuda import amp
        scaler = amp.GradScaler()
        autocast = amp.autocast
        print("Using mixed precision (FP16) training for faster computation")
    else:
        use_amp = False
        scaler = None
        # Define a no-op autocast context manager
        from contextlib import contextmanager
        @contextmanager
        def autocast():
            yield
        
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer and models
    print(f"Initializing models with {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    hidden_dim = base_model.config.hidden_size
    
    # Create embedding module
    embedding_module = EmbeddingModule(base_model).to(device)
    
    # Create trainable combined module with precision enhancements
    combined_module = TrainableCombinedModule(hidden_dim).to(device)
    
    # Create datasets and dataloaders
    print(f"Loading datasets...")
    full_train_dataset = RelevanceDataset(train_csv, tokenizer)
    
    # Split training data into train and validation sets
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    print(f"Splitting training data into {train_size} train and {val_size} validation examples")
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Create precision-optimized data loading with balanced sampling
    train_sampler = create_balanced_sampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=HP['batch_size'],
        sampler=train_sampler,
        collate_fn=lambda x: collate_fn(x, tokenizer, device)
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=HP['batch_size'],
        collate_fn=lambda x: collate_fn(x, tokenizer, device)
    )
    
    # Create test dataloader if test data is provided
    test_dataloader = None
    if test_csv and os.path.exists(test_csv):
        print("Loading test set for final evaluation (not used during training)")
        test_dataset = RelevanceDataset(test_csv, tokenizer)
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=HP['batch_size'],
            collate_fn=lambda x: collate_fn(x, tokenizer, device)
        )
    
    # Create combined model
    model = PruningModel(embedding_module, combined_module)
    
    # Use precision-focused loss function
    criterion = PrecisionFocalLoss()
    
    # Using standard Adam optimizer instead of AdamW to avoid compatibility issues
    print("Using standard Adam optimizer with weight decay applied manually")
    optimizer = torch.optim.Adam([
        {'params': embedding_module.parameters(), 'lr': HP['lr']/3},
        {'params': combined_module.parameters(), 'lr': HP['lr']}
    ], lr=HP['lr'], betas=(0.9, 0.999), eps=1e-8)
    
    # Apply weight decay manually outside the optimizer
    weight_decay = HP['weight_decay']
    
    # Linear warmup schedule for learning rate
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * HP['warmup_ratio']),
        num_training_steps=total_steps
    )
    
    # Track metrics
    history = {
        'train_loss': [],
        'train_metrics': [],
        'val_metrics': [],
        'test_metrics': []
    }
    
    # Variables for early stopping focused on precision
    best_precision = 0
    best_f1 = 0
    best_epoch = -1
    patience = max_patience
    patience_counter = 0
    
    # Training loop
    print(f"Starting precision-focused training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for query_inputs, chunk_inputs, labels in progress_bar:
            optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if use_amp:
                with autocast():
                    # Forward pass
                    query_emb = embedding_module(**query_inputs)
                    chunk_emb = embedding_module(**chunk_inputs)
                    
                    # Get combined scores with false positive detection
                    scores, mu, sigma, eta_uns, eta_ws = combined_module.forward_train(chunk_emb, query_emb)
                    scores = scores.view(-1)  # Flatten to match labels
                    
                    # Precision-focused loss
                    loss = criterion(scores, labels)
                    
                    # Apply manual weight decay (L2 regularization)
                    if weight_decay > 0:
                        l2_reg = torch.tensor(0.0).to(device)
                        for param in model.parameters():
                            l2_reg += torch.norm(param)**2
                        loss += weight_decay * l2_reg / 2
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping on scaled gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), HP['max_grad_norm'])
                
                # Update with scaled gradients
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward pass without mixed precision
                query_emb = embedding_module(**query_inputs)
                chunk_emb = embedding_module(**chunk_inputs)
                
                # Get combined scores with false positive detection
                scores, mu, sigma, eta_uns, eta_ws = combined_module.forward_train(chunk_emb, query_emb)
                scores = scores.view(-1)  # Flatten to match labels
                
                # Precision-focused loss
                loss = criterion(scores, labels)
                
                # Apply manual weight decay (L2 regularization)
                if weight_decay > 0:
                    l2_reg = torch.tensor(0.0).to(device)
                    for param in model.parameters():
                        l2_reg += torch.norm(param)**2
                    loss += weight_decay * l2_reg / 2
                
                # Backward pass with gradient clipping for stability
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), HP['max_grad_norm'])
                
                # Update parameters
                optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Track loss
            train_losses.append(loss.item())
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Calculate average loss for this epoch
        avg_train_loss = sum(train_losses) / len(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # Evaluate on train set
        print(f"Evaluating on train set...")
        train_metrics = evaluate_model(model, train_dataloader, device)
        history['train_metrics'].append(train_metrics)
        
        # Print training metrics
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
              f"Train Precision: {train_metrics['precision']:.4f}, "
              f"Recall: {train_metrics['recall']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        
        # Evaluate on validation set
        print(f"Evaluating on validation set...")
        val_metrics = evaluate_model(model, val_dataloader, device)
        history['val_metrics'].append(val_metrics)
        
        # Print validation metrics
        print(f"Validation - Precision: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, "
              f"Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Check for improvement on validation set (precision-focused)
        current_precision = val_metrics['precision']
        current_f1 = val_metrics['f1']
        
        # Early stopping based on minimum F1 score
        if early_stopping_f1 > 0 and epoch > 2 and current_f1 < early_stopping_f1:
            print(f"Early stopping due to low F1 score: {current_f1:.4f} < {early_stopping_f1:.4f}")
            break
            
        # Check for overall improvement
        if current_precision > best_precision:
            print(f"Validation precision improved from {best_precision:.4f} to {current_precision:.4f}")
            best_precision = current_precision
            best_f1 = current_f1
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model if save_model is True
            if save_model:
                best_model_path = os.path.join(output_dir, "best_precision_model.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'embedding_state_dict': embedding_module.state_dict(),
                    'combined_state_dict': combined_module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'config': {
                        'hidden_dim': hidden_dim,
                        'model_name': model_name,
                        'lambda_urs': HP['lambda_urs'],
                        'lambda_ws': HP['lambda_ws']
                    }
                }, best_model_path)
                print(f"Saved new best precision model to {best_model_path}")
        else:
            patience_counter += 1
            print(f"Validation precision did not improve. Patience: {patience_counter}/{patience}")
        
        # Early stopping based on patience
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    print(f"Training completed. Best validation precision: {best_precision:.4f} at epoch {best_epoch + 1}")
    
    # Post-training calibration
    if test_dataloader:
        print("\nPerforming final evaluation and calibration...")
        best_model_path = os.path.join(output_dir, "best_precision_model.pt")
        
        # Check if best model exists
        if save_model and os.path.exists(best_model_path):
            print(f"Loading best precision model from {best_model_path}")
            best_checkpoint = torch.load(best_model_path)
            embedding_module.load_state_dict(best_checkpoint['embedding_state_dict'])
            combined_module.load_state_dict(best_checkpoint['combined_state_dict'])
        else:
            if save_model:
                print(f"Warning: Best precision model not found at {best_model_path}")
            print("Using the final state of the model for calibration and evaluation.")
            # Continue with the current state of the model
        
        # Get validation predictions for calibration
        val_scores, val_labels = get_validation_predictions(model, val_dataloader)
        
        # Train isotonic calibration
        print("Training isotonic calibration model...")
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(val_scores.cpu().numpy(), val_labels.cpu().numpy())
        
        # Save calibrator if save_model is True
        if save_model:
            torch.save(calibrator, os.path.join(output_dir, "calibrator.pth"))
            print(f"Calibrator saved to {os.path.join(output_dir, 'calibrator.pth')}")
        
        # Final evaluation on test set
        test_metrics = evaluate_model(model, test_dataloader, device)
        history['test_metrics'] = [test_metrics]
        
        print(f"Test metrics - Precision: {test_metrics['precision']:.4f}, "
              f"Recall: {test_metrics['recall']:.4f}, "
              f"F1: {test_metrics['f1']:.4f}, "
              f"Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Save final model if save_model is True
    if save_model:
        final_model_path = os.path.join(output_dir, "final_model.pt")
        torch.save({
            'embedding_state_dict': embedding_module.state_dict(),
            'combined_state_dict': combined_module.state_dict(),
            'config': {
                'hidden_dim': hidden_dim,
                'model_name': model_name,
                'lambda_urs': HP['lambda_urs'],
                'lambda_ws': HP['lambda_ws']
            }
        }, final_model_path)
        print(f"Final model saved to {final_model_path}")
    
    # Save training history if save_model is True
    if save_model:
        history_path = os.path.join(output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            # Convert numpy values to Python scalars for JSON serialization
            json_history = {
                'train_loss': [float(x) for x in history['train_loss']],
                'train_metrics': [{k: float(v) for k, v in m.items()} for m in history['train_metrics']],
                'val_metrics': [{k: float(v) for k, v in m.items()} for m in history['val_metrics']],
                'test_metrics': [{k: float(v) for k, v in m.items()} for m in history['test_metrics']] if history['test_metrics'] else []
            }
            json.dump(json_history, f, indent=4)
        print(f"Training history saved to {history_path}")
    
    # Plot training curves with focus on precision if save_model is True
    if save_model:
        plt.figure(figsize=(15, 10))
        
        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot([m['accuracy'] for m in history['train_metrics']], label='Train Accuracy')
        plt.plot([m['accuracy'] for m in history['val_metrics']], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.legend()
        
        # Plot F1
        plt.subplot(2, 2, 3)
        plt.plot([m['f1'] for m in history['train_metrics']], label='Train F1')
        plt.plot([m['f1'] for m in history['val_metrics']], label='Validation F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('F1 Score')
        plt.legend()
        
        # Plot Precision (highlighted)
        plt.subplot(2, 2, 4)
        plt.plot([m['precision'] for m in history['train_metrics']], label='Train Precision', linewidth=2)
        plt.plot([m['precision'] for m in history['val_metrics']], label='Val Precision', linewidth=2)
        plt.plot([m['recall'] for m in history['train_metrics']], label='Train Recall', linestyle='--')
        plt.plot([m['recall'] for m in history['val_metrics']], label='Val Recall', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Precision & Recall (Precision Optimized)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precision_training_plots.png'))
        print(f"Training plots saved to {os.path.join(output_dir, 'precision_training_plots.png')}")
    
    # Return the model and history regardless of save_model setting
    return {
        'model': model,
        'embedding_module': embedding_module,
        'combined_module': combined_module,
        'tokenizer': tokenizer,
        'history': history
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train precision-focused pruning model with labeled data")
    parser.add_argument("--train", type=str, default="labeled_data/train_data.csv",
                        help="Path to training data CSV")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Proportion of training data to use for validation")
    parser.add_argument("--test", type=str, default="labeled_data/test_data.csv",
                        help="Path to test data CSV (only used for final evaluation)")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="Pretrained model name")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--output_dir", type=str, default="trained_models",
                        help="Directory to save trained models")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    # Precision-focused hyperparameters
    parser.add_argument("--lr", type=float, default=HP['lr'],
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=HP['batch_size'],
                        help="Training batch size")
    parser.add_argument("--lambda_urs", type=float, default=HP['lambda_urs'],
                        help="Weight for URS module")
    parser.add_argument("--lambda_ws", type=float, default=HP['lambda_ws'],
                        help="Weight for weak supervision module")
    parser.add_argument("--dropout", type=float, default=HP['dropout'],
                        help="Dropout rate for regularization")
    parser.add_argument("--fp_penalty_weight", type=float, default=HP['fp_penalty_weight'],
                        help="False positive penalty weight")
    parser.add_argument("--focal_gamma", type=float, default=HP['focal_gamma'],
                        help="Focal loss gamma parameter")
    parser.add_argument("--focal_alpha", type=float, default=HP['focal_alpha'],
                        help="Focal loss alpha parameter")
    parser.add_argument("--pos_weight", type=float, default=HP['pos_weight'],
                        help="Positive class weight")
    parser.add_argument("--weight_decay", type=float, default=HP['weight_decay'],
                        help="L2 regularization strength")
    parser.add_argument("--save_model", type=bool, default=True,
                        help="Whether to save models during training")
    parser.add_argument("--use_amp", type=bool, default=False,
                        help="Whether to use mixed precision training")
    parser.add_argument("--early_stopping_f1", type=float, default=0.0,
                        help="Minimum F1 score threshold below which training stops")
    parser.add_argument("--max_patience", type=int, default=3,
                        help="Number of epochs to wait before early stopping")
    
    args = parser.parse_args()
    
    # Update hyperparameters with command line arguments
    HP.update({
        'lr': args.lr,
        'batch_size': args.batch_size,
        'lambda_urs': args.lambda_urs,
        'lambda_ws': args.lambda_ws,
        'dropout': args.dropout,
        'fp_penalty_weight': args.fp_penalty_weight,
        'focal_gamma': args.focal_gamma,
        'focal_alpha': args.focal_alpha,
        'pos_weight': args.pos_weight,
        'weight_decay': args.weight_decay
    })
    
    # Print model summary
    print(f"Training precision-optimized model with hyperparameters:")
    for key, value in HP.items():
        print(f"  {key}: {value}")
    
    # Call the train_model function with parsed arguments
    train_model(
        train_csv=args.train,
        val_split=args.val_split,
        test_csv=args.test,
        model_name=args.model,
        epochs=args.epochs,
        output_dir=args.output_dir,
        seed=args.seed,
        save_model=args.save_model,
        use_amp=args.use_amp,
        early_stopping_f1=args.early_stopping_f1,
        max_patience=args.max_patience
    )