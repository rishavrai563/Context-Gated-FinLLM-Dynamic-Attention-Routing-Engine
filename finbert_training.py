"""
=============================================================================
FinBERT Financial News Sentiment Analysis — Training & Evaluation Script
=============================================================================
This script replaces the legacy Keras/TensorFlow Bi-LSTM notebook with a
modern PyTorch + Hugging Face transformers pipeline using ProsusAI/finbert.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
# FinBERT model identifier on Hugging Face Hub
MODEL_NAME = "ProsusAI/finbert"

# Path to the dataset CSV file
# NOTE: Update this path to match your local environment
DATASET_PATH = "News_sentiment_Jan2017_to_Apr2021.csv"

# Directory to save the fine-tuned model and tokenizer
SAVE_DIR = "./finbert-financial-sentiment"

# Training hyperparameters
BATCH_SIZE_TRAIN = 16        # Training batch size
BATCH_SIZE_EVAL = 32         # Evaluation batch size
NUM_EPOCHS = 4               # Number of fine-tuning epochs (no early stopping)
LEARNING_RATE = 2e-5         # AdamW learning rate (standard for BERT)
WARMUP_RATIO = 0.1           # Fraction of total steps used for linear warmup
MAX_SEQ_LENGTH = 128         # Maximum token sequence length for FinBERT
WEIGHT_DECAY = 0.01          # L2 regularization weight decay
TEST_SIZE = 0.2              # Fraction of data reserved for validation
RANDOM_STATE = 42            # Random seed for reproducibility

# FinBERT's native label mapping (must match pre-trained head)
# The pre-trained FinBERT model uses: {0: "positive", 1: "negative", 2: "neutral"}
LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}

# Bi-LSTM baseline metrics (from original notebook output)
BILSTM_BASELINE = {
    "val_accuracy": 0.7952,
    "val_loss": 0.4315,
    "train_accuracy": 0.8041,
    "train_loss": 0.4191,
}
def get_device():
    """Auto-detect the best available compute device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU (training will be slow)")
    return device
def load_and_preprocess_data(csv_path):
    """Load the financial news dataset and prepare it for FinBERT fine-tuning."""
    print("[INFO] Loading dataset from:", csv_path)
    df = pd.read_csv(csv_path)
    
    # Display dataset info
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    print(f"\n[INFO] First 5 rows:")
    print(df.head())
    initial_count = len(df)
    df = df.dropna(subset=["Title"])
    dropped = initial_count - len(df)
    if dropped > 0:
        print(f"[INFO] Dropped {dropped} rows with missing titles")
    # Original dataset has: "POSITIVE", "NEGATIVE" (uppercase strings)
    # FinBERT expects: 0=positive, 1=negative, 2=neutral
    sentiment_map = {"POSITIVE": 0, "NEGATIVE": 1}  # No neutral in this dataset
    df["label"] = df["sentiment"].map(sentiment_map)
    
    # Drop any rows where mapping failed (unexpected labels)
    unmapped = df["label"].isna().sum()
    if unmapped > 0:
        print(f"[WARNING] Dropping {unmapped} rows with unmapped sentiment labels")
        df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    print(f"\n[INFO] Label distribution:")
    for label_id, label_name in ID2LABEL.items():
        count = (df["label"] == label_id).sum()
        if count > 0:
            print(f"  {label_name} (id={label_id}): {count} samples")
    texts = df["Title"].values
    labels = df["label"].values
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,  # Preserve label distribution in both splits
    )
    
    print(f"\n[INFO] Train samples: {len(train_texts)}")
    print(f"[INFO] Validation samples: {len(val_texts)}")
    
    return train_texts, val_texts, train_labels, val_labels
class FinancialNewsDataset(Dataset):
    """Custom PyTorch Dataset for financial news headlines."""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

def tokenize_data(tokenizer, texts, max_length=MAX_SEQ_LENGTH):
    """Tokenize a list of text strings using the FinBERT tokenizer."""
    print(f"[INFO] Tokenizing {len(texts)} samples (max_length={max_length})...")
    encodings = tokenizer(
        list(texts),                  # Ensure it's a Python list
        padding=True,                 # Pad shorter sequences to max in batch
        truncation=True,              # Truncate sequences exceeding max_length
        max_length=max_length,        # Maximum token count
        return_tensors="pt",          # Return PyTorch tensors directly
    )
    print(f"[INFO] Tokenization complete. Shape: {encodings['input_ids'].shape}")
    return encodings
def train_model(model, train_loader, val_loader, device, num_epochs=NUM_EPOCHS):
    """Fine-tune the FinBERT model on the financial news dataset."""
    # AdamW separates weight decay from the gradient update (better for Transformers)
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    # Linear warmup: LR ramps from 0 to LEARNING_RATE over warmup steps
    # Then linearly decays to 0 over the remaining steps
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    print(f"\n{'='*70}")
    print(f" TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"  Epochs:          {num_epochs}")
    print(f"  Total steps:     {total_steps}")
    print(f"  Warmup steps:    {warmup_steps}")
    print(f"  Learning rate:   {LEARNING_RATE}")
    print(f"  Weight decay:    {WEIGHT_DECAY}")
    print(f"  Train batches:   {len(train_loader)}")
    print(f"  Val batches:     {len(val_loader)}")
    print(f"  Device:          {device}")
    print(f"{'='*70}\n")
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()  # Enable dropout, batch norm training mode
        total_train_loss = 0.0
        train_preds = []
        train_true = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch tensors to the compute device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Zero gradients from previous step
            optimizer.zero_grad()
            
            # Forward pass: model returns loss + logits when labels are provided
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients (common for Transformers)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights and learning rate
            optimizer.step()
            scheduler.step()
            
            # Accumulate metrics
            total_train_loss += loss.item()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            train_preds.extend(preds)
            train_true.extend(labels.cpu().numpy())
            
            # Progress logging every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_train_loss / (batch_idx + 1)
                print(f"  Epoch {epoch+1}/{num_epochs} | "
                      f"Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {avg_loss:.4f}")
        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = accuracy_score(train_true, train_preds)
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, device)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n  Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"    Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"    Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"    Time:       {epoch_time:.1f}s\n")
    
    return history
def evaluate_model(model, data_loader, device):
    """Evaluate the model on a dataset and compute loss + accuracy."""
    model.eval()  # Disable dropout, use eval mode for batch norm
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # No gradient tracking = faster + less memory
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)

def print_evaluation_report(val_preds, val_labels):
    """Print comprehensive evaluation metrics and compare with Bi-LSTM baseline."""
    acc = accuracy_score(val_labels, val_preds)
    f1_weighted = f1_score(val_labels, val_preds, average="weighted")
    f1_macro = f1_score(val_labels, val_preds, average="macro")
    unique_labels = sorted(set(val_labels) | set(val_preds))
    target_names = [ID2LABEL[l] for l in unique_labels]
    
    print(f"\n{'='*70}")
    print(f" FINBERT EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  F1 (weighted):     {f1_weighted:.4f}")
    print(f"  F1 (macro):        {f1_macro:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        val_labels, val_preds,
        labels=unique_labels,
        target_names=target_names,
        digits=4,
    ))
    print(f"\n{'='*70}")
    print(f" COMPARISON: FinBERT vs Bi-LSTM Baseline")
    print(f"{'='*70}")
    print(f"  {'Metric':<25} {'Bi-LSTM':<15} {'FinBERT':<15} {'Delta':<15}")
    print(f"  {'-'*65}")
    
    bilstm_acc = BILSTM_BASELINE["val_accuracy"]
    delta_acc = acc - bilstm_acc
    sign = "+" if delta_acc >= 0 else ""
    print(f"  {'Val Accuracy':<25} {bilstm_acc:<15.4f} {acc:<15.4f} {sign}{delta_acc:<15.4f}")
    
    bilstm_loss = BILSTM_BASELINE["val_loss"]
    print(f"  {'Val Loss':<25} {bilstm_loss:<15.4f} {'N/A':<15} {'N/A':<15}")
    print(f"  {'F1 (weighted)':<25} {'N/A':<15} {f1_weighted:<15.4f} {'N/A':<15}")
    print(f"{'='*70}\n")
    cm = confusion_matrix(val_labels, val_preds, labels=unique_labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names,
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("FinBERT Confusion Matrix — Financial News Sentiment")
    plt.tight_layout()
    plt.savefig("finbert_confusion_matrix.png", dpi=150)
    print("[INFO] Confusion matrix saved to: finbert_confusion_matrix.png")
    plt.show()

def plot_training_history(history):
    """Plot training and validation loss/accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("FinBERT Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy")
    ax2.plot(epochs, history["val_acc"], "r-o", label="Val Accuracy")
    # Add Bi-LSTM baseline reference line
    ax2.axhline(
        y=BILSTM_BASELINE["val_accuracy"],
        color="green", linestyle="--", alpha=0.7,
        label=f"Bi-LSTM Baseline ({BILSTM_BASELINE['val_accuracy']:.4f})"
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("FinBERT Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("finbert_training_curves.png", dpi=150)
    print("[INFO] Training curves saved to: finbert_training_curves.png")
    plt.show()
def save_model(model, tokenizer, save_dir=SAVE_DIR):
    """Save the fine-tuned FinBERT model and tokenizer to disk."""
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\n[INFO] Model saved to: {save_dir}")
    print(f"[INFO] Contents: {os.listdir(save_dir)}")
def main():
    """Main execution pipeline for FinBERT fine-tuning and evaluation."""
    print("=" * 70)
    print(" FinBERT Financial News Sentiment Analysis")
    print(" Migrating from Bi-LSTM to Transformer Architecture")
    print("=" * 70)
    device = get_device()
    train_texts, val_texts, train_labels, val_labels = load_and_preprocess_data(
        DATASET_PATH
    )
    print(f"\n[INFO] Loading FinBERT tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_encodings = tokenize_data(tokenizer, train_texts)
    val_encodings = tokenize_data(tokenizer, val_texts)
    train_dataset = FinancialNewsDataset(train_encodings, train_labels)
    val_dataset = FinancialNewsDataset(val_encodings, val_labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,   # Shuffle training data each epoch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,   # No shuffling for reproducible evaluation
    )
    
    print(f"\n[INFO] Train DataLoader: {len(train_loader)} batches "
          f"(batch_size={BATCH_SIZE_TRAIN})")
    print(f"[INFO] Val DataLoader:   {len(val_loader)} batches "
          f"(batch_size={BATCH_SIZE_EVAL})")
    print(f"\n[INFO] Loading pre-trained model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,         # 3-class: positive, negative, neutral
        id2label=ID2LABEL,    # Map integer IDs to human-readable labels
        label2id=LABEL2ID,    # Map human-readable labels to integer IDs
    )
    model.to(device)  # Move model weights to GPU/CPU
    
    # Display model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters:     {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")
    history = train_model(model, train_loader, val_loader, device, NUM_EPOCHS)
    print("\n[INFO] Running final evaluation on validation set...")
    val_loss, val_acc, val_preds, val_true = evaluate_model(
        model, val_loader, device
    )
    print_evaluation_report(val_preds, val_true)
    plot_training_history(history)
    save_model(model, tokenizer)
    
    print("\n" + "=" * 70)
    print(" Training & Evaluation Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
