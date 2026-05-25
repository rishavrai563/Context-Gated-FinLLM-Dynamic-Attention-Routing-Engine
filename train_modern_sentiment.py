"""
=============================================================================
QLoRA Financial Sentiment Analysis — Training Pipeline
=============================================================================
This script fine-tunes ModernBERT-base for financial sentiment classification
using QLoRA.
"""

import os
import time
import warnings

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# Suppress noisy warnings from transformers/bitsandbytes
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
MODEL_NAME = "answerdotai/ModernBERT-base"
DATASET_NAME = "PhraseBank + FinTwit (Balanced)"
DATASET_CONFIG = "Mixed"
# Where the trained LoRA adapters will be saved
OUTPUT_DIR = "./modernbert-financial-sentiment-lora"
# Financial PhraseBank label scheme
NUM_LABELS = 3
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = "all-linear"
NUM_EPOCHS = 4              # Full training, NO early stopping
BATCH_SIZE_TRAIN = 16       # Per-device training batch size
BATCH_SIZE_EVAL = 32        # Per-device evaluation batch size
LEARNING_RATE = 2e-4        # Higher than full FT (standard for LoRA)
WEIGHT_DECAY = 0.01         # L2 regularization
WARMUP_RATIO = 0.1          # 10% of total steps for linear LR warmup
MAX_SEQ_LENGTH = 128        # Max token length (financial headlines are short)
TEST_SPLIT_RATIO = 0.2      # 80/20 train/eval split
RANDOM_SEED = 42            # Reproducibility
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch = 16 * 2 = 32
def get_quantization_config():
    """Create a BitsAndBytesConfig for 4-bit model quantization."""
    if not torch.cuda.is_available():
        print("[WARNING] CUDA not available. Cannot use 4-bit quantization.")
        print("[WARNING] Falling back to FP32 (full precision).")
        return None

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("[INFO] 4-bit NF4 quantization configured (double quant enabled)")
    return bnb_config
def get_lora_config():
    """Create a LoRA configuration for PEFT."""
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    print(f"[INFO] LoRA config: r={LORA_R}, alpha={LORA_ALPHA}, "
          f"target={LORA_TARGET_MODULES}, dropout={LORA_DROPOUT}")
    return lora_config
def load_balanced_financial_data(tokenizer):
    """Load, merge, and balance the datasets."""
    import pandas as pd
    from datasets import Dataset, ClassLabel, concatenate_datasets

    print(f"\n[INFO] Loading datasets for combined mixture...")
    
    pb_dataset = load_dataset("FinanceMTEB/financial_phrasebank")
    phrasebank = concatenate_datasets([pb_dataset["train"], pb_dataset["test"]])
    
    print("  Loading Twitter Financial News...")
    fintwit = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")

    df_pb = pd.DataFrame(phrasebank)
    if 'text' in df_pb.columns:
        df_pb = df_pb.rename(columns={'text': 'sentence'})

    df_ft = pd.DataFrame(fintwit)
    label_mapping = {0: 0, 1: 2, 2: 1}
    df_ft['label'] = df_ft['label'].map(label_mapping)
    df_ft = df_ft.rename(columns={'text': 'sentence'})

    df_combined = pd.concat([df_pb[['sentence', 'label']], df_ft[['sentence', 'label']]], ignore_index=True)
    print(f"[INFO] Raw combined dataset size: {len(df_combined)}")

    min_class_size = df_combined['label'].value_counts().min()
    print(f"[INFO] Balancing classes. Downsampling to minority class size: {min_class_size}")
    
    balanced_dfs = []
    for label in [0, 1, 2]:
        df_label = df_combined[df_combined['label'] == label]
        df_label_sampled = df_label.sample(n=min_class_size, random_state=RANDOM_SEED)
        balanced_dfs.append(df_label_sampled)
        
    df_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"[INFO] Final balanced dataset size: {len(df_balanced)} ({min_class_size} samples per class)")

    raw_data = Dataset.from_pandas(df_balanced)
    raw_data = raw_data.cast_column("label", ClassLabel(names=["negative", "neutral", "positive"]))
    label_counts = {}
    for label_id, label_name in ID2LABEL.items():
        count = sum(1 for ex in raw_data if ex["label"] == label_id)
        label_counts[label_name] = count
        print(f"  {label_name}: {count} samples")
    split_data = raw_data.train_test_split(
        test_size=TEST_SPLIT_RATIO,
        seed=RANDOM_SEED,
        stratify_by_column="label",  # Preserve label distribution
    )
    train_data = split_data["train"]
    eval_data = split_data["test"]
    print(f"[INFO] Train: {len(train_data)} | Eval: {len(eval_data)}")
    def tokenize_fn(examples):
        """
        Tokenize a batch of sentences. Applied via dataset.map() for efficiency.
        """
        return tokenizer(
            examples["sentence"],
            padding="max_length",       # Pad to MAX_SEQ_LENGTH
            truncation=True,            # Truncate if longer than max
            max_length=MAX_SEQ_LENGTH,  # Financial headlines are typically short
        )

    # Apply tokenization to all samples (batched for speed)
    print("[INFO] Tokenizing dataset...")
    train_data = train_data.map(tokenize_fn, batched=True)
    eval_data = eval_data.map(tokenize_fn, batched=True)

    # Set format to PyTorch tensors
    train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    eval_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return train_data, eval_data
def initialize_model(bnb_config, lora_config):
    """Load base model and attach LoRA adapters."""
    print(f"\n[INFO] Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Ensure tokenizer has a pad token (some models don't by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("[INFO] Set pad_token = eos_token")
    print(f"[INFO] Loading model: {MODEL_NAME}")
    model_kwargs = {
        "num_labels": NUM_LABELS,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
    }

    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"  # Automatically place layers on GPU
        print("[INFO] Loading with 4-bit quantization...")
    else:
        print("[INFO] Loading in FP32 (no quantization)...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, **model_kwargs
    )
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
        print("[INFO] Model prepared for k-bit training")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer
def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for the Trainer callback.

    Called automatically by HuggingFace Trainer after each evaluation step.
    Receives the model's predictions and computes accuracy + F1 scores.

    Args:
        eval_pred: EvalPrediction object with .predictions and .label_ids

    Returns:
        dict: Metric name → value mapping
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    f1_weighted = f1_score(labels, predictions, average="weighted")
    f1_macro = f1_score(labels, predictions, average="macro")

    return {
        "accuracy": acc,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
    }
def create_trainer(model, tokenizer, train_data, eval_data):
    """Configure the HuggingFace Trainer for QLoRA fine-tuning."""
    # warmup_ratio is deprecated in transformers >=5.2; use warmup_steps instead
    import math
    total_samples = len(train_data)
    steps_per_epoch = math.ceil(total_samples / BATCH_SIZE_TRAIN)
    total_training_steps = steps_per_epoch * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_training_steps * WARMUP_RATIO)
    print(f"[INFO] Computed warmup_steps={warmup_steps} from ratio={WARMUP_RATIO} "
          f"(total_steps={total_training_steps})")
    if torch.cuda.is_available():
        optim_name = "paged_adamw_8bit"  # 8-bit AdamW: saves ~50% optimizer VRAM
    else:
        optim_name = "adamw_torch"  # Standard AdamW for CPU

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE_TRAIN,
        per_device_eval_batch_size=BATCH_SIZE_EVAL,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        optim=optim_name,
        # FP16 mixed precision: compute in FP16, accumulate gradients in FP32
        # This is compatible with 4-bit quantized base weights
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",     # Evaluate after every epoch
        save_strategy="epoch",     # Save checkpoint after every epoch
        load_best_model_at_end=False,  # No early stopping — train full epochs
        metric_for_best_model="accuracy",
        gradient_checkpointing=True,       # Recompute activations to save VRAM
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        logging_steps=25,
        logging_first_step=True,
        report_to="none",  # Disable W&B / MLflow logging
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
        dataloader_num_workers=2,
        remove_unused_columns=True,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    return trainer
def final_evaluation(trainer, eval_data):
    """Run evaluation and print report."""
    print(f"\n{'='*70}")
    print(" FINAL EVALUATION RESULTS")
    print(f"{'='*70}")

    # Get predictions
    predictions = trainer.predict(eval_data)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    # Overall metrics
    acc = accuracy_score(labels, preds)
    f1_w = f1_score(labels, preds, average="weighted")
    f1_m = f1_score(labels, preds, average="macro")

    print(f"\n  Accuracy:       {acc:.4f}")
    print(f"  F1 (weighted):  {f1_w:.4f}")
    print(f"  F1 (macro):     {f1_m:.4f}")

    unique_labels = sorted(set(labels) | set(preds))
    target_names = [ID2LABEL[l] for l in unique_labels]
    print(f"\n  Classification Report:")
    print(classification_report(
        labels, preds,
        labels=unique_labels,
        target_names=target_names,
        digits=4,
    ))

    print(f"\n{'='*70}")
    print(" COMPARISON WITH BASELINES")
    print(f"{'='*70}")
    print(f"  {'Model':<30} {'Accuracy':<12} {'F1 (weighted)':<15}")
    print(f"  {'-'*55}")
    print(f"  {'Bi-LSTM (original)':<30} {'0.7952':<12} {'N/A':<15}")
    print(f"  {'FinBERT (full FT)':<30} {'~0.86':<12} {'~0.86':<15}")
    print(f"  {'ModernBERT + QLoRA':<30} {acc:<12.4f} {f1_w:<15.4f}")
    print(f"{'='*70}\n")
def main():
    """Main execution pipeline."""
    print("=" * 70)
    print(" QLoRA Financial Sentiment Analysis — Training Pipeline")
    print(f" Model: {MODEL_NAME}")
    print(f" Dataset: {DATASET_NAME} ({DATASET_CONFIG})")
    print("=" * 70)

    start_time = time.time()
    bnb_config = get_quantization_config()
    lora_config = get_lora_config()
    model, tokenizer = initialize_model(bnb_config, lora_config)
    train_data, eval_data = load_balanced_financial_data(tokenizer)
    trainer = create_trainer(model, tokenizer, train_data, eval_data)
    print(f"\n{'='*70}")
    print(f" STARTING TRAINING: {NUM_EPOCHS} epochs, no early stopping")
    print(f"{'='*70}\n")

    train_result = trainer.train()

    # Log training metrics
    print(f"\n[INFO] Training complete!")
    print(f"  Total time:    {(time.time() - start_time) / 60:.1f} minutes")
    print(f"  Train loss:    {train_result.training_loss:.4f}")
    print(f"  Train samples: {train_result.metrics.get('train_samples_per_second', 'N/A')}")
    final_evaluation(trainer, eval_data)
    # Only the LoRA adapter weights are saved (~2MB), NOT the base model.
    # To use the model later:
    #   1. Load the base model (ModernBERT-base)
    #   2. Load LoRA adapters: PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    print(f"[INFO] Saving LoRA adapters to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # List saved files
    saved_files = os.listdir(OUTPUT_DIR)
    print(f"[INFO] Saved files: {saved_files}")
    total_size = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in saved_files
        if os.path.isfile(os.path.join(OUTPUT_DIR, f))
    )
    print(f"[INFO] Total adapter size: {total_size / 1024 / 1024:.1f} MB")
    print(f"       (vs ~600 MB for full FP32 model)")

    print("\n" + "=" * 70)
    print(" Training Pipeline Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
