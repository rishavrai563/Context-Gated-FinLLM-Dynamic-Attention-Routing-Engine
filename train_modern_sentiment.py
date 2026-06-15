"""
=============================================================================
QLoRA Financial Sentiment Analysis — Generative Instruction Tuning
=============================================================================
Fine-tunes Llama-3-8B-Instruct for financial sentiment extraction using
QLoRA (4-bit NF4) via Unsloth and SFTTrainer for structured JSON outputs.
"""

# Unsloth MUST be imported before transformers/trl/peft
from unsloth import FastLanguageModel

import os
import time
import warnings
import json

import torch
import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

warnings.filterwarnings("ignore", category=FutureWarning)

# ---- Configuration ----
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR = "./llama3-financial-sentiment-lora"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

NUM_EPOCHS = 2
BATCH_SIZE_TRAIN = 2
BATCH_SIZE_EVAL = 2
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_SEQ_LENGTH = 512
TEST_SPLIT_RATIO = 0.2
RANDOM_SEED = 42
GRADIENT_ACCUMULATION_STEPS = 4

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

def initialize_model():
    """Load 4-bit quantized Llama-3 via Unsloth and attach LoRA adapters."""
    print(f"\n[INFO] Loading model via Unsloth: {MODEL_NAME}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,  # Auto-detect (float16 on T4)
    )

    print(f"[INFO] Attaching LoRA adapters: r={LORA_R}, alpha={LORA_ALPHA}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 2x longer context, 30% less VRAM
        random_state=RANDOM_SEED,
    )

    model.print_trainable_parameters()
    return model, tokenizer

def load_balanced_financial_data():
    """Load, balance, and format datasets into Llama-3 Chat Template."""
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
    min_class_size = df_combined['label'].value_counts().min()
    print(f"[INFO] Balancing classes. Downsampling to: {min_class_size}")

    balanced_dfs = []
    for label in [0, 1, 2]:
        df_label = df_combined[df_combined['label'] == label]
        df_label_sampled = df_label.sample(n=min_class_size, random_state=RANDOM_SEED)
        balanced_dfs.append(df_label_sampled)

    df_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Contextual Gating System Prompt
    system_prompt = (
        "You are an expert Financial AI. You analyze financial news headlines and extract the sentiment (negative, neutral, or positive) and the reasoning.\n"
        "Sector Routing Rules:\n"
        "- Commodities: Focus on supply chains, raw material prices, and weather.\n"
        "- Macro: Focus on inflation, interest rates, and GDP.\n"
        "- Equities: Focus on earnings, M&A, and executive changes.\n"
        "Output strict JSON containing {\"sentiment\": \"<sentiment>\", \"reasoning_token_focus\": \"<reasoning>\"}."
    )

    def format_llama3_prompt(row):
        label_str = ID2LABEL[row['label']]
        reasoning = f"Semantic mapping to {label_str} based on contextual gating sector rules."

        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\nAnalyze this headline: {row['sentence']}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f'{{"sentiment": "{label_str}", "reasoning_token_focus": "{reasoning}"}}<|eot_id|>'
        )
        return prompt

    print("[INFO] Formatting dataset with Contextual Gating and Llama-3 Chat Template...")
    df_balanced['text'] = df_balanced.apply(format_llama3_prompt, axis=1)

    raw_data = Dataset.from_pandas(df_balanced[['text']])
    split_data = raw_data.train_test_split(test_size=TEST_SPLIT_RATIO, seed=RANDOM_SEED)

    train_data = split_data["train"]
    eval_data = split_data["test"]
    print(f"[INFO] Train: {len(train_data)} | Eval: {len(eval_data)}")

    return train_data, eval_data

def create_trainer(model, tokenizer, train_data, eval_data):
    """Configure trl.SFTTrainer for JSON instruction tuning."""
    import math
    total_samples = len(train_data)
    steps_per_epoch = math.ceil(total_samples / BATCH_SIZE_TRAIN)
    total_training_steps = steps_per_epoch * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_training_steps * WARMUP_RATIO)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE_TRAIN,
        per_device_eval_batch_size=BATCH_SIZE_EVAL,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        optim="adamw_8bit",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        eval_strategy="no",
        save_strategy="no",
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        logging_steps=10,
        report_to="none",
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
    )
    return trainer

def final_evaluation(trainer, eval_data, tokenizer):
    """Generative evaluation on a sample to verify strict JSON output."""
    print(f"\n{'='*70}")
    print(" GENERATIVE EVALUATION VERIFICATION")
    print(f"{'='*70}")

    model = trainer.model
    FastLanguageModel.for_inference(model)

    sample_text = eval_data[0]["text"]
    prompt_split = sample_text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")
    prompt = prompt_split[0] + "<|start_header_id|>assistant<|end_header_id|>\n\n"

    print("[INFO] Generating response for prompt...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)

    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    assistant_output = generated.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1]

    print("\n--- EXPECTED JSON OUTPUT ---")
    print(prompt_split[1])
    print("--- ACTUAL GENERATED OUTPUT ---")
    print(assistant_output)
    print("-------------------------------\n")

def main():
    """Main execution pipeline."""
    print("=" * 70)
    print(" QLoRA Financial Sentiment — Generative SFT Pipeline (Unsloth)")
    print(f" Model: {MODEL_NAME}")
    print("=" * 70)

    start_time = time.time()
    model, tokenizer = initialize_model()
    train_data, eval_data = load_balanced_financial_data()
    trainer = create_trainer(model, tokenizer, train_data, eval_data)

    print(f"\n{'='*70}")
    print(f" STARTING SFT TRAINING: {NUM_EPOCHS} epochs")
    print(f"{'='*70}\n")

    train_result = trainer.train()

    print(f"\n[INFO] Training complete! ({(time.time() - start_time) / 60:.1f} min)")
    final_evaluation(trainer, eval_data, tokenizer)

    print(f"[INFO] Saving LoRA adapters to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
