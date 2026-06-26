# Context-Gated FinLLM: Dynamic Attention Routing Engine

> A Wall-Street-inspired, two-tier hybrid sentiment analysis engine that routes financial headlines through a lightning-fast regex sieve before escalating complex cases to a fine-tuned Llama-3-8B model.

---

## Table of Contents

- [Background](#background)
- [Objective](#objective)
- [Architecture Evolution](#architecture-evolution)
  - [V1 — Bi-LSTM](#v1--bi-lstm)
  - [V2 — FinBERT](#v2--finbert)
  - [V3 — ModernBERT + QLoRA](#v3--modernbert--qlora)
  - [V4 — Generative Instruction Tuning](#v4--generative-instruction-tuning)
  - [V5 — GGUF Hybrid Routing Engine](#v5--gguf-hybrid-routing-engine)
- [Hybrid Routing Architecture](#hybrid-routing-architecture)
- [Training Strategy](#training-strategy)
- [Benchmark Results](#benchmark-results)
- [Improvement Roadmap](#improvement-roadmap)
- [Deployment](#deployment)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)

---

## Background

The financial industry generates vast amounts of news and data every day, which can significantly impact market sentiment. Traditional NLP models struggle with financial text because the same word can carry opposite meanings depending on sector context — a phenomenon known as **Semantic Domain Inversion** (e.g., "yields spike" is positive for bond sellers but negative for borrowers).

## Objective

Develop a highly accurate, State-of-the-Art (SOTA) financial news sentiment analysis system utilizing Parameter-Efficient Fine-Tuning, capable of:
- Classifying headlines as **POSITIVE**, **NEGATIVE**, or **NEUTRAL** with institutional-grade accuracy.
- Providing structured **reasoning** for each classification in JSON format.
- Operating within the constraints of consumer-grade hardware (6GB VRAM).

---

## Architecture Evolution

This project evolved through **5 architectural versions**, each addressing specific limitations of the previous approach.

### V1 — Bi-LSTM
- **Framework:** Keras / TensorFlow
- **Dataset:** India Financial News Headlines (Kaggle) — binary classification (Positive/Negative)
- **Approach:** Bidirectional LSTM with TF-Hub Universal Sentence Encoder embeddings.
- **Limitation:** No contextual understanding of financial jargon. Binary classification missed the critical "Neutral" class entirely.

### V2 — FinBERT
- **Framework:** PyTorch / Hugging Face Transformers
- **Base Model:** `ProsusAI/finbert` (domain-specific financial BERT)
- **Approach:** Full fine-tuning of all 110M parameters on the India Financial News dataset.
- **Limitation:** High VRAM usage (~8GB+) during training. Full fine-tuning risked catastrophic forgetting of FinBERT's pre-trained financial knowledge.

### V3 — ModernBERT + QLoRA
- **Framework:** PyTorch / PEFT / BitsAndBytes
- **Base Model:** `answerdotai/ModernBERT-base` (2024 SOTA encoder)
- **Approach:** Parameter-Efficient Fine-Tuning using QLoRA with 4-bit NF4 quantization. Only ~0.5% of parameters were trained via low-rank adapter matrices.
- **Dataset:** Introduced **Financial PhraseBank** (expert-annotated, 3-class) alongside the original dataset.
- **Limitation:** Encoder-only models can classify but cannot *explain* their reasoning — a critical requirement for institutional trust.

### V4 — Generative Instruction Tuning
- **Framework:** PyTorch / trl.SFTTrainer / Unsloth
- **Base Model:** `meta-llama/Meta-Llama-3-8B-Instruct`
- **Approach:** Transitioned from an encoder to a **Decoder LLM** for generative sentiment analysis. Fine-tuned using QLoRA (4-bit NF4, rank=16) with `trl.SFTTrainer`.
- **Key Innovation:** Introduced a **Contextual Gating System Prompt** — a set of sector-specific routing rules injected into the Llama-3 Chat Template to eliminate Semantic Domain Inversion.
- **Dataset:** Blended **Financial PhraseBank** + **Twitter Financial News Sentiment** into a unified instruction-tuning corpus formatted with Llama-3 Chat Templates.
- **Output Format:** The model natively outputs structured JSON: `{"sentiment": "...", "reasoning_token_focus": "..."}`.

### V5 — GGUF Hybrid Routing Engine *(Current)*
- **Inference Engine:** `llama-cpp-python` (GGUF)
- **Approach:** Merged LoRA adapters into the base model and exported to **Q4_K_M GGUF** format (~4.9GB). Replaced the memory-heavy `transformers` library with `llama.cpp` for optimized C++ inference.
- **Key Innovation:** Built the **Hybrid Routing Engine** — a two-tier inference pipeline that dramatically reduces cost by only sending ambiguous headlines to the expensive LLM.
- **Hardware:** Fits within a consumer **NVIDIA RTX 3050 (6GB VRAM)** using hybrid GPU/CPU layer offloading (`n_gpu_layers=25` of 33 total layers).

---

## Hybrid Routing Architecture

The engine mirrors how quantitative trading desks process news — only headlines that truly need deep reasoning ever touch the expensive LLM.

```
                    ┌─────────────────────┐
                    │   Input Headline    │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │   Paradox Trigger    │──── "inflation", "despite", "fed"
                    │      Detection       │     detected? ──────────────────┐
                    └─────────┬───────────┘                                 │
                         No   │                                             │
                    ┌─────────▼───────────┐                                 │
                    │  Bullish Keywords   │──── Match? → POSITIVE (85%)     │
                    └─────────┬───────────┘                                 │
                         No   │                                             │
                    ┌─────────▼───────────┐                                 │
                    │  Bearish Keywords   │──── Match? → NEGATIVE (85%)     │
                    └─────────┬───────────┘                                 │
                         No   │                                             │
                    ┌─────────▼───────────┐                                 │
                    │  Neutral Keywords   │──── Match? → NEUTRAL (80%)      │
                    └─────────┬───────────┘                                 │
                         No   │                                             │
                              ├─────────────────────────────────────────────┘
                    ┌─────────▼───────────┐
                    │  🧠 Llama-3-8B LLM  │
                    │  (Micro-Reasoning   │
                    │   Chain-of-Thought) │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │   JSON Response     │
                    │ {sentiment, reason} │
                    └─────────────────────┘
```

**Fast-Path (~2ms):** Regex keyword scan for obvious headlines. Catches bullish, bearish, and neutral patterns without invoking the LLM.

**Slow-Path (~2-7s):** Routes ambiguous or paradoxical headlines (containing triggers like "inflation", "despite", "guidance cut") to the Llama-3-8B model with a Micro-Reasoning Chain-of-Thought prompt capped at 64 tokens for latency control.

---

## Training Strategy

### Dataset Blending

A key challenge in financial sentiment analysis is the gap between **formal financial writing** and **real-time social media noise**. To build a model robust to both styles, we blended two fundamentally different data sources:

| Dataset | Source | Style | Classes |
|---------|--------|-------|---------|
| **Financial PhraseBank** | Expert-annotated corporate filings | Formal, structured | Positive, Negative, Neutral |
| **Twitter Financial News** | Social media posts & tweets | Noisy, informal, slang-heavy | Bearish, Bullish, Neutral |

### Instruction Tuning Format

The blended corpus was formatted into **Llama-3 Chat Templates** using `trl.SFTTrainer`, transforming each sample into an instruction-response pair:

```
<|start_header_id|>system<|end_header_id|>
You are a Quantitative Finance Sentiment Engine. [Contextual Gating Rules]...
<|start_header_id|>user<|end_header_id|>
Headline: "Apple expands operations into Southeast Asia"
<|start_header_id|>assistant<|end_header_id|>
{"sentiment": "POSITIVE", "reasoning_token_focus": "operational expansion signals growth"}
```

### QLoRA Configuration

| Parameter | Value |
|-----------|-------|
| Quantization | 4-bit NF4 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Target Modules | q_proj, k_proj, v_proj, o_proj |
| Trainable Parameters | ~0.5% of 8B |

---

## Benchmark Results

Evaluated on the **FiQA-2018** dataset (Financial Question Answering & Microblogs) — a completely **out-of-sample** dataset that the model has never seen during training. This ensures zero data leakage.

### Classification Metrics

| Metric | Score | Industry Benchmark |
|--------|-------|--------------------|
| **Accuracy** | 74.00% | — |
| **Weighted F1-Score** | 0.73 | > 0.70 |
| **Macro F1-Score** | 0.74 | > 0.65 |
| **MCC (Matthews)** | 0.6094 | > 0.50 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.76 | 0.84 | **0.80** | 19 |
| Neutral | 0.64 | 0.53 | 0.58 | 17 |
| Positive | 0.80 | 0.86 | **0.83** | 14 |

### Confusion Matrix

```
                Predicted
              Neg  Neu  Pos
Actual Neg  [ 16    3    0 ]
       Neu  [  5    9    3 ]
       Pos  [  0    2   12 ]
```

**Key Observations:**
- **Negative detection is strong** (84% recall) — the model rarely misses bearish signals.
- **Positive detection is excellent** (86% recall, 0.83 F1) — growth and expansion headlines are well-captured.
- **Neutral is the weakest class** (53% recall) — the model tends to over-classify neutral/factual headlines as negative (5 false negatives in the confusion matrix).

### Operational Metrics

| Metric | Score |
|--------|-------|
| Fast-Path Usage | 4.00% of headlines |
| Average Latency | 7,093 ms |
| P99 Latency | 10,331 ms |

---

## Improvement Roadmap

Based on the benchmark analysis, these are the identified areas for improvement:

### 1. Neutral Class Recall (Current: 53% → Target: 70%+)
**Problem:** The model misclassifies factual, boring news as "Negative" (5 out of 17 neutral samples).
**Solution:** Expand the Fast-Path `neutral_keywords` dictionary to catch more factual patterns (dates, appointments, meeting announcements) before they reach the LLM. Additionally, augment the training corpus with more pure-neutral examples.

### 2. Fast-Path Coverage (Current: 4% → Target: 25-30%)
**Problem:** Only 2 out of 50 headlines were resolved by the regex sieve, meaning 96% of requests hit the expensive LLM.
**Solution:** Broaden the bullish/bearish/neutral keyword dictionaries with domain-specific financial vocabulary. Introduce n-gram pattern matching for multi-word financial idioms (e.g., "beat the street", "priced in").

### 3. Latency Reduction (Current: 7.1s avg → Target: < 3s)
**Problem:** The current average latency of 7.1 seconds is too high for time-sensitive trading signals.
**Solution:**
  - Reduce `n_ctx` from 1024 to 512 (most headlines are < 50 tokens).
  - Investigate `Q3_K_M` quantization (smaller model, ~20% faster inference).
  - Increase `n_gpu_layers` from 25 to 30 on GPUs with > 6GB VRAM.

### 4. MCC Score (Current: 0.61 → Target: 0.75+)
**Problem:** While 0.61 is above the 0.50 institutional threshold, there is room for improvement.
**Solution:** Fine-tune on a larger, more diverse corpus including SEC filings, earnings call transcripts, and Bloomberg-style headlines to improve generalization across all three classes equally.

---

## Deployment

### Architecture

| Component | Technology | Port |
|-----------|-----------|------|
| **Backend API** | FastAPI + llama.cpp | Hugging Face Spaces (port 7860) |
| **Streamlit (Testing)** | Streamlit | localhost:8501 |

### Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 6 GB (RTX 3050) | 8 GB (RTX 4060) |
| System RAM | 16 GB | 32 GB |
| Disk Space | 10 GB (model weights) | 15 GB |

> **⚠️ Important:** Only one model-loading process (either FastAPI or Streamlit) can be active at a time on 6GB VRAM GPUs. Running both simultaneously will trigger a CUDA Out-Of-Memory crash.

---

## Repository Structure

```
├── hybrid_routing_engine.py        # Core two-tier routing engine
├── main.py                         # Streamlit testing interface
├── test_hybrid_model.py            # FiQA benchmark evaluation script
├── train_modern_sentiment.py       # V3/V4 training script (QLoRA + SFTTrainer)
├── finbert_training.py             # V2 FinBERT fine-tuning script
├── financial_news_sentiment_analysis.ipynb  # V1 Bi-LSTM notebook
├── requirements.txt                # Python dependencies
├── benchmark_results.csv           # Detailed per-headline benchmark output
├── frontend/                       # Production web interface
│   ├── index.html                  # Glassmorphism landing page
│   ├── api.py                      # Local FastAPI server
│   └── images/                     # Generated AI assets
├── hf_backend/                     # Hugging Face Spaces deployment
│   ├── api.py                      # Cloud FastAPI server
│   ├── Dockerfile                  # Container configuration
│   ├── hybrid_routing_engine.py    # Engine copy for cloud
│   └── requirements.txt            # Cloud dependencies
└── llama3-financial-sentiment-lora/  # Trained LoRA adapter weights
    ├── adapter_config.json
    └── adapter_model.safetensors
```

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the Benchmark

```bash
python3 test_hybrid_model.py
```

### Run the API Backend (Local)

```bash
cd frontend
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Run the Streamlit Testing Interface

```bash
streamlit run main.py
```

---

## Tech Stack

`Python` · `PyTorch` · `Llama-3-8B` · `QLoRA` · `PEFT` · `trl` · `llama-cpp-python` · `GGUF` · `FastAPI` · `Hugging Face` · `Docker` · `Streamlit`

---

## License

This project is for educational and research purposes.
