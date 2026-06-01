# Context-Gated FinLLM Dynamic Attention Routing Engine

![sentiment_gif_v02b](https://user-images.githubusercontent.com/88341388/232018228-e8e76fff-1b4b-4dc8-b7f5-26e6d74b29e2.gif)

### Background: 
The financial industry generates vast amounts of news and data every day, which can significantly impact market sentiment.

### Objective: 
Develop a highly accurate, State-of-the-Art (SOTA) financial news sentiment analysis system utilizing Parameter-Efficient Fine-Tuning.

### Evolution of Architectures: 
1. **V1 (Bi-LSTM):** Initial legacy implementation using Keras.
2. **V2 (FinBERT):** Migrated to a domain-specific Transformer (ProsusAI/finbert) via full fine-tuning.
3. **V3 (ModernBERT + QLoRA):** Implemented Parameter-Efficient Fine-Tuning (PEFT) on a 2024 SOTA encoder (`answerdotai/ModernBERT-base`). Uses 4-bit NF4 quantization to dramatically reduce VRAM usage while maintaining full fine-tuning quality.

### Deployment:
A low-latency Streamlit web application is provided for real-time inference using dynamically merged LoRA adapters.
