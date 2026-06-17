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
4. **V4 (Generative Instruction Tuning):** Transitioned to a Decoder LLM (`meta-llama/Meta-Llama-3-8B-Instruct`) using `trl.SFTTrainer`. Added a Global System Prompt for **Contextual Gating** (Sector Routing Rules) to eliminate Semantic Domain Inversion. The model natively outputs structured JSON schema `{"sentiment": "...", "reasoning_token_focus": "..."}`.
5. **V5 (GGUF Inference + Prompt Injection):** Merged LoRA adapters and exported to 4-bit `Q4_K_M` GGUF format. Replaced the memory-heavy `transformers` library with highly optimized `llama-cpp-python` to fit the 8B model within a consumer 6GB VRAM GPU (RTX 3050). Implemented manual Prompt Injection (Assistant pre-filling) and rigid quantitative rules to force the model into executing strict JSON structures over conversational summarization.

### Deployment
A low-latency Streamlit web application is provided for real-time inference. 
- **Hardware Profile:** Optimized for 6GB VRAM GPUs (e.g., NVIDIA RTX 3050).
- **Inference Engine:** `llama.cpp` using hybrid execution (`n_gpu_layers=25`) to prevent CUDA Out-Of-Memory errors while maintaining high throughput.
- **Frontend UI:** Built with Streamlit. Features dynamic sentiment extraction, a regex/brace-counting JSON fallback parser, and visual Confidence Distribution bars.
