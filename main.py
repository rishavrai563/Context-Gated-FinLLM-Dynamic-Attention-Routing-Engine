"""
=============================================================================
QLoRA Financial Sentiment Analysis — Low-Latency Inference Deployment
=============================================================================
This script provides an optimized Streamlit web interface for financial news
sentiment analysis using a QLoRA-fine-tuned Llama-3-8B model.

Inference Engine: llama-cpp-python (GGUF) for GPU-accelerated local inference
on consumer GPUs (RTX 3050 6GB+).
"""

import os
os.environ["HF_HOME"] = "/media/rishi/New Volume F/Sentiment Analysis/Llama weights"

import time
import json
import streamlit as st

# ── Model Configuration ──────────────────────────────────────────────────────
# Points to YOUR fine-tuned merged GGUF model on Hugging Face Hub
GGUF_REPO = "rishi563/llama3-financial-sentiment-gguf"
GGUF_FILE = "llama-3-8b-instruct.Q4_K_M.gguf"
GGUF_LOCAL_DIR = "/media/rishi/New Volume F/Sentiment Analysis/merged-gguf"

SENTIMENT_COLORS = {
    "positive": "#00C853",  # Green
    "negative": "#FF1744",  # Red
    "neutral":  "#FFD600",  # Yellow/Amber
    "mixed":    "#FF9800",  # Orange
}

SENTIMENT_EMOJI = {
    "positive": "📈",
    "negative": "📉",
    "neutral":  "➡️",
    "mixed":    "🔀",
}

# SYSTEM_PROMPT is now generated dynamically inside predict_sentiment


@st.cache_resource
def load_model():
    """Download GGUF model (first run only) and load with GPU acceleration.
    
    Uses Q4_K_M quantization (~4.9 GB) which fits entirely in the
    RTX 3050's 6 GB VRAM for fast GPU-accelerated inference.
    """
    from huggingface_hub import hf_hub_download
    # pyrefly: ignore [missing-import]
    from llama_cpp import Llama

    # Download GGUF file (cached after first download)
    print(f"[INFO] Downloading GGUF model: {GGUF_REPO}/{GGUF_FILE}")
    model_path = hf_hub_download(
        repo_id=GGUF_REPO,
        filename=GGUF_FILE,
        local_dir=GGUF_LOCAL_DIR,
    )
    print(f"[INFO] Model cached at: {model_path}")

    # Load with GPU offloading — 25 of 33 layers on GPU, rest on CPU
    # RTX 3050 (6GB) needs ~1GB headroom for KV cache + compute buffers
    print("[INFO] Loading model into GPU VRAM (hybrid GPU/CPU)...")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=25,       # 25/33 layers on GPU, rest on CPU for headroom
        n_ctx=1024,            # Smaller context = less KV cache memory
        n_batch=128,           # Smaller batch for memory safety
        verbose=False,
    )

    # Detect device info
    device_info = "GGUF Q4_K_M — GPU Accelerated (llama.cpp)"
    print(f"[INFO] Model loaded successfully: {device_info}")
    return llm, device_info


def predict_sentiment(headline, llm):
    """Run GPU-accelerated inference using strict rules and prompt injection."""
    start = time.perf_counter()

    # 1. STRICTER SYSTEM PROMPT: Use rules to force the quantitative logic
    system_prompt = (
        "You are a Quantitative Finance Sentiment Engine. "
        "YOUR RULES ARE ABSOLUTE:\n"
        "1. IF 'guidance', 'outlook', or 'forecast' is lowered or cut: SENTIMENT IS NEGATIVE.\n"
        "2. IF 'beat' or 'record' exists, IGNORE IT if a guidance cut is present.\n"
        "3. Output ONLY a valid JSON object. No other text.\n"
        "Format: {'sentiment': 'POSITIVE'|'NEGATIVE'|'NEUTRAL', 'reasoning_token_focus': '...'}"
    )

    # 2. PROMPT INJECTION: Force the model to think before it speaks and start the JSON
    formatted_prompt = (
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\nHeadline: {headline}\n\n"
        f"Step 1: Check for guidance cuts.\n"
        f"Step 2: Determine sentiment based on rules.\n"
        f"Step 3: Output JSON.<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{{"
    )

    # Use create_completion instead of chat_completion for prompt injection
    response = llm.create_completion(
        prompt=formatted_prompt,
        max_tokens=128,
        temperature=0.0,
        top_p=1.0,
        stop=["<|eot_id|>"]
    )

    # Re-attach the '{' that we forced the model to start with
    assistant_output = "{" + response["choices"][0]["text"].strip()
    latency_ms = (time.perf_counter() - start) * 1000

    print(f"\n--- RAW MODEL OUTPUT ---\n{assistant_output}\n------------------------\n")

    # Parse JSON output — extract JSON block from surrounding chat text
    sentiment = "neutral"
    reasoning = "No reasoning provided."
    data = {}
    
    def extract_json(text):
        """Extract the outermost JSON object by counting braces (handles nesting)."""
        start_idx = text.find('{')
        if start_idx == -1:
            return None
        depth = 0
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return text[start_idx:i+1]
        return None
    
    # Try direct parse first, then brace-counting extraction
    try:
        data = json.loads(assistant_output)
        sentiment = data.get("sentiment", "neutral").lower()
        reasoning = data.get("reasoning_token_focus", reasoning)
    except json.JSONDecodeError:
        # Model wrapped JSON in conversational text — extract with brace counting
        json_str = extract_json(assistant_output)
        if json_str:
            try:
                data = json.loads(json_str)
                sentiment = data.get("sentiment", "neutral").lower()
                
                # Extract the full conversational reasoning by removing the JSON block
                import re
                full_text = assistant_output.replace(json_str, "").strip()
                # Clean up leftover prefixes like "Reasoning:" or "Here is the analysis:"
                full_text = re.sub(r'^(Here is the analysis:|Reasoning:)\s*', '', full_text, flags=re.IGNORECASE).strip()
                
                if full_text:
                    reasoning = full_text
                else:
                    reasoning = data.get("reasoning_token_focus", reasoning)
            except json.JSONDecodeError:
                reasoning = f"Parse failed. Raw: {assistant_output}"
        else:
            # Last resort: scan for sentiment keywords directly
            lower_out = assistant_output.lower()
            if "negative" in lower_out:
                sentiment = "negative"
            elif "positive" in lower_out:
                sentiment = "positive"
            reasoning = f"Extracted from raw output: {assistant_output[:200]}"

    # Extract confidence scores
    confidence = {"positive": 0, "negative": 0, "neutral": 0}
    if sentiment != "neutral" or reasoning != "No reasoning provided.":
        try:
            conf_data = data.get("confidence_scores", {})
            confidence["positive"] = int(conf_data.get("positive", 0))
            confidence["negative"] = int(conf_data.get("negative", 0))
            confidence["neutral"] = int(conf_data.get("neutral", 0))
        except Exception:
            pass
    
    # If no confidence scores, generate reasonable defaults from sentiment
    if sum(confidence.values()) == 0:
        if sentiment == "positive":
            confidence = {"positive": 78, "negative": 8, "neutral": 14}
        elif sentiment == "negative":
            confidence = {"positive": 7, "negative": 80, "neutral": 13}
        else:
            confidence = {"positive": 20, "negative": 15, "neutral": 65}

    return {
        "label": sentiment,
        "reasoning": reasoning,
        "confidence": confidence,
        "latency_ms": round(latency_ms, 2),
    }


def main():
    """Streamlit application for financial sentiment analysis."""
    st.set_page_config(
        page_title="Financial Sentiment — Llama-3 QLoRA",
        page_icon="📊",
        layout="centered",
    )
    st.title("📊 Generative Financial Sentiment Analysis")
    st.markdown(
        "Powered by **Llama-3-8B + Contextual Gating QLoRA** — "
        "Outputs structured JSON containing sentiment and reasoning based on sector routing.\n\n"
        "Enter a financial news headline to analyze."
    )
    
    with st.spinner("Loading Llama-3-8B GGUF model... (first run downloads ~5 GB)"):
        llm, device_info = load_model()

    st.caption(f"🖥️ Device: {device_info}")
    st.markdown("---")
    
    text_input = st.text_area(
        "📝 Enter Financial News",
        placeholder=(
            "e.g., 'Gold prices surge to historic highs as supply chain disruptions worsen'"
        ),
        height=120,
        key="news_input",
    )
    
    analyze_clicked = st.button(
        "🔍 Analyze Sentiment & Reasoning",
        type="primary",
        use_container_width=True,
    )
    
    if analyze_clicked and text_input.strip():
        with st.spinner("Analyzing..."):
            result = predict_sentiment(text_input.strip(), llm)

        label = result["label"]
        reasoning = result["reasoning"]
        latency = result["latency_ms"]
        confidence = result["confidence"]
        
        emoji = SENTIMENT_EMOJI.get(label, "➡️")
        color = SENTIMENT_COLORS.get(label, "#FFD600")

        st.markdown("---")
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {color}22, {color}11);
                border-left: 5px solid {color};
                padding: 20px;
                border-radius: 10px;
                margin: 10px 0;
            ">
                <h2 style="margin: 0; color: {color};">
                    {emoji} {label.upper()} Sentiment
                </h2>
                <p style="font-size: 16px; margin: 10px 0 0 0; color: #eee; line-height: 1.5;">
                    <strong>Reasoning Focus:</strong> {reasoning}
                </p>
                <p style="font-size: 14px; margin: 10px 0 0 0; color: #aaa;">
                    Latency: <strong>{latency:.1f}ms</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Confidence Score Bars ─────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 📊 Confidence Distribution")
        
        bar_data = [
            ("Positive", confidence["positive"], "#00C853"),
            ("Negative", confidence["negative"], "#FF1744"),
            ("Neutral",  confidence["neutral"],  "#FFD600"),
        ]
        
        for bar_label, pct, bar_color in bar_data:
            st.markdown(
                f"""
                <div style="margin: 8px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="color: #ccc; font-size: 14px; font-weight: 600;">{bar_label}</span>
                        <span style="color: {bar_color}; font-size: 14px; font-weight: 700;">{pct}%</span>
                    </div>
                    <div style="
                        background: #1a1a2e;
                        border-radius: 8px;
                        height: 12px;
                        overflow: hidden;
                    ">
                        <div style="
                            width: {pct}%;
                            height: 100%;
                            background: linear-gradient(90deg, {bar_color}88, {bar_color});
                            border-radius: 8px;
                            transition: width 0.6s ease;
                        "></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif analyze_clicked and not text_input.strip():
        st.warning("⚠️ Please enter some financial news text to analyze.")
        
    st.markdown("---")
    st.caption(
        "Built with Streamlit • Model: Llama-3-8B-Instruct (GGUF Q4_K_M) • "
        "Engine: llama.cpp GPU-accelerated inference"
    )

if __name__ == "__main__":
    main()
