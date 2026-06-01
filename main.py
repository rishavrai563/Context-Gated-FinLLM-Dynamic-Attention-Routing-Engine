"""
=============================================================================
QLoRA Financial Sentiment Analysis — Low-Latency Inference Deployment
=============================================================================
This script provides an optimized Streamlit web interface for financial news
sentiment analysis using a QLoRA-fine-tuned ModernBERT model.
"""

import time
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
BASE_MODEL_NAME = "answerdotai/ModernBERT-base"
# Set to None to use the base model without fine-tuned adapters
# (useful for zero-shot inference or when adapters haven't been trained yet)
ADAPTER_PATH = "./modernbert-financial-sentiment-lora"
NUM_LABELS = 3
ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}
MAX_SEQ_LENGTH = 128  # Match training config for consistency
SENTIMENT_COLORS = {
    "Positive": "#00C853",  # Green
    "Negative": "#FF1744",  # Red
    "Neutral":  "#FFD600",  # Yellow/Amber
}
SENTIMENT_EMOJI = {
    "Positive": "📈",
    "Negative": "📉",
    "Neutral":  "➡️",
}
@st.cache_resource
def load_model_and_tokenizer():
    """Load the base model + LoRA adapters for inference."""
    import os
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # Try loading from adapter path first (has tokenizer config from training)
    # Fall back to base model if adapter path doesn't exist
    tokenizer_path = ADAPTER_PATH if (
        ADAPTER_PATH and os.path.isdir(ADAPTER_PATH)
    ) else BASE_MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if use_cuda:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            num_labels=NUM_LABELS,
            device_map="auto",
        )
        device_info = f"GPU ({torch.cuda.get_device_name(0)}) — 4-bit NF4"

    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME,
            num_labels=NUM_LABELS,
        )
        model = model.to(device)
        device_info = "CPU — FP32"
    if ADAPTER_PATH and os.path.isdir(ADAPTER_PATH):
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, ADAPTER_PATH)
            model = model.merge_and_unload()
            device_info += " + LoRA (merged)"

        except Exception as e:
            # If adapter loading fails, continue with base model
            device_info += " (no adapters — using base model)"
            print(f"[WARNING] Could not load adapters: {e}")
    else:
        device_info += " (base model only — no fine-tuned adapters)"
    model.eval()

    return tokenizer, model, device_info
def predict_sentiment(text, tokenizer, model):
    """Run optimized inference on a single financial news input."""
    start = time.perf_counter()
    # Determine the device the model is on
    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )
    # Move input tensors to the model's device (GPU or CPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    # outputs.logits: raw unnormalized scores, shape [1, NUM_LABELS]
    # softmax normalizes to valid probabilities that sum to 1.0
    probs = F.softmax(outputs.logits, dim=-1)
    predicted_id = torch.argmax(probs, dim=-1).item()
    predicted_label = ID2LABEL[predicted_id]
    confidence = probs[0][predicted_id].item()

    # Build full probability dict
    prob_dict = {
        ID2LABEL[i]: round(probs[0][i].item(), 4)
        for i in range(NUM_LABELS)
    }

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "label": predicted_label,
        "confidence": confidence,
        "probabilities": prob_dict,
        "latency_ms": round(latency_ms, 2),
    }
def main():
    """Streamlit application for financial sentiment analysis."""
    st.set_page_config(
        page_title="Financial Sentiment — QLoRA",
        page_icon="📊",
        layout="centered",
    )
    st.title("📊 Financial News Sentiment Analysis")
    st.markdown(
        "Powered by **ModernBERT + QLoRA** — "
        "Parameter-Efficient Fine-Tuning with 4-bit quantization.\n\n"
        "Enter a financial news headline to analyze its sentiment."
    )
    with st.spinner("Loading model... (first time only)"):
        tokenizer, model, device_info = load_model_and_tokenizer()

    # Display device/model info
    st.caption(f"🖥️ Device: {device_info}")
    st.markdown("---")
    text_input = st.text_area(
        "📝 Enter Financial News",
        placeholder=(
            "e.g., 'Federal Reserve signals potential rate cut amid "
            "cooling inflation data'"
        ),
        height=120,
        key="news_input",
    )
    analyze_clicked = st.button(
        "🔍 Analyze Sentiment",
        type="primary",
        use_container_width=True,
    )
    if analyze_clicked and text_input.strip():
        result = predict_sentiment(text_input.strip(), tokenizer, model)

        label = result["label"]
        confidence = result["confidence"]
        probs = result["probabilities"]
        latency = result["latency_ms"]
        emoji = SENTIMENT_EMOJI[label]
        color = SENTIMENT_COLORS[label]

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
                    {emoji} {label} Sentiment
                </h2>
                <p style="font-size: 18px; margin: 5px 0 0 0; color: #ccc;">
                    Confidence: <strong>{confidence:.1%}</strong>
                    &nbsp;|&nbsp;
                    Latency: <strong>{latency:.1f}ms</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.subheader("Probability Distribution")
        for sent_name in ["Positive", "Neutral", "Negative"]:
            prob = probs[sent_name]
            st.markdown(
                f"**{SENTIMENT_EMOJI[sent_name]} {sent_name}**: {prob:.2%}"
            )
            st.progress(prob)
        with st.expander("🔢 Raw Model Output"):
            st.json(result)
        with st.expander("⚡ Performance Details"):
            st.markdown(f"""
            **Inference Latency**: {latency:.1f}ms

            **Memory & Latency Tradeoffs**:

            | Precision | Memory | Latency | Use Case |
            |-----------|--------|---------|----------|
            | FP32 | ~600 MB | ~50ms (CPU) | CPU deployment |
            | FP16 | ~300 MB | ~8ms (GPU) | Best GPU speed |
            | 4-bit NF4 | ~100 MB | ~12ms (GPU) | Lowest memory |

            **Key insight**: 4-bit quantization is a *memory* optimization,
            not a *speed* optimization. On GPU, FP16 is actually faster
            because it avoids the dequantization overhead. Use 4-bit when
            VRAM is limited (e.g., T4 GPU on free-tier Spaces).
            """)

    elif analyze_clicked and not text_input.strip():
        st.warning("⚠️ Please enter some financial news text to analyze.")
    st.markdown("---")
    st.caption(
        "Built with Streamlit • Model: ModernBERT-base + QLoRA • "
        "Quantization: 4-bit NF4 via bitsandbytes"
    )
# Entry Point
if __name__ == "__main__":
    main()
