"""
=============================================================================
QLoRA Financial Sentiment Analysis — Low-Latency Inference Deployment
=============================================================================
This script provides an optimized Streamlit web interface for financial news
sentiment analysis using a QLoRA-fine-tuned Llama-3-8B model.
"""

import time
import json
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_PATH = "./llama3-financial-sentiment-lora"

SENTIMENT_COLORS = {
    "positive": "#00C853",  # Green
    "negative": "#FF1744",  # Red
    "neutral":  "#FFD600",  # Yellow/Amber
}

SENTIMENT_EMOJI = {
    "positive": "📈",
    "negative": "📉",
    "neutral":  "➡️",
}

@st.cache_resource
def load_model_and_tokenizer():
    """Load the base model + LoRA adapters for inference."""
    import os
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
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

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
        )
        device_info = f"GPU ({torch.cuda.get_device_name(0)}) — 4-bit NF4"

    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
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
            device_info += " (no adapters — using base model)"
            print(f"[WARNING] Could not load adapters: {e}")
    else:
        device_info += " (base model only — no fine-tuned adapters)"

    model.eval()
    return tokenizer, model, device_info

def predict_sentiment(text, tokenizer, model):
    """Run optimized inference on a single financial news input."""
    start = time.perf_counter()
    device = next(model.parameters()).device

    # Reconstruct system prompt matching training
    system_prompt = (
        "You are an expert Financial AI. You analyze financial news headlines and extract the sentiment (negative, neutral, or positive) and the reasoning.\n"
        "Sector Routing Rules:\n"
        "- Commodities: Focus on supply chains, raw material prices, and weather.\n"
        "- Macro: Focus on inflation, interest rates, and GDP.\n"
        "- Equities: Focus on earnings, M&A, and executive changes.\n"
        "Output strict JSON containing {\"sentiment\": \"<sentiment>\", \"reasoning_token_focus\": \"<reasoning>\"}."
    )

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\nAnalyze this headline: {text}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

    # Decode generation
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    assistant_output = generated_text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].replace("<|eot_id|>", "").strip()

    latency_ms = (time.perf_counter() - start) * 1000

    # Parse JSON output
    try:
        data = json.loads(assistant_output)
        sentiment = data.get("sentiment", "neutral").lower()
        reasoning = data.get("reasoning_token_focus", "No reasoning provided.")
    except Exception:
        sentiment = "neutral"
        reasoning = f"Raw output parsing failed. Raw response: {assistant_output}"

    return {
        "label": sentiment,
        "reasoning": reasoning,
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
    
    with st.spinner("Loading Llama-3-8B model... (this can take a moment)"):
        tokenizer, model, device_info = load_model_and_tokenizer()

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
        result = predict_sentiment(text_input.strip(), tokenizer, model)

        label = result["label"]
        reasoning = result["reasoning"]
        latency = result["latency_ms"]
        
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

    elif analyze_clicked and not text_input.strip():
        st.warning("⚠️ Please enter some financial news text to analyze.")
        
    st.markdown("---")
    st.caption(
        "Built with Streamlit • Model: Llama-3-8B-Instruct + QLoRA • "
        "Quantization: 4-bit NF4 via bitsandbytes"
    )

if __name__ == "__main__":
    main()
