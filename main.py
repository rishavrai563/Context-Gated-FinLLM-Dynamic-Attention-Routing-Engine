

import os
os.environ["HF_HOME"] = "/media/rishi/New Volume F/Sentiment Analysis/Llama weights"

import time
import json
import streamlit as st
from hybrid_routing_engine import HybridSentimentEngine

# Model Configuration
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



def main():
    """Streamlit application for financial sentiment analysis."""
    st.set_page_config(
        page_title="Financial Sentiment — Quant AI",
        page_icon="⚡",
        layout="centered",
    )
    
    # Custom CSS for Premium UI
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        /* Base Theme */
        .stApp {
            background-color: #0B0F19;
            font-family: 'Inter', sans-serif;
        }
        
        /* Hero Section */
        .hero-container {
            text-align: center;
            padding: 4rem 2rem;
            background: radial-gradient(100% 100% at 50% 0%, rgba(16, 185, 129, 0.1) 0%, rgba(11, 15, 25, 0) 100%);
            border-radius: 2rem;
            margin-bottom: 3rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            position: relative;
            overflow: hidden;
        }
        
        .hero-badge {
            display: inline-block;
            padding: 0.5rem 1.25rem;
            background: #111827;
            color: #10B981;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(16, 185, 129, 0.2);
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.1);
        }
        
        .hero-title {
            font-size: 3.5rem;
            font-weight: 800;
            color: #F3F4F6;
            margin-bottom: 1rem;
            line-height: 1.1;
            letter-spacing: -0.02em;
        }
        
        .hero-title span {
            background: linear-gradient(135deg, #10B981 0%, #059669 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .hero-subtitle {
            font-size: 1.15rem;
            color: #9CA3AF;
            max-width: 600px;
            margin: 0 auto;
            line-height: 1.6;
        }
        
        /* Input & Button Styling */
        .stTextArea textarea {
            background-color: #111827 !important;
            border: 1px solid #1F2937 !important;
            color: #F3F4F6 !important;
            border-radius: 1rem !important;
            padding: 1.25rem !important;
            font-size: 1.1rem !important;
            transition: all 0.3s ease;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1) !important;
        }
        
        .stTextArea textarea:focus {
            border-color: #10B981 !important;
            box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2) !important;
        }
        
        .stButton button {
            background: linear-gradient(135deg, #111827 0%, #1F2937 100%) !important;
            color: white !important;
            border: 1px solid #374151 !important;
            border-radius: 9999px !important;
            padding: 0.75rem 2rem !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
            margin-top: 1rem;
        }
        
        .stButton button:hover {
            border-color: #10B981 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 20px -5px rgba(16, 185, 129, 0.2) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Render Hero Section
    st.markdown("""
        <div class="hero-container">
            <div class="hero-badge">● LLAMA-3 8B LIVE</div>
            <div class="hero-title">Analyze Markets.<br><span>With AI Precision.</span></div>
            <div class="hero-subtitle">This is a quantitative finance engine. Input a headline, instantly detect market sentiment, and secure your trades with absolute certainty.</div>
        </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Loading AI Engine..."):
        llm, device_info = load_model()
        router = HybridSentimentEngine(llm)

    st.caption(f"🖥️ Device: {device_info} • Hybrid Routing: Fast-Path + Heavy LLM")
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
        with st.spinner("Routing headline through Hybrid Engine..."):
            result = router.analyze(text_input.strip())

        label = result["label"]
        reasoning = result["reasoning"]
        latency = result["latency_ms"]
        confidence = result["confidence"]
        engine_used = result.get("engine_used", "Unknown")
        
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
                <div style="display: flex; justify-content: space-between; margin-top: 12px;">
                    <span style="font-size: 13px; color: #aaa;">Engine: <strong style="color: #10B981;">{engine_used}</strong></span>
                    <span style="font-size: 13px; color: #aaa;">Latency: <strong>{latency:.1f}ms</strong></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Confidence Score Bars
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
        "Engine: Hybrid Routing (Fast-Path + llama.cpp GPU-accelerated inference)"
    )

if __name__ == "__main__":
    main()
