import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add backend logic to path
sys.path.append("/media/rishi/New Volume F/Sentiment Analysis/Finanical_news_Sentiment_analysis")
from hybrid_routing_engine import HybridSentimentEngine

os.environ["HF_HOME"] = "/media/rishi/New Volume F/Sentiment Analysis/Llama weights"
GGUF_REPO = "rishi563/llama3-financial-sentiment-gguf"
GGUF_FILE = "llama-3-8b-instruct.Q4_K_M.gguf"
GGUF_LOCAL_DIR = "/media/rishi/New Volume F/Sentiment Analysis/merged-gguf"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None

@app.on_event("startup")
def load_model():
    global engine
    print("[INFO] Starting FastAPI server...")
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    
    print(f"[INFO] Locating/Downloading GGUF model: {GGUF_REPO}/{GGUF_FILE}")
    model_path = hf_hub_download(
        repo_id=GGUF_REPO,
        filename=GGUF_FILE,
        local_dir=GGUF_LOCAL_DIR,
    )
    
    print("[INFO] Loading model into GPU VRAM (hybrid GPU/CPU)...")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=25,
        n_ctx=1024,
        n_batch=128,
        verbose=False,
    )
    
    engine = HybridSentimentEngine(llm)
    print("[INFO] API Ready!")

class HeadlineRequest(BaseModel):
    headline: str

@app.post("/analyze")
def analyze_headline(request: HeadlineRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Model still loading")
    
    result = engine.analyze(request.headline)
    return {
        "label": result["label"],
        "reasoning": result["reasoning"],
        "confidence": result["confidence"],
        "engine": result["engine_used"],
        "latency": result["latency_ms"]
    }
