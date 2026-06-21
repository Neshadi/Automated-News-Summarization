"""
NewsLens – FastAPI Backend
==========================
Connects the fine-tuned T5-small summarizer and Word2Vec semantic search
to the NewsLens demo frontend.

Usage:
    python app.py

Then open http://localhost:8000 in your browser.
The frontend is served automatically from NewsLens_Demo_UI.html.

Requirements:
    pip install fastapi uvicorn transformers torch gensim nltk scikit-learn numpy
"""

import os
import re
import time
import numpy as np
import nltk
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformers import T5Tokenizer, T5ForConditionalGeneration
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# ── Download NLTK data if needed ──────────────────────────────────────────────
nltk.download("punkt",     quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words("english"))

# ── Paths – adjust if your files are elsewhere ────────────────────────────────
MODEL_DIR   = "./t5-finetuned-news"   # saved by trainer.save_model()
W2V_PATH    = "./word2vec_bbc.model"  # saved separately (see note below)
CORPUS_PATH = "./bbc_articles.txt"    # one article per line (see note below)
FRONTEND    = "./NewsLens_Demo_UI.html"

# ── Global model holders (populated at startup) ───────────────────────────────
models: dict = {}


# ── Lifespan: load everything once at startup ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models …")

    # 1. T5 summarizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    t5_model  = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
    t5_model.eval()
    models["tokenizer"] = tokenizer
    models["t5"]        = t5_model
    models["device"]    = device
    print("  T5-small loaded.")

    # 2. Word2Vec + article corpus
    if Path(W2V_PATH).exists():
        w2v = Word2Vec.load(W2V_PATH)
        models["w2v"] = w2v
        print("  Word2Vec loaded.")
    else:
        print("  Word2Vec model not found – semantic search will be unavailable.")
        models["w2v"] = None

    if Path(CORPUS_PATH).exists():
        with open(CORPUS_PATH, encoding="utf-8") as f:
            articles = [line.strip() for line in f if line.strip()]
        models["articles"] = articles

        if models["w2v"]:
            vecs = np.array([_doc_vector(a, models["w2v"]) for a in articles])
            models["article_vecs"] = vecs
            print(f"  Article matrix: {vecs.shape}")
    else:
        print("  Corpus file not found – search will return placeholder results.")
        models["articles"]     = []
        models["article_vecs"] = None

    print("All models ready.\n")
    yield
    print("Shutting down.")


app = FastAPI(title="NewsLens API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────
class SummarizeRequest(BaseModel):
    article:  str
    strategy: str = "zero-shot"   # "zero-shot" | "few-shot" | "chain-of-thought"
    max_new_tokens: int = 80
    num_beams:      int = 4


class SearchRequest(BaseModel):
    query: str
    top_n: int = 5


# ── Helper: build prompt from strategy ───────────────────────────────────────
FEW_SHOT_EXAMPLE = (
    "Input: The UK economy showed strong growth in Q3 as businesses "
    "reopened after lockdown restrictions eased. "
    "Output: UK economy rebounds strongly in third quarter. "
)

def build_prompt(article: str, strategy: str) -> str:
    text = article[:800]  # truncation guard before tokenizer
    if strategy == "few-shot":
        return f"summarize: {FEW_SHOT_EXAMPLE}Input: {text} Output:"
    if strategy == "chain-of-thought":
        return f"summarize: key points: {text} summary:"
    return f"summarize: {text}"


def clean_output(text: str) -> str:
    """Strip leaked prompt fragments from T5 output."""
    text = text.strip()
    for prefix in ["Input:", "Output:", "Summary:", "key points:", "Now:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    words = text.split()
    if len(words) > 120:
        text = " ".join(words[:80])
    return text


# ── Helper: Word2Vec document vector ─────────────────────────────────────────
def _preprocess(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    return [w for w in tokens if w not in STOP_WORDS]


def _doc_vector(text: str, w2v: Word2Vec) -> np.ndarray:
    tokens = _preprocess(text)
    vecs   = [w2v.wv[w] for w in tokens if w in w2v.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(w2v.vector_size)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve the NewsLens demo UI."""
    html_path = Path(FRONTEND)
    if not html_path.exists():
        raise HTTPException(status_code=404, detail=f"Frontend not found at {FRONTEND}")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {
        "status":  "ok",
        "t5":      "loaded" if "t5" in models else "missing",
        "w2v":     "loaded" if models.get("w2v") else "missing",
        "device":  models.get("device", "unknown"),
        "articles": len(models.get("articles", [])),
    }


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    """
    Generate an abstractive summary using the fine-tuned T5-small model.

    Body:
        article         – raw news article text
        strategy        – "zero-shot" | "few-shot" | "chain-of-thought"
        max_new_tokens  – maximum tokens in the output (default 80)
        num_beams       – beam search width (default 4)

    Returns:
        summary, strategy, word counts, compression ratio, latency_ms
    """
    if "t5" not in models:
        raise HTTPException(status_code=503, detail="T5 model not loaded.")

    article = req.article.strip()
    if not article:
        raise HTTPException(status_code=400, detail="Article text is empty.")

    tokenizer = models["tokenizer"]
    t5        = models["t5"]
    device    = models["device"]

    prompt  = build_prompt(article, req.strategy)
    inputs  = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = t5.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            num_beams=req.num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    latency_ms = round((time.perf_counter() - t0) * 1000)

    raw_summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    summary     = clean_output(raw_summary)

    input_words  = len(article.split())
    output_words = len(summary.split())
    ratio        = round(output_words / max(input_words, 1) * 100)

    return {
        "summary":      summary,
        "strategy":     req.strategy,
        "input_words":  input_words,
        "output_words": output_words,
        "compression":  ratio,
        "latency_ms":   latency_ms,
    }


@app.post("/search")
def semantic_search(req: SearchRequest):
    """
    Find semantically similar BBC articles using Word2Vec cosine similarity.

    Body:
        query  – free-text search query
        top_n  – number of results to return (default 5)

    Returns:
        list of { rank, similarity, article_snippet }
    """
    w2v      = models.get("w2v")
    articles = models.get("articles", [])
    vecs     = models.get("article_vecs")

    if w2v is None or vecs is None or not articles:
        raise HTTPException(
            status_code=503,
            detail="Word2Vec model or corpus not available. "
                   "Ensure word2vec_bbc.model and bbc_articles.txt exist.",
        )

    query_vec = _doc_vector(req.query, w2v).reshape(1, -1)
    if np.all(query_vec == 0):
        raise HTTPException(status_code=400, detail="No known words in query.")

    sims    = cosine_similarity(query_vec, vecs)[0]
    top_idx = np.argsort(sims)[::-1][: req.top_n]

    results = []
    for rank, idx in enumerate(top_idx, 1):
        results.append({
            "rank":       rank,
            "similarity": round(float(sims[idx]), 4),
            "snippet":    articles[idx][:200],
        })

    return {"query": req.query, "results": results}


@app.get("/word-relations")
def word_relations():
    """
    Return nearest Word2Vec neighbours for a fixed set of news-domain words.
    """
    w2v = models.get("w2v")
    if w2v is None:
        raise HTTPException(status_code=503, detail="Word2Vec model not loaded.")

    words = ["government", "economy", "celebrity", "film", "crisis", "award"]
    out   = []
    for word in words:
        if word in w2v.wv:
            neighbours = [
                {"word": w, "similarity": round(float(s), 3)}
                for w, s in w2v.wv.most_similar(word, topn=3)
            ]
        else:
            neighbours = []
        out.append({"word": word, "neighbours": neighbours})

    # Analogy: economy + growth - crisis
    try:
        analogy = [
            {"word": w, "similarity": round(float(s), 3)}
            for w, s in w2v.wv.most_similar(
                positive=["economy", "growth"], negative=["crisis"], topn=3
            )
        ]
    except Exception:
        analogy = []

    return {"relations": out, "analogy": analogy}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
