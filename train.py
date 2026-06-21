"""
NewsLens – Training / Artifact Builder
======================================
Mirrors NewsContentSummarization.ipynb but runs locally and SAVES every file
that app.py needs:

    ./t5-finetuned-news/   - fine-tuned T5-small summarizer (+ tokenizer)
    ./word2vec_bbc.model   - Word2Vec model for semantic search
    ./bbc_articles.txt     - one article per line (search corpus)

Usage:
    python train.py

Speed knobs (env vars, optional):
    NEWS_SAMPLES=800   number of articles to use   (default 800)
    NEWS_EPOCHS=1      training epochs             (default 1)
    NEWS_MAXIN=256     max input tokens            (default 256)
    NEWS_MAXOUT=64     max target tokens           (default 64)

On a CPU the defaults finish in roughly 10-20 minutes. Increase NEWS_SAMPLES
and NEWS_EPOCHS (e.g. 2200 / 3) to match the notebook exactly if you have time
or a GPU.
"""

import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import numpy as np
import pandas as pd
import torch
import nltk

from datasets import load_dataset, Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from gensim.models import Word2Vec

# ── Config (overridable via env vars) ─────────────────────────────────────────
MODEL_NAME  = "t5-small"
OUTPUT_DIR  = "./t5-finetuned-news"
W2V_PATH    = "./word2vec_bbc.model"
CORPUS_PATH = "./bbc_articles.txt"

SAMPLES  = int(os.environ.get("NEWS_SAMPLES", 800))
EPOCHS   = int(os.environ.get("NEWS_EPOCHS", 1))
MAX_IN   = int(os.environ.get("NEWS_MAXIN", 256))
MAX_OUT  = int(os.environ.get("NEWS_MAXOUT", 64))

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words("english"))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: samples={SAMPLES} epochs={EPOCHS} max_in={MAX_IN} max_out={MAX_OUT}")

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    print("\nLoading BBC News Summary dataset …")
    dataset = load_dataset("gopalkalpande/bbc-news-summary", split="train")
    df = pd.DataFrame(dataset)
    df = df.rename(columns={"Articles": "Review", "Summaries": "summary"}).dropna()

    n = min(SAMPLES, len(df))
    df = df.sample(n, random_state=42).reset_index(drop=True)
    print(f"Using {len(df)} articles.")

    # ── 2. Train / val split ──────────────────────────────────────────────────
    split = int(0.9 * len(df))
    train_df = df.iloc[:split].reset_index(drop=True)

    train_dataset = Dataset.from_pandas(train_df)

    # ── 3. Tokenizer & model ──────────────────────────────────────────────────
    model_exists = os.path.exists(os.path.join(OUTPUT_DIR, "model.safetensors"))
    if model_exists:
        print(f"\nFine-tuned model already exists at {OUTPUT_DIR} - skipping training.")
        tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        _build_search_artifacts(df)
        print("\nAll artifacts ready. You can now run:  python app.py")
        return

    print(f"\nLoading base model: {MODEL_NAME}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

    def tokenize(batch):
        inputs = ["summarize: " + a for a in batch["Review"]]
        model_inputs = tokenizer(inputs, max_length=MAX_IN, truncation=True, padding=False)
        labels = tokenizer(text_target=batch["summary"], max_length=MAX_OUT,
                           truncation=True, padding=False)
        model_inputs["labels"] = [
            [tok if tok != tokenizer.pad_token_id else -100 for tok in label]
            for label in labels["input_ids"]
        ]
        return model_inputs

    tokenized_train = train_dataset.map(
        tokenize, batched=True, remove_columns=train_dataset.column_names
    )
    tokenized_train.set_format("torch")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=8,
        warmup_steps=50,
        weight_decay=0.01,
        learning_rate=3e-4,
        save_strategy="no",
        logging_steps=25,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
    )

    print("\nFine-tuning …")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved fine-tuned model -> {OUTPUT_DIR}")

    # ── 5. Word2Vec + corpus (for semantic search) ────────────────────────────
    _build_search_artifacts(df)

    print("\nAll artifacts ready. You can now run:  python app.py")


def _build_search_artifacts(df):
    """Train + save Word2Vec and write the one-article-per-line corpus."""
    print("\nTraining Word2Vec for semantic search ...")

    def preprocess(text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        tokens = word_tokenize(text)
        return [w for w in tokens if w not in STOP_WORDS]

    sentences = df["Review"].apply(preprocess).tolist()
    w2v = Word2Vec(sentences, vector_size=100, window=5,
                   min_count=1, workers=4, seed=42)
    w2v.save(W2V_PATH)
    print(f"Saved Word2Vec -> {W2V_PATH} (vocab={len(w2v.wv)})")

    # one article per line; flatten internal newlines so app.py reads correctly
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for article in df["Review"].tolist():
            f.write(re.sub(r"\s+", " ", str(article)).strip() + "\n")
    print(f"Saved corpus -> {CORPUS_PATH} ({len(df)} articles)")


if __name__ == "__main__":
    main()
