"""Quickstart: load a CSV, index it, retrieve, and answer — end to end.

Every step is a call you can read. The package ships no dataset URLs, so the
download stays here in the example (and, in the course, in the notebook).

Run:
    pip install "tai-aitutor[gemini,rag]"
    python examples/quickstart.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from tai_aitutor import (
    build_rag_prompt,
    chunk_document,
    configure,
    embed,
    generate,
    get_collection,
    load_csv,
    search,
    show_answer,
)

CSV_URL = (
    "https://raw.githubusercontent.com/AlaFalaki/tutorial_notebooks/main/data/"
    "mini-llama-articles.csv"
)
CSV_PATH = Path("mini-llama-articles.csv")

# 1. Configure the provider
configure(provider="gemini")

# 2. Download the data (the notebook does this too — the package never fetches)
if not CSV_PATH.exists():
    urllib.request.urlretrieve(CSV_URL, CSV_PATH)  # noqa: S310

docs = load_csv(
    CSV_PATH, text_col="content", meta_cols=("title", "url", "source"), id_col="title"
)

# 3. chunk -> embed -> upsert, the three visible steps
col = get_collection("quickstart", path="./quickstart-db")
chunks = [c for d in docs for c in chunk_document(d, chunk_size=512, chunk_overlap=128)]

BATCH_SIZE = 50
for start in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[start : start + BATCH_SIZE]
    col.add(
        ids=[c.id for c in batch],
        documents=[c.text for c in batch],
        embeddings=embed([c.text for c in batch], task="document"),
        metadatas=[c.metadata for c in batch],
    )
print(f"indexed {len(chunks)} chunks")

# 4. retrieve -> prompt -> generate
question = "What is the difference between RAG and fine-tuning?"
hits = search(question, col, top_k=5)
reply = generate(build_rag_prompt(question, hits))
show_answer(reply, hits)
