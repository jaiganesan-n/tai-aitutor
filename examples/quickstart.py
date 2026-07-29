"""The whole course pipeline, end to end.

Data policy: DOWNLOADS LIVE IN YOUR NOTEBOOK/SCRIPT, not in the package —
you decide where datasets are hosted and cached; the package only loads
whatever file you hand it (`load_csv` / `load_jsonl` / `load_directory`).

Needs: pip install "tai-aitutor[gemini,rag]" and GOOGLE_API_KEY in .env.
"""

import urllib.request
from pathlib import Path

from tai_aitutor import (
    answer,
    configure,
    get_collection,
    ingest,
    load_csv,
    setup_notebook,
    show_answer,
)

setup_notebook(required_keys=("GOOGLE_API_KEY",))
configure(provider="gemini")

# 1. Download the course dataset (explicit — swap this URL when the data moves)
DATA_URL = "https://raw.githubusercontent.com/AlaFalaki/tutorial_notebooks/main/data/mini-llama-articles.csv"
csv_path = Path("mini-llama-articles.csv")
if not csv_path.exists():
    urllib.request.urlretrieve(DATA_URL, csv_path)

# 2. Load → ingest → answer
docs = load_csv(csv_path, text_col="content", meta_cols=("title", "url", "source"),
                id_col="title")
col = get_collection("mini_articles", path="./db")
if col.count() == 0:
    print(ingest(docs, col, chunk_size=512, chunk_overlap=128))

show_answer(answer("What is the difference between RAG and fine-tuning?", col, top_k=5))
