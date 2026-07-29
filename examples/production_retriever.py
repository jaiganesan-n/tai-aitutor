"""The production retrieval chain (Section 7, stage for stage) + its evaluation.

Data downloads are explicit and yours to relocate (see quickstart.py).

Needs: pip install "tai-aitutor[gemini,rag,rerank,data]",
GOOGLE_API_KEY + COHERE_API_KEY in .env.
"""

import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

from tai_aitutor import (
    BM25Index,
    QADataset,
    answer,
    configure,
    evaluate_retrieval,
    get_all_chunks,
    get_collection,
    hybrid_search,
    ingest,
    load_csv,
    pack_context,
    rerank,
    search,
    setup_notebook,
    show_eval_table,
)

setup_notebook(required_keys=("GOOGLE_API_KEY", "COHERE_API_KEY"))
configure(provider="gemini")

# --- data (explicit downloads; host these wherever the course decides) ---
DATA_URL = "https://raw.githubusercontent.com/AlaFalaki/tutorial_notebooks/main/data/mini-llama-articles.csv"
csv_path = Path("mini-llama-articles.csv")
if not csv_path.exists():
    urllib.request.urlretrieve(DATA_URL, csv_path)
qa_path = hf_hub_download(
    repo_id="jaiganesan/ai_tutor_knowledge",
    filename="rag_eval_dataset_question_context_subset_50.json",
    repo_type="dataset",
)
qa = QADataset.load(qa_path)  # legacy-compatible JSON, loads unchanged

col = get_collection("mini_articles", path="./db")
if col.count() == 0:
    docs = load_csv(csv_path, text_col="content",
                    meta_cols=("title", "url", "source"), id_col="title")
    ingest(docs, col)

bm25 = BM25Index().build(get_all_chunks(col))  # build once; bm25.save()/load() to reuse


def production_retriever(query: str):
    hits = hybrid_search(query, col, bm25)          # dense 15 ∪ BM25 30 → RRF keep 30
    hits = rerank(query, hits)                      # Cohere v4-fast, top 5, floor 0.10
    return pack_context(hits, max_tokens=30_000)    # the cost knob


# The four-row ablation from the Hybrid Search lesson — same eval, four retrievers:
reports = {
    "dense only": evaluate_retrieval(qa, search_fn=lambda q, k: search(q, col, top_k=k)),
    "BM25 only": evaluate_retrieval(qa, search_fn=lambda q, k: bm25.search(q, top_k=k)),
    "fused (RRF)": evaluate_retrieval(qa, search_fn=lambda q, k: hybrid_search(q, col, bm25, keep=k)),
    "fused + rerank": evaluate_retrieval(qa, search_fn=lambda q, k: rerank(q, hybrid_search(q, col, bm25), top_n=k)),
}
show_eval_table(reports, extra_columns={
    "avg ctx tokens": {label: f"{r.avg_context_tokens(qa):.0f}" for label, r in reports.items()},
})

print(answer("How does hybrid search work?", retriever=production_retriever))
