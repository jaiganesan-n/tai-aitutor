"""The production retrieval chain, assembled from the parts the course teaches.

dense ∪ BM25 -> Reciprocal Rank Fusion -> Cohere rerank. The package ships each
stage separately and no composite that hides them, so the chain you run is the
chain you can read.

Run:
    pip install "tai-aitutor[gemini,rag,rerank]"
    python examples/production_retriever.py
"""

from __future__ import annotations

from tai_aitutor import (
    BM25Index,
    QADataset,
    configure,
    evaluate_retrieval,
    get_all_chunks,
    get_collection,
    rerank,
    rrf_fuse,
    search,
    show_eval_table,
)

configure(provider="gemini")

col = get_collection("ai_tutor_knowledge", path="./tai-knowledge-chroma")
bm25 = BM25Index().build(get_all_chunks(col))


def fused(query: str, top_k: int = 30):
    """Dense 15 ∪ BM25 30 -> RRF k=60, keeping top_k — production's constants."""
    return rrf_fuse(search(query, col, top_k=15), bm25.search(query, top_k=30), keep=top_k)


def production_retriever(query: str, top_k: int = 5):
    """The full chain: fuse a wide candidate set, then rerank it down."""
    return rerank(query, fused(query, top_k=30), top_n=top_k)


# The ablation the Hybrid Search and Re-Ranking lessons measure.
qa = QADataset.load("rag_eval_dataset.json")
reports = {
    "dense only": evaluate_retrieval(qa, search_fn=lambda q, k: search(q, col, top_k=k)),
    "BM25 only": evaluate_retrieval(qa, search_fn=lambda q, k: bm25.search(q, top_k=k)),
    "fused (RRF)": evaluate_retrieval(qa, search_fn=fused),
    "fused + rerank": evaluate_retrieval(qa, search_fn=production_retriever),
}
show_eval_table(reports)
