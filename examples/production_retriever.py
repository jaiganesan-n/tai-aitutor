"""The production retrieval chain (Section 7, stage for stage) + its evaluation.

Needs: pip install "tai-aitutor[gemini,rag,rerank,data]",
GOOGLE_API_KEY + COHERE_API_KEY in .env.
"""

from tai_aitutor import (
    BM25Index,
    answer,
    configure,
    evaluate_retrieval,
    get_all_chunks,
    get_collection,
    hybrid_search,
    ingest,
    mini_articles,
    pack_context,
    qa_dataset,
    rerank,
    search,
    setup_notebook,
    show_eval_table,
)

setup_notebook(required_keys=("GOOGLE_API_KEY", "COHERE_API_KEY"))
configure(provider="gemini")

col = get_collection("mini_articles", path="./db")
if col.count() == 0:
    ingest(mini_articles(), col)

bm25 = BM25Index().build(get_all_chunks(col))  # build once; bm25.save()/load() to reuse


def production_retriever(query: str):
    hits = hybrid_search(query, col, bm25)          # dense 15 ∪ BM25 30 → RRF keep 30
    hits = rerank(query, hits)                      # Cohere v4-fast, top 5, floor 0.10
    return pack_context(hits, max_tokens=30_000)    # the cost knob


# The four-row ablation from the Hybrid Search lesson — same eval, four retrievers:
qa = qa_dataset("rag_eval_50")
show_eval_table(
    {
        "dense only": evaluate_retrieval(qa, search_fn=lambda q, k: search(q, col, top_k=k)),
        "BM25 only": evaluate_retrieval(qa, search_fn=lambda q, k: bm25.search(q, top_k=k)),
        "fused (RRF)": evaluate_retrieval(qa, search_fn=lambda q, k: hybrid_search(q, col, bm25, keep=k)),
        "fused + rerank": evaluate_retrieval(qa, search_fn=lambda q, k: rerank(q, hybrid_search(q, col, bm25), top_n=k)),
    }
)

print(answer("How does hybrid search work?", retriever=production_retriever))
