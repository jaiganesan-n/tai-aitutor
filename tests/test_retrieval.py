"""retrieval.search unit tests against a fake collection (chroma-shaped)."""

from __future__ import annotations

import tai_aitutor as tai


class FakeCollection:
    def __init__(self):
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        n = kwargs["n_results"]
        ids = [f"c{i}" for i in range(min(n, 3))]
        return {
            "ids": [ids],
            "documents": [[f"text {i}" for i in range(len(ids))]],
            "metadatas": [[{"source": "tai_blog"} for _ in ids]],
            "distances": [[0.1 * (i + 1) for i in range(len(ids))]],
        }


def fake_embed(texts, task="document"):
    assert task == "query"
    return [0.5, 0.5]


def test_search_maps_chroma_result():
    col = FakeCollection()
    hits = tai.search("q", col, top_k=3, embed_fn=fake_embed)
    assert [h.id for h in hits] == ["c0", "c1", "c2"]
    assert [h.rank for h in hits] == [1, 2, 3]
    assert abs(hits[0].score - 0.9) < 1e-9  # 1 - distance
    assert hits[0].metadata == {"source": "tai_blog"}
    assert col.calls[0]["query_embeddings"] == [[0.5, 0.5]]
    assert "where" not in col.calls[0]


def test_search_passes_filters_through():
    col = FakeCollection()
    tai.search("q", col, top_k=2, where={"source": {"$eq": "x"}},
               where_document={"$contains": "needle"}, embed_fn=fake_embed)
    call = col.calls[0]
    assert call["where"] == {"source": {"$eq": "x"}}
    assert call["where_document"] == {"$contains": "needle"}
    assert call["n_results"] == 2


def test_search_accepts_list_returning_embed_fn():
    col = FakeCollection()
    tai.search("q", col, embed_fn=lambda texts, task="query": [[0.1, 0.2]])
    assert col.calls[0]["query_embeddings"] == [[0.1, 0.2]]
