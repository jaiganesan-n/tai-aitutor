"""vectorstore + retrieval integration tests on a real (ephemeral) Chroma.

Embeddings are faked with a deterministic keyword-count vectorizer so
similarity behaves predictably with zero network.
"""

from __future__ import annotations

import pytest

pytest.importorskip("chromadb")

import tai_aitutor as tai
from tai_aitutor import vectorstore

TOPICS = ("python", "cooking", "space")


def fake_embed(texts, task="document"):
    """Deterministic 3-dim vectors: counts of topic words (unit-normalised)."""
    single = isinstance(texts, str)
    items = [texts] if single else list(texts)
    out = []
    for text in items:
        low = text.lower()
        vec = [float(low.count(topic)) for topic in TOPICS]
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        out.append([v / norm for v in vec])
    return out[0] if single else out


DOCS = [
    tai.Document(text="python python decorators generators python typing", metadata={"source": "tai_blog", "title": "Python Tricks"}, id="py"),
    tai.Document(text="cooking cooking pasta sauce cooking basil", metadata={"source": "food_blog", "title": "Cooking Pasta"}, id="cook"),
    tai.Document(text="space rockets orbit space telescope", metadata={"source": "tai_blog", "title": "Space Tech"}, id="space"),
]


@pytest.fixture()
def collection():
    vectorstore._reset_clients()
    col = tai.reset_collection("test_kb")
    yield col
    vectorstore._reset_clients()


def test_ingest_and_stats(collection):
    stats = tai.ingest(DOCS, collection, chunk_size=64, chunk_overlap=8, embed_fn=fake_embed, show_progress=False)
    assert stats.documents == 3
    assert stats.chunks >= 3
    assert stats.collection == "test_kb"
    assert collection.count() == stats.chunks


def test_ingest_accepts_strings_and_prebuilt_chunks(collection):
    chunks = [tai.Chunk(id="c1", text="space suit", metadata={"k": ["a", "b"], "n": None})]
    stats = tai.ingest(["plain python string doc"] + chunks, collection, embed_fn=fake_embed, show_progress=False)
    assert stats.chunks == 2
    got = collection.get(ids=["c1"], include=["metadatas"])
    # metadata sanitised: list joined, None dropped
    assert got["metadatas"][0] == {"k": "a, b"}


def test_ingest_respects_precomputed_embeddings(collection):
    chunk = tai.Chunk(id="pre", text="whatever", embedding=[1.0, 0.0, 0.0])
    tai.ingest([chunk], collection, embed_fn=fake_embed, show_progress=False)
    hits = tai.search("python python", collection, top_k=1, embed_fn=fake_embed)
    assert hits[0].id == "pre"  # matched via the precomputed vector, not the text


def test_search_ranks_by_topic(collection):
    tai.ingest(DOCS, collection, chunk_size=64, chunk_overlap=8, embed_fn=fake_embed, show_progress=False)
    hits = tai.search("how do I use python typing", collection, top_k=3, embed_fn=fake_embed)
    assert hits[0].metadata["title"] == "Python Tricks"
    assert hits[0].rank == 1 and hits[1].rank == 2
    assert hits[0].score >= hits[1].score
    assert 0.0 <= hits[0].score <= 1.0 + 1e-6


def test_search_where_filter_scopes_sources(collection):
    tai.ingest(DOCS, collection, chunk_size=64, chunk_overlap=8, embed_fn=fake_embed, show_progress=False)
    where = tai.build_where_filter("food_blog")
    hits = tai.search("cooking", collection, top_k=5, where=where, embed_fn=fake_embed)
    assert hits and all(h.metadata["source"] == "food_blog" for h in hits)


def test_search_top_k_respected(collection):
    """Regression: the requested cap is actually applied (old hybrid bug class)."""
    tai.ingest(DOCS, collection, chunk_size=32, chunk_overlap=4, embed_fn=fake_embed, show_progress=False)
    assert collection.count() > 2
    hits = tai.search("python space cooking", collection, top_k=2, embed_fn=fake_embed)
    assert len(hits) == 2
    assert tai.search("x", collection, top_k=0, embed_fn=fake_embed) == []


def test_get_all_chunks_enumerates_everything(collection):
    tai.ingest(DOCS, collection, chunk_size=32, chunk_overlap=4, embed_fn=fake_embed, show_progress=False)
    everything = tai.get_all_chunks(collection, page_size=2)  # force pagination
    assert len(everything) == collection.count()
    assert all(c.text for c in everything)
    assert any(c.metadata.get("source") == "tai_blog" for c in everything)


def test_build_where_filter_shapes():
    assert tai.build_where_filter(None) is None
    assert tai.build_where_filter([]) is None
    assert tai.build_where_filter("tai_blog") == {"source": {"$eq": "tai_blog"}}
    assert tai.build_where_filter(["one"]) == {"source": {"$eq": "one"}}
    assert tai.build_where_filter(["a", "b"]) == {"source": {"$in": ["a", "b"]}}
    assert tai.build_where_filter(["a", "b"], key="topic") == {"topic": {"$in": ["a", "b"]}}


def test_reset_collection_clears(collection):
    tai.ingest(DOCS, collection, embed_fn=fake_embed, show_progress=False)
    assert collection.count() > 0
    fresh = tai.reset_collection("test_kb")
    assert fresh.count() == 0
