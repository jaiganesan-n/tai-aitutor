from __future__ import annotations

import pytest

import tai_aitutor as tai
from tai_aitutor import extractors
from tai_aitutor.chunking import Chunk


@pytest.fixture
def fake_extract(monkeypatch):
    def fake(prompt, schema, system=None, model=None, provider=None):
        name = schema.__name__
        if name == "_Keywords":
            return schema(keywords=["rag", "chunks", "embeddings", "retrieval"])
        if name == "_Summary":
            return schema(summary="  A short summary.  ")
        if name == "_Questions":
            return schema(questions=["What is X?", "Why Y?", "How Z?", "Extra?"])
        if name == "SituatedContext":
            return schema(context="This chunk covers setup, early in the document.")
        raise AssertionError(f"unexpected schema {name}")

    monkeypatch.setattr(extractors, "extract", fake)
    return fake


CHUNKS = [
    Chunk(id="a", text="alpha text", metadata={"source": "s1"}),
    Chunk(id="b", text="beta text", metadata={"source": "s2"}),
]


def test_extract_keywords_caps_and_joins(fake_extract):
    out = tai.extract_keywords(CHUNKS, n=3, show_progress=False)
    assert out[0].metadata["keywords"] == "rag, chunks, embeddings"
    assert out[1].metadata["source"] == "s2"  # existing metadata preserved
    # originals untouched
    assert "keywords" not in CHUNKS[0].metadata
    assert [c.id for c in out] == ["a", "b"]  # order preserved


def test_extract_summary_strips(fake_extract):
    out = tai.extract_summary(CHUNKS, show_progress=False)
    assert out[0].metadata["summary"] == "A short summary."


def test_extract_questions_joined_and_capped(fake_extract):
    out = tai.extract_questions(CHUNKS, n=2, show_progress=False)
    assert out[0].metadata["questions_answered"] == "What is X? | Why Y?"


def test_situate_chunk_and_chunks(fake_extract):
    context = tai.situate_chunk("chunk body", "full document")
    assert context.startswith("This chunk covers setup")

    out = tai.situate_chunks(CHUNKS, "full document", show_progress=False)
    assert out[0].text.startswith("This chunk covers setup")
    assert out[0].text.endswith("alpha text")
    assert out[0].metadata["original_text"] == "alpha text"
    assert CHUNKS[0].text == "alpha text"  # original unmutated


def test_extractors_plug_into_ingest(fake_extract):
    pytest.importorskip("chromadb")

    def fake_embed(texts, task="document"):
        items = [texts] if isinstance(texts, str) else list(texts)
        vecs = [[float(len(t) % 7), 1.0, 0.5] for t in items]
        return vecs[0] if isinstance(texts, str) else vecs

    from tai_aitutor import vectorstore

    vectorstore._reset_clients()
    col = tai.reset_collection("extract_kb")
    stats = tai.ingest(
        [tai.Document(text="python " * 40, id="d1")],
        col,
        chunk_size=64,
        chunk_overlap=8,
        enrich=[lambda chunks: tai.extract_keywords(chunks, show_progress=False)],
        embed_fn=fake_embed,
        show_progress=False,
    )
    assert stats.chunks >= 1
    stored = col.get(include=["metadatas"])
    assert all("keywords" in m for m in stored["metadatas"])
    vectorstore._reset_clients()
