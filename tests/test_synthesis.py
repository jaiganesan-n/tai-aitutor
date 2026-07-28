from __future__ import annotations

import pytest

import tai_aitutor as tai
from tai_aitutor import llm, synthesis
from tai_aitutor.chunking import Chunk
from tai_aitutor.errors import TaiAitutorError
from tai_aitutor.retrieval import ScoredChunk, expand_window


def make_hits():
    return [
        ScoredChunk(
            chunk=Chunk(id="c1", text="RAG grounds answers in retrieved context.",
                        metadata={"title": "RAG 101", "url": "https://x/rag"}),
            score=0.91, rank=1,
        ),
        ScoredChunk(
            chunk=Chunk(id="c2", text="Chunks are embedded and stored in a vector DB.",
                        metadata={"source_name": "tai_blog"}),
            score=0.84, rank=2,
        ),
    ]


def fake_complete(prompt, system, cfg, temperature, max_tokens, reasoning_effort):
    fake_complete.captured = {"prompt": prompt, "system": system, "cfg": cfg}
    return "Grounded answer [1].", tai.Usage(100, 20)


@pytest.fixture(autouse=True)
def patch_complete(monkeypatch):
    monkeypatch.setattr(llm, "_complete", fake_complete)


def test_build_rag_prompt_numbers_and_sources():
    prompt = tai.build_rag_prompt("What is RAG?", make_hits())
    assert "[1] RAG 101 (https://x/rag)" in prompt
    assert "[2] tai_blog" in prompt
    assert "RAG grounds answers" in prompt
    assert prompt.rstrip().endswith("Answer the question using only the context excerpts above.")
    assert "What is RAG?" in prompt


def test_answer_with_retriever():
    hits = make_hits()
    ans = tai.answer("What is RAG?", retriever=lambda q: hits)
    assert ans.text == "Grounded answer [1]."
    assert ans.sources == hits
    assert ans.usage.total_tokens == 120
    assert str(ans) == ans.text
    assert fake_complete.captured["system"] == synthesis.prompts.RAG_SYSTEM


def test_answer_requires_collection_or_retriever():
    with pytest.raises(TaiAitutorError):
        tai.answer("What is RAG?")


def test_answer_uses_search_when_collection_given(monkeypatch):
    hits = make_hits()
    calls = {}

    def fake_search(question, collection, top_k=5, where=None):
        calls.update(question=question, collection=collection, top_k=top_k, where=where)
        return hits

    monkeypatch.setattr(synthesis, "search", fake_search)
    ans = tai.answer("Q?", collection="COL", top_k=7, where={"source": {"$eq": "s"}})
    assert calls == {"question": "Q?", "collection": "COL", "top_k": 7,
                     "where": {"source": {"$eq": "s"}}}
    assert ans.sources == hits


def test_answer_with_sources_cited_prompt():
    tai.answer_with_sources("Q?", retriever=lambda q: make_hits())
    assert "Cite the excerpts" in fake_complete.captured["prompt"]
    assert fake_complete.captured["system"] == synthesis.prompts.RAG_SYSTEM_CITED


def test_answer_model_provider_override():
    tai.configure(provider="gemini")
    tai.answer("Q?", retriever=lambda q: make_hits(), provider="openai", model="gpt-5")
    cfg = fake_complete.captured["cfg"]
    assert cfg.provider == "openai" and cfg.chat_model == "gpt-5"
    assert tai.get_config().provider == "gemini"  # global untouched


def test_answer_stream_sources_upfront(monkeypatch):
    monkeypatch.setattr(llm, "generate_stream", lambda *a, **k: iter(["Hello", " world"]))
    stream = tai.answer_stream("Q?", retriever=lambda q: make_hits())
    assert len(stream.sources) == 2  # available before consuming
    assert "".join(stream) == "Hello world"
    assert stream.text == "Hello world"


def test_expand_window_swaps_text():
    hits = [
        ScoredChunk(chunk=Chunk(id="s1", text="Small sentence.",
                                metadata={"window": "Prev. Small sentence. Next."}),
                    score=0.9, rank=1),
        ScoredChunk(chunk=Chunk(id="s2", text="No window here.", metadata={}), score=0.8, rank=2),
    ]
    out = expand_window(hits)
    assert out[0].text == "Prev. Small sentence. Next."
    assert out[0].score == 0.9 and out[0].rank == 1
    assert out[1].text == "No window here."  # untouched when no window stored
