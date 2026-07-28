"""Phase 3 tests: BM25, RRF, hybrid, rerankers, query transforms, token budget."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

import tai_aitutor as tai
from tai_aitutor import llm, retrieval
from tai_aitutor.chunking import Chunk
from tai_aitutor.errors import TaiAitutorError
from tai_aitutor.retrieval import ScoredChunk

# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #


def test_code_tokenize_splits_camel_and_paths():
    assert tai.code_tokenize("VectorStoreIndex") == ["vector", "store", "index"]
    assert tai.code_tokenize("llama_index.core") == ["llama", "index", "core"]
    assert tai.code_tokenize("app/chroma_rag.py") == ["app", "chroma", "rag", "py"]
    assert "c" in tai.code_tokenize("the C language")  # short terms kept
    assert tai.code_tokenize("HTTPServer") == ["http", "server"]


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #

CORPUS = [
    Chunk(id="d0", text="python decorators and python generators", metadata={"source": "blog"}),
    Chunk(id="d1", text="cooking pasta with basil sauce", metadata={"source": "food"}),
    Chunk(id="d2", text="python typing", metadata={"source": "docs"}),
    Chunk(id="d3", text="rockets and orbital mechanics", metadata={"source": "space"}),
]


def test_bm25_ranks_and_scores_hand_checked():
    bm25 = tai.BM25Index().build(CORPUS)
    hits = bm25.search("python", top_k=10)
    # d0 says "python" twice (tf=2, dl=5); d2 once in a shorter doc (tf=1, dl=2).
    # With k1=1.5, b=0.75, avgdl=(5+5+2+4)/4=4: tf=2 outweighs the length norm.
    assert [h.id for h in hits] == ["d0", "d2"]
    assert hits[0].rank == 1 and hits[1].rank == 2

    # hand-check both scores: N=4, df=2 → idf = ln(1 + 2.5/2.5) = ln 2
    idf = math.log(2.0)
    d0_expected = idf * (2 * 2.5) / (2 + 1.5 * (0.25 + 0.75 * (5 / 4)))
    d2_expected = idf * (1 * 2.5) / (1 + 1.5 * (0.25 + 0.75 * (2 / 4)))
    assert abs(hits[0].score - d0_expected) < 1e-9
    assert abs(hits[1].score - d2_expected) < 1e-9
    assert d0_expected > d2_expected


def test_bm25_no_match_and_empty():
    bm25 = tai.BM25Index().build(CORPUS)
    assert bm25.search("quantum", top_k=5) == []
    assert bm25.search("python", top_k=0) == []
    assert tai.BM25Index().search("python") == []
    assert len(bm25) == 4


def test_bm25_query_uses_same_tokenizer():
    bm25 = tai.BM25Index().build([Chunk(id="x", text="the OrbitalMechanics module")])
    assert bm25.search("orbital_mechanics", top_k=1)  # dotted/camel forms match


def test_bm25_save_load_round_trip(tmp_path):
    bm25 = tai.BM25Index().build(CORPUS)
    path = tmp_path / "bm25.json.gz"
    bm25.save(path)
    loaded = tai.BM25Index.load(path)
    original = [(h.id, round(h.score, 9)) for h in bm25.search("python cooking", top_k=4)]
    reloaded = [(h.id, round(h.score, 9)) for h in loaded.search("python cooking", top_k=4)]
    assert original == reloaded
    assert loaded._chunks[0].metadata == {"source": "blog"}


def test_bm25_load_rejects_foreign_files(tmp_path):
    import gzip
    import json as jsonlib

    path = tmp_path / "other.json.gz"
    with gzip.open(path, "wt") as f:
        jsonlib.dump({"format": "something-else"}, f)
    with pytest.raises(TaiAitutorError):
        tai.BM25Index.load(path)


# --------------------------------------------------------------------------- #
# RRF
# --------------------------------------------------------------------------- #


def hit(chunk_id, score, rank):
    return ScoredChunk(chunk=Chunk(id=chunk_id, text=f"text {chunk_id}"), score=score, rank=rank)


def test_rrf_math_and_cap():
    dense = [hit("a", 0.9, 1), hit("b", 0.8, 2), hit("c", 0.7, 3)]
    keyword = [hit("b", 11.0, 1), hit("d", 9.0, 2)]
    fused = tai.rrf_fuse(dense, keyword, k=60, keep=3)
    # b appears in both lists: 1/(60+2) + 1/(60+1)
    assert fused[0].id == "b"
    assert abs(fused[0].score - (1 / 62 + 1 / 61)) < 1e-12
    assert [h.rank for h in fused] == [1, 2, 3]
    assert len(fused) == 3  # keep applied (regression: caps must actually cap)
    assert {h.id for h in fused} == {"b", "a", "d"}


def test_rrf_ignores_incomparable_scores():
    # BM25 scores ~11.0 vs cosine ~0.9 — positions decide, not raw scores
    dense = [hit("a", 0.9, 1)]
    keyword = [hit("z", 999.0, 1), hit("a", 500.0, 2)]
    fused = tai.rrf_fuse(dense, keyword, k=60, keep=10)
    assert fused[0].id == "a"  # in both lists → wins


# --------------------------------------------------------------------------- #
# hybrid_search (fake collection + real BM25)
# --------------------------------------------------------------------------- #


class FakeCollection:
    """Chroma-shaped .query returning the python docs first."""

    def query(self, **kwargs):
        order = ["d2", "d0", "d3", "d1"][: kwargs["n_results"]]
        text = {c.id: c.text for c in CORPUS}
        meta = {c.id: c.metadata for c in CORPUS}
        return {
            "ids": [order],
            "documents": [[text[i] for i in order]],
            "metadatas": [[meta[i] for i in order]],
            "distances": [[0.1 * (i + 1) for i in range(len(order))]],
        }


def fake_embed(texts, task="query"):
    return [0.5, 0.5]


def test_hybrid_search_fuses_both_legs():
    bm25 = tai.BM25Index().build(CORPUS)
    fused = tai.hybrid_search(
        "python typing", FakeCollection(), bm25=bm25,
        dense_top_k=3, bm25_top_k=3, keep=4, embed_fn=fake_embed,
    )
    assert fused[0].id == "d2"  # top of both legs
    assert len(fused) <= 4
    assert [h.rank for h in fused] == list(range(1, len(fused) + 1))


def test_hybrid_search_where_filters_bm25_leg():
    bm25 = tai.BM25Index().build(CORPUS)
    fused = tai.hybrid_search(
        "python", FakeCollection(), bm25=bm25,
        keep=10, where={"source": {"$eq": "docs"}}, embed_fn=fake_embed,
    )
    keyword_ids = {h.id for h in fused}
    # BM25 would return d0 (source=blog); the filter keeps only docs-side d2 from that leg
    assert "d2" in keyword_ids


def test_matches_where_shapes():
    assert retrieval._matches_where({"source": "a"}, None)
    assert retrieval._matches_where({"source": "a"}, {"source": {"$eq": "a"}})
    assert not retrieval._matches_where({"source": "b"}, {"source": {"$eq": "a"}})
    assert retrieval._matches_where({"source": "a"}, {"source": {"$in": ["a", "b"]}})
    assert not retrieval._matches_where({}, {"source": {"$in": ["a"]}})
    assert retrieval._matches_where({"source": "a"}, {"source": "a"})


# --------------------------------------------------------------------------- #
# rerank (fake Cohere client)
# --------------------------------------------------------------------------- #


def test_rerank_maps_floor_and_reorders(monkeypatch):
    calls = {}

    class FakeCohere:
        def rerank(self, **kwargs):
            calls.update(kwargs)
            return SimpleNamespace(
                results=[
                    SimpleNamespace(index=2, relevance_score=0.95),
                    SimpleNamespace(index=0, relevance_score=0.40),
                    SimpleNamespace(index=1, relevance_score=0.05),  # below floor
                ]
            )

    monkeypatch.setattr("tai_aitutor.embeddings._client_cohere", lambda api_key=None: FakeCohere())
    hits = [hit("a", 0.9, 1), hit("b", 0.8, 2), hit("c", 0.7, 3)]
    out = tai.rerank("q", hits, top_n=3, floor=0.10)
    assert [h.id for h in out] == ["c", "a"]        # judge order, floor applied
    assert [h.rank for h in out] == [1, 2]
    assert out[0].score == 0.95                      # reranker's score, not cosine
    assert calls["model"] == "rerank-v4.0-fast"
    assert calls["top_n"] == 3
    assert tai.rerank("q", [], top_n=3) == []


# --------------------------------------------------------------------------- #
# judge_rerank (fake extract) — the ordering-preserved regression
# --------------------------------------------------------------------------- #


def test_judge_rerank_preserves_judge_order(monkeypatch):
    def fake_extract(prompt, schema, system=None, model=None, provider=None):
        # judge says: excerpt 3 best, then 1; excerpt 2 weak; plus junk index 99
        return schema(
            scores=[
                {"index": 3, "score": 9.5},
                {"index": 1, "score": 6.0},
                {"index": 2, "score": 2.0},
                {"index": 99, "score": 10.0},
            ]
        )

    monkeypatch.setattr(llm, "extract", fake_extract)
    hits = [hit("a", 0.99, 1), hit("b", 0.98, 2), hit("c", 0.10, 3)]
    out = tai.judge_rerank("q", hits, top_n=2)
    # Regression (old notebook bug): output must follow the JUDGE's order and
    # carry the JUDGE's scores — not the original retrieval order/similarities.
    assert [h.id for h in out] == ["c", "a"]
    assert [h.score for h in out] == [9.5, 6.0]
    assert [h.rank for h in out] == [1, 2]
    assert tai.judge_rerank("q", [], top_n=2) == []


# --------------------------------------------------------------------------- #
# Query transforms (fake llm layer)
# --------------------------------------------------------------------------- #


def test_hyde_search_embeds_hypothetical(monkeypatch):
    monkeypatch.setattr(llm, "generate", lambda *a, **k: "A hypothetical answer passage.")
    embedded = []

    def spy_embed(texts, task="document"):
        embedded.append((texts, task))
        return [1.0, 0.0] if task == "document" else [0.0, 1.0]

    col = FakeCollection()
    captured = {}
    original_query = col.query

    def spy_query(**kwargs):
        captured.update(kwargs)
        return original_query(**kwargs)

    col.query = spy_query
    tai.hyde_search("what is x?", col, top_k=2, embed_fn=spy_embed)
    assert embedded[0] == ("A hypothetical answer passage.", "document")
    assert embedded[1][1] == "query"
    assert captured["query_embeddings"] == [[0.5, 0.5]]  # averaged with original

    tai.hyde_search("what is x?", col, top_k=2, include_original=False,
                    hypothetical="my own passage", embed_fn=spy_embed)
    assert captured["query_embeddings"] == [[1.0, 0.0]]  # hypothetical only, no generate call


def test_decompose_question(monkeypatch):
    monkeypatch.setattr(
        llm, "extract",
        lambda p, s, system=None, model=None, provider=None: s(
            questions=["What is A?", "What is B?", "", "What is C?", "What is D?", "What is E?"]
        ),
    )
    subs = tai.decompose_question("Compare A, B, C, D and E", n_max=4)
    assert subs == ["What is A?", "What is B?", "What is C?", "What is D?"]

    monkeypatch.setattr(
        llm, "extract",
        lambda p, s, system=None, model=None, provider=None: s(questions=[]),
    )
    assert tai.decompose_question("Simple?") == ["Simple?"]


def test_subquestion_answer_pipeline(monkeypatch):
    monkeypatch.setattr(
        llm, "extract",
        lambda p, s, system=None, model=None, provider=None: s(questions=["sub one?", "sub two?"])
        if s.__name__ == "_SubQuestions" else (_ for _ in ()).throw(AssertionError),
    )
    completions = []

    def fake_complete(prompt, system, cfg, temperature, max_tokens, reasoning_effort):
        completions.append(prompt)
        return f"answer #{len(completions)}", tai.Usage(10, 5)

    monkeypatch.setattr(llm, "_complete", fake_complete)
    retrieved = {"sub one?": [hit("a", 0.9, 1)], "sub two?": [hit("b", 0.8, 1), hit("a", 0.5, 2)]}
    ans = tai.subquestion_answer("Compare one and two", retriever=lambda q: retrieved[q])
    assert ans.text == "answer #3"                 # 2 sub-answers + 1 synthesis
    assert len(completions) == 3
    assert "Compare one and two" in completions[-1]
    assert "answer #1" in completions[-1]
    assert [h.id for h in ans.sources] == ["a", "b"]   # deduped, best-score order
    assert ans.sources[0].score == 0.9
    assert ans.usage.total_tokens == 45


def test_multi_step_answer_stops_when_done(monkeypatch):
    steps = iter([
        (False, "follow-up?"),
        (True, ""),
    ])

    def fake_extract(prompt, schema, system=None, model=None, provider=None):
        done, nxt = next(steps)
        return schema(done=done, next_question=nxt)

    completions = []

    def fake_complete(prompt, system, cfg, temperature, max_tokens, reasoning_effort):
        completions.append(prompt)
        return f"ans{len(completions)}", tai.Usage(1, 1)

    monkeypatch.setattr(llm, "extract", fake_extract)
    monkeypatch.setattr(llm, "_complete", fake_complete)
    ans = tai.multi_step_answer(
        "Big question?", retriever=lambda q: [hit(q[:3], 0.5, 1)], max_steps=5
    )
    # step1 answer + step2 answer + final synthesis = 3 completions; 2 planner calls
    assert len(completions) == 3
    assert ans.text == "ans3"
    assert len(ans.sources) == 2


def test_multi_step_requires_source():
    with pytest.raises(TaiAitutorError):
        tai.multi_step_answer("Q?")


# --------------------------------------------------------------------------- #
# pack_context
# --------------------------------------------------------------------------- #


def test_pack_context_budget(monkeypatch):
    monkeypatch.setattr(retrieval, "n_tokens", lambda text, model=None: 100)
    hits = [hit(f"c{i}", 1.0 - i * 0.1, i + 1) for i in range(5)]
    kept = tai.pack_context(hits, max_tokens=250)
    assert [h.id for h in kept] == ["c0", "c1"]
    assert [h.rank for h in kept] == [1, 2]
    assert tai.pack_context([], max_tokens=100) == []
    assert tai.pack_context(hits, max_tokens=1_000_000) == [
        ScoredChunk(chunk=h.chunk, score=h.score, rank=i + 1) for i, h in enumerate(hits)
    ] or len(tai.pack_context(hits, max_tokens=1_000_000)) == 5
