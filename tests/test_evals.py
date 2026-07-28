from __future__ import annotations

import json

import pytest

import tai_aitutor as tai
from tai_aitutor import evals
from tai_aitutor.chunking import Chunk
from tai_aitutor.errors import TaiAitutorError
from tai_aitutor.retrieval import ScoredChunk
from tai_aitutor.synthesis import Answer

# --------------------------------------------------------------------------- #
# QADataset — legacy compatibility is the acceptance test
# --------------------------------------------------------------------------- #

LEGACY_JSON = {
    # exactly what EmbeddingQAFinetuneDataset.save_json wrote
    "queries": {"q1": "What is RAG?", "q2": "What is chunking?"},
    "corpus": {"c1": "RAG retrieves context.", "c2": "Chunking splits documents."},
    "relevant_docs": {"q1": ["c1"], "q2": ["c2"]},
    "mode": "text",
}


def test_qadataset_loads_legacy_file(tmp_path):
    path = tmp_path / "rag_eval_dataset.json"
    path.write_text(json.dumps(LEGACY_JSON))
    qa = tai.QADataset.load(path)
    assert len(qa) == 2
    assert qa.queries["q1"] == "What is RAG?"
    assert qa.relevant_docs["q2"] == ["c2"]
    assert qa.mode == "text"


def test_qadataset_round_trip_stays_legacy_shaped(tmp_path):
    qa = tai.QADataset(**{k: v for k, v in LEGACY_JSON.items()})
    out = tmp_path / "saved.json"
    qa.save(out)
    raw = json.loads(out.read_text())
    assert set(raw) == {"queries", "corpus", "relevant_docs", "mode"}
    assert tai.QADataset.load(out).queries == qa.queries


def test_qadataset_coerces_string_relevant_docs(tmp_path):
    data = dict(LEGACY_JSON, relevant_docs={"q1": "c1", "q2": ["c2"]})
    path = tmp_path / "odd.json"
    path.write_text(json.dumps(data))
    qa = tai.QADataset.load(path)
    assert qa.relevant_docs["q1"] == ["c1"]


def test_qadataset_missing_keys_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"queries": {}}))
    with pytest.raises(TaiAitutorError) as err:
        tai.QADataset.load(path)
    assert "relevant_docs" in str(err.value)


def test_qadataset_sample_deterministic():
    queries = {f"q{i}": f"question {i}" for i in range(20)}
    corpus = {f"c{i}": f"chunk {i}" for i in range(20)}
    relevant = {f"q{i}": [f"c{i}"] for i in range(20)}
    qa = tai.QADataset(queries=queries, corpus=corpus, relevant_docs=relevant)
    a, b = qa.sample(5, seed=7), qa.sample(5, seed=7)
    assert list(a.queries) == list(b.queries)
    assert len(a) == 5
    # subset corpus keeps only needed chunks
    assert set(a.corpus) == {docs[0] for docs in a.relevant_docs.values()}
    assert qa.sample(50) is qa  # n >= len → unchanged


# --------------------------------------------------------------------------- #
# make_qa_pairs (extract is faked — no network)
# --------------------------------------------------------------------------- #


def test_make_qa_pairs_builds_dataset(monkeypatch):
    def fake_extract(prompt, schema, system=None, model=None, provider=None):
        assert "Context excerpt" in prompt
        return schema(questions=["Q about this?", "Another Q?"])

    monkeypatch.setattr(evals, "extract", fake_extract)
    chunks = [Chunk(id=f"c{i}", text=f"chunk text {i}") for i in range(6)]
    qa = tai.make_qa_pairs(chunks, n_chunks=4, questions_per_chunk=2, show_progress=False)
    assert len(qa.corpus) == 4
    assert len(qa.queries) == 8
    for qid, gold in qa.relevant_docs.items():
        assert len(gold) == 1
        assert qid.startswith(gold[0])
        assert qa.queries[qid].endswith("?")


def test_make_qa_pairs_deterministic_sampling(monkeypatch):
    monkeypatch.setattr(
        evals, "extract", lambda p, s, system=None, model=None, provider=None: s(questions=["Q?"])
    )
    chunks = [Chunk(id=f"c{i}", text=f"t{i}") for i in range(10)]
    first = tai.make_qa_pairs(chunks, n_chunks=3, seed=1, show_progress=False)
    second = tai.make_qa_pairs(chunks, n_chunks=3, seed=1, show_progress=False)
    assert set(first.corpus) == set(second.corpus)


def test_make_qa_pairs_empty_raises():
    with pytest.raises(TaiAitutorError):
        tai.make_qa_pairs([], show_progress=False)


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #


def make_qa() -> tai.QADataset:
    return tai.QADataset(
        queries={"q1": "one?", "q2": "two?", "q3": "three?"},
        corpus={"g1": "gold 1", "g2": "gold 2", "g3": "gold 3"},
        relevant_docs={"q1": ["g1"], "q2": ["g2"], "q3": ["g3"]},
    )


def fake_search_fn(query, top_k):
    # q1 → gold at rank 1; q2 → gold at rank 3; q3 → gold never retrieved
    table = {
        "one?": ["g1", "x", "y", "z", "w"],
        "two?": ["x", "y", "g2", "z", "w"],
        "three?": ["x", "y", "z", "w", "v"],
    }
    return table[query][:top_k]


def test_evaluate_retrieval_hit_rate_and_mrr():
    report = tai.evaluate_retrieval(make_qa(), search_fn=fake_search_fn, top_k=5)
    assert report.n_queries == 3
    assert abs(report.hit_rate - 2 / 3) < 1e-9
    assert abs(report.mrr - (1.0 + 1 / 3 + 0.0) / 3) < 1e-9
    misses = report.misses()
    assert [m.query_id for m in misses] == ["q3"]
    assert report.per_query[1].first_relevant_rank == 3


def test_top_k_changes_the_answer():
    # at top_k=2 the rank-3 gold becomes a miss
    report = tai.evaluate_retrieval(make_qa(), search_fn=fake_search_fn, top_k=2)
    assert abs(report.hit_rate - 1 / 3) < 1e-9
    assert report.top_k == 2


def test_hit_rate_and_mrr_wrappers():
    assert abs(tai.hit_rate(make_qa(), search_fn=fake_search_fn, top_k=5) - 2 / 3) < 1e-9
    assert abs(tai.mrr(make_qa(), search_fn=fake_search_fn, top_k=5) - 4 / 9) < 1e-9


def test_evaluate_retrieval_accepts_scoredchunk_results():
    def scored_fn(query, top_k):
        gold = {"one?": "g1", "two?": "g2", "three?": "g3"}[query]
        return [
            ScoredChunk(chunk=Chunk(id=gold, text="t"), score=0.9, rank=1),
            ScoredChunk(chunk=Chunk(id="other", text="t"), score=0.5, rank=2),
        ]

    report = tai.evaluate_retrieval(make_qa(), search_fn=scored_fn, top_k=2)
    assert report.hit_rate == 1.0 and report.mrr == 1.0


def test_evaluate_retrieval_collection_binding(monkeypatch):
    calls = []

    def fake_search(question, collection, top_k=5):
        calls.append((question, collection, top_k))
        return ["g1"] if question == "one?" else []

    monkeypatch.setattr(evals, "search", fake_search)
    report = tai.evaluate_retrieval(make_qa(), collection="COL", top_k=4)
    assert all(c[1] == "COL" and c[2] == 4 for c in calls)
    assert abs(report.hit_rate - 1 / 3) < 1e-9


def test_evaluate_retrieval_needs_source():
    with pytest.raises(TaiAitutorError):
        tai.evaluate_retrieval(make_qa())


def test_sweep_top_k_one_retrieval_scores_every_k():
    calls = []

    def counting_fn(query, top_k):
        calls.append((query, top_k))
        return fake_search_fn(query, top_k)

    reports = tai.sweep_top_k(make_qa(), [2, 5], search_fn=counting_fn)
    # one retrieval per question, at max(k) — not one per (question, k)
    assert len(calls) == 3
    assert all(k == 5 for _, k in calls)
    # every cutoff scored from the same ranked lists, matching a direct eval
    direct_2 = tai.evaluate_retrieval(make_qa(), search_fn=fake_search_fn, top_k=2)
    direct_5 = tai.evaluate_retrieval(make_qa(), search_fn=fake_search_fn, top_k=5)
    assert reports[2].hit_rate == direct_2.hit_rate and reports[2].mrr == direct_2.mrr
    assert reports[5].hit_rate == direct_5.hit_rate and reports[5].mrr == direct_5.mrr
    assert reports[2].top_k == 2 and reports[5].top_k == 5


def test_sweep_top_k_validates_k_values():
    with pytest.raises(TaiAitutorError):
        tai.sweep_top_k(make_qa(), [], search_fn=fake_search_fn)
    with pytest.raises(TaiAitutorError):
        tai.sweep_top_k(make_qa(), [0, 5], search_fn=fake_search_fn)


def test_context_tokens_and_avg(monkeypatch):
    monkeypatch.setattr("tai_aitutor.tokens.n_tokens", lambda text, model=None: len(text))
    hits = [ScoredChunk(chunk=Chunk(id="a", text="12345"), score=1.0, rank=1),
            ScoredChunk(chunk=Chunk(id="b", text="123"), score=0.9, rank=2)]
    assert tai.context_tokens(hits) == 8
    assert tai.context_tokens(["ab", "cd"]) == 4

    report = tai.evaluate_retrieval(make_qa(), search_fn=fake_search_fn, top_k=5)
    # retrieved ids outside qa.corpus are skipped; gold ids price their corpus text
    avg = report.avg_context_tokens(make_qa())
    assert avg == (len("gold 1") + len("gold 2") + 0) / 3


# --------------------------------------------------------------------------- #
# Judges (extract faked)
# --------------------------------------------------------------------------- #


def test_judges_build_typed_verdicts(monkeypatch):
    captured = {}

    def fake_extract(prompt, schema, system=None, model=None, provider=None):
        captured[schema.__name__] = {"prompt": prompt, "system": system}
        if schema is tai.FaithfulnessVerdict:
            return schema(faithful=True, reasoning="supported")
        if schema is tai.RelevancyVerdict:
            return schema(relevant=False, reasoning="off-topic")
        return schema(score=4.5, reasoning="close to reference")

    monkeypatch.setattr(evals, "extract", fake_extract)

    f = tai.judge_faithfulness("ans", "ctx")
    assert f.faithful is True
    assert "CONTEXT:\nctx" in captured["FaithfulnessVerdict"]["prompt"]

    hits = [ScoredChunk(chunk=Chunk(id="c", text="chunk text"), score=1.0, rank=1)]
    r = tai.judge_relevancy("q?", "ans", hits)
    assert r.relevant is False
    assert "chunk text" in captured["RelevancyVerdict"]["prompt"]

    c = tai.judge_correctness("q?", "ans", "ref")
    assert c.passing is True
    assert "REFERENCE ANSWER:\nref" in captured["CorrectnessVerdict"]["prompt"]


def test_correctness_passing_threshold():
    assert tai.CorrectnessVerdict(score=4.0, reasoning="").passing is True
    assert tai.CorrectnessVerdict(score=3.9, reasoning="").passing is False
    with pytest.raises(Exception):
        tai.CorrectnessVerdict(score=7.0, reasoning="")  # out of 1-5 range


# --------------------------------------------------------------------------- #
# run_judges (judge functions faked)
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_judges(monkeypatch):
    monkeypatch.setitem(
        evals._JUDGES, "faithfulness",
        lambda row, kw: tai.FaithfulnessVerdict(faithful="good" in row["answer"], reasoning=""),
    )
    monkeypatch.setitem(
        evals._JUDGES, "relevancy",
        lambda row, kw: tai.RelevancyVerdict(relevant=True, reasoning=""),
    )
    monkeypatch.setitem(
        evals._JUDGES, "correctness",
        lambda row, kw: tai.CorrectnessVerdict(score=5.0 if "good" in row["answer"] else 2.0, reasoning=""),
    )


def test_run_judges_aggregates(fake_judges):
    rows = [
        {"question": "q1", "answer": "good answer", "context": "ctx", "reference": "ref"},
        {"question": "q2", "answer": "bad answer", "context": "ctx", "reference": "ref"},
    ]
    report = tai.run_judges(
        rows, judges=("faithfulness", "relevancy", "correctness"), show_progress=False
    )
    assert report.n_rows == 2
    assert report.faithfulness_rate == 0.5
    assert report.relevancy_rate == 1.0
    assert report.correctness_mean == 3.5
    assert report.correctness_pass_rate == 0.5
    assert report.summary()["faithfulness_rate"] == 0.5
    # verdicts stay aligned with rows
    assert report.verdicts["faithfulness"][0].faithful is True
    assert report.verdicts["faithfulness"][1].faithful is False


def test_run_judges_accepts_answer_objects(fake_judges):
    hits = [ScoredChunk(chunk=Chunk(id="c", text="evidence"), score=1.0, rank=1)]
    ans = Answer(text="good answer", sources=hits)
    report = tai.run_judges(
        [{"question": "q", "answer": ans}], judges=("faithfulness",), show_progress=False
    )
    assert report.faithfulness_rate == 1.0
    assert report.relevancy_rate is None  # judge not run


def test_run_judges_unknown_judge():
    with pytest.raises(TaiAitutorError):
        tai.run_judges([{"question": "q", "answer": "a"}], judges=("vibes",), show_progress=False)


def test_run_judges_correctness_requires_reference(fake_judges):
    with pytest.raises(TaiAitutorError) as err:
        tai.run_judges(
            [{"question": "q", "answer": "a", "context": "c"}],
            judges=("correctness",),
            show_progress=False,
        )
    assert "reference" in str(err.value)
