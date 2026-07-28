from __future__ import annotations

import tai_aitutor as tai
from tai_aitutor import display
from tai_aitutor.chunking import Chunk
from tai_aitutor.evals import RetrievalReport
from tai_aitutor.retrieval import ScoredChunk


def render_capture(monkeypatch):
    out = {}
    monkeypatch.setattr(display, "_render", lambda text: out.setdefault("text", text))
    return out


def test_show_chunks_label_fallback_includes_name(monkeypatch):
    out = render_capture(monkeypatch)
    hits = [
        ScoredChunk(chunk=Chunk(id="kb-001", text="KB text",
                                metadata={"name": "Transformers Quicktour"}), score=0.9, rank=1),
        ScoredChunk(chunk=Chunk(id="raw-002", text="No labels at all", metadata={}),
                    score=0.5, rank=2),
    ]
    tai.show_chunks(hits)
    assert "Transformers Quicktour" in out["text"]  # finding 5: "name" renders, not the id
    assert "raw-002" in out["text"]                 # id stays the last resort


def test_show_eval_table_extra_columns(monkeypatch):
    out = render_capture(monkeypatch)
    reports = {
        "dense": RetrievalReport(hit_rate=0.8, mrr=0.7, top_k=5),
        "hybrid": RetrievalReport(hit_rate=0.9, mrr=0.85, top_k=5),
    }
    tai.show_eval_table(reports, extra_columns={
        "avg ctx tokens": {"dense": 1200, "hybrid": 1450},
        "note": {"dense": "baseline"},  # missing rows render blank
    })
    text = out["text"]
    assert "avg ctx tokens" in text and "note" in text
    assert "| dense | 0.800 | 0.700 | 5 | 0 | 1200 | baseline |" in text
    assert "| hybrid | 0.900 | 0.850 | 5 | 0 | 1450 |  |" in text


def test_show_eval_table_without_extras_unchanged(monkeypatch):
    out = render_capture(monkeypatch)
    tai.show_eval_table({"dense": RetrievalReport(hit_rate=1.0, mrr=1.0, top_k=3)})
    assert "| configuration | hit rate | MRR | top_k | queries |" in out["text"]
