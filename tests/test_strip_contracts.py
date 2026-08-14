"""Acceptance tests for the strip pass (tai_aitutor_strip_spec.md §6).

Each test pins one contract the spec asks for, so a regression is loud.
"""

from __future__ import annotations

import pytest

import tai_aitutor as tai
from tai_aitutor import _retry

REMOVED = [
    "Chat", "ToolLoop", "ChatEvent", "chat_completion", "Completion", "ToolCall", "Usage",
    "generate_stream", "generate_vision", "ask_batch",
    "answer", "answer_with_sources", "answer_stream", "AnswerStream", "Answer",
    "hybrid_search", "subquestion_answer", "multi_step_answer", "pack_context",
    "ingest", "IngestStats", "make_qa_pairs", "run_judges", "JudgeReport", "context_tokens",
    "make_retrieval_tool", "render_tool_result", "truncate", "estimate_cost", "mrr",
    "load_jsonl", "load_directory", "load_files", "load_wikipedia", "load_hf_dataset",
    "extract_keywords", "extract_summary", "extract_questions", "situate_chunk",
    "situate_chunks", "SituatedContext", "make_training_pairs", "train_embedder",
    "evaluate_embedder",
]


@pytest.mark.parametrize("name", REMOVED)
def test_removed_symbols_are_not_importable(name):
    assert name not in tai.__all__
    assert not hasattr(tai, name)


def test_removed_modules_are_gone():
    for module in ("chat", "extractors", "finetune"):
        with pytest.raises(ImportError):
            __import__(f"tai_aitutor.{module}")


def test_bm25_persistence_removed():
    assert not hasattr(tai.BM25Index, "save")
    assert not hasattr(tai.BM25Index, "load")


def test_unknown_provider_errors_are_value_errors():
    """`except ValueError` must catch what generate/embed/extract raise."""
    for err in (
        tai.TaiAitutorError, tai.UnsupportedProviderError, tai.ProviderNotInstalledError,
        tai.MissingKeyError, tai.EmbeddingsNotAvailableError, tai.StructuredOutputError,
    ):
        assert issubclass(err, ValueError)

    with pytest.raises(ValueError):
        tai.embed("x", provider="anthropic")


def test_no_retry_unless_asked():
    calls = []

    def flaky():
        calls.append(1)
        raise TimeoutError("timed out")

    with pytest.raises(TimeoutError):
        _retry.with_retries(flaky)
    assert len(calls) == 1, "a call must happen exactly once when retries=0"


def test_retries_when_explicitly_requested(monkeypatch):
    monkeypatch.setattr(_retry.time, "sleep", lambda _s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("timed out")
        return "ok"

    assert _retry.with_retries(flaky, retries=3) == "ok"
    assert len(calls) == 3


def test_permanent_errors_are_never_retried(monkeypatch):
    monkeypatch.setattr(_retry.time, "sleep", lambda _s: None)
    calls = []

    def broken():
        calls.append(1)
        raise ValueError("bad request: connection string malformed")

    with pytest.raises(ValueError):
        _retry.with_retries(broken, retries=5)
    assert len(calls) == 1, "'connection' in the message must not make an error transient"


def test_generate_takes_model_positionally():
    import inspect

    params = list(inspect.signature(tai.generate).parameters.values())
    assert [p.name for p in params[:3]] == ["prompt", "system", "model"]
    assert all(p.kind is not p.KEYWORD_ONLY for p in params[:3])


def test_extract_takes_model_positionally():
    import inspect

    params = list(inspect.signature(tai.extract).parameters.values())
    assert [p.name for p in params[:4]] == ["prompt", "schema", "system", "model"]
    assert all(p.kind is not p.KEYWORD_ONLY for p in params[:4])


def test_router_has_no_substring_fallback():
    from tai_aitutor.router import _match_route

    routes = {"knowledge": "...", "general": "..."}
    assert _match_route("knowledge", routes) == "knowledge"
    assert _match_route("KNOWLEDGE", routes) == "knowledge"
    assert _match_route("know", routes) is None
    assert _match_route("general knowledge questions", routes) is None


# --------------------------------------------------------------------------- #
# Credential and dependency failures must be actionable, never silent
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "call, env, needle",
    [
        (lambda: tai.generate("hi"), "GOOGLE_API_KEY", "GOOGLE_API_KEY"),
        (lambda: tai.generate("hi", provider="openai"), "OPENAI_API_KEY", "OPENAI_API_KEY"),
        (lambda: tai.generate("hi", provider="anthropic"), "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    ],
)
def test_missing_key_names_the_variable_to_set(monkeypatch, call, env, needle):
    monkeypatch.delenv(env, raising=False)
    with pytest.raises(tai.MissingKeyError) as err:
        call()
    assert needle in str(err.value)
    assert "Colab Secrets" in str(err.value)


def test_missing_cohere_key_is_actionable(monkeypatch):
    pytest.importorskip("cohere")
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    # A real candidate is required: rerank() returns [] for an empty list before it
    # ever builds a client, so passing [] would test the short-circuit, not the key.
    hits = [tai.ScoredChunk(chunk=tai.Chunk(id="c1", text="candidate"), score=1.0, rank=1)]
    with pytest.raises(tai.MissingKeyError) as err:
        tai.rerank("q", hits)
    assert "COHERE_API_KEY" in str(err.value)


def test_rerank_of_nothing_costs_nothing(monkeypatch):
    """Empty candidates short-circuit — no client, no key, no API call."""
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    assert tai.rerank("q", []) == []
