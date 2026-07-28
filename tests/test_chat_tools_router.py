"""Phase 4 tests: tools, ToolLoop/Chat memory modes, router, finetune helpers."""

from __future__ import annotations

import pytest

import tai_aitutor as tai
from tai_aitutor import chat as chat_module
from tai_aitutor import llm, router, tools
from tai_aitutor.errors import TaiAitutorError
from tai_aitutor.evals import QADataset
from tai_aitutor.llm import Completion, ToolCall, Usage

# --------------------------------------------------------------------------- #
# tools.tool()
# --------------------------------------------------------------------------- #


def test_tool_schema_from_signature():
    @tai.tool
    def lookup(query: str, limit: int = 5, strict: bool = False) -> str:
        """Look up things in the catalog."""
        return query

    assert lookup.name == "lookup"
    assert lookup.description == "Look up things in the catalog."
    assert lookup.parameters["properties"]["query"] == {"type": "string"}
    assert lookup.parameters["properties"]["limit"] == {"type": "integer", "default": 5}
    assert lookup.parameters["properties"]["strict"] == {"type": "boolean", "default": False}
    assert lookup.parameters["required"] == ["query"]
    assert lookup(query="x") == "x"


def test_tool_name_description_override_and_missing_doc():
    def fn(a: float):
        return a

    named = tai.tool(fn, name="halve", description="Halve a number.")
    assert named.name == "halve" and named.description == "Halve a number."
    with pytest.raises(TaiAitutorError):
        tai.tool(lambda x: x)  # no docstring, no description


def test_render_tool_result():
    assert tai.render_tool_result("plain") == "plain"
    assert tai.render_tool_result({"a": 1}) == '{"a": 1}'
    assert tai.render_tool_result([1, 2]) == "[1, 2]"


def test_make_retrieval_tool_formats_hits(monkeypatch):
    from tai_aitutor.chunking import Chunk
    from tai_aitutor.retrieval import ScoredChunk

    hits = [ScoredChunk(chunk=Chunk(id="c1", text="RAG retrieves.",
                                    metadata={"title": "RAG 101"}), score=0.9, rank=1)]
    monkeypatch.setattr("tai_aitutor.retrieval.search", lambda q, col, top_k=5: hits)
    retrieval_tool = tai.make_retrieval_tool("COL", top_k=3)
    out = retrieval_tool(query="what is rag?")
    assert "[1] RAG 101" in out and "RAG retrieves." in out
    assert retrieval_tool.parameters["required"] == ["query"]

    custom = tai.make_retrieval_tool("COL", retriever=lambda q: [])
    assert custom(query="x") == "No relevant passages found."


def test_search_web_via_fake_tavily(monkeypatch):
    class FakeTavily:
        def search(self, query, max_results):
            return {"results": [{"title": "T", "url": "https://u", "content": "C",
                                 "extra": "ignored"}][:max_results]}

    monkeypatch.setattr(tools, "_tavily_client", lambda api_key=None: FakeTavily())
    results = tai.search_web("python 3.14", max_results=1)
    assert results == [{"title": "T", "url": "https://u", "content": "C"}]


# --------------------------------------------------------------------------- #
# ToolLoop / Chat (fake chat_completion)
# --------------------------------------------------------------------------- #


def make_completions(script):
    """script: list of Completion to return in order; records the calls."""
    calls = []

    def fake(messages, tools=None, system=None, model=None, provider=None, max_tokens=None):
        calls.append({"messages": [dict(m) for m in messages], "tools": tools,
                      "system": system})
        return script[min(len(calls) - 1, len(script) - 1)]

    return fake, calls


def completion(text="", tool_calls=(), usage=(3, 1)):
    return Completion(text=text, tool_calls=list(tool_calls), usage=Usage(*usage),
                      stop_reason="tool_calls" if tool_calls else "stop")


@tai.tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def test_toolloop_runs_tools_then_answers(monkeypatch):
    fake, calls = make_completions([
        completion(tool_calls=[ToolCall("c1", "multiply", {"a": 3, "b": 7})]),
        completion(text="3 times 7 is 21."),
    ])
    monkeypatch.setattr(llm, "chat_completion", fake)
    loop = tai.ToolLoop(tools=[multiply], system="use tools")
    events = list(loop.run_events("what is 3*7?"))

    assert [e.type for e in events] == ["tool_call", "tool_result", "text"]
    assert events[1].result == "21"  # 3*7 with int args → int → rendered as JSON
    assert events[2].text == "3 times 7 is 21."
    # second call carried the tool traffic back to the model
    second = calls[1]["messages"]
    assert second[1]["tool_calls"][0]["name"] == "multiply"
    assert second[2] == {"role": "tool", "tool_call_id": "c1", "name": "multiply",
                         "content": "21"}
    assert loop.run("again?") in ("3 times 7 is 21.",)


def test_toolloop_unknown_tool_and_error_feed_back(monkeypatch):
    @tai.tool
    def boom(x: int) -> int:
        """Always fails."""
        raise ValueError("nope")

    fake, calls = make_completions([
        completion(tool_calls=[ToolCall("c1", "missing", {}), ToolCall("c2", "boom", {"x": 1})]),
        completion(text="recovered"),
    ])
    monkeypatch.setattr(llm, "chat_completion", fake)
    out = tai.ToolLoop(tools=[boom]).run("go")
    assert out == "recovered"
    tool_messages = [m for m in calls[1]["messages"] if m["role"] == "tool"]
    assert "unknown tool 'missing'" in tool_messages[0]["content"]
    assert "ValueError: nope" in tool_messages[1]["content"]


def test_toolloop_max_iters_raises(monkeypatch):
    fake, _ = make_completions([
        completion(tool_calls=[ToolCall("c", "multiply", {"a": 1, "b": 1})]),
    ])
    monkeypatch.setattr(llm, "chat_completion", fake)
    with pytest.raises(TaiAitutorError) as err:
        tai.ToolLoop(tools=[multiply], max_iters=2).run("loop forever")
    assert "max_iters" in str(err.value)


def test_chat_full_history_and_transcript(monkeypatch):
    fake, calls = make_completions([completion(text="answer one"),
                                    completion(text="answer two")])
    monkeypatch.setattr(llm, "chat_completion", fake)
    chat = tai.Chat(system="tutor mode")
    assert chat.ask("q1") == "answer one"
    assert chat.ask("q2") == "answer two"

    # second call resends the whole first turn (full history)
    sent = calls[1]["messages"]
    assert [m["content"] for m in sent] == ["q1", "answer one", "q2"]
    assert calls[1]["system"] == "tutor mode"
    assert [m["content"] for m in chat.messages] == ["q1", "answer one", "q2", "answer two"]
    assert chat.usage.total_tokens == 8  # 2 calls × (3+1)
    chat.reset()
    assert chat.messages == []


def test_chat_window_memory(monkeypatch):
    fake, calls = make_completions([completion(text=f"a{i}") for i in range(9)])
    monkeypatch.setattr(llm, "chat_completion", fake)
    chat = tai.Chat(history="window", window_turns=2)
    for i in range(4):
        chat.ask(f"q{i}")
    # 4th call: only the last 2 completed turns + the new user message were sent
    sent = calls[3]["messages"]
    assert [m["content"] for m in sent] == ["q1", "a1", "q2", "a2", "q3"]
    # but the full transcript keeps everything
    assert len(chat.messages) == 8


def test_chat_summary_memory_folds_old_turns(monkeypatch):
    fake, calls = make_completions([completion(text=f"a{i}") for i in range(10)])
    monkeypatch.setattr(llm, "chat_completion", fake)
    summarize_calls = []

    def fake_generate(prompt, system=None, model=None, provider=None, **kw):
        summarize_calls.append(prompt)
        return "SUMMARY-OF-OLD-TURNS"

    monkeypatch.setattr(llm, "generate", fake_generate)
    monkeypatch.setattr(chat_module, "n_tokens", lambda text, model=None: 1000)

    chat = tai.Chat(system="base", history="summary", window_turns=2,
                    summarize_after_tokens=5000)
    for i in range(4):  # by turn 4: candidate tokens 3*~2000 > 5000 → summarize
        chat.ask(f"q{i}")

    assert summarize_calls, "summary generation should have fired"
    assert "q0" in summarize_calls[0]
    final_system = calls[-1]["system"]
    assert "SUMMARY-OF-OLD-TURNS" in final_system and "base" in final_system
    sent = calls[-1]["messages"]
    assert sent[0]["content"] != "q0"  # old turns no longer resent verbatim


def test_chat_invalid_history_mode():
    with pytest.raises(TaiAitutorError):
        tai.Chat(history="everything")


def test_chat_tool_traffic_recorded_in_turn(monkeypatch):
    fake, _ = make_completions([
        completion(tool_calls=[ToolCall("c1", "multiply", {"a": 2, "b": 2})]),
        completion(text="4"),
    ])
    monkeypatch.setattr(llm, "chat_completion", fake)
    chat = tai.Chat(tools=[multiply])
    events = list(chat.ask_stream("2*2?"))
    assert [e.type for e in events] == ["tool_call", "tool_result", "text"]
    roles = [m["role"] for m in chat.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]


# --------------------------------------------------------------------------- #
# router
# --------------------------------------------------------------------------- #

ROUTES = {"knowledge": "course questions", "general": "other programming",
          "reject": "off-topic"}


def test_route_valid_and_case_insensitive(monkeypatch):
    monkeypatch.setattr(
        llm, "extract",
        lambda p, s, system=None, model=None, provider=None: s(route="Knowledge",
                                                               reason="course-y"),
    )
    decision = tai.route("What is RAG?", ROUTES)
    assert decision.route == "knowledge"
    assert decision.reason == "course-y"


def test_route_unmatchable_raises(monkeypatch):
    monkeypatch.setattr(
        llm, "extract",
        lambda p, s, system=None, model=None, provider=None: s(route="banana", reason="?"),
    )
    with pytest.raises(TaiAitutorError):
        tai.route("What is RAG?", ROUTES)
    with pytest.raises(TaiAitutorError):
        tai.route("q", {})


def test_match_route_substring():
    assert router._match_route("the knowledge route", ROUTES) == "knowledge"
    assert router._match_route("gen", ROUTES) == "general"
    assert router._match_route("re", ROUTES) == "reject"  # unambiguous substring
    assert router._match_route("ge", ROUTES) is None  # ambiguous: general AND knowledge
    assert router._match_route("REJECT", ROUTES) == "reject"


# --------------------------------------------------------------------------- #
# finetune helpers (no torch needed)
# --------------------------------------------------------------------------- #


def make_qa():
    return QADataset(
        queries={"q1": "What is A?", "q2": "What is B?", "q3": "orphan?"},
        corpus={"c1": "A is alpha.", "c2": "B is beta."},
        relevant_docs={"q1": ["c1"], "q2": ["c2"], "q3": ["missing"]},
    )


def test_training_rows_skip_orphans():
    from tai_aitutor.finetune import _training_rows

    anchors, positives = _training_rows(make_qa())
    assert anchors == ["What is A?", "What is B?"]
    assert positives == ["A is alpha.", "B is beta."]
    with pytest.raises(TaiAitutorError):
        _training_rows(QADataset())


def test_evaluate_embedder_with_fake_local_model(monkeypatch):
    vectors = {
        "A is alpha.": [1.0, 0.0], "B is beta.": [0.0, 1.0],
        "What is A?": [1.0, 0.0], "What is B?": [0.0, 1.0], "orphan?": [0.7, 0.7],
    }

    def fake_embed_local(texts, model_name=None, task="document", batch_size=32):
        return [vectors[t] for t in texts]

    monkeypatch.setattr("tai_aitutor.finetune.embed_local", fake_embed_local)
    report = tai.evaluate_embedder("fake-model", make_qa(), top_k=2)
    assert report.n_queries == 3
    assert abs(report.hit_rate - 2 / 3) < 1e-9   # q3's gold chunk doesn't exist
    assert abs(report.mrr - 2 / 3) < 1e-9        # both hits at rank 1


def test_make_training_pairs_delegates(monkeypatch):
    from tai_aitutor.chunking import Chunk

    captured = {}

    def fake_make_qa_pairs(chunks, n_chunks, questions_per_chunk, **kw):
        captured.update(n_chunks=n_chunks, questions_per_chunk=questions_per_chunk)
        return QADataset(queries={"q": "?"}, corpus={"c": "t"}, relevant_docs={"q": ["c"]})

    monkeypatch.setattr("tai_aitutor.finetune.make_qa_pairs", fake_make_qa_pairs)
    chunks = [Chunk(id=f"c{i}", text=f"t{i}") for i in range(7)]
    qa = tai.make_training_pairs(chunks, questions_per_chunk=3, show_progress=False)
    assert captured == {"n_chunks": 7, "questions_per_chunk": 3}
    assert len(qa) == 1
