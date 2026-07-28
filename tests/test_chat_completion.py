"""chat_completion provider conversions, against fake SDK clients."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import tai_aitutor as tai
from tai_aitutor import llm

CONVO = [
    {"role": "system", "content": "be helpful"},
    {"role": "user", "content": "what is 3*7?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "name": "multiply", "arguments": {"a": 3, "b": 7}}]},
    {"role": "tool", "tool_call_id": "call_1", "name": "multiply", "content": "21"},
]

TOOLS = [{"name": "multiply", "description": "Multiply two numbers.",
          "parameters": {"type": "object", "properties": {"a": {"type": "number"},
                                                          "b": {"type": "number"}},
                         "required": ["a", "b"]}}]


# --------------------------------------------------------------------------- #
# OpenAI-compatible branch
# --------------------------------------------------------------------------- #


def test_compat_round_trip(monkeypatch):
    captured = {}

    class FakeChat:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(content="21.", tool_calls=None)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
            )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeChat()))
    monkeypatch.setattr(llm, "_client_openai", lambda api_key=None, base_url=None: fake)
    tai.configure(provider="deepseek")
    completion = tai.chat_completion(CONVO, tools=TOOLS)

    sent = captured["messages"]
    assert sent[0] == {"role": "system", "content": "be helpful"}  # extracted + re-injected
    assert sent[2]["tool_calls"][0]["function"]["name"] == "multiply"
    assert json.loads(sent[2]["tool_calls"][0]["function"]["arguments"]) == {"a": 3, "b": 7}
    assert sent[3] == {"role": "tool", "tool_call_id": "call_1", "content": "21"}
    assert captured["tools"][0]["function"]["name"] == "multiply"
    assert completion.text == "21." and completion.tool_calls == []
    assert completion.stop_reason == "stop"
    assert completion.usage.total_tokens == 12


def test_compat_parses_tool_calls(monkeypatch):
    class FakeChat:
        def create(self, **kwargs):
            call = SimpleNamespace(
                id="c9", function=SimpleNamespace(name="multiply", arguments='{"a": 2, "b": 4}')
            )
            message = SimpleNamespace(content=None, tool_calls=[call])
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
                usage=None,
            )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeChat()))
    monkeypatch.setattr(llm, "_client_openai", lambda api_key=None, base_url=None: fake)
    tai.configure(provider="together")
    completion = tai.chat_completion([{"role": "user", "content": "2*4?"}], tools=TOOLS)
    assert completion.stop_reason == "tool_calls"
    assert completion.tool_calls == [llm.ToolCall(id="c9", name="multiply",
                                                  arguments={"a": 2, "b": 4})]


# --------------------------------------------------------------------------- #
# OpenAI native (Responses API) branch
# --------------------------------------------------------------------------- #


def test_openai_native_items_and_function_call(monkeypatch):
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            call = SimpleNamespace(type="function_call", call_id="fc1", name="multiply",
                                   arguments='{"a": 1, "b": 2}')
            return SimpleNamespace(output_text="", output=[call],
                                   usage=SimpleNamespace(input_tokens=5, output_tokens=1))

    fake = SimpleNamespace(responses=FakeResponses())
    monkeypatch.setattr(llm, "_client_openai", lambda api_key=None, base_url=None: fake)
    tai.configure(provider="openai")
    completion = tai.chat_completion(CONVO, tools=TOOLS)

    assert captured["instructions"] == "be helpful"
    items = captured["input"]
    assert items[0] == {"role": "user", "content": "what is 3*7?"}
    assert items[1]["type"] == "function_call" and items[1]["call_id"] == "call_1"
    assert items[2] == {"type": "function_call_output", "call_id": "call_1", "output": "21"}
    assert captured["tools"][0] == {"type": "function", **TOOLS[0]}
    assert completion.tool_calls[0].arguments == {"a": 1, "b": 2}
    assert completion.stop_reason == "tool_calls"


# --------------------------------------------------------------------------- #
# Anthropic branch
# --------------------------------------------------------------------------- #


def test_anthropic_blocks_merge_and_tool_result(monkeypatch):
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            blocks = [SimpleNamespace(type="text", text="It is 21."),
                      SimpleNamespace(type="tool_use", id="t1", name="multiply",
                                      input={"a": 5, "b": 5})]
            return SimpleNamespace(content=blocks, stop_reason="tool_use",
                                   usage=SimpleNamespace(input_tokens=8, output_tokens=3))

    fake = SimpleNamespace(messages=FakeMessages())
    monkeypatch.setattr(llm, "_client_anthropic", lambda api_key=None: fake)
    tai.configure(provider="anthropic")
    completion = tai.chat_completion(CONVO, tools=TOOLS)

    assert captured["system"] == "be helpful"
    sent = captured["messages"]
    assert sent[1]["role"] == "assistant"
    assert sent[1]["content"][0]["type"] == "tool_use"
    assert sent[2]["role"] == "user"
    assert sent[2]["content"][0] == {"type": "tool_result", "tool_use_id": "call_1",
                                     "content": "21"}
    assert captured["tools"][0]["input_schema"] == TOOLS[0]["parameters"]
    assert completion.text == "It is 21."
    assert completion.tool_calls[0].name == "multiply"


def test_anthropic_merges_adjacent_user_messages(monkeypatch):
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")],
                                   usage=None)

    fake = SimpleNamespace(messages=FakeMessages())
    monkeypatch.setattr(llm, "_client_anthropic", lambda api_key=None: fake)
    tai.configure(provider="anthropic")
    convo = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "name": "f", "arguments": {}},
            {"id": "b", "name": "g", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "a", "name": "f", "content": "r1"},
        {"role": "tool", "tool_call_id": "b", "name": "g", "content": "r2"},
    ]
    tai.chat_completion(convo)
    sent = captured["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]  # results merged
    assert len(sent[2]["content"]) == 2


# --------------------------------------------------------------------------- #
# Gemini branch
# --------------------------------------------------------------------------- #


def test_gemini_contents_and_function_call(monkeypatch):
    pytest.importorskip("google.genai")
    captured = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            call = SimpleNamespace(id=None, name="multiply", args={"a": 6, "b": 7})
            parts = [SimpleNamespace(function_call=call, text=None)]
            content = SimpleNamespace(parts=parts)
            return SimpleNamespace(
                candidates=[SimpleNamespace(content=content)],
                usage_metadata=SimpleNamespace(prompt_token_count=4, candidates_token_count=2),
            )

    fake = SimpleNamespace(models=FakeModels())
    monkeypatch.setattr(llm, "_client_gemini", lambda api_key=None: fake)
    tai.configure(provider="gemini")
    completion = tai.chat_completion(CONVO, tools=TOOLS)

    contents = captured["contents"]
    assert contents[0].role == "user"
    assert contents[1].role == "model"
    assert contents[1].parts[0].function_call.name == "multiply"
    assert contents[2].role == "user"  # function_response travels as user content
    assert contents[2].parts[0].function_response.response == {"result": "21"}
    config = captured["config"]
    assert config.system_instruction == "be helpful"
    assert config.tools[0].function_declarations[0].name == "multiply"
    assert completion.tool_calls[0] == llm.ToolCall(id="call_0", name="multiply",
                                                    arguments={"a": 6, "b": 7})
