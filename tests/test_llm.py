"""llm tests with fake SDK clients — no network, no real SDKs required."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

import tai_aitutor as tai
from tai_aitutor import llm
from tai_aitutor.errors import MissingKeyError, StructuredOutputError

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeGeminiModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            text="gemini says hi",
            parsed=None,
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5),
        )


class FakeOpenAIResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text="openai says hi",
            usage=SimpleNamespace(input_tokens=7, output_tokens=3),
        )

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        schema = kwargs["text_format"]
        return SimpleNamespace(output_parsed=schema(answer="parsed", score=1.0))


class FakeChatCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content="compat says hi")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2),
        )


class FakeAnthropicMessages:
    def __init__(self, blocks=None):
        self.calls = []
        self._blocks = blocks

    def create(self, **kwargs):
        self.calls.append(kwargs)
        blocks = self._blocks or [SimpleNamespace(type="text", text="claude says hi")]
        return SimpleNamespace(
            content=blocks,
            usage=SimpleNamespace(input_tokens=6, output_tokens=4),
        )


@pytest.fixture
def fake_clients(monkeypatch):
    gemini = SimpleNamespace(models=FakeGeminiModels())
    openai_native = SimpleNamespace(responses=FakeOpenAIResponses(), chat=SimpleNamespace(completions=FakeChatCompletions()))
    anthropic = SimpleNamespace(messages=FakeAnthropicMessages())

    monkeypatch.setattr(llm, "_client_gemini", lambda api_key=None: gemini)
    monkeypatch.setattr(llm, "_client_openai", lambda api_key=None, base_url=None: openai_native)
    monkeypatch.setattr(llm, "_client_anthropic", lambda api_key=None: anthropic)
    return SimpleNamespace(gemini=gemini, openai=openai_native, anthropic=anthropic)


# --------------------------------------------------------------------------- #
# generate() dispatch
# --------------------------------------------------------------------------- #


def test_generate_gemini_default(fake_clients):
    tai.configure(provider="gemini")
    out = tai.generate("hello", system="be brief")
    assert out == "gemini says hi"
    call = fake_clients.gemini.models.calls[0]
    assert call["model"] == "gemini-3.6-flash"
    assert call["config"].system_instruction == "be brief"


def test_generate_openai_uses_responses_api(fake_clients):
    tai.configure(provider="openai")
    out = tai.generate("hello", system="sys", reasoning_effort="minimal", max_tokens=64)
    assert out == "openai says hi"
    call = fake_clients.openai.responses.calls[0]
    assert call["model"] == "gpt-5.6-luna"
    assert call["instructions"] == "sys"
    assert call["reasoning"] == {"effort": "minimal"}
    assert call["max_output_tokens"] == 64


def test_generate_anthropic_sets_required_max_tokens(fake_clients):
    tai.configure(provider="anthropic")
    out = tai.generate("hello")
    assert out == "claude says hi"
    call = fake_clients.anthropic.messages.calls[0]
    assert call["model"] == "claude-sonnet-5"
    assert call["max_tokens"] == 4096  # Anthropic requires it; we default it
    assert "system" not in call  # not passed when None


def test_generate_compat_uses_chat_completions_with_base_url(fake_clients):
    tai.configure(provider="together")
    out = tai.generate("hello", system="sys")
    assert out == "compat says hi"
    call = fake_clients.openai.chat.completions.calls[0]
    assert call["messages"][0] == {"role": "system", "content": "sys"}
    assert call["messages"][1]["role"] == "user"


def test_per_call_provider_override_does_not_touch_global(fake_clients):
    tai.configure(provider="gemini")
    out = tai.generate("hello", provider="openai")
    assert out == "openai says hi"
    assert tai.get_config().provider == "gemini"


def test_kwarg_typo_raises_type_error(fake_clients):
    # The old LlamaIndex path silently accepted additional_kwrgs={...}; we must not.
    with pytest.raises(TypeError):
        tai.generate("hello", additional_kwrgs={"reasoning_effort": "minimal"})


def test_compat_without_key_raises_missing_key(monkeypatch, fake_clients):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    tai.configure(provider="deepseek")
    with pytest.raises(MissingKeyError) as err:
        tai.generate("hello")
    assert "DEEPSEEK_API_KEY" in str(err.value)


def test_ollama_needs_no_key(monkeypatch, fake_clients):
    tai.configure(provider="ollama")
    assert tai.generate("hello") == "compat says hi"


# --------------------------------------------------------------------------- #
# extract() structured outputs
# --------------------------------------------------------------------------- #


class Verdict(BaseModel):
    answer: str
    score: float


def test_extract_openai_parse(fake_clients):
    tai.configure(provider="openai")
    result = tai.extract("judge this", Verdict)
    assert isinstance(result, Verdict)
    assert result.answer == "parsed"


def test_extract_gemini_falls_back_to_json_text(fake_clients, monkeypatch):
    tai.configure(provider="gemini")

    def generate_content(**kwargs):
        return SimpleNamespace(text='```json\n{"answer": "from-json", "score": 0.5}\n```', parsed=None)

    monkeypatch.setattr(fake_clients.gemini.models, "generate_content", generate_content)
    result = tai.extract("judge this", Verdict)
    assert result.answer == "from-json"


def test_extract_anthropic_tool_use(monkeypatch, fake_clients):
    blocks = [SimpleNamespace(type="tool_use", input={"answer": "via-tool", "score": 0.9})]
    fake_clients.anthropic.messages._blocks = blocks
    tai.configure(provider="anthropic")
    result = tai.extract("judge this", Verdict)
    assert result.answer == "via-tool"
    call = fake_clients.anthropic.messages.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "emit_result"}


def test_extract_compat_json_fallback(monkeypatch, fake_clients):
    tai.configure(provider="deepseek")
    monkeypatch.setattr(
        llm, "generate", lambda *a, **k: 'Here you go: {"answer": "compat", "score": 1.0}'
    )
    result = tai.extract("judge this", Verdict)
    assert result.answer == "compat"


def test_extract_compat_repair_retry_then_error(monkeypatch, fake_clients):
    tai.configure(provider="deepseek")
    monkeypatch.setattr(llm, "generate", lambda *a, **k: "not json at all")
    with pytest.raises(StructuredOutputError):
        tai.extract("judge this", Verdict)


# --------------------------------------------------------------------------- #
# ask_batch
# --------------------------------------------------------------------------- #


def test_ask_batch_preserves_order(monkeypatch):
    monkeypatch.setattr(llm, "generate", lambda p, s=None, **k: f"answer:{p}")
    out = tai.ask_batch([f"q{i}" for i in range(20)], concurrency=5)
    assert out == [f"answer:q{i}" for i in range(20)]


def test_ask_batch_propagates_errors(monkeypatch):
    def boom(p, s=None, **k):
        if p == "q3":
            raise RuntimeError("provider down")
        return "ok"

    monkeypatch.setattr(llm, "generate", boom)
    with pytest.raises(RuntimeError):
        tai.ask_batch([f"q{i}" for i in range(5)])
