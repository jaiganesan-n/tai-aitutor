"""embeddings tests with fake clients — no network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import tai_aitutor as tai
from tai_aitutor import embeddings, llm
from tai_aitutor.errors import EmbeddingsNotAvailableError


class FakeGeminiModels:
    def __init__(self):
        self.calls = []

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        n = len(kwargs["contents"])
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3]) for _ in range(n)])


class FakeOpenAIEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        n = len(kwargs["input"])
        # deliberately shuffled indexes to prove we re-sort
        data = [SimpleNamespace(index=i, embedding=[float(i), 0.0]) for i in reversed(range(n))]
        return SimpleNamespace(data=data)


@pytest.fixture
def fake_gemini(monkeypatch):
    client = SimpleNamespace(models=FakeGeminiModels())
    monkeypatch.setattr(llm, "_client_gemini", lambda api_key=None: client)
    return client


@pytest.fixture
def fake_openai(monkeypatch):
    client = SimpleNamespace(embeddings=FakeOpenAIEmbeddings())
    monkeypatch.setattr(llm, "_client_openai", lambda api_key=None, base_url=None: client)
    return client


def test_single_string_returns_single_vector(fake_gemini):
    vec = tai.embed("hello", task="query")
    assert isinstance(vec[0], float)
    call = fake_gemini.models.calls[0]
    assert call["config"].task_type == "RETRIEVAL_QUERY"


def test_list_returns_list_of_vectors(fake_gemini):
    out = tai.embed(["a", "b", "c"])
    assert len(out) == 3 and all(len(v) == 3 for v in out)
    assert fake_gemini.models.calls[0]["config"].task_type == "RETRIEVAL_DOCUMENT"


def test_gemini_batching_respects_api_cap(fake_gemini):
    tai.embed([f"text {i}" for i in range(250)], batch_size=100)
    sizes = [len(c["contents"]) for c in fake_gemini.models.calls]
    assert sizes == [100, 100, 50]


def test_openai_branch_resorts_by_index(fake_openai):
    tai.configure(provider="openai")  # pairs with openai embeddings
    out = tai.embed(["a", "b", "c"])
    assert [v[0] for v in out] == [0.0, 1.0, 2.0]  # re-sorted despite shuffled response
    assert fake_openai.embeddings.calls[0]["model"] == "text-embedding-3-small"


def test_invalid_task_raises(fake_gemini):
    with pytest.raises(ValueError):
        tai.embed("hello", task="index")


def test_anthropic_embeddings_clear_error():
    with pytest.raises(EmbeddingsNotAvailableError) as err:
        tai.embed("hello", provider="anthropic")
    assert "Gemini" in str(err.value)


def test_cohere_input_type_mapping(monkeypatch):
    calls = []

    class FakeCohere:
        def embed(self, **kwargs):
            calls.append(kwargs)
            floats = [[0.5, 0.5]] * len(kwargs["texts"])
            return SimpleNamespace(embeddings=SimpleNamespace(float_=floats))

    monkeypatch.setattr(embeddings, "_client_cohere", lambda api_key=None: FakeCohere())
    out = tai.embed_cohere(["doc one", "doc two"], task="document")
    assert len(out) == 2
    assert calls[0]["input_type"] == "search_document"
    assert calls[0]["model"] == "embed-v4.0"

    tai.embed_cohere("a query", task="query")
    assert calls[1]["input_type"] == "search_query"


def test_local_e5_prefixes(monkeypatch):
    captured = {}

    class FakeModel:
        def encode(self, items, **kwargs):
            captured["items"] = items
            return [[0.0] * 4 for _ in items]

    monkeypatch.setattr(embeddings, "_get_local_model", lambda name: FakeModel())
    tai.embed_local(["some passage"], model_name="intfloat/e5-small-v2", task="document")
    assert captured["items"] == ["passage: some passage"]
    tai.embed_local("find me", model_name="intfloat/e5-small-v2", task="query")
    assert captured["items"] == ["query: find me"]
    # non-e5 models get no prefix
    tai.embed_local("plain", model_name="BAAI/bge-small-en-v1.5", task="query")
    assert captured["items"] == ["plain"]
