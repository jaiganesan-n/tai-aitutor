from __future__ import annotations

import pytest

import tai_aitutor as tai
from tai_aitutor import config as cfg
from tai_aitutor.errors import MissingKeyError, UnsupportedProviderError


def test_defaults_are_gemini():
    c = tai.get_config()
    assert c.provider == "gemini"
    assert c.chat_model == "gemini-3.6-flash"
    assert c.embed_provider == "gemini"
    assert c.embed_model == "gemini-embedding-001"


def test_openai_pairs_with_openai_embeddings():
    c = tai.configure(provider="openai")
    assert c.chat_model == "gpt-5.6-luna"
    assert c.embed_provider == "openai"
    assert c.embed_model == "text-embedding-3-small"


def test_anthropic_pairs_with_gemini_embeddings():
    c = tai.configure(provider="anthropic")
    assert c.chat_model == "claude-sonnet-5"
    assert c.embed_provider == "gemini"  # Decision 2: Anthropic has no embeddings API


def test_compat_providers_get_base_urls():
    for provider in ("together", "deepseek", "perplexity", "ollama"):
        c = tai.configure(provider=provider)
        assert c.base_url == cfg.BASE_URLS[provider]
        assert c.chat_model  # every registry provider has a default model


def test_unknown_provider_raises():
    with pytest.raises(UnsupportedProviderError):
        tai.configure(provider="not-a-provider")


def test_unknown_provider_allowed_with_base_url():
    c = tai.configure(provider="groq", base_url="https://api.groq.com/openai/v1",
                      chat_model="llama-3.3-70b")
    assert c.base_url.endswith("/v1")
    assert c.chat_model == "llama-3.3-70b"


def test_unknown_provider_with_base_url_but_no_model_raises():
    with pytest.raises(UnsupportedProviderError):
        tai.configure(provider="groq", base_url="https://api.groq.com/openai/v1")


def test_resolve_per_call_provider_override_leaves_global_untouched():
    tai.configure(provider="gemini")
    call_cfg = cfg.resolve(provider="openai")
    assert call_cfg.provider == "openai"
    assert call_cfg.chat_model == "gpt-5.6-luna"
    assert tai.get_config().provider == "gemini"  # global unchanged


def test_resolve_model_override_only():
    tai.configure(provider="gemini")
    call_cfg = cfg.resolve(chat_model="gemini-2.5-pro")
    assert call_cfg.chat_model == "gemini-2.5-pro"
    assert call_cfg.provider == "gemini"


def test_require_keys_lists_all_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(MissingKeyError) as err:
        tai.require_keys("OPENAI_API_KEY", "COHERE_API_KEY", "GOOGLE_API_KEY")
    assert "OPENAI_API_KEY" in str(err.value)
    assert "COHERE_API_KEY" in str(err.value)
    assert "GOOGLE_API_KEY" not in str(err.value)  # that one is set


def test_setup_notebook_local_loads_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("TAVILY_API_KEY=tvly-from-dotenv\n")
    in_colab = tai.setup_notebook(required_keys=("TAVILY_API_KEY",), dotenv_path=str(env_file))
    assert in_colab is False
    import os

    assert os.environ["TAVILY_API_KEY"] == "tvly-from-dotenv"


def test_bad_embed_provider_raises():
    with pytest.raises(UnsupportedProviderError):
        tai.configure(provider="gemini", embed_provider="anthropic")
