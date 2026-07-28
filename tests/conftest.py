"""Shared fixtures: clean config + fake API keys, no network anywhere."""

from __future__ import annotations

import pytest

from tai_aitutor import config as cfg
from tai_aitutor import llm


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """Reset global config/clients and provide dummy keys for every test."""
    cfg._reset()
    llm._reset_clients()
    for env in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "COHERE_API_KEY",
                "TOGETHER_API_KEY", "DEEPSEEK_API_KEY", "PERPLEXITY_API_KEY"):
        monkeypatch.setenv(env, f"test-{env.lower()}")
    yield
    cfg._reset()
    llm._reset_clients()
