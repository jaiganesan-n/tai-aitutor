from __future__ import annotations

import tai_aitutor as tai


def test_n_tokens_basic():
    assert tai.n_tokens("") == 0
    small = tai.n_tokens("hello")
    big = tai.n_tokens("hello " * 200)
    assert 0 < small < big


def test_truncate_bounds():
    text = "word " * 500
    cut = tai.truncate(text, 50)
    assert tai.n_tokens(cut) <= 50 or len(cut) <= 200  # tokenizer path or offline fallback
    assert tai.truncate(text, 0) == ""
    assert tai.truncate("short", 1000) in ("short", "short"[:4000])


def test_estimate_cost_known_model():
    # gpt-5-mini: $0.25 in / $2.00 out per 1M tokens (dated table)
    cost = tai.estimate_cost(1_000_000, 1_000_000, model="gpt-5-mini")
    assert cost == 0.25 + 2.00


def test_estimate_cost_unknown_model_returns_none():
    assert tai.estimate_cost(1000, 1000, model="mystery-model-9000") is None


def test_estimate_cost_uses_configured_model_by_default():
    tai.configure(provider="gemini")  # gemini-2.5-flash: 0.30 / 2.50
    cost = tai.estimate_cost(2_000_000, 0)
    assert cost == 0.60
