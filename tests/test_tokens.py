from __future__ import annotations

import tai_aitutor as tai


def test_n_tokens_basic():
    assert tai.n_tokens("") == 0
    small = tai.n_tokens("hello")
    big = tai.n_tokens("hello " * 200)
    assert 0 < small < big


