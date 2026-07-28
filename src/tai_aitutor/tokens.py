"""Token counting, truncation, and cost estimation.

Built in: "How To Use LLMs via API" (Section 1, reading token usage) and the
chunking lesson (Section 4), where chunk sizes are measured in tokens.

LlamaIndex used tiktoken invisibly inside its splitters; here the tokenizer is
a first-class, visible tool. Counts use OpenAI's ``o200k_base`` vocabulary —
other providers' tokenizers differ slightly, but for chunk budgeting and cost
estimates a single consistent ruler is what matters.

Offline-safe: if the tokenizer vocabulary can't be downloaded (no network),
falls back to a ~4-characters-per-token estimate with a one-time warning.
"""

from __future__ import annotations

import warnings

from . import config as _cfg

__all__ = ["n_tokens", "truncate", "estimate_cost"]

_encoding = None
_encoding_failed = False


def _get_encoding():
    global _encoding, _encoding_failed
    if _encoding is None and not _encoding_failed:
        try:
            import tiktoken

            _encoding = tiktoken.get_encoding("o200k_base")
        except Exception:
            _encoding_failed = True
            warnings.warn(
                "tiktoken vocabulary unavailable (offline?) — token counts are "
                "approximate (len/4) until it can be downloaded.",
                stacklevel=3,
            )
    return _encoding


def n_tokens(text: str, model: str | None = None) -> int:
    """Number of tokens in ``text`` (o200k_base; ``model`` reserved for future maps)."""
    enc = _get_encoding()
    if enc is None:
        return max(1, len(text) // 4) if text else 0
    return len(enc.encode(text))


def truncate(text: str, max_tokens: int, model: str | None = None) -> str:
    """Cut ``text`` to at most ``max_tokens`` tokens (used by context budgeting)."""
    if max_tokens <= 0:
        return ""
    enc = _get_encoding()
    if enc is None:
        return text[: max_tokens * 4]
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return text
    return enc.decode(ids[:max_tokens])


def estimate_cost(
    input_tokens: int,
    output_tokens: int = 0,
    model: str | None = None,
) -> float | None:
    """USD cost of a call at the package's dated price table (config.MODEL_PRICES).

    Returns ``None`` (never a guess) for models missing from the table — check
    the provider's pricing page and consider adding a row.
    """
    model = model or _cfg.get_config().chat_model
    prices = _cfg.MODEL_PRICES.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
