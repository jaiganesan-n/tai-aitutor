"""Token counting, truncation, and cost estimation.

Built in: "How To Use LLMs via API" (Section 1, reading token usage) and the
chunking lesson (Section 4), where chunk sizes are measured in tokens.

LlamaIndex used tiktoken invisibly inside its splitters; here the tokenizer is
a first-class, visible tool. Counts use the ``cl100k_base`` vocabulary — the
one the chunking lesson sets up (``ENC = tiktoken.get_encoding("cl100k_base")``)
— so a chunk measured in the notebook and the same chunk measured here come out
identical. Other providers' tokenizers differ slightly; one consistent ruler is
what chunk budgeting needs.

Offline-safe: if the tokenizer vocabulary can't be downloaded (no network),
falls back to a ~4-characters-per-token estimate with a one-time warning.
"""

from __future__ import annotations

import warnings

__all__ = ["n_tokens"]

_encoding = None
_encoding_failed = False


def _get_encoding():
    global _encoding, _encoding_failed
    if _encoding is None and not _encoding_failed:
        try:
            import tiktoken

            _encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoding_failed = True
            warnings.warn(
                "tiktoken vocabulary unavailable (offline?) — token counts are "
                "approximate (len/4) until it can be downloaded.",
                stacklevel=3,
            )
    return _encoding


def n_tokens(text: str, model: str | None = None) -> int:
    """Count the tokens in ``text``.

    Args:
        text: The text to measure.
        model: Reserved for future per-model vocabularies; ignored today.

    Returns:
        The token count under ``cl100k_base``, or a ~4-characters-per-token
        estimate if the vocabulary could not be downloaded.
    """
    enc = _get_encoding()
    if enc is None:
        return max(1, len(text) // 4) if text else 0
    return len(enc.encode(text))


