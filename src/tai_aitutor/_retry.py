"""Internal: opt-in retry-with-backoff for provider API calls.

Retries are **off by default**. Every provider helper takes ``retries=0``, so a
call is made exactly once unless the caller asks for more — nothing in this
package silently repeats a request you paid for. Section 13 Lesson 2 turns
``retries`` on and explains when a retry is the right answer.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable

_TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def _is_transient(exc: Exception) -> bool:
    """True only for errors worth repeating: rate limits, overload, and timeouts.

    Status codes are checked first because they are unambiguous. Message text is
    only consulted for the two SDK-agnostic markers that have no status code.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and status in _TRANSIENT_STATUS:
        return True
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    msg = str(exc).lower()
    return "rate limit" in msg or "rate_limit" in msg or "resource_exhausted" in msg


def with_retries[T](
    fn: Callable[[], T],
    retries: int = 0,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> T:
    """Call ``fn`` once, plus up to ``retries`` more times on transient errors.

    Args:
        fn: Zero-argument callable making the provider request.
        retries: Extra attempts after the first. ``0`` means no retry at all.
        base_delay: Seconds before the first retry; doubles each attempt.
        max_delay: Ceiling on the backoff delay, in seconds.

    Returns:
        Whatever ``fn`` returns.

    Raises:
        Exception: The last error raised by ``fn``, once retries are exhausted
            or the error is not transient.
    """
    attempts = max(1, retries + 1)
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — filtered by _is_transient
            if not _is_transient(exc) or attempt == attempts - 1:
                raise
            delay = min(max_delay, base_delay * (2**attempt)) * (0.5 + random.random())
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
