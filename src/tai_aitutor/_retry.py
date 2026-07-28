"""Internal: retry-with-backoff for provider API calls.

A production concern the notebooks skip; the package carries it so students
don't hit free-tier rate limits mid-lesson.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable

_TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
_TRANSIENT_MARKERS = (
    "rate limit",
    "rate_limit",
    "resource_exhausted",
    "resource has been exhausted",
    "overloaded",
    "temporarily",
    "timeout",
    "timed out",
    "connection",
    "unavailable",
    "quota",
    "503",
    "429",
)


def _is_transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and status in _TRANSIENT_STATUS:
        return True
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_MARKERS)


def with_retries[T](
    fn: Callable[[], T],
    attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> T:
    """Call ``fn``; on transient errors, retry with exponential backoff + jitter."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — filtered by _is_transient
            if not _is_transient(exc) or attempt == attempts - 1:
                raise
            last = exc
            delay = min(max_delay, base_delay * (2**attempt)) * (0.5 + random.random())
            time.sleep(delay)
    raise last  # pragma: no cover — unreachable
