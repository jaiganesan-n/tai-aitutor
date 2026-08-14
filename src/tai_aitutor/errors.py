"""Exceptions for tai-aitutor.

Every error message should tell a student exactly what to do next.

All of these subclass :class:`ValueError`, so the taught ``except ValueError``
catches them — the notebook cells raise a bare ``ValueError`` for an unknown
provider, and the package must be catchable the same way.
"""

from __future__ import annotations


class TaiAitutorError(ValueError):
    """Base class for all tai-aitutor errors. Subclasses ``ValueError``."""


class UnsupportedProviderError(TaiAitutorError):
    """Raised when a provider name is not in the registry and no base_url was given."""


class ProviderNotInstalledError(TaiAitutorError):
    """Raised when a provider SDK is missing. Message includes the pip extra to install."""


class MissingKeyError(TaiAitutorError):
    """Raised when a required API key is absent from the environment."""


class EmbeddingsNotAvailableError(TaiAitutorError):
    """Raised when the selected provider has no embeddings API (e.g. Anthropic)."""


class StructuredOutputError(TaiAitutorError):
    """Raised when a model response cannot be parsed into the requested schema."""
