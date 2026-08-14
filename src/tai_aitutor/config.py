"""Course-wide configuration: provider selection, model defaults, notebook setup.

Built in: "How To Use LLMs via API" (Section 1) and the shared three-provider
setup cell (Decisions 2 and 3 of the July 2026 course update).

This module replaces LlamaIndex's global ``Settings`` object. Instead of::

    Settings.llm = OpenAI(model="gpt-5-mini")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

you write::

    from tai_aitutor import configure
    configure(provider="openai")            # or the PROVIDER dropdown value

Every function in the package also accepts explicit ``provider=`` / ``model=``
overrides, so comparison lessons never need to touch global state.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace

from .errors import MissingKeyError, UnsupportedProviderError

# --------------------------------------------------------------------------- #
# Provider registry — ONE place to update when models change.
# --------------------------------------------------------------------------- #

#: Providers with a native SDK branch in tai_aitutor.llm / tai_aitutor.embeddings.
NATIVE_PROVIDERS = ("gemini", "openai", "anthropic")

#: Providers served through an OpenAI-compatible endpoint (openai SDK + base_url).
OPENAI_COMPATIBLE_PROVIDERS = ("together", "deepseek", "perplexity", "ollama")

PROVIDERS = NATIVE_PROVIDERS + OPENAI_COMPATIBLE_PROVIDERS

#: Default chat model per provider. The package carries no model defaults of its
#: own: these are Decision 2's, as quoted in the strip spec, as of 2026-08-13.
# TODO: [NEEDS UPDATE — confirm the three native chat defaults | source: internal doc
# (course_update_plan.md v4, Decision 2) | the strip spec says Decision 2 mandates
# gemini-3.5-flash-lite and claude-haiku-4-5, but the swept notebooks (02-Basic_RAG cell 6)
# set MODELS = {"gemini": "gemini-3.6-flash", "openai": "gpt-5.6-luna",
# "anthropic": "claude-sonnet-5"}. The spec's values are applied below per the
# spec-is-primary decision; the notebooks disagree and need reconciling.]
CHAT_MODEL_DEFAULTS: dict[str, str] = {
    "gemini": "gemini-3.5-flash-lite",
    "openai": "gpt-5.6-luna",
    "anthropic": "claude-haiku-4-5",
    "together": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "deepseek": "deepseek-chat",
    "perplexity": "sonar",
    "ollama": "llama3.2",
}

#: Default embedding model per embedding provider (Decision 2: Gemini is the course default).
EMBED_MODEL_DEFAULTS: dict[str, str] = {
    "gemini": "gemini-embedding-001",
    "openai": "text-embedding-3-small",
    "cohere": "embed-v4.0",
    "local": "BAAI/bge-small-en-v1.5",
}

#: Base URLs for the OpenAI-compatible providers.
BASE_URLS: dict[str, str] = {
    "together": "https://api.together.ai/v1",  # per Together's current docs (2026-07)
    "deepseek": "https://api.deepseek.com",
    "perplexity": "https://api.perplexity.ai",
    "ollama": "http://localhost:11434/v1",
}

#: Environment variable that holds each provider's API key (None = no key needed).
API_KEY_ENV: dict[str, str | None] = {
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "cohere": "COHERE_API_KEY",
    "together": "TOGETHER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "ollama": None,
}

#: Keys setup_notebook() tries to load from Colab Secrets (quietly skipping absent ones).
_KNOWN_KEYS = (
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "COHERE_API_KEY",
    "TOGETHER_API_KEY",
    "DEEPSEEK_API_KEY",
    "PERPLEXITY_API_KEY",
    "TAVILY_API_KEY",
    "FIRECRAWL_API_KEY",
    "HF_TOKEN",
)


# --------------------------------------------------------------------------- #
# Config object
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Config:
    """Resolved course configuration. Create via :func:`configure`."""

    provider: str = "gemini"
    chat_model: str = CHAT_MODEL_DEFAULTS["gemini"]
    embed_provider: str = "gemini"
    embed_model: str = EMBED_MODEL_DEFAULTS["gemini"]
    base_url: str | None = None
    api_key: str | None = None

    def __repr__(self) -> str:  # friendly notebook printing
        return (
            f"Config(provider={self.provider!r}, chat_model={self.chat_model!r}, "
            f"embed_provider={self.embed_provider!r}, embed_model={self.embed_model!r}"
            + (f", base_url={self.base_url!r}" if self.base_url else "")
            + ")"
        )


_config: Config | None = None


def _build(
    provider: str = "gemini",
    chat_model: str | None = None,
    embed_provider: str | None = None,
    embed_model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Config:
    """Pure resolver: provider name + overrides → a fully-resolved Config."""
    provider = provider.lower().strip()
    if provider not in PROVIDERS and base_url is None:
        raise UnsupportedProviderError(
            f"Unknown provider {provider!r}. Pick one of {', '.join(PROVIDERS)}, "
            "or pass base_url=... for any OpenAI-compatible endpoint."
        )

    if embed_provider is None:
        embed_provider = "openai" if provider == "openai" else "gemini"
    embed_provider = embed_provider.lower().strip()
    if embed_provider not in EMBED_MODEL_DEFAULTS:
        raise UnsupportedProviderError(
            f"Unknown embed_provider {embed_provider!r}. "
            f"Pick one of {', '.join(EMBED_MODEL_DEFAULTS)}. "
            "(Anthropic has no embeddings API — use Gemini or OpenAI embeddings.)"
        )

    resolved_chat = chat_model or CHAT_MODEL_DEFAULTS.get(provider, "")
    if not resolved_chat:
        raise UnsupportedProviderError(
            f"No default chat model known for provider {provider!r} — pass chat_model=..."
        )
    return Config(
        provider=provider,
        chat_model=resolved_chat,
        embed_provider=embed_provider,
        embed_model=embed_model or EMBED_MODEL_DEFAULTS[embed_provider],
        base_url=base_url or BASE_URLS.get(provider),
        api_key=api_key,
    )


def configure(
    provider: str = "gemini",
    chat_model: str | None = None,
    embed_provider: str | None = None,
    embed_model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Config:
    """Set the course-wide defaults (the package's only global state).

    Parameters
    ----------
    provider:
        ``"gemini"`` (course default), ``"openai"``, ``"anthropic"``, or an
        OpenAI-compatible provider (``"together"``, ``"deepseek"``,
        ``"perplexity"``, ``"ollama"``). Any other name is allowed **if**
        ``base_url`` is given (the escape hatch every OpenAI-compatible host fits).
    chat_model / embed_model:
        Override the per-provider defaults from the registry tables.
    embed_provider:
        ``"gemini"`` | ``"openai"`` | ``"cohere"`` | ``"local"``. Default: Gemini
        (Decision 2) — except when ``provider="openai"``, which pairs with OpenAI
        embeddings so a single API key covers the whole notebook. Anthropic has no
        embeddings API, so Anthropic chat pairs with Gemini embeddings.

    Args:
        provider: Chat provider — "gemini", "openai", "anthropic", or an
            OpenAI-compatible name ("together", "deepseek", "perplexity", "ollama").
        chat_model: Model id; defaults to the provider's course-standard model.
        embed_provider: Embedding provider; defaults to the chat provider, or to
            Gemini when the chat provider has no embeddings API.
        embed_model: Embedding model id; defaults to the provider's course standard.
        base_url: OpenAI-compatible endpoint, for a provider not in the registry.
        api_key: Key for this provider, overriding the environment variable.

    Returns:
        The resolved :class:`Config`, which is also stored as the module default.

    Raises:
        ValueError: The provider or embedding provider is not one this package knows.
    """
    global _config
    _config = _build(provider, chat_model, embed_provider, embed_model, base_url, api_key)
    return _config


def get_config() -> Config:
    """Return the current config, initialising course defaults (Gemini) on first use.

    Returns:
        The current :class:`Config`, initialising the Gemini defaults on first use.
    """
    global _config
    if _config is None:
        _config = _build()
    return _config


def _reset() -> None:
    """Testing hook: forget the global config."""
    global _config
    _config = None


def resolve(
    provider: str | None = None,
    chat_model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Config:
    """Current config with per-call overrides applied (never mutates the global).

    A ``provider=`` override resolves that provider's own registry defaults;
    other overrides replace individual fields on the current config.
    """
    cfg = get_config()
    if provider and provider.lower().strip() != cfg.provider:
        return _build(provider, chat_model=chat_model, base_url=base_url, api_key=api_key)
    updates = {
        k: v
        for k, v in {"chat_model": chat_model, "base_url": base_url, "api_key": api_key}.items()
        if v is not None
    }
    return replace(cfg, **updates) if updates else cfg


# --------------------------------------------------------------------------- #
# Notebook environment setup (Decision 3: one cell, Colab AND local)
# --------------------------------------------------------------------------- #


def in_colab() -> bool:
    """True when running inside Google Colab.

    Returns:
        ``True`` when ``google.colab`` is importable in this runtime.
    """
    return "google.colab" in sys.modules


def setup_notebook(
    required_keys: tuple[str, ...] | list[str] = (),
    dotenv_path: str | None = None,
) -> bool:
    """The environment half of the course's shared setup cell.

    In Colab: copies known API keys from Colab Secrets (``userdata``) into
    ``os.environ`` (absent secrets are skipped silently).
    Locally: loads ``.env`` via python-dotenv.
    Then verifies ``required_keys`` are present (:func:`require_keys`).

    Returns ``True`` in Colab, ``False`` locally — the same ``IN_COLAB`` flag
    the notebooks use for their guarded install branches.

    Args:
        required_keys: Environment variables that must be present afterwards.
        dotenv_path: Explicit ``.env`` path; defaults to the usual search.

    Returns:
        ``True`` when running inside Colab, ``False`` locally.

    Raises:
        ValueError: A key in ``required_keys`` is still missing after loading.
    """
    colab = in_colab()
    if colab:
        from google.colab import userdata  # type: ignore[import-not-found]

        for key in {*_KNOWN_KEYS, *required_keys}:
            if os.environ.get(key):
                continue
            try:
                value = userdata.get(key)
            except Exception:
                continue  # secret not set or not granted to this notebook
            if value:
                os.environ[key] = value
    else:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=dotenv_path)

    if required_keys:
        require_keys(*required_keys)
    return colab


def require_keys(*names: str) -> None:
    """Fail fast (with a fix-it message) if any API key is missing from the env.

    Args:
        *names: Environment variable names that must be set and non-empty.

    Returns:
        ``None``.

    Raises:
        ValueError: Any named key is missing, with the list of what to set.
    """
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        where = (
            "Colab: add them under the key icon (Secrets) and grant notebook access"
            if in_colab()
            else "Local: put them in a .env file next to this notebook"
        )
        raise MissingKeyError(f"Missing API key(s): {', '.join(missing)}. {where}.")


def api_key_for(provider: str) -> str | None:
    """The API key for a provider from explicit config or its environment variable."""
    cfg = get_config()
    if cfg.api_key and provider == cfg.provider:
        return cfg.api_key
    env = API_KEY_ENV.get(provider)
    return os.environ.get(env) if env else None
