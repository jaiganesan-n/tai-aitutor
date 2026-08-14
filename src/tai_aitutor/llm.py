"""Provider-neutral text generation: the course's `generate()` family.

Built in: "How To Use LLMs via API" (Section 1) — the lesson shows all three
native SDK calls side by side; this module is those same calls behind one
signature. Replaces the LlamaIndex LLM wrappers (``OpenAI``, ``GoogleGenAI``,
``Perplexity``, ``TogetherLLM``), ``ChatMessage``, ``structured_predict`` /
``as_structured_llm``, and ``BatchEvalRunner``-style fan-out.

Design notes
------------
- No LLM objects. Functions + a config. Per-call ``provider=`` / ``model=``
  override the configured default (comparison lessons stay natural).
- Together / DeepSeek / Perplexity / Ollama (and any custom ``base_url``) go
  through the OpenAI-compatible branch — the exact pattern the "open-weight
  models" lesson teaches.
- Typos in keyword arguments raise ``TypeError`` immediately. (The old
  LlamaIndex path silently swallowed ``additional_kwrgs={'reasoning_effort':
  'minimal'}`` — the setting never reached the model.)
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from . import config as _cfg
from ._retry import with_retries
from .errors import (
    MissingKeyError,
    ProviderNotInstalledError,
    StructuredOutputError,
    UnsupportedProviderError,
)

__all__ = [
    "generate",
    "extract",
]


# --------------------------------------------------------------------------- #
# SDK clients (lazy imports so `import tai_aitutor` stays instant, and a
# student who picked Gemini never needs the other SDKs installed).
# Factories are module-level so tests can monkeypatch them.
# --------------------------------------------------------------------------- #

_clients: dict[tuple, object] = {}


def _require_key(provider: str, api_key: str | None) -> str:
    """The key for ``provider``, or a package error naming the variable to set.

    Provider SDKs raise their own errors for a missing key, and those errors do
    not say which environment variable the course expects. This one does.
    """
    key = api_key or _cfg.api_key_for(provider)
    if not key:
        env = _cfg.API_KEY_ENV.get(provider) or f"{provider.upper()}_API_KEY"
        raise MissingKeyError(
            f"No API key found for {provider!r}. Set {env} in Colab Secrets or your "
            f".env file, or pass api_key=... to this call."
        )
    return key


def _client_gemini(api_key: str | None = None):
    key = ("gemini", api_key)
    if key not in _clients:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - message text tested via raise path
            raise ProviderNotInstalledError(
                "The Gemini SDK is not installed. Run: pip install 'tai-aitutor[gemini]'"
            ) from exc
        _clients[key] = genai.Client(api_key=_require_key("gemini", api_key))
    return _clients[key]


def _client_openai(api_key: str | None = None, base_url: str | None = None):
    key = ("openai", api_key, base_url)
    if key not in _clients:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ProviderNotInstalledError(
                "The OpenAI SDK is not installed. Run: pip install 'tai-aitutor[openai]'"
            ) from exc
        kwargs: dict = {"api_key": api_key} if api_key else {}
        if base_url:
            kwargs["base_url"] = base_url
        elif not api_key:
            kwargs["api_key"] = _require_key("openai", api_key)
        _clients[key] = OpenAI(**kwargs)
    return _clients[key]


def _client_anthropic(api_key: str | None = None):
    key = ("anthropic", api_key)
    if key not in _clients:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ProviderNotInstalledError(
                "The Anthropic SDK is not installed. Run: pip install 'tai-aitutor[anthropic]'"
            ) from exc
        _clients[key] = anthropic.Anthropic(api_key=_require_key("anthropic", api_key))
    return _clients[key]


def _client_compat(provider: str, api_key: str | None, base_url: str | None):
    """OpenAI-compatible endpoint (Together, DeepSeek, Perplexity, Ollama, custom)."""
    key = api_key or _cfg.api_key_for(provider)
    if key is None:
        if provider == "ollama" or provider not in _cfg.API_KEY_ENV:
            key = "not-needed"  # local/custom endpoints ignore the key
        else:
            from .errors import MissingKeyError

            env = _cfg.API_KEY_ENV.get(provider)
            raise MissingKeyError(
                f"No API key found for {provider!r}. Set {env} (Colab Secrets or .env), "
                "or pass api_key=... explicitly."
            )
    return _client_openai(api_key=key, base_url=base_url)


def _reset_clients() -> None:
    """Testing hook."""
    _clients.clear()


# --------------------------------------------------------------------------- #
# generate()
# --------------------------------------------------------------------------- #


def _resolve_call(provider, model, base_url, api_key) -> _cfg.Config:
    return _cfg.resolve(provider=provider, chat_model=model, base_url=base_url, api_key=api_key)


def generate(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    *,
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    retries: int = 0,
) -> str:
    """One prompt in, one answer out — on whichever provider is configured.

    Args:
        prompt: The user message.
        system: Optional system instruction.
        model: Model id; defaults to the configured chat model for the provider.
        provider: Override the configured provider for this call.
        temperature: Sampling temperature, or ``None`` for the provider default.
        max_tokens: Output cap, or ``None`` for the provider default.
        reasoning_effort: Reasoning budget where the provider supports one.
        base_url: OpenAI-compatible endpoint (Together / DeepSeek / Ollama / ...).
        api_key: Override the environment key for this call.
        retries: Retry attempts on transient errors. ``0`` (the default) means the
            call is made exactly once — no hidden retry loop. Section 13 turns
            this on and explains it.

    Returns:
        The model's reply text — an empty string if the model returned nothing,
        never ``None``.

    Raises:
        ValueError: The provider is not one this package knows and no
            ``base_url`` was given.

    >>> configure(provider="gemini")
    >>> generate("What is RAG?", system="Answer in one sentence.")
    """
    return _complete(
        prompt,
        system,
        cfg=_resolve_call(provider, model, base_url, api_key),
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        retries=retries,
    )


def _complete(
    prompt: str,
    system: str | None,
    cfg: _cfg.Config,
    temperature: float | None,
    max_tokens: int | None,
    reasoning_effort: str | None,
    retries: int = 0,
) -> str:
    provider, model = cfg.provider, cfg.chat_model
    api_key = cfg.api_key or _cfg.api_key_for(provider)

    if provider == "gemini":

        def call():
            from google.genai import types

            gen_cfg = types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens,
                thinking_config=(
                    types.ThinkingConfig(thinking_budget=0)
                    if reasoning_effort in ("none", "minimal")
                    else None
                ),
            )
            resp = _client_gemini(api_key).models.generate_content(
                model=model, contents=prompt, config=gen_cfg
            )
            return resp.text or ""

        return with_retries(call, retries=retries)

    if provider == "openai" and not cfg.base_url:

        def call():
            kwargs: dict = {}
            # The taught cell sends reasoning={"effort": "none"} unconditionally.
            kwargs["reasoning"] = {"effort": reasoning_effort or "none"}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_output_tokens"] = max_tokens
            resp = _client_openai(api_key).responses.create(
                model=model, input=prompt, instructions=system, **kwargs
            )
            return resp.output_text or ""

        return with_retries(call, retries=retries)

    if provider == "anthropic":

        def call():
            kwargs: dict = {}
            if system is not None:
                kwargs["system"] = system
            if temperature is not None:
                kwargs["temperature"] = temperature
            resp = _client_anthropic(api_key).messages.create(
                model=model,
                max_tokens=max_tokens or 4096,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

        return with_retries(call, retries=retries)

    # OpenAI-compatible endpoints (together / deepseek / perplexity / ollama / custom base_url)
    if cfg.base_url:

        def call():
            messages = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": prompt}
            ]
            kwargs: dict = {}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            resp = _client_compat(provider, api_key, cfg.base_url).chat.completions.create(
                model=model, messages=messages, **kwargs
            )
            return resp.choices[0].message.content or ""

        return with_retries(call, retries=retries)

    raise UnsupportedProviderError(f"No generation branch for provider {provider!r}.")


# --------------------------------------------------------------------------- #
# Structured outputs
# --------------------------------------------------------------------------- #


def extract[S: BaseModel](
    prompt: str,
    schema: type[S],
    system: str | None = None,
    model: str | None = None,
    *,
    provider: str | None = None,
    retries: int = 0,
) -> S:
    """Structured output: a validated instance of ``schema`` (a Pydantic model).

    Native mechanisms per provider — the exact three patterns the Structured
    Outputs lesson teaches: Gemini ``response_schema``, OpenAI
    ``responses.parse``, Anthropic ``messages.parse``. OpenAI-compatible
    endpoints ask for JSON by prompt and validate it.

    Args:
        prompt: The user message.
        schema: A ``pydantic.BaseModel`` subclass describing the desired shape.
        system: Optional system instruction.
        model: Model id; defaults to the configured chat model for the provider.
        provider: Override the configured provider for this call.
        retries: Retry attempts on transient errors. ``0`` (the default) means
            the call is made exactly once.

    Returns:
        A validated ``schema`` instance.

    Raises:
        ValueError: The response could not be parsed into ``schema``, or the
            provider is unknown. Never returns ``None`` and never repairs the
            JSON behind your back — a malformed response is a real failure.

    >>> class Ticket(BaseModel):
    ...     priority: str
    >>> extract("Server is down", Ticket)
    """
    cfg = _resolve_call(provider, model, None, None)
    prov, mdl = cfg.provider, cfg.chat_model
    api_key = cfg.api_key or _cfg.api_key_for(prov)

    if prov == "gemini":

        def call():
            from google.genai import types

            resp = _client_gemini(api_key).models.generate_content(
                model=mdl,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            parsed = getattr(resp, "parsed", None)
            if not isinstance(parsed, schema):
                raise StructuredOutputError(
                    f"Gemini returned no valid {schema.__name__}. Tighten the prompt or "
                    f"simplify the schema; the raw response was: {resp.text!r}"
                )
            return parsed

        return with_retries(call, retries=retries)

    if prov == "openai" and not cfg.base_url:

        def call():
            resp = _client_openai(api_key).responses.parse(
                model=mdl, input=prompt, instructions=system, text_format=schema
            )
            parsed = getattr(resp, "output_parsed", None)
            if parsed is None:
                raise StructuredOutputError(
                    f"OpenAI returned no parsed output for schema {schema.__name__}."
                )
            return parsed

        return with_retries(call, retries=retries)

    if prov == "anthropic":

        def call():
            kwargs: dict = {}
            if system is not None:
                kwargs["system"] = system
            resp = _client_anthropic(api_key).messages.parse(
                model=mdl,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                output_format=schema,
                **kwargs,
            )
            parsed = getattr(resp, "parsed_output", None)
            if parsed is None:
                raise StructuredOutputError(
                    f"Anthropic returned no parsed output for schema {schema.__name__}."
                )
            return parsed

        return with_retries(call, retries=retries)

    # OpenAI-compatible endpoints: ask for JSON by prompt, then validate it.
    schema_json = json.dumps(schema.model_json_schema(), indent=None)
    ask = (
        f"{prompt}\n\nRespond with ONLY a JSON object (no prose, no code fences) "
        f"matching this JSON Schema:\n{schema_json}"
    )
    raw = generate(ask, system, mdl, provider=prov, retries=retries)
    try:
        return schema.model_validate_json(raw)
    except Exception as exc:
        raise StructuredOutputError(
            f"Could not parse a valid {schema.__name__} from the model response. "
            f"Ask the endpoint for stricter JSON, or use a provider with native "
            f"structured output. Raw response: {raw!r}"
        ) from exc
