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

import base64
import json
import re
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from pydantic import BaseModel

from . import config as _cfg
from ._retry import with_retries
from .errors import ProviderNotInstalledError, StructuredOutputError, UnsupportedProviderError

__all__ = [
    "Usage",
    "generate",
    "generate_stream",
    "generate_vision",
    "extract",
    "ask_batch",
]


@dataclass(frozen=True)
class Usage:
    """Token accounting for one model call (drives ``tokens.estimate_cost``)."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# --------------------------------------------------------------------------- #
# SDK clients (lazy imports so `import tai_aitutor` stays instant, and a
# student who picked Gemini never needs the other SDKs installed).
# Factories are module-level so tests can monkeypatch them.
# --------------------------------------------------------------------------- #

_clients: dict[tuple, object] = {}


def _client_gemini(api_key: str | None = None):
    key = ("gemini", api_key)
    if key not in _clients:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - message text tested via raise path
            raise ProviderNotInstalledError(
                "The Gemini SDK is not installed. Run: pip install 'tai-aitutor[gemini]'"
            ) from exc
        _clients[key] = genai.Client(api_key=api_key) if api_key else genai.Client()
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
        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
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
        _clients[key] = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
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
    *,
    model: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """One prompt in, one answer out — on whichever provider is configured.

    >>> configure(provider="gemini")
    >>> generate("What is RAG?", system="Answer in one sentence.")
    """
    text, _ = _complete(
        prompt,
        system,
        cfg=_resolve_call(provider, model, base_url, api_key),
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
    return text


def _complete(
    prompt: str,
    system: str | None,
    cfg: _cfg.Config,
    temperature: float | None,
    max_tokens: int | None,
    reasoning_effort: str | None,
) -> tuple[str, Usage]:
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
            usage = getattr(resp, "usage_metadata", None)
            return resp.text or "", Usage(
                getattr(usage, "prompt_token_count", 0) or 0,
                getattr(usage, "candidates_token_count", 0) or 0,
            )

        return with_retries(call)

    if provider == "openai" and not cfg.base_url:

        def call():
            kwargs: dict = {}
            if reasoning_effort is not None:
                kwargs["reasoning"] = {"effort": reasoning_effort}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_output_tokens"] = max_tokens
            resp = _client_openai(api_key).responses.create(
                model=model, input=prompt, instructions=system, **kwargs
            )
            usage = getattr(resp, "usage", None)
            return resp.output_text or "", Usage(
                getattr(usage, "input_tokens", 0) or 0,
                getattr(usage, "output_tokens", 0) or 0,
            )

        return with_retries(call)

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
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            usage = getattr(resp, "usage", None)
            return text, Usage(
                getattr(usage, "input_tokens", 0) or 0,
                getattr(usage, "output_tokens", 0) or 0,
            )

        return with_retries(call)

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
            usage = getattr(resp, "usage", None)
            return resp.choices[0].message.content or "", Usage(
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
            )

        return with_retries(call)

    raise UnsupportedProviderError(f"No generation branch for provider {provider!r}.")


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


def generate_stream(
    prompt: str,
    system: str | None = None,
    *,
    model: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Iterator[str]:
    """Like :func:`generate`, but yields text deltas as they arrive."""
    cfg = _resolve_call(provider, model, base_url, api_key)
    prov, mdl = cfg.provider, cfg.chat_model
    api_key = cfg.api_key or _cfg.api_key_for(prov)

    if prov == "gemini":
        from google.genai import types

        gen_cfg = types.GenerateContentConfig(
            system_instruction=system, temperature=temperature, max_output_tokens=max_tokens
        )
        for chunk in _client_gemini(api_key).models.generate_content_stream(
            model=mdl, contents=prompt, config=gen_cfg
        ):
            if getattr(chunk, "text", None):
                yield chunk.text
        return

    if prov == "openai" and not cfg.base_url:
        kwargs: dict = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens
        with _client_openai(api_key).responses.stream(
            model=mdl, input=prompt, instructions=system, **kwargs
        ) as stream:
            for event in stream:
                if getattr(event, "type", "") == "response.output_text.delta":
                    yield event.delta
        return

    if prov == "anthropic":
        kwargs = {}
        if system is not None:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        with _client_anthropic(api_key).messages.stream(
            model=mdl,
            max_tokens=max_tokens or 4096,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        ) as stream:
            yield from stream.text_stream
        return

    if cfg.base_url:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        kwargs = {"stream": True}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        for chunk in _client_compat(prov, api_key, cfg.base_url).chat.completions.create(
            model=mdl, messages=messages, **kwargs
        ):
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
        return

    raise UnsupportedProviderError(f"No streaming branch for provider {prov!r}.")


# --------------------------------------------------------------------------- #
# Vision
# --------------------------------------------------------------------------- #


def generate_vision(
    prompt: str,
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    *,
    model: str | None = None,
    provider: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Ask a question about an image (Multimodal LLMs lesson)."""
    cfg = _resolve_call(provider, model, None, None)
    prov, mdl = cfg.provider, cfg.chat_model
    api_key = cfg.api_key or _cfg.api_key_for(prov)

    if prov == "gemini":

        def call():
            from google.genai import types

            resp = _client_gemini(api_key).models.generate_content(
                model=mdl,
                contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), prompt],
            )
            return resp.text or ""

        return with_retries(call)

    if prov == "openai" and not cfg.base_url:

        def call():
            b64 = base64.b64encode(image_bytes).decode()
            resp = _client_openai(api_key).responses.create(
                model=mdl,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": f"data:{mime_type};base64,{b64}"},
                            {"type": "input_text", "text": prompt},
                        ],
                    }
                ],
            )
            return resp.output_text or ""

        return with_retries(call)

    if prov == "anthropic":

        def call():
            b64 = base64.b64encode(image_bytes).decode()
            resp = _client_anthropic(api_key).messages.create(
                model=mdl,
                max_tokens=max_tokens or 4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": mime_type, "data": b64},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

        return with_retries(call)

    raise UnsupportedProviderError(
        f"generate_vision supports gemini/openai/anthropic; got {prov!r} "
        "(vision support on OpenAI-compatible endpoints is model-specific — call the SDK directly)."
    )


# --------------------------------------------------------------------------- #
# Structured outputs
# --------------------------------------------------------------------------- #

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _extract_json_text(raw: str) -> str:
    """Pull a JSON object out of a possibly-fenced model response."""
    match = _JSON_BLOCK.search(raw)
    if match:
        raw = match.group(1)
    raw = raw.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        return raw[start : end + 1]
    return raw


def extract[S: BaseModel](
    prompt: str,
    schema: type[S],
    system: str | None = None,
    *,
    model: str | None = None,
    provider: str | None = None,
) -> S:
    """Structured output: returns a validated instance of ``schema`` (a Pydantic model).

    Native mechanisms per provider — the exact three patterns the Structured
    Outputs lesson teaches: Gemini ``response_schema``, OpenAI structured
    outputs, Anthropic tool-schema. OpenAI-compatible endpoints fall back to
    JSON-by-prompt + validation with one repair retry.
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
            if isinstance(parsed, schema):
                return parsed
            return schema.model_validate_json(_extract_json_text(resp.text or ""))

        return with_retries(call)

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

        return with_retries(call)

    if prov == "anthropic":

        def call():
            kwargs: dict = {}
            if system is not None:
                kwargs["system"] = system
            resp = _client_anthropic(api_key).messages.create(
                model=mdl,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                tools=[
                    {
                        "name": "emit_result",
                        "description": f"Record the result as a {schema.__name__}.",
                        "input_schema": schema.model_json_schema(),
                    }
                ],
                tool_choice={"type": "tool", "name": "emit_result"},
                **kwargs,
            )
            for block in resp.content:
                if getattr(block, "type", "") == "tool_use":
                    return schema.model_validate(block.input)
            raise StructuredOutputError(
                f"Anthropic returned no tool_use block for schema {schema.__name__}."
            )

        return with_retries(call)

    # OpenAI-compatible fallback: JSON by prompt, validate, one repair retry.
    schema_json = json.dumps(schema.model_json_schema(), indent=None)
    ask = (
        f"{prompt}\n\nRespond with ONLY a JSON object (no prose, no code fences) "
        f"matching this JSON Schema:\n{schema_json}"
    )
    raw = generate(ask, system, model=mdl, provider=prov)
    try:
        return schema.model_validate_json(_extract_json_text(raw))
    except Exception:
        repair = (
            f"The following was supposed to be valid JSON for the schema "
            f"{schema_json}\nbut failed to parse:\n{raw}\n\n"
            "Return ONLY the corrected JSON object."
        )
        raw2 = generate(repair, system, model=mdl, provider=prov)
        try:
            return schema.model_validate_json(_extract_json_text(raw2))
        except Exception as exc:
            raise StructuredOutputError(
                f"Could not parse a valid {schema.__name__} from the model response."
            ) from exc


# --------------------------------------------------------------------------- #
# chat_completion — the raw messages + tools layer (agents lessons, Chat/ToolLoop)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation the model asked for."""

    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class Completion:
    """One model turn: text and/or tool calls (replaces agent-framework response objects)."""

    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    stop_reason: str  # "stop" | "tool_calls" | provider-specific


def _tool_spec(tool) -> dict:
    """Normalise a Tool object or dict into {name, description, parameters}."""
    if isinstance(tool, dict):
        return {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        }
    return {
        "name": tool.name,
        "description": getattr(tool, "description", "") or "",
        "parameters": getattr(tool, "parameters", None) or {"type": "object", "properties": {}},
    }


def _split_system(messages: list[dict], system: str | None) -> tuple[str | None, list[dict]]:
    """Allow system either as a parameter or as leading {"role": "system"} messages."""
    rest = list(messages)
    parts = [system] if system else []
    while rest and rest[0].get("role") == "system":
        parts.append(rest.pop(0)["content"])
    return ("\n\n".join(p for p in parts if p) or None), rest


def chat_completion(
    messages: list[dict],
    tools: list | None = None,
    *,
    system: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Completion:
    """One model turn over a normalized message list, with optional tool calling.

    This is the layer ``Chat`` and ``ToolLoop`` are built on — and the message
    format the memory lesson teaches (memory is just this list, resent):

    - ``{"role": "system" | "user" | "assistant", "content": str}``
    - assistant tool request: ``{"role": "assistant", "content": str,
      "tool_calls": [{"id", "name", "arguments": dict}]}``
    - tool result: ``{"role": "tool", "tool_call_id": str, "name": str,
      "content": str}``

    The same list drives all providers (Gemini, OpenAI Responses, Anthropic,
    OpenAI-compatible endpoints); each branch converts to its native shape.
    Replaces ``ChatMessage`` and the agent frameworks' hidden loops.
    """
    cfg = _resolve_call(provider, model, None, None)
    prov, mdl = cfg.provider, cfg.chat_model
    api_key = cfg.api_key or _cfg.api_key_for(prov)
    system, messages = _split_system(messages, system)
    specs = [_tool_spec(t) for t in (tools or [])]

    if prov == "gemini":
        return with_retries(
            lambda: _chat_gemini(api_key, mdl, messages, specs, system, temperature, max_tokens)
        )
    if prov == "openai" and not cfg.base_url:
        return with_retries(
            lambda: _chat_openai(api_key, mdl, messages, specs, system, temperature, max_tokens)
        )
    if prov == "anthropic":
        return with_retries(
            lambda: _chat_anthropic(api_key, mdl, messages, specs, system, temperature, max_tokens)
        )
    if cfg.base_url:
        return with_retries(
            lambda: _chat_compat(
                prov, api_key, cfg.base_url, mdl, messages, specs, system, temperature, max_tokens
            )
        )
    raise UnsupportedProviderError(f"No chat branch for provider {prov!r}.")


def _chat_gemini(api_key, model, messages, specs, system, temperature, max_tokens) -> Completion:
    from google.genai import types

    contents = []
    for message in messages:
        role = message["role"]
        if role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=message["content"])]))
        elif role == "assistant":
            parts = []
            if message.get("content"):
                parts.append(types.Part(text=message["content"]))
            for call in message.get("tool_calls", []):
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            name=call["name"], args=call["arguments"]
                        )
                    )
                )
            contents.append(types.Content(role="model", parts=parts or [types.Part(text="")]))
        elif role == "tool":
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=message.get("name", ""),
                                response={"result": message["content"]},
                            )
                        )
                    ],
                )
            )

    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        max_output_tokens=max_tokens,
        tools=[types.Tool(function_declarations=specs)] if specs else None,
    )
    resp = _client_gemini(api_key).models.generate_content(
        model=model, contents=contents, config=config
    )

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    candidates = getattr(resp, "candidates", None) or []
    parts = candidates[0].content.parts if candidates and candidates[0].content else []
    for i, part in enumerate(parts or []):
        call = getattr(part, "function_call", None)
        if call is not None:
            tool_calls.append(
                ToolCall(
                    id=getattr(call, "id", None) or f"call_{i}",
                    name=call.name,
                    arguments=dict(call.args or {}),
                )
            )
        elif getattr(part, "text", None):
            text_parts.append(part.text)
    usage = getattr(resp, "usage_metadata", None)
    return Completion(
        text="".join(text_parts),
        tool_calls=tool_calls,
        usage=Usage(
            getattr(usage, "prompt_token_count", 0) or 0,
            getattr(usage, "candidates_token_count", 0) or 0,
        ),
        stop_reason="tool_calls" if tool_calls else "stop",
    )


def _chat_openai(api_key, model, messages, specs, system, temperature, max_tokens) -> Completion:
    items: list[dict] = []
    for message in messages:
        role = message["role"]
        if role in ("user", "assistant") and not message.get("tool_calls"):
            items.append({"role": role, "content": message["content"]})
            continue
        if role == "assistant":
            if message.get("content"):
                items.append({"role": "assistant", "content": message["content"]})
            for call in message.get("tool_calls", []):
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"]),
                    }
                )
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                }
            )

    kwargs: dict = {}
    if specs:
        kwargs["tools"] = [{"type": "function", **spec} for spec in specs]
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_output_tokens"] = max_tokens
    resp = _client_openai(api_key).responses.create(
        model=model, input=items, instructions=system, **kwargs
    )

    tool_calls = [
        ToolCall(
            id=item.call_id,
            name=item.name,
            arguments=json.loads(getattr(item, "arguments", None) or "{}"),
        )
        for item in getattr(resp, "output", []) or []
        if getattr(item, "type", "") == "function_call"
    ]
    usage = getattr(resp, "usage", None)
    return Completion(
        text=getattr(resp, "output_text", "") or "",
        tool_calls=tool_calls,
        usage=Usage(
            getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0
        ),
        stop_reason="tool_calls" if tool_calls else "stop",
    )


def _chat_anthropic(api_key, model, messages, specs, system, temperature, max_tokens) -> Completion:
    converted: list[dict] = []
    for message in messages:
        role = message["role"]
        if role == "user":
            converted.append({"role": "user", "content": [{"type": "text", "text": message["content"]}]})
        elif role == "assistant":
            blocks: list[dict] = []
            if message.get("content"):
                blocks.append({"type": "text", "text": message["content"]})
            for call in message.get("tool_calls", []):
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["name"],
                        "input": call["arguments"],
                    }
                )
            converted.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message["tool_call_id"],
                            "content": message["content"],
                        }
                    ],
                }
            )
    # Anthropic requires strictly alternating roles: merge adjacent same-role messages.
    merged: list[dict] = []
    for message in converted:
        if merged and merged[-1]["role"] == message["role"]:
            merged[-1]["content"] = list(merged[-1]["content"]) + list(message["content"])
        else:
            merged.append(message)

    kwargs: dict = {}
    if system is not None:
        kwargs["system"] = system
    if temperature is not None:
        kwargs["temperature"] = temperature
    if specs:
        kwargs["tools"] = [
            {"name": s["name"], "description": s["description"], "input_schema": s["parameters"]}
            for s in specs
        ]
    resp = _client_anthropic(api_key).messages.create(
        model=model, max_tokens=max_tokens or 4096, messages=merged, **kwargs
    )

    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    tool_calls = [
        ToolCall(id=b.id, name=b.name, arguments=dict(b.input or {}))
        for b in resp.content
        if getattr(b, "type", "") == "tool_use"
    ]
    usage = getattr(resp, "usage", None)
    return Completion(
        text=text,
        tool_calls=tool_calls,
        usage=Usage(
            getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0
        ),
        stop_reason="tool_calls" if tool_calls else "stop",
    )


def _chat_compat(
    provider, api_key, base_url, model, messages, specs, system, temperature, max_tokens
) -> Completion:
    converted: list[dict] = []
    if system:
        converted.append({"role": "system", "content": system})
    for message in messages:
        role = message["role"]
        if role == "assistant" and message.get("tool_calls"):
            converted.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or None,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"]),
                            },
                        }
                        for call in message["tool_calls"]
                    ],
                }
            )
        elif role == "tool":
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": message["tool_call_id"],
                    "content": message["content"],
                }
            )
        else:
            converted.append({"role": role, "content": message["content"]})

    kwargs: dict = {}
    if specs:
        kwargs["tools"] = [{"type": "function", "function": spec} for spec in specs]
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    resp = _client_compat(provider, api_key, base_url).chat.completions.create(
        model=model, messages=converted, **kwargs
    )

    choice = resp.choices[0]
    raw_calls = getattr(choice.message, "tool_calls", None) or []
    tool_calls = [
        ToolCall(
            id=call.id,
            name=call.function.name,
            arguments=json.loads(call.function.arguments or "{}"),
        )
        for call in raw_calls
    ]
    usage = getattr(resp, "usage", None)
    return Completion(
        text=choice.message.content or "",
        tool_calls=tool_calls,
        usage=Usage(
            getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0
        ),
        stop_reason="tool_calls" if tool_calls else (choice.finish_reason or "stop"),
    )


# --------------------------------------------------------------------------- #
# Batch fan-out
# --------------------------------------------------------------------------- #


def ask_batch(
    prompts: Iterable[str],
    system: str | None = None,
    *,
    concurrency: int = 8,
    desc: str | None = None,
    **generate_kwargs,
) -> list[str]:
    """Run :func:`generate` over many prompts with a thread pool, order preserved.

    Replaces LlamaIndex's ``BatchEvalRunner`` fan-out — and the ``nest_asyncio``
    boilerplate cells that only existed to make its asyncio work in notebooks.
    """
    prompts = list(prompts)
    results: list[str | None] = [None] * len(prompts)

    def worker(idx: int) -> None:
        results[idx] = generate(prompts[idx], system, **generate_kwargs)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(worker, i): i for i in range(len(prompts))}
        iterator = futures
        if desc:
            from tqdm.auto import tqdm

            iterator = tqdm(futures, total=len(futures), desc=desc)
        for future in iterator:
            future.result()  # propagate the first error

    return [r if r is not None else "" for r in results]
