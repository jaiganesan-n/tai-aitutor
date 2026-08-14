"""Tools: plain Python functions the model can call.

Built in: the Web Search lesson (the course's first taste of tool calling) and
the agents lessons. Replaces ``QueryEngineTool`` / ``ToolMetadata`` /
``TavilyToolSpec`` — a tool here is a function plus a JSON schema derived from
its signature, nothing more.

>>> @tool
... def multiply(a: float, b: float) -> float:
...     \"\"\"Multiply two numbers.\"\"\"
...     return a * b
>>> Chat(tools=[multiply, make_retrieval_tool(col)]).ask("what is 3 times 7?")
"""

from __future__ import annotations

import inspect
import os
import typing
from dataclasses import dataclass

from .errors import MissingKeyError, ProviderNotInstalledError, TaiAitutorError

__all__ = ["Tool", "tool", "search_web"]

_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass(frozen=True)
class Tool:
    """A callable the model may invoke: name + description + JSON schema + the function."""

    name: str
    description: str
    parameters: dict  # JSON schema for the arguments object
    fn: object

    def __call__(self, **kwargs):
        return self.fn(**kwargs)


def _schema_from_signature(fn) -> dict:
    """Derive the arguments JSON schema from type hints and defaults.

    LlamaIndex asked for a ``ToolMetadata`` with a hand-written schema; here the
    signature is the schema, read straight off the annotations and defaults.
    """
    try:
        hints = typing.get_type_hints(fn)  # resolves "int" strings to int under PEP 563
    except Exception:
        hints = {}
    properties: dict = {}
    required: list[str] = []
    for name, param in inspect.signature(fn).parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(name, param.annotation)
        json_type = _TYPE_MAP.get(annotation, "string")
        properties[name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            properties[name]["default"] = param.default
    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def tool(fn=None, *, name: str | None = None, description: str | None = None) -> Tool:
    """Wrap a function as a :class:`Tool` (decorator or plain call).

    The schema comes from the signature's type hints; the description from
    ``description=`` or the docstring's first paragraph. Replaces
    ``QueryEngineTool.from_defaults(...)`` + ``ToolMetadata`` — the tool IS
    the function.

    Args:
        fn: The function to wrap, when used as a bare decorator.
        name: Override the tool name; defaults to the function's name.
        description: Override the description; defaults to the docstring's
            first paragraph.

    Returns:
        A :class:`Tool`, or a decorator returning one.

    Raises:
        ValueError: The function has neither a docstring nor a ``description``
            — the model chooses tools by their descriptions.
    """

    def wrap(func) -> Tool:
        doc = (description or inspect.getdoc(func) or "").strip().split("\n\n")[0]
        if not doc:
            raise TaiAitutorError(
                f"Tool {func.__name__!r} needs a description: add a docstring or "
                "pass description=... — the model chooses tools by their descriptions."
            )
        return Tool(
            name=name or func.__name__,
            description=doc,
            parameters=_schema_from_signature(func),
            fn=func,
        )

    return wrap(fn) if fn is not None else wrap


# --------------------------------------------------------------------------- #
# The course's two standard tools
# --------------------------------------------------------------------------- #


def _tavily_client(api_key: str | None = None):
    try:
        from tavily import TavilyClient
    except ImportError as exc:
        raise ProviderNotInstalledError(
            "tavily-python is not installed. Run: pip install 'tai-aitutor[web]'"
        ) from exc
    key = api_key or os.environ.get("TAVILY_API_KEY")
    if not key:
        raise MissingKeyError(
            "No TAVILY_API_KEY found. Set it in Colab Secrets or .env (free tier at tavily.com)."
        )
    return TavilyClient(api_key=key)


def search_web(query: str, max_results: int = 5, api_key: str | None = None) -> list[dict]:
    """Web search via Tavily (replaces ``TavilyToolSpec``): title, url, content per result.

    Use directly for grounded prompts, or hand it to the model as a tool:
    ``Chat(tools=[tool(search_web)])`` — the Web Search lesson does both.

    Args:
        query: The search query.
        max_results: How many results to return.
        api_key: Override ``TAVILY_API_KEY`` for this call.

    Returns:
        One dict per result with ``title``, ``url`` and ``content``.

    Raises:
        ValueError: ``tavily-python`` is not installed, or no API key is set.
    """
    response = _tavily_client(api_key).search(query=query, max_results=max_results)
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in response.get("results", [])
    ]
