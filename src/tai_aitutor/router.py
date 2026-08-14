"""Routing: which path should this question take?

Built in: "Adding Question Validation and Routing" (Section 9). Replaces
``RouterQueryEngine`` + ``LLMSingleSelector`` / ``PydanticSingleSelector`` —
and demystifies what a router is: ONE typed classification call. The dispatch
stays in the notebook as a plain ``if/else``, because that's the honest shape
of the pattern (and of the production agent's tool choice):

>>> decision = route(question, routes={
...     "knowledge": "AI/LLM course questions answerable from the knowledge base",
...     "general":   "general programming questions not covered by the course",
...     "reject":    "off-topic questions (politics, medical advice, homework-for-hire)",
... })
>>> if decision.route == "knowledge":
...     answer(question, collection=col)
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from . import llm as _llm
from .errors import TaiAitutorError

__all__ = ["RouteDecision", "route"]


class RouteDecision(BaseModel):
    """The router's typed verdict: which route, and why."""

    route: str = Field(description="Exactly one of the offered route names.")
    reason: str = Field(description="One sentence explaining the choice.")


def route(
    question: str,
    routes: dict[str, str],
    *,
    model: str | None = None,
    provider: str | None = None,
) -> RouteDecision:
    """Classify a question into one of the named routes (one typed LLM call).

    ``routes`` maps a route name to its description — descriptions are what the
    model chooses by, exactly like tool descriptions. The returned
    ``decision.route`` is guaranteed to be one of your keys (case-insensitive
    matching is applied; anything unresolvable raises rather than mis-routing
    silently).

    Args:
        question: The incoming question.
        routes: Route name to description. The model chooses by the
            descriptions, exactly as it chooses tools.
        model: Model id for the routing call.
        provider: Override the configured provider for this call.

    Returns:
        A :class:`RouteDecision` whose ``route`` is guaranteed to be one of your
        keys.

    Raises:
        ValueError: ``routes`` is empty, or the model's pick matched no route —
            better a loud failure than a silent mis-route.
    """
    if not routes:
        raise TaiAitutorError("route() needs at least one route (name -> description).")

    listing = "\n".join(f"- {name}: {description}" for name, description in routes.items())
    decision = _llm.extract(
        f"Question: {question}\n\nRoutes:\n{listing}\n\n"
        "Pick the single best route for this question.",
        RouteDecision,
        system="You are a router deciding how to handle an incoming question.",
        model=model,
        provider=provider,
    )

    matched = _match_route(decision.route, routes)
    if matched is None:
        raise TaiAitutorError(
            f"Router returned {decision.route!r}, which is not one of {sorted(routes)}. "
            "Tighten the route descriptions and retry."
        )
    return RouteDecision(route=matched, reason=decision.reason)


def _match_route(name: str, routes: dict[str, str]) -> str | None:
    """Exact match, then case-insensitive match. Anything else is unresolved."""
    if name in routes:
        return name
    by_lower = {key.lower(): key for key in routes}
    return by_lower.get(name.lower().strip())
