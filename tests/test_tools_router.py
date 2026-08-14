"""Tests for tools and the router."""

from __future__ import annotations

import pytest

import tai_aitutor as tai
from tai_aitutor import llm, tools
from tai_aitutor.errors import TaiAitutorError

# --------------------------------------------------------------------------- #
# tools.tool()
# --------------------------------------------------------------------------- #


def test_tool_schema_from_signature():
    @tai.tool
    def lookup(query: str, limit: int = 5, strict: bool = False) -> str:
        """Look up things in the catalog."""
        return query

    assert lookup.name == "lookup"
    assert lookup.description == "Look up things in the catalog."
    assert lookup.parameters["properties"]["query"] == {"type": "string"}
    assert lookup.parameters["properties"]["limit"] == {"type": "integer", "default": 5}
    assert lookup.parameters["properties"]["strict"] == {"type": "boolean", "default": False}
    assert lookup.parameters["required"] == ["query"]
    assert lookup(query="x") == "x"


def test_tool_name_description_override_and_missing_doc():
    def fn(a: float):
        return a

    named = tai.tool(fn, name="halve", description="Halve a number.")
    assert named.name == "halve" and named.description == "Halve a number."
    with pytest.raises(TaiAitutorError):
        tai.tool(lambda x: x)  # no docstring, no description


def test_search_web_via_fake_tavily(monkeypatch):
    class FakeTavily:
        def search(self, query, max_results):
            return {"results": [{"title": "T", "url": "https://u", "content": "C",
                                 "extra": "ignored"}][:max_results]}

    monkeypatch.setattr(tools, "_tavily_client", lambda api_key=None: FakeTavily())
    results = tai.search_web("python 3.14", max_results=1)
    assert results == [{"title": "T", "url": "https://u", "content": "C"}]


# --------------------------------------------------------------------------- #
# ToolLoop / Chat (fake chat_completion)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# router
# --------------------------------------------------------------------------- #

ROUTES = {"knowledge": "course questions", "general": "other programming",
          "reject": "off-topic"}


def test_route_valid_and_case_insensitive(monkeypatch):
    monkeypatch.setattr(
        llm, "extract",
        lambda p, s, system=None, model=None, provider=None: s(route="Knowledge",
                                                               reason="course-y"),
    )
    decision = tai.route("What is RAG?", ROUTES)
    assert decision.route == "knowledge"
    assert decision.reason == "course-y"


def test_route_unmatchable_raises(monkeypatch):
    monkeypatch.setattr(
        llm, "extract",
        lambda p, s, system=None, model=None, provider=None: s(route="banana", reason="?"),
    )
    with pytest.raises(TaiAitutorError):
        tai.route("What is RAG?", ROUTES)
    with pytest.raises(TaiAitutorError):
        tai.route("q", {})


# --------------------------------------------------------------------------- #
# finetune helpers (no torch needed)
# --------------------------------------------------------------------------- #


