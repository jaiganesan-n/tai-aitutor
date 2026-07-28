"""Answer synthesis: retrieved chunks + question → grounded answer.

Built in: "From Script to Pipeline" and "Improving Data Sources and Prompts"
(Section 4). Replaces ``index.as_query_engine(...).query(...)``,
``get_response_synthesizer``, and the response object's ``.response`` /
``.source_nodes`` — with a visible prompt (:func:`build_rag_prompt`) and a
plain :class:`Answer` dataclass.

There is no "response mode" machinery (the old refine-mode demo died with the
port, by design): synthesis is the prompt you can read, sent once.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from . import config as _cfg
from . import llm as _llm
from . import prompts
from .errors import TaiAitutorError
from .llm import Usage
from .retrieval import ScoredChunk, search

__all__ = [
    "Answer",
    "build_rag_prompt",
    "answer",
    "answer_with_sources",
    "answer_stream",
    "AnswerStream",
]


@dataclass
class Answer:
    """A grounded answer plus the evidence that produced it (was the Response object)."""

    text: str
    sources: list[ScoredChunk] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"Answer({self.text[:80]!r}..., sources={len(self.sources)})"


def build_rag_prompt(
    question: str,
    hits: list[ScoredChunk],
    cited: bool = False,
) -> str:
    """The course's visible RAG prompt: numbered excerpts with titles/sources, then the question.

    This f-string is what ``as_query_engine`` hid. Chunk metadata (``title``,
    ``source``/``source_name``, ``url``) becomes the citation header of each
    excerpt — the grounding pattern production's citation resolver builds on.
    """
    blocks = []
    for i, hit in enumerate(hits, 1):
        metadata = hit.metadata or {}
        title = metadata.get("title") or metadata.get("source_name") or metadata.get("source")
        url = metadata.get("url")
        blocks.append(prompts.context_block(i, hit.text, title=title, url=url))
    context = "\n\n".join(blocks) if blocks else "(no context retrieved)"

    cite_line = (
        "\nCite the excerpts you used with bracketed numbers like [1]." if cited else ""
    )
    return (
        f"Context excerpts:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer the question using only the context excerpts above.{cite_line}"
    )


def _retrieve(question, collection, top_k, where, retriever) -> list[ScoredChunk]:
    if retriever is not None:
        return list(retriever(question))
    if collection is None:
        raise TaiAitutorError(
            "answer() needs somewhere to retrieve from: pass collection=... "
            "(a Chroma collection) or retriever=... (a question -> hits callable)."
        )
    return search(question, collection, top_k=top_k, where=where)


def answer(
    question: str,
    collection=None,
    top_k: int = 5,
    where: dict | None = None,
    retriever=None,
    *,
    system: str | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> Answer:
    """Retrieve → prompt → generate. The whole query engine, in three visible steps.

    ``retriever`` accepts any ``question -> list[ScoredChunk]`` callable, which
    is how the later lessons plug in hybrid search, reranking, or HyDE::

        answer(q, retriever=lambda q: rerank(q, hybrid_search(q, col, bm25)))
    """
    hits = _retrieve(question, collection, top_k, where, retriever)
    prompt = build_rag_prompt(question, hits)
    cfg = _cfg.resolve(provider=provider, chat_model=model)
    text, usage = _llm._complete(
        prompt,
        system or prompts.RAG_SYSTEM,
        cfg=cfg,
        temperature=None,
        max_tokens=None,
        reasoning_effort=None,
    )
    return Answer(text=text, sources=hits, usage=usage)


def answer_with_sources(
    question: str,
    collection=None,
    top_k: int = 5,
    where: dict | None = None,
    retriever=None,
    *,
    model: str | None = None,
    provider: str | None = None,
) -> Answer:
    """Like :func:`answer`, but the model is instructed to cite excerpts ([1], [2]...).

    The "Improving Data Sources and Prompts" lesson's citation grounding:
    chunk titles/sources/urls go into the prompt, bracketed citations come out,
    and ``Answer.sources[i-1]`` resolves what ``[i]`` refers to.
    """
    hits = _retrieve(question, collection, top_k, where, retriever)
    prompt = build_rag_prompt(question, hits, cited=True)
    cfg = _cfg.resolve(provider=provider, chat_model=model)
    text, usage = _llm._complete(
        prompt,
        prompts.RAG_SYSTEM_CITED,
        cfg=cfg,
        temperature=None,
        max_tokens=None,
        reasoning_effort=None,
    )
    return Answer(text=text, sources=hits, usage=usage)


class AnswerStream:
    """Iterator of text deltas with ``.sources`` available up front.

    Retrieval happens before generation starts, so the evidence is inspectable
    while tokens stream — replaces ``print_response_stream()``. After
    iteration, the accumulated text is in ``.text``.
    """

    def __init__(self, deltas: Iterator[str], sources: list[ScoredChunk]):
        self._deltas = deltas
        self.sources = sources
        self.text = ""

    def __iter__(self) -> Iterator[str]:
        for delta in self._deltas:
            self.text += delta
            yield delta


def answer_stream(
    question: str,
    collection=None,
    top_k: int = 5,
    where: dict | None = None,
    retriever=None,
    *,
    system: str | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> AnswerStream:
    """Streaming :func:`answer`: iterate for deltas; ``.sources`` is ready immediately.

    >>> stream = answer_stream("What is RAG?", collection=col)
    >>> for token in stream: print(token, end="")
    >>> stream.sources
    """
    hits = _retrieve(question, collection, top_k, where, retriever)
    prompt = build_rag_prompt(question, hits)
    deltas = _llm.generate_stream(
        prompt, system or prompts.RAG_SYSTEM, model=model, provider=provider
    )
    return AnswerStream(deltas, hits)
