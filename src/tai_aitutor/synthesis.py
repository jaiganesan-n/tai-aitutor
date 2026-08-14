"""Answer synthesis: retrieved chunks + question → grounded answer.

Built in: "From Script to Pipeline" and "Improving Data Sources and Prompts"
(Section 4). Replaces ``index.as_query_engine(...).query(...)``,
``get_response_synthesizer``, and the response object's ``.response`` /
``.source_nodes`` — with a visible prompt (:func:`build_rag_prompt`) and a
plain :class:`Answer` dataclass.

LlamaIndex's response modes become explicit here. ``compact`` is what
:func:`answer` does: build one prompt from the retrieved chunks and send it
once. ``refine`` and ``tree_summarize`` are multi-call loops over the same two
functions — :func:`build_rag_prompt` to shape each call, :func:`answer` to make
it — written out in the lesson that needs them, where their cost is visible.
"""

from __future__ import annotations

from . import prompts
from .retrieval import ScoredChunk

__all__ = [
    "build_rag_prompt",
]


def build_rag_prompt(
    question: str,
    hits: list[ScoredChunk],
    cited: bool = False,
) -> str:
    """The course's visible RAG prompt: numbered excerpts with titles/sources, then the question.

    This f-string is what ``as_query_engine`` hid. Chunk metadata (``title``,
    ``source``/``source_name``, ``url``) becomes the citation header of each
    excerpt — the grounding pattern production's citation resolver builds on.

    Args:
        question: The user's question.
        hits: The retrieved chunks to ground the answer in.
        cited: Ask for bracketed citation numbers in the answer.

    Returns:
        The full prompt string. With no hits, the context block reads
        ``"(no context retrieved)"`` rather than being silently empty.
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


