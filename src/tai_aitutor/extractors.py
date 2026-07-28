"""Metadata extractors: enrich chunks with LLM-derived metadata before embedding.

Built in: "RAG Improve Chunking" / "Search With Metadata Filtering" (Sections
4 and 7). Replaces LlamaIndex's ``KeywordExtractor``, ``SummaryExtractor``,
and ``QuestionsAnsweredExtractor`` — each is one typed LLM call per chunk,
run concurrently, writing a metadata field you can see.

Every extractor is a ``list[Chunk] -> list[Chunk]`` function, so they plug
straight into ``ingest(docs, col, enrich=[extract_keywords])``. For custom
settings, use ``functools.partial(extract_keywords, n=5)``.

:func:`situate_chunk` is production's contextual-retrieval step
(``add_context_to_nodes.py`` / the ``SituatedContext`` pattern): a cheap model
writes 1-2 situating sentences per chunk before embedding, which measurably
lifts retrieval on ambiguous chunks — the Section 13 companion notebook
measures exactly this with the course eval functions.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from pydantic import BaseModel, Field

from . import prompts
from .chunking import Chunk
from .llm import extract

__all__ = [
    "extract_keywords",
    "extract_summary",
    "extract_questions",
    "situate_chunk",
    "situate_chunks",
]


def _map_chunks(chunks, fn, concurrency, desc, show_progress) -> list[Chunk]:
    """Apply ``fn(chunk) -> Chunk`` concurrently, preserving order, never mutating inputs."""
    results: list[Chunk | None] = [None] * len(chunks)

    def work(index: int) -> None:
        results[index] = fn(chunks[index])

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(work, i) for i in range(len(chunks))]
        iterator = futures
        if show_progress and len(futures) > 1:
            from tqdm.auto import tqdm

            iterator = tqdm(futures, total=len(futures), desc=desc)
        for future in iterator:
            future.result()
    return [r for r in results if r is not None]


def _with_metadata(chunk: Chunk, **fields) -> Chunk:
    return replace(chunk, metadata={**chunk.metadata, **fields})


class _Keywords(BaseModel):
    keywords: list[str] = Field(description="The most search-relevant terms in the text.")


def extract_keywords(
    chunks: list[Chunk],
    n: int = 10,
    *,
    model: str | None = None,
    provider: str | None = None,
    concurrency: int = 8,
    show_progress: bool = True,
) -> list[Chunk]:
    """Add ``metadata["keywords"]`` (comma-joined) to each chunk. Replaces ``KeywordExtractor``."""

    def one(chunk: Chunk) -> Chunk:
        result = extract(
            f"Text:\n{chunk.text}\n\nList the {n} most important keywords or key phrases.",
            _Keywords,
            model=model,
            provider=provider,
        )
        return _with_metadata(chunk, keywords=", ".join(result.keywords[:n]))

    return _map_chunks(chunks, one, concurrency, "extract_keywords", show_progress)


class _Summary(BaseModel):
    summary: str = Field(description="A 1-3 sentence summary of the text.")


def extract_summary(
    chunks: list[Chunk],
    *,
    model: str | None = None,
    provider: str | None = None,
    concurrency: int = 8,
    show_progress: bool = True,
) -> list[Chunk]:
    """Add ``metadata["summary"]`` to each chunk. Replaces ``SummaryExtractor``."""

    def one(chunk: Chunk) -> Chunk:
        result = extract(
            f"Text:\n{chunk.text}\n\nSummarize this text in 1-3 sentences.",
            _Summary,
            model=model,
            provider=provider,
        )
        return _with_metadata(chunk, summary=result.summary.strip())

    return _map_chunks(chunks, one, concurrency, "extract_summary", show_progress)


class _Questions(BaseModel):
    questions: list[str] = Field(description="Questions this text can answer.")


def extract_questions(
    chunks: list[Chunk],
    n: int = 3,
    *,
    model: str | None = None,
    provider: str | None = None,
    concurrency: int = 8,
    show_progress: bool = True,
) -> list[Chunk]:
    """Add ``metadata["questions_answered"]`` to each chunk. Replaces ``QuestionsAnsweredExtractor``."""

    def one(chunk: Chunk) -> Chunk:
        result = extract(
            f"Text:\n{chunk.text}\n\nWrite {n} questions this text can answer.",
            _Questions,
            model=model,
            provider=provider,
        )
        return _with_metadata(chunk, questions_answered=" | ".join(result.questions[:n]))

    return _map_chunks(chunks, one, concurrency, "extract_questions", show_progress)


class SituatedContext(BaseModel):
    """Production's contextual-retrieval schema: where does this chunk sit in its document?"""

    context: str = Field(description="1-2 sentences situating the chunk within its document.")


def situate_chunk(
    chunk_text: str,
    document_text: str,
    *,
    model: str | None = None,
    provider: str | None = None,
) -> str:
    """The situating sentences for one chunk (production's ``add_context_to_nodes`` call)."""
    result = extract(
        f"DOCUMENT:\n{document_text}\n\nCHUNK:\n{chunk_text}\n\nSituate the chunk.",
        SituatedContext,
        system=prompts.SITUATE_CHUNK,
        model=model,
        provider=provider,
    )
    return result.context.strip()


def situate_chunks(
    chunks: list[Chunk],
    document_text: str,
    *,
    model: str | None = None,
    provider: str | None = None,
    concurrency: int = 8,
    show_progress: bool = True,
) -> list[Chunk]:
    """Prefix every chunk with its situating context (contextual retrieval at ingest).

    The chunk text becomes ``"{situating context}\\n\\n{original text}"`` and the
    original text is kept in ``metadata["original_text"]`` — embed the situated
    version, show either.
    """

    def one(chunk: Chunk) -> Chunk:
        context = situate_chunk(chunk.text, document_text, model=model, provider=provider)
        situated = replace(
            chunk,
            text=f"{context}\n\n{chunk.text}",
            metadata={**chunk.metadata, "original_text": chunk.text},
        )
        return situated

    return _map_chunks(chunks, one, concurrency, "situate_chunks", show_progress)
