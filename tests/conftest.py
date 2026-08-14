"""Shared fixtures: clean config + fake API keys, no network anywhere."""

from __future__ import annotations

import pytest

from tai_aitutor import config as cfg
from tai_aitutor import llm


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """Reset global config/clients and provide dummy keys for every test."""
    cfg._reset()
    llm._reset_clients()
    for env in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "COHERE_API_KEY",
                "TOGETHER_API_KEY", "DEEPSEEK_API_KEY", "PERPLEXITY_API_KEY"):
        monkeypatch.setenv(env, f"test-{env.lower()}")
    yield
    cfg._reset()
    llm._reset_clients()


@pytest.fixture
def add_chunks():
    """Index documents into a collection the way the notebooks now do it.

    ``ingest()`` was removed in the strip pass (chunk -> embed -> upsert is a
    visible loop in every retrieval lesson), so tests write the same three steps.
    """
    from tai_aitutor.chunking import Chunk, chunk_document

    def _add(docs, collection, embed_fn, chunk_size=512, chunk_overlap=128):
        chunks: list[Chunk] = []
        for doc in docs:
            if isinstance(doc, Chunk):
                chunks.append(doc)
            elif isinstance(doc, str):
                from tai_aitutor.documents import Document

                chunks.extend(
                    chunk_document(
                        Document(id="doc", text=doc),
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                )
            else:
                chunks.extend(
                    chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                )
        if not chunks:
            return []
        collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embed_fn([c.text for c in chunks]),
            metadatas=[
                {k: v for k, v in (c.metadata or {}).items() if isinstance(v, (str, int, float, bool))}
                or {"_": ""}
                for c in chunks
            ],
        )
        return chunks

    return _add
