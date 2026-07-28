"""ChromaDB collections and ingestion — the course's vector store, unwrapped.

Built in: "Using a Vector Database" (Section 4) and "Improving Data Sources
and Prompts". Replaces LlamaIndex's ``ChromaVectorStore`` + ``StorageContext``
+ ``VectorStoreIndex.from_documents`` + ``IngestionPipeline`` — there is no
index object here. The Chroma collection IS the index; ingestion is a loop
you can read: chunk → (enrich) → embed → upsert.

Persistence is just a path: ``get_collection("ai_tutor_knowledge", path="./db")``
(replaces ``StorageContext`` / ``persist`` / ``load_index_from_storage``).

``build_where_filter`` produces the exact filter shape production uses to
scope retrieval to a student's selected sources (``$eq`` / ``$in``).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import embeddings as _embeddings
from .chunking import Chunk, chunk_document
from .documents import Document
from .errors import ProviderNotInstalledError

__all__ = [
    "get_collection",
    "reset_collection",
    "ingest",
    "IngestStats",
    "get_all_chunks",
    "build_where_filter",
]

_clients: dict[str | None, object] = {}


def _chromadb():
    try:
        import chromadb
    except ImportError as exc:
        raise ProviderNotInstalledError(
            "chromadb is not installed. Run: pip install 'tai-aitutor[rag]'"
        ) from exc
    return chromadb


def _client(path: str | None):
    if path not in _clients:
        chromadb = _chromadb()
        _clients[path] = (
            chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        )
    return _clients[path]


def get_collection(name: str, path: str | None = None):
    """Get or create a Chroma collection (cosine space).

    ``path=None`` → in-memory (gone when the runtime restarts);
    ``path="./db"`` → persisted on disk. That one argument is the whole
    "storage context" story.
    """
    return _client(path).get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )


def reset_collection(name: str, path: str | None = None):
    """Delete-and-recreate a collection (idempotent re-runs of ingest cells)."""
    client = _client(path)
    try:
        client.delete_collection(name)
    except Exception:
        pass  # didn't exist yet
    return get_collection(name, path)


def _reset_clients() -> None:
    """Testing hook."""
    _clients.clear()


# --------------------------------------------------------------------------- #
# Ingestion (replaces IngestionPipeline.run)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IngestStats:
    """What one ingest() call did."""

    documents: int
    chunks: int
    collection: str

    def __repr__(self) -> str:
        return (
            f"IngestStats({self.documents} documents → {self.chunks} chunks "
            f"→ collection {self.collection!r})"
        )


def _sanitize_metadata(metadata: dict) -> dict | None:
    """Chroma metadata values must be str/int/float/bool — flatten the rest."""
    out: dict = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        elif isinstance(value, (list, tuple)):
            out[key] = ", ".join(map(str, value))
        else:
            out[key] = str(value)
    return out or None


def ingest(
    docs,
    collection,
    chunker=None,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    enrich=(),
    embed_fn=None,
    batch_size: int = 64,
    show_progress: bool = True,
) -> IngestStats:
    """Chunk → (enrich) → embed → upsert. The visible ingestion loop.

    Parameters
    ----------
    docs:
        ``Document`` objects, plain strings, or ready-made ``Chunk`` lists.
    chunker:
        Optional ``text -> list[str]`` override (e.g.
        ``heading_aware_markdown_chunks``); default is token chunking with
        ``chunk_size``/``chunk_overlap``.
    enrich:
        Callables ``list[Chunk] -> list[Chunk]`` applied before embedding —
        the extractors module plugs in here (keywords, summaries, questions).
    embed_fn:
        ``(texts, task=...) -> vectors`` override; default is the configured
        :func:`tai_aitutor.embed`.
    """
    docs = list(docs)

    # 1. Chunk
    chunks: list[Chunk] = []
    n_docs = 0
    for doc in docs:
        if isinstance(doc, Chunk):
            chunks.append(doc)
            continue
        n_docs += 1
        if isinstance(doc, str):
            doc = Document(text=doc)
        chunks.extend(chunk_document(doc, chunk_size, chunk_overlap, chunker=chunker))

    # 2. Enrich (metadata extractors)
    for enrich_fn in enrich:
        chunks = enrich_fn(chunks)

    # 3 + 4. Embed and upsert, batch by batch
    embed_fn = embed_fn or _embeddings.embed
    batches = range(0, len(chunks), batch_size)
    if show_progress and len(chunks) > batch_size:
        from tqdm.auto import tqdm

        batches = tqdm(batches, desc=f"ingest → {collection.name}", unit="batch")

    for start in batches:
        batch = chunks[start : start + batch_size]
        texts = [c.text for c in batch]
        vectors = [
            c.embedding if c.embedding is not None else vec
            for c, vec in zip(batch, embed_fn(texts, task="document"))
        ]
        collection.upsert(
            ids=[c.id for c in batch],
            documents=texts,
            embeddings=vectors,
            metadatas=[_sanitize_metadata(c.metadata) for c in batch],
        )

    return IngestStats(documents=n_docs, chunks=len(chunks), collection=collection.name)


# --------------------------------------------------------------------------- #
# Enumeration and filters
# --------------------------------------------------------------------------- #


def get_all_chunks(collection, page_size: int = 500) -> list[Chunk]:
    """Every chunk in a collection, via real pagination.

    Replaces the old notebook hack of running a dummy query with
    ``similarity_top_k=100_000_000`` — the enumeration path BM25 indexing
    (Hybrid Search lesson) builds on.
    """
    out: list[Chunk] = []
    offset = 0
    while True:
        page = collection.get(
            limit=page_size, offset=offset, include=["documents", "metadatas"]
        )
        ids = page.get("ids") or []
        if not ids:
            break
        texts = page.get("documents") or [""] * len(ids)
        metas = page.get("metadatas") or [None] * len(ids)
        for chunk_id, text, metadata in zip(ids, texts, metas):
            out.append(Chunk(id=chunk_id, text=text or "", metadata=dict(metadata or {})))
        offset += len(ids)
        if len(ids) < page_size:
            break
    return out


def build_where_filter(sources, key: str = "source") -> dict | None:
    """The production source-scoping filter (``build_where_filter`` in the live tutor).

    >>> build_where_filter(None)                      # no filter
    >>> build_where_filter("tai_blog")                # {'source': {'$eq': 'tai_blog'}}
    >>> build_where_filter(["tai_blog", "hf_docs"])   # {'source': {'$in': [...]}}
    """
    if sources is None:
        return None
    if isinstance(sources, str):
        return {key: {"$eq": sources}}
    sources = list(sources)
    if not sources:
        return None
    if len(sources) == 1:
        return {key: {"$eq": sources[0]}}
    return {key: {"$in": sources}}
