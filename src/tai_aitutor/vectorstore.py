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

from .chunking import Chunk
from .errors import ProviderNotInstalledError

__all__ = [
    "get_collection",
    "reset_collection",
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

    Args:
        name: Collection name.
        path: Directory for an on-disk store, or ``None`` for in-memory.

    Returns:
        The Chroma collection, created if absent, using cosine distance.

    Raises:
        ValueError: ``chromadb`` is not installed.
    """
    return _client(path).get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )


def reset_collection(name: str, path: str | None = None):
    """Delete-and-recreate a collection (idempotent re-runs of ingest cells).

    Args:
        name: Collection name.
        path: Directory for an on-disk store, or ``None`` for in-memory.

    Returns:
        A new, empty collection at the same name and path.
    """
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


# --------------------------------------------------------------------------- #
# Enumeration and filters
# --------------------------------------------------------------------------- #


def get_all_chunks(collection, page_size: int = 500) -> list[Chunk]:
    """Every chunk in a collection, via real pagination.

    Replaces the old notebook hack of running a dummy query with
    ``similarity_top_k=100_000_000`` — the enumeration path BM25 indexing
    (Hybrid Search lesson) builds on.

    Args:
        collection: The Chroma collection to enumerate.
        page_size: Rows fetched per request.

    Returns:
        Every stored chunk, as :class:`Chunk` objects. This is the real
        enumeration path — no ``top_k=100000000`` hack.
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

    Args:
        sources: The selected metadata values.
        key: The metadata key to filter on.

    Returns:
        ``None`` for an empty selection (search everything), ``{key: {"$eq": v}}``
        for one value, ``{key: {"$in": [...]}}`` for several — the shapes
        production sends.
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
