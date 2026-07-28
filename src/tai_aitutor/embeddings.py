"""Provider-neutral embeddings: the course's `embed()` family.

Built in: "Building a Basic RAG Pipeline From Scratch" (Section 3) and
"How To Select the Right Embedding Model" (Section 5). Replaces the LlamaIndex
embedding wrappers (``OpenAIEmbedding``, ``CohereEmbedding``,
``HuggingFaceEmbedding``, ``resolve_embed_model``) and ``Settings.embed_model``.

The ``task`` argument ("document" vs "query") is the asymmetry lesson:
documents are embedded for being *found*, queries for *finding* — Gemini,
Cohere, and e5-style local models all encode that distinction; OpenAI does not.
Production uses the same split (`search_document` / `search_query`).

Anthropic note (Decision 2): Anthropic has no embeddings API. A student who
picks Anthropic for chat still embeds with Gemini (default) or OpenAI.
"""

from __future__ import annotations

from typing import overload

from . import config as _cfg
from ._retry import with_retries
from .errors import EmbeddingsNotAvailableError, ProviderNotInstalledError

__all__ = ["embed", "embed_cohere", "embed_local"]

Vector = list[float]

_GEMINI_TASKS = {"document": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"}
_COHERE_TASKS = {"document": "search_document", "query": "search_query"}

#: Per-request input caps (provider API limits; batches are split to fit).
_MAX_BATCH = {"gemini": 100, "openai": 2048, "cohere": 96}

_local_models: dict[str, object] = {}


def _check_task(task: str) -> str:
    if task not in ("document", "query"):
        raise ValueError(f'task must be "document" or "query", got {task!r}')
    return task


def _batched(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


@overload
def embed(texts: str, task: str = ..., *, provider: str | None = ..., model: str | None = ..., batch_size: int = ..., api_key: str | None = ...) -> Vector: ...
@overload
def embed(texts: list[str], task: str = ..., *, provider: str | None = ..., model: str | None = ..., batch_size: int = ..., api_key: str | None = ...) -> list[Vector]: ...


def embed(
    texts,
    task: str = "document",
    *,
    provider: str | None = None,
    model: str | None = None,
    batch_size: int = 100,
    api_key: str | None = None,
):
    """Embed one string (returns a vector) or a list of strings (returns a list).

    ``task``: ``"document"`` when indexing, ``"query"`` when searching.

    >>> vectors = embed([c.text for c in chunks])          # index time
    >>> qvec    = embed("What is RAG?", task="query")       # search time
    """
    _check_task(task)
    single = isinstance(texts, str)
    items: list[str] = [texts] if single else list(texts)
    if not items:
        return [] if not single else []

    cfg = _cfg.get_config()
    prov = (provider or cfg.embed_provider).lower()
    mdl = model or (cfg.embed_model if prov == cfg.embed_provider else _cfg.EMBED_MODEL_DEFAULTS.get(prov))

    if prov == "anthropic":
        raise EmbeddingsNotAvailableError(
            "Anthropic has no embeddings API. Use Gemini (default) or OpenAI embeddings: "
            'embed(..., provider="gemini") — see Decision 2 in the course setup lesson.'
        )

    if prov == "gemini":
        vectors = _embed_gemini(items, task, mdl, batch_size, api_key)
    elif prov == "openai":
        vectors = _embed_openai(items, mdl, batch_size, api_key)
    elif prov == "cohere":
        vectors = embed_cohere(items, task, model=mdl, api_key=api_key)
    elif prov == "local":
        vectors = embed_local(items, model_name=mdl, task=task)
    else:
        raise EmbeddingsNotAvailableError(
            f"No embeddings branch for provider {prov!r}. "
            'Use "gemini", "openai", "cohere", or "local".'
        )

    return vectors[0] if single else vectors


def _embed_gemini(items, task, model, batch_size, api_key):
    from .llm import _client_gemini

    client = _client_gemini(api_key or _cfg.api_key_for("gemini"))
    out: list[Vector] = []
    for batch in _batched(items, min(batch_size, _MAX_BATCH["gemini"])):

        def call(b=batch):
            from google.genai import types

            resp = client.models.embed_content(
                model=model,
                contents=b,
                config=types.EmbedContentConfig(task_type=_GEMINI_TASKS[task]),
            )
            return [list(e.values) for e in resp.embeddings]

        out.extend(with_retries(call))
    return out


def _embed_openai(items, model, batch_size, api_key):
    from .llm import _client_openai

    client = _client_openai(api_key or _cfg.api_key_for("openai"))
    out: list[Vector] = []
    for batch in _batched(items, min(batch_size, _MAX_BATCH["openai"])):

        def call(b=batch):
            resp = client.embeddings.create(model=model, input=b)
            data = sorted(resp.data, key=lambda d: d.index)
            return [list(d.embedding) for d in data]

        out.extend(with_retries(call))
    return out


def embed_cohere(
    texts,
    task: str = "document",
    *,
    model: str = "embed-v4.0",
    output_dimension: int | None = 1536,
    api_key: str | None = None,
) -> list[Vector] | Vector:
    """Cohere embeddings via ``ClientV2.embed`` — the production embedder.

    Teaches the ``input_type`` asymmetry explicitly: ``search_document`` at
    index time, ``search_query`` at search time (exactly as production does).

    ``embed-v4.0`` is Matryoshka-trained (256/512/1024/1536); the course
    standardises on **1536** so vectors stay comparable across notebooks and
    storage cost stays predictable — hence the explicit default. Pass a smaller
    ``output_dimension`` for the storage-vs-quality experiment, or ``None`` to
    take whatever the API's default is.
    """
    _check_task(task)
    single = isinstance(texts, str)
    items = [texts] if single else list(texts)

    client = _client_cohere(api_key)
    out: list[Vector] = []
    for batch in _batched(items, _MAX_BATCH["cohere"]):

        def call(b=batch):
            kwargs: dict = {}
            if output_dimension is not None:
                kwargs["output_dimension"] = output_dimension
            resp = client.embed(
                texts=b,
                model=model,
                input_type=_COHERE_TASKS[task],
                embedding_types=["float"],
                **kwargs,
            )
            floats = getattr(resp.embeddings, "float_", None)
            if floats is None:  # SDK versions differ on the alias
                floats = getattr(resp.embeddings, "float")
            return [list(v) for v in floats]

        out.extend(with_retries(call))
    return out[0] if single else out


def _client_cohere(api_key: str | None = None):
    try:
        import cohere
    except ImportError as exc:  # pragma: no cover
        raise ProviderNotInstalledError(
            "The Cohere SDK is not installed. Run: pip install 'tai-aitutor[rerank]'"
        ) from exc
    key = api_key or _cfg.api_key_for("cohere")
    return cohere.ClientV2(api_key=key) if key else cohere.ClientV2()


#: Known instruction-tuned retrieval models → the query instruction they expect.
#: Matched by substring against the lowercased model name; extend as lessons add models.
#: (e5's "query:"/"passage:" prefixes are handled separately below.)
_KNOWN_QUERY_PROMPTS: dict[str, str] = {
    "qwen3-embedding": (
        "Instruct: Given a web search query, retrieve relevant passages that answer "
        "the query\nQuery: "
    ),
    "gte-qwen": (
        "Instruct: Given a web search query, retrieve relevant passages that answer "
        "the query\nQuery: "
    ),
    "linq-embed": "Instruct: Given a question, retrieve passages that answer the question\nQuery: ",
}


def _default_query_prompt(model_name: str) -> str | None:
    lowered = model_name.lower()
    for marker, prompt in _KNOWN_QUERY_PROMPTS.items():
        if marker in lowered:
            return prompt
    return None


def embed_local(
    texts,
    *,
    model_name: str = "BAAI/bge-small-en-v1.5",
    task: str = "document",
    batch_size: int = 32,
    normalize: bool = True,
    query_prompt: str | None = None,
) -> list[Vector] | Vector:
    """Local embeddings with sentence-transformers (free, offline once downloaded).

    Prefix handling — the invisible failure the embedding lesson warns about,
    handled instead of skipped:

    - e5 family: the required ``"query: "`` / ``"passage: "`` prefixes are
      applied automatically.
    - Instruction-tuned retrievers (Qwen3-Embedding, gte-Qwen, Linq-Embed…):
      a query instruction is applied automatically for known models
      (``_KNOWN_QUERY_PROMPTS``), and ``query_prompt=`` sets it explicitly for
      anything else. Documents are embedded plain, queries get the instruction
      — skipping it silently costs retrieval accuracy.
    """
    _check_task(task)
    single = isinstance(texts, str)
    items = [texts] if single else list(texts)

    if "e5" in model_name.lower():
        prefix = "query: " if task == "query" else "passage: "
        items = [prefix + t for t in items]
    elif task == "query":
        instruction = query_prompt if query_prompt is not None else _default_query_prompt(model_name)
        if instruction:
            items = [instruction + t for t in items]

    model = _get_local_model(model_name)
    vectors = model.encode(
        items, batch_size=batch_size, normalize_embeddings=normalize, show_progress_bar=False
    )
    out = [list(map(float, v)) for v in vectors]
    return out[0] if single else out


def _get_local_model(model_name: str):
    if model_name not in _local_models:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ProviderNotInstalledError(
                "sentence-transformers is not installed. Run: pip install 'tai-aitutor[local]'"
            ) from exc
        _local_models[model_name] = SentenceTransformer(model_name)
    return _local_models[model_name]
