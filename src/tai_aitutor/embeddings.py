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

import warnings

from . import config as _cfg
from ._retry import with_retries
from .errors import EmbeddingsNotAvailableError, MissingKeyError, ProviderNotInstalledError

__all__ = ["embed", "embed_cohere", "embed_local"]

Vector = list[float]

_GEMINI_TASKS = {"document": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"}
_COHERE_TASKS = {"document": "search_document", "query": "search_query"}

#: The course pins every embedder to 1536 dimensions so vectors stay comparable
#: across lessons and providers, and so an embedding cache built with one
#: provider still loads under another.
EMBED_DIM = 1536

_local_models: dict[str, object] = {}


def _check_task(task: str) -> str:
    if task not in ("document", "query"):
        raise ValueError(f'task must be "document" or "query", got {task!r}')
    return task


def embed(
    texts,
    task: str = "document",
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    retries: int = 0,
) -> list[Vector]:
    """Embed text on whichever embedding provider is configured.

    Args:
        texts: One string, or a list of strings. A single string is treated as a
            one-item list.
        task: ``"document"`` when indexing, ``"query"`` when searching. Gemini,
            Cohere and e5-style local models encode the two sides differently;
            OpenAI does not.
        provider: Override the configured embedding provider for this call.
        model: Embedding model id; defaults to the configured model.
        api_key: Override the environment key for this call.
        retries: Retry attempts on transient errors. ``0`` (the default) means
            the call is made exactly once.

    Returns:
        A list of vectors — **always a list, even for a single input**, so
        ``embed(question, task="query")[0]`` is the query vector. Every vector is
        :data:`EMBED_DIM` floats long.

    Raises:
        ValueError: The provider has no embeddings API (Anthropic) or is not one
            this package knows.

    The caller does the batching: loop your corpus in slices and call ``embed``
    once per slice, exactly as the indexing lesson does.

    >>> vectors = embed([c.text for c in chunks])         # index time
    >>> qvec    = embed("What is RAG?", task="query")[0]  # search time
    """
    _check_task(task)
    items: list[str] = [texts] if isinstance(texts, str) else list(texts)
    if not items:
        return []
    # Embedding models work best on single-line inputs.
    items = [t.replace("\n", " ") for t in items]

    cfg = _cfg.get_config()
    prov = (provider or cfg.embed_provider).lower()
    mdl = model or (
        cfg.embed_model if prov == cfg.embed_provider else _cfg.EMBED_MODEL_DEFAULTS.get(prov)
    )

    if prov == "anthropic":
        raise EmbeddingsNotAvailableError(
            "Anthropic has no embeddings API. Use Gemini (default) or OpenAI embeddings: "
            'embed(..., provider="gemini") — see Decision 2 in the course setup lesson.'
        )

    if prov == "gemini":
        return _embed_gemini(items, task, mdl, api_key, retries)
    if prov == "openai":
        return _embed_openai(items, mdl, api_key, retries)
    if prov == "cohere":
        return embed_cohere(items, task, mdl, EMBED_DIM, api_key=api_key, retries=retries)
    if prov == "local":
        return embed_local(items, mdl, task)
    raise EmbeddingsNotAvailableError(
        f"No embeddings branch for provider {prov!r}. "
        'Use "gemini", "openai", "cohere", or "local".'
    )


def _embed_gemini(items, task, model, api_key, retries) -> list[Vector]:
    from .llm import _client_gemini

    client = _client_gemini(api_key or _cfg.api_key_for("gemini"))

    def call(batch):
        from google.genai import types

        resp = client.models.embed_content(
            model=model,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=_GEMINI_TASKS[task],
                output_dimensionality=EMBED_DIM,
            ),
        )
        return [list(e.values) for e in resp.embeddings]

    vectors = with_retries(lambda: call(items), retries=retries)
    if len(vectors) != len(items):
        # A multimodal Gemini embedder can read a list as ONE document and return a
        # single aggregated vector. Verify counts at the API boundary, then fall back
        # to one call per input. Behaviour-invisible: one vector per input either way.
        vectors = [with_retries(lambda t=t: call([t]), retries=retries)[0] for t in items]
    return vectors


def _embed_openai(items, model, api_key, retries) -> list[Vector]:
    from .llm import _client_openai

    client = _client_openai(api_key or _cfg.api_key_for("openai"))

    def call():
        resp = client.embeddings.create(model=model, input=items, dimensions=EMBED_DIM)
        data = sorted(resp.data, key=lambda d: d.index)
        return [list(d.embedding) for d in data]

    return with_retries(call, retries=retries)


def embed_cohere(
    texts,
    task: str = "document",
    model: str = "embed-v4.0",
    output_dimension: int | None = EMBED_DIM,
    batch_size: int = 96,
    *,
    api_key: str | None = None,
    retries: int = 0,
) -> list[Vector]:
    """Cohere embeddings via ``ClientV2.embed`` — the production embedder.

    Args:
        texts: One string, or a list of strings.
        task: ``"document"`` at index time, ``"query"`` at search time. This
            becomes Cohere's ``input_type`` — the asymmetry production relies on.
        model: Cohere embedding model id.
        output_dimension: ``embed-v4.0`` is Matryoshka-trained
            (256/512/1024/1536), so the dimension is a choice; the course pins
            :data:`EMBED_DIM`. Pass ``None`` to take the API default.
        batch_size: Texts per request — the API caps them.
        api_key: Override ``COHERE_API_KEY`` for this call.
        retries: Retry attempts on transient errors; ``0`` means one call.

    Returns:
        A list of vectors, one per input, always a list.

    Raises:
        ValueError: The ``cohere`` package is not installed, or no API key is set.
    """
    _check_task(task)
    items = [texts] if isinstance(texts, str) else list(texts)
    items = [t.replace("\n", " ") for t in items]

    client = _client_cohere(api_key)
    out: list[Vector] = []
    for start in range(0, len(items), batch_size):  # the API caps texts per call
        batch = items[start : start + batch_size]

        def call(b=batch):
            kwargs: dict = {}
            if output_dimension is not None:
                kwargs["output_dimension"] = output_dimension
            try:
                resp = client.embed(
                    texts=b,
                    model=model,
                    input_type=_COHERE_TASKS[task],
                    embedding_types=["float"],
                    **kwargs,
                )
            except TypeError:
                # Older cohere SDKs predate output_dimension — degrade loudly, not fatally.
                if not kwargs:
                    raise
                warnings.warn(
                    "This cohere SDK does not support output_dimension= — embedding at the "
                    "model default instead. pip install 'cohere>=5.13' to pin dimensions.",
                    stacklevel=3,
                )
                resp = client.embed(
                    texts=b,
                    model=model,
                    input_type=_COHERE_TASKS[task],
                    embedding_types=["float"],
                )
            floats = getattr(resp.embeddings, "float_", None)
            if floats is None:  # SDK versions differ on the alias
                floats = resp.embeddings.float
            return [list(v) for v in floats]

        out.extend(with_retries(call, retries=retries))
    return out


def _client_cohere(api_key: str | None = None):
    try:
        import cohere
    except ImportError as exc:  # pragma: no cover
        raise ProviderNotInstalledError(
            "The Cohere SDK is not installed. Run: pip install 'tai-aitutor[rerank]'"
        ) from exc
    key = api_key or _cfg.api_key_for("cohere")
    if not key:
        raise MissingKeyError(
            "No COHERE_API_KEY found. Set it in Colab Secrets or your .env file, or pass "
            "api_key=... — reranking and Cohere embeddings both need it (free tier at cohere.com)."
        )
    return cohere.ClientV2(api_key=key)


# Query-side instructions for instruction-tuned retrieval families (as of July 2026).
# Documents embed plain; ONLY queries carry the instruction. The e5 family is
# different again: it prefixes BOTH sides ("query: " / "passage: ").
_QUERY_PROMPTS: dict[str, str] = {
    "qwen3-embedding": (
        "Instruct: Given a web search query, retrieve relevant passages that "
        "answer the query\nQuery: "
    ),
    "gte-qwen": (
        "Instruct: Given a web search query, retrieve relevant passages that "
        "answer the query\nQuery: "
    ),
}


def embed_local(
    texts,
    model_name: str,
    task: str = "document",
    query_prompt: str | None = None,
) -> list[Vector]:
    """Embed with a local sentence-transformers model (free, offline once downloaded).

    Args:
        texts: One string, or a list of strings.
        model_name: A sentence-transformers model id, e.g. ``"BAAI/bge-small-en-v1.5"``.
        task: ``"document"`` or ``"query"``.
        query_prompt: Explicit query instruction, for models
            :data:`_QUERY_PROMPTS` does not recognise.

    Returns:
        A list of vectors, one per input, always a list. Vectors are
        L2-normalised, so a dot product is a cosine similarity.

    Raises:
        ValueError: ``sentence-transformers`` is not installed.

    Prefix handling is the invisible failure the embedding lesson warns about,
    so it is handled here rather than skipped: the e5 family gets its
    ``"query: "`` / ``"passage: "`` prefixes on both sides, instruction-tuned
    retrievers get a query instruction on the query side only.
    """
    _check_task(task)
    items = [texts] if isinstance(texts, str) else list(texts)

    if "e5" in model_name.lower():
        prefix = query_prompt if query_prompt is not None else (
            "query: " if task == "query" else "passage: "
        )
        items = [prefix + t for t in items]
    elif task == "query":
        instruction = query_prompt
        if instruction is None:
            lowered = model_name.lower()
            instruction = next(
                (p for marker, p in _QUERY_PROMPTS.items() if marker in lowered), None
            )
        if instruction:
            items = [instruction + t for t in items]

    model = _get_local_model(model_name)
    vectors = model.encode(items, normalize_embeddings=True, show_progress_bar=False)
    return [list(map(float, v)) for v in vectors]


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
