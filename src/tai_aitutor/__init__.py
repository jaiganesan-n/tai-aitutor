"""tai-aitutor — the Towards AI course toolkit.

Plain-Python building blocks for the RAG AI Tutor built across the
Full Stack AI Engineer course: provider-neutral LLM calls, embeddings,
chunking, retrieval, and evaluation.

Migrating from LlamaIndex
-------------------------
The course used to do four things through LlamaIndex: configure models on
``Settings``, build an ``Index``, query it through a ``QueryEngine``, and grade
the result with ``Evaluator`` classes. Those four jobs are still here, as
functions you can open: ``configure()``, a Chroma collection you hold yourself,
``search()`` + ``build_rag_prompt()`` + ``generate()``, and the metric and judge
functions in ``evals``. The mapping is symbol-for-symbol and written out in the
README's migration map, so the model you already have of a RAG pipeline
transfers intact — what changes is that every stage is now a call you can read
the source of, and the stages you used to get bundled are yours to arrange.

How it maps onto Sections 1-8
-----------------------------
Every function here is code you wrote by hand in a lesson first: ``chunk``,
``embed``, ``search``, ``build_rag_prompt``, ``hit_rate``, the judges. Same
names, same signatures, same returns. That is why the package starts at Section
9 — by then you have built each stage yourself, so importing it back is
consolidation, and later lessons can spend their cells on the new idea instead
of re-pasted plumbing. Each module docstring names the lesson that builds its
contents.

Quickstart::

    from tai_aitutor import configure, get_collection, chunk_document, embed, search

    configure(provider="gemini")                     # or "openai" / "anthropic"
    col = get_collection("kb", path="./db")          # the collection IS the index

    chunks = [c for d in docs for c in chunk_document(d)]
    col.add(                                         # chunk -> embed -> upsert,
        ids=[c.id for c in chunks],                  # the same three visible steps
        documents=[c.text for c in chunks],          # the indexing lesson writes out
        embeddings=embed([c.text for c in chunks]),
        metadatas=[c.metadata for c in chunks],
    )
    hits = search("What is RAG?", col, top_k=5)       # dense retrieval, with scores
"""

from .chunking import (
    Chunk,
    chunk,
    chunk_document,
    chunk_sentences,
    heading_aware_markdown_chunks,
    sentence_window_chunks,
)
from .config import (
    Config,
    configure,
    get_config,
    in_colab,
    require_keys,
    setup_notebook,
)
from .display import show_answer, show_chunks, show_eval_table
from .documents import Document, load_csv
from .embeddings import EMBED_DIM, embed, embed_cohere, embed_local
from .errors import (
    EmbeddingsNotAvailableError,
    MissingKeyError,
    ProviderNotInstalledError,
    StructuredOutputError,
    TaiAitutorError,
    UnsupportedProviderError,
)
from .evals import (
    CorrectnessVerdict,
    FaithfulnessVerdict,
    QADataset,
    QueryResult,
    RelevancyVerdict,
    RetrievalReport,
    evaluate_retrieval,
    hit_rate,
    judge_correctness,
    judge_faithfulness,
    judge_relevancy,
    reciprocal_rank,
    sweep_top_k,
)
from .llm import extract, generate
from .retrieval import (
    BM25Index,
    ScoredChunk,
    code_tokenize,
    decompose_question,
    expand_window,
    hyde_search,
    judge_rerank,
    rerank,
    rewrite_query,
    rrf_fuse,
    search,
)
from .router import RouteDecision, route
from .synthesis import build_rag_prompt
from .tokens import n_tokens
from .tools import Tool, search_web, tool
from .vectorstore import build_where_filter, get_all_chunks, get_collection, reset_collection

__version__ = "0.0.3"

__all__ = [
    # config — which provider and model the package calls
    "configure",
    "get_config",
    "setup_notebook",
    "require_keys",
    "in_colab",
    "Config",
    # llm — text in, text or a typed object out
    "generate",
    "extract",
    # embeddings
    "embed",
    "embed_cohere",
    "embed_local",
    "EMBED_DIM",
    # tokens
    "n_tokens",
    # documents
    "Document",
    "load_csv",
    # chunking
    "Chunk",
    "chunk",
    "chunk_document",
    "chunk_sentences",
    "heading_aware_markdown_chunks",
    "sentence_window_chunks",
    # vectorstore — the Chroma collection IS the index
    "get_collection",
    "reset_collection",
    "get_all_chunks",
    "build_where_filter",
    # retrieval
    "search",
    "ScoredChunk",
    "expand_window",
    "code_tokenize",
    "BM25Index",
    "rrf_fuse",
    "rerank",
    "judge_rerank",
    "rewrite_query",
    "hyde_search",
    "decompose_question",
    # synthesis — the prompt, visible
    "build_rag_prompt",
    # evals
    "QADataset",
    "hit_rate",
    "reciprocal_rank",
    "evaluate_retrieval",
    "sweep_top_k",
    "RetrievalReport",
    "QueryResult",
    "FaithfulnessVerdict",
    "RelevancyVerdict",
    "CorrectnessVerdict",
    "judge_faithfulness",
    "judge_relevancy",
    "judge_correctness",
    # tools
    "Tool",
    "tool",
    "search_web",
    # router
    "route",
    "RouteDecision",
    # display
    "show_chunks",
    "show_answer",
    "show_eval_table",
    # errors — all subclass ValueError
    "TaiAitutorError",
    "UnsupportedProviderError",
    "ProviderNotInstalledError",
    "MissingKeyError",
    "EmbeddingsNotAvailableError",
    "StructuredOutputError",
    "__version__",
]
