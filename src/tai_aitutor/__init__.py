"""tai-aitutor — the Towards AI course toolkit.

Plain-Python building blocks for the RAG AI Tutor built across the
Full Stack AI Engineer course: provider-neutral LLM calls, embeddings,
chunking, retrieval, and evaluation — every function is code a course
lesson builds inline first, then imports from here.

Not a framework: no hidden magic, no global objects beyond one `configure()`
call, and source you can read in one sitting.

Quickstart::

    from tai_aitutor import configure, get_collection, ingest, answer

    configure(provider="gemini")                    # or "openai" / "anthropic" / ...
    col = get_collection("kb", path="./db")         # the collection IS the index
    ingest(docs, col)                               # chunk → embed → upsert
    print(answer("What is RAG?", col))              # retrieve → prompt → generate
"""

from .chat import Chat, ChatEvent, ToolLoop
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
from .datasets import (
    ai_tutor_knowledge,
    mini_articles,
    prebuilt_chroma,
    qa_dataset,
    research_papers,
)
from .display import show_answer, show_chunks, show_eval_table
from .documents import (
    Document,
    load_csv,
    load_directory,
    load_files,
    load_hf_dataset,
    load_jsonl,
    load_wikipedia,
)
from .embeddings import embed, embed_cohere, embed_local
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
    JudgeReport,
    QADataset,
    QueryResult,
    RelevancyVerdict,
    RetrievalReport,
    evaluate_retrieval,
    hit_rate,
    judge_correctness,
    judge_faithfulness,
    judge_relevancy,
    make_qa_pairs,
    mrr,
    run_judges,
)
from .extractors import (
    SituatedContext,
    extract_keywords,
    extract_questions,
    extract_summary,
    situate_chunk,
    situate_chunks,
)
from .finetune import evaluate_embedder, make_training_pairs, train_embedder
from .llm import (
    Completion,
    ToolCall,
    Usage,
    ask_batch,
    chat_completion,
    extract,
    generate,
    generate_stream,
    generate_vision,
)
from .retrieval import (
    BM25Index,
    ScoredChunk,
    code_tokenize,
    decompose_question,
    expand_window,
    hybrid_search,
    hyde_search,
    judge_rerank,
    multi_step_answer,
    pack_context,
    rerank,
    rrf_fuse,
    search,
    subquestion_answer,
)
from .router import RouteDecision, route
from .synthesis import (
    Answer,
    AnswerStream,
    answer,
    answer_stream,
    answer_with_sources,
    build_rag_prompt,
)
from .tokens import estimate_cost, n_tokens, truncate
from .tools import Tool, make_retrieval_tool, render_tool_result, search_web, tool
from .vectorstore import (
    IngestStats,
    build_where_filter,
    get_all_chunks,
    get_collection,
    ingest,
    reset_collection,
)

__version__ = "1.0.0"

__all__ = [
    # config
    "configure",
    "get_config",
    "setup_notebook",
    "require_keys",
    "in_colab",
    "Config",
    # llm
    "generate",
    "generate_stream",
    "generate_vision",
    "extract",
    "ask_batch",
    "chat_completion",
    "Completion",
    "ToolCall",
    "Usage",
    # embeddings
    "embed",
    "embed_cohere",
    "embed_local",
    # tokens
    "n_tokens",
    "truncate",
    "estimate_cost",
    # documents
    "Document",
    "load_csv",
    "load_jsonl",
    "load_directory",
    "load_files",
    "load_wikipedia",
    "load_hf_dataset",
    # chunking
    "Chunk",
    "chunk",
    "chunk_document",
    "chunk_sentences",
    "heading_aware_markdown_chunks",
    "sentence_window_chunks",
    # vectorstore
    "get_collection",
    "reset_collection",
    "ingest",
    "IngestStats",
    "get_all_chunks",
    "build_where_filter",
    # retrieval
    "search",
    "ScoredChunk",
    "expand_window",
    "code_tokenize",
    "BM25Index",
    "rrf_fuse",
    "hybrid_search",
    "rerank",
    "judge_rerank",
    "hyde_search",
    "decompose_question",
    "subquestion_answer",
    "multi_step_answer",
    "pack_context",
    # synthesis
    "answer",
    "answer_with_sources",
    "answer_stream",
    "AnswerStream",
    "Answer",
    "build_rag_prompt",
    # evals
    "QADataset",
    "make_qa_pairs",
    "hit_rate",
    "mrr",
    "evaluate_retrieval",
    "RetrievalReport",
    "QueryResult",
    "FaithfulnessVerdict",
    "RelevancyVerdict",
    "CorrectnessVerdict",
    "judge_faithfulness",
    "judge_relevancy",
    "judge_correctness",
    "run_judges",
    "JudgeReport",
    # extractors
    "extract_keywords",
    "extract_summary",
    "extract_questions",
    "situate_chunk",
    "situate_chunks",
    "SituatedContext",
    # tools
    "Tool",
    "tool",
    "make_retrieval_tool",
    "search_web",
    "render_tool_result",
    # chat
    "Chat",
    "ToolLoop",
    "ChatEvent",
    # router
    "route",
    "RouteDecision",
    # finetune
    "make_training_pairs",
    "train_embedder",
    "evaluate_embedder",
    # datasets
    "mini_articles",
    "ai_tutor_knowledge",
    "prebuilt_chroma",
    "qa_dataset",
    "research_papers",
    # display
    "show_chunks",
    "show_answer",
    "show_eval_table",
    # errors
    "TaiAitutorError",
    "UnsupportedProviderError",
    "ProviderNotInstalledError",
    "MissingKeyError",
    "EmbeddingsNotAvailableError",
    "StructuredOutputError",
    "__version__",
]
