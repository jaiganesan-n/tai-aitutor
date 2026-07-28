# tai-aitutor

**The Towards AI course toolkit** — plain-Python building blocks for the RAG AI Tutor built across
the *Full Stack AI Engineer* course: provider-neutral LLM calls, embeddings, chunking, retrieval,
and evaluation.

This package replaces the LlamaIndex usage the course previously relied on — but it is **not a
framework**, and it is not a LlamaIndex clone. Every function here is code a course lesson builds
inline first; the package is where that code lives once you've written it yourself. Small flat
modules, plain functions, readable source. If you can't read a module in one sitting, that's a bug.

## Install

Requires **Python 3.12+** — 3.12 is Colab's current runtime, so the floor stays there
as long as Colab does (CI runs 3.12, 3.13, 3.14, plus the 3.15 pre-release).

```bash
# Course profile A (every notebook): the three providers
pip install "tai-aitutor[gemini,openai,anthropic]"

# Course profile B (retrieval lessons): + vector store and reranking
pip install "tai-aitutor[gemini,openai,anthropic,rag,rerank]"
```

During development (before the PyPI launch), install from GitHub pinned to a tag:

```bash
pip install "tai-aitutor[gemini,openai,anthropic] @ git+https://github.com/towardsai/tai-aitutor.git@v0.1.0"
```

Extras: `gemini` · `openai` · `anthropic` · `providers` (all three) · `rag` (chromadb) ·
`rerank` (cohere) · `local` (sentence-transformers) · `finetune` · `parse` (pypdf) ·
`web` (tavily, wikipedia) · `data` (huggingface-hub, pandas) · `all`

## Quickstart

```python
from tai_aitutor import configure, setup_notebook, generate, embed

IN_COLAB = setup_notebook(required_keys=("GOOGLE_API_KEY",))   # Colab Secrets or .env
PROVIDER = "gemini"   # @param ["gemini", "openai", "anthropic"]
configure(provider=PROVIDER)

print(generate("What is RAG, in one sentence?"))
qvec = embed("retrieval augmented generation", task="query")
```

Everything accepts per-call overrides, so comparisons never touch global state:

```python
generate("Same question", provider="openai", model="gpt-5-mini", reasoning_effort="minimal")
generate("Same question", provider="together")            # OpenAI-compatible endpoints built in
extract("Grade this answer...", schema=MyVerdict)         # native structured outputs, all providers
```

## The full pipeline in six lines

```python
from tai_aitutor import configure, load_jsonl, get_collection, ingest, answer, show_answer

configure(provider="gemini")
col = get_collection("ai_tutor_knowledge", path="./db")    # the collection IS the index
ingest(load_jsonl("kb.jsonl"), col)                        # chunk → embed → upsert (visible loop)
show_answer(answer("What is RAG?", col, top_k=5))          # retrieve → prompt → generate
```

## What's here (v1.0 — all five build-plan phases complete)

| Module | Public API | Replaces (LlamaIndex) |
|---|---|---|
| `config` | `configure`, `get_config`, `setup_notebook`, `require_keys`, `in_colab` | `Settings` |
| `llm` | `generate`, `generate_stream`, `generate_vision`, `extract`, `chat_completion` (+`Completion`, `ToolCall`), `ask_batch`, `Usage` | `OpenAI`, `GoogleGenAI`, `Perplexity`, `TogetherLLM`, `ChatMessage`, `structured_predict`, `BatchEvalRunner` fan-out |
| `embeddings` | `embed`, `embed_cohere`, `embed_local` | `OpenAIEmbedding`, `CohereEmbedding`, `HuggingFaceEmbedding`, `resolve_embed_model` |
| `tokens` | `n_tokens`, `truncate`, `estimate_cost` | (tiktoken hidden inside splitters) |
| `documents` | `Document`, `load_csv`, `load_jsonl`, `load_directory`, `load_files`, `load_wikipedia`, `load_hf_dataset` | `Document`, `SimpleDirectoryReader`, `WikipediaReader` |
| `chunking` | `Chunk`, `chunk`, `chunk_document`, `chunk_sentences`, `heading_aware_markdown_chunks`, `sentence_window_chunks` | `TokenTextSplitter`, `SentenceSplitter`, `SimpleNodeParser`, `SentenceWindowNodeParser`, `TextNode` |
| `vectorstore` | `get_collection`, `reset_collection`, `ingest`, `get_all_chunks`, `build_where_filter` | `ChromaVectorStore`, `StorageContext`, `IngestionPipeline`, `VectorStoreIndex.from_documents`, persist/load |
| `retrieval` | `search`, `ScoredChunk`, `expand_window`, `code_tokenize`, `BM25Index`, `rrf_fuse`, `hybrid_search`, `rerank`, `judge_rerank`, `hyde_search`, `decompose_question`, `subquestion_answer`, `multi_step_answer`, `pack_context` | `as_retriever().retrieve()`, `VectorIndexRetriever`, `NodeWithScore`, `MetadataReplacementPostProcessor`, `SimpleKeywordTableIndex`, `KeywordTableSimpleRetriever`, `CohereRerank`, `RankGPTRerank`, `BaseNodePostprocessor`, `HyDEQueryTransform`, `TransformQueryEngine`, `SubQuestionQueryEngine`, `LLMQuestionGenerator`, `QueryEngineTool`, `ToolMetadata`, `MultiStepQueryEngine`, `StepDecomposeQueryTransform`, `RetrieverQueryEngine` |
| `synthesis` | `answer`, `answer_with_sources`, `answer_stream`, `Answer`, `build_rag_prompt` | `as_query_engine().query()`, `get_response_synthesizer`, `Response.source_nodes` |
| `evals` | `QADataset` (legacy-JSON compatible), `make_qa_pairs`, `hit_rate`, `mrr`, `evaluate_retrieval`, `sweep_top_k`, `context_tokens`, `judge_faithfulness` / `judge_relevancy` / `judge_correctness`, `run_judges` | `generate_question_context_pairs`, `EmbeddingQAFinetuneDataset`, `RetrieverEvaluator`, `FaithfulnessEvaluator` / `RelevancyEvaluator` / `CorrectnessEvaluator`, `BatchEvalRunner` |
| `extractors` | `extract_keywords`, `extract_summary`, `extract_questions`, `situate_chunk(s)` | `KeywordExtractor`, `SummaryExtractor`, `QuestionsAnsweredExtractor` |
| `tools` | `tool()`, `Tool`, `make_retrieval_tool`, `search_web`, `render_tool_result` | `QueryEngineTool`, `ToolMetadata`, `TavilyToolSpec`, `LoadAndSearchToolSpec` |
| `chat` | `Chat` (full/window/summary memory), `ToolLoop`, `ChatEvent` | `as_chat_engine(chat_mode=...)`, `FunctionAgent`, `ReActAgent`, `AgentStream`, `ToolCallResult`, `Context`, `ChatSummaryMemoryBuffer` |
| `router` | `route`, `RouteDecision` | `RouterQueryEngine`, `LLMSingleSelector`, `PydanticSingleSelector` |
| `finetune` | `make_training_pairs`, `train_embedder`, `evaluate_embedder` | `EmbeddingAdapterFinetuneEngine`, `AdapterEmbeddingModel`, `EmbeddingQAFinetuneDataset` (training side) |
| `datasets` | `mini_articles`, `ai_tutor_knowledge`, `prebuilt_chroma`, `qa_dataset`, `research_papers` | the per-notebook `wget`/`hf_hub_download` cells |
| `display` | `show_chunks`, `show_answer`, `show_eval_table` | — |

Migrating a notebook? **[`MIGRATION.md`](MIGRATION.md)** has the symbol-by-symbol table.
Runnable end-to-end examples live in [`examples/`](examples/), including the Gradio tutor
app rebuilt on the package. Status and history: `BUILD_STATUS.md`, `CHANGELOG.md`;
the full design + LlamaIndex inventory: `docs/PACKAGE_PLAN.md`.

An agent from parts you built (the Gradio-app shape):

```python
from tai_aitutor import Chat, make_retrieval_tool, tool, search_web

chat = Chat(
    system="You are the course AI tutor. Ground answers in the knowledge base.",
    tools=[make_retrieval_tool(col, retriever=production_retriever), tool(search_web)],
    history="summary",                     # hand-rolled memory, packaged
)
for event in chat.ask_stream("What did we decide about chunk sizes?"):
    ...                                    # text / tool_call / tool_result events
```

The production retrieval chain, end to end:

```python
from tai_aitutor import (get_collection, get_all_chunks, BM25Index,
                         hybrid_search, rerank, pack_context, answer)

col = get_collection("ai_tutor_knowledge", path="./db")
bm25 = BM25Index().build(get_all_chunks(col))          # build once, save()/load() to reuse

def production_retriever(q):
    hits = hybrid_search(q, col, bm25)                  # dense 15 ∪ BM25 30 → RRF keep 30
    hits = rerank(q, hits)                              # Cohere v4, top 5, floor 0.10
    return pack_context(hits, max_tokens=30_000)        # the cost knob

print(answer("How does hybrid search work?", retriever=production_retriever))
```

## Design rules

1. **Teach-then-import.** A concept's first appearance is written out in its lesson notebook; later
   notebooks import it from here. Each module docstring names the lesson that builds it.
2. **Functions over object graphs.** No hidden state beyond one `configure()` call.
3. **Mirror production.** Where the production AI Tutor has an equivalent, constants and logic match.
4. **Loud failures.** Typo'd kwargs raise `TypeError` (the silently-swallowed
   `additional_kwrgs=` bug from the old stack is regression-tested), missing keys and missing
   extras say exactly what to install or set.

## Development

```bash
pip install -e ".[gemini,openai,anthropic,rerank]" pytest ruff
pytest
ruff check src tests
```

Tests run fully offline (provider SDKs are faked). Releases: tag `vX.Y.Z` → CI builds and publishes
to PyPI via Trusted Publishing (see `.github/workflows/release.yml`).

## License

MIT © Towards AI
