# Changelog

All notable changes to `tai-aitutor` are documented here.
Versioning: SemVer. Course notebooks always pin exact versions (`tai-aitutor==X.Y.Z`).

## [1.0.0] — 2026-07-28

Phase 5 (launch): datasets, migration guide, examples, release machinery. The package now
covers the complete LlamaIndex inventory from the course plan.

### Added
- `datasets` — one home for every download the notebooks make, filenames verified against
  the live notebooks: `mini_articles(with_embeddings=)`, `ai_tutor_knowledge()`,
  `prebuilt_chroma(variant="default"|"windowed")` (extracts the hosted Chroma store;
  loudly documents its OpenAI-embedding coupling), `qa_dataset("rag_eval_50")` (the shared
  50-question eval set), `research_papers(parsed=)`, `cache_dir()` (override with
  `TAI_AITUTOR_CACHE`). The org-account migration (Decision 6) is now a ONE-BLOCK edit in
  `datasets.py` instead of a 16-notebook sweep; a Gemini-embedded store slot is reserved.
- `MIGRATION.md` — the complete LlamaIndex → tai_aitutor mapping, symbol by symbol.
- `examples/`: `quickstart.py`, `production_retriever.py` (the four-row ablation +
  production chain), `gradio_tutor.py` (the Gradio lesson app rebuilt on `Chat` +
  `make_retrieval_tool` — the reference for the `ai-tutor-gradio-lesson` repo rewrite and
  the certification skeleton; no hard-coded models, `share=False`, `debug=False`).
- `CONTRIBUTING.md` — the teach-then-import rule, dev setup, release flow.

### Release
- Publish: `git tag v1.0.0 && git push --tags` after the one-time PyPI Trusted Publishing
  setup (header of `.github/workflows/release.yml`). First publish claims the (verified
  unclaimed) `tai-aitutor` name.

## [0.5.0] — 2026-07-28

Phase 4 (agents without a framework): router, tools, chat, fine-tuning.

### Added
- `llm.chat_completion` — the raw messages+tools layer under one normalized format
  (system/user/assistant/tool dicts), converted natively per provider: Gemini function
  calling, OpenAI Responses (`function_call` items), Anthropic tool_use blocks (adjacent
  same-role messages auto-merged), OpenAI-compatible endpoints. Returns a typed
  `Completion` (text, `ToolCall`s, usage, stop_reason). Replaces `ChatMessage` and the
  agent frameworks' hidden loops.
- `tools`: `tool()` decorator (JSON schema from type hints + docstring; loud error when a
  description is missing), `Tool`, `make_retrieval_tool` (the production
  `retrieve_tutor_context` shape; accepts a custom retriever for the hybrid+rerank chain),
  `search_web` (Tavily), `render_tool_result`. Replaces `QueryEngineTool` /
  `ToolMetadata` / `TavilyToolSpec`.
- `chat`: `ToolLoop` (the bare tool-calling loop, stateless, bounded by `max_iters`;
  tool errors go back to the model as text) and `Chat` — conversation memory as the
  visible message list with the lesson's three strategies: `history="full" | "window" |
  "summary"` (summary folds old turns via one `generate()` call and rides in the system
  prompt). `self.messages` is the full transcript; `self.last_context` is what was
  actually resent — the difference is the lesson. `ChatEvent` stream from `ask_stream`.
  Replaces `as_chat_engine(chat_mode=...)`, `FunctionAgent`, `ReActAgent`,
  `AgentStream`/`ToolCallResult`, workflow `Context`, `ChatSummaryMemoryBuffer`.
- `router`: `route(question, routes)` → typed `RouteDecision`, guaranteed to be one of
  your route names (case-insensitive + unambiguous-substring matching; raises instead of
  mis-routing). Replaces `RouterQueryEngine` + selectors — dispatch stays your `if/else`.
- `finetune`: `make_training_pairs` (the eval-lesson generator reused),
  `train_embedder` (sentence-transformers trainer + MultipleNegativesRankingLoss on
  `bge-small-en-v1.5` — the planned method upgrade over the old frozen-base adapter),
  `evaluate_embedder` (before/after hit rate + MRR with the same `RetrievalReport` ruler).
  Replaces `EmbeddingAdapterFinetuneEngine` / `AdapterEmbeddingModel` /
  `resolve_embed_model`.

## [0.4.0] — 2026-07-28

Phase 3 (advanced retrieval — the Section 7 stack) + Python floor raised.

### Changed
- **Requires Python >= 3.13** (team decision). CI matrix: 3.13, 3.14, and 3.15-dev
  (non-blocking); wheels build on 3.13. Code modernised to PEP 695 generics
  (`def extract[S: BaseModel](...)`).

### Added
- `code_tokenize` — the code-aware tokenizer (camelCase + dotted.path splitting, short
  terms kept), same function at index and query time.
- `BM25Index` — hand-rolled Okapi BM25 (k1=1.5, b=0.75, production constants) with
  versioned gzipped-JSON `save`/`load` (never pickle). Replaces
  `SimpleKeywordTableIndex` + `KeywordTableSimpleRetriever` — and actually computes BM25,
  which the old notebook never did.
- `rrf_fuse(k=60, keep=30)` — Reciprocal Rank Fusion; the `keep` cap is applied
  (regression test for the old ignored-top-k bug).
- `hybrid_search` — dense (top 15) ∪ BM25 (top 30) → RRF keep 30, production constants;
  `where` scopes both legs.
- `rerank` — Cohere `rerank-v4.0-fast`, top 5, score floor 0.10 (replaces `CohereRerank`);
  an explicit stage that eval tables can't silently skip.
- `judge_rerank` — LLM-as-judge reranking with the judge's ORDER and SCORES preserved
  (replaces `RankGPTRerank`/custom postprocessor; regression test for the
  thrown-away-ordering bug).
- Query transforms: `hyde_search` (replaces `HyDEQueryTransform`+`TransformQueryEngine`),
  `decompose_question` + `subquestion_answer` (replace `LLMQuestionGenerator` +
  `SubQuestionQueryEngine` + `QueryEngineTool`/`ToolMetadata`), `multi_step_answer`
  (replaces `MultiStepQueryEngine` + `StepDecomposeQueryTransform`).
- `pack_context` — the production 100k token budget knob.

## [0.3.0] — 2026-07-28

Phase 2 (evaluation + extractors): measure the pipeline, enrich the chunks.

### Added
- `evals`: `QADataset` — **drop-in reader/writer for the legacy
  `EmbeddingQAFinetuneDataset` JSON** (existing `rag_eval_dataset*.json` artifacts load
  unchanged; string `relevant_docs` coerced; deterministic `.sample(n, seed)` replaces the
  ad-hoc `subset_50` files); `make_qa_pairs` (typed question generation, seeded sampling —
  replaces `generate_question_context_pairs`); `hit_rate` / `mrr` / `evaluate_retrieval`
  (one retrieval pass computes both, keeps the per-query table + `misses()`; mirrors
  production `retrieval_metrics()`; works over ANY `(query, top_k)` retriever so rerankers
  actually get measured — replaces `RetrieverEvaluator.from_metric_names`); typed judges
  `judge_faithfulness` / `judge_relevancy` / `judge_correctness` with
  `FaithfulnessVerdict` / `RelevancyVerdict` / `CorrectnessVerdict` (1-5, ≥4.0 passes —
  replace the three Evaluator classes); `run_judges` thread-pool batch with `JudgeReport`
  aggregates (replaces `BatchEvalRunner` — no asyncio, no `nest_asyncio`).
- `extractors`: `extract_keywords` / `extract_summary` / `extract_questions` (replace
  `KeywordExtractor` / `SummaryExtractor` / `QuestionsAnsweredExtractor`; plug into
  `ingest(enrich=[...])`; never mutate inputs) and `situate_chunk` / `situate_chunks`
  (production's contextual-retrieval `SituatedContext` pattern).
- `display.show_eval_table` (the lesson ablation tables) and judge/QA prompts in `prompts`.

### Notes
- Acceptance check for the port: notebooks `06-Evaluate_RAG` and `Larger_Context_Larger_N`
  now reduce to package imports + their lesson-specific cells.

## [0.2.0] — 2026-07-28

Phase 1 (the pipeline core): documents → chunks → vector store → search → answer.

### Added
- `documents`: `Document`, `load_csv` (embeddings via `json.loads` — never `eval()`,
  regression-tested), `load_jsonl`, `load_directory` / `load_files` (replaces
  `SimpleDirectoryReader`), `load_wikipedia` (replaces `WikipediaReader`), `load_hf_dataset`.
- `chunking`: `Chunk`, `chunk` (replaces `TokenTextSplitter`), `chunk_document` (stable ids),
  `chunk_sentences` (replaces `SentenceSplitter`/`SimpleNodeParser`),
  `heading_aware_markdown_chunks` (the production chunker: 800/100, code blocks never split),
  `sentence_window_chunks` (replaces `SentenceWindowNodeParser`).
- `vectorstore`: `get_collection` / `reset_collection` (replaces `ChromaVectorStore` +
  `StorageContext` + persist/load), `ingest` (replaces `IngestionPipeline` +
  `VectorStoreIndex.from_documents`; visible chunk → enrich → embed → upsert loop),
  `get_all_chunks` (real enumeration — kills the `similarity_top_k=100000000` hack),
  `build_where_filter` (production `$eq`/`$in` source scoping).
- `retrieval`: `ScoredChunk` (replaces `NodeWithScore`), `search` (replaces
  `as_retriever().retrieve()`; requested `top_k` is enforced), `expand_window`
  (replaces `MetadataReplacementPostProcessor`).
- `synthesis`: `Answer` (replaces the Response object), `build_rag_prompt` (the visible
  RAG prompt), `answer` / `answer_with_sources` / `answer_stream` (replace
  `as_query_engine().query()` and `print_response_stream`), pluggable `retriever=` hook
  for the Section 7 lessons.
- `prompts` (shared constants) and `display` (`show_chunks`, `show_answer`).

### Notes
- The pipeline is now end-to-end: `configure → get_collection → ingest → answer`.
- No async anywhere — `nest_asyncio` cells die with the port.

## [0.1.0] — 2026-07-28

Phase 0 (scaffold + core). First installable version.

### Added
- `config`: `configure()` / `get_config()` (replaces LlamaIndex `Settings`), provider registry
  (Gemini default, OpenAI, Anthropic, plus OpenAI-compatible: Together, DeepSeek, Perplexity, Ollama),
  `setup_notebook()` (Colab/local detection, Colab Secrets vs `.env` key loading), `require_keys()`,
  model price table for `estimate_cost` (dated 2026-07).
- `llm`: `generate()`, `generate_stream()`, `generate_vision()`, `extract()` (native structured
  outputs on all three providers), `ask_batch()`, `Usage`.
  Replaces `llama_index.llms.openai.OpenAI`, `llama_index.llms.google_genai.GoogleGenAI`,
  `llama_index.llms.perplexity.Perplexity`, `llama_index.llms.together.TogetherLLM`,
  `ChatMessage`, `structured_predict`/`as_structured_llm`, and `BatchEvalRunner`-style fan-out.
- `embeddings`: `embed()` (Gemini default / OpenAI), `embed_cohere()` (v4, `input_type` asymmetry),
  `embed_local()` (sentence-transformers, e5 prefix handling). Replaces `OpenAIEmbedding`,
  `CohereEmbedding`, `HuggingFaceEmbedding`, `resolve_embed_model`.
- `tokens`: `n_tokens()`, `truncate()`, `estimate_cost()` (offline-safe fallback tokenizer).
- Packaging: PEP 621 metadata, extras (`gemini`, `openai`, `anthropic`, `rag`, `rerank`, `local`,
  `finetune`, `parse`, `web`, `data`, `all`), `py.typed`, CI (tests + lint), release workflow
  (PyPI Trusted Publishing).

### Notes
- Kwarg typos like `additional_kwrgs` now fail loudly (`TypeError`) instead of being silently
  swallowed — regression-tested.
