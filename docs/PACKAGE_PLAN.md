# `tai-aitutor` — Package Build Plan

**Replacing LlamaIndex across the Full Stack AI Engineer / LLM for Production course with our own package**

Draft v1.0 — July 28, 2026
Inputs: full code scan of both repos (`ai-tutor-rag-system` working copy + `Backup/ai-tutor-rag-system-main` original), Course Update Plan v2 (July 22, 2026), Live Course lesson index.

---

## 0. TL;DR

The course update (Decision 1, Course Update Plan v2) removes LlamaIndex everywhere and teaches every concept as a direct implementation. Twenty-plus notebooks have already been ported by copy-pasting the same helper functions (`generate()`, `embed()`, `chunk()`, `search()`, `answer()`, judges, metrics) into each notebook. This plan turns those repeated helpers into one package we own — **`tai-aitutor`** (import name `tai_aitutor`) — so the port finishes faster, the already-ported notebooks lose their duplicated cells, and future maintenance becomes "bump one pin," not forty bespoke fixes.

The plan contains: the complete inventory of every LlamaIndex class, function, import, method, and pip pin found in both repos (Section 2); the package architecture and full public API, module by module with signatures (Section 4); a symbol-by-symbol LlamaIndex → `tai_aitutor` migration map (Section 5); a notebook-by-notebook migration matrix (Section 6); build phases with sequencing (Section 7); testing/CI, versioning, risks, and open decisions (Sections 8–12).

Distribution: **GitHub-install first** (`pip install git+...@v0.x.y` pinned to tags), **PyPI at course launch** (verify the `tai-aitutor` name is free before announcing; reserve it early with a 0.0.1 placeholder if we want insurance).

One design rule governs everything: **the notebook teaches the concept inline first; the package is only allowed to carry code a previous lesson already taught.** It is the course's own code, packaged: every public symbol traces back to a cell the student has written, and its docstring names that lesson.

---

> **SUPERSEDED IN PART — 13 August 2026.** The API inventory in Section 4 below, and
> every symbol list that follows from it, predate the strip pass described in
> `tai_aitutor_strip_spec.md`. That pass removed `chat.py`, `extractors.py`,
> `finetune.py`, the `answer` family, `ingest`, `hybrid_search`, `subquestion_answer`,
> `multi_step_answer`, `pack_context`, `make_qa_pairs`, `run_judges`, `context_tokens`,
> `make_retrieval_tool`, `render_tool_result`, `truncate`, `estimate_cost`,
> `MODEL_PRICES`, the streaming/vision/batch wrappers, the message-and-tool layer, the
> non-CSV loaders, and `BM25Index` persistence; and it changed the signatures of
> `generate`, `extract`, `embed`, `embed_cohere`, `embed_local`, `hit_rate`, and
> `mrr` (now `reciprocal_rank`). **Read `README.md` for the current public surface.**
> Sections 1-3 (rationale, LlamaIndex inventory, principles) still stand.

## 1. Why a package, and what it carries

**Why.** The July decision replaced LlamaIndex with direct implementations. The first 20 ported notebooks prove the approach but expose the cost: every notebook re-defines the same 5–15 helper functions. Today `generate()` is defined in at least 17 notebooks, `embed()`/`chunk()`/`search()` in ~10, and the eval helpers (`make_qa_pairs`, `hit_rate`, `mrr`, three judges) are duplicated wherever evaluation appears. Every dataset URL change, model rename, or bug fix now multiplies across copies — exactly the maintenance trap the update was meant to end. A package we control gives us: one place to fix, one pin to bump per notebook, the same code in course and (eventually) production shape, and no dependence on a framework's roadmap.

**What it carries, and on what terms.** The pedagogical core of Decision 1 is that students *see* the code. So:

1. **Build-inline-first rule.** A concept's first appearance is always written out in the notebook (chunking in the chunking lesson, BM25 in the hybrid lesson, judges in the eval lesson). Only *later* notebooks import it from `tai_aitutor`. The package README and each docstring link back to the lesson where the code is taught.
2. **Readable source is a feature.** Small flat modules (target <300 lines each), plain functions, no inheritance trees, no `**kwargs` soup, no lazy loading tricks. A student who clicks through to the source should find the same code they wrote in the lesson, plus error handling.
3. **Thin over general.** We implement exactly what the course uses. Where LlamaIndex offered a configurable class covering many backends, we ship the one path the lessons take. When a lesson needs a variant, we add a function, not an abstraction layer.
4. **Mirror production.** Where production (`ai-tutor-app`) has an equivalent (`app/chroma_rag.py` chunking, BM25, RRF, rerank, token budget, `build_where_filter`; `evals/grade.py` retrieval metrics; `add_context_to_nodes.py` contextual retrieval), the package copies its logic and constants so course code and production code are the same thing at two sizes.

---

## 2. Complete LlamaIndex inventory (what the package must replace)

### 2.1 Footprint

Scanned: all 51 notebooks in the original repo, all 58 notebooks in the working repo, `scripts/`, and both `requirements.txt` files — code cells parsed from JSON, multi-line imports resolved, commented-out imports excluded from counts (but noted where meaningful).

| Measure | Original repo | Working repo (after the 20-notebook update) |
|---|---|---|
| Notebooks with LlamaIndex in code | **31** | **23** (of which ~9 are legacy files kept for archiving; `Parsing_PDFs.ipynb` matches only as a dataset filename, see 2.6) |
| Distinct classes/functions imported | **82 symbols** from ~60 `llama_index.*` module paths | subset of the same 82 |
| `Settings.*` global assignments | `Settings.llm` (53 uses), `Settings.embed_model` (38), `Settings.text_splitter` (2), `Settings.chunk_size` (1), `Settings.chunk_overlap` (1) | shrinking with each port |
| Method-level API surface | ~30 distinct methods (`.as_query_engine()` in 40 files, `.query()` 47, `.run()` 48, `.from_vector_store()` 34, `.retrieve()` 14, …) | subset |
| `llama-index-*` pip pins in install cells | 25+ distinct package pins (see 2.5) | subset |

Notebooks already ported clean (no LlamaIndex left): `02-Basic_RAG`, `03-From_Script_to_Pipeline` (new), `04-RAG_with_VectorStore`, `05-Improve_Prompts_+_Add_Source`, `06-Evaluate_RAG`, `15-Use_OpenSource_Models`, `Larger_Context_Larger_N`, `Selecting_Embedding_Models` (new, merges the 3 embedding notebooks), `Multimodal_LLMs` (new), `Parsing_PDFs` (new, replaces LlamaParse), `Structured(JSON)_PDF_Data_Extraction`, `Applied_Structured_Outputs` (new), `Crawl_a_Website`, `Firecrawl_Scraping`, `Perplexity_Web_Api`, `Web_Search_API_Tavily`, plus the non-RAG updates (`Prompting_101`, `RAG_101`, `Evaluating_and_Iterating_Prompts`, `Intro_to_Large_Language_Models`).

### 2.2 Full symbol inventory — core framework

Counts are distinct notebooks in **orig / working** repo. "Replacement" names the `tai_aitutor` API from Section 4.

**A. Data structures and global config**

| LlamaIndex symbol (import path) | Used in (orig/upd) | What it does for us | Replacement |
|---|---|---|---|
| `llama_index.core.Document` | 17 / 10 | Wraps text + metadata before indexing (`Document(text=..., metadata={title,url,source_name})`) | `tai_aitutor.Document` dataclass (`text`, `metadata`, `id`) — or plain dicts at notebook level |
| `llama_index.core.schema.Document` | 2 / 1 | Same class, alternate import path | same |
| `llama_index.core.schema.TextNode` | 2 / 1 | Chunk with stable `id_` (finetune corpus in `08`) | `tai_aitutor.Chunk` dataclass (`id`, `text`, `metadata`, `embedding?`) |
| `llama_index.core.schema.BaseNode` | 1 / 0 | Type hint in custom code | `Chunk` |
| `llama_index.core.schema.NodeWithScore` | 2 / 2 | Scored retrieval hit (custom retriever in `11`, judge postprocessor in `17`) | `tai_aitutor.ScoredChunk` (`chunk`, `score`, `rank`) |
| `llama_index.core.schema.MetadataMode` | 1 / 0 | `node.get_content(metadata_mode=NONE)` in eval corpus building (`06`) | `chunk.text` (explicit fields; no modes) |
| `llama_index.core.Settings` | 23 / 17 | Global default LLM / embed model / splitter | `tai_aitutor.config.configure(provider=..., chat_model=..., embed_model=..., embed_provider=...)` + `get_config()`; the notebook `PROVIDER` dropdown feeds it |
| `llama_index.core.QueryBundle` | 2 / 2 | Wraps a query string for retriever/postprocessor calls | plain `str` argument |
| `llama_index.core.llms.ChatMessage` | 1 / 0 | Chat message object (Perplexity workflow) | `{"role": ..., "content": ...}` dicts |
| `llama_index.core.llms.utils.LLM` | 1 / 0 | Type hint | n/a (functions take no LLM objects; provider chosen by config/argument) |
| `llama_index.core.PromptTemplate` / `llama_index.core.prompts.PromptTemplate` | 3 / 1 | Template for RankGPT prompt (`17`), workflow system prompt (Perplexity) | f-strings; shared prompt constants live in `tai_aitutor.prompts` |

**B. Readers / loaders**

| Symbol | Used in | What it does | Replacement |
|---|---|---|---|
| `llama_index.core.SimpleDirectoryReader` | 4 / 4 | Load a folder/files into Documents (`13` wiki data, `LlamaIndex_101`, `More_Api` w/ `file_extractor`) | `tai_aitutor.documents.load_directory(path)` / `load_files([...])` (plain text/markdown/pdf via `parsing`) |
| `llama_index.readers.wikipedia.WikipediaReader` | 1 / 1 | Wikipedia pages for router lesson (`13`) | `documents.load_wikipedia(titles)` using the `wikipedia` pkg directly |
| `llama_index.readers.web.FireCrawlWebReader` | 1 / 0 | Firecrawl crawl → Documents | already ported: `firecrawl-py` SDK directly (keep; no package code needed beyond `Document`) |
| `llama_parse.LlamaParse` (+ `LLAMA_CLOUD_API_KEY`, JSON mode, `file_extractor={".pdf": parser}`) | 2 / 2 | Managed PDF parsing (`LlamaParse.ipynb`, `More_Api_And_Tools`) | already replaced by `Parsing_PDFs.ipynb` approach: `parsing.parse_pdf()` (pypdf baseline) + `parsing.parse_pdf_with_llm()` (native file understanding); lesson keeps an honest "when managed parsers earn their keep" note |

**C. Splitters / tokenizer / node parsers** *(the "Splitter and tokenizer" start of the user's list)*

| Symbol | Used in | What it does | Replacement |
|---|---|---|---|
| `llama_index.core.node_parser.TokenTextSplitter` + `llama_index.core.text_splitter.TokenTextSplitter` (two import paths) | 5+5 / 2+4 | Token-window chunking, `separator=" ", chunk_size=512, chunk_overlap=128` (course standard); `.get_nodes_from_documents()`, `.split_text()` | `chunking.chunk(text, chunk_size=512, chunk_overlap=128)` (tiktoken-based; already written in ported notebooks) |
| `llama_index.core.node_parser.SentenceSplitter` | 7 / 3 | Sentence-aware chunking (`03` 768/64, `Crawl`/`Firecrawl` 512/30, `Settings.text_splitter`) | `chunking.chunk_sentences(text, chunk_size, chunk_overlap)` |
| `llama_index.core.node_parser.SimpleNodeParser` | 2 / 2 | `.from_defaults(chunk_size=768, chunk_overlap=64)` (`08`, `LlamaIndex_101`) | `chunking.chunk(...)` |
| `llama_index.core.node_parser.SentenceWindowNodeParser` | 1 / 1 | Sentence-window retrieval prep (`Advanced_Retriever`), `window` metadata | `chunking.sentence_window_chunks(text, window_size=3)` — stores neighbor window in `chunk.metadata["window"]` |
| (tokenizer, implicit) | everywhere | LlamaIndex used tiktoken internally | `tokens.n_tokens(text, model=...)` — explicit and taught; also `tokens.truncate(text, max_tokens)` |

**D. Metadata extractors** *(ingestion enrichment used in `07`, `13`, `Metadata_Filtering`, all three embedding notebooks, `Larger_Context`)*

| Symbol | Used in | What it does | Replacement |
|---|---|---|---|
| `llama_index.core.extractors.KeywordExtractor` | 7 / 6 | `keywords=10` per chunk via LLM | `extractors.extract_keywords(chunks, n=10)` — typed `generate()` call |
| `llama_index.core.extractors.SummaryExtractor` | 6 / 5 | Per-chunk summaries (`summaries=["prev","self"]`) | `extractors.extract_summary(chunks)` |
| `llama_index.core.extractors.QuestionsAnsweredExtractor` | 6 / 5 | "questions this chunk answers" (`questions=3`) | `extractors.extract_questions(chunks, n=3)` |
| — (production equivalent) | — | Contextual retrieval: situating summary per chunk (`add_context_to_nodes.py`, `SituatedContext`) | `extractors.situate_chunk(chunk, document)` — used by the Section 13 contextual-retrieval notebook |

**E. Ingestion pipeline**

| Symbol | Used in | What it does | Replacement |
|---|---|---|---|
| `llama_index.core.ingestion.IngestionPipeline` | 9 / 5 | `transformations=[splitter, extractors..., embed_model]` + `vector_store=`; `.run(documents=..., show_progress=True)` | `vectorstore.ingest(docs, collection, chunker=chunk, enrich=[...], embed_fn=embed, batch_size=...)` — a visible loop: chunk → (enrich) → embed → upsert (already written in `05`/`06`; package version adds retry + progress) |

**F. Indexes, vector stores, storage**

| Symbol | Used in | What it does | Replacement |
|---|---|---|---|
| `llama_index.core.VectorStoreIndex` | 28 / 20 | The central object: `.from_documents()` (14 files), `.from_vector_store()` (34), ctor over nodes, `.insert()`, and the three factories `.as_query_engine()` (40) / `.as_retriever()` (19) / `.as_chat_engine()` (2) | dissolved into explicit stages: `ingest()` + `search()` + `answer()` / `Chat` — there is no index object; Chroma *is* the index |
| `llama_index.vector_stores.chroma.ChromaVectorStore` | 20 / 15 | Adapter over a chroma collection | `chromadb` used directly: `vectorstore.get_collection(name, path=None)` (PersistentClient/EphemeralClient), plus `vectorstore.reset_collection`, `collection.upsert/query/get` |
| `llama_index.vector_stores.qdrant.QdrantVectorStore` | 1 / 1 | Metadata-filtering demo backend (`Metadata_Filtering`) | dropped — lesson moves to Chroma `where` filters (the production feature); Qdrant/pgvector stay as a one-line mention |
| `llama_index.core.StorageContext` (+ `storage.storage_context` path) | 3 / 2 | Binds vector store / persist dir | not needed; Chroma `PersistentClient(path=...)` is the persistence |
| `llama_index.core.load_index_from_storage` + `index.storage_context.persist()` | 1 / 1 | Local index persistence (`LlamaIndex_101`) | not needed (archive with `LlamaIndex_101`); persistence = Chroma path |
| `llama_index.core.SimpleKeywordTableIndex` | 1 / 1 | Keyword leg of old "hybrid" (`11`) | `retrieval.BM25Index` — real Okapi BM25 (k1=1.5, b=0.75), code-aware tokenizer, `save/load` (versioned gzipped JSON, mirrors production) |

**G. Retrievers, query engines, transforms, postprocessors** *(the "Advanced RAG" block)*

| Symbol | Used in | What it does | Replacement |
|---|---|---|---|
| `.as_query_engine(similarity_top_k=k, streaming=..., response_mode=...)` + `.query()` | 40 / 47 files use `.query()` | RAG in one call | `answer(question, top_k=k)` / `answer_with_sources(...)` / `answer_stream(...)` — prompt is a visible f-string |
| `.as_retriever(similarity_top_k=k)` + `.retrieve()` | 19 / 14 | Dense retrieval | `retrieval.search(query, top_k=5, where=None)` → `list[ScoredChunk]` |
| `llama_index.core.retrievers.VectorIndexRetriever` | 1 / 1 | Explicit dense retriever (`11`; also Gradio app, top-k 15) | `retrieval.search(...)` |
| `llama_index.core.retrievers.KeywordTableSimpleRetriever` | 1 / 1 | Keyword retriever (`11`) | `BM25Index.search(query, top_k)` |
| `llama_index.core.retrievers.BaseRetriever` (subclassed `CustomRetriever`) | 1 / 1 | Hand-rolled hybrid merge (`11`) — the round-robin bug lives here | `retrieval.rrf_fuse(dense_hits, bm25_hits, k=60, keep=30)` + `retrieval.hybrid_search(query, ...)`; result caps actually applied |
| `llama_index.core.query_engine.RetrieverQueryEngine` + `get_response_synthesizer` | 1+1 / 1+1 | Wrap custom retriever + LLM synthesis | `answer(question, retriever=hybrid_search)` — synthesis is just the prompt + `generate()` |
| `llama_index.core.query_engine.RouterQueryEngine` + `selectors.LLMSingleSelector` / `PydanticSingleSelector` | 1+1+1 / same | Route query to one of several engines (`13`) | `router.route(question, routes={...})` — one typed classification call + `if/else` dispatch |
| `llama_index.core.query_engine.SubQuestionQueryEngine` + `question_gen.LLMQuestionGenerator` (2 paths) + `tools.QueryEngineTool` + `tools.ToolMetadata` | 2+2+4+2 / 2+2+4+2 | Decompose → retrieve per sub-question → synthesize (`12`, `LlamaIndex_101`) | `retrieval.decompose_question(question)` (typed list) + loop of `search()` + `answer()`; shown as ~15 visible lines, packaged as `retrieval.subquestion_answer(...)` |
| `llama_index.core.query_engine.multistep_query_engine.MultiStepQueryEngine` + `StepDecomposeQueryTransform` | 1+1 / 1+1 | Iterative multi-step querying (`12`) | `retrieval.multi_step_answer(question, max_steps=3)` — a plain loop: ask, retrieve, refine |
| `TransformQueryEngine` + `HyDEQueryTransform(include_original=True)` | 1+1 / 1+1 | HyDE (`12`) | `retrieval.hyde_search(query, top_k)` — generate hypothetical answer, embed it, search; ~6 lines |
| `llama_index.postprocessor.cohere_rerank.CohereRerank` | 1 / 1 | Cohere rerank post-step (`10`; `top_n=2`, `rerank-english-v3.0`) | `retrieval.rerank(query, hits, model="rerank-v4.0-fast", top_n=5, floor=0.10)` — direct `cohere.ClientV2.rerank()`, production constants |
| `llama_index.core.postprocessor.rankGPT_rerank.RankGPTRerank` | 1 / 1 | LLM-as-reranker (`17`) | `retrieval.judge_rerank(query, hits, top_n=3)` — typed judge scores, order and scores preserved (kills the thrown-away-ordering bug) |
| `llama_index.core.postprocessor.MetadataReplacementPostProcessor` | 1 / 1 | Swap chunk for its stored window (`Advanced_Retriever`) | `retrieval.expand_window(hits)` — read `metadata["window"]`; mechanism visible |
| `llama_index.core.postprocessor.types.BaseNodePostprocessor` (subclassed `OpenaiAsJudgePostprocessor`) | 1 / 1 | Custom judge postprocessor base (`17`) | not needed — `judge_rerank` is a plain function |
| `node_postprocessors=[...]` kwarg | 3 / 3 | Attach postprocessors to engines — the `10` eval bug (passed to `.as_retriever()`, silently ignored) | explicit call order: `hits = search(...)`; `hits = rerank(...)`; impossible to silently skip |
| `llama_index.core.vector_stores.{MetadataFilters, MetadataFilter, FilterOperator, FilterCondition}` | 1 each / 1 each | Metadata filtering (`Metadata_Filtering`: `EQ`, `TEXT_MATCH`, `AND`) | Chroma `where=` dicts + `vectorstore.build_where_filter(sources)` (production's `$eq` / `$in` shape); full-text via Chroma `where_document={"$contains": ...}` |

**H. Agents, chat engines, workflows**

| Symbol | Used in | What it does | Replacement |
|---|---|---|---|
| `llama_index.core.agent.workflow.FunctionAgent` | 3 / 2 | Tool-calling agent (`13` router-as-agent, `Web_Search_API`, `Tavily`) | `chat.ToolLoop` — a native tool-calling loop over `generate()` with tool schemas; taught inline first in the Tavily lesson (already ported there as `answer_with_search`) |
| `llama_index.core.agent.workflow.ReActAgent` | 3 / 2 | ReAct agent (`14`, `Web_Search_API`, `Tavily` demo) | same `chat.ToolLoop`; ReAct pattern shown as prose + prompt in the lesson, not a class |
| `llama_index.core.agent.workflow.AgentStream` / `ToolCallResult` | 2+2 / 1+1 | Streaming agent events | `chat` streaming yields typed events (`text`, `tool_call`, `tool_result`) — plain generator |
| `llama_index.core.workflow.Context` | 3 / 2 | Agent session memory (`ctx = Context(agent)`) | `chat.Chat` object holds `messages`; memory strategies below |
| `llama_index.core.workflow.{Workflow, step, Event, StartEvent, StopEvent}` | 1 each / 0 | The Perplexity lesson's 400-tool analysis project | already ported: plain Python loop + Pydantic structured outputs; for orchestration patterns the course points to the Section 11 workflow-patterns lesson |
| `.as_chat_engine(chat_mode=..., memory=...)`, `.chat()`, `.stream_chat()` | 2 / 3 files | Conversational RAG (`14`) | `chat.Chat(system=..., tools=[retrieve_tool], history="full"|"window"|"summary")` with `.ask()` / `.ask_stream()`; the memory modes are hand-rolled in the lesson first (full history, sliding window, summarize-old-turns) |
| `llama_index.core.tools.QueryEngineTool` / `ToolMetadata` | 4+2 / 4+2 | Wrap a query engine as an agent tool | `tools.tool(fn, name, description)` — plain function + JSON schema; the retrieval tool is `tools.make_retrieval_tool(collection)` |
| `llama_index.tools.tavily_research.TavilyToolSpec` | 1 / 0 | Tavily search tool | already ported: `tavily-python` direct + `search_web` tool (promote to `tools.search_web`) |
| `llama_index.tools.google.GoogleSearchToolSpec` + `tool_spec.load_and_search.LoadAndSearchToolSpec` | 1+1 / 1+1 | Google CSE agent tools (`Web_Search_API`) | dropped — notebook archived (Google CSE closed to new customers; Tavily replaced it) |

**I. Evaluation** *(the "Eval" block)*

| Symbol | Used in | What it does | Replacement |
|---|---|---|---|
| `llama_index.core.evaluation.generate_question_context_pairs` | 4 / 3 | Synthetic QA dataset over chunks | `evals.make_qa_pairs(collection, n_chunks=25, questions_per_chunk=1)` — typed LLM call (already written in `06`) |
| `llama_index.core.evaluation.EmbeddingQAFinetuneDataset` (+ 2 legacy import paths in `llama_index.finetuning`) | 4+1+1 / 1+1+1 | QA dataset container: `queries`, `corpus`, `relevant_docs`; `.from_json()` (11 files), `.save_json()` (10) | `evals.QADataset` dataclass with the **same JSON schema** (drop-in: existing `rag_eval_dataset.json` artifacts keep loading) — `.save()` / `QADataset.load()` |
| `llama_index.core.evaluation.RetrieverEvaluator.from_metric_names(["mrr","hit_rate"])` + `.aevaluate_dataset()` | 7 / 5 | Retrieval eval | `evals.hit_rate(qa, search_fn, top_k)` and `evals.mrr(qa, search_fn, top_k)` (~10 lines each, mirror production `retrieval_metrics()` in `evals/grade.py`); `evals.evaluate_retrieval(...)` returns both |
| `llama_index.core.evaluation.FaithfulnessEvaluator` | 6 / 3 | Answer grounded in context? | `evals.judge_faithfulness(answer, context)` → `FaithfulnessVerdict` (Pydantic, already written in `06`) |
| `llama_index.core.evaluation.RelevancyEvaluator` | 6 / 3 | Answer+context relevant to question? | `evals.judge_relevancy(question, answer, context)` → `RelevancyVerdict` |
| `llama_index.core.evaluation.CorrectnessEvaluator` | 1 / 0 | Score vs reference answer (1–5) | `evals.judge_correctness(question, answer, reference)` → `CorrectnessVerdict` |
| `llama_index.core.evaluation.BatchEvalRunner` (+ `nest_asyncio` boilerplate) | 6 / 3 | Batch async judging | `evals.run_judges(questions, answers, contexts, judges=[...], concurrency=8)` — simple thread pool; **no `nest_asyncio` anywhere** |

**J. LLM wrappers** *(replaced by the provider-neutral `generate()` family)*

| Symbol | Used in | What it does | Replacement |
|---|---|---|---|
| `llama_index.llms.openai.OpenAI` | 24 / 17 | GPT-5 family chat; `additional_kwargs={'reasoning_effort':'minimal'}`; also the `additional_kwrgs` typo bug (silently accepted, setting never applied) — `.complete()`, `.chat()`, `.predict()`, `.structured_predict()`, `.as_structured_llm()` | `llm.generate(prompt, system=None, model=None, provider=None)`; reasoning effort handled inside the OpenAI branch; typos impossible (real kwargs, type-checked) |
| `llama_index.llms.google_genai.GoogleGenAI` | 10 / 5 | Gemini chat (`generation_config`, temperature, max_tokens) | `generate()` Gemini branch (google-genai SDK) — course default |
| `llama_index.llms.perplexity.Perplexity` | 1 / 0 | Perplexity sonar calls | already ported: OpenAI-compatible endpoint via `base_url` — `llm.generate(..., provider="perplexity")` or the lesson's raw client |
| `llama_index.llms.together.TogetherLLM` | 1 / 0 | Together AI open-weight models | already ported: OpenAI-compatible `base_url` — `provider="together"`; the lesson teaches the `base_url` pattern explicitly |
| — (Anthropic) | 0 today | Anthropic appears in zero old notebooks | first-class `provider="anthropic"` branch (Decision 2) |
| — (Ollama / DeepSeek) | 0 today | local + cheap tiers for Section 13 | `provider="ollama"` (local base_url) and `provider="deepseek"` branches |
| `llm.complete()` / `.chat()` / streaming / `.print_response_stream()` | various | direct generation + streaming | `generate()` / `generate_stream()`; `chat.Chat.ask_stream()` |
| structured output (`structured_predict`, `as_structured_llm`) | 2 / 2 | typed outputs | `llm.extract(prompt, schema: type[BaseModel], system=None)` — native structured outputs per provider (already written in `Structured(JSON)` + `Applied_Structured_Outputs`) |
| vision (multimodal calls) | new | image inputs | `llm.generate_vision(prompt, image_bytes, mime_type)` (already written in `Multimodal_LLMs`) |

**K. Embedding wrappers**

| Symbol | Used in | What it does | Replacement |
|---|---|---|---|
| `llama_index.embeddings.openai.OpenAIEmbedding` | 24 / 17 | `text-embedding-3-small` everywhere; `mode="text_search"` once (`13`) | `embeddings.embed(texts, task="document"|"query", provider=None)` — OpenAI branch |
| — (Gemini embeddings) | 0 today | none used Gemini embeddings | `gemini-embedding-001` as **default** (Decision 2 embeddings caveat) — this is why the prebuilt vector store must be re-embedded and re-hosted |
| `llama_index.embeddings.cohere.CohereEmbedding` | 2 / 2 | `embed-english-v3.0`, `input_type=search_document/search_query` | `embeddings.embed_cohere(texts, task)` — direct `cohere.ClientV2.embed`, upgrade demo to `embed-v4.0` (production model); the `input_type` asymmetry is taught |
| `llama_index.embeddings.huggingface.HuggingFaceEmbedding` | 3 / 2 | Local models (`BAAI/bge-small-en-v1.5`, `intfloat/e5-small-v2`) | `embeddings.embed_local(texts, model_name)` via `sentence-transformers` |
| `llama_index.core.embeddings.resolve_embed_model("local:...")` | 1 / 1 | String → model resolution (`08`, `15`) | explicit `embed_local(...)`; no string registry |
| `.get_text_embedding()` / `.get_query_embedding()` / batch | various | embed calls | `embed(texts, task=...)` with automatic batching + retry |
| `llama_index.embeddings.adapter.AdapterEmbeddingModel` | 1 / 1 | Load linear-adapter finetuned embedder (`08`) | dropped — method changes (see L) |

**L. Fine-tuning**

| Symbol | Used in | What it does | Replacement |
|---|---|---|---|
| `llama_index.finetuning.EmbeddingAdapterFinetuneEngine` (`.finetune()`, `.get_finetuned_model()`) | 1 / 1 | Linear adapter on frozen base embedder (`08`) | **method replaced per plan**: `finetune.train_embedder(base_model="BAAI/bge-small-en-v1.5", train, val, loss="MNRL", epochs, out_dir)` using the `sentence-transformers` trainer + MultipleNegativesRankingLoss |
| commented remnants: `generate_qa_embedding_pairs`, `LinearAdapterEmbeddingModel`, `SentenceTransformersFinetuneEngine` | (comments in `08`) | earlier experiments | `finetune.make_training_pairs(...)` (same typed-LLM pattern as `evals.make_qa_pairs`); before/after measured with `evals.hit_rate`/`mrr` |
| GPT fine-tuning (`GPT_4o_mini_Fine_Tuning`) | — | RAG scaffolding (index + query engine) used to *generate* training data; the fine-tune itself is the OpenAI API | scaffolding → `search()`/`answer()`; OpenAI fine-tuning stays native (vendor lesson) |

### 2.3 Method-level surface being replaced (for the API-parity checklist)

`.as_query_engine()` (40 files) · `.query()` (47) · `.run()` (48, IngestionPipeline/agents) · `.from_vector_store()` (34) · `.as_retriever()` (19) · `.from_defaults()` (15) · `.from_documents()` (14) · `.retrieve()` (14) · `.from_json()` (11) · `.save_json()` (10) · `.load_data()` (7) · `.get_nodes_from_documents()` (6) · `.generate()` (4) · `.chat()` (3) · `.as_chat_engine()` (2) · `.stream_chat()` (2) · `.finetune()` (2) · `.get_finetuned_model()` (2) · `.persist()` (2) · `.complete()` (2) · `.structured_predict()` (2) · `.print_response_stream()` (2) · `.evaluate()` / `.aevaluate_dataset()` · `.insert()` · `.predict()` · `.as_structured_llm()` · `.split_text()` · `.get_content()` · plus response object attributes (`response.response`, `response.source_nodes`, `node.score`, `node.text`, `node.metadata`) → replaced by `Answer` dataclass (`text`, `sources: list[ScoredChunk]`, `usage`).

Also `Settings.llm` (53), `Settings.embed_model` (38), `Settings.text_splitter` (2), `Settings.chunk_size` (1), `Settings.chunk_overlap` (1) → `config.configure(...)`.

### 2.4 Constructor-argument surface worth preserving (so ports feel familiar)

`chunk_size`, `chunk_overlap`, `separator`, `similarity_top_k`→`top_k`, `top_n`, `metadata`/`text` on Document, `show_progress`, `temperature`, `max_tokens`, `where` filters, `input_type`/`task`, `window_size`, `include_original` (HyDE), `streaming`. Names are kept where they exist in the ported notebooks already.

### 2.5 pip pins found in install cells (all must disappear)

Core pins seen across notebooks: `llama-index==0.13.3 / 0.14.0 / 0.14.10 / 0.14.12 / 0.14.19` and plugin pins: `llama-index-llms-openai` (0.2.9→0.6.12), `llama-index-llms-google-genai` (0.3.0/0.5.0/0.9.0), `llama-index-llms-perplexity==0.4.0`, `llama-index-llms-together==0.4.1`, `llama-index-llms-gemini==0.3.5` (legacy freeze), `llama-index-llms-azure-openai==0.4.1`, `llama-index-embeddings-openai` (0.2.5→0.6.0), `llama-index-embeddings-huggingface` (0.6.0/0.6.1), `llama-index-embeddings-cohere` (0.2.1/0.6.0/0.6.1), `llama-index-embeddings-adapter==0.4.1`, `llama-index-embeddings-instructor==0.4.0`, `llama-index-vector-stores-chroma` (0.2.0→0.5.5), `llama-index-vector-stores-qdrant==0.8.4`, `llama-index-postprocessor-cohere-rerank` (0.2.1/0.5.1), `llama-index-finetuning` (0.4.0/0.4.1), `llama-index-readers-web` (0.5.1/0.5.3/0.6.0), `llama-index-readers-wikipedia==0.4.0`, `llama-index-tools-google==0.6.2`, `llama-index-tools-tavily-research` (unpinned), `llama-index-question-gen-guidance==0.4.1`, `llama-parse` (0.5.6/0.6.54), plus the dead root `requirements.txt` freeze (16 more `llama-index-*` pins incl. `llama-index-agent-openai`, `llama-index-cli`, `llama-index-core==0.11.13.post1`, `llama-index-indices-managed-llama-cloud`, `llama-index-legacy`, `llama-index-multi-modal-llms-openai`, `llama-index-program-openai`, `llama-index-question-gen-openai`, `llama-index-readers-file`, `llama-index-readers-llama-parse`).

Replaced by the shared install profiles (Course Update Plan, provider sweep) + one package pin:

```
# profile A (all notebooks)
google-genai  openai  anthropic  tai-aitutor==<pinned>
# profile B (retrieval lessons) adds
chromadb  cohere
# per-lesson extras stay per-lesson (sentence-transformers, tavily-python, firecrawl-py, pypdf, wikipedia, ...)
```

### 2.6 LlamaIndex references outside notebook code (must be handled by the same program)

1. **Root `requirements.txt`** (both repos): September-2024 pip freeze with 16 `llama-index-*` pins — delete (already flagged in the update plan as dead).
2. **`nest_asyncio` cells** in ~24 notebooks: exist only for LlamaIndex's async internals — delete with each port (the package uses no asyncio-in-notebook).
3. **`LLAMA_CLOUD_API_KEY`** env var setup (4 notebooks) — dies with LlamaParse removal.
4. **Gradio lesson app repo** (`towardsai/ai-tutor-gradio-lesson`, audited in the update plan): `OpenAIAgent.from_tools` (deprecated), `RetrieverTool`, `VectorIndexRetriever` (top-k 15), `ChatSummaryMemoryBuffer`, UTF-16 `requirements.txt` with 16 llama-index pins → rebuild on `tai_aitutor` (`chat.Chat` + `tools.make_retrieval_tool` + summary memory); then derive the Section 16 certification skeleton from it (same code, per the update plan).
5. **Lesson pages / prose** (handled by the LlamaIndex-removal sweep, not the package): install lines in two Section 1 lessons, "we opted for LlamaIndex" prose, framework-comparison lesson rewrite, `docs.llamaindex.ai` links (26 occurrences / 14 lesson files), LlamaParse pricing claims, `synonym_expand_policy`/`KnowledgeGraphRAGRetriever` recommendation in the query lesson, the Long-Context lesson's LlamaIndex machinery on the page (retriever, similarity postprocessor, sub-question engine, SQL aside, LlamaParse ingestion), video scripts (Section 4 overview, RAG Evals narration).
6. **Deliberate exceptions (keep):** `llama_index` rows in eval tables and the source picker are a **corpus source label** (docs the tutor indexes), not the library; the dataset filename `research_papers_llamaparse.zip` and its `/content/research_papers_llamaparse` path referenced by the new `Parsing_PDFs.ipynb` (rename only if we re-host the dataset anyway); Jerry Liu's RAG definition quote in the What-is-RAG video (attribution, not endorsement).
7. **`LlamaIndex_101.ipynb`, `LlamaParse.ipynb`, old `03-RAG_with_LlamaIndex.ipynb`, `Web_Search_API.ipynb`, the two superseded embedding notebooks + `Cohere_and_Open_Source` pre-merge copies**: archived per the repo-chores sweep once replacements land (old students hold links — archive, don't delete).

---

## 3. The package: identity and principles

### 3.1 Name and distribution

- **pip name:** `tai-aitutor` · **import name:** `tai_aitutor` (hyphens can't appear in imports; if the team prefers a shorter import, `aitutor` is the alternative — decide before v0.1 since renaming later breaks every notebook).
- **Repo:** `towardsai/tai-aitutor` (public — students read the source; that's the point).
- **Distribution:** GitHub-first during the port: notebooks install `pip install git+https://github.com/towardsai/tai-aitutor.git@v0.4.0` (tag-pinned; never `@main` in a student notebook). Publish to **PyPI at course launch** as `tai-aitutor 1.0.0`, then notebooks switch to `pip install tai-aitutor==1.0.0`. Action item week 1: check `pip index versions tai-aitutor` / pypi.org for name availability; optionally park a 0.0.1 placeholder.
- **Python:** ≥3.10 (Colab default satisfies this; matches modern typing syntax).
- **License:** MIT (students reuse it in certification projects) — confirm with the team.

### 3.2 Design principles (the contract)

1. **Teach-then-import.** The package only contains code some lesson writes out first. Each module docstring says which lesson builds it ("Built in: From Script to Pipeline"). Notebooks import only functions from *earlier* lessons.
2. **Function-first API.** Top-level verbs, not object graphs: `generate`, `embed`, `chunk`, `ingest`, `search`, `answer`, `rerank`, `route`, judges, metrics. The only stateful classes are the ones whose state is the lesson (`Chat`, `BM25Index`, `QADataset`).
3. **Provider-neutral by config, explicit by argument.** `configure(provider="gemini")` once per notebook (fed by the Colab dropdown); every function accepts `provider=`/`model=` overrides so comparison lessons (open-source vs GPT, Cohere vs local embeddings) stay natural.
4. **Same names as the ported notebooks.** The 20 updated notebooks already standardized `generate(prompt, system, model)`, `embed(texts, task)`, `chunk(text, chunk_size, chunk_overlap)`, `search(query, top_k, where)`, `answer(question, top_k)`, `n_tokens`, `make_qa_pairs`, `hit_rate`, `mrr`, `judge_*`, `extract`. The package promotes these signatures verbatim — migrating an updated notebook is mostly *deleting* its helper cells and adding one import.
5. **Mirror production constants.** Heading-aware chunks 800/100 (code blocks never split); BM25 k1=1.5 b=0.75 with the code-aware tokenizer; RRF k=60; dense top 15 / BM25 top 30 / keep 30; rerank `rerank-v4.0-fast` top 5 floor 0.10; `build_where_filter` `$eq`/`$in` shape; retrieval metrics = rank-of-gold. Where a lesson teaches a simpler default (512/128 chunks, top-k 5), the simpler value is the function default and the production value is named in the docstring.
6. **No hidden I/O, no globals besides config.** Nothing writes files or calls networks unless the function name says so. No `nest_asyncio`, no background async.
7. **Typed edges.** Pydantic models only where structure is the point: judge verdicts, extraction schemas, route decisions. Everything else is dataclasses and dicts.
8. **Every historical bug becomes a regression test** (Section 8): the ignored `top_k`, the reranker that was never measured, the judge whose ordering was discarded, the round-robin "hybrid", `eval()` on data, `additional_kwrgs`.

### 3.3 Deliberately NOT in the package

- The **first-build teaching code** each lesson writes inline (numpy cosine similarity in `02`, the BM25 formula walkthrough in `11`, the judge prompts shown in `06`, memory modes built by hand in `14`) — the package carries the *reusable* version, the notebook carries the *lesson* version.
- **Workflow/agent frameworks** — plain loops per the update plan; LangChain/LangGraph stay a Section 12/13 topic, not a dependency.
- **The Section 13 agent-eval teaching harness** — separate package by prior decision (see Section 11).
- **Vendor lesson code** (OpenAI images, RFT, Realtime audio mechanics, GraphRAG CLI, HF Inference, FastAPI lessons) — not RAG-pipeline code; only their small RAG tails use this package.
- **Qdrant/Deep Lake/pgvector backends** — Chroma only, with a one-line "other stores do this too" mention in lessons.
- **Prompt-template engine** — f-strings; shared prompt text lives as plain constants in `tai_aitutor.prompts`.

---

## 4. Package architecture

### 4.1 Repository layout

```
tai-aitutor/
├── pyproject.toml            # hatchling; extras below; py.typed
├── README.md                 # install, 10-line quickstart, lesson map, "from LlamaIndex" pointer
├── MIGRATION.md              # the Section 5 mapping table, kept current
├── CHANGELOG.md
├── src/tai_aitutor/
│   ├── __init__.py           # re-exports the teaching API (flat: from tai_aitutor import generate, embed, ...)
│   ├── config.py             # configure(), get_config(), setup_notebook(), require_keys()
│   ├── llm.py                # generate, generate_stream, generate_vision, extract, ask_batch
│   ├── embeddings.py         # embed, embed_cohere, embed_local
│   ├── tokens.py             # n_tokens, truncate, estimate_cost
│   ├── documents.py          # Document, load_csv, load_jsonl, load_hf_dataset, load_directory, load_files, load_wikipedia
│   ├── chunking.py           # chunk, chunk_sentences, heading_aware_markdown_chunks, sentence_window_chunks
│   ├── parsing.py            # parse_pdf, parse_pdf_with_llm, slice_pdf, first_pages_text
│   ├── extractors.py         # extract_keywords, extract_summary, extract_questions, situate_chunk
│   ├── vectorstore.py        # get_collection, reset_collection, ingest, get_all_chunks, build_where_filter
│   ├── retrieval.py          # search, BM25Index, rrf_fuse, hybrid_search, rerank, judge_rerank,
│   │                         # hyde_search, decompose_question, subquestion_answer, multi_step_answer,
│   │                         # expand_window, pack_context
│   ├── synthesis.py          # build_rag_prompt, answer, answer_with_sources, answer_stream, Answer
│   ├── router.py             # route, RouteDecision
│   ├── chat.py               # Chat, ToolLoop, memory strategies (full/window/summary)
│   ├── tools.py              # tool(), make_retrieval_tool, search_web (tavily)
│   ├── evals.py              # QADataset, make_qa_pairs, hit_rate, mrr, evaluate_retrieval,
│   │                         # judge_faithfulness/relevancy/correctness (+ Verdict models), run_judges
│   ├── finetune.py           # make_training_pairs, train_embedder, evaluate_embedder
│   ├── datasets.py           # course data: mini_articles(), ai_tutor_knowledge(), prebuilt_chroma(), qa_dataset()
│   ├── prompts.py            # shared prompt constants (RAG answer prompt, judge prompts, situate prompt, router prompt)
│   └── display.py            # show_chunks, show_answer, show_eval_table (pretty notebook output)
├── tests/                    # unit + regression (Section 8)
├── examples/                 # one smoke script per module
└── .github/workflows/        # ci.yml (tests, lint, type-check), notebooks.yml (weekly nb run), release.yml
```

### 4.2 Public API, module by module

Signatures are the promotion of what the ported notebooks already define; defaults shown are the course defaults.

**`config`**

```python
configure(provider="gemini",              # "gemini" | "openai" | "anthropic" | "together" | "deepseek" | "ollama" | "perplexity"
          chat_model=None,                # default per provider: gemini-2.5-flash / gpt-5-mini / claude-sonnet-latest...
          embed_provider=None,            # "gemini" (default) | "openai" | "cohere" | "local" — Anthropic has no embeddings
          embed_model=None,               # gemini-embedding-001 / text-embedding-3-small / embed-v4.0 / BAAI-bge...
          base_url=None, api_key=None)    # the Together/Ollama/DeepSeek escape hatch
get_config() -> Config                    # inspectable, printable
setup_notebook(required_keys=(...)) -> bool  # IN_COLAB detection; Colab Secrets vs .env loading (Decision 3 cell calls this)
require_keys("GOOGLE_API_KEY", ...)       # fail fast with a friendly message
```

Model defaults live in ONE table in `config.py` (name, context window, prices for `estimate_cost`) so model swaps are single-line PRs.

**`llm`** — replaces `OpenAI`, `GoogleGenAI`, `Perplexity`, `TogetherLLM`, `ChatMessage`, structured predict, and every `Settings.llm`:

```python
generate(prompt, system=None, model=None, provider=None,
         temperature=None, max_tokens=None, reasoning_effort=None) -> str
generate_stream(...) -> Iterator[str]
generate_vision(prompt, image_bytes, mime_type="image/jpeg", model=None) -> str
extract(prompt, schema: type[BaseModel], system=None, model=None) -> BaseModel   # native structured outputs on all 3 providers
chat_completion(messages: list[dict], tools=None, **kw) -> Completion            # the raw layer Chat/ToolLoop use; returns text, tool_calls, usage
ask_batch(prompts, concurrency=8, desc=None) -> list[str]                        # replaces BatchEvalRunner-style fan-out
```

**`embeddings`** — replaces `OpenAIEmbedding`, `CohereEmbedding`, `HuggingFaceEmbedding`, `resolve_embed_model`, `Settings.embed_model`:

```python
embed(texts, task="document", provider=None, model=None, batch_size=100) -> list[list[float]]   # task: "document" | "query"
embed_cohere(texts, task="document", model="embed-v4.0") -> list[list[float]]                    # input_type asymmetry taught + enforced
embed_local(texts, model_name="BAAI/bge-small-en-v1.5", task="document") -> list[list[float]]    # sentence-transformers; e5 "query: " prefix handled
```

Retry with backoff and rate-limit friendliness built in (the production concern the notebooks skip).

**`tokens`**

```python
n_tokens(text, model=None) -> int          # tiktoken (o200k_base default)
truncate(text, max_tokens, model=None) -> str
estimate_cost(input_tokens, output_tokens, model=None) -> float   # priced from the config table, dated
```

**`documents`**

```python
@dataclass Document: text: str; metadata: dict = {}; id: str | None = None
load_csv(path_or_url, text_col, meta_cols=(...)) -> list[Document]      # mini-articles loader; json.loads for embeddings — never eval()
load_jsonl(path_or_url, text_key="content") -> list[Document]
load_hf_dataset(repo_id, filename=None, split=None) -> list[Document]   # ai_tutor_knowledge etc.
load_directory(path, exts=(".txt", ".md", ".pdf")) -> list[Document]    # replaces SimpleDirectoryReader
load_files(paths) -> list[Document]
load_wikipedia(titles) -> list[Document]                                 # replaces WikipediaReader
```

**`chunking`** — replaces both `TokenTextSplitter` paths, `SentenceSplitter`, `SimpleNodeParser`, `SentenceWindowNodeParser`:

```python
chunk(text, chunk_size=512, chunk_overlap=128, separator=" ") -> list[str]
chunk_document(doc, chunk_size=512, chunk_overlap=128) -> list[Chunk]        # carries metadata through, stable ids
chunk_sentences(text, chunk_size=512, chunk_overlap=128) -> list[str]
heading_aware_markdown_chunks(markdown, chunk_size=800, chunk_overlap=100) -> list[str]   # production chunker, code blocks never split
sentence_window_chunks(text, window_size=3) -> list[Chunk]                   # neighbors stored in metadata["window"]
```

**`extractors`** — replaces `KeywordExtractor`, `SummaryExtractor`, `QuestionsAnsweredExtractor`; adds production's contextual retrieval:

```python
extract_keywords(chunks, n=10, model=None) -> list[Chunk]      # writes metadata["keywords"]
extract_summary(chunks, model=None) -> list[Chunk]             # metadata["summary"]
extract_questions(chunks, n=3, model=None) -> list[Chunk]      # metadata["questions_answered"]
situate_chunk(chunk_text, document_text, model=None) -> str    # SituatedContext pattern from add_context_to_nodes.py
```

**`vectorstore`** — replaces `ChromaVectorStore`, `QdrantVectorStore`, `StorageContext`, `IngestionPipeline`, `VectorStoreIndex.from_documents/.from_vector_store/.insert`:

```python
get_collection(name, path=None) -> chromadb.Collection      # path=None → ephemeral; path → PersistentClient
reset_collection(name, path=None) -> chromadb.Collection
ingest(docs, collection, chunker=chunk, chunk_size=512, chunk_overlap=128,
       enrich=(), embed_fn=embed, batch_size=64, show_progress=True) -> IngestStats
get_all_chunks(collection) -> list[Chunk]                   # real enumeration (collection.get) — kills the top_k=100000000 hack
build_where_filter(sources) -> dict | None                  # {"source": {"$in": [...]}} / "$eq" — production shape
```

**`retrieval`** — replaces retrievers, query engines, transforms, postprocessors:

```python
search(query, collection=None, top_k=5, where=None, embed_fn=None) -> list[ScoredChunk]
class BM25Index:                                            # k1=1.5, b=0.75, code-aware tokenizer (camelCase, dotted.paths, "c")
    build(chunks); search(query, top_k=30) -> list[ScoredChunk]
    save(path); load(path)                                  # versioned gzipped JSON — never pickle
rrf_fuse(*ranked_lists, k=60, keep=30) -> list[ScoredChunk]  # score += 1/(k + rank)
hybrid_search(query, collection, bm25=None, dense_top_k=15, bm25_top_k=30, keep=30, where=None) -> list[ScoredChunk]
rerank(query, hits, model="rerank-v4.0-fast", top_n=5, floor=0.10) -> list[ScoredChunk]   # cohere ClientV2.rerank
judge_rerank(query, hits, top_n=3, model=None) -> list[ScoredChunk]   # LLM judge; judge's scores AND order kept
hyde_search(query, collection=None, top_k=5, include_original=True) -> list[ScoredChunk]
decompose_question(question, n_max=4, model=None) -> list[str]        # typed sub-question generation
subquestion_answer(question, collection=None, top_k=5) -> Answer      # decompose → search each → synthesize
multi_step_answer(question, collection=None, max_steps=3) -> Answer   # iterative refine loop
expand_window(hits) -> list[ScoredChunk]                              # metadata["window"] replacement
pack_context(hits, max_tokens=100_000, model=None) -> list[ScoredChunk]  # production token budget
```

**`synthesis`** — replaces `.as_query_engine().query()`, `get_response_synthesizer`, `response.source_nodes`:

```python
@dataclass Answer: text: str; sources: list[ScoredChunk]; usage: Usage
build_rag_prompt(question, hits, system_rules=prompts.RAG_SYSTEM) -> str   # the visible f-string with cited sources
answer(question, collection=None, top_k=5, where=None, retriever=None, model=None) -> Answer
answer_with_sources(question, ...) -> Answer          # citation-grounded variant (title/url per chunk in prompt)
answer_stream(question, ...) -> Iterator[str]         # + .sources after exhaustion
```

**`router`** — replaces `RouterQueryEngine` + selectors:

```python
class RouteDecision(BaseModel): route: str; reason: str
route(question, routes: dict[str, str], model=None) -> RouteDecision   # routes = {"name": "description"}; dispatch is the student's if/else
```

**`chat`** — replaces chat engines, agents, `Context`, memory buffers:

```python
class Chat:
    __init__(system=None, tools=(), history="full",      # "full" | "window" | "summary"
             window_turns=8, summarize_after_tokens=8000, model=None, provider=None)
    ask(user_msg) -> str        # runs the tool loop if tools attached
    ask_stream(user_msg) -> Iterator[Event]   # Event: text | tool_call | tool_result
    messages: list[dict]        # inspectable — memory is just the list you resend (the lesson's point)
class ToolLoop:                 # the bare agent loop (used by Chat; teachable standalone)
    run(user_msg, max_iters=6) -> str
```

**`tools`**

```python
tool(fn=None, *, name=None, description=None) -> Tool     # decorator: python fn + auto JSON schema
make_retrieval_tool(collection, top_k=5, name="search_course_knowledge") -> Tool
search_web(query, max_results=5) -> list[dict]            # tavily-python; the Tavily lesson's tool, packaged
```

**`evals`**

```python
class QADataset:  queries: dict[id, str]; corpus: dict[id, str]; relevant_docs: dict[id, list[id]]
    save(path); load(path)                       # SAME JSON schema as EmbeddingQAFinetuneDataset → old artifacts load unchanged
make_qa_pairs(collection, n_chunks=25, questions_per_chunk=1, model=None) -> QADataset
hit_rate(qa, search_fn=search, top_k=5) -> float
mrr(qa, search_fn=search, top_k=5) -> float
evaluate_retrieval(qa, search_fn=search, top_k=5) -> RetrievalReport      # both metrics + per-query table
class FaithfulnessVerdict(BaseModel): faithful: bool; reasoning: str
class RelevancyVerdict(BaseModel):    relevant: bool; reasoning: str
class CorrectnessVerdict(BaseModel):  score: float; reasoning: str        # 1–5, threshold 4.0
judge_faithfulness(answer, context, model=None) -> FaithfulnessVerdict
judge_relevancy(question, answer, context, model=None) -> RelevancyVerdict
judge_correctness(question, answer, reference, model=None) -> CorrectnessVerdict
run_judges(rows, judges=(...), concurrency=8) -> JudgeReport              # replaces BatchEvalRunner; no asyncio
```

**`finetune`**

```python
make_training_pairs(chunks, questions_per_chunk=2, model=None) -> QADataset
train_embedder(train: QADataset, val: QADataset, base_model="BAAI/bge-small-en-v1.5",
               epochs=2, batch_size=32, out_dir="ft-embedder") -> str     # sentence-transformers + MNRL
evaluate_embedder(model_path_or_name, qa: QADataset, top_k=5) -> RetrievalReport
```

**`datasets`** — one home for course data (Decision 6: org-owned, versioned):

```python
mini_articles() -> list[Document]                 # the refreshed mini-articles set (org HF repo)
ai_tutor_knowledge() -> list[Document]
prebuilt_chroma(embed_provider="gemini") -> path  # re-embedded org-hosted store; downloads + unzips
qa_dataset(name="rag_eval_50") -> QADataset
```

URLs point ONLY at the Towards AI org account. This module is how we finally kill the `AlaFalaki/*` and `jaiganesan/*` coupling in one place.

**`display`** — small quality-of-life for notebooks: `show_chunks(hits)`, `show_answer(ans)` (text + cited sources), `show_eval_table(reports)` (the four-row ablation table in the hybrid lesson).

### 4.3 Dependencies and extras

Core install stays light; heavy things are extras:

| Extra | Adds | Used by |
|---|---|---|
| (core) | `pydantic>=2`, `tiktoken`, `httpx`, `tqdm` | everything |
| `[gemini]` | `google-genai>=1.35` | default provider |
| `[openai]` | `openai>=2,<3` (settles the 1.x/2.x pin conflict — package requires 2.x) | provider |
| `[anthropic]` | `anthropic` (current) | provider |
| `[rag]` | `chromadb>=1.0.21` | vector store lessons |
| `[rerank]` | `cohere>=5.18` | reranking + Cohere embeddings |
| `[local]` | `sentence-transformers` | local embeddings, finetune eval |
| `[finetune]` | `sentence-transformers`, `datasets`, `accelerate` | notebook 08 |
| `[parse]` | `pypdf` | parsing lessons |
| `[web]` | `tavily-python`, `wikipedia` | web/tool lessons |
| `[data]` | `huggingface-hub`, `pandas` | datasets module |
| `[all]` | everything above | CI, maintainers |

Install cells in notebooks: `pip install "tai-aitutor[gemini,openai,anthropic]==X.Y.Z"` (profile A) or `...[gemini,openai,anthropic,rag,rerank]` (profile B). Providers not installed raise a clear "pip install tai-aitutor[x]" message at call time, so a student who picks only Gemini never downloads torch.

### 4.4 Provider support matrix (v1.0)

| Capability | Gemini (default) | OpenAI | Anthropic | Together/DeepSeek/Ollama/Perplexity (base_url) |
|---|---|---|---|---|
| `generate` / `generate_stream` | ✔ | ✔ | ✔ | ✔ (OpenAI-compatible) |
| `extract` (structured) | ✔ `response_schema` | ✔ structured outputs | ✔ tool-schema | best-effort JSON + validate |
| `generate_vision` | ✔ | ✔ | ✔ | model-dependent |
| `embed` | ✔ `gemini-embedding-001` | ✔ `text-embedding-3-small` | — (no embeddings API; falls back per Decision 2 with a clear message) | via Cohere/local instead |
| reasoning controls | thinking budget | `reasoning_effort` | thinking budget | pass-through |

---

## 5. Migration map: LlamaIndex → `tai_aitutor` (the porting cheat-sheet)

This is the table editors work from cell by cell. (Full inventory context in Section 2; this is the "what do I type instead" view.)

| You see (LlamaIndex) | You write (`tai_aitutor`) |
|---|---|
| `Settings.llm = OpenAI(model="gpt-5-mini", additional_kwargs={'reasoning_effort':'minimal'})` | `configure(provider=PROVIDER)` — or per-call `generate(..., provider="openai", model="gpt-5-mini", reasoning_effort="minimal")` |
| `Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")` | `configure(embed_provider="openai", embed_model="text-embedding-3-small")` |
| `llm.complete(prompt)` / `llm.chat(messages)` | `generate(prompt, system=...)` / `chat_completion(messages)` |
| `llm = GoogleGenAI(model="gemini-2.5-flash", generation_config=cfg)` | `generate(..., provider="gemini", temperature=..., max_tokens=...)` |
| `Perplexity(...)` / `TogetherLLM(...)` | `configure(provider="perplexity" / "together")` — OpenAI-compatible `base_url` under the hood |
| `Document(text=row[1], metadata={...})` | `Document(text=row[1], metadata={...})` (ours) or plain dict |
| `SimpleDirectoryReader("dir").load_data()` | `load_directory("dir")` |
| `WikipediaReader().load_data(pages=[...])` | `load_wikipedia([...])` |
| `LlamaParse(...)` + `file_extractor` | `parse_pdf(path)` (pypdf) or `parse_pdf_with_llm(path, schema=...)` |
| `TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)` | `chunk(text, 512, 128)` / `chunk_document(doc, 512, 128)` |
| `SentenceSplitter(chunk_size=768, chunk_overlap=64)` | `chunk_sentences(text, 768, 64)` |
| `SentenceWindowNodeParser.from_defaults(window_size=3, ...)` | `sentence_window_chunks(text, window_size=3)` |
| `splitter.get_nodes_from_documents(docs)` | `[c for d in docs for c in chunk_document(d)]` |
| `KeywordExtractor(keywords=10, llm=...)` etc. in `transformations=[...]` | `ingest(..., enrich=[extract_keywords, extract_summary, extract_questions])` |
| `IngestionPipeline(transformations=[splitter, embed_model], vector_store=vs).run(documents=docs)` | `ingest(docs, collection, chunk_size=..., chunk_overlap=...)` |
| `chromadb.PersistentClient(...)` + `ChromaVectorStore(chroma_collection=col)` + `StorageContext.from_defaults(...)` | `col = get_collection("ai_tutor_knowledge", path="./db")` — chromadb only, no wrappers |
| `VectorStoreIndex.from_documents(docs, transformations=[...])` | `ingest(docs, col)` |
| `VectorStoreIndex.from_vector_store(vector_store)` | nothing — the collection is already the index; go straight to `search`/`answer` |
| `index.insert(doc)` | `ingest([doc], col)` |
| `index.storage_context.persist("./storage")` + `load_index_from_storage(...)` | persistent Chroma path (no separate step) |
| `index.as_retriever(similarity_top_k=k).retrieve(q)` | `search(q, col, top_k=k)` |
| `index.as_query_engine(similarity_top_k=k).query(q)` | `answer(q, col, top_k=k)` → `Answer(text, sources)` |
| `response.response` / `response.source_nodes` / `node.score` | `ans.text` / `ans.sources` / `hit.score` |
| `query_engine.query(q).print_response_stream()` | `for tok in answer_stream(q, col): print(tok, end="")` |
| `response_mode="refine"` demo | dropped by plan (broken demo; do not rebuild) |
| `SimpleKeywordTableIndex(nodes)` + `KeywordTableSimpleRetriever` | `bm25 = BM25Index(); bm25.build(get_all_chunks(col)); bm25.search(q, 30)` |
| custom `BaseRetriever` merge (round-robin) | `hybrid_search(q, col, bm25)` = dense ∪ BM25 → `rrf_fuse(k=60)` — cap enforced |
| `RetrieverQueryEngine(retriever, response_synthesizer=get_response_synthesizer(llm))` | `answer(q, retriever=lambda q: hybrid_search(q, col, bm25))` |
| `CohereRerank(top_n=2, model='rerank-english-v3.0')` in `node_postprocessors` | `hits = rerank(q, hits, model="rerank-v4.0-fast", top_n=5, floor=0.10)` — explicit stage, measurable |
| `RankGPTRerank(top_n=3, llm=...)` / custom `BaseNodePostprocessor` | `judge_rerank(q, hits, top_n=3)` — judge order + scores survive |
| `MetadataReplacementPostProcessor(target_metadata_key="window")` | `expand_window(hits)` |
| `MetadataFilters(filters=[MetadataFilter(key="source", operator=FilterOperator.EQ, value="tai_blog")], condition=AND)` | `search(q, col, where={"source": "tai_blog"})`; multi-source: `where=build_where_filter(["tai_blog","hf"])`; text match: `where_document={"$contains": ...}` |
| `HyDEQueryTransform(include_original=True)` + `TransformQueryEngine` | `hyde_search(q, col)` |
| `LLMQuestionGenerator` + `QueryEngineTool` + `SubQuestionQueryEngine.from_defaults(...)` | `decompose_question(q)` + loop, or `subquestion_answer(q, col)` |
| `StepDecomposeQueryTransform` + `MultiStepQueryEngine` | `multi_step_answer(q, col, max_steps=3)` |
| `RouterQueryEngine(selector=LLMSingleSelector/PydanticSingleSelector, query_engine_tools=[...])` | `d = route(q, routes); if d.route == "knowledge": ...` |
| `FunctionAgent(tools=[...], llm=...)` / `ReActAgent(...)` + `Context(agent)` + `AgentStream`/`ToolCallResult` | `Chat(system=..., tools=[make_retrieval_tool(col), search_web_tool])`; events via `ask_stream()` |
| `index.as_chat_engine(chat_mode=..., memory=...)` / `.chat()` / `.stream_chat()` | `Chat(history="window"|"summary").ask()` / `.ask_stream()` |
| `Workflow` / `@step` / `StartEvent` / `StopEvent` (Perplexity project) | plain functions + a loop (already ported) |
| `TavilyToolSpec(api_key=...)` | `search_web(q)` / `tools.search_web` |
| `GoogleSearchToolSpec` + `LoadAndSearchToolSpec` | archived with `Web_Search_API.ipynb` |
| `generate_question_context_pairs(nodes, llm, num_questions_per_chunk)` | `make_qa_pairs(col, n_chunks, questions_per_chunk)` |
| `EmbeddingQAFinetuneDataset.from_json(p)` / `.save_json(p)` / `.queries/.corpus/.relevant_docs` | `QADataset.load(p)` / `.save(p)` — same JSON keys, old files load |
| `RetrieverEvaluator.from_metric_names(["mrr","hit_rate"], retriever)` + `aevaluate_dataset` | `evaluate_retrieval(qa, search_fn, top_k)` (or `hit_rate(...)`, `mrr(...)`) |
| `FaithfulnessEvaluator(llm).evaluate_response(...)` | `judge_faithfulness(ans.text, context)` |
| `RelevancyEvaluator(llm)` / `CorrectnessEvaluator(llm)` | `judge_relevancy(...)` / `judge_correctness(...)` |
| `BatchEvalRunner({...}, workers=8).aevaluate_queries(...)` + `nest_asyncio.apply()` | `run_judges(rows, judges=[...], concurrency=8)` — and delete the `nest_asyncio` cell |
| `resolve_embed_model("local:BAAI/bge-small-en-v1.5")` | `embed_local(texts, "BAAI/bge-small-en-v1.5")` |
| `CohereEmbedding(model_name=..., input_type="search_document")` | `embed_cohere(texts, task="document")` |
| `EmbeddingAdapterFinetuneEngine(ds, model_output_path=..., epochs=...)` + `AdapterEmbeddingModel` | `train_embedder(train_ds, val_ds, base_model="BAAI/bge-small-en-v1.5")` + `evaluate_embedder(...)` |
| `PromptTemplate("...")` | f-string or `prompts.*` constant |
| `QueryBundle(query_str)` | the string itself |
| `nest_asyncio.apply()` | delete |

**Four bugs the new API kills by design (and pins with tests):**
1. *Reranker never measured* (`10`): postprocessors could be silently ignored by `.as_retriever()`. Now reranking is an explicit call between search and eval — you can't pass it to something that drops it. Regenerate the published eval table from a real run.
2. *Judge ordering thrown away* (`17`): `_postprocess_nodes` re-looped the original list. `judge_rerank` returns the judge's order with judge scores attached.
3. *"Hybrid" that was round-robin with an ignored cap* (`11`): `rrf_fuse` implements real RRF, `keep` is applied, and `get_all_chunks` replaces the `similarity_top_k=100000000` dump.
4. *`eval()` on data* (`02` embeddings CSV): `load_csv` parses embeddings with `json.loads` only.

---

## 6. Notebook-by-notebook migration matrix

Order follows the course (port order per the update plan: Sections 3–7 first, then 8–10 stragglers). "Imports" = main `tai_aitutor` surface after migration. ✅ = already off LlamaIndex (helper cells → package imports is the remaining step). 🔨 = port still to do (LlamaIndex present today). 📦 = archive.

**Wave 1 — swap helper cells for imports (notebooks already ported):**

| Notebook | Status | Package imports after migration |
|---|---|---|
| `RAG_101.ipynb` | ✅ | `config`, `generate` |
| `Prompting_101.ipynb`, `Evaluating_and_Iterating_Prompts.ipynb`, `Intro_to_Large_Language_Models.ipynb` | ✅ | `config`, `generate` (+ `estimate_cost` in Intro) |
| `02-Basic_RAG.ipynb` | ✅ | `config`, `generate`, `embed` — chunking/cosine stay inline (it's the lesson) |
| `03-From_Script_to_Pipeline.ipynb` | ✅ | builds `chunk`/`heading_aware_markdown_chunks`/`search`/`answer` inline (it's the lesson), ends by showing the same API in the package ("what you wrote is what you'll import from now on") |
| `04-RAG_with_VectorStore.ipynb` | ✅ | `get_collection`, `ingest`, `search`, `answer` (chroma taught raw first) |
| `05-Improve_Prompts_+_Add_Source.ipynb` | ✅ | `ingest`, `search`, `build_rag_prompt`, `answer_with_sources` |
| `06-Evaluate_RAG.ipynb` | ✅ | builds judges/metrics inline (the lesson), then `evals.*` for reuse; `QADataset` |
| `15-Use_OpenSource_Models.ipynb` | ✅ | `configure(provider="together")`, `generate`, `evals.*` |
| `Selecting_Embedding_Models.ipynb` | ✅ | `embed`, `embed_cohere`, `embed_local`, `evals.evaluate_retrieval` |
| `Larger_Context_Larger_N.ipynb` | ✅ | `ingest`, `search`, `answer`, `evals.*` |
| `Multimodal_LLMs.ipynb` | ✅ | `generate_vision`, `embed`, `search` (image-description RAG) |
| `Crawl_a_Website.ipynb`, `Firecrawl_Scraping.ipynb` | ✅ | scraping SDKs stay direct; tail = `ingest` + `answer` |
| `Perplexity_Web_Api.ipynb` | ✅ | `configure(provider="perplexity")` or raw client; `extract` |
| `Web_Search_API_Tavily.ipynb` | ✅ | `generate`, `search_web`, `ToolLoop`/`Chat` for the tool-calling half |
| `Structured(JSON)_PDF_Data_Extraction.ipynb`, `Applied_Structured_Outputs.ipynb` | ✅ | `extract`, `parse_pdf`/`first_pages_text` |
| `Parsing_PDFs.ipynb` | ✅ | `parse_pdf`, `slice_pdf`, `extract` (only remaining "llama" match is the dataset filename `research_papers_llamaparse.zip`) |

**Wave 2 — Section 7 advanced RAG ports (🔨, the core remaining work):**

| Notebook | Port spec (from update plan + scan) | Package pieces |
|---|---|---|
| `10-Adding_Reranking.ipynb` 🔨 | Cohere rerank direct, production constants, re-run the eval table for real | `search`, `rerank`, `evals.evaluate_retrieval` |
| `11-Adding_Hybrid_Search.ipynb` 🔨 | real BM25 (taught inline) + RRF + code-aware tokenizer; 4-row ablation (dense / bm25 / fused / fused+rerank) | `BM25Index`, `rrf_fuse`, `hybrid_search`, `rerank`, `evals`, `get_all_chunks` |
| `12-Improve_Query.ipynb` 🔨 | HyDE, sub-questions, multi-step as visible few-liners; drop `synonym_expand_policy` prose | `hyde_search`, `decompose_question`, `subquestion_answer`, `multi_step_answer` |
| `17-Using_LLMs_to_rank_chunks_as_the_Judge.ipynb` 🔨 (mid-port) | finish: drop `BaseNodePostprocessor` subclass + llama retriever; typed judge scores kept | `search`, `judge_rerank` |
| `Metadata_Filtering.ipynb` 🔨 | Qdrant → Chroma `where`; full dataset (no silent 101–500 drop); production filter shape | `ingest(enrich=[extract_keywords])`, `search(where=...)`, `build_where_filter` |
| `Advanced_Retriever.ipynb` 🔨 | sentence-window/small-to-big as two visible functions | `sentence_window_chunks`, `search`, `expand_window` |
| `07-RAG_Improve_Chunking.ipynb` 🔨→📦 | salvage the chunk-size × top-k sweep into the new chunking lesson section; then archive | `chunk`, `ingest`, `evals.evaluate_retrieval` |

**Wave 3 — Sections 8–9 stragglers (🔨):**

| Notebook | Port spec | Package pieces |
|---|---|---|
| `08-Finetune_Embedding.ipynb` 🔨 | method change: sentence-transformers MNRL on `bge-small-en-v1.5`; typed pair generation; before/after with course metrics | `finetune.*`, `evals.*`, `QADataset` |
| `13-Adding_Router.ipynb` 🔨 | typed route + if/else; wiki data via `load_wikipedia`; extractors demo via `enrich=` | `route`, `load_wikipedia`, `ingest`, `answer` |
| `14-Adding_Chat.ipynb` 🔨 | memory built by hand (full/window/summary) then `Chat` with retrieval tool; fix `additional_kwrgs` everywhere | `Chat`, `make_retrieval_tool` |
| `GPT_4o_mini_Fine_Tuning.ipynb` 🔨 | RAG scaffolding → package; OpenAI FT stays native; re-run with current teacher/student | `search`, `answer`, `ask_batch` |
| `Audio_and_Realtime.ipynb` 🔨 | rebase on current audio models; RAG section → package; keep tool-calling flow | `search`, `answer`, `tools` |
| `Long_Context_Caching_vs_RAG.ipynb` 🔨 | RAG side → package; PDF via parsing lesson approach; drop SQL aside | `parse_pdf`, `ingest`, `answer`, `n_tokens` |
| `Knowledge_Base_for_RAG.ipynb` 🔨→S13 | salvage corpus-loading intro (port it) for the Section 13 KB lesson | `datasets.ai_tutor_knowledge`, `prebuilt_chroma` |
| `More_Api_And_Tools.ipynb` 🔨 | remove the LlamaParse segment (point to Parsing lesson); rest untouched | — |
| `GraphRAG_Implementation.ipynb` | no LlamaIndex in code; delete the stale "not compatible with LlamaIndex" prose | — |

**Wave 4 — apps and archives:**

| Item | Action |
|---|---|
| `ai-tutor-gradio-lesson` repo | rebuild on `Chat` + `make_retrieval_tool` + `datasets.prebuilt_chroma()` (re-embedded, org-hosted); fix requirements (UTF-8, short, no llama pins); then fork → Section 16 certification skeleton |
| `03-RAG_with_LlamaIndex.ipynb` (old), `LlamaIndex_101.ipynb`, `LlamaParse.ipynb`, `Web_Search_API.ipynb`, `Cohere_Better_Embedding_Model.ipynb`, `Open_source_BetterEmbedding_Model.ipynb`, `Cohere_and_Open_Source_Embedding_Model.ipynb` (post-merge) | 📦 archive folder (`notebooks/archive/`), pointer note at top; old students keep working links |
| root `requirements.txt` | delete (dead freeze) |
| Exercise notebooks (`notebook_exercise` variants, where they exist) | regenerate from migrated notebooks so gaps target package-taught concepts, not LlamaIndex calls |

---

## 7. Build phases and sequencing

Sized for one primary engineer + one reviewer; notebook migration can parallelize with lesson editors once each phase tags. Sequencing matches the update plan's "design the shared pieces once, then port in course order."

**Phase 0 — Scaffold (2–3 days) → tag `v0.1.0`**
Repo, `pyproject`, CI (pytest + ruff + mypy), name availability check on PyPI, `config` + `setup_notebook()` (the Decision 3 environment cell logic, packaged), `llm.generate/generate_stream/extract` for the three providers + base_url path, `embeddings.embed` (Gemini + OpenAI), `tokens`. Acceptance: the three-provider setup cell in a fresh Colab AND a local venv reduces to `%pip install ...` + `PROVIDER` dropdown + `configure(PROVIDER)`.

**Phase 1 — The pipeline core (1 week) → `v0.2.0`**
`documents`, `chunking` (incl. heading-aware + sentence-window), `vectorstore` (`get_collection`, `ingest`, `get_all_chunks`, `build_where_filter`), `retrieval.search`, `synthesis` (`Answer`, `build_rag_prompt`, `answer`, `answer_with_sources`, `answer_stream`), `display`. Migrate Wave 1 notebooks (delete duplicated helper cells → imports). Unblocks Sections 3–4 lesson editing.

**Phase 2 — Evals (3–4 days) → `v0.3.0`**
`evals` complete (QADataset with legacy-JSON compatibility, `make_qa_pairs`, metrics, judges, `run_judges`), `extractors`. Migrate `06`, `Larger_Context`, `Selecting_Embedding_Models` to package evals; regenerate any published eval numbers these lessons show. Unblocks every later notebook that measures anything.

**Phase 3 — Advanced RAG (1–1.5 weeks) → `v0.4.0`**
`BM25Index`, `rrf_fuse`, `hybrid_search`, `rerank`, `judge_rerank`, `hyde_search`, `decompose_question`, `subquestion_answer`, `multi_step_answer`, `expand_window`, `pack_context`. Port Wave 2 (10, 11, 12, 17, Metadata_Filtering, Advanced_Retriever; salvage 07). Regenerate the reranking table and the four-row hybrid ablation for real. This is the highest-value phase: Section 7 becomes "production stack, cell for cell."

**Phase 4 — Router, chat, tools, finetune (1 week) → `v0.5.0`**
`router`, `chat` (`Chat`, `ToolLoop`, memory modes), `tools`, `finetune`. Port Wave 3 (13, 14, 08, GPT-4o-mini scaffolding, Audio RAG tail, Long-Context RAG side, KB salvage). Tavily notebook adopts `tools.search_web`/`ToolLoop`.

**Phase 5 — Data + apps + launch (1 week, overlaps 3–4) → `v1.0.0` on PyPI**
`datasets` module against the new Towards AI org HF repos (mini-articles refresh, `ai_tutor_knowledge`, **re-embedded** prebuilt Chroma store — required anyway because Gemini becomes the default embedder), Gradio app rebuild + certification skeleton fork, `MIGRATION.md` + README + API docs (mkdocs-material or pdoc), publish to PyPI, switch all install cells from git+tag to the PyPI pin.

Total: **~5–6 calendar weeks** for the package + notebook migrations, parallel with the lesson-text sweeps the update plan already schedules. The package does not extend the update's critical path: Phases 0–1 land inside week 1, and lesson editing for Sections 3–4 can start against `v0.2.0`.

---

## 8. Testing and CI

**Unit tests (no network).** Chunking invariants (overlap, token bounds, code blocks never split, window neighbors correct); BM25 scores against a tiny hand-computed corpus; RRF math (`1/(k+rank)`, cap applied); `build_where_filter` shapes; QADataset JSON round-trip **including loading a real legacy `rag_eval_dataset.json` fixture**; hit-rate/MRR on synthetic rankings with known answers; prompt builders; provider dispatch (mocked SDK clients for all branches, including the Anthropic-embeddings error message and base_url providers).

**Regression tests for the historical bugs (named in the code):** `test_rerank_changes_metrics` (a rerank stage must be able to change eval output — guards bug 1), `test_judge_rerank_preserves_judge_order` (bug 2), `test_hybrid_respects_keep` + `test_get_all_chunks_no_giant_topk` (bug 3), `test_load_csv_never_evals` (bug 4), `test_generate_rejects_unknown_kwargs` (the `additional_kwrgs` class of typo → TypeError).

**Integration smoke (network, nightly + pre-release).** One script per provider path (Gemini free tier, OpenAI, Anthropic, Ollama in CI container): generate → embed → ingest 20 chunks → search → hybrid → rerank (Cohere trial key) → answer → one judge call. Budget-capped; skips cleanly when a secret is absent.

**Notebook CI (the real product).** Weekly + on release tag: run migrated notebooks top-to-bottom with papermill in two matrices — a Colab-sim container (fresh pip, `IN_COLAB` forced) and a plain venv (Decision 3's two environments). Fast tier per PR: the 6 cheapest notebooks. Full tier weekly. Any notebook that needs paid keys runs with spend caps and a cost report in the job summary.

**Golden numbers policy.** Published eval tables (hybrid ablation, reranking table, chunk-size sweep) are regenerated by scripts in `examples/`, with outputs and "as of" dates committed next to the lesson, so a re-run is one command when models change.

---

## 9. Versioning, releases, maintenance

- **SemVer with a teaching twist.** During the port: minor = new module, patch = fixes. After 1.0: breaking API changes require a major bump AND a notebook sweep ticket — the package version and the "course release" are pinned together. Notebooks always pin exact versions (`==`), never ranges.
- **One pin cell per notebook.** The install cell is generated from a template (profile A/B + extras), so the future "bump every notebook" is a scripted find-replace — the maintenance win this whole plan exists for.
- **Deprecation policy.** Nothing is removed within a course cycle; deprecated functions warn with the replacement name for one minor version.
- **Ownership.** Repo lives in the Towards AI org with CODEOWNERS = course team; issues templated as "notebook breakage" (notebook, cell, provider, traceback). Provider-SDK breakage is caught by the nightly smoke, fixed in the package, patch-released — zero notebook edits.
- **Docs.** README quickstart (10 lines: configure → ingest → answer), auto-generated API reference, `MIGRATION.md` (Section 5 table), and a "which lesson builds this" index — the package doubles as the course's code index.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Package becomes a framework (pedagogy inverted) | Teach-then-import rule enforced in review; module size budget; every public symbol must name its teaching lesson or it doesn't merge |
| Provider SDK churn (openai 2.x, google-genai, anthropic) breaks notebooks mid-cohort | All SDK contact isolated in `llm.py`/`embeddings.py`; nightly smoke; patch release path; notebooks pin the package, not the SDKs (package pins compatible SDK ranges) |
| openai 1.x/2.x conflict resurfaces via a stray lesson pin | package requires `openai>=2,<3`; the pin-sweep deletes all per-notebook `openai==1.107.0` pins (flagged in the update plan) |
| PyPI name taken / squatting | check week 1; fallback names ready (`tai-aitutor-kit`, `towardsai-aitutor`); GitHub-first means no launch dependency |
| Re-embedded prebuilt store lags (Gemini default coupling) | `datasets.prebuilt_chroma(embed_provider=...)` supports both stores during transition; OpenAI store stays available until the org-hosted Gemini store ships |
| Colab quirks (install/restart, `importlib.reload(site)`, torch downloads) | `setup_notebook()` owns the workaround in ONE place; heavy extras optional so default install has no torch |
| Free-tier rate limits (Gemini default) hit students mid-lesson | batching + retry with clear messages in `embed`/`generate`; lesson notes on free-tier limits; `ask_batch(concurrency=)` conservative defaults |
| Old cohorts' notebooks break when archives move | archive-not-delete policy; archived notebooks keep pinned old installs and a banner pointing to the replacement |
| Golden numbers drift from live model behavior | dated tables + regeneration scripts (Section 8); fragile-demo policy from the update plan applies to eval outputs too |
| Two-repo drift during the port (Backup vs working) | Backup repo is read-only reference; all migration lands in `ai-tutor-rag-system`; archives make the old state discoverable |

---

## 11. Relationship to the Section 13 teaching eval package

The Course Update Plan (carried decision 3) ships a **separate small teaching package** for the agent-eval harness: grading/reporting code, judge prompts, memory-preset configs, a synthetic multi-turn dataset, pre-run result bundles (offline grading), Ollama/DeepSeek tiers. That package stays separate — it ships datasets and presets, targets the production agent, and has its own privacy constraints (real student text never ships).

Coordination points: same org namespace and tooling (suggest `tai-agent-evals`); same judge-prompt style and verdict-schema conventions as `tai_aitutor.evals` so students see one idiom; `tai_aitutor` is a dependency it may reuse for `generate`/`extract`; no code duplication of metrics that already live in `tai_aitutor.evals` (single-answer judges live here; session/memory/cost grading lives there).

---

## 12. Open decisions for the team

1. **Import name:** `tai_aitutor` (matches pip name) vs shorter `aitutor` — decide before `v0.1.0`; it appears in every notebook cell. (Recommendation: `tai_aitutor`, unambiguous and brandable.)
2. **License:** MIT recommended (students fork it in certification projects) — confirm.
3. **Flat vs namespaced imports in notebooks:** `from tai_aitutor import generate, search, answer` (recommended for teaching) vs `import tai_aitutor as tai`. Pick one style and use it in every notebook (readability standard, Decision 4).
4. **How far `03-From_Script_to_Pipeline` goes:** end by importing the package (recommended: "what you built is what you'll import") vs keep the package invisible until Section 4's vector-store lesson.
5. **`datasets` URLs:** confirm the Towards AI org HF account names before Phase 5 freezes them into code.
6. **Gradio rebuild timing:** with Phase 4 (recommended, it's the `Chat` module's proof) vs after course launch.
7. **Does `judge_rerank` ship in the package or stay notebook-only?** It's a teaching-only technique (production uses Cohere). Recommendation: ship it — lesson 17 and the reranking lesson both reuse it, and the framing line ("concept here, dedicated reranker in practice") lives in the lesson.

---

## Appendix A — Scan method and provenance

- Repos scanned July 28, 2026 from the connected folders: `ai-tutor-rag-system` (working, 58 notebooks incl. 7 new; 20+ updated in the July pass) and `Backup/ai-tutor-rag-system-main` (original, 51 notebooks). All `.ipynb` parsed as JSON (code cells), plus `scripts/*.py` and both `requirements.txt`.
- Detection: regex over live code for `llama_index` / `llama-index` / `llama_parse` / `llama_cloud` / `LlamaParse` / `LlamaHub` (case-insensitive), single-line AND parenthesized multi-line imports resolved; commented imports excluded from counts. Method-level usage collected per file (constructor lines, `.method(` calls, `Settings.*` attributes).
- Cross-checked against the Course Update Plan v2 audit (31 LlamaIndex notebooks; 22 live lessons; per-lesson port specs) — counts agree.
- Numbers cited: 82 distinct imported symbols; 31 original-repo notebooks with LlamaIndex code; 23 working-repo files still matching (incl. archives-to-be and one dataset-filename-only match); `.query()` in 47 files, `.as_query_engine()` in 40, `VectorStoreIndex` in 48, `Settings` in 40, `OpenAIEmbedding`/`OpenAI` in 41 each, `ChromaVectorStore` in 35 (counts sum both repos).
- Scan artifacts (regenerable): import inventory, per-file usage map, per-repo symbol split — kept with this plan's source session; the scan script is 60 lines and should move into the repo as `tools/scan_llamaindex.py` so the **final "nothing left" sweep** (the update plan's LlamaIndex-removal net) is one command whose output must be empty (minus the documented corpus-label exceptions).





---

## Appendix B — Verbatim import inventory (every `from llama_index... import ...` found in code)

Distinct notebooks per repo: **orig** = `Backup/ai-tutor-rag-system-main`, **upd** = `ai-tutor-rag-system` (post-update working copy). Counts are notebooks, not occurrences.

| Import (module :: symbol) | orig | upd |
|---|---|---|
| `llama_index.core :: Document` | 17 | 10 |
| `llama_index.core :: PromptTemplate` | 1 | 0 |
| `llama_index.core :: QueryBundle` | 2 | 2 |
| `llama_index.core :: Settings` | 23 | 17 |
| `llama_index.core :: SimpleDirectoryReader` | 4 | 4 |
| `llama_index.core :: SimpleKeywordTableIndex` | 1 | 1 |
| `llama_index.core :: StorageContext` | 2 | 1 |
| `llama_index.core :: VectorStoreIndex` | 28 | 20 |
| `llama_index.core :: get_response_synthesizer` | 1 | 1 |
| `llama_index.core :: load_index_from_storage` | 1 | 1 |
| `llama_index.core.agent.workflow :: AgentStream` | 2 | 1 |
| `llama_index.core.agent.workflow :: FunctionAgent` | 3 | 2 |
| `llama_index.core.agent.workflow :: ReActAgent` | 3 | 2 |
| `llama_index.core.agent.workflow :: ToolCallResult` | 2 | 1 |
| `llama_index.core.embeddings :: resolve_embed_model` | 1 | 1 |
| `llama_index.core.evaluation :: BatchEvalRunner` | 6 | 3 |
| `llama_index.core.evaluation :: CorrectnessEvaluator` | 1 | 0 |
| `llama_index.core.evaluation :: EmbeddingQAFinetuneDataset` | 4 | 1 |
| `llama_index.core.evaluation :: FaithfulnessEvaluator` | 6 | 3 |
| `llama_index.core.evaluation :: RelevancyEvaluator` | 6 | 3 |
| `llama_index.core.evaluation :: RetrieverEvaluator` | 7 | 5 |
| `llama_index.core.evaluation :: generate_question_context_pairs` | 4 | 3 |
| `llama_index.core.extractors :: KeywordExtractor` | 7 | 6 |
| `llama_index.core.extractors :: QuestionsAnsweredExtractor` | 6 | 5 |
| `llama_index.core.extractors :: SummaryExtractor` | 6 | 5 |
| `llama_index.core.indices.query.query_transform :: HyDEQueryTransform` | 1 | 1 |
| `llama_index.core.indices.query.query_transform.base :: StepDecomposeQueryTransform` | 1 | 1 |
| `llama_index.core.ingestion :: IngestionPipeline` | 9 | 5 |
| `llama_index.core.llms :: ChatMessage` | 1 | 0 |
| `llama_index.core.llms.utils :: LLM` | 1 | 0 |
| `llama_index.core.node_parser :: SentenceSplitter` | 7 | 3 |
| `llama_index.core.node_parser :: SentenceWindowNodeParser` | 1 | 1 |
| `llama_index.core.node_parser :: SimpleNodeParser` | 2 | 2 |
| `llama_index.core.node_parser :: TokenTextSplitter` | 5 | 2 |
| `llama_index.core.postprocessor :: MetadataReplacementPostProcessor` | 1 | 1 |
| `llama_index.core.postprocessor.rankGPT_rerank :: RankGPTRerank` | 1 | 1 |
| `llama_index.core.postprocessor.types :: BaseNodePostprocessor` | 1 | 1 |
| `llama_index.core.prompts :: PromptTemplate` | 2 | 1 |
| `llama_index.core.query_engine :: RetrieverQueryEngine` | 1 | 1 |
| `llama_index.core.query_engine :: RouterQueryEngine` | 1 | 1 |
| `llama_index.core.query_engine :: SubQuestionQueryEngine` | 2 | 2 |
| `llama_index.core.query_engine.multistep_query_engine :: MultiStepQueryEngine` | 1 | 1 |
| `llama_index.core.query_engine.transform_query_engine :: TransformQueryEngine` | 1 | 1 |
| `llama_index.core.question_gen :: LLMQuestionGenerator` | 1 | 1 |
| `llama_index.core.question_gen.llm_generators :: LLMQuestionGenerator` | 1 | 1 |
| `llama_index.core.retrievers :: BaseRetriever` | 1 | 1 |
| `llama_index.core.retrievers :: KeywordTableSimpleRetriever` | 1 | 1 |
| `llama_index.core.retrievers :: VectorIndexRetriever` | 1 | 1 |
| `llama_index.core.schema :: BaseNode` | 1 | 0 |
| `llama_index.core.schema :: Document` | 2 | 1 |
| `llama_index.core.schema :: MetadataMode` | 1 | 0 |
| `llama_index.core.schema :: NodeWithScore` | 2 | 2 |
| `llama_index.core.schema :: TextNode` | 2 | 1 |
| `llama_index.core.selectors :: LLMSingleSelector` | 1 | 1 |
| `llama_index.core.selectors :: PydanticSingleSelector` | 1 | 1 |
| `llama_index.core.storage.storage_context :: StorageContext` | 1 | 1 |
| `llama_index.core.text_splitter :: TokenTextSplitter` | 5 | 4 |
| `llama_index.core.tools :: QueryEngineTool` | 4 | 4 |
| `llama_index.core.tools :: ToolMetadata` | 2 | 2 |
| `llama_index.core.tools.tool_spec.load_and_search :: LoadAndSearchToolSpec` | 1 | 1 |
| `llama_index.core.vector_stores :: FilterCondition` | 1 | 1 |
| `llama_index.core.vector_stores :: FilterOperator` | 1 | 1 |
| `llama_index.core.vector_stores :: MetadataFilter` | 1 | 1 |
| `llama_index.core.vector_stores :: MetadataFilters` | 1 | 1 |
| `llama_index.core.workflow :: Context` | 3 | 2 |
| `llama_index.core.workflow :: Event` | 1 | 0 |
| `llama_index.core.workflow :: StartEvent` | 1 | 0 |
| `llama_index.core.workflow :: StopEvent` | 1 | 0 |
| `llama_index.core.workflow :: Workflow` | 1 | 0 |
| `llama_index.core.workflow :: step` | 1 | 0 |
| `llama_index.embeddings.adapter :: AdapterEmbeddingModel` | 1 | 1 |
| `llama_index.embeddings.cohere :: CohereEmbedding` | 2 | 2 |
| `llama_index.embeddings.huggingface :: HuggingFaceEmbedding` | 3 | 2 |
| `llama_index.embeddings.openai :: OpenAIEmbedding` | 24 | 17 |
| `llama_index.finetuning :: EmbeddingAdapterFinetuneEngine` | 1 | 1 |
| `llama_index.finetuning :: EmbeddingQAFinetuneDataset` | 1 | 1 |
| `llama_index.finetuning.embeddings.common :: EmbeddingQAFinetuneDataset` | 1 | 1 |
| `llama_index.llms.google_genai :: GoogleGenAI` | 10 | 5 |
| `llama_index.llms.openai :: OpenAI` | 24 | 17 |
| `llama_index.llms.perplexity :: Perplexity` | 1 | 0 |
| `llama_index.llms.together :: TogetherLLM` | 1 | 0 |
| `llama_index.postprocessor.cohere_rerank :: CohereRerank` | 1 | 1 |
| `llama_index.readers.web :: FireCrawlWebReader` | 1 | 0 |
| `llama_index.readers.wikipedia :: WikipediaReader` | 1 | 1 |
| `llama_index.tools.google :: GoogleSearchToolSpec` | 1 | 1 |
| `llama_index.tools.tavily_research :: TavilyToolSpec` | 1 | 0 |
| `llama_index.vector_stores.chroma :: ChromaVectorStore` | 20 | 15 |
| `llama_index.vector_stores.qdrant :: QdrantVectorStore` | 1 | 1 |
| `llama_parse :: LlamaParse` | 2 | 2 |

Plain `import llama_index` statements: none (all usage is `from ... import ...`). Commented-out imports excluded from counts but catalogued in Section 2 where meaningful (`generate_qa_embedding_pairs`, `LinearAdapterEmbeddingModel`, `SentenceTransformersFinetuneEngine`, `QueryPipeline`, `KnowledgeGraphRAGRetriever`-adjacent prose).
