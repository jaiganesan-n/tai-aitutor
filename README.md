# tai-aitutor — Migrating from LlamaIndex

The Towards AI course toolkit — plain-Python building blocks for the RAG AI Tutor built across the Full Stack AI Engineer course: provider-neutral LLM calls, embeddings, chunking, retrieval, and evaluation.

**Coming from the LlamaIndex version of the course?** What you learned carries over; the surface is what changes. Where the course used to set `Settings`, build an `Index`, hand a question to a `QueryEngine`, and score it with an `Evaluator`, it now calls `configure()`, opens a Chroma collection, runs `search()` and then `build_rag_prompt()` + `generate()`, and scores with `hit_rate` / `reciprocal_rank` / the judge functions. Each step you knew as a class is a function here, and the [migration map](#migration-map-llamaindex--tai_aitutor) below pairs them symbol by symbol — read it as a translation table, not a rebuild.

**Coming from Sections 1–8?** You have already written this code. `chunk`, `embed`, `search`, `build_rag_prompt`, `hit_rate` — each one is the function you built inline in a lesson, with the same name, signature, and return. Importing it from here is picking your own work back up, which is why the package starts at Section 9 and not before. Every module docstring names the lesson that builds its contents, so you can always trace a function back to the cell you wrote it in.

Small flat modules, plain functions, readable source. If you can't read a module in one sitting, that's a bug.

## Install

Requires **Python 3.12+** (3.12 is Colab's current runtime, so the floor stays there as long as Colab does).

```bash
# Course profile A (every notebook): the three providers
pip install "tai-aitutor[gemini,openai,anthropic]"

# Course profile B (retrieval lessons): + the vector store
pip install "tai-aitutor[gemini,openai,anthropic,rag]"
```

`cohere` (reranking, Cohere embeddings) and `sentence-transformers` (local embeddings)
are **lesson-specific**: the package never imports them unless you call
`rerank` / `embed_cohere` / `embed_local`, so install them only in the lessons that use
them — via the extras (`[rerank]`, `[local]`) or directly
(`pip install cohere sentence-transformers`); any recent version works.

## Quickstart

```python
from tai_aitutor import (
    build_rag_prompt, chunk_document, configure, embed, generate,
    get_collection, load_csv, search, setup_notebook, show_answer,
)

IN_COLAB = setup_notebook(required_keys=("GOOGLE_API_KEY",))   # Colab Secrets or .env
PROVIDER = "gemini"   # @param ["gemini", "openai", "anthropic"]
configure(provider=PROVIDER)                                    # replaces Settings

docs = load_csv("mini-llama-articles.csv", text_col="content",
                meta_cols=("title", "url", "source"), id_col="title")
col = get_collection("kb", path="./db")                         # the collection IS the index

chunks = [c for d in docs for c in chunk_document(d)]           # chunk → embed → upsert:
col.add(                                                        # three steps, all visible
    ids=[c.id for c in chunks],
    documents=[c.text for c in chunks],
    embeddings=embed([c.text for c in chunks]),
    metadatas=[c.metadata for c in chunks],
)

question = "What is RAG?"
hits = search(question, col, top_k=5)                           # retrieve
show_answer(generate(build_rag_prompt(question, hits)), hits)   # prompt → generate
```

Notice what the package does **not** do for you: the indexing loop and the
retrieve → prompt → generate sequence stay in your code, because those are the
steps the lessons teach. The package carries the parts inside each step.

## Data stays in your notebooks — by design

This package ships **no dataset URLs and no downloaders**. Datasets are downloaded in the
notebook (wget / `urllib` / `hf_hub_download`), so the course decides where data is hosted
and can move it without a package release; the package only loads whatever file you hand it:

```python
# !wget https://.../mini-llama-articles.csv
docs = load_csv("mini-llama-articles.csv", text_col="content",
                meta_cols=("title", "url", "source"), id_col="title")
```

Other formats are loaded in the lesson that needs them — a JSONL reader is four
lines and the reading of it is part of the point.

`QADataset.load(path)` opens the course's existing `rag_eval_dataset*.json` files
byte-for-byte (the legacy `EmbeddingQAFinetuneDataset` JSON shape).

## Migration map: LlamaIndex → tai_aitutor

Also delete on sight: every `nest_asyncio.apply()` cell (only existed for LlamaIndex's
async), every `llama-index-*` pip pin, and `LLAMA_CLOUD_API_KEY` setup.

### Config

| LlamaIndex | tai_aitutor |
|---|---|
| `Settings.llm = OpenAI(model=..., additional_kwargs={'reasoning_effort':'minimal'})` | `configure(provider="openai")` — or per call: `generate(..., provider="openai", reasoning_effort="minimal")` |
| `Settings.embed_model = OpenAIEmbedding(...)` | `configure(embed_provider="openai")` |
| `Settings.text_splitter / chunk_size / chunk_overlap` | pass sizes to `chunk()` / `chunk_document()` directly |

### LLMs and embeddings

| LlamaIndex | tai_aitutor |
|---|---|
| `OpenAI(...).complete(p)` / `GoogleGenAI(...)` / `Perplexity(...)` / `TogetherLLM(...)` | `generate(p, system=...)` (+ `provider=` / `model=`; Together/Perplexity/DeepSeek/Ollama via built-in `base_url`s) |
| `llm.chat([ChatMessage(...)])` | the message list and the SDK call, written in the agents lesson |
| `llm.structured_predict(S, ...)` / `as_structured_llm` | `extract(prompt, S)` |
| streaming / `print_response_stream()` | the provider's own streaming call, written in the lesson that streams |
| `OpenAIEmbedding` / `CohereEmbedding(input_type=...)` / `HuggingFaceEmbedding` / `resolve_embed_model("local:...")` | `embed(texts, task="document"\|"query")` / `embed_cohere(..., output_dimension=1536)` / `embed_local(..., query_prompt=...)` |

### Documents, chunking, ingestion, storage

| LlamaIndex | tai_aitutor |
|---|---|
| `Document` / `TextNode` / `NodeWithScore` | `Document` / `Chunk` / `ScoredChunk` |
| `SimpleDirectoryReader("d").load_data()` / `WikipediaReader` | `load_csv(...)`; other loaders are written in the lesson that needs them (each is a screen of code) |
| `FireCrawlWebReader` | the `firecrawl-py` SDK directly → `Document(...)` |
| `LlamaParse` + `file_extractor` | `pypdf` baseline + native file understanding (Parsing lesson) |
| `node.get_content(metadata_mode=MetadataMode.NONE)` | `chunk.text` — explicit fields, no modes |
| `PromptTemplate("...")` | f-strings; shared constants in `tai_aitutor.prompts` |
| `TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)` | `chunk(text, 512, 128)` |
| `SentenceSplitter` / `SimpleNodeParser.from_defaults(...)` | `chunk_sentences(...)` |
| `SentenceWindowNodeParser.from_defaults(window_size=3)` | `sentence_window_chunks(text, window_size=3)` |
| `KeywordExtractor` / `SummaryExtractor` / `QuestionsAnsweredExtractor` in `transformations=` | one `extract()` call with your own schema, in the lesson that enriches |
| `IngestionPipeline(transformations=[...], vector_store=vs).run(documents=docs)` | the chunk → embed → `col.add()` loop, visible in every retrieval lesson |
| `chromadb` + `ChromaVectorStore` + `StorageContext.from_defaults(...)` | `col = get_collection(name, path="./db")` — chromadb only, no wrappers |
| `VectorStoreIndex.from_documents(docs)` | `get_collection(...)` + the indexing loop |
| `VectorStoreIndex.from_vector_store(vs)` | nothing — the collection already IS the index |
| `index.insert(doc)` / `persist` / `load_index_from_storage` | `col.add(...)` / the `path=` argument on `get_collection` |
| `QdrantVectorStore` + `MetadataFilters/MetadataFilter/FilterOperator/FilterCondition` | Chroma `where=` dicts; `build_where_filter(sources)`; text match via `where_document={"$contains": ...}` |

### Retrieval and answering

| LlamaIndex | tai_aitutor |
|---|---|
| `index.as_retriever(similarity_top_k=k).retrieve(q)` / `VectorIndexRetriever` | `search(q, col, top_k=k)` |
| `index.as_query_engine(...).query(q)` + `response.response` / `.source_nodes` | `hits = search(q, col)` then `generate(build_rag_prompt(q, hits))` — the two halves, separately |
| `get_response_synthesizer` / `RetrieverQueryEngine(retriever, ...)` | your retriever function + `build_rag_prompt` + `generate` |
| `response_mode="refine"` / `"tree_summarize"` | multi-call loops over `build_rag_prompt` + `generate`, written where their cost is visible |
| `SimpleKeywordTableIndex` + `KeywordTableSimpleRetriever` | `BM25Index().build(get_all_chunks(col))` — real Okapi BM25 |
| custom `BaseRetriever` round-robin merge | `rrf_fuse(search(...), bm25.search(...))` — real Reciprocal Rank Fusion, composed in your code |
| `CohereRerank(...)` in `node_postprocessors` | `rerank(q, hits)` — explicit stage, production constants (v4-fast, top 5, floor 0.10) |
| `RankGPTRerank` / custom `BaseNodePostprocessor` | `judge_rerank(q, hits)` — judge's order and scores kept |
| `MetadataReplacementPostProcessor(target_metadata_key="window")` | `expand_window(hits)` |
| `HyDEQueryTransform` + `TransformQueryEngine` | `hyde_search(q, col)` |
| `LLMQuestionGenerator` + `QueryEngineTool` + `SubQuestionQueryEngine` | `decompose_question(q)` + your loop over the sub-questions |
| `StepDecomposeQueryTransform` + `MultiStepQueryEngine` | the follow-up loop, written in the Query Variation lesson |
| `QueryBundle(q)` | the string itself |
| (production token budget) | `n_tokens()` + your own budget loop, in the context lesson |

### Evaluation

| LlamaIndex | tai_aitutor |
|---|---|
| `generate_question_context_pairs(...)` | the generator prompt + loop, written in the evaluation lesson |
| `EmbeddingQAFinetuneDataset.from_json/save_json` | `QADataset.load/save` — **same JSON**, old files open unchanged |
| `RetrieverEvaluator.from_metric_names(["mrr","hit_rate"]).aevaluate_dataset(...)` | `hit_rate(gold_id, retrieved_ids)` / `reciprocal_rank(gold_id, retrieved_ids)` per query; `evaluate_retrieval(qa, search_fn=..., top_k=k)` over a dataset — any retriever callable, so rerankers get measured |
| `FaithfulnessEvaluator/RelevancyEvaluator/CorrectnessEvaluator` | `judge_faithfulness/judge_relevancy/judge_correctness` → typed verdicts |
| `BatchEvalRunner(...).aevaluate_queries(...)` + `nest_asyncio` | a visible thread-pool cell over `judge_*` — no asyncio, no hidden fan-out |

### Agents, chat, tools, routing

| LlamaIndex | tai_aitutor |
|---|---|
| `FunctionAgent` / `ReActAgent` + `Context(agent)` | the tool-calling loop, written in the agents lessons |
| `AgentStream` / `ToolCallResult` events | the SDK's own streamed blocks, read in the lesson |
| `index.as_chat_engine(chat_mode=..., memory=...)` / `ChatSummaryMemoryBuffer` | the message list you keep, trimmed or summarised in the memory lesson |
| `QueryEngineTool.from_defaults(...)` + `ToolMetadata` | `@tool` on a plain function — the signature is the schema |
| `TavilyToolSpec` / `GoogleSearchToolSpec` + `LoadAndSearchToolSpec` | `search_web(q)` (+ `tool(search_web)`); Google CSE variant archived |
| `RouterQueryEngine` + `LLMSingleSelector`/`PydanticSingleSelector` | `route(q, routes)` + your `if/else` |
| `Workflow` / `@step` / `StartEvent` / `StopEvent` | plain functions and loops |

### Fine-tuning

| LlamaIndex | tai_aitutor |
|---|---|
| `EmbeddingAdapterFinetuneEngine.finetune()` / `AdapterEmbeddingModel` | a real `sentence-transformers` fine-tune, written in the Section 8 notebook (Sections 1–8 import nothing from this package) |
| `generate_qa_embedding_pairs` (legacy) | the pair-mining loop, in the same notebook |
| before/after measurement | `evaluate_retrieval(qa, search_fn=...)` — the same Hit Rate / MRR ruler |

## What's in the package

`config` (`configure`, `setup_notebook`, `require_keys`, `in_colab`) ·
`llm` (`generate`, `extract`) · `embeddings` (`embed`, `embed_cohere`, `embed_local`,
`EMBED_DIM`) · `tokens` (`n_tokens`) · `documents` (`Document`, `load_csv`) ·
`chunking` (`Chunk`, `chunk`, `chunk_document`, `chunk_sentences`,
`heading_aware_markdown_chunks`, `sentence_window_chunks`) ·
`vectorstore` (`get_collection`, `reset_collection`, `get_all_chunks`,
`build_where_filter`) · `retrieval` (`search`, `ScoredChunk`, `expand_window`,
`code_tokenize`, `BM25Index`, `rrf_fuse`, `rerank`, `judge_rerank`, `hyde_search`,
`decompose_question`) · `synthesis` (`build_rag_prompt`) ·
`evals` (`QADataset`, `hit_rate`, `reciprocal_rank`, `evaluate_retrieval`,
`sweep_top_k`, the three judges) · `tools` (`Tool`, `tool`, `search_web`) ·
`router` (`route`, `RouteDecision`) · `display` · `errors` (all subclass `ValueError`)

Every one of these has an inline twin in a Section 1–8 lesson. Composites that
would hide several taught steps behind one call are deliberately absent — you
assemble the pipeline, the package supplies the parts.

Runnable end-to-end examples (with explicit data downloads) in [`examples/`](examples/).
History: `CHANGELOG.md`; build tracker: `BUILD_STATUS.md`; the original design +
full LlamaIndex inventory: `docs/PACKAGE_PLAN.md`.

## Design rules

1. **Teach-then-import.** A concept's first appearance is written out in its lesson
   notebook; later notebooks import it from here. Each module docstring names the lesson
   that builds it.
2. **Functions over object graphs.** State lives in one `configure()` call and in the
   objects you pass explicitly — the collection, the index, the dataset. Where LlamaIndex
   handed you an object that carried the pipeline inside it, you hold the pieces.
3. **Data downloads live in notebooks, never in the package.**
4. **Mirror production.** Where the production AI Tutor has an equivalent, constants and
   logic match.
5. **Loud failures.** Typo'd kwargs raise `TypeError`, data parsing never uses `eval()`,
   missing keys and missing extras say exactly what to install or set.

## Development

```bash
pip install -e ".[gemini,openai,anthropic,rag,rerank,parse]" pytest ruff
pytest              # fully offline — provider SDKs are faked
ruff check src tests
```

Releases: bump `__version__` + `CHANGELOG.md`, then `git tag vX.Y.Z && git push --tags` —
`.github/workflows/release.yml` publishes to PyPI via Trusted Publishing.

## License

MIT © Towards AI
