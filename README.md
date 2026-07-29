# tai-aitutor — Migrating from LlamaIndex

The Towards AI course toolkit — plain-Python building blocks for the RAG AI Tutor built across the Full Stack AI Engineer course: provider-neutral LLM calls, embeddings, chunking, retrieval, and evaluation.

This package replaces the LlamaIndex usage the course previously relied on — but it is not a framework, and it is not a LlamaIndex clone. Every function here is code a course lesson builds inline first; the package is where that code lives once you've written it yourself. Small flat modules, plain functions, readable source. If you can't read a module in one sitting, that's a bug.

## Install

Requires **Python 3.12+** (3.12 is Colab's current runtime, so the floor stays there as long as Colab does).

```bash
# Course profile A (every notebook): the three providers
pip install "tai-aitutor[gemini,openai,anthropic]"

# Course profile B (retrieval lessons): + the vector store
pip install "tai-aitutor[gemini,openai,anthropic,rag]"
```

`cohere` (reranking, Cohere embeddings) and `sentence-transformers` (local embeddings,
fine-tuning) are **lesson-specific**: the package never imports them unless you call
`rerank` / `embed_cohere` / `embed_local` / `train_embedder`, so install them only in the
lessons that use them — via the extras (`[rerank]`, `[local]`, `[finetune]`) or directly
(`pip install cohere sentence-transformers`); any recent version works.

## Quickstart

```python
from tai_aitutor import configure, setup_notebook, generate, load_csv, get_collection, ingest, answer

IN_COLAB = setup_notebook(required_keys=("GOOGLE_API_KEY",))   # Colab Secrets or .env
PROVIDER = "gemini"   # @param ["gemini", "openai", "anthropic"]
configure(provider=PROVIDER)                                    # replaces Settings

docs = load_csv("mini-llama-articles.csv", text_col="content",
                meta_cols=("title", "url", "source"), id_col="title")
col = get_collection("kb", path="./db")                         # the collection IS the index
ingest(docs, col)                                               # chunk → embed → upsert
print(answer("What is RAG?", col, top_k=5))                     # retrieve → prompt → generate
```

## Data stays in your notebooks — by design

This package ships **no dataset URLs and no downloaders**. Datasets are downloaded in the
notebook (wget / `urllib` / `hf_hub_download`), so the course decides where data is hosted
and can move it without a package release; the package only loads whatever file you hand it:

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="<org>/ai_tutor_knowledge",
                       filename="ai_tutor_knowledge.jsonl", repo_type="dataset")
docs = load_jsonl(path, text_key="content", id_key="doc_id")

# or: !wget https://.../mini-llama-articles.csv
docs = load_csv("mini-llama-articles.csv", text_col="content",
                meta_cols=("title", "url", "source"), id_col="title")
```

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
| `Settings.text_splitter / chunk_size / chunk_overlap` | pass sizes to `chunk()` / `ingest()` directly |

### LLMs and embeddings

| LlamaIndex | tai_aitutor |
|---|---|
| `OpenAI(...).complete(p)` / `GoogleGenAI(...)` / `Perplexity(...)` / `TogetherLLM(...)` | `generate(p, system=...)` (+ `provider=` / `model=`; Together/Perplexity/DeepSeek/Ollama via built-in `base_url`s) |
| `llm.chat([ChatMessage(...)])` | `chat_completion(messages)` — plain dicts |
| `llm.structured_predict(S, ...)` / `as_structured_llm` | `extract(prompt, S)` |
| streaming / `print_response_stream()` | `generate_stream(...)` / `answer_stream(...)` |
| `OpenAIEmbedding` / `CohereEmbedding(input_type=...)` / `HuggingFaceEmbedding` / `resolve_embed_model("local:...")` | `embed(texts, task="document"\|"query")` / `embed_cohere(..., output_dimension=1536)` / `embed_local(..., query_prompt=...)` |

### Documents, chunking, ingestion, storage

| LlamaIndex | tai_aitutor |
|---|---|
| `Document` / `TextNode` / `NodeWithScore` | `Document` / `Chunk` / `ScoredChunk` |
| `SimpleDirectoryReader("d").load_data()` / `WikipediaReader` | `load_directory("d")` / `load_wikipedia([...])` (+ `load_csv`, `load_jsonl`, `load_hf_dataset`) |
| `FireCrawlWebReader` | the `firecrawl-py` SDK directly → `Document(...)` |
| `LlamaParse` + `file_extractor` | `pypdf` baseline + native file understanding (Parsing lesson) |
| `node.get_content(metadata_mode=MetadataMode.NONE)` | `chunk.text` — explicit fields, no modes |
| `PromptTemplate("...")` | f-strings; shared constants in `tai_aitutor.prompts` |
| `TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)` | `chunk(text, 512, 128)` |
| `SentenceSplitter` / `SimpleNodeParser.from_defaults(...)` | `chunk_sentences(...)` |
| `SentenceWindowNodeParser.from_defaults(window_size=3)` | `sentence_window_chunks(text, window_size=3)` |
| `KeywordExtractor` / `SummaryExtractor` / `QuestionsAnsweredExtractor` in `transformations=` | `ingest(..., enrich=[extract_keywords, extract_summary, extract_questions])` |
| `IngestionPipeline(transformations=[...], vector_store=vs).run(documents=docs)` | `ingest(docs, collection)` |
| `chromadb` + `ChromaVectorStore` + `StorageContext.from_defaults(...)` | `col = get_collection(name, path="./db")` — chromadb only, no wrappers |
| `VectorStoreIndex.from_documents(docs)` | `ingest(docs, col)` |
| `VectorStoreIndex.from_vector_store(vs)` | nothing — the collection already IS the index |
| `index.insert(doc)` / `persist` / `load_index_from_storage` | `ingest([doc], col)` / the `path=` argument |
| `QdrantVectorStore` + `MetadataFilters/MetadataFilter/FilterOperator/FilterCondition` | Chroma `where=` dicts; `build_where_filter(sources)`; text match via `where_document={"$contains": ...}` |

### Retrieval and answering

| LlamaIndex | tai_aitutor |
|---|---|
| `index.as_retriever(similarity_top_k=k).retrieve(q)` / `VectorIndexRetriever` | `search(q, col, top_k=k)` |
| `index.as_query_engine(...).query(q)` + `response.response` / `.source_nodes` | `ans = answer(q, col)` → `ans.text` / `ans.sources` |
| `get_response_synthesizer` / `RetrieverQueryEngine(retriever, ...)` | `answer(q, retriever=my_retriever)` |
| `response_mode="refine"` | dropped by design (broken demo; don't rebuild) |
| `SimpleKeywordTableIndex` + `KeywordTableSimpleRetriever` | `BM25Index().build(get_all_chunks(col))` — real Okapi BM25 |
| custom `BaseRetriever` round-robin merge | `rrf_fuse(...)` / `hybrid_search(q, col, bm25)` |
| `CohereRerank(...)` in `node_postprocessors` | `rerank(q, hits)` — explicit stage, production constants (v4-fast, top 5, floor 0.10) |
| `RankGPTRerank` / custom `BaseNodePostprocessor` | `judge_rerank(q, hits)` — judge's order and scores kept |
| `MetadataReplacementPostProcessor(target_metadata_key="window")` | `expand_window(hits)` |
| `HyDEQueryTransform` + `TransformQueryEngine` | `hyde_search(q, col)` |
| `LLMQuestionGenerator` + `QueryEngineTool` + `SubQuestionQueryEngine` | `decompose_question(q)` + loop, or `subquestion_answer(q, col)` |
| `StepDecomposeQueryTransform` + `MultiStepQueryEngine` | `multi_step_answer(q, col)` |
| `QueryBundle(q)` | the string itself |
| (production token budget) | `pack_context(hits, max_tokens)` |

### Evaluation

| LlamaIndex | tai_aitutor |
|---|---|
| `generate_question_context_pairs(...)` | `make_qa_pairs(col_or_chunks, n_chunks, questions_per_chunk)` |
| `EmbeddingQAFinetuneDataset.from_json/save_json` | `QADataset.load/save` — **same JSON**, old files open unchanged |
| `RetrieverEvaluator.from_metric_names(["mrr","hit_rate"]).aevaluate_dataset(...)` | `evaluate_retrieval(qa, search_fn=..., top_k=k)` — any retriever callable, so rerankers get measured; `sweep_top_k(qa, [2,4,6,8,10])` for one-pass ablations; `report.avg_context_tokens(qa)` for the cost column |
| `FaithfulnessEvaluator/RelevancyEvaluator/CorrectnessEvaluator` | `judge_faithfulness/judge_relevancy/judge_correctness` → typed verdicts |
| `BatchEvalRunner(...).aevaluate_queries(...)` + `nest_asyncio` | `run_judges(rows, judges=(...))` — threads, no asyncio |

### Agents, chat, tools, routing

| LlamaIndex | tai_aitutor |
|---|---|
| `FunctionAgent` / `ReActAgent` + `Context(agent)` | `ToolLoop(tools=[...])` (stateless) or `Chat(tools=[...])` (stateful) |
| `AgentStream` / `ToolCallResult` events | `ChatEvent` from `ask_stream()` / `run_events()` |
| `index.as_chat_engine(chat_mode=..., memory=...)` / `ChatSummaryMemoryBuffer` | `Chat(history="full"\|"window"\|"summary")` — inspect `chat.messages` vs `chat.last_context` |
| `QueryEngineTool.from_defaults(...)` + `ToolMetadata` | `@tool` on a plain function; `make_retrieval_tool(col)` |
| `TavilyToolSpec` / `GoogleSearchToolSpec` + `LoadAndSearchToolSpec` | `search_web(q)` (+ `tool(search_web)`); Google CSE variant archived |
| `RouterQueryEngine` + `LLMSingleSelector`/`PydanticSingleSelector` | `route(q, routes)` + your `if/else` |
| `Workflow` / `@step` / `StartEvent` / `StopEvent` | plain functions and loops |

### Fine-tuning

| LlamaIndex | tai_aitutor |
|---|---|
| `EmbeddingAdapterFinetuneEngine.finetune()` / `AdapterEmbeddingModel` | `train_embedder(train, val, base_model="BAAI/bge-small-en-v1.5")` — sentence-transformers + MNRL (method upgrade per the course plan) |
| `generate_qa_embedding_pairs` (legacy) | `make_training_pairs(chunks)` |
| before/after measurement | `evaluate_embedder(model_path, qa)` — same Hit Rate / MRR ruler |

## What's in the package

`config` · `llm` (`generate`, `generate_stream`, `generate_vision`, `extract`,
`chat_completion`, `ask_batch`) · `embeddings` (`embed`, `embed_cohere`, `embed_local`) ·
`tokens` (`n_tokens`, `truncate`, `estimate_cost`) · `documents` (loaders) · `chunking`
(incl. the production heading-aware chunker) · `vectorstore` (`get_collection`, `ingest`,
`get_all_chunks`, `build_where_filter`) · `retrieval` (`search`, `BM25Index`, `rrf_fuse`,
`hybrid_search`, `rerank`, `judge_rerank`, `hyde_search`, `decompose_question`,
`subquestion_answer`, `multi_step_answer`, `expand_window`, `pack_context`) · `synthesis`
(`answer` family) · `evals` (`QADataset`, metrics, judges, `sweep_top_k`) · `extractors` ·
`tools` · `chat` (`Chat`, `ToolLoop`) · `router` · `finetune` · `display`

Runnable end-to-end examples (with explicit data downloads) in [`examples/`](examples/),
including the Gradio tutor app rebuilt on the package. History: `CHANGELOG.md`; build
tracker: `BUILD_STATUS.md`; the original design + full LlamaIndex inventory:
`docs/PACKAGE_PLAN.md`.

## Design rules

1. **Teach-then-import.** A concept's first appearance is written out in its lesson
   notebook; later notebooks import it from here. Each module docstring names the lesson
   that builds it.
2. **Functions over object graphs.** No hidden state beyond one `configure()` call.
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
