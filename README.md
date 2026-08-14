# tai-aitutor — Migrating from LlamaIndex

Plain-Python building blocks for RAG: provider-neutral LLM calls, embeddings, chunking,
retrieval, and evaluation. Small flat modules, plain functions, readable source. If you
can't read a module in one sitting, that's a bug.

**Coming from LlamaIndex?** What you know carries over; the surface is what changes. Where
you set `Settings`, built an `Index`, handed a question to a `QueryEngine` and scored it
with an `Evaluator`, you now call `configure()`, open a Chroma collection, run `search()`
then `build_rag_prompt()` + `generate()`, and score with `hit_rate` / `reciprocal_rank` /
the judge functions. Each step you knew as a class is a function here. The
[migration map](#migration-map-llamaindex--tai_aitutor) below pairs them symbol by
symbol — read it as a translation table, not a rebuild.

The package supplies the parts inside each step. It does not supply the pipeline: the
indexing loop and the retrieve → prompt → generate sequence stay in your code, where you
can see and change them. Composites that would hide several steps behind one call are
deliberately absent.

## Install

Requires **Python 3.12+** (Colab's current runtime, so the floor stays there while it does).

```bash
# the three providers
pip install "tai-aitutor[gemini,openai,anthropic]"

# + the vector store
pip install "tai-aitutor[gemini,openai,anthropic,rag]"
```

`cohere` (reranking, Cohere embeddings) and `sentence-transformers` (local embeddings) are
**optional heavies**: the package never imports them unless you call `rerank` /
`embed_cohere` / `embed_local`. Install them only if you use those — via the extras
(`[rerank]`, `[local]`) or directly (`pip install cohere sentence-transformers`); any
recent version works.

## Quickstart

```python
from tai_aitutor import (
    build_rag_prompt, chunk_document, configure, embed, generate,
    get_collection, load_csv, search, setup_notebook, show_answer,
)

IN_COLAB = setup_notebook(required_keys=("GOOGLE_API_KEY",))   # Colab Secrets or .env
PROVIDER = "gemini"   # @param ["gemini", "openai", "anthropic"]
configure(provider=PROVIDER)                                    # replaces Settings

docs = load_csv("articles.csv", text_col="content",
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

## Loading data

The package ships **no dataset URLs and no downloaders**. Download however you like, then
hand the file to `load_csv`:

```python
docs = load_csv("articles.csv", text_col="content",
                meta_cols=("title", "url", "source"), id_col="title")
```

Other formats are four lines of your own code, and reading them is worth more than a
wrapper. `QADataset.load(path)` opens LlamaIndex's `EmbeddingQAFinetuneDataset` JSON
byte-for-byte, so existing eval files work unchanged.

## Migration map: LlamaIndex → tai_aitutor

Delete on sight: every `nest_asyncio.apply()` call (it existed only for LlamaIndex's
async), every `llama-index-*` pip pin, and `LLAMA_CLOUD_API_KEY` setup.

Where a row says the replacement is "your own code", that is the point — the operation was
small enough that a wrapper cost more than it saved.

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
| `llm.chat([ChatMessage(...)])` | the provider's own message list and SDK call |
| `llm.structured_predict(S, ...)` / `as_structured_llm` | `extract(prompt, S)` |
| streaming / `print_response_stream()` | the provider's own streaming call |
| `OpenAIEmbedding` / `CohereEmbedding(input_type=...)` / `HuggingFaceEmbedding` / `resolve_embed_model("local:...")` | `embed(texts, task="document"\|"query")` / `embed_cohere(..., output_dimension=1536)` / `embed_local(..., query_prompt=...)` |

### Documents, chunking, ingestion, storage

| LlamaIndex | tai_aitutor |
|---|---|
| `Document` / `TextNode` / `NodeWithScore` | `Document` / `Chunk` / `ScoredChunk` |
| `SimpleDirectoryReader("d").load_data()` / `WikipediaReader` | `load_csv(...)`; other sources are a screen of your own code |
| `FireCrawlWebReader` | the `firecrawl-py` SDK directly → `Document(...)` |
| `LlamaParse` + `file_extractor` | `pypdf`, or the provider's native file understanding |
| `node.get_content(metadata_mode=MetadataMode.NONE)` | `chunk.text` — explicit fields, no modes |
| `PromptTemplate("...")` | f-strings; shared constants in `tai_aitutor.prompts` |
| `TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)` | `chunk(text, 512, 128)` |
| `SentenceSplitter` / `SimpleNodeParser.from_defaults(...)` | `chunk_sentences(...)` |
| `SentenceWindowNodeParser.from_defaults(window_size=3)` | `sentence_window_chunks(text, window_size=3)` |
| `KeywordExtractor` / `SummaryExtractor` / `QuestionsAnsweredExtractor` in `transformations=` | one `extract()` call with your own schema |
| `IngestionPipeline(transformations=[...], vector_store=vs).run(documents=docs)` | the chunk → embed → `col.add()` loop, in your code |
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
| `response_mode="refine"` / `"tree_summarize"` | multi-call loops over `build_rag_prompt` + `generate`, where their cost is visible |
| `SimpleKeywordTableIndex` + `KeywordTableSimpleRetriever` | `BM25Index().build(get_all_chunks(col))` — real Okapi BM25 |
| custom `BaseRetriever` round-robin merge | `rrf_fuse(search(...), bm25.search(...))` — real Reciprocal Rank Fusion |
| `CohereRerank(...)` in `node_postprocessors` | `rerank(q, hits)` — an explicit stage (v4-fast, top 5, floor 0.10) |
| `RankGPTRerank` / custom `BaseNodePostprocessor` | `judge_rerank(q, hits)` — the judge's order and scores kept |
| `MetadataReplacementPostProcessor(target_metadata_key="window")` | `expand_window(hits)` |
| `HyDEQueryTransform` + `TransformQueryEngine` | `hyde_search(q, col)` |
| `LLMQuestionGenerator` + `QueryEngineTool` + `SubQuestionQueryEngine` | `decompose_question(q)` + your loop over the sub-questions |
| `StepDecomposeQueryTransform` + `MultiStepQueryEngine` | `rewrite_query(q)` + your follow-up loop |
| `QueryBundle(q)` | the string itself |
| (token budgeting) | `n_tokens()` + your own budget loop |

### Evaluation

| LlamaIndex | tai_aitutor |
|---|---|
| `generate_question_context_pairs(...)` | `extract()` with your own schema, over your corpus |
| `EmbeddingQAFinetuneDataset.from_json/save_json` | `QADataset.load/save` — **same JSON**, old files open unchanged |
| `RetrieverEvaluator.from_metric_names(["mrr","hit_rate"]).aevaluate_dataset(...)` | `hit_rate(gold_id, retrieved_ids)` / `reciprocal_rank(gold_id, retrieved_ids)` per query; `evaluate_retrieval(qa, search_fn=..., top_k=k)` over a dataset — any retriever callable, so rerankers get measured |
| `FaithfulnessEvaluator/RelevancyEvaluator/CorrectnessEvaluator` | `judge_faithfulness/judge_relevancy/judge_correctness` → typed verdicts |
| `BatchEvalRunner(...).aevaluate_queries(...)` + `nest_asyncio` | a thread pool over `judge_*` — no asyncio, no hidden fan-out |

### Agents, chat, tools, routing

| LlamaIndex | tai_aitutor |
|---|---|
| `FunctionAgent` / `ReActAgent` + `Context(agent)` | the tool-calling loop, in your code |
| `AgentStream` / `ToolCallResult` events | the SDK's own streamed blocks |
| `index.as_chat_engine(chat_mode=..., memory=...)` / `ChatSummaryMemoryBuffer` | the message list you keep, trimmed or summarised by you |
| `QueryEngineTool.from_defaults(...)` + `ToolMetadata` | `@tool` on a plain function — the signature is the schema |
| `TavilyToolSpec` / `GoogleSearchToolSpec` + `LoadAndSearchToolSpec` | `search_web(q)` (+ `tool(search_web)`) |
| `RouterQueryEngine` + `LLMSingleSelector`/`PydanticSingleSelector` | `route(q, routes)` + your `if/else` |
| `Workflow` / `@step` / `StartEvent` / `StopEvent` | plain functions and loops |

### Fine-tuning

No package equivalent — fine-tuning is a training script, not a building block.

| LlamaIndex | tai_aitutor |
|---|---|
| `EmbeddingAdapterFinetuneEngine.finetune()` / `AdapterEmbeddingModel` | `sentence-transformers` directly |
| `generate_qa_embedding_pairs` (legacy) | your own pair-mining loop |
| before/after measurement | `evaluate_retrieval(qa, search_fn=...)` — the same hit rate / MRR ruler |

## What's in the package

`config` (`configure`, `setup_notebook`, `require_keys`, `in_colab`) ·
`llm` (`generate`, `extract`) · `embeddings` (`embed`, `embed_cohere`, `embed_local`,
`EMBED_DIM`) · `tokens` (`n_tokens`) · `documents` (`Document`, `load_csv`) ·
`chunking` (`Chunk`, `chunk`, `chunk_document`, `chunk_sentences`,
`heading_aware_markdown_chunks`, `sentence_window_chunks`) ·
`vectorstore` (`get_collection`, `reset_collection`, `get_all_chunks`,
`build_where_filter`) · `retrieval` (`search`, `ScoredChunk`, `expand_window`,
`code_tokenize`, `BM25Index`, `rrf_fuse`, `rerank`, `judge_rerank`, `rewrite_query`,
`hyde_search`, `decompose_question`) · `synthesis` (`build_rag_prompt`) ·
`evals` (`QADataset`, `hit_rate`, `reciprocal_rank`, `evaluate_retrieval`,
`sweep_top_k`, the three judges) · `tools` (`Tool`, `tool`, `search_web`) ·
`router` (`route`, `RouteDecision`) · `display` · `errors` (all subclass `ValueError`)

Runnable end-to-end examples in [`examples/`](examples/).

## Design rules

1. **Functions over object graphs.** State lives in one `configure()` call and in the
   objects you pass explicitly — the collection, the index, the dataset. Where LlamaIndex
   handed you an object carrying the pipeline inside it, you hold the pieces.
2. **No hidden composites.** If a call would bundle several steps you should be able to
   see, it isn't here.
3. **No data downloads in the package.** It loads the file you hand it, nothing more.
4. **Loud failures.** Typo'd kwargs raise `TypeError`, data parsing never uses `eval()`,
   missing keys and missing extras say exactly what to install or set.

## Development

```bash
pip install -e ".[gemini,openai,anthropic,rag,rerank]" pytest ruff
pytest              # fully offline — provider SDKs are faked
ruff check src tests
```

Releases: bump `__version__` in `src/tai_aitutor/__init__.py`, commit, then
`git tag vX.Y.Z && git push --tags` — `.github/workflows/release.yml` publishes to PyPI
via Trusted Publishing.

## License

MIT — see [LICENSE](LICENSE).
