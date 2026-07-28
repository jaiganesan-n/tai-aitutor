# Migrating from LlamaIndex to `tai_aitutor`

The porting cheat-sheet, symbol by symbol. Every LlamaIndex class/function the course used
appears on the left; what you write instead appears on the right. Full design rationale:
[`docs/PACKAGE_PLAN.md`](docs/PACKAGE_PLAN.md).

Also delete on sight: every `nest_asyncio.apply()` cell (only existed for LlamaIndex's
async), every `llama-index-*` pip pin, and `LLAMA_CLOUD_API_KEY` setup.

## Config

| LlamaIndex | tai_aitutor |
|---|---|
| `Settings.llm = OpenAI(model="gpt-5-mini", additional_kwargs={'reasoning_effort':'minimal'})` | `configure(provider="openai")` — or per call: `generate(..., provider="openai", reasoning_effort="minimal")` |
| `Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")` | `configure(embed_provider="openai")` |
| `Settings.text_splitter / chunk_size / chunk_overlap` | pass sizes to `chunk()` / `ingest()` directly |

## LLMs and embeddings

| LlamaIndex | tai_aitutor |
|---|---|
| `OpenAI(...).complete(p)` / `GoogleGenAI(...)` / `Perplexity(...)` / `TogetherLLM(...)` | `generate(p, system=...)` (+ `provider=` / `model=`; Together/Perplexity/DeepSeek/Ollama via built-in `base_url`s) |
| `llm.chat([ChatMessage(...)])` | `chat_completion(messages)` — plain dicts |
| `llm.structured_predict(S, ...)` / `as_structured_llm` | `extract(prompt, S)` |
| streaming / `print_response_stream()` | `generate_stream(...)` / `answer_stream(...)` |
| `OpenAIEmbedding` / `CohereEmbedding(input_type=...)` / `HuggingFaceEmbedding` / `resolve_embed_model("local:...")` | `embed(texts, task="document"|"query")` / `embed_cohere(...)` / `embed_local(...)` |

## Documents, chunking, ingestion, storage

| LlamaIndex | tai_aitutor |
|---|---|
| `Document(text=..., metadata=...)` / `TextNode` / `NodeWithScore` | `Document` / `Chunk` / `ScoredChunk` |
| `SimpleDirectoryReader("d").load_data()` / `WikipediaReader` | `load_directory("d")` / `load_wikipedia([...])` (+ `load_csv`, `load_jsonl`, `load_hf_dataset`) |
| `FireCrawlWebReader` | the `firecrawl-py` SDK directly → `Document(text=..., metadata=...)` (already done in the ported scraping notebook) |
| `node.get_content(metadata_mode=MetadataMode.NONE)` | `chunk.text` — explicit fields, no modes |
| `PromptTemplate("...")` | f-strings; shared prompt text lives as constants in `tai_aitutor.prompts` |
| `LlamaParse` + `file_extractor` | `pypdf` baseline + native file understanding (see the Parsing lesson); `research_papers()` for the sample data |
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

## Retrieval and answering

| LlamaIndex | tai_aitutor |
|---|---|
| `index.as_retriever(similarity_top_k=k).retrieve(q)` / `VectorIndexRetriever` | `search(q, col, top_k=k)` |
| `index.as_query_engine(...).query(q)` + `response.response` / `.source_nodes` | `ans = answer(q, col)` → `ans.text` / `ans.sources` |
| `get_response_synthesizer` / `RetrieverQueryEngine(retriever, ...)` | `answer(q, retriever=my_retriever)` |
| `response_mode="refine"` | dropped by design (broken demo; don't rebuild) |
| `SimpleKeywordTableIndex` + `KeywordTableSimpleRetriever` | `BM25Index().build(get_all_chunks(col))` — real Okapi BM25 |
| custom `BaseRetriever` round-robin merge | `rrf_fuse(...)` / `hybrid_search(q, col, bm25)` |
| `CohereRerank(top_n=2, model='rerank-english-v3.0')` in `node_postprocessors` | `rerank(q, hits)` — explicit stage, production constants (v4-fast, top 5, floor 0.10) |
| `RankGPTRerank` / custom `BaseNodePostprocessor` | `judge_rerank(q, hits)` — judge's order and scores kept |
| `MetadataReplacementPostProcessor(target_metadata_key="window")` | `expand_window(hits)` |
| `HyDEQueryTransform` + `TransformQueryEngine` | `hyde_search(q, col)` |
| `LLMQuestionGenerator` + `QueryEngineTool` + `SubQuestionQueryEngine` | `decompose_question(q)` + loop, or `subquestion_answer(q, col)` |
| `StepDecomposeQueryTransform` + `MultiStepQueryEngine` | `multi_step_answer(q, col)` |
| `QueryBundle(q)` | the string itself |
| (production token budget) | `pack_context(hits, max_tokens)` |

## Evaluation

| LlamaIndex | tai_aitutor |
|---|---|
| `generate_question_context_pairs(...)` | `make_qa_pairs(col_or_chunks, n_chunks, questions_per_chunk)` |
| `EmbeddingQAFinetuneDataset.from_json/save_json` | `QADataset.load/save` — **same JSON**, old files open unchanged |
| `RetrieverEvaluator.from_metric_names(["mrr","hit_rate"]).aevaluate_dataset(...)` | `evaluate_retrieval(qa, search_fn=..., top_k=k)` (any retriever callable — rerank gets measured) |
| `FaithfulnessEvaluator/RelevancyEvaluator/CorrectnessEvaluator` | `judge_faithfulness/judge_relevancy/judge_correctness` → typed verdicts |
| `BatchEvalRunner(...).aevaluate_queries(...)` + `nest_asyncio` | `run_judges(rows, judges=(...))` — threads, no asyncio |

## Agents, chat, tools, routing

| LlamaIndex | tai_aitutor |
|---|---|
| `FunctionAgent` / `ReActAgent` + `Context(agent)` | `ToolLoop(tools=[...])` (stateless) or `Chat(tools=[...])` (stateful) |
| `AgentStream` / `ToolCallResult` events | `ChatEvent` from `ask_stream()` / `run_events()` |
| `index.as_chat_engine(chat_mode=..., memory=...)` / `ChatSummaryMemoryBuffer` | `Chat(history="full"|"window"|"summary")` — inspect `chat.messages` vs `chat.last_context` |
| `QueryEngineTool.from_defaults(...)` + `ToolMetadata` | `@tool` on a plain function; `make_retrieval_tool(col)` |
| `TavilyToolSpec` / `GoogleSearchToolSpec` + `LoadAndSearchToolSpec` | `search_web(q)` (+ `tool(search_web)`); Google CSE variant archived |
| `RouterQueryEngine` + `LLMSingleSelector`/`PydanticSingleSelector` | `route(q, routes)` + your `if/else` |
| `Workflow` / `@step` / `StartEvent` / `StopEvent` | plain functions and loops |

## Fine-tuning

| LlamaIndex | tai_aitutor |
|---|---|
| `EmbeddingAdapterFinetuneEngine.finetune()` + `get_finetuned_model()` / `AdapterEmbeddingModel` | `train_embedder(train, val, base_model="BAAI/bge-small-en-v1.5")` — sentence-transformers + MNRL (method upgrade per the course plan) |
| `generate_qa_embedding_pairs` (legacy) | `make_training_pairs(chunks)` |
| before/after measurement | `evaluate_embedder(model_path, qa)` — same Hit Rate / MRR ruler |

## Course data

| Old (scattered across notebooks) | tai_aitutor |
|---|---|
| `wget .../AlaFalaki/.../mini-llama-articles.csv` | `mini_articles()` / `mini_articles(with_embeddings=True)` |
| `hf_hub_download("jaiganesan/ai_tutor_knowledge", "ai_tutor_knowledge.jsonl")` | `ai_tutor_knowledge()` |
| `hf_hub_download(..., "vectorstore.zip")` + unzip | `prebuilt_chroma()` (⚠ embedded with OpenAI `text-embedding-3-small` — match your query embedder) |
| `hf_hub_download(..., "rag_eval_dataset_question_context_subset_50.json")` | `qa_dataset("rag_eval_50")` |
| research-paper zips | `research_papers(parsed=False)` |

When the data moves to the Towards AI org account (Decision 6), only the registry block at
the top of `src/tai_aitutor/datasets.py` changes.
