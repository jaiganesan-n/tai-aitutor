# tai-aitutor — Build Status

*The living tracker for the package build. Full design + LlamaIndex inventory: [`docs/PACKAGE_PLAN.md`](docs/PACKAGE_PLAN.md). Updated: **2026-07-28 (v1.0.0 — build complete)**.*

## Why this package exists (one paragraph)

The July 2026 course update removes LlamaIndex from the whole course and teaches every concept as
a direct implementation (Course Update Plan v2, Decision 1). The first ~20 ported notebooks proved
the approach by copy-pasting the same helpers (`generate`, `embed`, `chunk`, `search`, `answer`,
judges, metrics) into each notebook. `tai-aitutor` is those helpers as one package we own — same
names, same signatures — so remaining ports go faster, ported notebooks lose their duplicated
cells, and maintenance becomes "bump one pin." Rule that governs everything: **a lesson builds the
concept inline first; later notebooks import it from here.** The package is never allowed to be
"our LlamaIndex."

## Phase tracker

| Phase | Scope | Version | Status |
|---|---|---|---|
| 0 | Scaffold + `config` (Settings replacement, Colab/local setup cell) + `llm` (`generate`/`extract`/vision/stream/batch) + `embeddings` + `tokens` | v0.1.0 | ✅ done |
| 1 | Pipeline core: `documents`, `chunking` (incl. production heading-aware + sentence-window), `vectorstore` (Chroma, `ingest`, `build_where_filter`), `retrieval.search`, `synthesis` (`answer` family), `display` | v0.2.0 | ✅ done |
| 2 | `evals` — `QADataset` (legacy-JSON compatible with `EmbeddingQAFinetuneDataset` files), `make_qa_pairs`, `hit_rate`, `mrr`, `evaluate_retrieval`, typed judges (faithfulness / relevancy / correctness), `run_judges`; plus `extractors` (keywords / summary / questions / `situate_chunk`) | v0.3.0 | ✅ done |
| 3 | Advanced retrieval — `BM25Index` (k1=1.5, b=0.75, code-aware tokenizer, save/load), `rrf_fuse` (k=60), `hybrid_search`, `rerank` (Cohere v4, top 5, floor 0.10), `judge_rerank`, `hyde_search`, `decompose_question`, `subquestion_answer`, `multi_step_answer`, `pack_context` | v0.4.0 | ✅ done |
| 4 | `router` (typed route + dispatch), `chat` (`Chat`, `ToolLoop`, full/window/summary memory), `tools` (`tool()`, `make_retrieval_tool`, `search_web`), `finetune` (sentence-transformers MNRL), `llm.chat_completion` (messages+tools layer, all providers) | v0.5.0 | ✅ done |
| 5 | `datasets` (verified registry over today's hosts; org migration = one-block edit), `MIGRATION.md`, `examples/` (incl. the Gradio tutor rebuild reference), `CONTRIBUTING.md`, release machinery | v1.0.0 | ✅ done |

## What's built (v1.0.0 — complete)

- **109 public symbols across 19 modules** — the full replacement surface for the 82
  LlamaIndex symbols in the course inventory (`docs/PACKAGE_PLAN.md`, Appendix B), plus
  the course-data layer: pipeline (`ingest → search/hybrid/rerank → answer`), measurement
  (`make_qa_pairs → evaluate_retrieval → run_judges`), agents (`chat_completion` →
  `ToolLoop`/`Chat`, `route`, tools), fine-tuning, and `datasets` (registry verified
  against the live notebooks' own download calls).
- **154 tests**, all offline; `ruff` clean; `twine check` passes; verified on **Python
  3.13** (wheel `Requires-Python: >=3.13`, PEP 695 generics).
- One transcript format, four provider families, tested per branch; all historical-bug
  regressions locked in (kwarg typos, `eval()`, ignored caps, discarded judge ordering);
  `QADataset` byte-compatible with legacy eval files; `Chat.messages` vs
  `Chat.last_context` keeps the memory lesson inspectable.
- Docs complete: `README` (with the production-chain and agent examples), `MIGRATION.md`,
  `CONTRIBUTING.md`, `examples/` (quickstart, production retriever + four-row ablation,
  Gradio tutor rebuild), `docs/PACKAGE_PLAN.md`, this tracker.
- PyPI name **`tai-aitutor` unclaimed** (checked 2026-07-28); `dist/` holds the built,
  twine-checked 1.0.0 wheel + sdist — publishing is the runbook below.

## Working agreements

- Providers: Gemini default (`gemini-2.5-flash` / `gemini-embedding-001`), OpenAI (`gpt-5-mini` /
  `text-embedding-3-small`), Anthropic (`claude-sonnet-4-6`, embeddings via Gemini or OpenAI),
  plus OpenAI-compatible Together / DeepSeek / Perplexity / Ollama via `base_url`.
- Model names + prices live in ONE table (`config.py`), dated, verified each course release.
- **Python: >=3.13** (decided 2026-07-28). CI matrix: 3.13, 3.14, and 3.15-dev
  (non-blocking until 3.15 final ships ~Oct 2026); wheels build on 3.13.
- Notebooks pin exact versions. Dev installs use git tags; PyPI from launch (Trusted Publishing
  is set up in `.github/workflows/release.yml` — see its header comment for the one-time setup).
- `.github/workflows/` note: the CI files are also mirrored in `github-workflows-to-move/`
  because the desktop bridge can't write into `.github/` — keep them in sync (or delete the
  mirror folder once the real ones are committed).

## How to work on it

```bash
pip install -e ".[gemini,openai,anthropic,rag,rerank,parse]" pytest ruff
pytest                      # fully offline
ruff check src tests
# release: bump __version__ + CHANGELOG.md → git tag v0.X.0 → push --tags
```

## Release-day runbook (the remaining human steps)

The code is complete; these steps need repo/account access:

1. **Git + GitHub.** In the repo folder: move `github-workflows-to-move/` →
   `.github/workflows/`, then `git init && git add -A && git commit`, create
   `towardsai/tai-aitutor` on GitHub, push. CI runs on 3.13/3.14 (+3.15-dev).
2. **PyPI Trusted Publishing** (one-time, no tokens): pypi.org → Publishing → add pending
   publisher (owner `towardsai`, repo `tai-aitutor`, workflow `release.yml`, environment
   `pypi`); GitHub repo → Settings → Environments → create `pypi`.
3. **Publish:** `git tag v1.0.0 && git push --tags`. (Name verified unclaimed 2026-07-28;
   the built + twine-checked artifacts are also in `dist/` if a manual
   `twine upload dist/*` is ever preferred.)
4. **Notebook sweep:** switch install cells to `pip install "tai-aitutor[...]==1.0.0"`
   (profiles A/B from the plan) and replace the remaining per-notebook helper cells with
   imports — `MIGRATION.md` is the cheat-sheet.
5. **Data migration (Decision 6), when the org accounts exist:** upload the datasets +
   the Gemini-re-embedded vector store to the Towards AI org, then edit the ONE registry
   block at the top of `src/tai_aitutor/datasets.py` (`HF_ORG_REPO`, `GITHUB_DATA_BASE`,
   enable the `"gemini"` vector-store variant) and release 1.1.0.
6. **Gradio repo:** port `ai-tutor-gradio-lesson/app.py` from
   `examples/gradio_tutor.py` (same shape), then fork it into the Section 16
   certification skeleton.
7. **Post-launch:** notebook CI (papermill matrix, Colab-sim + local, spend-capped) per
   the plan's Section 8, so provider breakage is caught nightly — not by students.
