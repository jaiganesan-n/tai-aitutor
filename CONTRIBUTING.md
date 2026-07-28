# Contributing to tai-aitutor

This package is the course's own code, packaged. Two documents govern it:
[`docs/PACKAGE_PLAN.md`](docs/PACKAGE_PLAN.md) (the design + LlamaIndex inventory) and
[`BUILD_STATUS.md`](BUILD_STATUS.md) (what's built, what's next).

## The one rule

**Teach-then-import.** The package only contains code some lesson writes out first, and
every module docstring names that lesson. If a PR adds a public symbol no lesson teaches,
it doesn't merge. Corollaries: small flat modules (readable in one sitting), functions
over object graphs, no hidden state beyond `configure()`, loud failures with fix-it
messages.

## Dev setup

Python **>= 3.13**.

```bash
pip install -e ".[gemini,openai,anthropic,rag,rerank,parse]" pytest ruff
pytest              # fully offline — provider SDKs are faked, Chroma runs ephemeral
ruff check src tests
```

Rules the suite enforces (don't break them): provider kwargs are explicit (typos raise
`TypeError`), data parsing never uses `eval()`, requested caps (`top_k`, `keep`) are
applied, `judge_rerank` preserves the judge's ordering, `QADataset` stays byte-compatible
with the legacy `rag_eval_dataset*.json` shape.

## Where things live

- Model names, prices ("as of" dated), base URLs, API-key env vars: ONE table each in
  `config.py`.
- Dataset URLs: ONE registry block in `datasets.py` (the org-account migration is a
  one-block edit).
- Production constants (chunk sizes, BM25 k1/b, RRF k, rerank model/floor, token budget):
  mirrored from the live tutor; change them only together with production.

## Releasing

1. Bump `__version__` in `src/tai_aitutor/__init__.py` + add a `CHANGELOG.md` entry.
2. `pytest && ruff check src tests && pipx run build && pipx run twine check dist/*`
3. `git tag vX.Y.Z && git push --tags` — `release.yml` publishes to PyPI via Trusted
   Publishing (one-time setup in its header comment).
4. Course notebooks pin exact versions; bumping them is a scripted sweep, one pin cell per
   notebook.
