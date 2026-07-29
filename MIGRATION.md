# Migrating from LlamaIndex to `tai_aitutor`

The migration guide now lives in the [README](README.md#migration-map-llamaindex--tai_aitutor)
— the README *is* the migration doc, symbol by symbol.

Quick reminders while porting a notebook:

- Delete every `nest_asyncio.apply()` cell, every `llama-index-*` pip pin, and
  `LLAMA_CLOUD_API_KEY` setup.
- Data downloads stay in the notebook (wget / `hf_hub_download`) — the package ships no
  dataset URLs by design; hand the downloaded file to `load_csv` / `load_jsonl` /
  `QADataset.load`.
- Full design rationale + the complete 82-symbol LlamaIndex inventory:
  [`docs/PACKAGE_PLAN.md`](docs/PACKAGE_PLAN.md).
