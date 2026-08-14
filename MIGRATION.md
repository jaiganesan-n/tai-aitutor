# Migrating from LlamaIndex to `tai_aitutor`

The migration guide lives in the [README](README.md#migration-map-llamaindex--tai_aitutor)
— it pairs the two libraries symbol by symbol.

Quick reminders while porting:

- Delete every `nest_asyncio.apply()` call, every `llama-index-*` pip pin, and
  `LLAMA_CLOUD_API_KEY` setup.
- The package ships no dataset URLs and no downloaders. Download the file however you
  like, then hand it to `load_csv` / `QADataset.load`.
- `QADataset.load` reads LlamaIndex's `EmbeddingQAFinetuneDataset` JSON unchanged, so
  existing eval sets carry over as-is.
