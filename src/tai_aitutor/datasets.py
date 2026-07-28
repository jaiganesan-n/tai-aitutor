"""Course datasets: one home for every download the notebooks make.

Built for: Decision 6 of the course update — course data moves off personal
accounts onto Towards AI org accounts, versioned. Until that migration lands,
this module points at the CURRENT hosts (verbatim the URLs the notebooks use
today), so it works now and the org move becomes a one-block edit below —
instead of a sweep across 16+ notebooks.

Everything caches under ``~/.cache/tai-aitutor`` (override with the
``TAI_AITUTOR_CACHE`` env var) and is safe to call repeatedly.
"""

from __future__ import annotations

import os
import urllib.request
import zipfile
from pathlib import Path

from .documents import Document, load_csv, load_jsonl
from .errors import ProviderNotInstalledError, TaiAitutorError
from .evals import QADataset

__all__ = [
    "mini_articles",
    "ai_tutor_knowledge",
    "prebuilt_chroma",
    "qa_dataset",
    "research_papers",
    "cache_dir",
]

# --------------------------------------------------------------------------- #
# THE registry — the only block that changes when data moves to the org account
# (Decision 6). Filenames below are verified against the live course notebooks.
# --------------------------------------------------------------------------- #

#: TODO(course-launch): flip both to the Towards AI org accounts and re-verify.
HF_ORG_REPO = "jaiganesan/ai_tutor_knowledge"
GITHUB_DATA_BASE = "https://raw.githubusercontent.com/AlaFalaki/tutorial_notebooks/main/data"

_FILES = {
    "mini_articles": f"{GITHUB_DATA_BASE}/mini-llama-articles.csv",
    "mini_articles_embedded": f"{GITHUB_DATA_BASE}/mini-llama-articles-with_embeddings.csv",
    "kb_jsonl": (HF_ORG_REPO, "ai_tutor_knowledge.jsonl"),
    "vectorstore": (HF_ORG_REPO, "vectorstore.zip"),           # OpenAI text-embedding-3-small
    "vectorstore_windowed": (HF_ORG_REPO, "vectorstore-windowed.zip"),
    "qa_rag_eval_50": (HF_ORG_REPO, "rag_eval_dataset_question_context_subset_50.json"),
    "papers_raw": (HF_ORG_REPO, "rag_research_paper.zip"),
    "papers_parsed": (HF_ORG_REPO, "research_papers_llamaparse.zip"),
}

#: Which embedding model each prebuilt store was built with. The Gemini-default
#: re-embed (course plan, Decision 6 coupling) gets an entry here once hosted.
_VECTORSTORE_VARIANTS = {
    "default": ("vectorstore", "openai", "text-embedding-3-small"),
    "windowed": ("vectorstore_windowed", "openai", "text-embedding-3-small"),
    # "gemini": ("vectorstore_gemini", "gemini", "gemini-embedding-001"),  # pending re-embed + upload
}


def cache_dir() -> Path:
    """Where downloads land (override with the TAI_AITUTOR_CACHE env var)."""
    root = os.environ.get("TAI_AITUTOR_CACHE") or str(Path.home() / ".cache" / "tai-aitutor")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Download plumbing (cached, boring on purpose)
# --------------------------------------------------------------------------- #


def _fetch_url(url: str) -> Path:
    target = cache_dir() / url.rsplit("/", 1)[-1]
    if not target.exists():
        with urllib.request.urlopen(url) as resp:  # noqa: S310 — course data URLs
            target.write_bytes(resp.read())
    return target


def _fetch_hf(repo_id: str, filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ProviderNotInstalledError(
            "huggingface-hub is not installed. Run: pip install 'tai-aitutor[data]'"
        ) from exc
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(cache_dir() / repo_id.replace("/", "__")),
        )
    )


def _fetch(key: str) -> Path:
    spec = _FILES[key]
    if isinstance(spec, str):
        return _fetch_url(spec)
    repo_id, filename = spec
    return _fetch_hf(repo_id, filename)


def _extract_zip(archive: Path) -> Path:
    """Unzip next to the archive (once); return the extraction directory."""
    out = archive.with_suffix("")  # vectorstore.zip → vectorstore/
    if not out.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out)
    return out


# --------------------------------------------------------------------------- #
# The datasets
# --------------------------------------------------------------------------- #


def mini_articles(with_embeddings: bool = False) -> list[Document]:
    """The mini AI-articles set used from Section 3 onward.

    ``with_embeddings=True`` loads the precomputed-embeddings checkpoint
    (vectors land in ``doc.metadata["embedding"]``, parsed with ``json.loads``
    — the Basic RAG lesson's checkpoint, without the old ``eval()`` hazard).
    """
    key = "mini_articles_embedded" if with_embeddings else "mini_articles"
    path = _fetch(key)
    return load_csv(
        path,
        text_col="content",
        meta_cols=("title", "url", "source_name"),
        embedding_col="embedding" if with_embeddings else None,
    )


def ai_tutor_knowledge() -> list[Document]:
    """The full AI-tutor knowledge base (one JSONL document per source page)."""
    return load_jsonl(_fetch("kb_jsonl"), text_key="content")


def prebuilt_chroma(variant: str = "default") -> Path:
    """Download + extract a prebuilt Chroma store; returns the path for
    ``get_collection(name, path=str(...))``.

    Variants: ``"default"`` (the course store), ``"windowed"`` (sentence-window
    chunks for the Advanced Retrieval lesson). BOTH were embedded with OpenAI
    ``text-embedding-3-small`` — query them with
    ``configure(embed_provider="openai")`` or pass a matching ``embed_fn``;
    mixing embedders across index and query time silently ruins retrieval,
    which is exactly why this docstring shouts about it. A Gemini-embedded
    store ships with the org data migration (see ``_VECTORSTORE_VARIANTS``).
    """
    if variant not in _VECTORSTORE_VARIANTS:
        raise TaiAitutorError(
            f"Unknown vector store variant {variant!r}. "
            f"Available: {sorted(_VECTORSTORE_VARIANTS)} "
            "(the gemini-embedded store ships with the org data migration)."
        )
    key = _VECTORSTORE_VARIANTS[variant][0]
    path = _extract_zip(_fetch(key))
    # If the zip wraps a single top-level folder, return that folder.
    entries = [p for p in path.iterdir() if not p.name.startswith("__MACOSX")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return path


def qa_dataset(name: str = "rag_eval_50") -> QADataset:
    """A hosted QA eval set (legacy-compatible JSON) by name.

    ``"rag_eval_50"`` is the 50-question subset the retrieval lessons share, so
    every notebook's Hit Rate / MRR table is computed on the same questions.
    """
    registry = {"rag_eval_50": "qa_rag_eval_50"}
    if name not in registry:
        raise TaiAitutorError(
            f"Unknown QA dataset {name!r}. Available: {sorted(registry)}."
        )
    return QADataset.load(_fetch(registry[name]))


def research_papers(parsed: bool = False) -> Path:
    """The research-papers PDF set for the parsing lessons (extracted directory).

    ``parsed=True`` fetches the pre-parsed variant so lessons can skip ahead.
    """
    return _extract_zip(_fetch("papers_parsed" if parsed else "papers_raw"))
