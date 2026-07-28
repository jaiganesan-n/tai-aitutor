"""Loading source material into ``Document`` objects.

Built in: "Building Your Knowledge Base for RAG" (Section 3) and the data
lessons of Section 6. Replaces LlamaIndex's ``Document`` /
``llama_index.core.schema.Document``, ``SimpleDirectoryReader``, and
``WikipediaReader`` — with plain loaders you can read in one screen.

A ``Document`` is just text + metadata. Chunking (next lesson) turns
documents into ``Chunk`` objects; ingestion embeds and stores them.

Safety note carried from the course bug sweep: embeddings stored in CSV are
parsed with ``json.loads`` — never ``eval()`` (the old Basic RAG notebook
ran ``eval()`` on CSV strings, which executes arbitrary code).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ProviderNotInstalledError, TaiAitutorError

__all__ = [
    "Document",
    "load_csv",
    "load_jsonl",
    "load_directory",
    "load_files",
    "load_wikipedia",
    "load_hf_dataset",
]


@dataclass
class Document:
    """One source document: raw text plus whatever metadata should travel with it."""

    text: str
    metadata: dict = field(default_factory=dict)
    id: str | None = None

    def stable_id(self) -> str:
        """The document's id, or a short deterministic hash of its text."""
        if self.id:
            return self.id
        return hashlib.sha1(self.text.encode("utf-8", "ignore")).hexdigest()[:12]

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Document(id={self.stable_id()!r}, {len(self.text)} chars, {preview!r}...)"


# --------------------------------------------------------------------------- #
# Reading helpers
# --------------------------------------------------------------------------- #


def _read_text(path_or_url: str | Path) -> str:
    """Read a local file or an http(s) URL as UTF-8 text."""
    s = str(path_or_url)
    if s.startswith(("http://", "https://")):
        with urllib.request.urlopen(s) as resp:  # noqa: S310 — course data URLs
            return resp.read().decode("utf-8", "replace")
    return Path(s).read_text(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# CSV / JSONL
# --------------------------------------------------------------------------- #


def load_csv(
    path_or_url: str | Path,
    text_col: str,
    meta_cols: tuple[str, ...] | list[str] = (),
    embedding_col: str | None = None,
    id_col: str | None = None,
    id_max_chars: int = 40,
) -> list[Document]:
    """Load a CSV (local path or URL) into Documents.

    ``text_col`` names the column holding the document text; ``meta_cols`` are
    copied into ``Document.metadata``. If ``embedding_col`` is given, that
    column is parsed with ``json.loads`` (a JSON array of floats) and stored as
    ``metadata["embedding"]`` — the precomputed-embeddings checkpoint pattern
    from the Basic RAG lesson, without the old ``eval()`` hazard.

    ``id_col`` makes document ids READABLE: ``Document.id`` becomes that
    column's value (truncated to ``id_max_chars``, the course convention), so
    chunk ids read ``"Beyond GPT-4-0000"`` instead of a hash. Duplicate values
    get a ``~2``, ``~3``… suffix so chunk ids can never collide across
    documents. Stable, readable ids are what keep saved eval datasets and
    existing collections lined up across re-ingests.
    """
    csv.field_size_limit(sys.maxsize)  # course articles exceed the default field cap
    raw = _read_text(path_or_url)
    reader = csv.DictReader(io.StringIO(raw))
    if reader.fieldnames is None or text_col not in reader.fieldnames:
        raise TaiAitutorError(
            f"CSV has no {text_col!r} column. Found columns: {reader.fieldnames}"
        )
    if id_col is not None and id_col not in reader.fieldnames:
        raise TaiAitutorError(
            f"CSV has no {id_col!r} column (id_col). Found columns: {reader.fieldnames}"
        )

    docs: list[Document] = []
    seen_ids: dict[str, int] = {}
    for row in reader:
        text = (row.get(text_col) or "").strip()
        if not text:
            continue
        metadata = {c: row[c] for c in meta_cols if row.get(c) not in (None, "")}
        if embedding_col:
            raw_vec = row.get(embedding_col) or ""
            try:
                metadata["embedding"] = json.loads(raw_vec)
            except json.JSONDecodeError as exc:
                raise TaiAitutorError(
                    f"Column {embedding_col!r} is not valid JSON. Store embeddings as JSON "
                    "arrays (json.dumps) — never rely on eval()-style Python literals."
                ) from exc
        doc_id = None
        if id_col is not None:
            doc_id = str(row.get(id_col) or "").strip()[:id_max_chars] or None
            if doc_id is not None:
                seen_ids[doc_id] = seen_ids.get(doc_id, 0) + 1
                if seen_ids[doc_id] > 1:
                    doc_id = f"{doc_id}~{seen_ids[doc_id]}"
        docs.append(Document(text=text, metadata=metadata, id=doc_id))
    return docs


def load_jsonl(
    path_or_url: str | Path,
    text_key: str = "content",
    id_key: str = "id",
) -> list[Document]:
    """Load a JSON-Lines file (one object per line) into Documents.

    ``text_key`` becomes the document text; every other field goes to metadata.
    """
    docs: list[Document] = []
    for line_no, line in enumerate(_read_text(path_or_url).splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TaiAitutorError(f"Invalid JSON on line {line_no}: {line[:80]!r}") from exc
        text = str(obj.get(text_key) or "").strip()
        if not text:
            continue
        metadata = {k: v for k, v in obj.items() if k not in (text_key, id_key)}
        docs.append(Document(text=text, metadata=metadata, id=obj.get(id_key)))
    return docs


# --------------------------------------------------------------------------- #
# Files and directories (replaces SimpleDirectoryReader)
# --------------------------------------------------------------------------- #

_TEXT_EXTS = {".txt", ".md", ".markdown", ".rst", ".html", ".py", ".json", ".csv"}


def load_files(paths: list[str | Path]) -> list[Document]:
    """Load specific files into Documents (text-like files plus .pdf via pypdf)."""
    docs: list[Document] = []
    for p in map(Path, paths):
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            text = _pdf_text(p)
        elif suffix in _TEXT_EXTS:
            text = p.read_text(encoding="utf-8", errors="replace")
        else:
            continue  # silently skip binary/unknown types, like the lesson does
        if text.strip():
            docs.append(
                Document(text=text, metadata={"file_name": p.name, "file_path": str(p)})
            )
    return docs


def load_directory(
    path: str | Path,
    exts: tuple[str, ...] = (".txt", ".md", ".pdf"),
    recursive: bool = True,
) -> list[Document]:
    """Load every matching file under a directory (replaces ``SimpleDirectoryReader``)."""
    root = Path(path)
    if not root.is_dir():
        raise TaiAitutorError(f"{root} is not a directory.")
    pattern = "**/*" if recursive else "*"
    files = sorted(p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in exts)
    return load_files(files)


def _pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ProviderNotInstalledError(
            "pypdf is not installed (needed for .pdf files). "
            "Run: pip install 'tai-aitutor[parse]'"
        ) from exc
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


# --------------------------------------------------------------------------- #
# Wikipedia (replaces WikipediaReader) and HF datasets
# --------------------------------------------------------------------------- #


def load_wikipedia(titles: list[str], lang: str = "en") -> list[Document]:
    """Fetch Wikipedia pages as Documents (router lesson's knowledge source)."""
    try:
        import wikipedia
    except ImportError as exc:
        raise ProviderNotInstalledError(
            "The wikipedia package is not installed. Run: pip install 'tai-aitutor[web]'"
        ) from exc
    wikipedia.set_lang(lang)
    docs = []
    for title in titles:
        page = wikipedia.page(title, auto_suggest=False)
        docs.append(
            Document(
                text=page.content,
                metadata={"title": page.title, "url": page.url, "source": "wikipedia"},
                id=f"wikipedia-{page.pageid}",
            )
        )
    return docs


def load_hf_dataset(
    repo_id: str,
    filename: str,
    text_key: str = "content",
    repo_type: str = "dataset",
) -> list[Document]:
    """Download one file from a Hugging Face dataset repo and load it.

    Dispatches on extension: ``.jsonl`` → :func:`load_jsonl`; ``.csv`` needs an
    explicit follow-up call to :func:`load_csv` (column names vary), so it
    returns the downloaded path inside the raised message instead of guessing.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ProviderNotInstalledError(
            "huggingface-hub is not installed. Run: pip install 'tai-aitutor[data]'"
        ) from exc
    local = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
    if filename.lower().endswith(".jsonl"):
        return load_jsonl(local, text_key=text_key)
    if filename.lower().endswith(".json"):
        data = json.loads(Path(local).read_text(encoding="utf-8"))
        return [
            Document(
                text=str(obj.get(text_key, "")),
                metadata={k: v for k, v in obj.items() if k != text_key},
            )
            for obj in data
            if str(obj.get(text_key, "")).strip()
        ]
    raise TaiAitutorError(
        f"Downloaded {local} — load CSVs explicitly with "
        f"load_csv({local!r}, text_col=..., meta_cols=...) so column names are visible."
    )
