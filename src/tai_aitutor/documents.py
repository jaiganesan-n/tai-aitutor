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

from .errors import TaiAitutorError

__all__ = [
    "Document",
    "load_csv",
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

    Args:
        path_or_url: Local path or ``http(s)://`` URL of the CSV.
        text_col: Column holding the document body.
        meta_cols: Columns to carry through as metadata.
        embedding_col: Column holding a JSON list of floats, if the CSV is a
            pre-embedded checkpoint. Parsed with ``json.loads``, never ``eval``.
        id_col: Column to derive the document id from; falls back to a stable
            hash of the text.
        id_max_chars: Cap on a derived id's length.

    Returns:
        One :class:`Document` per row, in file order.

    Raises:
        ValueError: ``text_col`` is not a column in the file.
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


# --------------------------------------------------------------------------- #
# Files and directories (replaces SimpleDirectoryReader)
# --------------------------------------------------------------------------- #

_TEXT_EXTS = {".txt", ".md", ".markdown", ".rst", ".html", ".py", ".json", ".csv"}


# --------------------------------------------------------------------------- #
# Wikipedia (replaces WikipediaReader) and HF datasets
# --------------------------------------------------------------------------- #


