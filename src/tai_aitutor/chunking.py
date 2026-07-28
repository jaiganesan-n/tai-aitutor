"""Chunking: turning documents into retrievable pieces.

Built in: "From Script to Pipeline: Chunking and Reusable Retrieval Functions"
(Section 4). Replaces LlamaIndex's ``TokenTextSplitter``, ``SentenceSplitter``,
``SimpleNodeParser``, and ``SentenceWindowNodeParser`` — as four visible
functions instead of parser objects.

Course defaults: 512-token chunks with 128 overlap. Production defaults
(:func:`heading_aware_markdown_chunks`): 800-token chunks, 100 overlap, and
code blocks are NEVER split — the same rules as the live tutor's ingest
chunker in ``app/chroma_rag.py``, at notebook size.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .documents import Document
from .errors import TaiAitutorError
from .tokens import _get_encoding, n_tokens

__all__ = [
    "Chunk",
    "chunk",
    "chunk_document",
    "chunk_sentences",
    "heading_aware_markdown_chunks",
    "sentence_window_chunks",
]


@dataclass
class Chunk:
    """One retrievable piece of a document (LlamaIndex called this a Node/TextNode)."""

    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: list[float] | None = None

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(id={self.id!r}, {n_tokens(self.text)} tokens, {preview!r}...)"


def _check_sizes(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise TaiAitutorError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise TaiAitutorError(
            f"chunk_overlap must satisfy 0 <= overlap < chunk_size, "
            f"got overlap={chunk_overlap}, chunk_size={chunk_size}"
        )


# --------------------------------------------------------------------------- #
# Fixed-size token chunks (the course workhorse)
# --------------------------------------------------------------------------- #


def chunk(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    separator: str = " ",
) -> list[str]:
    """Split ``text`` into ~``chunk_size``-token windows overlapping by ``chunk_overlap``.

    Replaces ``TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)``.
    Offline fallback (no tokenizer vocabulary): approximate windows of
    ``chunk_size * 4`` characters split on ``separator``.
    """
    _check_sizes(chunk_size, chunk_overlap)
    if not text or not text.strip():
        return []

    enc = _get_encoding()
    step = chunk_size - chunk_overlap

    if enc is None:  # offline fallback: ~4 chars per token, respect separators
        words = text.split(separator)
        out, size = [], chunk_size * 4
        cur: list[str] = []
        cur_len = 0
        for word in words:
            cur.append(word)
            cur_len += len(word) + 1
            if cur_len >= size:
                out.append(separator.join(cur).strip())
                # keep a tail as overlap
                tail_len = 0
                tail: list[str] = []
                for w in reversed(cur):
                    tail_len += len(w) + 1
                    if tail_len > chunk_overlap * 4:
                        break
                    tail.insert(0, w)
                cur, cur_len = tail, sum(len(w) + 1 for w in tail)
        if cur and separator.join(cur).strip():
            out.append(separator.join(cur).strip())
        return out

    ids = enc.encode(text)
    out = []
    for start in range(0, len(ids), step):
        piece = enc.decode(ids[start : start + chunk_size]).strip()
        if piece:
            out.append(piece)
        if start + chunk_size >= len(ids):
            break
    return out


def chunk_document(
    doc: Document | str,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    chunker=None,
) -> list[Chunk]:
    """Chunk one Document into ``Chunk`` objects with stable ids and inherited metadata.

    ``chunker`` overrides the splitter (any ``text -> list[str]`` callable, e.g.
    ``heading_aware_markdown_chunks``); default is :func:`chunk` with the sizes given.
    """
    if isinstance(doc, str):
        doc = Document(text=doc)
    pieces = chunker(doc.text) if chunker else chunk(doc.text, chunk_size, chunk_overlap)
    base = doc.stable_id()
    return [
        Chunk(id=f"{base}-{i:04d}", text=piece, metadata=dict(doc.metadata))
        for i, piece in enumerate(pieces)
    ]


# --------------------------------------------------------------------------- #
# Sentence-aware chunks
# --------------------------------------------------------------------------- #

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


def chunk_sentences(text: str, chunk_size: int = 512, chunk_overlap: int = 128) -> list[str]:
    """Pack whole sentences into ~``chunk_size``-token chunks (replaces ``SentenceSplitter``).

    Sentences are never cut mid-way; overlap carries trailing sentences of the
    previous chunk (up to ``chunk_overlap`` tokens) into the next one.
    """
    _check_sizes(chunk_size, chunk_overlap)
    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for sentence in _sentences(text):
        s_tokens = n_tokens(sentence) + 1
        if cur and cur_tokens + s_tokens > chunk_size:
            chunks.append(" ".join(cur))
            tail: list[str] = []
            tail_tokens = 0
            for prev in reversed(cur):
                p_tokens = n_tokens(prev) + 1
                if tail_tokens + p_tokens > chunk_overlap:
                    break
                tail.insert(0, prev)
                tail_tokens += p_tokens
            cur, cur_tokens = tail, tail_tokens
        cur.append(sentence)
        cur_tokens += s_tokens
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# --------------------------------------------------------------------------- #
# Heading-aware markdown chunks (the production chunker, notebook-sized)
# --------------------------------------------------------------------------- #

_HEADING_LINE = re.compile(r"^#{1,6}\s+\S")
_FENCE_LINE = re.compile(r"^(```|~~~)")


def _markdown_blocks(markdown: str) -> list[tuple[str, str]]:
    """Parse markdown into ('heading' | 'text' | 'code', text) blocks.

    Fenced code blocks (``` or ~~~) are single atomic blocks, fences included.
    """
    blocks: list[tuple[str, str]] = []
    paragraph: list[str] = []
    code: list[str] = []
    in_code = False
    fence = ""

    def flush_paragraph() -> None:
        text = "\n".join(paragraph).strip("\n")
        if text.strip():
            blocks.append(("text", text))
        paragraph.clear()

    for line in markdown.split("\n"):
        stripped = line.strip()
        if in_code:
            code.append(line)
            if stripped.startswith(fence):
                blocks.append(("code", "\n".join(code)))
                code.clear()
                in_code = False
            continue
        fence_match = _FENCE_LINE.match(stripped)
        if fence_match:
            flush_paragraph()
            in_code = True
            fence = fence_match.group(1)
            code.append(line)
            continue
        if _HEADING_LINE.match(stripped):
            flush_paragraph()
            blocks.append(("heading", stripped))
            continue
        if not stripped:
            flush_paragraph()
            continue
        paragraph.append(line)

    if in_code:  # unclosed fence: keep it atomic anyway
        blocks.append(("code", "\n".join(code)))
    flush_paragraph()
    return blocks


def _tail_tokens(text: str, max_tokens: int) -> str:
    """The last ~``max_tokens`` tokens of ``text`` (overlap carry)."""
    if max_tokens <= 0:
        return ""
    enc = _get_encoding()
    if enc is None:
        return text[-max_tokens * 4 :]
    ids = enc.encode(text)
    return enc.decode(ids[-max_tokens:]) if len(ids) > max_tokens else text


def heading_aware_markdown_chunks(
    markdown: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[str]:
    """The production chunker, at notebook size (defaults: 800 tokens, 100 overlap).

    Rules (same as the live tutor's ingest chunking):

    1. Fenced code blocks are atomic — a code block is never split, even when
       it alone exceeds ``chunk_size`` (that chunk is allowed to run large).
    2. Chunks remember their section: when a section's content continues into
       a new chunk, the chunk starts with the current heading line again.
    3. ``chunk_overlap`` tokens of trailing prose (never code) carry over into
       the next chunk for continuity.
    4. Oversized prose paragraphs fall back to plain token windows.
    """
    _check_sizes(chunk_size, chunk_overlap)
    blocks = _markdown_blocks(markdown)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    heading: str | None = None
    last_prose: str = ""

    def flush() -> None:
        nonlocal current, current_tokens
        text = "\n\n".join(current).strip()
        if text:
            chunks.append(text)
        current, current_tokens = [], 0

    def start_new(carry_overlap: bool) -> None:
        """Begin a chunk with the section heading and optional prose overlap."""
        nonlocal current, current_tokens
        if heading is not None:
            current.append(heading)
            current_tokens += n_tokens(heading) + 1
        if carry_overlap and last_prose:
            tail = _tail_tokens(last_prose, chunk_overlap).strip()
            if tail:
                current.append(tail)
                current_tokens += n_tokens(tail) + 1

    for kind, text in blocks:
        block_tokens = n_tokens(text) + 1

        if kind == "heading":
            heading = text
            if current and current_tokens + block_tokens > chunk_size:
                flush()
            current.append(text)
            current_tokens += block_tokens
            continue

        if kind == "code" and block_tokens > chunk_size:
            # Rule 1: atomic even when oversized — flush, emit code with its heading.
            flush()
            piece = f"{heading}\n\n{text}" if heading else text
            chunks.append(piece)
            continue

        if kind == "text" and block_tokens > chunk_size:
            # Rule 4: giant paragraph → plain token windows within this section.
            flush()
            for piece in chunk(text, chunk_size, chunk_overlap):
                chunks.append(f"{heading}\n\n{piece}" if heading else piece)
            last_prose = text
            continue

        if current and current_tokens + block_tokens > chunk_size:
            flush()
            start_new(carry_overlap=(kind == "text"))

        current.append(text)
        current_tokens += block_tokens
        if kind == "text":
            last_prose = text

    flush()
    return chunks


# --------------------------------------------------------------------------- #
# Sentence-window chunks (small-to-big retrieval, Advanced Retrieval lesson)
# --------------------------------------------------------------------------- #


def sentence_window_chunks(
    text: str,
    window_size: int = 3,
    doc_id: str | None = None,
) -> list[Chunk]:
    """One Chunk per sentence, with its neighborhood stored in ``metadata["window"]``.

    Replaces ``SentenceWindowNodeParser`` + sets up
    ``retrieval.expand_window`` (which replaces ``MetadataReplacementPostProcessor``):
    embed the small sentence, answer with the big window.
    """
    sentences = _sentences(text)
    base = doc_id or hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:12]
    out: list[Chunk] = []
    for i, sentence in enumerate(sentences):
        lo = max(0, i - window_size)
        hi = min(len(sentences), i + window_size + 1)
        out.append(
            Chunk(
                id=f"{base}-s{i:04d}",
                text=sentence,
                metadata={"window": " ".join(sentences[lo:hi]), "sentence_index": i},
            )
        )
    return out
