from __future__ import annotations

import pytest

import tai_aitutor as tai
from tai_aitutor.chunking import _markdown_blocks
from tai_aitutor.errors import TaiAitutorError

TEXT = ("Retrieval augmented generation grounds model answers in your own data. " * 80).strip()


def test_chunk_produces_multiple_overlapping_pieces():
    pieces = tai.chunk(TEXT, chunk_size=128, chunk_overlap=32)
    assert len(pieces) > 2
    assert all(p.strip() for p in pieces)
    # each piece respects the token budget (tokenizer path) or ~4 chars/token (fallback)
    assert all(tai.n_tokens(p) <= 128 + 4 for p in pieces)


def test_chunk_empty_and_invalid():
    assert tai.chunk("") == []
    assert tai.chunk("   \n  ") == []
    with pytest.raises(TaiAitutorError):
        tai.chunk(TEXT, chunk_size=100, chunk_overlap=100)
    with pytest.raises(TaiAitutorError):
        tai.chunk(TEXT, chunk_size=0)


def test_chunk_short_text_is_single_piece():
    assert tai.chunk("hello world", chunk_size=512, chunk_overlap=128) == ["hello world"]


def test_chunk_document_stable_ids_and_metadata():
    doc = tai.Document(text=TEXT, metadata={"title": "RAG"}, id="doc7")
    chunks = tai.chunk_document(doc, chunk_size=128, chunk_overlap=32)
    assert chunks[0].id == "doc7-0000"
    assert chunks[1].id == "doc7-0001"
    assert all(c.metadata == {"title": "RAG"} for c in chunks)
    # deterministic across runs
    again = tai.chunk_document(doc, chunk_size=128, chunk_overlap=32)
    assert [c.id for c in again] == [c.id for c in chunks]
    assert [c.text for c in again] == [c.text for c in chunks]


def test_chunk_sentences_never_splits_sentences():
    sentences = [f"Sentence number {i} talks about topic {i}." for i in range(60)]
    text = " ".join(sentences)
    pieces = tai.chunk_sentences(text, chunk_size=100, chunk_overlap=20)
    assert len(pieces) > 1
    for piece in pieces:
        # every piece is made of whole input sentences
        for part in piece.split(". "):
            part = part.rstrip(".")
            if part:
                assert part + "." in text


def _fences_balanced(text: str) -> bool:
    return text.count("```") % 2 == 0


def test_heading_aware_keeps_code_blocks_atomic():
    code = "```python\n" + "\n".join(f"x_{i} = compute({i})" for i in range(400)) + "\n```"
    md = (
        "# Setup\n\nSome intro prose here.\n\n"
        "## Install\n\n" + ("Install instructions paragraph. " * 30) + "\n\n"
        + code
        + "\n\nClosing remarks paragraph.\n"
    )
    chunks = tai.heading_aware_markdown_chunks(md, chunk_size=200, chunk_overlap=40)
    assert len(chunks) >= 2
    # Rule 1: no chunk ever contains an unbalanced fence — code is never split
    assert all(_fences_balanced(c) for c in chunks)
    # the (oversized) code block survived intact in exactly one chunk
    holders = [c for c in chunks if "x_0 = compute(0)" in c]
    assert len(holders) == 1
    assert "x_399 = compute(399)" in holders[0]


def test_heading_aware_reprints_heading_on_continuation():
    md = "# Guide\n\n## Deep Section\n\n" + "\n\n".join(
        f"Paragraph {i}. " + ("More words here. " * 25) for i in range(8)
    )
    chunks = tai.heading_aware_markdown_chunks(md, chunk_size=150, chunk_overlap=30)
    assert len(chunks) >= 2
    # continuation chunks re-state the section heading (Rule 2)
    assert all("## Deep Section" in c or "# Guide" in c for c in chunks)


def test_markdown_blocks_parsing():
    md = "# H1\n\npara one\nstill para one\n\n```js\ncode()\n```\n\npara two"
    kinds = [k for k, _ in _markdown_blocks(md)]
    assert kinds == ["heading", "text", "code", "text"]


def test_sentence_window_chunks():
    text = " ".join(f"Sentence {i} is here." for i in range(10))
    chunks = tai.sentence_window_chunks(text, window_size=2, doc_id="w")
    assert len(chunks) == 10
    middle = chunks[5]
    assert middle.text == "Sentence 5 is here."
    for j in (3, 4, 5, 6, 7):
        assert f"Sentence {j} is here." in middle.metadata["window"]
    assert "Sentence 2 is here." not in middle.metadata["window"]
    # edges clamp cleanly
    assert "Sentence 0 is here." in chunks[0].metadata["window"]
