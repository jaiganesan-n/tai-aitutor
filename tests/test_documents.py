from __future__ import annotations

import json

import pytest

import tai_aitutor as tai
from tai_aitutor.errors import TaiAitutorError


def test_load_csv_with_meta_and_embeddings(tmp_path):
    path = tmp_path / "articles.csv"
    rows = [
        {"title": "A", "content": "First article text.", "url": "https://a", "emb": json.dumps([0.1, 0.2])},
        {"title": "B", "content": "Second article text.", "url": "https://b", "emb": json.dumps([0.3, 0.4])},
        {"title": "C", "content": "", "url": "https://c", "emb": ""},  # empty text skipped
    ]
    import csv

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "content", "url", "emb"])
        writer.writeheader()
        writer.writerows(rows)

    docs = tai.load_csv(path, text_col="content", meta_cols=("title", "url"), embedding_col="emb")
    assert len(docs) == 2
    assert docs[0].text == "First article text."
    assert docs[0].metadata["title"] == "A"
    assert docs[0].metadata["embedding"] == [0.1, 0.2]


def test_load_csv_never_evals(tmp_path):
    """Regression: embeddings parse with json.loads only — Python expressions are data, not code."""
    path = tmp_path / "evil.csv"
    payload = "__import__('os').system('echo pwned')"
    path.write_text(f'content,emb\n"text","{payload}"\n')
    with pytest.raises(TaiAitutorError) as err:
        tai.load_csv(path, text_col="content", embedding_col="emb")
    assert "json" in str(err.value).lower()
    # and a Python-literal-style list (single quotes) is also rejected, not eval'd
    path.write_text("content,emb\ntext,\"[0.1, 0.2]\"\n")
    docs = tai.load_csv(path, text_col="content", embedding_col="emb")
    assert docs[0].metadata["embedding"] == [0.1, 0.2]


def test_load_csv_missing_column(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(TaiAitutorError) as err:
        tai.load_csv(path, text_col="content")
    assert "content" in str(err.value)


def test_load_directory_pdf(tmp_path):
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    pdf_path = tmp_path / "doc.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with pdf_path.open("wb") as f:
        writer.write(f)
    # blank page → no text → document skipped, but no crash
    docs = tai.load_files([pdf_path])
    assert docs == []


def test_load_csv_id_col_readable_and_unique(tmp_path):
    path = tmp_path / "articles.csv"
    long_title = "An Extremely Long Article Title That Goes Way Past Forty Characters"
    path.write_text(
        "title,content\n"
        f'"{long_title}","text a"\n'
        '"Same Title","text b"\n'
        '"Same Title","text c"\n'
    )
    docs = tai.load_csv(path, text_col="content", id_col="title")
    assert docs[0].id == long_title[:40]          # course convention: title[:40]
    assert docs[1].id == "Same Title"
    assert docs[2].id == "Same Title~2"           # duplicates never collide
    # chunk ids inherit the readable id
    chunks = tai.chunk_document(docs[1], chunk_size=512, chunk_overlap=64)
    assert chunks[0].id == "Same Title-0000"


def test_load_csv_missing_id_col_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("content\nhello\n")
    with pytest.raises(TaiAitutorError) as err:
        tai.load_csv(path, text_col="content", id_col="title")
    assert "id_col" in str(err.value)


def test_document_stable_id():
    d1 = tai.Document(text="same text")
    d2 = tai.Document(text="same text")
    assert d1.stable_id() == d2.stable_id()
    assert tai.Document(text="x", id="explicit").stable_id() == "explicit"
