"""datasets tests — registry wiring and extraction, zero network."""

from __future__ import annotations

import json
import zipfile

import pytest

import tai_aitutor as tai
from tai_aitutor import datasets
from tai_aitutor.errors import TaiAitutorError


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TAI_AITUTOR_CACHE", str(tmp_path / "cache"))
    yield


def test_cache_dir_env_override(tmp_path):
    assert str(datasets.cache_dir()).endswith("cache")
    assert datasets.cache_dir().exists()


def test_registry_shapes():
    for key, spec in datasets._FILES.items():
        if isinstance(spec, str):
            assert spec.startswith("https://"), key
        else:
            repo_id, filename = spec
            assert "/" in repo_id and filename, key
    # every vector store variant points at a registered file
    for variant, (key, provider, model) in datasets._VECTORSTORE_VARIANTS.items():
        assert key in datasets._FILES, variant
        assert provider and model


def test_mini_articles_wiring(monkeypatch, tmp_path):
    csv_path = tmp_path / "mini-llama-articles.csv"
    csv_path.write_text(
        'title,content,url,source,embedding\n'                       # real column: "source"
        'Beyond GPT-4,"Body one",https://a,tai_blog,"[0.1, 0.2]"\n'
        'Fine-tuning 101,"Body two",https://b,tai_blog,"[0.3, 0.4]"\n'
    )
    monkeypatch.setattr(datasets, "_fetch", lambda key: csv_path)
    docs = tai.mini_articles()
    assert len(docs) == 2
    # Finding 2: "source" survives — build_where_filter(sources) depends on it
    assert docs[0].metadata == {"title": "Beyond GPT-4", "url": "https://a",
                                "source": "tai_blog"}
    # Finding 3: readable stable ids from the title, so chunk ids read
    # "Beyond GPT-4-0000" instead of a hash — and line up across re-ingests
    assert docs[0].id == "Beyond GPT-4"
    assert docs[0].stable_id() == "Beyond GPT-4"

    embedded = tai.mini_articles(with_embeddings=True)
    assert embedded[0].metadata["embedding"] == [0.1, 0.2]


def test_ai_tutor_knowledge_wiring(monkeypatch, tmp_path):
    jsonl = tmp_path / "kb.jsonl"
    jsonl.write_text(
        json.dumps({"doc_id": "hf_transformers/quicktour", "content": "Doc text.",
                    "name": "Transformers Quicktour", "source": "hf"}) + "\n"
    )
    monkeypatch.setattr(datasets, "_fetch", lambda key: jsonl)
    docs = tai.ai_tutor_knowledge()
    # Finding 4: the corpus's own doc_id is the document id (not a hash)
    assert docs[0].id == "hf_transformers/quicktour"
    assert "doc_id" not in docs[0].metadata
    # Finding 5: "name" is mirrored into "title" so displays/prompts show names
    assert docs[0].metadata["title"] == "Transformers Quicktour"
    assert docs[0].metadata["source"] == "hf"


def test_qa_dataset_wiring_and_unknown(monkeypatch, tmp_path):
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(json.dumps({
        "queries": {"q": "What?"}, "corpus": {"c": "Because."},
        "relevant_docs": {"q": ["c"]}, "mode": "text"}))
    monkeypatch.setattr(datasets, "_fetch", lambda key: qa_path)
    qa = tai.qa_dataset("rag_eval_50")
    assert len(qa) == 1
    with pytest.raises(TaiAitutorError) as err:
        tai.qa_dataset("nope")
    assert "rag_eval_50" in str(err.value)


def _make_zip(tmp_path, inner_dir: bool):
    archive = tmp_path / "vectorstore.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        prefix = "vector_store/" if inner_dir else ""
        zf.writestr(f"{prefix}chroma.sqlite3", "not-really-a-db")
        zf.writestr(f"{prefix}collection/data", "x")
    return archive


def test_prebuilt_chroma_extracts_and_caches(monkeypatch, tmp_path):
    archive = _make_zip(tmp_path, inner_dir=True)
    calls = []

    def fake_fetch(key):
        calls.append(key)
        return archive

    monkeypatch.setattr(datasets, "_fetch", fake_fetch)
    path = tai.prebuilt_chroma()
    assert path.name == "vector_store"          # unwrapped the single top-level folder
    assert (path / "chroma.sqlite3").exists()
    again = tai.prebuilt_chroma()               # second call reuses the extraction
    assert again == path
    assert calls == ["vectorstore", "vectorstore"]

    flat_dir = tmp_path / "flat"
    flat_dir.mkdir()
    flat = _make_zip(flat_dir, inner_dir=False)
    monkeypatch.setattr(datasets, "_fetch", lambda key: flat)
    flat_path = tai.prebuilt_chroma("windowed")
    assert (flat_path / "chroma.sqlite3").exists()


def test_prebuilt_chroma_unknown_variant():
    with pytest.raises(TaiAitutorError) as err:
        tai.prebuilt_chroma("gemini")
    assert "org data migration" in str(err.value)


def test_research_papers(monkeypatch, tmp_path):
    archive = tmp_path / "rag_research_paper.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("paper1.pdf", "pdf-bytes")
    monkeypatch.setattr(datasets, "_fetch", lambda key: archive)
    out = tai.research_papers()
    assert (out / "paper1.pdf").exists()
