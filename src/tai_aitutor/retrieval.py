"""Retrieval: from a question to scored chunks — dense, keyword, hybrid, reranked.

Built across Sections 4 and 7 of the course. Dense :func:`search` comes from
"From Script to Pipeline"; everything else is the Section 7 stack, which after
the port is, stage for stage, the production retrieval service in miniature:

- :class:`BM25Index` — hand-rolled Okapi BM25 (Hybrid Search lesson; replaces
  ``SimpleKeywordTableIndex`` + ``KeywordTableSimpleRetriever``)
- :func:`rrf_fuse` — Reciprocal Rank Fusion (replaces the round-robin merge)
- :func:`hybrid_search` — dense ∪ BM25 → RRF, production constants
- :func:`rerank` — Cohere rerank v4 (replaces ``CohereRerank``)
- :func:`judge_rerank` — LLM-as-judge reranking (replaces ``RankGPTRerank`` /
  custom ``BaseNodePostprocessor``); the judge's scores AND ordering survive
- :func:`hyde_search`, :func:`decompose_question` + :func:`subquestion_answer`,
  :func:`multi_step_answer` — query transforms (replace ``HyDEQueryTransform``,
  ``SubQuestionQueryEngine`` + ``LLMQuestionGenerator`` + ``QueryEngineTool``,
  ``MultiStepQueryEngine`` + ``StepDecomposeQueryTransform``)
- :func:`pack_context` — the production token budget knob

Production constants mirrored from the live tutor: dense top 15 / BM25 top 30
/ RRF k=60 keep 30 / rerank ``rerank-v4.0-fast`` top 5 floor 0.10 / 100k-token
budget; BM25 k1=1.5, b=0.75 with the code-aware tokenizer.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from . import config as _cfg
from . import embeddings as _embeddings
from . import llm as _llm
from .chunking import Chunk
from .errors import TaiAitutorError
from .tokens import n_tokens

__all__ = [
    "ScoredChunk",
    "search",
    "expand_window",
    "code_tokenize",
    "BM25Index",
    "rrf_fuse",
    "hybrid_search",
    "rerank",
    "judge_rerank",
    "hyde_search",
    "decompose_question",
    "subquestion_answer",
    "multi_step_answer",
    "pack_context",
]


@dataclass
class ScoredChunk:
    """A retrieved chunk with its score and 1-based rank (was ``NodeWithScore``)."""

    chunk: Chunk
    score: float
    rank: int

    # Convenience passthroughs so notebook code reads naturally (hit.text).
    @property
    def id(self) -> str:
        return self.chunk.id

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def metadata(self) -> dict:
        return self.chunk.metadata

    def __repr__(self) -> str:
        preview = self.text[:50].replace("\n", " ")
        return f"ScoredChunk(rank={self.rank}, score={self.score:.3f}, {preview!r}...)"


# --------------------------------------------------------------------------- #
# Dense search
# --------------------------------------------------------------------------- #


def _as_vector(result) -> list[float]:
    """Normalise an embed_fn result: a vector, or [vector]."""
    if result and isinstance(result[0], list):
        return result[0]
    return result


def _query_by_vector(
    collection,
    vector: list[float],
    top_k: int,
    where: dict | None,
    where_document: dict | None,
) -> list[ScoredChunk]:
    kwargs: dict = {"query_embeddings": [vector], "n_results": top_k}
    if where is not None:
        kwargs["where"] = where
    if where_document is not None:
        kwargs["where_document"] = where_document
    result = collection.query(**kwargs, include=["documents", "metadatas", "distances"])

    ids = (result.get("ids") or [[]])[0]
    texts = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    hits: list[ScoredChunk] = []
    for i, chunk_id in enumerate(ids):
        metadata = dict(metadatas[i] or {}) if i < len(metadatas) else {}
        distance = distances[i] if i < len(distances) else None
        score = 1.0 - distance if distance is not None else 0.0
        hits.append(
            ScoredChunk(
                chunk=Chunk(id=chunk_id, text=texts[i] or "", metadata=metadata),
                score=score,
                rank=i + 1,
            )
        )
    return hits


def search(
    query: str,
    collection,
    top_k: int = 5,
    where: dict | None = None,
    where_document: dict | None = None,
    embed_fn=None,
) -> list[ScoredChunk]:
    """Dense retrieval: embed the query, return the ``top_k`` most similar chunks.

    ``where`` takes a Chroma metadata filter — build source scoping with
    ``vectorstore.build_where_filter`` (the production ``$eq`` / ``$in`` shape).
    ``where_document={"$contains": "..."}`` adds full-text matching (the
    Metadata Filtering lesson's TEXT_MATCH case).

    The ``top_k`` you pass is the ``top_k`` you get (at most) — the old
    notebook bug where a stored cap was silently ignored cannot happen here.
    """
    if top_k <= 0:
        return []
    embed = embed_fn or _embeddings.embed
    query_vector = _as_vector(embed(query, task="query"))
    return _query_by_vector(collection, query_vector, top_k, where, where_document)


def expand_window(hits: list[ScoredChunk], window_key: str = "window") -> list[ScoredChunk]:
    """Swap each sentence-sized hit for its stored neighborhood window.

    The answer-time half of sentence-window retrieval (Advanced Retrieval
    lesson): embed small, answer big. Replaces
    ``MetadataReplacementPostProcessor(target_metadata_key="window")`` — and the
    mechanism is exactly this visible: read ``metadata["window"]``, use it as
    the text.
    """
    out: list[ScoredChunk] = []
    for hit in hits:
        window = hit.metadata.get(window_key)
        if window:
            expanded = Chunk(id=hit.id, text=window, metadata=dict(hit.metadata))
            out.append(ScoredChunk(chunk=expanded, score=hit.score, rank=hit.rank))
        else:
            out.append(hit)
    return out


# --------------------------------------------------------------------------- #
# BM25 (Hybrid Search lesson — the formula fits on a slide)
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"[A-Za-z0-9_./\-]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SPLIT_RE = re.compile(r"[._/\-]+")


def code_tokenize(text: str) -> list[str]:
    """The code-aware tokenizer: splits camelCase and dotted.paths, keeps terms like "c".

    ``"VectorStoreIndex"`` → ``vector, store, index``;
    ``"llama_index.core"`` → ``llama, index, core`` — so searches for either
    form match. Same tokenizer at index time and query time, always.
    """
    tokens: list[str] = []
    for raw in _WORD_RE.findall(text):
        for part in _SPLIT_RE.split(raw):
            if not part:
                continue
            for sub in _CAMEL_RE.split(part):
                if sub:
                    tokens.append(sub.lower())
    return tokens


_BM25_FORMAT = "tai-aitutor-bm25"
_BM25_VERSION = 1


class BM25Index:
    """Okapi BM25 over chunks — written out, not imported (k1=1.5, b=0.75, as in production).

    Replaces ``SimpleKeywordTableIndex`` + ``KeywordTableSimpleRetriever``:
    the old lesson's demo never actually computed BM25; this is the real
    scorer, ~60 lines with the tokenizer.

    >>> bm25 = BM25Index().build(get_all_chunks(col))
    >>> hits = bm25.search("chroma where filter", top_k=30)

    Persistence is versioned gzipped JSON (``save``/``load``) — never pickle,
    same as the production index artifact.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._tf: list[dict[str, int]] = []
        self._lengths: list[int] = []
        self._df: dict[str, int] = {}
        self._postings: dict[str, list[int]] = {}
        self._avgdl: float = 0.0

    def __len__(self) -> int:
        return len(self._chunks)

    def build(self, chunks: list[Chunk]) -> BM25Index:
        """Tokenise and index every chunk. Returns self for chaining."""
        self._chunks = list(chunks)
        self._tf, self._lengths, self._df, self._postings = [], [], {}, {}
        for doc_index, chunk in enumerate(self._chunks):
            counts: dict[str, int] = {}
            tokens = code_tokenize(chunk.text)
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self._tf.append(counts)
            self._lengths.append(len(tokens))
            for token in counts:
                self._df[token] = self._df.get(token, 0) + 1
                self._postings.setdefault(token, []).append(doc_index)
        self._avgdl = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        return self

    def search(self, query: str, top_k: int = 30) -> list[ScoredChunk]:
        """Score documents sharing terms with the query; return the top_k."""
        if not self._chunks or top_k <= 0:
            return []
        n_docs = len(self._chunks)
        scores: dict[int, float] = {}
        for token in set(code_tokenize(query)):
            doc_indexes = self._postings.get(token)
            if not doc_indexes:
                continue
            df = self._df[token]
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for doc_index in doc_indexes:
                tf = self._tf[doc_index][token]
                length_norm = 1.0 - self.b + self.b * (
                    self._lengths[doc_index] / self._avgdl if self._avgdl else 1.0
                )
                scores[doc_index] = scores.get(doc_index, 0.0) + idf * (
                    tf * (self.k1 + 1.0)
                ) / (tf + self.k1 * length_norm)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [
            ScoredChunk(chunk=self._chunks[doc_index], score=score, rank=position + 1)
            for position, (doc_index, score) in enumerate(ranked)
        ]

    def save(self, path: str | Path) -> None:
        """Versioned gzipped JSON — inspectable, diffable, never pickle."""
        payload = {
            "format": _BM25_FORMAT,
            "version": _BM25_VERSION,
            "k1": self.k1,
            "b": self.b,
            "docs": [
                {"id": c.id, "text": c.text, "metadata": c.metadata} for c in self._chunks
            ],
        }
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> BM25Index:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("format") != _BM25_FORMAT:
            raise TaiAitutorError(f"{path} is not a tai-aitutor BM25 index file.")
        if payload.get("version") != _BM25_VERSION:
            raise TaiAitutorError(
                f"BM25 index version {payload.get('version')} != {_BM25_VERSION} — rebuild it."
            )
        index = cls(k1=payload["k1"], b=payload["b"])
        index.build(
            [Chunk(id=d["id"], text=d["text"], metadata=d.get("metadata") or {}) for d in payload["docs"]]
        )
        return index


# --------------------------------------------------------------------------- #
# Fusion + hybrid (production: dense 15 ∪ BM25 30 → RRF k=60 keep 30)
# --------------------------------------------------------------------------- #


def rrf_fuse(*ranked_lists: list[ScoredChunk], k: int = 60, keep: int = 30) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion: ``score += 1 / (k + rank)`` across the input lists.

    The five-line idea that replaces the old notebook's round-robin merge —
    and unlike that merge, ``keep`` is actually applied (regression-tested).
    Positions in each list (1-based) are what count; input scores don't need
    to be comparable across lists (that's the point of RRF).
    """
    fused: dict[str, float] = {}
    first_seen: dict[str, ScoredChunk] = {}
    for ranked in ranked_lists:
        for position, hit in enumerate(ranked, start=1):
            fused[hit.id] = fused.get(hit.id, 0.0) + 1.0 / (k + position)
            if hit.id not in first_seen:
                first_seen[hit.id] = hit
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[: max(0, keep)]
    return [
        ScoredChunk(chunk=first_seen[chunk_id].chunk, score=score, rank=position + 1)
        for position, (chunk_id, score) in enumerate(ordered)
    ]


def _matches_where(metadata: dict, where: dict | None) -> bool:
    """Minimal matcher for the course filter shapes: {key: value | {$eq}| {$in}}."""
    if not where:
        return True
    for key, condition in where.items():
        value = metadata.get(key)
        if isinstance(condition, dict):
            if "$eq" in condition and value != condition["$eq"]:
                return False
            if "$in" in condition and value not in condition["$in"]:
                return False
        elif value != condition:
            return False
    return True


def hybrid_search(
    query: str,
    collection,
    bm25: BM25Index | None = None,
    dense_top_k: int = 15,
    bm25_top_k: int = 30,
    keep: int = 30,
    k: int = 60,
    where: dict | None = None,
    embed_fn=None,
) -> list[ScoredChunk]:
    """Dense ∪ BM25 → Reciprocal Rank Fusion, with the production constants.

    Pass a prebuilt ``bm25`` index for repeated calls (build once:
    ``BM25Index().build(get_all_chunks(collection))``); with ``bm25=None`` one
    is built on the fly, which is fine at course scale but wasteful in a loop.
    ``where`` scopes both legs (Chroma-side for dense; metadata match for BM25).
    """
    dense_hits = search(query, collection, top_k=dense_top_k, where=where, embed_fn=embed_fn)
    if bm25 is None:
        from .vectorstore import get_all_chunks  # local import: vectorstore ↔ retrieval

        bm25 = BM25Index().build(get_all_chunks(collection))
    keyword_hits = bm25.search(query, top_k=bm25_top_k)
    if where is not None:
        keyword_hits = [h for h in keyword_hits if _matches_where(h.metadata, where)]
        keyword_hits = [
            ScoredChunk(chunk=h.chunk, score=h.score, rank=i + 1)
            for i, h in enumerate(keyword_hits)
        ]
    return rrf_fuse(dense_hits, keyword_hits, k=k, keep=keep)


# --------------------------------------------------------------------------- #
# Reranking (production: rerank-v4.0-fast, top 5, score floor 0.10)
# --------------------------------------------------------------------------- #


def rerank(
    query: str,
    hits: list[ScoredChunk],
    model: str = "rerank-v4.0-fast",
    top_n: int = 5,
    floor: float = 0.10,
    api_key: str | None = None,
) -> list[ScoredChunk]:
    """Cohere cross-encoder reranking of retrieved candidates (replaces ``CohereRerank``).

    An explicit stage between retrieval and answering — it cannot be silently
    skipped the way ``node_postprocessors`` on ``as_retriever`` was, so eval
    tables built on it measure what actually ran. Results below ``floor``
    relevance are dropped (production behaviour).
    """
    if not hits:
        return []
    from .embeddings import _client_cohere

    client = _client_cohere(api_key)
    response = client.rerank(
        model=model,
        query=query,
        documents=[hit.text for hit in hits],
        top_n=min(top_n, len(hits)),
    )
    out: list[ScoredChunk] = []
    for result in response.results:
        score = float(result.relevance_score)
        if score < floor:
            continue
        source = hits[result.index]
        out.append(ScoredChunk(chunk=source.chunk, score=score, rank=len(out) + 1))
    return out


class _ChunkScore(BaseModel):
    index: int = Field(description="1-based index of the excerpt being scored.")
    score: float = Field(description="Relevance to the query, 0 (unrelated) to 10 (direct answer).")


class _JudgeRanking(BaseModel):
    scores: list[_ChunkScore]


def judge_rerank(
    query: str,
    hits: list[ScoredChunk],
    top_n: int = 3,
    *,
    model: str | None = None,
    provider: str | None = None,
) -> list[ScoredChunk]:
    """LLM-as-judge reranking: one typed call scores every chunk against the query.

    Replaces ``RankGPTRerank`` and the notebook's custom ``BaseNodePostprocessor``
    — and fixes its bug by construction: the judge's ORDER and SCORES are what
    you get back (the old ``_postprocess_nodes`` rebuilt results in original
    retrieval order with stale similarity scores, so the judge's work was
    thrown away).

    The concept lesson's framing stands: this teaches the idea; production uses
    a dedicated cross-encoder (:func:`rerank`).
    """
    if not hits:
        return []
    numbered = "\n\n".join(f"[{i}] {hit.text}" for i, hit in enumerate(hits, start=1))
    prompt = (
        f"Query: {query}\n\n"
        f"Excerpts:\n\n{numbered}\n\n"
        f"Score EVERY excerpt's relevance to the query from 0 to 10."
    )
    ranking = _llm.extract(
        prompt,
        _JudgeRanking,
        system="You are a strict relevance judge for a retrieval system.",
        model=model,
        provider=provider,
    )
    by_index: dict[int, float] = {}
    for item in ranking.scores:
        if 1 <= item.index <= len(hits) and item.index not in by_index:
            by_index[item.index] = float(item.score)
    ordered = sorted(by_index.items(), key=lambda item: (-item[1], item[0]))[: max(0, top_n)]
    return [
        ScoredChunk(chunk=hits[index - 1].chunk, score=score, rank=position + 1)
        for position, (index, score) in enumerate(ordered)
    ]


# --------------------------------------------------------------------------- #
# Query transforms (Query Variation and Augmentation lesson)
# --------------------------------------------------------------------------- #


def hyde_search(
    query: str,
    collection,
    top_k: int = 5,
    include_original: bool = True,
    hypothetical: str | None = None,
    where: dict | None = None,
    *,
    model: str | None = None,
    provider: str | None = None,
    embed_fn=None,
) -> list[ScoredChunk]:
    """HyDE in three visible steps: hypothesise → embed → search.

    Generate a hypothetical answer, embed THAT (as a document) instead of the
    short query, and retrieve by similarity to it. ``include_original=True``
    averages in the real query's embedding, matching the old
    ``HyDEQueryTransform(include_original=True)`` behaviour. Pass your own
    ``hypothetical=`` to inspect or reuse the generated passage (the lesson
    prints it — the failure mode where HyDE hallucinates the wrong topic is
    the teaching moment).
    """
    if hypothetical is None:
        hypothetical = _llm.generate(
            f"Write a short factual passage (3-5 sentences) that directly answers this "
            f"question, as it would appear in course documentation:\n\n{query}",
            model=model,
            provider=provider,
        )
    embed = embed_fn or _embeddings.embed
    vector = _as_vector(embed(hypothetical, task="document"))
    if include_original:
        query_vector = _as_vector(embed(query, task="query"))
        vector = [(h + q) / 2.0 for h, q in zip(vector, query_vector)]
    return _query_by_vector(collection, vector, top_k, where, None)


class _SubQuestions(BaseModel):
    questions: list[str] = Field(description="Independent, self-contained sub-questions.")


def decompose_question(
    question: str,
    n_max: int = 4,
    *,
    model: str | None = None,
    provider: str | None = None,
) -> list[str]:
    """Break a multi-part question into self-contained sub-questions (typed call).

    Replaces ``LLMQuestionGenerator``. A question that doesn't decompose comes
    back as itself — no machinery for the simple case.
    """
    result = _llm.extract(
        f"Question: {question}\n\n"
        f"If this question asks about multiple distinct things, split it into at most "
        f"{n_max} independent sub-questions, each answerable on its own. "
        f"If it is already a single question, return it unchanged as the only item.",
        _SubQuestions,
        system="You decompose questions for a retrieval system.",
        model=model,
        provider=provider,
    )
    questions = [q.strip() for q in result.questions if q.strip()][:n_max]
    return questions or [question]


def _merge_hits(*hit_lists: list[ScoredChunk]) -> list[ScoredChunk]:
    """Dedupe hits by id (keep best score), order by score, re-rank."""
    best: dict[str, ScoredChunk] = {}
    for hits in hit_lists:
        for hit in hits:
            if hit.id not in best or hit.score > best[hit.id].score:
                best[hit.id] = hit
    ordered = sorted(best.values(), key=lambda hit: (-hit.score, hit.id))
    return [
        ScoredChunk(chunk=hit.chunk, score=hit.score, rank=position + 1)
        for position, hit in enumerate(ordered)
    ]


def _retrieve_for(question, collection, top_k, where, retriever, embed_fn):
    if retriever is not None:
        return list(retriever(question))
    if collection is None:
        raise TaiAitutorError("Pass collection=... or retriever=... to retrieve context.")
    return search(question, collection, top_k=top_k, where=where, embed_fn=embed_fn)


def subquestion_answer(
    question: str,
    collection=None,
    top_k: int = 5,
    n_max: int = 4,
    where: dict | None = None,
    retriever=None,
    *,
    model: str | None = None,
    provider: str | None = None,
    embed_fn=None,
):
    """Decompose → retrieve per sub-question → answer each → synthesize.

    The whole ``SubQuestionQueryEngine`` + ``QueryEngineTool`` + ``ToolMetadata``
    stack as a visible loop. Returns an :class:`~tai_aitutor.synthesis.Answer`
    whose sources are the merged (deduped) evidence across sub-questions.
    """
    from .prompts import RAG_SYSTEM
    from .synthesis import Answer, build_rag_prompt

    cfg = _cfg.resolve(provider=provider, chat_model=model)
    sub_questions = decompose_question(question, n_max=n_max, model=model, provider=provider)

    findings: list[tuple[str, str]] = []
    all_hits: list[list[ScoredChunk]] = []
    total_in = total_out = 0
    for sub_question in sub_questions:
        hits = _retrieve_for(sub_question, collection, top_k, where, retriever, embed_fn)
        all_hits.append(hits)
        text, usage = _llm._complete(
            build_rag_prompt(sub_question, hits),
            RAG_SYSTEM,
            cfg=cfg,
            temperature=None,
            max_tokens=None,
            reasoning_effort=None,
        )
        findings.append((sub_question, text))
        total_in += usage.input_tokens
        total_out += usage.output_tokens

    joined = "\n\n".join(f"Sub-question: {q}\nAnswer: {a}" for q, a in findings)
    final_text, usage = _llm._complete(
        f"Original question: {question}\n\n{joined}\n\n"
        "Write the final answer to the original question, synthesising the sub-answers.",
        RAG_SYSTEM,
        cfg=cfg,
        temperature=None,
        max_tokens=None,
        reasoning_effort=None,
    )
    total_in += usage.input_tokens
    total_out += usage.output_tokens
    return Answer(
        text=final_text,
        sources=_merge_hits(*all_hits),
        usage=_llm.Usage(total_in, total_out),
    )


class _NextStep(BaseModel):
    done: bool = Field(description="True when enough is known to answer the original question.")
    next_question: str = Field(
        default="", description="The next follow-up question to retrieve for (empty when done)."
    )


def multi_step_answer(
    question: str,
    collection=None,
    max_steps: int = 3,
    top_k: int = 5,
    where: dict | None = None,
    retriever=None,
    *,
    model: str | None = None,
    provider: str | None = None,
    embed_fn=None,
):
    """Iterative retrieval: ask → retrieve → decide the next question → repeat.

    Replaces ``MultiStepQueryEngine`` + ``StepDecomposeQueryTransform`` with a
    plain loop: each step, a typed call decides whether enough is known or what
    to look up next (starting from the original question). Bounded by
    ``max_steps`` — the modern form of this technique is an agent rewriting its
    own retrieval query per tool call, which Section 13 shows.
    """
    from .prompts import RAG_SYSTEM
    from .synthesis import Answer, build_rag_prompt

    cfg = _cfg.resolve(provider=provider, chat_model=model)
    findings: list[tuple[str, str]] = []
    all_hits: list[list[ScoredChunk]] = []
    total_in = total_out = 0
    current_question = question

    for _step in range(max_steps):
        hits = _retrieve_for(current_question, collection, top_k, where, retriever, embed_fn)
        all_hits.append(hits)
        text, usage = _llm._complete(
            build_rag_prompt(current_question, hits),
            RAG_SYSTEM,
            cfg=cfg,
            temperature=None,
            max_tokens=None,
            reasoning_effort=None,
        )
        findings.append((current_question, text))
        total_in += usage.input_tokens
        total_out += usage.output_tokens

        known = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in findings)
        step = _llm.extract(
            f"Original question: {question}\n\nKnown so far:\n{known}\n\n"
            "Is this enough to answer the original question? If not, what single "
            "follow-up question should be retrieved next?",
            _NextStep,
            system="You plan retrieval steps for answering a question.",
            model=model,
            provider=provider,
        )
        if step.done or not step.next_question.strip():
            break
        current_question = step.next_question.strip()

    joined = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in findings)
    final_text, usage = _llm._complete(
        f"Original question: {question}\n\nGathered:\n{joined}\n\n"
        "Write the final answer to the original question.",
        RAG_SYSTEM,
        cfg=cfg,
        temperature=None,
        max_tokens=None,
        reasoning_effort=None,
    )
    total_in += usage.input_tokens
    total_out += usage.output_tokens
    return Answer(
        text=final_text,
        sources=_merge_hits(*all_hits),
        usage=_llm.Usage(total_in, total_out),
    )


# --------------------------------------------------------------------------- #
# Token budget (production: 100k on retrieval payloads — a real cost knob)
# --------------------------------------------------------------------------- #


def pack_context(
    hits: list[ScoredChunk],
    max_tokens: int = 100_000,
    model: str | None = None,
) -> list[ScoredChunk]:
    """Keep whole hits, in order, until the token budget is spent.

    The production budget knob: retrieval payloads dominate input tokens, and
    the evals found a 30k budget matched the 100k budget's recall — so this
    argument is where cost goes to be cut. Chunks are never truncated mid-way;
    the first one that doesn't fit is dropped along with everything after it
    counts against nothing.
    """
    kept: list[ScoredChunk] = []
    spent = 0
    for hit in hits:
        cost = n_tokens(hit.text, model=model)
        if spent + cost > max_tokens:
            continue
        spent += cost
        kept.append(ScoredChunk(chunk=hit.chunk, score=hit.score, rank=len(kept) + 1))
    return kept
