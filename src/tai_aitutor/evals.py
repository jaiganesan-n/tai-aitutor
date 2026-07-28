"""RAG evaluation: synthetic QA datasets, retrieval metrics, and LLM judges.

Built in: "Evaluating Your RAG Pipeline" (Section 4). Replaces the LlamaIndex
evaluation stack — ``generate_question_context_pairs``,
``EmbeddingQAFinetuneDataset``, ``RetrieverEvaluator.from_metric_names``,
``FaithfulnessEvaluator`` / ``RelevancyEvaluator`` / ``CorrectnessEvaluator``,
and ``BatchEvalRunner`` (plus its ``nest_asyncio`` boilerplate) — with a
dataclass, two ten-line metrics, and three typed judge calls.

Compatibility promise: :class:`QADataset` reads and writes the SAME JSON shape
as ``EmbeddingQAFinetuneDataset`` (``queries`` / ``corpus`` / ``relevant_docs``),
so every existing ``rag_eval_dataset*.json`` artifact keeps loading unchanged.

The metrics mirror production's ``retrieval_metrics()`` in ``evals/grade.py``:
the rank of the gold chunk among retrieved results drives both numbers.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from . import prompts
from .chunking import Chunk
from .errors import TaiAitutorError
from .llm import extract
from .retrieval import ScoredChunk, search
from .vectorstore import get_all_chunks

__all__ = [
    "QADataset",
    "make_qa_pairs",
    "hit_rate",
    "mrr",
    "evaluate_retrieval",
    "RetrievalReport",
    "QueryResult",
    "FaithfulnessVerdict",
    "RelevancyVerdict",
    "CorrectnessVerdict",
    "judge_faithfulness",
    "judge_relevancy",
    "judge_correctness",
    "run_judges",
    "JudgeReport",
]


# --------------------------------------------------------------------------- #
# QADataset (replaces EmbeddingQAFinetuneDataset, same JSON on disk)
# --------------------------------------------------------------------------- #


@dataclass
class QADataset:
    """Question → gold-chunk dataset for retrieval eval and embedding fine-tuning.

    Fields match the legacy JSON exactly: ``queries`` (query_id → question),
    ``corpus`` (chunk_id → chunk text), ``relevant_docs`` (query_id → gold
    chunk ids). ``QADataset.load("rag_eval_dataset.json")`` opens the files the
    old notebooks saved with ``EmbeddingQAFinetuneDataset.save_json``.
    """

    queries: dict[str, str] = field(default_factory=dict)
    corpus: dict[str, str] = field(default_factory=dict)
    relevant_docs: dict[str, list[str]] = field(default_factory=dict)
    mode: str = "text"

    def __len__(self) -> int:
        return len(self.queries)

    def __repr__(self) -> str:
        return (
            f"QADataset({len(self.queries)} questions over {len(self.corpus)} chunks)"
        )

    def save(self, path: str | Path) -> None:
        """Write the legacy-compatible JSON (old notebooks can read it back)."""
        payload = {
            "queries": self.queries,
            "corpus": self.corpus,
            "relevant_docs": self.relevant_docs,
            "mode": self.mode,
        }
        Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> QADataset:
        """Read a QA dataset — ours or a legacy ``EmbeddingQAFinetuneDataset`` file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        missing = [k for k in ("queries", "corpus", "relevant_docs") if k not in data]
        if missing:
            raise TaiAitutorError(
                f"{path} is missing key(s) {missing} — expected the "
                "queries/corpus/relevant_docs QA dataset shape."
            )
        relevant = {
            qid: [docs] if isinstance(docs, str) else list(docs)
            for qid, docs in data["relevant_docs"].items()
        }
        return cls(
            queries=dict(data["queries"]),
            corpus=dict(data["corpus"]),
            relevant_docs=relevant,
            mode=data.get("mode", "text"),
        )

    def sample(self, n: int, seed: int = 42) -> QADataset:
        """A deterministic n-question subset (the old ``subset_50`` files, reproducibly)."""
        if n >= len(self.queries):
            return self
        picked = random.Random(seed).sample(sorted(self.queries), n)
        queries = {qid: self.queries[qid] for qid in picked}
        relevant = {qid: self.relevant_docs.get(qid, []) for qid in picked}
        needed = {doc_id for docs in relevant.values() for doc_id in docs}
        corpus = {doc_id: text for doc_id, text in self.corpus.items() if doc_id in needed}
        return QADataset(queries=queries, corpus=corpus, relevant_docs=relevant, mode=self.mode)


class _GeneratedQuestions(BaseModel):
    questions: list[str] = Field(description="Self-contained exam questions for this excerpt.")


def make_qa_pairs(
    source,
    n_chunks: int = 25,
    questions_per_chunk: int = 1,
    *,
    model: str | None = None,
    provider: str | None = None,
    seed: int = 42,
    concurrency: int = 8,
    show_progress: bool = True,
) -> QADataset:
    """Generate a synthetic QA eval set from chunks — one typed LLM call per chunk.

    Replaces ``generate_question_context_pairs``. ``source`` is a Chroma
    collection or a list of :class:`Chunk` objects; ``n_chunks`` are sampled
    deterministically (``seed``) so re-runs build the same dataset.
    """
    chunks: list[Chunk] = source if isinstance(source, list) else get_all_chunks(source)
    if not chunks:
        raise TaiAitutorError("make_qa_pairs got no chunks — ingest something first.")
    if n_chunks < len(chunks):
        chunks = random.Random(seed).sample(chunks, n_chunks)

    dataset = QADataset()

    def one(chunk: Chunk) -> tuple[Chunk, list[str]]:
        prompt = (
            f"Context excerpt:\n---------------------\n{chunk.text}\n---------------------\n\n"
            f"Write exactly {questions_per_chunk} question(s) answerable only from this excerpt."
        )
        result = extract(
            prompt,
            _GeneratedQuestions,
            system=prompts.QA_GENERATION_SYSTEM,
            model=model,
            provider=provider,
        )
        return chunk, result.questions[:questions_per_chunk]

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(one, c) for c in chunks]
        iterator = futures
        if show_progress:
            from tqdm.auto import tqdm

            iterator = tqdm(futures, total=len(futures), desc="make_qa_pairs")
        for future in iterator:
            chunk, questions = future.result()
            dataset.corpus[chunk.id] = chunk.text
            for j, question in enumerate(questions):
                qid = f"{chunk.id}-q{j}"
                dataset.queries[qid] = question.strip()
                dataset.relevant_docs[qid] = [chunk.id]

    return dataset


# --------------------------------------------------------------------------- #
# Retrieval metrics (mirror production retrieval_metrics(): rank of the gold doc)
# --------------------------------------------------------------------------- #


@dataclass
class QueryResult:
    """One query's retrieval outcome."""

    query_id: str
    question: str
    gold_ids: list[str]
    retrieved_ids: list[str]
    first_relevant_rank: int | None  # 1-based; None = gold never retrieved

    @property
    def hit(self) -> bool:
        return self.first_relevant_rank is not None

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.first_relevant_rank if self.first_relevant_rank else 0.0


@dataclass
class RetrievalReport:
    """Hit rate + MRR over a QADataset, with the per-query table kept for inspection."""

    hit_rate: float
    mrr: float
    top_k: int
    per_query: list[QueryResult] = field(default_factory=list)

    @property
    def n_queries(self) -> int:
        return len(self.per_query)

    def misses(self) -> list[QueryResult]:
        """The queries whose gold chunk was never retrieved — read these, always."""
        return [r for r in self.per_query if not r.hit]

    def __repr__(self) -> str:
        return (
            f"RetrievalReport(top_k={self.top_k}, n={self.n_queries}, "
            f"hit_rate={self.hit_rate:.3f}, mrr={self.mrr:.3f})"
        )


def _retrieved_ids(result) -> list[str]:
    """Accept search() output (ScoredChunk list) or a plain list of id strings."""
    ids = []
    for item in result:
        if isinstance(item, ScoredChunk):
            ids.append(item.id)
        elif isinstance(item, Chunk):
            ids.append(item.id)
        else:
            ids.append(str(item))
    return ids


def _make_search_fn(search_fn, collection) -> Callable[[str, int], list]:
    if search_fn is not None:
        return search_fn
    if collection is None:
        raise TaiAitutorError(
            "Pass search_fn=(query, top_k) -> hits, or collection=... to evaluate "
            "the default dense search."
        )
    return lambda query, top_k: search(query, collection, top_k=top_k)


def evaluate_retrieval(
    qa: QADataset,
    search_fn: Callable[[str, int], list] | None = None,
    collection=None,
    top_k: int = 5,
    show_progress: bool = False,
) -> RetrievalReport:
    """Run every question through the retriever; score hit rate and MRR in one pass.

    Replaces ``RetrieverEvaluator.from_metric_names(["mrr", "hit_rate"], ...)``
    + ``aevaluate_dataset``. ``search_fn`` is any ``(query, top_k) -> hits``
    callable — dense today, hybrid or reranked in the Section 7 lessons, so the
    same eval measures every retriever variant (and the reranker actually gets
    measured, unlike the old notebook bug).
    """
    fn = _make_search_fn(search_fn, collection)
    items: Iterable = qa.queries.items()
    if show_progress:
        from tqdm.auto import tqdm

        items = tqdm(list(items), desc=f"evaluate_retrieval top_k={top_k}")

    per_query: list[QueryResult] = []
    for qid, question in items:
        gold = set(qa.relevant_docs.get(qid, []))
        retrieved = _retrieved_ids(fn(question, top_k))[:top_k]
        rank = next((i + 1 for i, rid in enumerate(retrieved) if rid in gold), None)
        per_query.append(
            QueryResult(
                query_id=qid,
                question=question,
                gold_ids=sorted(gold),
                retrieved_ids=retrieved,
                first_relevant_rank=rank,
            )
        )

    n = len(per_query) or 1
    return RetrievalReport(
        hit_rate=sum(r.hit for r in per_query) / n,
        mrr=sum(r.reciprocal_rank for r in per_query) / n,
        top_k=top_k,
        per_query=per_query,
    )


def hit_rate(qa: QADataset, search_fn=None, collection=None, top_k: int = 5) -> float:
    """Fraction of questions whose gold chunk appears in the top-k results.

    (Needs MRR too? :func:`evaluate_retrieval` computes both in one retrieval pass.)
    """
    return evaluate_retrieval(qa, search_fn, collection, top_k).hit_rate


def mrr(qa: QADataset, search_fn=None, collection=None, top_k: int = 5) -> float:
    """Mean reciprocal rank of the first gold chunk in the top-k results."""
    return evaluate_retrieval(qa, search_fn, collection, top_k).mrr


# --------------------------------------------------------------------------- #
# LLM judges (typed verdicts; replace the evaluator classes)
# --------------------------------------------------------------------------- #


class FaithfulnessVerdict(BaseModel):
    """Is every claim in the answer supported by the retrieved context?"""

    faithful: bool
    reasoning: str


class RelevancyVerdict(BaseModel):
    """Do the answer and its context actually address the question?"""

    relevant: bool
    reasoning: str


class CorrectnessVerdict(BaseModel):
    """1-5 score of the answer against a reference answer (>= 4.0 passes)."""

    score: float = Field(ge=1.0, le=5.0)
    reasoning: str

    @property
    def passing(self) -> bool:
        return self.score >= 4.0


def _context_text(context) -> str:
    """Accept a string, list of strings, or retrieval hits as judge context."""
    if isinstance(context, str):
        return context
    parts = []
    for item in context:
        parts.append(item.text if hasattr(item, "text") else str(item))
    return "\n\n".join(parts)


def judge_faithfulness(
    answer: str,
    context,
    *,
    model: str | None = None,
    provider: str | None = None,
) -> FaithfulnessVerdict:
    """Replaces ``FaithfulnessEvaluator.evaluate_response`` with one typed call."""
    prompt = (
        f"CONTEXT:\n{_context_text(context)}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Is the answer faithful to the context?"
    )
    return extract(
        prompt, FaithfulnessVerdict, system=prompts.FAITHFULNESS_JUDGE, model=model, provider=provider
    )


def judge_relevancy(
    question: str,
    answer: str,
    context,
    *,
    model: str | None = None,
    provider: str | None = None,
) -> RelevancyVerdict:
    """Replaces ``RelevancyEvaluator`` with one typed call."""
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{_context_text(context)}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Do the answer and context address the question?"
    )
    return extract(
        prompt, RelevancyVerdict, system=prompts.RELEVANCY_JUDGE, model=model, provider=provider
    )


def judge_correctness(
    question: str,
    answer: str,
    reference: str,
    *,
    model: str | None = None,
    provider: str | None = None,
) -> CorrectnessVerdict:
    """Replaces ``CorrectnessEvaluator`` with one typed call (1-5, >= 4.0 passes)."""
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"GENERATED ANSWER:\n{answer}\n\n"
        "Score the generated answer against the reference."
    )
    return extract(
        prompt, CorrectnessVerdict, system=prompts.CORRECTNESS_JUDGE, model=model, provider=provider
    )


# --------------------------------------------------------------------------- #
# Batch judging (replaces BatchEvalRunner — threads, no asyncio)
# --------------------------------------------------------------------------- #

_JUDGES = {
    "faithfulness": lambda row, kw: judge_faithfulness(row["answer"], row["context"], **kw),
    "relevancy": lambda row, kw: judge_relevancy(
        row["question"], row["answer"], row["context"], **kw
    ),
    "correctness": lambda row, kw: judge_correctness(
        row["question"], row["answer"], row["reference"], **kw
    ),
}


@dataclass
class JudgeReport:
    """Aggregated judge verdicts over many rows, with every verdict kept."""

    verdicts: dict[str, list]  # judge name -> verdict per row (aligned with rows)
    n_rows: int

    @property
    def faithfulness_rate(self) -> float | None:
        return self._rate("faithfulness", lambda v: v.faithful)

    @property
    def relevancy_rate(self) -> float | None:
        return self._rate("relevancy", lambda v: v.relevant)

    @property
    def correctness_mean(self) -> float | None:
        scores = [v.score for v in self.verdicts.get("correctness", [])]
        return sum(scores) / len(scores) if scores else None

    @property
    def correctness_pass_rate(self) -> float | None:
        return self._rate("correctness", lambda v: v.passing)

    def _rate(self, judge: str, predicate) -> float | None:
        verdicts = self.verdicts.get(judge)
        if not verdicts:
            return None
        return sum(bool(predicate(v)) for v in verdicts) / len(verdicts)

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, value in (
            ("faithfulness_rate", self.faithfulness_rate),
            ("relevancy_rate", self.relevancy_rate),
            ("correctness_mean", self.correctness_mean),
            ("correctness_pass_rate", self.correctness_pass_rate),
        ):
            if value is not None:
                out[name] = round(value, 4)
        return out

    def __repr__(self) -> str:
        stats = ", ".join(f"{k}={v}" for k, v in self.summary().items())
        return f"JudgeReport(n={self.n_rows}, {stats})"


def _normalize_row(row) -> dict:
    """Rows are dicts; an Answer in ``answer`` contributes its text and sources."""
    row = dict(row)
    answer = row.get("answer")
    if hasattr(answer, "text"):  # synthesis.Answer
        if "context" not in row and getattr(answer, "sources", None):
            row["context"] = answer.sources
        row["answer"] = answer.text
    row.setdefault("context", "")
    return row


def run_judges(
    rows: list[dict],
    judges: tuple[str, ...] = ("faithfulness", "relevancy"),
    *,
    model: str | None = None,
    provider: str | None = None,
    concurrency: int = 8,
    show_progress: bool = True,
) -> JudgeReport:
    """Judge many (question, answer, context[, reference]) rows concurrently.

    Replaces ``BatchEvalRunner(...).aevaluate_queries`` — a thread pool over
    typed judge calls, so there is no asyncio and no ``nest_asyncio`` cell.

    Each row is a dict with ``question``, ``answer`` (string or an
    :class:`~tai_aitutor.synthesis.Answer`, whose sources become the context),
    optional ``context``, and ``reference`` (required for ``"correctness"``).
    """
    unknown = [j for j in judges if j not in _JUDGES]
    if unknown:
        raise TaiAitutorError(f"Unknown judge(s) {unknown}. Available: {sorted(_JUDGES)}")
    normalized = [_normalize_row(r) for r in rows]
    if "correctness" in judges:
        missing = [i for i, r in enumerate(normalized) if "reference" not in r]
        if missing:
            raise TaiAitutorError(
                f"correctness judging needs a 'reference' answer; rows {missing[:5]} lack one."
            )

    kw = {"model": model, "provider": provider}
    results: dict[str, list] = {j: [None] * len(normalized) for j in judges}

    def work(judge_name: str, index: int):
        results[judge_name][index] = _JUDGES[judge_name](normalized[index], kw)

    tasks = [(j, i) for j in judges for i in range(len(normalized))]
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(work, j, i) for j, i in tasks]
        iterator = futures
        if show_progress and futures:
            from tqdm.auto import tqdm

            iterator = tqdm(futures, total=len(futures), desc="run_judges")
        for future in iterator:
            future.result()

    return JudgeReport(verdicts=results, n_rows=len(normalized))
