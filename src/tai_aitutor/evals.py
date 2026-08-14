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
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from . import prompts
from .chunking import Chunk
from .errors import TaiAitutorError
from .llm import extract
from .retrieval import ScoredChunk, search

__all__ = [
    "QADataset",
    "hit_rate",
    "reciprocal_rank",
    "evaluate_retrieval",
    "sweep_top_k",
    "RetrievalReport",
    "QueryResult",
    "FaithfulnessVerdict",
    "RelevancyVerdict",
    "CorrectnessVerdict",
    "judge_faithfulness",
    "judge_relevancy",
    "judge_correctness",
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

    def avg_context_tokens(self, qa: QADataset) -> float:
        """Mean tokens of retrieved context per query — the cost column every
        ablation table wants next to hit rate and MRR.

        Prices retrieved ids against ``qa.corpus`` texts; ids outside the eval
        corpus are skipped (they weren't part of the measured set), so treat
        this as the comparable-across-configs number, not an exact API bill —
        :func:`context_tokens` prices actual hits exactly.
        """
        from .tokens import n_tokens

        if not self.per_query:
            return 0.0
        totals = [
            sum(n_tokens(qa.corpus[cid]) for cid in result.retrieved_ids if cid in qa.corpus)
            for result in self.per_query
        ]
        return sum(totals) / len(totals)

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

    Args:
        qa: The evaluation dataset.
        search_fn: Any ``(query, top_k) -> hits`` callable — dense, hybrid or
            reranked, so every retriever variant is measured the same way.
        collection: Used to build a default dense ``search_fn`` when none is given.
        top_k: Cutoff to score at.
        show_progress: Show a progress bar.

    Returns:
        A :class:`RetrievalReport` carrying hit rate, MRR, and the per-query table.

    Raises:
        ValueError: Neither ``search_fn`` nor ``collection`` was given.
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


def _score_ranked(
    qa: QADataset, ranked_ids: dict[str, list[str]], top_k: int
) -> RetrievalReport:
    """Score already-retrieved ranked ids at a given cutoff (no retrieval calls)."""
    per_query: list[QueryResult] = []
    for qid, question in qa.queries.items():
        gold = set(qa.relevant_docs.get(qid, []))
        retrieved = ranked_ids.get(qid, [])[:top_k]
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


def sweep_top_k(
    qa: QADataset,
    k_values,
    search_fn: Callable[[str, int], list] | None = None,
    collection=None,
    show_progress: bool = False,
) -> dict[int, RetrievalReport]:
    """One retrieval pass, every top-k scored from it — the ablation workhorse.

    Retrieves each question ONCE at ``max(k_values)`` and scores every smaller
    cutoff from the same ranked list (a top-k cutoff is just a slice — that's
    the seam the Larger-Context lesson teaches inline before importing this).
    N questions cost N retrievals regardless of how many k values you sweep.

    >>> reports = sweep_top_k(qa, [2, 4, 6, 8, 10], collection=col)
    >>> show_eval_table({f"top_k={k}": r for k, r in reports.items()})

    Args:
        qa: The evaluation dataset.
        k_values: The cutoffs to score.
        search_fn: Any ``(query, top_k) -> hits`` callable.
        collection: Used to build a default dense ``search_fn`` when none is given.
        show_progress: Show a progress bar.

    Returns:
        One :class:`RetrievalReport` per cutoff, keyed by k. Retrieval runs once
        per question at ``max(k_values)``; the smaller cutoffs are slices of it.

    Raises:
        ValueError: ``k_values`` is empty or contains a non-positive value.
    """
    ks = sorted({int(k) for k in k_values})
    if not ks or ks[0] <= 0:
        raise TaiAitutorError(f"k_values must be positive ints, got {list(k_values)!r}")
    fn = _make_search_fn(search_fn, collection)
    max_k = ks[-1]

    items: Iterable = qa.queries.items()
    if show_progress:
        from tqdm.auto import tqdm

        items = tqdm(list(items), desc=f"sweep_top_k max_k={max_k}")

    ranked: dict[str, list[str]] = {
        qid: _retrieved_ids(fn(question, max_k))[:max_k] for qid, question in items
    }
    return {k: _score_ranked(qa, ranked, k) for k in ks}


def hit_rate(relevant_id: str, retrieved_ids: list[str]) -> float:
    """Did the gold chunk appear anywhere in this query's retrieved ids?

    This is the per-query metric. Average it over a dataset yourself, or use
    :func:`evaluate_retrieval`, which reports hit rate and MRR from one pass.

    >>> hit_rate("doc-3", ["doc-9", "doc-3"])
    1.0

    Args:
        relevant_id: The id of the chunk that should have been retrieved.
        retrieved_ids: The retrieved chunk ids, best first.

    Returns:
        ``1.0`` if ``relevant_id`` is present, ``0.0`` otherwise.
    """
    return 1.0 if relevant_id in retrieved_ids else 0.0


def reciprocal_rank(relevant_id: str, retrieved_ids: list[str]) -> float:
    """Where in the retrieved ids did the gold chunk land?

    Mean this over a dataset and you have MRR; :func:`evaluate_retrieval` does
    that for you.

    >>> reciprocal_rank("doc-3", ["doc-9", "doc-3"])
    0.5

    Args:
        relevant_id: The id of the chunk that should have been retrieved.
        retrieved_ids: The retrieved chunk ids, best first.

    Returns:
        ``1 / rank`` using 1-based ranks — ``1.0`` for first place, ``0.5`` for
        second — and ``0.0`` when the gold chunk is absent.
    """
    if relevant_id not in retrieved_ids:
        return 0.0
    return 1.0 / (retrieved_ids.index(relevant_id) + 1)


# --------------------------------------------------------------------------- #
# LLM judges (typed verdicts; replace the evaluator classes)
# --------------------------------------------------------------------------- #

#: A judge should be at least as strong as the model it grades, so judging does
#: not default to the configured chat model. Sourced from the strip spec's
#: Decision 2 quotation; as of 2026-08-13.
# TODO: [NEEDS UPDATE — judge model ids for openai and anthropic | source: internal doc
# (course_update_plan.md v4, Decision 2) | Decision 7 names only the Gemini judge
# (gemini-3.6-flash); the other two providers fall back to the configured chat model
# until the verified ids are supplied.]
JUDGE_MODELS: dict[str, str] = {
    "gemini": "gemini-3.6-flash",
}


def _judge_model(model: str | None, provider: str | None) -> str | None:
    """The judge model for this call: explicit ``model`` wins, else the per-provider
    default, else ``None`` (meaning the configured chat model)."""
    if model is not None:
        return model
    from . import config as _cfg

    prov = provider or _cfg.get_config().provider
    return JUDGE_MODELS.get(prov)



# TODO: [NEEDS UPDATE — decide the verdict schema shape | source: internal doc
# (tai_aitutor_strip_spec.md Fix 6) vs. test run (the swept 06-Evaluate_RAG.ipynb) |
# Fix 6 says 06 uses ONE shared JudgeVerdict(passing, reasoning) for faithfulness and
# relevancy. The swept notebook cell 20 defines TWO classes — FaithfulnessVerdict(faithful,
# reason) and RelevancyVerdict(relevant, reason) — i.e. the shape below, with the field
# named `reason` rather than `reasoning`. Applying Fix 6 literally would make the package
# contradict the cell students just read, so the current shape is left in place pending a
# decision. See the Task 2 report.]
class FaithfulnessVerdict(BaseModel):
    """Is every claim in the answer supported by the retrieved context?"""

    faithful: bool
    reasoning: str


class RelevancyVerdict(BaseModel):
    """Do the answer and its context actually address the question?"""

    relevant: bool
    reasoning: str


# TODO: [NEEDS UPDATE — decide the correctness scale and field names | source: internal doc
# (tai_aitutor_strip_spec.md Fix 6) vs. test run (the swept 06-Evaluate_RAG.ipynb) | Fix 6
# says 06 scores 0-5 ("0: entirely incorrect") with fields score/feedback. The swept notebook
# scores 1-5 (`score: float  # 1 (wrong) .. 5 (fully correct)`, prompt: "Return ONLY JSON:
# {"score": <number 1-5>}") — so the ge=1.0 bound below already matches the taught cell and
# only the field name differs (`feedback` taught vs `reasoning` here). Left as-is pending a
# decision. See the Task 2 report.]
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
    """Replaces ``FaithfulnessEvaluator.evaluate_response`` with one typed call.

    Args:
        answer: The generated answer.
        context: A string, a list of strings, or retrieval hits.
        model: Model id; defaults to the provider's judge model.
        provider: Override the configured provider for this call.

    Returns:
        A :class:`FaithfulnessVerdict` — ``faithful`` plus the reasoning.

    Raises:
        ValueError: The provider is unknown, its SDK is missing, or the response
            could not be parsed into the verdict schema.
    """
    prompt = (
        f"CONTEXT:\n{_context_text(context)}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Is the answer faithful to the context?"
    )
    return extract(
        prompt, FaithfulnessVerdict, system=prompts.FAITHFULNESS_JUDGE, model=_judge_model(model, provider), provider=provider
    )


def judge_relevancy(
    question: str,
    answer: str,
    context,
    *,
    model: str | None = None,
    provider: str | None = None,
) -> RelevancyVerdict:
    """Replaces ``RelevancyEvaluator`` with one typed call.

    Args:
        question: The question asked.
        answer: The generated answer.
        context: A string, a list of strings, or retrieval hits.
        model: Model id; defaults to the provider's judge model.
        provider: Override the configured provider for this call.

    Returns:
        A :class:`RelevancyVerdict` — ``relevant`` plus the reasoning.

    Raises:
        ValueError: The provider is unknown, its SDK is missing, or the response
            could not be parsed into the verdict schema.
    """
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{_context_text(context)}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Do the answer and context address the question?"
    )
    return extract(
        prompt, RelevancyVerdict, system=prompts.RELEVANCY_JUDGE, model=_judge_model(model, provider), provider=provider
    )


def judge_correctness(
    question: str,
    answer: str,
    reference: str,
    *,
    model: str | None = None,
    provider: str | None = None,
) -> CorrectnessVerdict:
    """Replaces ``CorrectnessEvaluator`` with one typed call (1-5, >= 4.0 passes).

    Args:
        question: The question asked.
        answer: The generated answer.
        reference: The gold answer to score against.
        model: Model id; defaults to the provider's judge model.
        provider: Override the configured provider for this call.

    Returns:
        A :class:`CorrectnessVerdict` — a 1-5 ``score`` plus the reasoning;
        ``verdict.passing`` is ``True`` at 4.0 or above.

    Raises:
        ValueError: The provider is unknown, its SDK is missing, or the response
            could not be parsed into the verdict schema.
    """
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"GENERATED ANSWER:\n{answer}\n\n"
        "Score the generated answer against the reference."
    )
    return extract(
        prompt, CorrectnessVerdict, system=prompts.CORRECTNESS_JUDGE, model=_judge_model(model, provider), provider=provider
    )
