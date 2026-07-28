"""Embedding fine-tuning with sentence-transformers (the standard practice).

Built in: "Fine-Tuning an Embedding Model" (Section 8). This is a METHOD
UPGRADE, not a port: the old notebook trained a linear adapter on a frozen
base with LlamaIndex-only machinery (``EmbeddingAdapterFinetuneEngine`` /
``AdapterEmbeddingModel``); per the course plan we fine-tune the base model
itself (``bge-small-en-v1.5``) with the sentence-transformers trainer and
MultipleNegativesRankingLoss, then measure before/after with the course's own
hit rate and MRR.

The honest paragraph from the lesson still applies: production did not
fine-tune an embedder — it upgraded to a stronger off-the-shelf one. Fine-tune
when your domain vocabulary is truly unusual; measure either way.

Requires the ``finetune`` extra: ``pip install 'tai-aitutor[finetune]'``.
"""

from __future__ import annotations

from .embeddings import embed_local
from .errors import ProviderNotInstalledError, TaiAitutorError
from .evals import QADataset, QueryResult, RetrievalReport, make_qa_pairs

__all__ = ["make_training_pairs", "train_embedder", "evaluate_embedder"]


def make_training_pairs(
    chunks,
    questions_per_chunk: int = 2,
    *,
    model: str | None = None,
    provider: str | None = None,
    seed: int = 42,
    concurrency: int = 8,
    show_progress: bool = True,
) -> QADataset:
    """Synthetic (question, positive-chunk) pairs for training — the eval-lesson
    generator reused (same typed call, same ``QADataset`` container).

    Replaces the commented-out ``generate_qa_embedding_pairs`` era: build a
    train set from one slice of chunks and a validation set from another, save
    both with ``.save()`` (legacy-compatible JSON).
    """
    chunks = list(chunks)
    return make_qa_pairs(
        chunks,
        n_chunks=len(chunks),
        questions_per_chunk=questions_per_chunk,
        model=model,
        provider=provider,
        seed=seed,
        concurrency=concurrency,
        show_progress=show_progress,
    )


def _training_rows(qa: QADataset) -> tuple[list[str], list[str]]:
    """(anchor, positive) rows for MultipleNegativesRankingLoss — one per query.

    MNRL treats every other in-batch positive as a negative, which is why no
    explicit negatives are mined here (the lesson explains this trade).
    """
    anchors: list[str] = []
    positives: list[str] = []
    for query_id, question in qa.queries.items():
        gold_ids = qa.relevant_docs.get(query_id, [])
        if not gold_ids:
            continue
        positive = qa.corpus.get(gold_ids[0])
        if not positive:
            continue
        anchors.append(question)
        positives.append(positive)
    if not anchors:
        raise TaiAitutorError("No usable (question, chunk) pairs in this QADataset.")
    return anchors, positives


def train_embedder(
    train: QADataset,
    val: QADataset | None = None,
    base_model: str = "BAAI/bge-small-en-v1.5",
    *,
    epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    out_dir: str = "ft-embedder",
) -> str:
    """Fine-tune ``base_model`` on (question, chunk) pairs; returns the saved model path.

    sentence-transformers trainer + MultipleNegativesRankingLoss — the standard
    recipe. With ``val=``, an information-retrieval evaluator runs during
    training. Compare before/after with :func:`evaluate_embedder`.
    """
    try:
        from datasets import Dataset
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
            losses,
        )
    except ImportError as exc:
        raise ProviderNotInstalledError(
            "Fine-tuning needs sentence-transformers + datasets + accelerate. "
            "Run: pip install 'tai-aitutor[finetune]'"
        ) from exc

    anchors, positives = _training_rows(train)
    train_dataset = Dataset.from_dict({"anchor": anchors, "positive": positives})

    model = SentenceTransformer(base_model)
    loss = losses.MultipleNegativesRankingLoss(model)

    evaluator = None
    if val is not None:
        from sentence_transformers.evaluation import InformationRetrievalEvaluator

        evaluator = InformationRetrievalEvaluator(
            queries=val.queries,
            corpus=val.corpus,
            relevant_docs={qid: set(ids) for qid, ids in val.relevant_docs.items()},
            name="val",
        )

    args = SentenceTransformerTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=25,
        save_strategy="no",
        report_to=[],
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_dataset, loss=loss, evaluator=evaluator
    )
    trainer.train()
    model.save_pretrained(out_dir)
    return out_dir


def evaluate_embedder(
    model_name_or_path: str,
    qa: QADataset,
    top_k: int = 5,
    batch_size: int = 32,
) -> RetrievalReport:
    """Hit rate + MRR for an embedding model over a QADataset — the before/after measure.

    Embeds the dataset's own corpus and queries with the given model (base or
    fine-tuned; e5-style prefixes handled by ``embed_local``), ranks by cosine
    similarity, and scores with the same metrics as the retrieval lessons, so
    the fine-tuning table and the pipeline tables are the same ruler.
    """
    corpus_ids = list(qa.corpus)
    corpus_vectors = embed_local(
        [qa.corpus[chunk_id] for chunk_id in corpus_ids],
        model_name=model_name_or_path,
        task="document",
        batch_size=batch_size,
    )
    query_ids = list(qa.queries)
    query_vectors = embed_local(
        [qa.queries[query_id] for query_id in query_ids],
        model_name=model_name_or_path,
        task="query",
        batch_size=batch_size,
    )

    per_query: list[QueryResult] = []
    for query_id, query_vector in zip(query_ids, query_vectors):
        scored = sorted(
            (
                (_dot(query_vector, corpus_vector), chunk_id)
                for chunk_id, corpus_vector in zip(corpus_ids, corpus_vectors)
            ),
            key=lambda pair: (-pair[0], pair[1]),
        )[:top_k]
        retrieved = [chunk_id for _, chunk_id in scored]
        gold = set(qa.relevant_docs.get(query_id, []))
        rank = next((i + 1 for i, cid in enumerate(retrieved) if cid in gold), None)
        per_query.append(
            QueryResult(
                query_id=query_id,
                question=qa.queries[query_id],
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


def _dot(a: list[float], b: list[float]) -> float:
    """Cosine similarity for normalised vectors is just the dot product."""
    return sum(x * y for x, y in zip(a, b))
