"""Pretty notebook output for chunks and answers (quality-of-life only).

No logic lives here — everything prints what the data already says, as
markdown in notebooks and plain text elsewhere.
"""

from __future__ import annotations

from .retrieval import ScoredChunk
from .synthesis import Answer

__all__ = ["show_chunks", "show_answer", "show_eval_table"]


def _render(markdown: str) -> None:
    try:
        from IPython.display import Markdown, display

        display(Markdown(markdown))
    except Exception:
        print(markdown)


def _label(metadata: dict, fallback: str) -> str:
    """Best human label for a chunk: title → name → source_name → source → id."""
    return (
        metadata.get("title")
        or metadata.get("name")
        or metadata.get("source_name")
        or metadata.get("source")
        or fallback
    )


def show_chunks(hits: list[ScoredChunk], max_chars: int = 300) -> None:
    """Readable listing of retrieval hits: rank, score, source, preview."""
    lines = []
    for hit in hits:
        metadata = hit.metadata or {}
        title = _label(metadata, hit.id)
        preview = hit.text[:max_chars].replace("\n", " ")
        ellipsis = "…" if len(hit.text) > max_chars else ""
        lines.append(f"**{hit.rank}. {title}** — score `{hit.score:.3f}`\n\n> {preview}{ellipsis}\n")
    _render("\n".join(lines) if lines else "_no results_")


def show_eval_table(reports: dict, extra_columns: dict | None = None) -> None:
    """Compare retrieval configurations side by side (the lesson ablation tables).

    ``reports`` maps a row label to a ``RetrievalReport`` — e.g. the Hybrid
    Search lesson's four rows: dense only, BM25 only, fused, fused + rerank.

    ``extra_columns`` adds columns without dropping to pandas:
    ``{column_name: {row_label: value}}`` — e.g. the context-token bill::

        show_eval_table(reports, extra_columns={
            "avg ctx tokens": {label: f"{r.avg_context_tokens(qa):.0f}"
                               for label, r in reports.items()},
        })
    """
    extra_columns = extra_columns or {}
    header = ["configuration", "hit rate", "MRR", "top_k", "queries", *extra_columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for label, report in reports.items():
        row = [
            str(label),
            f"{report.hit_rate:.3f}",
            f"{report.mrr:.3f}",
            str(report.top_k),
            str(report.n_queries),
        ]
        for column in extra_columns.values():
            value = column.get(label, "")
            row.append(f"{value}" if not isinstance(value, float) else f"{value:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    _render("\n".join(lines))


def show_answer(ans: Answer, show_sources: bool = True, max_chars: int = 160) -> None:
    """The answer, then its sources — the two things worth seeing after answer()."""
    parts = [ans.text]
    if show_sources and ans.sources:
        parts.append("\n**Sources**\n")
        for hit in ans.sources:
            metadata = hit.metadata or {}
            title = _label(metadata, hit.id)
            url = metadata.get("url")
            label = f"[{title}]({url})" if url else title
            preview = hit.text[:max_chars].replace("\n", " ")
            parts.append(f"- **[{hit.rank}]** {label} — `{hit.score:.3f}` — {preview}…")
    _render("\n".join(parts))
