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


def show_chunks(hits: list[ScoredChunk], max_chars: int = 300) -> None:
    """Readable listing of retrieval hits: rank, score, source, preview."""
    lines = []
    for hit in hits:
        metadata = hit.metadata or {}
        title = metadata.get("title") or metadata.get("source_name") or metadata.get("source") or hit.id
        preview = hit.text[:max_chars].replace("\n", " ")
        ellipsis = "…" if len(hit.text) > max_chars else ""
        lines.append(f"**{hit.rank}. {title}** — score `{hit.score:.3f}`\n\n> {preview}{ellipsis}\n")
    _render("\n".join(lines) if lines else "_no results_")


def show_eval_table(reports: dict) -> None:
    """Compare retrieval configurations side by side (the lesson ablation tables).

    ``reports`` maps a row label to a ``RetrievalReport`` — e.g. the Hybrid
    Search lesson's four rows: dense only, BM25 only, fused, fused + rerank.
    """
    lines = [
        "| configuration | hit rate | MRR | top_k | queries |",
        "|---|---|---|---|---|",
    ]
    for label, report in reports.items():
        lines.append(
            f"| {label} | {report.hit_rate:.3f} | {report.mrr:.3f} "
            f"| {report.top_k} | {report.n_queries} |"
        )
    _render("\n".join(lines))


def show_answer(ans: Answer, show_sources: bool = True, max_chars: int = 160) -> None:
    """The answer, then its sources — the two things worth seeing after answer()."""
    parts = [ans.text]
    if show_sources and ans.sources:
        parts.append("\n**Sources**\n")
        for hit in ans.sources:
            metadata = hit.metadata or {}
            title = metadata.get("title") or metadata.get("source_name") or hit.id
            url = metadata.get("url")
            label = f"[{title}]({url})" if url else title
            preview = hit.text[:max_chars].replace("\n", " ")
            parts.append(f"- **[{hit.rank}]** {label} — `{hit.score:.3f}` — {preview}…")
    _render("\n".join(parts))
