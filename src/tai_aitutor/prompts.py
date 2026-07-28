"""Shared prompt text, as plain readable constants.

Replaces LlamaIndex's ``PromptTemplate`` objects and its hidden internal RAG
prompts — the course shows every prompt as a visible f-string; these are the
ones reused across notebooks.
"""

RAG_SYSTEM = (
    "You are an AI tutor answering questions from students of an applied AI course. "
    "Answer using ONLY the numbered context excerpts provided. "
    "If the excerpts do not contain the answer, say you don't know — do not guess."
)

RAG_SYSTEM_CITED = (
    RAG_SYSTEM
    + " After each claim, cite the excerpt(s) you used with bracketed numbers like [1] or [1][3]."
)


def context_block(index: int, text: str, title: str | None = None, url: str | None = None) -> str:
    """Format one retrieved chunk for the prompt: numbered, titled, sourced."""
    header = f"[{index}]"
    if title:
        header += f" {title}"
    if url:
        header += f" ({url})"
    return f"{header}\n{text}"


# --------------------------------------------------------------------------- #
# Evaluation prompts (built in: "Evaluating Your RAG Pipeline", Section 4)
# --------------------------------------------------------------------------- #

QA_GENERATION_SYSTEM = (
    "You are a professor writing exam questions. Given a context excerpt from course "
    "material, write questions that can be answered using ONLY that excerpt. "
    "Rules: the questions must be self-contained (never refer to 'the context', "
    "'the excerpt', or 'the passage'), must vary in phrasing, and must not be "
    "answerable from general knowledge alone."
)

FAITHFULNESS_JUDGE = (
    "You are grading whether an ANSWER is faithful to the CONTEXT it was generated from. "
    "Faithful means every factual claim in the answer is supported by the context — "
    "no invented facts, no outside knowledge presented as if it came from the context. "
    "Refusals ('I don't know') are faithful. Set faithful=false if ANY claim lacks support, "
    "and explain which claim in the reasoning."
)

RELEVANCY_JUDGE = (
    "You are grading whether an ANSWER, produced from retrieved CONTEXT, actually addresses "
    "the QUESTION asked. Set relevant=false when the answer talks past the question, answers "
    "a different question, or the retrieved context was about something else entirely. "
    "A correct refusal ('the context doesn't cover this') counts as relevant when the "
    "context truly doesn't cover it."
)

CORRECTNESS_JUDGE = (
    "You are grading a generated ANSWER against a REFERENCE answer for the same question. "
    "Score 1.0-5.0: 5 = fully correct and complete vs the reference; 4 = correct with minor "
    "omissions; 3 = partially correct with real gaps; 2 = mostly incorrect; 1 = wrong or "
    "contradicts the reference. Judge substance, not wording. "
    "A score of 4.0 or above counts as passing."
)

SITUATE_CHUNK = (
    "You are indexing course material for retrieval. Given a full DOCUMENT and one CHUNK "
    "from it, write 1-2 sentences that situate the chunk within the document (what it is "
    "about and where it fits), so the chunk is easier to find with search. "
    "Answer with the situating context only."
)
