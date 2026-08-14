#!/usr/bin/env python3
"""Diff each kept package function against the notebook cell that teaches it.

The rule the package lives by: every public symbol has an *identical inline twin*
in a Section 1-8 lesson. This script checks that mechanically. For each mapping
below it extracts the named function from the notebook's code cells and from the
package source, normalises both, and reports whether they match.

Usage::

    python scripts/diff_taught_code.py --notebooks /path/to/ai-tutor-rag-system/notebooks
    python scripts/diff_taught_code.py --notebooks ... --symbol embed --show-diff

Exit code is 1 if any symbol is missing from either side, so this can become a
CI gate once the notebook repo is reachable from CI.

Exceptions are declared in EXPECTED_DIFFERENCES, with a reason each. An
undeclared difference is a finding; a declared one is printed but not fatal.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import sys
from pathlib import Path

#: symbol -> (package module, notebook filename that teaches it)
TAUGHT_IN: dict[str, tuple[str, str]] = {
    "generate": ("llm.py", "02-Basic_RAG.ipynb"),
    "extract": ("llm.py", "Applied_Structured_Outputs.ipynb"),
    "embed": ("embeddings.py", "02-Basic_RAG.ipynb"),
    "embed_cohere": ("embeddings.py", "Selecting_Embedding_Models.ipynb"),
    "embed_local": ("embeddings.py", "Selecting_Embedding_Models.ipynb"),
    "n_tokens": ("tokens.py", "03-From_Script_to_Pipeline.ipynb"),
    "chunk": ("chunking.py", "03-From_Script_to_Pipeline.ipynb"),
    "heading_aware_markdown_chunks": ("chunking.py", "03-From_Script_to_Pipeline.ipynb"),
    "search": ("retrieval.py", "04-RAG_with_VectorStore.ipynb"),
    "build_where_filter": ("vectorstore.py", "Metadata_Filtering.ipynb"),
    "rrf_fuse": ("retrieval.py", "10-Adding_Reranking.ipynb"),
    "rerank": ("retrieval.py", "10-Adding_Reranking.ipynb"),
    "judge_rerank": ("retrieval.py", "17-Using_LLMs_to_rank_chunks_as_the_Judge.ipynb"),
    "hyde_search": ("retrieval.py", "12-Improve_Query.ipynb"),
    "hit_rate": ("evals.py", "06-Evaluate_RAG.ipynb"),
    "reciprocal_rank": ("evals.py", "06-Evaluate_RAG.ipynb"),
    "judge_faithfulness": ("evals.py", "06-Evaluate_RAG.ipynb"),
    "judge_relevancy": ("evals.py", "06-Evaluate_RAG.ipynb"),
    "judge_correctness": ("evals.py", "06-Evaluate_RAG.ipynb"),
    "build_rag_prompt": ("synthesis.py", "05-Improve_Prompts_+_Add_Source.ipynb"),
    "show_chunks": ("display.py", "10-Adding_Reranking.ipynb"),
    "show_answer": ("display.py", "15-Use_OpenSource_Models.ipynb"),
    "code_tokenize": ("retrieval.py", "11-Adding_Hybrid_Search.ipynb"),
    "decompose_question": ("retrieval.py", "12-Improve_Query.ipynb"),
}

#: Symbols whose taught twin exists under a different name in the notebook.
#: A rename is itself a finding — it means a student cannot grep for the function
#: they wrote — so these are listed, not hidden.
RENAMED_IN_NOTEBOOK: dict[str, str] = {
    "code_tokenize": "tokenize",
}

#: symbol -> why the package copy legitimately differs from the taught cell.
EXPECTED_DIFFERENCES: dict[str, str] = {
    "generate": "package adds keyword-only extras (provider=, temperature=, retries=) and a docstring",
    "extract": "package adds provider=/retries= and raises a typed StructuredOutputError",
    "embed": "package resolves provider/model from config instead of module-level globals",
    "search": "package takes the collection as an argument; the notebook closes over a global",
    "hyde_search": "package takes the collection as an argument and accepts a caller-supplied hypothetical",
    "judge_rerank": "package returns ScoredChunk objects; the notebook returns dicts",
    "show_chunks": "package renders markdown in notebooks and plain text elsewhere",
    "show_answer": "package renders markdown in notebooks and plain text elsewhere",
    "rerank": "package takes hits as an argument; the notebook closes over a global client",
    "hit_rate": "notebook averages over a dataset; the package metric is per-query (strip spec Fix 4)",
    "reciprocal_rank": "notebook has no reciprocal_rank; mrr() is its corpus-level sibling (strip spec Fix 4)",
    "judge_faithfulness": "package uses extract() with a typed schema; the notebook parses JSON by hand",
    "judge_relevancy": "package uses extract() with a typed schema; the notebook parses JSON by hand",
    "judge_correctness": "package uses extract() with a typed schema; the notebook parses JSON by hand",
    "n_tokens": "package adds an offline len/4 fallback when the tiktoken vocabulary is unreachable",
    "chunk": "package adds the same offline fallback and validates chunk_size/chunk_overlap",
    "heading_aware_markdown_chunks": "package adds the offline fallback and size validation",
    "rrf_fuse": "notebook fuses id lists and returns a Counter; the package fuses and returns ScoredChunks",
    "embed_cohere": "package injects the client and scrubs newlines; the notebook uses a module-level client",
    "embed_local": "package caches the loaded model and injects the query instruction map",
    "build_where_filter": "package parameterises the metadata key; the notebook hard-codes 'source'",
    "code_tokenize": "notebook names it tokenize(); the package name is code_tokenize",
}


def function_source(source: str, name: str) -> str | None:
    """Return the source of top-level function ``name``, or None if absent."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    return None


def notebook_code(path: Path) -> str:
    """All code cells of a notebook, concatenated."""
    nb = json.loads(path.read_text())
    return "\n\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def normalise(src: str) -> list[str]:
    """Strip docstrings, comments and blank lines so the comparison is about code."""
    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                body = node.body
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                    if isinstance(body[0].value.value, str):
                        node.body = body[1:] or [ast.Pass()]
        src = ast.unparse(ast.fix_missing_locations(tree))
    except SyntaxError:
        pass
    return [line.rstrip() for line in src.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks", required=True, type=Path, help="notebooks/ directory")
    parser.add_argument("--package", type=Path, default=Path(__file__).parent.parent / "src" / "tai_aitutor")
    parser.add_argument("--symbol", help="check one symbol instead of all")
    parser.add_argument("--show-diff", action="store_true", help="print a unified diff per mismatch")
    args = parser.parse_args()

    targets = {args.symbol: TAUGHT_IN[args.symbol]} if args.symbol else TAUGHT_IN
    identical, declared, undeclared, missing = [], [], [], []

    for symbol, (module, notebook) in sorted(targets.items()):
        pkg_src = function_source((args.package / module).read_text(), symbol)
        nb_path = args.notebooks / notebook
        if not nb_path.exists():
            missing.append(f"{symbol}: notebook {notebook} not found")
            continue
        nb_code = notebook_code(nb_path)
        nb_src = function_source(nb_code, symbol)
        if nb_src is None and symbol in RENAMED_IN_NOTEBOOK:
            nb_src = function_source(nb_code, RENAMED_IN_NOTEBOOK[symbol])
        if pkg_src is None:
            missing.append(f"{symbol}: not defined in {module}")
            continue
        if nb_src is None:
            missing.append(f"{symbol}: no inline twin in {notebook}")
            continue

        a, b = normalise(nb_src), normalise(pkg_src)
        if a == b:
            identical.append(symbol)
        elif symbol in EXPECTED_DIFFERENCES:
            declared.append((symbol, EXPECTED_DIFFERENCES[symbol]))
        else:
            undeclared.append((symbol, notebook, module))
        if args.show_diff and a != b:
            print(f"\n--- {notebook}::{symbol}\n+++ {module}::{symbol}")
            print("\n".join(difflib.unified_diff(a, b, lineterm="", n=1)))

    print(f"\nidentical to the taught cell:  {len(identical)}")
    for s in identical:
        print(f"  = {s}")
    print(f"\ndeclared exceptions:           {len(declared)}")
    for s, why in declared:
        print(f"  ~ {s}: {why}")
    print(f"\nUNDECLARED differences:        {len(undeclared)}")
    for s, nb, mod in undeclared:
        print(f"  ! {s}: {mod} differs from {nb} with no declared reason")
    print(f"\nmissing on one side:           {len(missing)}")
    for m in missing:
        print(f"  ? {m}")

    return 1 if (undeclared or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
