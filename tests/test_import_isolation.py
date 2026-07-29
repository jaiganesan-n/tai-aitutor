"""The package must import and run WITHOUT cohere / sentence-transformers installed.

These are lesson-specific heavies: nothing may import them at module import time,
and calling into them without an install must raise the friendly extras message —
their absence is simulated even when they happen to be installed.
"""

from __future__ import annotations

import builtins
import subprocess
import sys

import pytest

import tai_aitutor as tai
from tai_aitutor.errors import ProviderNotInstalledError

BLOCKED = ("cohere", "sentence_transformers", "datasets", "accelerate")

_SUBPROCESS_CHECK = r"""
import builtins, sys
BLOCKED = {"cohere", "sentence_transformers", "datasets", "accelerate"}
real_import = builtins.__import__
def blocking_import(name, *args, **kwargs):
    if name.split(".")[0] in BLOCKED:
        raise ImportError(f"No module named {name!r} (blocked)")
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocking_import

import tai_aitutor  # must not touch any blocked module at import time
assert not (BLOCKED & {m.split(".")[0] for m in sys.modules}), "blocked SDK was imported"
print("IMPORT-CLEAN", tai_aitutor.__version__)
"""


def test_package_imports_without_optional_sdks():
    """Fresh interpreter, blocked SDKs → `import tai_aitutor` must still succeed."""
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_CHECK], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "IMPORT-CLEAN" in result.stdout


@pytest.fixture
def no_optional_sdks(monkeypatch):
    for name in list(sys.modules):
        if name.split(".")[0] in BLOCKED:
            monkeypatch.delitem(sys.modules, name, raising=False)
    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"No module named {name!r} (blocked by test)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)


def test_calls_raise_friendly_extras_errors(no_optional_sdks):
    with pytest.raises(ProviderNotInstalledError) as err:
        tai.embed_cohere("text")
    assert "tai-aitutor[rerank]" in str(err.value)

    with pytest.raises(ProviderNotInstalledError) as err:
        tai.embed_local("text")
    assert "tai-aitutor[local]" in str(err.value)

    with pytest.raises(ProviderNotInstalledError) as err:
        tai.train_embedder(
            tai.QADataset(queries={"q": "?"}, corpus={"c": "t"}, relevant_docs={"q": ["c"]})
        )
    assert "tai-aitutor[finetune]" in str(err.value)
