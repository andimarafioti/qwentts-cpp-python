"""Check diagnostics in the pinned native source after wheel-build patching."""

import os
from pathlib import Path
import shutil

import pytest

from scripts.build_native import native_diagnostic_compatibility


def test_missing_tokenizer_remains_an_error(tmp_path):
    source = os.environ.get("QWENTTS_CPP_SOURCE")
    if not source:
        pytest.skip("QWENTTS_CPP_SOURCE not set")

    shutil.copytree(Path(source).resolve() / "src", tmp_path / "src")
    with native_diagnostic_compatibility(tmp_path):
        bpe = (tmp_path / "src" / "bpe.h").read_text()
        assert 'qt_log(QT_LOG_ERROR, "[BPE] Tokenizer not found in %s", gguf_path);' in bpe
