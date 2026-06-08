from __future__ import annotations

import os

import pytest

from qwentts_cpp import LibraryNotFoundError, QwenLibrary


def test_missing_library_reports_search_paths(monkeypatch):
    monkeypatch.delenv("QWENTTS_CPP_LIBRARY", raising=False)
    monkeypatch.delenv("QWEN_LIBRARY_PATH", raising=False)
    with pytest.raises(LibraryNotFoundError, match="libqwen"):
        QwenLibrary("/definitely/missing/libqwen.so")


def test_loads_library_from_env_when_available():
    path = os.environ.get("QWENTTS_CPP_LIBRARY")
    if not path:
        pytest.skip("QWENTTS_CPP_LIBRARY not set")
    lib = QwenLibrary(path)
    assert lib.version()

