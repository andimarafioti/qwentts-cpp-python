from __future__ import annotations

import os

import numpy as np
import pytest

from qwentts_cpp import LibraryNotFoundError, QwenLibrary, load_rvq_codes, load_speaker_embedding
from qwentts_cpp._binding import QtTTSParams


def _pack_rvq_codes(codes, code_bits=11):
    mask = (1 << code_bits) - 1
    total_bits = len(codes) * code_bits
    out = bytearray((total_bits + 7) // 8)
    acc = 0
    bits_in_acc = 0
    out_pos = 0
    for code in codes:
        acc |= (int(code) & mask) << bits_in_acc
        bits_in_acc += code_bits
        while bits_in_acc >= 8:
            out[out_pos] = acc & 0xFF
            out_pos += 1
            acc >>= 8
            bits_in_acc -= 8
    if bits_in_acc > 0:
        out[out_pos] = acc & 0xFF
    return bytes(out)


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


def test_tts_params_contains_abi_v2_latent_tail_fields():
    assert [name for name, _ctype in QtTTSParams._fields_[-4:]] == [
        "ref_spk_emb",
        "ref_spk_dim",
        "ref_codes",
        "ref_T",
    ]


def test_load_speaker_embedding_reads_raw_float32(tmp_path):
    path = tmp_path / "speaker.spk"
    expected = np.array([0.25, -0.5, 1.0], dtype=np.float32)
    expected.tofile(path)

    loaded = load_speaker_embedding(path)

    assert loaded.dtype == np.float32
    assert loaded.flags.c_contiguous
    np.testing.assert_array_equal(loaded, expected)


def test_load_rvq_codes_unpacks_lsb_first_matrix(tmp_path):
    path = tmp_path / "reference.rvq"
    expected = np.array(
        [
            [1, 2, 3],
            [2047, 17, 42],
            [0, 999, 123],
            [456, 789, 1024],
        ],
        dtype=np.int32,
    )
    path.write_bytes(_pack_rvq_codes(expected.reshape(-1).tolist()))

    loaded = load_rvq_codes(path, num_codebooks=expected.shape[0])

    assert loaded.dtype == np.int32
    np.testing.assert_array_equal(loaded, expected)


def test_load_rvq_codes_rejects_wrong_codebook_count(tmp_path):
    path = tmp_path / "reference.rvq"
    path.write_bytes(_pack_rvq_codes([1, 2, 3, 4]))

    with pytest.raises(ValueError, match="num_codebooks"):
        load_rvq_codes(path, num_codebooks=3)
