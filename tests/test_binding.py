from __future__ import annotations

import ctypes
import os
import threading

import numpy as np
import pytest

from qwentts_cpp import (
    LibraryNotFoundError,
    QwenLibrary,
    QwenTTS,
    VoiceRef,
    load_rvq_codes,
    load_speaker_embedding,
    load_voice_ref,
    save_rvq_codes,
    save_speaker_embedding,
    save_voice_ref,
)
from qwentts_cpp._binding import QtTTSParams, QtVoiceRef


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


def test_voice_ref_struct_matches_abi():
    assert [name for name, _ctype in QtVoiceRef._fields_] == [
        "ref_spk_emb",
        "ref_spk_dim",
        "ref_codes",
        "ref_T",
        "num_codebooks",
    ]


def test_load_speaker_embedding_reads_raw_float32(tmp_path):
    path = tmp_path / "speaker.spk"
    expected = np.array([0.25, -0.5, 1.0], dtype=np.float32)
    expected.tofile(path)

    loaded = load_speaker_embedding(path)

    assert loaded.dtype == np.float32
    assert loaded.flags.c_contiguous
    np.testing.assert_array_equal(loaded, expected)


def test_save_speaker_embedding_writes_raw_float32(tmp_path):
    path = tmp_path / "nested" / "speaker.spk"
    expected = np.array([0.25, -0.5, 1.0], dtype=np.float32)

    saved = save_speaker_embedding(path, expected)

    assert saved == path
    np.testing.assert_array_equal(load_speaker_embedding(path), expected)


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


def test_save_rvq_codes_packs_lsb_first_matrix(tmp_path):
    path = tmp_path / "nested" / "reference.rvq"
    expected = np.array(
        [
            [1, 2, 3],
            [2047, 17, 42],
            [0, 999, 123],
            [456, 789, 1024],
        ],
        dtype=np.int32,
    )

    saved = save_rvq_codes(path, expected)

    assert saved == path
    assert path.read_bytes() == _pack_rvq_codes(expected.reshape(-1).tolist())
    np.testing.assert_array_equal(load_rvq_codes(path, num_codebooks=expected.shape[0]), expected)


def test_load_rvq_codes_rejects_wrong_codebook_count(tmp_path):
    path = tmp_path / "reference.rvq"
    path.write_bytes(_pack_rvq_codes([1, 2, 3, 4]))

    with pytest.raises(ValueError, match="num_codebooks"):
        load_rvq_codes(path, num_codebooks=3)


def test_save_rvq_codes_rejects_out_of_range_codes(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        save_rvq_codes(tmp_path / "bad.rvq", np.array([[0, 2048]], dtype=np.int32))


def test_voice_ref_save_and_load_round_trips_files(tmp_path):
    spk = np.array([0.25, -0.5, 1.0], dtype=np.float32)
    codes = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    ref = VoiceRef(spk, codes)

    spk_path, rvq_path = save_voice_ref(ref, tmp_path / "voice.spk", tmp_path / "voice.rvq")
    loaded = load_voice_ref(spk_path, rvq_path, num_codebooks=2)

    assert ref.num_codebooks == 2
    assert ref.ref_T == 3
    np.testing.assert_array_equal(loaded.ref_spk_emb, spk)
    np.testing.assert_array_equal(loaded.ref_codes, codes)


def test_voice_ref_instance_save_round_trips_files(tmp_path):
    ref = VoiceRef(
        np.array([0.5, 1.5], dtype=np.float32),
        np.array([[7, 8], [9, 10]], dtype=np.int32),
    )

    ref.save(tmp_path / "speaker.spk", tmp_path / "reference.rvq")

    loaded = load_voice_ref(tmp_path / "speaker.spk", tmp_path / "reference.rvq", num_codebooks=2)
    np.testing.assert_array_equal(loaded.ref_spk_emb, ref.ref_spk_emb)
    np.testing.assert_array_equal(loaded.ref_codes, ref.ref_codes)


class _FakeExtractLib:
    def __init__(self):
        self.free_calls = 0
        self.spk_buf = None
        self.codes_buf = None
        self.ctx = None
        self.ref_audio = None

    def qt_extract_voice_ref(self, ctx, audio_ptr, n_samples, out_ptr):
        self.ctx = ctx
        self.ref_audio = np.ctypeslib.as_array(audio_ptr, shape=(n_samples,)).copy()
        self.spk_buf = (ctypes.c_float * 3)(0.25, -0.5, 1.0)
        self.codes_buf = (ctypes.c_int32 * 6)(1, 2, 3, 4, 5, 6)
        out = out_ptr._obj
        out.ref_spk_emb = ctypes.cast(self.spk_buf, ctypes.POINTER(ctypes.c_float))
        out.ref_spk_dim = 3
        out.ref_codes = ctypes.cast(self.codes_buf, ctypes.POINTER(ctypes.c_int32))
        out.ref_T = 3
        out.num_codebooks = 2
        return 0

    def qt_voice_ref_free(self, out_ptr):
        self.free_calls += 1
        out = out_ptr._obj
        out.ref_spk_emb = ctypes.POINTER(ctypes.c_float)()
        out.ref_spk_dim = 0
        out.ref_codes = ctypes.POINTER(ctypes.c_int32)()
        out.ref_T = 0
        out.num_codebooks = 0


class _FakeLibrary:
    def __init__(self, lib):
        self._lib = lib
        self._has_qt_extract_voice_ref = True
        self._has_qt_voice_ref_free = True

    def last_error(self):
        return "fake error"


def test_extract_voice_ref_copies_native_buffers_before_free():
    fake_lib = _FakeExtractLib()
    tts = QwenTTS.__new__(QwenTTS)
    tts.library = _FakeLibrary(fake_lib)
    tts._ctx = 123
    tts._lock = threading.Lock()
    tts.last_extract_voice_ref_profile = None

    ref = tts.extract_voice_ref(np.array([0.0, 0.5, -0.5], dtype=np.float64))

    assert fake_lib.ctx == 123
    assert fake_lib.free_calls == 1
    np.testing.assert_array_equal(fake_lib.ref_audio, np.array([0.0, 0.5, -0.5], dtype=np.float32))
    np.testing.assert_array_equal(ref.ref_spk_emb, np.array([0.25, -0.5, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(ref.ref_codes, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))

    fake_lib.spk_buf[0] = 99.0
    fake_lib.codes_buf[0] = 99
    assert ref.ref_spk_emb[0] == np.float32(0.25)
    assert ref.ref_codes[0, 0] == 1
    assert tts.last_extract_voice_ref_profile["ref_spk_dim"] == 3
