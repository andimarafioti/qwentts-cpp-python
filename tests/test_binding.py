from __future__ import annotations

import ctypes
import gc
import os
import threading
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from qwentts_cpp import (
    LibraryNotFoundError,
    QwenLibrary,
    QwenTTS,
    QwenTTSError,
    VoiceRef,
    load_rvq_codes,
    load_speaker_embedding,
    load_voice_ref,
    save_rvq_codes,
    save_speaker_embedding,
    save_voice_ref,
)
from qwentts_cpp._binding import QtInitParams, QtTTSParams, QtVoiceRef, _LogCallbackState


@pytest.mark.parametrize("version", [b"unknown", b"abcdef0 (2026-01-01)", b"", None, b"7df"])
def test_unverified_library_rejected_before_binding(monkeypatch, tmp_path, version):
    from unittest.mock import Mock

    path = tmp_path / "libqwen.dylib"
    path.touch()
    native = Mock()
    native.qt_version.return_value = version
    monkeypatch.setattr(QwenLibrary, "_load_cdll", lambda self, path: native)
    bind = Mock()
    monkeypatch.setattr(QwenLibrary, "_bind", bind)
    with pytest.raises(QwenTTSError, match="Unverified qwentts.cpp ABI"):
        QwenLibrary(path)
    bind.assert_not_called()
    native.qt_init_default_params.assert_not_called()
    native.qt_tts_default_params.assert_not_called()


def test_missing_native_symbol_reports_incompatible_library(monkeypatch, tmp_path):
    path = tmp_path / "libqwen.dylib"
    path.touch()
    monkeypatch.setattr(QwenLibrary, "_load_cdll", lambda self, path: object())
    with pytest.raises(QwenTTSError, match="missing required C ABI symbol"):
        QwenLibrary(path)


@pytest.mark.parametrize("revision", ["7df559a", "7df559a8", "7df559a8ca25f66fee02970514ebe5f01dee9055"])
def test_verified_native_revision_is_accepted(revision):
    from unittest.mock import Mock

    library = QwenLibrary.__new__(QwenLibrary)
    library._lib = Mock()
    library._lib.qt_version.return_value = f"{revision} (2026-05-01)".encode()
    library._validate_native_revision()


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


def test_log_callback_survives_loader_and_is_reused(tmp_path, capsys):
    class FakeNativeLibrary:
        def qt_log_set(self, callback, _user_data):
            # Native code stores the pointer, not a Python reference.
            self.callback_address = ctypes.cast(callback, ctypes.c_void_p).value
            self.callback_ref = weakref.ref(callback)

    native = FakeNativeLibrary()
    path = tmp_path / "libqwen.so"
    messages = []

    def install(handler):
        loader = QwenLibrary.__new__(QwenLibrary)
        loader.path = path
        loader._lib = native
        loader._log_callback_handler = None
        loader.set_log_callback(handler)
        return loader

    first_loader = install(lambda level, message: messages.append((level, message)))
    first_loader_ref = weakref.ref(first_loader)
    first_address = native.callback_address
    first_callback_ref = native.callback_ref
    del first_loader
    gc.collect()

    assert first_loader_ref() is None
    assert first_callback_ref() is not None
    first_callback_ref()(1, b"first", None)
    assert messages == []
    assert capsys.readouterr().err == "first\n"

    second_loader = install(lambda level, message: messages.append((level, message)))
    second_loader_ref = weakref.ref(second_loader)
    del second_loader
    gc.collect()

    assert second_loader_ref() is None
    assert native.callback_address == first_address
    assert native.callback_ref() is first_callback_ref()
    native.callback_ref()(2, b"second", None)
    assert messages == []
    assert capsys.readouterr().err == "second\n"

    third_loader = install(lambda level, message: messages.append((level, message)))
    native.callback_ref()(3, b"third", None)
    assert messages == [(3, "third")]

    third_loader.set_log_callback(None)
    assert native.callback_address is None


def test_quiet_log_level_filters_both_native_sources(capsys):
    state = _LogCallbackState()
    state.level = "warning"
    for level, message in enumerate((b"qt debug", b"qt info", b"qt warning", b"qt error")):
        state.native_callback(level, message, None)
    for level, message in ((1, b"ggml debug\n"), (2, b"ggml info\n"),
                           (3, b"ggml warning"), (5, b" continued\n"),
                           (4, b"ggml error\n")):
        state.ggml_callback(level, message, None)
    assert capsys.readouterr().err == (
        "qt warning\nqt error\nggml warning continued\nggml error\n"
    )

    state.level = "debug"
    state.native_callback(0, b"qt debug", None)
    state.ggml_callback(1, b"ggml debug\n", None)
    assert capsys.readouterr().err == "qt debug\nggml debug\n"


def test_log_level_callbacks_survive_repeated_loaders(tmp_path, capsys):
    qt_set = Mock()
    ggml_set = Mock()
    native = SimpleNamespace(qt_log_set=qt_set)
    ggml = SimpleNamespace(ggml_log_set=ggml_set)
    path = tmp_path / "libqwen.so"

    def install(level):
        loader = QwenLibrary.__new__(QwenLibrary)
        loader.path = path
        loader._lib = native
        loader._dependency_handles = [ggml]
        loader._log_callback_handler = None
        loader.set_log_level(level)
        return loader

    first = install("quiet")
    qt_callback = qt_set.call_args.args[0]
    ggml_callback = ggml_set.call_args.args[0]
    first_ref = weakref.ref(first)
    del first
    gc.collect()
    assert first_ref() is None
    qt_callback(1, b"routine", None)
    ggml_callback(2, b"routine\n", None)
    qt_callback(2, b"warning", None)
    ggml_callback(3, b"warning\n", None)
    assert capsys.readouterr().err == "warning\nwarning\n"

    second = install("verbose")
    assert qt_set.call_args.args[0] is qt_callback
    assert ggml_set.call_args.args[0] is ggml_callback
    qt_callback(1, b"routine", None)
    ggml_callback(2, b"routine\n", None)
    assert capsys.readouterr().err == "routine\nroutine\n"
    with pytest.raises(ValueError, match="log_level"):
        second.set_log_level("silent")


def test_log_handler_does_not_keep_tts_context_alive(tmp_path):
    class FakeNativeLibrary:
        def __init__(self):
            self.freed = []

        def qt_log_set(self, callback, _user_data):
            pass

        def qt_free(self, ctx):
            self.freed.append(ctx)

    native = FakeNativeLibrary()
    loader = QwenLibrary.__new__(QwenLibrary)
    loader.path = tmp_path / "libqwen.so"
    loader._lib = native
    loader._log_callback_handler = None
    tts = QwenTTS.__new__(QwenTTS)
    tts.library = loader
    tts._ctx = 123
    tts_ref = weakref.ref(tts)
    loader.set_log_callback(lambda level, message, owner=tts: None)
    del tts, loader
    gc.collect()

    assert tts_ref() is None
    assert native.freed == [123]


def test_native_log_callback_survives_repeated_init_when_available():
    path = os.environ.get("QWENTTS_CPP_LIBRARY")
    if not path:
        pytest.skip("QWENTTS_CPP_LIBRARY not set")

    messages = []
    first = QwenLibrary(path)
    first.set_log_callback(lambda level, message: messages.append((level, message)))
    params = QtInitParams()
    first._lib.qt_init_default_params(ctypes.byref(params))
    assert not first._lib.qt_init(ctypes.byref(params))
    assert messages
    first_ref = weakref.ref(first)
    del first
    gc.collect()
    assert first_ref() is None

    second = QwenLibrary(path)
    assert not second._lib.qt_init(ctypes.byref(params))
    second.set_log_callback(lambda level, message: messages.append((level, message)))
    count = len(messages)
    assert not second._lib.qt_init(ctypes.byref(params))
    assert len(messages) > count
    second.set_log_callback(None)


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
