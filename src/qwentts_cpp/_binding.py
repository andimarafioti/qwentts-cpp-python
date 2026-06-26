from __future__ import annotations

import ctypes
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterator, Sequence, Tuple

import numpy as np

QT_ABI_VERSION = 2
RVQ_CODE_BITS = 11


class QwenStatus(IntEnum):
    OK = 0
    INVALID_PARAMS = -1
    MODE_INVALID = -2
    GENERATE_FAILED = -3
    OOM = -4
    CANCELLED = -5


class QwenTTSError(RuntimeError):
    pass


class LibraryNotFoundError(QwenTTSError):
    pass


QT_CANCEL_CB = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)
QT_AUDIO_CHUNK_CB = ctypes.CFUNCTYPE(
    ctypes.c_bool,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_void_p,
)
QT_LOG_CB = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)


class QtAudio(ctypes.Structure):
    _fields_ = [
        ("samples", ctypes.POINTER(ctypes.c_float)),
        ("n_samples", ctypes.c_int),
        ("sample_rate", ctypes.c_int),
        ("channels", ctypes.c_int),
    ]


class QtInitParams(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_int),
        ("talker_path", ctypes.c_char_p),
        ("codec_path", ctypes.c_char_p),
        ("use_fa", ctypes.c_bool),
        ("clamp_fp16", ctypes.c_bool),
    ]


class QtTTSParams(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_int),
        ("text", ctypes.c_char_p),
        ("lang", ctypes.c_char_p),
        ("instruct", ctypes.c_char_p),
        ("speaker", ctypes.c_char_p),
        ("ref_audio_24k", ctypes.POINTER(ctypes.c_float)),
        ("ref_n_samples", ctypes.c_int),
        ("ref_text", ctypes.c_char_p),
        ("seed", ctypes.c_int64),
        ("max_new_tokens", ctypes.c_int),
        ("do_sample", ctypes.c_bool),
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_int),
        ("top_p", ctypes.c_float),
        ("repetition_penalty", ctypes.c_float),
        ("subtalker_do_sample", ctypes.c_bool),
        ("subtalker_temperature", ctypes.c_float),
        ("subtalker_top_k", ctypes.c_int),
        ("subtalker_top_p", ctypes.c_float),
        ("dump_dir", ctypes.c_char_p),
        ("cancel", QT_CANCEL_CB),
        ("cancel_user_data", ctypes.c_void_p),
        ("on_chunk", QT_AUDIO_CHUNK_CB),
        ("on_chunk_user_data", ctypes.c_void_p),
        ("codec_chunk_sec", ctypes.c_float),
        ("codec_left_context_sec", ctypes.c_float),
        ("ref_spk_emb", ctypes.POINTER(ctypes.c_float)),
        ("ref_spk_dim", ctypes.c_int),
        ("ref_codes", ctypes.POINTER(ctypes.c_int32)),
        ("ref_T", ctypes.c_int),
    ]


class QtVoiceRef(ctypes.Structure):
    _fields_ = [
        ("ref_spk_emb", ctypes.POINTER(ctypes.c_float)),
        ("ref_spk_dim", ctypes.c_int),
        ("ref_codes", ctypes.POINTER(ctypes.c_int32)),
        ("ref_T", ctypes.c_int),
        ("num_codebooks", ctypes.c_int),
    ]


@dataclass(frozen=True)
class VoiceRef:
    """Reusable Base voice-clone conditioning extracted by qwentts.cpp."""

    ref_spk_emb: np.ndarray
    ref_codes: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref_spk_emb", _prepare_speaker_embedding(self.ref_spk_emb))
        object.__setattr__(self, "ref_codes", _prepare_rvq_matrix(self.ref_codes))

    @property
    def num_codebooks(self) -> int:
        return int(self.ref_codes.shape[0])

    @property
    def ref_T(self) -> int:
        return int(self.ref_codes.shape[1])

    def save(
        self,
        spk_path: str | os.PathLike[str],
        rvq_path: str | os.PathLike[str],
        *,
        code_bits: int = RVQ_CODE_BITS,
    ) -> tuple[Path, Path]:
        """Write this reference as qwentts.cpp-compatible `.spk` and `.rvq` files."""
        return save_voice_ref(self, spk_path, rvq_path, code_bits=code_bits)


def _prepare_speaker_embedding(embedding: np.ndarray) -> np.ndarray:
    spk = np.ascontiguousarray(embedding, dtype=np.float32).reshape(-1)
    if spk.size == 0:
        raise ValueError("Speaker embedding must not be empty")
    return spk


def _prepare_rvq_matrix(codes: np.ndarray) -> np.ndarray:
    rvq = np.asarray(codes, dtype=np.int32)
    if rvq.ndim != 2 or rvq.shape[0] <= 0 or rvq.shape[1] <= 0:
        raise ValueError("RVQ codes must have shape [num_codebooks, T] with positive dimensions")
    return np.ascontiguousarray(rvq, dtype=np.int32)


def load_speaker_embedding(path: str | os.PathLike[str]) -> np.ndarray:
    """Load a qwentts.cpp `.spk` file as a contiguous float32 vector."""
    data = np.fromfile(path, dtype=np.float32)
    if data.size == 0:
        raise ValueError(f"Speaker embedding file is empty: {path}")
    return _prepare_speaker_embedding(data)


def save_speaker_embedding(path: str | os.PathLike[str], embedding: np.ndarray) -> Path:
    """Write a qwentts.cpp `.spk` file containing raw float32 speaker values."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _prepare_speaker_embedding(embedding).tofile(output)
    return output


def load_rvq_codes(
    path: str | os.PathLike[str],
    num_codebooks: int,
    *,
    code_bits: int = RVQ_CODE_BITS,
) -> np.ndarray:
    """Load a packed qwentts.cpp `.rvq` file as `[num_codebooks, T]` int32 codes."""
    if num_codebooks <= 0:
        raise ValueError(f"num_codebooks must be positive, got {num_codebooks}")
    if code_bits <= 0 or code_bits >= 32:
        raise ValueError(f"code_bits must be in [1, 31], got {code_bits}")

    packed = np.fromfile(path, dtype=np.uint8)
    if packed.size == 0:
        raise ValueError(f"RVQ file is empty: {path}")

    n_codes = (int(packed.size) * 8) // int(code_bits)
    if n_codes == 0 or n_codes % int(num_codebooks) != 0:
        raise ValueError(
            f"RVQ file {path} yields {n_codes} codes, not a positive multiple "
            f"of num_codebooks={num_codebooks}"
        )
    codes = _unpack_rvq_codes(packed, n_codes, int(code_bits))
    return codes.reshape(int(num_codebooks), n_codes // int(num_codebooks))


def save_rvq_codes(
    path: str | os.PathLike[str],
    codes: np.ndarray,
    *,
    code_bits: int = RVQ_CODE_BITS,
) -> Path:
    """Write qwentts.cpp `.rvq` packed 11-bit reference codec codes."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_pack_rvq_codes(_prepare_rvq_matrix(codes).reshape(-1), int(code_bits)))
    return output


def load_voice_ref(
    spk_path: str | os.PathLike[str],
    rvq_path: str | os.PathLike[str],
    num_codebooks: int,
    *,
    code_bits: int = RVQ_CODE_BITS,
) -> VoiceRef:
    """Load reusable Base voice-clone conditioning from `.spk` and `.rvq` files."""
    return VoiceRef(
        ref_spk_emb=load_speaker_embedding(spk_path),
        ref_codes=load_rvq_codes(rvq_path, num_codebooks, code_bits=code_bits),
    )


def save_voice_ref(
    voice_ref: VoiceRef,
    spk_path: str | os.PathLike[str],
    rvq_path: str | os.PathLike[str],
    *,
    code_bits: int = RVQ_CODE_BITS,
) -> tuple[Path, Path]:
    """Write a reusable Base voice-clone reference to `.spk` and `.rvq` files."""
    ref = VoiceRef(voice_ref.ref_spk_emb, voice_ref.ref_codes)
    spk = save_speaker_embedding(spk_path, ref.ref_spk_emb)
    rvq = save_rvq_codes(rvq_path, ref.ref_codes, code_bits=code_bits)
    return spk, rvq


def _unpack_rvq_codes(packed: np.ndarray, n_codes: int, code_bits: int) -> np.ndarray:
    mask = (1 << code_bits) - 1
    out = np.empty(n_codes, dtype=np.int32)
    acc = 0
    bits_in_acc = 0
    in_pos = 0
    data = packed.tolist()
    for i in range(n_codes):
        while bits_in_acc < code_bits and in_pos < len(data):
            acc |= int(data[in_pos]) << bits_in_acc
            bits_in_acc += 8
            in_pos += 1
        out[i] = acc & mask
        acc >>= code_bits
        bits_in_acc -= code_bits
    return out


def _pack_rvq_codes(codes: np.ndarray, code_bits: int) -> bytes:
    if code_bits <= 0 or code_bits >= 32:
        raise ValueError(f"code_bits must be in [1, 31], got {code_bits}")
    flat = np.asarray(codes, dtype=np.int64).reshape(-1)
    if flat.size == 0:
        raise ValueError("RVQ codes must not be empty")

    max_code = (1 << code_bits) - 1
    invalid = (flat < 0) | (flat > max_code)
    if bool(np.any(invalid)):
        bad = int(flat[np.nonzero(invalid)[0][0]])
        raise ValueError(f"RVQ code {bad} is outside the {code_bits}-bit range [0, {max_code}]")

    total_bits = int(flat.size) * int(code_bits)
    out = bytearray((total_bits + 7) // 8)
    acc = 0
    bits_in_acc = 0
    out_pos = 0
    for code in flat.tolist():
        acc |= int(code) << bits_in_acc
        bits_in_acc += int(code_bits)
        while bits_in_acc >= 8:
            out[out_pos] = acc & 0xFF
            out_pos += 1
            acc >>= 8
            bits_in_acc -= 8
    if bits_in_acc > 0:
        out[out_pos] = acc & 0xFF
    return bytes(out)


def _as_utf8(value: str | os.PathLike[str] | None, keepalive: list[object]) -> bytes | None:
    if value is None:
        return None
    data = os.fspath(value).encode("utf-8")
    keepalive.append(data)
    return data


def _library_names() -> Sequence[str]:
    if sys.platform == "win32":
        return ("qwen.dll", "libqwen.dll")
    if sys.platform == "darwin":
        return ("libqwen.dylib", "qwen.dylib")
    return ("libqwen.so", "libqwen.so.0")


def _dependency_names() -> Sequence[str]:
    if sys.platform == "win32":
        return ("ggml-base.dll", "ggml-cpu.dll", "ggml.dll")
    if sys.platform == "darwin":
        return ("libggml-base.dylib", "libggml-cpu.dylib", "libggml.dylib")
    return (
        "libggml-base.so",
        "libggml-base.so.0",
        "libggml-cpu.so",
        "libggml-cpu.so.0",
        "libggml-cuda.so",
        "libggml-cuda.so.0",
        "libggml-vulkan.so",
        "libggml-vulkan.so.0",
        "libggml-sycl.so",
        "libggml-sycl.so.0",
        "libggml.so",
        "libggml.so.0",
    )


def find_library(explicit_path: str | os.PathLike[str] | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        path = Path(explicit_path)
        if path.is_file():
            return path
        raise LibraryNotFoundError(f"Could not find qwentts.cpp shared library at explicit path: {path}")
    for env_name in ("QWENTTS_CPP_LIBRARY", "QWEN_LIBRARY_PATH"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value))

    package_lib_dir = Path(__file__).resolve().parent / "lib"
    for name in _library_names():
        candidates.append(package_lib_dir / name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n".join(str(p) for p in candidates)
    raise LibraryNotFoundError(
        "Could not find qwentts.cpp shared library. Set QWENTTS_CPP_LIBRARY "
        f"or install a wheel that bundles libqwen. Searched:\n{searched}"
    )


class QwenLibrary:
    """Thin loader for the `qwentts.cpp` C ABI."""

    def __init__(self, library_path: str | os.PathLike[str] | None = None):
        self.path = find_library(library_path)
        self._dll_dir_handle = None
        self._dependency_handles: list[ctypes.CDLL] = []
        self._log_callback: QT_LOG_CB | None = None
        self._has_qt_num_codebooks = False
        self._has_qt_n_speakers = False
        self._has_qt_speaker_name = False
        self._has_qt_extract_voice_ref = False
        self._has_qt_voice_ref_free = False
        self._lib = self._load_cdll(self.path)
        self._bind()

    def _load_cdll(self, path: Path) -> ctypes.CDLL:
        mode = getattr(ctypes, "RTLD_GLOBAL", 0)
        lib_dir = str(path.parent)

        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            self._dll_dir_handle = os.add_dll_directory(lib_dir)

        for dep_name in _dependency_names():
            dep = path.parent / dep_name
            if dep.is_file():
                try:
                    self._dependency_handles.append(ctypes.CDLL(str(dep), mode=mode))
                except OSError:
                    # Some names are symlink/SONAME aliases; only the loadable ones matter.
                    pass

        return ctypes.CDLL(str(path), mode=mode)

    def _bind(self) -> None:
        lib = self._lib
        lib.qt_version.argtypes = []
        lib.qt_version.restype = ctypes.c_char_p
        lib.qt_last_error.argtypes = []
        lib.qt_last_error.restype = ctypes.c_char_p
        lib.qt_init_default_params.argtypes = [ctypes.POINTER(QtInitParams)]
        lib.qt_init_default_params.restype = None
        lib.qt_tts_default_params.argtypes = [ctypes.POINTER(QtTTSParams)]
        lib.qt_tts_default_params.restype = None
        lib.qt_init.argtypes = [ctypes.POINTER(QtInitParams)]
        lib.qt_init.restype = ctypes.c_void_p
        lib.qt_free.argtypes = [ctypes.c_void_p]
        lib.qt_free.restype = None
        lib.qt_synthesize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(QtTTSParams),
            ctypes.POINTER(QtAudio),
        ]
        lib.qt_synthesize.restype = ctypes.c_int
        lib.qt_audio_free.argtypes = [ctypes.POINTER(QtAudio)]
        lib.qt_audio_free.restype = None
        lib.qt_log_set.argtypes = [QT_LOG_CB, ctypes.c_void_p]
        lib.qt_log_set.restype = None
        lib.qt_duration_sec_to_tokens.argtypes = [ctypes.c_void_p, ctypes.c_float]
        lib.qt_duration_sec_to_tokens.restype = ctypes.c_int
        try:
            lib.qt_num_codebooks.argtypes = [ctypes.c_void_p]
            lib.qt_num_codebooks.restype = ctypes.c_int
            self._has_qt_num_codebooks = True
        except AttributeError:
            self._has_qt_num_codebooks = False
        try:
            lib.qt_n_speakers.argtypes = [ctypes.c_void_p]
            lib.qt_n_speakers.restype = ctypes.c_int
            self._has_qt_n_speakers = True
        except AttributeError:
            self._has_qt_n_speakers = False
        try:
            lib.qt_speaker_name.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.qt_speaker_name.restype = ctypes.c_char_p
            self._has_qt_speaker_name = True
        except AttributeError:
            self._has_qt_speaker_name = False
        try:
            lib.qt_extract_voice_ref.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.POINTER(QtVoiceRef),
            ]
            lib.qt_extract_voice_ref.restype = ctypes.c_int
            self._has_qt_extract_voice_ref = True
        except AttributeError:
            self._has_qt_extract_voice_ref = False
        try:
            lib.qt_voice_ref_free.argtypes = [ctypes.POINTER(QtVoiceRef)]
            lib.qt_voice_ref_free.restype = None
            self._has_qt_voice_ref_free = True
        except AttributeError:
            self._has_qt_voice_ref_free = False

    def version(self) -> str:
        return self._lib.qt_version().decode("utf-8", errors="replace")

    def last_error(self) -> str:
        err = self._lib.qt_last_error()
        return err.decode("utf-8", errors="replace") if err else ""

    def set_log_callback(self, callback) -> None:
        """Install a process-wide qwentts.cpp log callback.

        The native ABI exposes logging globally, mirroring llama.cpp-style
        callbacks. Keep the ctypes callback alive on this loader so the C
        function pointer remains valid.
        """
        if callback is None:
            self._lib.qt_log_set(QT_LOG_CB(), None)
            self._log_callback = None
            return

        def _callback(level: int, message: bytes, _user_data) -> None:
            callback(int(level), message.decode("utf-8", errors="replace") if message else "")

        self._log_callback = QT_LOG_CB(_callback)
        self._lib.qt_log_set(self._log_callback, None)


class QwenTTS:
    """High-level Python wrapper over `qt_init` and `qt_synthesize`."""

    def __init__(
        self,
        talker_path: str | os.PathLike[str],
        codec_path: str | os.PathLike[str],
        *,
        library_path: str | os.PathLike[str] | None = None,
        use_fa: bool = True,
        clamp_fp16: bool = False,
    ):
        self.library = QwenLibrary(library_path)
        self._ctx: int | None = None
        self._lock = threading.Lock()
        self.last_synthesize_profile: dict[str, Any] | None = None
        self.last_stream_profile: dict[str, Any] | None = None
        self.last_extract_voice_ref_profile: dict[str, Any] | None = None
        self._init(talker_path, codec_path, use_fa=use_fa, clamp_fp16=clamp_fp16)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        quant: str = "BF16",
        cache_dir: str | os.PathLike[str] | None = None,
        local_files_only: bool = False,
        library_path: str | os.PathLike[str] | None = None,
        use_fa: bool = True,
        clamp_fp16: bool = False,
    ) -> "QwenTTS":
        from .models import resolve_gguf_paths

        talker, codec = resolve_gguf_paths(
            model_id,
            quant=quant,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        return cls(
            talker,
            codec,
            library_path=library_path,
            use_fa=use_fa,
            clamp_fp16=clamp_fp16,
        )

    def _init(
        self,
        talker_path: str | os.PathLike[str],
        codec_path: str | os.PathLike[str],
        *,
        use_fa: bool,
        clamp_fp16: bool,
    ) -> None:
        keepalive: list[bytes] = []
        params = QtInitParams()
        self.library._lib.qt_init_default_params(ctypes.byref(params))
        params.talker_path = _as_utf8(talker_path, keepalive)
        params.codec_path = _as_utf8(codec_path, keepalive)
        params.use_fa = bool(use_fa)
        params.clamp_fp16 = bool(clamp_fp16)
        ctx = self.library._lib.qt_init(ctypes.byref(params))
        if not ctx:
            raise QwenTTSError(self.library.last_error())
        self._ctx = ctx

    def close(self) -> None:
        if self._ctx:
            self.library._lib.qt_free(self._ctx)
            self._ctx = None

    def __enter__(self) -> "QwenTTS":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _require_ctx(self) -> int:
        if not self._ctx:
            raise QwenTTSError("QwenTTS context is closed")
        return self._ctx

    def duration_sec_to_tokens(self, seconds: float) -> int:
        return int(self.library._lib.qt_duration_sec_to_tokens(self._require_ctx(), float(seconds)))

    def num_codebooks(self) -> int:
        if not self.library._has_qt_num_codebooks:
            raise QwenTTSError("qt_num_codebooks is unavailable; cached RVQ references require qwentts.cpp ABI v2")
        value = int(self.library._lib.qt_num_codebooks(self._require_ctx()))
        if value <= 0:
            raise QwenTTSError(self.library.last_error() or "qt_num_codebooks returned 0")
        return value

    def speaker_names(self) -> list[str]:
        if not (self.library._has_qt_n_speakers and self.library._has_qt_speaker_name):
            raise QwenTTSError("Speaker enumeration requires qwentts.cpp ABI v2")
        count = int(self.library._lib.qt_n_speakers(self._require_ctx()))
        names: list[str] = []
        for i in range(count):
            value = self.library._lib.qt_speaker_name(self._require_ctx(), i)
            if value:
                names.append(value.decode("utf-8", errors="replace"))
        return names

    def load_rvq_codes(self, path: str | os.PathLike[str], *, code_bits: int = RVQ_CODE_BITS) -> np.ndarray:
        return load_rvq_codes(path, self.num_codebooks(), code_bits=code_bits)

    def load_voice_ref(
        self,
        spk_path: str | os.PathLike[str],
        rvq_path: str | os.PathLike[str],
        *,
        code_bits: int = RVQ_CODE_BITS,
    ) -> VoiceRef:
        return load_voice_ref(spk_path, rvq_path, self.num_codebooks(), code_bits=code_bits)

    def extract_voice_ref(self, ref_audio_24k: np.ndarray) -> VoiceRef:
        """Extract reusable Base voice-clone conditioning from 24 kHz mono audio."""
        if not (self.library._has_qt_extract_voice_ref and self.library._has_qt_voice_ref_free):
            raise QwenTTSError("qt_extract_voice_ref is unavailable; voice reference extraction requires qwentts.cpp ABI v2")

        profile: dict[str, Any] = {}
        start = time.perf_counter()
        audio = np.ascontiguousarray(ref_audio_24k, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise ValueError("ref_audio_24k must not be empty")
        profile["audio_prepare_ms"] = (time.perf_counter() - start) * 1000
        profile["ref_n_samples"] = int(audio.size)

        out = QtVoiceRef()
        lock_start = time.perf_counter()
        with self._lock:
            profile["lock_wait_ms"] = (time.perf_counter() - lock_start) * 1000
            native_start = time.perf_counter()
            rc = self.library._lib.qt_extract_voice_ref(
                self._require_ctx(),
                audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(audio.size),
                ctypes.byref(out),
            )
            profile["native_extract_ms"] = (time.perf_counter() - native_start) * 1000

        try:
            if rc != QwenStatus.OK:
                raise QwenTTSError(self.library.last_error() or f"qt_extract_voice_ref failed with status {rc}")
            if not out.ref_spk_emb or out.ref_spk_dim <= 0:
                raise QwenTTSError("qt_extract_voice_ref returned an empty speaker embedding")
            if not out.ref_codes or out.num_codebooks <= 0 or out.ref_T <= 0:
                raise QwenTTSError("qt_extract_voice_ref returned empty RVQ codes")

            copy_start = time.perf_counter()
            spk = np.ctypeslib.as_array(out.ref_spk_emb, shape=(int(out.ref_spk_dim),)).copy()
            codes = np.ctypeslib.as_array(
                out.ref_codes,
                shape=(int(out.num_codebooks) * int(out.ref_T),),
            ).copy()
            codes = codes.reshape(int(out.num_codebooks), int(out.ref_T))
            profile["copy_ms"] = (time.perf_counter() - copy_start) * 1000
            profile["ref_spk_dim"] = int(out.ref_spk_dim)
            profile["num_codebooks"] = int(out.num_codebooks)
            profile["ref_T"] = int(out.ref_T)
            profile["total_ms"] = (time.perf_counter() - start) * 1000
            self.last_extract_voice_ref_profile = profile
            return VoiceRef(ref_spk_emb=spk, ref_codes=codes)
        finally:
            self.library._lib.qt_voice_ref_free(ctypes.byref(out))

    def save_voice_ref(
        self,
        ref_audio_24k: np.ndarray,
        spk_path: str | os.PathLike[str],
        rvq_path: str | os.PathLike[str],
        *,
        code_bits: int = RVQ_CODE_BITS,
    ) -> VoiceRef:
        """Extract and save reusable Base voice-clone conditioning from reference audio."""
        voice_ref = self.extract_voice_ref(ref_audio_24k)
        voice_ref.save(spk_path, rvq_path, code_bits=code_bits)
        return voice_ref

    def set_log_callback(self, callback) -> None:
        self.library.set_log_callback(callback)

    def synthesize(
        self,
        *,
        text: str,
        lang: str = "english",
        instruct: str | None = None,
        speaker: str | None = None,
        ref_audio_24k: np.ndarray | None = None,
        ref_spk_emb: np.ndarray | None = None,
        ref_codes: np.ndarray | None = None,
        ref_text: str | None = None,
        seed: int = -1,
        max_new_tokens: int = 2048,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        subtalker_do_sample: bool | None = None,
        subtalker_temperature: float | None = None,
        subtalker_top_k: int | None = None,
        subtalker_top_p: float | None = None,
        codec_chunk_sec: float = 24.0,
        codec_left_context_sec: float = 2.0,
        dump_dir: str | os.PathLike[str] | None = None,
    ) -> Tuple[np.ndarray, int]:
        profile: dict[str, Any] = {"mode": "buffered"}
        start = time.perf_counter()
        params_start = time.perf_counter()
        params, keepalive = self._make_tts_params(
            text=text,
            lang=lang,
            instruct=instruct,
            speaker=speaker,
            ref_audio_24k=ref_audio_24k,
            ref_spk_emb=ref_spk_emb,
            ref_codes=ref_codes,
            ref_text=ref_text,
            seed=seed,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            subtalker_do_sample=subtalker_do_sample,
            subtalker_temperature=subtalker_temperature,
            subtalker_top_k=subtalker_top_k,
            subtalker_top_p=subtalker_top_p,
            codec_chunk_sec=codec_chunk_sec,
            codec_left_context_sec=codec_left_context_sec,
            dump_dir=dump_dir,
        )
        profile["make_params_ms"] = (time.perf_counter() - params_start) * 1000

        audio = QtAudio()
        lock_start = time.perf_counter()
        with self._lock:
            profile["lock_wait_ms"] = (time.perf_counter() - lock_start) * 1000
            profile["native_enter_ms"] = (time.perf_counter() - start) * 1000
            native_start = time.perf_counter()
            rc = self.library._lib.qt_synthesize(self._require_ctx(), ctypes.byref(params), ctypes.byref(audio))
            profile["native_synthesize_ms"] = (time.perf_counter() - native_start) * 1000
        profile["native_return_ms"] = (time.perf_counter() - start) * 1000
        if rc != QwenStatus.OK:
            self.library._lib.qt_audio_free(ctypes.byref(audio))
            profile["total_ms"] = (time.perf_counter() - start) * 1000
            self.last_synthesize_profile = profile
            raise QwenTTSError(self.library.last_error() or f"qt_synthesize failed with status {rc}")

        try:
            if not audio.samples or audio.n_samples <= 0:
                profile["audio_copy_ms"] = 0.0
                profile["n_samples"] = 0
                profile["sample_rate"] = int(audio.sample_rate or 24000)
                profile["total_ms"] = (time.perf_counter() - start) * 1000
                self.last_synthesize_profile = profile
                return np.zeros(0, dtype=np.float32), int(audio.sample_rate or 24000)
            copy_start = time.perf_counter()
            samples = np.ctypeslib.as_array(audio.samples, shape=(audio.n_samples,)).copy()
            profile["audio_copy_ms"] = (time.perf_counter() - copy_start) * 1000
            profile["n_samples"] = int(audio.n_samples)
            profile["sample_rate"] = int(audio.sample_rate)
            profile["total_ms"] = (time.perf_counter() - start) * 1000
            self.last_synthesize_profile = profile
            return samples.astype(np.float32, copy=False), int(audio.sample_rate)
        finally:
            self.library._lib.qt_audio_free(ctypes.byref(audio))

    def stream(
        self,
        *,
        text: str,
        lang: str = "english",
        instruct: str | None = None,
        speaker: str | None = None,
        ref_audio_24k: np.ndarray | None = None,
        ref_spk_emb: np.ndarray | None = None,
        ref_codes: np.ndarray | None = None,
        ref_text: str | None = None,
        seed: int = -1,
        max_new_tokens: int = 2048,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        subtalker_do_sample: bool | None = None,
        subtalker_temperature: float | None = None,
        subtalker_top_k: int | None = None,
        subtalker_top_p: float | None = None,
        codec_chunk_sec: float = 1.0,
        codec_left_context_sec: float = 2.0,
        dump_dir: str | os.PathLike[str] | None = None,
    ) -> Iterator[Tuple[np.ndarray, int]]:
        profile: dict[str, Any] = {
            "mode": "stream",
            "codec_chunk_sec": float(codec_chunk_sec),
            "codec_left_context_sec": float(codec_left_context_sec),
            "callback_count": 0,
            "callback_copy_ms_total": 0.0,
            "callback_queue_ms_total": 0.0,
        }
        start = time.perf_counter()
        self.last_stream_profile = profile

        def elapsed_ms() -> float:
            return (time.perf_counter() - start) * 1000

        chunks: queue.Queue[object] = queue.Queue()
        done = object()
        cancel_event = threading.Event()

        def cancel_cb(_user_data) -> bool:
            return cancel_event.is_set()

        def on_chunk(samples, n_samples, _user_data) -> bool:
            if cancel_event.is_set():
                return False
            callback_enter_ms = elapsed_ms()
            is_first = profile["callback_count"] == 0
            if is_first:
                profile["first_callback_enter_ms"] = callback_enter_ms
                profile["first_callback_n_samples"] = int(n_samples)
                profile["first_callback_audio_s"] = float(n_samples) / 24000.0
            profile["callback_count"] += 1
            copy_start = time.perf_counter()
            chunk = np.ctypeslib.as_array(samples, shape=(n_samples,)).copy()
            copy_ms = (time.perf_counter() - copy_start) * 1000
            profile["callback_copy_ms_total"] += copy_ms
            if is_first:
                profile["first_callback_copy_ms"] = copy_ms
            queue_start = time.perf_counter()
            chunks.put((chunk.astype(np.float32, copy=False), 24000))
            queue_ms = (time.perf_counter() - queue_start) * 1000
            profile["callback_queue_ms_total"] += queue_ms
            if is_first:
                profile["first_callback_queue_ms"] = queue_ms
                profile["first_callback_return_ms"] = elapsed_ms()
            return True

        cancel_callback = QT_CANCEL_CB(cancel_cb)
        chunk_callback = QT_AUDIO_CHUNK_CB(on_chunk)

        def producer() -> None:
            profile["producer_thread_start_ms"] = elapsed_ms()
            audio = QtAudio()
            try:
                params_start = time.perf_counter()
                params, keepalive = self._make_tts_params(
                    text=text,
                    lang=lang,
                    instruct=instruct,
                    speaker=speaker,
                    ref_audio_24k=ref_audio_24k,
                    ref_spk_emb=ref_spk_emb,
                    ref_codes=ref_codes,
                    ref_text=ref_text,
                    seed=seed,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    subtalker_do_sample=subtalker_do_sample,
                    subtalker_temperature=subtalker_temperature,
                    subtalker_top_k=subtalker_top_k,
                    subtalker_top_p=subtalker_top_p,
                    codec_chunk_sec=codec_chunk_sec,
                    codec_left_context_sec=codec_left_context_sec,
                    dump_dir=dump_dir,
                )
                profile["make_params_ms"] = (time.perf_counter() - params_start) * 1000
                params.cancel = cancel_callback
                params.on_chunk = chunk_callback
                profile["before_lock_ms"] = elapsed_ms()
                lock_start = time.perf_counter()
                with self._lock:
                    profile["lock_wait_ms"] = (time.perf_counter() - lock_start) * 1000
                    profile["native_enter_ms"] = elapsed_ms()
                    native_start = time.perf_counter()
                    rc = self.library._lib.qt_synthesize(
                        self._require_ctx(),
                        ctypes.byref(params),
                        ctypes.byref(audio),
                    )
                    profile["native_synthesize_ms"] = (time.perf_counter() - native_start) * 1000
                profile["native_return_ms"] = elapsed_ms()
                if rc != QwenStatus.OK and not cancel_event.is_set():
                    chunks.put(QwenTTSError(self.library.last_error() or f"qt_synthesize failed with status {rc}"))
            except BaseException as exc:
                if not cancel_event.is_set():
                    chunks.put(exc)
            finally:
                self.library._lib.qt_audio_free(ctypes.byref(audio))
                profile["producer_done_ms"] = elapsed_ms()
                chunks.put(done)

        thread = threading.Thread(target=producer, daemon=True)
        profile["before_thread_start_ms"] = elapsed_ms()
        thread.start()
        profile["after_thread_start_ms"] = elapsed_ms()
        try:
            while True:
                item = chunks.get()
                profile["last_consumer_get_ms"] = elapsed_ms()
                if item is done:
                    profile["consumer_done_ms"] = elapsed_ms()
                    break
                if isinstance(item, BaseException):
                    profile["consumer_error_ms"] = elapsed_ms()
                    raise item
                if "first_yield_ms" not in profile:
                    profile["first_yield_ms"] = elapsed_ms()
                    if "first_callback_enter_ms" in profile:
                        profile["first_callback_to_yield_ms"] = (
                            profile["first_yield_ms"] - profile["first_callback_enter_ms"]
                        )
                yield item  # type: ignore[misc]
        finally:
            cancel_event.set()
            if thread.is_alive():
                thread.join(timeout=1.0)
            profile["stream_closed_ms"] = elapsed_ms()

    def _make_tts_params(
        self,
        *,
        text: str,
        lang: str,
        instruct: str | None,
        speaker: str | None,
        ref_audio_24k: np.ndarray | None,
        ref_spk_emb: np.ndarray | None,
        ref_codes: np.ndarray | None,
        ref_text: str | None,
        seed: int,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        subtalker_do_sample: bool | None,
        subtalker_temperature: float | None,
        subtalker_top_k: int | None,
        subtalker_top_p: float | None,
        codec_chunk_sec: float,
        codec_left_context_sec: float,
        dump_dir: str | os.PathLike[str] | None,
    ) -> tuple[QtTTSParams, list[object]]:
        keepalive: list[object] = []
        params = QtTTSParams()
        self.library._lib.qt_tts_default_params(ctypes.byref(params))

        if (ref_spk_emb is not None or ref_codes is not None) and params.abi_version < QT_ABI_VERSION:
            raise QwenTTSError("Cached speaker/RVQ references require qwentts.cpp ABI v2")
        if ref_audio_24k is not None and (ref_spk_emb is not None or ref_codes is not None):
            raise ValueError("ref_audio_24k is mutually exclusive with ref_spk_emb/ref_codes")
        if ref_codes is not None and ref_spk_emb is None:
            raise ValueError("ref_codes requires ref_spk_emb")
        if ref_codes is not None and not ref_text:
            raise ValueError("ref_codes requires ref_text")

        params.text = _as_utf8(text, keepalive)  # type: ignore[arg-type]
        params.lang = _as_utf8(lang, keepalive)  # type: ignore[arg-type]
        params.instruct = _as_utf8(instruct, keepalive)  # type: ignore[arg-type]
        params.speaker = _as_utf8(speaker, keepalive)  # type: ignore[arg-type]
        params.ref_text = _as_utf8(ref_text, keepalive)  # type: ignore[arg-type]
        params.dump_dir = _as_utf8(dump_dir, keepalive)  # type: ignore[arg-type]

        if ref_audio_24k is not None:
            audio = np.ascontiguousarray(ref_audio_24k, dtype=np.float32).reshape(-1)
            keepalive.append(audio)
            params.ref_audio_24k = audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            params.ref_n_samples = int(audio.shape[0])

        if ref_spk_emb is not None:
            spk = np.ascontiguousarray(ref_spk_emb, dtype=np.float32).reshape(-1)
            if spk.size == 0:
                raise ValueError("ref_spk_emb must not be empty")
            keepalive.append(spk)
            params.ref_spk_emb = spk.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            params.ref_spk_dim = int(spk.shape[0])

        if ref_codes is not None:
            codes, ref_T = self._prepare_ref_codes(ref_codes)
            keepalive.append(codes)
            params.ref_codes = codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            params.ref_T = int(ref_T)

        params.seed = int(seed)
        params.max_new_tokens = int(max_new_tokens)
        params.do_sample = bool(do_sample)
        params.temperature = float(temperature)
        params.top_k = int(top_k)
        params.top_p = float(top_p)
        params.repetition_penalty = float(repetition_penalty)
        params.subtalker_do_sample = bool(do_sample if subtalker_do_sample is None else subtalker_do_sample)
        params.subtalker_temperature = float(temperature if subtalker_temperature is None else subtalker_temperature)
        params.subtalker_top_k = int(top_k if subtalker_top_k is None else subtalker_top_k)
        params.subtalker_top_p = float(top_p if subtalker_top_p is None else subtalker_top_p)
        params.codec_chunk_sec = float(codec_chunk_sec)
        params.codec_left_context_sec = float(codec_left_context_sec)
        return params, keepalive

    def _prepare_ref_codes(self, ref_codes: np.ndarray) -> tuple[np.ndarray, int]:
        codes = np.asarray(ref_codes, dtype=np.int32)
        if codes.ndim == 2:
            if codes.shape[0] <= 0 or codes.shape[1] <= 0:
                raise ValueError("ref_codes must have shape [num_codebooks, T] with positive dimensions")
            expected = self.num_codebooks() if self.library._has_qt_num_codebooks else codes.shape[0]
            if codes.shape[0] != expected:
                raise ValueError(f"ref_codes has {codes.shape[0]} codebooks, expected {expected}")
            return np.ascontiguousarray(codes.reshape(-1), dtype=np.int32), int(codes.shape[1])
        if codes.ndim == 1:
            if codes.size == 0:
                raise ValueError("ref_codes must not be empty")
            num_codebooks = self.num_codebooks()
            if codes.size % num_codebooks != 0:
                raise ValueError(
                    f"flat ref_codes length {codes.size} is not divisible by num_codebooks={num_codebooks}"
                )
            return np.ascontiguousarray(codes, dtype=np.int32), int(codes.size // num_codebooks)
        raise ValueError("ref_codes must be a flat array or a [num_codebooks, T] matrix")
