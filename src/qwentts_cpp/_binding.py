from __future__ import annotations

import ctypes
import os
import queue
import sys
import threading
import time
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterator, Sequence, Tuple

import numpy as np


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
    ]


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
