"""The ROCm smoke gate must fail before synthesis if AMD is unavailable."""
from types import SimpleNamespace
import sys
from unittest.mock import MagicMock, Mock
import wave

import numpy as np
import pytest

from scripts import smoke_stream


@pytest.mark.parametrize("device,description,error", [
    (None, b"AMD Instinct MI300X", "ROCm0 was not registered"),
    (123, b"Other GPU", "not an AMD device"),
    (123, None, "not an AMD device"),
    (123, b"AMD Instinct MI300X", None),
])
def test_require_rocm_checks_registry(monkeypatch, tmp_path, device, description, error):
    ggml = SimpleNamespace(
        ggml_backend_load_all=Mock(),
        ggml_backend_dev_by_name=Mock(return_value=device),
        ggml_backend_dev_description=Mock(return_value=description),
    )
    load = Mock(return_value=ggml)
    monkeypatch.setattr(smoke_stream.ctypes, "CDLL", load)
    library = SimpleNamespace(path=tmp_path / "libqwen.so")
    if error:
        with pytest.raises(RuntimeError, match=error):
            smoke_stream.require_rocm(library)
    else:
        assert smoke_stream.require_rocm(library) == "AMD Instinct MI300X"
    load.assert_called_once_with(str(tmp_path / "libggml.so.0"))
    ggml.ggml_backend_load_all.assert_called_once_with()
    ggml.ggml_backend_dev_by_name.assert_called_once_with(b"ROCm0")


def test_rocm_missing_device_fails_before_model_load(monkeypatch):
    monkeypatch.setenv("GGML_BACKEND", "CPU")
    monkeypatch.setattr(sys, "argv", [
        "smoke_stream.py", "--talker", "talker.gguf", "--codec", "codec.gguf", "--require-rocm",
    ])
    library = Mock()
    monkeypatch.setattr(smoke_stream, "QwenLibrary", Mock(return_value=library))
    model = Mock()
    monkeypatch.setattr(smoke_stream, "QwenTTS", model)
    monkeypatch.setattr(smoke_stream, "require_rocm", Mock(side_effect=RuntimeError("missing AMD")))
    with pytest.raises(RuntimeError, match="missing AMD"):
        smoke_stream.main()
    assert smoke_stream.os.environ["GGML_BACKEND"] == "ROCm0"
    model.assert_not_called()


@pytest.mark.parametrize("audio,rate", [
    (np.zeros((2, 2)), 24000), (np.array([]), 24000),
    (np.array([np.nan]), 24000), (np.ones(10), 22050),
])
def test_smoke_rejects_invalid_audio(audio, rate):
    with pytest.raises(AssertionError):
        smoke_stream.check_audio(audio, rate)


def test_full_and_streaming_smoke_write_valid_wavs(monkeypatch, tmp_path):
    output = tmp_path / "amd.wav"
    monkeypatch.setenv("GGML_BACKEND", "CPU")
    monkeypatch.setattr(sys, "argv", [
        "smoke_stream.py", "--talker", "talker.gguf", "--codec", "codec.gguf",
        "--speaker", "Vivian", "--require-rocm", "--full", "--output", str(output),
    ])
    monkeypatch.setattr(smoke_stream, "QwenLibrary", Mock())
    gate = Mock(return_value="AMD Instinct MI300X")
    monkeypatch.setattr(smoke_stream, "require_rocm", gate)
    tts = MagicMock()
    tts.__enter__.return_value = tts
    audio = np.array([0.0, 0.25, -0.25], dtype=np.float32)
    tts.synthesize.return_value = (audio, 24000)
    tts.stream.return_value = iter([(audio, 24000), (audio, 24000)])
    monkeypatch.setattr(smoke_stream, "QwenTTS", Mock(return_value=tts))
    smoke_stream.main()
    gate.assert_called_once()
    assert smoke_stream.os.environ["GGML_BACKEND"] == "ROCm0"
    assert tts.synthesize.call_args.kwargs["speaker"] == "Vivian"
    assert tts.stream.call_args.kwargs["speaker"] == "Vivian"
    for path, frames in [(output, 6), (tmp_path / "amd-full.wav", 3)]:
        with wave.open(str(path)) as wav:
            assert wav.getframerate() == 24000
            assert wav.getnchannels() == 1
            assert wav.getnframes() == frames
            assert wav.getsampwidth() == 2
