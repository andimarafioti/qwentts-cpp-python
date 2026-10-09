"""Exercise an installed wheel with local GGUF weights and write playable PCM WAV."""
from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import wave

import numpy as np

from qwentts_cpp import QwenLibrary, QwenTTS, load_speaker_embedding


def require_rocm(library: QwenLibrary) -> str:
    """Check the installed GGML registry before loading any model weights."""
    ggml = ctypes.CDLL(str(library.path.parent / "libggml.so.0"))
    ggml.ggml_backend_load_all.argtypes = []
    ggml.ggml_backend_load_all.restype = None
    ggml.ggml_backend_dev_by_name.argtypes = [ctypes.c_char_p]
    ggml.ggml_backend_dev_by_name.restype = ctypes.c_void_p
    ggml.ggml_backend_dev_description.argtypes = [ctypes.c_void_p]
    ggml.ggml_backend_dev_description.restype = ctypes.c_char_p
    ggml.ggml_backend_load_all()
    device = ggml.ggml_backend_dev_by_name(b"ROCm0")
    if not device:
        raise RuntimeError("ROCm0 was not registered; an AMD ROCm device is required")
    raw_description = ggml.ggml_backend_dev_description(device)
    description = raw_description.decode("utf-8", errors="replace") if raw_description else ""
    if "AMD" not in description:
        raise RuntimeError(f"ROCm0 is not an AMD device: {description!r}")
    return description


def check_audio(audio: np.ndarray, rate: int) -> None:
    assert rate == 24000 and audio.ndim == 1 and audio.size, "Expected nonempty mono 24 kHz audio"
    assert np.isfinite(audio).all(), "Invalid audio samples"


def write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    check_audio(audio, rate)
    assert np.max(np.abs(audio)) > 1e-5, "Synthesized audio is silent"
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--talker", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("metal-smoke.wav"))
    parser.add_argument("--text", default="Hello from the Apple Silicon Metal wheel.")
    parser.add_argument("--speaker", help="Speaker name for a CustomVoice model")
    parser.add_argument("--ref-spk", type=Path, help="Local .spk embedding for a Base model")
    parser.add_argument("--instruct", help="Style instruction for a VoiceDesign model")
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--require-metal", action="store_true")
    device.add_argument("--require-rocm", action="store_true")
    parser.add_argument("--full", action="store_true", help="Also validate buffered synthesis and write a *-full.wav")
    args = parser.parse_args()
    if args.require_metal:
        # Explicit selection fails if Metal is unavailable, instead of silently
        # falling back to CPU. This is the native ggml device name.
        os.environ["GGML_BACKEND"] = "MTL0"
    if args.require_rocm:
        os.environ["GGML_BACKEND"] = "ROCm0"
        library = QwenLibrary(log_level="info")
        print(f"Native library: {library.path}")
        print(f"Native version: {library.version()}")
        print(f"AMD device: {require_rocm(library)}")
    chunks = []
    rate = None
    with QwenTTS(args.talker, args.codec) as tts:
        print(f"Native version: {tts.library.version()}")
        request = dict(
            text=args.text, speaker=args.speaker, instruct=args.instruct,
            ref_spk_emb=load_speaker_embedding(args.ref_spk) if args.ref_spk else None,
            seed=42, max_new_tokens=128,
        )
        if args.full:
            audio, sample_rate = tts.synthesize(**request)
            full_output = args.output.with_name(args.output.stem + "-full.wav")
            write_wav(full_output, audio, sample_rate)
            print(f"Wrote {full_output}: {audio.size / sample_rate:.2f} seconds")
        for chunk, sample_rate in tts.stream(**request, codec_chunk_sec=0.5):
            check_audio(chunk, sample_rate)
            assert rate in (None, sample_rate), "Sample rate changed during streaming"
            rate = sample_rate
            chunks.append(chunk)
            print(f"Chunk {len(chunks)}: {chunk.size} samples at {rate} Hz", flush=True)
    assert len(chunks) >= 2, "Expected multiple streaming callbacks"
    audio = np.concatenate(chunks)
    write_wav(args.output, audio, rate)
    print(f"Wrote {args.output}: {len(chunks)} chunks, {audio.size / rate:.2f} seconds")


if __name__ == "__main__":
    main()
