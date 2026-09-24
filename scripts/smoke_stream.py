"""Exercise an installed wheel with local GGUF weights and write playable PCM WAV."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import wave

import numpy as np

from qwentts_cpp import QwenTTS, load_speaker_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--talker", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("metal-smoke.wav"))
    parser.add_argument("--text", default="Hello from the Apple Silicon Metal wheel.")
    parser.add_argument("--speaker", help="Speaker name for a CustomVoice model")
    parser.add_argument("--ref-spk", type=Path, help="Local .spk embedding for a Base model")
    parser.add_argument("--instruct", help="Style instruction for a VoiceDesign model")
    parser.add_argument("--require-metal", action="store_true")
    args = parser.parse_args()
    if args.require_metal:
        # Explicit selection fails if Metal is unavailable, instead of silently
        # falling back to CPU. This is the native ggml device name.
        os.environ["GGML_BACKEND"] = "MTL0"
    chunks = []
    rate = None
    with QwenTTS(args.talker, args.codec) as tts:
        print(f"Native version: {tts.library.version()}")
        for chunk, sample_rate in tts.stream(
            text=args.text, speaker=args.speaker, instruct=args.instruct,
            ref_spk_emb=load_speaker_embedding(args.ref_spk) if args.ref_spk else None,
            seed=42, max_new_tokens=128, codec_chunk_sec=0.5,
        ):
            assert chunk.size and np.isfinite(chunk).all(), "Invalid streaming audio"
            assert rate in (None, sample_rate), "Sample rate changed during streaming"
            rate = sample_rate
            chunks.append(chunk)
            print(f"Chunk {len(chunks)}: {chunk.size} samples at {rate} Hz", flush=True)
    assert len(chunks) >= 2, "Expected multiple streaming callbacks"
    audio = np.concatenate(chunks)
    assert np.max(np.abs(audio)) > 1e-5, "Synthesized audio is silent"
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(args.output), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())
    print(f"Wrote {args.output}: {len(chunks)} chunks, {audio.size / rate:.2f} seconds")


if __name__ == "__main__":
    main()
