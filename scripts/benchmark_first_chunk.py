"""Compare first-packet latency and duration using local GGUF weights."""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import statistics

import numpy as np

from qwentts_cpp import QwenTTS, load_speaker_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--talker", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--speaker")
    parser.add_argument("--ref-spk", type=Path)
    parser.add_argument("--instruct")
    parser.add_argument("--require-metal", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--codec-chunk-sec", type=float, default=0.64)
    parser.add_argument("--output", type=Path, default=Path("first-chunk-benchmark.json"))
    parser.add_argument("--text", default=(
        "This is a streaming speech test. The first packet should arrive promptly, "
        "with enough audio to keep playback smooth while the next packet is generated."
    ))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.require_metal:
        os.environ["GGML_BACKEND"] = "MTL0"
    ref = load_speaker_embedding(args.ref_spk) if args.ref_spk else None
    results = []
    with QwenTTS(args.talker, args.codec) as tts:
        def run(first: int) -> dict:
            packets = list(tts.stream(
                text=args.text, speaker=args.speaker, instruct=args.instruct,
                ref_spk_emb=ref, seed=42, max_new_tokens=256,
                codec_chunk_sec=args.codec_chunk_sec, first_chunk_frames=first,
            ))
            assert len(packets) >= 3, "Use a longer utterance to measure steady-state packets"
            assert all(p.size and np.isfinite(p).all() and sr == 24000 for p, sr in packets)
            assert any(np.max(np.abs(p)) > 1e-5 for p, _ in packets), "Silent audio"
            assert packets[0][0].size == first * 1920, "Incorrect first packet size"
            profile = tts.last_stream_profile
            assert all(p.size == profile["packet_frames"] * 1920 for p, _ in packets[1:-1])
            assert packets[-1][0].size <= profile["packet_frames"] * 1920
            return {
                "first_chunk_frames": first,
                "native_first_callback_ms": profile["first_callback_enter_ms"],
                "native_first_callback_audio_ms": profile["first_callback_audio_s"] * 1000,
                "first_packet_ready_ms": profile["first_packet_ready_ms"],
                "first_yield_ms": profile["first_yield_ms"],
                "first_packet_audio_ms": profile["first_packet_audio_s"] * 1000,
                "packet_samples": [int(p.size) for p, _ in packets],
                "total_ms": profile["consumer_done_ms"],
            }

        # Warm the model/codec before measurements, then rotate the order to
        # reduce systematic cache and temperature bias between settings.
        print("Warming up (excluded from measurements)...", flush=True)
        run(4)
        for repeat in range(args.repeats):
            order = (1, 4, 8)
            shift = repeat % len(order)
            for first in order[shift:] + order[:shift]:
                result = run(first)
                results.append(result)
                print(f"frames={first}, run={repeat + 1}: native callback "
                      f"{result['native_first_callback_ms']:.1f} ms, first packet "
                      f"{result['first_packet_ready_ms']:.1f} ms, audio "
                      f"{result['first_packet_audio_ms']:.0f} ms", flush=True)
        report = {
            "platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "native_version": tts.library.version(),
            "backend": os.environ.get("GGML_BACKEND", "auto"),
            "talker": args.talker.name, "codec": args.codec.name,
            "codec_chunk_sec": args.codec_chunk_sec, "seed": 42,
            "max_new_tokens": 256, "text": args.text,
            "warmup_runs": 1, "repeats": args.repeats, "runs": results,
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("\nMedians: frames | native callback ms | packet ready ms | packet audio ms")
    for first in (1, 4, 8):
        runs = [r for r in results if r["first_chunk_frames"] == first]
        values = [statistics.median(r[key] for r in runs) for key in (
            "native_first_callback_ms", "first_packet_ready_ms", "first_packet_audio_ms",
        )]
        print(f"{first:>14} | {values[0]:>18.1f} | {values[1]:>15.1f} | {values[2]:>15.1f}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
