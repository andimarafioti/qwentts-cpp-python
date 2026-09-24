# qwentts-cpp-python

Python bindings and wheel packaging for Pascal's `qwentts.cpp` C ABI.

This package is intentionally small:

- it loads `libqwen` with `ctypes`
- it exposes buffered and streaming synthesis
- it can bundle prebuilt `libqwen`/`libggml` binaries in platform wheels
- it does not bundle GGUF model weights

CUDA development build with an existing `qwentts.cpp` checkout:

```bash
python scripts/build_native.py \
  --source /path/to/qwentts.cpp \
  --backend cuda \
  --clean
QWENTTS_CPP_WHEEL_BUILD_TAG=1cu128 python -m build --wheel
```

CPU development build:

```bash
python scripts/build_native.py \
  --source /path/to/qwentts.cpp \
  --backend cpu \
  --clean
QWENTTS_CPP_WHEEL_BUILD_TAG=1cpu python -m build --wheel
```

`--backend metal` is the default on macOS; other platforms default to CUDA.
CPU builds are still useful for development and smoke tests.

## Installation

The default PyPI package is built for CUDA 12.8:

```bash
pip install qwentts-cpp-python
```

Additional backend-specific wheels are published to Hugging Face Hub as local-version
variants. Use them when the PyPI CUDA 12.8 wheel does not match the runtime or
GPU target, for example DGX Spark / GB10 with CUDA 13:

```bash
pip install "qwentts-cpp-python==0.3.0+cpu" \
  -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cpu

pip install "qwentts-cpp-python==0.3.0+cu124" \
  -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cu124

pip install "qwentts-cpp-python==0.3.0+cu128" \
  -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cu128

pip install "qwentts-cpp-python==0.3.0+cu130" \
  -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cu130
```

These commands use pip's `--find-links` mode against the Hugging Face directory
page for the selected flavor. Dependencies still resolve from PyPI normally.
The wheels do not bundle CUDA runtime or cuBLAS libraries; use a base image or
system installation that provides the matching CUDA runtime.

The Hugging Face wheel pages may contain multiple Linux compatibility tags for
the same backend flavor. For example, the `cu128` page can host both
`manylinux_2_35` wheels for Ubuntu 22.04+ and `manylinux_2_39` wheels for
Ubuntu 24.04+. Pip selects the newest compatible wheel for the current machine.

Pull requests do not build the Linux wheel matrix. The PyPI and Hugging Face
publishing workflows each rebuild fresh wheels from the pinned qwentts.cpp
revision; validation artifacts are not reused for publishing.

### Apple Silicon (Metal)

The Hugging Face publisher builds a `+metal` wheel for **macOS 14 or newer,
arm64 Python 3.10+**. After this change is merged and the **Publish Hugging Face
Wheels** workflow has completed, install it with:

```bash
python -m pip install --only-binary=qwentts-cpp-python "qwentts-cpp-python==0.3.1+metal" \
  -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/metal
python -c "from qwentts_cpp import QwenLibrary; print(QwenLibrary().version())"
```

No local CMake build, Homebrew libraries, or explicit `library_path` is needed.
The wheel bundles libqwen and the ggml CPU, Metal, BLAS, and core libraries;
Metal shader source is embedded in the Metal library. Apple frameworks come
from macOS. The build applies a shader-only workaround for the pinned ggml's
invalid scalar-to-BF16-vector fill cast, restoring the source checkout afterward;
the C ABI is unchanged. Use a native arm64 Python, rather than an Intel Python
under Rosetta.
The `+metal` version identifies the backend on Hugging Face; it is never sent to
PyPI and cannot collide with the Linux CUDA or other backend variants.

Before publication, download the `hf-wheel-metal-macosx-arm64` artifact from
the PR's **Apple Silicon Metal wheel** check, unzip it, and install its `.whl`
with `python -m pip install /path/to/qwentts_cpp_python-0.3.1+metal-*.whl`.
This macOS check also runs for PRs; the Linux matrix remains dispatch-only.
The Hugging Face publisher rebuilds and checks its own Metal wheel.

Run a streaming smoke test using local GGUF weights (weights are not bundled):

```bash
python scripts/smoke_stream.py \
  --talker /path/to/qwen-talker-1.7b-base-Q8_0.gguf \
  --codec /path/to/qwen-tokenizer-12hz-Q8_0.gguf \
  --ref-spk /path/to/reference.spk \
  --require-metal --output metal-smoke.wav
```

For a CustomVoice talker, replace `--ref-spk` with `--speaker Vivian`; for a
VoiceDesign talker, use `--instruct "A calm, clear voice."`. `--require-metal`
forces the Metal device so the test fails if only CPU execution is available.
The script checks multiple nonempty, finite audio chunks and writes a mono
24 kHz PCM WAV. Listen to it with `afplay metal-smoke.wav`. Hosted macOS CI
checks packaging, ABI layouts, and installed loading; synthesis requires a Mac
with a working Metal device and local weights.

To build the same wheel locally with the pinned source checkout:

```bash
python -m pip install build delocate twine
export MACOSX_DEPLOYMENT_TARGET=14.0
python scripts/set_local_version.py metal  # changes local version metadata
python scripts/build_native.py --backend metal --clean
python -m build --wheel --config-setting=--build-option=--plat-name=macosx_14_0_arm64
delocate-wheel --require-archs arm64 -w wheelhouse -v dist/*.whl
python -m twine check --strict wheelhouse/*.whl
```

### Native ABI compatibility

The CI wheel build defaults to qwentts.cpp
`7df559a8ca25f66fee02970514ebe5f01dee9055`, which retains ABI v2 and includes
the latest static-graph, streaming-decode, and widened voice-route changes.

The loader verifies this native revision before calling functions that write
ctypes parameter buffers. Upstream does not expose an ABI-version or struct-size
query, so other revisions (including unknown builds) are rejected with an
actionable error even if they may be compatible. When updating the pin, update
`QWENTTS_NATIVE_REVISION` in the binding and run `tests/test_native_abi.py` with
`QWENTTS_CPP_SOURCE` pointing to the new checkout to verify every struct size
and field offset. An incompatible library fails before model loading.

`QWENTTS_CPP_WHEEL_BUILD_TAG` is useful for local wheelhouses. For public
indexes, publish one backend flavor per package/version/platform compatibility
tag; otherwise pip has no way to choose between CPU and CUDA binaries.

Local smoke test with a built library:

```bash
QWENTTS_CPP_LIBRARY=/path/to/libqwen.so python - <<'PY'
from qwentts_cpp import QwenLibrary
lib = QwenLibrary()
print(lib.version())
PY
```

Model files are resolved with `huggingface-hub` by `QwenTTS.from_pretrained(...)`
or passed directly to `QwenTTS(...)` as GGUF paths.

## Streaming packet sizes

`QwenTTS.stream()` defaults to a **4-frame first packet** (320 ms of mono
24 kHz audio). Choose `first_chunk_frames=1`, `2`, `4`, or `8` for 80, 160,
320, or 640 ms of initial audio. A larger packet gives playback more audio
to start with, at the cost of waiting longer before the first yield.

```python
for audio, sample_rate in tts.stream(
    text="The sky is blue today.",
    ref_spk_emb=spk,             # Base model; use speaker= for CustomVoice
    first_chunk_frames=4,
    codec_chunk_sec=0.64,       # Default later packets: 8 frames, independently sized
):
    play_audio(audio, sample_rate)  # Your playback/transport function
```

Later packets default to `codec_chunk_sec=0.64` (8 frames), matching the native
steady-state width without adding a larger batching delay after the first packet.
Explicit values round to the nearest 80 ms frame, with a one-frame minimum
(for example, 1.0 second rounds to 13 frames / 1.04 seconds).
The value must be finite and positive. Successful end-of-speech or the token
limit flushes any remaining audio as a short packet, including utterances
shorter than the requested first packet. Cancellation or errors discard the
unfinished packet; closing the iterator requests native cancellation.

Packet assembly happens in Python, using the existing verified ABI v2 library.
The native decoder still emits its fixed 1→2→4→8-frame ramp, then 8-frame
chunks. Consequently, first packets of 1, 2, 4, and 8 frames become available
after native output has reached 1, 3, 7, and 15 frames respectively (or earlier
at end-of-speech). Packet boundaries preserve every PCM sample but do not
change native decode scheduling. Later packets may become available together
when a native callback spans several packet boundaries; this is not a timed
playback scheduler. `codec_left_context_sec` is ignored by the stateful native
stream. Buffered `synthesize()` retains its native codec chunking behavior.

`last_stream_profile` keeps `first_callback_*` and `callback_count` for raw
native callbacks. `first_packet_ready_ms`, `first_packet_audio_s`, and
`packet_count` describe the assembled Python packets; `first_yield_ms` measures
delivery to the caller. This distinction includes the buffering cost in latency
measurements instead of treating the first native callback as audible output.

Benchmark 1, 4, and 8 frames on local hardware (one warm-up, then three measured
runs per setting, with rotated order):

```bash
python scripts/benchmark_first_chunk.py \
  --talker /path/to/qwen-talker-1.7b-base-Q8_0.gguf \
  --codec /path/to/qwen-tokenizer-12hz-Q8_0.gguf \
  --ref-spk /path/to/reference.spk \
  --require-metal --output /tmp/first-chunk-benchmark.json
```

For other model types, use `--speaker` or `--instruct` instead of `--ref-spk`.
Omit `--require-metal` on other backends. The report includes native first
callback time, first Python packet time/duration, and every packet's size.
The benchmark checks first/later packet sizes, finite non-silent audio, and the
final tail. Playback smoothness still depends on synthesis speed and the
player's buffering policy.

Example measurement on an Apple M3 Pro with Metal, macOS 26.6.2, Python
3.12.13, the Q8_0 1.7B Base talker and Q8_0 codec above, and a cached speaker
embedding (three runs per setting after warm-up, medians):

| First frames | Native first callback | First Python packet ready | First packet audio |
| --- | --- | --- | --- |
| 1 | 94.4 ms | 94.5 ms | 80 ms |
| 4 (default) | 84.2 ms | 369.8 ms | 320 ms |
| 8 | 85.7 ms | 826.4 ms | 640 ms |

All runs used the default 8-frame later packets (`codec_chunk_sec=0.64`), preserved the
115-frame utterance, and flushed the final short tail. Four frames provide a
middle ground on this machine; these measurements do not guarantee gap-free
playback on other hardware. Raw native callbacks remain one frame initially;
the larger first packet is assembled by the binding.

## Cached voice references

qwentts.cpp ABI v2 can skip reference WAV encoding for Base voice cloning by
passing precomputed latents:

- `.spk`: raw float32 speaker embedding from `qwen-codec --talker`
- `.rvq`: packed 11-bit reference codec stream from `qwen-codec`

The wrapper can create those files in-process from decoded mono float32 audio at
24 kHz:

```python
from qwentts_cpp import QwenTTS

tts = QwenTTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", quant="Q4_K_M")

# ref_audio_24k is a 1-D numpy float32 array, already resampled to 24 kHz.
voice_ref = tts.extract_voice_ref(ref_audio_24k)
voice_ref.save("reference.spk", "reference.rvq")
```

```python
from qwentts_cpp import QwenTTS, load_speaker_embedding

tts = QwenTTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", quant="Q4_K_M")

spk = load_speaker_embedding("reference.spk")
audio, sr = tts.synthesize(
    text="The sky is blue today.",
    lang="english",
    ref_spk_emb=spk,
    max_new_tokens=128,
)
```

For ICL clone mode, load the RVQ matrix with the model's codebook count and
also pass the reference transcript:

```python
from qwentts_cpp import load_rvq_codes

rvq = load_rvq_codes("reference.rvq", tts.num_codebooks())
audio, sr = tts.synthesize(
    text="The sky is blue today.",
    lang="english",
    ref_spk_emb=spk,
    ref_codes=rvq,
    ref_text="Transcript of the reference audio.",
)
```
