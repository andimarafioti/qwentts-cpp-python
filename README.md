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
python -m build --wheel
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
