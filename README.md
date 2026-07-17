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

`--backend cuda` is the default because `faster-qwen3-tts` is a CUDA-first
package. CPU builds are still useful for development and smoke tests, but they
are not the primary release target.

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

Pull requests do not build the wheel matrix. The PyPI and Hugging Face
publishing workflows each rebuild fresh wheels from the pinned qwentts.cpp
revision; validation artifacts are not reused for publishing.

The CI wheel build defaults to qwentts.cpp
`7df559a8ca25f66fee02970514ebe5f01dee9055`, which retains ABI v2 and includes
the latest static-graph, streaming-decode, and widened voice-route changes.

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
