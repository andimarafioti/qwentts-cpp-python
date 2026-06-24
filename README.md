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

The CI wheel build defaults to qwentts.cpp
`4536dcdce27c3764a93a06d6bf64026b124962f5`, which includes the scheduler
resets and ABI v2 cached voice-reference fields. CPU and CUDA both stay on the
backend prompt-projection path.

## CI Wheel Builds

The wheel workflow builds Linux CPU wheels, CUDA 12.8 wheels, and CUDA 13.0
wheels. Linux x86_64 CUDA wheels use Hugging Face Jobs via the
`hf-jobs-cpu-performance` runner label because compiling the full CUDA target
set can exceed GitHub-hosted runner time budgets.

Before the x86_64 CUDA jobs can run, set up `huggingface/jobs-actions` for this
repository:

1. Duplicate the `huggingface/jobs-actions-dispatcher` Space under your Hugging
   Face user or org and keep it on `cpu-upgrade` or better.
2. Use the dispatcher Space to create and install the generated GitHub App on
   `andimarafioti/qwentts-cpp-python`.
3. Add an `HF_TOKEN` with Jobs permissions to the dispatcher Space secrets.
4. Trigger the wheel workflow again. Jobs with `runs-on: hf-jobs-*` should then
   be picked up by ephemeral Hugging Face Jobs runners.

If the dispatcher is not installed or the Space is asleep, the x86_64 CUDA jobs
will stay queued with no available runner. The Linux aarch64 CUDA wheels still
use `ubuntu-24.04-arm`; Hugging Face Jobs runner labels do not currently select
an ARM64 GitHub Actions host.

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
