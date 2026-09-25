# Repository Instructions

- Never include `codex` in branch names or pull request titles.
- Keep release pull requests focused on version metadata and release documentation.
- Do not commit local build artifacts such as `dist/`, `build/`, generated wheels, generated sdists, or copied shared libraries.

## Publishing to PyPI

PyPI publishing is handled by GitHub Actions in `.github/workflows/publish.yml`.
The workflow runs on pushed tags that match `v*`, builds the public Linux CUDA
12.8 and macOS Metal wheels, checks the artifacts with `twine check --strict`, and publishes
through the configured `pypi` environment.

The public `qwentts-cpp-python` package publishes CUDA 12.8 wheels for Linux
and Metal wheels for macOS 14+ arm64, using plain public versions for both.
Their platform tags let pip select the appropriate wheel automatically.
Publishing multiple backend flavors under the same package name, version,
and platform tag would leave pip with no reliable way to choose the intended
runtime.

The validation wheel workflow is dispatch-only. Its downloadable artifacts are
not consumed by either publishing workflow; both publishers rebuild their own
wheels from the pinned qwentts.cpp source.

Additional backend-specific wheels can be published to Hugging Face Hub by
manually dispatching `.github/workflows/publish-hf-wheels.yml`. That workflow
builds local-version variants such as `0.2.0+cpu`, `0.2.0+cu124`,
`0.2.0+cu128`, `0.2.0+cu130`, and `0.3.1+metal`, prepares static `--find-links` pages,
creates the public dataset repo if needed, and uploads the wheel index using
the `HF_TOKEN` repository secret. Do not upload those local-version variants to
PyPI.

The `+metal` flavor is a macOS 14+ arm64 wheel with embedded Metal shaders and
bundled native dylibs. The reusable `.github/workflows/metal-wheel.yml` builds,
repairs, validates ABI layouts, and tests a clean wheel install. It runs for PRs
and is also called by validation, PyPI publishing (with `pypi: true` to omit
the local version suffix), and Hugging Face publishing. GPU synthesis is
tested locally with `scripts/smoke_stream.py --require-metal` and local weights.

Pull requests do not run the Linux wheel matrix. Do not run the validation
workflow solely as a publishing prerequisite, because the publishing workflows
perform fresh builds.

## Updating the qwentts.cpp Pin

To rebuild against a newer qwentts.cpp revision:

1. Resolve the latest upstream `master` commit to its full SHA.
2. Update every default and event fallback for `QWENTTS_REF` in
   `.github/workflows/wheels.yml`, `.github/workflows/publish.yml`,
   `.github/workflows/publish-hf-wheels.yml`, and `.github/workflows/metal-wheel.yml`.
   Update `QWENTTS_NATIVE_REVISION` in `src/qwentts_cpp/_binding.py` and verify
   ctypes layouts with `QWENTTS_CPP_SOURCE=/path/to/source python -m pytest tests/test_native_abi.py`.
3. Update the pinned revision and its summary in `README.md`.
4. Open a focused pull request containing the pin and documentation changes.
5. After merging, use the normal publishing paths when a release is intended:
   the version-bump and tag flow below for PyPI, and the Hugging Face publishing
   workflow for backend-specific wheels. Each publisher performs a fresh build.

Do not include locally generated libraries, wheels, sdists, `build/`, or
`dist/` contents in the pull request.

CUDA release wheels keep native cubins for sm_86, sm_90, and sm_120, plus PTX
fallbacks for sm_75 and the newest supported CUDA architecture. This keeps the
public CUDA 12.8 wheels under PyPI's default per-file upload limit while
preserving broad GPU compatibility. Ada sm_89 GPUs can run the sm_86 cubin
through CUDA's same-major binary compatibility. DGX Spark / GB10 sm_121 uses
PTX fallback in the public CUDA 12.8 wheels. A CUDA 13 build with native sm_121
can avoid that fallback, but it is not the PyPI default because the deployment
server image currently targets CUDA 12.8.

To prepare a release:

1. Confirm the intended version is not already published on PyPI.
2. Bump `version` in `pyproject.toml`.
3. Bump `__version__` in `src/qwentts_cpp/__init__.py`.
4. Open and merge a pull request with only the release preparation changes.

To publish after the release PR is merged:

1. Update `main` locally: `git checkout main && git pull origin main`.
2. Create an annotated tag for the version: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`.
3. Push the tag: `git push origin vX.Y.Z`.
4. Watch the `Publish` GitHub Actions workflow complete successfully.
5. Verify the new version appears at `https://pypi.org/project/qwentts-cpp-python/`.

Only upload manually if the GitHub Actions workflow is unavailable and the
maintainers have explicitly chosen that fallback.
