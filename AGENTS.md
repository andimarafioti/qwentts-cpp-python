# Repository Instructions

- Never include `codex` in branch names or pull request titles.
- Keep release pull requests focused on version metadata and release documentation.
- Do not commit local build artifacts such as `dist/`, `build/`, generated wheels, generated sdists, or copied shared libraries.

## Publishing to PyPI

PyPI publishing is handled by GitHub Actions in `.github/workflows/publish.yml`.
The workflow runs on pushed tags that match `v*`, builds the public Linux CUDA
12.8 wheels, checks the artifacts with `twine check --strict`, and publishes
through the configured `pypi` environment.

The public `qwentts-cpp-python` package currently publishes CUDA 12.8 wheels
only. Publishing multiple backend flavors under the same package name, version,
and platform tag would leave pip with no reliable way to choose the intended
runtime.

The PR wheel workflow may still build CPU and CUDA 13 wheels as downloadable
validation artifacts. Those artifacts are not uploaded to PyPI by the publish
workflow.

Additional CUDA wheels can be published to Hugging Face Hub by
`.github/workflows/publish-hf-wheels.yml`. That workflow builds local-version
variants such as `0.2.0+cu124`, `0.2.0+cu128`, and `0.2.0+cu130`, prepares
static `--find-links` pages, creates the public dataset repo if needed, and
uploads the wheel index using the `HF_TOKEN` repository secret. Do not upload
those local-version CUDA variants to PyPI.

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
