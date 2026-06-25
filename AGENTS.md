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
only. CPU and CUDA 13 wheels are useful validation artifacts, but publishing
multiple backend flavors under the same package name, version, and platform tag
would leave pip with no reliable way to choose the intended runtime.

CUDA release wheels keep native cubins for sm_86, sm_90, and sm_120, plus PTX
fallbacks for sm_75 and the newest supported CUDA architecture. This keeps the
public CUDA 12.8 wheels under PyPI's default per-file upload limit while
preserving broad GPU compatibility. Ada sm_89 GPUs can run the sm_86 cubin
through CUDA's same-major binary compatibility. DGX Spark / GB10 sm_121 uses
PTX fallback in the public CUDA 12.8 wheels; CUDA 13 validation wheels include
an sm_121 native cubin.

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
