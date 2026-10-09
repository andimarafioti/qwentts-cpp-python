from __future__ import annotations

import argparse
import html
import re
import shutil
from pathlib import Path, PurePosixPath


WHEEL_RE = re.compile(
    r"^qwentts_cpp_python-(?P<version>[^-]+)-py3-none-(?P<platform>.+)\.whl$"
)


def _flavor_from_wheel(path: Path) -> str:
    match = WHEEL_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected wheel filename: {path.name}")
    version = match.group("version")
    if "+" not in version:
        raise ValueError(f"Wheel does not include a local version flavor: {path.name}")
    return version.split("+", 1)[1]


def _version_from_wheel(path: Path) -> str:
    match = WHEEL_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected wheel filename: {path.name}")
    return match.group("version")


def _existing_wheels(repo_id: str, replaced_flavors: set[str]) -> list[Path]:
    """Read published filenames only, keeping the latest version per other flavor."""
    from huggingface_hub import HfApi
    from packaging.version import Version

    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    by_flavor: dict[str, list[Path]] = {}
    for filename in files:
        parts = PurePosixPath(filename).parts
        if len(parts) != 3 or parts[0] != "whl" or not WHEEL_RE.match(parts[2]):
            continue
        wheel = Path(parts[2])
        flavor = _flavor_from_wheel(wheel)
        if parts[1] == flavor and flavor not in replaced_flavors:
            by_flavor.setdefault(flavor, []).append(wheel)
    selected = []
    for wheels in by_flavor.values():
        latest = max(Version(_version_from_wheel(wheel)) for wheel in wheels)
        selected.extend(wheel for wheel in wheels if Version(_version_from_wheel(wheel)) == latest)
    return selected


def _write_links_page(path: Path, title: str, links: list[tuple[str, str]]) -> None:
    items = "\n".join(
        f'      <li><a href="{html.escape(href)}">{html.escape(label)}</a></li>' for href, label in links
    )
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "  <head>",
                '    <meta charset="utf-8">',
                f"    <title>{html.escape(title)}</title>",
                "  </head>",
                "  <body>",
                f"    <h1>{html.escape(title)}</h1>",
                "    <ul>",
                items,
                "    </ul>",
                "  </body>",
                "</html>",
                "",
            ]
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build static Hugging Face wheel index pages.")
    parser.add_argument("--dist", type=Path, required=True, help="Directory containing built wheels")
    parser.add_argument("--out", type=Path, required=True, help="Output directory to upload")
    parser.add_argument("--repo-id", required=True, help="HF dataset repo id used in generated README")
    parser.add_argument("--preserve-existing", action="store_true",
                        help="Keep indexes for other published flavors without downloading their wheels")
    args = parser.parse_args()

    wheels = sorted(args.dist.rglob("qwentts_cpp_python-*.whl"))
    if not wheels:
        raise SystemExit(f"No wheels found under {args.dist}")

    existing = _existing_wheels(args.repo_id, {_flavor_from_wheel(wheel) for wheel in wheels}) \
        if args.preserve_existing else []

    if args.out.exists():
        shutil.rmtree(args.out)
    wheel_root = args.out / "whl"
    wheel_root.mkdir(parents=True)

    by_flavor: dict[str, list[Path]] = {}
    versions_by_flavor: dict[str, str] = {}
    for wheel in wheels:
        flavor = _flavor_from_wheel(wheel)
        version = _version_from_wheel(wheel)
        previous = versions_by_flavor.setdefault(flavor, version)
        if previous != version:
            raise ValueError(f"Mixed versions for {flavor}: {previous} and {version}")
        target_dir = wheel_root / flavor
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / wheel.name
        shutil.copy2(wheel, target)
        by_flavor.setdefault(flavor, []).append(target)

    for wheel in existing:
        flavor = _flavor_from_wheel(wheel)
        versions_by_flavor[flavor] = _version_from_wheel(wheel)
        target_dir = wheel_root / flavor
        target_dir.mkdir(parents=True, exist_ok=True)
        # These links address files already on the Hub. Only new local wheels
        # are copied and uploaded; existing wheel payloads remain untouched.
        by_flavor.setdefault(flavor, []).append(target_dir / wheel.name)

    flavor_links: list[tuple[str, str]] = []
    for flavor in sorted(by_flavor):
        files = sorted(by_flavor[flavor])
        _write_links_page(
            wheel_root / f"{flavor}.html",
            f"qwentts-cpp-python {flavor} wheels",
            [(f"{flavor}/{wheel.name}", wheel.name) for wheel in files],
        )
        flavor_links.append((f"{flavor}.html", flavor))

    _write_links_page(wheel_root / "index.html", "qwentts-cpp-python wheels", flavor_links)

    args.out.joinpath("README.md").write_text(
        "\n".join(
            [
                "---",
                "license: mit",
                "---",
                "",
                "# qwentts-cpp-python wheels",
                "",
                "Optional backend-specific wheel variants for `qwentts-cpp-python`.",
                "",
                "The public PyPI package provides Linux CUDA 12.8 and macOS Metal wheels:",
                "",
                "```bash",
                "python -m pip install --upgrade qwentts-cpp-python",
                "```",
                "",
                "Install a backend-specific wheel from this repository with `--find-links`:",
                "",
                "```bash",
                *[
                    f'pip install "qwentts-cpp-python=={versions_by_flavor[flavor]}" -f https://huggingface.co/datasets/{args.repo_id}/tree/main/whl/{flavor}'
                    for flavor in sorted(versions_by_flavor)
                ],
                "```",
                "",
                "CUDA wheels require the matching system CUDA runtime and cuBLAS libraries.",
                "ROCm wheels require the matching system ROCm runtime, hipBLAS, hipBLASLt,",
                "rocBLAS, and rocBLAS/Tensile assets; these are not bundled in the wheel.",
                "The `rocm724` wheel targets Linux x86_64, Ubuntu 24.04, ROCm 7.2.4, and",
                "MI300X (`gfx942`). Its `linux_x86_64` tag does not check the Linux distribution",
                "or GPU architecture; install it only in the documented environment.",
                "It does not claim manylinux portability or support for other AMD targets.",
                "",
            ]
        )
    )
    print(f"Prepared {len(wheels)} new wheels and indexes for {', '.join(sorted(by_flavor))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
