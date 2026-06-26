from __future__ import annotations

import argparse
import html
import re
import shutil
from pathlib import Path


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
    args = parser.parse_args()

    wheels = sorted(args.dist.rglob("qwentts_cpp_python-*.whl"))
    if not wheels:
        raise SystemExit(f"No wheels found under {args.dist}")

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
                "The default PyPI package is CUDA 12.8:",
                "",
                "```bash",
                "pip install qwentts-cpp-python",
                "```",
                "",
                "Install a backend-specific wheel from this repository with `--find-links`:",
                "",
                "```bash",
                *[
                    f'pip install "qwentts-cpp-python=={versions_by_flavor[flavor]}" -f https://huggingface.co/datasets/{args.repo_id}/resolve/main/whl/{flavor}.html'
                    for flavor in sorted(versions_by_flavor)
                ],
                "```",
                "",
                "The wheels do not bundle CUDA runtime or cuBLAS libraries. Use a base image or",
                "system installation that provides the matching CUDA runtime.",
                "",
            ]
        )
    )
    print(f"Prepared {sum(len(v) for v in by_flavor.values())} wheels for {', '.join(sorted(by_flavor))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
