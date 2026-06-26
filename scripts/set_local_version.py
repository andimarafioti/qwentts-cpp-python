from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "src" / "qwentts_cpp" / "__init__.py"


def _replace(pattern: str, repl: str, text: str, path: Path) -> str:
    updated, count = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"Could not update version in {path}")
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Append a PEP 440 local version suffix for wheel builds.")
    parser.add_argument("suffix", help="Local version suffix without '+', for example cu130")
    args = parser.parse_args()

    suffix = args.suffix.strip().lstrip("+")
    if not re.fullmatch(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", suffix):
        raise SystemExit(f"Invalid local version suffix: {args.suffix!r}")

    pyproject_text = PYPROJECT.read_text()
    match = re.search(r'^version = "([^"]+)"$', pyproject_text, flags=re.MULTILINE)
    if not match:
        raise SystemExit(f"Could not find project version in {PYPROJECT}")
    base_version = match.group(1).split("+", 1)[0]
    version = f"{base_version}+{suffix}"

    PYPROJECT.write_text(
        _replace(r'^version = "[^"]+"$', f'version = "{version}"', pyproject_text, PYPROJECT)
    )
    INIT.write_text(
        _replace(r'^__version__ = "[^"]+"$', f'__version__ = "{version}"', INIT.read_text(), INIT)
    )
    print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
