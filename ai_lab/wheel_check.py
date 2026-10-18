"""Check that a built wheel serves the same application files as the checkout.

Run by CI after building: `python -m ai_lab.wheel_check dist/ai_lab-*.whl`.
Every `.py`, `.js`, `.css` and `.html` file under `ai_lab/` must be in the
wheel, and nothing from the private `opts/` folder may be.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile

SHIPPED_SUFFIXES = {".py", ".js", ".css", ".html"}


def application_files(source: Path) -> set[str]:
    """The checkout's application files, as paths inside a wheel."""
    return {
        path.relative_to(source).as_posix()
        for path in (source / "ai_lab").rglob("*")
        if path.is_file() and path.suffix in SHIPPED_SUFFIXES
        and "__pycache__" not in path.parts
    }


def differences(wheel: Path, source: Path) -> tuple[list[str], list[str]]:
    """(application files missing from the wheel, private files in it)."""
    with ZipFile(wheel) as archive:
        packaged = set(archive.namelist())
    missing = sorted(application_files(source) - packaged)
    private = sorted(name for name in packaged if name.startswith("opts/"))
    return missing, private


def main() -> None:
    """Compare the wheel named on the command line with the checkout; exit 1 on a difference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    arguments = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    missing, private = differences(arguments.wheel, source)
    if missing or private:
        raise SystemExit(f"Missing application files: {missing}; private files: {private}")
    print(f"Wheel contains all {len(application_files(source))} application files "
          f"and no private opts files")


if __name__ == "__main__":
    main()
