#!/usr/bin/env python3
"""Check that a built wheel serves the same application files as the checkout."""

import argparse
from pathlib import Path
from zipfile import ZipFile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    arguments = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    required = {
        path.relative_to(source).as_posix()
        for path in (source / "ai_lab").rglob("*")
        if path.is_file() and path.suffix in {".py", ".js", ".css", ".html"}
        and "__pycache__" not in path.parts
    }
    with ZipFile(arguments.wheel) as archive:
        packaged = set(archive.namelist())
    missing = sorted(required - packaged)
    private = sorted(name for name in packaged if name.startswith("opts/"))
    if missing or private:
        raise SystemExit(f"Missing application files: {missing}; private files: {private}")
    print(f"Wheel contains all {len(required)} application files and no private opts files")


if __name__ == "__main__":
    main()
