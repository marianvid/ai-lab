"""A private folder for one test, removed when the test ends (the reviewer's L on mkdtemp)."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path


def folder(test: unittest.TestCase, prefix: str = "ai-lab-test-") -> Path:
    """A new empty folder, removed with everything in it once `test` is done."""
    path = Path(tempfile.mkdtemp(prefix=prefix))
    test.addCleanup(shutil.rmtree, path, ignore_errors=True)
    return path


# A path that is never there: inside this checkout's test support, under a name
# nothing creates. "/nowhere" could exist on a runner (the reviewer's L).
ABSENT = Path(__file__).resolve().parent / "no-such-folder-for-tests"
