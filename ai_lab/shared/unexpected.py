"""Saying once that something failed where only an expected failure was allowed for.

Many readings here may fail in the ordinary run of things and are answered
with a safe default: a card probe that times out, an engine still starting,
an entry deleted while a page was open. Those are caught by name, with the
tuples below. Anything else is a bug of ours (a `KeyError` of a renamed field,
a `TypeError`) and must not hide behind the same default: the caller still
answers with its default, so a page or a decision never fails over a figure,
but `note` writes one line to the service log — once per place and kind of
error, so a loop that fails every second does not flood it.

A shared file: it decides nothing, and anything may import it.
"""

from __future__ import annotations

import http.client
import subprocess
import sys
import threading

# A probe of the machine that failed: a command that could not run, ran too
# long or printed something unreadable (`nvidia-smi`, `vm_stat`, a socket).
READING_FAILED = (OSError, subprocess.SubprocessError, ValueError)
# An answer over HTTP that did not come, or came unreadable.
CALL_FAILED = (OSError, http.client.HTTPException, ValueError)
# Something asked for by name that is gone or cannot be read: an entry, an
# engine (`KeyError` is their contract), a model's files.
LOOKUP_FAILED = (KeyError, OSError, ValueError)
# Asking upstream for a new version: the network, git, or the index refused.
UPSTREAM_FAILED = (OSError, http.client.HTTPException, subprocess.SubprocessError, ValueError)



class Notes:
    """Which places have already said which kind of failure."""

    def __init__(self) -> None:
        self._said: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def note(self, where: str, error: BaseException) -> None:
        """Log `error` raised in `where`, the first time that place fails that way."""
        key = (where, type(error).__name__)
        with self._lock:
            if key in self._said:
                return
            self._said.add(key)
        print(f"AI-Lab: unexpected {type(error).__name__} in {where}: {error} "
              "(answered with a default; said once)", file=sys.stderr, flush=True)

    def forget(self) -> None:
        """Say every failure again from now on, as after a restart."""
        with self._lock:
            self._said.clear()


# The manager's one record; `note` is what the places call.
NOTES = Notes()
note = NOTES.note
