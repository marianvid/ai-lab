"""What is waiting in a git checkout, read as commits.

llama.cpp is a checkout on this machine, so the changes are already on disk
after a fetch: no network beyond that, no service to be rate-limited by, and it
works when the machine is offline. Each commit's subject line is one change.

A commit is not a release note. Nobody wrote it for a reader, and there are a
lot of them — 138 waiting on the container in one measurement. Making them
readable is `sifting`'s job, not this file's; here they are only read out.
"""

from __future__ import annotations

import subprocess

from ..types import Change

# Long enough for a fetch of a busy repository over a slow line, short enough
# that a wedged network does not hold a page open indefinitely.
FETCH_TIMEOUT_S = 120
READ_TIMEOUT_S = 20

# No more than this many, newest first. The number is not a judgement about
# what matters — the sorting does that — but a checkout that is a year behind
# would otherwise put ten thousand lines into one page and make the browser
# unusable. Whatever is cut is reported, never dropped silently.
MOST = 400


def waiting(path: str, fetch: bool = True) -> tuple[list[Change], int]:
    """Commits on the remote that this checkout does not have.

    Returns them newest first, with the number left out by `MOST`.

    Raises `ValueError` with something a person can read: this runs behind a
    button somebody pressed, and "no git checkout at /opt/ai/llama.cpp" is an
    answer while a stack trace is not.
    """
    if fetch:
        _git(path, "fetch", "--tags", "--prune", timeout=FETCH_TIMEOUT_S)
    # %s is the subject — the first line, which is the one written to be read.
    # %h is the short hash, which is what identifies it upstream.
    output = _git(path, "log", "--format=%h%x1f%s", "HEAD..origin/HEAD",
                  timeout=READ_TIMEOUT_S)
    lines = [line for line in output.splitlines() if line.strip()]
    from .sifting import area_of
    changes = []
    for line in lines[:MOST]:
        reference, _, title = line.partition("\x1f")
        changes.append(Change(area=area_of(title), title=title.strip(),
                              reference=reference.strip()))
    return changes, max(0, len(lines) - MOST)


def _git(path: str, *arguments: str, timeout: int) -> str:
    try:
        result = subprocess.run(["git", *arguments], cwd=path,
                                capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"git {arguments[0]} failed: {error}") from None
    if result.returncode != 0:
        raise ValueError(result.stderr.strip().splitlines()[0]
                         if result.stderr.strip()
                         else f"git {arguments[0]} failed")
    return result.stdout
