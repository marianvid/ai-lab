"""Comparing two versions of one package the way Python's packaging does.

`0.27.1rc2.dev949` is BEFORE `0.27.1rc2`, which is before `0.27.1` (audit m25,
L32): a development build of a release candidate comes before the candidate,
and a candidate before the release. Comparing only the numbers called the
nightly "0.27.1" and dropped that release's notes; comparing every number
called rc2 -> final a step backwards.

Written here rather than imported from `packaging`, which is not one of the
manager's dependencies. It covers what the engines publish: `N(.N)*`, then
`aN`/`bN`/`rcN`, `.postN`, `.devN`, and a `+local` part, which is ignored. A
leading `v` is allowed.
"""

from __future__ import annotations

import re

SHAPE = re.compile(r"^v?(\d+(?:\.\d+)*)(?:\.?(a|b|rc)(\d+))?(?:\.?post(\d+))?(?:\.?dev(\d+))?",
                   re.IGNORECASE)
PRE = {"a": 0, "b": 1, "rc": 2}


def _stage(pre: str | None, pre_n: str | None, post: str | None, dev: str | None) -> tuple:
    """A candidate ranks by its letter and number; a bare development build comes before any."""
    if pre:
        return (1, PRE[pre.lower()], int(pre_n or 0))
    if dev is not None and post is None:
        return (0,)
    return (2,)


def version_key(text: str) -> tuple | None:
    """A key that sorts versions in packaging's order; None when `text` is not one."""
    match = SHAPE.match((text or "").strip().split("+")[0])
    if not match:
        return None
    numbers, pre, pre_n, post, dev = match.groups()
    release = [int(part) for part in numbers.split(".")]
    while len(release) > 1 and release[-1] == 0:
        release.pop()
    return (tuple(release), _stage(pre, pre_n, post, dev),
            (1, int(post)) if post is not None else (0,),
            (0, int(dev)) if dev is not None else (1,))
