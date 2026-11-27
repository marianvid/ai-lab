"""The names a YuE2 song and a window call go by, and the check that keeps a name in its folder.

Read by the manager (the music routes and `songs/`). The YuE2 adapter runs in
its own process and may import nothing of the manager's, so it keeps its own
copy of the same two rules (`music/backends/yue2_web_backend.py`); a test holds
the two equal, so a name the front door lets through is a name the Studio can
hold, and the reverse.
"""

from __future__ import annotations

import re

# The caller's name for one window call, used to cancel it (`DELETE .../{request_id}`).
REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
# A generation is a YuE2 Studio output folder; its name follows the studio's own rule.
# The first character is never a dot, so a name cannot climb out of its folder.
GENERATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}")


def check_generation_id(value) -> str:
    """A generation's id (a studio output folder's name), refused when it could leave that folder."""
    if not isinstance(value, str) or not GENERATION_ID.fullmatch(value):
        raise ValueError("generation_id is not a generation's name")
    return value
