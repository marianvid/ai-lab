"""A vendor's private environment file: how a CLI signs in without a browser.

The owner writes one line into a file that lives only on the machine (never
in a repository)::

    CLAUDE_CODE_OAUTH_TOKEN=<the token claude setup-token printed>

`providers.vendors.<vendor>.env_file` names that file. Its variables are given
to the vendor's CLI on every call, so a token written while AI-Lab runs is
used from the next call on, with no restart. A missing or empty file means the
CLI's own login, as before.

Values go only to the CLI's environment (`read`); nothing logs or shows one.
`describe`, for the page, says only whether a token is in use, since when
(the file's date: the day the owner wrote it), and what is wrong.
"""

from __future__ import annotations

import re
import stat
import time
from pathlib import Path

# `claude setup-token` makes a token valid for one year; warn a month before.
TOKEN_WARNING_AFTER_DAYS = 335
TOKEN_LIFETIME_DAYS = 365
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DAY_S = 86400.0
_QUOTES = ("\"", "'")


def read(path: str) -> dict[str, str]:
    """The variables in the file; empty when there is no file or it cannot be read."""
    if not path:
        return {}
    try:
        text = Path(path).read_text()
    except (OSError, UnicodeDecodeError):
        return {}
    return _parse(text)[0]


def describe(path: str, now: float | None = None) -> dict:
    """What the page shows: `token` or `login`, the date, any warning. No value."""
    if not path:
        return {"credential": "login"}
    file = Path(path)
    try:
        info = file.stat()
        values, unread = _parse(file.read_text())
    except FileNotFoundError:
        return {"credential": "login", "env_file": path}
    except (OSError, UnicodeDecodeError) as error:
        return {"credential": "login", "env_file": path,
                "warning": f"the file cannot be read ({type(error).__name__}): "
                           f"the CLI uses its own login"}
    if not values:
        return {"credential": "login", "env_file": path}
    age_days = max(int(((time.time() if now is None else now) - info.st_mtime) // _DAY_S), 0)
    state = {"credential": "token", "env_file": path,
             "set_on": time.strftime("%Y-%m-%d", time.localtime(info.st_mtime)),
             "age_days": age_days}
    warnings = _warnings(age_days, stat.S_IMODE(info.st_mode), unread)
    if warnings:
        state["warning"] = "; ".join(warnings)
    return state


def _warnings(age_days: int, mode: int, unread: list[int]) -> list[str]:
    """What needs the owner: an old token, a file others can read, a stray line."""
    warnings = []
    if age_days >= TOKEN_WARNING_AFTER_DAYS:
        left = TOKEN_LIFETIME_DAYS - age_days
        warnings.append(f"the token was set {age_days} days ago and lasts about "
                        f"{TOKEN_LIFETIME_DAYS}: make a new one with claude setup-token"
                        + (f" ({left} days left)" if left > 0 else " (it may have ended)"))
    if mode & 0o077:
        warnings.append("others on this machine can read the file: chmod 600 it")
    if unread:
        warnings.append(f"line(s) {', '.join(map(str, unread))} not understood "
                        f"(write NAME=value)")
    return warnings


def _parse(text: str) -> tuple[dict[str, str], list[int]]:
    """`NAME=value` lines (an `export ` before them and quotes around are allowed).

    Returns the variables with a value, and the numbers of the lines that are
    neither one of those, nor blank, nor a `#` comment.
    """
    values: dict[str, str] = {}
    unread: list[int] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pair = _pair(line.removeprefix("export ").strip())
        if pair is None:
            unread.append(number)
        elif pair[1]:
            values[pair[0]] = pair[1]
    return values, unread


def _pair(line: str) -> tuple[str, str] | None:
    """`NAME=value` as (name, value without surrounding quotes); None if not that."""
    name, equals, value = line.partition("=")
    name, value = name.strip(), value.strip()
    if not equals or not _NAME.match(name):
        return None
    if value[:1] in _QUOTES and value.endswith(value[:1]) and len(value) > 1:
        value = value[1:-1]
    return name, value
