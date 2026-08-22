"""Release notes, for an engine that is installed rather than checked out.

vLLM is not a checkout on this machine — it arrives as ready-built packages —
so there are no commits to read locally. What there is instead is better: notes
written by the people who made the release, for somebody about to install it.

Those notes are not sifted the way commits are. Sifting exists because 138
commit subjects are unreadable; a release note is already the short version,
written by a person, and picking it apart by guessing at prefixes would damage
it. What arrives here is shown as it was written.

One request per press, to a service that allows sixty an hour without any
account. That is plenty for a button, and it is why nothing here runs on a
timer.

Not every project writes notes worth reading, and this is only used where they
are. llama.cpp tags every build, and its releases carry a table of downloadable
binaries — five kilobytes of HTML per tag, no prose — so it is configured
without a `releases` name and read as commits instead.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

TIMEOUT_S = 20
API = "https://api.github.com/repos/{repo}/releases?per_page={count}"

# How many releases back to look. Reading notes means reading everything
# between what is installed and what would be installed, and a machine several
# months behind can be a dozen releases back.
HOW_MANY = 30

# 0.27.1rc2.dev949+geac636a7f -> (0, 27, 1). Everything after the numbers is
# dropped on purpose: a nightly is "somewhere after this version", which is
# exactly how it should be compared.
NUMBERS = re.compile(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?")


def between(repo: str, installed: str, latest: str) -> tuple[str, str]:
    """The notes for every release after `installed`, up to and including `latest`.

    Returns the notes and a warning, either of which may be empty. A warning is
    not an error: notes that could not be fetched leave the rest of the screen
    — what packages change, what goes backwards — perfectly usable, and refusing
    to show any of it because a web service was unreachable would be worse.
    """
    try:
        releases = _releases(repo)
    except Exception as error:
        return "", f"Could not read the release notes: {error}"

    # Two tags are only comparable when they are spelled the same way.
    # `b10448` and `v0.2.0` are both llama.cpp, both real, and on different
    # scales — the first is a count of builds, the second a version. Reading
    # one as the other would silently pick the wrong releases, so when the
    # shapes differ the floor is dropped and only the target is described.
    crossing = _shape(installed) != _shape(latest)
    floor = None if crossing else _version(installed)
    ceiling = _version(latest)

    wanted = []
    for release in releases:
        here = _version(release.get("tag_name") or "")
        if here is None or (floor is not None and here <= floor):
            continue
        if ceiling is not None and here > ceiling:
            continue
        if crossing and (release.get("tag_name") or "") != latest:
            continue
        wanted.append(release)

    crossed = ""
    if crossing and wanted:
        crossed = (f"{installed} and {latest} are on different lines, so what "
                   f"follows describes {latest} itself rather than everything "
                   "between the two.")

    if not wanted:
        return "", ("No release notes cover this update — upstream publishes "
                    "none for the version installed here.")

    parts = []
    for release in wanted:                    # newest first, as they arrive
        title = release.get("name") or release.get("tag_name") or ""
        body = (release.get("body") or "").strip()
        parts.append(f"# {title}\n\n{body}" if body
                     else f"# {title}\n\n_No notes were written for this release._")
    return "\n\n---\n\n".join(parts), crossed


def _releases(repo: str) -> list[dict]:
    request = urllib.request.Request(
        API.format(repo=repo, count=HOW_MANY),
        headers={"Accept": "application/vnd.github+json",
                 # Asked for by name so that a refusal names this application
                 # rather than looking like an anonymous script.
                 "User-Agent": "ai-lab"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as answer:
            found = json.loads(answer.read())
    except urllib.error.HTTPError as error:
        if error.code in (403, 429):
            raise ValueError("too many requests to GitHub for now — "
                             "it allows sixty an hour") from None
        raise ValueError(f"GitHub answered {error.code}") from None
    except urllib.error.URLError as error:
        raise ValueError(f"could not reach GitHub ({error.reason})") from None
    return found if isinstance(found, list) else []


def _shape(text: str) -> str:
    """How a tag is spelled, so two of them are only compared when they match.

    "v0.2.0" is a version; "b10448" is a count of builds; "0.27.1rc1.dev949" is
    a package version. Only the first two are ever seen together, and only
    because llama.cpp publishes both.
    """
    text = (text or "").strip()
    if re.match(r"^v?\d+\.\d+", text):
        return "version"
    if re.match(r"^b\d+$", text):
        return "builds"
    return "other"


def _version(text: str) -> tuple[int, int, int] | None:
    match = NUMBERS.search(text or "")
    if not match:
        return None
    return tuple(int(part or 0) for part in match.groups())      # type: ignore
