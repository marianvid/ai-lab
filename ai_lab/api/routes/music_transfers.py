"""Moving a YuE2 song (a generation) from one AI-Lab to another.

The machine that should have the song pulls it; the copy is kept only once it is
known to be whole:

- `GET /v1/audio/music/generations/{id}/bundle?model=<m>` answers the song's folder as an
  uncompressed tar. Its first member is `MANIFEST.json`: `{"generation_id", "files":
  [{"name", "size", "sha256"}]}`; the files follow, flat, in the manifest's order.
- `POST /v1/audio/music/generations/{id}/pull?model=<m>` with `{"from": "<peer>"}` and the
  header `X-Run-Id` fetches that bundle from the peer named in `engines.yue2.transfer`,
  writes it into a hidden folder beside the songs, checks every size and sha256 against the
  manifest, and renames it into place. It never deletes the peer's copy: the caller
  (music) does that itself once the answer says `checked: true`.
- `GET /v1/audio/music/generations?model=<m>[&limit=<n>][&offset=<k>]` lists this machine's
  generations, newest first, with what recognises each one. A read: nothing is changed.
- `POST /v1/audio/music/generations/{id}/put-back?model=<m>` with a bundle as the body
  (`Content-Type: application/x-tar`, the same tar `bundle` answers) and the header
  `X-Run-Id` puts a kept song back: every file is checked against the manifest as a pull
  checks it, written into a hidden folder, and renamed into place under the same id.
  Refused, never overwritten, when that id is already here or busy.

The AI-Labs never talk directly: the fetch goes through this machine's scheduler door,
`{scheduler}/ai/<peer>/v1/audio/music/generations/{id}/bundle`, carrying the caller's
`X-Run-Id` (the run the door checks) and no token. A pull without `X-Run-Id` is refused
(400). The door's own refusals (its `scheduler` key) are told apart from the other AI-Lab's
answers: a door's 404 is never taken for "the peer has no such song".

Both read and write the engine's folder, so they work whether YuE2 is loaded or not.

The routes only read the request (its query, body and `X-Run-Id`) and hand it to the
song service (`songs/`), which does the work and says every refusal.
"""
from __future__ import annotations

from http import HTTPStatus

from ...songs.bundles import BUNDLE_TYPE
from ...songs.listing import generations
from ...songs.transfers import RUN_HEADER, Pull, PutBack, Songs, bundle_pieces
from ..passthrough import Passthrough
from ..raw_body import RawBody


def bundle(files: Songs, query: dict | None, generation_id: str) -> Passthrough:
    """The generation as a bundle, sent as it is made."""
    return Passthrough(status=HTTPStatus.OK, content_type=BUNDLE_TYPE,
                       chunks=bundle_pieces(files, query, generation_id),
                       headers={"Content-Disposition": f'attachment; filename="{generation_id}.tar"'})


def run_id_of(headers) -> str | None:
    """The caller's `X-Run-Id`, if it sent one."""
    value = headers.get(RUN_HEADER) if headers is not None else None
    if not isinstance(value, str):
        return None
    return value.strip() or None


def put_back(files: Songs, query: dict | None, generation_id: str, body, headers) -> dict:
    """A kept bundle put back: only a raw bundle body is one."""
    bundle_body = body if isinstance(body, RawBody) else None
    return PutBack(files, query, generation_id, bundle_body, run_id_of(headers)).run()


def register_transfers(router, files: Songs) -> None:
    """The bundle, pull, put-back and list routes."""
    base = "/v1/audio/music/generations/{generation_id}"
    router.add("GET", base + "/bundle",
               lambda generation_id, query=None, **_: bundle(files, query, generation_id))
    router.add("POST", base + "/pull",
               lambda generation_id, query=None, body=None, headers=None, **_:
               Pull(files, query, generation_id, body, run_id_of(headers)).run())
    router.add("POST", base + "/put-back",
               lambda generation_id, query=None, body=None, headers=None, **_:
               put_back(files, query, generation_id, body, headers))
    router.add("GET", "/v1/audio/music/generations",
               lambda query=None, **_: generations(files, query))
