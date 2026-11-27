"""Why a song's move was refused; each names the HTTP status the server answers with."""

from __future__ import annotations

from http import HTTPStatus


class SongAlreadyHereError(RuntimeError):
    """A pull or a put-back would land on a song this machine already has: a 409.

    Said before anything is fetched, and again if another pull or put-back
    got there while this one was copying; the copy is removed, the song kept.
    """

    http_status = HTTPStatus.CONFLICT


class PeerConflictError(RuntimeError):
    """The other AI-Lab refused the bundle with its own 409 (a window call or a job on it): a 409."""

    http_status = HTTPStatus.CONFLICT


class TransferCheckError(RuntimeError):
    """A song fetched from another AI-Lab did not arrive as its manifest says: a 502.

    The copy is removed and the source kept; the other AI-Lab's answer was wrong, not the request.
    """

    http_status = HTTPStatus.BAD_GATEWAY


class SchedulerDoorError(RuntimeError):
    """This machine's scheduler door refused a pull's call, or did not answer: a 502.

    Told apart from the other AI-Lab's own answers by the door's `scheduler` key; the
    message carries the door's reason and detail (e.g. `run_stopped`, `partner_lost`).
    """

    http_status = HTTPStatus.BAD_GATEWAY
