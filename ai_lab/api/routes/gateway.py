"""The front door: one address for every configured model.

A client names the model it wants; if that model is not loaded it is loaded
first, and the client only notices that the first request took longer.

Requests arrive in one of two shapes. Nearly everything speaks the OpenAI one.
A client written against Anthropic's own library speaks the other, and only
some engines answer it — vLLM does, llama.cpp does not. Both are accepted here
and an entry that cannot answer a shape is refused by name, because the
alternative is forwarding the request to an engine that replies 404 about a
path the client never chose.

This file decides nothing about models. It reads the name out of the body, asks
`Gateway` for a lease, and forwards the request to whichever port that lease
points at. A name that belongs to a subscription model (Claude Code, Codex)
goes to `ProviderPool` instead, and a processor model (embeddings) to
`CpuLane`: those use no card, so they never take a lease.
"""

from __future__ import annotations

from ...cpu_models import EMBEDDINGS_PATH
from ...engines.base import (
    ALIGNMENT_PATHS,
    ANTHROPIC_PATHS,
    TOKENIZE_PATHS,
    DIARIZATION_PATHS,
    MUSIC_PATHS,
    OCR_PATHS,
    OPENAI_PATHS,
    SPEECH_PATHS,
    TRANSCRIPTION_PATHS,
    VAD_PATHS,
)
from ...gateway.core import Gateway
from ...lanes import SideLanes
from ...network import LOOPBACK
from ...gateway.state import Lease
from ...reservations import CardReservedError, ReservationBook, ReservationConflictError
from ...gateway.scheduler_parts import AdmissionWithdrawnError, StillWanted
from ...types import Task
from ..multipart import MultipartBody
from ..answer_cap import Hooks, PlaceKeeper, Waits, forward, send_capped
from ..uploads import UploadRejectedError, validate_image

# Every shape any engine here can answer. Registered as routes whatever is
# configured: a path that exists and explains why this model cannot serve it is
# more use than one that does not exist at all.
FORWARDED = tuple(dict.fromkeys(OPENAI_PATHS + ANTHROPIC_PATHS
                                + TRANSCRIPTION_PATHS + VAD_PATHS
                                + DIARIZATION_PATHS + OCR_PATHS + MUSIC_PATHS
                                + SPEECH_PATHS + ALIGNMENT_PATHS
                                + TOKENIZE_PATHS + (EMBEDDINGS_PATH,)))
NO_LANES = SideLanes()
# The header a call under a reservation carries: the reservation's id.
RESERVATION_HEADER = "X-AI-Lab-Reservation"
# How many times a call without a reservation, overtaken in the card's queue by
# a reservation granted meanwhile, goes back to wait at the door (ADR 0005).
DOOR_RETURNS = 3

# Which task a request path belongs to, for per-task timeouts (see
# `Gateway.timeouts_for`) and for which uploads get image validation. Paths
# that answer more than one task (`/v1/chat/completions` also transcribes)
# are not ambiguous in practice: this map is only consulted for the paths
# that are unique to a task.
_TASK_OF_PATH = {
    **dict.fromkeys(OPENAI_PATHS + ANTHROPIC_PATHS + TOKENIZE_PATHS, Task.TEXT_GENERATION),
    "/v1/audio/transcriptions": Task.TRANSCRIPTION,
    **dict.fromkeys(VAD_PATHS, Task.VAD),
    **dict.fromkeys(DIARIZATION_PATHS, Task.DIARIZATION),
    **dict.fromkeys(OCR_PATHS, Task.OCR),
    **dict.fromkeys(MUSIC_PATHS, Task.MUSIC_GENERATION),
    **dict.fromkeys(SPEECH_PATHS, Task.SPEECH_SYNTHESIS),
    **dict.fromkeys(ALIGNMENT_PATHS, Task.ALIGNMENT),
}

# Paths whose upload is an image and must pass the configured byte/pixel/
# dimension limits with a content-sniffed MIME type before it is forwarded.
_IMAGE_UPLOAD_PATHS = frozenset(OCR_PATHS)


def register(router, operations, gateway: Gateway, lanes: SideLanes = NO_LANES,
             book: ReservationBook | None = None) -> None:
    """The model routes, the catalogue, and the front door's own settings."""
    router.add("GET", "/v1/models", lambda **_: _catalogue(gateway, lanes))
    router.add("GET", "/v1/models/{model}",
               lambda model, **_: _describe(gateway, lanes, model))
    router.add("GET", "/api/gateway", lambda **_: _stats(gateway, lanes, book))

    def settings(body=None, **_):
        """Change the front door's own limits, and use them at once."""
        saved = operations.update_gateway(body or {})
        gateway.apply_settings(saved)
        return _stats(gateway, lanes, book)

    router.add("PATCH", "/api/gateway", settings)
    for path in FORWARDED:
        router.add("POST", path, _forwarder(gateway, path, lanes, book))
    if lanes.cpu is not None:
        cpu = lanes.cpu
        router.add("POST", "/api/cpu-models/{name}/start",
                   lambda name, **_: cpu.start(name))
        router.add("POST", "/api/cpu-models/{name}/stop",
                   lambda name, **_: cpu.stop(name))


def _stats(gateway: Gateway, lanes: SideLanes, book: ReservationBook | None = None) -> dict:
    """The card's queue, each subscription vendor's, and each processor model's."""
    stats = gateway.stats()
    if book is not None:
        # Calls without a reservation held at the door while the card is
        # reserved: not yet in the card's queue, so `waiting` does not count them.
        stats["at_the_door"] = book.door.view()
    if lanes.providers is not None:
        stats["providers"] = lanes.providers.stats()
    if lanes.cpu is not None:
        stats["cpu_models"] = lanes.cpu.stats()
    return stats


def _catalogue(gateway: Gateway, lanes: SideLanes) -> dict:
    """Every configured model, in the shape an OpenAI client expects.

    Models that are not loaded are listed too. That is the point: a client is
    supposed to be able to ask for one of them. Subscription and processor
    models follow the card's.
    """
    rows = [_one_model(row) for row in gateway.catalogue()] + lanes.catalogue()
    return {"object": "list", "data": rows}


def _describe(gateway: Gateway, lanes: SideLanes, name: str) -> dict:
    """One model, on the card, by subscription or on the processor, with its details."""
    return lanes.describe(name) or _one_model(gateway.describe(name), detailed=True)


# Extra details only `GET /v1/models/{model}` carries: too long for a listing.
# `live` is the running engine's own report (the context pool for vLLM).
_DETAIL_FIELDS = ("model_id", "params", "speech_form", "music_form", "live")


def _one_model(row: dict, detailed: bool = False) -> dict:
    """One model in the OpenAI shape, with this project's details beside it."""
    extra = {"loaded": row["loaded"], "ready": row["ready"],
             "port": row["port"], "shapes": row["shapes"],
             "task": row["task"], "capabilities": row["capabilities"]}
    if detailed:
        extra.update({key: row[key] for key in _DETAIL_FIELDS if key in row})
    return {
        "id": row["id"],
        "object": "model",
        "owned_by": row["engine"],
        # Not part of the OpenAI shape, and harmless to a client that ignores
        # unknown fields. It says which models are up and what each can do.
        "ai_lab": extra,
    }


# Where a client puts settings the model has to be *started* with, rather than
# ones that go in a request. Anything else in the body belongs to the engine and
# is passed through untouched.
SETTINGS_FIELD = "ai_lab"


def _forwarder(gateway: Gateway, path: str, lanes: SideLanes = NO_LANES,
               book: ReservationBook | None = None):
    """The handler for one forwarded path."""
    def handle(body=None, alive=None, headers=None, **_):
        payload = body or {}
        wanted = _model_name(payload)
        if path in TOKENIZE_PATHS:
            return _count_tokens(gateway, path, payload, wanted)
        # The card's reservations come first (ADR 0005): a call reaches a model
        # on this machine — the card's or an embeddings model — only under the
        # holder's reservation, or when the card is not reserved. Subscription
        # models run in the cloud and are never held back.
        guarded = book if book is not None and not _in_the_cloud(lanes, wanted) else None
        admission = _Admission(guarded, (headers or {}).get(RESERVATION_HEADER), alive,
                               wanted)
        admission.at_the_door()
        side = _side_lane(lanes, path, payload, wanted)
        if side is not None:
            return side
        _check_upload(gateway, path, payload)
        lease = _take_place(gateway, path, payload, admission)
        # Everything from here gives the place back if it fails: a leaked place
        # breaks nothing visibly, the card simply has one fewer place for ever.
        try:
            return _forward(gateway, lease, path, payload, admission)
        except BaseException:
            lease.release()
            raise
    return handle


def _count_tokens(gateway: Gateway, path: str, payload, wanted: str):
    """Count a text's tokens with a loaded model's tokenizer: no card place, no queue, no load.

    The engine counts on the processor, not the card, so a count neither waits
    for places nor for a reservation, and takes none from anybody. A model that
    is not loaded cannot count, and is not loaded for it: the answer is a 409
    with `ai_lab.reason` `model_not_loaded` (owner's decision, 22 October 2026).
    """
    if isinstance(payload, MultipartBody):
        raise ValueError("counting tokens takes a JSON request")
    port, name = gateway.where_loaded(wanted, path)
    request = {key: value for key, value in payload.items() if key != SETTINGS_FIELD}
    request["model"] = name
    first_byte_s, between_bytes_s = gateway.timeouts_for(Task.TEXT_GENERATION)
    return forward(f"http://{LOOPBACK}:{port}{path}", request,
                   Waits(first_byte_s, between_bytes_s), Hooks())


class _Admission:
    """Whether one call may reach the card: asked at the door, and again at the head of the queue.

    The door (`ReservationBook.admit`) lets the call in or makes it wait while
    the card is reserved. The call may then wait in the card's own queue, and
    by the time it is served the card may have been reserved by another team,
    or the call's own reservation may have ended. `still_wanted` is what the
    scheduler asks at the head of its queue, before anything is unloaded: it
    says whether the client is still there and, through the book, withdraws a
    call that may no longer reach the card (`AdmissionWithdrawnError`).
    """

    def __init__(self, book: ReservationBook | None, reservation_id: str | None,
                 alive: StillWanted | None, model: str = "") -> None:
        self.book = book
        self.reservation_id = reservation_id
        self.alive = alive
        self.model = model

    def at_the_door(self) -> None:
        """Let the call in, wait while the card is held, or refuse it (ADR 0005)."""
        if self.book is not None:
            self.book.admit(self.reservation_id, self.alive, self.model)

    def still_wanted(self) -> bool:
        """True while the client is there and the card is still its to use."""
        if self.alive is not None and not self.alive():
            return False
        if self.book is not None:
            try:
                self.book.permits(self.reservation_id)
            except ReservationConflictError as refused:
                raise AdmissionWithdrawnError(refused) from refused
        return True

    def may_wait_again(self, reason: BaseException) -> bool:
        """A call without a reservation, overtaken by one: it waits at the door again."""
        return self.reservation_id is None and isinstance(reason, CardReservedError)


def _take_place(gateway: Gateway, path: str, payload, admission: _Admission) -> Lease:
    """A place on the card for this call, with the model it names on it.

    A call without a reservation that queued while the card was free, and was
    overtaken by a reservation granted meanwhile, goes back to the door and
    waits there as if it had just arrived — at most `DOOR_RETURNS` times, then
    it gets the 409. A call whose own reservation ended while it queued gets
    the 409 at once, as it would have at the door.
    """
    returns = 0
    while True:
        try:
            return gateway.acquire(_model_name(payload), shape=path,
                                   settings=_start_settings(payload),
                                   still_wanted=admission.still_wanted)
        except AdmissionWithdrawnError as withdrawn:
            if returns >= DOOR_RETURNS or not admission.may_wait_again(withdrawn.reason):
                raise withdrawn.reason from None
            returns += 1
            admission.at_the_door()


def _in_the_cloud(lanes: SideLanes, wanted: str) -> bool:
    """Whether a subscription vendor answers this name (no reservation applies)."""
    return lanes.providers is not None and lanes.providers.serves(wanted)


def _side_lane(lanes: SideLanes, path: str, payload, wanted: str) -> dict | None:
    """The answer from a lane beside the card, or None when the card serves this name."""
    for lane, method, what in ((lanes.providers, "complete", "subscription"),
                               (lanes.cpu, "embed", "processor")):
        if lane is not None and lane.serves(wanted):
            if isinstance(payload, MultipartBody):
                raise ValueError(f"{what} models take a JSON request")
            return getattr(lane, method)(path, payload)
    return None


def _model_name(payload) -> str:
    wanted = (payload.field("model") if isinstance(payload, MultipartBody)
              else payload.get("model"))
    if not wanted:
        raise ValueError("the request must name a model")
    return wanted


def _check_upload(gateway: Gateway, path: str, payload) -> None:
    """An uploaded image must pass the configured size limits first."""
    if path not in _IMAGE_UPLOAD_PATHS or not isinstance(payload, MultipartBody):
        return
    uploaded = payload.raw("file")
    if uploaded is None:
        raise ValueError("the request must contain an image file")
    try:
        validate_image(uploaded, max_bytes=gateway.max_upload_bytes,
                       max_pixels=gateway.max_upload_pixels,
                       max_dimension=gateway.max_upload_dimension)
    except UploadRejectedError as error:
        raise ValueError(str(error)) from None


def _start_settings(payload) -> dict | None:
    """Settings the model must be started with, carried in our own field.

    The engine does not know them and would ignore them without a word, so
    they are read here and removed before forwarding.
    """
    settings = (None if isinstance(payload, MultipartBody)
                else payload.get(SETTINGS_FIELD) or None)
    if settings is not None and not isinstance(settings, dict):
        raise ValueError(f"{SETTINGS_FIELD} must be an object of settings")
    return settings


def _outgoing(lease, payload, wanted: str) -> tuple[bytes | dict, str, bool]:
    """The body for the engine, its content type, and whether streaming was asked for."""
    # The engine knows its own model by a different name than the entry does,
    # and rejects a name it does not recognise. The lease carries it.
    name = lease.model_name or wanted
    if isinstance(payload, MultipartBody):
        return payload.replace("model", name), payload.content_type, False
    # Fields the engine refuses with its settings are left out rather than
    # turned into an error every caller would have to learn to avoid.
    request = {key: value for key, value in payload.items()
               if key != SETTINGS_FIELD and key not in lease.drop}
    request["model"] = name
    return request, "application/json", bool(payload.get("stream"))


def _forward(gateway: Gateway, lease, path: str, payload, admission: _Admission):
    """Send the request to the leased engine, by the name that engine knows."""
    outgoing, content_type, streaming = _outgoing(lease, payload, admission.model)
    # Time to the first token only when streaming was asked for: otherwise the
    # first byte is the whole generation and the average measures neither.
    timed = ((lambda seconds: gateway.first_token(seconds, lease.instance_id))
             if streaming else None)
    keeper = PlaceKeeper(lease.release)
    # A stalled engine is restarted, once, so the next requests do not stall too.
    hooks = Hooks(on_close=keeper, on_first_chunk=timed,
                  on_stall=lambda reason: gateway.engine_stalled(lease.instance_id, reason))
    waits = _waits(gateway, lease, path, admission.alive)

    # A text too long for the model comes back with the limit and its size as
    # numbers next to the engine's sentence (`context_limit`).
    def send(body):
        return forward(f"http://{LOOPBACK}:{lease.port}{path}", body, waits, hooks,
                       content_type=content_type)
    if not isinstance(outgoing, dict) or not lease.text:
        return send(outgoing)
    return send_capped(send, outgoing, path, gateway.patience.default_max_tokens, keeper)


def _waits(gateway: Gateway, lease, path: str, alive) -> Waits:
    """How long to wait, and what to keep looking at: the client, and the engine's progress.

    An engine that keeps work counters (vLLM) is judged by them: no limit on
    waiting, a cut only when it does no work at all (`gateway.patience`). Any
    other keeps the two plain waits of its kind of request.
    """
    first_byte_s, between_bytes_s = gateway.timeouts_for(
        _TASK_OF_PATH.get(path, Task.TEXT_GENERATION))
    patience = gateway.patience
    progress = lease.progress
    return Waits(first_byte_s, between_bytes_s, client_here=alive,
                 idle_for=progress.idle_for if progress is not None else None,
                 stall_s=patience.stall_s, safety_s=patience.safety_s)
