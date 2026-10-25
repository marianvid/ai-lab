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
from ...gateway.admission.parts import StillWanted
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


def register(router, operations, gateway: Gateway, lanes: SideLanes = NO_LANES) -> None:
    """The model routes, the catalogue, and the front door's own settings."""
    router.add("GET", "/v1/models", lambda **_: _catalogue(gateway, lanes))
    router.add("GET", "/v1/models/{model}",
               lambda model, **_: _describe(gateway, lanes, model))
    router.add("GET", "/api/gateway", lambda **_: _stats(gateway, lanes))

    def settings(body=None, **_):
        """Change the front door's own limits, and use them at once."""
        saved = operations.update_gateway(body or {})
        gateway.apply_settings(saved)
        return _stats(gateway, lanes)

    router.add("PATCH", "/api/gateway", settings)
    for path in FORWARDED:
        router.add("POST", path, _forwarder(gateway, path, lanes))
    if lanes.cpu is not None:
        cpu = lanes.cpu
        router.add("POST", "/api/cpu-models/{name}/start",
                   lambda name, **_: cpu.start(name))
        router.add("POST", "/api/cpu-models/{name}/stop",
                   lambda name, **_: cpu.stop(name))


def _stats(gateway: Gateway, lanes: SideLanes) -> dict:
    """The card's queue, each subscription vendor's, and each processor model's."""
    stats = gateway.stats()
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


def _forwarder(gateway: Gateway, path: str, lanes: SideLanes = NO_LANES):
    """The handler for one forwarded path.

    Calls are served first come, first served (the approved list, A3): who may
    use the card when is the schedulers' business, not AI-Lab's. A reservation
    header an older client still sends is ignored.
    """
    def handle(body=None, alive=None, **_):
        payload = body or {}
        wanted = _model_name(payload)
        if path in TOKENIZE_PATHS:
            return _count_tokens(gateway, path, payload, wanted)
        side = _side_lane(lanes, path, payload, wanted)
        if side is not None:
            return side
        _check_upload(gateway, path, payload)
        lease = gateway.acquire(wanted, shape=path, settings=_start_settings(payload),
                                still_wanted=alive)
        # Everything from here gives the place back if it fails: a leaked place
        # breaks nothing visibly, the card simply has one fewer place for ever.
        try:
            return _forward(gateway, lease, path, payload, _Call(wanted, alive))
        except BaseException:
            lease.release()
            raise
    return handle


def _count_tokens(gateway: Gateway, path: str, payload, wanted: str):
    """Count a text's tokens with a loaded model's tokenizer: no card place, no queue, no load.

    The engine counts on the processor, not the card, so a count never waits
    for a place and takes none from anybody. A model that is not loaded cannot
    count, and is not loaded for it: the answer is a 409 with `ai_lab.reason`
    `model_not_loaded` (owner's decision, 22 October 2026).
    """
    if isinstance(payload, MultipartBody):
        raise ValueError("counting tokens takes a JSON request")
    port, name = gateway.where_loaded(wanted, path)
    request = {key: value for key, value in payload.items() if key != SETTINGS_FIELD}
    request["model"] = name
    first_byte_s, between_bytes_s = gateway.timeouts_for(Task.TEXT_GENERATION)
    return forward(f"http://{LOOPBACK}:{port}{path}", request,
                   Waits(first_byte_s, between_bytes_s), Hooks())


class _Call:
    """The model a call named, and how to ask whether its client is still there."""

    def __init__(self, model: str, alive: StillWanted | None) -> None:
        self.model = model
        self.alive = alive


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


def _forward(gateway: Gateway, lease, path: str, payload, call: _Call):
    """Send the request to the leased engine, by the name that engine knows."""
    outgoing, content_type, streaming = _outgoing(lease, payload, call.model)
    # Time to the first token only when streaming was asked for: otherwise the
    # first byte is the whole generation and the average measures neither.
    timed = ((lambda seconds: gateway.first_token(seconds, lease.instance_id))
             if streaming else None)
    keeper = PlaceKeeper(lease.release)
    # A stalled engine is restarted, once, so the next requests do not stall too.
    hooks = Hooks(on_close=keeper, on_first_chunk=timed,
                  on_stall=lambda reason: gateway.engine_stalled(lease.instance_id, reason))
    waits = _waits(gateway, lease, path, call.alive)

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
