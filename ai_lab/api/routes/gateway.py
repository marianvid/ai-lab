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
goes to `ProviderPool` instead: those use no card, so they never take a lease.
"""

from __future__ import annotations

from ...engines.base import (
    ALIGNMENT_PATHS,
    ANTHROPIC_PATHS,
    DIARIZATION_PATHS,
    MUSIC_PATHS,
    OCR_PATHS,
    OPENAI_PATHS,
    SPEECH_PATHS,
    TRANSCRIPTION_PATHS,
    VAD_PATHS,
)
from ...gateway import Gateway
from ...network import LOOPBACK
from ...providers import ProviderPool
from ...types import Task
from ..multipart import MultipartBody
from ..passthrough import forward
from ..uploads import UploadRejected, validate_image

# Every shape any engine here can answer. Registered as routes whatever is
# configured: a path that exists and explains why this model cannot serve it is
# more use than one that does not exist at all.
FORWARDED = tuple(dict.fromkeys(OPENAI_PATHS + ANTHROPIC_PATHS
                                + TRANSCRIPTION_PATHS + VAD_PATHS
                                + DIARIZATION_PATHS + OCR_PATHS + MUSIC_PATHS
                                + SPEECH_PATHS + ALIGNMENT_PATHS))

# Which task a request path belongs to, for per-task timeouts (see
# `Gateway.timeouts_for`) and for which uploads get image validation. Paths
# that answer more than one task (`/v1/chat/completions` also transcribes)
# are not ambiguous in practice: this map is only consulted for the paths
# that are unique to a task.
_TASK_OF_PATH = {
    **dict.fromkeys(OPENAI_PATHS + ANTHROPIC_PATHS, Task.TEXT_GENERATION),
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


def register(router, operations, gateway: Gateway,
             providers: ProviderPool | None = None) -> None:
    """The model routes, the catalogue, and the front door's own settings."""
    router.add("GET", "/v1/models", lambda **_: _catalogue(gateway, providers))
    router.add("GET", "/v1/models/{model}",
               lambda model, **_: _describe(gateway, providers, model))
    router.add("GET", "/api/gateway", lambda **_: _stats(gateway, providers))

    def settings(body=None, **_):
        """Change the front door's own limits, and use them at once."""
        saved = operations.update_gateway(body or {})
        gateway.apply_settings(saved)
        return _stats(gateway, providers)

    router.add("PATCH", "/api/gateway", settings)
    for path in FORWARDED:
        router.add("POST", path, _forwarder(gateway, path, providers))


def _stats(gateway: Gateway, providers: ProviderPool | None) -> dict:
    """The card's queue, and each subscription vendor's places and usage."""
    stats = gateway.stats()
    if providers is not None:
        stats["providers"] = providers.stats()
    return stats


def _catalogue(gateway: Gateway, providers: ProviderPool | None) -> dict:
    """Every configured model, in the shape an OpenAI client expects.

    Models that are not loaded are listed too. That is the point: a client is
    supposed to be able to ask for one of them. Subscription models follow
    the local ones.
    """
    rows = [_one_model(row) for row in gateway.catalogue()]
    if providers is not None:
        rows += providers.catalogue()
    return {"object": "list", "data": rows}


def _describe(gateway: Gateway, providers: ProviderPool | None, name: str) -> dict:
    """One model, local or subscription, with its details."""
    if providers is not None and providers.serves(name):
        return next(row for row in providers.catalogue() if row["id"] == name)
    return _one_model(gateway.describe(name), detailed=True)


# Extra details only `GET /v1/models/{model}` carries: too long for a listing.
_DETAIL_FIELDS = ("model_id", "params", "speech_form", "music_form")


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


def _forwarder(gateway: Gateway, path: str, providers: ProviderPool | None = None):
    """The handler for one forwarded path."""
    def handle(body=None, alive=None, **_):
        payload = body or {}
        wanted = _model_name(payload)
        if providers is not None and providers.serves(wanted):
            if isinstance(payload, MultipartBody):
                raise ValueError("subscription models take a JSON chat request")
            return providers.complete(path, payload)
        _check_upload(gateway, path, payload)
        # The lease is held until the last byte of the answer has been read, so
        # a swap cannot pull the model out from under a stream in progress.
        # `path` goes with it: an entry whose engine does not answer this shape
        # is refused before anything is loaded, not after.
        lease = gateway.acquire(wanted, shape=path, settings=_start_settings(payload),
                                still_wanted=alive)
        # Everything from here gives the place back if it fails: a leaked place
        # breaks nothing visibly, the card simply has one fewer place for ever.
        try:
            return _forward(gateway, lease, path, payload, wanted)
        except BaseException:
            lease.release()
            raise
    return handle


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
    except UploadRejected as error:
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


def _forward(gateway: Gateway, lease, path: str, payload, wanted: str):
    """Send the request to the leased engine, by the name that engine knows."""
    # The engine knows its own model by a different name than the entry does,
    # and rejects a name it does not recognise. The lease carries it.
    name = lease.model_name or wanted
    if isinstance(payload, MultipartBody):
        outgoing: bytes | dict = payload.replace("model", name)
        content_type = payload.content_type
        streaming = False
    else:
        request = {key: value for key, value in payload.items() if key != SETTINGS_FIELD}
        request["model"] = name
        outgoing = request
        content_type = "application/json"
        streaming = bool(payload.get("stream"))
    # Time to the first token only when streaming was asked for: otherwise the
    # first byte is the whole generation and the average measures neither.
    timed = ((lambda seconds: gateway.first_token(seconds, lease.instance_id))
             if streaming else None)
    first_byte_s, between_bytes_s = gateway.timeouts_for(
        _TASK_OF_PATH.get(path, Task.TEXT_GENERATION))
    return forward(f"http://{LOOPBACK}:{lease.port}{path}", outgoing,
                   on_close=lease.release, first_byte_s=first_byte_s,
                   between_bytes_s=between_bytes_s, on_first_chunk=timed,
                   content_type=content_type)
