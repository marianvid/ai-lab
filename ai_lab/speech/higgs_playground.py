"""The upstream Higgs playground page, served straight from the loaded model.

On Linux, Higgs runs inside an SGLang-Omni worker, and the playground that
ships with SGLang-Omni (a small web page plus a FastAPI program) forwards
the page's requests to that worker. On a Mac there is no worker: engine
`higgs_local` holds the model inside the speech host's own process. So this
file serves the same page itself and answers the page's two requests by
calling the already-loaded model. The checkpoint is never loaded twice.

What is reused unchanged: the page's HTML, JavaScript and styles, from
`native_ui/sglang_omni/playground/higgs/frontend/`. What this file replaces:
only the upstream FastAPI program, because that program needs FastAPI,
uvicorn and SGLang-Omni itself, none of which the Mac runtime has. This one
uses nothing beyond the standard library.

The page asks for:
- `GET /healthz` — is the model answering;
- `POST /api/synthesize` — one WAV file for a text;
- `POST /api/synthesize/stream` — the same speech as raw 16-bit PCM samples
  (headerless audio) so the page can start playing early. The transformers
  port produces a line all at once, so here the "stream" arrives in one
  piece; the page plays it the same way.

Both POSTs carry a multipart form (a browser form upload: text fields and a
file in one request). A reference clip, used to copy a voice, may be any
format the runtime can decode; it is converted to WAV before the model sees
it. A reference given only as a URL is refused: fetching arbitrary addresses
from the machine that runs the model is not something a page should be able
to ask for.
"""
from __future__ import annotations

import base64
import io
import json
import mimetypes
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from ai_lab.multipart import MultipartBody

from .higgs_local_backend import sampling_settings

FRONTEND = (Path(__file__).resolve().parents[1] / "native_ui" / "sglang_omni"
            / "playground" / "higgs" / "frontend")
# A form carries the text and one reference clip. Compressed clips are small;
# a minute of uncompressed studio WAV is about 11 MB.
MAX_FORM_BYTES = 32 * 1024 * 1024
STATIC_FILES = ("app.js", "styles.css", "favicon.ico", "sgl-omni-logo.png")


def serve(backend, port: int) -> ThreadingHTTPServer:
    """Start the playground on `port` in a background thread and return it."""
    handler = type("PlaygroundHandler", (Handler,), {"backend": backend})
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    Thread(target=server.serve_forever, name="higgs-playground",
           daemon=True).start()
    return server


class Handler(BaseHTTPRequestHandler):
    backend = None

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, index_html(), "text/html; charset=utf-8",
                       {"Cache-Control": "no-cache, must-revalidate"})
        elif path == "/healthz":
            self._json(200, {"backend": "ok", "api_base": "in-process"})
        elif path == "/favicon.ico" or path.startswith("/static/"):
            name = path.rsplit("/", 1)[-1]
            if name not in STATIC_FILES:  # a fixed list: no path can escape
                self._json(404, {"detail": "Not Found"})
                return
            kind = mimetypes.guess_type(name)[0] or "application/octet-stream"
            self._send(200, (FRONTEND / name).read_bytes(), kind)
        else:
            self._json(404, {"detail": "Not Found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path not in ("/api/synthesize", "/api/synthesize/stream"):
            self._json(404, {"detail": "Not Found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_FORM_BYTES:
                raise ValueError("the form is empty or larger than 32 MB")
            form = MultipartBody(self.headers.get("Content-Type", ""),
                                 self.rfile.read(size))
            body, sampling = speech_request(form)
            result = self.backend.generate(body, sampling)
            audio = base64.b64decode(result["data"][0]["b64_wav"])
        except (ValueError, TypeError) as error:
            self._json(400, {"detail": str(error)})
            return
        except Exception as error:
            self._json(500, {"detail": str(error)})
            return
        if path == "/api/synthesize":
            self._send(200, audio, "audio/wav")
            return
        rate, channels, width, samples = pcm_from_wav(audio)
        self._send(200, samples, "audio/pcm", {
            "Cache-Control": "no-store", "X-Sample-Rate": str(rate),
            "X-Channels": str(channels), "X-Bit-Depth": str(width * 8)})

    def log_message(self, *_args):  # the engine log has no use for page hits
        pass

    def _json(self, status: int, body: dict):
        self._send(status, json.dumps(body).encode(), "application/json")

    def _send(self, status: int, payload: bytes, kind: str,
              headers: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)


def index_html() -> bytes:
    """The page, with each script and style address stamped with its file
    date, as upstream does, so a browser never keeps a stale copy."""
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    for asset in ("app.js", "styles.css"):
        stamp = int((FRONTEND / asset).stat().st_mtime)
        html = html.replace(f"/static/{asset}", f"/static/{asset}?v={stamp}")
    return html.encode("utf-8")


def speech_request(form: MultipartBody) -> tuple[dict, dict]:
    """Turn the page's form into a speech-contract request plus its knobs."""
    text = (form.field("text") or "").strip()
    if not text:
        raise ValueError("Input text is empty.")
    if (form.field("ref_audio_url") or "").strip():
        raise ValueError("A reference given by URL is not supported here; "
                         "upload or record the clip instead.")
    body: dict = {"text": text}
    seed = _number(form, "seed", int)
    if seed is not None:
        body["seed"] = seed
    clip = form.raw("ref_audio")
    if clip:
        body["reference_audio"] = base64.b64encode(as_wav(clip)).decode()
        reference_text = (form.field("ref_text") or "").strip()
        if reference_text:
            body["reference_text"] = reference_text
    sampling = sampling_settings(
        temperature=_number(form, "temperature", float),
        top_p=_number(form, "top_p", float),
        top_k=_number(form, "top_k", int),
        max_new_tokens=_number(form, "max_new_tokens", int))
    return body, sampling


def _number(form: MultipartBody, name: str, kind):
    value = (form.field(name) or "").strip()
    if not value:
        return None
    try:
        return kind(value)
    except ValueError:
        raise ValueError(f"{name} must be a number") from None


def as_wav(clip: bytes) -> bytes:
    """The reference clip as a mono 16-bit WAV, whatever it arrived as.

    WAV passes straight through. Anything else (MP3, FLAC, or the WebM a
    browser records from a microphone) is decoded with the libraries the
    runtime already has: soundfile first, then torchaudio.
    """
    if clip.startswith(b"RIFF"):
        return clip
    samples, rate = _decode(clip)
    out = io.BytesIO()
    import soundfile as sf

    sf.write(out, samples, rate, format="WAV", subtype="PCM_16")
    return out.getvalue()


def _decode(clip: bytes):
    try:
        import soundfile as sf

        samples, rate = sf.read(io.BytesIO(clip), dtype="float32", always_2d=True)
        return samples.mean(axis=1), rate
    except Exception:
        pass
    try:
        import torchaudio

        samples, rate = torchaudio.load(io.BytesIO(clip))
        return samples.mean(dim=0).numpy(), rate
    except Exception:
        raise ValueError("This runtime cannot read that reference clip; "
                         "upload it as WAV, FLAC or MP3.") from None


def pcm_from_wav(audio: bytes) -> tuple[int, int, int, bytes]:
    """Sample rate, channel count, bytes per sample, and the bare samples."""
    with wave.open(io.BytesIO(audio)) as reader:
        return (reader.getframerate(), reader.getnchannels(),
                reader.getsampwidth(), reader.readframes(reader.getnframes()))
