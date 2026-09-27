"""Start the mlx-vlm server (text and pictures on Apple silicon), adapted.

This file runs inside the mlx-vlm environment, not inside the manager. It
imports only mlx-vlm and the web framework mlx-vlm already uses (Starlette,
under FastAPI). Every command-line option mlx-vlm knows is passed straight
through; the options below are this launcher's own and are removed first.

mlx-vlm serves one process per model here, like every AI-Lab engine. Its
server was written for a different setup — one process that loads whatever a
request names — so each request is adjusted before mlx-vlm sees it:

1. **The model.** mlx-vlm reads the request's `model` field as a folder or a
   Hugging Face name to load. The gateway puts the model's short name there,
   which is neither, so mlx-vlm would try to download it and throw out the
   loaded model. The field is replaced with the folder given by `--model`.

2. **Thinking.** mlx-vlm switches a model's thinking with a top-level
   `enable_thinking` field. llama.cpp and mlx-lm use
   `"chat_template_kwargs": {"enable_thinking": false}`. Both spellings work
   here: the second is copied into the first when the first is absent, so a
   client can use one field for all three engines.

3. **Defaults.** mlx-vlm has no command-line defaults for temperature,
   top-p, top-k or min-p. `--request-default key=value` (repeatable; the value
   is JSON) fills a field only when a request leaves it out, which is exactly
   what a server-side default means.

Readiness needs no change: mlx-vlm loads the model given with `--model`
before it starts answering, and its `/health` names the loaded model.
"""

from __future__ import annotations

import json
import sys


def split_arguments(argv: list[str]) -> tuple[dict, list[str]]:
    """Take this launcher's own options out; return them and the rest."""
    defaults: dict = {}
    rest: list[str] = []
    items = iter(argv)
    for item in items:
        if item == "--request-default":
            key, _, value = next(items).partition("=")
            defaults[key] = json.loads(value)
        else:
            rest.append(item)
    return defaults, rest


def model_of(argv: list[str]) -> str:
    """The folder given with `--model`, which every request is pointed at."""
    for index, item in enumerate(argv):
        if item == "--model" and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith("--model="):
            return item.split("=", 1)[1]
    raise SystemExit("--model is required")


def adjust(body: dict, model: str, defaults: dict) -> dict:
    """One request, as mlx-vlm needs it. See the module text for why."""
    body = dict(body)
    body["model"] = model
    template = body.get("chat_template_kwargs")
    if (isinstance(template, dict) and "enable_thinking" in template
            and "enable_thinking" not in body):
        body["enable_thinking"] = bool(template["enable_thinking"])
    for key, value in defaults.items():
        body.setdefault(key, value)
    return body


class AdjustRequests:
    """Rewrites each JSON request body before mlx-vlm reads it."""

    def __init__(self, app, model: str, defaults: dict) -> None:
        self.app = app
        self.model = model
        self.defaults = defaults

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers") or [])
        if b"json" not in headers.get(b"content-type", b""):
            return await self.app(scope, receive, send)
        chunks = []
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return await self.app(scope, receive, send)
            chunks.append(message.get("body", b""))
            if not message.get("more_body"):
                break
        raw = b"".join(chunks)
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            raw = json.dumps(adjust(parsed, self.model, self.defaults)).encode()
        replaced = [(name, value) for name, value in scope.get("headers") or []
                    if name.lower() != b"content-length"]
        replaced.append((b"content-length", str(len(raw)).encode()))
        scope = dict(scope, headers=replaced)
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": raw, "more_body": False}
            return await receive()

        return await self.app(scope, replay, send)


def main() -> None:
    defaults, rest = split_arguments(sys.argv[1:])
    model = model_of(rest)
    import mlx_vlm.server as server

    server.app.add_middleware(AdjustRequests, model=model, defaults=defaults)
    sys.argv = ["mlx_vlm.server", *rest]
    server.main()


if __name__ == "__main__":
    main()
