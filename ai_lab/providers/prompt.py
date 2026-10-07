"""An OpenAI chat request turned into one prompt, and the answer turned back.

A CLI takes one block of text. The conversation is written out with its roles,
system instructions first; a request for JSON becomes an instruction, since a
CLI has no `response_format`.
"""

from __future__ import annotations

import json
import time
import uuid

NO_TOOLS = "Do not use any tools. Answer directly."
JSON_ONLY = "Answer with one JSON value only, no prose and no code fences."
SYSTEM_ROLES = ("system", "developer")


def render(payload: dict) -> str:
    """The whole request as the text a CLI reads on standard input."""
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("the request must carry messages")
    system = [_text(item) for item in messages if item.get("role") in SYSTEM_ROLES]
    turns = [item for item in messages if item.get("role") not in SYSTEM_ROLES]
    if not turns:
        raise ValueError("the request must carry a user message")
    parts = [*system, NO_TOOLS, *_json_instruction(payload.get("response_format")),
             *_conversation(turns)]
    return "\n\n".join(part for part in parts if part)


def _conversation(turns: list[dict]) -> list[str]:
    """One question as itself; a longer exchange with each speaker named."""
    if len(turns) == 1:
        return [_text(turns[0])]
    return [f"{str(item.get('role', 'user')).capitalize()}: {_text(item)}"
            for item in turns]


def completion(model: str, text: str) -> dict:
    """The answer in the OpenAI `chat.completion` shape."""
    return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}]}


def _text(message: dict) -> str:
    """A message's text, from a plain string or a list of text parts."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if any(part.get("type") != "text" for part in content if isinstance(part, dict)):
            raise ValueError("subscription models take text only, not images or files")
        return "\n".join(str(part.get("text", "")) for part in content
                         if isinstance(part, dict))
    raise ValueError("every message must have text content")


def _json_instruction(response_format) -> list[str]:
    if not isinstance(response_format, dict):
        return []
    if response_format.get("type") == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema")
        return [JSON_ONLY, f"It must match this JSON schema: {json.dumps(schema)}"]
    if response_format.get("type") == "json_object":
        return [JSON_ONLY]
    return []
