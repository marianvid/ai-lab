"""A request body AI-Lab reads as a stream: a YuE2 bundle put back (`application/x-tar`).

It is not read before the handler runs: the handler checks `length` first, then reads
the body piece by piece, so a large bundle never sits whole in memory. The server
closes the connection after such a request, since a refused body may be left unread.
"""
from __future__ import annotations

from typing import Protocol

# The body types handed to the handler as a stream instead of read as JSON.
RAW_TYPES = ("application/x-tar",)


class Readable(Protocol):
    """Anything a body is read from: the request's socket file, or bytes in a test."""

    def read(self, size: int = -1, /) -> bytes:
        """Up to `size` bytes."""
        ...


def is_raw(content_type: str) -> bool:
    """Whether a body of this type is handed over as a stream."""
    return content_type.split(";", 1)[0].strip().lower() in RAW_TYPES


class RawBody:
    """The body's type, its announced length, and the stream it is read from (at most `length` bytes)."""

    def __init__(self, content_type: str, length: int, stream: Readable) -> None:
        self.content_type = content_type
        self.length = length
        self._stream = stream
        self._left = length

    def read(self, size: int = -1) -> bytes:
        """Up to `size` bytes (all that is left when negative), never past `length`."""
        if self._left <= 0:
            return b""
        size = self._left if size is None or size < 0 else min(size, self._left)
        data = self._stream.read(size)
        self._left -= len(data)
        return data
