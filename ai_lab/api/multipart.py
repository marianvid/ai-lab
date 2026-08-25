"""The small part of multipart requests AI-Lab needs to understand.

OpenAI's transcription API sends the model name and the audio file in one
multipart body. AI-Lab has to read the model name before choosing an engine,
then replace it with the name that engine was started under. Every other byte,
especially the audio, is forwarded unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from email.message import Message


@dataclass(frozen=True, slots=True)
class MultipartBody:
    content_type: str
    data: bytes

    def field(self, name: str) -> str | None:
        found = self._part(name)
        if found is None:
            return None
        _, _, content = found
        return content.removesuffix(b"\r\n").decode("utf-8")

    def replace(self, name: str, value: str) -> bytes:
        found = self._part(name)
        if found is None:
            raise ValueError(f"multipart request has no {name!r} field")
        start, end, content = found
        suffix = b"\r\n" if content.endswith(b"\r\n") else b""
        return self.data[:start] + value.encode("utf-8") + suffix + self.data[end:]

    def _part(self, wanted: str) -> tuple[int, int, bytes] | None:
        boundary = self._boundary()
        delimiter = b"--" + boundary
        position = 0
        for part in self.data.split(delimiter):
            part_start = position
            position += len(part) + len(delimiter)
            if not part.startswith(b"\r\n"):
                continue
            headers, separator, content = part[2:].partition(b"\r\n\r\n")
            if not separator or self._name(headers) != wanted:
                continue
            content_start = part_start + 2 + len(headers) + len(separator)
            return content_start, content_start + len(content), content
        return None

    def _boundary(self) -> bytes:
        message = Message()
        message["content-type"] = self.content_type
        boundary = message.get_param("boundary")
        if not boundary:
            raise ValueError("multipart request has no boundary")
        return str(boundary).encode("ascii")

    @staticmethod
    def _name(headers: bytes) -> str | None:
        match = re.search(br'(?i)content-disposition:[^\r\n]*\bname="([^"]+)"',
                          headers)
        return match.group(1).decode("utf-8") if match else None
