"""Checking an uploaded image before it is forwarded to an engine.

Two separate checks, both required by the completion spec: a byte/pixel/
dimension ceiling, and a MIME type read from the bytes themselves rather than
trusted from whatever filename the client sent. A client can name a file
`scan.png` and send anything inside it; only the bytes say what it actually
is.

Deliberately dependency-free: the handful of magic-byte signatures and header
layouts below cover the formats AI-Lab's OCR and image routes are contracted
to accept, and reading just enough of each header to learn its dimensions is
a few lines per format rather than a new library.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


class UploadRejected(ValueError):
    """The upload fails a configured limit or is not a recognized image."""


@dataclass(frozen=True, slots=True)
class ImageInfo:
    mime: str
    width: int
    height: int


def sniff_image(data: bytes) -> ImageInfo:
    """The real MIME type and pixel dimensions, read from the file's own bytes.

    Raises `UploadRejected` for anything that is not one of the image formats
    recognized here — never guesses, and never falls back to a client-supplied
    Content-Type.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return _png(data)
    if data.startswith(b"\xff\xd8\xff"):
        return _jpeg(data)
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return _gif(data)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _webp(data)
    raise UploadRejected(
        "The file is not a recognized image format (PNG, JPEG, GIF or WebP).")


def validate_image(data: bytes, *, max_bytes: int = 0, max_pixels: int = 0,
                    max_dimension: int = 0) -> ImageInfo:
    """Sniff the image, then enforce whichever limits are configured (nonzero).

    Byte size is checked first and cheaply, before the header is even parsed,
    so an oversized upload never costs the cost of parsing it.
    """
    if max_bytes and len(data) > max_bytes:
        raise UploadRejected(
            f"The image is {len(data)} bytes, over the {max_bytes} byte limit.")
    info = sniff_image(data)
    if max_dimension and max(info.width, info.height) > max_dimension:
        raise UploadRejected(
            f"The image is {info.width}x{info.height}, over the "
            f"{max_dimension} pixel dimension limit.")
    pixels = info.width * info.height
    if max_pixels and pixels > max_pixels:
        raise UploadRejected(
            f"The image is {pixels} pixels, over the {max_pixels} pixel limit.")
    return info


def _png(data: bytes) -> ImageInfo:
    if len(data) < 24 or data[12:16] != b"IHDR":
        raise UploadRejected("The PNG header is truncated or malformed.")
    width, height = struct.unpack(">II", data[16:24])
    return ImageInfo("image/png", width, height)


def _jpeg(data: bytes) -> ImageInfo:
    index = 2
    length = len(data)
    while index + 4 <= length:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        if marker == 0xD9:
            break
        segment_length = struct.unpack(">H", data[index + 2:index + 4])[0]
        # SOF0-SOF3, SOF5-SOF7, SOF9-SOF11, SOF13-SOF15 all carry the frame's
        # dimensions in the same layout; anything else is skipped whole.
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                     0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if index + 9 > length:
                break
            height, width = struct.unpack(">HH", data[index + 5:index + 9])
            return ImageInfo("image/jpeg", width, height)
        index += 2 + segment_length
    raise UploadRejected("The JPEG header is truncated or has no frame marker.")


def _gif(data: bytes) -> ImageInfo:
    if len(data) < 10:
        raise UploadRejected("The GIF header is truncated.")
    width, height = struct.unpack("<HH", data[6:10])
    return ImageInfo("image/gif", width, height)


def _webp(data: bytes) -> ImageInfo:
    if len(data) < 30:
        raise UploadRejected("The WebP header is truncated.")
    chunk = data[12:16]
    if chunk == b"VP8 ":
        width, height = struct.unpack("<HH", data[26:30])
        return ImageInfo("image/webp", width & 0x3FFF, height & 0x3FFF)
    if chunk == b"VP8L":
        packed = struct.unpack("<I", data[21:25])[0]
        width = (packed & 0x3FFF) + 1
        height = ((packed >> 14) & 0x3FFF) + 1
        return ImageInfo("image/webp", width, height)
    if chunk == b"VP8X":
        width = 1 + (data[24] | (data[25] << 8) | (data[26] << 16))
        height = 1 + (data[27] | (data[28] << 8) | (data[29] << 16))
        return ImageInfo("image/webp", width, height)
    raise UploadRejected("The WebP header has no recognized chunk type.")
