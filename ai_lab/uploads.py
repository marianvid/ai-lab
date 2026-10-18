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


# How many bytes each header needs before its dimensions can be read.
PNG_HEADER_BYTES = 24
GIF_HEADER_BYTES = 10
WEBP_HEADER_BYTES = 30
# JPEG markers: every marker starts with 0xFF; start and end of image, the
# standalone TEM marker and the restart markers carry no length; the frame
# (SOF) markers below carry the image's dimensions in the same layout.
JPEG_MARKER = 0xFF
JPEG_START, JPEG_END, JPEG_TEM = 0xD8, 0xD9, 0x01
JPEG_RESTART_FIRST, JPEG_RESTART_LAST = 0xD0, 0xD7
JPEG_FRAME_MARKERS = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                                0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})


class UploadRejectedError(ValueError):
    """The upload fails a configured limit or is not a recognized image."""


@dataclass(frozen=True, slots=True)
class ImageInfo:
    """What an image really is: its MIME type and its size in pixels."""

    mime: str
    width: int
    height: int


def sniff_image(data: bytes) -> ImageInfo:
    """The real MIME type and pixel dimensions, read from the file's own bytes.

    Raises `UploadRejectedError` for anything that is not one of the image formats
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
    raise UploadRejectedError(
        "The file is not a recognized image format (PNG, JPEG, GIF or WebP).")


def validate_image(data: bytes, *, max_bytes: int = 0, max_pixels: int = 0,
                    max_dimension: int = 0) -> ImageInfo:
    """Sniff the image, then enforce whichever limits are configured (nonzero).

    Byte size is checked first and cheaply, before the header is even parsed,
    so an oversized upload never costs the cost of parsing it.
    """
    if max_bytes and len(data) > max_bytes:
        raise UploadRejectedError(
            f"The image is {len(data)} bytes, over the {max_bytes} byte limit.")
    info = sniff_image(data)
    if max_dimension and max(info.width, info.height) > max_dimension:
        raise UploadRejectedError(
            f"The image is {info.width}x{info.height}, over the "
            f"{max_dimension} pixel dimension limit.")
    pixels = info.width * info.height
    if max_pixels and pixels > max_pixels:
        raise UploadRejectedError(
            f"The image is {pixels} pixels, over the {max_pixels} pixel limit.")
    return info


def _png(data: bytes) -> ImageInfo:
    if len(data) < PNG_HEADER_BYTES or data[12:16] != b"IHDR":
        raise UploadRejectedError("The PNG header is truncated or malformed.")
    width, height = struct.unpack(">II", data[16:24])
    return ImageInfo("image/png", width, height)


def _jpeg(data: bytes) -> ImageInfo:
    index = 2
    length = len(data)
    while index + 4 <= length:
        if data[index] != JPEG_MARKER:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (JPEG_START, JPEG_TEM) or JPEG_RESTART_FIRST <= marker <= JPEG_RESTART_LAST:
            index += 2
            continue
        if marker == JPEG_END:
            break
        segment_length = struct.unpack(">H", data[index + 2:index + 4])[0]
        # SOF0-SOF3, SOF5-SOF7, SOF9-SOF11, SOF13-SOF15 all carry the frame's
        # dimensions in the same layout; anything else is skipped whole.
        if marker in JPEG_FRAME_MARKERS:
            if index + 9 > length:
                break
            height, width = struct.unpack(">HH", data[index + 5:index + 9])
            return ImageInfo("image/jpeg", width, height)
        index += 2 + segment_length
    raise UploadRejectedError("The JPEG header is truncated or has no frame marker.")


def _gif(data: bytes) -> ImageInfo:
    if len(data) < GIF_HEADER_BYTES:
        raise UploadRejectedError("The GIF header is truncated.")
    width, height = struct.unpack("<HH", data[6:10])
    return ImageInfo("image/gif", width, height)


def _webp(data: bytes) -> ImageInfo:
    if len(data) < WEBP_HEADER_BYTES:
        raise UploadRejectedError("The WebP header is truncated.")
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
    raise UploadRejectedError("The WebP header has no recognized chunk type.")
