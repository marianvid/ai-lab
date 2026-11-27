"""Upload validation: MIME sniffed from bytes, not trusted from a filename."""

import struct
import unittest

from ai_lab.uploads import UploadRejectedError, sniff_image, validate_image


def png(width: int, height: int) -> bytes:
    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">II", width, height) + bytes(5)  # bit depth etc, unread
    chunk = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + struct.pack(">I", 0)
    return header + chunk


def jpeg(width: int, height: int) -> bytes:
    soi = b"\xff\xd8"
    # A minimal SOF0 segment: length(2) precision(1) height(2) width(2) ncomp(1)
    body = struct.pack(">B", 8) + struct.pack(">HH", height, width) + struct.pack(">B", 3)
    sof0 = b"\xff\xc0" + struct.pack(">H", len(body) + 2) + body
    eoi = b"\xff\xd9"
    return soi + sof0 + eoi


def gif(width: int, height: int) -> bytes:
    return b"GIF89a" + struct.pack("<HH", width, height) + bytes(3)


def webp(chunk: bytes, body: bytes) -> bytes:
    """A WebP's first 30 bytes: RIFF, the WEBP mark, one chunk and what it starts with."""
    return (b"RIFF" + struct.pack("<I", 1000) + b"WEBP" + chunk + body).ljust(30, b"\0")


class SniffTests(unittest.TestCase):
    def test_each_kind_of_webp_reports_its_real_dimensions(self):
        lossy = webp(b"VP8 ", bytes(10) + struct.pack("<HH", 640, 480))
        packed = (1023 - 1) | ((768 - 1) << 14)
        lossless = webp(b"VP8L", bytes(5) + struct.pack("<I", packed))
        extended = webp(b"VP8X", bytes(8) + (1919).to_bytes(3, "little")
                        + (1079).to_bytes(3, "little"))
        for data, size in ((lossy, (640, 480)), (lossless, (1023, 768)), (extended, (1920, 1080))):
            with self.subTest(chunk=data[12:16]):
                info = sniff_image(data)
                self.assertEqual((info.mime, info.width, info.height), ("image/webp", *size))

    def test_a_webp_that_is_cut_short_or_of_an_unknown_kind_is_refused(self):
        with self.assertRaisesRegex(UploadRejectedError, "WebP header is truncated"):
            sniff_image(webp(b"VP8 ", b"")[:20])
        with self.assertRaisesRegex(UploadRejectedError, "no recognized chunk type"):
            sniff_image(webp(b"ABCD", b""))

    def test_a_png_reports_its_real_dimensions(self):
        info = sniff_image(png(640, 480))
        self.assertEqual((info.mime, info.width, info.height), ("image/png", 640, 480))

    def test_a_jpeg_reports_its_real_dimensions(self):
        info = sniff_image(jpeg(800, 600))
        self.assertEqual((info.mime, info.width, info.height), ("image/jpeg", 800, 600))

    def test_a_gif_reports_its_real_dimensions(self):
        info = sniff_image(gif(100, 50))
        self.assertEqual((info.mime, info.width, info.height), ("image/gif", 100, 50))

    def test_a_renamed_text_file_is_not_believed(self):
        # The classic attack this guards against: `evil.exe` saved as `scan.png`.
        with self.assertRaises(UploadRejectedError):
            sniff_image(b"MZ\x90\x00 not actually an image, whatever the name says")

    def test_a_truncated_header_is_rejected_rather_than_crashing(self):
        with self.assertRaises(UploadRejectedError):
            sniff_image(b"\x89PNG\r\n\x1a\n\x00\x00\x00")


class ValidateTests(unittest.TestCase):
    def test_within_every_limit_is_accepted(self):
        info = validate_image(png(100, 100), max_bytes=10_000,
                              max_pixels=100_000, max_dimension=1000)
        self.assertEqual((info.width, info.height), (100, 100))

    def test_over_the_byte_limit_is_rejected_before_parsing(self):
        with self.assertRaises(UploadRejectedError):
            validate_image(png(100, 100), max_bytes=4)

    def test_over_the_dimension_limit_is_rejected(self):
        with self.assertRaises(UploadRejectedError):
            validate_image(png(9000, 100), max_dimension=8000)

    def test_over_the_pixel_limit_is_rejected(self):
        with self.assertRaises(UploadRejectedError):
            validate_image(png(5000, 5000), max_pixels=1_000_000)

    def test_a_zero_limit_means_unlimited(self):
        # No limit configured must not accidentally become "reject everything".
        info = validate_image(png(50000, 50000), max_bytes=0, max_pixels=0,
                              max_dimension=0)
        self.assertEqual((info.mime, info.width, info.height), ("image/png", 50000, 50000))


if __name__ == "__main__":
    unittest.main()
