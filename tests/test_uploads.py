"""Upload validation: MIME sniffed from bytes, not trusted from a filename."""

import struct
import unittest

from ai_lab.api.uploads import UploadRejected, sniff_image, validate_image


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


class SniffTests(unittest.TestCase):
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
        with self.assertRaises(UploadRejected):
            sniff_image(b"MZ\x90\x00 not actually an image, whatever the name says")

    def test_a_truncated_header_is_rejected_rather_than_crashing(self):
        with self.assertRaises(UploadRejected):
            sniff_image(b"\x89PNG\r\n\x1a\n\x00\x00\x00")


class ValidateTests(unittest.TestCase):
    def test_within_every_limit_is_accepted(self):
        info = validate_image(png(100, 100), max_bytes=10_000,
                              max_pixels=100_000, max_dimension=1000)
        self.assertEqual((info.width, info.height), (100, 100))

    def test_over_the_byte_limit_is_rejected_before_parsing(self):
        with self.assertRaises(UploadRejected):
            validate_image(png(100, 100), max_bytes=4)

    def test_over_the_dimension_limit_is_rejected(self):
        with self.assertRaises(UploadRejected):
            validate_image(png(9000, 100), max_dimension=8000)

    def test_over_the_pixel_limit_is_rejected(self):
        with self.assertRaises(UploadRejected):
            validate_image(png(5000, 5000), max_pixels=1_000_000)

    def test_a_zero_limit_means_unlimited(self):
        # No limit configured must not accidentally become "reject everything".
        validate_image(png(50000, 50000), max_bytes=0, max_pixels=0,
                       max_dimension=0)


if __name__ == "__main__":
    unittest.main()
