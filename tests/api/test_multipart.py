import unittest

from ai_lab.api.multipart import MultipartBody


BOUNDARY = "----ai-lab-test"


def request() -> MultipartBody:
    data = (
        f"--{BOUNDARY}\r\n"
        'Content-Disposition: form-data; name="model"\r\n\r\n'
        "whisper\r\n"
        f"--{BOUNDARY}\r\n"
        'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + b"\x00\x01audio\xff\r\n" + f"--{BOUNDARY}--\r\n".encode()
    return MultipartBody(f"multipart/form-data; boundary={BOUNDARY}", data)


class MultipartBodyTests(unittest.TestCase):
    def test_reads_a_text_field_without_touching_the_file(self):
        self.assertEqual(request().field("model"), "whisper")

    def test_replaces_only_the_requested_field(self):
        original = request()
        changed = original.replace("model", "whisper-large-v3")
        parsed = MultipartBody(original.content_type, changed)
        self.assertEqual(parsed.field("model"), "whisper-large-v3")
        self.assertIn(b"\x00\x01audio\xff", changed)

    def test_missing_fields_are_visible(self):
        self.assertIsNone(request().field("language"))
        with self.assertRaises(ValueError):
            request().replace("language", "ro")


if __name__ == "__main__":
    unittest.main()
