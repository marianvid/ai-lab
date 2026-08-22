"""What a model can do, read from its own files and remembered."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from ai_lab import capabilities
from ai_lab.capabilities import IMAGES, TOOLS, FILE_NAME, Known


def _gguf(path: Path, metadata: dict[str, str]) -> None:
    """Write the smallest GGUF file whose metadata can be read.

    Only string values, because that is all the capability reader looks at:
    it wants `tokenizer.chat_template` and nothing else.
    """
    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", 3)          # version
    out += struct.pack("<Q", 0)          # no tensors
    out += struct.pack("<Q", len(metadata))
    for key, value in metadata.items():
        raw = key.encode()
        out += struct.pack("<Q", len(raw)) + raw
        out += struct.pack("<I", 8)      # 8 is "string"
        raw = value.encode()
        out += struct.pack("<Q", len(raw)) + raw
    path.write_bytes(bytes(out))


class ReadingWeights(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.state = Path(tempfile.mkdtemp())
        self.known = Known(self.state)

    # -- GGUF, one file with its metadata at the head ----------------------

    def test_gguf_chat_template_offering_tools_means_tools(self):
        weights = self.dir / "model.gguf"
        _gguf(weights, {"tokenizer.chat_template":
                        "{% for t in tools %}{{ t.name }}{% endfor %}"})
        self.assertEqual(self.known.of(str(weights), True, ["model.gguf"]),
                         frozenset({TOOLS}))

    def test_gguf_without_the_word_tools_can_only_talk(self):
        weights = self.dir / "model.gguf"
        _gguf(weights, {"tokenizer.chat_template": "{{ messages[0].content }}"})
        self.assertEqual(self.known.of(str(weights), True, ["model.gguf"]),
                         frozenset())

    def test_gguf_with_a_projector_beside_it_can_see(self):
        weights = self.dir / "model.gguf"
        _gguf(weights, {"tokenizer.chat_template": "{{ messages }}"})
        (self.dir / "mmproj-model.gguf").write_bytes(b"")
        found = self.known.of(str(weights), True,
                              ["model.gguf", "mmproj-model.gguf"])
        self.assertEqual(found, frozenset({IMAGES}))

    # -- a directory of weights, where the answer is in the JSON -----------

    def test_directory_needs_both_marks_to_claim_pictures(self):
        # A text model's config can mention a vision tower it does not use.
        # Only the pair — a vision section *and* a token to put a picture in —
        # means the loaded model will accept one.
        (self.dir / "config.json").write_text(json.dumps(
            {"vision_config": {"hidden_size": 1}}))
        self.assertNotIn(IMAGES, self.known.of(str(self.dir), False,
                                               ["config.json"]))

        (self.dir / "config.json").write_text(json.dumps(
            {"vision_config": {"hidden_size": 1}, "image_token_id": 7}))
        Known(Path(tempfile.mkdtemp()))     # a fresh memory, so it reads again
        self.assertIn(IMAGES, Known(Path(tempfile.mkdtemp())).of(
            str(self.dir), False, ["config.json"]))

    def test_directory_reads_a_separate_template_file(self):
        (self.dir / "chat_template.jinja").write_text("{{ tools | length }}")
        self.assertIn(TOOLS, self.known.of(str(self.dir), False,
                                           ["chat_template.jinja"]))

    def test_directory_reads_the_template_inside_the_tokenizer_config(self):
        (self.dir / "tokenizer_config.json").write_text(json.dumps(
            {"chat_template": "{% if tools %}x{% endif %}"}))
        self.assertIn(TOOLS, self.known.of(str(self.dir), False,
                                           ["tokenizer_config.json"]))

    # -- failure is quiet, because an icon is not worth an error page ------

    def test_a_model_that_cannot_be_read_can_do_nothing_in_particular(self):
        self.assertEqual(self.known.of(str(self.dir / "gone.gguf"), True, []),
                         frozenset())

    def test_a_file_that_is_not_gguf_at_all_does_not_raise(self):
        weights = self.dir / "model.gguf"
        weights.write_bytes(b"not a model, just bytes")
        self.assertEqual(self.known.of(str(weights), True, ["model.gguf"]),
                         frozenset())


class Remembering(unittest.TestCase):
    """Reading a GGUF model costs about a quarter of a second. Once."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.state = Path(tempfile.mkdtemp())
        self.weights = self.dir / "model.gguf"
        _gguf(self.weights, {"tokenizer.chat_template": "{{ tools }}"})

    def test_a_second_manager_does_not_read_the_file_again(self):
        first = Known(self.state)
        self.assertEqual(first.of(str(self.weights), True, ["model.gguf"]),
                         frozenset({TOOLS}))
        self.assertTrue((self.state / FILE_NAME).exists())

        # A second manager, as after a restart, answers the same without
        # opening anything. Counted rather than timed: the point is that the
        # file is not read, and a stopwatch would only say it was quick.
        reads = []
        original = capabilities._read
        capabilities._read = lambda *args: reads.append(1) or frozenset()
        try:
            answer = Known(self.state).of(str(self.weights), True, ["model.gguf"])
        finally:
            capabilities._read = original
        self.assertEqual(answer, frozenset({TOOLS}))
        self.assertEqual(reads, [], "read the file again despite having the answer")

    def test_replacing_the_file_is_read_again(self):
        known = Known(self.state)
        self.assertEqual(known.of(str(self.weights), True, ["model.gguf"]),
                         frozenset({TOOLS}))
        # Same name, different content: a re-quantised model, or a download
        # that was resumed wrongly. Believing the old answer would show an
        # icon for something the new file cannot do.
        _gguf(self.weights, {"tokenizer.chat_template": "{{ messages }}"})
        self.assertEqual(Known(self.state).of(str(self.weights), True,
                                              ["model.gguf"]),
                         frozenset())

    def test_forgetting_clears_the_file_too(self):
        known = Known(self.state)
        known.of(str(self.weights), True, ["model.gguf"])
        known.forget()
        self.assertEqual(json.loads((self.state / FILE_NAME).read_text()), {})

    def test_without_a_state_directory_it_still_answers(self):
        # Tests build a catalog with no state directory, and so does anything
        # running before the host has said where state lives.
        self.assertEqual(Known(None).of(str(self.weights), True, ["model.gguf"]),
                         frozenset({TOOLS}))

    def test_an_unreadable_note_is_not_fatal(self):
        (self.state / FILE_NAME).write_text("{ this is not json")
        self.assertEqual(Known(self.state).of(str(self.weights), True,
                                              ["model.gguf"]),
                         frozenset({TOOLS}))


if __name__ == "__main__":
    unittest.main()
