"""The shared stand-ins behave like the parts they replace (the reviewer's items 12-13)."""

import gc
import unittest
from pathlib import Path

from ai_lab.types import ModelFile, ProcessSpec

from .support.fakes import CACHE_MB, FakeEngine, FakeHost


class Model:
    """A model of two files, 1 GiB and 512 MiB."""

    files = (ModelFile(path="/m/a.gguf", size_bytes=1 << 30), ModelFile(path="/m/b.gguf", size_bytes=1 << 29))
    entrypoint = "/m/a.gguf"

    @property
    def size_bytes(self) -> int:
        return sum(item.size_bytes for item in self.files)


class FakeEngineTests(unittest.TestCase):
    def test_it_answers_a_real_size_never_cannot_say(self):
        self.assertEqual(FakeEngine().needs_mb(Model(), {}, 32000.0), 1536.0 + CACHE_MB)

    def test_only_the_process_started_for_a_port_answers_on_it(self):
        host = FakeHost()
        engine = FakeEngine(host=host)
        plan = engine.plan(Model(), 8081, {})
        host.start(ProcessSpec(instance_id="other", argv=plan.argv))
        self.assertFalse(engine.ready(8080), "another entry's process answered for this port")
        self.assertTrue(engine.ready(8081))
        host.stop("other")
        self.assertFalse(engine.ready(8081), "a stopped process still answered")

    def test_the_host_keeps_its_state_in_a_folder_of_its_own_and_takes_it_away(self):
        host = FakeHost()
        folder = Path(host.state_dir())
        self.assertTrue(folder.is_dir())
        del host
        gc.collect()
        self.assertFalse(folder.exists(), "the host's folder outlived it")


if __name__ == "__main__":
    unittest.main()
