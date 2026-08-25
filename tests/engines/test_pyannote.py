import unittest

from ai_lab.engines.pyannote import PyannoteEngine
from ai_lab.types import Format, ModelFile, ModelSet, Task


class PyannoteEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = PyannoteEngine(binary="/opt/ai/pyannote/.venv/bin/python",
                                     server="/opt/ai-lab/ai_lab/audio/server.py")
        self.model = ModelSet(id="audio/community-1", name="community-1",
                              format=Format.PYANNOTE,
                              entrypoint="/models/audio/diarization/pyannote/community-1",
                              files=(ModelFile("config.yaml", 10),),
                              task=Task.DIARIZATION)

    def test_it_serves_diarization(self):
        self.assertEqual(self.engine.tasks(), frozenset({Task.DIARIZATION}))
        self.assertIn("/v1/audio/diarizations", self.engine.api_paths())

    def test_it_starts_the_isolated_adapter(self):
        argv = self.engine.plan(self.model, 8101, {}).argv
        self.assertEqual(argv[argv.index("--backend") + 1], "pyannote")
        self.assertNotIn("--precision", argv)


if __name__ == "__main__":
    unittest.main()
