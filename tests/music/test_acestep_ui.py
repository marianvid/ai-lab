"""ACE-Step's native editor must reuse the AI-Lab inference handlers."""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_lab.music.server import launch_native_ui


class AceStepNativeUiTests(unittest.TestCase):
    def test_ui_reuses_model_and_opens_on_its_companion_port(self):
        seen = {}
        demo = types.SimpleNamespace(
            queue=lambda **kwargs: seen.setdefault("queue", kwargs),
            launch=lambda **kwargs: seen.setdefault("launch", kwargs),
        )
        package = types.ModuleType("acestep")
        package.__path__ = []
        pipeline = types.ModuleType("acestep.acestep_v15_pipeline")
        pipeline.get_gpu_config = lambda: types.SimpleNamespace(recommended_backend="mlx")
        pipeline.set_global_gpu_config = lambda config: seen.setdefault("gpu", config)
        pipeline.create_demo = lambda **kwargs: seen.setdefault("init", kwargs) and demo
        backend = types.SimpleNamespace(dit=object(), llm=object(),
                                        output_root=Path("/state/music"))

        with patch.dict(sys.modules, {"acestep": package,
                                      "acestep.acestep_v15_pipeline": pipeline}):
            launch_native_ui(backend, "acestep-v15-xl-turbo",
                             Path("/models/ace"), 18112)

        params = seen["init"]["init_params"]
        self.assertIs(params["dit_handler"], backend.dit)
        self.assertIs(params["llm_handler"], backend.llm)
        self.assertTrue(params["pre_initialized"])
        self.assertEqual(seen["launch"]["server_port"], 18112)
        self.assertFalse(seen["launch"]["share"])
        self.assertEqual(seen["queue"]["default_concurrency_limit"], 1)


if __name__ == "__main__":
    unittest.main()
