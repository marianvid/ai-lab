"""The shared part of the engines that run one of AI-Lab's server scripts."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.engines.base import MUSIC_PATHS, LaunchPlan
from ai_lab.engines.isolated import (
    AI_LAB_PACKAGE,
    REPOSITORY,
    ConfiguredEngine,
    IsolatedEngine,
    model_directory,
    script_env,
    ui_port,
    with_headroom_mb,
)
from ai_lab.engines.text_launch import SAMPLING, sampling_argv
from ai_lab.types import Format, ModelFile, ModelSet, Task

MIB = 1024 * 1024


def model(name="song", entrypoint="/models/song", size_mb=100, task=Task.MUSIC_GENERATION):
    return ModelSet(id=f"music/{name}", name=name, format=Format.SAFETENSORS, task=task,
                    entrypoint=entrypoint, files=(ModelFile("w.safetensors", size_mb * MIB),))


class Plain(IsolatedEngine):
    id = "plain"
    display_name = "Plain"
    task = Task.MUSIC_GENERATION
    paths = MUSIC_PATHS
    script = ("music", "server.py")

    def plan(self, model, port, params):
        self._checked(model, params)
        return LaunchPlan(argv=[self.binary, self.server], env=script_env())


class Configured(ConfiguredEngine):
    id = "configured"
    display_name = "Configured"
    task = Task.MUSIC_GENERATION
    paths = MUSIC_PATHS
    reserves_memory = True

    def plan(self, model, port, params):
        self._options(model, params)
        return LaunchPlan(argv=[self.binary], env={})


class HelperTests(unittest.TestCase):
    def test_the_editor_page_sits_beside_the_api(self):
        self.assertEqual(ui_port(8100), "18100")

    def test_a_checkpoint_file_means_its_folder(self):
        with TemporaryDirectory() as root:
            weights = Path(root) / "w.safetensors"
            weights.touch()
            self.assertEqual(model_directory(model(entrypoint=str(weights))), root)
            self.assertEqual(model_directory(model(entrypoint=root)), root)

    def test_scripts_run_unbuffered_with_the_repository_on_their_path(self):
        env = script_env(EXTRA="1")
        self.assertEqual((env["PYTHONUNBUFFERED"], env["PYTHONPATH"], env["HF_HUB_OFFLINE"],
                          env["EXTRA"]), ("1", str(REPOSITORY), "1", "1"))
        self.assertNotIn("HF_HUB_OFFLINE", script_env(offline=False))

    def test_headroom_is_half_again_or_the_minimum_whichever_is_more(self):
        self.assertEqual(with_headroom_mb(model(size_mb=100), 2048), 2148)
        self.assertEqual(with_headroom_mb(model(size_mb=10000), 2048), 15000)

    def test_sampling_is_four_settings_on_the_command_line(self):
        settings = {spec.key: spec.default for spec in SAMPLING}
        self.assertEqual(sampling_argv(settings),
                         ["--temp", "0.8", "--top-p", "0.95", "--top-k", "40", "--min-p", "0.05"])


class IsolatedEngineTests(unittest.TestCase):
    def test_the_server_script_is_found_under_the_package(self):
        engine = Plain()
        self.assertEqual(engine.server, str(AI_LAB_PACKAGE / "music" / "server.py"))
        self.assertEqual(engine.binary, "python")

    def test_one_task_one_shape_one_request_at_a_time(self):
        engine = Plain()
        self.assertEqual(engine.tasks(), frozenset({Task.MUSIC_GENERATION}))
        self.assertEqual(engine.api_paths(), MUSIC_PATHS)
        self.assertEqual(engine.api_paths(Task.OCR), ())
        self.assertEqual((engine.params(), engine.concurrency({})), ((), 1))
        self.assertEqual(engine.needs_mb(model(size_mb=7), {}, 0.0), 7.0)

    def test_a_model_of_the_wrong_kind_or_with_settings_is_refused(self):
        with self.assertRaisesRegex(ValueError, "cannot perform ocr"):
            Plain().plan(model(task=Task.OCR), 1, {})
        with self.assertRaisesRegex(ValueError, "Unknown settings"):
            Plain().plan(model(), 1, {"x": 1})


class ConfiguredEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = Configured(model_options={"song": {"memory_reservation_mb": 900}})

    def test_only_models_the_configuration_names(self):
        self.assertTrue(self.engine.supports(model()))
        self.assertFalse(self.engine.supports(model(name="other")))
        with self.assertRaisesRegex(ValueError, "Configured is not configured for other"):
            self.engine.plan(model(name="other"), 1, {})

    def test_instance_settings_are_refused(self):
        with self.assertRaisesRegex(ValueError, "has no instance settings"):
            self.engine.plan(model(), 1, {"steps": 3})

    def test_the_reservation_stands_in_for_the_file_size(self):
        self.assertEqual(self.engine.needs_mb(model(), {}, 0.0), 900.0)
