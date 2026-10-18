"""The pieces the application services are built from, each on its own."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from ai_lab.application.downloads import _candidates
from ai_lab.application.engine_maintenance import EngineMaintenanceService, MaintenanceParts
from ai_lab.application.instance_config import _free_port, _supported_model_ids
from ai_lab.application.instances import InstanceParts, InstanceService
from ai_lab.application.move_jobs import MoveCancelledError, MoveJobRecords
from ai_lab.changes.frompackages import Move, venv_of
from ai_lab.changes.sifting import UNSORTED, area_of, sift
from ai_lab.hosts import detect
from ai_lab.runtime.checks import StartChecks
from ai_lab.runtime.state import Operation, Step
from ai_lab.builds.source_checkout import GitCheckout
from ai_lab.builds.source_compiled import CompiledVersions
from ai_lab.builds.source_versions import Version, mark_version
from ai_lab.types import AcceleratorSnapshot, Change, Interests, Phase


def repository(format_name, root_id="core", task="", name="r"):
    return SimpleNamespace(format=format_name, root_id=root_id, task=task, name=name)


class DownloadsTests(unittest.TestCase):
    def test_the_repository_for_the_task_comes_first(self):
        generation, edit = repository("comfyui", task="gen"), repository("comfyui", task="edit")
        config = SimpleNamespace(repositories=[generation, edit, repository("gguf")])
        self.assertEqual(_candidates(config, "comfyui", "core", "edit"), [edit])
        self.assertEqual(_candidates(config, "comfyui", "core", ""), [generation, edit])
        self.assertEqual(_candidates(config, "comfyui", "fast", ""), [])


class Status:
    def __init__(self, answer):
        self.answer = answer

    def get(self, engine_id):
        if self.answer is None:
            raise KeyError(engine_id)
        return SimpleNamespace(status=lambda: self.answer)


class Reader:
    def versions(self, moves):
        return "1.0", "2.0"


class MaintenanceTests(unittest.TestCase):
    def service(self, builds, installs):
        parts = MaintenanceParts(None, None, None, None, cast(Any, builds), cast(Any, installs))
        return EngineMaintenanceService(parts, list)

    def test_the_build_answers_first(self):
        service = self.service(Status({"installed": "b1", "latest": "b2"}), None)
        self.assertEqual(service._versions("llamacpp", cast(Any, Reader()), []), ("b1", "b2"))

    def test_then_the_install_then_the_packages(self):
        service = self.service(Status(None), Status({"installed": "0.26"}))
        self.assertEqual(service._versions("vllm", cast(Any, Reader()), []), ("0.26", "2.0"))

    def test_nothing_running_is_required_before_an_update(self):
        parts = MaintenanceParts(None, None, None, None)
        service = EngineMaintenanceService(parts, lambda: [{"id": "qwen", "running": True}])
        with self.assertRaisesRegex(ValueError, "Unload the running instances first: qwen"):
            service.install_engine("vllm")


class InstanceConfigTests(unittest.TestCase):
    def test_an_engine_that_names_its_models_lists_only_those(self):
        engine = SimpleNamespace(model_modes={"voice"})
        description = {"formats": ["safetensors"], "tasks": ["speech"]}
        models = [{"id": "a", "name": "voice", "format": "safetensors", "task": "speech"},
                  {"id": "b", "name": "other", "format": "safetensors", "task": "speech"}]
        self.assertEqual(_supported_model_ids(engine, description, models), ["a"])
        self.assertIsNone(_supported_model_ids(SimpleNamespace(), description, models))

    def test_a_port_taken_or_the_manager_s_own_is_refused(self):
        config = SimpleNamespace(port=8090, instances=[SimpleNamespace(id="a", port=8080)])
        self.assertEqual(_free_port(config, "b", 8081), 8081)
        self.assertEqual(_free_port(config, "a", 8080), 8080)
        with self.assertRaisesRegex(ValueError, "already taken"):
            _free_port(config, "b", 8080)
        with self.assertRaisesRegex(ValueError, "manager's own"):
            _free_port(config, "b", 8090)


class Store:
    def __init__(self, config):
        self.config = config

    def load(self):
        return self.config


class InstancesTests(unittest.TestCase):
    def test_the_first_free_port_is_suggested(self):
        config = SimpleNamespace(port=8081, instances=[SimpleNamespace(port=8080)])
        parts = InstanceParts(cast(Any, Store(config)), cast(Any, None), cast(Any, None),
                              cast(Any, None), None, print, list)
        self.assertEqual(InstanceService(parts).suggest_port(), 8082)


class Host:
    def __init__(self, directory, snapshot=None):
        self.directory = directory
        self.snapshot = snapshot

    def state_dir(self):
        return self.directory

    def accelerator(self, pid=None):
        return self.snapshot


class Records(MoveJobRecords):
    def __init__(self, directory):
        self.host = cast(Any, Host(directory))
        self.changes: list[str] = []
        self._changed = self.changes.append


class MoveJobTests(unittest.TestCase):
    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.records = Records(Path(self._temporary.name))
        self.job = {"id": "j1", "status": "copying", "updated_at": 1.0,
                    "staging": str(Path(self._temporary.name) / "staging")}
        self.records._write_job(self.job)

    def test_a_cancellation_reaches_the_move_through_the_file(self):
        self.records.cancel_move("j1")
        with self.assertRaises(MoveCancelledError):
            self.records._check_cancelled(self.job)

    def test_a_move_left_running_by_a_restart_is_marked_failed(self):
        recovered = self.records.recover_moves()
        self.assertEqual([job["status"] for job in recovered], ["failed"])
        self.assertIn("restart", self.records.move_jobs()[0]["error"])


class ChangesTests(unittest.TestCase):
    def test_a_package_going_backwards_is_said(self):
        self.assertTrue(Move("torch", "2.5.1", "2.4.0").backwards)
        self.assertFalse(Move("torch", "2.4.0", "2.5.1").backwards)
        self.assertEqual(venv_of("/opt/ai/vllm/.venv/bin/vllm"), "/opt/ai/vllm/.venv")

    def test_changes_are_sorted_into_mine_and_the_rest(self):
        self.assertEqual(area_of("an untagged sentence"), UNSORTED)
        cuda = Change(area="CUDA", title="cuda: faster", reference="a")
        metal = Change(area="Metal", title="metal: faster", reference="b")
        yours, others = sift([cuda, metal], Interests(accelerator_kind="metal"))
        self.assertEqual((yours, others), ((metal,), (cuda,)))


class DetectTests(unittest.TestCase):
    def test_each_engine_s_program_is_read_from_its_section(self):
        settings = {"vllm": {"binary": "/opt/ai/vllm/bin/vllm"}, "nemo": None}
        self.assertEqual(detect._binaries(settings, ("vllm", "nemo", "onnx")),
                         {"vllm": "/opt/ai/vllm/bin/vllm", "nemo": None, "onnx": None})

    def test_a_mac_gets_the_mac_host(self):
        with mock.patch.object(sys, "platform", "darwin"):
            host = detect.current_host({"comfyui": {"binary": "/x/python", "comfyui": "/x/main"}})
        self.assertEqual(type(host).__name__, "DarwinHost")


class Checks(StartChecks):
    def __init__(self, snapshot):
        self.host = cast(Any, Host(None, snapshot))


class StartChecksTests(unittest.TestCase):
    def test_a_model_bigger_than_the_free_card_is_refused(self):
        card = AcceleratorSnapshot(available=True, name="card", kind="cuda",
                                   memory_kind="dedicated", memory_used_mb=30000,
                                   memory_total_mb=32000)
        model = SimpleNamespace(name="big", size_bytes=10 * 1024 ** 3)
        with self.assertRaisesRegex(RuntimeError, "needs about 10.0 GB"):
            Checks(card)._refuse_if_it_cannot_fit(cast(Any, None), cast(Any, model),
                                                  SimpleNamespace())


class StateTests(unittest.TestCase):
    def test_an_operation_reads_as_its_steps(self):
        operation = Operation("qwen", "load", ok=True, total_ms=5,
                              steps=[Step(Phase.READY, 5)])
        self.assertEqual(operation.json()["steps"], [{"phase": "ready", "elapsed_ms": 5}])


class OneAnswer:
    """git that gives the same answer to every command."""

    def __init__(self, answer):
        self.answer = answer

    def run(self, path, *arguments, timeout=15):
        return self.answer

    def is_ancestor(self, path, tag, against):
        return None


def checkout_answering(answer):
    return GitCheckout(Path("/nowhere"), OneAnswer(answer))


class CheckoutTests(unittest.TestCase):
    def test_the_newest_tag_of_the_right_shape_is_taken(self):
        checkout = checkout_answering("b10448\nv0.3.0\nv0.2.0-rc1\n")
        self.assertEqual(checkout.newest_on("stable"), "v0.3.0")
        self.assertEqual(checkout.newest_on("nightly"), "b10448")


def compiled_under(root):
    return CompiledVersions(root, None, GitCheckout(Path("/nowhere"), OneAnswer("")))


class CompiledTests(unittest.TestCase):
    def test_builds_are_listed_and_the_one_in_use_marked(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for tag in ("b1", "b2"):
                (root / f"build-{tag}" / "bin").mkdir(parents=True)
                (root / f"build-{tag}" / "bin" / "llama-server").write_text("x")
                mark_version(root / f"build-{tag}", Version(tag, "c" + tag))
            compiled = compiled_under(root)
            compiled.point_at(root / "build-b2")
            found = {item.version.tag: item.active for item in compiled.environments()}
        self.assertEqual(found, {"b1": False, "b2": True})


if __name__ == "__main__":
    unittest.main()
