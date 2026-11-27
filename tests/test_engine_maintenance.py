"""Updating an engine: refused under a running model, otherwise handed to its build or install.

Every collaborator is a stand-in: nothing is installed, compiled or fetched.
"""

import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from ai_lab import changes
from ai_lab.application.engine_maintenance import EngineMaintenanceService, MaintenanceParts
from ai_lab.changes import Reader
from ai_lab.changes.frompackages import Move, installed_version, would_move
from ai_lab.types import IMAGES, TOOLS, Change, Format, Interests


class Recorder:
    """A build or an install: what it was asked, and a fixed answer."""

    job: Any = None

    def __init__(self, status=None, components=True):
        self.asked = []
        self._status = status or {}
        if components:
            self.update_component = lambda name: self._say("update_component", name)

    def _say(self, action, *args):
        self.asked.append((action, *args))
        return {"done": action}

    def status(self):
        return self._status

    def check(self):
        return self._say("check")

    def activate(self, name):
        return self._say("activate", name)

    def remove(self, name):
        return self._say("remove", name)

    def update(self):
        return self._say("update")

    def install(self, version):
        return self._say("install", version)


class Holder:
    """`builds` or `installs`: one engine's item, KeyError for any other."""

    def __init__(self, engine_id, item):
        self.engine_id, self.item = engine_id, item

    def get(self, engine_id):
        if engine_id != self.engine_id:
            raise KeyError(engine_id)
        return self.item

    def all(self, **_):
        return [self.item.status()]


def service(builds=None, installs=None, running=(), **parts):
    rows = [{"id": name, "running": True} for name in running]
    return EngineMaintenanceService(
        MaintenanceParts(parts.get("store"), parts.get("catalog"), parts.get("host"),
                         parts.get("engines"), cast(Any, builds), cast(Any, installs)),
        lambda: rows)


class GuardedActionsTests(unittest.TestCase):
    def test_what_replaces_the_engine_waits_until_nothing_runs(self):
        build, install = Recorder(), Recorder()
        refused = service(Holder("llamacpp", build), Holder("vllm", install), running=("qwen",))
        actions = {
            "activate_install": lambda: refused.activate_install("vllm", "v2"),
            "update_install_component": lambda: refused.update_install_component("vllm", "node"),
            "activate_build": lambda: refused.activate_build("llamacpp", "b2"),
            "update_engine": lambda: refused.update_engine("llamacpp"),
        }
        for name, act in actions.items():
            with self.subTest(name), self.assertRaisesRegex(
                    ValueError, "Unload the running instances first: qwen"):
                act()
        self.assertEqual((build.asked, install.asked), ([], []))

    def test_with_nothing_running_each_action_reaches_its_build_or_install(self):
        build, install = Recorder(), Recorder()
        free = service(Holder("llamacpp", build), Holder("vllm", install))
        free.activate_install("vllm", "v2")
        free.update_install_component("vllm", "node")
        free.install_engine("vllm", "0.27")
        free.activate_build("llamacpp", "b2")
        free.update_engine("llamacpp")
        free.check_for_update("llamacpp")
        self.assertEqual(install.asked, [("activate", "v2"), ("update_component", "node"),
                                         ("install", "0.27")])
        self.assertEqual(build.asked, [("activate", "b2"), ("update",), ("check",)])

    def test_a_job_is_checked_again_right_before_it_switches_the_engine(self):
        # Audit m12: a model loaded while the job ran must stop the switch.
        from ai_lab.builds.background_job import BackgroundJob
        build = Recorder()
        build.job = BackgroundJob("t", "a build")
        rows: list[dict] = []
        maintenance = EngineMaintenanceService(
            MaintenanceParts(None, None, None, None, cast(Any, Holder("llamacpp", build)), None),
            lambda: rows)
        maintenance.update_engine("llamacpp")
        build.job.swap_allowed()
        rows.append({"id": "qwen", "running": True})
        with self.assertRaisesRegex(ValueError, "qwen.*loaded while this ran"):
            build.job.swap_allowed()

    def test_removing_a_spare_version_is_allowed_while_a_model_runs(self):
        build, install = Recorder(), Recorder()
        busy = service(Holder("llamacpp", build), Holder("vllm", install), running=("qwen",))
        self.assertEqual(busy.remove_install("vllm", "v1"), {"done": "remove"})
        self.assertEqual(busy.remove_build("llamacpp", "b1"), {"done": "remove"})

    def test_an_install_without_components_says_so(self):
        plain = service(installs=Holder("vllm", Recorder(components=False)))
        with self.assertRaisesRegex(ValueError, "no separately managed components"):
            plain.update_install_component("vllm", "node")

    def test_a_machine_without_builds_or_installs_says_which_is_missing(self):
        bare = service()
        with self.assertRaisesRegex(KeyError, "built from source"):
            bare.update_engine("llamacpp")
        with self.assertRaisesRegex(KeyError, "installed as packages"):
            bare.install_status("vllm")
        self.assertEqual((bare.build_status(), bare.installs_available()), ([], []))


class Model(SimpleNamespace):
    pass


class InterestsTests(unittest.TestCase):
    def maintenance(self, entries, models):
        config = SimpleNamespace(repositories=[], instances=entries, engines={})
        return service(store=SimpleNamespace(load=lambda: config),
                       catalog=SimpleNamespace(scan=lambda _repositories: models),
                       host=SimpleNamespace(capabilities=lambda: SimpleNamespace(
                           accelerator_kind="cuda")))

    def test_what_the_configured_entries_use_is_what_an_update_is_read_against(self):
        models = [Model(id="g/qwen", format=Format.GGUF, capabilities=frozenset({IMAGES, TOOLS})),
                  Model(id="n/text", format=Format.NVFP4, capabilities=frozenset({IMAGES}))]
        entries = [SimpleNamespace(model_id="g/qwen", params={"language_model_only": True}),
                   SimpleNamespace(model_id="n/text", params={}),
                   SimpleNamespace(model_id="gone/model", params={})]
        found = self.maintenance(entries, models).interests()
        self.assertEqual(found, Interests(accelerator_kind="cuda",
                                          formats=frozenset({"gguf", "nvfp4"}),
                                          pictures=True, tools=True))

    def test_an_entry_that_switches_pictures_off_is_not_using_them(self):
        models = [Model(id="g/qwen", format=Format.GGUF, capabilities=frozenset({IMAGES}))]
        entries = [SimpleNamespace(model_id="g/qwen", params={"language_model_only": True})]
        self.assertFalse(self.maintenance(entries, models).interests().pictures)


class WhatWouldChangeTests(unittest.TestCase):
    def test_the_page_gets_versions_changes_packages_and_what_could_not_be_read(self):
        found = changes.Changes(installed="b1", latest="b2",
                                yours=(Change("CUDA", "faster kernels", "abc"),),
                                others=(Change("Metal", "mac only"),), notes="notes",
                                unreadable="no network.")
        reader = mock.Mock()
        reader.moves.return_value = ([Move("torch", "2.8", "2.7")], "uv was slow.")
        reader.read.return_value = found
        config = SimpleNamespace(repositories=[], instances=[],
                                 engines={"llamacpp": {"source": {"path": "/src"}}},
                                 machine_limits=SimpleNamespace(
                                     builds=SimpleNamespace(system_path="/opt/homebrew/bin:/bin")))
        maintenance = service(
            Holder("llamacpp", Recorder({"installed": "b1", "latest": "b2"})),
            store=SimpleNamespace(load=lambda: config),
            catalog=SimpleNamespace(scan=lambda _repositories: []),
            host=SimpleNamespace(capabilities=lambda: SimpleNamespace(accelerator_kind="cuda")),
            engines=SimpleNamespace(get=lambda _id: SimpleNamespace(name="llama.cpp")))
        with mock.patch("ai_lab.application.engine_maintenance.Reader",
                        return_value=reader) as made:
            answer = maintenance.what_would_change("llamacpp")
        # Audit m26: the dry run asks with the machine's own search path.
        self.assertEqual(made.call_args.kwargs["system_path"], "/opt/homebrew/bin:/bin")
        reader.read.assert_called_once()
        self.assertEqual(reader.read.call_args.args[:2], ("b1", "b2"))
        self.assertEqual(answer["engine_name"], "llama.cpp")
        self.assertEqual([item["title"] for item in answer["yours"]], ["faster kernels"])
        self.assertEqual(answer["by_area"], {"CUDA": 1})
        self.assertEqual(answer["packages"][0]["backwards"], True)
        self.assertEqual(answer["unreadable"], "no network. uv was slow.")


def script(folder: Path, name: str, body: str) -> str:
    path = folder / name
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        self.addCleanup(self._folder.cleanup)
        self.folder = Path(self._folder.name)

    def test_each_source_is_read_and_a_failing_one_costs_only_its_part(self):
        reader = Reader("llamacpp", {"path": "/src", "releases": "ggml-org/llama.cpp"})
        commits = [Change("CUDA", "one"), Change("Metal", "two")]
        with mock.patch.object(changes, "waiting", side_effect=ValueError("not a git checkout")), \
                mock.patch.object(changes, "between", return_value=("the notes", "")):
            found = reader.read("b1", "b2", Interests(accelerator_kind="cuda"))
        self.assertEqual((found.notes, found.yours), ("the notes", ()))
        self.assertIn("Could not read the checkout: not a git checkout", found.unreadable)
        with mock.patch.object(changes, "waiting", return_value=(commits, 5)), \
                mock.patch.object(changes, "between", return_value=("", "no release b2.")):
            found = reader.read("b1", "b2", Interests(accelerator_kind="cuda"))
        self.assertEqual([item.title for item in found.yours + found.others], ["one", "two"])
        self.assertIn("5 older changes beyond the 2 shown", found.unreadable)
        self.assertIn("no release b2.", found.unreadable)

    def test_an_engine_with_no_sources_reads_nothing_and_says_nothing_moves(self):
        reader = Reader("comfyui", None)
        self.assertEqual(reader.moves(), ([], ""))
        self.assertEqual(reader.versions([]), ("", ""))
        self.assertEqual(reader.read("", "", Interests(accelerator_kind="none")).unreadable, "")

    def test_a_package_engine_says_what_it_has_and_what_it_would_become(self):
        venv = self.folder / ".venv"
        (venv / "bin").mkdir(parents=True)
        script(venv / "bin", "python", "echo 0.26.0\n")
        reader = Reader("vllm", {"package": "vllm"}, binary=str(venv / "bin" / "vllm"))
        self.assertEqual(reader.versions([Move("VLLM", "0.26.0", "0.27.1")]), ("0.26.0", "0.27.1"))
        self.assertEqual(reader.versions([]), ("0.26.0", "0.26.0"))         # nothing to do

    def test_packages_that_would_move_and_a_uv_that_cannot_answer(self):
        uv = script(self.folder, "uv", "echo ' - torch==2.8.0'\necho ' + torch==2.7.1'\n"
                                       "echo ' + newdep==1.0'\necho ' - vllm==0.26.0'\n"
                                       "echo ' + vllm==0.27.1'\n")
        moves = would_move(str(self.folder), "vllm", uv=uv)
        self.assertEqual([(move.name, move.was, move.becomes) for move in moves],
                         [("torch", "2.8.0", "2.7.1"), ("newdep", "", "1.0"),
                          ("vllm", "0.26.0", "0.27.1")])
        self.assertTrue(moves[0].heavy and moves[0].backwards)
        broken = script(self.folder, "broken-uv", "echo 'no index' >&2\nexit 2\n")
        with self.assertRaisesRegex(ValueError, "no index"):
            would_move(str(self.folder), "vllm", uv=broken)
        reader = Reader("vllm", {"package": "vllm"}, binary=str(self.folder / "bin" / "vllm"),
                        uv=broken)
        self.assertEqual(reader.moves(), ([], "Could not work out which packages would change: "
                                              "no index"))

    def test_the_installed_version_is_empty_when_the_environment_cannot_say(self):
        self.assertEqual(installed_version(str(self.folder / "missing"), "vllm"), "")
        venv = self.folder / "venv"
        (venv / "bin").mkdir(parents=True)
        script(venv / "bin", "python", "exit 1\n")
        self.assertEqual(installed_version(str(venv), "vllm"), "")




if __name__ == "__main__":
    unittest.main()
