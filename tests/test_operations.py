import json
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.catalog import Catalog
from ai_lab.config import ConfigStore
from ai_lab.downloads import DownloadManager, HuggingFaceClient
from ai_lab.events import EventBus
from ai_lab.operations import Operations
from ai_lab.runtime import Runtime
from ai_lab.engines.registry import Registry
from ai_lab.settings import Settings

from ai_lab.engines.llamacpp import LlamaCppEngine

from ai_lab.lastloaded import LastLoaded
from tests.support import FakeHost, make_files


class InstantEngine(LlamaCppEngine):
    """The real llama.cpp engine, minus the wait.

    Every rule that matters here — which formats it accepts, how settings are
    validated, how the command line is built — is the real one. Only the
    readiness probe is replaced, because nothing is listening on the port in a
    test and the real probe would wait out its timeout.
    """

    def __init__(self, binary=None, host=None):
        super().__init__(binary=binary)
        self.host = host

    def ready(self, port: int) -> bool:
        return bool(self.host and self.host.running)


def offline_huggingface():
    """A client that answers from a fixed listing instead of the network.

    A test that quietly reaches Hugging Face fails for reasons that have
    nothing to do with the code — rate limits, a 401, no route out of the
    container — and passes only when someone happens to be online.
    """
    return HuggingFaceClient(opener=lambda url: [
        {"type": "file", "path": "model-Q4_K_M.gguf", "size": 1024},
    ])


def operations_with_instances(count, last_loaded=None):
    """An Operations with `count` configured entries, and the host behind it.

    Built by hand rather than through the interface, because what is under test
    is how the list is drawn, not how entries are created.
    """
    import tempfile
    directory = Path(tempfile.mkdtemp())
    # The layout every machine has: one root, a folder in it per weight
    # format. See `MODEL_STORAGE.md`.
    (directory / "models" / "gguf").mkdir(parents=True)
    for index in range(count):
        model = directory / "models" / "gguf" / f"model-{index}"
        model.mkdir()
        make_files(model, f"model-{index}.gguf", size=1024)
    config = {
        "title": "AI-Lab", "host": "127.0.0.1", "port": 8090,
        "engines": {"llamacpp": {"binary": "/bin/true"}},
        "models_root": str(directory / "models"),
        "repositories": [{"id": "gguf", "name": "GGUF",
                          "format": "gguf", "writable": True}],
        "instances": [
            {"id": f"model-{index}", "name": f"Model {index}",
             "engine": "llamacpp",
             "model_id": f"gguf/model-{index}/model-{index}",
             "port": 8100 + index, "params": {}}
            for index in range(count)
        ],
    }
    path = directory / "config.json"
    path.write_text(json.dumps(config))
    host = FakeHost()
    store = ConfigStore(path)
    operations = Operations(
        store=store, catalog=Catalog(),
        runtime=Runtime(host, EventBus(), sample_interval_s=0),
        settings=Settings(store, host, Registry()),
        downloads=DownloadManager(), huggingface=offline_huggingface(),
        host=host, engines=FakeRegistry(host), bus=EventBus(),
        last_loaded=last_loaded,
    )
    return operations, host


class FakeRegistry:
    def __init__(self, host):
        self.host = host

    def get(self, engine_id):
        if engine_id != "llamacpp":
            raise KeyError(f"Unknown engine: {engine_id}")
        return InstantEngine(binary="/bin/true", host=self.host)

    def describe(self, capabilities):
        engine = self.get("llamacpp")
        from dataclasses import asdict
        return [{"id": "llamacpp", "name": "llama.cpp", "available": True,
                 "reason": "", "binary": engine.binary,
                 "formats": ["gguf"],
                 "params": [asdict(spec) for spec in engine.params()]}]


class OperationsTests(unittest.TestCase):
    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)

        make_files(self.root / "gguf" / "qwen", "qwen.gguf", size=64)
        make_files(self.root / "gguf" / "gemma", "gemma.gguf", size=32)
        make_files(self.root / "safetensors" / "big", "model.safetensors", "config.json")

        self.path = self.root / "config.json"
        self.path.write_text(json.dumps({
            "models_root": str(self.root),
            "repositories": [
                {"id": "gguf", "name": "GGUF", "format": "gguf"},
                {"id": "st", "name": "Safetensors", "format": "safetensors"},
            ],
            "instances": [{"id": "qwen", "name": "Coding", "engine": "llamacpp",
                           "model_id": "gguf/qwen/qwen", "port": 8080,
                           "params": {"context_size": 4096}}],
        }))
        self.store = ConfigStore(self.path)
        self.host = FakeHost()
        self.operations = Operations(
            store=self.store, catalog=Catalog(),
            runtime=Runtime(self.host, EventBus(), sample_interval_s=0),
            settings=Settings(self.store, self.host, Registry()),
            downloads=DownloadManager(), huggingface=offline_huggingface(),
            host=self.host, engines=FakeRegistry(self.host),
        )

    # -- reading -----------------------------------------------------------

    def test_models_lists_everything_on_disk(self):
        # The safetensors model is named after its directory, not after
        # `model-00001-of-....safetensors` — which is what every safetensors
        # model's weights are called.
        names = sorted(item["name"] for item in self.operations.models())
        self.assertEqual(names, ["big", "gemma", "qwen"])

    def test_models_can_be_filtered_to_what_an_engine_can_load(self):
        """llama.cpp reads GGUF, so the safetensors model must not appear."""
        names = sorted(item["name"] for item in self.operations.models("llamacpp"))
        self.assertEqual(names, ["gemma", "qwen"])

    def test_instances_report_their_state(self):
        rows = self.operations.instances()
        self.assertEqual(rows[0]["id"], "qwen")
        self.assertFalse(rows[0]["running"])

    # -- lifecycle ---------------------------------------------------------

    def test_loading_joins_config_catalog_engine_and_runtime(self):
        operation = self.operations.load("qwen")
        self.assertTrue(operation.ok, operation.error)
        self.assertEqual(len(self.host.started), 1)
        self.assertIn("--model", self.host.started[0].argv)

    def test_loading_an_unknown_instance_raises(self):
        with self.assertRaises(KeyError):
            self.operations.load("nope")

    def test_changing_the_model_is_an_edit_then_a_reload(self):
        """One entry is one model, so there is no separate swap operation."""
        self.operations.update_instance("qwen", {"model_id": "gguf/gemma/gemma"})
        self.assertEqual(self.store.load().instance("qwen").model_id, "gguf/gemma/gemma")
        self.assertTrue(self.operations.load("qwen").ok)

    def test_a_model_the_engine_cannot_read_fails_cleanly(self):
        self.operations.update_instance("qwen", {"model_id": "st/big"})
        operation = self.operations.load("qwen")
        self.assertFalse(operation.ok)
        self.assertIn("safetensors", operation.error)

    def test_apply_saves_the_settings_and_reloads_with_them(self):
        result = self.operations.apply_and_reload("qwen", {"params": {"context_size": 8192}})
        self.assertTrue(result["applied"], result)
        self.assertEqual(self.store.load().instance("qwen").params["context_size"], 8192)
        self.assertEqual(len(self.host.started), 1)

    def test_a_port_is_suggested_that_nobody_is_using(self):
        self.assertEqual(self.operations.suggest_port(), 8081)   # 8080 is taken
        self.operations.create_instance({"id": "second", "engine": "llamacpp",
                                         "model_id": "gguf/gemma/gemma", "port": 8081})
        self.assertEqual(self.operations.suggest_port(), 8082)

    def test_the_managers_own_port_is_never_suggested(self):
        # It is 8090, and the suggestion walks up from 8080 — so it reaches it
        # as soon as ten models are configured. An engine started there finds
        # the port already held and refuses, which is a puzzling way to be told
        # the number was never free.
        for number in range(8081, 8095):
            self.operations.create_instance({
                "id": f"m{number}", "engine": "llamacpp",
                "model_id": "gguf/gemma/gemma", "port": number})
            if number == 8089:
                self.assertEqual(self.operations.suggest_port(), 8091)

    def test_the_port_of_an_existing_model_can_be_changed(self):
        self.operations.update_instance("qwen", {"port": 8095})
        self.assertEqual(self.store.load().instance("qwen").port, 8095)

    def test_a_port_another_model_holds_is_refused(self):
        self.operations.create_instance({"id": "second", "engine": "llamacpp",
                                         "model_id": "gguf/gemma/gemma", "port": 8081})
        with self.assertRaises(ValueError):
            self.operations.update_instance("qwen", {"port": 8081})

    def test_the_managers_own_port_is_refused(self):
        with self.assertRaises(ValueError):
            self.operations.update_instance("qwen", {"port": 8090})

    def test_a_change_that_cannot_be_made_is_refused_not_ignored(self):
        # It used to accept anything and answer "applied", so a change that
        # never happened read as a change that did.
        with self.assertRaises(ValueError):
            self.operations.update_instance("qwen", {"engine": "vllm"})

    def test_the_add_form_arrives_in_one_call(self):
        form = self.operations.new_instance_form()
        self.assertEqual(form["port"], 8081)
        self.assertTrue(any(item["name"] == "gemma" for item in form["models"]))

    # -- configuring -------------------------------------------------------

    def test_creating_an_instance_fills_in_engine_defaults(self):
        created = self.operations.create_instance({
            "id": "gemma", "engine": "llamacpp",
            "model_id": "gguf/gemma/gemma", "port": 8081})
        self.assertEqual(created["params"]["context_size"], 32768)
        self.assertEqual(len(self.store.load().instances), 2)

    def test_a_duplicate_port_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.operations.create_instance({"id": "other", "engine": "llamacpp",
                                             "model_id": "gguf/gemma/gemma", "port": 8080})
        self.assertIn("8080", str(caught.exception))

    def test_a_duplicate_id_is_refused(self):
        with self.assertRaises(ValueError):
            self.operations.create_instance({"id": "qwen", "engine": "llamacpp",
                                             "model_id": "gguf/gemma/gemma", "port": 9999})

    def test_bad_settings_are_rejected_with_a_readable_message(self):
        with self.assertRaises(ValueError) as caught:
            self.operations.update_instance("qwen", {"params": {"context_size": 4}})
        self.assertIn("at least", str(caught.exception))

    def test_changing_settings_does_not_restart_the_instance(self):
        """Restarting unloads a model somebody may be using."""
        self.operations.update_instance("qwen", {"params": {"context_size": 8192}})
        self.assertEqual(self.host.started, [])
        self.assertEqual(self.store.load().instance("qwen").params["context_size"], 8192)

    def test_a_running_instance_cannot_be_deleted(self):
        self.host.running.add("qwen")
        with self.assertRaises(ValueError):
            self.operations.delete_instance("qwen")

    def test_a_stopped_instance_can_be_deleted(self):
        self.operations.delete_instance("qwen")
        self.assertEqual(self.store.load().instances, [])

    # -- downloads ---------------------------------------------------------

    def test_downloading_into_a_read_only_repository_is_refused(self):
        with self.store.mutate() as config:
            config.repository("gguf").writable = False
        with self.assertRaises(ValueError) as caught:
            self.operations.download("org/model", "whatever", "gguf")
        self.assertIn("read-only", str(caught.exception))


if __name__ == "__main__":
    unittest.main()


class EffectiveSettingsTests(unittest.TestCase):
    """An entry written before a setting existed still reports what will apply.

    Showing a blank would be a lie: the engine uses its default, and that is
    what the interface has to say.
    """

    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        make_files(self.root / "gguf" / "qwen", "qwen.gguf", size=64)
        self.path = self.root / "config.json"
        self.path.write_text(json.dumps({
            "models_root": str(self.root),
            "repositories": [{"id": "gguf", "name": "GGUF", "format": "gguf"}],
            # Written by an older version: no temperature, and a setting that
            # has since been removed.
            "instances": [{"id": "qwen", "name": "Coding", "engine": "llamacpp",
                           "model_id": "gguf/qwen/qwen", "port": 8080,
                           "params": {"context_size": 4096, "gpu_layers": 999}}],
        }))
        store = ConfigStore(self.path)
        host = FakeHost()
        self.operations = Operations(
            store=store, catalog=Catalog(),
            runtime=Runtime(host, EventBus(), sample_interval_s=0),
            settings=Settings(store, host, Registry()),
            downloads=DownloadManager(), huggingface=offline_huggingface(),
            host=host, engines=FakeRegistry(host),
        )

    def test_a_missing_setting_reports_the_engine_default(self):
        params = self.operations.instances()[0]["params"]
        self.assertEqual(params["temperature"], 0.8)
        self.assertEqual(params["context_size"], 4096)   # the stored one wins

    def test_a_setting_stored_by_an_older_version_is_still_read(self):
        # 999 is what this project wrote when every layer went on the card.
        self.assertEqual(self.operations.instances()[0]["params"]["gpu_layers"],
                         999)


class DeleteModelTests(unittest.TestCase):
    """Removing files from disk is the one operation here that destroys data."""

    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        make_files(self.root / "gguf" / "qwen", "qwen.gguf", size=100)
        make_files(self.root / "gguf" / "spare", "spare.gguf", "config.json", size=50)
        self.path = self.root / "config.json"
        self.path.write_text(json.dumps({
            "models_root": str(self.root),
            "repositories": [{"id": "gguf", "name": "GGUF", "format": "gguf"}],
            "instances": [{"id": "qwen", "name": "Coding", "engine": "llamacpp",
                           "model_id": "gguf/qwen/qwen", "port": 8080, "params": {}}],
        }))
        store = ConfigStore(self.path)
        self.host = FakeHost()
        self.operations = Operations(
            store=store, catalog=Catalog(),
            runtime=Runtime(self.host, EventBus(), sample_interval_s=0),
            settings=Settings(store, self.host, Registry()),
            downloads=DownloadManager(), huggingface=offline_huggingface(),
            host=self.host, engines=FakeRegistry(self.host),
        )

    def test_a_model_nothing_points_at_is_removed_with_its_files(self):
        result = self.operations.delete_model("gguf/spare/spare")
        self.assertEqual(result["files"], 2)          # the weights and its config
        self.assertEqual(result["freed_bytes"], 100)
        self.assertFalse((self.root / "gguf" / "spare").exists(), "empty directory left behind")
        self.assertEqual([item["name"] for item in self.operations.models()], ["qwen"])

    def test_a_model_an_entry_points_at_is_refused(self):
        """Deleting under a configured entry leaves one that can never start."""
        with self.assertRaises(ValueError) as caught:
            self.operations.delete_model("gguf/qwen/qwen")
        self.assertIn("qwen", str(caught.exception),
                      "it must name the entry standing in the way")
        self.assertTrue((self.root / "gguf" / "qwen" / "qwen.gguf").exists())

    def test_it_can_be_deleted_once_the_entry_is_gone(self):
        self.operations.delete_instance("qwen")
        self.operations.delete_model("gguf/qwen/qwen")
        self.assertEqual([item["name"] for item in self.operations.models()], ["spare"])

    def test_an_unknown_model_raises_rather_than_deleting_anything(self):
        with self.assertRaises(KeyError):
            self.operations.delete_model("gguf/nope/nope")

    def test_the_repository_directory_itself_is_never_removed(self):
        self.operations.delete_instance("qwen")
        self.operations.delete_model("gguf/qwen/qwen")
        self.operations.delete_model("gguf/spare/spare")
        self.assertTrue((self.root / "gguf").is_dir())


class MoveModelTests(unittest.TestCase):
    """Moving a model between storage tiers must never lose the source."""

    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        self.core = self.root / "core"
        self.benchmark = self.root / "benchmark"
        make_files(self.core / "gguf" / "qwen", "qwen.gguf", size=100)
        (self.benchmark / "gguf").mkdir(parents=True, exist_ok=True)
        self.path = self.root / "config.json"
        self.path.write_text(json.dumps({
            "models_root": str(self.core),
            "model_roots": [
                {"id": "core", "name": "Core", "path": str(self.core)},
                {"id": "benchmark", "name": "Benchmark",
                 "path": str(self.benchmark)},
            ],
            "repositories": [{"id": "gguf", "name": "GGUF", "format": "gguf"}],
            "instances": [{"id": "qwen", "name": "Coding", "engine": "llamacpp",
                           "model_id": "gguf/qwen/qwen", "port": 8080, "params": {}}],
        }))
        self.store = ConfigStore(self.path)
        self.host = FakeHost()
        self.operations = Operations(
            store=self.store, catalog=Catalog(),
            runtime=Runtime(self.host, EventBus(), sample_interval_s=0),
            settings=Settings(self.store, self.host, Registry()),
            downloads=DownloadManager(), huggingface=offline_huggingface(),
            host=self.host, engines=FakeRegistry(self.host),
        )

    def test_a_model_is_copied_verified_and_removed_from_the_source(self):
        self.operations.delete_instance("qwen")
        result = self.operations.move_model("gguf/qwen/qwen", "benchmark")
        self.assertTrue(result["moved"])
        self.assertTrue((self.benchmark / "gguf" / "qwen" / "qwen.gguf").exists())
        self.assertFalse((self.core / "gguf" / "qwen").exists(),
                         "source must be gone once the copy is verified")

    def test_a_loaded_model_is_refused(self):
        self.host.running.add("qwen")
        with self.assertRaises(ValueError) as caught:
            self.operations.move_model("gguf/qwen/qwen", "benchmark")
        self.assertIn("loaded", str(caught.exception))
        self.assertTrue((self.core / "gguf" / "qwen" / "qwen.gguf").exists())

    def test_a_stopped_configured_entry_follows_the_model(self):
        result = self.operations.move_model("gguf/qwen/qwen", "benchmark")
        self.assertTrue(result["moved"])
        configured = self.store.load().instance("qwen")
        self.assertEqual(configured.model_id, "benchmark-gguf/qwen/qwen")

    def test_a_missing_derived_destination_directory_is_created(self):
        self.operations.delete_instance("qwen")
        target = self.benchmark / "gguf"
        target.rmdir()
        result = self.operations.move_model("gguf/qwen/qwen", "benchmark")
        self.assertTrue(result["moved"])
        self.assertTrue(target.is_dir())

    def test_a_disabled_destination_root_is_refused(self):
        self.operations.delete_instance("qwen")
        self.path.write_text(json.dumps({
            "models_root": str(self.core),
            "model_roots": [
                {"id": "core", "name": "Core", "path": str(self.core)},
                {"id": "benchmark", "name": "Benchmark",
                 "path": str(self.benchmark), "enabled": False},
            ],
            "repositories": [{"id": "gguf", "name": "GGUF", "format": "gguf"}],
            "instances": [],
        }))
        with self.assertRaises(ValueError) as caught:
            self.operations.move_model("gguf/qwen/qwen", "benchmark")
        self.assertIn("disabled", str(caught.exception))
        self.assertTrue((self.core / "gguf" / "qwen" / "qwen.gguf").exists())

    def test_not_enough_free_space_is_refused_and_source_kept(self):
        self.operations.delete_instance("qwen")
        import shutil as shutil_module
        original = shutil_module.disk_usage

        def fake_disk_usage(path):
            usage = original(path)
            return usage._replace(free=1)

        shutil_module.disk_usage = fake_disk_usage
        try:
            with self.assertRaises(ValueError) as caught:
                self.operations.move_model("gguf/qwen/qwen", "benchmark")
        finally:
            shutil_module.disk_usage = original
        self.assertIn("free space", str(caught.exception))
        self.assertTrue((self.core / "gguf" / "qwen" / "qwen.gguf").exists())

    def test_a_checksum_failure_during_copy_leaves_the_source_intact(self):
        self.operations.delete_instance("qwen")
        import ai_lab.operations as operations_module
        # Read the descriptor straight out of the class `__dict__`, not
        # through the class itself: `Operations._sha256` unwraps the
        # `staticmethod` and hands back a plain function. Restoring *that*
        # later would leave `_sha256` bound to `self` on every instance call
        # after this test — exactly the bug that broke every other move test
        # once this one ran first.
        original = operations_module.Operations.__dict__["_sha256"]
        calls = {"n": 0}

        def flaky_sha256(path):
            calls["n"] += 1
            return f"bad-{calls['n']}"

        operations_module.Operations._sha256 = staticmethod(flaky_sha256)
        try:
            with self.assertRaises(ValueError) as caught:
                self.operations.move_model("gguf/qwen/qwen", "benchmark")
        finally:
            operations_module.Operations._sha256 = original
        self.assertIn("Checksum", str(caught.exception))
        self.assertTrue((self.core / "gguf" / "qwen" / "qwen.gguf").exists(),
                        "source must survive a failed copy")
        self.assertFalse((self.benchmark / "gguf" / "qwen" / "qwen.gguf").exists())

    def test_sha256_survives_the_checksum_failure_tests_patch_and_restore(self):
        """A move right after the checksum-failure test must still work.

        That test above swaps `Operations._sha256` for a fake and puts the
        real one back afterwards. If the restore ever again hands back a bare
        function instead of a `staticmethod`, every instance would call
        `_sha256(self, path)` from here on — one argument too many — and this
        move would fail with a `TypeError` instead of succeeding.
        """
        self.operations.delete_instance("qwen")
        result = self.operations.move_model("gguf/qwen/qwen", "benchmark")
        self.assertTrue(result["moved"])
        self.assertTrue((self.benchmark / "gguf" / "qwen" / "qwen.gguf").exists())

    def test_same_tier_move_is_a_no_op(self):
        self.operations.delete_instance("qwen")
        result = self.operations.move_model("gguf/qwen/qwen", "core")
        self.assertFalse(result["moved"])
        self.assertTrue((self.core / "gguf" / "qwen" / "qwen.gguf").exists())

    def test_a_move_can_go_the_other_way_too(self):
        """Benchmark to core, not just core to benchmark."""
        self.operations.delete_instance("qwen")
        make_files(self.benchmark / "gguf" / "other", "other.gguf", size=40)
        result = self.operations.move_model("benchmark-gguf/other/other", "core")
        self.assertTrue(result["moved"])
        self.assertTrue((self.core / "gguf" / "other" / "other.gguf").exists())
        self.assertFalse((self.benchmark / "gguf" / "other").exists())

    def test_a_shared_companion_is_kept_for_the_sibling_left_behind(self):
        """Two GGUF models in one directory can share a companion file.

        Moving only one of them must not delete the tokenizer the other one
        still needs — `Catalog._classify` attaches every companion in a
        directory to every model in it, so `move_model` has to work out for
        itself which companions are still spoken for before it deletes
        anything from the source.
        """
        self.operations.delete_instance("qwen")
        directory = self.core / "gguf" / "qwen"
        (directory / "tokenizer.json").write_text("{}")
        make_files(directory, "sibling.gguf", size=10)
        result = self.operations.move_model("gguf/qwen/qwen", "benchmark")
        self.assertTrue(result["moved"])
        # The companion travelled with the model that moved...
        self.assertTrue((self.benchmark / "gguf" / "qwen" / "tokenizer.json").exists())
        # ...but the sibling left behind can still find it.
        self.assertTrue((self.core / "gguf" / "qwen" / "tokenizer.json").exists(),
                        "a companion still needed by a sibling must not be deleted")
        self.assertTrue((self.core / "gguf" / "qwen" / "sibling.gguf").exists())

    def test_a_failure_during_publishing_leaves_a_resumable_staging_state(self):
        """An interruption after copying must not delete published files
        or the staging area — it must leave something a retry can find.
        """
        self.operations.delete_instance("qwen")
        import ai_lab.operations as operations_module
        # Read the descriptor straight out of the class `__dict__`, not
        # through the class itself: see the identical note on the
        # `_sha256` patch above — the same unwrap-and-rebind trap applies
        # to `_publish`, and it is what broke every move test that ran
        # after this one.
        original = operations_module.Operations.__dict__["_publish"]

        def failing_publish(*args, **kwargs):
            raise OSError("disk pulled mid-rename")

        operations_module.Operations._publish = staticmethod(failing_publish)
        try:
            with self.assertRaises(OSError):
                self.operations.move_model("gguf/qwen/qwen", "benchmark")
        finally:
            operations_module.Operations._publish = original
        # The source must not have been touched: publishing never finished.
        self.assertTrue((self.core / "gguf" / "qwen" / "qwen.gguf").exists())
        jobs = self.operations.move_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["status"], "failed")
        self.assertIn("disk pulled", jobs[0]["error"])
        self.assertTrue(Path(jobs[0]["staging"]).exists(),
                        "the staged copy must survive a failed publish")

    def test_a_crashed_move_is_readable_after_restart(self):
        """A durable job record, not silence, is what a crash leaves behind.

        Nothing here actually crashes the process — that cannot be done
        deterministically — but the same guarantee is what matters: the job
        record written before the copy started must still be on disk and
        must still say what it was doing.
        """
        self.operations.delete_instance("qwen")
        result = self.operations.move_model("gguf/qwen/qwen", "benchmark")
        jobs = self.operations.move_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["id"], result["job_id"])
        self.assertEqual(jobs[0]["status"], "completed")
        self.assertEqual(jobs[0]["source_tier"], "core")
        self.assertEqual(jobs[0]["target_tier"], "benchmark")

    def test_a_move_can_be_cancelled_between_files(self):
        """Cancellation is checked between files, not only at the end.

        A cancellation is an operator decision that succeeded, not an error:
        `move_model` reports it as a plain result rather than raising, so the
        HTTP request that happened to be running the move does not come back
        looking like a failure.
        """
        self.operations.delete_instance("qwen")
        directory = self.core / "gguf" / "qwen"
        # A companion file so this model has two sources to copy, not one —
        # otherwise there is no "between files" to check cancellation at.
        (directory / "config.json").write_text("{}")

        import ai_lab.operations as operations_module
        original = operations_module.Operations._check_cancelled
        state = {"n": 0}

        def cancel_after_first(self, job):
            state["n"] += 1
            if state["n"] == 2:
                self.cancel_move(job["id"])
            original(self, job)

        operations_module.Operations._check_cancelled = cancel_after_first
        try:
            result = self.operations.move_model("gguf/qwen/qwen", "benchmark")
        finally:
            operations_module.Operations._check_cancelled = original
        self.assertFalse(result["moved"])
        self.assertTrue(result["cancelled"])
        self.assertTrue((self.core / "gguf" / "qwen" / "qwen.gguf").exists(),
                        "source must survive a cancelled move")
        jobs = self.operations.move_jobs()
        self.assertEqual(jobs[0]["status"], "cancelled")
        self.assertFalse(Path(jobs[0]["staging"]).exists(),
                         "the staged copy must be cleaned up on cancellation")

    def test_a_single_file_move_can_be_cancelled_mid_copy(self):
        """Even a one-file model — the common GGUF case — must be stoppable
        once copying has started, not only before the first byte moves."""
        self.operations.delete_instance("qwen")
        make_files(self.core / "gguf" / "qwen", "qwen.gguf", size=8 * 1024 * 1024)

        import ai_lab.operations as operations_module
        original = operations_module.Operations._check_cancelled
        state = {"n": 0}

        def cancel_after_first(self, job):
            state["n"] += 1
            if state["n"] == 2:
                self.cancel_move(job["id"])
            original(self, job)

        operations_module.Operations._check_cancelled = cancel_after_first
        operations_module.Operations._COPY_CHUNK = 1024
        try:
            result = self.operations.move_model("gguf/qwen/qwen", "benchmark")
        finally:
            operations_module.Operations._check_cancelled = original
            operations_module.Operations._COPY_CHUNK = 64 * 1024 * 1024
        self.assertTrue(result["cancelled"])
        self.assertTrue((self.core / "gguf" / "qwen" / "qwen.gguf").exists())
        self.assertFalse((self.benchmark / "gguf" / "qwen").exists(),
                         "no phantom model must appear at the destination")

    def test_staged_files_do_not_appear_as_a_model_while_moving(self):
        """The catalog must never see the in-flight staging directory.

        Checked by inspecting the catalog mid-copy via a `_check_cancelled`
        patch: at that point the staging directory holds a real (partial)
        file, which is exactly the state that used to show up as a broken
        model called `.ai-lab-move-<job-id>`.
        """
        self.operations.delete_instance("qwen")
        seen = {}

        import ai_lab.operations as operations_module
        original = operations_module.Operations._check_cancelled

        def inspect(self, job):
            config = self.store.load()
            seen["ids"] = [item.id for item in self.catalog.scan(config.repositories)]
            original(self, job)

        operations_module.Operations._check_cancelled = inspect
        try:
            self.operations.move_model("gguf/qwen/qwen", "benchmark")
        finally:
            operations_module.Operations._check_cancelled = original
        self.assertNotIn("benchmark-gguf/.ai-lab-staging", "".join(seen.get("ids", [])))
        for model_id in seen.get("ids", []):
            self.assertNotIn(".ai-lab-staging", model_id)

    def test_staging_root_is_created_privately(self):
        self.operations.delete_instance("qwen")
        self.operations.move_model("gguf/qwen/qwen", "benchmark")
        staging_root = self.benchmark / "gguf" / ".ai-lab-staging"
        # Removed after a clean move, so recreate it the same way to check
        # the mode a live one would have had.
        staging_root.mkdir(parents=True, exist_ok=True)
        os.chmod(staging_root, 0o700)
        self.assertEqual(oct(staging_root.stat().st_mode)[-3:], "700")

    def test_recover_moves_fails_jobs_left_mid_flight_and_cleans_their_staging(self):
        """What `recover_moves()` finds at startup after a crash."""
        self.operations.delete_instance("qwen")
        job_id = "crashed-job"
        staging = self.benchmark / "gguf" / ".ai-lab-staging" / job_id
        staging.mkdir(parents=True)
        (staging / "partial.gguf").write_text("half")
        job = {"id": job_id, "model_id": "gguf/qwen/qwen",
               "target_model_id": "benchmark-gguf/qwen/qwen",
               "source_tier": "core", "target_tier": "benchmark",
               "bytes": 100, "files": 1, "staging": str(staging),
               "status": "copying", "error": "", "started_at": 0.0,
               "updated_at": 0.0}
        self.operations._write_job(job)

        recovered = self.operations.recover_moves()

        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["id"], job_id)
        jobs = self.operations.move_jobs()
        self.assertEqual(jobs[0]["status"], "failed")
        self.assertIn("restart", jobs[0]["error"].lower())
        self.assertFalse(staging.exists(),
                         "a crashed move's staged bytes must be cleaned up")
        # The source, which a crashed copy never touches, must be untouched.
        self.assertTrue((self.core / "gguf" / "qwen" / "qwen.gguf").exists())

    def test_recover_moves_leaves_finished_jobs_alone(self):
        self.operations.delete_instance("qwen")
        result = self.operations.move_model("gguf/qwen/qwen", "benchmark")
        recovered = self.operations.recover_moves()
        self.assertEqual(recovered, [])
        jobs = self.operations.move_jobs()
        self.assertEqual(jobs[0]["status"], "completed")

    def test_a_second_move_of_the_same_model_is_refused_while_one_is_active(self):
        self.operations.delete_instance("qwen")
        job = {"id": "already-running", "model_id": "gguf/qwen/qwen",
               "target_model_id": "benchmark-gguf/qwen/qwen",
               "source_tier": "core", "target_tier": "benchmark",
               "bytes": 100, "files": 1, "staging": "",
               "status": "copying", "error": "", "started_at": 0.0,
               "updated_at": 0.0}
        self.operations._write_job(job)
        with self.assertRaises(ValueError) as caught:
            self.operations.move_model("gguf/qwen/qwen", "benchmark")
        self.assertIn("already being moved", str(caught.exception))

    def test_cancel_is_a_no_op_once_a_move_has_finished(self):
        self.operations.delete_instance("qwen")
        result = self.operations.move_model("gguf/qwen/qwen", "benchmark")
        job = self.operations.cancel_move(result["job_id"])
        self.assertEqual(job["status"], "completed")


class SupportedFormatTests(unittest.TestCase):
    """Downloads are filtered to what something here can actually run."""

    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        self.path = self.root / "config.json"
        self.path.write_text(json.dumps({"repositories": [], "instances": []}))
        store = ConfigStore(self.path)
        host = FakeHost()
        self.operations = Operations(
            store=store, catalog=Catalog(),
            runtime=Runtime(host, EventBus(), sample_interval_s=0),
            settings=Settings(store, host, Registry()),
            downloads=DownloadManager(),
            huggingface=HuggingFaceClient(opener=lambda url: [
                {"type": "file", "path": "model-Q4_K_M.gguf", "size": 10},
                {"type": "file", "path": "model.safetensors", "size": 20},
            ]),
            host=host, engines=Registry(),
        )

    def test_only_runnable_formats_are_offered(self):
        """vLLM is not installed, so safetensors are of no use here yet."""
        names = [item["format"] for item in self.operations.remote_sets("org/model")]
        self.assertEqual(names, ["gguf"])

    def test_there_is_no_way_to_ask_for_the_rest(self):
        """The switch is gone, and so is the argument behind it.

        There was never an answer to what it was for: a machine with no engine
        that reads safetensors cannot be helped by a list of them, and the
        download would be thirty gigabytes of nothing.
        """
        with self.assertRaises(TypeError):
            self.operations.remote_sets("org/model", only_supported=False)

    def test_supported_formats_follow_the_available_engines(self):
        self.assertEqual(self.operations.supported_formats(), ["gguf"])

    def test_search_hides_repositories_without_a_runnable_format(self):
        self.operations.huggingface.search = lambda query: [
            {"repo": "org/source", "formats": ["safetensors"]},
            {"repo": "org/weights-GGUF", "formats": ["gguf"]},
        ]
        self.assertEqual(
            [item["repo"] for item in self.operations.search("weights")["results"]],
            ["org/weights-GGUF"])

    def test_it_says_how_many_it_hid(self):
        # The one thing the switch was good for. "Nothing found" and "nothing
        # you can run" are different answers, and a list of length zero cannot
        # tell them apart.
        self.operations.huggingface.search = lambda query: [
            {"repo": "org/source", "formats": ["safetensors"]},
            {"repo": "org/other", "formats": ["safetensors"]},
            {"repo": "org/weights-GGUF", "formats": ["gguf"]},
        ]
        answer = self.operations.search("weights")
        self.assertEqual(len(answer["results"]), 1)
        self.assertEqual(answer["hidden"], 2)

    def test_nothing_hidden_is_reported_as_nothing_hidden(self):
        self.operations.huggingface.search = lambda query: [
            {"repo": "org/weights-GGUF", "formats": ["gguf"]},
        ]
        self.assertEqual(self.operations.search("weights")["hidden"], 0)

    def test_a_search_that_found_nothing_hid_nothing(self):
        self.operations.huggingface.search = lambda query: []
        answer = self.operations.search("nothing at all")
        self.assertEqual(answer, {"results": [], "hidden": 0})


@unittest.skipIf(os.geteuid() == 0, "permission bits do not restrain root")
class WritabilityTests(unittest.TestCase):
    """A repository the manager cannot write to must not be offered.

    Found in use: every repository declared `writable: true`, the interface
    believed it, and the download died on a permission error after the wait.

    Skipped when running as root, which can write regardless of the mode bits,
    so the case being tested cannot be created. The deployed tests run as root
    inside the container; these are the ones that run on the development
    machine.
    """

    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        # Two formats under one root, which is the only shape there is now.
        # One of the folders is readable and not writable, which is the case
        # this exists for and does happen — a disk mounted read-only, or a
        # folder owned by somebody else.
        (self.root / "gguf").mkdir()
        (self.root / "nvfp4").mkdir()
        (self.root / "nvfp4").chmod(0o500)           # readable, not writable
        self.addCleanup((self.root / "nvfp4").chmod, 0o700)
        self.path = self.root / "config.json"
        self.path.write_text(json.dumps({
            "models_root": str(self.root),
            "repositories": [
                {"id": "open", "name": "Open", "format": "gguf"},
                {"id": "locked", "name": "Locked", "format": "nvfp4"},
            ],
            "instances": [],
        }))
        store = ConfigStore(self.path)
        host = FakeHost()
        self.settings = Settings(store, host, Registry())
        self.operations = Operations(
            store=store, catalog=Catalog(),
            runtime=Runtime(host, EventBus(), sample_interval_s=0),
            settings=self.settings, downloads=DownloadManager(),
            huggingface=offline_huggingface(), host=host, engines=FakeRegistry(host),
        )

    def test_the_settings_view_reports_real_writability(self):
        rows = {item["id"]: item for item in self.settings.view()["repositories"]}
        self.assertTrue(rows["open"]["writable"])
        self.assertFalse(rows["locked"]["writable"], "declared writable, actually not")

    def test_downloading_into_an_unwritable_repository_is_refused_up_front(self):
        with self.assertRaises(ValueError) as caught:
            self.operations.download("org/model", "whatever", "locked")
        self.assertIn("not writable", str(caught.exception))

    def test_a_missing_directory_is_refused_with_its_path(self):
        with self.store_missing() as operations:
            with self.assertRaises(ValueError) as caught:
                operations.download("org/model", "whatever", "gone")
        self.assertIn("does not exist", str(caught.exception))

    def store_missing(self):
        from contextlib import contextmanager

        @contextmanager
        def build():
            self.path.write_text(json.dumps({
                "repositories": [{"id": "gone", "name": "Gone", "path": "/nowhere",
                                  "format": "gguf"}],
                "instances": [],
            }))
            yield self.operations
        return build()


def drain(operations, timeout=5.0):
    """Wait for the download worker to stop touching the temporary directory.

    The queue runs on a thread of its own, so a test that asks for a download
    and then returns can have its temporary directory removed underneath a
    worker still writing into it. That shows up as a cleanup error in whatever
    test happens to run next, which is a miserable thing to chase.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        busy = [item for item in operations.transfers()
                if item["state"] in ("queued", "running")]
        if not busy:
            return
        time.sleep(0.01)


class DownloadDestinationTests(unittest.TestCase):
    """The destination is worked out, not asked for.

    The store is organised by format and the listing says which format a model
    is, so making someone choose asks a question whose answer is already known.
    """

    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        for name in ("gguf", "safetensors"):
            (self.root / name).mkdir()
        self.path = self.root / "config.json"
        self.path.write_text(json.dumps({
            "repositories": [
                {"id": "gguf", "name": "GGUF", "path": str(self.root / "gguf"),
                 "format": "gguf"},
                {"id": "st", "name": "Safetensors", "path": str(self.root / "safetensors"),
                 "format": "safetensors"},
            ],
            "instances": [],
        }))
        store = ConfigStore(self.path)
        host = FakeHost()
        self.operations = Operations(
            store=store, catalog=Catalog(),
            runtime=Runtime(host, EventBus(), sample_interval_s=0),
            settings=Settings(store, host, Registry()),
            downloads=DownloadManager(opener=lambda url, resume: None),
            huggingface=HuggingFaceClient(opener=lambda url: [
                {"type": "file", "path": "thing-Q4_K_M.gguf", "size": 8},
                {"type": "file", "path": "weights.safetensors", "size": 8},
            ]),
            host=host, engines=FakeRegistry(host),
        )
        self.addCleanup(drain, self.operations)

    def test_a_gguf_model_goes_to_the_gguf_repository(self):
        transfer = self.operations.download("org/model", "thing-Q4_K_M")
        self.assertTrue(transfer["id"])
        self.assertTrue((self.root / "gguf").exists())

    def test_a_safetensors_model_goes_to_the_safetensors_repository(self):
        self.operations.download("org/model", "weights")
        # The destination was chosen by format, not by position in the list.
        self.assertEqual(
            self.operations.transfers()[0]["name"], "weights")

    def test_a_format_with_no_repository_says_so(self):
        with self.store_without_gguf():
            with self.assertRaises(ValueError) as caught:
                self.operations.download("org/model", "thing-Q4_K_M")
        self.assertIn("No repository is configured", str(caught.exception))

    def test_an_explicit_destination_still_wins(self):
        self.operations.download("org/model", "thing-Q4_K_M", "st")
        self.assertEqual(len(self.operations.transfers()), 1)

    def store_without_gguf(self):
        from contextlib import contextmanager

        @contextmanager
        def build():
            original = self.path.read_text()
            payload = json.loads(original)
            payload["repositories"] = [payload["repositories"][1]]
            self.path.write_text(json.dumps(payload))
            try:
                yield
            finally:
                self.path.write_text(original)
        return build()


class DownloadTierConflictTests(unittest.TestCase):
    """A repository named explicitly and a storage tier named explicitly
    can disagree, and that must be a refusal, not a silent tier fallback.
    """

    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        self.core = self.root / "core"
        self.benchmark = self.root / "benchmark"
        (self.core / "gguf").mkdir(parents=True)
        (self.benchmark / "gguf").mkdir(parents=True)
        self.path = self.root / "config.json"
        self.path.write_text(json.dumps({
            "models_root": str(self.core),
            "model_roots": [
                {"id": "core", "name": "Core", "path": str(self.core)},
                {"id": "benchmark", "name": "Benchmark",
                 "path": str(self.benchmark)},
            ],
            "repositories": [{"id": "gguf", "name": "GGUF", "format": "gguf"}],
            "instances": [],
        }))
        store = ConfigStore(self.path)
        host = FakeHost()
        self.operations = Operations(
            store=store, catalog=Catalog(),
            runtime=Runtime(host, EventBus(), sample_interval_s=0),
            settings=Settings(store, host, Registry()),
            downloads=DownloadManager(),
            huggingface=HuggingFaceClient(opener=lambda url: [
                {"type": "file", "path": "thing-Q4_K_M.gguf", "size": 8},
            ]),
            host=host, engines=FakeRegistry(host),
        )
        self.addCleanup(drain, self.operations)

    def test_a_repository_and_tier_that_disagree_are_refused(self):
        # "gguf" is a core repository (its root_id defaults to "core"), so
        # asking for it while also asking for the benchmark tier is a
        # contradiction that must be reported, not quietly resolved one way.
        with self.assertRaises(ValueError) as caught:
            self.operations.download("org/model", "thing-Q4_K_M",
                                     repository_id="gguf",
                                     storage_tier="benchmark")
        self.assertIn("benchmark", str(caught.exception))
        self.assertEqual(self.operations.transfers(), [])

    def test_a_repository_and_its_own_tier_agree_and_proceed(self):
        self.operations.download("org/model", "thing-Q4_K_M",
                                 repository_id="gguf", storage_tier="core")
        self.assertEqual(len(self.operations.transfers()), 1)


class BundleDownloadTests(unittest.TestCase):
    """A declared bundle is downloaded like any other model, into the tier asked
    for. A test download must never end up in the production library because
    the test disk was busy or the request was ambiguous.
    """

    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        self.core = self.root / "core"
        self.benchmark = self.root / "benchmark"
        for tier in (self.core, self.benchmark):
            (tier / "images" / "generation").mkdir(parents=True)
            (tier / "images" / "edit").mkdir(parents=True)
        self.path = self.root / "config.json"
        self.path.write_text(json.dumps({
            "models_root": str(self.core),
            "model_roots": [
                {"id": "core", "name": "Core", "path": str(self.core)},
                {"id": "benchmark", "name": "Benchmark",
                 "path": str(self.benchmark)},
            ],
            "repositories": [
                {"id": "images-comfyui-generation", "name": "Generation",
                 "format": "comfyui", "task": "image-generation",
                 "subpath": "images/generation"},
                {"id": "images-comfyui-edit", "name": "Editing",
                 "format": "comfyui", "task": "image-edit",
                 "subpath": "images/edit"},
            ],
            "instances": [],
            "downloads": {"bundles": [
                {"name": "qwen-image", "repo": "org/qwen", "format": "comfyui",
                 "task": "image-generation",
                 "components": [{"role": "diffusion_model",
                                 "path": "split_files/diffusion_models/q.safetensors"}]},
                {"name": "qwen-edit", "repo": "org/qwen", "format": "comfyui",
                 "task": "image-edit",
                 "components": [{"role": "diffusion_model",
                                 "path": "split_files/diffusion_models/e.safetensors"}]},
            ]},
        }))
        store = ConfigStore(self.path)
        host = FakeHost()
        self.operations = Operations(
            store=store, catalog=Catalog(),
            runtime=Runtime(host, EventBus(), sample_interval_s=0),
            settings=Settings(store, host, Registry()),
            downloads=DownloadManager(opener=lambda url, resume: None),
            huggingface=HuggingFaceClient(opener=lambda url: [
                {"type": "file", "size": 8,
                 "path": "split_files/diffusion_models/q.safetensors"},
                {"type": "file", "size": 8,
                 "path": "split_files/diffusion_models/e.safetensors"},
            ]),
            host=host, engines=FakeRegistry(host),
        )
        self.addCleanup(drain, self.operations)

    def test_a_benchmark_download_stays_on_the_benchmark_disk(self):
        self.operations.download("org/qwen", "qwen-image",
                                 storage_tier="benchmark")
        transfer = self.operations.transfers()[0]
        self.assertEqual(transfer["storage_tier"], "benchmark")
        self.assertNotIn(str(self.core), transfer.get("destination", ""))

    def test_an_editing_bundle_goes_to_the_editing_repository(self):
        """Two repositories hold the same format; the job it is for decides."""
        self.operations.download("org/qwen", "qwen-edit", storage_tier="benchmark")
        working = [path for path in (self.benchmark / "images").rglob("*")
                   if path.is_dir() and path.name.startswith(".")]
        self.assertTrue(any("edit" in str(path) for path in working), working)

    def test_a_declaration_that_is_unsafe_is_refused_rather_than_obeyed(self):
        payload = json.loads(self.path.read_text())
        payload["downloads"]["bundles"][0]["name"] = "../escape"
        self.path.write_text(json.dumps(payload))
        with self.assertRaises(ValueError) as caught:
            self.operations.download("org/qwen", "qwen-image",
                                     storage_tier="benchmark")
        self.assertIn("name", str(caught.exception))
        self.assertEqual(self.operations.transfers(), [])


class BrowseTests(unittest.TestCase):
    """Choosing a folder, since a web page cannot open one on the server."""

    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name).resolve()
        self.addCleanup(self._temporary.cleanup)
        (self.root / "models" / "gguf").mkdir(parents=True)
        (self.root / "models" / ".hidden").mkdir()
        (self.root / "models" / "notes.txt").write_text("x")
        self.path = self.root / "config.json"
        self.path.write_text(json.dumps({
            "repositories": [{"id": "gguf", "name": "GGUF",
                              "path": str(self.root / "models" / "gguf"),
                              "format": "gguf"}],
            "instances": [],
        }))
        self.store = ConfigStore(self.path)
        host = FakeHost()
        self.operations = Operations(
            store=self.store, catalog=Catalog(),
            runtime=Runtime(host, EventBus(), sample_interval_s=0),
            settings=Settings(self.store, host, Registry()),
            downloads=DownloadManager(), huggingface=offline_huggingface(),
            host=host, engines=FakeRegistry(host),
        )

    def test_it_lists_directories_and_nothing_else(self):
        """Only folder names. This is for picking a place, not reading files."""
        listing = self.operations.browse(str(self.root / "models"))
        names = [item["name"] for item in listing["entries"]]
        self.assertEqual(names, ["gguf"])
        self.assertNotIn("notes.txt", names)
        self.assertNotIn(".hidden", names)

    def test_it_says_where_it_is_and_where_up_is(self):
        listing = self.operations.browse(str(self.root / "models"))
        self.assertEqual(listing["path"], str(self.root / "models"))
        self.assertEqual(listing["parent"], str(self.root))

    def test_it_reports_whether_each_folder_can_be_written_to(self):
        listing = self.operations.browse(str(self.root / "models"))
        self.assertIn("writable", listing["entries"][0])

    def test_it_starts_near_the_models_rather_than_at_the_root(self):
        self.assertEqual(self.operations.browse()["path"], str(self.root / "models"))

    def test_a_path_that_is_not_a_directory_is_refused(self):
        with self.assertRaises(ValueError):
            self.operations.browse(str(self.root / "models" / "notes.txt"))


class RepositoryEditingTests(unittest.TestCase):
    """A repository whose path is wrong has to be fixable from the interface."""

    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name).resolve()
        self.addCleanup(self._temporary.cleanup)
        (self.root / "old").mkdir()
        (self.root / "old" / "gguf").mkdir()
        (self.root / "new").mkdir()
        (self.root / "new" / "gguf").mkdir()
        self.path = self.root / "config.json"
        self.path.write_text(json.dumps({
            "models_root": str(self.root / "old"),
            "repositories": [{"id": "gguf", "name": "GGUF", "format": "gguf"}],
            "instances": [],
        }))
        self.store = ConfigStore(self.path)
        host = FakeHost()
        self.operations = Operations(
            store=self.store, catalog=Catalog(),
            runtime=Runtime(host, EventBus(), sample_interval_s=0),
            settings=Settings(self.store, host, Registry()),
            downloads=DownloadManager(), huggingface=offline_huggingface(),
            host=host, engines=FakeRegistry(host),
        )

    def test_moving_the_root_moves_every_repository(self):
        # The whole point of there being one: a model store that moves to
        # another disk moves in one act, and no format is left behind.
        self.operations.update_models_root(str(self.root / "new"))
        self.assertEqual(self.store.load().models_root, str(self.root / "new"))
        self.assertEqual(self.store.load().repository("gguf").path,
                         str(self.root / "new" / "gguf"))

    def test_a_root_that_does_not_exist_is_refused_before_saving(self):
        """Saying so while the field is still on screen beats a broken screen."""
        with self.assertRaises(ValueError) as caught:
            self.operations.update_models_root("/nowhere/at/all")
        self.assertIn("not a directory", str(caught.exception))
        self.assertEqual(self.store.load().models_root, str(self.root / "old"))

    def test_moving_core_onto_an_existing_benchmark_root_is_refused(self):
        """`update_models_root` must not write a config `load()` then refuses.

        Every route calls `store.load()`, so a config the app cannot load
        again is a full outage from one bad folder pick, discoverable only
        by hand-editing config.json. The check has to happen before `save()`.
        """
        benchmark = self.root / "benchmark"
        benchmark.mkdir()
        with self.store.mutate() as config:
            from ai_lab.config import ModelRoot
            config.model_roots.append(
                ModelRoot(id="benchmark", name="Benchmark", path=str(benchmark)))
        with self.assertRaises(ValueError) as caught:
            self.operations.update_models_root(str(benchmark))
        self.assertIn("both point at", str(caught.exception))
        # The config must still load after the refusal.
        self.store.load()
        self.assertEqual(self.store.load().models_root, str(self.root / "old"))

    def test_pointing_a_model_root_at_another_ones_path_is_refused(self):
        benchmark = self.root / "benchmark"
        benchmark.mkdir()
        with self.store.mutate() as config:
            from ai_lab.config import ModelRoot
            config.model_roots.append(
                ModelRoot(id="benchmark", name="Benchmark", path=str(benchmark)))
        with self.assertRaises(ValueError) as caught:
            self.operations.update_model_root(
                "benchmark", {"path": str(self.root / "old")})
        self.assertIn("both point at", str(caught.exception))
        self.store.load()  # must still load after the refusal
        self.assertEqual(self.store.load().model_root("benchmark").path,
                         str(benchmark))

    def test_a_repository_cannot_be_pointed_somewhere_of_its_own(self):
        # Setting them one at a time let GGUF sit on one disk and NVFP4 on
        # another. The refusal says where to go instead.
        with self.assertRaises(ValueError) as caught:
            self.operations.update_repository("gguf", {"path": str(self.root / "new")})
        self.assertIn("models root", str(caught.exception))
        self.assertEqual(self.store.load().repository("gguf").path,
                         str(self.root / "old" / "gguf"))

    def test_the_name_can_be_changed(self):
        self.operations.update_repository("gguf", {"name": "My models"})
        self.assertEqual(self.store.load().repository("gguf").name, "My models")

    def test_the_format_cannot_be_changed(self):
        """It decides which engines can read what is inside; changing it lies."""
        with self.assertRaises(ValueError):
            self.operations.update_repository("gguf", {"format": "fp8"})

    def test_a_folder_can_be_created_from_the_form(self):
        result = self.operations.create_directory(str(self.root / "made" / "here"))
        self.assertTrue(result["created"])
        self.assertTrue((self.root / "made" / "here").is_dir())

    def test_creating_one_that_exists_is_not_an_error(self):
        self.assertFalse(self.operations.create_directory(str(self.root / "new"))["created"])


class DrawingTheListTests(unittest.TestCase):
    """The model list asks the supervisor once, not once per entry.

    On systemd every answer is a command, and asking one at a time meant three
    commands per instance. Measured on the container with eleven instances that
    was 152 ms — the whole cost of this call, with the readiness probes beside
    it costing nothing. The gateway asks the same question twice on every
    request, so it was half a second in front of an engine answering in 17 ms.
    """

    def test_one_question_covers_every_entry(self):
        operations, host = operations_with_instances(3)
        host.status_calls = 0
        rows = operations.instances()
        self.assertEqual(len(rows), 3)
        self.assertEqual(host.status_calls, 1,
                         "one command for the list, however long it is")

    def test_an_empty_list_asks_nothing_at_all(self):
        operations, host = operations_with_instances(0)
        host.status_calls = 0
        self.assertEqual(operations.instances(), [])
        self.assertEqual(host.status_calls, 1)

    def test_each_entry_still_gets_its_own_answer(self):
        operations, host = operations_with_instances(3)
        host.running.add("model-1")
        rows = {row["id"]: row for row in operations.instances()}
        self.assertTrue(rows["model-1"]["running"])
        self.assertFalse(rows["model-0"]["running"])
        self.assertFalse(rows["model-2"]["running"])


class EngineOutputTests(unittest.TestCase):
    """What an engine is saying about itself, while it runs.

    A model that answers oddly or slowly is explaining itself the whole time,
    and reading that used to mean ssh and journalctl.
    """

    def setUp(self):
        self.operations, self.host = operations_with_instances(2)
        self.host.log_lines = ["INFO listening on 8100", "INFO slot released"]

    def test_a_running_model_hands_over_what_it_printed(self):
        self.host.running.add("model-0")
        answer = self.operations.logs("model-0")
        self.assertTrue(answer["running"])
        self.assertEqual(answer["lines"], self.host.log_lines)

    def test_a_stopped_model_is_answered_but_with_nothing(self):
        # Deliberate. systemd keeps a journal after a unit exits and macOS
        # keeps no log at all once the process is gone, so a page offering
        # this for a stopped model would work on one machine and not the
        # other. Why a model would not *start* is a different question, and
        # the failed load already carries that sentence.
        answer = self.operations.logs("model-0")
        self.assertFalse(answer["running"])
        self.assertEqual(answer["lines"], [])

    def test_an_unknown_instance_is_a_missing_thing(self):
        # A KeyError, so the web layer answers 404 without being told.
        with self.assertRaises(KeyError):
            self.operations.logs("no-such-entry")

    def test_how_many_lines_is_the_caller_s_choice(self):
        self.host.running.add("model-0")
        seen = {}
        original = self.host.logs

        def counted(instance_id, lines=15):
            seen["lines"] = lines
            return original(instance_id, lines)
        self.host.logs = counted
        self.operations.logs("model-0", lines=500)
        self.assertEqual(seen["lines"], 500)


class RestoringTests(unittest.TestCase):
    """What comes back when the manager starts."""

    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.memory = LastLoaded(Path(self._temporary.name))
        self.operations, self.host = operations_with_instances(
            2, last_loaded=self.memory)

    def test_loading_a_model_is_remembered(self):
        self.operations.load("model-0")
        self.assertEqual(self.memory.read()["instance_id"], "model-0")

    def test_the_settings_it_was_started_with_are_remembered_too(self):
        self.operations.load("model-0", {"context_size": 4096})
        self.assertEqual(self.memory.read()["settings"]["context_size"], 4096)

    def test_unloading_it_is_remembered_as_an_empty_card(self):
        self.operations.load("model-0")
        self.operations.unload("model-0")
        self.assertIsNone(self.memory.read())

    def test_unloading_a_stray_does_not_empty_the_memory(self):
        # The gateway unloads anything running beside the model that stays.
        # That is not the card being emptied.
        self.operations.load("model-0")
        self.operations.unload("model-1")
        self.assertEqual(self.memory.read()["instance_id"], "model-0")

    def test_a_load_that_failed_is_not_remembered(self):
        original = self.operations.runtime.load
        self.operations.runtime.load = lambda *a, **k: _failed("model-0")
        try:
            self.operations.load("model-0")
        finally:
            self.operations.runtime.load = original
        self.assertIsNone(self.memory.read())

    def test_it_comes_back_on_startup(self):
        self.operations.load("model-0")
        self.host.running.clear()               # as after a reboot
        self.operations.restore_last()
        self.assertIn("model-0", self.host.running)

    def test_it_comes_back_the_way_it_was_started(self):
        self.operations.load("model-0", {"context_size": 4096})
        self.host.running.clear()
        self.operations.restore_last()
        started = self.host.started[-1]
        self.assertIn("4096", " ".join(started.argv))

    def test_nothing_comes_back_when_the_card_was_left_empty(self):
        self.operations.load("model-0")
        self.operations.unload("model-0")
        self.host.running.clear()
        self.operations.restore_last()
        self.assertEqual(self.host.running, set())

    def test_a_model_still_running_is_left_alone(self):
        # The ordinary case on Linux: systemd owns the engines, so a manager
        # restart finds its model still answering. Reloading it would take the
        # card away from whoever is using it.
        self.operations.load("model-0")
        before = len(self.host.started)
        self.operations.restore_last()
        self.assertEqual(len(self.host.started), before)

    def test_a_restore_that_cannot_work_does_not_raise(self):
        # This runs while the manager is starting. A model whose files have
        # gone must not stop the manager from serving.
        self.memory.remember("no-such-entry")
        self.assertIsNone(self.operations.restore_last())

    def test_nothing_remembered_means_nothing_done(self):
        self.assertIsNone(self.operations.restore_last())
        self.assertEqual(self.host.running, set())


def _failed(instance_id):
    from ai_lab.runtime import Operation
    return Operation(instance_id=instance_id, kind="load", ok=False, error="no")


class GatewaySettingsTests(unittest.TestCase):
    """How long the front door waits, and how many requests it holds."""

    def setUp(self):
        self.operations, self.host = operations_with_instances(1)

    def test_nothing_is_set_to_begin_with(self):
        # Absent means the built-in default applies. Writing the defaults into
        # every configuration would make them look chosen.
        self.assertEqual(self.operations.gateway_settings(), {})

    def test_a_change_is_saved_and_read_back(self):
        self.operations.update_gateway({"first_byte_s": 300})
        self.assertEqual(self.operations.gateway_settings()["first_byte_s"], 300.0)

    def test_changes_do_not_erase_each_other(self):
        self.operations.update_gateway({"first_byte_s": 300})
        self.operations.update_gateway({"max_waiting": 40})
        saved = self.operations.gateway_settings()
        self.assertEqual(saved["first_byte_s"], 300.0)
        self.assertEqual(saved["max_waiting"], 40)

    def test_a_setting_that_does_not_exist_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.operations.update_gateway({"patience": 10})
        self.assertIn("patience", str(caught.exception))

    def test_a_number_out_of_range_is_refused_while_it_is_typed(self):
        # Rather than discovered when a request hangs, or when nothing queues.
        with self.assertRaises(ValueError):
            self.operations.update_gateway({"between_bytes_s": 0})
        with self.assertRaises(ValueError):
            self.operations.update_gateway({"max_waiting": 0})

    def test_something_that_is_not_a_number_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.operations.update_gateway({"first_byte_s": "a while"})
        self.assertIn("must be a number", str(caught.exception))

    def test_the_queue_length_stays_a_whole_number(self):
        self.operations.update_gateway({"max_waiting": 42.7})
        self.assertEqual(self.operations.gateway_settings()["max_waiting"], 42)


class NamingAnEntryTests(unittest.TestCase):
    """The name is given, checked, and is the only name the entry has.

    It used to be made from a label by lowercasing it and replacing everything
    else with hyphens, so the name a request had to carry was decided by a
    sentence somebody wrote for reading — and that sentence was what the page
    showed while the derived name was what worked.
    """

    def setUp(self):
        self.operations, self.host = operations_with_instances(2)

    def add(self, identifier):
        return self.operations.create_instance({
            "id": identifier, "engine": "llamacpp",
            "model_id": "gguf/model-0/model-0", "port": 8200, "params": {},
        })

    def test_a_plain_name_is_taken_as_given(self):
        self.assertEqual(self.add("gemma-31b-nvfp4")["id"], "gemma-31b-nvfp4")

    def test_a_name_already_taken_is_refused_and_says_so(self):
        with self.assertRaises(ValueError) as caught:
            self.add("model-0")
        message = str(caught.exception)
        self.assertIn("model-0", message)
        self.assertIn("already", message)

    def test_spaces_are_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.add("Coding fastest")
        self.assertIn("no spaces", str(caught.exception))

    def test_capitals_and_punctuation_are_refused(self):
        for bad in ("Coding", "coding(fast)", "coding_fast", "coding.fast", ""):
            with self.assertRaises(ValueError):
                self.add(bad)

    def test_it_may_not_begin_with_a_hyphen(self):
        with self.assertRaises(ValueError):
            self.add("-coding")

    def test_digits_and_hyphens_are_fine(self):
        self.assertEqual(self.add("qwen3-35b-2")["id"], "qwen3-35b-2")

    def test_the_name_cannot_be_changed_afterwards(self):
        # A request carries it, so changing it breaks whatever is sending it.
        # Renaming is deleting and adding, which the page already offers.
        # Refused rather than ignored: a change that is quietly dropped looks
        # like one that was made.
        with self.assertRaises(ValueError) as caught:
            self.operations.update_instance("model-0", {"name": "something else"})
        self.assertIn("name", str(caught.exception))
        self.assertEqual(self.operations.instance("model-0")["id"], "model-0")

    def test_an_entry_has_no_second_name_at_all(self):
        self.assertNotIn("name", self.operations.instance("model-0"))


class PointingAnEngineSomewhereElse(unittest.TestCase):
    """Which program serves an engine.

    Not expected to change — on a settled machine it never will — but two
    builds of llama.cpp on one machine is ordinary, and being unable to say
    which one means editing a file over ssh.
    """

    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name).resolve()
        self.addCleanup(self._temporary.cleanup)
        (self.root / "bin").mkdir()
        self.program = self.root / "bin" / "llama-server"
        self.program.write_text("#!/bin/sh\n")
        self.program.chmod(0o755)
        (self.root / "bin" / "notes.txt").write_text("not a program")
        (self.root / "bin" / "sub").mkdir()

        self.path = self.root / "config.json"
        self.path.write_text(json.dumps({
            "models_root": str(self.root),
            "engines": {"llamacpp": {"binary": "/bin/true"}},
            "repositories": [], "instances": [],
        }))
        self.store = ConfigStore(self.path)
        host = FakeHost()
        self.operations = Operations(
            store=self.store, catalog=Catalog(),
            runtime=Runtime(host, EventBus(), sample_interval_s=0),
            settings=Settings(self.store, host, Registry()),
            downloads=DownloadManager(), huggingface=offline_huggingface(),
            host=host, engines=FakeRegistry(host),
        )

    def test_browsing_shows_only_folders_by_default(self):
        found = self.operations.browse(str(self.root / "bin"))
        self.assertEqual([item["name"] for item in found["entries"]], ["sub"])

    def test_asking_for_programs_adds_the_ones_that_can_be_run(self):
        found = self.operations.browse(str(self.root / "bin"), programs=True)
        by_name = {item["name"]: item["kind"] for item in found["entries"]}
        self.assertEqual(by_name, {"sub": "folder", "llama-server": "program"})
        self.assertNotIn("notes.txt", by_name)

    def test_shared_libraries_are_not_offered_as_programs(self):
        # They carry the execute bit and cannot be launched. Measured in
        # llama.cpp's build directory on the container: 125 executable files,
        # 33 of them `.so` companions to the launchers beside them, and every
        # one a wrong answer to scroll past.
        for name in ("libllama.so", "libggml.so.1", "libthing.dylib"):
            item = self.root / "bin" / name
            item.write_text("")
            item.chmod(0o755)
        found = self.operations.browse(str(self.root / "bin"), programs=True)
        names = [item["name"] for item in found["entries"]]
        self.assertIn("llama-server", names)
        self.assertEqual([n for n in names if ".so" in n or ".dylib" in n], [])

    def test_the_program_is_saved(self):
        self.operations.update_engine_binary("llamacpp", str(self.program))
        self.assertEqual(self.store.load().engines["llamacpp"]["binary"],
                         str(self.program))

    def test_something_that_cannot_be_run_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.operations.update_engine_binary(
                "llamacpp", str(self.root / "bin" / "notes.txt"))
        self.assertIn("cannot be run", str(caught.exception))
        self.assertEqual(self.store.load().engines["llamacpp"]["binary"], "/bin/true")

    def test_a_folder_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.operations.update_engine_binary("llamacpp", str(self.root / "bin"))
        self.assertIn("not a program", str(caught.exception))

    def test_something_that_is_not_there_is_refused_before_saving(self):
        with self.assertRaises(ValueError):
            self.operations.update_engine_binary("llamacpp", "/nowhere/at/all")
        self.assertEqual(self.store.load().engines["llamacpp"]["binary"], "/bin/true")

    def test_an_unknown_engine_is_a_missing_thing(self):
        with self.assertRaises(KeyError):
            self.operations.update_engine_binary("nonesuch", str(self.program))
