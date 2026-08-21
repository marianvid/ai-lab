import json
import os
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
    (directory / "models").mkdir()
    for index in range(count):
        model = directory / "models" / f"model-{index}"
        model.mkdir()
        make_files(model, f"model-{index}.gguf", size=1024)
    config = {
        "title": "AI-Lab", "host": "127.0.0.1", "port": 8090,
        "engines": {"llamacpp": {"binary": "/bin/true"}},
        "repositories": [{"id": "gguf", "name": "GGUF",
                          "path": str(directory / "models"),
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
            "repositories": [
                {"id": "gguf", "name": "GGUF", "path": str(self.root / "gguf"), "format": "gguf"},
                {"id": "st", "name": "Safetensors", "path": str(self.root / "safetensors"),
                 "format": "safetensors"},
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
            "repositories": [{"id": "gguf", "name": "GGUF",
                              "path": str(self.root / "gguf"), "format": "gguf"}],
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
            "repositories": [{"id": "gguf", "name": "GGUF",
                              "path": str(self.root / "gguf"), "format": "gguf"}],
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
        self.assertIn("Coding", str(caught.exception))
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
        (self.root / "open").mkdir()
        (self.root / "locked").mkdir()
        (self.root / "locked").chmod(0o500)          # readable, not writable
        self.addCleanup((self.root / "locked").chmod, 0o700)
        self.path = self.root / "config.json"
        self.path.write_text(json.dumps({
            "repositories": [
                {"id": "open", "name": "Open", "path": str(self.root / "open"),
                 "format": "gguf"},
                {"id": "locked", "name": "Locked", "path": str(self.root / "locked"),
                 "format": "gguf"},
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
        (self.root / "new").mkdir()
        self.path = self.root / "config.json"
        self.path.write_text(json.dumps({
            "repositories": [{"id": "gguf", "name": "GGUF",
                              "path": str(self.root / "old"), "format": "gguf"}],
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

    def test_the_path_can_be_changed(self):
        self.operations.update_repository("gguf", {"path": str(self.root / "new")})
        self.assertEqual(self.store.load().repository("gguf").path, str(self.root / "new"))

    def test_a_path_that_does_not_exist_is_refused_before_saving(self):
        """Saying so while the field is still on screen beats a broken screen."""
        with self.assertRaises(ValueError) as caught:
            self.operations.update_repository("gguf", {"path": "/nowhere/at/all"})
        self.assertIn("not a directory", str(caught.exception))
        self.assertEqual(self.store.load().repository("gguf").path, str(self.root / "old"))

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
