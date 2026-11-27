"""The `limits` section: this machine's waits and first port, read once and handed on."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.api.passthrough import Waits
from ai_lab.builds.core import Builds
from ai_lab.builds.gitapps import GitApplicationInstall, GitCommands
from ai_lab.builds.installs import Installs
from ai_lab.builds.package_install import PackageInstall
from ai_lab.clients.caps import ClientCaps, OverClientCapError
from ai_lab.config.policy import (BuildLimits, ClientCapLimits, EmbeddingsLimits, GatewayPolicy,
                                  MachineLimits)
from ai_lab.config.store import Config, ConfigStore
from ai_lab.config.validation import validate_configuration
from ai_lab.cpu_models.lane import CpuLane
from ai_lab.events import EventBus
from ai_lab.gateway.limits import GatewayLimits, Patience

TEXT = "/v1/chat/completions"


class TheSectionTests(unittest.TestCase):
    def test_a_machine_without_the_section_keeps_the_values_it_always_had(self):
        limits = MachineLimits.from_mapping(None)
        self.assertEqual((limits.runtime.start_s, limits.runtime.load_s, limits.runtime.unload_s),
                         (60, 900, 120))
        self.assertEqual((limits.builds.install_s, limits.builds.build_s,
                          limits.builds.check_every_s, limits.builds.builds_first_check_s,
                          limits.builds.installs_first_check_s, limits.builds.system_path),
                         (3600, 3600, 3600, 20, 25, "/usr/local/bin:/usr/bin:/bin"))
        self.assertEqual((limits.embeddings.start_s, limits.embeddings.stop_s,
                          limits.embeddings.drain_s, limits.embeddings.hold_s,
                          limits.embeddings.call_s), (120, 30, 60, 900, 600))
        self.assertEqual((limits.clients.chars_per_token, limits.clients.retry_after_s), (3.0, 5))
        self.assertEqual(limits.first_port, 8080)

    def test_a_value_set_is_read_and_the_others_keep_their_default(self):
        limits = MachineLimits.from_mapping({"runtime": {"load_s": 1800},
                                             "builds": {"system_path": "/opt/bin:/usr/bin"},
                                             "first_port": 9000})
        self.assertEqual(limits.runtime.load_s, 1800)
        self.assertEqual(limits.runtime.start_s, 60)
        self.assertEqual(limits.builds.system_path, "/opt/bin:/usr/bin")
        self.assertEqual(limits.first_port, 9000)

    def test_a_wrong_value_is_refused_with_its_name(self):
        wrong = {
            "Limits must be an object": [1],
            "Limits runtime must be an object": {"runtime": 5},
            "Limits runtime load_s": {"runtime": {"load_s": 0}},
            "Limits builds install_s": {"builds": {"install_s": "long"}},
            "Limits builds system_path": {"builds": {"system_path": "bin:/usr/bin"}},
            "Limits embeddings call_s": {"embeddings": {"call_s": -1}},
            "Limits clients retry_after_s": {"clients": {"retry_after_s": 2.5}},
            "Limits first_port": {"first_port": 70000},
        }
        for message, raw in wrong.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                MachineLimits.from_mapping(raw)  # type: ignore[arg-type]

    def test_the_configuration_keeps_the_section_and_its_check_reports_it(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"limits": {"runtime": {"load_s": 1800}}}))
            store = ConfigStore(path)
            self.assertEqual(store.load().machine_limits.runtime.load_s, 1800)
            with store.mutate() as config:
                config.title = "Edited"
            self.assertEqual(json.loads(path.read_text())["limits"],
                             {"runtime": {"load_s": 1800}})
        config = Config(limits={"runtime": {"load_s": 0}})
        with self.assertRaisesRegex(ValueError, "Limits runtime load_s"):
            validate_configuration(config, set(), sections={})


class OneOwnerForTheGatewayDefaultsTests(unittest.TestCase):
    def test_every_holder_of_a_gateway_default_means_the_same_number(self):
        policy, limits, patience = GatewayPolicy(), GatewayLimits(), Patience()
        answer = Waits()
        self.assertEqual((limits.first_byte_s, limits.between_bytes_s, limits.max_waiting),
                         (policy.first_byte_s, policy.between_bytes_s, policy.max_waiting))
        self.assertEqual((patience.stall_s, patience.safety_s, patience.default_max_tokens),
                         (policy.stall_s, policy.safety_s, policy.default_max_tokens))
        self.assertEqual((answer.first_byte_s, answer.between_bytes_s, answer.stall_s,
                          answer.safety_s),
                         (policy.first_byte_s, policy.between_bytes_s, policy.stall_s,
                          policy.safety_s))
        self.assertEqual(GatewayPolicy.from_mapping({}), policy)


class TheLimitsReachTheirUsersTests(unittest.TestCase):
    def setUp(self):
        self._folder = TemporaryDirectory()
        self.addCleanup(self._folder.cleanup)
        self.folder = Path(self._folder.name)

    def test_a_source_build_compiles_under_the_machines_limit(self):
        (self.folder / ".git").mkdir()
        builds = Builds({"llamacpp": {"source": {"path": str(self.folder)}}}, EventBus(),
                        BuildLimits(build_s=7200))
        self.assertEqual(builds.get("llamacpp").build_timeout_s, 7200)

    def test_a_package_install_uses_the_machines_limit_and_search_path(self):
        installs = Installs({"vllm": {"binary": f"{self.folder}/current/bin/vllm",
                                      "source": {"package": "vllm"}}}, EventBus(),
                            BuildLimits(install_s=5400, system_path="/opt/bin:/usr/bin"))
        install = installs.get("vllm")
        assert isinstance(install, PackageInstall)
        self.assertEqual((install.install_timeout_s, install.system_path),
                         (5400, "/opt/bin:/usr/bin"))

    def test_a_git_application_asks_upstream_with_the_machines_search_path(self):
        asked = []

        def remote_commit(repository, ref, **options):
            asked.append((repository, ref, options))
            return "b" * 40
        install = GitApplicationInstall(
            "comfyui", {"source": {"root": str(self.folder), "repository": "upstream"}},
            EventBus(), GitCommands(remote_commit=remote_commit),
            BuildLimits(system_path="/opt/bin"))
        install.check()
        self.assertEqual(asked, [("upstream", "HEAD", {"system_path": "/opt/bin"})])

    def test_a_client_over_its_cap_is_told_the_machines_wait(self):
        caps = ClientCaps(self.folder / "caps.json", ClientCapLimits(retry_after_s=9))
        caps.set("c", {"max_calls_at_once": 1})
        caps.admit("c", TEXT, {"messages": []}, 10)
        with self.assertRaises(OverClientCapError) as refused:
            caps.admit("c", TEXT, {"messages": []}, 10)
        self.assertEqual(refused.exception.headers, {"Retry-After": "9"})

    def test_a_prompt_is_counted_with_the_machines_characters_per_token(self):
        body = {"messages": [{"role": "user", "content": "x" * 3000}]}
        counted = []
        for chars in (3.0, 6.0):
            caps = ClientCaps(self.folder / f"caps-{chars}.json",
                              ClientCapLimits(chars_per_token=chars))
            caps.set("c", {"max_tokens_in_flight": 100_000})
            give_back = caps.admit("c", TEXT, body, 10)
            counted.append(caps.view("c")["in_flight"]["tokens"])
            give_back()
        self.assertGreater(counted[0], counted[1] * 1.9)

    def test_the_embeddings_lane_hands_its_limits_to_every_model(self):
        raw = {"binary": "/bin/llama-server", "models": {
            "bge": {"path": "/models/bge.gguf", "port": 8300}}}
        limits = EmbeddingsLimits(start_s=5, drain_s=7)
        lane = CpuLane.from_config(raw, self.folder, limits=limits)
        placement = lane.placements["bge"]
        self.assertIs(placement.limits, limits)
        self.assertIs(placement.cpu.limits, limits)
        self.assertEqual(lane.config.models["bge"].limits, limits)


if __name__ == "__main__":
    unittest.main()
