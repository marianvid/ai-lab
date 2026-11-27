import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from ai_lab.config.store import Config, Instance, ModelRoot, Repository
from ai_lab.config.validation import Section, editor_port_problems, validate_configuration
from ai_lab.engines.registry import Registry


def _base_config():
    return Config(
        port=8090, models_root='/models',
        model_roots=[ModelRoot('core', 'Core', '/models')],
        repositories=[Repository('images', 'Images', 'comfyui',
                                 task='image-generation')],
        instances=[Instance('picture', 'comfyui', 'images/picture', 8100)],
    )


class ConfigurationValidationTests(unittest.TestCase):
    def base(self):
        return _base_config()

    def test_an_engine_page_must_land_on_a_free_port_that_exists(self):
        config = self.base()
        self.assertEqual(editor_port_problems(config, {"comfyui"}), [])
        config.instances.append(Instance('chat', 'comfyui', 'images/chat', 18100))
        with self.assertRaisesRegex(ValueError, "picture: its page would be on port 18100 .port "
                                                "8100 plus the editor offset 10000., which is "
                                                "chat's port"):
            validate_configuration(config, {'comfyui'}, sections={}, page_engines={'comfyui'})
        # Only the engines with a page of their own are checked.
        validate_configuration(config, {'comfyui'}, sections={})
        config.instances[1].port = 9000
        config.limits = {"editor_port_offset": 50000}
        config.instances[0].port = 18000
        self.assertEqual(editor_port_problems(config, {"comfyui"}),
                         ["picture: its page would be on port 68000 (port 18000 plus the editor "
                          "offset 50000), above 65535"])
        config.instances[0].port = 8100
        config.limits = {"editor_port_offset": 1000}
        config.port = 9100
        self.assertIn("the manager's port", editor_port_problems(config, {"comfyui"})[0])

    def test_saving_an_entry_whose_page_lands_on_a_taken_port_is_refused(self):
        from ai_lab.application.instance_config import _refuse_page_clashes
        config = self.base()
        config.instances.append(Instance('chat', 'llamacpp', 'gguf/chat', 18100))
        with self.assertRaisesRegex(ValueError, "which is chat's port"):
            _refuse_page_clashes(config, Registry())
        config.instances[1].port = 18101
        _refuse_page_clashes(config, Registry())

    def test_the_engines_with_a_page_of_their_own(self):
        self.assertEqual(Registry().page_engines(), {
            "comfyui", "comfy_music", "heartmula", "yue2", "qwentts", "higgs", "higgs_local",
            "voxcpm", "kokoro"})

    def test_gateway_policy_has_typed_task_waits(self):
        config = self.base()
        config.gateway = {"task_timeouts": {
            "music-generation": {"first_byte_s": 1800, "between_bytes_s": 60}}}
        validate_configuration(config, {'comfyui'}, sections={})
        policy = config.gateway_policy
        self.assertEqual(policy.task_timeouts["music-generation"].first_byte_s, 1800)
        self.assertEqual(policy.max_waiting, 150)

    def test_sections_must_be_objects_even_when_empty(self):
        config = self.base()
        config.gateway = []
        with self.assertRaisesRegex(ValueError, "Gateway settings must be an object"):
            validate_configuration(config, {'comfyui'}, sections={})

    def test_invalid_gateway_task_wait_fails_at_startup(self):
        config = self.base()
        config.gateway = {"task_timeouts": {
            "music-generation": {"first_byte_s": -1}}}
        with self.assertRaisesRegex(ValueError, "Gateway first_byte_s"):
            validate_configuration(config, {'comfyui'}, sections={})

    def test_unknown_gateway_fields_survive_a_config_save(self):
        from ai_lab.config.store import ConfigStore
        import json
        with TemporaryDirectory() as root:
            path = Path(root) / "config.json"
            path.write_text(json.dumps({"gateway": {
                "max_waiting": 12, "future_control": {"enabled": True}}}))
            store = ConfigStore(path)
            self.assertEqual(store.load().gateway_policy.max_waiting, 12)
            with store.mutate() as config:
                config.title = "Updated"
            self.assertEqual(json.loads(path.read_text())["gateway"]["future_control"],
                             {"enabled": True})

    def test_valid_image_profile(self):
        config = self.base()
        with TemporaryDirectory() as root:
            Path(root, 'generate.json').write_text('{}')
            config.images = {'workflow_root': root, 'profiles': {
                'picture': {'model': 'picture', 'task': 'generation',
                            'workflow': 'generate.json'}}}
            validate_configuration(config, {'comfyui'}, check_workflows=True, sections={})
        self.assertEqual(config.images['profiles']['picture']['workflow'], 'generate.json')

    def test_reports_bad_references_before_work_starts(self):
        config = self.base()
        config.instances.append(Instance('other', 'unknown', 'missing/model', 8100))
        config.images = {'profiles': {'broken': {
            'model': 'absent', 'workflow': '../outside.json'}}}
        with self.assertRaisesRegex(ValueError, 'unknown engine') as error:
            validate_configuration(config, {'comfyui'}, sections={})
        self.assertIn('port 8100 is already assigned', str(error.exception))
        self.assertIn('unknown instance absent', str(error.exception))

    def test_rejects_mismatched_and_missing_workflow(self):
        config = self.base()
        config.images = {'workflow_root': '/missing', 'profiles': {
            'wrong': {'model': 'picture', 'task': 'edit',
                      'workflow': 'missing.json'}}}
        with self.assertRaisesRegex(ValueError, 'expected image-edit') as error:
            validate_configuration(config, {'comfyui'}, check_workflows=True, sections={})
        self.assertIn('workflow file is absent', str(error.exception))



def _owned(raw):
    """A section another package would own: an object with a `port`."""
    if not isinstance(raw, dict) or "port" not in raw:
        raise ValueError("owned section must name a port")
    return raw["port"]


def _owned_ports(port: object) -> list[tuple[str, int]]:
    return [("owned thing", cast(int, port))]


OWNED = {"gpu_health": Section(_owned, ports=_owned_ports)}


class SectionTests(unittest.TestCase):
    def base(self):
        return _base_config()

    def test_a_section_s_own_parser_reports_its_mistake(self):
        config = self.base()
        config.gpu_health = {}
        with self.assertRaisesRegex(ValueError, "owned section must name a port"):
            validate_configuration(config, {'comfyui'}, sections=OWNED)

    def test_a_port_a_section_claims_may_not_be_taken(self):
        config = self.base()
        config.gpu_health = {"port": 8100}
        with self.assertRaisesRegex(ValueError, "owned thing: port 8100 is already assigned"):
            validate_configuration(config, {'comfyui'}, sections=OWNED)

    def test_an_empty_section_map_checks_only_the_configuration_s_own_parts(self):
        config = self.base()
        config.gpu_health = {'interval_s': 'not checked here'}
        # The card's health section belongs to `gpu_health`: with no parser
        # named for it, a value it would refuse passes the configuration's check.
        validate_configuration(config, {'comfyui'}, sections={})
        self.assertEqual(config.gpu_health, {'interval_s': 'not checked here'})

    def test_the_sections_must_be_named(self):
        with self.assertRaises(TypeError):
            validate_configuration(self.base(), {'comfyui'})  # type: ignore[call-arg]


if __name__ == '__main__':
    unittest.main()
