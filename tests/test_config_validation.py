import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from ai_lab.config.store import Config, Instance, ModelRoot, Repository
from ai_lab.config.validation import Section, validate_configuration


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

    def test_invalid_media_retention_fails_at_startup(self):
        config = self.base()
        config.media = {"result_ttl_s": -1}
        with self.assertRaisesRegex(ValueError, "Media result_ttl_s"):
            validate_configuration(config, {'comfyui'}, sections={})

    def test_unknown_media_setting_is_refused(self):
        config = self.base()
        config.media = {"retention_days": 1}
        with self.assertRaisesRegex(ValueError, "Unknown media settings"):
            validate_configuration(config, {'comfyui'}, sections={})

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
        config.gateway = {}
        config.media = []
        with self.assertRaisesRegex(ValueError, "Media settings must be an object"):
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


OWNED = {"reservations": Section(_owned, ports=_owned_ports)}


class SectionTests(unittest.TestCase):
    def base(self):
        return _base_config()

    def test_a_section_s_own_parser_reports_its_mistake(self):
        config = self.base()
        config.reservations = {}
        with self.assertRaisesRegex(ValueError, "owned section must name a port"):
            validate_configuration(config, {'comfyui'}, sections=OWNED)

    def test_a_port_a_section_claims_may_not_be_taken(self):
        config = self.base()
        config.reservations = {"port": 8100}
        with self.assertRaisesRegex(ValueError, "owned thing: port 8100 is already assigned"):
            validate_configuration(config, {'comfyui'}, sections=OWNED)

    def test_an_empty_section_map_checks_only_the_configuration_s_own_parts(self):
        config = self.base()
        config.reservations = {}
        validate_configuration(config, {'comfyui'}, sections={})

    def test_the_sections_must_be_named(self):
        with self.assertRaises(TypeError):
            validate_configuration(self.base(), {'comfyui'})  # type: ignore[call-arg]


if __name__ == '__main__':
    unittest.main()
