import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.config import Config, Instance, ModelRoot, Repository
from ai_lab.config_validation import validate_configuration


class ConfigurationValidationTests(unittest.TestCase):
    def base(self):
        return Config(
            port=8090, models_root='/models',
            model_roots=[ModelRoot('core', 'Core', '/models')],
            repositories=[Repository('images', 'Images', 'comfyui',
                                     task='image-generation')],
            instances=[Instance('picture', 'comfyui', 'images/picture', 8100)],
        )

    def test_invalid_media_retention_fails_at_startup(self):
        config = self.base()
        config.media = {"result_ttl_s": -1}
        with self.assertRaisesRegex(ValueError, "Media result_ttl_s"):
            validate_configuration(config, {'comfyui'})

    def test_unknown_media_setting_is_refused(self):
        config = self.base()
        config.media = {"retention_days": 1}
        with self.assertRaisesRegex(ValueError, "Unknown media settings"):
            validate_configuration(config, {'comfyui'})

    def test_gateway_policy_has_typed_task_waits(self):
        config = self.base()
        config.gateway = {"task_timeouts": {
            "music-generation": {"first_byte_s": 1800, "between_bytes_s": 60}}}
        validate_configuration(config, {'comfyui'})
        policy = config.gateway_policy
        self.assertEqual(policy.task_timeouts["music-generation"].first_byte_s, 1800)
        self.assertEqual(policy.max_waiting, 150)

    def test_sections_must_be_objects_even_when_empty(self):
        config = self.base()
        config.gateway = []
        with self.assertRaisesRegex(ValueError, "Gateway settings must be an object"):
            validate_configuration(config, {'comfyui'})
        config.gateway = {}
        config.media = []
        with self.assertRaisesRegex(ValueError, "Media settings must be an object"):
            validate_configuration(config, {'comfyui'})

    def test_invalid_gateway_task_wait_fails_at_startup(self):
        config = self.base()
        config.gateway = {"task_timeouts": {
            "music-generation": {"first_byte_s": -1}}}
        with self.assertRaisesRegex(ValueError, "Gateway first_byte_s"):
            validate_configuration(config, {'comfyui'})

    def test_unknown_gateway_fields_survive_a_config_save(self):
        from ai_lab.config import ConfigStore
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
            validate_configuration(config, {'comfyui'}, check_workflows=True)

    def test_reports_bad_references_before_work_starts(self):
        config = self.base()
        config.instances.append(Instance('other', 'unknown', 'missing/model', 8100))
        config.images = {'profiles': {'broken': {
            'model': 'absent', 'workflow': '../outside.json'}}}
        with self.assertRaisesRegex(ValueError, 'unknown engine') as error:
            validate_configuration(config, {'comfyui'})
        self.assertIn('port 8100 is already assigned', str(error.exception))
        self.assertIn('unknown instance absent', str(error.exception))

    def test_rejects_mismatched_and_missing_workflow(self):
        config = self.base()
        config.images = {'workflow_root': '/missing', 'profiles': {
            'wrong': {'model': 'picture', 'task': 'edit',
                      'workflow': 'missing.json'}}}
        with self.assertRaisesRegex(ValueError, 'expected image-edit') as error:
            validate_configuration(config, {'comfyui'}, check_workflows=True)
        self.assertIn('workflow file is absent', str(error.exception))


if __name__ == '__main__':
    unittest.main()
