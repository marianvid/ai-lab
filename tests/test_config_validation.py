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
