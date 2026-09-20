import unittest

from ai_lab.engines.comfy_video import ComfyVideoEngine
from ai_lab.types import Format, ModelFile, ModelSet, Task


class ComfyVideoTests(unittest.TestCase):
    def test_two_configured_stacks_share_an_engine(self):
        options = {name: {'workflow': f'/private/{name}.json',
                          'model_root': f'/models/video/{name}',
                          'component_subdirs': ['diffusion_models', 'vae'],
                          'clip_seconds': 6.0, 'memory_reservation_mb': 30000}
                   for name in ('future-h3', 'future-ltx')}
        engine = ComfyVideoEngine(binary='/comfy/python',
                                  comfyui='/comfy/main.py', model_options=options)
        for name in options:
            model = ModelSet(id='video/' + name, name=name,
                             format=Format.COMFYUI, task=Task.VIDEO_GENERATION,
                             entrypoint=f'/models/video/{name}/diffusion_models/model.safetensors',
                             files=(ModelFile('model.safetensors', 1024),))
            plan = engine.plan(model, 8127, {})
            self.assertEqual(plan.argv[plan.argv.index('--workflow') + 1],
                             f'/private/{name}.json')
            self.assertIn('/models/video/' + name + '/vae', plan.argv)
            self.assertEqual(engine.needs_mb(model, {}, 32623), 30000)
            self.assertTrue(plan.splits_across_cpu)
            self.assertEqual(engine.video_form(name)['clip_seconds'], 6.0)


if __name__ == '__main__':
    unittest.main()
