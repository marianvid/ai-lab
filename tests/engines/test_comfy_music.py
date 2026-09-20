import unittest

from ai_lab.engines.comfy_music import ComfyMusicEngine
from ai_lab.types import Format, ModelFile, ModelSet, Task


class ComfyMusicTests(unittest.TestCase):
    def test_plan_uses_configured_components_and_workflow(self):
        engine = ComfyMusicEngine(binary='/comfy/python', comfyui='/comfy/main.py',
                                  model_options={'future-music': {
            'workflow': '/private/music.json',
            'component_subdirs': ['diffusion_models', 'text_encoders', 'vae'],
            'memory_reservation_mb': 30000}})
        model = ModelSet(id='music/future-music', name='future-music',
                         format=Format.SAFETENSORS,
                         task=Task.MUSIC_GENERATION,
                         entrypoint='/models/future-music/diffusion_models',
                         files=(ModelFile('model.safetensors', 1024),))
        plan = engine.plan(model, 8125, {})
        self.assertEqual(plan.argv[plan.argv.index('--workflow') + 1],
                         '/private/music.json')
        paths = [plan.argv[i + 1] for i, value in enumerate(plan.argv)
                 if value == '--model-path']
        self.assertEqual(paths, [
            '/models/future-music/diffusion_models',
            '/models/future-music/text_encoders',
            '/models/future-music/vae'])
        self.assertEqual(engine.needs_mb(model, {}, 32623), 30000)


if __name__ == '__main__':
    unittest.main()
