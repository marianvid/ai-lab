"""The Higgs playground page on a Mac, answered by the in-process model.

A stand-in model records what it was asked and answers with a short WAV, so
these tests need neither the weights nor torch.
"""
import base64
import importlib.util
import io
import json
import unittest
import urllib.error
import urllib.request
import wave

from ai_lab.speech import higgs_playground
from ai_lab.speech.higgs_local_backend import (
    HiggsLocalBackend, sampling_settings)


def tiny_wav(rate=24000, frames=240):
    out = io.BytesIO()
    with wave.open(out, 'wb') as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(b'\x01\x00' * frames)
    return out.getvalue()


class FakeBackend:
    def __init__(self):
        self.calls = []

    def generate(self, body, sampling=None):
        self.calls.append((body, sampling))
        return {'sample_rate': 24000, 'data': [{
            'mime_type': 'audio/wav',
            'b64_wav': base64.b64encode(tiny_wav()).decode()}]}


def form(fields, files=None):
    boundary = 'xYzBoundary'
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; '
                     f'name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name, data in (files or {}).items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; '
                     f'name="{name}"; filename="ref.wav"\r\n'
                     'Content-Type: audio/wav\r\n\r\n'.encode() + data + b'\r\n')
    parts.append(f'--{boundary}--\r\n'.encode())
    return b''.join(parts), f'multipart/form-data; boundary={boundary}'


class PlaygroundTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.server = higgs_playground.serve(self.backend, 0)
        self.base = f'http://127.0.0.1:{self.server.server_address[1]}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def post(self, path, fields, files=None):
        data, kind = form(fields, files)
        request = urllib.request.Request(self.base + path, data=data,
                                         headers={'Content-Type': kind})
        return urllib.request.urlopen(request, timeout=5)

    def test_serves_the_upstream_page_and_its_files(self):
        with urllib.request.urlopen(self.base + '/', timeout=5) as page:
            html = page.read().decode()
        self.assertIn('/static/app.js?v=', html)
        with urllib.request.urlopen(self.base + '/static/app.js', timeout=5) as js:
            self.assertIn(b'/api/synthesize', js.read())
        with urllib.request.urlopen(self.base + '/healthz', timeout=5) as health:
            self.assertEqual(json.load(health)['backend'], 'ok')

    def test_refuses_files_outside_the_page(self):
        for path in ('/static/../app.py', '/static/index.py', '/etc/passwd'):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(self.base + path, timeout=5)
            self.assertEqual(caught.exception.code, 404)

    def test_form_becomes_a_contract_request_with_its_own_knobs(self):
        clip = tiny_wav(16000)
        with self.post('/api/synthesize', {
                'text': ' Bună ziua ', 'ref_text': 'salut', 'seed': '7',
                'temperature': '0.8', 'top_p': '0.9', 'top_k': '40',
                'max_new_tokens': '512', 'ref_audio_url': ''},
                {'ref_audio': clip}) as answer:
            self.assertEqual(answer.headers['Content-Type'], 'audio/wav')
            self.assertTrue(answer.read().startswith(b'RIFF'))
        body, sampling = self.backend.calls[0]
        self.assertEqual(body['text'], 'Bună ziua')
        self.assertEqual(body['seed'], 7)
        self.assertEqual(base64.b64decode(body['reference_audio']), clip)
        self.assertEqual(body['reference_text'], 'salut')
        self.assertEqual(sampling, {'temperature': 0.8, 'top_p': 0.9,
                                    'top_k': 40, 'max_frames': 512})

    def test_stream_answers_bare_samples_with_their_format(self):
        with self.post('/api/synthesize/stream', {'text': 'Hi'}) as answer:
            self.assertEqual(answer.headers['X-Sample-Rate'], '24000')
            self.assertEqual(answer.headers['X-Bit-Depth'], '16')
            self.assertEqual(answer.headers['X-Channels'], '1')
            self.assertEqual(len(answer.read()), 480)

    def test_reference_by_url_and_bad_numbers_are_refused(self):
        for fields in ({'text': 'Hi', 'ref_audio_url': 'http://x/a.wav'},
                       {'text': 'Hi', 'top_k': 'many'},
                       {'text': 'Hi', 'temperature': '9'},
                       {'text': '   '}):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.post('/api/synthesize', fields)
            self.assertEqual(caught.exception.code, 400)
        self.assertEqual(self.backend.calls, [])


class SamplingTests(unittest.TestCase):
    def test_defaults_and_limits(self):
        self.assertEqual(sampling_settings(), {
            'temperature': 1.0, 'top_p': 0.95, 'top_k': 50, 'max_frames': 2048})
        self.assertEqual(sampling_settings(max_new_tokens=8192)['max_frames'], 2048)
        for bad in ({'temperature': 0}, {'top_p': 0}, {'top_p': 1.5},
                    {'top_k': 0}, {'top_k': 2.5}, {'max_new_tokens': 0}):
            with self.assertRaises(ValueError):
                sampling_settings(**bad)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'torch is not installed')
class PerLineSamplingTests(unittest.TestCase):
    def test_each_line_in_a_group_keeps_its_own_top_k(self):
        import torch

        logits = torch.arange(10, dtype=torch.float32).repeat(2, 3, 1)  # 2 lines, 3 codebooks
        probs = HiggsLocalBackend._probabilities(
            logits, torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0]),
            torch.tensor([1, 3]))
        self.assertEqual(int((probs[0] > 0).sum(-1).max()), 1)
        self.assertEqual(int((probs[1] > 0).sum(-1).min()), 3)
        self.assertTrue(torch.allclose(probs.sum(-1), torch.ones(2, 3)))

    def test_defaults_match_the_old_group_wide_sampler(self):
        import torch

        torch.manual_seed(0)
        logits = torch.randn(2, 4, 100)
        probs = HiggsLocalBackend._probabilities(
            logits, torch.tensor([1.0, 1.0]), torch.tensor([0.95, 0.95]),
            torch.tensor([50, 50]))
        kth = logits.topk(50, dim=-1).values[..., -1:]
        old = torch.where(logits < kth, float('-inf'), logits)
        ordered, order = torch.sort(old, descending=True, dim=-1)
        remove = ordered.softmax(-1).cumsum(-1) > 0.95
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scattered = torch.zeros_like(remove)
        scattered.scatter_(-1, order, remove)
        expected = torch.where(scattered, float('-inf'), old).softmax(-1)
        self.assertTrue(torch.allclose(probs, expected))


if __name__ == '__main__':
    unittest.main()
