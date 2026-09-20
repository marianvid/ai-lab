import base64
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ai_lab.media.jobs import MediaJobs
from ai_lab.types import Task


class Handler(BaseHTTPRequestHandler):
    requests = []
    block = None

    def do_POST(self):
        if self.path.endswith('/cancel'):
            self.send_response(200)
            self.end_headers()
            return
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.requests.append((self.path, body))
        if self.block is not None:
            self.block.wait(3)
        data = json.dumps({'data': [{'b64_wav': base64.b64encode(b'RIFF').decode()}]}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass


class Lease:
    def __init__(self, port):
        self.port = port
        self.model_name = 'worker-model'

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class Gateway:
    def __init__(self, port):
        self.port = port

    def resolve(self, model):
        if model != 'music':
            raise KeyError(model)
        return {'id': model}

    def acquire(self, model, shape):
        return Lease(self.port)

    def timeouts_for(self, task):
        return (5, 5)


class MediaJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        Handler.requests = []
        Handler.block = None
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.root = Path(self.temp.name)
        self.gateway = Gateway(self.server.server_port)
        self.jobs = MediaJobs(self.gateway, self.root)

    def wait_for(self, job_id, status):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = self.jobs.get(job_id)
            if job['status'] == status:
                return job
            time.sleep(.01)
        self.fail(f'job never reached {status}')

    def submit(self):
        return self.jobs.submit({'model': 'music',
            'task': Task.MUSIC_GENERATION.value,
            'input': {'prompt': 'quiet piano'}})

    def test_result_survives_restart_without_storing_the_prompt_in_metadata(self):
        job = self.submit()
        completed = self.wait_for(job['id'], 'succeeded')
        self.assertEqual(completed['result']['data'][0]['b64_wav'], 'UklGRg==')
        self.assertEqual(Handler.requests[0][1]['model'], 'worker-model')
        self.assertEqual(Handler.requests[0][0], '/v1/audio/music/generations')
        self.assertNotIn('prompt', json.dumps(self.jobs.list()))
        restored = MediaJobs(self.gateway, self.root)
        self.assertEqual(restored.get(job['id'])['status'], 'succeeded')
        self.assertEqual(restored.get(job['id'])['result'], completed['result'])

    def test_video_job_uses_the_video_shape(self):
        job = self.jobs.submit({'model': 'music',
            'task': Task.VIDEO_GENERATION.value,
            'input': {'prompt': 'slow camera'}})
        self.wait_for(job['id'], 'succeeded')
        self.assertEqual(Handler.requests[0][0], '/v1/videos/generations')

    def test_expired_result_and_metadata_are_removed(self):
        job = self.submit()
        self.wait_for(job['id'], 'succeeded')
        self.jobs.jobs[job['id']]['updated_at'] = 1
        self.jobs.store.save(self.jobs.jobs[job['id']])
        self.jobs.cleanup()
        with self.assertRaises(KeyError):
            self.jobs.get(job['id'])
        self.assertFalse((self.jobs.store.results_dir / (job['id'] + '.json')).exists())

    def test_queued_job_can_be_cancelled_before_it_runs(self):
        Handler.block = threading.Event()
        first = self.submit()
        self.wait_for(first['id'], 'running')
        second = self.submit()
        cancelled = self.jobs.cancel(second['id'])
        self.assertEqual(cancelled['status'], 'cancelled')
        Handler.block.set()
        self.wait_for(first['id'], 'succeeded')
        self.assertEqual(len(Handler.requests), 1)

    def test_restart_fails_unfinished_job(self):
        job = self.submit()
        self.wait_for(job['id'], 'succeeded')
        record = dict(self.jobs.jobs[job['id']])
        record['id'] = 'unfinished'
        record['status'] = 'running'
        self.jobs.store.save(record)
        restored = MediaJobs(self.gateway, self.root)
        self.assertEqual(restored.get('unfinished')['status'], 'failed')
        self.assertIn('restarted', restored.get('unfinished')['error'])

    def test_bad_task_or_model_is_refused_without_creating_a_job(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported media task'):
            self.jobs.submit({'model': 'music', 'task': 'unknown', 'input': {}})
        with self.assertRaises(KeyError):
            self.jobs.submit({'model': 'missing',
                'task': Task.MUSIC_GENERATION.value, 'input': {}})
        self.assertEqual(self.jobs.list(), [])
