import unittest

from ai_lab.runtime.diagnostics import RuntimeDiagnostics


class Host:
    def __init__(self, lines):
        self.lines = lines

    def logs(self, _instance_id, lines):
        return self.lines[-lines:]


class RuntimeDiagnosticsTests(unittest.TestCase):
    # The rules themselves are pinned in test_failure_reason; these read them through the host.
    def test_previous_run_is_excluded(self):
        reader = RuntimeDiagnostics(Host([
            'Started ai-lab-engine@coder.service',
            'ValueError: stale failure',
            'Started ai-lab-engine@coder.service',
            'RuntimeError: current failure',
        ]))
        self.assertIn('current failure', reader.why('coder'))
        self.assertNotIn('stale failure', reader.why('coder'))

    def test_no_log_says_why_diagnosis_is_unavailable(self):
        self.assertIn('could not be read', RuntimeDiagnostics(Host([])).why('coder'))
