import unittest

from ai_lab.runtime_diagnostics import RuntimeDiagnostics


class Host:
    def __init__(self, lines):
        self.lines = lines

    def logs(self, _instance_id, lines):
        return self.lines[-lines:]


class RuntimeDiagnosticsTests(unittest.TestCase):
    def test_first_real_exception_wins_over_later_generic_summary(self):
        reader = RuntimeDiagnostics(Host([
            'Started ai-lab-engine@coder.service',
            '(EngineCore pid=123) ERROR 08-21 20:42:04 [core.py:1] '
            'ValueError: context needs more cache',
            'RuntimeError: Engine core initialization failed. See root cause above.',
        ]))
        self.assertIn('ValueError: context needs more cache', reader.why('coder'))

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
