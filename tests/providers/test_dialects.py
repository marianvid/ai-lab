"""The two CLIs, run for real against small stand-in scripts."""

import os
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.providers.dialects import ClaudeCli, CodexCli, run
from ai_lab.providers.settings import Vendor, VendorLimits

FAKE_CLAUDE = """#!/bin/sh
prompt=$(cat)
echo "args=$* home=$HOME prompt=$prompt"
"""
FAKE_CODEX = """#!/bin/sh
out=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-o" ]; then out=$2; shift; fi
  shift
done
cat > "$out"
printf '\\ncodex_home=%s' "$CODEX_HOME" >> "$out"
"""
FAILING = """#!/bin/sh
echo "Claude AI usage limit reached" >&2
exit 1
"""
SLOW = """#!/bin/sh
sleep 5
"""


class DialectTests(unittest.TestCase):
    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)

    def script(self, name, body):
        path = self.root / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return str(path)

    def vendor(self, binary, dialect="claude", home="", timeout_s=30.0):
        return Vendor(id="v", dialect=dialect, binary=binary, home=home,
                      limits=VendorLimits(timeout_s=timeout_s))

    def test_claude_reads_the_prompt_on_standard_input(self):
        vendor = self.vendor(self.script("claude", FAKE_CLAUDE), home="/srv/anthropic")
        outcome = run(ClaudeCli(), vendor, "sonnet", "What is 2+2?", effort="low")
        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.answer, "args=-p --model sonnet --effort low "
                                         "home=/srv/anthropic prompt=What is 2+2?")

    def test_codex_answers_in_the_file_it_was_told_to_write(self):
        vendor = self.vendor(self.script("codex", FAKE_CODEX), dialect="codex",
                             home="/srv/openai")
        outcome = run(CodexCli(), vendor, "gpt-5.6-terra", "Hello", effort="high")
        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.answer, "Hello\ncodex_home=/srv/openai")
        argv = CodexCli().invocation(vendor, "gpt-5.6-terra", "x", "high", self.root).argv
        self.assertIn('model_reasoning_effort="high"', argv)
        self.assertEqual(argv[-1], "-")

    def test_codex_with_no_file_has_no_answer(self):
        self.assertEqual(CodexCli().answer("printed", self.root), "")

    def test_a_failure_keeps_the_cli_words(self):
        outcome = run(ClaudeCli(), self.vendor(self.script("claude", FAILING)), "opus", "x")
        self.assertFalse(outcome.succeeded)
        self.assertIn("usage limit", outcome.errors)

    def test_a_call_that_outlasts_its_timeout_is_stopped(self):
        vendor = self.vendor(self.script("claude", SLOW), timeout_s=0.3)
        outcome = run(ClaudeCli(), vendor, "opus", "x")
        self.assertTrue(outcome.timed_out)
        self.assertFalse(outcome.succeeded)

    def test_a_missing_binary_cannot_start(self):
        outcome = run(ClaudeCli(), self.vendor(str(self.root / "absent")), "opus", "x")
        self.assertIn("cannot start the CLI", outcome.errors)

    def test_no_home_leaves_the_environment_alone(self):
        call = ClaudeCli().invocation(self.vendor("/b"), "opus", "x", "", self.root)
        self.assertEqual(call.env, {})
        self.assertNotIn("--effort", call.argv)
        self.assertIn("HOME", os.environ)


if __name__ == "__main__":
    unittest.main()
