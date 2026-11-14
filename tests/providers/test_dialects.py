"""The three CLIs, run for real against small stand-in scripts."""

import os
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.providers.dialects import (
    CLAUDE_NO_SELF_UPDATE,
    CLAUDE_NOT_THE_PLAN,
    environment,
    OPENCODE_NO_TOOLS,
    ClaudeCli,
    CodexCli,
    OpencodeCli,
    run,
    user_names,
)
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
FAKE_OPENCODE = """#!/bin/sh
prompt=$(cat)
echo "banner" >&2
echo "args=$* config=$OPENCODE_CONFIG_CONTENT prompt=$prompt"
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

    def test_opencode_denies_every_tool_and_answers_on_standard_output(self):
        vendor = self.vendor(self.script("opencode", FAKE_OPENCODE), dialect="opencode")
        outcome = run(OpencodeCli(), vendor, "zen/kimi-k2", "Hello", effort="high")
        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.answer, "args=run -m zen/kimi-k2 --variant high "
                                         f"config={OPENCODE_NO_TOOLS} prompt=Hello")
        call = OpencodeCli().invocation(self.vendor("/b", home="/srv/oc"), "m", "x", "",
                                        self.root)
        self.assertEqual(call.env["HOME"], "/srv/oc")
        self.assertNotIn("--variant", call.argv)

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

    def test_no_home_sets_only_the_no_self_update_switch(self):
        call = ClaudeCli().invocation(self.vendor("/b"), "opus", "x", "", self.root)
        self.assertEqual(call.env, CLAUDE_NO_SELF_UPDATE)
        self.assertNotIn("--effort", call.argv)
        self.assertIn("HOME", os.environ)

    def test_a_manager_started_without_a_user_name_still_passes_one(self):
        # The AI-Lab Menu app gives the manager only HOME and PATH; Claude Code
        # then finds no Keychain sign-in. The CLI must get USER and LOGNAME.
        self.assertEqual(user_names({"USER": "owner", "LOGNAME": "owner"}), {})
        self.assertEqual(user_names({"LOGNAME": "owner"}), {"USER": "owner"})
        filled = user_names({})
        self.assertEqual(set(filled), {"USER", "LOGNAME"})
        self.assertTrue(filled["USER"])
        script = self.script("who", "#!/bin/sh\ncat > /dev/null\necho \"user=$USER\"\n")
        saved = {key: os.environ.pop(key, None) for key in ("USER", "LOGNAME")}
        try:
            outcome = run(ClaudeCli(), self.vendor(script), "opus", "x")
        finally:
            os.environ.update({key: value for key, value in saved.items() if value})
        self.assertNotEqual(outcome.answer, "user=")

    def test_the_yearly_token_reaches_claude_and_an_api_key_does_not(self):
        token_file = self.root / "claude-token.env"
        token_file.write_text("CLAUDE_CODE_OAUTH_TOKEN=stand-in-token\n"
                              "ANTHROPIC_API_KEY=stand-in-key\n")
        token_file.chmod(0o600)
        script = self.script("claude", "#!/bin/sh\ncat > /dev/null\n"
                             "echo \"token=$CLAUDE_CODE_OAUTH_TOKEN key=$ANTHROPIC_API_KEY "
                             "auth=$ANTHROPIC_AUTH_TOKEN\"\n")
        vendor = Vendor(id="anthropic", dialect="claude", binary=script,
                        env_file=str(token_file))
        saved = {key: os.environ.get(key) for key in CLAUDE_NOT_THE_PLAN}
        os.environ.update(dict.fromkeys(CLAUDE_NOT_THE_PLAN, "left-by-accident"))
        try:
            outcome = run(ClaudeCli(), vendor, "opus", "x")
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.assertEqual(outcome.answer, "token=stand-in-token key= auth=")

    def test_a_token_written_later_counts_from_the_next_call(self):
        token_file = self.root / "claude-token.env"
        vendor = Vendor(id="anthropic", dialect="claude", binary="/b", env_file=str(token_file))
        call = ClaudeCli().invocation(vendor, "opus", "x", "", self.root)
        self.assertNotIn("CLAUDE_CODE_OAUTH_TOKEN", environment({"PATH": "/bin"}, call, vendor))
        token_file.write_text("CLAUDE_CODE_OAUTH_TOKEN=stand-in-token\n")
        self.assertEqual(environment({"PATH": "/bin"}, call, vendor)["CLAUDE_CODE_OAUTH_TOKEN"],
                         "stand-in-token")

    def test_other_clis_keep_the_whole_environment(self):
        call = CodexCli().invocation(self.vendor("/b", dialect="codex"), "m", "x", "", self.root)
        env = environment({"ANTHROPIC_API_KEY": "k", "PATH": "/bin"}, call,
                          self.vendor("/b", dialect="codex"))
        self.assertEqual(env["ANTHROPIC_API_KEY"], "k")


if __name__ == "__main__":
    unittest.main()
