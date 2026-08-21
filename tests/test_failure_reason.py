"""Telling somebody why a model would not load.

The engine knows exactly why it died — a context that will not fit, a missing
package, a corrupt file — and that sentence is worth far more than "the process
exited". These tests run against journal output captured from the real
container on 21 August, when two loads failed for two different reasons and the
interface reported neither.

The fixtures are not trimmed to make this easy. They are what
`journalctl -n 300` actually returned, noise and all, because every one of the
three faults below was invisible against a tidier example.
"""

import unittest
from pathlib import Path

from ai_lab.runtime import Runtime

JOURNALS = Path(__file__).resolve().parent / "support" / "journals"


def journal(name):
    return (JOURNALS / f"{name}.txt").read_text().splitlines()


class RealFailuresTests(unittest.TestCase):
    def cause(self, name):
        return Runtime._cause(journal(name))

    def test_a_context_that_will_not_fit_says_so(self):
        # What actually happened: 128k was asked for on a 32 GB card. vLLM
        # worked out the largest that would fit and said so, which is the
        # number somebody needs. The interface showed "Engine core
        # initialization failed. See root cause above."
        cause = self.cause("kv-cache-too-small")
        self.assertIn("KV cache", cause)
        self.assertIn("131072", cause)

    def test_it_carries_the_number_that_would_have_worked(self):
        self.assertIn("117776", self.cause("kv-cache-too-small"))

    def test_a_missing_package_names_the_package(self):
        cause = self.cause("flashinfer-missing")
        self.assertIn("FlashInfer", cause)
        self.assertIn("install", cause.lower())

    def test_the_summary_that_explains_nothing_is_never_the_answer(self):
        # A traceback ends with a line that says something failed and nothing
        # about what. vLLM's reads "See root cause above" — the cause is above,
        # and taking the last error line reported the one line of no use.
        for name in ("kv-cache-too-small", "flashinfer-missing"):
            self.assertNotIn("See root cause above", self.cause(name))
            self.assertNotIn("initialization failed", self.cause(name))

    def test_no_traceback_frames(self):
        # "File "/opt/.../core.py", line 296, in _initialize_kv_caches" is
        # where it happened, not what happened.
        for name in ("kv-cache-too-small", "flashinfer-missing"):
            self.assertFalse(self.cause(name).startswith("File "))

    def test_the_logging_prefix_is_stripped(self):
        # vLLM prefixes every line with the process and its logger:
        # "(EngineCore pid=829699) ERROR 08-21 20:42:04 [core.py:1346] ".
        # None of it is part of the sentence.
        for name in ("kv-cache-too-small", "flashinfer-missing"):
            cause = self.cause(name)
            self.assertFalse(cause.startswith("("))
            self.assertNotIn("pid=", cause)
            self.assertNotIn("ERROR", cause)

    def test_an_exception_from_the_previous_run_is_not_reported(self):
        # Reading far enough back to find the cause also reaches over the end
        # of the run before it. The kv-cache journal contains a request that
        # failed a minute earlier with a different context limit, and that one
        # reads just as convincingly.
        cause = self.cause("kv-cache-too-small")
        self.assertNotIn("32768", cause,
                         "that limit belongs to the previous run")


class HeuristicTests(unittest.TestCase):
    """The rules, stated on small examples so a failure says which one broke."""

    def test_the_first_exception_wins_not_the_last(self):
        lines = [
            "Started ai-lab-engine@x.service - AI-Lab inference instance x.",
            "ValueError: the real reason, which came first",
            "Traceback (most recent call last):",
            "RuntimeError: Engine core initialization failed. See root cause above.",
        ]
        self.assertIn("the real reason", Runtime._cause(lines))

    def test_lines_before_this_run_are_ignored(self):
        lines = [
            "ValueError: something from last time",
            "Started ai-lab-engine@x.service - AI-Lab inference instance x.",
            "RuntimeError: what went wrong now",
        ]
        self.assertIn("what went wrong now", Runtime._cause(lines))

    def test_everything_is_searched_when_there_is_no_start_line(self):
        # A truncated log, or a host that writes no such line. Better a cause
        # that might be stale than none at all.
        self.assertIn("something", Runtime._cause(["ValueError: something"]))

    def test_a_death_with_no_exception_still_says_what_it_can(self):
        lines = [
            "Started ai-lab-engine@x.service - AI-Lab inference instance x.",
            "error while loading shared libraries: libcuda.so.1",
        ]
        self.assertIn("libcuda", Runtime._cause(lines))

    def test_silence_is_reported_as_silence(self):
        lines = ["Started ai-lab-engine@x.service - AI-Lab inference instance x.",
                 "loading weights", "reading tensors"]
        self.assertEqual(Runtime._cause(lines), "")


if __name__ == "__main__":
    unittest.main()
