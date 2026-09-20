"""Find a useful cause in an engine's noisy process log."""

from __future__ import annotations

import re

# How far back to read when a load fails. The sentence naming the cause sits a
# long way above the end of a Python traceback — measured on the container, 99
# lines for a context that would not fit and 163 for a missing package.
LOG_LINES_FOR_CAUSE = 300

# A line where something was actually raised: "ValueError: ...", "RuntimeError:
# ...", "torch.cuda.OutOfMemoryError: ...". The message after the colon is the
# part worth showing.
RAISED = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Exit)\s*:\s*\S")

# Lines that mention trouble and explain none of it.
NOISE = (
    "see root cause above",
    "traceback (most recent call last)",
    "for more info",
    "engine core initialization failed",
    "engine process failed to start",
    "see stack trace",
)


def _without_prefix(line: str) -> str:
    """Strip what the supervisor and the engine put in front of their output.

    systemd prefixes nothing, but vLLM prefixes every line with the process it
    came from — `(EngineCore pid=829699) ` — and its logger adds a level and a
    source: `ERROR 08-21 20:42:04 [core.py:1346] `. Neither is part of the
    sentence, and both stop it being recognised as a raised exception.
    """
    text = line.strip()
    if text.startswith("("):
        closing = text.find(") ")
        if closing != -1:
            text = text[closing + 2:].strip()
    text = LOG_PREFIX.sub("", text, count=1).strip()
    return text


LOG_PREFIX = re.compile(
    r"^(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\s+[\d:\- ]*\[[^\]]+\]\s*")


def _is_noise(text: str) -> bool:
    lowered = text.lower()
    return (not text
            or ".service:" in lowered
            or lowered.startswith(("file \"", "raise ", "self.", "return ",
                                   "await ", "with ", "yield "))
            or any(marker in lowered for marker in NOISE))


class RuntimeDiagnostics:
    def __init__(self, host) -> None:
        self.host = host

    def why(self, instance_id: str) -> str:
        """Explain a death using the engine's own words.

        The engine knows exactly why it died — not enough memory, a context
        that will not fit, a missing package — and that sentence is worth far
        more than "the process exited". Getting at it takes some care, because
        an engine that dies inside Python prints a great deal around it.

        **The first exception, not the last.** A traceback ends with a summary
        that says something failed and nothing about what. vLLM's literally
        reads "Engine core initialization failed. See root cause above." The
        cause is above, and taking the last line reported the one line that
        was no use.

        **Far enough back.** Measured on the container: the sentence naming a
        context that would not fit sat 99 lines above the end, and a missing
        package 163. Reading forty found neither.
        """
        try:
            lines = self.host.logs(instance_id, lines=LOG_LINES_FOR_CAUSE)
        except Exception:
            lines = []
        if not lines:
            return ("The engine stopped while loading, and its output could not "
                    "be read. On Linux the manager needs to be in the "
                    "systemd-journal group to see it.")
        detail = self.cause(lines)
        return f"The engine stopped while loading: {detail}" if detail else (
            "The engine stopped while loading, and said nothing about why. "
            "Its full output is in the journal.")

    @staticmethod
    def this_run(lines: list[str]) -> list[str]:
        """Only what this attempt printed.

        Reading three hundred lines back reaches over the end of the previous
        run, and an exception from *that* one reads exactly as convincingly.
        It happened while this was being written: the message named a context
        limit from a request answered a minute before the load even started.

        systemd writes one line when it starts a unit, which is the boundary.
        Without it — a log truncated, or a host that writes no such line —
        everything is searched, which is what happened before.
        """
        for index in range(len(lines) - 1, -1, -1):
            text = lines[index].strip()
            if text.startswith("Started ") and ".service" in text:
                return lines[index + 1:]
        return lines

    @classmethod
    def cause(cls, lines: list[str]) -> str:
        """The one sentence out of a few hundred that says what went wrong."""
        lines = cls.this_run(lines)
        candidates = []
        for line in lines:
            text = _without_prefix(line)
            if not text or _is_noise(text):
                continue
            if RAISED.match(text):
                candidates.append(text)
        if candidates:
            return candidates[0]
        # Nothing that looks like a raised exception. Fall back to the last
        # line that at least mentions trouble, which is what this used to do
        # for every case.
        mentions = [_without_prefix(line) for line in lines
                    if any(marker in line.lower()
                           for marker in ("error", "failed", "cannot",
                                          "out of memory", "unable"))
                    and not _is_noise(_without_prefix(line))]
        return mentions[-1] if mentions else ""
