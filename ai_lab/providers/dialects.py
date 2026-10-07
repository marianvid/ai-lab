"""How each CLI is asked a question and how its answer is read (Strategy).

Both CLIs are agents, built to edit files and run commands. Here they only
answer, so each call runs in an empty scratch directory: nothing there to read
or change, and no `AGENTS.md` or project notes for the CLI to load into the
prompt.

The prompt always goes in on standard input. As an argument, a long prompt
would pass Linux's limit on the length of one argument (128 KiB).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .settings import Vendor

# Where Codex writes its final message. Read back from the scratch directory.
CODEX_ANSWER_FILE = "answer.txt"


@dataclass(frozen=True)
class Invocation:
    """One CLI call, ready to run."""

    argv: list[str]
    prompt: str
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Outcome:
    """What the call left behind: exit code, the answer, and the error text."""

    returncode: int
    answer: str
    errors: str
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        """Exit code 0 and something to hand back."""
        return self.returncode == 0 and bool(self.answer.strip()) and not self.timed_out


class Dialect(Protocol):
    """One CLI's way of being called and of answering."""

    def invocation(self, vendor: Vendor, model: str, prompt: str, effort: str,
                   workdir: Path) -> Invocation:
        """The command line for one question."""

    def answer(self, stdout: str, workdir: Path) -> str:
        """The model's reply, wherever this CLI puts it."""


class ClaudeCli:
    """Claude Code: `claude -p --model <m>`; the reply is standard output."""

    def invocation(self, vendor: Vendor, model: str, prompt: str, effort: str,
                   workdir: Path) -> Invocation:
        """`claude -p`, reading the prompt from standard input."""
        argv = [vendor.binary, "-p", "--model", model]
        if effort:
            argv += ["--effort", effort]
        env = {"HOME": vendor.home} if vendor.home else {}
        return Invocation(argv=argv, prompt=prompt, env=env)

    def answer(self, stdout: str, workdir: Path) -> str:
        """Everything printed is the reply."""
        return stdout.strip()


class CodexCli:
    """Codex: `codex exec -m <m> -`; the reply is written to a file.

    `--ephemeral` keeps no session; `project_doc_max_bytes=0` stops it loading
    an `AGENTS.md` into the prompt; `CODEX_HOME` points at a configuration
    with no plugins, so a headless call starts nothing it does not need.
    """

    def invocation(self, vendor: Vendor, model: str, prompt: str, effort: str,
                   workdir: Path) -> Invocation:
        """`codex exec`, prompt on standard input (`-`), reply to a file."""
        argv = [vendor.binary, "exec", "-m", model, "--ephemeral",
                "--skip-git-repo-check", "-c", "project_doc_max_bytes=0"]
        if effort:
            argv += ["-c", f'model_reasoning_effort="{effort}"']
        argv += ["-o", str(workdir / CODEX_ANSWER_FILE), "-"]
        env = {"CODEX_HOME": vendor.home} if vendor.home else {}
        return Invocation(argv=argv, prompt=prompt, env=env)

    def answer(self, stdout: str, workdir: Path) -> str:
        """The file Codex was told to write; empty if it wrote none."""
        path = workdir / CODEX_ANSWER_FILE
        return path.read_text(errors="ignore").strip() if path.exists() else ""


DIALECTS: dict[str, Dialect] = {"claude": ClaudeCli(), "codex": CodexCli()}


def run(dialect: Dialect, vendor: Vendor, model: str, prompt: str,
        effort: str = "") -> Outcome:
    """Run one call in a fresh scratch directory and read what it left."""
    with tempfile.TemporaryDirectory(prefix="ai-lab-cli-") as scratch:
        workdir = Path(scratch)
        call = dialect.invocation(vendor, model, prompt, effort, workdir)
        try:
            completed = subprocess.run(
                call.argv, input=call.prompt, capture_output=True, text=True,
                timeout=vendor.limits.timeout_s, cwd=workdir, check=False,
                env={**os.environ, **call.env})
        except subprocess.TimeoutExpired:
            return Outcome(returncode=-1, answer="", errors="timeout", timed_out=True)
        except OSError as error:
            return Outcome(returncode=-1, answer="", errors=f"cannot start the CLI: {error}")
        return Outcome(returncode=completed.returncode,
                       answer=dialect.answer(completed.stdout, workdir),
                       errors="\n".join(part for part in (completed.stderr, completed.stdout)
                                        if part))
