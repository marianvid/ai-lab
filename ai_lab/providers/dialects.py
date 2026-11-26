"""How each CLI is asked a question and how its answer is read (Strategy).

Both CLIs are agents, built to edit files and run commands. Here they only
answer, so each call runs in an empty scratch directory: nothing there to read
or change, and no `AGENTS.md` or project notes for the CLI to load into the
prompt.

The prompt always goes in on standard input. As an argument, a long prompt
would pass Linux's limit on the length of one argument (128 KiB).
"""

from __future__ import annotations

import contextlib
import getpass
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from . import credentials
from .settings import Vendor, VendorLimits

# Where Codex writes its final message. Read back from the scratch directory.
CODEX_ANSWER_FILE = "answer.txt"
# Merged over opencode's own configuration for every call: every tool asks for
# a permission, and every permission is refused, so the agent can only answer.
OPENCODE_NO_TOOLS = json.dumps({"permission": {"*": "deny"}})
# Claude Code updates itself in the background unless told not to; the
# installed version is changed by hand, so every call runs the same binary.
CLAUDE_NO_SELF_UPDATE = {"DISABLE_AUTOUPDATER": "1"}
# Claude Code prefers these to the subscription (its yearly token or its
# login): one left in the manager's environment by accident would bill an API
# account, or fail, instead of using the plan. AI-Lab only drives the plan.
CLAUDE_NOT_THE_PLAN = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


@dataclass(frozen=True)
class Invocation:
    """One CLI call, ready to run."""

    argv: list[str]
    prompt: str
    env: dict[str, str] = field(default_factory=dict)
    # Variables taken out of the manager's environment before the call.
    unset: tuple[str, ...] = ()


@dataclass(frozen=True)
class Outcome:
    """What the call left behind: exit code, the answer, and the error text."""

    returncode: int
    answer: str
    errors: str
    timed_out: bool = False
    # The client left while the CLI ran, and the CLI was ended for it.
    cancelled: bool = False

    @property
    def succeeded(self) -> bool:
        """Exit code 0 and something to hand back."""
        return (self.returncode == 0 and bool(self.answer.strip())
                and not self.timed_out and not self.cancelled)


class Dialect(Protocol):
    """One CLI's way of being called and of answering.

    `efforts` lists the reasoning efforts the CLI accepts; empty means the
    CLI itself is the judge.
    """

    efforts: tuple[str, ...]
    # Variables this CLI must never get, from the manager or from the vendor's file.
    refused_env: tuple[str, ...]

    def invocation(self, vendor: Vendor, model: str, prompt: str, effort: str,
                   workdir: Path) -> Invocation:
        """The command line for one question."""

    def answer(self, stdout: str, workdir: Path) -> str:
        """The model's reply, wherever this CLI puts it."""


class ClaudeCli:
    """Claude Code: `claude -p --model <m>`; the reply is standard output."""

    # Claude Code ignores any other value with only a warning, so a typo
    # would quietly run at the default effort; the gateway refuses it instead.
    efforts: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
    refused_env: tuple[str, ...] = CLAUDE_NOT_THE_PLAN

    def invocation(self, vendor: Vendor, model: str, prompt: str, effort: str,
                   workdir: Path) -> Invocation:
        """`claude -p`, reading the prompt from standard input."""
        argv = [vendor.binary, "-p", "--model", model]
        if effort:
            argv += ["--effort", effort]
        env = dict(CLAUDE_NO_SELF_UPDATE)
        if vendor.home:
            env["HOME"] = vendor.home
        return Invocation(argv=argv, prompt=prompt, env=env, unset=self.refused_env)

    def answer(self, stdout: str, workdir: Path) -> str:
        """Everything printed is the reply."""
        return stdout.strip()


class CodexCli:
    """Codex: `codex exec -m <m> -`; the reply is written to a file.

    `--ephemeral` keeps no session; `project_doc_max_bytes=0` stops it loading
    an `AGENTS.md` into the prompt; `CODEX_HOME` points at a configuration
    with no plugins, so a headless call starts nothing it does not need.
    """

    efforts: tuple[str, ...] = ()
    refused_env: tuple[str, ...] = ()

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


class OpencodeCli:
    """opencode: `opencode run -m <provider/model>`; the reply is standard output.

    opencode is the gate to open-source models run in the cloud. Which
    providers and models exist is opencode's own configuration, kept by the
    owner in the vendor's home; an entry here only names one of them. Every
    tool is denied for the call (`OPENCODE_NO_TOOLS`).
    """

    # A variant's name belongs to the provider behind opencode.
    efforts: tuple[str, ...] = ()
    refused_env: tuple[str, ...] = ()

    def invocation(self, vendor: Vendor, model: str, prompt: str, effort: str,
                   workdir: Path) -> Invocation:
        """`opencode run`, reading the prompt from standard input."""
        argv = [vendor.binary, "run", "-m", model]
        if effort:
            argv += ["--variant", effort]
        env = {"OPENCODE_CONFIG_CONTENT": OPENCODE_NO_TOOLS}
        if vendor.home:
            env["HOME"] = vendor.home
        return Invocation(argv=argv, prompt=prompt, env=env)

    def answer(self, stdout: str, workdir: Path) -> str:
        """Everything printed is the reply; the banner goes to standard error."""
        return stdout.strip()


def user_names(environ: dict[str, str]) -> dict[str, str]:
    """`USER` and `LOGNAME` when the manager was started without them.

    Claude Code on macOS keeps its sign-in in the Keychain under the user's
    name, read from `USER`. The AI-Lab Menu app starts the manager with only
    `HOME` and `PATH`, so Claude Code answered "Not logged in" although the
    owner was signed in. Measured on the Mac, 2026-11-01.
    """
    name = environ.get("USER") or environ.get("LOGNAME") or getpass.getuser()
    return {key: name for key in ("USER", "LOGNAME") if not environ.get(key)}


def environment(environ: dict[str, str], call: Invocation, vendor: Vendor) -> dict[str, str]:
    """The CLI's environment: the manager's, less `call.unset`, plus the call's
    own variables and the vendor's private file (read at every call, so a token
    written while AI-Lab runs counts from the next call), which cannot bring
    back a name in `call.unset`."""
    env = {key: value for key, value in environ.items() if key not in call.unset}
    return {**env, **user_names(environ), **call.env,
            **credentials.read(vendor.env_file, refused=call.unset)}


DIALECTS: dict[str, Dialect] = {"claude": ClaudeCli(), "codex": CodexCli(),
                                "opencode": OpencodeCli()}


@dataclass(frozen=True)
class Ask:
    """One question for a CLI, and whether its caller still wants the answer.

    `still_wanted` turns False once the client has gone (its connection
    closed); None means nobody is watching.
    """

    model: str
    prompt: str
    effort: str = ""
    still_wanted: Callable[[], bool] | None = None


def run(dialect: Dialect, vendor: Vendor, model: str, prompt: str,
        effort: str = "") -> Outcome:
    """Run one call with nobody watching its client (see `run_ask`)."""
    return run_ask(dialect, vendor, Ask(model, prompt, effort))


def run_ask(dialect: Dialect, vendor: Vendor, ask: Ask) -> Outcome:
    """Run one call in a fresh scratch directory and read what it left.

    While the CLI works, AI-Lab looks every `client_check_s` whether the call
    is still wanted (`ask.still_wanted`). When it is not — the caller
    cancelled, as the scheduler does for a cancelled lot — the CLI is ended
    at once rather than left to spend the vendor's allowance for nobody
    (owner, 2026-11-26: a client's cancel is the server's cancel, "daca acest
    lucru mai este posibil"). What the vendor already used before that is
    spent.
    """
    still_wanted = ask.still_wanted
    with tempfile.TemporaryDirectory(prefix="ai-lab-cli-") as scratch:
        workdir = Path(scratch)
        call = dialect.invocation(vendor, ask.model, ask.prompt, ask.effort, workdir)
        try:
            # Its own process group, so ending it ends whatever it started.
            child = subprocess.Popen(
                call.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, cwd=workdir,
                env=environment(dict(os.environ), call, vendor), start_new_session=True)
        except OSError as error:
            return Outcome(returncode=-1, answer="", errors=f"cannot start the CLI: {error}")
        stdout, stderr, ended = _wait(child, call.prompt, vendor.limits, still_wanted)
        if ended == "timeout":
            return Outcome(returncode=-1, answer="", errors="timeout", timed_out=True)
        if ended == "cancelled":
            return Outcome(returncode=-1, answer="", errors="cancelled: the client left",
                           cancelled=True)
        return Outcome(returncode=child.returncode,
                       answer=dialect.answer(stdout, workdir),
                       errors="\n".join(part for part in (stderr, stdout) if part))


def _wait(child: subprocess.Popen, prompt: str, limits: VendorLimits,
          still_wanted: Callable[[], bool] | None) -> tuple[str, str, str]:
    """Feed the prompt and wait for the CLI, looking around between waits.

    Returns its output and how it ended: `done`, `timeout` or `cancelled`.
    The prompt is written and the output read by threads of their own, and
    this thread only waits for the process. Not `communicate(timeout=...)`
    retried: on Python 3.11 a retry stops writing the rest of a long prompt,
    so the CLI waits for it for ever (measured on the Mac, 2026-11-26; 3.13
    does resume).
    """
    pipes = _Pipes(child, prompt)
    deadline = time.monotonic() + limits.timeout_s
    while True:
        wait_s = max(0.01, min(limits.client_check_s, deadline - time.monotonic()))
        try:
            child.wait(timeout=wait_s)
            return (*pipes.collected(), "done")
        except subprocess.TimeoutExpired:
            pass
        if time.monotonic() >= deadline:
            _end(child, limits.stop_grace_s)
            return "", "", "timeout"
        if still_wanted is not None and not still_wanted():
            _end(child, limits.stop_grace_s)
            return "", "", "cancelled"


class _Pipes:
    """The CLI's three pipes, each served by a thread: the prompt in, the output out."""

    def __init__(self, child: subprocess.Popen, prompt: str) -> None:
        self._out: list[str] = []
        self._err: list[str] = []
        self._threads = [
            threading.Thread(target=_feed, args=(child.stdin, prompt), daemon=True),
            threading.Thread(target=_drain, args=(child.stdout, self._out), daemon=True),
            threading.Thread(target=_drain, args=(child.stderr, self._err), daemon=True)]
        for thread in self._threads:
            thread.start()

    def collected(self) -> tuple[str, str]:
        """Everything the CLI wrote, once its pipes have closed."""
        for thread in self._threads:
            thread.join()
        return "".join(self._out), "".join(self._err)


def _feed(stream, prompt: str) -> None:
    """Hand over the prompt and close standard input; a CLI that left early is no error."""
    with contextlib.suppress(OSError, ValueError):
        stream.write(prompt)
    with contextlib.suppress(OSError, ValueError):
        stream.close()


def _drain(stream, into: list[str]) -> None:
    with contextlib.suppress(OSError, ValueError):
        into.append(stream.read())


def _end(child: subprocess.Popen, grace_s: float) -> None:
    """Ask the CLI's whole group to stop; kill it if it is still there after `grace_s`."""
    for signal_number, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, None)):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(child.pid, signal_number)
        try:
            child.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue
