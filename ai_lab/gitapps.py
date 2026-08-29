"""Reversible installations of Git-based Python applications such as ComfyUI."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from .events import EventBus
from .types import ChangeEvent, LogEvent

SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
INSTALL_TIMEOUT_S = 3600


class GitApplicationInstall:
    """Build a full checkout and venv beside the active release, then switch."""

    def __init__(self, engine_id: str, settings: dict, bus: EventBus) -> None:
        source = settings.get("source") or {}
        self.engine_id, self.bus = engine_id, bus
        self.root = Path(source["root"])
        self.repository = source["repository"]
        self.application = source.get("application", "ComfyUI")
        self.python = source.get("python", "python3")
        self.uv = source.get("uv", "uv")
        self.verify_module = source.get("verify_module", "server")
        self.requires_cuda = bool(source.get("requires_cuda", False))
        self._lines = deque(maxlen=500)
        self._lock = threading.RLock()
        self._thread = None
        self._state, self._error, self._latest = "idle", "", ""
        self._component_latest: dict[str, str] = {}

    @property
    def link(self) -> Path:
        return self.root / "current"

    def active(self) -> Path | None:
        try:
            return self.link.resolve(strict=True)
        except OSError:
            return None

    @staticmethod
    def _metadata(release: Path) -> dict:
        try:
            return json.loads((release / ".ai-lab-release.json").read_text())
        except (OSError, ValueError):
            return {}

    def environments(self, with_sizes: bool = True) -> list[dict]:
        active, found = self.active(), []
        for entry in sorted(self.root.glob(".release-*")) if self.root.exists() else []:
            if not entry.is_dir():
                continue
            meta = self._metadata(entry)
            found.append({"name": entry.name,
                          "version": meta.get("version") or entry.name[9:],
                          "active": active == entry.resolve(),
                          "size_bytes": _size(entry) if with_sizes else 0,
                          "path": str(entry), "movable": True})
        return found

    def installed_now(self) -> str:
        active = self.active()
        return self._metadata(active).get("core_commit", "")[:12] if active else ""

    def check(self) -> dict:
        self._latest = self._remote_commit(self.repository, "HEAD")[:12]
        for row in self._components(False):
            try:
                self._component_latest[row["name"]] = self._remote_commit(
                    row["repository"], "HEAD")[:12]
            except Exception:
                self._component_latest[row["name"]] = ""
        self._announce()
        return self.status()

    def status(self) -> dict:
        environments, installed = self.environments(), self.installed_now()
        return {"engine": self.engine_id, "kind": "git-app", "root": str(self.root),
                "linked": bool(self.active()), "installed": installed,
                "latest": self._latest,
                "update_available": bool(installed and self._latest and installed != self._latest),
                "environments": environments,
                "spare_bytes": sum(x["size_bytes"] for x in environments if not x["active"]),
                "free_bytes": _free(self.root), "state": self._state,
                "error": self._error, "log": list(self._lines),
                "components": self._components(True)}

    def install(self, version: str = "") -> dict:
        return self._start(version or "HEAD", "")

    def update_component(self, name: str) -> dict:
        if not SAFE_NAME.fullmatch(name):
            raise ValueError("Invalid component name")
        active = self.active()
        if active is None:
            raise ValueError("Install ComfyUI before updating custom nodes")
        node = active / self.application / "custom_nodes" / name
        if not (node / ".git").is_dir():
            raise KeyError(f"No managed custom node called {name}")
        if self._git(node, "status", "--porcelain").strip():
            raise ValueError(f"{name} has local changes; refusing to overwrite them")
        return self._start(self._metadata(active).get("core_commit") or "HEAD", name)

    def _start(self, core_ref: str, component: str) -> dict:
        with self._lock:
            if self._state == "running":
                raise ValueError("A ComfyUI update is already running")
            self._lines.clear()
            self._error, self._state = "", "running"
            self._thread = threading.Thread(target=self._run, args=(core_ref, component),
                                            daemon=True, name=f"install-{self.engine_id}")
            self._thread.start()
        return self.status()

    def _run(self, core_ref: str, component: str) -> None:
        started, target = time.monotonic(), None
        elapsed = lambda: int((time.monotonic() - started) * 1000)
        try:
            commit = self._remote_commit(self.repository, core_ref)
            suffix = commit[:12] + (f"-{component}-{int(time.time())}" if component else "")
            target = self.root / f".release-{suffix}"
            if target.exists():
                raise ValueError(f"Release {target.name} already exists")
            target.mkdir(parents=True)
            app = target / self.application
            self._say("status", f"Fetching ComfyUI {commit[:12]}", elapsed())
            self._stream(["git", "clone", "--no-checkout", self.repository, str(app)], elapsed)
            self._stream(["git", "-C", str(app), "checkout", "--detach", commit], elapsed)
            self._copy_components(app)
            if component:
                node = app / "custom_nodes" / component
                latest = self._remote_commit(self._remote(node), "HEAD")
                self._stream(["git", "-C", str(node), "fetch", "--prune", "origin"], elapsed)
                self._stream(["git", "-C", str(node), "checkout", "--detach", latest], elapsed)
            self._say("status", "Creating an isolated Python environment", elapsed())
            self._stream([self.uv, "venv", str(target / ".venv"), "--python", self.python], elapsed)
            self._install_requirements(target, elapsed)
            self._verify(target, elapsed)
            components = {x["name"]: x["installed"] for x in self._scan_components(app)}
            meta = {"version": commit[:12], "core_commit": commit,
                    "created_at": int(time.time()), "components": components}
            (target / ".ai-lab-release.json").write_text(json.dumps(meta, indent=2) + "\n")
            self.point_at(target)
            self._state, self._latest = "done", commit[:12]
            self._say("status", f"ComfyUI now runs {commit[:12]}; previous release retained", elapsed())
        except Exception as error:
            self._state, self._error = "failed", str(error) or error.__class__.__name__
            self._say("err", self._error, elapsed())
            if target is not None and target.exists() and target != self.active():
                shutil.rmtree(target, ignore_errors=True)
        self._announce()

    def _copy_components(self, app: Path) -> None:
        active = self.active()
        if not active:
            return
        old, new = active / self.application / "custom_nodes", app / "custom_nodes"
        new.mkdir(exist_ok=True)
        if old.is_dir():
            for node in old.iterdir():
                if (node / ".git").is_dir() and not (new / node.name).exists():
                    shutil.copytree(node, new / node.name, symlinks=True)

    def _install_requirements(self, target: Path, elapsed) -> None:
        python, app = str(target / ".venv" / "bin" / "python"), target / self.application
        requirements = [app / "requirements.txt", *sorted((app / "custom_nodes").glob("*/requirements.txt"))]
        for path in requirements:
            if path.is_file():
                self._stream([self.uv, "pip", "install", "--python", python, "-r", str(path)], elapsed)

    def _verify(self, target: Path, elapsed) -> None:
        checks = ["import importlib"]
        if self.requires_cuda:
            checks += ["import torch", "assert torch.cuda.is_available(), 'CUDA unavailable'"]
        checks += [f"importlib.import_module({self.verify_module!r})", "print('ok')"]
        self._stream([str(target / ".venv" / "bin" / "python"), "-c", ";".join(checks)],
                     elapsed, cwd=target / self.application, timeout=180)

    def point_at(self, release: Path) -> None:
        if not (release / ".venv" / "bin" / "python").exists():
            raise ValueError(f"{release.name} is not usable")
        temporary = self.root / ".current-new"
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(release, target_is_directory=True)
        temporary.replace(self.link)

    def activate(self, name: str) -> dict:
        release = self.root / name
        if release not in [Path(x["path"]) for x in self.environments(False)]:
            raise KeyError(f"No installed release called {name}")
        self.point_at(release)
        self._announce()
        return self.status()

    def remove(self, name: str) -> dict:
        release = self.root / name
        if self.active() == release.resolve():
            raise ValueError(f"{name} is the release in use")
        if release not in [Path(x["path"]) for x in self.environments(False)]:
            raise KeyError(f"No installed release called {name}")
        shutil.rmtree(release)
        self._announce()
        return self.status()

    def _components(self, remote: bool) -> list[dict]:
        active = self.active()
        rows = self._scan_components(active / self.application) if active else []
        for row in rows:
            latest = self._component_latest.get(row["name"], "") if remote else ""
            row.update(latest=latest, update_available=bool(latest and latest != row["installed"]))
        return rows

    def _scan_components(self, app: Path) -> list[dict]:
        root, rows = app / "custom_nodes", []
        if not root.is_dir():
            return rows
        for node in sorted(root.iterdir()):
            if (node / ".git").is_dir():
                rows.append({"name": node.name, "repository": self._remote(node),
                             "installed": self._git(node, "rev-parse", "--short=12", "HEAD").strip(),
                             "dirty": bool(self._git(node, "status", "--porcelain").strip())})
        return rows

    def _remote(self, checkout: Path) -> str:
        return self._git(checkout, "remote", "get-url", "origin").strip()

    @staticmethod
    def _remote_commit(repository: str, ref: str) -> str:
        answer = subprocess.run(["git", "ls-remote", repository, ref], capture_output=True,
                                text=True, timeout=60, env=_environment())
        if answer.returncode or not answer.stdout.strip():
            raise ValueError(answer.stderr.strip() or f"Could not resolve {ref}")
        return answer.stdout.split()[0]

    @staticmethod
    def _git(checkout: Path, *args: str) -> str:
        answer = subprocess.run(["git", "-C", str(checkout), *args], capture_output=True,
                                text=True, timeout=60, env=_environment())
        if answer.returncode:
            raise ValueError(answer.stderr.strip() or "git failed")
        return answer.stdout

    def _stream(self, command: list[str], elapsed, timeout: int = INSTALL_TIMEOUT_S,
                cwd: Path | None = None) -> None:
        process = subprocess.Popen(command, cwd=str(cwd or self.root), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, env=_environment())
        deadline = time.monotonic() + timeout
        assert process.stdout is not None
        for line in process.stdout:
            self._say("out", line.rstrip(), elapsed())
            if time.monotonic() > deadline:
                process.kill()
                raise ValueError(f"{command[0]} took longer than {timeout}s")
        if process.wait():
            raise ValueError(f"{' '.join(command[:3])} failed")

    def _announce(self) -> None:
        if self.bus:
            self.bus.publish(ChangeEvent(topic="builds"))

    def _say(self, stream: str, text: str, elapsed_ms: int) -> None:
        self._lines.append(text)
        if self.bus:
            self.bus.publish(LogEvent(source=self.engine_id, stream=stream,
                                      text=text, elapsed_ms=elapsed_ms))


def _environment() -> dict[str, str]:
    return {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(Path.home()),
            "UV_NO_CONFIG": "1"}


def _size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def _free(path: Path) -> int:
    try:
        return shutil.disk_usage(path if path.exists() else path.parent).free
    except OSError:
        return 0
