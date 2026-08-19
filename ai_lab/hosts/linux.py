"""Linux host: systemd for processes, nvidia-smi for the accelerator.

systemd owns the engine processes rather than this application, so inference
survives a manager restart and returns after a reboot. Because the manager
runs unprivileged it cannot call systemctl directly; it goes through
`/usr/local/sbin/ai-lab-control`, allowed by one narrow sudoers rule.

Instances are created at runtime, so one templated unit serves all of them:
`ai-lab-engine@<id>.service`. The command line reaches the unit through a
launch file — see `launch.py`.
"""

from __future__ import annotations

from dataclasses import replace

from ..types import AcceleratorSnapshot, Capabilities, ProcessSpec, ProcessStatus
from . import launch
from .command import run, which

CONTROL_HELPER = "/usr/local/sbin/ai-lab-control"
UNIT = "ai-lab-engine@{instance_id}.service"

GPU_QUERY = "name,memory.used,memory.total,temperature.gpu,utilization.gpu"
APPS_QUERY = "pid,used_gpu_memory"


class LinuxHost:
    """See `base.Host` for what each method promises."""

    def __init__(self, control_helper: str = CONTROL_HELPER,
                 vllm_binary: str | None = None) -> None:
        self.control_helper = control_helper
        # Passed in from the engines section of config.json, because a
        # virtualenv install is invisible to PATH.
        self.vllm_binary = vllm_binary

    # -- capabilities ------------------------------------------------------

    def capabilities(self) -> Capabilities:
        engines = set()
        if which("llama-server"):
            engines.add("llamacpp")
        # vLLM is installed in its own virtual environment, so it is normally
        # absent from PATH even when present. A configured path counts as
        # installed; PATH is only the fallback.
        if self.vllm_binary or which("vllm"):
            engines.add("vllm")
        accelerator = self.accelerator()
        return Capabilities(
            supervisor="systemd",
            engines=frozenset(engines),
            accelerator_kind=accelerator.kind,
            can_configure_accelerator=False,
            operating_system="Linux",
        )

    # -- processes ---------------------------------------------------------

    def start(self, spec: ProcessSpec) -> None:
        launch.write_spec(spec)
        self._control("start", spec.instance_id)

    def stop(self, instance_id: str) -> None:
        self._control("stop", instance_id)

    def logs(self, instance_id: str, lines: int = 15) -> list[str]:
        unit = UNIT.format(instance_id=instance_id)
        result = run(["journalctl", "-u", unit, "-n", str(lines),
                      "--no-pager", "--output=cat"], timeout=10)
        if not result.ok:
            return []
        return [line for line in result.stdout.splitlines() if line.strip()]

    def stop_all(self) -> None:
        """Deliberately does nothing.

        systemd owns the engines here, so they are supposed to keep serving
        when the manager restarts — that is the whole reason for using it.
        Only macOS, where this application is the supervisor, has engines to
        clean up.
        """

    def status(self, instance_id: str) -> ProcessStatus:
        unit = UNIT.format(instance_id=instance_id)
        active = run(["systemctl", "is-active", unit]).stdout.strip() == "active"
        enabled = run(["systemctl", "is-enabled", unit]).stdout.strip() == "enabled"
        pid = None
        if active:
            result = run(["systemctl", "show", "-p", "MainPID", "--value", unit])
            pid = int(result.stdout.strip() or 0) or None
        return ProcessStatus(running=active, pid=pid, enabled=enabled)

    def _control(self, action: str, instance_id: str) -> None:
        result = run(["sudo", "-n", self.control_helper, action, instance_id], timeout=60)
        if not result.ok:
            message = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Could not {action} {instance_id}: {message}")

    # -- accelerator -------------------------------------------------------

    def accelerator(self, pid: int | None = None) -> AcceleratorSnapshot:
        result = run(["nvidia-smi", f"--query-gpu={GPU_QUERY}",
                      "--format=csv,noheader,nounits"], timeout=5)
        if not result.ok or not result.stdout.strip():
            return AcceleratorSnapshot(available=False, name="", kind="none",
                                       memory_kind="dedicated")
        snapshot = self._parse(result.stdout.splitlines()[0])
        if pid is None:
            return snapshot
        return replace(snapshot, process_memory_mb=self._process_memory(pid))

    @staticmethod
    def _process_memory(pid: int) -> float:
        """How much of the card one process is holding.

        nvidia-smi reports this per process, and inside this unprivileged
        container the pids it lists are the container's own — the same numbers
        systemd reports — so they can be matched directly.
        """
        result = run(["nvidia-smi", f"--query-compute-apps={APPS_QUERY}",
                      "--format=csv,noheader,nounits"], timeout=5)
        if not result.ok:
            return 0.0
        for line in result.stdout.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) == pid:
                try:
                    return float(fields[1])
                except ValueError:
                    return 0.0
        return 0.0

    @staticmethod
    def _parse(line: str) -> AcceleratorSnapshot:
        fields = [item.strip() for item in line.split(",")]
        if len(fields) < 5:
            return AcceleratorSnapshot(available=False, name="", kind="none",
                                       memory_kind="dedicated")
        name, used, total, temperature, utilization = fields[:5]

        def number(text: str) -> float | None:
            try:
                return float(text)
            except ValueError:
                return None

        return AcceleratorSnapshot(
            available=True, name=name, kind="cuda", memory_kind="dedicated",
            memory_used_mb=number(used) or 0.0,
            memory_total_mb=number(total) or 0.0,
            temperature_c=number(temperature),
            utilization_percent=number(utilization),
        )
