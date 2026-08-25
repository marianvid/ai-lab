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
from pathlib import Path

from ..types import AcceleratorSnapshot, Capabilities, ProcessSpec, ProcessStatus
from . import launch
from .command import run, which

CONTROL_HELPER = "/usr/local/sbin/ai-lab-control"
# Where the deployment gives the manager somewhere to write. The launch
# files live under it too — see `launch.py`.
STATE_DIR = Path("/var/lib/ai-lab")
UNIT = "ai-lab-engine@{instance_id}.service"

# `systemctl stop` blocks until the unit is actually stopped, and the unit
# gives an engine TimeoutStopSec=30 to go quietly before killing it. So this
# has to be comfortably longer than that.
#
# It used to be exactly 60, against a TimeoutStopSec of 60. Stopping a vLLM
# instance in the middle of generating then timed out here at the very moment
# systemd was finishing the job: the model really did stop and the card really
# was released, and the interface reported "sudo: timed out after 60s". A wait
# that equals what it is waiting for reports failure on success.
CONTROL_TIMEOUT_S = 90.0

GPU_QUERY = "name,memory.used,memory.total,temperature.gpu,utilization.gpu"
APPS_QUERY = "pid,used_gpu_memory"


class LinuxHost:
    """See `base.Host` for what each method promises."""

    def __init__(self, control_helper: str = CONTROL_HELPER,
                 vllm_binary: str | None = None,
                 nemo_binary: str | None = None,
                 onnx_binary: str | None = None,
                 pyannote_binary: str | None = None) -> None:
        self.control_helper = control_helper
        # Passed in from the engines section of config.json, because a
        # virtualenv install is invisible to PATH.
        self.vllm_binary = vllm_binary
        self.nemo_binary = nemo_binary
        self.onnx_binary = onnx_binary
        self.pyannote_binary = pyannote_binary
        # What kind of accelerator this machine has, once it has said. It does
        # not change while the machine is running, and asking costs 30 ms.
        self._kind = ""

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
        if self.nemo_binary or which("nemo"):
            engines.add("nemo")
        if self.onnx_binary or which("onnxruntime"):
            engines.add("onnx")
        if self.pyannote_binary:
            engines.add("pyannote")
        return Capabilities(
            supervisor="systemd",
            engines=frozenset(engines),
            accelerator_kind=self._accelerator_kind(),
            can_configure_accelerator=False,
            operating_system="Linux",
        )

    def _accelerator_kind(self) -> str:
        """Whether there is a card here, remembered once it says yes.

        A full reading is an `nvidia-smi` — 30 ms — and this is asked for every
        page draw, alongside the reading the page actually wants. What kind of
        accelerator a machine has does not change while it is running, so it is
        worth asking once.

        Only a positive answer is kept. A machine whose driver is still coming
        up answers "none", and remembering that would leave the card invisible
        until the manager was restarted.
        """
        if not self._kind:
            found = self.accelerator().kind
            if found and found != "none":
                self._kind = found
            return found
        return self._kind

    def system_memory(self) -> tuple[float, float]:
        """From /proc/meminfo, which is a file read rather than a command.

        `MemAvailable` rather than `MemFree`: the kernel counts cache it would
        drop under pressure as available, and free alone reads as an alarming
        number on a machine that is behaving perfectly.
        """
        try:
            values = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                name, _, rest = line.partition(":")
                if name in ("MemTotal", "MemAvailable"):
                    values[name] = float(rest.strip().split()[0]) / 1024.0
            total = values.get("MemTotal", 0.0)
            return max(0.0, total - values.get("MemAvailable", total)), total
        except (OSError, ValueError, IndexError):
            return 0.0, 0.0

    def state_dir(self) -> Path:
        """The directory the unit files already give the manager to write in."""
        return STATE_DIR

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
        return self.statuses([instance_id])[instance_id]

    def statuses(self, instance_ids: list[str]) -> dict[str, ProcessStatus]:
        """Ask systemd about every instance in one command.

        Asking one at a time meant three commands per instance — is-active,
        is-enabled, and show for the pid. Measured on the container with eleven
        instances: 152 ms to read them all, and that was the whole cost of
        drawing the model list. The gateway asks the same question twice on
        every request, so a workflow of short calls was paying half a second to
        reach an engine that answers in 17 ms.

        `systemctl show` takes as many units as it is given and answers with one
        block each, so the same information costs one command instead of
        thirty-three. A unit that does not exist still gets a block, reading
        inactive and disabled, which is the right answer for an instance whose
        engine has never been started.
        """
        if not instance_ids:
            return {}
        units = {UNIT.format(instance_id=identifier): identifier
                 for identifier in instance_ids}
        result = run(["systemctl", "show",
                      "-p", "Id", "-p", "ActiveState", "-p", "MainPID", *units],
                     timeout=15)
        found = self._parse_show(result.stdout, units) if result.ok else {}
        # Anything systemd did not mention is reported as stopped rather than
        # left out: a caller asking about an instance is entitled to an answer.
        return {identifier: found.get(identifier,
                                      ProcessStatus(running=False, pid=None))
                for identifier in instance_ids}

    @staticmethod
    def _parse_show(output: str, units: dict[str, str]) -> dict[str, ProcessStatus]:
        """One block per unit, separated by a blank line, in no fixed order.

        Keyed by the Id systemd reports rather than by position, because the
        properties inside a block come back in whatever order it likes and a
        unit can be missing from the answer entirely.
        """
        statuses = {}
        for block in output.split("\n\n"):
            fields = dict(line.split("=", 1) for line in block.splitlines()
                          if "=" in line)
            identifier = units.get(fields.get("Id", ""))
            if identifier is None:
                continue
            running = fields.get("ActiveState") == "active"
            pid = int(fields.get("MainPID") or 0) or None
            statuses[identifier] = ProcessStatus(running=running,
                                                 pid=pid if running else None)
        return statuses

    def _control(self, action: str, instance_id: str) -> None:
        result = run(["sudo", "-n", self.control_helper, action, instance_id],
                     timeout=CONTROL_TIMEOUT_S)
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
