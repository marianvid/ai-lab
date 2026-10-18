"""YuE2 song generation with an editable ABC composition."""
from __future__ import annotations

from ..types import ModelSet, Task
from .base import MUSIC_PATHS, LaunchPlan
from .isolated import ConfiguredEngine, model_directory, script_env, ui_port

# The accelerators YuE2 runs on: the NVIDIA card on Linux, Metal on a Mac.
DEVICES = ("cuda", "mps")


class Yue2Engine(ConfiguredEngine):
    """YuE2 songs from lyrics, with the YuE2 Studio page beside the API."""

    id = "yue2"
    display_name = "YuE2"
    task = Task.MUSIC_GENERATION
    paths = MUSIC_PATHS
    script = ("music", "yue2_server.py")
    reserves_memory = True
    location_settings = ("vae_path", "output_root", "webui_root")

    def __init__(self, binary: str | None = None, server: str | None = None,
                 model_options: dict[str, dict] | None = None,
                 locations: dict[str, str] | None = None, device: str = "cuda") -> None:
        super().__init__(binary, server, model_options, locations)
        # Which accelerator the YuE2 runtime and its Studio use. Configured
        # per host rather than guessed, so a Mac never silently falls back to
        # the CPU.
        if device not in DEVICES:
            raise ValueError("YuE2 device must be cuda or mps")
        self.device = device

    def music_form(self, model_name: str) -> dict:
        """What the music page asks for: lyrics, and an editable score."""
        return {"lyrics_required": True, "duration_kind": "none",
                "editable_score": True}

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        """The YuE2 host for this checkpoint, with the Studio on the page port."""
        options = self._options(
            model, params, "YuE2 VAE, output and Studio paths must be configured")
        where = self.locations
        return LaunchPlan(argv=[
            self.binary, self.server, "--model-path", model_directory(model),
            "--vae-path", where["vae_path"], "--output-root", where["output_root"],
            "--cot", options["cot"], "--memory-budget-gib",
            str(options["memory_budget_gib"]), "--port", str(port),
            "--ui-port", ui_port(port),
            "--webui-root", where["webui_root"], "--device", self.device],
            env=script_env())
