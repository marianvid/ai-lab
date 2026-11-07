"""YuE2 song generation with an editable ABC composition."""
from __future__ import annotations

import json
from pathlib import Path

from ...types import ModelSet, Task
from ..base import MUSIC_PATHS, MUSIC_WINDOW_PATHS, LaunchPlan, ParamSpec, validate
from ..isolated import ConfiguredEngine, model_directory, script_env, ui_port
from . import music_server
from .transfer import TransferSettings

# The accelerators YuE2 runs on: the NVIDIA card on Linux, Metal on a Mac.
DEVICES = ("cuda", "mps")
# Where the lab measured them: radio-lab/music's window tests on the Mac, 2026-11-05
# (`runtime/radio-lab/music/research/yue2-pseudo-inpainting.md`).
LAB = "Default from radio-lab/music's window lab on the Mac (2026-11-05)."


# Regenerating one window of an earlier song (`POST /v1/audio/music/window-generations`).
# A request may give context_s, edge_s, crossfade_s and the sampling; these are what it
# falls back on, and the limits it is held to. Each row: the key after `window_`, the
# label, the kind, the default, the lowest and highest values, and what it is.
_WINDOW_ROWS = (
    ("context_s", "Window context (s)", "float", 5.0, 0.0, 30.0,
     "Seconds kept on each side of the window and re-synthesised with it, held to "
     "the original (RePaint). Unit: seconds. " + LAB),
    ("edge_s", "Window edge ramp (s)", "float", 0.4, 0.0, 5.0,
     "How gradually the hold on the original lets go at the window's two ends. "
     "Unit: seconds. " + LAB),
    ("crossfade_s", "Window crossfade (s)", "float", 0.3, 0.0, 5.0,
     "Crossfade where the new audio is spliced into the original song. "
     "Unit: seconds. " + LAB),
    ("swap_margin_s", "Window swap margin (s)", "float", 2.5, 0.0, 30.0,
     "How far outside the window the new audio replaces the original; it must stay "
     "inside the context, where the new audio equals the original. Unit: seconds. "
     "From the lab's splice (half of the 5 s context)."),
    ("max_candidates", "Window candidates, most", "int", 5, 1, 10,
     "The most candidates one window request may ask for. Each takes about two "
     "minutes for a 30 s window on the Mac (estimate). From radio-lab/music's "
     "specification (2026-11-06)."),
    ("min_s", "Window, shortest (s)", "float", 1.0, 0.04, 60.0,
     "The shortest window a request may regenerate. Unit: seconds. From "
     "radio-lab/music's specification (2026-11-06)."),
    ("max_s", "Window, longest (s)", "float", 60.0, 1.0, 600.0,
     "The longest window a request may regenerate, so five candidates fit in the "
     "music call's time limit. Unit: seconds. Proposed by ai-lab (2026-11-06)."),
    ("temperature", "Window sampling: temperature", "float", 1.0, 0.0, 5.0,
     "Sampling of the window's new semantic tokens: higher is more varied. " + LAB),
    ("top_p", "Window sampling: top-p", "float", 0.95, 0.01, 1.0,
     "Sampling of the window's new tokens: the share of probability kept. " + LAB),
    ("top_k", "Window sampling: top-k", "int", 100, 1, 1000,
     "Sampling of the window's new tokens: how many candidates per step. " + LAB),
    ("repetition_penalty", "Window sampling: repetition penalty", "float", 1.2, 0.1, 5.0,
     "Sampling of the window's new tokens: how much a repeated token is held back. "
     + LAB),
    ("penalty_window", "Window sampling: penalty window", "int", 50, 1, 100,
     "Sampling of the window's new tokens: how many recent tokens the penalty "
     "looks at. Unit: tokens (25 per second). " + LAB),
)
WINDOW_SETTINGS = tuple(
    ParamSpec(f"window_{key}", label, kind, default, minimum=low, maximum=high,
              group="generation", help=what)
    for key, label, kind, default, low, high, what in _WINDOW_ROWS)


class Yue2Engine(ConfiguredEngine):
    """YuE2 songs from lyrics, with the YuE2 Studio page beside the API."""

    id = "yue2"
    display_name = "YuE2"
    task = Task.MUSIC_GENERATION
    paths = MUSIC_PATHS + MUSIC_WINDOW_PATHS
    settings = WINDOW_SETTINGS
    script = music_server("yue2")
    reserves_memory = True
    location_settings = ("vae_path", "output_root", "webui_root")
    # Read by `from_settings`: moving songs between AI-Labs (`transfer.py`).
    section_settings = ("transfer",)

    def __init__(self, binary: str | None = None, server: str | None = None,
                 model_options: dict[str, dict] | None = None,
                 locations: dict[str, str] | None = None, device: str = "cuda") -> None:
        super().__init__(binary, server, model_options, locations)
        # Moving songs between this AI-Lab and the others (`transfer.py`);
        # read from the section's `transfer` by `from_settings`.
        self.transfer = TransferSettings()
        # Which accelerator the YuE2 runtime and its Studio use. Configured
        # per host rather than guessed, so a Mac never silently falls back to
        # the CPU.
        if device not in DEVICES:
            raise ValueError("YuE2 device must be cuda or mps")
        self.device = device

    @classmethod
    def from_settings(cls, settings: dict) -> Yue2Engine:
        """Built from the engine's section, with its `transfer` settings checked."""
        engine = super().from_settings(settings)
        if not isinstance(engine, Yue2Engine):
            raise TypeError("from_settings built another engine")
        engine.transfer = TransferSettings.from_mapping(settings.get("transfer"))
        return engine

    def music_form(self, model_name: str) -> dict:
        """What the music page asks for: lyrics, and an editable score."""
        return {"lyrics_required": True, "duration_kind": "none",
                "editable_score": True}

    def generations_dir(self) -> Path:
        """Where YuE2 Studio keeps every song, one folder per generation id.

        The studio's data sits in `data/` beside the `config.yaml` that
        `yue2_server.studio_config` writes under `output_root/yue2-studio`.
        """
        return Path(self.locations["output_root"]) / "yue2-studio" / "data" / "outputs"

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        """The YuE2 host for this checkpoint, with the Studio on the page port."""
        options = self._options(
            model, {}, "YuE2 VAE, output and Studio paths must be configured")
        window = validate(self.settings, params)
        where = self.locations
        return LaunchPlan(argv=[
            self.binary, self.server, "--model-path", model_directory(model),
            "--vae-path", where["vae_path"], "--output-root", where["output_root"],
            "--cot", options["cot"], "--memory-budget-gib",
            str(options["memory_budget_gib"]), "--port", str(port),
            "--ui-port", ui_port(port),
            "--webui-root", where["webui_root"], "--device", self.device,
            "--window-settings", json.dumps(window, sort_keys=True)],
            env=script_env())
