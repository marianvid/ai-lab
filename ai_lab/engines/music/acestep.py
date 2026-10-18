"""ACE-Step music engine: isolated runtime with an AI-Lab HTTP contract."""
from __future__ import annotations

from ...types import ModelSet, Task
from ..base import MUSIC_PATHS, LaunchPlan
from ..isolated import IsolatedEngine, ui_port
from . import music_server


class AceStepEngine(IsolatedEngine):
    """ACE-Step 1.5, one checkpoint configuration per model name (`model_configs`)."""

    id = "acestep"
    display_name = "ACE-Step 1.5"
    task = Task.MUSIC_GENERATION
    paths = MUSIC_PATHS
    script = music_server("acestep")

    def __init__(self, binary: str | None = None, server: str | None = None,
                 project_root: str | None = None,
                 model_configs: dict[str, str] | None = None,
                 output_root: str | None = None) -> None:
        super().__init__(binary, server)
        self.project_root = project_root or ""
        self.model_configs = dict(model_configs or {})
        self.output_root = output_root or ""

    def supports(self, model: ModelSet) -> bool:
        """Whether this is a music checkpoint with a configuration named for it."""
        return (model.format in self.formats() and model.task is self.task
                and model.name in self.model_configs)

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        """The ACE-Step server for this checkpoint, with its own editor beside it."""
        self._checked(model, params)
        config_name = self.model_configs.get(model.name)
        if not config_name or not self.project_root or not self.output_root:
            raise ValueError(f"ACE-Step runtime is not configured for {model.name}")
        return LaunchPlan(argv=[
            self.binary, self.server, "--project-root", self.project_root,
            "--config-name", config_name, "--model-path", model.entrypoint,
            "--output-root", self.output_root, "--ui-port", ui_port(port),
            "--port", str(port),
        ], env={"PYTHONUNBUFFERED": "1"})
