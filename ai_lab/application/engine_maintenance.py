"""Engine build and package maintenance, with active-instance protection."""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable

from ..capabilities import IMAGES, TOOLS
from ..changes import Reader, counted
from ..types import Interests


class EngineMaintenanceService:
    def __init__(self, store, catalog, host, engines, builds, installs,
                 instances: Callable[[], list[dict]]) -> None:
        self.store = store
        self.catalog = catalog
        self.host = host
        self.engines = engines
        self.builds = builds
        self.installs = installs
        self.instances = instances

    def build_status(self, with_sizes: bool = True) -> list[dict]:
        return self.builds.all(with_sizes=with_sizes) if self.builds else []

    def check_for_update(self, engine_id: str) -> dict:
        return self.builds.get(engine_id).check()

    # -- reading an update before taking it --------------------------------

    def what_would_change(self, engine_id: str) -> dict:
        """Everything that can be found out about updating this engine.

        An update should be a decision rather than a hope, so this is what the
        page shows before offering the button: what is installed, what it would
        become, which changes matter *here*, what the people upstream wrote
        about it, and — for an engine installed as packages — exactly which
        packages would be replaced and which of them would go backwards.

        Nothing is changed by asking. Every source is read separately, so one
        being unreachable costs only that section.
        """
        settings = self.store.load().engines.get(engine_id) or {}
        engine = self.engines.get(engine_id)
        reader = Reader(engine_id, settings.get("source"),
                        binary=settings.get("binary", ""))
        installed, latest = "", ""
        try:
            build = self.builds.get(engine_id)
            status = build.status()
            installed = status.get("installed") or ""
            latest = status.get("latest") or ""
        except KeyError:
            pass                       # not built from source; notes still apply
        if not installed or not latest:
            try:
                managed = self._installs().get(engine_id).status()
                installed = installed or managed.get("installed", "")
                latest = latest or managed.get("latest", "")
            except KeyError:
                pass
        # For an engine installed as packages there is no build to ask, so the
        # versions come from the environment and from what the package manager
        # says it would do. Asked before the notes are fetched, because the
        # notes are chosen by which versions lie between the two.
        moves, trouble = reader.moves()
        if not installed or not latest:
            from_packages = reader.versions(moves)
            installed = installed or from_packages[0]
            latest = latest or from_packages[1]
        found = reader.read(installed, latest, self.interests())
        return {
            "engine": engine_id,
            "engine_name": getattr(engine, "name", engine_id),
            "installed": found.installed,
            "latest": found.latest,
            "yours": [asdict(item) for item in found.yours],
            "others": [asdict(item) for item in found.others],
            "by_area": counted(found.yours),
            "other_areas": counted(found.others),
            "notes": found.notes,
            "packages": [move.json() for move in moves],
            "unreadable": " ".join(part for part in
                                   (found.unreadable, trouble) if part),
        }

    def interests(self) -> Interests:
        """What this machine uses, so an update can be read against it.

        Worked out, never written down. The card decides which hardware
        changes are worth reading — the Mac wants Metal and the container wants
        CUDA, and neither is told which it is. The configured entries decide
        the rest: somebody running only text models does not need a hundred
        lines about the vision code.
        """
        config = self.store.load()
        models = {item.id: item for item in self.catalog.scan(config.repositories)}
        formats, pictures, tools = set(), False, False
        for entry in config.instances:
            model = models.get(entry.model_id)
            if model is None:
                continue
            formats.add(model.format.value)
            able = model.capabilities
            # An entry that switches pictures off is not using them, however
            # capable its weights are.
            if IMAGES in able and not entry.params.get("language_model_only"):
                pictures = True
            if TOOLS in able:
                tools = True
        return Interests(accelerator_kind=self.host.capabilities().accelerator_kind,
                         formats=frozenset(formats), pictures=pictures,
                         tools=tools)

    # -- engines that arrive as packages -----------------------------------

    def install_status(self, engine_id: str) -> dict:
        return self._installs().get(engine_id).status()

    def installs_available(self) -> list[dict]:
        return self.installs.all() if self.installs else []

    def _installs(self) -> Installs:
        if self.installs is None:
            raise KeyError("No engine on this machine is installed as packages")
        return self.installs

    def install_engine(self, engine_id: str, version: str = "") -> dict:
        """Install a new version beside the one in use.

        Refused while anything is running, for the same reason recompiling is:
        the engine ends up being launched from somewhere else, and a model
        already on the card would keep running the old one while the page said
        otherwise. Better to be plain about it than to be subtly wrong.

        The download itself is safe at any time — nothing existing is written
        to — but the swap at the end is not worth splitting into a second
        button nobody would remember to press.
        """
        self._nothing_running("The engine is about to be launched from "
                              "somewhere else.")
        return self._installs().get(engine_id).install(version)

    def activate_install(self, engine_id: str, name: str) -> dict:
        """Go back to, or forward to, an installed version."""
        self._nothing_running("The engine is about to be launched from "
                              "somewhere else.")
        return self._installs().get(engine_id).activate(name)

    def remove_install(self, engine_id: str, name: str) -> dict:
        """Delete an installed version that is not in use.

        Never automatic. The previous version is the way back from an update
        that turned out badly, and deciding it is no longer needed is a
        judgement about whether the new one has proved itself — which is not a
        judgement a timer can make.
        """
        return self._installs().get(engine_id).remove(name)

    def update_install_component(self, engine_id: str, name: str) -> dict:
        self._nothing_running("The engine and its extensions are being replaced.")
        install = self._installs().get(engine_id)
        if not hasattr(install, "update_component"):
            raise ValueError(f"{engine_id} has no separately managed components")
        return install.update_component(name)

    def activate_build(self, engine_id: str, name: str) -> dict:
        """Select a compiled source version without rebuilding it."""
        self._nothing_running("The engine is about to be launched from "
                              "somewhere else.")
        return self.builds.get(engine_id).activate(name)

    def remove_build(self, engine_id: str, name: str) -> dict:
        """Delete a compiled source version that is not in use."""
        return self.builds.get(engine_id).remove(name)

    def _nothing_running(self, why: str) -> None:
        running = [item["id"] for item in self.instances() if item["running"]]
        if running:
            raise ValueError("Unload the running instances first: "
                             + ", ".join(running) + ". " + why)

    def update_engine(self, engine_id: str) -> dict:
        """Pull and recompile an engine from source.

        Refused while anything is running. On Linux the linker cannot write
        over a binary that is executing, so the build would fail partway with
        a confusing message about a busy file. Better to say plainly that the
        models need unloading first.
        """
        self._nothing_running("The engine binary cannot be replaced while it "
                              "is executing.")
        return self.builds.get(engine_id).update()
