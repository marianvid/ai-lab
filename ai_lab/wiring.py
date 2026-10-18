"""Building the objects and connecting them. Nothing else.

This is the only place that knows which concrete host is in use and where the
configuration lives. Everything else receives what it needs as an argument,
which is what lets tests substitute a fake host and a fake engine. A
composition root: it imports most of the application, and that is its job.

If logic starts appearing in this file, it belongs in `operations.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from .builds.core import Builds
from .catalog.capabilities import Known
from .catalog.core import Catalog
from .config.store import Config, ConfigStore
from .config.validation import Section, validate_configuration
from .cpu_models.settings import CpuModelsConfig
from .downloads import DownloadManager, HuggingFaceClient
from .engines.registry import Registry
from .events import EventBus
from .gateway.core import Gateway
from .gateway.limits import GatewayLimits
from .hosts import current_host
from .hosts.base import Host
from .media.image_jobs import ImageJobs
from .builds.installs import Installs
from .runtime.lastloaded import LastLoaded
from .media.jobs import MediaJobs
from .operations import Operations
from .operations_services import OperationsParts
from .providers.settings import ProvidersConfig
from .reservations.settings import ReservationPolicy
from .runtime.core import Runtime
from .settings import Settings
from .storage import Storage



def _cpu_model_ports(parsed: object) -> list[tuple[str, int]]:
    """Every port an embeddings model claims (processor, card), labelled by the model."""
    models = cast(CpuModelsConfig, parsed).models
    return [(f"cpu model {model.name}", port)
            for model in models.values() for port in model.ports]


# The configuration sections other packages own, checked by their own parsers
# when the configuration is validated (`config` imports none of them).
CONFIG_SECTIONS: dict[str, Section] = {
    "providers": Section(ProvidersConfig.from_mapping),
    "cpu_models": Section(CpuModelsConfig.from_mapping, ports=_cpu_model_ports),
    "reservations": Section(ReservationPolicy.from_mapping),
}


def build(config_path: Path) -> tuple[Operations, EventBus, ConfigStore, Gateway]:
    """The whole manager, from the configuration at `config_path`.

    Returns the operations the routes call, the event bus the page listens
    on, the configuration store and the model gateway.
    """
    store = ConfigStore(config_path)
    bus = EventBus()
    config = store.load()
    engine_settings = engine_settings_of(config)
    engines = Registry(engine_settings)
    validate_configuration(config, set(engines.known()), check_workflows=True,
                           sections=CONFIG_SECTIONS)
    host = current_host(engine_settings)
    operations = _operations(store, bus, host, engines, engine_settings)
    model_gateway = Gateway(operations, gateway_limits(config))
    operations.image_jobs = ImageJobs(
        model_gateway, store.load().images, host.state_dir(), bus)
    operations.media_jobs = MediaJobs(
        model_gateway, host.state_dir(), bus, settings=config.media_policy)
    return operations, bus, store, model_gateway


def engine_settings_of(config: Config) -> dict:
    """The `engines` section, with ComfyUI told which workflow each image model uses.

    Engines are built from configuration, so a machine holding two llama.cpp
    builds uses the one named in config.json rather than whichever PATH
    happens to find first. The host needs the same section: vLLM lives in a
    virtual environment, so whether it is installed cannot be read from PATH.
    """
    engine_settings = dict(config.engines)
    by_instance = {instance.id: instance.model_id for instance in config.instances}
    presets: dict[str, str] = {}
    workflow_root = Path(config.images.get("workflow_root", ".")).resolve()
    for profile in config.images.get("profiles", {}).values():
        model_id = by_instance.get(profile.get("model"))
        if model_id and profile.get("workflow"):
            workflow = (workflow_root / profile["workflow"]).resolve()
            if workflow.is_relative_to(workflow_root):
                presets.setdefault(model_id, str(workflow))
    engine_settings["comfyui"] = {
        **engine_settings.get("comfyui", {}), "preset_workflows": presets,
    }
    return engine_settings


def gateway_limits(config: Config) -> GatewayLimits:
    """The front door's limits, from the configuration.

    How long to wait for an engine and how many requests to hold differ
    between the two machines this runs on; per-task overrides keep a slow
    image or OCR job from forcing a text client onto its clock (see
    `Gateway.timeouts_for`).
    """
    front_door = config.gateway_policy
    return GatewayLimits(
        first_byte_s=front_door.first_byte_s,
        between_bytes_s=front_door.between_bytes_s,
        max_waiting=front_door.max_waiting,
        task_timeouts=front_door.timeout_mapping(),
        max_upload_bytes=front_door.max_upload_bytes,
        max_upload_pixels=front_door.max_upload_pixels,
        max_upload_dimension=front_door.max_upload_dimension)


def _operations(store: ConfigStore, bus: EventBus, host: Host, engines: Registry,
                engine_settings: dict) -> Operations:
    """The operations the routes call, with every service they delegate to."""
    builds = Builds(engine_settings, bus)
    # Engines that arrive as packages rather than as source. A new
    # version is installed beside the one that works, never over it.
    installs = Installs(engine_settings, bus)
    # The page should already know what is waiting when it is opened, so there
    # is no button that only asks; and the version figures stay current
    # without anyone pressing anything.
    installs.watch()
    builds.watch()
    built: list[Operations] = []

    def learn_about_new_models(_destination) -> None:
        """Read a newly downloaded model's files now, not when a page asks.

        A full scan, because that is the only way in: everything already
        known is answered from memory, so what this actually costs is reading
        the one model that just arrived.
        """
        built[0].models()

    operations = Operations.assemble(OperationsParts(
        store=store,
        # What each model's files say it can do, read once and kept beside the
        # rest of the state: a quarter of a second per GGUF model otherwise, on
        # every scan of the library.
        catalog=Catalog(Known(host.state_dir())),
        runtime=Runtime(host, bus),
        settings=Settings(store, host, engines),
        downloads=DownloadManager(bus=bus, arrived=learn_about_new_models),
        huggingface=HuggingFaceClient(),
        host=host, engines=engines, builds=builds, installs=installs,
        storage=Storage(store.load().storage, bus), bus=bus,
        # What was on the card when the manager last stopped. The host says
        # where a machine keeps such things.
        last_loaded=LastLoaded(host.state_dir())))
    built.append(operations)
    return operations
