"""Building the objects and connecting them. Nothing else.

This is the only place that knows which concrete host is in use and where the
configuration lives. Everything else receives what it needs as an argument,
which is what lets tests substitute a fake host and a fake engine. A
composition root: it imports most of the application, and that is its job.

If logic starts appearing in this file, it belongs in `operations.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, cast

from .builds.core import Builds
from .catalog.capabilities import Known
from .catalog.core import Catalog
from .config.store import Config, ConfigStore
from .config.validation import Section, validate_configuration
from .cpu_models import CpuLane
from .cpu_models.settings import CpuModelsConfig
from .downloads import DownloadManager, HuggingFaceClient
from .engines.registry import Registry
from .events import EventBus
from .gateway.card_sharing import guard_loads
from .gateway.core import Gateway
from .gateway.limits import GatewayLimits, Patience
from .gateway.resources import GatewayResources
from .gpu_health.settings import GpuHealthSettings
from .hosts import current_host
from .hosts.base import Host
from .lanes import SideLanes
from .media.image_jobs import ImageJobs
from .builds.installs import Installs
from .runtime.lastloaded import LastLoaded
from .media.jobs import MediaJobs
from .operations import Operations
from .application.instances import InstanceService
from .operations_services import OperationsParts, build_services
from .providers import ProviderPool
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
    "gpu_health": Section(GpuHealthSettings.from_mapping),
}


class Manager(NamedTuple):
    """The built manager: what `main` starts and the routes are bound to."""

    operations: Operations
    host: Host
    bus: EventBus
    store: ConfigStore
    gateway: Gateway
    lanes: SideLanes
    image_jobs: ImageJobs
    media_jobs: MediaJobs


def build(config_path: Path) -> Manager:
    """The whole manager, from the configuration at `config_path`.

    The operations the routes call, the event bus the page listens on, the
    configuration store, the model gateway, the lanes beside the card and the
    two job queues. The embeddings lane is built before the gateway: it
    borrows the card, so the gateway's memory is built with it as the card's
    neighbour. The job queues send their work through the gateway, so they
    are built after it.
    """
    store = ConfigStore(config_path)
    bus = EventBus()
    config = store.load()
    engine_settings = engine_settings_of(config)
    engines = Registry(engine_settings)
    validate_configuration(config, set(engines.known()), check_workflows=True,
                           sections=CONFIG_SECTIONS)
    host = current_host(engine_settings)
    operations, instances = _operations(store, bus, host, engines, engine_settings)
    lanes = side_lanes(config, host, instances)
    limits = gateway_limits(config)
    memory = GatewayResources(operations, limits.quiet_mb, limits.quiet_timeout_s,
                              limits.poll_s, neighbours=lanes.cpu)
    model_gateway = Gateway(operations, limits, memory=memory)
    image_jobs = ImageJobs(model_gateway, store.load().images, host.state_dir(), bus)
    media_jobs = MediaJobs(model_gateway, host.state_dir(), bus, settings=config.media_policy)
    return Manager(operations, host, bus, store, model_gateway, lanes, image_jobs, media_jobs)


def side_lanes(config: Config, host: Host, instances: InstanceService) -> SideLanes:
    """Two lanes of their own beside the card.

    Subscription models (Claude Code, Codex, opencode) count their usage where
    the host keeps its state; processor models (embeddings) log there too.
    Embeddings may borrow the card; its own models come first, through the
    gateway (its memory has them as the card's neighbour) and the Load button
    alike (`guard_loads`).
    """
    state_dir = host.state_dir()
    cpu = CpuLane.from_config(config.cpu_models, state_dir,
                              card_free_mb=lambda: card_free_mb(host))
    lanes = SideLanes(providers=ProviderPool.from_config(config.providers, state_dir), cpu=cpu)
    if cpu.placements:
        cpu.start_reaper()
        guard_loads(instances, cpu)
    return lanes


def card_free_mb(host: Host) -> float | None:
    """Free memory on a card of its own, or None (no card, or unified memory)."""
    try:
        snapshot = host.accelerator()
    except Exception:
        return None
    if snapshot.memory_kind != "dedicated" or not snapshot.memory_total_mb:
        return None
    return float(snapshot.memory_total_mb - snapshot.memory_used_mb)


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
        max_upload_dimension=front_door.max_upload_dimension,
        patience=Patience(stall_s=front_door.stall_s, safety_s=front_door.safety_s,
                          default_max_tokens=front_door.default_max_tokens))


def _operations(store: ConfigStore, bus: EventBus, host: Host, engines: Registry,
                engine_settings: dict) -> tuple[Operations, InstanceService]:
    """The operations the routes call, and the instance service the side lanes guard."""
    builds = Builds(engine_settings, bus)
    # Engines that arrive as packages rather than as source. A new
    # version is installed beside the one that works, never over it.
    installs = Installs(engine_settings, bus)
    # The page knows what is waiting as soon as it opens (no button that only
    # asks), and the version figures stay current without anyone pressing.
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

    parts = OperationsParts(
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
        # What was on the card when the manager last stopped, where the host keeps state.
        last_loaded=LastLoaded(host.state_dir()))
    services = build_services(parts)
    operations = Operations(parts, services)
    built.append(operations)
    return operations, services.instance_service
