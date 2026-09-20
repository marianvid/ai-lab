"""Static configuration checks before the manager starts any background work."""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .types import Task

MODEL_MAP_FIELDS = {"acestep": "model_configs", "qwentts": "model_modes",
                    "kokoro": "model_options", "voxcpm": "model_options",
                    "khala": "model_options", "higgs": "model_options",
                    "heartmula": "model_options"}


def validate_configuration(config: Config, engine_ids: set[str],
                           *, check_workflows: bool = False) -> None:
    """Reject broken references and conflicting ports with one useful report.

    Model files are checked by the catalog when loaded. This validation only
    uses the declared configuration, so it also works against a snapshot on a
    machine that does not host the weights.
    """
    errors: list[str] = []
    roots = {item.id for item in config.model_roots}
    if config.download_root not in roots:
        errors.append(f"Unknown download root: {config.download_root}")
    repository_ids = [item.id for item in config.repositories]
    if len(repository_ids) != len(set(repository_ids)):
        errors.append("Repository IDs must be unique")
    instance_ids: set[str] = set()
    ports = {config.port}
    if not 1 <= config.port <= 65535:
        errors.append("Manager port must be between 1 and 65535")
    for item in config.instances:
        if item.id in instance_ids:
            errors.append(f"Duplicate instance ID: {item.id}")
        instance_ids.add(item.id)
        if item.engine not in engine_ids:
            errors.append(f"{item.id}: unknown engine {item.engine}")
        if item.engine in MODEL_MAP_FIELDS:
            field = MODEL_MAP_FIELDS[item.engine]
            configured = config.engines.get(item.engine, {}).get(field, {})
            model_name = item.model_id.rsplit("/", 1)[-1]
            if model_name not in configured:
                errors.append(f"{item.id}: {item.engine} has no configured checkpoint")
            elif item.engine == "kokoro":
                options = configured[model_name]
                required = ("language_code", "default_voice", "repo_id")
                if not isinstance(options, dict) or any(
                        not isinstance(options.get(key), str) or not options[key]
                        for key in required):
                    errors.append(f"{item.id}: Kokoro model options are incomplete")
            elif item.engine == "voxcpm":
                options = configured[model_name]
                if not isinstance(options, dict) or not (
                    type(options.get("cfg_value")) in (int, float) and
                    0 < options["cfg_value"] <= 10 and
                    type(options.get("inference_timesteps")) is int and
                    1 <= options["inference_timesteps"] <= 100):
                    errors.append(f"{item.id}: VoxCPM inference settings are invalid")
            elif item.engine == "khala":
                options = configured[model_name]
                if not isinstance(options, dict) or not (
                    type(options.get("default_bucket")) is int and
                    type(options.get("maximum_bucket")) is int and
                    0 <= options["default_bucket"] <= options["maximum_bucket"] <= 20):
                    errors.append(f"{item.id}: Khala length buckets are invalid")
            elif item.engine == "higgs":
                options = configured[model_name]
                if not isinstance(options, dict) or not (
                    type(options.get("worker_port")) is int and
                    1 <= options["worker_port"] <= 65535 and
                    options["worker_port"] != item.port and
                    options["worker_port"] not in ports and
                    type(options.get("memory_reservation_mb")) in (int, float) and
                    options["memory_reservation_mb"] > 0 and
                    type(options.get("mem_fraction_static")) in (int, float) and
                    0 < options["mem_fraction_static"] < 1):
                    errors.append(f"{item.id}: Higgs worker settings are invalid")
                else:
                    ports.add(options["worker_port"])
            elif item.engine == "heartmula":
                options = configured[model_name]
                if not isinstance(options, dict) or not (
                    isinstance(options.get("version"), str) and options["version"] and
                    isinstance(options.get("checkpoint_subdir"), str) and
                    options["checkpoint_subdir"] and
                    Path(options["checkpoint_subdir"]).name == options["checkpoint_subdir"] and
                    type(options.get("topk")) is int and 1 <= options["topk"] <= 1000 and
                    type(options.get("temperature")) in (int, float) and
                    0 <= options["temperature"] <= 5 and
                    type(options.get("cfg_scale")) in (int, float) and
                    0 < options["cfg_scale"] <= 10 and
                    type(options.get("memory_reservation_mb")) in (int, float) and
                    options["memory_reservation_mb"] > 0):
                    errors.append(f"{item.id}: HeartMuLa settings are invalid")
        repository_id = item.model_id.split("/", 1)[0]
        if repository_id not in repository_ids:
            errors.append(f"{item.id}: unknown repository {repository_id}")
        if item.port in ports:
            errors.append(f"{item.id}: port {item.port} is already assigned")
        if not 1 <= item.port <= 65535:
            errors.append(f"{item.id}: port must be between 1 and 65535")
        ports.add(item.port)

    images = config.images
    profiles = images.get("profiles", {})
    workflow_root = Path(images.get("workflow_root", ""))
    for profile_id, profile in profiles.items():
        model_id = profile.get("model", "")
        if model_id not in instance_ids:
            errors.append(f"Image profile {profile_id}: unknown instance {model_id}")
            continue
        task = profile.get("task", "generation")
        if task not in {"generation", "edit"}:
            errors.append(f"Image profile {profile_id}: unsupported task {task}")
            continue
        instance = config.instance(model_id)
        repository = config.repository(instance.model_id.split("/", 1)[0])
        expected = (Task.IMAGE_EDIT if task == "edit" else
                    Task.IMAGE_GENERATION).value
        if repository.task != expected:
            errors.append(f"Image profile {profile_id}: {model_id} is configured "
                          f"for {repository.task}, expected {expected}")
        workflow = profile.get("workflow", "")
        if not workflow or Path(workflow).name != workflow:
            errors.append(f"Image profile {profile_id}: invalid workflow name")
        elif check_workflows and not (workflow_root / workflow).is_file():
            errors.append(f"Image profile {profile_id}: workflow file is absent")
    if errors:
        raise ValueError("Invalid AI-Lab configuration:\n- " + "\n- ".join(errors))
