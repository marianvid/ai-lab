"""Static configuration checks before the manager starts any background work.

How it is built (Chain of Responsibility + Strategy):

- `validate_configuration` runs a fixed chain of section checkers — storage
  roots, repositories, the manager port, each instance, the policies, the
  image profiles. Each checker adds plain-language messages to one shared
  `_Report`, so the person reading the error sees every problem at once.
- How an engine's per-model options are judged is a table, `ENGINE_RULES`:
  one entry per engine, a short message and a tuple of small rules such as
  "`steps` is a whole number from 1 to 200". Adding an engine is one entry,
  not a new branch in a long function.

Model files are checked by the catalog when loaded. This validation only uses
the declared configuration, so it also works against a snapshot on a machine
that does not host the weights.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .types import Task

Rule = Callable[[dict], bool]

MAX_PORT = 65535
# Khala's largest length bucket.
KHALA_MAX_BUCKET = 20

MODEL_MAP_FIELDS = {"acestep": "model_configs", "qwentts": "model_modes",
                    "kokoro": "model_options", "voxcpm": "model_options",
                    "khala": "model_options", "higgs": "model_options",
                    "higgs_local": "model_options",
                    "heartmula": "model_options", "yue2": "model_options",
                    "comfy_music": "model_options",
                    "mulacover": "model_options",
                    "levo2": "model_options",
                    "stableaudio3": "model_options"}


# -- small rules, each about one option --------------------------------------

def _is_number(value) -> bool:
    # bool is an int in Python; a setting of `true` is not a number here.
    return type(value) in (int, float)


def text(key: str) -> Rule:
    """A non-empty string."""
    return lambda options: isinstance(options.get(key), str) and bool(options[key])


def file_name(key: str) -> Rule:
    """A bare file or folder name, never a path."""
    return lambda options: text(key)(options) and Path(options[key]).name == options[key]


def absolute_path(key: str) -> Rule:
    """An absolute path, written as text."""
    return lambda options: (isinstance(options.get(key), str)
                            and Path(options[key]).is_absolute())


def file_names(key: str) -> Rule:
    """A non-empty list of bare names."""
    def check(options: dict) -> bool:
        values = options.get(key)
        return (isinstance(values, list) and bool(values)
                and all(isinstance(value, str) and value and Path(value).name == value
                        for value in values))
    return check


def whole(key: str, low: int, high: int) -> Rule:
    """A whole number from `low` to `high`, both included."""
    return lambda options: type(options.get(key)) is int and low <= options[key] <= high


def number(key: str, low: float, high: float, *, above_low: bool = False,
           below_high: bool = False) -> Rule:
    """A number between `low` and `high`; either end may be excluded."""
    def check(options: dict) -> bool:
        value = options.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return False
        low_ok = value > low if above_low else value >= low
        high_ok = value < high if below_high else value <= high
        return low_ok and high_ok
    return check


def positive(key: str) -> Rule:
    """A number above zero."""
    return lambda options: _is_number(options.get(key)) and options[key] > 0


def one_of(key: str, allowed: set[str]) -> Rule:
    """One of the listed words (a list or other unhashable value is simply wrong)."""
    return lambda options: isinstance(options.get(key), str) and options[key] in allowed


def bucket_range(options: dict) -> bool:
    """Khala's default length bucket lies within 0 and its maximum (≤ 20)."""
    default, maximum = options.get("default_bucket"), options.get("maximum_bucket")
    return (type(default) is int and type(maximum) is int
            and 0 <= default <= maximum <= KHALA_MAX_BUCKET)


MEMORY = positive("memory_reservation_mb")
GUIDANCE = number("cfg_scale", 0, 10, above_low=True)
TOP_K = whole("topk", 1, 1000)
TEMPERATURE = number("temperature", 0, 5)

# Engine → (message when its options fail, the rules they must pass).
ENGINE_RULES: dict[str, tuple[str, tuple[Rule, ...]]] = {
    "kokoro": ("Kokoro model options are incomplete",
               (text("language_code"), text("default_voice"), text("repo_id"))),
    "voxcpm": ("VoxCPM inference settings are invalid",
               (number("cfg_value", 0, 10, above_low=True),
                whole("inference_timesteps", 1, 100))),
    "khala": ("Khala length buckets are invalid", (bucket_range,)),
    "higgs": ("Higgs worker settings are invalid",
              (whole("worker_port", 1, 65535), MEMORY,
               number("mem_fraction_static", 0, 1, above_low=True, below_high=True))),
    "higgs_local": ("Higgs (transformers) settings are invalid", (MEMORY,)),
    "heartmula": ("HeartMuLa settings are invalid",
                  (text("version"), file_name("checkpoint_subdir"), TOP_K,
                   TEMPERATURE, GUIDANCE, MEMORY)),
    "yue2": ("YuE2 settings are invalid",
             (one_of("cot", {"full", "melody"}),
              number("memory_budget_gib", 4, 64), MEMORY)),
    "comfy_music": ("ComfyUI music settings are invalid",
                    (absolute_path("workflow"), file_names("component_subdirs"), MEMORY)),
    "mulacover": ("MuLaCover settings are invalid",
                  (file_name("checkpoint_subdir"), TOP_K, TEMPERATURE, GUIDANCE, MEMORY)),
    "levo2": ("LeVo 2 settings are invalid",
              (file_name("lm"), file_name("flow"), file_name("vae"),
               whole("steps", 1, 200), number("cfg", 0, 10, above_low=True), MEMORY)),
    "stableaudio3": ("Stable Audio 3 settings are invalid",
                     (whole("steps", 1, 100), GUIDANCE, MEMORY)),
}


def options_pass(engine: str, options) -> bool:
    """Whether one model's options satisfy its engine's rules."""
    if engine not in ENGINE_RULES:
        return True
    _, rules = ENGINE_RULES[engine]
    return isinstance(options, dict) and all(rule(options) for rule in rules)


# -- the report every checker writes into -------------------------------------

@dataclass
class _Report:
    config: Config
    engine_ids: set[str]
    check_workflows: bool
    errors: list[str] = field(default_factory=list)
    instance_ids: set[str] = field(default_factory=set)
    ports: set[int] = field(default_factory=set)
    repository_ids: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.errors.append(message)


# -- the chain ----------------------------------------------------------------

def _check_roots(report: _Report) -> None:
    roots = {item.id for item in report.config.model_roots}
    if report.config.download_root not in roots:
        report.add(f"Unknown download root: {report.config.download_root}")


def _check_repositories(report: _Report) -> None:
    report.repository_ids = [item.id for item in report.config.repositories]
    if len(report.repository_ids) != len(set(report.repository_ids)):
        report.add("Repository IDs must be unique")


def _check_manager_port(report: _Report) -> None:
    report.ports.add(report.config.port)
    if not 1 <= report.config.port <= MAX_PORT:
        report.add(f"Manager port must be between 1 and {MAX_PORT}")


def _check_instances(report: _Report) -> None:
    for item in report.config.instances:
        _check_identity(report, item)
        _check_engine_options(report, item)
        _check_placement(report, item)


def _check_identity(report: _Report, item) -> None:
    if item.id in report.instance_ids:
        report.add(f"Duplicate instance ID: {item.id}")
    report.instance_ids.add(item.id)
    if item.engine not in report.engine_ids:
        report.add(f"{item.id}: unknown engine {item.engine}")


def _check_engine_options(report: _Report, item) -> None:
    """The model's entry in its engine's option map, judged by `ENGINE_RULES`."""
    if item.engine not in MODEL_MAP_FIELDS:
        return
    field_name = MODEL_MAP_FIELDS[item.engine]
    configured = report.config.engines.get(item.engine, {}).get(field_name, {})
    model_name = item.model_id.rsplit("/", 1)[-1]
    if model_name not in configured:
        report.add(f"{item.id}: {item.engine} has no configured checkpoint")
        return
    options = configured[model_name]
    if not options_pass(item.engine, options):
        report.add(f"{item.id}: {ENGINE_RULES[item.engine][0]}")
    elif item.engine == "higgs":
        _claim_worker_port(report, item, options)


def _claim_worker_port(report: _Report, item, options: dict) -> None:
    """Higgs runs a second process on its own port, which must be free too."""
    worker_port = options["worker_port"]
    if worker_port == item.port or worker_port in report.ports:
        report.add(f"{item.id}: {ENGINE_RULES['higgs'][0]}")
    else:
        report.ports.add(worker_port)


def _check_placement(report: _Report, item) -> None:
    repository_id = item.model_id.split("/", 1)[0]
    if repository_id not in report.repository_ids:
        report.add(f"{item.id}: unknown repository {repository_id}")
    if item.port in report.ports:
        report.add(f"{item.id}: port {item.port} is already assigned")
    if not 1 <= item.port <= MAX_PORT:
        report.add(f"{item.id}: port must be between 1 and {MAX_PORT}")
    report.ports.add(item.port)


def _check_policies(report: _Report) -> None:
    for policy_name in ("gateway_policy", "media_policy", "provider_policy"):
        try:
            getattr(report.config, policy_name)
        except ValueError as error:
            report.add(str(error))


def _check_image_profiles(report: _Report) -> None:
    images = report.config.images
    workflow_root = Path(images.get("workflow_root", ""))
    for profile_id, profile in images.get("profiles", {}).items():
        _check_image_profile(report, profile_id, profile, workflow_root)


def _check_image_profile(report: _Report, profile_id: str, profile: dict,
                         workflow_root: Path) -> None:
    model_id = profile.get("model", "")
    if model_id not in report.instance_ids:
        report.add(f"Image profile {profile_id}: unknown instance {model_id}")
        return
    task = profile.get("task", "generation")
    if task not in {"generation", "edit"}:
        report.add(f"Image profile {profile_id}: unsupported task {task}")
        return
    _check_profile_task(report, profile_id, model_id, task)
    _check_profile_workflow(report, profile_id, profile.get("workflow", ""), workflow_root)


def _check_profile_task(report: _Report, profile_id: str, model_id: str, task: str) -> None:
    instance = report.config.instance(model_id)
    repository = report.config.repository(instance.model_id.split("/", 1)[0])
    expected = (Task.IMAGE_EDIT if task == "edit" else Task.IMAGE_GENERATION).value
    if repository.task != expected:
        report.add(f"Image profile {profile_id}: {model_id} is configured "
                   f"for {repository.task}, expected {expected}")


def _check_profile_workflow(report: _Report, profile_id: str, workflow: str,
                            workflow_root: Path) -> None:
    if not workflow or Path(workflow).name != workflow:
        report.add(f"Image profile {profile_id}: invalid workflow name")
    elif report.check_workflows and not (workflow_root / workflow).is_file():
        report.add(f"Image profile {profile_id}: workflow file is absent")


CHAIN = (_check_roots, _check_repositories, _check_manager_port,
         _check_instances, _check_policies, _check_image_profiles)


def validate_configuration(config: Config, engine_ids: set[str],
                           *, check_workflows: bool = False) -> None:
    """Reject broken references and conflicting ports with one useful report."""
    report = _Report(config, engine_ids, check_workflows)
    for checker in CHAIN:
        checker(report)
    if report.errors:
        raise ValueError("Invalid AI-Lab configuration:\n- " + "\n- ".join(report.errors))
