"""The list of settings the Settings page shows: one `Field` each (`setting_fields.py` saves them).

Kept apart from the saving so each file stays one subject. See `setting_fields.py`
for the owner's rule these follow.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import policy
from ..config.store import Config
from ..network import ALL_INTERFACES, LOOPBACK
from ..gpu_health.settings import BOUNDS as MONITOR_BOUNDS
from ..gpu_health.settings import GpuHealthSettings

# The top of the file's own defaults (`title`, `host`, `port`).
_FILE = Config()
FIXED = "{label} is shown, not changed, here: change it in config.json"

# The groups the page shows, in order: the section title of each.
GROUPS = {
    "manager": "This AI-Lab",
    "loading": "Loading models",
    "updates": "Engine updates",
    "embeddings": "Embeddings lane",
    "clients": "Client caps",
    "uploads": "Uploaded images",
    "gpu_readings": "GPU health: readings",
    "gpu_windows": "GPU health: stable and resting windows",
    "gpu_trend": "GPU health: reference and trend",
    "gpu_alarms": "GPU health: alarms",
}
RESTART = "Takes effect when AI-Lab is next started."


@dataclass(frozen=True)
class Field:
    """One setting: where it is kept, how the page shows it, and what it means."""

    key: str
    group: str
    label: str
    unit: str
    default: float | int | str | bool | list
    low: float | None
    high: float | None
    help: str
    # "number", "integer", "text", "boolean", "list", or "fixed": shown, not
    # changed here (a value that, saved wrong from the page, would lock it out).
    kind: str = "number"
    restart: bool = True
    empty: bool = False             # a text that may be left empty


def _seconds(key: str, group: str, label: str, default: float, help_text: str) -> Field:
    """A wait in seconds, from one second to a week (`policy.LONGEST_S`), as its parser takes."""
    return Field(key, group, label, "s", default, 1, policy.LONGEST_S, help_text)


# The monitor's own defaults, so the fields hold no copy of them.
_MONITOR = GpuHealthSettings()


def _gpu(name: str, shown: tuple[str, str, str], help_text: str,
         kind: str = "number") -> Field:
    """One number of the card's health monitor, shown as (group, label, unit).

    Its range is the monitor's own (`gpu_health.settings.BOUNDS`), so the page
    and the start refuse the same values.
    """
    group, label, unit = shown
    low, high = MONITOR_BOUNDS[name]
    return Field(f"gpu_health.{name}", group, label, unit, getattr(_MONITOR, name), low, high,
                 help_text, kind=kind)


FIELDS: tuple[Field, ...] = (
    Field("title", "manager", "Name", "", _FILE.title, None, None,
          "The name in the browser tab, beside the machine's own (Linux, macOS), so two "
          "AI-Labs open side by side are told apart. Any short text; too long a name "
          "crowds the tab. Taken at once.", kind="text", restart=False),
    Field("host", "manager", "Listens on", "", _FILE.host, None, None,
          f"The network address AI-Lab answers on: {ALL_INTERFACES} is every interface of "
          f"this machine, {LOOPBACK} only itself. Change it in config.json, not here: a wrong "
          "value saved from this page would lock this page out.", kind="fixed"),
    Field("port", "manager", "Port", "", _FILE.port, None, None,
          "The port of this page and of every API call (the gateway). Change it in "
          "config.json, not here: a wrong value would lock this page out, and every "
          "client would need the new port.", kind="fixed"),
    _seconds("limits.runtime.start_s", "loading", "Process start", policy.START_S,
             "How long a model's engine process may take to appear after it is started. "
             "Too small: an engine that starts slowly is given up on and the load fails. "
             "Too large: a process that never appears is waited on for that long."),
    _seconds("limits.runtime.load_s", "loading", "Model load", policy.LOAD_S,
             "How long the engine may take to answer once started: reading the weights, "
             "a first compile. Too small: a large model, a slow disk or a first compile "
             "fails its load. Too large: a load that is stuck holds the card that long."),
    _seconds("limits.runtime.unload_s", "loading", "Model unload", policy.UNLOAD_S,
             "How long an engine may take to exit when stopped, and again for its memory "
             "to come back. Too small: a stop is reported as failed while memory is still "
             "being freed. Too large: a stuck stop holds the next load that long."),
    Field("limits.runtime.quiet_mb", "loading", "Quiet card", "MB", policy.QUIET_MB,
          *policy.QUIET_MB_BOUNDS,
          "How little memory may still be in use on the card for it to count as free "
          "before a new model loads (a driver keeps a little of its own: about 2 MB on "
          "CT 102). Too small: a load waits for memory that never comes back and fails. "
          "Too large: a model loads while another one's memory is still being freed."),
    _seconds("limits.runtime.quiet_s", "loading", "Wait for a quiet card", policy.QUIET_TIMEOUT_S,
             "How long a load waits for the card to become quiet (above). Too small: a "
             "driver that frees memory slowly fails the load with “did not exit”. "
             "Too large: a card that never frees keeps the waiting request that long."),
    Field("limits.runtime.quiet_poll_s", "loading", "Look at the card every", "s",
          policy.QUIET_POLL_S, *policy.QUIET_POLL_BOUNDS,
          "How often the card is read while waiting for it to become quiet. Too small: "
          "many readings of the card for nothing. Too large: a load starts that much later "
          "than it could."),
    Field("limits.first_port", "loading", "First port offered", "", policy.FIRST_PORT,
          1, 65535, "The first port suggested when a model entry is added; the next free "
          "one above it is offered. Only a suggestion: each entry's port can be changed. "
          "Pick a range nothing else on this machine uses.", kind="integer", restart=False),
    Field("limits.editor_port_offset", "loading", "Editor page offset", "",
          policy.EDITOR_PORT_OFFSET, 1000, 50000, "How far above a model's port its own "
          "page listens (ComfyUI, the speech editors, the music studios): the Create, "
          "Speak and Music links open that port. An offset that puts a page on another "
          "model's port, or past 65535, is refused (too small or too large). On CT 102 the firewall opens only "
          "18124 (YuE2 Studio); another offset needs home-lab to open the new port.",
          kind="integer"),
    _seconds("limits.builds.install_s", "updates", "Install step", policy.INSTALL_S,
             "How long one install command (a package install, a ComfyUI step) may run. "
             "Too small: a slow download is cut and the install fails, the old version "
             "kept. Too large: a hung install shows \u201cinstalling\u201d that long."),
    _seconds("limits.builds.build_s", "updates", "Compile step", policy.BUILD_S,
             "How long one compile step of an engine built from source (llama.cpp) may run. "
             "Too small: a full rebuild is cut and the old build kept. Too large: a hung "
             "compile shows \u201cbuilding\u201d that long."),
    _seconds("limits.builds.check_every_s", "updates", "Check for updates every",
             policy.UPDATE_CHECK_EVERY_S,
             "How often upstream is asked whether a newer version exists, so the page "
             "already knows. Too small: upstream is asked needlessly often. Too large: a new "
             "version is noticed late."),
    _seconds("limits.builds.builds_first_check_s", "updates", "First check, source builds",
             policy.BUILDS_FIRST_CHECK_S,
             "How long after start the source builds first ask upstream, so starting is not "
             "held up by the network. Kept apart from the installs' first check."),
    _seconds("limits.builds.installs_first_check_s", "updates", "First check, installs",
             policy.INSTALLS_FIRST_CHECK_S,
             "How long after start the package installs first ask the index. Kept apart "
             "from the source builds' first check, so the two do not ask at once."),
    Field("limits.builds.system_path", "updates", "Search path for installs", "",
          policy.SYSTEM_PATH, None, None,
          "Where the programs a new engine environment is built with (python, git, uv, "
          "compilers) are looked for: absolute folders joined by \u201c:\u201d. Wrong: an "
          "install or update cannot find its tools and fails, the old version kept.",
          kind="text"),
    _seconds("limits.embeddings.start_s", "embeddings", "Server start",
             policy.EMBEDDINGS_START_S,
             "How long an embeddings model's server may take to answer after it starts. Too "
             "small: a model loading from a slow disk fails its first call. Too large: a "
             "server that never comes up holds its first call that long."),
    _seconds("limits.embeddings.stop_s", "embeddings", "Server stop", policy.EMBEDDINGS_STOP_S,
             "How long that server may take to exit when asked before it is killed. Too "
             "small: it is killed mid-cleanup. Too large: stopping waits that long."),
    _seconds("limits.embeddings.drain_s", "embeddings", "Leave the card within",
             policy.EMBEDDINGS_DRAIN_S,
             "When a card model needs the card, how long the embeddings calls already on it "
             "may take to finish before their server is stopped anyway. Too small: those "
             "calls fail. Too large: the card model's load waits that long."),
    _seconds("limits.embeddings.hold_s", "embeddings", "Stay off the card for",
             policy.EMBEDDINGS_HOLD_S,
             "A safety net: if a card load never says it is done, the card is offered to the "
             "embeddings again after this long. Too small: embeddings may return while the "
             "load still needs the room. Too large: they stay on the processor that long."),
    _seconds("limits.embeddings.call_s", "embeddings", "One call", policy.EMBEDDINGS_CALL_S,
             "How long one embeddings call to its server may take, the queue wait not "
             "counted. Too small: a large batch of long texts fails. Too large: a stuck call "
             "holds its place that long."),
    Field("limits.clients.chars_per_token", "clients", "Characters per token", "",
          policy.CHARS_PER_TOKEN, 0.5, 20,
          "A capped client's prompt is counted as its characters divided by this, before "
          "the model sees it. Smaller counts more tokens (safer for the pool, refuses "
          "sooner); larger counts fewer (lets more in, may overfill the pool). English runs "
          "about 4, code about 3."),
    Field("limits.clients.retry_after_s", "clients", "Retry after", "s", policy.RETRY_AFTER_S,
          1, 3600, "How long a client over its cap is told to wait (the Retry-After header). "
          "Too small: it retries and is refused again. Too large: it waits longer than "
          "needed.", kind="integer"),
    Field("gateway.max_upload_bytes", "uploads", "Largest file", "bytes", 0, 0, 1024 ** 3,
          "The largest image a request may upload (OCR, an image edit); a larger one is "
          "refused before it is decoded. 0: no limit. Too small: ordinary photos are "
          "refused. Too large (or 0): one huge upload is held in memory whole.",
          kind="integer"),
    Field("gateway.max_upload_pixels", "uploads", "Most pixels", "pixels", 0, 0, 1024 ** 3,
          "The most pixels (width times height) an uploaded image may have. 0: no limit. "
          "Too small: large scans and photos are refused. Too large (or 0): a small file "
          "that decodes to a giant picture can fill the memory.", kind="integer"),
    Field("gateway.max_upload_dimension", "uploads", "Longest side", "px", 0, 0, 100000,
          "The longest side an uploaded image may have. 0: no limit. Too small: tall "
          "receipts and wide panoramas are refused. Too large (or 0): one very long side "
          "is accepted however thin the image.", kind="integer"),
    # -- the card's health monitor (`gpu_health/settings.py`, docs/gpu-health.md) --
    # Its parser takes any number that is not negative, so the fields do too.
    Field("gpu_health.enabled", "gpu_readings", "Monitor on", "", _MONITOR.enabled, None, None,
          "Whether the card's health is watched at all. Off: no readings, no alarms, and "
          "the page says nobody is watching the card. Only a machine without an NVIDIA "
          "card has no use for it.", kind="boolean"),
    _gpu("sample_s", ("gpu_readings", "Read every", "s"),
         "How often the card is read (temperature, power, fan, clocks). Too small: many "
         "readings and a busier history for nothing. Too large: a short overheat or "
         "slowdown happens between two readings and is missed."),
    _gpu("slow_s", ("gpu_readings", "Slow readings every", "s"),
         "How often the memory error counters and the driver's state are read. Too small: "
         "reads that change only rarely, repeated. Too large: a new memory error is "
         "noticed that much later."),
    _gpu("nvml_silent_s", ("gpu_readings", "Card silent alarm after", "s"),
         "How long the driver may not answer before that is an alarm. Too small: a "
         "short hiccup alarms. Too large: a card that stopped answering goes unnoticed "
         "that long."),
    _gpu("raw_keep_days", ("gpu_readings", "Keep raw readings", "days"),
         "How long every 15-second reading is kept before it is folded into 5-minute "
         "rows. Too small: the detail of a past problem is gone when you look. Too "
         "large: the history file grows.", kind="integer"),
    Field("gpu_health.pci_address", "gpu_readings", "Card's PCI address", "",
          _MONITOR.pci_address, None, None,
          "The card's address on the PCIe bus: a PCIe error line in the kernel log counts "
          "only when it names this card. Wrong: the card's PCIe errors are never counted, "
          "or another device's are.", kind="text"),
    Field("gpu_health.helper_events_path", "gpu_readings", "Host helper's file", "",
          _MONITOR.helper_events_path, None, None,
          "The file the host helper writes the kernel log's card lines to, as the "
          "container sees it. Empty: no helper, and the Xid and PCIe error alarms say "
          "they are not watched. A wrong path: the helper's heartbeat alarm.",
          kind="text", empty=True),
    _gpu("helper_heartbeat_s", ("gpu_readings", "Helper silent alarm after", "s"),
         "How long the host helper may write nothing (it writes a heartbeat every hour) "
         "before that is a warning. Too small: a late heartbeat warns. Too large: a "
         "stopped helper goes unnoticed that long."),
    _gpu("stable_min_s", ("gpu_windows", "Stable window, shortest", "s"),
         "How long the card must work at an even power for a stable window, the kind the "
         "thermal resistance is measured in. Too small: the chip has not yet warmed up and "
         "R reads low. Too large: few windows, and the reference takes longer."),
    _gpu("stable_max_s", ("gpu_windows", "Stable window, longest", "s"),
         "A long even load is cut into windows of at most this. Too small: windows too "
         "short to have warmed up. Too large: one long job counts as only a few windows."),
    _gpu("stable_band", ("gpu_windows", "Stable power band", "×"),
         "How far (as a fraction of the window's average) the power may wander and still "
         "be even: 0.10 is ±10 %. Too small: hardly any window is stable. Too large: "
         "uneven work is measured as if even, and R is noisy."),
    _gpu("stable_min_w", ("gpu_windows", "Stable window, least power", "W"),
         "A stable window must draw at least this. Too small: light work, where R says "
         "little, is measured. Too large: only the heaviest jobs count."),
    _gpu("idle_max_w", ("gpu_windows", "Idle, most power", "W"),
         "Below this power (and the use below) the card is resting, for the day's resting "
         "temperature. Too small: a resting card is not seen as resting. Too large: a card "
         "doing light work is taken for resting and the room reads warm."),
    _gpu("idle_max_util", ("gpu_windows", "Idle, most use", "%"),
         "Below this use (and the power above) the card is resting. Too small: background "
         "use of a few percent keeps every rest from counting. Too large: light work "
         "counts as rest."),
    _gpu("idle_min_s", ("gpu_windows", "Idle window, shortest", "s"),
         "How long a rest must last to give a resting temperature. Too small: a card still "
         "cooling from work is taken as resting. Too large: a busy day gives no rest."),
    _gpu("idle_max_s", ("gpu_windows", "Idle window, longest", "s"),
         "A long rest is cut into windows of at most this, so a quiet day still gives its "
         "temperature that day. Too small: many short windows. Too large: a day-long rest "
         "gives one window, late."),
    _gpu("idle_fallback_days", ("gpu_windows", "Borrow a resting temperature for", "days"),
         "A day with no rest uses the last day that had one, up to this many days back; "
         "after that the day has no reference. Too small: busy days go unmeasured. Too "
         "large: a weeks-old room temperature is used.", kind="integer"),
    _gpu("baseline_days", ("gpu_trend", "Reference over", "days"),
         "The reference R is the median of the first this-many days (with at least the "
         "windows below). Too small: one unusual week becomes the reference. Too large: "
         "no comparison for that long.", kind="integer"),
    _gpu("baseline_min_windows", ("gpu_trend", "Reference, fewest windows", ""),
         "The reference waits for at least this many stable windows. Too small: a "
         "reference from a handful of jobs. Too large: on a quiet machine it never forms.",
         kind="integer"),
    _gpu("trend_days", ("gpu_trend", "Trend over", "days"),
         "The trend is the median R of the last this-many days, compared with the "
         "reference. Too small: a noisy trend that alarms on one hot week. Too large: a "
         "real worsening shows late.", kind="integer"),
    _gpu("r_warn", ("gpu_trend", "R warning above reference", "×"),
         "The trend this much above the reference is a warning: 0.15 is 15 %. Too small: "
         "ordinary variation warns. Too large: cooling that is getting worse is not "
         "flagged."),
    _gpu("r_alarm", ("gpu_trend", "R alarm above reference", "×"),
         "The trend this much above the reference is an alarm: 0.25 is 25 %. Too small: "
         "alarms for what is still a warning. Too large: bad cooling alarms late."),
    _gpu("fan_warn_points", ("gpu_trend", "Fan warning above reference", "points"),
         "The fan this many percentage points faster than in the reference, at the same "
         "power, is a warning. Too small: normal fan changes warn. Too large: a fan "
         "working harder to cool the same load is not flagged."),
    _gpu("fan_band_w", ("gpu_trend", "Fan power band", "W"),
         "Windows are grouped in power bands this wide; the fan is compared only within "
         "one. Too small: too few windows in each band. Too large: different loads are "
         "compared."),
    _gpu("fan_std_factor", ("gpu_trend", "Fan spread warning", "×"),
         "The fan's spread (standard deviation) more than this many times the reference's "
         "is a warning. Too small: ordinary unevenness warns. Too large: a fan hunting up "
         "and down is not flagged."),
    _gpu("hotspot_delta_c", ("gpu_trend", "Hotspot above chip", "°C"),
         "The hotspot this much above the chip temperature is a warning (a card that "
         "reports a hotspot; this one does not). Too small: warns at normal spreads. Too "
         "large: bad paste contact is not flagged."),
    _gpu("ecc_corrected_week", ("gpu_trend", "Corrected memory errors per week", ""),
         "More corrected memory errors than this in a week is a warning. Too small: warns "
         "on the odd error every card has. Too large: memory that is starting to fail is "
         "not flagged.", kind="integer"),
    _gpu("idle_rise_days", ("gpu_trend", "Resting rise over", "days"),
         "The resting temperature is watched for a rise over this many days (for "
         "information only). Too small: the seasons look like a fault. Too large: a rise "
         "shows late.", kind="integer"),
    _gpu("idle_rise_c", ("gpu_trend", "Resting rise of at least", "°C"),
         "A rise of the resting temperature by at least this over those days is noted. "
         "Too small: noise is noted. Too large: a real rise is not."),
    _gpu("temp_gpu_c", ("gpu_alarms", "Chip too hot", "°C"),
         "The chip above this for the time below is an alarm. Too small: ordinary heavy "
         "work alarms. Too large: the card is near its own slowdown before anyone hears."),
    _gpu("temp_mem_c", ("gpu_alarms", "Memory too hot", "°C"),
         "The memory above this for the time below is an alarm. Too small: heavy work "
         "alarms. Too large: memory near its limit is not flagged."),
    _gpu("temp_hold_s", ("gpu_alarms", "Too hot for", "s"),
         "How long a temperature must stay above its limit to alarm. Too small: a short "
         "peak alarms. Too large: a real overheat alarms late."),
    _gpu("throttle_hold_s", ("gpu_alarms", "Heat slowdown for", "s"),
         "How long the card may slow itself for heat or a hardware signal before that is "
         "an alarm (a power-cap slowdown never is). Too small: a blip alarms. Too large: a "
         "card held back by heat alarms late."),
    _gpu("link_hold_s", ("gpu_alarms", "Slow PCIe link for", "s"),
         "How long the PCIe link may run below its normal speed, under load, before that "
         "is an alarm. Too small: a moment of renegotiation alarms. Too large: a "
         "degraded link alarms late."),
    _gpu("link_load_w", ("gpu_alarms", "PCIe link checked above", "W"),
         "The link speed is judged only when the card draws at least this (a resting "
         "card lowers its link on purpose). Too small: a resting card's slow link alarms. "
         "Too large: a bad link under light work is missed."),
    _gpu("replays_per_hour", ("gpu_alarms", "PCIe replays per hour", ""),
         "More packets sent again than this within an hour is a warning. Too small: the "
         "few replays every link has warn. Too large: a failing riser or slot is not "
         "flagged.", kind="integer"),
    Field("gpu_health.xid_critical", "gpu_alarms", "Critical Xid numbers", "",
          list(_MONITOR.xid_critical), None, None,
          "The driver's error numbers that mean the card or its memory may be failing, "
          "comma-separated: reported as critical, any other Xid as a warning. Fewer: a "
          "serious error is only a warning. More: a harmless one is critical.",
          kind="list"),
    Field("gpu_health.channel", "gpu_alarms", "Alarms go to", "", _MONITOR.channel, None, None,
          "Where an alarm is said. Only \u201clog\u201d exists: the event list on the GPU "
          "health page and the manager's service log (journalctl -u ai-lab-manager). "
          "A second channel would be a change of AI-Lab, not of this setting.", kind="fixed"),
    _gpu("repeat_s", ("gpu_alarms", "Repeat an alarm every", "s"),
         "How often an alarm whose condition lasts is said again. Too small: the log fills "
         "with the same alarm. Too large: a lasting problem is said once and forgotten."),
    _gpu("critical_repeat_s", ("gpu_alarms", "Repeat a critical alarm every", "s"),
         "The same, for a critical alarm. Too small: floods the log. Too large: a "
         "critical state is easy to miss."),
)
BY_KEY = {field.key: field for field in FIELDS}


