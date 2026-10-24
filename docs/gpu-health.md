# GPU health: watching the card for years

The card (NVIDIA RTX PRO 4500 Blackwell, 32 GB) is new. Cards age slowly:
dust fills the cooler, the thermal paste dries, a fan bearing wears, a memory
cell starts to fail. None of this shows from one day to the next. The GPU
health monitor writes down how the card behaves now, while it is new, and
compares every later week with that.

It runs inside the AI-Lab manager, in a thread of its own, only where an
NVIDIA driver is installed (CT 102). On the Mac the page only says so.

**It only reads.** It asks the driver for values (through NVML, the NVIDIA
driver's own library) and never changes a setting. It is not a program
running on the card, so it does not appear in `nvidia-smi`'s list of
processes, and it takes nothing from the models.

## Words used here

| Word | What it means |
|---|---|
| NVML | The library that comes with the NVIDIA driver; `nvidia-smi` uses it too. The monitor calls it directly (`ai_lab/gpu_health/nvml.py`), so AI-Lab needs nothing installed. |
| Reading (sample) | One set of values, taken every 15 seconds. |
| Stable load window | At least 5 minutes during which the card works at an even power: above 100 W, and within ±10 % of the window's average. Cut at 10 minutes. |
| Idle window | At least 10 minutes of rest: below 25 W and below 5 % use. Cut at 1 hour. |
| Resting temperature | The day's median chip temperature in idle windows. It follows the room. A day without one uses the last day that had one, up to 3 days back; after that the day has "no reference". |
| R, thermal resistance | `(temperature in a stable window − resting temperature that day) ÷ power`, in °C per watt. How hard it is for heat to leave the chip. Because the resting temperature is measured in the same room on the same day, the room's warmth mostly cancels out. R going up over months means the cooling is getting worse. |
| Reference (baseline) | The median R of the first 21 days, once there are at least 10 stable windows (if 21 days give fewer, it waits for the 10th). Saved with its dates. It changes only when somebody resets it, with a note. |
| Trend | The median R of the last 14 days. |
| Power band | Windows grouped by power in 25 W steps (100–125 W, 125–150 W, ...). The fan is compared only at the same power. |
| Throttling | The card slowing itself down. Because of its power limit (power cap) this is normal work and never an alarm; because of heat or a hardware signal it is an alarm. |
| ECC | The memory's error correction. A *corrected* error was fixed; an *uncorrectable* one was not. |
| Row remapping | The card replacing a bad memory row with a spare one. *Pending*: waiting for a reset of the card. *Failure*: no spare left. |
| Xid | An error number the NVIDIA driver writes in the kernel log. Some mean the card or its memory is failing (48, 63, 64, 74, 79, 92, 94, 95: critical); the others are warnings. |
| PCIe, replays, AER | The link between card and computer. A *replay* is a packet sent again; *AER* is the PCIe error report in the kernel log. |
| Host helper | A small program on the Proxmox host that reads the kernel log for the container, which may not read it (see below). |

## What is read, and how often

Every 15 seconds: chip temperature, memory temperature, hotspot, power and
power limit, fan, chip and memory clocks, use, throttle reasons, PCIe
generation and width, PCIe replays, and from AI-Lab the model the gateway has
on the card and whether a request is running.

Every hour: memory errors (corrected and uncorrectable, since the driver
started and over the card's life), replaced memory rows, the driver version.

All the time, when the host helper is installed: Xid and PCIe AER lines from
the kernel log.

**Not available on this card** (measured on CT 102, 24 Oct 2026): hotspot
and memory temperature. NVML gives neither for this card. They show "N/A on
this card" and are left out of every rule; the columns stay so that a card
which has them needs no change. The PCIe link runs at Gen 1 at rest and goes
up under load; x4 is normal on this OcuLink cable.

## Alarms

Immediate:

| Alarm | Level |
|---|---|
| A new uncorrectable memory error | critical |
| A newly replaced memory row: uncorrectable / correctable | critical / alarm |
| A row replacement pending / failed | alarm / critical |
| A new Xid | critical for 48, 63, 64, 74, 79, 92, 94, 95; warning otherwise |
| PCIe link below normal, under load, for more than 1 minute | alarm |
| More than 100 PCIe replays within an hour | warning |
| PCIe AER uncorrectable error | critical |
| Thermal or hardware slowdown for more than 30 seconds | alarm |
| Chip above 87 °C or memory above 95 °C for more than 1 minute | alarm |
| The power limit changed | warning |
| No answer from NVML for more than 5 minutes | alarm |
| No heartbeat from the host helper for more than 2 hours | warning |

"Normal" for the PCIe link is the best generation and width seen under load
since the monitor started (under load: above 50 W and some use). At rest the
link drops by itself, which is not a fault.

Slow (checked every 5 minutes, over the last 14 days):

| Alarm | Level |
|---|---|
| R 15 % above its reference / 25 % above | warning / alarm |
| Fan 15 points higher than in the reference, at the same power band | warning |
| Fan spread (standard deviation) more than 3× the reference's | warning |
| Hotspot more than 20 °C above the chip (not on this card: no hotspot) | warning |
| More than 10 corrected memory errors in a week | warning |
| Resting temperature rising over 30 days (at least 1.5 °C) | information only |

Each is checked for all models together, and for each model that has at
least 10 windows of its own (its own reference forms from its own first 21
days).

**Repeats.** A lasting alarm is repeated at most once every 6 hours. When the
condition ends, a "Resolved" message follows; if it comes back after that, or
gets worse (warning → alarm → critical), it is sent at once. Critical alarms are never held
back: every critical event is sent, and a critical condition that lasts is
repeated every 15 minutes. Everything, sent or held back, is written to the
events list.

**Where alarms go.** No notification channel exists yet in AI-Lab or in the
home lab, so the only channel is `log`: the events list on the page, and one
line per alarm in the manager's log:

    journalctl -u ai-lab-manager | grep "GPU health"

A real channel (Telegram, e-mail, ...) is the owner's choice; it is a new
value of `channel` and nothing else changes. The "Send test alarm" button
shows that the path works.

## Where it is kept

One SQLite file in AI-Lab's data folder: `/var/lib/ai-lab/gpu-health.sqlite`
on CT 102. Kept for years: readings older than 90 days are folded into
5-minute rows (minimum, maximum, average); windows, events and references are
never deleted. Roughly 1 MB a day of raw readings, so 90 days is about 100 MB;
the folded rows add about 30 MB a year.

## The page, the endpoint, the command

- The **GPU health** tab: the card now, the alarms in force, the reference
  and the trend, the last 24 hours as three small charts, memory and kernel
  log, the events, and two buttons (send a test alarm; reset the reference,
  with a note saying why — for example after cleaning the cooler).
- The endpoint, for anything that wants the numbers:

  | Address | What |
  |---|---|
  | `GET /api/gpu-health` | Everything the page shows |
  | `GET /api/gpu-health/history?hours=24` | Up to 360 averaged points for the charts |
  | `GET /api/gpu-health/report?days=7` | The report of the last days |
  | `POST /api/gpu-health/baseline/reset` with `{"note": "..."}` | Reset the reference |
  | `POST /api/gpu-health/test-alarm` | Send one test alarm |

- The command, in the container: `/opt/ai-lab/.venv/bin/ai-lab-gpu-health`
  (status) or `... ai-lab-gpu-health report --days 30`. It asks the running manager on
  this machine on port 8090 (`--url` for another); `--json` prints the raw answer.
- The weekly report is made every 7 days, written to the events and sent
  through the channel.

## Settings

The `gpu_health` section of `config.json`. Every number above is a setting
with that value as its default; an unknown key is refused, so a typo does not
silently keep a default. The ones you are most likely to touch:

| Key | Default | What |
|---|---|---|
| `enabled` | `true` | Switch the monitor off |
| `helper_events_path` | `""` | The host helper's file inside the container; empty: no helper |
| `channel` | `"log"` | Where alarms go |
| `temp_gpu_c`, `temp_mem_c` | `87`, `95` | Temperature alarms |
| `repeat_s`, `critical_repeat_s` | `21600`, `900` | How often a lasting alarm repeats |
| `raw_keep_days` | `90` | How long raw readings are kept before folding |

The rest (`sample_s`, `stable_*`, `idle_*`, `baseline_*`, `trend_days`,
`r_warn`, `r_alarm`, `fan_*`, `hotspot_delta_c`, `ecc_corrected_week`,
`idle_rise_*`, `link_*`, `replays_per_hour`, `nvml_silent_s`, `xid_critical`,
`helper_heartbeat_s`, `pci_address`) is listed in
`ai_lab/gpu_health/settings.py`, each with its meaning.

## The host helper (Xid and AER)

Inside CT 102 the kernel log cannot be read: `dmesg` is refused even to root,
`/dev/kmsg` does not exist, `journalctl -k` is empty. So the Xid and AER
alarms need a helper on the Proxmox host. Until it is installed, the page
says "not watched: the host helper is not installed" and every other alarm
works.

The helper is `ai_lab/gpu_health/kernel/host_helper.py` (Python 3, standard library
only, nothing from AI-Lab) with its unit `system/gpu-health/ai-lab-gpu-kernel-log.service`.
It follows `journalctl -k`, keeps the Xid and AER lines, writes them as JSON
lines to `/var/lib/ai-lab-gpu-helper/events.jsonl`, adds a heartbeat every
hour, and rotates the file at 10 MB. It does not run as root: systemd gives
it a throw-away user that may read the journal and write only its own folder,
with no network.

Installing it, on the Proxmox host, as root:

1. Copy the script to `/usr/local/lib/ai-lab/host_helper.py` and
   the unit to `/etc/systemd/system/ai-lab-gpu-kernel-log.service`.
2. `systemctl daemon-reload && systemctl enable --now ai-lab-gpu-kernel-log`
3. Give the container the folder, read-only. **This restarts nothing by
   itself, but the mount only appears after the container restarts**, which
   stops every model on the card for a minute or two:
   `pct set 102 -mp1 /var/lib/private/ai-lab-gpu-helper,mp=/var/lib/ai-lab-gpu-helper,ro=1`
   (`/var/lib/private/...` is where systemd really keeps the folder of a
   throw-away user; `/var/lib/ai-lab-gpu-helper` on the host is a link to it)
   (use a free `mpN` number), then restart CT 102 when the card is idle.
4. In CT 102's `config.json`: `"gpu_health": {"helper_events_path":
   "/var/lib/ai-lab-gpu-helper/events.jsonl"}`, then restart the manager.

The page then shows the helper's last heartbeat.

## Code

`ai_lab/gpu_health/`: `nvml.py` (the NVML calls), `readings.py` (one
reading), `store.py` (the SQLite file), `alarms.py` (levels, repeats,
"resolved"), `steps.py` and `monitor.py` (the thread), `service.py` (what the
manager starts), `settings.py`;
`analysis/`: `windows.py` (stable and idle windows), `checks.py` (immediate
rules), `trend.py` and `references.py` (resting temperature, R, reference,
trend);
`kernel/`: `events.py` (the helper's file), `host_helper.py` (the helper itself);
`output/`: `report.py`, `status.py`, `cli.py` (the command). The page is `web/js/views/gpu_health.js`.
