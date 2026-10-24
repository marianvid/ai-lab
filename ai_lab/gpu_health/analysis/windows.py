"""Finding the stretches of time the analysis is made of, one reading at a time.

Two kinds:
- a stable load window: the card works at an even power (within ±10 % of the
  window's mean, above 100 W) for at least 5 minutes. Its temperature, power
  and fan are the material of the thermal resistance R;
- an idle window: the card rests (below 25 W and below 5 % use) for at least
  10 minutes. Its median temperature is the day's resting temperature.

Readings are fed one by one as they arrive; a finished window comes out. A
gap of more than three readings (the monitor was stopped, or NVML did not
answer) ends a window, because nobody knows what the card did meanwhile.
"""

from __future__ import annotations

import statistics
import time

from ..settings import GpuHealthSettings

GAP_SAMPLES = 3
# A stretch needs a first and a last reading to have a length.
MIN_READINGS = 2


def day_of(ts: float) -> str:
    """The calendar day (YYYY-MM-DD, local time) a moment belongs to."""
    return time.strftime("%Y-%m-%d", time.localtime(ts))


class StableFinder:
    """Stable load windows, from readings fed in time order."""

    def __init__(self, settings: GpuHealthSettings) -> None:
        self._settings = settings
        self._run: list[dict] = []

    def feed(self, sample: dict) -> dict | None:
        """Take one reading; returns a finished window, or None."""
        finished = None
        if self._run and not self._continues(sample):
            finished = self._close()
        if self._loaded(sample):
            self._run.append(sample)
            if self._length() >= self._settings.stable_max_s:
                finished = self._close()
        return finished

    def _loaded(self, sample: dict) -> bool:
        power = sample.get("power_w")
        return power is not None and power > self._settings.stable_min_w

    def _continues(self, sample: dict) -> bool:
        """Whether `sample` belongs to the window being built."""
        last = self._run[-1]
        if sample["ts"] - last["ts"] > GAP_SAMPLES * self._settings.sample_s:
            return False
        if sample.get("model", "") != last.get("model", "") or not self._loaded(sample):
            return False
        powers = [item["power_w"] for item in self._run] + [sample["power_w"]]
        mean = statistics.fmean(powers)
        band = self._settings.stable_band * mean
        return max(powers) <= mean + band and min(powers) >= mean - band

    def _length(self) -> float:
        return self._run[-1]["ts"] - self._run[0]["ts"] if self._run else 0.0

    def _close(self) -> dict | None:
        run, self._run = self._run, []
        if len(run) < MIN_READINGS or run[-1]["ts"] - run[0]["ts"] < self._settings.stable_min_s:
            return None
        return summarize(run)


def summarize(run: list[dict]) -> dict:
    """A stable window's averages; the fan's spread; hotspot above chip, when known."""
    fans = known(run, "fan_pct")
    deltas = [item["temp_hotspot"] - item["temp_gpu"] for item in run
              if item.get("temp_hotspot") is not None and item.get("temp_gpu") is not None]
    return {
        "start": run[0]["ts"], "end": run[-1]["ts"], "day": day_of(run[0]["ts"]),
        "model": run[0].get("model", ""),
        "temp_avg": mean(known(run, "temp_gpu")),
        "power_avg": statistics.fmean(item["power_w"] for item in run),
        "fan_avg": mean(fans),
        "fan_std": statistics.pstdev(fans) if len(fans) >= MIN_READINGS else None,
        "hotspot_delta": mean(deltas),
    }


def known(rows: list[dict], name: str) -> list[float]:
    """The values of `name` that were read (not N/A)."""
    return [row[name] for row in rows if row.get(name) is not None]


def mean(values: list[float]) -> float | None:
    """The average, or None when there is nothing to average."""
    return statistics.fmean(values) if values else None


class IdleFinder:
    """Idle windows, from readings fed in time order."""

    def __init__(self, settings: GpuHealthSettings) -> None:
        self._settings = settings
        self._run: list[dict] = []

    def feed(self, sample: dict) -> dict | None:
        """Take one reading; returns a finished idle window, or None."""
        finished = None
        gap = self._run and sample["ts"] - self._run[-1]["ts"] > GAP_SAMPLES * self._settings.sample_s
        if self._run and (gap or not self._idle(sample)):
            finished = self._close()
        if self._idle(sample):
            self._run.append(sample)
            if self._run[-1]["ts"] - self._run[0]["ts"] >= self._settings.idle_max_s:
                finished = self._close()
        return finished

    def _idle(self, sample: dict) -> bool:
        power, use = sample.get("power_w"), sample.get("util_pct")
        if power is None or sample.get("temp_gpu") is None:
            return False
        quiet_use = use is None or use < self._settings.idle_max_util
        return power < self._settings.idle_max_w and quiet_use

    def _close(self) -> dict | None:
        run, self._run = self._run, []
        if len(run) < MIN_READINGS or run[-1]["ts"] - run[0]["ts"] < self._settings.idle_min_s:
            return None
        return {"start": run[0]["ts"], "end": run[-1]["ts"], "day": day_of(run[0]["ts"]),
                "temp_median": statistics.median(item["temp_gpu"] for item in run)}
