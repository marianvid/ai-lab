"""Where the monitor keeps everything, for years: one SQLite file in AI-Lab's data folder.

Tables:
- `samples`: every 15-second reading;
- `samples_5m`: readings older than `raw_keep_days`, folded into 5-minute rows
  (minimum, maximum, mean) — the only thing that ever replaces data;
- `windows`: stable load windows (the material of the thermal resistance);
- `idle_windows`: quiet stretches (the material of the day's resting temperature);
- `events`: alarms, their resolution, Xid, memory errors, PCIe, throttling;
- `slow_readings`: the hourly values (memory error counters, replaced rows, driver);
- `baselines`: the reference values, global and per model, with their date;
- `state`: small values that must survive a restart (last counters, offsets).

One connection, used under a lock: the monitor's thread writes, the page reads.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .readings import Sample

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
  ts REAL, temp_gpu REAL, temp_mem REAL, temp_hotspot REAL, power_w REAL,
  power_limit_w REAL, fan_pct REAL, clock_sm REAL, clock_mem REAL, util_pct REAL,
  throttle INTEGER, pcie_gen INTEGER, pcie_width INTEGER, pcie_replays INTEGER,
  model TEXT, running INTEGER);
CREATE INDEX IF NOT EXISTS samples_ts ON samples(ts);
CREATE TABLE IF NOT EXISTS samples_5m (
  ts REAL PRIMARY KEY, n INTEGER,
  temp_min REAL, temp_max REAL, temp_avg REAL,
  power_min REAL, power_max REAL, power_avg REAL,
  fan_min REAL, fan_max REAL, fan_avg REAL, util_avg REAL);
CREATE TABLE IF NOT EXISTS windows (
  start REAL, end REAL, day TEXT, model TEXT, temp_avg REAL, power_avg REAL,
  fan_avg REAL, fan_std REAL, hotspot_delta REAL);
CREATE INDEX IF NOT EXISTS windows_day ON windows(day);
CREATE TABLE IF NOT EXISTS idle_windows (start REAL, end REAL, day TEXT, temp_median REAL);
CREATE TABLE IF NOT EXISTS events (
  ts REAL, kind TEXT, level TEXT, key TEXT, message TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS baselines (
  scope TEXT PRIMARY KEY, r REAL, fan_std REAL, fan_bands TEXT, windows INTEGER,
  period_start TEXT, period_end TEXT, formed_at REAL, note TEXT);
CREATE TABLE IF NOT EXISTS slow_readings (ts REAL, ecc TEXT, rows TEXT, driver TEXT);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
"""
SAMPLE_COLUMNS = tuple(Sample.__dataclass_fields__)
WINDOW_COLUMNS = ("start", "end", "day", "model", "temp_avg", "power_avg",
                  "fan_avg", "fan_std", "hotspot_delta")
FIVE_MINUTES = 300
TABLES = frozenset({"samples", "samples_5m", "windows", "idle_windows", "events",
                    "baselines", "slow_readings"})


class Store:
    """The monitor's SQLite file."""

    def __init__(self, path: Path | str) -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            # WAL: the status command reads the file while the monitor writes it.
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)
            self._db.commit()

    def close(self) -> None:
        """Close the file."""
        with self._lock:
            self._db.close()

    # -- writing --------------------------------------------------------------

    def add_sample(self, sample: Sample) -> None:
        """One 15-second reading."""
        row = sample.row()
        self._write(f"INSERT INTO samples ({', '.join(SAMPLE_COLUMNS)}) VALUES "
                    f"({', '.join('?' for _ in SAMPLE_COLUMNS)})",
                    [row[name] for name in SAMPLE_COLUMNS])

    def add_window(self, window: dict) -> None:
        """One stable load window."""
        self._write(f"INSERT INTO windows ({', '.join(WINDOW_COLUMNS)}) VALUES "
                    f"({', '.join('?' for _ in WINDOW_COLUMNS)})",
                    [window.get(name) for name in WINDOW_COLUMNS])

    def add_idle_window(self, start: float, end: float, day: str, temp_median: float) -> None:
        """One quiet stretch and its median chip temperature."""
        self._write("INSERT INTO idle_windows VALUES (?, ?, ?, ?)", [start, end, day, temp_median])

    def add_event(self, event: dict) -> None:
        """One event: {ts, kind, level, key, message, detail (any JSON value)}."""
        self._write("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                    [event["ts"], event["kind"], event.get("level", "info"),
                     event.get("key", ""), event.get("message", ""),
                     json.dumps(event.get("detail"))])

    def add_slow_reading(self, ts: float, ecc: dict, rows: dict | None, driver: str) -> None:
        """One hourly reading."""
        self._write("INSERT INTO slow_readings VALUES (?, ?, ?, ?)",
                    [ts, json.dumps(ecc), json.dumps(rows), driver])

    def save_baseline(self, baseline: dict) -> None:
        """A reference value for one scope ("global" or a model id)."""
        self._write("INSERT OR REPLACE INTO baselines VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [baseline["scope"], baseline["r"], baseline.get("fan_std"),
                     json.dumps(baseline.get("fan_bands", {})),
                     baseline["windows"], baseline["period_start"], baseline["period_end"],
                     baseline["formed_at"], baseline.get("note", "")])

    def clear_baselines(self) -> None:
        """Forget every reference (a manual reset); the windows themselves stay."""
        self._write("DELETE FROM baselines", [])

    def set_state(self, key: str, value: Any) -> None:
        """A small value that must survive a restart."""
        self._write("INSERT OR REPLACE INTO state VALUES (?, ?)", [key, json.dumps(value)])

    def fold_old_samples(self, before: float) -> int:
        """Fold samples older than `before` into 5-minute rows; returns how many were folded.

        `before` is moved back to a 5-minute boundary, so a row is never made
        from half a bucket and then refused when the other half's turn comes.
        """
        before = int(before // FIVE_MINUTES) * FIVE_MINUTES
        with self._lock:
            folded = self._db.execute(
                "INSERT OR IGNORE INTO samples_5m SELECT "
                f"CAST(ts / {FIVE_MINUTES} AS INTEGER) * {FIVE_MINUTES}, COUNT(*), "
                "MIN(temp_gpu), MAX(temp_gpu), AVG(temp_gpu), "
                "MIN(power_w), MAX(power_w), AVG(power_w), "
                "MIN(fan_pct), MAX(fan_pct), AVG(fan_pct), AVG(util_pct) "
                f"FROM samples WHERE ts < ? GROUP BY CAST(ts / {FIVE_MINUTES} AS INTEGER)",
                [before]).rowcount
            self._db.execute("DELETE FROM samples WHERE ts < ?", [before])
            self._db.commit()
            return folded

    # -- reading --------------------------------------------------------------

    def state(self, key: str, default: Any = None) -> Any:
        """A saved small value, or `default`."""
        row = self._one("SELECT value FROM state WHERE key = ?", [key])
        return default if row is None else json.loads(row["value"])

    def samples_since(self, since: float) -> list[dict]:
        """Readings from `since` on, oldest first."""
        return self._all("SELECT * FROM samples WHERE ts >= ? ORDER BY ts", [since])

    def last_sample(self) -> dict | None:
        """The newest reading."""
        return self._one("SELECT * FROM samples ORDER BY ts DESC LIMIT 1", [])

    def windows(self, since_day: str = "", model: str | None = None) -> list[dict]:
        """Stable windows from `since_day` (YYYY-MM-DD) on, optionally for one model."""
        if model is None:
            return self._all("SELECT * FROM windows WHERE day >= ? ORDER BY start", [since_day])
        return self._all("SELECT * FROM windows WHERE day >= ? AND model = ? ORDER BY start",
                         [since_day, model])

    def idle_windows(self, since_day: str = "") -> list[dict]:
        """Quiet stretches from `since_day` on."""
        return self._all("SELECT * FROM idle_windows WHERE day >= ? ORDER BY start", [since_day])

    def events(self, limit: int = 50, since: float = 0.0) -> list[dict]:
        """The newest events first, each with its `id` (the order it was written in)."""
        rows = self._all("SELECT rowid AS id, * FROM events WHERE ts >= ? "
                         "ORDER BY ts DESC, rowid DESC LIMIT ?", [since, limit])
        return _with_detail(rows)

    def events_after(self, after: int, limit: int, sent_only: bool = True) -> list[dict]:
        """Events written after the one numbered `after`, oldest first.

        `sent_only`: only those the channel sent (the anti-spam applied), which
        is what a program passing the alarms on wants.
        """
        sent = " AND json_extract(detail, '$.sent') = 1" if sent_only else ""
        rows = self._all(f"SELECT rowid AS id, * FROM events WHERE rowid > ?{sent} "
                         "ORDER BY rowid LIMIT ?", [after, limit])
        return _with_detail(rows)

    def last_event_id(self, sent_only: bool = True) -> int:
        """The number of the newest event (0: none yet)."""
        sent = " WHERE json_extract(detail, '$.sent') = 1" if sent_only else ""
        row = self._one(f"SELECT MAX(rowid) AS id FROM events{sent}", [])
        return int(row["id"]) if row and row["id"] is not None else 0

    def baselines(self) -> dict[str, dict]:
        """Every reference, by scope."""
        rows = self._all("SELECT * FROM baselines", [])
        for row in rows:
            row["fan_bands"] = json.loads(row["fan_bands"]) if row["fan_bands"] else {}
        return {row["scope"]: row for row in rows}

    def slow_readings(self, since: float) -> list[dict]:
        """Hourly readings from `since` on, oldest first."""
        rows = self._all("SELECT * FROM slow_readings WHERE ts >= ? ORDER BY ts", [since])
        for row in rows:
            row["ecc"] = json.loads(row["ecc"]) if row["ecc"] else {}
            row["rows"] = json.loads(row["rows"]) if row["rows"] else None
        return rows

    def samples_5m_since(self, since: float) -> list[dict]:
        """Folded 5-minute rows from `since` on, oldest first."""
        return self._all("SELECT * FROM samples_5m WHERE ts >= ? ORDER BY ts", [since])

    def count(self, table: str) -> int:
        """How many rows a table has (for the status and the checks)."""
        if table not in TABLES:
            raise ValueError(f"no table {table}")
        row = self._one(f"SELECT COUNT(*) AS n FROM {table}", [])
        return int(row["n"]) if row else 0

    # -- inside ---------------------------------------------------------------

    def _write(self, sql: str, values: list) -> None:
        with self._lock:
            self._db.execute(sql, values)
            self._db.commit()

    def _all(self, sql: str, values: list) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self._db.execute(sql, values).fetchall()]

    def _one(self, sql: str, values: list) -> dict | None:
        with self._lock:
            row = self._db.execute(sql, values).fetchone()
            return None if row is None else dict(row)


def _with_detail(rows: list[dict]) -> list[dict]:
    """Events with their `detail` read back from JSON."""
    for row in rows:
        row["detail"] = json.loads(row["detail"]) if row["detail"] else None
    return rows
