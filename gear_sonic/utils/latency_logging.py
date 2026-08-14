"""Small buffered CSV logger used by latency instrumentation.

The logger is disabled unless ``GEAR_SONIC_LATENCY_LOG_DIR`` is set.  It uses
only the Python standard library and deliberately does not alter any transport
or queue settings.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import threading
import time
from typing import Mapping, Sequence


LATENCY_LOG_DIR_ENV = "GEAR_SONIC_LATENCY_LOG_DIR"


class CsvLatencyLogger:
    """Thread-safe CSV writer with periodic, buffered flushes."""

    def __init__(
        self,
        filename: str,
        fieldnames: Sequence[str],
        *,
        flush_interval_s: float = 1.0,
    ) -> None:
        self._lock = threading.Lock()
        self._file = None
        self._writer = None
        self._last_flush = time.monotonic()
        self._flush_interval_s = max(0.1, flush_interval_s)
        self.path: Path | None = None

        root = os.environ.get(LATENCY_LOG_DIR_ENV, "").strip()
        if not root:
            return

        try:
            log_dir = Path(root).expanduser()
            log_dir.mkdir(parents=True, exist_ok=True)
            self.path = log_dir / filename
            needs_header = not self.path.exists() or self.path.stat().st_size == 0
            self._file = self.path.open("a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=list(fieldnames),
                extrasaction="ignore",
            )
            if needs_header:
                self._writer.writeheader()
                self._file.flush()
            print(f"[LatencyMonitor] logging to {self.path}")
        except OSError as err:
            self.path = None
            self._file = None
            self._writer = None
            print(f"[LatencyMonitor] disabled: could not open {filename} ({err})")

    @property
    def enabled(self) -> bool:
        return self._writer is not None

    def write(self, row: Mapping[str, object]) -> None:
        with self._lock:
            if self._writer is None:
                return
            self._writer.writerow(row)
            now = time.monotonic()
            if now - self._last_flush >= self._flush_interval_s:
                self._file.flush()
                self._last_flush = now

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.flush()
                self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self) -> "CsvLatencyLogger":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
