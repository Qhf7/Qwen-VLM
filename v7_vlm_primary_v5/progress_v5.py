"""Log-friendly progress reporting for long v5 server stages."""
from __future__ import annotations

import json
import time
from pathlib import Path


def _format_seconds(seconds: float) -> str:
    if not (seconds >= 0) or seconds == float("inf"):
        return "--:--:--"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


class ProgressReporter:
    """Print persistent one-line progress and atomically update JSON state."""

    def __init__(
        self,
        stage: str,
        total: int,
        progress_path: Path | None = None,
        initial: int = 0,
        width: int = 24,
    ):
        self.stage = str(stage)
        self.total = max(int(total), 1)
        self.progress_path = progress_path
        self.initial = max(0, int(initial))
        self.width = max(10, int(width))
        self.started = time.time()
        self.last_time = self.started
        self.last_completed = self.initial
        self.ema_seconds_per_item: float | None = None

    def update(self, completed: int, metrics: dict | None = None, force: bool = False) -> dict:
        completed = min(max(int(completed), 0), self.total)
        now = time.time()
        delta_items = completed - self.last_completed
        if delta_items > 0:
            instant = (now - self.last_time) / delta_items
            self.ema_seconds_per_item = (
                instant if self.ema_seconds_per_item is None
                else 0.85 * self.ema_seconds_per_item + 0.15 * instant
            )
        elapsed = now - self.started
        if self.ema_seconds_per_item is not None:
            eta = max(self.total - completed, 0) * self.ema_seconds_per_item
            rate = 1.0 / max(self.ema_seconds_per_item, 1e-9)
        else:
            eta, rate = float("inf"), 0.0
        fraction = completed / self.total
        filled = min(self.width, int(round(fraction * self.width)))
        bar = "=" * filled + ">" * int(filled < self.width) + "." * max(self.width - filled - 1, 0)
        payload = {
            "stage": self.stage,
            "completed": completed,
            "total": self.total,
            "percent": round(100.0 * fraction, 3),
            "elapsed_seconds": round(elapsed, 2),
            "eta_seconds": None if eta == float("inf") else round(eta, 2),
            "items_per_second": round(rate, 5),
            "updated_unix": round(now, 3),
            "metrics": metrics or {},
        }
        if self.progress_path is not None:
            _atomic_json(self.progress_path, payload)
        metric_text = " ".join(
            f"{key}={value:.5f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in (metrics or {}).items()
        )
        print(
            f"PROGRESS stage={self.stage} [{bar}] {completed}/{self.total} "
            f"({100.0*fraction:6.2f}%) elapsed={_format_seconds(elapsed)} "
            f"eta={_format_seconds(eta)} rate={rate:.3f}/s"
            + (f" {metric_text}" if metric_text else ""),
            flush=True,
        )
        self.last_time = now
        self.last_completed = completed
        return payload
