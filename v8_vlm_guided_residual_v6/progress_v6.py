"""Dependency-free progress bars for long v6 stages.

The bar writes a compact carriage-return line for an interactive terminal and
also emits periodic newline snapshots so ``screen.log`` and ``tail`` retain a
useful history after remote execution.
"""
from __future__ import annotations

import sys
import time


class ProgressBar:
    def __init__(self, stage: str, total: int, initial: int = 0, width: int = 28, every: int = 1):
        self.stage = str(stage)
        self.total = max(int(total), 1)
        self.current = max(0, min(int(initial), self.total))
        self.width = max(10, int(width))
        self.every = max(1, int(every))
        self.started = time.time()
        self.last_printed = -1
        if self.current:
            self.update(self.current, {}, force=True)

    def _line(self, metrics: dict) -> str:
        fraction = min(max(self.current / self.total, 0.0), 1.0)
        filled = int(round(self.width * fraction))
        bar = "#" * filled + "." * (self.width - filled)
        elapsed = max(time.time() - self.started, 1e-6)
        rate = self.current / elapsed
        eta = (self.total - self.current) / rate if rate > 0 else float("inf")
        extra = " ".join(
            f"{key}={value:.5f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in metrics.items()
            if value is not None
        )
        eta_text = "--" if eta == float("inf") else f"{eta:.0f}s"
        return (
            f"V6_PROGRESS {self.stage} [{bar}] {fraction * 100:6.2f}% "
            f"{self.current}/{self.total} rate={rate:.2f}/s eta={eta_text} {extra}"
        ).rstrip()

    def update(self, current: int, metrics: dict | None = None, force: bool = False):
        self.current = max(0, min(int(current), self.total))
        if not force and self.current != self.total and self.current - self.last_printed < self.every:
            return
        self.last_printed = self.current
        line = self._line(metrics or {})
        # Newline snapshots are intentional: remote logs should remain useful.
        print(line, flush=True)

    def close(self, metrics: dict | None = None):
        self.update(self.total, metrics or {}, force=True)
