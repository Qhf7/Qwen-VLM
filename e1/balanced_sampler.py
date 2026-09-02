"""E1 四路等额 batch sampler：real-NG / normal / hard-normal / pseudo-NG。"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pandas as pd
from torch.utils.data import Sampler


SOURCES = ("real_anomaly", "normal", "hard_normal", "pseudo_anomaly")


class BalancedBatchSampler(Sampler[list[int]]):
    """每个 batch 对四个 source 精确等额，并在 source 内均匀抽 sampler_cell。"""

    def __init__(
        self,
        pool: pd.DataFrame,
        batch_size: int,
        steps_per_epoch: int,
        seed: int = 20260901,
    ) -> None:
        if batch_size % len(SOURCES):
            raise ValueError(f"batch_size 必须是 {len(SOURCES)} 的倍数")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch 必须为正")
        missing = set(SOURCES) - set(pool.source_type.unique())
        if missing:
            raise ValueError(f"sampler pool 缺少 source: {sorted(missing)}")
        self.pool = pool.reset_index(drop=True)
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.epoch = 0
        self.quota = batch_size // len(SOURCES)
        self.cells: dict[str, list[np.ndarray]] = {}
        for source in SOURCES:
            source_frame = self.pool[self.pool.source_type.eq(source)]
            self.cells[source] = [
                part.index.to_numpy(dtype=np.int64)
                for _, part in source_frame.groupby("sampler_cell", sort=True)
            ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.steps_per_epoch):
            batch: list[int] = []
            for source in SOURCES:
                cells = self.cells[source]
                chosen_cells = rng.integers(0, len(cells), size=self.quota)
                for cell_index in chosen_cells:
                    indices = cells[int(cell_index)]
                    batch.append(int(indices[rng.integers(0, len(indices))]))
            rng.shuffle(batch)
            yield batch


def default_steps_per_epoch(pool: pd.DataFrame, batch_size: int) -> int:
    """让一轮期望看到约一次完整 real-anomaly 池。"""
    quota = batch_size // len(SOURCES)
    n_real = int(pool.source_type.eq("real_anomaly").sum())
    return math.ceil(n_real / quota)
