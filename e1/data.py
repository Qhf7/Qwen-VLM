"""Dataset and collation utilities for E1 training and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from b0.b0plus_protocol import P0, P4_CLASSIFICATION_ONLY
from e1.pseudo_anomaly import make_pseudo_anomaly, sample_seed


SIDE = 448
REGION_NONE = 9


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        value = np.asarray(source.convert("RGB"), dtype=np.uint8)
    if value.shape != (SIDE, SIDE, 3):
        raise RuntimeError(f"cached image is not 448 RGB: {path} {value.shape}")
    return value


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        value = np.asarray(source.convert("L")) > 0
    if value.shape != (SIDE, SIDE):
        raise RuntimeError(f"cached mask is not 448x448: {path} {value.shape}")
    return value


def mask_region(mask: np.ndarray) -> int:
    """Map a binary mask to the 3x3 cell with the largest positive area."""
    if not np.any(mask):
        return REGION_NONE
    rows = np.array_split(np.arange(mask.shape[0]), 3)
    columns = np.array_split(np.arange(mask.shape[1]), 3)
    counts = [int(mask[np.ix_(ys, xs)].sum()) for ys in rows for xs in columns]
    return int(np.argmax(counts))


class E1TrainingDataset(Dataset):
    def __init__(
        self,
        pool: pd.DataFrame,
        cache_root: Path,
        global_seed: int = 20260901,
    ) -> None:
        self.pool = pool.reset_index(drop=True).copy()
        self.cache_root = Path(cache_root)
        self.global_seed = int(global_seed)
        self.epoch = 0
        self.pseudo_paths = sorted(
            self.pool.loc[self.pool.source_type.eq("pseudo_anomaly"), "image_path"].unique()
        )
        if not self.pseudo_paths:
            raise ValueError("pseudo anomaly donor pool is empty")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.pool)

    def _donor_path(self, image_path: str) -> str:
        seed = sample_seed(image_path, self.epoch, self.global_seed)
        index = seed % len(self.pseudo_paths)
        donor = self.pseudo_paths[index]
        if donor == image_path and len(self.pseudo_paths) > 1:
            donor = self.pseudo_paths[(index + 1) % len(self.pseudo_paths)]
        return donor

    def __getitem__(self, index: int) -> dict:
        row = self.pool.iloc[int(index)]
        image_path = str(row.image_path)
        source_type = str(row.source_type)
        image = read_rgb(self.cache_root / image_path)

        if source_type == "real_anomaly":
            mask_path = str(row.mask_path)
            mask = read_mask(self.cache_root / mask_path)
            if not np.any(mask):
                raise RuntimeError(f"real anomaly has an empty cached mask: {mask_path}")
            label = 1.0
        elif source_type == "pseudo_anomaly":
            donor = read_rgb(self.cache_root / self._donor_path(image_path))
            image, pseudo_mask = make_pseudo_anomaly(
                image=image,
                donor=donor,
                method=str(row.augmentation),
                image_path=image_path,
                epoch=self.epoch,
                global_seed=self.global_seed,
            )
            mask = pseudo_mask > 0
            if not np.any(mask):
                raise RuntimeError(f"pseudo anomaly has an empty mask: {image_path}")
            label = 1.0
        elif source_type in {"normal", "hard_normal"}:
            mask = np.zeros((SIDE, SIDE), dtype=bool)
            label = 0.0
        else:
            raise ValueError(f"unknown source_type: {source_type}")

        return {
            "image": image,
            "mask": mask.astype(np.float32)[None],
            "sample_label": label,
            "region_label": mask_region(mask),
            "pool_index": int(row.pool_index),
            "source_type": source_type,
            "class_name": str(row.class_name),
            "image_path": image_path,
        }


class E1ClassificationDataset(E1TrainingDataset):
    """Same frozen source schedule as E1, without loading or returning GT masks."""

    def __getitem__(self, index: int) -> dict:
        row = self.pool.iloc[int(index)]
        image_path = str(row.image_path)
        source_type = str(row.source_type)
        image = read_rgb(self.cache_root / image_path)

        if source_type == "real_anomaly":
            label = 1.0
        elif source_type == "pseudo_anomaly":
            donor = read_rgb(self.cache_root / self._donor_path(image_path))
            image, _ = make_pseudo_anomaly(
                image=image,
                donor=donor,
                method=str(row.augmentation),
                image_path=image_path,
                epoch=self.epoch,
                global_seed=self.global_seed,
            )
            label = 1.0
        elif source_type in {"normal", "hard_normal"}:
            label = 0.0
        else:
            raise ValueError(f"unknown source_type: {source_type}")

        return {
            "image": image,
            "sample_label": label,
            "pool_index": int(row.pool_index),
            "source_type": source_type,
            "class_name": str(row.class_name),
            "image_path": image_path,
        }


def list_collate(examples: list[dict]) -> list[dict]:
    return examples


def encode_examples(processor, examples: list[dict], device: torch.device) -> tuple[dict, dict]:
    conversations = [
        [{
            "role": "user",
            "content": [
                {"type": "image", "image": Image.fromarray(example["image"])},
                {"type": "text", "text": P0},
            ],
        }]
        for example in examples
    ]
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"text_kwargs": {"padding": True}},
    )
    inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
    targets = {
        "sample_labels": torch.tensor(
            [example["sample_label"] for example in examples], device=device
        ),
        "masks": torch.from_numpy(np.stack([example["mask"] for example in examples])).to(
            device, non_blocking=True
        ),
        "region_labels": torch.tensor(
            [example["region_label"] for example in examples], device=device, dtype=torch.long
        ),
    }
    return inputs, targets


def encode_classification_examples(
    processor, examples: list[dict], device: torch.device
) -> tuple[dict, torch.Tensor]:
    conversations = [
        [{
            "role": "user",
            "content": [
                {"type": "image", "image": Image.fromarray(example["image"])},
                {"type": "text", "text": P4_CLASSIFICATION_ONLY},
            ],
        }]
        for example in examples
    ]
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"text_kwargs": {"padding": True}},
    )
    inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
    labels = torch.tensor(
        [example["sample_label"] for example in examples], device=device, dtype=torch.float32
    )
    return inputs, labels
