"""Canonical dataset policy for all future experiments.

The active training protocol is the original fold1 split.  The expanded
2026-09-27 split is retained only as historical evaluation metadata and must
not be selected by a training entry point.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

BASE_MANIFEST = Path("artifacts/manifest.parquet")
BASE_SPLIT_DIR = Path("artifacts/splits/fold1")
BASE_TRAIN = BASE_SPLIT_DIR / "train.parquet"
BASE_DEV = BASE_SPLIT_DIR / "dev.parquet"
BASE_DEV_UNBIASED = BASE_SPLIT_DIR / "dev_unbiased.parquet"
BASE_HOLDOUT = BASE_SPLIT_DIR / "holdout.parquet"
BASE_HOLDOUT_CLASSES = BASE_SPLIT_DIR / "holdout_classes.txt"

# All classes in the original manifest.  The six holdout classes remain in
# the dataset for zero-shot testing; only the other 18 are used for training.
BASE_CLASSES = frozenset({
    "3_adapter", "PLCC_socket", "accurate_detection_switch",
    "audio_jack_socket", "balun_transformer", "battery_holder_connector",
    "battery_socket_connector", "blade_switch", "boost_converter_module",
    "ceramic_fuse", "ceramic_wave_filter", "detection_switch",
    "jam_jar_model", "knob_cap", "lilypad_led", "miniature_laser_module",
    "mobile_charging_connector", "multi_function_switch", "nylon_ball_head",
    "pneumatic_elbow", "side_press_switch", "silicon_cell_sensor",
    "slide_switch", "smd_receiver_module",
})
HOLDOUT_CLASSES = frozenset({
    "balun_transformer", "blade_switch", "boost_converter_module",
    "detection_switch", "multi_function_switch", "silicon_cell_sensor",
})
TRAIN_CLASSES = BASE_CLASSES - HOLDOUT_CLASSES
ADDED_CLASSES = frozenset({
    "insect_metal_parts", "lighting_connector", "limit_switch",
    "rotary_position_sensor", "jam_jar_model 2", "knob_cap 2",
    "lilypad_led 2", "mobile_charging_connector 2",
})


def assert_original_split(frame, split_name: str, allowed: Iterable[str] | None = None) -> None:
    """Reject expanded/new-category rows before a training run can start."""
    classes = set(frame["class_name"].dropna().astype(str))
    allowed_set = set(allowed if allowed is not None else TRAIN_CLASSES)
    unexpected = sorted(classes - allowed_set)
    if unexpected:
        raise ValueError(
            f"{split_name} contains non-original classes: {unexpected}. "
            "Use artifacts/splits/fold1 for future training."
        )


def assert_original_paths(frame, split_name: str) -> None:
    """Reject rows whose dataset path belongs to an added category."""
    paths = frame["image_path"].astype(str)
    hits = sorted({part for part in ADDED_CLASSES for value in paths if part in value})
    if hits:
        raise ValueError(f"{split_name} contains paths from removed categories: {hits}")
