"""Train/held-generator split for category-independent pseudo defects."""
from __future__ import annotations

from pseudo_v5 import make_group_targets_v5

TRAIN_GENERATORS = ("hairline", "scratch", "speck", "dent", "chip", "contamination", "cutpaste")
HELDOUT_GENERATORS = ("crack", "discolour", "misalign", "hole")


def make_targets(images, rng, force_normal=False, heldout=False):
    generators = HELDOUT_GENERATORS if heldout else TRAIN_GENERATORS
    return make_group_targets_v5(images, rng, force_normal=force_normal, generators=generators)
