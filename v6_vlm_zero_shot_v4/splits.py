from __future__ import annotations

import hashlib


def category_fold(category: str, folds: int = 5) -> int:
    digest = hashlib.sha1(category.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % folds


def split_categories(samples, heldout_fold: int, folds: int = 5):
    train = [sample for sample in samples if category_fold(sample.category, folds) != heldout_fold]
    heldout = [sample for sample in samples if category_fold(sample.category, folds) == heldout_fold]
    return train, heldout
