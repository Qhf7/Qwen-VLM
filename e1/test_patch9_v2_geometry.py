"""Tests for the independent Patch9-v2 geometry-only pipeline."""

import unittest

import numpy as np

from e1.patch9_v2_geometry import (
    audit_pair,
    crop_mask,
    fuse_probabilities,
    make_patch_specs,
    records_for,
)


class Patch9V2GeometryTests(unittest.TestCase):
    def test_grid_is_nine_and_covers_borders(self):
        specs = make_patch_specs(1000, 800)
        self.assertEqual(len(specs), 9)
        self.assertEqual(specs[0].as_tuple(), (0, 0, 400, 320))
        self.assertEqual(specs[-1].as_tuple(), (600, 480, 1000, 800))

    def test_image_and_mask_use_identical_coordinates(self):
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        mask = np.zeros((1000, 1000), dtype=bool)
        mask[390:410, 390:410] = True
        records = records_for(image, mask)
        positives = [r for r in records if r["contains_original_positive"]]
        self.assertGreaterEqual(len(positives), 1)
        self.assertTrue(all(r["mask"].shape == (448, 448) for r in records))
        self.assertTrue(any(r["mask"].any() for r in positives))

    def test_empty_patches_are_retained(self):
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        mask = np.zeros((1000, 1000), dtype=bool)
        mask[10:20, 10:20] = True
        records = records_for(image, mask)
        self.assertGreater(sum(not r["contains_original_positive"] for r in records), 0)

    def test_weighted_mean_does_not_equal_old_max(self):
        specs = make_patch_specs(1000, 1000)
        full = np.zeros((448, 448), dtype=np.float32)
        patch = {s.index: np.full((448, 448), 1.0 if s.index == 0 else 0.0, dtype=np.float32)
                 for s in specs}
        fused = fuse_probabilities(full, patch, specs, (1000, 1000), patch_weight=0.5)
        self.assertGreater(float(fused.max()), 0.0)
        self.assertLess(float(fused.max()), 1.0)

    def test_audit_reports_full_and_patch_pixels(self):
        image = np.zeros((800, 1000, 3), dtype=np.uint8)
        mask = np.zeros((800, 1000), dtype=bool)
        mask[350:360, 350:360] = True
        report = audit_pair(image, mask)
        self.assertEqual(report["patch_count"], 9)
        self.assertGreater(report["positive_patch_count"], 0)
        self.assertGreater(report["mask_pixels_full_448"], 0)


if __name__ == "__main__":
    unittest.main()
