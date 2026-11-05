"""CPU checks for shared VisA image geometry, masks, and standard metrics."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visa_pipeline import (
    IMAGEBIND_MEAN, IMAGEBIND_STD, PIPELINE_VERSION, auroc, average_precision,
    load_image_pil, load_image_tensor, load_mask_tensor, mask_path_for_image,
    pixel_metrics,
)


class VisaPipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "VisA" / "candle"
        self.image_path = self.root / "Data/Images/Anomaly/001.JPG"
        self.mask_path = self.root / "Data/Masks/Anomaly/001.png"
        self.image_path.parent.mkdir(parents=True)
        self.mask_path.parent.mkdir(parents=True)

    def test_non_square_geometry_and_palette_indices(self):
        # 8x16, with each positive palette index occupying a distinct stripe.
        raw = np.zeros((8, 16), dtype=np.uint8)
        raw[:, 1:9] = np.arange(1, 9, dtype=np.uint8)[None, :]
        rgb = np.repeat((raw > 0)[:, :, None], 3, axis=2).astype(np.uint8) * 255
        # Store losslessly despite the VisA-style image filename.
        Image.fromarray(rgb).save(self.image_path, format="PNG")
        palette_mask = Image.fromarray(raw, mode="P")
        # All colors are black: converting this palette to L would lose labels.
        palette_mask.putpalette([0] * 768)
        palette_mask.save(self.mask_path)
        image = load_image_pil(self.image_path, size=32)
        mask = load_mask_tensor(self.mask_path, device="cpu", size=32)
        expected = np.repeat(np.repeat(raw > 0, 4, axis=0), 2, axis=1)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (32, 32))
        self.assertEqual(tuple(mask.shape), (1, 32, 32))
        np.testing.assert_array_equal(mask[0].numpy(), expected.astype(np.float32))
        np.testing.assert_array_equal(np.array(image)[:, :, 0] > 127, expected)
        for label in range(1, 9):
            self.assertTrue(mask[0, 16, label * 2].item() == 1)
        # Foreground begins close to the left edge, proving no center crop.
        self.assertEqual(mask[0, 16, 2].item(), 1)
        self.assertEqual(mask[0, 16, 30].item(), 0)

    def test_imagebind_normalization_and_normal_mask(self):
        color = np.array([255, 128, 0], dtype=np.uint8)
        Image.fromarray(np.tile(color, (4, 9, 1))).save(self.image_path, format="PNG")
        actual = load_image_tensor(self.image_path, device="cpu", size=12)
        expected = (color.astype(np.float32) / 255 - IMAGEBIND_MEAN) / IMAGEBIND_STD
        self.assertEqual(tuple(actual.shape), (1, 3, 12, 12))
        self.assertEqual(actual.dtype, torch.float32)
        np.testing.assert_allclose(actual[0, :, 0, 0].numpy(), expected, rtol=1e-6)
        self.assertIsNone(mask_path_for_image(self.image_path, 0))
        normal = load_mask_tensor(None, device="cpu", size=12)
        self.assertEqual(tuple(normal.shape), (1, 12, 12))
        self.assertEqual(float(normal.sum()), 0)

    def test_missing_anomaly_mask_fails_and_uppercase_extension_works(self):
        with self.assertRaises(FileNotFoundError):
            mask_path_for_image(self.image_path, 1)
        uppercase = self.mask_path.with_suffix(".PNG")
        Image.fromarray(np.ones((2, 3), dtype=np.uint8)).save(uppercase)
        # The temporary volume may itself be case-insensitive (macOS).
        self.assertTrue(Path(mask_path_for_image(self.image_path, 1)).samefile(uppercase))
        Image.fromarray(np.ones((2, 3), dtype=np.uint8)).save(self.mask_path)
        self.assertEqual(mask_path_for_image(self.image_path, 1), str(self.mask_path))

    def test_ties_use_standard_auc_and_average_precision(self):
        y = np.array([0, 1, 0, 0, 1, 0, 0, 0])
        scores = np.full(y.shape, 0.25)
        self.assertEqual(auroc(y, scores), 0.5)
        self.assertEqual(average_precision(y, scores), float(y.mean()))
        # Reordering tied samples cannot change a standard metric.
        self.assertEqual(auroc(y[::-1], scores), 0.5)
        self.assertEqual(average_precision(y[::-1], scores), float(y.mean()))

    def test_metric_validation(self):
        for invalid in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                auroc([0, 1], [0, invalid])
            with self.assertRaises(ValueError):
                average_precision([0, 1], [0, invalid])
        with self.assertRaises(ValueError):
            auroc([0, 0], [0.1, 0.2])
        with self.assertRaises(ValueError):
            auroc([0, 2], [0.1, 0.2])
        with self.assertRaises(ValueError):
            average_precision([0, 1], [0.1])

    def test_pixel_metrics_aggregates_categories_and_counts(self):
        normal = np.zeros((2, 2), dtype=np.float32)
        gt_a = np.array([[1, 0], [0, 0]], dtype=np.float32)
        gt_b = np.array([[1, 1], [0, 0]], dtype=np.float32)
        items = [
            ("a", 0, normal, normal),
            ("a", 1, torch.from_numpy(gt_a)[None], gt_a),
            ("b", 0, normal, normal),
            ("b", 1, gt_b, 1 - gt_b),
        ]
        result = pixel_metrics(iter(items))
        # Category b has two low positives tied with four normal pixels,
        # and two higher background pixels: AUC=1/3, fg-weighted AUC=5/9.
        self.assertAlmostEqual(result["pixel_auroc_macro"], 2 / 3)
        self.assertAlmostEqual(result["pixel_auroc_weighted"], 5 / 9)
        self.assertAlmostEqual(result["pixel_ap_macro"], (1 + 0.25) / 2)
        self.assertEqual(result["image_auroc_from_map99"], 1)
        self.assertEqual(result["pipeline_version"], PIPELINE_VERSION)
        self.assertEqual(result["n_images"], 4)
        self.assertEqual(result["n_categories"], 2)
        self.assertEqual(result["counts"]["foreground_pixels"], 3)
        self.assertEqual(result["counts"]["pixels"], 16)
        self.assertEqual(result["foreground_ratio"], 3 / 16)

    def test_undefined_auc_is_explicit_and_misalignment_fails(self):
        zeros = np.zeros((2, 2), dtype=np.float32)
        result = pixel_metrics([("normal_only", 0, zeros, zeros)])
        self.assertIsNone(result["pixel_auroc_macro"])
        self.assertIsNone(result["image_auroc_from_map99"])
        self.assertEqual(result["n_categories"], 0)
        self.assertEqual(result["n_categories_total"], 1)
        with self.assertRaises(ValueError):
            pixel_metrics([("a", 1, zeros, np.zeros((2, 3)))])
        with self.assertRaises(ValueError):
            pixel_metrics([("a", 0, np.ones((2, 2)), zeros)])


if __name__ == "__main__":
    unittest.main()
