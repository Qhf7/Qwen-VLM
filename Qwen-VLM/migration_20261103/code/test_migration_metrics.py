"""Run with: python -m unittest -v test_migration_metrics.py (CPU only)."""
import unittest

import numpy as np

from migration_metrics import (donor_map_indices, evaluate_predictions, generated_metrics,
                               parse_generated, score_metrics, spatial_shuffle_maps, validate_cache)


class MetricsTests(unittest.TestCase):
    def test_tied_scores_use_standard_auc_and_ap(self):
        result = score_metrics([0, 1, 0, 1], [0, 0, 0, 0])
        self.assertEqual(result["image_auroc"], 0.5)
        self.assertEqual(result["image_ap"], 0.5)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["fpr"], 0.0)

    def test_zero_threshold_confusion(self):
        result = score_metrics([0, 0, 1, 1], [-2, 1, -1, 2])
        for key in ("accuracy", "macro_f1", "recall", "fpr"):
            self.assertEqual(result[key], 0.5)
        self.assertEqual(result["image_auroc"], 0.75)

    def test_bad_inputs_fail_and_one_class_metrics_are_explicit(self):
        for labels, scores in [([], []), ([0, 2], [1, 2]), ([0, 1], [1]), ([0, 1], [0, np.nan])]:
            with self.assertRaises(ValueError):
                score_metrics(labels, scores)
        result = score_metrics([0, 0], [0, 1])
        self.assertIsNone(result["image_auroc"])
        self.assertIsNone(result["image_ap"])
        self.assertIsNone(result["recall"])

    def test_parser_uses_only_first_decision_after_think(self):
        cases = {
            "  Yes, there is damage.": 1,
            "\nno.": 0,
            "<think>Yes may be wrong.\nInspect carefully.</think> No, normal.": 0,
            "<think>x</think> <think>y</think> YES": 1,
            "  </think>\nNo": 0,
            "Yes, although the text later says No.": 1,
            "The answer is Yes.": None,
            "Yesterday there was damage.": None,
            "Normal. No damage.": None,
            "<think>Yes": None,
            "<think>Yes</think> I conclude No.": None,
            "No_way": None,
            "": None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_generated(text), expected)
        self.assertIsNone(parse_generated(None))

    def test_invalid_answers_remain_in_accuracy_and_f1(self):
        result = generated_metrics([0, 0, 1, 1], ["No", "maybe", "Yes", "I think Yes"])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)
        self.assertEqual(result["invalid_rate"], 0.5)
        self.assertEqual(result["n_invalid_normal"], 1)
        self.assertEqual(result["n_invalid_anomaly"], 1)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["fpr"], 0.0)
        all_invalid = generated_metrics([0, 1], ["maybe", "maybe"])
        self.assertEqual(all_invalid["accuracy"], 0)
        self.assertEqual(all_invalid["macro_f1"], 0)
        self.assertEqual(all_invalid["invalid_rate"], 1)

    def test_categories_have_equal_weight_and_defined_counts(self):
        items = [{"category": "a", "label": label} for label in [0, 0, 1, 1]]
        items += [{"category": "b", "label": label} for label in [0, 1]]
        result = evaluate_predictions(items, [-1, -1, 1, 1, 1, -1], ["No", "No", "Yes", "Yes", "Yes", "No"])
        self.assertAlmostEqual(result["overall"]["scores"]["accuracy"], 4 / 6)
        self.assertEqual(result["category_macro"]["scores"]["accuracy"], 0.5)
        self.assertEqual(result["category_macro"]["generated"]["accuracy"], 0.5)
        self.assertEqual(result["category_macro_counts"]["scores"]["image_auroc"], 2)
        result = evaluate_predictions([{"category": "normal", "label": 0},
                                       {"category": "anomaly", "label": 1}], [-1, 1])
        self.assertNotIn("image_auroc", result["category_macro"]["scores"])
        self.assertEqual(result["category_macro_counts"]["scores"]["image_ap"], 1)


class InterventionTests(unittest.TestCase):
    def setUp(self):
        self.items = [{"path": f"{category}_{i}", "category": category, "label": label, "split": "val"}
                      for category in ("a", "b") for i, label in enumerate((0, 0, 1, 1))]

    def test_donors_are_reproducible_opposite_label_same_category(self):
        indices = donor_map_indices(self.items, seed=42)
        np.testing.assert_array_equal(indices, donor_map_indices(self.items, seed=42))
        for index, donor in enumerate(indices):
            self.assertNotEqual(index, donor)
            self.assertEqual(self.items[index]["category"], self.items[donor]["category"])
            self.assertNotEqual(self.items[index]["label"], self.items[donor]["label"])
        with self.assertRaises(ValueError):
            donor_map_indices(self.items[:1])
        with self.assertRaises(ValueError):
            donor_map_indices(self.items[:2])
        np.testing.assert_array_equal(donor_map_indices(self.items[:2], opposite_label=False), [1, 0])

    def test_donors_never_cross_splits(self):
        items = [dict(item) for item in self.items[:2]]
        items[1]["split"] = "test"
        with self.assertRaises(ValueError):
            donor_map_indices(items, opposite_label=False)

    def test_shuffle_preserves_values_but_changes_positions(self):
        maps = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
        original = maps.copy()
        shuffled = spatial_shuffle_maps(maps, seed=42)
        np.testing.assert_array_equal(maps, original)
        np.testing.assert_array_equal(shuffled, spatial_shuffle_maps(maps, seed=42))
        self.assertEqual(shuffled.dtype, maps.dtype)
        self.assertEqual(shuffled.shape, maps.shape)
        for before, after in zip(maps, shuffled):
            np.testing.assert_array_equal(np.sort(before.ravel()), np.sort(after.ravel()))
            self.assertFalse(np.array_equal(before, after))
        self.assertEqual(spatial_shuffle_maps(maps[0]).shape, maps[0].shape)
        with self.assertRaises(ValueError):
            spatial_shuffle_maps(np.zeros((3, 3), dtype=np.float32))

    def test_shuffle_repeated_values_and_optional_torch(self):
        maps = np.array([[0, 0, 1], [0, 1, 1]], dtype=np.float32)
        shuffled = spatial_shuffle_maps(maps)
        np.testing.assert_array_equal(np.sort(maps.ravel()), np.sort(shuffled.ravel()))
        self.assertFalse(np.array_equal(maps, shuffled))
        try:
            import torch
        except ImportError:
            return
        tensor = torch.from_numpy(maps)
        result = spatial_shuffle_maps(tensor)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.device, tensor.device)
        self.assertEqual(result.dtype, tensor.dtype)
        np.testing.assert_array_equal(result.numpy(), shuffled)


class CacheTests(unittest.TestCase):
    def test_schema_and_versions(self):
        cache = {"pipeline_version": "v1", "decoder_sha256": "a" * 64,
                 "items": [{"path": "normal", "category": "a", "label": 0, "split": "train"},
                           {"path": "anomaly", "category": "a", "label": 1, "split": "val"}],
                 "cls": np.zeros((2, 1024), dtype=np.float32),
                 "maps": np.zeros((2, 224, 224), dtype=np.float32)}
        result = validate_cache(cache, expected_pipeline_version="v1", expected_decoder_sha256="a" * 64)
        self.assertEqual(result["split_counts"], {"train": 1, "val": 1, "test": 0})
        with self.assertRaises(ValueError):
            validate_cache(cache, expected_pipeline_version="old")
        with self.assertRaises(ValueError):
            validate_cache({**cache, "maps": cache["maps"].astype(np.float16)})
        with self.assertRaises(ValueError):
            validate_cache({**cache, "decoder_sha256": "bad"})


if __name__ == "__main__":
    unittest.main()
