"""Fast CPU regressions for decoder layout, gradients, and feature caching."""
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


# Import the real implementation without loading ImageBind or a checkpoint.
# Its repository paths are only needed by main(), so preserve the test path.
_source = Path(__file__).resolve().parents[1] / "dec_big.py"
_saved_path = sys.path[:]
try:
    sys.path.insert(0, str(_source.parent))
    _spec = importlib.util.spec_from_file_location("decoder_under_test", _source)
    decoder = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(decoder)
finally:
    sys.path[:] = _saved_path


def _items(*paths):
    return [(path, "fake_category", 1) for path in paths]


class FakeEncoder:
    def __init__(self, nonfinite=False):
        self.calls = []
        self.grad_modes = []
        self.nonfinite = nonfinite

    def __call__(self, inputs):
        self.grad_modes.append(torch.is_grad_enabled())
        values = inputs["vision"][:, 0, 0, 0]
        self.calls.append(values.tolist())
        raw = []
        for level in range(4):
            feature = (values + level * 10).view(1, -1, 1).expand(257, -1, 1280).clone()
            feature[0] = -1000  # Distinct CLS values must be excluded.
            if self.nonfinite:
                feature[1, 0, 0] = float("nan")
            raw.append(feature)
        return {"vision": (None, raw)}


class DecoderLayoutTests(unittest.TestCase):
    def test_raw_layout_preserves_batch_and_patch_identity_and_excludes_cls(self):
        token = torch.arange(257, dtype=torch.float32).view(257, 1, 1)
        batch = torch.arange(2, dtype=torch.float32).view(1, 2, 1) * 1000
        raw = [(token + batch + level * 10000).expand(-1, -1, 1280) for level in range(4)]
        actual = decoder.patch_features(raw, batch_size=2)
        self.assertEqual(len(actual), 4)
        for level, feature in enumerate(actual):
            self.assertEqual(tuple(feature.shape), (2, 256, 1280))
            self.assertTrue(feature.is_contiguous())
            torch.testing.assert_close(feature, raw[level][1:].transpose(0, 1))
            self.assertEqual(feature[0, 0, 0].item(), level * 10000 + 1)
            self.assertEqual(feature[1, -1, 0].item(), level * 10000 + 1256)

    def test_raw_layout_rejects_swapped_axes_wrong_batch_and_missing_level(self):
        raw = [torch.zeros(257, 2, 1280) for _ in range(4)]
        with self.assertRaisesRegex(ValueError, "Unexpected ImageBind feature layout"):
            decoder.patch_features([feature.transpose(0, 1) for feature in raw], batch_size=2)
        with self.assertRaisesRegex(ValueError, "Unexpected ImageBind feature layout"):
            decoder.patch_features(raw, batch_size=1)
        with self.assertRaisesRegex(ValueError, "four ImageBind feature levels"):
            decoder.patch_features(raw[:3], batch_size=2)

    def test_decoder_produces_one_spatial_mask_per_image(self):
        model = decoder.BigDec(d=16).eval()
        features = [torch.full((2, 256, 1280), level / 10) for level in range(4)]
        with torch.no_grad():
            prediction = model(features)
        self.assertEqual(tuple(prediction.shape), (2, 1, 224, 224))
        self.assertTrue(torch.isfinite(prediction).all())
        with self.assertRaisesRegex(ValueError, "Decoder requires"):
            model([feature.transpose(0, 1) for feature in features])

    def test_wrong_saturated_logits_keep_correcting_gradients(self):
        for value, label, expected_sign in ((1000.0, 0.0, 1), (-1000.0, 1.0, -1)):
            with self.subTest(value=value, label=label):
                logits = torch.full((1, 1, 2, 2), value, requires_grad=True)
                loss = decoder.loss_fn(logits, torch.full_like(logits, label))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(torch.isfinite(logits.grad).all())
                self.assertTrue((logits.grad * expected_sign > 0).all())

    def test_cache_pins_batch_hits_deduplicates_misses_and_rejects_nonfinite(self):
        modality_module = ModuleType("ImageBind.models.imagebind_model")
        modality_module.ModalityType = SimpleNamespace(VISION="vision")
        ids = {"a": 1, "b": 2, "c": 3, "d": 4}

        def image_tensor(path):
            return torch.full((1, 3, 1, 1), float(ids[path]))

        def check(features, values):
            self.assertEqual(len(features), 4)
            for level, feature in enumerate(features):
                self.assertEqual(tuple(feature.shape), (len(values), 256, 1280))
                for row, value in zip(feature, values):
                    self.assertTrue(torch.all(row == value + level * 10))

        modules = {"ImageBind.models.imagebind_model": modality_module}
        with patch.dict(sys.modules, modules), patch.object(decoder, "load_image_tensor", image_tensor), \
                patch.object(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor), \
                redirect_stdout(io.StringIO()):
            encoder = FakeEncoder()
            cache = decoder.Features(encoder)
            per_image_bytes = 4 * 256 * 1280 * torch.tensor([], dtype=torch.float16).element_size()
            cache.limit = 2 * per_image_bytes
            check(cache(_items("a", "b")), [1, 2])
            # Inserting c evicts a, even though a is needed twice in this batch.
            check(cache(_items("a", "c", "a")), [1, 3, 1])
            self.assertNotIn("a", cache.cache)
            # A second eviction and duplicated miss exercise both branches.
            check(cache(_items("b", "c", "d", "d")), [2, 3, 4, 4])
            check(cache(_items("c", "d")), [3, 4])
            self.assertEqual(encoder.calls, [[1.0, 2.0], [3.0], [4.0]])
            self.assertFalse(any(encoder.grad_modes))
            actual_bytes = sum(feature.nelement() * feature.element_size()
                               for levels in cache.cache.values() for feature in levels)
            self.assertEqual(cache.nbytes, actual_bytes)
            self.assertLessEqual(cache.nbytes, cache.limit)

            disabled = decoder.Features(encoder, cache_gb=0)
            check(disabled(_items("a", "a", "b")), [1, 1, 2])
            self.assertEqual(encoder.calls[-1], [1.0, 2.0])
            self.assertFalse(disabled.cache)
            self.assertEqual(disabled.nbytes, 0)

            invalid = decoder.Features(FakeEncoder(nonfinite=True))
            with self.assertRaisesRegex(ValueError, "Nonfinite encoder features"):
                invalid(_items("a"))
            self.assertFalse(invalid.cache)


if __name__ == "__main__":
    unittest.main()
