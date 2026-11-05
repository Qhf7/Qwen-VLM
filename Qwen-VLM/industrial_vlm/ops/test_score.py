"""Focused boolean score fixtures; optional native processor CPU integration.

PYTHONPATH=src python ops/test_score.py
PYTHONPATH=src python ops/test_score.py --native-model /local/Qwen3-VL-8B-Instruct \
  --native-image prepared/interface_fixture/synthetic_interface_fixture.png

The optional check loads processor/tokenizer only, uses a dummy uniform causal
LM on CPU and never opens model weights or uses the GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

from industrial_vlm.score import (JSON_PREFIX, SCHEMA_VERSION, _calibration,
                                  _read_scores, _run, _score_image, boolean_token_span,
                                  calibrate_normal_threshold, candidate_plan,
                                  normalized_score, ranking_metrics, threshold_metrics)
from industrial_vlm.evaluate import _sha256

NATIVE_MODEL = None
NATIVE_IMAGE = None


def rows(labels: list[bool], scores: list[float]) -> list[dict]:
    return [{"source_id": str(i), "label": label, "anomaly_log_odds": score}
            for i, (label, score) in enumerate(zip(labels, scores))]


class TokenizerFixture:
    is_fast = True

    def __init__(self, *, merged_boundary: bool = False):
        self.merged_boundary = merged_boundary
        self.tokens = {}

    def __call__(self, text, **kwargs):
        value = "true" if text.startswith(JSON_PREFIX + "true") else "false"
        if self.merged_boundary:
            pieces = [JSON_PREFIX[:-1], ":" + value, text[len(JSON_PREFIX + value):]]
        else:
            pieces = [JSON_PREFIX, value[:2], value[2:], text[len(JSON_PREFIX + value):]]
        ids, offsets, cursor = [], [], 0
        for i, piece in enumerate(pieces):
            self.tokens[i] = piece
            ids.append(i)
            offsets.append((cursor, cursor + len(piece)))
            cursor += len(piece)
        return {"input_ids": ids, "offset_mapping": offsets}

    def decode(self, ids, **kwargs):
        return "".join(self.tokens[i] for i in ids)


class MetricsTest(unittest.TestCase):
    def test_perfect_and_reversed_rankings(self):
        perfect = ranking_metrics(rows([True, False], [2.0, 1.0]))
        reverse = ranking_metrics(rows([True, False], [1.0, 2.0]))
        self.assertEqual((perfect["auroc"], perfect["average_precision"]), (1.0, 1.0))
        self.assertEqual((reverse["auroc"], reverse["average_precision"]), (0.0, 0.5))

    def test_all_ties(self):
        value = ranking_metrics(rows([True, False, True, False], [1.0] * 4))
        self.assertEqual(value["auroc"], 0.5)
        self.assertEqual(value["average_precision"], 0.5)

    def test_partial_ties(self):
        # Scores 3:(positive), 2:(positive, negative), 1:(negative).
        value = ranking_metrics(rows([True, True, False, False], [3.0, 2.0, 2.0, 1.0]))
        self.assertAlmostEqual(value["auroc"], 0.875)
        self.assertAlmostEqual(value["average_precision"], 5 / 6)

    def test_one_class(self):
        self.assertIsNone(ranking_metrics(rows([False, False], [0.0, 1.0]))["auroc"])
        self.assertIsNone(ranking_metrics(rows([False, False], [0.0, 1.0]))["average_precision"])
        self.assertEqual(ranking_metrics(rows([True, True], [0.0, 1.0]))["average_precision"], 1.0)

    def test_duplicate_or_nonfinite_sources_rejected(self):
        repeated = rows([True, False], [1.0, 0.0])
        repeated[1]["source_id"] = "0"
        with self.assertRaises(ValueError):
            ranking_metrics(repeated)
        with self.assertRaises(ValueError):
            ranking_metrics(rows([True], [float("nan")]))

    def test_log_softmax_and_extremes(self):
        result = normalized_score(-1.0, -3.0)
        self.assertAlmostEqual(result["anomaly_score"], 1 / (1 + math.exp(-2)))
        self.assertAlmostEqual(math.exp(result["log_normalized_true"]) +
                               math.exp(result["log_normalized_false"]), 1.0)
        self.assertEqual(normalized_score(-1.0, -10001.0)["anomaly_score"], 1.0)
        self.assertEqual(normalized_score(-10001.0, -1.0)["anomaly_score"], 0.0)
        self.assertEqual(normalized_score(-1e307, -1e307)["anomaly_score"], 0.5)
        with self.assertRaises(ValueError):
            normalized_score(float("nan"), -1.0)

    def test_normal_only_threshold_and_boundary_ties(self):
        data = rows([False, False, False, False, True], [4.0, 3.0, 3.0, 1.0, 100.0])
        calibration = calibrate_normal_threshold(data, 0.5)
        self.assertEqual(calibration["threshold_log_odds"], 3.0)
        self.assertEqual(calibration["actual_calibration_false_positives"], 1)
        # Changing anomaly scores cannot affect a normal-only threshold.
        data[-1]["anomaly_log_odds"] = -100.0
        self.assertEqual(calibrate_normal_threshold(data, 0.5)["threshold_log_odds"], 3.0)

    def test_small_calibration_has_zero_allowed_false_positives(self):
        data = rows([False] * 10 + [True], [float(i) for i in range(11)])
        calibration = calibrate_normal_threshold(data, 0.05)
        self.assertEqual(calibration["allowed_calibration_false_positives"], 0)
        self.assertEqual(calibration["threshold_log_odds"], 9.0)
        measured = threshold_metrics(data, calibration["threshold_log_odds"])
        self.assertEqual((measured["fp"], measured["tp"]), (0, 1))


class BoundariesTest(unittest.TestCase):
    def test_actual_multiple_value_tokens(self):
        for value in ("true", "false"):
            plan = candidate_plan(TokenizerFixture(), JSON_PREFIX + value + "}<|im_end|>\n", value)
            self.assertEqual((plan["value_begin"], plan["value_end"]), (1, 3))
            self.assertEqual(plan["value_token_ids"], [1, 2])

    def test_merged_prefix_value_token_rejected(self):
        with self.assertRaisesRegex(ValueError, "boundary"):
            candidate_plan(TokenizerFixture(merged_boundary=True), JSON_PREFIX + "true}", "true")

    def test_unreliable_offsets_rejected(self):
        with self.assertRaisesRegex(ValueError, "gap"):
            boolean_token_span([0, 1], [(0, 1), (2, 4)], "true", 0, 4)

    def test_slow_tokenizer_rejected(self):
        tokenizer = TokenizerFixture()
        tokenizer.is_fast = False
        with self.assertRaisesRegex(ValueError, "fast tokenizer"):
            candidate_plan(tokenizer, JSON_PREFIX + "true}", "true")


class OfflineArtifactTest(unittest.TestCase):
    def test_completed_resume_rejects_internally_consistent_tampering(self):
        from industrial_vlm.data import DECISION_PROMPT
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "model.safetensors").write_bytes(b"isolated test fixture; not real weights")
            image = root / "image.fixture"
            image.write_bytes(b"isolated source fixture; not decoded")
            source = {"id": "fixture_decision", "source_id": "fixture_source", "image": str(image),
                      "category": "fixture", "is_anomalous": False, "prompt": DECISION_PROMPT,
                      "answer": '{"is_anomalous":false}'}
            data = root / "data.jsonl"
            data.write_text(json.dumps(source) + "\n")
            output = root / "output"
            args = argparse.Namespace(model=str(model), adapter=None, data=data,
                                      image_root=root, output=output, data_role="validation",
                                      max_pixels=524288, max_length=2048, limit=None,
                                      run_name="isolated_fixture", resume=False, calibration=None)
            # Let the real runner write its exact config, then stop before any
            # model/GPU access. A fake torch module keeps this fixture CPU-free.
            loader = ModuleType("industrial_vlm.modeling")
            loader.input_device = mock.Mock()
            loader.load_model = mock.Mock(side_effect=RuntimeError("fixture stops before model load"))
            with mock.patch.dict(sys.modules, {"industrial_vlm.modeling": loader,
                                                "torch": ModuleType("torch")}):
                with self.assertRaisesRegex(RuntimeError, "fixture stops"):
                    _run(args)
            meta_path = output / "meta.json"
            meta = json.loads(meta_path.read_text())
            scores_path = output / "scores.jsonl"
            record = {"schema_version": SCHEMA_VERSION, "id": source["id"],
                      "source_id": source["source_id"], "image": str(image),
                      "image_sha256": _sha256(image), "category": source["category"],
                      "label": False, "latency_seconds": 0.0,
                      "log_likelihood_true": -3.0, "log_likelihood_false": -1.0,
                      **normalized_score(-3.0, -1.0)}
            scores_path.write_text(json.dumps(record) + "\n")
            meta.update(status="complete", n_completed=1, scores_sha256=_sha256(scores_path))
            meta_path.write_text(json.dumps(meta))
            # Change raw LL AND its derived values consistently, retaining the
            # old completed hash. Resume must reject instead of blessing it.
            record.update(log_likelihood_true=-5.0, **normalized_score(-5.0, -1.0))
            scores_path.write_text(json.dumps(record) + "\n")
            args.resume = True
            loader.load_model.reset_mock()
            with mock.patch.dict(sys.modules, {"industrial_vlm.modeling": loader,
                                                "torch": ModuleType("torch")}):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    _run(args)
            loader.load_model.assert_not_called()
            self.assertEqual(json.loads(meta_path.read_text())["scores_sha256"], meta["scores_sha256"])

    def test_calibration_metadata_and_score_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [{"source_id": "normal_fixture", "label": False,
                        "log_likelihood_true": -3.0, "log_likelihood_false": -1.0,
                        **normalized_score(-3.0, -1.0)},
                       {"source_id": "anomaly_fixture", "label": True,
                        "log_likelihood_true": -1.0, "log_likelihood_false": -3.0,
                        **normalized_score(-1.0, -3.0)}]
            scores = root / "scores.jsonl"
            scores.write_text("".join(json.dumps(r) + "\n" for r in records))
            meta = {"schema_version": SCHEMA_VERSION, "status": "complete",
                    "scores_sha256": _sha256(scores),
                    "config": {"n_expected": 2, "data_role": "validation",
                               "data_sha256": "fixture", "subset_sha256": "fixture",
                               "score_model_contract": {"fixture_only": True}}}
            metadata = root / "meta.json"
            metadata.write_text(json.dumps(meta))
            output = root / "threshold.json"
            _calibration(argparse.Namespace(scores=root, output=output, target_fpr=0.05))
            value = json.loads(output.read_text())
            self.assertEqual(value["threshold_log_odds"], -2.0)
            self.assertEqual(value["n_calibration_normals"], 1)
            self.assertEqual(value["score_model_contract"], {"fixture_only": True})
            self.assertEqual(value["source_scores_sha256"], _sha256(scores))
            meta["config"]["data_role"] = "regression"
            metadata.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "validation"):
                _calibration(argparse.Namespace(scores=root, output=root / "other.json", target_fpr=0.05))
            scores.write_text(scores.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                _read_scores(root)


class NativeProcessorTest(unittest.TestCase):
    def test_native_full_candidate_tokens_and_no_labels(self):
        if not (NATIVE_MODEL and NATIVE_IMAGE):
            self.skipTest("provide --native-model and --native-image for CPU integration")
        import torch
        from types import SimpleNamespace
        from transformers import AutoProcessor
        from industrial_vlm.data import DECISION_PROMPT

        processor = AutoProcessor.from_pretrained(NATIVE_MODEL, max_pixels=524288,
                                                  local_files_only=True)
        vocabulary = len(processor.tokenizer)

        class UniformCausalLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = torch.nn.Embedding(1, 1)
                self.calls = 0

            def get_input_embeddings(self):
                return self.embeddings

            def forward(self, **inputs):
                if "labels" in inputs:
                    raise AssertionError("candidate labels were passed to model.forward")
                if inputs["image_grid_thw"].shape[0] != 1:
                    raise AssertionError("expected exactly one native image grid")
                self.calls += 1
                count = inputs["input_ids"].shape[-1]
                return SimpleNamespace(logits=torch.zeros(1, 1, vocabulary).expand(1, count, vocabulary))

        model = UniformCausalLM().eval()
        result = _score_image(model, processor, Path(NATIVE_IMAGE), DECISION_PROMPT, 2048)
        self.assertEqual(model.calls, 2)
        for value in ("true", "false"):
            self.assertGreater(result["n_value_tokens"][value], 0)
            self.assertAlmostEqual(result[f"log_likelihood_{value}"],
                                   -math.log(vocabulary) * result["n_value_tokens"][value], places=5)
        self.assertEqual(result["token_boundary_audit"],
                         "native full token IDs == prompt IDs + offset-verified continuation IDs")
        print("NATIVE_CPU_AUDIT", result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--native-model")
    parser.add_argument("--native-image")
    options, remaining = parser.parse_known_args()
    NATIVE_MODEL, NATIVE_IMAGE = options.native_model, options.native_image
    unittest.main(argv=[sys.argv[0], *remaining])
