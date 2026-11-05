"""Frozen native Qwen3-VL visual features + normal-patch memory, validation only.

No language-model weights, decoder training, test-set inference, or feature
disk cache. The CLI defaults to source audit; --run explicitly enables GPU.
Use --cpu-test-only for the synthetic spatial-order regression on any host.

  python migration_visual_probe.py --cpu-test-only
  python migration_visual_probe.py --run --out runs/native_memory/val.json

The audited Qwen2VL fast image processor emits merge-block order:
  [T, block_row, block_col, intra_row, intra_col, C].
Qwen3VL visual blocks preserve this order; the main merger groups consecutive
2x2 blocks. Its input must be inverted to [T,H,W,C], whereas its output is
already [T,H/2,W/2,C]. No grid shape is inferred from a square root.
"""
import argparse
from collections import Counter
import gc
import hashlib
import inspect
import json
from pathlib import Path
import random
import tempfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from migration_native_parity import DEFAULT_MODEL, atomic_json, source_audit
from visa_pipeline import load_image_pil, load_mask_tensor, mask_path_for_image, pixel_metrics


PIPELINE_VERSION = "qwen3vl-bf16-normal-memory-square-v1"
EXPLICIT_GEOMETRY_POLICY = "RGB direct bicubic square resize; processor do_resize=False; actual patch grid must equal requested size"
AUDITED_PROCESSOR_SHA256 = "09bfa9b17df7c3f0c6159bc34008ee50f21d2472cd5bae7e5c21ba1ca13a423c"


def restore_premerger_grid(features, grid_thw, merge_size=2):
    """Inverse the exact merge-block permutation in the audited image processor."""
    t, h, w = map(int, grid_thw)
    if features.ndim != 2 or features.shape[0] != t * h * w:
        raise ValueError("Pre-merger feature count does not match actual grid_thw")
    if min(t, h, w, merge_size) < 1 or h % merge_size or w % merge_size:
        raise ValueError("Grid height/width must be divisible by spatial_merge_size")
    channels = features.shape[1]
    return (features.reshape(t, h // merge_size, w // merge_size, merge_size, merge_size, channels)
            .permute(0, 1, 3, 2, 4, 5).reshape(t, h, w, channels))


def restore_postmerger_grid(features, grid_thw, merge_size=2):
    t, h, w = map(int, grid_thw)
    if min(t, h, w, merge_size) < 1 or h % merge_size or w % merge_size:
        raise ValueError("Grid height/width must be divisible by spatial_merge_size")
    if features.ndim != 2 or features.shape[0] != t * (h // merge_size) * (w // merge_size):
        raise ValueError("Post-merger feature count does not match actual merged grid")
    return features.reshape(t, h // merge_size, w // merge_size, features.shape[1])


def spatial_order_self_test():
    """Non-square grids and temporal frames detect a plausible-looking wrong reshape."""
    cases = []
    for t, h, w in ((1, 4, 6), (2, 6, 10), (1, 16, 16)):
        positions = torch.arange(t * h * w, dtype=torch.float32).reshape(t, h, w, 1)
        # This forward permutation is copied directly from the audited processor.
        block_order = (positions.view(t, h // 2, 2, w // 2, 2, 1)
                       .permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
        restored = restore_premerger_grid(block_order, (t, h, w), 2)
        torch.testing.assert_close(restored, positions, rtol=0, atol=0)
        merged = block_order.reshape(-1, 4, 1).mean(1)
        post = restore_postmerger_grid(merged, (t, h, w), 2)
        expected = positions.reshape(t, h // 2, 2, w // 2, 2, 1).mean((2, 4))
        torch.testing.assert_close(post, expected, rtol=0, atol=0)
        # Confirm this fixture actually catches the old naive square/grid view.
        assert not torch.equal(block_order.reshape(t, h, w, 1), positions)
        cases.append({"grid_thw": [t, h, w], "pre_shape": list(restored.shape), "post_shape": list(post.shape)})
    for function in (restore_premerger_grid, restore_postmerger_grid):
        try:
            function(torch.zeros(5, 2), (1, 4, 6), 2)
        except ValueError:
            pass
        else:
            raise AssertionError("Incorrect feature counts must fail")
    return {"passed": True, "cases": cases, "invalid_feature_counts_rejected": True}


def sample_spatial_features(grid, maximum_side):
    """Deterministic, spatially spread patch subset; no averaging across defects."""
    if grid.ndim != 3 or maximum_side < 1:
        raise ValueError("Expected [H,W,C] and a positive memory sampling side")
    height, width = grid.shape[:2]
    rows = torch.linspace(0, height - 1, min(height, maximum_side), device=grid.device).round().long().unique()
    cols = torch.linspace(0, width - 1, min(width, maximum_side), device=grid.device).round().long().unique()
    return grid[rows[:, None], cols[None, :]].reshape(-1, grid.shape[-1])


def normalized_features(features):
    features = features.float()
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Nonfinite native visual features")
    norms = features.norm(dim=-1, keepdim=True)
    if bool((norms <= 1e-12).any()):
        raise ValueError("Zero-norm native visual features")
    return features / norms


@torch.inference_mode()
def nearest_cosine_map(query_grid, bank, query_chunk=256, memory_chunk=2048):
    """Exact cosine nearest neighbour, chunked along query and memory dimensions."""
    height, width, channels = query_grid.shape
    query = normalized_features(query_grid.reshape(-1, channels))
    if bank.ndim != 2 or not len(bank) or bank.shape[1] != channels:
        raise ValueError("Normal memory bank and query channels disagree")
    if query_chunk < 1 or memory_chunk < 1:
        raise ValueError("Nearest-neighbour chunk sizes must be positive")
    scores = []
    for start in range(0, len(query), query_chunk):
        block = query[start:start + query_chunk]
        maximum = block.new_full((len(block),), -float("inf"))
        for memory_start in range(0, len(bank), memory_chunk):
            similarity = block @ bank[memory_start:memory_start + memory_chunk].T
            maximum = torch.maximum(maximum, similarity.max(dim=1).values)
        scores.append(1 - maximum.clamp(-1, 1))
    result = torch.cat(scores).reshape(height, width)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Nonfinite nearest-neighbour anomaly map")
    return result


def memory_distance_self_test():
    bank = normalized_features(torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 0.]]))
    query = torch.tensor([[[1., 0., 0.], [-1., 0., 0.]], [[0., 1., 1.], [1., 2., 3.]]])
    expected = (1 - (normalized_features(query.reshape(-1, 3)) @ bank.T).max(1).values).reshape(2, 2)
    actual = nearest_cosine_map(query, bank, query_chunk=1, memory_chunk=2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
    selected = sample_spatial_features(torch.arange(16 * 24).reshape(16, 24, 1), 14)
    assert selected.shape == (196, 1)
    assert selected[0].item() == 0 and selected[-1].item() == 16 * 24 - 1
    return {"passed": True, "exact_chunked_cosine": True, "sampling_limit": 196}


def actual_processor_order_self_test(processor):
    """Check the installed processor using a real pixel position grid on CPU."""
    positions = np.arange(24, dtype=np.uint8).reshape(4, 6)
    pixels = np.repeat(np.repeat(positions, 16, axis=0), 16, axis=1)
    pixels = np.repeat(pixels[:, :, None], 3, axis=-1)
    encoded = processor(images=Image.fromarray(pixels), do_resize=False,
                        do_rescale=False, do_normalize=False, return_tensors="pt")
    patch_means = encoded["pixel_values"].float().mean(-1, keepdim=True)
    restored = restore_premerger_grid(patch_means, encoded["image_grid_thw"][0].tolist(), 2)
    torch.testing.assert_close(restored[0, :, :, 0], torch.from_numpy(positions).float(), rtol=0, atol=0)
    return {"passed": True, "actual_grid_thw": encoded["image_grid_thw"].tolist(),
            "fixture": "4x6 distinct 16x16 RGB pixel patches, real processor, no resize/normalization"}


def visual_source_audit(model_path):
    report = source_audit(model_path)
    from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    processor_source = Path(inspect.getsourcefile(Qwen2VLImageProcessorFast))
    report["image_processor_source"] = {
        "file": str(processor_source), "sha256": hashlib.sha256(processor_source.read_bytes()).hexdigest(),
        "preprocess_sha256": hashlib.sha256(inspect.getsource(Qwen2VLImageProcessorFast._preprocess).encode()).hexdigest(),
        "expected_sha256": AUDITED_PROCESSOR_SHA256,
    }
    report["passed"] = report["passed"] and report["image_processor_source"]["sha256"] == AUDITED_PROCESSOR_SHA256
    report["visual_from_config_source_sha256"] = hashlib.sha256(inspect.getsource(Qwen3VLVisionModel._from_config).encode()).hexdigest()
    index = json.loads((Path(model_path) / "model.safetensors.index.json").read_text())
    visual_keys = {key: shard for key, shard in index["weight_map"].items() if key.startswith("model.visual.")}
    if not visual_keys:
        raise ValueError("Checkpoint index has no model.visual weights")
    report["visual_checkpoint"] = {"n_tensors": len(visual_keys), "shards": sorted(set(visual_keys.values())),
                                    "index_sha256": hashlib.sha256((Path(model_path) / "model.safetensors.index.json").read_bytes()).hexdigest()}
    report["configuration_sha256"] = {
        filename: hashlib.sha256((Path(model_path) / filename).read_bytes()).hexdigest()
        for filename in ("config.json", "preprocessor_config.json")
    }
    return report


def load_visual_tower(model_path, device):
    """Load ONLY indexed model.visual tensors; never instantiate the language model."""
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    root = Path(model_path)
    config = AutoConfig.from_pretrained(root, local_files_only=True).vision_config
    visual = Qwen3VLVisionModel._from_config(config, dtype=torch.bfloat16, attn_implementation="sdpa")
    weights = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.visual."
    by_shard = {}
    for name, shard in weights.items():
        if name.startswith(prefix):
            by_shard.setdefault(shard, []).append(name)
    state = {}
    for shard, names in sorted(by_shard.items()):
        with safe_open(root / shard, framework="pt", device="cpu") as tensors:
            for name in names:
                tensor = tensors.get_tensor(name)
                if tensor.dtype != torch.bfloat16:
                    raise ValueError("Visual checkpoint tensor must be BF16: %s %s" % (name, tensor.dtype))
                state[name[len(prefix):]] = tensor
    visual.load_state_dict(state, strict=True)
    del state
    visual.eval().requires_grad_(False)
    if any(parameter.dtype != torch.bfloat16 for parameter in visual.parameters()):
        raise ValueError("Vision/merger/DeepStack parameters must all remain BF16")
    visual = visual.to(device=device)
    report = {"parameter_count": sum(parameter.numel() for parameter in visual.parameters()),
              "parameter_dtype": "torch.bfloat16", "loaded_tensors": sum(map(len, by_shard.values())),
              "loaded_shards": sorted(by_shard), "language_model_instantiated": False,
              "patch_size": visual.patch_size, "spatial_merge_size": visual.spatial_merge_size,
              "vision_hidden_size": config.hidden_size, "merged_hidden_size": config.out_hidden_size,
              "deepstack_visual_indexes": list(config.deepstack_visual_indexes)}
    return visual, report


class VisualExtractor:
    def __init__(self, visual, image_processor, device):
        self.visual = visual
        self.processor = image_processor
        self.device = device
        self.premerger = None
        self.grids = Counter()
        self.hook = visual.merger.register_forward_pre_hook(self._capture)

    def _capture(self, module, args):
        if len(args) != 1 or self.premerger is not None:
            raise ValueError("Unexpected main-merger invocation")
        self.premerger = args[0].detach()

    @torch.inference_mode()
    def __call__(self, path, resolution):
        factor = int(self.visual.patch_size) * int(self.visual.spatial_merge_size)
        if not isinstance(resolution, int) or isinstance(resolution, bool) or resolution < 1 or resolution % factor:
            raise ValueError("Explicit square resolution must be a positive multiple of patch_size * spatial_merge_size")
        image = load_image_pil(path, size=resolution)
        encoded = self.processor(images=image, do_resize=False, return_tensors="pt")
        grid = encoded["image_grid_thw"]
        if grid.shape != (1, 3) or int(grid[0, 0]) != 1:
            raise ValueError("Memory probe expects one still image per visual call")
        actual = tuple(map(int, grid[0].tolist()))
        actual_hw = [actual[1] * self.visual.patch_size, actual[2] * self.visual.patch_size]
        if actual_hw != [resolution, resolution]:
            raise ValueError("Processor changed explicit square resolution: requested %s, actual %s" % (resolution, actual_hw))
        pixels = encoded["pixel_values"].to(device=self.device, dtype=torch.bfloat16)
        self.premerger = None
        post, deepstack = self.visual(pixels, grid_thw=grid.to(self.device))
        if self.premerger is None:
            raise RuntimeError("Main-merger pre-hook did not capture final visual features")
        before = restore_premerger_grid(self.premerger, actual, self.visual.spatial_merge_size)
        after = restore_postmerger_grid(post, actual, self.visual.spatial_merge_size)
        self.premerger = None
        self.grids[actual] += 1
        metadata = {"image_grid_thw": list(actual), "pixel_values_shape": list(pixels.shape),
                    "requested_image_hw": [resolution, resolution], "actual_image_hw": actual_hw,
                    "geometry_policy": EXPLICIT_GEOMETRY_POLICY, "processor_do_resize": False,
                    "pre_grid_shape": list(before.shape), "post_grid_shape": list(after.shape),
                    "deepstack_shapes_observed": [list(value.shape) for value in deepstack]}
        return {"pre": before[0], "post": after[0]}, metadata

    def close(self):
        self.hook.remove()
        self.premerger = None


def extractor_geometry_self_test():
    """CPU real-image/fake-processor check of order and exact 224/448/672 geometry."""
    class Processor:
        def __init__(self):
            self.calls = []
            self.wrong_grid = False

        def __call__(self, images, do_resize, return_tensors):
            assert do_resize is False and return_tensors == "pt"
            self.calls.append(images.size)
            height, width = images.height // 16, images.width // 16
            if self.wrong_grid:
                height += 2
            positions = torch.arange(height * width, dtype=torch.float32).reshape(1, height, width, 1)
            block = positions.view(1, height // 2, 2, width // 2, 2, 1).permute(0, 1, 3, 2, 4, 5).reshape(-1, 1)
            return {"pixel_values": block, "image_grid_thw": torch.tensor([[1, height, width]])}

    class Visual(torch.nn.Module):
        patch_size, spatial_merge_size = 16, 2

        def __init__(self):
            super().__init__()
            self.merger = torch.nn.Identity()

        def forward(self, pixels, grid_thw):
            assert pixels.dtype == torch.bfloat16 and not torch.is_grad_enabled()
            pre = self.merger(pixels)
            return pre.reshape(-1, 4, 1).float().mean(1).to(pre.dtype), []

    processor, visual = Processor(), Visual()
    extractor = VisualExtractor(visual, processor, "cpu")
    cases = []
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "source.png"
        Image.new("RGB", (91, 57), (100, 140, 190)).save(path)
        try:
            for resolution in (224, 448, 672):
                features, metadata = extractor(path, resolution)
                height = resolution // 16
                expected = torch.arange(height * height, dtype=torch.float32).to(torch.bfloat16).reshape(height, height, 1)
                torch.testing.assert_close(features["pre"], expected, rtol=0, atol=0)
                assert metadata["actual_image_hw"] == metadata["requested_image_hw"] == [resolution, resolution]
                assert metadata["processor_do_resize"] is False
                cases.append({"resolution": resolution, "grid_thw": metadata["image_grid_thw"]})
            processor.wrong_grid = True
            try:
                extractor(path, 224)
            except ValueError:
                pass
            else:
                raise AssertionError("A processor changing the requested square geometry must fail")
            processor.wrong_grid = False
            try:
                extractor(path, 225)
            except ValueError:
                pass
            else:
                raise AssertionError("An incompatible explicit grid size must fail")
        finally:
            extractor.close()
    assert not visual.merger._forward_pre_hooks
    return {"passed": True, "cases": cases, "processor_resize_disabled": True,
            "actual_size_mismatch_rejected": True, "merge_order_restored": True}


def choose_supports(train, validation, shots, seed):
    train_paths = {item[0] for item in train}
    validation_paths = {item[0] for item in validation}
    if train_paths & validation_paths:
        raise ValueError("Training and validation image sources overlap")
    classes = sorted({item[1] for item in validation})
    supports = {}
    for category in classes:
        normal = sorted(item[0] for item in train if item[1] == category and item[2] == 0)
        if len(normal) < max(shots):
            raise ValueError("Insufficient training normals for %s: %d" % (category, len(normal)))
        # Stable per-class stream; the first 16 of 64 supports are exactly the 16-shot bank.
        category_seed = int(hashlib.sha256((str(seed) + ":" + category).encode()).hexdigest(), 16)
        supports[category] = random.Random(category_seed).sample(normal, max(shots))
    return supports


@torch.inference_mode()
def evaluate_resolution(extractor, validation, supports, args, resolution):
    groups = [(feature, shot) for feature in args.features for shot in args.shots]
    predictions = {group: [] for group in groups}
    memory_report = {}
    observed_geometry = {}
    for category in sorted(supports):
        samples = {feature: [] for feature in args.features}
        for path in supports[category]:
            features, geometry = extractor(path, resolution)
            observed_geometry[json.dumps(geometry, sort_keys=True)] = geometry
            for feature in args.features:
                samples[feature].append(sample_spatial_features(features[feature], args.memory_side).cpu())
            del features
        banks = {}
        memory_report[category] = {}
        for feature, shot in groups:
            bank = normalized_features(torch.cat(samples[feature][:shot]).to(args.device))
            banks[(feature, shot)] = bank
            memory_report[category][feature + ":" + str(shot)] = {
                "normal_images": shot, "patches": len(bank), "channels": bank.shape[1],
                "patches_per_image": [len(value) for value in samples[feature][:shot]],
                "bank_gpu_bytes": bank.nelement() * bank.element_size(),
            }
        del samples
        category_validation = [item for item in validation if item[1] == category]
        for path, _, label in category_validation:
            features, geometry = extractor(path, resolution)
            observed_geometry[json.dumps(geometry, sort_keys=True)] = geometry
            # A missing anomaly annotation raises; palette indices are binarized before resize.
            truth = load_mask_tensor(mask_path_for_image(path, label), device="cpu", size=224)[0].numpy()
            for feature, shot in groups:
                score = nearest_cosine_map(features[feature], banks[(feature, shot)], args.query_chunk, args.memory_chunk)
                score = F.interpolate(score[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
                predictions[(feature, shot)].append((category, label, truth, score))
            del features
        del banks, bank
        print(json.dumps({"resolution": resolution, "category": category,
                          "support_images": len(supports[category]), "validation_images": len(category_validation)}), flush=True)
    metrics = {}
    for group in groups:
        feature, shot = group
        result = pixel_metrics(predictions.pop(group))
        result.update({"feature": feature, "shots": shot, "requested_square_resolution": resolution,
                       "feature_pipeline_version": PIPELINE_VERSION, "split": "val"})
        metrics[feature + ":" + str(shot)] = result
    return {"requested_square_resolution": resolution, "actual_geometries": list(observed_geometry.values()),
            "memory": memory_report, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run", action="store_true", help="Explicitly enable visual-tower GPU inference")
    parser.add_argument("--cpu-test-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[224, 448, 672])
    parser.add_argument("--shots", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--features", nargs="+", choices=["pre", "post"], default=["pre", "post"])
    parser.add_argument("--memory-side", type=int, default=14)
    parser.add_argument("--query-chunk", type=int, default=256)
    parser.add_argument("--memory-chunk", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="runs/native_memory/val.json")
    args = parser.parse_args()
    if min(args.resolutions + args.shots + [args.memory_side, args.query_chunk, args.memory_chunk]) < 1:
        parser.error("Resolution, shots, sampling side, and chunk sizes must be positive")
    args.resolutions = sorted(set(args.resolutions))
    args.shots = sorted(set(args.shots))
    args.features = list(dict.fromkeys(args.features))
    cpu = {"spatial_order": spatial_order_self_test(), "normal_memory": memory_distance_self_test(),
           "explicit_geometry": extractor_geometry_self_test()}
    if args.cpu_test_only:
        print(json.dumps(cpu, indent=2))
        return
    report = {"pipeline_version": PIPELINE_VERSION, "config": vars(args), "cpu_tests": cpu,
              "source_audit": visual_source_audit(args.model), "resolutions": [],
              "protocol": "class-aware frozen Qwen visual nearest-normal-patch; train normals only; validation only",
              "feature_disk_cache": False, "head_training": False, "test_set_used": False,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    atomic_json(report, args.out)
    if not report["source_audit"]["passed"]:
        raise RuntimeError("Qwen source differs from the audited 4.57.1 implementation")
    if not args.run:
        print(json.dumps({"source_audit_passed": True, "gpu_model_loaded": False, "report": args.out}))
        return
    if not args.device.startswith("cuda") or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("--run requires a CUDA device with BF16 support")
    from transformers import AutoImageProcessor
    from split_visa import splits
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train, validation, _ = splits()  # Only train/val images are read below.
    supports = choose_supports(train, validation, args.shots, args.seed)
    manifest = {"supports": supports, "validation": validation}
    report["data"] = {"train_images": len(train), "validation_images": len(validation),
                      "validation_anomaly_images": sum(item[2] for item in validation),
                      "support_paths": supports,
                      "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()}
    processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True, use_fast=True)
    report["image_processor"] = processor.to_dict()
    report["actual_processor_order_cpu_test"] = actual_processor_order_self_test(processor)
    visual, report["visual_model"] = load_visual_tower(args.model, args.device)
    extractor = VisualExtractor(visual, processor, args.device)
    try:
        for resolution in args.resolutions:
            result = evaluate_resolution(extractor, validation, supports, args, resolution)
            report["resolutions"].append(result)
            report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
            atomic_json(report, args.out)
            print(json.dumps({"resolution": resolution, "metrics": result["metrics"]}), flush=True)
            gc.collect()
        report["complete"] = True
    except Exception as error:
        report["complete"] = False
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        extractor.close()
        atomic_json(report, args.out)
    print(json.dumps({"complete": True, "report": args.out, "test_set_used": False}), flush=True)


if __name__ == "__main__":
    main()
