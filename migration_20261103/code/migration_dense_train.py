"""Frozen visual tower + a small spatial head, supervised VisA train/val.

The baseline (--distill-weight 0) and distillation arm use the same feature
source, head, seed, deterministic sample plan, optimizer and schedule. Pre
and post features have different input widths and therefore different head
parameter counts; neither head is a capacity-matched replacement of BigDec.
ImageBind1280 and Qwen pre1152 use identical head structure with 697345 and
664577 parameters respectively (4.93% difference); encoder precision differs.

Defaults only write a source/config/plan preview. --run enables GPU training;
--eval --run validates a strictly compatible head checkpoint, never test.
Qwen pre/post use its BF16 visual tower; imagebind uses the FP16 final
vision block (zero-based layer 31) at fixed square224. No language model
or pretrained anomaly decoder is loaded. Preview instantiates neither tower.

Examples:
  python migration_dense_train.py --cpu-test-only
  python migration_dense_train.py --features pre --size 448 --out runs/dense_gt
  python migration_dense_train.py --run --features pre --size 448 --out runs/dense_gt
  python migration_dense_train.py --features imagebind --size 224 --out runs/ib_dense_gt
  python migration_dense_train.py --run --features pre --size 448 \
      --distill-weight .5 --teacher-maps runs/teacher_train/manifest.json --out runs/dense_kd

Teacher cache contract (read only, no teacher inference here):
  {"schema_version":"visa-teacher-prob-v1", "split":"train",
   "coordinate_system":"direct-square-224",
   "visa_pipeline_version": <visa_pipeline.PIPELINE_VERSION>,
   "teacher_checkpoint": <provenance>, "maps": {<absolute train image>:<relative .npy>}}
Exported .npy files are finite float32 probability [224,224] in [0,1]. The
export payload, decoder/source-cache identity, current plan and planned map
file/array hashes are verified before GPU construction. All keys must belong
to train, and all images in this run's deterministic plan must be covered.
Only training reads this cache. The explicit soft loss is mean absolute
error(sigmoid(student_logits), teacher_probability), added to GT focal+Dice.
"""
import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from migration_native_parity import DEFAULT_MODEL, atomic_json
from migration_visual_probe import (EXPLICIT_GEOMETRY_POLICY, VisualExtractor, extractor_geometry_self_test,
                                    load_visual_tower, visual_source_audit)
from visa_pipeline import PIPELINE_VERSION as VISA_PIPELINE_VERSION
from visa_pipeline import load_image_tensor, load_mask_tensor, mask_path_for_image, pixel_metrics


PIPELINE_VERSION = "qwen3vl-frozen-singlelayer-spatialhead-v1"
IMAGEBIND_PIPELINE_VERSION = "imagebind-fp16-lastlayer-spatialhead-v1"
DEFAULT_IMAGEBIND_REPO = os.environ.get("ANOMALYGPT_REPO", "/root/autodl-tmp/AnomalyGPT")
AUDITED_IMAGEBIND_WEIGHT_SHA256 = "d6f6c22bedcc90708448d5d2fbb7b2db9c73f505dc89bd0b2e09b23af1b62157"
AUDITED_IMAGEBIND_SOURCE_SHA256 = {
    "__init__.py": "8426c3ef2aa09b2452e66be3734570d3433ee2bdd729cb4e73180cdefea5dba4",
    "models/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "models/helpers.py": "ea157fc6f95a806305c680fe56bb8aaf0b6fc5dfb83e80ccd80af2e802d78bb0",
    "models/imagebind_model.py": "65f1ca944b157f53b898075f450a01173e0761dd98524d251c9fcb219368cc5b",
    "models/multimodal_preprocessors.py": "fab69ddfe599b8ca471086b24db2c9c171622fa61478a10d73ef1641e4764930",
    "models/transformer.py": "1f005158baca13d10dbe15fdc710050961f5c3983a7b6ad5160da4426d8eec25",
}


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def head_pipeline_version(compatibility):
    return IMAGEBIND_PIPELINE_VERSION if compatibility.get("features") == "imagebind" else PIPELINE_VERSION


def imagebind_source_audit(repo, weights=None):
    """Hash source and checkpoint bytes only; no import/construction/torch.load."""
    repo = Path(repo).resolve()
    package = repo / "code/code/model/ImageBind"
    weights = Path(weights).resolve() if weights else repo / "pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth"
    source = {name: file_sha(package / name) for name in AUDITED_IMAGEBIND_SOURCE_SHA256}
    stat = weights.stat()
    weight_sha = file_sha(weights)
    return {"passed": source == AUDITED_IMAGEBIND_SOURCE_SHA256 and weight_sha == AUDITED_IMAGEBIND_WEIGHT_SHA256,
            "family": "ImageBind-Huge", "package": str(package), "source_files_sha256": source,
            "expected_source_files_sha256": AUDITED_IMAGEBIND_SOURCE_SHA256,
            "source_sha256": json_sha(source),
            "checkpoint": {"path": str(weights), "sha256": weight_sha,
                           "expected_sha256": AUDITED_IMAGEBIND_WEIGHT_SHA256,
                           "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            "vision_config": {"input_hw": [224, 224], "patch_size": 14, "grid_hw": [16, 16],
                              "hidden_size": 1280, "depth": 32, "out_layers": [31]},
            "model_instantiated": False}


def restore_imagebind_last_grid(raw, batch_size):
    """Layer31 before the trunk's output transpose: [CLS+patch,B,C] -> row-major grid."""
    if (not isinstance(raw, torch.Tensor) or raw.ndim != 3
            or tuple(raw.shape) != (257, batch_size, 1280) or batch_size < 1):
        raise ValueError("ImageBind layer31 must be [257,B,1280], CLS followed by 16x16 patches")
    return raw.transpose(0, 1)[:, 1:, :].contiguous().reshape(batch_size, 16, 16, 1280)


def load_imagebind_tower(audit, device):
    """Strictly load the audited checkpoint, then retain only frozen FP16 vision modules."""
    if not audit["passed"]:
        raise ValueError("ImageBind source/checkpoint audit failed")
    package = Path(audit["package"])
    if {name: file_sha(package / name) for name in AUDITED_IMAGEBIND_SOURCE_SHA256} != audit["source_files_sha256"]:
        raise ValueError("ImageBind source changed after audit")
    weights = Path(audit["checkpoint"]["path"])
    stat = weights.stat()
    if (stat.st_size, stat.st_mtime_ns) != (audit["checkpoint"]["bytes"], audit["checkpoint"]["mtime_ns"]):
        raise ValueError("ImageBind checkpoint changed after audit")
    sys.path.insert(0, str(package.parent))
    from ImageBind import imagebind_model
    if Path(imagebind_model.__file__).resolve() != package / "models/imagebind_model.py":
        raise ValueError("Imported ImageBind does not match the audited package")
    encoder, _ = imagebind_model.imagebind_huge({"layers": [31]})
    if encoder.out_layers != [31]:
        raise ValueError("ImageBind must return only its final visual block")
    encoder.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    # Other modalities are unused; removing them leaves the vision path unchanged.
    for name in ("modality_preprocessors", "modality_trunks", "modality_heads", "modality_postprocessors"):
        modules = getattr(encoder, name)
        setattr(encoder, name, nn.ModuleDict({"vision": modules["vision"]}))
    encoder = encoder.half().eval().requires_grad_(False).to(device=device)
    if any(parameter.dtype != torch.float16 or parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("ImageBind vision parameters must be frozen FP16")
    report = {"parameter_count": sum(parameter.numel() for parameter in encoder.parameters()),
              "parameter_dtype": "torch.float16", "vision_only_retained": True,
              "checkpoint_sha256": audit["checkpoint"]["sha256"], "out_layers": [31],
              "feature_stage": "visual block31 output before post_transformer_layer",
              "language_model_instantiated": False, "pretrained_anomaly_decoder_loaded": False}
    return encoder, report


class ImageBindExtractor:
    def __init__(self, encoder, device):
        self.encoder = encoder
        self.device = device

    @torch.inference_mode()
    def __call__(self, path, resolution):
        if resolution != 224:
            raise ValueError("ImageBind has fixed square224 input; --size must be 224")
        pixels = load_image_tensor(path, device=self.device, size=224).to(dtype=torch.float16)
        result = self.encoder({"vision": pixels})["vision"]
        if not isinstance(result, (list, tuple)) or len(result) != 2:
            raise ValueError("Unexpected ImageBind vision return contract")
        levels = result[1]
        if not isinstance(levels, (list, tuple)) or len(levels) != 1:
            raise ValueError("ImageBind must return exactly the requested single layer31")
        grid = restore_imagebind_last_grid(levels[0], batch_size=1)
        if grid.dtype != torch.float16:
            raise ValueError("ImageBind layer31 features must remain FP16")
        metadata = {"actual_image_hw": [224, 224], "pixel_values_shape": list(pixels.shape),
                    "imagebind_grid_shape": list(grid.shape), "raw_feature_shape": list(levels[0].shape),
                    "feature_grid_hw": [16, 16], "patch_size": 14, "out_layers": [31],
                    "feature_dtype": str(grid.dtype), "coordinate_system": "direct-square-224"}
        return {"imagebind": grid[0]}, metadata

    def close(self):
        pass


def feature_configuration(feature, size, audit):
    """Bind checkpoints to the feature source and explicit processor geometry."""
    if feature == "imagebind":
        if size != 224:
            raise ValueError("ImageBind has fixed square224 input; --size must be 224")
        channels = 1280
    else:
        vision = audit["model_config"]["vision_config"]
        channels = vision["hidden_size"] if feature == "pre" else vision["out_hidden_size"]
    head_config = {"in_channels": channels, "width": 256,
                   "normalization": "per_patch_layernorm_no_affine", "spatial_sizes": [56, 112, 224]}
    compatibility = {"features": feature, "size": size, "head_config": head_config,
                     "source_sha256": audit["source_sha256"]}
    if feature == "imagebind":
        compatibility.update({"source_files_sha256": audit["source_files_sha256"],
                              "checkpoint_sha256": audit["checkpoint"]["sha256"],
                              "vision_config": audit["vision_config"],
                              "feature_stage": "visual block31 output before post_transformer_layer",
                              "encoder_dtype": "torch.float16", "cache_dtype": "torch.float16",
                              "preprocessing": "RGB direct bicubic square224; ImageBind mean/std",
                              "preprocessing_source_sha256": file_sha(Path(__file__).with_name("visa_pipeline.py"))})
    else:
        compatibility.update({"processor_sha256": audit["image_processor_source"]["sha256"],
                              "configuration_sha256": audit["configuration_sha256"],
                              "geometry_policy": EXPLICIT_GEOMETRY_POLICY,
                              "extractor_source_sha256": file_sha(Path(__file__).with_name("migration_visual_probe.py"))})
    compatibility["visa_pipeline_version"] = VISA_PIPELINE_VERSION
    return head_config, compatibility


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class SpatialHead(nn.Module):
    """Per-patch normalization -> 1x1/256 -> spatial convolutions/upsampling."""
    def __init__(self, in_channels, width=256):
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.project = nn.Conv2d(in_channels, width, 1)
        self.refine56 = nn.Conv2d(width, 128, 3, padding=1)
        self.refine112 = nn.Conv2d(128, 64, 3, padding=1)
        self.readout = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, features):
        if features.ndim != 4 or features.shape[-1] != self.in_channels:
            raise ValueError("SpatialHead expects [B,H,W,C] with its configured channels")
        # No learned normalization; the same transformation is used in GT/KD arms.
        x = F.layer_norm(features.float(), (self.in_channels,)).permute(0, 3, 1, 2).contiguous()
        x = F.gelu(self.project(x))
        x = F.interpolate(x, size=(56, 56), mode="bilinear", align_corners=False)
        x = F.gelu(self.refine56(x))
        x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
        x = F.gelu(self.refine112(x))
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.readout(x)


def supervised_loss(logits, truth):
    if logits.shape != truth.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("GT and logits must have identical [B,1,H,W] shapes")
    bce = F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
    probability = logits.sigmoid()
    pt = probability * truth + (1 - probability) * (1 - truth)
    focal = ((1 - pt).square() * bce).mean()
    dimensions = (1, 2, 3)
    dice = 1 - (2 * (probability * truth).sum(dimensions) + 1) / (
        probability.sum(dimensions) + truth.sum(dimensions) + 1
    )
    return focal + dice.mean()


def soft_teacher_loss(logits, teacher_probability):
    if logits.shape != teacher_probability.shape:
        raise ValueError("Teacher and student must have the same 224 square coordinates")
    if not bool(torch.isfinite(teacher_probability).all()):
        raise ValueError("Nonfinite teacher probabilities")
    if bool(((teacher_probability < 0) | (teacher_probability > 1)).any()):
        raise ValueError("Teacher cache must contain probabilities in [0,1]")
    return F.l1_loss(logits.sigmoid(), teacher_probability)


class FeatureLRU:
    """CPU feature cache, preserving each tower's dtype, with per-batch pinning."""
    def __init__(self, extractor, feature, size, device, cache_gb=1.0):
        self.extractor = extractor
        self.feature = feature
        self.dtype = torch.float16 if feature == "imagebind" else torch.bfloat16
        self.size = size
        self.device = device
        self.limit = int(cache_gb * 1024 ** 3)
        self.cache = OrderedDict()
        self.nbytes = 0
        self.geometry = None

    def __call__(self, items):
        # Pin requested hits even if inserting a miss evicts their LRU entries.
        available = {path: self.cache[path] for path, _, _ in items if path in self.cache}
        missing = list(dict.fromkeys(path for path, _, _ in items if path not in available))
        for path in missing:
            features, geometry = self.extractor(path, self.size)
            # Frozen extractors use inference_mode. This clone is intentionally
            # OUTSIDE that context, so the trainable head may save its input.
            value = features[self.feature].detach().to(device="cpu", dtype=self.dtype).clone()
            del features
            if torch.is_inference(value):
                raise RuntimeError("Frozen cached features must be ordinary tensors before head training")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Nonfinite frozen visual features")
            if self.geometry is None:
                self.geometry = geometry
            available[path] = value
            nbytes = value.nelement() * value.element_size()
            if nbytes <= self.limit:
                while self.cache and self.nbytes + nbytes > self.limit:
                    _, old = self.cache.popitem(last=False)
                    self.nbytes -= old.nelement() * old.element_size()
                self.cache[path] = value
                self.nbytes += nbytes
        values = []
        for path, _, _ in items:
            values.append(available[path])
            if path in self.cache:
                self.cache.move_to_end(path)
        return torch.stack(values).to(self.device)


def make_training_plan(train, steps, batch_size, seed):
    normal = [item for item in train if item[2] == 0]
    anomaly = [item for item in train if item[2] == 1]
    if not normal or not anomaly:
        raise ValueError("Balanced supervised training requires normal and anomaly training images")
    rng = random.Random(seed)
    return [[rng.choice(anomaly if rng.random() < .5 else normal) for _ in range(batch_size)]
            for _ in range(steps)]


def targets(items, device):
    return torch.stack([load_mask_tensor(mask_path_for_image(path, label), device=device, size=224)
                        for path, _, label in items])


class TeacherCache:
    """Read only strict train-only teacher probabilities; validation has no access."""
    def __init__(self, manifest_path, train, validation, required_paths, cache_mb=128,
                 expected_plan_sha256=None, expected_decoder_sha256=None):
        self.path = Path(manifest_path)
        self.manifest = json.loads(self.path.read_text())
        required = {"schema_version": "visa-teacher-prob-v1", "split": "train",
                    "coordinate_system": "direct-square-224", "visa_pipeline_version": VISA_PIPELINE_VERSION}
        for key, expected in required.items():
            if self.manifest.get(key) != expected:
                raise ValueError("Teacher manifest %s must be %r" % (key, expected))
        if not self.manifest.get("teacher_checkpoint"):
            raise ValueError("Teacher manifest requires checkpoint provenance")
        self.maps = self.manifest["maps"]
        if not isinstance(self.maps, dict) or not self.maps or any(
                not isinstance(path, str) or not isinstance(name, str) or not name for path, name in self.maps.items()):
            raise ValueError("Teacher maps must be a nonempty image-path to .npy-path object")
        required_paths = set(required_paths)
        train_paths = {item[0] for item in train}
        train_items = {item[0]: (item[1], item[2]) for item in train}
        validation_paths = {item[0] for item in validation}
        if set(self.maps) - train_paths or set(self.maps) & validation_paths:
            raise ValueError("Teacher cache contains image keys outside the training split")
        missing = required_paths - set(self.maps)
        if missing:
            raise FileNotFoundError("Teacher cache misses %d planned training images; example: %s" % (len(missing), min(missing)))
        self.allowed = train_paths
        self.map_records = self.manifest.get("map_files")
        self.exported = any(key in self.manifest for key in
                            ("export_version", "payload_sha256", "training_plan", "map_files", "teacher_cache"))
        self.audit = {"exported_manifest": self.exported, "required_images": len(required_paths),
                      "payload_verified": False, "planned_map_files_verified": 0}
        if self.exported:
            if self.manifest.get("export_version") != "planned-train-teacher-export-v1":
                raise ValueError("Unsupported teacher export version")
            payload = dict(self.manifest)
            payload_sha = payload.pop("payload_sha256", None)
            if payload_sha != json_sha(payload):
                raise ValueError("Teacher manifest payload SHA256 mismatch")
            self.audit["payload_verified"] = True
            plan = self.manifest.get("training_plan", {})
            if not isinstance(plan, dict) or not self._is_sha(plan.get("batches_sha256")):
                raise ValueError("Exported teacher manifest requires its training-plan batches SHA256")
            if expected_plan_sha256 is not None and plan["batches_sha256"] != expected_plan_sha256:
                raise ValueError("Teacher export training-plan SHA256 differs from this run")
            teacher = self.manifest["teacher_checkpoint"]
            cached = self.manifest.get("teacher_cache", {})
            if not isinstance(teacher, dict) or not isinstance(cached, dict):
                raise ValueError("Exported teacher requires checkpoint and source-cache provenance objects")
            decoder_sha = teacher.get("decoder_sha256")
            if not self._is_sha(decoder_sha) or decoder_sha != cached.get("decoder_sha256"):
                raise ValueError("Teacher checkpoint and source cache decoder SHA256 disagree")
            if expected_decoder_sha256 is not None and decoder_sha != expected_decoder_sha256:
                raise ValueError("Teacher decoder SHA256 differs from the expected frozen decoder")
            for key in ("cache_sha256", "manifest_sha256"):
                if not self._is_sha(cached.get(key)):
                    raise ValueError("Teacher source cache requires " + key)
            if cached.get("pipeline_version") != VISA_PIPELINE_VERSION or cached.get("geometry") != "square224":
                raise ValueError("Exported teacher source cache must use the current square224 pipeline")
            if self.manifest.get("test_or_validation_maps_exported") is not False:
                raise ValueError("Teacher export must explicitly contain no validation/test maps")
            if not isinstance(self.map_records, dict) or set(self.map_records) != set(self.maps):
                raise ValueError("Exported teacher map_files must match every maps key")
            if self.manifest.get("n_images") != len(self.maps):
                raise ValueError("Teacher export n_images disagrees with its maps")
            for path, record in self.map_records.items():
                category, label = train_items[path]
                if (not isinstance(record, dict) or record.get("file") != self.maps[path]
                        or record.get("shape") != [224, 224] or record.get("dtype") != "float32"
                        or record.get("split") != "train" or record.get("category") != category
                        or record.get("label") != label
                        or not self._is_sha(record.get("sha256")) or not self._is_sha(record.get("array_sha256"))):
                    raise ValueError("Exported teacher map metadata disagrees with the training image: " + path)
            self.audit.update({"training_plan_sha256": plan["batches_sha256"], "decoder_sha256": decoder_sha,
                               "teacher_cache_sha256": cached["cache_sha256"],
                               "teacher_cache_manifest_sha256": cached["manifest_sha256"]})
            # Detect replaced/corrupted arrays before any GPU construction.
            # Each subsequent cache miss rechecks the bytes it actually reads.
            for path in sorted(required_paths):
                self._read(path)
            self.audit["planned_map_files_verified"] = len(required_paths)
        elif expected_plan_sha256 is not None or expected_decoder_sha256 is not None:
            raise ValueError("Matched KD requires the exported teacher manifest with plan/decoder/map hashes")
        self.cache = OrderedDict()
        self.limit = int(cache_mb * 1024 ** 2)
        self.nbytes = 0

    @staticmethod
    def _is_sha(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None

    def _read(self, path):
        if path not in self.allowed:
            raise ValueError("Teacher cache is only available for training images")
        filename = self.path.parent / self.maps[path]
        record = None if self.map_records is None else self.map_records[path]
        if record is not None:
            if Path(self.maps[path]).is_absolute() or not filename.resolve().is_relative_to(self.path.parent.resolve()):
                raise ValueError("Exported teacher .npy path must remain inside the export directory")
            if file_sha(filename) != record["sha256"] or filename.stat().st_size != record.get("bytes"):
                raise ValueError("Teacher probability file SHA256/bytes mismatch: " + str(filename))
        if filename.suffix.lower() != ".npy":
            raise ValueError("Teacher probabilities must be .npy arrays")
        values = np.load(filename, allow_pickle=False)
        if values.shape != (224, 224) or values.dtype.kind != "f":
            raise ValueError("Teacher probability must be floating point [224,224]: %s" % filename)
        if not np.isfinite(values).all() or np.min(values) < 0 or np.max(values) > 1:
            raise ValueError("Teacher probability must be finite and in [0,1]: %s" % filename)
        if record is not None:
            if values.dtype != np.float32 or hashlib.sha256(values.tobytes(order="C")).hexdigest() != record["array_sha256"]:
                raise ValueError("Teacher probability float32 array SHA256 mismatch: " + str(filename))
        return torch.from_numpy(values.astype(np.float32, copy=True)).unsqueeze(0)

    def batch(self, items, device):
        pinned = {path: self.cache[path] for path, _, _ in items if path in self.cache}
        for path, _, _ in items:
            if path not in pinned:
                value = self._read(path)
                pinned[path] = value
                size = value.nelement() * value.element_size()
                if size <= self.limit:
                    while self.cache and self.nbytes + size > self.limit:
                        _, previous = self.cache.popitem(last=False)
                        self.nbytes -= previous.nelement() * previous.element_size()
                    self.cache[path] = value
                    self.nbytes += size
        for path, _, _ in items:
            if path in self.cache:
                self.cache.move_to_end(path)
        return torch.stack([pinned[path] for path, _, _ in items]).to(device)


@torch.no_grad()
def evaluate(head, features, validation, batch_size, device):
    """Validation uses GT exclusively; no teacher argument is accepted."""
    was_training = head.training
    head.eval()
    records = []
    total_loss = 0.
    for start in range(0, len(validation), batch_size):
        items = validation[start:start + batch_size]
        truth = targets(items, device)
        logits = head(features(items))
        total_loss += float(supervised_loss(logits, truth)) * len(items)
        probability = logits.sigmoid().cpu().numpy()
        for (_, category, label), gt, score in zip(items, truth.cpu().numpy(), probability):
            records.append((category, label, gt[0], score[0]))
    result = pixel_metrics(records)
    result.update({"loss_gt": total_loss / len(validation), "split": "val", "teacher_used": False})
    head.train(was_training)
    return result


def load_head_checkpoint(head, path, compatibility):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("pipeline_version") != head_pipeline_version(compatibility):
        raise ValueError("Checkpoint belongs to a different head pipeline")
    if checkpoint.get("compatibility") != compatibility:
        raise ValueError("Checkpoint feature/size/head/source configuration is incompatible")
    head.load_state_dict(checkpoint["head"], strict=True)
    return checkpoint


def imagebind_cpu_self_test():
    """CPU contract/preview tests with a tiny fake encoder and real image preprocessing."""
    import contextlib
    import copy
    import io
    import types
    from unittest.mock import patch
    from PIL import Image

    # Distinct batch and token values catch both a swapped axis and CLS leakage.
    raw = (torch.arange(257).view(257, 1, 1) * 4
           + torch.arange(2).view(1, 2, 1) + torch.arange(1280).view(1, 1, 1280) / 4096)
    grid = restore_imagebind_last_grid(raw, 2)
    for batch in range(2):
        torch.testing.assert_close(grid[batch].reshape(256, 1280), raw[1:, batch], rtol=0, atol=0)
    for bad in (raw.transpose(0, 1), raw[1:], raw[:, :, :-1]):
        try:
            restore_imagebind_last_grid(bad, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid ImageBind feature layout must fail")

    class FakeEncoder(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.out_layers = layers
            self.calls = 0
            self.bad_levels = False
            for name in ("modality_preprocessors", "modality_trunks", "modality_heads", "modality_postprocessors"):
                setattr(self, name, nn.ModuleDict({"vision": nn.Linear(1, 1), "text": nn.Linear(1, 1)}))

        def forward(self, inputs):
            assert set(inputs) == {"vision"}
            assert inputs["vision"].shape == (1, 3, 224, 224) and inputs["vision"].dtype == torch.float16
            assert not torch.is_grad_enabled()
            self.calls += 1
            # This FP16 value would change if accidentally rounded to BF16.
            level = torch.full((257, 1, 1280), 1 + 1 / 1024, dtype=torch.float16)
            level[0] = -10
            return {"vision": (torch.zeros(1, 1024), [level, level] if self.bad_levels else [level])}

    module = sys.modules[__name__]
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        package = temporary / "code/code/model/ImageBind"
        fixture_source = {}
        for name in AUDITED_IMAGEBIND_SOURCE_SHA256:
            path = package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# CPU source fixture: " + name + "\n")
            fixture_source[name] = file_sha(path)
        weights = temporary / "pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth"
        weights.parent.mkdir(parents=True)
        atomic_save(FakeEncoder([31]).state_dict(), weights)
        with patch.object(module, "AUDITED_IMAGEBIND_SOURCE_SHA256", fixture_source), \
                patch.object(module, "AUDITED_IMAGEBIND_WEIGHT_SHA256", file_sha(weights)):
            with patch.object(torch, "load", side_effect=AssertionError("Preview must not deserialize weights")):
                audit = imagebind_source_audit(temporary)
            assert audit["passed"] and audit["vision_config"]["out_layers"] == [31]
            fake_model_module = types.ModuleType("ImageBind.models.imagebind_model")
            fake_model_module.__file__ = str(package / "models/imagebind_model.py")
            fake_model_module.imagebind_huge = lambda args: (FakeEncoder(args["layers"]), 1024)
            fake_package = types.ModuleType("ImageBind")
            fake_package.imagebind_model = fake_model_module
            with patch.dict(sys.modules, {"ImageBind": fake_package}), patch.object(sys, "path", list(sys.path)):
                encoder, model_meta = load_imagebind_tower(audit, "cpu")
            assert model_meta["parameter_dtype"] == "torch.float16" and not encoder.training
            assert all(not p.requires_grad and p.dtype == torch.float16 for p in encoder.parameters())
            assert all(set(getattr(encoder, name)) == {"vision"} for name in (
                "modality_preprocessors", "modality_trunks", "modality_heads", "modality_postprocessors"))
            image_path = temporary / "nonsquare.png"
            Image.fromarray(np.zeros((30, 70, 3), dtype=np.uint8)).save(image_path)
            extractor = ImageBindExtractor(encoder, "cpu")
            cache = FeatureLRU(extractor, "imagebind", 224, "cpu", cache_gb=.01)
            values = cache([(str(image_path), "fixture", 0)] * 2)
            assert values.shape == (2, 16, 16, 1280) and values.dtype == torch.float16
            assert encoder.calls == 1 and values[0, 0, 0, 0].item() == 1 + 1 / 1024
            assert not torch.is_inference(values) and cache.geometry["actual_image_hw"] == [224, 224]
            head = SpatialHead(1280)
            assert sum(p.numel() for p in head.parameters()) == 697345
            assert sum(p.numel() for p in SpatialHead(1152).parameters()) == 664577
            head(values[:1]).square().mean().backward()
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters())
            encoder.bad_levels = True
            try:
                extractor(image_path, 224)
            except ValueError:
                pass
            else:
                raise AssertionError("Multiple ImageBind levels must be rejected")
            encoder.bad_levels = False
            try:
                extractor(image_path, 448)
            except ValueError:
                pass
            else:
                raise AssertionError("ImageBind resolution must be fixed at224")
            _, compatibility = feature_configuration("imagebind", 224, audit)
            atomic_save({"pipeline_version": IMAGEBIND_PIPELINE_VERSION, "compatibility": compatibility,
                         "head": head.state_dict()}, temporary / "ib_head.pt")
            load_head_checkpoint(SpatialHead(1280), temporary / "ib_head.pt", compatibility)
            for key in ("checkpoint_sha256", "size", "encoder_dtype", "vision_config"):
                changed = copy.deepcopy(compatibility)
                changed[key] = "incompatible"
                try:
                    load_head_checkpoint(SpatialHead(1280), temporary / "ib_head.pt", changed)
                except ValueError:
                    pass
                else:
                    raise AssertionError("ImageBind strict checkpoint compatibility must reject " + key)
            # Main preview must use no Qwen audit, model construction, weight load, or CUDA check.
            argv = ["migration_dense_train.py", "--features", "imagebind", "--size", "224",
                    "--imagebind-repo", str(temporary), "--steps", "2", "--out", str(temporary / "preview")]
            fixture_splits = ([("train0", "fixture", 0), ("train1", "fixture", 1)], [("val1", "fixture", 1)], [])
            with patch.object(sys, "argv", argv), patch("split_visa.splits", return_value=fixture_splits), \
                    patch.object(module, "visual_source_audit", side_effect=AssertionError("No Qwen audit for ImageBind")), \
                    patch.object(module, "load_imagebind_tower", side_effect=AssertionError("No preview model loading")), \
                    patch.object(torch, "load", side_effect=AssertionError("No preview weight deserialization")), \
                    patch.object(torch.cuda, "is_available", side_effect=AssertionError("No preview CUDA check")), \
                    contextlib.redirect_stdout(io.StringIO()):
                main()
            preview = json.loads((temporary / "preview/run.json").read_text())
            assert preview["compatibility"] == compatibility and preview["test_set_used"] is False
            weights.write_bytes(weights.read_bytes() + b"changed")
            assert imagebind_source_audit(temporary)["passed"] is False
    # Qwen compatibility explicitly binds the corrected no-second-resize geometry.
    qwen_audit = {"model_config": {"vision_config": {"hidden_size": 1152, "out_hidden_size": 4096}},
                  "source_sha256": "qwen-source", "image_processor_source": {"sha256": "qwen-processor"},
                  "configuration_sha256": {"config.json": "qwen-config"}}
    for feature, channels in (("pre", 1152), ("post", 4096)):
        head_config, compatibility = feature_configuration(feature, 448, qwen_audit)
        assert head_config["in_channels"] == channels
        assert compatibility == {"features": feature, "size": 448, "head_config": head_config,
                                 "source_sha256": "qwen-source", "processor_sha256": "qwen-processor",
                                 "configuration_sha256": {"config.json": "qwen-config"},
                                 "geometry_policy": EXPLICIT_GEOMETRY_POLICY,
                                 "extractor_source_sha256": file_sha(Path(__file__).with_name("migration_visual_probe.py")),
                                 "visa_pipeline_version": VISA_PIPELINE_VERSION}
        assert head_pipeline_version(compatibility) == PIPELINE_VERSION
    return {"passed": True, "batch_token_order_and_cls": True, "fp16_cache_preserved": True,
            "frozen_vision_only_load": True, "head_parameters": 697345, "finite_head_backward": True,
            "strict_compatibility": True, "preview_no_model_or_cuda": True, "qwen_explicit_geometry_compatibility": True}


def teacher_export_contract_self_test():
    """Use the actual exporter schema; reject plan, identity and map tampering."""
    import copy

    from migration_export_teacher import export_teacher, canonical_sha, atomic_json as write_json

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        train = [("/fixture/normal", "fixture", 0), ("/fixture/anomaly", "fixture", 1)]
        validation = [("/fixture/val", "fixture", 1)]
        items = [dict(path=path, category=category, label=label, split="train") for path, category, label in train]
        items.append(dict(path=validation[0][0], category="fixture", label=1, split="val"))
        metadata = {"pipeline_version": VISA_PIPELINE_VERSION, "decoder_sha256": "a" * 64,
                    "decoder": "/fixture/decoder.pt", "geometry": "square224", "teacher_seed": 42,
                    "training_protocol": "supervised_custom_60_20_20",
                    "manifest_sha256": hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()}
        maps = torch.stack([torch.full((224, 224), value, dtype=torch.float32) for value in (.1, .8, .4)])
        cache = {**metadata, "items": items, "cls": torch.ones(3, 1024), "maps": maps}
        cache_path = root / "teacher.pt"
        torch.save(cache, cache_path)
        write_json({**metadata, "cache_sha256": file_sha(cache_path)}, cache_path.with_suffix(".json"))
        batches = [[list(train[0]), list(train[1])]]
        plan_sha = canonical_sha(batches)
        plan_path = root / "training_plan.json"
        write_json({"seed": 42, "steps": 1, "batch_size": 2, "sha256": plan_sha, "batches": batches}, plan_path)
        exported = root / "export"
        manifest = export_teacher(cache_path, plan_path, exported, progress=False)
        path = exported / "manifest.json"
        required = {item[0] for item in train}

        def reader(**kwargs):
            return TeacherCache(path, train, validation, required, expected_plan_sha256=kwargs.get("plan", plan_sha),
                                expected_decoder_sha256=kwargs.get("decoder", "a" * 64))

        accepted = reader()
        assert accepted.audit["payload_verified"] and accepted.audit["planned_map_files_verified"] == 2
        torch.testing.assert_close(accepted.batch(train, "cpu")[:, 0], maps[:2], rtol=0, atol=0)
        for kwargs in ({"plan": "b" * 64}, {"decoder": "b" * 64}):
            try:
                reader(**kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError("Mismatched exported plan/expected decoder must fail")

        def publish(changed, resign=True):
            changed = copy.deepcopy(changed)
            if resign:
                changed.pop("payload_sha256", None)
                changed["payload_sha256"] = canonical_sha(changed)
            write_json(changed, path)

        bad_payload = copy.deepcopy(manifest)
        bad_payload["n_images"] = 999
        publish(bad_payload, resign=False)
        try:
            reader()
        except ValueError:
            pass
        else:
            raise AssertionError("Teacher payload tampering must fail")
        changed = copy.deepcopy(manifest)
        changed["teacher_cache"]["decoder_sha256"] = "b" * 64
        publish(changed)
        try:
            reader()
        except ValueError:
            pass
        else:
            raise AssertionError("Teacher checkpoint/source-cache decoder disagreement must fail")
        publish(manifest)
        filename = exported / manifest["maps"][train[0][0]]
        original_bytes = filename.read_bytes()
        np.save(filename, np.zeros((224, 224), dtype=np.float32))
        try:
            reader()
        except ValueError:
            pass
        else:
            raise AssertionError("Replaced teacher map must fail before GPU construction")
        filename.write_bytes(original_bytes)
        changed = copy.deepcopy(manifest)
        changed["map_files"][train[0][0]]["array_sha256"] = "b" * 64
        publish(changed)
        try:
            reader()
        except ValueError:
            pass
        else:
            raise AssertionError("Teacher map array SHA256 disagreement must fail")
    return {"passed": True, "real_exporter_manifest": True, "plan_sha_rejected": True,
            "decoder_identity_rejected": True, "payload_tampering_rejected": True,
            "planned_file_hashes_checked_before_gpu": True, "array_hash_tampering_rejected": True}


def cpu_self_test():
    class FakeExtractor:
        def __init__(self):
            self.calls = []

        @torch.inference_mode()
        def __call__(self, path, size):
            self.calls.append(path)
            value = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8) + int(path)
            return {"pre": value}, {"grid": [1, 2, 3]}

    extractor = FakeExtractor()
    cache = FeatureLRU(extractor, "pre", 448, "cpu", cache_gb=0)
    feature = cache([("1", "a", 1), ("1", "a", 1)])
    assert extractor.calls == ["1"] and not torch.is_inference(feature)
    head = SpatialHead(8, width=16)
    logits = head(feature)
    truth = torch.zeros_like(logits)
    truth[:, :, 70:100, 80:110] = 1
    teacher = torch.full_like(logits, .25)
    loss = supervised_loss(logits, truth) + .5 * soft_teacher_loss(logits, teacher)
    loss.backward()
    assert logits.shape == (2, 1, 224, 224)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    # Cache eviction must not discard a hit requested in the current batch.
    cache.limit = 2 * 2 * 3 * 8 * 2
    cache([("2", "a", 0), ("3", "a", 0)])
    pinned = cache([("2", "a", 0), ("4", "a", 0), ("2", "a", 0)])
    assert torch.equal(pinned[0], pinned[2]) and "2" not in cache.cache
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        np.save(temporary / "map.npy", np.full((224, 224), .3, dtype=np.float16))
        manifest = {"schema_version": "visa-teacher-prob-v1", "split": "train",
                    "coordinate_system": "direct-square-224", "visa_pipeline_version": VISA_PIPELINE_VERSION,
                    "teacher_checkpoint": "cpu-fixture", "maps": {"train": "map.npy"}}
        atomic_json(manifest, temporary / "manifest.json")
        teacher_cache = TeacherCache(temporary / "manifest.json", [("train", "a", 1)], [("val", "a", 1)], {"train"})
        assert teacher_cache.batch([("train", "a", 1)], "cpu").shape == (1, 1, 224, 224)
        try:
            teacher_cache.batch([("val", "a", 1)], "cpu")
        except ValueError:
            pass
        else:
            raise AssertionError("Validation must not read teacher cache")
        compatibility = {"fixture": True}
        atomic_save({"pipeline_version": PIPELINE_VERSION, "compatibility": compatibility,
                     "head": head.state_dict()}, temporary / "head.pt")
        reloaded = SpatialHead(8, width=16)
        load_head_checkpoint(reloaded, temporary / "head.pt", compatibility)
        assert all(torch.equal(x, y) for x, y in zip(head.state_dict().values(), reloaded.state_dict().values()))
    return {"passed": True, "inference_tensor_clone": True, "cache_batch_pinning": True,
            "finite_head_gradients": True, "teacher_validation_rejected": True, "strict_checkpoint_reload": True,
            "imagebind": imagebind_cpu_self_test(), "explicit_qwen_geometry": extractor_geometry_self_test(),
            "exported_teacher": teacher_export_contract_self_test()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--imagebind-repo", default=DEFAULT_IMAGEBIND_REPO)
    parser.add_argument("--imagebind-weights", help="Audited ImageBind-Huge .pth; defaults below --imagebind-repo")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--cpu-test-only", action="store_true")
    parser.add_argument("--features", choices=["pre", "post", "imagebind"], default="pre")
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--cache-gb", type=float, default=1.)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distill-weight", type=float, default=0.)
    parser.add_argument("--teacher-maps", help="Read-only train teacher manifest.json")
    parser.add_argument("--teacher-decoder-sha256", help="Optional external expected decoder identity; always verify export/source-cache decoder agreement")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--ckpt")
    parser.add_argument("--out", default="runs/qwen_dense_gt")
    args = parser.parse_args()
    if args.cpu_test_only:
        print(json.dumps(cpu_self_test(), indent=2))
        return
    if min(args.size, args.steps, args.bs, args.val_every) < 1 or args.lr <= 0 or args.cache_gb < 0 or args.distill_weight < 0:
        parser.error("Positive size/steps/batch/interval/LR and nonnegative cache/distillation weight required")
    if args.features == "imagebind" and args.size != 224:
        parser.error("--features imagebind requires --size 224 (fixed ImageBind input)")
    if args.distill_weight and not args.teacher_maps:
        parser.error("Positive distillation weight requires --teacher-maps")
    if args.teacher_decoder_sha256 is not None and not TeacherCache._is_sha(args.teacher_decoder_sha256):
        parser.error("--teacher-decoder-sha256 must be a SHA256 hex digest")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if not args.eval and any((output / name).exists() for name in ("best.pt", "final.pt")):
        raise FileExistsError("Choose a fresh output directory; existing weights are preserved")
    audit = (imagebind_source_audit(args.imagebind_repo, args.imagebind_weights)
             if args.features == "imagebind" else visual_source_audit(args.model))
    if not audit["passed"]:
        raise RuntimeError("ImageBind source/checkpoint differs from the audited implementation" if args.features == "imagebind"
                           else "Qwen/processor source differs from the audited implementation")
    from split_visa import splits
    train, validation, _ = splits()
    if {item[0] for item in train} & {item[0] for item in validation}:
        raise ValueError("Training and validation images overlap")
    plan = make_training_plan(train, args.steps, args.bs, args.seed)
    plan_sha = json_sha(plan)
    atomic_json({"seed": args.seed, "steps": args.steps, "batch_size": args.bs,
                 "sha256": plan_sha, "batches": plan}, output / ("eval_plan_preview.json" if args.eval else "training_plan.json"))
    head_config, compatibility = feature_configuration(args.features, args.size, audit)
    pipeline_version = head_pipeline_version(compatibility)
    matched = {"compatibility": compatibility, "seed": args.seed, "steps": args.steps,
               "batch_size": args.bs, "lr": args.lr, "plan_sha256": plan_sha,
               "val_every": args.val_every,
               "optimizer": "AdamW(weight_decay=1e-4)", "schedule": "cosine to lr*1e-3"}
    meta = {"pipeline_version": pipeline_version, "config": vars(args), "source_audit": audit,
            "compatibility": compatibility, "matched_run_key": json_sha(matched),
            "train_plan_sha256": plan_sha, "train_images": len(train), "validation_images": len(validation),
            "protocol": "supervised_custom_60_20_20", "selection": "validation macro pixel AP",
            "feature_source": {"pre": "final visual block input to main merger", "post": "main merger output",
                               "imagebind": "ImageBind visual block31 output before post_transformer_layer; CLS removed"}[args.features],
            "feature_fusion": "single final level; no multi-layer fusion", "test_set_used": False,
            "numerical_protocol": {"encoder_dtype": "torch.float16" if args.features == "imagebind" else "torch.bfloat16",
                                   "cache_dtype": "torch.float16" if args.features == "imagebind" else "torch.bfloat16",
                                   "head_dtype": "torch.float32",
                                   "cross_source_precision": "ImageBind historical FP16; Qwen native BF16"},
            "head_loss": "GT focal(gamma=2)+per-image smoothed Dice",
            "distillation": {"weight": args.distill_weight, "loss": "L1(sigmoid(logits),train_teacher_probability)",
                             "teacher_used_in_validation": False},
            "comparison_scope": ("same-feature GT vs KD share head capacity and training budget; cross-source SpatialHead structure is shared; "
                                 "ImageBind1280/pre1152/post4096 input widths give different parameter counts; ImageBind FP16/Qwen BF16"),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    teacher = None
    if args.distill_weight and not args.eval:
        required_paths = {item[0] for batch in plan for item in batch}
        teacher = TeacherCache(args.teacher_maps, train, validation, required_paths,
                               expected_plan_sha256=plan_sha,
                               expected_decoder_sha256=args.teacher_decoder_sha256)
        meta["distillation"]["manifest_sha256"] = hashlib.sha256(Path(args.teacher_maps).read_bytes()).hexdigest()
        meta["distillation"]["teacher_checkpoint"] = teacher.manifest["teacher_checkpoint"]
        meta["distillation"]["consumer_audit"] = teacher.audit
    metadata_path = output / ("eval_run.json" if args.eval else "run.json")
    atomic_json(meta, metadata_path)
    if not args.run:
        print(json.dumps({"source_and_plan_ready": True, "gpu_model_loaded": False,
                          "train_plan_sha256": plan_sha, "matched_run_key": meta["matched_run_key"], "out": str(output)}))
        return
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("--run requires a CUDA device")
    if args.features != "imagebind" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Qwen --run requires a CUDA device with BF16 support")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.features == "imagebind":
        visual, visual_meta = load_imagebind_tower(audit, args.device)
        extractor = ImageBindExtractor(visual, args.device)
    else:
        from transformers import AutoImageProcessor
        visual, visual_meta = load_visual_tower(args.model, args.device)
        processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True, use_fast=True)
        extractor = VisualExtractor(visual, processor, args.device)
    features = FeatureLRU(extractor, args.features, args.size, args.device, args.cache_gb)
    # Reset after visual construction so cached loading cannot affect head initialization.
    torch.manual_seed(args.seed)
    head = SpatialHead(head_config["in_channels"], width=head_config["width"]).float().to(args.device)
    meta["visual_model"] = visual_meta
    meta["head_parameters"] = sum(parameter.numel() for parameter in head.parameters())
    atomic_json(meta, metadata_path)
    try:
        if args.eval:
            checkpoint = load_head_checkpoint(head, args.ckpt or output / "best.pt", compatibility)
            result = evaluate(head, features, validation, args.bs, args.device)
            result.update({"pipeline_version_head": pipeline_version, "checkpoint_step": checkpoint["step"],
                           "checkpoint": str(args.ckpt or output / "best.pt"),
                           "trained_distill_weight": checkpoint["config"]["distill_weight"]})
            atomic_json(result, output / "metrics.json")
            print(json.dumps(result), flush=True)
            return
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 1e-3)
        best = -1.
        start = time.time()
        for step, items in enumerate(plan, 1):
            head.train()
            optimizer.zero_grad(set_to_none=True)
            logits = head(features(items))
            truth = targets(items, args.device)
            gt_loss = supervised_loss(logits, truth)
            kd_loss = soft_teacher_loss(logits, teacher.batch(items, args.device)) if teacher is not None else logits.new_zeros(())
            loss = gt_loss + args.distill_weight * kd_loss
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite spatial head loss")
            loss.backward()
            gradient = float(nn.utils.clip_grad_norm_(head.parameters(), 1.))
            if not np.isfinite(gradient):
                raise ValueError("Nonfinite spatial head gradient")
            optimizer.step()
            schedule.step()
            if step == 1 or step % 20 == 0:
                print(json.dumps({"step": step, "steps": args.steps, "loss": float(loss),
                                  "loss_gt": float(gt_loss), "loss_soft_l1": float(kd_loss), "gradient": gradient,
                                  "lr": schedule.get_last_lr()[0], "elapsed_seconds": time.time() - start}), flush=True)
            if step % args.val_every == 0 or step == args.steps:
                metrics = evaluate(head, features, validation, args.bs, args.device)
                if metrics["pixel_ap_macro"] is None:
                    raise ValueError("Validation AP is undefined; no valid anomaly categories")
                record = {"step": step, **metrics}
                with (output / "validation.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                print("[validation] " + json.dumps(record), flush=True)
                checkpoint = {"pipeline_version": pipeline_version, "compatibility": compatibility,
                              "head": {name: value.detach().cpu().clone()
                                       for name, value in head.state_dict().items()},
                              "step": step, "seed": args.seed,
                              "matched_run_key": meta["matched_run_key"], "validation": metrics,
                              "config": vars(args), "geometry": features.geometry}
                if metrics["pixel_ap_macro"] > best:
                    best = metrics["pixel_ap_macro"]
                    atomic_save(checkpoint, output / "best.pt")
                if step == args.steps:
                    atomic_save(checkpoint, output / "final.pt")
        # Exercise the strict compatibility loader on the selected on-disk head,
        # then repeat the complete validation forward under its reloaded state.
        selected = load_head_checkpoint(head, output / "best.pt", compatibility)
        reloaded = evaluate(head, features, validation, args.bs, args.device)
        errors = {}
        for key in ("pixel_ap_macro", "pixel_auroc_macro", "pixel_auroc_weighted", "image_auroc_from_map99"):
            expected, actual = selected["validation"][key], reloaded[key]
            if expected is None or actual is None:
                if expected != actual:
                    raise AssertionError("Best head reload changed undefined metric: " + key)
                errors[key] = 0.
            else:
                errors[key] = abs(float(actual) - float(expected))
                if not np.isfinite(errors[key]) or errors[key] > 1e-10:
                    raise AssertionError("Best head reload changed %s by %.12g" % (key, errors[key]))
        reload_max_error = max(errors.values())
        atomic_json({"passed": True, "selected_best_step": selected["step"], "split": "val",
                     "checkpoint": str(output / "best.pt"), "tolerance": 1e-10,
                     "metric_abs_errors": errors, "reload_max_abs_metric_error": reload_max_error,
                     "validation_original": selected["validation"], "validation_reloaded": reloaded},
                    output / "best_validation_reloaded.json")
        atomic_json({"steps": args.steps, "best_validation_pixel_ap": best, "test_set_used": False,
                     "selected_best_step": selected["step"], "reload_max_abs_metric_error": reload_max_error,
                     "matched_run_key": meta["matched_run_key"], "elapsed_seconds": time.time() - start,
                     "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(args.device)}, output / "train_complete.json")
    except Exception as error:
        atomic_json({"type": type(error).__name__, "message": str(error)}, output / "failure.json")
        raise
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
