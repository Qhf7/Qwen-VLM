"""Export planned training-only teacher probabilities to the dense-head schema.

CPU only; no model is imported or instantiated. The shared cache and sidecar
SHA, pipeline, geometry, item-manifest SHA, and dense training-plan SHA are
verified before export. An existing final manifest is never overwritten.
Interrupted exports resume only with identical inputs; each existing .npy
must exactly match the expected serialized float32 source map SHA256.

  python migration_export_teacher.py \
      --cache runs/migration_20261103/teacher_cache.pt \
      --plan runs/qwen_dense_gt/training_plan.json --out runs/teacher_train

The resulting manifest.json is accepted directly by migration_dense_train's
TeacherCache. All maps share the direct-square-224 coordinate system.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from migration_roi_eval import load_shared_cache, sha256_file
from visa_pipeline import PIPELINE_VERSION as VISA_PIPELINE_VERSION


EXPORT_VERSION = "planned-train-teacher-export-v1"


def canonical_sha(value):
    # Matches migration_dense_train.json_sha exactly.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def final_json_exclusive(value, path):
    """Publish a complete manifest atomically, without replacing any prior one."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".publish.tmp")
    atomic_json(value, temporary)
    try:
        # Same-directory hard link is atomic and fails if the destination exists.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def export_lock(directory):
    """A crashed process releases flock automatically, so recovery needs no override."""
    with (directory / ".export.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another teacher export is active in " + str(directory)) from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def validate_plan(path):
    plan = json.loads(Path(path).read_text())
    batches = plan.get("batches")
    if not isinstance(batches, list) or not batches:
        raise ValueError("Training plan needs nonempty batches")
    if type(plan.get("steps")) is not int or plan["steps"] != len(batches):
        raise ValueError("Training plan steps do not match its batches")
    if type(plan.get("batch_size")) is not int or plan["batch_size"] < 1:
        raise ValueError("Training plan needs a positive batch_size")
    if type(plan.get("seed")) is not int:
        raise ValueError("Training plan requires an integer seed")
    if plan.get("sha256") != canonical_sha(batches):
        raise ValueError("Training plan batches SHA256 mismatch")
    images = {}
    for batch in batches:
        if not isinstance(batch, list) or len(batch) != plan["batch_size"]:
            raise ValueError("Training plan batch size is inconsistent")
        for item in batch:
            if (not isinstance(item, list) or len(item) != 3
                    or not isinstance(item[0], str) or not item[0] or not Path(item[0]).is_absolute()
                    or not isinstance(item[1], str) or not item[1] or type(item[2]) is not int
                    or item[2] not in (0, 1)):
                raise ValueError("Training plan items must be [absolute_path,category,binary_label]")
            image = {"path": item[0], "category": item[1], "label": item[2]}
            if item[0] in images and images[item[0]] != image:
                raise ValueError("Training plan gives conflicting metadata for " + item[0])
            images[item[0]] = image
    return plan, images


def map_array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.array(value, dtype=np.float32, copy=True, order="C")
    if array.shape != (224, 224) or not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError("Expected finite teacher probability [224,224] in [0,1]")
    return array


def map_bytes(array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def export_teacher(cache_path, plan_path, output_directory, progress=True):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Existing teacher manifest is preserved: " + str(manifest_path))
    with export_lock(output):
        if manifest_path.exists():
            raise FileExistsError("Existing teacher manifest is preserved: " + str(manifest_path))
        plan, planned = validate_plan(plan_path)
        cache, audit = load_shared_cache(cache_path)
        by_path = {item["path"]: (index, item) for index, item in enumerate(cache["items"])}
        for path, expected in planned.items():
            if path not in by_path:
                raise FileNotFoundError("Planned training image is absent from shared cache: " + path)
            _, item = by_path[path]
            if item["split"] != "train":
                raise ValueError("Plan requests a non-training teacher map: " + path)
            if item["category"] != expected["category"] or item["label"] != expected["label"]:
                raise ValueError("Training plan metadata disagrees with teacher cache: " + path)
        if cache["pipeline_version"] != VISA_PIPELINE_VERSION:
            raise ValueError("Teacher and dense-head mask pipelines disagree")
        job = {"export_version": EXPORT_VERSION,
               "teacher_cache_sha256": audit["cache_sha256"],
               "teacher_cache_manifest_sha256": audit["manifest_sha256"],
               "teacher_decoder_sha256": cache["decoder_sha256"],
               "training_plan_sha256": plan["sha256"],
               "training_plan_file_sha256": sha256_file(plan_path),
               "visa_pipeline_version": VISA_PIPELINE_VERSION,
               "coordinate_system": "direct-square-224", "dtype": "float32",
               "planned_unique_images": len(planned)}
        job_sha = canonical_sha(job)
        state_path = output / "export_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get("job_sha256") != job_sha or state.get("job") != job:
                raise ValueError("Interrupted export belongs to different cache/plan/pipeline inputs")
            if set(state.get("files", {})) - set(planned):
                raise ValueError("Interrupted export contains paths outside this training plan")
        else:
            state = {"job_sha256": job_sha, "job": job, "files": {}, "complete": False}
        maps_directory = output / "maps"
        maps_directory.mkdir(exist_ok=True)
        names = {path: "maps/" + hashlib.sha256(path.encode()).hexdigest() + ".npy" for path in planned}
        expected_names = {Path(name).name for name in names.values()}
        unexpected = {path.name for path in maps_directory.glob("*.npy")} - expected_names
        if unexpected:
            raise ValueError("Output contains .npy files outside the requested training plan")
        atomic_json(state, state_path)
        started = time.monotonic()
        reused = written = 0
        records = {}
        for ordinal, path in enumerate(sorted(planned), 1):
            index, item = by_path[path]
            array = map_array(cache["maps"][index])
            serialized = map_bytes(array)
            expected_sha = hashlib.sha256(serialized).hexdigest()
            destination = output / names[path]
            if destination.exists():
                if sha256_file(destination) != expected_sha:
                    raise ValueError("Existing probability file does not match source map SHA256: " + str(destination))
                reused += 1
            else:
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                with temporary.open("wb") as stream:
                    stream.write(serialized)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(destination)
                if sha256_file(destination) != expected_sha:
                    raise IOError("Written probability file failed SHA256 verification: " + str(destination))
                written += 1
            records[path] = {"file": names[path], "sha256": expected_sha,
                             "array_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                             "shape": [224, 224], "dtype": "float32", "bytes": len(serialized),
                             "source_cache_index": index, "category": item["category"],
                             "label": int(item["label"]), "split": "train"}
            state["files"][path] = records[path]
            if ordinal % 25 == 0 or ordinal == len(planned):
                atomic_json(state, state_path)
                if progress:
                    print(json.dumps({"exported": ordinal, "total": len(planned), "written": written,
                                      "reused_verified": reused, "elapsed_seconds": time.monotonic() - started}), flush=True)
        manifest = {
            "schema_version": "visa-teacher-prob-v1", "split": "train",
            "coordinate_system": "direct-square-224", "visa_pipeline_version": VISA_PIPELINE_VERSION,
            "export_version": EXPORT_VERSION, "job_sha256": job_sha,
            "teacher_checkpoint": {"path": cache.get("decoder"), "decoder_sha256": cache["decoder_sha256"],
                                   "teacher_seed": cache.get("teacher_seed"),
                                   "training_protocol": cache.get("training_protocol")},
            "teacher_cache": {"path": str(Path(cache_path).resolve()), **audit,
                              "sidecar_sha256": sha256_file(Path(cache_path).with_suffix(".json"))},
            "training_plan": {"path": str(Path(plan_path).resolve()), "file_sha256": job["training_plan_file_sha256"],
                              "batches_sha256": plan["sha256"], "seed": plan["seed"], "steps": plan["steps"],
                              "batch_size": plan["batch_size"], "sampled_positions": plan["steps"] * plan["batch_size"]},
            "maps": names, "map_files": records, "n_images": len(records),
            "n_normal": sum(record["label"] == 0 for record in records.values()),
            "n_anomaly": sum(record["label"] == 1 for record in records.values()),
            "test_or_validation_maps_exported": False,
            "export_statistics": {"written": written, "reused_verified": reused,
                                  "elapsed_seconds": time.monotonic() - started},
        }
        manifest["payload_sha256"] = canonical_sha(manifest)
        final_json_exclusive(manifest, manifest_path)
        state["complete"] = True
        state["manifest_sha256"] = sha256_file(manifest_path)
        atomic_json(state, state_path)
        return manifest


def cpu_self_test():
    import torch
    from migration_dense_train import TeacherCache

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        items = [{"path": "/fixture/normal", "category": "a", "label": 0, "split": "train"},
                 {"path": "/fixture/anomaly", "category": "a", "label": 1, "split": "train"},
                 {"path": "/fixture/validation", "category": "a", "label": 1, "split": "val"},
                 {"path": "/fixture/test", "category": "a", "label": 0, "split": "test"}]
        maps = torch.stack([torch.full((224, 224), value, dtype=torch.float32) for value in (.1, .8, .4, .2)])
        metadata = {"pipeline_version": VISA_PIPELINE_VERSION, "decoder_sha256": "a" * 64,
                    "decoder": "/fixture/teacher.pt", "geometry": "square224", "teacher_seed": 42,
                    "training_protocol": "supervised_custom_60_20_20",
                    "manifest_sha256": hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()}
        cache = {**metadata, "items": items, "cls": torch.ones(4, 1024), "maps": maps}
        cache_path = root / "teacher_cache.pt"
        torch.save(cache, cache_path)
        atomic_json({**metadata, "cache_sha256": sha256_file(cache_path)}, cache_path.with_suffix(".json"))
        batches = [[["/fixture/normal", "a", 0], ["/fixture/anomaly", "a", 1]],
                   [["/fixture/anomaly", "a", 1], ["/fixture/normal", "a", 0]]]
        plan = {"seed": 42, "steps": 2, "batch_size": 2, "sha256": canonical_sha(batches), "batches": batches}
        plan_path = root / "training_plan.json"
        atomic_json(plan, plan_path)
        output = root / "export"
        (output / "maps").mkdir(parents=True)
        # Simulate a crash after saving one map, before writing progress metadata.
        first_name = hashlib.sha256(b"/fixture/normal").hexdigest() + ".npy"
        np.save(output / "maps" / first_name, maps[0].numpy(), allow_pickle=False)
        manifest = export_teacher(cache_path, plan_path, output, progress=False)
        assert manifest["n_images"] == 2 and manifest["export_statistics"]["reused_verified"] == 1
        train = [(item["path"], item["category"], item["label"]) for item in items if item["split"] == "train"]
        validation = [("/fixture/validation", "a", 1)]
        reader = TeacherCache(output / "manifest.json", train, validation, {item[0] for item in train})
        read = reader.batch(train, "cpu")
        torch.testing.assert_close(read[:, 0], maps[:2], rtol=0, atol=0)
        try:
            export_teacher(cache_path, plan_path, output, progress=False)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing manifest must never be overwritten")
        bad_plan = dict(plan, sha256="0" * 64)
        atomic_json(bad_plan, root / "bad_plan.json")
        try:
            export_teacher(cache_path, root / "bad_plan.json", root / "bad_plan", progress=False)
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupt plan SHA must fail")
        val_batches = [[["/fixture/validation", "a", 1]]]
        atomic_json({"seed": 42, "steps": 1, "batch_size": 1,
                     "batches": val_batches, "sha256": canonical_sha(val_batches)}, root / "val_plan.json")
        try:
            export_teacher(cache_path, root / "val_plan.json", root / "bad_val", progress=False)
        except ValueError:
            pass
        else:
            raise AssertionError("A validation path must never be exported")
        # Resume must verify map bytes against source, rather than trusting state.
        (output / "manifest.json").unlink()
        np.save(output / manifest["maps"]["/fixture/normal"], np.zeros((224, 224), np.float32))
        try:
            export_teacher(cache_path, plan_path, output, progress=False)
        except ValueError:
            pass
        else:
            raise AssertionError("Tampered resumed map must fail")
    return {"passed": True, "resume_verified": True, "dense_reader_compatible": True,
            "existing_manifest_preserved": True, "bad_plan_sha_rejected": True,
            "validation_path_rejected": True, "tampered_map_rejected": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="runs/migration_20261103/teacher_cache.pt")
    parser.add_argument("--plan")
    parser.add_argument("--out")
    parser.add_argument("--cpu-test-only", action="store_true")
    args = parser.parse_args()
    if args.cpu_test_only:
        print(json.dumps(cpu_self_test(), indent=2))
        return
    if not args.plan or not args.out:
        parser.error("--plan and --out are required for export")
    manifest = export_teacher(args.cache, args.plan, args.out)
    print(json.dumps({"manifest": str(Path(args.out) / "manifest.json"), "n_images": manifest["n_images"],
                      "decoder_sha256": manifest["teacher_checkpoint"]["decoder_sha256"],
                      "plan_sha256": manifest["training_plan"]["batches_sha256"], "gpu_model_loaded": False}))


if __name__ == "__main__":
    main()
