"""Run deterministic, resumable evaluation with the training input pipeline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import SCHEMA_VERSION, compute_metrics, parse_prediction, validate_boxes


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_data(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records, seen = [], set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("id", "image", "category", "is_anomalous", "prompt", "answer"):
                if key not in row:
                    raise ValueError(f"{path}:{line_number}: missing {key}")
            identifier = str(row["id"])
            if identifier in seen:
                raise ValueError(f"duplicate evaluation id: {identifier}")
            seen.add(identifier)
            if type(row["is_anomalous"]) is not bool:
                raise ValueError(f"{identifier}: is_anomalous must be a boolean")
            if not isinstance(row["prompt"], str) or not row["prompt"].strip():
                raise ValueError(f"{identifier}: prompt must be a nonempty string")
            if not isinstance(row["answer"], str):
                raise ValueError(f"{identifier}: answer must be a JSON string")
            answer, error = parse_prediction(row["answer"], require_boxes=False)
            if error:
                raise ValueError(f"{identifier}: invalid ground-truth answer: {error}")
            if answer["is_anomalous"] != row["is_anomalous"]:
                raise ValueError(f"{identifier}: answer/label disagree")
            row["localization_eligible"] = "boxes" in answer
            row["gt_boxes"] = validate_boxes(row.get("boxes", answer.get("boxes", [])))
            if row["localization_eligible"] and row["gt_boxes"] != answer["boxes"]:
                raise ValueError(f"{identifier}: top-level/answer boxes disagree")
            if row["localization_eligible"] and row["is_anomalous"] and not row["gt_boxes"]:
                raise ValueError(f"{identifier}: anomalous region answer has no boxes")
            if not row["is_anomalous"] and row["gt_boxes"]:
                raise ValueError(f"{identifier}: normal ground truth has nonempty boxes")
            fraction = row.get("anomaly_area_fraction")
            if fraction is not None and (type(fraction) not in (int, float) or not 0 <= fraction <= 1):
                raise ValueError(f"{identifier}: anomaly_area_fraction must be in [0,1]")
            row["id"] = identifier
            row["source_id"] = str(row.get("source_id") or row["image"])
            records.append(row)
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError("evaluation data is empty")
    return records


def _paths(output: Path) -> tuple[Path, Path, Path]:
    if output.suffix == ".jsonl":
        output.parent.mkdir(parents=True, exist_ok=True)
        return output, output.with_suffix(".metrics.json"), output.with_suffix(".meta.json")
    output.mkdir(parents=True, exist_ok=True)
    return output / "predictions.jsonl", output / "metrics.json", output / "meta.json"


def _read_predictions(path: Path, expected_ids: set[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records, seen = [], set()
    with path.open("r+b") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                # A crash can leave only the last line incomplete; never repair
                # a bad record followed by further data.
                if handle.read().strip():
                    raise ValueError("corrupt prediction JSONL before final line")
                handle.truncate(offset)
                break
            identifier = str(record.get("id"))
            if identifier not in expected_ids or identifier in seen:
                raise ValueError(f"unexpected or duplicate resumed id: {identifier}")
            seen.add(identifier)
            records.append(record)
            if not line.endswith(b"\n"):
                # A complete final object may have been flushed before its
                # newline; appending must still produce valid JSONL.
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
    return records


def _package_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="run directory, or a .jsonl file")
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name")
    parser.add_argument("--branch", default="unspecified", help="e.g. dataquality_sampling_ablation or region_supervision_ablation")
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if min(args.max_pixels, args.max_new_tokens, args.max_length) <= 0:
        parser.error("pixel/token limits must be positive")
    if args.bootstrap_samples != 0 and args.bootstrap_samples < 20:
        parser.error("--bootstrap-samples must be 0 or at least 20")
    data = _load_data(args.data, args.limit)
    predictions_path, metrics_path, meta_path = _paths(args.output)
    config = {
        "model": args.model, "adapter": args.adapter, "data": str(args.data.resolve()),
        "data_sha256": _sha256(args.data), "image_root": str(args.image_root.resolve()),
        "max_pixels": args.max_pixels, "max_new_tokens": args.max_new_tokens,
        "max_length": args.max_length, "limit": args.limit,
        "generation": {"do_sample": False, "num_beams": 1},
        "quantization": "shared modeling.load_model NF4 policy for base and adapter",
        "branch": args.branch, "run_name": args.run_name or args.output.stem,
        "n_expected": len(data),
        "adapter_files_sha256": {
            path.name: _sha256(path)
            for path in sorted(Path(args.adapter).glob("*"))
            if args.adapter and path.is_file() and path.suffix in {".safetensors", ".bin", ".json"}
        } if args.adapter else None,
        "base_config_sha256": _sha256(Path(args.model) / "config.json"),
        "n_region_task": sum(row["localization_eligible"] for row in data),
        "n_decision_task": sum(not row["localization_eligible"] for row in data),
        "subset_sha256": hashlib.sha256(json.dumps([(r["id"], r["prompt"], r["localization_eligible"]) for r in data], ensure_ascii=False).encode()).hexdigest(),
    }
    if args.resume:
        if not meta_path.exists():
            raise ValueError("--resume requires existing metadata")
        old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if old_meta.get("config") != config:
            raise ValueError("resume config/data differs; use a new output directory")
        predictions = _read_predictions(predictions_path, {r["id"] for r in data})
        meta = old_meta
    else:
        if any(path.exists() for path in (predictions_path, metrics_path, meta_path)):
            raise FileExistsError("output exists; use --resume or another output directory")
        predictions = []
        meta = {"schema_version": SCHEMA_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
                "config": config, "versions": _package_versions(), "status": "running"}
        _atomic_json(meta_path, meta)
    completed = {r["id"] for r in predictions}
    pending = [row for row in data if row["id"] not in completed]
    try:
        if pending:
            import torch
            from .modeling import input_device, load_model, make_inputs
            model, processor = load_model(args.model, adapter=args.adapter, training=False, max_pixels=args.max_pixels)
            model.eval()
            device = input_device(model)
            if device.type != "cuda":
                raise RuntimeError("CUDA evaluation required for the shared NF4 comparison")
            meta["hardware"] = {"gpu": torch.cuda.get_device_name(device), "device": str(device)}
            meta["model_loading_audit"] = getattr(model, "_industrial_vlm_audit", {})
            _atomic_json(meta_path, meta)
            warmed_up = False
            with predictions_path.open("a", encoding="utf-8") as output:
                for row in pending:
                    image_path = Path(row["image"])
                    if not image_path.is_absolute():
                        image_path = args.image_root / image_path
                    if not image_path.is_file():
                        raise FileNotFoundError(image_path)
                    inputs = make_inputs(processor, image_path, row["prompt"], answer=None, max_length=args.max_length)
                    if "labels" in inputs:
                        raise ValueError("inference inputs unexpectedly contain training labels")
                    if inputs["input_ids"].shape[-1] + args.max_new_tokens > args.max_length:
                        raise ValueError(f"{row['id']}: input + generation budget exceeds --max-length; use a common larger budget")
                    # Warm up once outside latency/memory measurements.
                    if not warmed_up:
                        warm = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
                        with torch.inference_mode():
                            warm_output = model.generate(**warm, do_sample=False, num_beams=1, max_new_tokens=1, use_cache=True)
                        torch.cuda.synchronize(device)
                        del warm_output, warm
                        warmed_up = True
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
                    with torch.inference_mode():
                        generated = model.generate(**inputs, do_sample=False, num_beams=1,
                                                   max_new_tokens=args.max_new_tokens, use_cache=True)
                    torch.cuda.synchronize(device)
                    response = generated[:, inputs["input_ids"].shape[-1]:]
                    raw_text = processor.batch_decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    latency = time.perf_counter() - start
                    peak = torch.cuda.max_memory_allocated(device)
                    parsed, parse_error = parse_prediction(raw_text, row["localization_eligible"])
                    record = {"schema_version": SCHEMA_VERSION, "id": row["id"], "source_id": row["source_id"],
                              "image": row["image"], "category": row["category"], "label": row["is_anomalous"],
                              "gt_boxes": row["gt_boxes"], "localization_eligible": row["localization_eligible"],
                              "anomaly_area_fraction": row.get("anomaly_area_fraction"),
                              "prompt": row["prompt"], "raw_text": raw_text, "parsed": parsed, "parse_error": parse_error,
                              "latency_seconds": latency, "peak_memory_allocated_bytes": peak,
                              "input_tokens": int(inputs["input_ids"].shape[-1]), "generated_tokens": int(response.shape[-1])}
                    output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    output.flush()
                    predictions.append(record)
                    del generated, response, inputs
                    if len(predictions) % 10 == 0:
                        print(f"Evaluated {len(predictions)}/{len(data)}", flush=True)
        metrics = compute_metrics(predictions, bootstrap_samples=args.bootstrap_samples)
        metrics["metadata"] = {**config, "complete": len(predictions) == len(data)}
        _atomic_json(metrics_path, metrics)
        meta.update(status="complete", n_completed=len(predictions), finished_at=datetime.now(timezone.utc).isoformat())
        _atomic_json(meta_path, meta)
        print(json.dumps({"predictions": str(predictions_path), "metrics": str(metrics_path),
                          "overall": metrics["overall"]}, ensure_ascii=False), flush=True)
    except Exception as exc:
        meta.update(status="failed", n_completed=len(predictions), error=f"{type(exc).__name__}: {exc}")
        _atomic_json(meta_path, meta)
        raise


if __name__ == "__main__":
    main()
