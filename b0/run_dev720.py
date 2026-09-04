"""Run a fixed Dev720 manifest with a specified B0 prompt and binary token score."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from b0.b0plus_protocol import BATCH_SINGLE_ATOL, PROMPTS, PROTOCOL_VERSION
except ModuleNotFoundError:
    from b0plus_protocol import BATCH_SINGLE_ATOL, PROMPTS, PROTOCOL_VERSION


SIDE = 448
EXPECTED_CLASSES = 18
EXPECTED_GROUPS = 720
EXPECTED_VIEWS = 3600


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_ids(tokenizer, word: str) -> list[int]:
    ids = {
        int(encoded[0])
        for text in (word, " " + word)
        if len(encoded := tokenizer.encode(text, add_special_tokens=False)) == 1
    }
    if not ids:
        raise ValueError(f"{word!r} 不是单 token")
    return sorted(ids)


def binary_scores(logits: torch.Tensor, yes_ids: list[int], no_ids: list[int]) -> dict[str, float]:
    values = logits.float()
    yes_logit = torch.logsumexp(values[yes_ids], dim=0)
    no_logit = torch.logsumexp(values[no_ids], dim=0)
    probabilities = torch.softmax(torch.stack([no_logit, yes_logit]), dim=0)
    return {
        "p_yes": float(probabilities[1]),
        "p_no": float(probabilities[0]),
        "yes_logit": float(yes_logit),
        "no_logit": float(no_logit),
        "logit_margin_no_minus_yes": float(no_logit - yes_logit),
    }


def load_direct(path: Path) -> Image.Image:
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"))
    resized = cv2.resize(rgb, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized)


def conversations(images: list[Image.Image], prompt: str) -> list[list[dict]]:
    return [
        [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        for image in images
    ]


def encode(processor, images: list[Image.Image], prompt: str, device: torch.device) -> dict:
    inputs = processor.apply_chat_template(
        conversations(images, prompt),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"text_kwargs": {"padding": True}},
    )
    return {key: value.to(device) for key, value in inputs.items()}


def forward_logits(model, inputs: dict) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(**inputs, use_cache=False, logits_to_keep=1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    logits = output.logits[:, -1, :].float().cpu()
    del output
    return logits, elapsed


def generate_logits(model, inputs: dict) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    logits = output.scores[0].float().cpu()
    del output
    return logits, elapsed


def generation_audit(
    model, processor, subset: pd.DataFrame, root: Path, device: torch.device,
    prompt: str, out_dir: Path, count: int,
) -> None:
    """Record a balanced set of complete responses without using them for scores."""
    if count <= 0:
        return
    audit_path = out_dir / "generation_audit.csv"
    if audit_path.is_file() and len(pd.read_csv(audit_path)) >= count:
        print(f"[generation audit] reuse {audit_path}", flush=True)
        return
    per_stratum = max(1, count // (subset.class_name.nunique() * 2))
    selected = (
        subset.groupby(["class_name", "sample_label"], group_keys=False, sort=True)
        .head(per_stratum)
        .head(count)
    )
    records = []
    for index, row in enumerate(selected.itertuples(index=False), 1):
        image = load_direct(root / row.image_path)
        inputs = encode(processor, [image], prompt, device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True)
        continuation = generated[:, inputs["input_ids"].shape[1]:]
        response = processor.batch_decode(continuation, skip_special_tokens=True)[0].strip()
        records.append({
            "class_name": row.class_name, "group_path": row.group_path,
            "image_path": row.image_path, "sample_label": row.sample_label,
            "response": response, "first_word": response.split(maxsplit=1)[0] if response else "",
        })
        if index % 12 == 0 or index == len(selected):
            print(f"[generation audit] {index}/{len(selected)}", flush=True)
    pd.DataFrame(records).to_csv(audit_path, index=False)


def validate_cuda(
    model, processor, group: pd.DataFrame, root: Path, device: torch.device,
    yes_ids: list[int], no_ids: list[int], prompt: str, anomaly_token: str,
    out_dir: Path, forward_generate_atol: float,
) -> str:
    path = out_dir / "cuda_validation.csv"
    if path.is_file():
        prior = pd.read_csv(path)
        if (
            len(prior) == 5
            and prior.abs_batch_single.max() <= BATCH_SINGLE_ATOL
            and prior.abs_forward_generate.max() <= forward_generate_atol
        ):
            print(f"[validation] reuse {path}", flush=True)
            return "batch5"
    score_key = "p_yes" if anomaly_token == "yes" else "p_no"
    images = [load_direct(root / image_path) for image_path in group.image_path]
    singles = []
    records = []
    for index, ((_, row), image) in enumerate(zip(group.iterrows(), images)):
        inputs = encode(processor, [image], prompt, device)
        logits, forward_s = forward_logits(model, inputs)
        score = binary_scores(logits[0], yes_ids, no_ids)[score_key]
        singles.append(score)
        record = {
            "image_path": row.image_path,
            "p_single": score,
            "single_forward_s": forward_s,
        }
        if index == 0:
            generated, generate_s = generate_logits(model, inputs)
            generated_score = binary_scores(generated[0], yes_ids, no_ids)[score_key]
            record.update({
                "p_generate": generated_score,
                "generate_one_s": generate_s,
                "abs_forward_generate": abs(score - generated_score),
            })
        records.append(record)
    batch_inputs = encode(processor, images, prompt, device)
    batch_logits, batch_s = forward_logits(model, batch_inputs)
    batch_scores = [binary_scores(value, yes_ids, no_ids)[score_key] for value in batch_logits]
    for record, single, batch in zip(records, singles, batch_scores):
        record["p_batch5"] = batch
        record["abs_batch_single"] = abs(batch - single)
        record["batch5_forward_s"] = batch_s
    frame = pd.DataFrame(records)
    frame["abs_forward_generate"] = frame.abs_forward_generate.fillna(0.0)
    frame.to_csv(path, index=False)
    max_fg = float(frame.abs_forward_generate.max())
    max_bs = float(frame.abs_batch_single.max())
    print(f"[validation] forward-generate={max_fg:.3g}; batch-single={max_bs:.3g}", flush=True)
    if max_fg > forward_generate_atol:
        raise RuntimeError("forward-only 与 generate 首 token 未通过冻结容差")

    execution_mode = "batch5" if max_bs <= BATCH_SINGLE_ATOL else "serial"
    decision = {
        "execution_mode": execution_mode,
        "forward_generate_max_abs": max_fg,
        "forward_generate_atol": forward_generate_atol,
        "batch_single_max_abs": max_bs,
        "batch_single_atol": BATCH_SINGLE_ATOL,
        "reason": (
            "batch5 passed the preregistered equivalence gate"
            if execution_mode == "batch5"
            else "batch5 failed the preregistered equivalence gate; use single-view B0f "
                 "for every formal score without relaxing the tolerance"
        ),
    }
    (out_dir / "execution_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if execution_mode == "serial":
        print(
            "[validation] batch5 未通过冻结容差；正式分数自动切换为逐视角 B0f",
            flush=True,
        )
    return execution_mode


def write_protocol(
    args, subset: pd.DataFrame, yes_ids: list[int], no_ids: list[int], prompt: str, out_dir: Path,
) -> None:
    holdout = {
        line.strip()
        for line in args.holdout_classes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    classes = sorted(subset.class_name.unique())
    overlap = sorted(set(classes) & holdout)
    if overlap:
        raise RuntimeError(f"Dev-mini 混入 holdout 类: {overlap}")
    protocol = {
        "parent_protocol": PROTOCOL_VERSION,
        "protocol": (
            "B0plus-dev720-frozen-v1" if args.prompt_id == "P0"
            else "B0plus-dev720-prompt-diagnostic-v1"
        ),
        "model": str(args.model_id),
        "prompt_id": args.prompt_id,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "anomaly_token": args.anomaly_token,
        "preprocess": "direct 448x448 cv2.INTER_AREA",
        "input_root": str(args.root),
        "preprocessed_cache_manifest": (
            str(args.preprocessed_cache_manifest) if args.preprocessed_cache_manifest else None
        ),
        "preprocessed_cache_manifest_sha256": (
            sha256_file(args.preprocessed_cache_manifest)
            if args.preprocessed_cache_manifest else None
        ),
        "forward": {"use_cache": False, "logits_to_keep": 1, "dtype": "float16"},
        "primary_aggregation": "max",
        "offline_aggregations": ["max", "top2_mean", "mean"],
        "subset_path": str(args.subset),
        "subset_sha256": sha256_file(args.subset),
        "holdout_classes_path": str(args.holdout_classes),
        "holdout_classes_sha256": sha256_file(args.holdout_classes),
        "holdout_intersection": overlap,
        "classes": classes,
        "counts": {
            "classes": int(subset.class_name.nunique()),
            "groups": int(subset.group_path.nunique()),
            "views": len(subset),
        },
        "yes_token_ids": yes_ids,
        "no_token_ids": no_ids,
        "batch_single_atol": BATCH_SINGLE_ATOL,
        "bootstrap": {
            "repeats": 2000,
            "seed": 20260901,
            "sampling": "stratified within class_name x sample_label",
        },
        "resolution_strata": ["small: median long side <=640", "medium: 641-1280", "large: >1280"],
        "defect_area_strata": ["small: max ratio <=1%", "medium: 1-3%", "large: >3%"],
    }
    (out_dir / "frozen_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--subset", type=Path, default=Path("artifacts/splits/fold1/dev.parquet"))
    parser.add_argument("--holdout-classes", type=Path, default=Path("artifacts/splits/fold1/holdout_classes.txt"))
    parser.add_argument("--root", type=Path, default=Path("Dataset"))
    parser.add_argument("--preprocessed-cache-manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/b0plus_dev720"))
    parser.add_argument("--prompt-id", choices=sorted(PROMPTS), default="P0")
    parser.add_argument("--anomaly-token", choices=("yes", "no"), default="yes")
    parser.add_argument("--generation-audit-count", type=int, default=0)
    parser.add_argument("--generation-audit-only", action="store_true")
    parser.add_argument("--forward-generate-atol", type=float, default=1e-5)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompt = PROMPTS[args.prompt_id]

    subset = pd.read_parquet(args.subset).sort_values(
        ["class_name", "condition", "group_path", "view_idx"], kind="mergesort"
    )
    if (
        len(subset) != EXPECTED_VIEWS
        or subset.group_path.nunique() != EXPECTED_GROUPS
        or subset.class_name.nunique() != EXPECTED_CLASSES
        or not subset.groupby("group_path").size().eq(5).all()
    ):
        raise RuntimeError("Dev-mini 不是冻结的 18类/720组/3600视角/每组5视角")
    label_counts = subset.groupby(["class_name", "sample_label"]).group_path.nunique()
    if not label_counts.eq(20).all():
        raise RuntimeError("Dev-mini 不是每类 20 OK + 20 NG")
    missing = [path for path in subset.image_path if not (args.root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少 {len(missing)} 张图，首个: {missing[0]}")
    if args.preprocessed_cache_manifest:
        cache = pd.read_csv(args.preprocessed_cache_manifest)
        if len(cache) != EXPECTED_VIEWS or set(cache.image_path) != set(subset.image_path):
            raise RuntimeError("预处理缓存清单与冻结 Dev-mini 不一致")
        for index, row in enumerate(cache.itertuples(index=False), 1):
            with Image.open(args.root / row.image_path) as source:
                cached_rgb = np.asarray(source.convert("RGB"))
            if cached_rgb.shape != (SIDE, SIDE, 3):
                raise RuntimeError(f"缓存尺寸错误: {row.image_path} {cached_rgb.shape}")
            digest = hashlib.sha256(cached_rgb.tobytes()).hexdigest()
            if digest != row.pixel_sha256:
                raise RuntimeError(f"缓存像素哈希错误: {row.image_path}")
            if index % 600 == 0:
                print(f"[cache audit] {index}/{EXPECTED_VIEWS}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(
        args.model_id, min_pixels=SIDE * SIDE, max_pixels=SIDE * SIDE
    )
    processor.tokenizer.padding_side = "left"
    yes_ids = candidate_ids(processor.tokenizer, "Yes")
    no_ids = candidate_ids(processor.tokenizer, "No")
    write_protocol(args, subset, yes_ids, no_ids, prompt, args.out_dir)
    print("[protocol] frozen; holdout intersection=0", flush=True)

    load_started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval().to(device)
    print(f"[load] {time.perf_counter()-load_started:.2f}s", flush=True)

    generation_audit(
        model, processor, subset, args.root, device, prompt, args.out_dir,
        args.generation_audit_count,
    )
    if args.generation_audit_only:
        print("[done] generation audit only", flush=True)
        return

    first_group = next(iter(subset.groupby("group_path", sort=False)))[1]
    execution_mode = validate_cuda(
        model, processor, first_group, args.root, device, yes_ids, no_ids,
        prompt, args.anomaly_token, args.out_dir, args.forward_generate_atol,
    )

    views_path = args.out_dir / f"{args.prompt_id}_views.csv"
    timings_path = args.out_dir / "group_timings.csv"
    existing = pd.read_csv(views_path) if views_path.is_file() else pd.DataFrame()
    if len(existing):
        counts = existing.groupby("group_path").size()
        done = set(counts[counts.eq(5)].index)
        existing = existing[existing.group_path.isin(done)].copy()
    else:
        done = set()
    timing_records = pd.read_csv(timings_path).to_dict("records") if timings_path.is_file() else []
    timing_records = [record for record in timing_records if record["group_path"] in done]
    new_views = []
    groups = list(subset.groupby("group_path", sort=False))
    run_started = time.perf_counter()
    for index, (group_path, group) in enumerate(groups, 1):
        if group_path in done:
            continue
        total_started = time.perf_counter()
        decode_started = time.perf_counter()
        images = [load_direct(args.root / image_path) for image_path in group.image_path]
        decode_s = time.perf_counter() - decode_started
        processor_s = 0.0
        forward_s = 0.0
        scores = []
        if execution_mode == "batch5":
            processor_started = time.perf_counter()
            inputs = encode(processor, images, prompt, device)
            processor_s = time.perf_counter() - processor_started
            logits, forward_s = forward_logits(model, inputs)
            scores = [binary_scores(value, yes_ids, no_ids) for value in logits]
        else:
            for image in images:
                processor_started = time.perf_counter()
                inputs = encode(processor, [image], prompt, device)
                processor_s += time.perf_counter() - processor_started
                logits, one_forward_s = forward_logits(model, inputs)
                forward_s += one_forward_s
                scores.append(binary_scores(logits[0], yes_ids, no_ids))
        for (_, row), score in zip(group.iterrows(), scores):
            anomaly_score = score["p_yes" if args.anomaly_token == "yes" else "p_no"]
            new_views.append({
                "prompt_id": args.prompt_id, "class_name": row.class_name,
                "group_path": group_path, "image_path": row.image_path,
                "view_idx": row.view_idx, "condition": row.condition,
                "defect_norm": row.defect_norm, "sample_label": row.sample_label,
                "view_label": row.view_label, "width": row.width, "height": row.height,
                "defect_ratio": row.defect_ratio, "area_bucket": row.area_bucket,
                "p_anomaly": anomaly_score,
                **score,
            })
        timing_records.append({
            "class_name": group.iloc[0].class_name,
            "group_path": group_path,
            "execution_mode": execution_mode,
            "batch_size": 5 if execution_mode == "batch5" else 1,
            "decode_resize_s": decode_s,
            "processor_s": processor_s,
            "forward_s": forward_s,
            "total_s": time.perf_counter() - total_started,
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        })
        combined = pd.concat([existing, pd.DataFrame(new_views)], ignore_index=True)
        combined.to_csv(views_path, index=False)
        pd.DataFrame(timing_records).to_csv(timings_path, index=False)
        if index % 20 == 0 or index == len(groups):
            elapsed = time.perf_counter() - run_started
            completed_now = len(new_views) // 5
            rate = completed_now / elapsed if elapsed else 0.0
            remaining = len(groups) - len(done) - completed_now
            eta = remaining / rate if rate else float("nan")
            print(
                f"[dev720] {index:03d}/{len(groups)} total_done={len(done)+completed_now}; "
                f"last={timing_records[-1]['total_s']:.3f}s eta={eta/60:.1f}m",
                flush=True,
            )

    final = pd.read_csv(views_path)
    if len(final) != EXPECTED_VIEWS or final.image_path.nunique() != EXPECTED_VIEWS:
        raise RuntimeError("最终视角产物数量不完整")
    run_config = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": "float16",
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (args.out_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] {views_path} ({len(final)} views)", flush=True)


if __name__ == "__main__":
    main()
