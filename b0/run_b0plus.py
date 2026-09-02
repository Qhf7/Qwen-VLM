"""Qwen2.5-VL-3B 的 B0f / B0p / B0w-5crop 一次性本地实验。"""

from __future__ import annotations

import argparse
import gc
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
    from b0.analyze_b0plus import analyze_five_crop, analyze_prompts
    from b0.b0plus_protocol import (
        BATCH_SINGLE_ATOL,
        FIVE_CROP_DEFINITION,
        FIVE_CROPS,
        FORWARD_GENERATE_ATOL,
        PREREGISTERED_AT,
        PROMPTS,
        PROTOCOL_VERSION,
        SELECTION_RULE,
    )
    from b0.run_smoke import PROMPT as ORIGINAL_P0
except ModuleNotFoundError:
    from analyze_b0plus import analyze_five_crop, analyze_prompts
    from b0plus_protocol import (
        BATCH_SINGLE_ATOL,
        FIVE_CROP_DEFINITION,
        FIVE_CROPS,
        FORWARD_GENERATE_ATOL,
        PREREGISTERED_AT,
        PROMPTS,
        PROTOCOL_VERSION,
        SELECTION_RULE,
    )
    from run_smoke import PROMPT as ORIGINAL_P0


SIDE = 448


def candidate_ids(tokenizer, word: str) -> list[int]:
    ids = {
        int(encoded[0])
        for text in (word, " " + word)
        if len(encoded := tokenizer.encode(text, add_special_tokens=False)) == 1
    }
    if not ids:
        raise ValueError(f"{word!r} 不是单 token")
    return sorted(ids)


def yes_probability(logits: torch.Tensor, yes_ids: list[int], no_ids: list[int]) -> float:
    values = logits.float()
    yes_logit = torch.logsumexp(values[yes_ids], dim=0)
    no_logit = torch.logsumexp(values[no_ids], dim=0)
    return float(torch.softmax(torch.stack([no_logit, yes_logit]), dim=0)[1].cpu())


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def move_inputs(inputs: dict, device: torch.device, dtype: torch.dtype) -> dict:
    return {
        key: value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device)
        for key, value in inputs.items()
    }


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"))


def direct_image(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.resize(rgb, (SIDE, SIDE), interpolation=cv2.INTER_AREA))


def five_crops(rgb: np.ndarray) -> list[tuple[str, Image.Image]]:
    height, width = rgb.shape[:2]
    middle_y, middle_x = height // 2, width // 2
    arrays = (
        ("global", rgb),
        ("top-left", rgb[:middle_y, :middle_x]),
        ("top-right", rgb[:middle_y, middle_x:]),
        ("bottom-left", rgb[middle_y:, :middle_x]),
        ("bottom-right", rgb[middle_y:, middle_x:]),
    )
    assert tuple(name for name, _ in arrays) == FIVE_CROPS
    return [(name, direct_image(array)) for name, array in arrays]


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


def encode(processor, images: list[Image.Image], prompt: str, device, dtype) -> dict:
    inputs = processor.apply_chat_template(
        conversations(images, prompt),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"text_kwargs": {"padding": True}},
    )
    return move_inputs(inputs, device, dtype)


def forward_logits(model, inputs: dict, device_name: str) -> tuple[torch.Tensor, float]:
    started = time.perf_counter()
    with torch.inference_mode():
        # B0f 只读取 prefill 最后一个位置的 logits，不生成 token，也不保留 KV cache。
        output = model(**inputs, use_cache=False, logits_to_keep=1)
    sync(device_name)
    elapsed = time.perf_counter() - started
    # MPS 统一内存下不能让返回的 GPU view 继续持有整个 forward
    # 输出（特别是 KV cache），否则连续推理会触发压缩/交换。
    logits = output.logits[:, -1, :].float().cpu()
    del output
    if device_name == "mps":
        torch.mps.empty_cache()
    return logits, elapsed


def generate_first_logits(model, inputs: dict, device_name: str) -> tuple[torch.Tensor, float]:
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
    sync(device_name)
    logits = output.scores[0].float().cpu()
    elapsed = time.perf_counter() - started
    del output
    if device_name == "mps":
        torch.mps.empty_cache()
    return logits, elapsed


def validate_scores(
    model, processor, subset: pd.DataFrame, root: Path, device, dtype, device_name: str,
    yes_ids: list[int], no_ids: list[int], baseline_path: Path, out_dir: Path,
) -> None:
    validation_path = out_dir / "forward_validation.csv"
    if validation_path.is_file():
        frame = pd.read_csv(validation_path)
        if "forward_use_cache" in frame and not frame.forward_use_cache.astype(bool).any():
            max_fg = float(frame.abs_forward_generate.max())
            max_bs = float(frame.abs_batch_single.max())
            if max_fg > FORWARD_GENERATE_ATOL or max_bs > BATCH_SINGLE_ATOL:
                raise RuntimeError("已有 validation 产物没有通过预注册误差")
            print(
                f"[validation] reuse {validation_path}; forward-generate={max_fg:.3g}, "
                f"batch-single={max_bs:.3g}", flush=True,
            )
            return
        print("[validation] 旧产物使用 KV cache，按 B0f 定义重测", flush=True)
    rows = subset.head(5).to_dict("records")
    baseline = pd.read_csv(baseline_path).set_index("image_path")
    records = []
    single_scores = []
    images = []
    for row in rows:
        image = direct_image(load_rgb(root / row["image_path"]))
        images.append(image)
        inputs = encode(processor, [image], PROMPTS["P0"], device, dtype)
        forward, forward_s = forward_logits(model, inputs, device_name)
        generated, generate_s = generate_first_logits(model, inputs, device_name)
        p_forward = yes_probability(forward[0], yes_ids, no_ids)
        p_generate = yes_probability(generated[0], yes_ids, no_ids)
        p_baseline = float(baseline.loc[row["image_path"], "p_yes"])
        single_scores.append(p_forward)
        records.append({
            "image_path": row["image_path"],
            "p_forward": p_forward,
            "p_generate": p_generate,
            "p_saved_baseline": p_baseline,
            "abs_forward_generate": abs(p_forward - p_generate),
            "abs_forward_saved": abs(p_forward - p_baseline),
            "forward_use_cache": False,
            "forward_s": forward_s,
            "generate_one_s": generate_s,
        })
    batch_inputs = encode(processor, images, PROMPTS["P0"], device, dtype)
    batch_logits, batch_s = forward_logits(model, batch_inputs, device_name)
    batch_scores = [yes_probability(logits, yes_ids, no_ids) for logits in batch_logits]
    for record, batch_score, single_score in zip(records, batch_scores, single_scores):
        record["p_batch5"] = batch_score
        record["abs_batch_single"] = abs(batch_score - single_score)
        record["batch5_forward_s"] = batch_s
    frame = pd.DataFrame(records)
    frame.to_csv(validation_path, index=False)
    max_fg = float(frame.abs_forward_generate.max())
    max_bs = float(frame.abs_batch_single.max())
    print(f"[validation] max forward-generate={max_fg:.3g}, batch-single={max_bs:.3g}", flush=True)
    if max_fg > FORWARD_GENERATE_ATOL:
        raise RuntimeError(f"forward/generate 超出预注册误差: {max_fg}")
    if max_bs > BATCH_SINGLE_ATOL:
        raise RuntimeError(f"batch/single 超出预注册误差: {max_bs}")


def run_single_latency(
    model, processor, subset: pd.DataFrame, root: Path, device, dtype, device_name: str,
    yes_ids: list[int], no_ids: list[int], out_dir: Path,
) -> None:
    path = out_dir / "single_view_latency.csv"
    if path.is_file():
        prior = pd.read_csv(path)
        if (
            len(prior) == len(subset)
            and "forward_use_cache" in prior
            and not prior.forward_use_cache.astype(bool).any()
        ):
            print(f"[latency] reuse {path}", flush=True)
            return
        print("[single latency] 旧产物使用 KV cache，按 B0f 定义重测", flush=True)
    records = []
    for index, row in enumerate(subset.to_dict("records"), 1):
        total_start = time.perf_counter()
        decode_start = time.perf_counter()
        image = direct_image(load_rgb(root / row["image_path"]))
        decode_s = time.perf_counter() - decode_start
        processor_start = time.perf_counter()
        inputs = encode(processor, [image], PROMPTS["P0"], device, dtype)
        processor_s = time.perf_counter() - processor_start
        logits, forward_s = forward_logits(model, inputs, device_name)
        records.append({
            "group_path": row["group_path"], "image_path": row["image_path"],
            "p_yes": yes_probability(logits[0], yes_ids, no_ids),
            "forward_use_cache": False,
            "decode_resize_s": decode_s, "processor_s": processor_s,
            "forward_s": forward_s, "total_s": time.perf_counter() - total_start,
        })
        pd.DataFrame(records).to_csv(path, index=False)
        if index % 10 == 0:
            print(f"[single latency] {index}/{len(subset)}", flush=True)


def run_batch5_latency(
    model, processor, subset: pd.DataFrame, root: Path, device, dtype, device_name: str,
    yes_ids: list[int], no_ids: list[int], out_dir: Path, repeats: int = 5,
) -> None:
    """真实 5 图 batch prefill；只做延迟对照，不用于后续打分。"""
    path = out_dir / "batch5_latency.csv"
    existing = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    if (
        len(existing) == repeats
        and "forward_use_cache" in existing
        and not existing.forward_use_cache.astype(bool).any()
    ):
        print(f"[batch5 latency] reuse {path}", flush=True)
        return
    records = []
    groups = list(subset.groupby("group_path", sort=False))[:repeats]
    for index, (group_path, group) in enumerate(groups, 1):
        total_start = time.perf_counter()
        decode_start = time.perf_counter()
        images = [direct_image(load_rgb(root / image_path)) for image_path in group.image_path]
        decode_s = time.perf_counter() - decode_start
        processor_start = time.perf_counter()
        inputs = encode(processor, images, PROMPTS["P0"], device, dtype)
        processor_s = time.perf_counter() - processor_start
        logits, forward_s = forward_logits(model, inputs, device_name)
        records.append({
            "repeat": index, "group_path": group_path,
            "batch_size": len(images), "forward_use_cache": False,
            "mean_p_yes": float(np.mean([
                yes_probability(value, yes_ids, no_ids) for value in logits
            ])),
            "decode_resize_s": decode_s, "processor_s": processor_s,
            "forward_s": forward_s, "total_s": time.perf_counter() - total_start,
        })
        pd.DataFrame(records).to_csv(path, index=False)
        print(f"[batch5 latency] {index}/{repeats} {records[-1]['total_s']:.2f}s", flush=True)


def run_prompt(
    prompt_id: str, model, processor, subset: pd.DataFrame, root: Path, device, dtype,
    device_name: str, yes_ids: list[int], no_ids: list[int], out_dir: Path,
) -> None:
    view_path = out_dir / f"{prompt_id}_views.csv"
    batch_path = out_dir / f"{prompt_id}_batches.csv"
    existing = pd.read_csv(view_path) if view_path.is_file() else pd.DataFrame()
    done = set(existing.group_path) if len(existing) else set()
    new_views = []
    batch_records = pd.read_csv(batch_path).to_dict("records") if batch_path.is_file() else []
    for record in batch_records:
        record.setdefault("execution_mode", "batch5_use_cache_true")
    groups = list(subset.groupby("group_path", sort=False))
    for index, (group_path, group) in enumerate(groups, 1):
        if group_path in done:
            continue
        total_start = time.perf_counter()
        decode_start = time.perf_counter()
        images = [direct_image(load_rgb(root / path)) for path in group.image_path]
        decode_s = time.perf_counter() - decode_start
        processor_s = 0.0
        forward_s = 0.0
        scores = []
        # 分数已验证与 batch5 完全一致；串行执行避免 18GB MPS 统一内存换页。
        for image in images:
            processor_start = time.perf_counter()
            inputs = encode(processor, [image], PROMPTS[prompt_id], device, dtype)
            processor_s += time.perf_counter() - processor_start
            logits, elapsed = forward_logits(model, inputs, device_name)
            forward_s += elapsed
            scores.append(yes_probability(logits[0], yes_ids, no_ids))
        total_s = time.perf_counter() - total_start
        for (_, row), score in zip(group.iterrows(), scores):
            new_views.append({
                "prompt_id": prompt_id, "class_name": row.class_name,
                "group_path": group_path, "image_path": row.image_path,
                "view_idx": row.view_idx, "condition": row.condition,
                "sample_label": row.sample_label, "view_label": row.view_label,
                "p_yes": score,
            })
        batch_records.append({
            "prompt_id": prompt_id, "class_name": group.iloc[0].class_name,
            "group_path": group_path, "sample_label": group.iloc[0].sample_label,
            "decode_resize_s": decode_s, "processor_s": processor_s,
            "forward_s": forward_s, "total_s": total_s,
            "execution_mode": "single_serial_use_cache_false",
        })
        combined = pd.concat([existing, pd.DataFrame(new_views)], ignore_index=True)
        combined.to_csv(view_path, index=False)
        pd.DataFrame(batch_records).to_csv(batch_path, index=False)
        print(f"[{prompt_id}] batch {index:02d}/{len(groups)} {total_s:.2f}s", flush=True)


def run_five_crop(
    prompt_id: str, model, processor, subset: pd.DataFrame, root: Path, device, dtype,
    device_name: str, yes_ids: list[int], no_ids: list[int], out_dir: Path,
) -> None:
    score_path = out_dir / f"{prompt_id}_5crop_scores.csv"
    batch_path = out_dir / f"{prompt_id}_5crop_batches.csv"
    existing = pd.read_csv(score_path) if score_path.is_file() else pd.DataFrame()
    done = set(existing.image_path) if len(existing) else set()
    new_scores = []
    batches = pd.read_csv(batch_path).to_dict("records") if batch_path.is_file() else []
    rows = subset.to_dict("records")
    for index, row in enumerate(rows, 1):
        if row["image_path"] in done:
            continue
        total_start = time.perf_counter()
        decode_start = time.perf_counter()
        crops = five_crops(load_rgb(root / row["image_path"]))
        decode_s = time.perf_counter() - decode_start
        processor_s = 0.0
        forward_s = 0.0
        scores = []
        for _, image in crops:
            processor_start = time.perf_counter()
            inputs = encode(processor, [image], PROMPTS[prompt_id], device, dtype)
            processor_s += time.perf_counter() - processor_start
            logits, elapsed = forward_logits(model, inputs, device_name)
            forward_s += elapsed
            scores.append(yes_probability(logits[0], yes_ids, no_ids))
        total_s = time.perf_counter() - total_start
        for (crop_name, _), score in zip(crops, scores):
            new_scores.append({
                "prompt_id": prompt_id, "class_name": row["class_name"],
                "group_path": row["group_path"], "image_path": row["image_path"],
                "sample_label": row["sample_label"], "view_label": row["view_label"],
                "crop_name": crop_name, "p_yes": score,
            })
        batches.append({
            "prompt_id": prompt_id, "class_name": row["class_name"],
            "group_path": row["group_path"], "image_path": row["image_path"],
            "sample_label": row["sample_label"], "decode_resize_s": decode_s,
            "processor_s": processor_s, "forward_s": forward_s, "total_s": total_s,
            "execution_mode": "five_crop_serial_use_cache_false",
        })
        pd.concat([existing, pd.DataFrame(new_scores)], ignore_index=True).to_csv(score_path, index=False)
        pd.DataFrame(batches).to_csv(batch_path, index=False)
        if index % 5 == 0:
            print(f"[5crop {prompt_id}] {index:03d}/{len(rows)} {total_s:.2f}s", flush=True)


def write_protocol(out_dir: Path, args, yes_ids: list[int], no_ids: list[int]) -> None:
    data = {
        "version": PROTOCOL_VERSION,
        "preregistered_at": PREREGISTERED_AT,
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "subset": str(args.subset),
        "side": SIDE,
        "prompt_order": list(PROMPTS),
        "prompts": PROMPTS,
        "prompt_sha256": {
            key: hashlib.sha256(value.encode()).hexdigest() for key, value in PROMPTS.items()
        },
        "selection_rule": SELECTION_RULE,
        "five_crops": list(FIVE_CROPS),
        "five_crop_definition": FIVE_CROP_DEFINITION,
        "yes_token_ids": yes_ids,
        "no_token_ids": no_ids,
        "forward_generate_atol": FORWARD_GENERATE_ATOL,
        "batch_single_atol": BATCH_SINGLE_ATOL,
        "forward_use_cache": False,
        "batch5_latency_repeats": 5,
    }
    (out_dir / "preregistered_protocol.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-id", default="models/modelscope/models/Qwen--Qwen2.5-VL-3B-Instruct/snapshots/master"
    )
    ap.add_argument("--subset", type=Path, default=Path("artifacts/b0/smoke20_manifest.parquet"))
    ap.add_argument("--root", type=Path, default=Path("Dataset"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/b0plus"))
    ap.add_argument("--device", choices=["mps", "cpu"], default="mps")
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = ap.parse_args()
    assert PROMPTS["P0"] == ORIGINAL_P0, "P0 与 B0 prompt 不一致"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    subset = pd.read_parquet(args.subset).sort_values(
        ["class_name", "condition", "group_path", "view_idx"], kind="mergesort"
    )
    assert len(subset) == 100 and subset.group_path.nunique() == 20
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    processor = AutoProcessor.from_pretrained(
        args.model_id, min_pixels=SIDE * SIDE, max_pixels=SIDE * SIDE
    )
    processor.tokenizer.padding_side = "left"
    yes_ids = candidate_ids(processor.tokenizer, "Yes")
    no_ids = candidate_ids(processor.tokenizer, "No")
    write_protocol(args.out_dir, args, yes_ids, no_ids)
    print(f"[protocol] {PROTOCOL_VERSION} P0/P1/P2 已冻结", flush=True)

    load_start = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, dtype=dtype, low_cpu_mem_usage=True, attn_implementation="eager"
    )
    model.eval().to(device)
    print(f"[load] {time.perf_counter()-load_start:.1f}s", flush=True)

    validate_scores(
        model, processor, subset, args.root, device, dtype, args.device,
        yes_ids, no_ids, Path("artifacts/b0/qwen25_vl_3b_views.csv"), args.out_dir,
    )
    run_single_latency(
        model, processor, subset, args.root, device, dtype, args.device,
        yes_ids, no_ids, args.out_dir,
    )
    run_batch5_latency(
        model, processor, subset, args.root, device, dtype, args.device,
        yes_ids, no_ids, args.out_dir,
    )
    for prompt_id in PROMPTS:
        run_prompt(
            prompt_id, model, processor, subset, args.root, device, dtype, args.device,
            yes_ids, no_ids, args.out_dir,
        )
    best_prompt = analyze_prompts(args.out_dir)
    print(f"[selection] {best_prompt}", flush=True)
    run_five_crop(
        best_prompt, model, processor, subset, args.root, device, dtype, args.device,
        yes_ids, no_ids, args.out_dir,
    )
    analyze_five_crop(args.out_dir, best_prompt)

    config = {
        "model": "Qwen/Qwen2.5-VL-3B-Instruct", "device": args.device,
        "dtype": args.dtype, "best_prompt": best_prompt,
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    (args.out_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del model, processor
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
