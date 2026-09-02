"""在冻结的 20 样本子集上运行 Qwen-VL B0 冒烟测试。

统一输出契约：第一个生成 token 必须是 Yes/No，随后输出一个 JSON 位置词。
``generate(..., output_scores=True)`` 的第一个 score 同时用于 Yes/No 概率，避免
为了 logits 再做一次前向。
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from transformers import AutoModelForImageTextToText, AutoProcessor


SIDE = 448
REGIONS = (
    "top-left", "top-center", "top-right",
    "middle-left", "center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right", "none",
)
PROMPT = """You are inspecting one view of an industrial component for visible defects.
A defect means a visible scratch, crack, deformation, discoloration, stain, missing or broken part, pit, burn mark, coating problem, contaminant, or another localized abnormality. Normal edges, holes, reflections, shadows, text, and regular texture are not defects.

The first generated token must be exactly Yes or No. Then output exactly one JSON object with a region selected from: top-left, top-center, top-right, middle-left, center, middle-right, bottom-left, bottom-center, bottom-right, none.
Format: Yes {\"region\":\"center\"}
For a normal image use: No {\"region\":\"none\"}
Do not output Markdown or an explanation."""


def _device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS 不可用")
    return torch.device(name)


def _dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _candidate_ids(tokenizer, word: str) -> list[int]:
    ids: set[int] = set()
    for text in (word, " " + word):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) == 1:
            ids.add(int(encoded[0]))
    if not ids:
        raise ValueError(f"{word!r} 不是单 token，无法读取 next-token 概率")
    return sorted(ids)


def _yes_probability(first_scores: torch.Tensor, yes_ids: list[int], no_ids: list[int]) -> tuple[float, float, float]:
    logits = first_scores.float()
    yes_logit = torch.logsumexp(logits[yes_ids], dim=0)
    no_logit = torch.logsumexp(logits[no_ids], dim=0)
    probability = torch.softmax(torch.stack([no_logit, yes_logit]), dim=0)[1]
    return float(probability.cpu()), float(yes_logit.cpu()), float(no_logit.cpu())


def _normalize_region(value: str) -> str:
    value = value.strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "middle-center": "center",
        "center-center": "center",
        "centre": "center",
        "no-defect": "none",
        "normal": "none",
    }
    return aliases.get(value, value)


def parse_contract(text: str) -> tuple[str, str, bool, bool]:
    stripped = text.strip()
    answer_match = re.match(r"^(Yes|No)\b", stripped, flags=re.IGNORECASE)
    answer = answer_match.group(1).title() if answer_match else ""
    region = ""
    json_ok = False
    object_match = re.search(r"\{.*?\}", stripped, flags=re.DOTALL)
    if object_match:
        try:
            obj = json.loads(object_match.group(0))
            region = _normalize_region(str(obj.get("region", "")))
            json_ok = region in REGIONS
        except (json.JSONDecodeError, TypeError):
            pass
    return answer, region, bool(answer_match), json_ok


def gt_region(row: dict, dataset_root: Path) -> str:
    if not row["view_label"] or not row["mask_path"]:
        return "none"
    mask = cv2.imread(str(dataset_root / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return ""
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return "none"
    col = min(2, int(3 * float(xs.mean()) / mask.shape[1]))
    band = min(2, int(3 * float(ys.mean()) / mask.shape[0]))
    return (
        ("top-left", "top-center", "top-right"),
        ("middle-left", "center", "middle-right"),
        ("bottom-left", "bottom-center", "bottom-right"),
    )[band][col]


def _f1max(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(np.max(2 * precision * recall / (precision + recall + 1e-10)))


def build_report(results: pd.DataFrame, config: dict) -> str:
    group = (
        results.groupby(["class_name", "group_path"], as_index=False)
        .agg(
            sample_label=("sample_label", "first"),
            anomaly_score=("p_yes", "max"),
            latency_s=("total_s", "sum"),
        )
    )
    labels = group.sample_label.to_numpy()
    scores = group.anomaly_score.to_numpy()
    overall_auroc = float(roc_auc_score(labels, scores))
    overall_aupr = float(average_precision_score(labels, scores))
    overall_f1 = _f1max(labels, scores)

    per_class = []
    for class_name, sub in group.groupby("class_name"):
        if sub.sample_label.nunique() == 2:
            per_class.append(float(roc_auc_score(sub.sample_label, sub.anomaly_score)))

    # 探针与断点续跑各有一次 MPS 图编译；排除第一个 5 视觉样本，
    # 避免把编译抖动误计入 warm 分位数。
    cold_group = results.iloc[0].group_path
    warm = results[results.group_path != cold_group]
    warm_groups = group[group.group_path != cold_group]
    positive_regions = results[results.view_label == 1]
    region_valid = positive_regions[positive_regions.json_ok]
    region_accuracy = (
        float((region_valid.parsed_region == region_valid.gt_region).mean())
        if len(region_valid) else float("nan")
    )
    lines = [
        "=" * 82,
        f"B0 冒烟报告 | {config['model_id']}",
        "=" * 82,
        f"设备 {config['device']} / {config['dtype']} | 参数 {config['n_parameters']/1e9:.2f}B | "
        f"加载 {config['load_s']:.1f}s",
        f"子集 {group.group_path.nunique()} 个完整样本 / {len(results)} 视角 / "
        f"{group.class_name.nunique()} 类 | direct {SIDE}",
        "",
        "【图像级，5视角 max】",
        f"  overall I-AUROC={overall_auroc:.4f} I-AUPR={overall_aupr:.4f} F1max={overall_f1:.4f}",
        f"  10 类宏平均 I-AUROC={float(np.mean(per_class)):.4f}",
        "",
        "【输出契约】",
        f"  首词 Yes/No 可解析率 = {results.answer_ok.mean():.2%}",
        f"  JSON + 合法位置词率 = {results.json_ok.mean():.2%}",
        f"  完整契约率 = {(results.answer_ok & results.json_ok).mean():.2%}",
        f"  异常视角位置准确率（仅合法 JSON） = {region_accuracy:.2%} (N={len(region_valid)})",
        "",
        "【分类诊断】",
        f"  视角级 accuracy@0.5 = {((results.p_yes >= .5).astype(int) == results.view_label).mean():.2%}",
        f"  样本级 accuracy@0.5 = {((scores >= .5).astype(int) == labels).mean():.2%}",
        f"  P(Yes) OK/NG中位 = {group.loc[group.sample_label==0, 'anomaly_score'].median():.4f} / "
        f"{group.loc[group.sample_label==1, 'anomaly_score'].median():.4f}",
        "",
        "【延迟】",
        f"  冷启动首视角 = {results.iloc[0].total_s:.2f}s",
        f"  单视角 warm p50/p95（排除首个5视角样本） = "
        f"{np.percentile(warm.total_s,50):.2f}s / "
        f"{np.percentile(warm.total_s,95):.2f}s",
        f"  其中 generate warm p50/p95 = {np.percentile(warm.generate_s,50):.2f}s / "
        f"{np.percentile(warm.generate_s,95):.2f}s",
        f"  5视角样本 warm p50/p95 = "
        f"{np.percentile(warm_groups.latency_s,50):.2f}s / "
        f"{np.percentile(warm_groups.latency_s,95):.2f}s",
        f"  ≤1s/样本："
        f"{'达成' if np.percentile(warm_groups.latency_s,95) <= 1 else '未达成'}",
        "=" * 82,
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--display-model-id", default="")
    ap.add_argument("--slug", required=True)
    ap.add_argument(
        "--subset", type=Path,
        default=Path("artifacts/b0/smoke20_manifest.parquet"),
    )
    ap.add_argument("--root", type=Path, default=Path("Dataset"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/b0"))
    ap.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    ap.add_argument("--device", choices=["mps", "cpu"], default="mps")
    ap.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--limit-views", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    display_model_id = args.display_model_id or args.model_id

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / f"{args.slug}_views.csv"
    config_path = args.out_dir / f"{args.slug}_config.json"
    report_path = args.out_dir / f"{args.slug}_report.txt"
    device = _device(args.device)
    dtype = _dtype(args.dtype)

    full_subset = pd.read_parquet(args.subset)
    subset = full_subset
    if args.limit_views:
        subset = subset.head(args.limit_views)
    rows = subset.to_dict("records")

    existing = pd.DataFrame()
    done: set[str] = set()
    if args.resume and result_path.is_file():
        existing = pd.read_csv(result_path)
        done = set(existing.image_path)
        print(f"[resume] 已有 {len(done)} 个视角", flush=True)

    print(f"[load] processor {args.model_id}", flush=True)
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        min_pixels=SIDE * SIDE,
        max_pixels=SIDE * SIDE,
    )
    print(f"[tokens] Yes={_candidate_ids(processor.tokenizer, 'Yes')} "
          f"No={_candidate_ids(processor.tokenizer, 'No')}", flush=True)
    yes_ids = _candidate_ids(processor.tokenizer, "Yes")
    no_ids = _candidate_ids(processor.tokenizer, "No")

    print(f"[load] model {args.model_id}", flush=True)
    load_start = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval().to(device)
    load_s = time.perf_counter() - load_start
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"[load] 完成 {load_s:.1f}s, {n_parameters/1e9:.2f}B params", flush=True)

    new_results: list[dict] = []
    for index, row in enumerate(rows, 1):
        if row["image_path"] in done:
            continue
        total_start = time.perf_counter()
        decode_start = time.perf_counter()
        with Image.open(args.root / row["image_path"]) as source:
            rgb = np.asarray(source.convert("RGB"))
        resized = cv2.resize(rgb, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
        image = Image.fromarray(resized)
        decode_resize_s = time.perf_counter() - decode_start

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        processor_start = time.perf_counter()
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(device=device, dtype=dtype) if value.is_floating_point()
            else value.to(device)
            for key, value in inputs.items()
        }
        processor_s = time.perf_counter() - processor_start

        generate_start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        if args.device == "mps":
            torch.mps.synchronize()
        generate_s = time.perf_counter() - generate_start
        prompt_len = inputs["input_ids"].shape[1]
        new_ids = generated.sequences[:, prompt_len:]
        text = processor.batch_decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        p_yes, yes_logit, no_logit = _yes_probability(
            generated.scores[0][0], yes_ids, no_ids
        )
        answer, region, answer_ok, json_ok = parse_contract(text)
        grid = inputs.get("image_grid_thw")
        grid_values = grid[0].detach().cpu().tolist() if grid is not None else [0, 0, 0]
        total_s = time.perf_counter() - total_start
        record = {
            "model_id": display_model_id,
            "class_name": row["class_name"],
            "group_path": row["group_path"],
            "image_path": row["image_path"],
            "condition": row["condition"],
            "view_idx": row["view_idx"],
            "sample_label": row["sample_label"],
            "view_label": row["view_label"],
            "gt_region": gt_region(row, args.root),
            "p_yes": p_yes,
            "yes_logit": yes_logit,
            "no_logit": no_logit,
            "generated_text": text.replace("\n", "\\n"),
            "parsed_answer": answer,
            "parsed_region": region,
            "answer_ok": answer_ok,
            "json_ok": json_ok,
            "grid_t": int(grid_values[0]),
            "grid_h": int(grid_values[1]),
            "grid_w": int(grid_values[2]),
            "decode_resize_s": decode_resize_s,
            "processor_s": processor_s,
            "generate_s": generate_s,
            "total_s": total_s,
        }
        new_results.append(record)
        combined = pd.concat([existing, pd.DataFrame(new_results)], ignore_index=True)
        combined.to_csv(result_path, index=False)
        print(
            f"[{index:03d}/{len(rows)}] C{int(row['view_idx'])+1} "
            f"{row['condition']} pYes={p_yes:.4f} {total_s:.2f}s | {text[:90]!r}",
            flush=True,
        )

    results = pd.concat([existing, pd.DataFrame(new_results)], ignore_index=True)
    config = {
        "model_id": display_model_id,
        "device": args.device,
        "dtype": args.dtype,
        "side": SIDE,
        "max_new_tokens": args.max_new_tokens,
        "prompt": PROMPT,
        "yes_token_ids": yes_ids,
        "no_token_ids": no_ids,
        "load_s": load_s,
        "n_parameters": n_parameters,
        "n_views": len(results),
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    complete = set(results.image_path) == set(full_subset.image_path)
    if complete:
        report = build_report(
            results[results.image_path.isin(full_subset.image_path)].copy(), config
        )
        report_path.write_text(report + "\n", encoding="utf-8")
        print(report)
        print(f"[write] {result_path} / {config_path} / {report_path}")
    else:
        done_in_full = results.image_path.isin(full_subset.image_path).sum()
        print(f"[partial] 当前 {done_in_full}/{len(full_subset)}，未生成最终报告")

    del model, processor
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
