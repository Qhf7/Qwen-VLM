"""Create fixed pilot settings before inspecting real test predictions."""
import json
from pathlib import Path

folder = Path(__file__).resolve().parents[1] / "configs"
folder.mkdir(exist_ok=True)
remote = "/root/autodl-tmp/industrial_vlm"
for name in ["quality_region", "random_region", "quality_decision"]:
    suffix = "_decision" if name.endswith("decision") else ""
    config = {
        "model": remote + "/models/Qwen3-VL-8B-Instruct",
        "train_file": remote + "/prepared/visa/train_" + name + ".jsonl",
        "val_file": remote + "/prepared/visa/val_balanced_120" + suffix + ".jsonl",
        "image_root": remote + "/datasets/VisA",
        "output": remote + "/runs/" + name,
        "max_steps": 96, "seed": 42, "max_pixels": 524288, "max_length": 2048,
        "gradient_accumulation_steps": 8, "learning_rate": 1e-4,
        "warmup_steps": 5, "save_steps": 32, "eval_steps": 32,
        "val_max_samples": 24, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    }
    (folder / (name + ".json")).write_text(json.dumps(config, indent=2) + "\n")
plan = {
    "status": "preregistered_before_real_test_predictions", "date": "2026-10-23",
    "task": "Single-image supervised industrial inspection with decision and one union box",
    "dataset": "VisA custom SHA256-deduplicated stratified70/15/15 split, seed42",
    "train_images_per_arm": 768,
    "primary_test": "Fixed category/label-balanced 120-image VisA heldout test",
    "external_test": "Fixed 15 categories x2 labels x5 images MVTec mirror pilot,150 images,seed31415",
    "max_pixels": 524288,
    "model_selection": "Fixed final96-step adapter; validation loss audit only; no test-based checkpoint choice",
    "arms": ["base_region", "quality_region", "random_region", "quality_decision",
             "base_decision", "quality_region_under_decision_prompt"],
    "training_budget": "96 updates x8 images=768 source exposures per arm; same optimizer and seed; target token counts differ with task",
    "quality_ablation": "Matched normal sources and per-step category-label order; anomaly mask-area coverage sampling vs uniform random; valid labels in both arms",
    "region_ablation": "Identical source images; common decision-only eval prompt for region-trained and decision-trained adapters",
    "metrics": ["schema-valid output rate", "macroF1", "balancedaccuracy", "normalFPR", "abnormalFNR",
                "normalfailurerate", "unionbox IoU recall0.3/0.5 over all anomalous images", "source-level bootstrap CI"],
    "limits": ["one training seed", "small training sample and pilot tests",
               "exact-byte duplicate isolation; no physical part identity metadata",
               "supervised training protocol", "public benchmark pretraining contamination cannot be excluded"],
}
(folder / "experiment_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
