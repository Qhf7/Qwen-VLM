"""Read compact second-round progress; no image or model access."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
exp = root / "experiments/multitask_v2"
result = {}
for name in ("state", "scoring_state"):
    path = exp / (name + ".json")
    result[name] = json.loads(path.read_text()) if path.exists() else {"status": "not_started"}
result["training"] = {}
for arm in ("region_region", "decision_decision", "region_decision"):
    directory = root / "runs/multitask_v2" / arm
    events = directory / "metrics.jsonl"
    if not events.exists():
        continue
    rows = [json.loads(line) for line in events.read_text().splitlines() if line.strip()]
    trains = [r for r in rows if r["event"] == "train"]
    validations = [r for r in rows if r["event"] == "validation"]
    result["training"][arm] = {
        "latest_update": trains[-1]["optimizer_step"] if trains else 0,
        "last_train_loss": trains[-1]["loss"] if trains else None,
        "last_validation_ce": validations[-1]["validation_loss"] if validations else None,
        "peak_allocated_gib": trains[-1]["peak_allocated_gib"] if trains else None,
        "complete": (directory / "summary.json").exists()
                    and json.loads((directory / "summary.json").read_text())["status"] == "complete",
    }
for group, filename in (("multitask_v2", "predictions.jsonl"), ("multitask_v2_scores", "scores.jsonl")):
    result[group] = {}
    directory = root / "evaluations" / group
    if directory.exists():
        for path in sorted(directory.glob("*/meta.json")):
            meta = json.loads(path.read_text())
            predictions = path.with_name(filename)
            result[group][path.parent.name] = {
                "status": meta["status"],
                "rows": sum(bool(line.strip()) for line in predictions.read_text().splitlines()) if predictions.exists() else 0,
            }
print(json.dumps(result, ensure_ascii=False, indent=2))
