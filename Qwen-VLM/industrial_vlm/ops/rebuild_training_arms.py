"""Regenerate source selections from the validated inventory, without image IO."""
import json
from pathlib import Path
from industrial_vlm.data import training_subsets, instruction, _jsonl_write, _json_write

root = Path(__file__).resolve().parents[1]
out = root / "prepared/visa"
manifest = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
quality, random_rows, sampling = training_subsets([r for r in manifest if r["split"] == "train"], 768, 42)
_jsonl_write(out / "train_quality_region.jsonl", [instruction(r) for r in quality])
_jsonl_write(out / "train_random_region.jsonl", [instruction(r) for r in random_rows])
_jsonl_write(out / "train_quality_decision.jsonl", [instruction(r, decision_only=True) for r in quality])
audit = json.loads((out / "audit_report.json").read_text())
audit["sampling"] = sampling
audit["quality_train_n"], audit["random_train_n"] = len(quality), len(random_rows)
audit["checks"].update({key: sampling[key] for key in ("normal_source_ids_equal", "category_label_sequence_equal")})
_json_write(out / "audit_report.json", audit)
print(json.dumps(sampling, indent=2))
