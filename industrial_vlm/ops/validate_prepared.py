"""Fail before training if source isolation or the fixed experiment matrix differs."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "prepared/visa"
audit = json.loads((folder / "audit_report.json").read_text())
assert audit["valid_unique_images"] == 10821, audit["valid_unique_images"]
assert audit["rejected_rows"] == 0, audit.get("reject_reasons")
split = json.loads((folder / "split.json").read_text())
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    assert not set(split[a]).intersection(split[b]), (a, b)
names = ["train_quality_region", "train_random_region", "train_quality_decision",
         "val_balanced_120", "test_balanced_120", "val_balanced_120_decision", "test_balanced_120_decision"]
rows = {name: [json.loads(line) for line in (folder / (name + ".jsonl")).read_text().splitlines()] for name in names}
quality, random, decision = (rows[name] for name in names[:3])
assert len(quality) == len(random) == len(decision) == 768
assert [r["source_id"] for r in quality] == [r["source_id"] for r in decision]
quota = lambda records: Counter((r["category"], r["is_anomalous"]) for r in records)
assert quota(quality) == quota(random)
assert [r["source_id"] for r in quality if not r["is_anomalous"]] == [r["source_id"] for r in random if not r["is_anomalous"]]
assert [(r["category"], r["is_anomalous"]) for r in quality] == [(r["category"], r["is_anomalous"]) for r in random]
assert Counter(r["is_anomalous"] for r in quality) == {False: 384, True: 384}
for name, records in rows.items():
    expected = "train" if name.startswith("train") else "val" if name.startswith("val") else "test"
    assert all(r["source_id"] in split[expected] for r in records)
    if not name.startswith("train"):
        assert len(records) == 120
        assert Counter(r["is_anomalous"] for r in records) == {False: 60, True: 60}
    for record in records:
        assert "anomaly_area_fraction" in record
        target = json.loads(record["answer"])
        assert target["is_anomalous"] == record["is_anomalous"]
        assert bool(record["boxes"]) == record["is_anomalous"]
        assert ("boxes" in target) == (not name.endswith("decision"))
checks = {"status": "passed", "source_splits_disjoint": True, "n_sources": audit["valid_unique_images"],
          "train_samples_each_arm": 768, "train_normal_each_arm": 384, "train_anomaly_each_arm": 384,
          "test_pilot": 120, "test_pilot_normal": 60, "test_pilot_anomaly": 60,
          "config_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "configs").glob("*.json"))},
          "data_hashes": {name: hashlib.sha256((folder / (name + ".jsonl")).read_bytes()).hexdigest() for name in names}}
(ROOT / "ops/prepared_validation.json").write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2), flush=True)
