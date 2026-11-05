"""Recompute reports from all saved validation predictions; no model inference."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
METRICS_SOURCE = ROOT.parent / 'code' / 'migration_metrics.py'
sys.path.insert(0, str(METRICS_SOURCE.parent))
from migration_metrics import evaluate_predictions


def compare(actual, expected, location=''):
    if isinstance(actual, dict):
        for key, value in actual.items():
            compare(value, expected[key], location + '/' + key)
    elif isinstance(actual, (float, int)):
        assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), (location, actual, expected)
    else:
        assert actual == expected, (location, actual, expected)


def main():
    checks = []
    reference_items = None
    for seed in (42, 43, 44):
        for arm in ('B', 'C'):
            d = ROOT / ('arm_%s_s%d' % (arm, seed))
            assert (d / 'complete.json').exists(), d
            for mode, predictions, metrics in (
                ('A', 'initial_val_predictions.json', 'initial_val.json'),
                ('best', 'best_val_generated_predictions.json', 'best_val_metrics.json'),
                ('final', 'final_val_predictions.json', 'final_val_metrics.json'),
            ):
                records = json.loads((d / predictions).read_text())
                assert len(records) == len({x['path'] for x in records}) == 380
                assert all(x['split'] == 'val' for x in records)
                assert sum(x['label'] == 1 for x in records) == 208
                items = [(x['path'], x['category'], x['label']) for x in records]
                if reference_items is None:
                    reference_items = items
                assert items == reference_items, (d, mode, 'validation identity/order changed')
                recalculated = evaluate_predictions(records, [x['score'] for x in records],
                                                     [x['generated'] for x in records])
                compare(recalculated, json.loads((d / metrics).read_text()), d.name + '/' + mode)
                checks.append(dict(run=d.name, checkpoint=mode, n=len(records), state='passed'))
    result = dict(state='complete', checks=checks, evaluated_records=sum(x['n'] for x in checks),
                  unique_validation_images=len(reference_items), new_model_queries=0, test_queries=0,
                  metrics_source_sha256=hashlib.sha256(METRICS_SOURCE.read_bytes()).hexdigest(),
                  audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT / 'prediction_metrics_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
