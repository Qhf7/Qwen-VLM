"""Audit complete tool runs; optional GT geometry only after saved-run gates.

No model, optimization or tool selection. Saved coordinates never change.
All record/RGB checks finish before any mask path is resolved or opened.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from tool_workflow import (HERE, REPO, CONDITIONS, atomic_json, sha256_file,
                           parse_request, request_for_condition, crop_bounds)
for folder in (HERE.parent / 'code', REPO):
    sys.path.insert(0, str(folder))


def mean(values):
    return sum(values) / len(values) if values else None


def audit_records(run, rows):
    if run.get('status') != 'complete':
        raise ValueError('All predictions must be complete before post-hoc audit')
    conditions = run['conditions']
    if len(set(conditions)) != len(conditions) or any(c not in CONDITIONS for c in conditions):
        raise ValueError('Invalid condition list')
    n = run['n']
    if n <= 0 or len(rows) != n * len(conditions):
        raise ValueError('Prediction denominator incomplete')
    by_condition = {condition: [] for condition in conditions}
    for row in rows:
        by_condition[row['condition']].append(row)
    for condition, values in by_condition.items():
        if len(values) != n or [r['path'] for r in values] != run['selected_paths']:
            raise ValueError('Condition order differs from frozen selection')
        for row in values:
            if row['split'] != 'val' or type(row['label']) is not int or row['label'] not in (0, 1):
                raise ValueError('Only frozen validation metadata permitted')
            if sha256_file(row['path']) != row['image_sha256']:
                raise ValueError('RGB identity differs: ' + row['path'])
            plan, error = parse_request(row['planner']['raw_text'])
            if plan != row['planner']['request'] or error != row['planner']['error']:
                raise ValueError('Saved tool request differs from exact parser')
            request, rejected = request_for_condition(condition, plan, error, row['path'], run['config']['seed'])
            bounds = crop_bounds(request['center']) if request['action'] == 'crop' else None
            if (row['request_executed'] != request or row['bbox_xyxy_relative'] != bounds
                    or row['tool_calls'] != int(bounds is not None) or row['request_rejected'] != bool(rejected)):
                raise ValueError('Tool execution differs from model/control request')
            expected_images = 1 if condition == 'no_tool' else 2
            if row['n_images'] != expected_images or row['image_tokens_total'] != expected_images * 196:
                raise ValueError('Visual budgets differ')
            if row['score_yes_minus_no'] != row['yes_logit'] - row['no_logit']:
                raise ValueError('Score definition differs')
    first = by_condition[conditions[0]]
    for pos, base in enumerate(first):
        for condition in conditions[1:]:
            row = by_condition[condition][pos]
            if any(row[k] != base[k] for k in ('path', 'category', 'label', 'split', 'cache_index', 'planner')):
                raise ValueError('Paired conditions differ in sample or exact planner output')
    return by_condition


def posthoc_geometry(by_condition):
    import numpy as np
    from visa_pipeline import load_mask_tensor, mask_path_for_image
    first = next(iter(by_condition.values()))
    coordinate = (np.arange(224) + 0.5) / 224
    masks, sources = {}, []
    for row in first:
        path = mask_path_for_image(row['path'], row['label'])
        before = sha256_file(path) if path is not None else None
        mask = load_mask_tensor(path, size=224, device='cpu').detach().numpy().reshape(224, 224) > 0
        if bool(mask.any()) != bool(row['label']):
            raise ValueError('GT/label mismatch')
        if path is not None and sha256_file(path) != before:
            raise ValueError('GT changed during annotation read')
        masks[row['path']] = mask
        sources.append({'path': row['path'], 'mask_path': path, 'mask_sha256': before})
    results, records = {}, []
    for condition, rows in by_condition.items():
        fractions, hits, called_fractions = [], [], []
        for row in rows:
            if not row['label']:
                continue
            mask = masks[row['path']]
            bounds = row['bbox_xyxy_relative']
            if bounds is None:
                coverage, pixel_count = 0.0, 0
            else:
                x1, y1, x2, y2 = bounds
                inside = ((coordinate[None, :] >= x1) & (coordinate[None, :] < x2)
                          & (coordinate[:, None] >= y1) & (coordinate[:, None] < y2))
                pixel_count = int((inside & mask).sum())
                coverage = pixel_count / int(mask.sum())
                called_fractions.append(coverage)
            fractions.append(coverage)
            hits.append(int(pixel_count > 0))
            records.append({'condition': condition, 'path': row['path'], 'tool_calls': row['tool_calls'],
                            'gt_pixels224': int(mask.sum()), 'retained_gt_pixels224': pixel_count,
                            'defect_fraction_inside_crop': coverage, 'crop_hits_any_gt': pixel_count > 0})
        results[condition] = {'n_anomaly': len(fractions), 'n_anomaly_with_crop': len(called_fractions),
                              'mean_defect_fraction_all_anomalies': mean(fractions),
                              'crop_hit_rate_all_anomalies': mean(hits),
                              'mean_defect_fraction_called_anomalies_only': mean(called_fractions)}
    return {'results': results, 'records': records, 'mask_sources': sources,
            'geometry': 'direct square224 binary GT; pixel centers in normalized original coordinates',
            'interpretation': 'post-hoc geometry; coverage does not establish defect visibility or causal language attention'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True, help='Completed tool output directory')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--with-masks', action='store_true', help='Read GT after all saved-result/RGB gates pass')
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    run_path, pred_path = args.run / 'run.json', args.run / 'predictions.jsonl'
    run = json.loads(run_path.read_text())
    rows = [json.loads(line) for line in pred_path.read_text().splitlines()]
    hashes = {'run_json': sha256_file(run_path), 'predictions_jsonl': sha256_file(pred_path)}
    by_condition = audit_records(run, rows)
    from migration_metrics import evaluate_predictions
    classification = {}
    for condition, values in by_condition.items():
        recalculated = evaluate_predictions(values, [r['score_yes_minus_no'] for r in values],
                                            [r['generated_text'] for r in values])
        if recalculated != run['results'][condition]['metrics']:
            raise ValueError('Recomputed metrics differ: ' + condition)
        classification[condition] = recalculated
    output = {'passed': True, 'schema': 'native-crop-tool-posthoc-audit-v1',
              'n': run['n'], 'conditions': run['conditions'], 'sources': hashes,
              'classification': classification, 'gt_reads_before_saved_run_and_rgb_gates': 0}
    if args.with_masks:
        output['posthoc_gt_geometry'] = posthoc_geometry(by_condition)
    if hashes['run_json'] != sha256_file(run_path) or hashes['predictions_jsonl'] != sha256_file(pred_path):
        raise ValueError('Saved inference changed during audit')
    atomic_json(output, args.out)
    print(json.dumps({'passed': True, 'n': run['n'], 'with_masks': args.with_masks, 'output': str(args.out)}))


if __name__ == '__main__':
    main()
