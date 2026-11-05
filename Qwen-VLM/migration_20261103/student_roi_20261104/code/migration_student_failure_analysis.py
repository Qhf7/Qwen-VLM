"""CPU-only, post-hoc 224 geometry diagnostic of already completed ROI runs.

Requires full val380 teacher + pre_gt + pre_kd predictions and dense maps.
GT masks are first resolved/read only after ALL saved-run, prediction, RGB,
head and map checks pass. Saved boxes are never changed or selected using GT.
No model, teacher cache, SSH, GPU, pixel-metric recomputation or training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
from migration_metrics import parse_generated

SCHEMA = 'student-roi-posthoc-224-failure-analysis-v1'
N = 380
SIZE = 224
ARMS = ('teacher', 'pre_gt', 'pre_kd')
CONDITIONS = {'teacher': 'original_teacher_roi', 'pre_gt': 'pre_gt', 'pre_kd': 'pre_kd'}
MATCHED = 'e79f550e88f58e8a51829608f716fb0049e057cea8beadbb73f6b5ccb827dacf'
TOKENS = {'image_grid_thw': [[1, 28, 28], [1, 28, 28]],
          'image_tokens_per_view': [196, 196], 'image_tokens_total': 392,
          'prompt_tokens_total': 470, 'patch_size': 16, 'spatial_merge_size': 2, 'n_images': 2}
BUCKETS = ('zero', 'partial', 'all')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    raw = Path(path).read_text().splitlines()
    need(bool(raw) and all(line.strip() for line in raw), str(path) + ': empty/blank JSONL row')
    rows = [json.loads(line) for line in raw]
    need(all(isinstance(row, dict) for row in rows), str(path) + ': rows must be objects')
    return rows


def close_list(actual, expected, tolerance=1e-12):
    return (isinstance(actual, list) and len(actual) == len(expected)
            and all(finite(a) and math.isclose(a, b, rel_tol=0, abs_tol=tolerance) for a, b in zip(actual, expected)))


def shifted_box(peak):
    row, col = peak
    center = [(col + .5) / SIZE, (row + .5) / SIZE]
    left = min(max(center[0] - 1 / 6, 0.), 1 - 1 / 3)
    top = min(max(center[1] - 1 / 6, 0.), 1 - 1 / 3)
    return center, [left, top, left + 1 / 3, top + 1 / 3]


def validate_row(row, item, arm):
    tag = arm + ': ' + item['path']
    for key in ('path', 'label', 'category', 'split', 'cache_index'):
        need(key in row and row[key] == item[key], tag + ': metadata differs: ' + key)
    need(type(row['label']) is int and type(row['cache_index']) is int, tag + ': metadata types differ')
    need(row.get('condition') == CONDITIONS[arm], tag + ': wrong condition')
    need(all(row.get(key) == value for key, value in TOKENS.items()), tag + ': two-view token budget differs')
    need(row.get('view_sources') == [item['path']] * 2
         and row.get('view_roles') == ['target_original', 'target_roi'], tag + ': wrong view sources/roles')
    need(row.get('target_path', item['path']) == item['path'] and row.get('reference') is None, tag + ': wrong target/reference')
    need(all(finite(row.get(key)) for key in ('yes_logit', 'no_logit', 'score_yes_minus_no')), tag + ': nonfinite score')
    need(row['score_yes_minus_no'] == row['yes_logit'] - row['no_logit'], tag + ': score != Yes-No')
    ids = row.get('generated_token_ids')
    need(isinstance(row.get('generated_text'), str) and isinstance(ids, list) and 1 <= len(ids) <= 32
         and all(type(x) is int and x >= 0 for x in ids), tag + ': invalid generation')
    roi = row.get('roi', {})
    need(isinstance(roi, dict), tag + ': ROI must be object')
    box = roi.get('bbox_xyxy_relative')
    need(isinstance(box, list) and len(box) == 4 and all(finite(x) for x in box), tag + ': malformed box')
    left, top, right, bottom = box
    need(0 <= left < right <= 1 and 0 <= top < bottom <= 1, tag + ': box outside RGB')
    need(roi.get('area_fraction') == 1 / 9 and math.isclose(right - left, 1 / 3, rel_tol=0, abs_tol=1e-12)
         and math.isclose(bottom - top, 1 / 3, rel_tol=0, abs_tol=1e-12), tag + ': fixed ROI area/side differs')
    wh = row.get('source_image_wh')
    need(isinstance(wh, list) and len(wh) == 2 and all(type(x) is int and x > 0 for x in wh), tag + ': bad RGB size')
    need(close_list(roi.get('bbox_xyxy_raw_pixels_float'), [left * wh[0], top * wh[1], right * wh[0], bottom * wh[1]], 1e-9), tag + ': raw/relative box differs')
    prefix = 'teacher' if arm == 'teacher' else 'student'
    need(roi.get('kind') == ('teacher' if arm == 'teacher' else 'student'), tag + ': selector kind differs')
    peak = roi.get(prefix + '_peak_row_col')
    if peak is None:
        need(arm == 'teacher', tag + ': student peak absent')
        return
    need(isinstance(peak, list) and len(peak) == 2 and all(type(x) is int and 0 <= x < SIZE for x in peak), tag + ': bad saved peak')
    need(roi.get(prefix + '_map_hw') == [SIZE, SIZE]
         and roi.get('argmax_tie_rule') == 'first in row-major order', tag + ': map/tie rule differs')
    center, expected = shifted_box(peak)
    need(close_list(roi.get('requested_center_xy_relative'), center) and close_list(box, expected), tag + ': saved peak/center/fixed box differ')
    need(finite(roi.get(prefix + '_peak_value')), tag + ': bad peak value')


def validate_result(result, n, tag):
    need(isinstance(result, dict) and result.get('n') == n, tag + ': full result denominator absent')
    need(result.get('visual_tokens_total_min_max') == [392, 392]
         and result.get('prompt_tokens_total_min_max') == [470, 470]
         and result.get('actual_image_grids') == [[1, 28, 28]], tag + ': result token/grid summary differs')


def validate_inputs(maps_path, teacher_path, student_dir):
    """Complete all pre-GT gates; no mask resolution or mask reads here."""
    maps_path, teacher_path, student_dir = Path(maps_path).resolve(), Path(teacher_path).resolve(), Path(student_dir).resolve()
    paths = {'maps_manifest': maps_path, 'teacher_predictions': teacher_path,
             'teacher_run': teacher_path.parent / 'run.json', 'student_run': student_dir / 'run.json',
             'pre_gt_predictions': student_dir / 'pre_gt_predictions.jsonl',
             'pre_kd_predictions': student_dir / 'pre_kd_predictions.jsonl'}
    sources = {name: {'path': str(path), 'sha256': sha_file(path)} for name, path in paths.items()}
    manifest, teacher_run, run = (read_json(paths[x]) for x in ('maps_manifest', 'teacher_run', 'student_run'))
    need(manifest.get('schema_version') == 'qwen-dense-map-export-v1' and manifest.get('status') == 'complete', 'Map export is not complete')
    payload = dict(manifest)
    need(payload.pop('payload_sha256', None) == canonical(payload), 'Map manifest payload hash mismatch')
    items = manifest.get('items')
    need(isinstance(items, list) and len(items) == N, 'Exactly full val380 required')
    order = []
    for item in items:
        need(isinstance(item, dict) and isinstance(item.get('path'), str) and Path(item['path']).is_absolute(), 'Bad absolute item path')
        need(isinstance(item.get('category'), str) and item['category'] and item.get('split') == 'val'
             and type(item.get('label')) is int and item['label'] in (0, 1)
             and type(item.get('cache_index')) is int and item['cache_index'] >= 0, 'Invalid val item metadata')
        order.append(item['path'])
    need(len(set(order)) == N and len({str(Path(p).resolve()) for p in order}) == N, 'Duplicate item path/resolved path')
    need(len({x['cache_index'] for x in items}) == N, 'Duplicate cache index')
    need(sum(x['label'] for x in items) == 208 and len({x['category'] for x in items}) == 12, 'Expected full 172 normal/208 anomaly/12 categories')
    need(manifest.get('n_images') == N and manifest.get('items_sha256') == canonical(items), 'Map item hash/count differs')
    geom = manifest.get('geometry', {})
    need(geom.get('coordinate_system') == 'direct-square-224' and geom.get('map_hw') == [224, 224]
         and geom.get('requested_image_hw') == [448, 448] and geom.get('processor_do_resize') is False, 'Map coordinate protocol differs')
    need(all(manifest.get(k) is True for k in ('no_teacher', 'no_imagebind', 'no_gt_masks'))
         and all(manifest.get(k) is False for k in ('test_set_used', 'train_fit', 'language_model_instantiated')), 'Map inference scope flags differ')
    need(run.get('status') == teacher_run.get('status') == 'complete', 'Both saved ROI runs must be complete before GT reads')
    need(run.get('pipeline_version') == 'qwen3vl-student-map-native-roi-v1'
         and teacher_run.get('pipeline_version') == 'qwen3vl-native-roi-val-v1', 'Unexpected saved ROI pipeline')
    need(run.get('n') == teacher_run.get('n') == teacher_run.get('full_validation_n') == N
         and run.get('split') == teacher_run.get('split') == 'val', 'Saved run not full val380')
    need(run.get('selected_paths') == teacher_run.get('selected_paths') == order
         and teacher_run.get('selected_cache_indices') == [x['cache_index'] for x in items], 'Saved run path/cache order differs')
    need(run.get('items_sha256') == canonical(items) and set(run.get('conditions', [])) == {'pre_gt', 'pre_kd'}, 'Student manifest/conditions differ')
    legacy_items = [{k: x[k] for k in ('path', 'category', 'label', 'split')} for x in items]
    legacy_sha = hashlib.sha256(json.dumps(legacy_items, sort_keys=True).encode()).hexdigest()
    need(teacher_run.get('selected_manifest_sha256') == legacy_sha, 'Teacher item metadata hash differs')
    need(run.get('map_manifest_sha256') == sources['maps_manifest']['sha256']
         and run.get('control_predictions_sha256') == sources['teacher_predictions']['sha256']
         and run.get('control_run_sha256') == sources['teacher_run']['sha256'], 'Saved source file identity differs')
    need(all(run.get(k) is False for k in ('teacher_model_used', 'teacher_cache_read', 'imagebind_used', 'gt_masks_read', 'train_fit', 'test_set_used')), 'Student inference scope flags differ')
    cfg = teacher_run.get('config', {})
    need(run.get('question') == teacher_run.get('question') and isinstance(run.get('question'), str) and run['question'], 'Question differs/absent')
    need(run.get('image_size') == cfg.get('image_size') == 448 and run.get('roi_area') == cfg.get('roi_area') == 1 / 9
         and run.get('max_new_tokens') == cfg.get('max_new_tokens') == 32
         and run.get('config', {}).get('model') == cfg.get('model') and cfg.get('seed') == 42, 'Native ROI generation protocol differs')
    rows = {arm: read_rows(paths['teacher_predictions' if arm == 'teacher' else arm + '_predictions']) for arm in ARMS}
    for arm in ARMS:
        rs = rows[arm]
        need(len(rs) == N and len({r.get('path') for r in rs}) == N, arm + ': missing/duplicate full prediction rows')
        need([r.get('path') for r in rs] == order, arm + ': prediction path order differs')
        for row, item in zip(rs, items):
            validate_row(row, item, arm)
        result = (teacher_run if arm == 'teacher' else run).get('results', {}).get(CONDITIONS[arm])
        validate_result(result, N, arm)
    for index, item in enumerate(items):
        sizes = [rows[arm][index]['source_image_wh'] for arm in ARMS]
        need(sizes[0] == sizes[1] == sizes[2], item['path'] + ': RGB sizes differ across arms')
    heads = manifest.get('heads', {})
    need(set(heads) == {'pre_gt', 'pre_kd'} and heads['pre_gt'].get('compatibility') == heads['pre_kd'].get('compatibility'), 'Head pairing differs')
    input_hashes = manifest.get('input_sha256', {})
    need(set(input_hashes) == set(order) and all(is_sha(v) for v in input_hashes.values()), 'Input hash coverage differs')
    # Verify RGB and selected checkpoint bytes without loading a model/checkpoint.
    for item in items:
        need(sha_file(item['path']) == input_hashes[item['path']], 'Original RGB bytes changed: ' + item['path'])
    maps_verified = {}
    for name, dirname, weight in [('pre_gt', 'dense_pre_gt_s42', 0.), ('pre_kd', 'dense_pre_kd_s42', .5)]:
        head = heads[name]
        checkpoint = Path(head.get('checkpoint', ''))
        need(checkpoint.is_absolute() and checkpoint.name == 'best.pt' and checkpoint.parent.name == dirname
             and head.get('features') == 'pre' and head.get('size') == 448 and head.get('step') == 1200
             and head.get('distill_weight') == weight and head.get('matched_run_key') == MATCHED, name + ': selected head identity differs')
        need(is_sha(head.get('sha256')) and is_sha(head.get('state_sha256'))
             and sha_file(checkpoint) == head['sha256'], name + ': selected head bytes hash differs')
        compatibility = head.get('compatibility', {})
        need(compatibility.get('features') == 'pre' and compatibility.get('size') == 448
             and is_sha(compatibility.get('source_sha256')), name + ': feature compatibility differs')
        need(set(head.get('maps', {})) == set(order), name + ': map coverage differs')
        audit = run.get('map_audits', {}).get(name, {})
        need(audit.get('maps_verified') == N and audit.get('checkpoint_sha256') == head['sha256']
             and all(audit.get(k) == head.get(k) for k in ('features', 'size', 'step', 'distill_weight')), name + ': saved map consumer audit differs')
        seen_map_paths = set()
        for index, item in enumerate(items):
            record = head['maps'][item['path']]
            relative = Path(record.get('path', ''))
            source = (maps_path.parent / relative).resolve()
            need(not relative.is_absolute() and maps_path.parent in source.parents and source not in seen_map_paths, name + ': unsafe/duplicate map file')
            seen_map_paths.add(source)
            need(all(is_sha(record.get(k)) for k in ('sha256', 'array_sha256', 'input_sha256', 'head_sha256', 'source_sha256')), name + ': map hash absent')
            need(sha_file(source) == record['sha256'] and source.stat().st_size == record.get('bytes'), name + ': map file hash/size differs')
            array = np.load(source, allow_pickle=False)
            need(array.dtype == np.float32 and array.shape == (SIZE, SIZE) and np.isfinite(array).all()
                 and 0 <= float(array.min()) <= float(array.max()) <= 1, name + ': invalid probability array')
            need(hashlib.sha256(array.tobytes(order='C')).hexdigest() == record['array_sha256'], name + ': map array hash differs')
            need(record.get('shape') == [SIZE, SIZE] and record.get('dtype') == 'float32'
                 and record.get('min') == float(array.min()) and record.get('max') == float(array.max()), name + ': array metadata differs')
            need(record['input_sha256'] == input_hashes[item['path']] and record['head_sha256'] == head['sha256']
                 and record['source_sha256'] == compatibility['source_sha256'], name + ': map input/head/source differs')
            g = record.get('geometry', {})
            need(g.get('requested_image_hw') == g.get('actual_image_hw') == [448, 448]
                 and g.get('processor_do_resize') is False and g.get('image_grid_thw') == [1, 28, 28]
                 and g.get('pre_grid_shape') == [1, 28, 28, 1152]
                 and g.get('geometry_policy') == compatibility.get('geometry_policy'), name + ': actual dense geometry differs')
            peak = [int(x) for x in np.unravel_index(int(array.argmax()), array.shape)]
            roi = rows[name][index]['roi']
            need(roi['student_peak_row_col'] == peak and roi['student_peak_value'] == float(array[tuple(peak)]), name + ': saved ROI not from verified map argmax')
        maps_verified[name] = {'n': N, 'head_sha256': head['sha256'], 'state_sha256_recorded': head['state_sha256'],
                               'features': head['features'], 'size': head['size'], 'step': head['step'], 'distill_weight': weight}
    for name, path in paths.items():
        need(sha_file(path) == sources[name]['sha256'], 'Source changed while validating: ' + name)
    return items, rows, {'sources': sources, 'maps_verified': maps_verified, 'items_sha256': canonical(items),
                         'ordered_paths_sha256': canonical(order), 'all_pre_gt_checks_passed': True,
                         'input_rgb_hashes_verified': N, 'head_file_hashes_verified': 2,
                         'gt_mask_reads_before_validation_complete': 0,
                         'saved_roi_spec_sha256': {arm: canonical([r['roi'] for r in rows[arm]]) for arm in ARMS}}


def geometry_diagnostic(truth, box, peak):
    """Use frozen predicted bbox; no adjustment, new argmax or GT crop."""
    foreground = int(truth.sum())
    axis = (np.arange(SIZE) + .5) / SIZE
    left, top, right, bottom = box
    chosen = ((axis[:, None] >= top) & (axis[:, None] < bottom)
              & (axis[None, :] >= left) & (axis[None, :] < right))
    retained = int(truth[chosen].sum())
    return {'gt_pixels_in_fixed_bbox_224': retained, 'bbox_selected_pixels_224': int(chosen.sum()),
            'gt_retained_fraction_224': retained / foreground if foreground else None,
            'crop_contains_any_gt_224': retained > 0 if foreground else None,
            'coverage_bucket_224': ('zero' if retained == 0 else 'all' if retained == foreground else 'partial') if foreground else None,
            'peak_hit_224': bool(truth[tuple(peak)]) if peak is not None else None}


def analyze(items, predictions, mask_loader, mask_resolver):
    """Called only after validate_inputs completes; injectable CPU loaders for mocks."""
    output = []
    frozen = {arm: canonical([r['roi'] for r in predictions[arm]]) for arm in ARMS}
    mask_sources = []
    for index, item in enumerate(items):
        mask_path = mask_resolver(item['path'], item['label'])
        before = sha_file(mask_path) if mask_path is not None else None
        tensor = mask_loader(mask_path, size=SIZE, device='cpu')
        need(str(tensor.device) == 'cpu' and tuple(tensor.shape) == (1, SIZE, SIZE), 'GT loader must return CPU [1,224,224]')
        truth = tensor[0].numpy()
        need(np.isfinite(truth).all() and np.isin(truth, (0, 1)).all(), 'GT must be finite binary224')
        truth = truth.astype(bool, copy=False)
        gt_n = int(truth.sum())
        need((item['label'] == 0 and gt_n == 0 and mask_path is None) or (item['label'] == 1 and gt_n > 0 and mask_path is not None), 'GT224 empty/label mismatch: ' + item['path'])
        if mask_path is not None:
            need(sha_file(mask_path) == before, 'Mask changed while reading: ' + str(mask_path))
        mask_sources.append({'path': item['path'], 'mask_path': mask_path, 'mask_sha256': before})
        result = {**{k: item[k] for k in ('path', 'category', 'label', 'split', 'cache_index')},
                  'sequence': index, 'gt_foreground_pixels_224': gt_n, 'gt_foreground_fraction_224': gt_n / (SIZE * SIZE),
                  'mask_path': mask_path, 'mask_sha256': before, 'arms': {}}
        for arm in ARMS:
            row = predictions[arm][index]
            roi = row['roi']
            prefix = 'teacher' if arm == 'teacher' else 'student'
            decision = parse_generated(row['generated_text'])
            score_decision = int(row['score_yes_minus_no'] > 0)
            result['arms'][arm] = {'condition': row['condition'], 'generated_text': row['generated_text'],
                'generated_token_ids': list(row['generated_token_ids']), 'generated_decision': decision,
                'score_yes_minus_no': row['score_yes_minus_no'], 'yes_logit': row['yes_logit'], 'no_logit': row['no_logit'],
                'score_decision_gt_zero': score_decision, 'generated_score_disagree': decision != score_decision,
                'invalid_generated': decision is None,
                'fn': item['label'] == 1 and decision != 1, 'fp': item['label'] == 0 and decision == 1,
                'score_fn': item['label'] == 1 and score_decision != 1, 'score_fp': item['label'] == 0 and score_decision == 1,
                'bbox_xyxy_relative': list(roi['bbox_xyxy_relative']),
                'bbox_xyxy_raw_pixels_float': list(roi['bbox_xyxy_raw_pixels_float']),
                'source_image_wh': list(row['source_image_wh']), 'saved_peak_row_col': roi.get(prefix + '_peak_row_col'),
                **geometry_diagnostic(truth, roi['bbox_xyxy_relative'], roi.get(prefix + '_peak_row_col'))}
        result['common_fn_three_arms'] = all(result['arms'][arm]['fn'] for arm in ARMS)
        result['pre_gt_pre_kd_decision_differs'] = result['arms']['pre_gt']['generated_decision'] != result['arms']['pre_kd']['generated_decision']
        output.append(result)
    need(all(frozen[arm] == canonical([r['roi'] for r in predictions[arm]]) for arm in ARMS), 'Saved ROI changed during GT diagnostic')
    return output, mask_sources


def summarize(rows, audit):
    result = {'schema_version': SCHEMA, 'status': 'complete', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'script_sha256': sha_file(__file__), 'n_images': len(rows), 'normal_n': sum(r['label'] == 0 for r in rows),
              'anomaly_n': sum(r['label'] == 1 for r in rows), 'categories_n': len({r['category'] for r in rows}),
              'scope': '后验224几何诊断：只使用已保存预测框及binary GT224像素中心；不改变框，不重算像素指标；anyGT不证明缺陷可见或语言模型关注。',
              'test_set_used': False, 'gpu_used': False, 'model_loaded': False, 'teacher_cache_read': False,
              'gt_used_for_inference_or_crop_selection': False, 'saved_bboxes_unchanged_after_gt': True,
              'coverage_definition': 'left <= (col+.5)/224 < right and top <= (row+.5)/224 < bottom; retained GT count / total GT224 count',
              'coverage_bucket_definition': {'zero': '0 foreground pixels retained', 'partial': '0 < retained < total foreground pixels', 'all': 'all foreground pixels retained'},
              'decision_definition': 'migration_metrics.parse_generated; invalid is distinct and anomaly invalid counts FN; score decision is strictly Yes-No > 0',
              'audit': audit, 'arms': {}, 'by_category': {}, 'generated_fn_intersections': {}, 'gt_vs_kd_decisions': {}}
    for category in [None] + sorted({r['category'] for r in rows}):
        subset = rows if category is None else [r for r in rows if r['category'] == category]
        arm_summary = {}
        for arm in ARMS:
            confusion = Counter(tp=0, fn=0, fp=0, tn=0, invalid_normal=0, invalid_anomaly=0)
            coverage = {bucket: {'anomaly_n': 0, 'tp': 0, 'fn': 0} for bucket in BUCKETS}
            fractions, peaks = [], []
            for row in subset:
                r = row['arms'][arm]
                decision = r['generated_decision']
                if decision is None:
                    confusion['invalid_anomaly' if row['label'] else 'invalid_normal'] += 1
                if row['label']:
                    confusion['tp' if decision == 1 else 'fn'] += 1
                    bucket = coverage[r['coverage_bucket_224']]
                    bucket['anomaly_n'] += 1
                    bucket['tp' if decision == 1 else 'fn'] += 1
                    fractions.append(r['gt_retained_fraction_224'])
                    if r['peak_hit_224'] is not None:
                        peaks.append(r['peak_hit_224'])
                elif decision == 1:
                    confusion['fp'] += 1
                elif decision == 0:
                    confusion['tn'] += 1
            arm_summary[arm] = {'generated_counts': dict(confusion), 'coverage_by_generated_fn_224': coverage,
                                'fn_by_coverage_224': {b: coverage[b]['fn'] for b in BUCKETS},
                                'crop_contains_any_gt_anomaly_n': sum(coverage[b]['anomaly_n'] for b in ('partial', 'all')),
                                'mean_gt_retained_fraction_anomaly_224': float(np.mean(fractions)) if fractions else None,
                                'peak_hit_anomaly_n': sum(peaks) if peaks else None, 'peak_available_anomaly_n': len(peaks),
                                'generated_score_disagreement_n': sum(r['arms'][arm]['generated_score_disagree'] for r in subset),
                                'score_fn_n': sum(r['arms'][arm]['score_fn'] for r in subset),
                                'score_fp_n': sum(r['arms'][arm]['score_fp'] for r in subset)}
            need(sum(arm_summary[arm]['fn_by_coverage_224'].values()) == confusion['fn'], 'FN coverage decomposition does not sum')
        if category is None:
            result['arms'] = arm_summary
        else:
            result['by_category'][category] = {'n': len(subset), 'normal_n': sum(r['label'] == 0 for r in subset),
                                               'anomaly_n': sum(r['label'] == 1 for r in subset), 'arms': arm_summary,
                                               'common_fn_three_arms_n': sum(r['common_fn_three_arms'] for r in subset)}
    anomalies = [r for r in rows if r['label'] == 1]
    groups = [('teacher_and_pre_gt', ('teacher', 'pre_gt')), ('teacher_and_pre_kd', ('teacher', 'pre_kd')),
              ('pre_gt_and_pre_kd', ('pre_gt', 'pre_kd')), ('all_three', ARMS)]
    for key, arms in groups:
        matches = [r['path'] for r in anomalies if all(r['arms'][a]['fn'] for a in arms)]
        result['generated_fn_intersections'][key] = {'n': len(matches), 'paths': matches}
    exact = Counter(''.join('1' if r['arms'][a]['fn'] else '0' for a in ARMS) for r in anomalies)
    result['generated_fn_intersections']['exact_membership_order'] = list(ARMS)
    result['generated_fn_intersections']['exact_membership_counts'] = {format(i, '03b'): exact[format(i, '03b')] for i in range(8)}
    for label, name in [(0, 'normal'), (1, 'anomaly')]:
        subset = [r for r in rows if r['label'] == label]
        transitions = Counter()
        changed = []
        for r in subset:
            a, b = (r['arms'][x]['generated_decision'] for x in ('pre_gt', 'pre_kd'))
            key = ('invalid' if a is None else str(a)) + '->' + ('invalid' if b is None else str(b))
            transitions[key] += 1
            if a != b:
                changed.append(r['path'])
        result['gt_vs_kd_decisions'][name] = {'n': len(subset), 'different_n': len(changed), 'transition_counts': dict(sorted(transitions.items())), 'different_paths': changed}
    result['gt_vs_kd_decisions']['total_different_n'] = sum(r['pre_gt_pre_kd_decision_differs'] for r in rows)
    result['gt_vs_kd_decisions']['interpretation'] = '相同TP/FP/F1总计不能推出逐图决策相同。'
    result['limitations'] = ['单seed、同一dev/val380的后验诊断；不构成独立test、因果验证、等价或非劣结论。',
        '224二值mask与归一化bbox的像素中心几何仅近似原图缺陷保留；不度量缩放后的细节可辨识性、遮挡、上下文或语言注意力。',
        '包含部分或全部GT不证明模型看到或理解缺陷；零覆盖也只描述ROI，原始整图仍作为第一视图输入。',
        'teacher peak仅来自已保存ROI记录；未加载teacher map/cache或验证其历史argmax。学生peak来自已校验map数组。',
        'head仅重新核对checkpoint文件SHA；state_sha256保留manifest记录，不重新反序列化checkpoint。']
    return result


def report_markdown(summary):
    lines = ['# Student ROI 后验224几何失败诊断', '', summary['scope'], '',
             f"验证集 {summary['n_images']}：normal {summary['normal_n']} / anomaly {summary['anomaly_n']} / 类别 {summary['categories_n']}。三臂full run、预测、token预算、head/RGB/map/array哈希全部先校验，随后读取GT；保存框未改动。", '',
             '生成答案按初始Yes/No解析；无效答案单独记录，异常图无效计FN。score使用严格 >0。正常图coverage比例为null；peak可用时在零GT上为false，否则null。', '',
             '| Arm | TP | FN | FP | TN | FN零覆盖 | FN部分覆盖 | FN全部覆盖 | AnyGT /208 | Peak hit / available |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        a = summary['arms'][arm]; c = a['generated_counts']; f = a['fn_by_coverage_224']
        lines.append(f"|{arm}|{c['tp']}|{c['fn']}|{c['fp']}|{c['tn']}|{f['zero']}|{f['partial']}|{f['all']}|{a['crop_contains_any_gt_anomaly_n']}|{a['peak_hit_anomaly_n']} / {a['peak_available_anomaly_n']}|")
    lines += ['', '零/部分/全部覆盖依据GT224前景像素保留数分解，仅为后验几何；anyGT不能等同可见或语言关注。', '', '## 类别FN分解', '',
              '| 类别 | Arm | 异常数 | FN | 零覆盖FN | 部分覆盖FN | 全部覆盖FN |', '|---|---|---:|---:|---:|---:|---:|']
    for category, value in summary['by_category'].items():
        for arm in ARMS:
            a = value['arms'][arm]; f = a['fn_by_coverage_224']
            lines.append(f"|{category}|{arm}|{value['anomaly_n']}|{a['generated_counts']['fn']}|{f['zero']}|{f['partial']}|{f['all']}|")
    lines += ['', '## 三臂漏检交集', '']
    for key in ('teacher_and_pre_gt', 'teacher_and_pre_kd', 'pre_gt_and_pre_kd', 'all_three'):
        lines.append(f"- {key}: {summary['generated_fn_intersections'][key]['n']}")
    lines += ['', f"GT vs KD逐图生成决策差异：共 {summary['gt_vs_kd_decisions']['total_different_n']}；normal {summary['gt_vs_kd_decisions']['normal']['different_n']}，anomaly {summary['gt_vs_kd_decisions']['anomaly']['different_n']}。相同汇总计数不代表相同预测。", '',
              '| 标签 | GT → KD | 数量 |', '|---|---|---:|']
    for name in ('normal', 'anomaly'):
        for transition, count in summary['gt_vs_kd_decisions'][name]['transition_counts'].items():
            lines.append(f'|{name}|{transition}|{count}|')
    lines += ['', '## 解释边界', ''] + ['- ' + x for x in summary['limitations']]
    lines += ['', '逐图数据：`per_image.jsonl`、`per_image.csv`（CSV一图一行，各arm展开为列）；交集路径与完整校验来源见`summary.json`。', '']
    return '\n'.join(lines)


def csv_rows(rows):
    result = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key != 'arms'}
        for arm, value in row['arms'].items():
            flat.update({arm + '_' + key: val for key, val in value.items()})
        result.append({key: json.dumps(value, ensure_ascii=False, separators=(',', ':')) if isinstance(value, (list, dict))
                       else '' if value is None else value for key, value in flat.items()})
    return result


def write_outputs(out, rows, summary):
    """Reserve fresh directory exclusively; never overwrite prior output."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    temporary = Path(tempfile.mkdtemp(prefix='.failure-analysis-', dir=out))
    try:
        (temporary / 'per_image.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False, allow_nan=False) + '\n' for x in rows))
        flat = csv_rows(rows)
        with (temporary / 'per_image.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader(); writer.writerows(flat)
        (temporary / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        (temporary / 'REPORT.md').write_text(report_markdown(summary))
        for filename in ('per_image.jsonl', 'per_image.csv', 'REPORT.md', 'summary.json'):
            os.link(temporary / filename, out / filename)
    except BaseException:
        # Preserve any published output for inspection; it can never be reused.
        raise
    finally:
        shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps', required=True, help='Complete dense export manifest.json')
    parser.add_argument('--teacher-predictions', required=True, help='Old full380 original_teacher_roi_predictions.jsonl; sibling run.json required')
    parser.add_argument('--student-dir', required=True, help='Completed student roi_full directory')
    parser.add_argument('--out', required=True, help='New output directory; existing paths are rejected')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError('Existing diagnostic output is preserved: ' + str(out))
    items, predictions, audit = validate_inputs(args.maps, args.teacher_predictions, args.student_dir)
    # This import and all GT reads deliberately occur after every complete-file gate.
    from visa_pipeline import load_mask_tensor, mask_path_for_image
    audit['diagnostic_dependency_sha256'] = {
        'visa_pipeline.py': sha_file(Path(load_mask_tensor.__code__.co_filename)),
        'migration_metrics.py': sha_file(Path(parse_generated.__code__.co_filename))}
    rows, mask_sources = analyze(items, predictions, load_mask_tensor, mask_path_for_image)
    audit['mask_sources_sha256'] = canonical(mask_sources)
    audit['gt_masks_read_after_all_checks'] = len([x for x in mask_sources if x['mask_path'] is not None])
    audit['normal_zero_tensors_loaded'] = len([x for x in mask_sources if x['mask_path'] is None])
    summary = summarize(rows, audit)
    write_outputs(out, rows, summary)
    print(json.dumps({'status': 'complete', 'out': str(out), 'n': len(rows),
                      'fn': {a: summary['arms'][a]['generated_counts']['fn'] for a in ARMS},
                      'common_fn': summary['generated_fn_intersections']['all_three']['n'],
                      'gt_vs_kd_decision_differs': summary['gt_vs_kd_decisions']['total_different_n']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
