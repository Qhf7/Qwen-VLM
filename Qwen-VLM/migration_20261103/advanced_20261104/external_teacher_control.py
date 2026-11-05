"""Post-hoc fixed teacher-box controls after the complete frozen external audit.

Reads only saved metadata, predictions, posthoc targets, and the CPU tensor
cache. It never opens an RGB/mask, imports a model, or tunes a parameter. The
imagewise >=95th-percentile union-box rule and score>0 gate are copied from
the existing S validation control. External controls cannot select a model.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile

import numpy as np

from structured_geometry import (SCHEMA, localization_metrics,
                                 mask_to_target, parse_structured, target_answer)

HERE = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)
MODELS = ('A', 'C', 'D', 'Cpair', 'R', 'Cstructured', 'S')
PIPELINE = 'visa-square-binary-sklearn-v1'
RULE = 'union box of teacher map >= its imagewise 95th percentile; no GT/calibration; box on every image'
GATE = 'saved Cstructured score > 0; equality at zero produces No and no box'
METRICS = ('anomaly_bbox_iou_mean_all', 'anomaly_iou_at_05_all',
           'anomaly_box_center_inside_gt_box_all', 'anomaly_grid3_accuracy_all',
           'anomaly_bbox_iou_mean_given_valid_positive', 'anomaly_iou_at_05_given_valid_positive')


@dataclass(frozen=True)
class Contract:
    manifest_sha256: str = '4cf2709614c8fd6aabe1268ef3c9f8687a07a0849d3d0f0341d1756570f9497a'
    n: int = 1575
    normal: int = 392
    anomaly: int = 1183
    categories: int = 15


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def read(path):
    return json.loads(Path(path).read_text())


def checked_read(path, digest, jsonl=False):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, 'Committed bytes changed: ' + str(path))
    return ([json.loads(line) for line in raw.splitlines() if line.strip()] if jsonl
            else json.loads(raw))


def compare(actual, expected, location='metrics'):
    if isinstance(actual, dict):
        require(isinstance(expected, dict) and set(actual) == set(expected), 'Keys differ: ' + location)
        for key in actual:
            compare(actual[key], expected[key], location + '.' + key)
    elif type(actual) is float:
        require(type(expected) in (float, int) and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), 'Value differs: ' + location)
    else:
        require(type(actual) is type(expected) and actual == expected, 'Value/type differs: ' + location)


def summary_gate(summary, contract):
    """This gate runs before opening any saved prediction, target or tensor."""
    require(summary.get('state') == 'complete' and summary.get('expected_runs') == 21
            and summary.get('completed_recomputed_runs') == 21 and summary.get('unavailable') == [],
            'Requires external_summary complete with all21 recomputed runs')
    require(summary.get('requested_models') == list(MODELS) and summary.get('requested_seeds') == list(SEEDS)
            and summary.get('manifest_sha256') == contract.manifest_sha256
            and summary.get('n') == contract.n and summary.get('no_model_or_image_or_mask_queries') is True,
            'External summary model/seed/protocol identity differs')
    rows = summary.get('rows', [])
    lookup = {(r['model'], r['seed']): r for r in rows}
    require(len(rows) == len(lookup) == 21 and set(lookup) == {(m, s) for m in MODELS for s in SEEDS},
            'External summary has missing/duplicate model-seed rows')
    for row in rows:
        require(row.get('state') == 'complete_and_recomputed' and row.get('n') == contract.n
                and row.get('n_normal') == contract.normal and row.get('n_anomaly') == contract.anomaly
                and row.get('category_count') == contract.categories
                and row.get('manifest_sha256') == contract.manifest_sha256,
                'An external summary row is incomplete or has a different population')
    require(len({r['external_cache_sha256'] for r in rows}) == 1, 'External summary mixes tensor caches')
    return lookup


def manifest_gate(records, contract):
    require(len(records) == contract.n and len({r['source_id'] for r in records}) == contract.n
            and len({r['path'] for r in records}) == contract.n
            and sum(r['label'] for r in records) == contract.anomaly
            and len({r['category'] for r in records}) == contract.categories,
            'Manifest population differs')
    for record in records:
        require(type(record['label']) is int and record['label'] in (0, 1)
                and record['source_id'] == 'mvtec_' + record['image_sha256']
                and type(record['width']) is int and record['width'] > 0
                and type(record['height']) is int and record['height'] > 0,
                'Invalid manifest metadata')
        path = Path(record['path'])
        require(not path.is_absolute() and '..' not in path.parts, 'Invalid relative manifest path')


def bundle_metadata(root, model, seed, row, freeze, cache_sha, protocol_sha, contract, snapshots):
    """Validate all completion/identity metadata before any raw output is read."""
    run = root / ('arm_S_s%d' % seed)
    output = run / ('external_mvtec_residual_v1_initial_c' if model == 'Cstructured' else 'external_mvtec_residual_v1')
    paths = {name: output / name for name in ('summary.json', 'frozen_config.json', 'inference_complete.json',
                                            'predictions.jsonl', 'predictions.sha256', 'posthoc_targets.json')}
    for path in paths.values():
        require(path.is_file(), 'Missing completed structured artifact: ' + str(path))
    summary_digest = sha256(paths['summary.json'])
    require(summary_digest == row['summary_sha256'], 'Structured summary differs from complete21 report')
    summary = read(paths['summary.json'])
    config = read(paths['frozen_config.json'])
    marker = read(paths['inference_complete.json'])
    expected = freeze['models'][model + '_s%d' % seed]
    require(summary.get('state') == 'complete' and summary.get('n') == contract.n
            and summary.get('seed') == seed and summary.get('schema') == SCHEMA
            and summary.get('condition') == ('initial_C_structured_prompt' if model == 'Cstructured' else 'S_best')
            and summary.get('external_queries') == contract.n and summary.get('historical_visa_test_queries') == 0
            and summary.get('mask_reads_during_inference') == 0 and summary.get('posthoc_mask_reads') == contract.anomaly,
            'Structured summary is not fully completed under the frozen protocol')
    identity = dict(manifest_sha256=contract.manifest_sha256, protocol_sha256=protocol_sha,
                    decoder_sha256=freeze['decoder_sha256'], external_cache_sha256=cache_sha,
                    checkpoint_sha256=expected['checkpoint_sha256'], checkpoint_step=expected['checkpoint_step'],
                    structured_run_sha256=expected['run_sha256'], base_manifest_sha256=expected['base_manifest_sha256'],
                    max_new_tokens=80, batch_size=4, schema=SCHEMA, seed=seed,
                    evaluator_sha256=freeze['source_sha256']['structured_external.py'],
                    model_source_sha256=freeze['source_sha256']['structured_model.py'], mask_reads_at_start=0)
    for key, value in identity.items():
        require(summary.get(key) == value and config.get(key) == value, 'Structured frozen identity differs: ' + key)
    for key, value in config.items():
        require(summary.get(key) == value, 'Summary/config metadata differs: ' + key)
    require(marker.get('state') == 'inference_complete' and marker.get('processed') == contract.n
            and marker.get('mask_reads') == 0 and marker.get('frozen_config_sha256') == sha256(paths['frozen_config.json']),
            'Missing inference-before-GT completion proof')
    digest = summary.get('predictions_sha256')
    require(isinstance(digest, str) and len(digest) == 64 and digest == row['predictions_sha256']
            and marker.get('predictions_sha256') == digest
            and paths['predictions.sha256'].read_text().split() == [digest, 'predictions.jsonl'],
            'Structured prediction SHA declarations differ')
    target_digest = summary.get('posthoc_targets_sha256')
    require(isinstance(target_digest, str) and len(target_digest) == 64, 'Missing posterior target SHA')
    for filename, key in (('run.json', 'run_sha256'), ('complete.json', 'complete_sha256')):
        source = run / filename
        require(source.is_file() and sha256(source) == expected[key], 'Frozen training metadata changed: ' + filename)
        snapshots[source] = expected[key]
    policy_name = 'initial_generation_policy.json' if model == 'Cstructured' else 'best_generation_policy.json'
    policy_path = run / policy_name
    policy_digest = freeze['structured_generation_policy_sha256'][str(Path(expected['run']) / policy_name)]
    require(sha256(policy_path) == policy_digest == summary.get('generation_policy_sha256'),
            'Generation policy differs from the pre-external freeze')
    for name in ('summary.json', 'frozen_config.json', 'inference_complete.json', 'predictions.sha256'):
        snapshots[paths[name]] = sha256(paths[name])
    snapshots[policy_path] = policy_digest
    return dict(model=model, seed=seed, output=output, summary=summary, paths=paths,
                predictions_sha256=digest, targets_sha256=target_digest)


def validate_targets(targets, manifest):
    require(len(targets) == len(manifest), 'Posterior target count differs')
    for target, record in zip(targets, manifest):
        require(all(target[key] == record[key] for key in ('source_id', 'category', 'label'))
                and type(target['label']) is int, 'Posterior targets are not aligned to the frozen manifest')
        parsed = parse_structured(target_answer(target))
        require(parsed is not None and all(parsed[key] == target[key] for key in ('label', 'bbox', 'location')),
                'Saved posterior target has invalid box/grid geometry')
        require(type(target['foreground_pixels']) is int
                and (target['foreground_pixels'] > 0 if target['label'] else target['foreground_pixels'] == 0),
                'Posterior target foreground denominator differs')
        if target['label']:
            require(target.get('mask_width') == record['width'] and target.get('mask_height') == record['height'],
                    'Posterior target geometry differs from manifest')


def validate_predictions(predictions, manifest):
    require(len(predictions) == len(manifest), 'Prediction count differs')
    for prediction, record in zip(predictions, manifest):
        require(all(prediction[key] == record[key] for key in ('source_id', 'path', 'category', 'label'))
                and prediction.get('split') == 'external', 'Prediction population/order differs')
        require(type(prediction['score']) in (int, float) and math.isfinite(prediction['score'])
                and isinstance(prediction['generated'], str), 'Invalid saved score/text')
        require(prediction.get('parsed') == parse_structured(prediction['generated']), 'Saved strict parsing differs')


def validate_cache(cache, sidecar, manifest, freeze, contract):
    import torch
    identity = dict(schema='imagebind-bigdec-external-cache-v1', pipeline_version=PIPELINE,
                    manifest_sha256=contract.manifest_sha256, decoder_sha256=freeze['decoder_sha256'],
                    geometry='direct-square224', images=contract.n, gt_masks_opened=False,
                    images_for_training=False, generation_or_qwen_queries=0)
    for metadata in (cache, sidecar):
        for key, value in identity.items():
            require(type(metadata.get(key)) is type(value) and metadata.get(key) == value,
                    'CPU cache identity differs: ' + key)
    for key, value in sidecar.items():
        if key != 'cache_sha256':
            require(canonical(cache.get(key)) == canonical(value), 'Cache/sidecar metadata differs: ' + key)
    require(len(cache['items']) == len(manifest), 'Cache item count differs')
    for item, record in zip(cache['items'], manifest):
        require(item['source_id'] == record['source_id'] and item['label'] == record['label']
                and item['category_original'] == record['category'] and item['category'] == 'industrial_component'
                and item['split'] == 'external' and Path(item['path']).as_posix().endswith('/' + record['path']),
                'CPU cache item/order differs from frozen manifest')
    for key, shape in (('cls', (contract.n, 1024)), ('maps', (contract.n, 224, 224))):
        tensor = cache[key]
        require(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu'
                and tensor.dtype == torch.float32 and tuple(tensor.shape) == shape
                and bool(torch.isfinite(tensor).all()), 'Invalid CPU cache tensor: ' + key)
    require(float(cache['maps'].min()) >= 0 and float(cache['maps'].max()) <= 1, 'Teacher maps outside [0,1]')


def teacher_target(probability):
    """Identical no-parameter rule used by structured_train.py on VisA val."""
    return mask_to_target(probability >= np.quantile(probability, .95), 1)


def metrics_package(targets, texts):
    categories = sorted({t['category'] for t in targets})
    def calculate(truth, answers):
        metric = localization_metrics(truth, answers)
        parsed = [parse_structured(text) for text in answers]
        metric['n_positive_box_on_normal'] = sum(bool(p and p['label']) for p, t in zip(parsed, truth) if t['label'] == 0)
        metric['normal_positive_box_rate'] = (metric['n_positive_box_on_normal'] / metric['n_normal']
                                               if metric['n_normal'] else None)
        metric['anomaly_positive_box_coverage'] = (metric['n_valid_positive_on_anomaly'] / metric['n_anomaly']
                                                   if metric['n_anomaly'] else None)
        return metric
    overall = calculate(targets, texts)
    per_category = {c: calculate([t for t in targets if t['category'] == c],
                                  [s for t, s in zip(targets, texts) if t['category'] == c]) for c in categories}
    category_macro = {key: statistics.mean(v[key] for v in per_category.values()) for key in METRICS
                      if all(v[key] is not None for v in per_category.values())}
    return dict(overall=overall, per_category=per_category, category_macro=category_macro)


def describe(values):
    values = list(values)
    return dict(n=len(values), mean=statistics.mean(values), sample_sd=statistics.stdev(values) if len(values) > 1 else None)


def aggregate(packages):
    result = {}
    for scope in ('overall', 'category_macro'):
        common = set.intersection(*(set(p[scope]) for p in packages))
        result[scope] = {key: describe(p[scope][key] for p in packages) for key in sorted(common)
                         if all(type(p[scope][key]) in (int, float) for p in packages)}
    result['per_category'] = {c: {key: describe(p['per_category'][c][key] for p in packages)
                                  for key in sorted(packages[0]['per_category'][c])
                                  if all(type(p['per_category'][c][key]) in (int, float) for p in packages)}
                              for c in packages[0]['per_category']}
    return result


def build_report(root, manifest_path, protocol_path, cache_path, contract=Contract()):
    summary_path = root / 'external_summary.json'
    external_summary = read(summary_path)
    lookup = summary_gate(external_summary, contract)
    # No raw prediction/target or tensor has been opened above this line.
    freeze_path = root / 'external_selection_freeze.json'
    freeze = read(freeze_path)
    require(freeze.get('state') == 'all_selection_frozen_before_external_model_queries'
            and freeze.get('manifest_sha256') == contract.manifest_sha256
            and set(freeze['models']) == {m + '_s%d' % s for m in MODELS for s in SEEDS}, 'Invalid selection freeze')
    require(sha256(manifest_path) == contract.manifest_sha256, 'Frozen manifest bytes changed')
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    manifest_gate(manifest, contract)
    protocol_sha = sha256(protocol_path)
    protocol = read(protocol_path)
    require(protocol_sha == freeze['protocol_sha256']
            and protocol.get('protocol') == 'visa_to_mvtec_residual_official_test_v1'
            and protocol.get('manifest_sha256') == contract.manifest_sha256
            and protocol.get('retained_n') == contract.n and protocol.get('n_normal') == contract.normal
            and protocol.get('n_anomaly') == contract.anomaly, 'Frozen external protocol differs')
    require(sha256(HERE / 'structured_geometry.py') == freeze['source_sha256']['structured_geometry.py'],
            'Frozen geometry implementation changed')
    sidecar_path = cache_path.with_suffix('.json')
    sidecar = read(sidecar_path)
    cache_sha = sidecar['cache_sha256']
    require({r['external_cache_sha256'] for r in lookup.values()} == {cache_sha}
            and sidecar['manifest_sha256'] == contract.manifest_sha256
            and sidecar['decoder_sha256'] == freeze['decoder_sha256'], 'Cache/decoder identity differs from freeze/report')
    snapshots = {path: sha256(path) for path in (summary_path, freeze_path, manifest_path, protocol_path, sidecar_path)}
    bundles = [bundle_metadata(root, model, seed, lookup[(model, seed)], freeze, cache_sha, protocol_sha, contract, snapshots)
               for model in ('Cstructured', 'S') for seed in SEEDS]
    # Every one of the six completion and inference-before-GT markers passed.
    # Validate every immutable raw/target checksum before parsing any of them.
    for bundle in bundles:
        for filename, digest in (('predictions.jsonl', bundle['predictions_sha256']),
                                 ('posthoc_targets.json', bundle['targets_sha256'])):
            require(sha256(bundle['paths'][filename]) == digest,
                    'Committed bytes changed: ' + str(bundle['paths'][filename]))
    targets = None
    for bundle in bundles:
        paths = bundle['paths']
        current = checked_read(paths['posthoc_targets.json'], bundle['targets_sha256'])
        validate_targets(current, manifest)
        if targets is None:
            targets = current
        require(canonical(current) == canonical(targets), 'Saved posthoc targets differ across the six structured runs')
        predictions = checked_read(paths['predictions.jsonl'], bundle['predictions_sha256'], jsonl=True)
        validate_predictions(predictions, manifest)
        bundle['predictions'] = predictions
        metrics = localization_metrics(targets, [r['generated'] for r in predictions])
        per_category = {c: localization_metrics([t for t in targets if t['category'] == c],
                        [p['generated'] for p in predictions if p['category'] == c]) for c in sorted({t['category'] for t in targets})}
        for expected in (bundle['summary']['metrics'], lookup[(bundle['model'], bundle['seed'])]['metrics']):
            compare(metrics, expected['localization'])
            compare(per_category, expected['localization_per_category'])
        snapshots[paths['predictions.jsonl']] = bundle['predictions_sha256']
        snapshots[paths['posthoc_targets.json']] = bundle['targets_sha256']
    # The tensor is read only after complete21, all6 immutable raw SHAs and
    # exactly equal saved targets passed; weights_only prevents pickle objects.
    require(sha256(cache_path) == cache_sha, 'CPU cache bytes changed')
    import torch
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    validate_cache(cache, sidecar, manifest, freeze, contract)
    teacher = [teacher_target(probability.numpy()) for probability in cache['maps']]
    baseline_texts = [target_answer(t) for t in teacher]
    normal_text = target_answer(dict(label=0, bbox=None, location='none'))
    shared = metrics_package(targets, baseline_texts)
    results = []
    for seed in SEEDS:
        c = next(b for b in bundles if b['model'] == 'Cstructured' and b['seed'] == seed)
        s = next(b for b in bundles if b['model'] == 'S' and b['seed'] == seed)
        gated = [text if r['score'] > 0 else normal_text for text, r in zip(baseline_texts, c['predictions'])]
        packages = dict(Cstructured_score_gt0_teacher_box=metrics_package(targets, gated),
                        Cstructured_generated=metrics_package(targets, [r['generated'] for r in c['predictions']]),
                        S_generated=metrics_package(targets, [r['generated'] for r in s['predictions']]))
        differences = {name: {key: packages['S_generated']['overall'][key] - reference['overall'][key] for key in METRICS
                              if packages['S_generated']['overall'][key] is not None and reference['overall'][key] is not None}
                       for name, reference in (('S_minus_unconditional_teacher_box', shared),
                                               ('S_minus_Cstructured_score_gt0_teacher_box', packages['Cstructured_score_gt0_teacher_box']))}
        results.append(dict(seed=seed, conditions=packages, descriptive_differences=differences))
    c_lookup = {b['seed']: b['predictions'] for b in bundles if b['model'] == 'Cstructured'}
    boxes = [dict(source_id=r['source_id'], category=r['category'], label=r['label'],
                  teacher_bbox=t['bbox'], teacher_location=t['location'],
                  Cstructured_score_gt0={str(seed): bool(c_lookup[seed][i]['score'] > 0) for seed in SEEDS})
             for i, (r, t) in enumerate(zip(manifest, teacher))]
    for path, expected in snapshots.items():
        require(sha256(path) == expected, 'Input artifact changed while computing controls: ' + str(path))
    require(sha256(cache_path) == cache_sha, 'Tensor cache changed during control audit')
    report = dict(state='complete', role='posthoc_external_fixed_teacher_box_control',
                  n=contract.n, n_normal=contract.normal, n_anomaly=contract.anomaly, categories=contract.categories,
                  rule=RULE, score_gate=GATE, quantile=.95, quantile_method='numpy default linear; >= includes ties',
                  schema=SCHEMA, source_rule='structured_train.py existing VisA validation teacher_map_bbox_baseline',
                  unconditional_teacher_box=shared, by_seed=results,
                  aggregate_3seed={condition: aggregate([r['conditions'][condition] for r in results])
                                   for condition in results[0]['conditions']},
                  evidence=dict(external_summary_sha256=snapshots[summary_path],
                                external_selection_freeze_sha256=snapshots[freeze_path],
                                manifest_sha256=contract.manifest_sha256, protocol_sha256=protocol_sha,
                                decoder_sha256=freeze['decoder_sha256'], cache_sha256=cache_sha,
                                geometry_source_sha256=freeze['source_sha256']['structured_geometry.py'],
                                posthoc_targets_equal_across_all_six=True,
                                structured_inputs=[dict(model=b['model'], seed=b['seed'],
                                                        summary_sha256=sha256(b['paths']['summary.json']),
                                                        predictions_sha256=b['predictions_sha256'],
                                                        posthoc_targets_sha256=b['targets_sha256']) for b in bundles],
                                source_sha256=sha256(__file__)),
                  access=dict(new_rgb_reads=0, new_mask_reads=0, new_model_queries=0, new_training=0,
                              tensor_device='cpu', all21_complete_before_raw_reads=True,
                              all6_complete_before_raw_reads=True, no_checkpoint_or_decoder_weights_opened=True),
                  limitations=[
                      'Post-hoc fixed control on external evaluation; no external model selection or threshold tuning.',
                      'No causal claim: S structured generation, deterministic teacher geometry, and C score gating are different systems.',
                      'Teacher geometry comes from the pre-existing ImageBind/BigDec maps; it is not LLM-generated localization.',
                      'Control text/schema validity and grid consistency are assigned deterministically, not learned language abilities.',
                      'Unconditional teacher emits a box on every normal image; localization on all anomalies does not establish detection quality.',
                      'All-anomaly metrics include every anomaly, including No/invalid S outputs and score-gated misses as zero; conditional metrics require coverage.',
                      'Centers are tested inside GT union boxes, not against defect mask pixels; disconnected regions are represented by a union box.',
                      'Three-seed standard deviations are descriptive; no significance or unseen-pretraining claim.'])
    return report, boxes


def render_markdown(report):
    lines = ['# External post-hoc fixed teacher-box control', '',
             'State: **complete**. All21 external runs and all6 structured completion/inference/SHA proofs passed before raw prediction, saved target or CPU cache use.', '',
             'The existing VisA validation rule is reused without fitting: map >= imagewise95th percentile, union bbox, normalized0..1000 and outward rounding. Gated controls emit this box only when the saved same-seed Cstructured score >0. No new RGB, masks or model queries.', '',
             '| Condition | Seed | All anomalies | IoU | IoU≥0.5 | Center in GT box | Grid3 | Anomaly box coverage | Normal box rate |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def row(name, seed, package):
        metric = package['overall']
        values = [metric[k] for k in ('anomaly_bbox_iou_mean_all', 'anomaly_iou_at_05_all',
                  'anomaly_box_center_inside_gt_box_all', 'anomaly_grid3_accuracy_all',
                  'anomaly_positive_box_coverage', 'normal_positive_box_rate')]
        return '| %s | %s | %d | %s |' % (name, seed, metric['n_anomaly'], ' | '.join('%.6f' % v for v in values))
    lines.append(row('Unconditional teacher box (shared)', '—', report['unconditional_teacher_box']))
    for result in report['by_seed']:
        for condition, package in result['conditions'].items():
            lines.append(row(condition, result['seed'], package))
    lines += ['', 'Complete per-category denominators/results, three-seed descriptive means/SD, and same-seed differences are in the JSON report. The unconditional control is one shared deterministic result, not three independent seeds.', '',
              'This is an external post-hoc fixed control. It does not identify a causal LoRA, language reasoning or mask supervision effect and cannot select a model. Teacher box/grid validity is assigned by code. Unconditional boxes on normal images do not demonstrate useful detection; center-in-GT-box is not pixel-mask localization.']
    return '\n'.join(lines) + '\n'


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def self_test():
    """Small synthetic CPU fixtures; never touches actual external outputs."""
    import torch
    checks = []
    with tempfile.TemporaryDirectory(prefix='external-teacher-control-') as folder:
        root = Path(folder)
        manifest = [dict(source_id='mvtec_' + str(i) * 64, image_sha256=str(i) * 64,
                         path='category%d/test/%d.png' % (i // 2, i), category='category%d' % (i // 2),
                         label=i % 2, width=224, height=224) for i in range(4)]
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
            return sha256(path)
        manifest_path = root / 'external_manifest.jsonl'
        manifest_path.write_text(''.join(json.dumps(r) + '\n' for r in manifest))
        contract = Contract(sha256(manifest_path), 4, 2, 2, 2)
        protocol = dict(protocol='visa_to_mvtec_residual_official_test_v1', manifest_sha256=contract.manifest_sha256,
                        retained_n=4, n_normal=2, n_anomaly=2)
        protocol_path = root / 'external_protocol.json'
        protocol_sha = write(protocol_path, protocol)
        metadata = dict(schema='imagebind-bigdec-external-cache-v1', pipeline_version=PIPELINE,
                        manifest_sha256=contract.manifest_sha256, decoder_sha256='d' * 64,
                        geometry='direct-square224', images=4, gt_masks_opened=False,
                        images_for_training=False, generation_or_qwen_queries=0)
        cache_path = root / 'external_teacher_cache.pt'
        maps = torch.zeros((4, 224, 224)); maps[:, 112:, 112:] = 1
        cache = dict(**metadata, items=[dict(path='/never/open/' + r['path'], category='industrial_component',
                     label=r['label'], split='external', source_id=r['source_id'], category_original=r['category']) for r in manifest],
                     cls=torch.zeros(4, 1024), maps=maps)
        torch.save(cache, cache_path)
        cache_sha = sha256(cache_path)
        write(cache_path.with_suffix('.json'), dict(**metadata, cache_sha256=cache_sha))
        freeze = dict(state='all_selection_frozen_before_external_model_queries', manifest_sha256=contract.manifest_sha256,
                      protocol_sha256=protocol_sha, decoder_sha256='d' * 64, models={},
                      source_sha256={'structured_geometry.py': sha256(HERE / 'structured_geometry.py'),
                                     'structured_external.py': 'e' * 64, 'structured_model.py': 'f' * 64},
                      structured_generation_policy_sha256={})
        targets = [dict(source_id=r['source_id'], category=r['category'], label=r['label'],
                        bbox=[0, 0, 1000, 1000] if r['label'] else None,
                        location='middle_center' if r['label'] else 'none', foreground_pixels=50176 if r['label'] else 0,
                        **(dict(mask_width=224, mask_height=224) if r['label'] else {})) for r in manifest]
        rows = []
        for model in MODELS:
            for seed in SEEDS:
                expected = dict(checkpoint_sha256='a' * 64, checkpoint_step=100, run='/frozen/arm_S_s%d' % seed,
                                run_sha256='b' * 64, complete_sha256='c' * 64, base_manifest_sha256='0' * 64)
                row = dict(model=model, seed=seed, state='complete_and_recomputed', n=4, n_normal=2,
                           n_anomaly=2, category_count=2, manifest_sha256=contract.manifest_sha256,
                           external_cache_sha256=cache_sha)
                if model in ('Cstructured', 'S'):
                    run = root / ('arm_S_s%d' % seed)
                    expected['run_sha256'] = write(run / 'run.json', dict(seed=seed))
                    expected['complete_sha256'] = write(run / 'complete.json', dict(state='complete', seed=seed))
                    policy_name = 'initial_generation_policy.json' if model == 'Cstructured' else 'best_generation_policy.json'
                    policy_sha = write(run / policy_name, dict(selected='scalar', seed=seed, model=model))
                    freeze['structured_generation_policy_sha256'][str(Path(expected['run']) / policy_name)] = policy_sha
                    output = run / ('external_mvtec_residual_v1_initial_c' if model == 'Cstructured' else 'external_mvtec_residual_v1')
                    output.mkdir(parents=True)
                    predictions = [dict(source_id=r['source_id'], path=r['path'], category=r['category'], label=r['label'],
                                        split='external', score=(-1, 1, 0, -1)[i], generated=target_answer(t),
                                        parsed=parse_structured(target_answer(t))) for i, (r, t) in enumerate(zip(manifest, targets))]
                    prediction_path = output / 'predictions.jsonl'
                    prediction_path.write_text(''.join(json.dumps(r) + '\n' for r in predictions))
                    prediction_sha = sha256(prediction_path)
                    (output / 'predictions.sha256').write_text(prediction_sha + '  predictions.jsonl\n')
                    target_sha = write(output / 'posthoc_targets.json', targets)
                    config = dict(manifest_sha256=contract.manifest_sha256, protocol_sha256=protocol_sha,
                                  decoder_sha256=freeze['decoder_sha256'], external_cache_sha256=cache_sha,
                                  checkpoint_sha256=expected['checkpoint_sha256'], checkpoint_step=100,
                                  structured_run_sha256=expected['run_sha256'], base_manifest_sha256=expected['base_manifest_sha256'],
                                  max_new_tokens=80, batch_size=4, schema=SCHEMA, seed=seed,
                                  evaluator_sha256='e' * 64, model_source_sha256='f' * 64,
                                  mask_reads_at_start=0, generation_policy_sha256=policy_sha)
                    config_sha = write(output / 'frozen_config.json', config)
                    write(output / 'inference_complete.json', dict(state='inference_complete', processed=4, mask_reads=0,
                          predictions_sha256=prediction_sha, frozen_config_sha256=config_sha))
                    metrics = dict(localization=localization_metrics(targets, [p['generated'] for p in predictions]),
                                   localization_per_category={c: localization_metrics([t for t in targets if t['category'] == c],
                                      [p['generated'] for p in predictions if p['category'] == c]) for c in ('category0', 'category1')})
                    summary = dict(**config, state='complete', n=4, condition='initial_C_structured_prompt' if model == 'Cstructured' else 'S_best',
                                   external_queries=4, historical_visa_test_queries=0, mask_reads_during_inference=0,
                                   posthoc_mask_reads=2, predictions_sha256=prediction_sha, posthoc_targets_sha256=target_sha, metrics=metrics)
                    row.update(summary_sha256=write(output / 'summary.json', summary), predictions_sha256=prediction_sha, metrics=metrics)
                freeze['models'][model + '_s%d' % seed] = expected
                rows.append(row)
        write(root / 'external_selection_freeze.json', freeze)
        summary = dict(state='complete', expected_runs=21, completed_recomputed_runs=21, unavailable=[],
                       requested_models=list(MODELS), requested_seeds=list(SEEDS), manifest_sha256=contract.manifest_sha256,
                       n=4, no_model_or_image_or_mask_queries=True, rows=rows)
        write(root / 'external_summary.json', summary)
        report, boxes = build_report(root, manifest_path, protocol_path, cache_path, contract)
        require(report['state'] == 'complete' and len(boxes) == 4, 'Fixture integration failed')
        require(report['unconditional_teacher_box']['overall']['anomaly_bbox_iou_mean_all'] == .25,
                'Known teacher union box IoU differs')
        gated = report['by_seed'][0]['conditions']['Cstructured_score_gt0_teacher_box']['overall']
        require(gated['anomaly_bbox_iou_mean_all'] == .125 and gated['anomaly_positive_box_coverage'] == .5
                and gated['normal_positive_box_rate'] == 0 and boxes[2]['Cstructured_score_gt0']['42'] is False,
                'Strict >0 gate or all-anomaly denominator differs')
        require(report['unconditional_teacher_box']['per_category']['category1']['n_anomaly'] == 1,
                'Per-category denominator differs')
        checks.append('full synthetic21/six-run CPU integration, known IoU/gate/category denominators')
        saved = summary['state']; summary['state'] = 'partial'; write(root / 'external_summary.json', summary)
        try:
            build_report(root, root / 'never_open_manifest', protocol_path, root / 'never_open_cache', contract)
        except ValueError as error:
            require('complete' in str(error), 'Incomplete21 did not stop before missing raw/cache path')
        else:
            raise AssertionError('Incomplete21 was accepted')
        summary['state'] = saved; write(root / 'external_summary.json', summary)
        checks.append('incomplete21 rejects before manifest/raw/cache access')
        bad_marker = root / 'arm_S_s44/external_mvtec_residual_v1/inference_complete.json'
        marker = read(bad_marker); marker['state'] = 'inferring'; write(bad_marker, marker)
        try:
            build_report(root, manifest_path, protocol_path, root / 'absent_cache.pt', contract)
        except FileNotFoundError:
            # Its sidecar is read before six-run gating, so make metadata-only
            # sidecar available; the tensor itself must remain unavailable.
            write(root / 'absent_cache.json', dict(**metadata, cache_sha256=cache_sha))
            try:
                build_report(root, manifest_path, protocol_path, root / 'absent_cache.pt', contract)
            except ValueError as error:
                require('inference-before-GT' in str(error), 'Six-run gate failed for the wrong reason')
            else:
                raise AssertionError('Incomplete structured marker accepted')
        else:
            raise AssertionError('Fixture expected unavailable sidecar')
        checks.append('incomplete sixth inference marker rejects before tensor access')
        marker['state'] = 'inference_complete'; write(bad_marker, marker)
        # Update declared hashes consistently: a changed valid target must still
        # be rejected for cross-run disagreement, not merely a stale checksum.
        output = root / 'arm_S_s44/external_mvtec_residual_v1'
        changed = json.loads(json.dumps(targets)); changed[1]['bbox'] = [0, 0, 500, 500]; changed[1]['location'] = 'top_left'
        target_sha = write(output / 'posthoc_targets.json', changed)
        saved_summary = read(output / 'summary.json'); saved_summary['posthoc_targets_sha256'] = target_sha
        for row in summary['rows']:
            if row['model'] == 'S' and row['seed'] == 44:
                row['summary_sha256'] = write(output / 'summary.json', saved_summary)
        write(root / 'external_summary.json', summary)
        try:
            build_report(root, manifest_path, protocol_path, cache_path, contract)
        except ValueError as error:
            require('targets differ across' in str(error), 'Target disagreement rejected for wrong reason')
        else:
            raise AssertionError('Cross-run target disagreement accepted')
        checks.append('SHA-consistent cross-run target disagreement rejects')
    tie = teacher_target(np.zeros((4, 4), np.float32))
    require(tie['bbox'] == [0, 0, 1000, 1000], '>= quantile tie rule differs')
    singleton = np.zeros((4, 4), np.float32); singleton[3, 3] = 1
    require(teacher_target(singleton)['bbox'] == [750, 750, 1000, 1000], 'Outward-normalized union box differs')
    checks.append('quantile ties and one-pixel outward box geometry')
    return dict(state='passed', checks=checks, actual_external_outputs_opened=False, device='cpu')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False)); return
    root = args.root.resolve()
    manifest = args.manifest or root / 'external_manifest.jsonl'
    if not manifest.is_file() and args.manifest is None:
        manifest = root / 'external_mvtec/manifest.jsonl'
    protocol = args.protocol or root / 'external_protocol.json'
    if not protocol.is_file() and args.protocol is None:
        protocol = root / 'external_mvtec/protocol.json'
    report, boxes = build_report(root, manifest, protocol, args.cache or root / 'external_teacher_cache.pt')
    output = args.out or root / 'external_teacher_control.json'
    boxes_path = output.with_name(output.stem + '_boxes.jsonl')
    atomic_text(boxes_path, ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in boxes))
    report['evidence']['derived_boxes_sha256'] = sha256(boxes_path)
    atomic_text(output, json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    atomic_text(output.with_suffix('.md'), render_markdown(report))
    print(json.dumps(dict(state=report['state'], n=report['n'], n_anomaly=report['n_anomaly'],
                          all21_complete=True, all6_saved_targets_equal=True, output=str(output))))


if __name__ == '__main__':
    main()
