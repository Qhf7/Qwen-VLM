"""Recompute saved external predictions and report complete/partial evidence.

Reads saved JSON/JSONL only. It does not import a model, open RGB/masks, query a
server, or select a checkpoint. All comparisons retain source order and same-
seed pairing. Structured localization is recomputed from saved posthoc targets.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'code'))
from migration_metrics import evaluate_predictions

SEEDS = (42, 43, 44)
BASIC = ('A', 'C', 'D', 'Cpair', 'R')
STRUCTURED = ('Cstructured', 'S')
CORE_KEYS = ('image_auroc', 'image_ap', 'accuracy', 'macro_f1', 'recall', 'fpr', 'invalid_rate')
LOCAL_KEYS = ('format_valid_rate', 'format_valid_rate_anomaly', 'format_valid_rate_normal',
              'anomaly_bbox_iou_mean_all', 'anomaly_iou_at_05_all',
              'anomaly_box_center_inside_gt_box_all', 'anomaly_grid3_accuracy_all',
              'anomaly_bbox_iou_mean_given_valid_positive', 'anomaly_iou_at_05_given_valid_positive')
EXPECTED_N = 1575
EXPECTED_MANIFEST = '4cf2709614c8fd6aabe1268ef3c9f8687a07a0849d3d0f0341d1756570f9497a'
CONTRASTS = (
    ('C_minus_A', 'C', 'A', 'LoRA continuation on the same frozen bridge; selected validation checkpoints.'),
    ('D_minus_C', 'D', 'C', 'Joint bridge+LoRA versus LoRA-only; different trainables and bridge learning rate.'),
    ('D_minus_A', 'D', 'A', 'Joint bridge+LoRA continuation versus its original bridge.'),
    ('R_minus_Cpair', 'R', 'Cpair', 'Matched paired data sequence/microbatch/optimizer; ranking objective contrast.'),
    ('Cpair_minus_C', 'Cpair', 'C', 'Compound category-paired sampling and microbatch change; not ranking effect.'),
    ('Cstructured_minus_C', 'Cstructured', 'C', 'Same C weights; structured instructions and generation budget differ.'),
    ('S_minus_Cstructured', 'S', 'Cstructured', 'Structured supervised continuation; same schema/prompt/budget; adds masks and tokens.'),
)


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def exact_metrics(actual, expected, location='metrics'):
    if isinstance(actual, dict):
        require(isinstance(expected, dict) and set(actual) == set(expected), 'Metric keys differ: ' + location)
        for key, value in actual.items():
            exact_metrics(value, expected[key], location + '.' + key)
    elif isinstance(actual, (float, int)) and not isinstance(actual, bool):
        require(type(expected) in (float, int) and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), 'Metric differs: ' + location)
    else:
        require(actual == expected, 'Metric differs: ' + location)


def numeric_metrics(metrics):
    values = {}
    for group in ('overall', 'category_macro'):
        for channel in ('scores', 'generated'):
            for key in CORE_KEYS:
                value = metrics.get(group, {}).get(channel, {}).get(key)
                if type(value) in (int, float) and math.isfinite(value):
                    values[group + '.' + channel + '.' + key] = float(value)
    for key in LOCAL_KEYS:
        value = metrics.get('localization', {}).get(key)
        if type(value) in (int, float) and math.isfinite(value):
            values['localization.' + key] = float(value)
    if 'localization_per_category' in metrics:
        for key in LOCAL_KEYS:
            values_by_category = [x.get(key) for x in metrics['localization_per_category'].values()]
            finite = [float(x) for x in values_by_category if type(x) in (int, float) and math.isfinite(x)]
            if finite:
                values['category_macro.localization.' + key] = statistics.mean(finite)
    return values


def describe(values):
    values = list(values)
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values) > 1 else None)


def run_paths(advanced, historical, model, seed):
    if model in ('A', 'C'):
        run = historical / ('arm_C_s%d' % seed)
    elif model in STRUCTURED:
        run = advanced / ('arm_S_s%d' % seed)
    else:
        run = advanced / (('arm_D' if model == 'D' else model) + '_s%d' % seed)
    dirname = ('external_mvtec_residual_v1_A' if model == 'A' else
               'external_mvtec_residual_v1_initial_c' if model == 'Cstructured' else
               'external_mvtec_residual_v1')
    return run, run / dirname


def category_values(metrics):
    result = {}
    for category, channels in metrics['per_category'].items():
        result[category] = {channel + '.' + key: float(value)
                            for channel, fields in channels.items() for key, value in fields.items()
                            if key in CORE_KEYS and type(value) in (int, float) and math.isfinite(value)}
        for key in LOCAL_KEYS:
            value = metrics.get('localization_per_category', {}).get(category, {}).get(key)
            if type(value) in (int, float) and math.isfinite(value):
                result[category]['localization.' + key] = float(value)
    return result



def validate_saved_policy(advanced, run, output, model, seed, cfg, summary):
    """Verify the saved implementation decision without re-running generation."""
    frozen_path = advanced / 'external_selection_freeze.json'
    frozen = read(frozen_path)
    expected = frozen['models'][model + '_s%d' % seed]
    require(frozen.get('manifest_sha256') == summary['manifest_sha256']
            and expected['checkpoint_sha256'] == summary['checkpoint_sha256']
            and expected['checkpoint_step'] == summary['checkpoint_step'],
            'Summary checkpoint/manifest differs from pre-external freeze')
    structured = model in STRUCTURED
    maximum = 80 if structured else 32
    policy_name = ('initial_generation_policy.json' if model == 'Cstructured' else 'best_generation_policy.json')
    path = run / policy_name if structured else output / 'generation_policy.json'
    policy = read(path)
    source_sha = frozen['source_sha256']['advanced_generation.py']
    require(policy.get('version') == 'exact-gated-leftpad-greedy-v1'
            and policy.get('selected') in ('scalar', 'batched')
            and policy.get('max_new_tokens') == maximum
            and policy.get('batch_size') == cfg['eval_batch_size']
            and policy.get('source_sha256') == source_sha
            and policy.get('gate_split') == 'VisA_existing_val_only'
            and policy.get('scalar_parent_exact') is True, 'Generation policy source/budget/gate differs')
    require(len(policy.get('gate_indices', [])) == len(policy.get('gate_paths', [])) == 8
            and len(set(policy['gate_indices'])) == len(set(policy['gate_paths'])) == 8
            and len(policy.get('gate_labels', [])) == len(policy.get('gate_categories', [])) == 8
            and set(policy['gate_labels']) == {0, 1} and len(set(policy['gate_categories'])) == 4,
            'Generation policy is not the fixed val8 engineering gate')
    if policy['selected'] == 'batched':
        require(policy.get('exact_token_ids_and_stop_ids') is True and policy.get('failure') is None
                and policy.get('scalar_token_ids') == policy.get('batched_token_ids')
                and policy.get('scalar_stop_ids') == policy.get('batched_stop_ids'),
                'Batched policy lacks exact saved tokens/EOS')
    digest = sha256(path)
    require(summary.get('generation_policy_sha256') == digest
            and summary.get('generation_policy') == policy['selected']
            and summary.get('max_new_tokens') == maximum
            and summary.get('batch_size') == policy['batch_size'], 'Summary generation policy evidence differs')
    require(expected.get('generation_max_new_tokens') == maximum
            and expected.get('generation_batch_size') == policy['batch_size'],
            'Generation budget differs from pre-external freeze')
    evaluator = 'structured_external.py' if structured else 'evaluate_external_checkpoint.py'
    require(summary.get('evaluator_sha256') == frozen['source_sha256'][evaluator],
            'Evaluator differs from pre-external freeze')
    if structured:
        frozen_key = str(Path(expected['run']) / policy_name)
        require(frozen['structured_generation_policy_sha256'].get(frozen_key) == digest,
                'Structured initial/best policy differs from pre-external freeze')
    else:
        require(summary.get('generation_policy_source_sha256') == source_sha
                and summary.get('generation_policy_gate_split') == policy['gate_split'],
                'Basic summary generation source/split differs')
    return dict(selected=policy['selected'], sha256=digest, source_sha256=source_sha,
                max_new_tokens=maximum, batch_size=policy['batch_size'], gate_split=policy['gate_split'],
                frozen_selection_sha256=sha256(frozen_path))

def load_result(advanced, historical, model, seed, manifest, manifest_sha):
    run, output = run_paths(advanced, historical, model, seed)
    summary_path, prediction_path = output / 'summary.json', output / 'predictions.jsonl'
    if not summary_path.exists():
        return None, dict(model=model, seed=seed, state='partial' if output.exists() else 'missing',
                          expected_summary=str(summary_path), reason='No completed summary; raw partial rows are not counted')
    summary = read(summary_path)
    if summary.get('state') != 'complete':
        return None, dict(model=model, seed=seed, state='partial', expected_summary=str(summary_path),
                          reason='Saved summary state is not complete')
    require(summary['seed'] == seed and summary['n'] == EXPECTED_N, 'Result identity/count differs')
    require(summary['manifest_sha256'] == manifest_sha, 'Result manifest SHA differs')
    require(summary.get('external_queries') == EXPECTED_N and summary.get('historical_visa_test_queries') == 0,
            'Query denominator/history differs')
    predictions = jsonl(prediction_path)
    require(len(predictions) == EXPECTED_N and len({x['source_id'] for x in predictions}) == EXPECTED_N,
            'Predictions are incomplete or duplicate source IDs')
    for record, expected in zip(predictions, manifest):
        require(all(record[key] == expected[key] for key in ('source_id', 'path', 'category', 'label'))
                and record['split'] == 'external', 'Prediction order/source metadata differs from frozen manifest')
        require(type(record['score']) in (int, float) and math.isfinite(record['score'])
                and isinstance(record['generated'], str), 'Invalid score/generation raw data')
    digest = sha256(prediction_path)
    checksum_file = output / 'predictions.sha256'
    require(checksum_file.exists() and checksum_file.read_text().split()[0] == digest,
            'Prediction checksum absent or mismatched')
    if 'predictions_sha256' in summary:
        require(summary['predictions_sha256'] == digest, 'Summary prediction SHA differs')
    metrics = evaluate_predictions(predictions, [r['score'] for r in predictions], [r['generated'] for r in predictions])
    structured = model in STRUCTURED
    if structured:
        from structured_geometry import SCHEMA, localization_metrics, parse_structured
        require(summary.get('schema') == SCHEMA and summary.get('max_new_tokens') == 80,
                'Structured schema/generation budget differs')
        require(summary.get('condition') == ('initial_C_structured_prompt' if model == 'Cstructured' else 'S_best'),
                'Structured condition differs')
        require(summary.get('mask_reads_during_inference') == 0 and summary.get('posthoc_mask_reads') == 1183,
                'Structured posterior mask-read contract differs')
        target_path = output / 'posthoc_targets.json'
        require(sha256(target_path) == summary['posthoc_targets_sha256'], 'Saved posterior targets SHA differs')
        targets = read(target_path)
        require(len(targets) == EXPECTED_N and all(all(t[k] == r[k] for k in ('source_id', 'category', 'label'))
                for t, r in zip(targets, predictions)), 'Posterior targets order/source differs')
        for record in predictions:
            require(record.get('parsed') == parse_structured(record['generated']), 'Saved structured parsing differs')
        metrics['localization'] = localization_metrics(targets, [r['generated'] for r in predictions])
        metrics['localization_per_category'] = {
            c: localization_metrics([t for t in targets if t['category'] == c],
                                   [r['generated'] for r in predictions if r['category'] == c])
            for c in sorted({r['category'] for r in predictions})}
        inference = read(output / 'inference_complete.json')
        require(inference.get('state') == 'inference_complete' and inference.get('processed') == EXPECTED_N
                and inference.get('mask_reads') == 0 and inference.get('predictions_sha256') == digest,
                'Inference-before-GT marker differs')
    else:
        require(summary.get('arm') == ('C' if model == 'Cpair' else model), 'Basic arm identity differs')
        require(summary.get('mask_opened') is False, 'Basic inference unexpectedly opened GT masks')
    exact_metrics(metrics, summary['metrics'])
    metric_path = output / 'metrics.json'
    require(metric_path.exists(), 'Separate metrics artifact missing')
    exact_metrics(metrics, read(metric_path))
    source_run, complete = read(run / 'run.json'), read(run / 'complete.json')
    cfg, identity = source_run['config'], source_run['identity']
    require(identity['seed'] == complete['seed'] == seed and complete.get('preflight') is False
            and complete.get('steps') == 500, 'Selected source training identity differs')
    checkpoint_step = identity['bridge_step'] if model == 'A' else identity['source_c_step'] if model == 'Cstructured' else complete['best_step']
    require(summary['checkpoint_step'] == checkpoint_step, 'External checkpoint differs from validation-selected checkpoint')
    require(complete.get('reload_max_abs_score_error') == 0, 'Source training reload gate failed')
    generation_policy = validate_saved_policy(advanced, run, output, model, seed, cfg, summary)
    checkpoint_sha = summary.get('checkpoint_sha256')
    require(isinstance(checkpoint_sha, str) and len(checkpoint_sha) == 64, 'Missing checkpoint SHA')
    if model == 'A':
        require(checkpoint_sha == identity['bridge_sha256'], 'A checkpoint is not source bridge')
    if model == 'Cstructured':
        require(checkpoint_sha == identity['source_c_best_sha256'], 'Cstructured checkpoint is not original C best')
    weight_path = (Path(identity['bridge_path']) if model == 'A' else
                   Path(cfg['source_c']) / 'best.pt' if model == 'Cstructured' else run / 'best.pt')
    weights_verified = weight_path.is_file()
    if weights_verified:
        require(sha256(weight_path) == checkpoint_sha, 'Checkpoint bytes changed')
    cache_sha = summary.get('external_cache_sha256') if structured else summary.get('cache_sha256')
    require(isinstance(cache_sha, str) and len(cache_sha) == 64, 'Missing external cache SHA')
    inference_cost = {key: summary.get(key) for key in ('mean_latency_seconds', 'peak_memory_bytes',
                      'elapsed_seconds', 'inference_seconds', 'total_elapsed_seconds')}
    source_cost = {key: complete.get(key) for key in ('train_seconds', 'exposures_per_second', 'peak_memory_bytes')}
    source_trainable = source_run.get('trainable_parameters')
    if model == 'A':
        # A weights are the original trained bridge, not this hosting C run's adapter.
        original_run = Path(identity['bridge_path']).parent / 'run.json'
        original_spec = read(original_run) if original_run.is_file() else {}
        source_trainable = original_spec.get('trainable_parameters')
        source_cost = {}
    elif model == 'Cstructured':
        original_run = Path(cfg['source_c']) / 'run.json'
        original_done = Path(cfg['source_c']) / 'complete.json'
        if original_run.is_file():
            source_trainable = read(original_run).get('trainable_parameters')
        if original_done.is_file():
            original = read(original_done)
            source_cost = {key: original.get(key) for key in ('train_seconds', 'exposures_per_second', 'peak_memory_bytes')}
    added_trainable = 0 if model in ('A', 'Cstructured') else source_run.get('trainable_parameters')
    base_identity = identity.get('base_manifest_sha256')
    source_order_sha = hashlib.sha256(json.dumps([(r['source_id'], r['path'], r['category'], r['label'])
                                                for r in predictions], separators=(',', ':')).encode()).hexdigest()
    result = dict(model=model, seed=seed, state='complete_and_recomputed', run=str(run), output=str(output),
                  n=EXPECTED_N, n_normal=sum(r['label'] == 0 for r in predictions), n_anomaly=sum(r['label'] == 1 for r in predictions),
                  category_count=len(metrics['per_category']), checkpoint_step=checkpoint_step,
                  checkpoint_sha256=checkpoint_sha, checkpoint_bytes_verified_locally=weights_verified,
                  source_trainable_parameters=source_trainable,
                  newly_trained_parameters_for_this_condition=added_trainable,
                  training_exposures_for_this_condition=0 if model in ('A', 'Cstructured') else complete.get('exposures', 2000),
                  maximum_generated_tokens=80 if structured else 32,
                  schema=summary.get('schema', 'historical_YesNo'),
                  inference_cost=inference_cost, source_training_cost=source_cost,
                  manifest_sha256=manifest_sha, external_cache_sha256=cache_sha,
                  base_manifest_sha256=base_identity, source_order_sha256=source_order_sha,
                  summary_sha256=sha256(summary_path), predictions_sha256=digest,
                  evaluator_sha256=summary.get('evaluator_sha256'), generation_policy=generation_policy, metrics=metrics,
                  flat_metrics=numeric_metrics(metrics), category_metrics=category_values(metrics))
    return result, None


def aggregate_rows(rows, models):
    aggregate = {}
    for model in models:
        group = [row for row in rows if row['model'] == model]
        metric_keys = sorted(set.intersection(*(set(row['flat_metrics']) for row in group))) if group else []
        category_keys = sorted(set.intersection(*(set(row['category_metrics']) for row in group))) if group else []
        categories = {}
        for category in category_keys:
            keys = sorted(set.intersection(*(set(row['category_metrics'][category]) for row in group)))
            categories[category] = {key: describe(row['category_metrics'][category][key] for row in group) for key in keys}
        cost = {}
        for key in ('mean_latency_seconds', 'peak_memory_bytes', 'inference_seconds', 'elapsed_seconds'):
            values = [row['inference_cost'][key] for row in group if type(row['inference_cost'].get(key)) in (int, float)]
            if values:
                cost[key] = describe(values)
        aggregate[model] = dict(state='complete_3seed' if {row['seed'] for row in group} == set(SEEDS) else 'partial',
                               seeds=[row['seed'] for row in group], n_seeds=len(group),
                               metrics={key: describe(row['flat_metrics'][key] for row in group) for key in metric_keys},
                               per_category=categories, inference_cost=cost)
    return aggregate


def paired_results(rows, models):
    lookup = {(row['model'], row['seed']): row for row in rows}
    result = {}
    for contrast, model, reference, boundary in CONTRASTS:
        if model not in models or reference not in models:
            continue
        pairs = []
        for seed in SEEDS:
            if (model, seed) not in lookup or (reference, seed) not in lookup:
                continue
            current, baseline = lookup[(model, seed)], lookup[(reference, seed)]
            for key in ('manifest_sha256', 'external_cache_sha256', 'base_manifest_sha256', 'source_order_sha256'):
                require(current[key] == baseline[key], 'Pair identity mismatch: ' + contrast + ' ' + key)
            keys = sorted(set(current['flat_metrics']) & set(baseline['flat_metrics']))
            delta = {key: current['flat_metrics'][key] - baseline['flat_metrics'][key] for key in keys}
            category_delta = {}
            for category in sorted(set(current['category_metrics']) & set(baseline['category_metrics'])):
                left, right = current['category_metrics'][category], baseline['category_metrics'][category]
                category_delta[category] = {key: left[key] - right[key] for key in sorted(set(left) & set(right))}
            pairs.append(dict(seed=seed, delta=delta, per_category_delta=category_delta))
        keys = sorted(set.intersection(*(set(pair['delta']) for pair in pairs))) if pairs else []
        aggregate = {key: dict(**describe(pair['delta'][key] for pair in pairs),
                              positive_seeds=sum(pair['delta'][key] > 0 for pair in pairs),
                              negative_seeds=sum(pair['delta'][key] < 0 for pair in pairs)) for key in keys}
        category_aggregate = {}
        if pairs:
            for category in pairs[0]['per_category_delta']:
                common = set.intersection(*(set(pair['per_category_delta'][category]) for pair in pairs))
                category_aggregate[category] = {key: describe(pair['per_category_delta'][category][key] for pair in pairs)
                                                for key in sorted(common)}
        result[contrast] = dict(state='complete_3seed' if len(pairs) == 3 else 'partial',
                               attribution_boundary=boundary, pairs=pairs, aggregate=aggregate,
                               per_category_aggregate=category_aggregate,
                               delta_units='raw metric difference; multiply by100 for percentage points; FPR increase is worse')
    return result


def render_markdown(summary):
    lines = ['# Frozen external transfer audit', '', 'State: **' + summary['state'] + '**. Each included run is recomputed from all1575 saved raw predictions in the frozen source order.', '',
             '| Model | Seeds | AUROC mean±SD | AP | Generated F1 | Recall | FPR | Category AUROC |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    def value(stats, key, sd=False):
        metric = stats.get(key)
        if not metric or metric['mean'] is None:
            return '—'
        return ('%.6f±%.6f' % (metric['mean'], metric['sample_sd'])) if sd and metric['sample_sd'] is not None else '%.6f' % metric['mean']
    for model, group in summary['aggregate'].items():
        stats = group['metrics']
        lines.append('| %s | %d/3 | %s | %s | %s | %s | %s | %s |' % (
            model, group['n_seeds'], value(stats, 'overall.scores.image_auroc', True), value(stats, 'overall.scores.image_ap'),
            value(stats, 'overall.generated.macro_f1'), value(stats, 'overall.generated.recall'), value(stats, 'overall.generated.fpr'),
            value(stats, 'category_macro.scores.image_auroc')))
    lines += ['', 'Differences are computed within the same seed before aggregation. Partial seed groups remain explicitly partial.', '',
              '| Contrast | Pairs | ΔAUROC | ΔAP | ΔGenerated F1 | ΔRecall | ΔFPR |', '|---|---:|---:|---:|---:|---:|---:|']
    for contrast, result in summary['paired'].items():
        stats = result['aggregate']
        lines.append('| %s | %d/3 | %s | %s | %s | %s | %s |' % (contrast, len(result['pairs']),
            value(stats, 'overall.scores.image_auroc'), value(stats, 'overall.scores.image_ap'), value(stats, 'overall.generated.macro_f1'),
            value(stats, 'overall.generated.recall'), value(stats, 'overall.generated.fpr')))
    lines += ['', 'Checkpoint steps, source hashes, training parameters, inference costs, per-category seed means/differences and localization denominators are in the JSON artifact.', '',
              'A is the original selected source bridge; Cstructured reuses C weights with structured instructions. S adds mask supervision and answer tokens. Basic max generation32 and structured80 differ; cross-schema comparisons do not isolate a LoRA mechanism.', '',
              'This is a public cross-dataset audit. VisA validation selects checkpoints; MVTec results cannot select models or prompts. Foundation pretraining exposure is unknown. Sample SD across3 seeds is not a significance test.']
    if summary['unavailable']:
        lines += ['', 'Unavailable or rejected runs:', '']
        lines.extend('- %s seed%d: %s — %s' % (x['model'], x['seed'], x['state'], x['reason']) for x in summary['unavailable'])
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--advanced-root', '--root', dest='advanced_root', type=Path, default=HERE)
    ap.add_argument('--historical-root', type=Path)
    ap.add_argument('--manifest', type=Path)
    ap.add_argument('--include-structured', action='store_true')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--markdown-out', type=Path)
    args = ap.parse_args()
    advanced = args.advanced_root.resolve()
    historical = (args.historical_root or advanced.parent / 'lora_20261104').resolve()
    manifest_path = args.manifest or advanced / 'external_manifest.jsonl'
    if not manifest_path.exists() and args.manifest is None:
        manifest_path = advanced / 'external_mvtec/manifest.jsonl'
    manifest = jsonl(manifest_path)
    manifest_sha = sha256(manifest_path)
    require(manifest_sha == EXPECTED_MANIFEST and len(manifest) == EXPECTED_N
            and len({r['source_id'] for r in manifest}) == EXPECTED_N
            and sum(r['label'] for r in manifest) == 1183 and len({r['category'] for r in manifest}) == 15,
            'Manifest does not match frozen1575/392normal/1183anomaly/15category protocol')
    models = BASIC + (STRUCTURED if args.include_structured else ())
    rows, unavailable = [], []
    for model in models:
        for seed in SEEDS:
            try:
                row, issue = load_result(advanced, historical, model, seed, manifest, manifest_sha)
                if row:
                    rows.append(row)
                if issue:
                    unavailable.append(issue)
            except Exception as error:
                unavailable.append(dict(model=model, seed=seed, state='invalid', reason=str(error)))
    # A cache identity is shared by every comparison; reject mixed caches globally.
    common_cache = {row['external_cache_sha256'] for row in rows}
    require(len(common_cache) <= 1, 'Saved results use different external caches; refusing pooled report')
    summary = dict(state='complete' if len(rows) == len(models) * 3 and not unavailable else 'partial',
                   requested_models=list(models), requested_seeds=list(SEEDS), expected_runs=len(models) * 3,
                   completed_recomputed_runs=len(rows), manifest_sha256=manifest_sha, n=EXPECTED_N,
                   rows=rows, unavailable=unavailable, aggregate=aggregate_rows(rows, models),
                   paired=paired_results(rows, models), summarizer_sha256=sha256(__file__),
                   no_model_or_image_or_mask_queries=True,
                   limitations=['Saved raw predictions are the source of metrics; no incomplete result enters a mean.',
                                'Partial means use only listed seeds and are not complete three-seed results.',
                                'Parameters/cost separate source training from newly trained condition parameters; A/Cstructured have zero new training.',
                                'Basic and structured instructions/max output tokens differ; cross-schema differences are compound.',
                                'No external selection, private unseen-data guarantee, or significance claim.'])
    output = args.out or advanced / 'external_summary.json'
    markdown = args.markdown_out or output.with_suffix('.md')
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(output)
    markdown.write_text(render_markdown(summary))
    print(json.dumps(dict(state=summary['state'], completed_recomputed_runs=len(rows), expected_runs=len(models) * 3,
                          unavailable=len(unavailable), output=str(output))))


if __name__ == '__main__':
    main()
