#!/usr/bin/env python3
"""Supplementary diagnostics after complete360 only; never calls a model or alters frozen inputs.

Usage: python supplement_results.py --output runs/modelgo_448
Run --self-test for in-memory and temporary-fixture checks, without reading formal predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import fcntl
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import runner as r

PAIRS = ((1, 0), (4, 0), (4, 1))
REPLICATES = 2000
SEED = 42
PHASES = ('original_workers3', 'rolling6', 'transport_rolling6')
LIMITATIONS = [
    'Supplementary category-macro diagnostics are not replacement primary endpoints.',
    'Previously viewed 120-image development validation, not a new blind or cross-domain test.',
    'One fixed normal-support selection and order; nested 1-of-4 supports; no support-selection variance measured.',
    'Increasing supports also increases input information, image tokens and gateway work; no pure sample-count effect isolated.',
    'Bootstrap is conditional on these 12 categories and fixed supports, excludes service randomness, and has no multiplicity correction.',
    'Each category has only five normal and five anomalous queries; confidence intervals and calibration are small-sample diagnostics.',
    'The balanced 50-percent anomaly prevalence differs from production; AP and probability diagnostics do not transfer automatically.',
    'TypeSafe image functionality is experimental; public images may have appeared in upstream pretraining, which cannot be independently excluded.',
    'Gateway logical model-name echo is not independent evidence of upstream weights or immutable provider routing.',
    'Operational concurrency and timeout phases differ; request latency is not intrinsic model speed or a controlled phase comparison.',
    'Transport errors can have unknown server completion; absent failure usage does not mean zero tokens or zero charges.',
    'No currency cost is estimated; reported token totals include only observed usage fields.',
    'Historical Qwen differs in preprocessing, prompt, training exposure, budget and runtime; cross-method gaps are not architectural causal effects.',
]


def require_complete(output):
    """Must run before opening predictions, metrics, frozen labels or image files."""
    status = r.read_json(output / 'status.json')
    if (status.get('status') != 'complete' or status.get('completed') != 360
            or status.get('expected') != 360 or status.get('mode') != 'formal_development_baseline'):
        raise ValueError('Refusing supplemental performance on incomplete or non-formal run; need status complete360.')
    return status


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError('Missing recorded timestamp')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Timestamp has no timezone')
    return parsed.astimezone(timezone.utc)


def same_number(actual, expected, name):
    if type(actual) not in (int, float) or type(expected) not in (int, float):
        raise ValueError('Non-numeric primary metric: ' + name)
    if not math.isfinite(actual) or not math.isfinite(expected) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Supplement does not reproduce primary: ' + name)


def check_primary_metrics(actual, stored, name):
    for key, value in actual.items():
        if key not in stored:
            raise ValueError('Primary metric missing: ' + name + '.' + key)
        same_number(value, stored[key], name + '.' + key)


def confusion(y, p):
    y = np.asarray(y, dtype=int)
    positive = np.asarray(p, dtype=float) >= 0.5
    return {'threshold': 0.5, 'comparison': 'probability >= threshold',
            'TP': int(np.sum(positive & (y == 1))), 'FP': int(np.sum(positive & (y == 0))),
            'TN': int(np.sum(~positive & (y == 0))), 'FN': int(np.sum(~positive & (y == 1))),
            'n': len(y)}


def calibration(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    bins = np.minimum((p * 10).astype(int), 9)
    records = []
    for b in range(10):
        chosen = bins == b
        n = int(chosen.sum())
        mean = float(p[chosen].mean()) if n else None
        rate = float(y[chosen].mean()) if n else None
        records.append({'bin': b, 'lower': b / 10, 'upper': (b + 1) / 10,
                        'interval': 'closed_both' if b == 9 else 'closed_left_open_right',
                        'count': n, 'positive_count': int(y[chosen].sum()),
                        'mean_probability': mean, 'positive_fraction': rate,
                        'absolute_gap': abs(mean - rate) if n else None})
    ece = sum(z['count'] / len(y) * z['absolute_gap'] for z in records if z['count'])
    return {'bins': 10, 'definition': 'equal-width on predicted probability; p=1 belongs to bin9',
            'records': records, 'ece_10_equal_width': ece}


def weighted_rank_metrics(y, p, weights):
    """Exact tied-score AUROC/AP for bootstrap replication counts; no score jitter."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2 or weights.shape[1] != len(y) or len(y) != len(p):
        raise ValueError('Invalid bootstrap weight shape')
    order = np.argsort(-p, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(p[order]) != 0) + 1]
    group_positive = np.add.reduceat(weights[:, order] * y[order][None, :], starts, axis=1)
    group_negative = np.add.reduceat(weights[:, order] * (1 - y[order])[None, :], starts, axis=1)
    tp, fp = np.cumsum(group_positive, axis=1), np.cumsum(group_negative, axis=1)
    positives, negatives = tp[:, -1], fp[:, -1]
    if np.any(positives <= 0) or np.any(negatives <= 0):
        raise ValueError('Bootstrap replicate lacks a binary class')
    previous_tp = np.c_[np.zeros(len(weights)), tp[:, :-1]]
    auc = np.sum(group_negative * (tp + previous_tp) * 0.5, axis=1) / (positives * negatives)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    ap = np.sum(group_positive * precision, axis=1) / positives
    return np.column_stack((auc, ap))


def bootstrap_weights(queries, y, replicates=REPLICATES, seed=SEED):
    strata = defaultdict(list)
    for index, query in enumerate(queries):
        strata[query['category'], int(y[index])].append(index)
    strata = [np.asarray(indices, dtype=int) for _, indices in sorted(strata.items())]
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, len(y)), dtype=int)
    # Loop and stratum order exactly match the frozen primary bootstrap.
    for b in range(replicates):
        indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in strata])
        weights[b] = np.bincount(indices, minlength=len(y))
    return weights


def metric_supplement(queries, y, probs, primary):
    categories = sorted({q['category'] for q in queries})
    index_by_category = {c: np.asarray([i for i, q in enumerate(queries) if q['category'] == c]) for c in categories}
    arms = {}
    for shots in r.SHOTS:
        p = probs[shots]
        check_primary_metrics(r.metrics(y, p), primary['shots'][str(shots)], str(shots))
        classes = {}
        for category, indices in index_by_category.items():
            actual = r.metrics(y[indices], p[indices])
            check_primary_metrics(actual, primary['by_category'][category][str(shots)], category + ':' + str(shots))
            classes[category] = {'auroc': actual['auroc'], 'ap': actual['ap'],
                                 'confusion': confusion(y[indices], p[indices])}
        calibr = calibration(y, p)
        same_number(calibr['ece_10_equal_width'], primary['shots'][str(shots)]['ece_10_equal_width'], 'calibration')
        arms[str(shots)] = {
            'n': len(y), 'category_count': len(categories),
            'category_macro': {m: float(np.mean([classes[c][m] for c in categories])) for m in ('auroc', 'ap')},
            'confusion': confusion(y, p), 'by_category': classes, 'calibration': calibr,
            'binary_label_macro_f1_note': 'Primary macro_f1 averages normal/anomalous labels, not object categories.'}
    weights = bootstrap_weights(queries, y)
    pooled_samples = {s: weighted_rank_metrics(y, probs[s], weights) for s in r.SHOTS}
    macro_samples = {s: np.zeros((REPLICATES, 2), dtype=float) for s in r.SHOTS}
    for category, indices in index_by_category.items():
        for shots in r.SHOTS:
            macro_samples[shots] += weighted_rank_metrics(y[indices], probs[shots][indices], weights[:, indices]) / len(categories)
    comparisons = {}
    for a, b in PAIRS:
        name = str(a) + '-' + str(b)
        comparisons[name] = {}
        for j, metric in enumerate(('auroc', 'ap')):
            # Also reproduce pooled primary percentile intervals with identical draws.
            pooled = np.percentile(pooled_samples[a][:, j] - pooled_samples[b][:, j], [2.5, 97.5])
            old = primary['paired_bootstrap'][name][metric]
            if old.get('replicates') != REPLICATES or old.get('seed') != SEED or old.get('strata') != 'category_and_label':
                raise ValueError('Primary bootstrap configuration differs')
            for actual, expected in zip(pooled, old['percentile_95_ci']):
                same_number(float(actual), expected, 'pooled_ci:' + name + ':' + metric)
            same_number(primary['shots'][str(a)][metric] - primary['shots'][str(b)][metric], old['point_difference'], 'pooled_point')
            point = arms[str(a)]['category_macro'][metric] - arms[str(b)]['category_macro'][metric]
            ci = np.percentile(macro_samples[a][:, j] - macro_samples[b][:, j], [2.5, 97.5]).tolist()
            comparisons[name][metric] = {'point_difference': point, 'percentile_95_ci': ci,
                'point_difference_percentage_points': point * 100,
                'percentile_95_ci_percentage_points': [x * 100 for x in ci],
                'replicates': REPLICATES, 'seed': SEED, 'strata': 'category_and_label',
                'endpoint_status': 'supplementary_not_primary',
                'limitation': 'Fixed categories and support set; no provider or support-selection variance; no multiplicity correction.'}
    return {'definition': 'Unweighted arithmetic mean of 12 within-category AUROCs or APs; not pooled cross-category ranking.',
            'endpoint_status': 'supplementary_not_primary', 'shots': arms,
            'paired_category_macro_bootstrap': comparisons,
            'primary_recomputation': 'All pooled and per-category primary metrics and pooled bootstrap intervals reproduced.'}


def elapsed_summary(values):
    values = np.asarray(values, dtype=float)
    return {'count': len(values), 'sum': float(values.sum()),
            'mean': float(values.mean()) if len(values) else None,
            'median': float(np.median(values)) if len(values) else None,
            'p95': float(np.percentile(values, 95)) if len(values) else None}


def attempt_phase(row):
    phase = row.get('operational_phase', 'original_workers3')
    if phase not in PHASES:
        raise ValueError('Unknown attempt phase')
    return phase


def summarize_attempt_records(attempts, predictions, expected_hashes, contract_hash):
    starts, failures = {}, {}
    sequence = Counter()
    for row in attempts:
        pair = r.result_key(row)
        if pair not in expected_hashes or row.get('contract_sha256') != contract_hash or row.get('request_sha256') != expected_hashes[pair]:
            raise ValueError('Attempt identity/hash differs from frozen request')
        if type(row.get('attempt')) is not int or row['attempt'] < 1:
            raise ValueError('Invalid attempt sequence')
        identity = pair + (row['attempt'],)
        timestamp(row.get('timestamp_utc'))
        attempt_phase(row)
        if row.get('event') == 'started':
            sequence[pair] += 1
            if row['attempt'] != sequence[pair] or identity in starts:
                raise ValueError('Duplicate or nonmonotonic started attempt')
            starts[identity] = row
        elif row.get('event') == 'failed':
            if identity not in starts or identity in failures:
                raise ValueError('Duplicate or orphan failed attempt')
            if attempt_phase(row) != attempt_phase(starts[identity]):
                raise ValueError('Failure phase differs from start')
            sec = row.get('seconds')
            if type(sec) not in (int, float) or not math.isfinite(sec) or sec < 0:
                raise ValueError('Invalid failure elapsed time')
            failures[identity] = row
        else:
            raise ValueError('Unexpected attempt event')
    successes = {}
    for pair, prediction in predictions.items():
        if not sequence[pair]:
            raise ValueError('Successful prediction has no started attempt')
        identity = pair + (sequence[pair],)
        start = starts[identity]
        if identity in failures or attempt_phase(start) != attempt_phase(prediction):
            raise ValueError('Final prediction cannot match latest successful start')
        if start.get('invocation_id') != prediction.get('invocation_id'):
            raise ValueError('Success invocation differs from latest start')
        if timestamp(prediction['timestamp_utc']) < timestamp(start['timestamp_utc']):
            raise ValueError('Success precedes latest started attempt')
        successes[identity] = prediction
    enriched = []
    for identity, start in starts.items():
        failed, success = failures.get(identity), successes.get(identity)
        outcome = 'success' if success is not None else ('failed_observed' if failed is not None else 'interrupted_unknown')
        unknown_server = outcome == 'interrupted_unknown' or (failed is not None and 'unknown_server_completion' in str(failed.get('error', '')))
        terminal = success if success is not None else failed
        usage = r.safe_usage({'usage': terminal.get('usage')}) if terminal else {}
        sec = terminal.get('seconds') if terminal else None
        started = timestamp(start['timestamp_utc'])
        # Failed events inherit start timestamp; elapsed reconstructs a client finish, not a server finish.
        finished = timestamp(success['timestamp_utc']) if success is not None else (started + timedelta(seconds=sec) if failed is not None else None)
        enriched.append({'source_id': identity[0], 'shots': identity[1], 'attempt': identity[2],
            'phase': attempt_phase(start), 'outcome': outcome, 'server_completion_unknown': bool(unknown_server),
            'retry': identity[2] > 1, 'seconds': sec, 'usage': usage, 'started': started,
            'finished': finished, 'error': failed.get('error') if failed else None,
            'http_status': failed.get('http_status') if failed else 200,
            'exception_class': failed.get('exception_class') if failed else None,
            'stage': failed.get('stage') if failed else None})
    def group(items):
        keys = sorted({k for x in items for k in x['usage']})
        counts = Counter(x['outcome'] for x in items)
        return {'started_attempts': len(items), 'distinct_targets': len({(x['source_id'], x['shots']) for x in items}),
            'successful_attempts': counts['success'], 'failed_observed_attempts': counts['failed_observed'],
            'interrupted_without_terminal_event': counts['interrupted_unknown'],
            'server_completion_unknown_attempts': sum(x['server_completion_unknown'] for x in items),
            'retry_attempts': sum(x['retry'] for x in items),
            'successful_after_retry': sum(x['retry'] and x['outcome'] == 'success' for x in items),
            'targets_with_retry_attempt': len({(x['source_id'], x['shots']) for x in items if x['retry']}),
            'elapsed_seconds_success': elapsed_summary([x['seconds'] for x in items if x['outcome'] == 'success']),
            'elapsed_seconds_failed_observed': elapsed_summary([x['seconds'] for x in items if x['outcome'] == 'failed_observed']),
            'usage_reported_totals': {k: sum(x['usage'].get(k, 0) for x in items) for k in keys},
            'usage_reported_by_outcome': {outcome: {k: sum(x['usage'].get(k, 0) for x in items if x['outcome'] == outcome) for k in keys}
                for outcome in ('success', 'failed_observed', 'interrupted_unknown')},
            'usage_present_attempts': sum(bool(x['usage']) for x in items),
            'usage_missing_attempts': sum(not bool(x['usage']) for x in items),
            'usage_missing_failed_or_unknown': sum(not bool(x['usage']) and x['outcome'] != 'success' for x in items),
            'failure_error_counts': dict(Counter(str(x['error']) for x in items if x['outcome'] == 'failed_observed')),
            'failure_http_status_counts': dict(Counter(str(x['http_status']) for x in items if x['outcome'] == 'failed_observed')),
            'failure_exception_class_counts': dict(Counter(x['exception_class'] for x in items if x['exception_class'])),
            'failure_stage_counts': dict(Counter(x['stage'] for x in items if x['stage']))}
    summary = {'overall': group(enriched),
        'by_shots': {str(s): group([x for x in enriched if x['shots'] == s]) for s in r.SHOTS},
        'by_phase': {p: group([x for x in enriched if x['phase'] == p]) for p in PHASES},
        'by_phase_and_shots': {p: {str(s): group([x for x in enriched if x['phase'] == p and x['shots'] == s]) for s in r.SHOTS} for p in PHASES},
        'unknown_attempts': [{k: x[k] for k in ('source_id', 'shots', 'attempt', 'phase', 'outcome', 'error')} for x in enriched if x['server_completion_unknown']],
        'accounting_note': 'Success/failure/interrupted partition starts; server_completion_unknown overlaps observed transport failures. Retry=attempt number >1. Missing usage is not zero usage or cost.',
        'currency_cost': 'Not estimated: no independently verified monetary billing field.'}
    return summary, enriched


def operational_provenance(output, contract_hash, contract_path, attempt_rows, predictions):
    provenance = {}
    for phase, filename, executor in [('rolling6', 'operational_amendment.json', 'resume_executor.py'),
                                      ('transport_rolling6', 'operational_amendment_v2.json', 'transport_executor.py')]:
        path = output / filename
        if not path.exists():
            if any(attempt_phase(x) == phase for x in attempt_rows + list(predictions.values())):
                raise ValueError('Operational phase missing its amendment')
            continue
        amendment = r.read_json(path)
        if (amendment.get('original_contract_canonical_sha256') != contract_hash
                or amendment.get('original_contract_file_sha256') != r.file_sha(contract_path)
                or amendment.get('original_runner_sha256') != r.file_sha(r.__file__)
                or amendment.get('executor_sha256') != r.file_sha(Path(__file__).parent / executor)):
            raise ValueError('Operational amendment provenance mismatch')
        for logname, field in [('predictions.jsonl', 'initial_predictions_file'), ('attempts.jsonl', 'initial_attempts_file')]:
            old = amendment[field]
            if old['exists']:
                with (output / logname).open('rb') as stream:
                    prefix = stream.read(old['bytes'])
                if len(prefix) != old['bytes'] or r.sha(prefix) != old['sha256']:
                    raise ValueError('Operational historical log prefix changed')
        digest = r.sha(r.canonical(amendment))
        for row in attempt_rows + list(predictions.values()):
            if attempt_phase(row) == phase and row.get('amendment_sha256') != digest:
                raise ValueError('Operational row amendment mismatch')
        provenance[phase] = {'file': filename, 'file_sha256': r.file_sha(path), 'canonical_sha256': digest,
            'executor': executor, 'actual_workers': amendment.get('actual_workers'),
            'read_timeout_seconds': amendment.get('actual_read_timeout_seconds', amendment.get('actual_timeout_seconds')),
            'connect_upload_timeout_seconds': amendment.get('actual_connect_upload_timeout_seconds')}
    v2 = output / 'operational_amendment_v2.json'
    if v2.exists():
        doc = r.read_json(v2)
        prev = output / 'operational_amendment.json'
        if (doc.get('previous_amendment_file_sha256') != r.file_sha(prev)
                or doc.get('previous_amendment_canonical_sha256') != r.sha(r.canonical(r.read_json(prev)))):
            raise ValueError('Operational amendment chain mismatch')
    return provenance


def recorded_timing(output, status, enriched):
    starts = [x['started'] for x in enriched]
    finishes = [x['finished'] for x in enriched if x['finished'] is not None]
    sessions, files = [], {}
    for filename in ('operational_sessions.jsonl', 'operational_sessions_v2.jsonl'):
        path = output / filename
        if not path.exists():
            continue
        files[filename] = r.file_sha(path)
        for row in r.jsonl(path):
            began, ended = timestamp(row['started_utc']), timestamp(row['finished_utc'])
            if ended < began:
                raise ValueError('Session timestamps reversed')
            sessions.append({'invocation_id': row['invocation_id'], 'phase': row['operational_phase'],
                'started_utc': began.isoformat(), 'finished_utc': ended.isoformat(),
                'recorded_seconds': (ended - began).total_seconds(), 'actual_workers': row.get('actual_workers'),
                'api_calls_started_this_invocation': row.get('api_calls_started_this_invocation'),
                'amendment_sha256': row.get('amendment_sha256')})
    first = min(starts)
    end_observed = max(finishes)
    last_status = timestamp(status['timestamp_utc'])
    recorded_ends = [last_status, end_observed] + [timestamp(s['finished_utc']) for s in sessions]
    last = max(recorded_ends)
    return {'first_attempt_started_utc': first.isoformat(),
        'last_observed_or_elapsed_reconstructed_client_finish_utc': end_observed.isoformat(),
        'request_observation_span_seconds': (end_observed - first).total_seconds(),
        'last_recorded_status_or_session_finish_utc': last.isoformat(),
        'recorded_total_wall_span_seconds': (last - first).total_seconds(),
        'sessions': sessions, 'session_file_sha256': files,
        'timestamp_sources': ['attempts.jsonl started.timestamp_utc', 'predictions.jsonl timestamp_utc',
            'failed client completion reconstructed as started.timestamp_utc + seconds',
            'status.json timestamp_utc', 'operational_sessions*.jsonl started_utc/finished_utc when available'],
        'limitation': 'Wall span includes pauses/restarts and is not sum of request seconds. Failed events inherit start timestamp; reconstructed finish is client-only. Unknown interrupted attempt duration is not imputed.'}


def build(args):
    output = args.output.resolve()
    status = require_complete(output)  # Never inspect partial formal predictions.
    input_files = ['status.json', 'contract.json', 'predictions.jsonl', 'metrics.json', 'attempts.jsonl']
    before = {name: r.file_sha(output / name) for name in input_files}
    config, plan, labels = r.load_frozen(args.plan_dir)
    queries = plan['queries']
    if len(queries) != 120 or len({q['source_id'] for q in queries}) != 120 or set(labels) != {q['source_id'] for q in queries}:
        raise ValueError('Need exactly 120 unique frozen queries and matching labels')
    y = np.asarray([labels[q['source_id']] for q in queries], dtype=int)
    counts = Counter((q['category'], int(y[i])) for i, q in enumerate(queries))
    categories = sorted({q['category'] for q in queries})
    if (len(categories) != 12 or len(counts) != 24 or any(n != 5 for n in counts.values())
            or set(plan['supports_by_category']) != set(categories)
            or any(len(v) != 4 for v in plan['supports_by_category'].values())):
        raise ValueError('Frozen category/label/support counts differ')
    if config.get('threshold') != 0.5 or config.get('ece_bins') != 10 or config.get('bootstrap') != {'seed': SEED, 'replicates': REPLICATES, 'strata': 'category_and_label'}:
        raise ValueError('Frozen metric configuration differs')
    supports = [x for rows in plan['supports_by_category'].values() for x in rows]
    for field in ('source_id', 'sha256', 'relative_image'):
        if len({x[field] for x in supports}) != 48 or {x[field] for x in supports} & {x[field] for x in queries}:
            raise ValueError('Query/support identity overlap or duplicate supports')
    materialized, images = r.verified_images(args.plan_dir, config, plan)
    contract = r.read_json(output / 'contract.json')
    if (contract.get('mode') != 'formal_development_baseline' or contract.get('limit_queries') is not None
            or contract.get('config_sha256') != r.sha(r.canonical(config))
            or contract.get('runner_sha256') != r.file_sha(r.__file__)
            or contract.get('materialized_records_sha256') != r.sha(r.canonical(materialized['records']))
            or contract.get('versions') != r.versions()):
        raise ValueError('Formal contract mismatch')
    r.validate_endpoint(contract['endpoint'])
    schedule = r.schedule_for(plan, None)
    if contract.get('schedule') != [{'source_id': q['source_id'], 'shots': s} for q, s in schedule]:
        raise ValueError('Formal schedule differs')
    contract_hash = r.sha(r.canonical(contract))
    allowed = {(q['source_id'], s) for q, s in schedule}
    rows = r.validate_records(r.jsonl(output / 'predictions.jsonl'), allowed, contract_hash)
    if len(allowed) != 360 or set(rows) != allowed:
        raise ValueError('Need complete unique 360 predictions')
    request_hashes = {}
    for query, shots in schedule:
        pair = query['source_id'], shots
        payload = r.request_payload(query, plan['supports_by_category'][query['category']], shots, images,
                                    contract_hash + '|' + query['source_id'] + '|' + str(shots))
        digest = r.sha(r.canonical(payload))
        if rows[pair]['request_sha256'] != digest or rows[pair].get('category') != query['category']:
            raise ValueError('Prediction differs from frozen request or category')
        request_hashes[pair] = digest
    del images
    primary = r.read_json(output / 'metrics.json')
    if (primary.get('status') != 'complete' or primary.get('prediction_file_sha256') != before['predictions.jsonl']
            or primary.get('model') != r.MODEL or primary.get('threshold') != 0.5
            or primary.get('data_role') != plan['data_role']):
        raise ValueError('Primary summary identity mismatch')
    probs = {s: np.asarray([rows[q['source_id'], s]['probability'] for q in queries]) for s in r.SHOTS}
    supplement = metric_supplement(queries, y, probs, primary)
    attempts = r.jsonl(output / 'attempts.jsonl')
    amendments = operational_provenance(output, contract_hash, output / 'contract.json', attempts, rows)
    operations, enriched = summarize_attempt_records(attempts, rows, request_hashes, contract_hash)
    if operations['overall']['successful_attempts'] != 360:
        raise ValueError('Attempt accounting does not reproduce 360 successes')
    for s in r.SHOTS:
        if operations['by_shots'][str(s)]['successful_attempts'] != 120:
            raise ValueError('Attempt arm is incomplete')
        observed_success_usage = {k: v for k, v in operations['by_shots'][str(s)]['usage_reported_by_outcome']['success'].items()
                                  if k in primary['shots'][str(s)]['usage_reported_totals']}
        if observed_success_usage != primary['shots'][str(s)]['usage_reported_totals']:
            raise ValueError('Usage totals do not reproduce primary')
    result = {'schema': 'jev-industrial-supplement-v1', 'status': 'complete', 'generated_utc': r.now(),
        'endpoint_status': 'supplementary_not_primary', 'model_requested': r.MODEL, 'data_role': plan['data_role'],
        'formal_target_judgments': 360, 'query_count': 120, 'support_count': 48,
        'metrics': supplement, 'attempt_accounting': operations,
        'timing': recorded_timing(output, status, enriched), 'operational_amendments': amendments,
        'limitations': LIMITATIONS, 'source_file_sha256': before,
        'frozen_file_sha256': {name: r.file_sha(args.plan_dir / name) for name in ('config.json', 'plan.json', 'labels.json', 'materialized.json')},
        'script_sha256': r.file_sha(__file__), 'runtime_versions': r.versions()}
    require_complete(output)
    if any(r.file_sha(output / name) != digest for name, digest in before.items()):
        raise ValueError('Formal inputs changed during supplement calculation; no output written')
    r.write_json(output / 'supplemental_metrics.json', result)
    return result


def self_test():
    """Temporary/mocked checks; no access to the real formal run."""
    rng = np.random.default_rng(17)
    y = np.array([0, 1, 0, 1, 1, 0])
    for p in (np.array([.2, .8, .2, .8, .5, .5]), np.ones(6) * .5, np.array([0, 1, 0, 1, 1, 0])):
        weights = np.vstack([np.ones(6, dtype=int), rng.integers(1, 4, size=(9, 6))])
        actual = weighted_rank_metrics(y, p, weights)
        for i, w in enumerate(weights):
            indices = np.repeat(np.arange(6), w)
            expected = [roc_auc_score(y[indices], p[indices]), average_precision_score(y[indices], p[indices])]
            if not np.allclose(actual[i], expected, atol=1e-12, rtol=1e-12):
                raise AssertionError('Weighted tied-score rank metric mismatch')
    cal = calibration(np.array([0, 1, 1]), np.array([0., .1, 1.]))
    if [x['count'] for x in cal['records']] != [1, 1, 0, 0, 0, 0, 0, 0, 0, 1]:
        raise AssertionError('Calibration edge placement differs')
    queries = [{'category': 'a'}, {'category': 'a'}, {'category': 'b'}, {'category': 'b'}]
    weights = bootstrap_weights(queries, np.array([0, 1, 0, 1]), replicates=7)
    if not np.array_equal(weights, np.ones((7, 4), dtype=int)):
        raise AssertionError('Category-label bootstrap failed')
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        r.write_json(root / 'status.json', {'status': 'incomplete', 'completed': 359, 'expected': 360, 'mode': 'formal_development_baseline'})
        try:
            require_complete(root)
        except ValueError:
            pass
        else:
            raise AssertionError('Incomplete formal run was accepted')
    contract_hash = 'b' * 64
    digest = 'a' * 64
    base = {'source_id': 'q', 'shots': 0, 'contract_sha256': contract_hash, 'request_sha256': digest}
    first = dict(base, event='started', attempt=1, timestamp_utc='2026-11-10T00:00:00+00:00')
    failure = dict(first, event='failed', seconds=2., error='transport_error_unknown_server_completion', http_status=None)
    second = dict(base, event='started', attempt=2, timestamp_utc='2026-11-10T00:00:03+00:00')
    prediction = dict(base, timestamp_utc='2026-11-10T00:00:04+00:00', seconds=1., usage={'input_tokens': 10, 'output_tokens': 1})
    report, _ = summarize_attempt_records([first, failure, second], {('q', 0): prediction}, {('q', 0): digest}, contract_hash)
    overall = report['overall']
    if (overall['started_attempts'], overall['successful_attempts'], overall['failed_observed_attempts'],
            overall['server_completion_unknown_attempts'], overall['retry_attempts']) != (2, 1, 1, 1, 1):
        raise AssertionError('Attempt outcome accounting failed')
    interrupted, _ = summarize_attempt_records([first, second], {('q', 0): prediction}, {('q', 0): digest}, contract_hash)
    if (interrupted['overall']['interrupted_without_terminal_event'] != 1
            or interrupted['overall']['server_completion_unknown_attempts'] != 1):
        raise AssertionError('Interrupted attempt was not preserved as unknown')
    # Full-size synthetic metric fixture, entirely independent of formal files.
    queries = [{'category': 'c' + str(c).zfill(2)} for c in range(12) for _ in range(10)]
    y = np.tile(np.r_[np.zeros(5, dtype=int), np.ones(5, dtype=int)], 12)
    probs = {s: np.round(rng.random(120), 1) for s in r.SHOTS}
    primary = {'shots': {str(s): r.metrics(y, probs[s]) for s in r.SHOTS},
               'by_category': {}, 'paired_bootstrap': {}}
    for c in range(12):
        indices = np.arange(c * 10, (c + 1) * 10)
        primary['by_category']['c' + str(c).zfill(2)] = {str(s): r.metrics(y[indices], probs[s][indices]) for s in r.SHOTS}
    weights = bootstrap_weights(queries, y)
    ranks = {s: weighted_rank_metrics(y, probs[s], weights) for s in r.SHOTS}
    for replicate in (0, 1, 17, 1999):
        indices = np.repeat(np.arange(120), weights[replicate])
        for s in r.SHOTS:
            direct = [roc_auc_score(y[indices], probs[s][indices]), average_precision_score(y[indices], probs[s][indices])]
            if not np.allclose(ranks[s][replicate], direct, atol=1e-12, rtol=1e-12):
                raise AssertionError('Full-size bootstrap tied rank mismatch')
    for a, b in PAIRS:
        primary['paired_bootstrap'][str(a) + '-' + str(b)] = {metric: {
            'point_difference': primary['shots'][str(a)][metric] - primary['shots'][str(b)][metric],
            'percentile_95_ci': np.percentile(ranks[a][:, j] - ranks[b][:, j], [2.5, 97.5]).tolist(),
            'replicates': REPLICATES, 'seed': SEED, 'strata': 'category_and_label'} for j, metric in enumerate(('auroc', 'ap'))}
    supplemented = metric_supplement(queries, y, probs, primary)
    for s in r.SHOTS:
        for metric in ('auroc', 'ap'):
            expected = np.mean([v[str(s)][metric] for v in primary['by_category'].values()])
            same_number(supplemented['shots'][str(s)]['category_macro'][metric], float(expected), 'mock_category_macro')
    return {'self_test': 'passed', 'checks': ['tied weighted AUROC/AP versus sklearn', 'calibration boundaries',
        'category-label bootstrap', 'refusal before incomplete run reads', 'attempt retry/unknown accounting',
        'complete120 synthetic category macro and paired bootstrap'],
        'formal_predictions_read': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument('--plan-dir', type=Path, default=root / 'frozen')
    parser.add_argument('--output', type=Path, default=root / 'runs' / 'modelgo_448')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    try:
        if args.self_test:
            print(json.dumps(self_test(), ensure_ascii=False))
            return 0
        # Open existing lock read-only, never create/alter the executor lock file.
        with (args.output / '.run.lock').open('r') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('Executor still active; supplemental evaluation requires completed drained run')
            try:
                result = build(args)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        print(json.dumps({'status': result['status'], 'targets': 360,
            'output': str(args.output / 'supplemental_metrics.json'), 'primary_recomputed': True}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

