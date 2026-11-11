#!/usr/bin/env python3
"""Operational transport amendment v2; scientific contract remains immutable."""
from __future__ import annotations
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import getpass
import json
import math
import os
from pathlib import Path
import sys
import time
import threading
import runner as r

PHASE = 'transport_rolling6'
WORKERS = 6
CONNECT_UPLOAD_TIMEOUT = 60.0
READ_TIMEOUT = 120.0
SESSION_LOCAL = threading.local()
MAX_ATTEMPTS = 5
TRANSIENT_HTTP = {429, 502, 503, 504}

def snapshot(path):
    path = Path(path)
    return {'exists': path.exists(), 'sha256': r.file_sha(path) if path.exists() else None,
            'bytes': path.stat().st_size if path.exists() else 0}

def rows(path):
    return r.jsonl(path) if path.exists() else []

def verify_prefix(path, old):
    if not old['exists']:
        return
    if not path.exists():
        raise ValueError('Historical log disappeared after operational amendment')
    with path.open('rb') as stream:
        prefix = stream.read(old['bytes'])
    if len(prefix) != old['bytes'] or r.sha(prefix) != old['sha256']:
        raise ValueError('Historical log prefix changed after operational amendment')

def successful_response(endpoint, key, body):
    start = time.perf_counter(); response_row = None; error = None; status = None
    exception_class = None; cause_classes = []; stage = None
    def classify(exc):
        names = []; current = exc
        for _ in range(5):
            if current is None: break
            name = type(current).__name__
            names.append(name if name.isidentifier() and len(name) <= 80 else 'OtherException')
            current = current.__cause__ or current.__context__
        return names
    session = getattr(SESSION_LOCAL, 'session', None)
    if session is None:
        session = r.requests.Session()
        # Preserve requests' default trust_env=True and the existing proxy environment.
        SESSION_LOCAL.session = session
    try:
        response = session.post(endpoint, json=body, headers={'Authorization': 'Bearer ' + key,
            'TypeSafe-Danger-Zone-Images': 'true'}, timeout=(CONNECT_UPLOAD_TIMEOUT, READ_TIMEOUT), allow_redirects=False)
        status = response.status_code
        if status != 200:
            error = 'http_' + str(status)
        else:
            data = response.json(); probability = r.response_probability(data)
            response_row = {'probability': probability, 'response_model': data['model'], 'usage': r.safe_usage(data),
                'request_id': r.safe_request_id(response.headers.get('x-request-id') or response.headers.get('request-id') or data.get('id'), key)}
    except r.requests.Timeout as exc:
        error = 'timeout_unknown_server_completion_no_automatic_retry_this_invocation'
        cause_classes = classify(exc); exception_class = cause_classes[0]
        stage = 'connect_or_upload_timeout' if isinstance(exc, r.requests.ConnectTimeout) else ('read_timeout' if isinstance(exc, r.requests.ReadTimeout) else 'unknown_timeout')
    except r.requests.RequestException as exc:
        error = 'transport_error_unknown_server_completion_no_automatic_retry_this_invocation'
        cause_classes = classify(exc); exception_class = cause_classes[0]; stage = 'transport_unknown'
    except (ValueError, KeyError, TypeError) as exc:
        error = 'invalid_typed_response_or_model'
        cause_classes = classify(exc); exception_class = cause_classes[0]; stage = 'response_parse_or_contract'
    return response_row, error, status, time.perf_counter() - start, {'exception_class': exception_class, 'cause_classes': cause_classes, 'stage': stage}

def inspect(args):
    config, plan, _ = r.load_frozen(args.plan_dir)
    materialized, images = r.verified_images(args.plan_dir, config, plan)
    contract_path = args.output / 'contract.json'; contract = r.read_json(contract_path)
    if contract['schema'] != r.SCHEMA or contract['mode'] != 'formal_development_baseline' or contract['limit_queries'] is not None:
        raise ValueError('Operational executor requires existing complete formal schedule')
    r.validate_endpoint(contract['endpoint'])
    if (contract['config_sha256'] != r.sha(r.canonical(config)) or contract['runner_sha256'] != r.file_sha(r.__file__) or
            contract['versions'] != r.versions() or contract['materialized_records_sha256'] != r.sha(r.canonical(materialized['records']))):
        raise ValueError('Scientific contract, runtime or materialized data changed')
    schedule = r.schedule_for(plan, None)
    if len(schedule) != 360 or contract['schedule'] != [{'source_id': q['source_id'], 'shots': s} for q, s in schedule]:
        raise ValueError('Frozen schedule differs')
    contract_hash = r.sha(r.canonical(contract)); allowed = {(q['source_id'], s) for q, s in schedule}
    predictions_path = args.output / 'predictions.jsonl'; attempts_path = args.output / 'attempts.jsonl'
    predictions = rows(predictions_path)
    completed = r.validate_records(predictions, allowed, contract_hash)
    query_map = {q['source_id']: q for q in plan['queries']}
    request_hashes = {}
    for q, shots in schedule:
        identity = contract_hash + '|' + q['source_id'] + '|' + str(shots)
        body = r.request_payload(q, plan['supports_by_category'][q['category']], shots, images, identity)
        request_hashes[q['source_id'], shots] = r.sha(r.canonical(body))
    for pair, row in completed.items():
        if row['request_sha256'] != request_hashes[pair] or row.get('category') != query_map[pair[0]]['category']:
            raise ValueError('Persisted success differs from original frozen payload')
    attempts = rows(attempts_path); starts = Counter(); latest = {}; unresolved = []
    for row in attempts:
        pair = r.result_key(row)
        if (pair not in allowed or row.get('contract_sha256') != contract_hash or
                row.get('request_sha256') != request_hashes[pair] or row.get('event') not in ('started', 'failed')):
            raise ValueError('Persisted attempt differs from original frozen payload')
        if type(row.get('attempt')) is not int or row['attempt'] < 1:
            raise ValueError('Invalid historical attempt number')
        if row['event'] == 'started':
            starts[pair] += 1
            if row['attempt'] != starts[pair]:
                raise ValueError('Historical attempt sequence is not monotonic')
        elif row['attempt'] > starts[pair]:
            raise ValueError('Historical failed event has no started event')
        latest[pair] = row
    for pair, row in latest.items():
        if pair not in completed and row['event'] == 'started':
            unresolved.append({'source_id': pair[0], 'shots': pair[1], 'attempt': row['attempt'],
                               'request_sha256': row['request_sha256']})
    previous_amendment_path = args.output / 'operational_amendment.json'
    previous_amendment = r.read_json(previous_amendment_path)
    previous_amendment_hash = r.sha(r.canonical(previous_amendment))
    verify_prefix(predictions_path, previous_amendment['initial_predictions_file'])
    verify_prefix(attempts_path, previous_amendment['initial_attempts_file'])
    amendment_path = args.output / 'operational_amendment_v2.json'
    if amendment_path.exists():
        amendment = r.read_json(amendment_path)
        expected = {'original_contract_file_sha256': r.file_sha(contract_path), 'original_contract_canonical_sha256': contract_hash,
                    'original_runner_sha256': r.file_sha(r.__file__), 'executor_sha256': r.file_sha(__file__),
                    'actual_workers': WORKERS, 'actual_connect_upload_timeout_seconds': CONNECT_UPLOAD_TIMEOUT, 'actual_read_timeout_seconds': READ_TIMEOUT, 'max_total_started_per_target': MAX_ATTEMPTS, 'previous_amendment_file_sha256': r.file_sha(previous_amendment_path), 'previous_amendment_canonical_sha256': previous_amendment_hash,
                    'operational_phase': PHASE}
        for key, value in expected.items():
            if amendment.get(key) != value:
                raise ValueError('Operational amendment differs: ' + key)
        verify_prefix(predictions_path, amendment['initial_predictions_file'])
        verify_prefix(attempts_path, amendment['initial_attempts_file'])
    else:
        if len(unresolved) > WORKERS:
            raise ValueError('More interrupted unknown targets than original concurrent workers')
        amendment = {'schema': 'jev-operational-amendment-v2', 'timestamp_utc': r.now(),
            'reason': 'Large-image transport failures clustered near 15 seconds; increase connect/upload timeout and read timeout with per-thread connection reuse. Scientific inputs and request identities remain unchanged.',
            'operational_phase': PHASE, 'original_contract_file_sha256': r.file_sha(contract_path),
            'original_contract_canonical_sha256': contract_hash, 'original_runner_sha256': r.file_sha(r.__file__),
            'executor_sha256': r.file_sha(__file__), 'actual_workers': WORKERS, 'actual_connect_upload_timeout_seconds': CONNECT_UPLOAD_TIMEOUT, 'actual_read_timeout_seconds': READ_TIMEOUT, 'previous_amendment_file_sha256': r.file_sha(previous_amendment_path), 'previous_amendment_canonical_sha256': previous_amendment_hash,
            'original_workers': contract['workers'], 'original_timeout_seconds': contract['timeout'],
            'max_total_started_per_target': MAX_ATTEMPTS, 'initial_completed_count': len(completed),
            'initial_predictions_file': snapshot(predictions_path), 'initial_attempts_file': snapshot(attempts_path),
            'interrupted_unknown_targets': unresolved, 'maximum_previous_inflight_at_interrupt': WORKERS,
            'retry_authorization': 'Explicit --retry-failed required; each target max 5 total started across all phases.',
            'same_invocation_retry': 'HTTP 502 only. Transport/timeout/other failures wait until next explicit invocation.',
            'latency_note': 'Report original, rolling6 and transport_rolling6 phases separately; higher timeout changes observed successful latency and failure rate.', 'connection_reuse': 'requests.Session per worker via thread-local storage; trust_env left at default True.'}
        r.write_json(amendment_path, amendment)
    amendment_hash = r.sha(r.canonical(amendment))
    for row in predictions + attempts:
        phase = row.get('operational_phase')
        if phase == 'rolling6':
            if row.get('amendment_sha256') != previous_amendment_hash: raise ValueError('Invalid previous operational provenance')
        elif phase == PHASE:
            if row.get('amendment_sha256') != amendment_hash: raise ValueError('Invalid transport operational provenance')
        elif phase is not None: raise ValueError('Unknown operational phase')
    return config, plan, contract, contract_hash, images, schedule, completed, starts, amendment_hash, unresolved

def engine_summary(output, amendment_hash):
    predictions = rows(output / 'predictions.jsonl'); attempts = rows(output / 'attempts.jsonl'); phases = {}
    for phase in ('original_workers3', 'rolling6', PHASE):
        selected = [x for x in predictions if x.get('operational_phase', 'original_workers3') == phase]
        started = [x for x in attempts if x.get('operational_phase', 'original_workers3') == phase and x['event'] == 'started']
        failed = [x for x in attempts if x.get('operational_phase', 'original_workers3') == phase and x['event'] == 'failed']
        latency = [x['seconds'] for x in selected]
        phases[phase] = {'actual_workers': 3 if phase == 'original_workers3' else WORKERS,
            'successful_targets': len(selected), 'started_attempts': len(started), 'failed_attempts': len(failed),
            'errors': dict(Counter(str(x.get('error')) for x in failed)),
            'exception_classes': dict(Counter(str(x.get('exception_class')) for x in failed if x.get('exception_class'))),
            'cause_classes': dict(Counter(c for x in failed for c in x.get('cause_classes', []))),
            'stages': dict(Counter(str(x.get('stage')) for x in failed if x.get('stage'))),
            'mean_success_seconds': sum(latency) / len(latency) if latency else None,
            'median_success_seconds': float(r.np.median(latency)) if latency else None,
            'p95_success_seconds': float(r.np.percentile(latency, 95)) if latency else None}
    doc = {'timestamp_utc': r.now(), 'amendment_sha256': amendment_hash, 'phases': phases,
           'total_completed': len(predictions), 'expected': 360,
           'note': 'Success latency by phase; rates are operational and not controlled model comparisons. Started attempts may include interrupted unknown outcomes.'}
    r.write_json(output / 'phase_engineering_summary_v2.json', doc)
    return doc

def execute(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.run.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('Another executor holds the experiment lock; stop and drain it first')
        try: return execute_locked(args)
        finally: fcntl.flock(lock, fcntl.LOCK_UN)

def execute_locked(args):
    _, plan, contract, contract_hash, images, schedule, completed, starts, amendment_hash, unknown = inspect(args)
    incomplete_attempted = [pair for pair in starts if pair not in completed]
    if incomplete_attempted and not args.retry_failed:
        raise ValueError('Incomplete failed/interrupted targets require explicit --retry-failed')
    pending = deque((q, shots) for q, shots in schedule if (q['source_id'], shots) not in completed and starts[q['source_id'], shots] < MAX_ATTEMPTS)
    if args.dry_run:
        print(json.dumps({'phase': PHASE, 'completed': len(completed), 'eligible_pending': len(pending),
                          'interrupted_unknown': len(unknown), 'workers': WORKERS, 'api_calls': 0})); return 0
    if len(completed) == 360:
        engine_summary(args.output, amendment_hash)
        print(json.dumps({'status': 'complete', 'completed': 360, 'api_calls': 0, 'key_requested': False})); return 0
    key = os.environ.get(args.key_env) or getpass.getpass('API key (not persisted): ')
    if not key: raise ValueError('Missing API credential')
    predictions_path = args.output / 'predictions.jsonl'; attempts_path = args.output / 'attempts.jsonl'
    session_id = str(r.uuid.uuid4()); session_start = r.now(); consecutive_provider_errors = 0
    waiting_for_explicit_retry = set(); stop_observed = False; futures = {}
    def submit(pool, query, shots):
        pair = query['source_id'], shots
        if pair in completed or starts[pair] >= MAX_ATTEMPTS: return
        body = r.request_payload(query, plan['supports_by_category'][query['category']], shots, images,
            contract_hash + '|' + query['source_id'] + '|' + str(shots))
        starts[pair] += 1
        attempt = {'source_id': query['source_id'], 'shots': shots, 'contract_sha256': contract_hash,
            'request_sha256': r.sha(r.canonical(body)), 'timestamp_utc': r.now(), 'attempt': starts[pair], 'event': 'started',
            'operational_phase': PHASE, 'amendment_sha256': amendment_hash, 'invocation_id': session_id}
        r.append_json(attempts_path, attempt)
        futures[pool.submit(successful_response, contract['endpoint'], key, body)] = (query, shots, attempt)
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            while pending or futures:
                stop_observed = stop_observed or args.stop_file.exists()
                while pending and len(futures) < WORKERS and not stop_observed:
                    q, shots = pending.popleft(); submit(pool, q, shots)
                if not futures: break
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    query, shots, attempt = futures.pop(future); pair = query['source_id'], shots
                    response_row, error, status, elapsed, diagnostics = future.result()
                    if response_row is not None:
                        row = {**response_row, 'source_id': query['source_id'], 'shots': shots, 'category': query['category'],
                            'request_sha256': attempt['request_sha256'], 'contract_sha256': contract_hash,
                            'timestamp_utc': r.now(), 'seconds': elapsed, 'operational_phase': PHASE,
                            'amendment_sha256': amendment_hash, 'invocation_id': session_id}
                        r.append_json(predictions_path, row); completed[pair] = row; consecutive_provider_errors = 0
                        print(json.dumps({'completed': len(completed), 'expected': 360, 'shots': shots, 'seconds': elapsed, 'phase': PHASE}), flush=True)
                    else:
                        r.append_json(attempts_path, {**attempt, 'event': 'failed', 'error': error, 'http_status': status, 'seconds': elapsed, **diagnostics})
                        print(json.dumps({'error': error, 'http_status': status, 'attempt': starts[pair], 'phase': PHASE}), flush=True)
                        if status in TRANSIENT_HTTP: consecutive_provider_errors += 1
                        else: consecutive_provider_errors = 0
                        if args.retry_failed and status == 502 and starts[pair] < MAX_ATTEMPTS:
                            pending.append((query, shots))
                        else: waiting_for_explicit_retry.add(pair)
                        if consecutive_provider_errors >= 3:
                            time.sleep(3); consecutive_provider_errors = 0
                    stop_observed = stop_observed or args.stop_file.exists()
                    r.write_json(args.output / 'status.json', {'timestamp_utc': r.now(), 'mode': contract['mode'],
                        'expected': 360, 'completed': len(completed), 'status': 'complete' if len(completed) == 360 else 'incomplete',
                        'operational_phase': PHASE, 'actual_workers': WORKERS, 'amendment_sha256': amendment_hash,
                        'stop_file_observed': stop_observed,
                        'model_version_note': 'ModelGo logical echo is not independent proof of upstream version.'})
    finally:
        key = None
        summary = engine_summary(args.output, amendment_hash)
        r.append_json(args.output / 'operational_sessions_v2.jsonl', {'invocation_id': session_id, 'started_utc': session_start,
            'finished_utc': r.now(), 'operational_phase': PHASE, 'amendment_sha256': amendment_hash,
            'actual_workers': WORKERS, 'connect_upload_timeout': CONNECT_UPLOAD_TIMEOUT, 'read_timeout': READ_TIMEOUT, 'retry_failed': args.retry_failed,
            'stop_file_observed': stop_observed, 'completed': len(completed), 'api_calls_started_this_invocation':
            sum(1 for x in rows(attempts_path) if x.get('invocation_id') == session_id and x['event'] == 'started')})
    print(json.dumps({'status': 'complete' if len(completed) == 360 else 'incomplete', 'completed': len(completed),
        'waiting_for_next_explicit_retry': len(waiting_for_explicit_retry), 'stop_file_observed': stop_observed,
        'exhausted_incomplete': sum(pair not in completed and n >= MAX_ATTEMPTS for pair, n in starts.items())}))
    return 0 if len(completed) == 360 else 2

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan-dir', type=Path, default=Path(__file__).resolve().parent / 'frozen')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--key-env', default='MODELGO_API_KEY')
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--stop-file', type=Path)
    return p

def main():
    args = parser().parse_args()
    if args.stop_file is None: args.stop_file = args.output / 'STOP_V2'
    try: return execute(args)
    except (ValueError, KeyError, OSError) as exc:
        print('ERROR: ' + str(exc) if isinstance(exc, ValueError) else 'ERROR: ' + type(exc).__name__, file=sys.stderr); return 2

if __name__ == '__main__': raise SystemExit(main())
