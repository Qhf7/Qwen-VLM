#!/usr/bin/env python3
"""Frozen paired Jev ROI review; execute tests and API calls on AutoDL only.

prepare --root RUN_ROOT --output runs/modelgo_roi
dry-run --root RUN_ROOT --output runs/modelgo_roi
run --root RUN_ROOT --output runs/modelgo_roi [--retry-failed]
self-test

The duplicate arm supplies a real duplicate, never a mislabeled crop. No GT,
labels, file names, source IDs or teacher split are sent to the API. Credentials
and raw bodies/responses are never persisted. Successful targets are immutable.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import sys
import tempfile
import threading
import time
import uuid

from PIL import Image
import requests
from requests.adapters import HTTPAdapter

SCHEMA = 'jev-fixed-candidate-paired-roi-v1'
MODEL = 'jev-1.14.0'
ENDPOINT = 'https://api.modelgo.ai/v1/systemone'
ARMS = ('roi', 'duplicate')
SIZE = 448
WORKERS = 3
CONNECT_TIMEOUT = 60.0
READ_TIMEOUT = 120.0
MAX_STARTED = 3
ERROR_STOP = 3
SERVER_HOSTNAME = 'autodl-container-132444827f-9c594038'
THREAD_LOCAL = threading.local()
INSTRUCTIONS = (
    'Does the candidate region in the image identified by query contain at least one '
    'visible industrial anomaly (surface damage, contamination, missing material or '
    'component, or structural irregularity)? candidate_bbox_in_query is normalized '
    '[left, top, right, bottom] in query, with x increasing rightward and y downward. '
    'Judge that candidate region, not unrelated regions. second_view contains either '
    'a close-up of this same candidate region or an unchanged duplicate of query, as '
    'declared by second_view_kind. A candidate_crop is an enlarged view of the '
    'recorded original-image bounds with centered white padding; padding is not a '
    'defect. A duplicate_original adds no new visual region. Use only visible evidence. '
    'Ordinary differences in lighting, viewpoint, background or object identity alone '
    'are not defects.'
)
CRITERIA = {
    'true': 'At least one visible industrial defect is present inside the candidate region.',
    'false': 'No visible industrial defect is present inside the candidate region.'
}


class ProtocolError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    if not Path(path).exists():
        return []
    data = Path(path).read_bytes()
    require(not data or data.endswith(b'\n'), 'Incomplete JSONL tail requires explicit audit; refusing append')
    return [json.loads(row) for row in data.splitlines() if row.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_json(path, value):
    with Path(path).open('ab') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def immutable_json(path, value):
    if Path(path).exists():
        require(canonical(read_json(path)) == canonical(value), 'Existing frozen contract differs')
    else:
        write_json(path, value)


def versions():
    packages = {}
    for name in ('Pillow', 'requests', 'urllib3', 'certifi', 'charset-normalizer', 'idna'):
        packages[name] = importlib.metadata.version(name)
    return {'python': platform.python_version(), 'packages': packages}


def inside(root, path):
    target = Path(path)
    target = (root / target if not target.is_absolute() else target).resolve()
    require(target.is_relative_to(root), 'Asset or protocol path escapes experiment root')
    return target


def relative(root, path):
    return Path(path).resolve().relative_to(root).as_posix()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_box(value):
    require(isinstance(value, list) and len(value) == 4, 'Invalid candidate box shape')
    require(all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in value),
            'Candidate coordinates must be finite numbers in [0,1]')
    require(value[0] < value[2] and value[1] < value[3], 'Empty or reversed candidate box')
    return [float(x) for x in value]


def validated_asset(root, metadata):
    require(isinstance(metadata, dict), 'Missing asset metadata')
    require(metadata.get('size') == [SIZE, SIZE], 'Asset size metadata differs from frozen 448')
    require(valid_hash(metadata.get('png_sha256')), 'Invalid asset SHA256')
    require(type(metadata.get('png_bytes')) is int and metadata['png_bytes'] > 0, 'Invalid asset byte count')
    path = inside(root, metadata['path'])
    require(path.is_file(), 'Missing candidate asset')
    raw = path.read_bytes()
    require(len(raw) == metadata['png_bytes'] and digest(raw) == metadata['png_sha256'], 'Asset integrity mismatch')
    with Image.open(io.BytesIO(raw)) as image:
        require(image.format == 'PNG' and image.size == (SIZE, SIZE) and image.mode == 'RGB',
                'Asset must be an RGB 448x448 PNG')
        image.load()
    return {'path': relative(root, path), 'png_sha256': digest(raw),
            'png_bytes': len(raw), 'size': [SIZE, SIZE]}, raw


def build_contract(root, candidates_path, baseline_path, limit_queries=None):
    root = root.resolve()
    candidate_file = inside(root, candidates_path)
    baseline = inside(root, baseline_path)
    config_file = baseline / 'frozen/config.json'
    materialized_file = baseline / 'frozen/materialized.json'
    plan_file = baseline / 'frozen/plan.json'
    protocol_file = candidate_file.with_name('03_protocol.json')
    config, materialized, candidates = (read_json(p) for p in (config_file, materialized_file, candidate_file))
    plan, candidate_protocol = read_json(plan_file), read_json(protocol_file)
    require(config.get('model') == MODEL and config.get('size') == SIZE, 'Unexpected source baseline model or resolution')
    require(config.get('plan_sha256') == digest(canonical(plan)), 'Source plan/config mismatch')
    require(candidates.get('schema') == 'jev-fixed-candidates-v1' and candidates.get('status') == 'complete', 'Candidate stage incomplete')
    require(candidates.get('protocol_sha256') == file_sha(protocol_file), 'Candidate protocol SHA differs')
    require(candidates.get('confirmation_opened') is False and candidates.get('ground_truth_masks_opened') is False, 'Candidate inference isolation gate failed')
    require(candidate_protocol.get('model') == MODEL and candidate_protocol.get('expected_targets') == 240
            and candidate_protocol.get('normal_references') == 0 and candidate_protocol.get('assets_per_request') == 2,
            'Candidate protocol differs from paired ROI design')
    require(materialized.get('status') == 'complete' and not materialized.get('failures'), 'Source materialization is not complete')
    require(materialized.get('config_sha256') == digest(canonical(config)), 'Source materialization/config mismatch')
    source_rows = materialized.get('records', [])
    source_lookup = {r['source_id']: r for r in source_rows}
    require(len(source_lookup) == len(source_rows) == 168, 'Source materialization requires 168 unique images')
    rows = candidates.get('records')
    require(isinstance(rows, list) and len(rows) == 120, 'Need exactly 120 frozen candidate records')
    require(len({r['source_id'] for r in rows}) == 120, 'Duplicate candidate source identity')
    require(len(plan.get('queries', [])) == 120 and {r['source_id'] for r in rows} ==
            {r['source_id'] for r in plan['queries']}, 'Candidate/query source identities differ')
    query_lookup = {r['source_id']: r for r in plan['queries']}
    require(candidates.get('full_image_boxes') == sum(validate_box(r['bbox_original_norm']) == [0., 0., 1., 1.] for r in rows),
            'Candidate full-image count metadata differs')
    require(candidates.get('versions', {}).get('Pillow') == importlib.metadata.version('Pillow'), 'Candidate renderer Pillow version differs')
    require(limit_queries is None or type(limit_queries) is int and 1 <= limit_queries <= 120, 'Invalid preflight query limit')
    selected = sorted(rows, key=lambda row: row['source_id'])
    if limit_queries is not None:
        selected = selected[:limit_queries]
    records, images = [], {}
    for row in selected:
        sid = row['source_id']
        require(isinstance(sid, str) and re.fullmatch('visa_[0-9a-f]{64}', sid) is not None, 'Invalid source ID')
        require(sid in source_lookup, 'Candidate does not match source materialization')
        require(isinstance(row.get('category'), str) and re.fullmatch('[a-z0-9_]+', row['category']) is not None, 'Invalid category')
        require(row['category'] == query_lookup[sid]['category'], 'Candidate/source category differs')
        require(row.get('teacher_split') in ('train', 'val', 'test'), 'Missing teacher training exposure')
        query_meta, query = validated_asset(root, row['assets']['query'])
        crop_meta, crop = validated_asset(root, row['assets']['crop'])
        source = source_lookup[sid]
        require(row.get('original_sha256') == source.get('original_sha256') == sid[5:], 'Candidate original SHA differs from baseline source')
        require(row.get('exif_orientation') == 1, 'Teacher-to-query EXIF orientation gate failed')
        full_image = validate_box(row['bbox_original_norm']) == [0., 0., 1., 1.]
        same_bytes = query == crop
        require(not full_image or same_bytes, 'Full-image crop must exactly equal the newly rendered query PNG')
        record = {'source_id': sid, 'category': row['category'], 'teacher_split': row['teacher_split'],
                  'bbox_original_norm': validate_box(row['bbox_original_norm']),
                  'bbox_448_norm': validate_box(row['bbox_448_norm']),
                  'full_image_candidate': full_image, 'query_crop_bytes_identical': same_bytes,
                  'query_png_matches_phase1': query_meta['png_sha256'] == source.get('png_sha256'),
                  'assets': {'query': query_meta, 'crop': crop_meta}}
        records.append(record)
        images[sid] = {'query': query, 'crop': crop}
    dependencies = versions()
    contract = {'schema': SCHEMA, 'model': MODEL, 'endpoint': ENDPOINT, 'size': SIZE,
                'mode': 'formal_development_candidate_review' if limit_queries is None else 'independent_preflight_not_formal',
                'limit_queries': limit_queries, 'arms': list(ARMS), 'normal_references': 0,
                'instructions': INSTRUCTIONS, 'criteria': CRITERIA, 'threshold': 0.5,
                'threshold_rule': 'probability >= 0.5 retains the fixed candidate box',
                'semantic_target': 'visible anomaly inside the fixed candidate region; not whole-image calibrated probability',
                'source_candidates': relative(root, candidate_file), 'source_baseline': relative(root, baseline),
                'source_sha256': {'candidates': file_sha(candidate_file), 'baseline_config': file_sha(config_file),
                                  'baseline_materialized': file_sha(materialized_file), 'baseline_plan': file_sha(plan_file),
                                  'candidate_protocol': file_sha(protocol_file)},
                'renderer_versions': {'phase1': config.get('versions', {}).get('pillow'),
                                      'stage3': candidates.get('versions', {}).get('Pillow')},
                'rendering_comparison': 'Both new arms use the same new stage3 query PNG. Phase1 scores are historical controls when rendering differs.',
                'teacher_exposure_counts': dict(Counter(row['teacher_split'] for row in records)),
                'teacher_unseen_definition': 'teacher_split != train; development diagnostic, not new blind test',
                'full_image_candidate_count': sum(row['full_image_candidate'] for row in records),
                'query_crop_bytes_identical_count': sum(row['query_crop_bytes_identical'] for row in records),
                'query_png_changed_from_phase1_count': sum(not row['query_png_matches_phase1'] for row in records),
                'identical_asset_rule': 'When crop PNG equals query PNG, both arms declare duplicate_original; payload semantic content is identical except neutral asset UUIDs.',
                'runner_sha256': file_sha(__file__), 'dependencies': dependencies,
                'dependencies_sha256': digest(canonical(dependencies)),
                'workers': WORKERS, 'connect_upload_timeout_seconds': CONNECT_TIMEOUT,
                'read_timeout_seconds': READ_TIMEOUT, 'max_total_started_per_target': MAX_STARTED,
                'consecutive_error_submission_stop': ERROR_STOP, 'automatic_retries': False,
                'retry_rule': 'Only a later explicit --retry-failed invocation; unknown completion may duplicate billing; never retry accepted targets',
                'data_role': 'previously_viewed_development_not_blind_test',
                'confirmation_images_read': False, 'records': records,
                'schedule': [{'source_id': row['source_id'], 'arm': arm} for row in records for arm in ARMS]}
    contract_hash = digest(canonical(contract))
    for record in records:
        if record['query_crop_bytes_identical']:
            roi = request_payload(record, 'roi', images[record['source_id']], contract_hash)
            duplicate = request_payload(record, 'duplicate', images[record['source_id']], contract_hash)
            require(semantic_payload(roi) == semantic_payload(duplicate), 'Identical asset arms differ in semantic payload')
    return contract, images


def request_payload(record, arm, images, contract_hash):
    require(arm in ARMS, 'Unexpected ROI arm')
    assets = {}
    identity = contract_hash + '|' + record['source_id'] + '|' + arm
    def add(role, raw):
        reference = str(uuid.uuid5(uuid.NAMESPACE_URL, identity + '|' + role))
        assets[reference] = {'type': 'image', 'format': 'png', 'data_base64': base64.b64encode(raw).decode('ascii')}
        return reference
    query = add('first', images['query'])
    second = add('second', images['crop'] if arm == 'roi' else images['query'])
    state = {'query': query, 'second_view': second,
             'second_view_kind': 'candidate_crop' if arm == 'roi' and images['query'] != images['crop'] else 'duplicate_original',
             'category': record['category'].replace('_', ' '),
             'candidate_bbox_in_query': record['bbox_448_norm'],
             'candidate_bbox_in_original': record['bbox_original_norm']}
    return {'model': MODEL, 'state': state, 'assets': assets,
            'questions': {'anomaly': {'type': 'noul', 'instructions': INSTRUCTIONS, 'criteria': CRITERIA}}}


def semantic_payload(body):
    """Normalize only neutral asset IDs, retaining roles and image evidence."""
    state = dict(body['state'])
    roles = {}
    for name in ('query', 'second_view'):
        reference = state[name]
        asset = body['assets'][reference]
        roles[name] = {'type': asset['type'], 'format': asset['format'],
                       'sha256': digest(base64.b64decode(asset['data_base64'], validate=True))}
        state[name] = name
    return canonical({'model': body['model'], 'state': state, 'assets': roles, 'questions': body['questions']})


def response_probability(data):
    require(isinstance(data, dict) and data.get('model') == MODEL, 'Missing or mismatched response model')
    require(isinstance(data.get('answers'), dict) and set(data['answers']) == {'anomaly'}, 'Unexpected question identities')
    answer = data['answers']['anomaly']
    require(isinstance(answer, dict) and answer.get('type') == 'noul', 'Response is not typed Noul')
    probability = answer.get('noul')
    require(type(probability) in (float, int) and math.isfinite(probability) and 0 <= probability <= 1, 'Invalid Noul probability')
    return float(probability)


def safe_usage(data):
    usage = data.get('usage')
    if not isinstance(usage, dict):
        return {}
    return {k: value for k, value in usage.items() if isinstance(k, str) and k.endswith('_tokens')
            and type(value) in (int, float) and math.isfinite(value) and value >= 0}


def safe_request_id(value, key):
    if not isinstance(value, str) or not value or len(value) > 200 or key in value:
        return None
    return value if all(c.isalnum() or c in '-_.:' for c in value) else None


def send(body, key):
    start = time.perf_counter()
    session = getattr(THREAD_LOCAL, 'session', None)
    if session is None:
        session = requests.Session()
        session.mount('https://', HTTPAdapter(max_retries=0))
        THREAD_LOCAL.session = session
    status = None
    try:
        response = session.post(ENDPOINT, json=body, headers={'Authorization': 'Bearer ' + key,
            'TypeSafe-Danger-Zone-Images': 'true'}, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), allow_redirects=False)
        status = int(response.status_code)
        if status != 200:
            return None, 'http_' + str(status), status, time.perf_counter() - start
        data = response.json()
        probability = response_probability(data)
        value = {'probability': probability, 'response_model': MODEL, 'usage': safe_usage(data),
                 'request_id': safe_request_id(response.headers.get('x-request-id') or
                     response.headers.get('request-id') or data.get('id'), key)}
        return value, None, status, time.perf_counter() - start
    except requests.Timeout:
        error = 'timeout_unknown_server_completion'
    except requests.RequestException:
        error = 'transport_error_unknown_server_completion'
    except (ValueError, KeyError, TypeError):
        error = 'invalid_typed_response_or_model'
    return None, error, status, time.perf_counter() - start


def target_key(row):
    return row['source_id'], row['arm']


def eligible(target, completed, starts, retry_failed):
    return target not in completed and starts[target] < MAX_STARTED and (starts[target] == 0 or retry_failed)


def validate_logs(output, contract, images):
    contract_hash = digest(canonical(contract))
    records = {r['source_id']: r for r in contract['records']}
    schedule = [target_key(r) for r in contract['schedule']]
    request_hashes = {}
    for sid, arm in schedule:
        request_hashes[sid, arm] = digest(canonical(request_payload(records[sid], arm, images[sid], contract_hash)))
    completed = {}
    predictions = read_rows(output / 'predictions.jsonl')
    for row in predictions:
        pair = target_key(row)
        require(pair in request_hashes and pair not in completed, 'Unexpected or duplicate accepted target')
        require(row.get('contract_sha256') == contract_hash and row.get('request_sha256') == request_hashes[pair], 'Accepted target contract or payload differs')
        p = response_probability({'model': row.get('response_model'), 'answers': {'anomaly': {'type': 'noul', 'noul': row.get('probability')}}})
        require(row.get('retain_candidate') is (p >= 0.5), 'Saved decision differs from fixed threshold')
        require(type(row.get('seconds')) in (int, float) and math.isfinite(row['seconds']) and row['seconds'] >= 0, 'Invalid accepted latency')
        require(safe_usage({'usage': row.get('usage')}) == row.get('usage'), 'Invalid saved usage')
        require(row.get('category') == records[pair[0]]['category'] and row.get('teacher_split') == records[pair[0]]['teacher_split'], 'Accepted metadata differs')
        completed[pair] = row
    attempts = read_rows(output / 'attempts.jsonl')
    starts = Counter()
    terminal = set()
    start_rows = {}
    for row in attempts:
        pair = target_key(row)
        require(pair in request_hashes and row.get('contract_sha256') == contract_hash and
                row.get('request_sha256') == request_hashes[pair], 'Attempt contract or payload differs')
        require(type(row.get('attempt')) is int and 1 <= row['attempt'] <= MAX_STARTED, 'Invalid attempt number')
        identity = pair + (row['attempt'],)
        if row.get('event') == 'started':
            starts[pair] += 1
            require(row['attempt'] == starts[pair] and identity not in start_rows, 'Attempt sequence differs')
            start_rows[identity] = row
        else:
            require(row.get('event') in ('failed', 'succeeded') and identity in start_rows and identity not in terminal, 'Missing or duplicate attempt terminal')
            require(row.get('invocation_id') == start_rows[identity].get('invocation_id'), 'Attempt invocation differs')
            terminal.add(identity)
            if row['event'] == 'succeeded':
                require(pair in completed, 'Succeeded attempt has no durable accepted prediction')
                require(completed[pair]['attempt'] == row['attempt'], 'Succeeded attempt differs from accepted prediction')
    for pair, row in completed.items():
        require(type(row.get('attempt')) is int and pair + (row['attempt'],) in start_rows, 'Accepted prediction lacks durable intent')
    unknown = [identity for identity in start_rows if identity not in terminal and identity[:2] not in completed]
    return completed, starts, attempts, request_hashes, unknown


def status_document(contract, completed, starts, attempts, unknown, state, invocation_id=None, stop_reason=None):
    schedule = [target_key(r) for r in contract['schedule']]
    return {'schema': SCHEMA, 'timestamp_utc': now(), 'status': state, 'mode': contract['mode'],
            'contract_sha256': digest(canonical(contract)), 'expected': len(schedule), 'completed': len(completed),
            'remaining': len(schedule) - len(completed), 'started_attempts': sum(starts.values()),
            'failed_attempts': sum(r.get('event') == 'failed' for r in attempts),
            'interrupted_unknown_attempts': len(unknown),
            'exhausted_unaccepted_targets': sum(pair not in completed and starts[pair] >= MAX_STARTED for pair in schedule),
            'by_arm': {arm: sum(pair[1] == arm for pair in completed) for arm in ARMS},
            'invocation_id': invocation_id, 'stop_reason': stop_reason, 'automatic_retries': False,
            'no_probability_imputation': True, 'confirmation_images_read': False}


@contextmanager
def output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ProtocolError('Another process holds this output directory') from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def setup(args):
    root = args.root.resolve()
    output = inside(root, args.output)
    contract, images = build_contract(root, args.candidates, args.source_baseline, args.limit_queries)
    return root, output, contract, images


def prepare(args):
    root, output, contract, images = setup(args)
    if args.command == 'dry-run':
        if (output / 'contract.json').exists():
            require(canonical(read_json(output / 'contract.json')) == canonical(contract), 'Existing contract differs during dry-run')
        completed, starts, attempts, hashes, unknown = validate_logs(output, contract, images)
        print(json.dumps(status_document(contract, completed, starts, attempts, unknown, 'dry_run_validated'), ensure_ascii=False))
        return 0
    with output_lock(output):
        immutable_json(output / 'contract.json', contract)
        completed, starts, attempts, hashes, unknown = validate_logs(output, contract, images)
        state = 'complete' if len(completed) == len(contract['schedule']) else 'prepared'
        status = status_document(contract, completed, starts, attempts, unknown, state)
        write_json(output / 'status.json', status)
        print(json.dumps(status, ensure_ascii=False))
    return 0


def run(args):
    require(socket.gethostname() == SERVER_HOSTNAME, 'API execution is restricted to the designated AutoDL server')
    root = args.root.resolve()
    output = inside(root, args.output)
    with output_lock(output):
        frozen = read_json(output / 'contract.json')
        contract, images = build_contract(root, frozen['source_candidates'], frozen['source_baseline'], frozen['limit_queries'])
        require(canonical(contract) == canonical(frozen), 'Code, dependencies, source data or asset contract changed')
        completed, starts, attempts, request_hashes, unknown = validate_logs(output, contract, images)
        schedule = [target_key(row) for row in contract['schedule']]
        if len(completed) == len(schedule):
            status = status_document(contract, completed, starts, attempts, unknown, 'complete')
            write_json(output / 'status.json', status)
            print(json.dumps(status, ensure_ascii=False))
            return 0
        pending = [pair for pair in schedule if eligible(pair, completed, starts, args.retry_failed)]
        if not pending:
            status = status_document(contract, completed, starts, attempts, unknown, 'incomplete_requires_audit_or_explicit_retry')
            write_json(output / 'status.json', status)
            print(json.dumps(status, ensure_ascii=False))
            return 2
        if (output / 'STOP').exists():
            status = status_document(contract, completed, starts, attempts, unknown, 'stopped', stop_reason='STOP_file')
            write_json(output / 'status.json', status)
            print(json.dumps(status, ensure_ascii=False))
            return 2
        key = os.environ.get(args.key_env)
        if not key:
            require(sys.stdin.isatty(), 'Credential environment is unset; interactive hidden input requires a TTY')
            key = getpass.getpass('ModelGo API key (hidden, never saved): ')
        require(isinstance(key, str) and key.strip() == key and bool(key), 'Missing or invalid API credential')
        invocation_id = str(uuid.uuid4())
        contract_hash = digest(canonical(contract))
        source = {row['source_id']: row for row in contract['records']}
        append_json(output / 'invocations.jsonl', {'invocation_id': invocation_id, 'timestamp_utc': now(),
            'contract_sha256': contract_hash, 'retry_failed': args.retry_failed,
            'unknown_completion_retry_risk_acknowledged': bool(args.retry_failed and unknown),
            'pending_at_start': len(pending), 'workers': WORKERS})
        futures = {}
        position = 0
        consecutive_errors = 0
        stopped = None
        def submit(pool, pair):
            sid, arm = pair
            body = request_payload(source[sid], arm, images[sid], contract_hash)
            require(digest(canonical(body)) == request_hashes[pair], 'Payload changed before submission')
            starts[pair] += 1
            intent = {'source_id': sid, 'arm': arm, 'attempt': starts[pair], 'event': 'started',
                      'timestamp_utc': now(), 'contract_sha256': contract_hash,
                      'request_sha256': request_hashes[pair], 'invocation_id': invocation_id}
            append_json(output / 'attempts.jsonl', intent)
            attempts.append(intent)
            futures[pool.submit(send, body, key)] = (pair, intent)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            while position < len(pending) or futures:
                if (output / 'STOP').exists():
                    stopped = stopped or 'STOP_file'
                while not stopped and position < len(pending) and len(futures) < WORKERS:
                    submit(pool, pending[position])
                    position += 1
                if not futures:
                    break
                done, _ = wait(futures, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in done:
                    pair, intent = futures.pop(future)
                    value, error, http_status, seconds = future.result()
                    if value is not None:
                        row = {**value, 'source_id': pair[0], 'arm': pair[1], 'attempt': intent['attempt'],
                               'category': source[pair[0]]['category'], 'teacher_split': source[pair[0]]['teacher_split'],
                               'retain_candidate': value['probability'] >= 0.5, 'seconds': seconds,
                               'request_sha256': intent['request_sha256'], 'contract_sha256': contract_hash,
                               'invocation_id': invocation_id, 'timestamp_utc': now()}
                        append_json(output / 'predictions.jsonl', row)
                        completed[pair] = row
                        terminal = {**intent, 'event': 'succeeded', 'timestamp_utc': now(), 'http_status': http_status, 'seconds': seconds}
                        consecutive_errors = 0
                    else:
                        terminal = {**intent, 'event': 'failed', 'timestamp_utc': now(), 'error': error,
                                    'http_status': http_status, 'seconds': seconds}
                        consecutive_errors += 1
                        if consecutive_errors >= ERROR_STOP:
                            stopped = stopped or 'consecutive_errors'
                    append_json(output / 'attempts.jsonl', terminal)
                    attempts.append(terminal)
                    live = status_document(contract, completed, starts, attempts, unknown,
                                           'draining_inflight' if stopped else 'running', invocation_id, stopped)
                    write_json(output / 'status.json', live)
                    print(json.dumps({'completed': len(completed), 'expected': len(schedule),
                                      'arm': pair[1], 'http_status': http_status, 'error': error, 'seconds': seconds}), flush=True)
        completed, starts, attempts, request_hashes, unknown = validate_logs(output, contract, images)
        state = 'complete' if len(completed) == len(schedule) else ('stopped' if stopped else 'incomplete_requires_explicit_retry')
        status = status_document(contract, completed, starts, attempts, unknown, state, invocation_id, stopped)
        write_json(output / 'status.json', status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        return 0 if state == 'complete' else 2


def self_test():
    """Synthetic CPU checks only; no dataset, remote API or credentials."""
    require(validate_box([0, 0.2, 1, 0.8]) == [0., .2, 1., .8], 'Box check failed')
    bad = ([0, 0, 0, 1], [0, 0, float('nan'), 1], [False, 0, 1, 1], [-.1, 0, 1, 1])
    for value in bad:
        try:
            validate_box(value)
        except ProtocolError:
            pass
        else:
            raise ProtocolError('Invalid geometry was accepted')
    buffer = io.BytesIO()
    Image.new('RGB', (SIZE, SIZE), 'white').save(buffer, format='PNG')
    query = buffer.getvalue()
    buffer = io.BytesIO()
    Image.new('RGB', (SIZE, SIZE), 'black').save(buffer, format='PNG')
    crop = buffer.getvalue()
    record = {'source_id': 'visa_' + 'a' * 64, 'category': 'pcb1', 'teacher_split': 'train',
              'label': 1, 'bbox_original_norm': [.2, .3, .6, .7], 'bbox_448_norm': [.2, .3, .6, .7]}
    image_bytes = {'query': query, 'crop': crop}
    roi = request_payload(record, 'roi', image_bytes, 'b' * 64)
    duplicate = request_payload(record, 'duplicate', image_bytes, 'b' * 64)
    require(len(roi['assets']) == len(duplicate['assets']) == 2, 'Asset count differs')
    require(roi['questions'] == duplicate['questions'] and
            roi['state']['candidate_bbox_in_query'] == duplicate['state']['candidate_bbox_in_query'], 'Paired task differs')
    require('source_id' not in canonical(roi).decode() and 'teacher_split' not in canonical(roi).decode()
            and 'label' not in roi['state'], 'Privileged metadata entered payload')
    require(base64.b64decode(duplicate['assets'][duplicate['state']['second_view']]['data_base64']) == query, 'Duplicate arm is not a duplicate')
    require(base64.b64decode(roi['assets'][roi['state']['second_view']]['data_base64']) == crop, 'ROI arm is not the crop')
    require(canonical(roi) == canonical(request_payload(record, 'roi', image_bytes, 'b' * 64)), 'Payload identity is not stable')
    same = {'query': query, 'crop': query}
    require(semantic_payload(request_payload(record, 'roi', same, 'b' * 64)) ==
            semantic_payload(request_payload(record, 'duplicate', same, 'b' * 64)), 'Full-image duplicate content differs by arm')
    good = {'model': MODEL, 'answers': {'anomaly': {'type': 'noul', 'noul': .5}}}
    require(response_probability(good) == .5, 'Typed response check failed')
    for p in (True, float('nan'), -0.1, 1.1, '0.5'):
        try:
            response_probability({'model': MODEL, 'answers': {'anomaly': {'type': 'noul', 'noul': p}}})
        except ProtocolError:
            pass
        else:
            raise ProtocolError('Invalid response probability was accepted')
    target = (record['source_id'], 'roi')
    require(eligible(target, {}, Counter(), False), 'New target was skipped')
    require(not eligible(target, {target: {}}, Counter(), True), 'Accepted target was rescheduled')
    require(not eligible(target, {}, Counter({target: 1}), False), 'Failed target auto-retried')
    require(eligible(target, {}, Counter({target: 1}), True), 'Explicit bounded retry rejected')
    require(not eligible(target, {}, Counter({target: MAX_STARTED}), True), 'Attempt cap exceeded')
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        asset = root / 'fixture.png'
        asset.write_bytes(query)
        metadata = {'path': 'fixture.png', 'png_sha256': digest(query), 'png_bytes': len(query), 'size': [SIZE, SIZE]}
        checked, raw = validated_asset(root, metadata)
        require(raw == query and checked == metadata, 'Asset integrity check failed')
        append_json(root / 'audit.jsonl', {'event': 'synthetic'})
        require(read_rows(root / 'audit.jsonl') == [{'event': 'synthetic'}], 'Durable JSONL check failed')
    print(json.dumps({'status': 'passed', 'synthetic_only': True, 'network_calls': 0,
                      'checks': ['geometry', 'paired_payload', 'no_privileged_metadata', 'typed_response',
                                 'skip_accepted', 'explicit_retry', 'attempt_cap', 'PNG_integrity', 'durable_JSONL']}))
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'dry-run'):
        item = sub.add_parser(name)
        item.add_argument('--root', type=Path, required=True)
        item.add_argument('--candidates', default='stages/03_candidates.json')
        item.add_argument('--source-baseline', default='source_baseline')
        item.add_argument('--output', default='runs/modelgo_roi')
        item.add_argument('--limit-queries', type=int)
        item.set_defaults(func=prepare)
    item = sub.add_parser('run')
    item.add_argument('--root', type=Path, required=True)
    item.add_argument('--output', default='runs/modelgo_roi')
    item.add_argument('--key-env', default='MODELGO_API_KEY')
    item.add_argument('--retry-failed', action='store_true')
    item.set_defaults(func=run)
    item = sub.add_parser('self-test')
    item.set_defaults(func=lambda args: self_test())
    return result


def main():
    args = parser().parse_args()
    try:
        return args.func(args)
    except ProtocolError as error:
        print(json.dumps({'status': 'blocked_protocol', 'reason': str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    except (OSError, KeyError, TypeError, ValueError, importlib.metadata.PackageNotFoundError) as error:
        # Never print arbitrary exception text: transport/config data may contain secrets.
        print(json.dumps({'status': 'blocked_input_or_runtime', 'exception_class': type(error).__name__}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
