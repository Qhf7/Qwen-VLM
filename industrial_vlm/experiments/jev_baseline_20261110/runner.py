#!/usr/bin/env python3
"""Frozen Jev image baseline; never persist credentials or raw requests."""
from __future__ import annotations
import argparse, base64, fcntl, getpass, hashlib, io, json, math, os, platform, sys, time, uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
import numpy as np
from PIL import Image, ImageOps, __version__ as PILLOW_VERSION
import requests, sklearn
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

MODEL = 'jev-1.14.0'
SHOTS = (0, 1, 4)
SCHEMA = 'jev-industrial-baseline-v1'
SIZE = 448
REMOTE_ROOT = PurePosixPath('/root/autodl-tmp/industrial_vlm/datasets/VisA')
INSTRUCTIONS = ('Does the image in state field query show at least one visible industrial anomaly '
    '(surface damage, contamination, missing material or component, or structural irregularity)? '
    'Treat normal_references, if provided, as normal examples of the same object category. '
    'Judge query, not the references. Ordinary differences in viewpoint, lighting, background, '
    'or object identity alone are not defects.')
CRITERIA = {'true': 'At least one visible industrial defect is present in query.',
            'false': 'No visible industrial defect is present in query.'}
ROOT = Path(__file__).resolve().parent

def now(): return datetime.now(timezone.utc).isoformat()
def sha(data): return hashlib.sha256(data).hexdigest()
def canonical(obj): return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def file_sha(path): return sha(Path(path).read_bytes())
def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)
def append_json(path, obj):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(canonical(obj).decode() + '\n'); stream.flush(); os.fsync(stream.fileno())
def jsonl(path): return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
def versions(): return {'python': platform.python_version(), 'pillow': PILLOW_VERSION, 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'requests': requests.__version__}
def validate_endpoint(endpoint):
    u = urlsplit(endpoint)
    if u.scheme != 'https' or not u.hostname or u.path != '/v1/systemone' or u.username or u.password or u.query or u.fragment:
        raise ValueError('Endpoint must be HTTPS /v1/systemone without URL credentials or query')
    return endpoint

def clean_row(row, audit):
    sid = row['source_id']; original = str(PurePosixPath(row['image']))
    if sid not in audit: raise ValueError('Missing historical source hash')
    hist = audit[sid]
    if original != str(PurePosixPath(hist['image'])) or sid != 'visa_' + hist['sha256']:
        raise ValueError('Manifest and historical byte identity disagree')
    relative = PurePosixPath(original).relative_to(REMOTE_ROOT)
    if '..' in relative.parts: raise ValueError('Unsafe image path')
    return {'source_id': sid, 'category': row['category'], 'relative_image': str(relative),
            'sha256': hist['sha256'], 'source_bytes': hist['bytes']}

def immutable_write(path, obj):
    if Path(path).exists():
        if canonical(read_json(path)) != canonical(obj): raise ValueError('Frozen artifact differs: ' + Path(path).name)
    else: write_json(path, obj)

def prepare(args):
    query_rows = jsonl(args.queries); train_rows = jsonl(args.train); audit = {}
    for row in read_json(args.audit)['images']:
        if row['source_id'] in audit and audit[row['source_id']] != row: raise ValueError('Conflicting historical byte identities')
        audit[row['source_id']] = row
    if len(query_rows) != 120 or len({r['source_id'] for r in query_rows}) != 120: raise ValueError('Need exactly 120 unique queries')
    if any(type(r['is_anomalous']) is not bool for r in query_rows + train_rows): raise ValueError('Labels must be booleans')
    counts = Counter((r['category'], r['is_anomalous']) for r in query_rows)
    categories = sorted({r['category'] for r in query_rows})
    if len(categories) != 12 or any(counts[c, y] != 5 for c in categories for y in (False, True)):
        raise ValueError('Need 12 classes, five normal and five anomalous each')
    unique = {}
    for row in train_rows:
        sid = row['source_id']
        if sid in unique and any(unique[sid][k] != row[k] for k in ('image', 'category', 'is_anomalous')):
            raise ValueError('Conflicting duplicate training source')
        unique[sid] = row
    if len(unique) != 768: raise ValueError('Need 768 unique training sources')
    queries = [clean_row(r, audit) for r in sorted(query_rows, key=lambda r: r['source_id'])]
    train = [clean_row(r, audit) for r in unique.values()]
    for key in ('source_id', 'sha256', 'relative_image'):
        if {r[key] for r in queries} & {r[key] for r in train}: raise ValueError('Train/query overlap: ' + key)
    supports = {}
    for c in categories:
        normals = [r for r in unique.values() if r['category'] == c and not r['is_anomalous']]
        normals.sort(key=lambda r: sha(('jev-support-v1|42|' + r['source_id']).encode()))
        if len(normals) < 4: raise ValueError('Insufficient normal supports')
        supports[c] = [clean_row(r, audit) for r in normals[:4]]
    flat = [r for rows in supports.values() for r in rows]
    if len({r['sha256'] for r in flat}) != 48: raise ValueError('Duplicate support bytes')
    labels = {r['source_id']: int(r['is_anomalous']) for r in query_rows}
    plan = {'schema': SCHEMA, 'data_role': 'previously_viewed_development_validation_not_blind_test',
            'queries': queries, 'supports_by_category': supports,
            'support_selection': 'sha256(jev-support-v1|42|source_id) ascending; first 4; 1-shot first 1',
            'training_rows': len(train_rows), 'unique_training_sources': len(unique)}
    config = {'schema': SCHEMA, 'model': MODEL, 'shots': list(SHOTS), 'size': SIZE,
              'preprocessing': 'EXIF transpose; RGB; LANCZOS contain; centered white padding; PNG compress_level=6',
              'instructions': INSTRUCTIONS, 'criteria': CRITERIA, 'threshold': 0.5, 'ece_bins': 10,
              'bootstrap': {'seed': 42, 'replicates': 2000, 'strata': 'category_and_label'},
              'source_hashes': {k: file_sha(getattr(args, k)) for k in ('queries', 'train', 'audit')},
              'plan_sha256': sha(canonical(plan)), 'labels_sha256': sha(canonical(labels)),
              'runner_sha256': file_sha(__file__), 'versions': versions()}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, obj in [('plan', plan), ('labels', labels), ('config', config)]: immutable_write(args.output / (name + '.json'), obj)
    missing = [r['relative_image'] for r in queries + flat if args.image_root is None or not (args.image_root / r['relative_image']).is_file()]
    status = {'timestamp_utc': now(), 'queries': 120, 'supports': 48, 'expected_target_judgments': 360,
              'historical_identity_checks_passed': True, 'current_bytes_verified': False,
              'missing_images': missing, 'available_path_count_only': 168 - len(missing),
              'status': 'prepared_pending_materialization', 'confirmation_images_opened': False}
    write_json(args.output / 'prepare_status.json', status)
    print(json.dumps({'status': status['status'], 'queries': 120, 'supports': 48, 'missing': len(missing)}))

def load_frozen(plan_dir):
    config = read_json(plan_dir / 'config.json'); plan = read_json(plan_dir / 'plan.json'); labels = read_json(plan_dir / 'labels.json')
    if config['schema'] != SCHEMA or config['model'] != MODEL or config['shots'] != list(SHOTS): raise ValueError('Unexpected protocol')
    for name, obj in [('plan', plan), ('labels', labels)]:
        if sha(canonical(obj)) != config[name + '_sha256']: raise ValueError('Frozen artifact hash mismatch: ' + name)
    if config['runner_sha256'] != file_sha(__file__): raise ValueError('Runner changed after freeze; use new experiment directory')
    if config['versions'] != versions(): raise ValueError('Runtime versions changed after freeze')
    if config['instructions'] != INSTRUCTIONS or config['criteria'] != CRITERIA or config['size'] != SIZE:
        raise ValueError('Question or preprocessing differs from protocol')
    return config, plan, labels

def all_images(plan): return plan['queries'] + [r for rows in plan['supports_by_category'].values() for r in rows]
def processed_png(raw):
    with Image.open(io.BytesIO(raw)) as opened:
        im = ImageOps.exif_transpose(opened).convert('RGB')
        im = ImageOps.contain(im, (SIZE, SIZE), method=Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (SIZE, SIZE), (255, 255, 255))
        canvas.paste(im, ((SIZE - im.width) // 2, (SIZE - im.height) // 2))
        stream = io.BytesIO(); canvas.save(stream, format='PNG', compress_level=6); return stream.getvalue()

def materialize(args):
    config, plan, _ = load_frozen(args.plan_dir); dest = args.plan_dir / 'data_images'; dest.mkdir(exist_ok=True)
    records = []; failures = []
    for row in all_images(plan):
        source = args.image_root / row['relative_image']
        if not source.is_file(): failures.append({'source_id': row['source_id'], 'error': 'missing_image'}); continue
        raw = source.read_bytes()
        if sha(raw) != row['sha256'] or len(raw) != row['source_bytes']:
            failures.append({'source_id': row['source_id'], 'error': 'original_bytes_mismatch'}); continue
        try: png = processed_png(raw)
        except (OSError, ValueError): failures.append({'source_id': row['source_id'], 'error': 'invalid_image'}); continue
        filename = row['source_id'] + '.png'; target = dest / filename
        if target.exists() and target.read_bytes() != png: raise ValueError('Existing processed image differs')
        target.write_bytes(png)
        records.append({'source_id': row['source_id'], 'original_sha256': sha(raw), 'png_sha256': sha(png),
                        'png_bytes': len(png), 'image': 'data_images/' + filename, 'size': [SIZE, SIZE]})
    result = {'schema': SCHEMA, 'timestamp_utc': now(), 'status': 'complete' if not failures else 'blocked_missing_or_invalid_images',
              'config_sha256': sha(canonical(config)), 'records': records, 'failures': failures,
              'current_original_byte_verification': True, 'confirmation_images_opened': False}
    path = args.plan_dir / 'materialized.json'
    if path.exists() and read_json(path).get('status') == 'complete':
        old = read_json(path)
        if old['records'] != records or old['config_sha256'] != result['config_sha256']: raise ValueError('Completed materialization differs')
    write_json(path, result); print(json.dumps({'status': result['status'], 'verified': len(records), 'failures': len(failures)}))
    return 0 if not failures else 2

def verified_images(plan_dir, config, plan):
    doc = read_json(plan_dir / 'materialized.json')
    if doc['status'] != 'complete' or doc['failures'] or doc['config_sha256'] != sha(canonical(config)):
        raise ValueError('Images have not passed complete materialization')
    rows = {r['source_id']: r for r in doc['records']}
    if len(rows) != len(doc['records']) or set(rows) != {r['source_id'] for r in all_images(plan)}:
        raise ValueError('Unexpected or duplicate image identities')
    images = {}
    for source in all_images(plan):
        r = rows[source['source_id']]; path = plan_dir / r['image']
        if path.resolve().parent != (plan_dir / 'data_images').resolve(): raise ValueError('Unsafe image path')
        raw = path.read_bytes()
        if r['original_sha256'] != source['sha256'] or sha(raw) != r['png_sha256'] or len(raw) != r['png_bytes'] or r['size'] != [SIZE, SIZE]:
            raise ValueError('Processed image integrity mismatch')
        images[source['source_id']] = raw
    return doc, images

def request_payload(query, supports, shots, images, identity):
    assets = {}; refs = []
    def add(row, role):
        ref = str(uuid.uuid5(uuid.NAMESPACE_URL, identity + '|' + role))
        assets[ref] = {'type': 'image', 'format': 'png', 'data_base64': base64.b64encode(images[row['source_id']]).decode('ascii')}
        return ref
    qref = add(query, 'query')
    for i, r in enumerate(supports[:shots]): refs.append(add(r, 'reference-' + str(i)))
    return {'model': MODEL, 'state': {'query': qref, 'category': query['category'].replace('_', ' '), 'normal_references': refs},
            'assets': assets, 'questions': {'anomaly': {'type': 'noul', 'instructions': INSTRUCTIONS, 'criteria': CRITERIA}}}

def response_probability(data):
    if not isinstance(data, dict) or data.get('model') != MODEL: raise ValueError('Missing or mismatched response model')
    answers = data.get('answers')
    if not isinstance(answers, dict) or set(answers) != {'anomaly'}: raise ValueError('Unexpected question identities')
    answer = answers['anomaly']
    if not isinstance(answer, dict) or answer.get('type') != 'noul': raise ValueError('Response is not typed Noul')
    p = answer.get('noul')
    if type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1: raise ValueError('Invalid Noul probability')
    return float(p)

def safe_usage(data):
    usage = data.get('usage')
    if not isinstance(usage, dict): return {}
    return {k: v for k, v in usage.items() if isinstance(k, str) and k.endswith('_tokens') and
            type(v) in (float, int) and math.isfinite(v) and v >= 0}
def safe_request_id(value, key=None):
    if not isinstance(value, str) or len(value) > 200 or (key and key in value): return None
    return value if all(c.isalnum() or c in '-_.:' for c in value) else None
def schedule_for(plan, limit): return [(q, s) for q in (plan['queries'] if limit is None else plan['queries'][:limit]) for s in SHOTS]
def result_key(row): return row['source_id'], row['shots']
def validate_records(records, allowed, contract_hash):
    found = {}
    for row in records:
        key = result_key(row)
        if key not in allowed or key in found: raise ValueError('Unexpected or duplicate prediction identity')
        if row.get('contract_sha256') != contract_hash: raise ValueError('Prediction contract mismatch')
        response_probability({'model': row.get('response_model'), 'answers': {'anomaly': {'type': 'noul', 'noul': row.get('probability')}}})
        digest = row.get('request_sha256')
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest): raise ValueError('Malformed prediction request hash')
        if type(row.get('seconds')) not in (float, int) or not math.isfinite(row['seconds']) or row['seconds'] < 0: raise ValueError('Invalid persisted latency')
        if not isinstance(row.get('usage'), dict) or safe_usage({'usage': row['usage']}) != row['usage']: raise ValueError('Invalid persisted usage')
        found[key] = row
    return found

def run_impl(args):
    endpoint = validate_endpoint(args.endpoint); config, plan, _ = load_frozen(args.plan_dir)
    materialized, images = verified_images(args.plan_dir, config, plan)
    if args.limit_queries is not None and not 1 <= args.limit_queries <= len(plan['queries']): raise ValueError('limit-queries outside range')
    schedule = schedule_for(plan, args.limit_queries)
    mode = 'formal_development_baseline' if args.limit_queries is None else 'preflight_subset_not_formal_baseline'
    contract = {'schema': SCHEMA, 'endpoint': endpoint, 'mode': mode, 'limit_queries': args.limit_queries,
        'config_sha256': sha(canonical(config)), 'materialized_records_sha256': sha(canonical(materialized['records'])),
        'runner_sha256': file_sha(__file__), 'versions': versions(), 'workers': args.workers, 'timeout': args.timeout,
        'schedule': [{'source_id': q['source_id'], 'shots': s} for q, s in schedule]}
    contract_hash = sha(canonical(contract)); args.output.mkdir(parents=True, exist_ok=True)
    immutable_write(args.output / 'contract.json', contract)
    predictions = args.output / 'predictions.jsonl'; attempts = args.output / 'attempts.jsonl'
    allowed = {(q['source_id'], s) for q, s in schedule}
    completed = validate_records(jsonl(predictions) if predictions.exists() else [], allowed, contract_hash)
    for q, shots in schedule:
        pair = q['source_id'], shots
        if pair in completed:
            body = request_payload(q, plan['supports_by_category'][q['category']], shots, images, contract_hash + '|' + q['source_id'] + '|' + str(shots))
            if completed[pair]['request_sha256'] != sha(canonical(body)) or completed[pair].get('category') != q['category']:
                raise ValueError('Persisted request or category differs from frozen inputs')
    previous = jsonl(attempts) if attempts.exists() else []
    if any(r.get('contract_sha256') != contract_hash or result_key(r) not in allowed for r in previous): raise ValueError('Attempt contract mismatch')
    attempted = Counter(result_key(r) for r in previous if r.get('event') == 'started')
    if any(k not in completed for k in attempted) and not args.retry_failed:
        raise ValueError('Unresolved attempts: inspect log; explicit --retry-failed may cause duplicate billing')
    if args.dry_run:
        print(json.dumps({'mode': mode, 'expected': len(schedule), 'completed': len(completed), 'api_calls': 0})); return 0
    key = os.environ.get(args.key_env) or getpass.getpass('API key (not persisted): ')
    if not key: raise ValueError('Missing credential')
    consecutive = 0
    pending = [(q, shots) for q, shots in schedule if (q['source_id'], shots) not in completed]
    def send(body):
        start = time.perf_counter(); row = None; error = None; status = None
        try:
            response = requests.post(endpoint, json=body, headers={'Authorization': 'Bearer ' + key,
                'TypeSafe-Danger-Zone-Images': 'true'}, timeout=(15, args.timeout), allow_redirects=False)
            status = response.status_code
            if status != 200: error = 'http_' + str(status)
            else:
                data = response.json(); probability = response_probability(data)
                row = {'probability': probability, 'response_model': data['model'], 'usage': safe_usage(data),
                    'request_id': safe_request_id(response.headers.get('x-request-id') or response.headers.get('request-id') or data.get('id'), key)}
        except requests.Timeout: error = 'timeout_unknown_server_completion_no_automatic_retry'
        except requests.RequestException: error = 'transport_error_unknown_server_completion_no_automatic_retry'
        except (ValueError, KeyError, TypeError): error = 'invalid_typed_response_or_model'
        return row, error, status, time.perf_counter() - start
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for offset in range(0, len(pending), args.workers):
                batch = pending[offset:offset + args.workers]; futures = {}
                for query, shots in batch:
                    pair = query['source_id'], shots
                    body = request_payload(query, plan['supports_by_category'][query['category']], shots, images,
                                           contract_hash + '|' + query['source_id'] + '|' + str(shots))
                    digest = sha(canonical(body))
                    attempt = {'source_id': query['source_id'], 'shots': shots, 'contract_sha256': contract_hash,
                        'request_sha256': digest, 'timestamp_utc': now(), 'attempt': attempted[pair] + 1, 'event': 'started'}
                    append_json(attempts, attempt); attempted[pair] += 1
                    futures[pool.submit(send, body)] = (query, shots, attempt)
                for future in as_completed(futures):
                    query, shots, attempt = futures[future]; pair = query['source_id'], shots
                    response_row, error, status, elapsed = future.result()
                    if response_row is not None:
                        row = {**response_row, 'source_id': query['source_id'], 'shots': shots, 'category': query['category'],
                            'request_sha256': attempt['request_sha256'], 'contract_sha256': contract_hash,
                            'timestamp_utc': now(), 'seconds': elapsed}
                        append_json(predictions, row); completed[pair] = row; consecutive = 0
                        print(json.dumps({'completed': len(completed), 'expected': len(schedule), 'shots': shots, 'seconds': elapsed}), flush=True)
                    else:
                        append_json(attempts, {**attempt, 'event': 'failed', 'error': error, 'http_status': status, 'seconds': elapsed})
                        consecutive += 1; print(json.dumps({'error': error, 'http_status': status}), flush=True)
                    write_json(args.output / 'status.json', {'timestamp_utc': now(), 'mode': mode, 'expected': len(schedule),
                        'completed': len(completed), 'status': 'complete' if len(completed) == len(schedule) else 'incomplete',
                        'model_version_note': 'ModelGo logical model echo is not independent proof of upstream version.'})
                if consecutive >= args.max_consecutive_errors: break
    finally: key = None
    return 0 if len(completed) == len(schedule) else 2

def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.run.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('Another runner holds this output directory')
        try: return run_impl(args)
        finally: fcntl.flock(lock, fcntl.LOCK_UN)

def metrics(labels, probabilities):
    y = np.asarray(labels, dtype=int); p = np.asarray(probabilities, dtype=float)
    if len(y) == 0 or len(y) != len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)): raise ValueError('Invalid metric inputs')
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2: raise ValueError('Need both binary classes')
    pred = p >= 0.5; bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(np.mean(bins == b) * abs(float(np.mean(y[bins == b])) - float(np.mean(p[bins == b]))) for b in range(10) if np.any(bins == b))
    normals = y == 0; abnormal = y == 1
    return {'n': len(y), 'normal': int(normals.sum()), 'anomalous': int(abnormal.sum()),
        'auroc': float(roc_auc_score(y, p)), 'ap': float(average_precision_score(y, p)),
        'macro_f1': float(f1_score(y, pred, average='macro', labels=[0, 1], zero_division=0)),
        'recall': float(np.mean(pred[abnormal])), 'fpr': float(np.mean(pred[normals])),
        'brier': float(brier_score_loss(y, p)), 'ece_10_equal_width': float(ece)}

def evaluate(args):
    config, plan, labels = load_frozen(args.plan_dir); contract = read_json(args.output / 'contract.json')
    doc, images = verified_images(args.plan_dir, config, plan)
    if contract['config_sha256'] != sha(canonical(config)) or contract['runner_sha256'] != file_sha(__file__) or contract['materialized_records_sha256'] != sha(canonical(doc['records'])):
        raise ValueError('Evaluation contract mismatch')
    validate_endpoint(contract['endpoint']); contract_hash = sha(canonical(contract)); schedule = schedule_for(plan, contract['limit_queries'])
    if contract['schedule'] != [{'source_id': q['source_id'], 'shots': s} for q, s in schedule]: raise ValueError('Schedule mismatch')
    allowed = {(q['source_id'], s) for q, s in schedule}; predictions = args.output / 'predictions.jsonl'
    rows = validate_records(jsonl(predictions) if predictions.exists() else [], allowed, contract_hash)
    for q, shots in schedule:
        pair = q['source_id'], shots
        if pair in rows:
            body = request_payload(q, plan['supports_by_category'][q['category']], shots, images, contract_hash + '|' + q['source_id'] + '|' + str(shots))
            if rows[pair]['request_sha256'] != sha(canonical(body)) or rows[pair].get('category') != q['category']:
                raise ValueError('Prediction request differs from frozen inputs')
    del images
    if len(rows) != len(allowed) or len(allowed) != 360 or contract['mode'] != 'formal_development_baseline':
        status = {'status': 'incomplete_or_preflight_not_formal', 'completed': len(rows), 'expected_for_this_run': len(allowed), 'required_for_formal': 360, 'formal_metrics_produced': False}
        write_json(args.output / 'partial_status.json', status); print(json.dumps(status)); return 2
    queries = plan['queries']; y = np.asarray([labels[q['source_id']] for q in queries], dtype=int)
    probs = {s: np.asarray([rows[q['source_id'], s]['probability'] for q in queries]) for s in SHOTS}
    summary = {'status': 'complete', 'data_role': plan['data_role'], 'model': MODEL, 'threshold': 0.5, 'shots': {}, 'by_category': {}, 'paired_bootstrap': {},
        'cost': 'Not estimated; no verified currency-cost field.', 'version_limit': 'Gateway logical echo does not independently verify upstream model version.'}
    for s in SHOTS:
        subset = [r for r in rows.values() if r['shots'] == s]; seconds = np.asarray([r['seconds'] for r in subset]); keys = sorted({k for r in subset for k in r['usage']})
        summary['shots'][str(s)] = {**metrics(y, probs[s]), 'latency_seconds': {'mean': float(seconds.mean()), 'median': float(np.median(seconds)), 'p95': float(np.percentile(seconds, 95))},
            'usage_reported_totals': {k: sum(r['usage'].get(k, 0) for r in subset) for k in keys}, 'usage_present_requests': sum(bool(r['usage']) for r in subset)}
    for c in sorted(plan['supports_by_category']):
        idx = np.asarray([i for i, q in enumerate(queries) if q['category'] == c], dtype=int)
        summary['by_category'][c] = {str(s): metrics(y[idx], probs[s][idx]) for s in SHOTS}
    strata = defaultdict(list)
    for i, q in enumerate(queries): strata[q['category'], int(y[i])].append(i)
    strata = [np.asarray(v) for _, v in sorted(strata.items())]; rng = np.random.default_rng(42)
    diffs = {(a, b, metric): [] for a, b in ((1, 0), (4, 0), (4, 1)) for metric in ('auroc', 'ap')}
    for _ in range(2000):
        idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        scores = {s: (float(roc_auc_score(y[idx], probs[s][idx])), float(average_precision_score(y[idx], probs[s][idx]))) for s in SHOTS}
        for a, b in ((1, 0), (4, 0), (4, 1)):
            for j, metric in enumerate(('auroc', 'ap')): diffs[a, b, metric].append(scores[a][j] - scores[b][j])
    for a, b in ((1, 0), (4, 0), (4, 1)):
        summary['paired_bootstrap'][str(a) + '-' + str(b)] = {metric: {'point_difference': summary['shots'][str(a)][metric] - summary['shots'][str(b)][metric],
            'percentile_95_ci': np.percentile(diffs[a, b, metric], [2.5, 97.5]).tolist(), 'replicates': 2000, 'seed': 42, 'strata': 'category_and_label',
            'limitation': 'Fixed-query resampling excludes provider randomness and support selection variation.'} for metric in ('auroc', 'ap')}
    summary['prediction_file_sha256'] = file_sha(predictions); write_json(args.output / 'metrics.json', summary)
    lines = ['# Jev 0/1/4 正常参考图开发基线', '', '同一批120张已用于开发的验证图；不是新盲测。阈值0.5，Noul连续概率。', '',
        '| 正常参考数 | AUROC | AP | Macro-F1 | Recall | FPR | Brier | ECE | 平均秒/图 |', '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in SHOTS:
        m = summary['shots'][str(s)]; vals = [str(s)] + [format(m[k], '.6f') for k in ('auroc', 'ap', 'macro_f1', 'recall', 'fpr', 'brier', 'ece_10_equal_width')] + [format(m['latency_seconds']['mean'], '.4f')]
        lines.append('| ' + ' | '.join(vals) + ' |')
    lines += ['', '## 配对差值（百分点）', '', '| 比较 | 指标 | 点估计 | 95%区间 |', '|---|---|---:|---|']
    for comparison, values in summary['paired_bootstrap'].items():
        for metric, r in values.items():
            low, high = r['percentile_95_ci']; lines.append('| {} | {} | {:.3f} | [{:.3f}, {:.3f}] |'.format(comparison, metric, r['point_difference'] * 100, low * 100, high * 100))
    lines += ['', '逐类指标、usage与延迟分位数见metrics.json。区间不包含训练/服务随机性与支持集变化，未做多重比较校正。',
        'ModelGo逻辑模型名回显不独立证明上游版本；无可靠货币费用字段，未估造成本。', '']
    (args.output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status': 'complete', 'predictions': len(rows), 'metrics': str(args.output / 'metrics.json')})); return 0

def parser():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('prepare'); q.add_argument('--queries', type=Path, required=True); q.add_argument('--train', type=Path, required=True); q.add_argument('--audit', type=Path, required=True)
    q.add_argument('--output', type=Path, default=ROOT / 'frozen'); q.add_argument('--image-root', type=Path); q.set_defaults(func=prepare)
    q = sub.add_parser('materialize'); q.add_argument('--plan-dir', type=Path, default=ROOT / 'frozen'); q.add_argument('--image-root', type=Path, required=True); q.set_defaults(func=materialize)
    q = sub.add_parser('run'); q.add_argument('--plan-dir', type=Path, default=ROOT / 'frozen'); q.add_argument('--endpoint', required=True); q.add_argument('--output', type=Path, required=True)
    q.add_argument('--key-env', default='TYPESAFE_API_KEY'); q.add_argument('--limit-queries', type=int); q.add_argument('--timeout', type=float, default=90)
    q.add_argument('--workers', type=int, default=1); q.add_argument('--max-consecutive-errors', type=int, default=1); q.add_argument('--retry-failed', action='store_true'); q.add_argument('--dry-run', action='store_true'); q.set_defaults(func=run)
    q = sub.add_parser('evaluate'); q.add_argument('--plan-dir', type=Path, default=ROOT / 'frozen'); q.add_argument('--output', type=Path, required=True); q.set_defaults(func=evaluate)
    return p

def main():
    args = parser().parse_args()
    try:
        if getattr(args, 'timeout', 1) <= 0 or getattr(args, 'max_consecutive_errors', 1) < 1 or not 1 <= getattr(args, 'workers', 1) <= 8: raise ValueError('Invalid timeout or error limit')
        return args.func(args) or 0
    except (ValueError, KeyError, OSError) as exc:
        print('ERROR: ' + str(exc) if isinstance(exc, ValueError) else 'ERROR: ' + type(exc).__name__, file=sys.stderr); return 2
if __name__ == '__main__': raise SystemExit(main())
