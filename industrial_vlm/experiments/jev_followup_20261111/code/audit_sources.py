"""Server-only audit of the frozen Jev sources; no model or GT access."""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(str(path))
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--images', required=True, type=Path)
    args = parser.parse_args()
    if socket.gethostname() != 'autodl-container-132444827f-9c594038':
        raise RuntimeError('This audit must execute on the designated server')
    root = args.root.resolve()
    frozen = root / 'source_baseline/frozen'
    plan = json.loads((frozen / 'plan.json').read_text())
    queries = plan['queries']
    supports = [r for group in plan['supports_by_category'].values() for r in group]
    assert len(queries) == 120 and len(supports) == 48
    assert not ({r['source_id'] for r in queries} & {r['source_id'] for r in supports})
    records = []
    for role, rows in [('query', queries), ('support', supports)]:
        for row in rows:
            relative = Path(row['relative_image'])
            assert not relative.is_absolute() and '..' not in relative.parts
            image = (args.images / relative).resolve()
            assert image.is_relative_to(args.images.resolve())
            actual = digest(image) if image.is_file() else None
            size = image.stat().st_size if image.is_file() else None
            records.append(dict(source_id=row['source_id'], category=row['category'],
                role=role, path=str(image), sha256=actual, bytes=size,
                expected_sha256=row['sha256'], expected_bytes=row['source_bytes'],
                passed=(actual == row['sha256'] and size == row['source_bytes'])))
    result = dict(schema='jev-followup-source-audit-v1',
        time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        hostname=socket.gethostname(), plan_sha256=digest(frozen/'plan.json'),
        script_sha256=digest(__file__), total=len(records),
        passed=sum(r['passed'] for r in records),
        counts_by_role=dict(collections.Counter(r['role'] for r in records)),
        query_categories=dict(collections.Counter(r['category'] for r in records if r['role']=='query')),
        source_overlap=0, confirmation_opened=False, ground_truth_masks_opened=False,
        status='complete' if all(r['passed'] for r in records) else 'failed', records=records)
    publish(root/'stages/01_source_audit.json', result)
    summary = {k:v for k,v in result.items() if k!='records'}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if result['status'] != 'complete':
        raise SystemExit(2)

if __name__ == '__main__':
    main()
