"""Fail-fast serial B/C queue with explicit preflight gates and GPU lock."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def atomic_json(obj, path):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='/root/autodl-tmp/AnomalyGPT')
    ap.add_argument('--out', default='runs/migration_20261103/lora_20261104')
    ap.add_argument('--preflight-only', action='store_true')
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    os.chdir(repo)
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('OMP_NUM_THREADS', '8')
    os.environ.setdefault('MKL_NUM_THREADS', '8')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    lock_path = repo / 'runs/migration_20261103/gpu.lock'
    with open(lock_path, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(state='running', stage='base_identity', started=time.time(),
                      runner_pid=os.getpid(), jobs=[], test_queries=0)
        atomic_json(status, root / 'status.json')
        try:
            base = Path(os.environ.get('QWEN3VL_MODEL',
                        '/root/autodl-tmp/industrial_vlm/models/Qwen3-VL-8B-Instruct'))
            base_files = sorted(set(list(base.glob('*.json')) + list(base.glob('*.safetensors'))))
            manifest = dict(model_path=str(base), files={str(p.name): dict(
                bytes=p.stat().st_size, sha256=digest(p)) for p in base_files})
            assert any(p.suffix == '.safetensors' for p in base_files)
            atomic_json(manifest, root / 'base_manifest.json')
            sources = ['migration_lora_train.py', 'migration_lora_runner.py',
                       'migration_language.py', 'migration_metrics.py', 'migration_train.py',
                       'dec_big.py', 'visa_pipeline.py', 'split_visa.py',
                       'code/code/model/AnomalyGPT_models.py']
            atomic_json({p:dict(sha256=digest(repo / p), bytes=(repo / p).stat().st_size)
                         for p in sources}, root / 'source_manifest.json')
            jobs = [('preflight_B_s42', 'B', 42, True), ('preflight_C_s42', 'C', 42, True)]
            if not args.preflight_only:
                jobs += [('arm_%s_s%d' % (arm, seed), arm, seed, False)
                         for seed in (42, 43, 44) for arm in ('B', 'C')]
            for name, arm, seed, preflight in jobs:
                target = root / name
                if (target / 'complete.json').exists():
                    result = json.load(open(target / 'complete.json'))
                    assert result['steps'] == (40 if preflight else 500)
                    assert result['arm'] == arm and result['seed'] == seed
                    status['jobs'].append(dict(name=name, state='already_complete'))
                    continue
                if not preflight:
                    for gate in ('preflight_B_s42', 'preflight_C_s42'):
                        result = json.load(open(root / gate / 'complete.json'))
                        assert result['preflight'] and result['reload_max_abs_score_error'] == 0.
                        assert result['zero_adapter_max_abs_logit_error'] == 0.
                        assert result['last5_mean_loss'] < result['first5_mean_loss']
                cmd = [sys.executable, '-u', 'migration_lora_train.py', '--arm', arm,
                       '--seed', str(seed), '--out', str(target),
                       '--cache', str(repo / 'runs/migration_20261103/teacher_cache.pt'),
                       '--bridge', str(repo / ('runs/migration_20261103/lm_qwen_s%d/best.pt' % seed)),
                       '--base-manifest', str(root / 'base_manifest.json')]
                if preflight:
                    cmd += ['--preflight', '--steps', '40', '--val-every', '20']
                status.update(stage=name, command=cmd, stage_started=time.time())
                atomic_json(status, root / 'status.json')
                print('[start]', name, flush=True)
                with open(root / (name + '.log'), 'w') as log:
                    child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                    status['child_pid'] = child.pid
                    atomic_json(status, root / 'status.json')
                    rc = child.wait()
                if rc:
                    raise RuntimeError('%s failed, exit %d; see %s.log' % (name, rc, name))
                result = json.load(open(target / 'complete.json'))
                assert result['steps'] == (40 if preflight else 500)
                status['jobs'].append(dict(name=name, state='complete', result=result))
                atomic_json(status, root / 'status.json')
                print('[done]', name, flush=True)
            status.update(state='complete', stage='complete', finished=time.time())
        except Exception as exc:
            status.update(state='failed', error=str(exc), traceback=traceback.format_exc(),
                          finished=time.time())
            raise
        finally:
            atomic_json(status, root / 'status.json')


if __name__ == '__main__':
    main()
