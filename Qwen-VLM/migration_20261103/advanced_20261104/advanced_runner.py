"""Serial D and paired ranking-loss experiments on the shared single GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def atomic_json(value, path):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='/root/autodl-tmp/AnomalyGPT')
    ap.add_argument('--root', default='runs/migration_20261103/advanced_20261104')
    ap.add_argument('--rank-weight', type=float, default=None)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    root = (repo / args.root).resolve()
    os.chdir(repo)
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('OMP_NUM_THREADS', '8')
    os.environ.setdefault('MKL_NUM_THREADS', '8')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    status_path = root / 'status.json'
    lock_path = repo / 'runs/migration_20261103/gpu.lock'
    with open(lock_path, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(state='running', stage='start', started=time.time(), rank_weight=args.rank_weight,
                      jobs=[], test_queries=0, external_queries=0)
        atomic_json(status, status_path)
        try:
            if args.rank_weight is None:
                calibration = root / 'rank_weight_calibration.json'
                cmd = [sys.executable, '-u', 'calibrate_rank_weight.py',
                       '--cache', str(repo / 'runs/migration_20261103/teacher_cache.pt'),
                       '--bridge', str(repo / 'runs/migration_20261103/lm_qwen_s42/best.pt'),
                       '--base-manifest', str(repo / 'runs/migration_20261103/lora_20261104/base_manifest.json'),
                       '--out', str(calibration)]
                status.update(stage='rank_weight_calibration', command=cmd, stage_started=time.time())
                atomic_json(status, status_path)
                with open(root / 'rank_weight_calibration.log', 'w') as log:
                    code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
                if code:
                    raise RuntimeError('Train-only ranking weight calibration failed')
                args.rank_weight = json.loads(calibration.read_text())['fixed_rank_weight']
                status['rank_weight'] = args.rank_weight
                atomic_json(status, status_path)
            phases = [
                [('preflight_D_s42', 'D', 42, True, False),
                 *[('arm_D_s%d' % seed, 'D', seed, False, False) for seed in (42, 43, 44)]],
                [('preflight_Cpair_s42', 'C', 42, True, True),
                 ('preflight_R_s42', 'R', 42, True, True),
                 *[('%s_s%d' % (arm, seed), arm, seed, False, True)
                   for seed in (42, 43, 44) for arm in ('Cpair', 'R')]],
            ]
            for phase in phases:
                for name, arm_name, seed, preflight, paired in phase:
                    arm = 'C' if arm_name == 'Cpair' else arm_name
                    out = root / name
                    if (out / 'complete.json').exists():
                        result = json.loads((out / 'complete.json').read_text())
                        assert result['preflight'] == preflight and result['seed'] == seed
                        status['jobs'].append(dict(name=name, state='already_complete'))
                        atomic_json(status, status_path)
                        continue
                    cmd = [sys.executable, '-u', 'migration_advanced_train.py', '--arm', arm,
                           '--seed', str(seed), '--out', str(out),
                           '--cache', str(repo / 'runs/migration_20261103/teacher_cache.pt'),
                           '--bridge', str(repo / ('runs/migration_20261103/lm_qwen_s%d/best.pt' % seed)),
                           '--base-manifest', str(repo / 'runs/migration_20261103/lora_20261104/base_manifest.json')]
                    if arm == 'D':
                        cmd += ['--lr-bridge', '1e-5', '--lr-lora', '1e-4']
                    if paired:
                        cmd += ['--paired-sampling', '--microbatch', '2', '--accum', '2']
                    if arm == 'R':
                        cmd += ['--objective', 'ce_pair_rank', '--rank-weight', str(args.rank_weight)]
                    if preflight:
                        cmd += ['--preflight', '--steps', '40', '--val-every', '20']
                    else:
                        for gate in (('preflight_D_s42',) if arm == 'D' else ('preflight_Cpair_s42', 'preflight_R_s42')):
                            if gate not in [j['name'] for j in status['jobs']] and not (root / gate / 'complete.json').exists():
                                raise RuntimeError('Missing preflight gate: ' + gate)
                            check = json.loads((root / gate / 'complete.json').read_text())
                            assert check['preflight'] and check['steps'] == 40
                            assert check['reload_max_abs_score_error'] == 0.
                            assert check['zero_adapter_max_abs_logit_error'] == 0.
                            assert check['frozen_parameter_versions_unchanged']
                            assert check['last5_mean_loss'] < check['first5_mean_loss']
                    status.update(stage=name, command=cmd, stage_started=time.time(), child_pid=None)
                    atomic_json(status, status_path)
                    print('[start]', name, flush=True)
                    with open(root / (name + '.log'), 'w') as log:
                        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                        status['child_pid'] = child.pid
                        atomic_json(status, status_path)
                        code = child.wait()
                    if code:
                        raise RuntimeError('%s failed with exit code %d' % (name, code))
                    result = json.loads((out / 'complete.json').read_text())
                    expected_steps = 40 if preflight else 500
                    assert result['steps'] == expected_steps and result['test_queries'] == 0
                    status['jobs'].append(dict(name=name, state='complete', result=result))
                    atomic_json(status, status_path)
                    print('[done]', name, flush=True)
            status.update(state='complete', stage='complete', finished=time.time())
        except Exception as error:
            status.update(state='failed', error=str(error), traceback=traceback.format_exc(), finished=time.time())
            raise
        finally:
            atomic_json(status, status_path)


if __name__ == '__main__':
    main()
