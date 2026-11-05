"""Fail-fast serial experiment runner; writes a reviewable stage status."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path('/root/autodl-tmp/AnomalyGPT')
ROOT = REPO / 'runs/migration_20261103'
PYTHON = '/root/autodl-tmp/conda-envs/qwen3vl2/bin/python'


def save(value, path):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['smoke', 'phase1_seed42', 'phase1_repeats', 'native_parity',
                                       'native_smoke', 'native_main', 'visual_probe', 'dense_gt', 'roi_pilot',
                                       'roi_full', 'dense_kd'], required=True)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '8')
    os.environ.setdefault('MKL_NUM_THREADS', '8')
    if Path('/dev/shm/migration_vicuna_bf16/model.safetensors.index.json').exists():
        os.environ['VICUNA_BF16_MODEL'] = '/dev/shm/migration_vicuna_bf16'
    ROOT.mkdir(exist_ok=True, parents=True)
    with open(ROOT / 'gpu.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = []
        if args.stage == 'native_parity':
            jobs.append(('native_parity' + args.tag, [PYTHON, '-u', 'migration_native_parity.py', '--run',
                                          '--out', str(ROOT / 'native_parity/report.json')]))
        elif args.stage in ('native_smoke', 'native_main'):
            names = [('native_smoke_s42', ['--smoke'])] if args.stage == 'native_smoke' else [
                ('native_baseline', ['--baseline']), ('native_map_s42', [])]
            for name, extra in names:
                name += args.tag
                jobs.append((name, [PYTHON, '-u', 'migration_native_train.py',
                                   '--parity-report', str(ROOT / 'native_parity/report.json'),
                                   '--out', str(ROOT / name), *extra]))
        elif args.stage == 'visual_probe':
            name = 'visual_probe' + args.tag
            jobs.append((name, [PYTHON, '-u', 'migration_visual_probe.py', '--run',
                               '--out', str(ROOT / name / 'val.json')]))
        elif args.stage in ('dense_gt', 'dense_kd'):
            # The 224 controls separate feature-source changes from resolution.
            # This cache uses CPU RAM and does not create feature files.
            configurations = [('imagebind', 224), ('pre', 224), ('post', 224),
                              ('pre', 448), ('post', 448)] if args.stage == 'dense_gt' else [
                                  ('pre', 448), ('post', 448)]
            for feature, size in configurations:
                name = ('dense_' + feature + ('_224' if size == 224 and feature != 'imagebind' else '')
                        + ('_kd' if args.stage == 'dense_kd' else '_gt') + '_s42' + args.tag)
                cmd = [PYTHON, '-u', 'migration_dense_train.py', '--run',
                       '--features', feature, '--size', str(size), '--cache-gb', '8',
                       '--out', str(ROOT / name)]
                if args.stage == 'dense_kd':
                    cmd += ['--distill-weight', '0.5', '--teacher-maps',
                            str(ROOT / 'teacher_train_subset/manifest.json')]
                jobs.append((name, cmd))
        elif args.stage in ('roi_pilot', 'roi_full'):
            if args.stage == 'roi_pilot':
                preflight = 'roi_preflight' + args.tag
                jobs.append((preflight, [PYTHON, '-u', 'migration_roi_preflight.py', '--run',
                                         '--parity-report', str(ROOT / 'native_parity/report.json'),
                                         '--out', str(ROOT / preflight)]))
            name = args.stage + args.tag
            jobs.append((name, [PYTHON, '-u', 'migration_roi_eval.py', '--run',
                               '--limit', '48' if args.stage == 'roi_pilot' else '0',
                               '--with-reference', '--parity-report', str(ROOT / 'native_parity/report.json'),
                               '--out', str(ROOT / name)]))
        else:
            seeds = [43, 44] if args.stage == 'phase1_repeats' else [42]
            for seed in seeds:
                for backend in ('vicuna', 'qwen'):
                    name = ('smoke_' if args.stage == 'smoke' else 'lm_') + backend + '_s' + str(seed) + args.tag
                    cmd = [PYTHON, '-u', 'migration_train.py', '--backend', backend, '--seed', str(seed),
                           '--out', str(ROOT / name), '--batch-size', '4', '--accum', '1', '--eval-split', 'val']
                    if args.stage == 'smoke':
                        cmd += ['--steps', '4', '--val-every', '4', '--val-limit', '24',
                                '--generate-limit', '8', '--skip-map-controls']
                    jobs.append((name, cmd))
        status_path = ROOT / (args.stage + args.tag + '_status.json')
        if status_path.exists():
            raise FileExistsError('Stage has already been dispatched: ' + str(status_path))
        status = dict(stage=args.stage, started_at=time.strftime('%Y-%m-%d %H:%M:%S %z'),
                      state='waiting_for_teacher_cache', jobs=[])
        save(status, status_path)
        while not (ROOT / 'teacher_cache.json').exists():
            time.sleep(5)
        for name, cmd in jobs:
            record = dict(name=name, command=cmd, state='running', started_at=time.strftime('%Y-%m-%d %H:%M:%S %z'))
            status.update(state='running', current_job=name)
            status['jobs'].append(record)
            save(status, status_path)
            print('[start]', name, flush=True)
            begin = time.time()
            with open(ROOT / (name + '.log'), 'xb') as log:
                result = subprocess.run(cmd, cwd=REPO, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT)
            record.update(returncode=result.returncode, seconds=time.time()-begin,
                          state='complete' if result.returncode == 0 else 'failed')
            save(status, status_path)
            if result.returncode:
                status['state'] = 'failed'
                save(status, status_path)
                raise SystemExit(result.returncode)
            print('[complete]', name, record['seconds'], flush=True)
        status.update(state='complete', completed_at=time.strftime('%Y-%m-%d %H:%M:%S %z'))
        save(status, status_path)
        print('[stage complete]', args.stage, flush=True)


if __name__ == '__main__':
    main()
