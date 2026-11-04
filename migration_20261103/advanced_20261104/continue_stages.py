"""Run S and native ROI serially after verified D/ranking completion.

An existing marker is accepted only after its full stage contract passes.
Incomplete/invalid output directories are preserved and fail explicitly; no
failed evidence is overwritten or silently treated as completed work.
"""
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

SCHEMA = 'visa-yesno-bbox1000-grid3-v1'
TOOL_CONDITIONS = ('no_tool', 'duplicate_original', 'center_crop', 'random_crop', 'model_tool')
FORMAL_PRIOR = (['arm_D_s%d' % seed for seed in (42, 43, 44)]
                + ['%s_s%d' % (arm, seed) for seed in (42, 43, 44) for arm in ('Cpair', 'R')])
PREFLIGHT_PRIOR = ('preflight_D_s42', 'preflight_Cpair_s42', 'preflight_R_s42')


def write(value, path):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def read(path):
    with Path(path).open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_prior(root):
    status, verification = read(root / 'status.json'), read(root / 'verification_status.json')
    require(status.get('state') == verification.get('state') == 'complete', 'Prior queues are not complete')
    jobs = status.get('jobs', [])
    expected = set(FORMAL_PRIOR) | set(PREFLIGHT_PRIOR)
    require(len(jobs) == 12 and {job.get('name') for job in jobs} == expected,
            'Prior training must contain exactly the twelve expected jobs')
    require(all(job.get('state') in ('complete', 'already_complete') for job in jobs), 'Prior job is not complete')
    for name in expected:
        result = read(root / name / 'complete.json')
        seed = int(name.rsplit('s', 1)[1])
        preflight = name.startswith('preflight_')
        arm = 'D' if '_D_' in name else 'R' if name.startswith(('R_', 'preflight_R_')) else 'C'
        require(result.get('seed') == seed and result.get('arm') == arm
                and result.get('preflight') is preflight
                and result.get('steps') == (40 if preflight else 500), 'Prior identity/step mismatch: ' + name)
        require(result.get('test_queries') == 0 and result.get('reload_max_abs_score_error') == 0
                and result.get('zero_adapter_max_abs_logit_error') == 0
                and result.get('frozen_parameter_versions_unchanged') is True, 'Prior engineering gate failed: ' + name)
        if preflight:
            require(result.get('last5_mean_loss', float('inf')) < result.get('first5_mean_loss', -float('inf')),
                    'Prior fixed-example fit gate failed: ' + name)
    checks = verification.get('checks', [])
    require(len(checks) == 9 and {check.get('run') for check in checks} == set(FORMAL_PRIOR),
            'Prior verification must contain the nine expected formal runs')
    require(all(check.get('state') == 'passed' and check.get('n', 0) > 0
                and check.get('both_modes_verified') is True for check in checks),
            'A prior fresh-process check did not pass')
    for name in FORMAL_PRIOR:
        complete = read(root / name / 'complete.json')
        for mode in ('final', 'best'):
            result = read(root / name / ('fresh_' + mode + '_reload.json'))
            expected_step = 500 if mode == 'final' else complete['best_step']
            require(result.get('state') == 'complete' and result.get('checkpoint') == mode
                    and result.get('seed') == int(name.rsplit('s', 1)[1])
                    and result.get('arm') == ('D' if name.startswith('arm_D_') else 'R' if name.startswith('R_') else 'C')
                    and result.get('step') == expected_step and result.get('n', 0) > 0
                    and result.get('both_modes_verified') is True
                    and result.get('original_batch_composition_preserved') is True
                    and result.get('fresh_reload_max_abs_score_error') == 0
                    and result.get('adapter_disabled_max_abs_score_error') == 0
                    and result.get('generated_exact_match') is True
                    and (result.get('adapter_B_nonzero') is True or (mode == 'best' and expected_step == 0))
                    and result.get('checkpoint_sha256') == sha256(root / name / (mode + '.pt')),
                    'Prior fresh reload marker failed: ' + name + ' ' + mode)



def validate_targets(targets, cache):
    meta = read(targets.with_suffix('.json'))
    require(meta.get('schema') == SCHEMA and meta.get('manifest_sha256') == sha256(targets)
            and meta.get('cache_sha256') == sha256(cache) and meta.get('test_mask_reads') == 0,
            'Structured target metadata/hash gate failed')
    rows = [json.loads(line) for line in targets.read_text().splitlines() if line.strip()]
    require(len(rows) == 10365 and len({row['index'] for row in rows}) == 10365,
            'Structured targets must cover train9985 + val380 exactly')
    counts = {split: sum(row['split'] == split for row in rows) for split in ('train', 'val')}
    require(counts == {'train': 9985, 'val': 380} and all(row['split'] in counts for row in rows),
            'Structured target split counts differ')


def validate_structured(out, seed, preflight):
    complete, run = read(out / 'complete.json'), read(out / 'run.json')
    expected_steps, expected_n = (40, 8) if preflight else (500, 380)
    cfg, identity = run['config'], run['identity']
    require(complete.get('state') == 'complete' and complete.get('schema') == SCHEMA
            and complete.get('seed') == seed and complete.get('preflight') is preflight
            and complete.get('steps') == expected_steps and complete.get('exposures') == expected_steps * 4,
            'Structured completion identity/steps mismatch: ' + str(out))
    require(cfg.get('seed') == identity.get('seed') == seed and cfg.get('preflight') is preflight
            and cfg.get('steps') == expected_steps and identity.get('schema') == SCHEMA,
            'Structured run identity differs: ' + str(out))
    require(complete.get('reload_max_abs_score_error') == 0
            and complete.get('bridge_unchanged') is True
            and complete.get('frozen_parameter_versions_unchanged') is True
            and complete.get('test_queries') == complete.get('external_queries') == 0,
            'Structured engineering gate failed: ' + str(out))
    require(complete.get('best_step') in range(0, expected_steps + 1)
            and all((out / (mode + '.pt')).is_file() for mode in ('best', 'final')),
            'Missing structured weights/invalid selected step')
    for mode in ('best', 'final'):
        metrics = read(out / (mode + '_val_metrics.json'))
        require(metrics['overall']['scores']['n'] == metrics['overall']['generated']['n'] == expected_n
                and metrics['localization']['n'] == expected_n, 'Structured validation denominator differs')
    if preflight:
        require(complete.get('last5_mean_loss', float('inf')) < complete.get('first5_mean_loss', -float('inf')),
                'Structured fixed-example fit did not improve')


def validate_structured_reload(out, mode):
    result = read(out / ('fresh_' + mode + '_reload.json'))
    complete = read(out / 'complete.json')
    require(result.get('state') == 'complete' and result.get('mode') == mode
            and result.get('step') == (complete['best_step'] if mode == 'best' else complete['steps'])
            and result.get('n', 0) > 0 and result.get('fresh_reload_max_abs_score_error') == 0
            and result.get('generated_exact_match') is True and result.get('adapter_tensor_exact_match') is True
            and result.get('bridge_unchanged') is True and result.get('original_batch_composition_preserved') is True
            and result.get('test_queries') == result.get('external_queries') == 0,
            'Structured fresh-process reload failed: ' + str(out) + ' ' + mode)


def validate_tool(out, n):
    run = read(out / 'run.json')
    require(run.get('status') == 'complete' and run.get('n') == n
            and tuple(run.get('conditions', ())) == TOOL_CONDITIONS
            and run.get('test_queries') == run.get('external_queries') == 0
            and run.get('gpu_model_loaded') is True, 'Tool run completion/size/conditions gate failed: ' + str(out))
    rows = [json.loads(line) for line in (out / 'predictions.jsonl').read_text().splitlines() if line.strip()]
    require(len(rows) == n * 5 and len(run.get('selected_paths', [])) == n
            and len(set(run['selected_paths'])) == n and set(run.get('results', {})) == set(TOOL_CONDITIONS),
            'Tool predictions/results denominator incomplete: ' + str(out))
    for condition in TOOL_CONDITIONS:
        selected = [row for row in rows if row.get('condition') == condition]
        require(len(selected) == n and [row['path'] for row in selected] == run['selected_paths']
                and all(row.get('split') == 'val' and row.get('n_images') == (1 if condition == 'no_tool' else 2)
                        and row.get('image_tokens_total') == (196 if condition == 'no_tool' else 392)
                        for row in selected), 'Tool per-condition metadata/budget mismatch: ' + condition)
        result = run['results'][condition]
        require(result.get('n') == n and result['metrics']['overall']['scores']['n'] == n
                and result['metrics']['overall']['generated']['n'] == n, 'Tool metric denominator mismatch')


def validate_tool_audit(out, n):
    validate_tool(out, n)
    audit = read(out / 'audit.json')
    require(audit.get('passed') is True and audit.get('n') == n
            and tuple(audit.get('conditions', ())) == TOOL_CONDITIONS
            and audit.get('gt_reads_before_saved_run_and_rgb_gates') == 0
            and set(audit.get('classification', {})) == set(TOOL_CONDITIONS)
            and set(audit.get('posthoc_gt_geometry', {}).get('results', {})) == set(TOOL_CONDITIONS),
            'Tool posterior audit did not pass all five arms: ' + str(out))
    require(audit['sources'].get('run_json') == sha256(out / 'run.json')
            and audit['sources'].get('predictions_jsonl') == sha256(out / 'predictions.jsonl'),
            'Tool run changed after audit: ' + str(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=Path('/root/autodl-tmp/AnomalyGPT'))
    ap.add_argument('--root', type=Path, default=Path('runs/migration_20261103/advanced_20261104'))
    args = ap.parse_args()
    repo = args.repo.resolve()
    os.chdir(repo)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state = dict(state='waiting_for_D_rank_and_reload', jobs=[], test_queries=0, external_queries=0)
    sp = root / 'continuation_status.json'
    write(state, sp)

    def job(name, cmd, marker, validate, output_dir=None):
        if marker.exists():
            validate()
            state['jobs'].append(dict(name=name, state='already_complete'))
            write(state, sp)
            return
        if output_dir is not None and output_dir.exists():
            raise FileExistsError('Preserve incomplete output; move it explicitly before retry: ' + str(output_dir))
        state.update(state='running', stage=name, command=[str(x) for x in cmd])
        write(state, sp)
        with open(root / (name + '.log'), 'a') as log:
            process = subprocess.Popen([str(x) for x in cmd], stdout=log, stderr=subprocess.STDOUT)
            state['pid'] = process.pid
            write(state, sp)
            code = process.wait()
        require(code == 0, '%s exit=%s; partial output preserved' % (name, code))
        require(marker.exists(), 'Missing completion marker: ' + str(marker))
        validate()
        state['jobs'].append(dict(name=name, state='complete'))
        write(state, sp)

    try:
        for name in ('status.json', 'verification_status.json'):
            while True:
                value = read(root / name) if (root / name).exists() else {}
                if value.get('state') in ('failed', 'blocked_by_training_failure'):
                    raise RuntimeError('Prior stage failed: ' + name)
                if value.get('state') == 'complete':
                    break
                time.sleep(15)
        validate_prior(root)
        os.environ.setdefault('OMP_NUM_THREADS', '8')
        os.environ.setdefault('MKL_NUM_THREADS', '8')
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        cache = repo / 'runs/migration_20261103/teacher_cache.pt'
        targets = root / 'structured_targets.jsonl'
        manifest = repo / 'runs/migration_20261103/lora_20261104/base_manifest.json'
        with open(root.parent / 'gpu.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            require(targets.exists() or not targets.with_suffix('.json').exists(), 'Incomplete target sidecar preserved')
            job('structured_targets', [sys.executable, 'structured_geometry.py', '--cache', cache, '--out', targets],
                targets, lambda: validate_targets(targets, cache))
            for preflight, seed in [(True, 42)] + [(False, seed) for seed in (42, 43, 44)]:
                name = ('preflight_S' if preflight else 'arm_S') + '_s' + str(seed)
                out = root / name
                if not preflight:
                    validate_structured(root / 'preflight_S_s42', 42, True)
                cmd = [sys.executable, '-u', 'structured_train.py', '--cache', cache, '--targets', targets,
                       '--source-c', repo / ('runs/migration_20261103/lora_20261104/arm_C_s%d' % seed),
                       '--base-manifest', manifest, '--out', out, '--seed', seed]
                if preflight:
                    cmd += ['--preflight', '--steps', 40, '--val-every', 20]
                job(name, cmd, out / 'complete.json',
                    lambda out=out, seed=seed, preflight=preflight: validate_structured(out, seed, preflight), out)
                if not preflight:
                    for mode in ('best', 'final'):
                        job(name + '_verify_' + mode, [sys.executable, 'structured_verify.py', '--run', out, '--mode', mode],
                            out / ('fresh_' + mode + '_reload.json'),
                            lambda out=out, mode=mode: validate_structured_reload(out, mode))
            for name, limit, n in [('tool_preflight', 8, 8), ('tool_val', 0, 380)]:
                out = root / name
                if name == 'tool_val':
                    validate_tool_audit(root / 'tool_preflight', 8)
                job(name, [sys.executable, '-u', 'tool_workflow.py', '--run', '--limit', limit, '--out', out],
                    out / 'run.json', lambda out=out, n=n: validate_tool(out, n), out)
                job(name + '_audit', [sys.executable, 'tool_audit.py', '--run', out, '--out', out / 'audit.json', '--with-masks'],
                    out / 'audit.json', lambda out=out, n=n: validate_tool_audit(out, n))
            # Completion is based on revalidated artifacts, including skipped work.
            validate_targets(targets, cache)
            for seed in (42, 43, 44):
                out = root / ('arm_S_s%d' % seed)
                validate_structured(out, seed, False)
                for mode in ('best', 'final'):
                    validate_structured_reload(out, mode)
            validate_tool_audit(root / 'tool_preflight', 8)
            validate_tool_audit(root / 'tool_val', 380)
            state.update(state='complete', stage='complete', finished=time.time())
    except Exception as error:
        state.update(state='failed', error=str(error), traceback=traceback.format_exc(), finished=time.time())
        raise
    finally:
        write(state, sp)


if __name__ == '__main__':
    main()
