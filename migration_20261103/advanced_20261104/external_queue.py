"""Open frozen external data only after all selections and audits complete.

GPU ownership: cache and structured evaluators run under this queue's lock;
basic evaluator owns its own lock. Existing invalid/partial outputs fail with
their original evidence preserved. The first freeze is immutable on retries.
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

EXPECTED_MANIFEST = '4cf2709614c8fd6aabe1268ef3c9f8687a07a0849d3d0f0341d1756570f9497a'
SEEDS = (42, 43, 44)


def write(value, path):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return value


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_staging(data, manifest, protocol):
    manifest_sha = sha256(manifest)
    require(manifest_sha == EXPECTED_MANIFEST, 'External manifest changed')
    spec = read(protocol)
    require(spec.get('manifest_sha256') == manifest_sha and spec.get('retained_n') == 1575
            and spec.get('n_normal') == 392 and spec.get('n_anomaly') == 1183,
            'Frozen external protocol/counts changed')
    audit = read(data / 'staging_audit.json')
    require(audit.get('state') == 'complete' and audit.get('image_count') == 1575
            and audit.get('manifest_sha256') == manifest_sha and not audit.get('failed')
            and audit.get('model_queries') == 0, 'External staging failed/incomplete/mismatched')
    return manifest_sha


def freeze_sources(repo, root, manifest, protocol, decoder):
    historical = repo / 'runs/migration_20261103/lora_20261104'
    models = {}
    for seed in SEEDS:
        c_run = historical / ('arm_C_s%d' % seed)
        c_spec, c_done = read(c_run / 'run.json'), read(c_run / 'complete.json')
        cid = c_spec['identity']
        require(cid['arm'] == 'C' and cid['seed'] == seed and not c_done['preflight']
                and c_done['steps'] == 500, 'Original C source incomplete')
        for model in ('A', 'C', 'D', 'Cpair', 'R', 'Cstructured', 'S'):
            run = (c_run if model in ('A', 'C') else root / ('arm_S_s%d' % seed)
                   if model in ('Cstructured', 'S') else root / (('arm_D' if model == 'D' else model) + '_s%d' % seed))
            spec, done = read(run / 'run.json'), read(run / 'complete.json')
            require(done.get('seed') == seed and done.get('preflight') is False
                    and done.get('steps') == 500 and done.get('test_queries') == 0,
                    'Source run incomplete: ' + str(run))
            checkpoint = Path(cid['bridge_path']) if model == 'A' else c_run / 'best.pt' if model == 'Cstructured' else run / 'best.pt'
            step = cid['bridge_step'] if model == 'A' else c_done['best_step'] if model == 'Cstructured' else done['best_step']
            models[model + '_s%d' % seed] = dict(model=model, seed=seed, run=str(run),
                checkpoint=str(checkpoint), checkpoint_step=step, checkpoint_sha256=sha256(checkpoint),
                run_sha256=sha256(run / 'run.json'), complete_sha256=sha256(run / 'complete.json'),
                original_bridge_path=cid['bridge_path'], original_bridge_sha256=cid['bridge_sha256'],
                base_manifest_sha256=cid['base_manifest_sha256'])
            require(sha256(cid['bridge_path']) == cid['bridge_sha256'], 'Original source bridge changed')
    filenames = ('evaluate_external_checkpoint.py', 'structured_external.py', 'migration_language.py',
                 'structured_model.py', 'structured_geometry.py', 'migration_metrics.py',
                 'migration_advanced_train.py', 'advanced_generation.py', 'cache_external_mvtec.py')
    sources = {name: sha256(repo / name) for name in filenames}
    policies = {}
    for seed in SEEDS:
        run = root / ('arm_S_s%d' % seed)
        for name in ('initial_generation_policy.json', 'best_generation_policy.json'):
            path = run / name
            policies[str(path)] = sha256(path) if path.exists() else None
    return dict(state='all_selection_frozen_before_external_model_queries', models=models,
                manifest_sha256=sha256(manifest), protocol_sha256=sha256(protocol),
                decoder_sha256=sha256(decoder), source_sha256=sources, structured_generation_policy_sha256=policies,
                prompts=dict(basic='industrial_component; historical YesNo max32',
                             structured='industrial_component; bbox1000 grid3 max80'),
                prompt_source_sha256={name: sources[name] for name in ('migration_language.py', 'structured_model.py')})


def validate_cache(cache, frozen):
    sidecar = read(cache.with_suffix('.json'))
    require(cache.is_file() and sidecar.get('cache_sha256') == sha256(cache)
            and sidecar.get('manifest_sha256') == frozen['manifest_sha256']
            and sidecar.get('decoder_sha256') == frozen['decoder_sha256']
            and sidecar.get('schema') == 'imagebind-bigdec-external-cache-v1'
            and sidecar.get('pipeline_version') == 'visa-square-binary-sklearn-v1'
            and sidecar.get('geometry') == 'direct-square224' and sidecar.get('images') == 1575
            and sidecar.get('gt_masks_opened') is False and sidecar.get('images_for_training') is False
            and sidecar.get('generation_or_qwen_queries') == 0,
            'External cache/sidecar identity or content contract failed')
    return sidecar['cache_sha256']


def validate_result(output, model, seed, frozen, cache_sha, records):
    summary, prediction_path = read(output / 'summary.json'), output / 'predictions.jsonl'
    expected = frozen['models'][model + '_s%d' % seed]
    require(summary.get('state') == 'complete' and summary.get('seed') == seed and summary.get('n') == 1575
            and summary.get('manifest_sha256') == frozen['manifest_sha256']
            and summary.get('checkpoint_step') == expected['checkpoint_step']
            and summary.get('checkpoint_sha256') == expected['checkpoint_sha256']
            and summary.get('external_queries') == 1575 and summary.get('historical_visa_test_queries') == 0,
            'External result identity/completion mismatch: ' + str(output))
    structured = model in ('Cstructured', 'S')
    require(summary.get('external_cache_sha256' if structured else 'cache_sha256') == cache_sha,
            'External result cache differs')
    evaluator = 'structured_external.py' if structured else 'evaluate_external_checkpoint.py'
    require(summary.get('evaluator_sha256') == frozen['source_sha256'][evaluator], 'External evaluator changed after freeze')
    rows = [json.loads(line) for line in prediction_path.read_text().splitlines() if line.strip()]
    require(len(rows) == 1575 and len({row['source_id'] for row in rows}) == 1575
            and all(all(row[key] == record[key] for key in ('source_id', 'path', 'category', 'label'))
                    and row['split'] == 'external' for row, record in zip(rows, records)),
            'External prediction source order/count mismatch')
    digest = sha256(prediction_path)
    require((output / 'predictions.sha256').read_text().split()[0] == digest, 'Prediction checksum differs')
    require(summary['metrics']['overall']['scores']['n'] == summary['metrics']['overall']['generated']['n'] == 1575,
            'External metric denominator incomplete')
    if structured:
        condition = 'initial_C_structured_prompt' if model == 'Cstructured' else 'S_best'
        require(summary.get('condition') == condition and summary.get('schema') == 'visa-yesno-bbox1000-grid3-v1'
                and summary.get('max_new_tokens') == 80 and summary.get('predictions_sha256') == digest
                and summary.get('mask_reads_during_inference') == 0 and summary.get('posthoc_mask_reads') == 1183
                and summary.get('model_source_sha256') == frozen['source_sha256']['structured_model.py']
                and summary.get('posthoc_targets_sha256') == sha256(output / 'posthoc_targets.json')
                and summary['metrics']['localization']['n'] == 1575, 'Structured external posterior contract failed')
        inference = read(output / 'inference_complete.json')
        require(inference.get('state') == 'inference_complete' and inference.get('processed') == 1575
                and inference.get('mask_reads') == 0 and inference.get('predictions_sha256') == digest,
                'Structured predictions were not frozen before posterior masks')
    else:
        require(summary.get('arm') == ('C' if model == 'Cpair' else model) and summary.get('mask_opened') is False,
                'Basic external arm/mask contract failed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=Path('/root/autodl-tmp/AnomalyGPT'))
    args = ap.parse_args()
    repo = args.repo.resolve()
    os.chdir(repo)
    root = repo / 'runs/migration_20261103/advanced_20261104'
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / 'external_status.json'
    status = dict(state='waiting_for_all_stage_selection', jobs=[])
    write(status, status_path)
    data = repo / 'datasets/MVTec_AD_residual_v1'
    manifest, protocol = root / 'external_manifest.jsonl', root / 'external_protocol.json'
    cache, decoder = root / 'external_teacher_cache.pt', repo / 'runs/dec_big_fixed_s42/best.pt'

    def job(name, cmd, marker, check, output_dir=None):
        if marker.exists():
            check()
            status['jobs'].append(dict(name=name, state='already_complete'))
            write(status, status_path)
            return
        if output_dir is not None and output_dir.exists():
            raise FileExistsError('Incomplete output preserved; move explicitly before retry: ' + str(output_dir))
        status.update(state='running', stage=name, command=[str(x) for x in cmd])
        write(status, status_path)
        with open(root / (name + '.log'), 'a') as log:
            code = subprocess.call([str(x) for x in cmd], stdout=log, stderr=subprocess.STDOUT)
        require(code == 0, '%s failed exit=%d; partial evidence preserved' % (name, code))
        require(marker.exists(), 'Missing completion marker: ' + str(marker))
        check()
        status['jobs'].append(dict(name=name, state='complete'))
        write(status, status_path)

    try:
        while True:
            continuation = read(root / 'continuation_status.json') if (root / 'continuation_status.json').exists() else {}
            if continuation.get('state') == 'failed':
                raise RuntimeError('Prior continuation failed')
            if continuation.get('state') == 'complete':
                break
            time.sleep(15)
        from continue_stages import validate_prior, validate_structured, validate_structured_reload, validate_tool_audit
        validate_prior(root)
        for seed in SEEDS:
            run = root / ('arm_S_s%d' % seed)
            validate_structured(run, seed, False)
            for mode in ('best', 'final'):
                validate_structured_reload(run, mode)
        validate_tool_audit(root / 'tool_preflight', 8)
        validate_tool_audit(root / 'tool_val', 380)
        while True:
            marker = data / 'staging_audit.json'
            if marker.exists():
                audit = read(marker)
                if audit.get('state') == 'failed':
                    raise RuntimeError('External data staging failed; failed files are preserved')
                if audit.get('state') == 'complete':
                    break
                require(audit.get('state') in ('running', 'staging'), 'Unknown external staging state')
            time.sleep(15)
        validate_staging(data, manifest, protocol)
        frozen = freeze_sources(repo, root, manifest, protocol, decoder)
        freeze_path = root / 'external_selection_freeze.json'
        if freeze_path.exists():
            existing = read(freeze_path)
            require({key: value for key, value in existing.items() if key != 'timestamp'} == frozen,
                    'Frozen source/checkpoint/prompt identity changed; original freeze preserved')
        else:
            write(dict(**frozen, timestamp=time.time()), freeze_path)
        records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        os.environ.setdefault('OMP_NUM_THREADS', '8')
        os.environ.setdefault('MKL_NUM_THREADS', '8')
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        with open(root.parent / 'gpu.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            require(cache.with_suffix('.json').exists() or not cache.exists(), 'Incomplete external cache preserved')
            job('external_cache', [sys.executable, '-u', 'cache_external_mvtec.py', '--manifest', manifest,
                '--data-root', data, '--decoder', decoder, '--out', cache], cache.with_suffix('.json'),
                lambda: validate_cache(cache, frozen))
        cache_sha = validate_cache(cache, frozen)
        historical = repo / 'runs/migration_20261103/lora_20261104'
        for seed in SEEDS:
            for model in ('A', 'C', 'D', 'Cpair', 'R'):
                run = historical / ('arm_C_s%d' % seed) if model in ('A', 'C') else root / (('arm_D' if model == 'D' else model) + '_s%d' % seed)
                output = run / ('external_mvtec_residual_v1_A' if model == 'A' else 'external_mvtec_residual_v1')
                cmd = [sys.executable, '-u', 'evaluate_external_checkpoint.py', '--root', root, '--run', run,
                       '--external-cache', cache, '--manifest', manifest]
                if model == 'A':
                    cmd += ['--baseline-a']
                # Basic evaluator takes the shared lock itself; do not hold it here.
                job('external_%s_s%d' % (model, seed), cmd, output / 'summary.json',
                    lambda output=output, model=model, seed=seed: validate_result(output, model, seed, frozen, cache_sha, records), output)
        with open(root.parent / 'gpu.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for seed in SEEDS:
                for initial in (True, False):
                    run = root / ('arm_S_s%d' % seed)
                    model = 'Cstructured' if initial else 'S'
                    output = run / ('external_mvtec_residual_v1_initial_c' if initial else 'external_mvtec_residual_v1')
                    cmd = [sys.executable, '-u', 'structured_external.py', '--run', run, '--external-cache', cache,
                           '--manifest', manifest, '--protocol', protocol, '--data-root', data, '--out', output]
                    if initial:
                        cmd += ['--initial-c']
                    job('external_%s_s%d' % (model, seed), cmd, output / 'summary.json',
                        lambda output=output, model=model, seed=seed: validate_result(output, model, seed, frozen, cache_sha, records), output)
        # The summary is recomputed from every immutable raw prediction; no GPU or masks.
        output = root / 'external_summary.json'
        job('external_summary', [sys.executable, 'summarize_external.py', '--advanced-root', root,
                                '--historical-root', historical, '--manifest', manifest, '--include-structured', '--out', output],
            output, lambda: require(read(output).get('state') == 'complete'
                                    and read(output).get('completed_recomputed_runs') == 21,
                                    'Recomputed external summary incomplete'))
        require(len(status['jobs']) == 23, 'Expected cache +21 evaluations +summary jobs')
        status.update(state='complete', stage='complete', finished=time.time())
    except Exception as error:
        status.update(state='failed', error=str(error), traceback=traceback.format_exc(), finished=time.time())
        raise
    finally:
        write(status, status_path)


if __name__ == '__main__':
    main()
