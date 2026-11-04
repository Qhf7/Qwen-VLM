"""Execute the authorized migration routes serially after an accepted smoke.

No retries or result-based hyperparameter search. Completed stage statuses
are checked and skipped; failed stages stop the sequence for inspection.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

REPO = Path('/root/autodl-tmp/AnomalyGPT')
ROOT = REPO / 'runs/migration_20261103'
STAGES = ('native_main', 'visual_probe', 'dense_gt', 'roi_pilot', 'roi_full', 'dense_kd')
PIXEL_KEYS = ('pixel_ap_macro', 'pixel_auroc_macro', 'pixel_auroc_weighted', 'image_auroc_from_map99')
SCORE_KEYS = ('image_auroc', 'image_ap', 'accuracy', 'macro_f1', 'recall', 'fpr', 'invalid_rate')
GENERATED_KEYS = ('accuracy', 'macro_f1', 'recall', 'fpr', 'invalid_rate')
EXACT_CHECKS = ('prefill_logits', 'cached_step_logits', 'prefill_and_cached_language_inputs',
                'prefill_rope_delta', 'cached_rope_delta', 'full_replay_logits',
                'full_replay_language_inputs', 'full_replay_shadow_ids', 'full_replay_rope_delta',
                'official_full_vs_cached_last_position', 'cache_lengths', 'cached_step_has_no_visual_inputs')
ROI_CONDITIONS = ('original_only', 'original_center_crop', 'original_random_crop',
                  'original_teacher_roi', 'original_teacher_roi_reference')
DENSE_GT = (('imagebind', 224, 'dense_imagebind_gt_s42'), ('pre', 224, 'dense_pre_224_gt_s42'),
            ('post', 224, 'dense_post_224_gt_s42'), ('pre', 448, 'dense_pre_gt_s42'),
            ('post', 448, 'dense_post_gt_s42'))
DENSE_KD = (('pre', 448, 'dense_pre_kd_s42'), ('post', 448, 'dense_post_kd_s42'))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    try:
        def reject_constant(value):
            raise ValueError('Nonfinite JSON: ' + value)
        value = json.loads(path.read_text(), parse_constant=reject_constant)
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeError('Missing/invalid required JSON %s: %s' % (path, error)) from error
    require(isinstance(value, dict), 'Required JSON is not an object: ' + str(path))
    return value


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def required_metrics(value, keys, label):
    require(isinstance(value, dict), 'Missing metric object: ' + label)
    require(all(finite(value.get(key)) and 0 <= value[key] <= 1 for key in keys),
            'Missing/nonfinite/out-of-range metrics: ' + label)


def channel(value, name, n, label):
    metrics = value.get('overall', {}).get(name, {})
    require(metrics.get('n') == n, 'Wrong sample count: ' + label + '.' + name)
    required_metrics(metrics, SCORE_KEYS if name == 'scores' else GENERATED_KEYS, label + '.' + name)
    return metrics


def checkpoint_files(directory):
    # Presence/size only; no torch import or checkpoint deserialization.
    for name in ('best.pt', 'final.pt'):
        path = directory / name
        require(path.is_file() and path.stat().st_size > 0, 'Missing/empty checkpoint: ' + str(path))


def exact_gate(path, baseline, phase):
    report = read_json(path)
    require(report.get('passed') is True and report.get('status') == 'complete', 'Exact gate incomplete: ' + str(path))
    require(report.get('expected_auxiliary_tokens') == (0 if baseline else 18), 'Wrong auxiliary branch: ' + str(path))
    require(report.get('baseline') is baseline and report.get('auxiliary_tokens') == (0 if baseline else 18),
            'Wrong gate branch/token count: ' + str(path))
    require(report.get('context', {}).get('phase') == phase, 'Wrong gate checkpoint phase: ' + str(path))
    checks = report.get('checks', {})
    require(all(checks.get(name, {}).get('passed') is True for name in EXACT_CHECKS), 'Required exact checks missing: ' + str(path))
    for name in ('prefill_logits', 'cached_step_logits', 'full_replay_logits'):
        check = checks[name]
        require(check.get('dtype_equal') is True and check.get('bitwise_equal') is True and check.get('max_abs_diff') == 0,
                'Same-shape official/manual logits not exact: ' + str(path))
    require(not report.get('failed_checks') and not report.get('error'), 'Exact gate failure recorded: ' + str(path))
    return report


def validate_native(directory, baseline=False, smoke=False):
    run = read_json(directory / 'run.json')
    config = run.get('config', {})
    n, generated_n = (24, 8) if smoke else (380, 380)
    require(run.get('pipeline_version') == 'qwen3vl-native-map-increment-v1', 'Wrong native pipeline: ' + str(directory))
    require(config.get('baseline') is baseline and config.get('smoke') is smoke and config.get('eval_only') is False,
            'Native train/baseline/smoke contract mismatch: ' + str(directory))
    require(config.get('seed') == 42 and config.get('steps') == (4 if smoke else 1500), 'Wrong native seed/budget: ' + str(directory))
    require(run.get('selection_split') == 'val' and run.get('val_n') == n and run.get('counts', {}).get('val') == 380,
            'Wrong native val scope: ' + str(directory))
    require(run.get('dtype_audit', {}).get('passed') is True, 'Native dtype audit failed: ' + str(directory))
    require(run.get('auxiliary_tokens') == (0 if baseline else 18), 'Native auxiliary count wrong: ' + str(directory))
    initial = exact_gate(directory / 'auxiliary_cache_check.json', baseline, 'before_training_or_checkpoint_load')
    selected = exact_gate(directory / 'selected_auxiliary_cache_check.json', baseline, 'selected_checkpoint_reloaded')
    complete = read_json(directory / 'complete.json')
    trained = read_json(directory / 'train_complete.json')
    metrics = read_json(directory / 'val_metrics.json')
    require(complete.get('passed') is True and complete.get('split') == 'val', 'Native final completion not passed: ' + str(directory))
    require(metrics.get('split') == 'val' and metrics.get('val_n') == n and metrics.get('full_validation_n') == 380
            and metrics.get('smoke') is smoke, 'Native metrics scope mismatch: ' + str(directory))
    step = metrics.get('selected_step')
    require(isinstance(step, int) and (step == 0 if baseline else 1 <= step <= config['steps']), 'Invalid selected native step')
    require(step == trained.get('selected_step') == complete.get('selected_step') == selected.get('context', {}).get('selected_step'),
            'Native selected step mismatch: ' + str(directory))
    require(trained.get('selected_auxiliary_cache_check_passed') is True and finite(trained.get('reload_max_abs_score_error'))
            and 0 <= trained['reload_max_abs_score_error'] <= 1e-5, 'Native reload audit incomplete: ' + str(directory))
    conditions = ('real_map',) if baseline else ('real_map', 'zero_map', 'donor_opposite_label_map', 'spatial_shuffle_map')
    require(set(metrics.get('conditions', {})) == set(conditions), 'Native controls incomplete: ' + str(directory))
    for name in conditions:
        channel(metrics['conditions'][name], 'scores', n, name)
    channel(metrics.get('generated_real_map', {}), 'generated', generated_n, 'native generation')
    require(metrics.get('generated_n') == generated_n and metrics.get('generated_is_subset') is smoke,
            'Native generation subset count mismatch: ' + str(directory))
    require(metrics.get('selected_auxiliary_cache_check') == selected, 'Selected gate differs from metrics copy')
    for key in ('cache_sha256', 'decoder_sha256'):
        require(run.get(key) is not None and run[key] == metrics.get(key), 'Native cache provenance mismatch: ' + key)
    if not baseline:
        gradient = read_json(directory / 'gradient_check.json')
        require(gradient.get('frozen_qwen_gradients') == 0 and gradient.get('labels_positions_checked') is True, 'Native gradient audit missing')
        require(all(finite(gradient.get(key)) and gradient[key] > 0 for key in ('base_prompt_gradient_norm', 'adaptive_gradient_norm_sum')),
                'Native trainable branch gradient missing')
    checkpoint_files(directory)
    return {'kind': 'native_smoke' if smoke else 'native', 'directory': str(directory), 'n': n,
            'generated_n': generated_n, 'selected_step': step, 'baseline': baseline,
            'result_sha256': digest(directory / 'val_metrics.json'), 'complete_sha256': digest(directory / 'complete.json')}


def validate_visual(path):
    report = read_json(path)
    require(report.get('pipeline_version') == 'qwen3vl-bf16-normal-memory-square-v1' and report.get('complete') is True
            and not report.get('error'), 'Visual probe report not complete: ' + str(path))
    require(report.get('test_set_used') is False and report.get('source_audit', {}).get('passed') is True, 'Visual source/split audit missing')
    require(report.get('data', {}).get('validation_images') == 380, 'Visual fullval380 missing')
    config = report.get('config', {})
    require(config.get('seed') == 42 and set(config.get('features', [])) == {'pre', 'post'}
            and set(config.get('resolutions', [])) == {224, 448, 672} and set(config.get('shots', [])) == {16, 64}, 'Visual planned matrix mismatch')
    resolutions = report.get('resolutions', [])
    require(len(resolutions) == 3 and {r.get('requested_square_resolution') for r in resolutions} == {224, 448, 672}, 'Visual resolutions incomplete/duplicated')
    for resolution in resolutions:
        require(set(resolution.get('metrics', {})) == {'pre:16', 'pre:64', 'post:16', 'post:64'}, 'Visual feature/shot grid incomplete')
        require(bool(resolution.get('actual_geometries')), 'Visual actual geometry missing')
        size = resolution['requested_square_resolution']
        for geometry in resolution['actual_geometries']:
            require(geometry.get('processor_do_resize') is False and geometry.get('requested_image_hw') == [size, size]
                    and geometry.get('actual_image_hw') == [size, size], 'Visual explicit resolution changed')
        for key, metrics in resolution['metrics'].items():
            feature, shot = key.split(':')
            require(metrics.get('split') == 'val' and metrics.get('n_images') == 380 and metrics.get('feature') == feature
                    and metrics.get('shots') == int(shot), 'Visual cell scope mismatch')
            required_metrics(metrics, PIXEL_KEYS, 'visual ' + key)
    return {'kind': 'normal_memory', 'path': str(path), 'cells': 12, 'result_sha256': digest(path)}


def validate_dense(directory, feature, size, weight):
    run = read_json(directory / 'run.json')
    trained = read_json(directory / 'train_complete.json')
    reload = read_json(directory / 'best_validation_reloaded.json')
    plan = read_json(directory / 'training_plan.json')
    config, compatibility = run.get('config', {}), run.get('compatibility', {})
    require(not (directory / 'failure.json').exists(), 'Dense failure.json exists: ' + str(directory))
    require(config.get('run') is True and config.get('eval') is False and config.get('features') == feature and config.get('size') == size, 'Dense wrong feature/run mode')
    require(config.get('seed') == 42 and config.get('steps') == 1200 and config.get('bs') == 4 and config.get('val_every') == 200 and config.get('lr') == .001, 'Dense budget mismatch')
    require(config.get('distill_weight') == weight and run.get('distillation', {}).get('weight') == weight, 'Dense loss arm mismatch')
    require(compatibility.get('features') == feature and compatibility.get('size') == size and run.get('test_set_used') is False
            and run.get('validation_images') == 380 and run.get('source_audit', {}).get('passed') is True, 'Dense source/scope mismatch')
    require(plan.get('seed') == 42 and plan.get('steps') == 1200 and plan.get('batch_size') == 4
            and len(plan.get('batches', [])) == 1200 and all(len(batch) == 4 for batch in plan['batches']), 'Dense training plan schema mismatch')
    require(canonical(plan['batches']) == plan.get('sha256') == run.get('train_plan_sha256'), 'Dense plan SHA mismatch')
    require(trained.get('steps') == 1200 and trained.get('matched_run_key') == run.get('matched_run_key')
            and bool(run.get('matched_run_key')) and trained.get('test_set_used') is False, 'Dense completion contract mismatch')
    require(reload.get('passed') is True and reload.get('split') == 'val', 'Dense reload not passed')
    step = reload.get('selected_best_step')
    require(isinstance(step, int) and 1 <= step <= 1200 and step == trained.get('selected_best_step'), 'Dense selected step mismatch')
    for name in ('validation_original', 'validation_reloaded'):
        metrics = reload.get(name, {})
        require(metrics.get('split') == 'val' and metrics.get('n_images') == 380 and metrics.get('teacher_used') is False, 'Dense reload validation scope mismatch')
        required_metrics(metrics, PIXEL_KEYS, name)
    for key in PIXEL_KEYS:
        require(abs(reload['validation_original'][key] - reload['validation_reloaded'][key]) <= 1e-10, 'Dense reloaded metric changed')
    require(finite(reload.get('reload_max_abs_metric_error')) and reload['reload_max_abs_metric_error'] <= 1e-10
            and trained.get('reload_max_abs_metric_error') == reload['reload_max_abs_metric_error'], 'Dense reload error report mismatch')
    require(abs(trained.get('best_validation_pixel_ap', -1) - reload['validation_reloaded']['pixel_ap_macro']) <= 1e-10, 'Dense best AP mismatch')
    if weight:
        teacher = run.get('distillation', {})
        require(teacher.get('manifest_sha256') and teacher.get('teacher_checkpoint') and teacher.get('teacher_used_in_validation') is False, 'Dense KD provenance incomplete')
    checkpoint_files(directory)
    return {'kind': 'dense', 'directory': str(directory), 'feature': feature, 'size': size, 'distill_weight': weight,
            'matched_run_key': run['matched_run_key'], 'plan_sha256': plan['sha256'], 'compatibility': compatibility,
            'result_sha256': digest(directory / 'best_validation_reloaded.json')}


def validate_roi(directory, n):
    run = read_json(directory / 'run.json')
    require(run.get('pipeline_version') == 'qwen3vl-native-roi-val-v1' and run.get('status') == 'complete'
            and not run.get('error'), 'ROI run incomplete: ' + str(directory))
    require(run.get('split') == 'val' and run.get('n') == n and run.get('full_validation_n') == 380, 'ROI pilot/fullval mismatch')
    require(run.get('source_audit', {}).get('passed') is True and run.get('dtype_audit', {}).get('passed') is True, 'ROI source/dtype audit incomplete')
    require(set(run.get('conditions', [])) == set(ROI_CONDITIONS) and set(run.get('results', {})) == set(ROI_CONDITIONS), 'ROI conditions incomplete')
    require(len(run.get('selected_paths', [])) == n == len(set(run['selected_paths'])), 'ROI selected paths missing/duplicated')
    for name in ROI_CONDITIONS:
        separate = read_json(directory / (name + '_metrics.json'))
        require(separate == run['results'][name] and separate.get('n') == n, 'ROI condition summary mismatch: ' + name)
        channel(separate.get('metrics', {}), 'scores', n, name)
        channel(separate.get('metrics', {}), 'generated', n, name)
        require(bool(separate.get('actual_image_grids')), 'ROI actual image grids absent')
    return {'kind': 'ROI', 'directory': str(directory), 'n': n, 'conditions': 5, 'result_sha256': digest(directory / 'run.json')}


def expected_jobs(stage):
    if stage == 'roi_pilot':
        return ['roi_preflight', 'roi_pilot']
    if stage == 'native_main':
        return ['native_baseline', 'native_map_s42']
    if stage == 'dense_gt':
        return [name for _, _, name in DENSE_GT]
    if stage == 'dense_kd':
        return [name for _, _, name in DENSE_KD]
    return [stage]


def validate_stage(stage, stage_status, root=ROOT):
    require(stage_status.get('stage') == stage, 'Stage status identity mismatch: ' + stage)
    require([job.get('name') for job in stage_status['jobs']] == expected_jobs(stage), 'Unexpected/incomplete planned jobs: ' + stage)
    for job in stage_status['jobs']:
        command = job.get('command', [])
        require(isinstance(command, list) and '--out' in command, 'Missing recorded stage command: ' + stage)
        position = command.index('--out')
        expected = root / job['name'] / 'val.json' if stage == 'visual_probe' else root / job['name']
        require(position + 1 < len(command) and Path(command[position + 1]) == expected, 'Stage command/output path mismatch: ' + stage)
    if stage == 'native_main':
        return [validate_native(root / 'native_baseline', baseline=True), validate_native(root / 'native_map_s42')]
    if stage == 'visual_probe':
        return [validate_visual(root / stage / 'val.json')]
    if stage == 'roi_pilot':
        preflight = read_json(root / 'roi_preflight/run.json')
        require(preflight.get('pipeline_version') == 'qwen3vl-native-roi-preflight-v1'
                and preflight.get('status') == 'complete' and preflight.get('passed') is True
                and preflight.get('gpu_model_loaded') is True, 'ROI three-image preflight incomplete')
        required = preflight.get('required_checks', [])
        require(len(required) >= 14 and all(preflight.get('checks', {}).get(key) is True for key in required)
                and not preflight.get('failed_checks'), 'ROI preflight required check failed')
        require(preflight.get('split') == 'val' and preflight.get('n') == 1
                and preflight.get('condition') == 'original_teacher_roi_reference'
                and preflight.get('image_size') == 448
                and preflight.get('generation_cached_steps', 0) >= 1
                and preflight.get('performance_metrics_computed') is False, 'ROI preflight scope mismatch')
        return [{'kind': 'ROI_three_image_preflight', 'path': str(root / 'roi_preflight/run.json'),
                 'passed': True, 'result_sha256': digest(root / 'roi_preflight/run.json')},
                validate_roi(root / stage, 48)]
    if stage in ('dense_gt', 'dense_kd'):
        results = [validate_dense(root / name, feature, size, 0 if stage == 'dense_gt' else .5)
                   for feature, size, name in (DENSE_GT if stage == 'dense_gt' else DENSE_KD)]
        if stage == 'dense_kd':
            for kd in results:
                gt = validate_dense(root / ('dense_' + kd['feature'] + '_gt_s42'), kd['feature'], kd['size'], 0)
                require(all(kd[key] == gt[key] for key in ('matched_run_key', 'plan_sha256', 'compatibility')), 'KD/GT pair contract mismatch')
        return results
    return [validate_roi(root / stage, 48 if stage == 'roi_pilot' else 380)]


def validate_teacher_subset(root, require_gt_plans=True):
    path = root / 'teacher_train_subset/manifest.json'
    manifest = read_json(path)
    require(manifest.get('schema_version') == 'visa-teacher-prob-v1' and manifest.get('split') == 'train'
            and manifest.get('coordinate_system') == 'direct-square-224' and manifest.get('visa_pipeline_version') == 'visa-square-binary-sklearn-v1'
            and manifest.get('test_or_validation_maps_exported') is False, 'Teacher subset export contract invalid')
    payload = dict(manifest)
    payload_sha = payload.pop('payload_sha256', None)
    require(payload_sha == canonical(payload), 'Teacher manifest payload SHA mismatch')
    require(isinstance(manifest.get('maps'), dict) and bool(manifest['maps'])
            and manifest.get('n_images') == len(manifest['maps']), 'Teacher manifest map count invalid')
    for feature in ('pre', 'post') if require_gt_plans else ():
        plan = read_json(root / ('dense_' + feature + '_gt_s42') / 'training_plan.json')
        require(canonical(plan.get('batches')) == plan.get('sha256') == manifest.get('training_plan', {}).get('batches_sha256'), 'Teacher subset does not match GT plan')
        required = {item[0] for batch in plan['batches'] for item in batch}
        require(required <= set(manifest.get('maps', {})), 'Teacher subset misses planned paths')
    for relative in manifest.get('maps', {}).values():
        target = (path.parent / relative).resolve()
        require(path.parent.resolve() in target.parents and target.suffix == '.npy' and target.is_file() and target.stat().st_size > 0,
                'Teacher map missing/unsafe path: ' + str(relative))
    return {'manifest': str(path), 'sha256': digest(path), 'n_images': manifest.get('n_images')}


def preserve_existing_outputs(stage, root=ROOT):
    for name in expected_jobs(stage):
        directory = root / name
        require(not (root / (name + '.log')).exists(), 'Existing log preserved; inspect before dispatch: ' + name)
        require(not directory.exists() or not any(directory.iterdir()), 'Existing output preserved; inspect before dispatch: ' + str(directory))


def pid_identity(pid):
    """Linux proc starttime protects against accepting an unrelated reused PID."""
    require(isinstance(pid, int) and not isinstance(pid, bool) and pid > 0, 'Expected a positive existing runner PID')
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError) as error:
        raise RuntimeError('Expected native_main runner PID is not accessible/alive: %s' % pid) from error
    try:
        # comm may contain spaces; fields after its final ')' start at field3.
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        arguments = [part.decode(errors='replace') for part in (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0') if part]
    except (OSError, IndexError) as error:
        raise RuntimeError('Cannot verify native_main PID starttime: %s' % pid) from error
    require(len(fields) >= 20 and fields[0] != 'Z', 'Native_main PID is dead/zombie: %s' % pid)
    require(any(Path(argument).name == 'migration_runner.py' for argument in arguments) and '--stage' in arguments,
            'Supplied PID is not the migration runner: %s' % pid)
    position = arguments.index('--stage')
    require(position + 1 < len(arguments) and arguments[position + 1] == 'native_main',
            'Supplied runner PID is not native_main: %s' % pid)
    return {'pid': pid, 'starttime_ticks': fields[19]}


def existing_wait_contract(stage_status, root):
    require(stage_status.get('stage') == 'native_main' and stage_status.get('state') == 'running',
            '--wait-existing-pid is allowed only for a running native_main status')
    require(bool(stage_status.get('started_at')), 'Running native_main status lacks start identity')
    jobs = stage_status.get('jobs', [])
    names = [job.get('name') for job in jobs]
    expected = expected_jobs('native_main')
    require(bool(names) and names == expected[:len(names)] and len(names) <= len(expected), 'Running native_main job list is not the planned prefix')
    require(stage_status.get('current_job') == names[-1] and jobs[-1].get('state') == 'running', 'Native_main current job not running')
    require(all(job.get('state') == 'complete' and job.get('returncode') == 0 for job in jobs[:-1]), 'Native_main prior job failed/incomplete')
    for job in jobs:
        command = job.get('command', [])
        require(isinstance(command, list) and '--out' in command, 'Missing existing native job command')
        position = command.index('--out')
        require(position + 1 < len(command) and Path(command[position + 1]) == root / job['name'], 'Existing native command output mismatch')
    return stage_status


def wait_existing_native_main(path, identity, root, *, sleep_seconds=10):
    """Observe one authorized runner; status completion is checked before PID death."""
    started = read_json(path)
    started_at = started.get('started_at')
    while True:
        value = read_json(path)
        require(value.get('started_at') == started_at, 'Existing native_main status was replaced while waiting')
        if value.get('state') == 'complete':
            verified = checked_status(path)
            results = validate_stage('native_main', verified, root)
            return verified, results
        existing_wait_contract(value, root)
        current = pid_identity(identity['pid'])
        require(current == identity, 'Existing native_main PID was reused/stale while status not complete')
        time.sleep(sleep_seconds)


def save(value, path):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def checked_status(path):
    status = read_json(path)
    if status.get('state') != 'complete' or not status.get('jobs'):
        raise RuntimeError('Stage has not completed successfully: ' + str(path))
    if any(job.get('state') != 'complete' or job.get('returncode') != 0 for job in status['jobs']):
        raise RuntimeError('Incomplete stage job: ' + str(path))
    return status


def cpu_self_test():
    import copy
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        def put(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        def must_reject(call):
            try:
                call()
            except RuntimeError:
                return
            raise AssertionError('Invalid completion must be rejected')
        def scores(n):
            return {'overall': {'scores': {'n': n, **{key: .5 for key in SCORE_KEYS}}}}
        def gate(phase):
            checks = {name: {'passed': True} for name in EXACT_CHECKS}
            for name in ('prefill_logits', 'cached_step_logits', 'full_replay_logits'):
                checks[name].update(dtype_equal=True, bitwise_equal=True, max_abs_diff=0)
            return {'passed': True, 'status': 'complete', 'expected_auxiliary_tokens': 18,
                    'auxiliary_tokens': 18, 'baseline': False, 'checks': checks, 'failed_checks': [],
                    'context': {'phase': phase, 'selected_step': 4}}
        directory = root / 'native_smoke_s42_v3'
        initial, selected = gate('before_training_or_checkpoint_load'), gate('selected_checkpoint_reloaded')
        run = {'pipeline_version': 'qwen3vl-native-map-increment-v1',
               'config': {'baseline': False, 'smoke': True, 'eval_only': False, 'seed': 42, 'steps': 4},
               'selection_split': 'val', 'val_n': 24, 'counts': {'val': 380}, 'dtype_audit': {'passed': True},
               'auxiliary_tokens': 18, 'cache_sha256': 'cache', 'decoder_sha256': 'decoder'}
        metrics = {'split': 'val', 'val_n': 24, 'full_validation_n': 380, 'smoke': True, 'selected_step': 4,
                   'conditions': {name: scores(24) for name in ('real_map', 'zero_map', 'donor_opposite_label_map', 'spatial_shuffle_map')},
                   'generated_real_map': {'overall': {'generated': {'n': 8, **{key: .5 for key in GENERATED_KEYS}}}},
                   'generated_n': 8, 'generated_is_subset': True, 'selected_auxiliary_cache_check': selected,
                   'cache_sha256': 'cache', 'decoder_sha256': 'decoder'}
        for name, value in {'run.json': run, 'val_metrics.json': metrics, 'auxiliary_cache_check.json': initial,
                            'selected_auxiliary_cache_check.json': selected,
                            'complete.json': {'passed': True, 'split': 'val', 'selected_step': 4},
                            'train_complete.json': {'selected_step': 4, 'selected_auxiliary_cache_check_passed': True, 'reload_max_abs_score_error': 0},
                            'gradient_check.json': {'frozen_qwen_gradients': 0, 'labels_positions_checked': True,
                                                    'base_prompt_gradient_norm': 1, 'adaptive_gradient_norm_sum': 1}}.items():
            put(directory / name, value)
        for name in ('best.pt', 'final.pt'):
            (directory / name).write_bytes(b'CPU presence fixture; never deserialize')
        assert validate_native(directory, smoke=True)['n'] == 24
        wrong = copy.deepcopy(initial)
        wrong['context']['phase'] = 'wrong'
        put(directory / 'auxiliary_cache_check.json', wrong)
        must_reject(lambda: validate_native(directory, smoke=True))
        put(directory / 'auxiliary_cache_check.json', initial)
        must_reject(lambda: validate_native(directory, smoke=False))
        stage_status = {'stage': 'visual_probe', 'state': 'complete', 'jobs': [{'name': 'visual_probe', 'state': 'complete', 'returncode': 0,
                        'command': ['python', 'migration_visual_probe.py', '--out', str(root / 'visual_probe/val.json')]}]}
        put(root / 'visual_probe_status.json', stage_status)
        must_reject(lambda: validate_stage('visual_probe', checked_status(root / 'visual_probe_status.json'), root))
        resolutions = []
        for size in (224, 448, 672):
            cells = {}
            for feature in ('pre', 'post'):
                for shot in (16, 64):
                    cells[feature + ':' + str(shot)] = {'split': 'val', 'n_images': 380, 'feature': feature, 'shots': shot,
                                                       **{key: .5 for key in PIXEL_KEYS}}
            resolutions.append({'requested_square_resolution': size, 'actual_geometries': [
                {'processor_do_resize': False, 'requested_image_hw': [size, size], 'actual_image_hw': [size, size]}], 'metrics': cells})
        visual = {'pipeline_version': 'qwen3vl-bf16-normal-memory-square-v1', 'complete': True, 'test_set_used': False,
                  'source_audit': {'passed': True}, 'data': {'validation_images': 380},
                  'config': {'seed': 42, 'features': ['pre', 'post'], 'resolutions': [224, 448, 672], 'shots': [16, 64]},
                  'resolutions': resolutions}
        put(root / 'visual_probe/val.json', visual)
        assert validate_stage('visual_probe', stage_status, root)[0]['cells'] == 12
        visual['complete'] = False
        put(root / 'visual_probe/val.json', visual)
        must_reject(lambda: validate_stage('visual_probe', stage_status, root))
        must_reject(lambda: preserve_existing_outputs('visual_probe', root))
        assert expected_jobs('dense_gt') == [name for _, _, name in DENSE_GT]
        assert len(expected_jobs('dense_gt')) == 5 and len(expected_jobs('dense_kd')) == 2
        batches = [[['/fixture/train', 'fixture', 0]] * 4 for _ in range(1200)]
        plan = {'seed': 42, 'steps': 1200, 'batch_size': 4, 'batches': batches, 'sha256': canonical(batches)}
        dense_dir = root / 'dense_pre_gt_s42'
        dense_metrics = {'split': 'val', 'n_images': 380, 'teacher_used': False, **{key: .5 for key in PIXEL_KEYS}}
        dense_run = {'config': {'run': True, 'eval': False, 'features': 'pre', 'size': 448, 'seed': 42, 'steps': 1200,
                                'bs': 4, 'val_every': 200, 'lr': .001, 'distill_weight': 0},
                     'compatibility': {'features': 'pre', 'size': 448}, 'test_set_used': False, 'validation_images': 380,
                     'source_audit': {'passed': True}, 'distillation': {'weight': 0},
                     'train_plan_sha256': plan['sha256'], 'matched_run_key': 'fixture-matched'}
        dense_trained = {'steps': 1200, 'matched_run_key': 'fixture-matched', 'test_set_used': False, 'selected_best_step': 400,
                         'reload_max_abs_metric_error': 0, 'best_validation_pixel_ap': .5}
        dense_reload = {'passed': True, 'split': 'val', 'selected_best_step': 400, 'validation_original': dense_metrics,
                        'validation_reloaded': dense_metrics, 'reload_max_abs_metric_error': 0}
        for name, value in {'run.json': dense_run, 'training_plan.json': plan, 'train_complete.json': dense_trained,
                            'best_validation_reloaded.json': dense_reload}.items():
            put(dense_dir / name, value)
        for name in ('best.pt', 'final.pt'):
            (dense_dir / name).write_bytes(b'CPU fixture')
        assert validate_dense(dense_dir, 'pre', 448, 0)['plan_sha256'] == plan['sha256']
        dense_reload['passed'] = False
        put(dense_dir / 'best_validation_reloaded.json', dense_reload)
        must_reject(lambda: validate_dense(dense_dir, 'pre', 448, 0))
        roi_dir = root / 'roi_pilot'
        roi_results = {}
        for name in ROI_CONDITIONS:
            value = {'n': 48, 'actual_image_grids': [[1, 28, 28]], 'metrics': {'overall': {
                'scores': {'n': 48, **{key: .5 for key in SCORE_KEYS}},
                'generated': {'n': 48, **{key: .5 for key in GENERATED_KEYS}}}}}
            roi_results[name] = value
            put(roi_dir / (name + '_metrics.json'), value)
        roi_run = {'pipeline_version': 'qwen3vl-native-roi-val-v1', 'status': 'complete', 'split': 'val', 'n': 48,
                   'full_validation_n': 380, 'source_audit': {'passed': True}, 'dtype_audit': {'passed': True},
                   'conditions': list(ROI_CONDITIONS), 'results': roi_results, 'selected_paths': ['path%d' % i for i in range(48)]}
        put(roi_dir / 'run.json', roi_run)
        assert validate_roi(roi_dir, 48)['conditions'] == 5
        must_reject(lambda: validate_roi(roi_dir, 380))
        roi_run['status'] = 'running'
        put(roi_dir / 'run.json', roi_run)
        must_reject(lambda: validate_roi(roi_dir, 48))
        teacher_path = root / 'teacher_train_subset/manifest.json'
        teacher_map = teacher_path.parent / 'maps/map.npy'
        teacher_map.parent.mkdir(parents=True)
        teacher_map.write_bytes(b'presence only; loader validates real npy')
        teacher = {'schema_version': 'visa-teacher-prob-v1', 'split': 'train', 'coordinate_system': 'direct-square-224',
                   'visa_pipeline_version': 'visa-square-binary-sklearn-v1', 'test_or_validation_maps_exported': False,
                   'training_plan': {'batches_sha256': plan['sha256']}, 'maps': {'/fixture/train': 'maps/map.npy'}, 'n_images': 1}
        teacher['payload_sha256'] = canonical(teacher)
        put(teacher_path, teacher)
        put(root / 'dense_post_gt_s42/training_plan.json', plan)
        assert validate_teacher_subset(root)['n_images'] == 1
        teacher['training_plan']['batches_sha256'] = 'wrong'
        payload = dict(teacher)
        payload.pop('payload_sha256')
        teacher['payload_sha256'] = canonical(payload)
        put(teacher_path, teacher)
        must_reject(lambda: validate_teacher_subset(root))
        native_status_path = root / 'native_main_status.json'
        running = {'stage': 'native_main', 'state': 'running', 'started_at': 'fixture-start', 'current_job': 'native_baseline',
                   'jobs': [{'name': 'native_baseline', 'state': 'running', 'command': ['python', 'migration_native_train.py',
                            '--out', str(root / 'native_baseline')]}]}
        complete_status = copy.deepcopy(running)
        complete_status['state'] = 'complete'
        complete_status['jobs'] = [{'name': name, 'state': 'complete', 'returncode': 0,
                                   'command': ['python', 'migration_native_train.py', '--out', str(root / name)]}
                                  for name in expected_jobs('native_main')]
        identity = {'pid': 123, 'starttime_ticks': '999'}
        put(native_status_path, running)
        with patch.object(sys.modules[__name__], 'pid_identity', return_value=identity), \
                patch.object(sys.modules[__name__], 'validate_stage', return_value=[{'verified': True}]), \
                patch.object(time, 'sleep', side_effect=lambda seconds: put(native_status_path, complete_status)):
            assert wait_existing_native_main(native_status_path, identity, root)[1] == [{'verified': True}]
        put(native_status_path, running)
        with patch.object(sys.modules[__name__], 'pid_identity', side_effect=RuntimeError('dead PID')):
            must_reject(lambda: wait_existing_native_main(native_status_path, identity, root))
        with patch.object(sys.modules[__name__], 'pid_identity', return_value={'pid': 123, 'starttime_ticks': 'reused'}):
            must_reject(lambda: wait_existing_native_main(native_status_path, identity, root))
        failed = copy.deepcopy(running)
        failed['state'] = 'failed'
        put(native_status_path, failed)
        must_reject(lambda: wait_existing_native_main(native_status_path, identity, root))
        # Verify dispatcher stops before launching a stage when outputs already exist.
        with patch.object(sys.modules[__name__], 'ROOT', root), \
                patch.object(sys.modules[__name__], 'validate_native', return_value={'fixture': True}), \
                patch.object(sys.modules[__name__], 'validate_teacher_subset', return_value={'fixture': True}), \
                patch.object(subprocess, 'run', side_effect=AssertionError('Must preserve outputs before dispatch')):
            put(root / 'native_smoke_v3_status.json', {'stage': 'native_smoke', 'state': 'complete',
                'jobs': [{'name': 'native_smoke_s42_v3', 'state': 'complete', 'returncode': 0}]})
            (root / 'native_baseline').mkdir()
            (root / 'native_baseline/existing.json').write_text('{}')
            must_reject(lambda: run_sequence(argparse.Namespace(smoke_tag='_v3')))
            assert read_json(root / 'ordered_sequence_status.json')['state'] == 'failed'
        require('torch' not in sys.modules, 'CPU validator test unexpectedly imported torch')
    return {'passed': True, 'native_smoke_schema': True, 'phase_and_val_scope_enforced': True,
            'exit_zero_not_completion': True, 'full_visual_matrix_required': True,
            'dense_reload_required': True, 'ROI_complete_and_scope_required': True, 'teacher_plan_binding': True,
            'explicit_PID_wait_complete_dead_failed_reuse_checks': True,
            'existing_outputs_preserved': True, 'runner_GT5_KD2_contract': True, 'no_model_or_GPU_import': True}


def run_sequence(args):
    smoke = checked_status(ROOT / ('native_smoke' + args.smoke_tag + '_status.json'))
    smoke_dir = ROOT / ('native_smoke_s42' + args.smoke_tag)
    require(smoke.get('stage') == 'native_smoke' and [job.get('name') for job in smoke['jobs']] == [smoke_dir.name],
            'Smoke job identity mismatch')
    smoke_audit = validate_native(smoke_dir, smoke=True)
    teacher_audit = validate_teacher_subset(ROOT, require_gt_plans=False)
    waiting = None
    wait_pid = getattr(args, 'wait_existing_pid', None)
    if wait_pid is not None:
        running = existing_wait_contract(read_json(ROOT / 'native_main_status.json'), ROOT)
        identity = pid_identity(wait_pid)
        waiting = {'identity': identity, 'stage': 'native_main', 'status_started_at': running.get('started_at')}
    state_path = ROOT / 'ordered_sequence_status.json'
    if state_path.exists():
        previous = read_json(state_path)
        if previous.get('state') == 'complete':
            # Re-read real artifacts before accepting a prior sequence result.
            for stage in STAGES:
                validate_stage(stage, checked_status(ROOT / (stage + '_status.json')), ROOT)
            print('[already complete] verified existing sequence; status preserved', flush=True)
            return
        raise RuntimeError('Inspect existing sequence status before a new dispatch: ' + str(state_path))
    state = {'state': 'running', 'started_at': time.strftime('%Y-%m-%d %H:%M:%S %z'),
             'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'smoke_tag': args.smoke_tag, 'smoke_result_audit': smoke_audit, 'teacher_subset_audit': teacher_audit,
             'wait_existing': waiting,
             'stages': [],
             'protocol': 'validation only; supervised custom60/20/20; no new test inference'}
    save(state, state_path)
    try:
        for stage in STAGES:
            stage_status = ROOT / (stage + '_status.json')
            state['current_stage'] = stage
            save(state, state_path)
            if stage == 'dense_kd':
                state['teacher_subset_audit'] = validate_teacher_subset(ROOT)
                save(state, state_path)
            if stage_status.exists():
                if stage == 'native_main' and waiting is not None:
                    print('[wait existing]', stage, waiting['identity']['pid'], flush=True)
                    _, results = wait_existing_native_main(stage_status, waiting['identity'], ROOT)
                    record = {'stage': stage, 'state': 'complete', 'reused_existing_complete': True,
                              'waited_for_existing_pid': waiting['identity']['pid'], 'result_audits': results}
                else:
                    results = validate_stage(stage, checked_status(stage_status), ROOT)
                    record = {'stage': stage, 'state': 'complete', 'reused_existing_complete': True, 'result_audits': results}
            else:
                preserve_existing_outputs(stage, ROOT)
                print('[dispatch]', stage, flush=True)
                began = time.time()
                result = subprocess.run([sys.executable, '-u', 'migration_runner.py', '--stage', stage],
                                        cwd=REPO, stdin=subprocess.DEVNULL)
                if result.returncode:
                    raise RuntimeError('Stage failed: %s (exit %s)' % (stage, result.returncode))
                results = validate_stage(stage, checked_status(stage_status), ROOT)
                record = {'stage': stage, 'state': 'complete', 'elapsed_seconds': time.time() - began, 'result_audits': results}
            state['stages'].append(record)
            save(state, state_path)
        state.update(state='complete', completed_at=time.strftime('%Y-%m-%d %H:%M:%S %z'))
    except Exception as error:
        state.update(state='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        save(state, state_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-tag', default='_v3')
    parser.add_argument('--wait-existing-pid', type=int, help='Explicit running native_main runner PID; wait10s, no retries')
    parser.add_argument('--cpu-test-only', action='store_true')
    args = parser.parse_args()
    if args.cpu_test_only:
        print(json.dumps(cpu_self_test(), indent=2))
        return
    require(ROOT.is_dir(), 'Missing experiment root: ' + str(ROOT))
    # Separate from runner's gpu.lock: serializes continuation dispatchers only.
    with (ROOT / 'ordered_sequence.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_sequence(args)


if __name__ == '__main__':
    main()
