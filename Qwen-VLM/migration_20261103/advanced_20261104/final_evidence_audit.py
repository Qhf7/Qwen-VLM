"""Final CPU audit of saved evidence; no model, weights, RGB, mask or cache reads.

All 21 external summaries must already be complete. Pending evidence writes an
explicit pending report and exits 2; failed integrity checks exit 1. Success is
possible only after the existing independent summarizers recompute all results.
Only FINAL_EVIDENCE_AUDIT.json/.md (or explicit --out) are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'code'))
from summarize_advanced import same as compare_advanced_metrics, stats as advanced_stats, KEYS as ADVANCED_KEYS
from migration_metrics import evaluate_predictions
import summarize_structured as structured_summary
import summarize_external as external_summary
import summarize_tool as tool_summary
from diagnose_external_structured import completion_gate, checked_jsonl, validate_manifest

SEEDS = (42, 43, 44)
MODELS = external_summary.BASIC + external_summary.STRUCTURED


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def compare_saved(actual, expected, location='metrics'):
    # Existing comparison recurses through both dictionaries and lists, with
    # 1e-12 float tolerance. Compare both directions to require identical keys.
    structured_summary.recursive_compare(actual, expected, location)
    structured_summary.recursive_compare(expected, actual, location)


def require_result_only_snapshot(advanced, historical):
    """Refuse optional tensor paths before invoking existing JSON summarizers."""
    candidates = set()
    directories = [advanced/(('arm_D' if arm == 'D' else arm)+'_s%d'%seed)
                   for arm in ('D','Cpair','R') for seed in SEEDS]
    directories += [advanced/('arm_S_s%d'%seed) for seed in SEEDS]
    directories += [historical/('arm_C_s%d'%seed) for seed in SEEDS]
    for directory in directories:
        candidates.update(directory/name for name in ('best.pt','final.pt'))
        run_path = directory/'run.json'
        if run_path.is_file():
            run=read(run_path);cfg=run.get('config',{});identity=run.get('identity',{})
            for value in (cfg.get('cache'),identity.get('bridge_path')):
                if value:candidates.add(Path(value))
            if cfg.get('source_c'):candidates.add(Path(cfg['source_c'])/'best.pt')
    freeze_path=advanced/'external_selection_freeze.json'
    if freeze_path.is_file():
        for selected in read(freeze_path).get('models',{}).values():
            for key in ('checkpoint','original_bridge_path'):
                if selected.get(key):candidates.add(Path(selected[key]))
    present=sorted(str(path) for path in candidates if path.is_file())
    require(not present,'This audit requires the local result-only snapshot. Tensor/cache/weight files exist: '+', '.join(present))


def pending_evidence(advanced, historical):
    """Metadata-only gate; never opens any raw predictions or target files."""
    pending = []
    for model in MODELS:
        for seed in SEEDS:
            _, output = external_summary.run_paths(advanced, historical, model, seed)
            path = output / 'summary.json'
            if not path.is_file():
                pending.append(dict(section='external', model=model, seed=seed, reason='summary_missing', path=str(path)))
                continue
            summary = read(path)
            if summary.get('state') != 'complete' or summary.get('n') != 1575:
                pending.append(dict(section='external', model=model, seed=seed,
                                    reason='summary_not_complete1575', state=summary.get('state'), n=summary.get('n')))
    for name, checks in (
        ('summary.json', lambda d:d.get('state') == 'complete' and d.get('completed_runs') == 9 and len(d.get('rows',[])) == 18),
        ('verification_status.json', lambda d:d.get('state') == 'complete' and len(d.get('checks',[])) == 9),
        ('structured_summary/summary.json', lambda d:d.get('state') == 'complete' and d.get('fresh_reload_all_passed') is True),
        ('tool_summary.json', lambda d:d.get('state') == 'complete'),
        ('external_summary.json', lambda d:d.get('state') == 'complete' and d.get('expected_runs') == 21
         and d.get('completed_recomputed_runs') == 21 and not d.get('unavailable'))):
        path = advanced / name
        if not path.is_file():
            pending.append(dict(section='summary', path=str(path), reason='summary_missing'))
        elif not checks(read(path)):
            pending.append(dict(section='summary', path=str(path), reason='required_complete_summary_pending'))
    return pending


def audit_advanced(advanced, historical):
    """Verify the existing eighteen recomputation bindings and official metrics."""
    saved = read(advanced / 'summary.json')
    require(saved['state'] == 'complete' and saved['completed_runs'] == saved['expected_runs'] == 9
            and len(saved['rows']) == len(saved['prediction_recomputations']) == 18,
            'Advanced nine-run/eighteen-raw coverage differs')
    recomputations = {entry['file']:entry for entry in saved['prediction_recomputations']}
    require(len(recomputations) == 18, 'Advanced recomputation filenames are duplicated')
    rows = {(row['arm'],row['seed'],row['mode']):row for row in saved['rows']}
    require(len(rows) == 18, 'Advanced summary identities are duplicated')
    for seed in SEEDS:
        for arm in ('D','Cpair','R'):
            name=('arm_D' if arm == 'D' else arm)+'_s%d'%seed
            directory=advanced/name
            for mode in ('best','final'):
                filename='best_val_generated_predictions.json' if mode=='best' else 'final_val_predictions.json'
                path=directory/filename;records=read(path)
                require(len(records)==380 and len({record['path'] for record in records})==380
                        and sum(record['label']==1 for record in records)==208
                        and len({record['category'] for record in records})==12,
                        'Advanced raw validation denominator differs')
                computed=evaluate_predictions(records,[record['score'] for record in records],
                                              [record['generated'] for record in records])
                compare_advanced_metrics(computed,read(directory/(mode+'_val_metrics.json')))
                entry=recomputations[str(path.relative_to(advanced))]
                require(entry.get('n')==380 and entry.get('matches') is True and entry.get('sha256')==sha256(path),
                        'Advanced saved recomputation SHA differs')
                row=rows[arm,seed,mode];score=computed['overall']['scores'];generated=computed['overall']['generated']
                values=dict(auroc=score['image_auroc'],ap=score['image_ap'],gen_f1=generated['macro_f1'],
                            recall=generated['recall'],fpr=generated['fpr'],invalid=generated['invalid_rate'],
                            category_auroc=computed['category_macro']['scores']['image_auroc'])
                for key,value in values.items():compare_saved(value,row[key])
    verification = read(advanced / 'verification_status.json')
    checks = {entry['run']:entry for entry in verification['checks']}
    require(len(checks) == 9, 'Advanced fresh run identities are not unique')
    fresh, sources = [], {}
    for seed in SEEDS:
        for arm in ('D','Cpair','R'):
            name = ('arm_D' if arm == 'D' else arm) + '_s%d' % seed
            run = advanced / name
            config, complete = read(run / 'run.json'), read(run / 'complete.json')
            require(config['identity']['seed'] == complete['seed'] == seed, 'Advanced source seed differs')
            require(complete.get('steps') == 500 and complete.get('preflight') is False
                    and complete.get('test_queries') == 0
                    and complete.get('reload_max_abs_score_error') == complete.get('zero_adapter_max_abs_logit_error') == 0
                    and complete.get('frozen_parameter_versions_unchanged') is True,
                    'Advanced completed training engineering gate failed')
            schedule=read(run/'sample_indices.json')
            require(len(schedule)==2000 and hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
                    == config['identity']['sample_schedule_sha256'], 'Advanced exposure schedule SHA differs')
            require(config['source_sha256']['migration_advanced_train.py'] == sha256(advanced/'migration_advanced_train.py'),
                    'Advanced training source changed')
            check = checks[name]
            require(check.get('state') == 'passed' and check.get('both_modes_verified') is True
                    and set(check.get('modes', [])) == {'best','final'}, 'Advanced verification queue is incomplete')
            for mode in ('best','final'):
                path = run / ('fresh_'+mode+'_reload.json'); audit = read(path)
                reference = run / ('best_val_generated_predictions.json' if mode == 'best' else 'final_val_predictions.json')
                require(audit.get('state') == 'complete' and audit.get('checkpoint') == mode
                        and audit.get('seed') == seed and audit.get('arm') == ('C' if arm == 'Cpair' else arm)
                        and audit.get('step') == (complete['best_step'] if mode == 'best' else 500),
                        'Advanced fresh checkpoint identity differs')
                require(audit.get('n') == 28 and audit.get('audited_labels') == [0,1]
                        and len(set(audit.get('audited_categories',[]))) == 4
                        and audit.get('original_batch_composition_preserved') is True,
                        'Advanced fresh validation coverage differs')
                require(audit.get('fresh_reload_max_abs_score_error') == audit.get('adapter_disabled_max_abs_score_error') == 0
                        and audit.get('generated_exact_match') is True and audit.get('both_modes_verified') is True
                        and audit.get('immutable_base_loads_in_process') == 1
                        and audit.get('verification_invocation') == 'both' and audit.get('test_queries') == 0,
                        'Advanced fresh exactness/dual-mode gate failed')
                require(audit.get('reference_predictions_sha256') == sha256(reference)
                        and audit.get('verifier_sha256') == sha256(advanced/'verify_advanced_checkpoint.py'),
                        'Advanced fresh source/reference SHA differs')
                require(audit.get('adapter_B_nonzero') is True or (mode == 'best' and audit['step'] == 0),
                        'Advanced trained adapter is unexpectedly zero')
                require(check.get('n' if mode == 'final' else 'best_n') == audit['n']
                        and (mode != 'best' or check.get('best_step') == audit['step']),
                        'Verification queue differs from fresh reports')
                require(rows[arm,seed,mode]['step'] == audit['step'], 'Advanced summary step differs from fresh checkpoint')
                fresh.append(dict(run=name,mode=mode,n=audit['n'],report_sha256=sha256(path),
                                  checkpoint_sha256=audit['checkpoint_sha256'],
                                  reference_predictions_sha256=sha256(reference)))
            sources[name] = dict(run_sha256=sha256(run/'run.json'), complete_sha256=sha256(run/'complete.json'))
    for arm in ('D','Cpair','R'):
        for mode in ('best','final'):
            group=[rows[arm,seed,mode] for seed in SEEDS]
            aggregate={key:advanced_stats([row[key] for row in group])
                       for key in ADVANCED_KEYS+('category_auroc','peak_gib','exposures_per_second')}
            compare_saved(aggregate,saved['aggregate'][arm+'_'+mode])
    return dict(state='passed', runs=9, recomputed_prediction_files=18, prediction_rows=18*380,
                both_modes_fresh_runs=9, fresh_reports=fresh, source_metadata=sources,
                summary_sha256=sha256(advanced/'summary.json'), verification_sha256=sha256(advanced/'verification_status.json'),
                summarizer_sha256=sha256(advanced/'summarize_advanced.py'),
                saved_recomputation_sha_and_metrics_reverified=True)


def audit_generation_policy(path, run, seed, mode):
    policy = read(path)
    require(policy.get('source_sha256') == sha256(path.parent.parent/'advanced_generation.py')
            and policy.get('version') == 'exact-gated-leftpad-greedy-v1'
            and policy.get('gate_split') == 'VisA_existing_val_only'
            and policy.get('max_new_tokens') == 80 and policy.get('batch_size') == run['config']['eval_batch_size']
            and policy.get('scalar_parent_exact') is True and policy.get('selected') in ('scalar','batched'),
            'S generation source/schema/gate differs')
    require(len(policy.get('gate_indices',[])) == len(policy.get('gate_paths',[])) == 8
            and len(set(policy['gate_indices'])) == len(set(policy['gate_paths'])) == 8
            and set(policy.get('gate_labels',[])) == {0,1} and len(set(policy.get('gate_categories',[]))) == 4,
            'S generation fixed val8 coverage differs')
    require(policy.get('context',{}).get('seed') == seed, 'S generation policy seed differs')
    if mode == 'initial':
        require(policy['context'].get('condition') == 'initial', 'Initial policy context differs')
    if policy['selected'] == 'batched':
        require(policy.get('exact_token_ids_and_stop_ids') is True and policy.get('failure') is None
                and policy.get('scalar_token_ids') == policy.get('batched_token_ids')
                and policy.get('scalar_stop_ids') == policy.get('batched_stop_ids'),
                'S batched generation lacks exact token/EOS gate')
    return policy


def audit_structured(advanced, historical):
    saved = read(advanced/'structured_summary/summary.json')
    require(saved.get('state') == 'complete' and saved.get('seeds') == list(SEEDS)
            and saved.get('fresh_reload_all_passed') is True
            and saved.get('summarizer_sha256') == sha256(advanced/'summarize_structured.py'),
            'S summary completeness/source differs')
    saved_runs = {row['seed']:row for row in saved['runs']}
    require(len(saved['runs']) == 3 and set(saved_runs) == set(SEEDS), 'S summary seed coverage differs')
    computed_runs, raw_reports, fresh_reports, canonical_rows = [], [], [], None
    for seed in SEEDS:
        directory = advanced/('arm_S_s%d' % seed)
        # Reuse existing recomputations after refusing actual local tensor inputs.
        computed = structured_summary.summarize_run(directory)
        computed_runs.append(computed)
        prior = saved_runs[seed]
        for key in ('run','seed','best_step','identity','modes','paired_differences','teacher_baseline',
                    'best_generation_reuse_audit','fresh_reload','cost','generation_policies',
                    'predictions_sha256','run_sha256','complete_sha256'):
            compare_saved(computed[key], prior[key], 'S.%d.%s' % (seed,key))
        run = read(directory/'run.json'); complete = read(directory/'complete.json')
        for filename, expected in run['source_sha256'].items():
            require(sha256(advanced/filename) == expected, 'S frozen source differs: '+filename)
        source_c = historical/('arm_C_s%d' % seed)
        require(sha256(source_c/'run.json') == run['identity']['source_c_run_sha256'], 'S original C source SHA differs')
        if prior.get('original_C_binary_prompt') is not None:
            compare_saved(prior['original_C_binary_prompt']['metrics'],read(source_c/'best_val_metrics.json'))
        for mode, filename in (('initial','initial_val_predictions.json'),('best','best_val_generated_predictions.json'),
                               ('final','final_val_predictions.json')):
            path = directory/filename; records = read(path)
            canonical = [{key:record[key] for key in ('index','path','category','label','split','target')} for record in records]
            if canonical_rows is None:
                canonical_rows = canonical
            require(canonical == canonical_rows, 'S raw target/source order changed across seeds or stages')
            policy_path = directory/(mode+'_generation_policy.json')
            audit_generation_policy(policy_path,run,seed,mode)
            raw_reports.append(dict(seed=seed,mode=mode,n=len(records),sha256=sha256(path),policy_sha256=sha256(policy_path)))
        for mode in ('best','final'):
            path = directory/('fresh_'+mode+'_reload.json'); fresh = read(path)
            policy_path = directory/(mode+'_generation_policy.json'); policy=read(policy_path)
            require(fresh.get('state') == 'complete' and fresh.get('mode') == mode
                    and fresh.get('step') == (complete['best_step'] if mode == 'best' else 500)
                    and fresh.get('n') == 104 and len(set(fresh.get('categories',[]))) == 12
                    and fresh.get('labels') == [0,1], 'S fresh checkpoint/coverage differs')
            require(all(fresh.get(key) is True for key in ('original_batch_composition_preserved','generated_exact_match',
                        'adapter_tensor_exact_match','adapter_B_nonzero','bridge_unchanged'))
                    and fresh.get('fresh_reload_max_abs_score_error') == 0
                    and fresh.get('test_queries') == fresh.get('external_queries') == 0,
                    'S fresh exactness gate failed')
            require(fresh.get('source_sha256') == sha256(advanced/'structured_verify.py')
                    and fresh.get('generation_policy_sha256') == sha256(policy_path)
                    and fresh.get('generation_policy') == policy['selected'], 'S fresh source/policy SHA differs')
            fresh_reports.append(dict(seed=seed,mode=mode,n=104,sha256=sha256(path),policy_sha256=sha256(policy_path)))
    for key in ('schema','cache_sha256','targets_sha256','base_manifest_sha256'):
        require(len({row['identity'][key] for row in computed_runs}) == 1, 'S cross-seed identity differs: '+key)
    aggregates = {mode:structured_summary.aggregate([row['modes'][mode]['flat'] for row in computed_runs])
                  for mode in ('initial','best','final')}
    compare_saved(aggregates,saved['aggregate'])
    paired = {label:structured_summary.aggregate([row['paired_differences'][label] for row in computed_runs])
              for label in ('best_minus_initial','final_minus_initial')}
    compare_saved(paired,saved['aggregate_paired_differences'])
    existing_audit=read(advanced/'STRUCTURED_FULL_AUDIT.json')
    require(existing_audit.get('state')=='complete' and existing_audit.get('checked_raw_prediction_files')==9
            and existing_audit.get('checked_fresh_reports')==6
            and existing_audit.get('checked_prediction_rows')==3420
            and existing_audit.get('fresh_reload_all_passed') is True,
            'Existing independent S audit is incomplete')
    return dict(state='passed', runs=3, raw_prediction_files=raw_reports, prediction_rows=9*380,
                fresh_reports=fresh_reports, canonical_source_and_entire_target_sha256=canonical_sha(canonical_rows),
                source_and_entire_target_identical_across_nine_raw_files=True,
                summary_sha256=sha256(advanced/'structured_summary/summary.json'),
                existing_independent_audit_sha256=sha256(advanced/'STRUCTURED_FULL_AUDIT.json'),
                teacher_control='Saved control metrics checked; original tensor cache intentionally not opened by final audit.')


def audit_tool(advanced):
    computed = tool_summary.build(advanced)
    saved = read(advanced/'tool_summary.json')
    require(computed['state'] == saved['state'] == 'complete', 'Tool preflight/validation not fully audited')
    for key in ('schema','preflight','validation','external_queries','model_loaded','gt_masks_opened_by_summarizer','summarizer_sha256'):
        compare_saved(computed[key],saved[key],'tool.'+key)
    for stage in ('tool_preflight','tool_val'):
        for key in ('state','expected_n','reported_n','sources'):
            compare_saved(computed['stages'][stage][key],saved['stages'][stage][key],'tool.'+stage+'.'+key)
        run = read(advanced/stage/'run.json')
        require(run['source_sha256'] == sha256(advanced/'tool_workflow.py'), 'Tool inference source SHA differs')
    validation = computed['validation']; model = validation['arms']['model_tool']
    return dict(state='passed',preflight_n=computed['preflight']['n'],validation_n=validation['n'],
                n_conditions=5, validation_prediction_rows=validation['n']*5,
                completed_stage_sha256={key:value['sources'] for key,value in computed['stages'].items()},
                model_tool_calls=model['actual_calls'], model_planner_actions=model['planner_action_counts'],
                model_selected_crop_observed=validation['model_selected_crop_observed'],
                complete_zero_crop_result_is_legal=True,
                observed_interpretation=validation['autonomous_crop_evidence'],
                summary_sha256=sha256(advanced/'tool_summary.json'),
                summarizer_sha256=sha256(advanced/'summarize_tool.py'))


def source_path(advanced, filename):
    return advanced.parent/'code'/filename if filename in ('migration_language.py','migration_metrics.py') else advanced/filename


def known_snapshot_relative(value, expected, advanced):
    """Accept one exact relative identity under a known local/server root."""
    local_root=PurePosixPath(str(advanced.parent))
    server_root=PurePosixPath('/root/autodl-tmp/AnomalyGPT/runs/migration_20261103')
    require(isinstance(value,str), 'Snapshot path must be a string')
    path=PurePosixPath(value)
    require(path.is_absolute() and str(path)==value and '\x00' not in value
            and all(part not in ('.','..') for part in path.parts), 'Illegal snapshot path: '+repr(value))
    expected_relative=PurePosixPath(str(expected)).relative_to(local_root)
    roots=[root for root in (local_root,server_root) if path.is_relative_to(root)]
    require(len(roots)==1, 'Unknown snapshot path root: '+value)
    relative=path.relative_to(roots[0])
    require(relative==expected_relative, 'Snapshot path differs from expected identity: '+value)
    return str(relative)


def normalize_external_row_paths(rows, advanced, historical):
    """Normalize only run/output fields; require exact model/seed directories."""
    normalized=[]
    for row in rows:
        require(row.get('model') in MODELS and row.get('seed') in SEEDS,
                'Unknown external model/seed while validating path metadata')
        expected_run,expected_output=external_summary.run_paths(advanced,historical,row['model'],row['seed'])
        result=dict(row)
        for key,expected in (('run',expected_run),('output',expected_output)):
            result[key]=known_snapshot_relative(row.get(key),expected,advanced)
        normalized.append(result)
    return normalized


def split_external_environment_attestations(computed, saved, advanced, historical, freeze):
    """Keep server tensor-check attestations separate from local invariants."""
    require(len(computed)==len(saved)==21, 'External attestation coverage differs')
    environment=[];local_rows=[];saved_rows=[]
    for actual,prior in zip(computed,saved):
        require((actual['model'],actual['seed'])==(prior['model'],prior['seed']),
                'External attestation row order differs')
        model,seed=actual['model'],actual['seed']
        local_value=actual.get('checkpoint_bytes_verified_locally')
        saved_value=prior.get('checkpoint_bytes_verified_locally')
        require(type(local_value) is bool and local_value is False and type(saved_value) is bool,
                'External checkpoint execution attestation must be boolean; local result-only must be False')
        run,_=external_summary.run_paths(advanced,historical,model,seed)
        source=read(run/'run.json');selected=freeze['models'][model+'_s%d'%seed]
        checkpoint=(Path(source['identity']['bridge_path']) if model=='A' else
                    Path(source['config']['source_c'])/'best.pt' if model=='Cstructured' else run/'best.pt')
        expected=(advanced.parent/('lm_qwen_s%d'%seed)/'best.pt' if model=='A' else
                  historical/('arm_C_s%d'%seed)/'best.pt' if model=='Cstructured' else run/'best.pt')
        relative=known_snapshot_relative(str(checkpoint),expected,advanced)
        require(not checkpoint.is_file() and not expected.is_file(),
                'Local checkpoint bytes exist, violating result-only attestation scope')
        require(actual['checkpoint_sha256']==prior['checkpoint_sha256']==selected['checkpoint_sha256'],
                'Attested checkpoint SHA differs from frozen checkpoint')
        environment.append(dict(model=model,seed=seed,field='checkpoint_bytes_verified_locally',
                                local_result_only_value=local_value,saved_summary_value=saved_value,
                                expected_checkpoint_relative_path=relative,checkpoint_sha256=actual['checkpoint_sha256'],
                                interpretation='Server summary attests a checkpoint-byte verification; this audit did not read or verify checkpoint bytes.'
                                   if saved_value else 'Saved summary does not attest checkpoint-byte verification.'))
        local_rows.append({key:value for key,value in actual.items() if key!='checkpoint_bytes_verified_locally'})
        saved_rows.append({key:value for key,value in prior.items() if key!='checkpoint_bytes_verified_locally'})
    return local_rows,saved_rows,environment


def recover_external_source_metadata(row, advanced, historical, freeze):
    """Resolve two optional source lookups lost by a server-to-local snapshot."""
    model,seed=row['model'],row['seed']
    if model not in ('A','Cstructured'):
        return row,None
    restored=dict(row)
    run,_=external_summary.run_paths(advanced,historical,model,seed)
    source=read(run/'run.json')
    if model=='A':
        original=advanced.parent/('lm_qwen_s%d'%seed)
        known_snapshot_relative(source['identity']['bridge_path'],original/'best.pt',advanced)
        path=original/'run.json';metadata=read(path);identity=source['identity']
        require(metadata.get('backend')=='qwen' and metadata['config']['backend']=='qwen'
                and metadata['config']['seed']==metadata['initialization_audit']['init_seed']==seed,
                'Original A bridge seed/backend metadata differs')
        for key in ('cache_sha256','cache_manifest_sha256','decoder_sha256','pipeline_version'):
            require(metadata[key]==identity[key],'Original A bridge identity differs: '+key)
        count=metadata.get('trainable_parameters');initialization=metadata['initialization_audit']
        require(type(count) is int and count>0
                and type(initialization['prompt_parameters']) is int
                and type(initialization['cls_projection_parameters']) is int
                and count==initialization['prompt_parameters']+initialization['cls_projection_parameters'],
                'Original A bridge parameter decomposition differs')
        require(row['source_trainable_parameters'] is None or row['source_trainable_parameters']==count,
                'Original A bridge parameter count conflicts with loader')
        restored['source_trainable_parameters']=count
        return restored,dict(model=model,seed=seed,field='source_trainable_parameters',
                              loader_value=row['source_trainable_parameters'],recovered_value=count,
                              source_relative_path=str(path.relative_to(advanced.parent)),source_json_sha256=sha256(path),
                              checks=['seed','qwen backend','cache SHA','cache manifest SHA','decoder SHA',
                                      'pipeline version','prompt plus CLS parameter decomposition','exact saved-summary equality'],
                              limitation='Original bridge run.json SHA was not pre-bound in the external selection freeze; this local metadata evidence has weaker provenance than the frozen C metadata.')
    original=historical/('arm_C_s%d'%seed)
    known_snapshot_relative(source['config']['source_c'],original,advanced)
    original_run,original_complete=original/'run.json',original/'complete.json'
    frozen_c=freeze['models']['C_s%d'%seed]
    require(sha256(original_run)==frozen_c['run_sha256']==source['identity']['source_c_run_sha256']
            and sha256(original_complete)==frozen_c['complete_sha256'],
            'Original C source cost metadata differs from pre-external freeze')
    config,done=read(original_run),read(original_complete)
    require(config['identity']['arm']=='C' and config['identity']['seed']==done['seed']==seed
            and done.get('preflight') is False and done.get('steps')==500,
            'Original C source cost identity differs')
    cost={key:done[key] for key in ('train_seconds','exposures_per_second','peak_memory_bytes')}
    restored['source_training_cost']=cost
    restored['source_trainable_parameters']=config['trainable_parameters']
    return restored,dict(model=model,seed=seed,field='source_training_cost',loader_value=row['source_training_cost'],
                          recovered_value=cost,source_relative_path=str(original.relative_to(advanced.parent)),
                          source_run_sha256=sha256(original_run),source_complete_sha256=sha256(original_complete),
                          pre_external_frozen_C_metadata_sha_exact=True,
                          attribution='Original C training cost; no new Cstructured training, distinct from the later S continuation cost.')


def audit_external(advanced, historical):
    manifest_path=advanced/'external_manifest.jsonl'
    if not manifest_path.is_file():
        manifest_path=advanced/'external_mvtec/manifest.jsonl'
    manifest=validate_manifest(manifest_path)
    freeze=read(advanced/'external_selection_freeze.json')
    require(freeze.get('state') == 'all_selection_frozen_before_external_model_queries'
            and freeze.get('manifest_sha256') == external_summary.EXPECTED_MANIFEST,
            'External selection freeze identity differs')
    source_checks={}
    require(len(freeze['source_sha256']) == 9, 'Expected nine frozen external source files')
    for filename,expected in freeze['source_sha256'].items():
        actual=sha256(source_path(advanced,filename))
        require(actual == expected, 'External frozen source SHA differs: '+filename)
        source_checks[filename]=actual
    expected_keys={(model,seed) for model in MODELS for seed in SEEDS}
    require(set(freeze['models']) == {model+'_s%d'%seed for model,seed in expected_keys},
            'Pre-external frozen model set differs')
    rows=[]; target_bytes=None; targets_sha=None;target_reports=[];source_recoveries=[]
    for model in MODELS:
        for seed in SEEDS:
            run,output=external_summary.run_paths(advanced,historical,model,seed)
            summary=read(output/'summary.json');selected=freeze['models'][model+'_s%d'%seed]
            require(summary.get('state')=='complete' and summary.get('n')==1575
                    and summary.get('seed')==seed, 'External summary is incomplete before raw read')
            require(sha256(run/'run.json') == selected['run_sha256']
                    and sha256(run/'complete.json') == selected['complete_sha256'],
                    'External source metadata differs from selection freeze')
            if model in external_summary.STRUCTURED:
                gate,pending=completion_gate(advanced,historical,model,seed,external_summary.EXPECTED_MANIFEST)
                require(pending is None, 'Structured completion gate became pending')
                digest=gate['prediction_sha256']
            else:
                require(summary.get('mask_opened') is False, 'Basic external result opened masks')
                checksum=(output/'predictions.sha256').read_text().split()
                require(bool(checksum), 'Basic external prediction checksum missing')
                digest=checksum[0]
            checked_jsonl(output/'predictions.jsonl',digest)
            row,issue=external_summary.load_result(advanced,historical,model,seed,manifest,external_summary.EXPECTED_MANIFEST)
            require(issue is None and row is not None and row['state'] == 'complete_and_recomputed',
                    'External result failed full official recomputation')
            require(row['n'] == 1575 and row['n_normal'] == 392 and row['n_anomaly'] == 1183
                    and row['category_count'] == 15, 'External denominator differs')
            row,recovery=recover_external_source_metadata(row,advanced,historical,freeze)
            if recovery is not None:
                source_recoveries.append(recovery)
            rows.append(row)
            if model in external_summary.STRUCTURED:
                path=output/'posthoc_targets.json';raw=path.read_bytes();actual=hashlib.sha256(raw).hexdigest()
                require(actual == summary['posthoc_targets_sha256'], 'External target SHA differs')
                targets=json.loads(raw)
                require(len(targets)==1575 and all(all(target[key]==entry[key] for key in ('source_id','category','label'))
                        for target,entry in zip(targets,manifest)), 'External target source order differs')
                if target_bytes is None:
                    target_bytes,targets_sha=raw,actual
                require(raw == target_bytes, 'Six external complete target artifacts differ')
                target_reports.append(dict(model=model,seed=seed,n=1575,sha256=actual))
    require({(row['model'],row['seed']) for row in rows} == expected_keys and len(rows)==21,
            'External expected21 coverage differs')
    expected_order=canonical_sha([(row['source_id'],row['path'],row['category'],row['label']) for row in manifest])
    require({row['source_order_sha256'] for row in rows} == {expected_order}, 'External source order differs')
    for key in ('external_cache_sha256','base_manifest_sha256','manifest_sha256'):
        require(len({row[key] for row in rows})==1,'External shared identity differs: '+key)
    require(len(target_reports)==6,'Six external structured targets not covered')
    saved=read(advanced/'external_summary.json')
    require(saved.get('state')=='complete' and saved.get('requested_models')==list(MODELS)
            and saved.get('requested_seeds')==list(SEEDS) and saved.get('expected_runs')==21
            and saved.get('completed_recomputed_runs')==21 and not saved.get('unavailable')
            and saved.get('summarizer_sha256')==sha256(advanced/'summarize_external.py'),
            'External saved summary completeness/source differs')
    comparable,saved_comparable,environment=split_external_environment_attestations(rows,saved['rows'],advanced,historical,freeze)
    compare_saved(normalize_external_row_paths(comparable,advanced,historical),
                  normalize_external_row_paths(saved_comparable,advanced,historical))
    compare_saved(external_summary.aggregate_rows(rows,MODELS),saved['aggregate'])
    compare_saved(external_summary.paired_results(rows,MODELS),saved['paired'])
    return dict(state='passed',completed_recomputed_runs=21,expected_runs=21,prediction_rows=21*1575,
                summary_sha256=sha256(advanced/'external_summary.json'),
                selection_freeze_sha256=sha256(advanced/'external_selection_freeze.json'),
                frozen_source_sha256=source_checks,source_order_sha256=expected_order,
                individual_summary_sha256={row['model']+'_s%d'%row['seed']:row['summary_sha256'] for row in rows},
                individual_prediction_sha256={row['model']+'_s%d'%row['seed']:row['predictions_sha256'] for row in rows},
                structured_target_artifacts=target_reports,six_target_artifacts_byte_identical=True,
                common_target_sha256=targets_sha,
                path_metadata_normalization=dict(fields=['run','output'],
                    accepted_roots=[str(advanced.parent),'/root/autodl-tmp/AnomalyGPT/runs/migration_20261103'],
                    rule='Known local/server roots only; exact expected model/seed relative paths; other fields unchanged'),
                checkpoint_execution_environment_attestations=environment,
                source_path_recovery_evidence=source_recoveries,
                metadata_scope_limits=['Only run/output directory strings are normalized under the two known snapshot roots.',
                    'Checkpoint-byte verification is a saved server attestation; the local result-only audit checks absence and SHA bindings, not weight bytes.',
                    'A bridge run.json SHA was not pre-bound by external freeze. Its local SHA, parameter decomposition, seed/backend/cache/decoder and exact saved-summary count are checked; provenance is weaker than frozen original C metadata.',
                    'Cstructured source cost is recovered from exact frozen original C run/complete JSON; S continuation cost is not substituted.'],
                checkpoint_tensor_bytes_checked_by_this_audit=False,
                note='Checkpoint identity is bound to the pre-external freeze and completed fresh reports. No tensor bytes are opened.')


def markdown(report):
    lines=['# Final saved evidence audit','','State: **'+report['state']+'**. CPU saved-artifact audit; no model, GPU, SSH, RGB, masks, tensor weights or caches are read.','']
    if report.get('pending'):
        lines += ['Pending evidence (no prediction raw files were opened):','']
        lines += ['- '+json.dumps(item,ensure_ascii=False) for item in report['pending']]
    if report.get('error'):
        lines += ['Integrity check failed: '+report['error'],'']
    for name,value in report.get('checks',{}).items():
        lines.append('- '+name+': '+value['state'])
    if report['state']=='passed':
        lines += ['', 'Coverage: D/Cpair/R 9 runs, 18 recomputed validation files and 18 fresh reports; '
                  'S 3 runs, 9 recomputed validation files and 6 fresh104 reports; tool preflight8 and validation380 '
                  'with five conditions and completed posterior SHA audits; external21 ×1575 with six identical target artifacts.',
                  'The complete tool result remains valid with zero model-selected crops; audit success does not imply a positive performance result.',
                  'Existing completed fresh reports provide model reconstruction evidence. This script verifies saved report/source/policy bindings and never reruns a model.', '']
        external=report['checks']['external']
        lines += ['| Model | Seeds | Local checkpoint bytes verified | Saved server attestation |',
                  '|---|---|---|---|']
        for model in MODELS:
            group=[row for row in external['checkpoint_execution_environment_attestations'] if row['model']==model]
            lines.append('| '+model+' | '+','.join(str(row['seed']) for row in group)+' | '
                         +','.join(str(row['local_result_only_value']) for row in group)+' | '
                         +','.join(str(row['saved_summary_value']) for row in group)+' |')
        lines += ['', 'These booleans describe different execution environments. Saved True values are server attestations; '
                  'this audit verifies local checkpoint absence and the exact frozen checkpoint SHA binding. It does not verify tensor bytes.', '',
                  '| Source recovery | Seeds | Source and verification |', '|---|---|---|',
                  '| A parameter count | 42,43,44 | Local original lm_qwen run.json; prompt plus CLS parameter decomposition, seed/backend/cache/decoder, and exact server-summary equality |',
                  '| Cstructured training cost | 42,43,44 | Exact pre-external frozen original C run/complete JSON; distinct from S continuation cost |', '',
                  'The original A bridge run.json SHA was not pre-bound in the external selection freeze. Its local SHA is recorded; '
                  'this metadata has weaker provenance than the frozen original C source. All recovered numbers are compared exactly within the existing numeric tolerance.',
                  'Only run/output path strings under the two known local/server roots are normalized. Every relative path must match the expected model and seed; unknown roots and illegal paths are rejected.', '']
    lines += ['Audit script SHA: `'+report['audit_source_sha256']+'`. All input summary/source hashes are in the JSON.','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=HERE)
    parser.add_argument('--historical',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();advanced=args.root.resolve();historical=(args.historical or advanced.parent/'lora_20261104').resolve()
    out=(args.out or advanced/'FINAL_EVIDENCE_AUDIT.json').resolve()
    require(out.suffix=='.json' and out.name.startswith('FINAL_EVIDENCE_AUDIT'),
            'Use a separate FINAL_EVIDENCE_AUDIT*.json output; input artifacts cannot be overwritten')
    require(not any(part.startswith(('arm_','Cpair_s','R_s','tool_val','tool_preflight')) for part in out.relative_to(advanced).parts[:-1]),
            'Final output must be outside frozen run directories')
    report=dict(state='pending',audit_source_sha256=sha256(__file__),
                scope='Final saved evidence consistency audit; foundation-model inference is not rerun.',
                cpu_only=True,only_saved_json_jsonl_checksums_and_local_sources=True,
                model_queries=0,rgb_reads=0,mask_reads=0,tensor_cache_or_weight_reads=0,
                root=str(advanced),checks={},pending=[])
    exit_code=2
    try:
        require_result_only_snapshot(advanced,historical)
        report['pending']=pending_evidence(advanced,historical)
        report['summary_sha256']={name:sha256(advanced/name) for name in (
            'summary.json','verification_status.json','structured_summary/summary.json','tool_summary.json','external_summary.json')
            if (advanced/name).is_file()}
        if not report['pending']:
            report['checks']['advanced']=audit_advanced(advanced,historical)
            report['checks']['structured']=audit_structured(advanced,historical)
            report['checks']['tool']=audit_tool(advanced)
            report['checks']['external']=audit_external(advanced,historical)
            report['state']='passed';exit_code=0
    except Exception as error:
        report.update(state='failed',error=type(error).__name__+': '+str(error));exit_code=1
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    out.with_suffix('.md').write_text(markdown(report))
    print(json.dumps(dict(state=report['state'],pending=len(report['pending']),checks=list(report['checks']),
                          error=report.get('error'),output=str(out)),ensure_ascii=False))
    raise SystemExit(exit_code)


if __name__=='__main__':
    main()
