"""Recompute VisA S reports from predictions; never opens external results."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
for path in (HERE,HERE.parent/'code'):
    sys.path.insert(0,str(path))
from structured_geometry import SCHEMA,sha256_file,localization_metrics,parse_structured
from migration_metrics import evaluate_predictions


COST_KEYS=('train_seconds','exposures_per_second','peak_memory_bytes','elapsed_seconds')


def atomic_json(value,path):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def recursive_compare(expected,actual,path='root',tolerance=1e-12):
    """Check every stored key; allow recomputed reports to add new diagnostics."""
    if isinstance(expected,dict):
        assert isinstance(actual,dict),(path,type(actual))
        for key,value in expected.items():
            assert key in actual,(path,key)
            recursive_compare(value,actual[key],path+'.'+key,tolerance)
    elif isinstance(expected,list):
        assert isinstance(actual,list) and len(expected)==len(actual),path
        for index,(left,right) in enumerate(zip(expected,actual)):
            recursive_compare(left,right,path+'.'+str(index),tolerance)
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        assert isinstance(actual,(int,float)) and abs(expected-actual)<=tolerance,(path,expected,actual)
    else:
        assert expected==actual,(path,expected,actual)


def recompute(records):
    assert records and all(r['split']=='val' for r in records)
    assert len({r['path'] for r in records})==len(records)
    assert all(math.isfinite(r['score']) and isinstance(r['generated'],str) for r in records)
    targets=[r['target'] for r in records];texts=[r['generated'] for r in records]
    for r,t in zip(records,targets):
        assert all(r[key]==t[key] for key in ('path','category','split','label'))
        parsed=parse_structured(t['answer'])
        assert parsed is not None and all(parsed[k]==t[k] for k in ('label','bbox','location'))
        if 'parsed' in r:assert r['parsed']==parse_structured(r['generated'])
    metrics=evaluate_predictions(records,[r['score'] for r in records],texts)
    metrics['localization']=localization_metrics(targets,texts)
    metrics['localization_per_category']={c:localization_metrics([r['target'] for r in records if r['category']==c],
        [r['generated'] for r in records if r['category']==c]) for c in sorted({r['category'] for r in records})}
    return metrics


def extract(metrics):
    """Compact numeric view with explicit global and equal-category metrics."""
    result={}
    for channel,keys in (('scores',('image_auroc','image_ap','macro_f1','recall','fpr')),
                          ('generated',('macro_f1','recall','fpr','invalid_rate'))):
        for key in keys:
            result[channel+'_'+key]=metrics['overall'][channel][key]
            macro=metrics['category_macro'][channel]
            if key in macro:result['category_macro_'+channel+'_'+key]=macro[key]
    result['localization_format_invalid_rate']=1-metrics['localization']['format_valid_rate']
    for key,value in metrics['localization'].items():
        if isinstance(value,(int,float)) and not isinstance(value,bool):result['localization_'+key]=value
    return result


def paired_difference(left,right):
    return {key: right[key]-left[key] for key in left.keys()&right.keys()
            if left[key] is not None and right[key] is not None and not key.startswith('localization_n')}


def aggregate(rows):
    """Sample SD across seeds; never call three runs independent test sets."""
    keys=sorted(set().union(*(row.keys() for row in rows)))
    result={}
    for key in keys:
        values=[row[key] for row in rows if row.get(key) is not None and isinstance(row[key],(int,float))]
        if values:
            result[key]=dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values) if len(values)>1 else None,n_seeds=len(values),values=values)
    return result


def summarize_run(run):
    config=json.loads((run/'run.json').read_text());identity=config['identity'];cfg=config['config']
    complete=json.loads((run/'complete.json').read_text())
    assert identity['schema']==SCHEMA and identity['seed']==complete['seed']
    assert complete['state']=='complete' and not cfg['preflight'] and not complete['preflight']
    assert complete['steps']==500 and complete['exposures']==2000
    assert complete['test_queries']==complete['external_queries']==0
    assert cfg['max_new_tokens']==80 and (cfg['microbatch'],cfg['accum'],cfg['lr'])==(1,4,1e-4)
    modes={};raw_hashes={};raw={}
    for mode,pred_name,metric_name in (
        ('initial','initial_val_predictions.json','initial_val_metrics.json'),
        ('best','best_val_generated_predictions.json','best_val_metrics.json'),
        ('final','final_val_predictions.json','final_val_metrics.json')):
        records=json.loads((run/pred_name).read_text())
        assert len(records)==380 and sum(int(r['label']) for r in records)==208
        assert len({r['index'] for r in records})==len(records)
        assert len({r['category'] for r in records})==12
        metrics=recompute(records)
        recursive_compare(json.loads((run/metric_name).read_text()),metrics,run.name+'/'+mode)
        raw[mode]=records;raw_hashes[pred_name]=sha256_file(run/pred_name)
        modes[mode]=dict(metrics=metrics,flat=extract(metrics))
    fixed_targets=Path(cfg['targets'])
    if fixed_targets.is_file():
        assert sha256_file(fixed_targets)==identity['targets_sha256']
        target_lookup={int(t['index']):t for t in map(json.loads,fixed_targets.read_text().splitlines())}
        assert [r['target'] for r in raw['initial']]==[target_lookup[r['index']] for r in raw['initial']]
    for mode in ('best','final'):
        assert [(r['path'],r['label'],r['category'],r['index']) for r in raw[mode]]==[(r['path'],r['label'],r['category'],r['index']) for r in raw['initial']]
        assert [r['target'] for r in raw[mode]]==[r['target'] for r in raw['initial']]
    for mode in ('initial','best','final'):
        key={'initial':'initial_val_auroc','best':'best_val_auroc','final':'final_val_auroc'}[mode]
        assert abs(modes[mode]['flat']['scores_image_auroc']-complete[key])<1e-12
    selection=[(0,modes['initial']['flat']['scores_image_auroc'])]
    for step in range(100,501,100):
        metrics=json.loads((run/('val_step_%d.json'%step)).read_text())
        selection.append((step,metrics['overall']['scores']['image_auroc']))
    expected_step,expected_auc=max(selection,key=lambda entry:entry[1])
    assert expected_step==complete['best_step'] and expected_auc==complete['best_val_auroc']
    baseline=json.loads((run/'teacher_map_bbox_baseline.json').read_text())
    assert '95th percentile' in baseline['rule']
    # Recompute map baseline if the original tensor cache is available. A local
    # result-only checkout can still report it clearly as stored/unrecomputed.
    baseline_audit='stored_metrics_only_original_cache_not_local'
    cache_path=Path(cfg['cache'])
    if cache_path.is_file():
        import torch
        import numpy as np
        from structured_geometry import mask_to_target,target_answer
        assert sha256_file(cache_path)==identity['cache_sha256']
        cache=torch.load(cache_path,map_location='cpu');plain=[];gated=[]
        for record in raw['initial']:
            i=record['index'];assert cache['items'][i]['path']==record['path']
            probability=cache['maps'][i].numpy();target=mask_to_target(probability>=np.quantile(probability,.95),1)
            plain.append(target_answer(target))
            gated.append(target_answer(target) if record['score']>0 else target_answer(dict(label=0,bbox=None,location='none')))
        expected=dict(unconditional_box=localization_metrics([r['target'] for r in raw['initial']],plain),
                      initial_score_gated=localization_metrics([r['target'] for r in raw['initial']],gated))
        for key in expected:recursive_compare(baseline[key],expected[key],run.name+'/teacherbaseline/'+key)
        baseline_audit='recomputed_from_original_cache_and_initial_logits'
    reuse_path=run/'best_generation_reuse_audit.json'
    if reuse_path.exists():
        reuse=json.loads(reuse_path.read_text())
        if 'best_generation_reuse_audit_sha256' in complete:
            assert sha256_file(reuse_path)==complete['best_generation_reuse_audit_sha256']
        if reuse['state']=='complete':
            assert reuse['identity']==identity and reuse['selected_step'] in (0,complete['steps'])
            assert reuse['full_score_n']==380 and reuse['full_score_max_abs_error']==0.
            assert reuse['generation_n']>=8 and reuse['generation_exact_match']
            assert reuse['original_evaluation_batch_composition_preserved']
            assert reuse['reused_generation_n']==380
            refmode=reuse['reference_mode'];refpath=run/(refmode+'_val_predictions.json')
            assert sha256_file(refpath)==reuse['reference_predictions_sha256']
            assert sha256_file(run/(refmode+'_generation_policy.json'))==reuse['reference_policy_sha256']
            assert sha256_file(run/'best_generation_policy.json')==reuse['copied_best_policy_sha256']==reuse['reference_policy_sha256']
            assert raw['best']==raw[refmode]
    else:
        reuse=dict(state='not_recorded',note='Run predates endpoint generation reuse')
    source_c=Path(cfg['source_c'])
    source_original=None
    if (source_c/'best_val_metrics.json').is_file():
        assert sha256_file(source_c/'run.json')==identity['source_c_run_sha256']
        metrics=json.loads((source_c/'best_val_metrics.json').read_text())
        source_original=dict(metrics=metrics,note='Original C binary prompt, 32 generated tokens; not matched to the new 80-token structured prompt or additional mask supervision')
    fresh={}
    for mode in ('best','final'):
        path=run/('fresh_'+mode+'_reload.json')
        if path.is_file():
            audit=json.loads(path.read_text())
            assert audit['state']=='complete' and audit['fresh_reload_max_abs_score_error']==0 and audit['generated_exact_match']
            fresh[mode]=audit
        else:fresh[mode]=dict(state='missing')
    generation_policies={}
    for mode in ('initial','best','final'):
        path=run/(mode+'_generation_policy.json')
        generation_policies[mode]=json.loads(path.read_text()) if path.exists() else dict(selected='historical_scalar')
    cost={key:complete[key] for key in COST_KEYS}
    cost.update(peak_memory_gib=complete['peak_memory_bytes']/2**30,exposures=complete['exposures'],steps=complete['steps'],
                answer_tokens=config.get('target_tokens'),trainable_parameters=config['trainable_parameters'])
    return dict(run=run.name,seed=identity['seed'],best_step=complete['best_step'],identity=identity,
                modes=modes,paired_differences=dict(best_minus_initial=paired_difference(modes['initial']['flat'],modes['best']['flat']),
                final_minus_initial=paired_difference(modes['initial']['flat'],modes['final']['flat'])),
                teacher_baseline=baseline,teacher_baseline_audit=baseline_audit,original_C_binary_prompt=source_original,
                best_generation_reuse_audit=reuse,
                fresh_reload=fresh,cost=cost,generation_policies=generation_policies,predictions_sha256=raw_hashes,
                run_sha256=sha256_file(run/'run.json'),complete_sha256=sha256_file(run/'complete.json'))


def write_markdown(result,path):
    lines=['# Structured localization continuation: VisA development validation','',
           'All classification and localization results are recomputed from raw predictions. Each run uses the same380 validation images (172normal/208anomaly); seeds are training repeats, not independent test sets. Best checkpoints are selected and reported on that development validation set.','',
           '| Condition | AUROC mean±sample SD | AP | generated macro F1 | recall | FPR | format valid | anomaly mean box IoU | IoU≥.5 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    keys=('scores_image_auroc','scores_image_ap','generated_macro_f1','generated_recall','generated_fpr',
          'localization_format_valid_rate','localization_anomaly_bbox_iou_mean_all','localization_anomaly_iou_at_05_all')
    for mode in ('initial','best','final'):
        values=result['aggregate'][mode];cells=[]
        for key in keys:
            stat=values[key];cells.append('%.6f ± %.6f'%(stat['mean'],stat['sample_sd']) if stat['sample_sd'] is not None else '%.6f'%stat['mean'])
        lines.append('| '+mode+' | '+' | '.join(cells)+' |')
    lines+=['','## Paired differences','',
            'Differences are computed within matching seed before aggregation; absolute fractional differences, not relative percentages.','',
            '| Difference | AUROC mean±sample SD | generated macro F1 | anomaly mean box IoU | format valid |',
            '|---|---:|---:|---:|---:|']
    for label,values in result['aggregate_paired_differences'].items():
        cells=[]
        for key in ('scores_image_auroc','generated_macro_f1','localization_anomaly_bbox_iou_mean_all','localization_format_valid_rate'):
            stat=values[key];cells.append('%.6f ± %.6f'%(stat['mean'],stat['sample_sd']) if stat['sample_sd'] is not None else '%.6f'%stat['mean'])
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines+=['','## Teacher-map bbox controls','',
            '| Fixed control | anomaly mean box IoU | IoU≥.5 | box center in GT box |',
            '|---|---:|---:|---:|']
    for condition,values in result['aggregate_teacher_baseline'].items():
        cells=[]
        for key in ('anomaly_bbox_iou_mean_all','anomaly_iou_at_05_all','anomaly_box_center_inside_gt_box_all'):
            stat=values[key];cells.append('%.6f ± %.6f'%(stat['mean'],stat['sample_sd']) if stat['sample_sd'] is not None else '%.6f'%stat['mean'])
        lines.append('| '+condition+' | '+' | '.join(cells)+' |')
    lines+=['','## Resource and audit boundaries','',
            'Teacher-map bbox controls and complete per-seed/per-category results are in summary.json. Cache availability determines whether the teacher baseline was independently recomputed; no stored-only control is mislabeled as recomputed.','',
            'S adds mask-derived target supervision, a schema prompt, longer answers, and 2000 further image exposures while updating language LoRA. Its difference from original binary C cannot isolate a LoRA mechanism or demonstrate equal-token/equal-supervision gains. Spatial outputs are union-box predictions, not segmentation or causal explanations. Invalid/missing/No boxes score zero on all-anomaly localization metrics; conditional valid-positive metrics retain their coverage denominator.','',
            'This summary never opens external evaluation directories and does not choose a checkpoint from external data.','',
            '| Seed | best step | training seconds | exposures/s | peak GiB | best reload | final reload |',
            '|---|---:|---:|---:|---:|---|---|']
    for row in result['runs']:
        cost=row['cost'];lines.append('| %d | %d | %.2f | %.3f | %.3f | %s | %s |'%(row['seed'],row['best_step'],cost['train_seconds'],cost['exposures_per_second'],cost['peak_memory_gib'],row['fresh_reload']['best']['state'],row['fresh_reload']['final']['state']))
    Path(path).write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=[42,43,44]);ap.add_argument('--out',type=Path)
    args=ap.parse_args();assert len(set(args.seeds))==len(args.seeds)
    runs=[summarize_run(args.root/('arm_S_s%d'%seed)) for seed in args.seeds]
    assert [r['seed'] for r in runs]==args.seeds
    for key in ('schema','cache_sha256','targets_sha256','base_manifest_sha256'):
        assert len({r['identity'][key] for r in runs})==1,('Cross-seed protocol mismatch',key)
    result=dict(schema=SCHEMA,state='complete',seeds=args.seeds,runs=runs,
                aggregate={mode:aggregate([r['modes'][mode]['flat'] for r in runs]) for mode in ('initial','best','final')},
                aggregate_paired_differences={label:aggregate([r['paired_differences'][label] for r in runs]) for label in ('best_minus_initial','final_minus_initial')},
                aggregate_cost=aggregate([{k:v for k,v in r['cost'].items() if isinstance(v,(int,float))} for r in runs]),
                aggregate_teacher_baseline={condition:aggregate([{k:v for k,v in r['teacher_baseline'][condition].items() if isinstance(v,(int,float))} for r in runs]) for condition in ('unconditional_box','initial_score_gated')},
                fresh_reload_all_passed=all(r['fresh_reload'][mode]['state']=='complete' for r in runs for mode in ('best','final')),
                scope='VisA val380 only; no external results opened; original C frozen per seed',
                interpretation='Additional mask supervision, prompt/answer tokens and continuation compute differ from original C; no equal-token or LoRA-only causal attribution.',
                summarizer_sha256=sha256_file(__file__))
    out=args.out or args.root/'structured_summary';out.mkdir(parents=True,exist_ok=True)
    atomic_json(result,out/'summary.json');write_markdown(result,out/'RESULTS.md')
    print(json.dumps(dict(state='complete',seeds=args.seeds,summary=str(out/'summary.json'),report=str(out/'RESULTS.md'))))


if __name__=='__main__':main()
