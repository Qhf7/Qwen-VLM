"""Strict CPU-only saved-result report: --root runs/migration_20261103.
Reads JSON/JSONL and source bytes only. Never loads images/maps/checkpoints/GPU.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from migration_metrics import evaluate_predictions, parse_generated

SCHEMA = 'student-roi-comparison-summary-v1'
N = 380
MATCHED = 'e79f550e88f58e8a51829608f716fb0049e057cea8beadbb73f6b5ccb827dacf'
ARMS = [('center', 'original_center_crop', 'center_predictions.jsonl', '旧中心裁剪'),
        ('random', 'original_random_crop', 'random_predictions.jsonl', '旧随机裁剪'),
        ('teacher', 'original_teacher_roi', 'teacher_roi_predictions.jsonl', '旧BigDec teacher ROI'),
        ('pre_gt', 'pre_gt', 'roi_full/pre_gt_predictions.jsonl', 'Qwen pre448 GT ROI'),
        ('pre_kd', 'pre_kd', 'roi_full/pre_kd_predictions.jsonl', 'Qwen pre448 GT＋KD ROI')]
TOKEN = {'image_grid_thw': [[1,28,28],[1,28,28]], 'image_tokens_per_view': [196,196],
         'image_tokens_total':392, 'prompt_tokens_total':470, 'patch_size':16,
         'spatial_merge_size':2, 'n_images':2}
PRED_KEYS = ('yes_logit','no_logit','score_yes_minus_no','generated_text','generated_token_ids')
METRICS = ('image_auroc','image_ap','macro_f1','recall','fpr','invalid_rate')


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def need(ok,message):
    if not ok:
        raise ValueError(message)


def finite(x):
    return isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x)


def is_sha(x):
    return isinstance(x,str) and re.fullmatch('[0-9a-f]{64}',x) is not None


def atomic_json(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def compact(m,category=False):
    x=m['category_macro' if category else 'overall']
    return {'image_auroc':x['scores']['image_auroc'],'image_ap':x['scores']['image_ap'],
            **{k:x['generated'][k] for k in ('macro_f1','recall','fpr','invalid_rate')}}


def validate_rows(rows,items,condition,student=False):
    need(len(rows)<=N,condition+': excess rows')
    need([r.get('path') for r in rows]==[x['path'] for x in items[:len(rows)]],condition+': ordered path mismatch')
    need(len({r['path'] for r in rows})==len(rows),condition+': duplicate path')
    for i,(r,item) in enumerate(zip(rows,items)):
        tag=condition+': row '+str(i)
        need(r.get('condition')==condition,tag+' wrong condition')
        for k in ('path','category','label','split','cache_index'):
            need(k in r and r[k]==item[k] and (k!='label' or type(r[k]) is int),tag+' metadata differs: '+k)
        need(all(r.get(k)==v for k,v in TOKEN.items()),tag+' actual tokens/grid differ')
        need(r.get('view_roles')==['target_original','target_roi'] and r.get('view_sources')==[item['path']]*2,tag+' wrong image roles/sources')
        need(r.get('target_path',r['path'])==r['path'] and r.get('reference') is None,tag+' target/reference differs')
        need(all(finite(r.get(k)) for k in ('yes_logit','no_logit','score_yes_minus_no')),tag+' nonfinite score')
        need(r['score_yes_minus_no']==r['yes_logit']-r['no_logit'],tag+' score != Yes-No')
        ids=r.get('generated_token_ids')
        need(isinstance(r.get('generated_text'),str) and isinstance(ids,list) and 1<=len(ids)<=32 and all(type(x) is int and x>=0 for x in ids),tag+' invalid generation')
        need(all(finite(r.get(k)) and r[k]>=0 for k in ('prepare_seconds','score_seconds','generate_seconds','elapsed_seconds')),tag+' invalid timing')
        roi=r.get('roi',{});box=roi.get('bbox_xyxy_relative',[])
        need(len(box)==4 and all(finite(x) for x in box),tag+' bad bbox')
        l,t,rr,b=box
        need(0<=l<rr<=1 and 0<=t<b<=1,tag+' out-of-RGB bbox')
        need(roi.get('area_fraction')==1/9 and math.isclose(rr-l,1/3,rel_tol=0,abs_tol=1e-12) and math.isclose(b-t,1/3,rel_tol=0,abs_tol=1e-12),tag+' changed area')
        wh=r.get('source_image_wh');raw=roi.get('bbox_xyxy_raw_pixels_float')
        need(isinstance(wh,list) and len(wh)==2 and all(type(x) is int and x>0 for x in wh),tag+' bad raw size')
        need(isinstance(raw,list) and len(raw)==4 and all(math.isclose(x,y,rel_tol=0,abs_tol=1e-9) for x,y in zip(raw,(l*wh[0],t*wh[1],rr*wh[0],b*wh[1]))),tag+' raw/normalized bbox differs')
        if student:
            peak=roi.get('student_peak_row_col');center=roi.get('requested_center_xy_relative')
            need(roi.get('kind')=='student' and roi.get('student_map_hw')==[224,224] and roi.get('argmax_tie_rule')=='first in row-major order',tag+' wrong selector')
            need(isinstance(peak,list) and len(peak)==2 and all(type(x) is int and 0<=x<224 for x in peak),tag+' bad peak')
            need(finite(roi.get('student_peak_value')) and 0<=roi['student_peak_value']<=1,tag+' bad peak probability')
            need(center==[(peak[1]+.5)/224,(peak[0]+.5)/224],tag+' peak center differs')
            left=min(max(center[0]-1/6,0.),1-1/3);top=min(max(center[1]-1/6,0.),1-1/3)
            need(all(math.isclose(x,y,rel_tol=0,abs_tol=1e-12) for x,y in zip(box,(left,top,left+1/3,top+1/3))),tag+' shifted crop differs')


def decision_counts(rows,score=False):
    c={k:0 for k in ('tp','fn','fp','tn','invalid_normal','invalid_anomaly')}
    for r in rows:
        pred=int(r['score_yes_minus_no']>0) if score else parse_generated(r['generated_text'])
        if pred is None:c['invalid_anomaly' if r['label'] else 'invalid_normal']+=1
        if r['label']:c['tp' if pred==1 else 'fn']+=1
        elif pred==1:c['fp']+=1
        elif pred==0:c['tn']+=1
    return c


def paired_changes(old,new):
    oldby={r['path']:r for r in old};newby={r['path']:r for r in new}
    need(set(oldby)==set(newby) and len(oldby)==len(old)==len(new), 'paired paths differ')
    out={}
    for label,name in [(0,'normal'),(1,'anomaly')]:
        counts={k:0 for k in ('n','old_correct_new_wrong','old_wrong_new_correct','same_decision','old_invalid','new_invalid','both_invalid')}
        transitions={}
        for path,r in oldby.items():
            if r['label']!=label:continue
            other=newby[path];need(other['label']==label,'paired label differs')
            a=parse_generated(r['generated_text']);b=parse_generated(other['generated_text'])
            counts['n']+=1;counts['old_correct_new_wrong']+=int(a==label and b!=label)
            counts['old_wrong_new_correct']+=int(a!=label and b==label);counts['same_decision']+=int(a==b)
            counts['old_invalid']+=int(a is None);counts['new_invalid']+=int(b is None);counts['both_invalid']+=int(a is None and b is None)
            k=('invalid' if a is None else str(a))+'->'+('invalid' if b is None else str(b));transitions[k]=transitions.get(k,0)+1
        out[name]={**counts,'transition_matrix':transitions}
    return out


def build_summary(root):
    root=Path(root).resolve();base=root/'student_roi_20261104'
    s={'schema_version':SCHEMA,'created_at_utc':datetime.now(timezone.utc).isoformat(),'root':str(root),'status':'pending',
       'expected_n':N,'errors':[],'pending':[],'checks':{},'sources':{},'arms':{},'deltas':{},'paired_generated_changes':{},
       'map_export':None,'map_diagnostics':None,'replay':None,
       'input_scope':'Saved JSON/JSONL and source byte hashes only; no image, .npy, checkpoint, GPU, SSH or test inference.'}
    def read(relative,lines=False,optional=False):
        path=root/relative
        if not path.exists():
            if not optional:s['pending'].append('Missing '+str(relative))
            return None
        raw=path.read_bytes();s['sources'][str(relative)]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
        try:return [json.loads(x) for x in raw.decode().splitlines() if x.strip()] if lines else json.loads(raw)
        except Exception as e:s['errors'].append(str(relative)+': '+str(e));return None
    def check(name,fn):
        try:fn();s['checks'][name]={'passed':True}
        except Exception as e:s['checks'][name]={'passed':False,'error':str(e)};s['errors'].append(name+': '+str(e))
    old=read(Path('roi_full/run.json'));val=read(Path('student_roi_20261104/val_manifest.json'))
    maps=read(Path('student_roi_20261104/maps/manifest.json'));run=read(Path('student_roi_20261104/roi_full/run.json'))
    if old is None or val is None:
        s['status']='failed' if s['errors'] else 'pending';return s
    items=val.get('items',[])
    def identity():
        need(len(items)==N and len({x['path'] for x in items})==N,'need 380 unique items')
        need(all(x.get('split')=='val' and type(x.get('label')) is int and x['label'] in (0,1) for x in items),'bad val labels/split')
        need(len({x['category'] for x in items})==12 and sum(x['label'] for x in items)==208,'need 12 classes / normal172 / anomaly208')
        need(old.get('status')=='complete' and old.get('n')==old.get('full_validation_n')==N and old.get('split')=='val','legacy control not full complete')
        need(old['selected_paths']==[x['path'] for x in items] and old['selected_cache_indices']==[x['cache_index'] for x in items],'legacy order/identity differs')
        minimal=[{k:x[k] for k in ('path','category','label','split')} for x in items]
        digest=hashlib.sha256(json.dumps(minimal,sort_keys=True).encode()).hexdigest()
        need(digest==old['selected_manifest_sha256']==val['selected_manifest_sha256'],'legacy manifest SHA differs')
        need(val['source_roi_run_sha256']==s['sources']['roi_full/run.json']['sha256'],'validation source run changed')
        need(val.get('test_set_used') is False and val.get('n')==N and val['selected_items_sha256']==digest,'val manifest metadata differs')
        cfg=old['config'];need(cfg['image_size']==448 and cfg['roi_area']==1/9 and cfg['max_new_tokens']==32 and cfg['seed']==42,'legacy protocol differs')
        s['identity']={'items_sha256':canonical(items),'legacy_selected_manifest_sha256':digest,'ordered_paths_sha256':canonical([x['path'] for x in items]),'n':N,'categories':12,'normal':172,'anomaly':208}
    check('validation_identity',identity)
    if not s['checks']['validation_identity']['passed']:s['status']='failed';return s
    allrows={}
    for key,condition,filename,label in ARMS:
        rows=read(Path('student_roi_20261104')/filename,lines=True)
        expected=(old if key in ('center','random','teacher') else (run or {})).get('results',{}).get(condition)
        arm={'label':label,'condition':condition,'status':'pending','n_observed':None if rows is None else len(rows),
             'metrics':None,'overall':None,'category_macro':None,'generated_counts':None,'score_counts':None,'timing':None}
        s['arms'][key]=arm
        if rows is None:continue
        check(key+'_prediction_contract',lambda rows=rows,c=condition,student=key.startswith('pre_'):validate_rows(rows,items,c,student))
        if not s['checks'][key+'_prediction_contract']['passed']:arm['status']='failed';continue
        if len(rows)!=N:s['pending'].append(key+': only '+str(len(rows))+'/380 prediction rows');continue
        if expected is None:s['pending'].append(key+': result summary not yet available');continue
        def verify_metrics():
            need(expected.get('n')==N,key+': wrong summary denominator')
            if key.startswith('pre_') and maps is not None:
                old_by_path={x['path']:x for x in allrows.get('teacher',[])}
                for row in rows:
                    record=maps['heads'][key]['maps'][row['path']]
                    need(row['roi']['student_peak_value']==record['max'],key+': ROI peak differs from selected head map maximum')
                    if old_by_path:need(row['source_image_wh']==old_by_path[row['path']]['source_image_wh'],key+': original RGB dimensions changed')

            m=evaluate_predictions(items,scores=[r['score_yes_minus_no'] for r in rows],generated_texts=[r['generated_text'] for r in rows])
            need(m==expected['metrics'],key+': independently recomputed metrics differ (exact object comparison)')
            need(expected.get('visual_tokens_total_min_max')==[392,392] and expected.get('prompt_tokens_total_min_max')==[470,470] and expected.get('actual_image_grids')==[[1,28,28]],key+': summary geometry differs')
            sums={k:sum(r[k] for r in rows) for k in ('prepare_seconds','score_seconds','generate_seconds','elapsed_seconds')}
            for field,source in [('score_seconds_total','score_seconds'),('generate_seconds_total','generate_seconds'),('prepare_seconds_total','prepare_seconds')]:
                if field in expected:need(expected[field]==sums[source],key+': timing sum differs: '+field)
            arm.update(status='complete',metrics=m,overall=compact(m),category_macro=compact(m,True),generated_counts=decision_counts(rows),score_counts=decision_counts(rows,True),
                       timing={'condition_wall_seconds':expected.get('elapsed_seconds'),'prepare_seconds_sum':sums['prepare_seconds'],'score_seconds_sum':sums['score_seconds'],
                               'generate_seconds_sum':sums['generate_seconds'],'row_elapsed_seconds_sum':sums['elapsed_seconds'],'map_extraction_included':False,
                               **{k:expected.get(k) for k in ('peak_gpu_allocated_bytes','peak_gpu_reserved_bytes','baseline_gpu_allocated_bytes')}})
            allrows[key]=rows
        check(key+'_metrics_exact',verify_metrics)
        if not s['checks'][key+'_metrics_exact']['passed']:arm['status']='failed'
    if maps is not None:
        def map_identity():
            need(maps.get('schema_version')=='qwen-dense-map-export-v1' and maps.get('status')=='complete','map export not complete')
            payload=dict(maps);digest=payload.pop('payload_sha256',None);need(digest==canonical(payload),'map payload SHA differs')
            need(maps['items']==items and maps['items_sha256']==canonical(items) and maps['n_images']==N,'map item identity differs')
            need(maps['source_manifest']['sha256']==s['sources']['student_roi_20261104/val_manifest.json']['sha256'],'map source manifest differs')
            need(all(maps.get(k) is True for k in ('no_teacher','no_imagebind','no_gt_masks')) and all(maps.get(k) is False for k in ('test_set_used','train_fit','language_model_instantiated')),'map inference dependency audit failed')
            g=maps['geometry'];need(g['coordinate_system']=='direct-square-224' and g['map_hw']==[224,224] and g['requested_image_hw']==[448,448] and g['processor_do_resize'] is False,'map geometry differs')
            need(set(maps['heads'])=={'pre_gt','pre_kd'} and maps['heads']['pre_gt']['compatibility']==maps['heads']['pre_kd']['compatibility'],'map head feature pairing differs')
            paths=[x['path'] for x in items];need(set(maps['input_sha256'])==set(paths) and all(is_sha(x) for x in maps['input_sha256'].values()),'image SHA coverage differs')
            for name,weight in [('pre_gt',0.),('pre_kd',.5)]:
                h=maps['heads'][name];need(h['features']=='pre' and h['size']==448 and h['step']==1200 and h['distill_weight']==weight and h['matched_run_key']==MATCHED and is_sha(h['sha256']) and is_sha(h['state_sha256']),'wrong selected head: '+name)
                need(set(h['maps'])==set(paths),'map coverage differs')
                for path,r in h['maps'].items():
                    need(all(is_sha(r[k]) for k in ('sha256','array_sha256','input_sha256','head_sha256','source_sha256')),'missing map SHA')
                    need(r['input_sha256']==maps['input_sha256'][path] and r['head_sha256']==h['sha256'] and r['source_sha256']==h['compatibility']['source_sha256'],'map input/head/source differs')
                    need(r['shape']==[224,224] and r['dtype']=='float32' and 0<=r['min']<=r['max']<=1,'map values metadata differs')
                    gg=r['geometry'];need(gg['requested_image_hw']==gg['actual_image_hw']==[448,448] and gg['processor_do_resize'] is False and gg['image_grid_thw']==[1,28,28] and gg['pre_grid_shape']==[1,28,28,1152],'map actual geometry differs')
                    other=maps['heads']['pre_kd' if name=='pre_gt' else 'pre_gt']['maps'][path]
                    need(r['inference_seconds']==other['inference_seconds'],'shared feature timing differs')
            need(maps['feature_extractions']==N and maps['shared_visual_features_across_heads'] is True,'shared feature count differs')
            timing={k:maps.get(k) for k in ('elapsed_seconds','visual_load_seconds','export_map_seconds','inference_seconds_total','serialization_seconds_total')}
            need(all(finite(x) and x>=0 for x in timing.values()),'missing/invalid map timing')
            need(sum(x['inference_seconds'] for x in maps['heads']['pre_gt']['maps'].values())==timing['inference_seconds_total'],'shared inference sum differs')
            need(sum(x['serialization_seconds'] for h in maps['heads'].values() for x in h['maps'].values())==timing['serialization_seconds_total'],'serialization sum differs')
            s['map_export']={'status':'complete','n':N,'feature_extractions':N,'shared_visual_features_across_heads':True,'timing':timing,
                             'heads':{n:{k:h[k] for k in ('sha256','state_sha256','features','size','step','distill_weight','matched_run_key')} for n,h in maps['heads'].items()},
                             'input_sha256_manifest':canonical(maps['input_sha256']),'payload_sha256':digest,'visual_weight_identity':maps.get('visual_weight_identity'),
                             'evidence_boundary':'Saved manifest identity/metadata checked; raw image/.npy/checkpoint byte verification is recorded by exporter/evaluator, not repeated by this summarizer.'}
        check('map_export_identity',map_identity)
    if run is not None:
        def student_identity():
            need(run.get('n')==N and run.get('split')=='val' and run['selected_paths']==[x['path'] for x in items] and run['items_sha256']==canonical(items) and set(run['conditions'])=={'pre_gt','pre_kd'},'student data/conditions differ')
            need(run['question']==old['question'] and run['image_size']==448 and run['roi_area']==1/9 and run['max_new_tokens']==32,'student prompt/protocol differs')
            need(run['config']['model']==old['config']['model'],'native model path differs')
            need(run['control_run_sha256']==s['sources']['roi_full/run.json']['sha256'] and run['control_predictions_sha256']==s['sources']['student_roi_20261104/teacher_roi_predictions.jsonl']['sha256'],'legacy source file changed')
            if maps is not None:need(run['map_manifest_sha256']==s['sources']['student_roi_20261104/maps/manifest.json']['sha256'],'student map manifest changed')
            need(all(run.get(k) is False for k in ('teacher_model_used','teacher_cache_read','imagebind_used','gt_masks_read','train_fit','test_set_used')),'student inference dependency audit failed')
            need(run.get('source_audit',{}).get('passed') is True and run.get('dtype_audit',{}).get('passed') is True,'source/dtype audit missing/failed')
            need(all(run['source_audit'][k]==old['source_audit'][k] for k in ('source_sha256','model_config','processor_config','api')),'native source/model/processor differs')
            rt=run['runtime'];need(rt['attention_backend']=='sdpa' and rt['yes_id']==9454 and rt['no_id']==2753,'runtime differs')
            for name in ('pre_gt','pre_kd'):
                if maps is not None:
                    h=maps['heads'][name];a=run['map_audits'][name]
                    need(a['maps_verified']==N and a['checkpoint_sha256']==h['sha256'] and all(a[k]==h[k] for k in ('features','size','step','distill_weight')),'map consumer audit differs: '+name)
        check('student_run_identity',student_identity)
        def replay_check():
            need('teacher' in allrows,'validated teacher predictions unavailable')
            rows=allrows['teacher'];cats=sorted({r['category'] for r in rows})[:3]
            expected=[next(r for r in rows if r['category']==c and r['label']==label) for c in cats for label in (0,1)]
            replay=run.get('replay',{});cases=replay.get('cases',[])
            need(replay.get('passed') is True and replay.get('n')==len(cases)==6 and [c['path'] for c in cases]==[r['path'] for r in expected],'fixed replay six cases absent/differ')
            for case,row in zip(cases,expected):
                need(case.get('passed') is True and case['max_abs_score_error']==0 and case['token_budget']==TOKEN,'replay status/tokens/score not exact')
                need(all(case['expected'][k]==case['actual'][k]==row[k] for k in PRED_KEYS),'replay content not exact')
                a=case['actual'];need(a['score_yes_minus_no']==a['yes_logit']-a['no_logit'],'replay score differs')
            s['replay']={'status':'complete','passed':True,'n':6,'max_abs_score_error':0.,'paths':[c['path'] for c in cases],
                         'scope':'Recorded teacher boxes only; three categories x normal/anomaly, not 380 individually replayed.'}
        if run.get('replay',{}).get('passed') is True or run.get('status')=='complete':check('six_case_exact_replay',replay_check)
        else:s['pending'].append('Six-case exact replay not yet available')
        if run.get('status')!='complete':
            if run.get('status')=='failed':s['errors'].append('Student run failed: '+str(run.get('error')))
            else:s['pending'].append('Student run status='+str(run.get('status')))
    sources=read(Path('student_roi_20261104/code/source_manifest.json'))
    if sources is not None:
        def source_check():
            for name,r in sources.items():
                p=base/'code'/name;need(p.is_file() and sha_file(p)==r['sha256'] and p.stat().st_size==r['bytes'],'executed source snapshot differs: '+name)
                s['sources'][str(p.relative_to(root))]={'sha256':sha_file(p),'bytes':p.stat().st_size}
            if run is not None:need(sources['migration_student_roi.py']['sha256']==run['script_sha256'],'executed evaluator differs')
            if maps is not None:need(sources['migration_dense_infer.py']['sha256']==maps['script_sha256'],'executed exporter differs')
        check('executed_source_snapshot',source_check)
    metric_snapshot=root/'code'/'migration_metrics.py'
    if metric_snapshot.exists():
        def metric_source_check():
            current=Path(evaluate_predictions.__code__.co_filename).resolve()
            need(sha_file(metric_snapshot)==sha_file(current),'metric implementation differs from historical executed snapshot')
            s['sources']['code/migration_metrics.py']={'sha256':sha_file(metric_snapshot),'bytes':metric_snapshot.stat().st_size}
        check('historical_metric_implementation',metric_source_check)
    diagnostic=read(Path('student_roi_20261104/map_metrics.json'),optional=True)
    if diagnostic is not None:
        def diagnostic_check():
            need(diagnostic['n_images']==N and diagnostic['test_set_used'] is False and diagnostic['gpu_used'] is False,'post-prediction diagnostic scope differs')
            need(diagnostic['manifest_sha256']==s['sources']['student_roi_20261104/maps/manifest.json']['sha256'],'diagnostic map source changed')
            for name,dirname in [('pre_gt','dense_pre_gt_s42'),('pre_kd','dense_pre_kd_s42')]:
                prior=read(Path(dirname)/'best_validation_reloaded.json');need(prior is not None,'missing prior dense reload')
                h=diagnostic['heads'][name];m=h['metrics'];need(m['n_images']==N and m['n_categories']==m['n_categories_total']==12,'map diagnostic denominator differs')
                keys=('pixel_ap_macro','pixel_auroc_macro','pixel_auroc_weighted','image_auroc_from_map99')
                need(all(m[k]==prior['validation_reloaded'][k] for k in keys),'export map metrics differ from original selected reload')
                need(h['anomaly_n']==208 and 0<=h['peak_hits']<=208 and h['peak_in_gt_rate']==h['peak_hits']/208 and 0<=h['crop_contains_any_gt_n']<=208 and 0<=h['mean_gt_foreground_in_crop']<=1,'crop diagnostic counts differ')
            s['map_diagnostics']=diagnostic
        check('post_prediction_map_diagnostics',diagnostic_check)
    for left,right in [(a,b) for a in ('pre_gt','pre_kd') for b in ('teacher','center','random')]+[('pre_kd','pre_gt')]:
        a=s['arms'].get(left,{});b=s['arms'].get(right,{})
        if a.get('status')==b.get('status')=='complete':
            s['deltas'][left+'_minus_'+right]={scope+'_percentage_points':{k:100*(a[scope][k]-b[scope][k]) for k in METRICS} for scope in ('overall','category_macro')}
    if 'teacher' in allrows:
        for name in ('pre_gt','pre_kd'):
            if name in allrows:s['paired_generated_changes'][name+'_vs_teacher']=paired_changes(allrows['teacher'],allrows[name])
    s['sources']['summarizer']={'path':str(Path(__file__).resolve()),'sha256':sha_file(__file__)}
    metric_path=Path(evaluate_predictions.__code__.co_filename).resolve();s['sources']['metrics_implementation']={'path':str(metric_path),'sha256':sha_file(metric_path)}
    complete=all(s['arms'].get(a[0],{}).get('status')=='complete' for a in ARMS) and maps is not None and run is not None and run.get('status')=='complete' and (s.get('replay') or {}).get('passed') is True and not s['pending']
    s['status']='failed' if s['errors'] else ('complete' if complete else 'pending')
    return s


def render_markdown(s):
    fmt=lambda x:'pending' if x is None else f'{x:.6f}'
    lines=['# 学生地图替换 ROI 选择器：验证结果','','日期：'+s['created_at_utc']+'。状态：**'+s['status']+'**。','',
           '同一val380：正常172、异常208、12类，seed42。五臂均为原图＋ROI两图、448/图、196×2=392 visual tokens、总prompt470、normalized面积1/9；同QUESTION、BF16原生Qwen/SDPA、greedy最多32 tokens。','',
           '## 完整集指标','','AUROC/AP来自LM首答案raw logit(Yes)−logit(No)，其他指标来自独立free generation。生成macro F1是完整集二分类macro F1，类别等权另列。pending不以0补齐。','',
           '| 条件 | 状态 / 已见行数 | Image AUROC | Image AP | 生成 F1 | recall | FPR | invalid |','|---|---|---:|---:|---:|---:|---:|---:|']
    for key,_,_,label in ARMS:
        a=s['arms'].get(key,{});m=a.get('overall') or {};lines.append('| '+label+' | '+str(a.get('status','pending'))+' / '+str(a.get('n_observed'))+' | '+' | '.join(fmt(m.get(k)) for k in METRICS)+' |')
    lines+=['','## 类别等权','','| 条件 | Image AUROC | Image AP | 生成 F1 | recall | FPR | invalid |','|---|---:|---:|---:|---:|---:|---:|']
    for key,_,_,label in ARMS:
        m=s['arms'].get(key,{}).get('category_macro') or {};lines.append('| '+label+' | '+' | '.join(fmt(m.get(k)) for k in METRICS)+' |')
    lines+=['','## 差值与取舍','','Δ=前者−后者，百分点；FPR/invalid越低越好。summary另保存类别等权差值。','','| 比较 | Δ AUROC | Δ AP | Δ F1 | Δ recall | Δ FPR | Δ invalid |','|---|---:|---:|---:|---:|---:|---:|']
    for name,d in s['deltas'].items():lines.append('| '+name+' | '+' | '.join(f'{d["overall_percentage_points"][k]:+.6f}' for k in METRICS)+' |')
    if not s['deltas']:lines.append('| 学生结果尚未完整核验 | pending | pending | pending | pending | pending | pending |')
    if 'pre_kd_minus_pre_gt' in s['deltas']:
        d=s['deltas']['pre_kd_minus_pre_gt']['overall_percentage_points'];lines+=['',f'KD−GT的AUROC/AP/F1为{d["image_auroc"]:+.3f}/{d["image_ap"]:+.3f}/{d["macro_f1"]:+.3f}个百分点；recall/FPR变化{d["recall"]:+.3f}/{d["fpr"]:+.3f}个百分点。排序、召回和误报需共同解释，不能将单项变化概括为普遍改善。']
    lines+=['','## 生成计数','','FN含异常invalid；正常invalid单列而不充当TN。','','| 条件 | TP / 208 | FN / 208 | FP / 172 | TN / 172 | 正常invalid | 异常invalid |','|---|---:|---:|---:|---:|---:|---:|']
    for key,_,_,label in ARMS:
        c=s['arms'].get(key,{}).get('generated_counts') or {};lines.append('| '+label+' | '+' | '.join('pending' if c.get(k) is None else str(c[k]) for k in ('tp','fn','fp','tn','invalid_normal','invalid_anomaly'))+' |')
    lines+=['','## 真正逐图配对变化','','按完整path匹配teacher与student生成决策，正常/异常分别统计。same_decision含both-invalid；invalid另列属于重叠描述，不能与各列相加当互斥分组。完整0/1/invalid转移矩阵保存在summary，以下正确/错误变化来自逐图片配对，非仅净TP/FP差。','','| Student vs teacher | 真实标签 | n | old正确→new错 | old错→new正确 | 相同决策 | old invalid | new invalid | 双invalid |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,groups in s['paired_generated_changes'].items():
        for label,c in groups.items():lines.append('| '+name+' | '+label+' | '+' | '.join(str(c[k]) for k in ('n','old_correct_new_wrong','old_wrong_new_correct','same_decision','old_invalid','new_invalid','both_invalid'))+' |')
    if not s['paired_generated_changes']:lines.append('| pending | pending | pending | pending | pending | pending | pending | pending | pending |')
    lines+=['','## 地图与crop离线诊断','','GT mask仅在预测完成后的独立CPU评估脚本中使用，不进入student infer或ROI中心选择；地图指标不是LM分数。']
    diag=s.get('map_diagnostics')
    if diag:
        lines+=['','| Head | Pixel AP macro | Pixel AUROC macro | Weighted pixel AUROC | Map-p99 image AUROC | Peak hit / 208 | Mean GT前景保留比例 | Crop含任意GT / 208 |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for name,h in diag['heads'].items():
            m=h['metrics'];lines.append('| '+name+' | '+' | '.join(fmt(m[k]) for k in ('pixel_ap_macro','pixel_auroc_macro','pixel_auroc_weighted','image_auroc_from_map99'))+' | '+str(h['peak_hits'])+' | '+fmt(h['mean_gt_foreground_in_crop'])+' | '+str(h['crop_contains_any_gt_n'])+' |')
        lines+=['','导出地图四项指标与原selected checkpoint reload记录完全相等。crop覆盖定义为224二值GT像素中心落入normalized框，属于后验诊断，不能当作推理使用GT或语言模型确实关注缺陷的证据。']
    else:lines+=['','未同步独立map诊断；不以0补齐。']
    lines+=['','## 计时与地图成本','','秒。condition wall和prepare/score/generate不含预计算地图；score与generate各有独立prefill。旧teacher图也已预计算，其离线地图成本本次未测。','','| 条件 | condition wall | prepare合计 | score合计 | generate合计 | allocated peak GiB |','|---|---:|---:|---:|---:|---:|']
    for key,_,_,label in ARMS:
        t=s['arms'].get(key,{}).get('timing') or {};peak=t.get('peak_gpu_allocated_bytes');lines.append('| '+label+' | '+' | '.join(fmt(t.get(k)) for k in ('condition_wall_seconds','prepare_seconds_sum','score_seconds_sum','generate_seconds_sum'))+' | '+fmt(None if peak is None else peak/1024**3)+' |')
    if s.get('map_export'):
        lines+=['','380次visual提取由两个head共享，成本只计一次；不是两臂各自独立部署延迟。','','| 地图阶段 | 秒 |','|---|---:|']+['| '+k+' | '+fmt(v)+' |' for k,v in s['map_export']['timing'].items()]
    lines+=['','两图/token预算不匹配地图产生的额外计算。map job wall含CPU审计/hash/加载；export/inference/serialization互有包含，不能全部相加。地图读取校验在ROI条件计时前完成；未记录的在线端到端延迟不作推算。','',
           '## 完成与证据','','五臂各380条path唯一且原序一致，label/category/split/cache_index与旧manifest逐项一致，actual tokens/geometry一致；独立CPU重算metrics与run完整metric对象exact对照，另验map payload/head/input/source身份及执行源码SHA。']
    lines+=['','固定旧teacher框replay：'+('6/6通过，logits/score/generated IDs/text/tokens exact；只覆盖三类×正常/异常各一例。' if s.get('replay') else 'pending，尚无全部六例exact证据。')]
    if s['pending']:lines+=['','待完成：']+['- '+x for x in s['pending']]
    if s['errors']:lines+=['','核验失败：']+['- '+x for x in s['errors']]
    lines+=['','## 解释边界','',
            '- 学生组合推理用Qwen视觉塔＋SpatialHead定位，再把原RGB和ROI交给完整Qwen。学生infer不加载ImageBind/BigDec或teacher概率图，旧teacher框只在独立连续性replay使用；历史teacher对照仍使用BigDec地图。',
            '- GT head训练用过异常GT mask；KD head还用过GT与weight0.5的训练teacher概率监督。推理移除外部teacher依赖不等于无teacher训练、无监督或纯原生Qwen零样本定位。',
            '- 两head由同开发val选择best1200，这里再用val380报告ROI；无新增test查询、无独立holdout、单seed42，无显著性或跨seed稳定性证明。',
            '- 汇总器只读JSON/JSONL及源码字节，不重开原图/.npy/checkpoint，不SSH/GPU。原图/数组/权重字节核验由exporter/evaluator执行并记录，此处核验其manifest身份链。','',
            '## 来源 SHA256','','| 本地来源 | SHA256 |','|---|---|']
    lines+=['| '+name+' | '+r['sha256']+' |' for name,r in s['sources'].items()]
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    args=parser.parse_args(argv);s=build_summary(args.root);base=Path(args.root).resolve()/'student_roi_20261104'
    atomic_json(s,base/'summary.json');report=base/'STUDENT_ROI_RESULTS.md';temp=report.with_suffix('.md.tmp');temp.write_text(render_markdown(s));temp.replace(report)
    print(json.dumps({'status':s['status'],'schema_version':SCHEMA,'errors':s['errors'],'pending':s['pending'],'summary':str(base/'summary.json'),'report':str(report)},ensure_ascii=False))
    return 1 if s['status']=='failed' else 0


if __name__=='__main__':
    raise SystemExit(main())
