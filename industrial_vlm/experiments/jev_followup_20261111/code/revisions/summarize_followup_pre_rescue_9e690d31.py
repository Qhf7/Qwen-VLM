#!/usr/bin/env python3
"""Server-only summaries of frozen experiments; no fitting or model/API calls.

audit-qwen --root ROOT
summary --root ROOT
self-test

summary publishes final artifacts only after stage08=600, stage09=240 and
stage10=120 are complete and evaluated. Otherwise it writes a pending status,
without partial scientific statistics. Only approved120 boolean labels are read.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import numpy as np

HOSTNAME='autodl-container-132444827f-9c594038'
SCHEMA='jev-qwen-followup-final-summary-v1'
BOOTSTRAP=dict(replicates=2000,seed=42,strata='category x image_label',paired=True,
    includes_provider_repeat_randomness=False,includes_training_seed_variance=False,
    includes_support_selection_variance=False,multiplicity_corrected=False)
METRICS=('auroc','ap','macro_f1','recall','fpr')

def require(ok,message):
    if not ok:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def host_gate():require(socket.gethostname()==HOSTNAME,'Execute only on designated AutoDL server')
def rows(path):
    raw=Path(path).read_bytes();require(not raw or raw.endswith(b'\n'),'Incomplete JSONL tail')
    return [json.loads(line) for line in raw.splitlines() if line.strip()]
def publish(path,value,text=False):
    path=Path(path);raw=value.encode() if text else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode()+b'\n'
    if path.exists():require(path.read_bytes()==raw,'Frozen output differs: '+str(path));return
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with temporary.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    try:os.link(temporary,path)
    finally:temporary.unlink(missing_ok=True)
def mutable_status(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:stream.write(json.dumps(value,ensure_ascii=False,indent=2).encode()+b'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
@contextmanager
def locked(root):
    with (root/'.summary.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)

def identities(root):
    base=root/'source_baseline/frozen';plan,labels,config=[read(base/(n+'.json')) for n in ('plan','labels','config')]
    require(digest(plan)==config['plan_sha256'] and digest(labels)==config['labels_sha256'],'Frozen query/label identity changed')
    q=sorted(plan['queries'],key=lambda x:x['source_id']);ids=[x['source_id'] for x in q]
    require(len(q)==len(set(ids))==120 and set(ids)==set(labels),'Need exactly approved120 labels')
    require(all(type(labels[s]) is int and labels[s] in (0,1) for s in ids),'Invalid boolean labels')
    counts=Counter((x['category'],labels[x['source_id']]) for x in q)
    require(len(counts)==24 and set(counts.values())=={5},'Expected12 categories x2 labels x5')
    return q,np.array([labels[s] for s in ids],dtype=int)

def weighted(y,scores,prediction,weights):
    """Tie-aware rank and fixed-working-point metrics; no fitting/jitter."""
    y=np.asarray(y,dtype=int);scores=np.asarray(scores,dtype=float);prediction=np.asarray(prediction,dtype=bool)
    w=np.asarray(weights,dtype=float);require(w.ndim==2 and w.shape[1]==len(y),'Weight dimensions differ')
    pos=w@y;neg=w@(1-y);require(np.all(pos>0) and np.all(neg>0),'Both labels required')
    order=np.argsort(-scores,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(scores[order])!=0)+1]
    gp=np.add.reduceat(w[:,order]*y[order][None,:],starts,axis=1)
    gn=np.add.reduceat(w[:,order]*(1-y[order])[None,:],starts,axis=1)
    tp_rank=np.cumsum(gp,axis=1);fp_rank=np.cumsum(gn,axis=1);prev=np.c_[np.zeros(len(w)),tp_rank[:,:-1]]
    auc=np.sum(gn*(tp_rank+prev)*.5,axis=1)/(pos*neg)
    precision=np.divide(tp_rank,tp_rank+fp_rank,out=np.zeros_like(tp_rank),where=tp_rank+fp_rank>0)
    ap=np.sum(gp*precision,axis=1)/pos
    tp=w@(y*prediction);fp=w@((1-y)*prediction);fn=pos-tp;tn=neg-fp
    fpos=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp),where=2*tp+fp+fn>0)
    fneg=np.divide(2*tn,2*tn+fp+fn,out=np.zeros_like(tn),where=2*tn+fp+fn>0)
    return dict(auroc=auc,ap=ap,recall=tp/pos,fpr=fp/neg,macro_f1=(fpos+fneg)/2)
def point(y,score,pred):
    result={k:float(v[0]) for k,v in weighted(y,score,pred,np.ones((1,len(y)))).items()}
    y=np.asarray(y);pred=np.asarray(pred,dtype=bool)
    result.update(n=len(y),n_anomaly=int(y.sum()),n_normal=int(len(y)-y.sum()),
        tp=int(np.sum((y==1)&pred)),fp=int(np.sum((y==0)&pred)),
        tn=int(np.sum((y==0)&~pred)),fn=int(np.sum((y==1)&~pred)))
    return result
def weights_for(q,y):
    groups=defaultdict(list)
    for i,x in enumerate(q):groups[x['category'],int(y[i])].append(i)
    rng=np.random.default_rng(42);w=np.zeros((2000,len(y)))
    for _,members in sorted(groups.items()):w[:,members]=rng.multinomial(len(members),np.full(len(members),1/len(members)),size=2000)
    return w
def paired(q,y,scores,predictions,pairs):
    w=weights_for(q,y);boot={a:weighted(y,s,predictions[a],w) for a,s in scores.items()}
    points={a:point(y,s,predictions[a]) for a,s in scores.items()}
    categories=sorted({x['category'] for x in q});cat_points={};cat_boot={}
    for c in categories:
        idx=np.array([i for i,x in enumerate(q) if x['category']==c])
        cat_points[c]={a:point(y[idx],s[idx],predictions[a][idx]) for a,s in scores.items()}
        cat_boot[c]={a:weighted(y[idx],s[idx],predictions[a][idx],w[:,idx]) for a,s in scores.items()}
    def interval(delta,array):return dict(delta=float(delta),ci95=np.quantile(array,[.025,.975],method='linear').tolist(),**BOOTSTRAP)
    result={}
    for a,b in pairs:
        result[a+'-minus-'+b]=dict(
            pooled={k:interval(points[a][k]-points[b][k],boot[a][k]-boot[b][k]) for k in METRICS},
            category_macro={k:interval(np.mean([cat_points[c][a][k]-cat_points[c][b][k] for c in categories]),
                np.mean([cat_boot[c][a][k]-cat_boot[c][b][k] for c in categories],axis=0)) for k in METRICS})
    return points,result

def qwen_audit(root):
    stage=root/'stage09_qwen_matched';contract=read(stage/'contract.json');freeze=read(stage/'prediction_freeze.json')
    require(freeze['accepted']==240 and freeze['scores_sha256']==sha(stage/'scores.jsonl') and
            freeze['contract_sha256']==digest(contract),'Qwen prediction freeze changed')
    require(read(stage/'status.json')['status']=='complete','Qwen run incomplete')
    q,y=identities(root);ids=[x['source_id'] for x in q];rr=rows(stage/'scores.jsonl')
    maps={a:{r['source_id']:r for r in rr if r['arm']==a} for a in ('base','decision_decision')}
    require(len(rr)==240 and all(set(v)==set(ids) and len(v)==120 for v in maps.values()),'Need unique Qwen240')
    png={x['source_id']:x['png_sha256'] for x in contract['queries']}
    for row in rr:
        require(row['contract_sha256']==digest(contract) and row['image_sha256']==png[row['source_id']],'Qwen row provenance changed')
        require(not(set(row)&{'label','is_anomalous','boxes','answer'}),'Ground truth in inference record')
        require(row['image_grid_thw']==[[1,28,28]] and row['merged_visual_tokens']==196 and row['native_input_calls']==3,'Qwen input audit failed')
        require(all(math.isfinite(row[k]) for k in ('log_likelihood_true','log_likelihood_false','anomaly_log_odds','anomaly_score')),'Nonfinite Qwen score')
        require(row['anomaly_log_odds']==row['log_likelihood_true']-row['log_likelihood_false'],'Qwen margin differs')
    scores={a:np.array([mp[s]['anomaly_log_odds'] for s in ids]) for a,mp in maps.items()}
    predictions={a:s>0 for a,s in scores.items()}
    points,deltas=paired(q,y,scores,predictions,[('decision_decision','base')])
    metrics=read(stage/'metrics.json')
    for a,p in points.items():
        saved=metrics['arms'][a];require(math.isclose(p['auroc'],saved['overall']['auroc'],abs_tol=1e-14),'AUROC recomputation differs')
        require(math.isclose(p['ap'],saved['overall']['average_precision'],abs_tol=1e-14),'AP recomputation differs')
        require(all(p[k]==saved['default_equal_candidate_threshold'][k] for k in ('tp','fp','tn','fn')),'Working-point recomputation differs')
    mismatches=sum(maps['base'][s]['input_audit']['pixel_values_sha256']!=maps['decision_decision'][s]['input_audit']['pixel_values_sha256'] for s in ids)
    require(mismatches==0,'Cross-arm native pixel tensors differ')
    loaders={}
    for a in maps:
        d=read(stage/(a+'_loading_audit.json'))['model_loading_audit']
        require(len(d['quantized_modules'])==252 and len(d['loaded_lora_modules'])==(144 if a=='decision_decision' else 0) and
                d['visual_dtype']=='torch.bfloat16' and d['attention_implementation']=='sdpa' and d['all_parameters_frozen'],'Qwen loading audit failed')
        loaders[a]=dict(quantized_modules=252,lora_modules=len(d['loaded_lora_modules']),visual_dtype=d['visual_dtype'],attention='sdpa')
    history={}
    for a in maps:
        p=root/'source_qwen'/('base_scores.jsonl' if a=='base' else 'decision_decision_scores.jsonl');old=rows(p);mp={x['source_id']:x for x in old}
        require(set(mp)==set(ids) and all(mp[x['source_id']]['image_sha256']==x['sha256'] for x in q),'Historical Qwen source mismatch')
        s=np.array([mp[x]['anomaly_log_odds'] for x in ids]);previous=point(y,s,s>0)
        history[a]=dict(scores_sha256=sha(p),original_input=previous,new448=points[a],
            new_minus_original={k:points[a][k]-previous[k] for k in METRICS},
            interpretation='Historical original-input control; render geometry budget changed; not a new matched-input run')
    result=dict(schema='qwen-matched-input-final-audit-v1',status='complete',script_sha256=sha(__file__),
        n_targets=240,n_unique_pairs=240,source_counts={a:120 for a in maps},cross_arm_pixel_tensor_hash_mismatches=0,
        all_rows_grid=[1,28,28],merged_visual_tokens=196,loaders=loaders,arms=points,paired_dd_minus_base=deltas['decision_decision-minus-base'],
        hashes={name:sha(stage/name) for name in ('contract.json','scores.jsonl','metrics.json','prediction_freeze.json','preflight.json')},
        history_comparison=history,bootstrap=BOOTSTRAP,model_forward_budget=dict(image_decisions=240,boolean_candidate_forwards=480,extra_warmups=0),
        no_new_training=True,no_new_api=True,labels_opened_after_prediction_freeze=True,confirmation_opened=False,limitations=contract['protocol']['limitations'])
    # Preserve the already-published audit's producing-code provenance when a
    # later report-only revision reruns the same unchanged numerical audit.
    audit_path=stage/'final_audit.json'
    if audit_path.exists():
        previous=read(audit_path)
        require({k:v for k,v in previous.items() if k!='script_sha256'}==
                {k:v for k,v in result.items() if k!='script_sha256'},'Existing Qwen audit differs')
        result=previous
    publish(audit_path,result)
    lines=['# Qwen共同448输入结果审计','','服务器完成base/DD各120次图像判断，全部240行先冻结后读取既有开发集布尔标签。','','| 路线 | AUROC | AP | Recall | FPR | TP/FP/TN/FN |','|---|---:|---:|---:|---:|---|']
    for a,v in points.items():lines.append('| '+a+' | '+' | '.join(f'{100*v[k]:.2f}%' for k in ('auroc','ap','recall','fpr'))+' | '+'/'.join(str(v[k]) for k in ('tp','fp','tn','fn'))+' |')
    lines+=['','固定工作点为严格anomaly_log_odds > 0；未拟合阈值。','', '## 配对DD减底座','']
    for k,v in result['paired_dd_minus_base']['pooled'].items():lines.append(f"- {k}: {100*v['delta']:+.2f}pp，95%分层配对区间[{100*v['ci95'][0]:+.2f},{100*v['ci95'][1]:+.2f}]pp。")
    lines+=['','区间2000次category×label分层、seed42；固定本次query/模型，不含训练seed与服务重复随机性，未做多重比较校正。','', '## 核验与限制','','- 全240行实际grid=[1,28,28]、196个merged视觉token；两路逐query native像素tensor散列错配0。','- 两路252语言NF4线性层，DD144注意力LoRA，视觉BF16/SDPA；逐行持久化后完整冻结。','- 480候选forward，无额外预热；每次load首图有冷启动开销，时延含输入审计，不作跨系统速度排名。']
    for a,v in history.items():lines.append(f"- {a}: 历史原图AUROC{100*v['original_input']['auroc']:.2f}% → 新448 {100*v['new448']['auroc']:.2f}%（{100*v['new_minus_original']['auroc']:+.2f}pp），属于输入渲染预算变化对照。")
    lines+=['- DD在pcb3/capsules等类别仍有弱项，pooled改善不能解释为所有类别提升。','- 与Jev只匹配PNG像素和图数；两者问题/结构化输出协议不同，Jev内部token与算力未知。','- 既有120开发query不是新盲测；confirmation未读取。','']
    publish(stage/'ANALYSIS.md','\n'.join(lines),text=True)
    return result

def completion(root):
    progress={};pending=[]
    for stage,n in [('stage08_support',600),('stage09_qwen_matched',240),('stage10_resolution',120)]:
        p=root/stage/'status.json';state=read(p) if p.exists() else {}
        good=(state.get('status')=='complete' and state.get('completed')==state.get('expected')==n and
            (root/stage/'metrics.json').exists() and (root/stage/'prediction_freeze.json').exists())
        progress[stage]=dict(status=state.get('status','missing'),completed=state.get('completed'),expected=n,
                            evaluation_exists=(root/stage/'metrics.json').exists(),ready=good)
        if not good:pending.append(stage)
    return progress,pending

def api_audit(root,stage,expected):
    path=root/stage;contract=read(path/'contract.json');freeze=read(path/'prediction_freeze.json');status=read(path/'status.json')
    rr=rows(path/'predictions.jsonl');attempts=rows(path/'attempts.jsonl')
    key=lambda row:(row['source_id'],row.get('arm','size320_0shot'))
    require(len(rr)==len({key(r) for r in rr})==expected,'Accepted API target count differs')
    require(status['status']=='complete' and status['completed']==status['expected']==expected,'API status incomplete')
    require(freeze['contract_sha256']==sha(path/'contract.json'),'API frozen contract changed')
    prediction_key='predictions320_sha256' if stage=='stage10_resolution' else 'predictions_sha256'
    require(freeze[prediction_key]==sha(path/'predictions.jsonl'),'API frozen predictions changed')
    contract_hash=digest(contract);require(all(r['contract_sha256']==contract_hash for r in rr),'API row contract changed')
    require(all(type(r['probability']) in (float,int) and math.isfinite(r['probability']) and 0<=r['probability']<=1 for r in rr),'Invalid accepted probability')
    scheduled={(r['source_id'],r.get('arm','size320_0shot')) for r in contract['schedule']}
    require({key(r) for r in rr}==scheduled,'API accepted/scheduled target mismatch')
    starts=[r for r in attempts if r['event']=='started'];terminal=[r for r in attempts if r['event'] in ('succeeded','failed')]
    eventkey=lambda r:(*key(r),r['attempt'],r.get('invocation_id'))
    unknown={eventkey(r) for r in starts}-{eventkey(r) for r in terminal}
    require(not unknown and sum(r['event']=='succeeded' for r in terminal)==expected,'API unsettled/duplicate success attempts')
    require(not status.get('confirmation_opened',True),'Unexpected confirmation access flag')
    metrics=read(path/'metrics.json');require(metrics['status']=='complete' and not metrics['confirmation_opened'],'API metric status invalid')
    report=dict(stage=stage,accepted=expected,started=len(starts),explicit_failed=sum(r['event']=='failed' for r in terminal),unknown=0,
        per_arm=dict(Counter(key(r)[1] for r in rr)),files_sha256={n:sha(path/n) for n in ('contract.json','predictions.jsonl','attempts.jsonl','prediction_freeze.json','metrics.json')},
        currency_cost=None,missing_usage_policy='Unknown, never zero imputation')
    return contract,rr,metrics,report

def summarize(root):
    progress,pending=completion(root)
    if pending:
        value=dict(schema=SCHEMA,status='pending',pending_stages=pending,progress=progress,
                   partial_formal_statistics_produced=False,confirmation_opened=False)
        mutable_status(root/'stages/11_summary_status.json',value)
        print(json.dumps(value,ensure_ascii=False),flush=True);return
    qa=qwen_audit(root);c8,r8,m8,a8=api_audit(root,'stage08_support',600);c10,r10,m10,a10=api_audit(root,'stage10_resolution',120)
    require(c10['reference_contract_sha256']==sha(root/'stage08_support/contract.json') if 'reference_contract_sha256' in c10 else True,'Resolution reference mismatch')
    f10=read(root/'stage10_resolution/prediction_freeze.json')
    require(f10['reference_contract_sha256']==a8['files_sha256']['contract.json'] and
            f10['reference_predictions_sha256']==a8['files_sha256']['predictions.jsonl'],'Resolution shared448 evidence differs')
    q,y=identities(root);ids=[x['source_id'] for x in q]
    mq=rows(root/'stage09_qwen_matched/scores.jsonl');mp={a:{r['source_id']:r for r in mq if r['arm']==a} for a in ('base','decision_decision')}
    shared={r['source_id']:r for r in r8 if r['arm']=='shared0'};require(set(shared)==set(ids),'Shared448 source mismatch')
    qcontract=read(root/'stage09_qwen_matched/contract.json')
    require(qcontract['source_stage_contract']['sha256']==a8['files_sha256']['contract.json'],'Qwen/JeV stage08 contract differs')
    scores={a:np.array([mp[a][s]['anomaly_log_odds'] for s in ids]) for a in mp};scores['jev_shared0']=np.array([shared[s]['probability'] for s in ids])
    predictions={a:s>0 for a,s in scores.items() if a!='jev_shared0'};predictions['jev_shared0']=scores['jev_shared0']>=.5
    ranking,cross=paired(q,y,scores,predictions,[('decision_decision','base'),('decision_decision','jev_shared0'),('base','jev_shared0')])
    seed_summary={}
    for shots in (1,4):
        selected={str(seed):m8['arms'][f's{seed}_{shots}']['metrics'] for seed in (43,44)}
        seed_summary[str(shots)]=dict(seed_metrics=selected,descriptive_mean={k:float(np.mean([v[k] for v in selected.values()])) for k in selected['43']},
            descriptive_range={k:[min(v[k] for v in selected.values()),max(v[k] for v in selected.values())] for k in selected['43']},
            interpretation='Two fixed support-set point estimates; not two independent query datasets or a support-population confidence interval')
    stagefiles={str(n):name for n,name in [(1,'01_source_audit.json'),(2,'02_teacher_alignment.json'),(3,'03_candidates.json'),(4,'04_prediction_freeze.json'),(5,'05_evaluation.json'),(6,'06_routing_results.json'),(7,'07_calibration_results.json')]}
    old={k:read(root/'stages'/v) for k,v in stagefiles.items()}
    require(all(not d.get('confirmation_opened',False) for d in old.values()),'Unexpected confirmation flag in prior stage')
    roi=old['5'];route=old['6'];cal=old['7'];allroi=roi['groups']['all120'];allroute=route['groups']['all120']
    roi_summary=dict(methods={a:allroi['methods'][a]['pooled'] for a in ('unconditional_teacher','roi','duplicate')},
        roi_minus_duplicate=allroi['roi_minus_duplicate'],teacher_exposure_counts=roi['teacher_exposure_counts'],
        teacher_unseen=roi['groups']['teacher_unseen']['methods']['roi']['pooled'],candidate_diagnostics=allroi['candidate_and_repeat_diagnostics'],
        operations=roi['operations'],source_code='code/evaluate_roi.py',evidence='stages/05_evaluation.json')
    route_summary=dict(methods={a:dict(performance=v['performance']['pooled'],budget=v['budget']) for a,v in allroute['methods'].items()},
        uncertain_minus_duplicate=allroute['uncertain_minus_duplicate'],offline_replay_only=True,
        source_code='code/routing_replay.py',evidence='stages/06_routing_results.json',no_actual_call_savings_claim=True)
    cal_summary=dict(models={a:{v:spec[v]['pooled'] for v in ('raw','oof_logistic')} for a,spec in cal['models'].items()},
        rejection={a:{version:{coverage:{k:result[k] for k in ('n_accepted','n_abstained','realized_coverage',
            'selected_accuracy','selective_risk','selected_anomaly','selected_normal','full_anomaly_automatic_detection_rate',
            'anomaly_acceptance_coverage','normal_acceptance_coverage')} for coverage,result in spec[version]['rejection'].items()}
            for version in ('raw','oof_logistic')} for a,spec in cal['models'].items()},
        n_sources=cal['n_sources'],n_folds=cal['n_folds'],fit_count=cal['calibration_fit_count'],
        protocol_freeze_timing=cal['protocol_freeze_timing'],candidate_evaluation_available_at_prepare=cal['candidate_evaluation_available_at_prepare'],
        independent_calibration_task_completed=False,production_calibrator=False,
        source_code='code/calibration_diagnostics.py',evidence='stages/07_calibration_results.json')
    source_hashes={('stages/'+v):sha(root/'stages'/v) for v in stagefiles.values()}
    source_hashes.update({('code/'+n):sha(root/'code'/n) for n in ['summarize_followup.py','qwen_matched_input.py','support_robustness.py','resolution_ablation.py']})
    next_steps=[
        dict(order=1,task='Improve candidate generator before another ROI gain claim',reason='Teacher mean IoU is low and half candidates cover whole image',gate='Freeze a train-only candidate rule; report all-anomaly IoU, IoU>=.5, fullbox fraction and teacher exposure without tuning on confirmation',code='code/extract_teacher.py; code/evaluate_roi.py'),
        dict(order=2,task='Separate calibration data and final confirmation',reason='OOF reused viewed development120; independent calibration and confirmation remain incomplete',gate='Fix system/prompt/threshold using training and independent calibration; open confirmation only at final approved evaluation',code='code/calibration_diagnostics.py; source_baseline/frozen/plan.json'),
        dict(order=3,task='Broader supports and provider repeat controls',reason='Seeds43/44 are only two fixed banks;0shot is shared once',gate='Predeclare more support banks and interleaved repeated requests; paired source evaluation plus support/provider variance',code='code/support_robustness.py'),
        dict(order=4,task='Evaluate online routing after predictive benefit is supported',reason='Current uncertain replay increases logical calls and normal positive boxes',gate='Predeclare escalation budget and FP/FN utility; compare online wall time and real billed usage on independent data',code='code/routing_replay.py'),
    ]
    value=dict(schema=SCHEMA,status='complete',hostname=HOSTNAME,script_sha256=sha(__file__),source_files_sha256=source_hashes,
        population=dict(n_query=120,normal=60,anomaly=60,categories=12,role='previously_viewed_development_not_blind',confirmation_opened=False),
        source_and_teacher=dict(source_audit=old['1'].get('status'),teacher_query_split=old['2']['counts_by_query_split'],teacher_support_split=old['2']['counts_by_support_split'],full_image_candidates=old['3']['full_image_boxes'],
            warning='Teacher custom supervised split exposure differs from native Qwen source-disjoint training; nontrain subset is not a new blind test'),
        roi=roi_summary,routing=route_summary,development_oof=cal_summary,
        supports=dict(arms=m8['arms'],seed_point_summary=seed_summary,paired_bootstrap=m8['paired_bootstrap'],category_macro=m8['category_macro'],normal_bank_verified=m8['normal_bank_verified'],cross_seed_support_overlap=m8['cross_seed_support_overlap'],limitations=m8['limitations'],source_code='code/support_robustness.py',evidence='stage08_support/metrics.json'),
        matched448=dict(arms=ranking,paired_bootstrap=cross,qwen_audit=qa,thresholds={'qwen':'anomaly_log_odds > 0','jev':'probability >= 0.5'},
            score_interpretation='Cross-system AUROC/AP ranking comparison only; no equal calibration or compute claim; native prompts and output protocols differ',
            source_code='code/qwen_matched_input.py',evidence=['stage09_qwen_matched/metrics.json','stage08_support/metrics.json']),
        resolution=dict(arms=m10['arms'],paired_bootstrap=m10['paired_bootstrap'],canvas_pixel_ratio=m10['canvas_pixel_ratio_320_over448'],limitations=m10['limitations'],source_code='code/resolution_ablation.py',evidence='stage10_resolution/metrics.json'),
        execution_audits=dict(roi=roi['operations'],support=a8,qwen=qa['model_forward_budget'],resolution=a10,stage09_files=qa['hashes']),
        bootstrap=BOOTSTRAP,next_scientific_gates=next_steps,no_new_fitting=True,no_new_model_or_api_calls=True,no_gt_masks_opened_by_summary=True,confirmation_opened=False)
    publish(root/'stages/11_summary.json',value)
    report=make_report(value);publish(root/'FINAL_FOLLOWUP_REPORT.md',report,text=True)
    mutable_status(root/'stages/11_summary_status.json',dict(schema=SCHEMA,status='complete',summary_sha256=sha(root/'stages/11_summary.json'),report_sha256=sha(root/'FINAL_FOLLOWUP_REPORT.md'),confirmation_opened=False))
    print(json.dumps(dict(status='complete',report=str(root/'FINAL_FOLLOWUP_REPORT.md'),summary=str(root/'stages/11_summary.json'),accepted_new_api=240+600+120,qwen_image_decisions=240),ensure_ascii=False),flush=True)

def pct(x):return f'{100*x:.2f}%'
def delta(v):return f"{100*v['delta']:+.2f}pp [{100*v['ci95'][0]:+.2f},{100*v['ci95'][1]:+.2f}]"
def make_report(v):
    r=v['roi'];t=v['routing'];s=v['supports'];matched=v['matched448'];resolution=v['resolution'];qa=matched['qwen_audit']
    lines=['# Jev与Qwen工业异常检测后续研究总结','','## 结论与数据边界','',
        '全部研究固定既有120张VisA开发query（12类各5正常/5异常），未读取confirmation。后续实验与统计均在指定AutoDL服务器完成。',
        'ROI复核尚未证明定位净收益；共同448输入下DD相对量化底座有明确开发集排序改善；支持与分辨率结论限定于本次固定query和服务调用。','',
        '## 共同448图像的全图判断','','| 系统 | AUROC | AP | Recall | FPR | 工作点 |','|---|---:|---:|---:|---:|---|']
    for a,m in matched['arms'].items():lines.append('| '+a+' | '+' | '.join(pct(m[k]) for k in ('auroc','ap','recall','fpr'))+' | '+('p≥0.5' if a=='jev_shared0' else 'logodds>0')+' |')
    dd=matched['paired_bootstrap']['decision_decision-minus-base']['pooled']['auroc']
    lines+=['',f'- DD−base AUROC：{delta(dd)}；这对应新共同448输入，不能与历史原图82.39%→94.56%的差值混用。',
        '- 代码`code/qwen_matched_input.py`；证据`stage09_qwen_matched/scores.jsonl`、`metrics.json`、`final_audit.json`及`stage08_support/metrics.json`。',
        '- Qwen两路都使用语言NF4 double quant/BF16 compute、视觉BF16、SDPA；DD为r16/alpha32/qkvo LoRA。240图像判断=480候选forward。',
        '- 两路逐query native像素tensor相同，实际grid=[1,28,28]与196个merged视觉token；Jev共享同PNG字节。',
        '- Qwen JSON布尔候选评分与Jev Noul问题协议不同；Jev内部token/算力不可见，只匹配PNG/图数。跨系统比较AUROC/AP排名，不宣称相同校准或相同算力。',
        '- DD−base逐类别宏平均AUROC：'+delta(matched['paired_bootstrap']['decision_decision-minus-base']['category_macro']['auroc'])+'。宏平均区间跨零，pooled改善不能解释为每类都改善，pcb3和capsules仍有弱项。','',
        '## 正常参考图的种子敏感性','','| 新采集条件 | AUROC | AP | Recall | FPR |','|---|---:|---:|---:|---:|']
    for a,d in s['arms'].items():m=d['metrics'];lines.append('| '+a+' | '+' | '.join(pct(m[k]) for k in ('auroc','ap','recall','fpr'))+' |')
    for shot,d in s['seed_point_summary'].items():
        ran=d['descriptive_range']['auroc'];lines.append(f"- {shot}-shot seeds43/44 AUROC均值{pct(d['descriptive_mean']['auroc'])}，范围[{pct(ran[0])},{pct(ran[1])}]；这是两组点估计描述，不是独立query或总体支持分布区间。")
    for pair,d in s['paired_bootstrap'].items():
        if pair in ('s43_4-minus-shared0','s44_4-minus-shared0','s44_4-minus-s43_4'):lines.append('- '+pair+' AUROC：'+delta(d['pooled']['auroc'])+'。')
    lines+=['- 384张训练正常候选每类32张，query来源/字节无交集；1-shot嵌套4-shot首张，固定种子哈希选择，不按结果重选。',
        '- 新shared0单独采集120，两seed1/4各120，总600判断；旧seed42因renderer差异仅为历史对照。',
        '- 这里zero/few-shot指推理时不给/给正常参考图，不更新Jev参数；Qwen DD则已做768训练来源的监督LoRA，不能将它与参考图数量当等训练预算比较。',
        '- 代码`code/support_robustness.py`；证据`stage08_support/{contract,predictions,prediction_freeze,metrics}.json`，实际JSONL后缀见目录。下一门槛是更多bank和服务重复，而非把两seed当大样本。','',
        '## 第一阶段保留的历史零少样本基线','','| 旧seed42条件 | AUROC | Recall(p≥0.5) | FPR(p≥0.5) |','|---|---:|---:|---:|']
    for arm in ('jev_0shot','jev_1shot','jev_4shot'):
        m=v['development_oof']['models'][arm]['raw'];rec=m['TP']/(m['TP']+m['FN']);fpr=m['FP']/(m['FP']+m['TN'])
        lines.append('| '+arm+' | '+pct(m['auroc'])+' | '+pct(rec)+' | '+pct(fpr)+' |')
    lines+=['','历史360条完整响应保留不覆盖；本轮后续API/GPU均改由服务器执行。旧renderer与新renderer不同，旧seed42与新43/44的差异不能纯归因为支持种子。','',
        '## 320与448零参考分辨率','','| 条件 | AUROC | AP | Recall | FPR |','|---|---:|---:|---:|---:|']
    for a,d in resolution['arms'].items():m=d['metrics'];lines.append('| '+a+' | '+' | '.join(pct(m[k]) for k in ('auroc','ap','recall','fpr'))+' |')
    lines+=['- 320−448 AUROC：'+delta(resolution['paired_bootstrap']['pooled']['auroc'])+'。',
        f"- 320/448画布像素比{resolution['canvas_pixel_ratio']:.4f}；这同时改变可见细节与输入预算，不等同纯算力因果或精确token比例。",
        '- 新320接受120判断，仅复用stage08 shared0的120个448响应；其余480支持响应不混入。服务采集时间未交错，无法独立排除provider变化。',
        '- 代码`code/resolution_ablation.py`；证据`stage10_resolution/metrics.json`。下一门槛是独立数据及交错重复的质量/成本测量。','',
        '## 固定候选区域复核与瓶颈','','| 方法 | 全60异常平均IoU | IoU≥0.5 | 异常保留召回 | 全60正常出框率 |','|---|---:|---:|---:|---:|']
    for a,m in r['methods'].items():lines.append('| '+a+f" | {m['anomaly_bbox_iou_mean_all']:.6f} | {pct(m['anomaly_iou_at_05_all'])} | {pct(m['anomaly_retained_recall'])} | {pct(m['normal_positive_box_rate'])} |")
    for k,d in r['roi_minus_duplicate']['pooled'].items():lines.append('- ROI−duplicate '+k+'：'+delta(d)+'。')
    source=v['source_and_teacher'];lines+=['',f"Teacher query split={source['teacher_query_split']}；96/120在teacher训练划分中，未见24张仅22异常/2正常；未见子集不是新盲测，FPR分母极小。",
        f"候选中{source['full_image_candidates']}/120为整图框。当前teacher无条件平均IoU低，门控只能保留或删除现有框，无法修正候选几何，候选质量是主要科学瓶颈。",
        'ROI−duplicate的定位/保留召回区间跨或触及0，误报也有代价，保留负结果。历史全图0/1/4与原生Qwen只作输入/语义预算不同的历史门控。',
        '代码`code/extract_teacher.py`、`code/evaluate_roi.py`；证据`stages/02_teacher_alignment.json`、`03_candidates.json`、`05_evaluation.json`。下一步先冻结train-only候选改进，再用新确认方案验证。','',
        '## 离线顺序路由与成本','','| 策略 | 逻辑成功请求 | ROI升级 | Recall | 正常出框率 |','|---|---:|---:|---:|---:|']
    for a,d in t['methods'].items():m=d['performance'];b=d['budget'];lines.append('| '+a+f" | {b['logical_successful_requests']} | {b['extra_roi_calls']} | {pct(m['anomaly_retained_recall'])} | {pct(m['normal_positive_box_rate'])} |")
    lines+=['- 不确定性路由仅根据duplicate概率、几何及固定中性哈希，策略冻结后回放，无oracle选择；并非在线Agent部署。',
        '- 实际已采集ROI/duplicate共240成功目标及重试；逻辑162请求不抵消已发生采集成本，未估造货币账单。',
        '- 不确定性路由−duplicate Recall：'+delta(t['uncertain_minus_duplicate']['pooled']['anomaly_retained_recall'])+'；正常出框率：'+delta(t['uncertain_minus_duplicate']['pooled']['normal_positive_box_rate'])+'。',
        '- 代码`code/routing_replay.py`；证据`stages/06_routing_results.json`。下一门槛是独立数据上预声明误报/漏检效用及在线时延/账单测量。','',
        '## 开发集OOF校准与拒答诊断','','| 历史全图分数 | raw Brier | OOF Brier | raw ECE10 | OOF ECE10 |','|---|---:|---:|---:|---:|']
    for a,d in v['development_oof']['models'].items():x,z=d['raw'],d['oof_logistic'];lines.append('| '+a+f" | {x['brier']:.6f} | {z['brier']:.6f} | {x['ece_10_equal_width']:.6f} | {z['ece_10_equal_width']:.6f} |")
    lines+=['','拒答代表例使用历史全图分数；这些诊断采用其既有p≥0.5协议，不替换stage09原生严格logodds>0工作点。','',
        '| 历史分数 | 版本 | 覆盖率 | 接收异常/正常 | 接收准确率 | Selective risk | 全异常自动检出率 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for a in ('jev_4shot','qwen_decision_decision'):
        for version,coverages in v['development_oof']['rejection'][a].items():
            for coverage,d in sorted(coverages.items(),key=lambda item:float(item[0])):
                lines.append('| '+a+' | '+version+' | '+pct(d['realized_coverage'])+f" | {d['selected_anomaly']}/{d['selected_normal']} | "+
                    ' | '.join(pct(d[k]) for k in ('selected_accuracy','selective_risk','full_anomaly_automatic_detection_rate'))+' |')
    lines+=['- 固定5折category×label分层，每折96拟合/24验证；25个既有Logistic校准器，不在汇总中重拟合或调超参。',
        '- 这是已查看120开发图上的交叉拟合诊断，协议冻结时ROI结果已存在；独立校准/confirmation仍未完成，也未拟合生产校准器。',
        '- ECE对分箱和小样本敏感；拒答准确率应同时报告覆盖率、接收异常/正常分母及全异常自动检出率。区域Noul未错误套用整图校准标签。',
        '- 代码`code/calibration_diagnostics.py`；证据`stages/07_calibration_results.json`、`CALIBRATION_REPORT.md`。下一门槛是独立校准数据与固定最终方案后的确认。','',
        '## 完整执行与审计','','| 实验 | 接受目标 | Started | 显式失败 | 未决 |','|---|---:|---:|---:|---:|']
    audits=v['execution_audits']
    for name in ('roi','support','resolution'):
        a=audits[name];lines.append('| '+name+f" | {a['accepted']} | {a['started']} | {a.get('explicit_failed',a.get('failed'))} | {a.get('unknown',0)} |")
    lines+=['','Qwen单独240个图像判断/480个候选forward。新增API成功目标合计960（ROI240+支持600+320分辨率120），448共同对照被复用一次，未重复算作新增调用。',
        '费用只依据实际返回usage；失败或缺失usage保持未知，不补0。逐run文件SHA、计数、配对区间与下一门槛保存于`stages/11_summary.json`。',
        '阶段1 source审计、阶段2 teacher对齐、阶段3固定候选、阶段4预测冻结、阶段5区域评价、阶段6路由回放、阶段7开发OOF、阶段8支持种子、阶段9共同输入Qwen、阶段10分辨率均完整留存。','',
        '## 下一科学门槛','']
    for x in v['next_scientific_gates']:lines.append(f"{x['order']}. {x['task']}。原因：{x['reason']}。门槛：{x['gate']}。对应：`{x['code']}`。")
    lines+=['','所有区间固定当前query与条件，未纳入服务重复/训练seed/支持总体不确定性，未校正多重比较。此文档不把开发结果表述为跨域、盲测、生产概率保证或部署收益。','']
    return '\n'.join(lines)

def self_test():
    host_gate();y=np.array([0,0,1,1]);s=np.array([0.,.2,.8,1.]);p=s>=.5
    m=point(y,s,p);require(m['auroc']==m['ap']==m['recall']==1 and m['fpr']==0,'Perfect metric failed')
    tied=np.array([0.,.5,.5,1.]);w=np.array([[1,1,1,1],[2,0,1,1]])
    got=weighted(y,tied,tied>.5,w)
    for j,count in enumerate(w):
        idx=np.repeat(np.arange(4),count);reference=point(y[idx],tied[idx],tied[idx]>.5)
        require(all(math.isclose(float(got[k][j]),reference[k],abs_tol=1e-13) for k in METRICS),'Replication weighting failed')
    require(point(y,tied,tied>.5)['recall']!=point(y,tied,tied>=.5)['recall'],'Strict Qwen threshold test failed')
    print(json.dumps(dict(status='passed',model_calls=0,api_calls=0,calibrator_fits=0)),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('audit-qwen','summary'):
        x=sub.add_parser(name);x.add_argument('--root',type=Path,required=True)
    sub.add_parser('self-test');args=p.parse_args();host_gate()
    if args.command=='self-test':return self_test()
    root=args.root.resolve()
    with locked(root):
        if args.command=='audit-qwen':
            value=qwen_audit(root);print(json.dumps({k:value[k] for k in ('status','n_targets','paired_dd_minus_base','cross_arm_pixel_tensor_hash_mismatches')},ensure_ascii=False),flush=True)
        else:summarize(root)

if __name__=='__main__':main()
