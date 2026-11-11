#!/usr/bin/env python3
"""Server-only evaluation of original stage10 against certified stage08 rescue.

No inference, paid API, model training, dependency mutation, or original runner
modification occurs. The original run CLI remains resolution_ablation.py run.
This evaluator explicitly freezes its added evaluation contract before labels.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import numpy as np
import roi_api as api
import support_robustness as support
import resolution_ablation as resolution
import support_rescue as rescue

HOSTNAME='autodl-container-132444827f-9c594038'
SCHEMA='jev-resolution-certified-reference-evaluation-v1'
RESOLUTION_SHA='4458848524108d22220d1196cffef65967d06f730425e1cb73277e651b1db9af'
SUPPORT_SHA='fc6ddc02eb2e2a8c0556f3843a5e73321f9d9bfcf1b5f14314de1aea2077d0bd'
TRANSPORT_SHA='4f7b5ed1f703b3751f9fd4bb8c86131d684cec0bed3a9011f5108621e7aa9e01'

def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return api.file_sha(Path(path))
def digest(value):return api.digest(api.canonical(value))
def host_gate():
    require(socket.gethostname()==HOSTNAME,'Execute evaluator only on designated AutoDL server')
    require(sha(resolution.__file__)==RESOLUTION_SHA and sha(support.__file__)==SUPPORT_SHA and
        sha(api.__file__)==TRANSPORT_SHA,'Original frozen runner source differs')
def publish(path,value,text=False):
    path=Path(path);raw=value.encode('utf-8') if text else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8')+b'\n'
    if path.exists():require(path.read_bytes()==raw,'Frozen evaluator artifact differs: '+path.name);return
    temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with temp.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    try:os.link(temp,path)
    finally:temp.unlink(missing_ok=True)

def align(queries,reference_contract,complete):
    ids=[q['source_id'] for q in queries]
    require(len(ids)==len(set(ids))==120 and ids==[q['source_id'] for q in reference_contract['queries']],
        '320/448 query alignment differs')
    require(len(complete)==600,'Certified reference must include600 accepted targets')
    selected=[complete[sid,'shared0'] for sid in ids]
    require(len(selected)==120 and all(r['arm']=='shared0' and r['source_id']==sid for sid,r in zip(ids,selected)),
        'Only shared0 may be reused for the448 reference')
    return selected

def evaluate(args):
    host_gate();root=args.root.resolve();stage=api.inside(root,args.stage)
    with api.output_lock(stage),api.output_lock(root/'stage08_rescue'):
        contract,images=resolution.verify(root,stage);status=read(stage/'status.json')
        require(contract['mode']=='formal_independent_resolution' and status['status']=='complete' and
            status['completed']==status['expected']==120,'Need complete original stage10 formal320120')
        completed,starts,attempts,hashes,unknown=resolution.validate_logs(stage,contract,images)
        require(len(completed)==120 and not unknown,'320 request audit is incomplete')
        rc,ri,reference,rstarts,rattempts,rhashes,runknown=rescue.load_complete(root)
        require(contract['reference_stage']=='stage08_support' and not runknown,'Unexpected reference stage')
        cert_path=root/'stage08_rescue/completion_certificate.json';certificate=read(cert_path)
        old_status=read(root/'stage08_support/status.json')
        require(certificate['original_stage_status']==old_status['status']!='complete' and
            certificate['original_status_unchanged'] is True,'Original support status must remain explicit')
        require(rc['mode']=='formal_independent_support_robustness' and
            contract['reference_contract_sha256']==sha(root/'stage08_support/contract.json'),'Reference contract changed')
        queries=contract['queries'];rows448=align(queries,rc,reference)
        for q,rq in zip(queries,rc['queries']):
            require(q['reference448']['png_sha256']==rq['asset']['png_sha256'],'448 query PNG identity differs')
        rows320=[completed[q['source_id']] for q in queries]
        provenance=dict(reference_completion_status=certificate['status'],original_stage_status=old_status['status'],
            original_status_unchanged=True,scientific_change=False,
            certificate_path='stage08_rescue/completion_certificate.json',certificate_sha256=sha(cert_path),
            combined_predictions_path='stage08_rescue/combined_predictions.jsonl',
            combined_predictions_sha256=sha(root/'stage08_rescue/combined_predictions.jsonl'),
            operational_amendment_sha256=sha(root/'stage08_rescue/operational_amendment.json'),
            original_contract_sha256=sha(root/'stage08_support/contract.json'),
            original_predictions_sha256=sha(root/'stage08_support/predictions.jsonl'),
            reference_source_sha256=sha(rescue.__file__),evaluator_source_sha256=sha(__file__),
            reused_arm='shared0',reused_accepted=120,excluded_support_arm_observations=480)
        evaluation_contract=dict(schema=SCHEMA,status='evaluation_protocol_frozen',hostname=HOSTNAME,
            original_resolution_contract_sha256=sha(stage/'contract.json'),
            original_resolution_source_sha256=sha(resolution.__file__),evaluator_source_sha256=sha(__file__),
            reference=provenance,query_order_sha256=digest([q['source_id'] for q in queries]),
            thresholds=dict(size320_0shot='probability >= 0.5',shared0_448='probability >= 0.5'),
            metrics=list(resolution.METRICS),bootstrap=dict(replicates=2000,seed=42,strata='category x image_label',paired=True),
            statistical_implementation='Original resolution.paired_bootstrap / support.metrics; no fit or threshold selection',
            budgets='120new320;120shared0448reused;480support observations excluded',
            operational_change_only=True,scientific_contract_unchanged=True,
            labels_not_opened_by_wrapper_before_freeze=True,previously_viewed_development_not_blind=True,confirmation_opened=False)
        evaluation_path=stage/'evaluation_contract_amended_reference.json';publish(evaluation_path,evaluation_contract)
        frozen=dict(schema=resolution.SCHEMA,contract_sha256=sha(stage/'contract.json'),
            predictions320_sha256=sha(stage/'predictions.jsonl'),
            reference_contract_sha256=sha(root/'stage08_support/contract.json'),
            reference_predictions_sha256=provenance['combined_predictions_sha256'],
            support_source_sha256=sha(support.__file__),resolution_source_sha256=sha(resolution.__file__),
            evaluator_source_sha256=sha(__file__),evaluation_contract_sha256=sha(evaluation_path),
            reference_completion_certificate_sha256=sha(cert_path),reference_operational_amendment=provenance,
            labels_not_opened_by_wrapper_before_prediction_freeze=True,confirmation_opened=False)
        publish(stage/'prediction_freeze.json',frozen)
        # Only already-approved120 boolean labels are read, after both freezes.
        labels=read(root/'source_baseline/frozen/labels.json');config=read(root/'source_baseline/frozen/config.json')
        require(digest(labels)==config['labels_sha256'],'Frozen development label identity differs')
        y=np.array([labels[q['source_id']] for q in queries],dtype=int)
        require(len(y)==120 and sum(y)==60 and set(y)=={0,1},'Need approved60normal/60anomaly')
        p320=np.array([r['probability'] for r in rows320]);p448=np.array([r['probability'] for r in rows448])
        arms={resolution.ARM:dict(metrics=support.metrics(y,p320),budget=resolution.budget(rows320,320)),
            resolution.REFERENCE:dict(metrics=support.metrics(y,p448),budget=resolution.budget(rows448,448))}
        by_category={}
        for category in sorted({q['category'] for q in queries}):
            idx=np.array([i for i,q in enumerate(queries) if q['category']==category])
            by_category[category]={resolution.ARM:support.metrics(y[idx],p320[idx]),resolution.REFERENCE:support.metrics(y[idx],p448[idx])}
        macro={arm:{k:float(np.mean([by_category[c][arm][k] for c in by_category])) for k in resolution.METRICS} for arm in arms}
        result=dict(schema=resolution.SCHEMA,status='complete',timestamp_utc=api.now(),hostname=HOSTNAME,prediction_freeze=frozen,
            arms=arms,by_category=by_category,category_macro=macro,paired_bootstrap=resolution.paired_bootstrap(queries,y,p320,p448),
            operations320=dict(accepted=120,started=sum(starts.values()),failed=sum(r['event']=='failed' for r in attempts),unknown=0),
            reused448_accepted=120,excluded_support_arm_observations=480,canvas_pixel_ratio_320_over448=320**2/448**2,
            semantic_note='Whole-image anomaly Noul with zero references; same wording as stage08 shared0',
            operational_reference_provenance=provenance,evaluation_contract_sha256=sha(evaluation_path),
            limitations=['Viewed development queries, not confirmation or cross-domain evidence',
                '448observations predate320; provider/time variation not independently excluded; HTTP502rescue collected later',
                'Downsampling changes visible detail and image budget simultaneously',
                'Bootstrap fixed queries/renderings, excludes provider repeat randomness; no multiple-comparison correction',
                'No exact image-token formula or verified currency bill; missing usage remains unknown'],
            no_gt_tuning=True,confirmation_opened=False)
        output=stage/'metrics.json'
        if output.exists():
            previous=read(output);require({k:v for k,v in previous.items() if k!='timestamp_utc'}==
                {k:v for k,v in result.items() if k!='timestamp_utc'},'Frozen amended resolution metrics differ');result=previous
        lines=['# Jev 320/448零参考分辨率研究','','原stage10科学合同、采集源码与预测保持不变；参考来自stage08_rescue完整运维修订证书，旧stage08/status仍保留原状态。',
            '新320接受120判断，只复用shared0的120个448响应，其余480个支持上下文不进入本比较。','',
            '| 输入 | AUROC | AP | Recall | FPR | Brier | ECE10 |','|---|---:|---:|---:|---:|---:|---:|']
        for arm in (resolution.ARM,resolution.REFERENCE):lines.append('| '+arm+' | '+' | '.join(f"{arms[arm]['metrics'][k]:.6f}" for k in ('auroc','ap','recall','fpr','brier','ece_10_equal_width'))+' |')
        d=result['paired_bootstrap']['pooled']['auroc']
        lines+=['',f"- 320−448 AUROC差{100*d['delta']:+.2f}pp；95%分层配对区间[{100*d['ci95'][0]:+.2f},{100*d['ci95'][1]:+.2f}]pp。",
            f'- 320/448画布像素比{320**2/448**2:.4f}，像素数不是精确图像token数。',
            '- 原run的120条响应先冻结，新增评价合同明确绑定救援combined/证书/源码SHA；之后只读取approved120布尔标签。',
            '- 接口HTTP502额外重试属于显式运维修订，旧科学合同、问题、图片与已接受响应均未改写；原stage08状态未伪装为complete。',
            '- 320/448采集时间未交错、provider重复方差未测量，不能单独归因为分辨率；阈值固定0.5。',
            '- 未读取confirmation或GT mask；逐usage缺失保持未知，不估货币账单。','']
        report='\n'.join(lines)
        publish(output,result);publish(stage/'REPORT.md',report,text=True)
        require(sha(stage/'predictions.jsonl')==frozen['predictions320_sha256'] and
            sha(root/'stage08_rescue/combined_predictions.jsonl')==frozen['reference_predictions_sha256'],
            'Frozen predictions changed during amended resolution evaluation')
        print(json.dumps(dict(status='complete',new320=120,reused448=120,metrics=str(output),
            evaluation_contract=str(evaluation_path),reference_completion_status=certificate['status'],
            no_new_api=True,confirmation_opened=False)),flush=True)
    return 0

def self_test(args):
    host_gate();queries=[dict(source_id=str(i)) for i in range(120)]
    complete={(str(i),arm):dict(source_id=str(i),arm=arm,probability=.9 if arm!='shared0' else .2)
        for i in range(120) for arm in ('shared0','s43_1','s43_4','s44_1','s44_4')}
    selected=align(queries,dict(queries=queries),complete)
    require(len(selected)==120 and all(r['probability']==.2 for r in selected),'Support arms leaked into448reference')
    y=np.array([0,0,1,1]);p=np.array([0.,.2,.8,1.]);require(support.metrics(y,p)['auroc']==1,'Synthetic metrics failed')
    weights=np.ones((2,4));w=support.weighted_metrics(y,p,weights)
    require(all(np.allclose(w[k],support.metrics(y,p)[k],atol=1e-12) for k in resolution.METRICS),'Original weighted implementation differs')
    require(digest({'x':1})==hashlib.sha256(api.canonical({'x':1})).hexdigest(),'Canonical binding differs')
    print(json.dumps(dict(status='passed',synthetic_only=True,model_calls=0,api_calls=0,label_files_opened=0,
        checks=['shared0only120','exclude_support480','original_metrics','canonical_sha','original_source_hashes'])),flush=True)
    return 0

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    evaluate_parser=sub.add_parser('evaluate');evaluate_parser.add_argument('--root',type=Path,required=True);evaluate_parser.add_argument('--stage',default='stage10_resolution')
    sub.add_parser('self-test');args=parser.parse_args()
    return self_test(args) if args.command=='self-test' else evaluate(args)
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        print(json.dumps(dict(status='blocked',exception_class=type(error).__name__,
            reason=str(error) if isinstance(error,ValueError) else 'Inspect frozen evaluation artifacts; arbitrary exception suppressed')),file=sys.stderr)
        raise SystemExit(2)
