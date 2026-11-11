#!/usr/bin/env python3
"""Explicit operational amendment for stage08 targets exhausted by HTTP502.

No original contract, code, ledger, result, support or question is changed.
The same original request body may receive at most two additional starts,
one per explicit invocation, sequentially and only after audited HTTP502.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import uuid

import numpy as np
import roi_api as api
import support_robustness as support

HOSTNAME='autodl-container-132444827f-9c594038'
SCHEMA='jev-support-http502-operational-amendment-v1'
OLD='stage08_support'
RESCUE='stage08_rescue'
OLD_CODE_SHA='fc6ddc02eb2e2a8c0556f3843a5e73321f9d9bfcf1b5f14314de1aea2077d0bd'
TRANSPORT_SHA='4f7b5ed1f703b3751f9fd4bb8c86131d684cec0bed3a9011f5108621e7aa9e01'


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return api.file_sha(Path(path))
def canonical(value):return api.canonical(value)
def digest(value):return api.digest(value)
def pair(row):return row['source_id'],row['arm']


def publish(path,value,text=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=value.encode('utf-8') if text else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8')+b'\n'
    if path.exists():
        require(path.read_bytes()==raw,'Frozen rescue artifact differs: '+path.name);return
    temp=path.with_name(path.name+'.'+str(uuid.uuid4())+'.tmp')
    with temp.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    try:os.link(temp,path)
    finally:temp.unlink(missing_ok=True)


def host_gate():
    require(socket.gethostname()==HOSTNAME,'Execute rescue only on designated server')
    require(sha(support.__file__)==OLD_CODE_SHA and sha(api.__file__)==TRANSPORT_SHA,'Frozen support/transport source differs')


def file_identity(path):
    return dict(bytes=Path(path).stat().st_size,sha256=sha(path))


def pure_502(terminals):
    return bool(terminals) and all(r.get('event')=='failed' and r.get('http_status')==502 and r.get('error')=='http_502' for r in terminals)


def inspect_original(root):
    stage=root/OLD;contract,images=support.verify_contract(stage)
    complete,starts,attempts,hashes,unknown=support.validate_logs(stage,contract,images)
    require(contract['mode']=='formal_independent_support_robustness' and len(contract['schedule'])==600,'Formal600 original contract required')
    require(not unknown,'Original unknown requests cannot be rescued')
    todo=[]
    queries={r['source_id']:r for r in contract['queries']}
    for target in contract['schedule']:
        key=pair(target)
        if key in complete:continue
        require(starts[key]==3,'First complete all original targets below cap3 before freezing rescue')
        terminals=[r for r in attempts if pair(r)==key and r['event']!='started']
        require(len(terminals)==3 and pure_502(terminals),'Only three settled pureHTTP502 failures can be rescued')
        todo.append(dict(source_id=key[0],arm=key[1],category=queries[key[0]]['category'],
            request_sha256=hashes[key],original_started=3,original_failed=3,
            original_terminal_sha256=digest(canonical(terminals))))
    return contract,images,complete,starts,attempts,hashes,todo


def prepare(args):
    host_gate();root=args.root.resolve();stage=root/RESCUE
    with api.output_lock(root/OLD),api.output_lock(stage):
        contract,images,complete,starts,attempts,hashes,todo=inspect_original(root)
        require(todo,'No cap3 HTTP502 targets require rescue')
        names=('contract.json','protocol.json','predictions.jsonl','attempts.jsonl','status.json','invocations.jsonl')
        original_files={name:file_identity(root/OLD/name) for name in names}
        amendment=dict(schema=SCHEMA,status='operational_amendment_frozen',original_stage=OLD,
            original_contract_sha256=sha(root/OLD/'contract.json'),original_contract_canonical_sha256=digest(canonical(contract)),
            original_files=original_files,original_accepted=len(complete),original_started=sum(starts.values()),
            original_failures=sum(r['event']=='failed' for r in attempts),eligible_targets=todo,expected_total=600,
            scientific_change=False,reason='At most two extra explicit starts for fully settled threeHTTP502 failures',
            max_additional_starts_per_target=2,max_total_starts_per_target=5,workers=1,
            one_start_per_target_per_invocation=True,min_seconds_before_second_extra_start=5,
            retry_requires_new_explicit_invocation=True,retry_only_after_http502=True,unknown_completion_not_retried=True,
            original_logs_exactly_immutable_after_amendment=True,original_log_prefix_must_remain_identical=True,
            request_builder='Unmodified support.payload(original_query,original_arm,original_contract,original_images,original_canonical_contract_sha)',
            request_uuid_assets_png_question_support_order_unchanged=True,no_gt_or_metrics_read=True,
            runner_sha256=sha(__file__),support_source_sha256=sha(support.__file__),transport_source_sha256=sha(api.__file__),
            dependencies=support.dependencies(),confirmation_opened=False)
        publish(stage/'operational_amendment.json',amendment)
        _,_,_,_,_,_,_,_,status=audit(root)
        api.write_json(stage/'status.json',status)
        print(json.dumps(dict(status='prepared',eligible=len(todo),old_accepted=len(complete),
            amendment_sha256=sha(stage/'operational_amendment.json'),no_gt_or_metrics_read=True)),flush=True)
    return 0


def load_original(root):
    host_gate();stage=root/RESCUE;amendment=read(stage/'operational_amendment.json')
    require(amendment['schema']==SCHEMA and amendment['scientific_change'] is False,'Operational amendment differs')
    require(amendment['runner_sha256']==sha(__file__) and amendment['support_source_sha256']==sha(support.__file__)
        and amendment['transport_source_sha256']==sha(api.__file__),'Frozen rescue source changed')
    require(amendment['dependencies']==support.dependencies(),'Frozen rescue dependencies changed')
    for name,identity in amendment['original_files'].items():
        require(file_identity(root/OLD/name)==identity,'Original file/prefix changed after rescue freeze: '+name)
    contract,images,complete,starts,attempts,hashes,todo=inspect_original(root)
    require(todo==amendment['eligible_targets'] and len(complete)==amendment['original_accepted'],'Frozen eligible rescue targets differ')
    return amendment,contract,images,complete,starts,attempts,hashes


def audit(root):
    stage=root/RESCUE;amendment,contract,images,complete,oldstarts,oldattempts,hashes=load_original(root)
    original_complete=dict(complete);eligible={pair(r):r for r in amendment['eligible_targets']}
    ah=sha(stage/'operational_amendment.json');ch=digest(canonical(contract))
    starts=Counter();intents={};terminals={};attempts=api.read_rows(stage/'attempts.jsonl')
    invocation_rows=api.read_rows(stage/'invocations.jsonl');invocations={r['invocation_id']:r for r in invocation_rows}
    require(len(invocations)==len(invocation_rows),'Duplicate rescue invocation')
    for row in invocation_rows:
        require(row['amendment_sha256']==ah and row['contract_sha256']==ch and row['explicit_invocation'] is True,'Invalid rescue invocation binding')
    seen_invocation_targets=set()
    for row in attempts:
        key=pair(row);identity=key+(row['attempt'],)
        require(key in eligible and row['request_sha256']==hashes[key] and row['contract_sha256']==ch
            and row['amendment_sha256']==ah,'Rescue request identity differs')
        require(type(row['attempt']) is int and 1<=row['attempt']<=2 and row['total_attempt']==3+row['attempt'],'Rescue extra-start cap exceeded')
        require(row['invocation_id'] in invocations,'Rescue attempt lacks invocation')
        if row['event']=='started':
            starts[key]+=1;ivkey=(row['invocation_id'],)+key
            require(starts[key]==row['attempt'] and identity not in intents and ivkey not in seen_invocation_targets,'Invalid repeated rescue start')
            seen_invocation_targets.add(ivkey)
            if row['attempt']==2:
                previous=terminals.get(key+(1,));require(previous is not None and pure_502([previous]),'Second extra start requires settledHTTP502')
                require(previous['invocation_id']!=row['invocation_id'] and invocations[row['invocation_id']]['retry_failed'] is True,'Second extra start needs explicit newretry invocation')
                elapsed=(datetime.fromisoformat(row['timestamp_utc'])-datetime.fromisoformat(previous['timestamp_utc'])).total_seconds()
                require(elapsed>=5,'Second extra start occurred before5second cooling interval')
            intents[identity]=row
        else:
            require(row['event'] in ('failed','succeeded') and identity in intents and identity not in terminals,'Invalid rescue terminal')
            require(row['invocation_id']==intents[identity]['invocation_id'],'Rescue terminal invocation changed')
            terminals[identity]=row
    rescue_predictions=api.read_rows(stage/'predictions.jsonl');rescued={}
    for row in rescue_predictions:
        key=pair(row);identity=key+(row['attempt'],)
        require(key in eligible and key not in complete and key not in rescued,'Unexpected/duplicate rescued acceptance')
        require(row['contract_sha256']==ch and row['request_sha256']==hashes[key] and row['amendment_sha256']==ah,'Rescue prediction request differs')
        require(identity in intents and row['invocation_id']==intents[identity]['invocation_id']
            and row['total_attempt']==3+row['attempt'],'Rescue acceptance lacks durable intent')
        api.response_probability(dict(model=row['response_model'],answers={'anomaly':dict(type='noul',noul=row['probability'])}))
        require(row['category']==eligible[key]['category'] and api.safe_usage({'usage':row['usage']})==row['usage'],'Rescue acceptance metadata differs')
        require(type(row['seconds']) in (int,float) and math.isfinite(row['seconds']) and row['seconds']>=0,'Invalid rescue seconds')
        rescued[key]=row
    for identity,row in terminals.items():
        if row['event']=='succeeded':require(identity[:2] in rescued and rescued[identity[:2]]['attempt']==identity[2],'Rescue success absent')
    unknown=[identity for identity in intents if identity not in terminals and identity[:2] not in rescued]
    complete.update(rescued)
    status=dict(schema=SCHEMA,status='complete_pending_evaluation' if len(complete)==600 and not unknown else 'incomplete',
        timestamp_utc=api.now(),expected=600,completed=len(complete),original_completed=len(original_complete),
        rescued_completed=len(rescued),eligible=len(eligible),additional_started=sum(starts.values()),
        additional_failed=sum(r['event']=='failed' for r in attempts),unknown=len(unknown),
        exhausted_unaccepted=sum(starts[key]>=2 and key not in rescued for key in eligible),
        amendment_sha256=ah,original_contract_sha256=sha(root/OLD/'contract.json'),
        original_logs_unchanged=True,scientific_change=False,confirmation_opened=False)
    return amendment,contract,images,complete,starts,attempts,hashes,unknown,status


def dry_run(args):
    root=args.root.resolve();*_,status=audit(root);print(json.dumps(status),flush=True);return 0


def run(args):
    root=args.root.resolve();stage=root/RESCUE
    with api.output_lock(root/OLD),api.output_lock(stage):
        amendment,contract,images,complete,starts,attempts,hashes,unknown,status=audit(root)
        require(not unknown,'Unsettled rescue request: no automatic or explicit resend of unknown completion')
        if len(complete)==600:api.write_json(stage/'status.json',status);return 0
        eligible=[pair(row) for row in amendment['eligible_targets']];pending=[]
        for key in eligible:
            if key in complete or starts[key]>=2:continue
            if starts[key]==0:pending.append(key)
            elif args.retry_failed:
                previous=[r for r in attempts if pair(r)==key and r['event']!='started']
                if len(previous)==1 and pure_502(previous):pending.append(key)
        if not pending or (stage/'STOP').exists():
            status.update(status='incomplete_requires_audit_or_explicit_invocation',stop_reason='STOP' if (stage/'STOP').exists() else 'no_eligible_targets')
            api.write_json(stage/'status.json',status);print(json.dumps(status),flush=True);return 2
        credential=os.environ.get(args.key_env)
        if not credential:
            require(sys.stdin.isatty(),'Credential env unset; hidden input needsTTY');credential=getpass.getpass('ModelGo key (hidden, never saved): ')
        require(isinstance(credential,str) and credential and credential.strip()==credential,'Missing credential')
        invocation=str(uuid.uuid4());ah=sha(stage/'operational_amendment.json');ch=digest(canonical(contract))
        api.append_json(stage/'invocations.jsonl',dict(invocation_id=invocation,timestamp_utc=api.now(),explicit_invocation=True,
            retry_failed=args.retry_failed,amendment_sha256=ah,contract_sha256=ch,pending=len(pending),automatic_http_retries=False))
        queries={r['source_id']:r for r in contract['queries']};consecutive=0;stop=None
        for key in pending:
            if (stage/'STOP').exists():stop='STOP';break
            if starts[key]:
                previous=[r for r in attempts if pair(r)==key and r['event']=='failed'][-1]
                elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(previous['timestamp_utc'])).total_seconds()
                if elapsed<5:time.sleep(5-elapsed)
                if (stage/'STOP').exists():stop='STOP';break
            body=support.payload(queries[key[0]],key[1],contract,images,ch)
            require(digest(canonical(body))==hashes[key],'Rescue body/UUID/PNG changed before send')
            attempt=starts[key]+1
            intent=dict(source_id=key[0],arm=key[1],attempt=attempt,total_attempt=3+attempt,event='started',
                invocation_id=invocation,timestamp_utc=api.now(),request_sha256=hashes[key],contract_sha256=ch,amendment_sha256=ah)
            api.append_json(stage/'attempts.jsonl',intent)
            value,error,http_status,seconds=api.send(body,credential)
            if value is not None:
                row=dict(**value,source_id=key[0],arm=key[1],category=queries[key[0]]['category'],attempt=attempt,total_attempt=3+attempt,
                    invocation_id=invocation,request_sha256=hashes[key],contract_sha256=ch,amendment_sha256=ah,seconds=seconds,timestamp_utc=api.now())
                api.append_json(stage/'predictions.jsonl',row);consecutive=0
                terminal=dict(intent,event='succeeded',http_status=http_status,seconds=seconds,timestamp_utc=api.now())
            else:
                consecutive+=1;terminal=dict(intent,event='failed',error=error,http_status=http_status,seconds=seconds,timestamp_utc=api.now())
            api.append_json(stage/'attempts.jsonl',terminal)
            amendment,contract,images,complete,starts,attempts,hashes,unknown,status=audit(root)
            status.update(status='running' if len(complete)<600 else 'complete_pending_evaluation')
            api.write_json(stage/'status.json',status)
            print(json.dumps(dict(completed=len(complete),expected=600,http_status=http_status,error=error,total_attempt=3+attempt)),flush=True)
            if error and error!='http_502':stop='non502_requires_audit';break
            if consecutive>=3:stop='consecutive_errors';break
        *_,status=audit(root)
        if status['completed']<600:status.update(status='incomplete_requires_audit_or_explicit_invocation',stop_reason=stop)
        api.write_json(stage/'status.json',status);print(json.dumps(status),flush=True)
        return 0 if status['completed']==600 else 2


def combined_rows(contract,complete):
    return [complete[pair(row)] for row in contract['schedule']]


def evaluate(args):
    root=args.root.resolve();stage=root/RESCUE
    with api.output_lock(root/OLD),api.output_lock(stage):
        amendment,contract,images,complete,starts,attempts,hashes,unknown,status=audit(root)
        require(len(complete)==600 and not unknown,'Need all600 settled acceptances before reading labels')
        rows=combined_rows(contract,complete)
        text=''.join(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n' for row in rows)
        publish(stage/'combined_predictions.jsonl',text,text=True)
        freeze=dict(schema=SCHEMA,contract_sha256=sha(root/OLD/'contract.json'),predictions_sha256=sha(stage/'combined_predictions.jsonl'),
            original_predictions_sha256=sha(root/OLD/'predictions.jsonl'),original_attempts_sha256=sha(root/OLD/'attempts.jsonl'),
            rescue_predictions_sha256=sha(stage/'predictions.jsonl'),rescue_attempts_sha256=sha(stage/'attempts.jsonl'),
            operational_amendment_sha256=sha(stage/'operational_amendment.json'),runner_sha256=sha(__file__),
            original_support_runner_sha256=sha(support.__file__),predictions_before_metric_labels=True,confirmation_opened=False)
        publish(stage/'prediction_freeze.json',freeze)
        config=read(root/'source_baseline/frozen/config.json');labels=read(root/'source_baseline/frozen/labels.json')
        require(digest(canonical(labels))==config['labels_sha256'],'Frozen development label identity differs')
        queries=contract['queries'];y=np.array([labels[q['source_id']] for q in queries],dtype=int)
        require(len(y)==120 and sum(y)==60,'Need60normal/60anomaly development queries')
        probabilities={arm:np.array([complete[q['source_id'],arm]['probability'] for q in queries]) for arm in support.ARMS}
        summary={}
        for arm,p in probabilities.items():
            rr=[complete[q['source_id'],arm] for q in queries];fields=sorted({k for r in rr for k in r['usage']})
            summary[arm]=dict(metrics=support.metrics(y,p),usage={k:dict(observed_sum=float(sum(r['usage'][k] for r in rr if k in r['usage'])),
                observed_calls=sum(k in r['usage'] for r in rr),missing_calls=sum(k not in r['usage'] for r in rr)) for k in fields},
                seconds=dict(mean=float(np.mean([r['seconds'] for r in rr])),median=float(np.median([r['seconds'] for r in rr])),p95=float(np.quantile([r['seconds'] for r in rr],.95))))
        by_category={}
        for category in sorted({q['category'] for q in queries}):
            idx=np.array([i for i,q in enumerate(queries) if q['category']==category])
            by_category[category]={arm:support.metrics(y[idx],p[idx]) for arm,p in probabilities.items()}
        macro={arm:{k:float(np.mean([by_category[c][arm][k] for c in by_category])) for k in support.METRIC_NAMES} for arm in support.ARMS}
        original_attempts=api.read_rows(root/OLD/'attempts.jsonl')
        result=dict(schema=support.SCHEMA,status='complete',timestamp_utc=api.now(),hostname=socket.gethostname(),prediction_freeze=freeze,
            arms=summary,by_category=by_category,category_macro=macro,paired_bootstrap=support.bootstrap(queries,y,probabilities),
            normal_bank_verified=contract['normal_bank_verified'],selected_supports=contract['n_unique_selected_supports'],
            cross_seed_support_overlap=contract['cross_seed_support_overlap'],
            operations=dict(accepted=600,started=sum(r['event']=='started' for r in original_attempts)+sum(starts.values()),
                failed=sum(r['event']=='failed' for r in original_attempts+attempts),unknown=0,original_accepted=amendment['original_accepted'],
                rescue_accepted=600-amendment['original_accepted'],rescue_started=sum(starts.values())),
            operational_amendment=dict(path=RESCUE+'/operational_amendment.json',sha256=sha(stage/'operational_amendment.json'),
                scientific_change=False,additional_starts_cap=2,original_logs_unchanged=True),
            limitations=['Previously viewed120 development queries, not confirmation','Two fixed support banks; not broad support-distribution evidence',
                'Shared0 collected once; excludes provider-repeat variance','Bootstrap conditions on fixed supports and does not capture provider variation',
                'More supports changes information/image/token budgets','Original seed42 renderer remains historical control',
                'Training-normal bank independent of teacher supervised split','AdditionalHTTP502-only retries occur later; possible provider/time variation',
                'No currency bill or cross-domain evaluation'],confirmation_opened=False,no_gt_tuning=True)
        metric_path=stage/'metrics.json'
        if metric_path.exists():
            old=read(metric_path);require(all(old[k]==v for k,v in result.items() if k!='timestamp_utc'),'Frozen rescue metrics differ');result=old
        else:publish(metric_path,result)
        lines=['# Jev stage08 HTTP502运维修订与完整结果','','原科学contract、支持seed、PNG、问题、请求UUID/body与已接受响应均保持不变。',
            '仅对原3次均明确HTTP502的耗尽目标增加最多2次显式单并发请求；原日志和status未改写。','',
            '| 条件 | AUROC | AP | Recall | FPR | Brier | ECE10 |','|---|---:|---:|---:|---:|---:|---:|']
        for arm in support.ARMS:lines.append('| '+arm+' | '+' | '.join(f"{summary[arm]['metrics'][k]:.6f}" for k in ('auroc','ap','recall','fpr','brier','ece_10_equal_width'))+' |')
        lines+=['','## 结果分析','']
        for seed in support.SEEDS:
            value=summary['s'+str(seed)+'_4']['metrics']['auroc']-summary['shared0']['metrics']['auroc']
            lines.append(f'- seed{seed} 4参考相对shared0 AUROC变化{100*value:+.2f}个百分点；完整配对区间见metrics.json。')
        lines+=['','## 运维与解释边界','',
            f"- 原接受{amendment['original_accepted']}；救援接受{600-amendment['original_accepted']}；额外started={sum(starts.values())}。",
            '- completion_certificate明确为operational amendment完整，不将原stage08/status伪装为complete。',
            '- 120均为已看开发query；source/byte隔离与teacher exposure按原contract，未打开confirmation。',
            '- 两支持seed仅作固定bank描述；bootstrap不包括支持选择、provider随机性、时间变化或多重比较校正。',
            '- 重试为显式HTTP502运维恢复，不据此宣称新科学效果；缺失usage不计零，不估货币账单。','']
        publish(stage/'REPORT.md','\n'.join(lines),text=True)
        certificate=dict(schema=SCHEMA,status='complete_with_operational_amendment',expected=600,completed=600,
            original_stage_status=read(root/OLD/'status.json')['status'],original_status_unchanged=True,
            original_contract_sha256=sha(root/OLD/'contract.json'),original_files=amendment['original_files'],
            operational_amendment_sha256=sha(stage/'operational_amendment.json'),
            combined_predictions_sha256=sha(stage/'combined_predictions.jsonl'),prediction_freeze_sha256=sha(stage/'prediction_freeze.json'),
            rescue_predictions_sha256=sha(stage/'predictions.jsonl'),rescue_attempts_sha256=sha(stage/'attempts.jsonl'),
            metrics_sha256=sha(stage/'metrics.json'),report_sha256=sha(stage/'REPORT.md'),runner_sha256=sha(__file__),
            scientific_change=False,no_gt_tuning=True,confirmation_opened=False)
        require(certificate['original_stage_status']!='complete','Original stage must not be relabeled complete')
        publish(stage/'completion_certificate.json',certificate)
        status.update(status='complete_with_operational_amendment',metrics_sha256=sha(stage/'metrics.json'),certificate_sha256=sha(stage/'completion_certificate.json'))
        api.write_json(stage/'status.json',status)
        print(json.dumps(dict(status=status['status'],completed=600,certificate=str(stage/'completion_certificate.json'))),flush=True)
    return 0


def load_complete(root):
    """Read-only explicit downstream adapter; never masquerades as old complete."""
    root=Path(root).resolve();stage=root/RESCUE
    amendment,contract,images,complete,starts,attempts,hashes,unknown,status=audit(root)
    certificate=read(stage/'completion_certificate.json')
    require(certificate['status']=='complete_with_operational_amendment' and certificate['completed']==certificate['expected']==600
        and len(complete)==600 and not unknown,'Amendment completion certificate incomplete')
    for field,name in [('operational_amendment_sha256','operational_amendment.json'),('combined_predictions_sha256','combined_predictions.jsonl'),
        ('prediction_freeze_sha256','prediction_freeze.json'),('rescue_predictions_sha256','predictions.jsonl'),
        ('rescue_attempts_sha256','attempts.jsonl'),('metrics_sha256','metrics.json'),('report_sha256','REPORT.md')]:
        require(certificate[field]==sha(stage/name),'Completion certificate artifact differs: '+name)
    require(api.read_rows(stage/'combined_predictions.jsonl')==combined_rows(contract,complete),'Combined accepted rows differ from original+rescue')
    oldattempts=api.read_rows(root/OLD/'attempts.jsonl');oldstarts=Counter(pair(r) for r in oldattempts if r['event']=='started')
    return contract,images,complete,oldstarts+starts,oldattempts+attempts,hashes,unknown


def self_test(args):
    host_gate()
    require(pure_502([dict(event='failed',http_status=502,error='http_502')]),'Pure502 classification failed')
    require(not pure_502([dict(event='failed',http_status=None,error='timeout')]),'Unknown timeout must not qualify')
    require(not pure_502([dict(event='failed',http_status=503,error='http_503')]),'Non502 must not qualify')
    require(not pure_502([]),'Empty evidence must not qualify')
    contract=dict(schedule=[dict(source_id='b',arm='shared0'),dict(source_id='a',arm='s43_4')])
    fixture={('a','s43_4'):dict(source_id='a',arm='s43_4'),('b','shared0'):dict(source_id='b',arm='shared0')}
    require([r['source_id'] for r in combined_rows(contract,fixture)]==['b','a'],'Combined schedule order failed')
    print(json.dumps(dict(status='passed',synthetic_only=True,network_calls=0,
        checks=['pure502_only','unknown_timeout_excluded','otherHTTP_excluded','empty_evidence_excluded','original_schedule_order'])),flush=True);return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','dry-run','run','evaluate','self-test'):group.add_argument('--'+name,action='store_true')
    parser.add_argument('--retry-failed',action='store_true');parser.add_argument('--key-env',default='MODELGO_API_KEY')
    args=parser.parse_args();functions=dict(prepare=prepare,dry_run=dry_run,run=run,evaluate=evaluate,self_test=self_test)
    try:return next(fn(args) for name,fn in functions.items() if getattr(args,name))
    except Exception as error:
        print(json.dumps(dict(status='blocked',exception_class=type(error).__name__,reason=str(error) if isinstance(error,ValueError) else 'Inspect runtime/frozen rescue artifacts; arbitrary exception suppressed')),file=sys.stderr);return 2


if __name__=='__main__':sys.exit(main())
