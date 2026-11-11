#!/usr/bin/env python3
"""Server-only independent 320-versus448 whole-image Jev0-shot ablation.

prepare --root ROOT [--stage stage10_resolution] [--limit-queries 1]
run --root ROOT [--stage stage10_resolution] [--retry-failed]
evaluate --root ROOT [--stage stage10_resolution]
self-test

New320 has120 accepted targets. 448 is the immutable stage08 shared0 arm;
the stage08 support conditions are not reused as resolution observations.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import getpass
import io
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import uuid

import numpy as np
from PIL import Image,ImageOps
import roi_api as api
import support_robustness as support

SCHEMA='jev-independent-resolution-ablation-v1'
HOSTNAME='autodl-container-132444827f-9c594038'
SIZE=320
ARM='size320_0shot'
REFERENCE='shared0_448'
METRICS=support.METRIC_NAMES


def require(value,message):
    if not value: raise ValueError(message)


def host_gate(): require(socket.gethostname()==HOSTNAME,'Execute resolution experiment only on designated AutoDL server')
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return api.file_sha(path)
def digest(value): return api.digest(value)
def canonical(value): return api.canonical(value)


def render(raw):
    with Image.open(io.BytesIO(raw)) as opened:
        original=list(opened.size)
        image=ImageOps.exif_transpose(opened).convert('RGB')
        oriented=list(image.size)
        image=ImageOps.contain(image,(SIZE,SIZE),method=Image.Resampling.LANCZOS)
        resized=list(image.size); padding=[(SIZE-image.width)//2,(SIZE-image.height)//2]
        canvas=Image.new('RGB',(SIZE,SIZE),'white'); canvas.paste(image,tuple(padding))
        output=io.BytesIO(); canvas.save(output,format='PNG',compress_level=6)
    return output.getvalue(),dict(original_size=original,oriented_size=oriented,resized_size=resized,padding=padding)


def validated320(stage,metadata):
    require(metadata.get('size')==[320,320] and api.valid_hash(metadata.get('png_sha256')),'Invalid320 asset metadata')
    path=api.inside(stage,metadata['path']); raw=path.read_bytes()
    require(len(raw)==metadata['png_bytes'] and digest(raw)==metadata['png_sha256'],'320 PNG integrity differs')
    with Image.open(io.BytesIO(raw)) as image:
        require(image.format=='PNG' and image.mode=='RGB' and image.size==(320,320),'ExpectedRGB320PNG')
        image.load()
    return raw


def protocol():
    return dict(schema=SCHEMA,model=support.MODEL,endpoint=api.ENDPOINT,arms=[ARM,REFERENCE],
        new_targets=120,reference_source='stage08 shared0 accepted predictions; never support-arm480',
        query_population='Same120 previously viewed development queries',normal_references=0,
        instructions=support.INSTRUCTIONS,criteria=support.CRITERIA,threshold=.5,ece_bins=10,
        preprocessing='EXIF transpose; RGB; LANCZOS contain; centered white pad; PNG compress_level6',
        controlled_difference='Canvas maximum320 vs448; same Pillow renderer/source SHA/question/model; size changes detail/image-token budget',
        size320_interface_note='Each submitted edge320; capability must pass independent preflight on current gateway',
        reference_caveat='448responses collected earlier in stage08; no interleaved-time control or provider-repeat measurement',
        workers=3,connect_timeout=60,read_timeout=120,max_started_per_target=3,automatic_retries=False,
        consecutive_error_submission_stop=3,bootstrap=dict(replicates=2000,seed=42,strata='category x image_label',paired=True),
        new_resolution_experiment=True,no_gt_in_requests=True,confirmation_opened=False)


def prepare(args):
    host_gate(); root=args.root.resolve(); stage=api.inside(root,args.stage); reference_stage=api.inside(root,args.reference_stage)
    with api.output_lock(stage):
        reference_contract_path=reference_stage/'contract.json'; reference_contract=read(reference_contract_path)
        require(reference_contract.get('schema')==support.SCHEMA and reference_contract.get('mode')=='formal_independent_support_robustness'
            and len(reference_contract.get('queries',[]))==120 and len(reference_contract.get('schedule',[]))==600,'Need formal stage08 contract')
        require(reference_contract['runner_sha256']==sha(support.__file__) and reference_contract['transport_source_sha256']==sha(api.__file__),'Reference sources changed')
        require(reference_contract['dependencies']==support.dependencies(),'Reference/new runtime differs')
        require(reference_contract['new_renderer']=='12.3.0','Pillow12.3 renderer required')
        queries=reference_contract['queries']
        require(queries==sorted(queries,key=lambda q:q['source_id']) and len({q['source_id'] for q in queries})==120,'Reference population/order differs')
        require(args.limit_queries is None or 1<=args.limit_queries<=120,'Invalid independent preflight size')
        selected=queries if args.limit_queries is None else queries[:args.limit_queries]
        records=[]
        for row in selected:
            path=Path(row['original_path'])
            raw=path.read_bytes()
            require(digest(raw)==row['original_sha256'] and len(raw)==row['original_bytes'],'Original query SHA/size changed')
            _,reference_png=api.validated_asset(reference_stage,row['asset'])
            png,geometry=render(raw)
            destination=stage/'assets'/(row['source_id']+'.png'); destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists(): require(destination.read_bytes()==png,'Existing320 asset differs')
            else:
                with destination.open('xb') as stream: stream.write(png);stream.flush();os.fsync(stream.fileno())
            records.append(dict(source_id=row['source_id'],category=row['category'],original_path=str(path),
                original_sha256=row['original_sha256'],original_bytes=row['original_bytes'],
                asset320=dict(path=destination.relative_to(stage).as_posix(),png_sha256=digest(png),png_bytes=len(png),size=[320,320],**geometry),
                reference448=dict(stage=args.reference_stage,**row['asset'])))
        api.immutable_json(stage/'protocol.json',protocol())
        candidates_path=root/'stages/03_candidates.json'
        contract=dict(schema=SCHEMA,model=support.MODEL,endpoint=api.ENDPOINT,size=320,
            mode='formal_independent_resolution' if args.limit_queries is None else 'separate_preflight_not_formal',
            limit_queries=args.limit_queries,reference_stage=args.reference_stage,reference_contract_sha256=sha(reference_contract_path),
            candidates_sha256=sha(candidates_path),protocol_sha256=sha(stage/'protocol.json'),
            runner_sha256=sha(__file__),support_source_sha256=sha(support.__file__),transport_source_sha256=sha(api.__file__),
            dependencies=support.dependencies(),queries=records,
            schedule=[dict(source_id=row['source_id'],arm=ARM) for row in records],
            no_reference_predictions_read_during_prepare=True,no_gt_read_during_prepare=True,confirmation_opened=False)
        api.immutable_json(stage/'contract.json',contract)
        images=load_assets(stage,contract)
        completed,starts,attempts,hashes,unknown=validate_logs(stage,contract,images)
        status=state(contract,completed,starts,attempts,unknown,'complete' if len(completed)==len(records) else 'prepared')
        api.write_json(stage/'status.json',status);print(json.dumps(status),flush=True)
    return 0


def load_assets(stage,contract): return {q['source_id']:validated320(stage,q['asset320']) for q in contract['queries']}


def verify(root,stage):
    contract=read(stage/'contract.json')
    require(contract['schema']==SCHEMA and contract['model']==support.MODEL and contract['size']==320,'Resolution contract identity differs')
    require(contract['runner_sha256']==sha(__file__) and contract['support_source_sha256']==sha(support.__file__)
        and contract['transport_source_sha256']==sha(api.__file__),'Frozen resolution source changed')
    require(contract['dependencies']==support.dependencies(),'Frozen resolution dependencies changed')
    require(contract['protocol_sha256']==sha(stage/'protocol.json') and read(stage/'protocol.json')==protocol(),'Resolution protocol changed')
    reference_stage=api.inside(root,contract['reference_stage'])
    require(contract['reference_contract_sha256']==sha(reference_stage/'contract.json'),'Frozen448 contract changed')
    require(contract['candidates_sha256']==sha(root/'stages/03_candidates.json'),'Frozen query candidate provenance changed')
    for row in contract['queries']: api.validated_asset(reference_stage,row['reference448'])
    return contract,load_assets(stage,contract)


def payload(row,images,contract_hash):
    import base64
    reference=str(uuid.uuid5(uuid.NAMESPACE_URL,contract_hash+'|'+row['source_id']+'|'+ARM+'|query'))
    return dict(model=support.MODEL,state=dict(query=reference,category=row['category'].replace('_',' '),normal_references=[]),
        assets={reference:dict(type='image',format='png',data_base64=base64.b64encode(images[row['source_id']]).decode('ascii'))},
        questions={'anomaly':dict(type='noul',instructions=support.INSTRUCTIONS,criteria=support.CRITERIA)})


def validate_logs(stage,contract,images):
    contract_hash=digest(canonical(contract))
    queries={q['source_id']:q for q in contract['queries']}
    hashes={sid:digest(canonical(payload(q,images,contract_hash))) for sid,q in queries.items()}
    completed={}
    for row in api.read_rows(stage/'predictions.jsonl'):
        sid=row['source_id']
        require(sid in hashes and sid not in completed and row['arm']==ARM,'Unexpected or duplicate resolution accepted target')
        require(row['contract_sha256']==contract_hash and row['request_sha256']==hashes[sid],'Resolution prediction identity differs')
        api.response_probability(dict(model=row['response_model'],answers={'anomaly':dict(type='noul',noul=row['probability'])}))
        require(row['category']==queries[sid]['category'] and api.safe_usage({'usage':row['usage']})==row['usage'],'Saved resolution metadata differs')
        require(type(row['seconds']) in (int,float) and math.isfinite(row['seconds']) and row['seconds']>=0,'Invalid resolution latency')
        completed[sid]=row
    attempts=api.read_rows(stage/'attempts.jsonl');starts=Counter();intents={};terminal=set()
    for row in attempts:
        sid=row['source_id'];identity=sid,row['attempt']
        require(sid in hashes and row['arm']==ARM and row['contract_sha256']==contract_hash and row['request_sha256']==hashes[sid],'Resolution attempt identity differs')
        require(type(row['attempt']) is int and 1<=row['attempt']<=3,'Resolution attempt budget exceeded')
        if row['event']=='started':
            starts[sid]+=1;require(starts[sid]==row['attempt'] and identity not in intents,'Resolution intent sequence differs');intents[identity]=row
        else:
            require(row['event'] in ('failed','succeeded') and identity in intents and identity not in terminal,'Invalid resolution terminal')
            require(row['invocation_id']==intents[identity]['invocation_id'],'Resolution invocation differs');terminal.add(identity)
            if row['event']=='succeeded': require(sid in completed and completed[sid]['attempt']==row['attempt'],'Succeeded resolution target missing')
    for sid,row in completed.items(): require((sid,row['attempt']) in intents,'Resolution success lacks durable intent')
    unknown=[identity for identity in intents if identity not in terminal and identity[0] not in completed]
    return completed,starts,attempts,hashes,unknown


def state(contract,completed,starts,attempts,unknown,status,reason=None):
    return dict(schema=SCHEMA,status=status,timestamp_utc=api.now(),mode=contract['mode'],
        contract_sha256=digest(canonical(contract)),expected=len(contract['schedule']),completed=len(completed),
        started_attempts=sum(starts.values()),failed_attempts=sum(r['event']=='failed' for r in attempts),
        interrupted_unknown_attempts=len(unknown),exhausted_unaccepted=sum(starts[r['source_id']]>=3 and r['source_id'] not in completed for r in contract['schedule']),
        stop_reason=reason,automatic_retries=False,confirmation_opened=False)


def run(args):
    host_gate();root=args.root.resolve();stage=api.inside(root,args.stage)
    with api.output_lock(stage):
        contract,images=verify(root,stage);completed,starts,attempts,hashes,unknown=validate_logs(stage,contract,images)
        pending=[r['source_id'] for r in contract['schedule'] if r['source_id'] not in completed and starts[r['source_id']]<3
                 and (starts[r['source_id']]==0 or args.retry_failed)]
        if len(completed)==len(contract['schedule']): api.write_json(stage/'status.json',state(contract,completed,starts,attempts,unknown,'complete'));return 0
        if not pending or (stage/'STOP').exists():
            api.write_json(stage/'status.json',state(contract,completed,starts,attempts,unknown,'incomplete_requires_audit_or_explicit_retry',
                'STOP' if (stage/'STOP').exists() else 'no_eligible_targets'));return 2
        key=os.environ.get(args.key_env)
        if not key:
            require(sys.stdin.isatty(),'Credential env unset; hidden input requires TTY');key=getpass.getpass('ModelGo API key (hidden, never saved): ')
        require(isinstance(key,str) and key and key.strip()==key,'Missing or invalid credential')
        invocation=str(uuid.uuid4());contract_hash=digest(canonical(contract));queries={q['source_id']:q for q in contract['queries']}
        api.append_json(stage/'invocations.jsonl',dict(invocation_id=invocation,timestamp_utc=api.now(),contract_sha256=contract_hash,
            retry_failed=args.retry_failed,unknown_completion_retry_risk_acknowledged=bool(args.retry_failed and unknown)))
        futures={};position=0;consecutive=0;stop=None
        with ThreadPoolExecutor(max_workers=3) as pool:
            while position<len(pending) or futures:
                if (stage/'STOP').exists(): stop=stop or 'STOP_file'
                while not stop and position<len(pending) and len(futures)<3:
                    sid=pending[position];position+=1;body=payload(queries[sid],images,contract_hash)
                    require(digest(canonical(body))==hashes[sid],'Resolution payload changed before send');starts[sid]+=1
                    intent=dict(source_id=sid,arm=ARM,attempt=starts[sid],event='started',timestamp_utc=api.now(),
                        contract_sha256=contract_hash,request_sha256=hashes[sid],invocation_id=invocation)
                    api.append_json(stage/'attempts.jsonl',intent);attempts.append(intent);futures[pool.submit(api.send,body,key)]=(sid,intent)
                if not futures: break
                done,_=wait(futures,timeout=.5,return_when=FIRST_COMPLETED)
                for future in done:
                    sid,intent=futures.pop(future);value,error,http_status,seconds=future.result()
                    if value is not None:
                        row=dict(**value,source_id=sid,arm=ARM,category=queries[sid]['category'],attempt=intent['attempt'],seconds=seconds,
                            contract_sha256=contract_hash,request_sha256=hashes[sid],invocation_id=invocation,timestamp_utc=api.now())
                        api.append_json(stage/'predictions.jsonl',row);completed[sid]=row;consecutive=0
                        terminal=dict(intent,event='succeeded',timestamp_utc=api.now(),http_status=http_status,seconds=seconds)
                    else:
                        terminal=dict(intent,event='failed',timestamp_utc=api.now(),http_status=http_status,seconds=seconds,error=error);consecutive+=1
                        if consecutive>=3: stop=stop or 'consecutive_errors'
                    api.append_json(stage/'attempts.jsonl',terminal);attempts.append(terminal)
                    api.write_json(stage/'status.json',state(contract,completed,starts,attempts,unknown,'draining_inflight' if stop else 'running',stop))
                    print(json.dumps(dict(completed=len(completed),expected=len(contract['schedule']),http_status=http_status,error=error,seconds=seconds)),flush=True)
        completed,starts,attempts,hashes,unknown=validate_logs(stage,contract,images)
        complete=len(completed)==len(contract['schedule']);status=state(contract,completed,starts,attempts,unknown,'complete' if complete else 'incomplete_requires_explicit_retry',stop)
        api.write_json(stage/'status.json',status);print(json.dumps(status),flush=True);return 0 if complete else 2


def paired_bootstrap(queries,y,p320,p448):
    strata=defaultdict(list)
    for i,q in enumerate(queries): strata[q['category'],int(y[i])].append(i)
    rng=np.random.default_rng(42);weights=np.zeros((2000,len(y)))
    for _,members in sorted(strata.items()): weights[:,members]=rng.multinomial(len(members),np.full(len(members),1/len(members)),size=2000)
    a=support.weighted_metrics(y,p320,weights);b=support.weighted_metrics(y,p448,weights)
    ma=support.metrics(y,p320);mb=support.metrics(y,p448)
    categories=sorted({q['category'] for q in queries});category_boot=[];category_points=[]
    for category in categories:
        idx=np.array([i for i,q in enumerate(queries) if q['category']==category])
        ca=support.weighted_metrics(y[idx],p320[idx],weights[:,idx]);cb=support.weighted_metrics(y[idx],p448[idx],weights[:,idx])
        cma=support.metrics(y[idx],p320[idx]);cmb=support.metrics(y[idx],p448[idx])
        category_boot.append({k:ca[k]-cb[k] for k in METRICS});category_points.append({k:cma[k]-cmb[k] for k in METRICS})
    def item(point,values): return dict(delta=float(point),ci95=np.quantile(values,[.025,.975],method='linear').tolist(),replicates=2000,seed=42,strata='category x image_label',paired=True)
    return dict(direction='320-minus448',pooled={k:item(ma[k]-mb[k],a[k]-b[k]) for k in METRICS},
        category_macro={k:item(np.mean([c[k] for c in category_points]),np.mean([c[k] for c in category_boot],axis=0)) for k in METRICS})


def budget(rows,size):
    fields=sorted({k for r in rows for k in r['usage']})
    return dict(accepted_requests=len(rows),images_per_request=1,canvas_pixels_per_request=size*size,
        total_canvas_pixels=len(rows)*size*size,observed_usage={k:dict(observed_sum=float(sum(r['usage'][k] for r in rows if k in r['usage'])),
            observed_calls=sum(k in r['usage'] for r in rows),missing_calls=sum(k not in r['usage'] for r in rows)) for k in fields},
        latency_seconds=dict(mean=float(np.mean([r['seconds'] for r in rows])),median=float(np.median([r['seconds'] for r in rows])),
            p95=float(np.quantile([r['seconds'] for r in rows],.95))),
        limitation='Observed token totals only; canvas pixel count is not an exact model image-token formula; no currency estimate')


def evaluate(args):
    host_gate();root=args.root.resolve();stage=api.inside(root,args.stage)
    with api.output_lock(stage):
        contract,images=verify(root,stage);status=read(stage/'status.json')
        require(contract['mode']=='formal_independent_resolution' and status['status']=='complete'
            and status['completed']==status['expected']==120,'Complete formal320120 required')
        completed,starts,attempts,hashes,unknown=validate_logs(stage,contract,images)
        require(len(completed)==120 and not unknown,'Resolution accepted target audit incomplete')
        reference_stage=api.inside(root,contract['reference_stage']);rs=read(reference_stage/'status.json')
        rc,ri=support.verify_contract(reference_stage);reference,rstarts,rattempts,_,runknown=support.validate_logs(reference_stage,rc,ri)
        require(rc['mode']=='formal_independent_support_robustness' and rs['status']=='complete'
            and rs['completed']==rs['expected']==600 and len(reference)==600 and not runknown,'Complete stage08 formal600 required')
        queries=contract['queries'];require([q['source_id'] for q in queries]==[q['source_id'] for q in rc['queries']],'320/448 query alignment differs')
        frozen=dict(schema=SCHEMA,contract_sha256=sha(stage/'contract.json'),predictions320_sha256=sha(stage/'predictions.jsonl'),
            reference_contract_sha256=sha(reference_stage/'contract.json'),reference_predictions_sha256=sha(reference_stage/'predictions.jsonl'),
            support_source_sha256=sha(support.__file__),resolution_source_sha256=sha(__file__),confirmation_opened=False)
        api.immutable_json(stage/'prediction_freeze.json',frozen)
        labels=read(root/'source_baseline/frozen/labels.json');config=read(root/'source_baseline/frozen/config.json')
        require(digest(canonical(labels))==config['labels_sha256'],'Frozen evaluation labels changed')
        y=np.array([labels[q['source_id']] for q in queries],dtype=int);require(len(y)==120 and sum(y)==60,'Need60normal/60anomaly')
        rows320=[completed[q['source_id']] for q in queries];rows448=[reference[q['source_id'],'shared0'] for q in queries]
        p320=np.array([r['probability'] for r in rows320]);p448=np.array([r['probability'] for r in rows448])
        arms={ARM:dict(metrics=support.metrics(y,p320),budget=budget(rows320,320)),
              REFERENCE:dict(metrics=support.metrics(y,p448),budget=budget(rows448,448))}
        by_category={}
        for category in sorted({q['category'] for q in queries}):
            idx=np.array([i for i,q in enumerate(queries) if q['category']==category])
            by_category[category]={ARM:support.metrics(y[idx],p320[idx]),REFERENCE:support.metrics(y[idx],p448[idx])}
        macro={arm:{k:float(np.mean([by_category[c][arm][k] for c in by_category])) for k in METRICS} for arm in arms}
        result=dict(schema=SCHEMA,status='complete',timestamp_utc=api.now(),hostname=socket.gethostname(),prediction_freeze=frozen,
            arms=arms,by_category=by_category,category_macro=macro,paired_bootstrap=paired_bootstrap(queries,y,p320,p448),
            operations320=dict(accepted=120,started=sum(starts.values()),failed=sum(r['event']=='failed' for r in attempts),unknown=len(unknown)),
            reused448_accepted=120,excluded_support_arm_observations=480,
            canvas_pixel_ratio_320_over448=320**2/448**2,
            semantic_note='Whole-image anomaly Noul with zero references; same wording as stage08 shared0',
            limitations=['Viewed development queries, no confirmation or cross-domain evidence',
                '448 observations predate320; provider version/routing/time variation not independently excluded',
                'Downsampling changes visible detail and image budget simultaneously',
                'Bootstrap fixed queries/renderings, excludes provider repeat randomness',
                'No exact image-token formula or verified currency bill; missing usage remains unknown'],
            no_gt_tuning=True,confirmation_opened=False)
        output=stage/'metrics.json'
        if output.exists():
            old=read(output);require(all(old[k]==result[k] for k in result if k!='timestamp_utc'),'Frozen resolution result differs');result=old
        else: api.immutable_json(output,result)
        lines=['# Jev 320/448零参考分辨率研究','','同120开发query，全图问题和模型名固定；新320接受120判断，对照stage08已冻结shared0的120个448响应。',
            '其余480个支持上下文响应不混入分辨率比较；旧第一阶段结果不覆盖。','',
            '| 输入 | AUROC | AP | Recall | FPR | Brier | ECE10 |','|---|---:|---:|---:|---:|---:|---:|']
        for arm in (ARM,REFERENCE): lines.append('| '+arm+' | '+' | '.join(f"{arms[arm]['metrics'][k]:.6f}" for k in ('auroc','ap','recall','fpr','brier','ece_10_equal_width'))+' |')
        delta=result['paired_bootstrap']['pooled']
        lines+=['','## 分析与预算','',f"- 320−448 AUROC差{100*delta['auroc']['delta']:+.2f}pp；95%配对区间[{100*delta['auroc']['ci95'][0]:+.2f},{100*delta['auroc']['ci95'][1]:+.2f}]pp。",
            f'- 320每请求102,400画布像素，448为200,704，比例{320**2/448**2:.4f}；像素数不等同精确图像token数。',
            '- 每臂一图一请求，逐usage字段观察总量、缺失和延迟分位数见metrics.json；不估货币费用。',
            '- 320与448包含不同细节与预算，本轮不隔离纯算力因果；调用时间不同，未排除provider变化。',
            '- 全部阈值固定0.5；分层bootstrap固定query和渲染，不含服务重复随机性，未做多重比较校正。','']
        text='\n'.join(lines);report=stage/'REPORT.md'
        if report.exists(): require(report.read_text(encoding='utf-8')==text,'Frozen resolution report differs')
        else:
            with report.open('x',encoding='utf-8') as stream: stream.write(text);stream.flush();os.fsync(stream.fileno())
        require(sha(stage/'predictions.jsonl')==frozen['predictions320_sha256'] and sha(reference_stage/'predictions.jsonl')==frozen['reference_predictions_sha256'],'Predictions changed during resolution evaluation')
        print(json.dumps(dict(status='complete',new320=120,reused448=120,metrics=str(output),report=str(report),confirmation_opened=False)),flush=True)
    return 0


def self_test():
    host_gate()
    raw=io.BytesIO();Image.new('RGB',(640,320),'white').save(raw,format='PNG');png,g=render(raw.getvalue())
    with Image.open(io.BytesIO(png)) as image: require(image.size==(320,320) and image.mode=='RGB','320 render failed')
    require(g['resized_size']==[320,160] and g['padding']==[0,80],'320 contain/pad differs')
    q=dict(source_id='synthetic',category='pcb1');body=payload(q,{'synthetic':png},'a'*64)
    require(len(body['assets'])==1 and body['state']['normal_references']==[],'Zero-shot image budget differs')
    require(body['questions']['anomaly']['instructions']==support.INSTRUCTIONS,'Whole-image wording differs')
    y=np.array([0,0,1,1]);p=np.array([0.,.2,.8,1.]);w=np.ones((2,4))
    m=support.weighted_metrics(y,p,w);require(all(np.allclose(m[k],support.metrics(y,p)[k],atol=1e-12) for k in METRICS),'Paired metric dependency differs')
    require(not api.eligible('x',{'x':{}},Counter(),True) and not api.eligible('x',{},Counter({'x':3}),True),'Retry guard failed')
    print(json.dumps(dict(status='passed',synthetic_only=True,network_calls=0,checks=['320RGB','contain_pad','zero_reference_payload','same_question','weighted_metrics','retry_guard'])))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name,func in [('prepare',prepare),('run',run),('evaluate',evaluate)]:
        item=sub.add_parser(name);item.add_argument('--root',type=Path,required=True);item.add_argument('--stage',default='stage10_resolution')
        if name=='prepare': item.add_argument('--reference-stage',default='stage08_support');item.add_argument('--limit-queries',type=int)
        if name=='run': item.add_argument('--key-env',default='MODELGO_API_KEY');item.add_argument('--retry-failed',action='store_true')
        item.set_defaults(func=func)
    item=sub.add_parser('self-test');item.set_defaults(func=lambda args:self_test())
    args=parser.parse_args()
    try: return args.func(args)
    except Exception as error:
        print(json.dumps(dict(status='blocked',exception_class=type(error).__name__,reason=str(error) if isinstance(error,ValueError) else 'Inspect runtime/frozen inputs; arbitrary exception suppressed')),file=sys.stderr);return 2


if __name__=='__main__':sys.exit(main())
