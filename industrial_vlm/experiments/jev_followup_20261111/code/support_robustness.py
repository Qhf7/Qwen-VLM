#!/usr/bin/env python3
"""Independent server-only Jev normal-support robustness experiment.

prepare --root ROOT [--stage stage08_support] [--limit-queries 1]
run --root ROOT [--stage stage08_support] [--retry-failed]
evaluate --root ROOT [--stage stage08_support]
self-test

Formal design: new-render shared0 + seed43/44 x shots1/4 = 600 decisions.
No320 arm, GT masks, training, confirmation access or phase1 mutation.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import getpass
import hashlib
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
from PIL import Image, ImageOps
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
import roi_api as api

SCHEMA='jev-independent-support-robustness-v1'
SERVER_HOSTNAME='autodl-container-132444827f-9c594038'
SEEDS=(43,44)
ARMS=('shared0','s43_1','s43_4','s44_1','s44_4')
SIZE=448
MODEL='jev-1.14.0'
TRAIN_SHA='214778943ccb8be292c533708d12f42944ae8c8f23364560d2e221798e610da4'
AUDIT_SHA='ca5ba84aa4dc5932e127bbf62698ffe196f7c4181b33b74f8e72be9586f685b0'
DEFAULT_DATA=Path('/root/autodl-tmp/industrial_vlm')
INSTRUCTIONS=('Does the image in state field query show at least one visible industrial anomaly '
    '(surface damage, contamination, missing material or component, or structural irregularity)? '
    'Treat normal_references, if provided, as normal examples of the same object category. '
    'Judge query, not the references. Ordinary differences in viewpoint, lighting, background, '
    'or object identity alone are not defects.')
CRITERIA={'true':'At least one visible industrial defect is present in query.',
          'false':'No visible industrial defect is present in query.'}
METRIC_NAMES=('auroc','ap','macro_f1','recall','fpr','brier','ece_10_equal_width')
PAIRS=(('s43_1','shared0'),('s43_4','shared0'),('s43_4','s43_1'),
       ('s44_1','shared0'),('s44_4','shared0'),('s44_4','s44_1'),('s44_1','s43_1'),('s44_4','s43_4'))


def require(value,message):
    if not value: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return api.file_sha(path)
def canonical(value): return api.canonical(value)
def digest(value): return api.digest(value)


def host_gate():
    require(socket.gethostname()==SERVER_HOSTNAME,'Execute this experiment only on the designated AutoDL server')


def dependencies():
    value=api.versions()
    value.update(numpy=np.__version__,sklearn=__import__('sklearn').__version__)
    return value


def render(raw):
    with Image.open(io.BytesIO(raw)) as opened:
        original_size=list(opened.size)
        image=ImageOps.exif_transpose(opened).convert('RGB')
        oriented_size=list(image.size)
        image=ImageOps.contain(image,(SIZE,SIZE),method=Image.Resampling.LANCZOS)
        resized=list(image.size)
        padding=[(SIZE-image.width)//2,(SIZE-image.height)//2]
        canvas=Image.new('RGB',(SIZE,SIZE),'white')
        canvas.paste(image,tuple(padding))
        output=io.BytesIO()
        canvas.save(output,format='PNG',compress_level=6)
    return output.getvalue(),dict(original_size=original_size,oriented_size=oriented_size,resized_size=resized,padding=padding)


def select_supports(bank,seed):
    categories=sorted({r['category'] for r in bank})
    return {category:sorted([r for r in bank if r['category']==category],
        key=lambda r:(digest(('jev-support-v1|'+str(seed)+'|'+r['source_id']).encode()),r['source_id']))[:4]
        for category in categories}


def arm_details(arm):
    require(arm in ARMS,'Unknown support arm')
    if arm=='shared0': return None,0
    seed,shots=arm[1:].split('_')
    return int(seed),int(shots)


def payload(query,arm,contract,images,contract_hash):
    seed,shots=arm_details(arm)
    supports=[] if seed is None else contract['supports'][str(seed)][query['category']][:shots]
    assets={}
    identity=contract_hash+'|'+query['source_id']+'|'+arm
    def add(source,role):
        reference=str(uuid.uuid5(uuid.NAMESPACE_URL,identity+'|'+role))
        import base64
        assets[reference]=dict(type='image',format='png',data_base64=base64.b64encode(images[source['source_id']]).decode('ascii'))
        return reference
    qref=add(query,'query')
    refs=[add(source,'reference-'+str(i)) for i,source in enumerate(supports)]
    return dict(model=MODEL,state=dict(query=qref,category=query['category'].replace('_',' '),normal_references=refs),
        assets=assets,questions={'anomaly':dict(type='noul',instructions=INSTRUCTIONS,criteria=CRITERIA)})


def publish_asset(stage,sid,raw):
    png,geometry=render(raw)
    path=stage/'assets'/(sid+'.png')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): require(path.read_bytes()==png,'Existing stage08 PNG differs')
    else:
        with path.open('xb') as stream:
            stream.write(png); stream.flush(); os.fsync(stream.fileno())
    return dict(path=path.relative_to(stage).as_posix(),png_sha256=digest(png),png_bytes=len(png),size=[SIZE,SIZE],**geometry)


def protocol():
    return dict(schema=SCHEMA,model=MODEL,endpoint=api.ENDPOINT,size=SIZE,arms=list(ARMS),
        seeds=list(SEEDS),shots=[0,1,4],shared_zero_shot=True,formal_targets=600,
        instructions=INSTRUCTIONS,criteria=CRITERIA,threshold=.5,ece_bins=10,
        preprocessing='EXIF transpose; RGB; LANCZOS contain; centered white pad; PNG compress_level6',
        support_selection='Within each category sort sha256(jev-support-v1|seed|source_id) ascending then source_id; first4; 1shot nested first1',
        query_definition='Same120 approved previously viewed development queries; no new confirmation',
        isolation='All unique training sources and byte SHA disjoint from query; supports selected only from declared training normal bank',
        seed42_note='Phase1 seed42 remains historical renderer control only; new0 is independently collected for matching new pixels',
        teacher_note='Support seeds refer to original native training-normal bank, independently of supervised teacher train/val/test exposure',
        width320_included=False,workers=3,connect_timeout=60,read_timeout=120,max_started_per_target=3,
        automatic_retries=False,consecutive_error_submission_stop=3,
        bootstrap=dict(replicates=2000,seed=42,strata='category x image_label',paired=True),
        no_query_gt_in_request=True,confirmation_opened=False,independent_experiment=True)


def prepare(args):
    host_gate()
    root=args.root.resolve(); stage=api.inside(root,args.stage)
    with api.output_lock(stage):
        plan_path=root/'source_baseline/frozen/plan.json'
        config_path=root/'source_baseline/frozen/config.json'
        plan,config=read(plan_path),read(config_path)
        require(config['instructions']==INSTRUCTIONS and config['criteria']==CRITERIA and config['model']==MODEL,'Historical full-image task wording changed')
        require(config['plan_sha256']==digest(canonical(plan)),'Historical query plan identity differs')
        require(sha(args.train)==TRAIN_SHA and sha(args.audit)==AUDIT_SHA,'Training manifest or historical source audit SHA differs')
        train=api.read_rows(args.train)
        require(len(train)==1536,'Expected1536 original native training rows')
        unique={}
        for original in train:
            row={key:original[key] for key in ('source_id','image','category','is_anomalous')}
            require(type(row['is_anomalous']) is bool,'Training normal eligibility needs explicit boolean label')
            sid=row['source_id']
            require(sid not in unique or unique[sid]==row,'Conflicting duplicate training source')
            unique[sid]=row
        require(len(unique)==768 and all(v==2 for v in Counter(r['source_id'] for r in train).values()),'Expected768unique sources repeated twice')
        audit=read(args.audit)
        history={}
        for row in audit['images']:
            if row['source_id'] not in unique and row['source_id'] not in {q['source_id'] for q in plan['queries']}: continue
            require(row['source_id'] not in history or history[row['source_id']]==row,'Conflicting byte history')
            history[row['source_id']]=row
        query_ids={r['source_id'] for r in plan['queries']}
        query_sha={r['sha256'] for r in plan['queries']}
        train_sha={history[sid]['sha256'] for sid in unique}
        require(not query_ids.intersection(unique) and not query_sha.intersection(train_sha),'Train/query source or byte overlap')
        bank=[]
        for sid,row in unique.items():
            if row['is_anomalous']: continue
            previous=history[sid]
            require(sid=='visa_'+previous['sha256'] and row['image']==previous['image'],'Training byte identity differs')
            path=Path(row['image']).resolve()
            require(path.is_relative_to(args.images_root.resolve()) and path.is_file(),'Normal support outside approved image root')
            raw=path.read_bytes()
            require(digest(raw)==previous['sha256'] and len(raw)==previous['bytes'],'Training-normal original SHA/size mismatch')
            bank.append(dict(source_id=sid,category=row['category'],original_path=str(path),
                             original_sha256=previous['sha256'],original_bytes=len(raw)))
        require(len(bank)==384 and len({r['original_sha256'] for r in bank})==384,'Need384 byte-unique training normals')
        counts=Counter(r['category'] for r in bank)
        require(len(counts)==12 and set(counts.values())=={32},'Need32 training-normal candidates per category')
        selected={str(seed):select_supports(bank,seed) for seed in SEEDS}
        selected_unique={r['source_id']:r for groups in selected.values() for records in groups.values() for r in records}
        query_rows=sorted(plan['queries'],key=lambda r:r['source_id'])
        require(len(query_rows)==120 and len(query_ids)==120,'Need120 approved queries')
        require(args.limit_queries is None or 1<=args.limit_queries<=120,'Invalid separate preflight size')
        query_rows=query_rows if args.limit_queries is None else query_rows[:args.limit_queries]
        queries=[]
        all_sources={}
        for row in query_rows:
            path=(args.images_root/row['relative_image']).resolve()
            require(path.is_relative_to(args.images_root.resolve()) and path.is_file(),'Query outside approved image root')
            raw=path.read_bytes()
            require(digest(raw)==row['sha256'] and len(raw)==row['source_bytes'],'Approved query original SHA differs')
            clean=dict(source_id=row['source_id'],category=row['category'],original_path=str(path),
                       original_sha256=row['sha256'],original_bytes=len(raw))
            clean['asset']=publish_asset(stage,row['source_id'],raw)
            queries.append(clean); all_sources[row['source_id']]=clean
        support_assets={}
        for sid,row in sorted(selected_unique.items()):
            clean=dict(row); clean['asset']=publish_asset(stage,sid,Path(row['original_path']).read_bytes())
            support_assets[sid]=clean; all_sources[sid]=clean
        supports={seed:{category:[support_assets[r['source_id']] for r in records] for category,records in groups.items()}
                  for seed,groups in selected.items()}
        require(not query_ids.intersection(selected_unique),'Selected support/query identities overlap')
        require(not query_sha.intersection(r['original_sha256'] for r in selected_unique.values()),'Selected support/query bytes overlap')
        design=protocol()
        api.immutable_json(stage/'protocol.json',design)
        deps=dependencies()
        require(deps['packages']['Pillow']=='12.3.0','Stage08 renderer requires frozen Pillow12.3.0')
        contract=dict(schema=SCHEMA,mode='formal_independent_support_robustness' if args.limit_queries is None else 'separate_preflight_not_formal',
            limit_queries=args.limit_queries,root_stage=args.stage,model=MODEL,endpoint=api.ENDPOINT,size=SIZE,
            protocol_sha256=sha(stage/'protocol.json'),runner_sha256=sha(__file__),transport_source_sha256=sha(api.__file__),
            source_files={key:dict(path=str(path.resolve()),sha256=sha(path)) for key,path in
                [('query_plan',plan_path),('historical_config',config_path),('train',args.train),('source_audit',args.audit)]},
            dependencies=deps,queries=queries,supports=supports,normal_bank_counts=dict(counts),
            normal_bank_originals=sorted(bank,key=lambda r:r['source_id']),normal_bank_verified=384,
            n_unique_selected_supports=len(selected_unique),
            cross_seed_support_overlap={c:len({r['source_id'] for r in supports['43'][c]} &
                {r['source_id'] for r in supports['44'][c]}) for c in sorted(counts)},
            train_query_source_overlap=0,train_query_byte_overlap=0,confirmation_opened=False,
            phase1_renderer=config.get('versions',{}).get('pillow'),new_renderer=deps['packages']['Pillow'],
            schedule=[dict(source_id=q['source_id'],arm=arm) for q in queries for arm in ARMS])
        api.immutable_json(stage/'contract.json',contract)
        images=load_assets(stage,contract)
        completed,starts,attempts,hashes,unknown=validate_logs(stage,contract,images)
        status=state_document(contract,completed,starts,attempts,unknown,'complete' if len(completed)==len(contract['schedule']) else 'prepared')
        api.write_json(stage/'status.json',status)
        print(json.dumps(status,ensure_ascii=False),flush=True)
    return 0


def load_assets(stage,contract):
    sources={q['source_id']:q for q in contract['queries']}
    for groups in contract['supports'].values():
        for records in groups.values():
            for row in records: sources[row['source_id']]=row
    images={}
    for sid,row in sources.items():
        metadata=row['asset']
        checked,raw=api.validated_asset(stage,metadata)
        images[sid]=raw
    return images


def verify_contract(stage):
    contract=read(stage/'contract.json')
    require(contract['schema']==SCHEMA and contract['model']==MODEL and contract['endpoint']==api.ENDPOINT,'Stage08 contract schema/model mismatch')
    require(contract['runner_sha256']==sha(__file__) and contract['transport_source_sha256']==sha(api.__file__),'Frozen experiment source changed')
    require(contract['dependencies']==dependencies(),'Frozen stage08 dependencies changed')
    require(contract['protocol_sha256']==sha(stage/'protocol.json') and read(stage/'protocol.json')==protocol(),'Stage08 protocol changed')
    for metadata in contract['source_files'].values():
        require(sha(metadata['path'])==metadata['sha256'],'Frozen stage08 source manifest changed')
    return contract,load_assets(stage,contract)


def key_for(row): return row['source_id'],row['arm']


def validate_logs(stage,contract,images):
    contract_hash=digest(canonical(contract))
    queries={r['source_id']:r for r in contract['queries']}
    hashes={(q['source_id'],arm):digest(canonical(payload(q,arm,contract,images,contract_hash)))
            for q in contract['queries'] for arm in ARMS}
    completed={}
    for row in api.read_rows(stage/'predictions.jsonl'):
        pair=key_for(row)
        require(pair in hashes and pair not in completed,'Unexpected or duplicate stage08 accepted target')
        require(row['contract_sha256']==contract_hash and row['request_sha256']==hashes[pair],'Stage08 prediction identity changed')
        api.response_probability(dict(model=row['response_model'],answers={'anomaly':dict(type='noul',noul=row['probability'])}))
        require(row['category']==queries[pair[0]]['category'],'Stage08 accepted category changed')
        require(type(row['seconds']) in (int,float) and math.isfinite(row['seconds']) and row['seconds']>=0,'Invalid call seconds')
        require(api.safe_usage({'usage':row['usage']})==row['usage'],'Invalid saved usage')
        completed[pair]=row
    attempts=api.read_rows(stage/'attempts.jsonl'); starts=Counter(); intents={}; terminal=set()
    for row in attempts:
        pair=key_for(row); identity=pair+(row['attempt'],)
        require(pair in hashes and row['contract_sha256']==contract_hash and row['request_sha256']==hashes[pair],'Stage08 attempt identity changed')
        require(type(row['attempt']) is int and 1<=row['attempt']<=3,'Stage08 attempt cap exceeded')
        if row['event']=='started':
            starts[pair]+=1
            require(row['attempt']==starts[pair] and identity not in intents,'Stage08 intent sequence differs')
            intents[identity]=row
        else:
            require(row['event'] in ('failed','succeeded') and identity in intents and identity not in terminal,'Invalid stage08 terminal')
            require(row['invocation_id']==intents[identity]['invocation_id'],'Stage08 invocation identity differs')
            terminal.add(identity)
            if row['event']=='succeeded': require(pair in completed and completed[pair]['attempt']==row['attempt'],'Succeeded stage08 target missing')
    for pair,row in completed.items(): require(pair+(row['attempt'],) in intents,'Stage08 prediction lacks durable intent')
    unknown=[identity for identity in intents if identity not in terminal and identity[:2] not in completed]
    return completed,starts,attempts,hashes,unknown


def state_document(contract,completed,starts,attempts,unknown,state,reason=None):
    return dict(schema=SCHEMA,status=state,timestamp_utc=api.now(),mode=contract['mode'],
        contract_sha256=digest(canonical(contract)),expected=len(contract['schedule']),completed=len(completed),
        by_arm={arm:sum(pair[1]==arm for pair in completed) for arm in ARMS},
        started_attempts=sum(starts.values()),failed_attempts=sum(r['event']=='failed' for r in attempts),
        interrupted_unknown_attempts=len(unknown),exhausted_unaccepted=sum(starts[key_for(r)]>=3 and key_for(r) not in completed for r in contract['schedule']),
        stop_reason=reason,confirmation_opened=False,automatic_retries=False)


def run(args):
    host_gate(); root=args.root.resolve(); stage=api.inside(root,args.stage)
    with api.output_lock(stage):
        contract,images=verify_contract(stage)
        completed,starts,attempts,hashes,unknown=validate_logs(stage,contract,images)
        expected=len(contract['schedule'])
        if len(completed)==expected:
            api.write_json(stage/'status.json',state_document(contract,completed,starts,attempts,unknown,'complete')); return 0
        pending=[key_for(row) for row in contract['schedule'] if key_for(row) not in completed and starts[key_for(row)]<3
                 and (starts[key_for(row)]==0 or args.retry_failed)]
        if not pending or (stage/'STOP').exists():
            status=state_document(contract,completed,starts,attempts,unknown,'incomplete_requires_audit_or_explicit_retry',
                                  'STOP' if (stage/'STOP').exists() else 'no_eligible_targets')
            api.write_json(stage/'status.json',status); print(json.dumps(status),flush=True); return 2
        key=os.environ.get(args.key_env)
        if not key:
            require(sys.stdin.isatty(),'API key environment unset; hidden input requires TTY')
            key=getpass.getpass('ModelGo API key (hidden, never saved): ')
        require(isinstance(key,str) and key and key.strip()==key,'Missing or invalid credential')
        invocation=str(uuid.uuid4()); contract_hash=digest(canonical(contract))
        api.append_json(stage/'invocations.jsonl',dict(invocation_id=invocation,timestamp_utc=api.now(),
            retry_failed=args.retry_failed,contract_sha256=contract_hash,pending=len(pending),
            unknown_completion_retry_risk_acknowledged=bool(args.retry_failed and unknown)))
        queries={row['source_id']:row for row in contract['queries']}; futures={}; position=0; consecutive=0; stop=None
        with ThreadPoolExecutor(max_workers=3) as pool:
            while position<len(pending) or futures:
                if (stage/'STOP').exists(): stop=stop or 'STOP_file'
                while not stop and position<len(pending) and len(futures)<3:
                    pair=pending[position]; position+=1
                    body=payload(queries[pair[0]],pair[1],contract,images,contract_hash)
                    require(digest(canonical(body))==hashes[pair],'Payload changed before send')
                    starts[pair]+=1
                    intent=dict(source_id=pair[0],arm=pair[1],attempt=starts[pair],event='started',
                        timestamp_utc=api.now(),request_sha256=hashes[pair],contract_sha256=contract_hash,invocation_id=invocation)
                    api.append_json(stage/'attempts.jsonl',intent); attempts.append(intent)
                    futures[pool.submit(api.send,body,key)]=(pair,intent)
                if not futures: break
                done,_=wait(futures,timeout=.5,return_when=FIRST_COMPLETED)
                for future in done:
                    pair,intent=futures.pop(future)
                    value,error,http_status,seconds=future.result()
                    if value is not None:
                        seed,shots=arm_details(pair[1])
                        row=dict(**value,source_id=pair[0],arm=pair[1],support_seed=seed,shots=shots,
                            category=queries[pair[0]]['category'],attempt=intent['attempt'],seconds=seconds,
                            request_sha256=hashes[pair],contract_sha256=contract_hash,invocation_id=invocation,timestamp_utc=api.now())
                        api.append_json(stage/'predictions.jsonl',row); completed[pair]=row; consecutive=0
                        terminal=dict(intent,event='succeeded',timestamp_utc=api.now(),http_status=http_status,seconds=seconds)
                    else:
                        consecutive+=1
                        terminal=dict(intent,event='failed',timestamp_utc=api.now(),http_status=http_status,seconds=seconds,error=error)
                        if consecutive>=3: stop=stop or 'consecutive_errors'
                    api.append_json(stage/'attempts.jsonl',terminal); attempts.append(terminal)
                    api.write_json(stage/'status.json',state_document(contract,completed,starts,attempts,unknown,'draining_inflight' if stop else 'running',stop))
                    print(json.dumps(dict(completed=len(completed),expected=expected,arm=pair[1],http_status=http_status,error=error,seconds=seconds)),flush=True)
        completed,starts,attempts,hashes,unknown=validate_logs(stage,contract,images)
        state='complete' if len(completed)==expected else 'incomplete_requires_explicit_retry'
        status=state_document(contract,completed,starts,attempts,unknown,state,stop)
        api.write_json(stage/'status.json',status); print(json.dumps(status),flush=True)
        return 0 if state=='complete' else 2


def metrics(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    require(len(y)==len(p)>0 and np.isfinite(p).all() and np.all((p>=0)&(p<=1)) and np.isin(y,[0,1]).all(),'Invalid metric arrays')
    if len(np.unique(y))!=2: return dict(n=len(y),normal=int(sum(y==0)),anomalous=int(sum(y==1)),**{k:None for k in METRIC_NAMES})
    pred=p>=.5; bins=np.minimum((p*10).astype(int),9)
    ece=sum(np.mean(bins==b)*abs(float(y[bins==b].mean())-float(p[bins==b].mean())) for b in range(10) if np.any(bins==b))
    return dict(n=len(y),normal=int(sum(y==0)),anomalous=int(sum(y==1)),auroc=float(roc_auc_score(y,p)),
        ap=float(average_precision_score(y,p)),macro_f1=float(f1_score(y,pred,average='macro',labels=[0,1],zero_division=0)),
        recall=float(pred[y==1].mean()),fpr=float(pred[y==0].mean()),brier=float(brier_score_loss(y,p)),ece_10_equal_width=float(ece))


def weighted_metrics(y,p,weights):
    """Exact replication-weight bootstrap with grouped ties, no score jitter."""
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float); weights=np.asarray(weights,dtype=float)
    require(weights.ndim==2 and weights.shape[1]==len(y)==len(p),'Invalid bootstrap dimensions')
    positives=weights@y; negatives=weights@(1-y); total=positives+negatives
    require(np.all(positives>0) and np.all(negatives>0),'Bootstrap needs both labels')
    order=np.argsort(-p,kind='stable')
    starts=np.r_[0,np.flatnonzero(np.diff(p[order])!=0)+1]
    group_positive=np.add.reduceat(weights[:,order]*y[order][None,:],starts,axis=1)
    group_negative=np.add.reduceat(weights[:,order]*(1-y[order])[None,:],starts,axis=1)
    tp_rank=np.cumsum(group_positive,axis=1); fp_rank=np.cumsum(group_negative,axis=1)
    previous=np.c_[np.zeros(len(weights)),tp_rank[:,:-1]]
    auc=np.sum(group_negative*(tp_rank+previous)*.5,axis=1)/(positives*negatives)
    precision=np.divide(tp_rank,tp_rank+fp_rank,out=np.zeros_like(tp_rank),where=(tp_rank+fp_rank)>0)
    ap=np.sum(group_positive*precision,axis=1)/positives
    predicted=p>=.5
    tp=weights@(y*predicted); fp=weights@((1-y)*predicted)
    fn=positives-tp; tn=negatives-fp
    f1_positive=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp),where=(2*tp+fp+fn)>0)
    f1_negative=np.divide(2*tn,2*tn+fp+fn,out=np.zeros_like(tn),where=(2*tn+fp+fn)>0)
    bins=np.minimum((p*10).astype(int),9); ece=np.zeros(len(weights))
    for b in range(10):
        chosen=bins==b
        if np.any(chosen): ece+=np.abs(weights[:,chosen]@(y[chosen]-p[chosen]))/total
    return dict(auroc=auc,ap=ap,macro_f1=(f1_positive+f1_negative)*.5,
        recall=tp/positives,fpr=fp/negatives,brier=(weights@((y-p)**2))/total,ece_10_equal_width=ece)


def bootstrap(queries,y,probabilities,replicates=2000):
    strata=defaultdict(list)
    for i,q in enumerate(queries): strata[q['category'],int(y[i])].append(i)
    rng=np.random.default_rng(42); weights=np.zeros((replicates,len(y)),dtype=float)
    for _,members in sorted(strata.items()):
        weights[:,members]=rng.multinomial(len(members),np.full(len(members),1/len(members)),size=replicates)
    pooled={arm:weighted_metrics(y,p,weights) for arm,p in probabilities.items()}
    categories=sorted({q['category'] for q in queries}); category_boot={}; category_points={}
    for category in categories:
        idx=np.array([i for i,q in enumerate(queries) if q['category']==category])
        category_boot[category]={arm:weighted_metrics(y[idx],p[idx],weights[:,idx]) for arm,p in probabilities.items()}
        category_points[category]={arm:metrics(y[idx],p[idx]) for arm,p in probabilities.items()}
    points={arm:metrics(y,p) for arm,p in probabilities.items()}
    def interval(delta,values):
        return dict(delta=float(delta),ci95=np.quantile(values,[.025,.975],method='linear').tolist(),
            replicates=replicates,seed=42,strata='category x image_label',paired=True,
            includes_provider_randomness=False,includes_support_selection_variance=False)
    result={}
    for a,b in PAIRS:
        result[a+'-minus-'+b]=dict(
            pooled={k:interval(points[a][k]-points[b][k],pooled[a][k]-pooled[b][k]) for k in METRIC_NAMES},
            category_macro={k:interval(np.mean([category_points[c][a][k]-category_points[c][b][k] for c in categories]),
                np.mean([category_boot[c][a][k]-category_boot[c][b][k] for c in categories],axis=0)) for k in METRIC_NAMES})
    return result


def evaluate(args):
    host_gate(); root=args.root.resolve(); stage=api.inside(root,args.stage)
    with api.output_lock(stage):
        status=read(stage/'status.json'); contract,images=verify_contract(stage)
        require(status.get('status')=='complete' and status.get('completed')==status.get('expected')==600
                and contract['mode']=='formal_independent_support_robustness','Need complete independent600 before evaluation')
        completed,starts,attempts,hashes,unknown=validate_logs(stage,contract,images)
        require(len(completed)==600 and not unknown,'Formal600 response audit incomplete')
        predictions_sha=sha(stage/'predictions.jsonl')
        freeze=dict(schema=SCHEMA,contract_sha256=sha(stage/'contract.json'),predictions_sha256=predictions_sha,
            attempts_sha256=sha(stage/'attempts.jsonl'),runner_sha256=sha(__file__),confirmation_opened=False)
        api.immutable_json(stage/'prediction_freeze.json',freeze)
        config=read(root/'source_baseline/frozen/config.json'); labels=read(root/'source_baseline/frozen/labels.json')
        require(digest(canonical(labels))==config['labels_sha256'],'Frozen label identity changed')
        queries=contract['queries']; y=np.array([labels[q['source_id']] for q in queries],dtype=int)
        require(len(y)==120 and sum(y)==60,'Need60 normal and60 anomaly queries')
        probabilities={arm:np.array([completed[q['source_id'],arm]['probability'] for q in queries]) for arm in ARMS}
        summary={}
        for arm,p in probabilities.items():
            rows=[completed[q['source_id'],arm] for q in queries]; fields=sorted({k for r in rows for k in r['usage']})
            summary[arm]=dict(metrics=metrics(y,p),usage={k:dict(observed_sum=float(sum(r['usage'][k] for r in rows if k in r['usage'])),
                observed_calls=sum(k in r['usage'] for r in rows),missing_calls=sum(k not in r['usage'] for r in rows)) for k in fields},
                seconds=dict(mean=float(np.mean([r['seconds'] for r in rows])),median=float(np.median([r['seconds'] for r in rows])),
                    p95=float(np.quantile([r['seconds'] for r in rows],.95))))
        by_category={}
        for category in sorted({q['category'] for q in queries}):
            idx=np.array([i for i,q in enumerate(queries) if q['category']==category])
            by_category[category]={arm:metrics(y[idx],p[idx]) for arm,p in probabilities.items()}
        macro={arm:{k:float(np.mean([by_category[c][arm][k] for c in by_category])) for k in METRIC_NAMES} for arm in ARMS}
        result=dict(schema=SCHEMA,status='complete',timestamp_utc=api.now(),hostname=socket.gethostname(),prediction_freeze=freeze,
            arms=summary,by_category=by_category,category_macro=macro,paired_bootstrap=bootstrap(queries,y,probabilities),
            normal_bank_verified=contract['normal_bank_verified'],selected_supports=contract['n_unique_selected_supports'],
            cross_seed_support_overlap=contract['cross_seed_support_overlap'],
            operations=dict(accepted=600,started=sum(starts.values()),failed=sum(r['event']=='failed' for r in attempts),unknown=len(unknown)),
            limitations=['Previously viewed120 development queries, not confirmation',
                'Two fixed additional support sets are descriptive, not broad support distribution evidence',
                'Shared0 collected once; excludes provider repeat variance',
                'Bootstrap conditions on each fixed support set, not support-selection uncertainty',
                'More supports increases information/image/token budgets',
                'Original seed42 and renderer remain historical controls only',
                'Training normal bank is independent of teacher supervised split membership',
                'No320 resolution arm, currency bill or cross-domain evaluation'],confirmation_opened=False,no_gt_tuning=True)
        result_path=stage/'metrics.json'
        if result_path.exists():
            old=read(result_path); require(all(old[k]==result[k] for k in result if k!='timestamp_utc'),'Frozen support result differs'); result=old
        else: api.immutable_json(result_path,result)
        lines=['# Jev 多支持种子独立研究','','同一120开发query，重新渲染并重新采集shared0；seed43/44各1/4正常参考，共600接受判断。',
            '问题定义逐字沿用第一阶段全图Noul，448等比白填充；这是新独立实验，不是修改旧冻结结果。','',
            '| 条件 | AUROC | AP | Recall | FPR | Brier | ECE10 |','|---|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            m=summary[arm]['metrics']; lines.append('| '+arm+' | '+' | '.join(f'{m[k]:.6f}' for k in ('auroc','ap','recall','fpr','brier','ece_10_equal_width'))+' |')
        lines+=['','## 结果分析','']
        for seed in SEEDS:
            s=str(seed); d=summary['s'+s+'_4']['metrics']['auroc']-summary['shared0']['metrics']['auroc']
            lines.append(f'- seed{seed}的4参考相对新shared0 AUROC差值为{100*d:+.2f}个百分点；配对区间及其他工作点变化见metrics.json。')
        lines+=['','## 预算与局限','',f"- 600成功目标，{sum(starts.values())}次started，{sum(r['event']=='failed' for r in attempts)}次显式失败；缺失usage不视为零。",
            '- 支持从384张训练正常图中按固定种子哈希选取，每类4张，1shot为首张；与query来源和字节均隔离。',
            '- 两个支持seed不是两套独立query数据，也不是模型训练seed；正常训练bank与teacher自定义监督划分是不同概念。',
            '- 第一阶段seed42仅保留历史renderer对照，不把新旧差值归为支持seed因果。未开展320分辨率分支。',
            '- 置信区间固定支持集、不含服务随机性和支持选择方差；未做多重比较校正，不据开发结果声称部署收益。','']
        document='\n'.join(lines); report_path=stage/'REPORT.md'
        if report_path.exists(): require(report_path.read_text()==document,'Frozen support report differs')
        else:
            with report_path.open('x',encoding='utf-8') as stream: stream.write(document); stream.flush(); os.fsync(stream.fileno())
        require(sha(stage/'predictions.jsonl')==predictions_sha,'Predictions changed during evaluation')
        print(json.dumps(dict(status='complete',accepted=600,metrics=str(result_path),report=str(report_path),confirmation_opened=False)),flush=True)
    return 0


def self_test():
    host_gate()
    bank=[dict(source_id='visa_'+str(i).zfill(64),category='fixture') for i in range(8)]
    selected=select_supports(bank,43)['fixture']
    require(len(selected)==4 and selected==select_supports(bank,43)['fixture'],'Support selection unstable')
    query=dict(source_id='visa_'+'f'*64,category='fixture')
    contract=dict(supports={'43':{'fixture':selected},'44':{'fixture':select_supports(bank,44)['fixture']}})
    pixels={q['source_id']:b'synthetic' for q in bank+[query]}
    one=payload(query,'s43_1',contract,pixels,'a'*64); four=payload(query,'s43_4',contract,pixels,'a'*64)
    require(len(one['assets'])==2 and len(four['assets'])==5 and len(one['state']['normal_references'])==1,'Reference budget failed')
    require(one['questions']==four['questions'] and one['questions']['anomaly']['instructions']==INSTRUCTIONS,'Full-image question changed')
    y=np.array([0,0,1,1]); p=np.array([0.,.1,.8,1.])
    m=metrics(y,p); require(m['auroc']==1. and m['recall']==1. and m['fpr']==0.,'Metric sanity failed')
    tied=np.array([.2,.5,.5,1.]); weights=np.array([[1,1,1,1],[2,0,1,1]],dtype=float)
    weighted=weighted_metrics(y,tied,weights)
    for j,counts in enumerate(weights.astype(int)):
        idx=np.repeat(np.arange(4),counts); reference=metrics(y[idx],tied[idx])
        require(all(math.isclose(weighted[k][j],reference[k],rel_tol=0,abs_tol=1e-12) for k in METRIC_NAMES),'Weighted bootstrap/tie metrics differ')
    raw=io.BytesIO(); Image.new('RGB',(200,100),'white').save(raw,format='PNG')
    png,geometry=render(raw.getvalue())
    with Image.open(io.BytesIO(png)) as image: require(image.size==(448,448),'Render size failed')
    require(geometry['resized_size']==[448,224] and geometry['padding']==[0,112],'Padding geometry failed')
    pair=(query['source_id'],'shared0')
    require(not api.eligible(pair,{pair:{}},Counter(),True),'Accepted target retry guard failed')
    require(not api.eligible(pair,{},Counter({pair:3}),True),'Retry budget guard failed')
    print(json.dumps(dict(status='passed',synthetic_only=True,network_calls=0,checks=['deterministic_supports','nested1of4',
        '2and5assets','unchanged_question','wholeimage_metrics','448whitepad','skip_accepted','attempt_cap'])))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    for name,func in [('prepare',prepare),('run',run),('evaluate',evaluate)]:
        item=sub.add_parser(name); item.add_argument('--root',type=Path,required=True); item.add_argument('--stage',default='stage08_support')
        if name=='prepare':
            item.add_argument('--train',type=Path,default=DEFAULT_DATA/'prepared/multitask_v2/train_decision_decision.jsonl')
            item.add_argument('--audit',type=Path,default=DEFAULT_DATA/'experiments/multitask_v2/source_byte_audit.json')
            item.add_argument('--images-root',type=Path,default=DEFAULT_DATA/'datasets/VisA'); item.add_argument('--limit-queries',type=int)
        if name=='run': item.add_argument('--retry-failed',action='store_true'); item.add_argument('--key-env',default='MODELGO_API_KEY')
        item.set_defaults(func=func)
    item=sub.add_parser('self-test'); item.set_defaults(func=lambda args:self_test())
    args=parser.parse_args()
    try: return args.func(args)
    except Exception as error:
        print(json.dumps(dict(status='blocked',exception_class=type(error).__name__,
            reason=str(error) if isinstance(error,ValueError) else 'Inspect runtime and frozen inputs; arbitrary exception suppressed')),file=sys.stderr); return 2


if __name__=='__main__': sys.exit(main())
