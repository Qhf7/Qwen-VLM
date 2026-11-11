#!/usr/bin/env python3
"""Server-only real Noul crop gate and newly collected downstream pilot."""
from __future__ import annotations
import argparse,base64,getpass,hashlib,io,json,math,os,socket,sys,tempfile,time,uuid
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import requests
from requests.adapters import HTTPAdapter

OLD=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111')
sys.path.insert(0,str(OLD/'code'))
import roi_api as api
import evaluate_roi as evaluation
import support_robustness as support
HOST='autodl-container-132444827f-9c594038'
SCHEMA='jev-live-noul-crop-gate-pilot-v1'
GATE_INSTRUCTIONS=('Would inspecting an enlarged close-up of the frozen candidate region provide useful additional visible evidence '
    'for deciding whether that candidate region contains an industrial defect? query is the original448 image. '
    'candidate_bbox_in_query is [left,top,right,bottom], normalized0to1, x rightward and y downward. '
    'Only the original image is available at this stage; the proposed close-up is sampled from the recorded same-image bounds. '
    'Judge need for a closer view, not whether an anomaly is present. Use visible ambiguity, small details or uncertain texture. '
    'An unchanged full-image duplicate gives no additional detail. Do not invent unseen defects.')
GATE_CRITERIA={'true':'An enlarged close-up of the recorded candidate would provide useful additional visual evidence.',
    'false':'The recorded close-up is unnecessary or adds no useful visible evidence.'}
MODEL='jev-1.14.0'
MAX_STARTS=216
MAX_FAILURES=30
_SESSIONS=__import__('threading').local()


def req(v,m):
    if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return api.file_sha(Path(p))
def canon(v):return api.canonical(v)
def digest(v):return api.digest(v)
def now():return api.now()
def neutral(sid):return digest(('jev-live-route-pilot-neutral-v1|'+sid).encode())
def key(row):return row['source_id'],row['arm']
def host():req(socket.gethostname()==HOST,'Execute pilot only on designated server')
def put(p,v,text=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    raw=v.encode() if text else json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False).encode()+b'\n'
    if p.exists():req(p.read_bytes()==raw,'Frozen pilot artifact differs: '+p.name);return
    temp=p.with_name(p.name+'.'+str(uuid.uuid4())+'.tmp')
    with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    try:os.link(temp,p)
    finally:temp.unlink(missing_ok=True)
def rows_text(rows):return ''.join(json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n' for r in rows)


def protocol():
    return dict(schema=SCHEMA,model=MODEL,endpoint=api.ENDPOINT,n_queries=24,
        selection='Sort original120 by sha256(source_id UTF8), then source_id; first24; no label selection',
        eligibility='Candidate crop PNG SHA differs from query PNG SHA; no substitute samples; area/fullimage recorded',
        common_K='min(8,eligible_count)',gate_question=GATE_INSTRUCTIONS,gate_criteria=GATE_CRITERIA,
        gate_inputs='Original448RGB plus fixed candidatebbox and category; no cropRGB,teacher map,label,oldprobability',
        policies=dict(learned_gate='Highest crop_needed Noul among eligible; neutralhash tie',
            matched_random='sha256(jev-live-route-random-v1|20261111|source_id) ascending among eligible',
            historical_uncertainty='Original stage04 duplicate abs(p-.5) ascending among eligible; neutralhash tie',
            geometry_small='Original normalized candidate area ascending among eligible; neutralhash tie'),
        final_requests='New duplicate for all24 and new ROI for union of4 policy sets; no cached downstream outcomes',
        anomaly_question=api.INSTRUCTIONS,anomaly_criteria=api.CRITERIA,threshold=.5,
        sequence='All gate24 first; freeze gate/route/actual crop hashes; final newdups/ROI interleaved by sourcehash and armhash',
        successful_targets_upper_bound=72,max_started_global=MAX_STARTS,max_started_per_target=3,
        workers=3,failed_submission_stop_threshold=MAX_FAILURES,inflight_failed_drain_bound=MAX_FAILURES+2,
        consecutive_failure_stop=3,unknown_completion_hardstop=True,explicit_newinvocation_retry=True,no_automatic_http_retry=True,
        evaluation='Exploratory previously viewed24; all strategies share same newly collected potential outcome table; gate24cost extra explicit',
        comparison='Each strategy vs newduplicate24; only selectedsubset has newly measured pairedROI potential outcomes',
        post_evaluation='After all newpredictions freeze: read only approved24 historical development labels and stage05 frozen candidateIoU, no new mask reads',
        bootstrap=dict(replicates=2000,seed=42,strata='image_label only',fixed_routes=True,
            category='Descriptive only; do not use categoryxlabel singleton bootstrap',
            limitation='Conditions on fixed24/routes/provider outcomes; excludes route selection/model/provider randomness'),
        masks_or_new_GT_opened=False,stage13_data_read=False,no_new_training=True,
        api_capability='Only already verified gateway image Noul; no Choice Score or Box request',
        limitation='Oldduplicate uncertainty has earlier collection time; selected24 may be unbalanced; small strata and provider timing not causal proof')


def prepare(a):
    host();root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    with api.output_lock(root):
        oldstage=OLD/'runs/modelgo_roi';old=read(oldstage/'contract.json')
        req(old['mode']=='formal_development_candidate_review' and len(old['records'])==120,'Original approved120 ROI contract required')
        req(old['runner_sha256']==sha(api.__file__),'Frozen ROI source differs')
        selected=sorted(old['records'],key=lambda r:(digest(r['source_id'].encode()),r['source_id']))[:24]
        previous={key(r):r for r in api.read_rows(oldstage/'predictions.jsonl')}
        req(len(previous)==240,'Complete old240 needed only for fixed uncertainty control')
        sources=read(OLD/'stages/03_candidates.json')['records'];source={r['source_id']:r for r in sources}
        records=[]
        for r in selected:
            sid=r['source_id'];candidate=source[sid]
            req(r['bbox_original_norm']==candidate['bbox_original_norm'] and r['bbox_448_norm']==candidate['bbox_448_norm'],'Candidate source geometry differs')
            for which in ('query','crop'):api.validated_asset(OLD,r['assets'][which])
            b=r['bbox_original_norm'];area=(b[2]-b[0])*(b[3]-b[1])
            records.append(dict(source_id=sid,category=r['category'],teacher_split=r['teacher_split'],
                bbox_original_norm=b,bbox_448_norm=r['bbox_448_norm'],assets=r['assets'],
                eligible=r['assets']['query']['png_sha256']!=r['assets']['crop']['png_sha256'],area=area,
                full_image_candidate=r['full_image_candidate'],crop_bounds_original=candidate['crop_bounds_original'],
                original_sha256=candidate['original_sha256'],original_size=candidate['original_size'],
                historical_duplicate_probability=previous[sid,'duplicate']['probability']))
        p=protocol();put(root/'protocol.json',p)
        contract=dict(schema=SCHEMA,status='input_protocol_frozen',model=MODEL,records=records,
            selected_source_order=[r['source_id'] for r in records],eligible_count=sum(r['eligible'] for r in records),
            K=min(8,sum(r['eligible'] for r in records)),protocol_sha256=sha(root/'protocol.json'),
            source_files={name:sha(path) for name,path in [('oldcontract',oldstage/'contract.json'),('oldpredictions',oldstage/'predictions.jsonl'),
                ('oldcandidates',OLD/'stages/03_candidates.json'),('old_source_plan',OLD/'source_baseline/frozen/plan.json'),
                ('old_source_audit',OLD/'stages/01_source_audit.json'),('old_evaluation',OLD/'stages/05_evaluation.json')]},
            runner_sha256=sha(__file__),roi_source_sha256=sha(api.__file__),evaluation_source_sha256=sha(evaluation.__file__),
            support_source_sha256=sha(support.__file__),dependencies=support.dependencies(),no_label_or_mask_reads=True,stage13_data_read=False)
        put(root/'contract.json',contract)
        gate=dict(schema=SCHEMA,phase='gate',parent_contract_sha256=sha(root/'contract.json'),
            schedule=[dict(source_id=r['source_id'],arm='gate') for r in records])
        put(root/'gate/contract.json',gate)
        images=load_images(contract);hashes=request_hashes(contract,gate,images)
        put(root/'gate/request_hashes.json',{'|'.join(k):v for k,v in hashes.items()})
        print(json.dumps(dict(status='prepared',n=24,eligible=contract['eligible_count'],K=contract['K'],
            max_successes=48+contract['eligible_count'],contract_sha256=sha(root/'contract.json'),no_label_or_GT_reads=True)),flush=True)
    return 0


def verify(root):
    host();c=read(root/'contract.json');req(c['schema']==SCHEMA and c['runner_sha256']==sha(__file__),'Pilot source changed')
    req(c['roi_source_sha256']==sha(api.__file__) and c['evaluation_source_sha256']==sha(evaluation.__file__)
        and c['support_source_sha256']==sha(support.__file__) and c['dependencies']==support.dependencies(),'Pilot dependency/source changed')
    req(read(root/'protocol.json')==protocol() and c['protocol_sha256']==sha(root/'protocol.json'),'Pilot protocol changed')
    for name,path in [('oldcontract',OLD/'runs/modelgo_roi/contract.json'),('oldpredictions',OLD/'runs/modelgo_roi/predictions.jsonl'),
        ('oldcandidates',OLD/'stages/03_candidates.json'),('old_source_plan',OLD/'source_baseline/frozen/plan.json'),
        ('old_source_audit',OLD/'stages/01_source_audit.json'),('old_evaluation',OLD/'stages/05_evaluation.json')]:
        req(c['source_files'][name]==sha(path),'Original pilot source changed')
    return c,load_images(c)


def load_images(c):
    result={}
    for r in c['records']:
        result[r['source_id']]={}
        for name in ('query','crop'):
            _,raw=api.validated_asset(OLD,r['assets'][name]);result[r['source_id']][name]=raw
    return result


def payload(row,arm,images,contract_hash):
    sid=row['source_id']
    if arm=='gate':
        reference=str(uuid.uuid5(uuid.NAMESPACE_URL,contract_hash+'|'+sid+'|gate|query'))
        result=dict(model=MODEL,state=dict(query=reference,category=row['category'].replace('_',' '),
            candidate_bbox_in_query=row['bbox_448_norm'],candidate_bbox_in_original=row['bbox_original_norm']),
            assets={reference:dict(type='image',format='png',data_base64=base64.b64encode(images[sid]['query']).decode())},
            questions={'crop_needed':dict(type='noul',instructions=GATE_INSTRUCTIONS,criteria=GATE_CRITERIA)})
        validate_gate_payload(result);return result
    return api.request_payload(row,arm,images[sid],contract_hash)


def validate_gate_payload(body):
    req(set(body)=={'model','state','assets','questions'} and body['model']==MODEL,'Gate envelope differs')
    state=body['state']
    req(set(state)=={'query','category','candidate_bbox_in_query','candidate_bbox_in_original'},'Forbidden gate state key')
    req(len(body['assets'])==1 and state['query'] in body['assets'],'Gate must contain only onequery image')
    req(state['category'] in {'candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4','pipe fryum'},
        'Gate category outside declared taxonomy')
    for name in ('candidate_bbox_in_query','candidate_bbox_in_original'):api.validate_box(state[name])
    req(set(body['questions'])=={'crop_needed'} and body['questions']['crop_needed']==dict(type='noul',instructions=GATE_INSTRUCTIONS,criteria=GATE_CRITERIA),
        'Gate question changed')
    encoded=canon(state).decode().lower()
    req(not any(token in encoded for token in ('mask','ground_truth','anomalous','normal/','anomaly/','source_id','oldprobability','grade','label')),
        'Privileged path/grade/mask/label content in gate state')


def request_hashes(c,phase,images):
    ch=digest(canon(phase));lookup={r['source_id']:r for r in c['records']}
    return {key(r):digest(canon(payload(lookup[r['source_id']],r['arm'],images,ch))) for r in phase['schedule']}


def response_probability(data,question):
    req(isinstance(data,dict) and data.get('model')==MODEL and isinstance(data.get('answers'),dict)
        and set(data['answers'])=={question},'Wrong gateway model/question identity')
    answer=data['answers'][question];p=answer.get('noul') if isinstance(answer,dict) else None
    req(isinstance(answer,dict) and answer.get('type')=='noul' and type(p) in (int,float) and math.isfinite(p) and 0<=p<=1,
        'Invalid typedNoul probability')
    return float(p)


def send(body,credential):
    begin=time.perf_counter();status=None
    session=getattr(_SESSIONS,'value',None)
    if session is None:session=requests.Session();session.mount('https://',HTTPAdapter(max_retries=0));_SESSIONS.value=session
    question=next(iter(body['questions']))
    try:
        response=session.post(api.ENDPOINT,json=body,headers={'Authorization':'Bearer '+credential,'TypeSafe-Danger-Zone-Images':'true'},
            timeout=(60,120),allow_redirects=False);status=int(response.status_code)
        if status!=200:return None,'http_'+str(status),status,time.perf_counter()-begin
        data=response.json();p=response_probability(data,question)
        value=dict(probability=p,response_model=MODEL,question=question,usage=api.safe_usage(data),
            request_id=api.safe_request_id(response.headers.get('x-request-id') or response.headers.get('request-id') or data.get('id'),credential))
        return value,None,status,time.perf_counter()-begin
    except requests.Timeout:error='timeout_unknown_server_completion'
    except requests.RequestException:error='transport_unknown_server_completion'
    except (ValueError,TypeError,KeyError):error='invalid_typed_response_unknown_server_completion'
    return None,error,status,time.perf_counter()-begin


def settled(row):return type(row.get('http_status')) is int and row['http_status']!=200 and row.get('error')=='http_'+str(row['http_status'])


def audit_phase(root,c,phase,images,prior_started=0,prior_failed=0):
    folder=root/phase['phase'];hashes=request_hashes(c,phase,images);ch=digest(canon(phase))
    req(read(folder/'request_hashes.json')=={'|'.join(k):v for k,v in hashes.items()},'Frozen request hashes changed')
    attempts=api.read_rows(folder/'attempts.jsonl');predictions=api.read_rows(folder/'predictions.jsonl');starts=Counter();intents={};terminals={}
    invs_list=api.read_rows(folder/'invocations.jsonl');invs={r['invocation_id']:r for r in invs_list}
    req(len(invs)==len(invs_list),'Duplicate invocation')
    for r in invs_list:
        req(r['contract_sha256']==ch and r['explicit_invocation'] is True,'Invocation contract/explicit flag differs')
    reserved=0;failures=0;consecutive=0;current=None;blocked=set();ambiguous=False;seen=set()
    for r in attempts:
        k=key(r);identity=k+(r['attempt'],)
        req(k in hashes and r['request_sha256']==hashes[k] and r['contract_sha256']==ch and r['invocation_id'] in invs,'Ledger request identity differs')
        req(type(r['attempt']) is int and 1<=r['attempt']<=3,'Target cap3 exceeded')
        if r['event']=='started':
            if current!=r['invocation_id']:
                req(reserved==0 and not ambiguous and r['invocation_id'] not in seen,'New invocation with unsettled/ambiguous prior request')
                current=r['invocation_id'];seen.add(current);consecutive=0
                req(invs[current]['prior_global_started']==prior_started+sum(starts.values())
                    and invs[current]['prior_global_failed']==prior_failed+failures,'Invocation persistent prior counters differ')
            req(not ambiguous and current not in blocked and reserved<3,'Submit after unknown/circuit or rolling3 exceeded')
            req(prior_started+sum(starts.values())<MAX_STARTS and prior_failed+failures<MAX_FAILURES,'Start after global budget/failure limit')
            if starts[k]:
                prev=terminals.get(k+(starts[k],));req(prev is not None and prev['event']=='failed' and settled(prev),'Unknown or accepted retry forbidden')
                req(prev['invocation_id']!=current and invs[current]['retry_failed'] is True,'Retry needs new explicit invocation')
            starts[k]+=1;req(r['attempt']==starts[k] and identity not in intents,'Intent sequence differs');intents[identity]=r;reserved+=1
        else:
            req(r['event'] in ('succeeded','failed') and identity in intents and identity not in terminals,'Invalid terminal')
            req(r['invocation_id']==intents[identity]['invocation_id'],'Terminal invocation differs');terminals[identity]=r;reserved-=1
            if r['event']=='failed':
                failures+=1;consecutive+=1
                if not settled(r):ambiguous=True
            else:consecutive=0
            if consecutive>=3:blocked.add(r['invocation_id'])
    complete={}
    for r in predictions:
        k=key(r);identity=k+(r['attempt'],)
        req(k in hashes and k not in complete and identity in intents and starts[k]==r['attempt'],'Duplicate/unintended/retried accepted target')
        req(r['request_sha256']==hashes[k] and r['contract_sha256']==ch and r['invocation_id']==intents[identity]['invocation_id'],'Accepted identity differs')
        req(identity not in terminals or terminals[identity]['event']=='succeeded','Accepted/failed terminal conflict')
        response_probability(dict(model=r['response_model'],answers={r['question']:dict(type='noul',noul=r['probability'])}),
            'crop_needed' if k[1]=='gate' else 'anomaly')
        req(api.safe_usage({'usage':r['usage']})==r['usage'],'Invalid saved usage');complete[k]=r
        req(type(r['seconds']) in (int,float) and math.isfinite(r['seconds']) and r['seconds']>=0,'Invalid saved call duration')
    for identity,r in terminals.items():
        if r['event']=='succeeded':req(identity[:2] in complete and complete[identity[:2]]['attempt']==identity[2],'Succeeded response absent')
    unknown=[i for i in intents if i not in terminals or (terminals[i]['event']=='failed' and not settled(terminals[i]))]
    return complete,starts,attempts,hashes,unknown


def phase_data(root,name,c,images):
    phase=read(root/name/'contract.json');req(phase['phase']==name and phase['parent_contract_sha256']==sha(root/'contract.json'),'Phase contract differs')
    req(len({key(r) for r in phase['schedule']})==len(phase['schedule']),'Duplicate phase target')
    if name=='gate':
        req(phase['schedule']==[dict(source_id=r['source_id'],arm='gate') for r in c['records']],'Gate schedule differs')
        return phase,audit_phase(root,c,phase,images)
    gate,values=phase_data(root,'gate',c,images);gc,gs,ga,gh,gu=values
    req(len(gc)==24 and not gu,'Finalphase before settledgate24')
    routing=read(root/'routing.json');freeze=read(root/'gate/prediction_freeze.json')
    req(freeze['contract_sha256']==sha(root/'gate/contract.json') and freeze['predictions_sha256']==sha(root/'gate/predictions.jsonl')
        and freeze['attempts_sha256']==sha(root/'gate/attempts.jsonl'),'Frozen gate evidence changed')
    req(phase['routing_sha256']==sha(root/'routing.json') and phase['gate_prediction_freeze_sha256']==sha(root/'gate/prediction_freeze.json')
        ==routing['gate_prediction_freeze_sha256'],'Routing evidence chain differs')
    expected=choose_routes(c,gc)
    req(routing['routes']==expected and routing['union_roi']==sorted(set().union(*map(set,expected.values()))),'Frozen policy/union differs')
    req(phase['schedule']==final_schedule(c,routing['union_roi']),'Frozen finalschedule differs')
    for execution in routing['crop_executions']:
        sid=execution['source_id'];path=api.inside(root,execution['executed_crop_path']);raw=path.read_bytes()
        req(digest(raw)==execution['png_sha256'] and len(raw)==execution['png_bytes'] and raw==images[sid]['crop'],'Executedcrop identity changed')
        images[sid]['crop']=raw
    req({r['source_id'] for r in routing['crop_executions']}==set(routing['union_roi']),'Executedcrop set differs')
    return phase,audit_phase(root,c,phase,images,sum(gs.values()),sum(r['event']=='failed' for r in ga))


def global_counts(root,c,images):
    started=failed=unknown=0
    for name in ('gate','final'):
        if not (root/name/'contract.json').exists():continue
        _,(_,counts,attempts,_,unresolved)=phase_data(root,name,c,images)
        started+=sum(counts.values());failed+=sum(r['event']=='failed' for r in attempts);unknown+=len(unresolved)
    req(started<=MAX_STARTS and failed<=MAX_FAILURES+2,'Persistent global pilot budget exceeded')
    return started,failed,unknown


def status_doc(root,c,phase,complete,starts,attempts,unknown,status,reason=None):
    total_started,total_failed,total_unknown=global_counts(root,c,load_images(c))
    return dict(schema=SCHEMA,status=status,timestamp_utc=now(),phase=phase['phase'],expected=len(phase['schedule']),completed=len(complete),
        by_arm=dict(Counter(k[1] for k in complete)),contract_sha256=digest(canon(phase)),
        global_started=total_started,global_failed=total_failed,global_unknown=total_unknown,
        phase_unknown=len(unknown),stop_reason=reason,max_started_global=MAX_STARTS,max_started_per_target=3,
        failure_submission_stop_threshold=MAX_FAILURES,inflight_drain_failure_bound=MAX_FAILURES+2,no_auto_retry=True,stage13_data_read=False)


def dry(a):
    root=a.root.resolve();c,images=verify(root);phase,values=phase_data(root,a.phase,c,images)
    print(json.dumps(status_doc(root,c,phase,*values[:3],values[4],'dry_run_passed')),flush=True);return 0


def run(a):
    root=a.root.resolve()
    with api.output_lock(root):
        c,images=verify(root);phase,(complete,starts,attempts,hashes,unknown)=phase_data(root,a.phase,c,images)
        folder=root/a.phase;allstarted,allfailed,allunknown=global_counts(root,c,images)
        req(not allunknown,'Unknown/missing terminal hardstop; do not resend')
        if len(complete)==len(phase['schedule']):
            api.write_json(folder/'status.json',status_doc(root,c,phase,complete,starts,attempts,unknown,'complete'));return 0
        pending=[key(r) for r in phase['schedule'] if key(r) not in complete and starts[key(r)]<3 and (starts[key(r)]==0 or a.retry_failed)]
        if allfailed>=MAX_FAILURES or allstarted>=MAX_STARTS or not pending or (root/'STOP').exists():
            api.write_json(folder/'status.json',status_doc(root,c,phase,complete,starts,attempts,unknown,'incomplete_requires_audit','budget_STOP_or_noeligible'));return 2
        credential=os.environ.get(a.key_env)
        if not credential:
            req(sys.stdin.isatty(),'Credential environment unset; hidden input needsTTY');credential=getpass.getpass('ModelGo key (hidden, never saved): ')
        req(isinstance(credential,str) and credential and credential.strip()==credential,'Missing credential')
        invocation=str(uuid.uuid4());ch=digest(canon(phase));lookup={r['source_id']:r for r in c['records']}
        api.append_json(folder/'invocations.jsonl',dict(invocation_id=invocation,timestamp_utc=now(),retry_failed=a.retry_failed,
            explicit_invocation=True,contract_sha256=ch,prior_global_started=allstarted,prior_global_failed=allfailed))
        futures={};position=0;consecutive=0;stop=None
        with ThreadPoolExecutor(max_workers=3) as pool:
            while position<len(pending) or futures:
                if (root/'STOP').exists():stop=stop or 'STOP'
                while not stop and position<len(pending) and len(futures)<3:
                    if allstarted>=MAX_STARTS or allfailed>=MAX_FAILURES:stop='persistent_budget';break
                    k=pending[position];position+=1;body=payload(lookup[k[0]],k[1],images,ch)
                    req(digest(canon(body))==hashes[k],'Request changed before submit');starts[k]+=1;allstarted+=1
                    intent=dict(source_id=k[0],arm=k[1],attempt=starts[k],event='started',timestamp_utc=now(),invocation_id=invocation,
                        contract_sha256=ch,request_sha256=hashes[k])
                    api.append_json(folder/'attempts.jsonl',intent);attempts.append(intent);futures[pool.submit(send,body,credential)]=(k,intent)
                if not futures:break
                done,_=wait(futures,timeout=.5,return_when=FIRST_COMPLETED)
                for f in done:
                    k,intent=futures.pop(f);value,error,http_status,seconds=f.result()
                    if value is not None:
                        row=dict(**value,source_id=k[0],arm=k[1],attempt=intent['attempt'],timestamp_utc=now(),invocation_id=invocation,
                            contract_sha256=ch,request_sha256=hashes[k],seconds=seconds)
                        api.append_json(folder/'predictions.jsonl',row);complete[k]=row;consecutive=0
                        terminal=dict(intent,event='succeeded',timestamp_utc=now(),http_status=http_status,seconds=seconds)
                    else:
                        allfailed+=1;consecutive+=1;terminal=dict(intent,event='failed',timestamp_utc=now(),http_status=http_status,seconds=seconds,error=error)
                        if consecutive>=3:stop=stop or 'consecutive_errors'
                        if allfailed>=MAX_FAILURES:stop=stop or 'persistent_failure_lock'
                        if not settled(terminal):stop=stop or 'unknown_completion'
                    api.append_json(folder/'attempts.jsonl',terminal);attempts.append(terminal)
                    api.write_json(folder/'status.json',status_doc(root,c,phase,complete,starts,attempts,unknown,'draining' if stop else 'running',stop))
                    print(json.dumps(dict(phase=a.phase,completed=len(complete),expected=len(phase['schedule']),http_status=http_status,error=error,
                        global_started=allstarted,global_failed=allfailed)),flush=True)
        phase,(complete,starts,attempts,hashes,unknown)=phase_data(root,a.phase,c,images)
        status=status_doc(root,c,phase,complete,starts,attempts,unknown,'complete' if len(complete)==len(phase['schedule']) else 'incomplete_requires_audit_or_explicit_retry',stop)
        api.write_json(folder/'status.json',status);print(json.dumps(status),flush=True);return 0 if status['status']=='complete' else 2


def freeze_routes(a):
    host();root=a.root.resolve()
    with api.output_lock(root):
        c,images=verify(root);phase,(complete,starts,attempts,hashes,unknown)=phase_data(root,'gate',c,images)
        req(len(complete)==24 and not unknown,'Need complete settledgate24 before route selection')
        put(root/'gate/prediction_freeze.json',dict(schema=SCHEMA,contract_sha256=sha(root/'gate/contract.json'),
            predictions_sha256=sha(root/'gate/predictions.jsonl'),attempts_sha256=sha(root/'gate/attempts.jsonl'),no_GT_reads=True))
        eligible=[r for r in c['records'] if r['eligible']];K=c['K'];routes=choose_routes(c,complete)
        union=sorted(set().union(*map(set,routes.values())));req(len(union)<=24 and all(len(v)==K for v in routes.values()),'CommonK/union budget differs')
        # Real crop execution regenerates only union-selected crops from approved originals.
        plan=read(OLD/'source_baseline/frozen/plan.json');by_id={r['source_id']:r for r in plan['queries']}
        audit=read(OLD/'stages/01_source_audit.json');base={r['source_id']:r for r in audit['records'] if r['role']=='query'}
        lookup={r['source_id']:r for r in c['records']};executions=[]
        for sid in union:
            row=lookup[sid];original=base[sid];path=Path(original['path']) if 'path' in original else None
            if path is None or not path.is_file():
                relative=by_id[sid]['relative_image'];path=Path('/root/autodl-tmp/industrial_vlm/datasets/VisA')/relative
            raw=path.read_bytes();req(digest(raw)==row['original_sha256'],'Crop original SHA differs')
            with Image.open(io.BytesIO(raw)) as opened:
                req(int(opened.getexif().get(274,1))==1,'Unexpected EXIF crop transform')
                image=ImageOps.exif_transpose(opened).convert('RGB');req(list(image.size)==row['original_size'],'Original geometry changed')
                cropped=image.crop(tuple(row['crop_bounds_original']));rendered,_=support.render(image_bytes(cropped))
            req(rendered==images[sid]['crop'],'Executed crop bytes differ from fixed candidate PNG')
            destination=root/'executed_crops'/(sid+'.png');destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():req(destination.read_bytes()==rendered,'Executed crop output changed')
            else:
                with destination.open('xb') as f:f.write(rendered);f.flush();os.fsync(f.fileno())
            executions.append(dict(source_id=sid,original_sha256=digest(raw),crop_bounds_original=row['crop_bounds_original'],
                executed_crop_path=destination.relative_to(root).as_posix(),png_sha256=digest(rendered),png_bytes=len(rendered),
                original_opened_after_gatefreeze=True,GT_or_mask_opened=False))
        routing=dict(schema=SCHEMA,status='route_frozen_before_final_requests',parent_contract_sha256=sha(root/'contract.json'),
            gate_prediction_freeze_sha256=sha(root/'gate/prediction_freeze.json'),K=K,eligible_count=len(eligible),routes=routes,
            union_roi=union,crop_executions=executions,no_GT_or_label_reads=True,stage13_data_read=False)
        put(root/'routing.json',routing)
        schedule=final_schedule(c,union)
        final=dict(schema=SCHEMA,phase='final',parent_contract_sha256=sha(root/'contract.json'),routing_sha256=sha(root/'routing.json'),
            expected=len(schedule),schedule=schedule,gate_prediction_freeze_sha256=sha(root/'gate/prediction_freeze.json'))
        put(root/'final/contract.json',final);put(root/'final/request_hashes.json',{'|'.join(k):v for k,v in request_hashes(c,final,images).items()})
        print(json.dumps(dict(status='routes_frozen',K=K,eligible=len(eligible),unionROI=len(union),newfinal=len(schedule),
            total_successes=24+len(schedule),routing_sha256=sha(root/'routing.json'),no_GT_or_label_reads=True)),flush=True)
    return 0


def image_bytes(image):
    output=io.BytesIO();image.save(output,format='PNG',compress_level=6);return output.getvalue()


def choose_routes(c,complete):
    eligible=[r for r in c['records'] if r['eligible']];K=c['K']
    return dict(learned_gate=[r['source_id'] for r in sorted(eligible,key=lambda r:(-complete[r['source_id'],'gate']['probability'],neutral(r['source_id']),r['source_id']))[:K]],
        matched_random=[r['source_id'] for r in sorted(eligible,key=lambda r:(digest(('jev-live-route-random-v1|20261111|'+r['source_id']).encode()),r['source_id']))[:K]],
        historical_uncertainty=[r['source_id'] for r in sorted(eligible,key=lambda r:(abs(r['historical_duplicate_probability']-.5),neutral(r['source_id']),r['source_id']))[:K]],
        geometry_small=[r['source_id'] for r in sorted(eligible,key=lambda r:(r['area'],neutral(r['source_id']),r['source_id']))[:K]])


def final_schedule(c,union):
    schedule=[]
    for row in sorted(c['records'],key=lambda r:(neutral(r['source_id']),r['source_id'])):
        arms=['duplicate']+(['roi'] if row['source_id'] in union else [])
        for arm in sorted(arms,key=lambda arm:digest((neutral(row['source_id'])+'|'+arm).encode())):schedule.append(dict(source_id=row['source_id'],arm=arm))
    return schedule


def usage_and_seconds(rows):
    fields=sorted({k for r in rows for k in r['usage']})
    return dict(successful_calls=len(rows),successful_seconds_sum=float(sum(r['seconds'] for r in rows)),
        successful_seconds_note='Sum of individual request durations; not concurrent wallclock or independent policy deployment timing',
        usage={k:dict(observed_sum=float(sum(r['usage'][k] for r in rows if k in r['usage'])),
            observed_calls=sum(k in r['usage'] for r in rows),missing_calls=sum(k not in r['usage'] for r in rows)) for k in fields})


def evaluate(a):
    root=a.root.resolve()
    with api.output_lock(root):
        c,images=verify(root);gate,(g,gs,ga,gh,gu)=phase_data(root,'gate',c,images)
        final,(f,fs,fa,fh,fu)=phase_data(root,'final',c,images)
        req(len(g)==24 and len(f)==len(final['schedule']) and not gu and not fu,'Complete gate and newfinal required before approvedGT')
        frozen=dict(schema=SCHEMA,parent_contract_sha256=sha(root/'contract.json'),routing_sha256=sha(root/'routing.json'),
            protocol_sha256=sha(root/'protocol.json'),gate_request_hashes_sha256=sha(root/'gate/request_hashes.json'),
            final_request_hashes_sha256=sha(root/'final/request_hashes.json'),
            gate_contract_sha256=sha(root/'gate/contract.json'),gate_predictions_sha256=sha(root/'gate/predictions.jsonl'),
            gate_attempts_sha256=sha(root/'gate/attempts.jsonl'),final_contract_sha256=sha(root/'final/contract.json'),
            final_predictions_sha256=sha(root/'final/predictions.jsonl'),final_attempts_sha256=sha(root/'final/attempts.jsonl'),
            runner_sha256=sha(__file__),predictions_before_pilot_GT=True,stage13_data_read=False)
        put(root/'prediction_freeze.json',frozen)
        # Approved previously seen development targets only; no new mask/13 reads.
        labels=read(OLD/'source_baseline/frozen/labels.json');config=read(OLD/'source_baseline/frozen/config.json')
        req(digest(canon(labels))==config['labels_sha256'],'Approved historical labels changed')
        oldeval=read(OLD/'stages/05_evaluation.json')
        target_rows=oldeval.get('records',oldeval.get('per_query',[]))
        req(isinstance(target_rows,list) and target_rows,'Frozen stage05 per-source records required')
        targets={r['source_id']:r for r in target_rows}
        records=[]
        for r in c['records']:
            sid=r['source_id'];old=targets[sid];label=labels[sid]
            req(type(label) is int and label in (0,1),'Approved label type differs')
            req(old['label']==label,'Historical GT label mismatch')
            records.append(dict(source_id=sid,category=r['category'],label=label,candidate_iou=old['candidate_iou']))
        put(root/'approved_targets24.json',dict(schema=SCHEMA,records=records,
            source_label_container_sha256=sha(OLD/'source_baseline/frozen/labels.json'),
            source_target_container_sha256=sha(OLD/'stages/05_evaluation.json'),
            source_container_scope='Opened approved historical120 label and frozen target containers; extracted/evaluated selected24 only',
            no_new_mask_reads=True,stage13_data_read=False))
        put(root/'approved_labels24.json',dict(schema=SCHEMA,labels={r['source_id']:r['label'] for r in records},
            source_label_container_sha256=sha(OLD/'source_baseline/frozen/labels.json'),
            selected_source_ids_sha256=digest(canon([r['source_id'] for r in records])),
            prediction_freeze_sha256=sha(root/'prediction_freeze.json'),
            source_container_scope='Approvedhistorical120containeropened; selected24labels derived only after fullpredictionfreeze',stage13_data_read=False))
        routing=read(root/'routing.json');n=24;K=c['K'];y=np.array([r['label'] for r in records]);duplicate=np.array([f[r['source_id'],'duplicate']['probability'] for r in records])
        results={};probabilities={'new_duplicate':duplicate}
        for name,selected in routing['routes'].items():
            chosen=[f[r['source_id'],'roi' if r['source_id'] in selected else 'duplicate'] for r in records]
            p=np.array([r['probability'] for r in chosen]);probabilities[name]=p
            controls=[f[sid,'duplicate'] for sid in selected];roi=[f[sid,'roi'] for sid in selected]
            results[name]=dict(selected_source_ids=selected,classification=pilot_metrics(y,p),
                candidate_retention=evaluation.metrics(records,p>=.5),
                budgets=dict(actual_shared_collection='See operations; all4policies share same newly collected potential outcomes',
                    selected_final_responses=usage_and_seconds(chosen),
                    pilot_single_policy_evidence_calls=24+24+K if name=='learned_gate' else 24+K,
                    theoretical_minimal_live_policy_calls=48 if name=='learned_gate' else 24,
                    theoretical_note='Gate24+onefinal24 versus controlonefinal24; uncertainty control additionally depends on old model score prepass not charged as newcall',
                    extra_duplicate_control_calls_for_selected_crop_sources=K,
                    duplicate_control_note='K selectedROI sources also have duplicatecontrol; other24-K duplicatecalls are chosenfinal outcomes',
                    gate24_extra_cost=usage_and_seconds(list(g.values())) if name=='learned_gate' else None,
                    selected_paired_newROI=usage_and_seconds(roi),selected_paired_newduplicate=usage_and_seconds(controls)),
                interpretation='Exploratory fixed24; shared potential outcomes, not four separately timed online deployments')
        results['new_duplicate']=dict(classification=pilot_metrics(y,duplicate),candidate_retention=evaluation.metrics(records,duplicate>=.5),
            budgets=usage_and_seconds([f[r['source_id'],'duplicate'] for r in records]))
        # Label-only bootstrap avoids categoryxlabel singleton zero-width artifacts.
        strata=defaultdict(list)
        for i,r in enumerate(records):strata[int(y[i])].append(i)
        rng=np.random.default_rng(42);weights=np.zeros((2000,n))
        for _,members in sorted(strata.items()):weights[:,members]=rng.multinomial(len(members),np.full(len(members),1/len(members)),size=2000)
        paired={}
        base=support.weighted_metrics(y,duplicate,weights)
        for name,p in probabilities.items():
            if name=='new_duplicate':continue
            boot=support.weighted_metrics(y,p,weights)
            paired[name]={k:dict(delta=results[name]['classification'][k]-results['new_duplicate']['classification'][k],
                ci95=np.quantile(boot[k]-base[k],[.025,.975],method='linear').tolist()) for k in ('auroc','ap','macro_f1','recall','fpr')}
        totalstarted,totalfailed,totalunknown=global_counts(root,c,images)
        result=dict(schema=SCHEMA,status='complete',timestamp_utc=now(),hostname=socket.gethostname(),prediction_freeze=frozen,
            n=24,normal=int(sum(y==0)),anomalous=int(sum(y==1)),K=K,eligible=c['eligible_count'],unionROI=len(routing['union_roi']),
            results=results,paired_bootstrap_vs_newduplicate=paired,bootstrap=dict(replicates=2000,seed=42,strata='image_label only',
                stratum_sizes={str(k):len(v) for k,v in strata.items()},fixed_routes=True,
                limitation='Fixed24/routes/provider outcomes; excludes policyselection/model/provider variation; categorydescriptiveonly'),
            operations=dict(accepted=len(g)+len(f),gate_accepted=len(g),final_accepted=len(f),started=totalstarted,failed=totalfailed,unknown=totalunknown,
                actual_collection_budget=usage_and_seconds(list(g.values())+list(f.values())),failed_usage='unknown',currency_bill='unavailable'),
            approved_label_sha256=sha(OLD/'source_baseline/frozen/labels.json'),historical_target_sha256=sha(OLD/'stages/05_evaluation.json'),
            approved_targets24_sha256=sha(root/'approved_targets24.json'),
            approved_labels24_sha256=sha(root/'approved_labels24.json'),
            source_label_container_opened='Approved historical120 container; evaluatedselected24 only after complete predictionfreeze',
            gate_is_live_model_decision=True,final_outcomes_newly_collected=True,crops_executed_after_gatefreeze=True,
            no_new_mask_reads=True,no_GT_route_selection=True,stage13_data_read=False,
            limitations=['Previously viewed24 development subset, not new blind data','FixedK model ranking, not autonomous unbounded tool loop',
                'Gatep measures crop usefulness, not anomalyconfidence','Candidate geometry frozen and teacher exposure persists',
                'Four strategies share same new potential outcomes; not four independent deployment timings',
                'Input view and timing may affect provider responses; olduncertaintycontrol has historicaltime',
                'No learned policy head training, no directBox/Choice/Score requests'])
        target=root/'metrics.json'
        if target.exists():
            previous=read(target);req(all(previous[k]==v for k,v in result.items() if k!='timestamp_utc'),'Frozen pilot result differs');result=previous
        else:put(target,result)
        lines=['# 真正Noul门控与新末级请求的24图试验','','这是已看开发图上的探索性pilot。gate为新API请求，选路后真正执行固定crop，duplicate和ROI末级也全部重新采集；不使用旧ROI作为末级结果。','',
            f'固定24中eligible={c["eligible_count"]}，共同K={K}，4策略ROI并集={len(routing["union_roi"])}。',
            f'真实采集成功={len(g)+len(f)}（gate24+duplicate24+unionROI），started={totalstarted}，失败={totalfailed}，unknown={totalunknown}。','',
            '| 策略 | AUROC | AP | Recall | FPR | 全异常候选IoU | 正常出框率 |','|---|---:|---:|---:|---:|---:|---:|']
        for name,v in results.items():
            m=v['classification'];z=v['candidate_retention'];lines.append('| '+name+' | '+' | '.join(str(m[k]) for k in ('auroc','ap','recall','fpr'))+f' | {z["anomaly_bbox_iou_mean_all"]} | {z["normal_positive_box_rate"]} |')
        lines+=['','## 成本与解释','',
            '- 真实总采集为gate24+dup24+并集ROI。单policy取证含额外duplicate对照：gatepolicy=48+K，控制=24+K。',
            '- 理论最简部署若只取每图最终一次，则gatepolicy=48请求、控制=24；历史不确定性控制依赖旧评分预pass，其成本另计，不把该理论数当已测独立时延。',
            '- 策略共享同一新潜在结果表，不声称分别真实部署四遍。gate24是额外成本，不能因复用抹掉。',
            '- 判定针对固定候选区域，非全图校准概率；gate概率是crop有用性，不是异常概率。未做Choice/Box/Score或policyhead训练。',
            '- 全部24与GT先冻结预测再评价；13确认数据未读取。主2000次bootstrap仅按正常/异常分层，类别仅描述；不含策略选择/模型/服务随机性，不据小pilot承诺部署收益。',
            '- 不选择更优结果换主策略；原始失败/usage缺失完整保留，不估货币账单。','']
        put(root/'REPORT.md','\n'.join(lines),text=True)
        put(root/'completion_certificate.json',dict(schema=SCHEMA,status='complete',n=24,accepted=len(g)+len(f),
            metrics_sha256=sha(root/'metrics.json'),report_sha256=sha(root/'REPORT.md'),prediction_freeze_sha256=sha(root/'prediction_freeze.json'),
            no_GT_route_selection=True,no_new_mask_reads=True,stage13_data_read=False))
        print(json.dumps(dict(status='complete',accepted=len(g)+len(f),report=str(root/'REPORT.md'))),flush=True)
    return 0


def pilot_metrics(y,p):
    # Regional candidate question and image-level labels have different targets:
    # report ranking/decision diagnostics, never regional Brier/ECE calibration.
    value=support.metrics(y,p)
    return {k:v for k,v in value.items() if k not in ('brier','ece_10_equal_width')}


def self_test(a):
    host();r=dict(source_id='fixture',category='pcb1',bbox_448_norm=[.1,.2,.6,.8],bbox_original_norm=[.1,.2,.6,.8])
    body=payload(r,'gate',{'fixture':{'query':b'fixture','crop':b'crop'}},'a'*64)
    req(len(body['assets'])==1 and set(body['questions'])=={'crop_needed'} and 'historical_duplicate_probability' not in body['state'],'Gate leakage')
    for forbidden in ('source_id','label','grade','mask_path','historical_duplicate_probability','teacher_map'):
        changed=json.loads(canon(body));changed['state'][forbidden]='sensitive'
        try:validate_gate_payload(changed)
        except ValueError:pass
        else:raise ValueError('Forbidden gate state key accepted')
    for value in ('Anomaly/ground_truth.png','grade=defect','mask.png','normal/label.txt'):
        changed=json.loads(canon(body));changed['state']['category']=value
        try:validate_gate_payload(changed)
        except ValueError:pass
        else:raise ValueError('Privileged string accepted')
    req(response_probability(dict(model=MODEL,answers={'crop_needed':dict(type='noul',noul=.4)}),'crop_needed')==.4,'Typed gate parse failed')
    req(not settled(dict(http_status=None,error='timeout_unknown_server_completion')),'Unknown completion settled')
    req(MAX_STARTS==72*3 and protocol()['n_queries']==24,'Pilot budget differs')
    print(json.dumps(dict(status='passed',synthetic_only=True,network_calls=0,no_GT_or_data_reads=True,
        checks=['gate_oneimage_no_oldscore','forbidden_state_key_mutations','grade_mask_label_path_strings_rejected',
            'typedNoul_question_identity','unknown_hardstop','24_72_216_bounds'])),flush=True);return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    group=p.add_mutually_exclusive_group(required=True)
    for name in ('prepare','dry-run','run','freeze-routes','evaluate','self-test'):group.add_argument('--'+name,action='store_true')
    p.add_argument('--phase',choices=['gate','final'],default='gate');p.add_argument('--retry-failed',action='store_true');p.add_argument('--key-env',default='MODELGO_API_KEY')
    a=p.parse_args()
    try:return prepare(a) if a.prepare else dry(a) if a.dry_run else run(a) if a.run else freeze_routes(a) if a.freeze_routes else evaluate(a) if a.evaluate else self_test(a)
    except Exception as error:
        print(json.dumps(dict(status='blocked',exception_class=type(error).__name__,reason=str(error) if isinstance(error,ValueError) else 'Inspect frozen pilot/runtime; arbitrary exception suppressed')),file=sys.stderr);return 2


if __name__=='__main__':sys.exit(main())
