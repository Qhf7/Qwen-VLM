#!/usr/bin/env python3
"""Server-only source-disjoint calibration and heldout development study.

--prepare-selection --root ROOT freezes the system-choice rule without
reading stage08 probabilities, labels, metrics or any stage12 evaluation.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime,timezone
import hashlib
import getpass
import io
import json
import math
import os
from pathlib import Path
import socket
import sys
import uuid

import numpy as np
from PIL import Image,ImageOps
from sklearn.linear_model import LogisticRegression
import roi_api as api
import support_robustness as support
import support_rescue as rescue

HOSTNAME='autodl-container-132444827f-9c594038'
SELECTION_SCHEMA='jev-source-disjoint-system-selection-v1'
MODEL='jev-1.14.0'
STUDY_SCHEMA='jev-source-disjoint-calibration-v1'
ROLES=('calibration','heldout_eval')
PLAN_HASHES={'preprotocol.json':'0f5021f67e2c345ed253b4cb81b07cb6d7c41398eae0bf6b9b1cf9b293fdc844',
    'amendment_01_allow1shot.json':'283facd5bb5bac924236bb9d95f715811c9469d3454d435a12b7bf584107bb1d',
    'metadata_audit.json':'b4b4539d8a5d02c584c5a4452d68b2f1df8472fe3afbfa515de2056746150933',
    'calibration_inputs.jsonl':'cacb9522e4c50df7388400aa9904b07c1a5284f4e0a72d2102d2b23e14bf8bfa',
    'heldout_eval_inputs.jsonl':'1e5ca69c49347861222397fadab2cf85d4b3090a17faa56ffc25ed1b14a06f99'}
LABEL_HASHES={'calibration':'aaaf4898122c685ed5a511af5b1d0d85d6eaa4d699a7822174eaa2a84a56fd1d',
    'heldout_eval':'7cc2853e08c36af1491a29b60622d42cdc2a377d803f07b962dd5a81942761d3'}


def require(value,message):
    if not value: raise ValueError(message)


def now(): return datetime.now(timezone.utc).isoformat()
def canonical(value): return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def digest(value): return hashlib.sha256(value).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): value.update(block)
    return value.hexdigest()


def publish(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): require(canonical(read(path))==canonical(value),'Frozen stage12 artifact differs');return
    temporary=path.with_name(path.name+'.'+str(uuid.uuid4())+'.publish.tmp')
    with temporary.open('xb') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode()+b'\n');stream.flush();os.fsync(stream.fileno())
    try: os.link(temporary,path)
    finally: temporary.unlink(missing_ok=True)


def selection_body(stage08_contract_path):
    contract=read(stage08_contract_path)
    require(contract.get('schema')=='jev-independent-support-robustness-v1'
        and contract.get('mode')=='formal_independent_support_robustness'
        and len(contract.get('schedule',[]))==600,'Formal stage08 contract required')
    require(contract.get('model')==MODEL and contract.get('size')==448,'Stage08 model/resolution differs')
    return dict(schema=SELECTION_SCHEMA,model=MODEL,size=448,
        stage08_contract_file_sha256=sha(stage08_contract_path),
        stage08_contract_canonical_sha256=digest(canonical(contract)),
        selection_metric='Whole-image development pooled AUROC from complete stage08 metrics.json only',
        candidates={'0':['shared0'],'1':['s43_1','s44_1'],'4':['s43_4','s44_4']},
        aggregation='Arithmetic mean of the listed AUROCs; shared0 collected once',
        tolerance_absolute_auroc=.01,
        rule='Find largest candidate mean AUROC; choose smallest shot count with mean >= maximum-0.01, inclusive; no best-seed selection',
        support_seed_if_nonzero=43,normal_reference_order='Frozen stage08 seed43 hash order; first1 or first4',
        prompt='Exact stage08 full-image INSTRUCTIONS and criteria; never regional ROI prompt',
        no320_fewshot_extrapolation=True,
        prerequisites=['stage08 complete600 and immutable metrics',
            'stage09 matched Qwen complete and stage10 resolution complete',
            'final research summary published before system choice'],
        forbidden_selection_inputs=['stage12 predictions','stage12 calibration labels/outcomes','stage12 heldout labels/outcomes',
            'best seed','heldout FPR/AUROC','320 fewshot unmeasured outcomes','confirmation'],
        source_policy='Two mutually exclusive new120 subsets, each12categories x5normal x5anomaly; source/byte disjoint from train and prior120',
        calibration=dict(fit_split='calibration120 only',feature='logit(clip(raw_probability,1e-6,1-1e-6))',
            model='sklearn LogisticRegression',C=1.,solver='lbfgs',penalty='l2',max_iter=1000,random_state=42,
            decision_threshold=.5,heldout_fit_forbidden=True),
        refusal=dict(confidence='max(p,1-p)',coverages=[.5,.75,1.],
            selection='Sort confidence descending; tie break by source_id ascending; retain exactly ceil(coverage*N)',
            interpretation='Retrospective fixed-coverage development risk analysis; coverage rule predeclared, no heldout metric optimization'),
        metrics=['AUROC','AP','Brier','ECE10_equal_width','Recall_at0.5','FPR_at0.5','MacroF1','fixed_coverage_risk'],
        bootstrap=dict(split='heldout120 only',replicates=2000,seed=42,strata='category x image_label',
            paired_raw_vs_calibrated=True,refit_on_heldout=False),
        historical_exposure_note='Native validation remainder may have participated in historical validation; source-disjoint is not blind or confirmation',
        no_stage08_metrics_read_at_rule_freeze=True,no_stage12_data_read_at_rule_freeze=True,confirmation_opened=False)


def prepare_selection(args):
    require(socket.gethostname()==HOSTNAME,'Freeze selection on designated server')
    root=args.root.resolve();source=root/'stage08_support/contract.json'
    body=selection_body(source);target=root/'stages/12_selection_protocol.json'
    if target.exists():
        document=read(target);require(document['protocol']==body and document['protocol_sha256']==digest(canonical(body)),'Frozen system-choice protocol differs')
    else:
        document=dict(schema=SELECTION_SCHEMA,status='frozen',timestamp_utc=now(),hostname=socket.gethostname(),
            protocol=body,protocol_sha256=digest(canonical(body)),stage08_predictions_or_metrics_read=False,
            stage12_data_or_results_read=False,confirmation_opened=False)
        publish(target,document)
    print(json.dumps(dict(status='frozen',protocol=str(target),protocol_sha256=document['protocol_sha256'],
        stage08_predictions_or_metrics_read=False,stage12_data_or_results_read=False),ensure_ascii=False),flush=True)
    return 0


def choose_shots(aucs):
    means={0:float(aucs['shared0']),1:float((aucs['s43_1']+aucs['s44_1'])/2),
           4:float((aucs['s43_4']+aucs['s44_4'])/2)}
    require(all(math.isfinite(v) and 0<=v<=1 for v in means.values()),'Invalid candidate AUROCs')
    maximum=max(means.values())
    return min(shots for shots in (0,1,4) if means[shots]>=maximum-.01-1e-12),means


def adapter_body(root):
    protocol=read(root/'stages/12_selection_protocol.json')
    require(protocol['protocol']==selection_body(root/'stage08_support/contract.json'),'Selection protocol changed')
    previous=root/'stages/12_selection_operational_adapter.json'
    require(sha(previous)=='d650bd4a8bb3538b4cc7cac5fd25a6619186fcb9adef4266ed345256e4028fdb','Prior operational adapter changed')
    return dict(schema='jev-stage12-operational-prerequisite-adapter-v2',scientific_selection_rule_changed=False,
        supersedes_operational_adapter_sha256=sha(previous),
        operational_correction='Stage09 Qwen freeze uses canonical contract digest and scoresSHA; stage10 resolution uses fileSHA and predictions320SHA',
        selection_protocol_sha256=sha(root/'stages/12_selection_protocol.json'),
        selection_protocol_canonical_sha256=protocol['protocol_sha256'],
        original_stage08_contract_sha256=sha(root/'stage08_support/contract.json'),
        operational_amendment_sha256=sha(root/'stage08_rescue/operational_amendment.json'),
        operational_adapter_source_sha256=sha(rescue.__file__),runner_sha256=sha(__file__),
        accepted_completion='Original597 plus exactly frozen eligibleHTTP502-only rescued3 via explicit completion_certificate; original status remains incomplete',
        stage08_metrics='stage08_rescue/metrics.json only after certified combined600',
        stage10_evaluation='stage10_resolution/evaluation_contract_amended_reference.json plus certified combined448reference',
        final_summary='stages/11_summary.json and FINAL_FOLLOWUP_REPORT.md with complete status/hash',
        unchanged=['0/1/4 candidates','mean AUROC with1pp inclusive tolerance','fewest shots within tolerance','fixedseed43 if nonzero',
            '448 resolution','question/criteria','training-normal support and order','source split','calibration procedure','heldout exclusion','confirmation exclusion'],
        no_stage08_metrics_or_stage12_data_read=True,confirmation_opened=False)


def prepare_operational_adapter(args):
    require(socket.gethostname()==HOSTNAME,'Freeze operational adapter on designated server')
    root=args.root.resolve();body=adapter_body(root);target=root/'stages/12_selection_operational_adapter_v2.json'
    publish(target,body)
    print(json.dumps(dict(status='frozen',adapter=str(target),sha256=sha(target),scientific_selection_rule_changed=False,
        no_stage08_metrics_or_stage12_data_read=True)),flush=True);return 0


def select_system(args):
    require(socket.gethostname()==HOSTNAME,'Select system on designated server')
    root=args.root.resolve();stage=root/'stage12';stage.mkdir(exist_ok=True)
    with api.output_lock(stage):
        frozen=read(root/'stages/12_selection_protocol.json')
        body=selection_body(root/'stage08_support/contract.json')
        require(frozen['protocol']==body and frozen['protocol_sha256']==digest(canonical(body)),'Pre-outcome selection rule changed')
        require(frozen.get('stage08_predictions_or_metrics_read') is False and frozen.get('stage12_data_or_results_read') is False,'Pre-outcome rule isolation missing')
        adapter=root/'stages/12_selection_operational_adapter_v2.json'
        require(read(adapter)==adapter_body(root),'Explicit operational prerequisite adapter not frozen or changed')
        # Explicit operational adapter: original incomplete08 evidence stays intact.
        # This does not change the previously frozen AUROC selection rule.
        contract,_,complete,_,_,_,unknown=rescue.load_complete(root)
        require(len(complete)==600 and not unknown,'Certified combined stage08 incomplete')
        rescue_stage=root/'stage08_rescue';certificate=read(rescue_stage/'completion_certificate.json')
        require(certificate['status']=='complete_with_operational_amendment' and certificate['scientific_change'] is False,
            'Explicit operational-only completion certificate required')
        prerequisites={'stage08_support':dict(status='complete_with_operational_amendment',
            original_status_sha256=sha(root/'stage08_support/status.json'),
            completion_certificate_sha256=sha(rescue_stage/'completion_certificate.json'),
            operational_amendment_sha256=sha(rescue_stage/'operational_amendment.json'),
            combined_predictions_sha256=sha(rescue_stage/'combined_predictions.jsonl'),
            metrics_sha256=sha(rescue_stage/'metrics.json'))}
        for folder,expected in [('stage09_qwen_matched',240),('stage10_resolution',120)]:
            status_path=root/folder/'status.json';status=read(status_path)
            require(status.get('status')=='complete' and status.get('completed')==status.get('expected')==expected,'Prerequisite stage incomplete: '+folder)
            metrics_path=root/folder/'metrics.json'
            require(metrics_path.is_file(),'Prerequisite evaluation missing: '+folder)
            freeze_path=root/folder/'prediction_freeze.json';stage_freeze=read(freeze_path)
            expected_contract=digest(canonical(read(root/folder/'contract.json'))) if folder=='stage09_qwen_matched' else sha(root/folder/'contract.json')
            require(stage_freeze['contract_sha256']==expected_contract,'Prerequisite prediction contract mismatch: '+folder)
            if folder=='stage09_qwen_matched':require(stage_freeze['scores_sha256']==sha(root/folder/'scores.jsonl'),'Prerequisite Qwen score freeze changed')
            else:require(stage_freeze['predictions320_sha256']==sha(root/folder/'predictions.jsonl'),'Prerequisite resolution predictions changed')
            prerequisites[folder]=dict(status_sha256=sha(status_path),metrics_sha256=sha(metrics_path),prediction_freeze_sha256=sha(freeze_path))
        resolution_freeze=read(root/'stage10_resolution/prediction_freeze.json')
        resolution_eval=root/'stage10_resolution/evaluation_contract_amended_reference.json'
        require(resolution_freeze['reference_completion_certificate_sha256']==sha(rescue_stage/'completion_certificate.json')
            and resolution_freeze['reference_predictions_sha256']==sha(rescue_stage/'combined_predictions.jsonl')
            and resolution_freeze['evaluation_contract_sha256']==sha(resolution_eval),
            'Resolution amended reference evidence mismatch')
        prerequisites['stage10_resolution']['amended_reference_evaluation_contract_sha256']=sha(resolution_eval)
        summary=api.inside(root,args.final_summary)
        require(summary.is_file() and summary.stat().st_size>0,'Final research summary required before system selection')
        summary_json=root/'stages/11_summary.json';summary_status=root/'stages/11_summary_status.json'
        final=read(summary_json);final_status=read(summary_status)
        require(final['status']=='complete' and final_status['status']=='complete'
            and final_status['summary_sha256']==sha(summary_json) and final_status['report_sha256']==sha(summary),
            'Final research summary complete certificate/hash missing')
        require(final.get('confirmation_opened') is False,'Final summary confirmation flag differs')
        prerequisite_summary=dict(path=summary.relative_to(root).as_posix(),sha256=sha(summary),
            json_sha256=sha(summary_json),status_sha256=sha(summary_status))
        metrics_path=rescue_stage/'metrics.json';metrics=read(metrics_path)
        require(metrics.get('status')=='complete' and metrics['prediction_freeze']['contract_sha256']==sha(root/'stage08_support/contract.json'),
                'Complete stage08 metrics contract mismatch')
        require(metrics['prediction_freeze']['predictions_sha256']==sha(rescue_stage/'combined_predictions.jsonl'),'Certified stage08 accepted predictions changed')
        require(frozen['timestamp_utc']<metrics['timestamp_utc'],'Selection rule was not frozen before stage08 outcomes')
        aucs={arm:metrics['arms'][arm]['metrics']['auroc'] for arm in support.ARMS}
        shots,means=choose_shots(aucs)
        supports={} if shots==0 else {c:records[:shots] for c,records in contract['supports']['43'].items()}
        for records in supports.values():
            for row in records: api.validated_asset(root/'stage08_support',row['asset'])
        selection=dict(schema=STUDY_SCHEMA,status='system_frozen',model=MODEL,size=448,shots=shots,
            support_seed=None if shots==0 else 43,supports=supports,instructions=support.INSTRUCTIONS,criteria=support.CRITERIA,
            selection_protocol_sha256=sha(root/'stages/12_selection_protocol.json'),selection_protocol_canonical_sha256=frozen['protocol_sha256'],
            candidate_mean_auroc={str(k):v for k,v in means.items()},stage08_metrics_sha256=sha(metrics_path),
            source_stage08_contract_sha256=sha(root/'stage08_support/contract.json'),
            prerequisites=prerequisites,final_summary=prerequisite_summary,
            runner_sha256=sha(__file__),transport_source_sha256=sha(api.__file__),support_source_sha256=sha(support.__file__),
            operational_completion_adapter_source_sha256=sha(rescue.__file__),
            operational_prerequisite_adapter_sha256=sha(adapter),
            dependencies=support.dependencies(),stage12_prediction_or_label_reads=0,confirmation_opened=False)
        target=stage/'system_contract.json'
        if target.exists():
            previous=read(target);require(previous==selection,'Frozen system choice differs')
        else: publish(target,selection)
        print(json.dumps(dict(status='system_frozen',shots=shots,support_seed=selection['support_seed'],
            system_contract_sha256=sha(target),candidate_mean_auroc=selection['candidate_mean_auroc'],stage12_prediction_or_label_reads=0)),flush=True)
    return 0


def load_system(root):
    path=root/'stage12/system_contract.json';system=read(path)
    require(system['schema']==STUDY_SCHEMA and system['status']=='system_frozen' and system['model']==MODEL and system['size']==448,'Frozen stage12 system missing')
    require(system['runner_sha256']==sha(__file__) and system['transport_source_sha256']==sha(api.__file__)
        and system['support_source_sha256']==sha(support.__file__)
        and system['operational_completion_adapter_source_sha256']==sha(rescue.__file__),'Frozen stage12 source changed')
    require(system['dependencies']==support.dependencies(),'Frozen stage12 dependencies changed')
    require(system['selection_protocol_sha256']==sha(root/'stages/12_selection_protocol.json'),'Selection protocol changed')
    require(system['operational_prerequisite_adapter_sha256']==sha(root/'stages/12_selection_operational_adapter_v2.json'),'Operational prerequisite adapter changed')
    require(system['source_stage08_contract_sha256']==sha(root/'stage08_support/contract.json'),'System source contract changed')
    p=system['prerequisites']['stage08_support']
    require(p['completion_certificate_sha256']==sha(root/'stage08_rescue/completion_certificate.json')
        and p['combined_predictions_sha256']==sha(root/'stage08_rescue/combined_predictions.jsonl')
        and p['metrics_sha256']==sha(root/'stage08_rescue/metrics.json'),'System certified stage08 evidence changed')
    require(system['instructions']==support.INSTRUCTIONS and system['criteria']==support.CRITERIA,'Full-image task changed')
    return system


def prepare_data(args):
    require(socket.gethostname()==HOSTNAME,'Prepare new data on designated server')
    root=args.root.resolve();stage=api.inside(root,args.stage);system=load_system(root)
    with api.output_lock(stage):
        plan=root/'stage12_calibration_plan'
        for name,expected in PLAN_HASHES.items(): require(sha(plan/name)==expected,'Frozen calibration source plan changed: '+name)
        inputs=[]
        for role in ROLES:
            rows=api.read_rows(plan/(role+'_inputs.jsonl'))
            require(len(rows)==len({r['source_id'] for r in rows})==120,'Need120 unique new inputs per split')
            require(all(r['role']==role for r in rows),'Input split role mismatch')
            inputs.extend(rows)
        require(len({r['source_id'] for r in inputs})==240,'Calibration/heldout source overlap')
        reference=read(root/'stage08_support/contract.json')
        require(sha(reference['source_files']['train']['path'])==reference['source_files']['train']['sha256'],'Frozen training manifest changed')
        train=api.read_rows(reference['source_files']['train']['path'])
        old_query_ids={r['source_id'] for r in reference['queries']}
        train_ids={r['source_id'] for r in train}
        new_ids={r['source_id'] for r in inputs};new_sha={r['original_sha256'] for r in inputs}
        require(len(new_sha)==240 and not new_ids.intersection(old_query_ids|train_ids),'New source/training/current query overlap')
        require(not new_sha.intersection({sid[5:] for sid in old_query_ids|train_ids}),'New byte/training/current query overlap')
        require(args.limit_per_split is None or args.limit_per_split==1,'Only1-target independent preflight is supported')
        selected=inputs if args.limit_per_split is None else sorted([r for r in inputs if r['role']=='calibration'],key=lambda r:r['source_id'])[:1]
        records=[]
        for row in sorted(selected,key=lambda r:(ROLES.index(r['role']),r['source_id'])):
            require(row['source_id']=='visa_'+row['original_sha256'],'New source ID/SHA mismatch')
            path=Path(row['image']).resolve()
            require(path.is_relative_to(Path('/root/autodl-tmp/industrial_vlm/datasets/VisA')),'New image outside approved data root')
            raw=path.read_bytes();require(digest(raw)==row['original_sha256'],'New query original SHA mismatch')
            png,geometry=support.render(raw)
            destination=stage/'assets'/(row['source_id']+'.png');destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists(): require(destination.read_bytes()==png,'Existing new query render differs')
            else:
                with destination.open('xb') as stream: stream.write(png);stream.flush();os.fsync(stream.fileno())
            records.append(dict(source_id=row['source_id'],category=row['category'],role=row['role'],
                original_path=str(path),original_sha256=row['original_sha256'],original_bytes=len(raw),
                asset=dict(path=destination.relative_to(stage).as_posix(),png_sha256=digest(png),png_bytes=len(png),size=[448,448],**geometry)))
        contract=dict(schema=STUDY_SCHEMA,mode='formal_source_disjoint_calibration' if args.limit_per_split is None else 'independent_one_target_preflight',
            system_contract_sha256=sha(root/'stage12/system_contract.json'),model=MODEL,size=448,shots=system['shots'],
            source_plan_files=PLAN_HASHES,source_train_file_sha256=sha(reference['source_files']['train']['path']),
            current_query_contract_sha256=sha(root/'stage08_support/contract.json'),source_and_byte_overlap=0,
            records=records,schedule=[dict(source_id=r['source_id'],role=r['role']) for r in records],
            runner_sha256=sha(__file__),no_gt_or_label_reads=True,confirmation_opened=False)
        publish(stage/'contract.json',contract)
        images=load_images(root,stage,contract,system)
        completed,starts,attempts,hashes,unknown=validate_logs(stage,contract,system,images)
        status=state(contract,completed,starts,attempts,unknown,'complete' if len(completed)==len(records) else 'prepared')
        api.write_json(stage/'status.json',status);print(json.dumps(status),flush=True)
    return 0


def load_images(root,stage,contract,system):
    images={}
    for row in contract['records']:
        _,raw=api.validated_asset(stage,row['asset']);images[row['source_id']]=raw
    for rows in system['supports'].values():
        for row in rows:
            _,raw=api.validated_asset(root/'stage08_support',row['asset']);images[row['source_id']]=raw
    return images


def verify_data(root,stage):
    system=load_system(root);contract=read(stage/'contract.json')
    require(contract['schema']==STUDY_SCHEMA and contract['runner_sha256']==sha(__file__)
        and contract['system_contract_sha256']==sha(root/'stage12/system_contract.json'),'Frozen new data contract changed')
    for name,expected in contract['source_plan_files'].items(): require(sha(root/'stage12_calibration_plan'/name)==expected,'New source plan changed')
    reference=read(root/'stage08_support/contract.json')
    require(contract['current_query_contract_sha256']==sha(root/'stage08_support/contract.json')
        and contract['source_train_file_sha256']==sha(reference['source_files']['train']['path'])==reference['source_files']['train']['sha256'],
        'Original train/query identity changed after data freeze')
    return contract,system,load_images(root,stage,contract,system)


def payload(row,system,images,contract_hash):
    import base64
    assets={}
    def add(source,role):
        reference=str(uuid.uuid5(uuid.NAMESPACE_URL,contract_hash+'|'+row['source_id']+'|'+role))
        assets[reference]=dict(type='image',format='png',data_base64=base64.b64encode(images[source['source_id']]).decode('ascii'));return reference
    query=add(row,'query')
    refs=[add(s,'reference-'+str(i)) for i,s in enumerate(system['supports'].get(row['category'],[]))]
    require(len(refs)==system['shots'],'Frozen support count mismatch')
    return dict(model=MODEL,state=dict(query=query,category=row['category'].replace('_',' '),normal_references=refs),
        assets=assets,questions={'anomaly':dict(type='noul',instructions=support.INSTRUCTIONS,criteria=support.CRITERIA)})


def validate_logs(stage,contract,system,images):
    ch=digest(canonical(contract));records={r['source_id']:r for r in contract['records']}
    hashes={sid:digest(canonical(payload(row,system,images,ch))) for sid,row in records.items()}
    complete={}
    for row in api.read_rows(stage/'predictions.jsonl'):
        sid=row['source_id'];require(sid in hashes and sid not in complete,'Unexpected/duplicate new accepted target')
        require(row['contract_sha256']==ch and row['request_sha256']==hashes[sid] and row['role']==records[sid]['role'],'New prediction identity differs')
        api.response_probability(dict(model=row['response_model'],answers={'anomaly':dict(type='noul',noul=row['probability'])}))
        require(row['category']==records[sid]['category'] and api.safe_usage({'usage':row['usage']})==row['usage'],'New prediction metadata differs')
        complete[sid]=row
    attempts=api.read_rows(stage/'attempts.jsonl');starts=Counter();intents={};terminals=set()
    for row in attempts:
        sid=row['source_id'];identity=sid,row['attempt']
        require(sid in hashes and row['contract_sha256']==ch and row['request_sha256']==hashes[sid],'New attempt identity differs')
        require(type(row['attempt']) is int and 1<=row['attempt']<=3,'New attempt budget exceeded')
        if row['event']=='started':
            starts[sid]+=1;require(starts[sid]==row['attempt'] and identity not in intents,'New intent sequence differs');intents[identity]=row
        else:
            require(row['event'] in ('failed','succeeded') and identity in intents and identity not in terminals,'Invalid new terminal')
            require(row['invocation_id']==intents[identity]['invocation_id'],'New invocation differs');terminals.add(identity)
            if row['event']=='succeeded': require(sid in complete and complete[sid]['attempt']==row['attempt'],'Succeeded new target absent')
    for sid,row in complete.items(): require((sid,row['attempt']) in intents,'New prediction lacks durable intent')
    unknown=[identity for identity in intents if identity not in terminals and identity[0] not in complete]
    return complete,starts,attempts,hashes,unknown


def state(contract,complete,starts,attempts,unknown,status,reason=None):
    return dict(schema=STUDY_SCHEMA,status=status,timestamp_utc=now(),mode=contract['mode'],
        expected=len(contract['schedule']),completed=len(complete),contract_sha256=digest(canonical(contract)),
        by_role=dict(Counter(r['role'] for r in complete.values())),started_attempts=sum(starts.values()),
        failed_attempts=sum(r['event']=='failed' for r in attempts),interrupted_unknown_attempts=len(unknown),
        exhausted_unaccepted=sum(starts[r['source_id']]>=3 and r['source_id'] not in complete for r in contract['records']),
        stop_reason=reason,automatic_retries=False,confirmation_opened=False)


def dry_run(args):
    require(socket.gethostname()==HOSTNAME,'Dry-run on designated server');root=args.root.resolve();stage=api.inside(root,args.stage)
    contract,system,images=verify_data(root,stage);complete,starts,attempts,hashes,unknown=validate_logs(stage,contract,system,images)
    print(json.dumps(state(contract,complete,starts,attempts,unknown,'dry_run_passed')),flush=True);return 0


def run(args):
    require(socket.gethostname()==HOSTNAME,'Run API on designated server');root=args.root.resolve();stage=api.inside(root,args.stage)
    with api.output_lock(stage):
        contract,system,images=verify_data(root,stage);complete,starts,attempts,hashes,unknown=validate_logs(stage,contract,system,images)
        if len(complete)==len(contract['records']): api.write_json(stage/'status.json',state(contract,complete,starts,attempts,unknown,'complete'));return 0
        pending=[r['source_id'] for r in contract['records'] if r['source_id'] not in complete and starts[r['source_id']]<3
                 and (starts[r['source_id']]==0 or args.retry_failed)]
        if not pending or (stage/'STOP').exists():
            api.write_json(stage/'status.json',state(contract,complete,starts,attempts,unknown,'incomplete_requires_audit_or_explicit_retry',
                'STOP' if (stage/'STOP').exists() else 'no_eligible_targets'));return 2
        key=os.environ.get(args.key_env)
        if not key:
            require(sys.stdin.isatty(),'Credential env unset; hidden input requiresTTY');key=getpass.getpass('ModelGo API key (hidden, never saved): ')
        require(isinstance(key,str) and key and key.strip()==key,'Missing or invalid credential')
        invocation=str(uuid.uuid4());ch=digest(canonical(contract));records={r['source_id']:r for r in contract['records']}
        api.append_json(stage/'invocations.jsonl',dict(invocation_id=invocation,timestamp_utc=now(),retry_failed=args.retry_failed,
            unknown_completion_retry_risk_acknowledged=bool(args.retry_failed and unknown),contract_sha256=ch))
        futures={};position=0;consecutive=0;stop=None
        with ThreadPoolExecutor(max_workers=3) as pool:
            while position<len(pending) or futures:
                if (stage/'STOP').exists(): stop=stop or 'STOP_file'
                while not stop and position<len(pending) and len(futures)<3:
                    sid=pending[position];position+=1;body=payload(records[sid],system,images,ch)
                    require(digest(canonical(body))==hashes[sid],'New payload changed before send');starts[sid]+=1
                    intent=dict(source_id=sid,role=records[sid]['role'],attempt=starts[sid],event='started',
                        contract_sha256=ch,request_sha256=hashes[sid],invocation_id=invocation,timestamp_utc=now())
                    api.append_json(stage/'attempts.jsonl',intent);attempts.append(intent);futures[pool.submit(api.send,body,key)]=(sid,intent)
                if not futures:break
                done,_=wait(futures,timeout=.5,return_when=FIRST_COMPLETED)
                for future in done:
                    sid,intent=futures.pop(future);value,error,http_status,seconds=future.result()
                    if value is not None:
                        row=dict(**value,source_id=sid,category=records[sid]['category'],role=records[sid]['role'],attempt=intent['attempt'],
                            contract_sha256=ch,request_sha256=hashes[sid],invocation_id=invocation,seconds=seconds,timestamp_utc=now())
                        api.append_json(stage/'predictions.jsonl',row);complete[sid]=row;consecutive=0
                        terminal=dict(intent,event='succeeded',timestamp_utc=now(),http_status=http_status,seconds=seconds)
                    else:
                        consecutive+=1;terminal=dict(intent,event='failed',timestamp_utc=now(),http_status=http_status,seconds=seconds,error=error)
                        if consecutive>=3:stop=stop or 'consecutive_errors'
                    api.append_json(stage/'attempts.jsonl',terminal);attempts.append(terminal)
                    api.write_json(stage/'status.json',state(contract,complete,starts,attempts,unknown,'draining_inflight' if stop else 'running',stop))
                    print(json.dumps(dict(completed=len(complete),expected=len(contract['records']),role=records[sid]['role'],http_status=http_status,error=error)),flush=True)
        complete,starts,attempts,hashes,unknown=validate_logs(stage,contract,system,images)
        finished=len(complete)==len(contract['records']);status=state(contract,complete,starts,attempts,unknown,'complete' if finished else 'incomplete_requires_explicit_retry',stop)
        api.write_json(stage/'status.json',status);print(json.dumps(status),flush=True);return 0 if finished else 2


def clipped_logit(probabilities):
    values=np.asarray(probabilities,dtype=float)
    require(np.isfinite(values).all() and np.all((values>=0)&(values<=1)),'Invalid raw probabilities')
    clipped=np.clip(values,1e-6,1-1e-6)
    return np.log(clipped/(1-clipped)).reshape(-1,1)


def label_file(plan,role,records):
    path=plan/(role+'_labels.jsonl')
    require(sha(path)==LABEL_HASHES[role],'Frozen source-plan labels changed')
    rows=api.read_rows(path)
    found={r['source_id']:r for r in rows}
    require(len(found)==len(rows)==120 and set(found)=={r['source_id'] for r in records},'Label/source role identities differ')
    require(all(r['role']==role and type(r['is_anomalous']) is bool for r in rows),'Invalid label role/type')
    y=np.array([int(found[r['source_id']]['is_anomalous']) for r in records])
    counts=Counter((r['category'],int(value)) for r,value in zip(records,y))
    require(len({r['category'] for r in records})==12 and all(counts[c,k]==5 for c in {r['category'] for r in records} for k in (0,1)),
            'Role label strata differ from frozen5normal5anomaly design')
    return y,sha(path)


def coverage_analysis(records,y,p):
    confidence=np.maximum(p,1-p)
    order=sorted(range(len(records)),key=lambda i:(-float(confidence[i]),records[i]['source_id']))
    result={}
    for coverage in (.5,.75,1.):
        indices=np.array(order[:int(math.ceil(coverage*len(records)))],dtype=int)
        prediction=p[indices]>=.5
        result[str(coverage)]=dict(requested_coverage=coverage,accepted=len(indices),total=len(records),
            actual_coverage=len(indices)/len(records),error_rate=float(np.mean(prediction!=y[indices])),
            confidence_cutoff=float(confidence[indices[-1]]),
            accepted_source_ids=[records[i]['source_id'] for i in indices],
            metrics=support.metrics(y[indices],p[indices]),
            note='Fixed fraction retrospective risk; confidence tie break source_id ascending; no heldout threshold optimization')
    return result


def heldout_bootstrap(records,y,raw,calibrated):
    strata=defaultdict(list)
    for i,row in enumerate(records):strata[row['category'],int(y[i])].append(i)
    rng=np.random.default_rng(42);weights=np.zeros((2000,len(y)))
    for _,members in sorted(strata.items()):weights[:,members]=rng.multinomial(len(members),np.full(len(members),1/len(members)),size=2000)
    a=support.weighted_metrics(y,calibrated,weights);b=support.weighted_metrics(y,raw,weights)
    ma=support.metrics(y,calibrated);mb=support.metrics(y,raw)
    result={k:dict(delta=ma[k]-mb[k],ci95=np.quantile(a[k]-b[k],[.025,.975],method='linear').tolist(),
        replicates=2000,seed=42,strata='category x image_label',paired=True,calibrator_refitted=False) for k in support.METRIC_NAMES}
    return dict(direction='calibrated-minus-raw',pooled=result,
        limitation='Fixed calibration120 fit; heldout resampling does not include calibrator-fit or provider uncertainty')


def evaluate(args):
    require(socket.gethostname()==HOSTNAME,'Evaluate on designated server');root=args.root.resolve();stage=api.inside(root,args.stage)
    with api.output_lock(stage):
        contract,system,images=verify_data(root,stage);status=read(stage/'status.json')
        require(contract['mode']=='formal_source_disjoint_calibration' and status['status']=='complete'
            and status['completed']==status['expected']==240 and status['contract_sha256']==digest(canonical(contract)),
            'Need formal240 complete before labels/calibration')
        complete,starts,attempts,hashes,unknown=validate_logs(stage,contract,system,images)
        require(len(complete)==240 and not unknown,'Formal240 accepted response audit incomplete')
        freeze=dict(schema=STUDY_SCHEMA,contract_sha256=sha(stage/'contract.json'),system_contract_sha256=sha(root/'stage12/system_contract.json'),
            predictions_sha256=sha(stage/'predictions.jsonl'),attempts_sha256=sha(stage/'attempts.jsonl'),
            runner_sha256=sha(__file__),confirmation_opened=False,predictions_before_labels=True)
        publish(stage/'prediction_freeze.json',freeze)
        by_role={role:[r for r in contract['records'] if r['role']==role] for role in ROLES}
        require(all(len(v)==120 for v in by_role.values()),'Need two formal120 splits')
        plan=root/'stage12_calibration_plan'
        calibration_rows=by_role['calibration'];heldout_rows=by_role['heldout_eval']
        ycal,cal_label_sha=label_file(plan,'calibration',calibration_rows)
        pcal=np.array([complete[r['source_id']]['probability'] for r in calibration_rows])
        # Fit only calibration120. Persist coefficients before opening heldout labels.
        calibrator=LogisticRegression(C=1.,penalty='l2',solver='lbfgs',max_iter=1000,random_state=42)
        calibrator.fit(clipped_logit(pcal),ycal)
        fit=dict(schema=STUDY_SCHEMA,feature='logit(clip(p,1e-6,1-1e-6))',C=1.,penalty='l2',solver='lbfgs',
            max_iter=1000,random_state=42,coefficient=calibrator.coef_.tolist(),intercept=calibrator.intercept_.tolist(),
            classes=calibrator.classes_.tolist(),n_iter=calibrator.n_iter_.tolist(),calibration_label_sha256=cal_label_sha,
            calibration_prediction_records_sha256=digest(canonical([complete[r['source_id']] for r in calibration_rows])),
            heldout_labels_used_for_fit=False,confirmation_opened=False)
        publish(stage/'calibrator.json',fit)
        raw=np.array([complete[r['source_id']]['probability'] for r in heldout_rows])
        calibrated=calibrator.predict_proba(clipped_logit(raw))[:,1]
        yh,heldout_label_sha=label_file(plan,'heldout_eval',heldout_rows)
        methods={'raw':raw,'calibrated':calibrated}
        by_category={}
        for category in sorted({r['category'] for r in heldout_rows}):
            idx=np.array([i for i,r in enumerate(heldout_rows) if r['category']==category])
            by_category[category]={name:support.metrics(yh[idx],p[idx]) for name,p in methods.items()}
        all_results={name:dict(metrics=support.metrics(yh,p),coverage=coverage_analysis(heldout_rows,yh,p)) for name,p in methods.items()}
        bootstrap=heldout_bootstrap(heldout_rows,yh,raw,calibrated)
        brier_delta=all_results['calibrated']['metrics']['brier']-all_results['raw']['metrics']['brier']
        ece_delta=all_results['calibrated']['metrics']['ece_10_equal_width']-all_results['raw']['metrics']['ece_10_equal_width']
        usage_fields=sorted({k for r in complete.values() for k in r['usage']})
        result=dict(schema=STUDY_SCHEMA,status='complete',timestamp_utc=now(),hostname=socket.gethostname(),prediction_freeze=freeze,
            system_shots=system['shots'],support_seed=system['support_seed'],calibrator=fit,
            calibration_label_sha256=cal_label_sha,heldout_label_sha256=heldout_label_sha,
            calibration_fit_metrics=dict(raw=support.metrics(ycal,pcal),calibrated=support.metrics(ycal,calibrator.predict_proba(clipped_logit(pcal))[:,1])),
            heldout=all_results,by_category=by_category,
            category_macro={method:{k:float(np.mean([by_category[c][method][k] for c in by_category])) for k in support.METRIC_NAMES} for method in methods},
            paired_bootstrap=bootstrap,
            operations=dict(accepted=240,started=sum(starts.values()),failed=sum(r['event']=='failed' for r in attempts),unknown=len(unknown),
                usage={k:dict(observed_sum=float(sum(r['usage'][k] for r in complete.values() if k in r['usage'])),
                    observed_calls=sum(k in r['usage'] for r in complete.values()),missing_calls=sum(k not in r['usage'] for r in complete.values())) for k in usage_fields}),
            analysis=[f'固定calibration120拟合后，heldout120 Brier变化{brier_delta:+.6f}，ECE10变化{ece_delta:+.6f}；负值表示该指标改善。',
                '校准不自动提高排序或所有阈值指标；报告完整配对区间与拒答覆盖率，不在heldout重选系统或阈值。'],
            limitations=['Source-disjoint new subsets may have appeared in historical native validation',
                'Not confirmation, blind or cross-domain testing','Balanced50percent prevalence differs from production',
                'Only120 calibration observations, fixed C and clip; no heldout refit',
                'Retrospective fixed coverage is not deployment coverage guarantee',
                'Bootstrap conditions on fitted calibrator, excludes calibration fit/provider uncertainty',
                'Logical gateway model echo is not upstream immutable weight proof','No currency bill; missing usage unknown'],
            no_gt_tuning=True,heldout_used_for_fit=False,confirmation_opened=False)
        target=stage/'metrics.json'
        if target.exists():
            old=read(target);require(all(old[k]==result[k] for k in result if k!='timestamp_utc'),'Frozen calibration result differs');result=old
        else:publish(target,result)
        lines=['# Jev 新来源校准与留出开发评价','',
            f"系统按先冻结的stage08规则选择：{system['shots']}正常参考，支持seed={system['support_seed']}，448输入。",
            '校准120与heldout120互斥，并与训练来源、先前120 query按source/byte隔离；本阶段不读取confirmation。',
            '完整240预测先冻结，再仅用calibration120拟合固定logistic；系数保存后才读取heldout标签。','',
            '| Heldout120 | AUROC | AP | Recall | FPR | Brier | ECE10 |','|---|---:|---:|---:|---:|---:|---:|']
        for name,value in all_results.items():lines.append('| '+name+' | '+' | '.join(f"{value['metrics'][k]:.6f}" for k in ('auroc','ap','recall','fpr','brier','ece_10_equal_width'))+' |')
        lines+=['','## 分析','',*['- '+line for line in result['analysis']],'',
            '| 方法 | 覆盖率 | 保留数 | 分类错误率 |','|---|---:|---:|---:|']
        for name,value in all_results.items():
            for coverage,item in value['coverage'].items():lines.append(f"| {name} | {coverage} | {item['accepted']} | {item['error_rate']:.6f} |")
        lines+=['','## 解释边界','',
            '- 这是与当前120来源互斥的开发校准/评价；native验证剩余池可能曾被历史验证查看，不能称全新盲测或最终confirmation。',
            '- 系统只根据先前stage08 AUROC规则选择；heldout不用于选prompt、支持seed、shots、输入尺寸或阈值。',
            '- 固定覆盖率按置信度排序及source ID破同分；不是在heldout上寻找最优拒答阈值。',
            '- 每类仅5正常/5异常；校准样本量、类别异质性与先验变化限制外推。无真实货币账单不估费用。','']
        text='\n'.join(lines);report=stage/'REPORT.md'
        if report.exists():require(report.read_text(encoding='utf-8')==text,'Frozen calibration report differs')
        else:
            with report.open('x',encoding='utf-8') as stream:stream.write(text);stream.flush();os.fsync(stream.fileno())
        require(sha(stage/'predictions.jsonl')==freeze['predictions_sha256'],'Predictions changed during calibration evaluation')
        print(json.dumps(dict(status='complete',accepted=240,calibration=120,heldout=120,metrics=str(target),report=str(report),confirmation_opened=False)),flush=True)
    return 0


def self_test(args):
    require(socket.gethostname()==HOSTNAME,'Self-test on designated server')
    shots,means=choose_shots(dict(shared0=.8,s43_1=.82,s44_1=.82,s43_4=.825,s44_4=.825))
    require(shots==1,'Fewest within1pp choice failed')
    require(choose_shots(dict(shared0=.82,s43_1=.82,s44_1=.82,s43_4=.82,s44_4=.82))[0]==0,'Tie should prefer0')
    require(choose_shots(dict(shared0=.82,s43_1=.83,s44_1=.83,s43_4=.83,s44_4=.83))[0]==0,'Inclusive1pp edge failed')
    values=clipped_logit([0.,.5,1.]);require(np.isfinite(values).all() and values[1,0]==0,'Clip/logit failed')
    labels=np.array([0,0,1,1]);p=np.array([.1,.3,.7,.9])
    model=LogisticRegression(C=1.,penalty='l2',solver='lbfgs',max_iter=1000,random_state=42).fit(clipped_logit(p),labels)
    calibrated=model.predict_proba(clipped_logit(p))[:,1];require(np.isfinite(calibrated).all(),'Calibration fit failed')
    records=[dict(source_id='fixture'+str(i),category='fixture') for i in range(4)]
    coverage=coverage_analysis(records,labels,p);require(coverage['0.5']['accepted']==2 and coverage['1.0']['accepted']==4,'Fixed coverage failed')
    system=dict(shots=0,supports={});row=dict(source_id='fixture',category='pcb1')
    body=payload(row,system,{'fixture':b'synthetic'},'a'*64)
    require(len(body['assets'])==1 and body['state']['normal_references']==[],'System payload failed')
    require('role' not in body['state'] and 'source_id' not in body['state'],'Privileged metadata in request')
    print(json.dumps(dict(status='passed',synthetic_only=True,network_calls=0,checks=['fixed_system_choice','fewest_shots_tie',
        'clip_logit','fixed_logistic','fixed_coverage','normal_reference_payload','no_privileged_metadata'])),flush=True)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    action=parser.add_mutually_exclusive_group(required=True)
    for flag in ('prepare-selection','prepare-operational-adapter','select-system','prepare-data','dry-run','run','evaluate','self-test'):
        action.add_argument('--'+flag,action='store_true')
    parser.add_argument('--stage',default='stage12/run')
    parser.add_argument('--final-summary',default='FINAL_FOLLOWUP_REPORT.md')
    parser.add_argument('--limit-per-split',type=int,choices=[1],help='Independent one-calibration-target preflight only')
    parser.add_argument('--key-env',default='MODELGO_API_KEY')
    parser.add_argument('--retry-failed',action='store_true')
    args=parser.parse_args()
    functions={'prepare_selection':prepare_selection,'prepare_operational_adapter':prepare_operational_adapter,
        'select_system':select_system,'prepare_data':prepare_data,
        'dry_run':dry_run,'run':run,'evaluate':evaluate,'self_test':self_test}
    try: return next(functions[name](args) for name in functions if getattr(args,name))
    except Exception as error:
        print(json.dumps(dict(status='blocked',exception_class=type(error).__name__,reason=str(error) if isinstance(error,ValueError) else 'Inspect frozen protocol/runtime')),file=sys.stderr);return 2


if __name__=='__main__':sys.exit(main())
