#!/usr/bin/env python3
"""Server-only one-time confirmation under a result-independent final policy."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime,timezone
import getpass
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
import uuid

import numpy as np
import roi_api as api
import support_robustness as support
import isolated_calibration as calibration

HOSTNAME='autodl-container-132444827f-9c594038'
SCHEMA='jev-one-time-final-confirmation-v1'
EXPECTED=240
MAX_STARTED=720
MAX_FAILED=30
STAGE='stage13_confirmation'
DATA_ROOT=Path('/root/autodl-tmp/industrial_vlm/datasets/VisA')
PROJECT=Path('/root/autodl-tmp/industrial_vlm')


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return api.file_sha(Path(path))
def canonical(value):return api.canonical(value)
def digest(value):return api.digest(value)


def publish(path,value,plain=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=value.encode('utf-8') if plain else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode()+b'\n'
    if path.exists():require(path.read_bytes()==raw,'Frozen confirmation artifact differs: '+path.name);return
    temporary=path.with_name(path.name+'.'+str(uuid.uuid4())+'.tmp')
    descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    try:os.link(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def host_gate():require(socket.gethostname()==HOSTNAME,'Confirmation operations only on designated server')


def policy_body(root):
    system=calibration.load_system(root)
    require(system['shots']==4 and system['support_seed']==43 and system['size']==448,'Only already selected4shot43/448 system')
    return dict(schema=SCHEMA,model='jev-1.14.0',shots=4,support_seed=43,size=448,
        selected_system_contract_sha256=sha(root/'stage12/system_contract.json'),
        primary='Jev raw probability mapped by fixed calibration120 logistic logit mapping; probability >=0.5',
        secondary='Same single raw probability per image; raw probability >=0.5; never replaces primary based on heldout or confirmation',
        calibration=dict(data='stage12 calibration120 only',feature='logit(clip(p,1e-6,1-1e-6))',
            C=1.,penalty='l2',solver='lbfgs',max_iter=1000,random_state=42,heldout_refit=False),
        fixed_question_instructions=system['instructions'],fixed_question_criteria=system['criteria'],
        primary_denominator='All240; no primary refusal or abstention',
        refusal=dict(coverages=[.5,.75,1.],confidence='max(p,1-p)',
            order='descending confidence; source_id ascending for ties; ceil(fraction*N)',
            interpretation='Predeclared retrospective coverage-risk analysis only; no optimized threshold or deployment guarantee'),
        confirmation=dict(expected_sources=240,categories=12,per_category_per_label=10,
            immutable_legacy_path='prepared/multitask_v2/confirmation_sources.jsonl',
            construction_source='ops/prepare_multitask_v2.py',
            source_policy='Original fixed test240; no resampling/replacement; abort on any count/hash/source-disjoint violation',
            disjoint_from=['all native train768','previous query120','stage12 calibration120','stage12 heldout120']),
        opening_order=['stage12 complete240 raw predictions and durable prediction_freeze',
            'calibration120 fit saved before heldout labels','stage12 heldout metrics saved',
            'freeze final system bundle including exact coefficient/intercept and all code/runtime hashes',
            'only then privileged broker may open legacy confirmation metadata and images',
            'broker extracts input and separately protected boolean labels; no masks opened',
            'API runner consumes inputs/images/supports only; no protected labels in payload',
            'all240 predictions frozen before final evaluator opens confirmation labels'],
        budgets=dict(expected_successes=240,max_started_global=720,max_started_per_target=3,workers=3,
            max_cumulative_failed_global=30,consecutive_error_submission_stop=3,
            inflight_reserved_before_submit=True,counters_rebuilt_from_durable_ledger_each_invocation=True,
            retry_requires_explicit_new_invocation=True,unknown_completion_hard_stop=True,
            no_automatic_http_retry=True,no_retry_after_cumulative_failure_cap=True,
            failure_drain_note='Atfailure30 stop further starts, drain previously reserved inflight; observed terminal failures may finish up to32',
            missing_usage='unknown; neverzero',currency_cost='unavailable; no inferred monetary bill'),
        metrics=list(support.METRIC_NAMES),bootstrap=dict(replicates=2000,seed=42,strata='category x image_label',
            paired_primary_vs_raw=True,calibrator_refit=False),
        scientific_choices_independent_of_stage12_heldout_results=True,
        confirmation_opened=False,confirmation_used_for_selection=False,
        limitations=['Single fixed test sample; not crossdomain','Public benchmark pretraining contamination unknown',
            'Source/byte isolation does not prove physical part or near-duplicate independence',
            'Balanced50percent prevalence differs from production','Provider/model echo is not immutable upstream weights proof'])


def prepare_policy(args):
    host_gate();root=args.root.resolve();stage=root/STAGE
    with api.output_lock(stage):
        body=policy_body(root);target=stage/'preprotocol.json'
        if target.exists():require(read(target)['policy']==body,'Predeclared final policy changed')
        else:publish(target,dict(schema=SCHEMA,status='policy_frozen_before_stage12_outcomes',timestamp_utc=api.now(),
            policy=body,policy_sha256=digest(canonical(body)),confirmation_metadata_or_images_read=False,
            stage12_predictions_or_metrics_read=False,confirmation_opened=False))
        print(json.dumps(dict(status='policy_frozen',preprotocol_sha256=sha(target),
            policy_sha256=read(target)['policy_sha256'],confirmation_opened=False)),flush=True)
    return 0


def load_policy(root):
    path=root/STAGE/'preprotocol.json';document=read(path)
    require(sha(path)=='1630df051d37301944cd30baae2a7e4107e4684edcf43b0e90dbe0fa37a57986','Pre-outcome confirmation policy changed')
    require(document['policy']==policy_body(root) and document['policy_sha256']==digest(canonical(document['policy'])),
        'Frozen confirmation policy differs')
    require(document['confirmation_metadata_or_images_read'] is False and document['stage12_predictions_or_metrics_read'] is False,
        'Pre-outcome policy isolation missing')
    return document


def validate_fit(fit):
    for key,value in dict(feature='logit(clip(p,1e-6,1-1e-6))',C=1.,penalty='l2',solver='lbfgs',max_iter=1000,random_state=42,
        classes=[0,1],heldout_labels_used_for_fit=False,confirmation_opened=False).items():
        require(fit[key]==value,'Fixed calibration procedure differs: '+key)
    coefficient=np.asarray(fit['coefficient'],dtype=float);intercept=np.asarray(fit['intercept'],dtype=float)
    require(coefficient.shape==(1,1) and intercept.shape==(1,) and np.isfinite(coefficient).all() and np.isfinite(intercept).all(),
        'Invalid finite logistic mapping shape')
    require(fit['calibration_label_sha256']==calibration.LABEL_HASHES['calibration'],'Fixed calibration label SHA differs')
    return coefficient,intercept


def apply_fit(probabilities,fit):
    coefficient,intercept=validate_fit(fit);z=calibration.clipped_logit(probabilities)[:,0]*coefficient[0,0]+intercept[0]
    # Algebraic logistic with sign split avoids overflow, without changing the fitted mapping.
    result=np.empty_like(z);positive=z>=0
    result[positive]=1/(1+np.exp(-z[positive]));ez=np.exp(z[~positive]);result[~positive]=ez/(1+ez)
    require(np.isfinite(result).all() and np.all((result>=0)&(result<=1)),'Invalid calibrated probability')
    return result


def freeze_final(args):
    host_gate();root=args.root.resolve();stage=root/STAGE
    with api.output_lock(stage):
        policy=load_policy(root);source=root/'stage12/run'
        contract,system,images=calibration.verify_data(root,source)
        status=read(source/'status.json')
        require(contract['mode']=='formal_source_disjoint_calibration' and status['status']=='complete'
            and status['completed']==status['expected']==240 and status['contract_sha256']==digest(canonical(contract)),
            'Final bundle requires completed formal stage12')
        complete,starts,attempts,hashes,unknown=calibration.validate_logs(source,contract,system,images)
        require(len(complete)==240 and not unknown,'Final bundle requires settled240 and no unknowns')
        require(all(policy['timestamp_utc']<row['timestamp_utc'] for row in complete.values()),'Main policy not frozen before stage12 outcomes')
        frozen=read(source/'prediction_freeze.json');fit=read(source/'calibrator.json');metrics=read(source/'metrics.json')
        for field,name in [('contract_sha256','contract.json'),('predictions_sha256','predictions.jsonl'),('attempts_sha256','attempts.jsonl')]:
            require(frozen[field]==sha(source/name),'Stage12 freeze mismatch: '+name)
        require(frozen['system_contract_sha256']==sha(root/'stage12/system_contract.json') and frozen['predictions_before_labels'] is True,
            'Stage12 system or prelabel freeze differs')
        validate_fit(fit)
        calrows=[r for r in contract['records'] if r['role']=='calibration']
        require(len(calrows)==120 and fit['calibration_prediction_records_sha256']==digest(canonical([complete[r['source_id']] for r in calrows])),
            'Calibration coefficient prediction/source identity differs')
        require(metrics['status']=='complete' and metrics['prediction_freeze']==frozen and metrics['calibrator']==fit
            and metrics['heldout_used_for_fit'] is False and metrics['confirmation_opened'] is False
            and metrics['heldout_label_sha256']==calibration.LABEL_HASHES['heldout_eval'],'Heldout final evaluation provenance invalid')
        require((source/'REPORT.md').is_file(),'Stage12 analysis report missing')
        bundle=dict(schema=SCHEMA,status='final_system_frozen_before_confirmation',
            preprotocol_sha256=sha(stage/'preprotocol.json'),policy_sha256=policy['policy_sha256'],
            system=system,calibrator=fit,primary='calibrated',secondary='raw',threshold=.5,
            source_files={name:sha(source/name) for name in ('contract.json','predictions.jsonl','attempts.jsonl','prediction_freeze.json',
                'calibrator.json','metrics.json','REPORT.md')},
            selected_system_contract_sha256=sha(root/'stage12/system_contract.json'),
            source_sha256={name:sha(path) for name,path in [('final_confirmation',__file__),('isolated_calibration',calibration.__file__),
                ('support_robustness',support.__file__),('roi_api',api.__file__)]},
            dependencies=support.dependencies(),selection_independent_of_heldout_metrics=True,
            no_new_fit_or_parameter_choice=True,confirmation_metadata_or_images_read=False,confirmation_opened=False)
        target=stage/'final_system_bundle.json';publish(target,bundle)
        print(json.dumps(dict(status='final_system_frozen',bundle_sha256=sha(target),primary='calibrated',secondary='raw',
            confirmation_opened=False)),flush=True)
    return 0


def load_bundle(root):
    host_gate();stage=root/STAGE;policy=load_policy(root);bundle=read(stage/'final_system_bundle.json')
    require(bundle['schema']==SCHEMA and bundle['status']=='final_system_frozen_before_confirmation'
        and bundle['preprotocol_sha256']==sha(stage/'preprotocol.json') and bundle['policy_sha256']==policy['policy_sha256'],
        'Final system bundle absent or changed')
    require(bundle['selected_system_contract_sha256']==sha(root/'stage12/system_contract.json')
        and bundle['system']==calibration.load_system(root),'Final selected system changed')
    for name,path in [('final_confirmation',__file__),('isolated_calibration',calibration.__file__),
        ('support_robustness',support.__file__),('roi_api',api.__file__)]:
        require(bundle['source_sha256'][name]==sha(path),'Final system source changed: '+name)
    require(bundle['dependencies']==support.dependencies(),'Final system runtime changed')
    for name,expected in bundle['source_files'].items():require(sha(root/'stage12/run'/name)==expected,'Frozen stage12 artifact changed: '+name)
    validate_fit(bundle['calibrator']);return bundle


def jsonl(rows):return ''.join(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n' for row in rows)


def open_inputs(args):
    """Privileged one-time broker; never sends requests and never opens masks."""
    root=args.root.resolve();stage=root/STAGE
    with api.output_lock(stage):
        bundle=load_bundle(root);marker=stage/'protected_opening.json'
        require(not marker.exists(),'Confirmation opening already reserved; do not reopen or replace sample; audit partial state')
        # Durable exclusive reservation is published before the first legacy byte read.
        publish(marker,dict(schema=SCHEMA,status='one_time_opening_reserved',timestamp_utc=api.now(),
            bundle_sha256=sha(stage/'final_system_bundle.json'),preprotocol_sha256=sha(stage/'preprotocol.json'),
            protected_metadata_may_be_opened_after_this_marker=True,no_reselection_or_second_opening=True))
        legacy=PROJECT/'prepared/multitask_v2/confirmation_sources.jsonl'
        provenance_path=PROJECT/'prepared/multitask_v2/provenance_audit.json'
        provenance=read(provenance_path)
        require(sha(legacy)==provenance['prepared_artifact_sha256']['confirmation_sources.jsonl'],'Original confirmation metadata SHA differs from construction provenance')
        construction=PROJECT/'ops/prepare_multitask_v2.py'
        construction_rows=[r for r in provenance['inputs'] if r['path']=='ops/prepare_multitask_v2.py']
        require(len(construction_rows)==1 and construction_rows[0]['sha256']==sha(construction),'Legacy confirmation construction source changed')
        rows=api.read_rows(legacy)
        require(len(rows)==EXPECTED and len({r['source_id'] for r in rows})==EXPECTED,'Original confirmation count/uniqueness differs; no replacement')
        require(all(type(r['is_anomalous']) is bool and r['split']=='test' for r in rows),'Original confirmation type/testsplit mismatch')
        counts=Counter((r['category'],r['is_anomalous']) for r in rows);categories={r['category'] for r in rows}
        require(len(categories)==12 and len(counts)==24 and all(counts[c,label]==10 for c in categories for label in (False,True)),
            'Original confirmation category/label design differs; no replacement')
        require(digest(''.join(r['source_id']+'\n' for r in rows).encode())==provenance['confirmation_source_order_sha256'],
            'Original confirmation source order differs from construction provenance')
        oldcontract=read(root/'stage08_support/contract.json');train=api.read_rows(oldcontract['source_files']['train']['path'])
        prior=read(root/'stage12/run/contract.json')['records']
        forbidden_ids={r['source_id'] for r in train+oldcontract['queries']+prior}
        forbidden_sha={sid[5:] for sid in forbidden_ids}
        ids={r['source_id'] for r in rows};hashes={r['image_sha256'] for r in rows}
        require(len(hashes)==EXPECTED and not ids.intersection(forbidden_ids) and not hashes.intersection(forbidden_sha),
            'Confirmation intersects train/previous query/calibration/heldout source or bytes; stop without substitution')
        inputs=[];labels=[];rendered=[]
        for row in sorted(rows,key=lambda r:r['source_id']):
            require(row['source_id']=='visa_'+row['image_sha256'],'Confirmation source/SHA identity differs')
            path=Path(row['image']).resolve()
            require(path.is_relative_to(DATA_ROOT),'Confirmation original path outside approved data root')
            raw=path.read_bytes();require(digest(raw)==row['image_sha256'],'Confirmation original bytes changed')
            png,geometry=support.render(raw);asset=stage/'assets'/(row['source_id']+'.png')
            asset.parent.mkdir(parents=True,exist_ok=True)
            with asset.open('xb') as stream:stream.write(png);stream.flush();os.fsync(stream.fileno())
            inputs.append(dict(source_id=row['source_id'],category=row['category'],asset=dict(path=asset.relative_to(stage).as_posix(),
                png_sha256=digest(png),png_bytes=len(png),size=[448,448],**geometry)))
            labels.append(dict(source_id=row['source_id'],is_anomalous=row['is_anomalous']))
            rendered.append(dict(source_id=row['source_id'],original_sha256=row['image_sha256'],original_bytes=len(raw),png_sha256=digest(png)))
        publish(stage/'inputs.jsonl',jsonl(inputs),plain=True)
        publish(stage/'protected_labels.jsonl',jsonl(labels),plain=True)
        input_contract=dict(schema=SCHEMA,mode='one_time_final_confirmation',model='jev-1.14.0',size=448,shots=4,support_seed=43,
            expected=EXPECTED,bundle_sha256=sha(stage/'final_system_bundle.json'),opening_marker_sha256=sha(marker),
            legacy_metadata_sha256=sha(legacy),legacy_provenance_sha256=sha(provenance_path),
            legacy_construction_source_sha256=sha(construction),legacy_source_order_sha256=provenance['confirmation_source_order_sha256'],
            input_sha256=sha(stage/'inputs.jsonl'),protected_label_sha256=sha(stage/'protected_labels.jsonl'),
            source_and_byte_overlap=0,records=inputs,render_provenance=rendered,schedule=[r['source_id'] for r in inputs],
            runner_sha256=sha(__file__),broker_only_labels_opened=True,inference_labels_opened=False,masks_opened=False,
            confirmation_opened=True,no_sample_replacement=True)
        publish(stage/'contract.json',input_contract)
        publish(stage/'protected_opening_complete.json',dict(schema=SCHEMA,status='input_label_broker_complete',
            opening_marker_sha256=sha(marker),contract_sha256=sha(stage/'contract.json'),expected=EXPECTED,
            inputs_sha256=sha(stage/'inputs.jsonl'),labels_sha256=sha(stage/'protected_labels.jsonl'),
            source_and_byte_overlap=0,masks_opened=False))
        contract,bundle,images=verify_inputs(root)
        complete,starts,attempts,hashes,unknown=audit_logs(stage,contract,bundle,images)
        status=state(contract,complete,starts,attempts,unknown,'prepared');api.write_json(stage/'status.json',status)
        print(json.dumps(dict(status='prepared',expected=EXPECTED,source_and_byte_overlap=0,contract_sha256=sha(stage/'contract.json'),
            protected_labels_only_broker_opened=True,masks_opened=False)),flush=True)
    return 0


def verify_inputs(root):
    """Inference gate reads no legacy metadata or protected label bytes."""
    stage=root/STAGE;bundle=load_bundle(root);contract=read(stage/'contract.json')
    require(contract['schema']==SCHEMA and contract['mode']=='one_time_final_confirmation' and contract['expected']==EXPECTED
        and contract['runner_sha256']==sha(__file__) and contract['bundle_sha256']==sha(stage/'final_system_bundle.json'),
        'Final confirmation input contract changed')
    require(contract['input_sha256']==sha(stage/'inputs.jsonl'),'Confirmation input file changed')
    require(api.read_rows(stage/'inputs.jsonl')==contract['records'] and len(contract['records'])==EXPECTED
        and len({r['source_id'] for r in contract['records']})==EXPECTED
        and contract['schedule']==[r['source_id'] for r in contract['records']],'Confirmation schedule changed')
    opening=read(stage/'protected_opening_complete.json')
    require(opening['contract_sha256']==sha(stage/'contract.json') and opening['opening_marker_sha256']==sha(stage/'protected_opening.json')
        ==contract['opening_marker_sha256'] and opening['expected']==EXPECTED and opening['inputs_sha256']==contract['input_sha256']
        and opening['labels_sha256']==contract['protected_label_sha256'] and opening['masks_opened'] is False,'One-time opening provenance changed')
    images={}
    for row in contract['records']:
        require(set(row)=={'source_id','category','asset'},'Privileged metadata in inference inputs')
        require(set(row['asset'])=={'path','png_sha256','png_bytes','size','original_size','oriented_size','resized_size','padding'},
            'Unexpected metadata in inference asset')
        _,raw=api.validated_asset(stage,row['asset']);images[row['source_id']]=raw
    for rows in bundle['system']['supports'].values():
        for row in rows:
            _,raw=api.validated_asset(root/'stage08_support',row['asset']);images[row['source_id']]=raw
    return contract,bundle,images


def payload(row,bundle,images,contract_hash):
    return calibration.payload(row,bundle['system'],images,contract_hash)


def audit_logs(stage,contract,bundle,images):
    ch=digest(canonical(contract));records={r['source_id']:r for r in contract['records']}
    hashes={sid:digest(canonical(payload(row,bundle,images,ch))) for sid,row in records.items()}
    predictions=api.read_rows(stage/'predictions.jsonl');complete={}
    attempts=api.read_rows(stage/'attempts.jsonl');starts=Counter();intents={};terminals={}
    invocations_list=api.read_rows(stage/'invocations.jsonl');invocations={r['invocation_id']:r for r in invocations_list}
    require(len(invocations)==len(invocations_list),'Duplicate confirmation invocation')
    for row in invocations_list:
        require(row['contract_sha256']==ch and row['bundle_sha256']==contract['bundle_sha256'] and row['explicit_invocation'] is True,
            'Confirmation invocation binding differs')
    reservations=0;failed=0;consecutive=0;current_invocation=None;blocked_invocations=set();seen_invocations=set();ambiguous_seen=False
    for row in attempts:
        sid=row['source_id'];identity=(sid,row['attempt'])
        require(sid in hashes and row['contract_sha256']==ch and row['request_sha256']==hashes[sid]
            and row['bundle_sha256']==contract['bundle_sha256'] and row['invocation_id'] in invocations,
            'Confirmation request/ledger identity changed')
        require(type(row['attempt']) is int and 1<=row['attempt']<=3,'Confirmation target startcap exceeded')
        if row['event']=='started':
            if row['invocation_id']!=current_invocation:
                require(reservations==0 and not ambiguous_seen,'New invocation before settled drain or after ambiguous completion')
                require(row['invocation_id'] not in seen_invocations,'Prior confirmation invocation reopened')
                record=invocations[row['invocation_id']]
                require(record['prior_cumulative_started']==sum(starts.values()) and record['prior_cumulative_failed']==failed
                    and record['no_unknown_completion'] is True,'Persistent invocation prior counters differ')
                current_invocation=row['invocation_id'];seen_invocations.add(current_invocation);consecutive=0
            require(failed<MAX_FAILED and sum(starts.values())<MAX_STARTED,'Confirmation persistent global budget exceeded')
            require(not ambiguous_seen,'Submission after ambiguous completion')
            require(reservations<3,'Confirmation rolling concurrency exceeded')
            require(row['invocation_id'] not in blocked_invocations,'Submission after invocation circuit stop')
            if starts[sid]:
                previous=terminals.get((sid,starts[sid]));require(previous is not None and previous['event']=='failed',
                    'Confirmation unknown/success request cannot be resubmitted')
                require(previous['invocation_id']!=row['invocation_id'] and invocations[row['invocation_id']]['retry_failed'] is True,
                    'Confirmation retry requires new explicit invocation')
            starts[sid]+=1;require(row['attempt']==starts[sid] and identity not in intents,'Confirmation intent sequence differs')
            intents[identity]=row;reservations+=1
        else:
            require(row['event'] in ('succeeded','failed') and identity in intents and identity not in terminals,
                'Confirmation terminal missing/duplicate intent')
            require(row['invocation_id']==intents[identity]['invocation_id'],'Confirmation terminal invocation differs')
            terminals[identity]=row;reservations-=1
            if row['event']=='failed':
                failed+=1;consecutive+=1
                if not settled_http_failure(row):ambiguous_seen=True;blocked_invocations.add(row['invocation_id'])
            else:consecutive=0
            if consecutive>=3:blocked_invocations.add(row['invocation_id'])
    for row in predictions:
        sid=row['source_id'];identity=(sid,row['attempt'])
        require(sid in hashes and sid not in complete and identity in intents,'Unexpected/duplicate confirmation acceptance')
        require(row['contract_sha256']==ch and row['request_sha256']==hashes[sid] and row['bundle_sha256']==contract['bundle_sha256']
            and row['invocation_id']==intents[identity]['invocation_id'],'Confirmation accepted request differs')
        require(identity not in terminals or terminals[identity]['event']=='succeeded','Accepted confirmation response has conflicting failed terminal')
        require(starts[sid]==row['attempt'],'Accepted confirmation target was subsequently resubmitted')
        api.response_probability(dict(model=row['response_model'],answers={'anomaly':dict(type='noul',noul=row['probability'])}))
        require(row['category']==records[sid]['category'] and api.safe_usage({'usage':row['usage']})==row['usage'],'Confirmation accepted metadata invalid')
        require(type(row['seconds']) in (int,float) and math.isfinite(row['seconds']) and row['seconds']>=0,'Invalid confirmation call seconds')
        complete[sid]=row
    for identity,row in terminals.items():
        if row['event']=='succeeded':require(identity[0] in complete and complete[identity[0]]['attempt']==identity[1],'Confirmation success absent')
    # Ambiguous transport/timeout/model parsing may have consumed a paid request:
    # they remain a permanent unknown completion, even with a failed ledger row.
    unknown=[identity for identity in intents if identity not in terminals and identity[0] not in complete]
    unknown.extend(identity for identity,row in terminals.items() if row['event']=='failed' and not settled_http_failure(row))
    require(sum(starts.values())<=MAX_STARTED and failed<=MAX_FAILED+2,'Persistent confirmation budget/drain bound exceeded')
    return complete,starts,attempts,hashes,unknown


def settled_http_failure(row):
    status=row.get('http_status')
    return type(status) is int and status!=200 and row.get('error')=='http_'+str(status)


def state(contract,complete,starts,attempts,unknown,status,reason=None):
    failed=sum(r['event']=='failed' for r in attempts)
    return dict(schema=SCHEMA,status=status,timestamp_utc=api.now(),expected=EXPECTED,completed=len(complete),
        contract_sha256=digest(canonical(contract)),bundle_sha256=contract['bundle_sha256'],
        cumulative_started=sum(starts.values()),cumulative_failed=failed,unknown=len(unknown),
        max_started_global=MAX_STARTED,max_started_per_target=3,failure_submission_stop_threshold=MAX_FAILED,
        permitted_inflight_drain_failure_upper_bound=MAX_FAILED+2,
        exhausted_unaccepted=sum(starts[sid]>=3 and sid not in complete for sid in contract['schedule']),
        cumulative_failure_lock=failed>=MAX_FAILED,stop_reason=reason,automatic_retries=False,
        inference_label_reads=0,masks_opened=False,confirmation_opened=True)


def dry_run(args):
    root=args.root.resolve();stage=root/STAGE;contract,bundle,images=verify_inputs(root)
    complete,starts,attempts,hashes,unknown=audit_logs(stage,contract,bundle,images)
    print(json.dumps(state(contract,complete,starts,attempts,unknown,'dry_run_passed')),flush=True);return 0


def run(args):
    root=args.root.resolve();stage=root/STAGE
    with api.output_lock(stage):
        contract,bundle,images=verify_inputs(root);complete,starts,attempts,hashes,unknown=audit_logs(stage,contract,bundle,images)
        require(not unknown,'Unknown confirmation completion: hard stop without resend')
        if len(complete)==EXPECTED:
            api.write_json(stage/'status.json',state(contract,complete,starts,attempts,unknown,'complete'));return 0
        failed=sum(r['event']=='failed' for r in attempts)
        pending=[sid for sid in contract['schedule'] if sid not in complete and starts[sid]<3 and (starts[sid]==0 or args.retry_failed)]
        if failed>=MAX_FAILED or sum(starts.values())>=MAX_STARTED or not pending or (stage/'STOP').exists():
            status=state(contract,complete,starts,attempts,unknown,'incomplete_requires_audit',
                'persistent_budget_lock' if failed>=MAX_FAILED or sum(starts.values())>=MAX_STARTED else 'STOP_or_noeligible')
            api.write_json(stage/'status.json',status);print(json.dumps(status),flush=True);return 2
        credential=os.environ.get(args.key_env)
        if not credential:
            require(sys.stdin.isatty(),'Credential env unset; hidden input needsTTY');credential=getpass.getpass('ModelGo key (hidden, never saved): ')
        require(isinstance(credential,str) and credential and credential.strip()==credential,'Missing or invalid credential')
        invocation=str(uuid.uuid4());ch=digest(canonical(contract))
        api.append_json(stage/'invocations.jsonl',dict(invocation_id=invocation,timestamp_utc=api.now(),explicit_invocation=True,
            retry_failed=args.retry_failed,contract_sha256=ch,bundle_sha256=contract['bundle_sha256'],
            prior_cumulative_started=sum(starts.values()),prior_cumulative_failed=failed,no_unknown_completion=True))
        records={r['source_id']:r for r in contract['records']};futures={};position=0;consecutive=0;stop=None
        with ThreadPoolExecutor(max_workers=3) as pool:
            while position<len(pending) or futures:
                if (stage/'STOP').exists():stop=stop or 'STOP'
                while not stop and position<len(pending) and len(futures)<3:
                    if failed>=MAX_FAILED or sum(starts.values())>=MAX_STARTED:
                        stop='persistent_budget_lock';break
                    sid=pending[position];position+=1;body=payload(records[sid],bundle,images,ch)
                    require(digest(canonical(body))==hashes[sid],'Final request body changed before send')
                    starts[sid]+=1;intent=dict(source_id=sid,attempt=starts[sid],event='started',invocation_id=invocation,
                        timestamp_utc=api.now(),contract_sha256=ch,request_sha256=hashes[sid],bundle_sha256=contract['bundle_sha256'])
                    api.append_json(stage/'attempts.jsonl',intent);attempts.append(intent)
                    futures[pool.submit(api.send,body,credential)]=(sid,intent)
                if not futures:break
                done,_=wait(futures,timeout=.5,return_when=FIRST_COMPLETED)
                for future in done:
                    sid,intent=futures.pop(future);value,error,http_status,seconds=future.result()
                    if value is not None:
                        row=dict(**value,source_id=sid,category=records[sid]['category'],attempt=intent['attempt'],invocation_id=invocation,
                            timestamp_utc=api.now(),contract_sha256=ch,request_sha256=hashes[sid],bundle_sha256=contract['bundle_sha256'],seconds=seconds)
                        api.append_json(stage/'predictions.jsonl',row);complete[sid]=row;consecutive=0
                        terminal=dict(intent,event='succeeded',http_status=http_status,seconds=seconds,timestamp_utc=api.now())
                    else:
                        failed+=1;consecutive+=1;terminal=dict(intent,event='failed',error=error,http_status=http_status,seconds=seconds,timestamp_utc=api.now())
                        if consecutive>=3:stop=stop or 'consecutive_errors'
                        if failed>=MAX_FAILED:stop=stop or 'persistent_failure_lock'
                        if not settled_http_failure(terminal):
                            stop=stop or 'unknown_server_completion';unknown.append((sid,intent['attempt']))
                    api.append_json(stage/'attempts.jsonl',terminal);attempts.append(terminal)
                    api.write_json(stage/'status.json',state(contract,complete,starts,attempts,unknown,'draining_inflight' if stop else 'running',stop))
                    print(json.dumps(dict(completed=len(complete),expected=EXPECTED,http_status=http_status,error=error,
                        cumulative_started=sum(starts.values()),cumulative_failed=failed)),flush=True)
        complete,starts,attempts,hashes,unknown=audit_logs(stage,contract,bundle,images)
        status=state(contract,complete,starts,attempts,unknown,'complete' if len(complete)==EXPECTED else 'incomplete_requires_audit_or_explicit_retry',stop)
        api.write_json(stage/'status.json',status);print(json.dumps(status),flush=True);return 0 if len(complete)==EXPECTED else 2


def evaluate(args):
    root=args.root.resolve();stage=root/STAGE
    with api.output_lock(stage):
        contract,bundle,images=verify_inputs(root);complete,starts,attempts,hashes,unknown=audit_logs(stage,contract,bundle,images)
        status=read(stage/'status.json')
        require(status['status']=='complete' and status['completed']==status['expected']==EXPECTED
            and len(complete)==EXPECTED and not unknown,'Need all240 settled predictions before confirmation labels')
        frozen=dict(schema=SCHEMA,contract_sha256=sha(stage/'contract.json'),predictions_sha256=sha(stage/'predictions.jsonl'),
            attempts_sha256=sha(stage/'attempts.jsonl'),bundle_sha256=sha(stage/'final_system_bundle.json'),
            runner_sha256=sha(__file__),predictions_before_confirmation_label_evaluation=True,
            no_confirmation_fit_or_selection=True,confirmation_opened=True,masks_opened=False)
        publish(stage/'prediction_freeze.json',frozen)
        records=contract['records'];raw=np.array([complete[r['source_id']]['probability'] for r in records])
        primary=apply_fit(raw,bundle['calibrator'])
        mapped=[dict(source_id=r['source_id'],raw_probability=float(p),calibrated_probability=float(q),
            primary_prediction=bool(q>=.5),secondary_prediction=bool(p>=.5)) for r,p,q in zip(records,raw,primary)]
        publish(stage/'fixed_mapped_predictions.jsonl',jsonl(mapped),plain=True)
        # Only this final evaluator opens protected boolean label bytes.
        label_path=stage/'protected_labels.jsonl'
        require(sha(label_path)==contract['protected_label_sha256'],'Protected labels changed after one-time broker')
        labels=api.read_rows(label_path);lookup={r['source_id']:r for r in labels}
        require(len(labels)==len(lookup)==EXPECTED and set(lookup)=={r['source_id'] for r in records},'Confirmation labels/source identities differ')
        require(all(set(row)=={'source_id','is_anomalous'} and type(row['is_anomalous']) is bool for row in labels),'Protected label shape invalid')
        y=np.array([int(lookup[r['source_id']]['is_anomalous']) for r in records]);counts=Counter((r['category'],int(value)) for r,value in zip(records,y))
        require(len(counts)==24 and all(counts[c,k]==10 for c in {r['category'] for r in records} for k in (0,1)),
            'Confirmation final label strata differ; no sample replacement')
        methods=dict(primary_calibrated=primary,secondary_raw=raw)
        results={name:dict(metrics=support.metrics(y,p),retrospective_coverage=calibration.coverage_analysis(records,y,p)) for name,p in methods.items()}
        by_category={}
        for category in sorted({r['category'] for r in records}):
            idx=np.array([i for i,r in enumerate(records) if r['category']==category])
            by_category[category]={name:support.metrics(y[idx],p[idx]) for name,p in methods.items()}
        macro={name:{k:float(np.mean([by_category[c][name][k] for c in by_category])) for k in support.METRIC_NAMES} for name in methods}
        strata=defaultdict(list)
        for i,row in enumerate(records):strata[row['category'],int(y[i])].append(i)
        rng=np.random.default_rng(42);weights=np.zeros((2000,EXPECTED))
        for _,members in sorted(strata.items()):weights[:,members]=rng.multinomial(len(members),np.full(len(members),1/len(members)),size=2000)
        boot={name:support.weighted_metrics(y,p,weights) for name,p in methods.items()}
        intervals={name:{k:dict(value=results[name]['metrics'][k],ci95=np.quantile(v[k],[.025,.975],method='linear').tolist())
            for k in support.METRIC_NAMES} for name,v in boot.items()}
        delta={k:dict(value=results['primary_calibrated']['metrics'][k]-results['secondary_raw']['metrics'][k],
            ci95=np.quantile(boot['primary_calibrated'][k]-boot['secondary_raw'][k],[.025,.975],method='linear').tolist()) for k in support.METRIC_NAMES}
        usage_fields=sorted({k for row in complete.values() for k in row['usage']})
        result=dict(schema=SCHEMA,status='complete',timestamp_utc=api.now(),hostname=socket.gethostname(),
            primary='primary_calibrated',secondary='secondary_raw',threshold=.5,denominator=EXPECTED,
            prediction_freeze=frozen,fixed_mapped_predictions_sha256=sha(stage/'fixed_mapped_predictions.jsonl'),
            protected_labels_sha256=sha(label_path),system_policy_sha256=read(stage/'preprotocol.json')['policy_sha256'],
            results=results,by_category=by_category,category_macro=macro,
            bootstrap=dict(replicates=2000,seed=42,strata='category x image_label',fixed_calibrator=True,
                intervals=intervals,primary_minus_secondary=delta),
            operations=dict(accepted=EXPECTED,started=sum(starts.values()),failed=sum(r['event']=='failed' for r in attempts),unknown=0,
                failure_submission_stop_threshold=MAX_FAILED,inflight_drain_failure_upper_bound=MAX_FAILED+2,
                max_started_global=MAX_STARTED,max_started_per_target=3,
                usage={k:dict(observed_sum=float(sum(row['usage'][k] for row in complete.values() if k in row['usage'])),
                    observed_calls=sum(k in row['usage'] for row in complete.values()),missing_calls=sum(k not in row['usage'] for row in complete.values())) for k in usage_fields},
                failed_request_usage='unknown',currency_bill='unavailable'),
            selection_independent_of_confirmation=True,no_refit=True,no_threshold_optimization=True,
            primary_all240_without_refusal=True,confirmation_opened=True,masks_opened=False,
            limitations=read(stage/'preprotocol.json')['policy']['limitations']+[
                'Historical metadata stratification used labels; image predictions and label evaluation sealed until final system',
                'Teacher training exposure requires separate audit; not guaranteed unseen by every prior model',
                'Bootstrap conditions on fixed calibration/provider responses; no fit/provider uncertainty',
                'Retrospective fixed coverage is not deployment coverage guarantee'])
        target=stage/'metrics.json'
        if target.exists():
            previous=read(target);require(all(previous[k]==v for k,v in result.items() if k!='timestamp_utc'),'Frozen final result differs');result=previous
        else:publish(target,result)
        lines=['# Jev 一次性最终确认评价','','## 固定系统与结论边界','',
            '主系统在stage12结果前声明为Jev4正常参考、seed43、448输入及固定calibration120 logistic映射；同一次raw概率仅作次要诊断。',
            '最终bundle在stage12完整预测、校准与heldout评价后冻结；随后一次性broker才开启原固定240来源，不替换样本，不用确认结果再选择系统或阈值。',
            '主评价分母全部240，不拒答；阈值固定p≥0.5。这是本轮确认流程终点，不代表跨域或部署保证。','',
            '| 预声明系统 | AUROC | AP | MacroF1 | Recall | FPR | Brier | ECE10 |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for name,v in results.items():lines.append('| '+name+' | '+' | '.join(f"{v['metrics'][k]:.6f}" for k in support.METRIC_NAMES)+' |')
        brier=delta['brier'];ece=delta['ece_10_equal_width']
        lines+=['','## 分析','',
            f"- 主映射−raw Brier={brier['value']:+.6f}，95%配对区间[{brier['ci95'][0]:+.6f},{brier['ci95'][1]:+.6f}]；负值为改善。",
            f"- 主映射−raw ECE10={ece['value']:+.6f}，95%配对区间[{ece['ci95'][0]:+.6f},{ece['ci95'][1]:+.6f}]；不因其优劣切换主系统。",
            '- 固定概率映射可能改变阈值与可靠性，不能假定所有指标提升；逐类/宏平均/2000分层配对区间见metrics.json。','',
            '| 系统 | 回顾覆盖率 | 保留数 | 分类错误率 |','|---|---:|---:|---:|']
        for name,v in results.items():
            for coverage,item in v['retrospective_coverage'].items():lines.append(f"| {name} | {coverage} | {item['accepted']} | {item['error_rate']:.6f} |")
        lines+=['','## 运行预算与局限','',
            f"- 240接受，started={sum(starts.values())}，显式失败={sum(r['event']=='failed' for r in attempts)}，unknown=0；global720、target3。",
            '- 累计失败30为停止新提交阈值，已预留inflight完整drain最多32终态；缺失或失败usage未知，不估货币账单。',
            '- 历史元数据按类/标签抽样；本轮预测和评价标签在最终系统冻结前封存，不宣称元数据从未查看。',
            '- 来源/字节隔离不证明物理零件/近重复独立，公开基准预训练污染未知，teacher暴露需单列审计。',
            '- 类均衡240不等于生产先验；bootstrap不含calibration拟合、provider或支持选择方差。','']
        publish(stage/'REPORT.md','\n'.join(lines),plain=True)
        completion=dict(schema=SCHEMA,status='complete',expected=EXPECTED,completed=EXPECTED,
            contract_sha256=sha(stage/'contract.json'),bundle_sha256=sha(stage/'final_system_bundle.json'),
            prediction_freeze_sha256=sha(stage/'prediction_freeze.json'),metrics_sha256=sha(stage/'metrics.json'),
            report_sha256=sha(stage/'REPORT.md'),no_refit=True,no_threshold_optimization=True,no_reselection=True,
            inference_label_reads=0,evaluator_boolean_labels_opened=True,masks_opened=False,confirmation_opened=True)
        publish(stage/'completion_certificate.json',completion)
        print(json.dumps(dict(status='complete',accepted=EXPECTED,primary='calibrated',secondary='raw',
            metrics=str(target),report=str(stage/'REPORT.md'),no_reselection=True)),flush=True)
    return 0


def self_test(args):
    host_gate();fit=dict(feature='logit(clip(p,1e-6,1-1e-6))',C=1.,penalty='l2',solver='lbfgs',max_iter=1000,random_state=42,
        classes=[0,1],heldout_labels_used_for_fit=False,confirmation_opened=False,
        calibration_label_sha256=calibration.LABEL_HASHES['calibration'],coefficient=[[1.]],intercept=[0.])
    values=apply_fit([0,.25,.5,.75,1],fit)
    require(np.allclose(values,[1e-6,.25,.5,.75,1-1e-6],rtol=0,atol=1e-12),'Fixed logistic mapping failed')
    require(settled_http_failure(dict(error='http_502',http_status=502)),'ExplicitHTTP502 classification failed')
    for row in [dict(error='timeout_unknown_server_completion',http_status=None),
        dict(error='transport_error_unknown_server_completion',http_status=None),dict(error='invalid_typed_response_or_model',http_status=200)]:
        require(not settled_http_failure(row),'Ambiguous completion incorrectly settled')
    bundle=dict(system=dict(shots=0,supports={}))
    body=payload(dict(source_id='fixture',category='pcb1'),bundle,dict(fixture=b'synthetic'),'a'*64)
    require(set(body['state'])=={'query','category','normal_references'} and len(body['assets'])==1,'Privileged metadata payload leakage')
    require(MAX_STARTED==EXPECTED*3 and MAX_FAILED==30,'Declared persistent budgets differ')
    # Exercise the real ledger validator with synthetic bodies and no network.
    contract=dict(records=[dict(source_id='fixture',category='pcb1')],bundle_sha256='b'*64)
    ch=digest(canonical(contract));request=digest(canonical(payload(contract['records'][0],bundle,dict(fixture=b'synthetic'),ch)))
    def invocation(name,started=0,failed=0,retry=False):
        return dict(invocation_id=name,contract_sha256=ch,bundle_sha256='b'*64,explicit_invocation=True,
            prior_cumulative_started=started,prior_cumulative_failed=failed,no_unknown_completion=True,retry_failed=retry)
    def intent(number,name):
        return dict(source_id='fixture',attempt=number,event='started',invocation_id=name,contract_sha256=ch,
            request_sha256=request,bundle_sha256='b'*64)
    def accepted(number,name):
        return dict(source_id='fixture',category='pcb1',attempt=number,invocation_id=name,contract_sha256=ch,
            request_sha256=request,bundle_sha256='b'*64,probability=.2,response_model='jev-1.14.0',usage={},seconds=1.)
    def check(invocations,attempts,predictions,should_block=False,unknown_count=0):
        with tempfile.TemporaryDirectory(prefix='confirmation-synthetic-') as temporary:
            stage=Path(temporary)
            for name,rows in [('invocations',invocations),('attempts',attempts),('predictions',predictions)]:
                (stage/(name+'.jsonl')).write_text(jsonl(rows),encoding='utf-8')
            try:
                result=audit_logs(stage,contract,bundle,dict(fixture=b'synthetic'))
            except ValueError:
                require(should_block,'Synthetic ledger unexpectedly blocked');return
            require(not should_block,'Synthetic unsafe ledger was accepted')
            require(len(result[-1])==unknown_count,'Synthetic unknown accounting differs')
    first=intent(1,'i1');second=intent(2,'i2')
    http=dict(first,event='failed',http_status=502,error='http_502')
    ambiguous=dict(first,event='failed',http_status=None,error='timeout_unknown_server_completion')
    success=dict(first,event='succeeded',http_status=200)
    check([invocation('i1')],[first,success],[accepted(1,'i1')])
    check([invocation('i1')],[first],[accepted(1,'i1')])
    check([invocation('i1')],[first],[],unknown_count=1)
    check([invocation('i1')],[first,ambiguous],[],unknown_count=1)
    check([invocation('i1')],[first,http],[accepted(1,'i1')],should_block=True)
    check([invocation('i1'),invocation('i2',1,1,True)],
        [first,http,second,dict(second,event='succeeded',http_status=200)],[accepted(2,'i2')])
    check([invocation('i1'),invocation('i2',1,1,False)],[first,http,second],[],should_block=True)
    check([invocation('i1'),invocation('i2',1,1,True)],[first,ambiguous,second],[],should_block=True)
    print(json.dumps(dict(status='passed',synthetic_only=True,network_calls=0,confirmation_metadata_or_images_read=False,
        checks=['fixed_logistic_mapping','clip_extremes','explicitHTTP_only_settled','timeout_transport_invalid_unknown',
            'no_privileged_payload','persistent_budget_constants','durable_success','missing_terminal_unknown',
            'failed_and_accepted_conflict_rejected','explicit_newinvocation_retry_only','ambiguous_retry_rejected'])),flush=True);return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare-policy','freeze-final','open-inputs','dry-run','run','evaluate','self-test'):group.add_argument('--'+name,action='store_true')
    parser.add_argument('--retry-failed',action='store_true');parser.add_argument('--key-env',default='MODELGO_API_KEY')
    args=parser.parse_args();functions=dict(prepare_policy=prepare_policy,freeze_final=freeze_final,open_inputs=open_inputs,
        dry_run=dry_run,run=run,evaluate=evaluate,self_test=self_test)
    try:return next(function(args) for name,function in functions.items() if getattr(args,name))
    except Exception as error:
        print(json.dumps(dict(status='blocked',exception_class=type(error).__name__,
            reason=str(error) if isinstance(error,ValueError) else 'Inspect frozen confirmation gate; arbitrary exception suppressed')),file=sys.stderr);return 2


if __name__=='__main__':sys.exit(main())
