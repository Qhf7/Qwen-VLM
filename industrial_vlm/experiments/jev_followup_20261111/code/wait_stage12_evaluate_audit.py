#!/usr/bin/env python3
from pathlib import Path
from collections import Counter
import json,hashlib,socket,time,sys,subprocess,fcntl,datetime,os
ROOT=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111');STAGE=ROOT/'stage12/run';PY='/root/autodl-tmp/conda-envs/qwen3vl2/bin/python'
assert socket.gethostname()=='autodl-container-132444827f-9c594038'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def publish(p,d):
 assert not p.exists()
 with p.open('x') as f:f.write(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
claim=ROOT/'stages/12_evaluation_waiter_claim.json'
publish(claim,dict(schema='jev-stage12-completion-waiter-v1',status='started',timestamp_utc=now(),hostname=socket.gethostname(),pid=os.getpid(),runner_sha256=sha(ROOT/'code/isolated_calibration.py'),independent_auditor_sha256=sha(ROOT/'code/audit_stage12_results.py'),poll_seconds=10,lock_release_buffer_seconds=3,no_api_calls=True,no_partial_predictions_or_labels_read=True,confirmation_opened=False))
print(json.dumps(dict(event='waiting_for_complete240',timestamp_utc=now())),flush=True)
deadline=time.monotonic()+7200
while True:
 status=read(STAGE/'status.json')
 if status.get('status')=='complete' and status.get('completed')==status.get('expected')==240:
  with (STAGE/'.run.lock').open('a') as lock:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:time.sleep(3);continue
   fcntl.flock(lock,fcntl.LOCK_UN)
  time.sleep(3)
  status=read(STAGE/'status.json')
  assert status['status']=='complete' and status['completed']==status['expected']==240
  break
 if time.monotonic()>deadline:raise TimeoutError('Complete240 not observed within bounded2h waiter; preserve all data')
 time.sleep(10)
assert sha(ROOT/'code/isolated_calibration.py')=='d8489c953ec4cb5e9d7804643d35f43d5ad0801211abc118bcc8e0003fec37d1'
assert sha(ROOT/'code/audit_stage12_results.py')==read(ROOT/'stages/12_supplement_methodology_freeze.json')['auditor_sha256']
c=read(STAGE/'contract.json');expected={r['source_id'] for r in c['records']};intents=set();term=set();success=set();counts=Counter()
for line in (STAGE/'attempts.jsonl').open():
 if not line.strip():continue
 r=json.loads(line);identity=(r['source_id'],r['attempt']);counts[r['event']]+=1
 assert r['source_id'] in expected and 1<=r['attempt']<=3
 if r['event']=='started':assert identity not in intents;intents.add(identity)
 else:
  assert identity in intents and identity not in term;term.add(identity)
  if r['event']=='failed':assert r.get('http_status')==502 and r.get('error')=='http_502'
  else:assert r['event']=='succeeded';success.add(r['source_id'])
assert len(expected)==len(success)==240 and intents==term
assert status['interrupted_unknown_attempts']==0 and status['exhausted_unaccepted']==0
publish(ROOT/'stages/12_pre_evaluation_completion_gate.json',dict(schema='jev-stage12-completion-before-label-gate-v1',status='passed',timestamp_utc=now(),hostname=socket.gethostname(),predictions_sha256=sha(STAGE/'predictions.jsonl'),attempts_sha256=sha(STAGE/'attempts.jsonl'),accepted240=True,unknown0=True,all_failed_attempts_explicit_http502=True,attempt_counts=dict(counts),lock_released_and3s_buffer=True,partial_prediction_values_opened=False,label_files_opened=False,confirmation_opened=False,no_api_calls=True))
log=ROOT/'stages/12_completion_evaluation.log'
for name,args,marker in [
 ('frozen_evaluate',[PY,str(ROOT/'code/isolated_calibration.py'),'--root',str(ROOT),'--stage','stage12/run','--evaluate'],STAGE/'metrics.json'),
 ('independent_postaudit_and_supplement',[PY,str(ROOT/'code/audit_stage12_results.py')],ROOT/'stages/12_postexecution_audit.json')]:
 if marker.exists():
  print(json.dumps(dict(event='already_complete_skip',step=name,marker=str(marker))),flush=True);continue
 print(json.dumps(dict(event='starting',step=name,timestamp_utc=now())),flush=True)
 with log.open('a') as handle:
  handle.write('\n'+json.dumps(dict(step=name,timestamp_utc=now()))+'\n');handle.flush()
  result=subprocess.run(args,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
 assert result.returncode==0,(name,result.returncode)
 print(json.dumps(dict(event='complete',step=name,timestamp_utc=now())),flush=True)
report=read(STAGE/'metrics.json');audit=read(ROOT/'stages/12_postexecution_audit.json')
assert report['status']==audit['status']=='complete' and audit['failed']==0
publish(ROOT/'stages/12_completion_waiter_result.json',dict(schema='jev-stage12-completion-waiter-result-v1',status='complete',timestamp_utc=now(),hostname=socket.gethostname(),metrics_sha256=sha(STAGE/'metrics.json'),audit_sha256=sha(ROOT/'stages/12_postexecution_audit.json'),supplement_sha256=sha(STAGE/'STATISTICAL_SUPPLEMENT.json'),accepted=240,independent_passed=audit['passed'],independent_failed=0,confirmation_opened=False,no_api_calls=True))
print(json.dumps(dict(event='all_complete',accepted=240,independent_passed=audit['passed'],timestamp_utc=now())),flush=True)

