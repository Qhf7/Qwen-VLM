#!/usr/bin/env python3
"""One unique complete-only evaluation claim; no API and no partial outcomes."""
from pathlib import Path
from collections import Counter
import json,sys,time,fcntl,os,socket,subprocess,datetime,hashlib
ROOT=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111');STAGE=ROOT/'stage13_confirmation';PYTHON='/root/autodl-tmp/conda-envs/qwen3vl2/bin/python'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
 with p.open('x') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2)+chr(10));f.flush();os.fsync(f.fileno())
def lockfree():
 with (STAGE/'.run.lock').open('a') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return False
  fcntl.flock(f,fcntl.LOCK_UN);return True
assert socket.gethostname()=='autodl-container-132444827f-9c594038'
claim=ROOT/'stages/13_evaluation_waiter_claim.json';put(claim,dict(status='claimed',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),host=socket.gethostname(),exclusive_final_evaluator='stage2_evaluation',waiter_sha256=sha(Path(__file__)),no_api=True,no_partial_metric_or_label_reads=True))
print(json.dumps(dict(status='waiting_full240_and_released_lock',pid=os.getpid())),flush=True)
last=None;deadline=time.time()+7200
while time.time()<deadline:
 s=read(STAGE/'status.json');snap=(s['status'],s['completed'],s['cumulative_started'],s['cumulative_failed'],s['unknown'])
 if snap!=last:print(json.dumps(dict(status=snap[0],completed=snap[1],started=snap[2],failed=snap[3],unknown=snap[4])),flush=True);last=snap
 if s['status']=='complete' and s['completed']==s['expected']==240 and s['unknown']==0 and lockfree():
  time.sleep(3)
  s=read(STAGE/'status.json')
  if not(s['status']=='complete' and s['completed']==240 and s['unknown']==0 and lockfree()):continue
  # Only metadata of the durable ledger is inspected here; probabilities are untouched.
  events=[json.loads(l) for l in (STAGE/'attempts.jsonl').open() if l.strip()];starts={(r['source_id'],r['attempt']) for r in events if r['event']=='started'};terminal=[r for r in events if r['event']!='started'];ends={(r['source_id'],r['attempt']) for r in terminal};counts=Counter(r['event'] for r in events);failed=[r for r in terminal if r['event']=='failed']
  assert starts==ends and len(ends)==len(terminal) and counts['succeeded']==240 and len(failed)==s['cumulative_failed'] and counts['started']==s['cumulative_started']
  assert all(r.get('error')=='http_502' and r.get('http_status')==502 for r in failed)
  gate=dict(status='complete_gate_passed',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted=240,started=counts['started'],failed=counts['failed'],unknown=0,lock_released=True,settle_seconds=3,labels_read=False,partial_probabilities_read=False,status_sha256=sha(STAGE/'status.json'),attempts_sha256=sha(STAGE/'attempts.jsonl'),predictions_sha256=sha(STAGE/'predictions.jsonl'))
  put(ROOT/'stages/13_pre_evaluation_completion_gate.json',gate)
  with (ROOT/'stages/13_completion_evaluation.log').open('x') as log:
   for argv in [[PYTHON,str(ROOT/'code/final_confirmation.py'),'--root',str(ROOT),'--evaluate'],[PYTHON,str(ROOT/'code/audit_stage13_results.py')]]:
    p=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT);log.flush();assert p.returncode==0,'Evaluation/audit failed; inspect exclusive log, no automatic repeat'
  result=dict(status='complete',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),metrics_sha256=sha(STAGE/'metrics.json'),postexecution_audit_sha256=sha(ROOT/'stages/13_postexecution_audit.json'),supplement_sha256=sha(STAGE/'STATISTICAL_SUPPLEMENT.json'),completion_certificate_sha256=sha(STAGE/'completion_certificate.json'),no_api_requests=True,no_refit_or_selection=True)
  put(ROOT/'stages/13_completion_waiter_result.json',result);print(json.dumps(result),flush=True);sys.exit(0)
 time.sleep(10)
raise TimeoutError('Twohour evaluationwaiter deadline; no label read or automatic API request')
