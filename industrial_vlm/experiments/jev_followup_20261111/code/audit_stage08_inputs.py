#!/usr/bin/env python3
"""Independent stage08 inputs and attempt-envelope audit; no prediction/label reads."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib,struct,socket,datetime,re
ROOT=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111')
STAGE=ROOT/'stage08_support'
DATA=Path('/root/autodl-tmp/industrial_vlm/datasets/VisA')
sys.path.insert(0,str(ROOT/'code'))
import support_robustness as support
import roi_api as api
checks=[]
def check(name,condition,detail=None):
    checks.append(dict(name=name,passed=bool(condition),detail=detail))
    if not condition:raise AssertionError(name)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
check('designated_server',socket.gethostname()=='autodl-container-132444827f-9c594038')
c=read(STAGE/'contract.json');p=read(STAGE/'protocol.json')
check('formal_mode_and_no_query_limit',c['mode']=='formal_independent_support_robustness' and c['limit_queries'] is None)
check('frozen_model_endpoint_geometry',c['model']==p['model']=='jev-1.14.0' and c['endpoint']==p['endpoint']=='https://api.modelgo.ai/v1/systemone' and c['size']==p['size']==448)
check('support_runner_frozen',sha(ROOT/'code/support_robustness.py')==c['runner_sha256']=='fc6ddc02eb2e2a8c0556f3843a5e73321f9d9bfcf1b5f14314de1aea2077d0bd')
check('transport_runner_frozen',sha(ROOT/'code/roi_api.py')==c['transport_source_sha256']=='4f7b5ed1f703b3751f9fd4bb8c86131d684cec0bed3a9011f5108621e7aa9e01')
check('protocol_frozen',sha(STAGE/'protocol.json')==c['protocol_sha256'] and p==support.protocol())
check('dependency_snapshot',c['dependencies']==support.dependencies())
for k,v in c['source_files'].items():check('source_file_sha_'+k,sha(Path(v['path']))==v['sha256'])
check('explicit_stage_host_gate',"host_gate(); root=args.root.resolve()" in (ROOT/'code/support_robustness.py').read_text())
check('retry_policy',p['max_started_per_target']==3 and p['automatic_retries'] is False and p['consecutive_error_submission_stop']==3)
check('bounded_transport_and_concurrency',p['workers']==3 and p['connect_timeout']==60 and p['read_timeout']==120)
check('confirmation_declared_closed',c['confirmation_opened'] is False and p['confirmation_opened'] is False)
check('no320_arm',p['width320_included'] is False)
trainpath=Path(c['source_files']['train']['path'])
train=[]
for line in trainpath.open():
 if line.strip():
  v=json.loads(line);train.append({k:v[k] for k in ('source_id','image','category','is_anomalous')})
u={r['source_id']:r for r in train}
check('train768_double_exposure',len(train)==1536 and len(u)==768 and set(Counter(r['source_id'] for r in train).values())=={2})
check('explicit_boolean_training_labels',all(type(r['is_anomalous']) is bool for r in train))
normal={sid:r for sid,r in u.items() if not r['is_anomalous']}
check('normal_bank384_12x32',len(normal)==384 and len(Counter(r['category'] for r in normal.values()))==12 and set(Counter(r['category'] for r in normal.values()).values())=={32})
bank={r['source_id']:r for r in c['normal_bank_originals']}
check('contract_bank_matches_training_normals',set(bank)==set(normal) and len(bank)==384)
q={r['source_id']:r for r in c['queries']}
check('query120_unique',len(c['queries'])==len(q)==120)
check('query_vs_all_train_source_and_encoded_bytes_disjoint',not set(q)&set(u) and not {s[5:] for s in q}&{s[5:] for s in u})
for sid,r in bank.items():
 t=normal[sid];path=Path(r['original_path']).resolve()
 check('bank_meta_'+sid[:18],path.is_relative_to(DATA) and str(path)==t['image'] and r['category']==t['category'] and sid=='visa_'+r['original_sha256'])
 check('bank_bytes_'+sid[:18],path.stat().st_size==r['original_bytes'] and sha(path)==r['original_sha256'])
for sid,r in q.items():
 path=Path(r['original_path']).resolve()
 check('query_bytes_'+sid[:18],path.is_relative_to(DATA) and sid=='visa_'+r['original_sha256'] and path.stat().st_size==r['original_bytes'] and sha(path)==r['original_sha256'])
check('byte_unique_bank',len({r['original_sha256'] for r in bank.values()})==384)
check('query_byte_unique',len({r['original_sha256'] for r in q.values()})==120)
check('query_bank_byte_disjoint',not {r['original_sha256'] for r in bank.values()}&{r['original_sha256'] for r in q.values()})
selected={}
for seed in (43,44):
 recomputed=support.select_supports(list(bank.values()),seed)
 for cat,rs in recomputed.items():
  actual=c['supports'][str(seed)][cat]
  check('support_selection_'+str(seed)+'_'+cat,[r['source_id'] for r in actual]==[r['source_id'] for r in rs] and len(actual)==4)
  for r in actual:
   check('support_from_bank_'+str(seed)+'_'+r['source_id'][:18],all(r[k]==bank[r['source_id']][k] for k in ('category','original_path','original_sha256','original_bytes')))
   selected[r['source_id']]=r
check('selected90_unique_supports',len(selected)==c['n_unique_selected_supports']==90)
check('cross_seed_overlap_matches',{cat:len({r['source_id'] for r in c['supports']['43'][cat]}&{r['source_id'] for r in c['supports']['44'][cat]}) for cat in c['supports']['43']}==c['cross_seed_support_overlap'])
sources=dict(q);sources.update(selected);assets={}
for sid,r in sources.items():
 a=r['asset'];path=(STAGE/a['path']).resolve();raw=path.read_bytes()
 check('png_asset_'+sid[:18],path.is_relative_to(STAGE.resolve()) and len(raw)==a['png_bytes'] and hashlib.sha256(raw).hexdigest()==a['png_sha256'] and raw[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',raw[16:24])==(448,448) and a['size']==[448,448])
 assets[sid]=raw
arms=('shared0','s43_1','s43_4','s44_1','s44_4')
expected={(sid,arm) for sid in q for arm in arms};schedule=[(r['source_id'],r['arm']) for r in c['schedule']]
check('formal600_schedule',len(schedule)==len(set(schedule))==len(expected)==p['formal_targets']==600 and set(schedule)==expected)
check('each_arm120',dict(Counter(arm for _,arm in schedule))=={a:120 for a in arms})
contract_hash=hashlib.sha256(canon(c)).hexdigest();hashes={}
for sid,arm in schedule:
 payload=support.payload(q[sid],arm,c,assets,contract_hash);body=canon(payload)
 seed,shots=support.arm_details(arm)
 check('request_envelope_'+sid[:14]+'_'+arm,set(payload)=={'model','state','assets','questions'} and set(payload['state'])=={'query','category','normal_references'} and payload['model']=='jev-1.14.0' and len(payload['assets'])==shots+1 and len(payload['state']['normal_references'])==shots and payload['state']['query'] in payload['assets'] and all(k in payload['assets'] for k in payload['state']['normal_references']))
 check('request_task_and_no_identity_'+sid[:14]+'_'+arm,payload['questions']=={'anomaly':{'type':'noul','instructions':p['instructions'],'criteria':p['criteria']}} and sid.encode() not in body and b'source_id' not in body and b'is_anomalous' not in body and b'original_path' not in body and b'candidate_bbox' not in body)
 hashes[(sid,arm)]=hashlib.sha256(body).hexdigest()
# Read only attempt envelope metadata. Never open predictions.jsonl or label files.
ledger_path=STAGE/'attempts.jsonl';ledger_raw=ledger_path.read_bytes() if ledger_path.exists() else b''
lines=ledger_raw.splitlines();partial=bool(ledger_raw and not ledger_raw.endswith(b'\n'))
if partial:lines=lines[:-1]
starts=Counter();intents=set();terminal=set();events=Counter()
for line in lines:
 row=json.loads(line);pair=(row['source_id'],row['arm']);key=(pair,row['attempt']);events[row['event']]+=1
 check('attempt_identity_'+str(sum(events.values())),pair in expected and row['contract_sha256']==contract_hash and row['request_sha256']==hashes[pair])
 if row['event']=='started':
  starts[pair]+=1
  check('attempt_sequence_'+str(sum(events.values())),starts[pair]==row['attempt'] and row['attempt']<=3 and key not in intents);intents.add(key)
 else:
  check('attempt_terminal_'+str(sum(events.values())),row['event'] in ('succeeded','failed') and key in intents and key not in terminal);terminal.add(key)
check('max_started_per_target_snapshot',max(starts.values(),default=0)<=3)
check('finite_started_total_ceiling',sum(starts.values())<=600*3)
result={
 'schema':'jev-stage08-independent-preexecution-audit-v1','status':'complete','audit_scope':'Frozen contract, source/byte integrity, bank selection, PNG envelope and attempt metadata; performed while run may be active; no predictions/labels/metrics read',
 'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),
 'contract_sha256':sha(STAGE/'contract.json'),'contract_canonical_sha256':contract_hash,'protocol_sha256':sha(STAGE/'protocol.json'),'auditor_sha256':sha(Path(__file__)),
 'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks),'checks':checks,
 'design':{'query_sources':120,'formal_targets':600,'arms':list(arms),'training_sources':768,'normal_bank':384,'normal_per_category':32,'support_seeds':[43,44],'unique_selected_supports':90,'resolution':448,'per_target_max_started':3,'finite_total_max_started':1800},
 'attempt_snapshot':{'read_complete_lines':len(lines),'partial_last_line_ignored':partial,'events':dict(events),'started':sum(starts.values()),'max_target_started':max(starts.values(),default=0),'inflight_or_unknown_at_snapshot':len(intents-terminal),'ledger_prefix_sha256':hashlib.sha256(b'\n'.join(lines)+b'\n').hexdigest()},
 'no_prediction_values_opened':True,'no_query_labels_opened':True,'ground_truth_masks_opened':False,'confirmation_files_opened':False,'new_api_requests':0,
 'limits':['Not a post-execution metric audit','No independent accumulated-error or monetary stop in frozen support runner;3 consecutive errors and3starts per target bound attempts, with inflight drain','Transport4f7 lacks host gate; support runner itself enforces hostname','Two support seeds share the same120 previously viewed development queries','Cross-seed support overlap exists and is disclosed,1shot nested in4shot perseed','Historical phase1 Pillow10.3 vs this12.3 means independent shared0 is the matched renderer control'],
}
out=ROOT/'stages/08_preexecution_audit.json'
assert not out.exists(),'Do not overwrite existing audit'
with out.open('x') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('checks',)},ensure_ascii=False,indent=2))

