#!/usr/bin/env python3
"""Independent stage10 input/geometry audit; no prediction/label reads."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib,struct,socket,datetime
ROOT=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111')
STAGE=ROOT/'stage10_resolution';REF=ROOT/'stage08_support'
sys.path.insert(0,str(ROOT/'code'))
import support_robustness as support
import resolution_ablation as res
checks=[]
def check(name,v):
 checks.append({'name':name,'passed':bool(v)})
 if not v:raise AssertionError(name)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
check('designated_server',socket.gethostname()=='autodl-container-132444827f-9c594038')
c=read(STAGE/'contract.json');p=read(STAGE/'protocol.json');rc=read(REF/'contract.json')
check('formal120_only',c['mode']=='formal_independent_resolution' and c['limit_queries'] is None and len(c['queries'])==len(c['schedule'])==120)
check('sources_frozen',sha(ROOT/'code/resolution_ablation.py')==c['runner_sha256']=='4458848524108d22220d1196cffef65967d06f730425e1cb73277e651b1db9af' and sha(ROOT/'code/support_robustness.py')==c['support_source_sha256']=='fc6ddc02eb2e2a8c0556f3843a5e73321f9d9bfcf1b5f14314de1aea2077d0bd' and sha(ROOT/'code/roi_api.py')==c['transport_source_sha256']=='4f7b5ed1f703b3751f9fd4bb8c86131d684cec0bed3a9011f5108621e7aa9e01')
check('reference448_contract_frozen',c['reference_stage']=='stage08_support' and c['reference_contract_sha256']==sha(REF/'contract.json'))
check('protocol_exact',sha(STAGE/'protocol.json')==c['protocol_sha256'] and p==res.protocol())
check('dependency_snapshot',c['dependencies']==support.dependencies()==rc['dependencies'])
check('candidate_provenance_hash',c['candidates_sha256']==sha(ROOT/'stages/03_candidates.json'))
check('same_query_order_and_sources',[r['source_id'] for r in c['queries']]==[r['source_id'] for r in rc['queries']])
check('same_wording_model_zero_references',p['instructions']==support.INSTRUCTIONS and p['criteria']==support.CRITERIA and p['normal_references']==0 and c['model']==rc['model']=='jev-1.14.0')
check('new320_and_reused448_design',p['new_targets']==120 and p['arms']==['size320_0shot','shared0_448'] and c['size']==320)
check('host_gate_in_resolution',"host_gate();root=args.root.resolve()" in (ROOT/'code/resolution_ablation.py').read_text())
check('retry_and_stop_limits',p['max_started_per_target']==3 and p['automatic_retries'] is False and p['consecutive_error_submission_stop']==3 and p['workers']==3)
check('no_gt_or_reference_pred_read_prepare_flags',c['no_gt_read_during_prepare'] is True and c['no_reference_predictions_read_during_prepare'] is True)
check('confirmation_closed',c['confirmation_opened'] is False and p['confirmation_opened'] is False)
images={};q={r['source_id']:r for r in c['queries']};refq={r['source_id']:r for r in rc['queries']}
for sid,r in q.items():
 rr=refq[sid]
 check('original_identity_'+sid[:16],all(r[k]==rr[k] for k in ['original_sha256','original_bytes','original_path','category']) and sid=='visa_'+r['original_sha256'])
 path=Path(r['original_path']);raw=path.read_bytes()
 check('original_bytes_'+sid[:16],len(raw)==r['original_bytes'] and hashlib.sha256(raw).hexdigest()==r['original_sha256'])
 a=r['asset320'];file=(STAGE/a['path']).resolve();png=file.read_bytes()
 check('320_asset_'+sid[:16],file.is_relative_to(STAGE) and len(png)==a['png_bytes'] and hashlib.sha256(png).hexdigest()==a['png_sha256'] and png[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',png[16:24])==(320,320) and a['size']==[320,320])
 check('320_render_reproduced_'+sid[:16],res.render(raw)[0]==png)
 a448=r['reference448'];b=(REF/a448['path']).read_bytes()
 check('448_reference_asset_'+sid[:16],a448['stage']=='stage08_support' and all(a448[k]==rr['asset'][k] for k in rr['asset']) and len(b)==a448['png_bytes'] and hashlib.sha256(b).hexdigest()==a448['png_sha256'] and struct.unpack('>II',b[16:24])==(448,448))
 images[sid]=png
schedule=[(r['source_id'],r['arm']) for r in c['schedule']]
check('schedule120_unique',len(schedule)==len(set(schedule))==120 and set(schedule)=={(sid,'size320_0shot') for sid in q})
ch=hashlib.sha256(canon(c)).hexdigest();hashes={}
for sid,r in q.items():
 body=res.payload(r,images,ch);raw=canon(body)
 check('payload_'+sid[:16],body['model']=='jev-1.14.0' and len(body['assets'])==1 and body['state']['normal_references']==[] and set(body['state'])=={'query','category','normal_references'} and body['questions']=={'anomaly':dict(type='noul',instructions=support.INSTRUCTIONS,criteria=support.CRITERIA)} and sid.encode() not in raw and b'source_id' not in raw and b'is_anomalous' not in raw and b'original_path' not in raw)
 hashes[sid]=hashlib.sha256(raw).hexdigest()
pth=STAGE/'attempts.jsonl';raw=pth.read_bytes() if pth.exists() else b'';lines=raw.splitlines();partial=bool(raw and not raw.endswith(b'\n'))
if partial:lines=lines[:-1]
starts=Counter();intents=set();term=set();events=Counter()
for line in lines:
 v=json.loads(line);sid=v['source_id'];key=(sid,v['attempt']);events[v['event']]+=1
 check('attempt_identity_'+str(sum(events.values())),sid in q and v['arm']=='size320_0shot' and v['contract_sha256']==ch and v['request_sha256']==hashes[sid])
 if v['event']=='started':
  starts[sid]+=1;check('start_sequence_'+str(sum(events.values())),starts[sid]==v['attempt'] and v['attempt']<=3 and key not in intents);intents.add(key)
 else:
  check('terminal_'+str(sum(events.values())),v['event'] in ['failed','succeeded'] and key in intents and key not in term);term.add(key)
check('finite_attempt_ceiling',sum(starts.values())<=360 and max(starts.values(),default=0)<=3)
result={'schema':'jev-stage10-independent-preexecution-audit-v1','status':'complete','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'passed':sum(v['passed'] for v in checks),'failed':sum(not v['passed'] for v in checks),'checks':checks,'contract_sha256':sha(STAGE/'contract.json'),'protocol_sha256':sha(STAGE/'protocol.json'),'auditor_sha256':sha(Path(__file__)),'design':{'new320_targets':120,'reused448_shared0':120,'excluded_reference_support_observations':480,'new_http_accepted_target':120,'max_started_per_target':3,'finite_max_started':360,'canvas_pixel_ratio':320**2/448**2},'attempt_snapshot':{'started':sum(starts.values()),'events':dict(events),'partial_line_ignored':partial,'inflight_or_unknown':len(intents-term)},'no_prediction_values_opened':True,'no_labels_opened':True,'gt_masks_opened':False,'confirmation_opened':False,'new_api_requests':0,'limits':['448 shared0 is earlier in time, not interleaved with320','Pixel ratio is not exact token or money ratio','Development120, not confirmation','No cumulative monetary/error stop beyond fixed per-target3 and3consecutive errors']}
out=ROOT/'stages/10_preexecution_audit.json';assert not out.exists()
with out.open('x') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False,indent=2))

