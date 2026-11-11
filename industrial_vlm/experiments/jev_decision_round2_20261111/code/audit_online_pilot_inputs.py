#!/usr/bin/env python3
"""Independent label-free input audit for real crop-gate pilot."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib,socket,datetime,base64,io,argparse,math
from PIL import Image
R=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_decision_round2_20261111');S=R/'online_route_pilot';OLD=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111')
sys.path.insert(0,str(R/'code'))
import online_route_pilot as online
checks=[]
def check(n,v):
 checks.append(dict(name=n,passed=bool(v)))
 if not v:raise AssertionError(n)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []
def canon(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def main(phase):
 check('server',socket.gethostname()=='autodl-container-132444827f-9c594038');c,images=online.verify(S);proto=read(S/'protocol.json');old=read(OLD/'runs/modelgo_roi/contract.json');cand={r['source_id']:r for r in read(OLD/'stages/03_candidates.json')['records']};oldpred={(r['source_id'],r['arm']):r for r in rows(OLD/'runs/modelgo_roi/predictions.jsonl')};expected=sorted(old['records'],key=lambda r:(hashlib.sha256(r['source_id'].encode()).hexdigest(),r['source_id']))[:24]
 check('24_hash_no_replacement',[r['source_id'] for r in c['records']]==[r['source_id'] for r in expected] and len({r['source_id'] for r in c['records']})==24)
 check('fixed_version_source',c['model']=='jev-1.14.0' and c['runner_sha256']==sha(R/'code/online_route_pilot.py') and c['protocol_sha256']==sha(S/'protocol.json') and c['no_label_or_mask_reads'] and c['stage13_data_read'] is False)
 check('budget_caps',online.MAX_STARTS==216 and online.MAX_FAILURES==30 and proto['successful_targets_upper_bound']==72 and proto['max_started_per_target']==3 and proto['workers']==3 and proto['unknown_completion_hardstop'])
 for i,r in enumerate(c['records']):
  sid=r['source_id'];q=images[sid]['query'];crop=images[sid]['crop'];v=cand[sid];b=r['bbox_original_norm'];area=(b[2]-b[0])*(b[3]-b[1]);check('candidate_'+str(i),r['bbox_original_norm']==v['bbox_original_norm'] and r['bbox_448_norm']==v['bbox_448_norm'] and r['original_sha256']==v['original_sha256'] and r['crop_bounds_original']==v['crop_bounds_original'])
  check('eligibility_'+str(i),r['eligible']==(q!=crop)==(hashlib.sha256(q).hexdigest()!=hashlib.sha256(crop).hexdigest()) and math.isclose(area,r['area']))
  check('oldscore_control_only_'+str(i),r['historical_duplicate_probability']==oldpred[sid,'duplicate']['probability'])
  for name,raw in [('query',q),('crop',crop)]:
   a=r['assets'][name];im=Image.open(io.BytesIO(raw));im.load();check('asset_'+str(i)+'_'+name,im.size==(448,448) and im.mode=='RGB' and hashlib.sha256(raw).hexdigest()==a['png_sha256'] and len(raw)==a['png_bytes'])
 check('commonK',c['eligible_count']==sum(r['eligible'] for r in c['records']) and c['K']==min(8,c['eligible_count']))
 phasec=read(S/phase/'contract.json');ch=hashlib.sha256(canon(phasec)).hexdigest();lookup={r['source_id']:r for r in c['records']};hashes=read(S/phase/'request_hashes.json')
 for i,record in enumerate(phasec['schedule']):
  row=lookup[record['source_id']];arm=record['arm'];body=online.payload(row,arm,images,ch);encoded=canon(body);check('request_hash_'+str(i),hashlib.sha256(encoded).hexdigest()==hashes[row['source_id']+'|'+arm])
  check('payload_noprivileged_'+str(i),all(t not in encoded for t in [row['source_id'].encode(),b'source_id',b'historical_duplicate_probability',b'is_anomalous',b'label',b'teacher_split',b'GT_mask',b'original_path']))
  if arm=='gate':
   check('gate_single_query_'+str(i),set(body)=={'model','state','assets','questions'} and len(body['assets'])==1 and set(body['state'])=={'query','category','candidate_bbox_in_query','candidate_bbox_in_original'} and set(body['questions'])=={'crop_needed'} and body['questions']['crop_needed']==dict(type='noul',instructions=online.GATE_INSTRUCTIONS,criteria=online.GATE_CRITERIA))
   asset=body['assets'][body['state']['query']];check('gate_no_crop_'+str(i),base64.b64decode(asset['data_base64'])==images[row['source_id']]['query'])
  else:
   check('final_fixed_question_'+str(i),body['questions']=={'anomaly':dict(type='noul',instructions=online.api.INSTRUCTIONS,criteria=online.api.CRITERIA)} and len(body['assets'])==2)
   second=base64.b64decode(body['assets'][body['state']['second_view']]['data_base64']);check('final_actual_view_'+str(i),second==images[row['source_id']]['crop' if arm=='roi' else 'query'])
 if phase=='gate':check('schedule_gate24',phasec['schedule']==[dict(source_id=r['source_id'],arm='gate') for r in c['records']])
 else:
  routing=read(S/'routing.json');gf=read(S/'gate/prediction_freeze.json');gaterows=rows(S/'gate/predictions.jsonl');gate={(r['source_id'],r['arm']):r for r in gaterows};check('gatefreeze_before_final',len(gate)==24 and gf['predictions_sha256']==sha(S/'gate/predictions.jsonl') and gf['attempts_sha256']==sha(S/'gate/attempts.jsonl') and gf['contract_sha256']==sha(S/'gate/contract.json') and phasec['routing_sha256']==sha(S/'routing.json'))
  eligible=[r for r in c['records'] if r['eligible']];K=c['K'];neutral=lambda sid:hashlib.sha256(('jev-live-route-pilot-neutral-v1|'+sid).encode()).hexdigest();route=dict(learned_gate=[r['source_id'] for r in sorted(eligible,key=lambda r:(-gate[r['source_id'],'gate']['probability'],neutral(r['source_id']),r['source_id']))[:K]],matched_random=[r['source_id'] for r in sorted(eligible,key=lambda r:(hashlib.sha256(('jev-live-route-random-v1|20261111|'+r['source_id']).encode()).hexdigest(),r['source_id']))[:K]],historical_uncertainty=[r['source_id'] for r in sorted(eligible,key=lambda r:(abs(r['historical_duplicate_probability']-.5),neutral(r['source_id']),r['source_id']))[:K]],geometry_small=[r['source_id'] for r in sorted(eligible,key=lambda r:(r['area'],neutral(r['source_id']),r['source_id']))[:K]])
  check('four_policy_selection_independent',route==routing['routes'] and all(len(v)==K and len(set(v))==K for v in route.values()));union=sorted(set().union(*map(set,route.values())));check('union_trueROI',union==routing['union_roi'] and len(union)<=24 and len(phasec['schedule'])==24+len(union) and len(routing['crop_executions'])==len(union))
  for i,e in enumerate(routing['crop_executions']):
   raw=(S/e['executed_crop_path']).read_bytes();check('executed_crop_'+str(i),e['source_id'] in union and sha(S/e['executed_crop_path'])==e['png_sha256'] and len(raw)==e['png_bytes'] and raw==images[e['source_id']]['crop'] and e['GT_or_mask_opened'] is False)
  check('all24newduplicates',Counter(r['arm'] for r in phasec['schedule'])==Counter(duplicate=24,roi=len(union)))
 runtime_phase,(complete,starts,attempts,hashes2,unknown)=online.phase_data(S,phase,c,images);check('runtime_phase_binding',runtime_phase==phasec);check('api_unknown_free',not unknown and max(starts.values(),default=0)<=3)
 result=dict(schema='jev-online-pilot-independent-input-audit-v1',status='complete',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=socket.gethostname(),phase=phase,passed=sum(x['passed'] for x in checks),failed=sum(not x['passed'] for x in checks),checks=checks,auditor_sha256=sha(Path(__file__)),runner_sha256=sha(R/'code/online_route_pilot.py'),protocol_sha256=sha(S/'protocol.json'),contract_sha256=sha(S/'contract.json'),phase_contract_sha256=sha(S/phase/'contract.json'),n=24,eligible=c['eligible_count'],K=c['K'],api_snapshot=dict(accepted=len(complete),started=sum(starts.values()),failed=sum(r['event']=='failed' for r in attempts),unknown=len(unknown)),approved24_labels_read=False,GT_masks_read=False,stage13_data_read=False,new_api_calls=0,note='Gate semantics are need-for-closeup, not anomaly; gate score not used as final anomaly probability; exploratory previously viewed sample')
 with (S/(phase+'_independent_input_audit.json')).open('x') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2)+chr(10))
 print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['gate','final'],required=True);main(ap.parse_args().phase)
