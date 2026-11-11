#!/usr/bin/env python3
"""Input-only independent audit; never opens protected/legacy labels or masks."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib,socket,datetime,struct,io
from PIL import Image
ROOT=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111')
sys.path.insert(0,str(ROOT/'code'))
import final_confirmation as final
checks=[]
def check(n,v):
 checks.append(dict(name=n,passed=bool(v)))
 if not v:raise AssertionError(n)
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def main():
 check('designated_server',socket.gethostname()=='autodl-container-132444827f-9c594038')
 stage=ROOT/'stage13_confirmation';c=read(stage/'contract.json');bundle=read(stage/'final_system_bundle.json');pre=read(stage/'preprotocol.json');marker=read(stage/'protected_opening.json');opened=read(stage/'protected_opening_complete.json');status=read(stage/'status.json')
 check('runner_frozen',sha(ROOT/'code/final_confirmation.py')==c['runner_sha256']==bundle['source_sha256']['final_confirmation']=='ba312df2a904122e1a9dab72c62fe01d5beed40dbe507e1091bc8c21e8366bc4')
 check('bundle_frozen',sha(stage/'final_system_bundle.json')==c['bundle_sha256']==marker['bundle_sha256']=='72ab9b8706cdf77c102f7fb53c372e3a183ecdf69e67599d9d58732838fd392e')
 check('contract_frozen',sha(stage/'contract.json')=='60ac5b7c7d8a8e69e07bb26817751c19c6aff25870d7998eb7bdbbbf3ab0645c')
 check('preprotocol_fixed',sha(stage/'preprotocol.json')==bundle['preprotocol_sha256']==marker['preprotocol_sha256']=='1630df051d37301944cd30baae2a7e4107e4684edcf43b0e90dbe0fa37a57986')
 check('preprotocol_before_any12response',pre['timestamp_utc']<min(r['timestamp_utc'] for r in rows(ROOT/'stage12/run/predictions.jsonl')))
 check('bundle_primary_secondary_fixed',bundle['primary']=='calibrated' and bundle['secondary']=='raw' and bundle['threshold']==.5 and bundle['no_new_fit_or_parameter_choice'] and bundle['selection_independent_of_heldout_metrics'])
 check('frozen_system4shot43_448',c['model']=='jev-1.14.0' and c['size']==448 and c['shots']==4 and c['support_seed']==43 and c['expected']==240)
 check('coefficient_no_refit',bundle['calibrator']==read(ROOT/'stage12/run/calibrator.json') and bundle['calibrator']['heldout_labels_used_for_fit'] is False)
 check('opening_chain',marker['status']=='one_time_opening_reserved' and marker['no_reselection_or_second_opening'] and opened['status']=='input_label_broker_complete' and opened['contract_sha256']==sha(stage/'contract.json') and opened['opening_marker_sha256']==c['opening_marker_sha256']==sha(stage/'protected_opening.json') and opened['inputs_sha256']==c['input_sha256']==sha(stage/'inputs.jsonl') and opened['labels_sha256']==c['protected_label_sha256'])
 check('input_no_labels_masks',c['broker_only_labels_opened'] and c['inference_labels_opened'] is False and c['masks_opened'] is False and opened['masks_opened'] is False and c['no_sample_replacement'])
 rr=rows(stage/'inputs.jsonl');check('records_exact240',rr==c['records'] and len(rr)==240 and len({r['source_id'] for r in rr})==240 and c['schedule']==[r['source_id'] for r in rr])
 check('12categories20each',len(Counter(r['category'] for r in rr))==12 and set(Counter(r['category'] for r in rr).values())=={20})
 provenance={r['source_id']:r for r in c['render_provenance']};check('provenance240',len(provenance)==len(c['render_provenance'])==240)
 old=read(ROOT/'stage08_support/contract.json');train=rows(Path(old['source_files']['train']['path']));prior=read(ROOT/'stage12/run/contract.json')['records'];forbidden={r['source_id'] for r in train+old['queries']+prior};forbidden_sha={s[5:] for s in forbidden}
 ids={r['source_id'] for r in rr};originals={r['original_sha256'] for r in provenance.values()};check('source_byte_disjoint',not ids&forbidden and not originals&forbidden_sha and len(originals)==240 and c['source_and_byte_overlap']==opened['source_and_byte_overlap']==0)
 images={}
 for i,r in enumerate(rr):
  sid=r['source_id'];a=r['asset'];p=(stage/a['path']).resolve();raw=p.read_bytes();v=provenance[sid]
  check('row_schema_'+str(i),set(r)=={'source_id','category','asset'} and set(a)=={'path','png_sha256','png_bytes','size','original_size','oriented_size','resized_size','padding'})
  check('provenance_identity_'+str(i),sid=='visa_'+v['original_sha256'] and v['png_sha256']==a['png_sha256'] and v['original_bytes']>0)
  check('asset_bytes_'+str(i),p.is_relative_to((stage/'assets').resolve()) and hashlib.sha256(raw).hexdigest()==a['png_sha256'] and len(raw)==a['png_bytes'] and raw[:8]==bytes.fromhex('89504e470d0a1a0a') and struct.unpack('>II',raw[16:24])==(448,448) and a['size']==[448,448])
  im=Image.open(io.BytesIO(raw));im.load();check('asset_decode_'+str(i),im.size==(448,448) and im.mode=='RGB');images[sid]=raw
 for cat,refs in bundle['system']['supports'].items():
  check('support4_'+cat,len(refs)==4 and len({r['source_id'] for r in refs})==4)
  for r in refs:
   a=r['asset'];p=(ROOT/'stage08_support'/a['path']).resolve();raw=p.read_bytes();check('support_bytes_'+cat+'_'+r['source_id'][:14],p.is_relative_to((ROOT/'stage08_support').resolve()) and hashlib.sha256(raw).hexdigest()==a['png_sha256']);images[r['source_id']]=raw
 ch=hashlib.sha256(canon(c)).hexdigest()
 for i,r in enumerate(rr):
  body=final.payload(r,bundle,images,ch);encoded=canon(body)
  check('payload4ref_'+str(i),set(body)=={'model','state','assets','questions'} and set(body['state'])=={'query','category','normal_references'} and len(body['assets'])==5 and len(body['state']['normal_references'])==4 and body['questions']=={'anomaly':dict(type='noul',instructions=bundle['system']['instructions'],criteria=bundle['system']['criteria'])})
  check('payload_no_privileged_'+str(i),all(t not in encoded for t in [r['source_id'].encode(),b'source_id',b'is_anomalous',b'protected_labels',b'original_path',b'confirmation',b'calibration',b'heldout_eval']))
 c2,b2,im2=final.verify_inputs(ROOT);check('runtime_inputgate_no_labels',c2==c and b2==bundle and im2==images)
 complete,starts,attempts,hashes,unknown=final.audit_logs(stage,c,bundle,images);check('ledger_bound_no_unknown',not unknown and max(starts.values(),default=0)<=3 and sum(starts.values())<=720)
 result=dict(schema='jev-stage13-independent-input-audit-v1',status='complete',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=socket.gethostname(),passed=sum(x['passed'] for x in checks),failed=sum(not x['passed'] for x in checks),checks=checks,auditor_sha256=sha(Path(__file__)),contract_sha256=sha(stage/'contract.json'),bundle_sha256=sha(stage/'final_system_bundle.json'),preprotocol_sha256=sha(stage/'preprotocol.json'),opening_marker_sha256=sha(stage/'protected_opening.json'),records=240,categories=12,per_category=20,source_byte_overlap=0,shots=4,support_seed=43,size=448,api_snapshot=dict(accepted=len(complete),started=sum(starts.values()),failed=sum(x['event']=='failed' for x in attempts),unknown=len(unknown)),protected_label_bytes_opened=False,legacy_metadata_opened=False,gt_masks_opened=False,new_api_requests=0,limits=['Per-label count and original raw rendering identity checked by one-time privileged broker; independent input auditor checks committed provenance/source SHA, all rendered PNG SHA/RGB/dimensions and frozen broker code without reopening privileged legacy metadata','Source/byte isolation is not physical-part/near-duplicate independence or public pretraining nonexposure','Primary remains fixed calibrated; raw is secondary; no result-driven reselection'])
 out=ROOT/'stages/13_preexecution_audit.json'
 with out.open('x') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2)+chr(10))
 print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
