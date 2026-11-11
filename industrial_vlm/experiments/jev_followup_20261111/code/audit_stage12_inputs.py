#!/usr/bin/env python3
"""Independent stage12 input audit, no labels or predictions opened."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib,socket,datetime,struct
ROOT=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111')
sys.path.insert(0,str(ROOT/'code'))
import isolated_calibration as iso
import support_robustness as support
import roi_api as api
checks=[]
def check(n,v):
 checks.append({'name':n,'passed':bool(v)})
 if not v:raise AssertionError(n)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def rows(p):return [json.loads(l) for l in p.open() if l.strip()]
def main():
 check('server',socket.gethostname()=='autodl-container-132444827f-9c594038')
 stage=ROOT/'stage12_data';sysfile=ROOT/'stage12/system_contract.json'
 # Use only contract artifacts; never select/fit or read labels from this audit.
 if not sysfile.exists():print(json.dumps({'status':'pending','reason':'final system not frozen'}));return
 candidates=[p for p in [ROOT/'stage12/run',ROOT/'stage12_data',ROOT/'stage12_calibration',ROOT/'stage12'] if (p/'contract.json').exists()]
 if not candidates:print(json.dumps({'status':'pending','reason':'formal240 data contract not frozen'}));return
 stage=None
 for p in candidates:
  c0=read(p/'contract.json')
  if c0.get('mode')=='formal_source_disjoint_calibration':stage=p;break
 if stage is None:print(json.dumps({'status':'pending','reason':'onlypreflight contract exists'}));return
 c=read(stage/'contract.json');s=read(sysfile);plan=ROOT/'stage12_calibration_plan'
 check('runner_source',sha(ROOT/'code/isolated_calibration.py')==c['runner_sha256']=='d8489c953ec4cb5e9d7804643d35f43d5ad0801211abc118bcc8e0003fec37d1')
 check('model448_and_system',c['model']==s['model']=='jev-1.14.0' and c['size']==s['size']==448 and c['system_contract_sha256']==sha(sysfile))
 check('formal240_roles',len(c['records'])==len(c['schedule'])==240 and dict(Counter(r['role'] for r in c['records']))=={'calibration':120,'heldout_eval':120})
 check('input_only_record_fields',all(set(r)=={'source_id','category','role','original_path','original_sha256','original_bytes','asset'} for r in c['records']))
 for n,v in c['source_plan_files'].items():check('frozen_plan_sha_'+n,sha(plan/n)==v)
 inp={}
 for role in ['calibration','heldout_eval']:
  rr=rows(plan/(role+'_inputs.jsonl'))
  check('input_no_labels_'+role,all(set(r)=={'source_id','category','image','original_sha256','role'} and r['role']==role for r in rr))
  check('input_role120_and_category10_'+role,len(rr)==120 and len(Counter(r['category'] for r in rr))==12 and set(Counter(r['category'] for r in rr).values())=={10})
  for r in rr:check('role_unique_'+r['source_id'][:16],r['source_id'] not in inp);inp[r['source_id']]=r
 records={r['source_id']:r for r in c['records']}
 check('all_input_identities',set(records)==set(inp) and len(records)==240)
 ref=read(ROOT/'stage08_support/contract.json');trainpath=Path(ref['source_files']['train']['path']);train=rows(trainpath);trainids={r['source_id'] for r in train};oldids={r['source_id'] for r in ref['queries']}
 check('train_manifest_sha',sha(trainpath)==c['source_train_file_sha256']==ref['source_files']['train']['sha256'])
 check('source_byte_isolation',not set(records)&(oldids|trainids) and not {r['original_sha256'] for r in records.values()}&{sid[5:] for sid in oldids|trainids} and len({r['original_sha256'] for r in records.values()})==240)
 shots=s['shots'];check('selected_shots_allowed',shots in [0,1,4])
 check('shot_matches_system',c['shots']==shots)
 if shots:
  check('seed43_fixed',s.get('support_seed')==43)
  for cat,rr in s['supports'].items():check('support_seed43_'+cat,[r['source_id'] for r in rr]==[r['source_id'] for r in ref['supports']['43'][cat][:shots]])
 else:check('zero_has_no_support',not any(s['supports'].values()))
 images={}
 for sid,r in records.items():
  original=Path(r['original_path']);check('input_metadata_'+sid[:16],r['role']==inp[sid]['role'] and r['category']==inp[sid]['category'] and str(original)==inp[sid]['image'] and r['original_sha256']==inp[sid]['original_sha256'] and sid=='visa_'+r['original_sha256'])
  raw=original.read_bytes();check('original_sha_'+sid[:16],len(raw)==r['original_bytes'] and hashlib.sha256(raw).hexdigest()==r['original_sha256'])
  a=r['asset'];p=(stage/a['path']).resolve();png=p.read_bytes()
  check('448asset_'+sid[:16],p.is_relative_to(stage.resolve()) and len(png)==a['png_bytes'] and hashlib.sha256(png).hexdigest()==a['png_sha256'] and png[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',png[16:24])==(448,448))
  check('render_reproduced_'+sid[:16],support.render(raw)[0]==png)
  images[sid]=png
 for rr in s['supports'].values():
  for r in rr:
   raw=(ROOT/'stage08_support'/r['asset']['path']).read_bytes();check('support_asset_'+r['source_id'][:16],hashlib.sha256(raw).hexdigest()==r['asset']['png_sha256']);images[r['source_id']]=raw
 check('schedule_exact_unique',len({(r['source_id'],r['role']) for r in c['schedule']})==240 and {(r['source_id'],r['role']) for r in c['schedule']}=={(r['source_id'],r['role']) for r in c['records']})
 ch=hashlib.sha256(canon(c)).hexdigest()
 for sid,r in records.items():
  b=iso.payload(r,s,images,ch);encoded=canon(b)
  check('payload_'+sid[:16],set(b)=={'model','state','assets','questions'} and set(b['state'])=={'query','category','normal_references'} and len(b['assets'])==shots+1 and len(b['state']['normal_references'])==shots and b['questions']=={'anomaly':dict(type='noul',instructions=support.INSTRUCTIONS,criteria=support.CRITERIA)})
  check('payload_no_labels_roles_identity_'+sid[:16],sid.encode() not in encoded and b'source_id' not in encoded and b'is_anomalous' not in encoded and b'heldout_eval' not in encoded and b'calibration' not in encoded and b'original_path' not in encoded)
 check('no_gt_flag',c['no_gt_or_label_reads'] is True and c['confirmation_opened'] is False)
 result={'schema':'jev-stage12-independent-input-audit-v1','status':'complete','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'stage':str(stage),'passed':sum(v['passed'] for v in checks),'failed':sum(not v['passed'] for v in checks),'checks':checks,'system_sha256':sha(sysfile),'data_contract_sha256':sha(stage/'contract.json'),'runner_sha256':sha(ROOT/'code/isolated_calibration.py'),'auditor_sha256':sha(Path(__file__)),'records':240,'role_counts':{'calibration':120,'heldout_eval':120},'per_role_category_count':10,'shots':shots,'selection_label_stratification_note':'Originalmetadata selection used known label; this runtime audit does not reopen either label file','calibration_or_heldout_labels_opened':False,'predictions_opened':False,'gt_masks_opened':False,'confirmation_opened':False,'new_api_requests':0,'limits':['Nativevalidation remainder, not blind','Does not establish historical allmodel nonexposure','RGB/SHA/render verified only after finalsystem/data contract freeze']}
 out=ROOT/'stages/12_preexecution_audit.json';assert not out.exists()
 with out.open('x') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()

