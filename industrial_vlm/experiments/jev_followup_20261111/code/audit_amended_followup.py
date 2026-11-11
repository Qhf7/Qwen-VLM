#!/usr/bin/env python3
"""Independent read-only audit of stage08 rescue and amended stage10 outputs."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import argparse,sys,json,hashlib,socket,math
import numpy as np
ROOT=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111')
sys.path.insert(0,str(ROOT/'code'))
import support_robustness as support
import audit_followup_metrics as stats
checks=[]
def check(name,v):
 checks.append({'name':name,'passed':bool(v)})
 if not v:raise AssertionError(name)
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(l) for l in p.open() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def key(v):return v['source_id'],v['arm']
def close(a,b):return stats.close(a,b)
def timestamp(s):return datetime.fromisoformat(s)
def source_assets(c,stage):
 rs={r['source_id']:r for r in c['queries']}
 for groups in c['supports'].values():
  for group in groups.values():
   for r in group:rs[r['source_id']]=r
 images={}
 for sid,r in rs.items():
  raw=(stage/r['asset']['path']).read_bytes()
  check('asset_sha_'+sid[:16],len(raw)==r['asset']['png_bytes'] and hashlib.sha256(raw).hexdigest()==r['asset']['png_sha256'])
  images[sid]=raw
 return images

def old_ledger(stage,hashes,ch,expected):
 by={}
 for i,r in enumerate(rows(stage/'predictions.jsonl')):
  k=key(r)
  check('old_prediction_'+str(i),k in expected and k not in by and r['contract_sha256']==ch and r['request_sha256']==hashes[k] and r['response_model']=='jev-1.14.0' and 0<=r['probability']<=1)
  by[k]=r
 starts=Counter();intents={};term={};events=Counter()
 for i,r in enumerate(rows(stage/'attempts.jsonl')):
  k=key(r);identity=k+(r['attempt'],);events[r['event']]+=1
  check('old_attempt_identity_'+str(i),k in expected and r['contract_sha256']==ch and r['request_sha256']==hashes[k] and type(r['attempt']) is int and 1<=r['attempt']<=3)
  if r['event']=='started':
   starts[k]+=1;check('old_start_'+str(i),r['attempt']==starts[k] and identity not in intents);intents[identity]=r
  else:
   check('old_terminal_'+str(i),r['event'] in ('failed','succeeded') and identity in intents and identity not in term and r['invocation_id']==intents[identity]['invocation_id']);term[identity]=r
   if r['event']=='succeeded':check('old_success_'+str(i),k in by and by[k]['attempt']==r['attempt'])
 check('old_no_unknown',set(intents)==set(term))
 check('old_accepts_have_intents',all(k+(r['attempt'],) in intents for k,r in by.items()))
 check('old_success_count',events['succeeded']==len(by))
 return by,starts,term,events

def validate_metrics(c,complete,metric):
 qs=c['queries'];sids=[q['source_id'] for q in qs]
 config=read(ROOT/'source_baseline/frozen/config.json');label=read(ROOT/'source_baseline/frozen/labels.json')
 check('approved120_label_sha',hashlib.sha256(canon(label)).hexdigest()==config['labels_sha256'])
 y=np.array([label[sid] for sid in sids]);check('balanced120',len(y)==120 and y.sum()==60)
 ps={arm:np.array([complete[sid,arm]['probability'] for sid in sids]) for arm in support.ARMS}
 points={arm:stats.point(y,p) for arm,p in ps.items()};cats=sorted({q['category'] for q in qs});ix={cat:np.array([i for i,q in enumerate(qs) if q['category']==cat]) for cat in cats}
 catpoints={cat:{arm:stats.point(y[idx],p[idx]) for arm,p in ps.items()} for cat,idx in ix.items()}
 for arm,p in ps.items():
  check('point_'+arm,close(points[arm],metric['arms'][arm]['metrics']))
  for cat in cats:check('category_'+cat+'_'+arm,close(catpoints[cat][arm],metric['by_category'][cat][arm]))
  check('macro_'+arm,close({k:float(np.mean([catpoints[cat][arm][k] for cat in cats])) for k in stats.METRICS},metric['category_macro'][arm]))
  rr=[complete[sid,arm] for sid in sids];fields=sorted({k for r in rr for k in r['usage']})
  usage={k:dict(observed_sum=float(sum(r['usage'][k] for r in rr if k in r['usage'])),observed_calls=sum(k in r['usage'] for r in rr),missing_calls=sum(k not in r['usage'] for r in rr)) for k in fields}
  secs=np.array([r['seconds'] for r in rr]);expected_seconds=dict(mean=float(secs.mean()),median=float(np.median(secs)),p95=float(np.quantile(secs,.95)))
  check('usage_'+arm,close(usage,metric['arms'][arm]['usage']));check('latency_'+arm,close(expected_seconds,metric['arms'][arm]['seconds']))
 w=stats.bootstrap_weights(qs,y);boot={arm:stats.wm(y,p,w) for arm,p in ps.items()};cb={cat:{arm:stats.wm(y[idx],p[idx],w[:,idx]) for arm,p in ps.items()} for cat,idx in ix.items()}
 for arm,ref in support.PAIRS:
  saved=metric['paired_bootstrap'][arm+'-minus-'+ref]
  for k in stats.METRICS:
   for group,delta,values in [('pooled',points[arm][k]-points[ref][k],boot[arm][k]-boot[ref][k]),('category_macro',np.mean([catpoints[cat][arm][k]-catpoints[cat][ref][k] for cat in cats]),np.mean([cb[cat][arm][k]-cb[cat][ref][k] for cat in cats],axis=0))]:
    v=saved[group][k];check('bootstrap_'+arm+'_'+ref+'_'+group+'_'+k,close(float(delta),v['delta']) and close(np.quantile(values,[.025,.975]).tolist(),v['ci95']) and v['replicates']==2000 and v['seed']==42 and v['paired'] is True)
 return points

def audit08():
 stage=ROOT/'stage08_rescue';old=ROOT/'stage08_support'
 required=['completion_certificate.json','operational_amendment.json','prediction_freeze.json','metrics.json','combined_predictions.jsonl','predictions.jsonl','attempts.jsonl','invocations.jsonl']
 if any(not (stage/n).exists() for n in required):return {'status':'pending','stage':8,'reason':'amended completion artifacts not all present'}
 cert=read(stage/'completion_certificate.json');am=read(stage/'operational_amendment.json');freeze=read(stage/'prediction_freeze.json');metric=read(stage/'metrics.json');c=read(old/'contract.json')
 check('certificate_complete',cert['status']=='complete_with_operational_amendment' and cert['expected']==cert['completed']==600 and cert['scientific_change'] is False)
 check('original_stage_honestly_incomplete',read(old/'status.json')['status']==cert['original_stage_status']!='complete')
 for name,v in am['original_files'].items():check('original_exact_file_'+name,(old/name).stat().st_size==v['bytes'] and sha(old/name)==v['sha256'] and cert['original_files'][name]==v)
 check('amended_runner_frozen',sha(ROOT/'code/support_rescue.py')==am['runner_sha256']==cert['runner_sha256'])
 check('support_and_transport_frozen',sha(ROOT/'code/support_robustness.py')==am['support_source_sha256']=='fc6ddc02eb2e2a8c0556f3843a5e73321f9d9bfcf1b5f14314de1aea2077d0bd' and sha(ROOT/'code/roi_api.py')==am['transport_source_sha256']=='4f7b5ed1f703b3751f9fd4bb8c86131d684cec0bed3a9011f5108621e7aa9e01')
 check('operational_not_scientific_change',am['scientific_change'] is False and am['max_additional_starts_per_target']==2 and am['max_total_starts_per_target']==5 and am['workers']==1 and am['one_start_per_target_per_invocation'] is True and am['retry_only_after_http502'] is True and am['unknown_completion_not_retried'] is True)
 ch=hashlib.sha256(canon(c)).hexdigest();ah=sha(stage/'operational_amendment.json')
 check('contract_identity',sha(old/'contract.json')==am['original_contract_sha256']==cert['original_contract_sha256']==freeze['contract_sha256'] and ch==am['original_contract_canonical_sha256'])
 for field,name in [('operational_amendment_sha256','operational_amendment.json'),('combined_predictions_sha256','combined_predictions.jsonl'),('prediction_freeze_sha256','prediction_freeze.json'),('rescue_predictions_sha256','predictions.jsonl'),('rescue_attempts_sha256','attempts.jsonl'),('metrics_sha256','metrics.json'),('report_sha256','REPORT.md')]:check('certificate_hash_'+name,cert[field]==sha(stage/name))
 images=source_assets(c,old);qs={q['source_id']:q for q in c['queries']};expected={(sid,arm) for sid in qs for arm in support.ARMS}
 hashes={(sid,arm):hashlib.sha256(canon(support.payload(qs[sid],arm,c,images,ch))).hexdigest() for sid,arm in expected}
 oldcomplete,oldstarts,oldterm,oldevents=old_ledger(old,hashes,ch,expected)
 eligible={key(r):r for r in am['eligible_targets']}
 check('eligible_exact_missing_targets',set(eligible)==expected-set(oldcomplete) and len(eligible)==len(am['eligible_targets']))
 for k,r in eligible.items():
  terminal=[oldterm[k+(n,)] for n in [1,2,3]]
  check('eligible_pure_three502_'+k[0][:16]+'_'+k[1],oldstarts[k]==3 and all(v['event']=='failed' and v['http_status']==502 and v['error']=='http_502' for v in terminal) and r['request_sha256']==hashes[k] and r['original_started']==r['original_failed']==3 and r['original_terminal_sha256']==hashlib.sha256(canon(terminal)).hexdigest())
 check('original_aggregate_counts',len(oldcomplete)==am['original_accepted'] and sum(oldstarts.values())==am['original_started'] and oldevents['failed']==am['original_failures'])
 ivs=rows(stage/'invocations.jsonl');iv={v['invocation_id']:v for v in ivs};check('unique_invocations',len(iv)==len(ivs))
 for i,v in enumerate(ivs):check('invocation_'+str(i),v['amendment_sha256']==ah and v['contract_sha256']==ch and v['explicit_invocation'] is True and v['automatic_http_retries'] is False)
 starts=Counter();intents={};terms={};seen_iv=set();succeeded=set();events=Counter();previous_terminal=None
 for i,v in enumerate(rows(stage/'attempts.jsonl')):
  k=key(v);identity=k+(v['attempt'],);events[v['event']]+=1
  check('rescue_request_'+str(i),k in eligible and v['request_sha256']==hashes[k] and v['contract_sha256']==ch and v['amendment_sha256']==ah and v['invocation_id'] in iv and 1<=v['attempt']<=2 and v['total_attempt']==3+v['attempt'])
  if v['event']=='started':
   starts[k]+=1;ivk=(v['invocation_id'],)+k
   check('rescue_start_'+str(i),starts[k]==v['attempt'] and identity not in intents and ivk not in seen_iv and k not in succeeded)
   check('serial_start_'+str(i),len(intents)==len(terms) and (previous_terminal is None or timestamp(v['timestamp_utc'])>=timestamp(previous_terminal['timestamp_utc'])))
   if v['attempt']==2:
    prev=terms[k+(1,)]
    check('rescue_retry_pure502_'+str(i),prev['event']=='failed' and prev.get('http_status')==502 and prev.get('error')=='http_502' and prev['invocation_id']!=v['invocation_id'] and iv[v['invocation_id']]['retry_failed'] is True and (timestamp(v['timestamp_utc'])-timestamp(prev['timestamp_utc'])).total_seconds()>=5)
   intents[identity]=v;seen_iv.add(ivk)
  else:
   check('rescue_terminal_'+str(i),v['event'] in ('failed','succeeded') and identity in intents and identity not in terms and v['invocation_id']==intents[identity]['invocation_id']);terms[identity]=v;previous_terminal=v
   if v['event']=='succeeded':succeeded.add(k)
 check('rescue_no_unknown',set(intents)==set(terms))
 check('rescue_bounded_total',max(starts.values(),default=0)<=2 and sum(starts.values())<=2*len(eligible))
 rescued={}
 for i,v in enumerate(rows(stage/'predictions.jsonl')):
  k=key(v);identity=k+(v['attempt'],)
  check('rescued_acceptance_'+str(i),k in eligible and k not in oldcomplete and k not in rescued and identity in intents and terms[identity]['event']=='succeeded' and v['request_sha256']==hashes[k] and v['contract_sha256']==ch and v['amendment_sha256']==ah and v['total_attempt']==3+v['attempt'] and v['response_model']=='jev-1.14.0' and math.isfinite(v['probability']) and 0<=v['probability']<=1)
  rescued[k]=v
 check('rescue_success_link',set(rescued)==succeeded and events['succeeded']==len(rescued))
 complete=dict(oldcomplete);complete.update(rescued);combined=rows(stage/'combined_predictions.jsonl')
 check('combined_unique_complete600',len(complete)==len(combined)==600 and set(complete)==expected and combined==[complete[key(r)] for r in c['schedule']])
 check('freeze_all_sources',freeze['predictions_sha256']==sha(stage/'combined_predictions.jsonl') and freeze['original_predictions_sha256']==sha(old/'predictions.jsonl') and freeze['original_attempts_sha256']==sha(old/'attempts.jsonl') and freeze['rescue_predictions_sha256']==sha(stage/'predictions.jsonl') and freeze['rescue_attempts_sha256']==sha(stage/'attempts.jsonl') and freeze['operational_amendment_sha256']==ah and freeze['predictions_before_metric_labels'] is True and metric['prediction_freeze']==freeze)
 check('no_confirmation',not am['confirmation_opened'] and not cert['confirmation_opened'] and not freeze['confirmation_opened'] and not metric['confirmation_opened'])
 points=validate_metrics(c,complete,metric)
 expectedops=dict(accepted=600,started=sum(oldstarts.values())+sum(starts.values()),failed=oldevents['failed']+events['failed'],unknown=0,original_accepted=len(oldcomplete),rescue_accepted=len(rescued),rescue_started=sum(starts.values()))
 check('combined_operation_counts',metric['operations']==expectedops)
 return {'status':'complete','stage':8,'point_metrics':points,'operations':expectedops,'original_stage_unchanged_status':cert['original_stage_status'],'original_max_started':max(oldstarts.values()),'rescue_extra_max_started':max(starts.values(),default=0),'total_per_target_max_started':max(oldstarts[k]+starts[k] for k in expected),'eligible_targets':len(eligible),'original_contract_sha256':sha(old/'contract.json'),'operational_amendment_sha256':ah,'completion_certificate_sha256':sha(stage/'completion_certificate.json'),'metrics_sha256':sha(stage/'metrics.json'),'predictions_sha256':sha(stage/'combined_predictions.jsonl')}


def audit10():
 stage=ROOT/'stage10_resolution';rescue_stage=ROOT/'stage08_rescue';old=ROOT/'stage08_support'
 needed=['status.json','metrics.json','prediction_freeze.json','evaluation_contract_amended_reference.json']
 if any(not (stage/n).exists() for n in needed):return {'status':'pending','stage':10,'reason':'amended resolution artifacts absent'}
 status=read(stage/'status.json')
 if status['status']!='complete':return {'status':'pending','stage':10,'completed':status.get('completed')}
 # Verify the full600 reference and its metrics independently before reusing shared0.
 reference_audit=audit08()
 if reference_audit['status']!='complete':return {'status':'pending','stage':10,'reason':'reference amended stage08 incomplete'}
 import resolution_ablation as res
 c=read(stage/'contract.json');ec=read(stage/'evaluation_contract_amended_reference.json');freeze=read(stage/'prediction_freeze.json');metric=read(stage/'metrics.json');cert=read(rescue_stage/'completion_certificate.json')
 check('resolution_complete120',status['completed']==status['expected']==120)
 check('resolution_frozen_source',sha(ROOT/'code/resolution_ablation.py')==c['runner_sha256']=='4458848524108d22220d1196cffef65967d06f730425e1cb73277e651b1db9af')
 check('amended_evaluator_source',sha(ROOT/'code/resolution_amended_reference.py')==ec['evaluator_source_sha256']==freeze['evaluator_source_sha256'])
 check('evaluation_contract_binding',sha(stage/'contract.json')==ec['original_resolution_contract_sha256']==freeze['contract_sha256'] and sha(stage/'evaluation_contract_amended_reference.json')==freeze['evaluation_contract_sha256']==metric['evaluation_contract_sha256'])
 check('evaluation_protocol_flags',ec['schema']=='jev-resolution-certified-reference-evaluation-v1' and ec['operational_change_only'] is True and ec['scientific_contract_unchanged'] is True and ec['labels_not_opened_by_wrapper_before_freeze'] is True and ec['previously_viewed_development_not_blind'] is True and ec['confirmation_opened'] is False)
 prov=ec['reference']
 check('reference_provenance_shared',prov==freeze['reference_operational_amendment']==metric['operational_reference_provenance'])
 check('reference_hashes',prov['certificate_sha256']==sha(rescue_stage/'completion_certificate.json')==freeze['reference_completion_certificate_sha256'] and prov['combined_predictions_sha256']==sha(rescue_stage/'combined_predictions.jsonl')==freeze['reference_predictions_sha256'] and prov['operational_amendment_sha256']==sha(rescue_stage/'operational_amendment.json') and prov['reference_source_sha256']==sha(ROOT/'code/support_rescue.py'))
 check('original_reference_status_disclosed',prov['original_stage_status']==read(old/'status.json')['status']==cert['original_stage_status']!='complete' and prov['original_status_unchanged'] is True)
 check('reference_only_shared0_120',prov['reused_arm']=='shared0' and prov['reused_accepted']==metric['reused448_accepted']==120 and prov['excluded_support_arm_observations']==metric['excluded_support_arm_observations']==480)
 check('320_prediction_freeze',freeze['predictions320_sha256']==sha(stage/'predictions.jsonl') and metric['prediction_freeze']==freeze and freeze['labels_not_opened_by_wrapper_before_prediction_freeze'] is True)
 by,starts,events=stats.accepted(stage,c,10)
 qs=c['queries'];sids=[q['source_id'] for q in qs];rc=read(old/'contract.json');combined=rows(rescue_stage/'combined_predictions.jsonl');ref={key(r):r for r in combined}
 check('query_order_aligned',[q['source_id'] for q in rc['queries']]==sids and ec['query_order_sha256']==hashlib.sha256(canon(sids)).hexdigest())
 config=read(ROOT/'source_baseline/frozen/config.json');labels=read(ROOT/'source_baseline/frozen/labels.json');check('labels_identity_after_freeze',hashlib.sha256(canon(labels)).hexdigest()==config['labels_sha256'])
 y=np.array([labels[sid] for sid in sids]);check('resolution_balanced120',len(y)==120 and y.sum()==60)
 ps={res.ARM:np.array([by[sid,res.ARM]['probability'] for sid in sids]),res.REFERENCE:np.array([ref[sid,'shared0']['probability'] for sid in sids])}
 points={a:stats.point(y,p) for a,p in ps.items()};cats=sorted({q['category'] for q in qs});ix={cat:np.array([i for i,q in enumerate(qs) if q['category']==cat]) for cat in cats}
 cp={cat:{a:stats.point(y[idx],p[idx]) for a,p in ps.items()} for cat,idx in ix.items()}
 for a in ps:
  check('resolution_points_'+a,close(points[a],metric['arms'][a]['metrics']))
  for cat in cats:check('resolution_category_'+cat+'_'+a,close(cp[cat][a],metric['by_category'][cat][a]))
  check('resolution_macro_'+a,close({k:float(np.mean([cp[cat][a][k] for cat in cats])) for k in stats.METRICS},metric['category_macro'][a]))
  rr=[by[sid,res.ARM] if a==res.ARM else ref[sid,'shared0'] for sid in sids];size=320 if a==res.ARM else 448
  fields=sorted({k for r in rr for k in r['usage']});secs=np.array([r['seconds'] for r in rr])
  budget=dict(accepted_requests=120,images_per_request=1,canvas_pixels_per_request=size*size,total_canvas_pixels=120*size*size,observed_usage={k:dict(observed_sum=float(sum(r['usage'][k] for r in rr if k in r['usage'])),observed_calls=sum(k in r['usage'] for r in rr),missing_calls=sum(k not in r['usage'] for r in rr)) for k in fields},latency_seconds=dict(mean=float(secs.mean()),median=float(np.median(secs)),p95=float(np.quantile(secs,.95))),limitation='Observed token totals only; canvas pixel count is not an exact model image-token formula; no currency estimate')
  check('resolution_budget_'+a,close(budget,metric['arms'][a]['budget']))
 w=stats.bootstrap_weights(qs,y);boot={a:stats.wm(y,p,w) for a,p in ps.items()};cb={cat:{a:stats.wm(y[idx],p[idx],w[:,idx]) for a,p in ps.items()} for cat,idx in ix.items()}
 a=res.ARM;b=res.REFERENCE;stored=metric['paired_bootstrap'];check('resolution_delta_direction',stored['direction']=='320-minus448')
 for k in stats.METRICS:
  for group,delta,values in [('pooled',points[a][k]-points[b][k],boot[a][k]-boot[b][k]),('category_macro',np.mean([cp[cat][a][k]-cp[cat][b][k] for cat in cats]),np.mean([cb[cat][a][k]-cb[cat][b][k] for cat in cats],axis=0))]:
   v=stored[group][k];check('resolution_bootstrap_'+group+'_'+k,close(float(delta),v['delta']) and close(np.quantile(values,[.025,.975]).tolist(),v['ci95']) and v['replicates']==2000 and v['seed']==42 and v['paired'] is True)
 expectedops=dict(accepted=120,started=sum(starts.values()),failed=events['failed'],unknown=0)
 check('resolution_operation_counts',metric['operations320']==expectedops)
 check('resolution_pixel_ratio',close(metric['canvas_pixel_ratio_320_over448'],320**2/448**2))
 check('resolution_confirmation_closed',metric['confirmation_opened'] is False and freeze['confirmation_opened'] is False)
 return dict(status='complete',stage=10,point_metrics=points,operations=expectedops,reference_amended600_audited=True,reused_shared0_448=120,excluded_reference_support_observations=480,reference_completion_certificate_sha256=sha(rescue_stage/'completion_certificate.json'),evaluation_contract_sha256=sha(stage/'evaluation_contract_amended_reference.json'),contract_sha256=sha(stage/'contract.json'),metrics_sha256=sha(stage/'metrics.json'),prediction_freeze_sha256=sha(stage/'prediction_freeze.json'))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',type=int,choices=[8,10],required=True);args=ap.parse_args()
 check('designated_server',socket.gethostname()=='autodl-container-132444827f-9c594038')
 value=audit08() if args.stage==8 else audit10()
 if value['status']!='complete':print(json.dumps(value));return
 value.update(schema='jev-independent-amended-postexecution-audit-v1',timestamp_utc=datetime.now().astimezone().isoformat(),hostname=socket.gethostname(),passed=sum(v['passed'] for v in checks),failed=sum(not v['passed'] for v in checks),checks=checks,auditor_sha256=sha(Path(__file__)),metrics_implementation='Independent pairwise weighted AUC, grouped threshold AP, confusion/Brier/ECE and paired stratified bootstrap2000; no frozen metric functions called',gt_masks_opened=False,confirmation_opened=False,new_api_requests=0)
 out=ROOT/'stages'/f'{args.stage:02d}_postexecution_audit.json';check('new_audit_path',not out.exists())
 with out.open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in value.items() if k!='checks'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()

