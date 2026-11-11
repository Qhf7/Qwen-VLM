#!/usr/bin/env python3
"""Completed-only independent audit and prespecified fixed-confirmation supplement."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib,socket,datetime,math,argparse
import numpy as np
ROOT=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111')
sys.path.insert(0,str(ROOT/'code'))
import final_confirmation as final
import audit_followup_metrics as stats
import audit_stage12_results as common
checks=[]
def check(n,v):
 checks.append(dict(name=n,passed=bool(v)))
 if not v:raise AssertionError(n)
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(l) for l in p.open() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def put(p,v):
 with p.open('x') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+chr(10))
def mapped(raw,fit):
 a=float(fit['coefficient'][0][0]);b=float(fit['intercept'][0]);p=np.clip(raw,1e-6,1-1e-6);z=a*np.log(p/(1-p))+b
 out=np.empty_like(z);ix=z>=0;out[ix]=1/(1+np.exp(-z[ix]));ez=np.exp(z[~ix]);out[~ix]=ez/(1+ez);return out

def selftest():
 y=np.array([0,0,1,1]);p=np.array([.1,.7,.4,.8]);v=common.extra(y,p)
 assert v['confusion']==dict(TP=1,FP=1,TN=1,FN=1)
 assert math.isclose(v['binary_logloss'],-np.log([.9,.3,.4,.8]).mean())
 assert np.allclose(mapped(np.array([0,.25,.5,.75,1]),dict(coefficient=[[1]],intercept=[0])),[1e-6,.25,.5,.75,1-1e-6])
 q=[dict(source_id=x,category='fixture') for x in ['b','a','d','c']];cov=common.coverage(q,y,np.array([.5,.5,.5,.5]));assert cov['0.5']['accepted_source_ids']==['a','b']
 w=stats.bootstrap_weights(q,y);assert w.shape==(2000,4) and np.all(w[:,:2].sum(1)==2) and np.all(w[:,2:].sum(1)==2)
 print(json.dumps(dict(status='selftest_passed',synthetic_only=True,api_calls=0,confirmation_inputs_or_labels_read=False)))

def freeze_method():
 check('server',socket.gethostname()=='autodl-container-132444827f-9c594038')
 stage=ROOT/'stage13_confirmation';s=read(stage/'status.json');c=read(stage/'contract.json')
 check('before_complete_and_labels',s['completed']<240 and not any((stage/n).exists() for n in ['prediction_freeze.json','fixed_mapped_predictions.jsonl','metrics.json']))
 method=dict(schema='jev-stage13-independent-methodology-freeze-v1',status='frozen_before_complete_confirmation',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=socket.gethostname(),auditor_sha256=sha(Path(__file__)),dependencies_sha256={n:sha(ROOT/'code'/n) for n in ['audit_followup_metrics.py','audit_stage12_results.py','final_confirmation.py']},contract_sha256=sha(stage/'contract.json'),bundle_sha256=c['bundle_sha256'],snapshot=dict(completed=s['completed'],started=s['cumulative_started'],failed=s['cumulative_failed'],unknown=s['unknown']),primary='primary_calibrated',secondary='secondary_raw',threshold=.5,methods=dict(independent_points=list(stats.METRICS),by_category=True,category_macro=True,intervals='2000 multinomial paired source bootstrap, strata category x boolean label, seed42; percentile linear quantiles .025/.975',macro_bootstrap='same paired weights; average12 category metrics each replicate',calibrator='frozen stage12 coefficient/intercept; independent algebraic mapping; no refit',supplement=['binary logloss clip1e-15','confusion TP FP TN FN','Wilson two-sided95 z1.959963984540054','fixed coverage .5/.75/1 descending max(p,1-p), sourceID ascending ties, ceil(fraction*240), conditional subset denominators','paired pooled and category-macro logloss differences']),label_gate='Complete240 settled durable prediction_freeze and fixed_mapped_predictions; original evaluator complete; independent rechecks before opening protected labels',protected_label_bytes_opened=False,legacy_metadata_opened=False,gt_masks_opened=False,new_api_requests=0,no_result_driven_changes=True)
 put(ROOT/'stages/13_supplement_methodology_freeze.json',method);print(json.dumps(method,ensure_ascii=False,indent=2))

def main():
 check('server',socket.gethostname()=='autodl-container-132444827f-9c594038')
 stage=ROOT/'stage13_confirmation';method=read(ROOT/'stages/13_supplement_methodology_freeze.json')
 check('method_unchanged',method['auditor_sha256']==sha(Path(__file__)) and all(sha(ROOT/'code'/n)==h for n,h in method['dependencies_sha256'].items()))
 required=['status.json','contract.json','final_system_bundle.json','prediction_freeze.json','fixed_mapped_predictions.jsonl','metrics.json','completion_certificate.json']
 check('completed_artifacts_present',all((stage/n).exists() for n in required))
 status=read(stage/'status.json');c,bundle,images=final.verify_inputs(ROOT);freeze=read(stage/'prediction_freeze.json');m=read(stage/'metrics.json');certificate=read(stage/'completion_certificate.json')
 check('240complete',status['status']==m['status']==certificate['status']=='complete' and status['completed']==status['expected']==m['denominator']==240)
 check('frozen_contract_bundle',freeze['contract_sha256']==sha(stage/'contract.json')==method['contract_sha256'] and freeze['bundle_sha256']==sha(stage/'final_system_bundle.json')==method['bundle_sha256'])
 check('full_predictionfreeze',freeze['predictions_sha256']==sha(stage/'predictions.jsonl') and freeze['attempts_sha256']==sha(stage/'attempts.jsonl') and freeze['predictions_before_confirmation_label_evaluation'] and freeze['no_confirmation_fit_or_selection'] and m['prediction_freeze']==freeze)
 check('mapfreeze',m['fixed_mapped_predictions_sha256']==sha(stage/'fixed_mapped_predictions.jsonl'))
 check('fixed_primary_secondary',m['primary']=='primary_calibrated' and m['secondary']=='secondary_raw' and m['threshold']==.5 and m['no_refit'] and m['no_threshold_optimization'] and m['selection_independent_of_confirmation'] and m['primary_all240_without_refusal'])
 check('certificate_binding',certificate['metrics_sha256']==sha(stage/'metrics.json') and certificate['report_sha256']==sha(stage/'REPORT.md') and certificate['prediction_freeze_sha256']==sha(stage/'prediction_freeze.json') and certificate['inference_label_reads']==0 and certificate['masks_opened'] is False)
 complete,starts,attempts,hashes,unknown=final.audit_logs(stage,c,bundle,images);rr=c['records'];records={r['source_id']:r for r in rr};ch=hashlib.sha256(canon(c)).hexdigest()
 check('allsettled240',len(complete)==len(records)==240 and set(complete)==set(records) and not unknown)
 for i,(sid,r) in enumerate(complete.items()):check('response_'+str(i),r['request_sha256']==hashes[sid] and r['contract_sha256']==ch and r['bundle_sha256']==c['bundle_sha256'] and r['response_model']=='jev-1.14.0' and math.isfinite(r['probability']) and 0<=r['probability']<=1 and r['category']==records[sid]['category'])
 raw=np.array([complete[r['source_id']]['probability'] for r in rr]);primary=mapped(raw,bundle['calibrator']);mappedrows=rows(stage/'fixed_mapped_predictions.jsonl')
 for i,(r,z,p,q) in enumerate(zip(rr,mappedrows,raw,primary)):check('fixed_map_'+str(i),set(z)=={'source_id','raw_probability','calibrated_probability','primary_prediction','secondary_prediction'} and z['source_id']==r['source_id'] and stats.close(z['raw_probability'],float(p)) and stats.close(z['calibrated_probability'],float(q)) and z['primary_prediction']==bool(q>=.5) and z['secondary_prediction']==bool(p>=.5))
 check('map240',len(mappedrows)==240)
 check('labels_guard_before_open',len(complete)==240 and not unknown and freeze['predictions_before_confirmation_label_evaluation'] and len(mappedrows)==240 and m['fixed_mapped_predictions_sha256']==sha(stage/'fixed_mapped_predictions.jsonl'))
 # The first protected label byte access in this independent program occurs here.
 labelpath=stage/'protected_labels.jsonl';check('protectedlabel_sha',sha(labelpath)==c['protected_label_sha256']==m['protected_labels_sha256']);labels=rows(labelpath);look={r['source_id']:r for r in labels}
 check('labels240identity',len(look)==len(labels)==240 and set(look)==set(records) and all(set(r)=={'source_id','is_anomalous'} and type(r['is_anomalous']) is bool for r in labels))
 y=np.array([int(look[r['source_id']]['is_anomalous']) for r in rr]);counts=Counter((r['category'],int(v)) for r,v in zip(rr,y));check('12x2x10',len(counts)==24 and set(counts.values())=={10})
 probs=dict(primary_calibrated=primary,secondary_raw=raw);cats=sorted({r['category'] for r in rr});ix={cat:np.array([i for i,r in enumerate(rr) if r['category']==cat]) for cat in cats};point={name:common.point(y,p) for name,p in probs.items()};bycat={cat:{name:common.point(y[idx],p[idx]) for name,p in probs.items()} for cat,idx in ix.items()}
 w=stats.bootstrap_weights(rr,y);boot={name:stats.wm(y,p,w) for name,p in probs.items()};categoryboot={cat:{name:stats.wm(y[idx],p[idx],w[:,idx]) for name,p in probs.items()} for cat,idx in ix.items()}
 for name,p in probs.items():
  check('pooled_'+name,stats.close(point[name],m['results'][name]['metrics']))
  check('coverage_'+name,stats.close(common.coverage(rr,y,p),m['results'][name]['retrospective_coverage']))
  for cat in cats:check('category_'+cat+'_'+name,stats.close(bycat[cat][name],m['by_category'][cat][name]))
  check('macro_'+name,stats.close({k:float(np.mean([bycat[cat][name][k] for cat in cats])) for k in stats.METRICS},m['category_macro'][name]))
  for k in stats.METRICS:check('interval_'+name+'_'+k,stats.close(dict(value=point[name][k],ci95=np.quantile(boot[name][k],[.025,.975],method='linear').tolist()),m['bootstrap']['intervals'][name][k]))
 for k in stats.METRICS:check('paireddelta_'+k,stats.close(dict(value=point['primary_calibrated'][k]-point['secondary_raw'][k],ci95=np.quantile(boot['primary_calibrated'][k]-boot['secondary_raw'][k],[.025,.975],method='linear').tolist()),m['bootstrap']['primary_minus_secondary'][k]))
 check('bootmethod',m['bootstrap']['replicates']==2000 and m['bootstrap']['seed']==42 and m['bootstrap']['strata']=='category x image_label' and m['bootstrap']['fixed_calibrator'])
 usagefields=sorted({k for r in complete.values() for k in r['usage']});usage={k:dict(observed_sum=float(sum(r['usage'][k] for r in complete.values() if k in r['usage'])),observed_calls=sum(k in r['usage'] for r in complete.values()),missing_calls=sum(k not in r['usage'] for r in complete.values())) for k in usagefields};events=Counter(r['event'] for r in attempts)
 check('operations',m['operations']['accepted']==240 and m['operations']['started']==sum(starts.values()) and m['operations']['failed']==events['failed'] and m['operations']['unknown']==0 and stats.close(usage,m['operations']['usage']))
 audit=dict(schema='jev-stage13-independent-postexecution-audit-v1',status='complete',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=socket.gethostname(),passed=sum(v['passed'] for v in checks),failed=sum(not v['passed'] for v in checks),checks=checks,auditor_sha256=sha(Path(__file__)),methodology_sha256=sha(ROOT/'stages/13_supplement_methodology_freeze.json'),contract_sha256=sha(stage/'contract.json'),bundle_sha256=sha(stage/'final_system_bundle.json'),prediction_freeze_sha256=sha(stage/'prediction_freeze.json'),metrics_sha256=sha(stage/'metrics.json'),operations=dict(accepted=240,started=sum(starts.values()),failed=events['failed'],unknown=0),point_metrics=point,legacy_metadata_opened=False,protected_boolean_labels_opened_after_full_freeze=True,gt_masks_opened=False,new_api_requests=0,no_refit_or_selection=True)
 out=ROOT/'stages/13_postexecution_audit.json';put(out,audit)
 delta={scope:{} for scope in ['pooled','category_macro']}
 for k in stats.METRICS:
  for scope in delta:
   if scope=='pooled':d=point['primary_calibrated'][k]-point['secondary_raw'][k];b=boot['primary_calibrated'][k]-boot['secondary_raw'][k]
   else:d=float(np.mean([bycat[cat]['primary_calibrated'][k]-bycat[cat]['secondary_raw'][k] for cat in cats]));b=np.mean([categoryboot[cat]['primary_calibrated'][k]-categoryboot[cat]['secondary_raw'][k] for cat in cats],axis=0)
   delta[scope][k]=dict(delta=float(d),ci95=np.quantile(b,[.025,.975],method='linear').tolist())
 ld=common.losses(y,primary)-common.losses(y,raw);ll=dict(pooled=dict(delta=float(ld.mean()),ci95=np.quantile(w@ld/w.sum(axis=1),[.025,.975]).tolist()),category_macro=dict(delta=float(np.mean([ld[idx].mean() for idx in ix.values()])),ci95=np.quantile(np.mean([w[:,idx]@ld[idx]/w[:,idx].sum(axis=1) for idx in ix.values()],axis=0),[.025,.975]).tolist()))
 supp=dict(schema='jev-stage13-fixed-statistical-supplement-v1',status='complete',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=socket.gethostname(),inputs_sha256=dict(metrics=sha(stage/'metrics.json'),bundle=sha(stage/'final_system_bundle.json'),freeze=sha(stage/'prediction_freeze.json'),audit=sha(out)),methods={name:common.extra(y,p) for name,p in probs.items()},by_category={cat:{name:common.extra(y[idx],p[idx]) for name,p in probs.items()} for cat,idx in ix.items()},coverage={},paired_primary_minus_secondary=delta,paired_logloss_difference=ll,bootstrap=dict(replicates=2000,seed=42,strata='category x label',calibrator_refitted=False,provider_uncertainty=False),definitions=dict(confusion='p>=.5 anomalypositive; primary all240 no abstention',logloss_clip=1e-15,mapping_clip=1e-6,coverage='predeclared .5/.75/1 ceil fraction, descendingmax(p,1-p),sourceIDtie; subset conditional Wilson denominators',macro='Arithmetic average12 categories at point and each paired replicate'),limitations=['Single balanced fixed240, not deployment prevalence or crossdomain','Fixedcalibrator bootstrap excludes fit/provider/support selection uncertainty','Sourcebyte isolation does not establish physicalpart or nearduplicate independence; public pretraining unknown','Historical metadata stratification is distinct from sealed final prediction/label evaluation; teacher exposure separate audit','Supplement does not alter primary or tune thresholds'],gt_masks_opened=False,new_api_requests=0)
 for name,p in probs.items():
  order=sorted(range(240),key=lambda i:(-float(max(p[i],1-p[i])),rr[i]['source_id']));supp['coverage'][name]={}
  for cov in [.5,.75,1.]:
   idx=np.array(order[:math.ceil(cov*240)]);supp['coverage'][name][str(cov)]=dict(accepted=len(idx),abstained=240-len(idx),coverage=len(idx)/240,conditional=common.extra(y[idx],p[idx]),accepted_source_set_sha256=hashlib.sha256(canon([rr[i]['source_id'] for i in idx])).hexdigest())
 put(stage/'STATISTICAL_SUPPLEMENT.json',supp)
 lines=['# Stage13独立审计与固定补充统计','','全部240预测和映射冻结后读取布尔标签，独立复算点指标、逐类、类宏平均与2000分层配对区间；不重拟合、不选新阈值。',f"独立审计{audit['passed']}通过，0失败；主系统固定calibrated，raw只作次要诊断。",'','| 方法 | TP | FP | TN | FN | Recall及Wilson95% | FPR及Wilson95% | Logloss |','|---|---:|---:|---:|---:|---|---|---:|']
 for name,v in supp['methods'].items():
  cm=v['confusion'];fmt=lambda z:f"{z['estimate']:.4f} [{z['ci95'][0]:.4f},{z['ci95'][1]:.4f}]";lines.append('| '+name+' | '+' | '.join(str(cm[k]) for k in ['TP','FP','TN','FN'])+' | '+fmt(v['recall'])+' | '+fmt(v['fpr'])+f" | {v['binary_logloss']:.6f} |")
 lines+=['','| 指标 | 主系统−raw | 95%配对区间 |','|---|---:|---|']
 for k,v in delta['pooled'].items():lines.append(f"| {k} | {v['delta']:+.6f} | [{v['ci95'][0]:+.6f},{v['ci95'][1]:+.6f}] |")
 v=ll['pooled'];lines.append(f"| logloss | {v['delta']:+.6f} | [{v['ci95'][0]:+.6f},{v['ci95'][1]:+.6f}] |")
 lines+=['','| 方法 | 固定覆盖率 | 保留 | 拒答 | 条件错误率及Wilson95% |','|---|---:|---:|---:|---|']
 for name,values in supp['coverage'].items():
  for cov,v in values.items():
   z=v['conditional']['error'];lines.append(f"| {name} | {cov} | {v['accepted']} | {v['abstained']} | {z['estimate']:.4f} [{z['ci95'][0]:.4f},{z['ci95'][1]:.4f}] |")
 lines+=['','拒答不是正常；固定coverage回顾分析不代表部署保证，subset条件区间不能用全240分母。正斜率单调校准保留排序，但可改分类阈值及可靠性；完整宏平均差异、逐类混淆与区间见JSON。','','均衡固定240不能外推产线先验；bootstrap条件于既有calibrator和响应，不含其不确定性。本轮确认是冻结方案的终点，负结果同样保留。','','- metrics SHA: '+sha(stage/'metrics.json'),'- supplement SHA: '+sha(stage/'STATISTICAL_SUPPLEMENT.json'),'']
 with (stage/'STATISTICAL_SUPPLEMENT.md').open('x') as f:f.write(chr(10).join(lines))
 print(json.dumps({k:v for k,v in audit.items() if k!='checks'},ensure_ascii=False,indent=2));print(json.dumps(dict(supplement=str(stage/'STATISTICAL_SUPPLEMENT.json'),methods=supp['methods'],logloss_difference=ll),ensure_ascii=False,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');ap.add_argument('--freeze-method',action='store_true');args=ap.parse_args()
 if args.self_test:selftest()
 elif args.freeze_method:freeze_method()
 else:main()
