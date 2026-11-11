#!/usr/bin/env python3
"""Server-only predeclared three-way decision analysis of frozen stage12 outputs."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,socket,datetime,math,argparse
import numpy as np
OLD=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111')
ROOT=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_decision_round2_20261111')
HOST='autodl-container-132444827f-9c594038'
READS=[]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def read(p):
 p=Path(p);assert 'stage13' not in str(p) and 'confirmation' not in p.name;READS.append(str(p));return json.loads(p.read_text())
def rows(p):
 p=Path(p);assert 'stage13' not in str(p) and 'confirmation' not in p.name;READS.append(str(p));return [json.loads(l) for l in p.open() if l.strip()]
def put(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+chr(10))
def txt(p,v):
 with p.open('x') as f:f.write(v)
def wilson(k,n):
 if n==0:return dict(count=int(k),denominator=0,estimate=None,ci95=None,method='Wilson two-sided95; undefined denominator0')
 z=1.959963984540054;v=k/n;den=1+z*z/n;mid=(v+z*z/(2*n))/den;half=z*math.sqrt(v*(1-v)/n+z*z/(4*n*n))/den
 return dict(count=int(k),denominator=int(n),estimate=v,ci95=[max(0.,mid-half),min(1.,mid+half)],method='Wilson two-sided95')
def mapping(p,fit):
 p=np.clip(np.asarray(p,float),1e-6,1-1e-6);z=fit['coefficient'][0][0]*np.log(p/(1-p))+fit['intercept'][0]
 out=np.empty_like(z);ix=z>=0;out[ix]=1/(1+np.exp(-z[ix]));ez=np.exp(z[~ix]);out[~ix]=ez/(1+ez);return out

def freeze():
 assert socket.gethostname()==HOST
 ROOT.mkdir(parents=True,exist_ok=True)
 files=['stage12/run/contract.json','stage12/run/predictions.jsonl','stage12/run/attempts.jsonl','stage12/run/prediction_freeze.json','stage12/run/calibrator.json','stage12/run/metrics.json','stage12_calibration_plan/calibration_labels.jsonl','stage12_calibration_plan/heldout_eval_labels.jsonl','stages/12_postexecution_audit.json']
 # Hashing binds existing files without selecting thresholds or reading outcome records.
 p=dict(schema='jev-decision-round2-frozen-protocol-v1',status='frozen_before_round2_threshold_selection',timestamp_utc=now(),hostname=socket.gethostname(),analysis_code_sha256=sha(Path(__file__)),source_root=str(OLD),input_sha256={n:sha(OLD/n) for n in files},inference='No new API, model forward, image opening, training or refit',data_roles=dict(calibration='stage12 calibration120, same fixed logistic fit data; reuse for thresholds is in-sample development',heldout='stage12 heldout120 has already been evaluated in prior work; once-only fixed round2 policy evaluation, not new blind holdout',forbidden='No stage13 inputs, probabilities, labels, GT masks or legacy confirmation reads; no new routing evaluation on confirmation'),probability='Only fixed stage12 logit-logistic mapping, clip1e-6; no refit',actions=dict(auto_normal='p_cal < lower',auto_anomaly='p_cal >= upper',human_review='lower <= p_cal < upper',tie='At equal lower and upper there is no human band, p>=threshold anomalous'),threshold_grid=dict(values='np.linspace(0,1,101)',pairs='all lower<=upper,5151 pairs; includes0/1',selection='Calibration120 only'),scenarios=dict(fn_to_fp=[1,5,10,20],human_penalty_fp_units=[.1,.25,.5],primary=dict(fn_to_fp=10,human_penalty_fp_units=.25),interpretation='Transparent illustrative assumptions; not measured business costs, not measured human performance'),selection=dict(objective='[r * anomaly_autonormal + normal_autoanomaly + h * human_count] / N',human_model='Deferral incurs penalty h only in this surrogate. Eventual human errors are not measured; objective omits them, so is not real end-to-end loss',tie_break=['objective rounded14decimals','smaller anomaly_autonormal count','larger human_count','smaller normal_autoanomaly count','smaller lower','larger upper'],primary_reason='Illustrative predeclared r10 h.25; no claim this is deployment-optimal'),report_metrics=dict(normal_fpr='normal auto_anomaly / all normal',unsafe_release='anomaly auto_normal / all anomaly',auto_normal_contamination='anomaly auto_normal / all auto_normal; None if no release',automatic_coverage='(auto_normal+auto_anomaly)/allN',human_rate='human/allN',wilson='Two-sided95, z1.959963984540054, every numerator and denominator explicit'),baselines=['raw probability threshold.5','calibrated probability threshold.5','all human review'],uncertainty=dict(bootstrap_replicates=2000,seed=42,strata='category x image_label',paired='fixed policy surrogate cost vs calibrated binary.5',limits='conditions on fitted calibrator/selected thresholds/responses; excludes fitting and threshold-selection uncertainty; Wilson has no deployment guarantee'),exploratory_curve='Save full5151 heldout threshold grid and single-threshold curves clearly exploratory; no heldout optimum, new threshold choice or rule switch',scientific_limits=['Balanced50percent anomaly prevalence differs from production','Each role has5normal+5anomaly per category; small risk denominators','Auto-normal contamination is prevalence dependent','Native validation remainder may have historical exposure; already seen heldout is development analysis','No guarantee from empirical or Wilson risk estimates','Existing final confirmation is not reused to select or evaluate new policies'])
 put(ROOT/'protocol.json',p)
 txt(ROOT/'PROTOCOL.md',chr(10).join(['# 三态决策分析冻结协议','','使用既有stage12 calibration120选阈值；heldout120此前已看，本轮为开发性复盘，不称新盲测。整个分析不读取stage13数据、不再API或模型推理。','','主情景预声明FN:FP=10:1、人工惩罚h=0.25个FP单位；附加1/5/10/20×h0.1/0.25/0.5共12情景。这些是透明假设，不是已测业务成本或人工准确率。','','固定logistic概率p：p<lower自动正常，p>=upper自动异常，中间人工。网格0到1步长0.01、lower<=upper，共5151对。仅cal120最小化[r×异常自动放行+正常自动误报+h×人工数]/120。相同值先更少异常放行、再更多人工、再少误报、再小lower/大upper。','','人工只计惩罚的理想化surrogate不包含人工后续漏检，不能称真实端到端成本。heldout一次性评价冻结的12规则；完整曲线只供探索，不反选最优。','','明确分母：正常FPR=正常自动异常/全部正常；unsafe release=异常自动正常/全部异常；自动正常污染=异常自动正常/自动正常；coverage=自动处置/全部；humanrate=人工/全部。空自动正常时污染率为未定义，不写0。','','全部风险给计数/分母及Wilson95%；2000固定策略配对分层bootstrap只描述既定策略，不包含校准与阈值选择不确定性。','','本协议JSON含代码与输入SHA。下一阶段只据cal冻结policy，再打开本轮heldout评价，确认集不参与。','']))
 print(json.dumps(dict(status='protocol_frozen',protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(Path(__file__)),labels_or_predictions_read=False,new_api_requests=0)))

def route(y,p,l,u):
 y=np.asarray(y,int);p=np.asarray(p,float);normal=p<l;anomaly=p>=u;human=~(normal|anomaly);assert not np.any(normal&anomaly)
 fn=int(sum(normal&(y==1)));fp=int(sum(anomaly&(y==0)));n=len(y);nn=int(sum(y==0));na=int(sum(y==1));hn=int(human.sum());an=int(normal.sum());aa=int(anomaly.sum())
 return dict(n=n,normal=nn,anomalous=na,actions=dict(auto_normal=an,auto_anomaly=aa,human=hn),confusion=dict(anomaly_auto_normal=fn,anomaly_auto_anomaly=int(sum(anomaly&(y==1))),anomaly_human=int(sum(human&(y==1))),normal_auto_normal=int(sum(normal&(y==0))),normal_auto_anomaly=fp,normal_human=int(sum(human&(y==0)))),normal_fpr=wilson(fp,nn),unsafe_release=wilson(fn,na),auto_normal_contamination=wilson(fn,an),automatic_coverage=wilson(an+aa,n),human_rate=wilson(hn,n))
def objective(v,r,h):return (r*v['confusion']['anomaly_auto_normal']+v['confusion']['normal_auto_anomaly']+h*v['actions']['human'])/v['n']
def action_losses(y,p,l,u,r,h):return r*((y==1)&(p<l)).astype(float)+((y==0)&(p>=u)).astype(float)+h*((p>=l)&(p<u)).astype(float)
def grid_values(y,p):
 q=np.linspace(0,1,101);out=[]
 for i,l in enumerate(q):
  for u in q[i:]:
   normal=p<l;anomaly=p>=u;human=~(normal|anomaly)
   out.append(dict(lower=float(l),upper=float(u),fn=int(sum(normal&(y==1))),fp=int(sum(anomaly&(y==0))),human=int(sum(human)),auto_normal=int(sum(normal)),auto_anomaly=int(sum(anomaly))))
 return out

def analyze():
 assert socket.gethostname()==HOST
 proto=read(ROOT/'protocol.json');assert proto['analysis_code_sha256']==sha(Path(__file__))
 assert not (ROOT/'results.json').exists() and not (ROOT/'selected_policies.json').exists()
 for n,h in proto['input_sha256'].items():assert sha(OLD/n)==h,n
 stage=OLD/'stage12/run';c=read(stage/'contract.json');freeze=read(stage/'prediction_freeze.json');fit=read(stage/'calibrator.json');oldm=read(stage/'metrics.json');audit=read(OLD/'stages/12_postexecution_audit.json');predrows=rows(stage/'predictions.jsonl');pred={r['source_id']:r for r in predrows}
 assert audit['failed']==0 and oldm['status']=='complete' and len(pred)==len(predrows)==240 and fit['heldout_labels_used_for_fit'] is False
 assert freeze['predictions_sha256']==sha(stage/'predictions.jsonl') and freeze['contract_sha256']==sha(stage/'contract.json') and freeze['predictions_before_labels']
 role={r:[v for v in c['records'] if v['role']==r] for r in ['calibration','heldout_eval']};assert all(len(v)==120 for v in role.values())
 # Calibration labels only are opened for selection. Heldout label bytes are opened after policies are durable.
 labcal=rows(OLD/'stage12_calibration_plan/calibration_labels.jsonl');lc={r['source_id']:r['is_anomalous'] for r in labcal};cal=role['calibration'];ycal=np.array([int(lc[r['source_id']]) for r in cal]);rawcal=np.array([pred[r['source_id']]['probability'] for r in cal]);pcal=mapping(rawcal,fit)
 assert len(lc)==120 and ycal.sum()==60 and Counter((r['category'],int(y)) for r,y in zip(cal,ycal))==Counter({(cat,label):5 for cat in {r['category'] for r in cal} for label in [0,1]})
 grid=grid_values(ycal,pcal);assert len(grid)==5151
 policies=[]
 for r in [1,5,10,20]:
  for h in [.1,.25,.5]:
   def key(v):return (round((r*v['fn']+v['fp']+h*v['human'])/120,14),v['fn'],-v['human'],v['fp'],v['lower'],-v['upper'])
   v=min(grid,key=key);policies.append(dict(name='r'+str(r)+'_h'+str(h),fn_to_fp=r,human_penalty=h,lower=v['lower'],upper=v['upper'],primary=r==10 and h==.25,calibration_selection=dict(objective=key(v)[0],counts=v,tie_tuple=list(key(v))),selected_only_on_calibration=True))
 selected=dict(schema='jev-three-way-calibration-selected-policies-v1',status='frozen_before_round2_heldout_label_read',timestamp_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),calibrator_sha256=sha(stage/'calibrator.json'),calibration_source_count=120,calibration_labels_sha256=sha(OLD/'stage12_calibration_plan/calibration_labels.jsonl'),policies=policies,heldout_labels_read_in_round2_before_policy_freeze=False,stage13_data_read=False)
 put(ROOT/'selected_policies.json',selected)
 held=role['heldout_eval'];labheld=rows(OLD/'stage12_calibration_plan/heldout_eval_labels.jsonl');lh={r['source_id']:r['is_anomalous'] for r in labheld};yh=np.array([int(lh[r['source_id']]) for r in held]);rawh=np.array([pred[r['source_id']]['probability'] for r in held]);ph=mapping(rawh,fit)
 assert len(lh)==120 and yh.sum()==60 and not set(lc)&set(lh)
 strata=defaultdict(list)
 for i,(record,y) in enumerate(zip(held,yh)):strata[record['category'],int(y)].append(i)
 assert len(strata)==24 and set(map(len,strata.values()))=={5}
 rng=np.random.default_rng(42);w=np.zeros((2000,120))
 for _,idx in sorted(strata.items()):w[:,idx]=rng.multinomial(len(idx),np.full(len(idx),1/len(idx)),size=2000)
 out=[];primary=None
 for pol in policies:
  r=pol['fn_to_fp'];h=pol['human_penalty'];l=pol['lower'];u=pol['upper'];cm=route(ycal,pcal,l,u);hm=route(yh,ph,l,u);bs=dict(raw_binary_half=route(yh,rawh,.5,.5),calibrated_binary_half=route(yh,ph,.5,.5),all_human_review=route(yh,ph,0.,1.0000000001))
  loss=action_losses(yh,ph,l,u,r,h);base=action_losses(yh,ph,.5,.5,r,h);delta=loss-base;ci=np.quantile(w@delta/w.sum(axis=1),[.025,.975],method='linear').tolist()
  value=dict(policy=pol,calibration=cm,heldout=hm,calibration_surrogate_cost=objective(cm,r,h),heldout_surrogate_cost=objective(hm,r,h),heldout_baselines={k:dict(metrics=v,surrogate_cost=objective(v,r,h)) for k,v in bs.items()},paired_cost_minus_calibrated_binary=dict(delta=float(delta.mean()),ci95=ci,replicates=2000,seed=42,fixed_thresholds=True,calibrator_refit=False),by_category={cat:route(yh[idx],ph[idx],l,u) for cat,idx in sorted(defaultdict(list,{cat:np.array([i for i,v in enumerate(held) if v['category']==cat]) for cat in {v['category'] for v in held}}).items())})
  out.append(value)
  if pol['primary']:primary=value
 # Full curve is exploratory; no heldout argmin or new selection is computed.
 full=[]
 for v in grid_values(yh,ph):full.append(dict(**v,n=120,normal=60,anomalous=60,normal_fpr=v['fp']/60,unsafe_release=v['fn']/60,auto_normal_contamination=(v['fn']/v['auto_normal'] if v['auto_normal'] else None),automatic_coverage=(v['auto_normal']+v['auto_anomaly'])/120,human_rate=v['human']/120,exploratory_only=True))
 curve=ROOT/'EXPLORATORY_HELDOUT_GRID.jsonl'
 with curve.open('x') as f:
  for v in full:f.write(json.dumps(v,ensure_ascii=False,allow_nan=False)+chr(10))
 points=ROOT/'analysis_source_table.jsonl'
 with points.open('x') as f:
  for name,rr,yy,pp,raw in [('calibration',cal,ycal,pcal,rawcal),('heldout_eval',held,yh,ph,rawh)]:
   for rec,y,p,z in zip(rr,yy,pp,raw):f.write(json.dumps(dict(source_id=rec['source_id'],category=rec['category'],role=name,label=int(y),raw_probability=float(z),calibrated_probability=float(p)),sort_keys=True)+chr(10))
 result=dict(schema='jev-three-way-round2-results-v1',status='complete',timestamp_utc=now(),hostname=socket.gethostname(),protocol_sha256=sha(ROOT/'protocol.json'),selected_policy_sha256=sha(ROOT/'selected_policies.json'),input_sha256=proto['input_sha256'],scenarios=out,primary_name=primary['policy']['name'],data_counts=dict(calibration=120,heldout=120,normal_per_role=60,anomaly_per_role=60),same_source_different_role_intersection=0,source_table_sha256=sha(points),exploratory_grid_sha256=sha(curve),warning='Development analysis on alreadyseen heldout; no new blind or confirmation evaluation; costs/humanaccuracy not business facts',no_new_api_calls=True,no_model_forward=True,no_model_refit=True,stage13_inputs_predictions_labels_read=False,gt_masks_read=False,file_reads=READS,limitations=proto['scientific_limits'])
 put(ROOT/'results.json',result)
 # Compact Chinese report with all denominators; None stays undefined.
 fmt=lambda v:'未定义(分母0)' if v['estimate'] is None else f"{v['count']}/{v['denominator']}={100*v['estimate']:.2f}% [{100*v['ci95'][0]:.2f}%,{100*v['ci95'][1]:.2f}%]"
 lines=['# 三态决策路由：calibration120选阈值、既有heldout120复盘','','## 结果如何解释','','这是新的开发分析。heldout120此前已经看过，不称新盲测；stage13确认数据完全不参与本轮选阈值、统计或画曲线。固定既有Jev4shot43/448与logistic，不新增API/推理/训练。','','阈值只用cal120按冻结规则选择，先保存selected_policies.json再打开本轮heldout标签。cal数据同时曾用于logistic拟合与本轮阈值选择，因此cal结果是重用后的in-sample结果，不能当泛化证据。','','## 主情景与分母','',f"预声明主情景FN:FP=10:1、人工惩罚h=0.25个FP单位；cal-only选择lower={primary['policy']['lower']:.2f}、upper={primary['policy']['upper']:.2f}。p<lower自动正常，p>=upper自动异常，中间人工。成本为透明假设，不是已测业务事实；人工惩罚surrogate没有计入人工后续判断错误。",'', '| heldout指标 | 计数/分母、点估计与Wilson95% |','|---|---|']
 for key,title in [('normal_fpr','正常FPR：正常自动异常/全部正常'),('unsafe_release','unsafe release：异常自动正常/全部异常'),('auto_normal_contamination','自动正常污染：异常自动正常/全部自动正常'),('automatic_coverage','自动覆盖率：自动处置/全部'),('human_rate','人工比例：人工/全部')]:lines.append('| '+title+' | '+fmt(primary['heldout'][key])+' |')
 lines+=['','| FN:FP | h | cal lower/upper | heldout自动正常/异常/人工 | 正常FPR | unsafe release | 自动正常污染 | 假设surrogate成本 |','|---:|---:|---|---|---:|---:|---:|---:|']
 for v in out:
  p=v['policy'];a=v['heldout']['actions'];val=lambda k:'未定义' if v['heldout'][k]['estimate'] is None else f"{100*v['heldout'][k]['estimate']:.2f}%"
  lines.append(f"| {p['fn_to_fp']} | {p['human_penalty']} | {p['lower']:.2f}/{p['upper']:.2f} | {a['auto_normal']}/{a['auto_anomaly']}/{a['human']} | {val('normal_fpr')} | {val('unsafe_release')} | {val('auto_normal_contamination')} | {v['heldout_surrogate_cost']:.4f} |")
 lines+=['','## 对比与成本假设','', '| 主情景方法 | 自动正常/异常/人工 | unsafe release | 正常FPR | 假设成本 |','|---|---|---|---|---:|']
 for name,v in [('三态主策略',dict(metrics=primary['heldout'],surrogate_cost=primary['heldout_surrogate_cost']))]+list(primary['heldout_baselines'].items()):
  a=v['metrics']['actions'];lines.append(f"| {name} | {a['auto_normal']}/{a['auto_anomaly']}/{a['human']} | {fmt(v['metrics']['unsafe_release'])} | {fmt(v['metrics']['normal_fpr'])} | {v['surrogate_cost']:.4f} |")
 d=primary['paired_cost_minus_calibrated_binary'];lines+=['',f"主三态−calibrated二分类的假设成本差{d['delta']:+.6f}，固定策略2000配对分层bootstrap95%[{d['ci95'][0]:+.6f},{d['ci95'][1]:+.6f}]。仅条件于当前calibrator和已选阈值，不含选择不确定性；不能写成节省真实业务费用。",'','## 限制与后续','', '0次观察到unsafe release也不保证未来为0；样本只有60异常，Wilson上界仍可能明显非零。污染率分母是自动正常数，不能与异常60分母混用，也强依赖50%异常先验。人工区只转交，不代表人工一定判断正确；端到端风险还需要人工执行数据。', '', '12种情景全部按cal选好后一次性评价heldout，不根据heldout选“最好”情景。EXPLORATORY_HELDOUT_GRID.jsonl完整5151对曲线仅供探索，未计算heldout最优点、更未回写策略。', '', '后续先取得真实业务FN/FP、人工时延与准确率，重新预注册成本/风险政策，并用新的来源互斥留出集验证。既有stage13确认保持终点地位，不能拿来调这轮新路由。','']
 txt(ROOT/'REPORT.md',chr(10).join(lines))
 print(json.dumps(dict(status='complete',primary=primary['policy'],heldout=primary['heldout'],surrogate_cost=primary['heldout_surrogate_cost'],cost_delta=primary['paired_cost_minus_calibrated_binary'],results_sha256=sha(ROOT/'results.json'),stage13_data_read=False),ensure_ascii=False,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');a=ap.parse_args()
 if a.freeze:freeze()
 else:analyze()
