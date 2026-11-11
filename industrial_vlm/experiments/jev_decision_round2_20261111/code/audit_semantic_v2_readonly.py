#!/usr/bin/env python3
"""Third-party read-only verification of completed8 semantic proxy targets; no mask reads."""
from pathlib import Path
from collections import Counter,defaultdict
import sys,json,hashlib,socket,datetime,math,base64
R=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_decision_round2_20261111');S=R/'semantic_review_pilot_v2';PRIOR=R/'semantic_review_pilot'
sys.path.insert(0,str(S/'code'))
import semantic_review as sem
checks=[]
def check(n,v):
 checks.append(dict(name=n,passed=bool(v)))
 if not v:raise AssertionError(n)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(l) for l in p.open() if l.strip()]
def canon(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
check('server',socket.gethostname()=='autodl-container-132444827f-9c594038');c=read(S/'source_contract.json');t=read(S/'contract.json');f=read(S/'prediction_freeze.json');m=read(S/'metrics.json');p=read(S/'protocol.json');h=read(S/'request_hashes.json');revision=read(S/'revision_manifest.json')
check('runnerfrozen',sha(S/'code/semantic_review.py')==c['runner_sha256']==f['runner_sha256']=='5a5a614fa139fe5ff9a8e098d8e591b2ab8db0e171288d3340062e554872cafe')
check('known_contract_metric_freezes',sha(S/'metrics.json')=='7610a1389f2a333ddaff6fdb668dd4f29c6bebad7d304dbdf80c0a7ba44d60e9' and sha(S/'prediction_freeze.json')=='5eb458ae43131aa386333ba3e9f7888f671ad9af4dc4de43507ee78735003fd0' and sha(S/'contract.json')==f['contract_sha256']=='2f8fec5fc19d2c5e77ef8e17161ca413f5f1aa2b5a78ed32874f7b9975db7d1f')
check('source_protocol_chain',c['protocol_sha256']==t['protocol_sha256']==sha(S/'protocol.json') and t['source_contract_sha256']==sha(S/'source_contract.json') and p==sem.protocol())
check('oldroot_preserved_preGT_API',all(sha(PRIOR/n)==z for n,z in c['prior_sha256'].items()) and not any((PRIOR/n).exists() for n in ['feasibility.json','attempts.jsonl','predictions.jsonl']) and revision['API_started_before_revision']==revision['real_GT_reads_before_revision']==0 and revision['prior_preserved'])
check('onlinefreeze_preGT_chain',read(R/'online_route_pilot/prediction_freeze.json')['predictions_before_pilot_GT'] and t['n_sources']==2 and t['n_targets']==len(t['records'])==8)
source={r['source_id']:r for r in c['records']};online=read(R/'online_route_pilot/contract.json');check('same24nosubstitution',set(source)=={r['source_id'] for r in online['records']} and len(source)==24)
for i,r in enumerate(c['records']):
 w,he=r['resized_size'];x,y=r['padding'];bounds=[x/448,y/448,(x+w)/448,(y+he)/448]
 check('contentbounds_'+str(i),r['content_bounds_in_query448']==bounds and all(0<=z<=1 for z in bounds) and w<=448 and he<=448 and r['exif_orientation']==1)
ch=hashlib.sha256(canon(t)).hexdigest()
for i,r in enumerate(t['records']):
 body=sem.payload(r,ch);enc=canon(body);check('requesthash_'+str(i),hashlib.sha256(enc).hexdigest()==h[r['source_id']+'|'+r['arm']])
 image=r['arm'].startswith('image');check('state_allowlist_'+str(i),set(body)=={'model','state','assets','questions'} and set(body['state'])==({'claim','content_bounds_in_query448','query'} if image else {'claim','content_bounds_in_query448'}) and len(body['assets'])==int(image) and body['state']['content_bounds_in_query448']==source[r['source_id']]['content_bounds_in_query448'])
 check('no_privileged_state_'+str(i),all(z not in enc for z in [r['source_id'].encode(),b'source_id',b'proxy_label',b'candidate_box',b'old_probability',b'ground_truth',b'mask_path',b'correct_answer']))
 check('question_fixed_'+str(i),body['questions']=={'supported':dict(type='noul',instructions=sem.INSTRUCTIONS,criteria=sem.CRITERIA)})
 if image:
  asset=body['assets'][body['state']['query']];check('querysamebytes_'+str(i),hashlib.sha256(base64.b64decode(asset['data_base64'])).hexdigest()==r['query_asset']['png_sha256'])
pred=rows(S/'predictions.jsonl');attempts=rows(S/'attempts.jsonl');invs=rows(S/'invocations.jsonl');check('predictionfreeze_hash',f['predictions_sha256']==sha(S/'predictions.jsonl') and f['attempts_sha256']==sha(S/'attempts.jsonl'))
starts={};ends={};prev=Counter();active=0;failed=0
for i,r in enumerate(attempts):
 k=(r['source_id'],r['arm']);identity=k+(r['attempt'],);check('attempt_identity_'+str(i),r['contract_sha256']==ch and r['request_sha256']==h['|'.join(k)] and 1<=r['attempt']<=3)
 if r['event']=='started':
  check('startseq_'+str(i),identity not in starts and r['attempt']==prev[k]+1 and active<3 and len(starts)<144 and failed<10);starts[identity]=r;prev[k]+=1;active+=1
 else:
  check('terminal_'+str(i),r['event']=='succeeded' and identity in starts and identity not in ends and r['http_status']==200 and r['invocation_id']==starts[identity]['invocation_id']);ends[identity]=r;active-=1
check('no_unknown8success',len(starts)==len(ends)==len(pred)==8 and set(starts)==set(ends) and active==0 and set(prev.values())=={1} and len(invs)==1 and invs[0]['explicit_invocation'] and invs[0]['prior_started']==invs[0]['prior_failed']==0)
lookup={}
for i,r in enumerate(pred):
 k=(r['source_id'],r['arm']);identity=k+(r['attempt'],);check('accepted_'+str(i),k not in lookup and identity in ends and r['request_sha256']==h['|'.join(k)] and r['contract_sha256']==ch and r['question']=='supported' and r['response_model']=='jev-1.14.0' and 0<=r['probability']<=1 and math.isfinite(r['probability']) and r['seconds']>=0);lookup[k]=r
# Only after all predictionfreeze/ledger guards are checked, read the completed mask-derived proxy labels; never open masks.
labels=read(S/'proxy_labels.json');check('proxylabelsha',sha(S/'proxy_labels.json')==t['proxy_labels_sha256'])
by=defaultdict(list)
for r in labels:by[r['source_id']].append(r)
check('2sources4arms',len(by)==2 and len(labels)==8 and set(by)=={r['source_id'] for r in t['records']} and all(Counter((r['arm'].split('_')[0],r['proxy_label']) for r in rr)==Counter({('image',0):1,('image',1):1,('text',0):1,('text',1):1}) for rr in by.values()))
computed=[]
for sid,rr in by.items():
 probs={mode:{r['proxy_label']:lookup[sid,r['arm']]['probability'] for r in rr if r['arm'].startswith(mode)} for mode in ['image','text']};im=probs['image'][1]-probs['image'][0];tx=probs['text'][1]-probs['text'][0];computed.append(dict(source_id=sid,image_true=probs['image'][1],image_false=probs['image'][0],text_true=probs['text'][1],text_false=probs['text'][0],image_separation=im,text_separation=tx,image_minus_text_separation=im-tx,image_proxy_correct=int(probs['image'][1]>=.5)+int(probs['image'][0]<.5)))
check('persource_metrics',computed==m['records']);expected=dict(image_proxy_accuracy=sum(v['image_proxy_correct'] for v in computed)/4,image_separation_mean=sum(v['image_separation'] for v in computed)/2,text_prior_separation_mean=sum(v['text_separation'] for v in computed)/2,paired_separation_gain_mean=sum(v['image_minus_text_separation'] for v in computed)/2)
for k,v in expected.items():check('aggregate_'+k,math.isclose(v,m[k],rel_tol=0,abs_tol=1e-14))
check('full24andsemanticlimits',m['denominator']==24 and m['eligible']==m['selected']==2 and m['targets']==8 and m['operations']==dict(accepted=8,started=8,failed=0,unknown=0) and m['labels_are_mask_derived_proxy'] and m['human_visual_fact_gold'] is False and m['defect_type_or_cause_validated'] is False)
usage={k:dict(observed_sum=float(sum(r['usage'][k] for r in pred if k in r['usage'])),observed_calls=sum(k in r['usage'] for r in pred),missing_calls=sum(k not in r['usage'] for r in pred)) for k in sorted({k for r in pred for k in r['usage']})}
result=dict(schema='jev-semanticv2-thirdparty-readonly-post-audit-v1',status='complete',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=socket.gethostname(),passed=sum(r['passed'] for r in checks),failed=sum(not r['passed'] for r in checks),checks=checks,auditor_sha256=sha(Path(__file__)),protocol_sha256=sha(S/'protocol.json'),source_contract_sha256=sha(S/'source_contract.json'),contract_sha256=sha(S/'contract.json'),prediction_freeze_sha256=sha(S/'prediction_freeze.json'),metrics_sha256=sha(S/'metrics.json'),denominator=24,selected_sources=2,targets=8,operations=dict(accepted=8,started=8,failed=0,unknown=0),aggregate_metrics=expected,per_source_proxy_probability_pairs=[{k:v for k,v in r.items() if k!='source_id'} for r in computed],usage=usage,mask_bytes_read=False,historical_GT_container_read=False,stage13_data_read=False,new_api_calls=0,limitations=['Two sources are not eight independent examples; no generalization confidence claim','Mask-derived spatialproxy is not human-verified complete semanticgold, defecttype, or causaltruth','Text-only prior/missingevidence behavior is not equivalent visualtask accuracy','Sourcecontentbounds verified from frozengeometry, no newmask eligibility reenactment','Previously viewed strict-filtered development subset; no newblind data'])
with (S/'thirdparty_readonly_post_audit.json').open('x') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2)+chr(10))
lines=['# 语义v2第三方只读复核','',f"独立检查{result['passed']}项通过，0失败；只读已冻结源码、协议、24source几何、8载荷/预测/账本及全预测冻结后的proxy标签。不调用API，不开任何mask字节、旧GT容器或stage13数据。",'', '固定源分母24，其中严格mask派生筛选仅2源参与；2源×true/false命题×有图/无图=8请求。8accepted/8started，均attempt1、HTTP200、0failed、unknown0。8请求不是8个独立来源。', '', '| 来源 | 有图true/false | 无图true/false | 有图分离 | 无图分离 | 分离增量 |','|---|---|---|---:|---:|---:|']
for i,r in enumerate(computed):lines.append(f"| {i+1} | {r['image_true']:.2f}/{r['image_false']:.2f} | {r['text_true']:.2f}/{r['text_false']:.2f} | {r['image_separation']:.3f} | {r['text_separation']:.3f} | {r['image_minus_text_separation']:.3f} |")
lines+=['',f"有图proxy accuracy={expected['image_proxy_accuracy']:.3f}（4/4命题）；有图平均分离{expected['image_separation_mean']:.3f}，无图prior分离{expected['text_prior_separation_mean']:.3f}，逐source配对分离增量{expected['paired_separation_gain_mean']:.3f}。仅两个来源，不计算泛化承诺，不把4/4作为完整semantic正确率。",'','v2网格按原图contentbounds分3×3，显式排除letterboxpadding，冻结坐标与queryPNG一致。旧v1根保留preGT/API0；复核其SHA不变。proxy真/假由mask空间派生，未做人类完整可见事实标注，不能验证缺陷类型或类别因果解释。','','无图条件同时移除了视觉证据，其低支持概率与prior分离不等于同一视觉任务准确率；这里只描述输入条件差异，不把二来源配对差当因果泛化。','','复核指标与原metrics逐source及聚合完全一致；完整哈希、计数、usage与检查项见thirdparty_readonly_post_audit.json。','']
with (S/'THIRDPARTY_READONLY_AUDIT.md').open('x') as f:f.write(chr(10).join(lines))
print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False,indent=2))
