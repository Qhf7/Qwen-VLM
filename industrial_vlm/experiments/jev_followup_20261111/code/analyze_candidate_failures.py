#!/usr/bin/env python3
"""Post-hoc diagnostics of fixed ROI candidates; no threshold tuning, maps or masks."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,socket,datetime,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111')
assert socket.gethostname()=='autodl-container-132444827f-9c594038'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads((ROOT/'stages/03_candidates.json').read_text())
e=json.loads((ROOT/'stages/05_evaluation.json').read_text())
assert c['status']==e['status']=='complete' and not c['confirmation_opened'] and not e['confirmation_opened']
cs={r['source_id']:r for r in c['records']};es={r['source_id']:r for r in e['records']}
assert len(cs)==len(es)==120 and set(cs)==set(es)
rows=[]
for sid,r in es.items():
 x=cs[sid]
 assert r['candidate_area_fraction']==x['bbox_area_fraction'] and r['teacher_split']==x['teacher_split']
 assert r['candidate_bbox_original_norm']==x['bbox_original_norm']
 rows.append(dict(source_id=sid,category=r['category'],label=r['label'],teacher_split=r['teacher_split'],teacher_training_exposed=r['teacher_training_exposed'],area=r['candidate_area_fraction'],candidate_iou=r['candidate_iou'],fullbox=r['full_image_candidate'],same_pixels=r['query_crop_bytes_identical'],quantile_threshold=x['quantile_threshold'],map_min=x['map_range'][0],map_max=x['map_range'][1],gt_bbox=r['gt_bbox_original_norm'],gates={k:bool(r['gates'][k]) for k in ['roi','duplicate']}))
def desc(v):
 if not v:return None
 a=np.asarray(v,dtype=float)
 return dict(n=len(a),minimum=float(a.min()),q25=float(np.quantile(a,.25)),median=float(np.median(a)),mean=float(a.mean()),q75=float(np.quantile(a,.75)),maximum=float(a.max()))
def summary(rs):
 pos=[r for r in rs if r['label']==1];neg=[r for r in rs if r['label']==0]
 ret={'sources':len(rs),'anomalous':len(pos),'normal':len(neg),'fullboxes':sum(r['fullbox'] for r in rs),'same_pixels':sum(r['same_pixels'] for r in rs),'area_summary':desc([r['area'] for r in rs]),'candidate_mean_iou_all_anomaly_denominator':float(np.mean([r['candidate_iou'] for r in pos])) if pos else None,'normal_unconditional_boxes':len(neg),'arms':{}}
 for arm in ['roi','duplicate']:
  keptp=[r for r in pos if r['gates'][arm]]
  keptn=[r for r in neg if r['gates'][arm]]
  base=sum(r['candidate_iou'] for r in pos);after=sum(r['candidate_iou'] for r in keptp)
  ret['arms'][arm]={'anomaly_kept':len(keptp),'anomaly_dropped':len(pos)-len(keptp),'recall':len(keptp)/len(pos) if pos else None,'normal_boxes_kept':len(keptn),'normal_box_fpr':len(keptn)/len(neg) if neg else None,'mean_iou_all_anomaly_denominator':after/len(pos) if pos else None,'discarded_iou_mass':base-after,'mean_iou_loss_vs_unconditional':(base-after)/len(pos) if pos else None,'retained_iou_mass_fraction':after/base if base>0 else None,'anomaly_dropped_with_positive_iou':sum(not r['gates'][arm] and r['candidate_iou']>0 for r in pos)}
 return ret
bins=[('area_lt_0.01',0,.01),('area_0.01_to_0.1',.01,.1),('area_0.1_to_0.5',.1,.5),('area_0.5_to_1',.5,1),('fullbox',1,1.0000001)]
byarea={name:summary([r for r in rows if lo<=r['area']<hi]) for name,lo,hi in bins}
bycat={cat:summary([r for r in rows if r['category']==cat]) for cat in sorted({r['category'] for r in rows})}
byexposure={name:summary([r for r in rows if r['teacher_training_exposed']==flag]) for name,flag in [('teacher_training_exposed',True),('teacher_unseen',False)]}
full=[r for r in rows if r['fullbox']];nonfull=[r for r in rows if not r['fullbox']]
fg=[r for r in full if r['label']==1]
full_equivalence=[]
for r in fg:
 b=r['gt_bbox'];gtarea=(b[2]-b[0])*(b[3]-b[1]);full_equivalence.append(abs(gtarea-r['candidate_iou']))
assert max(full_equivalence,default=0)<1e-12
pos=[r for r in rows if r['label']==1]
areas=np.asarray([r['area'] for r in pos]);ious=np.asarray([r['candidate_iou'] for r in pos])
result={'schema':'jev-fixed-candidate-failure-analysis-v1','status':'complete','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'inputs_sha256':{'stages/03_candidates.json':sha(ROOT/'stages/03_candidates.json'),'stages/05_evaluation.json':sha(ROOT/'stages/05_evaluation.json')},'auditor_sha256':sha(Path(__file__)),'all120':summary(rows),'by_area':byarea,'by_category':bycat,'by_teacher_exposure':byexposure,'fullbox':summary(full),'nonfullbox':summary(nonfull),'map_scalar_diagnostics':{'fullbox_quantile':desc([r['quantile_threshold'] for r in full]),'nonfullbox_quantile':desc([r['quantile_threshold'] for r in nonfull]),'fullbox_map_min':desc([r['map_min'] for r in full]),'fullbox_map_max':desc([r['map_max'] for r in full]),'fullbox_threshold_equals_min':sum(r['quantile_threshold']==r['map_min'] for r in full),'fullbox_gt_area_equals_candidate_iou_max_abs_error':max(full_equivalence,default=0),'fullbox_same_pixels':sum(r['same_pixels'] for r in full)},'probability_semantics':'ROI gate judges fixed candidate region, not calibrated whole-image probability','no_new_thresholds_or_candidates':True,'original_map_distribution_opened':False,'gt_masks_reopened':False,'rgb_reopened':False,'confirmation_opened':False,'new_api_requests':0,'limits':['Current120 previously viewed development sources','Teacher training exposure96/120; unseen24 has only2normal','Scalar map range andquantile cannot identify spatial disconnected components,tie mass or background score distribution','No alternative quantile/threshold tried; no GT-guided tuning','ROI keep gate can only retain or discard this fixed bbox; it cannot repair bbox geometry']}
out=ROOT/'stages/candidate_failure_analysis.json';assert not out.exists()
with out.open('x') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(18,6.8),gridspec_kw={'width_ratios':[1.1,1.55,1.0]})
for exposed,color,marker,name in [(True,'#2867A4','o','Teacher train exposed (n=38 anomaly)'),(False,'#DC8632','^','Teacher unseen (n=22 anomaly)')]:
 rr=[r for r in pos if r['teacher_training_exposed']==exposed]
 axs[0].scatter([r['area'] for r in rr],[r['candidate_iou'] for r in rr],s=32,color=color,marker=marker,alpha=.72,label=name,edgecolors='white',linewidths=.4)
axs[0].set(xlabel='Fixed candidate area / image area',ylabel='Candidate union-box IoU against GT',xlim=(-.025,1.04),ylim=(-.02,1.02),title='A  Fixed candidate geometry (60 anomalies)')
axs[0].legend(loc='upper left',fontsize=8)
axs[0].text(.04,.54,'Full-image candidate: IoU = GT bbox area\n'+str(len(fg))+' / 60 anomaly queries use a full-image box',transform=axs[0].transAxes,fontsize=8.5,bbox={'facecolor':'#F4F4F4','edgecolor':'none'})
cats=list(bycat);x=np.arange(len(cats));w=.23
vals={a:[] for a in ['unconditional','roi','duplicate']}
for cat in cats:
 s=bycat[cat];vals['unconditional'].append(s['candidate_mean_iou_all_anomaly_denominator']);vals['roi'].append(s['arms']['roi']['mean_iou_all_anomaly_denominator']);vals['duplicate'].append(s['arms']['duplicate']['mean_iou_all_anomaly_denominator'])
for j,(a,color,label) in enumerate([('unconditional','#555C68','Unconditional candidate'),('duplicate','#8DA6C1','Duplicate gate'),('roi','#319D8E','ROI gate')]):axs[1].bar(x+(j-1)*w,vals[a],w,label=label,color=color)
axs[1].set_xticks(x,cats,rotation=60,ha='right',fontsize=8)
axs[1].set(ylabel='Mean IoU over all 5 anomalies / category',ylim=(0,max(vals['unconditional'])*1.22),title='B  Keep/drop cannot repair box geometry')
axs[1].legend(fontsize=8,loc='upper right')
S=result['all120'];labels=['Normal boxes\nretained / 60','Anomaly IoU mass\nretained fraction'];xx=np.arange(2)
for j,(name,color,vals2) in enumerate([('Unconditional','#555C68',[1,1]),('Duplicate gate','#8DA6C1',[S['arms']['duplicate']['normal_box_fpr'],S['arms']['duplicate']['retained_iou_mass_fraction']]),('ROI gate','#319D8E',[S['arms']['roi']['normal_box_fpr'],S['arms']['roi']['retained_iou_mass_fraction']])]):
 bars=axs[2].bar(xx+(j-1)*.23,vals2,.23,color=color,label=name)
 for b,v in zip(bars,vals2):axs[2].text(b.get_x()+b.get_width()/2,b.get_height()+.025,f'{100*v:.1f}%',ha='center',fontsize=8)
axs[2].set_xticks(xx,labels,fontsize=9);axs[2].set(ylim=(0,1.14),ylabel='Fraction (0 to 1)',title='C  Rejection removes false boxes and IoU')
axs[2].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.15))
fig.subplots_adjust(left=.055,right=.985,bottom=.29,top=.88,wspace=.31)
fig.text(.055,.975,'Fixed candidate bottleneck analysis: 95th-percentile union box + frozen Jev keep gate',fontsize=15,weight='bold',va='top')
caption='Approved viewed development120 only; Teacher train exposed96/120. Full-image candidates60/120; normal boxes60/60 before rejection.\nA/B use all60 anomalies (5/category), not only kept detections. No candidate threshold search, new map, GT mask, RGB or API access.\nScalar map min/max/quantile cannot establish disconnected-background mechanism. These post-hoc diagnostics do not establish unseen performance.'
fig.text(.055,.105,caption,va='top',fontsize=9,color='#444444',linespacing=1.55)
png=figdir/'candidate_failure_analysis.png';svg=figdir/'candidate_failure_analysis.svg'
fig.savefig(png,dpi=300,facecolor='white');fig.savefig(svg,facecolor='white');plt.close(fig)
provenance={'schema':'jev-candidate-failure-figure-provenance-v1','analysis_sha256':sha(out),'script_sha256':sha(Path(__file__)),'inputs_sha256':result['inputs_sha256'],'outputs_sha256':{p.name:sha(p) for p in [png,svg]},'rendered_on':socket.gethostname(),'new_api_requests':0,'confirmation_opened':False}
with (figdir/'candidate_failure_analysis.provenance.json').open('x') as f:f.write(json.dumps(provenance,indent=2)+'\n')
S=result['all120'];text=[]
text+=['# 固定候选框瓶颈与复核损失分析','','本分析仅使用已冻结stage03候选与stage05评价records，120个已查看开发来源；不重新打开RGB、Teacher map或GT mask，不尝试新quantile/阈值，不发新API，confirmation未开启。','', '## 主要发现','',f"- 无条件Teacher候选全异常meanIoU={S['candidate_mean_iou_all_anomaly_denominator']:.6f}；duplicate门控={S['arms']['duplicate']['mean_iou_all_anomaly_denominator']:.6f}；ROI门控={S['arms']['roi']['mean_iou_all_anomaly_denominator']:.6f}。分母固定60异常。",f"- 120个来源中{len(full)}个候选是全图，其中异常{len(fg)}、正常{len(full)-len(fg)}；全图candidate与crop PNG相同{sum(r['same_pixels'] for r in full)}个。",f"- 固定候选生成器对全部60正常都出框；duplicate只保留{S['arms']['duplicate']['normal_boxes_kept']}个正常框，ROI保留{S['arms']['roi']['normal_boxes_kept']}个。过滤降低正常出框，但同时丢弃真实异常IoU质量。",'', '| 门控 | 异常保留/60 | 正常框保留/60 | 全异常meanIoU | 相对无条件IoU均值损失 | 保留IoU总质量比例 |','|---|---:|---:|---:|---:|---:|']
for arm in ['duplicate','roi']:
 a=S['arms'][arm];text.append(f"| {arm} | {a['anomaly_kept']} | {a['normal_boxes_kept']} | {a['mean_iou_all_anomaly_denominator']:.6f} | {a['mean_iou_loss_vs_unconditional']:.6f} | {100*a['retained_iou_mass_fraction']:.2f}% |")
text+=['','## 为什么95%分位的union可能退化成整图','','冻结规则是对224×224 map取numpy.quantile(...,0.95,method=linear)，保留>=阈值的像素，然后把所有保留位置包成一个union bbox。这里“95%分位”描述分数排名，不代表候选框面积只有5%，也不表示95%置信度。只要入选位置的最小x/y触达左/上边、最大x/y触达右/下边，包围框就覆盖整图；少量相距很远的高分位置也会让union很大。>=还可能在分数有并列时选中超过5%的像素。','','这些是规则的数学性质。现有记录只有map_range、quantile_threshold和最终bbox，没有完整空间分布或并列计数，因此不能断言本数据的全图框一定由离散背景点、分数并列或某一种具体机制造成。本分析不重开map、不补试阈值寻优，只能确认几何退化已经发生。','',f"在{len(fg)}个全图异常候选中，candidate IoU等于GT union bbox归一化面积，最大绝对误差{max(full_equivalence,default=0):.2g}。例如一个GT框只占全图0.15%，全图候选的IoU约0.0015；即使Jev准确判断有异常，保留原全图框也不会提高几何IoU。",'','## 按面积、训练暴露和类别分解','','| 面积组 | 全部 | 异常 | 正常 | 全异常candidate meanIoU | ROI meanIoU |','|---|---:|---:|---:|---:|---:|']
for name,s in byarea.items():
 f=lambda v:'—' if v is None else f'{v:.6f}'
 text.append(f"| {name} | {s['sources']} | {s['anomalous']} | {s['normal']} | {f(s['candidate_mean_iou_all_anomaly_denominator'])} | {f(s['arms']['roi']['mean_iou_all_anomaly_denominator'])} |")
text+=['','| Teacher暴露 | 来源 | 异常 | 正常 | candidate meanIoU | ROI meanIoU |','|---|---:|---:|---:|---:|---:|']
for name,s in byexposure.items():text.append(f"| {name} | {s['sources']} | {s['anomalous']} | {s['normal']} | {s['candidate_mean_iou_all_anomaly_denominator']:.6f} | {s['arms']['roi']['mean_iou_all_anomaly_denominator']:.6f} |")
text+=['','Teacher训练暴露96/120；unseen24中22异常、2正常。子组分布差异很大，不能把均值差异解释成训练暴露的因果作用，2张unseen正常也无法验证低FPR。逐类全分母值见JSON及图B。','','## 研究含义与下一动作','','当前复核只做keep/drop，不移动或细化框，因此几何上界由固定candidate决定；分类正确也可能定位很差。先前ROI−duplicate meanIoU差值约0.00118且配对95%区间跨0，不能据此宣称稳定定位改进。','','后续可以另立协议研究连通域、top-k局部候选、面积/空间约束或学习式定位，但必须在新的开发来源预先固定规则和预算，不能反复在这60异常上调quantile取最好结果。若Teacher用于新路线，需先审其训练来源暴露，严格未见主张不能依赖当前120。当前继续执行既定stage08–12，不让本归因分析触发额外API或修改冻结候选。','','## 图与溯源','','![candidate diagnostics](figures/candidate_failure_analysis.png)','','- JSON：stages/candidate_failure_analysis.json','- 新脚本：code/analyze_candidate_failures.py','- 图：figures/candidate_failure_analysis.png / .svg / .provenance.json',f"- stage03 SHA256：{result['inputs_sha256']['stages/03_candidates.json']}",f"- stage05 SHA256：{result['inputs_sha256']['stages/05_evaluation.json']}",f"- 分析JSON SHA256：{sha(out)}",'']
with (ROOT/'CANDIDATE_FAILURE_ANALYSIS.md').open('x') as f:f.write('\n'.join(text))
print(json.dumps({'status':'complete','analysis':str(out),'all120':S,'fullbox':{'all':len(full),'anomalous':len(fg),'normal':len(full)-len(fg)},'figure':str(png),'document':str(ROOT/'CANDIDATE_FAILURE_ANALYSIS.md'),'confirmation_opened':False,'new_api_requests':0},ensure_ascii=False,indent=2))

