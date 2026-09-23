"""Recompute comparable diagnostics from completed per-view histogram audits."""
from pathlib import Path
import json, html
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
R=Path('artifacts/version_audit_20260923')
def table(d):
    cols=list(d.columns)
    def f(x): return f'{x:.5f}' if isinstance(x,(float,np.floating)) else str(x)
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(f(x) for x in row)+' |' for row in d.itertuples(index=False,name=None))
def ratio(a,b): return np.divide(a,b,out=np.zeros_like(np.asarray(a,dtype=float)),where=np.asarray(b)!=0)
rows=[]; slices=[]; provenance=[]; base=None; badframes=[]; clsrows=[]
for v in ['v1','v2','v3','v4','v5']:
    d=R/v; report=json.loads((d/'submission_fusion_metrics.json').read_text()); sel=json.loads((d/'selection.json').read_text())
    f=pd.read_csv(d/'per_image.csv'); cache=np.load(d/'per_image_histograms.npz'); h=cache['hist'][:,list(cache['weights']).index(.5)]
    assert len(f)==report['rows']==3600 and h.shape==(3600,2,256)
    assert f.image_path.is_unique and np.isfinite(f.p_yes).all()
    assert (h.sum(axis=(1,2))==448**2).all() and (h[:,0].sum(1)==f.positive_pixels).all()
    if base is None: base=f
    else: assert f[['image_path','p_yes','view_label']].equals(base[['image_path','p_yes','view_label']])
    provenance.append(dict(version=v,checkpoint=report['checkpoint'],sha256=sel['checkpoint_sha256'],rows=len(f)))
    for gate in [.5,0.]:
        g=h.copy(); blocked=f.p_yes.to_numpy()<gate; g[blocked]=0;g[blocked,:,0]=h[blocked].sum(axis=2)
        tp=g[:,0,128:].sum(1);fp=g[:,1,128:].sum(1);pos=g[:,0].sum(1);fn=pos-tp;neg=g[:,1].sum(1)
        anomaly=f.view_label.to_numpy()==1; normal=~anomaly
        dice=ratio(2*tp,2*tp+fp+fn);iou=ratio(tp,tp+fp+fn);recall=ratio(tp,pos)
        m=report['results'][f'gate{gate}|w0.5']
        rows.append(dict(version=v,gate=gate,**m,anomaly_Dice=dice[anomaly].mean(),anomaly_IoU=iou[anomaly].mean(),anomaly_Recall=recall[anomaly].mean(),pixel_Precision=float(ratio(tp.sum(),(tp+fp).sum())),pixel_Recall=float(ratio(tp.sum(),pos.sum())),normal_FP_views=int(((tp+fp)[normal]>0).sum()),normal_FP_pixel_rate=float(ratio(fp[normal].sum(),neg[normal].sum())),anomaly_empty=int(((tp+fp)[anomaly]==0).sum())))
        detail=f[['image_path','class_name','group','view_label','ratio','p_yes']].copy();detail['version']=v;detail['gate']=gate
        detail['Dice']=dice;detail['IoU']=iou;detail['Recall']=recall;detail['TP']=tp;detail['FP']=fp;detail['FN']=fn
        detail['classification_miss']=anomaly&(f.p_yes.to_numpy()<.5)
        detail['gate_blocks_defect']=anomaly&blocked
        detail['empty_anomaly']=anomaly&((tp+fp)==0);detail['low_recall']=anomaly&(recall<.2)
        detail['normal_false_positive']=normal&(fp>0)
        rawtp=h[:,0,128:].sum(1);detail['lost_TP_due_gate']=rawtp-tp
        detail['bad']=detail[['classification_miss','empty_anomaly','low_recall','normal_false_positive']].any(axis=1)
        detail.sort_values(['bad','Dice','FP'],ascending=[False,True,False]).to_csv(d/f'diagnostics_gate{gate}.csv',index=False)
        badframes.append(detail)
        selections={f'class/{c}':f.class_name.eq(c).to_numpy() for c in sorted(f.class_name.unique())}
        for name,lo,hi in [('tiny',0,.005),('small',.005,.02),('large',.02,1.01)]: selections['area/'+name]=anomaly&(f.ratio.to_numpy()>lo if lo==0 else f.ratio.to_numpy()>=lo)&(f.ratio.to_numpy()<hi)
        for key,mask in selections.items():
            a=mask&anomaly;n=mask&normal
            slices.append(dict(version=v,gate=gate,slice=key,views=int(mask.sum()),anomalies=int(a.sum()),Dice=float(dice[a].mean()) if a.any() else np.nan,Recall=float(recall[a].mean()) if a.any() else np.nan,normal_FP_views=int((fp[n]>0).sum()),empty_anomalies=int(((tp+fp)[a]==0).sum())))
summary=pd.DataFrame(rows); summary.to_csv(R/'summary_detailed.csv',index=False)
sliced=pd.DataFrame(slices);sliced.to_csv(R/'slices_detailed.csv',index=False)
allbad=pd.concat(badframes);allbad.to_csv(R/'diagnostics_all_versions.csv',index=False)
y=base.view_label.to_numpy();scores=base.p_yes.to_numpy()
for t in [.1,.3,.5,.7,.9]:
    pred=scores>=t;tp=int(((y==1)&pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum());tn=int(((y==0)&~pred).sum())
    clsrows.append(dict(threshold=t,TP=tp,FP=fp,FN=fn,TN=tn,Precision=tp/(tp+fp),Recall=tp/(tp+fn),F1=2*tp/(2*tp+fp+fn)))
cls=pd.DataFrame(clsrows);cls.to_csv(R/'classification_checked.csv',index=False)
gate=summary[summary.gate==0].set_index('version').select_dtypes('number')-summary[summary.gate==.5].set_index('version').select_dtypes('number');gate=gate.drop(columns='gate').reset_index();gate.to_csv(R/'gate_deltas.csv',index=False)
lines=['# v1–v5 全量 dev 评测对比','', '五个版本均已完成；每版 3600 张视角，有门控和无门控共 10 组结果。数据已同步到本地。', '', '## 口径与限制','',f'18 类，{base.group.nunique()} 个记录中的样本组，1775 异常视角、1825 正常视角。所有版本图像顺序、标签、分类概率一致，逐图直方图像素数已校验。', '', '所有版本使用 mean_w0.5 融合，输出 448×448 uint8 概率图；gate=0.5 拦截 p_yes<0.5 的图，gate=0 保留所有真实预测。固定阈值指标使用量化值≥128；AUROC/AP/F1max 使用全体像素直方图。异常 Dice/IoU/Recall 是异常视角宏平均，pixel Precision/Recall 是全体像素微平均，二者不可混用。', '', '**v1 在这里也是统一 mean_w0.5 融合，不能直接当作历史 max 融合提交的复现。v4 使用 24000 step。** 此 dev 曾用于模型选择，属于已见类别的开发集；并非独立测试集，不能用绝对分数推断比赛榜单。历史 tune/holdout 也不能当作全新未见测试集。', '', '## 分类模型（所有版本共用）','',f'重新用 sklearn 按并列分数处理计算 AUROC={roc_auc_score(y,scores):.6f}，AP={average_precision_score(y,scores):.6f}。分类器冻结，去门控只影响分割输出，不改变分类预测。', '',table(cls),'','## 分割总表（越高越好）','',table(summary[['version','gate','pixel_auroc','pixel_ap','pixel_f1max','anomaly_Dice','anomaly_IoU','anomaly_Recall','pixel_Precision','pixel_Recall']]),'','## 错误统计（越低越好）','',table(summary[['version','gate','normal_FP_views','normal_FP_pixel_rate','anomaly_empty']]),'','## 去门控的变化（无门控减有门控）','',table(gate[['version','pixel_auroc','pixel_ap','pixel_f1max','anomaly_Dice','anomaly_Recall','normal_FP_views','anomaly_empty']]),'','## 面积分桶','',table(sliced[sliced['slice'].str.startswith('area/')]),'','## 逐类别结果','',table(sliced[sliced['slice'].str.startswith('class/')]),'','## Bad case 追踪','', 'diagnostics_all_versions.csv 包含每图每配置的 Dice/IoU/Recall、TP/FP/FN、独立布尔问题标签，以及门控损失的正确缺陷像素 lost_TP_due_gate。可用 image_path 对齐版本，追踪改善与退化。各版本 diagnostics_gate*.csv 已按问题优先排序。原 bad_cases 子表采用互斥标签且只截取前100行，本报告新增诊断表保留全部记录和可重叠标签。', '', '直方图能重算阈值指标，但不保留空间位置；此次运行未保存完整概率图，不能据此直接生成预测叠加图、边界指标或区域 AUPRO。这些需要对选定坏例重推理并保存预测。', '', '## 权重与可复现信息','',table(pd.DataFrame(provenance))]
(R/'完整评测对比报告.md').write_text('\n'.join(lines),encoding='utf-8')
# Standalone browser-readable tables, no external dependency.
body='<h1>v1–v5 全量 dev 评测</h1><p>3600 张 / 10 配置。详细口径及局限见同目录 Markdown 报告。固定阈值为 0.5；分类器共用。</p>'
for title,df in [('分割总表',summary),('分类阈值',cls),('去门控变化',gate),('类别与面积分桶',sliced)]:body+='<h2>'+title+'</h2><div>'+df.to_html(index=False,float_format=lambda x:f'{x:.5f}')+'</div>'
(R/'对比表.html').write_text('<!doctype html><meta charset="utf-8"><title>模型评测对比</title><style>body{font:14px system-ui;margin:28px;color:#223}div{overflow:auto}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd;white-space:nowrap}th{background:#eef3fa;position:sticky;top:0}tr:nth-child(even){background:#f7f9fc}</style>'+body,encoding='utf-8')
print(summary.to_string(index=False))
