"""统一分析两个模型的逐视角 B0f 分数。

输入必须来自同一 manifest；脚本以 group 为统计单位，输出 max/top2_mean/mean、
分层 bootstrap 和一个不参与主结果的 calibration 诊断。
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

AGGS=("max","top2_mean","mean")

def f1max(y,s):
    p,r,_=precision_recall_curve(y,s); return float(np.max(2*p*r/(p+r+1e-12)))
def metric(y,s): return dict(auroc=roc_auc_score(y,s),aupr=average_precision_score(y,s),f1max=f1max(y,s))
def samples(v):
    k=["class_name","group_path"]
    if "width" not in v: v=v.assign(width=0,defect_ratio=0)
    out=v.groupby(k,as_index=False).agg(sample_label=("sample_label","first"),resolution=("width","median"),area=("defect_ratio","max"))
    for a in AGGS:
        out[a]=v.groupby(k).p_yes.apply(lambda x: np.sort(x)[-1] if a=="max" else np.sort(x)[-2:].mean() if a=="top2_mean" else x.mean()).to_numpy()
    return out
def summarize(x):
    rows=[]
    for a in AGGS:
        y=x.sample_label.to_numpy(); s=x[a].to_numpy(); m=metric(y,s)
        rows.append({"aggregation":a,"overall_auroc":m["auroc"],"overall_aupr":m["aupr"],"overall_f1max":m["f1max"],"macro_auroc":np.mean([metric(g.sample_label,g[a])["auroc"] for _,g in x.groupby("class_name")]),"macro_aupr":np.mean([metric(g.sample_label,g[a])["aupr"] for _,g in x.groupby("class_name")]),"macro_f1max":np.mean([metric(g.sample_label,g[a])["f1max"] for _,g in x.groupby("class_name")]),"ok_median":np.median(x.loc[y==0,a]),"ng_median":np.median(x.loc[y==1,a])})
    return pd.DataFrame(rows)
def bootstrap(x,n=2000,seed=20260901):
    rng=np.random.default_rng(seed); strata=[g.index.to_numpy() for _,g in x.groupby(["class_name","sample_label"])]
    rec=[]
    for i in range(n):
        ix=np.concatenate([rng.choice(z,len(z),replace=True) for z in strata]); z=x.loc[ix]
        for a in AGGS:
            m=metric(z.sample_label,z[a]); rec.append({"repeat":i,"aggregation":a,**m})
    raw=pd.DataFrame(rec); rows=[]
    for (a,m),g in raw.melt(id_vars=["repeat","aggregation"],value_vars=["auroc","aupr","f1max"]).groupby(["aggregation","variable"]):
        rows.append({"aggregation":a,"metric":m,"ci_low":g.value.quantile(.025),"median":g.value.quantile(.5),"ci_high":g.value.quantile(.975)})
    return pd.DataFrame(rows)
def calibration(x):
    # 50/50 deterministic group split by sorted group path; temperature is a grid-only diagnostic.
    x=x.sort_values(["class_name","group_path"]).reset_index(drop=True); cal=x.groupby("class_name",group_keys=False).head(max(1, x.groupby("class_name").size().iloc[0]//2)); ev=x.drop(cal.index)
    rows=[]
    for a in AGGS:
        def loss(t,z):
            p=np.clip(z[a].to_numpy(float),1e-6,1-1e-6)
            logit=np.clip(np.log(p/(1-p)),-30,30)
            q=np.clip(1/(1+np.exp(-logit/t)),1e-6,1-1e-6)
            y=z.sample_label.to_numpy(float)
            return -np.mean(y*np.log(q)+(1-y)*np.log(1-q))
        ts=np.linspace(.25,4,151); best=float(ts[np.argmin([loss(t,cal) for t in ts])])
        logits=np.log(np.clip(ev[a],1e-6,1-1e-6)/(1-np.clip(ev[a],1e-6,1-1e-6))); p=1/(1+np.exp(-logits/best))
        rows.append({"aggregation":a,"temperature":best,"cal_n":len(cal),"eval_n":len(ev),"eval_auroc_raw":roc_auc_score(ev.sample_label,ev[a]),"eval_aupr_raw":average_precision_score(ev.sample_label,ev[a]),"eval_accuracy_calibrated":np.mean((p>=.5)==ev.sample_label)})
    return pd.DataFrame(rows)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",nargs=2,required=True); ap.add_argument("--names",nargs=2,required=True); ap.add_argument("--out-dir",type=Path,required=True); args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    report=[]
    for path,name in zip(args.input,args.names):
        v=pd.read_csv(path); x=samples(v); x.to_csv(args.out_dir/f"{name}_sample_scores.csv",index=False); sm=summarize(x); sm.insert(0,"model",name); sm.to_csv(args.out_dir/f"{name}_aggregation_metrics.csv",index=False); ci=bootstrap(x); ci.insert(0,"model",name); ci.to_csv(args.out_dir/f"{name}_bootstrap_ci.csv",index=False); ca=calibration(x); ca.insert(0,"model",name); ca.to_csv(args.out_dir/f"{name}_calibration.csv",index=False); report.append((name,len(v),len(x),sm,ci,ca))
    lines=["# B0f cross-model Dev 720 comparison","","This report is generated only when both inputs use the same frozen manifest. Calibration is diagnostic and never replaces the frozen max result.",""]
    for name,nv,ng,sm,ci,ca in report:
        lines += [f"## {name}",f"{ng} groups / {nv} views","",sm.to_csv(index=False),"Bootstrap 95% intervals (group-stratified):",ci.to_csv(index=False),"Calibration split:",ca.to_csv(index=False),""]
    (args.out_dir/"CROSS_MODEL_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
if __name__=="__main__": main()
