"""Per-class threshold calibration and error analysis for saved single-view predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, balanced_accuracy_score

ROOT = Path("artifacts/e1_real_classification_qwen3_full_3000")

def best_threshold(frame):
    y = frame.view_label.to_numpy(int); s = frame.p_yes.to_numpy(float)
    candidates = np.unique(np.r_[np.linspace(0.01, .99, 99), s])
    rows=[]
    for t in candidates:
        pred=s>=t
        rows.append((f1_score(y,pred,zero_division=0), float(t), precision_score(y,pred,zero_division=0), recall_score(y,pred,zero_division=0), balanced_accuracy_score(y,pred)))
    return max(rows)

def main():
    steps=[]
    for path in sorted(ROOT.glob("step*_views.csv")):
        tag=path.stem.replace("_views",""); frame=pd.read_csv(path)
        global_best=best_threshold(frame)
        class_rows=[]
        for cls,g in frame.groupby("class_name",sort=True):
            f1,t,p,r,ba=best_threshold(g)
            y=g.view_label.to_numpy(int); s=g.p_yes.to_numpy(float); pred05=s>=.5; predcal=s>=t
            small=(y==1)&g.defect_ratio.le(.01).to_numpy()
            class_rows.append(dict(step=int(tag[4:]),class_name=cls,n=len(g),n_normal=int((y==0).sum()),n_anomaly=int((y==1).sum()),threshold=t,f1max=f1,precision_at_threshold=p,recall_at_threshold=r,balanced_accuracy=ba,recall_05=recall_score(y,pred05,zero_division=0),f1_05=f1_score(y,pred05,zero_division=0),small_recall_05=float(pred05[small].mean()) if small.any() else None,small_recall_calibrated=float(predcal[small].mean()) if small.any() else None))
        steps.append((tag,global_best,class_rows))
    rows=[]
    for tag,g,classes in steps:
        rows.append(dict(step=int(tag[4:]),global_f1max=g[0],global_threshold=g[1],global_precision=g[2],global_recall=g[3],global_balanced_accuracy=g[4],mean_class_threshold=float(np.mean([r['threshold'] for r in classes])),macro_calibrated_f1=float(np.mean([r['f1max'] for r in classes])),macro_recall_at_class_threshold=float(np.mean([r['recall_at_threshold'] for r in classes]))))
    summary=pd.DataFrame(rows).sort_values('step'); summary.to_csv(ROOT/'threshold_summary.csv',index=False)
    per=pd.DataFrame([r for _,_,rs in steps for r in rs]); per.to_csv(ROOT/'per_class_thresholds.csv',index=False)
    final=pd.read_csv(ROOT/'step3000_views.csv')
    audio=final[final.class_name.eq('audio_jack_socket')].copy(); audio['pred05']=audio.p_yes>=.5
    fp=audio[(audio.view_label==0)&audio.pred05].sort_values('p_yes',ascending=False)
    fn=audio[(audio.view_label==1)&~audio.pred05].sort_values('p_yes',ascending=True)
    for x in (fp,fn):
        x['rank_abs_error']=(x.p_yes-x.view_label).abs()
    fp.head(20).to_csv(ROOT/'audio_jack_socket_false_positives_top20.csv',index=False)
    fn.head(20).to_csv(ROOT/'audio_jack_socket_false_negatives_top20.csv',index=False)
    audio_stats=dict(n=len(audio),normal=int((audio.view_label==0).sum()),anomaly=int((audio.view_label==1).sum()),accuracy_05=float((audio.view_label==audio.pred05).mean()),false_positives=int(((audio.view_label==0)&audio.pred05).sum()),false_negatives=int(((audio.view_label==1)&~audio.pred05).sum()),normal_mean=float(audio.loc[audio.view_label==0,'p_yes'].mean()),anomaly_mean=float(audio.loc[audio.view_label==1,'p_yes'].mean()),best_threshold=best_threshold(audio),top_false_positive_paths=fp.head(10).image_path.tolist(),top_false_negative_paths=fn.head(10).image_path.tolist())
    report=dict(task='single-image classification',checkpoint='step3000',threshold_summary=summary.to_dict('records'),audio_jack_socket=audio_stats,artifacts=['threshold_summary.csv','per_class_thresholds.csv','audio_jack_socket_false_positives_top20.csv','audio_jack_socket_false_negatives_top20.csv'])
    (ROOT/'THRESHOLD_ERROR_ANALYSIS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    table='| Step | Global F1max | Global threshold | Mean class threshold | Macro calibrated F1 | Macro recall at class threshold |\n|---:|---:|---:|---:|---:|---:|\n' + '\n'.join(f"| {int(r.step)} | {r.global_f1max:.4f} | {r.global_threshold:.4f} | {r.mean_class_threshold:.4f} | {r.macro_calibrated_f1:.4f} | {r.macro_recall_at_class_threshold:.4f} |" for r in summary.itertuples())
    lines=['# 逐类别阈值校准与错误样本分析','','固定阈值为 0.5；校准阈值在每个类别的 Dev 图像上搜索最大 F1，仅用于诊断。','',table,'','## audio_jack_socket（step3000）','',f"样本 {audio_stats['n']} 张；正常 {audio_stats['normal']} 张；异常 {audio_stats['anomaly']} 张。0.5 阈值误报 {audio_stats['false_positives']} 张、漏报 {audio_stats['false_negatives']} 张；正常平均 p(Yes)={audio_stats['normal_mean']:.4f}，异常平均 p(Yes)={audio_stats['anomaly_mean']:.4f}。",'',f"该类别最优阈值为 {audio_stats['best_threshold'][1]:.6f}，对应 F1={audio_stats['best_threshold'][0]:.4f}；详细误报/漏报路径见 CSV。",'','## 如何判断问题来源','','如果校准阈值后 F1 和召回明显上升，主要是分数标定问题；如果阈值扫描仍无法同时提升正常识别和异常召回，则是该类别视觉特征分离不足。完整阈值曲线数据已保存在 `per_class_thresholds.csv`。','']
    (ROOT/'THRESHOLD_ERROR_ANALYSIS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(summary.to_string(index=False)); print(audio_stats)

if __name__=='__main__': main()
