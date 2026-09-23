"""Create detailed v5-hard diagnostics from a completed full-audit directory."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from e1.evaluate_leaderboard_metrics import pooled_pixel_metrics

def main(root='artifacts/version_audit_20260923/v5hard'):
    d=Path(root); f=pd.read_csv(d/'per_image.csv'); z=np.load(d/'per_image_histograms.npz'); weights=list(z['weights']); h=z['hist']; j=weights.index(.5)
    rows=[]
    for gate in [.5,0.0]:
        g=h.copy(); blocked=f.p_yes.to_numpy()<gate; g[blocked]=0; g[blocked,:,:,0]=h[blocked].sum(axis=-1)
        pos,neg=g[:,j].sum(0); m=pooled_pixel_metrics(pos,neg)
        tp=g[:,j,0,128:].sum(1);fp=g[:,j,1,128:].sum(1);area=g[:,j,0].sum(1);fn=area-tp; an=f.view_label.to_numpy()==1; no=~an
        dice=2*tp/np.maximum(2*tp+fp+fn,1); rec=tp/np.maximum(area,1)
        rows.append(dict(version='v5-hard',gate=gate,**m,anomaly_Dice=float(dice[an].mean()),anomaly_Recall=float(rec[an].mean()),normal_FP_views=int(((tp+fp)[no]>0).sum()),anomaly_empty=int(((tp+fp)[an]==0).sum()),pixel_Precision=float(tp.sum()/max((tp+fp).sum(),1)),pixel_Recall=float(tp.sum()/max(area.sum(),1))) )
        q=f[['image_path','class_name','group','partition','view_label','ratio','p_yes']].copy();q['gate']=gate;q['Dice']=dice;q['Recall']=rec;q['TP']=tp;q['FP']=fp;q['FN']=fn;q['classification_miss']=(an&(f.p_yes.to_numpy()<.5));q['gate_blocks_defect']=an&blocked;q['empty_anomaly']=an&((tp+fp)==0);q['low_recall']=an&(rec<.2);q['normal_false_positive']=no&(fp>0);q['bad']=q[['classification_miss','empty_anomaly','low_recall','normal_false_positive']].any(axis=1);q.sort_values(['bad','Dice','FP'],ascending=[False,True,False]).to_csv(d/f'diagnostics_gate{gate}.csv',index=False)
    pd.DataFrame(rows).to_csv(d/'v5hard_detailed_summary.csv',index=False)
    rep=json.loads((d/'submission_fusion_metrics.json').read_text())
    text='# v5-hard 完整 dev 诊断\n\n'
    text+=f"测试集：`{rep['test']}`，{rep['rows']} 张视角；融合：mean_w0.5；分类 gate：0.5 与 0。\n\n"
    text+='## 总指标\n\n'+pd.DataFrame(rows).to_markdown(index=False)+'\n\n'
    text+='## 输出文件\n\n- `v5hard_detailed_summary.csv`：两种 gate 的总指标。\n- `diagnostics_gate0.5.csv`、`diagnostics_gate0.0.csv`：每图 Dice、Recall、TP/FP/FN、分类漏检、门控拦截、空预测和正常图误报。\n- `bad_cases.csv` 及分类/分割 bad-case 子表：由完整评测器生成。\n'
    (d/'v5hard完整诊断报告.md').write_text(text,encoding='utf-8')
if __name__=='__main__': main()
