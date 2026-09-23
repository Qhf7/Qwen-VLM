"""Aggregate completed v1--v5 audit directories into one comparison table."""
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--out',type=Path)
    a=p.parse_args(); out=a.out or a.root/'summary.csv'; rows=[]
    for v in ('v1','v2','v3','v4','v5'):
        d=a.root/v; m=d/'submission_fusion_metrics.json'; c=d/'classification_metrics.json'
        if not m.exists(): continue
        report=json.loads(m.read_text()); cls=json.loads(c.read_text()) if c.exists() else {}
        for key,val in report['results'].items():
            gate,weight=key.split('|w'); rows.append({'version':v,'gate':float(gate[4:]),'fusion_weight':float(weight),
                'image_cls_auroc':cls.get('image_auroc'),'image_cls_ap':cls.get('image_ap'),
                'pixel_auroc':val.get('pixel_auroc'),'pixel_ap':val.get('pixel_ap'),'pixel_f1max':val.get('pixel_f1max')})
    t=pd.DataFrame(rows).sort_values(['gate','pixel_f1max'],ascending=[False,False]); out.parent.mkdir(parents=True,exist_ok=True); t.to_csv(out,index=False); t.to_markdown(out.with_suffix('.md'),index=False); print(t.to_string(index=False))
if __name__=='__main__': main()
