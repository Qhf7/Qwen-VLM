"""Plot only completed validation results, keeping endpoint types separate."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    args=parser.parse_args()
    root=Path(args.root)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    ax=axes[0,0]
    for backend,color in [('vicuna','#607D8B'),('qwen','#1976D2')]:
        vals=[]
        for seed in (42,43,44):
            d=read(root/('lm_%s_s%d'%(backend,seed))/'val_metrics.json')
            if d:vals.append(d['conditions']['real_map']['overall']['scores']['image_auroc'])
        if len(vals)==3:ax.plot([42,43,44],vals,'o-',label=backend,color=color)
    ax.set(xlabel='Paired seed',ylabel='Image AUROC',title='1. Frozen language replacement · val380',xticks=[42,43,44],ylim=(.5,1))
    ax.legend()
    ax=axes[0,1]
    names=[];vals=[]
    for directory,name in [('native_baseline','Native RGB'),('native_map_s42','Native RGB + map')]:
        d=read(root/directory/'val_metrics.json')
        c=read(root/directory/'complete.json')
        if d and c and c.get('passed'):
            names.append(name);vals.append(d['conditions']['real_map']['overall']['scores']['image_auroc'])
    if vals:
        bars=ax.bar(names,vals,color=['#90A4AE','#1976D2'])
        ax.bar_label(bars,fmt='%.3f',padding=3)
    ax.set(ylabel='Image AUROC',title='2. Native vision + learned map · seed42',ylim=(0,1))
    ax=axes[1,0]
    features=[('imagebind','IB224'),('pre_224','Qpre224'),('post_224','Qpost224'),('pre','Qpre448'),('post','Qpost448')]
    x=np.arange(len(features));gt=[];kd=[]
    for feature,_ in features:
        for arm,values in [('gt',gt),('kd',kd)]:
            d=read(root/('dense_%s_%s_s42'%(feature,arm))/'best_validation_reloaded.json')
            values.append(d['validation_reloaded']['pixel_ap_macro'] if d and d.get('passed') else np.nan)
    bars=ax.bar(x-.18,gt,.36,label='GT',color='#1976D2')
    ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
    if np.isfinite(kd).any():
        bars=ax.bar(x+.18,kd,.36,label='GT + KD(.5)',color='#00897B')
        ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
    ax.set(xticks=x,xticklabels=[label for _,label in features],ylabel='Macro pixel AP',title='3/5. Independent dense head · seed42',ylim=(0,.55))
    ax.legend()
    ax=axes[1,1]
    d=read(root/'roi_full/run.json')
    labels={'original_only':'Original','original_center_crop':'+ Center','original_random_crop':'+ Random','original_teacher_roi':'+ ROI','original_teacher_roi_reference':'+ ROI + Ref'}
    if d and d.get('status')=='complete':
        vals=[d['results'][key]['metrics']['overall']['scores']['image_auroc'] for key in labels]
        bars=ax.bar(list(labels.values()),vals,color=['#90A4AE','#90A4AE','#90A4AE','#1976D2','#00897B'])
        ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
    else:ax.text(.5,.5,'Full ROI validation ongoing',ha='center',va='center',transform=ax.transAxes)
    ax.set(ylabel='Image AUROC',title='4. Native Qwen + crop/reference · val380',ylim=(0,1))
    fig.suptitle('Qwen3-VL → AnomalyGPT · supervised custom split · validation evidence',fontsize=14)
    fig.text(.5,-.02,'Panels use different inputs/protocols; compare within each panel. Single seed except panel 1. No new test confirmation.',ha='center',fontsize=9)
    path=root/'migration_results.png'
    fig.savefig(path,dpi=160,bbox_inches='tight')
    plt.close(fig)
    print(path)


if __name__=='__main__':main()
