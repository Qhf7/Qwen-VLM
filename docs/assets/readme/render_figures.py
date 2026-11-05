#!/usr/bin/env python3
"""Render README figures from audited, explicitly recorded experiment summaries.

Run from any directory: python docs/assets/readme/render_figures.py
Requires only matplotlib and numpy. No model weights or inference are used.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator
from matplotlib.patches import FancyBboxPatch
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NAVY, BLUE, TEAL, AMBER = '#142840', '#3675DA', '#087F83', '#BA6A16'
GRAY, LIGHT, GRID, MUTED = '#7B8EA3', '#F3F7FC', '#DBE4EF', '#5A6E83'
SOURCE_PATHS = [
    'migration_20261103/PHASE1_RESULTS.md',
    'migration_20261103/lora_20261104/RESULTS.md',
    'migration_20261103/advanced_20261104/FINAL_REPORT.md',
    'industrial_vlm/reports/multitask_v2/scoring_matrix.md',
    'industrial_vlm/reports/multitask_v2/实验报告.md',
]
DATA = {
    'schema_version': 1,
    'created_for': 'README visualization; no new training or inference',
    'source_path_base': 'repository root',
    'sources': [
        {'path': p, 'sha256': hashlib.sha256((ROOT / p).read_bytes()).hexdigest()}
        for p in SOURCE_PATHS
    ],
    'language_transfer': {
        'metric': 'global image AUROC', 'seeds': [42, 43, 44],
        'dataset': 'custom supervised VisA development val380',
        'checkpoint_selection': 'development validation AUROC',
        'models': ['Vicuna + trained bridge', 'Qwen + trained bridge (A)', 'Qwen + language LoRA (C)'],
        'mean': [0.889684, 0.925779, 0.936983],
        'sample_sd_ddof1': [0.013285, 0.032698, 0.030098],
        'sources': SOURCE_PATHS[:2],
        'notes': ['Shared ImageBind/BigDec front end; A trains a new bridge with frozen LLM.',
                  'C continues from the matched-seed A with frozen bridge and q/v LoRA.',
                  'Error bars show training-seed sample SD, not confidence intervals.'],
    },
    'development_vs_external': {
        'metric': 'global image AUROC', 'seeds': [42, 43, 44],
        'arms': ['A', 'C', 'D', 'Cpair', 'R', 'S'],
        'development_val380': [0.925779, 0.936983, 0.940258, 0.935334, 0.947348, 0.939489],
        'external_mvtec1575': [0.708834, 0.711363, 0.714118, 0.709581, 0.703438, 0.717814],
        'external_total_predictions_all_7_arms_3_seeds': 33075,
        'external_generated_fpr_range': [0.612245, 0.715986],
        'sources': [SOURCE_PATHS[2]],
        'notes': ['Same selected checkpoints evaluated on external data; no external tuning.',
                  'R must be compared with Cpair, not historical C.',
                  'S uses a structured prompt schema; cross-schema differences are not isolated LoRA effects.',
                  'A/C/D/Cpair/R are separate branches from matched A; S continues from C.',
                  'The plot shows six arms; the seven-arm external audit additionally includes Cstructured.'],
    },
    'structured_output': {
        'dataset': 'external MVTec1575', 'seeds': [42, 43, 44],
        'arms': ['Cstructured', 'S'],
        'strict_structure_validity': [0.641270, 0.919577],
        'all_anomaly_box_iou': [0.071108, 0.069815],
        'anomaly_denominator': 1183,
        'sources': [SOURCE_PATHS[2]],
        'notes': ['Means across seeds, identical structured schema and 80-token budget.',
                  'No, invalid and missing-box outputs contribute zero IoU in the full anomaly denominator.',
                  'Strict validity includes consistency between the location phrase and its own box center.',
                  'S adds training exposure and targets; this is not a pure mask-supervision causal estimate.'],
    },
    'native_qlora': {
        'dataset': 'independent native Qwen branch, VisA development val120',
        'seed': 42, 'arms': ['NF4 base', 'RR', 'DD', 'RD'],
        'auroc': [0.8239, 0.9149, 0.9456, 0.9201],
        'ap': [0.8558, 0.9301, 0.9526, 0.9354],
        'train_sources': 768, 'image_exposures': 1536, 'optimizer_updates': 192,
        'sources': SOURCE_PATHS[3:],
        'precision_note': 'Values are the published report percentages rounded to 2 decimal places.',
        'notes': ['The NF4 base has no task fine-tuning; each RR/DD/RD arm has the same exposure budget.',
                  'RR = two region tasks; DD = two decision tasks; RD = one of each.',
                  'This protocol is different from the AnomalyGPT migration; no direct between-branch subtraction.'],
    },
}
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'text.color': NAVY, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
    'ytick.color': NAVY, 'axes.edgecolor': GRID, 'axes.titleweight': 'bold',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'svg.fonttype': 'none', 'svg.hashsalt': 'qwen-vlm-readme-v1', 'savefig.facecolor': 'white',
})

def decorate(fig, title, subtitle, note):
    fig.text(.055, .935, title, size=22, weight='bold', color=NAVY)
    fig.text(.055, .885, subtitle, size=10.5, color=MUTED)
    fig.text(.055, .045, note, size=9, color=MUTED)
    fig.add_artist(plt.Line2D([.055, .945], [.845, .845], transform=fig.transFigure, color=GRID, lw=1))

def axes_style(ax, direction='x'):
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='both', length=0, pad=10)
    ax.set_axisbelow(True)
    ax.grid(axis=direction, color=GRID, linewidth=.7)

def save(fig, stem):
    fig.savefig(HERE / f'{stem}.png', dpi=165)
    fig.savefig(HERE / f'{stem}.svg', metadata={'Date': None})
    plt.close(fig)

def language_transfer():
    d=DATA['language_transfer']
    fig=plt.figure(figsize=(12, 5.7))
    decorate(fig, 'A stronger language backbone, then targeted adaptation',
             'Shared ImageBind / BigDec  |  supervised VisA development: 380 images  |  3 training seeds',
             'Dots: seed means. Error bars: sample SD (n=3), not confidence intervals. Validation selects checkpoints.')
    ax=fig.add_axes([.25,.22,.51,.53]);colors=[GRAY, BLUE, TEAL]
    labels=['Vicuna + bridge','Qwen + bridge (A)','Qwen + q/v LoRA (C)']
    for i,(v,sd,c) in enumerate(zip(d['mean'],d['sample_sd_ddof1'],colors)):
        y=2-i
        ax.plot([.75,1],[y,y],color=LIGHT,lw=23,zorder=0,solid_capstyle='round')
        ax.errorbar(v,y,xerr=sd,fmt='o',ms=11,color=c,ecolor=c,lw=2.2,capsize=5)
        ax.text(1.015,y,f'{v:.4f}',va='center',size=15,color=c,weight='bold')
    ax.set_yticks([2,1,0],labels)
    ax.set_xlim(.75,1.0);ax.set_ylim(-.6,2.6)
    ax.xaxis.set_major_locator(MultipleLocator(.05));ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.set_xlabel('Development image AUROC',labelpad=13)
    axes_style(ax)
    fig.text(.055,.755,'MODEL',size=9,color=MUTED,weight='bold')
    fig.text(.795,.755,'MEAN AUROC',size=9,color=MUTED,weight='bold')
    fig.text(.80,.578,'+3.61 pp',size=18,color=BLUE,weight='bold')
    fig.text(.80,.542,'Qwen vs Vicuna',size=10,color=MUTED)
    fig.text(.80,.414,'+1.12 pp',size=18,color=TEAL,weight='bold')
    fig.text(.80,.378,'C vs matched A',size=10,color=MUTED)
    save(fig,'language_transfer')

def development_vs_external():
    d=DATA['development_vs_external']
    fig=plt.figure(figsize=(12,7.0))
    decorate(fig,'Development improvements meet a cross-dataset test',
             'Three-seed means  |  VisA val380 selects checkpoints  |  frozen MVTec AD external: 1,575 images',
             'Global image AUROC, same selected weights. R is paired with Cpair. S uses a structured prompt schema.')
    ax=fig.add_axes([.26,.18,.62,.60])
    names=['A  |  Qwen bridge','C  |  language LoRA','D  |  bridge + LoRA','Cpair  |  matched CE','R  |  CE + ranking','S  |  structured SFT']
    for i,(dev,ext) in enumerate(zip(d['development_val380'],d['external_mvtec1575'])):
        y=5-i
        if i in [3,4]:ax.axhspan(y-.40,y+.40,color='#FFF6E9',zorder=-1)
        ax.plot([ext,dev],[y,y],color=GRID,lw=4,solid_capstyle='round')
        ax.scatter([ext],[y],s=90,color=AMBER,zorder=3)
        ax.scatter([dev],[y],s=90,color=BLUE,zorder=3)
        ax.text(ext-.009,y,f'{ext*100:.2f}',ha='right',va='center',size=10,color=AMBER,weight='bold')
        ax.text(dev+.009,y,f'{dev*100:.2f}',ha='left',va='center',size=10,color=BLUE,weight='bold')
    ax.set_yticks(range(5,-1,-1),names);ax.set_ylim(-.65,5.65)
    ax.set_xlim(.65,1.00);ax.xaxis.set_major_locator(MultipleLocator(.05));ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.set_xlabel('Image AUROC  |  common 65%–100% display range',labelpad=9)
    axes_style(ax)
    fig.text(.29,.805,'●  MVTec external',color=AMBER,size=10,weight='bold')
    fig.text(.59,.805,'●  VisA development',color=BLUE,size=10,weight='bold')
    fig.text(.055,.075,'External audit: 7 arms × 3 seeds = 33,075 predictions; the additional arm is Cstructured.',size=9.5,color=MUTED)
    save(fig,'development_external')

def structured_output():
    d=DATA['structured_output']
    fig=plt.figure(figsize=(12,5.7))
    decorate(fig,'More consistent structure; spatial accuracy remains limited',
             'Cstructured vs S  |  identical structured schema and 80-token budget  |  external MVTec AD, 3 seeds',
             'All 1,183 anomaly images enter the IoU denominator; No, invalid and missing-box outputs score zero.')
    axes=[fig.add_axes([.08,.23,.38,.52]),fig.add_axes([.58,.23,.34,.52])]
    specs=[('Strict structure validity',d['strict_structure_validity'],(0,1),PercentFormatter(1,decimals=0)),
           ('All-anomaly bounding-box IoU',d['all_anomaly_box_iou'],(0,.10),None)]
    for ax,(title,values,lim,fmt) in zip(axes,specs):
        ax.set_title(title,loc='left',size=13,pad=20)
        bars=ax.bar([0,1],values,color=[GRAY,TEAL],width=.50,zorder=3)
        for bar,v in zip(bars,values):
            txt=f'{v*100:.2f}%' if fmt else f'{v:.6f}'
            ax.text(bar.get_x()+bar.get_width()/2,v+(lim[1]*.035),txt,ha='center',size=14,weight='bold')
        ax.set_xticks([0,1],['Cstructured','S']);ax.set_xlim(-.65,1.65);ax.set_ylim(*lim)
        if fmt:ax.yaxis.set_major_formatter(fmt);ax.yaxis.set_major_locator(MultipleLocator(.2))
        else:ax.yaxis.set_major_locator(MultipleLocator(.02));ax.set_ylabel('IoU  (display: 0.00–0.10)',labelpad=8)
        axes_style(ax,'y')
    fig.text(.11,.125,'+27.83 percentage points',color=TEAL,size=13,weight='bold')
    fig.text(.60,.125,'0.071108 → 0.069815',color=AMBER,size=13,weight='bold')
    save(fig,'structured_output')

def native_qlora():
    d=DATA['native_qlora']
    fig=plt.figure(figsize=(12,6.0))
    decorate(fig,'Native Qwen QLoRA: a separate controlled task ablation',
             'Qwen3-VL-8B-Instruct  |  seed42  |  independent VisA development val120  |  native visual pathway',
             'RR: region + region. DD: decision + decision. RD: one of each. Separate protocol from language migration.')
    ax=fig.add_axes([.09,.22,.83,.53]);x=np.arange(4);w=.28
    auroc=ax.bar(x-w/2,d['auroc'],w,color=BLUE,label='AUROC',zorder=3)
    ap=ax.bar(x+w/2,d['ap'],w,color=TEAL,label='AP',zorder=3)
    for bars in [auroc,ap]:
        for b in bars:
            v=b.get_height();ax.text(b.get_x()+b.get_width()/2,v+.018,f'{v*100:.2f}',ha='center',size=11,weight='bold',color=b.get_facecolor())
    ax.set_ylim(0,1.08);ax.set_xlim(-.6,3.6);ax.set_xticks(x,d['arms'])
    ax.set_ylabel('Validation score',labelpad=12);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.yaxis.set_major_locator(MultipleLocator(.2))
    ax.legend(ncol=2,frameon=False,loc='lower left',bbox_to_anchor=(.01,1.03),fontsize=10)
    axes_style(ax,'y')
    fig.text(.10,.125,'Each training arm: 768 source images  ·  1,536 exposures  ·  192 optimizer updates',size=11,color=MUTED)
    save(fig,'native_qlora')

def architecture():
    # Diagram, not a quantitative plot. SVG-native keeps the README asset small and crisp.
    from html import escape
    blocks=[]
    def text(x,y,s,size=15,fill=NAVY,weight='400',anchor='start'):
        blocks.append(f'<text x="{x}" y="{y}" fill="{fill}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{escape(s)}</text>')
    def box(x,y,w,h,title,lines,color=BLUE):
        blocks.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="15" fill="white" stroke="#DDE5EF"/>')
        blocks.append(f'<rect x="{x}" y="{y+18}" width="4" height="{h-36}" rx="2" fill="{color}"/>')
        text(x+19,y+30,title,17,NAVY,'700')
        for i,l in enumerate(lines):text(x+19,y+55+i*19,l,12,MUTED)
    def arrow(x1,y1,x2,y2,c='#8FA4BD'):
        blocks.append(f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{c}" stroke-width="2.2" marker-end="url(#arrow)"/>')
    blocks.append('''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="620" viewBox="0 0 1280 620"><defs><linearGradient id="bg" x1="0" x2="1" y1="0" y2="1"><stop stop-color="#F4F8FE"/><stop offset="1" stop-color="#ECF7F6"/></linearGradient><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10" fill="#8FA4BD"/></marker></defs><rect width="1280" height="620" rx="24" fill="url(#bg)"/>''')
    text(45,58,'Qwen-VLM',34,NAVY,'700');text(45,88,'Industrial anomaly adaptation & parameter-efficient post-training',17,MUTED)
    blocks.append('<rect x="1010" y="34" width="225" height="42" rx="21" fill="#E1EFED"/>');text(1122,61,'8B backbone · audited evidence',13,TEAL,'700','middle')
    text(45,139,'01  LANGUAGE-BACKBONE MIGRATION',12,BLUE,'700')
    box(45,160,160,115,'Image',['Industrial input','No GT mask at inference'])
    box(250,160,225,115,'Shared visual front end',['Frozen ImageBind','BigDec anomaly map'])
    box(520,160,225,115,'Multimodal bridge',['CLS projection','Map → prompt tokens'])
    box(790,160,215,115,'Qwen language model',['Base frozen; optional q/v LoRA','A / C / D / Cpair / R / S'],TEAL)
    box(1050,160,185,115,'Outputs',['Decision / score','Structured bbox'],TEAL)
    for a,b in [(205,250),(475,520),(745,790),(1005,1050)]:arrow(a,217,b-8,217)
    text(250,300,'Vicuna ↔ Qwen: matched visual input, bridge initialization, data order and training budget.',13,MUTED)
    text(45,351,'02  NATIVE QWEN VISUAL PATHWAY  ·  SEPARATE EXPERIMENT',12,TEAL,'700')
    box(45,372,160,104,'Image',['Native processor'],TEAL)
    box(250,372,355,104,'Qwen3-VL visual pathway',['Frozen vision encoder, merger and base','NF4 language base + q/k/v/o LoRA'],TEAL)
    box(650,372,355,104,'Fixed-exposure task ablation',['RR: region × 2   |   DD: decision × 2','RD: region + decision'],TEAL)
    box(1050,372,185,104,'Evaluation',['Generated answers','Likelihood scores'],TEAL)
    for a,b in [(205,250),(605,650),(1005,1050)]:arrow(a,424,b-8,424)
    blocks.append('<rect x="45" y="518" width="1190" height="63" rx="14" fill="#142840"/>')
    text(66,544,'EVIDENCE CHAIN',11,'#94B6DF','700')
    text(66,565,'Pipeline repair  →  paired seeds  →  LoRA / ranking / structured targets  →  reload audits  →  frozen external evaluation',15,'white')
    blocks.append('</svg>')
    (HERE/'architecture.svg').write_text('\n'.join(blocks))
    # A raster preview makes native image inspection possible without extra dependencies.
    try:
        import cairosvg
        cairosvg.svg2png(url=str(HERE/'architecture.svg'),write_to=str(HERE/'architecture.png'),output_width=1600)
    except ImportError:
        pass


def main():
    (HERE/'evidence.json').write_text(json.dumps(DATA,ensure_ascii=False,indent=2)+'\n')
    language_transfer();development_vs_external();structured_output();native_qlora();architecture()
    for p in sorted(HERE.iterdir()):
        if p.is_file():print(p.name,p.stat().st_size)

if __name__=='__main__':main()
