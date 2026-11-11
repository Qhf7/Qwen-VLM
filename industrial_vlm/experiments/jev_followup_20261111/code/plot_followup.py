#!/usr/bin/env python3
"""Render audited stage05/06/07 results on the designated AutoDL server.

Only new figure assets, figure provenance and FIGURES.md are written.
No inference, fitting, API requests, RGB or GT masks are accessed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np

HOSTNAME='autodl-container-132444827f-9c594038'
BLUE='#2471A3'
ORANGE='#D77A24'
GREEN='#268168'
PURPLE='#8D62A6'
GRAY='#69727B'

def require(condition,message):
    if not condition:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def publish(path,value):
    path=Path(path)
    if path.exists():
        require(canonical(read(path))==canonical(value),'Existing figure manifest differs')
        return
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())

def style_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#A9B0B7')
    ax.spines['bottom'].set_color('#A9B0B7')
    ax.tick_params(colors='#34414D',labelsize=10)
    ax.yaxis.grid(True,color='#DFE4E8',linewidth=.7)
    ax.set_axisbelow(True)

def draw(args):
    require(socket.gethostname()==HOSTNAME,'Plotting must execute on designated server')
    root=args.root.resolve()
    paths={name:root/'stages'/filename for name,filename in [
        ('stage05','05_evaluation.json'),('stage06','06_routing_results.json'),
        ('stage07','07_calibration_results.json')]}
    source_hashes={name:sha(path) for name,path in paths.items()}
    roi,routing,calibration=(read(paths[name]) for name in ('stage05','stage06','stage07'))
    require(all(d.get('status')=='complete' for d in (roi,routing,calibration)),'Need complete stage05/06/07')
    require(len(roi['records'])==120 and sum(r['label'] for r in roi['records'])==60,'120/60 denominators differ')
    require(routing['stage05_sha256']==source_hashes['stage05']
            and calibration['stage05_sha256']==source_hashes['stage05'],'Stages are not linked to same frozen evaluation')
    require(roi['confirmation_opened'] is False and routing['confirmation_opened'] is False
            and calibration['confirmation_opened'] is False,'Confirmation isolation differs')
    require(routing['offline_replay_only'] is True
            and calibration['independent_calibration_task_completed'] is False,'Research scope flags differ')
    methods=roi['groups']['all120']['methods']
    for name in ('roi','duplicate'):
        require(methods[name]['pooled']['n_anomaly']==methods[name]['pooled']['n_normal']==60,'ROI primary denominators changed')
    route_methods=routing['groups']['all120']['methods']
    font_path=font_manager.findfont('DejaVu Sans',fallback_to_default=False)
    glyph_font=FT2Font(font_path)
    required_chars='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,:;()[]+-=/%_'
    require(all(glyph_font.get_char_index(ord(char))>0 for char in required_chars),'Figure font misses required English glyphs')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,
        'axes.labelsize':11,'axes.titlesize':12,'axes.titleweight':'bold',
        'svg.fonttype':'none','axes.unicode_minus':False,'figure.facecolor':'white',
        'savefig.facecolor':'white','pdf.fonttype':42})
    figure=plt.figure(figsize=(18,8.2))
    grid=figure.add_gridspec(1,3,left=.055,right=.985,top=.79,bottom=.24,
                             width_ratios=[1.02,1.16,1.32],wspace=.35)
    agrid=grid[0].subgridspec(2,1,height_ratios=[1.75,1],hspace=.62)
    arate=figure.add_subplot(agrid[0]);aiou=figure.add_subplot(agrid[1])
    b=figure.add_subplot(grid[1]);c=figure.add_subplot(grid[2])
    figure.suptitle('Jev follow-up: fixed ROI gates, routing replay and OOF calibration',
                     fontsize=18,fontweight='bold',x=.055,ha='left',y=.965,color='#1F2D3A')
    figure.text(.055,.915,'Viewed development set: 120 sources | 60 anomalies + 60 normal images | fixed candidate and decision rules',
                fontsize=12,color='#536575',ha='left')
    # Panel A uses separate axes so IoU is never plotted as a percentage.
    values={name:methods[name]['pooled'] for name in ('duplicate','roi')}
    x=np.arange(2);width=.31
    for offset,name,color,label in [(-width/2,'duplicate',BLUE,'Duplicate'),(width/2,'roi',ORANGE,'ROI')]:
        heights=[values[name]['anomaly_retained_recall'],values[name]['normal_positive_box_rate']]
        bars=arate.bar(x+offset,heights,width,color=color,label=label)
        for rect,value in zip(bars,heights):
            arate.text(rect.get_x()+rect.get_width()/2,value+.025,f'{value:.3f}',ha='center',fontsize=10.5)
    arate.set_xticks(x,['Anomaly recall','Normal FPR'])
    arate.set_ylabel('Fraction (0-1)')
    arate.set_ylim(0,1.03);arate.yaxis.set_major_locator(MultipleLocator(.25))
    arate.set_title('(A) ROI vs duplicate',loc='left',pad=15)
    arate.legend(loc='upper right',frameon=False,ncol=2,fontsize=9.5,handlelength=1)
    style_axis(arate)
    ious=[values[name]['anomaly_bbox_iou_mean_all'] for name in ('duplicate','roi')]
    bars=aiou.bar([0,1],ious,.55,color=[BLUE,ORANGE])
    for rect,value in zip(bars,ious):
        aiou.text(rect.get_x()+rect.get_width()/2,value+.003,f'{value:.5f}',ha='center',fontsize=10.5)
    aiou.set_xticks([0,1],['Duplicate','ROI'])
    aiou.set_ylabel('Mean IoU (0-1)')
    aiou.set_ylim(0,.10);aiou.yaxis.set_major_locator(MultipleLocator(.05))
    aiou.set_title('All 60 anomalies, including rejected boxes',loc='left',fontsize=9.8,fontweight='normal',pad=8)
    style_axis(aiou)
    # Panel B plots exact logical budgets and quality values: no point jitter.
    route_info=[('duplicate','Duplicate',BLUE,'o'),('roi','ROI',ORANGE,'s'),
                ('sequential_uncertain','Uncertainty route',GREEN,'D'),
                ('geometry_small','Small-area route',PURPLE,'^')]
    for method,label,color,marker in route_info:
        data=route_methods[method];metric=data['performance']['pooled'];budget=data['budget']
        b.scatter(budget['logical_successful_requests'],metric['anomaly_bbox_iou_mean_all'],
                  s=95,color=color,marker=marker,edgecolors='white',linewidths=.8,zorder=5,label=label)
    random_names=['matched_random_s42','matched_random_s43','matched_random_s44']
    b.scatter([route_methods[name]['budget']['logical_successful_requests'] for name in random_names],
              [route_methods[name]['performance']['pooled']['anomaly_bbox_iou_mean_all'] for name in random_names],
              marker='x',s=100,color=GRAY,linewidths=1.4,zorder=6,label='Matched random (3 seeds)')
    for method,label,xytext in [('duplicate','Duplicate: 120',(3,-19)),('roi','ROI: 120',(3,12)),
                              ('sequential_uncertain','Uncertainty: 162',(-84,15)),
                              ('geometry_small','Small-area: 153',(-109,-17))]:
        data=route_methods[method];point=(data['budget']['logical_successful_requests'],data['performance']['pooled']['anomaly_bbox_iou_mean_all'])
        b.annotate(label,point,xytext=xytext,textcoords='offset points',fontsize=9.2,
                   color='#2D3E4B',arrowprops=dict(arrowstyle='-',lw=.6,color='#9AA5AE'))
    b.set_title('(B) Routing quality vs logical cost',loc='left',pad=15)
    b.set_xlabel('Successful requests for 120 queries')
    b.set_ylabel('Mean IoU, all 60 anomalies')
    b.set_xlim(115,170);b.set_xticks([120,140,153,162])
    b.set_ylim(.0682,.0702);b.yaxis.set_major_locator(MultipleLocator(.0005))
    b.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    style_axis(b)
    b.text(.02,.025,'Y axis truncated to show small differences.\nOffline replay; no online latency claim.',
            transform=b.transAxes,fontsize=8.7,color='#607080',va='bottom')
    b.legend(loc='upper left',frameon=False,ncol=1,fontsize=8.5,
             borderaxespad=.4,labelspacing=.35,handletextpad=.4)
    # Panel C reports Brier only: probability quality and ranking are distinct.
    model_order=['jev_0shot','jev_1shot','jev_4shot','qwen_base','qwen_decision_decision']
    labels=['Jev\n0-shot','Jev\n1-shot','Jev\n4-shot','Qwen\nbase','Qwen\nDD']
    raw=[calibration['models'][name]['raw']['pooled']['brier'] for name in model_order]
    oof=[calibration['models'][name]['oof_logistic']['pooled']['brier'] for name in model_order]
    positions=np.arange(len(model_order));width=.34
    for shift,values_,color,label in [(-width/2,raw,BLUE,'Raw'),(width/2,oof,ORANGE,'OOF logistic')]:
        bars=c.bar(positions+shift,values_,width,color=color,label=label)
        for rect,value in zip(bars,values_):
            c.text(rect.get_x()+rect.get_width()/2,value+.007,f'{value:.3f}',
                   ha='center',va='bottom',rotation=90,fontsize=9.5)
    c.set_title('(C) Whole-image calibration diagnostic',loc='left',pad=15)
    c.set_ylabel('Brier score (lower is better)')
    c.set_xticks(positions,labels)
    c.set_ylim(0,.37);c.yaxis.set_major_locator(MultipleLocator(.1))
    style_axis(c)
    c.legend(loc='upper right',frameon=False,fontsize=9.5)
    c.text(.02,.975,'Shared source-level 5-fold split; C=1, L2.\nQwen DD Brier slightly worsens after OOF.',
            transform=c.transAxes,fontsize=9.2,color='#536575',ha='left',va='top')
    # Self-contained scientific caption, with limitations visible on the image.
    caption=[
        'Scope: the Teacher trained on 96/120 query sources; the remaining 24 contain only 2 normal images. These are viewed development diagnostics.',
        'Candidates: 60/120 union-95% boxes cover the full image and have identical query/crop pixels. ROI mean-IoU delta = +0.00118; paired 95% CI crosses zero.',
        'Budget: panel B shows replay budgets, not actual bills; all 240 paired calls were collected (+38 failed attempts). Random seed 44 overlaps the uncertainty point.',
        'Calibration: whole-image scores only; OOF protocol formed after ROI evaluation but before calibration fitting. No independent calibration or confirmation was completed.'
    ]
    for index,text in enumerate(caption):
        figure.text(.055,.137-index*.027,text,fontsize=9.2,color='#536575',ha='left',va='top')
    directory=root/'figures';directory.mkdir(exist_ok=True)
    png=directory/'followup_roi_routing_calibration.png';svg=png.with_suffix('.svg')
    require(not png.exists() and not svg.exists(),'Figure output exists; preserve it or choose a new reviewed script')
    for path,format_ in [(png,'png'),(svg,'svg')]:
        figure.savefig(path,format=format_,dpi=300,metadata={'Title':'Jev follow-up development diagnostics',
                       'Description':'Frozen stage05/06/07 results, full anomaly/normal denominators; no independent validation claim'})
    plt.close(figure)
    from PIL import Image
    with Image.open(png) as image:
        dimensions=list(image.size)
        require(image.width>=5000 and image.height>=2000,'Rendered figure too small')
    require(source_hashes=={name:sha(path) for name,path in paths.items()},'Frozen inputs changed while rendering')
    diagnostics=dict(schema='jev-followup-figure-provenance-v1',timestamp_utc=datetime.now(timezone.utc).isoformat(),
        hostname=socket.gethostname(),source_sha256=source_hashes,script_sha256=sha(__file__),
        versions={name:importlib.metadata.version(name) for name in ('matplotlib','numpy','Pillow')},
        font='DejaVu Sans',font_path=font_path,font_sha256=sha(font_path),font_glyph_check=True,
        figure_files={path.name:dict(sha256=sha(path),bytes=path.stat().st_size) for path in (png,svg)},
        png_dimensions=dimensions,complete_denominators=dict(anomaly=60,normal=60),
        ROI_IoU_unit='0..1 fraction, not percent; separate axis from recall/FPR',
        routing_y_axis_truncated=True,routing_points_jittered=False,
        displayed_confidence_intervals=False,caption_reports_roi_paired_interval_crosses_zero=True,
        confirmation_opened=False,new_api_calls=0,model_calls=0,gt_masks_opened=0,
        existing_frozen_scientific_artifacts_modified=False)
    publish(directory/'followup_roi_routing_calibration.provenance.json',diagnostics)
    lines=['# Jev follow-up figures','',
        '## Files','',
        '- figures/followup_roi_routing_calibration.png: 300-DPI raster, '+str(dimensions[0])+' x '+str(dimensions[1])+' pixels.',
        '- figures/followup_roi_routing_calibration.svg: scalable figure with editable English text.',
        '- figures/followup_roi_routing_calibration.provenance.json: source, script, font and output hashes.','',
        '## What the three panels show','',
        '(A) ROI and duplicate gates use full denominators: all60 anomalies for recall and mean IoU; all60 normal images for FPR. Rates and IoU are shown in original0..1 units on separate axes. Rejected anomaly boxes contribute zero IoU.',
        '(B) Fixed routing policies compare all-anomaly mean IoU with successful logical request count for120 queries. The Y axis is explicitly truncated. Points are exact, without jitter or interpolating curves. Matched-random seed44 and the uncertainty route coincide. Logical replay budgets do not undo the240 calls and38 failures actually collected.',
        '(C) Whole-image raw and5fold OOF logistic Brier scores share the same120 source-level folds. Calibration is fixedC1/L2 and not a foundation-model training result. QwenDD Brier worsens slightly after OOF; the plot preserves this negative result.','',
        '## Caption','']
    lines+=caption
    lines+=['','## Exact values','',
        '| Gate | Anomaly recall | Normal FPR | All-anomaly mean IoU |','|---|---:|---:|---:|']
    for name in ('duplicate','roi'):
        value=values[name]
        lines.append(f"| {name} | {value['anomaly_retained_recall']:.9f} | {value['normal_positive_box_rate']:.9f} | {value['anomaly_bbox_iou_mean_all']:.9f} |")
    lines+=['','| Routing policy | Logical successful requests | All-anomaly mean IoU | Anomaly recall | Normal FPR |','|---|---:|---:|---:|---:|']
    for name,data in route_methods.items():
        value=data['performance']['pooled']
        lines.append(f"| {name} | {data['budget']['logical_successful_requests']} | {value['anomaly_bbox_iou_mean_all']:.9f} | {value['anomaly_retained_recall']:.9f} | {value['normal_positive_box_rate']:.9f} |")
    lines+=['','| Score | Raw Brier | OOF logistic Brier |','|---|---:|---:|']
    for name,r,o in zip(model_order,raw,oof):
        lines.append(f'| {name} | {r:.9f} | {o:.9f} |')
    lines+=['','## Interpretation boundaries','',
        '- The figure shows diagnostic trade-offs, not an independent test or a production guarantee.',
        '- ROI/duplicate is the matched new-arm comparison. Historical Jev rendering differs, and Qwen max_pixels524288 differs from Jev448; cross-model values are not an architectural causal comparison.',
        '- The ROI paired mean-IoU difference and interval remain in stage05; the caption says the interval crosses zero. No error bars are drawn; omission is not a claim of significance.',
        '- Stage07 raw/OOF uses p>=0.5; native Qwen default uses strict logodds>0. Equality cases differ and are documented in the calibration report.',
        '- Font is verified DejaVu Sans. All figure text is English to avoid missing CJK glyphs.','',
        '## Provenance','',
        f"Script SHA256: {diagnostics['script_sha256']}",
        *[f'{name} SHA256: {value}' for name,value in source_hashes.items()],
        'Generated on the designated AutoDL server. No API/model call, RGB or GT-mask read, or frozen scientific artifact change.','']
    report=root/'FIGURES.md'
    require(not report.exists(),'FIGURES.md already exists')
    with report.open('x',encoding='utf-8') as stream:
        stream.write('\n'.join(lines));stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(status='complete',png=str(png),svg=str(svg),
        documentation=str(report),png_dimensions=dimensions,font_glyph_check=True,
        source_sha256=source_hashes),ensure_ascii=False),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    draw(parser.parse_args())

if __name__=='__main__':
    main()
