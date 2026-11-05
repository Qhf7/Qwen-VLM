"""Plot completed same-budget full-validation ROI conditions only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    old = json.loads((root / 'roi_full/run.json').read_text())
    new = json.loads((root / 'student_roi_20261104/roi_full/run.json').read_text())
    if old['status'] != 'complete' or new['status'] != 'complete' or not new['replay']['passed']:
        raise ValueError('Complete student validation and exact control replay are required')
    names = ['Center', 'Random', 'BigDec ROI', 'Qwen GT ROI', 'Qwen KD ROI']
    records = [old['results'][k]['metrics']['overall'] for k in ('original_center_crop', 'original_random_crop', 'original_teacher_roi')]
    records += [new['results'][k]['metrics']['overall'] for k in ('pre_gt', 'pre_kd')]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#90A4AE', '#90A4AE', '#607D8B', '#1976D2', '#00897B']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    series = [('Image AUROC', [r['scores']['image_auroc'] for r in records]),
              ('Generated binary macro F1', [r['generated']['macro_f1'] for r in records]),
              ('Generated anomaly recall / FPR', None)]
    for ax, (title, values) in zip(axes, series):
        if values is not None:
            bars = ax.bar(names, values, color=colors)
            ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
        else:
            x = np.arange(len(names))
            bars = ax.bar(x - .18, [r['generated']['recall'] for r in records], .36, label='Recall', color=colors)
            ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
            bars = ax.bar(x + .18, [r['generated']['fpr'] for r in records], .36, label='FPR', color='#D88E6A')
            ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
            ax.set_xticks(x, names)
            ax.legend(loc='upper right')
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', labelrotation=28)
        ax.set_ylabel('Validation metric (0–1)')
    fig.suptitle('Qwen student maps → native Qwen ROI answers · full val380', fontsize=15)
    fig.text(.5, -.06, 'Same 2×448 views / 392 visual tokens / 470 prompt tokens / 1⁄9 crop area. '
             'Students use no ImageBind/BigDec at inference; supervised development split, seed42.', ha='center', fontsize=9)
    path = root / 'student_roi_20261104/student_roi_results.png'
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(path)


if __name__ == '__main__':
    main()
