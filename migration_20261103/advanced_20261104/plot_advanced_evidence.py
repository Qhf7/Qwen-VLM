"""Static scientific figures from recomputed VisA summaries only.

This script never reads external evaluation results or opens a model, image,
mask, checkpoint or prediction file. It records every summary's SHA256 and
explicit panel denominators. Incomplete S results are skipped by default; an
explicit option may plot listed seeds as PARTIAL, never as three seeds.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SEEDS = (42, 43, 44)
COLORS = {42: '#0072B2', 43: '#D55E00', 44: '#009E73'}
PREDICTION_N, NORMAL_N, ANOMALY_N = 380, 172, 208
HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def mean_sd(values):
    values = list(values)
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values) > 1 else None)


def get_rows(advanced, historical, mode):
    """Normalize C's historical generated recall/FPR field names."""
    rows = {}
    for row in historical.get('rows', []):
        if row['arm'] == 'C' and row['mode'] == mode and row['seed'] in SEEDS:
            rows['C', row['seed']] = dict(auroc=row['auroc'], recall=row['gen_recall'],
                                          fpr=row['gen_fpr'], step=row['step'])
    for row in advanced.get('rows', []):
        if row['mode'] == mode and row['arm'] in ('D', 'Cpair', 'R') and row['seed'] in SEEDS:
            rows[row['arm'], row['seed']] = {key: row[key] for key in ('auroc', 'recall', 'fpr', 'step')}
    for key, row in rows.items():
        require(all(finite(row[metric]) and 0 <= row[metric] <= 1 for metric in ('auroc', 'recall', 'fpr')),
                'Invalid metric: ' + str(key))
    return rows


def validate_recomputed(advanced):
    require(advanced.get('external_results_opened') is False, 'Expected VisA-only recomputed advanced summary')
    records = advanced.get('prediction_recomputations', [])
    require(records and all(record.get('matches') is True and record.get('n') == PREDICTION_N
                            and len(record.get('sha256', '')) == 64 for record in records),
            'Advanced summary lacks complete380-row recomputation evidence')


def annotate(ax, label):
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontsize=11, weight='bold')


def paired_panel(ax, rows, left, right, mode):
    seeds = [seed for seed in SEEDS if (left, seed) in rows and (right, seed) in rows]
    for seed in seeds:
        values = [rows[left, seed]['auroc'] * 100, rows[right, seed]['auroc'] * 100]
        ax.plot([0, 1], values, marker='o', markersize=5, lw=1.3, color=COLORS[seed], label='Seed%d' % seed)
    if seeds:
        means = [statistics.mean(rows[arm, seed]['auroc'] * 100 for seed in seeds) for arm in (left, right)]
        ax.plot([0, 1], means, color='#222222', lw=2.1, marker='D', markersize=5, label='Seed mean', zorder=4)
        deltas = [(rows[right, seed]['auroc'] - rows[left, seed]['auroc']) * 100 for seed in seeds]
        ax.text(.04, .12, 'Mean paired change: %+.2f pp' % statistics.mean(deltas),
                transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=2))
    ax.set_xticks([0, 1], [left, right])
    ax.set_xlim(-.2, 1.2)
    ax.set_ylabel('Image AUROC (%)')
    ax.set_title('%s to %s | %s | %d/3 seeds%s' %
                 (left, right, mode, len(seeds), ' | PARTIAL' if len(seeds) < 3 else ''), fontsize=10)
    ax.grid(axis='y', alpha=.22)
    return dict(panel='%s_to_%s_auroc' % (left, right), mode=mode, seeds=seeds,
                n_seeds=len(seeds), state='complete_3seed' if len(seeds) == 3 else 'partial',
                denominator=dict(images_per_seed=PREDICTION_N, normal=NORMAL_N, anomaly=ANOMALY_N),
                source='historical/advanced recomputed overall image AUROC rows',
                pair_values={str(seed): {arm: rows[arm, seed]['auroc'] for arm in (left, right)} for seed in seeds},
                paired_change_percentage_points=mean_sd((rows[right, seed]['auroc'] - rows[left, seed]['auroc']) * 100 for seed in seeds))


def tradeoff_panel(ax, rows, left, right, mode):
    seeds = [seed for seed in SEEDS if (left, seed) in rows and (right, seed) in rows]
    for seed in seeds:
        xs = [rows[arm, seed]['fpr'] * 100 for arm in (left, right)]
        ys = [rows[arm, seed]['recall'] * 100 for arm in (left, right)]
        ax.plot(xs, ys, color=COLORS[seed], lw=1.2, alpha=.75)
        ax.scatter(xs[0], ys[0], marker='o', facecolors='white', edgecolors=COLORS[seed], s=43, zorder=3)
        ax.scatter(xs[1], ys[1], marker='^', color=COLORS[seed], s=43, zorder=3)
        ax.annotate('', xy=(xs[1], ys[1]), xytext=(xs[0], ys[0]),
                    arrowprops=dict(arrowstyle='->', lw=.8, color=COLORS[seed]))
    ax.set_xlabel('Generated false-positive rate (%) [normal n=172]')
    ax.set_ylabel('Generated anomaly recall (%) [anomaly n=208]')
    ax.set_title('%s (circle) to %s (triangle) | %s' % (left, right, mode), fontsize=10)
    ax.grid(alpha=.22)
    ax.legend(handles=[Line2D([], [], marker='o', linestyle='', markerfacecolor='white',
                            markeredgecolor='#222222', label=left),
                       Line2D([], [], marker='^', linestyle='', color='#222222', label=right)],
              fontsize=8, loc='best', frameon=False)
    return dict(panel='%s_to_%s_recall_fpr' % (left, right), mode=mode, seeds=seeds, n_seeds=len(seeds),
                denominator=dict(recall_anomalies_per_seed=ANOMALY_N, false_positive_normals_per_seed=NORMAL_N,
                                 invalid_answers_retained=True),
                source='generated initial Yes/No decisions, not a threshold sweep or score ROC curve',
                pair_values={str(seed): {arm: dict(recall=rows[arm, seed]['recall'], fpr=rows[arm, seed]['fpr'])
                                         for arm in (left, right)} for seed in seeds},
                recall_change_percentage_points=mean_sd((rows[right, seed]['recall'] - rows[left, seed]['recall']) * 100 for seed in seeds),
                fpr_change_percentage_points=mean_sd((rows[right, seed]['fpr'] - rows[left, seed]['fpr']) * 100 for seed in seeds))


def save_figure(fig, output_base):
    png, pdf = output_base.with_suffix('.png'), output_base.with_suffix('.pdf')
    fig.savefig(png, dpi=220, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)
    return dict(png=str(png), pdf=str(pdf), png_sha256=sha256(png), pdf_sha256=sha256(pdf))


def plot_classification(advanced, historical, mode, output):
    rows = get_rows(advanced, historical, mode)
    # Recheck historical C values against the already-audited advanced pairing;
    # this catches a summary from another experiment being plotted by accident.
    for pair in advanced.get('paired_deltas', []):
        if pair['mode'] != mode:
            continue
        right, left = ('D', 'C') if pair['contrast'] == 'D_minus_C' else ('R', 'Cpair')
        seed = pair['seed']
        if (left, seed) in rows and (right, seed) in rows:
            require(all(abs((rows[right, seed][key] - rows[left, seed][key]) - pair['delta'][key]) < 1e-12
                        for key in ('auroc', 'recall', 'fpr')), 'Plot input differs from recomputed same-seed deltas')
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), constrained_layout=True)
    panels = [
        paired_panel(axes[0, 0], rows, 'C', 'D', mode),
        paired_panel(axes[0, 1], rows, 'Cpair', 'R', mode),
        tradeoff_panel(axes[1, 0], rows, 'C', 'D', mode),
        tradeoff_panel(axes[1, 1], rows, 'Cpair', 'R', mode)]
    for axis, label in zip(axes.flat, ('A', 'B', 'C', 'D')):
        annotate(axis, label)
    handles = [Line2D([], [], marker='o', color=COLORS[seed], label='Seed%d' % seed) for seed in SEEDS]
    handles.append(Line2D([], [], marker='D', color='#222222', label='Seed mean (AUROC panels)'))
    axes[0, 0].legend(handles=handles, fontsize=8, frameon=False, loc='upper left')
    fig.suptitle('VisA development validation | %s checkpoints | summary: %s\n'
                 'Same 380 images per run; seed 42/43/44 are training repeats. Best is selected on this validation set.'
                 % (mode, advanced['state']), fontsize=11)
    fig.supxlabel('C to D: joint bridge+LoRA vs LoRA-only. Cpair to R: matched paired CE vs CE+ranking.\n'
                  'Recall/FPR use generated answers; higher FPR means more normal-image false alarms.', fontsize=8)
    return dict(artifacts=save_figure(fig, output / ('classification_' + mode)), panels=panels)


def structured_panels(summary, output, mode, allow_partial):
    runs = summary.get('runs', [])
    seeds = [run['seed'] for run in runs]
    require(len(set(seeds)) == len(seeds) and set(seeds).issubset(SEEDS), 'Invalid structured seed list')
    complete = set(seeds) == set(SEEDS) and summary.get('state') == 'complete' and summary.get('fresh_reload_all_passed') is True
    if not complete and not allow_partial:
        return dict(state='skipped_partial', seeds=seeds, n_seeds=len(seeds),
                    reason='S requires all3 seeds and fresh reloads; use --allow-partial-structured to label a partial plot explicitly')
    require(runs, 'Structured summary has no completed runs')
    for run in runs:
        for condition in ('initial', mode):
            metrics = run['modes'][condition]['metrics']
            require(metrics['overall']['scores']['n'] == PREDICTION_N
                    and metrics['localization']['n_anomaly'] == ANOMALY_N
                    and metrics['localization']['n_normal'] == NORMAL_N, 'Structured panel denominator differs')
    # Separate bbox IoU and hit-rate panels; all-anomaly denominators penalize
    # invalid/No/missing boxes. Teacher baselines retain explicit control names.
    labels = ('C structured', 'S ' + mode, 'Teacher box\n(always)', 'Teacher box\n(C score gated)')
    keys = ('anomaly_bbox_iou_mean_all', 'anomaly_iou_at_05_all')
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), constrained_layout=True)
    panels = []
    for axis, key, ylabel in zip(axes, keys, ('Mean union-box IoU [all 208 anomalies]', 'IoU >=0.5 success rate [all 208 anomalies]')):
        values = []
        for run in runs:
            initial = run['modes']['initial']['metrics']['localization'][key]
            final = run['modes'][mode]['metrics']['localization'][key]
            teacher = run['teacher_baseline']
            row = [initial, final, teacher['unconditional_box'][key], teacher['initial_score_gated'][key]]
            require(all(finite(value) and 0 <= value <= 1 for value in row), 'Invalid localization metric')
            values.append(row)
            axis.plot([0, 1], row[:2], color=COLORS[run['seed']], marker='o', lw=1.1, alpha=.8)
            axis.scatter([2, 3], row[2:], color=COLORS[run['seed']], marker='o', s=30, alpha=.8)
        means = [statistics.mean(row[index] for row in values) for index in range(4)]
        axis.plot([0, 1], means[:2], marker='D', color='#222222', lw=2, markersize=5)
        axis.scatter([2, 3], means[2:], marker='D', color='#222222', s=34, zorder=4)
        axis.axvline(1.5, color='#999999', lw=.7, linestyle=':')
        axis.set_xticks(range(4), labels, fontsize=8)
        axis.set_ylim(bottom=0)
        axis.set_ylabel(ylabel)
        axis.grid(axis='y', alpha=.22)
        panels.append(dict(panel=key, seeds=seeds, n_seeds=len(seeds),
                           denominator=dict(anomalies_per_seed=ANOMALY_N, malformed_no_missing_as_zero=True),
                           values={str(run['seed']): dict(zip(labels, row)) for run, row in zip(runs, values)},
                           teacher_baseline_audit={str(run['seed']):run['teacher_baseline_audit'] for run in runs},
                           source='structured recomputed localization plus explicitly labeled stored/recomputed teacher controls'))
    annotate(axes[0], 'A'); annotate(axes[1], 'B')
    title_state = 'complete 3 seeds' if complete else 'PARTIAL %d/3 seeds; fresh reloads may be pending' % len(seeds)
    fig.suptitle('Structured localization | %s | %s\nAdditional mask supervision and answer tokens; union boxes, not segmentation.' % (mode, title_state), fontsize=11)
    fig.supxlabel('C structured uses original C weights with the same schema prompt. Teacher box uses fixed map 95th percentile.\n'
                  'Teacher control verification status is recorded in the provenance JSON.', fontsize=8)
    axes[0].legend(handles=[Line2D([], [], marker='o', color=COLORS[seed], label='Seed%d'%seed) for seed in seeds]
                          + [Line2D([], [], marker='D', color='#222222', label='Seed mean')], fontsize=8, frameon=False)
    return dict(state='complete_3seed' if complete else 'partial', seeds=seeds, n_seeds=len(seeds),
                artifacts=save_figure(fig, output / ('structured_' + mode)), panels=panels)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=HERE)
    ap.add_argument('--advanced-summary', type=Path)
    ap.add_argument('--historical-summary', type=Path)
    ap.add_argument('--structured-summary', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--mode', choices=('best', 'final', 'both'), default='both')
    ap.add_argument('--allow-partial-structured', action='store_true')
    args = ap.parse_args()
    root = args.root.resolve()
    advanced_path = args.advanced_summary or root / 'summary.json'
    historical_path = args.historical_summary or root.parent / 'lora_20261104/summary.json'
    structured_path = args.structured_summary or root / 'structured_summary/summary.json'
    # Restrict to named local VisA summary inputs; external data is never scanned.
    for path in (advanced_path, historical_path, structured_path):
        require('external' not in str(path).lower(), 'External summaries are outside this script scope')
    advanced, historical = read(advanced_path), read(historical_path)
    validate_recomputed(advanced)
    require(historical.get('state') == 'complete', 'Historical C summary must be complete')
    output = args.out or root / 'figures'
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    sources = dict(advanced=dict(path=str(advanced_path), sha256=sha256(advanced_path), state=advanced['state']),
                   historical=dict(path=str(historical_path), sha256=sha256(historical_path), state=historical['state']))
    modes = ('best', 'final') if args.mode == 'both' else (args.mode,)
    figures = {mode:plot_classification(advanced, historical, mode, output) for mode in modes}
    structured = {}
    if structured_path.exists():
        spec = read(structured_path)
        sources['structured'] = dict(path=str(structured_path), sha256=sha256(structured_path), state=spec['state'], seeds=spec.get('seeds', []))
        structured = {mode:structured_panels(spec, output, mode, args.allow_partial_structured) for mode in modes}
    else:
        structured = dict(state='missing', reason='No structured summary available')
    for figure in figures.values():
        for panel in figure['panels']:
            panel['source_summary_sha256'] = {key:sources[key]['sha256'] for key in
                                               (('advanced', 'historical') if panel['panel'].startswith('C_to_D') else ('advanced',))}
    for figure in structured.values():
        if isinstance(figure, dict) and 'panels' in figure:
            for panel in figure['panels']:
                panel['source_summary_sha256'] = {'structured':sources['structured']['sha256']}
    provenance = dict(schema='advanced-visa-evidence-figures-v1', source_summaries=sources,
                      script_sha256=sha256(__file__), figures=figures, structured=structured,
                      scope='Saved recomputed VisA val380 summaries only; zero external/model/image/mask queries',
                      denominators=dict(images=380,normal=172,anomaly=208,expected_training_seeds=list(SEEDS)),
                      interpretation='Checkpoint selection and reporting share development validation. Three seeds are repeats, not independent datasets. No significance or reasoning improvement claim.')
    (output / 'figure_provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    notes = ['# Figure provenance and denominators', '',
             'A/B: paired image AUROC on the same VisA val380 images (172normal,208anomaly), by training seed.',
             'C/D: generated decision recall on208 anomalies and false-positive rate on172 normals; invalid answers remain in the denominator. These are paired operating points, not ROC curves.',
             'Best is selected on this development validation set; final is fixed update500. Seed repeats are not independent datasets.',
             'Structured localization uses all208 anomalies, with invalid/No/missing boxes as zero. Teacher control verification status is recorded per seed; union boxes are not segmentation.',
             'Each panel has its source-summary SHA256, seeds, denominator and plotted values in figure_provenance.json.', '']
    for key, value in sources.items():
        notes += [key + ': ' + value['path'], 'SHA256: ' + value['sha256'], '']
    (output/'FIGURE_NOTES.md').write_text('\\n'.join(notes)+'\\n')
    print(json.dumps(dict(state='complete', output=str(output), classification_modes=list(figures),
                         structured={mode: value['state'] for mode, value in structured.items()} if 'state' not in structured else structured)))


if __name__ == '__main__':
    main()
