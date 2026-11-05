"""Plot observed pilot training/evaluation JSON as PNG and vector PDF.

Example: python ops/plot_results.py --root /root/autodl-tmp/industrial_vlm \
    --output reports/figures
No model, image, or training dataset is loaded. Missing results are never zero-filled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARMS = {
    'quality_region': ('Coverage SFT', '#0072B2', '-'),
    'random_region': ('Random SFT', '#D55E00', '-'),
    'quality_decision': ('Decision SFT', '#009E73', '--'),
}
PANELS = [
    {'key': 'region_test', 'title': 'VisA region test', 'expected_n': 120,
     'region': True, 'runs': [
         ('base_region', 'Base', '#707070'),
         ('quality_region', 'Coverage SFT', '#0072B2'),
         ('random_region', 'Random SFT', '#D55E00')]},
    {'key': 'common_decision_test', 'title': 'VisA common decision test', 'expected_n': 120,
     'region': False, 'runs': [
         ('base_decision', 'Base', '#707070'),
         ('quality_region_decision_prompt', 'Region SFT', '#0072B2'),
         ('quality_decision', 'Decision SFT', '#009E73')]},
    {'key': 'external_pilot', 'title': 'MVTec external pilot', 'expected_n': 150,
     'region': True, 'runs': [
         ('external_base_region', 'Base', '#707070'),
         ('external_quality_region', 'Coverage SFT', '#0072B2')]},
]
ALIASES = {'quality_region_decision_prompt': 'quality_region_under_decision_prompt'}
MATCHED_KEYS = ('model', 'data_sha256', 'subset_sha256', 'max_pixels',
                'max_length', 'max_new_tokens', 'generation', 'quantization', 'limit')


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _rate(value: Any, source: str) -> float | None:
    number = _number(value)
    if number is not None and not 0 <= number <= 1:
        raise ValueError(f'Expected a rate within [0,1], got {value}: {source}')
    return number


def load_training(root: Path, warnings: list[str]) -> dict:
    result = {}
    for arm in ARMS:
        path = root / 'runs' / arm / 'metrics.jsonl'
        if not path.is_file():
            warnings.append(f'Missing training events: {path}')
            continue
        series: dict[str, dict[int, dict]] = {'train': {}, 'validation': {}}
        for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                warnings.append(f'Skipped malformed event at {path}:{line_number}')
                continue
            kind = event.get('event')
            if kind not in series:
                continue
            step = event.get('optimizer_step')
            loss = _number(event.get('loss' if kind == 'train' else 'validation_loss'))
            if not isinstance(step, int) or isinstance(step, bool) or step < 0 or loss is None or loss < 0:
                warnings.append(f'Skipped invalid {kind} event at {path}:{line_number}')
                continue
            if step in series[kind]:
                raise ValueError(f'Duplicate {kind} optimizer step {step} in {path}; resolve mixed run history')
            series[kind][step] = {'step': step, 'loss': loss,
                                  'samples': event.get('validation_samples') if kind == 'validation' else None}
        summary_path = path.parent / 'summary.json'
        summary = _read(summary_path) if summary_path.is_file() else {}
        result[arm] = {'events_file': str(path), 'sha256': _sha(path),
                       'status': summary.get('status', 'unknown'),
                       **{kind: [rows[s] for s in sorted(rows)] for kind, rows in series.items()}}
    return result


def load_evaluations(root: Path, warnings: list[str]) -> dict:
    runs = {}
    names = [name for panel in PANELS for name, _, _ in panel['runs']]
    for name in names:
        path = root / 'evaluations' / name / 'metrics.json'
        if not path.is_file() and name in ALIASES:
            path = root / 'evaluations' / ALIASES[name] / 'metrics.json'
        if not path.is_file():
            warnings.append(f'Missing evaluation: {name}')
            continue
        payload = _read(path)
        meta_path = path.with_name('meta.json')
        meta = _read(meta_path) if meta_path.is_file() else {}
        config = {**meta.get('config', {}), **payload.get('metadata', {})}
        if not config.get('complete', meta.get('status') == 'complete'):
            warnings.append(f'Excluded incomplete evaluation: {name}')
            continue
        overall = payload.get('overall')
        if not isinstance(overall, dict) or not isinstance(overall.get('n'), int) or overall['n'] < 1:
            warnings.append(f'Excluded evaluation without a positive sample count: {name}')
            continue
        if config.get('n_expected') is not None and overall['n'] != config['n_expected']:
            raise ValueError(f'Completed evaluation count differs from n_expected: {path}')
        for key in ('macro_f1', 'union_box_recall_iou_0.5', 'format_valid_rate'):
            _rate(overall.get(key), f'{path}:{key}')
        runs[name] = {'metrics_file': str(path), 'sha256': _sha(path), 'overall': overall,
                      'metadata': config, 'bootstrap': payload.get('bootstrap', {})}
    for panel in PANELS:
        present = [runs[name] for name, _, _ in panel['runs'] if name in runs]
        for key in MATCHED_KEYS:
            values = {json.dumps(r['metadata'][key], sort_keys=True) for r in present
                      if key in r['metadata']}
            if len(values) > 1:
                raise ValueError(f"Unmatched {key} in panel {panel['key']}; do not combine these conditions")
        counts = {r['overall']['n'] for r in present}
        if len(counts) > 1:
            raise ValueError(f"Different sample counts in panel {panel['key']}")
        if counts and counts != {panel['expected_n']}:
            warnings.append(f"{panel['key']}: actual n={next(iter(counts))}; planned n={panel['expected_n']}")
        if present and any(not r['metadata'].get('data_sha256') or not r['metadata'].get('subset_sha256')
                           for r in present):
            warnings.append(f"{panel['key']}: missing source/prompt hash metadata; matched inputs not fully established")
    return runs


def _style() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
        'axes.labelsize': 9, 'legend.fontsize': 8, 'xtick.labelsize': 8,
        'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.7, 'grid.linewidth': 0.5, 'grid.alpha': 0.25,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white',
    })


def _save(fig: Any, output: Path, name: str) -> list[str]:
    import matplotlib.pyplot as plt
    paths = []
    for suffix in ('png', 'pdf'):
        path = output / f'{name}.{suffix}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        paths.append(str(path))
    plt.close(fig)
    return paths


def _empty(ax: Any, message: str = 'No completed results') -> None:
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha='center', va='center', color='#666666')


def plot_training(training: dict, output: Path) -> list[str]:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    handles, labels = [], []
    for arm, (label, color, linestyle) in ARMS.items():
        if arm not in training:
            continue
        for index, kind in enumerate(('train', 'validation')):
            rows = training[arm][kind]
            if not rows:
                continue
            line, = axes[index].plot([r['step'] for r in rows], [r['loss'] for r in rows],
                                     color=color, linestyle=linestyle, linewidth=1.4,
                                     marker='o' if kind == 'validation' else None,
                                     markersize=4, label=label)
            if label not in labels:
                handles.append(line)
                labels.append(label)
    for index, (kind, title) in enumerate((('train', 'Training assistant CE'),
                                          ('validation', 'Validation assistant CE'))):
        ax = axes[index]
        ax.set_title(f'({chr(97 + index)}) {title}', loc='left')
        ax.set_xlabel('Optimizer update')
        ax.set_ylabel('Cross-entropy')
        ax.grid(axis='y')
        ax.set_ylim(bottom=0)
        if not any(training.get(arm, {}).get(kind) for arm in ARMS):
            _empty(ax, 'No recorded events')
        if kind == 'validation':
            counts = {r['samples'] for arm in training for r in training[arm][kind] if r['samples'] is not None}
            if len(counts) == 1:
                ax.text(0.98, 0.96, f'n = {next(iter(counts))}', transform=ax.transAxes,
                        ha='right', va='top', color='#555555', fontsize=8)
    fig.suptitle('Training dynamics | single-seed pilot', y=1.01, fontsize=12)
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False)
    fig.text(0.5, -0.01, 'Raw logged values. CE is task-specific; decision and region targets differ.',
             ha='center', fontsize=8, color='#555555')
    fig.subplots_adjust(top=0.78, bottom=0.18, wspace=0.28)
    return _save(fig, output, 'training_curves')


def _panel_title(panel: dict, runs: dict, index: int) -> str:
    counts = {runs[name]['overall']['n'] for name, _, _ in panel['runs'] if name in runs}
    n = str(next(iter(counts))) if len(counts) == 1 else 'unavailable'
    return f"({chr(97 + index)}) {panel['title']}\nn = {n}"


def _interval(run: dict, metric: str) -> tuple[float, float] | None:
    interval = run.get('bootstrap', {}).get('intervals', {}).get(metric, {})
    low, high = _number(interval.get('low')), _number(interval.get('high'))
    if low is None or high is None:
        return None
    if not 0 <= low <= high <= 1:
        raise ValueError(f'Invalid bootstrap interval in {run["metrics_file"]}: {metric}')
    return low, high


def _draw_interval(ax: Any, x: float, interval: tuple[float, float] | None, color: str = '#222222') -> None:
    if interval is not None:
        low, high = (100 * value for value in interval)
        ax.vlines(x, low, high, color=color, linewidth=0.9, zorder=4)
        ax.hlines([low, high], x - 0.045, x + 0.045, color=color, linewidth=0.9, zorder=4)


def plot_bars(runs: dict, output: Path) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 3, figsize=(12.1, 4.0), sharey=True)
    any_intervals = False
    for index, panel in enumerate(PANELS):
        ax = axes[index]
        ax.set_title(_panel_title(panel, runs, index), loc='left', pad=12)
        metrics = [('macro_f1', 'Two-class macro F1', '', -0.18)]
        if panel['region']:
            metrics.append(('union_box_recall_iou_0.5', 'Union-box R@IoU 0.5', '//', 0.18))
        observed = False
        for x, (name, label, color) in enumerate(panel['runs']):
            run = runs.get(name)
            for metric, _, hatch, offset in metrics:
                shift = offset if panel['region'] else 0
                value = _rate(run['overall'].get(metric), name) if run else None
                if value is None:
                    ax.text(x + shift, 3, 'N/A', rotation=90, ha='center', va='bottom', color='#777777', fontsize=7)
                    continue
                observed = True
                ax.bar(x + shift, value * 100, width=0.32 if panel['region'] else 0.5,
                       color=color, alpha=0.9 if not hatch else 0.42, hatch=hatch,
                       edgecolor=color, linewidth=0.6)
                interval = _interval(run, metric)
                any_intervals |= interval is not None
                _draw_interval(ax, x + shift, interval)
                # Labels show observed estimates; interval endpoints stay in source JSON.
                top = max(value * 100, interval[1] * 100 if interval else 0)
                ax.text(x + shift, min(top + 2, 104), f'{value * 100:.1f}',
                        ha='center', va='bottom', fontsize=7)
        ax.set_xticks(range(len(panel['runs'])), [label for _, label, _ in panel['runs']])
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(axis='y')
        ax.set_axisbelow(True)
        if not observed:
            _empty(ax)
    axes[0].set_ylabel('Score (%)')
    fig.suptitle('Held-out inspection performance | single-seed pilot', y=1.02, fontsize=12)
    fig.legend(handles=[Patch(facecolor='#707070', label='Two-class macro F1'),
                        Patch(facecolor='#c0c0c0', edgecolor='#707070', hatch='//', label='Union-box R@IoU 0.5')],
               loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
    caption = 'Macro F1 averages normal and anomalous F1. Box recall uses all anomalous region-task images.'
    if any_intervals:
        caption += '\nWhiskers: recorded 95% source-bootstrap intervals; these do not measure seed variation.'
    fig.text(0.5, -0.01, caption, ha='center', fontsize=8, color='#555555')
    fig.subplots_adjust(top=0.75, bottom=0.18, wspace=0.25)
    return _save(fig, output, 'evaluation_bars')


def plot_latency(runs: dict, output: Path) -> list[str]:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.1, 3.7), sharey=True)
    markers = ('o', 's', '^')
    for index, panel in enumerate(PANELS):
        ax = axes[index]
        ax.set_title(_panel_title(panel, runs, index), loc='left', pad=12)
        observed = False
        for position, (name, label, color) in enumerate(panel['runs']):
            run = runs.get(name)
            if not run:
                continue
            f1 = _rate(run['overall'].get('macro_f1'), name)
            latency = _number(run['overall'].get('mean_latency_seconds'))
            if f1 is None or latency is None or latency < 0:
                continue
            observed = True
            ax.scatter(latency, f1 * 100, color=color, marker=markers[position],
                       s=56, edgecolor='white', linewidth=0.7, zorder=5, label=label)
            interval = _interval(run, 'macro_f1')
            if interval is not None:
                ax.vlines(latency, interval[0] * 100, interval[1] * 100, color=color, alpha=0.7, linewidth=1)
        if observed:
            ax.legend(frameon=False, loc='lower right', fontsize=7)
            ax.margins(x=0.2)
        else:
            _empty(ax, 'No completed latency measurements')
        ax.set_xlabel('Mean generation latency (s / image)')
        ax.set_xlim(left=0)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid()
    axes[0].set_ylabel('Two-class macro F1 (%)')
    fig.suptitle('Observed quality and inference cost | single-seed pilot', y=1.02, fontsize=12)
    fig.text(0.5, -0.01, 'Latency excludes preprocessing and initial warm-up. Panels use distinct evaluation tasks.',
             ha='center', fontsize=8, color='#555555')
    fig.subplots_adjust(top=0.76, bottom=0.19, wspace=0.25)
    return _save(fig, output, 'latency_macro_f1')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Project root containing runs/ and evaluations/')
    parser.add_argument('--output', type=Path, default=Path('reports/figures'),
                        help='Absolute path, or a path relative to --root')
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f'Project root not found: {root}')
    output = args.output.expanduser()
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    warnings: list[str] = []
    training = load_training(root, warnings)
    evaluations = load_evaluations(root, warnings)
    if not training and not evaluations:
        parser.error('No observed training events or completed evaluation metrics found; no figures generated')
    _style()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = (plot_training(training, output) + plot_bars(evaluations, output) +
                 plot_latency(evaluations, output))
    provenance = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'root': str(root),
        'scope': 'Single-seed pilot. Raw logged CE and completed observed evaluations; no imputed results.',
        'figure_files': artifacts, 'training': training, 'evaluations': evaluations,
        'warnings': warnings,
        'definitions': {
            'macro_f1': 'Mean of normal and anomalous F1; not macro over product categories.',
            'localization': 'End-to-end union-box recall at IoU >= 0.5 over all anomalous region-task images.',
            'confidence_intervals': 'Only recorded bootstrap intervals are drawn; no uncertainty across training seeds.',
            'training_curves': 'Raw token-weighted assistant cross-entropy per optimizer update; no smoothing.',
            'decision_region_ce': 'Different target formats; CE magnitudes do not establish cross-task superiority.',
            'external': 'MVTec mirror pilot subset, not the complete official benchmark.',
        },
    }
    (output / 'figure_data.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'figures': artifacts, 'provenance': str(output / 'figure_data.json'),
                      'warnings': warnings}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
