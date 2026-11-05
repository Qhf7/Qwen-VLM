"""Render deterministic post-hoc paired examples from completed evaluations.

python ops/plot_examples.py --root /root/autodl-tmp/industrial_vlm \
    --output reports/examples

Selection is diagnostic, not an estimator of overall performance. No crops,
model calls, inference-time labels, or edits to predictions are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Work when launched directly, without requiring an editable package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from industrial_vlm.metrics import enclosing_box, union_box_iou, validate_boxes

DATASETS = (
    ('visa', 'VisA held-out test', 'base_region', 'quality_region'),
    ('mvtec', 'MVTec external pilot', 'external_base_region', 'external_quality_region'),
)
GROUPS = (
    ('base_wrong_sft_right', 'Base wrong / SFT correct', False, True),
    ('base_right_sft_wrong', 'Base correct / SFT wrong', True, False),
    ('both_wrong', 'Both wrong', False, False),
    ('both_right', 'Both correct', True, True),
)
MATCHED_KEYS = ('model', 'data_sha256', 'subset_sha256', 'max_pixels', 'max_length',
                'max_new_tokens', 'generation', 'quantization', 'limit')
GT_COLOR, PRED_COLOR = '#009E73', '#D55E00'


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return payload


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_run(root: Path, name: str) -> dict | None:
    directory = root / 'evaluations' / name
    meta_path, metrics_path, predictions_path = (directory / f for f in ('meta.json', 'metrics.json', 'predictions.jsonl'))
    if not all(p.is_file() for p in (meta_path, metrics_path, predictions_path)):
        return None
    meta, metrics = _json(meta_path), _json(metrics_path)
    if meta.get('status') != 'complete' or not metrics.get('metadata', {}).get('complete', False):
        return None
    config = {**meta.get('config', {}), **metrics.get('metadata', {})}
    records = {}
    for line_number, line in enumerate(predictions_path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        source = row.get('source_id')
        if not isinstance(source, str) or not source:
            raise ValueError(f'Missing source_id: {predictions_path}:{line_number}')
        if source in records:
            raise ValueError(f'Multiple predictions for source {source}: {predictions_path}')
        if type(row.get('label')) is not bool:
            raise ValueError(f'Non-boolean ground truth for {source}')
        if not row.get('localization_eligible'):
            raise ValueError(f'This renderer requires region-task predictions: {source}')
        validate_boxes(row['gt_boxes'])
        parsed = row.get('parsed')
        if parsed is not None:
            if not isinstance(parsed, dict) or type(parsed.get('is_anomalous')) is not bool:
                raise ValueError(f'Invalid stored parsed prediction: {source}')
            validate_boxes(parsed.get('boxes'))
        records[source] = row
    expected = metrics.get('overall', {}).get('n')
    if expected != len(records) or (config.get('n_expected') is not None and config['n_expected'] != len(records)):
        raise ValueError(f'Completed prediction count mismatch: {directory}')
    return {'name': name, 'config': config, 'records': records,
            'sources': {key: {'path': str(path), 'sha256': _sha(path)} for key, path in
                        [('metadata', meta_path), ('metrics', metrics_path), ('predictions', predictions_path)]}}


def _correct(row: dict) -> bool:
    parsed = row.get('parsed')
    return parsed is not None and parsed['is_anomalous'] == row['label']


def pair_and_select(base: dict, sft: dict) -> tuple[list[dict], dict]:
    """Match full input inventories, then pick the first source in each class."""
    for key in MATCHED_KEYS:
        if key not in base['config'] or key not in sft['config']:
            raise ValueError(f'Missing matched-input metadata: {key}')
        if base['config'][key] != sft['config'][key]:
            raise ValueError(f'Base/SFT evaluation conditions differ: {key}')
    if set(base['records']) != set(sft['records']):
        raise ValueError('Base/SFT source inventories differ; an intersection is not a matched evaluation')
    candidates: dict[str, list[str]] = {key: [] for key, _, _, _ in GROUPS}
    for source in sorted(base['records']):
        a, b = base['records'][source], sft['records'][source]
        for key in ('id', 'image', 'prompt', 'category', 'label', 'gt_boxes', 'localization_eligible', 'input_tokens'):
            if key not in a or key not in b or a[key] != b[key]:
                raise ValueError(f'Base/SFT row inputs differ: {source}:{key}')
        correctness = (_correct(a), _correct(b))
        group = next(key for key, _, base_ok, sft_ok in GROUPS if correctness == (base_ok, sft_ok))
        candidates[group].append(source)
    selected = []
    for key, label, _, _ in GROUPS:
        if candidates[key]:
            source = candidates[key][0]
            selected.append({'group': key, 'group_label': label, 'source_id': source,
                             'base': base['records'][source], 'sft': sft['records'][source]})
    return selected, {'matched_sources': len(base['records']),
                      'candidate_counts': {key: len(values) for key, values in candidates.items()},
                      'missing_groups': [key for key, values in candidates.items() if not values]}


def _description(row: dict) -> dict:
    parsed = row.get('parsed')
    iou = union_box_iou(parsed.get('boxes', []), row['gt_boxes']) if parsed is not None and row['label'] else None
    return {'classification_correct': _correct(row), 'label': row['label'],
            'prediction': parsed['is_anomalous'] if parsed is not None else None,
            'union_box_iou': iou, 'format_error': row.get('parse_error'),
            'raw_text': row.get('raw_text'), 'gt_boxes': row['gt_boxes'],
            'predicted_boxes': parsed.get('boxes', []) if parsed is not None else None}


def _rectangle(ax: Any, box: list[float] | None, width: int, height: int,
               color: str, linestyle: str) -> None:
    if box is None:
        return
    from matplotlib.patches import Rectangle
    x1, y1, x2, y2 = box
    ax.add_patch(Rectangle((x1 * width / 1000, y1 * height / 1000),
                           (x2 - x1) * width / 1000, (y2 - y1) * height / 1000,
                           fill=False, edgecolor=color, linewidth=2.0,
                           linestyle=linestyle, zorder=4))


def render(selected: list[dict], title: str, matched_n: int, output: Path, stem: str) -> list[str]:
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    # Reserve the same physical header/footer space for one through four rows.
    # Fractional positions alone make the legend collide with titles in tall grids.
    figure_height = 3.4 * len(selected) + 1.0
    fig, axes = plt.subplots(len(selected), 2, figsize=(8.4, figure_height), squeeze=False)
    for index, selection in enumerate(selected):
        image_path = Path(selection['base']['image'])
        if not image_path.is_absolute():
            raise ValueError('Prepared absolute image paths are required for example rendering')
        with Image.open(image_path) as original:
            orientation = original.getexif().get(274, 1)
            if orientation not in (None, 1):
                raise ValueError(f'Nontrivial EXIF orientation cannot be silently changed: {image_path}')
            image = original.convert('RGB').copy()
        width, height = image.size
        for column, (name, row) in enumerate((('Base', selection['base']), ('Coverage SFT', selection['sft']))):
            ax = axes[index, column]
            ax.imshow(image, interpolation='nearest')
            _rectangle(ax, enclosing_box(row['gt_boxes']), width, height, GT_COLOR, '-')
            parsed = row.get('parsed')
            _rectangle(ax, enclosing_box(parsed.get('boxes', [])) if parsed is not None else None,
                       width, height, PRED_COLOR, '--')
            detail = _description(row)
            label = 'anomalous' if row['label'] else 'normal'
            prediction = ('anomalous' if detail['prediction'] else 'normal') if parsed is not None else 'invalid'
            correctness = 'correct' if detail['classification_correct'] else 'wrong'
            iou = f"{detail['union_box_iou']:.3f}" if detail['union_box_iou'] is not None else 'N/A'
            ax.set_title(f'{name} | Label: {label} | Pred: {prediction}\n'
                         f'Class: {correctness} | Union-box IoU: {iou}' +
                         (' | Format error' if parsed is None else ''), fontsize=8.5, loc='left')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"{selection['group_label']} | {row['category']}", fontsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle(f'{title} | single-seed post-hoc examples\n'
                 f'Paired evaluation n = {matched_n}; deterministic diagnostic selection',
                 fontsize=11, y=1 - 0.08 / figure_height)
    fig.legend(handles=[Line2D([0], [0], color=GT_COLOR, linewidth=2, label='Ground-truth union box'),
                        Line2D([0], [0], color=PRED_COLOR, linewidth=2, linestyle='--', label='Predicted union box')],
               loc='upper center', bbox_to_anchor=(0.5, 1 - 0.53 / figure_height),
               ncol=2, frameon=False, fontsize=8)
    fig.text(0.5, 0.08 / figure_height, 'Examples do not estimate aggregate performance. Identical full images in both columns.\n'
             'Ground truth is overlaid after inference; no inference-time crop or label input.',
             ha='center', fontsize=7.5, color='#555555')
    fig.subplots_adjust(top=1 - 1.30 / figure_height, bottom=0.60 / figure_height,
                        left=0.035, right=0.985, hspace=0.36, wspace=0.08)
    files = []
    for suffix in ('png', 'pdf'):
        path = output / f'{stem}.{suffix}'
        fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
        files.append(str(path))
    plt.close(fig)
    return files


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('reports/examples'))
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    output = args.output.expanduser()
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    output.mkdir(parents=True, exist_ok=True)
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'root': str(root),
              'selection_rule': 'Within each of four classification-correctness groups, sort by source_id and select first.',
              'correctness': 'Stored parsed prediction is non-null and its boolean equals the ground-truth label; localization is separate.',
              'scope': 'Post-hoc diagnostic visualization only; not a representative sample or aggregate performance estimate. '
                       'Ground-truth boxes are overlaid after inference; no inference-time cropping or label input.',
              'iou': 'Local industrial_vlm.metrics.union_box_iou; enclosing predicted rectangle vs enclosing GT rectangle. '
                     'Undefined for normal GT or invalid format; empty anomaly prediction gives zero.',
              'datasets': {}, 'warnings': []}
    for key, title, base_name, sft_name in DATASETS:
        base, sft = _load_run(root, base_name), _load_run(root, sft_name)
        if base is None or sft is None:
            report['warnings'].append(f'Skipped {key}: both evaluations must be complete with metadata, metrics and predictions.')
            continue
        selected, audit = pair_and_select(base, sft)
        files = render(selected, title, audit['matched_sources'], output, f'{key}_paired_examples') if selected else []
        report['datasets'][key] = {
            **audit, 'base_run': base_name, 'sft_run': sft_name,
            'inputs_matched_on': list(MATCHED_KEYS), 'files': files,
            'evaluation_sources': {'base': base['sources'], 'sft': sft['sources']},
            'examples': [{'group': item['group'], 'source_id': item['source_id'],
                          'image': item['base']['image'], 'category': item['base']['category'],
                          'image_sha256': _sha(Path(item['base']['image'])),
                          'base': _description(item['base']), 'sft': _description(item['sft'])}
                         for item in selected],
        }
    destination = output / 'example_selection.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'selection': str(destination),
                      'figures': [f for dataset in report['datasets'].values() for f in dataset['files']],
                      'warnings': report['warnings']}, indent=2))


if __name__ == '__main__':
    main()
