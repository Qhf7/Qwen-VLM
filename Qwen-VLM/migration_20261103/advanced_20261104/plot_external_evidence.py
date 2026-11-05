"""Static figures from a complete, recomputed 21-condition external summary.

Partial inputs are rejected before any output directory is created. Reads only
one saved summary JSON; never reads predictions, RGB, masks, model weights or a
server. This independent reporting script does not alter the frozen evaluators.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)
MODELS = ('A', 'C', 'D', 'Cpair', 'R', 'Cstructured', 'S')
BASIC = MODELS[:5]
STRUCTURED = MODELS[5:]
N, NORMAL, ANOMALY, CATEGORIES = 1575, 392, 1183, 15
MANIFEST = '4cf2709614c8fd6aabe1268ef3c9f8687a07a0849d3d0f0341d1756570f9497a'
SCHEMA = 'visa-yesno-bbox1000-grid3-v1'
AUROC = 'overall.scores.image_auroc'
F1 = 'overall.generated.macro_f1'
RECALL = 'overall.generated.recall'
FPR = 'overall.generated.fpr'
MACRO = 'category_macro.scores.image_auroc'
IOU = 'localization.anomaly_bbox_iou_mean_all'
BASE_KEYS = (AUROC, F1, RECALL, FPR, MACRO)
COLORS = {42: '#0072B2', 43: '#D55E00', 44: '#009E73'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def metric(row, key):
    value = row['metrics']
    for component in key.split('.'):
        value = value[component]
    require(finite(value) and 0 <= value <= 1, 'Invalid metric: ' + key)
    return value


def close(left, right, context):
    require(finite(left) and finite(right) and math.isclose(left, right, abs_tol=1e-12, rel_tol=1e-12),
            'Summary/row mismatch: ' + context)


def describe(values):
    values = list(values)
    return dict(n=len(values), mean=statistics.mean(values), sample_sd=statistics.stdev(values))


def validate_stats(saved, values, context):
    computed = describe(values)
    require(saved.get('n') == len(values), 'Wrong aggregate seed count: ' + context)
    for key in ('mean', 'sample_sd'):
        close(saved.get(key), computed[key], context + '.' + key)


def validate_complete(summary):
    """Check all model/seed, metric, pairing and denominator contracts."""
    require(summary.get('state') == 'complete', 'Refusing partial external summary: state must be complete')
    require(summary.get('expected_runs') == summary.get('completed_recomputed_runs') == 21,
            'External figures require exactly 21/21 recomputed runs')
    require(summary.get('n') == N and summary.get('manifest_sha256') == MANIFEST,
            'External sample count or frozen manifest differs')
    require(summary.get('requested_models') == list(MODELS) and summary.get('requested_seeds') == list(SEEDS),
            'Expected all seven frozen conditions and seed42/43/44')
    require(summary.get('unavailable') == [] and summary.get('no_model_or_image_or_mask_queries') is True,
            'Summary has unavailable conditions or lacks saved-result-only provenance')
    require(valid_sha(summary.get('summarizer_sha256')), 'Missing summarizer SHA256')
    saved_rows = summary.get('rows', [])
    require(len(saved_rows) == 21, 'External summary must contain 21 rows')
    rows = {}
    for row in saved_rows:
        key = (row.get('model'), row.get('seed'))
        require(key not in rows and key[0] in MODELS and key[1] in SEEDS, 'Duplicate or unknown condition/seed')
        rows[key] = row
        require(row.get('state') == 'complete_and_recomputed', 'Run is not complete and recomputed: ' + str(key))
        require((row.get('n'), row.get('n_normal'), row.get('n_anomaly'), row.get('category_count'))
                == (N, NORMAL, ANOMALY, CATEGORIES), 'Run denominator differs: ' + str(key))
        require(row.get('manifest_sha256') == MANIFEST, 'Run manifest differs: ' + str(key))
        for hash_key in ('source_order_sha256', 'predictions_sha256', 'summary_sha256',
                         'checkpoint_sha256', 'external_cache_sha256', 'evaluator_sha256'):
            require(valid_sha(row.get(hash_key)), 'Missing source hash: ' + str(key) + '.' + hash_key)
        require(type(row.get('checkpoint_step')) is int and row['checkpoint_step'] >= 0,
                'Missing selected checkpoint step: ' + str(key))
        structured = row['model'] in STRUCTURED
        require(row.get('maximum_generated_tokens') == (80 if structured else 32), 'Generation budget differs')
        require(row.get('schema') == (SCHEMA if structured else 'historical_YesNo'), 'Schema differs')
        metrics = row['metrics']
        for channel in ('scores', 'generated'):
            counts = metrics['overall'][channel]
            require((counts.get('n'), counts.get('n_normal'), counts.get('n_anomaly')) == (N, NORMAL, ANOMALY),
                    'Metric denominator differs: ' + str(key))
        per_category = metrics.get('per_category', {})
        require(len(per_category) == CATEGORIES
                and metrics.get('category_macro_counts', {}).get('scores', {}).get('image_auroc') == CATEGORIES,
                'All15 categories must contribute to macro AUROC')
        close(statistics.mean(c['scores']['image_auroc'] for c in per_category.values()), metric(row, MACRO),
              str(key) + '.equal_category_macro_AUROC')
        require(sum(c['scores']['n_normal'] for c in per_category.values()) == NORMAL
                and sum(c['scores']['n_anomaly'] for c in per_category.values()) == ANOMALY,
                'Per-category label counts differ: ' + str(key))
        keys = BASE_KEYS + ((IOU,) if structured else ())
        for name in keys:
            close(row.get('flat_metrics', {}).get(name), metric(row, name), str(key) + '.' + name)
        if structured:
            local = metrics.get('localization', {})
            require((local.get('n'), local.get('n_normal'), local.get('n_anomaly')) == (N, NORMAL, ANOMALY),
                    'Localization must use all1183 anomalies')
            coverage = local.get('n_valid_positive_on_anomaly')
            require(type(coverage) is int and 0 <= coverage <= ANOMALY, 'Missing localization coverage count')
            require(finite(local.get('format_valid_rate')) and 0 <= local['format_valid_rate'] <= 1,
                    'Missing strict schema validity')
    require(set(rows) == {(model, seed) for model in MODELS for seed in SEEDS}, 'Incomplete seven-by-three matrix')
    require(len({r['source_order_sha256'] for r in rows.values()}) == 1, 'Run source order differs')
    for model in MODELS:
        aggregate = summary.get('aggregate', {}).get(model, {})
        require(aggregate.get('state') == 'complete_3seed' and aggregate.get('seeds') == list(SEEDS)
                and aggregate.get('n_seeds') == 3, 'Incomplete aggregate: ' + model)
        for name in BASE_KEYS + ((IOU,) if model in STRUCTURED else ()):
            validate_stats(aggregate.get('metrics', {}).get(name, {}),
                           [metric(rows[model, seed], name) for seed in SEEDS], model + '.' + name)
    for contrast, left, right in (('D_minus_C', 'C', 'D'), ('R_minus_Cpair', 'Cpair', 'R'),
                                 ('S_minus_Cstructured', 'Cstructured', 'S')):
        paired = summary.get('paired', {}).get(contrast, {})
        pairs = paired.get('pairs', [])
        require(paired.get('state') == 'complete_3seed' and len(pairs) == 3
                and sorted(p.get('seed') for p in pairs) == list(SEEDS), 'Missing same-seed pairs: ' + contrast)
        for name in BASE_KEYS + ((IOU,) if right == 'S' else ()):
            values = []
            for pair in pairs:
                seed = pair['seed']
                value = metric(rows[right, seed], name) - metric(rows[left, seed], name)
                close(pair.get('delta', {}).get(name), value, contrast + '.' + name)
                values.append(value)
            validate_stats(paired.get('aggregate', {}).get(name, {}), values, contrast + '.' + name)
    return rows


def annotate(ax, label):
    ax.text(-.10, 1.04, label, transform=ax.transAxes, fontsize=11, weight='bold')


def seed_legend(mean=True):
    handles = [Line2D([], [], color=COLORS[s], marker='o', label='Seed%d' % s) for s in SEEDS]
    if mean:
        handles.append(Line2D([], [], color='#222222', marker='D', lw=2, label='Seed mean'))
    return handles


def paired_panel(ax, rows, left, right, key, ylabel, title):
    values = {str(s): {m: metric(rows[m, s], key) for m in (left, right)} for s in SEEDS}
    for seed in SEEDS:
        ax.plot((0, 1), [100 * values[str(seed)][m] for m in (left, right)], color=COLORS[seed],
                lw=1.3, marker='o', markersize=5)
    ax.plot((0, 1), [100 * statistics.mean(values[str(s)][m] for s in SEEDS) for m in (left, right)],
            color='#222222', lw=2.0, marker='D', markersize=5, zorder=4)
    deltas = [(values[str(s)][right] - values[str(s)][left]) * 100 for s in SEEDS]
    ax.set_xticks((0, 1), (left, right), fontsize=8)
    ax.set_xlim(-.22, 1.22)
    ax.set_ylabel(ylabel)
    ax.set_title(title + '\nMean paired change: %+.2f pp' % statistics.mean(deltas), fontsize=10)
    ax.grid(axis='y', alpha=.22)
    return dict(metric=key, seeds=list(SEEDS), pair_values=values,
                paired_change_percentage_points=describe(deltas),
                denominator=dict(images=N, normal=NORMAL, anomaly=ANOMALY,
                                 all_anomalies=ANOMALY if key == IOU else None,
                                 invalid_no_missing_boxes_as_zero=key == IOU))


def tradeoff_panel(ax, rows, left, right):
    values = {}
    for seed in SEEDS:
        xs = [100 * metric(rows[m, seed], FPR) for m in (left, right)]
        ys = [100 * metric(rows[m, seed], RECALL) for m in (left, right)]
        ax.plot(xs, ys, color=COLORS[seed], lw=1.1, alpha=.8)
        ax.scatter(xs[0], ys[0], marker='o', s=43, facecolor='white', edgecolor=COLORS[seed], zorder=3)
        ax.scatter(xs[1], ys[1], marker='^', s=43, color=COLORS[seed], zorder=3)
        ax.annotate('', xy=(xs[1], ys[1]), xytext=(xs[0], ys[0]),
                    arrowprops=dict(arrowstyle='->', color=COLORS[seed], lw=.9))
        values[str(seed)] = {m: dict(fpr=metric(rows[m, seed], FPR), recall=metric(rows[m, seed], RECALL))
                             for m in (left, right)}
    ax.set_xlabel('Generated FPR (%) | all 392 normal images')
    ax.set_ylabel('Generated recall (%) | all 1183 anomalies')
    ax.set_title('%s (circle) to %s (triangle)' % (left, right), fontsize=10)
    ax.grid(alpha=.22)
    ax.legend(handles=[Line2D([], [], marker='o', ls='', markerfacecolor='white', markeredgecolor='#222222', label=left),
                       Line2D([], [], marker='^', ls='', color='#222222', label=right)], frameon=False, fontsize=8)
    return dict(metrics=[FPR, RECALL], seeds=list(SEEDS), pair_values=values,
                fpr_change_percentage_points=describe(100 * (metric(rows[right, s], FPR) - metric(rows[left, s], FPR)) for s in SEEDS),
                recall_change_percentage_points=describe(100 * (metric(rows[right, s], RECALL) - metric(rows[left, s], RECALL)) for s in SEEDS),
                denominator=dict(false_positive_normals=NORMAL, recall_anomalies=ANOMALY, invalid_answers_retained=True),
                interpretation='Generated-answer operating points, not a score-threshold sweep or ROC curve')


def finish(fig, path, fixture, title, detail):
    fig.suptitle(title + '\nPublic MVTec official-test remainder | 1575 images: 392 normal, 1183 anomaly; 15 categories', fontsize=11)
    fig.supxlabel(detail + '\nVisA-selected checkpoints; three training seeds on the same images. '
                  'Native Qwen external: not evaluated; foundation pretraining exposure unknown.', fontsize=8)
    if fixture:
        fig.text(.5, .5, 'SYNTHETIC FIXTURE\nNOT EXPERIMENTAL RESULTS', ha='center', va='center',
                 fontsize=27, color='#b00020', alpha=.25, rotation=18, weight='bold', zorder=100)
    png, pdf = path.with_suffix('.png'), path.with_suffix('.pdf')
    fig.savefig(png, dpi=220, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)
    return dict(png=str(png.resolve()), pdf=str(pdf.resolve()), png_sha256=sha256(png), pdf_sha256=sha256(pdf))


def basic_figure(rows, output, fixture):
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8), constrained_layout=True)
    panels = {
        'C_to_D_auroc': paired_panel(axes[0, 0], rows, 'C', 'D', AUROC, 'Global image AUROC (%)', 'C to D | joint bridge + LoRA'),
        'Cpair_to_R_auroc': paired_panel(axes[0, 1], rows, 'Cpair', 'R', AUROC, 'Global image AUROC (%)', 'Cpair to R | matched CE + ranking'),
        'C_to_D_generated_tradeoff': tradeoff_panel(axes[1, 0], rows, 'C', 'D'),
        'Cpair_to_R_generated_tradeoff': tradeoff_panel(axes[1, 1], rows, 'Cpair', 'R')}
    for ax, label in zip(axes.flat, 'ABCD'):
        annotate(ax, label)
    axes[0, 0].legend(handles=seed_legend(), frameon=False, fontsize=8, loc='best')
    artifacts = finish(fig, output / 'external_basic', fixture, 'External basic conditions | complete 21/21 | matched seed comparisons',
                       'Basic generation budget: 32 tokens. AUROC uses Yes-minus-No logits; recall/FPR use actual generated answers.')
    return dict(artifacts=artifacts, panels=panels)


def scale_panel(ax, rows, models, title):
    values = {}
    for index, model in enumerate(models):
        values[model] = {}
        for seed_index, seed in enumerate(SEEDS):
            offset = (seed_index - 1) * .055
            xs = [index - .16 + offset, index + .16 + offset]
            ys = [metric(rows[model, seed], key) * 100 for key in (AUROC, MACRO)]
            ax.plot(xs, ys, color=COLORS[seed], lw=1.0, alpha=.8)
            ax.scatter(xs[0], ys[0], marker='o', s=29, color=COLORS[seed], zorder=3)
            ax.scatter(xs[1], ys[1], marker='^', s=29, facecolor='white', edgecolor=COLORS[seed], zorder=3)
            values[model][str(seed)] = dict(global_auroc=ys[0] / 100, equal_category_macro_auroc=ys[1] / 100)
        ax.plot([index - .16, index + .16],
                [100 * statistics.mean(metric(rows[model, s], key) for s in SEEDS) for key in (AUROC, MACRO)],
                color='#222222', marker='D', lw=1.8, markersize=4, zorder=4)
    ax.set_xticks(range(len(models)), models, fontsize=8)
    ax.set_xlim(-.5, len(models) - .5)
    ax.set_ylabel('AUROC (%)')
    ax.set_title(title, fontsize=10)
    ax.grid(axis='y', alpha=.22)
    return dict(metrics=[AUROC, MACRO], values=values, seeds=list(SEEDS),
                denominator=dict(global_images=N, equal_weight_categories=CATEGORIES),
                interpretation='Global includes cross-category positive-negative comparisons; macro ranks within each category then equally averages15 categories')


def scale_figure(rows, output, fixture):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), gridspec_kw={'width_ratios': (1.8, 1)}, constrained_layout=True)
    panels = {'basic': scale_panel(axes[0], rows, BASIC, 'Basic | max 32 output tokens'),
              'structured': scale_panel(axes[1], rows, STRUCTURED, 'Structured | max 80 output tokens')}
    annotate(axes[0], 'A'); annotate(axes[1], 'B')
    axes[0].legend(handles=seed_legend(False) + [
        Line2D([], [], color='#333333', marker='o', ls='', label='Global (left)'),
        Line2D([], [], color='#333333', marker='^', markerfacecolor='white', ls='', label='Category macro (right)'),
        Line2D([], [], color='#222222', marker='D', label='Seed mean')], frameon=False, fontsize=7, ncol=2, loc='best')
    artifacts = finish(fig, output / 'external_auroc_scales', fixture, 'External AUROC definitions | global versus equal-category macro',
                       'Different aggregation questions. Basic and structured prompts/output budgets differ; cross-schema changes are compound.')
    return dict(artifacts=artifacts, panels=panels)


def structured_figure(rows, output, fixture):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5), constrained_layout=True)
    panels = {}
    for ax, name, ylabel, title in zip(axes, (AUROC, F1, IOU),
                                      ('Global image AUROC (%)', 'Generated binary macro F1 (%)', 'Mean union-box IoU (%)'),
                                      ('Anomaly score ranking', 'Generated anomaly decision', 'Localization | all 1183 anomalies')):
        panels[name] = paired_panel(ax, rows, 'Cstructured', 'S', name, ylabel, title)
    for ax, label in zip(axes, 'ABC'):
        annotate(ax, label)
    axes[0].legend(handles=seed_legend(), frameon=False, fontsize=8, loc='best')
    panels['localization_coverage'] = {str(seed): {model: dict(
        n_valid_positive_on_anomaly=rows[model, seed]['metrics']['localization']['n_valid_positive_on_anomaly'],
        format_valid_rate=rows[model, seed]['metrics']['localization']['format_valid_rate'])
        for model in STRUCTURED} for seed in SEEDS}
    artifacts = finish(fig, output / 'external_structured', fixture, 'External structured continuation | C weights + schema to S best',
                       'Same 80-token schema; S adds mask-derived supervision and training exposures. All-anomaly IoU: invalid/No/missing boxes=0.\n'
                       'Compressed teacher-map input; union boxes are not segmentation or causal explanations.')
    return dict(artifacts=artifacts, panels=panels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=HERE / 'external_summary.json')
    parser.add_argument('--out', type=Path, default=HERE / 'external_figures')
    args = parser.parse_args()
    path = args.summary.resolve()
    blob = path.read_bytes()
    summary = json.loads(blob)
    rows = validate_complete(summary)
    fixture = summary.get('synthetic_fixture') is True
    output = args.out.resolve()
    # No output is created before the complete21/21 validation above.
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    figures = dict(basic=basic_figure(rows, output, fixture), auroc_definitions=scale_figure(rows, output, fixture),
                   structured=structured_figure(rows, output, fixture))
    identities = {model: {str(seed): {key: rows[model, seed].get(key) for key in (
        'checkpoint_step', 'checkpoint_sha256', 'checkpoint_bytes_verified_locally',
        'summary_sha256', 'predictions_sha256', 'external_cache_sha256', 'source_order_sha256',
        'evaluator_sha256', 'maximum_generated_tokens', 'schema', 'generation_policy',
        'source_trainable_parameters', 'newly_trained_parameters_for_this_condition',
        'training_exposures_for_this_condition', 'inference_cost', 'source_training_cost')}
        for seed in SEEDS} for model in MODELS}
    provenance = dict(schema='external-evidence-figures-v1', state='complete21/21',
                      data_kind='synthetic_fixture_not_results' if fixture else 'complete_saved_external_results',
                      created_utc=datetime.now(timezone.utc).isoformat(),
                      source_summary=dict(path=str(path), sha256=hashlib.sha256(blob).hexdigest(),
                                          state=summary['state'], completed_recomputed_runs=21,
                                          summarizer_sha256=summary['summarizer_sha256']),
                      script_sha256=sha256(__file__), manifest_sha256=MANIFEST,
                      denominators=dict(images_per_run=N, normal=NORMAL, anomaly=ANOMALY,
                                        categories=CATEGORIES, training_seeds=list(SEEDS)),
                      model_identities=identities, figures=figures,
                      scope='Saved recomputed summary only; no model, prediction, RGB, mask, checkpoint or server access',
                      native_qwen_external_evaluated=False,
                      interpretation='Public MVTec official-test remainder; foundation pretraining exposure unknown. '
                      'Checkpoints selected on reused VisA development validation. Same-seed changes, no significance claim. '
                      'Basic/structured instruction and output budgets differ. S adds spatial supervision and exposures; '
                      'compressed teacher-map inputs remain. No reasoning, causal explanation or native-vision transfer claim.')
    (output / 'external_figure_provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    notes = ['# External figure notes', '',
             '**SYNTHETIC FIXTURE; NOT EXPERIMENTAL RESULTS.**' if fixture else '**Complete21/21 saved-result report.**', '',
             'All runs: same public MVTec official-test remainder1575 images (392normal,1183anomaly),15categories, seeds42/43/44.',
             'external_basic: C→D and Cpair→R pair matching seeds. AUROC is from Yes-minus-No logits; generated FPR/recall are operating points, not ROC curves.',
             'external_auroc_scales: global AUROC versus equal-category macro AUROC. Global includes cross-category comparisons; macro ranks within category and gives15 categories equal weight.',
             'external_structured: Cstructured→S AUROC, generated binary macro F1 and union-box IoU on all1183 anomalies. Invalid/No/missing boxes count zero. Coverage and strict format validity are in provenance.',
             'Cstructured and S use the same schema prompt and80-token budget. S adds training-mask-derived targets, answer tokens and continuation exposures; teacher-map visual inputs remain.',
             'Native Qwen external was not evaluated. Foundation pretraining exposure is unknown. VisA development validation selected checkpoints; three training repeats are not independent test sets or significance evidence.',
             'The plotting script reads the summary only. The upstream summarizer performs raw-prediction recomputation; plotting does not repeat model or raw-prediction audits.',
             'Source summary SHA256: ' + hashlib.sha256(blob).hexdigest(),
             'Frozen manifest SHA256: ' + MANIFEST]
    (output / 'EXTERNAL_FIGURE_NOTES.md').write_text('\n\n'.join(notes) + '\n')
    print(json.dumps(dict(state='complete', data_kind=provenance['data_kind'], output=str(output),
                          figures=list(figures)), ensure_ascii=False))


if __name__ == '__main__':
    main()
