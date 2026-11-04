"""Summarize only complete, posterior-audited native ROI tool experiments.

Incomplete runs contribute status only, never partial efficacy numbers.
All metrics, request/crop replay, paired deltas, call/error counts, geometry
and costs are recomputed from records tied to a completed audit by SHA.
No model, GPU, SSH, cache, source RGB or GT mask is loaded by this script.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'code'))
from tool_workflow import (CONDITIONS, atomic_json, sha256_file, parse_request,
                           request_for_condition, crop_bounds)
from migration_metrics import evaluate_predictions, parse_generated

SCHEMA = 'native-crop-tool-recomputed-summary-v1'
CONTROLS = ('duplicate_original', 'center_crop', 'random_crop')
METRIC_KEYS = ('image_auroc', 'image_ap', 'accuracy', 'macro_f1', 'recall', 'fpr', 'invalid_rate')
COST_KEYS = ('final_seconds', 'deployment_seconds', 'standalone_seconds_including_planner',
             'standalone_seconds_without_planner', 'score_seconds', 'generate_seconds',
             'image_load_hash_seconds', 'peak_gpu_allocated_bytes_final',
             'peak_gpu_reserved_bytes_final', 'peak_gpu_allocated_bytes_end_to_end',
             'image_tokens_total', 'prompt_tokens_total')


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def mean(values):
    return sum(values) / len(values) if values else None


def describe(values):
    require(values and all(finite(x) for x in values), 'Expected nonempty finite numeric observations')
    ordered = sorted(values)
    return {'n': len(values), 'mean': mean(values), 'median': ordered[(len(values)-1)//2]
            if len(values) % 2 else (ordered[len(values)//2-1]+ordered[len(values)//2])/2,
            'min': ordered[0], 'max': ordered[-1]}


def metric_delta(left, right):
    return {channel: {key: (left[channel][key] - right[channel][key])
                      for key in METRIC_KEYS
                      if key in left[channel] and key in right[channel]
                      and left[channel][key] is not None and right[channel][key] is not None}
            for channel in ('scores', 'generated')}


def completed_status(directory, expected_n):
    directory = Path(directory)
    run_path = directory / 'run.json'
    if not run_path.exists():
        return {'state': 'pending', 'expected_n': expected_n, 'path': str(directory)}, None
    run = read(run_path)
    status = {'state': run.get('status', 'unknown'), 'expected_n': expected_n,
              'path': str(directory), 'reported_n': run.get('n')}
    if run.get('status') != 'complete':
        return status, None
    audit_path, predictions_path = directory / 'audit.json', directory / 'predictions.jsonl'
    if not audit_path.exists():
        status['state'] = 'waiting_for_posterior_audit'
        return status, None
    audit = read(audit_path)
    if audit.get('passed') is not True:
        status['state'] = 'posterior_audit_not_passed'
        return status, None
    require(run.get('n') == audit.get('n') == expected_n, 'Completed run/audit denominator differs')
    require(tuple(run.get('conditions', ())) == tuple(audit.get('conditions', ())) == CONDITIONS,
            'Five paired conditions changed')
    require(run.get('test_queries') == run.get('external_queries') == 0,
            'This summary only permits the validation-only protocol')
    require(audit.get('gt_reads_before_saved_run_and_rgb_gates') == 0, 'Post-hoc GT gate changed')
    require(audit.get('sources', {}).get('run_json') == sha256_file(run_path)
            and audit.get('sources', {}).get('predictions_jsonl') == sha256_file(predictions_path),
            'Completed inference changed after posterior audit')
    raw = predictions_path.read_text().splitlines()
    require(raw and all(line.strip() for line in raw), 'Blank prediction rows are not permitted')
    rows = [json.loads(line) for line in raw]
    require(len(rows) == expected_n * len(CONDITIONS), 'Prediction denominator incomplete')
    selected = run['selected_paths']
    require(len(selected) == len(set(selected)) == expected_n, 'Selected image identity/order invalid')
    by_condition = {c: [r for r in rows if r.get('condition') == c] for c in CONDITIONS}
    require(set(run.get('results', {})) == set(CONDITIONS), 'Missing completed result condition')
    for condition, values in by_condition.items():
        require([r['path'] for r in values] == selected, 'Prediction path order differs: ' + condition)
        for row in values:
            require(row.get('split') == 'val' and type(row.get('label')) is int
                    and row['label'] in (0, 1), 'Unexpected query metadata')
            plan, error = parse_request(row['planner']['raw_text'])
            require(plan == row['planner']['request'] and error == row['planner']['error'],
                    'Planner parse changed')
            request, rejected = request_for_condition(condition, plan, error, row['path'], run['config']['seed'])
            box = crop_bounds(request['center']) if request['action'] == 'crop' else None
            require(row['request_executed'] == request and row['request_rejected'] == bool(rejected)
                    and row['bbox_xyxy_relative'] == box and row['tool_calls'] == int(box is not None),
                    'Request execution/crop replay mismatch')
            n_images = 1 if condition == 'no_tool' else 2
            require(row['n_images'] == n_images and row['image_tokens_total'] == n_images * 196
                    and row['image_grid_thw'] == [[1, 28, 28]] * n_images, 'Visual token/grid budget changed')
            require(all(finite(row.get(k)) for k in ('yes_logit', 'no_logit', 'score_yes_minus_no'))
                    and row['score_yes_minus_no'] == row['yes_logit'] - row['no_logit'], 'Nonfinite/incorrect score')
            require(type(row['generated_text']) is str, 'Generation must remain a raw string')
            require(row['planner_invalid'] == bool(error), 'Planner invalid flag differs')
    base = by_condition['model_tool']
    original_items = [{key: row[key] for key in ('path', 'category', 'label', 'split')} for row in base]
    require(hashlib.sha256(json.dumps(original_items, sort_keys=True).encode()).hexdigest()
            == run['selected_manifest_sha256'], 'Frozen selected sample metadata hash mismatch')
    require([row['cache_index'] for row in base] == run['selected_cache_indices'], 'Cache indices differ from selection')
    for position, row in enumerate(base):
        for condition in CONDITIONS:
            require(all(by_condition[condition][position].get(k) == row.get(k)
                        for k in ('path', 'label', 'category', 'split', 'cache_index', 'image_sha256', 'planner')),
                    'Paired sample/planner identities differ')
    geometry = audit.get('posthoc_gt_geometry', {})
    require(set(geometry.get('results', {})) == set(CONDITIONS), 'Complete posterior geometry audit required')
    status.update(state='complete', sources={'run_json': sha256_file(run_path),
        'predictions_jsonl': sha256_file(predictions_path), 'posterior_audit_json': sha256_file(audit_path)})
    return status, recompute(run, audit, by_condition)


def recompute(run, audit, by_condition):
    n = run['n']
    arms = {}
    anomaly_n = sum(r['label'] for r in by_condition['model_tool'])
    normal_n = n - anomaly_n
    geometry_rows = audit['posthoc_gt_geometry']['records']
    require(len(geometry_rows) == anomaly_n * len(CONDITIONS), 'Geometry all-anomaly denominator incomplete')
    for condition, rows in by_condition.items():
        metrics = evaluate_predictions(rows, [r['score_yes_minus_no'] for r in rows],
                                       [r['generated_text'] for r in rows])
        require(metrics == run['results'][condition]['metrics']
                and metrics == audit['classification'][condition], 'Classification recomputation differs')
        planner_errors = Counter(r['planner']['error'] or 'valid' for r in rows)
        planner_actions = Counter('invalid' if r['planner']['error'] else r['planner']['request']['action'] for r in rows)
        executed = [r['tool_calls'] for r in rows]
        per_label = {}
        for label, label_name in ((0, 'normal'), (1, 'anomaly')):
            subset = [r for r in rows if r['label'] == label]
            per_label[label_name] = {'n': len(subset), 'actual_calls': sum(r['tool_calls'] for r in subset),
                                    'actual_call_rate': mean([r['tool_calls'] for r in subset]),
                                    'invalid_planner_rate': mean([int(r['planner_invalid']) for r in subset])}
        costs = {key: describe([r[key] for r in rows]) for key in COST_KEYS}
        costs['planner_seconds'] = describe([r['planner']['seconds'] for r in rows])
        costs['planner_generate_seconds'] = describe([r['planner']['generate_seconds'] for r in rows])
        costs['planner_image_tokens'] = describe([r['planner']['image_tokens_total'] for r in rows])
        crops = [r['bbox_xyxy_relative'] for r in rows if r['tool_calls']]
        g_rows = [r for r in geometry_rows if r['condition'] == condition]
        require(len(g_rows) == anomaly_n and {r['path'] for r in g_rows}
                == {r['path'] for r in rows if r['label']}, 'Geometry paths differ')
        indexed = {r['path']: r for r in rows}
        for r in g_rows:
            require(r['tool_calls'] == indexed[r['path']]['tool_calls']
                    and type(r['gt_pixels224']) is int and r['gt_pixels224'] > 0
                    and type(r['retained_gt_pixels224']) is int
                    and 0 <= r['retained_gt_pixels224'] <= r['gt_pixels224']
                    and r['defect_fraction_inside_crop'] == r['retained_gt_pixels224'] / r['gt_pixels224']
                    and r['crop_hits_any_gt'] == (r['retained_gt_pixels224'] > 0),
                    'Geometry pixel/count ratio mismatch')
            require(r['tool_calls'] or r['retained_gt_pixels224'] == 0, 'Uncalled crop has positive retained pixels')
        called = [r for r in g_rows if r['tool_calls']]
        geometry = {'n_anomaly': anomaly_n, 'n_anomaly_with_crop': len(called),
                    'mean_defect_fraction_all_anomalies': mean([r['defect_fraction_inside_crop'] for r in g_rows]),
                    'crop_hit_rate_all_anomalies': mean([int(r['crop_hits_any_gt']) for r in g_rows]),
                    'mean_defect_fraction_called_anomalies_only': mean([r['defect_fraction_inside_crop'] for r in called])}
        saved_geometry = audit['posthoc_gt_geometry']['results'][condition]
        require(set(geometry) == set(saved_geometry), 'Geometry aggregate schema differs')
        for key in ('n_anomaly', 'n_anomaly_with_crop'):
            require(type(saved_geometry[key]) is int and geometry[key] == saved_geometry[key],
                    'Geometry aggregate integer denominator differs: ' + key)
        for key in ('mean_defect_fraction_all_anomalies', 'crop_hit_rate_all_anomalies',
                    'mean_defect_fraction_called_anomalies_only'):
            actual, saved = geometry[key], saved_geometry[key]
            require((actual is None and saved is None)
                    or (finite(actual) and finite(saved) and math.isclose(actual, saved, rel_tol=0, abs_tol=1e-12)),
                    'Geometry aggregate floating value differs: ' + key)
        arms[condition] = {'metrics': metrics, 'n': n, 'actual_calls': sum(executed),
            'actual_call_rate': mean(executed), 'invalid_planner_rate': mean([int(r['planner_invalid']) for r in rows]),
            'rejected_request_rate': mean([int(r['request_rejected']) for r in rows]),
            'planner_error_counts': dict(sorted(planner_errors.items())),
            'planner_action_counts': dict(sorted(planner_actions.items())),
            'per_label_tool_behavior': per_label, 'unique_executed_bboxes': len({tuple(box) for box in crops}),
            'geometry': geometry, 'costs': costs}
    comparisons = {}
    model = by_condition['model_tool']
    for control in CONTROLS + ('no_tool',):
        comparison = metric_delta(arms['model_tool']['metrics']['overall'], arms[control]['metrics']['overall'])
        paired = Counter()
        for left, right in zip(model, by_condition[control]):
            l = parse_generated(left['generated_text'])
            r = parse_generated(right['generated_text'])
            left_correct, right_correct = l == left['label'], r == right['label']
            paired['model_only_correct'] += int(left_correct and not right_correct)
            paired['control_only_correct'] += int(right_correct and not left_correct)
            paired['both_correct'] += int(left_correct and right_correct)
            paired['both_wrong'] += int(not left_correct and not right_correct)
            paired['generated_decisions_differ'] += int(l != r)
        comparisons['model_tool_minus_' + control] = {'metric_delta': comparison,
            'paired_generated_outcomes': dict(paired),
            'mean_deployment_seconds_delta': arms['model_tool']['costs']['deployment_seconds']['mean']
                                            - arms[control]['costs']['deployment_seconds']['mean'],
            'all_anomaly_crop_hit_rate_delta': arms['model_tool']['geometry']['crop_hit_rate_all_anomalies']
                                              - arms[control]['geometry']['crop_hit_rate_all_anomalies']
                                              if anomaly_n else None}
    executed = arms['model_tool']['actual_calls']
    return {'n': n, 'n_normal': normal_n, 'n_anomaly': anomaly_n, 'arms': arms, 'paired_comparisons': comparisons,
            'model_selected_crop_observed': executed > 0,
            'autonomous_crop_evidence': ('At least one valid model request was executed as an RGB crop; '
                'this alone does not show image-dependent selection or benefit over controls.'
                if executed else 'No valid model crop was executed; the complete result does not demonstrate autonomous cropping.'),
            'interpretation_limits': ['Original RGB only at inference; no category/label/mask/map inputs.',
                'Final classifier is a fresh prompt with returned crop evidence; this is a bounded one-call workflow.',
                'Four two-image arms match visual tokens, not text tokens or wall time; no_tool has fewer visual tokens.',
                'Planner is reused across controls; deployment_seconds excludes unnecessary planner cost for fixed controls.',
                'All-answer or all-invalid planners remain legitimate complete negative results.',
                'GT geometry was read post-inference and does not prove detail visibility or language attention.',
                'Native vision experiment is distinct from compressed bridge C/D LoRA; validation is development data.']}


def report(summary):
    lines = ['# 原生Qwen ROI工具结果', '', '状态：' + summary['state'],
             '', '只纳入推理complete且后验审计passed、输入报告SHA匹配的结果。未完成阶段仅列状态，不报告部分效能。', '']
    for key, value in summary['stages'].items():
        lines.append('- ' + key + '：' + value['state'])
    result = summary.get('validation')
    if not result:
        lines.extend(['', '完整val380尚未通过所有门禁，当前没有正式工具效能结论。'])
        return '\n'.join(lines) + '\n'
    lines += ['', '下表为完整开发验证；AUROC/AP是首答案logit排序，F1/recall/FPR是实际生成。', '',
        '| 条件 | AUROC | AP | 生成F1 | 召回 | FPR | 实际调用率 | planner无效率 | 部署秒/图 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt = lambda x: 'NA' if x is None else format(x, '.6f')
    for condition in CONDITIONS:
        arm=result['arms'][condition];metric=arm['metrics']['overall']
        vals=[metric['scores']['image_auroc'],metric['scores']['image_ap'],metric['generated']['macro_f1'],
              metric['generated']['recall'],metric['generated']['fpr'],arm['actual_call_rate'],
              arm['invalid_planner_rate'],arm['costs']['deployment_seconds']['mean']]
        lines.append('| '+condition+' | '+' | '.join(fmt(x) for x in vals)+' |')
    lines += ['', '| model_tool减控制（百分点） | ΔAUROC | Δ生成F1 | Δ召回 | ΔFPR |',
              '|---|---:|---:|---:|---:|']
    for control in CONTROLS + ('no_tool',):
        delta=result['paired_comparisons']['model_tool_minus_'+control]['metric_delta']
        vals=[delta['scores'].get('image_auroc'),delta['generated'].get('macro_f1'),
              delta['generated'].get('recall'),delta['generated'].get('fpr')]
        lines.append('| '+control+' | '+' | '.join(fmt(None if x is None else x*100) for x in vals)+' |')
    model=result['arms']['model_tool']
    lines += ['', '模型真实裁剪次数：'+str(model['actual_calls'])+'/'+str(result['n'])+'；独特执行bbox：'
              +str(model['unique_executed_bboxes'])+'。', '',
              'planner action计数：'+json.dumps(model['planner_action_counts'],ensure_ascii=False)+'。',
              'planner error计数：'+json.dumps(model['planner_error_counts'],ensure_ascii=False)+'。', '',
              result['autonomous_crop_evidence'], '', '解释边界：']
    lines += ['- '+line for line in result['interpretation_limits']]
    lines += ['', '全部配对逐图正确/错误迁移、正常/异常调用率、几何覆盖与成本分布详见JSON。']
    return '\n'.join(lines)+'\n'


def build(root):
    stages={};details={}
    for stage, n in (('tool_preflight',8),('tool_val',380)):
        status, value=completed_status(Path(root)/stage,n)
        stages[stage]=status
        if value is not None: details[stage]=value
    return {'schema':SCHEMA, 'state':'complete' if all(s['state']=='complete' for s in stages.values()) else 'partial',
            'stages':stages, 'preflight':details.get('tool_preflight'), 'validation':details.get('tool_val'),
            'external_queries':0, 'model_loaded':False, 'gt_masks_opened_by_summarizer':False,
            'summarizer_sha256':sha256_file(__file__)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('runs/migration_20261103/advanced_20261104'))
    ap.add_argument('--out',type=Path)
    ap.add_argument('--markdown',type=Path)
    args=ap.parse_args()
    result=build(args.root)
    out=args.out or args.root/'tool_summary.json'
    markdown=args.markdown or args.root/'TOOL_RESULTS.md'
    atomic_json(result,out)
    markdown.parent.mkdir(parents=True,exist_ok=True)
    temp=markdown.with_suffix(markdown.suffix+'.tmp');temp.write_text(report(result));temp.replace(markdown)
    print(json.dumps({'state':result['state'],'out':str(out),'markdown':str(markdown),
                      'validation_n':None if result['validation'] is None else result['validation']['n']}))


if __name__=='__main__':
    main()
