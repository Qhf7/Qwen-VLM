#!/usr/bin/env python3
"""Independent stdlib-only audit of frozen existing scores, on approved server only."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import sys


ROOT = Path('/root/autodl-tmp/industrial_vlm')
OLD = ROOT / 'experiments/jev_followup_20261111'
ROUND = ROOT / 'experiments/jev_decision_round2_20261111'
OUT = ROUND / 'fixed_fpr_operating_points'
HOST = 'autodl-container-132444827f-9c594038'
PYTHON = '/root/autodl-tmp/conda-envs/qwen3vl2/bin/python'
EXPECTED = {
    'code/isolated_calibration.py': 'd8489c953ec4cb5e9d7804643d35f43d5ad0801211abc118bcc8e0003fec37d1',
    'stage12/run/calibrator.json': '0bac063df108bd44938c25a8bb2c07261d79bfe2350b6d1bdb8b8cf1c1bcd9ec',
    'stage12/run/predictions.jsonl': 'f81d0006dcd17d305f768f4065cd5ce4b7da5a991200955a5f7ff2c46af9116a',
    'stage12/run/contract.json': '45d593a9833ab2b1dd003afb954b6eca10cd85a574055f3d0e821ac7aa515ea1',
    'stage12_calibration_plan/calibration_labels.jsonl': 'aaaf4898122c685ed5a511af5b1d0d85d6eaa4d699a7822174eaa2a84a56fd1d',
    'stage12_calibration_plan/heldout_eval_labels.jsonl': '7cc2853e08c36af1491a29b60622d42cdc2a377d803f07b962dd5a81942761d3',
}
CHECKS = []
READS = []


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(name, condition, detail=None):
    CHECKS.append({'check': name, 'passed': bool(condition), 'detail': detail})
    if not condition:
        raise AssertionError(name + ': ' + repr(detail))


def read(path):
    data = path.read_bytes()
    READS.append({'path': str(path), 'sha256': digest(data)})
    return data


def parse_rows(data):
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def close(left, right):
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-14)


def equal(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(equal(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and close(actual, expected)
    return type(actual) is type(expected) and actual == expected


def ci95(successes, total):
    # Independently derived Wilson score roots, z fixed by protocol.
    z2 = 1.959963984540054 ** 2
    denominator = 2.0 * (total + z2)
    root = math.sqrt(z2 * (z2 + 4.0 * successes * (1.0 - successes / total)))
    numerator = 2.0 * successes + z2
    return [max(0.0, (numerator - root) / denominator), min(1.0, (numerator + root) / denominator)]


def summarize(ids, labels, scores, threshold):
    counts = Counter((labels[i], scores[i] > threshold) for i in ids)
    tp, fn, fp, tn = (counts[(1, True)], counts[(1, False)], counts[(0, True)], counts[(0, False)])
    n0, n1 = fp + tn, tp + fn
    ties0 = sum(labels[i] == 0 and scores[i] == threshold for i in ids)
    ties1 = sum(labels[i] == 1 and scores[i] == threshold for i in ids)
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, normal_n=n0, anomalous_n=n1,
                fpr=fp / n0, recall=tp / n1, fpr_wilson95=ci95(fp, n0), recall_wilson95=ci95(tp, n1),
                scores_equal_threshold_total=ties0 + ties1, normal_scores_equal_threshold=ties0,
                anomalous_scores_equal_threshold=ties1)


def order_audit(ids, raw, calibrated):
    raw_ties = cal_ties = mismatches = pairs = 0
    for j, i in enumerate(ids):
        for k in ids[j + 1:]:
            pairs += 1
            r = (raw[i] > raw[k]) - (raw[i] < raw[k])
            c = (calibrated[i] > calibrated[k]) - (calibrated[i] < calibrated[k])
            raw_ties += r == 0
            cal_ties += c == 0
            mismatches += r != c
    return dict(unordered_pairs=pairs, pairwise_order_or_tie_mismatches=mismatches,
                raw_tied_pairs=raw_ties, calibrated_tied_pairs=cal_ties,
                raw_unique_scores=len({raw[i] for i in ids}), calibrated_unique_scores=len({calibrated[i] for i in ids}),
                scores_clipped=sum(raw[i] < 1e-6 or raw[i] > 1 - 1e-6 for i in ids))


def main():
    check('approved_hostname', socket.gethostname() == HOST)
    check('approved_python', sys.executable == PYTHON)
    check('approved_workspace', Path.cwd() == ROOT)
    check('new_output_paths_only', all(not (OUT / f).exists() for f in ('independent_audit.json', 'independent_audit.md', 'independent_audit.sha256')))
    started = datetime.now(timezone.utc).isoformat()
    own_sha = digest(Path(__file__).read_bytes())
    artifact_names = ('PROTOCOL.md', 'thresholds.json', 'results.json', 'heldout_decisions.json', 'REPORT.md')
    artifacts = {name: read(OUT / name) for name in artifact_names}
    manifest_bytes = read(OUT / 'SHA256_MANIFEST.json')
    manifest = json.loads(manifest_bytes)
    runner_bytes = read(ROUND / 'code/fixed_fpr_operating_points.py')
    runner_sha = digest(runner_bytes)
    for name, data in artifacts.items():
        check('manifest_sha:' + name, digest(data) == manifest['files'][name], digest(data))
    check('manifest_runner_sha', runner_sha == manifest['runner_sha256'], runner_sha)
    thresholds = json.loads(artifacts['thresholds.json'])
    results = json.loads(artifacts['results.json'])
    decision_rows = json.loads(artifacts['heldout_decisions.json'])
    for docname, doc in (('thresholds', thresholds), ('results', results)):
        check(docname + '_runner_sha', doc['runner_sha256'] == runner_sha)
        check(docname + '_protocol_sha', doc['protocol_sha256'] == digest(artifacts['PROTOCOL.md']))
    check('results_threshold_sha', results['thresholds_sha256'] == digest(artifacts['thresholds.json']))

    # Read no old-experiment files except the six literal allowlisted inputs.
    old = {}
    for rel, wanted in EXPECTED.items():
        old[rel] = read(OLD / rel)
        check('frozen_input_sha:' + rel, digest(old[rel]) == wanted, digest(old[rel]))
    check('threshold_input_hashes', thresholds['input_sha256'] == {k: v for k, v in EXPECTED.items() if 'heldout_eval_labels' not in k})
    ledger = results['inputs']
    check('ledger_exact_six_inputs', len(ledger) == 6 and {str(OLD / k): v for k, v in EXPECTED.items()} == {r['path']: r['sha256'] for r in ledger})

    contract = json.loads(old['stage12/run/contract.json'])
    calibrator = json.loads(old['stage12/run/calibrator.json'])
    predictions = parse_rows(old['stage12/run/predictions.jsonl'])
    roles = ('calibration', 'heldout_eval')
    ids = {role: [r['source_id'] for r in contract['records'] if r['role'] == role] for role in roles}
    check('contract_240_unique_sources', all(len(ids[r]) == 120 and len(set(ids[r])) == 120 for r in roles) and len(set(ids['calibration']) | set(ids['heldout_eval'])) == 240)
    pred = {r['source_id']: r for r in predictions}
    check('prediction_240_exact_alignment', len(predictions) == len(pred) == 240 and set(pred) == set(ids['calibration']) | set(ids['heldout_eval']))
    check('prediction_roles_match_contract', all(pred[i]['role'] == role for role in roles for i in ids[role]))
    labels = {}
    for role, rel in (('calibration', 'stage12_calibration_plan/calibration_labels.jsonl'), ('heldout_eval', 'stage12_calibration_plan/heldout_eval_labels.jsonl')):
        rows = parse_rows(old[rel])
        check(role + '_labels_alignment_and_types', len(rows) == 120 and len({r['source_id'] for r in rows}) == 120 and {r['source_id'] for r in rows} == set(ids[role]) and all(r['role'] == role and type(r['is_anomalous']) is bool for r in rows))
        labels[role] = {r['source_id']: int(r['is_anomalous']) for r in rows}
        check(role + '_60normal_60anomaly', Counter(labels[role].values()) == Counter({0: 60, 1: 60}))
    raw = {i: float(p['probability']) for i, p in pred.items()}
    check('finite_probabilities_in_unit_interval', all(math.isfinite(p) and 0 <= p <= 1 for p in raw.values()))
    coefficient, intercept = float(calibrator['coefficient'][0][0]), float(calibrator['intercept'][0])
    check('frozen_positive_logit_mapping', coefficient > 0 and calibrator['classes'] == [0, 1] and calibrator['feature'] == 'logit(clip(p,1e-6,1-1e-6))' and calibrator['heldout_labels_used_for_fit'] is False)
    calibrated = {}
    for i, probability in raw.items():
        bounded = min(max(probability, 1e-6), 1 - 1e-6)
        linear = coefficient * (math.log(bounded) - math.log1p(-bounded)) + intercept
        calibrated[i] = 1.0 / (1.0 + math.exp(-linear))
    scores = {'raw': raw, 'calibrated': calibrated}
    check('predeclared_alpha_and_strict_comparison', thresholds['alphas'] == [0.05, 0.10] and thresholds['calibration_normal_n'] == 60 and thresholds['comparison'] == 'score > threshold')
    frozen_rows = {(r['alpha'], r['method']): r for r in thresholds['rows']}
    result_rows = {(r['alpha'], r['method']): r for r in results['rows']}
    expected_keys = {(a, m) for a in (0.05, 0.10) for m in scores}
    check('four_frozen_and_result_operating_points', len(thresholds['rows']) == len(results['rows']) == 4 and set(frozen_rows) == set(result_rows) == expected_keys)
    recomputed = []
    decisions = {}
    for alpha in (0.05, 0.10):
        for method, mapping in scores.items():
            key = (alpha, method)
            frozen, reported = frozen_rows[key], result_rows[key]
            rank = math.floor(alpha * 60) + 1
            normal_sorted = sorted((mapping[i] for i in ids['calibration'] if labels['calibration'][i] == 0), reverse=True)
            threshold = normal_sorted[rank - 1]
            check(str(key) + '_rank4_or_rank7_threshold', frozen['normal_threshold_one_based_rank'] == rank and frozen['maximum_calibration_fp'] == rank - 1 and close(frozen['threshold'], threshold))
            # Strict comparison uses independently recomputed threshold. Exact raw ties
            # remain exact after applying this scalar deterministic mapping.
            summary = {role: summarize(ids[role], labels[role], mapping, threshold) for role in roles}
            check(str(key) + '_calibration_counts_rates_Wilson_ties', equal(frozen['calibration'], summary['calibration']) and equal(reported['calibration'], summary['calibration']))
            check(str(key) + '_heldout_counts_rates_Wilson_ties', equal(reported['heldout'], summary['heldout_eval']))
            check(str(key) + '_threshold_copy_unchanged', equal({k: reported[k] for k in frozen}, frozen))
            check(str(key) + '_calibration_FP_upper_bound', summary['calibration']['fp'] <= rank - 1)
            for role in roles:
                decisions[(alpha, method, role)] = {i: mapping[i] > threshold for i in ids[role]}
            recomputed.append(dict(alpha=alpha, method=method, threshold=threshold, normal_threshold_one_based_rank=rank,
                                   calibration=summary['calibration'], heldout=summary['heldout_eval']))
    check('heldout_480_decision_rows', len(decision_rows) == 480)
    seen = set()
    max_score_error = max_threshold_error = 0.0
    stored_threshold_false = 0
    for row in decision_rows:
        alpha, method, i = row['alpha'], row['method'], row['source_id']
        key = (alpha, method, i)
        check('decision_row_key_unique_and_allowed', key not in seen and (alpha, method) in expected_keys and i in labels['heldout_eval'])
        seen.add(key)
        frozen = frozen_rows[(alpha, method)]['threshold']
        max_score_error = max(max_score_error, abs(row['score'] - scores[method][i]))
        max_threshold_error = max(max_threshold_error, abs(row['threshold'] - frozen))
        check('decision_row_fields_match_independent_recomputation', close(row['score'], scores[method][i]) and row['threshold'] == frozen and type(row['predicted_anomalous']) is bool and row['predicted_anomalous'] == decisions[(alpha, method, 'heldout_eval')][i] and row['label'] == labels['heldout_eval'][i])
        if row['score'] == row['threshold']:
            check('strict_threshold_tie_is_normal', row['predicted_anomalous'] is False)
            stored_threshold_false += 1
    check('heldout_all_4x120_keys_complete', seen == {(a, m, i) for a, m in expected_keys for i in ids['heldout_eval']})
    invariance = {}
    for role in roles:
        audit = order_audit(ids[role], raw, calibrated)
        audit['decision_mismatches_by_alpha'] = {str(a): sum(decisions[(a, 'raw', role)][i] != decisions[(a, 'calibrated', role)][i] for i in ids[role]) for a in (0.05, 0.10)}
        check(role + '_rank_tie_and_decision_audit_matches', equal(results['monotonicity_audit'][role], audit))
        check(role + '_positive_monotonic_mapping_preserves_rank_ties_and_decisions', audit['pairwise_order_or_tie_mismatches'] == 0 and all(v == 0 for v in audit['decision_mismatches_by_alpha'].values()))
        for method in scores:
            check(role + '_' + method + '_alpha5_detections_subset_alpha10', all(not decisions[(0.05, method, role)][i] or decisions[(0.10, method, role)][i] for i in ids[role]))
        invariance[role] = audit
    combined = order_audit(ids['calibration'] + ids['heldout_eval'], raw, calibrated)
    check('combined_240_rank_and_ties_preserved', combined['pairwise_order_or_tie_mismatches'] == 0)

    # A static/recorded-evidence check, not a reconstruction of unseen process history.
    source = runner_bytes.decode()
    tree = ast.parse(source)
    main_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    input_calls = [n for n in ast.walk(main_node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'read_input']
    check('runner_static_exact_allowlisted_input_calls', len(input_calls) == 6 and {n.args[0].value for n in input_calls} == set(EXPECTED))
    heldout_line = next(n.lineno for n in input_calls if n.args[0].value.endswith('heldout_eval_labels.jsonl'))
    publish_line = next(n.lineno for n in ast.walk(main_node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'publish' and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == 'thresholds.json')
    publish_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'publish')
    publish_text = ast.get_source_segment(source, publish_node)
    read_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'read_input')
    read_text = ast.get_source_segment(source, read_node)
    check('runner_static_threshold_publish_before_heldout_label_read', publish_line < heldout_line, {'threshold_publish_line': publish_line, 'heldout_label_read_line': heldout_line})
    check('runner_static_exclusive_publish_flush_fsync', ".open('xb')" in publish_text and 'f.flush()' in publish_text and 'os.fsync(f.fileno())' in publish_text)
    check('runner_static_allowlist_guard', "assert rel in EXPECTED" in read_text)
    frozen_time = datetime.fromisoformat(thresholds['frozen_at_utc'])
    saved_time = datetime.fromisoformat(results['thresholds_durably_saved_at_utc'])
    heldout_read_time = datetime.fromisoformat(next(r['read_at_utc'] for r in ledger if r['path'].endswith('heldout_eval_labels.jsonl')))
    check('ledger_records_freeze_save_before_heldout_read', frozen_time <= saved_time < heldout_read_time)
    check('frozen_no_refit_flags', thresholds['calibrator_refit'] is False and results['calibrator_refit'] is False)
    check('results_zero_API_GPU_confirmation_flags', results['api_calls'] == 0 and results['gpu_inference'] is False and results['confirmation_opened'] is False)
    check('source_counts_reported_correctly', results['calibration_source_n'] == results['heldout_source_n'] == 120 and results['source_overlap'] == 0)
    original_hashes_after = {name: digest((OUT / name).read_bytes()) for name in artifact_names}
    check('all_frozen_artifacts_unmodified', original_hashes_after == {name: digest(data) for name, data in artifacts.items()} and digest((OUT / 'SHA256_MANIFEST.json').read_bytes()) == digest(manifest_bytes) and digest((ROUND / 'code/fixed_fpr_operating_points.py').read_bytes()) == runner_sha)
    output = dict(schema='fixed-fpr-independent-audit-v1', status='PASS', started_at_utc=started,
                  completed_at_utc=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(), executable=sys.executable,
                  independent_code_sha256=own_sha, audited_runner_sha256=runner_sha,
                  audited_manifest_sha256=digest(manifest_bytes), frozen_artifact_sha256=original_hashes_after,
                  checks_passed=len(CHECKS), checks_failed=0, checks=CHECKS, input_read_manifest=READS,
                  old_inputs_allowlist_sha256=EXPECTED, recomputed_rows=recomputed,
                  source_counts={'calibration': 120, 'heldout_eval': 120, 'overlap': 0},
                  decision_rows_checked=480, unique_heldout_sources=120, decisions_per_alpha=240,
                  max_stored_calibrated_score_abs_error=max_score_error, max_stored_threshold_copy_abs_error=max_threshold_error,
                  stored_heldout_ties_verified_normal=stored_threshold_false,
                  mapping={'coefficient': coefficient, 'intercept': intercept, 'feature': calibrator['feature'], 'refit': False},
                  monotonicity_audit=invariance, combined240_order_audit=combined,
                  freeze_order_evidence={'type': 'static code and recorded ledger consistency only', 'threshold_publish_line': publish_line,
                                         'heldout_label_read_line': heldout_line, 'threshold_frozen_at_utc': thresholds['frozen_at_utc'],
                                         'threshold_saved_at_utc': results['thresholds_durably_saved_at_utc'], 'recorded_heldout_read_at_utc': heldout_read_time.isoformat(),
                                         'limitation': 'This later audit cannot independently reconstruct or prove invisible wall-clock execution/read history.'},
                  execution_scope={'new_API_calls': 0, 'GPU_inference': False, 'refit': False, 'threshold_tuning': False,
                                   'old_experiment_files_read': list(EXPECTED), 'stage13_or_confirmation_read_by_this_audit': False},
                  interpretation='Existing development heldout diagnostic only; empirical calibration FPR is not a population guarantee; intervals are descriptive and unadjusted.')
    json_bytes = (json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with (OUT / 'independent_audit.json').open('xb') as handle:
        handle.write(json_bytes)
    lines = ['# 固定经验 FPR 独立复核', '',
             f'结论：PASS，{len(CHECKS)} 项检查通过，0 项失败。全部统计在指定 AutoDL 实例 `{HOST}`、指定 Python 上完成；独立实现仅用 Python 标准库，未导入或执行原 runner。', '',
             '只读取六个明确 allowlisted 的 stage12 旧输入和本次 fixedFPR 冻结产物。未运行 API、GPU 推理、训练、重拟合或调阈值；本审计未读取 stage13/confirmation。原协议、代码、阈值、结果、逐样本决策、原报告和 SHA manifest 均保持原字节。', '',
             '| α | 分数 | normal降序rank | 阈值 | 校准 TP/FN/FP/TN | heldout TP/FN/FP/TN | heldout FPR（Wilson95%） | heldout Recall（Wilson95%） |',
             '|---:|---|---:|---:|---|---|---|---|']
    def pct(value):
        return f'{100 * value:.2f}%'
    def counts(row):
        return '/'.join(str(row[k]) for k in ('tp', 'fn', 'fp', 'tn'))
    for row in recomputed:
        h = row['heldout']
        lines.append(f"| {pct(row['alpha'])} | {row['method']} | {row['normal_threshold_one_based_rank']} | {row['threshold']:.16g} | {counts(row['calibration'])} | {counts(h)} | {pct(h['fpr'])} [{pct(h['fpr_wilson95'][0])}, {pct(h['fpr_wilson95'][1])}] | {pct(h['recall'])} [{pct(h['recall_wilson95'][0])}, {pct(h['recall_wilson95'][1])}] |")
    lines += ['', '两个 split 各120来源（60 normal、60 anomaly），交集0；预测240唯一来源完全对齐。冻结的四个运行点均重新得到第4/7高 normal 分数，严格 `score > threshold`，等于阈值均判正常。α=5%校准 FP=3/60（5%）；α=10%校准 FP=5/60（8.33%），后者因阈值上有2个 normal 并列而低于6/60。', '',
              f"逐条核完480条 heldout 决策（120来源×2方法×2α，每α240条），标签、映射分数、阈值和布尔决策全部一致；math标量实现与原存储分数最大绝对差 `{max_score_error:.3g}`，容差 `1e-14`。两个 split 的 raw/calibrated 决策在两α均0差异，5%检测集合均为10%检测集合子集。", '',
              f"calibration 的7140对中 ties=93，heldout 的7140对中 ties=87；两侧各62个不同分数，raw/calibrated 的顺序或并列差异均0，clip数均0。合并240来源的{combined['unordered_pairs']}对也全部保持顺序/并列。", '',
              f"阈值冻结顺序：原 runner 第{publish_line}行持久化 `thresholds.json`（独占创建、flush、fsync）后，第{heldout_line}行读取 heldout 标签；账本记录 `{thresholds['frozen_at_utc']}` 冻结，`{results['thresholds_durably_saved_at_utc']}` 保存，`{heldout_read_time.isoformat()}` 读取标签。此处仅审查静态代码及已记录时间的一致性；事后审计无法逆推出或独立证明不可见的实际读取时序。", '',
              '解释限度：这是已使用过的开发 heldout 上的运行点诊断；normal n=60，一次误报对应1.67个百分点。经验FPR不保证总体FPR；Wilson95%是边际、未校正的描述性区间，校准侧区间不包括阈值选择不确定性。正单调校准仅改变分数尺度，本结果没有固定FPR召回率或排名提升。', '',
              f'独立审计 JSON SHA256：`{digest(json_bytes)}`。',
              f'独立审计代码 SHA256：`{own_sha}`。',
              f'原 runner SHA256：`{runner_sha}`。',
              f'原 manifest SHA256：`{digest(manifest_bytes)}`。',
              '原冻结文件 SHA256：', '']
    lines += [f'- `{name}`：`{value}`' for name, value in original_hashes_after.items()]
    lines.append('')
    md_bytes = '\n'.join(lines).encode()
    with (OUT / 'independent_audit.md').open('xb') as handle:
        handle.write(md_bytes)
    checksum_text = f'{digest(json_bytes)}  independent_audit.json\n{digest(md_bytes)}  independent_audit.md\n{own_sha}  ../code/audit_fixed_fpr_independent.py\n'
    with (OUT / 'independent_audit.sha256').open('x') as handle:
        handle.write(checksum_text)
    print(json.dumps({'status': 'PASS', 'checks_passed': len(CHECKS), 'checks_failed': 0,
                      'audit_json_sha256': digest(json_bytes), 'audit_md_sha256': digest(md_bytes),
                      'code_sha256': own_sha, 'decision_rows_checked': 480, 'rows': recomputed,
                      'monotonicity_audit': invariance, 'combined240_order_audit': combined}, ensure_ascii=False))


if __name__ == '__main__':
    main()
