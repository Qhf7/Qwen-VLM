#!/usr/bin/env python3
"""One frozen server-only operating-point diagnostic; no API, GPU or fitting."""
import argparse
import hashlib
import json
import math
import socket
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT = Path('/root/autodl-tmp/industrial_vlm')
OLD = ROOT / 'experiments/jev_followup_20261111'
OUT = ROOT / 'experiments/jev_decision_round2_20261111/fixed_fpr_operating_points'
EXPECTED = {
    'code/isolated_calibration.py': 'd8489c953ec4cb5e9d7804643d35f43d5ad0801211abc118bcc8e0003fec37d1',
    'stage12/run/calibrator.json': '0bac063df108bd44938c25a8bb2c07261d79bfe2350b6d1bdb8b8cf1c1bcd9ec',
    'stage12/run/predictions.jsonl': 'f81d0006dcd17d305f768f4065cd5ce4b7da5a991200955a5f7ff2c46af9116a',
    'stage12/run/contract.json': '45d593a9833ab2b1dd003afb954b6eca10cd85a574055f3d0e821ac7aa515ea1',
    'stage12_calibration_plan/calibration_labels.jsonl': 'aaaf4898122c685ed5a511af5b1d0d85d6eaa4d699a7822174eaa2a84a56fd1d',
    'stage12_calibration_plan/heldout_eval_labels.jsonl': '7cc2853e08c36af1491a29b60622d42cdc2a377d803f07b962dd5a81942761d3',
}
READS = []

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(data):
    return hashlib.sha256(data).hexdigest()

def read_input(rel):
    assert rel in EXPECTED, 'Input is outside explicit allowlist'
    data = (OLD / rel).read_bytes()
    assert sha(data) == EXPECTED[rel], 'Frozen input hash mismatch: ' + rel
    READS.append(dict(path=str(OLD / rel), sha256=sha(data), read_at_utc=now()))
    return data

def publish(name, value):
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with (OUT / name).open('xb') as f:
        f.write(data)
        f.flush()
        import os
        os.fsync(f.fileno())
    return sha(data)

def rows(data):
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]

def wilson(k, n):
    assert n > 0 and 0 <= k <= n
    z = 1.959963984540054
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [max(0., center-half), min(1., center+half)]

def count_summary(y, decision, scores, threshold):
    y = np.asarray(y, dtype=int)
    d = np.asarray(decision, dtype=bool)
    tp = int(np.sum((y == 1) & d))
    fp = int(np.sum((y == 0) & d))
    fn = int(np.sum((y == 1) & ~d))
    tn = int(np.sum((y == 0) & ~d))
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, normal_n=fp+tn, anomalous_n=tp+fn,
        fpr=fp/(fp+tn), fpr_wilson95=wilson(fp, fp+tn),
        recall=tp/(tp+fn), recall_wilson95=wilson(tp, tp+fn),
        scores_equal_threshold_total=int(np.sum(scores == threshold)),
        normal_scores_equal_threshold=int(np.sum((y == 0) & (scores == threshold))),
        anomalous_scores_equal_threshold=int(np.sum((y == 1) & (scores == threshold))))

def label_map(data, role, ids):
    rr = rows(data)
    assert len(rr) == 120 and len({r['source_id'] for r in rr}) == 120
    assert {r['source_id'] for r in rr} == set(ids)
    assert all(r['role'] == role and type(r['is_anomalous']) is bool for r in rr)
    lookup = {r['source_id']: int(r['is_anomalous']) for r in rr}
    assert Counter(lookup.values()) == {0: 60, 1: 60}
    return np.array([lookup[i] for i in ids], dtype=int)

def transform(p, calibrator):
    x = np.clip(p, 1e-6, 1-1e-6)
    a = float(calibrator['coefficient'][0][0])
    b = float(calibrator['intercept'][0])
    assert a > 0 and calibrator['classes'] == [0, 1]
    return expit(a * np.log(x/(1-x)) + b)

def rank_audit(raw, cal):
    sr = np.sign(raw[:, None] - raw[None, :])
    sc = np.sign(cal[:, None] - cal[None, :])
    upper = np.triu(np.ones(sr.shape, dtype=bool), 1)
    return dict(unordered_pairs=int(np.sum(upper)),
        pairwise_order_or_tie_mismatches=int(np.sum((sr != sc) & upper)),
        raw_tied_pairs=int(np.sum((sr == 0) & upper)),
        calibrated_tied_pairs=int(np.sum((sc == 0) & upper)),
        raw_unique_scores=int(len(np.unique(raw))), calibrated_unique_scores=int(len(np.unique(cal))),
        scores_clipped=int(np.sum((raw < 1e-6) | (raw > 1-1e-6))))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true', required=True)
    parser.parse_args()
    assert str(Path.cwd()).startswith(str(ROOT)), 'Execute only on approved server workspace'
    assert socket.gethostname() == 'autodl-container-132444827f-9c594038'
    assert OUT.is_dir() and (OUT / 'PROTOCOL.md').is_file()
    assert not (OUT / 'thresholds.json').exists() and not (OUT / 'results.json').exists()
    started = now()
    code_sha = sha(Path(__file__).read_bytes())
    protocol_sha = sha((OUT / 'PROTOCOL.md').read_bytes())
    read_input('code/isolated_calibration.py')
    contract = json.loads(read_input('stage12/run/contract.json'))
    calibrator = json.loads(read_input('stage12/run/calibrator.json'))
    predictions = rows(read_input('stage12/run/predictions.jsonl'))
    assert len(predictions) == 240 and len({p['source_id'] for p in predictions}) == 240
    pred = {p['source_id']: p for p in predictions}
    records = {role: [r for r in contract['records'] if r['role'] == role] for role in ('calibration', 'heldout_eval')}
    assert all(len(v) == 120 for v in records.values())
    ids = {role: [r['source_id'] for r in rr] for role, rr in records.items()}
    assert not set(ids['calibration']) & set(ids['heldout_eval'])
    for role, rr in records.items():
        assert all(pred[r['source_id']]['role'] == role for r in rr)
    raw = {role: np.array([pred[i]['probability'] for i in role_ids], dtype=float) for role, role_ids in ids.items()}
    assert all(np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)) for p in raw.values())
    cal = {role: transform(p, calibrator) for role, p in raw.items()}
    ycal = label_map(read_input('stage12_calibration_plan/calibration_labels.jsonl'), 'calibration', ids['calibration'])
    scoring = {'raw': raw, 'calibrated': cal}
    thresholds = dict(schema='fixed-empirical-fpr-operating-points-v1', frozen_at_utc=now(),
        protocol_sha256=protocol_sha, runner_sha256=code_sha,
        calibration_normal_n=60, alphas=[0.05, 0.10], comparison='score > threshold',
        rule='descending normal score at one-based rank floor(alpha*n_normal)+1',
        heldout_labels_read_before_threshold_freeze=False, calibrator_refit=False,
        input_sha256={k:v for k,v in EXPECTED.items() if 'heldout_eval_labels' not in k},
        rows=[])
    for alpha in (0.05, 0.10):
        k = math.floor(alpha * 60)
        for method, by_role in scoring.items():
            p = by_role['calibration']
            normal = np.sort(p[ycal == 0])[::-1]
            threshold = float(normal[k])
            decisions = p > threshold
            summary = count_summary(ycal, decisions, p, threshold)
            assert summary['fp'] <= k
            thresholds['rows'].append(dict(alpha=alpha, method=method, threshold=threshold,
                normal_threshold_one_based_rank=k+1, maximum_calibration_fp=k, calibration=summary))
    threshold_sha = publish('thresholds.json', thresholds)
    threshold_saved_at = now()
    # The held-out labels are first read only after thresholds.json is durably written.
    yh = label_map(read_input('stage12_calibration_plan/heldout_eval_labels.jsonl'), 'heldout_eval', ids['heldout_eval'])
    result_rows = []
    decision_rows = []
    for entry in thresholds['rows']:
        row = dict(entry)
        ph = scoring[row['method']]['heldout_eval']
        decision = ph > row['threshold']
        row['heldout'] = count_summary(yh, decision, ph, row['threshold'])
        result_rows.append(row)
        for i, source_id in enumerate(ids['heldout_eval']):
            decision_rows.append(dict(source_id=source_id, alpha=row['alpha'], method=row['method'],
                score=float(ph[i]), threshold=row['threshold'], predicted_anomalous=bool(decision[i]), label=int(yh[i])))
    invariance = {}
    for role in ('calibration', 'heldout_eval'):
        invariance[role] = rank_audit(raw[role], cal[role])
        invariance[role]['decision_mismatches_by_alpha'] = {}
        for alpha in (0.05, 0.10):
            tr = next(r['threshold'] for r in thresholds['rows'] if r['alpha'] == alpha and r['method'] == 'raw')
            tc = next(r['threshold'] for r in thresholds['rows'] if r['alpha'] == alpha and r['method'] == 'calibrated')
            invariance[role]['decision_mismatches_by_alpha'][str(alpha)] = int(np.sum((raw[role] > tr) != (cal[role] > tc)))
    results = dict(schema='fixed-empirical-fpr-operating-points-v1', started_at_utc=started, completed_at_utc=now(),
        hostname=socket.gethostname(), executable=sys.executable, protocol_sha256=protocol_sha, runner_sha256=code_sha,
        thresholds_sha256=threshold_sha, thresholds_durably_saved_at_utc=threshold_saved_at,
        inputs=READS, rows=result_rows, monotonicity_audit=invariance,
        calibration_source_n=120, heldout_source_n=120, source_overlap=0,
        api_calls=0, gpu_inference=False, calibrator_refit=False, confirmation_opened=False,
        interpretation='new development diagnostic on previously evaluated heldout; not a new blind test or true-FPR guarantee',
        alpha_selection='both predeclared alpha values reported; no heldout-based alpha choice',
        wilson_note='Marginal binomial Wilson intervals are descriptive, unadjusted; calibration FPR interval does not cover threshold selection uncertainty or guarantee population risk.')
    results_sha = publish('results.json', results)
    decisions_sha = publish('heldout_decisions.json', decision_rows)
    def pc(v): return f'{100*v:.2f}%'
    def ci(v): return f'[{100*v[0]:.2f}%, {100*v[1]:.2f}%]'
    md = ['# 固定经验 FPR 运行点诊断', '',
        '本诊断仅使用 stage12 已有 calibration120（60 normal、60 anomaly）与既有 heldout120（60 normal、60 anomaly）。未发 API、未运行 GPU、未重拟合校准器，未读取 stage13。', '',
        '## 事前固定规则', '',
        '同时报告 α=5% 与 10%，不根据 held-out 结果反选。将 60 个校准 normal 分数降序排列，取第 floor(α×60)+1 高为阈值，且只有 score > threshold 判异常；因此分别使用第 4/7 高分数。阈值先持久化，再首次读取 held-out 标签。', '',
        '冻结映射为 sigmoid(1.56720531322167×logit(clip(p,1e−6,1−1e−6))−0.15632219069119976)。正系数单调映射改变数值尺度，一般保持排名和对应阈值决策；本次逐对核验 ties、clip 及全部决策。', '',
        '| 目标经验FPR | 分数 | 阈值 | 校准FP / normal | 校准实际FPR | Held-out FP / normal（FPR; Wilson95%） | Held-out TP / anomaly（Recall; Wilson95%） |',
        '|---:|---|---:|---:|---:|---|---|']
    for r in result_rows:
        c, h = r['calibration'], r['heldout']
        md.append(f"| {pc(r['alpha'])} | {r['method']} | {r['threshold']:.12g} | {c['fp']}/{c['normal_n']} | {pc(c['fpr'])} | {h['fp']}/{h['normal_n']}（{pc(h['fpr'])}; {ci(h['fpr_wilson95'])}） | {h['tp']}/{h['anomalous_n']}（{pc(h['recall'])}; {ci(h['recall_wilson95'])}） |")
    md += ['', '## 并列与单调性核验', '', '```json', json.dumps(invariance, ensure_ascii=False, indent=2), '```', '',
        '所有等于阈值的样本均按 strict > 规则判正常。并列可能使校准实际 FPR 低于目标；本次 ties 数量及 raw/calibrated 决策差异见表和 JSON。', '',
        '## 解释边界', '',
        '- 这是已看开发验证集上的新运行点诊断，不是新的盲测，不读取或代表 stage13 确认集。',
        '- 目标是校准集经验 FPR，不是保证真实总体 FPR 固定为 5%/10%。60 个 normal 使一张误报就相当于 1.67 个百分点，尾部阈值分别只由第 4/7 个极值决定；存在抽样误差、类别组成和分布迁移风险。',
        '- Held-out FPR 和 Recall 必须同时报告计数与区间。Wilson95% 为边际描述性区间，未做多重比较校正；对用同一校准 normal 自适应选择的阈值，校准 FPR 的二项区间也不能担保阈值选择后总体风险。',
        '- raw 与 calibrated 共享同一模型预测与正单调映射，排名不变时固定经验 FPR 下的 Recall 不会凭校准凭空提高。不得把尺度改善误写成检测排名提升。',
        '- 两个 α 都是事前固定的报告点，不依据 held-out 表挑选后宣称最优方案。', '',
        f'协议 SHA256：`{protocol_sha}`；runner：`{code_sha}`；阈值：`{threshold_sha}`；结果：`{results_sha}`。', '']
    with (OUT / 'REPORT.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(md))
    manifest = dict(created_at_utc=now(), files={name: sha((OUT / name).read_bytes()) for name in ('PROTOCOL.md', 'thresholds.json', 'results.json', 'heldout_decisions.json', 'REPORT.md')}, runner_sha256=code_sha)
    publish('SHA256_MANIFEST.json', manifest)
    print(json.dumps(dict(status='complete', output=str(OUT), rows=result_rows, monotonicity=invariance, results_sha256=results_sha), ensure_ascii=False))

if __name__ == '__main__':
    main()
