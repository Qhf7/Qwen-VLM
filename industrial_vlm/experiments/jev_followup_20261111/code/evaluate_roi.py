"""Evaluate complete paired Jev ROI runs on the designated server only.

No model/API is called. Formal predictions, request identities, assets and
dependencies are verified and their hashes are frozen before any GT mask is
opened. The frozen 120 development queries are the only allowed RGB/GT inputs.

  python evaluate_roi.py --root RUN_ROOT --run runs/modelgo_roi
  python evaluate_roi.py --self-test

The ROI Noul concerns a candidate region, so whole-image Brier/ECE is not
reported as regional probability calibration. Historical full-image gates
remain historical controls, including renderer/semantic/budget differences.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import tempfile

import numpy as np
from PIL import Image

import roi_api as api

HOSTNAME = 'autodl-container-132444827f-9c594038'
SCHEMA = 'jev-fixed-candidate-paired-evaluation-v1'
METRICS = ('anomaly_bbox_iou_mean_all', 'anomaly_iou_at_05_all',
           'anomaly_retained_recall', 'normal_positive_box_rate')
REPLICATES = 2000
SEED = 42
QWEN_EXPECTED_SHA = {
    'base': 'dc109510af9e58701257ef943e05cee4b100f46df80a14620b36fdfb64f96499',
    'decision_decision': 'd19c70935d3369d400a921d5a9281c6382566c327667b9b3f6de0bb3a7b4681c',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def publish(path, value):
    """Publish once; an identical rerun may reuse the frozen artifact."""
    path = Path(path)
    if path.exists():
        require(canonical(read(path)) == canonical(value), 'Frozen artifact differs: ' + path.name)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.publish.tmp')
    with temporary.open('wb') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def wilson_interval(successes, total):
    """Descriptive binomial Wilson95; not a claim of deployment sampling."""
    if not total:
        return None
    z=1.959963984540054
    p=successes/total
    denominator=1+z*z/total
    center=(p+z*z/(2*total))/denominator
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0.,center-half),min(1.,center+half)]


def box_iou(first, second):
    if first is None or second is None:
        return 0.0
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1]))
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    union = area(first) + area(second) - intersection
    return float(intersection / union) if union > 0 else 0.0


def mask_box(binary):
    binary = np.asarray(binary) > 0
    if binary.ndim == 3:
        binary = binary.any(axis=2)
    require(binary.ndim == 2 and binary.any(), 'Expected a nonempty anomaly mask')
    ys, xs = np.nonzero(binary)
    height, width = binary.shape
    return [float(xs.min() / width), float(ys.min() / height),
            float((xs.max() + 1) / width), float((ys.max() + 1) / height)]


def validate_formal(root, output):
    """No labels/masks are opened in this integrity phase."""
    status = read(output / 'status.json')
    require(status.get('status') == 'complete' and status.get('completed') == 240
            and status.get('expected') == 240
            and status.get('mode') == 'formal_development_candidate_review',
            'Need a complete formal 240-target paired run before GT access')
    saved = read(output / 'contract.json')
    require(saved.get('schema') == api.SCHEMA
            and saved.get('mode') == 'formal_development_candidate_review'
            and saved.get('limit_queries') is None, 'Formal contract required')
    contract, images = api.build_contract(root, saved['source_candidates'],
                                          saved['source_baseline'], None)
    require(canonical(saved) == canonical(contract), 'Rebuilt ROI contract differs')
    contract_sha = api.digest(api.canonical(contract))
    require(status.get('contract_sha256') == contract_sha, 'Status contract differs')
    completed, starts, attempts, _, unknown = api.validate_logs(output, contract, images)
    schedule = {(r['source_id'], r['arm']) for r in contract['schedule']}
    require(len(schedule) == len(contract['schedule']) == len(completed) == 240
            and set(completed) == schedule, 'Accepted targets are incomplete or duplicated')
    require(status.get('by_arm') == {'roi': 120, 'duplicate': 120}, 'Status arm counts differ')
    require(not unknown, 'Unresolved started intents remain')
    require(status.get('confirmation_images_read') is False, 'Confirmation isolation flag differs')
    return status, contract, completed, starts, attempts


def freeze_inputs(root, output, contract, status):
    paths = {
        'formal_predictions': output / 'predictions.jsonl',
        'formal_attempts': output / 'attempts.jsonl',
        'formal_contract': output / 'contract.json',
        'formal_status': output / 'status.json',
        'candidates': root / contract['source_candidates'],
        'protocol': root / 'stages/03_protocol.json',
        'source_audit': root / 'stages/01_source_audit.json',
        'teacher_alignment': root / 'stages/02_teacher_alignment.json',
        'source_plan': root / 'source_baseline/frozen/plan.json',
        'source_config': root / 'source_baseline/frozen/config.json',
        'source_materialized': root / 'source_baseline/frozen/materialized.json',
        'source_labels': root / 'source_baseline/frozen/labels.json',
        'historical_predictions': root / 'source_baseline/runs/modelgo_448/predictions.jsonl',
        'evaluator': Path(__file__),
        'api_runner': Path(api.__file__),
    }
    qwen_paths={name:root/'source_qwen'/f'{name}_scores.jsonl' for name in QWEN_EXPECTED_SHA}
    if any(path.exists() for path in qwen_paths.values()):
        require(all(path.exists() for path in qwen_paths.values()),'Both audited Qwen controls required')
        for name,path in qwen_paths.items():
            require(sha_file(path)==QWEN_EXPECTED_SHA[name],'Audited Qwen control file hash differs')
            paths['qwen_'+name]=path
    hashes = {name: sha_file(path) for name, path in paths.items()}
    path = root / 'stages/04_prediction_freeze.json'
    if path.exists():
        frozen = read(path)
        require(frozen.get('files_sha256') == hashes
                and frozen.get('run') == output.relative_to(root).as_posix(),
                'Prediction freeze inputs changed')
        return frozen
    frozen = dict(schema='jev-paired-predictions-frozen-before-gt-v1',
        timestamp_utc=now(), hostname=socket.gethostname(),
        run=output.relative_to(root).as_posix(), n_predictions=240,
        files_sha256=hashes, contract_sha256=status['contract_sha256'],
        confirmation_opened=False, ground_truth_masks_opened=False,
        evaluation_policy='Fixed threshold0.5, all-anomaly/normal denominators; no GT tuning',
        bootstrap=dict(replicates=REPLICATES, seed=SEED, strata='category x image_label',
                       paired=True, interval='percentile95', multiplicity_corrected=False))
    publish(path, frozen)
    return frozen


def validate_sources(root, contract):
    plan = read(root / 'source_baseline/frozen/plan.json')
    config = read(root / 'source_baseline/frozen/config.json')
    labels = read(root / 'source_baseline/frozen/labels.json')
    require(digest(canonical(labels)) == config['labels_sha256'], 'Historical labels hash differs')
    require(digest(canonical(plan)) == config['plan_sha256'], 'Historical plan hash differs')
    queries = plan['queries']
    by_sid = {r['source_id']: r for r in queries}
    require(len(by_sid) == len(queries) == 120 and set(by_sid) == set(labels), 'Query/label set differs')
    require({r['source_id'] for r in contract['records']} == set(by_sid), 'ROI and source query set differs')
    require(all(type(y) is int and y in (0, 1) for y in labels.values()), 'Labels must be binary integers')
    counts = Counter((r['category'], labels[r['source_id']]) for r in queries)
    categories = sorted({r['category'] for r in queries})
    require(len(categories) == 12 and all(counts[c, y] == 5 for c in categories for y in (0, 1)),
            'Need 12 categories with five normal and five anomaly queries each')
    audit = read(root / 'stages/01_source_audit.json')
    require(audit.get('status') == 'complete' and audit.get('passed') == audit.get('total') == 168,
            'Original image source audit incomplete')
    originals = {r['source_id']: r for r in audit['records'] if r['role'] == 'query'}
    for sid, row in by_sid.items():
        require(sid == 'visa_' + row['sha256'] and sid in originals, 'Original source identity differs')
        current = originals[sid]
        image = Path(current['path'])
        require(current['sha256'] == row['sha256'] and current['passed'] is True
                and image.is_file() and image.stat().st_size == row['source_bytes']
                and sha_file(image) == row['sha256'], 'Current original RGB changed')
    return by_sid, labels, originals


def load_targets(root, by_sid, labels, originals, candidates):
    """Called only after the formal prediction freeze has been published."""
    records = []
    for sid in sorted(by_sid):
        source = by_sid[sid]
        candidate = candidates[sid]
        require(candidate['category'] == source['category'], 'Candidate category differs')
        y = labels[sid]
        image = Path(originals[sid]['path'])
        target = None
        metadata = None
        if y:
            parts = list(image.parts)
            position = next((i for i in range(len(parts) - 2)
                             if parts[i:i+3] == ['Data', 'Images', 'Anomaly']), None)
            require(position is not None, 'Anomaly path does not follow VisA layout')
            parts[position + 1] = 'Masks'
            mask = Path(*parts).with_suffix('.png')
            require(mask.is_file(), 'Approved query GT mask missing')
            raw = mask.read_bytes()
            import io
            with Image.open(io.BytesIO(raw)) as opened:
                pixels = np.asarray(opened)
                size = list(opened.size)
            target = mask_box(pixels)
            require(size == candidate['original_size'], 'GT and original RGB dimensions differ')
            metadata = dict(path=str(mask), sha256=digest(raw), bytes=len(raw), size=size,
                            target_bbox_original_norm=target)
        pred_box = api.validate_box(candidate['bbox_original_norm'])
        area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        full = area == 1.0
        identical = candidate['assets']['query']['png_sha256'] == candidate['assets']['crop']['png_sha256']
        if 'full_image_candidate' in candidate:
            require(bool(candidate['full_image_candidate']) == full, 'Full-image candidate flag differs')
        records.append(dict(source_id=sid, category=source['category'], label=y,
            original_sha256=source['sha256'], teacher_split=candidate['teacher_split'],
            teacher_training_exposed=candidate['teacher_split'] == 'train',
            candidate_bbox_original_norm=pred_box, gt_bbox_original_norm=target,
            candidate_iou=box_iou(pred_box, target) if y else 0.0,
            candidate_area_fraction=float(area), full_image_candidate=full,
            query_crop_bytes_identical=identical, ground_truth=metadata))
    require(sum(r['label'] for r in records) == 60, 'Need exactly60 anomaly queries')
    return records


def historical_predictions(root, records):
    rows = api.read_rows(root / 'source_baseline/runs/modelgo_448/predictions.jsonl')
    allowed = {r['source_id']: r for r in records}
    found = {}
    for row in rows:
        pair = row.get('source_id'), row.get('shots')
        require(pair[0] in allowed and pair[1] in (0, 1, 4) and pair not in found,
                'Historical target unexpected or duplicate')
        require(row.get('category') == allowed[pair[0]]['category'], 'Historical category differs')
        api.response_probability({'model': row.get('response_model'),
            'answers': {'anomaly': {'type': 'noul', 'noul': row.get('probability')}}})
        require(api.valid_hash(row.get('request_sha256'))
                and api.valid_hash(row.get('contract_sha256')), 'Historical identity hash invalid')
        found[pair] = row
    expected = {(sid, shot) for sid in allowed for shot in (0, 1, 4)}
    require(len(found) == len(rows) == 360 and set(found) == expected, 'Historical360 incomplete')
    require(len({r['contract_sha256'] for r in rows}) == 1, 'Historical contracts mixed')
    return found


def qwen_predictions(root, rows):
    sources={r['source_id']:r for r in rows}
    files={name:root/'source_qwen'/f'{name}_scores.jsonl' for name in QWEN_EXPECTED_SHA}
    if not any(path.exists() for path in files.values()):
        return {},dict(status='not_aligned_not_evaluated',
            reason='No audited same-source Qwen scorer manifest supplied; MVTec1575 results not substituted')
    require(all(path.exists() for path in files.values()),'Both audited Qwen score files required')
    controls,summary={},{}
    for name,path in files.items():
        require(sha_file(path)==QWEN_EXPECTED_SHA[name],'Qwen source SHA differs')
        records=api.read_rows(path)
        found={}
        for value in records:
            sid=value.get('source_id')
            require(sid in sources and sid not in found,'Unexpected or duplicate Qwen source')
            expected=sources[sid]
            require(value.get('schema_version')=='industrial-vlm-boolean-score-v1'
                    and value.get('image_sha256')==expected['original_sha256']
                    and value.get('category')==expected['category']
                    and type(value.get('label')) is bool and int(value['label'])==expected['label'],
                    'Qwen identity/category/image SHA/label differs')
            odds=value.get('anomaly_log_odds')
            require(type(odds) in (int,float) and math.isfinite(odds),'Invalid Qwen log odds')
            score=value.get('anomaly_score')
            require(type(score) in (int,float) and math.isfinite(score) and 0<=score<=1,
                    'Invalid Qwen score')
            found[sid]=value
        require(len(records)==len(found)==120 and set(found)==set(sources),'Qwen120 not aligned')
        controls['historical_qwen_'+name]=[found[r['source_id']]['anomaly_log_odds']>0 for r in rows]
        summary[name]=dict(file=path.relative_to(root).as_posix(),sha256=sha_file(path),n=120,
            gate='anomaly_log_odds > 0 strictly; equality rejects',
            normalized_score_note='sigmoid of candidate log odds; not independently calibrated')
    return controls,dict(status='audited_same_source_historical_controls',models=summary,
        comparison_scope='All120 source/image SHA/category/label aligned; native max_pixels524288 differs from Jev448',
        not_same_input_ablation=True,validation_tuned_FPR5_threshold_used=False,
        training_audit='Decision-decision training768 unique sources has0 query overlap;48 supports are training subset')


def metrics(rows, keep, indices=None):
    selected = list(range(len(rows))) if indices is None else list(indices)
    anomaly = [i for i in selected if rows[i]['label']]
    normal = [i for i in selected if not rows[i]['label']]
    tp = sum(bool(keep[i]) for i in anomaly)
    fp = sum(bool(keep[i]) for i in normal)
    values = [rows[i]['candidate_iou'] if keep[i] else 0.0 for i in anomaly]
    return dict(n=len(selected), n_anomaly=len(anomaly), n_normal=len(normal),
        TP=tp, FN=len(anomaly)-tp, FP=fp, TN=len(normal)-fp,
        anomaly_bbox_iou_mean_all=fraction(sum(values), len(anomaly)),
        anomaly_iou_at_05_all=fraction(sum(value >= .5 for value in values), len(anomaly)),
        anomaly_retained_recall=fraction(tp, len(anomaly)),
        normal_positive_box_rate=fraction(fp, len(normal)),
        anomaly_retained_recall_wilson95=wilson_interval(tp,len(anomaly)),
        normal_positive_box_rate_wilson95=wilson_interval(fp,len(normal)),
        binomial_interval_note='Descriptive Wilson95 with observed group denominator; fixed development set, not deployment guarantee',
        anomaly_retained_iou_mean=fraction(sum(values), tp))


def package(rows, keep, indices=None):
    selected = list(range(len(rows))) if indices is None else list(indices)
    by_category = {category: metrics(rows, keep, [i for i in selected if rows[i]['category'] == category])
                   for category in sorted({rows[i]['category'] for i in selected})}
    macro = {}
    for name in METRICS:
        values = [value[name] for value in by_category.values() if value[name] is not None]
        macro[name] = dict(value=fraction(sum(values), len(values)), categories_with_denominator=len(values))
    return dict(pooled=metrics(rows, keep, selected), category_macro=macro, by_category=by_category)


def bootstrap_difference(rows, first, second, indices=None, replicates=REPLICATES, seed=SEED):
    selected = list(range(len(rows))) if indices is None else list(indices)
    if not selected:
        return dict(n=0, replicates=replicates, seed=seed, pooled={}, category_macro={})
    groups = {}
    for local, index in enumerate(selected):
        groups.setdefault((rows[index]['category'], rows[index]['label']), []).append(local)
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, len(selected)), dtype=np.float64)
    for key in sorted(groups):
        members = groups[key]
        draws = rng.multinomial(len(members), np.full(len(members), 1/len(members)), size=replicates)
        weights[:, members] = draws
    labels = np.array([rows[i]['label'] for i in selected], dtype=bool)
    ious = np.array([rows[i]['candidate_iou'] for i in selected], dtype=float)
    a = np.array([first[i] for i in selected], dtype=float)
    b = np.array([second[i] for i in selected], dtype=float)
    arrays = dict(anomaly_bbox_iou_mean_all=(a-b)*ious,
        anomaly_iou_at_05_all=(a-b)*(ious >= .5),
        anomaly_retained_recall=a-b, normal_positive_box_rate=a-b)
    full_a, full_b = metrics(rows, first, selected), metrics(rows, second, selected)
    pooled, macro = {}, {}
    categories = sorted({rows[i]['category'] for i in selected})
    for name, difference in arrays.items():
        mask = ~labels if name == 'normal_positive_box_rate' else labels
        n = int(mask.sum())
        if n:
            values = weights[:, mask] @ difference[mask] / n
            pooled[name] = dict(delta=full_a[name]-full_b[name],
                ci95=[float(v) for v in np.quantile(values, [.025,.975], method='linear')],
                denominator=n)
        else:
            pooled[name] = dict(delta=None, ci95=None, denominator=0)
        category_values, point_values = [], []
        for category in categories:
            category_mask = mask & np.array([rows[i]['category'] == category for i in selected])
            denominator = int(category_mask.sum())
            if denominator:
                category_values.append(weights[:, category_mask] @ difference[category_mask] / denominator)
                point_values.append(float(difference[category_mask].mean()))
        if category_values:
            values = np.mean(category_values, axis=0)
            macro[name] = dict(delta=float(np.mean(point_values)),
                ci95=[float(v) for v in np.quantile(values,[.025,.975],method='linear')],
                categories_with_denominator=len(category_values))
        else:
            macro[name] = dict(delta=None, ci95=None, categories_with_denominator=0)
    return dict(n=len(selected), replicates=replicates, seed=seed,
        strata='category x image_label, resample within selected subset', paired=True,
        method='percentile95; fixed categories, support, model version and candidate rule',
        no_multiplicity_correction=True, includes_provider_randomness=False,
        pooled=pooled, category_macro=macro)


def operation_summary(completed, starts, attempts):
    result = dict(accepted=len(completed), started=sum(starts.values()),
        explicit_failed=sum(r.get('event') == 'failed' for r in attempts),
        failed_categories=dict(Counter(r.get('error') or r.get('error_class') or 'unspecified'
            for r in attempts if r.get('event') == 'failed')),
        usage_scope='Observed accepted-response usage only; missing fields/failed usage are unknown, not zero',
        currency_cost=None, currency_cost_note='No reliable currency billing data available', arms={})
    for arm in ('roi','duplicate'):
        rows=[r for (sid,a),r in completed.items() if a==arm]
        seconds=np.array([r['seconds'] for r in rows],dtype=float)
        fields=sorted({key for row in rows for key in row.get('usage',{})})
        observed={}
        for field in fields:
            values=[row['usage'][field] for row in rows if field in row.get('usage',{})]
            observed[field]=dict(total_observed=float(sum(values)),responses_with_field=len(values),
                                 responses_missing_field=len(rows)-len(values))
        result['arms'][arm]=dict(n_accepted=len(rows), usage=observed,
            latency_seconds=dict(mean=float(seconds.mean()),median=float(np.median(seconds)),
                p95=float(np.quantile(seconds,.95,method='linear')),total_observed=float(seconds.sum())),
            timing_scope='Client request latency under shared gateway load; not controlled intrinsic model speed')
    return result


def candidate_diagnostics(rows,completed,indices):
    if not indices:
        return dict(n=0)
    area=np.array([rows[i]['candidate_area_fraction'] for i in indices],dtype=float)
    roi=np.array([completed[rows[i]['source_id'],'roi']['probability'] for i in indices],dtype=float)
    duplicate=np.array([completed[rows[i]['source_id'],'duplicate']['probability'] for i in indices],dtype=float)
    return dict(n=len(indices),candidate_area_fraction=dict(
        minimum=float(area.min()),median=float(np.median(area)),maximum=float(area.max()),
        quantile25=float(np.quantile(area,.25,method='linear')),quantile75=float(np.quantile(area,.75,method='linear'))),
        paired_probability_mean_abs_difference=float(np.abs(roi-duplicate).mean()),
        paired_probability_max_abs_difference=float(np.abs(roi-duplicate).max()),
        paired_exact_probability_equal=int(np.sum(roi==duplicate)),
        paired_threshold_decision_equal=int(np.sum((roi>=.5)==(duplicate>=.5))),
        note='Same-pixel duplicate cases diagnose provider repeat variability; no additional randomized replicates')


def fmt(value, percent=False):
    return 'NA' if value is None else (f'{value*100:.2f}%' if percent else f'{value:.6f}')


def report_document(result):
    lines=['# Jev 固定候选区域复核结果', '',
        f"评价时间（UTC）：{result['timestamp_utc']}；服务器：`{result['hostname']}`。", '',
        '## 评价边界', '',
        '- 正式240个目标完整，预测/请求/图像/依赖一致性通过并冻结后，才读取当前120张清单内60张异常图的GT。',
        '- 主比较为同一服务器处理图、同两资产预算和同区域问题的 ROI 与 duplicate 配对；阈值固定为 p≥0.5。',
        '- 全部120张是已查看开发集；Teacher训练暴露按逐图split披露，未见子集也不是新的独立盲测。',
        '- 历史0/1/4-shot是全图判断，渲染器、问题语义和输入预算存在差异，只作历史门控对照。',
        '- 新服务器图与历史Jev图均非像素完全相同，不能把跨阶段分数变化归因于裁片；Qwen原生max_pixels524288也不同于Jev448。',
        '- ROI概率的目标是候选区域异常，不能直接解释为整图概率校准，故不报告对应Brier/ECE。', '',
        '## 完整120张开发集', '',
        '| 门控 | 全60异常平均IoU | 全60异常IoU≥0.5 | 异常保留召回 | 全60正常出框率 |',
        '|---|---:|---:|---:|---:|']
    for name, package_value in result['groups']['all120']['methods'].items():
        value=package_value['pooled']
        lines.append(f"| {name} | {fmt(value[METRICS[0]])} | {fmt(value[METRICS[1]],True)} | {fmt(value[METRICS[2]],True)} | {fmt(value[METRICS[3]],True)} |")
    lines+=['', '## 配对主差异：ROI − duplicate', '',
        '| 指标 | 差值 | 配对bootstrap 95%区间 |', '|---|---:|---:|']
    comparison=result['groups']['all120']['roi_minus_duplicate']['pooled']
    for name,value in comparison.items():
        ci=value['ci95']
        interval='NA' if ci is None else f"[{fmt(ci[0])}, {fmt(ci[1])}]"
        lines.append(f"| {name} | {fmt(value['delta'])} | {interval} |")
    lines+=['', '差值采用原单位；率指标乘100后为百分点。固定类别×图像标签分层、2000次、seed42；未做多重比较校正。', '',
        '## 分组与退化候选', '',
        '| 分组 | 图数 | 异常 | 正常 | ROI平均IoU | duplicate平均IoU | ROI正常出框率 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,group in result['groups'].items():
        r=group['methods']['roi']['pooled']; d=group['methods']['duplicate']['pooled']
        lines.append(f"| {name} | {r['n']} | {r['n_anomaly']} | {r['n_normal']} | {fmt(r[METRICS[0]])} | {fmt(d[METRICS[0]])} | {fmt(r[METRICS[3]],True)} |")
    lines+=['',
        'Teacher未见子集只有少量正常图时，误报率估计非常不稳定；配对bootstrap不能创造额外正常样本。',
        '各组正常出框率与异常召回的描述性Wilson95区间保存在JSON；分层bootstrap若仅有单样本stratum可能退化，不能据此声称精确。',
        'full_image_candidate与query_crop_bytes_identical分别报告；后者两臂像素相同，其差异用于检查服务随机性，不能宣称裁剪带来收益。',
        '详细逐类、宏平均、各分组配对区间、候选面积和逐图错误归因见05_evaluation.json。', '',
        '## 结果分析', '']
    main=result['groups']['all120']['methods']
    r,d=main['roi']['pooled'],main['duplicate']['pooled']
    lines.append(f"ROI相对duplicate：全异常平均IoU变化 {fmt(r[METRICS[0]]-d[METRICS[0]])}，异常保留召回变化 {fmt(r[METRICS[2]]-d[METRICS[2]],True)}，正常出框率变化 {fmt(r[METRICS[3]]-d[METRICS[3]],True)}。应共同判断定位收益、漏检和误报的取舍。")
    for name,value in comparison.items():
        interval=value['ci95']
        if interval is not None:
            assessment='区间跨零，当前开发集证据不足以排除零差异' if interval[0]<=0<=interval[1] else '区间未跨零，但仍仅覆盖该固定开发设置'
            lines.append(f'- {name}：{assessment}。')
    lines+=['', '## 执行与成本', '',
        f"接受判断 {result['operations']['accepted']} 个；started {result['operations']['started']} 次；显式失败 {result['operations']['explicit_failed']} 次。",
        '仅累加实际返回usage，失败或缺失usage保留未知；不估造货币费用。请求时延不是模型固有速度比较。', '',
        '## 下一阶段依据', '',
        '先审查nonfullbox的配对净收益及完整召回/误报权衡，再冻结工具路由和预算协议。',
        '若未证明ROI净收益，保留该负结果，不通过调阈值或挑子集改写当前结论。校准/拒答须另立独立数据协议，confirmation仍保持隔离。', '',
        f"Qwen同source对照：{result['qwen_scorer_control']['status']}；使用默认严格log_odds>0，不使用同120张调过的FPR5阈值。", '',
        '## 完整性记录', '',
        f"- 正式预测SHA256：`{result['prediction_freeze']['files_sha256']['formal_predictions']}`",
        f"- GT目标记录SHA256：`{result['gt_targets_sha256']}`",
        f"- 评价源码SHA256：`{result['script_sha256']}`",
        '- confirmation未读取；GT仅用于冻结后的评价，不进入候选、请求或阈值选择。', '']
    return '\n'.join(lines)


def evaluate(args):
    require(socket.gethostname()==HOSTNAME, 'Evaluation must execute on designated AutoDL server')
    root=args.root.resolve()
    output=api.inside(root,args.run)
    with api.output_lock(output):
        status,contract,completed,starts,attempts=validate_formal(root,output)
        frozen=freeze_inputs(root,output,contract,status)
        by_sid,labels,originals=validate_sources(root,contract)
        candidates_document=read(root/contract['source_candidates'])
        candidates={r['source_id']:r for r in candidates_document['records']}
        require(set(candidates)==set(by_sid), 'Candidate source set differs')
        rows=load_targets(root,by_sid,labels,originals,candidates)
        historical=historical_predictions(root,rows)
        gates={'unconditional_teacher':[True]*len(rows)}
        for shot in (0,1,4):
            gates[f'historical_jev_{shot}shot']=[historical[r['source_id'],shot]['probability']>=.5 for r in rows]
        for arm in ('roi','duplicate'):
            gates[arm]=[completed[r['source_id'],arm]['probability']>=.5 for r in rows]
        qwen_gates,qwen_summary=qwen_predictions(root,rows)
        gates.update(qwen_gates)
        partitions=dict(all120=list(range(len(rows))),
            teacher_unseen=[i for i,r in enumerate(rows) if not r['teacher_training_exposed']],
            teacher_training_exposed=[i for i,r in enumerate(rows) if r['teacher_training_exposed']],
            full_image_candidate=[i for i,r in enumerate(rows) if r['full_image_candidate']],
            nonfull_image_candidate=[i for i,r in enumerate(rows) if not r['full_image_candidate']],
            query_crop_bytes_identical=[i for i,r in enumerate(rows) if r['query_crop_bytes_identical']],
            query_crop_bytes_different=[i for i,r in enumerate(rows) if not r['query_crop_bytes_identical']])
        groups={}
        for name,indices in partitions.items():
            groups[name]=dict(definition=name,methods={method:package(rows,keep,indices) for method,keep in gates.items()},
                roi_minus_duplicate=bootstrap_difference(rows,gates['roi'],gates['duplicate'],indices),
                candidate_and_repeat_diagnostics=candidate_diagnostics(rows,completed,indices))
        for index,row in enumerate(rows):
            row['gates']={method:bool(keep[index]) for method,keep in gates.items()}
            row['probabilities']={arm:completed[row['source_id'],arm]['probability'] for arm in ('roi','duplicate')}
            row['historical_fullimage_probabilities']={str(shot):historical[row['source_id'],shot]['probability'] for shot in (0,1,4)}
        targets=[dict(source_id=r['source_id'],label=r['label'],bbox=r['gt_bbox_original_norm'],
                      ground_truth=r['ground_truth']) for r in rows]
        gt_sha=digest(canonical(targets))
        result_path=root/'stages/05_evaluation.json'
        if result_path.exists():
            previous=read(result_path)
            require(previous.get('prediction_freeze')==frozen and previous.get('gt_targets_sha256')==gt_sha
                    and previous.get('script_sha256')==sha_file(__file__), 'Existing evaluation inputs changed')
            require(canonical(previous['groups'])==canonical(groups), 'Recomputed metrics differ')
            result=previous
        else:
            result=dict(schema=SCHEMA,status='complete',timestamp_utc=now(),hostname=socket.gethostname(),
                script_sha256=sha_file(__file__),prediction_freeze=frozen,gt_targets_sha256=gt_sha,
                versions={k:importlib.metadata.version(k) for k in ('numpy','Pillow')},
                bbox_geometry='float original-image normalized xyxy; exclusive upper pixel boundaries; no1000 rounding',
                threshold=.5,threshold_rule='p>=0.5 retains candidate; No gives zero localization on every anomaly',
                denominator_policy='Full fixed group anomaly/normal denominators, no accepted-only primary metrics',
                teacher_exposure_counts={s:dict(Counter(str(r['label']) for r in rows if r['teacher_split']==s)) for s in ('train','val','test')},
                historical_control_note='Old Jev probabilities gate new fixed boxes as historical full-image decisions; renderer/prompt/information budgets differ; no causal crop attribution',
                qwen_scorer_control=qwen_summary,
                probability_calibration_note='Regional question cannot use whole-image label as regional calibrated target; Brier/ECE omitted',
                operations=operation_summary(completed,starts,attempts),groups=groups,records=rows,
                gt_masks_opened=60,confirmation_opened=False,no_gt_tuning=True,
                limitations=['Viewed120 development queries; not blind testing',
                    'Teacher custom supervised split overlaps original native validation split',
                    'Teacher-unseen normal denominator may be tiny; FPR uncertain',
                    'Union95 boxes may degenerate to full image; retained as frozen rule',
                    'Bootstrap fixed categories and candidates; excludes repeated service randomness',
                    'Historical full-image gates differ in renderer/prompt/information budget',
                    'No known monetary bill; missing usage remains unknown'])
            publish(result_path,result)
        report=report_document(result)
        report_path=root/'REPORT.md'
        if report_path.exists():
            require(report_path.read_text(encoding='utf-8')==report,'Existing report differs')
        else:
            with report_path.open('x',encoding='utf-8') as stream:
                stream.write(report)
                stream.flush()
                os.fsync(stream.fileno())
        require(sha_file(output/'predictions.jsonl')==frozen['files_sha256']['formal_predictions'],
                'Predictions changed during evaluation')
        print(json.dumps(dict(status='complete',n=120,gt_masks_opened=60,
            evaluation=str(result_path),report=str(report_path),
            roi=result['groups']['all120']['methods']['roi']['pooled'],
            duplicate=result['groups']['all120']['methods']['duplicate']['pooled'],
            confirmation_opened=False),ensure_ascii=False),flush=True)


def self_test():
    require(socket.gethostname()==HOSTNAME,'Self-tests must run on designated server')
    mask=np.zeros((10,20),dtype=np.uint8);mask[2:4,3:8]=7
    require(mask_box(mask)==[.15,.2,.4,.4],'Mask boundaries differ')
    require(box_iou([0,0,1,1],[0,0,1,1])==1 and box_iou(None,[0,0,1,1])==0,'IoU semantics differ')
    try:
        mask_box(np.zeros((2,2),dtype=np.uint8))
    except ValueError:
        pass
    else:
        raise AssertionError('Empty anomaly mask accepted')
    rows=[dict(category='a',label=1,candidate_iou=1.),dict(category='a',label=1,candidate_iou=.2),
          dict(category='a',label=0,candidate_iou=0.),dict(category='a',label=0,candidate_iou=0.)]
    a=[True,False,True,False];b=[True,True,False,False]
    value=metrics(rows,a)
    require(value['anomaly_bbox_iou_mean_all']==.5 and value['anomaly_retained_recall']==.5
            and value['normal_positive_box_rate']==.5 and value['FN']==1,'Full-denominator metric differs')
    require(metrics(rows,[False]*4)['anomaly_bbox_iou_mean_all']==0,'No did not count zero')
    first=bootstrap_difference(rows,a,b,replicates=200)
    second=bootstrap_difference(rows,a,b,replicates=200)
    require(canonical(first)==canonical(second),'Bootstrap not deterministic')
    require(first['pooled']['normal_positive_box_rate']['delta']==.5,'Paired FPR point difference differs')
    require(metrics(rows,a,[0])['normal_positive_box_rate'] is None,'Missing denominator not null')
    interval=wilson_interval(0,2)
    require(interval[0]==0 and interval[1]>.65,'Tiny normal denominator uncertainty hidden')
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'freeze.json'
        publish(path,{'v':1});publish(path,{'v':1})
        try:
            publish(path,{'v':2})
        except ValueError:
            pass
        else:
            raise AssertionError('Frozen artifact overwritten')
    print(json.dumps(dict(status='self_test_passed',hostname=socket.gethostname(),
        tests=['mask_geometry','empty_mask_rejection','iou_null','all_anomaly_denominator',
               'normal_denominator','deterministic_paired_bootstrap','immutable_publish'],
        models_called=False,api_calls=0,formal_data_opened=False)),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path)
    parser.add_argument('--run',default='runs/modelgo_roi')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        require(args.root is not None,'--root is required')
        evaluate(args)


if __name__=='__main__':
    main()
