#!/usr/bin/env python3
"""Server-only whole-image OOF calibration and rejection diagnostics.

prepare freezes method, sources and shared source-level folds before fitting.
evaluate requires completed stage05 and never opens an RGB or GT mask.
The reused development120 is not an independent calibration/test set.

  python calibration_diagnostics.py prepare --root RUN_ROOT
  python calibration_diagnostics.py evaluate --root RUN_ROOT
  python calibration_diagnostics.py self-test
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HOSTNAME = 'autodl-container-132444827f-9c594038'
SCHEMA = 'jev-wholeimage-development-oof-calibration-v1'
CLIP = 1e-6
FOLDS = 5
SEED = 42
COVERAGES = (.5, .75, 1.)
TIE_PREFIX = 'jev-calibration-reject-v1'
QWEN_SHA = {
    'base': 'dc109510af9e58701257ef943e05cee4b100f46df80a14620b36fdfb64f96499',
    'decision_decision': 'd19c70935d3369d400a921d5a9281c6382566c327667b9b3f6de0bb3a7b4681c',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    raw = Path(path).read_bytes()
    require(not raw or raw.endswith(b'\n'), 'Incomplete JSONL tail')
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def publish(path, value):
    path = Path(path)
    if path.exists():
        require(canonical(read(path)) == canonical(value), 'Existing frozen artifact differs: ' + path.name)
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


@contextmanager
def lock(root):
    directory = root / 'stages/calibration_lock'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def identities(root):
    base = root / 'source_baseline/frozen'
    plan, labels, config = (read(base / (name + '.json')) for name in ('plan', 'labels', 'config'))
    require(sha_bytes(canonical(plan)) == config['plan_sha256'], 'Frozen plan hash differs')
    require(sha_bytes(canonical(labels)) == config['labels_sha256'], 'Frozen label hash differs')
    queries = plan.get('queries', [])
    require(len(queries) == len({row['source_id'] for row in queries}) == 120, 'Need120 unique query sources')
    require(set(labels) == {row['source_id'] for row in queries}, 'Frozen labels/source set differs')
    rows = []
    for source in sorted(queries, key=lambda row: row['source_id']):
        sid = source['source_id']
        require(sid == 'visa_' + source['sha256'], 'Source identity/hash differs')
        require(type(labels[sid]) is int and labels[sid] in (0, 1), 'Label must be binary integer')
        rows.append(dict(source_id=sid, category=source['category'], label=labels[sid],
                         original_sha256=source['sha256']))
    counts = Counter((row['category'], row['label']) for row in rows)
    categories = sorted({row['category'] for row in rows})
    require(len(categories) == 12 and len(counts) == 24
            and all(counts[category, label] == 5 for category in categories for label in (0, 1)),
            'Need exactly24 category-label strata, each with5 sources')
    return rows


def make_folds(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['category'], row['label']), []).append(row['source_id'])
    require(len(groups) == 24 and all(len(members) == FOLDS for members in groups.values()),
            'Five-fold protocol requires five sources per category-label stratum')
    rng = np.random.default_rng(SEED)
    assignment = {}
    for group in sorted(groups):
        members = sorted(groups[group])
        permuted = rng.permutation(len(members))
        for fold, position in enumerate(permuted):
            assignment[members[int(position)]] = fold
    require(len(assignment) == len(rows) == 120 and set(assignment.values()) == set(range(FOLDS)),
            'Fold source partition incomplete')
    result = [dict(**row, fold=assignment[row['source_id']]) for row in rows]
    for fold in range(FOLDS):
        held = [row for row in result if row['fold'] == fold]
        require(len(held) == 24 and sum(row['label'] for row in held) == 12, 'Fold balance differs')
        require(len({(row['category'], row['label']) for row in held}) == 24,
                'Fold does not hold one source per category-label stratum')
    return result


def dependencies():
    return dict(python=platform.python_version(),
        packages={name: importlib.metadata.version(name) for name in ('numpy','scipy','scikit-learn')})


def protocol_body(root):
    rows = identities(root)
    paths = {
        'source_plan': root / 'source_baseline/frozen/plan.json',
        'source_labels': root / 'source_baseline/frozen/labels.json',
        'source_config': root / 'source_baseline/frozen/config.json',
        'wholeimage_jev_predictions': root / 'source_baseline/runs/modelgo_448/predictions.jsonl',
        'code': Path(__file__),
    }
    qwen_files = {name: root / 'source_qwen' / (name + '_scores.jsonl') for name in QWEN_SHA}
    models = ['jev_0shot','jev_1shot','jev_4shot']
    if any(path.exists() for path in qwen_files.values()):
        require(all(path.exists() for path in qwen_files.values()), 'Both audited Qwen score files required')
        for name, path in qwen_files.items():
            require(sha_file(path) == QWEN_SHA[name], 'Qwen score source hash differs')
            paths['qwen_' + name] = path
            models.append('qwen_' + name)
    folds = make_folds(rows)
    return dict(schema=SCHEMA, models=models, files_sha256={name:sha_file(path) for name,path in paths.items()},
        dependencies=dependencies(), source_count=120, source_unit='one original-image source; same source never crosses train/heldout',
        data_role='Reused previously viewed120 development queries; not independent calibration/test data',
        target='whole-image visible industrial anomaly label, not candidate-region label',
        allowed_label_access='Only source_baseline frozen120 labels for predefined folds and evaluation; no GT masks',
        crossfit=dict(folds=FOLDS,seed=SEED,strata='category x image_label',
            assignment='Sorted strata, sorted source_id; numpy default_rng42 permutation within each five-source stratum; assignfold0..4',
            calibration_train_folds=4, calibration_test_folds=1, heldout_per_fold=24,
            fit_scope='Single scalar calibrator per model on96 sources; no category-specific fit',
            shared_folds_across_models=True, outer_hyperparameter_search=False,
            refit_on_all120=False, fold_records=folds, fold_map_sha256=sha_bytes(canonical(folds))),
        logistic=dict(feature='log(p_clip/(1-p_clip))',clip=CLIP,C=1.0,penalty='l2',
            solver='lbfgs',fit_intercept=True,max_iter=1000,tol=1e-8,class_weight=None,
            random_state=SEED,convergence_warning_action='fail',
            input_note='Qwen sigmoid preference scores are not independently calibrated probabilities'),
        metric_policy=dict(metrics=['Brier','binary_log_loss','ECE10_equal_width','AUROC'],
            brier='mean((p-y)^2) on raw or heldout probability without extra clipping',
            logloss='binary logloss after fixed clip1e-6 on raw or heldout probability',
            ece='10 bins [b/10,(b+1)/10), last includes1; sum n_bin/n * abs(mean_p-mean_y)',
            auroc='sklearn roc_auc_score on raw or OOF heldout probability; tied scores retained',
            uncertainty='Descriptive fixed-fold diagnostics; no confidence interval claiming repeated-calibration uncertainty'),
        rejection=dict(confidence='max(p,1-p)',coverages=list(COVERAGES),
            ranking='descending confidence, then sha256(prefix|source_id), then source_id',
            tie_prefix=TIE_PREFIX, ranking_uses_label=False,
            accepted_count='round(coverage *120), exactly60/90/120',prediction='p>=0.5',
            metrics=['selected_accuracy','selective_risk','selected_class_counts','selected_FPR','selected_recall',
                     'full_anomaly_automatic_detection_rate','class_specific_acceptance_coverage'],
            abstention='Unselected cases abstain; not imputed negative; report class-specific selected denominators',
            no_threshold_or_coverage_tuning=True),
        evaluation_gate='stage05_evaluation.status==complete and source set identical; stage07 protocol pre-frozen',
        independent_calibration_task_completed=False, confirmation_task_completed=False,
        model_training='Only one-dimensional logistic calibration fit; no foundation-model/LoRA training',
        no_api_calls=True, no_new_predictions=True, no_rgb_or_masks_opened=True, confirmation_opened=False)


def prepare(args):
    require(socket.gethostname() == HOSTNAME, 'Prepare must execute on designated server')
    root = args.root.resolve()
    with lock(root):
        body = protocol_body(root)
        path = root / 'stages/07_calibration_protocol.json'
        if path.exists():
            document = read(path)
            require(document.get('protocol') == body, 'Frozen calibration protocol differs')
        else:
            require(not (root/'stages/07_calibration_results.json').exists(),
                    'Calibration results already exist without this frozen protocol')
            stage05_path=root/'stages/05_evaluation.json'
            stage05_available=stage05_path.exists()
            document = dict(schema=SCHEMA,status='frozen',timestamp_utc=now(),hostname=socket.gethostname(),
                protocol_sha256=sha_bytes(canonical(body)),probabilities_evaluated=False,
                approved120_labels_read_for_strata=True,ground_truth_masks_opened=False,
                candidate_evaluation_available_at_prepare=stage05_available,
                stage05_sha256_at_prepare=sha_file(stage05_path) if stage05_available else None,
                freeze_timing='Before any calibration fitting or calibration/rejection metric computation; stage05 availability recorded explicitly',
                protocol=body)
            publish(path,document)
        print(json.dumps(dict(status='frozen',path=str(path),
            protocol_sha256=document['protocol_sha256'],folds=FOLDS,models=body['models'],
            candidate_evaluation_available_at_prepare=document['candidate_evaluation_available_at_prepare'],
            independent_calibration_task_completed=False,confirmation_opened=False)),flush=True)
    return 0


def valid_probability(value):
    return type(value) in (float,int) and math.isfinite(value) and 0 <= value <= 1


def load_probabilities(root, folds, models):
    expected = {row['source_id']:row for row in folds}
    historical = read_rows(root/'source_baseline/runs/modelgo_448/predictions.jsonl')
    found = {}
    for row in historical:
        pair = row.get('source_id'),row.get('shots')
        require(pair[0] in expected and pair[1] in (0,1,4) and pair not in found,
                'Unexpected or duplicate wholeimage Jev target')
        require(row.get('category') == expected[pair[0]]['category']
                and row.get('response_model') == 'jev-1.14.0'
                and valid_probability(row.get('probability')), 'Jev metadata/probability invalid')
        found[pair] = row
    require(len(found) == len(historical) == 360
            and set(found) == {(sid,shot) for sid in expected for shot in (0,1,4)}, 'Jev360 not complete')
    require(len({row.get('contract_sha256') for row in historical}) == 1, 'Historical Jev contracts mixed')
    probabilities = {'jev_'+str(shot)+'shot':np.array([found[row['source_id'],shot]['probability'] for row in folds],dtype=float)
                     for shot in (0,1,4)}
    for name in QWEN_SHA:
        if 'qwen_'+name not in models:
            continue
        path = root/'source_qwen'/(name+'_scores.jsonl')
        require(sha_file(path) == QWEN_SHA[name], 'Qwen file hash differs')
        rows = read_rows(path)
        values = {}
        for value in rows:
            sid=value.get('source_id')
            require(sid in expected and sid not in values,'Unexpected/duplicate Qwen source')
            source = expected[sid]
            require(value.get('image_sha256') == source['original_sha256']
                    and value.get('category') == source['category']
                    and type(value.get('label')) is bool and int(value['label']) == source['label']
                    and valid_probability(value.get('anomaly_score')), 'Qwen identity/label/score differs')
            odds = value.get('anomaly_log_odds')
            require(type(odds) in (float,int) and math.isfinite(odds), 'Invalid Qwen log odds')
            require(math.isclose(float(value['anomaly_score']),float(np.exp(-np.logaddexp(0.,-odds))),
                                 rel_tol=1e-10,abs_tol=1e-12),'Qwen sigmoid/logodds mismatch')
            values[sid] = value['anomaly_score']
        require(len(values) == len(rows) == 120 and set(values) == set(expected), 'Qwen120 incomplete')
        probabilities['qwen_'+name] = np.array([values[row['source_id']] for row in folds],dtype=float)
    require(set(probabilities) == set(models), 'Protocol/model scores differ')
    return probabilities


def feature(values):
    values=np.asarray(values,dtype=float)
    require(values.ndim == 1 and np.isfinite(values).all() and (values>=0).all() and (values<=1).all(),
            'Invalid scalar probability feature')
    clipped=np.clip(values,CLIP,1-CLIP)
    return (np.log(clipped)-np.log1p(-clipped)).reshape(-1,1)


def fit_oof(raw, labels, folds):
    raw,labels=np.asarray(raw,dtype=float),np.asarray(labels,dtype=int)
    require(len(raw)==len(labels)==len(folds)==120,'OOF needs aligned120')
    x=feature(raw)
    assignments=np.array([row['fold'] for row in folds],dtype=int)
    output=np.full(120,np.nan,dtype=float)
    fit_records=[]
    for fold in range(FOLDS):
        train=np.flatnonzero(assignments!=fold)
        test=np.flatnonzero(assignments==fold)
        require(len(train)==96 and len(test)==24 and len(set(labels[train]))==2,
                'OOF fold class counts invalid')
        train_ids=[folds[i]['source_id'] for i in train]
        test_ids=[folds[i]['source_id'] for i in test]
        require(not(set(train_ids)&set(test_ids)),'Calibration source leakage')
        model=LogisticRegression(C=1.,penalty='l2',solver='lbfgs',fit_intercept=True,
            max_iter=1000,tol=1e-8,class_weight=None,random_state=SEED)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model.fit(x[train],labels[train])
        require(list(model.classes_)==[0,1], 'Calibrator classes differ')
        probability=model.predict_proba(x[test])[:,1]
        require(np.isfinite(probability).all() and (probability>=0).all() and (probability<=1).all(),
                'Invalid heldout calibrator output')
        output[test]=probability
        fit_records.append(dict(fold=fold,n_train=96,n_heldout=24,
            train_anomaly_count=int(labels[train].sum()),heldout_anomaly_count=int(labels[test].sum()),
            train_source_ids_sha256=sha_bytes(canonical(train_ids)),
            heldout_source_ids_sha256=sha_bytes(canonical(test_ids)),
            coefficient=float(model.coef_[0,0]),intercept=float(model.intercept_[0]),
            iterations=[int(v) for v in model.n_iter_],source_intersection=0))
    require(np.isfinite(output).all(),'Missing OOF probabilities')
    return output,fit_records


def metrics(labels,probabilities):
    y,p=np.asarray(labels,dtype=int),np.asarray(probabilities,dtype=float)
    require(len(y)==len(p)>0 and set(y.tolist()) <= {0,1},'Invalid metric input')
    require(np.isfinite(p).all() and (p>=0).all() and (p<=1).all(),'Invalid metric probability')
    clipped=np.clip(p,CLIP,1-CLIP)
    bins=np.minimum((p*10).astype(int),9)
    bin_records=[]
    for b in range(10):
        selected=bins==b
        n=int(selected.sum())
        mean_p=float(p[selected].mean()) if n else None
        mean_y=float(y[selected].mean()) if n else None
        bin_records.append(dict(bin=b,lower=b/10,upper=(b+1)/10,includes_upper=b==9,
            n=n,n_anomaly=int(y[selected].sum()),mean_probability=mean_p,
            anomaly_fraction=mean_y,absolute_gap=abs(mean_p-mean_y) if n else None))
    ece=sum(row['n']/len(y)*row['absolute_gap'] for row in bin_records if row['n'])
    positive=p>=.5
    normal=y==0;anomaly=y==1
    return dict(n=len(y),n_anomaly=int(anomaly.sum()),n_normal=int(normal.sum()),
        brier=float(np.mean((p-y)**2)),
        binary_log_loss=float(np.mean(-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)))),
        ece_10_equal_width=float(ece),
        auroc=float(roc_auc_score(y,p)) if len(set(y.tolist()))==2 else None,
        threshold=.5,TP=int(np.sum(positive&anomaly)),FN=int(np.sum(~positive&anomaly)),
        FP=int(np.sum(positive&normal)),TN=int(np.sum(~positive&normal)),
        accuracy=float(np.mean(positive==y)),calibration_bins=bin_records)


def selection_order(probabilities,source_ids):
    confidence=np.maximum(probabilities,1-probabilities)
    require(len(set(source_ids))==len(source_ids),'Duplicate source in rejection ranking')
    return sorted(range(len(source_ids)),key=lambda i:(-float(confidence[i]),
        sha_bytes((TIE_PREFIX+'|'+source_ids[i]).encode()),source_ids[i]))


def selective_metrics(labels,probabilities,source_ids):
    y,p=np.asarray(labels,dtype=int),np.asarray(probabilities,dtype=float)
    order=selection_order(p,source_ids)
    positive=p>=.5
    total_anomaly=int((y==1).sum());total_normal=int((y==0).sum())
    result={}
    for coverage in COVERAGES:
        count=int(round(coverage*len(y)))
        selected=np.array(order[:count],dtype=int)
        rejected=np.array(order[count:],dtype=int)
        truth=y[selected];decision=positive[selected]
        normal=truth==0;anomaly=truth==1
        n_normal=int(normal.sum());n_anomaly=int(anomaly.sum())
        tp=int(np.sum(decision&anomaly));fp=int(np.sum(decision&normal))
        accuracy=float(np.mean(decision==truth))
        result[str(coverage)]=dict(target_coverage=coverage,n_accepted=count,n_abstained=len(y)-count,
            realized_coverage=count/len(y),selected_accuracy=accuracy,selective_risk=1-accuracy,
            selected_anomaly=n_anomaly,selected_normal=n_normal,
            accepted_source_ids=[source_ids[i] for i in selected],
            abstained_source_ids=[source_ids[i] for i in rejected],
            TP=tp,FN=n_anomaly-tp,FP=fp,TN=n_normal-fp,
            selected_FPR=fp/n_normal if n_normal else None,selected_recall=tp/n_anomaly if n_anomaly else None,
            full_anomaly_automatic_detection_rate=tp/total_anomaly if total_anomaly else None,
            full_normal_automatic_false_positive_rate=fp/total_normal if total_normal else None,
            anomaly_acceptance_coverage=n_anomaly/total_anomaly if total_anomaly else None,
            normal_acceptance_coverage=n_normal/total_normal if total_normal else None,
            selected_confidence_minimum=float(np.min(np.maximum(p[selected],1-p[selected]))),
            label_access_in_ranking=False,abstention_note='Unselected cases abstain; not assigned No')
    return result


def metric_package(rows,labels,probabilities):
    categories=sorted({row['category'] for row in rows})
    by_category={}
    for category in categories:
        indices=np.array([i for i,row in enumerate(rows) if row['category']==category],dtype=int)
        by_category[category]=metrics(labels[indices],probabilities[indices])
    macro={}
    for name in ('brier','binary_log_loss','ece_10_equal_width','auroc'):
        values=[row[name] for row in by_category.values() if row[name] is not None]
        macro[name]=float(np.mean(values)) if values else None
    return dict(pooled=metrics(labels,probabilities),category_macro=macro,by_category=by_category,
        rejection=selective_metrics(labels,probabilities,[row['source_id'] for row in rows]))


def report_document(result):
    lines=['# 全图判断分数的交叉拟合校准与拒答诊断','',
        f"服务器：`{result['hostname']}`；时间（UTC）：{result['timestamp_utc']}。",'',
        '**这是重用120张已查看开发图的固定5折诊断，不是独立校准集、盲测或生产概率保证。**',
        '没有完成任务清单中的“使用独立数据校准/新的留出确认”；confirmation继续未读。','',
        '## 固定方法','',
        '- 按类别×图像标签分层，每个stratum5来源，seed42分配5fold；每折24图，4折96图拟合、1折24图评价。',
        '- 所有Jev 0/1/4-shot和Qwen base/DD共用同一来源分折，每图恰有一个未参与该校准器拟合的OOF预测。',
        '- 固定clip1e-6的logit特征，LogisticRegression C1/L2/lbfgs，不搜索超参数，不用全120重拟合后自评。',
        '- 仅评价第一阶段全图Noul和Qwen全图候选偏好分数，不对候选区域Noul套用整图校准标签。',
        '- Qwen原分数是候选logodds的sigmoid，称为偏好分数，不是独立校准概率；模型输入与Jev不同，跨模型表不是同输入消融。',
        f"- 协议冻结时stage05候选评价已存在：{result['candidate_evaluation_available_at_prepare']}。协议在本校准阶段任何正式分数评价和拟合前冻结；不声称先于候选实验结果预注册。",'',
        '## 全部120图指标','',
        '| 分数 | 版本 | Brier↓ | LogLoss↓ | ECE10↓ | AUROC↑ |',
        '|---|---|---:|---:|---:|---:|']
    for name,model in result['models'].items():
        for variant in ('raw','oof_logistic'):
            value=model[variant]['pooled']
            lines.append(f"| {name} | {variant} | {value['brier']:.6f} | {value['binary_log_loss']:.6f} | {value['ece_10_equal_width']:.6f} | {value['auroc']*100:.2f}% |")
    lines+=['','ECE10在120图上对分箱敏感；AUROC可能因跨折校准尺度和有限样本变化，不能把校准当成提高基座判别能力的证据。','',
        '## 固定覆盖率拒答','',
        '排序只用confidence=max(p,1−p)，同分按source哈希打破；不使用待评价图标签选择接收样本。',
        '未接收图计为abstain，不计作正常或自动检测成功。风险为接收样本中的错误比例。','',
        '| 分数 | 版本 | 覆盖率 | 接收图数 | 接收异常/正常 | 接收准确率 | Selective Risk | 全异常自动检出率 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,model in result['models'].items():
        for variant in ('raw','oof_logistic'):
            for coverage in COVERAGES:
                value=model[variant]['rejection'][str(coverage)]
                recall=value['full_anomaly_automatic_detection_rate']
                lines.append(f"| {name} | {variant} | {coverage*100:.0f}% | {value['n_accepted']} | {value['selected_anomaly']}/{value['selected_normal']} | {value['selected_accuracy']*100:.2f}% | {value['selective_risk']*100:.2f}% | {recall*100:.2f}% |")
    lines+=['','各覆盖率所接收的异常/正常分母不同，不能只根据接收准确率上升声称系统召回提高；完整接收ID和类条件覆盖率见JSON。','',
        '## 结果分析','']
    for name,model in result['models'].items():
        delta=model['oof_minus_raw']
        direction='降低' if delta['brier']<0 else ('提高' if delta['brier']>0 else '保持')
        lines.append(f"- {name}：OOF相对raw的Brier {direction}，差值 {delta['brier']:+.6f}；LogLoss差值 {delta['binary_log_loss']:+.6f}；ECE10差值 {delta['ece_10_equal_width']:+.6f}。这些是固定开发图和分折下的诊断。")
    lines+=['','## 边界与下一步','',
        '- 标签已参与开发与分层，交叉拟合只阻止本图用于自身校准器拟合，不能撤销早前选题、模型或prompt开发暴露。',
        '- 每类正常/异常各5张，固定5折的校准参数可能不稳定；未测新的支持选择、模型服务重复随机性或生产分布。',
        '- 不根据本表重新挑阈值、coverage或C；本协议不发布生产校准器或用全120训练最终版本。',
        '- 后续独立校准和confirmation仍需单独冻结来源隔离及选型协议，不能将本阶段勾选为独立验证完成。',
        '- 本阶段无新增API调用、无基座/LoRA训练、未打开RGB/GT mask、未读取confirmation。','',
        '## 完整性','',
        f"- 协议SHA256：`{result['protocol_sha256']}`",
        f"- 来源分折SHA256：`{result['fold_map_sha256']}`",
        f"- stage05 SHA256：`{result['stage05_sha256']}`",
        f"- 脚本SHA256：`{result['script_sha256']}`",'']
    return '\n'.join(lines)


def evaluate(args):
    require(socket.gethostname()==HOSTNAME,'Evaluate must execute on designated server')
    root=args.root.resolve()
    with lock(root):
        protocol_path=root/'stages/07_calibration_protocol.json'
        frozen=read(protocol_path)
        require(frozen.get('status')=='frozen' and frozen.get('schema')==SCHEMA
                and frozen.get('probabilities_evaluated') is False,'Pre-fit frozen stage07 protocol required')
        body=protocol_body(root)
        require(canonical(body)==canonical(frozen['protocol'])
                and sha_bytes(canonical(body))==frozen['protocol_sha256'],'Code, dependencies, source or fold protocol changed')
        stage05_path=root/'stages/05_evaluation.json'
        stage05=read(stage05_path)
        require(stage05.get('status')=='complete'
                and stage05.get('schema')=='jev-fixed-candidate-paired-evaluation-v1'
                and stage05.get('confirmation_opened') is False,'Completed isolated stage05 required')
        if frozen.get('candidate_evaluation_available_at_prepare'):
            require(frozen.get('stage05_sha256_at_prepare')==sha_file(stage05_path),
                    'stage05 changed after calibration protocol freeze')
        else:
            require(datetime.fromisoformat(frozen['timestamp_utc']) <= datetime.fromisoformat(stage05['timestamp_utc']),
                    'Recorded stage05 availability/timing inconsistent')
        folds=body['crossfit']['fold_records']
        require(len(stage05.get('records',[]))==120
                and {row['source_id'] for row in stage05['records']}=={row['source_id'] for row in folds},
                'stage05 source set differs')
        prediction_freeze=read(root/'stages/04_prediction_freeze.json')
        require(stage05.get('prediction_freeze')==prediction_freeze,'stage05 prediction freeze differs')
        formal_predictions=root/prediction_freeze['run']/'predictions.jsonl'
        require(sha_file(formal_predictions)==prediction_freeze['files_sha256']['formal_predictions'],
                'Formal candidate predictions changed after stage05')
        probabilities=load_probabilities(root,folds,body['models'])
        labels=np.array([row['label'] for row in folds],dtype=int)
        models={}
        records=[dict(**row,predictions={}) for row in folds]
        for name in body['models']:
            raw=probabilities[name]
            calibrated,fit_records=fit_oof(raw,labels,folds)
            raw_result=metric_package(folds,labels,raw)
            oof_result=metric_package(folds,labels,calibrated)
            models[name]=dict(raw=raw_result,oof_logistic=oof_result,calibrator_folds=fit_records,
                oof_minus_raw={key:oof_result['pooled'][key]-raw_result['pooled'][key]
                    for key in ('brier','binary_log_loss','ece_10_equal_width','auroc')})
            for i,row in enumerate(records):
                row['predictions'][name]=dict(raw=float(raw[i]),oof_logistic=float(calibrated[i]),
                                              calibration_heldout_fold=row['fold'])
        result_path=root/'stages/07_calibration_results.json'
        if result_path.exists():
            result=read(result_path)
            require(result.get('protocol_sha256')==frozen['protocol_sha256']
                    and result.get('stage05_sha256')==sha_file(stage05_path)
                    and result.get('script_sha256')==sha_file(__file__)
                    and canonical(result['models'])==canonical(models)
                    and canonical(result['records'])==canonical(records),'Existing calibration results differ')
        else:
            result=dict(schema=SCHEMA,status='complete',timestamp_utc=now(),hostname=socket.gethostname(),
                protocol_file=protocol_path.relative_to(root).as_posix(),protocol_sha256=frozen['protocol_sha256'],
                stage05_sha256=sha_file(stage05_path),script_sha256=sha_file(__file__),
                files_sha256=body['files_sha256'],dependencies=dependencies(),
                fold_map_sha256=body['crossfit']['fold_map_sha256'],models=models,records=records,
                candidate_evaluation_available_at_prepare=frozen['candidate_evaluation_available_at_prepare'],
                protocol_freeze_timing=frozen['freeze_timing'],
                n_sources=120,n_folds=5,calibration_fit_count=5*len(models),
                wholeimage_labels_only=True,region_noul_probabilities_used=False,
                rgb_files_opened=0,gt_masks_opened=0,new_api_calls=0,confirmation_opened=False,
                source_is_independent_calibration=False,independent_calibration_task_completed=False,
                confirmation_task_completed=False,final_production_calibrator_fitted=False,
                interpretation='Fixed5fold heldout calibration/rejection diagnostic on previously viewed development120; no independent validity claim')
            publish(result_path,result)
        report=report_document(result)
        report_path=root/'CALIBRATION_REPORT.md'
        if report_path.exists():
            require(report_path.read_text(encoding='utf-8')==report,'Existing calibration report differs')
        else:
            with report_path.open('x',encoding='utf-8') as stream:
                stream.write(report);stream.flush();os.fsync(stream.fileno())
        print(json.dumps(dict(status='complete',n_sources=120,n_models=len(models),
            protocol_sha256=frozen['protocol_sha256'],results=str(result_path),report=str(report_path),
            delta_brier={name:value['oof_minus_raw']['brier'] for name,value in models.items()},
            independent_calibration_task_completed=False,confirmation_opened=False)),flush=True)
    return 0


def self_test():
    require(socket.gethostname()==HOSTNAME,'Self-tests must execute on designated server')
    rows=[]
    for c in range(12):
        for y in (0,1):
            for i in range(5):
                sid='visa_'+sha_bytes(f'fixture|{c}|{y}|{i}'.encode())
                rows.append(dict(source_id=sid,category=f'fixture{c}',label=y,original_sha256=sid[5:]))
    rows=sorted(rows,key=lambda row:row['source_id'])
    folds=make_folds(rows)
    require(canonical(folds)==canonical(make_folds(rows)),'Folds not deterministic')
    labels=np.array([row['label'] for row in folds],dtype=int)
    raw=np.array([.65 if row['label'] else .35 for row in folds],dtype=float)
    raw[0]=0;raw[1]=1
    require(np.isfinite(feature(raw)).all(),'Clipped logits not finite')
    oof,fit_records=fit_oof(raw,labels,folds)
    require(len(oof)==120 and len(fit_records)==5 and all(r['source_intersection']==0 for r in fit_records),
            'OOF partition incomplete')
    value=metrics(np.array([0,1]),np.array([0.,1.]))
    require(value['brier']==0 and value['auroc']==1 and value['ece_10_equal_width']==0
            and math.isfinite(value['binary_log_loss']),'Metric endpoints invalid')
    constant=np.full(120,.5)
    sources=[row['source_id'] for row in folds]
    order=selection_order(constant,sources)
    require(order==selection_order(constant,sources),'Tie selection not deterministic')
    selective=selective_metrics(labels,constant,sources)
    require([selective[str(c)]['n_accepted'] for c in COVERAGES]==[60,90,120],
            'Coverage accepted counts differ')
    require(selective['1.0']['n_abstained']==0 and selective['0.5']['n_abstained']==60,
            'Abstention accounting differs')
    require(selection_order(constant,sources)==order,'Rejection ranking depends on labels')
    try:
        feature(np.array([np.nan]))
    except ValueError:
        pass
    else:
        raise AssertionError('NaN probability accepted')
    print(json.dumps(dict(status='self_test_passed',hostname=socket.gethostname(),
        tests=['source_fold_balance','deterministic_folds','source_disjoint_oof','clipped_endpoints',
               'Brier_logloss_ECE_AUC','deterministic_ties','coverage_and_abstention','NaN_rejection'],
        formal_data_opened=False,new_api_calls=0,gt_masks_opened=0)),flush=True)
    return 0


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','evaluate'):
        item=sub.add_parser(name);item.add_argument('--root',required=True,type=Path)
    sub.add_parser('self-test')
    args=parser.parse_args()
    return {'prepare':prepare,'evaluate':evaluate,'self-test':lambda _:self_test()}[args.command](args)


if __name__=='__main__':
    raise SystemExit(main())
