"""Read-only analysis of stored single-view predictions; no model inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


def metrics(frame):
    y = frame.view_label.to_numpy(dtype=int)
    s = frame.p_yes.to_numpy(dtype=float)
    assert np.isfinite(s).all() and ((s >= 0) & (s <= 1)).all()
    pred = s >= 0.5
    precision, recall, thresholds = precision_recall_curve(y, s)
    f1s = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1],
                    out=np.zeros(len(thresholds)), where=(precision[:-1] + recall[:-1]) != 0)
    best = int(np.argmax(f1s))
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    small = (y == 1) & frame.defect_ratio.le(.01).to_numpy()
    return dict(
        n=len(y), n_normal=int((y == 0).sum()), n_anomaly=int((y == 1).sum()),
        accuracy=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        auroc=float(roc_auc_score(y, s)), aupr=float(average_precision_score(y, s)),
        f1max=float(f1s[best]), f1max_threshold=float(thresholds[best]),
        tn=tn, fp=fp, fn=fn, tp=tp, positive_rate=float(pred.mean()),
        normal_recall=tn / (tn + fp), score_std=float(s.std()),
        normal_score_mean=float(s[y == 0].mean()), anomaly_score_mean=float(s[y == 1].mean()),
        small_view_n=int(small.sum()), small_recall_at_05=float(pred[small].mean()),
    )


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])


def main():
    root = Path('artifacts/e1_real_classification_qwen3')
    out = root / 'single_view_analysis'
    out.mkdir(parents=True, exist_ok=True)
    dev_path = Path('artifacts/splits/fold1/dev.parquet')
    dev = pd.read_parquet(dev_path).set_index('image_path')
    inputs = {'baseline': Path('artifacts/baseline_retest/qwen3/P4_CLASSIFICATION_ONLY_views.csv')}
    inputs.update({p.stem.replace('_views', ''): p for p in sorted(root.glob('step*_views.csv'))})
    assert set(inputs) == {'baseline', 'step0250', 'step0500', 'step0750', 'step1000'}
    results = {}
    for tag, path in inputs.items():
        frame = pd.read_csv(path)
        assert len(frame) == 3600 and frame.image_path.is_unique
        assert set(frame.image_path) == set(dev.index)
        labels = dev.loc[frame.image_path, 'view_label'].to_numpy()
        assert np.array_equal(labels, frame.view_label.to_numpy())
        assert np.array_equal(dev.loc[frame.image_path, 'class_name'].to_numpy(), frame.class_name.to_numpy())
        detail = {cls: metrics(g) for cls, g in frame.groupby('class_name')}
        overall = metrics(frame)
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auroc', 'aupr', 'f1max']:
            overall['macro_' + metric] = float(np.mean([v[metric] for v in detail.values()]))
        results[tag] = dict(overall=overall, per_class=detail)
    history = pd.read_csv(root / 'training_history.csv')
    assert history.step.tolist() == list(range(1, 1001))
    assert np.isfinite(history[['loss', 'grad_norm']].to_numpy()).all()
    pool_path = Path('artifacts/e1_real_classification/real_classification_pool.parquet')
    sched_path = Path('artifacts/e1_real_classification/real_classification_schedule.parquet')
    pool = pd.read_parquet(pool_path)
    schedule = pd.read_parquet(sched_path)
    train = pd.read_parquet('artifacts/splits/fold1/train.parquet')
    assert pool.image_path.is_unique and set(pool.image_path) <= set(train.image_path)
    assert not (set(pool.group_path) & set(dev.group_path))
    assert not (set(pool.visual_group_id) & set(dev.visual_group_id))
    joined = schedule.merge(pool[['image_path', 'class_name', 'group_path', 'view_label']], on='image_path', validate='many_to_one')
    assert joined.groupby(['step', 'view_label']).size().eq(4).all()
    assert set(joined.source_type) == {'normal', 'real_anomaly'}
    assert joined.view_label.eq(joined.source_type.eq('real_anomaly').astype(int)).all()
    counts = joined.groupby(['class_name', 'source_type']).size().unstack(fill_value=0)
    loss_windows = {}
    for a, b in [(1, 100), (151, 250), (401, 500), (651, 750), (901, 1000)]:
        loss_windows[f'{a}-{b}'] = history.loc[history.step.between(a, b),
            ['loss', 'source_normal_loss_mean', 'source_real_anomaly_loss_mean']].mean().to_dict()
    summary = json.loads((root / 'training_summary.json').read_text())
    report = dict(
        task='single image; view_label; no aggregation', fixed_threshold=.5,
        inference='stored predictions from completed GPU evaluation; metrics recomputed locally',
        training=summary, metrics=results, loss_windows=loss_windows,
        train_manifest=dict(views=len(train), groups=train.group_path.nunique(),
            normal_groups=train.loc[train.sample_label.eq(0), 'group_path'].nunique(),
            anomaly_groups=train.loc[train.sample_label.eq(1), 'group_path'].nunique()),
        actual_training=dict(pool_views=len(pool), normal_views=int(pool.view_label.eq(0).sum()),
            anomaly_views=int(pool.view_label.eq(1).sum()), groups=pool.group_path.nunique(),
            draws=len(schedule), unique_views=schedule.image_path.nunique(),
            first250_classes=int(joined.loc[joined.step.le(250), 'class_name'].nunique()),
            per_class_draws=counts.to_dict('index')),
    )
    sources = list(inputs.values()) + [dev_path, root / 'training_history.csv', pool_path, sched_path,
        Path('e1/model.py'), Path('e1/evaluate_classification_fold1.py'), Path('e1/make_real_classification_schedule.py')]
    report['input_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (out / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=int), encoding='utf-8')
    main_rows = [[tag] + [f'{row["overall"][k]:.4f}' for k in ['accuracy', 'precision', 'recall', 'f1', 'auroc', 'aupr', 'f1max', 'macro_aupr']]
                 for tag, row in results.items()]
    lines = ['# Qwen3 真实数据单图分类训练结果', '',
        '2026-09-05 完成。1000 步和四次 3600 视角评测全部结束，训练与评测耗时 1505 秒（约 25.1 分钟）。', '',
        '本报告依据保存的逐图预测重新计算指标，没有再次运行模型推理。五份预测均通过完整性和 Dev 的逐图标签一致性检查。', '',
        '结论：本轮未改善单图 baseline，loss 没有稳定下降。存在采样、评测口径及 Qwen3 接口适配错误，因此不能把结果归因为分类任务本身无法学习。', '',
        '## 单图指标', '',
        '正类为可见异常，使用 view_label。Accuracy/Precision/Recall/F1 固定阈值 0.5；F1max 在本 Dev 上搜索，仅作诊断。宏 AUPR 为 18 类等权平均。', '',
        table(['阶段', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUROC', 'AUPR', 'F1max', '宏 AUPR'], main_rows), '',
        'step500 的单图宏 AUPR 在本次四个 checkpoint 中最高，但仍低于 baseline。原 best.pt 恰好也是 step500；原选择过程使用的是零件级宏 AUPR，不能视作按单图指标预先选定。', '',
        '## 混淆矩阵与小缺陷', '',
        table(['阶段', 'TN 正常正确', 'FP 正常误报', 'FN 异常漏检', 'TP 异常正确', '预测异常比例', '小缺陷图片数', '小缺陷召回@0.5'],
            [[tag] + [r['overall'][k] for k in ['tn','fp','fn','tp']] + [f"{r['overall']['positive_rate']:.4f}",r['overall']['small_view_n'],f"{r['overall']['small_recall_at_05']:.4f}"] for tag,r in results.items()]), '',
        '## 训练 loss', '',
        table(['步数窗口', '平均总 BCE', '正常 BCE', '真实异常 BCE'], [[w]+[f'{r[k]:.4f}' for k in ['loss','source_normal_loss_mean','source_real_anomaly_loss_mean']] for w,r in loss_windows.items()]), '',
        '![loss 与单图验证指标](loss_and_single_view_metrics.png)', '',
        '最后100步正常 loss 高于异常 loss，结合误报数可见模型对正常图仍处理不好。单批 loss 下降不能证明验证集分类已收敛。', '',
        '## 数据与协议审计', '',
        '- Dev：18 类、3600 图（1825 正常、1775 异常），每类 200 图。baseline 和四次评测逐图一致。',
        '- 完整 fold1/train：18775 图、3755 组，其中正常零件 539 组，异常零件 3216 组。此前将 2695 个正常视角误写为正常零件组，此处更正。',
        '- 本次实际训练池：4371 图（1750 异常、2621 正常）、2004 个零件组；训练/Dev group_path 和 visual_group_id 无交集。',
        '- 1000步×8图=8000次抽样，覆盖4371张独立图片。正常/异常每步各4张，来源仅真实图；原图固定缩放为448输入，没有伪异常生成。',
        '- 缓存缺失使训练池从18775图缩到4371图，这不是全量真实数据实验。',
        '- 采样脚本按类别拼接，再顺序取图；没有按类别或零件组均衡打散。前250步只接触11类，时序分布偏移可能影响loss和验证曲线。',
        '- 原评测器仍按五视角max聚合，日志的宏AUPR和小缺陷召回是零件级；本报告单独从view_label重算，保留原产物。',
        '- Qwen3视觉适配错误：hook位于合并前的vision blocks，但_restore_spatial按合并后的token数切片。448输入的28×28=784个token仅取196个，decoder缺少其余75%的patch特征；主图像嵌入仍保留，不能称为整张模型只看了1/4图片。',
        '- 当前E1手动调用language_model时未传递Qwen3原生的deepstack_visual_embeds和visual_pos_masks；baseline原生前向包含该通路。这使E1与baseline的视觉信息通路不一致。该结论已对照远端安装的Transformers源码核实。', '',
        '上述接口与采样问题使本轮不足以评价正确实现的Qwen3分类-only方案，也不能仅凭此次结果判断数据量或损失函数是根因。', '',
        '## 每类单图结果', '',
        table(['零件类别', '正常图', '异常图', 'Baseline AUPR', 'Step500 AUPR', 'Step1000 AUPR', 'Step1000 Recall@0.5', 'Step1000 F1@0.5'],
            [[cls, r['n_normal'],r['n_anomaly'],f"{r['aupr']:.4f}",f"{results['step0500']['per_class'][cls]['aupr']:.4f}",f"{results['step1000']['per_class'][cls]['aupr']:.4f}",f"{results['step1000']['per_class'][cls]['recall']:.4f}",f"{results['step1000']['per_class'][cls]['f1']:.4f}"] for cls,r in results['baseline']['per_class'].items()]), '',
        '完整逐类指标、混淆矩阵、数据指纹及采样次数见 [results.json](results.json)。', '',
        '## 后续处理', '',
        '先修复单图评测、类别/组均衡采样和Qwen3适配，补全训练缓存；完成baseline前向一致性与单批梯度检查后再从step1训练。当前不满足进入像素分割阶段的条件。', '']
    (out / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    axes[0].plot(history.step, history.loss, color='#cad7e8', lw=.6, label='Per-step BCE')
    axes[0].plot(history.step, history.loss.rolling(50).mean(), color='#215caa', label='Trailing 50-step mean')
    axes[0].set(title='Qwen3 real-only: completed run (implementation issues found)', ylabel='BCE loss')
    for key,label in [('source_normal_loss_mean','Real normal'),('source_real_anomaly_loss_mean','Real anomaly')]:
        axes[1].plot(history.step, history[key].rolling(50).mean(), label=label)
    axes[1].set(title='Source loss: trailing 50-step mean', ylabel='BCE loss')
    checkpoints = ['step0250','step0500','step0750','step1000']
    for key,label,color in [('macro_aupr','Macro AUPR','#2461a6'),('aupr','Overall AUPR','#ba7f1c')]:
        axes[2].plot([250,500,750,1000],[results[t]['overall'][key] for t in checkpoints], marker='o',color=color,label=label)
        axes[2].axhline(results['baseline']['overall'][key],color=color,ls='--',alpha=.6,label='Baseline '+label)
    axes[2].set(title='Single-view Dev: 3600 images, view_label targets', ylabel='AUPR')
    for ax in axes:
        ax.grid(alpha=.15); ax.legend(fontsize=9); ax.set_xlabel('Optimizer step')
    fig.savefig(out / 'loss_and_single_view_metrics.png', dpi=150)
    plt.close(fig)
    print(table(['Stage','Accuracy','Precision','Recall','F1','AUROC','AUPR','F1max','Macro AUPR'],main_rows))
    print(out / 'RESULTS.md')


if __name__ == '__main__':
    main()
