"""Report only completed paired runs, including step0 and final outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).parent)
    args = ap.parse_args()
    root = args.root
    rows, identities = [], {}
    for seed in (42, 43, 44):
        for arm in ('B', 'C'):
            d = root / ('arm_%s_s%d' % (arm, seed))
            if not (d / 'complete.json').exists():
                continue
            run = json.loads((d / 'run.json').read_text())
            done = json.loads((d / 'complete.json').read_text())
            assert done['steps'] == 500 and not done['preflight'] and done['test_queries'] == 0
            assert done['reload_max_abs_score_error'] == 0
            assert done['paired_A_max_abs_score_error'] == 0
            assert done['frozen_parameter_versions_unchanged']
            if arm == 'C':
                assert done['frozen_bridge_unchanged']
            schedule = json.loads((d / 'sample_indices.json').read_text())
            assert hashlib.sha256(json.dumps(schedule).encode()).hexdigest() == run['identity']['sample_schedule_sha256']
            identities[(seed, arm)] = run['identity']
            for mode, filename in [('A', 'initial_val.json'), ('best', 'best_val_metrics.json'),
                                   ('final', 'final_val_metrics.json')]:
                m = json.loads((d / filename).read_text())
                score = m['overall']['scores']; gen = m['overall']['generated']
                assert score['n'] == gen['n'] == 380
                rows.append(dict(seed=seed, arm=arm, mode=mode,
                                 step=0 if mode == 'A' else done['best_step'] if mode == 'best' else 500,
                                 auroc=score['image_auroc'], ap=score['image_ap'],
                                 score_f1=score['macro_f1'], score_recall=score['recall'], score_fpr=score['fpr'],
                                 gen_f1=gen['macro_f1'], gen_recall=gen['recall'], gen_fpr=gen['fpr'],
                                 invalid=gen['invalid_rate'], category_macro=m['category_macro'],
                                 cat_auroc=m['category_macro']['scores']['image_auroc'],
                                 cat_ap=m['category_macro']['scores']['image_ap'],
                                 cat_gen_f1=m['category_macro']['generated']['macro_f1'],
                                 trainable=0 if mode == 'A' else done['trainable_parameters'],
                                 peak_gib=None if mode == 'A' else done['peak_memory_bytes']/2**30,
                                 exposures_per_second=None if mode == 'A' else done['exposures_per_second']))
    paired_seeds = [s for s in (42, 43, 44) if (s, 'B') in identities and (s, 'C') in identities]
    pairs = []
    for s in paired_seeds:
        for k in ('bridge_sha256', 'cache_sha256', 'base_manifest_sha256', 'sample_schedule_sha256'):
            assert identities[(s, 'B')][k] == identities[(s, 'C')][k], (s, k)
        def row(arm, mode):
            return next(r for r in rows if r['seed'] == s and r['arm'] == arm and r['mode'] == mode)
        assert row('B', 'A') == row('C', 'A') | {'arm': 'B'}
        for mode in ('best', 'final'):
            a, b, c = row('B', 'A'), row('B', mode), row('C', mode)
            pairs.append(dict(seed=s, mode=mode, B_minus_A={k:b[k]-a[k] for k in ('auroc','ap','gen_f1','gen_recall','gen_fpr')},
                              C_minus_A={k:c[k]-a[k] for k in ('auroc','ap','gen_f1','gen_recall','gen_fpr')},
                              C_minus_B={k:c[k]-b[k] for k in ('auroc','ap','gen_f1','gen_recall','gen_fpr')}))
    def mean_sd(values):
        return dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values) if len(values)>1 else None,
                    n=len(values))
    aggregate, paired_aggregate = {}, {}
    if len(paired_seeds) == 3:
        for arm, mode in [('B','A'),('B','best'),('C','best'),('B','final'),('C','final')]:
            group = [r for r in rows if r['arm']==arm and r['mode']==mode]
            aggregate[arm+'_'+mode] = {k:mean_sd([r[k] for r in group]) for k in
                                      ('auroc','ap','gen_f1','gen_recall','gen_fpr','invalid',
                                       'cat_auroc','cat_ap','cat_gen_f1')}
            if mode != 'A':
                aggregate[arm+'_'+mode].update({k:mean_sd([r[k] for r in group]) for k in
                                               ('peak_gib','exposures_per_second')})
        for mode in ('best', 'final'):
            group = [p for p in pairs if p['mode'] == mode]
            paired_aggregate[mode] = {contrast: {
                k: dict(**mean_sd([p[contrast][k] for p in group]),
                        positive_seeds=sum(p[contrast][k] > 0 for p in group))
                for k in ('auroc','ap','gen_f1','gen_recall','gen_fpr')}
                for contrast in ('B_minus_A','C_minus_A','C_minus_B')}
    verification = json.loads((root/'verify_status.json').read_text()) if (root/'verify_status.json').exists() else {'state':'missing'}
    if verification['state'] == 'complete':
        assert len(verification['checks']) == 5
        for check in verification['checks']:
            name = ('preflight_' if check['preflight'] else 'arm_') + check['arm'] + '_s' + str(check['seed'])
            fresh = json.loads((root/name/'fresh_final_reload.json').read_text())
            assert fresh['fresh_process_max_abs_score_error'] == 0 and fresh['generated_exact_match']
            assert fresh['verifier_sha256'] == hashlib.sha256((root/'verify_lora_checkpoint.py').read_bytes()).hexdigest()
            if check['arm'] == 'C':
                assert fresh['trained_lora_B_nonzero'] and fresh['disabled_adapter_max_abs_score_error'] == 0
    result = dict(state='complete' if len(paired_seeds)==3 else 'partial', paired_seeds=paired_seeds,
                  rows=rows, paired_deltas=pairs, aggregate=aggregate, paired_aggregate=paired_aggregate,
                  independent_verification=verification, test_queries=0)
    (root/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    lines = ['# B/C 语言 LoRA 延续实验结果', '',
             '仅纳入完成500updates且通过工程门槛的实验。A为该seed的已有桥接起点；best允许选择step0，final固定为500。', '',
             '| Seed | Arm | Checkpoint | Step | AUROC | AP | 生成F1 | 召回 | FPR | Invalid |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| {seed} | {arm} | {mode} | {step} | {auroc:.6f} | {ap:.6f} | {gen_f1:.6f} | {gen_recall:.6f} | {gen_fpr:.6f} | {invalid:.6f} |'.format(**r))
    lines += ['', '已完成配对seed：'+str(paired_seeds)+'。结果状态：'+result['state']+'。', '']
    if aggregate:
        lines += ['| Arm/checkpoint | AUROC均值±样本SD | AP均值 | 生成F1均值±样本SD | 召回均值 | FPR均值 |',
                  '|---|---:|---:|---:|---:|---:|']
        for k,m in aggregate.items():
            lines.append('| %s | %.6f±%.6f | %.6f | %.6f±%.6f | %.6f | %.6f |' % (
                k,m['auroc']['mean'],m['auroc']['sample_sd'],m['ap']['mean'],m['gen_f1']['mean'],
                m['gen_f1']['sample_sd'],m['gen_recall']['mean'],m['gen_fpr']['mean']))
        lines += ['', '以下差值先按同seed相减，再取3seed均值；单位为百分点（×100），SD为配对差的样本标准差。FPR增加代表更多正常图误报。', '',
                  '| Checkpoint/contrast | ΔAUROC均值±SD | ΔAP | Δ生成F1 | Δ召回 | ΔFPR | AUROC提高seed数 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for mode, contrasts in paired_aggregate.items():
            for contrast, m in contrasts.items():
                lines.append('| %s/%s | %+.4f±%.4f | %+.4f | %+.4f | %+.4f | %+.4f | %d/3 |' % (
                    mode,contrast,100*m['auroc']['mean'],100*m['auroc']['sample_sd'],100*m['ap']['mean'],
                    100*m['gen_f1']['mean'],100*m['gen_recall']['mean'],100*m['gen_fpr']['mean'],m['auroc']['positive_seeds']))
        lines += ['', '| Arm/checkpoint | 类别等权AUROC | 类别等权AP | 类别等权生成F1 |', '|---|---:|---:|---:|']
        for k,m in aggregate.items():
            lines.append('| %s | %.6f | %.6f | %.6f |' % (k,m['cat_auroc']['mean'],m['cat_ap']['mean'],m['cat_gen_f1']['mean']))
    lines += ['', '| Seed/arm | 可训练参数 | 峰值GiB | 训练曝光/秒 |', '|---|---:|---:|---:|']
    for r in rows:
        if r['mode']=='best':
            lines.append('| %d/%s | %d | %.3f | %.2f |' % (r['seed'],r['arm'],r['trainable'],r['peak_gib'],r['exposures_per_second']))
    lines += ['', '独立进程重载验证状态：`'+verification['state']+'`；具体覆盖见 `verify_status.json` 与各run的 `fresh_final_reload.json`。', '',
              'B/C参数组和候选学习率不同；同曝光与同采样不等于同计算量。生成与连续分数分开评估，完整类别等权指标见各run JSON。', '',
              'val380已多轮用于开发，训练含真实异常；本结果不代表官方无监督基准或未见盲测泛化。没有新test查询。', '']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(state=result['state'],paired_seeds=paired_seeds,
                          mean_auroc={k:v['auroc']['mean'] for k,v in aggregate.items()},
                          independent_verification=verification['state']),ensure_ascii=False))


if __name__ == '__main__':
    main()
