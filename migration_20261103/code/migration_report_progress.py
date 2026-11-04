"""Write one consistent progress snapshot from the validated JSON summary."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def chosen(value):
    if value.get('status') != 'complete':
        return None
    return value.get('selected_complete', value)


def number(value):
    return '—' if value is None else '%.6f' % value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    summary = json.loads((root / 'all_summary.json').read_text())
    routes = summary['routes']
    sequence_file = root / 'ordered_sequence_status.json'
    sequence = json.loads(sequence_file.read_text()) if sequence_file.exists() else {}
    phase1 = routes['phase1_language_backbone']
    native = routes['native_map_increment']
    memory = routes['frozen_normal_memory']
    dense = routes['supervised_dense_GT_and_KD']
    roi = routes['native_ROI']
    gt = [x for x in dense['matrix'] if x['arm'] == 'GT']
    kd = [x for x in dense['matrix'] if x['arm'] == 'GT_KD']
    complete_gt = sum(chosen(x['result']) is not None for x in gt)
    complete_kd = sum(chosen(x['result']) is not None for x in kd)
    time = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    lines = [
        '# Qwen3-VL → AnomalyGPT 迁移进度', '',
        '快照时间：%s。来源为已经同步的真实 JSON；缺失或未完成项不计为0。' % time, '',
        '结果合同状态：**%s**；远程串行队列：**%s**，current_stage=%s。'
        % (summary['status'], sequence.get('state', 'unknown'), sequence.get('current_stage', 'unknown')), '',
        '| 顺序 | 本轮完成范围 | 状态 |', '|---|---|---|',
        '| 1. 语言骨干替换 | Vicuna/Qwen × seed42/43/44，六组完整 val380 | %s |' % phase1['status'],
        '| 2. 原生视觉＋异常图 | Native baseline＋18 map tokens，两臂训练/重载/官方接口检查 | %s |' % native['status'],
        '| 3. Qwen特征定位 | normal-memory 12格：%s；独立GT定位头：%d/5 | %s |'
        % (memory['status'], complete_gt, 'complete' if memory['status'] == 'complete' and complete_gt == 5 else 'ongoing'),
        '| 4. 原图＋ROI／参考 | pilot48：%s；full val380五条件：%s | %s |'
        % (roi['pilot']['status'], roi['full_validation']['status'], roi['full_validation']['status']),
        '| 5. 训练期教师蒸馏 | pre448/post448，GT＋0.5×KD，%d/2；匹配合同见配对结果 | %s |'
        % (complete_kd, 'complete' if complete_kd == 2 and all(p['status'] == 'complete' for p in dense['pairs']) else 'ongoing'), '',
        '## 已验收的主要数值', '',
    ]
    aggregate = phase1.get('aggregate')
    if aggregate:
        stats = aggregate['by_backend']
        q, v = stats['qwen']['image_auroc'], stats['vicuna']['image_auroc']
        lines += ['语言骨干三配对seed image AUROC（均值±样本SD）：Vicuna **%.6f±%.6f**，Qwen **%.6f±%.6f**。'
                  % (v['mean'], v['sample_sd'], q['mean'], q['sample_sd']),
                  '三seed Qwen AUROC均高；生成F1并非逐seed都提高，不能据此宣称统计显著或普遍更稳定。', '']
    lines += ['| Native方案 | Image AUROC | 生成macro F1 | 生成invalid率 |', '|---|---:|---:|---:|']
    for cell in native['matrix']:
        record = chosen(cell['result'])
        if record:
            score = record['conditions']['real_map']['scores']
            gen = record['generated_real_map']['generated']
            lines += ['| %s | %s | %s | %s |' % ('baseline' if cell['baseline'] else '+18 map tokens', number(score['image_auroc']), number(gen['macro_f1']), number(gen['invalid_rate']))]
    lines += ['', 'Native baseline的invalid来自非空解释性前缀；原prompt未强制Yes/No首词，训练收益混入回答格式适配。详见NATIVE_PROTOCOL_AUDIT.md。', '',
              '| 独立定位头（seed42） | 监督 | Best step | Pixel AP macro | Pixel AUROC macro |', '|---|---|---:|---:|---:|']
    for cell in dense['matrix']:
        record = chosen(cell['result'])
        if record:
            metric = record['metrics']
            lines += ['| %s%d | %s | %s | %s | %s |' % (cell['feature'], cell['requested_size'], cell['arm'], record['best_step'], number(metric['pixel_ap_macro']), number(metric['pixel_auroc_macro']))]
    for pair in dense['pairs']:
        if pair['status'] == 'complete':
            lines += ['', '同特征配对 %s%d：KD−GT Pixel AP=%s，Pixel AUROC=%s。'
                      % (pair['feature'], pair['requested_size'], number(pair['delta_KD_minus_GT']['pixel_ap_macro']), number(pair['delta_KD_minus_GT']['pixel_auroc_macro']))]
    lines += ['', '| Full ROI条件（各380图） | 合同状态 | Image AUROC | 生成macro F1 |', '|---|---|---:|---:|']
    for cell in roi['full_validation']['matrix']:
        record = chosen(cell['result'])
        lines += ['| %s | %s | %s | %s |' % (cell['condition'], cell['result']['status'], number(record['metrics']['scores']['image_auroc']) if record else '—', number(record['metrics']['generated']['macro_f1']) if record else '—')]
    lines += ['', '只有整份run标记complete后，full ROI各条件才进入完成表；执行中已有单条件指标可在roi_full/run.json查看。', '',
              '## 验证与边界', '',
              '- 统一visa-square-binary-sklearn-v1。监督式自定义60/20/20切分：train9985（含真实异常732）、val380。',
              '- 历史dev test456已参与以往开发；本轮无新test推理，不能把本轮val结果表述为未见测试集泛化。',
              '- 语言骨干两臂同seed的初始化、采样计划、teacher/cache已匹配；训练骨干冻结，无LoRA或量化。',
              '- Native采用官方同形状exact gate；官方自身跨attention形状的BF16差异仅作诊断。',
              '- ROI只用teacher预测峰值，不用GT mask选框；center/random/teacher两图预算相同。参考条件额外增加第三图、train-normal信息及ImageBind检索。',
              '- Native baseline与ROI original的prompt、分辨率和token预算不同；各路线不可直接当成编码器单变量实验。',
              '- Dense同source GT/KD配对相同head、初始化、计划和预算；跨source维度、参数量、FP16/BF16不同。',
              '- KD的teacher只用于训练；验证/推理不读teacher。目前完成范围是视觉定位头，尚未将蒸馏头接回语言回答系统。',
              '- 第2—5路线以seed42为探索性证据；多seed泛化尚未建立。', '',
              '## 产物入口', '',
              '- [ALL_RESULTS.md](ALL_RESULTS.md)：全量数字和协议；[all_summary.json](all_summary.json)：完成合同、输入路径和SHA。',
              '- [PHASE1_RESULTS.md](PHASE1_RESULTS.md)、[PHASE2_RESULTS.md](PHASE2_RESULTS.md)：语言替换与原生融合。',
              '- [PHASE3_5_RESULTS.md](PHASE3_5_RESULTS.md)、[PHASE4_RESULTS.md](PHASE4_RESULTS.md)：定位、蒸馏及ROI完整结果。',
              '- [FINAL_RECOMMENDATION.md](FINAL_RECOMMENDATION.md)：当前路线判断；[INFERENCE_USAGE.md](INFERENCE_USAGE.md)：已验证推理命令。',
              '- [ROI_PROTOCOL_AUDIT.md](ROI_PROTOCOL_AUDIT.md)、[NATIVE_PROTOCOL_AUDIT.md](NATIVE_PROTOCOL_AUDIT.md)：协议审计。',
              '- [migration_results.png](migration_results.png)：按协议分区比较；code/source_manifest.json：执行代码快照。', '']
    destination = root / 'PROGRESS.md'
    destination.write_text('\n'.join(lines))
    print(destination)


if __name__ == '__main__':
    main()
