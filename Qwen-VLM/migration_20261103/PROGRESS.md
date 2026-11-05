# Qwen3-VL → AnomalyGPT 迁移进度

快照时间：2026-11-04T00:18:18+08:00。来源为已经同步的真实 JSON；缺失或未完成项不计为0。

结果合同状态：**complete**；远程串行队列：**complete**，current_stage=dense_kd。

| 顺序 | 本轮完成范围 | 状态 |
|---|---|---|
| 1. 语言骨干替换 | Vicuna/Qwen × seed42/43/44，六组完整 val380 | complete |
| 2. 原生视觉＋异常图 | Native baseline＋18 map tokens，两臂训练/重载/官方接口检查 | complete |
| 3. Qwen特征定位 | normal-memory 12格：complete；独立GT定位头：5/5 | complete |
| 4. 原图＋ROI／参考 | pilot48：complete；full val380五条件：complete | complete |
| 5. 训练期教师蒸馏 | pre448/post448，GT＋0.5×KD，2/2；匹配合同见配对结果 | complete |

## 已验收的主要数值

语言骨干三配对seed image AUROC（均值±样本SD）：Vicuna **0.889684±0.013285**，Qwen **0.925779±0.032698**。
三seed Qwen AUROC均高；生成F1并非逐seed都提高，不能据此宣称统计显著或普遍更稳定。

| Native方案 | Image AUROC | 生成macro F1 | 生成invalid率 |
|---|---:|---:|---:|
| baseline | 0.626342 | 0.218069 | 0.702632 |
| +18 map tokens | 0.866391 | 0.812690 | 0.000000 |

Native baseline的invalid来自非空解释性前缀；原prompt未强制Yes/No首词，训练收益混入回答格式适配。详见NATIVE_PROTOCOL_AUDIT.md。

| 独立定位头（seed42） | 监督 | Best step | Pixel AP macro | Pixel AUROC macro |
|---|---|---:|---:|---:|
| imagebind224 | GT | 1000 | 0.390244 | 0.974284 |
| pre224 | GT | 1200 | 0.217631 | 0.907415 |
| post224 | GT | 1200 | 0.181908 | 0.935581 |
| pre448 | GT | 1200 | 0.336404 | 0.952411 |
| pre448 | GT_KD | 1200 | 0.327643 | 0.953943 |
| post448 | GT | 1200 | 0.295317 | 0.966730 |
| post448 | GT_KD | 1200 | 0.302921 | 0.967149 |

同特征配对 pre448：KD−GT Pixel AP=-0.008761，Pixel AUROC=0.001532。

同特征配对 post448：KD−GT Pixel AP=0.007603，Pixel AUROC=0.000419。

| Full ROI条件（各380图） | 合同状态 | Image AUROC | 生成macro F1 |
|---|---|---:|---:|
| original_only | complete | 0.753480 | 0.446888 |
| original_center_crop | complete | 0.713174 | 0.639874 |
| original_random_crop | complete | 0.726996 | 0.548301 |
| original_teacher_roi | complete | 0.837265 | 0.765826 |
| original_teacher_roi_reference | complete | 0.830417 | 0.731000 |

只有整份run标记complete后，full ROI各条件才进入完成表；执行中已有单条件指标可在roi_full/run.json查看。

## 验证与边界

- 统一visa-square-binary-sklearn-v1。监督式自定义60/20/20切分：train9985（含真实异常732）、val380。
- 历史dev test456已参与以往开发；本轮无新test推理，不能把本轮val结果表述为未见测试集泛化。
- 语言骨干两臂同seed的初始化、采样计划、teacher/cache已匹配；训练骨干冻结，无LoRA或量化。
- Native采用官方同形状exact gate；官方自身跨attention形状的BF16差异仅作诊断。
- ROI只用teacher预测峰值，不用GT mask选框；center/random/teacher两图预算相同。参考条件额外增加第三图、train-normal信息及ImageBind检索。
- Native baseline与ROI original的prompt、分辨率和token预算不同；各路线不可直接当成编码器单变量实验。
- Dense同source GT/KD配对相同head、初始化、计划和预算；跨source维度、参数量、FP16/BF16不同。
- KD的teacher只用于训练；验证/推理不读teacher。目前完成范围是视觉定位头，尚未将蒸馏头接回语言回答系统。
- 第2—5路线以seed42为探索性证据；多seed泛化尚未建立。

## 产物入口

- [ALL_RESULTS.md](ALL_RESULTS.md)：全量数字和协议；[all_summary.json](all_summary.json)：完成合同、输入路径和SHA。
- [PHASE1_RESULTS.md](PHASE1_RESULTS.md)、[PHASE2_RESULTS.md](PHASE2_RESULTS.md)：语言替换与原生融合。
- [PHASE3_5_RESULTS.md](PHASE3_5_RESULTS.md)、[PHASE4_RESULTS.md](PHASE4_RESULTS.md)：定位、蒸馏及ROI完整结果。
- [FINAL_RECOMMENDATION.md](FINAL_RECOMMENDATION.md)：当前路线判断；[INFERENCE_USAGE.md](INFERENCE_USAGE.md)：已验证推理命令。
- [ROI_PROTOCOL_AUDIT.md](ROI_PROTOCOL_AUDIT.md)、[NATIVE_PROTOCOL_AUDIT.md](NATIVE_PROTOCOL_AUDIT.md)：协议审计。
- [migration_results.png](migration_results.png)：按协议分区比较；code/source_manifest.json：执行代码快照。
