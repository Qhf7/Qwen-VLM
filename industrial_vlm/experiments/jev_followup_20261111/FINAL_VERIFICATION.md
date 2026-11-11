# 最终结果独立核验补充

本补充文件在最终汇总后生成，不修改已冻结的科学报告、预测、合同和审计。所有统计与核验均在指定AutoDL服务器执行。

| 核验 | 通过 | 失败 | 范围 |
|---|---:|---:|---|
| stage08救援独立审计 | 4437 | 0 | 600目标、原文件不变、请求身份、指标及2000次配对区间 |
| stage10含stage08独立审计 | 4498 | 0 | 120个320与复用shared0120；排除支持480；预算与区间 |
| stage09 Qwen独立输入审计 | 240行完整 | 0个tensor错配 | base/DD同PNG、同native像素tensor，480个布尔候选forward |

stage10独立检查包含stage08检查，不把两项通过数相加作为独立检查量。独立数值实现使用pairwise AUC、grouped AP与配对分层重采样，与正式指标一致。

## 执行状态

- ROI API：240接受／278started／38明确502。
- 支持API：600接受／699started／99明确502；原597与救援3显式合并，旧stage08状态原样保留。
- 分辨率API：120接受／120started／0失败／0未决。
- Qwen：240图像判断／480候选forward；未进行新训练。
- 仅使用既有120张开发query的已批准布尔标签；confirmation继续未读取。
- 候选ROI仍没有已证实的定位净收益；分辨率AUROC区间跨零；这些限制保持在正式报告中。

## 冻结文件SHA256

- `FINAL_FOLLOWUP_REPORT.md`: `8dced54c1da46b6ae5dfd7f5051537a265ccd3352a1d0e36cd4a4e772b8781f7`
- `stages/11_summary.json`: `f89bb7adfb19dcd0708460729e9e959cc2c2596ecb655495bbed22ff627b9280`
- `stages/08_postexecution_audit.json`: `33481864faea246dfa2779c97c5087afa579c3d0204ee1920afc934762eaabf2`
- `stages/10_postexecution_audit.json`: `dcdef3e9374860986b96016820c65b66acf08e8bd178092f024e3bc598ea2362`
- `stage09_qwen_matched/final_audit.json`: `81875eea94f990605fe7e427b340e5cb600ff2443e0883a56386e746e0f7efbd`
- `stage10_resolution/evaluation_contract_amended_reference.json`: `79e0b0dab9fb166b1da7f922459642d621eed93c67ce31626cde0396da3fc256`

核验本身不发API、不调用模型、不重拟合校准器、不调阈值、不打开GT mask。下一阶段系统选择、独立校准与最终确认由各自冻结协议推进。
