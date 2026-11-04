# Student ROI 后验224几何失败诊断

后验224几何诊断：只使用已保存预测框及binary GT224像素中心；不改变框，不重算像素指标；anyGT不证明缺陷可见或语言模型关注。

验证集 380：normal 172 / anomaly 208 / 类别 12。三臂full run、预测、token预算、head/RGB/map/array哈希全部先校验，随后读取GT；保存框未改动。

生成答案按初始Yes/No解析；无效答案单独记录，异常图无效计FN。score使用严格 >0。正常图coverage比例为null；peak可用时在零GT上为false，否则null。

| Arm | TP | FN | FP | TN | FN零覆盖 | FN部分覆盖 | FN全部覆盖 | AnyGT /208 | Peak hit / available |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|teacher|126|82|6|166|13|18|51|192|96 / 208|
|pre_gt|118|90|6|166|30|17|43|177|110 / 208|
|pre_kd|118|90|6|166|30|21|39|176|108 / 208|

零/部分/全部覆盖依据GT224前景像素保留数分解，仅为后验几何；anyGT不能等同可见或语言关注。

## 类别FN分解

| 类别 | Arm | 异常数 | FN | 零覆盖FN | 部分覆盖FN | 全部覆盖FN |
|---|---|---:|---:|---:|---:|---:|
|candle|teacher|15|7|0|1|6|
|candle|pre_gt|15|9|2|3|4|
|candle|pre_kd|15|10|1|3|6|
|capsules|teacher|17|13|1|1|11|
|capsules|pre_gt|17|15|3|2|10|
|capsules|pre_kd|17|14|4|3|7|
|cashew|teacher|12|5|2|0|3|
|cashew|pre_gt|12|5|0|0|5|
|cashew|pre_kd|12|5|1|0|4|
|chewinggum|teacher|11|0|0|0|0|
|chewinggum|pre_gt|11|0|0|0|0|
|chewinggum|pre_kd|11|0|0|0|0|
|fryum|teacher|19|7|2|3|2|
|fryum|pre_gt|19|9|4|3|2|
|fryum|pre_kd|19|7|2|3|2|
|macaroni1|teacher|23|18|3|3|12|
|macaroni1|pre_gt|23|18|8|2|8|
|macaroni1|pre_kd|23|18|6|3|9|
|macaroni2|teacher|16|9|3|1|5|
|macaroni2|pre_gt|16|8|4|2|2|
|macaroni2|pre_kd|16|8|4|2|2|
|pcb1|teacher|17|9|1|1|7|
|pcb1|pre_gt|17|11|2|0|9|
|pcb1|pre_kd|17|12|6|0|6|
|pcb2|teacher|16|1|1|0|0|
|pcb2|pre_gt|16|2|2|0|0|
|pcb2|pre_kd|16|2|2|0|0|
|pcb3|teacher|17|9|0|4|5|
|pcb3|pre_gt|17|6|3|1|2|
|pcb3|pre_kd|17|8|3|3|2|
|pcb4|teacher|25|0|0|0|0|
|pcb4|pre_gt|25|2|2|0|0|
|pcb4|pre_kd|25|1|1|0|0|
|pipe_fryum|teacher|20|4|0|4|0|
|pipe_fryum|pre_gt|20|5|0|4|1|
|pipe_fryum|pre_kd|20|5|0|4|1|

## 三臂漏检交集

- teacher_and_pre_gt: 74
- teacher_and_pre_kd: 76
- pre_gt_and_pre_kd: 85
- all_three: 74

GT vs KD逐图生成决策差异：共 18；normal 8，anomaly 10。相同汇总计数不代表相同预测。

| 标签 | GT → KD | 数量 |
|---|---|---:|
|normal|0->0|162|
|normal|0->1|4|
|normal|1->0|4|
|normal|1->1|2|
|anomaly|0->0|85|
|anomaly|0->1|5|
|anomaly|1->0|5|
|anomaly|1->1|113|

## 解释边界

- 单seed、同一dev/val380的后验诊断；不构成独立test、因果验证、等价或非劣结论。
- 224二值mask与归一化bbox的像素中心几何仅近似原图缺陷保留；不度量缩放后的细节可辨识性、遮挡、上下文或语言注意力。
- 包含部分或全部GT不证明模型看到或理解缺陷；零覆盖也只描述ROI，原始整图仍作为第一视图输入。
- teacher peak仅来自已保存ROI记录；未加载teacher map/cache或验证其历史argmax。学生peak来自已校验map数组。
- head仅重新核对checkpoint文件SHA；state_sha256保留manifest记录，不重新反序列化checkpoint。

逐图数据：`per_image.jsonl`、`per_image.csv`（CSV一图一行，各arm展开为列）；交集路径与完整校验来源见`summary.json`。
