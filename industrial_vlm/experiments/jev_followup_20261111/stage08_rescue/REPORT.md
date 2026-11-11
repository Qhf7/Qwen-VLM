# Jev stage08 HTTP502运维修订与完整结果

原科学contract、支持seed、PNG、问题、请求UUID/body与已接受响应均保持不变。
仅对原3次均明确HTTP502的耗尽目标增加最多2次显式单并发请求；原日志和status未改写。

| 条件 | AUROC | AP | Recall | FPR | Brier | ECE10 |
|---|---:|---:|---:|---:|---:|---:|
| shared0 | 0.834583 | 0.868369 | 0.783333 | 0.350000 | 0.172938 | 0.117250 |
| s43_1 | 0.892083 | 0.909668 | 0.866667 | 0.300000 | 0.146962 | 0.112500 |
| s43_4 | 0.913333 | 0.928396 | 0.850000 | 0.233333 | 0.132546 | 0.132083 |
| s44_1 | 0.891389 | 0.904805 | 0.883333 | 0.250000 | 0.148015 | 0.134167 |
| s44_4 | 0.911250 | 0.927796 | 0.850000 | 0.233333 | 0.132563 | 0.129833 |

## 结果分析

- seed43 4参考相对shared0 AUROC变化+7.87个百分点；完整配对区间见metrics.json。
- seed44 4参考相对shared0 AUROC变化+7.67个百分点；完整配对区间见metrics.json。

## 运维与解释边界

- 原接受597；救援接受3；额外started=3。
- completion_certificate明确为operational amendment完整，不将原stage08/status伪装为complete。
- 120均为已看开发query；source/byte隔离与teacher exposure按原contract，未打开confirmation。
- 两支持seed仅作固定bank描述；bootstrap不包括支持选择、provider随机性、时间变化或多重比较校正。
- 重试为显式HTTP502运维恢复，不据此宣称新科学效果；缺失usage不计零，不估货币账单。
