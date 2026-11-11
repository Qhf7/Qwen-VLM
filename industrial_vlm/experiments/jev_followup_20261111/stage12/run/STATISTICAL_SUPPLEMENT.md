# Stage12独立审计与补充统计

独立审计通过1550项，0失败。240预测及完整attempt账本冻结后，先仅calibration120复拟固定logistic并核系数，再读heldout120标签。此补充不改runner/metrics，不选新阈值或方法。

| 方法 | TP | FP | TN | FN | Recall及Wilson95% | FPR及Wilson95% | Logloss |
|---|---:|---:|---:|---:|---|---|---:|
| raw | 53 | 11 | 49 | 7 | 0.8833 [0.7782,0.9423] | 0.1833 [0.1056,0.2992] | 0.404792 |
| calibrated | 53 | 6 | 54 | 7 | 0.8833 [0.7782,0.9423] | 0.1000 [0.0466,0.2015] | 0.348591 |

校准−raw Logloss差-0.056200，95%配对来源bootstrap区间[-0.076143,-0.033828]；固定calibrator，不包含校准拟合不确定性。Logloss数值clip1e-15，与拟合featureclip1e-6区别已明确。

| 方法 | 覆盖率 | 保留 | 拒答 | 条件错误率及Wilson95% |
|---|---:|---:|---:|---|
| raw | 0.5 | 60 | 60 | 0.0333 [0.0092,0.1136] |
| raw | 0.75 | 90 | 30 | 0.0889 [0.0457,0.1657] |
| raw | 1.0 | 120 | 0 | 0.1500 [0.0970,0.2247] |
| calibrated | 0.5 | 60 | 60 | 0.0333 [0.0092,0.1136] |
| calibrated | 0.75 | 90 | 30 | 0.0889 [0.0457,0.1657] |
| calibrated | 1.0 | 120 | 0 | 0.1083 [0.0644,0.1766] |

拒答不是正常预测；固定coverage是事后风险描述，不保证部署coverage。条件subset的FPR/Recall与全120分母不可混用。完整混淆矩阵、逐类区间、coverage子集聚合与logloss差异见同名JSON。

来源隔离新采样仍属于nativeval remainder，可能历史验证已看，不称blind或confirmation；平衡异常先验50%不代表产线缺陷率。校准有改善也有退化可能，保留完整负结果与区间。confirmation仍未读取。

- runner metrics SHA：075f823dc6fc0617844e3d5578ffd0c04b3b0d077e07052f5cff9628408afea4
- calibrator SHA：0bac063df108bd44938c25a8bb2c07261d79bfe2350b6d1bdb8b8cf1c1bcd9ec
- independent audit SHA：595c853d71a3dee79bfa11d40497cf9c0017ab5a07b54f5625bc6a6f3e8a31bb
- supplement JSON SHA：25d0cf2b74f2b9c9441cb9bdaeae7eb11a22b96b6f6c89e6811cb6acb0e38912
