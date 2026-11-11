# 校准与最终确认图表补充索引

这是对既有FIGURE_SUPPLEMENT_INDEX.md的追加记录，不覆盖原图与已冻结指标。

| 图表 | 数据角色 | 核心信息 |
|---|---|---|
| figures/calibration_heldout_analysis.png/.svg | calibration120拟合，heldout120评价 | AUROC不变92.82%，FPR11/60到6/60，Brier/logloss改善；ECE差区间跨0 |
| figures/confirmation_fixed_system_analysis.png/.svg | 原固定最终confirmation240 | AUROC90.18%；Recall106/120到102/120，FP30/120到25/120；properloss/ECE改善；主系统仍固定calibrated |

两幅图各有provenance.json、INDEPENDENT_ANALYSIS.md及STATISTICAL_SUPPLEMENT，统计/绘图均在指定服务器；本机只查看与同步。

确认集检验已完成，不依据确认结果重新选择映射、阈值或参考图；后续新假设须新协议和新留出来源。

来源互斥的开发120+calibration/heldout240+confirmation240共600张唯一query，逻辑API目标可能重复引用同图，分母不可混同。

完整路径及SHA见stages/final_calibration_confirmation_figure_index.json。
