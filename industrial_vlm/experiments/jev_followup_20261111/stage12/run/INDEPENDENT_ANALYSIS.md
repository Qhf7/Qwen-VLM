# Stage12留出结果独立解释

指定服务器完成240预测/251started/11纯502，unknown0；独立审计1550项通过、0失败。先冻结预测，使用calibration120拟合固定logit-logistic，冻结系数后才访问heldout120标签。参数a=1.567205313222、b=-0.156322190691，映射为p_cal=sigmoid(a*logit(clip(p_raw,1e-6,1-1e-6))+b)。这是单调映射，所以本数据AUROC/AP不变：92.82%/94.24%。它调整概率与固定决策边界，而非改模型排序。

校准后p_cal>=0.5等价于p_raw约>=0.524916，该数值由calibration拟合系数推导，没有在heldout调阈值。heldout异常53/60检出不变；正常误报由11/60降到6/60。FPR配对差-8.33pp，95%区间[-15,-3.33]pp。calibrated FPR自身Wilson95%区间[4.66%,20.15%]，不能保证未来10%或5%。

Brier从0.12423降到0.10774，配对95%差区间[-0.02366,-0.00872]；Logloss从0.40479降到0.34859，差区间[-0.07614,-0.03383]。这两项支持固定映射在本留出来源上改善概率质量。ECE10从0.15833降到0.11042，但差区间[-0.07082,+0.00843]跨0，不能把点估计下降直接写成稳定ECE收益。ECE分箱对120小样本敏感，应与proper loss一起看。

固定coverage50%/75%下raw和calibrated条件错误率分别同为3.33%/8.89%；100%覆盖错误率由15%降到10.83%。拒答不当正常，coverage是在这份留出集上按预声明置信排序取得固定比例，不保证未来部署coverage，也没有重新选择最优拒答阈值。

这是来源互斥的新采样native validation remainder研究，不是盲测或最终confirmation。每类只有5正常/5异常，平衡50%异常先验不同于产线；bootstrap固定本次calibrator，未包含calibration拟合或API/provider变化。后续最终确认必须保持唯一system/映射/阈值/拒答规则冻结，不能依据这轮留出再换prompt、参考、size或shots。

图为figures/calibration_heldout_analysis.png/.svg，逐bin/provenance和完整STATISTICAL_SUPPLEMENT均已保存。confirmation在本阶段未读。
