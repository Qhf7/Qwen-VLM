# Stage12预协议修订01：允许1-shot候选

生效时间：stage12任何API请求、新响应和最终系统选择之前。本修订不覆盖原preprotocol.json或README.md，不改变已选240来源与120/120角色。

最终候选由仅0/4具体化为0/1/4。父级规定的选择分数：0-shot使用stage08 shared0 AUROC，1-shot使用s43_1/s44_1 AUROC均值，4-shot使用s43_4/s44_4 AUROC均值。选择与最高分差距不超过0.01绝对AUROC的最少shots，包含等号；非零参考固定seed43，尺寸448×448。独立执行器还须在读stage08指标前冻结精确selection_protocol及SHA。

stage12仍只执行所选唯一arm的calibration120+heldout120=240 logical targets；不各跑三个arm。不根据这240新响应选择prompt、参考、size或shots。固定校准方法、标签隔离、全分母评价、无Teacher和confirmation未开启的所有条款保持生效。原文“0-shot或4-shot”由本修订允许“0-shot、1-shot或4-shot”。

元数据选样阶段已知分层标签；这不是严格盲测。修订时stage12 API为0，响应为0。
