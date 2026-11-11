# 接口文档核验记录

2026-11-11（北京时间），指定 AutoDL 服务器逐页抓取 TypeSafe Noul、Box、Images 和 ModelGo System One 文档，四页均 HTTP 200。原 HTML、初步文本和逐页 SHA 保存在 reference_docs/manifest.json。系统检索未返回可引用结果，因此本记录依据已保存的原文快照，不编造搜索引用。

## 当前可依据的内容

- TypeSafe Noul 原文说明没有独立 confidence 字段；二元分布由 noul 概率表示。不能为本项目 Noul 额外发明置信度输出。
- TypeSafe Box 以 top、bottom、left、right 表示四条归一化边界；不是 xywh，也不是像素分割。原文包含独立 confidence 示意，不能与 Noul 混淆。
- Images 原文要求图片宽高各 280–1568 像素，并要求图片或 Box 请求携带 TypeSafe-Danger-Zone-Images:true。Images/Box 仍标 Dreamer-ready。当前项目固定 448×448；256/224 不是原生合法输入。
- ModelGo 文档同日仍写“不支持 image assets、box 与需要 TypeSafe-Danger-Zone-* 的功能”。这与本项目已经成功完成的真实图片 Noul 请求冲突，应分别保存文档与实测。不能由文档否认已经发生的成功，也不能由 Noul 成功推定同渠道 Box/Choice 全部可用。

## 对本轮执行的影响

在线工具 pilot 继续使用已验证的 /v1/systemone、jev-1.14.0、图片 assets、Noul。crop-needed 是新的问题定义，但仍须验证同样的严格 typed response。Box/Choice 不进入本轮付费试验；没有伪称已经做过定位题型。

候选框是 Teacher 给出的固定视觉输入，本轮 Jev 仅选择是否检查已有候选；这不等于 Jev 直接产框。现有模型使用对真/假候选的因果语言模型似然评分，也不等于新建非自回归判别头或实现了 RLCD。

## 原始来源

- https://docs.typesafe.ai/primitives/noul
- https://docs.typesafe.ai/primitives/box
- https://docs.typesafe.ai/images
- https://modelgo.ai/docs/models/text/typesafe-systemone

HTML 抓取本身不调用模型、不读取实验 GT、不产生推理费用。图片性能和运行点的结论始终来自本项目冻结实验，而不是厂商宣称。
