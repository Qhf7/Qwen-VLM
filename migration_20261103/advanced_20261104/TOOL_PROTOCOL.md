# 单次 ROI 工具调用协议

## 问题与归因边界

检验 Qwen3-VL-8B 能否根据当前 RGB 自主选择是否查看局部、生成可执行坐标，并利用工具返回的真实局部图像完成异常判断。本实验使用完整原生 Qwen 视觉 + 语言路径，沿用官方 processor、MRoPE、DeepStack 和历史 source/dtype gates，并逐文件复核基座 manifest SHA。它是独立的原生视觉路线，不能据此称为压缩桥接 C/D 的 LoRA 增益，也不与 C/D 作为等输入机制消融混报。

当前压缩路线仅把 ImageBind CLS + teacher map 编为 19 个 token，训练答案只有 Yes/No。该特征接口没有经过多图/工具请求训练。把原生视觉 token 或第二组桥接 token 临时接上，不能直接证明原有 LoRA 的工具能力。因此先执行可审计的原生工具基线；若实际调用和证据覆盖有价值，再给压缩桥接做独立工具 SFT 与相应零初始化/重载门禁。

## 固定决策与工具契约

- 输入只有当前 RGB。类别、二值标签、mask、teacher/student map 均不传入推理函数。
- 第一轮原图固定 448×448，模型贪心输出一项严格 JSON：{"action":"crop","center":[x,y]} 或 {"action":"answer"}。
- 坐标为原图归一化 x/y，范围 [0,1]，最多调用 1 次。crop 边长固定为原图宽/高的 1/3，总面积 1/9；边界采用平移，保持面积。
- JSON 没有额外字段、重复键、代码围栏、前后解释、布尔坐标、NaN/Infinity。非法请求拒绝执行并记录，不能默默换成 teacher ROI。
- 有效请求实际裁剪原 RGB，使用分数边界 + bicubic 输出 448×448。第二轮把原图、工具返回图像、执行坐标交给模型；以 Yes/No 开头回答一条可见证据。模型原始请求和答案全部保留。
- 模型选择 answer 或请求无效时，固定双图控制臂提供显式标记的原图副本以保持视觉 token 上限。该协议为 bounded one-call workflow，不声称多步 agent 规划或深层推理已提升。

## 配对控制

所有条件在同一图像上复用完全相同的第一轮模型规划结果，以省 GPU，同时分别记录完整 planner 等预算成本，以及控制臂实际部署时无需 planner 的成本。区别只在执行策略：

| 条件 | 实际执行 | 最终视觉 token |
|---|---|---:|
| no_tool | 仅原图，模型规划结果不执行 | 196 |
| duplicate_original | 不裁剪，原图 + 原图副本 | 392 |
| center_crop | 固定中心，面积 1/9 | 392 |
| random_crop | 固定 path+seed 随机中心，面积 1/9 | 392 |
| model_tool | 执行模型有效请求；否则原图副本 | 392 |

final prompt 一致；工具状态文字和坐标按真实执行填写。坐标文本 token 数会变化，逐条记录总 prompt token，不能宣称等 text token、等 FLOPs 或等 wall time。no_tool 是较低视觉预算基线，中心/随机/副本是视觉预算匹配控制。第一轮 196 个视觉 token；生成上限 64，最终答案上限 32，greedy/no sampling，完整 BF16/SDPA，不量化。

推理函数 run_case(... image_path ...) 没有标签/类别/mask 参数；外层验证驱动只使用 cache 的 path/category/label/split 做预定子集和事后指标。cache CLS/maps 不进入推理，在载入元数据审计后释放。

## 工程预检和正式验证

先运行 CPU 自测，再在共享 GPU 锁内执行 8 张分层验证工程预检。工程门禁为 source/dtype gate、单图/双图 token/shape gate、有限 logits、已保存完整请求和答案；不能把小样本 AUROC 当作收益。之后固定提示和 crop 参数，运行全部 val380。验证集已反复用于开发，该结果只属开发验证。本程序不打开旧 test456 或外部 MVTec。

    python tool_workflow.py --self-test
    flock runs/migration_20261103/gpu.lock /root/autodl-tmp/conda-envs/qwen3vl2/bin/python -u tool_workflow.py --run --limit 8 --out runs/migration_20261103/advanced_20261104/tool_preflight
    flock runs/migration_20261103/gpu.lock /root/autodl-tmp/conda-envs/qwen3vl2/bin/python -u tool_workflow.py --run --limit 0 --out runs/migration_20261103/advanced_20261104/tool_val
    python tool_audit.py --run runs/migration_20261103/advanced_20261104/tool_val --out runs/migration_20261103/advanced_20261104/tool_val/audit.json --with-masks

远端平铺部署需要已有 migration_roi_eval.py、migration_metrics.py、migration_native_parity.py、visa_pipeline.py、dec_big.py；当前远端实验环境已有这些模块。GPU 命令必须等结构监督阶段完成后再排队。输出目录互斥，失败目录保留，不能覆盖旧预测。

## 报告分母与判读

- AUROC/AP：最终答案位置 logit(Yes)-logit(No)，同图同序配对；完整 overall/per-category/category-macro。
- 生成：macro F1、recall、FPR、accuracy、invalid rate。非法生成保留，按现有 migration_metrics 规则计错。
- 工具：planner 无效率、拒绝率、真实调用率。所有样本计入分母，不能只展示成功调用案例。
- 成本：planner、final 和含图像载入/哈希+planner+final 的等预算时间，以及控制臂省略 planner 的实际部署时间；每轮视觉/prompt token，规划与最终生成显存。模型载入时间另报，预热/驱动条件见 run.json。
- tool_audit.py 在整个 inference run 已完成、行数/配对/原图 SHA/请求重算/指标复算通过后，才可读 GT mask。报告所有异常的 crop hit rate、缺陷像素保留比例及实际调用异常子集的条件均值。未调用的异常 coverage 记 0；正常样本单独计数。224 二值 mask 几何覆盖不代表缩放后可辨性、注意力或解释忠实度。
- 除非模型臂在匹配控制下取得可复核收益，否则只表述完成工具调度系统及能力边界，不能表述推理性能提高。一次确定性推理不需把42/43/44当训练seed；random_crop 当前为42，若需稳定性，再固定追加随机裁剪seed并完整报告。

## 当前交付状态

本地已实现可执行程序与严格 schema/几何 CPU 自测。没有在本子任务中运行 GPU、打开任何外部数据结果或声称收益；远端工程预检和完整验证需主队列执行。
