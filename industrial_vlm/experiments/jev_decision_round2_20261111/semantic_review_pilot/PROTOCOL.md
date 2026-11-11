# 可见缺陷空间命题支持：最小诊断协议

状态：实现与协议冻结阶段。尚未构造真实标签、未打开 online24 的 mask、未发送语义诊断 API。只有服务器完成 online24 gate/final 全部预测冻结、并得到明确的开发 GT 授权后，才运行 feasibility。

## 问题与边界

固定 jev-1.14.0，使用已经验证的 Noul 请求。问：**Does the image provide visible evidence supporting the claim: a visible defect is present in [grid]?** 输入为原图448 PNG、声明与同一固定 rubric。对照为相同声明和 rubric 的无图请求。

真实标签来自 GT mask 的空间位置，称为 **mask-derived spatial proxy**。它只给出部分空间诊断，不是人工核验的完整可见事实 gold，不证明缺陷类型、解释质量、成因推理或因果关系。API 不看 mask，也不要求它猜测不可见的“标注在哪里”。

## 固定来源与时间顺序

1. prepare 只读 online_route_pilot/contract.json 的24个已冻结 source/hash 及 query448 PNG；不读 labels/mask。记录来源 contract、protocol、runner 与依赖 SHA。
2. feasibility 要求 online prediction_freeze.json 存在，并核验 gate/final contract、predictions、attempts SHA 和完整父链；还必须显式传入 --authorize-development-gt。
3. 此时仅对这24个开发来源构造空间 proxy，完整保留所有24的分母和所有排除原因。不增补24以外来源，不动stage13或native test。
4. 合格来源按预固定 source hash 排序，取前 min(12, eligible_count)；冻结 true/false 声明、request hashes 和 proxy label 独立文件后才允许请求。
5. 全部语义预测再次冻结后生成诊断指标；不按结果改格子、样本或阈值。

## 标签构造与排除规则

- 只给异常来源构造 paired 声明；正常来源仅保留 all-negative 描述性分母，不占未定义的请求预算。
- mask二值化；如果多通道取任意通道非零。8邻域单连通、非空；多缺陷连通分量直接排除。
- 原始图像使用固定3×3等分网格，按原始像素中心落格；不对 mask 做插值或 resize。
- dominant格须含至少80%的GT像素，其余各格不超过10%。
- 每一个GT像素中心须位于dominant格四条边以内至少2原图像素；这比80%规则更严格，避免跨格或边界含糊。
- false格必须无GT像素，并与true格Chebyshev距离>1（不相邻、不对角接触）。若true为中间格，则无合格false格，排除。false格按预固定hash选择。
- true声明为“该格有可见缺陷”；false声明为选定空格有可见缺陷。真伪只是mask空间proxy，背景/未标注异常和可见性仍可能使proxy与人的视觉语义不一致。
- 请求隐藏 true/false 映射：每图两条声明的a/b顺序由固定hash决定。state只允许 claim 与有图条件的资产名 query；不得放label、mask、GT计数、候选框、teacher map或旧概率。

## 请求与预算

每个合格来源最多4个成功target：true声明有图、true声明无图、false声明有图、false声明无图。最多12来源／48成功target／144started；单target最多3次，rolling concurrency=3。累计10次失败停止新发，最多再排空2个inflight；连续3次失败也停。任何超时、未知完成或无明确非200失败结果硬停，不重发。只有明确失败且新的一次显式调用带 --retry-failed 才可重试。无自动HTTP retry，不保存或输出key。

无合格来源时报告真实零可行性，发送0 API；不能放松规则或补挑图片。

## 输出与解释

- 固定0.5，仅描述图像条件对proxy的accuracy；输出每图true-minus-false支持分离度。
- 无图条件描述措辞／位置prior与缺失证据行为；它缺少视觉证据，不把它当与有图完全相同的任务准确率。
- 逐source报告图像分离度减去无图分离度，不把4次请求当4个独立来源。
- 保留完整24分母、所有mask排除理由、eligible数与cap后的selected数、逐targetledger、usage缺失与unknown状态。
- 这是此前看过的开发子集；小样本、strict筛选和mask proxy限制使结果只能支持下一步可行性判断。

## 服务器命令顺序

脚本位于 /root/autodl-tmp/industrial_vlm/experiments/jev_decision_round2_20261111/semantic_review_pilot/code/semantic_review.py；解释器使用 /root/autodl-tmp/conda-envs/qwen3vl2/bin/python。根目录为同级 semantic_review_pilot。

1. --self-test：只用合成数组，零API、零真实GT。
2. --prepare：冻结协议与label-free24来源；零GT。
3. online全部预测冻结且获GT授权后：--feasibility --authorize-development-gt。
4. --audit-inputs：校验目标、哈希、state allowlist与ledger。
5. 预算批准且凭证以环境变量或隐藏交互进入进程后：--run；仅明确已失败target可新调用 --run --retry-failed。
6. 全部settled后：--evaluate，先冻结预测再汇总proxy指标。

本文件的命令步骤不表示feasibility或API已执行。
