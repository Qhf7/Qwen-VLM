# Jev 第二轮：概念、已有证据与代码对应

日期：2026-11-11。本文来自现有源码只读审计与指定 AutoDL 的紧凑源码核验。不重算旧指标、不读取确认集标签、不运行新模型或 API，不修改已完成的 01—13 阶段。厂商与网关当前能力以本轮单独保存的官方资料及实际小预检为准。

本轮官方证据来自服务器 HTTP 抓取快照，非网页搜索引用：reference_docs/manifest.json 记录 UTC 2026-11-10 19:57:35（北京时间2026-11-11），四页均 HTTP200。Noul 明确没有独立 confidence；Box 为 top/bottom/left/right 四边 [0,1]，原生图片/Box 需危险功能 header；图片宽高各280—1568。ModelGo 文档同次抓取仍写 image assets、Box、危险 header 不支持，与本项目已经成功的图片 Noul 实测冲突。因此只将已实测 Noul 视作该网关能力，不由原生文档推断 Box 可用。

原始页面：docs.typesafe.ai/primitives/noul、docs.typesafe.ai/primitives/box、docs.typesafe.ai/images、modelgo.ai/docs/models/text/typesafe-systemone。HTML SHA：noul=a2be4e49a8ea6890a0c29287299a67873d975ef6b55765b283547698d5cef130；box=a6a2bbacf2c5edfd0bff7d2c2336489da73db088b364fe22aa7e4a6320871f6c；gateway=d9be82777a81902486db6dab78c771a602ab15bbafc9a2de90d757f12cfb5868。完整图片页 SHA 与文本见 reference_docs/manifest.json，四份 html/txt 均保留。

## 1. 项目有四条不同的技术路径

| 路径 | 真正做了什么 | 当前接口（相对 Qwen-VLM 根） | 不应混称 |
|---|---|---|---|
| 原生 Qwen 文本生成 | 图像经官方 processor，LM 自回归输出 Yes/No 或 JSON | migration_20261103/advanced_20261104/tool_workflow.py:169 的 planner，调用 model.generate | 不是 TypeSafe 原生结构化头 |
| 原生 Qwen 固定候选评分 | 同图同 prompt，分别算 true/false 的条件 log-likelihood，以差值排序 | industrial_vlm/src/industrial_vlm/score.py:226 的 _score_image；:31 的 normalized_score | 无自由文本解码，不等于架构换成非生成分类头 |
| Jev 全图 Noul | ModelGo /v1/systemone，图像 assets 与结构化问题，接收异常概率 | jev_followup_20261111/code/support_robustness.py:103 的 payload；roi_api.py:316 的 response_probability | Noul 成功不证明 Box/Choice/Score 已实测 |
| Jev ROI 与离线路由 | Teacher 先给固定框，Jev 判断保留/删除；已有响应上回放升级策略 | roi_api.py:283 的 request_payload；extract_teacher.py；routing_replay.py | 不是 Jev 直接产框，也不是在线 Agent 或训练好的 tool policy head |

实验代码位于 industrial_vlm/experiments/jev_followup_20261111/code。服务器项目根是 /root/autodl-tmp/industrial_vlm。迁移源码本机存在，不假定其服务器部署路径与本机相同。

## 2. Noul、Choice、Score、Box 的任务边界

### Noul：问题定义的二元判定

已经验证的具体问题是“query 中是否存在至少一个可见工业异常”。support_robustness.py:42 固定 INSTRUCTIONS 与 CRITERIA，:103 组装 state.query、category、normal_references、assets 和 questions.anomaly.type=noul。roi_api.py:316 严格要求响应 model 匹配、question 恰为 anomaly、类型为 noul、概率有限且位于 [0,1]。

例如 p=.8 是这一个问题在当前图像、正常参考与文字定义下的异常分数，不是“八成所有问题都能判断”，也不是缺陷面积 80%。全图二元判断不能自动替代多缺陷定位、类别识别或严重程度。

### Choice：有限选项的单选决策，项目尚未接入 Jev Choice

可设计动作选项 answer、inspect_top_left、inspect_middle_center 等，也可定义九宫格位置。正常/无目标另配存在性 Noul，或在动作定义中明确 answer。单题 Choice 不能同时返回多个独立缺陷；九宫格分类准确率不是框 IoU。

现有九宫格在 migration_20261103/advanced_20261104/structured_geometry.py:15：LOCATIONS 是 top/middle/bottom 与 left/center/right 的组合；:26 的 location_for_box(box) 根据框中心映射一个格子。它是几何标签与生成校验函数，不是 Jev Choice API。

例：[0,0,200,200] 的中心为 (100,100)，属于 top_left；细长缺陷即使跨多个格子，仍只得到一个框中心格子。未来 Choice 选位置只能提供粗候选或裁剪动作，不能直接宣称精确定位。

### Score：事先定义等级的有序评分，项目尚未接入 Jev Score

可以定义 0=无可见缺陷、1=轻微可疑、2=明确局部缺陷、3=严重结构异常。等级含义、证据要求和评价标准应先冻结。等级分布或其集中程度不能直接当真实尺寸、任意连续回归值或正确率。

特别注意：industrial_vlm/src/industrial_vlm/score.py 的 score 是 Qwen 候选 token 似然，不是 TypeSafe 的 Score 题型。当前没有已经验证的 Jev 严重程度输出。

### Box：直接定位，不同于判断已有候选是否有效

本轮服务器原生文档快照描述 Box 为 top/bottom/left/right 的归一化单框，但 ModelGo 同次文档仍声称不支持。必须实际核该网关是否接受相同 schema，不能凭原生文档或 Noul 的 HTTP200 就断言本渠道 Box 可用。

若返回 top=.2、bottom=.5、left=.1、right=.4，转换后的 xyxy 是 [.1,.2,.4,.5]，不能照字段顺序直接装数组。asset 引用名、坐标约定、目标不存在的语义、confidence 与错误格式都要依据渠道契约。若 Box 总返回一框，应配独立存在性 Noul 决定是否采用，并披露正常图出框率。

## 3. 原生 tool_workflow 能规划一次裁剪，尚非训练好的策略头

源码：migration_20261103/advanced_20261104/tool_workflow.py。

| 接口 | 真实行为与限制 | 例子 |
|---|---|---|
| PLAN_PROMPT，第26行 | 单 JSON：action=crop 配 center=[x,y]，或 action=answer | center=[.5,.5] 请求原图中心附近 |
| parse_request(text)，第68行 | 拒额外/重复 key、bool 坐标、NaN、代码块、超界坐标 | 带 label 字段的 planner 输出被拒 |
| crop_bounds(center)，第90行 | 宽高各为原图 1/3，面积 1/9，边缘平移保持框在图内 | [.5,.5]→[1/3,1/3,2/3,2/3] |
| build_final_inputs，第119行 | 只接 RGB 与验证后的请求，返回原图和最多一个裁剪/重复图 | 无效请求被拒，以重复原图受控回退 |
| planner，第169行 | 实际调用 model.generate 产生工具 JSON | 语言生成规划，不是新分类 head |
| run_case，第194行 | 对照 no_tool/duplicate_original/center_crop/random_crop/model_tool | 部署需计 planner 与最终判定两段成本 |

最终 score_definition 是 logit(Yes)-logit(No)。每幅 448 视图经原生 processor 核为 196 个 merged 视觉 token；两图变 392。加入额外图的收益不能都解释为动作策略更聪明。代码区分是否支付 planner 开销的时延。

真正的 tool policy head 需要另设可审计训练协议：输入可用状态 s，输出有限动作 a；监督与 reward 不能把 GT 或 oracle 未来收益当推理输入。应固定行动预算，比较非法动作率、调用率、净任务效用、时延和实际费用，并保留 duplicate/center/random 的同预算控制。01—13 没有训练/验证这种 Jev 或 Qwen 专用策略头。

## 4. Qwen 候选评分不是架构上的非生成头

score.py:27 声明共同 JSON 前缀及 true/false 两候选。candidate_plan 在第78行通过 fast-tokenizer offset 核 boolean token 边界。_score_image 在第226行对两候选各一次 model.forward，只累加 boolean token 的条件 log 概率，共同 JSON 前缀、闭括号和结束符不计分。

定义为 s=log P(true tokens|image,prompt,prefix)-log P(false tokens|image,prompt,prefix)，再以 sigmoid(s) 得 anomaly_score。例：两候选 log-likelihood=-.2 与 -1.2，则 s=1，二候选归一化约 .731。这不是未经校准就能解释为“正确率 73.1%”。

该路径无需自由文本解码与解析，适合稳定排序/AUROC/AP；底层仍是生成 LM 的词表 logits 与因果 token 概率。没有把 Qwen 变成 TypeSafe System One，也没有添加独立分类 head。Jev 内部架构、训练与算力不可由这段代码推出；同 PNG 不等于同内部 token、算力或校准。

## 5. parse_prediction 与坐标协议

industrial_vlm/src/industrial_vlm/metrics.py:45 的 parse_prediction(raw_text,require_boxes=True) 接受单 JSON，可有一层 JSON 代码块。is_anomalous 必须是真 bool，字段仅 is_anomalous/boxes。第28行 validate_boxes 要求每框四个有限数、正面积、范围 [0,1000]；不自动 clip、修反序、截取散文或把字符串 true 转 bool。

例如 is_anomalous=true 且 boxes=[[100,200,400,500]] 结构有效；is_anomalous="true" 无效。正常判断附非空框语法仍可有效，语义矛盾另计，不能悄悄删框改善指标。

structured_geometry.py:57 是另一套格式：第一行 Yes/No，下一行含 bbox/location 的 JSON；异常需整数框与几何一致的 grid3，正常要求 bbox=null/location=none。这不能与 industrial_vlm JSON parser 直接互换。

### 原图、448 画布与 [0,1000] 不可混用

extract_teacher.py 从 224 热图取 union95，分别保存 bbox_original_norm 与 bbox_448_norm。448 使用等比缩放、居中白填充：x448=(x_original_norm*resized_width+pad_x)/448，y 同理。

例：原图 800×400 缩成 448×224，上下各 pad112。原图归一化框 [.25,.25,.75,.75] 在画布是 [.25,.375,.75,.625]。若 Box 输出画布坐标，需减 pad 再逆缩放才回原图；直接乘1000是画布框，不能与原图 GT 直接算 IoU。越界、落白边、反序与空框应预声明失败策略，不静默修正。

metrics.py:82 的 union_box_iou 比较各自最外包矩形，不是 mask/polygon union IoU，不能称像素分割精度。

## 6. 两种校准要分开

score.py:165 的 calibrate_normal_threshold(records,target_fpr) 对 normal 的 logodds 降序，允许错误数=floor(target_fpr*Nnormal)，取对应顺序统计量为阈值，预测严格 s>threshold。第307行 CLI 限制输入 meta.data_role=validation，并绑定 score/data/subset/model 契约。

例如 60 正常、目标经验 FPR=.05，则允许3个；取第4高为阈值，严格 > 排除边界 ties，所以经验 FPR 可低于5%。这只控制该校准样本的经验 FPR，不保证未来≤5%，也不是概率校准，不能用确认集再挑阈值。

Jev 的 isolated_calibration.py:433 先 clip(p,1e-6,1-1e-6) 再 logit，第483行只用 calibration120 拟合固定 L2 logistic。完整240预测先冻结，系数先保存，再读 heldout 标签。final_confirmation.py 将同一映射预声明为主系统，raw 为次要对照，阈值仍 .5。概率校准可能少误报也多漏检；不能按确认表现切换主系统。原确认240已查看，第二轮不能用它调方案后再称新盲测。

## 7. 已做与未做

| 内容 | 事实状态 | 第二轮用途 |
|---|---|---|
| 固定448、同query、0/1/4正常参考 Noul | 首阶段360和新render/seed43/44的600均完成 | 历史冻结对照，不重新宣称尚未做 |
| ROI vs matched duplicate | 固定 teacher 框、双图 Noul 复核240已完成 | 复用审计方法，不称 direct Box |
| 独立校准和一次确认 | cal120/heldout120/固定确认240完成 | 保留结果，另定第二轮开发与新确认门 |
| 原生自由生成裁剪 planner | 本机有源码与受控流程 | 需核服务器部署版本和旧产物，源码存在≠新实验完成 |
| Jev 直接 Box | 01—13 未发送或接收 Box 题型 | 先能力预检，渠道不支持则记录阻塞 |
| Jev Choice 九宫格 | 未接入与验证 | 能力通过后作粗位置/有限动作，需存在性与预算控制 |
| Jev Score 有序严重度 | 未接入与验证 | 先 rubric 与标注一致性，不当精确尺寸回归 |
| 真正训练 tool policy head | 未做 | 另需动作监督、损失、训练与独立评价协议 |
| RLCD 等强化方法迁移 | 没有项目内收益保证 | 仅待证研究，需匹配架构、数据、reward、动作空间与算力 |

RLCD 不能仅凭“判定模型”三个字直接迁移到 Qwen：需要可访问且可训练的输出、可计算的行为概率或明确替代目标、恰当 reward 与独立验证。API 型 Jev 的能力、源码/权重开放程度与训练接口也不可假设。先读当前官方材料，不能把厂商方法名当本项目已做的强化学习。

## 8. 下一步：先能力预检，再配对小试验

### A. 冻结能力契约，不消费新 GT

1. 保存当前原生及 ModelGo schema、端点、模型、图片限制、Box 字段/空目标语义/错误格式、多问题支持，记录抓取时间与原文 SHA。
2. 从已批准且已看的开发来源按 hash/类别事前固定极小清单，用既有448 PNG；不拿确认240作新开发，不按性能挑图。
3. 最多1个现有 Noul 通道检查，再最多1个直接 Box 请求。若同请求 Noul+Box，先核渠道明确支持多问题，并严格验证返回 question 集合。
4. 不支持类型、400/403、invalid typed 均原样记录并停扩展；不反复猜 schema 碰200后隐藏尝试。凭证只环境/隐藏输入，日志不含 Authorization 或 base64。
5. 旧 roi_api.send 的 Danger-Zone header 只证明其用于 Noul 图片请求；不证明网关透传所有上游 header 或支持 Box。不改旧代码试探。

### B. 能力通过再冻配对设计

一个保守小试可用24张已看开发 query（12类各2，事前 hash 固定），比较同任务预算的定位方案。两臂返回题型必须一致：不能一臂 Noul+Box、另一臂纯 Noul，却把差异全归因输入证据。若比原图+候选 crop 与原图+duplicate，应两臂同问存在性 Noul 与同一 Box，并分别报判定和定位。

建议预算：24query×2arms=48逻辑目标，预检另账；预检 rolling1，正式 rolling≤3；每目标≤2started、全局≤96started、累计失败10停新提交并 drain、unknown 硬阻重发、重试必须新显式 invocation。真实账单未知，记录请求和 observed usage，不估钱。

定位预测冻结后才读取该批准开发来源的既有 GT；仍不读取13确认标签。报全部异常分母的 IoU/IoU≥.5、全部正常出框率、异常保留召回、Noul AUROC/AP、时延和 usage。不能只在模型说异常的少数图上报 conditional IoU。能力预检不读任何 GT、不报定位精度。

### C. CLI 草案，仅设计，尚非可运行功能

依次建议提供：direct_box_probe.py --freeze-capability、--prepare、--dry-run、--probe-noul、--probe-box、--freeze-formal、--run、--audit、--evaluate。每步显式使用 ROUND2 根和冻结渠道契约，prepared 输入绑定已批准开发契约。只有真实实现、服务器自测、独立输入审计及既有预算授权均满足后才发请求。父流程将结合官方核验决定实现，不在本机替代测试。

## 9. 本次服务器源码核验

hostname=autodl-container-132444827f-9c594038，SSH34855。仅源码读取，没有数据、标签或预测读取。

| 文件（相对 industrial_vlm 根） | SHA256 |
|---|---|
| src/industrial_vlm/score.py | b2dd6dc5330804f884560c1cd35c238f253d2b8b30b73a9e14458b9e878928e0 |
| src/industrial_vlm/metrics.py | 4a9c7904499c1e12f915810bafdf283ca1129b67bc85706ce413ab90e7e094cc |
| experiments/jev_followup_20261111/code/roi_api.py | 4f7b5ed1f703b3751f9fd4bb8c86131d684cec0bed3a9011f5108621e7aa9e01 |
| experiments/jev_followup_20261111/code/isolated_calibration.py | d8489c953ec4cb5e9d7804643d35f43d5ad0801211abc118bcc8e0003fec37d1 |
| experiments/jev_followup_20261111/code/final_confirmation.py | ba312df2a904122e1a9dab72c62fe01d5beed40dbe507e1091bc8c21e8366bc4 |

迁移 planner 本机存在。本次只探查的两个服务器候选路径未见文件，不能据此称全服务器不存在，部署位置需单独定位。本文未运行本机测试、推理、统计、训练或 API。

## 10. 本轮已开始执行的真实门控试验

新增 code/online_route_pilot.py（冻结源码 SHA52b53e8d95351ab28ea4660302d2426733f9c44b732179303390542d5d624ca7），已在指定服务器 compile、自测并完成 gate 输入独立223项审计，随后父进程安全注入凭证启动真实 Noul 请求。本节只记录已发生的步骤，不提前写未完成结果。

从原已看开发120按 sha256(source_id) 排序固定前24，不看标签。13个候选 crop PNG 与原图字节不同，共同 K=min(8,13)=8；gate、固定随机seed20261111、旧duplicate不确定性、面积四策略使用同13个eligible。gate只见原448图和候选bbox，不见crop RGB、旧分数、GT、路径标签或13确认数据。

待24 gate全部完成并冻结后，四策略固定top8，其ROI并集≤13；实际执行固定crop并核PNG字节，再真正新发24 duplicate与并集ROI。总新成功目标≤61（方案上限72）、每目标3、全局started上限216、累计失败30停新发并drain≤32；unknown永久硬阻，不重复已接受目标。所有策略共用同一批新潜在结果，区别于旧离线结果回放。

评价只在完整gate/final预测冻结后派生批准24开发标签；仅复用旧阶段05已冻结candidateIoU，不开新mask。不读取13确认标签。主区间按正常/异常标签分层2000次bootstrap，类别仅描述。真实union采集、单policy取证（含额外duplicate对照）、理论最简部署请求数分开，不把合并采集的墙钟时间假装每个策略单独在线时延。

Gate24已全部真实HTTP200完成，24started、零失败/未决。随后完成route冻结和13个真实crop重执行核字节，final输入独立审计前0started。最终新请求为37=24duplicate+13ROI，总真实成功目标61。24中11为整图候选，严格eligible排除相同PNG，未为得到更好候选替换样本。gate top8与随机/历史不确定性/面积top8的交集分别6/5/5，四组并集13；这些是无标签选择结构描述，不是准确率或收益。
