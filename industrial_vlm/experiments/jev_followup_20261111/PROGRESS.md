# Jev 服务器持续研究日志

更新：2026-11-11T01:38:27.593228+08:00

## 阶段01：冻结数据远端对齐（完成）

- 指定服务器实际执行，168/168原图SHA与大小匹配，120query、48正常支持，来源交集0。
- 未读取GT mask或confirmation。

## 阶段02：Teacher来源审计（完成）

- 缓存2.2GB完整SHA、items manifest与decoder权重SHA通过。
- query训练暴露：train96 / val9 / test15；支持train44 / val4。
- 预定义未参与teacher训练子集为teacher_split != train，共24图。它仍是开发诊断，不是新盲测。
- split×全图标签计数：{"('train', 1)": 38, "('train', 0)": 58, "('val', 1)": 9, "('test', 1)": 13, "('test', 0)": 2}。
- 若未见子集缺某类标签，对应FPR/AUROC必须报告不可计算，不能补零。

## 阶段03：固定候选与几何（完成）

- 规则：每图teacher map自身95%分位（linear），>=阈值后union box，不用GT调参。
- 只提取批准120query的概率图，未执行全缓存验证/其他split推理。
- 120query的EXIF orientation全1；明确224方形map→原图→448实际resize/padding转换。
- 60/120候选为整图，60对crop与query PNG完全相同；保留这个退化现象，不事后改规则。
- 第二阶段服务器Pillow12.3.0，第一阶段10.3.0；新query PNG与旧PNG 0/120字节相同。两新臂共享新PNG；旧Jev概率仅历史全图门控对照，不能归因跨阶段差异为裁剪收益。
- 240张新资产（query/crop）及候选坐标、map SHA已落盘冻结；无GT mask读取。

## 阶段04：真正ROI语义复核（准备中）

- 独立目录与新contract，比较原图+候选裁片和原图+重复原图，均2资产、0参考，同区域问题。
- 固定jev-1.14.0、448输入、p>=0.5，120query×2=240接受判断；预检独立2判断。
- 并发3，connect60/read120，自动HTTP重试0，每目标最多3次started且显式恢复；未知完成保留审计。
- 每阶段完成立刻分析并转下一门槛，遇失败先修复；不无限提交付费请求。

## 阶段05及以后

- ROI预测完整并冻结SHA后，再只打开120清单中异常图mask，报告完整60异常分母与60正常误报，另报teacher-unseen子集。
- 后续ROI路由、校准和多支持实验以本阶段证据决定实施细节；保持confirmation隔离。

## 阶段04启动记录

- 服务器自测9类检查通过。独立预检2/2成功，HTTP200，约5.96/6.31秒，无失败。
- 正式240目标已冻结contract并启动，并发3；凭证只经隐藏TTY进入服务器进程。
- 第一阶段旧PNG120/120通过SHA；新旧像素相同0/120，最大单通道差12，已记录03_render_comparison.json。
- 已找到同120来源Qwen base/DD历史score，SHA和source/image/category/label全部对齐；严格logodds>0门控作为历史系统对照。
- 评价器已在服务器通过合成自测，只有formal complete240且合同一致才冻结预测再打开批准60异常mask。

## 阶段06协议已提前冻结

- 路由自测通过，06_routing_protocol.json已在stage05开GT前冻结，协议SHA f71f946972c53eb861859216f368c5e4b1046d859368eca3d3d41c9d0117f5bc。
- 固定不确定区间[.25,.75]且裁片不同才升级；同K哈希随机42/43/44与面积<=.5控制。
- 离线回放逻辑成本为先duplicate再按需ROI，不把提前采集全表的实际费用当节省；不声称在线Agent收益。
- 协调器已在服务器监控：stage04完成后立即stage05评价→stage06回放。失败受每target3started硬限，不无限重试。
- 正在准备stage07开发集交叉拟合校准诊断及stage08多支持种子独立实验；stage07不替代独立数据校准，confirmation继续不读。

## 阶段04—06完成与分析

- formal ROI240/240接受，278started=240成功+38显式502，无unknown/未完成，成功未重发。
- 预测SHA冻结后才读批准60异常mask；confirmation仍未读。
- ROI vs duplicate：保留召回71.67% vs65.00%，正常出框21.67% vs18.33%，全异常meanIoU .069719 vs .068542，IoU>=.5均5%。
- 配对meanIoU差.001176，95%[-.000681,.003213]；召回差+6.67pp区间[0,15]pp，误报差+3.33pp区间[-3.33,10]pp。不能确认ROI净收益。
- Teacher无条件候选meanIoU仅.089529、IoU>=.5仅6.67%，60/120整图候选，候选质量是主要限制。96训练暴露与未见24（只有2正常）限制泛化解释。
- 不确定性路由42次升级，逻辑请求162比duplicate120多35%，召回+5pp、误报+5pp，meanIoU微增且区间跨零；与同K随机控制接近，不宣称Agent净收益。
- 历史Qwen DD门控81.67%召回/11.67%误报更强，但输入/目标/训练不同，只作历史系统对照。
- 下一步在服务器继续：stage07开发集交叉拟合校准诊断；stage08新render共同0shot+43/44的1/4shot支持鲁棒性；stage09同448图像Qwen base/DD对照。

## 阶段07完成：开发集交叉拟合校准与拒答诊断

- 固定category×label分层5fold，每折24，4折拟合1折预测；未在本图拟合自身校准器，固定C1/L2，无超参搜索。
- Jev4shot Brier .141269→.134716，ECE .114917→.072830；AUROC .8885→.8840，校准不等于判别变强。
- Jev1shot Brier .157224→.147661；Qwen底座 .298766→.165891，DD .093157→.094009，指标并非全面改善。
- 50%覆盖率下Jev4shot raw接受准确率96.67%，完整异常自动检出仅58.33%；不能以高子集准确率替代完整召回。
- 这是已看120开发集诊断，协议在stage05后/本校准拟合前冻结；未完成独立校准或confirmation任务。
- 下一步独立stage08：新render shared0及43/44各1/4shot共600；stage09同448PNG Qwen base/DD，共240GPU推理。

## 阶段08正式启动

- 384训练正常来源server逐SHA核验，与120query的source/byte交集0。seed43/44正常支持各48、重合6，保留固定抽样。
- 新render共同0shot120+2seed×1/4shot480，共600接受目标，独立contract b8438cec41443aa2db8fddba2a6107b70ee843d5161fafae362400152137aadf。
- 独立preflight5/5通过，6started/1次502，未混入正式结果。正式600于北京时间2026-11-11 02:04启动，协调器完成后立即评价。
- support代码初次无API自测遇host常量导出差异，已在新support文件独立修复并自测通过，未改已冻结ROI传输器。
- Stage09原生Qwen同PNG对照准备中，所有无forward准备错误合同均可逆归档，不覆盖原生环境或训练。
- Stage10将固定320 vs448全图0shot输入分辨率，448复用stage08新shared0，320单独新120目标。

## 阶段09完成：共同448PNG的Qwen量化基底/DD LoRA

- 原生Python环境实际GPU完成240/240图像判断=480候选forward，实际grid全[1,28,28]、196merged视觉token。
- Loader核验252语言NF4/BF16视觉与DD144LoRA r16/a32/qkvo门均通过，原score._score_image因果与布尔token计分不改。
- 底座AUROC82.4444%/AP85.9678%/Recall23.33%/FPR0；DD AUROC93.4861%/AP94.3288%/Recall71.67%/FPR5%，默认strict logodds>0。
- DD较base AUROC+11.0417pp，严格同PNG及同Qwen评分机制；与Jev模型内部算力/token不等价。
- 准备期flush打印与TF5.14 SizeDict兼容修复均发生在forward前，旧空合同归档保留；正式无代码/依赖/合同变化。
- Stage08与10后续完整后再产联合报告，不把partialAPI结果当正式指标。

## 中段运行与独立审计记录

- Stage04—07独立177项审核全部通过；本地code已回同步为真实server4f7运行版，晚加hostname版本只归档，原预测/合同保持不变。
- Stage09 pooled AUROC DD-base+11.0417pp，category×label配对bootstrap95%[7.4028,14.9729]pp；类别宏+1pp区间跨0，不宣称每类都进步。
- 320独立预检1/1成功，约4.61秒；stage10协调器等待stage08完整评价后立即运行120，避免两个正式API阶段叠加并发。
- Stage08已在连续502后显式有界恢复，所有成功不重发，单target最多3started；运行部分结果不作正式性能评价。
- 新source-disjoint校准/留出数据角色已metadata冻结120+120，与train768/current120交集0，仍属nativeval remainder非盲测，confirmation保持隔离。

## 下一研究阶段已冻结，未使用新评价结果选型

- Stage08输入审计4427检查全过：train/query来源字节隔离、384支持池SHA、90唯一支持、210PNG和600payload哈希/无GT泄漏；不修改运行结果。
- Stage12数据角色固定calibration120和heldout_eval120，各60正常60异常，与train768/current120和彼此来源/字节交集0；nativeval remainder非blind。
- Stage12选择规则在读取stage08指标前冻结：shared0与两seed平均1shot/4shot AUROC比较，距最高<=.01选最少shots；非0固定seed43，输入448。
- 新数据240个响应不参与prompt/size/support/shot选型，先固定最终系统再推理；固定C1/L2校准只拟合calibration120，heldout120一次评价。
- 数据预协议0/1/4允许项用独立amendment追加，原预协议/manifest保持不变；confirmation仍不读。

## 阶段08原协议请求收尾与运维修订

- 北京时间2026-11-11 02:31，原cap3补跑完成：597/600，696 started=597 accepted+99显式HTTP502失败；0 unknown。
- 其余3目标各有3次明确HTTP502终态，不能当作0分，也不能把attempt4追加到原cap3账本。
- 采用独立stage08_rescue运维修订：只允许这3个耗尽目标最多额外2次显式单并发提交；使用同一原contract、PNG、支持顺序、问题、UUID/body和request SHA。原文件冻结为exactSHA，原status继续保持incomplete。
- 完整600后才生成combined结果、评价及complete_with_operational_amendment证书；阶段10/12使用显式适配引用该证书，保留时间与provider变化的局限。
- 凭证提示进程因终端未保持stdin而未能输入；核实无started后，仅终止该等待凭证进程，重新使用可写TTY。未重发任何已接受目标。

## 阶段08完成：两个支持集的完整0/1/4参考比较

- 3项运维修订各第1次额外请求成功，600/600；总699 started=600 accepted+99显式502，unknown0。原6文件exactSHA不变；完整证书在stage08_rescue/completion_certificate.json。
- 独立post审计4437项通过0失败：逐来源、body/UUID/requestSHA、日志、2,000次配对bootstrap、pooled/类别宏/usage/延迟全部一致；救援最多实际total4。
- Pooled AUROC：新0参考83.4583%；seed43/44的1参考89.2083%/89.1389%；4参考91.3333%/91.1250%。4参考相对0参考分别+7.875/+7.667pp。
- 4参考两seed召回均85.00%、FPR均23.33%；0参考召回78.33%、FPR35.00%。固定两个支持bank的方向一致，不能当作广泛支持分布或低误报部署证明。
- 新448结果不覆盖旧seed42/Pillow10.3历史结果。3项救援发生较晚，保留服务时间/路由变化局限。
- 阶段10已立即启动：同120query，原320新请求与新448 shared0比较；评价使用显式修订来源合同，原科学contract和runner不改。

## 阶段10—11完成：分辨率与联合报告

- 新320正式120/120，120started、0failed、0unknown；全部HTTP200，原科学合同与源码未改。新增评价合同在标签前冻结，显式引用08救援combined600，仅取shared0的120条。
- AUROC320为79.0417%、448为83.4583%；配对差-4.4167pp，95%区间[-9.18,+0.08]pp，跨0。Recall66.67%/78.33%、FPR21.67%/35.00%，低误报伴随更多漏检。
- Brier320 .180094、448 .172938；不能只凭FPR更低判定320更好。画布像素比.5102不是精确token/货币成本比，非交错采集未排除服务变化。
- 阶段10独立post审计4498项通过0失败。阶段11联合报告与summary完整发布，汇总960新增API接受目标、240Qwen图像判断；不改旧阶段冻结结果。
- 依冻结选择规则（最大均AUROC的1pp内选最少参考）选择4参考、seed43、448：0/1/4均值分别83.4583%/89.1736%/91.2292%，1参考距最好2.0556pp。stage12系统先冻结，再准备新120校准+120留出，不用新结果反选系统。

## 阶段12启动：新来源校准与留出评价

- system_contract SHA55292eb602330e62f4ca6d7badee45eddf6911dc7a7e71539ded74969c36fc9a；固定Jev1.14、4正常参考、seed43、448、全图Noul，选择规则未因新数据改动。
- 正式240输入独立审计1762/0：240原图SHA及448重渲染一致，校准/留出各120，与train768/current120及彼此source/byte交集0；API payload不含label、role、source ID或文件路径。
- 独立1图预检1/1成功，HTTP200；补审计23/0，明确预检响应不混入240正式结果。正式240在服务器运行，成功不重发，完整预测先冻结，再cal120拟合、系数保存后heldout120评价。
- 这是native val remainder的来源互斥开发研究，不称全新盲测；历史完整val曝光未知，metadata分层本来使用标签，API/评价角色隔离不代表研究者从未知标签。
- 在任何stage12结果出现前，stage13 prepolicy已冻结：主系统固定calibratedJev4/43/448、p>=0.5，raw为预声明secondary；不依据heldout结果选择校准或阈值。confirmation继续未读取，待12全阶段与最终系统freeze门通过。

## 阶段12完成：校准120拟合与互斥留出120评价

- 240/240完整预测，251started=240accepted+11纯HTTP502、unknown0；cal120/heldout120齐，预测与账本先冻结。正式评价及独立复核1550项通过0失败。
- 固定logistic仅cal120拟合，5次迭代：p_cal=sigmoid(1.56720531322167×logit(clip(p,1e-6,1-1e-6))−0.15632219069119976)，系数保存后才开heldout标签；未选超参。
- Heldout raw→cal：AUROC92.8194%与AP94.2361%不变（正斜率单调映射），Recall88.3333%=53/60不变；FPR18.3333%=11/60→10%=6/60，配对差−8.333pp，95%[-15,−3.333]pp。
- Brier .1242283→.1077424，差−.0164859，95%[-.023661,−.008720]；logloss .404792→.348591，差−.056200，95%[-.076143,−.033828]。ECE .158333→.110424，但差区间[-.070821,+.008433]跨0。
- 校准FPR的Wilson95区间约[4.664%,20.149%]，不承诺产线低误报；50%/75%回顾覆盖率两版本错误率均3.333%/8.889%，完整错误率15%→10.833%。
- stage12仍是nativeval remainder开发研究，不是最终confirmation；bootstrap固定cal拟合，不含拟合/provider/bank总体不确定性。主系统政策于正式12首API前9.526秒冻结，未根据这批收益选择calibrated。
- 下一步按已审源码：冻结最终bundle→独立SHA/系数门审计→一次性input/label broker开启原固定240→独立input-only审核→API→全预测冻结后评价；不重新抽样、不调阈值。

## 阶段13一次性确认准备

- 最终系统bundle SHA72ab9b8706cdf77c102f7fb53c372e3a183ecdf69e67599d9d58732838fd392e；独立42项通过0失败，绑定stage12预测/校准/评价及源码、依赖、主calibrated/secondary raw、阈值0.5。
- API runner源码ba312df2a904122e1a9dab72c62fe01d5beed40dbe507e1091bc8c21e8366bc4已通过producer11组与独立7组server合成审计；不在本机执行。
- 北京时间03:03左右，所有前置门通过后唯一一次broker开启原固定confirmation元数据和原图，分离无标签inputs与0600布尔labels，不读取mask；240唯一、各类各标签10、来源/字节与train/current120/stage12交集0，未换样。
- 状态prepared240/0started；待input-only独立审核后发API。推理只读输入PNG/category/固定支持，不读protectedlabels或旧含标签清单，标签评价等全240预测及映射冻结后。
- 持久预算global720started、target3、累计failure30停止新提交（已预留inflight最多drain至32）、连续3暂停；timeout/transport/200无效响应永久unknown硬停且不重发。
- 输入独立审计1517项通过0失败，未开protectedlabels/legacy/mask，0请求快照通过；父随后安全输入隐藏key正式启动确认240，无额外confirmation预检或重新抽样。评价代理独占全240完成、锁释放与3秒缓冲后的正式评价及独立post，避免重复评价。

## 阶段13完成：一次性最终确认240

- 北京时间2026-11-11 03:14:16，正式240/240全部HTTP200，240started/0failed/0unknown；全预测与固定映射冻结后才由评价端读取布尔标签，GT mask未读。正式评价、独立post和补充统计于03:14:44完成，548项通过0失败。
- 主calibrated：AUROC90.1840%、AP91.6971%、MacroF1 .820681；TP102/FP25/TN95/FN18，Recall85%=102/120、FPR20.8333%=25/120。Recall Wilson95[77.53%,90.30%]、FPR[14.53%,28.95%]。
- 同次raw secondary：AUROC/AP相同，TP106/FP30/TN90/FN14，Recall88.3333%、FPR25%。校准少5误报但多4漏检，不能称全面改善；主方案不因该结果切换。
- 主−raw：Recall−3.333pp，95%[-6.667,−.833]pp；FPR−4.167pp，95%[-7.5,−.833]pp；MacroF1差+.004833区间[-.016533,+.029791]跨0。
- Brier .140102→.126974，差−.013128，95%[-.018390,−.007617]；ECE .123→.072835，差−.050165，95%[-.072121,−.007087]；logloss .440692→.399576，差−.041115，95%[-.058343,−.022773]。本确认数据支持概率损失改善，同时存在召回代价。
- 固定50%/75%回顾覆盖率下主错误率5.833%/10.556%，完整240错误率17.917%；不将拒答当正常，不以子集准确率替代全异常召回。
- confirmation已被本轮一次性查看，不能再据其调prompt/seed/size/阈值并称同轮新盲测。只评固定Jev，无确认集Qwen跨模型优越性证据；均衡先验/物理近重复/预训练污染和固定拟合bootstrap限制保留。
- metrics SHA2838095b23bba31a4c233be7a385f3e010c4af7a978673de40bccf10f6e63976；completion certificate415f40f52644c0d716c3f563add5d4d03c540c37cbd249c197b6b8a80d899241。

## 最终交付与全链核验完成

- 阶段01—13全部完成，正式1440 logical targets、1588started、148明确失败、unknown0；三组来源/原图SHA并集600，Qwen240判断/480候选forward另列。
- 新增CONTINUOUS_RESEARCH_COMPLETION.md及stages/14_completion_report_manifest.json，逐阶段中文总结、所有负结果、区间、数据角色、代码/结果SHA与四组图索引完整；旧科学产物不覆盖。
- 总报告SHA f5eacc0ba5f726ab175179fb15ca00e6eb0aa92ae8c3898c66f7ec099287c755；manifest SHA dc28712c8a894ba955dde5b65a7fdc4fc65a34da75f0b997fd7bd3b79bc2fdd1。服务器最终核全部manifest来源链一致，status.json已标complete。
- 图PNG/SVG/provenance均在服务器生成、全图视觉QA通过，非保护报告/代码/审计/图表同步本地；protectedlabels与legacy确认清单未复制。所有测试统计在指定服务器完成。
- 本轮按文档到一次性确认终点结束；后续候选器改进/更多bank/provider重复/生产先验与线上Agent需另立新开发和新留出协议，不能在已看确认240上继续选型。
