# Qwen pre448 GT / KD 地图替换 ROI 选择器：最小运行协议

## 本轮实际执行约定（2026-11-04）

学生map已由pre448 GT与同feature KD(0.5)两个已选best1200头导出，各380图，复用一次视觉前向；不存在新的训练或按结果选臂。实际continuity replay采用3个类别各正常/异常一例，共6例（而非下文建议的9例），直接复用记录的teacher crop bbox。六例的Yes/No logits、score、生成token IDs、文本以及实际geometry/token预算全部exact相同，最大score差0。旧中心/随机/teacher完整prediction已按path/label/category/split/cache_index核验，并可独立重算原指标。该replay是有限样例的路径连续性验证，不是对全部380对照的逐张重跑。

推理分两阶段：纯Qwen视觉+FP32head导出概率图；释放视觉后用完整native Qwen处理原图+student crop。新学生臂不读teacher cache/模型、ImageBind或GT mask。旧teacher bbox仅用于上述独立replay。另有独立CPU post-export map_metrics脚本在预测完成后读取GT，用于指标及ROI覆盖诊断；它不参与map生成、框选择或语言输出。


日期：2026-11-04。状态：**审查协议，尚未运行学生 ROI**。授权范围是只读既有源码与 `roi_full/run.json` 并新增本文件；本次未SSH、未运行GPU、未修改脚本或其他报告。

## 立即采用的最小矩阵

新增两臂：`original_student_pre448_gt_roi`、`original_student_pre448_kd_roi`；每臂完整相同val380、两图。第一图是原始RGB直接bicubic到448，第二图从原始RGB按地图峰值裁剪并输出448。原生Qwen只看这两张RGB和既有统一QUESTION，模型参数冻结。比较对象为既有 `original_teacher_roi`、`original_center_crop`、`original_random_crop`；三图reference条件不进入同预算主比较。

变更变量仅为ROI中心来源；地图不送入语言模型，prompt不透露GT/KD/teacher来源。两学生checkpoint固定使用已有 `best.pt`（两者均selected step1200），不根据新ROI结果改选checkpoint、阈值、裁剪面积或生成设置。原始dense GT/KD的AP高低不预先决定哪一臂可保留。

## 必须通过的检查项

| Gate | 严格验收条件 | 最小落盘证据 |
|---|---|---|
| `same_fullval_identity` | ordered fullval380与旧run完全一致，无缺失、重复、额外path；12类、172正常/208异常仅作metric核对 | 380条原始item、ordered paths、旧cache index与new local position对照、manifest SHA |
| `student_checkpoint_identity` | pre/448、head664577、同compatibility与matched key；GT distill0、KD distill0.5；selected step1200，strict load | 各checkpoint文件SHA、selected step、trained weight、run/completion/reload来源及SHA、strict compatibility结果 |
| `student_map_224` | 448直接square RGB，processor不resize，grid[1,28,28]；pre恢复为[28,28,1152]；FP32 head最终sigmoid为float32[224,224]、全finite且[0,1] | 原图SHA/size、actual geometry、extractor/head/source/config SHA、每map shape/dtype/min/max/array SHA |
| `same_crop_math` | row-major第一最大；中心像素+.5再除224；normalized宽高1/3、面积1/9，边界平移且不缩小；原RGB浮点bbox bicubic EXTENT到448 | peak row/col/value/tie rule、relative center、relative及raw浮点bbox、crop像素SHA |
| `no_label_teacher_in_student_infer` | student infer只消费RGB、固定模型/processor/head；crop仅消费本图student map；不加载teacher/GT mask/label、不用path的Normal/Anomaly字串分支 | 输入字段白名单、GT/teacher读取spy或fail-on-access、model frozen/inference_mode、两臂数据流审计 |
| `same_native_inference` | 同原生Qwen3-VL-8B-Instruct、BF16全参数、不量化、SDPA、TF32false、eval/frozen/inference_mode，原生top-level forward/generate | 当前source/processor/model/tokenizer identities、dtype audit、parity/forward trace |
| `same_two_view_tokens` | 原图→ROI顺序，两图均448；grid[[1,28,28],[1,28,28]]、196×2=392 visual tokens、总prompt470，attention全1且单query无padding | 每条实际grids/token数/roles；模板文本与input_ids SHA |
| `same_decode_score` | 原QUESTION/user-only，同tokenizer Yes9454/No2753；首答案raw logit差、score>0判Yes；greedy、beam1、max_new_tokens32、use_cacheTrue | yes/no logits、score、generated IDs/text、完整generation config、prefill/cache trace |
| `prediction_key_join` | 条件×path唯一；map→query按显式path join；380条每臂顺序能重建为旧selected_paths；不按文件排序或blind zip | keyed prediction JSONL、condition/path/index/map source/map SHA；独立重算metrics |
| `legacy_control_replay` | 固定少量旧center/random/teacher样例重放，几何/tokens/crop与原prediction相同；token IDs与logits按下述规则核验 | replay report、旧prediction SHA、固定选择清单、所有差值与pass/fail |
| `honest_timing` | 地图提取/磁盘IO/裁剪processor/LM score/LM generate分别计时；CUDA同步；另列cold load、end-to-end与GPU峰值 | 每图时长＋总量、warm/cold定义、共享feature策略及设备/runtime |

任何必需gate失败必须标记failed并保留报告，不以0/空数组替代缺失map，不回退teacher/center，不跳过失败图；complete只在两臂各380条score与generation、所有gate及指标重算通过后给出。

## 数据 manifest 身份与键对齐

旧 `roi_full/run.json` 的 `selected_cache_indices` 为9985..10364（闭区间），有序380条path，`selected_manifest_sha256` 为 `73b294cd66e143719cf263a8142d984085d15215bbf9688a293418aa507e1af2`。旧计算定义是 `SHA256(json.dumps(selected_item_dicts, sort_keys=True).encode())`，使用Python默认separators，item字段为path/category/label/split；不是路径列表SHA，也不是compact JSON SHA。本次在CPU按已保存path重建这四个字段并重算得到完全相同SHA。

新运行应从旧run的完整有序manifest或既有split函数生成完整manifest后校验；旧cache index只作身份注释，不能把该index用于一个新的380行student map数组。独立记录paths-only compact SHA便于无label推理接口对齐：`9e9cdaf7b3e8dceffb84eac6b9977a4326517edc5bdae248c82403a566eba08d`。其定义为 `json.dumps(selected_paths, sort_keys=True, separators=(',', ':'))`。路径迁移必须有明确old path→实际文件映射，并核对原图字节SHA；不得只用basename匹配。

将metric manifest（含category/label）与infer manifest（只有opaque sample_id/path及原图内容SHA）分离。所有GT标签只在输出落盘后按path join用于metrics；完整val选样使用既有380路径，无需category/label分层。路径可能含Normal/Anomaly目录名，推理将其作为文件定位字符串，不检查这些字串，不把path/category/label或地图来源写入QUESTION。metric类别划分只在评估器中使用。

学生map export manifest建议每臂记录 `schema_version, arm, feature=pre, input_size=448, output_size=224, dtype=float32, coordinate_system=direct-square-224, model/head/checkpoint/source identities, selected_paths_sha256, maps`；`maps`以完整old path为key，每条含本图RGB SHA、相对`.npy`文件名、文件SHA及C-order float32数组SHA、shape/dtype、峰值row/col、提取时长。所有key集合及按selected_paths读取后的顺序必须精确对应380图。重复map文件/数组并非自动错误（正常图可能一致），但一个key映射到别图或缺失应拒绝。

## 学生地图与无泄漏推理入口

两checkpoint来源为 `dense_pre_gt_s42`、`dense_pre_kd_s42`；它们共享matched key `e79f550e88f58e8a51829608f716fb0049e057cea8beadbb73f6b5ccb827dacf`。同pre448 extractor实际RGB448、patch16、merge2、grid[1,28,28]，须沿用 `restore_premerger_grid` 恢复空间顺序。BF16冻结encoder/pre特征，head FP32；LayerNorm、卷积及56→112→224插值沿用原SpatialHead，最终sigmoid概率图不做min-max标准化、阈值化、平滑、mask裁切、按标签取反或padding。head输出[B,1,224,224]，每图明确取[0,0]，避免把batch/channel误当空间。

必须使用strict checkpoint loader并额外核验checkpoint step、seed、trained distill_weight、head state与原run一致；现有 `load_head_checkpoint`仅强制pipeline/compatibility/strict state，不自动检查selected step或trained weight，需要新runner补检查。本地没有checkpoint，当前只能确定JSON记录selected step1200，真实checkpoint SHA及state身份需执行时落盘。

**不能直接调用 `migration_dense_train.evaluate` 生成ROI地图。** 它会调用`targets`读取GT mask并计算supervised_loss。最小学生推理链是RGB→冻结visual extractor→pre448→加载的head.eval→sigmoid224，并在`torch.inference_mode()`下执行；不需要TeacherCache、teacher decoder、ImageBind CLS、normal bank、GT mask或label。旧`FeatureLRU`虽接受(path,category,label)三元组，实际只消费path，仍建议给纯path入口，防止将标签经接口传入推理。

**旧 `migration_roi_eval.main` 也不能原样充当无teacher学生臂。** 它必然`load_shared_cache`并给`build_views`传`cache['maps'][index]`。新学生runner应使用已校验的val路径manifest，显式把本图student map传入共享几何函数，且student条件名/metadata真实标注student来源。teacher replay在单独分支读旧teacher map，不能把它当学生map缺失回退。使用同一完整Qwen的visual塔可省重复常驻权重，但必须验证extractor与原预训练visual等价，退出/移除hook，不能更改native model权重、精度或MRoPE/cache状态。

## Crop的精确定义与CPU边界用例

在最终float32[224,224]概率图上先检查finite/[0,1]，再用NumPy C-order `argmax`取第一个最大值；同值时按row-major，保留tie_rule文字。令(row,col)为该位置，`cx=(col+0.5)/224, cy=(row+0.5)/224`，side=sqrt(1/9)=1/3，`left=clip(cx-side/2,0,1-side)`、top同理，right=left+side、bottom=top+side。不能改成col/223、直接乘2、先量化/归一化或按logits取峰值；sigmoid饱和可能改变tie，应以已落盘概率图为准。

relative bbox按raw原图宽高分别缩放成浮点像素bbox；调用PIL `Image.Transform.EXTENT`、BICUBIC直接输出448，不先整数crop再resize。raw非正方形时，normalized square对应像素矩形，面积仍为原图1/9；不要为了像素正方形而改边长或letterbox。CPU最少检查：单峰中心/四角、两个相等峰、全常数（取[0,0]）、非正方形raw、NaN/Inf或坏shape拒绝；地图标签/teacher文件访问需fail-on-access。

## 固定 native 模型、prompt 与生成

原QUESTION必须逐字一致：

> The first image is the target product. If a second image is supplied, it is a close-up crop of the same target. If a third image is supplied, it is a normal reference product of the same category. Is there any anomaly in the target product? Answer with Yes or No first, followed by one brief sentence.

即使本次只有两图，也保留第三图说明句；仅user消息，content依次image,image,text，`apply_chat_template(tokenize=False, add_generation_prompt=True)`。processor `images_kwargs={'do_resize': False}`，每图patch16/merge2，实际grid[1,28,28]、per-view196、总392、prompt470；单query无padding、attention_mask全1。固定两图原图在前、ROI在后，roles=`target_original,target_roi`、两条view_sources都为同一目标path，无reference。

模型是既有Qwen3-VL-8B-Instruct完整官方top-level模型，source SHA `dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da`、transformers4.57.1；完整vision/merger/DeepStack/language/head参数BF16、不量化、SDPA，model.eval、requires_grad false、TF32false。记录模型shard/index、config、processor、tokenizer及generation配置的身份（路径、SHA/版本），以现有source/parity audit为门，不把模型目录名视为权重身份的完整证明。旧run未保存所有权重或input_ids的SHA，新结果补记录并靠control replay验证。

首答案用Yes token9454、No token2753；logits转FP32，score=Yes−No，不softmax、不按teacher概率校准；分类严格score>0，等于0为No。score forward `use_cache=False, logits_to_keep=1, return_dict=True`；generation另一次prefill，`max_new_tokens=32, do_sample=False, num_beams=1, use_cache=True, use_model_defaults=False`，BOS/EOS/PAD从原模型generation_config复制。两次调用之前均`model.model.rope_deltas=None`。记录实际generated suffix token IDs并按旧tokenizer `skip_special_tokens=True`解码；最长32是上限，不能强迫生成满32或改变提前EOS。保留原初始Yes/No parser、invalid在分母，不重评解释语义。

至少一个固定两图学生样例做native执行preflight：score无cache的带pixel prefill、generation带pixel且cache_position0的prefill，以及随后无pixel/cache_position>0的cached decode均应观察到；finite logits、BF16 pixels、正确tokens。数值原生parity必须对应当前模型/source与相同batch形状；旧三图preflight不能单独证明新两图学生路径正确。

## 旧对照是否需要 replay

若直接复用旧center/random/teacher全380作为配对效果对照，**需要先做固定小样例replay**，这是新runner复用旧forward/prompt/crop的回归门。无需为了不变的模型和对照重复完整380。建议预先固定旧位置0、190、379（cache index9985、10175、10364），对三个旧两图条件各重放一次，共9条，不根据新结果或标签挑样。若未覆盖实际边界裁剪，另加由旧已保存bbox预先选出的边界样例；几何合成用例不能替代真实native replay。

replay按path查旧完整prediction记录，比较relative/raw bbox、original/crop RGB像素SHA或同算法重建、grids/roles/每图196 tokens、总392 / prompt470、QUESTION/模板IDs；记录old/new logits与差值、生成IDs/text。geometry、keys、tokens必须精确；相同runtime/device/backend下目标是generated IDs精确相等，float logits应先争取精确一致，并用预先声明的小容差（建议max_abs≤1e-5）检查yes/no与score，不能只靠score分类一致放行。若小数差异超过门限或generation不同，标记比较门失败、保留诊断，查模型/processor/runtime后决定重跑全部旧两图对照或将新结果与历史结果分开报告，禁止事后放宽门限。greedy本身不保证跨CUDA/硬件bitwise确定性。

本地结果包仅有`roi_full/run.json`汇总，没有旧五个`*_predictions.jsonl`；不能从汇总AUROC重建逐图片replay期望值。执行侧应使用原文件并记录SHA，若原记录缺失，需重新完整运行旧三对照或明确取消“经过旧对照逐图片replay验证”的表述。teacher replay允许该独立验证分支使用旧teacher缓存；学生两臂应保持无teacher消费。

## Prediction输出与metric复算

每臂JSONL恰好380条：`condition, path/sample_id, target_path, selected_position, legacy_cache_index, student_arm, map_file/map_sha, checkpoint_sha, split='val', source_image_wh, view_sources/roles, roi(peak/center/bboxes/area/tie), actual grids/tokens, yes_logit/no_logit/score, generated_token_ids/text, timing`。label/category可在metric join后放到独立评分文件，不能影响前述输出；manifest与JSONL各记录文件SHA。condition×path唯一、target_path=path、图源两条都等于path；map的source_rgb_sha与该原图一致。

验证器先按path建字典并检查key集合，再依旧selected_paths顺序组装score/text及labels，才调用原`evaluate_predictions`。独立复算run汇总、12类category_macro及counts；不能以“两个list长度同为380”代替键对齐。image AUROC/AP用连续LM首token分数；生成macro F1/recall/FPR/invalid来自独立生成parser。主比较含两学生与三旧对照的AUROC、AP、生成F1、recall、FPR、invalid及类别等权结果；报告signed delta与召回/误报取舍，不能用dense地图AP预测语言提升，或将KD对teacher的差值称为纯teacher知识因果。

## 计时与内存边界

- `map_extract_seconds`：原图读取/预处理、student视觉tower、BF16特征→FP32 head、sigmoid、D2H/必要写出分别可拆列；至少有包含全部这些步骤的map wall time，CUDA前后同步。预计算地图时另记map文件load/hash时间。
- `prepare_seconds`：原RGB读入、固定两图/crop构建、processor编码、H2D；保持与旧ROI的定义一致。旧run的prepare不包含teacher模型计算，其teacher map已在cache中，因此不能拿新含student地图提取的prepare与旧值直接宣称速度优劣。
- `score_seconds`与`generate_seconds`继续按旧函数CUDA同步边界分列：两个独立完整prefill的成本均计入。不要将生成时间等同全query成本，也不能从缓存prefill跳过原控制中存在的计算。
- `end_to_end_seconds`单列在线RGB→student map→crop→LM score＋generate；另列cold模型/head加载和map预计算总时长。预计算后仅LM两图相同是视觉token预算匹配，额外student map前向与teacher离线地图成本未被它匹配，不能称整个系统总计算量相同。
- GT/KD可共享同一张RGB冻结pre448特征，但必须声明为shared模式，分别记录feature一次＋两个head两次成本；若宣称每臂独立部署延迟，应各自承担一次feature提取。不把共享成本只记给第一臂，让第二臂看似零成本。
- 分别记录map阶段、LM阶段、整流程GPU baseline/allocated peak/reserved peak、并存模型/复用visual策略、device/GPU/torch/CUDA/backend。CUDA allocator reserved跨阶段可延续，不能当作各臂新增显存；计时比较需同硬件/冷暖条件。

## 本地审查证据

只读来源：`roi_full/run.json`、`code/migration_roi_eval.py`、`code/migration_roi_preflight.py`、`code/migration_dense_train.py`、`code/migration_visual_probe.py`、`code/migration_metrics.py`及pre GT/KD的run/completion/reload。上述代码快照与 `/Users/qiaohongfei/anomalygpt_remote/files` 对应文件的本次字节完全一致。原run状态complete、两图实际392 visual /470 prompt已记录，但新学生路径的执行证据尚未产生。

| 来源 | SHA256 |
|---|---|
| `roi_full/run.json` | `10e02b111c120ddc5b1cff5a0826b957ff37b4980dfb06d5e80b66be24250495` |
| `code/migration_roi_eval.py` | `324aaeb97eea674d63b322129a01cc6694547f8f884726450f327f390e9cc711` |
| `code/migration_roi_preflight.py` | `4dbdc5249baaadf4ccd97b3c91071a2de8bf039e4d7489e2aa308aad9c4af8b3` |
| `code/migration_dense_train.py` | `a05336a7e94aea1fff50bec009eebc1ee1fbba327c6da5da3a140415df5a7a3d` |
| `code/migration_visual_probe.py` | `af80a1b9d9fd585cacb25b1e7a63c483f73a15bb1811ac3ed3492fb19c33aea6` |
| `code/migration_metrics.py` | `18780dda38a176fbe438f54c65d5f25bd6879f74153ac1eebd7e1816d00f08d0` |
