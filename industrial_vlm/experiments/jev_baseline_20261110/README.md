# Jev 0/1/4 正常参考图判断基线

固定模型 jev-1.14.0、同一批120张VisA开发query与448×448处理，比较0、1、4张同类训练正常参考图，共360个目标判断。这是上下文支持推理，不是微调或LoRA；120图已用于开发，不是新盲测。

## 数据协议

来源是外层industrial_vlm/prepared/multitask_v2的val_decision_120与train_decision_decision，以及既有source_byte_audit.json。Query为12类各5正常、5异常。训练先按source去重为768来源，每类正常来源按SHA256("jev-support-v1|42|"+source_id)排序取4，共48支持；1-shot取4-shot首张。

检查训练/query来源、路径、历史原字节SHA无交集。168张原图必须逐字节匹配历史SHA与大小。frozen/plan.json保存来源身份、类别、相对路径、hash、字节大小；标签单独放labels.json；GT框、答案、面积不进入API请求。不读取confirmation。

EXIF转正、RGB、LANCZOS等比缩放、白色居中填充为448×448、PNG compress_level=6。冻结Pillow与其他依赖版本、处理后SHA，三臂query用相同PNG。

## 接口与凭证

正确接口HTTPS /v1/systemone。assets使用中性UUID与PNG data_base64，state只有query、category、normal_references，固定Noul问题；读取answers.anomaly.noul并检查模型名。需要TypeSafe-Danger-Zone-Images:true。

ModelGo通用文档与当前模型卡存在差异，以独立真实图像预检为准；/v1/content_generation/tasks不用于本基线。ModelGo model是逻辑名称回显，不独立证明上游实际版本。图片功能仍属TypeSafe实验能力。

凭证只通过环境变量或getpass隐藏输入。不写入文件、Authorization、原始请求、base64或原始错误响应；仅记录允许字段、HTTP状态与固定错误类别。没有可靠货币费用字段时不估造成本。

## 准备与运行

从外层项目根目录执行。以下图像根目录已经由父任务恢复，里面直接含各类别目录。

    PY=/opt/anaconda3/bin/python
    BASE=Qwen-VLM/industrial_vlm/experiments/jev_baseline_20261110
    $PY "$BASE/runner.py" prepare --queries industrial_vlm/prepared/multitask_v2/val_decision_120.jsonl --train industrial_vlm/prepared/multitask_v2/train_decision_decision.jsonl --audit industrial_vlm/experiments/multitask_v2/source_byte_audit.json --image-root "$BASE/data_images/VisA"
    $PY "$BASE/runner.py" materialize --image-root "$BASE/data_images/VisA"
    $PY "$BASE/runner.py" run --endpoint https://api.modelgo.ai/v1/systemone --output "$BASE/runs/modelgo_448" --workers 3 --timeout 90 --max-consecutive-errors 3 --dry-run

正式运行去掉dry-run，并指定环境变量名；未配置时隐藏输入key：

    $PY "$BASE/runner.py" run --endpoint https://api.modelgo.ai/v1/systemone --key-env MODELGO_API_KEY --output "$BASE/runs/modelgo_448" --workers 3 --timeout 90 --max-consecutive-errors 3
    $PY "$BASE/runner.py" evaluate --output "$BASE/runs/modelgo_448"

可用limit-queries 1做3个判断的预检，必须独立output如runs/preflight_1query，不混入正式结果。端点可替换成支持图片的原生TypeSafe /v1/systemone，但必须新建output，不能在原输出续跑换渠道。

## 恢复与并发

contract.json冻结endpoint、code/config/data hashes、依赖版本、timeout、workers与目标顺序。按query×0/1/4安排任务，workers默认1、可用3；每批有限并发，主线程统一append/fsync，并等待本批完成再判断是否停止。并发请求完成顺序可能不同，按(source_id, shots)对齐。输出目录文件锁防止两个runner并发改写。

发送前写started intent。成功预测逐条fsync。超时/中断可能已经在服务端完成，不自动重发；检查attempts.jsonl后才用显式--retry-failed，可能重复计费。每次run每个未完成目标最多发送一次；建议重试最多3轮，仍失败则保留不完整状态。恢复会校验配置、数据、code、类别、请求hash、概率、模型和重复身份。

- predictions.jsonl：允许的概率、模型、usage、请求ID、延迟、hash和本地来源标识；不存图片或key。
- attempts.jsonl：发送前和失败记录。
- status.json：完成数量；preflight不当作正式实验。
- 完整360个配对正式结果才输出metrics.json/RESULTS.md。预检或缺失只输出partial状态。

## 指标

AUROC、非插值AP、Macro-F1/召回/FPR（阈值0.5）、Brier、10等宽bin ECE，逐类指标，延迟均值/中位数/p95与可用token usage。配对1−0、4−0、4−1按类别+标签分层bootstrap 2000次seed42，AUROC/AP百分位95%区间。区间不含服务随机性或支持选择变化，未多重比较校正。

## 离线测试

    $PY -m unittest discover -s "$BASE" -p test_runner.py -v

测试使用临时合成图像、合成标签和mock响应，无API调用，不进入正式结果。覆盖ties指标、Noul类型/版本、标签与GT不入payload、原字节及PNG篡改、配置/重复拒绝、断点跳过、超时无自动重试、preflight不产正式指标。

冻结后不要改runner或升级依赖续跑；方法变更应新建冻结目录。

## 本次实际执行的操作修订

科学配置、原runner和contract保持不变。原阶段每批3并发，保存61个成功目标后切换独立`resume_executor.py`滚动6并发，记录`operational_amendment.json`。该阶段保存135个成功目标，累计196。

多个4参考图请求在约15秒上传/连接阶段中断，因此停止提交并等待在途请求结束，新增独立`transport_executor.py`。`operational_amendment_v2.json`串联旧修订、代码、历史前缀哈希与实际传输设置：线程独享Session、滚动6并发、connect/upload 60秒、read 120秒、每目标累计最多5次started。原始问题、图片字节、JSON body、UUID和request hash均一致。成功目标不重发。

现有输出恢复应使用下列命令，不能删除或覆盖contract来改变旧参数：

    $PY "$BASE/transport_executor.py" --output "$BASE/runs/modelgo_448" --key-env MODELGO_API_KEY --retry-failed

传输/超时本次不自动重发；再次显式恢复才可重试，仍受总预算5次限制。HTTP502可在显式retry模式下有界补发。`STOP_V2`文件使执行器停止新提交并等待在途完成。旧`STOP`文件保留用于历史阶段，不影响新执行器。已complete时执行器不读取key、不调用API。

完成后：

    $PY "$BASE/runner.py" evaluate --output "$BASE/runs/modelgo_448"
    $PY "$BASE/supplement_results.py" --output "$BASE/runs/modelgo_448"
    $PY "$BASE/plot_results.py"

后处理只有360个唯一目标全部完成后才产生正式图表和指标。主指标保持冻结协议；类别宏平均、混淆计数、校准分箱与服务可靠性作为补充分析。阶段延迟不能当作受控模型速度比较。
