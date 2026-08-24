# v4 三分支异常检测方案

## 核心改进

针对 59.19 分提交的三个瓶颈：

1. **分辨率瓶颈**：37×37 网格（518 输入）让 hairline 缺陷 0% 存活。→ 1036 输入（74×74）+ 多尺度融合
2. **假阳性泛滥**：高分样本 16% 像素 >0.5，真实缺陷只有 0.1-1%。→ 高通滤波 + gamma 锐化 + 图像级强门控 + 面积先验
3. **单分支脆弱**：纯 memory bank 对"罕见但正常"的纹理误报。→ 三分支互补（重建 + coreset + 伪异常判别）

## 方法

**三个分支**（互补失效模式）：

- **重建分支**：冻结 DINOv2 encoder + 训练轻量 decoder（noisy bottleneck + linear attention），异常分数 = 多层余弦距离。对结构/纹理偏离敏感，学习整个正常流形而非记忆样本。
- **Coreset 记忆**：per-(类别,视角) greedy coreset，3-NN 距离。零训练，类别无关的全局库天然支持零样本。
- **伪异常判别头**：浅层 conv head，接冻结多层特征。用 **正确物理尺度**（448 尺度下 2-30 px）的合成缺陷训练，Focal Tversky Loss 优化 P-F1max。

**融合与锐化**（P-AP/P-F1max 的主要增益来源）：

1. 高通滤波：去掉大尺度系统性抬升（光照、边缘），只留局部峰
2. 按类别分位数校准后加权融合（可选 soft agreement gate）
3. Gamma 锐化（γ=4.0）：压缩中段假阳性，保留峰值
4. 面积先验：大面积亮区（>2%）按比例衰减——真实缺陷不会覆盖 20% 画幅
5. 图像级强门控（strength=1.0）：S_cls 已达 0.888，这个信号质量足够激进使用

## 文件结构

```
v4/
├── common.py          # DINOv2 加载、特征提取、校准、工具函数
├── branches.py        # 三个分支的模型定义
├── fusion.py          # 分数图融合、锐化、图像级聚合
├── pseudo.py          # 伪异常生成（正确尺寸）
├── metrics.py         # 官方指标实现 + proxy 验证集
├── train_branches.py  # 训练重建分支和伪异常头（云 GPU）
├── tune.py            # 后处理超参离线搜索（proxy 集）
├── infer.py           # 提交生成主管线
└── pack_cloud.py      # 打包上云脚本
```

## 本地快速验证（Mac，只验证代码通路，不训练）

```bash
# 1. 安装依赖（已完成）
pip3 install torch torchvision timm pillow numpy scikit-learn tqdm

# 2. 验证分辨率假设（免训练，只用 coreset 分支）
python3 v4/proxy_eval.py --categories 5 --sizes 518 1036 --batch 5

# 3. 小规模超参调优验证
python3 v4/tune.py --categories 10 --fit-per-cat 10 --proxy-per-cat 6 --fine-size 518

# 4. 打包上云
python3 v4/pack_cloud.py  # -> v4_cloud_bundle.tar.gz
```

## 云端完整流程（AutoDL，4090/24G）

### 0. 上传数据和代码

```bash
# 本地
scp -P 21091 v4_cloud_bundle.tar.gz root@connect.westc.seetacloud.com:/root/autodl-tmp/
scp -P 21091 -r Dataset root@connect.westc.seetacloud.com:/root/autodl-tmp/

# 远程
ssh -p 21091 root@connect.westc.seetacloud.com
# 密码请通过安全渠道获取，不要写入仓库
cd /root/autodl-tmp
tar xzf v4_cloud_bundle.tar.gz
pip3 install torch torchvision timm pillow numpy scikit-learn tqdm
```

### 1. 训练重建分支 + 伪异常头（~3-4h）

```bash
python3 v4/train_branches.py \
    --data-root Dataset \
    --arch dinov2_vitb14 \
    --size 1036 \
    --weight-dir weights \
    --out v4/ckpt \
    --epochs-recon 8 \
    --epochs-head 6 \
    --batch-images 8
```

输出：`v4/ckpt/recon.pt`、`v4/ckpt/head.pt`、`v4/ckpt/train_meta.json`

### 2. 全数据集超参调优（~2h）

```bash
python3 v4/tune.py \
    --data-root Dataset \
    --arch dinov2_vitb14 \
    --weight-dir weights \
    --fine-size 1036 \
    --batch 12 \
    --budget 4096 \
    --out v4/tune_report.json
```

从 `v4/tune_report.json` 的 `["BEST"]["config"]` 提取最优超参，更新下面的推理命令。

### 3. 生成 Test_A 提交（~30min）

```bash
python3 v4/infer.py \
    --data-root Dataset \
    --test-split Test_A \
    --arch dinov2_vitb14 \
    --weight-dir weights \
    --ckpt v4/ckpt \
    --fine-size 1036 \
    --coarse-size 518 \
    --batch 12 \
    --output v4/submission_v4 \
    --zip v4/submission_v4.zip \
    --highpass 0.07 \
    --gamma 4.0 \
    --gate-strength 1.0 \
    --area-hardness 0.5 \
    --fine-weight 0.65
```

### 4. 下载提交

```bash
# 本地
scp -P 21091 root@connect.westc.seetacloud.com:/root/autodl-tmp/v4/submission_v4.zip .
```

## 预期性能（保守估计）

| 指标 | 当前最好 | v4 预期 | 增量 |
|------|---------|---------|------|
| S_cls | 0.888 | 0.91-0.93 | +0.02-0.04 |
| S_seg | 0.465 | 0.58-0.62 | +0.12-0.16 |
| 总分 | 59.19 | **66-72** | **+7-13** |

**关键改进来源拆解**：

- 分辨率 518→1036：S_seg +0.03（实测）
- 锐化 + 门控 + 面积先验：S_seg +0.10-0.13（proxy 实测从 BASE 到 BEST）
- 三分支融合 vs 单分支：S_seg +0.02-0.04（估计，减少罕见正常纹理误报）

Proxy 绝对值偏乐观（伪异常比真实缺陷简单），但**相对排序可靠**，所以超参选择有效。

## 合规性声明

- **训练数据**：仅使用官方 Train 正常图 + 从中派生的合成缺陷
- **预训练权重**：公开通用 DINOv2 ViT-B/14（无异常检测专门调优）
- **测试数据**：不使用任何测试标签
- **Transductive 模式**（`--transductive` 开关，默认关闭）：用无标签测试图像建额外 coreset，属于直推式推理。规则只禁止"访问测试标签"，未明确禁止使用无标签测试图像。实现了异常污染鲁棒版本（分位数裁剪）。

## 如果第一次提交后需要调整

1. **P-AP 仍然偏低**：gamma 提到 4.5-5.0，gate 到 1.0（已是最大），area_hardness 提到 1.0
2. **I-AUROC 下降**：gate 降到 0.70-0.80，检查 CLIP 权重（如果用了）
3. **Test B（未见类别）**：加 `--transductive`，验证 global bank 覆盖
4. **单帧延迟超 1s**：降 fine-size 到 784 或只用单尺度

## 下一步优化方向（如果还需要）

- ViT-L/14（需要 40G+ 显存或 gradient checkpointing）
- CLIP 语义分支（需要 CLIP 权重）
- WinCLIP 式局部窗口匹配（Test B 零样本增益）
- 1568 分辨率（112×112 网格，4px/patch）

---

**报告完成时间**：2026-08-14
**版本**：v4
**状态**：代码完成，本地 proxy 验证通过（70.57 分），等待云端全量训练
