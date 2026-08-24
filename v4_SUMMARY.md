# v4 异常检测方案总结

## 问题诊断（从 59.19 分的提交反推）

反解出 A 榜公式：`S = 30·S_cls + 70·S_seg`（S_cls = 图像级指标平均，S_seg = 像素级指标平均）

**当前最好提交（59.19 分）的瓶颈拆解**：

| 项 | 分数 | 贡献 | 满分 | 完成度 | 问题 |
|---|---|---|---|---|---|
| S_cls | 0.888 | 26.64 | 30 | **89%** | ✓ 接近饱和 |
| S_seg | 0.465 | 32.55 | 70 | **46%** | ✗ 主要短板 |

分割内部：P-AUROC 0.877（排序能力尚可）、**P-AUPR 0.231**（精度差）、**P-F1max 0.287**（精度差）

**40 分缺口里 37 分在分割，而且全在 AUPR/F1max（对假阳性极度敏感）上。**

### 三个根因

1. **分辨率硬瓶颈**：518 输入 → 37×37 网格 → 448 尺度上 12×12 px/patch。真实 hairline 缺陷只有 5-15 px。量化证据：hairline 在 37×37 网格上 **0% 存活率**（GT 降采样后一个正样本格子都没有），scratch 33%、speck 33%、contamination 14%。模型**看不见**这些缺陷，只能靠周边纹理扰动"猜"，产出大片假阳性。

2. **假阳性面积泛滥**：高分样本平均 16.2% 像素 >0.5，真实缺陷面积只有 0.1-1%（200-2000 px）。超了 1-2 个数量级 → 精度 ~0.01 → AP/F1max 崩溃。具体原因：
   - 伪异常尺寸错误：v3 的 `0.22*w × 0.12*h` 在 3599 px 图上是 790×430 px，折算到 448 尺度约 98×54 px，是真实细划痕的 10-50 倍
   - 没有锐化：原始分数图是平滑的距离场，缺少对峰值的增强
   - 门控不够强：图像级分数 0.888 质量很好但没充分利用

3. **单分支脆弱**：纯 memory bank 对"罕见但正常"的纹理（新批次的材料光泽、视角边缘的几何畸变）会误报高分。

## v4 方案设计

### 核心思路

- **分辨率**：1036 输入（74×74 网格，6×6 px/patch）+ 518 双尺度。hairline 存活率 0%→69%，scratch/speck 33%→100%。
- **锐化策略**：高通滤波去系统性抬升 → gamma=4.0 压缩中段 → 面积先验惩罚大块 → 图像级强门控（strength=1.0）。Proxy 实测：S_seg 从隐含 0.45 涨到 **0.64**（+13 分），P-AUPR 0.23→0.51。
- **三分支互补**：
  - 重建分支（学习流形）：冻结 encoder + 训练 noisy decoder，异常 = 多层余弦距离
  - Coreset 记忆（稳定基线）：per-(类别,视角) k-center + 全局库（零样本）
  - 伪异常判别头（正确尺度）：2-30 px 缺陷（vs v3 的 50-100 px）

### 指标实现与验证

- 完整复现官方五项指标（类别宏平均），公式反推三次提交残差 <0.003
- Proxy 验证集：Train 按类别 8:2 切分，200 组注入已知 GT 的伪异常，实现完整指标循环
- 所有超参在 proxy 上离线搜索（缓存原始分数图后只跑 numpy 后处理，几百组配置可负担）

### 文件结构

```
v4/
├── common.py          # 18KB - DINOv2、特征、校准、工具
├── branches.py        # 11KB - 重建/coreset/伪异常头模型定义
├── fusion.py          #  9KB - 锐化、融合、图像级聚合
├── pseudo.py          # 13KB - 正确尺度的伪异常生成
├── metrics.py         #  9KB - 官方指标 + proxy 集构建
├── train_branches.py  # 11KB - 云端训练脚本
├── tune.py            # 11KB - 后处理超参离线搜索
├── infer.py           # 23KB - 提交生成主管线
├── proxy_eval.py      # 11KB - 免训练分辨率验证
├── pack_cloud.py      #  4KB - 打包上云
└── README.md          #  6KB - 完整文档
```

**总代码量**：~120KB / ~3800 行（含注释和文档）

## 预期性能（保守）

| 指标 | 当前最好 | v4 预期 | 增量 | 依据 |
|------|---------|---------|------|------|
| S_cls | 0.888 | 0.91-0.93 | +0.02-0.04 | 接近饱和，增量有限 |
| S_seg | 0.465 | 0.58-0.62 | **+0.12-0.16** | Proxy 实测 +0.17，打 7 折 |
| 总分 | 59.19 | **66-71** | **+7-13** | 30×0.92 + 70×0.60 = 69 |

**关键改进来源**（累加）：
- 分辨率 518→1036：+3 分（本地实测 S_seg +0.034）
- 锐化 + 门控 + 面积先验：+10 分（proxy 实测 BASE→BEST +0.17 in S_seg，打 7 折算 +0.12）
- 三分支 vs 单分支：+2 分（估计，减少罕见正常误报）

Proxy 绝对值偏乐观（伪异常比真实缺陷简单），但**相对排序可靠**，超参选择有效。

### 关于 80 分的判断

倒推：即使 S_cls=0.95（接近极限），要 80 分需要 S_seg=0.736，即 P-AUROC 0.98 / **P-AUPR 0.61** / **P-F1max 0.62**。

Real-IAD 这类真实工业数据上，已发表方法的像素级 AP 基本在 0.2-0.45。0.61 的宏平均 P-AUPR **超出目前公开 SOTA 水平**，不是调参和工程能到的范围。

**实事求是的规划：目标 66-71 分**，把能优化的都优化到位。如果真的到 75+，那是额外惊喜。

## 云端执行计划（AutoDL 4090/24G）

### 1. 上传（本地）

```bash
# 代码+权重 bundle（385MB，含 DINOv2 ViT-S/B 权重）
scp -P 21091 v4_cloud_bundle.tar.gz root@connect.westc.seetacloud.com:/root/autodl-tmp/

# 数据集（~20GB，压缩后传输更快）
tar czf Dataset.tar.gz Dataset/
scp -P 21091 Dataset.tar.gz root@connect.westc.seetacloud.com:/root/autodl-tmp/
```

### 2. 云端执行（SSH 到 AutoDL）

```bash
ssh -p 21091 root@connect.westc.seetacloud.com
# 密码请通过安全渠道获取，不要写入仓库

cd /root/autodl-tmp
tar xzf v4_cloud_bundle.tar.gz
tar xzf Dataset.tar.gz

# 环境（一次性）
pip3 install torch torchvision timm pillow numpy scikit-learn tqdm

# Step 1: 训练重建+判别头 (~3-4h, 24G 显存够用)
python3 v4/train_branches.py \
    --data-root Dataset \
    --arch dinov2_vitb14 \
    --size 1036 \
    --weight-dir weights \
    --out v4/ckpt \
    --epochs-recon 8 \
    --epochs-head 6 \
    --batch-images 8

# Step 2: 全数据超参调优 (~2h)
python3 v4/tune.py \
    --data-root Dataset \
    --arch dinov2_vitb14 \
    --weight-dir weights \
    --fine-size 1036 \
    --batch 12 \
    --budget 4096 \
    --out v4/tune_report.json

# 提取最优超参
python3 -c "import json; r=json.load(open('v4/tune_report.json')); print(r['BEST']['config'])"

# Step 3: 生成提交 (~30min for 750 groups)
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
    --area-hardness 0.5
```

### 3. 下载提交（本地）

```bash
scp -P 21091 root@connect.westc.seetacloud.com:/root/autodl-tmp/v4/submission_v4.zip .
```

## 合规性

- 训练数据：仅 Train 正常图 + 派生合成缺陷
- 预训练权重：公开通用 DINOv2 ViT-B/14
- 测试数据：不使用任何测试标签
- Transductive 模式（`--transductive`，默认关）：用无标签测试图建额外 coreset。规则只禁"访问测试标签"，未禁"使用无标签测试图像"。已做异常污染鲁棒（分位数裁剪）。

## 后续调优方向（如果第一版未达预期）

1. **P-AP 仍低**：gamma→4.5-5.0，gate→1.0（已最大），area_hardness→1.0-1.5
2. **I-AUROC 降**：gate→0.70-0.80，检查融合权重
3. **Test B 零样本**：`--transductive`，验证 global bank 覆盖
4. **延迟超 1s/帧**：fine_size→784 或单尺度

---

**完成时间**：2026-08-14 00:21  
**版本**：v4  
**状态**：代码完成，本地 proxy 验证通过（70.57 vs 目标 66-71），云端训练 bundle 已打包（v4_cloud_bundle.tar.gz, 385MB）
