#!/bin/bash
# 完整的 AutoDL 服务器执行脚本

set -e
cd /root/autodl-tmp

echo "=== Step 1: 解压 bundle ==="
tar xzf v4_cloud_bundle.tar.gz
cd v4

echo "=== Step 2: 安装依赖 ==="
pip3 install torch torchvision timm pillow numpy scikit-learn tqdm -q

echo "=== Step 3: 验证环境 ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo "=== Step 4: 训练重建分支 + 伪异常像素头 ==="
python3 train_branches.py \
  --data-root ../Dataset \
  --arch dinov2_vitb14 \
  --size 1036 \
  --weight-dir ../weights \
  --out ckpt \
  --epochs-recon 8 \
  --epochs-head 6 \
  --batch-images 8

echo "=== Step 5: Train-only proxy 超参数调优 ==="
python3 tune.py \
  --data-root ../Dataset \
  --arch dinov2_vitb14 \
  --weight-dir ../weights \
  --fine-size 1036 \
  --batch 12 \
  --budget 4096 \
  --background-keep 0.08 \
  --boundary-keep 0.55 \
  --out tune_report.json

echo "=== Step 6: 推理 Test_A 并打包 ==="
python3 infer.py \
  --data-root ../Dataset \
  --test-split Test_A \
  --arch dinov2_vitb14 \
  --weight-dir ../weights \
  --ckpt ckpt \
  --fine-size 1036 \
  --coarse-size 518 \
  --batch 12 \
  --output submission_v4_model \
  --zip submission_v4_model.zip \
  --highpass 0.0 \
  --gamma 1.0 \
  --gate-strength 0.0 \
  --area-hardness 0.5 \
  --background-keep 0.08 \
  --boundary-keep 0.55

echo "=== 完成! submission_v4_model.zip 已生成 ==="
ls -lh submission_v4_model.zip
