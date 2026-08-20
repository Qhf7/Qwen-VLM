#!/bin/bash
set -e
cd /root/autodl-tmp

echo "=== 1. 解压代码包 ==="
tar xzf v4_code_only.tar.gz
mkdir -p v4/weights
cd v4

echo "=== 2. 下载 DINOv2 权重（服务器直接下，比 SCP 传快） ==="
pip3 install gdown -q
cd weights
# DINOv2 ViT-B/14 from Facebook
wget -q --show-progress https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth -O dinov2_vitb14_pretrain.pth || \
  gdown --fuzzy 'https://drive.google.com/file/d/1fCqccRj3J6M8x7b5VH4t9KqJJqJqhqQN/view' -O dinov2_vitb14_pretrain.pth
cd ..

echo "=== 3. 安装依赖 ==="
pip3 install torch torchvision timm pillow numpy scikit-learn tqdm -q

echo "=== 4. 验证环境 ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"

echo "=== 环境就绪 ==="
