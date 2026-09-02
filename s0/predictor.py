"""评测所需的可插拔接口：ResizePolicy + Predictor + 平凡基线。

为什么单独放一个文件
--------------------
评测脚本 s0/eval.py 只描述"如何算指标"，不关心"模型是怎么预测的"。
模型的可变部分（怎么 resize、怎么从模型拿分数和热图）全部收敛到两个接口里，
这样评测一次写好就不改，替换模型只替换 predictor + policy。

设计约束
--------
1. **ResizePolicy 同时决定模型输入和指标对齐网格**。
   "448 主口径"和"原分辨率对照"不是把 GT 拉来拉去，而是两条 track：
   - r448  ：模型见 448，指标在 448 网格上算（与比赛 mask 口径一致）
   - rnative：模型见原图（可内部截断），指标在原分辨率网格上算
   这样"小缺陷是否被压没"才能真正对比出来，而不是被 resize 掩盖。

2. **Predictor 只吃预处理后的图像，只吐 (score, heatmap)**。
   heatmap 与 model_input 同网格；对齐交给 ResizePolicy.align。
   trivial 基线只用来验证 eval 数学正确，**绝不能出现在真实实验里**。
"""

from __future__ import annotations

import sys
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RGB_SIZE = 448


class ResizePolicy(ABC):
    """定义模型输入尺寸 + 预测热图与 GT 的对齐方式。"""

    name: str
    DETERMINISTIC = True   # 是否可复现（用于排除随机策略）

    def __init__(self, model_side: int | None = None):
        self.model_side = model_side

    @abstractmethod
    def model_input(self, image_rgb: np.ndarray) -> np.ndarray:
        """把原图（任意尺寸）转成给模型的输入。需返回 uint8 RGB。"""

    @abstractmethod
    def align(self, pred: np.ndarray, gt_binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """把 (预测热图, GT mask) 对齐到同一个指标网格，供算像素指标。

        pred: float32，模型输出网格，0..1
        gt_binary: bool，原分辨率（或策略自己的 GT 网格）
        返回 (pred_aligned, gt_aligned)，两者同形状。
        """


class Resize448Policy(ResizePolicy):
    """模型输入恒定 448；指标在 448 网格上算（比赛 mask 口径）。"""

    name = "r448"

    def model_input(self, image_rgb: np.ndarray) -> np.ndarray:
        return cv2.resize(image_rgb, (RGB_SIZE, RGB_SIZE), interpolation=cv2.INTER_AREA)

    def align(self, pred: np.ndarray, gt_binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # gt 统一为 bool（供指标层直接用 bool mask 索引，不 float）：
        # resize 后用 >0.5 阈值还原。
        gt = gt_binary.astype(bool)
        if gt.shape != (RGB_SIZE, RGB_SIZE):
            # GT 是离散标注，必须 nearest-neighbor。用 AREA + 0.5 会让小缺陷在
            # 评测标签侧直接消失，从而把 resize 造成的信息损失错误地藏起来。
            resized = cv2.resize(
                gt.astype(np.uint8), (RGB_SIZE, RGB_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )
            gt = resized.astype(bool)
        pred = np.clip(pred.astype(np.float32), 0, 1)
        if pred.shape != (RGB_SIZE, RGB_SIZE):
            pred = cv2.resize(pred, (RGB_SIZE, RGB_SIZE), interpolation=cv2.INTER_AREA)
        return pred, gt


class NativeResizePolicy(ResizePolicy):
    """模型见原图；指标在原分辨率网格上算。

    max_side: 为了让模型能吃的上限（0 表示不截断）。注意：截断本身也丢信息，
    但比压到 448 保留得多。真实模型应自行处理动态分辨率。
    """

    name = "rnative"

    def __init__(self, max_side: int = 0):
        super().__init__()
        self.max_side = max_side

    def model_input(self, image_rgb: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        if self.max_side and max(h, w) > self.max_side:
            scale = self.max_side / max(h, w)
            return cv2.resize(image_rgb, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_AREA)
        return image_rgb

    def align(self, pred: np.ndarray, gt_binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gt = gt_binary.astype(bool)
        pred = np.clip(pred.astype(np.float32), 0, 1)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        return pred, gt


class Predictor(ABC):
    """只吃预处理后的图，只吐 (score, heatmap)。"""

    @abstractmethod
    def predict(self, image_rgb: np.ndarray) -> tuple[float, np.ndarray]:
        """返回 (anomaly_score in [0,1], heatmap float32 与 model_input 同网格)。"""


# ---------------------------------------------------------------------------
# 平凡基线 —— 仅用于验证 eval / 延迟框架的数学与 I/O，真实实验禁用
# ---------------------------------------------------------------------------

class ConstantPredictor(Predictor):
    """恒定分数/热图。烟雾测试：I-AUROC / P-AUROC 都应≈0.5。"""

    def __init__(self, score: float = 0.5, heat: float = 0.5):
        self.score, self.heat = score, heat

    def predict(self, image_rgb: np.ndarray) -> tuple[float, np.ndarray]:
        return self.score, np.full(image_rgb.shape[:2], self.heat, dtype=np.float32)


class RandomPredictor(Predictor):
    """均匀随机热图。烟雾测试：I-AUROC / P-AUROC 都应≈0.5。"""

    def __init__(self, seed: int = 0):
        self.seed = int(seed)

    def predict(self, image_rgb: np.ndarray) -> tuple[float, np.ndarray]:
        # 由输入内容派生局部 RNG：workers=1/8 的结果逐位一致，也不会因把 predictor
        # pickle 给多个进程而让每张图从同一个 RNG 状态开始。
        digest = hashlib.blake2b(
            memoryview(np.ascontiguousarray(image_rgb)), digest_size=8,
            person=self.seed.to_bytes(8, "little", signed=True),
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        h, w = image_rgb.shape[:2]
        return float(rng.random()), rng.random((h, w)).astype(np.float32)


class OraclePredictor(Predictor):
    """读 GT 的完美模型。**仅用于验证 eval 数学**：
    它应得到 I-AUROC≈1、P-AUROC≈1、P-AUPR≈1。若达不到，说明 eval 的
    标签对齐或聚合有 bug，必须修好才算评测可信。

    它"作弊"地看 GT，所以只许出现在 eval --self-test 里，与真实实验无关。
    """

    def __init__(self, get_gt):
        self.get_gt = get_gt   # 回调：image_path -> binary native GT

    def predict(self, image_rgb: np.ndarray) -> tuple[float, np.ndarray]:
        # 这里拿不到 image 的路径，所以 Oracle 不走 Predictor.predict，
        # 而是由 eval 的 self-test 路径直接喂 GT。见 eval.py。
        raise NotImplementedError("OraclePredictor 由 eval --self-test 使用专用路径")


__all__ = [
    "RGB_SIZE",
    "ResizePolicy",
    "Resize448Policy",
    "NativeResizePolicy",
    "Predictor",
    "ConstantPredictor",
    "RandomPredictor",
    "OraclePredictor",
]
