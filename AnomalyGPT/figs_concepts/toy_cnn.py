"""28x28 玩具例子：conv3x3+ReLU+MaxPool 两次，最后 5x5 卷积滑出 3x3=9 个 token。
   卷积核全部手工设定，保证每一步都能手算核对。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True, linewidth=200)
torch.set_grad_enabled(False)

# ---------------- 输入：28x28，10~13 行/列有一个 4x4 亮块（"缺陷"）
X = np.full((28,28), 0.05, np.float32)
X[10:14, 10:14] = 0.90
x = torch.tensor(X)[None,None]

# ---------------- 第 1 层：4 个手工核
K1 = torch.tensor([
    [[[0,0,0],[0,1,0],[0,0,0]]],                                   # 0 原样
    [[[1,1,1],[1,1,1],[1,1,1]]],                                   # 1 均值
    [[[-1,0,1],[-1,0,1],[-1,0,1]]],                                # 2 竖直边缘
    [[[1,1,1],[0,0,0],[-1,-1,-1]]],                                # 3 水平边缘
], dtype=torch.float32)
B1 = torch.zeros(4)
c1 = F.relu(F.conv2d(x, K1, B1, padding=1))          # [1,4,28,28]
p1 = F.max_pool2d(c1, 2)                              # [1,4,14,14]

# ---------------- 第 2 层：4→6，核为"对每个输入通道取 1/4 权重之和"
K2 = torch.zeros(6,4,3,3)
for o in range(6):
    for i in range(4):
        if (o+i) % 4 == 0 or o >= 4:
            K2[o,i] = 1/4
B2 = torch.zeros(6)
c2 = F.relu(F.conv2d(p1, K2, B2, padding=1))          # [1,6,14,14]
p2 = F.max_pool2d(c2, 2)                              # [1,6,7,7]

# ---------------- 第 3 层：6→8，5x5，每个输出通道只读一个输入通道，取 5x5 平均
# 5x5 高斯核（中心权重大），总和归一化为 1
g = torch.tensor([[1,2,3,2,1],[2,4,6,4,2],[3,6,9,6,3],[2,4,6,4,2],[1,2,3,2,1]], dtype=torch.float32)
g = g / g.sum()
K3 = torch.zeros(8,6,5,5)
for o in range(8):
    K3[o, o % 6] = g
B3 = torch.zeros(8)
c3 = F.conv2d(p2, K3, B3)                             # [1,8,3,3]
tokens = c3.reshape(1,8,9).transpose(-2,-1)[0]        # [9,8]

print("形状流水：", tuple(x.shape), "→", tuple(c1.shape), "→", tuple(p1.shape),
      "→", tuple(c2.shape), "→", tuple(p2.shape), "→", tuple(c3.shape), "→ tokens", tuple(tokens.shape))

# ---------------- ASCII 输入
chars = " .:-=+*#%@"
print("\n=== 输入 28x28（每格 1 个字符，越亮字符越重）===")
for r in range(28):
    print("".join(chars[min(9, int(v*10))] for v in X[r]))

print("\n=== conv1 第 0 个核（原样）在 (9,9) 的手算 ===")
win = X[9:12, 9:12]
print("输入 3x3 窗口:\n", win)
print("权重:\n", K1[0,0].numpy())
print("逐项乘积之和 =", float((torch.tensor(win)*K1[0,0]).sum()), " → 过 ReLU 不变")

print("\n=== pool1 对 conv1 第 0 通道左上 2x2 取 max ===")
print(c1[0,0,:2,:2].numpy(), "→ max =", float(c1[0,0,:2,:2].max()))

print("\n=== conv2 第 0 通道在 (3,3) 的输入窗口（4 个通道各 3x3）===")
for i in range(4):
    print(" in ch%d:\n" % i, p1[0,i,2:5,2:5].numpy())
print("该输出通道的权重（每个 in 通道都是 1/4）: 0.25")
val = float((p1[0,:,2:5,2:5]*K2[0]).sum())
print("求和 =", val, "（= 4 通道 3x3 全部相加 / 4）")

print("\n=== 池化两次后的 7x7 特征图（第 0、2 通道）===")
print("ch0:\n", p2[0,0].numpy())
print("ch2:\n", p2[0,2].numpy())

print("\n=== 最后一步：3x3 个位置，每个取 5x5 平均 ===")
for (i,j) in [(0,0),(1,1)]:
    w = p2[0,0,i:i+5, j:j+5]
    print("位置(%d,%d) 的 5x5 窗口:\n" % (i,j), w.numpy())
    print("  高斯权重:\n", np.round(g.numpy(),3))
    print("  加权和 = %.4f" % float((w*g).sum()))

print("\n=== 9 个 token（每个 8 维）===")
print(tokens.numpy())
print("\ntoken 的 L2 范数:", np.round(tokens.norm(dim=-1).numpy(), 3))

# ---------------- 图
fig, ax = plt.subplots(3, 4, figsize=(15, 10))
def show(a, img, title, cmap="viridis"):
    im = a.imshow(img, cmap=cmap); a.set_title(title, fontsize=10)
    a.set_xticks([]); a.set_yticks([]); plt.colorbar(im, ax=a, fraction=0.046)
show(ax[0,0], X, "input 28x28 (4x4 bright block)")
show(ax[0,1], c1[0,0], "conv1 ch0 (identity) 28x28")
show(ax[0,2], c1[0,2], "conv1 ch2 (vertical edge) 28x28")
show(ax[0,3], c1[0,3], "conv1 ch3 (horizontal edge)", "coolwarm")
show(ax[1,0], p1[0,0], "maxpool -> 14x14, ch0")
show(ax[1,1], p1[0,2], "maxpool -> 14x14, ch2")
show(ax[1,2], c2[0,0], "conv2 ch0 14x14")
show(ax[1,3], p2[0,0], "maxpool -> 7x7, ch0")
im = ax[2,0].imshow(p2[0,2], cmap="viridis"); ax[2,0].set_title("7x7, ch2", fontsize=10)
plt.colorbar(im, ax=ax[2,0], fraction=0.046); ax[2,0].set_xticks([]); ax[2,0].set_yticks([])
im = ax[2,1].imshow(c3[0,0], cmap="viridis"); ax[2,1].set_title("3x3 = 9 token positions", fontsize=10)
for i in range(3):
    for j in range(3):
        ax[2,1].text(j, i, "%.3f" % c3[0,0,i,j], ha="center", va="center", fontsize=9)
ax[2,1].set_xticks([]); ax[2,1].set_yticks([]); plt.colorbar(im, ax=ax[2,1], fraction=0.046)
im = ax[2,2].imshow(tokens.numpy(), cmap="viridis", aspect="auto")
ax[2,2].set_title("9 tokens x 8 dims", fontsize=10); ax[2,2].set_xlabel("token"); ax[2,2].set_ylabel("dim")
plt.colorbar(im, ax=ax[2,2], fraction=0.046)
ax[2,3].bar(range(1,10), tokens.norm(dim=-1).numpy())
ax[2,3].set_title("L2 norm of each token", fontsize=10); ax[2,3].set_xlabel("token")
fig.suptitle("28x28 toy: conv -> pool -> conv -> pool -> 5x5 conv -> 9 tokens", fontsize=13)
fig.tight_layout(); fig.savefig("18_toy_cnn.png", dpi=140, bbox_inches="tight")
print("\nsaved 18_toy_cnn.png")
