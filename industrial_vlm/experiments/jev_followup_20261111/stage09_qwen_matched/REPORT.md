# Qwen共同448输入比较

Qwen量化底座与DD LoRA采用stage08新Jev shared0逐字节相同的120张PNG，每次只含一张query、无参考图。
复用原生score._score_image：只对JSON布尔值token计分，保留共同前后缀核验与causal shift。

| 模型 | AUROC | AP | Recall(logodds>0) | FPR(logodds>0) |
|---|---:|---:|---:|---:|
| base | 0.824444 | 0.859678 | 0.233333 | 0.000000 |
| decision_decision | 0.934861 | 0.943288 | 0.716667 | 0.050000 |

实际native image_grid_thw=[1,28,28]，merged视觉token=196/图；240图像判断对应480次候选forward，无额外GPU预热。

## 解释边界

- Previously viewed development validation; no blind confirmation or deployment claim.
- Exact same stage08 query PNG bytes, one image and no references in both Qwen arms.
- Qwen JSON boolean prompt/scoring and Jev typed Noul protocol differ.
- Jev internal visual tokens and compute are unknown; matching pixels/images is not matching FLOPs or tokens.
- Qwen two-candidate normalized score is not calibrated confidence.
- Historical Qwen original-resolution scores and historical Jev renderer remain separate evidence.
