"""缺陷词全局词表 —— S0 的唯一真相源。

数据集中共出现 22 种原始缺陷字符串，存在三处不一致。所有下游代码
（manifest / splits / 评测 / prompt 构造）只允许通过本模块取词，
不得在别处写字符串字面量或临时映射。

归一化决策与理由记录在 `NORMALIZATION_NOTES`，改动请连同理由一起改。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 原始标签 → 归一化标签
# ---------------------------------------------------------------------------

#: 大小写 / 后缀不统一，实为同一缺陷。三者分别出现在
#: PLCC_socket(Porosity) / audio_jack_socket(porosity) / nylon_ball_head(PorosityDS)。
_POROSITY_ALIASES = ("Porosity", "porosity", "PorosityDS")

#: ImpactDamage 仅出现在 smd_receiver_module，是 Damage 的子类。
#: 默认**不合并**——见 NORMALIZATION_NOTES 第 2 条。
_IMPACT_DAMAGE = "ImpactDamage"

_BASE_MAP = {alias: "Porosity" for alias in _POROSITY_ALIASES}

#: 归一化后的规范词表（20 类，按出现的类别数降序）。
CANONICAL_DEFECTS: tuple[str, ...] = (
    "Scratch",                # 12 / 24 类
    "Deformation",            # 10
    "Damage",                 # 10
    "Discoloration",          #  9
    "OilStain",               #  7
    "GelatinousContaminant",  #  7
    "Fracture",               #  6
    "BurnMark",               #  5
    "Pit",                    #  4
    "Warpage",                #  4
    "MissingPart",            #  4
    "Porosity",               #  3  (合并自 Porosity/porosity/PorosityDS)
    "Abrasion",               #  3
    "CharacterOmission",      #  3
    "UnevenCoating",          #  2
    "BrokenWire",             #  2
    "MetallicContaminant",    #  2
    "FiberorBurrs",           #  1  独有词 → 3_adapter
    "TwistedWire",            #  1  独有词 → silicon_cell_sensor
    "ImpactDamage",           #  1  独有词 → smd_receiver_module
)

#: 归一化后只出现在单一类别的缺陷词。这些词在其他类别里学不到，
#: 是零样本的天然难点，也是 holdout 划分必须记录的量。
UNIQUE_DEFECTS: tuple[str, ...] = ("FiberorBurrs", "TwistedWire", "ImpactDamage")

#: 覆盖 5 个及以上类别的高频词。holdout 里必须有，用来分离
#: "缺陷语义迁移成功" 与 "这个类恰好简单"。
HIGH_FREQ_DEFECTS: tuple[str, ...] = (
    "Scratch", "Deformation", "Damage", "Discoloration",
    "OilStain", "GelatinousContaminant", "Fracture", "BurnMark",
)

NORMALIZATION_NOTES = """\
1. Porosity / porosity / PorosityDS → Porosity
   大小写与后缀差异，实为同一缺陷。合并后 Porosity 覆盖 3 个类别，
   因此**不再是独有词**——这一点直接影响 holdout 划分，见第 3 条。

2. ImpactDamage 默认保持独立，不并入 Damage
   理由：Damage 已覆盖 10 个类别、语义极宽；把 ImpactDamage 并进去
   等于抹掉一个可用的独有词。而独有词是零样本能力上界的探针——
   归一化后全数据集只剩 3 个独有词（见第 3 条），非常稀缺，不宜再减。
   若要复现 19 类口径，用 normalize(..., merge_impact_damage=True)。

3. 归一化的副作用：独有词从 6 个降到 3 个
   归一化前，6 个缺陷词各只出现在 1 个类别；合并 Porosity 三兄弟后，
   只剩 FiberorBurrs / TwistedWire / ImpactDamage 三个真正的独有词，
   分属 3_adapter / silicon_cell_sensor / smd_receiver_module。
   → 4 折划分中最多只有 3 折能分到独有词类，第 4 折必然没有。
   这是数据的硬约束，不是划分算法的缺陷，必须在结论里如实说明。
"""


def normalize(raw: str, *, merge_impact_damage: bool = False) -> str:
    """把原始缺陷字符串映射到规范词。

    Args:
        raw: 数据集里 NG 子目录的原始名字。
        merge_impact_damage: 置 True 则 ImpactDamage → Damage（19 类口径）。

    Raises:
        KeyError: 出现词表外的字符串。宁可炸掉也不要静默放行——
            新类别引入新缺陷词时必须显式更新本文件。
    """
    canon = _BASE_MAP.get(raw, raw)
    if merge_impact_damage and canon == _IMPACT_DAMAGE:
        canon = "Damage"
    if canon not in CANONICAL_DEFECTS:
        raise KeyError(
            f"未知缺陷词 {raw!r}（归一化后 {canon!r}）。"
            f"新缺陷词必须显式加入 vocab.CANONICAL_DEFECTS 并说明归并决策，"
            f"不要在调用处打补丁。"
        )
    return canon


def canonical_index(defect: str) -> int:
    """规范词 → 稳定的整数 id（用于分类头 / 混淆矩阵）。"""
    return CANONICAL_DEFECTS.index(defect)


__all__ = [
    "CANONICAL_DEFECTS",
    "UNIQUE_DEFECTS",
    "HIGH_FREQ_DEFECTS",
    "NORMALIZATION_NOTES",
    "normalize",
    "canonical_index",
]
