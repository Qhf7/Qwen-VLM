"""B0+ 预注册协议。

这个文件在运行 P0/P1/P2 之前冻结；不根据结果追加 prompt。
"""

from __future__ import annotations

PROTOCOL_VERSION = "B0plus-preregistered-v1"
PREREGISTERED_AT = "2026-09-01"

P0 = """You are inspecting one view of an industrial component for visible defects.
A defect means a visible scratch, crack, deformation, discoloration, stain, missing or broken part, pit, burn mark, coating problem, contaminant, or another localized abnormality. Normal edges, holes, reflections, shadows, text, and regular texture are not defects.

The first generated token must be exactly Yes or No. Then output exactly one JSON object with a region selected from: top-left, top-center, top-right, middle-left, center, middle-right, bottom-left, bottom-center, bottom-right, none.
Format: Yes {"region":"center"}
For a normal image use: No {"region":"none"}
Do not output Markdown or an explanation."""

P1 = """Inspect this image of an industrial component and classify only what is visibly present.

Normal: the component is intact and its edges, holes, reflections, shadows, text, color, and repeated texture are internally consistent.
Anomalous: a localized region visibly breaks that consistency through damage, missing material, contamination, deformation, or an irregular surface or color change.

Treat the normal and anomalous hypotheses symmetrically. The first generated token must be exactly Yes for anomalous or No for normal. Then output exactly one JSON object whose region is one of: top-left, top-center, top-right, middle-left, center, middle-right, bottom-left, bottom-center, bottom-right, none.
Use No {"region":"none"} when the normal hypothesis is stronger. Do not output Markdown or an explanation."""

P2 = """Compare two hypotheses for the pictured <object> using only visible image evidence.

H0: this is a normal instance; visible structure and appearance are mutually consistent.
H1: this instance contains a localized anomaly; some visible region is inconsistent with the rest of the object.

Choose the better-supported hypothesis. The first generated token must be exactly No for H0 or Yes for H1. Then output exactly one JSON object whose region is one of: top-left, top-center, top-right, middle-left, center, middle-right, bottom-left, bottom-center, bottom-right, none.
For H0 use No {"region":"none"}. Do not output Markdown or an explanation."""

# This post-registration diagnostic deliberately reverses P0's label wording:
# Yes means normal and No means defective.  The runner records P(No) as the
# anomaly score, rather than relying on a prompt-specific column name.
P3_COMPONENT_FIRST = """Inspect one view of an industrial component.

First identify internally what kind of component it is and its expected repeated structure, edges, holes, markings, color, and surface. Then decide whether any visible local region breaks that expected structure.

The first generated token must be exactly Yes or No. Yes means the component is normal. No means the component has a visible defect.
A defect includes a visible scratch, crack, deformation, discoloration, stain, missing or broken material, pit, burn mark, coating problem, contaminant, or other localized inconsistency. Normal edges, holes, reflections, shadows, text, and regular texture are not defects.

Then output exactly one JSON object. Its component value must be a short component-type description, and its region must be one of: top-left, top-center, top-right, middle-left, center, middle-right, bottom-left, bottom-center, bottom-right, none.
For normal use: Yes {"component":"short component type","region":"none"}
For defective use: No {"component":"short component type","region":"center"}
Do not output Markdown or an explanation."""

P4_CLASSIFICATION_ONLY = """Inspect one view of an industrial component.

First infer internally what kind of component it is and what its normal structure, edges, holes, markings, color, and surface should look like. Then decide only whether this image contains a visible defect.

A defect means a visible scratch, crack, deformation, discoloration, stain, missing or broken material, pit, burn mark, coating problem, contaminant, or another localized abnormality. Normal edges, holes, reflections, shadows, text, and regular texture are not defects.

The first generated token must be exactly Yes or No. Yes means the image contains a visible defect. No means the image is normal.
Do not locate or segment the defect. Do not output JSON, Markdown, or an explanation."""

PROMPTS = {
    "P0": P0,
    "P1": P1,
    "P2": P2,
    "P3_COMPONENT_FIRST": P3_COMPONENT_FIRST,
    "P4_CLASSIFICATION_ONLY": P4_CLASSIFICATION_ONLY,
}

FORWARD_GENERATE_ATOL = 1e-5
BATCH_SINGLE_ATOL = 5e-4
