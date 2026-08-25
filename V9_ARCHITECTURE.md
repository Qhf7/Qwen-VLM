# V9 Architecture

GitHub renders the following Mermaid diagram directly. The model is a dense
DINOv2-first anomaly detector with a bounded local Qwen3-VL semantic branch.

Editable draw.io source: [v9_best_hybrid_vlm_architecture.drawio](docs/v9_best_hybrid_vlm_architecture.drawio)

Static SVG preview: [v9_architecture.svg](docs/v9_architecture.svg)

Publication-style SVG: [v9_best_hybrid_vlm_architecture.svg](docs/v9_best_hybrid_vlm_architecture.svg)

![Best Hybrid VLM v9 architecture](docs/v9_architecture.svg)

```mermaid
flowchart LR
    A[Official Train normal images\n5 fixed views] --> B[Frozen DINOv2 ViT-B/14]
    B --> C1[1036px multi-layer patch features]
    B --> C2[518px multi-layer patch features]
    C1 --> D1[Reconstruction branch]
    C1 --> D2[Category/view memory bank\n+ global bank]
    C1 --> D3[Pseudo anomaly pixel head]
    C2 --> E[Multi-scale feature fusion]
    D1 --> E
    D2 --> E
    D3 --> E
    F[Train normal calibration\ncategory-stratified] --> E
    G[Procedural pseudo defects\nmicrocrack, low-contrast stain,\nscratch, dent, chip, stain] --> D3
    H[Hard normal supervision\nbackground, contour, highlight,\nstrong texture, normal edge] --> D3
    E --> I[DINO dense control\ncontinuous pixel anomaly map]

    J[Test_A five-view images] --> K[Local Qwen3-VL crop audit]
    K --> K1[is_anomaly + score tiers]
    K --> K2[reason keywords + defect type]
    K --> K3[compact box area]
    K --> K4[crop overlap + five-view agreement]
    K1 --> L[Bounded VLM evidence]
    K2 --> L
    K3 --> L
    K4 --> L
    I --> M[DINO support gate]
    L --> M
    M --> N[VLM residual <= 0.12\nonly where DINO responds]
    I --> O[Dense control submission]
    N --> P[Hybrid submission]
    O --> Q[448x448 grayscale masks\n+ submission.csv]
    P --> Q
```

## Data and supervision

The trainable reconstruction and pseudo-head modules receive only official
normal Train images. Defect masks come from procedural transformations of those
images. Hard normal regions are explicitly supervised as non-anomalous so that
silhouettes, reflections, highlights, and repeated textures are not learned as
defects.

## VLM role

Qwen3-VL does not generate the final pixel mask. It supplies local structured
evidence from crop-level audits. The proposal is converted to a bounded
semantic residual and multiplied by a DINO support gate. Therefore a VLM box
cannot paint a region where the dense visual model has no anomaly response.

## Outputs

The pipeline produces two independently validated archives:

- `submission_v9_dense.zip`: DINO dense-control ablation;
- `submission_v9_hybrid.zip`: dense control plus bounded Qwen3-VL residual.

The v9 run is still in training and has no official leaderboard score yet.
