# v9 architecture refinement log

This log records the static refinement passes required by the local
`drawio-diagram-builder` skill. Browser automation is not available in this
workspace, so no browser screenshot is claimed; the draw.io preview HTML is
generated for manual opening.

## Pass 1 — structural preflight

- Found: the old diagram had nested swimlanes counted as overlaps, arrows
  crossing parent containers, and inconsistent color/stroke families.
- Fix: created `v9_best_hybrid_vlm_architecture_paper.drawio` with flat,
  non-overlapping editable cards, explicit waypoints, and five semantic fills.

## Pass 2 — connector audit

- Found: pseudo-label and calibration routes could cross intermediate cards.
- Fix: routed those paths through deliberate whitespace corridors between
  feature, head, and output cards.

## Pass 3 — final static gate

- Result: `validate_visual_quality.py` reports `FAIL: 0`; strict
  `validate_drawio.py` reports `OK`.
- Remaining warnings are density/spacing suggestions only; no clipped,
  overlapping, off-page, or invalid-geometry findings remain.
