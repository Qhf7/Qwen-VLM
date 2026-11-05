# Data protocol

This is a new supervised industrial inspection project. VisA is re-split for training and evaluation; these results must not be presented as the official VisA normal-only anomaly detection benchmark. All images, masks, labels and derived examples remain in one source split. No test annotation is used to construct training examples or sampling thresholds.

## Prepare VisA

The input is the extracted official VisA layout, including `split_csv/1cls.csv` with columns `object,split,label,image,mask`. Image and mask paths resolve relative to the extracted data root; a single containing archive directory and paths prefixed with that directory name are supported. The original CSV split is retained as metadata, not used for this custom supervised split.

```bash
PYTHONPATH=src python -m industrial_vlm.data \
  --data-root /data/VisA --output /data/industrial_vlm/visa \
  --train-size 768 --eval-size 120 --seed 42
```

Run from the project directory. `prepare_visa` may optionally precede the flags. Only Python and Pillow are required. The command reads original files and writes manifests; it does not copy, resize, crop, train, or download anything. Paths in output instruction rows are absolute, so the prepared data must remain mounted at the same path or be regenerated for another machine.

## Validation and exact duplicates

Images must decode successfully. Every anomaly needs an existing, nonempty mask of exactly the image dimensions. A normal image may have no mask, or an empty mask. A foreground mask on a normal image is rejected. **Original VisA masks are class-indexed:** 0 is background and every positive integer defect-class ID is foreground. Values such as `{0,1,2,7}` are valid; masks are binarized with `value > 0` for area and union-box extraction. Original values and `mask_label_semantics` remain in the manifest. Palette-indexed VisA masks retain their integer indices rather than palette display colors. MVTec masks, separately, must be binary with values within `{0,1}` or `{0,255}`. Grayscale RGB masks are supported; colored masks are rejected. Missing files, unknown labels, corrupt images/masks and inconsistent annotations are recorded in `rejects.jsonl`; no missing mask is silently interpreted as a normal anomaly label.

Coordinates use stored image pixels: no EXIF transpose is applied. The manifest records `exif_orientation` and `coordinate_frame`; nontrivial EXIF rotations/flips (orientation other than absent/1) are rejected so a mask cannot silently become misaligned. Training and inference must decode with `Image.open(path).convert("RGB")` using the same stored-pixel frame. Any later crop/resize/flip must transform boxes consistently.

SHA256 is computed on each original image file's bytes before splitting. Identical hashes form one group. Consistently annotated copies are reduced to one deterministic canonical image, with all source aliases in `manifest.jsonl` and `duplicates.jsonl`. The canonical path is selected by category, relative path, then CSV row. A same-label duplicate spanning categories is assigned to its canonical category and the original categories remain in aliases. Conflicting labels or binary foreground masks reject the entire hash group. Exact duplicates cannot enter different splits. This does not detect recompressed duplicates or distinct views of the same physical part; claims of physical-instance disjointness would require additional product-instance metadata.

Accepted unique sources are stratified by `(category, is_anomalous)` and shuffled reproducibly with seed 42. Each stratum is apportioned 70% train / 15% validation / 15% test by largest remainder, with ties in train/validation/test order. Very small strata can have an empty holdout; actual counts appear in the audit report. Stable source IDs are `visa_<full-original-image-sha256>`. The seed, sorted source IDs and stratum determine assignment; CSV row order does not affect a unique image's split. Changing the accepted data inventory can change the split, so archive `manifest.jsonl`, `split.json` and `audit_report.json` with the experiment.

## Boxes and answer contract

Each mask produces one union bounding box covering every foreground pixel. We do not infer separate defect instances from connected components. A mask with several disconnected components therefore yields one possibly loose box; this avoids assuming that component count equals defect count. Normal images use `[]`.

Bounds use pixel edges `(left, top, right-exclusive, bottom-exclusive)`. Coordinates are scaled to `[0,1000]`, with lower edges rounded down and upper edges rounded up so even a one-pixel defect keeps positive box width and height. Full-image foreground maps to `[0,0,1000,1000]`. Coordinates always refer to the entire unmodified input image. Do not evaluate these union boxes as instance detection labels.

Instruction row fields are:

```json
{"id":"visa_<sha256>__region","source_id":"visa_<sha256>","image":"/absolute/image.png","category":"candle","is_anomalous":true,"boxes":[[100,200,350,600]],"prompt":"...","answer":"{\"is_anomalous\":true,\"boxes\":[[100,200,350,600]]}"}
```

Generation inference receives only `image` and `prompt`. `answer` is used only as an SFT target or for teacher-forced validation loss. `category`, `is_anomalous`, `boxes` and `anomaly_area_fraction` are ground truth metadata, never additional model inputs. `anomaly_area_fraction` is foreground mask pixels divided by original image pixels, with zero for normal images. Prompts are uniform short English instructions. Region answers contain the boolean and boxes. Decision-only prompts ask only for a defect decision, and their answers contain only `{"is_anomalous":true}` or `false`; their top-level boxes remain ground truth metadata for audit. Decision examples use the same source IDs as their region counterparts and an `__decision` ID suffix.

## Honest sampling and supervision ablations

The default 768 is the total number of examples in each training arm, not the count per category. Capacity-aware equal quotas are allocated over category/label strata; normal/anomaly and category counts therefore match as closely as available data and integer rounding permit. A small stratum's unused quota is redistributed. Requesting more training samples than available training sources fails rather than repeating images.

- `train_quality_region.jsonl`: within each anomalous category's training sources, mask area divided by original image area is binned using empirical one-third and two-thirds quantiles computed from **training anomalies only**. Sampling covers nonempty small/medium/large bins with equal, capacity-aware quotas. Equal area values can produce empty bins, which are not fabricated. Normal strata use the exact same randomly selected source images as the random arm.
- `train_random_region.jsonl`: uniform seeded random sampling without replacement inside exactly the same category/label quotas. It uses the same valid candidate pool and verified labels. It may overlap with the quality arm; overlap is recorded, not prevented or hidden.
- `train_quality_decision.jsonl`: exactly the quality-region source set, in the same order, with the box requirement removed from prompt and answer. This measures added regional supervision at matched source images. Classification evaluation should use the paired decision-only prompts; localization scores are not claimed for that arm.

“Quality” names **defect area coverage**, not cleaner annotations, better reasoning, or deliberately corrupted random data. Both arms use valid labels. Normal membership is matched exactly. Within each category/label block selected sources are sorted by source ID, then both arms receive the same seeded global permutation, matching the category/label sequence at every training-file position. Only anomalous membership varies; downstream dataloader shuffling must also use the same seed and batching settings across arms to preserve this control. The audit records `normal_source_ids_equal`, `category_label_sequence_equal`, area thresholds, selection overlap and actual quotas. Membership differences beyond this sampling factor, or adding extra examples to one arm, invalidate this ablation. The project does not generate synthetic reasoning chains from labels.

## Output files and evaluation discipline

`manifest.jsonl` contains all accepted, deduplicated original sources with dimensions, mask area, union boxes, hashes, aliases and assigned split. `split.json` contains train/val/test source-ID arrays. Every original image in validation and test remains in `val.jsonl` / `test.jsonl`; neither file is just a 120-image sample.

Each holdout also has a reproducibly sampled, capacity-aware category/label-balanced subset: `val_balanced_120.jsonl` and `test_balanced_120.jsonl` by default. The numeric suffix follows `--eval-size`. When fewer sources are available, all available sources are used and the audit records requested versus actual size. If all category/label strata can supply the quota, their counts differ by at most one; scarce or missing strata are explicitly visible. These are separate validation and test sets, never a combined 120-image pool.

Paired decision-only versions are `val_decision.jsonl`, `test_decision.jsonl`, `val_balanced_120_decision.jsonl`, and `test_balanced_120_decision.jsonl`, with exactly the same sources as their region counterparts. Use validation for hyperparameters, prompt choice, training length and parsing rules. Freeze the protocol before test evaluation; do not select models by the test subset or use test masks to pick crops. Report full holdout or clearly name the balanced subset and its actual sample count. Archive reject and duplicate reports even when empty.

## Independent MVTec AD external evaluation

```bash
PYTHONPATH=src python -m industrial_vlm.data prepare_mvtec \
  --data-root /data/mvtec_anomaly_detection \
  --output /data/industrial_vlm/mvtec_external --eval-size 120 --seed 42
```

The reader accepts standard extracted tarball / Hugging Face file-mirror directory layouts: optional `category/train/good`, `category/test/good`, `category/test/<defect>`, and `category/ground_truth/<defect>/<stem>_mask.png` (also `<stem>.png`). A test-only partial download is allowed. One containing archive directory is allowed. An Arrow/Parquet dataset export without these image directories is not the same layout and must first be explicitly converted.

`external_manifest.jsonl` records all valid originals. `external_train_normal_manifest.jsonl` preserves official train/good as **reference-only**, not new SFT training data; it can be empty for test-only downloads. `external_test.jsonl` and `external_test_balanced_120.jsonl` contain supplied official test images; matching `_decision.jsonl` versions share sources. `split.json` uses `reference_only` and `external_test` keys. Duplicate hashes across official train and test are quarantined in the reject report; ambiguous labels/masks are handled like VisA. No training instructions are generated in this mode. The command does not verify the completeness of the official dataset. Report a 150-image mirror selection as an **external pilot subset**, with its actual count, selection rule, mirror and revision; never label it the complete 1,725-image MVTec test. MVTec external test is used after VisA-based model/prompt selection. If future experiments condition on train-normal references, report that additional inference access and keep reference IDs separate from test IDs.
