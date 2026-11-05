"""CPU metrics and controlled map interventions for the migration pilot.

No language model or ImageBind import is required. Invalid generated answers
are abstentions (-1): they are incorrect for accuracy and count as a false
negative for their true class in the binary macro F1. They are never silently
removed. FPR counts explicit Yes answers among all normal images; invalids
are separately reported because a low FPR alone can hide abstention.
"""
import re

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        try:
            return value.numpy()
        except TypeError:  # numpy does not support a bfloat16 tensor directly.
            return value.float().numpy()
    return np.asarray(value)


def _labels(labels):
    values = _numpy(labels).reshape(-1)
    if not len(values) or not np.isin(values, (0, 1)).all():
        raise ValueError("Expected a nonempty sequence of binary labels 0/1")
    return values.astype(np.int64, copy=False)


def _scores(scores, n):
    values = _numpy(scores).astype(np.float64, copy=False).reshape(-1)
    if len(values) != n or not np.isfinite(values).all():
        raise ValueError("Scores must be finite and match the number of labels")
    return values


def _classification(labels, predictions):
    y = _labels(labels)
    pred = np.asarray(predictions).reshape(-1)
    if len(pred) != len(y) or not np.isin(pred, (-1, 0, 1)).all():
        raise ValueError("Predictions must have matching length and values -1/0/1")
    normal = y == 0
    anomaly = y == 1
    invalid = pred == -1
    return {
        "n": int(len(y)), "n_normal": int(normal.sum()), "n_anomaly": int(anomaly.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
        "recall": float((pred[anomaly] == 1).mean()) if anomaly.any() else None,
        "fpr": float((pred[normal] == 1).mean()) if normal.any() else None,
        "invalid_rate": float(invalid.mean()), "n_invalid": int(invalid.sum()),
        "n_invalid_normal": int((invalid & normal).sum()),
        "n_invalid_anomaly": int((invalid & anomaly).sum()),
    }


def score_metrics(labels, scores, threshold=0.0):
    """Standard sklearn image AUROC/AP and scores > threshold decisions.

    AUROC is undefined (None) for a one-class group. AP is undefined (None)
    when a group has no positives. A score exactly equal to zero predicts No.
    """
    y = _labels(labels)
    s = _scores(scores, len(y))
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    result = _classification(y, (s > threshold).astype(np.int64))
    result.update({
        "image_auroc": float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None,
        "image_ap": float(average_precision_score(y, s)) if y.any() else None,
        "threshold": float(threshold),
    })
    return result


_THINK = re.compile(r"\A<think\s*>.*?</think\s*>\s*", flags=re.I | re.S)
_THINK_END = re.compile(r"\A</think\s*>\s*", flags=re.I)
_FIRST_DECISION = re.compile(r"\A(yes|no)(?![\w])", flags=re.I)


def parse_generated(text):
    """Read only an initial Yes/No after whitespace and leading think tags.

    Complete leading <think>...</think> blocks are ignored. One leading
    </think> is accepted for generation that began inside an already-open
    think block. Unterminated think blocks, prose before a decision, and
    words such as 'yesterday'/'normal' are invalid. Later Yes/No words do
    not repair an invalid prefix. The first valid decision is authoritative.
    """
    if not isinstance(text, str):
        return None
    rest = text.lstrip()
    while True:
        match = _THINK.match(rest)
        if match is None:
            break
        rest = rest[match.end():]
    match = _THINK_END.match(rest)
    if match is not None:
        rest = rest[match.end():]
    match = _FIRST_DECISION.match(rest)
    return int(match.group(1).lower() == "yes") if match else None


def generated_metrics(labels, texts):
    """Score all generated answers, including invalids in the denominator."""
    predictions = [parse_generated(text) for text in texts]
    return _classification(labels, [-1 if pred is None else pred for pred in predictions])


_MACRO_KEYS = ("image_auroc", "image_ap", "accuracy", "macro_f1", "recall", "fpr", "invalid_rate")


def evaluate_predictions(items, scores=None, generated_texts=None):
    """Return overall, per-category and equal-category-mean metrics.

    Counts state how many categories contributed to each mean; undefined
    one-class AUROCs and APs without positives are omitted from that mean.
    Scores and generations remain separate evaluation channels.
    """
    items = list(items)
    if not items or (scores is None and generated_texts is None):
        raise ValueError("Nonempty items and at least one prediction channel are required")
    labels = _labels([item["label"] for item in items])
    categories = [str(item["category"]) for item in items]
    channels = {}
    if scores is not None:
        channels["scores"] = (_scores(scores, len(items)), score_metrics)
    if generated_texts is not None:
        texts = list(generated_texts)
        if len(texts) != len(items):
            raise ValueError("Generated texts must match the number of items")
        channels["generated"] = (texts, generated_metrics)
    overall = {name: metric(labels, values) for name, (values, metric) in channels.items()}
    per_category = {}
    for category in sorted(set(categories)):
        indices = [i for i, value in enumerate(categories) if value == category]
        per_category[category] = {
            name: metric(labels[indices], [values[i] for i in indices])
            for name, (values, metric) in channels.items()
        }
    macro, counts = {}, {}
    for name in channels:
        macro[name], counts[name] = {}, {}
        for key in _MACRO_KEYS:
            values = [metrics[name].get(key) for metrics in per_category.values()]
            values = [value for value in values if value is not None]
            if values:
                macro[name][key] = float(np.mean(values))
                counts[name][key] = len(values)
    return {"overall": overall, "per_category": per_category,
            "category_macro": macro, "category_macro_counts": counts,
            "generated_rule": "initial Yes/No; invalid is wrong and a false negative for its true class"}


def donor_map_indices(items, seed=42, opposite_label=True):
    """Choose deterministic same-category donors with no self pairing.

    Opposite-label selection is a diagnostic control using evaluation labels,
    not part of the inference algorithm. Donors may be reused when category
    class counts differ. Categories without an eligible donor fail explicitly.
    """
    items = list(items)
    labels = _labels([item["label"] for item in items])
    rng = np.random.default_rng(seed)
    result = []
    for index, item in enumerate(items):
        candidates = [j for j, donor in enumerate(items)
                      if j != index and donor["category"] == item["category"]
                      and (not opposite_label or labels[j] != labels[index])
                      and ("path" not in item or donor.get("path") != item["path"])
                      and donor.get("split") == item.get("split")]
        if not candidates:
            raise ValueError("No eligible same-category donor for item %d" % index)
        result.append(int(rng.choice(candidates)))
    return np.asarray(result, dtype=np.int64)


def spatial_shuffle_maps(maps, seed=42):
    """Shuffle every map's pixels, preserving its exact value multiset.

    Accept [H,W] or [N,H,W] numpy arrays or tensors, preserving input type,
    dtype and tensor device. Constant maps cannot produce a different spatial
    control and raise ValueError. A deterministic roll handles the rare case
    where the random permutation leaves a nonconstant map unchanged.
    """
    source = _numpy(maps)
    if source.ndim not in (2, 3) or not source.size or not np.isfinite(source).all():
        raise ValueError("Expected finite nonempty maps [H,W] or [N,H,W]")
    batched = source[None] if source.ndim == 2 else source
    rng = np.random.default_rng(seed)
    result = np.empty_like(batched)
    for index, array in enumerate(batched):
        flat = array.reshape(-1)
        if np.all(flat == flat[0]):
            raise ValueError("Cannot spatially change constant map %d" % index)
        shuffled = flat[rng.permutation(len(flat))]
        if np.array_equal(shuffled, flat):
            shuffled = np.roll(flat, 1)
        result[index] = shuffled.reshape(array.shape)
    result = result[0] if source.ndim == 2 else result
    return maps.new_tensor(result) if hasattr(maps, "new_tensor") else result


def validate_cache(cache, expected_pipeline_version=None, expected_decoder_sha256=None):
    """Validate the shared cache schema without importing model code.

    This validates metadata and finite float32 cached values. It does not
    compute metrics or choose a split/checkpoint.
    """
    for key in ("pipeline_version", "decoder_sha256", "items", "cls", "maps"):
        if key not in cache:
            raise ValueError("Cache is missing %s" % key)
    if not isinstance(cache["pipeline_version"], str) or not cache["pipeline_version"]:
        raise ValueError("Cache requires a nonempty pipeline_version")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", cache["decoder_sha256"]):
        raise ValueError("Cache decoder_sha256 must be a SHA256 hex digest")
    for key, expected in (("pipeline_version", expected_pipeline_version),
                          ("decoder_sha256", expected_decoder_sha256)):
        if expected is not None and cache[key] != expected:
            raise ValueError("Cache %s does not match expected value" % key)
    items = list(cache["items"])
    if not items:
        raise ValueError("Cache items are empty")
    paths = []
    for item in items:
        if not all(key in item for key in ("path", "category", "label", "split")):
            raise ValueError("Each item requires path/category/label/split")
        if not item["path"] or not item["category"] or item["split"] not in {"train", "val", "test"}:
            raise ValueError("Invalid cache item metadata")
        paths.append(item["path"])
    _labels([item["label"] for item in items])
    if len(set(paths)) != len(paths):
        raise ValueError("Cache source paths must be unique across splits")
    for key, shape in (("cls", (len(items), 1024)), ("maps", (len(items), 224, 224))):
        values = _numpy(cache[key])
        if values.shape != shape or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError("Cache %s must be finite float32 with shape %s" % (key, shape))
        if key == "maps" and (values.min() < 0 or values.max() > 1):
            raise ValueError("Cache maps must be probabilities in [0,1]")
    return {"n": len(items), "pipeline_version": cache["pipeline_version"],
            "decoder_sha256": cache["decoder_sha256"],
            "split_counts": {split: sum(item["split"] == split for item in items)
                             for split in ("train", "val", "test")}}
