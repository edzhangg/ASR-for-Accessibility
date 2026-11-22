"""Category-first, hard-threshold speaker sampler (CPU-compatible).

Sampling: category -> hard/non-hard group -> speaker -> recording.
Categories not listed in the config receive probability zero.
The manifest order is preserved, including zero-weight excluded rows.
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def load_category_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    proportions = config.get("category_proportions")
    if not isinstance(proportions, dict) or not proportions:
        raise ValueError("category_proportions must be a nonempty mapping")
    if any(not isinstance(k, str) or not k.strip() for k in proportions):
        raise ValueError("Category names must be nonempty strings")
    if any(not isinstance(v, (int, float)) or isinstance(v, bool)
           or not math.isfinite(v) or v <= 0 for v in proportions.values()):
        raise ValueError("All configured category probabilities must be finite and positive")
    if not math.isclose(sum(proportions.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Category proportions must sum to 1")
    hard_probability = config.get("hard_group_probability", 0.5)
    if (not isinstance(hard_probability, (int, float))
            or isinstance(hard_probability, bool)
            or not math.isfinite(hard_probability)
            or not 0 < hard_probability < 1):
        raise ValueError("hard_group_probability must be strictly between 0 and 1")
    return {"category_proportions": proportions,
            "hard_group_probability": float(hard_probability)}


def build_category_first_weights(train_rows, speaker_info, config):
    """Return one probability per input row, in unchanged manifest order.

    speaker_info: {speaker_id: {'hard': bool, ...}} as produced by
    hard_speaker_sampler.load_speaker_weights(..., strategy='e3').
    """
    proportions = config["category_proportions"]
    hard_probability = config["hard_group_probability"]
    if not train_rows:
        raise ValueError("Empty training rows")
    counts = Counter()
    speakers_by_group = defaultdict(set)
    for index, row in enumerate(train_rows):
        speaker = row.get("speaker_id")
        category = row.get("category")
        if not speaker or not category:
            raise ValueError(f"Row {index} requires speaker_id and category")
        if speaker not in speaker_info:
            raise ValueError(f"Speaker {speaker!r} missing from weights CSV")
        if category not in proportions:
            continue
        group = bool(speaker_info[speaker]["hard"])
        counts[(speaker, category)] += 1
        speakers_by_group[(category, group)].add(speaker)

    for category in proportions:
        for group in (True, False):
            if not speakers_by_group[(category, group)]:
                label = "hard" if group else "non-hard"
                raise ValueError(f"Category {category!r} has no {label} speakers")

    weights = []
    for row in train_rows:
        speaker, category = row["speaker_id"], row["category"]
        if category not in proportions:
            weights.append(0.0)
            continue
        group = bool(speaker_info[speaker]["hard"])
        group_probability = hard_probability if group else 1.0 - hard_probability
        weight = (proportions[category] * group_probability
                  / len(speakers_by_group[(category, group)])
                  / counts[(speaker, category)])
        weights.append(weight)

    if not math.isclose(math.fsum(weights), 1.0, abs_tol=1e-9):
        raise AssertionError("Sampling probabilities do not sum to 1")
    for category, target in proportions.items():
        actual = math.fsum(w for row, w in zip(train_rows, weights)
                           if row["category"] == category)
        if not math.isclose(actual, target, abs_tol=1e-9):
            raise AssertionError(f"Category {category!r}: {actual} != {target}")
    actual_hard = math.fsum(w for row, w in zip(train_rows, weights)
                            if speaker_info[row["speaker_id"]]["hard"])
    if not math.isclose(actual_hard, hard_probability, abs_tol=1e-9):
        raise AssertionError("Hard-speaker mass does not match config")
    return weights


def make_category_first_sampler(train_rows, speaker_info, config, *,
                                num_samples=None, seed=42):
    """Create a replacement WeightedRandomSampler; requires PyTorch, not CUDA."""
    import torch
    from torch.utils.data import WeightedRandomSampler

    weights = build_category_first_weights(train_rows, speaker_info, config)
    count = len(train_rows) if num_samples is None else num_samples
    if count <= 0:
        raise ValueError("num_samples must be positive")
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=count,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
