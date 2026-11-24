"""Shared sampling strategies for Parkinson's ASR fine-tuning.

Only TRAIN metadata is used. This module never loads audio or rewrites a manifest.
`train_rows[i]` must refer to the same recording as `train_dataset[i]`.

Strategies:
- continuous: implemented by the E4 owner (alpha-based difficulty weighting)
- hard_threshold: TODO for the other team member
"""




from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence





def load_speaker_wer_csv(path: str | Path) -> dict[str, float]:
    """Read the team's TRAIN-only difficulty CSV.

    Required columns: speaker_id, difficulty_wer_percent.
    Converts WER percentages to fractions: 15 (%) -> 0.15.
    Confirm separately that the CSV really comes from TRAIN zero-shot inference.
    """
    speaker_wer = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {
            "speaker_id", "difficulty_wer_percent"
        }.issubset(reader.fieldnames):
            raise ValueError("CSV must contain speaker_id and difficulty_wer_percent")
        for line_no, row in enumerate(reader, start=2):
            speaker_id = (row.get("speaker_id") or "").strip()
            if not speaker_id or speaker_id in speaker_wer:
                raise ValueError(f"Missing/duplicate speaker on CSV line {line_no}")
            try:
                wer = float(row["difficulty_wer_percent"]) / 100
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid WER on CSV line {line_no}") from exc
            if not math.isfinite(wer) or wer < 0:
                raise ValueError(f"Invalid WER on CSV line {line_no}")
            speaker_wer[speaker_id] = wer
    if not speaker_wer:
        raise ValueError("Empty speaker WER CSV")
    return speaker_wer


def load_hard_labels_csv(path: str | Path) -> dict[str, bool]:
    """Read the existing hard_top25 labels from the TRAIN difficulty CSV."""
    hard_labels = {}

    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or not {
            "speaker_id", "hard_top25"
        }.issubset(reader.fieldnames):
            raise ValueError("CSV must contain speaker_id and hard_top25")

        for line_no, row in enumerate(reader, start=2):
            speaker_id = (row.get("speaker_id") or "").strip()
            label = (row.get("hard_top25") or "").strip()

            if not speaker_id or speaker_id in hard_labels:
                raise ValueError(
                    f"Missing/duplicate speaker on CSV line {line_no}"
                )

            if label not in {"0", "1"}:
                raise ValueError(
                    f"Invalid hard_top25 on CSV line {line_no}"
                )

            hard_labels[speaker_id] = label == "1"

    if not hard_labels:
        raise ValueError("Empty hard-label CSV")

    return hard_labels






def _continuous_weights(
    train_rows: Sequence[Mapping],
    speaker_wer: Mapping[str, float],
    alpha: float,
    epsilon: float,
    max_wer: float,
) -> list[float]:
    """Compute per-recording weights for speaker-level continuous sampling.

    Speaker weight:
        w_s = (epsilon + min(speaker_WER, max_wer)) ** alpha

    Recording weight:
        w_i = w_s / n_(s,c)

    where n_(s,c) is the number of eligible recordings for speaker s
    in category c.

    WER is a fraction (e.g. 0.15 = 15%).

    After category normalization, speakers are sampled proportionally
    to their difficulty weights, independently of recording counts.

    When alpha=0, speakers are sampled uniformly within each category.
    """
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and >= 0")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and > 0")
    if not math.isfinite(max_wer) or max_wer <= 0:
        raise ValueError("max_wer must be finite and > 0")

    # Count eligible recordings for each (speaker, category) pair
    counts = defaultdict(int)

    for row in train_rows:
        key = (row["speaker_id"], row["category"])
        counts[key] += 1

    weights = []
    for i, row in enumerate(train_rows):
        speaker_id = row.get("speaker_id")
        if not speaker_id or speaker_id not in speaker_wer:
            raise ValueError(f"Missing speaker ID / TRAIN WER at row {i}")
        wer = float(speaker_wer[speaker_id])
        if not math.isfinite(wer) or wer < 0:
            raise ValueError(f"Invalid speaker WER at row {i}")
        speaker_weight = (epsilon + min(wer, max_wer)) ** alpha

        key = (row["speaker_id"], row["category"])
        recording_weight = speaker_weight / counts[key]

        weights.append(recording_weight)

    return weights





# =========== TO DO ====================
def _hard_threshold_weights(
    train_rows: Sequence[Mapping],
    hard_labels: Mapping[str, bool],
) -> list[float]:
    """Compute recording weights for hierarchical hard-threshold sampling.

    Category -> Hard/Other (50/50) -> Speaker (uniform)
             -> Recording (uniform).

    Category probabilities are applied later by calculate_probabilities().
    """
    # Number of eligible recordings for each speaker-category pair
    counts = defaultdict(int)

    # Unique speakers in each (category, hard/other) group
    group_speakers = defaultdict(set)
    categories = set()

    for i, row in enumerate(train_rows):
        speaker = row.get("speaker_id")
        category = row.get("category")

        if not speaker or not category:
            raise ValueError(f"Missing speaker/category at row {i}")
        if speaker not in hard_labels:
            raise ValueError(f"Missing hard label for speaker {speaker}")

        is_hard = hard_labels[speaker]

        counts[(speaker, category)] += 1
        group_speakers[(category, is_hard)].add(speaker)
        categories.add(category)

    # Each category must contain both groups to achieve 50/50.
    for category in categories:
        if not group_speakers[(category, True)]:
            raise ValueError(f"No hard speakers in category {category}")
        if not group_speakers[(category, False)]:
            raise ValueError(f"No other speakers in category {category}")

    weights = []

    for row in train_rows:
        speaker = row["speaker_id"]
        category = row["category"]
        is_hard = hard_labels[speaker]

        n_speakers = len(group_speakers[(category, is_hard)])
        n_recordings = counts[(speaker, category)]

        weight = 0.5 / (n_speakers * n_recordings)
        weights.append(weight)

    return weights






def calculate_probabilities(
    train_rows: Sequence[Mapping],
    speaker_wer: Mapping[str, float],
    *,
    strategy: str = "continuous",
    alpha: float = 1.0,
    epsilon: float = 0.02,
    max_wer: float = 0.30,
    category_probs: Mapping[str, float] | None = None,
    hard_labels: Mapping[str, bool] | None = None,
) -> list[float]:
    """Return P(i) for each recording, aligned to the original manifest order.

    Without category balancing: P(i) = weight_i / sum_j(weight_j).
    With category shares q_c: P(i) = q_c * weight_i / sum_{j in c}(weight_j).

    IMPORTANT: category_probs must be identical across BOTH strategies.
    """
    if not train_rows:
        raise ValueError("Training rows are empty")
    if any(row.get("split", "train") != "train" for row in train_rows):
        raise ValueError("Sampler received a non-TRAIN recording")

    # Validate metadata for both sampling strategies
    for i, row in enumerate(train_rows):
        speaker = row.get("speaker_id")
        category = row.get("category")

        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError(f"Missing speaker ID at row {i}")

        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"Missing category at row {i}")

        if speaker not in speaker_wer:
            raise ValueError(f"Missing speaker WER at row {i}")

        wer = speaker_wer[speaker]

        if not math.isfinite(wer) or wer < 0:
            raise ValueError(f"Invalid speaker WER at row {i}")

    if strategy == "continuous":
        weights = _continuous_weights(train_rows, speaker_wer, alpha, epsilon, max_wer)
    elif strategy == "hard_threshold":
        if hard_labels is None:
            raise ValueError(
                "hard_threshold requires hard_labels from the TRAIN difficulty CSV"
            )
        weights = _hard_threshold_weights(train_rows, hard_labels)
    else:
        raise ValueError("strategy must be 'continuous' or 'hard_threshold'")

    if len(weights) != len(train_rows) or any(
        not math.isfinite(w) or w <= 0 for w in weights
    ):
        raise ValueError("Strategy must return one positive, finite weight per row")

    if category_probs is None:
        total = math.fsum(weights)
        probabilities = [w / total for w in weights]
    else:
        shares = dict(category_probs)
        if not shares or any(not math.isfinite(p) or p <= 0 for p in shares.values()):
            raise ValueError("Each category probability must be positive and finite")
        if not math.isclose(math.fsum(shares.values()), 1.0, abs_tol=1e-9):
            raise ValueError("Category probabilities must sum to 1")
        categories = [row.get("category") for row in train_rows]
        if any(not isinstance(c, str) or not c.strip() for c in categories):
            raise ValueError("Category-aware sampling requires category on every row")
        if set(categories) != set(shares):
            raise ValueError("category_probs must cover all and only TRAIN categories")
        totals = defaultdict(float)
        for category, weight in zip(categories, weights):
            totals[category] += weight
        probabilities = [
            shares[category] * weight / totals[category]
            for category, weight in zip(categories, weights)
        ]

    if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-8):
        raise ValueError("Probabilities do not sum to 1")
    return probabilities







def build_sampler(
    train_rows: Sequence[Mapping],
    speaker_wer: Mapping[str, float],
    *,
    strategy: str = "continuous",
    seed: int = 42,
    alpha: float = 1.0,
    epsilon: float = 0.02,
    max_wer: float = 0.30,
    category_probs: Mapping[str, float] | None = None,
    num_samples: int | None = None,
    hard_labels: Mapping[str, bool] | None = None,
):
    """Build the PyTorch sampler that the shared Trainer will use.

    Draws WITH replacement. One epoch defaults to len(train_rows) draws.
    `train_rows` must be in exactly the same order as the Trainer's Dataset.
    Exact resume requires Trainer-side sampler RNG state restoration.
    """
    import torch
    from torch.utils.data import WeightedRandomSampler

    probabilities = calculate_probabilities(
        train_rows, speaker_wer, strategy=strategy,
        alpha=alpha, epsilon=epsilon, max_wer=max_wer,
        category_probs=category_probs,
        hard_labels=hard_labels,
    )
    draws = len(train_rows) if num_samples is None else num_samples
    if not isinstance(draws, int) or draws <= 0:
        raise ValueError("num_samples must be a positive integer")
    return WeightedRandomSampler(
        weights=torch.tensor(probabilities, dtype=torch.double),
        num_samples=draws,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def summarize_sampling(
    train_rows: Sequence[Mapping],
    probabilities: Sequence[float],
    hard_labels: Mapping[str, bool] | None = None,
) -> dict:
    """Summarize expected sampling exposure."""
    if len(train_rows) != len(probabilities):
        raise ValueError("Rows and probabilities have different lengths")

    category_mass = defaultdict(float)
    speaker_mass = defaultdict(float)
    hard_category_mass = defaultdict(float)

    for row, probability in zip(train_rows, probabilities):
        speaker = row["speaker_id"]
        category = row["category"]

        category_mass[category] += probability
        speaker_mass[speaker] += probability

        if hard_labels is not None and hard_labels[speaker]:
            hard_category_mass[category] += probability

    summary = {
        "num_recordings": len(train_rows),
        "num_speakers": len(speaker_mass),
        "total_probability": math.fsum(probabilities),
        "category_probability": dict(category_mass),
        "speaker_probability": dict(speaker_mass),
    }

    if hard_labels is not None:
        summary["hard_probability"] = math.fsum(
            hard_category_mass.values()
        )

        summary["hard_probability_by_category"] = {
            category: hard_category_mass[category] / mass
            for category, mass in category_mass.items()
        }

    return summary