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





def _continuous_weights(
    train_rows: Sequence[Mapping],
    speaker_wer: Mapping[str, float],
    alpha: float,
    epsilon: float,
    max_wer: float,
) -> list[float]:
    """One recording weight: (epsilon + min(speaker_WER, max_wer)) ** alpha.

    WER is a fraction (e.g. 0.15 = 15%). Each recording of the same speaker
    gets the same weight. When alpha=0, recording weights are uniform.
    """
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and >= 0")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and > 0")
    if not math.isfinite(max_wer) or max_wer <= 0:
        raise ValueError("max_wer must be finite and > 0")

    weights = []
    for i, row in enumerate(train_rows):
        speaker_id = row.get("speaker_id")
        if not speaker_id or speaker_id not in speaker_wer:
            raise ValueError(f"Missing speaker ID / TRAIN WER at row {i}")
        wer = float(speaker_wer[speaker_id])
        if not math.isfinite(wer) or wer < 0:
            raise ValueError(f"Invalid speaker WER at row {i}")
        weights.append((epsilon + min(wer, max_wer)) ** alpha)
    return weights





# =========== TO DO ====================
def _hard_threshold_weights(
    train_rows: Sequence[Mapping], speaker_wer: Mapping[str, float]
) -> list[float]:
    """TODO (teammate): implement the team's hard-threshold strategy.

    Return ONE positive weight PER RECORDING, in train_rows order.
    Decide the hard threshold and hard-group probability allocation together.
    Do not build another DataLoader or apply category balancing here; the shared
    functions below already handle normalization and category proportions.
    """
    raise NotImplementedError("hard_threshold is reserved for a teammate")






def calculate_probabilities(
    train_rows: Sequence[Mapping],
    speaker_wer: Mapping[str, float],
    *,
    strategy: str = "continuous",
    alpha: float = 1.0,
    epsilon: float = 0.02,
    max_wer: float = 0.30,
    category_probs: Mapping[str, float] | None = None,
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

    if strategy == "continuous":
        weights = _continuous_weights(train_rows, speaker_wer, alpha, epsilon, max_wer)
    elif strategy == "hard_threshold":
        weights = _hard_threshold_weights(train_rows, speaker_wer)
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
