"""Weighted sampling of training recordings for Qwen fine-tuning (strategies E1-E4).

Reads the per-speaker probability columns of the weights CSV (built by
make_sampling_weights.py) and turns them into a PyTorch WeightedRandomSampler:

  e1  uniform over recordings      e2  uniform over speakers
  e3  hard-speaker 50/50           e4  difficulty-weighted

Key rules
  * A speaker's probability mass comes from the CSV; it is split evenly over that
    speaker's recordings *in the manifest actually used for training*. The manifest
    may therefore be smaller than the one the CSV was built from (clips over
    --max-duration are dropped) but never larger.
  * The probabilities must sum to 1 and, for e3 only, hard speakers must hold 50%.
  * Sampling is with replacement; the generator is seeded once and keeps advancing,
    so different epochs get different, reproducible draws.

Does not load audio or require a GPU (torch is imported only when a sampler is built).
Run it directly to simulate the sampler and print the observed hard-speaker share:

  python hard_speaker_sampler.py --manifest train.jsonl --weights weights.csv
"""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def load_manifest(path):
    """Load training recordings from train.jsonl."""
    rows = []

    with Path(path).open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            row = json.loads(line)

            if not row.get("speaker_id"):
                raise ValueError(
                    f"Missing speaker_id at line {line_number}"
                )

            if row.get("split") != "train":
                raise ValueError(
                    f"Non-training recording at line {line_number}"
                )

            rows.append(row)

    if not rows:
        raise ValueError("Training manifest is empty")

    return rows


STRATEGY_COLUMNS = {
    # strategy -> (per-speaker probability column, per-recording probability column)
    "e1": ("e1_uniform_recording_p_speaker", "e1_per_recording_probability"),
    "e2": ("e2_speaker_balanced_p_speaker", "e2_per_recording_probability"),
    "e3": ("e3_hard50_p_speaker", "e3_per_recording_probability"),
    "e4": ("e4_difficulty_p_speaker", "e4_per_recording_probability"),
}


def load_speaker_weights(path, strategy="e3"):
    """Read precomputed sampling probabilities for each speaker."""
    speakers = {}
    speaker_col, recording_col = STRATEGY_COLUMNS[strategy]

    required = {
        "speaker_id",
        "hard_top25",
        "n_manifest",
        speaker_col,
        recording_col,
    }

    with Path(path).open(
        encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Weights CSV is missing required columns")

        for row in reader:
            speaker_id = row["speaker_id"].strip()

            if not speaker_id or speaker_id in speakers:
                raise ValueError("Missing or duplicate speaker ID")

            if row["hard_top25"] not in ("0", "1"):
                raise ValueError(
                    f"Invalid hard_top25 for {speaker_id}"
                )

            n = int(row["n_manifest"])
            p_speaker = float(row[speaker_col])
            p_recording = float(row[recording_col])

            if n <= 0:
                raise ValueError(f"Invalid recording count for {speaker_id}")

            if not all(
                math.isfinite(p) and p > 0
                for p in (p_speaker, p_recording)
            ):
                raise ValueError(f"Invalid probabilities for {speaker_id}")

            # Check: P(recording) = P(speaker) / number of recordings
            if not math.isclose(
                p_speaker / n, p_recording,
                rel_tol=1e-5, abs_tol=1e-12
            ):
                raise ValueError(
                    f"Inconsistent probabilities for {speaker_id}"
                )

            speakers[speaker_id] = {
                "hard": row["hard_top25"] == "1",
                "n": n,
                "p_speaker": p_speaker,
                "p_recording": p_recording,
            }

    if not speakers:
        raise ValueError("Weights CSV is empty")

    return speakers


def build_sampling_weights(train_rows, speaker_info, check_hard_mass=True):
    """Assign a sampling probability to each recording.

    check_hard_mass enforces the E3 rule (hard speakers hold 50% of the mass);
    pass False for E1, which has no such constraint.
    """

    # Count recordings per speaker in the actual training manifest
    counts = Counter(row["speaker_id"] for row in train_rows)

    # Every training speaker must appear in the weights CSV
    if set(counts) != set(speaker_info):
        raise ValueError("Manifest and weights CSV have different speakers")

    # The manifest may be a filtered version of the one the CSV was built from
    # (e.g. clips over --max-duration dropped), so it can have FEWER recordings
    # per speaker than the CSV, never more.
    for speaker_id, count in counts.items():
        csv_n = speaker_info[speaker_id]["n"]

        if count > csv_n:
            raise ValueError(
                f"Recording count exceeds CSV for {speaker_id}: "
                f"manifest={count}, CSV={csv_n}"
            )

    # Keep each speaker's probability mass from the CSV and split it evenly over
    # the recordings that actually exist, so the probabilities still sum to 1.
    weights = [
        speaker_info[row["speaker_id"]]["p_speaker"] / counts[row["speaker_id"]]
        for row in train_rows
    ]

    # Check that probabilities sum to 1
    total = sum(weights)

    if not math.isclose(total, 1.0, abs_tol=1e-5):
        raise ValueError(
            f"Sampling probabilities sum to {total}, expected 1"
        )

    # Check that hard speakers receive 50% of probability mass
    if check_hard_mass:
        hard_mass = sum(
            weight
            for row, weight in zip(train_rows, weights)
            if speaker_info[row["speaker_id"]]["hard"]
        )

        if not math.isclose(hard_mass, 0.5, abs_tol=1e-5):
            raise ValueError(
                f"Hard-speaker sampling share is {hard_mass:.4f}, "
                "expected 0.5"
            )

    return weights


def make_sampler(train_rows, speaker_info, num_samples=None, seed=42,
                 check_hard_mass=True):
    """Construct a PyTorch WeightedRandomSampler.

    The generator is created once and advances across epochs, so each epoch gets
    different (but reproducible) draws.
    """

    import torch
    from torch.utils.data import WeightedRandomSampler

    weights = build_sampling_weights(train_rows, speaker_info, check_hard_mass)

    if num_samples is None:
        num_samples = len(train_rows)

    generator = torch.Generator().manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )

    return sampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draws", type=int, default=100000)
    args = parser.parse_args()

    if args.draws <= 0:
        parser.error("--draws must be positive")

    train_rows = load_manifest(args.manifest)
    speaker_info = load_speaker_weights(args.weights)

    sampler = make_sampler(
        train_rows,
        speaker_info,
        num_samples=args.draws,
        seed=args.seed,
    )

    # Simulate sampling and measure the actual hard-speaker share
    hard_count = sum(
        speaker_info[train_rows[i]["speaker_id"]]["hard"]
        for i in sampler
    )

    print(f"Training recordings: {len(train_rows):,}")
    print(f"Training speakers: {len(speaker_info)}")
    print(f"Hard speakers: {sum(s['hard'] for s in speaker_info.values())}")
    print("Expected hard-speaker share: 50.00%")
    print(f"Observed hard-speaker share: {hard_count / args.draws:.2%}")


if __name__ == "__main__":
    main()
