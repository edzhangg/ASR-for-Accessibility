"""hard-speaker 50/50 sampling for Qwen fine-tuning.

Uses precomputed sampling probabilities from training_sampling_weights.csv.
Does not load audio or require a GPU.
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


def load_speaker_weights(path):
    """Read precomputed E3 probabilities for each speaker."""
    speakers = {}

    required = {
        "speaker_id",
        "hard_top25",
        "n_manifest",
        "e3_hard50_p_speaker",
        "e3_per_recording_probability",
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
            p_speaker = float(row["e3_hard50_p_speaker"])
            p_recording = float(row["e3_per_recording_probability"])

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


def build_sampling_weights(train_rows, speaker_info):
    """Assign an E3 probability to each recording."""

    # Count recordings per speaker in the actual training manifest
    counts = Counter(row["speaker_id"] for row in train_rows)

    # Every training speaker must appear in the weights CSV
    if set(counts) != set(speaker_info):
        raise ValueError("Manifest and weights CSV have different speakers")

    # Verify the CSV matches the training manifest
    for speaker_id, count in counts.items():
        expected = speaker_info[speaker_id]["n"]

        if count != expected:
            raise ValueError(
                f"Recording count mismatch for {speaker_id}: "
                f"manifest={count}, CSV={expected}"
            )

    # Assign each recording its precomputed probability
    weights = [
        speaker_info[row["speaker_id"]]["p_recording"]
        for row in train_rows
    ]

    # Check that probabilities sum to 1
    total = sum(weights)

    if not math.isclose(total, 1.0, abs_tol=1e-5):
        raise ValueError(
            f"Sampling probabilities sum to {total}, expected 1"
        )

    # Check that hard speakers receive 50% of probability mass
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


def make_sampler(train_rows, speaker_info, num_samples=None, seed=42):
    """Construct a PyTorch WeightedRandomSampler."""

    import torch
    from torch.utils.data import WeightedRandomSampler

    weights = build_sampling_weights(train_rows, speaker_info)

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
