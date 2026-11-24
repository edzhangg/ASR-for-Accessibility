import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling import (
    calculate_probabilities,
    build_sampler,
    load_speaker_wer_csv,
    load_hard_labels_csv,
    summarize_sampling,
)

def test_continuous_equal_speaker_exposure():
    """Equal-WER speakers must have equal probability,
    regardless of their recording counts.
    """
    category = "Digital Assistant Commands"

    # Create synthetic training recordings
    train_rows = [
        {"speaker_id": "A", "category": category}
        for _ in range(100)
    ] + [
        {"speaker_id": "B", "category": category}
        for _ in range(10)
    ] + [
        {"speaker_id": "C", "category": category}
        for _ in range(50)
    ]

    # Speaker-level WER (fractions, not percentages)
    speaker_wer = {
        "A": 0.10,
        "B": 0.10,
        "C": 0.20,
    }

    # Calculate per-recording probabilities
    probabilities = calculate_probabilities(
        train_rows,
        speaker_wer,
        strategy="continuous",
        alpha=1.0,
        epsilon=0.02,
        max_wer=0.30,
        category_probs={category: 1.0},
    )

    # Aggregate recording probabilities by speaker
    speaker_probs = {"A": 0.0, "B": 0.0, "C": 0.0}

    for row, probability in zip(train_rows, probabilities):
        speaker_probs[row["speaker_id"]] += probability

    # Check probability normalization
    assert sum(probabilities) == pytest.approx(1.0)

    # Check expected speaker probabilities
    assert speaker_probs["A"] == pytest.approx(0.12 / 0.46)
    assert speaker_probs["B"] == pytest.approx(0.12 / 0.46)
    assert speaker_probs["C"] == pytest.approx(0.22 / 0.46)

    # Equal WER must imply equal speaker exposure
    assert speaker_probs["A"] == pytest.approx(speaker_probs["B"])



def test_continuous_alpha_zero():
    """Alpha=0 must produce uniform speaker selection
    within each category.
    """
    category = "Digital Assistant Commands"

    train_rows = [
        {"speaker_id": "A", "category": category}
        for _ in range(100)
    ] + [
        {"speaker_id": "B", "category": category}
        for _ in range(10)
    ] + [
        {"speaker_id": "C", "category": category}
        for _ in range(50)
    ]

    # Different speaker difficulties
    speaker_wer = {
        "A": 0.05,
        "B": 0.15,
        "C": 0.30,
    }

    probabilities = calculate_probabilities(
        train_rows,
        speaker_wer,
        strategy="continuous",
        alpha=0.0,
        category_probs={category: 1.0},
    )

    # Aggregate probabilities by speaker
    speaker_probs = {"A": 0.0, "B": 0.0, "C": 0.0}

    for row, probability in zip(train_rows, probabilities):
        speaker_probs[row["speaker_id"]] += probability

    assert sum(probabilities) == pytest.approx(1.0)

    for probability in speaker_probs.values():
        assert probability == pytest.approx(1 / 3)



def test_continuous_alpha_sensitivity():
    """Increasing alpha should increase exposure to
    the highest-WER speaker.
    """
    category = "Digital Assistant Commands"

    train_rows = [
        {"speaker_id": "A", "category": category}
        for _ in range(100)
    ] + [
        {"speaker_id": "B", "category": category}
        for _ in range(10)
    ] + [
        {"speaker_id": "C", "category": category}
        for _ in range(50)
    ]

    speaker_wer = {
        "A": 0.05,
        "B": 0.15,
        "C": 0.30,
    }

    hard_speaker_probs = []

    for alpha in [0.0, 1.0, 2.0]:
        probabilities = calculate_probabilities(
            train_rows,
            speaker_wer,
            strategy="continuous",
            alpha=alpha,
            category_probs={category: 1.0},
        )

        # Aggregate probabilities for the highest-WER speaker
        prob_C = sum(
            p for row, p in zip(train_rows, probabilities)
            if row["speaker_id"] == "C"
        )

        hard_speaker_probs.append(prob_C)

        assert sum(probabilities) == pytest.approx(1.0)

    # Alpha=0 gives uniform speaker selection
    assert hard_speaker_probs[0] == pytest.approx(1 / 3)

    # Higher alpha should increase exposure to speaker C
    assert (
        hard_speaker_probs[0]
        < hard_speaker_probs[1]
        < hard_speaker_probs[2]
    )



@pytest.mark.parametrize("strategy", ["continuous", "hard_threshold"])
def test_category_probabilities(strategy):
    """Both sampling strategies must preserve the target
    category distribution.
    """
    category_probs = {
        "Digital Assistant Commands": 0.50,
        "Novel Sentences": 0.30,
        "Spontaneous Speech Prompts": 0.20,
    }

    # Recording counts for each speaker-category pair
    recording_counts = {
        "A": [100, 10, 2],
        "B": [50, 0, 3],
        "C": [20, 30, 0],
        "D": [10, 5, 5],
    }

    categories = list(category_probs)

    train_rows = []

    for speaker, counts in recording_counts.items():
        for category, count in zip(categories, counts):
            train_rows.extend([
                {"speaker_id": speaker, "category": category}
                for _ in range(count)
            ])

    speaker_wer = {
        "A": 0.40,
        "B": 0.25,
        "C": 0.10,
        "D": 0.05,
    }

    hard_labels = {
        "A": True,
        "B": True,
        "C": False,
        "D": False,
    }

    probabilities = calculate_probabilities(
        train_rows,
        speaker_wer,
        strategy=strategy,
        category_probs=category_probs,
        hard_labels=hard_labels,
    )

    # Aggregate recording probabilities by category
    actual_probs = {
        category: 0.0 for category in categories
    }

    for row, probability in zip(train_rows, probabilities):
        actual_probs[row["category"]] += probability

    assert len(probabilities) == len(train_rows)
    assert sum(probabilities) == pytest.approx(1.0)

    # Verify target category probabilities
    for category, expected in category_probs.items():
        assert actual_probs[category] == pytest.approx(expected)



def test_hard_threshold_group_and_speaker_balance():
    """Verify equal group probability and uniform
    speaker selection within each group.
    """
    category = "Digital Assistant Commands"

    recording_counts = {
        "A": 10,
        "B": 20,
        "C": 100,
        "D": 200,
        "E": 300,
    }

    train_rows = [
        {"speaker_id": speaker, "category": category}
        for speaker, count in recording_counts.items()
        for _ in range(count)
    ]

    speaker_wer = {
        "A": 0.60,
        "B": 0.40,
        "C": 0.15,
        "D": 0.10,
        "E": 0.05,
    }

    hard_labels = {
        "A": True,
        "B": True,
        "C": False,
        "D": False,
        "E": False,
    }

    probabilities = calculate_probabilities(
        train_rows,
        speaker_wer,
        strategy="hard_threshold",
        category_probs={category: 1.0},
        hard_labels=hard_labels,
    )

    # Aggregate probabilities by group
    group_probs = {True: 0.0, False: 0.0}

    for row, probability in zip(train_rows, probabilities):
        is_hard = hard_labels[row["speaker_id"]]
        group_probs[is_hard] += probability

    assert sum(probabilities) == pytest.approx(1.0)

    # Both groups must receive equal probability mass
    assert group_probs[True] == pytest.approx(0.50)
    assert group_probs[False] == pytest.approx(0.50)

        # Aggregate probabilities by speaker
    speaker_probs = {
        speaker: 0.0 for speaker in recording_counts
    }

    for row, probability in zip(train_rows, probabilities):
        speaker_probs[row["speaker_id"]] += probability

    # Hard group: two speakers share 50% probability
    assert speaker_probs["A"] == pytest.approx(0.50 / 2)
    assert speaker_probs["B"] == pytest.approx(0.50 / 2)

    # Other group: three speakers share 50% probability
    assert speaker_probs["C"] == pytest.approx(0.50 / 3)
    assert speaker_probs["D"] == pytest.approx(0.50 / 3)
    assert speaker_probs["E"] == pytest.approx(0.50 / 3)




@pytest.mark.parametrize("strategy", ["continuous", "hard_threshold"])
def test_recording_and_joint_probabilities(strategy):
    """Recordings within the same speaker-category pair
    must have equal sampling probabilities.
    """
    category_probs = {
        "Digital Assistant Commands": 0.50,
        "Novel Sentences": 0.30,
        "Spontaneous Speech Prompts": 0.20,
    }

    recording_counts = {
        "A": [100, 10, 2],
        "B": [50, 0, 3],
        "C": [20, 30, 0],
        "D": [10, 5, 5],
    }

    train_rows = [
        {"speaker_id": speaker, "category": category}
        for speaker, counts in recording_counts.items()
        for category, count in zip(category_probs, counts)
        for _ in range(count)
    ]

    speaker_wer = {
        "A": 0.40,
        "B": 0.25,
        "C": 0.10,
        "D": 0.05,
    }

    hard_labels = {
        "A": True,
        "B": True,
        "C": False,
        "D": False,
    }

    probabilities = calculate_probabilities(
        train_rows,
        speaker_wer,
        strategy=strategy,
        category_probs=category_probs,
        hard_labels=hard_labels,
    )

    # Collect recording probabilities by speaker-category pair
    pair_probs = {}

    for row, probability in zip(train_rows, probabilities):
        key = (row["speaker_id"], row["category"])
        pair_probs.setdefault(key, []).append(probability)

    # Check uniformity within each speaker-category pair
    for probs in pair_probs.values():
        expected = sum(probs) / len(probs)

        for probability in probs:
            assert probability == pytest.approx(expected)

        # Verify the total probability of each speaker-category pair
    if strategy == "hard_threshold":
        expected_pair_probs = {
            "Digital Assistant Commands": {
                "A": 0.125,
                "B": 0.125,
                "C": 0.125,
                "D": 0.125,
            },
            "Novel Sentences": {
                "A": 0.15,
                "C": 0.075,
                "D": 0.075,
            },
            "Spontaneous Speech Prompts": {
                "A": 0.05,
                "B": 0.05,
                "D": 0.10,
            },
        }

    else:
        expected_pair_probs = {
            "Digital Assistant Commands": {
                "A": 0.50 * 0.32 / 0.78,
                "B": 0.50 * 0.27 / 0.78,
                "C": 0.50 * 0.12 / 0.78,
                "D": 0.50 * 0.07 / 0.78,
            },
            "Novel Sentences": {
                "A": 0.30 * 0.32 / 0.51,
                "C": 0.30 * 0.12 / 0.51,
                "D": 0.30 * 0.07 / 0.51,
            },
            "Spontaneous Speech Prompts": {
                "A": 0.20 * 0.32 / 0.66,
                "B": 0.20 * 0.27 / 0.66,
                "D": 0.20 * 0.07 / 0.66,
            },
        }

    for (speaker, category), probs in pair_probs.items():
        expected = expected_pair_probs[category][speaker]

        assert sum(probs) == pytest.approx(expected)



@pytest.mark.parametrize(
    "is_hard, expected_error",
    [
        (True, "No other speakers"),
        (False, "No hard speakers"),
    ],
)
def test_hard_threshold_empty_group(is_hard, expected_error):
    """Hard threshold must reject categories that
    do not contain both speaker groups.
    """
    category = "Digital Assistant Commands"

    train_rows = [
        {"speaker_id": "A", "category": category},
        {"speaker_id": "B", "category": category},
    ]

    speaker_wer = {
        "A": 0.20,
        "B": 0.10,
    }

    # Both speakers belong to the same group
    hard_labels = {
        "A": is_hard,
        "B": is_hard,
    }

    with pytest.raises(ValueError, match=expected_error):
        calculate_probabilities(
            train_rows,
            speaker_wer,
            strategy="hard_threshold",
            category_probs={category: 1.0},
            hard_labels=hard_labels,
        )




@pytest.mark.parametrize("strategy", ["continuous", "hard_threshold"])
def test_missing_speaker_wer(strategy):
    """Both strategies must reject missing speaker WER."""
    category = "Digital Assistant Commands"

    train_rows = [
        {"speaker_id": "A", "category": category},
        {"speaker_id": "B", "category": category},
    ]

    # Speaker B is missing from the WER dictionary
    speaker_wer = {
        "A": 0.30,
    }

    hard_labels = {
        "A": True,
        "B": False,
    }

    with pytest.raises(ValueError, match="Missing speaker WER"):
        calculate_probabilities(
            train_rows,
            speaker_wer,
            strategy=strategy,
            category_probs={category: 1.0},
            hard_labels=hard_labels,
        )




@pytest.mark.parametrize("strategy", ["continuous", "hard_threshold"])
def test_sampler_reproducibility(strategy):
    """Verify reproducible sampling and consistent
    sampling settings for both strategies.
    """
    category = "Digital Assistant Commands"

    train_rows = [
        {"speaker_id": "A", "category": category}
        for _ in range(10)
    ] + [
        {"speaker_id": "B", "category": category}
        for _ in range(20)
    ]

    speaker_wer = {
        "A": 0.30,
        "B": 0.10,
    }

    hard_labels = {
        "A": True,
        "B": False,
    }

    # Shared sampling configuration
    kwargs = {
        "strategy": strategy,
        "category_probs": {category: 1.0},
        "hard_labels": hard_labels,
        "num_samples": len(train_rows),
    }

    # Create three independent samplers
    sampler_1 = build_sampler(
        train_rows, speaker_wer, seed=42, **kwargs
    )

    sampler_2 = build_sampler(
        train_rows, speaker_wer, seed=42, **kwargs
    )

    sampler_3 = build_sampler(
        train_rows, speaker_wer, seed=43, **kwargs
    )

    # Generate sampled recording indices
    indices_1 = list(sampler_1)
    indices_2 = list(sampler_2)
    indices_3 = list(sampler_3)

    # Same seed: identical sampling sequence
    assert indices_1 == indices_2

    # Different seed: different sampling sequence
    assert indices_1 != indices_3

    # Same number of samples per epoch
    assert len(indices_1) == len(train_rows)
    assert len(indices_2) == len(train_rows)
    assert len(indices_3) == len(train_rows)

    # Sampling with replacement
    assert sampler_1.replacement is True
    assert sampler_2.replacement is True
    assert sampler_3.replacement is True

    # All sampled indices must be valid
    for indices in [indices_1, indices_2, indices_3]:
        assert all(0 <= i < len(train_rows) for i in indices)



def test_load_speaker_metadata(tmp_path):
    """Both metadata loaders must read the same
    TRAIN difficulty CSV correctly.
    """
    csv_path = tmp_path / "speaker_difficulty.csv"

    csv_path.write_text(
        "speaker_id,difficulty_wer_percent,hard_top25\n"
        "A,60.0,1\n"
        "B,25.0,0\n"
        "C,10.0,0\n"
        "D,5.0,0\n",
        encoding="utf-8",
    )

    speaker_wer = load_speaker_wer_csv(csv_path)
    hard_labels = load_hard_labels_csv(csv_path)

    # WER percentages must be converted to fractions
    assert speaker_wer == pytest.approx({
        "A": 0.60,
        "B": 0.25,
        "C": 0.10,
        "D": 0.05,
    })

    # Hard labels must be converted to booleans
    assert hard_labels == {
        "A": True,
        "B": False,
        "C": False,
        "D": False,
    }

    assert all(
        isinstance(label, bool)
        for label in hard_labels.values()
    )

    # Both loaders must identify the same speakers
    assert set(speaker_wer) == set(hard_labels)




@pytest.mark.parametrize("strategy", ["continuous", "hard_threshold"])
def test_csv_sampler_integration(tmp_path, strategy):
    """Verify the complete metadata-to-sampler workflow."""
    csv_path = tmp_path / "speaker_difficulty.csv"

    csv_path.write_text(
        "speaker_id,difficulty_wer_percent,hard_top25\n"
        "A,60.0,1\n"
        "B,25.0,0\n"
        "C,10.0,0\n"
        "D,5.0,0\n",
        encoding="utf-8",
    )

    category_probs = {
        "Digital Assistant Commands": 0.50,
        "Novel Sentences": 0.30,
        "Spontaneous Speech Prompts": 0.20,
    }

    recording_counts = {
        "A": [10, 5, 2],
        "B": [20, 10, 0],
        "C": [30, 0, 5],
        "D": [40, 15, 10],
    }

    train_rows = [
        {
            "speaker_id": speaker,
            "category": category,
            "split": "train",
        }
        for speaker, counts in recording_counts.items()
        for category, count in zip(category_probs, counts)
        for _ in range(count)
    ]

    # Follow the same metadata-loading logic as train.py
    speaker_wer = load_speaker_wer_csv(csv_path)

    hard_labels = (
        load_hard_labels_csv(csv_path)
        if strategy == "hard_threshold"
        else None
    )

    sampler = build_sampler(
        train_rows,
        speaker_wer,
        strategy=strategy,
        seed=42,
        alpha=1.0,
        epsilon=0.02,
        max_wer=0.30,
        category_probs=category_probs,
        num_samples=len(train_rows),
        hard_labels=hard_labels,
    )

    # Inspect the actual weights stored in the sampler
    probabilities = sampler.weights.tolist()

    assert len(probabilities) == len(train_rows)
    assert sum(probabilities) == pytest.approx(1.0)

    # Verify category probabilities
    for category, expected in category_probs.items():
        actual = sum(
            p for row, p in zip(train_rows, probabilities)
            if row["category"] == category
        )

        assert actual == pytest.approx(expected)

    # Verify global hard-speaker probability
    if strategy == "hard_threshold":
        hard_mass = sum(
            p for row, p in zip(train_rows, probabilities)
            if hard_labels[row["speaker_id"]]
        )

        assert hard_mass == pytest.approx(0.50)

    # Verify that the sampler can generate training indices
    indices = list(sampler)

    assert len(indices) == len(train_rows)
    assert all(0 <= i < len(train_rows) for i in indices)

        # Verify sampling diagnostics
    summary = summarize_sampling(
        train_rows,
        probabilities,
        hard_labels=hard_labels,
    )

    assert summary["total_probability"] == pytest.approx(1.0)
    assert summary["num_speakers"] == 4
    assert summary["num_recordings"] == len(train_rows)

    assert summary["category_probability"] == pytest.approx(
        category_probs
    )

    assert sum(
        summary["speaker_probability"].values()
    ) == pytest.approx(1.0)

    if strategy == "hard_threshold":
        assert summary["hard_probability"] == pytest.approx(0.50)

        for probability in summary["hard_probability_by_category"].values():
            assert probability == pytest.approx(0.50)