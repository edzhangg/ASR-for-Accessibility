"""Unit tests using synthetic data. No SAP data or GPU required."""

import unittest

import csv
import tempfile
from pathlib import Path

from hard_speaker_sampler import (
    build_sampling_weights,
    load_speaker_weights,
    make_sampler,
)

HEADER = ["speaker_id", "hard_top25", "n_manifest", "e3_hard50_p_speaker",
          "e3_per_recording_probability", "e4_difficulty_p_speaker", "e4_per_recording_probability"]


def write_csv(rows):
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
    w = csv.writer(f)
    w.writerow(HEADER)
    w.writerows(rows)
    f.close()
    return Path(f.name)


class TestHardSpeakerSampler(unittest.TestCase):

    def setUp(self):
        # Two recordings from a hard speaker,
        # three recordings from an other speaker.
        self.rows = (
            [{"speaker_id": "hard", "split": "train"}] * 2
            + [{"speaker_id": "other", "split": "train"}] * 3
        )

        self.info = {
            "hard": {
                "hard": True,
                "n": 2,
                "p_speaker": 0.5,
                "p_recording": 0.25,
            },
            "other": {
                "hard": False,
                "n": 3,
                "p_speaker": 0.5,
                "p_recording": 1 / 6,
            },
        }

    def test_probability_mass(self):
        weights = build_sampling_weights(self.rows, self.info)

        self.assertAlmostEqual(sum(weights[:2]), 0.5)
        self.assertAlmostEqual(sum(weights[2:]), 0.5)
        self.assertAlmostEqual(sum(weights), 1.0)

    def test_manifest_with_more_rows_than_csv_is_rejected(self):
        self.info["hard"]["n"] = 1

        with self.assertRaisesRegex(ValueError, "exceeds CSV"):
            build_sampling_weights(self.rows, self.info)

    def test_filtered_manifest_keeps_speaker_mass(self):
        # CSV was built with 4 and 6 recordings; the filtered manifest has 2 and 3.
        self.info["hard"].update(n=4, p_recording=0.125)
        self.info["other"].update(n=6, p_recording=1 / 12)
        weights = build_sampling_weights(self.rows, self.info)

        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertAlmostEqual(sum(weights[:2]), 0.5)
        self.assertAlmostEqual(weights[0], 0.25)

    def test_csv_loads_e3_and_e4(self):
        path = write_csv([["a", "1", "2", "0.5", "0.25", "0.4", "0.2"],
                          ["b", "0", "3", "0.5", str(0.5 / 3), "0.6", "0.2"]])
        self.assertAlmostEqual(load_speaker_weights(path, "e3")["a"]["p_recording"], 0.25)
        self.assertAlmostEqual(load_speaker_weights(path, "e4")["b"]["p_speaker"], 0.6)

    def test_csv_rejects_inconsistent_and_duplicate(self):
        bad = write_csv([["a", "1", "2", "0.5", "0.4", "0.4", "0.2"]])
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            load_speaker_weights(bad, "e3")
        dup = write_csv([["a", "1", "2", "0.5", "0.25", "0.4", "0.2"]] * 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_speaker_weights(dup, "e3")

    def test_non_e3_skips_hard_mass_check(self):
        rows = [{"speaker_id": "a"}] * 2 + [{"speaker_id": "b"}] * 2
        info = {"a": {"hard": True, "n": 2, "p_speaker": 0.7, "p_recording": 0.35},
                "b": {"hard": False, "n": 2, "p_speaker": 0.3, "p_recording": 0.15}}
        with self.assertRaisesRegex(ValueError, "Hard-speaker"):
            build_sampling_weights(rows, info)
        self.assertAlmostEqual(sum(build_sampling_weights(rows, info, check_hard_mass=False)), 1.0)

    def test_sampled_share(self):
        sampler = make_sampler(
            self.rows,
            self.info,
            num_samples=10000,
            seed=42,
        )

        hard_count = sum(
            self.rows[i]["speaker_id"] == "hard"
            for i in sampler
        )

        observed = hard_count / 10000

        self.assertLess(abs(observed - 0.5), 0.025)


if __name__ == "__main__":
    unittest.main()
