"""Unit tests using synthetic data. No SAP data or GPU required."""

import unittest

from hard_speaker_sampler import (
    build_sampling_weights,
    make_sampler,
)


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

    def test_count_mismatch(self):
        self.info["hard"]["n"] = 5

        with self.assertRaisesRegex(ValueError, "count mismatch"):
            build_sampling_weights(self.rows, self.info)

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
