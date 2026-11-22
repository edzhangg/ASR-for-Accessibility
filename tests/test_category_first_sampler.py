"""Run: python3 -m unittest test_category_first_sampler.py"""
import unittest
from category_first_sampler import build_category_first_weights, load_category_config


class CategoryFirstTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"speaker_id": "h1", "category": "A"},
            {"speaker_id": "h1", "category": "A"},
            {"speaker_id": "e1", "category": "A"},
            {"speaker_id": "h2", "category": "B"},
            {"speaker_id": "e2", "category": "B"},
            {"speaker_id": "h3", "category": "C"},
            {"speaker_id": "e3", "category": "C"},
            {"speaker_id": "h3", "category": "excluded"},
        ]
        self.info = {s: {"hard": s.startswith("h")}
                     for s in ("h1", "h2", "h3", "e1", "e2", "e3")}
        self.config = {"category_proportions": {"A": .5, "B": .3, "C": .2},
                       "hard_group_probability": .5}

    def test_marginals(self):
        weights = build_category_first_weights(self.rows, self.info, self.config)
        self.assertAlmostEqual(sum(weights), 1)
        self.assertEqual(weights[-1], 0)
        for category, target in self.config["category_proportions"].items():
            self.assertAlmostEqual(sum(w for row, w in zip(self.rows, weights)
                                       if row["category"] == category), target)
        self.assertAlmostEqual(sum(w for row, w in zip(self.rows, weights)
                                   if self.info[row["speaker_id"]]["hard"]), .5)
        self.assertAlmostEqual(weights[0], .125)
        self.assertAlmostEqual(weights[1], .125)
        self.assertAlmostEqual(weights[2], .25)

    def test_missing_group(self):
        rows = [r for r in self.rows if r["speaker_id"] != "e3"]
        with self.assertRaisesRegex(ValueError, "no non-hard speakers"):
            build_category_first_weights(rows, self.info, self.config)

    def test_repository_config_and_fixed_targets(self):
        config = load_category_config("configs/category_sampling.json")
        proportions = config["category_proportions"]
        self.assertEqual(
            proportions,
            {
                "Digital Assistant Commands": 0.5,
                "Novel Sentences": 0.3,
                "Spontaneous Speech Prompts": 0.2,
            },
        )
        self.assertEqual(config["hard_group_probability"], 0.5)

    def test_hard_and_nonhard_masses_within_each_category(self):
        weights = build_category_first_weights(self.rows, self.info, self.config)
        for category, target in self.config["category_proportions"].items():
            hard_mass = sum(
                weight
                for row, weight in zip(self.rows, weights)
                if row["category"] == category and self.info[row["speaker_id"]]["hard"]
            )
            self.assertAlmostEqual(hard_mass, target * 0.5)
            self.assertAlmostEqual(target - hard_mass, target * 0.5)


if __name__ == "__main__":
    unittest.main()
