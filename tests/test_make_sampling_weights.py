"""Synthetic-data tests for make_sampling_weights. No SAP data required."""

import csv
import math
import tempfile
import unittest
from pathlib import Path

import make_sampling_weights as msw


def speakers(n_speakers=8):
    # speaker i has WER 2*(i+1) and (i+1)*10 recordings
    return [{"speaker_id": f"s{i}", "wer": 2.0 * (i + 1), "n": (i + 1) * 10} for i in range(n_speakers)]


class TestComputeColumns(unittest.TestCase):
    def setUp(self):
        self.rows = msw.compute_columns(speakers())
        self.by_id = {r["speaker_id"]: r for r in self.rows}

    def test_every_strategy_sums_to_one(self):
        for k, col in (("e1", "e1_uniform_recording_p_speaker"), ("e2", "e2_speaker_balanced_p_speaker"),
                       ("e3", "e3_hard50_p_speaker"), ("e4", "e4_difficulty_p_speaker")):
            self.assertAlmostEqual(sum(r[col] for r in self.rows), 1.0, msg=k)
            self.assertAlmostEqual(
                sum(r[f"{k}_per_recording_probability"] * r["n_manifest"] for r in self.rows), 1.0, msg=k)

    def test_hard_group_is_top_quarter_by_wer(self):
        hard = {r["speaker_id"] for r in self.rows if r["hard_top25"]}
        self.assertEqual(hard, {"s7", "s6"})              # ceil(0.25 * 8) = 2 hardest
        self.assertEqual(self.by_id["s7"]["difficulty_rank"], 1)

    def test_e3_gives_hard_group_half(self):
        mass = sum(r["e3_hard50_p_speaker"] for r in self.rows if r["hard_top25"])
        self.assertAlmostEqual(mass, 0.5)

    def test_e1_is_natural_and_e2_is_speaker_uniform(self):
        total = sum(s["n"] for s in speakers())
        self.assertAlmostEqual(self.by_id["s3"]["e1_per_recording_probability"], 1 / total)
        self.assertAlmostEqual(self.by_id["s3"]["e2_speaker_balanced_p_speaker"], 1 / 8)

    def test_e4_is_clipped_and_monotone_in_wer(self):
        ps = [r["e4_difficulty_p_speaker"] for r in self.rows]    # rows sorted hardest first
        self.assertEqual(ps, sorted(ps, reverse=True))
        # WER 16 and 14 are above the cap of 12 here -> equal weight
        capped = msw.compute_columns(speakers(), e4_cap=12.0)
        by = {r["speaker_id"]: r for r in capped}
        self.assertAlmostEqual(by["s7"]["e4_difficulty_p_speaker"], by["s5"]["e4_difficulty_p_speaker"])
        # default floor = median WER (9): easiest speakers share the floor weight
        self.assertAlmostEqual(self.by_id["s0"]["e4_difficulty_p_speaker"],
                               self.by_id["s3"]["e4_difficulty_p_speaker"])

    def test_ties_are_deterministic(self):
        tied = [{"speaker_id": k, "wer": 5.0, "n": 10} for k in "dcba"]
        ranks = {r["speaker_id"]: r["difficulty_rank"] for r in msw.compute_columns(tied)}
        self.assertEqual(ranks, {"a": 1, "b": 2, "c": 3, "d": 4})

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            msw.compute_columns([{"speaker_id": "a", "wer": 1.0, "n": 0}, {"speaker_id": "b", "wer": 1.0, "n": 1}])
        with self.assertRaises(ValueError):
            msw.compute_columns([{"speaker_id": "a", "wer": 1.0, "n": 1}] * 2)


class TestCli(unittest.TestCase):
    def test_roundtrip_and_verify(self):
        with tempfile.TemporaryDirectory() as d:
            src, out, out2 = Path(d) / "in.csv", Path(d) / "out.csv", Path(d) / "out2.csv"
            with src.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["speaker_id", "difficulty_wer_percent", "n_manifest"])
                for s in speakers():
                    w.writerow([s["speaker_id"], s["wer"], s["n"]])
            msw.main(["--in", str(src), "--out", str(out)])
            # the output is itself a valid input, and verifying it against itself must pass
            msw.main(["--in", str(out), "--out", str(out2), "--verify"])
            with out.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(list(rows[0]), msw.OUT_COLUMNS)
            self.assertTrue(math.isclose(sum(float(r["e4_difficulty_p_speaker"]) for r in rows), 1.0, rel_tol=1e-9))


if __name__ == "__main__":
    unittest.main()
