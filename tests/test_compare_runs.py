"""Tests for compare_runs.py using tiny synthetic runs. No model, GPU or SAP data."""

import json
import tempfile
import unittest
from pathlib import Path

import compare_runs as cr


def make_row(path, speaker, ref, hyp):
    """A row as written by run_asr_baselines.py to scored_predictions.jsonl."""
    return {"audio_path": path, "speaker_id": speaker, "category": "x", "text": ref,
            "hyp": hyp, "ref_norm": ref, "hyp_norm": hyp}


def write_run(runs_dir, name, rows, split="val"):
    folder = Path(runs_dir) / name / f"eval_{split}" / "qwen"
    folder.mkdir(parents=True)
    with (folder / "scored_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


# Four speakers, 2 utterances each, 4 reference words per utterance.
# Zero-shot WER per speaker: A=0%, B=12.5%, C=25%, D=50%  -> hardest 25% = {D}
REFS = {"A": "one two three four", "B": "one two three four", "C": "one two three four", "D": "one two three four"}


def zero_shot_rows():
    hyp = {"A": ["one two three four"] * 2,
           "B": ["one two three four", "one two three five"],         # 1 error in 8 words
           "C": ["one two three five", "one two three five"],         # 2 errors in 8
           "D": ["one two five six", "one two five six"]}             # 4 errors in 8
    return [make_row(f"{s}{i}.wav", s, REFS[s], hyp[s][i]) for s in REFS for i in range(2)]


def tuned_rows(fixed):
    """Fine-tuned model that gets every speaker in `fixed` perfectly right, others as zero-shot."""
    zs = zero_shot_rows()
    return [dict(r, hyp_norm=r["ref_norm"]) if r["speaker_id"] in fixed else r for r in zs]


class TestScoring(unittest.TestCase):
    def test_edit_counts(self):
        self.assertEqual(cr.edit_counts("a b c".split(), "a b c".split()), (0, 0, 0))
        self.assertEqual(cr.edit_counts("a b c".split(), "a x c".split()), (1, 0, 0))
        self.assertEqual(cr.edit_counts("a b c".split(), "a c".split()), (0, 1, 0))
        self.assertEqual(cr.edit_counts("a c".split(), "a b c".split()), (0, 0, 1))

    def test_corpus_wer_is_errors_over_total_words(self):
        # 1 error in 4 words + 0 errors in 8 words = 1/12 (NOT the mean of 25% and 0%)
        rows = [make_row("a", "s", "a b c d", "a b c x"),
                make_row("b", "s", "a b c d e f g h", "a b c d e f g h")]
        self.assertAlmostEqual(cr.score_rows(rows)["wer"], 100 / 12)

    def test_matches_jiwer_when_available(self):
        if not cr.HAVE_JIWER:
            self.skipTest("jiwer not installed")
        rows = zero_shot_rows()
        refs, hyps = [r["ref_norm"] for r in rows], [r["hyp_norm"] for r in rows]
        self.assertAlmostEqual(cr.score_rows(rows)["wer"], 100 * cr.jiwer.wer(refs, hyps))

    def test_empty(self):
        self.assertIsNone(cr.score_rows([])["wer"])


class TestHardSpeakers(unittest.TestCase):
    def test_picks_hardest_quarter_by_zero_shot_wer(self):
        hard, ranking = cr.pick_hard_speakers(zero_shot_rows(), 0.25)
        self.assertEqual(hard, {"D"})
        self.assertEqual([s for s, _ in ranking], ["D", "C", "B", "A"])

    def test_uses_ceiling_and_breaks_ties_by_id(self):
        # 25 speakers -> ceil(6.25) = 7 hard; with all-equal WER the first 7 ids win.
        rows = [make_row(f"{i}.wav", f"s{i:02d}", "a b", "a c") for i in range(25)]
        hard, _ = cr.pick_hard_speakers(rows, 0.25)
        self.assertEqual(hard, {f"s{i:02d}" for i in range(7)})


class TestCompare(unittest.TestCase):
    def test_end_to_end_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d, "zero_shot", zero_shot_rows())
            write_run(d, "E3_L1", tuned_rows({"D"}))              # fixes only the hard speaker
            write_run(d, "E4_L1", tuned_rows({"A", "B", "C"}))    # fixes only the easy speakers
            result = cr.compare(d, "val")
            by_name = {m["model"]: m for m in result["models"]}

            self.assertEqual(result["hard_speakers"], ["D"])
            self.assertEqual(result["n_common_utterances"], 8)
            # zero-shot: 7 errors / 32 words overall; hard (D) 4/8 = 50%; others 3/24 = 12.5%
            self.assertAlmostEqual(by_name["zero_shot"]["overall"]["wer"], 100 * 7 / 32)
            self.assertAlmostEqual(by_name["zero_shot"]["hard"]["wer"], 50.0)
            self.assertAlmostEqual(by_name["zero_shot"]["other"]["wer"], 12.5)
            # E3_L1 fixes speaker D only
            self.assertAlmostEqual(by_name["E3_L1"]["hard"]["wer"], 0.0)
            self.assertAlmostEqual(by_name["E3_L1"]["other"]["wer"], 12.5)
            self.assertEqual(by_name["E3_L1"]["n_speakers_improved"], 1)
            self.assertEqual(by_name["E3_L1"]["n_hard_speakers_improved"], 1)
            self.assertAlmostEqual(by_name["E3_L1"]["speaker_delta_pp"]["D"], -50.0)
            # E4_L1 fixes the three easy speakers; speaker A was already perfect, so only B and C improve
            self.assertAlmostEqual(by_name["E4_L1"]["hard"]["wer"], 50.0)
            self.assertEqual(by_name["E4_L1"]["n_speakers_improved"], 2)
            self.assertEqual(by_name["E4_L1"]["n_hard_speakers_improved"], 0)
            self.assertAlmostEqual(by_name["E3_L1"]["relative_wer_reduction_pct"], 100 * 4 / 7)

    def test_intersection_drops_unshared_utterances(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d, "zero_shot", zero_shot_rows())
            write_run(d, "E3_L1", tuned_rows({"D"})[:-1])          # one utterance missing
            result = cr.compare(d, "val")
            self.assertEqual(result["n_common_utterances"], 7)
            self.assertEqual(result["n_dropped_by_intersection"]["zero_shot"], 1)

    def test_different_reference_text_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d, "zero_shot", zero_shot_rows())
            changed = tuned_rows({"D"})
            changed[0]["ref_norm"] = "something else entirely"
            write_run(d, "E3_L1", changed)
            with self.assertRaises(SystemExit):
                cr.compare(d, "val")

    def test_outputs_and_gpu_usage(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d, "zero_shot", zero_shot_rows())
            write_run(d, "E3_L1", tuned_rows({"D"}))
            (Path(d) / "E3_L1" / "gpu_usage.json").write_text(json.dumps({"peak_memory_allocated_gb": 9.5}))
            result = cr.compare(d, "val")
            out = cr.write_outputs(result, Path(d) / "out")
            for name in ("comparison_val.json", "comparison_summary_val.csv",
                         "comparison_per_speaker_val.csv", "comparison_val.md"):
                self.assertTrue((out / name).is_file(), name)
            self.assertIn("9.5", (out / "comparison_val.md").read_text())

    def test_test_split_requires_final_flag(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                cr.main(["--runs-dir", d, "--split", "test"])


if __name__ == "__main__":
    unittest.main()
