"""Tests for run_experiment.py. Only builds/prints commands; never launches training."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import run_experiment as rx

CONFIG = rx.load_config(rx.DEFAULT_CONFIG)


def option(cmd, flag):
    """Value following `flag` in a command list."""
    return cmd[cmd.index(flag) + 1]


class TestConfig(unittest.TestCase):
    def test_shared_config_loads_and_has_all_sections(self):
        for section in ("data", "model", "training", "eval"):
            self.assertIn(section, CONFIG)

    def test_override_changes_copy_only(self):
        changed = rx.apply_overrides(CONFIG, ["training.learning_rate=5e-5", "eval.max_duration=120"])
        self.assertEqual(changed["training"]["learning_rate"], 5e-5)
        self.assertEqual(changed["eval"]["max_duration"], 120)
        self.assertEqual(CONFIG["training"]["learning_rate"], 0.0001)      # original untouched

    def test_override_rejects_typos_and_bad_format(self):
        for bad in ("training.learning_rat=1", "nosection=1", "training=1", "_notes.x=1"):
            with self.assertRaises(ValueError, msg=bad):
                rx.apply_overrides(CONFIG, [bad])


class TestCommands(unittest.TestCase):
    runs = Path("/tmp/runs")

    def test_name(self):
        self.assertEqual(rx.experiment_name("e3", "L1"), "E3_L1")
        self.assertEqual(rx.experiment_name("none", "L2"), "NONE_L2")

    def test_train_cmd_only_differs_by_sampling_and_lora(self):
        a = rx.build_train_cmd(CONFIG, "e3", "L1", self.runs)
        b = rx.build_train_cmd(CONFIG, "e4", "L2", self.runs)
        differing = {i for i, (x, y) in enumerate(zip(a, b)) if x != y}
        flags = {a[i - 1] for i in differing if i > 0}
        # output dir, lora config and sampling mode are the ONLY differences
        self.assertEqual(flags, {"--output-dir", "--lora-config", "--sampling"})
        self.assertEqual(len(a), len(b))

    def test_weights_csv_only_for_weighted_sampling(self):
        self.assertIn("--weights-csv", rx.build_train_cmd(CONFIG, "e4", "L1", self.runs))
        self.assertNotIn("--weights-csv", rx.build_train_cmd(CONFIG, "none", "L1", self.runs))

    def test_eval_cmd_identical_for_zero_shot_and_adapter_except_adapter_and_out(self):
        zs = rx.build_eval_cmd(CONFIG, "val", self.runs, rx.ZERO_SHOT_NAME)
        ft = rx.build_eval_cmd(CONFIG, "val", self.runs, "E3_L1", self.runs / "E3_L1" / "adapter-final")
        self.assertNotIn("--qwen-adapter", zs)
        self.assertEqual(option(ft, "--qwen-adapter"), str(self.runs / "E3_L1" / "adapter-final"))
        for flag in ("--manifest", "--qwen-backend", "--qwen-model", "--max-utts",
                     "--max-duration", "--max-new-tokens", "--seed"):
            self.assertEqual(option(zs, flag), option(ft, flag), flag)
        self.assertTrue(option(ft, "--out").endswith("E3_L1/eval_val"))

    def test_compare_cmd(self):
        cmd = rx.build_compare_cmd("test", self.runs, ["E3_L1"], final=True)
        self.assertIn("--final", cmd)
        self.assertEqual(cmd[cmd.index("--experiments") + 1], "E3_L1")


class TestFlagsAreAccepted(unittest.TestCase):
    """The commands the runner builds must be accepted by the scripts' real argument parsers."""

    runs = Path("/tmp/runs")

    def test_train_flags_parse(self):
        import train_qwen_lora as trainer
        for sampling in rx.SAMPLINGS:
            cmd = rx.build_train_cmd(CONFIG, sampling, "L2", self.runs, resume_latest=True)
            args = trainer.parse_args(cmd[2:])          # drop interpreter and script path
            self.assertEqual(args.sampling, sampling)
            self.assertEqual(args.lora_config, "L2")
            self.assertEqual(args.learning_rate, CONFIG["training"]["learning_rate"])

    def test_eval_flags_parse(self):
        import subprocess, sys
        help_text = subprocess.run([sys.executable, str(rx.REPO / "run_asr_baselines.py"), "--help"],
                                   capture_output=True, text=True).stdout
        cmd = rx.build_eval_cmd(CONFIG, "val", self.runs, "E3_L1", self.runs / "E3_L1" / "adapter-final")
        for flag in [c for c in cmd if c.startswith("--")]:
            self.assertIn(flag, help_text, flag)

    def test_compare_flags_parse(self):
        import compare_runs
        cmd = rx.build_compare_cmd("val", self.runs, ["E3_L1"], final=False)
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()):
            # parsing succeeds; the run itself then exits because the folder has no experiments
            with self.assertRaises(SystemExit) as ctx:
                compare_runs.main(["--runs-dir", d] + cmd[cmd.index("--split"):])
        self.assertNotEqual(ctx.exception.code, 2)      # 2 would mean an argparse error


class TestCli(unittest.TestCase):
    def run_main(self, *argv):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(out):
            code = rx.main([*argv, "--dry-run", "--runs-dir", d])
        return code, out.getvalue()

    def test_grid_dry_run_lists_four_trainings_and_evals(self):
        code, text = self.run_main("--grid", "--stages", "zero-shot", "train", "eval", "compare")
        self.assertEqual(code, 0)
        self.assertEqual(text.count("train_qwen_lora.py"), 4)
        self.assertEqual(text.count("run_asr_baselines.py"), 5)           # 4 adapters + zero-shot
        self.assertEqual(text.count("compare_runs.py"), 1)
        for name in ("E3_L1", "E3_L2", "E4_L1", "E4_L2"):
            self.assertIn(name, text)

    def test_single_experiment(self):
        code, text = self.run_main("--sampling", "e4", "--lora", "L2")
        self.assertEqual(text.count("train_qwen_lora.py"), 1)
        self.assertIn("E4_L2", text)

    def test_rejects_bad_launches(self):
        for argv in (["--stages", "train"],                              # no sampling/lora
                     ["--grid", "--sampling", "e3"],                      # grid + explicit
                     ["--sampling", "e3", "--lora", "L1", "--split", "test", "--stages", "eval"],  # test without --final
                     ["--sampling", "e3", "--lora", "L1", "--split", "test", "--final"]):  # train + test together
            with self.assertRaises(SystemExit, msg=argv), contextlib.redirect_stderr(io.StringIO()):
                rx.parse_args(argv)

    def test_existing_adapter_skips_training(self):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(out):
            (Path(d) / "E3_L1" / "adapter-final").mkdir(parents=True)
            rx.main(["--sampling", "e3", "--lora", "L1", "--stages", "train", "--dry-run", "--runs-dir", d])
        self.assertIn("skipping training", out.getvalue())
        self.assertNotIn("train_qwen_lora.py", out.getvalue())


if __name__ == "__main__":
    unittest.main()
