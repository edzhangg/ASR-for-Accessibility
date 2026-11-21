#!/usr/bin/env python3
"""Master runner: launch any experiment (or the whole 2x2 grid) with one command.

Everything that must be identical across experiments lives in ONE file,
``configs/shared.json``. The only things you choose when launching are the two
experimental factors:

  --sampling  {none,e1,e2,e3,e4}   how training recordings are drawn
                 e1 uniform over recordings | e2 uniform over speakers
                 e3 hard-speaker 50/50      | e4 difficulty-weighted
  --lora      {L1,L2}              L1 decoder-only LoRA | L2 audio + decoder LoRA

Stages (``--stages``, default ``train eval``)
  zero-shot  score the un-tuned model on the evaluation split (the comparison baseline)
  train      fine-tune with train_qwen_lora.py          -> runs/<name>/adapter-final
  eval       score the trained adapter with run_asr_baselines.py -> runs/<name>/eval_<split>
  compare    build the final tables with compare_runs.py -> runs/comparison_<split>

Every stage is a separate child process, so each experiment starts from the original
pretrained weights and a crash in one run cannot affect another.

Examples
  # one experiment of the grid (train, then evaluate on validation)
  python run_experiment.py --sampling e4 --lora L1

  # the whole project grid: E3/E4 x L1/L2, with the zero-shot baseline and the final table
  python run_experiment.py --grid --stages zero-shot train eval compare

  # see exactly what would run, without running anything
  python run_experiment.py --grid --dry-run

  # change one shared hyper-parameter for this launch
  python run_experiment.py --sampling e3 --lora L2 --set training.learning_rate=5e-5

Output layout (identical for every run)
  runs/<name>/                      <name> = E3_L1, E4_L2, NONE_L1, ...
      adapter-final/  checkpoint-*/  LoRA weights (base model is never saved)
      training_config.json           arguments, GPU, package versions, LoRA info
      gpu_usage.json  log_history.json
      config_used.json  run_meta.json  logs/<stage>.log
      eval_<split>/qwen/             predictions, scored_predictions.jsonl, summary.json ...
  runs/zero_shot/eval_<split>/qwen/  the baseline
  runs/comparison_<split>/           tables from compare_runs.py

The test split is reserved for the final evaluation: --split test needs --final.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO / "configs" / "shared.json"
SAMPLINGS = ("none", "e1", "e2", "e3", "e4")
LORAS = ("L1", "L2")
STAGES = ("zero-shot", "train", "eval", "compare")
# The project's 2x2: hard-speaker (E3) and difficulty-weighted (E4) sampling, each with L1 and L2.
GRID_SAMPLINGS = ("e3", "e4")
ZERO_SHOT_NAME = "zero_shot"


# ----------------------------------------------------------------------------- config
def load_config(path):
    """Read the shared JSON config."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_overrides(config, overrides):
    """Apply ``section.key=value`` overrides (value parsed as JSON, else kept as a string).

    Returns a modified COPY; unknown keys are rejected so a typo cannot silently
    leave a hyper-parameter unchanged.
    """
    config = deepcopy(config)
    for item in overrides or []:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"--set expects section.key=value, got '{item}'")
        dotted, raw = item.split("=", 1)
        section, key = dotted.split(".", 1)
        if section not in config or section.startswith("_") or key not in config[section]:
            raise ValueError(f"--set: unknown setting '{dotted}'")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        config[section][key] = value
    return config


def experiment_name(sampling, lora):
    """Folder name for one experiment, e.g. ('e3', 'L1') -> 'E3_L1'."""
    return f"{sampling.upper()}_{lora}"


def abs_path(path):
    """Resolve a config path against the repo root (absolute paths pass through)."""
    path = Path(path)
    return path if path.is_absolute() else REPO / path


# ----------------------------------------------------------------------------- commands
# These are pure functions (no side effects) so they can be unit-tested and so
# --dry-run can print exactly what a real launch would execute.
def build_train_cmd(config, sampling, lora, runs_dir, resume_latest=False):
    """Command line for train_qwen_lora.py."""
    t, d = config["training"], config["data"]
    cmd = [
        sys.executable, str(REPO / "train_qwen_lora.py"),
        "--model-path", config["model"]["base"],
        "--train-file", str(abs_path(d["train_file"])),
        "--output-dir", str(runs_dir / experiment_name(sampling, lora)),
        "--lora-config", lora,
        "--sampling", sampling,
        "--seed", str(t["seed"]),
        "--epochs", str(t["epochs"]),
        "--max-steps", str(t["max_steps"]),
        "--batch-size", str(t["batch_size"]),
        "--grad-acc", str(t["grad_acc"]),
        "--learning-rate", str(t["learning_rate"]),
        "--warmup-ratio", str(t["warmup_ratio"]),
        "--log-steps", str(t["log_steps"]),
        "--save-steps", str(t["save_steps"]),
        "--save-total-limit", str(t["save_total_limit"]),
        "--load-in-4bit", str(t["load_in_4bit"]),
        "--gradient-checkpointing", str(t["gradient_checkpointing"]),
        "--optimizer", t["optimizer"],
        "--num-workers", str(t["num_workers"]),
    ]
    if sampling != "none":                      # weighted sampling needs the per-speaker probabilities
        cmd += ["--weights-csv", str(abs_path(d["weights_csv"]))]
    if resume_latest:
        cmd.append("--resume-latest")
    return cmd


def manifest_for(config, split):
    """Evaluation manifest path for 'val' or 'test'."""
    return abs_path(config["data"][f"{split}_manifest"])


def build_eval_cmd(config, split, runs_dir, run_name, adapter=None):
    """Command line for run_asr_baselines.py (adapter=None -> zero-shot baseline).

    Identical flags for every run: that is what keeps validation "exactly as the
    baseline did it".
    """
    e = config["eval"]
    cmd = [
        sys.executable, str(REPO / "run_asr_baselines.py"),
        "--manifest", str(manifest_for(config, split)),
        "--models", "qwen",
        "--qwen-backend", e["backend"],
        "--qwen-model", config["model"]["base"],
        "--max-utts", str(e["max_utts"]),
        "--max-duration", str(e["max_duration"]),
        "--max-new-tokens", str(e["max_new_tokens"]),
        "--seed", str(e["seed"]),
        "--out", str(runs_dir / run_name / f"eval_{split}"),
    ]
    if adapter is not None:
        cmd += ["--qwen-adapter", str(adapter)]
    return cmd


def build_compare_cmd(split, runs_dir, experiments, final):
    """Command line for compare_runs.py."""
    cmd = [sys.executable, str(REPO / "compare_runs.py"), "--runs-dir", str(runs_dir), "--split", split]
    if experiments:
        cmd += ["--experiments", *experiments]
    if final:
        cmd.append("--final")
    return cmd


# ----------------------------------------------------------------------------- execution
def git_commit():
    """Current git commit hash (recorded in run_meta.json), or None outside a repo."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_command(cmd, log_path, dry_run):
    """Run a command, echoing its output to the console AND to a log file. Returns the exit code."""
    print("\n$ " + shlex.join(cmd), flush=True)
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")   # live progress, UTF-8 child output
    # Decode the child's output explicitly as UTF-8 (not the Windows console code page) and never
    # let an odd character crash the runner mid-training: undecodable bytes become '?'.
    with log_path.open("a", encoding="utf-8") as log, subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
            encoding="utf-8", errors="replace") as proc:
        for line in proc.stdout:
            try:
                sys.stdout.write(line)
            except UnicodeEncodeError:                       # console cannot show this character
                sys.stdout.write(line.encode("ascii", "replace").decode("ascii"))
            log.write(line)
        return proc.wait()


def update_meta(run_dir, dry_run, **fields):
    """Merge `fields` into runs/<name>/run_meta.json (stage timings, commands, git commit...)."""
    if dry_run:
        return
    path = run_dir / "run_meta.json"
    meta = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"stages": {}}
    stages = fields.pop("stages", {})
    meta["stages"].update(stages)
    meta.update(fields)
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run_stage(stage, cmd, run_dir, dry_run):
    """Run one stage as a child process, recording its command, duration and exit code."""
    started = time.time()
    code = run_command(cmd, run_dir / "logs" / f"{stage}.log", dry_run)
    update_meta(run_dir, dry_run, stages={stage: {
        "command": cmd, "exit_code": code, "seconds": round(time.time() - started, 1)}})
    return code


def check_files(paths, dry_run):
    """Fail early (before a long GPU job) if an input file is missing; only warn on --dry-run."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing and dry_run:
        print("WARNING (dry run): missing input(s): " + ", ".join(missing))
    elif missing:
        raise SystemExit("Missing input file(s) - fix before launching:\n  " + "\n  ".join(missing))


def run_experiment(args, config, sampling, lora):
    """Run the train and/or eval stages for one (sampling, lora) experiment. Returns exit code."""
    name = experiment_name(sampling, lora)
    run_dir = args.runs_dir / name
    update_meta(run_dir, args.dry_run, name=name, sampling=sampling, lora=lora, git_commit=git_commit())
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        # Save the exact shared config used, so runs can be diffed for comparability.
        (run_dir / "config_used.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    adapter = run_dir / args.checkpoint
    if "train" in args.stages:
        if (run_dir / "adapter-final").is_dir() and not args.force and not args.resume_latest:
            print(f"[{name}] adapter-final already exists; skipping training (use --force to retrain).")
        else:
            check_files([abs_path(config["data"]["train_file"])]
                        + ([abs_path(config["data"]["weights_csv"])] if sampling != "none" else []),
                        args.dry_run)
            code = run_stage("train", build_train_cmd(config, sampling, lora, args.runs_dir, args.resume_latest),
                             run_dir, args.dry_run)
            if code:
                print(f"[{name}] TRAINING FAILED (exit {code}); see {run_dir / 'logs' / 'train.log'}")
                return code
    if "eval" in args.stages:
        check_files([manifest_for(config, args.split)] + ([adapter] if "train" not in args.stages else []),
                    args.dry_run)
        # run_asr_baselines.py resumes from cached predictions, so re-running eval is cheap and safe.
        code = run_stage(f"eval_{args.split}", build_eval_cmd(config, args.split, args.runs_dir, name, adapter),
                         run_dir, args.dry_run)
        if code:
            print(f"[{name}] EVALUATION FAILED (exit {code})")
            return code
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sampling", nargs="+", choices=SAMPLINGS, help="sampling strategy (one or more)")
    ap.add_argument("--lora", nargs="+", choices=LORAS, help="LoRA config (one or more)")
    ap.add_argument("--grid", action="store_true",
                    help="shorthand for --sampling e3 e4 --lora L1 L2 (the project's 2x2)")
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=["train", "eval"])
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--final", action="store_true", help="required for --split test (final evaluation only)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="shared config (default configs/shared.json)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a shared setting for this launch, e.g. training.learning_rate=5e-5")
    ap.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    ap.add_argument("--checkpoint", default="adapter-final",
                    help="adapter folder inside the run to evaluate (e.g. checkpoint-600)")
    ap.add_argument("--resume-latest", action="store_true", help="resume training from the newest checkpoint")
    ap.add_argument("--force", action="store_true", help="retrain even if adapter-final exists")
    ap.add_argument("--keep-going", action="store_true", help="in a grid, continue after a failed experiment")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = ap.parse_args(argv)

    if args.grid:
        if args.sampling or args.lora:
            ap.error("--grid already sets --sampling and --lora")
        args.sampling, args.lora = list(GRID_SAMPLINGS), list(LORAS)
    wants_experiment = any(s in args.stages for s in ("train", "eval"))
    if wants_experiment and not (args.sampling and args.lora):
        ap.error("train/eval stages need --sampling and --lora (or --grid)")
    if args.split == "test" and not args.final:
        ap.error("--split test requires --final (the test set is only for the final evaluation)")
    if args.split == "test" and "train" in args.stages:
        ap.error("do not train and score the test set in one launch; train first, then run --stages eval --split test --final")
    return args


def main(argv=None):
    args = parse_args(argv)
    config = apply_overrides(load_config(args.config), args.overrides)
    print(f"shared config: {args.config}" + (f"  overrides: {args.overrides}" if args.overrides else ""))

    experiments, failures = [], []

    # 1) Zero-shot baseline: same evaluator and settings, no adapter.
    if "zero-shot" in args.stages:
        check_files([manifest_for(config, args.split)], args.dry_run)
        zs_dir = args.runs_dir / ZERO_SHOT_NAME
        update_meta(zs_dir, args.dry_run, name=ZERO_SHOT_NAME, git_commit=git_commit())
        code = run_stage(f"eval_{args.split}", build_eval_cmd(config, args.split, args.runs_dir, ZERO_SHOT_NAME),
                         zs_dir, args.dry_run)
        if code:
            raise SystemExit(f"zero-shot evaluation failed (exit {code}); later stages need it")

    # 2) Every (sampling, lora) combination, one child process per stage.
    if args.sampling and args.lora and any(s in args.stages for s in ("train", "eval")):
        for sampling in args.sampling:
            for lora in args.lora:
                experiments.append(experiment_name(sampling, lora))
                if run_experiment(args, config, sampling, lora):
                    failures.append(experiment_name(sampling, lora))
                    if not args.keep_going:
                        raise SystemExit(f"stopping: {failures[-1]} failed (use --keep-going to continue a grid)")

    # 3) Final comparison table over everything that finished.
    if "compare" in args.stages:
        ok = [e for e in experiments if e not in failures] or None     # None -> auto-discover
        code = run_command(build_compare_cmd(args.split, args.runs_dir, ok, args.final),
                           args.runs_dir / "logs" / "compare.log", args.dry_run)
        if code:
            failures.append("compare")

    print("\nDone." if not failures else f"\nFinished with failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
