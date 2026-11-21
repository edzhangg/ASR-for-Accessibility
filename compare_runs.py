#!/usr/bin/env python3
"""Compare the fine-tuned runs (and the zero-shot baseline) on the same utterances.

This script NEVER runs a model and NEVER changes how validation is scored. It only
reads the ``scored_predictions.jsonl`` files that ``run_asr_baselines.py`` already
wrote (normalized reference + hypothesis for every utterance, with speaker IDs) and
re-aggregates them. Evaluation itself therefore stays exactly as the baseline did it.

What it adds on top of the per-run ``summary.json`` files
---------------------------------------------------------
* One table with every run side by side (overall WER/CER, S/D/I error counts).
* **Hard-speaker vs other-speaker WER.** "Hard" speakers are the hardest 25% of the
  evaluation speakers *by zero-shot WER* (same rule used to pick the hard training
  speakers: ``ceil(0.25 * n_speakers)``, ties broken by speaker_id). The split comes
  from the zero-shot run, so the SAME speakers are "hard" for every model.
* **Per-speaker improvement** over zero-shot (percentage points), how many speakers
  improved, and the median change.
* Peak GPU memory of each training run (read from ``gpu_usage.json`` if present).

Fairness guarantee
------------------
All runs are compared on the INTERSECTION of the utterances every run scored
(a run could have skipped a clip on an error). The number of utterances dropped by
this intersection is printed, so partial-set WERs are never silently compared.

Expected directory layout (written by ``run_experiment.py``)::

    runs/
      zero_shot/eval_val/qwen/scored_predictions.jsonl     <- zero-shot baseline
      E3_L1/eval_val/qwen/scored_predictions.jsonl         <- one folder per experiment
      E3_L1/gpu_usage.json                                    (optional)
      ...

Usage::

    python compare_runs.py --runs-dir runs --split val
    python compare_runs.py --runs-dir runs --split test --final   # test set: final only

Output files go to ``<runs-dir>/comparison_<split>/``. They contain speaker IDs and
must stay out of git (``runs/`` is already in ``.gitignore``).
"""

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Experiment folders are named like E3_L1 / E4_L2 (sampling strategy + LoRA config);
# NONE_L1 is the no-weighting control (the Trainer's normal shuffled epochs).
EXPERIMENT_DIR_RE = re.compile(r"^(E\d+|NONE)_L\d+$")
# Name of the zero-shot baseline folder inside --runs-dir.
ZERO_SHOT_NAME = "zero_shot"
# Fraction of evaluation speakers labelled "hard" (same as the training E3 rule).
HARD_FRACTION = 0.25


# --------------------------------------------------------------------------- scoring
# The baseline (run_asr_baselines.py) scores with `jiwer`. We use jiwer here too when
# it is installed so the numbers are identical. A small pure-python fallback exists so
# the unit tests (and machines without jiwer) still work; it gives the same total
# error count, and therefore the same WER/CER.
try:  # pragma: no cover - depends on the environment
    import jiwer

    HAVE_JIWER = True
except ImportError:  # pragma: no cover
    jiwer = None
    HAVE_JIWER = False


def edit_counts(ref, hyp):
    """Word/char-level alignment of two token lists -> (substitutions, deletions, insertions).

    Classic Levenshtein dynamic program followed by a back-trace that counts which
    kind of edit each step was. Used only when jiwer is unavailable.
    """
    n, m = len(ref), len(hyp)
    # dist[i][j] = minimum edits to turn ref[:i] into hyp[:j]
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dist[i][0] = i                      # i deletions
    for j in range(1, m + 1):
        dist[0][j] = j                      # j insertions
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dist[i][j] = min(dist[i - 1][j - 1] + sub_cost,   # match / substitution
                             dist[i - 1][j] + 1,              # deletion
                             dist[i][j - 1] + 1)              # insertion
    # Back-trace from the bottom-right corner to count each edit type.
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dist[i][j] == dist[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            subs += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return subs, dels, ins


def score_rows(rows):
    """Corpus-level WER/CER (percent) and S/D/I counts for rows with ref_norm/hyp_norm.

    Corpus level = total errors / total reference words (not an average of
    per-utterance WERs), exactly like the baseline.
    """
    if not rows:
        return {"n": 0, "ref_words": 0, "wer": None, "cer": None,
                "substitutions": None, "deletions": None, "insertions": None}
    refs = [r["ref_norm"] for r in rows]
    hyps = [r["hyp_norm"] for r in rows]
    if HAVE_JIWER:
        w = jiwer.process_words(refs, hyps)
        subs, dels, ins = w.substitutions, w.deletions, w.insertions
        ref_words = w.hits + subs + dels
        cer = 100 * jiwer.cer(refs, hyps)
    else:
        subs = dels = ins = ref_words = 0
        char_errors = char_total = 0
        for ref, hyp in zip(refs, hyps):
            s, d, i = edit_counts(ref.split(), hyp.split())
            subs, dels, ins, ref_words = subs + s, dels + d, ins + i, ref_words + len(ref.split())
            cs, cd, ci = edit_counts(list(ref), list(hyp))   # characters incl. spaces, like jiwer.cer
            char_errors, char_total = char_errors + cs + cd + ci, char_total + len(ref)
        cer = 100 * char_errors / char_total if char_total else None
    wer = 100 * (subs + dels + ins) / ref_words if ref_words else None
    return {"n": len(rows), "ref_words": ref_words, "wer": wer, "cer": cer,
            "substitutions": subs, "deletions": dels, "insertions": ins}


# --------------------------------------------------------------------------- loading
def find_scored_predictions(run_dir, split, model_name="qwen"):
    """Locate scored_predictions.jsonl inside a run folder, trying the known layouts."""
    run_dir = Path(run_dir)
    candidates = [
        run_dir / f"eval_{split}" / model_name / "scored_predictions.jsonl",  # run_experiment.py layout
        run_dir / model_name / "scored_predictions.jsonl",                     # run_asr_baselines.py --out <run_dir>
        run_dir / "scored_predictions.jsonl",                                  # file placed directly
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No scored_predictions.jsonl for split '{split}' under {run_dir}. "
        f"Looked in: {[str(c) for c in candidates]}")


def read_scored(path):
    """Read scored_predictions.jsonl -> {audio_path: row}. Rejects duplicate audio paths."""
    rows = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("audio_path", "speaker_id", "ref_norm", "hyp_norm"):
                if key not in row:
                    raise ValueError(f"{path}:{line_number}: missing field '{key}'")
            if row["audio_path"] in rows:
                raise ValueError(f"{path}:{line_number}: duplicate audio_path")
            rows[row["audio_path"]] = row
    if not rows:
        raise ValueError(f"{path}: no predictions")
    return rows


def discover_experiments(runs_dir, split):
    """Names of experiment folders (E3_L1, ...) that have an evaluation for `split`."""
    found = []
    for child in sorted(Path(runs_dir).iterdir()):
        if child.is_dir() and EXPERIMENT_DIR_RE.match(child.name):
            try:
                find_scored_predictions(child, split)
            except FileNotFoundError:
                continue
            found.append(child.name)
    return found


# --------------------------------------------------------------------------- analysis
def by_speaker(rows):
    """Group rows by speaker_id."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["speaker_id"]].append(row)
    return groups


def pick_hard_speakers(zero_shot_rows, fraction=HARD_FRACTION):
    """Hardest `fraction` of speakers by zero-shot WER (ceil, ties by speaker_id).

    Returns (hard_set, ranking) where ranking is [(speaker_id, wer), ...] hardest first.
    """
    per_speaker = {s: score_rows(r)["wer"] for s, r in by_speaker(zero_shot_rows).items()}
    # Hardest first; the speaker_id tie-break makes the choice deterministic.
    ranking = sorted(per_speaker.items(), key=lambda item: (-item[1], item[0]))
    n_hard = math.ceil(fraction * len(ranking))
    if not 0 < n_hard < len(ranking):
        raise ValueError("hard group must be non-empty and smaller than the speaker set")
    return {s for s, _ in ranking[:n_hard]}, ranking


def summarize_model(name, rows, hard, zero_shot_speaker_wer):
    """All headline numbers for one model on the common utterance set."""
    hard_rows = [r for r in rows if r["speaker_id"] in hard]
    other_rows = [r for r in rows if r["speaker_id"] not in hard]
    speaker_wers = {s: score_rows(r)["wer"] for s, r in by_speaker(rows).items()}
    # Change vs zero-shot per speaker, in percentage points (negative = improvement).
    deltas = {s: speaker_wers[s] - zero_shot_speaker_wer[s] for s in speaker_wers}
    overall = score_rows(rows)
    return {
        "model": name,
        "overall": overall,
        "hard": score_rows(hard_rows),
        "other": score_rows(other_rows),
        "mean_speaker_wer": statistics.mean(speaker_wers.values()),
        "median_speaker_wer": statistics.median(speaker_wers.values()),
        "speaker_wer": speaker_wers,
        "speaker_delta_pp": deltas,
        "n_speakers": len(speaker_wers),
        "n_speakers_improved": sum(d < 0 for d in deltas.values()),
        "n_hard_speakers_improved": sum(d < 0 for s, d in deltas.items() if s in hard),
        "median_speaker_delta_pp": statistics.median(deltas.values()),
    }


def read_gpu_peak(run_dir):
    """Peak allocated GPU memory (GB) recorded by train_qwen_lora.py, or None."""
    path = Path(run_dir) / "gpu_usage.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("peak_memory_allocated_gb")


def compare(runs_dir, split, experiments=None, fraction=HARD_FRACTION):
    """Load every run, restrict to common utterances, and build the comparison dict."""
    runs_dir = Path(runs_dir)
    experiments = experiments or discover_experiments(runs_dir, split)
    if not experiments:
        raise SystemExit(f"No experiment folders with an eval_{split} found in {runs_dir}")

    # Load every run: zero-shot first, then the experiments in the order given.
    loaded = {}
    for name in [ZERO_SHOT_NAME] + list(experiments):
        try:
            loaded[name] = read_scored(find_scored_predictions(runs_dir / name, split))
        except FileNotFoundError as exc:
            hint = ("  -> run the baseline first: python run_experiment.py --stages zero-shot"
                    if name == ZERO_SHOT_NAME else "")
            raise SystemExit(f"{exc}\n{hint}".rstrip())

    # Keep only utterances EVERY run scored, so no model is judged on an easier subset.
    common = set.intersection(*(set(rows) for rows in loaded.values()))
    if not common:
        raise SystemExit("The runs share no scored utterances; are they on the same manifest?")
    dropped = {name: len(rows) - len(common) for name, rows in loaded.items()}
    aligned = {name: [rows[p] for p in sorted(common)] for name, rows in loaded.items()}

    # The reference text must be identical across runs for the same utterance.
    base = loaded[ZERO_SHOT_NAME]
    for name, rows in loaded.items():
        bad = [p for p in common if rows[p]["ref_norm"] != base[p]["ref_norm"]]
        if bad:
            raise SystemExit(f"{name}: reference text differs from zero-shot on {len(bad)} utterances "
                             "(different manifest or normalizer?)")

    hard, ranking = pick_hard_speakers(aligned[ZERO_SHOT_NAME], fraction)
    zero_shot_speaker_wer = dict(ranking)
    models = [summarize_model(name, rows, hard, zero_shot_speaker_wer) for name, rows in aligned.items()]
    for model in models:
        # Relative WER reduction vs zero-shot (positive = better); 0 for zero-shot itself.
        zs = models[0]["overall"]["wer"]
        model["relative_wer_reduction_pct"] = 100 * (zs - model["overall"]["wer"]) / zs if zs else None
        model["peak_gpu_gb"] = None if model["model"] == ZERO_SHOT_NAME else read_gpu_peak(runs_dir / model["model"])

    return {
        "split": split,
        "n_common_utterances": len(common),
        "n_dropped_by_intersection": dropped,
        "n_speakers": len(ranking),
        "hard_fraction": fraction,
        "hard_speakers": sorted(hard),
        "zero_shot_speaker_ranking": ranking,
        "models": models,
    }


# --------------------------------------------------------------------------- output
def fmt(value, digits=2):
    """Format a number for tables; None -> '-'."""
    return "-" if value is None else f"{value:.{digits}f}"


def markdown_table(result):
    """The headline table as GitHub-flavoured markdown."""
    header = ["model", "overall WER", "overall CER", "hard WER", "other WER", "mean spk WER",
              "rel. WER red. %", "spk improved", "hard spk improved", "median Δ pp", "peak GPU GB"]
    n_hard = len(result["hard_speakers"])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for m in result["models"]:
        lines.append("| " + " | ".join([
            m["model"], fmt(m["overall"]["wer"]), fmt(m["overall"]["cer"]),
            fmt(m["hard"]["wer"]), fmt(m["other"]["wer"]), fmt(m["mean_speaker_wer"]),
            fmt(m["relative_wer_reduction_pct"], 1),
            f"{m['n_speakers_improved']}/{m['n_speakers']}",
            f"{m['n_hard_speakers_improved']}/{n_hard}",
            fmt(m["median_speaker_delta_pp"]), fmt(m["peak_gpu_gb"], 1),
        ]) + " |")
    return "\n".join(lines)


def write_outputs(result, out_dir):
    """Write json / csv / markdown files and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = result["split"]

    # Full machine-readable record.
    (out_dir / f"comparison_{split}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # One row per model.
    with (out_dir / f"comparison_summary_{split}.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n", "wer", "cer", "substitutions", "deletions", "insertions", "hard_wer",
                    "other_wer", "mean_speaker_wer", "median_speaker_wer", "relative_wer_reduction_pct",
                    "speakers_improved", "hard_speakers_improved", "median_speaker_delta_pp", "peak_gpu_gb"])
        for m in result["models"]:
            o = m["overall"]
            w.writerow([m["model"], o["n"], o["wer"], o["cer"], o["substitutions"], o["deletions"],
                        o["insertions"], m["hard"]["wer"], m["other"]["wer"], m["mean_speaker_wer"],
                        m["median_speaker_wer"], m["relative_wer_reduction_pct"], m["n_speakers_improved"],
                        m["n_hard_speakers_improved"], m["median_speaker_delta_pp"], m["peak_gpu_gb"]])

    # One row per speaker: WER of every model and its change vs zero-shot.
    hard = set(result["hard_speakers"])
    with (out_dir / f"comparison_per_speaker_{split}.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        names = [m["model"] for m in result["models"]]
        w.writerow(["speaker_id", "group"] + [f"{n}_wer" for n in names]
                   + [f"{n}_delta_pp" for n in names[1:]])
        for speaker, _ in result["zero_shot_speaker_ranking"]:          # hardest first
            row = [speaker, "hard" if speaker in hard else "other"]
            row += [m["speaker_wer"][speaker] for m in result["models"]]
            row += [m["speaker_delta_pp"][speaker] for m in result["models"][1:]]
            w.writerow(row)

    table = markdown_table(result)
    notes = (f"\n\nSplit: **{split}** | common utterances: {result['n_common_utterances']:,} | "
             f"speakers: {result['n_speakers']} (hard = {len(hard)}, hardest by zero-shot WER)\n"
             f"Δ pp = per-speaker WER change vs zero-shot in percentage points (negative is better).\n")
    (out_dir / f"comparison_{split}.md").write_text(table + notes, encoding="utf-8")
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--split", choices=("val", "test"), default="val",
                    help="val is for development; test is the final evaluation only")
    ap.add_argument("--final", action="store_true",
                    help="required for --split test: acknowledges this is the one final test-set comparison")
    ap.add_argument("--experiments", nargs="*", help="experiment folder names (default: auto-discover E*_L*)")
    ap.add_argument("--hard-fraction", type=float, default=HARD_FRACTION)
    ap.add_argument("--out", type=Path, help="default: <runs-dir>/comparison_<split>")
    args = ap.parse_args(argv)

    # The test set is reserved for the final evaluation; make it an explicit choice.
    if args.split == "test" and not args.final:
        ap.error("--split test requires --final (the test set is only for the final evaluation)")
    if not 0 < args.hard_fraction < 1:
        ap.error("--hard-fraction must be between 0 and 1")

    result = compare(args.runs_dir, args.split, args.experiments, args.hard_fraction)
    out_dir = write_outputs(result, args.out or args.runs_dir / f"comparison_{args.split}")

    print(markdown_table(result))
    print(f"\ncommon utterances: {result['n_common_utterances']:,}; "
          f"dropped by intersection: {result['n_dropped_by_intersection']}")
    print(f"hard speakers (top {args.hard_fraction:.0%} by zero-shot WER): {len(result['hard_speakers'])} "
          f"of {result['n_speakers']}")
    if not HAVE_JIWER:
        print("NOTE: jiwer not installed; used the built-in scorer (same totals, S/D/I split may differ slightly).",
              file=sys.stderr)
    print(f"wrote results to {out_dir}")


if __name__ == "__main__":
    main()
