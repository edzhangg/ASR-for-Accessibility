"""Run four fresh 1,000-step models and score checkpoints 500 and 1,000 on the pilot subset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PILOT = ROOT / "runs/qwen_100step_30s_20261122"
OUT = ROOT / "runs/qwen_1000step_30s"
STEPS = (500, 1000)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_score(folder):
    summary = read(folder / "eval_val/qwen/summary.json")
    if (summary["n_selected"], summary["n_predicted"], summary["overall"]["n"]) != (1000, 1000, 1000):
        raise ValueError(f"Incomplete evaluation in {folder}")
    return summary


def report(status, rows):
    lines = ["# Qwen 1,000-step validation results", "", status, "",
             "All models use the pilot's frozen 1,000 recordings. Times are minutes.",
             "The 100-step scores come from the earlier pilot. Both later checkpoints come from one fresh training run per configuration.", "",
             "| Model | Scored | WER % | CER % | Mean speaker WER % |",
             "|---|---:|---:|---:|---:|"]
    categories = ["", "| Model | Category | Scored | WER % | CER % |", "|---|---|---:|---:|---:|"]
    for name, summary in rows.items():
        scores = summary["overall"]
        lines.append(f"| {name} | {scores['n']} | {scores['wer']} | {scores['cer']} | {summary['mean_speaker_wer']} |")
        for category, score in summary["by_category"].items():
            categories.append(f"| {name} | {category} | {score['n']} | {score['wer']} | {score['cer']} |")
    (OUT / "RESULTS.md").write_text("\n".join(lines + categories) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="verify pilot inputs and print commands")
    args = parser.parse_args()
    sys.path.insert(0, str(PILOT))
    import queue_experiment as pilot
    pilot.check_inputs()
    sys.path.insert(0, str(pilot.SOURCE))
    import run_experiment as runner
    cfg = read(pilot.CONFIG)
    cfg["training"].update(max_steps=1000, save_steps=500, save_total_limit=3)
    cfg["_notes"].append("Fresh 1000-step grid, evaluation at steps 500 and 1000.")
    scores = {}
    pilot.validate_eval("zero_shot")
    baseline = PILOT / "zero_shot"
    scores["zero_shot"] = validate_score(baseline)
    if read(baseline / "eval_val/qwen/run_config.json")["model_id"] != cfg["model"]["base"]:
        raise ValueError("Unexpected baseline model")
    for sampling, lora in pilot.ARMS:
        name = runner.experiment_name(sampling, lora)
        scores[name + "_100"] = validate_score(PILOT / name)

    commands = []
    evaluations = []
    for sampling, lora in pilot.ARMS:
        name = runner.experiment_name(sampling, lora)
        folder = OUT / name
        commands.append((name, runner.build_train_cmd(cfg, sampling, lora, OUT), folder))
        for step in STEPS:
            label = f"{name}_{step}"
            commands.append((label, runner.build_eval_cmd(cfg, "val", OUT, label, folder / f"checkpoint-{step}"), None))
            evaluations.append(label)
    commands.append(("comparison", runner.build_compare_cmd("val", OUT, evaluations, False), None))
    if args.dry_run:
        for label, cmd, _ in commands:
            print(label, subprocess.list2cmdline(cmd))
        return

    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "queue.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        config_path = OUT / "experiment_config.json"
        if config_path.exists() and read(config_path) != cfg:
            raise ValueError("Output folder contains a different configuration")
        pilot.save_json(config_path, cfg)
        shutil.copyfile(PILOT / "source_provenance.json", OUT / "source_provenance.json")
        source = baseline / "eval_val/qwen"
        dest = OUT / "zero_shot/eval_val/qwen"
        dest.mkdir(parents=True, exist_ok=True)
        for file in ("scored_predictions.jsonl", "summary.json", "run_config.json"):
            shutil.copyfile(source / file, dest / file)
        status_path = OUT / "queue_status.json"
        status = read(status_path) if status_path.exists() else {}
        timings = status.setdefault("completed_stage_minutes", {})
        for label, cmd, folder in commands:
            status.update(state="running", stage=label)
            pilot.save_json(status_path, status)
            if label not in timings:
                if folder and folder.exists() and any(folder.iterdir()):
                    raise RuntimeError(f"Inspect partial training outputs before retrying: {folder}")
                report("Current stage: " + label, scores)
                print(label, subprocess.list2cmdline(cmd), flush=True)
                started = time.monotonic()
                env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           HF_DATASETS_CACHE=str(PILOT / "cache/datasets"))
                with (OUT / "execution.log").open("a", encoding="utf-8") as log:
                    print(label, subprocess.list2cmdline(cmd), file=log, flush=True)
                    subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
                minutes = round((time.monotonic() - started) / 60, 2)
            if folder:
                for step in STEPS:
                    checkpoint = folder / f"checkpoint-{step}"
                    state = read(checkpoint / "trainer_state.json")
                    if (state["global_step"], state["max_steps"]) != (step, 1000):
                        raise ValueError(f"Unexpected checkpoint state: {checkpoint}")
                if not read(folder / "gradient_check.json"):
                    raise ValueError(f"Missing gradient checks: {folder}")
                adapters = [folder / sub / "adapter_model.safetensors" for sub in ("adapter-final", "checkpoint-1000")]
                if len({hashlib.sha256(p.read_bytes()).digest() for p in adapters}) != 1:
                    raise ValueError(f"Final weights differ from checkpoint 1000: {folder}")
            elif label in evaluations:
                scores[label] = validate_score(OUT / label)
            if label not in timings:
                timings[label] = minutes
            pilot.save_json(status_path, status)
        status.update(state="complete", stage=None)
        pilot.save_json(status_path, status)
        report("Complete", scores)
        with (OUT / "RESULTS.md").open("a", encoding="utf-8") as stream:
            stream.write("\n| Stage | Minutes |\n|---|---:|\n")
            for label, minutes in timings.items():
                stream.write(f"| {label} | {minutes} |\n")
        print(OUT / "RESULTS.md", flush=True)
    except Exception as exc:
        if "status" in locals():
            status.update(state="failed", error=str(exc))
            pilot.save_json(status_path, status)
        report(f"Stopped: {exc}", scores)
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
