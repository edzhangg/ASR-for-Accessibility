# Qwen 100-step experiment: preparation and pending GPU queue

Prepared on November 22, 2026. CPU preparation is complete. GPU work is authorized but must wait for the existing job. Read `runs/qwen_100step_30s_20261122/queue_status.json` for the live state; do not assume that GPU stages have completed merely because these instructions exist.

## Walkthrough

### How the three datasets were chosen

The original split is by **speaker**. A person belongs to only one of training, validation or test. In the shared preparation code, 25 speakers are randomly selected from the official Train speakers using seed 42 for validation. The remaining Train speakers go into training. Official Dev speakers go into the project test set. This split does not balance speech categories or speaker difficulty. The saved speaker assignment file records the actual original assignment; this experiment reuses the existing splits.

The new helper then prepares this smaller experiment from those existing splits:

| Set | Selection and size | Location relative to the repository root |
|---|---|---|
| Training pool | Keep clips up to 30 seconds. 91,035 original rows become 87,964 rows, with all 221 training speakers retained. | `runs/qwen_100step_30s_20261122/prepared/train.jsonl` |
| Validation exam | From 10,733 original validation rows, keep 10,260 clips up to 30 seconds. Select 1,000 once using seed 42, balancing speech categories. There are 334 commands, 333 novel sentences and 333 spontaneous prompts. All 25 validation speakers appear, with unequal recording counts. | `runs/qwen_100step_30s_20261122/prepared/val_30s_seed42_1000.jsonl` |
| Test | Reuse the original held-out test split. This helper does not filter, relocate or evaluate it. Its row count and audio accessibility require inspection on the GPU machine. | Original manifest expected at `data/pd_prepared/test.jsonl` |

Category balancing here applies to the 1,000 validation recordings. Training draws instead follow speaker difficulty. The preparation helper creates experiment copies of the source train/validation manifests under `prepared/source_manifests/`, with corrected audio paths. The shared converter also writes a converted `prepared/val.jsonl` and a training pilot. The fixed evaluation uses `val_30s_seed42_1000.jsonl`; training uses the full prepared training pool.

According to the remote preparation report, training and the frozen validation inputs are ready. The three synced files on this Mac contain instructions and code. Live progress and generated data remain on the GPU machine. Read its `queue_status.json` for current GPU progress. A `preparation.json` file records successful completion of the helper. The test manifest is an existing input whose availability needs a separate check.

### What the training settings mean

Batch size 1 means one recording is processed at a time. Gradient accumulation 8 means the model collects learning signals from eight such recordings before updating its adapters once. One optimizer step is that update. Therefore 100 steps use about 800 recording selections for each model.

The sampler draws from the 87,964-row training pool **with replacement**. A recording can be selected multiple times, and many recordings will never be selected during this short run. Each speaker's probability is divided equally among that speaker's retained recordings. E3 gives the hardest quarter of training speakers half the total draw probability. E4 gradually increases speaker probability with their existing difficulty WER, using the shared floor and cap. The weights stay fixed during training. Each configuration starts from the original pretrained model and trains its own adapters.

### Which scripts do the work

`prepare_pd_dataset.py`, `prepare_qwen_finetuning.py`, `make_sampling_weights.py`, `hard_speaker_sampler.py`, `lora_configs.py`, `train_qwen_lora.py`, `run_asr_baselines.py`, `compare_runs.py` and the master `run_experiment.py` belong to the shared pipeline. `run_experiment.py` launches the existing training, evaluation and comparison scripts.

The GPU agent added this guide, `prepare_experiment.py` and `queue_experiment.py` for this particular run. The queue wraps the team's runner. It adds waiting for the GPU, checks for exactly 100 completed steps, output checks, status files and reports. It uses a frozen copy of the shared code in the experiment's `source/` directory.

The Fixed GPU protocol below describes the GPU work after CPU preparation. The sequence is E4_L1 training then validation, original-model validation, E4_L2 training then validation, E3_L1 training then validation, E3_L2 training then validation, followed by CPU comparison. All five evaluations use the same frozen validation file. The baseline loads the original Qwen model. Each of the four adapted evaluations loads that original model with its own trained adapter; the evaluator merges the adapter into the loaded base for inference.

### Starting, waiting, recovery and results

A running queue automatically launches its next stage after its GPU checks pass. It requires the old PID 11200 to disappear and GPU memory/use to stay low for three readings. Creating the script alone does not start it. You can start the queue yourself using the commands below after checking that another copy is not already running. A manually launched individual runner command bypasses the waiting logic.

Recovery is partial. Fully completed models listed in the queue state are checked and skipped on restart. The evaluator can reuse cached predictions. Interrupted training can use the shared trainer's checkpoint-resume support, but the queue itself stops when it sees an unfinished model folder and requires agent inspection. Its checkpoint interval is 100 steps, so an interruption before step 100 may leave no resumable checkpoint. A killed queue can leave a stale lock; check the recorded PID and any child training/evaluation processes before removing it.

`RESULTS.md` is generated automatically when the queue starts and refreshed after completed models, failures and final comparison. Early tables can contain pending rows. The `report` command below refreshes it from existing outputs without training or inference. The queue's state `complete`, five complete model rows, and the final comparison together establish completion.

## Git Bash commands on the Windows GPU machine

Use these in **Git Bash on Windows** from the repository root. They use the same Windows Python environment as the PowerShell examples. For command comparison, PowerShell `$gridRoot = ...` becomes Bash `gridRoot=...`, PowerShell `Join-Path` becomes `"$gridRoot/file"`, and the PowerShell call operator `&` is replaced by calling `"$gridPy"` directly.

```bash
cd '/c/MIRE team folder/ASR-for-Accessibility'
gridPy='./.venv-qwen-lora/Scripts/python.exe'
gridRoot='runs/qwen_100step_30s_20261122'
gridCode="$gridRoot/source"
gridConfig="$gridRoot/experiment_config.json"

# Check preparation and current status. Missing files are meaningful here.
ls "$gridRoot/preparation.json" "$gridRoot/experiment_config.json"
cat "$gridRoot/queue_status.json"
tail -n 30 "$gridRoot/execution.log"
nvidia-smi

# List Python processes and their command lines through Windows PowerShell.
powershell.exe -NoProfile -Command 'Get-CimInstance Win32_Process | Where-Object { $_.Name -match "^python.*\.exe$" } | Select-Object ProcessId,ParentProcessId,CommandLine | Format-List'

# A lock contains the queue PID. Check that PID and child processes above.
if [ -f "$gridRoot/queue.lock" ]; then cat "$gridRoot/queue.lock"; fi

# Preview commands and validate saved inputs.
"$gridPy" "$gridRoot/queue_experiment.py" dry-run
```

After checking there is no active queue or its child job, start a single queue in the foreground. Keep this terminal open until it ends. It waits automatically for GPU availability.

```bash
"$gridPy" -u "$gridRoot/queue_experiment.py" queue
```

If you already have the GPU reserved, use this command in place of the preceding launch. It takes the same experiment lock, checks frozen inputs and honors `STOP_QUEUE` before each GPU stage. It skips both the old PID and GPU utilization/memory checks for the entire launch. Other GPU users can still start independent jobs.

```bash
"$gridPy" -u "$gridRoot/queue_experiment.py" queue --start-now
```

Same as the following, our immediate next step (nov23)

```bash
./.venv-qwen-lora/Scripts/python.exe -u \
  runs/qwen_100step_30s_20261122/queue_experiment.py queue --start-now
```

In another Git Bash terminal, initialize the variables again and monitor or refresh reports:

```bash
gridPy='./.venv-qwen-lora/Scripts/python.exe'
gridRoot='runs/qwen_100step_30s_20261122'
cat "$gridRoot/queue_status.json"
tail -n 30 "$gridRoot/execution.log"
cat "$gridRoot/RESULTS.md"
"$gridPy" "$gridRoot/queue_experiment.py" report

# Optional request to stop before the next GPU stage.
touch "$gridRoot/STOP_QUEUE"
```

A prior `STOP_QUEUE` file must be removed when you deliberately resume. Ask the agent to inspect partial outputs before resuming a failed run. The equivalent individual commands below are for troubleshooting after checking the queue is stopped:

```bash
"$gridPy" "$gridCode/run_experiment.py" --config "$gridConfig" --runs-dir "$gridRoot" --sampling e4 --lora L1 --stages train eval
"$gridPy" "$gridCode/run_experiment.py" --config "$gridConfig" --runs-dir "$gridRoot" --stages zero-shot
"$gridPy" "$gridCode/run_experiment.py" --config "$gridConfig" --runs-dir "$gridRoot" --sampling e4 --lora L2 --stages train eval
"$gridPy" "$gridCode/run_experiment.py" --config "$gridConfig" --runs-dir "$gridRoot" --sampling e3 --lora L1 --stages train eval
"$gridPy" "$gridCode/run_experiment.py" --config "$gridConfig" --runs-dir "$gridRoot" --sampling e3 --lora L2 --stages train eval
"$gridPy" "$gridCode/run_experiment.py" --config "$gridConfig" --runs-dir "$gridRoot" --stages compare
```

## Experiment location and source

- Fresh output folder: `runs/qwen_100step_30s_20261122/`.
- Source: isolated copy of `shared-finetuning`, commit `5df52d1329e64e47b3c83b640f7bd4cdc5ebe2c3`, under `source/` inside that folder.
- Existing checkout and local edits were preserved. No branch checkout or training-code edits were needed.
- Source hashes and the exact commit are saved in `source_provenance.json`.
- Interpreter: `.venv-qwen-lora/Scripts/python.exe`, Python 3.12.10; torch 2.14.1+cu130, transformers 4.57.6, qwen-asr 0.0.6, peft 0.21.3, bitsandbytes 0.50.2.

## Completed CPU preparation

1. Verified that the existing baseline training-manifest SHA-256 matches `data/pd_prepared/train.jsonl`.
2. Verified that existing predictions cover all 91,035 original training recordings and that all 221 speaker WER scores and recording counts in `runs/qwen_train_analysis/training_sampling_weights.csv` agree with the saved baseline.
3. Corrected stale audio paths only in experiment-local copies: original `C:/SAP downsampled database` paths now resolve through unchanged `audio_relpath` under `C:/SAP_downsampled_database`. Original manifests, baseline results, and audio files were not changed.
4. Used the shared preparation script with `--max-duration 30 --verify-audio` to regenerate `prepared/train.jsonl`, including `speaker_id`: **87,964 recordings, 221 speakers**. Dropped 3,071 over-duration training recordings. Retained audio headers and existence were verified on CPU.
5. Built `prepared/training_sampling_weights_v2.csv` with the shared script and `--verify`: **1,768 checks, zero mismatches**. Validated both E3 and E4 against all 87,964 retained training rows. E3 hard-speaker probability mass is 50%; E4 hard-speaker mass is approximately 48.5039%. Each strategy's total probability is 1 within numerical precision.
6. Filtered the original validation split by audio-header duration <=30 seconds **before** selection: 10,260 eligible recordings from 10,733 original rows.
7. Used the project's `run_asr_baselines.select_rows` with seed 42 to select exactly 1,000 eligible recordings once, saved to `prepared/val_30s_seed42_1000.jsonl`.

| Frozen validation category | Recordings |
|---|---:|
| Digital Assistant Commands | 334 |
| Novel Sentences | 333 |
| Spontaneous Speech Prompts | 333 |
| Total | 1,000 |

All 25 validation speakers are represented, with 8–56 recordings per speaker. Training and validation speakers are disjoint. This is a category-balanced development subset, not the natural full-validation category distribution.

### Difficulty evidence policy

Speaker difficulty was measured on the **original full training split**, including long utterances, using the saved Qwen HF baseline. These scores are reused unchanged; they were not recalculated only on <=30-second clips. Speaker probability mass is redistributed across that speaker's retained training rows by the shared sampler.

There are 91,035 original predictions and 91,033 scored rows. Two references normalize to empty text; their predictions exist, and they remain eligible for training, but the existing speaker WER excludes those unscorable references. Every training speaker has a valid aggregate difficulty score. No new training inference is needed or scheduled.

## Fixed GPU protocol

`experiment_config.json` is the shared configuration for every arm:

- Train: exactly **100 optimizer steps**, batch size 1, gradient accumulation 8, seed 42; approximately 800 weighted draws with replacement per arm, not necessarily 800 distinct recordings.
- Base: `Qwen/Qwen3-ASR-1.7B`; 4-bit training; shared rank 16 LoRA and shared learning rate 1e-4. L1 and L2 differ only as defined by the shared branch.
- Save step 100, verify `checkpoint-100/trainer_state.json`, verify final-adapter weights match checkpoint 100, and require gradient checks.
- Evaluate only the final adapter once on the frozen validation manifest; `eval.max_utts=0`, `eval.max_duration=30`, `eval.seed=42`, `eval.max_new_tokens=256`, backend `qwen-asr`.
- No in-training evaluation or early stopping for this fixed-budget pilot.
- Execution order: E4_L1 training and evaluation; original-model baseline evaluation; E4_L2 training and evaluation; E3_L1 training and evaluation; E3_L2 training and evaluation; CPU comparison/report.
- Original-model baseline is evaluated once through the same qwen-asr backend and frozen subset. Historical HF predictions are not substituted for this controlled comparison.
- No test-set evaluation.

The GPU was occupied at preparation time: RTX 4070, approximately 11.4 GB used and 100% utilization. The visible existing training worker was PID 11200, launched as `lora_test.py C:/SAP_downsampled_database` from `C:/GitHub/Care2.0-Fork` (launcher PID 27720). Some other process command lines are inaccessible. The existing job is never terminated by this workflow.

The queue waits until PID 11200 is absent **and** GPU memory used <=2,000 MiB and utilization <=10%, for three readings 20 seconds apart. It repeats this check before every GPU stage. Its idle check does not reserve the GPU against unrelated jobs launched by others.

## Commands and later operation

Run from the repository root in PowerShell. These paths are specific to this prepared experiment.

```powershell
$gridPy = (Resolve-Path '.venv-qwen-lora/Scripts/python.exe').Path
$gridRoot = (Resolve-Path 'runs/qwen_100step_30s_20261122').Path
$gridCode = Join-Path $gridRoot 'source'
$gridConfig = Join-Path $gridRoot 'experiment_config.json'
```

CPU preparation was performed with:

```powershell
& $gridPy (Join-Path $gridRoot 'prepare_experiment.py')
```

**Do not rerun that command on these frozen inputs.** It intentionally refuses to overwrite a completed or nonempty preparation. Its exact subprocess commands and provenance checks are in the helper file. To prepare a changed experiment, create a fresh output folder and a new source snapshot.

Preview all commands without model loading, training, or inference:

```powershell
& $gridPy (Join-Path $gridRoot 'queue_experiment.py') dry-run
```

Read current status and reports:

```powershell
Get-Content (Join-Path $gridRoot 'queue_status.json') -Raw
Get-Content (Join-Path $gridRoot 'RESULTS.md') -Raw
Get-Content (Join-Path $gridRoot 'queue.stdout.log') -Tail 20
Get-Content (Join-Path $gridRoot 'execution.log') -Tail 30 -ErrorAction SilentlyContinue
```

The authorized background queue is started once by Codex. **Do not start a second queue while `queue.lock` exists or the recorded queue PID is alive.** If it has stopped, inspect its logs and partial outputs before restarting. Partial adapter outputs require manual inspection; the helper deliberately refuses automatic retraining with a changed or ambiguous budget.

If no queue is running and outputs are safe to continue, the foreground command is:

```powershell
& $gridPy -u (Join-Path $gridRoot 'queue_experiment.py') queue
```

For reference, the individual shared-runner commands are below. **Do not run these alongside the queue.** They bypass its GPU-idle check and its post-stage assertions, so use the queue for normal execution.

```powershell
# E4_L1 first
& $gridPy (Join-Path $gridCode 'run_experiment.py') --config $gridConfig --runs-dir $gridRoot --sampling e4 --lora L1 --stages train eval

# Original model once on exactly the same frozen validation manifest
& $gridPy (Join-Path $gridCode 'run_experiment.py') --config $gridConfig --runs-dir $gridRoot --stages zero-shot

# The remaining three arms, unchanged budget and validation input
& $gridPy (Join-Path $gridCode 'run_experiment.py') --config $gridConfig --runs-dir $gridRoot --sampling e4 --lora L2 --stages train eval
& $gridPy (Join-Path $gridCode 'run_experiment.py') --config $gridConfig --runs-dir $gridRoot --sampling e3 --lora L1 --stages train eval
& $gridPy (Join-Path $gridCode 'run_experiment.py') --config $gridConfig --runs-dir $gridRoot --sampling e3 --lora L2 --stages train eval

# CPU-only comparison and report refresh after saved predictions exist
& $gridPy (Join-Path $gridCode 'run_experiment.py') --config $gridConfig --runs-dir $gridRoot --stages compare
& $gridPy (Join-Path $gridRoot 'queue_experiment.py') report
```

The queue runs model stages offline using the already cached original Qwen model. Dataset caches and new experiment artifacts are kept under this output folder.

To request a safe stop **between** stages, create `STOP_QUEUE` under the output folder. This does not terminate a currently running training or evaluation process:

```powershell
New-Item -ItemType File -Path (Join-Path $gridRoot 'STOP_QUEUE')
```

## Timing and reporting

Initial planning estimate after GPU access: approximately **2–3.5 hours for four arms**, plus **15–25 minutes** for the original-model baseline. GPU waiting and CPU preparation are separate. Actual L2 rate is unmeasured.

After E4_L1 evaluation, the queue writes measured training/evaluation seconds and an updated remaining-time and finish-time range to `queue_status.json` and `RESULTS.md`, then continues. 

Reports are updated after each completed model and at the end:

- `RESULTS.md`: overall and per-category WER/CER, scored/selected counts, stage durations, status and first-arm estimate.
- `results_overall.csv`, `results_categories.csv`, `results.json`: machine-readable results as they become available.
- `comparison_val/comparison_val.md`: shared comparison over common successful recordings after all models finish.
- `<model>/eval_val/qwen/summary.json`, `category_scores.csv`, `scored_predictions.jsonl`, and `errors.jsonl` if present: primary scoring evidence.
- `<arm>/checkpoint-100/trainer_state.json`, `training_config.json`, `gradient_check.json`, `gpu_usage.json`, `log_history.json`, and `run_meta.json`: training and timing evidence.

The queue treats any missing prediction or unscored frozen-validation row as an execution issue instead of silently reporting it as a complete comparison. Small 100-step training budgets and this small, category-balanced validation subset limit conclusions about full-dataset generalization.
