# Qwen ASR fine-tuning grid on one Windows GPU

Train and validate four adapters in this order: E4_L1, E4_L2, E3_L1, E3_L2. E4 uses continuous difficulty-weighted sampling and E3 oversamples hard speakers. L1 trains decoder adapters and L2 trains audio-encoder and decoder adapters. All four use the same training-speaker evidence and validation split.

## 0. Remote agent handoff: find the starting point

Run from the repository root in Windows PowerShell. Keep the same shell for the launch variables. Start with a read-only inventory. Include any known private data or prior-run folders on other drives in the search. Check for an active job before starting another on this GPU.

```powershell
git status --short --branch
git rev-parse --short HEAD
nvidia-smi
Get-Item run_experiment.py, configs/shared.json, data/pd_prepared/train.jsonl, data/pd_prepared/val.jsonl -ErrorAction SilentlyContinue | Select-Object FullName
$roots = @(Get-ChildItem -Directory | Where-Object { $_.Name -eq 'data' -or $_.Name -like 'runs*' })
Get-ChildItem -Path . -File -Filter '*sampling_weights*.csv'
$roots | ForEach-Object { Get-ChildItem $_.FullName -Recurse -File -Include preparation_report.json,*sampling_weights*.csv,run_config.json,summary.json,training_config.json,trainer_state.json,run_meta.json,adapter_config.json } | Select-Object -ExpandProperty FullName
```

Use the shared-finetuning codebase. Report the GPU, chosen cutoff, usable inputs and missing inputs, four arm statuses, next command and preliminary time budget. Use aggregate counts and paths in that report. Continue from the first missing dependency.

| What you find | Where to start |
|---|---|
| Missing prepared Parkinson’s splits | Locate the team's original split manifests and audio. If starting from raw SAP, use `prepare_pd_dataset.py --root <SAP-root> --out data/pd_prepared` with the team’s split settings. Check source audio paths on this machine. |
| Split manifests exist, converted training JSONL missing or cutoff differs | Step 3 after setting variables and checking CUDA. |
| Converted data ready | Step 4. Prefer compatible v2 CSV, then source CSV, then scored training predictions, then fresh inference. |
| Both sampling strategies validated | Step 5. Reuse matching successful smoke tests. |
| Full adapter or checkpoint exists | Step 6. Check model, paths, duration, weights and training budget against the saved configuration before reusing a folder. |
| All four adapters and validation scores complete | Step 7 for a missing validation baseline or comparison. |

## 1. Select the shared inputs and output root

The shared protocol in `FINETUNING.md` prepares training audio at 30 seconds. The evaluator reads `eval.max_duration` from `configs/shared.json`, currently 30. These are independent controls. The trainer reads whichever recordings are in its JSONL. Changing the evaluation setting alone cannot recover training clips excluded during preparation. Both duration filters skip whole clips longer than the limit.

To match a completed teammate run, open its `runs/<name>/training_config.json` and locate `train_file`. The `preparation_report.json` beside that file records `max_duration_seconds` and the numbers dropped. The evaluator records its actual `max_duration`, manifest, and model in `eval_val/qwen/run_config.json`. For an older baseline, look under its output directory in `qwen/run_config.json`. The older Monica branch shows 30 seconds for training-data preparation and 120 seconds in evaluation examples. A saved Qwen HF test run confirms 120 seconds for that evaluation. It supplies neither the training cutoff nor the training-speaker WERs. For a team-grid comparison use the same training manifest, difficulty CSV and evaluation settings across all four runs. The commands below implement the **120-second grid**. Set `$seconds = 30` instead for the documented shared cutoff. Use a fresh run root if changing data, weights or hyperparameters after a run.

```powershell
$seconds = 120
$prepared = "data/qwen_finetuning/${seconds}s"
$weights = "$prepared/training_sampling_weights_v2.csv"
$runs = "runs_e4grid_${seconds}s"
$smoke = "runs_smoke_e4_${seconds}s"
$settings = @('--set', "data.train_file=$prepared/train.jsonl", '--set', "data.weights_csv=$weights", '--set', "eval.max_duration=$seconds")
```

## 2. Verify the environment and install only if needed

On the project's Windows RTX 4070 machine, the repository documents a CUDA 13.0 PyTorch wheel. Check the driver with `nvidia-smi`. If using another GPU machine, choose the PyTorch CUDA wheel index compatible with its driver. The dedicated Python 3.11 environment below can be reused between sessions. Calling its interpreter directly avoids PowerShell activation settings.

```powershell
nvidia-smi
$py = ".\.venv-qwen-lora\Scripts\python.exe"
if (Test-Path $py) { & $py -c "import torch, torchvision, bitsandbytes, qwen_asr, peft, datasets, librosa, soundfile, jiwer, whisper_normalizer; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))" }
```

If that probe passed, go to step 3. For a missing or broken environment on the documented machine:

```powershell
if (!(Test-Path .venv-qwen-lora/Scripts/python.exe)) { py -3.11 -m venv .venv-qwen-lora }
$py = ".\.venv-qwen-lora\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
& $py -m pip install -r requirements-finetune.txt
& $py -m pip check
& $py -c "import torch, bitsandbytes, qwen_asr, peft; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

## 3. Prepare the training manifest

If `$prepared/train.jsonl` already comes from these same source manifests, its report confirms the chosen duration, and each row has `speaker_id`, reuse it. Otherwise run:

```powershell
& $py prepare_qwen_finetuning.py --train data/pd_prepared/train.jsonl --val data/pd_prepared/val.jsonl --out $prepared --max-duration $seconds --verify-audio
Get-Content "$prepared/preparation_report.json"
```

The evaluation input stays `data/pd_prepared/val.jsonl`. The evaluator needs its original paths, speaker metadata and reference text. The trainer needs the converted `train.jsonl`.

## 4. Reuse or build the shared E3/E4 weights

The evidence for E4 is one zero-shot WER **percentage per training speaker**, plus the speaker ID and recording count. The CSV does not record which duration, split or model produced those scores. Check its source run with the team. To reproduce the team grid, use the same evidence across the arms even if the team deliberately estimated difficulty on a different clip set. For a new 120-second protocol with newly estimated difficulty, use the matching training baseline in C or generate it in D.

Choose **one** route:

**A. Existing team v2 CSV.** If it has E3 and E4 columns and the intended provenance, point the runner at it by copying it to the expected location. Skip the copy if it is already there.

```powershell
Copy-Item data/qwen_finetuning/training_sampling_weights_v2.csv $weights
```

**B. Existing team source CSV.** With `training_sampling_weights.csv` in the repo root, run the following. Adjust the input path if the private copy is elsewhere.

```powershell
& $py make_sampling_weights.py --in training_sampling_weights.csv --out $weights --verify
```

**C. Training baseline available, CSV missing.** Set `$evidence` to the directory containing its `qwen` folder. Check `qwen/run_config.json` for the training manifest, base model, backend and cutoff, and `qwen/summary.json` for full-manifest coverage. With matching scored predictions, skip the inference command and go directly to the conversion block. With partial matching predictions, reuse the same output directory and run the evaluator to finish the pass.

**D. No compatible training baseline or CSV.** Score the base model on the original **train** manifest once. This can take a full inference pass. The runner's own `zero-shot` stage scores validation and serves the later model comparison. Use this direct evaluator command for training-speaker evidence:

```powershell
$evidence = "runs_train_difficulty_${seconds}s"
& $py run_asr_baselines.py --manifest data/pd_prepared/train.jsonl --models qwen --qwen-backend qwen-asr --qwen-model Qwen/Qwen3-ASR-1.7B --max-utts 0 --max-duration $seconds --max-new-tokens 256 --seed 42 --out $evidence
```

Convert the evaluator's per-speaker corpus WER into the three input columns. This block checks that every retained recording has a scored prediction before using a fresh baseline. If it stops, inspect the baseline's `errors.jsonl`, `summary.json` and any empty normalized references. Clips excluded by the chosen duration already fall outside the prepared manifest.

```powershell
@'
import csv, json, sys
from collections import Counter
from pathlib import Path
prepared, evidence = map(Path, sys.argv[1:])
def rows(path):
    with path.open(encoding="utf-8") as src:
        return [json.loads(line) for line in src if line.strip()]
train = rows(prepared / "train.jsonl")
scored = rows(evidence / "qwen" / "scored_predictions.jsonl")
expected = {(r["speaker_id"], r["audio"]) for r in train}
actual = {(r["speaker_id"], r["audio_path"]) for r in scored}
if expected != actual:
    sys.exit(f"Coverage mismatch: {len(expected-actual)} missing, {len(actual-expected)} extra")
counts = Counter(r["speaker_id"] for r in train)
summary = json.loads((evidence / "qwen" / "summary.json").read_text(encoding="utf-8"))
with (prepared / "training_sampling_weights_source.csv").open("w", newline="", encoding="utf-8") as out:
    writer = csv.writer(out)
    writer.writerow(["speaker_id", "difficulty_wer_percent", "n_manifest"])
    for speaker, count in sorted(counts.items()):
        writer.writerow([speaker, summary["per_speaker"][speaker]["wer"], count])
'@ | & $py - $prepared $evidence
if ($LASTEXITCODE -ne 0) { throw "Check training baseline coverage before building weights." }
& $py make_sampling_weights.py --in "$prepared/training_sampling_weights_source.csv" --out $weights
```

**Check the chosen CSV against the actual manifest** before loading the model:

```powershell
@'
import sys
from pathlib import Path
from train_qwen_lora import read_speaker_ids
from hard_speaker_sampler import build_sampling_weights, load_speaker_weights
rows = read_speaker_ids(Path(sys.argv[1]))
for strategy in ("e3", "e4"):
    speakers = load_speaker_weights(Path(sys.argv[2]), strategy=strategy)
    weights = build_sampling_weights(rows, speakers, check_hard_mass=(strategy == "e3"))
    print(strategy, len(speakers), "speakers,", len(weights), "recordings, mass", sum(weights))
'@ | & $py - "$prepared/train.jsonl" $weights
```

E4 clips each training speaker's WER between the median speaker WER and 30 **percent**, then normalizes across speakers. The trainer reads `e4_difficulty_p_speaker` and divides a speaker's probability equally among their retained recordings. It draws with replacement, one draw per training row per epoch. If changing from 30 to 120 seconds adds speakers or makes retained counts exceed the CSV counts, the check fails. Use weights based on compatible counts and documented WER evidence.

## 5. Smoke-test both adapter configurations

Reuse a successful smoke test from the same setup. Otherwise run the corresponding line below. E4_L1 tests training and the evaluation path. E4_L2 tests gradients in the extra audio adapters. Both use a separate smoke root.

```powershell
& $py run_experiment.py --sampling e4 --lora L1 @settings --runs-dir $smoke --set training.max_steps=2 --set training.save_steps=1 --set eval.max_utts=4
& $py run_experiment.py --sampling e4 --lora L2 @settings --runs-dir $smoke --stages train --set training.max_steps=2 --set training.save_steps=1
Get-Content "$smoke/E4_L1/gradient_check.json","$smoke/E4_L2/gradient_check.json"
(Get-Content "$smoke/E4_L1/eval_val/qwen/summary.json" -Raw | ConvertFrom-Json).overall
```

Require successful exits, positive scored count, decoder gradients for L1, and audio plus decoder gradients for L2. Inspect errors on failure. Two steps check the code path; the longest eligible clips may use more memory.

## 6. Queue the full runs and measure time on this GPU

If an arm has a matching final adapter and weights, the runner skips its training and reuses matching cached validation predictions. If an arm has only a checkpoint, first resume **that arm alone** with its original settings, for example:

```powershell
# Only for an interrupted E4_L1 with a valid checkpoint and no final adapter
& $py run_experiment.py --sampling e4 --lora L1 @settings --runs-dir $runs --stages train --resume-latest
```

Repeat targeted recovery for any other interrupted arm. Use a fresh output root for a changed configuration. If a run is already active, monitor it until the GPU is available. For a fresh or partially completed grid, preview and then launch one sequential queue:

```powershell
& $py run_experiment.py --sampling e4 e3 --lora L1 L2 @settings --runs-dir $runs --stages train eval --dry-run
& $py run_experiment.py --sampling e4 e3 --lora L1 L2 @settings --runs-dir $runs --stages train eval --keep-going
```

The explicit order finishes training **and validation** for E4_L1 first. Each arm gets its own folder under `$runs`. Check failed stages after the queue exits. Use step 7 after all four validation results are complete.

### Estimate remaining GPU hours

Start the budget during inventory. If previous runs exist on this machine, inspect their `training_config.json` (GPU, batch size, accumulation, rows, LoRA), `log_history.json` (final training speed), and `run_meta.json` (elapsed seconds per stage). Prefer rates from runs with the same cutoff and workload. Otherwise give a provisional range with explicit assumed rates and revise it during the first real job. Use a second PowerShell window for monitoring; reinitialize the path variables from step 1 there.

1. **Count the workload.** This command reads the actual filtered training and validation JSONLs. The single-GPU defaults use one epoch, batch size 1 and eight batches per optimizer step. The trainer’s displayed total step count supersedes the planning estimate.

   ```powershell
   $nTrain = ([System.IO.File]::ReadLines((Resolve-Path "$prepared/train.jsonl").Path) | Measure-Object).Count
   $nVal = ([System.IO.File]::ReadLines((Resolve-Path "$prepared/val.jsonl").Path) | Measure-Object).Count
   $steps = [math]::Ceiling($nTrain / 8)
   "Train rows: $nTrain | eligible validation rows: $nVal | estimated steps per full arm: $steps"
   ```

2. **Time training.** At roughly steps 50–100 of the real E4_L1, read seconds per optimizer step from the live trainer progress bar, or divide elapsed seconds between two progress readings by steps advanced. Invert an `it/s` reading to get seconds per step. Measure a second window and use the two rates for a provisional range. Repeat on E4_L2. Use L1 rates for E3_L1 and L2 rates for E3_L2 until each has its own reading. Allow for checkpoint saves and longer clips. For each arm, **hours remaining = remaining optimizer steps × seconds per step / 3600**. Use zero for completed training and the latest resumable step for interrupted runs. For scale only, 91,000 rows at 5 seconds/step would take about 15.8 training hours per fresh arm.

3. **Time inference separately.** Include the optional training-split zero-shot pass from step 4D, four adapter validation passes and one shared validation baseline. For a running pass, use two wall-clock readings of the evaluator’s `processed X/Y` counter. Divide the change in processed rows by elapsed seconds, then divide remaining pending rows by that rate and by 3600. The evaluator prints cached and pending counts at startup. For future uncached validation passes use the first full validation’s elapsed time from `run_meta.json`; until then use a measured inference rate and mark the estimate provisional. Account for model loading, duration skips and scoring overhead. A four-clip smoke evaluation includes startup overhead.

4. **Report a total against the reservation.** List each arm with status, remaining steps, seconds/step, training hours, and validation hours. Add the remaining training-evidence and validation-baseline hours once each. Sum the ranges because this queue runs sequentially on one GPU. Include the measurement time, estimated finish time and any gap against the available reservation. Mark unmeasured rates as assumptions. Update after the first full validation, first L2 speed reading and any restart.

## 7. Validate and compare the four results

For each arm check `adapter-final/`, `gradient_check.json`, `eval_val/qwen/summary.json` and `scored_predictions.jsonl`. Inspect successful prediction counts and errors. All four full validation runs should use the same manifest, cutoff and full-evaluation setting (`max_utts=0`).

With all four results present, run the missing shared zero-shot validation baseline and compare:

```powershell
& $py run_experiment.py @settings --runs-dir $runs --stages zero-shot compare
```

If a compatible full validation baseline already exists at `$runs/zero_shot/eval_val/qwen/`, use `--stages compare` alone. Read `$runs/comparison_val/comparison_val.md` and verify the table includes the baseline and all four arms. Record measured stage durations and the number of common scored utterances alongside the WER/CER comparison.
