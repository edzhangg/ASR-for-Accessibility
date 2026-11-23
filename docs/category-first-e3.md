# Category-first E3 sampling

Category-first E3 controls the training mix by sampling in this order:

**Category → hard/non-hard group → speaker → recording**

The default config assigns 50% to Digital Assistant Commands, 30% to Novel Sentences, and 20% to Spontaneous Speech Prompts. Non-spontaneous Speech Prompts and any other category absent from the config receive zero probability.

Within each configured category, hard and non-hard speakers each receive half of that category's probability. Speakers are sampled uniformly within each category and difficulty group; recordings are sampled uniformly within each speaker and category. Hard labels come from the existing `hard_top25` column.

This option was added to hold the speech-category mix fixed while comparing hard-speaker sampling. Original E3 balances hard and non-hard speakers overall without fixing category proportions. The older `--category-stratified` option rebalances categories within each speaker and does not impose category-first 50/30/20 targets. All three remain separate strategies.


## Sampling methodology

For a recording $i$ belonging to speaker $s$, category $c$, and difficulty group $g$, its sampling probability is:

$$
P(i) = P(c) \cdot P(g \mid c) \cdot P(s \mid c,g) \cdot P(i \mid s,c)
$$

Equivalently,

$$
P(i) = \frac{q_c \, p_g}{N_{c,g} \, n_{s,c}}
$$

where:

- $q_c$ is the configured probability of category $c$.
- $p_g$ is the probability of the hard or non-hard group within the category ($0.5$ by default).
- $N_{c,g}$ is the number of unique speakers in category $c$ and difficulty group $g$.
- $n_{s,c}$ is the number of recordings belonging to speaker $s$ in category $c$.

Sampling uses replacement through PyTorch's `WeightedRandomSampler`.

The resulting distribution satisfies:

$$
\sum_i P(i) = 1
$$

For every configured category:

$$
P(c) = q_c
$$

$$
P(c,\text{hard}) = 0.5q_c
$$

$$
P(c,\text{non-hard}) = 0.5q_c
$$

Categories absent from the configuration receive zero sampling probability.


## Configure and run

Edit `configs/category_sampling.json` to change the positive category proportions; they must sum to 1. `hard_group_probability` sets the hard share within every category (0.5 by default). Each configured category must contain at least one hard and one non-hard speaker in the training manifest.

Prepare manifests with `prepare_qwen_finetuning.py`, which writes `speaker_id` and `category` alongside Qwen's `audio` and `text` fields. The trainer ignores these metadata fields when building model inputs.

Enable category-first E3 through the unified experiment launcher:

```sh
python run_experiment.py \
  --sampling e3 \
  --lora L1 \
  --category-config configs/category_sampling.json \
  --stages train
```

The launcher forwards the option to `train_qwen_lora.py`. Existing experiments without this option retain their prior sampling behavior.

Direct trainer usage is also supported with:

```sh
python train_qwen_lora.py \
  --train-file TRAIN.jsonl \
  --weights-csv WEIGHTS.csv \
  --category-config configs/category_sampling.json \
  --output-dir runs/category_e3_l1 \
  --sampling e3 \
  --lora-config L1
```

**Important:** Category-first sampling is currently implemented specifically for E3. Passing `--category-config` does not automatically apply category balancing to E1, E2, or E4.

## Validation

### Unit tests

Run the complete unit suite and category-first tests from the repository root:

```sh
python -m unittest discover -s tests -v
python -m unittest discover -s tests -p "test_category_first_sampler.py" -v
```

The previously completed unit-test run reported:

- 68 tests passed.
- 1 test skipped because `jiwer` was unavailable.
- `git diff --check` passed.

Unit tests use synthetic rows to verify theoretical marginals, hard/non-hard group masses, excluded categories, and configuration validation.

### Real-data validation — 2026-11-23

Category-first E3 was validated against the actual training manifest on the team's Windows GPU VM.

**Environment**

| Property | Value |
|---|---|
| Machine | Windows VM |
| GPU | NVIDIA GeForce RTX 4070 (12 GB) |
| Worktree | `ASR-category-e3` |
| Branch | `category-stratified-sampling` |
| Training recordings | 87,875 |
| Speakers in weights CSV | 221 |
| Missing speaker IDs | 0 |

**Input files**

- Training manifest: `../ASR-for-Accessibility/data/qwen_finetuning_category_smoke/train.jsonl`
- Speaker weights: `../ASR-for-Accessibility/runs/qwen_100step_30s_20261122/prepared/training_sampling_weights_v2.csv`
- Category configuration: `configs/category_sampling.json`

The paths above reflect the VM's worktree layout and should be adapted for other machines.

#### Dataset composition

| Category | Recordings | Hard speakers | Non-hard speakers |
|---|---:|---:|---:|
| Digital Assistant Commands | 60,913 | 56 | 165 |
| Novel Sentences | 20,134 | 55 | 165 |
| Spontaneous Speech Prompts | 6,828 | 54 | 162 |
| **Total** | **87,875** | | |

Speaker counts are category-specific; the same speaker can appear in multiple categories.

All 87,875 recordings had speaker IDs present in the weights CSV.

#### Exact probability validation

An independent CPU calculation using the real manifest and speaker metadata verified the theoretical probability masses.

| Category | Total probability | Hard | Non-hard |
|---|---:|---:|---:|
| Digital Assistant Commands | 0.50 | 0.25 | 0.25 |
| Novel Sentences | 0.30 | 0.15 | 0.15 |
| Spontaneous Speech Prompts | 0.20 | 0.10 | 0.10 |
| **Total** | **1.00** | **0.50** | **0.50** |

**Result: PASS**

The total probability was 1.0, and every category and difficulty-group mass matched the configured targets.

#### Actual sampler simulation: 100,000 draws

A separate validation directly executed the implemented sampler using:

- `load_category_config()`
- `load_speaker_weights()`
- `build_category_first_weights()`
- `make_category_first_sampler()`

The simulation used 100,000 replacement draws with random seed 42.

| Category | Target | Observed | Hard | Non-hard |
|---|---:|---:|---:|---:|
| Digital Assistant Commands | 50.00% | 49.82% | 24.97% | 24.85% |
| Novel Sentences | 30.00% | 30.05% | 15.13% | 14.91% |
| Spontaneous Speech Prompts | 20.00% | 20.13% | 9.96% | 10.17% |

**Result: PASS**

The observed frequencies closely matched the theoretical category and hard/non-hard proportions within expected random sampling variation.

This confirms that the implemented sampler produces the intended distribution on the real training manifest, not only on synthetic test data.

#### Reproducing the real-data simulation

Run the following Python code from the repository root, replacing the input paths as needed:

```python
import json
from collections import Counter

from category_first_sampler import (
    load_category_config,
    build_category_first_weights,
    make_category_first_sampler,
)
from hard_speaker_sampler import load_speaker_weights

with open("TRAIN.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

speakers = load_speaker_weights("WEIGHTS.csv", strategy="e3")
config = load_category_config("configs/category_sampling.json")

weights = build_category_first_weights(rows, speakers, config)

print("Recordings:", len(rows))
print("Total probability:", sum(weights))

sampler = make_category_first_sampler(
    rows,
    speakers,
    config,
    num_samples=100_000,
    seed=42,
)

counts = Counter()

for index in sampler:
    row = rows[index]
    hard = bool(speakers[row["speaker_id"]]["hard"])
    counts[(row["category"], hard)] += 1

for category, target in config["category_proportions"].items():
    hard = counts[(category, True)] / 100_000
    nonhard = counts[(category, False)] / 100_000

    print(
        f"{category}: "
        f"target={target:.1%}, "
        f"observed={hard + nonhard:.2%}, "
        f"hard={hard:.2%}, "
        f"nonhard={nonhard:.2%}"
    )
```

This validation runs on CPU and does not load the ASR model.

## GPU integration smoke test — 2026-11-23

Following the successful CPU validation, a two-step GPU smoke test was completed to verify that category-first E3 integrates with the Qwen3-ASR decoder-only LoRA training pipeline.

### Objective

Verify that the training pipeline can:

1. Load the pretrained Qwen3-ASR model.
2. Apply 4-bit quantization.
3. Inject decoder-only LoRA adapters.
4. Load the real training manifest.
5. Initialize the category-first E3 sampling configuration.
6. Complete forward and backward passes.
7. Verify gradient flow through the LoRA adapters.
8. Execute optimizer steps and log training loss.
9. Save the resulting LoRA adapter and checkpoint.

This is an integration test, not a model-quality evaluation.

### Environment and configuration

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen3-ASR-1.7B` |
| GPU | NVIDIA GeForce RTX 4070 (12 GB) |
| Python | 3.12.10 |
| PyTorch | 2.14.1+cu130 |
| Transformers | 4.57.6 |
| qwen-asr | 0.0.6 |
| PEFT | 0.21.3 |
| bitsandbytes | 0.50.2 |
| datasets | 5.0.1 |
| Sampling | E3, category-first |
| LoRA configuration | L1 (decoder-only) |
| Training recordings | 87,875 |
| Maximum optimizer steps | 2 |
| Batch size | 1 |
| Gradient accumulation | 1 |
| Learning rate | 1e-4 |
| Quantization | 4-bit |
| Gradient checkpointing | Enabled |
| Optimizer | `paged_adamw_8bit` |
| Random seed | 42 |
| Evaluation | Disabled |

The test used the existing `.venv-qwen-lora` Python environment on the VM.

### Reproduction command

The following PowerShell commands reproduce the test in the same VM worktree layout:

```powershell
$python = (Resolve-Path "..\ASR-for-Accessibility\.venv-qwen-lora\Scripts\python.exe").Path
$base = (Resolve-Path "..\ASR-for-Accessibility").Path

& $python train_qwen_lora.py `
  --train-file "$base\data\qwen_finetuning_category_smoke\train.jsonl" `
  --weights-csv "$base\runs\qwen_100step_30s_20261122\prepared\training_sampling_weights_v2.csv" `
  --category-config "configs/category_sampling.json" `
  --output-dir "runs\category_e3_l1_smoke_2step" `
  --sampling e3 `
  --lora-config L1 `
  --max-steps 2 `
  --batch-size 1 `
  --grad-acc 1 `
  --num-workers 0 `
  --load-in-4bit 1 `
  --gradient-checkpointing 1 `
  --log-steps 1 `
  --eval-steps 0
```

Use a fresh output directory when repeating the experiment to avoid overwriting previous artifacts.

### LoRA injection and gradient-flow verification

The model loaded successfully, and the L1 configuration produced:

| Metric | Result |
|---|---:|
| Decoder LoRA target modules | 112 |
| Audio encoder LoRA target modules | 0 |
| Trainable parameters | 6,422,528 |
| Total reported parameters | 1,183,331,968 |
| Trainable percentage | 0.543% |

The training script reported:

```text
[L1] 112 decoder + 0 audio target modules;
trainable 6,422,528 / 1,183,331,968 (0.543%)

Gradient-flow check passed:
{
    'decoder': {
        'lora_B_matrices': 112,
        'with_gradient': 112
    },
    'audio': {
        'lora_B_matrices': 0,
        'with_gradient': 0
    }
}
```

**Result: PASS**

All 112 decoder LoRA B matrices checked received gradients, while no audio-encoder LoRA modules were targeted.

### Training results

Both optimizer steps completed successfully.

| Metric | Result |
|---|---:|
| Step 1 loss | 2.0623 |
| Step 2 loss | 0.0662 |
| Reported average training loss | 1.0643 |
| Training runtime | 4.57 seconds |
| Completed steps | 2/2 |

The reported loss reduction is not evidence of generalization or improved ASR accuracy. Two steps are sufficient only for a basic training integration check.

### Training configuration verification

The generated `training_config.json` confirmed:

```text
sampling = e3
lora_config = L1
category_stratified = False
category_config = configs/category_sampling.json
category_first_config = {...}
load_in_4bit = 1
gradient_checkpointing = 1
```

`category_stratified=False` is expected: it refers to the legacy speaker-first category-stratification option, not the new category-first sampler.

The presence of `category_first_config` confirms that the new configuration was loaded and recorded by the training pipeline.

Together with the successful training run, this supports the conclusion that the category-first E3 integration is operational.

### Generated artifacts

The smoke test successfully generated:

```text
runs/category_e3_l1_smoke_2step/
├── adapter-final/
├── checkpoint-2/
├── gpu_usage.json
├── gradient_check.json
├── log_history.json
└── training_config.json
```

The working tree remained clean after training, consistent with the generated run artifacts being excluded from version control.

**Do not commit training outputs or checkpoints.** They may contain private dataset paths or other experiment-specific information.

### Warnings observed

Several non-fatal warnings appeared during training:

- `triton not found`: FLOP counting for Triton kernels was unavailable.
- `temperature` generation flag: a generation parameter may be ignored.
- `use_cache=True` was automatically disabled because gradient checkpointing was enabled.
- Tokenizer PAD/EOS token IDs were aligned with the model configuration.

None prevented the two-step smoke test from completing. Token configuration should be reviewed before full-scale training and inference evaluation.

### Smoke-test conclusion

**Status: PASS**

The two-step GPU smoke test successfully demonstrated that:

- Qwen3-ASR loads under the available GPU environment.
- Decoder-only LoRA adapters are injected correctly.
- The real training manifest can be processed.
- Category-first E3 configuration is accepted by the training pipeline.
- Gradient flow reaches the intended decoder adapters.
- The training loop completes optimizer steps.
- Final adapter and checkpoint artifacts are saved.

This establishes basic end-to-end training compatibility for category-first E3 + L1.

It does not establish training convergence, improved WER/CER, or robustness over longer runs.

## Remaining work

### 1. Investigate manifest differences

The original Qwen training manifest contains 87,964 recordings, while the category-enabled training manifest contains 87,875 recordings.

The difference of **89 recordings** has not yet been explained.

Before conducting formal comparisons, investigate whether these recordings were excluded because of missing metadata, preprocessing failures, filtering, or other causes.

### 2. Longer training validation

Run a longer smoke test (e.g., 10 optimizer steps) to verify continued training stability, loss logging, and checkpoint behavior.

The two-step test has already passed, so this is a follow-up validation rather than a prerequisite for establishing basic integration.

### 3. Unified category sampling across experiments

Currently, category-first sampling is available specifically for E3.

If the experimental design requires identical category proportions across E1–E4, extend the unified sampling framework so that category selection is a shared factor while each sampling strategy retains its intended within-category behavior.

Any such extension should explicitly document how it changes the definitions of E1–E4.

### 4. Full fine-tuning and evaluation

After finalizing the experimental design:

- Train the selected LoRA configurations using consistent hyperparameters.
- Evaluate WER and CER on held-out data.
- Report overall and per-category performance.
- Compare hard-speaker and non-hard-speaker performance.
- Analyze whether category-first sampling improves recognition for underrepresented speech categories or difficult speakers.

The held-out test set should remain reserved for final evaluation.

### 5. Team integration

Before merging into the team's unified training branch:

- Compare changes against the latest team pipeline.
- Confirm compatibility with teammate updates.
- Run the complete unit-test suite.
- Verify that existing sampling strategies retain their intended behavior.
- Submit the category-first implementation and documentation through a Pull Request.

## Current status

| Milestone | Status |
|---|---|
| Category-first sampler implementation | Complete |
| Unit tests | Passed (68 passed, 1 skipped in prior run) |
| Real-data probability validation | Passed |
| 100,000-draw sampling simulation | Passed |
| GPU model loading and LoRA injection | Passed |
| Gradient-flow verification | Passed |
| 2-step GPU smoke test | Passed |
| Adapter and checkpoint saving | Passed |
| 10-step training test | Pending |
| Full fine-tuning | Pending |
| WER/CER evaluation | Pending |
| Unified category sampling across E1–E4 | Pending |

**As of November 23, 2026, category-first E3 + L1 has passed CPU sampling validation and a two-step GPU training integration smoke test. Model-quality evaluation and full experimental comparisons remain future work.**