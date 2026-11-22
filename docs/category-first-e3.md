# Category-first E3 sampling

Category-first E3 controls the training mix by sampling in this order:

**Category → hard/non-hard group → speaker → recording**

The default config assigns 50% to Digital Assistant Commands, 30% to Novel
Sentences, and 20% to Spontaneous Speech Prompts. Non-spontaneous Speech Prompts
and any other category absent from the config receive zero probability. Within
each configured category, hard and non-hard speakers each receive half of that
category's probability. Speakers are uniform within category and difficulty
group; recordings are uniform within speaker and category. Hard labels come
from the existing `hard_top25` column.

This option was added to hold the speech-category mix fixed while comparing
hard-speaker sampling. Original E3 balances hard and non-hard speakers overall
without fixing category proportions. The older `--category-stratified` option
rebalances categories within each speaker and does not impose category-first
50/30/20 targets. All three remain separate strategies.

## Configure and run

Edit `configs/category_sampling.json` to change the positive category
proportions; they must sum to 1. `hard_group_probability` sets the hard share
within every category (0.5 by default). Each configured category must contain
at least one hard and one non-hard speaker in the training manifest.

Prepare manifests with `prepare_qwen_finetuning.py`, which writes `speaker_id`
and `category` alongside Qwen's `audio` and `text` fields. The trainer ignores
these metadata fields when building model inputs. Then opt in through
the experiment launcher:

```sh
python run_experiment.py --sampling e3 --lora L1 --category-config configs/category_sampling.json --stages train
```

The launcher forwards the option to `train_qwen_lora.py`. Existing experiments
without this option retain their prior sampling behavior. Direct trainer usage
is also supported with `--sampling e3 --weights-csv PATH --category-config PATH`.

## Validation

Run the category-first tests and the complete unit suite from the repository
root:

```sh
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s tests -p 'test_category_first_sampler.py' -v
```

Run category-first sampling on a real manifest from Python without loading a
model or GPU:

```python
from category_first_sampler import load_category_config, simulate_category_first_sampling
from hard_speaker_sampler import load_manifest, load_speaker_weights

rows = load_manifest("TRAIN.jsonl")
speakers = load_speaker_weights("WEIGHTS.csv", strategy="e3")
config = load_category_config("configs/category_sampling.json")
categories, hard_groups = simulate_category_first_sampling(
    rows, speakers, config, draws=100_000, seed=42
)
```

The returned counters report category totals and per-category hard/non-hard
counts. Unit tests use synthetic rows and verify exact theoretical marginals,
50/50 group masses, excluded rows, and deterministic configuration parsing.

The probability checks establish the distribution mathematically, not model
quality. No GPU training or WER/CER evaluation is claimed as completed by this
feature. Validate against the private real training manifest before using it
for a research run.
