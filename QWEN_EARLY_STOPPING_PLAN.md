# Proposed Qwen early-stopping plan

Date: 2026-11-21

Status: saved for review only. This proposal has not been implemented. Training code, shared configuration, and the existing grid protocol are unchanged.

## Objective

Reduce GPU time for E4_L1, E4_L2, E3_L1, and E3_L2 by stopping when validation transcription quality plateaus, while retaining a 5,000-optimizer-step ceiling. Stopping before 5,000 steps is possible but not guaranteed.

Keep the proposed 120-second training-duration ceiling and all 221 training speakers eligible. Continue using the existing E3/E4 sampling strategies. Reducing the training budget controls the number of weighted draws without first removing longer speech from the eligible pool.

## Proposed starting policy

| Setting | Proposal |
|---|---|
| Monitoring set | Approximately 1,000 fixed recordings from validation, covering all 25 validation speakers and a mix of durations and categories |
| Selection | Deterministic, documented seed; identical recording IDs across checkpoints and all four arms |
| Metric | Mean per-speaker corpus WER, calculated with the established text normalizer |
| Evaluation interval | Every 500 optimizer steps |
| Minimum training | No stopping before 1,000 steps |
| Meaningful improvement | At least 0.2 absolute WER percentage points below the last qualifying best score |
| Patience | Three consecutive evaluations without meaningful improvement |
| Maximum training | 5,000 optimizer steps |
| Selected checkpoint | Lowest monitored WER among evaluated checkpoints, including improvements smaller than the patience threshold |
| Final validation | Restore the selected checkpoint and evaluate on the full validation split |

These values are initial heuristics for review, not measured optimal settings. Fix the monitoring subset before training; do not repeatedly select a subset based on observed results. Keep the test split untouched during stopping and checkpoint selection.

## Metric and stopping semantics

Calculate each speaker's corpus WER as total word errors divided by total reference words across that speaker's monitoring recordings, then average those 25 speaker WERs. Do not average individual utterance WERs. Use a fixed policy for empty normalized references and report prediction failures and coverage; failures must not make a checkpoint appear better by silently shrinking its evaluation set.

Track two quantities separately:

- The qualifying best score used to reset patience. An improvement of at least 0.2 percentage points resets patience and updates this score.
- The absolute lowest monitored WER used to select the checkpoint to restore. A smaller improvement can update this checkpoint without resetting patience.

For clarity, 10.0% to 9.8% WER is an improvement of 0.2 percentage points. If code stores WER as a fraction, the corresponding threshold is 0.002, not 0.2.

Suggested minimum-training behavior: record the 500-step evaluation, but start the patience counter at step 1,000. At step 1,000, initialize the qualifying best from the lowest score observed so far. Count subsequent non-improving checks toward patience.

Example: if the best qualifying score is reached at step 1,000, and steps 1,500, 2,000, and 2,500 do not improve sufficiently, stop at 2,500 and restore the lowest-WER checkpoint observed. A step-zero baseline is useful context but should not substitute for the initial adapter checkpoint when initializing patience.

## Current implementation gap

The inspected `train_qwen_lora.py` disables in-training evaluation by default. Enabling `--eval-steps` alone would evaluate loss, not generated-transcript WER. It currently has no early-stopping callback, generated-WER monitoring path, or best-checkpoint restoration policy. The experiment runner currently evaluates the adapter after training.

Implementation would therefore need to:

1. Create and record the fixed monitoring manifest from validation.
2. Add periodic transcription and scoring for the current adapter using consistent inference settings and normalization.
3. Implement the minimum-step, threshold, and patience rules explicitly, with restart-safe state.
4. Coordinate checkpoint saving and evaluation, preferably at the same 500-step interval.
5. Preserve the selected best checkpoint despite checkpoint-retention limits, restore it at training end, and save it as the final adapter used by the runner.
6. Record monitoring scores, coverage, stopping reason, selected step, total training steps, and stage timings.
7. Verify checkpoint selection, resume behavior, and the stopping rule before full grid execution.

Hugging Face documents metric-based early stopping through `EarlyStoppingCallback`, but the exact policy above may require custom state handling, particularly for minimum training and the distinction between patience-reset and absolute-best scores:

https://huggingface.co/docs/transformers/v4.57.0/en/main_classes/callback

## Comparison and time accounting

Apply the same stopping policy and monitoring data to all four arms. Actual stopping steps can differ. Report full-validation WER/CER alongside selected checkpoint step, executed optimizer steps, sampled draws, and GPU hours.

This evaluates the four configurations under a shared stopping policy. It changes the original fixed-budget comparison; adopt and document that protocol change before running it.

At batch size 1 and gradient accumulation 8, 5,000 steps correspond to approximately 40,000 weighted draws per arm. Sampling is with replacement, so these are not necessarily distinct recordings. Training time should decrease roughly with executed steps, but periodic inference and checkpoint saves add overhead. Measure the first monitoring pass and the first L1/L2 step rates before promising a revised completion time.

Retain a full validation pass for the selected checkpoint of each arm. Avoid running the entire validation set at every 500-step check: the previous full adapter validation took approximately two hours. Monitoring-subset timing remains unmeasured, and long recordings can make its cost differ from a simple recording-count ratio.

## Items to confirm before implementation

- Whether to adopt validation-based stopping instead of the existing fixed-budget grid.
- Monitoring subset size and selection method, including representation of longer speech.
- Mean per-speaker WER as the stopping metric, even if the final report also emphasizes corpus WER.
- The proposed 500-step interval, 1,000-step minimum, 0.2-percentage-point threshold, and patience of three.
- A learning-rate schedule appropriate to the declared maximum budget; stopping early ends that schedule before its planned completion.

No runs have been launched or modified as part of saving this proposal.
