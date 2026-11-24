# Qwen 100-step grid results and status

Updated: 2026-11-23T23:39:37.293214+00:00

**State:** complete

Source: shared-finetuning `5df52d1329e64e47b3c83b640f7bd4cdc5ebe2c3`; isolated snapshot, existing checkout preserved.

Training: 87,964 recordings, 221 speakers; <=30 seconds; exactly 100 optimizer steps per adapter; batch 1, accumulation 8; approximately 800 weighted draws with replacement per arm.

Difficulty: verified scores from the original full training split, retained unchanged and redistributed over eligible recordings. No new training inference.

Validation: frozen category-balanced 1,000-row <=30s subset, seed 42, all 25 validation speakers; max_utts=0. This is a development subset, not a full-distribution validation estimate.

Category counts: {"Novel Sentences": 333, "Digital Assistant Commands": 334, "Spontaneous Speech Prompts": 333}.

| Model | Scored / selected | Overall WER % | CER % | Mean speaker WER % | Train minutes | Eval minutes |
|---|---:|---:|---:|---:|---:|---:|
| zero_shot | 1000 / 1000 | 8.7291 | 5.1644 | 12.5791 | 0.00 | 12.41 |
| E4_L1 | 1000 / 1000 | 8.3557 | 5.0047 | 11.765 | 6.03 | 13.60 |
| E4_L2 | 1000 / 1000 | 9.5694 | 5.9972 | 32.3963 | 7.92 | 13.82 |
| E3_L1 | 1000 / 1000 | 8.3965 | 5.0505 | 11.5951 | 5.81 | 14.23 |
| E3_L2 | 1000 / 1000 | 8.3499 | 5.0763 | 11.6388 | 8.18 | 14.34 |

| Model | Category | Scored | WER % | CER % |
|---|---|---:|---:|---:|
| zero_shot | Digital Assistant Commands | 334 | 12.7638 | 6.8276 |
| zero_shot | Novel Sentences | 333 | 7.4673 | 3.8065 |
| zero_shot | Spontaneous Speech Prompts | 333 | 8.4986 | 5.4136 |
| E4_L1 | Digital Assistant Commands | 334 | 8.8523 | 5.0197 |
| E4_L1 | Novel Sentences | 333 | 7.5149 | 4.0484 |
| E4_L1 | Spontaneous Speech Prompts | 333 | 8.5896 | 5.3948 |
| E4_L2 | Digital Assistant Commands | 334 | 8.8008 | 5.05 |
| E4_L2 | Novel Sentences | 333 | 12.176 | 7.7042 |
| E4_L2 | Spontaneous Speech Prompts | 333 | 8.7079 | 5.4717 |
| E3_L1 | Digital Assistant Commands | 334 | 9.5728 | 5.656 |
| E3_L1 | Novel Sentences | 333 | 7.61 | 4.0666 |
| E3_L1 | Spontaneous Speech Prompts | 333 | 8.4895 | 5.3423 |
| E3_L2 | Digital Assistant Commands | 334 | 8.8008 | 5.3227 |
| E3_L2 | Novel Sentences | 333 | 7.4673 | 3.9069 |
| E3_L2 | Spontaneous Speech Prompts | 333 | 8.6078 | 5.5111 |

## Timing and limitations

GPU stages run sequentially. Stage durations include subprocess startup and model loading. GPU availability checks are skipped for launches using --start-now.

Initial reservation: approximately 2–3.5 hours for four arms, plus approximately 15–25 minutes for the matching original-model baseline. Unmeasured L2 speed remains a source of uncertainty.

Coverage and exceptions: [{"zero_shot": {"unpredicted": 0, "failures": {}}}, {"E4_L1": {"unpredicted": 0, "failures": {}}}, {"E4_L2": {"unpredicted": 0, "failures": {}}}, {"E3_L1": {"unpredicted": 0, "failures": {}}}, {"E3_L2": {"unpredicted": 0, "failures": {}}}]

The original-model baseline uses the same qwen-asr backend, model, token limit, normalizer and frozen manifest as the adapters; older HF baseline metrics are not substituted.

Inputs and hashes: preparation.json. Source provenance: source_provenance.json. Effective configuration: experiment_config.json. Queue state and revised estimates: queue_status.json. Logs: execution.log and each model folder/logs/.

## First-arm timing update

{
  "measured_at_utc": "2026-11-23T22:22:53.351292+00:00",
  "train_seconds": 361.7,
  "eval_seconds": 816.36,
  "arm_wall_seconds_including_gpu_gates": 1178.2,
  "remaining_hours_low": 1.21,
  "remaining_hours_high": 1.64,
  "finish_utc_low_if_gpu_stays_available": "2026-11-23T23:35:23.891292+00:00",
  "finish_utc_high_if_gpu_stays_available": "2026-11-24T00:01:03.651292+00:00",
  "basis": "Three remaining adapter evaluations plus one baseline; L2 training provisionally 1\u20132x measured L1; overhead allowance in upper estimate."
}
