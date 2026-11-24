#!/usr/bin/env python3
"""Model-agnostic ASR LoRA training entry point.

This file owns ONLY the shared training workflow:
  1. Read a speaker-disjoint training manifest (keep original row order).
  2. Build a weighted sampler from sampling.py.
  3. Ask a model backend (e.g. qwen_backend.py) to load/collate/adapt its model.
  4. Run Hugging Face Trainer, check gradients, and save private artifacts.

Model-specific audio processing, model loading, and LoRA target names belong
in <model>_backend.py, NOT here. All restricted data stays on the local machine.

Important limitations of this first integration:
- hard_threshold is a teammate TODO in sampling.py.
- exact sampler-RNG restoration on resume is NOT implemented; use fresh runs.
- WER evaluation is handled by the team's existing evaluator, not this trainer.
- not GPU-verified; run a two-step local smoke test before a full training run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from importlib.metadata import PackageNotFoundError, version
import math
from pathlib import Path
from typing import Any, Mapping


# To add another model later: implement the same create_backend(args) contract,
# then register its module here. The shared training logic below stays unchanged.
BACKEND_MODULES = {"qwen": "qwen_backend"}


class ManifestDataset:
    """Minimal map-style dataset; indices exactly match the JSONL row order."""

    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def read_manifest(path: Path, *, split: str = "train") -> list[dict[str, Any]]:
    """Validate and load a private JSONL manifest, without touching audio.

    Training rows must contain audio, text, speaker_id and category.
    Category sampling is mandatory for every experiment arm.
    """
    rows: list[dict[str, Any]] = []
    seen_audio: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            if not all(isinstance(row.get(key), str) and row[key].strip()
                       for key in ("audio", "text", "speaker_id", "category")):
                raise ValueError(f"{path}:{line_no}: missing audio/text/speaker_id/category")
            if row.get("split", split) != split:
                raise ValueError(f"{path}:{line_no}: expected split={split!r}")
            audio_key = row["audio"].replace("\\", "/").casefold()
            if audio_key in seen_audio:
                raise ValueError(f"{path}:{line_no}: duplicate audio path")
            seen_audio.add(audio_key)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no eligible rows")
    return rows


def check_val_speaker_separation(train_rows: list[Mapping[str, Any]],
                                 val_manifest: Path | None) -> None:
    """Optional preflight check only; validation WER is scored elsewhere.

    The team's data-preparation script remains responsible for the full
    Train/Validation/Test disjointness check.
    """
    if val_manifest is None:
        return
    val_speakers: set[str] = set()
    with val_manifest.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("speaker_id"):
                raise ValueError(f"{val_manifest}:{line_no}: missing speaker_id")
            val_speakers.add(str(row["speaker_id"]))
    overlap = {r["speaker_id"] for r in train_rows} & val_speakers
    if overlap:
        raise ValueError(f"Train/validation speaker overlap: {len(overlap)} speakers")


def sha256_file(path: Path) -> str:
    """Fingerprint private input bytes; hash only, never copy contents to logs."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_version(package: str) -> str | None:
    """Record package versions without requiring optional packages for tests."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def parse_category_probs(value: str) -> dict[str, float]:
    """Read a REQUIRED category policy, inline or from a local JSON file."""
    # Inline JSON containing quotes/colons is not a valid Windows filename.
    data = json.loads(value if value.lstrip().startswith("{") else Path(value).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("--category-probs must be a JSON object")
    return {str(k): float(v) for k, v in data.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=tuple(BACKEND_MODULES), default="qwen")
    parser.add_argument("--model-id", default=None,
                        help="pretrained model identifier; backend chooses its default when omitted")
    parser.add_argument("--train-manifest", type=Path, required=True,
                        help="private Qwen-format JSONL in Dataset index order")
    parser.add_argument("--val-manifest", type=Path,
                        help="optional speaker-overlap check; not in-training WER evaluation")
    parser.add_argument("--speaker-wer-csv", type=Path, required=True,
                        help="TRAIN-only speaker zero-shot difficulty_wer_percent CSV")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="must be a new / empty private output folder")

    # All experiment arms use the same sampler API. Only strategy differs.
    parser.add_argument("--sampling", choices=("continuous", "hard_threshold"), required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--max-wer", type=float, default=0.30)
    parser.add_argument("--category-probs", required=True,
                        help="REQUIRED shared category distribution: a local JSON file path or JSON object")
    parser.add_argument("--seed", type=int, default=42)

    # A model backend can map these common names to its own adapter targets.
    parser.add_argument("--lora", choices=("decoder_only", "encoder_decoder"), required=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--audio-layers", default="",
                        help="Qwen backend only: optional comma-separated audio layer indices")
    parser.add_argument("--load-in-4bit", type=int, choices=(0, 1), default=1)
    parser.add_argument("--gradient-checkpointing", type=int, choices=(0, 1), default=1)
    parser.add_argument("--sampling-rate", type=int, default=16000)

    # Safe default is a TWO optimizer-step smoke test, not an accidental 90k-row job.
    parser.add_argument("--max-steps", type=int, default=2,
                        help="2 by default for smoke testing; -1 uses --epochs")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-acc", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--optimizer", default="paged_adamw_8bit")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args(argv)
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be -1 or a positive integer")
    if any(x <= 0 for x in (args.batch_size, args.grad_acc, args.save_steps,
                            args.log_steps, args.save_total_limit, args.lora_r)):
        parser.error("batch/grad-acc/save/log/LoRA-r options must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be finite and positive")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("warmup ratio must be in [0,1)")
    if not 0 <= args.lora_dropout < 1:
        parser.error("LoRA dropout must be in [0,1)")
    if args.lora == "decoder_only" and args.audio_layers:
        parser.error("--audio-layers only applies to encoder_decoder")
    return args


def create_backend(args: argparse.Namespace):
    """Instantiate model-specific code without putting it in the shared Trainer."""
    module = importlib.import_module(BACKEND_MODULES[args.backend])
    return module.create_backend(args)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for path in (args.train_manifest, args.val_manifest, args.speaker_wer_csv):
        if path is not None and not path.is_file():
            raise SystemExit(f"Missing input file: {path}")
    # No force/reuse/resume yet: protects experimental identity and sampling RNG.
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise SystemExit(
            f"Output directory is not empty: {args.output_dir}. "
            "Use a NEW directory; overwrite/resume is intentionally unsupported."
        )

    from sampling import (
        build_sampler,
        load_speaker_wer_csv,
        load_hard_labels_csv,
        summarize_sampling,
    )
    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for this QLoRA training implementation")

    train_rows = read_manifest(args.train_manifest)
    check_val_speaker_separation(train_rows, args.val_manifest)
    speaker_wer = load_speaker_wer_csv(args.speaker_wer_csv)
    hard_labels = (load_hard_labels_csv(args.speaker_wer_csv) if args.sampling == "hard_threshold" else None)
    category_probs = parse_category_probs(args.category_probs)
    set_seed(args.seed)
    sampler = build_sampler(
        train_rows, speaker_wer,
        strategy=args.sampling, seed=args.seed,
        alpha=args.alpha, epsilon=args.epsilon, max_wer=args.max_wer,
        category_probs=category_probs, num_samples=len(train_rows),
        hard_labels=hard_labels
    )

    # Report expected sampling distribution
    sampling_summary = summarize_sampling(
        train_rows,
        sampler.weights.tolist(),
        hard_labels=hard_labels,
    )

    print("\n--- Sampling Diagnostics ---")
    print(f"Strategy: {args.sampling}")
    print(f"Recordings: {sampling_summary['num_recordings']}")
    print(f"Speakers: {sampling_summary['num_speakers']}")

    print("Expected category distribution:")
    for category, probability in sampling_summary["category_probability"].items():
        print(f"  {category}: {probability:.2%}")

    if hard_labels is not None:
        print(f"Overall hard probability: {sampling_summary['hard_probability']:.2%}")

        print("Hard probability within each category:")
        for category, probability in sampling_summary["hard_probability_by_category"].items():
            print(f"  {category}: {probability:.2%}")

    speaker_probs = sampling_summary["speaker_probability"].values()

    print(f"Minimum speaker probability: {min(speaker_probs):.4%}")
    print(f"Maximum speaker probability: {max(speaker_probs):.4%}")
    print("----------------------------\n")

    # CRITICAL: ManifestDataset and sampler are built from THE SAME list of rows.
    dataset = ManifestDataset(train_rows)

    backend = create_backend(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_record = {
        "backend": args.backend, "model_id": backend.model_id,
        "sampling": args.sampling, "lora": args.lora,
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_records": len(train_rows),
        "train_speakers": len({row["speaker_id"] for row in train_rows}),
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "speaker_wer_csv_sha256": sha256_file(args.speaker_wer_csv),
        "backend_info": backend.info,
        "packages": {name: installed_version(name) for name in (
            "torch", "transformers", "qwen-asr", "peft", "bitsandbytes")},
        "code_sha256": {name: sha256_file(Path(__file__).parent / name) for name in (
            "train.py", "sampling.py", "qwen_backend.py")
            if (Path(__file__).parent / name).is_file()},
        "warning": "Private paths/configs and adapters: do not commit output_dir to git.",
    }
    (args.output_dir / "training_config.json").write_text(
        json.dumps(private_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Shared HF optimization loop; backend owns only model-specific details.
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        logging_steps=args.log_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=backend.compute_dtype == torch.bfloat16,
        fp16=backend.compute_dtype == torch.float16,
        optim=args.optimizer,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        eval_strategy="no",  # use the team's shared WER evaluator afterwards
    )

    class WeightedASRTrainer(Trainer):
        """Generic Trainer extension: weighted indexing, tensor precision, grad gate."""

        _did_check_gradients = False

        def _get_train_sampler(self, *a, **kw):
            return sampler

        def _prepare_inputs(self, inputs):
            inputs = super()._prepare_inputs(inputs)
            for key, value in inputs.items():
                if torch.is_tensor(value) and value.is_floating_point():
                    inputs[key] = value.to(dtype=backend.compute_dtype)
            return inputs

        def training_step(self, model, inputs, *a, **kw):
            loss = super().training_step(model, inputs, *a, **kw)
            if not self._did_check_gradients:
                # Backend reports gradient coverage of trainable adapter groups.
                report = backend.check_first_gradients(model, args.lora)
                (args.output_dir / "gradient_check.json").write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
                self._did_check_gradients = True
            return loss

    trainer = WeightedASRTrainer(
        model=backend.model,
        args=training_args,
        train_dataset=dataset,
        data_collator=backend.collate,
        processing_class=backend.processor,
    )
    trainer.model_accepts_loss_kwargs = False

    torch.cuda.reset_peak_memory_stats()
    trainer.train()

    adapter_dir = args.output_dir / "adapter-final"
    trainer.save_model(str(adapter_dir))
    backend.save_processor(adapter_dir)
    (args.output_dir / "gpu_usage.json").write_text(json.dumps({
        "peak_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "peak_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
    }, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Training complete. Private adapter saved to: {adapter_dir}")
    print("Next: evaluate this adapter with the TEAM's shared validation evaluator.")


if __name__ == "__main__":
    main()
