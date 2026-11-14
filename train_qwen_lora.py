#!/usr/bin/env python3
"""Memory-conscious LoRA/QLoRA pilot training for Qwen3-ASR.

This is an experimental, project-owned alternative to Qwen's official
full-parameter trainer. It follows the official audio/text preprocessing and
label masking, freezes the base model, and trains PEFT LoRA adapters. Defaults
target a single 12 GB NVIDIA GPU: 4-bit base weights, batch size 1, gradient
accumulation, gradient checkpointing, and a paged 8-bit optimizer.

Use a dedicated environment. The qwen-asr package currently pins a different
Transformers line than the native ``-hf`` baseline used elsewhere in this repo.
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


QWEN_ASR_SFT_SOURCE = (
    "https://github.com/QwenLM/Qwen3-ASR/blob/main/finetuning/qwen3_asr_sft.py"
)
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-acc", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="positive value overrides epochs; use 2 for an initial smoke test")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--eval-steps", type=int, default=0,
                        help="0 disables in-training eval; checkpoint WER should be measured separately")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="0 is safest on Windows and avoids duplicate worker memory")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="all-linear",
                        help="PEFT target modules; 'all-linear' adapts linear audio/text layers")
    parser.add_argument("--load-in-4bit", type=int, choices=(0, 1), default=1)
    parser.add_argument("--gradient-checkpointing", type=int, choices=(0, 1), default=1)
    parser.add_argument("--optimizer", default="paged_adamw_8bit")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--resume-latest", action="store_true")
    return parser.parse_args(argv)


def validate_jsonl_schema(path):
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if set(("audio", "text")) - row.keys():
                raise ValueError(f"{path}:{line_number}: expected nonempty audio and text fields")
            if not str(row["audio"]).strip() or not str(row["text"]).strip():
                raise ValueError(f"{path}:{line_number}: expected nonempty audio and text fields")
            count += 1
    if not count:
        raise ValueError(f"{path}: no training rows")
    return count


def find_latest_checkpoint(output_dir):
    if not output_dir.is_dir():
        return None
    checkpoints = []
    for child in output_dir.iterdir():
        match = CHECKPOINT_RE.fullmatch(child.name)
        if child.is_dir() and match:
            checkpoints.append((int(match.group(1)), child))
    return max(checkpoints, default=(None, None))[1]


def patch_outer_forward(model):
    """Expose the official model's thinker loss to Transformers Trainer."""
    cls = model.__class__
    if getattr(cls, "_pd_lora_forward_patched", False):
        return
    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError("Qwen3-ASR model has no thinker.forward; incompatible qwen-asr version")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._pd_lora_forward_patched = True


def build_prefix_messages(prompt, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn(processor):
    def preprocess(example):
        prompt = example.get("prompt", "")
        prefix_text = processor.apply_chat_template(
            [build_prefix_messages(prompt, None)],
            add_generation_prompt=True,
            tokenize=False,
        )[0]
        return {
            "prompt": prompt,
            "audio": example["audio"],
            "target": example["text"],
            "prefix_text": prefix_text,
        }

    return preprocess


@dataclass
class QwenASRCollator:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features):
        import librosa

        audio_paths = [feature["audio"] for feature in features]
        prefix_texts = [feature["prefix_text"] for feature in features]
        targets = [feature["target"] for feature in features]
        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [prefix + target + eos for prefix, target in zip(prefix_texts, targets)]
        audios = [librosa.load(path, sr=self.sampling_rate, mono=True)[0] for path in audio_paths]
        full_inputs = self.processor(
            text=full_texts, audio=audios, return_tensors="pt", padding=True, truncation=False
        )
        prefix_inputs = self.processor(
            text=prefix_texts, audio=audios, return_tensors="pt", padding=True, truncation=False
        )
        prefix_lengths = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for index, prefix_length in enumerate(prefix_lengths):
            labels[index, :prefix_length] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        full_inputs["labels"] = labels
        return full_inputs


def package_versions(packages):
    from importlib.metadata import PackageNotFoundError, version

    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def main(argv=None):
    args = parse_args(argv)
    for path in (args.train_file, args.eval_file):
        if path is not None and not path.is_file():
            raise SystemExit(f"Manifest does not exist: {path}")
    if args.batch_size <= 0 or args.grad_acc <= 0 or args.save_steps <= 0:
        raise SystemExit("batch size, gradient accumulation and save steps must be positive")
    train_rows = validate_jsonl_schema(args.train_file)
    eval_rows = validate_jsonl_schema(args.eval_file) if args.eval_file else 0

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from qwen_asr import Qwen3ASRModel
        from transformers import BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise SystemExit(
            f"Missing training dependency: {exc}. Install qwen-asr, datasets, peft, "
            "bitsandbytes, librosa and accelerate in a dedicated environment."
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for Qwen3-ASR LoRA training")
    set_seed(args.seed)
    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    load_kwargs = {
        "dtype": compute_dtype,
        "device_map": {"": 0},
        "max_inference_batch_size": 1,
    }
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    wrapper = Qwen3ASRModel.from_pretrained(args.model_path, **load_kwargs)
    model = wrapper.model
    processor = wrapper.processor
    patch_outer_forward(model)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=bool(args.gradient_checkpointing)
        )
    elif args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    target_modules = args.target_modules
    if "," in target_modules:
        target_modules = [name.strip() for name in target_modules.split(",") if name.strip()]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    files = {"train": str(args.train_file)}
    if args.eval_file:
        files["validation"] = str(args.eval_file)
    dataset = load_dataset("json", data_files=files)
    dataset = dataset.map(make_preprocess_fn(processor), num_proc=1)
    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in dataset:
        drop = [column for column in dataset[split].column_names if column not in keep]
        if drop:
            dataset[split] = dataset[split].remove_columns(drop)

    evaluate_during_training = bool(args.eval_file and args.eval_steps > 0)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        logging_steps=args.log_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if evaluate_during_training else "no",
        eval_steps=args.eval_steps if evaluate_during_training else None,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        bf16=use_bf16,
        fp16=not use_bf16,
        optim=args.optimizer,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )

    class FloatInputTrainer(Trainer):
        def _prepare_inputs(self, inputs):
            inputs = super()._prepare_inputs(inputs)
            for key, value in list(inputs.items()):
                if torch.is_tensor(value) and value.is_floating_point():
                    inputs[key] = value.to(dtype=compute_dtype)
            return inputs

    trainer = FloatInputTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation") if evaluate_during_training else None,
        data_collator=QwenASRCollator(processor=processor, sampling_rate=args.sampling_rate),
        processing_class=processor,
    )
    # The upstream patched forward accepts **kwargs, which can make Trainer
    # incorrectly assume the model normalizes loss for gradient accumulation.
    trainer.model_accepts_loss_kwargs = False

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_record = {
        "purpose": "experimental Qwen3-ASR LoRA/QLoRA pilot",
        "upstream_preprocessing_source": QWEN_ASR_SFT_SOURCE,
        "model_path": args.model_path,
        "train_file": str(args.train_file.resolve()),
        "eval_file": str(args.eval_file.resolve()) if args.eval_file else None,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "python": sys.version,
        "packages": package_versions(
            ["torch", "transformers", "qwen-asr", "datasets", "peft", "bitsandbytes"]
        ),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "warning": "Adapter checkpoints and logs may reveal private source paths; do not commit output_dir.",
    }
    (args.output_dir / "training_config.json").write_text(
        json.dumps(run_record, indent=2) + "\n", encoding="utf-8"
    )

    resume = args.resume_from
    if args.resume_latest and not resume:
        resume = find_latest_checkpoint(args.output_dir)
    print(f"Train rows: {train_rows:,}; eval rows: {eval_rows:,}")
    print(f"GPU: {run_record['gpu']}; 4-bit: {bool(args.load_in_4bit)}")
    print(f"Resume: {resume or 'no'}")
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_model(str(args.output_dir / "adapter-final"))
    processor.save_pretrained(str(args.output_dir / "adapter-final"))


if __name__ == "__main__":
    main()
