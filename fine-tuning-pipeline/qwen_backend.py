#!/usr/bin/env python3
"""Qwen3-ASR-specific backend for the model-agnostic train.py.

Adapted from the team's train_qwen_lora.py and lora_configs.py. This file owns:
  * Qwen3-ASR-1.7B loading (optional 4-bit NF4 quantization)
  * official-style audio/text collation + prompt/padding label masking
  * LoRA targets for decoder_only vs encoder_decoder (L2)
  * Qwen outer-wrapper compatibility and first-backward gradient checks

Nothing here chooses speakers or sampling probabilities: that lives in sampling.py.
This backend is experimental and still requires a real Qwen/CUDA smoke test.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"

# Qwen's transformers-backed thinker has a text decoder at thinker.model.layers
# and audio encoder at thinker.audio_tower.layers. Fail loudly if architecture
# changes rather than silently adapting unrelated self-attention projections.
DECODER_RE = re.compile(
    r"(?:^|\.)thinker\.model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
)
AUDIO_RE = re.compile(
    r"(?:^|\.)thinker\.audio_tower\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|out_proj)$"
)


def select_lora_targets(linear_module_names: list[str], mode: str,
                        audio_layers: set[int] | None = None) -> tuple[list[str], list[str]]:
    """Pure selection logic, testable without Qwen, PEFT, torch or private data.

    `encoder_decoder` is a strict superset of `decoder_only`. Layer selection
    is intentionally narrow so other multimodal attention is never adapted.
    """
    if mode not in ("decoder_only", "encoder_decoder"):
        raise ValueError(f"Unknown LoRA mode: {mode}")
    decoder = sorted(name for name in linear_module_names if DECODER_RE.search(name))
    if not decoder:
        raise RuntimeError("No Qwen decoder attention modules matched: inspect model.named_modules()")
    audio = []
    if mode == "encoder_decoder":
        for name in linear_module_names:
            match = AUDIO_RE.search(name)
            if match and (audio_layers is None or int(match.group(1)) in audio_layers):
                audio.append(name)
        audio.sort()
        if not audio:
            raise RuntimeError("L2 requested, but no audio encoder attention modules matched")
        if audio_layers is not None:
            found = {int(AUDIO_RE.search(name).group(1)) for name in audio}
            if found != audio_layers:
                raise RuntimeError(f"Requested audio layers not matched: {sorted(audio_layers - found)}")
    return decoder, audio


def patch_outer_forward(model) -> None:
    """Forward the outer multimodal wrapper's training args to thinker.forward."""
    cls = type(model)
    if getattr(cls, "_pd_lora_forward_patched", False):
        return
    if not hasattr(model, "thinker"):
        raise RuntimeError("Expected Qwen ASR model.thinker; incompatible qwen-asr model")

    def forward(self, input_ids=None, attention_mask=None, input_features=None,
                feature_attention_mask=None, labels=None, **kwargs):
        return self.thinker.forward(
            input_ids=input_ids, attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels, **kwargs,
        )

    cls.forward = forward
    cls._pd_lora_forward_patched = True


def bridge_input_embeddings(model) -> None:
    """Expose inner thinker embeddings to PEFT's gradient preparation."""
    try:
        embeddings = model.get_input_embeddings()
    except (AttributeError, NotImplementedError):
        thinker = getattr(model, "thinker", None)
        if thinker is None or not hasattr(thinker, "get_input_embeddings"):
            raise RuntimeError("Qwen thinker has no get_input_embeddings()")
        model.get_input_embeddings = thinker.get_input_embeddings
        if hasattr(thinker, "set_input_embeddings"):
            model.set_input_embeddings = thinker.set_input_embeddings
        embeddings = model.get_input_embeddings()
    if embeddings is None:
        raise RuntimeError("Qwen returned no usable input embeddings")


def apply_qwen_lora(model, *, mode: str, audio_layers: set[int] | None,
                    r: int, alpha: int, dropout: float):
    """Reuse the team's LoRA settings, with safer module path filtering."""
    import torch
    from peft import LoraConfig, get_peft_model

    for parameter in model.parameters():
        parameter.requires_grad = False
    # bitsandbytes Linear4bit inherits nn.Linear; verify actual target counts.
    names = [name for name, layer in model.named_modules() if isinstance(layer, torch.nn.Linear)]
    decoder, audio = select_lora_targets(names, mode, audio_layers)
    model = get_peft_model(model, LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=decoder + audio, bias="none",
    ))
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Expected ONLY LoRA parameters to be trainable")
    audio_trainable = [n for n in trainable if "audio_tower" in n]
    decoder_trainable = [n for n in trainable if "thinker.model.layers" in n]
    if not decoder_trainable or bool(audio_trainable) != (mode == "encoder_decoder"):
        raise RuntimeError("LoRA trainable parameter groups do not match requested mode")
    if len(decoder_trainable) + len(audio_trainable) != len(trainable):
        raise RuntimeError("Unexpected trainable parameter outside Qwen decoder/audio tower")
    info = {
        "mode": mode, "r": r, "lora_alpha": alpha, "dropout": dropout,
        "decoder_modules": len(decoder), "audio_modules": len(audio),
        "audio_layers": sorted(audio_layers) if audio_layers else "all",
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "examples": (decoder[:2] + audio[:2]),
    }
    print(f"[{mode}] LoRA targets: {len(decoder)} decoder, {len(audio)} audio")
    return model, info


class QwenBackend:
    """Small contract consumed by train.py; another ASR backend can implement it."""

    def __init__(self, model, processor, compute_dtype, model_id: str, info: dict):
        self.model = model
        self.processor = processor
        self.compute_dtype = compute_dtype
        self.model_id = model_id
        self.info = info

    def collate(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Qwen official-style labels: only ASR target tokens contribute to loss."""
        import librosa

        sampling_rate = self.info["sampling_rate"]
        audio_arrays = [librosa.load(row["audio"], sr=sampling_rate, mono=True)[0] for row in rows]
        prompts = [row.get("prompt", "") for row in rows]
        prefix_texts = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": prompt or ""},
                {"role": "user", "content": [{"type": "audio", "audio": None}]},
            ]
            rendered = self.processor.apply_chat_template(
                [messages], add_generation_prompt=True, tokenize=False
            )
            prefix_texts.append(rendered[0] if isinstance(rendered, (list, tuple)) else rendered)
        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [prefix + row["text"] + eos for prefix, row in zip(prefix_texts, rows)]
        full = self.processor(
            text=full_texts, audio=audio_arrays, return_tensors="pt", padding=True, truncation=False
        )
        prefix = self.processor(
            text=prefix_texts, audio=audio_arrays, return_tensors="pt", padding=True, truncation=False
        )
        labels = full["input_ids"].clone()
        for i, length in enumerate(prefix["attention_mask"].sum(dim=1).tolist()):
            labels[i, :length] = -100
        pad = self.processor.tokenizer.pad_token_id
        if pad is not None:
            labels[labels == pad] = -100
        if not bool((labels != -100).any()):
            raise RuntimeError("Batch has no target tokens after prefix/pad masking")
        full["labels"] = labels
        return dict(full)

    def check_first_gradients(self, model, mode: str) -> dict[str, Any]:
        """Check first backward pass actually reaches encoder+decoder LoRA B."""
        groups: dict[str, list[bool]] = {"decoder": [], "audio": []}
        for name, p in model.named_parameters():
            if p.requires_grad and "lora_B" in name:
                group = "audio" if "audio_tower" in name else "decoder"
                groups[group].append(p.grad is not None and bool((p.grad != 0).any()))
        report = {key: {"total": len(flags), "nonzero": sum(flags)}
                  for key, flags in groups.items()}
        for key in (("decoder", "audio") if mode == "encoder_decoder" else ("decoder",)):
            flags = groups[key]
            if not flags or sum(flags) / len(flags) < 0.5:
                raise RuntimeError(f"{key} LoRA first-backward gradients missing: {report[key]}")
        if mode == "decoder_only" and groups["audio"]:
            raise RuntimeError("decoder_only unexpectedly has audio LoRA gradients")
        return report

    def save_processor(self, output_dir: Path) -> None:
        self.processor.save_pretrained(str(output_dir))


def create_backend(args) -> QwenBackend:
    """Load Qwen and its LoRA adapters. All heavy packages imported lazily."""
    import torch
    from peft import prepare_model_for_kbit_training
    from qwen_asr import Qwen3ASRModel
    from transformers import BitsAndBytesConfig

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model_id = args.model_id or DEFAULT_MODEL_ID
    kwargs = {"dtype": dtype, "device_map": {"": 0}, "max_inference_batch_size": 1}
    if quant_config:
        kwargs["quantization_config"] = quant_config
    wrapper = Qwen3ASRModel.from_pretrained(model_id, **kwargs)
    model = wrapper.model
    patch_outer_forward(model)
    bridge_input_embeddings(model)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=bool(args.gradient_checkpointing),
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    audio_layers = {int(x) for x in args.audio_layers.split(",") if x.strip()} or None
    if audio_layers is not None and any(layer < 0 for layer in audio_layers):
        raise ValueError("audio layer indices must be >= 0")
    model, lora_info = apply_qwen_lora(
        model, mode=args.lora, audio_layers=audio_layers,
        r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
    )
    info = {"sampling_rate": args.sampling_rate, "load_in_4bit": bool(args.load_in_4bit),
            "lora": lora_info}
    return QwenBackend(model, wrapper.processor, dtype, model_id, info)
