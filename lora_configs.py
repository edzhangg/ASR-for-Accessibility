"""Shared LoRA configurations for the 2x2 experiment (L1 / L2).

L1: decoder-only   -> attention projections (q/k/v/o) of the text decoder.
L2: audio + decoder -> L1 targets PLUS attention projections of the audio encoder.

L2 is a strict superset of L1 so the only difference between them is the
audio-side adapters. Rank / alpha / dropout are shared constants, not per-config.

Target modules are discovered from the loaded model's module names (not hardcoded
paths) so the same code works whichever Qwen3-ASR wrapper is used, and the
resulting list is asserted non-empty and printed for the run log.
"""

import re

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

DECODER_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
AUDIO_PROJ = ("q_proj", "k_proj", "v_proj", "out_proj")
AUDIO_MARKER = "audio_tower"
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def select_targets(linear_names, config, audio_layers=None):
    """Pick target module names from an iterable of Linear-module names.

    Pure function (no torch needed). `audio_layers` optionally restricts L2 to a
    set of audio-encoder layer indices (None = all layers).
    Returns (decoder_targets, audio_targets) as sorted lists.
    """
    if config not in ("L1", "L2"):
        raise ValueError(f"Unknown LoRA config {config!r}; expected 'L1' or 'L2'")

    decoder, audio = [], []
    for name in linear_names:
        if AUDIO_MARKER in name:
            if config == "L2" and ".self_attn." in name and name.rsplit(".", 1)[-1] in AUDIO_PROJ:
                m = LAYER_RE.search(name)
                if audio_layers is None or (m and int(m.group(1)) in audio_layers):
                    audio.append(name)
        elif ".self_attn." in name and name.rsplit(".", 1)[-1] in DECODER_PROJ:
            decoder.append(name)

    if not decoder:
        raise RuntimeError("No decoder attention projections found; module names unexpected")
    if config == "L2" and not audio:
        raise RuntimeError("L2 requested but no audio-encoder attention projections found")
    return sorted(decoder), sorted(audio)


def check_trainable(trainable_names, config):
    """Assert only LoRA params train, and audio params train iff config == 'L2'."""
    if not trainable_names:
        raise RuntimeError("No trainable parameters")
    bad = [n for n in trainable_names if "lora_" not in n]
    if bad:
        raise RuntimeError(f"Non-LoRA trainable parameters: {bad[:5]}")
    has_audio = any(AUDIO_MARKER in n for n in trainable_names)
    if config == "L1" and has_audio:
        raise RuntimeError("L1 must not train audio-encoder parameters")
    if config == "L2" and not has_audio:
        raise RuntimeError("L2 must train audio-encoder parameters")


def build_lora(model, config, audio_layers=None, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT):
    """Freeze the base model and attach LoRA adapters for `config` ('L1' | 'L2').

    Returns (peft_model, info) where info is JSON-serializable for the run record.
    """
    import torch
    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad = False

    linear_names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    decoder, audio = select_targets(linear_names, config, audio_layers)
    targets = decoder + audio

    peft_model = get_peft_model(
        model,
        LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets, bias="none"),
    )
    check_trainable([n for n, p in peft_model.named_parameters() if p.requires_grad], config)

    n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in peft_model.parameters())
    info = {
        "lora_config": config,
        "r": r, "alpha": alpha, "dropout": dropout,
        "n_decoder_targets": len(decoder), "n_audio_targets": len(audio),
        "audio_layers": sorted(audio_layers) if audio_layers is not None else "all",
        "example_targets": targets[:3] + targets[-3:],
        "trainable_params": n_train, "total_params": n_total,
    }
    print(f"[{config}] {len(decoder)} decoder + {len(audio)} audio target modules; "
          f"trainable {n_train:,} / {n_total:,} ({100 * n_train / n_total:.3f}%)")
    return peft_model, info
