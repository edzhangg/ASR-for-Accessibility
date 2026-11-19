"""Decoder-only LoRA configuration for Qwen3-ASR-1.7B."""

from peft import LoraConfig, get_peft_model

def configure_decoder_lora(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
):
    """
    Freeze the pretrained Qwen3-ASR model and apply LoRA
    only to language-model attention projections.
    """

    # Freeze all pretrained parameters
    for param in model.parameters():
        param.requires_grad = False

    # Target only decoder attention projections
    target_modules = [
        f"model.language_model.layers.{i}.self_attn.{proj}"
        for i in range(len(model.model.language_model.layers))
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
    ]

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, config)

    # Verify that the audio encoder remains frozen
    trainable_audio = [
        name
        for name, param in model.named_parameters()
        if "audio_tower" in name and param.requires_grad
    ]

    if trainable_audio:
        raise RuntimeError(
            f"Audio encoder has trainable parameters: {trainable_audio[:5]}"
        )

    # Verify all trainable parameters belong to decoder LoRA
    trainable = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
    ]

    if not trainable:
        raise RuntimeError("No trainable LoRA parameters found")

    for name in trainable:
        if "language_model" not in name or "lora_" not in name:
            raise RuntimeError(
                f"Unexpected trainable parameter: {name}"
            )

    model.print_trainable_parameters()

    return model
