# This file encapsulates model loading, specific patching needed for Isaac Sim containers, and the LoRA configuration.
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model


def _patch_transformers_attention_dispatch() -> None:
    """Patch transformers attention dispatch checks for container compatibility.

    In some Isaac Sim container builds, transformers will attempt to read
    `self._supports_sdpa` during model initialization, but that attribute may
    not exist on the base class. This leads to a hard crash when loading
    OpenVLA with `trust_remote_code=True`.

    We take a conservative approach:
      - define the expected `_supports_*` flags if missing
      - override the SDPA dispatch check to always return False

    This forces "eager" attention paths and avoids SDPA/FlashAttention dispatch.
    """

    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return

    for attr in ("_supports_sdpa", "_supports_flash_attn_2", "_supports_flex_attn"):
        if not hasattr(PreTrainedModel, attr):
            setattr(PreTrainedModel, attr, False)

    # Some transformer versions call `_sdpa_can_dispatch` during init and access
    # `_supports_sdpa` internally. Override to avoid touching the attribute.
    def _sdpa_can_dispatch(self, is_init_check: bool = False) -> bool:  # noqa: ARG001
        return False

    PreTrainedModel._sdpa_can_dispatch = _sdpa_can_dispatch  # type: ignore[assignment]


def load_vla_model_and_processor(vla_path: str, device: torch.device):
    _patch_transformers_attention_dispatch()
    
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    
    model = AutoModelForVision2Seq.from_pretrained(
        vla_path, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to(device)

    # Preference: Move LoRA config into the model wrapper
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, processor