"""
Model registry for supported LLM models.

Each model entry defines its HuggingFace ID, quantization, VRAM requirements,
and any model-specific vLLM configuration (e.g., GDN backend for Qwen3.5,
thinking mode handling for DeepSeek-R1).

Usage:
    from config.model_registry import get_model_config, MODEL_REGISTRY

    config = get_model_config("qwen3.5-27b")
    # config.hf_id -> "Qwen/Qwen3.5-27B-FP8"
    # config.vllm_kwargs -> {"gdn_prefill_backend": "triton"}
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class ModelConfig:
    """Configuration for a specific model variant."""
    hf_id: str
    short_name: str
    quantization: str  # "fp8", "awq", "bf16", "gptq"
    vram_gb: float
    min_gpu_vram_gb: float
    max_model_len: int = 32768
    has_thinking_mode: bool = False
    vllm_kwargs: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


# ============================================================================
# Model Registry
# ============================================================================
# All models sized to fit on RTX 6000 (48 GB) as the standard target GPU.
# ============================================================================

MODEL_REGISTRY: Dict[str, ModelConfig] = {

    # --- Qwen 3.5 ---
    "qwen3.5-27b": ModelConfig(
        hf_id="QuantTrio/Qwen3.5-27B-AWQ",
        short_name="qwen3.5-27b",
        quantization="awq",
        vram_gb=14,
        min_gpu_vram_gb=24,
        max_model_len=32768,
        has_thinking_mode=True,
        vllm_kwargs={"gdn_prefill_backend": "triton", "quantization": "awq"},
        description="Qwen3.5 27B AWQ 4-bit — fits all GPUs (RTX 5000/6000/H100)",
    ),

    "qwen3.5-27b-fp8": ModelConfig(
        hf_id="Qwen/Qwen3.5-27B-FP8",
        short_name="qwen3.5-27b-fp8",
        quantization="fp8",
        vram_gb=28.5,
        min_gpu_vram_gb=60,
        max_model_len=32768,
        has_thinking_mode=True,
        vllm_kwargs={"gdn_prefill_backend": "triton"},
        description="Qwen3.5 27B FP8 — higher quality, needs H100 (80GB)",
    ),

    # --- DeepSeek R1 ---
    "deepseek-r1-14b": ModelConfig(
        hf_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        short_name="deepseek-r1-14b",
        quantization="bf16",
        vram_gb=28,
        min_gpu_vram_gb=40,
        max_model_len=32768,
        has_thinking_mode=True,
        vllm_kwargs={},
        description="DeepSeek R1 distilled from Qwen2.5-14B, BF16",
    ),

    "deepseek-r1-32b": ModelConfig(
        hf_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        short_name="deepseek-r1-32b",
        quantization="bf16",
        vram_gb=64,
        min_gpu_vram_gb=80,
        max_model_len=32768,
        has_thinking_mode=True,
        vllm_kwargs={"tensor_parallel_size": 2},
        description="DeepSeek R1 distilled from Qwen2.5-32B, BF16 — needs 2 GPUs (TP=2)",
    ),

    # --- LLaMA ---
    "llama3.1-8b": ModelConfig(
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        short_name="llama3.1-8b",
        quantization="bf16",
        vram_gb=16,
        min_gpu_vram_gb=24,
        max_model_len=32768,
        has_thinking_mode=False,
        vllm_kwargs={},
        description="LLaMA 3.1 8B Instruct, BF16 — baseline model",
    ),

    "llama3.3-70b": ModelConfig(
        hf_id="meta-llama/Llama-3.3-70B-Instruct",
        short_name="llama3.3-70b",
        quantization="bf16",
        vram_gb=140,
        min_gpu_vram_gb=160,
        max_model_len=32768,
        has_thinking_mode=False,
        vllm_kwargs={"tensor_parallel_size": 4},
        description="LLaMA 3.3 70B Instruct, BF16 — SOTA, needs 4 GPUs (TP=4)",
    ),
}


def get_model_config(model_key: str) -> ModelConfig:
    """
    Look up a model by its short key (e.g., 'qwen3.5-27b').
    Falls back to treating the key as a raw HuggingFace ID.
    """
    if model_key in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_key]

    # Check if it matches a HuggingFace ID
    for config in MODEL_REGISTRY.values():
        if config.hf_id == model_key:
            return config

    # Unknown model — create a minimal config so the pipeline can still try
    return ModelConfig(
        hf_id=model_key,
        short_name=model_key.split("/")[-1].lower(),
        quantization="unknown",
        vram_gb=0,
        min_gpu_vram_gb=0,
        max_model_len=32768,
        has_thinking_mode=False,
        vllm_kwargs={},
        description=f"Unknown model: {model_key}",
    )


def list_models() -> None:
    """Print available models."""
    print("\nAvailable models:")
    print("-" * 80)
    for key, cfg in MODEL_REGISTRY.items():
        print(f"  {key:25s}  {cfg.hf_id:45s}  {cfg.vram_gb:.0f} GiB ({cfg.quantization})")
    print()
