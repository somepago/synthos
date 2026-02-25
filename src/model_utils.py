"""
Model loading utilities for Z-Image pipelines.

Usage:
    from src.model_utils import load_pipeline
    pipe = load_pipeline("z-image-turbo")              # text2image only
    pipe = load_pipeline("z-image-turbo-img2img")      # + SigLip encoder + projection layers
"""

import torch
import torch.nn as nn
from diffsynth.core.loader.config import ModelConfig


MODEL_REGISTRY = {
    "z-image-base": {
        "display_name": "Z-Image (Base)",
        "role": "base",
        "model_configs": [
            {"model_id": "Tongyi-MAI/Z-Image", "origin_file_pattern": "transformer/*.safetensors"},
            {"model_id": "Tongyi-MAI/Z-Image-Turbo", "origin_file_pattern": "text_encoder/*.safetensors"},
            {"model_id": "Tongyi-MAI/Z-Image-Turbo", "origin_file_pattern": "vae/diffusion_pytorch_model.safetensors"},
        ],
        "tokenizer_config": {"model_id": "Tongyi-MAI/Z-Image-Turbo", "origin_file_pattern": "tokenizer/"},
        "num_inference_steps": 50,
        "cfg_scale": 4.0,
    },
    "z-image-turbo": {
        "display_name": "Z-Image-Turbo (Distilled)",
        "role": "distilled",
        "model_configs": [
            {"model_id": "Tongyi-MAI/Z-Image-Turbo", "origin_file_pattern": "transformer/*.safetensors"},
            {"model_id": "Tongyi-MAI/Z-Image-Turbo", "origin_file_pattern": "text_encoder/*.safetensors"},
            {"model_id": "Tongyi-MAI/Z-Image-Turbo", "origin_file_pattern": "vae/diffusion_pytorch_model.safetensors"},
        ],
        "tokenizer_config": {"model_id": "Tongyi-MAI/Z-Image-Turbo", "origin_file_pattern": "tokenizer/"},
        "num_inference_steps": 8,
        "cfg_scale": 1.0,
    },
    # Turbo + SigLip for img2img (SigLip loaded separately from HF)
    "z-image-turbo-img2img": {
        "display_name": "Z-Image-Turbo + SigLip (img2img)",
        "role": "distilled",
        "model_configs": [
            {"model_id": "Tongyi-MAI/Z-Image-Turbo", "origin_file_pattern": "transformer/*.safetensors"},
            {"model_id": "Tongyi-MAI/Z-Image-Turbo", "origin_file_pattern": "text_encoder/*.safetensors"},
            {"model_id": "Tongyi-MAI/Z-Image-Turbo", "origin_file_pattern": "vae/diffusion_pytorch_model.safetensors"},
        ],
        "tokenizer_config": {"model_id": "Tongyi-MAI/Z-Image-Turbo", "origin_file_pattern": "tokenizer/"},
        "num_inference_steps": 8,
        "cfg_scale": 1.0,
    },
    # Base + SigLip for img2img (standard flow matching, not distilled)
    "z-image-base-img2img": {
        "display_name": "Z-Image-Base + SigLip (img2img)",
        "role": "base",
        "model_configs": [
            {"model_id": "Tongyi-MAI/Z-Image", "origin_file_pattern": "transformer/*.safetensors"},
            {"model_id": "Tongyi-MAI/Z-Image-Turbo", "origin_file_pattern": "text_encoder/*.safetensors"},
            {"model_id": "Tongyi-MAI/Z-Image-Turbo", "origin_file_pattern": "vae/diffusion_pytorch_model.safetensors"},
        ],
        "tokenizer_config": {"model_id": "Tongyi-MAI/Z-Image-Turbo", "origin_file_pattern": "tokenizer/"},
        "num_inference_steps": 50,
        "cfg_scale": 4.0,
    },
}


def load_pipeline(model_key: str, device: str = "cuda", torch_dtype: str = "bfloat16",
                   text_encoder: str = "qwen3"):
    """Load a Z-Image pipeline by model key.

    Args:
        text_encoder: Which text encoder to use.
            "qwen3" — Z-Image's Qwen3-4B via pipeline (default, t2i only)
            "qwen3vl" — Full Qwen3-VL-4B with Z-Image LLM weights spliced in.
                         Replaces text encoder (~9GB VL model, saves ~8GB text encoder).
    """
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)
    entry = MODEL_REGISTRY[model_key]

    model_configs = [ModelConfig(**cfg) for cfg in entry["model_configs"]]

    # qwen3vl: strip text_encoder from pipeline (VL model with spliced weights replaces it)
    if text_encoder == "qwen3vl":
        model_configs = [
            cfg for cfg in model_configs
            if "text_encoder" not in cfg.origin_file_pattern
        ]

    tokenizer_config = ModelConfig(**entry["tokenizer_config"])

    from diffsynth.pipelines.z_image import ZImagePipeline

    print(f"Loading {entry['display_name']} (dtype={torch_dtype}, text_encoder={text_encoder})")
    pipe = ZImagePipeline.from_pretrained(
        torch_dtype=dtype, device=device,
        model_configs=model_configs, tokenizer_config=tokenizer_config,
    )

    # Attach VL encoder with spliced Z-Image LLM weights
    if text_encoder == "qwen3vl":
        _setup_vl_splice(pipe, device, dtype)
    else:
        pipe.vl_model = None
        pipe.vl_processor = None

    # For img2img: load SigLip2 from HF and add projection layers
    if model_key.endswith("-img2img"):
        _setup_siglip_conditioning(pipe, device, dtype)

    return pipe


def _setup_vl_splice(pipe, device: str, dtype: torch.dtype):
    """Load Qwen3-VL-4B and splice Z-Image's trained LLM weights into it.

    This gives us a VL model that:
    - Has Qwen3-VL's vision tower (ViT + PatchMerger) for image understanding
    - Has Z-Image's trained LLM weights that the DiT was trained to consume
    - Processes images through the full VL pipeline (ViT → merger → LLM)
      with chat template context, producing better embeddings than split encoding.
    """
    import gc
    from pathlib import Path
    from safetensors.torch import load_file
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print("Loading Qwen3-VL-4B + splicing Z-Image LLM weights...")
    vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct", torch_dtype=dtype,
    ).to(device).eval()

    # Splice Z-Image's trained LLM weights into VL model's language backbone
    model_dir = Path("models/Tongyi-MAI/Z-Image-Turbo/text_encoder")
    z_sd = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        z_sd.update(load_file(str(f), device="cpu"))
    z_sd = {k.removeprefix("model."): v for k, v in z_sd.items()}

    result = vl_model.model.language_model.load_state_dict(z_sd, strict=False)
    del z_sd; gc.collect()
    print(f"  Spliced: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")

    vl_model.requires_grad_(False)
    pipe.vl_model = vl_model
    pipe.vl_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    print("  VL splice model attached")


def _setup_siglip_conditioning(pipe, device: str, dtype: torch.dtype):
    """Load SigLip2 encoder and create a standalone projection to text embedding space.

    The projection maps SigLip features (1152-dim) to the text encoder output space
    (2560-dim) so they can be concatenated with text embeddings and processed by the
    frozen cap_embedder + DiT as additional caption tokens.

    DiT is NOT modified — dit.siglip_embedder stays None, so model_fn routes to
    the standard turbo (non-omni) code path.
    """
    from diffsynth.models.siglip2_image_encoder import Siglip2ImageEncoder428M

    print("Loading SigLip2 image encoder (Siglip2ImageEncoder428M)...")
    image_encoder = Siglip2ImageEncoder428M()
    from transformers import Siglip2VisionModel
    hf_model = Siglip2VisionModel.from_pretrained("google/siglip2-so400m-patch16-naflex")
    image_encoder.load_state_dict(hf_model.state_dict(), strict=False)
    del hf_model
    image_encoder = image_encoder.to(device=device, dtype=dtype)
    image_encoder.eval()
    image_encoder.requires_grad_(False)
    pipe.image_encoder = image_encoder
    print("  SigLip2 loaded and frozen")

    # Standalone projection: SigLip features → text embedding space
    siglip_dim = 1152   # SigLip2 so400m hidden size
    cap_dim = 2560      # text encoder output dim (= cap_embedder input dim)
    pipe.siglip_projection = nn.Sequential(
        nn.LayerNorm(siglip_dim),
        nn.Linear(siglip_dim, cap_dim, bias=True),
    ).to(device=device, dtype=dtype)

    trainable = sum(p.numel() for p in pipe.siglip_projection.parameters())
    print(f"  SigLip projection (standalone): {trainable:,} params")


def get_defaults(model_key: str) -> dict:
    """Return default inference params (num_inference_steps, cfg_scale)."""
    entry = MODEL_REGISTRY[model_key]
    return {"num_inference_steps": entry["num_inference_steps"], "cfg_scale": entry["cfg_scale"]}
