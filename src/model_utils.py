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
}


def load_pipeline(model_key: str, device: str = "cuda", torch_dtype: str = "bfloat16"):
    """Load a Z-Image pipeline by model key."""
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)
    entry = MODEL_REGISTRY[model_key]

    model_configs = [ModelConfig(**cfg) for cfg in entry["model_configs"]]
    tokenizer_config = ModelConfig(**entry["tokenizer_config"])

    from diffsynth.pipelines.z_image import ZImagePipeline

    print(f"Loading {entry['display_name']} (dtype={torch_dtype})")
    pipe = ZImagePipeline.from_pretrained(
        torch_dtype=dtype, device=device,
        model_configs=model_configs, tokenizer_config=tokenizer_config,
    )

    # For img2img: load SigLip2 from HF and add projection layers to DiT
    if model_key == "z-image-turbo-img2img":
        _setup_siglip_conditioning(pipe, device, dtype)

    return pipe


def _setup_siglip_conditioning(pipe, device: str, dtype: torch.dtype):
    """Load SigLip2 from HuggingFace and add projection layers to the DiT.

    This enables the omni path in model_fn_z_image by:
    1. Loading google/siglip2-so400m-patch16-naflex as pipe.image_encoder
    2. Adding siglip_embedder (RMSNorm + Linear, 1152 -> 3840) to the DiT
    3. Adding siglip_refiner (2 transformer blocks) to the DiT
    4. Adding siglip_pad_token to the DiT

    All added layers are randomly initialized and intended to be trained.
    """
    from diffsynth.models.siglip2_image_encoder import Siglip2ImageEncoder428M

    # Load SigLip2 image encoder (uses HF transformers Siglip2VisionModel internally)
    print("Loading SigLip2 image encoder (Siglip2ImageEncoder428M)...")
    image_encoder = Siglip2ImageEncoder428M()
    # Load pretrained weights from HF
    from transformers import Siglip2VisionModel
    hf_model = Siglip2VisionModel.from_pretrained("google/siglip2-so400m-patch16-naflex")
    image_encoder.load_state_dict(hf_model.state_dict(), strict=False)
    del hf_model
    image_encoder = image_encoder.to(device=device, dtype=dtype)
    image_encoder.eval()
    image_encoder.requires_grad_(False)
    pipe.image_encoder = image_encoder
    print("  SigLip2 loaded and frozen")

    # Add projection layers to DiT for omni path
    _add_siglip_layers_to_dit(pipe.dit, device, dtype)


def _add_siglip_layers_to_dit(dit, device: str, dtype: torch.dtype):
    """Add siglip_embedder, siglip_refiner, and siglip_pad_token to a Turbo DiT.

    These are the layers the omni path in model_fn_z_image expects.
    They are randomly initialized — training will learn the projection.
    """
    from diffsynth.models.general_modules import RMSNorm
    from diffsynth.models.z_image_dit import ZImageTransformerBlock

    siglip_feat_dim = 1152  # SigLip2 so400m hidden size
    dim = 3840              # DiT hidden dimension
    n_heads = 30
    n_kv_heads = 30
    norm_eps = 1e-5
    qk_norm = True
    n_refiner_layers = 2

    # Projection: RMSNorm(1152) -> Linear(1152, 3840)
    dit.siglip_embedder = nn.Sequential(
        RMSNorm(siglip_feat_dim, eps=norm_eps),
        nn.Linear(siglip_feat_dim, dim, bias=True),
    ).to(device=device, dtype=dtype)

    # Refiner: 2 transformer blocks (no AdaLN modulation)
    dit.siglip_refiner = nn.ModuleList([
        ZImageTransformerBlock(
            2000 + layer_id, dim, n_heads, n_kv_heads,
            norm_eps, qk_norm, modulation=False,
        )
        for layer_id in range(n_refiner_layers)
    ]).to(device=device, dtype=dtype)

    # Pad token for variable-length siglip sequences
    dit.siglip_pad_token = nn.Parameter(
        torch.empty((1, dim), device=device, dtype=dtype)
    )
    nn.init.normal_(dit.siglip_pad_token, std=0.02)

    # Store feat dim so forward() can use it
    dit.siglip_feat_dim = siglip_feat_dim

    trainable = sum(
        p.numel() for name, p in dit.named_parameters()
        if "siglip" in name
    )
    print(f"  Added siglip layers to DiT: embedder + {n_refiner_layers} refiner blocks + pad_token")
    print(f"  SigLip projection params: {trainable:,}")


def get_defaults(model_key: str) -> dict:
    """Return default inference params (num_inference_steps, cfg_scale)."""
    entry = MODEL_REGISTRY[model_key]
    return {"num_inference_steps": entry["num_inference_steps"], "cfg_scale": entry["cfg_scale"]}
