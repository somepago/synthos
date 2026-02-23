"""
Shared diffusion pipeline utilities for Z-Image.

Provides:
- Full text2image generation (manual denoising loop)
- img2img via native pipeline (SigLip-conditioned edit_image)
- SigLip image encoding for training
- Helpers for noise generation, latent decoding
"""

import torch
from PIL import Image

from src.constants import SCHEDULER_SCALE


# =============================================================================
# Utilities
# =============================================================================

def get_latent_shape(height: int, width: int, latent_channels: int = 16) -> tuple:
    """Get latent tensor shape for given image dimensions."""
    return (1, latent_channels, height // 8, width // 8)


def generate_noise(seed: int, shape: tuple, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Generate noise tensor with specific seed (CPU generator for reproducibility)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(shape, generator=generator, dtype=dtype)
    return noise.to(device)


def decode_latent(pipe, latent: torch.Tensor) -> Image.Image:
    """Decode a latent tensor to PIL Image via VAE."""
    pipe.load_models_to_device(["vae_decoder"])
    latent = latent.to(device=pipe.device, dtype=pipe.torch_dtype)
    image = pipe.vae_decoder(latent)
    image = pipe.vae_output_to_image(image)
    return image


def encode_image_vae(pipe, image: Image.Image) -> torch.Tensor:
    """Encode a PIL Image to VAE latents.

    Returns:
        Tensor of shape (1, 16, H/8, W/8) — the clean latent z_0.
    """
    pipe.load_models_to_device(["vae_encoder"])
    image_tensor = pipe.preprocess_image(image)
    latent = pipe.vae_encoder(image_tensor)
    return latent


def encode_image_siglip(pipe, image: Image.Image) -> torch.Tensor:
    """Encode a PIL Image to SigLip spatial features using the pipeline's image_encoder.

    Returns:
        Tensor of shape (H', W', 1152) — spatial SigLip features.
        H', W' depend on image size (roughly image_size / 16).

    Raises:
        RuntimeError if pipe.image_encoder is None (SigLip not loaded).
    """
    if pipe.image_encoder is None:
        raise RuntimeError(
            "Pipeline has no image_encoder (SigLip). "
            "Load with model_key='z-image-turbo-img2img' to include SigLip weights."
        )
    pipe.load_models_to_device(["image_encoder"])
    return pipe.image_encoder(image, device=pipe.device)


# =============================================================================
# Internal helpers (for manual denoising loops)
# =============================================================================

def _prepare_diffusion(pipe, prompt: str, num_inference_steps: int):
    """Common diffusion setup: set scheduler, encode prompt."""
    from diffsynth.pipelines.z_image import ZImageUnit_PromptEmbedder

    pipe.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=None)
    pipe.load_models_to_device(["text_encoder"])

    prompt_embedder = ZImageUnit_PromptEmbedder()
    prompt_embeds = prompt_embedder.encode_prompt(pipe, prompt, pipe.device)
    negative_embeds = prompt_embedder.encode_prompt(pipe, "", pipe.device)

    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    return prompt_embeds, negative_embeds, models


def _denoise_step(pipe, cfg_scale, inputs_shared, inputs_posi, inputs_nega, models, timestep, progress_id):
    """Execute a single denoising step. Mutates inputs_shared['latents'] in place."""
    timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
    noise_pred = pipe.cfg_guided_model_fn(
        pipe.model_fn, cfg_scale,
        inputs_shared, inputs_posi, inputs_nega,
        **models, timestep=timestep, progress_id=progress_id,
    )
    inputs_shared["latents"] = pipe.step(
        pipe.scheduler, progress_id=progress_id,
        noise_pred=noise_pred, **inputs_shared,
    )


def _decode_final(pipe, inputs_shared) -> Image.Image:
    """Decode final latents to PIL Image."""
    pipe.load_models_to_device(["vae_decoder"])
    image = pipe.vae_decoder(inputs_shared["latents"])
    image = pipe.vae_output_to_image(image)
    return image


# =============================================================================
# Public inference functions
# =============================================================================

@torch.no_grad()
def run_full_diffusion(
    pipe, prompt: str, noise: torch.Tensor,
    num_inference_steps: int, cfg_scale: float,
) -> Image.Image:
    """Run full text2image diffusion: noise in, image out."""
    prompt_embeds, negative_embeds, models = _prepare_diffusion(pipe, prompt, num_inference_steps)

    inputs_shared = {"latents": noise.to(device=pipe.device, dtype=pipe.torch_dtype)}
    inputs_posi = {"prompt_embeds": prompt_embeds}
    inputs_nega = {"prompt_embeds": negative_embeds}

    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        _denoise_step(pipe, cfg_scale, inputs_shared, inputs_posi, inputs_nega, models, timestep, progress_id)

    return _decode_final(pipe, inputs_shared)


@torch.no_grad()
def run_img2img(
    pipe, prompt: str, edit_image: Image.Image,
    num_inference_steps: int = 8, cfg_scale: float = 1.0,
    height: int = 512, width: int = 512, seed: int = 42,
) -> Image.Image:
    """Image-to-image via native pipeline with SigLip conditioning.

    Uses the pipeline's built-in edit_image pathway:
    - SigLip encoder produces spatial features from edit_image
    - Features are passed to the DiT as image_embeds (via model_fn)
    - Generation is conditioned on both text prompt and image features

    Args:
        edit_image: Input/reference image for conditioning.
        height, width: Output image dimensions.
        seed: Random seed for noise generation.
    """
    return pipe(
        prompt=prompt,
        edit_image=edit_image,
        height=height,
        width=width,
        seed=seed,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
    )
