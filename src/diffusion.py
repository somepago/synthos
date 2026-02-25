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
# VL encoder helpers (Qwen3-VL with Z-Image LLM weights spliced in)
# =============================================================================

def _cap_resolution(image: Image.Image, max_pixels: int) -> Image.Image:
    """Resize image if total pixels exceed max_pixels, preserving aspect ratio."""
    w, h = image.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


@torch.no_grad()
def encode_image_vl(pipe, image: Image.Image, device: str,
                    max_pixels: int = 768 * 768) -> torch.Tensor:
    """Encode an image via spliced VL model. Returns (L, 2560).

    Full VL forward pass: image goes through ViT → merger → spliced LLM
    with chat template context. Hidden states include both visual and text
    template tokens, filtered by attention mask.

    Args:
        max_pixels: Cap total pixels to limit token count (default 768*768 ~750 tokens).
    """
    if getattr(pipe, 'vl_model', None) is None:
        raise RuntimeError("No VL model available (load with text_encoder='qwen3vl')")

    image = _cap_resolution(image, max_pixels)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": ""},
    ]}]
    text = pipe.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = pipe.vl_processor(text=[text], images=[image], return_tensors="pt").to(device)

    out = pipe.vl_model.model(**inputs, output_hidden_states=True)
    h = out.hidden_states[-2][0]
    mask = inputs["attention_mask"][0].bool()
    return h[mask]


@torch.no_grad()
def encode_text_vl(pipe, text: str, device: str) -> torch.Tensor:
    """Encode text via spliced VL model. Returns (L, 2560).

    Used for negative prompts when cfg_scale > 1.0 with VL conditioning.
    """
    if getattr(pipe, 'vl_model', None) is None:
        raise RuntimeError("No VL model available (load with text_encoder='qwen3vl')")

    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    chat_text = pipe.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = pipe.vl_processor(text=[chat_text], return_tensors="pt").to(device)

    out = pipe.vl_model.model(**inputs, output_hidden_states=True)
    h = out.hidden_states[-2][0]
    mask = inputs["attention_mask"][0].bool()
    return h[mask]


@torch.no_grad()
def encode_images_vl(pipe, images: list, device: str,
                     max_pixels: int = 512 * 512) -> torch.Tensor:
    """Encode multiple images via spliced VL model in a single forward pass.

    The LLM's self-attention lets visual tokens from different images attend
    to each other, producing entangled representations.

    Returns (L, 2560) where L includes visual + text template tokens.
    """
    if getattr(pipe, 'vl_model', None) is None:
        raise RuntimeError("No VL model available (load with text_encoder='qwen3vl')")

    resized = [_cap_resolution(img, max_pixels) for img in images]

    content = [{"type": "image", "image": img} for img in resized]
    content.append({"type": "text", "text": ""})
    messages = [{"role": "user", "content": content}]

    text = pipe.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = pipe.vl_processor(text=[text], images=resized, return_tensors="pt").to(device)

    out = pipe.vl_model.model(**inputs, output_hidden_states=True)
    h = out.hidden_states[-2][0]
    mask = inputs["attention_mask"][0].bool()
    return h[mask]


@torch.no_grad()
def encode_interleaved_vl(pipe, content_list: list, device: str,
                          max_pixels: int = 512 * 512) -> torch.Tensor:
    """Encode interleaved image+text content via spliced VL model.

    Args:
        content_list: list of dicts, each {"img": PIL.Image} or {"txt": str}.
            Images should already be opened as PIL.
        device: compute device
        max_pixels: cap per image

    Returns (L, 2560) prompt embeddings.
    """
    if getattr(pipe, 'vl_model', None) is None:
        raise RuntimeError("No VL model available (load with text_encoder='qwen3vl')")

    chat_content = []
    pil_images = []
    for item in content_list:
        if "img" in item:
            img = _cap_resolution(item["img"], max_pixels)
            chat_content.append({"type": "image", "image": img})
            pil_images.append(img)
        elif "txt" in item:
            chat_content.append({"type": "text", "text": item["txt"]})

    # Ensure there's at least some text (empty string) at the end for chat template
    if not any("txt" in item for item in content_list):
        chat_content.append({"type": "text", "text": ""})

    messages = [{"role": "user", "content": chat_content}]
    text = pipe.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = pipe.vl_processor(
        text=[text], images=pil_images if pil_images else None,
        return_tensors="pt"
    ).to(device)

    out = pipe.vl_model.model(**inputs, output_hidden_states=True)
    h = out.hidden_states[-2][0]
    mask = inputs["attention_mask"][0].bool()
    return h[mask]


# =============================================================================
# B5: Weighted token averaging (no cross-image attention)
# =============================================================================

@torch.no_grad()
def encode_weighted_avg_vl(pipe, images: list, device: str,
                           alpha: float = 0.5,
                           max_pixels: int = 512 * 512) -> torch.Tensor:
    """B5: Encode images SEPARATELY, then concatenate with per-image alpha scaling.

    Each image gets its own VL forward pass — no cross-image attention.
    Output is [alpha * h_img0 ; (1-alpha) * h_img1 ; ...] for 2 images,
    or equal-weight scaling for 3+ images.

    Args:
        images: list of PIL Images
        alpha: weight for first image (second gets 1-alpha). For 3+ images, ignored — uses equal weights.
        max_pixels: cap per image

    Returns (L_total, 2560) concatenated scaled embeddings.
    """
    if getattr(pipe, 'vl_model', None) is None:
        raise RuntimeError("No VL model available (load with text_encoder='qwen3vl')")

    if len(images) < 2:
        raise ValueError("Need at least 2 images for weighted averaging")

    # Determine weights
    if len(images) == 2:
        weights = [alpha, 1.0 - alpha]
    else:
        w = 1.0 / len(images)
        weights = [w] * len(images)

    # Encode each image separately
    scaled_parts = []
    for img, w in zip(images, weights):
        h = encode_image_vl(pipe, img, device, max_pixels)
        scaled_parts.append(h * w)

    return torch.cat(scaled_parts, dim=0)


# =============================================================================
# B6: Weighted token concatenation (cross-attention + per-image scaling)
# =============================================================================

VISION_START_ID = 151652
VISION_END_ID = 151653

def _find_image_token_ranges(input_ids: torch.Tensor):
    """Find (start, end) index ranges for each image's visual tokens in input_ids.

    Looks for <|vision_start|> ... <|vision_end|> boundaries.
    Returns list of (start_idx, end_idx) tuples (exclusive end).
    The range covers tokens BETWEEN vision_start and vision_end (the visual tokens).
    """
    ids = input_ids.squeeze().tolist()
    ranges = []
    i = 0
    while i < len(ids):
        if ids[i] == VISION_START_ID:
            start = i + 1  # first visual token is after vision_start
            j = start
            while j < len(ids) and ids[j] != VISION_END_ID:
                j += 1
            ranges.append((start, j))  # exclusive end
            i = j + 1
        else:
            i += 1
    return ranges


@torch.no_grad()
def encode_weighted_concat_vl(pipe, content_list: list, device: str,
                              alpha: float = 0.5,
                              max_pixels: int = 512 * 512) -> torch.Tensor:
    """B6: Encode images TOGETHER (cross-attention), then scale output tokens by image-of-origin.

    Full VL forward pass with all images seeing each other via self-attention.
    After encoding, visual tokens from each image are scaled by their weight.
    Non-visual tokens (chat template, text) are left unscaled.

    Args:
        content_list: list of dicts, each {"img": PIL.Image} or {"txt": str}
        alpha: weight for first image's visual tokens (second gets 1-alpha)
        max_pixels: cap per image

    Returns (L, 2560) prompt embeddings with per-image visual token scaling.
    """
    if getattr(pipe, 'vl_model', None) is None:
        raise RuntimeError("No VL model available (load with text_encoder='qwen3vl')")

    # Build chat content (same as encode_interleaved_vl)
    chat_content = []
    pil_images = []
    for item in content_list:
        if "img" in item:
            img = _cap_resolution(item["img"], max_pixels)
            chat_content.append({"type": "image", "image": img})
            pil_images.append(img)
        elif "txt" in item:
            chat_content.append({"type": "text", "text": item["txt"]})

    if not any("txt" in item for item in content_list):
        chat_content.append({"type": "text", "text": ""})

    n_images = len(pil_images)
    if n_images < 2:
        raise ValueError("Need at least 2 images for weighted concat")

    # Determine weights
    if n_images == 2:
        weights = [alpha, 1.0 - alpha]
    else:
        w = 1.0 / n_images
        weights = [w] * n_images

    messages = [{"role": "user", "content": chat_content}]
    text = pipe.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = pipe.vl_processor(
        text=[text], images=pil_images if pil_images else None,
        return_tensors="pt"
    ).to(device)

    # Find image token boundaries BEFORE forward pass
    image_ranges = _find_image_token_ranges(inputs["input_ids"])

    out = pipe.vl_model.model(**inputs, output_hidden_states=True)
    h = out.hidden_states[-2][0]  # (seq_len, 2560)
    mask = inputs["attention_mask"][0].bool()

    # Scale visual tokens by image-of-origin weight
    for (start, end), w in zip(image_ranges, weights):
        h[start:end] *= w

    return h[mask]


# =============================================================================
# Internal helpers (for manual denoising loops)
# =============================================================================

def _prepare_diffusion(pipe, prompt: str, num_inference_steps: int):
    """Common diffusion setup: set scheduler, encode prompt via pipeline's text encoder."""
    pipe.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=None)

    from diffsynth.pipelines.z_image import ZImageUnit_PromptEmbedder
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
def run_img2img_omni(
    pipe, prompt: str, edit_image: Image.Image,
    num_inference_steps: int = 8, cfg_scale: float = 1.0,
    height: int = 512, width: int = 512, seed: int = 42,
) -> Image.Image:
    """Image-to-image via native pipeline with SigLip + VAE reference (omni mode).

    Uses the pipeline's built-in edit_image pathway which passes BOTH
    SigLip features AND VAE-encoded reference latents to the DiT.
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


@torch.no_grad()
def run_img2img_siglip_caption(
    pipe, prompt: str, edit_image: Image.Image,
    num_inference_steps: int = 8, cfg_scale: float = 1.0,
    height: int = 512, width: int = 512, seed: int = 42,
) -> Image.Image:
    """SigLip-as-caption image conditioning via the standard turbo (non-omni) path.

    Projects SigLip features to text embedding space and concatenates with
    text embeddings as extra caption tokens. Uses the frozen cap_embedder +
    DiT text conditioning path — no omni mode, no noise masks.
    """
    from diffsynth.pipelines.z_image import ZImageUnit_PromptEmbedder

    # Encode SigLip features and project to text space
    siglip_raw = encode_image_siglip(pipe, edit_image)  # (H', W', 1152)
    siglip_flat = siglip_raw.reshape(-1, siglip_raw.shape[-1])
    siglip_projected = pipe.siglip_projection(siglip_flat)  # (H'*W', 2560)

    # Encode text prompt via pipeline's text encoder
    embedder = ZImageUnit_PromptEmbedder()
    pipe.load_models_to_device(["text_encoder"])
    text_embeds = embedder.encode_prompt(pipe, prompt, pipe.device)[0]  # (L, 2560)
    neg_text_embeds = embedder.encode_prompt(pipe, "", pipe.device)[0]

    # Build extended prompts: text + SigLip tokens
    posi_prompt = torch.cat([text_embeds, siglip_projected], dim=0)
    nega_prompt = torch.cat([neg_text_embeds, siglip_projected], dim=0)

    # Generate noise
    latent_shape = get_latent_shape(height, width)
    noise = generate_noise(seed, latent_shape, pipe.device, pipe.torch_dtype)

    # Denoise via standard turbo path
    pipe.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=None)
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    inputs_shared = {"latents": noise}
    inputs_posi = {"prompt_embeds": posi_prompt}
    inputs_nega = {"prompt_embeds": nega_prompt}

    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        _denoise_step(pipe, cfg_scale, inputs_shared, inputs_posi, inputs_nega,
                      models, timestep, progress_id)

    return _decode_final(pipe, inputs_shared)
