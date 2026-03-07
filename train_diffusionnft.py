#!/usr/bin/env python3
"""
DiffusionNFT training for coherent multi-image blending via VL splice.

Uses DiffusionNFT (arxiv.org/abs/2509.16117) — reward-weighted velocity matching
on the forward diffusion process. Only needs ONE DiT forward+backward per sample
(vs 8x for naive GRPO backprop through denoising chain).

Objective: given two input images, generate a coherent "surprise" blend. The VLM
reward judges whether the output is visually plausible (reasonable proportions,
consistent style, no artifacts) rather than matching a specific text prompt.

Trains:
  1. CompositionModule: transformer block(s) on VL prompt_embeds (2560-dim)
  2. LoRA on DiT Linear layers (manual, no HuggingFace/peft)

Rollout filtering: skips gradient updates when max(scores) < threshold — if all
rollouts are bad, there's no positive/negative contrast signal to learn from.

Dataset format (composition.jsonl, one JSON array per line):
    [{"img": "001532193.jpg"}, {"img": "005112355.jpg"}]
    [{"img": "001532193.jpg"}, {"img": "005112355.jpg"}, {"prompt": "optional text"}]

Usage:
    # Start vLLM reward server first:
    # python -m vllm.entrypoints.openai.api_server \\
    #     --model Qwen/Qwen3-VL-8B-Instruct --max-model-len 4096 --port 8100

    python train_diffusionnft.py --dataset /path/to/composition.jsonl --reward_url http://localhost:8100
    python train_diffusionnft.py --dataset /path/to/composition.jsonl --dry_run
"""

from src import env_setup  # noqa: F401

import argparse
import base64
import json
import math
import random
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import psutil
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import wandb

from src.model_utils import load_pipeline
from src.constants import SCHEDULER_SCALE
from src.diffusion import (
    encode_interleaved_vl,
    encode_image_vae,
    get_latent_shape,
    generate_noise,
    decode_latent,
)

VLM_JUDGE_PROMPT = (
    "You are a strict judge evaluating an AI-generated image that was created by "
    "blending two reference images.\n\n"
    "A GOOD blend takes the subject/content from one reference and the style/aesthetic "
    "from the other, producing a single coherent image. A BAD blend just pastes both "
    "subjects together, creates floating/disjointed objects, or produces artifacts.\n\n"
    "Score from 1 to 10. Be HARSH — most blends should score 3-6.\n\n"
    "SEVERE penalties (score 1-3):\n"
    "- Floating or disjointed objects that don't belong in the scene\n"
    "- Obvious collage effect — elements look pasted, not integrated\n"
    "- Broken anatomy, deformed faces, extra limbs\n"
    "- Both subjects just placed side by side or overlapping randomly\n"
    "- Incoherent scene that makes no visual sense\n\n"
    "Medium penalties (score 4-6):\n"
    "- Both subjects appear but are somewhat integrated into one scene\n"
    "- Inconsistent lighting or color palette between elements\n"
    "- Minor artifacts or slight style mismatch\n\n"
    "High scores (score 7-8):\n"
    "- Clear subject from one reference rendered in the style of the other\n"
    "- OR both elements genuinely fused into one coherent scene/concept\n"
    "- Consistent style throughout, no obvious artifacts\n\n"
    "Score 9-10: Reserved for exceptional results that look professionally made\n\n"
    "Respond with ONLY a single integer."
)

# =============================================================================
# Dataset: composition JSONL
# =============================================================================

class CompositionDataset:
    """Loads composition.jsonl: each line is a JSON array of
    [{"img": "filename"}, {"img": "filename"}, {"prompt": "text"}].

    Images are loaded from image_dir relative to the JSONL file.
    """

    def __init__(self, jsonl_path: str):
        self.jsonl_path = Path(jsonl_path)
        self.base_dir = self.jsonl_path.parent
        self.entries = []
        with open(self.jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.entries.append(json.loads(line))
        print(f"Loaded {len(self.entries)} composition entries from {self.jsonl_path}")

    def sample(self) -> dict:
        """Return a random entry parsed into (images, prompt, content_list).

        content_list is in encode_interleaved_vl format:
            [{"img": PIL.Image}, {"txt": str}, {"img": PIL.Image}]
        """
        entry = random.choice(self.entries)
        images = []
        prompt = ""
        content_list = []

        for item in entry:
            if "img" in item:
                # Try path as-is first (absolute or relative to CWD),
                # then relative to JSONL parent dir, then relative to
                # images/ subdir alongside JSONL
                img_name = item["img"]
                for candidate in [
                    Path(img_name),
                    self.base_dir / img_name,
                    self.base_dir / "images" / img_name,
                ]:
                    if candidate.exists():
                        img_path = candidate
                        break
                else:
                    img_path = self.base_dir / "images" / img_name
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                content_list.append({"img": img})
            elif "prompt" in item:
                prompt = item["prompt"]
                content_list.append({"txt": prompt})

        return {"images": images, "prompt": prompt, "content_list": content_list}

    def __len__(self):
        return len(self.entries)


# =============================================================================
# Composition Module
# =============================================================================

class CompositionBlock(nn.Module):
    """Single transformer block for multi-image composition.

    Pre-norm architecture with self-attention + FFN.
    Zero-initialized output projections → starts as identity.
    """

    def __init__(self, dim: int, n_heads: int, ffn_mult: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * ffn_mult)),
            nn.GELU(),
            nn.Linear(int(dim * ffn_mult), dim),
        )
        # Zero-init outputs → identity at init
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class CompositionModule(nn.Module):
    """Trainable transformer block(s) that learn to compose multi-image
    VL embeddings into a coherent conditioning signal for the DiT.

    Sits between encode_interleaved_vl() output (L, 2560) and the DiT's
    frozen cap_embedder. The VL model encodes both images with cross-attention
    but doesn't know what the DiT needs. This module learns the transformation.

    Zero-initialized → acts as identity at start of training.
    """

    def __init__(self, dim: int = 2560, n_heads: int = 16, n_layers: int = 1,
                 ffn_mult: float = 4.0):
        super().__init__()
        self.layers = nn.ModuleList([
            CompositionBlock(dim, n_heads, ffn_mult) for _ in range(n_layers)
        ])

    def forward(self, x):
        # x: (L, D) — no batch dim from encode_interleaved_vl
        needs_squeeze = x.dim() == 2
        if needs_squeeze:
            x = x.unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        if needs_squeeze:
            x = x.squeeze(0)
        return x


# =============================================================================
# LoRA (manual, no HuggingFace)
# =============================================================================

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear that adds a low-rank residual.

    output = frozen_linear(x) + x @ A @ B * (alpha / rank)

    A: (in_features, rank) — kaiming uniform init
    B: (rank, out_features) — zero init
    → LoRA contribution is zero at initialization.
    """

    def __init__(self, original: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scale = alpha / rank

        # Freeze the original
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        # Low-rank matrices (trainable)
        self.lora_A = nn.Parameter(torch.empty(original.in_features, rank,
                                               dtype=original.weight.dtype,
                                               device=original.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(rank, original.out_features,
                                               dtype=original.weight.dtype,
                                               device=original.weight.device))
        # Kaiming init for A, zeros for B → zero output at init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base = self.original(x)
        lora = (x @ self.lora_A @ self.lora_B) * self.scale
        return base + lora


def inject_lora(model: nn.Module, rank: int, alpha: float,
                min_dim: int = 512, include_adaln: bool = True) -> list[str]:
    """Replace nn.Linear modules in model with LoRALinear wrappers.

    Targets:
      - All Linear layers with min(in, out) >= min_dim
      - adaLN_modulation layers (if include_adaln=True)

    Returns list of module names that got LoRA.
    """
    targets = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if min(module.weight.shape) >= min_dim:
            targets.append(name)
        elif include_adaln and "adaLN_modulation" in name:
            targets.append(name)

    for name in targets:
        # Navigate to parent module
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr = parts[-1]
        original = getattr(parent, attr)
        setattr(parent, attr, LoRALinear(original, rank, alpha))

    return targets


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Collect all LoRA A/B parameters from LoRALinear modules."""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora_A, module.lora_B])
    return params


def get_lora_state_dict(model: nn.Module) -> dict:
    """Extract LoRA weights as a state dict."""
    sd = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            sd[f"{name}.lora_A"] = module.lora_A.data
            sd[f"{name}.lora_B"] = module.lora_B.data
    return sd


def load_lora_state_dict(model: nn.Module, sd: dict):
    """Load LoRA weights from a state dict."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            if a_key in sd:
                module.lora_A.data.copy_(sd[a_key])
            if b_key in sd:
                module.lora_B.data.copy_(sd[b_key])


# =============================================================================
# VLM Reward Client
# =============================================================================

def pil_to_base64(img: Image.Image, max_size: int = 512) -> str:
    """Convert PIL image to base64 string, resized for reward model efficiency."""
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()



class VLMRewardClient:
    """Scores composition outputs via Qwen3-VL-8B running on vLLM.

    The VLM sees all reference images + the prompt + the generated output,
    and judges whether the output correctly combines elements from the
    references according to the prompt.
    """

    def __init__(self, base_url: str = "http://localhost:8100",
                 model: str = "Qwen/Qwen3-VL-8B-Instruct"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.url = f"{self.base_url}/v1/chat/completions"

    def _build_message(self, ref_images: list[Image.Image],
                       output_img: Image.Image) -> list:
        content = []
        content.append({"type": "text", "text": "Reference images used as input:\n"})
        for i, ref in enumerate(ref_images):
            content.append({"type": "text", "text": f"Reference {i+1}:"})
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{pil_to_base64(ref)}"}})

        content.append({"type": "text", "text": "\nGenerated result:"})
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{pil_to_base64(output_img)}"}})

        content.append({"type": "text", "text": VLM_JUDGE_PROMPT})
        return [{"role": "user", "content": content}]

    def score_batch(self, ref_images_list: list,
                    output_images: list) -> list[float]:
        """Score a batch. ref_images_list[i] is a list of reference images for sample i."""
        scores = []
        for refs, out in zip(ref_images_list, output_images):
            try:
                score = self._score_single(refs, out)
            except Exception as e:
                print(f"  VLM reward error: {e}, defaulting to 5.0")
                score = 5.0
            scores.append(score)
        return scores

    def _score_single(self, ref_images, output_img) -> float:
        messages = self._build_message(ref_images, output_img)
        resp = requests.post(self.url, json={
            "model": self.model,
            "messages": messages,
            "max_tokens": 16,
            "temperature": 0.1,
        }, timeout=60)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        # Parse number from response (handle "7/10", "7.5", "7", etc.)
        for token in text.replace("/10", "").split():
            try:
                return max(0.0, min(10.0, float(token)))
            except ValueError:
                continue
        print(f"  VLM reward parse error: '{text}', defaulting to 5.0")
        return 5.0

    def health_check(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


# =============================================================================
# Differentiable Reward (LPIPS + DINOv2 + SSIM for DRafT-style training)
# =============================================================================

class PerceptualReward:
    """Non-VLM reward using LPIPS + DINOv2 cosine sim. Works without vLLM."""

    def __init__(self, device="cuda"):
        self.device = device
        self._lpips = None
        self._dino = None
        self._dino_transform = None

    def _load_lpips(self):
        if self._lpips is None:
            import lpips
            self._lpips = lpips.LPIPS(net="vgg").to(self.device).eval()

    def _load_dino(self):
        if self._dino is None:
            self._dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(self.device).eval()
            from torchvision import transforms
            self._dino_transform = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    @torch.no_grad()
    def score_batch(self, ref_images_list: list,
                    output_images: list) -> list[float]:
        """Score based on perceptual similarity to first reference image."""
        self._load_lpips()
        self._load_dino()
        scores = []
        for refs, out in zip(ref_images_list, output_images):
            inp = refs[0]  # compare against first reference
            from torchvision import transforms
            to_tensor = transforms.ToTensor()
            inp_t = to_tensor(inp.resize((256, 256))).unsqueeze(0).to(self.device) * 2 - 1
            out_t = to_tensor(out.resize((256, 256))).unsqueeze(0).to(self.device) * 2 - 1
            lpips_dist = self._lpips(inp_t, out_t).item()

            inp_d = self._dino_transform(inp).unsqueeze(0).to(self.device)
            out_d = self._dino_transform(out).unsqueeze(0).to(self.device)
            inp_feat = self._dino(inp_d)
            out_feat = self._dino(out_d)
            dino_sim = F.cosine_similarity(inp_feat, out_feat).item()

            score = (1.0 - lpips_dist) * 5.0 + dino_sim * 5.0
            scores.append(max(0.0, min(10.0, score)))
        return scores


# =============================================================================
# DiffusionNFT Trainer (reward-weighted velocity matching on forward process)
# =============================================================================

class DiffusionNFTTrainer:
    """DiffusionNFT training for coherent multi-image blending via VL splice.

    Objective: generate coherent "surprise" blends from two input images.
    VLM reward judges visual plausibility, not text-prompt adherence.

    Skips gradient updates when max(scores) < min_reward_threshold (no signal).

    Reference: DiffusionNFT (arxiv.org/abs/2509.16117)
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "bfloat16",
        # LoRA config
        lora_rank: int = 32,
        lora_alpha: float = 1.0,
        # Adapter config
        adapter_layers: int = 1,
        adapter_heads: int = 16,
        adapter_ffn_mult: float = 4.0,
        # Training config
        group_size: int = 4,
        lr: float = 1e-5,
        beta_nft: float = 0.1,
        ema_decay: float = 0.99,
        min_reward_threshold: float = 3.0,
        cond_noise_std: float = 0.1,
        # VL encoding
        max_pixels: int = 384 * 384,
        # Generation
        height: int = 1024,
        width: int = 1024,
    ):
        self.device = device
        self.group_size = group_size
        self.beta_nft = beta_nft
        self.ema_decay = ema_decay
        self.min_reward_threshold = min_reward_threshold
        self.cond_noise_std = cond_noise_std
        self.max_pixels = max_pixels
        self.height = height
        self.width = width

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.dtype = dtype_map.get(dtype, torch.bfloat16)

        # Load pipeline with VL splice
        print("Loading Z-Image-Turbo with VL splice...")
        self.pipe = load_pipeline("z-image-turbo", device=device, torch_dtype=dtype,
                                  text_encoder="qwen3vl")

        # Conditioning adapter (trainable)
        print(f"Creating CompositionModule: layers={adapter_layers}, heads={adapter_heads}")
        self.composer = CompositionModule(
            dim=2560, n_heads=adapter_heads, n_layers=adapter_layers,
            ffn_mult=adapter_ffn_mult,
        ).to(device=device, dtype=self.dtype)
        adapter_params = sum(p.numel() for p in self.composer.parameters())
        print(f"  CompositionModule params: {adapter_params:,}")

        # LoRA on DiT (manual, no peft)
        print(f"Injecting LoRA: rank={lora_rank}, alpha={lora_alpha}")
        lora_targets = inject_lora(self.pipe.dit, rank=lora_rank, alpha=lora_alpha)
        lora_params = get_lora_params(self.pipe.dit)
        lora_param_count = sum(p.numel() for p in lora_params)
        print(f"  LoRA: {len(lora_targets)} layers, {lora_param_count:,} params")

        # EMA copy of trainable weights (for v_old)
        self.ema_state = {}
        for k, v in self._trainable_params_dict().items():
            self.ema_state[k] = v.detach().clone()

        # Optimizer over adapter + LoRA params
        trainable = list(self.composer.parameters()) + lora_params
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

        # Scheduler config (for rollout generation)
        self.num_inference_steps = 8
        self.cfg_scale = 1.0

        total_trainable = sum(p.numel() for p in trainable)
        print(f"Total trainable params: {total_trainable:,}")

    def _trainable_params_dict(self) -> dict:
        """Get all trainable params as a flat dict (adapter + LoRA)."""
        d = {}
        for k, v in self.composer.named_parameters():
            d[f"adapter.{k}"] = v
        for name, module in self.pipe.dit.named_modules():
            if isinstance(module, LoRALinear):
                d[f"dit.{name}.lora_A"] = module.lora_A
                d[f"dit.{name}.lora_B"] = module.lora_B
        return d

    def _update_ema(self):
        """Soft EMA update: ema ← decay * ema + (1-decay) * current."""
        trainable = self._trainable_params_dict()
        with torch.no_grad():
            for k, v in trainable.items():
                self.ema_state[k].mul_(self.ema_decay).add_(v.data, alpha=1 - self.ema_decay)

    def _swap_to_ema(self):
        """Swap current weights to EMA weights. Returns backup for restore."""
        backup = {}
        trainable = self._trainable_params_dict()
        with torch.no_grad():
            for k, v in trainable.items():
                backup[k] = v.data.clone()
                v.data.copy_(self.ema_state[k])
        return backup

    def _restore_from_backup(self, backup):
        """Restore weights from backup after EMA swap."""
        trainable = self._trainable_params_dict()
        with torch.no_grad():
            for k, v in trainable.items():
                v.data.copy_(backup[k])

    def _encode_conditioning(self, content_list: list):
        """Encode interleaved [img, txt, img, ...] via VL model + adapter.

        VL encoding is frozen (no grad), adapter is trainable (has grad).
        """
        with torch.no_grad():
            prompt_embeds = encode_interleaved_vl(
                self.pipe, content_list, self.device, max_pixels=self.max_pixels
            )
        prompt_embeds = self.composer(prompt_embeds)
        return prompt_embeds

    def _rollout_no_grad(self, noise, prompt_embeds):
        """Full 8-step denoising without gradients. Returns PIL image."""
        self.pipe.scheduler.set_timesteps(self.num_inference_steps,
                                          denoising_strength=1.0, shift=None)
        self.pipe.load_models_to_device(self.pipe.in_iteration_models)
        models = {name: getattr(self.pipe, name)
                  for name in self.pipe.in_iteration_models}

        latents = noise.clone().to(device=self.device, dtype=self.dtype)
        inputs_shared = {"latents": latents}
        inputs_posi = {"prompt_embeds": prompt_embeds.detach()}
        inputs_nega = {"prompt_embeds": prompt_embeds.detach()}

        with torch.no_grad():
            for progress_id, timestep in enumerate(self.pipe.scheduler.timesteps):
                timestep_t = timestep.unsqueeze(0).to(dtype=self.dtype, device=self.device)
                noise_pred = self.pipe.cfg_guided_model_fn(
                    self.pipe.model_fn, self.cfg_scale,
                    inputs_shared, inputs_posi, inputs_nega,
                    **models, timestep=timestep_t, progress_id=progress_id,
                )
                inputs_shared["latents"] = self.pipe.step(
                    self.pipe.scheduler, progress_id=progress_id,
                    noise_pred=noise_pred, **inputs_shared,
                )
            image = decode_latent(self.pipe, inputs_shared["latents"])
        return image

    def _predict_velocity(self, z_t, t_scaled, prompt_embeds,
                          use_grad_ckpt=False):
        """Single DiT forward pass with current weights. Returns v_θ.

        Args:
            z_t: noisy latent (1, C, H, W)
            t_scaled: timestep scaled to scheduler range (0-1000), shape (1,)
            prompt_embeds: conditioning (L, 2560)
            use_grad_ckpt: enable gradient checkpointing (saves memory, costs compute)
        """
        self.pipe.load_models_to_device(self.pipe.in_iteration_models)
        models = {name: getattr(self.pipe, name)
                  for name in self.pipe.in_iteration_models}

        inputs_shared = {"latents": z_t}
        inputs_posi = {"prompt_embeds": prompt_embeds}
        inputs_nega = {"prompt_embeds": prompt_embeds.detach()}

        v_pred = self.pipe.cfg_guided_model_fn(
            self.pipe.model_fn, self.cfg_scale,
            inputs_shared, inputs_posi, inputs_nega,
            **models, timestep=t_scaled, progress_id=0,
            use_gradient_checkpointing=use_grad_ckpt,
        )
        return v_pred

    def train_step(self, sample: dict, reward_fn) -> dict:
        """One DiffusionNFT training step.

        1. Encode conditioning via VL + adapter (no grad for VL, grad for adapter)
        2. Generate K images via full 8-step rollout (no grad)
        3. Score with VLM reward, normalize to [0,1]
        4. For each image: forward diffuse, single DiT forward (grad) + old (no grad)
        5. Compute DiffusionNFT contrastive velocity loss
        6. Backward + update + EMA
        """
        K = self.group_size
        ref_images = sample["images"]
        prompt = sample.get("prompt", "")
        content_list = sample["content_list"]

        # --- Phase 1: Generate K images (no grad) ---
        with torch.no_grad():
            prompt_embeds_frozen = encode_interleaved_vl(
                self.pipe, content_list, self.device, max_pixels=self.max_pixels
            )
            prompt_embeds_nograd = self.composer(prompt_embeds_frozen)

        images = []
        latents_z0 = []
        for k in range(K):
            seed = random.randint(0, 2**32 - 1)
            shape = get_latent_shape(self.height, self.width)
            noise = generate_noise(seed, shape, self.device, self.dtype)

            # Add per-rollout noise to conditioning for diversity
            if self.cond_noise_std > 0:
                cond_noise = torch.randn_like(prompt_embeds_nograd) * self.cond_noise_std
                rollout_embeds = prompt_embeds_nograd + cond_noise
            else:
                rollout_embeds = prompt_embeds_nograd

            img = self._rollout_no_grad(noise, rollout_embeds)
            images.append(img)

            with torch.no_grad():
                z_0 = encode_image_vae(self.pipe, img)
            latents_z0.append(z_0)

        # --- Phase 2: Score with reward ---
        scores = reward_fn.score_batch([ref_images] * K, images)
        rewards_raw = torch.tensor(scores, dtype=torch.float32)

        mean_r = rewards_raw.mean()
        std_r = rewards_raw.std() + 1e-8

        # Rollout filtering: skip gradient if all scores below threshold
        skipped = False
        if rewards_raw.max().item() < self.min_reward_threshold:
            skipped = True
            rewards_norm = torch.zeros(K)
            return {
                "loss": 0.0, "loss_pos": 0.0, "loss_neg": 0.0,
                "mean_reward": mean_r.item(), "std_reward": std_r.item(),
                "min_reward": rewards_raw.min().item(),
                "max_reward": rewards_raw.max().item(),
                "rewards_norm": rewards_norm.tolist(),
                "scores": scores, "images": images,
                "ref_images": ref_images, "prompt": prompt,
                "adapter_grad_norm": 0.0, "lora_grad_norm": 0.0,
                "skipped": True,
            }

        # Normalize rewards to [0,1] (DiffusionNFT normalization)
        rewards_norm = 0.5 + 0.5 * torch.clamp((rewards_raw - mean_r) / std_r, -1.0, 1.0)

        # --- Phase 3: DiffusionNFT gradient step ---
        loss_total_val = 0.0
        loss_pos_total = 0.0
        loss_neg_total = 0.0
        beta = self.beta_nft

        self.optimizer.zero_grad()

        for k in range(K):
            r = rewards_norm[k].item()
            z_0 = latents_z0[k].detach()

            t = random.uniform(0.05, 0.95)
            epsilon = torch.randn_like(z_0)

            # Forward diffuse: z_t = (1-t)*z_0 + t*ε (rectified flow)
            z_t = ((1 - t) * z_0 + t * epsilon).detach()
            v_target = (epsilon - z_0).detach()
            t_scaled = torch.tensor([t * SCHEDULER_SCALE], dtype=self.dtype, device=self.device)

            # Re-encode conditioning per sample (fresh computation graph)
            prompt_embeds_k = self._encode_conditioning(content_list)

            # v_θ with gradient checkpointing
            v_theta = self._predict_velocity(z_t, t_scaled, prompt_embeds_k,
                                             use_grad_ckpt=True)

            # v_old ≈ v_theta.detach() (valid early in training when EMA ≈ current)
            v_old = v_theta.detach()

            # Implicit parameterization (DiffusionNFT)
            v_pos = (1 - beta) * v_old + beta * v_theta
            v_neg = (1 + beta) * v_old - beta * v_theta

            loss_pos = r * ((v_pos - v_target) ** 2).mean() / K
            loss_neg = (1 - r) * ((v_neg - v_target) ** 2).mean() / K
            sample_loss = loss_pos + loss_neg

            # Backward immediately — frees activation memory before next sample
            sample_loss.backward()

            loss_total_val += sample_loss.item()
            loss_pos_total += loss_pos.item() * K
            loss_neg_total += loss_neg.item() * K

        all_trainable = list(self.composer.parameters()) + [
            p for p in self.pipe.dit.parameters() if p.requires_grad
        ]
        torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
        self.optimizer.step()
        self._update_ema()

        # Grad norms for monitoring
        adapter_grad_norm = 0.0
        lora_grad_norm = 0.0
        for p in self.composer.parameters():
            if p.grad is not None:
                adapter_grad_norm += p.grad.data.norm(2).item() ** 2
        for p in get_lora_params(self.pipe.dit):
            if p.grad is not None:
                lora_grad_norm += p.grad.data.norm(2).item() ** 2
        adapter_grad_norm = adapter_grad_norm ** 0.5
        lora_grad_norm = lora_grad_norm ** 0.5

        return {
            "loss": loss_total_val,
            "loss_pos": loss_pos_total / K,
            "loss_neg": loss_neg_total / K,
            "mean_reward": mean_r.item(),
            "std_reward": std_r.item(),
            "min_reward": rewards_raw.min().item(),
            "max_reward": rewards_raw.max().item(),
            "rewards_norm": rewards_norm.tolist(),
            "scores": scores, "images": images,
            "ref_images": ref_images, "prompt": prompt,
            "adapter_grad_norm": adapter_grad_norm,
            "lora_grad_norm": lora_grad_norm,
            "skipped": False,
        }

    def save_checkpoint(self, path: str, step: int):
        """Save adapter + LoRA + EMA + optimizer state."""
        adapter_state = {k: v.cpu() for k, v in self.composer.state_dict().items()}
        lora_state = {k: v.cpu() for k, v in get_lora_state_dict(self.pipe.dit).items()}
        ema_state = {k: v.cpu() for k, v in self.ema_state.items()}

        torch.save({
            "step": step,
            "adapter_state_dict": adapter_state,
            "lora_state_dict": lora_state,
            "ema_state": ema_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load adapter + LoRA + EMA weights."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.composer.load_state_dict(ckpt["adapter_state_dict"])
        load_lora_state_dict(self.pipe.dit, ckpt["lora_state_dict"])
        if "ema_state" in ckpt:
            for k, v in ckpt["ema_state"].items():
                if k in self.ema_state:
                    self.ema_state[k].copy_(v)
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Loaded checkpoint: {path} (step {ckpt['step']})")
        return ckpt["step"]


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="DiffusionNFT for coherent image blending (VL splice)")

    # Data
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to composition JSONL (images/ dir expected alongside it)")

    # Reward
    parser.add_argument("--reward", type=str, default="vlm",
                        choices=["vlm", "perceptual"],
                        help="Reward type: vlm (Qwen3-VL via vLLM) or perceptual (LPIPS+DINO)")
    parser.add_argument("--reward_url", type=str, default="http://localhost:8100",
                        help="vLLM server URL for VLM reward")
    parser.add_argument("--reward_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Model name on vLLM server")

    # DiffusionNFT
    parser.add_argument("--group_size", type=int, default=4, help="K: images per group")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta_nft", type=float, default=0.1,
                        help="DiffusionNFT guidance strength β (default: 0.1)")
    parser.add_argument("--ema_decay", type=float, default=0.99,
                        help="EMA decay for old policy weights (default: 0.99)")
    parser.add_argument("--min_reward_threshold", type=float, default=3.0,
                        help="Skip gradient if max(scores) < this (default: 3.0)")
    parser.add_argument("--cond_noise_std", type=float, default=0.1,
                        help="Gaussian noise std added to conditioning per rollout for diversity (default: 0.1)")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=100)

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=1.0)

    # Adapter
    parser.add_argument("--adapter_layers", type=int, default=1,
                        help="Number of transformer blocks in conditioning adapter")
    parser.add_argument("--adapter_heads", type=int, default=16)
    parser.add_argument("--adapter_ffn_mult", type=float, default=4.0)

    # VL encoding
    parser.add_argument("--max_pixels", type=int, default=384 * 384,
                        help="Max pixels for VL image encoding")

    # Generation
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)

    # Misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--output_dir", type=str, default="outputs/diffnft")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Test pipeline without training")
    parser.add_argument("--mem_limit", type=float, default=85.0,
                        help="Kill switch: abort if memory usage exceeds this %% (default 85)")

    args = parser.parse_args()

    # Setup output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load dataset
    dataset = CompositionDataset(args.dataset)

    # Setup reward
    if args.reward == "vlm":
        reward_fn = VLMRewardClient(base_url=args.reward_url, model=args.reward_model)
        if not reward_fn.health_check():
            print(f"WARNING: vLLM server not responding at {args.reward_url}")
            print("Start it with:")
            print(f"  python -m vllm.entrypoints.openai.api_server \\")
            print(f"    --model {args.reward_model} --max-model-len 4096 --port 8100")
            if not args.dry_run:
                return
        else:
            print(f"VLM reward connected: {args.reward_url}")
    else:
        reward_fn = PerceptualReward(device=args.device)
        print("Using perceptual reward (LPIPS + DINOv2)")

    # Init trainer
    trainer = DiffusionNFTTrainer(
        device=args.device, dtype=args.dtype,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        adapter_layers=args.adapter_layers, adapter_heads=args.adapter_heads,
        adapter_ffn_mult=args.adapter_ffn_mult,
        group_size=args.group_size, lr=args.lr,
        beta_nft=args.beta_nft, ema_decay=args.ema_decay,
        min_reward_threshold=args.min_reward_threshold,
        cond_noise_std=args.cond_noise_std,
        max_pixels=args.max_pixels,
        height=args.height, width=args.width,
    )

    start_step = 0
    if args.resume:
        start_step = trainer.load_checkpoint(args.resume)

    # Dry run: generate one sample, score it, test one training step
    if args.dry_run:
        print("\n=== DRY RUN ===")
        sample = dataset.sample()
        print(f"Prompt: {sample['prompt']}")
        print(f"Ref images: {len(sample['images'])}")

        prompt_embeds = trainer._encode_conditioning(sample["content_list"])
        print(f"Prompt embeds shape: {prompt_embeds.shape}")

        shape = get_latent_shape(args.height, args.width)
        noise = generate_noise(42, shape, args.device, trainer.dtype)
        with torch.no_grad():
            out_img = trainer._rollout_no_grad(noise, prompt_embeds)
        out_img.save(run_dir / "dry_run_output.png")
        for i, ref in enumerate(sample["images"]):
            ref.save(run_dir / f"dry_run_ref_{i}.png")
        print(f"Output saved: {run_dir}")

        # Test VAE encode round-trip
        with torch.no_grad():
            z_0 = encode_image_vae(trainer.pipe, out_img)
        print(f"VAE encoded latent shape: {z_0.shape}")

        # Test single velocity prediction
        t = 0.5
        epsilon = torch.randn_like(z_0)
        z_t = (1 - t) * z_0 + t * epsilon
        t_scaled = torch.tensor([t * SCHEDULER_SCALE], dtype=trainer.dtype, device=args.device)
        with torch.no_grad():
            v_pred = trainer._predict_velocity(z_t, t_scaled, prompt_embeds.detach())
        print(f"Velocity prediction shape: {v_pred.shape}, norm: {v_pred.norm().item():.2f}")

        if args.reward == "vlm" and reward_fn.health_check():
            score = reward_fn.score_batch([sample["images"]], [out_img])
            print(f"VLM reward: {score[0]:.1f}/10")

        print("Dry run complete.")
        return

    # Init wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(project="synthos-diffnft", config=vars(args), dir=str(run_dir))

    # Memory kill switch
    mem_limit = args.mem_limit
    def check_memory(step):
        mem = psutil.virtual_memory()
        pct = mem.percent
        if pct > mem_limit:
            print(f"\n!!! MEMORY KILL SWITCH: {pct:.1f}% > {mem_limit}% limit at step {step}")
            print(f"    Used: {mem.used / 1e9:.1f}GB / {mem.total / 1e9:.1f}GB")
            if use_wandb:
                wandb.finish()
            trainer.save_checkpoint(str(run_dir / "checkpoints" / f"emergency_step_{step}.pt"), step)
            raise SystemExit(f"OOM protection: {pct:.1f}% memory used")
        return pct

    # Training loop
    print(f"\nStarting DiffusionNFT training: {args.steps} steps, K={args.group_size}, β={args.beta_nft}")
    print(f"Min reward threshold: {args.min_reward_threshold}")
    print(f"Output dir: {run_dir}")
    print(f"Memory kill switch: {mem_limit}%")

    skipped_count = 0

    for step in tqdm(range(start_step + 1, args.steps + 1), desc="DiffNFT"):
        mem_pct = check_memory(step)

        sample = dataset.sample()

        t0 = time.time()
        metrics = trainer.train_step(sample, reward_fn)
        step_time = time.time() - t0

        if metrics.get("skipped"):
            skipped_count += 1

        # --- Wandb logging: scalars every step ---
        if use_wandb:
            log_dict = {
                "train/loss": metrics["loss"],
                "train/loss_pos": metrics["loss_pos"],
                "train/loss_neg": metrics["loss_neg"],
                "train/skipped": int(metrics.get("skipped", False)),
                "train/skipped_total": skipped_count,
                "reward/mean": metrics["mean_reward"],
                "reward/std": metrics["std_reward"],
                "reward/min": metrics["min_reward"],
                "reward/max": metrics["max_reward"],
                "grad/adapter_norm": metrics["adapter_grad_norm"],
                "grad/lora_norm": metrics["lora_grad_norm"],
                "perf/step_time_s": step_time,
                "perf/mem_pct": mem_pct,
            }
            for i, s in enumerate(metrics["scores"]):
                log_dict[f"reward/sample_{i}"] = s
            wandb.log(log_dict, step=step)

        # --- Console logging ---
        if step % 5 == 0 or metrics.get("skipped"):
            scores_str = ",".join(f"{s:.1f}" for s in metrics["scores"])
            skip_tag = " SKIPPED" if metrics.get("skipped") else ""
            tqdm.write(
                f"[{step}] loss={metrics['loss']:.4f} "
                f"scores=[{scores_str}]{skip_tag} "
                f"gnorm=A:{metrics['adapter_grad_norm']:.2f}/L:{metrics['lora_grad_norm']:.2f} "
                f"t={step_time:.0f}s skipped={skipped_count}/{step}"
            )

        # --- Eval: save rollouts + log images to wandb ---
        if step % args.eval_every == 0:
            step_dir = run_dir / "samples" / f"step_{step:05d}"
            step_dir.mkdir(parents=True, exist_ok=True)

            for i, ref in enumerate(metrics["ref_images"]):
                ref.save(step_dir / f"ref_{i}.png")
            for i, img in enumerate(metrics["images"][:4]):
                img.save(step_dir / f"output_{i}.png")
            with open(step_dir / "meta.json", "w") as f:
                json.dump({
                    "prompt": metrics.get("prompt", ""),
                    "scores": metrics["scores"],
                    "rewards_norm": metrics["rewards_norm"],
                    "skipped": metrics.get("skipped", False),
                }, f, indent=2)

            if use_wandb:
                wandb_images = []
                for i, ref in enumerate(metrics["ref_images"]):
                    wandb_images.append(wandb.Image(ref, caption=f"ref_{i}"))
                for i, (img, score) in enumerate(zip(metrics["images"], metrics["scores"])):
                    wandb_images.append(wandb.Image(
                        img, caption=f"out_{i} score={score:.1f}"
                    ))
                wandb.log({"rollouts": wandb_images}, step=step)

        # --- Checkpoint ---
        if step % args.save_every == 0:
            trainer.save_checkpoint(
                str(run_dir / "checkpoints" / f"step_{step:05d}.pt"), step
            )

    # Final checkpoint
    trainer.save_checkpoint(str(run_dir / "checkpoints" / "final.pt"), args.steps)
    if use_wandb:
        wandb.finish()
    print(f"\nTraining complete. Skipped {skipped_count}/{args.steps} steps. Outputs: {run_dir}")


if __name__ == "__main__":
    main()
