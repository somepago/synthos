#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) training for Z-Image-Turbo img2img.

Core idea:
  1. Sample a prompt + reference image
  2. Generate a GROUP of K images from the policy (LoRA'd Z-Image-Turbo)
     conditioned on the reference image via SigLip features
  3. Score each image with a reward model (HPS, CLIP similarity, etc.)
  4. Compute advantages using group-relative baseline (no separate critic needed)
  5. Update policy with clipped surrogate objective (PPO-style)

Adapting GRPO from language (DeepSeek-R1) to diffusion models:
  - "tokens" → denoising steps (each step is an "action")
  - "policy" → the DiT velocity predictions at each timestep
  - "reward" → image quality/similarity scores on final decoded image

Image conditioning:
  - Reference image is encoded via SigLip to spatial features (H', W', 1152)
  - Features are passed to DiT as image_embeds at each denoising step
  - This is the native Z-Image edit_image pathway

Usage:
    # img2img GRPO (optimize for quality)
    python train_grpo.py --reward hps --input_images data/inputs/

    # img2img GRPO (optimize for similarity + quality)
    python train_grpo.py --reward hps+clip_sim --input_images data/inputs/

    # text2image GRPO (no image conditioning, pure quality optimization)
    python train_grpo.py --reward hps
"""

from src import env_setup  # noqa: F401

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import wandb

from src.model_utils import load_pipeline
from src.constants import SCHEDULER_SCALE, TURBO_SCHEDULER_TIMESTEPS, DEFAULT_PROMPTS
from src.diffusion import (
    get_latent_shape,
    generate_noise,
    decode_latent,
    encode_image_siglip,
    _prepare_diffusion,
)


# =============================================================================
# Reward functions
# =============================================================================

class HPSReward:
    """Human Preference Score v2 as reward signal."""

    def __init__(self, device="cuda"):
        import hpsv2
        self.device = device
        self._score_fn = hpsv2.score

    @torch.no_grad()
    def score(self, images: list[Image.Image], prompts: list[str]) -> list[float]:
        scores = []
        for img, prompt in zip(images, prompts):
            s = self._score_fn(img, prompt)
            scores.append(float(s[0]) if isinstance(s, (list, tuple)) else float(s))
        return scores


class CLIPSimReward:
    """CLIP-based similarity between generated image and reference image."""

    def __init__(self, device="cuda"):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device,
        )
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def score(self, images: list[Image.Image], references: list[Image.Image]) -> list[float]:
        scores = []
        for img, ref in zip(images, references):
            img_t = self.preprocess(img).unsqueeze(0).to(self.device)
            ref_t = self.preprocess(ref).unsqueeze(0).to(self.device)
            img_feat = self.model.encode_image(img_t)
            ref_feat = self.model.encode_image(ref_t)
            sim = F.cosine_similarity(img_feat, ref_feat).item()
            scores.append(sim)
        return scores


def build_reward_fn(reward_name: str, device: str = "cuda"):
    """Build reward function(s) by name. Supports 'hps', 'clip_sim', 'hps+clip_sim'."""
    rewards = {}
    for name in reward_name.split("+"):
        name = name.strip()
        if name == "hps":
            rewards[name] = HPSReward(device)
        elif name == "clip_sim":
            rewards[name] = CLIPSimReward(device)
        else:
            raise ValueError(f"Unknown reward: {name}")
    return rewards


# =============================================================================
# GRPO Trainer
# =============================================================================

class GRPOTrainer:
    """GRPO training for Z-Image-Turbo with LoRA and SigLip image conditioning."""

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "bfloat16",
        lora_rank: int = 64,
        lora_alpha: float = 1.0,
        group_size: int = 4,
        lr: float = 1e-5,
        clip_epsilon: float = 0.2,
        beta_kl: float = 0.01,
        use_img2img: bool = False,
        text_encoder: str = "qwen3",
    ):
        self.device = device
        self.dtype = dtype
        self.group_size = group_size
        self.clip_epsilon = clip_epsilon
        self.beta_kl = beta_kl
        self.use_img2img = use_img2img

        # Load policy model (with SigLip if img2img)
        model_key = "z-image-turbo-img2img" if use_img2img else "z-image-turbo"
        print(f"Loading policy model ({model_key})...")
        self.pipe = load_pipeline(model_key, device=device, torch_dtype=dtype,
                                  text_encoder=text_encoder)

        # Verify SigLip loaded if img2img
        if use_img2img:
            if self.pipe.image_encoder is None:
                raise RuntimeError("SigLip image encoder failed to load.")
            if self.pipe.dit.siglip_embedder is None:
                raise RuntimeError("SigLip projection layers not added to DiT.")

        # Add LoRA to existing DiT layers (excludes siglip layers which are already trainable)
        self._add_lora(lora_rank, lora_alpha)

        # Ensure siglip projection layers are trainable (they already are from init,
        # but LoRA injection may have frozen them)
        if use_img2img:
            for name, param in self.pipe.dit.named_parameters():
                if "siglip" in name:
                    param.requires_grad_(True)

        # Reference model: store initial state of ALL trainable params
        # (LoRA weights = zeros, siglip projection = random init)
        self.ref_lora_state = {
            k: v.clone() for k, v in self.pipe.dit.named_parameters() if v.requires_grad
        }

        # Optimizer over all trainable params (LoRA + siglip projection)
        trainable_params = [p for p in self.pipe.dit.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

        # Scheduler params
        self.num_inference_steps = 8
        self.cfg_scale = 1.0

    def _add_lora(self, lora_rank: int, lora_alpha: float):
        """Add LoRA adapter to DiT (skips siglip layers — those are trained directly)."""
        from peft import LoraConfig, inject_adapter_in_model

        target_modules = []
        for name, module in self.pipe.dit.named_modules():
            # Skip siglip layers — trained directly, not via LoRA
            if "siglip" in name:
                continue
            if isinstance(module, nn.Linear):
                if min(module.weight.shape) >= 512:
                    target_modules.append(name)
                elif "adaLN_modulation" in name:
                    target_modules.append(name)

        print(f"  LoRA: rank={lora_rank}, alpha={lora_alpha}, {len(target_modules)} target modules")

        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha,
            target_modules=target_modules, init_lora_weights=True,
        )
        self.pipe.dit = inject_adapter_in_model(lora_config, self.pipe.dit)

        trainable = sum(p.numel() for p in self.pipe.dit.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.pipe.dit.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def _get_prompt_embeds(self, prompt: str):
        """Encode prompt and negative prompt."""
        from diffsynth.pipelines.z_image import ZImageUnit_PromptEmbedder
        self.pipe.load_models_to_device(["text_encoder"])
        embedder = ZImageUnit_PromptEmbedder()
        with torch.no_grad():
            prompt_embeds = embedder.encode_prompt(self.pipe, prompt, self.device)
            negative_embeds = embedder.encode_prompt(self.pipe, "", self.device)

        return prompt_embeds, negative_embeds

    def _encode_edit_image(self, image: Image.Image) -> list[torch.Tensor]:
        """Encode reference image via SigLip for image conditioning.

        Returns list of image_embeds matching pipeline's expected format.
        """
        with torch.no_grad():
            image_embeds = encode_image_siglip(self.pipe, image)
        return [image_embeds]

    def _rollout_single(self, noise, prompt_embeds, negative_embeds, image_embeds=None):
        """Run a single denoising trajectory with optional SigLip image conditioning.

        For GRPO we need:
          - The final image (for reward computation)
          - Log-probability of the trajectory under current policy

        We approximate log-prob as negative L2 of the velocity prediction
        (treats the model output as a Gaussian mean with fixed variance).

        Args:
            image_embeds: Optional SigLip features from _encode_edit_image().
                         Passed to DiT as image conditioning at each step.
        """
        self.pipe.scheduler.set_timesteps(self.num_inference_steps, denoising_strength=1.0, shift=None)
        self.pipe.load_models_to_device(self.pipe.in_iteration_models)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}

        latents = noise.clone().to(device=self.pipe.device, dtype=self.pipe.torch_dtype)
        inputs_shared = {"latents": latents}
        inputs_posi = {"prompt_embeds": prompt_embeds}
        inputs_nega = {"prompt_embeds": negative_embeds}

        # Pass SigLip features as image_embeds if available
        if image_embeds is not None:
            inputs_posi["image_embeds"] = image_embeds
            inputs_nega["image_embeds"] = image_embeds

        log_probs = []

        for progress_id, timestep in enumerate(self.pipe.scheduler.timesteps):
            timestep_t = timestep.unsqueeze(0).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

            # Get velocity prediction (this is the "action")
            noise_pred = self.pipe.cfg_guided_model_fn(
                self.pipe.model_fn, self.cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep_t, progress_id=progress_id,
            )

            # Step (get next latent)
            inputs_shared["latents"] = self.pipe.step(
                self.pipe.scheduler, progress_id=progress_id,
                noise_pred=noise_pred, **inputs_shared,
            )

            # Approximate log-prob: negative L2 of the velocity prediction
            # (treats model output as Gaussian mean with fixed variance)
            log_prob = -0.5 * (noise_pred ** 2).sum()
            log_probs.append(log_prob)

        # Decode final image
        image = decode_latent(self.pipe, inputs_shared["latents"])

        total_log_prob = sum(log_probs)
        return image, total_log_prob

    def _compute_ref_log_probs(self, noise, prompt_embeds, negative_embeds, image_embeds=None):
        """Compute log-probs under the reference (frozen) policy.

        Temporarily swaps in reference LoRA weights, runs forward pass,
        then restores current weights.
        """
        # Save current weights
        current_state = {
            k: v.clone() for k, v in self.pipe.dit.named_parameters() if v.requires_grad
        }

        # Load reference weights
        with torch.no_grad():
            for name, param in self.pipe.dit.named_parameters():
                if name in self.ref_lora_state:
                    param.copy_(self.ref_lora_state[name])

        # Forward pass under reference
        with torch.no_grad():
            _, ref_log_prob = self._rollout_single(noise, prompt_embeds, negative_embeds, image_embeds)

        # Restore current weights
        with torch.no_grad():
            for name, param in self.pipe.dit.named_parameters():
                if name in current_state:
                    param.copy_(current_state[name])

        return ref_log_prob

    def train_step(self, prompt: str, reward_fns: dict, input_image: Image.Image = None):
        """One GRPO training step.

        1. Encode reference image via SigLip (if img2img)
        2. Generate K images (group) from current policy
        3. Score with reward
        4. Compute group-relative advantages
        5. Compute clipped surrogate loss with KL penalty
        6. Backward + optimize

        Returns dict of metrics.
        """
        prompt_embeds, negative_embeds = self._get_prompt_embeds(prompt)

        # Encode reference image via SigLip if provided
        image_embeds = None
        if input_image is not None:
            image_embeds = self._encode_edit_image(input_image)

        K = self.group_size
        height, width = 512, 512

        # Generate group of K images
        images = []
        log_probs = []
        noises = []

        for k in range(K):
            seed = random.randint(0, 2**32 - 1)
            shape = get_latent_shape(height, width)
            noise = generate_noise(seed, shape, self.pipe.device, self.pipe.torch_dtype)
            noises.append(noise)

            image, log_prob = self._rollout_single(noise, prompt_embeds, negative_embeds, image_embeds)
            images.append(image)
            log_probs.append(log_prob)

        # Compute rewards
        all_rewards = {}
        total_rewards = torch.zeros(K)
        for rname, rfn in reward_fns.items():
            if rname == "clip_sim" and input_image is not None:
                scores = rfn.score(images, [input_image] * K)
            elif rname == "hps":
                scores = rfn.score(images, [prompt] * K)
            else:
                continue
            all_rewards[rname] = scores
            total_rewards += torch.tensor(scores)

        # Group-relative advantages: A_i = (r_i - mean(r)) / (std(r) + eps)
        mean_r = total_rewards.mean()
        std_r = total_rewards.std() + 1e-8
        advantages = (total_rewards - mean_r) / std_r

        # Compute policy loss with clipped surrogate + KL penalty
        policy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        kl_total = 0.0

        for k in range(K):
            advantage = advantages[k].item()

            # Recompute log-prob with gradients
            _, log_prob_k = self._rollout_single(noises[k], prompt_embeds, negative_embeds, image_embeds)

            # Reference log-prob (no gradients)
            ref_log_prob_k = self._compute_ref_log_probs(
                noises[k], prompt_embeds, negative_embeds, image_embeds,
            )

            # KL divergence approximation
            kl = (log_prob_k - ref_log_prob_k).detach()
            kl_total += kl.item()

            # Clipped surrogate
            ratio = torch.exp(log_prob_k - log_prob_k.detach())  # 1 on first iter
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            surrogate = torch.min(ratio * advantage, clipped_ratio * advantage)

            policy_loss = policy_loss - surrogate + self.beta_kl * kl

        policy_loss = policy_loss / K

        # Backward + optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.pipe.dit.parameters() if p.requires_grad], max_norm=1.0,
        )
        self.optimizer.step()

        return {
            "loss": policy_loss.item(),
            "mean_reward": mean_r.item(),
            "std_reward": std_r.item(),
            "kl": kl_total / K,
            "advantages": advantages.tolist(),
            **{f"reward_{k}": v for k, v in all_rewards.items()},
            "images": images,
        }

    def save_checkpoint(self, path: str, step: int):
        """Save LoRA weights + siglip projection weights + optimizer state."""
        lora_state = {}
        siglip_state = {}
        for k, v in self.pipe.dit.named_parameters():
            if not v.requires_grad:
                continue
            if "siglip" in k:
                siglip_state[k] = v.cpu()
            else:
                lora_state[k] = v.cpu()

        torch.save({
            "step": step,
            "lora_state_dict": lora_state,
            "siglip_state_dict": siglip_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"Saved checkpoint to {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GRPO training for Z-Image-Turbo")
    parser.add_argument("--reward", type=str, default="hps",
                        help="Reward function(s), e.g. 'hps', 'clip_sim', 'hps+clip_sim'")
    parser.add_argument("--group_size", type=int, default=4,
                        help="Number of images per group (K)")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of training steps")
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--beta_kl", type=float, default=0.01,
                        help="KL penalty coefficient")

    # Text encoder
    parser.add_argument("--text_encoder", type=str, default="qwen3",
                        choices=["qwen3", "qwen3vl"],
                        help="Text encoder: qwen3 (default), qwen3vl (VL splice for i2i)")

    # img2img
    parser.add_argument("--input_images", type=str, default=None,
                        help="Directory of input images for SigLip-conditioned img2img")

    # prompts
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="Text file with one prompt per line")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--output_dir", type=str, default="outputs/grpo")
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()

    # Setup output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "images").mkdir(exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load prompts
    if args.prompts_file:
        prompts = Path(args.prompts_file).read_text().strip().splitlines()
    else:
        prompts = DEFAULT_PROMPTS

    # Load input images if img2img
    input_images = None
    use_img2img = args.input_images is not None
    if use_img2img:
        img_dir = Path(args.input_images)
        input_images = [
            Image.open(p).convert("RGB")
            for p in sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
        ]
        print(f"Loaded {len(input_images)} input images for img2img")

    # Init wandb
    if not args.no_wandb:
        wandb.init(project="synthos-grpo", config=vars(args), dir=str(run_dir))

    # Build reward
    reward_fns = build_reward_fn(args.reward, args.device)
    print(f"Reward functions: {list(reward_fns.keys())}")

    # Init trainer
    trainer = GRPOTrainer(
        device=args.device, dtype=args.dtype,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        group_size=args.group_size, lr=args.lr,
        clip_epsilon=args.clip_epsilon, beta_kl=args.beta_kl,
        use_img2img=use_img2img, text_encoder=args.text_encoder,
    )

    # Training loop
    for step in tqdm(range(1, args.steps + 1), desc="GRPO Training"):
        prompt = random.choice(prompts)
        input_image = random.choice(input_images) if input_images else None

        metrics = trainer.train_step(prompt, reward_fns, input_image)

        # Log
        log_dict = {k: v for k, v in metrics.items() if k != "images"}
        if not args.no_wandb:
            wandb.log(log_dict, step=step)

        if step % 10 == 0:
            tqdm.write(
                f"[Step {step}] loss={metrics['loss']:.4f} "
                f"reward={metrics['mean_reward']:.4f}±{metrics['std_reward']:.4f} "
                f"kl={metrics['kl']:.4f}"
            )

        # Save sample images
        if step % args.eval_every == 0:
            for i, img in enumerate(metrics["images"][:4]):
                img.save(run_dir / "images" / f"step{step:05d}_sample{i}.png")

        # Save checkpoint
        if step % args.save_every == 0:
            trainer.save_checkpoint(
                str(run_dir / "checkpoints" / f"step_{step:05d}.pt"), step,
            )

    # Final save
    trainer.save_checkpoint(str(run_dir / "checkpoints" / "final.pt"), args.steps)
    print(f"\nTraining complete. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
