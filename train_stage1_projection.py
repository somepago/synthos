#!/usr/bin/env python3
"""
Stage 1: Train SigLip projection to text embedding space.

Trainable: siglip_projection (LayerNorm + Linear, 1152 → 2560, ~2.95M params)
Frozen: SigLip2 image encoder, DiT (all weights), text encoder, VAE

SigLip features are projected to text embedding space and concatenated with
text embeddings as additional caption tokens. The DiT processes them through
its standard (non-omni) text conditioning path.

Usage:
    python train_stage1_projection.py
    python train_stage1_projection.py --steps 5000 --lr 1e-4
"""

from src import env_setup  # noqa: F401

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import wandb

from src.model_utils import load_pipeline, get_defaults
from src.constants import SCHEDULER_SCALE, TURBO_SCHEDULER_TIMESTEPS, FLOW_MATCHING_SHIFT
from src.diffusion import (
    encode_image_vae,
    encode_image_siglip,
    run_img2img_siglip_caption,
)


# =============================================================================
# Dataset
# =============================================================================

class ArtDataset:
    """Dataset backed by downloaded.parquet + images on disk."""

    def __init__(self, data_dir: str, max_size: int = 768):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.max_size = max_size

        df = pd.read_parquet(self.data_dir / "downloaded.parquet")
        df = df[df["download_status"] == "ok"].reset_index(drop=True)

        self.samples = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
            if (self.image_dir / row["file_name"]).exists():
                self.samples.append({"file_name": row["file_name"], "text": row["text"]})

        print(f"Dataset: {len(self.samples)} images (from {len(df)} downloaded)")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _round_to_16(size, max_size):
        """Resize preserving aspect ratio, round dims to multiple of 16."""
        w, h = size
        scale = min(max_size / max(w, h), 1.0)
        w, h = round(w * scale), round(h * scale)
        return max(w // 16 * 16, 16), max(h // 16 * 16, 16)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(self.image_dir / s["file_name"]).convert("RGB")
        new_w, new_h = self._round_to_16(image.size, self.max_size)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        return image, s["text"]

    def random_sample(self):
        while True:
            try:
                return self[random.randint(0, len(self) - 1)]
            except Exception:
                continue



# =============================================================================
# Stage 1 Trainer
# =============================================================================

class Stage1Trainer:
    """Train SigLip → text-space projection with diffusion velocity matching."""

    def __init__(self, device="cuda", dtype="bfloat16", lr=1e-4,
                 warmup_steps=100, total_steps=5000, lr_min_ratio=0.01,
                 max_grad_norm=5.0, model="turbo", text_encoder="qwen3"):
        self.device = device
        self.model_type = model  # "turbo" or "base"

        model_key = "z-image-turbo-img2img" if model == "turbo" else "z-image-base-img2img"
        self.pipe = load_pipeline(model_key, device=device, torch_dtype=dtype,
                                  text_encoder=text_encoder)
        assert self.pipe.image_encoder is not None, "SigLip image encoder not loaded"
        assert hasattr(self.pipe, "siglip_projection"), "SigLip projection not created"
        # DiT should NOT have siglip_embedder — we use the non-omni path
        assert self.pipe.dit.siglip_embedder is None, "DiT should not have siglip layers"

        # Inference defaults for eval generation
        defaults = get_defaults(model_key)
        self.eval_num_steps = defaults["num_inference_steps"]
        self.eval_cfg_scale = defaults["cfg_scale"]

        # Freeze everything
        for param in self.pipe.dit.parameters():
            param.requires_grad_(False)

        # Only siglip_projection is trainable
        trainable_params = list(self.pipe.siglip_projection.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)
        print(f"Trainable: {trainable_count:,} params (siglip_projection)")

        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

        # LR schedule: linear warmup + cosine decay
        from torch.optim.lr_scheduler import LambdaLR
        def warmup_cosine_schedule(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return lr_min_ratio + (1.0 - lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        self.scheduler = LambdaLR(self.optimizer, warmup_cosine_schedule)

        self.pipe.scheduler.set_timesteps(1, denoising_strength=1.0, shift=None)
        # Turbo: sample from 8 fixed distilled timesteps
        # Base: sample t uniformly from [0, 1] (standard flow matching)
        self.timesteps = TURBO_SCHEDULER_TIMESTEPS if model == "turbo" else None
        self.shift = FLOW_MATCHING_SHIFT
        self.max_grad_norm = max_grad_norm
        self._cached_text_embeds = None

    def _compute_sigma(self, t: float) -> float:
        return self.shift * t / (1 + (self.shift - 1) * t)

    @torch.no_grad()
    def _get_text_embeds(self):
        """Encode empty prompt once and cache. Returns tensor (L, 2560)."""
        if self._cached_text_embeds is None:
            from diffsynth.pipelines.z_image import ZImageUnit_PromptEmbedder
            self.pipe.load_models_to_device(["text_encoder"])
            embedder = ZImageUnit_PromptEmbedder()
            embeds_list = embedder.encode_prompt(self.pipe, "", self.device)
            self._cached_text_embeds = embeds_list[0]  # (L, 2560)
        return self._cached_text_embeds

    @torch.no_grad()
    def _encode_frozen(self, image: Image.Image):
        """Encode image through frozen encoders (VAE + SigLip). No grad needed."""
        z_0 = encode_image_vae(self.pipe, image)
        siglip_raw = encode_image_siglip(self.pipe, image)  # (H', W', 1152)
        text_embeds = self._get_text_embeds()  # (L, 2560)
        return z_0, siglip_raw, text_embeds

    def _build_extended_prompt(self, siglip_raw, text_embeds):
        """Project SigLip features and concat with text embeddings. Grad flows here."""
        siglip_flat = siglip_raw.reshape(-1, siglip_raw.shape[-1])  # (H'*W', 1152)
        siglip_projected = self.pipe.siglip_projection(siglip_flat)  # (H'*W', 2560)
        return torch.cat([text_embeds, siglip_projected], dim=0)  # (L + H'*W', 2560)

    def _forward_loss(self, image: Image.Image, t: float = None):
        """Compute velocity matching loss for a single sample."""
        z_0, siglip_raw, text_embeds = self._encode_frozen(image)

        # Project SigLip → text space (trainable, grad flows)
        extended_prompt = self._build_extended_prompt(siglip_raw, text_embeds)

        if t is None:
            if self.timesteps is not None:
                t = random.choice(self.timesteps)  # turbo: 8 fixed timesteps
            else:
                t = random.random()  # base: uniform [0, 1)
        sigma = self._compute_sigma(t)

        noise = torch.randn_like(z_0)
        z_t = (1 - sigma) * z_0 + sigma * noise
        v_target = noise - z_0

        self.pipe.load_models_to_device(self.pipe.in_iteration_models)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}

        timestep_t = torch.tensor([t * SCHEDULER_SCALE], dtype=self.pipe.torch_dtype, device=self.device)
        inputs_shared = {"latents": z_t}
        inputs_posi = {"prompt_embeds": extended_prompt}
        inputs_nega = {}  # cfg_scale=1.0 → negative path never runs

        v_pred = self.pipe.cfg_guided_model_fn(
            self.pipe.model_fn, 1.0,
            inputs_shared, inputs_posi, inputs_nega,
            **models, timestep=timestep_t, progress_id=0,
        )
        return F.mse_loss(v_pred, v_target), t

    def train_step(self, images: list[Image.Image], grad_accum_steps: int = 1) -> dict:
        """One optimizer step over `grad_accum_steps` sequential forward passes."""
        self.optimizer.zero_grad()
        total_loss = 0.0
        last_t = 0.0

        for image in images:
            loss, t = self._forward_loss(image)
            (loss / grad_accum_steps).backward()
            total_loss += loss.item()
            last_t = t

        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.pipe.siglip_projection.parameters()), max_norm=self.max_grad_norm,
        )
        self.optimizer.step()
        self.scheduler.step()

        return {
            "loss": total_loss / grad_accum_steps,
            "grad_norm": grad_norm.item(),
            "timestep": last_t,
            "sigma": self._compute_sigma(last_t),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def compute_val_loss(self, val_images: list) -> float:
        """Average loss over fixed validation images, random timesteps."""
        losses = []
        for image in val_images:
            loss, _ = self._forward_loss(image)
            losses.append(loss.item())
        return sum(losses) / len(losses)

    @torch.no_grad()
    def generate_sample(self, image: Image.Image, seed: int = 42) -> Image.Image:
        return run_img2img_siglip_caption(
            self.pipe, "", image,
            num_inference_steps=self.eval_num_steps,
            cfg_scale=self.eval_cfg_scale,
            height=512, width=512, seed=seed,
        )

    def save_checkpoint(self, path: str, step: int):
        torch.save({
            "step": step,
            "siglip_projection_state_dict": self.pipe.siglip_projection.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str, weights_only: bool = False) -> int:
        """Load checkpoint. If weights_only=True, only load projection weights
        (fresh optimizer + scheduler for new LR/schedule)."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.pipe.siglip_projection.load_state_dict(checkpoint["siglip_projection_state_dict"])
        step = checkpoint.get("step", 0)
        if weights_only:
            print(f"Loaded projection weights from {path} (step {step}, fresh optimizer)")
            return 0  # restart step counter
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Loaded checkpoint from {path} (step {step})")
        return step


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Train SigLip projection layers")
    parser.add_argument("--data_dir", type=str,
                        default="/home/gnan/projects/data/datasets/relaion-art-lowres/")
    parser.add_argument("--eval_dir", type=str,
                        default="eval",
                        help="Directory with eval images + eval_set.txt")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--n_val", type=int, default=8,
                        help="Number of fixed validation samples for val loss")
    parser.add_argument("--n_eval_images", type=int, default=12,
                        help="Number of eval images to generate at each eval step")
    parser.add_argument("--max_size", type=int, default=768,
                        help="Max image dimension (preserves aspect ratio, rounds to 16)")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Fixed size for eval images")
    parser.add_argument("--model", type=str, default="turbo", choices=["turbo", "base"],
                        help="Model variant: turbo (8-step distilled) or base (50-step flow matching)")
    parser.add_argument("--text_encoder", type=str, default="qwen3",
                        choices=["qwen3", "qwen3vl"],
                        help="Text encoder: qwen3 (default), qwen3vl (VL splice for i2i)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--output_dir", type=str, default="outputs/stage1")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (weights + optimizer + scheduler)")
    parser.add_argument("--init_weights", type=str, default=None,
                        help="Init projection weights from checkpoint (fresh optimizer/scheduler)")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch size)")
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--lr_min_ratio", type=float, default=0.01)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_hps", action="store_true",
                        help="Skip HPSv2.1 scoring")

    args = parser.parse_args()

    # Setup output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load eval images from eval_set.txt
    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_absolute():
        eval_dir = Path(__file__).parent / eval_dir
    eval_set_file = eval_dir / "eval_set.txt"
    eval_filenames = [
        l.strip() for l in eval_set_file.read_text().splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    eval_filenames = eval_filenames[:args.n_eval_images]

    eval_images = []
    for fname in eval_filenames:
        img_path = eval_dir / fname
        img = Image.open(img_path).convert("RGB")
        img = img.resize((args.image_size, args.image_size), Image.LANCZOS)
        eval_images.append(img)
    n_eval = len(eval_images)
    print(f"Eval images: {n_eval} from {eval_set_file}")

    # Save eval inputs to run dir
    for i, img in enumerate(eval_images):
        img.save(run_dir / "samples" / f"eval_input_{i}.png")

    # Load dataset
    dataset = ArtDataset(args.data_dir, max_size=args.max_size)

    # Init trainer
    trainer = Stage1Trainer(
        device=args.device, dtype=args.dtype, lr=args.lr,
        warmup_steps=args.lr_warmup_steps, total_steps=args.steps,
        lr_min_ratio=args.lr_min_ratio, max_grad_norm=args.max_grad_norm,
        model=args.model, text_encoder=args.text_encoder,
    )

    # Resume or init from checkpoint
    start_step = 0
    if args.resume:
        start_step = trainer.load_checkpoint(args.resume)
    elif args.init_weights:
        trainer.load_checkpoint(args.init_weights, weights_only=True)

    # Pick fixed validation images (for val loss)
    val_images = []
    while len(val_images) < args.n_val:
        try:
            img, _ = dataset.random_sample()
            val_images.append(img)
        except Exception:
            continue
    random.seed()  # reseed for training randomness

    # HPSv2.1 scorer
    use_hps = not args.no_hps
    if use_hps:
        try:
            import hpsv2
            print("HPSv2.1 scoring enabled")
        except ImportError:
            print("Warning: hpsv2 not installed, skipping HPS scoring")
            use_hps = False

    # Init wandb
    if not args.no_wandb:
        wandb.init(
            project=f"synthos-train-stg1-{args.model}",
            config=vars(args),
            dir=str(run_dir),
            resume="allow" if args.resume else None,
        )
        # Log eval input images once at step 0
        wandb.log({
            "eval/inputs": [
                wandb.Image(img, caption=f"p{i}: {eval_filenames[i][:30]}")
                for i, img in enumerate(eval_images)
            ]
        }, step=0)

    # Training loop
    running_loss = 0.0
    for step in tqdm(range(start_step + 1, args.steps + 1), desc="Stage 1"):
        images = [dataset.random_sample()[0] for _ in range(args.grad_accum_steps)]
        metrics = trainer.train_step(images, grad_accum_steps=args.grad_accum_steps)
        running_loss += metrics["loss"]

        # --- Step-level logging (train loss + val loss) ---
        if step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            val_loss = trainer.compute_val_loss(val_images)

            tqdm.write(
                f"[Step {step}] train={avg_loss:.6f} val={val_loss:.6f} "
                f"grad={metrics['grad_norm']:.4f} t={metrics['timestep']:.3f}"
            )

            log_dict = {
                "train/loss": avg_loss,
                "train/grad_norm": metrics["grad_norm"],
                "train/timestep": metrics["timestep"],
                "train/sigma": metrics["sigma"],
                "train/lr": metrics["lr"],
                "eval/val_loss": val_loss,
            }
            if not args.no_wandb:
                wandb.log(log_dict, step=step)
            running_loss = 0.0

        # --- Eval: image generation + HPS ---
        if step % args.eval_every == 0:
            tqdm.write(f"  [Eval @ step {step}] generating {n_eval} samples...")
            eval_log = {}
            generated = []

            for i, img in enumerate(eval_images):
                recon = trainer.generate_sample(img, seed=42)
                recon.save(run_dir / "samples" / f"step_{step:05d}_s{i}.png")
                generated.append(recon)

            # Log as wandb image list
            eval_log["eval/samples"] = [
                wandb.Image(recon, caption=f"p{i}_s0: {eval_filenames[i][:30]}")
                for i, recon in enumerate(generated)
            ]

            # HPSv2.1
            if use_hps:
                hps_scores = []
                for recon in generated:
                    try:
                        score = hpsv2.score(recon, "", hps_version="v2.1")[0]
                        hps_scores.append(float(score))
                    except Exception as e:
                        tqdm.write(f"    HPSv2 error: {e}")
                if hps_scores:
                    mean_hps = sum(hps_scores) / len(hps_scores)
                    tqdm.write(f"    hpsv2={mean_hps:.3f}")
                    eval_log["eval/hpsv2_mean"] = mean_hps

            if not args.no_wandb:
                wandb.log(eval_log, step=step)

        # --- Save checkpoint ---
        if step % args.save_every == 0:
            trainer.save_checkpoint(
                str(run_dir / "checkpoints" / f"step_{step:05d}.pt"), step,
            )

    # Final save
    trainer.save_checkpoint(str(run_dir / "checkpoints" / "final.pt"), args.steps)
    print(f"\nStage 1 complete. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
