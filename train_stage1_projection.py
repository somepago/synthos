#!/usr/bin/env python3
"""
Stage 1: Train SigLip projection layers with diffusion velocity-matching loss.

Trainable: siglip_embedder + siglip_refiner + siglip_pad_token (~358M params)
Frozen: SigLip2 image encoder, DiT (all original weights), text encoder, VAE

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

from src.model_utils import load_pipeline
from src.constants import SCHEDULER_SCALE, TURBO_SCHEDULER_TIMESTEPS, FLOW_MATCHING_SHIFT
from src.diffusion import (
    encode_image_vae,
    encode_image_siglip,
    run_img2img,
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
    """Train SigLip projection layers with diffusion velocity matching."""

    def __init__(self, device="cuda", dtype="bfloat16", lr=1e-4,
                 warmup_steps=100, total_steps=5000, lr_min_ratio=0.01,
                 max_grad_norm=5.0):
        self.device = device

        self.pipe = load_pipeline("z-image-turbo-img2img", device=device, torch_dtype=dtype)
        assert self.pipe.image_encoder is not None, "SigLip image encoder not loaded"
        assert self.pipe.dit.siglip_embedder is not None, "SigLip projection layers not on DiT"

        # Freeze everything, unfreeze only siglip projection layers
        for param in self.pipe.dit.parameters():
            param.requires_grad_(False)
        for name, param in self.pipe.dit.named_parameters():
            if "siglip" in name:
                param.requires_grad_(True)

        trainable_count = sum(p.numel() for p in self.pipe.dit.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in self.pipe.dit.parameters())
        print(f"Trainable: {trainable_count:,} / {total_count:,} params")

        trainable_params = [p for p in self.pipe.dit.parameters() if p.requires_grad]
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
        self.timesteps = TURBO_SCHEDULER_TIMESTEPS
        self.shift = FLOW_MATCHING_SHIFT
        self.max_grad_norm = max_grad_norm
        self._cached_empty_prompt = None

    def _compute_sigma(self, t: float) -> float:
        return self.shift * t / (1 + (self.shift - 1) * t)

    @torch.no_grad()
    def _get_empty_prompt_embeds(self, image: Image.Image):
        """Encode empty prompt once and cache (omni mode structure)."""
        if self._cached_empty_prompt is None:
            from diffsynth.pipelines.z_image import ZImageUnit_PromptEmbedder
            self.pipe.load_models_to_device(["text_encoder"])
            embedder = ZImageUnit_PromptEmbedder()
            self._cached_empty_prompt = embedder.encode_prompt_omni(
                self.pipe, "", edit_image=image, device=self.device,
            )
        return self._cached_empty_prompt

    @torch.no_grad()
    def _encode_inputs(self, image: Image.Image):
        z_0 = encode_image_vae(self.pipe, image)
        image_embeds = encode_image_siglip(self.pipe, image)
        prompt_embeds = self._get_empty_prompt_embeds(image)
        return z_0, image_embeds, prompt_embeds

    def _forward_loss(self, image: Image.Image, t: float = None):
        """Compute velocity matching loss for a single sample. Returns loss tensor."""
        z_0, image_embeds, prompt_embeds = self._encode_inputs(image)

        if t is None:
            t = random.choice(self.timesteps)
        sigma = self._compute_sigma(t)

        noise = torch.randn_like(z_0)
        z_t = (1 - sigma) * z_0 + sigma * noise
        v_target = noise - z_0

        self.pipe.load_models_to_device(self.pipe.in_iteration_models)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}

        timestep_t = torch.tensor([t * SCHEDULER_SCALE], dtype=self.pipe.torch_dtype, device=self.device)
        inputs_shared = {"latents": z_t}
        inputs_posi = {"prompt_embeds": prompt_embeds, "image_embeds": [image_embeds], "image_latents": [z_0]}
        # cfg_scale=1.0 → negative path never runs, pass empty dict
        inputs_nega = {}

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
            [p for p in self.pipe.dit.parameters() if p.requires_grad], max_norm=self.max_grad_norm,
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
    def generate_sample(self, image: Image.Image, caption: str, seed: int = 42) -> Image.Image:
        return run_img2img(
            self.pipe, caption, image,
            num_inference_steps=8, cfg_scale=1.0,
            height=512, width=512, seed=seed,
        )

    def save_checkpoint(self, path: str, step: int):
        siglip_state = {}
        for name, param in self.pipe.dit.named_parameters():
            if "siglip" in name:
                siglip_state[name] = param.cpu()

        torch.save({
            "step": step,
            "siglip_state_dict": siglip_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        siglip_state = checkpoint["siglip_state_dict"]
        for name, param in self.pipe.dit.named_parameters():
            if name in siglip_state:
                param.data.copy_(siglip_state[name].to(param.device, param.dtype))
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        step = checkpoint.get("step", 0)
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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--output_dir", type=str, default="outputs/stage1")
    parser.add_argument("--resume", type=str, default=None)
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
    )

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        start_step = trainer.load_checkpoint(args.resume)

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
            project="synthos-train-stg1",
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

        # --- Eval: image generation + HPSv3 ---
        if step % args.eval_every == 0:
            tqdm.write(f"  [Eval @ step {step}] generating {n_eval} samples...")
            eval_log = {}
            generated = []

            for i, img in enumerate(eval_images):
                recon = trainer.generate_sample(img, "", seed=42)
                recon.save(run_dir / "samples" / f"step_{step:05d}_s{i}.png")
                generated.append(recon)

            # Log as wandb image list (diffscapes pattern)
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
