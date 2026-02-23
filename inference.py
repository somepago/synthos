#!/usr/bin/env python3
"""
Inference script for Z-Image text2image and img2img.

Usage:
    # Text-to-image with turbo model
    python inference.py --prompt "a cat sitting on a chair" --seed 42

    # Text-to-image with base model
    python inference.py --prompt "a cat sitting on a chair" --model z-image-base

    # Image-to-image (SigLip-conditioned)
    python inference.py --prompt "oil painting of a cat" --edit_image cat.png

    # With LoRA checkpoint
    python inference.py --prompt "a cat" --lora_path outputs/checkpoints/best.pt
"""

from src import env_setup  # noqa: F401

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.model_utils import load_pipeline, get_defaults
from src.diffusion import get_latent_shape, generate_noise, run_full_diffusion, run_img2img


def load_checkpoint(pipe, checkpoint_path: str):
    """Load checkpoint: supports LoRA+siglip (GRPO) or siglip-only (stage1)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Load LoRA weights if present
    lora_state = checkpoint.get("lora_state_dict", {})
    if lora_state:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(pipe.dit, lora_state)
        print(f"Loaded LoRA weights ({len(lora_state)} params)")

    # Load siglip projection weights if present
    siglip_state = checkpoint.get("siglip_state_dict", {})
    if siglip_state:
        for name, param in pipe.dit.named_parameters():
            if name in siglip_state:
                param.data.copy_(siglip_state[name].to(param.device, param.dtype))
        print(f"Loaded siglip projection weights ({len(siglip_state)} params)")

    print(f"Loaded checkpoint from {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Z-Image inference")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model", type=str, default=None,
                        choices=["z-image-base", "z-image-turbo", "z-image-turbo-img2img"],
                        help="Model key (auto-selects turbo-img2img if --edit_image is set)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--output", type=str, default="output.png")

    # img2img via SigLip conditioning
    parser.add_argument("--edit_image", type=str, default=None,
                        help="Reference image for SigLip-conditioned img2img")

    # LoRA
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA checkpoint")

    # Override defaults
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")

    args = parser.parse_args()

    # Auto-select model: use img2img variant if edit_image is provided
    model_key = args.model
    if model_key is None:
        model_key = "z-image-turbo-img2img" if args.edit_image else "z-image-turbo"

    # Load pipeline
    pipe = load_pipeline(model_key, device=args.device, torch_dtype=args.dtype)
    defaults = get_defaults(model_key)
    num_steps = args.num_steps or defaults["num_inference_steps"]
    cfg_scale = args.cfg_scale or defaults["cfg_scale"]

    # Load LoRA if specified
    if args.lora_path:
        load_checkpoint(pipe, args.lora_path)

    if args.edit_image:
        # img2img: SigLip-conditioned generation
        edit_img = Image.open(args.edit_image).convert("RGB")
        image = run_img2img(
            pipe, args.prompt, edit_img,
            num_inference_steps=num_steps, cfg_scale=cfg_scale,
            height=args.height, width=args.width, seed=args.seed,
        )
        print(f"img2img (SigLip): {args.height}x{args.width}, steps={num_steps}, cfg={cfg_scale}")
    else:
        # text2image
        shape = get_latent_shape(args.height, args.width)
        noise = generate_noise(args.seed, shape, args.device, getattr(torch, args.dtype))
        image = run_full_diffusion(pipe, args.prompt, noise, num_steps, cfg_scale)
        print(f"text2image: {args.height}x{args.width}, steps={num_steps}, cfg={cfg_scale}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
