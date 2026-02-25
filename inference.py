#!/usr/bin/env python3
"""
Inference script for Z-Image text2image and img2img.

Usage:
    # Text-to-image with turbo model
    python inference.py --prompt "a cat sitting on a chair" --seed 42

    # Text-to-image with base model
    python inference.py --prompt "a cat sitting on a chair" --model z-image-base

    # Image-to-image via VL splice (Qwen3-VL visual encoder + Z-Image text encoder)
    python inference.py --prompt "" --text_encoder qwen3vl --edit_image cat.png

    # With LoRA checkpoint
    python inference.py --prompt "a cat" --lora_path outputs/checkpoints/best.pt
"""

from src import env_setup  # noqa: F401

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.model_utils import load_pipeline, get_defaults
from src.diffusion import (
    get_latent_shape, generate_noise, run_full_diffusion,
    encode_image_vl, encode_text_vl, _denoise_step, _decode_final,
)


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
                        choices=["z-image-base", "z-image-turbo"],
                        help="Model key (default: z-image-turbo)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--output", type=str, default="output.png")

    # img2img via VL splice
    parser.add_argument("--edit_image", type=str, default=None,
                        help="Reference image for VL-conditioned img2img (requires --text_encoder qwen3vl)")

    # LoRA
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA checkpoint")

    # Text encoder
    parser.add_argument("--text_encoder", type=str, default="qwen3",
                        choices=["qwen3", "qwen3vl"],
                        help="Text encoder: qwen3 (default), qwen3vl (VL splice for i2i)")

    # Override defaults
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")

    args = parser.parse_args()

    model_key = args.model or "z-image-turbo"

    # Load pipeline
    pipe = load_pipeline(model_key, device=args.device, torch_dtype=args.dtype,
                         text_encoder=args.text_encoder)
    defaults = get_defaults(model_key)
    num_steps = args.num_steps or defaults["num_inference_steps"]
    cfg_scale = args.cfg_scale or defaults["cfg_scale"]

    # Load LoRA if specified
    if args.lora_path:
        load_checkpoint(pipe, args.lora_path)

    if args.edit_image:
        # img2img via VL splice
        edit_img = Image.open(args.edit_image).convert("RGB")
        prompt_embeds = encode_image_vl(pipe, edit_img, pipe.device)

        shape = get_latent_shape(args.height, args.width)
        noise = generate_noise(args.seed, shape, pipe.device, pipe.torch_dtype)

        pipe.scheduler.set_timesteps(num_steps, denoising_strength=1.0, shift=None)
        pipe.load_models_to_device(pipe.in_iteration_models)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

        shared = {"latents": noise}
        posi = {"prompt_embeds": prompt_embeds}
        nega = {}
        if cfg_scale > 1.0:
            neg_embeds = encode_text_vl(pipe, "", pipe.device)
            nega = {"prompt_embeds": neg_embeds}
        with torch.no_grad():
            for pid, ts in enumerate(pipe.scheduler.timesteps):
                _denoise_step(pipe, cfg_scale, shared, posi, nega, models, ts, pid)
            image = _decode_final(pipe, shared)
        print(f"img2img (VL splice): {args.height}x{args.width}, steps={num_steps}, cfg={cfg_scale}")
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
