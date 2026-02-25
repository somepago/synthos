#!/bin/bash
# Test 1: Z-Image Base (50-step) with qwen3vl splice
# Test 2: Multi-image input via qwen3vl splice
set -e
cd /home/gnan/projects/synthos

OUT=outputs/splice_test2
mkdir -p $OUT

echo "=== Test 1: Z-Image Base + qwen3vl splice (50 steps, cfg=4.0) ==="

echo "[base i2i 1/3] worker @ 512x512"
venv/bin/python3 inference.py \
  --prompt "" --text_encoder qwen3vl --model z-image-base \
  --edit_image eval/000371083.jpg \
  --output $OUT/base_worker_512x512.png --height 512 --width 512 --seed 42

echo "[base i2i 2/3] fashion @ 512x512"
venv/bin/python3 inference.py \
  --prompt "" --text_encoder qwen3vl --model z-image-base \
  --edit_image eval/000481713.jpg \
  --output $OUT/base_fashion_512x512.png --height 512 --width 512 --seed 42

echo "[base i2i 3/3] captain @ 512x512"
venv/bin/python3 inference.py \
  --prompt "" --text_encoder qwen3vl --model z-image-base \
  --edit_image eval/001574009.jpg \
  --output $OUT/base_captain_512x512.png --height 512 --width 512 --seed 42

echo ""
echo "=== Test 2: Multi-image input (turbo, qwen3vl splice) ==="

venv/bin/python3 -c "
from src import env_setup
import torch
from PIL import Image
from src.model_utils import load_pipeline, get_defaults
from src.diffusion import encode_images_vl, get_latent_shape, generate_noise, _denoise_step, _decode_final

pipe = load_pipeline('z-image-turbo', device='cuda', torch_dtype='bfloat16', text_encoder='qwen3vl')
defaults = get_defaults('z-image-turbo')

# Combo 1: worker + fashion
imgs = [Image.open('eval/000371083.jpg').convert('RGB'), Image.open('eval/000481713.jpg').convert('RGB')]
print(f'Combo 1: worker + fashion ({len(imgs)} images)')
embeds = encode_images_vl(pipe, imgs, 'cuda')
print(f'  Embedding shape: {embeds.shape}')

noise = generate_noise(42, get_latent_shape(512, 512), 'cuda', torch.bfloat16)
pipe.scheduler.set_timesteps(8, denoising_strength=1.0, shift=None)
pipe.load_models_to_device(pipe.in_iteration_models)
models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
shared = {'latents': noise}
posi = {'prompt_embeds': embeds}
with torch.no_grad():
    for pid, ts in enumerate(pipe.scheduler.timesteps):
        _denoise_step(pipe, 1.0, shared, posi, {}, models, ts, pid)
    image = _decode_final(pipe, shared)
image.save('outputs/splice_test2/multi_worker_fashion.png')
print('  Saved: multi_worker_fashion.png')

# Combo 2: worker + captain
imgs = [Image.open('eval/000371083.jpg').convert('RGB'), Image.open('eval/001574009.jpg').convert('RGB')]
print(f'Combo 2: worker + captain ({len(imgs)} images)')
embeds = encode_images_vl(pipe, imgs, 'cuda')
print(f'  Embedding shape: {embeds.shape}')

noise = generate_noise(42, get_latent_shape(512, 512), 'cuda', torch.bfloat16)
pipe.scheduler.set_timesteps(8, denoising_strength=1.0, shift=None)
shared = {'latents': noise}
posi = {'prompt_embeds': embeds}
with torch.no_grad():
    for pid, ts in enumerate(pipe.scheduler.timesteps):
        _denoise_step(pipe, 1.0, shared, posi, {}, models, ts, pid)
    image = _decode_final(pipe, shared)
image.save('outputs/splice_test2/multi_worker_captain.png')
print('  Saved: multi_worker_captain.png')

# Combo 3: fashion + captain + worker (3 images)
imgs = [
    Image.open('eval/000481713.jpg').convert('RGB'),
    Image.open('eval/001574009.jpg').convert('RGB'),
    Image.open('eval/000371083.jpg').convert('RGB'),
]
print(f'Combo 3: fashion + captain + worker ({len(imgs)} images)')
embeds = encode_images_vl(pipe, imgs, 'cuda')
print(f'  Embedding shape: {embeds.shape}')

noise = generate_noise(42, get_latent_shape(512, 512), 'cuda', torch.bfloat16)
pipe.scheduler.set_timesteps(8, denoising_strength=1.0, shift=None)
shared = {'latents': noise}
posi = {'prompt_embeds': embeds}
with torch.no_grad():
    for pid, ts in enumerate(pipe.scheduler.timesteps):
        _denoise_step(pipe, 1.0, shared, posi, {}, models, ts, pid)
    image = _decode_final(pipe, shared)
image.save('outputs/splice_test2/multi_fashion_captain_worker.png')
print('  Saved: multi_fashion_captain_worker.png')

# Combo 4: two very different images (MJ art + photo)
imgs = [
    Image.open('eval/43b35bfe-d099-4619-8c84-ecc2722d8adb.jpeg').convert('RGB'),
    Image.open('eval/001574009.jpg').convert('RGB'),
]
print(f'Combo 4: MJ art + captain ({len(imgs)} images)')
embeds = encode_images_vl(pipe, imgs, 'cuda')
print(f'  Embedding shape: {embeds.shape}')

noise = generate_noise(42, get_latent_shape(512, 512), 'cuda', torch.bfloat16)
pipe.scheduler.set_timesteps(8, denoising_strength=1.0, shift=None)
shared = {'latents': noise}
posi = {'prompt_embeds': embeds}
with torch.no_grad():
    for pid, ts in enumerate(pipe.scheduler.timesteps):
        _denoise_step(pipe, 1.0, shared, posi, {}, models, ts, pid)
    image = _decode_final(pipe, shared)
image.save('outputs/splice_test2/multi_mjart_captain.png')
print('  Saved: multi_mjart_captain.png')

print('Done!')
"

echo ""
echo "=== ALL DONE ==="
ls -la $OUT/*.png
