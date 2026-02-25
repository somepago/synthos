#!/bin/bash
# Test full VL splice (full model with Z-Image LLM weights)
# 1) Turbo i2i - same images as before to compare quality
# 2) Base i2i - 50 steps
# 3) Multi-image input
set -e
cd /home/gnan/projects/synthos

OUT=outputs/splice_test3
mkdir -p $OUT

echo "=== Turbo i2i (full VL splice) ==="

echo "[turbo 1/3] worker 512x512"
venv/bin/python3 inference.py --prompt "" --text_encoder qwen3vl \
  --edit_image eval/000371083.jpg --output $OUT/turbo_worker_512.png \
  --height 512 --width 512 --seed 42

echo "[turbo 2/3] fashion 512x512"
venv/bin/python3 inference.py --prompt "" --text_encoder qwen3vl \
  --edit_image eval/000481713.jpg --output $OUT/turbo_fashion_512.png \
  --height 512 --width 512 --seed 42

echo "[turbo 3/3] captain 512x512"
venv/bin/python3 inference.py --prompt "" --text_encoder qwen3vl \
  --edit_image eval/001574009.jpg --output $OUT/turbo_captain_512.png \
  --height 512 --width 512 --seed 42

echo ""
echo "=== Base i2i (full VL splice, 50 steps) ==="

echo "[base 1/3] worker 512x512"
venv/bin/python3 inference.py --prompt "" --text_encoder qwen3vl --model z-image-base \
  --edit_image eval/000371083.jpg --output $OUT/base_worker_512.png \
  --height 512 --width 512 --seed 42

echo "[base 2/3] fashion 512x512"
venv/bin/python3 inference.py --prompt "" --text_encoder qwen3vl --model z-image-base \
  --edit_image eval/000481713.jpg --output $OUT/base_fashion_512.png \
  --height 512 --width 512 --seed 42

echo "[base 3/3] captain 512x512"
venv/bin/python3 inference.py --prompt "" --text_encoder qwen3vl --model z-image-base \
  --edit_image eval/001574009.jpg --output $OUT/base_captain_512.png \
  --height 512 --width 512 --seed 42

echo ""
echo "=== Multi-image (turbo, full VL splice) ==="

venv/bin/python3 -c "
from src import env_setup
import torch
from PIL import Image
from src.model_utils import load_pipeline, get_defaults
from src.diffusion import encode_images_vl, get_latent_shape, generate_noise, _denoise_step, _decode_final

pipe = load_pipeline('z-image-turbo', device='cuda', torch_dtype='bfloat16', text_encoder='qwen3vl')

def gen_from_embeds(pipe, embeds, seed, h=512, w=512, out_path='out.png'):
    noise = generate_noise(seed, get_latent_shape(h, w), 'cuda', torch.bfloat16)
    pipe.scheduler.set_timesteps(8, denoising_strength=1.0, shift=None)
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    shared = {'latents': noise}
    posi = {'prompt_embeds': embeds}
    with torch.no_grad():
        for pid, ts in enumerate(pipe.scheduler.timesteps):
            _denoise_step(pipe, 1.0, shared, posi, {}, models, ts, pid)
        image = _decode_final(pipe, shared)
    image.save(out_path)
    print(f'  Saved: {out_path}')

# Combo 1: worker + fashion
imgs = [Image.open('eval/000371083.jpg').convert('RGB'), Image.open('eval/000481713.jpg').convert('RGB')]
print(f'Combo 1: worker + fashion')
embeds = encode_images_vl(pipe, imgs, 'cuda')
print(f'  Embedding shape: {embeds.shape}')
gen_from_embeds(pipe, embeds, 42, out_path='outputs/splice_test3/multi_worker_fashion.png')

# Combo 2: worker + captain
imgs = [Image.open('eval/000371083.jpg').convert('RGB'), Image.open('eval/001574009.jpg').convert('RGB')]
print(f'Combo 2: worker + captain')
embeds = encode_images_vl(pipe, imgs, 'cuda')
print(f'  Embedding shape: {embeds.shape}')
gen_from_embeds(pipe, embeds, 42, out_path='outputs/splice_test3/multi_worker_captain.png')

# Combo 3: 3 images
imgs = [
    Image.open('eval/000481713.jpg').convert('RGB'),
    Image.open('eval/001574009.jpg').convert('RGB'),
    Image.open('eval/000371083.jpg').convert('RGB'),
]
print(f'Combo 3: fashion + captain + worker')
embeds = encode_images_vl(pipe, imgs, 'cuda')
print(f'  Embedding shape: {embeds.shape}')
gen_from_embeds(pipe, embeds, 42, out_path='outputs/splice_test3/multi_3way.png')

print('Done!')
"

echo ""
echo "=== ALL DONE ==="
ls -la $OUT/*.png
