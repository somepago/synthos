#!/bin/bash
# Multi-resolution i2i + text-guided multi-image composition tests
set -e
cd /home/gnan/projects/synthos

OUT=outputs/splice_gen4
mkdir -p $OUT

echo "=== Multi-resolution i2i (turbo, VL splice) ==="

# 1) 512x512 baseline - worker
echo "[1/6] worker 512x512"
venv/bin/python3 inference.py --prompt "" --text_encoder qwen3vl \
  --edit_image eval/000371083.jpg --output $OUT/worker_512x512.png \
  --height 512 --width 512 --seed 42

# 2) 768x512 landscape - captain
echo "[2/6] captain 768x512"
venv/bin/python3 inference.py --prompt "" --text_encoder qwen3vl \
  --edit_image eval/001574009.jpg --output $OUT/captain_768x512.png \
  --height 512 --width 768 --seed 42

# 3) 512x768 portrait - fashion
echo "[3/6] fashion 512x768"
venv/bin/python3 inference.py --prompt "" --text_encoder qwen3vl \
  --edit_image eval/000481713.jpg --output $OUT/fashion_512x768.png \
  --height 768 --width 512 --seed 42

# 4) 768x768 square - large landscape photo
echo "[4/6] landscape 768x768"
venv/bin/python3 inference.py --prompt "" --text_encoder qwen3vl \
  --edit_image eval/002368074.jpg --output $OUT/landscape_768x768.png \
  --height 768 --width 768 --seed 42

# 5) 1024x512 wide - MJ art
echo "[5/6] mjart 1024x512"
venv/bin/python3 inference.py --prompt "" --text_encoder qwen3vl \
  --edit_image eval/43b35bfe-d099-4619-8c84-ecc2722d8adb.jpeg --output $OUT/mjart_1024x512.png \
  --height 512 --width 1024 --seed 42

# 6) 1024x1024 large square - tall painting
echo "[6/6] painting 1024x1024"
venv/bin/python3 inference.py --prompt "" --text_encoder qwen3vl \
  --edit_image eval/000708629.webp --output $OUT/painting_1024x1024.png \
  --height 1024 --width 1024 --seed 42

echo ""
echo "=== Text-guided multi-image composition ==="

venv/bin/python3 -c "
from src import env_setup
import torch
from PIL import Image
from src.model_utils import load_pipeline
from src.diffusion import get_latent_shape, generate_noise, _denoise_step, _decode_final

pipe = load_pipeline('z-image-turbo', device='cuda', torch_dtype='bfloat16', text_encoder='qwen3vl')

def encode_interleaved(pipe, content_list, device):
    \"\"\"Encode interleaved image+text content through VL model.\"\"\"
    from src.diffusion import _cap_resolution
    images = []
    content = []
    for item in content_list:
        if isinstance(item, Image.Image):
            img = _cap_resolution(item, 512*512)
            images.append(img)
            content.append({'type': 'image', 'image': img})
        else:
            content.append({'type': 'text', 'text': item})
    messages = [{'role': 'user', 'content': content}]
    text = pipe.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = pipe.vl_processor(text=[text], images=images, return_tensors='pt').to(device)
    with torch.no_grad():
        out = pipe.vl_model.model(**inputs, output_hidden_states=True)
    h = out.hidden_states[-2][0]
    mask = inputs['attention_mask'][0].bool()
    return h[mask]

def gen(pipe, embeds, seed, h=512, w=512, out_path='out.png'):
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

worker = Image.open('eval/000371083.jpg').convert('RGB')
fashion = Image.open('eval/000481713.jpg').convert('RGB')
captain = Image.open('eval/001574009.jpg').convert('RGB')
mjart = Image.open('eval/43b35bfe-d099-4619-8c84-ecc2722d8adb.jpeg').convert('RGB')

# Combo 1: fashion in style of worker
print('Combo 1: fashion + \"in the style of\" + worker')
embeds = encode_interleaved(pipe, [fashion, 'in the style of', worker], 'cuda')
print(f'  Embedding shape: {embeds.shape}')
gen(pipe, embeds, 42, out_path='outputs/splice_gen4/compose_fashion_style_worker.png')

# Combo 2: captain in style of MJ art
print('Combo 2: captain + \"in the style of\" + MJ art')
embeds = encode_interleaved(pipe, [captain, 'in the artistic style of', mjart], 'cuda')
print(f'  Embedding shape: {embeds.shape}')
gen(pipe, embeds, 42, out_path='outputs/splice_gen4/compose_captain_style_mjart.png')

# Combo 3: worker + \"wearing\" + fashion
print('Combo 3: worker + \"wearing the outfit from\" + fashion')
embeds = encode_interleaved(pipe, [worker, 'wearing the outfit from', fashion], 'cuda')
print(f'  Embedding shape: {embeds.shape}')
gen(pipe, embeds, 42, out_path='outputs/splice_gen4/compose_worker_wearing_fashion.png')

# Combo 4: plain text between two images (no style transfer prompt)
print('Combo 4: fashion + \"and\" + captain (plain merge)')
embeds = encode_interleaved(pipe, [fashion, 'and', captain], 'cuda')
print(f'  Embedding shape: {embeds.shape}')
gen(pipe, embeds, 42, out_path='outputs/splice_gen4/compose_fashion_and_captain.png')

print('Done!')
"

echo ""
echo "=== ALL DONE ==="
ls -la $OUT/*.png
