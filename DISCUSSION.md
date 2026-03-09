# Discussion / Research Notes

## Stage 1: SigLip Projection Training

### Setup
- **Trainable**: siglip_embedder + siglip_refiner + siglip_pad_token (~358M params)
- **Frozen**: DiT (6.5B), SigLip2 encoder, text encoder, VAE
- **Dataset**: relaion-art-lowres, ~49K images (256-700px), native resolution (rounded to 16px multiples, capped 768)
- **Loss**: Flow matching velocity MSE, `v_target = noise - z_0`
- **Prompts**: Empty (`""`) for both train and eval — pure image conditioning
- **Optimizer**: AdamW, lr=1e-4, weight_decay=0.01, warmup=100 steps, cosine decay to 1%
- **Effective batch size**: 8 (grad accumulation, pipeline only supports B=1)
- **Eval**: 12 curated images (6 real + 6 MJ), seed=42, HPSv2.1

### Run: robust-jazz-6 (wandb)

Config: `--lr 1e-4 --grad_accum_steps 8 --steps 5000 --n_val 32 --n_eval_images 12 --eval_every 100`

#### Loss curve
| Step | Train Loss | Val Loss | Grad Norm |
|------|-----------|----------|-----------|
| 10   | 0.598     | 0.522    | 5.03      |
| 100  | 0.254     | 0.257    | 0.61      |
| 500  | 0.084     | 0.079    | 0.63      |
| 1000 | 0.048     | 0.043    | 0.61      |
| 2000 | 0.052     | 0.041    | 0.32      |
| 2500 | 0.046     | 0.036    | 0.32      |

Loss drops quickly in first 500 steps (0.6 → 0.08), then gradually settles around 0.04. No train/val divergence — no overfitting.

#### HPSv2.1 scores
- Steps 100-800 (old wrong-sign run): ~0.191
- Steps 100-2500 (current run): ~0.210-0.212, plateaued

#### Speed
- ~14s per optimizer step (8 accum samples) on GB10 (128GB unified memory)
- ~1.75s per sample forward+backward
- Bottleneck: backward pass through frozen 6.5B DiT (~3.5s/sample in benchmark)
- Prompt caching saved ~780ms/sample (empty prompt encoded once)

### Bugs Fixed
1. **Missing `image_latents`**: Omni mode DiT expects both `image_embeds` (SigLip features) AND `image_latents` (VAE-encoded reference). Without `image_latents`, `image_noise_mask` structure mismatches → CUDA assert.
2. **Wrong prompt encoding**: Omni mode needs `encode_prompt_omni()` (multi-segment prompts), not `encode_prompt()`. The omni path creates `[vision_start, vision_end, prompt, assistant]` segments — one per image in the sequence.
3. **Velocity sign error**: Original code used `v_target = z_0 - noise`. Correct target is `noise - z_0` because `model_fn` negates DiT output (`v_pred = -dit_output`) and the scheduler step is `z_{t-1} = z_t + v_pred * (σ' - σ)`. Confirmed by `FlowMatchScheduler.training_target()` returning `noise - sample`. Wrong sign caused complementary color inversion in generations.
4. **Corrupt images**: Some images in relaion-art-lowres fail PIL open. Added retry loop in `random_sample()`.

### Observations
- Projection trained very fast (loss 0.6 → 0.04 in ~500 steps). Likely because the Z-Image omni model was already trained with SigLip projection — the DiT attention layers already know how to use SigLip features, so our random projection just needs to learn the right mapping.
- Generation quality (visual) plateaus despite loss still dropping. HPSv2.1 flat at ~0.211.
- **Root cause**: Training 8-step Turbo model with per-timestep MSE loss doesn't work — Turbo was distilled with adversarial losses to produce coherent 8-step trajectories. MSE at independent random timesteps teaches per-step prediction but not multi-step coherence. Proper MSE training needs ~1000 denoising steps, then distillation into Turbo — too expensive.

### Architecture Fix: SigLip-Only Conditioning (post robust-jazz-6)

**Problem**: In omni mode, the DiT token sequence is `[ref_VAE_patches + noisy_patches]` with `image_noise_mask=[[0, 1]]`. The reference VAE patches give the DiT pixel-level access to the input image — a shortcut that bypasses SigLip features entirely. The SigLip projection wasn't learning meaningful representations because the DiT could reconstruct directly from the clean reference patches via self-attention.

**Evidence**: Fast convergence (loss 0.6 → 0.04 in 500 steps) was not the SigLip projection learning — it was the DiT exploiting the existing omni-mode copy pathway it was already trained on. Quality plateau at HPSv2.1 ~0.211 further suggests the SigLip features were decorative, not functional.

**Fix v1** (fearless-puddle-7): Removed `image_latents` from the forward pass entirely. Token sequence: `[noisy_patches]` only, `image_noise_mask=[[1]]`. SigLip features passed as `image_embeds`. Prompt encoding uses `encode_prompt_omni(edit_image=None)` — single text segment.

**Bug in Fix v1**: With `image_noise_mask=[[1]]`, ALL tokens (including SigLip features) get `noise_mask=1`. But the DiT uses per-token adaptive layer norm: `noise_mask=1` → `adaln_noisy` (timestep-dependent), `noise_mask=0` → `adaln_clean` (t=1.0). In the original omni mode, SigLip features always had `noise_mask=0` (clean conditioning). Giving them `noise_mask=1` applies the wrong scale/gate modulation in every transformer layer (~30+ layers). This explains why loss plateaued at ~0.50 — the model couldn't properly process SigLip features with wrong conditioning.

**Fix v2** (curious-dawn-9): Pass `image_latents=[None]` (list containing None) to create a dummy reference slot at index 0 with `noise_mask=0`. Structure becomes `[dummy_ref(None), noisy_target]` with mask `[[0, 1]]`. SigLip at index 0 gets `noise_mask=0` (clean conditioning). Prompt encoding uses `encode_prompt_omni(edit_image="dummy")` for 3-segment template matching 2-image omni structure. Modified `model_fn_z_image` to handle None in image_latents list. Minor limitation: SigLip position IDs not rescaled (x_size[0]=None for dummy), but RoPE spatial alignment is approximate anyway.

| Run | Step | Train | Val | HPSv2.1 | Notes |
|-----|------|-------|-----|---------|-------|
| fearless-puddle-7 (v1) | 100 | 0.485 | 0.565 | 0.193 | Wrong noise_mask |
| fearless-puddle-7 (v1) | 300 | 0.502 | 0.535 | 0.177 | Plateaued |
| curious-dawn-9 (v2) | 100 | 0.530 | 0.533 | - | Correct noise_mask |
| curious-dawn-9 (v2) | 300 | 0.484 | 0.499 | 0.175 | In progress |

### Why Omni Mode Failed (dead end — do not revisit)

**Key facts established:**
- Z-Image-Turbo checkpoint has **no siglip weights** — it's a text-to-image only model
- Omni weights (with pretrained siglip_embedder/refiner) are **not publicly released**
- Our `_add_siglip_layers_to_dit` created siglip_embedder/refiner from **random init**
- The frozen DiT's 30+ transformer layers have **never seen SigLip tokens** in attention
- Training only the 358M projection cannot teach frozen 6.1B DiT to attend to a new modality
- All omni-mode runs produced identical blurry blobs regardless of input — SigLip features completely ignored
- The omni path's noise_mask/dual-adaln complexity added debugging surface with no benefit

### Architecture v3: SigLip as Caption Tokens (current)

Bypass omni mode entirely. Project SigLip features to text embedding space (2560-dim) and concatenate with text embeddings as extra caption tokens. Uses the standard turbo (non-omni) DiT code path that the model was actually trained with.

```
UNIFIED SEQUENCE in DiT self-attention:

┌──────────────┬───────────────┬──────────────────────────┐
│  x_patches   │  text_tokens  │  siglip_tokens (NEW)     │
│              │               │                          │
│  noisy image │  "" (empty    │  projected SigLip feats  │
│  latent      │   prompt)     │  from reference image    │
│              │               │                          │
│  (H/16)²     │  ~20 tokens   │  up to 256 tokens        │
│  e.g. 1024   │  (cached)     │  (NaFlex max_patches)    │
│              │               │                          │
│  dim=3840    │  dim=3840     │  dim=3840                │
│ (x_embedder) │(cap_embedder) │ (cap_embedder — shared)  │
└──────────────┴───────────────┴──────────────────────────┘

adaln_input = t_noisy (single timestep, all tokens)
RoPE: 1D sequential positions for all caption-like tokens
```

**Data flow:**
```
Reference Image
      │
      ▼
SigLip2 encoder (frozen, NaFlex, max 256 patches)
      │ (H', W', 1152)
      ▼
siglip_projection (TRAINABLE: LayerNorm(1152) + Linear(1152, 2560))
      │ (H'*W', 2560)               ┌─────────────────────┐
      ▼                              │                     ▼
   concat ◄── text_encoder (frozen) ──┘              cap_embedder (frozen)
      │       "" → ~20 tokens × 2560                 RMSNorm(2560)+Linear(2560,3840)
      │                                                    │
      └────────────────────────────────────────────────────┘
                              │
                              ▼
                      context_refiner (2 frozen blocks)
                              │
                              ▼
                      concat with x_patches → unified sequence
                              │
                              ▼
                      main transformer (30+ frozen layers, single adaln)
                              │
                              ▼
                      unpatchify → v_pred
```

**Why this should work:**
- DiT already knows how to attend to caption tokens (trained as text-to-image)
- SigLip tokens just look like a longer caption to the frozen attention layers
- No omni mode, no noise_mask, no dual adaln — zero masking complexity
- Only ~3M trainable params (single linear projection)

**SigLip2 details:**
- Model: `google/siglip2-so400m-patch16-naflex` (428M params, frozen)
- NaFlex: variable resolution, preserves aspect ratio, does NOT force fixed square
- `max_num_patches=256` caps total patches — images >256px get downscaled to fit
- Output: `(H_patches, W_patches, 1152)` spatial grid, at most 256 tokens
- For 512px input: downscaled to ~256px → 16×16 = 256 tokens
- Could increase max_num_patches later for higher-res features

**Trainable params:**
- `siglip_projection`: LayerNorm(1152) + Linear(1152, 2560) = ~2.95M params
- Two linear layers in series for SigLip: our trainable projection (1152→2560) then frozen cap_embedder (2560→3840)

### Stage 2: GRPO Plan

GRPO generates K complete 8-step trajectories per reference image, scores them with rewards, and uses relative advantages to update. This naturally handles the multi-step Turbo constraint because it optimizes end-to-end output quality.

**Reward function for reconstruction fidelity** — existing reward models (HPSv2, ImageReward) track aesthetics/prompt-alignment, not reconstruction quality. A GAN discriminator with BS=1 is impractical (too noisy, easy overfit). Instead, use perceptual similarity rewards that work per-image:
- **LPIPS**: perceptual distance in VGG space — directly measures reconstruction quality, lightweight
- **DINOv2 cosine similarity**: semantic/structural similarity
- **SSIM**: pixel-level structural similarity
- **SigLip2 self-similarity**: compare SigLip features of input vs output

Combined reward: `reward = -lpips(gen, ref) + α * ssim(gen, ref) + β * dino_sim(gen, ref)`

### Text Encoder: qwen3 / qwen3vl (splice)

`--text_encoder` flag across inference.py, train_stage1_projection.py, train_grpo.py:

| Key | Model | LLM Weights | Vision | Extra Memory |
|-----|-------|-------------|--------|-------------|
| `qwen3` | Z-Image's Qwen3-4B (via pipeline) | Z-Image trained | No | 0 (default) |
| `qwen3vl` | Full Qwen3-VL-4B with spliced Z-Image LLM | Z-Image trained | Yes | ~1GB net (VL replaces text encoder) |

**qwen3vl (splice) architecture — final**:

Load full `Qwen3VLForConditionalGeneration` from HuggingFace, then splice Z-Image's trained LLM weights into `model.model.language_model`. The pipeline's text encoder is NOT loaded (stripped from model configs) — the VL model replaces it entirely.

```
Image → Qwen3-VL ViT (frozen) → PatchMerger → visual tokens
                                                    ↓
                                   LLM (36 layers, Z-Image's trained weights)
                                   + chat template tokens + M-RoPE
                                                    ↓
                                   hidden_states[-2] (layer 34)
                                   filtered by attention_mask
                                                    ↓
                                   prompt_embeds (L, 2560) → DiT conditioning
```

Key: visual tokens pass through ALL 36 LLM layers alongside chat template context tokens. The LLM's self-attention lets the model reason about spatial relationships in the image. Output embeddings are in the same 2560-dim space the DiT was trained to consume.

For i2i, the text portion of the chat message is `""` (empty string). The embedding sequence includes both visual tokens and chat template tokens — the DiT sees it all as one conditioning sequence.

**Why full VL model, not visual-only**: An earlier approach loaded only the ViT+PatchMerger (~0.8GB) and fed visual tokens into Z-Image's text encoder separately. This produced worse quality because the LLM needs the chat template context to properly process visual tokens. The full forward pass with chat template produces significantly better embeddings.

**Memory**: VL model (~9GB) replaces text encoder (~8GB), net ~1GB extra. Fine on 128GB unified memory.

**Multi-image support**: `encode_images_vl()` packs multiple images into one chat message. The LLM's self-attention lets visual tokens from different images attend to each other, producing entangled representations. Tested with 2-image (542 tokens) and 3-image (804 tokens) combos — see `outputs/splice_test3/multi_*.png`.

**CFG with VL conditioning**: For base model (cfg=4.0), negative prompt embeddings are needed. Since there's no text encoder loaded, we encode empty text `""` through the same VL model via `encode_text_vl()`. Turbo (cfg=1.0) skips negative entirely.

#### Bugs encountered
1. **Missing `torch.no_grad()` in inference.py VL i2i path**: The denoising loop was outside any no_grad context. PyTorch built autograd graphs for all 8 DiT forward passes — ~72GB of activation memory. Fix: wrap denoising loop in `with torch.no_grad():`.
2. **CUBLAS OOM on large images**: Image 000481713.jpg (2560x1707 = 4.3M pixels) produced ~4250 visual tokens, exceeding CUBLAS limits. Fix: added `max_pixels` cap (default 768*768) in `encode_image_vl()` that resizes images before VL encoding.
3. **Base model crash with empty negative**: `cfg_guided_model_fn` calls `model_fn` with negative prompts when cfg > 1.0. Passing `{}` (no prompt_embeds) → `len(None)` crash in DiT's `patchify_and_embed`. Fix: `encode_text_vl()` encodes `""` through VL model as negative embeddings.
4. **Quality degradation from visual-only approach**: Splitting ViT from LLM (loading only visual encoder, ~0.8GB) produced artifacts vs original `test_qwen3vl.py` which used full VL model. Root cause: LLM needs chat template context for proper visual token processing. Fix: reverted to full VL model with spliced weights.

#### Test results
- Turbo i2i (8 steps, cfg=1.0): 3/3 passed — `outputs/splice_test3/turbo_*.png`
- Multi-image (turbo): 3/3 passed — `outputs/splice_test3/multi_*.png`
- Base i2i (50 steps, cfg=4.0): in progress — `outputs/splice_test3/base_*.png`

### Z-Image Text Encoder Base Model

Z-Image's text encoder is **Qwen/Qwen3-4B (base)** — not Instruct or Thinking variant.

Evidence:
- config.json identical to Qwen/Qwen3-4B: hidden_size=2560, 36 layers, 32 heads, 8 KV heads
- generation_config.json identical (bos=151643, eos=[151645,151643])
- First 2 of 3 safetensors files have identical blob hashes to Qwen3-4B
- Third file differs only slightly (likely shard boundary)
- diffsynth's ZImageTextEncoder hardcodes matching config

This means Z-Image likely did not finetune the LLM weights significantly (or at all) — the text encoder is essentially frozen Qwen3-4B base used as a feature extractor.

### M-RoPE Ablation (qwen3vl-native i2i)

**Hypothesis**: Qwen3-VL uses Multi-dimensional RoPE (M-RoPE) which encodes 3D spatial positions (temporal, height, width) for visual tokens. Z-Image's text encoder (Qwen3-4B base) uses standard 1D RoPE. The mismatch in positional encoding could hurt generation quality when VL embeddings condition the DiT.

**Experiment**: `run_encoder_test_i2i.sh` — ran i2i with qwen3vl-native on `eval/000371083.jpg` (512x512, 8 steps, cfg=1.0, seed=42):
1. Default (M-RoPE active): `i2i_qwen3vl_native.png`
2. `--disable_mrope` flag: `i2i_qwen3vl_native_no_mrope.png`

**Result**: Outputs are **byte-identical** (md5: `be47561180a6fde029c239bad52a7595`).

**Root cause**: The `disable_mrope` implementation checks `"position_ids" in inputs` but **Qwen3-VL's processor does not return `position_ids`** — it only returns `[input_ids, attention_mask, pixel_values, image_grid_thw]`. The model constructs M-RoPE position_ids internally in `Qwen3VLModel.forward()` from `image_grid_thw`. So the disable check silently passes without effect — both runs used M-RoPE.

**Fix needed**: To actually disable M-RoPE, must construct flat 1D `position_ids` externally and pass them in (the model's `forward()` accepts `position_ids` kwarg and will skip internal computation if provided). The current code needs to generate position_ids unconditionally when `disable_mrope=True`, not check if they exist in the processor output.

**Fix**: Changed `_encode_image_vl_full()` to construct `position_ids` unconditionally when `disable_mrope=True`, using `inputs["input_ids"].shape[1]` for seq_len. Shape `(3, 1, L)` with all three axes set to `arange(L)`.

**Re-run result** (after fix):
- With M-RoPE: md5 `be47561180a6fde029c239bad52a7595` (same as before — confirms determinism)
- Without M-RoPE: md5 `d3fb54f94c45bd6ec1fd066c248341b7` (different — fix works)

**Visual comparison** (input: `eval/000371083.jpg`, worker with wrench illustration):
- **With M-RoPE**: Cleaner composition. Wrench prominent and well-placed. Circular design element. Face cartoony but coherent. Better spatial layout overall.
- **Without M-RoPE**: More chaotic. Wrench position off, extra burst/explosion artifacts at bottom. Face has different expression. Less spatially coherent.

**Conclusion**: M-RoPE helps. The 3D positional encoding preserves spatial relationships between visual patches (which patch is above/below/left/right). With flat 1D positions, the model loses spatial structure — visual tokens treated as flat sequence, so spatial reasoning about the image degrades. **Keep M-RoPE enabled for VL i2i.**

This also means M-RoPE is NOT a problem for DiT conditioning — even though Z-Image's Qwen3-4B text encoder uses 1D RoPE, the DiT only sees the final hidden states, not the positional encoding scheme. The positional encoding affects how the LLM internally processes visual tokens, and M-RoPE's spatial awareness produces better hidden states for the DiT to consume.

### Generation Tests (splice_test3, splice_gen4)

#### Single-image i2i — turbo (8 steps, cfg=1.0)

Outputs: `outputs/splice_test3/turbo_*.png`, `outputs/splice_gen4/`

**What worked well:**
- **Worker (flat illustration)**: Excellent reconstruction at 512x512. Preserves color palette (yellow helmet, blue shirt, red wrench), composition, and cartoon style faithfully. Nearly identical to input.
- **Captain (photo)**: Good at 512x512 and 768x512 landscape. Preserves uniform, binoculars, harbor background, ship. Face is coherent (no deformities). Slight differences in background details but overall faithful.
- **Fashion (photo)**: Good at 512x512 and 512x768 portrait. Pink tulle dress, runway setting, audience all preserved. Pose changes (front view vs back view in original) but semantic content is correct.
- **Landscape (768x768)**: Input was a golden ornamental castle relief. Output is a golden castle scene — strong style/content preservation, impressive at higher resolution.
- **MJ art (1024x512 wide)**: Input was minimalist line art of girls on chairs. Output faithfully recreates the style and content at wide aspect ratio.
- **Painting (1024x1024)**: Input was a tall painting of Indian market scene. Output recreates market scene with women in red saris, produce stalls, architecture. Good semantic preservation at high resolution.

**Observations on resolution:**
- Multi-resolution works well across the board (512x512 up to 1024x1024).
- High resolution (1024x1024) produces coherent images — no breakdown.
- Non-square aspect ratios (768x512, 512x768, 1024x512) all work correctly.
- Input images are capped at `max_pixels=768*768` before VL encoding regardless of output resolution — the DiT handles upscaling.

#### Single-image i2i — base (50 steps, cfg=4.0)

Outputs: `outputs/splice_test3/base_*.png`

- **Worker**: Shifts to a more realistic/pop-art style — bold outlines, saturated colors, black background. Different artistic interpretation but preserves core content (man, hard hat, wrench, overalls).
- **Captain**: Higher detail than turbo. Face is coherent. Uniform details (stripes, cap badge) are sharper. Background harbor scene preserved.
- **Fashion**: Shows model from behind (matching original input perspective better than turbo). Pink dress, runway, audience all present.

Base model with CFG=4.0 produces different stylistic interpretations — more dramatic, higher contrast. Not necessarily "better" than turbo, just different. The CFG amplifies the difference between conditioned and unconditioned, pushing outputs further from the mean.

#### Multi-image composition (no text guidance)

Outputs: `outputs/splice_test3/multi_*.png`

- **Worker + fashion** (542 tokens): Worker character appears in fashion runway setting. Both concepts clearly present — worker with wrench stands among fashion models/dresses. Composition is coherent.
- **Worker + captain** (542 tokens): Worker character in captain's harbor scene. Worker with binoculars, harbor and ships in background. Clean blend of both concepts into a single scene.
- **3-way (fashion + captain + worker)** (804 tokens): Fashion model in pink dress on runway, with harbor/port in background and worker character visible. All three concepts represented, though composition is more crowded.

**Key finding**: Multi-image input produces images containing elements from all inputs, but it's more like collage/compositing than true blending — the worker character appears as a separate cartoon element pasted into the photo scene rather than the concepts being fused into a unified style/scene. The VL model encodes all images but the DiT doesn't deeply merge them.

#### Text-guided multi-image composition

Outputs: `outputs/splice_gen4/compose_*.png`

Interleaved `[image] "text" [image]` content in the VL chat message.

- **Fashion "in the style of" worker**: Fashion model in pink dress on runway, with worker character (flat cartoon style) composited into the scene. Both images clearly visible but they coexist as separate elements rather than true style transfer — the worker remains in cartoon style, the fashion scene remains photographic. The text instruction did NOT cause style transfer.
- **Captain "in the artistic style of" MJ art**: Captain scene rendered in the minimalist line-art style of the MJ input. This one shows more style influence — the captain and harbor are rendered with flat colors and simplified forms matching the MJ art style. Partial success.
- **Worker "wearing the outfit from" fashion**: Worker character in center wearing overalls (not the pink dress). Fashion models/dresses visible in background. The text instruction did NOT cause the worker to wear the fashion outfit. Both images just coexist.
- **Fashion "and" captain (plain merge)**: Fashion model on runway with harbor/port as background replacing the indoor venue. Clean scene merge. Similar to the no-text multi-image results.

**Key finding on text guidance**: The text between images has **weak influence** on how images are composed. The VL model processes the text tokens but the DiT's conditioning path doesn't strongly respond to compositional instructions like "in the style of" or "wearing the outfit from". The text tokens are a small fraction of the total sequence (a few tokens vs hundreds of visual tokens) — they get diluted.

Exception: when the text instruction aligns with what the model would naturally do (captain+MJ art style transfer partially worked because the style difference was dramatic), some influence is visible. But fine-grained compositional control ("wearing the outfit from") does not work zero-shot.

**Why**: The DiT was trained on text captions, not on compositional instructions between images. The VL model understands the instructions semantically (it's a language model), but the DiT's frozen attention layers don't know how to interpret "in the style of" as a compositional operation on the conditioning tokens. This would likely require training/finetuning.

### Eval: t2i vs i2i (run_eval_40.py)

**Setup**: 40 random images from relaion-art-lowres (seed=42, validated non-corrupt). Single qwen3vl pipeline for both:
- **t2i**: `encode_text_vl(pipe, caption)` — text caption through VL model (same spliced LLM weights)
- **i2i**: `encode_image_vl(pipe, image)` — image through VL ViT + spliced LLM
- Same seed, same resolution (matched to input AR, rounded to 16px multiples, capped 768)
- Turbo: 8 steps, cfg=1.0

**Metrics**: HPSv2.1 (aesthetics), CLIP score (prompt adherence), DINOv2 cosine sim (structural similarity to input)

**Outputs**: `outputs/eval_40_vl/` — `t2i_000.png`, `i2i_000.png`, `input_000.png` x40 + `results.json`

**Status**: Done.

**Captions**: Dense Qwen3-VL-8B captions from `somepago/relaion-art-qwen3vl-8b-caps`. Joined with local images by `_row_id` — 2457 images have both dense captions and images on disk, 40 sampled.

**Results** (40 images, turbo 8-step, cfg=1.0):

| Metric | t2i (mean ± std) | i2i (mean ± std) | diff |
|--------|------------------|-------------------|------|
| HPSv2.1 | 0.295 ± 0.025 | 0.286 ± 0.031 | -0.009 |
| CLIP score | 0.366 ± 0.031 | 0.337 ± 0.036 | -0.029 |
| DINOv2 sim | 0.720 ± 0.132 | 0.710 ± 0.123 | -0.010 |

- i2i slightly worse across all metrics, but gaps are small
- CLIP score has the biggest delta (-0.029) — t2i gets the caption directly, i2i must reconstruct content from visual features
- HPSv2.1 (aesthetics) nearly tied — both produce visually decent images
- DINOv2 sim (structural similarity to input) nearly tied — surprising, means t2i from dense caption recovers structure almost as well as i2i from the actual image

### Unified Inference CLI

Consolidated `inference.py` + `run_eval_40.py` into a single `inference.py` with unified CLI:
- Single t2i: `python inference.py --prompt "a cat"`
- Single i2i: `python inference.py --image cat.png`
- Batch i2i from folder: `python inference.py --image eval_unified/images/`
- Batch t2i from txt: `python inference.py --prompt captions.txt`
- Paired batch: `python inference.py --image eval_unified/images/ --prompt captions.txt --metrics`
- Metrics only: `python inference.py --metrics --skip_generation --output_dir outputs/some_run/`

Metrics extracted to `src/metrics.py` (HPSv2.1, CLIP score, DINOv2 cosine sim).

### Multi-Image Inference + Composition Baselines

`inference_multi_image.py` — generates images from JSONL files with interleaved image/text content.

**JSONL format** (one JSON array per line):
```jsonl
[{"img": "eval_unified/images/005.jpg"}, {"txt": "in the style of"}, {"img": "eval_unified/images/014.jpeg"}]
[{"img": "eval_unified/images/000.jpg"}, {"img": "eval_unified/images/003.jpg"}]
```

**Blend modes** (`--blend_mode`):

| Mode | Flag | Description |
|------|------|-------------|
| B3: Concat | `concat` (default) | Full interleaved encoding — all images + text in one VL forward pass. Cross-image attention in LLM. |
| B5: Weighted avg | `avg` | Encode each image SEPARATELY (no cross-attention), concatenate with per-image scaling: `[alpha * h1 ; (1-alpha) * h2]` |
| B6: Weighted scale | `scale` | Encode TOGETHER (cross-attention), then scale output visual tokens by image-of-origin using `<\|vision_start\|>`/`<\|vision_end\|>` boundaries |

`--alpha` controls first image weight (second gets 1-alpha). Default 0.5.

**Key question B5 vs B3**: Token averaging (B5) destroys cross-image attention — image A's tokens never see image B's. If B3 beats B5, LLM reasoning between visual tokens matters. If B5 ≈ B3, cross-image attention isn't contributing.

**Key question B6**: Whether per-image weighting after cross-attention produces controllable composition. Token positions correspond to input image order, but representations are entangled by layer 34.

**Implementation** (`src/diffusion.py`):
- `encode_interleaved_vl()` — B3: builds VL chat message from interleaved content, single forward pass
- `encode_weighted_avg_vl()` — B5: separate `encode_image_vl()` calls per image, concatenated with alpha scaling
- `encode_weighted_concat_vl()` — B6: single forward pass, then scales visual tokens by `_find_image_token_ranges()` using vision_start/end token IDs (151652/151653)

### Baseline Runs (baselines_feb25)

`run_baselines.sh` — overnight batch with resume support (`run_if_needed` checks for `meta.json`).

**Runs:**
1. i2i on all 84 eval images
2. t2i on all 84 eval captions (paired with images, + metrics)
3. Dense composition prompts (50 entries, `composition_prompts.jsonl`):
   - B3 concat
   - B5 avg: alpha ∈ {0.3, 0.5, 0.7, 0.9}
   - B6 scale: alpha ∈ {0.3, 0.5, 0.7, 0.9}
4. Light composition prompts (50 entries, `composition_light.jsonl`):
   - B3 concat, B5 avg 0.3, B6 scale 0.3
5. Caption-drop ablation (`composition_light_notext.jsonl` — images only, no text):
   - B3 concat, B5 avg 0.3, B6 scale 0.3

Outputs: `outputs/baselines_feb25/`

### Eval Dataset

`eval_unified/` — 84 curated images from 3 sources:
- 12 relaion-pop (real photos)
- 12 midjourney-top (generated, 1024px+)
- 12 relaion-art-lowres (real art, 256-700px)
- 24 more relaion-pop
- 24 more midjourney-top (high-res, 1632-2688px)

Structure:
```
eval_unified/
  eval.csv                          # manifest: filepath, orig_name, dataset, type, dims, caption
  images/                           # all 84 images (000.jpg — 083.jpeg)
  composition_prompts.jsonl         # 50 dense multi-image composition prompts
  composition_light.jsonl           # 50 lighter composition prompts
  composition_light_notext.jsonl    # same 50 pairs, images only (caption-drop ablation)
  multi_test.jsonl                  # 5-entry test set
```

### Resolution: 512 vs 1024

**Finding**: Z-Image-Turbo was trained/distilled at **1024x1024** native resolution. Official DiffSynth-Studio examples and HuggingFace Space both default to 1024x1024. Our baselines were initially generated at 512x512, which likely degraded quality — especially fine details like eyes and faces in portraits.

**Evidence from official code**:
- DiffSynth-Studio `ZImagePipeline.__call__` defaults: `height=1024, width=1024`
- HF Space uses 1024px with various aspect ratios (1024/1280/1536)
- HF Space uses `guidance_scale=0.0` (equivalent to `cfg_scale=1.0`), `num_inference_steps=9`
- Scheduler: `FlowMatchEulerDiscreteScheduler(shift=3.0)` — matches our setup

**Other findings from DiffSynth source**:
- Native Qwen3-4B text encoder uses `apply_chat_template(enable_thinking=True)` — adds thinking tokens to prompt. Our VL path does NOT use `enable_thinking`.
- Text encoder output: `hidden_states[-2]` (second-to-last layer), padded to `max_length=512`, filtered by attention_mask to variable-length output
- Z-Image's text encoder is essentially frozen Qwen3-4B base used as feature extractor

**Action**: Updated all inference defaults from 512→1024. AR-matching uses `round_to_16(w, h, max_size=1024)` — scales images to fit within 1024 while preserving aspect ratio, rounded to 16px multiples.

**Baseline runs at 1024px** (in `outputs/baselines_feb25/`):
- `native_t2i_512`, `t2i_all_512`, `i2i_all_512` — old 512px runs (kept for comparison)
- `native_t2i` — native Qwen3 pipeline t2i at 1024px
- `t2i_all` — VL splice t2i at 1024px
- `i2i_all` — VL splice i2i at 1024px

Script: `run_1024_baselines.sh`

**Timeout safety**: Added per-image timeout (120s for native, 180s for VL) using `signal.SIGALRM` to prevent GPU hangs. Timed-out images are skipped, generation continues.

**Result**: 1024px runs completed successfully (84 images each, all three variants). Quality dramatically improved — face/eye artifacts from 512px runs are gone. Fine details (skin texture, iris detail, hair strands) now render cleanly across all three pipelines (native t2i, VL t2i, VL i2i).

**Conclusion**: The 512px artifacts were NOT caused by our VL splice encoding — they were purely a resolution issue. The model produces clean output at its native 1024px training resolution. Our VL pipeline introduces no quality degradation compared to the native Qwen3 text encoder path.

### Variation Strength Ablation (max_pixels sweep)

Tested how VL token count (controlled via `max_pixels`) affects i2i variation strength. Lower `max_pixels` = fewer visual tokens = coarser representation = stronger variation from the original.

**Token counts** (measured on image 000, plumber illustration):

| Level | max_pixels | Tokens | Expected |
|-------|-----------|--------|----------|
| `very_strong` | 16,384 (128x128) | 82 | Very abstract |
| `strong` | 65,536 (256x256) | 82 | Loose interpretation |
| `medium` | 147,456 (384x384) | 153 | Balanced variation |
| `default` | 262,144 (512x512) | 280 | Moderate fidelity |
| `subtle` | 589,824 (768x768) | 582 | Close to original |
| `very_subtle` | 802,816 (896x896) | 760 | Near-faithful |

Note: `very_strong` and `strong` both produce 82 tokens — Qwen3-VL has a minimum patch count floor.

**Run**: 20 images x 6 levels = 120 generations. Outputs: `outputs/baselines_feb25/vary_{level}/`

Script: `run_variations.sh` (uses `--max_pixels` and `--n_samples` flags added to `inference.py`)

### Text-Guided Variation Ablation

Tested whether adding text alongside the image via `encode_interleaved_vl()` can steer the output.

**Token counts at medium (384x384)**:
- Image-only: **153 tokens**
- Image + "wearing a top hat": **158 tokens** (5 text tokens added)
- Text-only "wearing a top hat": **13 tokens**
- Empty text: **8 tokens**

**Key finding: text is ~3% of the conditioning sequence** — 5 tokens vs 153 image tokens. This is why text guidance barely works zero-shot.

**Prompts tested** (9 per image, 12 images = 108 variants):
- Style transfer: "in watercolor style", "as a pencil sketch", "in cyberpunk neon style", "as an oil painting"
- Scene modification: "at sunset", "in winter with snow", "underwater"
- Subject alteration: "but as a cat", "wearing a top hat"

**Results**: 3/108 somewhat worked (~3% hit rate):
- `000_sunset` — plumber illustration "at sunset" (color/lighting shift)
- `011_top_hat` — rhino pencil sketch "wearing a top hat"
- `015_top_hat` — orange fashion photo "wearing a top hat"
- `017_top_hat` — face closeup "wearing a top hat"

**Pattern**: Only additive/concrete modifications worked ("top hat" = single object overlay, "sunset" = color shift). Style transfers (watercolor, pencil, cyberpunk) and scene changes (winter, underwater) all failed — these require global changes that 5 text tokens can't override against 153 image tokens.

**Conclusion**: Zero-shot text-guided variation doesn't work with the current architecture because text tokens are massively outnumbered by visual tokens. Finetuning needed to amplify text influence.

Output: `outputs/baselines_feb25/vary_text/`, picks in `picks.json`.

Script: `run_text_variations.py`

### Stage 2: GRPO for Text-Guided Variations

**Goal**: Train the model to follow text instructions when generating image variations. Zero-shot text guidance failed (3/108 hit rate) because text tokens are ~3% of the conditioning sequence (5 tokens vs 153 image tokens at medium max_pixels).

**Script**: `train_grpo.py`

**What we train (two components)**:

1. **ConditioningAdapter** — 1+ trainable transformer block(s) on VL prompt_embeds (2560-dim), inserted between `encode_interleaved_vl()` output and the DiT's frozen `cap_embedder`. Self-attention lets text tokens attend to image tokens and learn to amplify the text signal. Zero-initialized output projections → starts as identity (no disruption at init).
   - Architecture: pre-norm transformer block (LayerNorm → MultiheadAttention → LayerNorm → FFN)
   - Default: 1 layer, 16 heads, FFN mult 4.0
   - Params: ~78M per layer (dim=2560)

2. **LoRA on DiT** — low-rank adapters on DiT Linear layers. Manual implementation (no HuggingFace/peft). Each target `nn.Linear` gets wrapped in `LoRALinear`: `output = original(x) + x @ A @ B * (alpha/rank)`. A is kaiming-init, B is zero-init → zero LoRA contribution at start.
   - Default: rank=32, targets all Linear layers with min dim ≥ 512 + adaLN_modulation

**Data flow during training**:
```
Input Image + Text Instruction
         │
         ▼
encode_interleaved_vl() (frozen VL model)
         │ (L, 2560) — ~158 tokens at medium (153 image + 5 text)
         ▼
ConditioningAdapter (TRAINABLE) — self-attention on conditioning tokens
         │ (L, 2560)
         ▼
DiT forward (LoRA TRAINABLE, base frozen):
  cap_embedder (frozen) → context_refiner (frozen) → main transformer (LoRA)
         │
         ▼
8-step denoising → output image
         │
         ▼
VLM Reward (Qwen3-VL-8B via vLLM) — scores instruction-following + content preservation
```

**GRPO training loop**:
1. Sample random (image, text_instruction) pair
2. Encode via VL model + pass through adapter
3. Generate K=4 images (no grad, for scoring)
4. Score with VLM reward
5. Compute group-relative advantages: `A_i = (r_i - mean(r)) / std(r)`
6. For each sample: re-run with gradients, compute clipped surrogate loss
7. Backward through adapter + LoRA, optimizer step

**Reward model**: Qwen3-VL-8B-Instruct via vLLM server. Sees (input_image, text_instruction, output_image), scores 0-10 on instruction following + content preservation. Also supports perceptual reward (LPIPS + DINOv2) as a simpler alternative that doesn't require vLLM.

**Memory budget (128GB unified, DGX Spark)**:

| Component | Estimated |
|-----------|-----------|
| DiT (6.1B, bf16) | ~13GB |
| Qwen3-VL-4B (VL splice, frozen) | ~9GB |
| VAE (frozen) | ~1GB |
| LoRA adapters (rank 32) | ~0.1GB |
| ConditioningAdapter (1 layer) | ~0.3GB |
| 8-step rollout activations | ~20-30GB |
| Qwen3-VL-8B reward (vLLM, separate process) | ~16GB |
| OS + buffers | ~5GB |
| **Total** | **~65-75GB** |

Fits comfortably. K=4 rollouts run sequentially (not parallel) so peak memory is 1 rollout.

**VLM reward server** (run before training):
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-8B-Instruct --max-model-len 4096 --port 8100
```

**Training data**: `composition.jsonl` — each line is a JSON array of `{"img": ...}` and `{"prompt": ...}` items. Images loaded from `images/` dir alongside the JSONL. Source: `/home/gnan/projects/data/datasets/laion__relaion-pop/composition.jsonl` (246 entries, images from relaion-pop ~114K images).

Current format: `[{"img": "file1"}, {"img": "file2"}, {"prompt": "text"}]` (2 images + prompt).

**TODO**: Support variable-length entries — 1 image + prompt (text-guided variation), 2 images + prompt (composition), 3+ images + prompt (multi-source composition). The `CompositionDataset` and `encode_interleaved_vl` already handle arbitrary interleaved content; the VLM reward prompt needs to adapt to variable number of references.

**Turbo diversity concern**: With cfg=1.0 and 8 steps, K=4 rollouts from different seeds may produce similar outputs, making GRPO advantages noisy. Potential mitigations: increase K, use base model (50 steps, cfg=4.0) for more diversity, or add noise to conditioning.

**VLM reward for composition**: VLM sees all 3 images (ref1, ref2, output) inline-labeled + the composition prompt. Scoring criteria: (1) follows composition prompt, (2) takes RIGHT elements from each reference, (3) avoids leaking UNWANTED elements, (4) coherent single image not collage, (5) visually plausible. Scores 0-10 integer with calibration guide (9-10 = accurate, 6-8 = imperfect, 3-5 = major issues, 0-2 = failure).

**Wandb logging** (project: `synthos-grpo`):
- Every step: `train/loss`, `train/kl`, `reward/mean`, `reward/std`, `reward/min`, `reward/max`, `reward/sample_0..K`, `grad/adapter_norm`, `grad/lora_norm`, `perf/step_time_s`
- Every `eval_every` steps: rollout images (refs + K outputs with scores as captions), prompt text

**Training command**:
```bash
python train_grpo.py \
    --dataset /home/gnan/projects/data/datasets/laion__relaion-pop/composition.jsonl \
    --reward vlm --reward_url http://localhost:8100 \
    --steps 500 --group_size 4 --lr 1e-5 \
    --lora_rank 32 --adapter_layers 1
```

### Stage 2: DiffusionNFT Training

**Script**: `train_diffusionnft.py` (previously `train_grpo.py`)

**Algorithm**: DiffusionNFT (arxiv.org/abs/2509.16117) — reward-weighted velocity matching on the forward diffusion process. Instead of backpropagating through the full 8-step denoising chain (OOMed on DGX Spark), DiffusionNFT needs only ONE DiT forward+backward per training sample:

1. Generate K images from current model (no grad, full 8-step rollout)
2. Score with VLM reward → normalize to [0,1]
3. For each generated image z_0:
   - Forward diffuse to random timestep: z_t = (1-t)*z_0 + t*ε
   - v_θ = current model prediction (single DiT forward, WITH grad + gradient checkpointing)
   - v_old = v_θ.detach() (EMA swap crashes on unified memory; approximation valid early in training)
   - Loss = r·‖v_+ - v‖² + (1-r)·‖v_- - v‖² where v_± = (1∓β)·v_old ± β·v_θ
4. Per-sample backward (frees activations between samples) → optimizer step → EMA update

**Memory solution**: gradient checkpointing (native DiT support via `use_gradient_checkpointing=True`) + per-sample gradient accumulation. Without these, backward pass at 1024x1024 OOMed even for a single sample.

**Test run** (5 steps, K=2): Completed successfully at ~47s/step, 53% memory (67.7GB / 130.7GB). Scores ranged 3-8, losses ~0.42-0.64. Output: `outputs/grpo_test/run_20260226_090425/`.

#### What we train (two components with distinct roles)

1. **CompositionModule** (previously "ConditioningAdapter") — trainable transformer block(s) sitting between `encode_interleaved_vl()` output and the DiT's frozen `cap_embedder`. The VL model encodes both input images with cross-attention in its 36 LLM layers, producing entangled visual token representations. But the VL model doesn't know what the DiT needs — it was never trained to produce "composed" conditioning. The CompositionModule learns to transform multi-image VL embeddings into a single coherent conditioning signal that the DiT can generate from. Its role is **how to mix the inputs**.
   - Architecture: pre-norm transformer block (LayerNorm → MultiheadAttention → LayerNorm → FFN)
   - Zero-initialized output projections → starts as identity (no disruption at init)
   - Default: 1 layer, 16 heads, FFN mult 4.0, ~78M params

2. **LoRA on DiT** — low-rank adapters on DiT Linear layers. The DiT was trained on text conditioning from a single image's caption, not on multi-image composed representations. LoRA teaches the DiT **how to generate from the composed signal** that the CompositionModule produces.
   - Manual implementation (no HuggingFace/peft)
   - Default: rank=32, all Linear layers with min dim ≥ 512 + adaLN_modulation, ~96M params

```
Image A ──┐
          ├── encode_interleaved_vl() (frozen VL, cross-image attention)
Image B ──┘          │
                     │ (L, 2560) — visual tokens from both images
                     ▼
             CompositionModule (TRAINABLE) — learns to compose multi-image
             embeddings into coherent conditioning for the DiT
                     │ (L, 2560)
                     ▼
             DiT forward (LoRA TRAINABLE, base frozen) — learns to generate
             from composed representations
                     │
                     ▼
             8-step denoising → output image
                     │
                     ▼
             VLM Reward (coherence scoring)
```

#### Pivot: Composition → Coherence-Based RL

**Problem**: Base model can't produce decent multi-image compositions. All K rollouts score poorly (3-5 range), so there's no positive/negative contrast signal for RL. Finetuning for composition capability from scratch is too ambitious — RL refines existing capabilities, it doesn't teach new ones.

**New objective**: Instead of text-guided composition ("combine elements from image A and B according to prompt"), train for coherent image blending:
- Mix two input images → generate a "surprise" blend
- VLM rates on coherence axes: reasonable proportions, consistent style/lighting, no artifacts, visually plausible
- The prompt text is no longer the primary control — it's just two images being blended
- Drop rollouts where max(scores) < threshold (no positive signal to learn from)

**VLM scoring criteria** (new, coherence-focused):
- Proportions/scale: are objects reasonably sized relative to each other?
- Style coherence: consistent lighting, color palette, rendering style?
- Artifacts: obvious seams, floating elements, broken anatomy?
- Visual plausibility: does this look like a real/intentional image (not a bad collage)?
- Score 1-10, single integer

**Rollout filtering**: Skip gradient update when `max(scores) < min_reward_threshold`. Start threshold at 3 (permissive), raise as model improves. This avoids learning from all-bad groups which just add gradient noise.

**Adaptive threshold idea**: Instead of fixed threshold, require at least one rollout to be >1 std above group mean. This is relative and works regardless of absolute score level.

#### DiffusionNFT test runs (lyric-surf-1, wise-oath-2)

**Run 1 (lyric-surf-1)**: 4 steps, K=4, eval set (50 pairs, no text). Scores 7-8, 0 skipped. VLM prompt was too lenient.

**Run 2 (wise-oath-2)**: Same config, stricter VLM prompt (harsh scoring, penalizes floating objects, collage effects). Added `--cond_noise_std 0.1` for diversity. Scores dropped to ~5 — all identical per group, no contrast signal. 0 skipped.

**Conclusion**: Base model's multi-image composition capability doesn't exist. All rollouts produce equally bad outputs. RL can't bridge this gap — it refines existing capabilities, doesn't teach new ones.

### Conditioning Schedule Experiment

**Script**: `experiment_cond_schedule.py`

**Idea**: Instead of conditioning on both images from step 0, introduce the second image's conditioning partway through the 8-step denoising. Early steps establish structure from one image, late steps add influence from the other.

**Three experiments per pair**:
- **A→A+B** (`a_to_ab`): Start with image A only, introduce A+B (joint encoding) at switch point
- **B→A+B** (`b_to_ab`): Start with image B only, introduce A+B at switch point
- **A→B** (`a_to_b`): Start with image A only, switch to image B only at switch point (no joint encoding)

**Switch points**: 0, 2, 4, 6, 8 (out of 8 total steps)
- `s0` = use late conditioning for ALL steps
- `s8` = use early conditioning for ALL steps (never switch)
- `s4` = early for 4 steps, late for 4 steps

**Config**: 20 pairs from `eval_unified/composition_light_notext.jsonl`, seed=42 (same noise for all gens), turbo 8 steps cfg=1.0, 1024px output.

**Output**: `outputs/cond_schedule/pair_000/` through `pair_019/` — 17 files each (2 inputs + 15 generated). Total 300 generated images.

**Viewer**: `viz/cond_schedule_viewer.py` (Streamlit, port 8503)

**Results**: TODO — visually review outputs.

### TODOs
- [x] **Evaluate t2i vs i2i quality** — `run_eval_40.py`, results in `outputs/eval_40_vl/`
- [x] **Unified inference CLI** — `inference.py` (t2i + i2i, single + batch, metrics)
- [x] **Multi-image inference** — `inference_multi_image.py` with blend modes
- [x] **Composition baselines** — B3/B5/B6 with alpha sweeps, light prompts, caption-drop ablation
- [x] **Variation strength ablation** — `run_variations.sh`, 6 max_pixels levels x 20 images
- [x] **Text-guided variation ablation** — `run_text_variations.py`, 9 text prompts x 12 images
- [x] **DiffusionNFT pipeline** — `train_diffusionnft.py`, ConditioningAdapter + LoRA + DiffusionNFT loss
- [x] Set up vLLM server for Qwen3-VL-8B reward model
- [x] Test DiffusionNFT training (5 steps, K=2, verified working)
- [x] DiffusionNFT coherence runs — ran 2 test runs, both showed base model can't compose (no RL signal)
- [x] Conditioning schedule experiment — 20 pairs × 3 experiments × 5 switch points = 300 images
- [ ] Review conditioning schedule results — visually compare switch points
- [ ] Z-Image Base model for training — may converge better with MSE loss since it's not distilled
- [ ] Analyze baseline results — compare B3 vs B5 vs B6, effect of alpha, text vs no-text

### Ideas to Explore

1. **Cut-paste collage as input**: Take elements cut/pasted from multiple images as a single composite input image. The model might blend them into a coherent output that preserves most elements from the original while smoothing out the collage seams.

2. **SigLip text embeddings for creativity**: Add SigLip text embeddings (e.g. encoding the word "creative") to the image embeddings during i2i. Could nudge the output to be more open-ended/creative while still grounded by the image tokens.

3. **Intentional channel misalignment — text as SigLip embedding**: Instead of passing text through the regular text encoder channel, encode it as a SigLip text embedding (SigLip can encode text too). The DiT expects image-like features from that pathway, so text features there might produce unexpected/creative interpretations.

4. **Noise on projection layer**: Add small noise to the SigLip projection weights or activations at inference time. Could introduce controlled variation/creativity in i2i outputs.

5. **Noise on input image**: Add noise directly to the input image before encoding through VL. Different from denoising noise — this perturbs the visual features the model receives, potentially leading to more creative reinterpretations.

## VL Splice Architecture

The core discovery of this project: Z-Image-Turbo's text encoder and Qwen3-VL-4B's LLM have **identical architecture** (36 layers, 2560-dim). Splicing Z-Image's trained weights into Qwen3-VL's LLM slot gives us zero-shot i2i.

```
═══════════════════════════════════════════════════════════════════
  Z-Image-Turbo (original t2i)
═══════════════════════════════════════════════════════════════════

  Text ──→ [ Z-Image LLM (36 layers, 2560-dim) ] ──→ (L, 2560) ──→ DiT ──→ Image
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           This IS the text encoder. Same arch as Qwen3-VL's LLM.


═══════════════════════════════════════════════════════════════════
  Qwen3-VL-4B (original VLM)
═══════════════════════════════════════════════════════════════════

  Image ──→ [ ViT (24 layers, 1024-dim) ] ──→ [ PatchMerger (4096→2560) ]─┐
                                                                           ├──→ [ Qwen LLM (36 layers, 2560-dim) ] ──→ text
  Text  ──→ [ tokenizer + embed_tokens ]───────────────────────────────────┘


═══════════════════════════════════════════════════════════════════
  VL SPLICE (what we did)
═══════════════════════════════════════════════════════════════════

  Image ──→ [ ViT (24 layers, 1024-dim) ] ──→ [ PatchMerger (4096→2560) ]─┐
                 from Qwen3-VL                     from Qwen3-VL           │
                                                                           ├──→ [ Z-Image LLM weights ] ──→ hidden_states[-2] ──→ (L, 2560) ──→ DiT ──→ Image
  Text  ──→ [ tokenizer + embed_tokens ]───────────────────────────────────┘     SPLICED INTO
                 from Qwen3-VL                                                   Qwen3-VL's LLM
```

**How it works**: `vl_model.model.language_model.load_state_dict(z_image_weights)` — literally swap the LLM weights. ViT + PatchMerger feed visual tokens (2560-dim) into the LLM, which now has Z-Image's weights that the DiT was trained to consume. Works zero-shot because the dimensions match exactly.

**Code**: `src/model_utils.py:_setup_vl_splice()` loads Qwen3-VL-4B, splices Z-Image LLM weights, attaches as `pipe.vl_model`.

**Encoding**: `src/diffusion.py:encode_image_vl()` — full VL forward pass with `output_hidden_states=True`, extracts `hidden_states[-2]` (penultimate LLM layer), filters by attention mask → `(L, 2560)` prompt embeddings for DiT.

## Layer Tap Experiments (Mar 8, 2026)

### Motivation

The standard pipeline always uses `hidden_states[-2]` (LLM layer 34, penultimate) as conditioning for DiT. But there are many internal tapping points in the VL forward pass, each carrying different information:

```
Image ──→ ViT ──→ PatchMerger ──→ [LLM Layer 0] ──→ ... ──→ [LLM Layer 34] ──→ [LLM Layer 35]
                       ↑                ↑                          ↑                  ↑
                  post_merger      early layers              BASELINE (current)    final layer
                  (raw visual)    (lightly processed)        hidden_states[-2]    hidden_states[-1]
```

`hidden_states` tuple has 37 entries: [embedding_output, layer_0, layer_1, ..., layer_35]. Index 0 = post-PatchMerger (before any LLM layer). All are (seq_len, 2560).

### Experiment 1: Single-Image Layer Tap

**Script**: `experiment_layer_tap.py`
**Output**: `outputs/layer_tap_exp/` (200 files: 20 images × 9 layers + 20 inputs)

Layers tapped: post_merger (idx 0), layer 4, 8, 12, 18, 24, 30, 34 (baseline), 35 (final).

**Finding**: Layers 12 and beyond produce interesting results. Earlier layers (post-merger, layer 4, 8) produce more abstract/noisy outputs. The transition around layer 12 is where outputs start looking coherent.

### Experiment 2: Multi-Image Blend + Layer Tap

**Script**: `experiment_layer_tap_blend.py`
**Output**: `outputs/layer_tap_blend/` (210 files: 15 pairs × 6 layers × 2 modes)

Two blend modes at each layer:
- **avg**: Encode each image separately, scale by alpha (0.3/0.7), concatenate. No cross-image attention.
- **scale**: Encode both images together (cross-attention in LLM), then scale visual tokens per-image.

Layers tapped: 12, 18, 24, 30, 34 (baseline), 35 (final).

Used image pairs from existing multi-image experiment (`multi_avg_a0.3` entries).

### Experiment 3: Multi-Image Blend + Layer Tap (Light Prompts)

**Script**: `experiment_layer_tap_blend.py --entries_file eval_unified/composition_light.jsonl --output_dir outputs/layer_tap_blend_light`
**Output**: `outputs/layer_tap_blend_light/` (210 files: 15 pairs × 6 layers × 2 modes)

Same as Experiment 2 but using light text prompts alongside image pairs. The scale mode encodes images + text together (text attends to images via causal LLM attention).

### Experiment 4: Text-Only Vision-Aware Conditioning

**Script**: `experiment_layer_tap_textonly.py`
**Output**: `outputs/layer_tap_textonly/` (90 files: 15 pairs × 6 layers)

Images + text go through full VL forward pass, but vision tokens are **stripped** from the output. Only text tokens (which attended to image tokens via causal self-attention) are kept as DiT conditioning.

**Finding**: Outputs look like regular t2i — the ~20-50 text tokens don't carry enough visual signal from attention alone. The DiT needs the actual visual tokens.

### Experiment 5: Composites Layer Tap (rough composition cleanup)

**Script**: `experiment_layer_tap_composites.py`
**Output**: `outputs/layer_tap_composites/`

Rough cut-paste composites from `eval_obj_stitch/composites/` fed through VL at different layers. Tests whether VL+DiT can "clean up" rough compositions zero-shot — the model sees the composite as input and generates a coherent version.

5 composites × 7 layers (post_merger through layer 35).

### Experiment 6: Cross-Image Attention Blocking (Isolated Blend)

**Script**: `experiment_isolated_blend.py`
**Output**: `outputs/isolated_blend/` (60 files: 15 pairs × 2 layers × 2 modes)

**Problem**: In scale blend mode, both images go through the VL LLM together — Image B's tokens attend to Image A's tokens (causal self-attention). This causes unwanted object leakage: features from one image contaminate the other.

**Solution**: Custom 4D attention mask that blocks cross-image attention while preserving text↔image attention:

```
Normal (scale mode):       Isolated (cross-image blocked):
┌─────────────────────┐    ┌─────────────────────┐
│ A attends to A  ✓   │    │ A attends to A  ✓   │
│ B attends to A  ✓   │    │ B attends to A  ✗   │  ← BLOCKED
│ B attends to B  ✓   │    │ B attends to B  ✓   │
│ Text attends all ✓  │    │ Text attends all ✓  │
└─────────────────────┘    └─────────────────────┘
```

**Implementation**: Build causal mask, then zero out cross-image regions. Convert to additive float mask (0.0=attend, -inf=block). Pre-compute position_ids with original 1D mask to avoid `get_rope_index` crash with 4D mask, then pass 4D mask + position_ids to the model.

Compares scale (normal cross-attention) vs isolated (blocked) at layers 24 and 34 for 15 image pairs.

### Experiment 7: SDEdit Composites (partial noise denoising)

**Script**: `experiment_sdedit_composites.py`
**Output**: `outputs/sdedit_composites/`

**Problem**: Current composite i2i starts from pure noise, so the spatial layout of the composite is completely lost. The model generates a new image that captures the content/style but not the spatial arrangement.

**Approach**: SDEdit-style — encode the composite through VAE to get clean latents z_0, add noise to an intermediate timestep, then denoise with VL conditioning. This preserves spatial layout while letting the model clean up seams and rough edges.

```
Composite image ──→ VAE encode ──→ z_0 (clean latents)
                                    │
                    noise ──→ scheduler.add_noise(z_0, noise, t_start) ──→ z_t
                                    │
                    VL encode ──→ prompt_embeds ──→ denoise from z_t ──→ output
```

**Denoising strength** controls how much noise:
- 0.2: Very subtle cleanup (mostly preserves composite)
- 0.4: Moderate cleanup
- 0.6: Stronger reinterpretation
- 0.8: Heavy reinterpretation (close to pure i2i)
- 1.0: Pure noise (current i2i baseline)

**Sweep**: 5 denoising strengths × 3 conditioning levels × all composites from `eval_obj_stitch/composites/`.

**Conditioning levels**:
- **No VL conditioning** (nocond): Empty text `""` through VL model — pure SDEdit spatial preservation with no semantic guidance from the composite image
- **Medium** (384x384, ~400 tokens): Coarser VL representation + SDEdit
- **Default** (768x768, ~1000 tokens): Fine VL representation + SDEdit

The nocond row isolates the SDEdit spatial preservation effect from VL semantic guidance. Comparing nocond vs medium/default shows how much the VL conditioning contributes beyond just the noised VAE latents.

### Experiment 8: Text Before vs After Image Embeddings

**Script**: `experiment_text_before.py`
**Output**: `outputs/text_before/` (30 files: 15 pairs × 2 modes)

**Idea**: In causal LLM attention, token order determines what can attend to what:
- **Text-after** (current): `[img_A] [img_B] [text]` — image tokens are processed blind to text, text attends to images but can't influence how they're encoded
- **Text-before** (new): `[text] [img_A] [img_B]` — image tokens attend to preceding text via causal self-attention, so visual representations are "steered" by the text context

If text-before produces visibly different/better results, it means the LLM is actually modulating visual feature processing based on the text context — the text "primes" the image encoding rather than being a weak afterthought (~3% of tokens).

**Setup**: Scale mode, layer 34 (baseline), alpha=0.3, light prompts from `composition_light.jsonl`. Same settings as layer_tap_blend_light baseline — only variable is text position.

### Viewer

All experiments viewable in Streamlit: `viz/cond_schedule_viewer.py` (port 8503). Tabs: Cond Schedule, Blend Modes, Variation Strength, Text-Guided Variations, Layer Tap, Layer Tap Blend, Layer Tap Blend (Light), Text-Only (Vision-Aware), Composites, SDEdit Composites, Isolated Blend, Text Before vs After, Layer Tap Text.
