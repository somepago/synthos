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

### TODOs
- [x] **Evaluate t2i vs i2i quality** — `run_eval_40.py`, results in `outputs/eval_40_vl/`
- [ ] Investigate `max_pixels` effect: old `test_qwen3vl.py` resized to 512x512 before encoding (~334 visual tokens), current code caps at 768*768 (~500-750 tokens). Lower token count may produce cleaner outputs — test 512*512 cap vs 768*768.
- [ ] Z-Image Base model for training — may converge better with MSE loss since it's not distilled
- [ ] Add relaion-pop dataset (`/home/gnan/projects/data/datasets/laion__relaion-pop/`) — higher resolution images
- [ ] GRPO with LPIPS + SSIM + DINOv2 reward
- [ ] Text-guided composition training: finetune (LoRA?) to make text instructions between images actually control composition
- [ ] Multi-image composition as a research direction — zero-shot compositing puts elements from multiple images into one scene but doesn't truly blend styles/concepts. Training needed for real fusion.
