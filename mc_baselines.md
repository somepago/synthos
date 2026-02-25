# Baselines: Multi-Image Composition & Visual Breeding

Papers and tools relevant to language-guided multi-image composition via VL reasoning.

## Interactive Evolutionary / Breeding Systems

**Picbreeder** (Evolutionary Computation 2011) — Secretan et al.
Collaborative online platform where users evolve images by selecting from populations of CPPN-generated candidates. Key insight: open-ended exploration via branching lineages discovers images unreachable by direct optimization. Foundational work on interactive evolutionary image generation.
- [ACM DL](https://dl.acm.org/doi/10.1162/EVCO_a_00030) | [Project](http://picbreeder.org/behind.php)

**Artbreeder** (2018–present) — Joel Simon
Successor to Picbreeder/Ganbreeder. Users breed images by interpolating latent vectors in BigGAN/StyleGAN, later SDXL. Adjustable "genes" (latent directions) control attributes. Collaborative: anyone can continue anyone's lineage. Mechanism is purely geometric (lerp/slerp in latent space) — no semantic understanding of image content.
- [artbreeder.com](https://www.artbreeder.com/) | [Wikipedia](https://en.wikipedia.org/wiki/Artbreeder)

**Collaborative Interactive Evolution in GAN Latent Space** (2024) — Bontrager et al.
Extends Picbreeder-style collaborative evolution to GAN latent spaces. Studies how evolutionary computing explores the latent space of deep generative models for art creation.
- [arXiv:2403.19620](https://arxiv.org/abs/2403.19620)

**ImageBreeder** (GECCO 2025) — Sobania et al.
Evolutionary inference-time framework for diffusion models. Maintains a population of images per prompt, scores with ImageReward, applies selection + 10 variation operators (pixel blending, latent mutations). Shows evolutionary search outperforms random sampling on 75%+ of benchmarks. No training — purely inference-time optimization.
- [ACM DL](https://dl.acm.org/doi/10.1145/3712256.3726439) | [GitHub](https://github.com/domsob/ImageBreeder)

## Multi-Image Conditioning in Diffusion Models

**IP-Adapter** (Aug 2023) — Ye et al.
Image prompt adapter for pretrained T2I diffusion. CLIP image features → decoupled cross-attention (separate from text cross-attention). ~417M adapter params, trained on image-text pairs. Multi-image support via averaging CLIP embeddings. Limitation: CLIP is a contrastive encoder, no reasoning about image content — multi-image composition is feature averaging, not semantic combination.
- [arXiv:2308.06721](https://arxiv.org/abs/2308.06721) | [Project](https://ip-adapter.github.io/) | [GitHub](https://github.com/tencent-ailab/IP-Adapter)

**MS-Diffusion** (ICLR 2025) — Wang et al.
Multi-subject zero-shot image personalization with layout guidance. Grounding resampler maintains per-subject detail fidelity; multi-subject cross-attention ensures each subject condition acts on specific spatial regions. Requires layout annotations (bounding boxes) and trained grounding modules.
- [arXiv:2406.07209](https://arxiv.org/abs/2406.07209) | [GitHub](https://github.com/MS-Diffusion/MS-Diffusion) | [Project](https://ms-diffusion.github.io/)

**Canvas-to-Image** (Nov 2025) — Dalva et al. (Snap Research)
Unified framework: all controls (subject refs, pose, layout, text) consolidated into a single canvas RGB image. Multi-Task Canvas Training fine-tunes diffusion model to reason across tasks from the composite input. Strong results on multi-person composition, pose control, layout constraints. Requires heavy training on curated multi-task datasets.
- [arXiv:2511.21691](https://arxiv.org/abs/2511.21691) | [Project](https://snap-research.github.io/canvas-to-image/)

## Concept Blending in Diffusion

**How to Blend Concepts in Diffusion Models** (Jul 2024)
Compares four blending methods exploiting different diffusion pipeline stages (prompt scheduling, embedding interpolation, layer-wise conditioning). Training-free. Shows modern diffusion models have inherent blending capability across diverse concept categories.
- [arXiv:2407.14280](https://arxiv.org/abs/2407.14280)

**Diffusion Blend: Inference-Time Multi-Preference Alignment** (May 2025)
Fine-tunes diffusion model once, then at inference generates images aligned with any user-specified linear combination of rewards — no per-combination retraining. Addresses multi-preference alignment without repeated fine-tuning.
- [arXiv:2505.18547](https://arxiv.org/abs/2505.18547)

## Evolutionary + Diffusion Crossover

**Diffusion Models are Evolutionary Algorithms** (Oct 2024)
Draws theoretical parallels between diffusion denoising and evolutionary search. Shows diffusion's iterative refinement can be interpreted as population-based optimization.
- [arXiv:2410.02543](https://arxiv.org/abs/2410.02543)

**Inference-Time Alignment via Evolutionary Algorithms** (Jun 2025)
Uses evolutionary algorithms to align diffusion model outputs at inference time without retraining. Population-based search over denoising trajectories.
- [arXiv:2506.00299](https://arxiv.org/abs/2506.00299)

## Gap Analysis

| Method | Multi-Image | Semantic Understanding | Language-Guided | Training Required |
|--------|------------|----------------------|----------------|------------------|
| Artbreeder | Yes (2-parent breeding) | No (latent interpolation) | No | No |
| IP-Adapter | Yes (averaged CLIP) | Shallow (CLIP features) | Partial (text + image separate) | Yes (~417M params) |
| MS-Diffusion | Yes (with layout) | Shallow (CLIP features) | Partial (text + layout) | Yes |
| Canvas-to-Image | Yes (canvas) | Moderate (trained jointly) | Yes | Yes (heavy) |
| ImageBreeder | No (single prompt) | No (reward-based selection) | No | No |
| **VL-Splice (ours)** | Yes (N-parent, native) | Deep (36-layer LLM reasoning) | Yes (natural language) | Minimal (optional LoRA) |

Key differentiator: VL model processes multiple images through full LLM self-attention stack — visual tokens from image A attend to tokens from image B across 36 layers. The composition is the output of *reasoning*, not interpolation or feature averaging. Language instructions guide the reasoning process.
