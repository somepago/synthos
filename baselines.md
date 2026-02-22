# Baselines: Creative Image Interpolation & Composition

Papers relevant to Synthos — creative interpolation via SigLIP2-conditioned DiT with LoRA training.

## Latent/LoRA Interpolation (closest to our approach)

**DiffMorpher** (CVPR 2024) — Zhang et al.
Fits two separate LoRAs to two input images, then interpolates in LoRA parameter space + latent noise space for smooth morphing. Also proposes attention interpolation/injection and adaptive normalization. Closest prior art — key difference is we use SigLIP2 image conditioning natively in the DiT rather than per-image LoRA fitting at inference time.
- [arXiv:2312.07409](https://arxiv.org/abs/2312.07409) | [Project page](https://kevin-thu.github.io/DiffMorpher_page/)

**Interpolating between Images with Diffusion Models** (ICML 2023 Workshop) — Wang & Golland
Zero-shot interpolation in latent space at decreasing noise levels, conditioned on interpolated text embeddings via textual inversion. Uses CLIP-based candidate selection.
- [arXiv:2307.12560](https://arxiv.org/abs/2307.12560) | [Project page](https://clintonjwang.github.io/interpolation)

**Linear Combinations of Latents in Diffusion Models** (2024)
Theoretical analysis of spherical vs linear vs norm-aware interpolation in diffusion latent space. Shows when each method preserves semantic meaning.
- [arXiv:2408.08558](https://arxiv.org/abs/2408.08558)

**Image Interpolation with Score-based Riemannian Metrics** (2025)
Uses the intrinsic manifold structure of pretrained diffusion models for geodesic interpolation — no retraining needed.
- [arXiv:2504.20288](https://arxiv.org/abs/2504.20288)

## Concept Blending

**FreeBlend** (Feb 2025)
Training-free concept blending via staged feedback-driven interpolation between latents. Stepwise increasing interpolation with auxiliary latent updates. Worth studying for interpolation schedules.
- [arXiv:2502.05606](https://arxiv.org/abs/2502.05606)

**Blending Concepts with Text-to-Image Diffusion Models** (Jun 2025)
Compares four blending methods exploiting different diffusion pipeline stages (prompt scheduling, embedding interpolation, layer-wise conditioning). No fine-tuning needed.
- [arXiv:2506.23630](https://arxiv.org/abs/2506.23630)

**Black-Scholes-Inspired Concept Blending** (2025)
Applies financial math (Black-Scholes) to derive optimal concept blending schedules in diffusion.
- [arXiv:2405.13685](https://arxiv.org/abs/2405.13685)

## Image Morphing / Transitions

**FreeMorph** (ICCV 2025) — Cao et al.
Tuning-free image morphing using guidance-aware spherical interpolation and step-oriented self-attention blending.
- [ICCV 2025 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Cao_FreeMorph_Tuning-Free_Generalized_Image_Morphing_with_Diffusion_Model_ICCV_2025_paper.pdf)

**DreamMover** (ECCV 2024)
Leverages diffusion priors for image interpolation with large motion — handles significant geometric changes between frames.
- [Springer](https://link.springer.com/chapter/10.1007/978-3-031-72633-0_19)

## Composition / Editing

**Composer: Creative and Controllable Image Synthesis** (ICML 2023)
Decomposes images into composable conditions (style, structure, semantics) and recomposes them with controllable creativity.
- [ICML 2023 PDF](https://proceedings.mlr.press/v202/huang23b/huang23b.pdf)

**Latent Diffusion Multi-Dimension Explorer** (Sep 2025)
Framework for direct manipulation of conceptual/spatial representations in latent space — enables concept blending and dynamic generation.
- [arXiv:2509.22038](https://arxiv.org/abs/2509.22038)
