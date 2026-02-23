"""
Minimal HPSv3 implementation that bypasses trainer dependency issues.

HPSv3 is based on Qwen2-VL-7B-Instruct with a reward head (rm_head).
This implementation loads the model directly without the problematic imports.
"""

import torch
import torch.nn as nn
from PIL import Image
from typing import Optional, List
import huggingface_hub


# ---- Prompt templates from HPSv3 ----
INSTRUCTION = """
You are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best.

**Visual Quality:**
Evaluate the overall visual quality of the image. The following sub-dimensions should be considered:
- **Reasonableness:** The image should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The image should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible.

**Text Alignment:**
Assess how well the image matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the image (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the image adheres to this style.
- **Contextual Consistency**: Assess whether the background, setting, and surrounding elements in the image logically fit the scenario described in the prompt. The environment should support and enhance the subject without contradictions.
- **Attribute Fidelity**: Check if specific attributes mentioned in the prompt (e.g., colors, clothing, accessories, expressions, actions) are faithfully represented in the image. Minor deviations may be acceptable, but critical attributes should be preserved.
- **Semantic Coherence**: Evaluate whether the overall meaning and intent of the prompt are captured in the image. The generated content should not introduce elements that conflict with or distort the original description.
Textual prompt - {text_prompt}


"""

PROMPT_WITH_SPECIAL_TOKEN = """
Please provide the overall ratings of this image: <|Reward|>

END
"""


def _convert_hpsv3_state_dict(state_dict, base_model_state_dict):
    """
    Convert HPSv3 checkpoint keys to match base Qwen2VLForConditionalGeneration structure.

    HPSv3 checkpoint has:
      - model.embed_tokens.weight
      - model.layers.X...
      - visual.X...
      - rm_head.X...

    Base model has:
      - model.language_model.embed_tokens.weight
      - model.language_model.layers.X...
      - model.visual.X...

    We need to map HPSv3 keys to base model keys.
    """
    new_state_dict = {}
    rm_head_dict = {}

    for key, value in state_dict.items():
        if key.startswith("rm_head."):
            # Keep rm_head separate
            rm_head_dict[key] = value
        elif key.startswith("model."):
            # Map model.X -> model.language_model.X
            new_key = key.replace("model.", "model.language_model.", 1)
            new_state_dict[new_key] = value
        elif key.startswith("visual."):
            # Map visual.X -> model.visual.X
            new_key = "model." + key
            new_state_dict[new_key] = value
        elif key.startswith("lm_head."):
            # lm_head stays the same
            new_state_dict[key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict, rm_head_dict


class Qwen2VLRewardModel(nn.Module):
    """
    Qwen2-VL with a reward head for HPSv3 scoring.

    Wraps Qwen2VLForConditionalGeneration and adds rm_head.
    """

    def __init__(self, base_model, output_dim=2):
        super().__init__()
        # Store the full base model
        self.base_model = base_model
        self.config = base_model.config

        # RankNet-style reward head (matches HPSv3 checkpoint)
        hidden_size = self.config.hidden_size
        self.rm_head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(1024, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )
        self.rm_head.to(torch.float32)

        self.special_token_ids = None

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        # Access model components through base_model
        model = self.base_model.model  # Qwen2VLModel
        visual = model.visual
        language_model = model.language_model

        # Get input embeddings
        inputs_embeds = language_model.embed_tokens(input_ids)

        # Process images if provided
        if pixel_values is not None:
            pixel_values = pixel_values.type(visual.get_dtype())
            image_embeds = visual(pixel_values, grid_thw=image_grid_thw)

            # Replace image tokens with image embeddings
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        attention_mask = attention_mask.to(inputs_embeds.device)

        # Forward through the language model
        outputs = language_model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state  # [B, L, D]

        # Apply reward head
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            logits = self.rm_head(hidden_states)  # [B, L, output_dim]

        batch_size = input_ids.shape[0]

        # Get sequence lengths (position of last non-pad token)
        if self.config.pad_token_id is not None:
            sequence_lengths = (
                torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            )
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(logits.device)
        else:
            sequence_lengths = -1

        # Pool using special token position if available
        if self.special_token_ids is not None:
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (input_ids == special_token_id)
            pooled_logits = logits[special_token_mask, ...]
            pooled_logits = pooled_logits.view(batch_size, -1)
        else:
            # Use last token
            pooled_logits = logits[
                torch.arange(batch_size, device=logits.device), sequence_lengths
            ]

        return {"logits": pooled_logits}


class HPSv3Scorer:
    """
    HPSv3 Human Preference Score scorer.

    Usage:
        scorer = HPSv3Scorer(device="cuda")
        score = scorer.score(image, prompt)
    """

    def __init__(self, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy load the model and processor."""
        if self._model is not None:
            return

        print("  Loading HPSv3 (Qwen2-VL-7B)...")

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        # Load processor first and add special token
        self._processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            padding_side="right"
        )
        special_tokens = ["<|Reward|>"]
        self._processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )

        # Load base model with automatic device placement for memory efficiency
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=self.dtype,
            attn_implementation="sdpa",  # Use SDPA instead of flash attention
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        # Resize embeddings for new token BEFORE loading checkpoint
        base_model.resize_token_embeddings(len(self._processor.tokenizer))

        # Create reward model wrapper
        self._model = Qwen2VLRewardModel(base_model, output_dim=2)
        self._model.special_token_ids = self._processor.tokenizer.convert_tokens_to_ids(special_tokens)

        # Load HPSv3 checkpoint
        checkpoint_path = huggingface_hub.hf_hub_download(
            "MizzenAI/HPSv3", "HPSv3.safetensors", repo_type="model"
        )

        import safetensors.torch
        hpsv3_state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")

        # Convert HPSv3 checkpoint keys to match our model structure
        converted_state_dict, rm_head_dict = _convert_hpsv3_state_dict(
            hpsv3_state_dict, base_model.state_dict()
        )

        # Load base model weights
        missing, unexpected = base_model.load_state_dict(converted_state_dict, strict=False)
        if missing:
            # Filter out expected missing (lm_head may differ)
            missing = [k for k in missing if not k.startswith("lm_head")]
            if missing:
                print(f"  Warning: Missing base model keys: {len(missing)}")
        if unexpected:
            print(f"  Warning: Unexpected base model keys: {len(unexpected)}")

        # Load rm_head weights
        rm_missing, rm_unexpected = self._model.rm_head.load_state_dict(
            {k.replace("rm_head.", ""): v for k, v in rm_head_dict.items()},
            strict=False
        )
        if rm_missing:
            print(f"  Warning: Missing rm_head keys: {rm_missing}")

        self._model.eval()
        self._model.rm_head.to(self.device)

        print("  HPSv3 loaded successfully!")

    def _prepare_input(self, image: Image.Image, prompt: str):
        """Prepare input for the model."""
        max_pixels = 256 * 28 * 28

        # Format the message
        text_content = INSTRUCTION.format(text_prompt=prompt) + PROMPT_WITH_SPECIAL_TOKEN

        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": max_pixels,
                        "max_pixels": max_pixels,
                    },
                    {
                        "type": "text",
                        "text": text_content,
                    },
                ],
            }
        ]

        # Process with the processor
        text = self._processor.apply_chat_template(
            [message], tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(
            text=text,
            images=[image.convert("RGB")],
            padding=True,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs

    @torch.inference_mode()
    def score(self, image: Image.Image, prompt: str = "") -> float:
        """
        Compute HPSv3 score for an image.

        Args:
            image: PIL Image to score
            prompt: Text prompt used to generate the image

        Returns:
            HPSv3 score (higher = better, typically 0-10 range)
        """
        self._load_model()

        inputs = self._prepare_input(image, prompt)

        outputs = self._model(
            return_dict=True,
            **inputs
        )

        logits = outputs["logits"]
        # HPSv3 outputs [mu, sigma] for uncertainty estimation, we use mu
        score = logits[0, 0].item()

        return score

    def score_batch(self, images: List[Image.Image], prompts: List[str]) -> List[float]:
        """Score multiple images (processes one at a time for simplicity)."""
        return [self.score(img, prompt) for img, prompt in zip(images, prompts)]


# Convenience function
_scorer = None

def hpsv3_score(image: Image.Image, prompt: str = "", device: str = "cuda") -> float:
    """
    Compute HPSv3 score for an image.

    Args:
        image: PIL Image to score
        prompt: Text prompt used to generate the image
        device: Device to run on

    Returns:
        HPSv3 score (higher = better, typically 0-10 range)
    """
    global _scorer
    if _scorer is None:
        _scorer = HPSv3Scorer(device=device)
    return _scorer.score(image, prompt)


if __name__ == "__main__":
    # Test the implementation
    import numpy as np

    # Create a random test image
    test_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

    scorer = HPSv3Scorer(device="cuda")
    score = scorer.score(test_img, "a beautiful landscape")
    print(f"HPSv3 score: {score}")
