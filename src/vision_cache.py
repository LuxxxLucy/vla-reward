"""
Vision-cached inference for ensemble speedup.

Encodes video frames through the vision tower ONCE, then runs
multiple text prompts against the cached vision embeddings.

This avoids re-encoding the same frames N times for an N-prompt ensemble.

Usage:
    cache = VisionCache(model, processor)
    cache.encode_frames(pil_images)  # expensive, done once
    lp1 = cache.log_prob_true(prompt1)  # cheap, reuses vision
    lp2 = cache.log_prob_true(prompt2)  # cheap, reuses vision
"""

import numpy as np
from PIL import Image

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


class VisionCache:
    """Cache vision tower outputs for reuse across multiple prompts."""

    def __init__(self, model, processor, true_token_id=3007, false_token_id=3557):
        self._model = model
        self._processor = processor
        self._true_id = true_token_id
        self._false_id = false_token_id
        self._cached_vision_hidden = None
        self._cached_deepstack = None
        self._cached_grid_thw = None
        self._cached_pixel_values = None
        self._cached_images = None
        self._n_images = 0

    def encode_frames(self, pil_images: list[Image.Image]):
        """Encode images through vision tower. Call once per prefix."""
        import mlx.core as mx
        from mlx_vlm import prepare_inputs

        self._n_images = len(pil_images)
        self._cached_images = pil_images

        # We need to do a dummy prepare_inputs to get pixel_values and grid_thw
        dummy_prompt = IMAGE_PLACEHOLDER * len(pil_images) + "\ndummy"
        inputs = prepare_inputs(self._processor, images=pil_images, prompts=dummy_prompt)

        pixel_values = inputs["pixel_values"]
        grid_thw = inputs.get("image_grid_thw")

        # Run vision tower only
        dtype = self._model.vision_tower.patch_embed.proj.weight.dtype
        pv = pixel_values.astype(dtype)
        hidden_states, deepstack = self._model.vision_tower(pv, grid_thw)
        mx.eval(hidden_states)
        if deepstack is not None:
            mx.eval(deepstack)

        self._cached_vision_hidden = hidden_states
        self._cached_deepstack = deepstack
        self._cached_grid_thw = grid_thw
        self._cached_pixel_values = pixel_values

    def _run_with_prompt(self, prompt_text: str):
        """Run the language model with cached vision + new prompt text."""
        import mlx.core as mx
        from mlx_vlm import prepare_inputs

        raw_prompt = IMAGE_PLACEHOLDER * self._n_images + "\n" + prompt_text
        inputs = prepare_inputs(self._processor, images=self._cached_images, prompts=raw_prompt)

        input_ids = inputs["input_ids"]

        # Get text embeddings
        inputs_embeds = self._model.language_model.model.embed_tokens(input_ids)

        # Merge with CACHED vision hidden states (skip re-encoding)
        inputs_embeds, image_mask = self._model.merge_input_ids_with_image_features(
            self._cached_vision_hidden,
            inputs_embeds,
            input_ids,
            self._model.config.image_token_index,
            self._model.config.video_token_index,
        )

        image_mask = image_mask[..., 0]

        # Compute position IDs
        image_grid_thw = self._cached_grid_thw
        mask = inputs.get("attention_mask")
        position_ids, rope_deltas = self._model.language_model.get_rope_index(
            input_ids, image_grid_thw, None, mask
        )
        self._model.language_model._position_ids = position_ids
        self._model.language_model._rope_deltas = rope_deltas

        if self._cached_deepstack is not None:
            mx.eval(self._cached_deepstack)
        deepstack_visual_embeds = self._cached_deepstack

        # Run language model
        logits = self._model.language_model(
            input_ids,
            mask=mask,
            cache=None,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=image_mask,
            deepstack_visual_embeds=deepstack_visual_embeds,
            pixel_values=self._cached_pixel_values,
            image_grid_thw=image_grid_thw,
        )

        last_logits = logits.logits[0, -1, :]
        log_probs = last_logits - mx.logsumexp(last_logits)
        mx.eval(log_probs)
        return log_probs

    def log_prob_true(self, prompt_text: str) -> float:
        lp = self._run_with_prompt(prompt_text)
        return lp[self._true_id].item()

    def log_prob_false(self, prompt_text: str) -> float:
        lp = self._run_with_prompt(prompt_text)
        return lp[self._false_id].item()

    def log_prob_both(self, prompt_text: str) -> tuple[float, float]:
        lp = self._run_with_prompt(prompt_text)
        return lp[self._true_id].item(), lp[self._false_id].item()
