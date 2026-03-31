"""
VLM backends for vla-reward experiments.

Adapted from philfung/reward-scope/backends.py with additions:
  1. log_prob_false() — reads P("False") from the same forward pass as P("True")
  2. OpenAICompatibleBackend — works with Fireworks AI, Together AI, OpenAI, etc.

Usage:
    # Local (2B/4B on Mac MPS):
    backend = QwenVLBackend("Qwen/Qwen3-VL-2B-Instruct")

    # Fireworks AI (8B via API):
    backend = make_backend("fireworks-8b")  # reads FIREWORKS_API_KEY env var

    # Any OpenAI-compatible API:
    backend = OpenAICompatibleBackend(base_url="...", api_key="...", model="...")
"""

import os
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image

# Auto-load .env from project root
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    import cv2
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _frame_to_jpeg_b64(frame: np.ndarray) -> str:
    import cv2, base64
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class VLMBackend(ABC):
    @abstractmethod
    def log_prob_true(self, frames: list[np.ndarray], prompt_text: str) -> float:
        """Return log P(" True") from the output logit distribution."""

    @abstractmethod
    def log_prob_false(self, frames: list[np.ndarray], prompt_text: str) -> float:
        """Return log P(" False") from the output logit distribution."""

    def log_prob_both(self, frames: list[np.ndarray], prompt_text: str) -> tuple[float, float]:
        """Return (log P("True"), log P("False")) in one forward pass.

        Default: two separate calls. Subclasses override for efficiency.
        """
        return self.log_prob_true(frames, prompt_text), self.log_prob_false(frames, prompt_text)


# ---------------------------------------------------------------------------
# Local Qwen backend (2B / 4B on Mac MPS)
# ---------------------------------------------------------------------------

class QwenVLBackend(VLMBackend):
    """Local Qwen3-VL backend via HuggingFace transformers + PyTorch MPS.

    Reads both " True" and " False" logits in a single forward pass.
    use_chat_template=False matches the paper's best configuration (Section 5.4).
    """

    _IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: str = "auto",
        use_chat_template: bool = False,
        torch_dtype: str = "auto",
    ):
        import torch
        from transformers import AutoProcessor

        if "Qwen3" in model_name:
            try:
                from transformers import Qwen3VLForConditionalGeneration as ModelClass
            except ImportError:
                from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass

        self.model_name = model_name
        self.use_chat_template = use_chat_template

        print(f"Loading {model_name} …")
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = ModelClass.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device
        )
        self._model.eval()

        # Resolve token IDs — must use " True" and " False" with leading space.
        # Wrong token IDs silently give near-zero probability on every query.
        true_ids = self._processor.tokenizer.encode(" True", add_special_tokens=False)
        false_ids = self._processor.tokenizer.encode(" False", add_special_tokens=False)
        assert len(true_ids) == 1, f"' True' should be a single token, got {true_ids}"
        assert len(false_ids) == 1, f"' False' should be a single token, got {false_ids}"
        self._true_token_id = true_ids[0]
        self._false_token_id = false_ids[0]
        print(f"  Loaded. ' True'={self._true_token_id}  ' False'={self._false_token_id}")

    def _build_inputs(self, frames: list[np.ndarray], prompt_text: str):
        import torch
        pil_images = [_frame_to_pil(f) for f in frames]
        if self.use_chat_template:
            content = [{"type": "image", "image": img} for img in pil_images]
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = self._IMAGE_PLACEHOLDER * len(pil_images) + "\n" + prompt_text
        inputs = self._processor(text=[text], images=pil_images, padding=True, return_tensors="pt")
        device = next(self._model.parameters()).device
        return {k: v.to(device) for k, v in inputs.items()}

    def _get_logits(self, frames: list[np.ndarray], prompt_text: str):
        import torch
        inputs = self._build_inputs(frames, prompt_text)
        with torch.no_grad():
            outputs = self._model(**inputs)
        logits = outputs.logits[0, -1, :]
        return torch.nn.functional.log_softmax(logits, dim=-1)

    def log_prob_true(self, frames, prompt_text):
        return self._get_logits(frames, prompt_text)[self._true_token_id].item()

    def log_prob_false(self, frames, prompt_text):
        return self._get_logits(frames, prompt_text)[self._false_token_id].item()

    def log_prob_both(self, frames, prompt_text):
        """Single forward pass returns both logits."""
        lp = self._get_logits(frames, prompt_text)
        return lp[self._true_token_id].item(), lp[self._false_token_id].item()


# ---------------------------------------------------------------------------
# MLX backend (fast local inference on Apple Silicon)
# ---------------------------------------------------------------------------

class MLXQwenVLBackend(VLMBackend):
    """Local Qwen3-VL via mlx-vlm. Much faster than PyTorch MPS on Apple Silicon.

    Uses raw mode (no chat template) to match the paper's best configuration.
    """

    _IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct"):
        from mlx_vlm import load, prepare_inputs
        import mlx.core as mx

        self.model_name = model_name
        self._prepare_inputs = prepare_inputs
        self._mx = mx

        print(f"Loading {model_name} (MLX) …")
        self._model, self._processor = load(model_name)

        # Resolve token IDs via the tokenizer
        tokenizer = self._processor.tokenizer if hasattr(self._processor, 'tokenizer') else self._processor
        true_ids = tokenizer.encode(" True", add_special_tokens=False)
        false_ids = tokenizer.encode(" False", add_special_tokens=False)
        assert len(true_ids) == 1, f"' True' should be a single token, got {true_ids}"
        assert len(false_ids) == 1, f"' False' should be a single token, got {false_ids}"
        self._true_token_id = true_ids[0]
        self._false_token_id = false_ids[0]
        print(f"  Loaded. ' True'={self._true_token_id}  ' False'={self._false_token_id}")

    def _get_log_probs(self, frames: list[np.ndarray], prompt_text: str):
        """Single forward pass, return full log-prob vector at last token."""
        mx = self._mx
        pil_images = [_frame_to_pil(f) for f in frames]
        raw_prompt = self._IMAGE_PLACEHOLDER * len(pil_images) + "\n" + prompt_text
        inputs = self._prepare_inputs(self._processor, images=pil_images, prompts=raw_prompt)
        output = self._model(**inputs)
        logits = output.logits[0, -1, :]
        log_probs = logits - mx.logsumexp(logits)
        mx.eval(log_probs)
        return log_probs

    def log_prob_true(self, frames, prompt_text):
        return self._get_log_probs(frames, prompt_text)[self._true_token_id].item()

    def log_prob_false(self, frames, prompt_text):
        return self._get_log_probs(frames, prompt_text)[self._false_token_id].item()

    def log_prob_both(self, frames, prompt_text):
        lp = self._get_log_probs(frames, prompt_text)
        return lp[self._true_token_id].item(), lp[self._false_token_id].item()


# ---------------------------------------------------------------------------
# OpenAI-compatible API backend (Fireworks, Together AI, OpenAI, etc.)
# ---------------------------------------------------------------------------

class OpenAICompatibleBackend(VLMBackend):
    """Any OpenAI-compatible VLM API with logprobs support.

    Works with Fireworks AI, Together AI, OpenAI, etc.
    Requires: pip install openai
    """

    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import OpenAI
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        print(f"API backend: {model} via {base_url}")

    def _image_content(self, frame: np.ndarray) -> dict:
        b64 = _frame_to_jpeg_b64(frame)
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

    def _query_logprobs(self, frames: list[np.ndarray], prompt_text: str) -> dict[str, float]:
        """Return {token: logprob} for the top first-generated tokens."""
        content = [self._image_content(f) for f in frames]
        content.append({"type": "text", "text": prompt_text})
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=2,
            temperature=0.0,
            logprobs=True,
            top_logprobs=5,  # Fireworks caps at 5
        )
        token_logprobs = response.choices[0].logprobs
        if not token_logprobs or not token_logprobs.content:
            return {}
        return {lp.token: lp.logprob for lp in token_logprobs.content[0].top_logprobs}

    def _find_token(self, logprob_map: dict[str, float], target: str, fallback: float = -20.0) -> float:
        for token, lp in logprob_map.items():
            if token.strip().lower() == target.lower():
                return lp
        return fallback

    def log_prob_true(self, frames, prompt_text):
        return self._find_token(self._query_logprobs(frames, prompt_text), "True")

    def log_prob_false(self, frames, prompt_text):
        return self._find_token(self._query_logprobs(frames, prompt_text), "False")

    def log_prob_both(self, frames, prompt_text):
        lp_map = self._query_logprobs(frames, prompt_text)
        return (
            self._find_token(lp_map, "True"),
            self._find_token(lp_map, "False"),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Presets for common providers
# ---------------------------------------------------------------------------

FIREWORKS = {
    "base_url": "https://api.fireworks.ai/inference/v1",
    "env_key": "FIREWORKS_API_KEY",
    "models": {
        "8b": "accounts/fireworks/models/qwen3-vl-8b-instruct",
        "32b": "accounts/fireworks/models/qwen3-vl-32b-instruct",
        "72b": "accounts/fireworks/models/qwen2p5-vl-72b-instruct",
    },
}

TOGETHER = {
    "base_url": "https://api.together.xyz/v1",
    "env_key": "TOGETHER_API_KEY",
    "models": {
        "8b": "Qwen/Qwen3-VL-8B-Instruct",
    },
}


def make_backend(
    kind: str,
    model: str | None = None,
    api_key: str | None = None,
) -> VLMBackend:
    """Create a backend.

    kind:
        "qwen-2b"      → local Qwen3-VL-2B via PyTorch MPS
        "qwen-4b"      → local Qwen3-VL-4B via PyTorch MPS
        "fireworks-8b"  → Fireworks AI Qwen3-VL-8B (reads FIREWORKS_API_KEY env)
        "fireworks-32b" → Fireworks AI Qwen3-VL-32B
        "fireworks-72b" → Fireworks AI Qwen2.5-VL-72B
        "together-8b"   → Together AI Qwen3-VL-8B (reads TOGETHER_API_KEY env)
        "qwen"          → local, custom model string
    """
    # Local backends
    if kind == "qwen-2b":
        return QwenVLBackend(model or "Qwen/Qwen3-VL-2B-Instruct")
    elif kind == "qwen-4b":
        return QwenVLBackend(model or "Qwen/Qwen3-VL-4B-Instruct")
    elif kind == "qwen":
        return QwenVLBackend(model or "Qwen/Qwen3-VL-2B-Instruct")

    # API backends
    for prefix, cfg in [("fireworks", FIREWORKS), ("together", TOGETHER)]:
        for size, model_id in cfg["models"].items():
            if kind == f"{prefix}-{size}":
                key = api_key or os.environ.get(cfg["env_key"])
                if not key:
                    raise ValueError(
                        f"Set {cfg['env_key']} env var or pass --api-key for {kind}"
                    )
                return OpenAICompatibleBackend(
                    base_url=cfg["base_url"],
                    api_key=key,
                    model=model or model_id,
                )

    # MLX backends (Apple Silicon, fast local inference)
    if kind == "qwen-vl-2b":
        return MLXQwenVLBackend(model or "Qwen/Qwen3-VL-2B-Instruct")
    elif kind == "qwen-vl-4b":
        return MLXQwenVLBackend(model or "Qwen/Qwen3-VL-4B-Instruct")
    elif kind == "qwen-vl-8b":
        return MLXQwenVLBackend(model or "Qwen/Qwen3-VL-8B-Instruct")

    available = ["qwen-vl-2b", "qwen-vl-4b", "qwen-vl-8b",
                 "qwen-2b", "qwen-4b", "qwen",
                 "fireworks-8b", "fireworks-32b", "fireworks-72b",
                 "together-8b"]
    raise ValueError(f"Unknown backend {kind!r}. Choose from: {available}")
