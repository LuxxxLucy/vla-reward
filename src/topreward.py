"""
TOPreward scoring functions.

Four modes:
  baseline           — R = log P("True")                         (paper original)
  contrastive        — R = log P("True") - log P("False")        (zero extra cost)
  ensemble           — average R across N prompt variants         (N× cost)
  contrast_ensemble  — average contrastive across N prompts       (N× cost)

All return a dict with keys:
    raw:         list[float]  — raw reward values per prefix
    normalized:  list[float]  — min-max normalized to [0, 1]
    voc:         float        — VOC for this episode
"""

import numpy as np
from .voc import compute_voc
from .prompts import VARIANTS, build_prompt


def _normalize(values: list[float]) -> list[float]:
    arr = np.array(values, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return [0.5] * len(values)
    return ((arr - lo) / (hi - lo)).tolist()


def score_episode(
    frames_list: list[list[np.ndarray]],
    instruction: str,
    backend,
    mode: str = "baseline",
    prompt_indices: list[int] | None = None,
) -> dict:
    """Score one episode under a given mode.

    Args:
        frames_list: List of frame sets, one per prefix timepoint.
                     frames_list[k] = frames 0..k (growing prefix).
        instruction: Task instruction string.
        backend:     A VLMBackend instance.
        mode:        "baseline" | "contrastive" | "ensemble" | "contrast_ensemble"
        prompt_indices: Which VARIANTS indices to use for ensemble.
                        Default: all 10.

    Returns:
        {"raw": [...], "normalized": [...], "voc": float}
    """
    if mode == "baseline":
        return _score_baseline(frames_list, instruction, backend)
    elif mode == "contrastive":
        return _score_contrastive(frames_list, instruction, backend)
    elif mode == "ensemble":
        return _score_ensemble(frames_list, instruction, backend, prompt_indices)
    elif mode == "contrast_ensemble":
        return _score_contrast_ensemble(frames_list, instruction, backend, prompt_indices)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


def _score_baseline(frames_list, instruction, backend) -> dict:
    prompt = build_prompt(VARIANTS[0], instruction)
    raw = [backend.log_prob_true(frames, prompt) for frames in frames_list]
    normalized = _normalize(raw)
    return {"raw": raw, "normalized": normalized, "voc": compute_voc(normalized)}


def _score_contrastive(frames_list, instruction, backend) -> dict:
    prompt = build_prompt(VARIANTS[0], instruction)
    raw = []
    for frames in frames_list:
        lp_true, lp_false = backend.log_prob_both(frames, prompt)
        raw.append(lp_true - lp_false)
    normalized = _normalize(raw)
    return {"raw": raw, "normalized": normalized, "voc": compute_voc(normalized)}


def _score_ensemble(frames_list, instruction, backend, prompt_indices=None) -> dict:
    if prompt_indices is None:
        prompt_indices = list(range(len(VARIANTS)))
    prompts = [build_prompt(VARIANTS[i], instruction) for i in prompt_indices]

    raw = []
    for frames in frames_list:
        # Average log-probs across prompts (geometric mean of probabilities)
        log_probs = [backend.log_prob_true(frames, p) for p in prompts]
        raw.append(float(np.mean(log_probs)))

    normalized = _normalize(raw)
    return {"raw": raw, "normalized": normalized, "voc": compute_voc(normalized)}


def _score_contrast_ensemble(frames_list, instruction, backend, prompt_indices=None) -> dict:
    if prompt_indices is None:
        prompt_indices = list(range(len(VARIANTS)))
    prompts = [build_prompt(VARIANTS[i], instruction) for i in prompt_indices]

    raw = []
    for frames in frames_list:
        contrasts = []
        for p in prompts:
            lp_t, lp_f = backend.log_prob_both(frames, p)
            contrasts.append(lp_t - lp_f)
        raw.append(float(np.mean(contrasts)))

    normalized = _normalize(raw)
    return {"raw": raw, "normalized": normalized, "voc": compute_voc(normalized)}


def build_prefix_list(all_frames: list[np.ndarray], num_prefixes: int = 10) -> list[list[np.ndarray]]:
    """Build a list of growing frame prefixes from a full episode.

    prefix[k] contains frames 0..k (inclusive) sampled uniformly.

    all_frames: The full set of frames for this episode (already sampled at num_prefixes points).
    Returns: List of num_prefixes lists, where list[k] = all_frames[:k+1].
    """
    return [all_frames[:k + 1] for k in range(len(all_frames))]
