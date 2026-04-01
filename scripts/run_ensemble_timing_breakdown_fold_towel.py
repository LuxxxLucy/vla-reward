"""
Timing breakdown: vision encoding vs language inference.

Measures where inference time goes to show that vision caching
makes the N=3 ensemble essentially free.

Phase A: Uncached baseline (1 prompt, K prefixes) — full forward pass each time
Phase B: Cached single prompt — vision encode + language-only, measured separately
Phase C: Cached N=3 ensemble — same vision encode + 3× language-only

Usage:
    uv run python scripts/run_ensemble_timing_breakdown.py
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
from PIL import Image
from src.video import extract_frames
from src.vision_cache import VisionCache
from src.prompts import VARIANTS, build_prompt

VIDEO = "data/videos/fold_towel.mp4"
INSTRUCTION = "Fold the towel."
NUM_FRAMES = 10


def frames_to_pil(frames):
    return [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]


def p(msg):
    print(msg, flush=True)


def main():
    p("=== Timing Breakdown: fold_towel ===\n")

    frames_bgr = extract_frames(VIDEO, NUM_FRAMES)
    frames_pil = frames_to_pil(frames_bgr)

    prompt = build_prompt(VARIANTS[0], INSTRUCTION)
    prompts_3 = [build_prompt(VARIANTS[i], INSTRUCTION) for i in range(3)]

    # Load model once
    from mlx_vlm import load
    p("Loading Qwen3-VL-2B-Instruct (MLX)...")
    model, processor = load("Qwen/Qwen3-VL-2B-Instruct")

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    true_id = tokenizer.encode(" True", add_special_tokens=False)[0]
    false_id = tokenizer.encode(" False", add_special_tokens=False)[0]

    # We need backend for Phase A (uncached)
    from src.backend import MLXQwenVLBackend
    # Build a backend that shares the already-loaded model
    backend = MLXQwenVLBackend.__new__(MLXQwenVLBackend)
    backend.model_name = "Qwen/Qwen3-VL-2B-Instruct"
    backend._model = model
    backend._processor = processor
    backend._true_token_id = true_id
    backend._false_token_id = false_id
    from mlx_vlm import prepare_inputs
    backend._prepare_inputs = prepare_inputs
    import mlx.core as mx
    backend._mx = mx

    cache = VisionCache(model, processor, true_id, false_id)

    # --- Phase A: Uncached baseline ---
    p("\n--- Phase A: Uncached baseline (1 prompt, K=10) ---")
    phase_a_times = []
    for k in range(1, NUM_FRAMES + 1):
        t0 = time.time()
        backend.log_prob_true(frames_bgr[:k], prompt)
        dt = time.time() - t0
        phase_a_times.append(dt)
        p(f"  prefix {k:2d}: {dt:.2f}s")
    phase_a_total = sum(phase_a_times)
    p(f"  Total: {phase_a_total:.1f}s")

    # --- Phase B: Cached single prompt ---
    p("\n--- Phase B: Cached single prompt (1 prompt, K=10) ---")
    phase_b_vision = []
    phase_b_language = []
    for k in range(1, NUM_FRAMES + 1):
        t0 = time.time()
        cache.encode_frames(frames_pil[:k])
        t_v = time.time() - t0

        t1 = time.time()
        cache.log_prob_true(prompt)
        t_l = time.time() - t1

        phase_b_vision.append(t_v)
        phase_b_language.append(t_l)
        p(f"  prefix {k:2d}: vision={t_v:.2f}s  language={t_l:.2f}s  total={t_v+t_l:.2f}s")
    phase_b_total = sum(phase_b_vision) + sum(phase_b_language)
    p(f"  Total: {phase_b_total:.1f}s  (vision={sum(phase_b_vision):.1f}s  language={sum(phase_b_language):.1f}s)")

    # --- Phase C: Cached N=3 ensemble ---
    p("\n--- Phase C: Cached N=3 ensemble (3 prompts, K=10) ---")
    phase_c_vision = []
    phase_c_language = []
    for k in range(1, NUM_FRAMES + 1):
        t0 = time.time()
        cache.encode_frames(frames_pil[:k])
        t_v = time.time() - t0

        t1 = time.time()
        for pi in range(3):
            cache.log_prob_true(prompts_3[pi])
        t_l = time.time() - t1

        phase_c_vision.append(t_v)
        phase_c_language.append(t_l)
        p(f"  prefix {k:2d}: vision={t_v:.2f}s  3×language={t_l:.2f}s  total={t_v+t_l:.2f}s")
    phase_c_total = sum(phase_c_vision) + sum(phase_c_language)
    p(f"  Total: {phase_c_total:.1f}s  (vision={sum(phase_c_vision):.1f}s  language={sum(phase_c_language):.1f}s)")

    # --- Summary ---
    vision_frac = sum(phase_b_vision) / phase_b_total * 100
    ensemble_overhead = phase_c_total / phase_a_total
    naive_3x = 3 * phase_a_total
    caching_speedup = naive_3x / phase_c_total

    p(f"\n=== Summary ===")
    p(f"  Vision fraction (Phase B):     {vision_frac:.1f}%")
    p(f"  Phase A (uncached, 1 prompt):  {phase_a_total:.1f}s")
    p(f"  Phase C (cached, 3 prompts):   {phase_c_total:.1f}s")
    p(f"  Naive 3× cost:                 {naive_3x:.1f}s")
    p(f"  Ensemble overhead vs baseline:  {ensemble_overhead:.2f}×")
    p(f"  Caching speedup vs naive:       {caching_speedup:.2f}×")
    p(f"")
    p(f"  → N=3 ensemble costs {ensemble_overhead:.2f}× a single run, not 3×")
    p(f"  → Vision caching saves {caching_speedup:.1f}× vs running 3 full passes")

    result = {
        "video": "fold_towel",
        "model": "Qwen3-VL-2B",
        "num_frames": NUM_FRAMES,
        "phase_a_uncached": {
            "per_prefix_total": phase_a_times,
            "total": phase_a_total,
        },
        "phase_b_cached_single": {
            "per_prefix_vision": phase_b_vision,
            "per_prefix_language": phase_b_language,
            "total": phase_b_total,
        },
        "phase_c_cached_ensemble3": {
            "per_prefix_vision": phase_c_vision,
            "per_prefix_language": phase_c_language,
            "total": phase_c_total,
        },
        "summary": {
            "vision_fraction_pct": vision_frac,
            "ensemble_overhead_vs_baseline": ensemble_overhead,
            "naive_3x_cost": naive_3x,
            "caching_speedup": caching_speedup,
        },
    }

    out = "results/timing_breakdown_fold_towel.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    p(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
