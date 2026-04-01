"""
Timing breakdown: vision encoding vs language inference for Qwen3-VL-8B.

Measures where inference time goes for the 8B model on fold_towel,
so we can fill in the missing vision/language split for 8B.

Usage:
    uv run python scripts/run_timing_breakdown_8b.py
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
    p("=== Timing Breakdown: 8B on fold_towel ===\n")

    frames_bgr = extract_frames(VIDEO, NUM_FRAMES)
    frames_pil = frames_to_pil(frames_bgr)

    prompt = build_prompt(VARIANTS[0], INSTRUCTION)

    # Load 8B model
    from mlx_vlm import load
    p("Loading Qwen3-VL-8B-Instruct (MLX)...")
    model, processor = load("Qwen/Qwen3-VL-8B-Instruct")

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    true_id = tokenizer.encode(" True", add_special_tokens=False)[0]
    false_id = tokenizer.encode(" False", add_special_tokens=False)[0]
    p(f"  Loaded. ' True'={true_id}  ' False'={false_id}")

    cache = VisionCache(model, processor, true_id, false_id)

    # --- Single prompt with vision/language split ---
    p("\n--- Single prompt timing (1 prompt, K=10) ---")
    vision_times = []
    language_times = []
    for k in range(1, NUM_FRAMES + 1):
        t0 = time.time()
        cache.encode_frames(frames_pil[:k])
        t_v = time.time() - t0

        t1 = time.time()
        cache.log_prob_true(prompt)
        t_l = time.time() - t1

        vision_times.append(t_v)
        language_times.append(t_l)
        p(f"  prefix {k:2d}: vision={t_v:.2f}s  language={t_l:.2f}s  total={t_v+t_l:.2f}s")

    total_vision = sum(vision_times)
    total_language = sum(language_times)
    total = total_vision + total_language
    p(f"\n  Total: {total:.1f}s  (vision={total_vision:.1f}s  language={total_language:.1f}s)")
    p(f"  Vision fraction: {total_vision/total*100:.1f}%")

    result = {
        "video": "fold_towel",
        "model": "Qwen3-VL-8B",
        "num_frames": NUM_FRAMES,
        "single_prompt_breakdown": {
            "per_prefix_vision": vision_times,
            "per_prefix_language": language_times,
            "total_vision": total_vision,
            "total_language": total_language,
            "total": total,
            "vision_fraction_pct": total_vision / total * 100,
        },
    }

    out = "results/timing_breakdown_8b_fold_towel.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    p(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
