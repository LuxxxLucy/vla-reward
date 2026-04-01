"""
Timing breakdown for N=10 prompt ensemble with vision caching (2B, fold_towel).
Shows that 10 prompts costs ~4× baseline instead of 10× thanks to caching.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
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
    p("=== N=10 Ensemble Timing: 2B on fold_towel ===\n")

    frames_bgr = extract_frames(VIDEO, NUM_FRAMES)
    frames_pil = frames_to_pil(frames_bgr)

    prompts_10 = [build_prompt(VARIANTS[i], INSTRUCTION) for i in range(10)]

    from mlx_vlm import load
    p("Loading Qwen3-VL-2B-Instruct (MLX)...")
    model, processor = load("Qwen/Qwen3-VL-2B-Instruct")
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    true_id = tokenizer.encode(" True", add_special_tokens=False)[0]
    false_id = tokenizer.encode(" False", add_special_tokens=False)[0]
    p(f"  Loaded. ' True'={true_id}")

    cache = VisionCache(model, processor, true_id, false_id)

    # --- Cached N=10 ensemble ---
    p("\n--- Cached N=10 ensemble (10 prompts, K=10) ---")
    vision_times = []
    language_times = []
    for k in range(1, NUM_FRAMES + 1):
        t0 = time.time()
        cache.encode_frames(frames_pil[:k])
        t_v = time.time() - t0

        t1 = time.time()
        for pi in range(10):
            cache.log_prob_true(prompts_10[pi])
        t_l = time.time() - t1

        vision_times.append(t_v)
        language_times.append(t_l)
        p(f"  prefix {k:2d}: vision={t_v:.2f}s  10×language={t_l:.2f}s  total={t_v+t_l:.2f}s")

    total_vision = sum(vision_times)
    total_language = sum(language_times)
    total = total_vision + total_language
    p(f"\n  Total: {total:.1f}s  (vision={total_vision:.1f}s  language={total_language:.1f}s)")
    p(f"  Vision fraction: {total_vision/total*100:.1f}%")

    result = {
        "video": "fold_towel",
        "model": "Qwen3-VL-2B",
        "num_frames": NUM_FRAMES,
        "n_prompts": 10,
        "cached_ensemble10": {
            "per_prefix_vision": vision_times,
            "per_prefix_language": language_times,
            "total_vision": total_vision,
            "total_language": total_language,
            "total": total,
            "vision_fraction_pct": total_vision / total * 100,
        },
    }

    out = "results/timing_n10_fold_towel.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    p(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
