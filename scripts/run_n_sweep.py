"""
N=1..10 ensemble sweep using vision caching.

For each prefix k, encodes frames once, then runs all 10 prompts.
Reports VOC for ensemble sizes N=1,2,...,10 (using the first N prompts)
and also for the "best-N" (oracle selection).

Usage:
    uv run python scripts/run_n_sweep.py --video data/videos/put_pen_cup.mp4 --instruction "Put the pen into the cup."
"""

import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np, cv2
from PIL import Image
from mlx_vlm import load
from src.video import extract_frames
from src.vision_cache import VisionCache
from src.prompts import VARIANTS, build_prompt
from src.voc import compute_voc


def frames_to_pil(frames):
    """Convert BGR numpy frames to PIL images."""
    return [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]


def p(msg):
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    p(f"N-sweep: {args.video}")
    p(f"Instruction: {args.instruction!r}")
    p(f"K={args.num_frames}\n")

    frames = frames_to_pil(extract_frames(args.video, args.num_frames))
    model, processor = load("Qwen/Qwen3-VL-2B-Instruct")
    true_ids = processor.tokenizer.encode(" True", add_special_tokens=False)
    false_ids = processor.tokenizer.encode(" False", add_special_tokens=False)
    cache = VisionCache(model, processor, true_ids[0], false_ids[0])

    prompts = [build_prompt(v, args.instruction) for v in VARIANTS]
    n_prompts = len(prompts)

    # per_prompt_raw[prompt_i][prefix_k] = log P(True)
    per_prompt_raw = [[] for _ in range(n_prompts)]

    t_total = time.time()
    for k in range(1, len(frames) + 1):
        t0 = time.time()
        cache.encode_frames(frames[:k])
        t_encode = time.time() - t0

        t1 = time.time()
        for i in range(n_prompts):
            lp = cache.log_prob_true(prompts[i])
            per_prompt_raw[i].append(lp)
        t_prompts = time.time() - t1

        p(f"  prefix {k:2d}/{len(frames)}: encode={t_encode:.1f}s  {n_prompts} prompts={t_prompts:.1f}s  total={time.time()-t0:.1f}s")

    total_time = time.time() - t_total
    p(f"\nTotal time: {total_time:.1f}s")

    # Compute per-prompt VOC
    prompt_vocs = [(i, compute_voc(per_prompt_raw[i])) for i in range(n_prompts)]
    prompt_vocs_sorted = sorted(prompt_vocs, key=lambda x: x[1], reverse=True)

    p(f"\nPer-prompt VOC:")
    for i, voc in prompt_vocs_sorted:
        snippet = VARIANTS[i][:50].replace("{task}", "...")
        p(f"  prompt {i:2d}: VOC={voc:.4f}  {snippet}")

    # Compute ensemble VOC for N=1..10 (first-N and best-N)
    p(f"\nEnsemble VOC vs N:")
    p(f"  {'N':>3} {'first-N':>10} {'best-N':>10}")
    p(f"  {'-'*3} {'-'*10} {'-'*10}")

    sweep_first = []
    sweep_best = []
    for n in range(1, n_prompts + 1):
        # First-N: use prompts 0..N-1
        first_n_raw = [float(np.mean([per_prompt_raw[i][k] for i in range(n)])) for k in range(len(frames))]
        first_n_voc = compute_voc(first_n_raw)
        sweep_first.append(first_n_voc)

        # Best-N: use top-N by individual VOC
        best_indices = [idx for idx, _ in prompt_vocs_sorted[:n]]
        best_n_raw = [float(np.mean([per_prompt_raw[i][k] for i in best_indices])) for k in range(len(frames))]
        best_n_voc = compute_voc(best_n_raw)
        sweep_best.append(best_n_voc)

        p(f"  {n:3d} {first_n_voc:>10.4f} {best_n_voc:>10.4f}")

    result = {
        "video": args.video, "instruction": args.instruction,
        "num_frames": args.num_frames, "total_time": total_time,
        "per_prompt_voc": [{"prompt_index": i, "voc": v} for i, v in prompt_vocs],
        "per_prompt_raw": per_prompt_raw,
        "sweep_first_n": sweep_first,
        "sweep_best_n": sweep_best,
    }

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        p(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
