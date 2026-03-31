"""
Run each of the 10 prompt variants individually on a video.
Shows which prompts help vs hurt.

Usage:
    uv run python scripts/run_prompt_sweep.py --backend qwen-vl-2b --video data/videos/put_pen_cup.mp4 --instruction "Put the pen into the cup."
"""

import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from src.backend import make_backend
from src.video import extract_frames
from src.prompts import VARIANTS, build_prompt
from src.voc import compute_voc


def p(msg):
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="qwen-vl-2b")
    parser.add_argument("--video", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    p(f"Prompt sweep: {args.video}")
    p(f"Instruction: {args.instruction!r}")
    p(f"K={args.num_frames}  Backend={args.backend}\n")

    frames = extract_frames(args.video, args.num_frames)
    backend = make_backend(args.backend)
    p("")

    results = []
    for i, template in enumerate(VARIANTS):
        prompt = build_prompt(template, args.instruction)
        raw = []
        t0 = time.time()
        for k in range(1, len(frames) + 1):
            lp = backend.log_prob_true(frames[:k], prompt)
            raw.append(lp)
        dt = time.time() - t0
        voc = compute_voc(raw)
        # Show first 30 chars of the prompt variant (after the common prefix)
        snippet = template[:60].replace("{task}", "...")
        p(f"  prompt {i:2d}: VOC={voc:7.4f}  ({dt:.0f}s)  {snippet}")
        results.append({"prompt_index": i, "voc": voc, "raw": raw, "time": dt})

    # Sort by VOC
    ranked = sorted(results, key=lambda r: r["voc"], reverse=True)
    p(f"\nRanking:")
    for rank, r in enumerate(ranked):
        marker = " ★" if rank < 3 else " ✗" if r["voc"] < results[0]["voc"] else ""
        p(f"  #{rank+1} prompt {r['prompt_index']:2d}: VOC={r['voc']:.4f}{marker}")

    # Best-3 ensemble (average of top 3 prompts' raw values)
    best_indices = [r["prompt_index"] for r in ranked[:3]]
    best_raws = [results[i]["raw"] for i in best_indices]
    ensemble_raw = [float(np.mean([r[k] for r in best_raws])) for k in range(len(frames))]
    ensemble_voc = compute_voc(ensemble_raw)
    p(f"\nBest-3 ensemble (prompts {best_indices}): VOC={ensemble_voc:.4f}")

    # All-10 ensemble
    all_raws = [r["raw"] for r in results]
    all_ensemble_raw = [float(np.mean([r[k] for r in all_raws])) for k in range(len(frames))]
    all_ensemble_voc = compute_voc(all_ensemble_raw)
    p(f"All-10 ensemble: VOC={all_ensemble_voc:.4f}")

    # Random-3 ensemble (first 3, which is what our earlier runs used)
    first3_raws = [results[i]["raw"] for i in range(3)]
    first3_raw = [float(np.mean([r[k] for r in first3_raws])) for k in range(len(frames))]
    first3_voc = compute_voc(first3_raw)
    p(f"First-3 ensemble (prompts 0,1,2): VOC={first3_voc:.4f}")

    output_data = {
        "video": args.video, "instruction": args.instruction,
        "backend": args.backend, "num_frames": args.num_frames,
        "per_prompt": results,
        "best_3_indices": best_indices, "best_3_voc": ensemble_voc,
        "all_10_voc": all_ensemble_voc, "first_3_voc": first3_voc,
    }
    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        p(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
