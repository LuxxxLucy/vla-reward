"""
Run TOPreward methods on a video.

Usage:
    uv run python scripts/run.py --method baseline --backend qwen-vl-2b
    uv run python scripts/run.py --method contrastive --backend qwen-vl-2b
    uv run python scripts/run.py --method ensemble --backend qwen-vl-2b --n-prompts 3
    uv run python scripts/run.py --method contrast_ensemble --backend qwen-vl-2b --n-prompts 3
"""

import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.backend import make_backend
from src.video import extract_frames
from src.topreward import score_episode, build_prefix_list


METHODS = ["baseline", "contrastive", "ensemble", "contrast_ensemble"]


def p(msg):
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--backend", default="qwen-vl-2b")
    parser.add_argument("--video", default="data/videos/put_pen_cup.mp4")
    parser.add_argument("--instruction", default="create a tower of 5 cubes")
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--n-prompts", type=int, default=3)
    parser.add_argument("--prompt-indices", default=None,
                        help="Comma-separated prompt indices, e.g. '2,5,9'")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    prompt_indices = None
    if args.prompt_indices:
        prompt_indices = [int(x) for x in args.prompt_indices.split(",")]
        args.n_prompts = len(prompt_indices)
    elif args.method in ("ensemble", "contrast_ensemble"):
        prompt_indices = list(range(args.n_prompts))

    p(f"method={args.method}  backend={args.backend}  K={args.num_frames}")
    p(f"video={args.video}")
    p(f"instruction={args.instruction!r}")
    if args.method in ("ensemble", "contrast_ensemble"):
        p(f"n_prompts={args.n_prompts}")
    p("")

    frames = extract_frames(args.video, args.num_frames)
    backend = make_backend(args.backend)
    prefix_list = build_prefix_list(frames)
    p("")

    t_total = time.time()
    result = score_episode(prefix_list, args.instruction, backend, args.method, prompt_indices)
    total_time = time.time() - t_total

    raw = result["raw"]
    voc = result["voc"]

    for k, r in enumerate(raw, 1):
        p(f"  prefix {k:2d}/{len(frames)}: {r:7.4f}")

    p(f"\nVOC: {voc:.4f}")
    p(f"Time: {total_time:.1f}s  ({total_time/len(frames):.1f}s/prefix)")

    output = {
        "method": args.method, "backend": args.backend,
        "video": args.video, "instruction": args.instruction,
        "num_frames": args.num_frames, "n_prompts": args.n_prompts,
        "raw": raw, "voc": voc, "time": total_time,
    }

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        p(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
