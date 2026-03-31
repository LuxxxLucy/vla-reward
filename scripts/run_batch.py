"""
Run a method across all videos in the manifest.

Usage:
    uv run python scripts/run_batch.py --method baseline --backend qwen-vl-2b
    uv run python scripts/run_batch.py --method baseline --backend qwen-vl-8b
    uv run python scripts/run_batch.py --method ensemble --backend qwen-vl-2b --n-prompts 3
"""

import sys, os, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--backend", default="qwen-vl-2b")
    parser.add_argument("--n-prompts", type=int, default=3)
    parser.add_argument("--manifest", default="data/videos/manifest.json")
    parser.add_argument("--prompt-indices", default=None,
                        help="Comma-separated prompt indices, e.g. '2,5,9'")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Video IDs to skip (default: none)")
    args = parser.parse_args()

    with open(args.manifest) as f:
        videos = json.load(f)

    # Filter
    videos = [v for v in videos if v["id"] not in args.skip]
    print(f"Running {args.method} with {args.backend} on {len(videos)} videos", flush=True)

    for v in videos:
        vid_id = v["id"]
        # Resolve video path relative to project root
        video_path = v["file"]
        if not os.path.isabs(video_path) and not os.path.exists(video_path):
            video_path = os.path.join("data/videos", video_path)

        method_tag = args.method
        if args.method in ("ensemble", "contrast_ensemble"):
            if args.prompt_indices:
                method_tag = f"{args.method}-best{len(args.prompt_indices.split(','))}"
            else:
                method_tag = f"{args.method}-{args.n_prompts}"

        output = f"results/{method_tag}_{args.backend}_{vid_id}.json"

        if os.path.exists(output):
            print(f"\n[SKIP] {vid_id} — already exists: {output}", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"[{vid_id}] {v['instruction']!r}", flush=True)
        print(f"  video={video_path}  K={v['num_frames']}", flush=True)
        print(f"{'='*60}", flush=True)

        cmd = [
            sys.executable, "scripts/run.py",
            "--method", args.method,
            "--backend", args.backend,
            "--video", video_path,
            "--instruction", v["instruction"],
            "--num-frames", str(v["num_frames"]),
            "--n-prompts", str(args.n_prompts),
            "--output", output,
        ]
        if args.prompt_indices:
            cmd.extend(["--prompt-indices", args.prompt_indices])
        subprocess.run(cmd, check=False)

    # Print summary
    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY: {args.method} / {args.backend}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Video':<30} {'VOC':>8} {'Time':>8}", flush=True)
    print("-" * 48, flush=True)

    vocs = []
    for v in videos:
        vid_id = v["id"]
        method_tag = args.method
        if args.method in ("ensemble", "contrast_ensemble"):
            if args.prompt_indices:
                method_tag = f"{args.method}-best{len(args.prompt_indices.split(','))}"
            else:
                method_tag = f"{args.method}-{args.n_prompts}"
        output = f"results/{method_tag}_{args.backend}_{vid_id}.json"
        if os.path.exists(output):
            with open(output) as f:
                r = json.load(f)
            vocs.append(r["voc"])
            print(f"  {vid_id:<28} {r['voc']:>8.4f} {r['time']:>7.0f}s", flush=True)
        else:
            print(f"  {vid_id:<28} {'MISSING':>8}", flush=True)

    if vocs:
        import numpy as np
        print(f"\n  {'Mean VOC':<28} {np.mean(vocs):>8.4f}", flush=True)
        print(f"  {'Std':<28} {np.std(vocs):>8.4f}", flush=True)


if __name__ == "__main__":
    main()
