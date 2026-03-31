"""
Generate publication-quality plots from TOPreward results.

Usage:
    uv run python scripts/plot_results.py
"""

import os
import json
import glob
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'Helvetica Neue', 'Arial', 'sans-serif'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linewidth': 0.6,
    'figure.dpi': 150,
})

COLORS = {
    '2b_baseline':    '#3b82f6',   # blue
    '2b_contrastive': '#f97316',   # orange
    '2b_ens3':        '#22c55e',   # green
    '2b_best3':       '#a855f7',   # purple
    '8b_baseline':    '#ef4444',   # red
}

VIDEOS = [
    'fold_towel',
    'remove_cap',
    'put_block_cup',
    'put_marker_cup',
    'stack_cubes_lerobot',
    'put_pen_cup',
    'stackcubes_rewardscope',
]

SHORT_NAMES = {
    'fold_towel':           'Fold\ntowel',
    'remove_cap':           'Remove\ncap',
    'put_block_cup':        'Block\n->cup',
    'put_marker_cup':       'Marker\n->cup',
    'stack_cubes_lerobot':  'Stack cubes\n(LeRobot)',
    'put_pen_cup':          'Pen\n->cup',
    'stackcubes_rewardscope': 'Stack cubes\n(Scope)',
}

PROMPT_LABELS = {
    0: '#0 — default ask',
    1: '#1 — step judge',
    2: '#2 — progress 1-5',
    3: '#3 — boolean q',
    4: '#4 — neg framing',
    5: '#5 — ordinal score',
    6: '#6 — chain-of-thought',
    7: '#7 — completion %',
    8: '#8 — binary done?',
    9: '#9 — rank best',
}

METHOD_KEYS = {
    '2b_baseline':    'baseline_qwen-vl-2b',
    '2b_contrastive': 'contrastive_qwen-vl-2b',
    '2b_ens3':        'ensemble-3_qwen-vl-2b',
    '2b_best3':       'ensemble-best3_qwen-vl-2b',
    '8b_baseline':    'baseline_qwen-vl-8b',
}

METHOD_LABELS = {
    '2b_baseline':    '2B baseline',
    '2b_contrastive': '2B contrastive',
    '2b_ens3':        '2B ensemble first-3',
    '2b_best3':       '2B ensemble best-3',
    '8b_baseline':    '8B baseline',
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir='results'):
    """Return dict: vid_id -> method_key -> result_dict."""
    data = {}
    for f in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
        fname = os.path.basename(f).replace('.json', '')
        # Skip sweep files
        if fname.startswith('n_sweep') or fname.startswith('prompt_sweep'):
            continue
        with open(f) as fh:
            r = json.load(fh)
        # Parse method key and video id
        for backend in ['qwen-vl-8b', 'qwen-vl-2b']:
            tag = f'_{backend}_'
            if tag in fname:
                method = fname.split(tag)[0]
                vid_id = fname.split(tag)[1]
                key = f'{method}_{backend}'
                data.setdefault(vid_id, {})[key] = r
                break
    return data


def load_n_sweep(vid_id, results_dir='results'):
    path = os.path.join(results_dir, f'n_sweep_{vid_id}.json')
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def load_prompt_sweep(vid_id, results_dir='results'):
    path = os.path.join(results_dir, f'prompt_sweep_qwen-vl-2b_{vid_id}.json')
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def get_voc(data, vid_id, slot):
    key = METHOD_KEYS[slot]
    r = data.get(vid_id, {}).get(key)
    return r['voc'] if r else None


def normalize(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-9)


# ---------------------------------------------------------------------------
# Figure 1 — Grouped bar chart: all methods × all videos
# ---------------------------------------------------------------------------

def fig1_voc_comparison(data, out='figures/voc_comparison.png'):
    slots = ['2b_baseline', '2b_contrastive', '2b_ens3', '2b_best3', '8b_baseline']
    n_methods = len(slots)
    n_vids = len(VIDEOS)

    # Collect values — None → 0 for missing
    vals = {}
    for s in slots:
        vals[s] = [get_voc(data, v, s) or 0.0 for v in VIDEOS]

    x = np.arange(n_vids)
    total_w = 0.75
    w = total_w / n_methods
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * w

    fig, ax = plt.subplots(figsize=(14, 5.5))

    for i, slot in enumerate(slots):
        bars = ax.bar(x + offsets[i], vals[slot], w * 0.88,
                      label=METHOD_LABELS[slot],
                      color=COLORS[slot], edgecolor='white', linewidth=0.4,
                      zorder=3)
        for bar, v in zip(bars, vals[slot]):
            if abs(v) > 0.01:
                ypos = bar.get_height() + 0.015 if v >= 0 else bar.get_height() - 0.06
                ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                        f'{v:.2f}', ha='center', va='bottom',
                        fontsize=5.5, color='#333333')

    # Mean reference lines
    for slot in slots:
        m = np.mean(vals[slot])
        ax.axhline(m, color=COLORS[slot], linestyle='--', linewidth=0.9,
                   alpha=0.45, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES[v] for v in VIDEOS], fontsize=8.5)
    ax.set_ylabel('VOC  (higher is better)', fontsize=10)
    ax.set_title('TOPreward — VOC by method and video', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.85,
              ncol=n_methods, bbox_to_anchor=(0, -0.22))
    ax.set_ylim(-0.6, 1.18)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.axhline(0, color='#999', linewidth=0.5, zorder=1)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


# ---------------------------------------------------------------------------
# Figure 2 — Reward curves: 3 videos × 4 methods
# ---------------------------------------------------------------------------

def fig2_reward_curves(data, out='figures/reward_curves.png'):
    vids = ['put_pen_cup', 'fold_towel', 'stackcubes_rewardscope']
    vid_titles = {
        'put_pen_cup':            'Pen -> cup',
        'fold_towel':             'Fold towel',
        'stackcubes_rewardscope': 'Stack cubes (Scope)',
    }
    slots = ['2b_baseline', '2b_contrastive', '2b_ens3', '8b_baseline']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)

    for ax, vid in zip(axes, vids):
        for slot in slots:
            key = METHOD_KEYS[slot]
            r = data.get(vid, {}).get(key)
            if r is None:
                continue
            raw = np.array(r['raw'])
            norm = normalize(raw)
            voc = r['voc']
            ax.plot(range(1, len(norm) + 1), norm,
                    marker='o', markersize=3.5, linewidth=1.6,
                    color=COLORS[slot],
                    label=f'{METHOD_LABELS[slot]}  (VOC={voc:.2f})',
                    zorder=3)

        ax.set_xlabel('Prefix k', fontsize=9)
        ax.set_ylabel('Normalised reward', fontsize=9)
        ax.set_title(vid_titles[vid], fontsize=10, fontweight='semibold')
        ax.set_ylim(-0.08, 1.12)
        ax.set_xticks(range(1, 11))
        ax.legend(fontsize=6.5, loc='lower right', framealpha=0.85)

    plt.suptitle('Reward curves — normalised per video', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


# ---------------------------------------------------------------------------
# Figure 3 — N-sweep curves
# ---------------------------------------------------------------------------

def fig3_n_sweep(data, out='figures/n_sweep.png'):
    vids = ['put_pen_cup', 'fold_towel', 'stackcubes_rewardscope']
    vid_titles = {
        'put_pen_cup':            'Pen -> cup',
        'fold_towel':             'Fold towel',
        'stackcubes_rewardscope': 'Stack cubes (Scope)',
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
    N = np.arange(1, 11)

    for ax, vid in zip(axes, vids):
        sweep = load_n_sweep(vid)
        if sweep is None:
            ax.set_title(f'{vid_titles[vid]}\n(no data)')
            continue

        first_n = np.array(sweep['sweep_first_n'])
        best_n = np.array(sweep['sweep_best_n'])

        ax.plot(N, first_n, color='#3b82f6', linewidth=2, marker='o', markersize=4,
                label='First-N VOC', zorder=3)
        ax.plot(N, best_n, color='#22c55e', linewidth=2, marker='s', markersize=4,
                linestyle='--', label='Best-N VOC', zorder=3)

        # Reference lines: 2B baseline and 8B baseline
        ref_2b = get_voc(data, vid, '2b_baseline')
        ref_8b = get_voc(data, vid, '8b_baseline')
        if ref_2b is not None:
            ax.axhline(ref_2b, color=COLORS['2b_baseline'], linewidth=1,
                       linestyle=':', alpha=0.8, label=f'2B baseline ({ref_2b:.2f})')
        if ref_8b is not None:
            ax.axhline(ref_8b, color=COLORS['8b_baseline'], linewidth=1,
                       linestyle=':', alpha=0.8, label=f'8B baseline ({ref_8b:.2f})')

        ax.set_xlabel('N (number of prompts)', fontsize=9)
        ax.set_ylabel('VOC', fontsize=9)
        ax.set_title(vid_titles[vid], fontsize=10, fontweight='semibold')
        ax.set_xticks(N)
        ax.legend(fontsize=7, framealpha=0.85)

    plt.suptitle('N-prompt sweep — First-N vs Best-N ensemble VOC', fontsize=12,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


# ---------------------------------------------------------------------------
# Figure 4 — Prompt × Video heatmap
# ---------------------------------------------------------------------------

def fig4_prompt_heatmap(out='figures/prompt_heatmap.png'):
    vids = ['put_pen_cup', 'fold_towel', 'stackcubes_rewardscope']
    vid_labels = ['Pen -> cup', 'Fold towel', 'Stack cubes (Scope)']
    n_prompts = 10

    # Build matrix: rows=prompts, cols=videos
    mat = np.full((n_prompts, len(vids)), np.nan)
    for j, vid in enumerate(vids):
        sweep = load_prompt_sweep(vid)
        if sweep is None:
            continue
        for entry in sweep['per_prompt']:
            i = entry['prompt_index']
            mat[i, j] = entry['voc']

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(mat, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')

    # Annotate cells
    for i in range(n_prompts):
        for j in range(len(vids)):
            v = mat[i, j]
            if not np.isnan(v):
                color = 'black' if abs(v) < 0.6 else 'white'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=9, color=color, fontweight='semibold')

    ax.set_xticks(range(len(vids)))
    ax.set_xticklabels(vid_labels, fontsize=10)
    ax.set_yticks(range(n_prompts))
    ax.set_yticklabels([PROMPT_LABELS[i] for i in range(n_prompts)], fontsize=9)
    ax.set_title('Per-prompt VOC — Prompt × Video', fontsize=12, fontweight='bold', pad=12)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('VOC', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Turn off the grid for this one (it clashes with the heatmap cells)
    ax.grid(False)
    ax.tick_params(axis='x', bottom=False)
    ax.tick_params(axis='y', left=False)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


# ---------------------------------------------------------------------------
# Figure 5 — Method summary bar chart (mean VOC across videos)
# ---------------------------------------------------------------------------

def fig5_method_summary(data, out='figures/method_summary.png'):
    slots = ['2b_baseline', '2b_contrastive', '2b_ens3', '2b_best3', '8b_baseline']

    means, stds = [], []
    for slot in slots:
        vocs = [get_voc(data, v, slot) for v in VIDEOS]
        vocs = [x for x in vocs if x is not None]
        means.append(np.mean(vocs) if vocs else 0.0)
        stds.append(np.std(vocs) if vocs else 0.0)

    # Sort best → worst
    order = np.argsort(means)[::-1]
    sorted_slots  = [slots[i] for i in order]
    sorted_means  = [means[i]  for i in order]
    sorted_stds   = [stds[i]   for i in order]
    sorted_labels = [METHOD_LABELS[s] for s in sorted_slots]
    sorted_colors = [COLORS[s] for s in sorted_slots]

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(sorted_slots))

    bars = ax.barh(y, sorted_means, xerr=sorted_stds,
                   color=sorted_colors, edgecolor='white', linewidth=0.4,
                   error_kw=dict(elinewidth=1.2, capsize=4, capthick=1.2, ecolor='#555'),
                   height=0.55, zorder=3)

    # Value labels at end of bars
    for bar, m, s in zip(bars, sorted_means, sorted_stds):
        xpos = max(m, 0) + s + 0.01
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f'{m:.3f} ± {s:.3f}', va='center', fontsize=8.5, color='#333')

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_labels, fontsize=10)
    ax.set_xlabel('Mean VOC across 7 videos  (higher is better)', fontsize=10)
    ax.set_title('Method summary — mean VOC (excl. ride_bike)', fontsize=12,
                 fontweight='bold', pad=10)
    ax.axvline(0, color='#999', linewidth=0.6)
    ax.set_xlim(left=min(0, min(sorted_means) - max(sorted_stds) - 0.15))
    ax.invert_yaxis()   # best at top

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs('figures', exist_ok=True)
    data = load_results()

    print(f'Loaded data for {len(data)} videos: {sorted(data.keys())}')

    fig1_voc_comparison(data)
    fig2_reward_curves(data)
    fig3_n_sweep(data)
    fig4_prompt_heatmap()
    fig5_method_summary(data)

    print('\nAll figures saved to figures/')


if __name__ == '__main__':
    main()
