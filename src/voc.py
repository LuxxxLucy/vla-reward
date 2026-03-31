"""
VOC (Value-Order Correlation) metric.

Measures whether a reward curve correctly orders video prefixes by
temporal progress. A perfect reward curve (monotone increasing) scores 1.0.
Near-zero means the curve is no better than random.

This is Spearman rank correlation between the reward values and the
chronological frame indices [0, 1, ..., K-1].
"""

import numpy as np
from scipy.stats import spearmanr


def compute_voc(rewards: list[float]) -> float:
    """Spearman correlation of rewards vs. chronological order.

    Args:
        rewards: List of reward values at K timepoints (must be length >= 2).

    Returns:
        VOC in [-1, 1]. Higher is better. 1.0 = perfectly monotone increasing.
    """
    if len(rewards) < 2:
        return 0.0
    indices = list(range(len(rewards)))
    corr, _ = spearmanr(indices, rewards)
    return float(corr) if not np.isnan(corr) else 0.0


def mean_voc(episodes: list[list[float]]) -> tuple[float, float]:
    """Compute mean ± std VOC across a list of episodes.

    Returns:
        (mean_voc, std_voc)
    """
    vocs = [compute_voc(ep) for ep in episodes]
    return float(np.mean(vocs)), float(np.std(vocs))
