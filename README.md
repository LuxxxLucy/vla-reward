# vla-reward

Cheap reward signals for VLA robotics via VLM token probabilities.

Reimplements [TOPreward](https://arxiv.org/abs/2602.19313) (Chen et al., 2026) and investigates whether prompt ensembles on a 2B model can match 8B quality. Apple Silicon / MLX. PyTorch and API backends available in `src/backend.py`.

## Setup

```bash
uv sync
```

## Usage

```bash
uv run python scripts/run.py --method baseline --backend qwen-vl-2b \
  --video data/videos/put_pen_cup.mp4 --instruction "Put the pen into the cup."

uv run python scripts/run_batch.py --method ensemble --backend qwen-vl-2b --n-prompts 3
uv run python scripts/plot_results.py
```

## Methods

| Method | Formula |
|--------|---------|
| `baseline` | R = log P("True") |
| `contrastive` | R = log P("True") − log P("False") |
| `ensemble` | mean baseline over N prompts |
| `contrast_ensemble` | mean contrastive over N prompts |
