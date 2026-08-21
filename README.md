# awq-two-param-scaling

Standalone re-implementation of **AWQ** (simulated INT4) with a weight-aware second scaling exponent.

## Original work
**AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** (Lin et al.; arXiv 2306.00978, MLSys 2024).
A small fraction of weight channels are salient, identified by activation magnitude. Scaling those input channels up before group-wise low-bit quantization (folding the inverse scale into the activations) reduces quantization error without mixed precision. The scale is `s = mean|x|^α`, with `α` grid-searched per layer to minimise output MSE on a calibration set.

Re-implemented here as simulated quantization (W4, group size 128, asymmetric) on OPT models (default `facebook/opt-125m`), calibrated on WikiText-2 and evaluated by WikiText-2 perplexity.
Deviations: fake-quant only (no packed kernels), no weight-clipping search.

## Issue
The scale depends only on activation statistics and one exponent. Channels with large weights are boosted by the same rule as small-weight channels, although the group quantization step is set by the weight range within each group. Boosting a channel that already has large weights widens the range and coarsens the step for its neighbours.

## Proposed solution (implemented)
Add a second exponent `β` discounting the scale by weight magnitude: `s = mean|x|^α / mean|W_:j|^β`, searched jointly on a 2-D `(α, β)` grid (`--betas 0 0.25 0.5`). `β = 0` is exactly the original method, so the search space contains it. Details in [docs/METHOD.md](docs/METHOD.md).

## Layout
```
src/awq_lite/
  quant.py        group-wise fake quantization
  search.py       scale search (original + proposed)
  calibration.py  forward-hook input capture
  data.py         WikiText-2 batches
  evaluation.py   perplexity
  pipeline.py     end-to-end run
  cli.py          entry point
scripts/sweep_betas.py        original vs proposed grids
tests/test_quant_search.py    unit tests (no downloads)
docs/METHOD.md
```

## Run (uv)
```
uv sync
uv run awq-lite                        # original AWQ
uv run awq-lite --betas 0 0.25 0.5     # proposed
uv run python scripts/sweep_betas.py
uv run pytest
```

## Status
Written but not executed: no tests were run and no results exist. Paper details are from memory; verify the citation. Lower calibration error does not guarantee lower perplexity.
