# Method notes

## AWQ in one paragraph
Quantizing `W` (group-wise INT4) injects error `ΔW`; the layer output error is `X ΔWᵀ`, so input
channels with large activations amplify error from their weights. Multiplying weight column `j`
by `s_j > 1` before quantization (and the activations by `1/s_j`) shrinks that channel's
*relative* quantization error. AWQ sets `s = mean|x|^α` and picks `α ∈ [0, 1)` per layer by grid
search on output MSE over a small calibration set (`α = 0` is plain round-to-nearest).

## What this repo simulates
`W' = Q(W·s)/s`, which is numerically what a kernel computes after folding `1/s` into the
previous op. No packed INT4 storage, no scale fusion, no clipping search.

## Proposed change
`s = mean|x|^α / mean|W_:j|^β`, searched jointly over `(α, β)`; `β = 0` is the original method,
so the search space strictly contains it and calibration MSE per layer can only decrease or
stay equal (checked by `tests/test_quant_search.py::test_beta_grid_contains_original_awq`).

Rationale: group-wise quantization step is set by the weight range in a group. Boosting a
channel that already has large weights widens that range, coarsening the step for neighbouring
channels. Dividing by weight magnitude counteracts this.

## Caveats
* Lower calibration MSE does not guarantee lower perplexity (possible overfit to calibration data).
* Cost grows linearly with the number of β values.
* Targets OPT's layout; other architectures need a different `layers` accessor.
