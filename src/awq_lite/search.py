"""Per-layer scale search.

Original AWQ:   s = mean|x|^alpha                      (beta = 0), alpha on a grid in [0, 1)
Proposed:       s = mean|x|^alpha / mean|W_:j|^beta    jointly searched over (alpha, beta)

The scale is normalised by sqrt(max * min). The quantized weight is Q(W * s) / s, which is
what a deployed kernel computes once 1/s is folded into the preceding op, so the layer output
error measured here equals the deployed error. The objective is the MSE between the FP output
and the quantized output on calibration inputs.
"""
import torch

from .quant import fake_quant


@torch.no_grad()
def scales_for(x_mean, w_mean, alpha, beta):
    s = x_mean.pow(alpha) / w_mean.pow(beta)
    return s / (s.max() * s.min()).sqrt()


@torch.no_grad()
def search_scales(weight, x, bits=4, group=128, grid=20, betas=(0.0,)):
    """Return (best_quantized_weight, best_alpha, best_beta, best_error, plain_rtn_error).

    weight: [out, in]; x: [tokens, in] calibration inputs of this layer.
    """
    w = weight.float()
    x = x.float()
    x_mean = x.abs().mean(0).clamp(min=1e-5)
    w_mean = w.abs().mean(0).clamp(min=1e-5)
    ref = x @ w.T

    best = (float("inf"), None, 0.0, 0.0)
    rtn_err = None
    for beta in betas:
        for i in range(grid):
            alpha = i / grid
            s = scales_for(x_mean, w_mean, alpha, beta)
            wq = fake_quant(w * s[None, :], bits, group) / s[None, :]
            err = (ref - x @ wq.T).pow(2).mean().item()
            if alpha == 0.0 and beta == 0.0:
                rtn_err = err  # s == const -> plain round-to-nearest
            if err < best[0]:
                best = (err, wq, alpha, beta)
    if rtn_err is None:
        rtn_err = (ref - x @ fake_quant(w, bits, group).T).pow(2).mean().item()
    err, wq, alpha, beta = best
    return wq.to(weight.dtype), alpha, beta, err, rtn_err
