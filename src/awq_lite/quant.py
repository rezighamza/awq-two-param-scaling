"""Group-wise asymmetric integer fake quantization (quantize then dequantize)."""
import torch


def fake_quant(w, bits=4, group=128):
    """w: [out, in] with in % group == 0. Each run of `group` input weights shares scale/zero-point."""
    if w.shape[-1] % group:
        raise ValueError(f"in_features {w.shape[-1]} not divisible by group size {group}")
    shape = w.shape
    w = w.reshape(-1, group)
    mx, mn = w.amax(1, keepdim=True), w.amin(1, keepdim=True)
    qmax = 2 ** bits - 1
    scale = (mx - mn).clamp(min=1e-5) / qmax
    zp = (-mn / scale).round()
    q = (torch.round(w / scale) + zp).clamp(0, qmax)
    return ((q - zp) * scale).reshape(shape)
