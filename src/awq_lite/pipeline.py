"""End-to-end: load OPT, collect calibration inputs, quantize every decoder Linear, evaluate."""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .calibration import collect_inputs
from .data import calibration_batches, test_batches
from .evaluation import perplexity
from .search import search_scales


def run(model_name, bits=4, group=128, grid=20, betas=(0.0,), nsamples=64, seqlen=512, device=None):
    """Quantize with the given beta grid and return a result dict. betas=(0.0,) is original AWQ."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device).eval()
    layers = model.model.decoder.layers  # OPT layout
    test = test_batches(tok, seqlen)
    result = {"betas": list(betas), "bits": bits, "group": group, "fp_ppl": perplexity(model, test, device)}

    xs = collect_inputs(model, calibration_batches(tok, nsamples, seqlen), device, layers)
    errs, rtn_errs, chosen = [], [], []
    for li, layer in enumerate(layers):
        for n, m in layer.named_modules():
            if isinstance(m, nn.Linear):
                wq, a, b, err, rtn = search_scales(m.weight.data, xs[f"{li}.{n}"], bits, group, grid, betas)
                m.weight.data = wq
                errs.append(err); rtn_errs.append(rtn); chosen.append((a, b))
    result.update(
        ppl=perplexity(model, test, device),
        mean_layer_mse=sum(errs) / len(errs),
        mean_rtn_mse=sum(rtn_errs) / len(rtn_errs),
        frac_beta_nonzero=sum(b > 0 for _, b in chosen) / len(chosen),
    )
    return result
