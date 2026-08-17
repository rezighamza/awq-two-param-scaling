"""Capture a random subset of input tokens for every Linear inside the decoder layers."""
import torch
import torch.nn as nn


@torch.no_grad()
def collect_inputs(model, batches, device, layers, tokens_per_batch=256):
    store, hooks = {}, []

    def make_hook(name):
        def hook(_, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1])
            sel = torch.randperm(x.shape[0], device=x.device)[:tokens_per_batch]
            store.setdefault(name, []).append(x[sel].float())
        return hook

    for li, layer in enumerate(layers):
        for n, m in layer.named_modules():
            if isinstance(m, nn.Linear):
                hooks.append(m.register_forward_pre_hook(make_hook(f"{li}.{n}")))
    for x in batches:
        model(x.to(device))
    for h in hooks:
        h.remove()
    return {k: torch.cat(v) for k, v in store.items()}
