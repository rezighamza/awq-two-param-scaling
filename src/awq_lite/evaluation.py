import torch


@torch.no_grad()
def perplexity(model, batches, device):
    nll, count = 0.0, 0
    for x in batches:
        x = x.to(device)
        nll += model(x, labels=x).loss.item() * (x.shape[1] - 1)
        count += x.shape[1] - 1
    return torch.exp(torch.tensor(nll / count)).item()
