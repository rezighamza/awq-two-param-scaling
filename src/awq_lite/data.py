"""WikiText-2 calibration windows (train) and contiguous evaluation windows (test)."""
import torch
from datasets import load_dataset


def _ids(tok, split):
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split=split)["text"])
    return tok(text, return_tensors="pt").input_ids


def calibration_batches(tok, n_samples, seqlen, seed=0):
    ids = _ids(tok, "train")
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, ids.shape[1] - seqlen - 1, (n_samples,), generator=g)
    return [ids[:, s:s + seqlen] for s in starts.tolist()]


def test_batches(tok, seqlen):
    ids = _ids(tok, "test")
    n = ids.shape[1] // seqlen
    return [ids[:, i * seqlen:(i + 1) * seqlen] for i in range(n)]
