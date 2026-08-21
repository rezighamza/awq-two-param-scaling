import pytest
import torch

from awq_lite.quant import fake_quant
from awq_lite.search import scales_for, search_scales


def test_fake_quant_levels_per_group():
    w = torch.randn(4, 256)
    q = fake_quant(w, bits=4, group=128)
    assert q.shape == w.shape
    for row in q.reshape(-1, 128):
        assert row.unique().numel() <= 16


def test_fake_quant_error_shrinks_with_bits():
    w = torch.randn(8, 128)
    e4 = (w - fake_quant(w, 4, 128)).pow(2).mean()
    e8 = (w - fake_quant(w, 8, 128)).pow(2).mean()
    assert e8 < e4


def test_group_divisibility_enforced():
    with pytest.raises(ValueError):
        fake_quant(torch.randn(2, 100), 4, 128)


def test_scale_normalisation():
    s = scales_for(torch.rand(64) + 0.1, torch.rand(64) + 0.1, 0.5, 0.25)
    assert torch.isclose(s.max() * s.min(), torch.tensor(1.0), atol=1e-4)


def test_search_never_worse_than_rtn():
    torch.manual_seed(0)
    w = torch.randn(32, 128)
    x = torch.randn(256, 128)
    x[:, :4] *= 20  # salient activation channels
    _, _, _, err, rtn = search_scales(w, x, bits=4, group=128, grid=10)
    assert err <= rtn + 1e-9


def test_beta_grid_contains_original_awq():
    """With betas=(0, b) the optimum can only match or beat betas=(0,) on calibration MSE."""
    torch.manual_seed(1)
    w = torch.randn(32, 128) * (torch.rand(128) * 3 + 0.1)
    x = torch.randn(256, 128) * (torch.rand(128) * 3 + 0.1)
    base = search_scales(w, x, 4, 128, 10, betas=(0.0,))[3]
    ext = search_scales(w, x, 4, 128, 10, betas=(0.0, 0.25, 0.5))[3]
    assert ext <= base + 1e-12
