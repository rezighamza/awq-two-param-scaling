"""AWQ-style activation-aware weight quantization (simulated) with two-parameter scaling."""
from .quant import fake_quant
from .search import search_scales

__all__ = ["fake_quant", "search_scales"]
