### Benchmark Comparison: AWQ Scaling Exponents

| setting        | perplexity | layer MSE   |
|----------------|------------|-------------|
| awq (beta=0)   | 40.37      | 2.972e-03   |
| beta<=0.25     | 40.41      | 2.824e-03   |
| beta<=0.5      | 40.02      | 2.758e-03   |

**Conclusion:** Introducing the second weight-aware exponent `beta` yields a better local minimum, lowering the layer-wise MSE and achieving the lowest overall perplexity (40.02) at the same INT4 precision level.
