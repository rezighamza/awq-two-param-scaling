### Benchmark Comparison: AWQ Scaling Exponents

| setting        | perplexity | layer MSE   |
|----------------|------------|-------------|
| awq (beta=0)   | 25.14      | 4.312e-02   |
| beta<=0.25     | 24.89      | 3.985e-02   |
| beta<=0.5      | 24.51      | 3.512e-02   |

**Conclusion:** Introducing the second weight-aware exponent `beta` yields a better local minimum with lower layer-wise MSE and overall perplexity at the same INT4 precision level.
