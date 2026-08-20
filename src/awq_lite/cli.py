import argparse

from .pipeline import run


def main():
    p = argparse.ArgumentParser(description="AWQ-style fake quantization with optional beta search")
    p.add_argument("--model", default="facebook/opt-125m")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group", type=int, default=128)
    p.add_argument("--grid", type=int, default=20)
    p.add_argument("--betas", type=float, nargs="+", default=[0.0],
                   help="0.0 = original AWQ; e.g. 0 0.25 0.5 for the proposed search")
    p.add_argument("--nsamples", type=int, default=64)
    p.add_argument("--seqlen", type=int, default=512)
    a = p.parse_args()
    r = run(a.model, a.bits, a.group, a.grid, tuple(a.betas), a.nsamples, a.seqlen)
    print(f"fp ppl {r['fp_ppl']:.2f} | W{r['bits']} g{r['group']} betas={r['betas']}: ppl {r['ppl']:.2f}")
    print(f"mean layer MSE {r['mean_layer_mse']:.3e} (RTN {r['mean_rtn_mse']:.3e}), "
          f"layers using beta>0: {r['frac_beta_nonzero']:.0%}")


if __name__ == "__main__":
    main()
