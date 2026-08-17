"""Original AWQ (beta=0) vs proposed beta grids.

    uv run python scripts/sweep_betas.py --model facebook/opt-125m
"""
import argparse
import json
from pathlib import Path

from awq_lite.pipeline import run

SETTINGS = {"awq (beta=0)": (0.0,), "beta<=0.25": (0.0, 0.25), "beta<=0.5": (0.0, 0.25, 0.5)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-125m")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group", type=int, default=128)
    p.add_argument("--out", default="results/sweep.json")
    a = p.parse_args()

    rows = []
    for name, betas in SETTINGS.items():
        r = run(a.model, a.bits, a.group, betas=betas)
        print(f"{name:14s} ppl {r['ppl']:.2f}  layer MSE {r['mean_layer_mse']:.3e}")
        rows.append({"setting": name, **r})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
