#!/usr/bin/env python3
"""Writes the bound-constrained logistic-regression data as raw float64.

The C++ benchmark and the scipy benchmark both read these files, so the two
solvers are handed bit-identical problem data.

    python3 gen_data.py <outdir> [n] [m] [seed]
"""
import sys

import numpy as np


def main() -> int:
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    m = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 20261029
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(m, n))
    wstar = rng.uniform(-1.0, 1.0, size=n)
    # Labels with a margin band so the problem is not separable: the classes
    # overlap near the boundary.
    z = X @ wstar
    noise = rng.normal(0.0, 0.5 * float(np.std(z)), size=m)
    y = np.where(z + noise >= 0.0, 1.0, -1.0)
    X.astype(np.float64).tofile(f"{out}/logreg_X.bin")
    y.astype(np.float64).tofile(f"{out}/logreg_y.bin")
    print(f"wrote {out}/logreg_X.bin {X.shape} {X.dtype} and logreg_y.bin ({m} labels), "
          f"positive rate {(y > 0).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
