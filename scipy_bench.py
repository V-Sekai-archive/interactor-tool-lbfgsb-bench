#!/usr/bin/env python3
"""Traditional-solver side of the comparison: scipy.optimize.minimize with
method='L-BFGS-B' (the Fortran L-BFGS-B 3.0 of Byrd/Lu/Nocedal/Zhu that scipy
wraps) on the same problems the C++ benchmark solves.

Problem data is read from <outdir>/problems/*.txt, written by bench.cpp, and
the logistic data from <outdir>/logreg_X.bin / logreg_y.bin, written by
gen_data.py -- so all three solvers see identical problems.

    python3 scipy_bench.py <outdir> [reps_small] [reps_big] [variant]

variant: "default" (scipy's own options) or "matched" (LBFGSpp-like options).
"""
import sys
import time

import numpy as np
from scipy.optimize import minimize


# --------------------------------------------------------------- problems
def read_problem(path):
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            k, rest = parts[0], parts[1:]
            if k in ("n",):
                d[k] = int(rest[0])
            elif k in ("A_offdiag", "lambda"):
                d[k] = float(rest[0])
            elif k in ("probe_x0_f", "probe_x0_ginf", "probe_p_f", "probe_p_ginf"):
                d[k] = float(rest[0])
            else:
                d[k] = np.array([np.inf if t == "inf" else -np.inf if t == "-inf" else float(t) for t in rest])
    return d


def make_fg(name, d, outdir):
    n = d["n"]
    if name.startswith("rosen_n"):
        def fg(x):
            g = np.zeros(n)
            a = x[1:] - x[:-1] ** 2
            b = 1.0 - x[:-1]
            f = float(np.sum(100.0 * a * a + b * b))
            g[:-1] += -400.0 * x[:-1] * a - 2.0 * b
            g[1:] += 200.0 * a
            return f, g
        return fg
    if name.startswith("rosenbox_upstream"):
        def fg(x):
            g = np.zeros(n)
            f = float((x[0] - 1.0) ** 2)
            g[0] = 2 * (x[0] - 1) + 16 * (x[0] ** 2 - x[1]) * x[0]
            for i in range(1, n):
                f += 4.0 * (x[i] - x[i - 1] ** 2) ** 2
                if i == n - 1:
                    g[i] = 8 * (x[i] - x[i - 1] ** 2)
                else:
                    g[i] = 8 * (x[i] - x[i - 1] ** 2) + 16 * (x[i] ** 2 - x[i + 1]) * x[i]
            return f, g
        return fg
    if name.startswith("boxqp"):
        diag, off, b = d["A_diag"], d["A_offdiag"], d["b"]

        def fg(x):
            ax = diag * x
            ax[:-1] += off * x[1:]
            ax[1:] += off * x[:-1]
            return float(0.5 * x @ ax - b @ x), ax - b
        return fg
    if name.startswith("logreg"):
        X = np.fromfile(f"{outdir}/logreg_X.bin", dtype=np.float64).reshape(-1, n)
        y = np.fromfile(f"{outdir}/logreg_y.bin", dtype=np.float64)
        m = X.shape[0]
        lam = d["lambda"]

        def fg(w):
            z = X @ w
            t = y * z
            # softplus(-t) written stably
            f = float(np.sum(np.where(t > 0.0, np.log1p(np.exp(-np.abs(t))), -t + np.log1p(np.exp(-np.abs(t))))) / m)
            s = 1.0 / (1.0 + np.exp(t))  # sigmoid(-t)
            g = -(X.T @ (y * s)) / m + lam * w
            f += 0.5 * lam * float(w @ w)
            return f, g
        return fg
    raise SystemExit(f"unknown problem {name}")


def proj_grad_norm(x, g, lb, ub):
    v = np.where(x <= lb, np.minimum(0.0, g), np.where(x >= ub, np.maximum(0.0, g), g))
    return float(np.max(np.abs(v)))


def time_fg_us(fg, x, calls):
    """Cost of one objective call in this process, so the solver's own cost can
    be separated from the objective's (the same separation the C++ side does)."""
    t0 = time.perf_counter()
    for i in range(calls):
        fg(x)
    t1 = time.perf_counter()
    return (t1 - t0) / calls * 1e6


def main():
    outdir = sys.argv[1]
    reps_small = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    reps_big = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    variant = sys.argv[4] if len(sys.argv) > 4 else "default"

    if variant == "default":
        opts = {}
        label = "scipy-default"
    elif variant == "matched":
        # LBFGSpp's defaults: m=10, ftol=1e-4, eps=1e-5, maxls=20.
        opts = {"maxcor": 10, "ftol": 1e-4, "gtol": 1e-5, "maxls": 20, "maxiter": 20000, "maxfun": 200000}
        label = "scipy-matched"
    elif variant == "tight":
        opts = {"maxcor": 10, "ftol": 1e-12, "gtol": 1e-5, "maxls": 20, "maxiter": 20000, "maxfun": 200000}
        label = "scipy-tight"
    else:
        raise SystemExit(f"unknown variant {variant}")

    names = ["rosen_n2", "rosen_n10", "rosen_n100", "rosenbox_upstream_n25", "boxqp_n1000",
             "rosen_n1000", "boxqp_n10000", "logreg_n1000_m4000"]
    rows = []
    print("problem,solver,niter,nfev,f,pg,xstar_err,ms_median,ms_min,ms_max,note")
    for name in names:
        d = read_problem(f"{outdir}/problems/{name}.txt")
        fg = make_fg(name, d, outdir)
        lb, ub, x0 = d["lb"], d["ub"], d["x0"]
        n = d["n"]
        if "probe_x0_f" in d:
            f0, g0 = fg(x0)
            fp, gp = fg(np.full(n, 0.1))
            ok = (abs(f0 - d["probe_x0_f"]) < 1e-12 * max(1.0, abs(f0))
                  and abs(fp - d["probe_p_f"]) < 1e-12 * max(1.0, abs(fp))
                  and abs(np.max(np.abs(g0)) - d["probe_x0_ginf"]) < 1e-9 * max(1.0, np.max(np.abs(g0))))
            print(f"# {name}: python f(x0)={f0!r} ginf={np.max(np.abs(g0))!r} f(0.1)={fp!r} "
                  f"| C++ f(x0)={d['probe_x0_f']!r} ginf={d['probe_x0_ginf']!r} f(0.1)={d['probe_p_f']!r} "
                  f"-> {'AGREE' if ok else 'MISMATCH'}")
        reps = reps_big if n >= 1000 else reps_small
        times = []
        res = None
        nfev = 0
        for r in range(reps + 1):
            counter = [0]

            def fun(x, counter=counter):
                counter[0] += 1
                f, g = fg(x)
                return f, g

            t0 = time.perf_counter()
            res = minimize(fun, x0.copy(), jac=True, method="L-BFGS-B", bounds=list(zip(lb, ub)), options=dict(opts))
            t1 = time.perf_counter()
            if r > 0:
                times.append((t1 - t0) * 1e3)
            nfev = counter[0]
        times.sort()
        med = times[len(times) // 2]
        pg = proj_grad_norm(res.x, res.jac, lb, ub)
        xerr = float(np.max(np.abs(res.x - d["xstar"]))) if "xstar" in d else float("nan")
        note = f"{res.status}:{res.message.splitlines()[0][:40]}"
        calls = 200 if n >= 1000 else 20000
        fg_us = time_fg_us(fg, np.full(n, 0.1), calls)
        solver_ms = med - nfev * fg_us / 1e3
        print(f"{name:<22} {label:<13} niter={res.nit:<5d} nfev={nfev:<5d} f={res.fun:.10g} pg={pg:.3g} "
              f"xerr={xerr:.2g}  median {med:8.3f} ms (solver-only {solver_ms:8.3f})  "
              f"[min {times[0]:8.3f} max {times[-1]:8.3f}] fg={fg_us:.1f} us  {note}")
        rows.append((name, label, res.nit, nfev, res.fun, pg, xerr, med, times[0], times[-1], fg_us, solver_ms, note))

    with open(f"{outdir}/scipy_results_{variant}.csv", "w") as f:
        f.write("problem,solver,niter,nfev,f,pg,xstar_err,ms_median,ms_min,ms_max,fg_us,solver_ms,note\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
