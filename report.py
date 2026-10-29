#!/usr/bin/env python3
"""Renders results/report.md from the CSVs the two benchmark programs write.

    python3 report.py <results dir>

Reads:
  <dir>/dc/cpp_results.csv            driver + LBFGSpp, delta 1e-3 (the repo's
                                      diffcloth parameters)
  <dir>/tight/cpp_results.csv         driver + LBFGSpp, delta 1e-10
  <dir>/dc/scipy_results_default.csv  scipy's own L-BFGS-B options
  <dir>/dc/scipy_results_matched.csv  scipy at LBFGSpp-like options
  <dir>/dc/scipy_results_tight.csv    scipy at a near-machine ftol
"""
import csv
import sys
from pathlib import Path

ORDER = ["rosen_n2", "rosen_n10", "rosen_n100", "rosenbox_upstream_n25", "boxqp_n1000",
         "rosen_n1000", "boxqp_n10000", "logreg_n1000_m4000"]
N = {"rosen_n2": 2, "rosen_n10": 10, "rosen_n100": 100, "rosenbox_upstream_n25": 25,
     "boxqp_n1000": 1000, "rosen_n1000": 1000, "boxqp_n10000": 10000, "logreg_n1000_m4000": 1000}


def load(path):
    """problem -> solver -> row"""
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    with p.open() as f:
        for r in csv.DictReader(f):
            out.setdefault(r["problem"], {})[r["solver"]] = r
    return out


def num(row, key, fmt="{:.4g}"):
    if row is None:
        return "-"
    try:
        return fmt.format(float(row[key]))
    except (ValueError, TypeError, KeyError):
        return "-"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    dc = load(root / "dc/cpp_results.csv")
    tight = load(root / "tight/cpp_results.csv")
    sci = {v: load(root / f"dc/scipy_results_{v}.csv") for v in ("default", "matched", "tight")}
    out = []
    w = out.append

    env = root / "environment.txt"
    if env.exists():
        w("## Environment\n\n```")
        w(env.read_text().strip())
        w("```\n")

    # ---- algorithmic agreement: driver vs the solver it mirrors
    for name, data, delta in (("dc", dc, "1e-3 (the repo's diffcloth parameters)"),
                              ("tight", tight, "1e-10 (the delta test out of the way)")):
        w(f"### Iterations and objective evaluations -- delta = {delta}\n")
        rows = []
        for prob in ORDER:
            d = data.get(prob, {}).get("driver")
            l = data.get(prob, {}).get("lbfgspp")
            rows.append([prob, N[prob], num(d, "niter", "{:.0f}"), num(d, "nfev", "{:.0f}"),
                         num(l, "niter", "{:.0f}"), num(l, "nfev", "{:.0f}"),
                         num(d, "f", "{:.10g}"), num(l, "f", "{:.10g}"),
                         num(d, "pg", "{:.3g}"), num(l, "pg", "{:.3g}")])
        w(table(["problem", "n", "driver iters", "driver nfev", "LBFGSpp iters", "LBFGSpp nfev",
                 "driver f", "LBFGSpp f", "driver pg", "LBFGSpp pg"], rows))
        w("")

    # ---- wall time, dc variant, three solvers
    for name, data, variant, title in (("dc", dc, "matched", "delta 1e-3 vs scipy at LBFGSpp-like options"),
                                       ("tight", tight, "tight", "delta 1e-10 vs scipy at a near-machine ftol")):
        w(f"### Wall time -- {title}\n")
        w("Median of the timed repetitions, in milliseconds. `solver` excludes the objective "
          "calls both programs time separately, so the number is the solver's own cost.\n")
        rows = []
        for prob in ORDER:
            d = data.get(prob, {}).get("driver")
            l = data.get(prob, {}).get("lbfgspp")
            s = sci[variant].get(prob, {}).get("scipy-" + variant)
            ratio = "-"
            if d and l and float(l["solver_ms"]) > 0:
                ratio = f"{float(d['solver_ms']) / float(l['solver_ms']):.2f}x"
            sratio = "-"
            if d and s and float(s["solver_ms"]) > 0:
                sratio = f"{float(d['solver_ms']) / float(s['solver_ms']):.2f}x"
            rows.append([prob, num(d, "ms_median", "{:.3f}"), num(l, "ms_median", "{:.3f}"),
                         num(s, "ms_median", "{:.3f}"), num(d, "solver_ms", "{:.3f}"),
                         num(l, "solver_ms", "{:.3f}"), num(s, "solver_ms", "{:.3f}"), ratio, sratio])
        w(table(["problem", "driver total", "LBFGSpp total", "scipy total", "driver solver",
                 "LBFGSpp solver", "scipy solver", "driver/LBFGSpp", "driver/scipy"], rows))
        w("")

    # ---- what each solver stopped at
    w("### Where each solver stopped\n")
    rows = []
    for prob in ORDER:
        d = dc.get(prob, {}).get("driver")
        l = dc.get(prob, {}).get("lbfgspp")
        s = sci["default"].get(prob, {}).get("scipy-default")
        rows.append([prob, num(d, "f", "{:.10g}"), num(s, "f", "{:.10g}"),
                     num(d, "pg", "{:.3g}"), num(l, "pg", "{:.3g}"), num(s, "pg", "{:.3g}"),
                     num(d, "xstar_err", "{:.2g}"), num(s, "xstar_err", "{:.2g}"),
                     (d["note"].split(":")[-1] if d else "-"), (s["note"].split(":")[0] if s else "-")])
    w(table(["problem", "driver f (dc)", "scipy f (default)", "driver pg", "LBFGSpp pg", "scipy pg",
             "driver x* err", "scipy x* err", "driver stop", "scipy status"], rows))
    w("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
