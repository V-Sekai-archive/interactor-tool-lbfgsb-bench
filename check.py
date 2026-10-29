#!/usr/bin/env python3
"""The gate: the Lean-emitted driver must reproduce the solver it mirrors.

Fails (exit 1) if, for any problem in either parameter setting:

  * either solver failed to converge,
  * the driver took more than max(2, 25%) more (or fewer) iterations than
    LBFGSpp -- the same tolerance contract-lbfgsb's own Gate 5 G2 applies,
  * the final objective differs by more than 1e-3 relative,
  * the driver's final projected-gradient norm is more than 4x LBFGSpp's and
    above 1e-4 absolute,

or if fewer than half of the problem/setting pairs match LBFGSpp's iteration
AND evaluation counts exactly (the fidelity claim the README makes: identical
control flow, so identical work per problem).

    python3 check.py <results dir>
"""
import csv
import sys
from pathlib import Path

FAILS = []


def load(path):
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    with p.open() as f:
        for r in csv.DictReader(f):
            out.setdefault(r["problem"], {})[r["solver"]] = r
    return out


def load_variant(root, variant):
    """The CPU pass, plus the GPU pass where the run produced one."""
    data = load(root / variant / "cpp_results.csv")
    if not data:
        FAILS.append(f"missing {root / variant / 'cpp_results.csv'}")
    for prob, solvers in load(root / variant / "cpp_results_vk.csv").items():
        data.setdefault(prob, {}).update(solvers)
    return data


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    exact = 0
    vk_exact = 0
    pairs = 0
    for variant in ("dc", "tight"):
        data = load_variant(root, variant)
        for prob, solvers in sorted(data.items()):
            d, l = solvers.get("driver"), solvers.get("lbfgspp")
            if d is None or l is None:
                FAILS.append(f"{variant}/{prob}: a solver row is missing")
                continue
            pairs += 1
            tag = f"{variant}/{prob}"
            for name, row in (("driver", d), ("lbfgspp", l)):
                if not row["note"].startswith("CONVERGED"):
                    FAILS.append(f"{tag}: {name} did not converge ({row['note']})")
            di, li = int(d["niter"]), int(l["niter"])
            dn, ln = int(d["nfev"]), int(l["nfev"])
            if di == li and dn == ln:
                exact += 1
            if di > 1.25 * li + 2 or li > 1.25 * di + 2:
                FAILS.append(f"{tag}: iterations {di} vs {li} outside max(2, 25%)")
            df, lf = float(d["f"]), float(l["f"])
            if abs(df - lf) > 1e-3 * max(1.0, abs(lf)):
                FAILS.append(f"{tag}: f {df!r} vs {lf!r} differ by more than 1e-3 relative")
            dpg, lpg = float(d["pg"]), float(l["pg"])
            if dpg > max(1e-4, 4.0 * lpg):
                FAILS.append(f"{tag}: projected gradient {dpg!r} vs {lpg!r} (driver 4x worse and above 1e-4)")

            # The GPU backend, where the run has one. It is the same driver and
            # the same kernels, but a different instruction stream, so it is
            # held to the tolerance contract-lbfgsb's own Gate 5 G2 applies to
            # a GPU result (iterations within max(2, 25%), f within 1e-3
            # relative) rather than to bit-equality with the CPU. The exact
            # matches are counted and reported.
            g = solvers.get("driver-vk")
            if g is None:
                continue
            if not g["note"].startswith("CONVERGED"):
                FAILS.append(f"{tag}: driver-vk did not converge ({g['note']})")
                continue
            gi = int(g["niter"])
            if gi > 1.25 * di + 2 or di > 1.25 * gi + 2:
                FAILS.append(f"{tag}: driver-vk {gi} iterations vs the CPU driver's {di}, outside max(2, 25%)")
            gf = float(g["f"])
            if abs(gf - df) > 1e-3 * max(1.0, abs(df)):
                FAILS.append(f"{tag}: driver-vk f {gf!r} vs the CPU driver's {df!r}, more than 1e-3 relative")
            if gi == di and int(g["nfev"]) == dn:
                vk_exact += 1
            elif gi != di or int(g["nfev"]) != dn:
                print(f"note {tag}: driver-vk did {gi} iterations and {g['nfev']} evaluations, "
                      f"the CPU driver {di} and {dn}")

    if pairs and exact * 2 < pairs:
        FAILS.append(f"only {exact}/{pairs} problem/setting pairs matched iteration and evaluation counts exactly")

    print(f"{exact}/{pairs} problem/setting pairs matched LBFGSpp's iteration and evaluation counts exactly")
    if vk_exact:
        print(f"{vk_exact}/{pairs} problem/setting pairs had the GPU backend match the CPU driver's "
              f"iteration and evaluation counts exactly")
    for f in FAILS:
        print(f"FAIL {f}")
    if FAILS:
        return 1
    print("OK: the driver reproduces LBFGSpp's work and result within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
