# interactor-tool-lbfgsb-bench

Measures the L-BFGS-B that [contract-lbfgsb](https://github.com/V-Sekai-fire/contract-lbfgsb)
writes in Lean and emits through Slang, against traditional solvers of the same
type:

* **LBFGSpp 0.3.0** — the bounded quasi-Newton solver contract-lbfgsb mirrors,
  compiled from its own headers, at the same parameters.
* **scipy's L-BFGS-B** — the Fortran L-BFGS-B 3.0 of Byrd, Lu, Nocedal and Zhu,
  the reference implementation most people reach for.

All three solve the same problems, defined once in `bench.cpp` and dumped as
text so the Python side reads identical data.

## What is measured

The subject is contract-lbfgsb's in-guest driver: `guest/drape/lbfgsb.cpp` (the
scalar control flow) over `guest/drape/vec_cpu.cpp` (the Lean-emitted kernels,
`slangc -target cpp`). That is the same pair of files the `drape.elf` guest
compiles, so this is the driver's real arithmetic — float32 vectors, double
scalars, one kernel dispatch per vector operation — run on a host CPU.

Problems, all bound-constrained:

| problem | n | what it is |
|---|---|---|
| `rosen_n2`, `rosen_n10`, `rosen_n100` | 2, 10, 100 | chained Rosenbrock with mixed, partly one-sided and one fixed coordinate |
| `rosenbox_upstream_n25` | 25 | LBFGSpp's own `example-rosenbrock-box.cpp` setup |
| `boxqp_n1000`, `boxqp_n10000` | 1000, 10000 | box QP built from a known KKT point with strict complementarity |
| `rosen_n1000` | 1000 | the chained Rosenbrock at scale |
| `logreg_n1000_m4000` | 1000 | bound-constrained L2 logistic regression, 4000 samples |

The first five are contract-lbfgsb's own oracle problems
(`tests/lbfgsb_oracle/gen.cpp`), so the comparison sits on the same ground the
repository's correctness gate uses.

Two parameter settings:

* **dc** — `m 10, delta 1e-3, max_linesearch 20, epsilon 1e-5`: the parameters
  cloth-dynamics' `BackwardTaskSolver` uses, and the ones the driver ships with.
* **tight** — the same with `delta 1e-10`, so the function-difference test stays
  out of the way and the gradient test decides.

Reported per solve: iterations, objective evaluations, the objective and the
projected-gradient norm at the stopping point, and wall time. The objective is
timed on its own (`fg_us`), so `solver_ms = median - nfev * fg_us` separates the
solver's cost from the objective's — both C++ solvers call the same objective
code, so that split is exact for them.

## Run

```sh
./run.sh              # clones the pinned dependencies into .deps/, runs all
                      # solvers, writes results/ and results/report.md
CXX=clang++ ./run.sh
REPS_SMALL=50 REPS_BIG=10 ./run.sh
```

`run.sh` fetches three public repositories at the revisions the committed
results were measured at: contract-lbfgsb, interactor-aria-lbfgspp (which
carries LBFGSpp 0.3.0 and Eigen) and contract-guest-runtime (the `slang-rt`
headers `vec_cpu.cpp` includes). LBFGSpp and Eigen are used only here, by a
host-native benchmark — they are linked into no guest ELF.

## Results

Apple M2 Pro, macOS 26.6.2, Apple clang 21.0.0, scipy 1.13.1, numpy 2.0.2, at
contract-lbfgsb `5fe3581`. Median of 20 repetitions (5 above n=1000), after a
warm-up. The full tables, including both scipy parameter variants, are in
[`results/report.md`](results/report.md).

### 1. The driver does exactly LBFGSpp's work

Iteration and evaluation counts, driver / LBFGSpp, are identical on all 16
problem-and-setting pairs:

| problem | n | iters dc | nfev dc | iters tight | nfev tight | f dc (driver / LBFGSpp) | f tight (driver / LBFGSpp) |
|---|---|---|---|---|---|---|---|
| rosen_n2 | 2 | 19/19 | 25/25 | 21/21 | 27/27 | 0.0506256 / 0.0506045 | 0.0505961 / 0.0505961 |
| rosen_n10 | 10 | 11/11 | 12/12 | 40/40 | 45/45 | 32.7987 / 32.7987 | 21.9431 / 21.9431 |
| rosen_n100 | 100 | 12/12 | 13/13 | 43/43 | 49/49 | 122.105 / 122.105 | 118.811 / 118.811 |
| rosenbox_upstream_n25 | 25 | 10/10 | 11/11 | 13/13 | 14/14 | 360.285 / 360.285 | 360.284 / 360.284 |
| boxqp_n1000 | 1000 | 5/5 | 8/8 | 11/11 | 14/14 | -1932.88 / -1932.88 | -1932.99 / -1932.99 |
| rosen_n1000 | 1000 | 11/11 | 12/12 | 43/43 | 49/49 | 1013.04 / 1013.04 | 1009.72 / 1009.72 |
| boxqp_n10000 | 10000 | 5/5 | 11/11 | 11/11 | 17/17 | -19315.1 / -19315.1 | -19316.1 / -19316.1 |
| logreg_n1000_m4000 | 1000 | 10/10 | 11/11 | 17/17 | 18/18 | 0.159539 / 0.159539 | 0.157505 / 0.157505 |

The returned points agree to `max|x_driver - x_LBFGSpp|` <= 1.6e-4 in the dc
setting (the float32 vectors make the function-difference test fire at a
slightly different accepted point; `rosen_n2` is the worst case) and <= 1.8e-6
in the tight setting. `check.py` gates on this.

### 2. Wall time on the CPU, solver only (ms)

dc — the shipped parameters:

| problem | driver | LBFGSpp | scipy (matched options) | driver/LBFGSpp | driver/scipy |
|---|---|---|---|---|---|
| rosen_n2 | 0.060 | 0.102 | 0.338 | 0.59x | 0.18x |
| rosen_n10 | 0.041 | 0.046 | 1.132 | 0.90x | 0.04x |
| rosen_n100 | 0.447 | 0.099 | 0.551 | 4.54x | 0.81x |
| rosenbox_upstream_n25 | 0.067 | 0.051 | 0.330 | 1.32x | 0.20x |
| boxqp_n1000 | 1.836 | 0.460 | 2.648 | 4.00x | 0.69x |
| rosen_n1000 | 3.579 | 0.571 | 3.002 | 6.26x | 1.19x |
| boxqp_n10000 | 21.677 | 4.278 | 24.302 | 5.07x | 0.89x |
| logreg_n1000_m4000 | 2.873 | 0.986 | 5.635 | 2.91x | 0.51x |

tight — the delta test out of the way:

| problem | driver | LBFGSpp | scipy (near-machine ftol) | driver/LBFGSpp | driver/scipy |
|---|---|---|---|---|---|
| rosen_n2 | 0.056 | 0.086 | 0.376 | 0.65x | 0.15x |
| rosen_n10 | 0.179 | 0.193 | 1.243 | 0.93x | 0.14x |
| rosen_n100 | 1.676 | 0.437 | 1.331 | 3.84x | 1.26x |
| rosenbox_upstream_n25 | 0.083 | 0.057 | 0.359 | 1.46x | 0.23x |
| boxqp_n1000 | 3.591 | 0.703 | 3.238 | 5.11x | 1.11x |
| rosen_n1000 | 17.628 | 3.458 | 6.223 | 5.10x | 2.83x |
| boxqp_n10000 | 42.831 | 7.812 | 28.125 | 5.48x | 1.52x |
| logreg_n1000_m4000 | 8.506 | 5.833 | 5.861 | 1.46x | 1.45x |

Two shapes show up:

* **Small problems: the driver wins.** Below n~25 the Lean driver's kernel
  dispatch costs less than LBFGSpp's Eigen setup and exception scaffolding, and
  far less than scipy's per-iteration Python round trip.
* **Large problems: the driver costs 4-6x LBFGSpp.** Each vector operation is
  one dispatched kernel over float32 buffers; LBFGSpp does the same arithmetic
  in double with Eigen's vectorised loops. The ratio is flat (4.0x-6.3x) across
  n=100..10000, so it is per-operation overhead, not an n-dependent term.
* **When the objective dominates, the difference disappears.** On the logistic
  problem one evaluation costs 4.2 ms in C++ against a whole solve's 2.9 ms of
  solver time: total wall time is 48.9 ms for the driver and 47.0 ms for
  LBFGSpp.

### 3. Against scipy's Fortran L-BFGS-B

scipy's defaults (`ftol` 2.2e-9, `gtol` 1e-5, `maxls` 20) reach a much tighter
projected gradient than either C++ solver in the dc setting, at more iterations
and at the cost of a Python objective callback of 6-24 us per call:

| problem | driver pg (dc) | driver pg (tight) | scipy pg (default) | scipy iters | scipy nfev | scipy total ms |
|---|---|---|---|---|---|---|
| rosen_n2 | 0.169 | 3.81e-06 | 6.16e-07 | 13 | 20 | 0.61 |
| rosen_n10 | 1.96 | 0.000101 | 0.00213 | 49 | 66 | 1.63 |
| rosen_n100 | 0.642 | 0.000273 | 0.00048 | 41 | 47 | 1.65 |
| rosenbox_upstream_n25 | 0.435 | 1.22e-05 | 8.7e-06 | 13 | 14 | 0.61 |
| boxqp_n1000 | 0.317 | 0.000183 | 0.000189 | 11 | 12 | 3.03 |
| rosen_n1000 | 1.53 | 0.000692 | 0.0065 | 42 | 48 | 6.28 |
| boxqp_n10000 | 0.351 | 0.000186 | 0.000192 | 11 | 12 | 27.22 |
| logreg_n1000_m4000 | 0.000497 | 0.000102 | 5.71e-06 | 30 | 32 | 48.59 |

At the shipped dc parameters the driver's whole solve is 0.04x-1.19x scipy's
wall time, but that is not a like-for-like stop: `delta 1e-3` is a looser
function-difference test than scipy's default `ftol`, so the driver stops
earlier on a worse point. With the delta test out of the way (tight) the
comparison is fair, and the driver is faster than scipy on the small problems
and 1.5x-2.8x slower on `rosen_n1000` and `boxqp_n10000`.

scipy's `ftol` and LBFGSpp's `delta` are not the same test, which is why the
report carries three scipy variants: `default`, `matched` (`ftol` 1e-4, `gtol`
1e-5, `maxcor` 10, `maxls` 20 — LBFGSpp-like) and `tight` (`ftol` 1e-12).

## The gate

`check.py` fails the run if either solver stops without converging, if the
driver's iteration count leaves `max(2, 25%)` of LBFGSpp's (the tolerance
contract-lbfgsb's own Gate 5 G2 applies), if the final objective differs by
more than 1e-3 relative, if the driver's projected gradient is both more than
4x LBFGSpp's and above 1e-4, or if fewer than half of the problem/setting pairs
match LBFGSpp's iteration *and* evaluation counts exactly.

## What this does not measure

* **The GPU and guest path.** `vec_rd.cpp` (the same kernels as SPIR-V over a
  sandbox device) is not runnable on a host without the guest runtime. The
  numbers here are the CPU backend, which is a scalar emulation of the kernels
  with the groupshared reductions run as their serial siblings; they are not
  the design's intended throughput.
* **float32 vs double accuracy.** The driver's vectors are float32, LBFGSpp's
  and scipy's are double. That is a property of the driver, not of the harness,
  and it is what the x-agreement figures above show.
* **The objective's own speed.** The C++ problems are plain loops; the scipy
  ones are numpy, which uses threads for the logistic matvecs (1.4 ms per call
  against 4.2 ms in C++). Only the driver-vs-LBFGSpp comparison holds the
  objective exactly constant.
* **Correctness beyond these eight problems.** That is contract-lbfgsb's own
  Gate 5, on its 20 component fixtures and 20 recorded traces.

## Licence

MIT. LBFGSpp is MIT, Eigen is MPL-2.0, scipy is BSD-3-Clause; none of them is
redistributed here, `run.sh` fetches them.
