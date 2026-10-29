# interactor-tool-lbfgsb-bench

Measures the L-BFGS-B that [contract-lbfgsb](https://github.com/V-Sekai-fire/contract-lbfgsb)
writes in Lean and emits through Slang, against traditional solvers of the same
type:

* **LBFGSpp 0.3.0** — the bounded quasi-Newton solver contract-lbfgsb mirrors,
  compiled from its own headers, at the same parameters.
* **scipy's L-BFGS-B** — the Fortran L-BFGS-B 3.0 of Byrd, Lu, Nocedal and Zhu,
  the reference implementation most people reach for.

All three solve the same problems, defined once in `bench.cpp` and dumped as
text so the Python side reads identical data. The driver runs on two backends,
CPU and GPU, and both are measured.

## What is measured

The subject is contract-lbfgsb's in-guest driver: `guest/drape/lbfgsb.cpp` (the
scalar control flow) over a vector backend. Two are benchmarked:

* **CPU** — `guest/drape/vec_cpu.cpp`, the `slangc -target cpp` emits of the
  Lean kernels. This is the pair the host-native test compiles, and the
  arithmetic is the guest's: float32 vectors, double scalars, one kernel
  dispatch per vector operation.
* **GPU** — `vec_vk.cpp` (this repository), the same kernels as SPIR-V
  dispatched through Vulkan. In the guest those go through `vec_rd.cpp` and
  Godot's RenderingDevice across the sandbox boundary; here they go through the
  Vulkan loader directly, which is the same dispatch sequence without the
  boundary. MoltenVK on macOS, the real driver on Linux.

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
./run.sh              # clones the pinned dependencies into .deps/, runs every
                      # solver and backend, writes results/ and results/report.md
CXX=clang++ ./run.sh
REPS_SMALL=50 REPS_BIG=10 ./run.sh
SLANGC=/path/to/slangc ./run.sh
```

`run.sh` fetches three public repositories at the revisions the committed
results were measured at: contract-lbfgsb, interactor-aria-lbfgspp (which
carries LBFGSpp 0.3.0 and Eigen) and contract-guest-runtime (the `slang-rt`
headers `vec_cpu.cpp` includes). LBFGSpp and Eigen are used only here, by a
host-native benchmark — they are linked into no guest ELF.

The GPU pass needs `slangc` (to emit SPIR-V from the committed `.slang`) and a
Vulkan device; without either it is skipped and the CPU results stand alone. The
CPU and GPU passes run in separate processes and write separate CSVs, so neither
one's timing disturbs the other's.

## Results

Apple M2 Pro, macOS 26.6.2, Apple clang 21.0.0, scipy 1.13.1, numpy 2.0.2, at
contract-lbfgsb `5fe3581`, GPU via MoltenVK 1.4.363. Median of 20 repetitions
(5 above n=1000), after a warm-up. The full tables, including both scipy
parameter variants, are in [`results/report.md`](results/report.md).

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
| rosen_n2 | 0.059 | 0.101 | 0.316 | 0.59x | 0.19x |
| rosen_n10 | 0.042 | 0.050 | 1.120 | 0.83x | 0.04x |
| rosen_n100 | 0.443 | 0.093 | 0.570 | 4.78x | 0.78x |
| rosenbox_upstream_n25 | 0.067 | 0.051 | 0.310 | 1.31x | 0.22x |
| boxqp_n1000 | 1.840 | 0.445 | 2.871 | 4.13x | 0.64x |
| rosen_n1000 | 3.560 | 0.610 | 3.071 | 5.83x | 1.16x |
| boxqp_n10000 | 21.281 | 4.187 | 24.478 | 5.08x | 0.87x |
| logreg_n1000_m4000 | 3.780 | 2.022 | 5.600 | 1.87x | 0.68x |

tight — the delta test out of the way:

| problem | driver | LBFGSpp | scipy (near-machine ftol) | driver/LBFGSpp | driver/scipy |
|---|---|---|---|---|---|
| rosen_n2 | 0.051 | 0.086 | 0.393 | 0.59x | 0.13x |
| rosen_n10 | 0.189 | 0.191 | 1.227 | 0.99x | 0.15x |
| rosen_n100 | 1.794 | 0.465 | 1.340 | 3.86x | 1.34x |
| rosenbox_upstream_n25 | 0.087 | 0.061 | 0.349 | 1.42x | 0.25x |
| boxqp_n1000 | 3.936 | 0.831 | 3.241 | 4.74x | 1.21x |
| rosen_n1000 | 18.173 | 3.288 | 6.285 | 5.53x | 2.89x |
| boxqp_n10000 | 42.611 | 7.716 | 29.923 | 5.52x | 1.42x |
| logreg_n1000_m4000 | 6.702 | 1.832 | 14.260 | 3.66x | 0.47x |

Two shapes show up:

* **Small problems: the driver wins.** Below n~25 the Lean driver's kernel
  dispatch costs less than LBFGSpp's Eigen setup and exception scaffolding, and
  far less than scipy's per-iteration Python round trip.
* **Large problems: the driver costs 4-6x LBFGSpp.** Each vector operation is
  one dispatched kernel over float32 buffers; LBFGSpp does the same arithmetic
  in double with Eigen's vectorised loops. The ratio is flat (4.1x-5.8x) across
  n=100..10000, so it is per-operation overhead, not an n-dependent term.
* **When the objective dominates, the difference disappears.** On the logistic
  problem one evaluation costs 4.2 ms in C++ against a whole solve's 3.8 ms of
  solver time: total wall time is 49.5 ms for the driver and 47.7 ms for
  LBFGSpp.

### 3. Against scipy's Fortran L-BFGS-B

scipy's defaults (`ftol` 2.2e-9, `gtol` 1e-5, `maxls` 20) reach a much tighter
projected gradient than either C++ solver in the dc setting, at more iterations
and at the cost of a Python objective callback of 6-24 us per call:

| problem | driver pg (dc) | driver pg (tight) | LBFGSpp pg (tight) | scipy pg (default) | scipy iters | scipy nfev | scipy total ms |
|---|---|---|---|---|---|---|---|
| rosen_n2 | 0.169 | 3.81e-06 | 3.47e-08 | 6.16e-07 | 13 | 20 | 0.52 |
| rosen_n10 | 1.96 | 0.000101 | 7.48e-05 | 0.00213 | 49 | 66 | 1.62 |
| rosen_n100 | 0.642 | 0.000273 | 0.000273 | 0.00048 | 41 | 47 | 1.60 |
| rosenbox_upstream_n25 | 0.435 | 1.22e-05 | 7.02e-06 | 8.7e-06 | 13 | 14 | 0.63 |
| boxqp_n1000 | 0.317 | 0.000183 | 0.000183 | 0.000189 | 11 | 12 | 3.47 |
| rosen_n1000 | 1.53 | 0.000692 | 0.000358 | 0.0065 | 42 | 48 | 6.24 |
| boxqp_n10000 | 0.351 | 0.000186 | 0.000186 | 0.000192 | 11 | 12 | 26.15 |
| logreg_n1000_m4000 | 0.000497 | 0.000102 | 0.000102 | 5.71e-06 | 30 | 32 | 48.70 |

At the shipped dc parameters the driver's whole solve is 0.04x-1.16x scipy's
wall time, but that is not a like-for-like stop: `delta 1e-3` is a looser
function-difference test than scipy's default `ftol`, so the driver stops
earlier on a worse point. With the delta test out of the way (tight) the
comparison is fair, and the driver is faster than scipy on the small problems
and 1.3x-2.9x slower from `rosen_n100` upward.

scipy's `ftol` and LBFGSpp's `delta` are not the same test, which is why the
report carries three scipy variants: `default`, `matched` (`ftol` 1e-4, `gtol`
1e-5, `maxcor` 10, `maxls` 20 — LBFGSpp-like) and `tight` (`ftol` 1e-12).

### 4. The GPU backend is far slower than the CPU backend

Same driver, same kernels, dispatched as SPIR-V instead of as the `slangc
-target cpp` emits. Iterations are identical to the CPU driver on 13 of the 16
problem-and-setting pairs; three differ by one iteration or by extra
line-search evaluations, and every one of those lands on the same objective —
the GPU is a different instruction stream, not a different algorithm.

dc:

| problem | iters cpu/gpu | nfev cpu/gpu | cpu ms | gpu ms | LBFGSpp ms | gpu/cpu | gpu f | cpu f |
|---|---|---|---|---|---|---|---|---|
| rosen_n2 | 19/19 | 25/25 | 0.060 | 34.296 | 0.101 | 575x | 0.050636828 | 0.050625587 |
| rosen_n10 | 11/11 | 12/12 | 0.042 | 32.106 | 0.050 | 764x | 32.798661 | 32.798662 |
| rosen_n100 | 12/12 | 13/13 | 0.445 | 69.918 | 0.095 | 157x | 122.10495 | 122.10495 |
| rosenbox_upstream_n25 | 10/10 | 11/31 | 0.067 | 30.057 | 0.051 | 447x | 360.28461 | 360.28461 |
| boxqp_n1000 | 5/5 | 8/8 | 1.854 | 99.586 | 0.460 | 54x | -1932.8821 | -1932.8821 |
| rosen_n1000 | 11/11 | 12/12 | 3.577 | 439.207 | 0.627 | 123x | 1013.0395 | 1013.0395 |
| boxqp_n10000 | 5/5 | 11/11 | 21.614 | 991.959 | 4.520 | 46x | -19315.055 | -19315.055 |
| logreg_n1000_m4000 | 10/10 | 11/11 | 49.480 | 753.469 | 47.721 | 15x | 0.15953927 | 0.15953927 |

tight:

| problem | iters cpu/gpu | nfev cpu/gpu | cpu ms | gpu ms | LBFGSpp ms | gpu/cpu | gpu f | cpu f |
|---|---|---|---|---|---|---|---|---|
| rosen_n2 | 21/21 | 27/27 | 0.051 | 40.468 | 0.087 | 795x | 0.050596091 | 0.050596091 |
| rosen_n10 | 40/40 | 45/45 | 0.190 | 108.341 | 0.192 | 570x | 21.943057 | 21.943057 |
| rosen_n100 | 43/42 | 49/48 | 1.801 | 408.921 | 0.471 | 227x | 118.81112 | 118.81112 |
| rosenbox_upstream_n25 | 13/13 | 14/34 | 0.088 | 44.377 | 0.062 | 506x | 360.28359 | 360.28359 |
| boxqp_n1000 | 11/11 | 14/14 | 3.960 | 436.250 | 0.855 | 110x | -1932.9854 | -1932.9854 |
| rosen_n1000 | 43/43 | 49/49 | 18.238 | 3423.221 | 3.353 | 188x | 1009.7183 | 1009.7183 |
| boxqp_n10000 | 11/11 | 17/17 | 42.869 | 4223.225 | 7.975 | 99x | -19316.068 | -19316.068 |
| logreg_n1000_m4000 | 17/17 | 18/18 | 81.147 | 1883.021 | 76.276 | 23x | 0.15750526 | 0.15750526 |

The GPU is 15x to 795x the CPU driver's time, and it loses to LBFGSpp by 9x to
440x. The next section says why.

### 5. Where the GPU time goes, and why vectorizing is not the lever

`./bench --micro <spv dir> <n> <dispatches>` puts N identical dispatches in one
phase, so the submit cost is amortised and each kernel's own cost is visible. At
n=1000, with an empty submit+fence of 24.1 us (`results/micro.txt`):

| kernel | threads | CPU us/dispatch | GPU us/dispatch | GPU/CPU |
|---|---|---|---|---|
| box_project | 256 | 0.10 | 6.26 | 63x |
| saxpby | 256 | 0.07 | 6.78 | 97x |
| ring_store | 256 | 0.30 | 5.87 | 20x |
| pg_inf | 256 | 0.17 | 8.60 | 51x |
| step_max | 256 | 1.30 | 8.71 | 7x |
| multi_dot (10 columns) | 256 per column | 64.58 | 33.55 | **0.5x** |
| compact | 1 | 0.01 | 6.25 | 625x |
| **cauchy** | **1** | **16.92** | **3743.58** | **221x** |
| **subspace (maxit 10)** | **1** | **0.06** | **161.49** | **2691x** |

The picture is not "the kernels are slow on the GPU":

* **Every parallel kernel is 6-9 us, and the submit floor is 24 us per phase.**
  They are already at the floor. A float4 load would not move them; the driver's
  phase structure — one submit per phase, by design, as `rd_compute.h` argues —
  is what sets that floor.
* **`multi_dot` is the one kernel the GPU wins** — 33.6 us against the CPU's
  64.6 us, because it is the one with ten independent 256-thread reductions.
  That is parallelism, not vector width.
* **Four kernels are single-threaded** (`[numthreads(1,1,1)]`): `compact`,
  `apply_m`, and — fatally — `cauchy` and `subspace`. `cauchy` is the
  breakpoint sweep and sort of the generalized Cauchy point; at n=1000 it costs
  3744 us on the GPU against 17 us on the CPU, a 221x penalty for putting a
  serial O(n log n)-ish sweep on one GPU thread. It runs once per iteration, so
  it alone accounts for the GPU column above: 11 iterations x 3.7 ms is 41 ms
  of `boxqp_n1000`'s 100 ms, and it is why `boxqp_n10000` takes 4.2 s.

So, to the question "should lean-shader-slang be vectorized": **no, not for
this.** Vectorizing the memory-bound kernels would touch 6-9 us dispatches
sitting under a 24 us submit floor. The lever is the two sequential kernels —
parallelize the Cauchy breakpoint sweep and the subspace gather/scatter, or run
them on the CPU and keep the GPU for the parallel ones. `multi_dot` already
shows what the GPU does when a kernel has real parallelism to give it.

### 6. The same counts on another compiler and platform

The `bench` workflow runs `run.sh` on ubuntu-24.04 with Ubuntu clang 18.1.3
(x86_64) and uploads its results as an artifact. It reports the same 16/16
exact iteration and evaluation match against LBFGSpp, and the same final
objectives to the last printed digit as the Apple clang 21 / arm64 run above:
the emitted kernels' arithmetic does not depend on the compiler or the
platform. The Azure runner has neither `slangc` nor a Vulkan device, so its GPU
pass is skipped and its `results/` carry the CPU tables only; its wall times are
1.2x-2.1x the M2 Pro's (fewer repetitions, shared hardware).

## The gate

`check.py` fails the run if either C++ solver stops without converging, if the
driver's iteration count leaves `max(2, 25%)` of LBFGSpp's (the tolerance
contract-lbfgsb's own Gate 5 G2 applies), if the final objective differs by
more than 1e-3 relative, if the driver's projected gradient is both more than
4x LBFGSpp's and above 1e-4, or if fewer than half of the problem/setting pairs
match LBFGSpp's iteration *and* evaluation counts exactly.

Where the GPU backend ran, it is held to the same tolerance against the CPU
driver rather than to bit-equality — the same algorithm in a different
instruction stream — and the number of exact iteration-and-evaluation matches
is reported. Three of the sixteen differ, as the tables above show. The gate
does catch a broken backend: fed the stale-descriptor-set symptom (`driver-vk`
converging in one iteration with f = 0), it fails on both the iteration count
and the objective.

## What this does not measure

* **The guest path itself.** `vec_rd.cpp` reaches the same SPIR-V through
  Godot's RenderingDevice across the sandbox boundary, and `rd_compute.h`
  measures a submit+fence there at ~245 us. The host Vulkan numbers here are
  therefore a *lower bound* on the guest's GPU cost, not an estimate of it.
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
