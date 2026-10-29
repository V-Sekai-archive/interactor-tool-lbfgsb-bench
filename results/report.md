## Environment

```
host: Darwin 25.6.0 arm64
cpu: Apple M2 Pro
compiler: Apple clang version 21.0.0 (clang-2100.1.1.101)
python3: Python 3.9.6
numpy: 2.0.2
scipy: 1.13.1
LBFGSpp: 0.3.0 (interactor-aria-lbfgspp thirdparty/LBFGSpp, 15e37f49bafd1b6908170f615002f1d638103c9d)
Eigen: interactor-aria-lbfgspp thirdparty/eigen, 15e37f49bafd1b6908170f615002f1d638103c9d
contract-lbfgsb: 5fe35817f02b4ad7bc552556295d4636af65cea6
contract-guest-runtime: 1d8341825ac6fcfa8b8b2a11ae908e5d6aa71b33
```

### Iterations and objective evaluations -- delta = 1e-3 (the repo's diffcloth parameters)

| problem | n | driver iters | driver nfev | LBFGSpp iters | LBFGSpp nfev | driver f | LBFGSpp f | driver pg | LBFGSpp pg |
|---|---|---|---|---|---|---|---|---|---|
| rosen_n2 | 2 | 19 | 25 | 19 | 25 | 0.05062558722 | 0.05060453614 | 0.169 | 0.0903 |
| rosen_n10 | 10 | 11 | 12 | 11 | 12 | 32.79866173 | 32.79866166 | 1.96 | 1.96 |
| rosen_n100 | 100 | 12 | 13 | 12 | 13 | 122.104946 | 122.104946 | 0.642 | 0.642 |
| rosenbox_upstream_n25 | 25 | 10 | 11 | 10 | 11 | 360.284608 | 360.2846081 | 0.435 | 0.435 |
| boxqp_n1000 | 1000 | 5 | 8 | 5 | 8 | -1932.882068 | -1932.882069 | 0.317 | 0.317 |
| rosen_n1000 | 1000 | 11 | 12 | 11 | 12 | 1013.039503 | 1013.039504 | 1.53 | 1.53 |
| boxqp_n10000 | 10000 | 5 | 11 | 5 | 11 | -19315.05483 | -19315.05514 | 0.351 | 0.351 |
| logreg_n1000_m4000 | 1000 | 10 | 11 | 10 | 11 | 0.1595392702 | 0.1595392731 | 0.000497 | 0.000497 |

### Iterations and objective evaluations -- delta = 1e-10 (the delta test out of the way)

| problem | n | driver iters | driver nfev | LBFGSpp iters | LBFGSpp nfev | driver f | LBFGSpp f | driver pg | LBFGSpp pg |
|---|---|---|---|---|---|---|---|---|---|
| rosen_n2 | 2 | 21 | 27 | 21 | 27 | 0.05059609148 | 0.05059609838 | 3.81e-06 | 3.47e-08 |
| rosen_n10 | 10 | 40 | 45 | 40 | 45 | 21.94305743 | 21.94305917 | 0.000101 | 7.48e-05 |
| rosen_n100 | 100 | 43 | 49 | 43 | 49 | 118.8111182 | 118.8111183 | 0.000273 | 0.000273 |
| rosenbox_upstream_n25 | 25 | 13 | 14 | 13 | 14 | 360.2835856 | 360.2835856 | 1.22e-05 | 7.02e-06 |
| boxqp_n1000 | 1000 | 11 | 14 | 11 | 14 | -1932.985351 | -1932.985351 | 0.000183 | 0.000183 |
| rosen_n1000 | 1000 | 43 | 49 | 43 | 49 | 1009.718305 | 1009.718305 | 0.000692 | 0.000358 |
| boxqp_n10000 | 10000 | 11 | 17 | 11 | 17 | -19316.06836 | -19316.06836 | 0.000186 | 0.000186 |
| logreg_n1000_m4000 | 1000 | 17 | 18 | 17 | 18 | 0.15750526 | 0.1575052603 | 0.000102 | 0.000102 |

### Wall time -- delta 1e-3 vs scipy at LBFGSpp-like options

Median of the timed repetitions, in milliseconds. `solver` excludes the objective calls both programs time separately, so the number is the solver's own cost.

| problem | driver total | LBFGSpp total | scipy total | driver solver | LBFGSpp solver | scipy solver | driver/LBFGSpp | driver/scipy |
|---|---|---|---|---|---|---|---|---|
| rosen_n2 | 0.060 | 0.102 | 0.450 | 0.060 | 0.102 | 0.338 | 0.59x | 0.18x |
| rosen_n10 | 0.042 | 0.046 | 1.535 | 0.041 | 0.046 | 1.132 | 0.90x | 0.04x |
| rosen_n100 | 0.450 | 0.101 | 0.644 | 0.447 | 0.099 | 0.551 | 4.54x | 0.81x |
| rosenbox_upstream_n25 | 0.067 | 0.051 | 0.567 | 0.067 | 0.051 | 0.330 | 1.32x | 0.20x |
| boxqp_n1000 | 1.851 | 0.475 | 2.707 | 1.836 | 0.460 | 2.648 | 4.00x | 0.69x |
| rosen_n1000 | 3.595 | 0.588 | 3.123 | 3.579 | 0.571 | 3.002 | 6.26x | 1.19x |
| boxqp_n10000 | 21.858 | 4.460 | 24.473 | 21.677 | 4.278 | 24.302 | 5.07x | 0.89x |
| logreg_n1000_m4000 | 48.877 | 46.989 | 28.908 | 2.873 | 0.986 | 5.635 | 2.91x | 0.51x |

### Wall time -- delta 1e-10 vs scipy at a near-machine ftol

Median of the timed repetitions, in milliseconds. `solver` excludes the objective calls both programs time separately, so the number is the solver's own cost.

| problem | driver total | LBFGSpp total | scipy total | driver solver | LBFGSpp solver | scipy solver | driver/LBFGSpp | driver/scipy |
|---|---|---|---|---|---|---|---|---|
| rosen_n2 | 0.056 | 0.086 | 0.518 | 0.056 | 0.086 | 0.376 | 0.65x | 0.15x |
| rosen_n10 | 0.180 | 0.193 | 1.687 | 0.179 | 0.193 | 1.243 | 0.93x | 0.14x |
| rosen_n100 | 1.683 | 0.443 | 1.653 | 1.676 | 0.437 | 1.331 | 3.84x | 1.26x |
| rosenbox_upstream_n25 | 0.083 | 0.057 | 0.644 | 0.083 | 0.057 | 0.359 | 1.46x | 0.23x |
| boxqp_n1000 | 3.614 | 0.726 | 3.366 | 3.591 | 0.703 | 3.238 | 5.11x | 1.11x |
| rosen_n1000 | 17.703 | 3.532 | 6.712 | 17.628 | 3.458 | 6.223 | 5.10x | 2.83x |
| boxqp_n10000 | 43.107 | 8.088 | 28.497 | 42.831 | 7.812 | 28.125 | 5.48x | 1.52x |
| logreg_n1000_m4000 | 83.124 | 80.451 | 49.031 | 8.506 | 5.833 | 5.861 | 1.46x | 1.45x |

### Where each solver stopped

| problem | driver f (dc) | scipy f (default) | driver pg | LBFGSpp pg | scipy pg | driver x* err | scipy x* err | driver stop | scipy status |
|---|---|---|---|---|---|---|---|---|---|
| rosen_n2 | 0.05062558722 | 0.05059609838 | 0.169 | 0.0903 | 6.16e-07 | nan | nan | delta | 0 |
| rosen_n10 | 32.79866173 | 21.94305918 | 1.96 | 1.96 | 0.00213 | nan | nan | delta | 0 |
| rosen_n100 | 122.104946 | 118.8111183 | 0.642 | 0.642 | 0.00048 | nan | nan | delta | 0 |
| rosenbox_upstream_n25 | 360.284608 | 360.2835856 | 0.435 | 0.435 | 8.7e-06 | nan | nan | delta | 0 |
| boxqp_n1000 | -1932.882068 | -1932.985351 | 0.317 | 0.317 | 0.000189 | 0.041 | 8.4e-05 | delta | 0 |
| rosen_n1000 | 1013.039503 | 1009.718305 | 1.53 | 1.53 | 0.0065 | nan | nan | delta | 0 |
| boxqp_n10000 | -19315.05483 | -19316.06836 | 0.351 | 0.351 | 0.000192 | 0.045 | 8.8e-05 | delta | 0 |
| logreg_n1000_m4000 | 0.1595392702 | 0.157419961 | 0.000497 | 0.000497 | 5.71e-06 | nan | nan | delta | 0 |

