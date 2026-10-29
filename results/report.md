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
| rosen_n2 | 0.060 | 0.101 | 0.428 | 0.059 | 0.101 | 0.316 | 0.59x | 0.19x |
| rosen_n10 | 0.042 | 0.050 | 1.527 | 0.042 | 0.050 | 1.120 | 0.83x | 0.04x |
| rosen_n100 | 0.445 | 0.095 | 0.664 | 0.443 | 0.093 | 0.570 | 4.78x | 0.78x |
| rosenbox_upstream_n25 | 0.067 | 0.051 | 0.551 | 0.067 | 0.051 | 0.310 | 1.31x | 0.22x |
| boxqp_n1000 | 1.854 | 0.460 | 2.928 | 1.840 | 0.445 | 2.871 | 4.13x | 0.64x |
| rosen_n1000 | 3.577 | 0.627 | 3.188 | 3.560 | 0.610 | 3.071 | 5.83x | 1.16x |
| boxqp_n10000 | 21.614 | 4.520 | 24.642 | 21.281 | 4.187 | 24.478 | 5.08x | 0.87x |
| logreg_n1000_m4000 | 49.480 | 47.721 | 28.800 | 3.780 | 2.022 | 5.600 | 1.87x | 0.68x |

### Wall time -- delta 1e-10 vs scipy at a near-machine ftol

Median of the timed repetitions, in milliseconds. `solver` excludes the objective calls both programs time separately, so the number is the solver's own cost.

| problem | driver total | LBFGSpp total | scipy total | driver solver | LBFGSpp solver | scipy solver | driver/LBFGSpp | driver/scipy |
|---|---|---|---|---|---|---|---|---|
| rosen_n2 | 0.051 | 0.087 | 0.532 | 0.051 | 0.086 | 0.393 | 0.59x | 0.13x |
| rosen_n10 | 0.190 | 0.192 | 1.659 | 0.189 | 0.191 | 1.227 | 0.99x | 0.15x |
| rosen_n100 | 1.801 | 0.471 | 1.654 | 1.794 | 0.465 | 1.340 | 3.86x | 1.34x |
| rosenbox_upstream_n25 | 0.088 | 0.062 | 0.624 | 0.087 | 0.061 | 0.349 | 1.42x | 0.25x |
| boxqp_n1000 | 3.960 | 0.855 | 3.366 | 3.936 | 0.831 | 3.241 | 4.74x | 1.21x |
| rosen_n1000 | 18.238 | 3.353 | 6.747 | 18.173 | 3.288 | 6.285 | 5.53x | 2.89x |
| boxqp_n10000 | 42.869 | 7.975 | 30.285 | 42.611 | 7.716 | 29.923 | 5.52x | 1.42x |
| logreg_n1000_m4000 | 81.147 | 76.276 | 57.807 | 6.702 | 1.832 | 14.260 | 3.66x | 0.47x |

### The GPU backend -- delta 1e-3

The same driver and the same kernels, dispatched as SPIR-V through Vulkan instead of as the `slangc -target cpp` emits. Both backends must do the same iterations; the evaluation count can differ when a line-search step falls differently under the GPU's rounding.

| problem | iters cpu | iters gpu | nfev cpu / gpu | cpu ms | gpu ms | LBFGSpp ms | gpu/cpu | gpu f | cpu f |
|---|---|---|---|---|---|---|---|---|---|
| rosen_n2 | 19 | 19 | 25 / 25 | 0.060 | 34.296 | 0.101 | 575.4x | 0.05063682831 | 0.05062558722 |
| rosen_n10 | 11 | 11 | 12 / 12 | 0.042 | 32.106 | 0.050 | 764.4x | 32.79866133 | 32.79866173 |
| rosen_n100 | 12 | 12 | 13 / 13 | 0.445 | 69.918 | 0.095 | 157.1x | 122.1049459 | 122.104946 |
| rosenbox_upstream_n25 | 10 | 10 | 11 / 31 | 0.067 | 30.057 | 0.051 | 447.3x | 360.2846081 | 360.284608 |
| boxqp_n1000 | 5 | 5 | 8 / 8 | 1.854 | 99.586 | 0.460 | 53.7x | -1932.882067 | -1932.882068 |
| rosen_n1000 | 11 | 11 | 12 / 12 | 3.577 | 439.207 | 0.627 | 122.8x | 1013.039503 | 1013.039503 |
| boxqp_n10000 | 5 | 5 | 11 / 11 | 21.614 | 991.959 | 4.520 | 45.9x | -19315.05481 | -19315.05483 |
| logreg_n1000_m4000 | 10 | 10 | 11 / 11 | 49.480 | 753.469 | 47.721 | 15.2x | 0.1595392717 | 0.1595392702 |

### The GPU backend -- delta 1e-10

The same driver and the same kernels, dispatched as SPIR-V through Vulkan instead of as the `slangc -target cpp` emits. Both backends must do the same iterations; the evaluation count can differ when a line-search step falls differently under the GPU's rounding.

| problem | iters cpu | iters gpu | nfev cpu / gpu | cpu ms | gpu ms | LBFGSpp ms | gpu/cpu | gpu f | cpu f |
|---|---|---|---|---|---|---|---|---|---|
| rosen_n2 | 21 | 21 | 27 / 27 | 0.051 | 40.468 | 0.087 | 795.1x | 0.05059609148 | 0.05059609148 |
| rosen_n10 | 40 | 40 | 45 / 45 | 0.190 | 108.341 | 0.192 | 569.9x | 21.94305743 | 21.94305743 |
| rosen_n100 | 43 | 42 | 49 / 48 | 1.801 | 408.921 | 0.471 | 227.1x | 118.8111182 | 118.8111182 |
| rosenbox_upstream_n25 | 13 | 13 | 14 / 34 | 0.088 | 44.377 | 0.062 | 506.0x | 360.2835856 | 360.2835856 |
| boxqp_n1000 | 11 | 11 | 14 / 14 | 3.960 | 436.250 | 0.855 | 110.2x | -1932.985351 | -1932.985351 |
| rosen_n1000 | 43 | 43 | 49 / 49 | 18.238 | 3423.221 | 3.353 | 187.7x | 1009.718305 | 1009.718305 |
| boxqp_n10000 | 11 | 11 | 17 / 17 | 42.869 | 4223.225 | 7.975 | 98.5x | -19316.06836 | -19316.06836 |
| logreg_n1000_m4000 | 17 | 17 | 18 / 18 | 81.147 | 1883.021 | 76.276 | 23.2x | 0.1575052602 | 0.15750526 |

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

