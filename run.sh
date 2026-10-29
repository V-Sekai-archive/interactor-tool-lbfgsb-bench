#!/bin/sh
# Build and run the L-BFGS-B benchmark: contract-lbfgsb's Lean-emitted driver
# against LBFGSpp 0.3.0 (the solver it mirrors) and scipy's Fortran L-BFGS-B.
#
#   ./run.sh              # clones the pinned dependencies into .deps/, runs
#                         # everything, writes results/
#   CXX=clang++ ./run.sh
#   DEPS=/somewhere ./run.sh
#
# Environment: CXX, PY, DEPS, OUT, REPS_SMALL, REPS_BIG.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

DEPS=${DEPS:-$HERE/.deps}
OUT=${OUT:-$HERE/results}
CXX=${CXX:-c++}
PY=${PY:-python3}
REPS_SMALL=${REPS_SMALL:-20}
REPS_BIG=${REPS_BIG:-5}

# The revisions the results in results/ were measured at.
CONTRACT_LBFGSB_SHA=5fe35817f02b4ad7bc552556295d4636af65cea6
ARIA_LBFGSPP_SHA=15e37f49bafd1b6908170f615002f1d638103c9d
GUEST_RUNTIME_SHA=1d8341825ac6fcfa8b8b2a11ae908e5d6aa71b33

fetch() { # url sha dir
	if [ -d "$3/.git" ]; then
		return 0
	fi
	mkdir -p "$3"
	git -C "$3" init -q
	git -C "$3" remote add origin "$1"
	git -C "$3" fetch -q --depth 1 origin "$2"
	git -C "$3" checkout -q FETCH_HEAD
}

mkdir -p "$DEPS" "$OUT" "$OUT/dc" "$OUT/tight"
fetch https://github.com/V-Sekai-fire/contract-lbfgsb.git "$CONTRACT_LBFGSB_SHA" "$DEPS/contract-lbfgsb"
fetch https://github.com/V-Sekai-fire/interactor-aria-lbfgspp.git "$ARIA_LBFGSPP_SHA" "$DEPS/interactor-aria-lbfgspp"
fetch https://github.com/V-Sekai-fire/contract-guest-runtime.git "$GUEST_RUNTIME_SHA" "$DEPS/contract-guest-runtime"

{
	echo "host: $(uname -srm)"
	if [ "$(uname -s)" = "Darwin" ]; then echo "cpu: $(sysctl -n machdep.cpu.brand_string)"; fi
	echo "compiler: $($CXX --version | head -1)"
	echo "$PY: $($PY --version 2>&1)"
	echo "numpy: $($PY -c 'import numpy; print(numpy.__version__)')"
	echo "scipy: $($PY -c 'import scipy; print(scipy.__version__)')"
	echo "LBFGSpp: 0.3.0 (interactor-aria-lbfgspp thirdparty/LBFGSpp, $ARIA_LBFGSPP_SHA)"
	echo "Eigen: interactor-aria-lbfgspp thirdparty/eigen, $ARIA_LBFGSPP_SHA"
	echo "contract-lbfgsb: $CONTRACT_LBFGSB_SHA"
	echo "contract-guest-runtime: $GUEST_RUNTIME_SHA"
} > "$OUT/environment.txt"

INC="-I$DEPS/interactor-aria-lbfgspp/thirdparty/LBFGSpp/include"
INC="$INC -I$DEPS/interactor-aria-lbfgspp/thirdparty/eigen"
INC="$INC -I$DEPS/contract-guest-runtime/guest/avbd/slang-rt"

# The GPU path needs the Vulkan headers and loader. On macOS Homebrew keeps
# them outside the default search path; on Linux the system ones are found.
VKINC=""
if [ -f /opt/homebrew/include/vulkan/vulkan.h ]; then
	VKINC="-I/opt/homebrew/include -L/opt/homebrew/lib"
fi
VK="-lvulkan"

# The kernels as SPIR-V, from the .slang that contract-lbfgsb commits (its
# gen.sh's spirv step). Without slangc the benchmark runs the CPU backend only.
SPV=""
if command -v "${SLANGC:-slangc}" >/dev/null 2>&1; then
	SPV="$OUT/spv"
	mkdir -p "$SPV"
	for k in lb_box_project lb_pg_inf lb_step_max lb_multi_dot lb_compact lb_ring_store lb_apply_m lb_cauchy \
		lb_subspace saxpby; do
		"${SLANGC:-slangc}" -target spirv -profile sm_6_5 -stage compute -entry main -fp-mode precise \
			-o "$SPV/$k.spv" "$DEPS/contract-lbfgsb/kernels/drape/slang/$k.slang"
	done
else
	echo "no slangc: the GPU backend is skipped"
fi

# -ffp-contract=off: the guest builds these same files with it (contract-lbfgsb
# tests/lbfgsb_driver/build.sh), so the CPU results here match the guest's.
$CXX -O2 -std=c++17 -ffp-contract=off -DNDEBUG -Wall -Wno-unused-function -Wno-unused-lambda-capture \
	$INC $VKINC \
	-DGUEST_LBFGSB_H="\"$DEPS/contract-lbfgsb/guest/drape/lbfgsb.h\"" \
	-DGUEST_VEC_CPU_H="\"$DEPS/contract-lbfgsb/guest/drape/vec_cpu.h\"" \
	bench.cpp vec_vk.cpp \
	"$DEPS/contract-lbfgsb/guest/drape/lbfgsb.cpp" \
	"$DEPS/contract-lbfgsb/guest/drape/vec_cpu.cpp" \
	$VK -o bench

$PY gen_data.py "$OUT/dc" 1000 4000
$PY gen_data.py "$OUT/tight" 1000 4000
# The CPU pass: the driver against LBFGSpp, in its own process so neither
# disturbs the other's timing.
# delta 1e-3: the parameters cloth-dynamics' BackwardTaskSolver uses.
./bench "$OUT/dc" "$REPS_SMALL" "$REPS_BIG" 1e-3 > "$OUT/dc/bench.log"
# delta 1e-10: the same solvers with the function-difference test out of the way.
./bench "$OUT/tight" "$REPS_SMALL" "$REPS_BIG" 1e-10 > "$OUT/tight/bench.log"
# The GPU pass, into its own file, where there is a device.
if [ -n "$SPV" ]; then
	if ./bench "$OUT/dc" "$REPS_SMALL" "$REPS_BIG" 1e-3 "$SPV" --vk > "$OUT/dc/bench_vk.log" 2>&1; then
		./bench "$OUT/tight" "$REPS_SMALL" "$REPS_BIG" 1e-10 "$SPV" --vk > "$OUT/tight/bench_vk.log" 2>&1 || true
	else
		echo "no GPU device: the GPU pass is skipped"
		rm -f "$OUT/dc/cpp_results_vk.csv" "$OUT/tight/cpp_results_vk.csv"
	fi
	# Per-kernel cost, CPU against GPU, with the submit amortised.
	./bench --micro "$SPV" 1000 50 > "$OUT/micro.txt" 2>&1 || true
fi

for variant in default matched tight; do
	$PY scipy_bench.py "$OUT/dc" "$REPS_SMALL" "$REPS_BIG" "$variant" > "$OUT/scipy_$variant.log"
done

$PY report.py "$OUT" > "$OUT/report.md"

# The gate. Run it without a pipe: a pipeline's status is the last command's,
# which would hide a failing check.
if ! $PY check.py "$OUT" > "$OUT/check.log" 2>&1; then
	cat "$OUT/check.log"
	exit 1
fi
cat "$OUT/check.log"

echo "results in $OUT (report: $OUT/report.md)"
