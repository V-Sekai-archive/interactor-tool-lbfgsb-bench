// bench.cpp -- contract-lbfgsb's Lean-emitted L-BFGS-B (guest/drape/lbfgsb.cpp
// over vec_cpu.cpp) vs LBFGSpp 0.3.0's own LBFGSBSolver, on the five problems
// the repository's oracle generator defines (tests/lbfgsb_oracle/gen.cpp).
// Same problems, same parameters, same objective code (double) for both.
//
//   bench <outdir> [reps_small] [reps_big]
//
// Writes <outdir>/problems/*.txt (for the scipy comparison) and prints a CSV
// to stdout plus <outdir>/cpp_results.csv.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <LBFGSB.h>

// The guest's own headers are reached by an explicit path, never through -I:
// on a case-insensitive filesystem guest/drape/lbfgsb.h shadows LBFGSpp's
// LBFGSB.h when both directories are on the include path. Override the two
// macros to point at another checkout.
#ifndef GUEST_LBFGSB_H
#define GUEST_LBFGSB_H ".deps/contract-lbfgsb/guest/drape/lbfgsb.h"
#endif
#ifndef GUEST_VEC_CPU_H
#define GUEST_VEC_CPU_H ".deps/contract-lbfgsb/guest/drape/vec_cpu.h"
#endif
#include GUEST_LBFGSB_H
#include GUEST_VEC_CPU_H

using Vec = Eigen::VectorXd;
using Clock = std::chrono::steady_clock;
static const double INF = std::numeric_limits<double>::infinity();

// ---------------------------------------------------------------- problems
// Verbatim from tests/lbfgsb_oracle/gen.cpp (MIT), so both solvers face the
// exact problems the repository's own gate is defined on.
struct Problem {
	std::string name;
	int n;
	std::vector<double> lb, ub, x0, xstar;
	std::vector<double> a_diag, a_b;  // boxqp only
	double a_off = 0.0;
	double lr_lambda = 0.0;  // logreg only
	std::function<double(const double *, double *)> fg;
};

static Problem make_rosen(int n) {
	Problem p;
	p.name = "rosen_n" + std::to_string(n);
	p.n = n;
	p.lb.assign(n, 0.0);
	p.ub.assign(n, 0.0);
	p.x0.assign(n, 0.0);
	for (int i = 0; i < n; i++) {
		p.lb[i] = -1.5;
		p.ub[i] = 1.5;
		if (i % 3 == 1) p.ub[i] = 0.6;
		if (n >= 10 && i % 5 == 4) { p.lb[i] = -INF; p.ub[i] = INF; }
		p.x0[i] = (i % 2 == 0) ? -1.2 : 1.0;
	}
	if (n >= 10) p.lb[n / 2] = p.ub[n / 2] = 0.5;
	p.fg = [n](const double *x, double *g) {
		double f = 0.0;
		for (int i = 0; i < n; i++) g[i] = 0.0;
		for (int i = 0; i + 1 < n; i++) {
			const double a = x[i + 1] - x[i] * x[i], b = 1.0 - x[i];
			f += 100.0 * a * a + b * b;
			g[i] += -400.0 * x[i] * a - 2.0 * b;
			g[i + 1] += 200.0 * a;
		}
		return f;
	};
	return p;
}

static Problem make_upstream() {
	Problem p;
	const int n = 25;
	p.name = "rosenbox_upstream_n25";
	p.n = n;
	p.lb.assign(n, 2.0);
	p.ub.assign(n, 4.0);
	p.lb[2] = -INF;
	p.ub[2] = INF;
	p.x0.assign(n, 3.0);
	p.x0[0] = p.x0[1] = 2.0;
	p.x0[5] = p.x0[7] = 4.0;
	p.fg = [n](const double *x, double *grad) {
		double fx = (x[0] - 1.0) * (x[0] - 1.0);
		grad[0] = 2 * (x[0] - 1) + 16 * (x[0] * x[0] - x[1]) * x[0];
		for (int i = 1; i < n; i++) {
			fx += 4 * std::pow(x[i] - x[i - 1] * x[i - 1], 2);
			if (i == n - 1) grad[i] = 8 * (x[i] - x[i - 1] * x[i - 1]);
			else grad[i] = 8 * (x[i] - x[i - 1] * x[i - 1]) + 16 * (x[i] * x[i] - x[i + 1]) * x[i];
		}
		return fx;
	};
	return p;
}

static void qp_Ax(const std::vector<double> &d, double off, const double *x, int n, double *y) {
	for (int i = 0; i < n; i++) {
		double v = d[i] * x[i];
		if (i > 0) v += off * x[i - 1];
		if (i + 1 < n) v += off * x[i + 1];
		y[i] = v;
	}
}

static Problem make_qp(int n) {
	Problem p;
	p.name = "boxqp_n" + std::to_string(n);
	p.n = n;
	p.lb.assign(n, -1.0);
	p.ub.assign(n, 1.0);
	p.x0.assign(n, 0.0);
	std::vector<double> d(n), xs(n), gs(n);
	for (int i = 0; i < n; i++) {
		d[i] = 2.5 + 0.5 * ((i * 37) % 11);
		const int r = i % 4;
		if (r == 0) { xs[i] = -1.0; gs[i] = 0.25 + 0.25 * (i % 5); }
		else if (r == 1) { xs[i] = 1.0; gs[i] = -(0.25 + 0.25 * (i % 3)); }
		else { xs[i] = (((i * 13) % 17) - 8) / 16.0; gs[i] = 0.0; }
	}
	std::vector<double> b(n), ax(n);
	qp_Ax(d, -1.0, xs.data(), n, ax.data());
	for (int i = 0; i < n; i++) b[i] = ax[i] - gs[i];
	p.xstar = xs;
	p.a_diag = d;
	p.a_b = b;
	p.a_off = -1.0;
	auto dp = std::make_shared<std::vector<double>>(d);
	auto bp = std::make_shared<std::vector<double>>(b);
	p.fg = [n, dp, bp](const double *x, double *g) {
		std::vector<double> ax(n);
		qp_Ax(*dp, -1.0, x, n, ax.data());
		double f = 0.0;
		for (int i = 0; i < n; i++) {
			g[i] = ax[i] - (*bp)[i];
			f += 0.5 * x[i] * ax[i] - (*bp)[i] * x[i];
		}
		return f;
	};
	return p;
}

// Bound-constrained L2 logistic regression: a real L-BFGS-B workload.
// f(w) = (1/m) sum_i softplus(-y_i x_i.w) + 0.5 lambda |w|^2, lb/ub = -1/1.
// Data comes from a raw float64 file written by gen_data.py, so both the C++
// and the scipy run read identical numbers.
static Problem make_logreg(int n, int m, double lambda, const std::string &xpath, const std::string &ypath) {
	Problem p;
	p.name = "logreg_n" + std::to_string(n) + "_m" + std::to_string(m);
	p.n = n;
	auto X = std::make_shared<std::vector<double>>(size_t(n) * m, 0.0);
	auto y = std::make_shared<std::vector<double>>(m, 0.0);
	FILE *fx = std::fopen(xpath.c_str(), "rb");
	FILE *fy = std::fopen(ypath.c_str(), "rb");
	if (!fx || !fy) {
		std::fprintf(stderr, "missing logistic data %s / %s\n", xpath.c_str(), ypath.c_str());
		std::exit(3);
	}
	if (std::fread(X->data(), sizeof(double), X->size(), fx) != X->size()) std::exit(3);
	if (std::fread(y->data(), sizeof(double), y->size(), fy) != y->size()) std::exit(3);
	std::fclose(fx);
	std::fclose(fy);
	p.lb.assign(n, -1.0);
	p.ub.assign(n, 1.0);
	p.x0.assign(n, 0.0);
	p.fg = [n, m, lambda, X, y](const double *w, double *g) {
		double f = 0.0;
		for (int j = 0; j < n; j++) g[j] = 0.0;
		for (int i = 0; i < m; i++) {
			const double *xi = &(*X)[size_t(i) * n];
			double z = 0.0;
			for (int j = 0; j < n; j++) z += xi[j] * w[j];
			const double t = (*y)[i] * z;                // want softplus(-t)
			const double s = 1.0 / (1.0 + std::exp(t));   // sigmoid(-t)
			f += (t > 0.0) ? std::log1p(std::exp(-t)) : (-t + std::log1p(std::exp(t)));
			const double c = -(*y)[i] * s;
			for (int j = 0; j < n; j++) g[j] += c * xi[j];
		}
		const double inv = 1.0 / m;
		f *= inv;
		for (int j = 0; j < n; j++) {
			f += 0.5 * lambda * w[j] * w[j];
			g[j] = g[j] * inv + lambda * w[j];
		}
		return f;
	};
	p.lr_lambda = lambda;
	return p;
}

static double proj_grad_norm(const std::vector<double> &x, const std::vector<double> &g,
							 const std::vector<double> &lb, const std::vector<double> &ub) {
	double mx = 0.0;
	for (size_t i = 0; i < x.size(); i++) {
		double v;
		if (x[i] <= lb[i]) v = std::min(0.0, g[i]);
		else if (x[i] >= ub[i]) v = std::max(0.0, g[i]);
		else v = g[i];
		mx = std::max(mx, std::fabs(v));
	}
	return mx;
}

// ---------------------------------------------------------------- results
struct Out {
	bool ok = false;
	int niter = 0, nfev = 0;
	double f = NAN, pg = NAN, xstar_err = NAN;
	double ms_med = NAN, ms_min = NAN, ms_max = NAN;
	std::string note;
	std::vector<double> x;
};

static double median(std::vector<double> v) {
	std::sort(v.begin(), v.end());
	return v[v.size() / 2];
}

// --------------------------------------------------- solver A: the driver
// guest/drape/lbfgsb.cpp over vec_cpu.cpp: the Lean-emitted kernels, float32
// vectors, double scalars, reverse communication.
static Out run_driver(const Problem &p, int m, double delta, int maxit, int reps) {
	Out o;
	std::vector<double> times;
	for (int r = 0; r < reps + 1; r++) {
		const int n = p.n;
		std::vector<float> x0(n), lb(n), ub(n);
		for (int i = 0; i < n; i++) {
			x0[i] = (float)p.x0[i];
			lb[i] = std::isinf(p.lb[i]) ? (p.lb[i] > 0 ? INFINITY : -INFINITY) : (float)p.lb[i];
			ub[i] = std::isinf(p.ub[i]) ? (p.ub[i] > 0 ? INFINITY : -INFINITY) : (float)p.ub[i];
		}
		VecCpu v;
		Lbfgsb s(v);
		LbfgsbParams pr;
		pr.m = m;
		pr.delta = delta;
		pr.max_linesearch = 20;
		pr.max_iterations = maxit;
		int nfev = 0;
		const auto t0 = Clock::now();
		Lbfgsb::Status st = s.start((uint32_t)n, x0.data(), lb.data(), ub.data(), pr);
		while (true) {
			if (st == Lbfgsb::NEED_EVAL || st == Lbfgsb::TRY) {
				std::vector<float> xf;
				if (!s.readX(xf)) { o.note = "readX failed"; break; }
				std::vector<double> xd(n), gd(n);
				for (int i = 0; i < n; i++) xd[i] = xf[i];
				const double f = p.fg(xd.data(), gd.data());
				++nfev;
				std::vector<float> gf(n);
				for (int i = 0; i < n; i++) gf[i] = (float)gd[i];
				if (!s.setGradient(gf.data())) { o.note = "setGradient failed"; break; }
				st = s.next(f);
			} else if (st == Lbfgsb::ACCEPT || st == Lbfgsb::BUSY) {
				st = s.next();
			} else {
				break;
			}
		}
		const auto t1 = Clock::now();
		const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
		if (r > 0) times.push_back(ms);
		if (r == reps) {
			o.ok = (st == Lbfgsb::CONVERGED);
			o.note = Lbfgsb::status_name(st) + std::string(":") + (st == Lbfgsb::CONVERGED ? s.reason() : s.error());
			o.niter = s.iterations();
			o.nfev = nfev;
			o.f = s.fx();
			o.pg = s.pgNorm();
			std::vector<float> xf;
			s.readX(xf);
			o.x.assign(n, 0.0);
			for (int i = 0; i < n; i++) o.x[i] = xf[i];
		}
	}
	o.ms_med = median(times);
	o.ms_min = *std::min_element(times.begin(), times.end());
	o.ms_max = *std::max_element(times.begin(), times.end());
	if (!p.xstar.empty()) {
		double e = 0.0;
		for (int i = 0; i < p.n; i++) e = std::max(e, std::fabs(o.x[i] - p.xstar[i]));
		o.xstar_err = e;
	}
	return o;
}

// ------------------------------------------- solver B: LBFGSpp's own solver
static Out run_lbfgspp(const Problem &p, int m, double delta, int maxit, int reps) {
	Out o;
	std::vector<double> times;
	for (int r = 0; r < reps + 1; r++) {
		LBFGSpp::LBFGSBParam<double> prm;
		prm.m = m;
		prm.delta = delta;
		prm.max_linesearch = 20;
		prm.max_iterations = maxit;
		LBFGSpp::LBFGSBSolver<double> solver(prm);
		Vec x = Eigen::Map<const Vec>(p.x0.data(), p.n);
		const Vec lb = Eigen::Map<const Vec>(p.lb.data(), p.n);
		const Vec ub = Eigen::Map<const Vec>(p.ub.data(), p.n);
		double fx = 0.0;
		int nfev = 0;
		std::vector<double> xd(p.n), gd(p.n);
		auto fun = [&](const Vec &xv, Vec &gv) {
			++nfev;
			for (int i = 0; i < p.n; i++) xd[i] = xv[i];
			const double f = p.fg(xd.data(), gd.data());
			for (int i = 0; i < p.n; i++) gv[i] = gd[i];
			return f;
		};
		int ret = -1;
		std::string note = "ok";
		const auto t0 = Clock::now();
		try {
			ret = solver.minimize(fun, x, fx, lb, ub);
		} catch (const std::exception &e) {
			note = std::string("exception: ") + e.what();
		}
		const auto t1 = Clock::now();
		const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
		if (r > 0) times.push_back(ms);
		if (r == reps) {
			o.ok = (note == "ok");
			o.note = note == "ok" ? ("CONVERGED:" + std::to_string(ret)) : note;
			o.niter = ret;
			o.nfev = nfev;
			o.f = fx;
			o.pg = solver.final_grad_norm();
			o.x.assign(p.n, 0.0);
			for (int i = 0; i < p.n; i++) o.x[i] = x[i];
		}
	}
	o.ms_med = median(times);
	o.ms_min = *std::min_element(times.begin(), times.end());
	o.ms_max = *std::max_element(times.begin(), times.end());
	if (!p.xstar.empty()) {
		double e = 0.0;
		for (int i = 0; i < p.n; i++) e = std::max(e, std::fabs(o.x[i] - p.xstar[i]));
		o.xstar_err = e;
	}
	return o;
}

// ---------------------------------------------------------------- output
static void dump_problem(const std::string &dir, const Problem &p) {
	const std::string path = dir + "/" + p.name + ".txt";
	FILE *f = std::fopen(path.c_str(), "w");
	std::fprintf(f, "n %d\n", p.n);
	auto vec = [&](const char *k, const std::vector<double> &v) {
		std::fprintf(f, "%s", k);
		for (double x : v) {
			if (std::isinf(x)) std::fprintf(f, " %s", x > 0 ? "inf" : "-inf");
			else std::fprintf(f, " %.17g", x);
		}
		std::fputc('\n', f);
	};
	vec("lb", p.lb);
	vec("ub", p.ub);
	vec("x0", p.x0);
	if (p.name.rfind("boxqp", 0) == 0) {
		vec("A_diag", p.a_diag);
		std::fprintf(f, "A_offdiag %.17g\n", p.a_off);
		vec("b", p.a_b);
		vec("xstar", p.xstar);
	}
	if (p.name.rfind("logreg", 0) == 0) {
		std::fprintf(f, "lambda %.17g\n", p.lr_lambda);
		// A probe so the Python objective can be proved identical to this one.
		std::vector<double> g(p.n);
		const double f0 = p.fg(p.x0.data(), g.data());
		double g0 = 0.0;
		for (int i = 0; i < p.n; i++) g0 = std::max(g0, std::fabs(g[i]));
		std::vector<double> xp(p.n, 0.1);
		const double fp = p.fg(xp.data(), g.data());
		double gp = 0.0;
		for (int i = 0; i < p.n; i++) gp = std::max(gp, std::fabs(g[i]));
		std::fprintf(f, "probe_x0_f %.17g\nprobe_x0_ginf %.17g\nprobe_p_f %.17g\nprobe_p_ginf %.17g\n", f0, g0, fp, gp);
		std::printf("%-22s probe f(x0)=%.12g |g|inf=%.12g  f(0.1)=%.12g |g|inf=%.12g\n", p.name.c_str(), f0, g0, fp, gp);
	}
	std::fclose(f);
}

// Cost of one objective evaluation, so the solver's own cost can be separated
// from the objective's (both solvers call the same code).
static double time_fg_us(const Problem &p, int calls) {
	std::vector<double> g(p.n);
	std::vector<double> x = p.x0;
	const auto t0 = Clock::now();
	for (int i = 0; i < calls; i++) {
		p.fg(x.data(), g.data());
		x[i % p.n] = x[i % p.n] == 0.0 ? 1e-3 : x[i % p.n];  // stop the compiler folding it
	}
	const auto t1 = Clock::now();
	return std::chrono::duration<double, std::micro>(t1 - t0).count() / calls;
}

static void row(FILE *csv, const Problem &p, const char *solver, const Out &o, double fg_us) {
	const double solver_ms = o.ms_med - o.nfev * fg_us / 1e3;
	std::fprintf(csv, "%s,%s,%d,%d,%.12g,%.6g,%.3g,%.4f,%.4f,%.4f,%.4f,%.4f,%s\n", p.name.c_str(), solver, o.niter,
				 o.nfev, o.f, o.pg, o.xstar_err, o.ms_med, o.ms_min, o.ms_max, fg_us, solver_ms, o.note.c_str());
	std::fprintf(stdout,
				 "%-22s %-9s niter=%-5d nfev=%-5d f=%.10g pg=%.3g xerr=%.2g  median %8.3f ms (solver-only %8.3f)  "
				 "[min %8.3f max %8.3f] fg=%.2f us  %s\n",
				 p.name.c_str(), solver, o.niter, o.nfev, o.f, o.pg, o.xstar_err, o.ms_med, solver_ms, o.ms_min,
				 o.ms_max, fg_us, o.note.c_str());
}

int main(int argc, char **argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <outdir> [reps_small] [reps_big] [delta]\n", argv[0]);
		return 2;
	}
	const std::string out = argv[1];
	const int reps_small = argc > 2 ? std::atoi(argv[2]) : 20;
	const int reps_big = argc > 3 ? std::atoi(argv[3]) : 5;
	const double delta = argc > 4 ? std::atof(argv[4]) : 1e-3;
	std::string mk = "mkdir -p " + out + "/problems";
	if (std::system(mk.c_str()) != 0) return 2;

	std::vector<Problem> probs = {make_rosen(2), make_rosen(10), make_rosen(100), make_upstream(), make_qp(1000),
								  make_rosen(1000), make_qp(10000)};
	probs.push_back(make_logreg(1000, 4000, 1e-4, out + "/logreg_X.bin", out + "/logreg_y.bin"));
	for (const auto &p : probs) dump_problem(out + "/problems", p);

	const std::string csvpath = out + "/cpp_results.csv";
	FILE *csv = std::fopen(csvpath.c_str(), "w");
	std::fprintf(csv, "problem,solver,niter,nfev,f,pg,xstar_err,ms_median,ms_min,ms_max,fg_us,solver_ms,note\n");
	std::printf("problem,solver,niter,nfev,f,pg,xstar_err,ms_median,ms_min,ms_max,fg_us,solver_ms,note\n");
	for (const auto &p : probs) {
		const int reps = p.n >= 1000 ? reps_big : reps_small;
		// Enough objective calls for a stable per-call cost, bounded so the
		// heavy problems do not dominate the run.
		const int calls = p.n >= 1000 ? 200 : 20000;
		const double fg_us = time_fg_us(p, calls);
		Out a = run_driver(p, 10, delta, 20000, reps);
		row(csv, p, "driver", a, fg_us);
		Out b = run_lbfgspp(p, 10, delta, 20000, reps);
		row(csv, p, "lbfgspp", b, fg_us);
		double dx = 0.0, nx = 0.0;
		for (int i = 0; i < p.n; i++) {
			dx = std::max(dx, std::fabs(a.x[i] - b.x[i]));
			nx = std::max(nx, std::fabs(b.x[i]));
		}
		std::printf("    max|x_driver - x_LBFGSpp| = %.3g (inf-norm of x %.3g)\n", dx, nx);
	}
	std::fclose(csv);
	return 0;
}
