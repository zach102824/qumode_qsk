"""Unlimited-query BFGS baseline: 16 cold IID starts, 8 narrow / 8 broad, energy selection."""
from __future__ import annotations
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[k] = "1"

import numpy as np
from scipy.optimize import minimize
from .hamiltonian import param_count, energy_grad, central_diff_grad
from .io_util import x_sha256

MAXITER = 2000
GTOL = 1e-10
TIGHT_INF = 1e-8
TIGHT_L2 = 1e-8
N_STARTS = 16


def start_rng_seed(seed, L, method, start_index):
    off = 11 if method == "nov3" else 29
    return int(int(seed) * 1000003 + int(L) * 10007 + off + int(start_index) * 97)


def make_start(seed, L, method, start_index):
    k = int(start_index)
    p = param_count(method, L)
    width = 0.16 if (k % 2) else 0.6
    width_class = "narrow" if (k % 2) else "broad"
    rng_seed = start_rng_seed(seed, L, method, k)
    rng = np.random.default_rng(rng_seed)
    x = rng.normal(0.0, width, p)
    if method == "nov3":
        x[0] += 0.2
    else:
        x[::2] += 0.12
        x[1::2] -= 0.22
    return dict(
        start_index=k, kind="iid_fresh_cold", width=width, width_class=width_class,
        rng_seed=rng_seed, x=x, x_sha256=x_sha256(x),
    )


def all_starts(seed, L, method):
    return [make_start(seed, L, method, k) for k in range(N_STARTS)]


def run_start(m, method, L, start, maxiter=MAXITER, gtol=GTOL):
    x0 = np.asarray(start["x"] if isinstance(start, dict) else start, float)
    evals = {"n": 0}

    def fun(x):
        E, g, v = energy_grad(m, x, method, L)
        evals["n"] += 1
        evals["last_v"] = v
        evals["last_E"] = E
        evals["last_g"] = g
        return E, g

    E0, g0, v0 = energy_grad(m, x0, method, L)
    F0 = m.fidelity(v0) if m.gs is not None else None
    res = minimize(lambda x: fun(x), x0, jac=True, method="BFGS",
                   options=dict(maxiter=maxiter, gtol=gtol, norm=np.inf, disp=False))
    x = np.asarray(res.x, float)
    E, g, v = energy_grad(m, x, method, L)
    nrm = float(np.vdot(v, v).real)
    E_ref = m.energy(m.state(x, L, method))
    v_ref = m.state(x, L, method)
    state_err = float(np.max(np.abs(v - v_ref)))
    g_inf = float(np.max(np.abs(g)))
    g_l2 = float(np.linalg.norm(g))
    # Per-start central-difference is expensive at large N; the independent
    # kernel audit covers FD vs analytic. Optional hook: QSK_FD_CHECK=1.
    if os.environ.get("QSK_FD_CHECK", "0") not in ("0", "false", "False"):
        g_fd = central_diff_grad(m, x, method, L, eps=1e-7)
        fd_err = float(np.max(np.abs(g - g_fd)))
    else:
        fd_err = None
    success = bool(res.success)
    hit_cap = bool((not success) and (int(res.nit) >= maxiter or "iteration" in str(res.message).lower()))
    tight = bool(success and g_inf < TIGHT_INF and g_l2 < TIGHT_L2 and abs(nrm - 1) < 1e-10)
    F = m.fidelity(v) if m.gs is not None else None
    out = dict(
        start_index=int(start.get("start_index", -1)) if isinstance(start, dict) else -1,
        start_kind=start.get("kind") if isinstance(start, dict) else None,
        start_width=start.get("width") if isinstance(start, dict) else None,
        start_width_class=start.get("width_class") if isinstance(start, dict) else None,
        start_rng_seed=start.get("rng_seed") if isinstance(start, dict) else None,
        start_x_sha256=start.get("x_sha256") if isinstance(start, dict) else x_sha256(x0),
        E_initial=float(E0), F_initial=F0, grad_l2_initial=float(np.linalg.norm(g0)),
        x=x.tolist(), x_sha256=x_sha256(x),
        best_E=float(E), F=F, norm=nrm,
        E_vs_driver=abs(E - E_ref),
        state_max_abs_err_vs_driver=state_err,
        final_grad=g.tolist(), final_grad_inf=g_inf, final_grad_l2=g_l2,
        final_grad_max_abs_err_vs_central=fd_err,
        success=success, status=int(res.status), message=str(res.message),
        nit=int(res.nit), nfev=int(res.nfev), njev=int(getattr(res, "njev", -1)),
        unique_energy_grad_evals=int(evals["n"]),
        hit_iteration_cap=hit_cap, tightly_converged=tight,
        objective="energy_only", fidelity_not_optimized=True,
        maxiter=int(maxiter), gtol=float(gtol),
        init_scheme="ALL_16_IID_COLD_8NARROW_8BROAD_NO_LAYER_TRANSFER_NO_SPSA_WARM",
    )
    if m.e0 is not None:
        out.update(E0=m.e0, energy_error=float(E - m.e0), energy_error_over_N=float((E - m.e0) / m.n))
    return out


def select_by_energy(start_results):
    best = min(start_results, key=lambda z: (z["best_E"], z.get("start_index", 0)))
    Fvals = [z for z in start_results if z.get("F") is not None]
    bestF = max(Fvals, key=lambda z: (z["F"], -z.get("start_index", 0))) if Fvals else None
    return dict(
        selected_start=best.get("start_index"),
        selected_E=best["best_E"],
        selected_F=best.get("F"),
        selected_success=best.get("success"),
        selected_tightly_converged=best.get("tightly_converged"),
        selected_hit_cap=best.get("hit_iteration_cap"),
        selection="lowest_energy_among_16_cold_starts",
        max_F_start=None if bestF is None else bestF.get("start_index"),
        max_F=None if bestF is None else bestF.get("F"),
        max_F_from_different_start=bool(bestF is not None and bestF.get("start_index") != best.get("start_index")),
        n_success=sum(1 for z in start_results if z.get("success")),
        n_tight=sum(1 for z in start_results if z.get("tightly_converged")),
        n_hit_cap=sum(1 for z in start_results if z.get("hit_iteration_cap")),
    )
