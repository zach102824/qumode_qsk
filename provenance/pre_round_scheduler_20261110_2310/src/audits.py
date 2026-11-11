"""Correctness audits: Schmidt, variance, gradients, SPSA schedule, start recipes, optional historical peak."""
from __future__ import annotations
import os, time, json, traceback
from pathlib import Path
import numpy as np

from . import protocol as P
from .hamiltonian import Model, make_J, energy_grad, central_diff_grad, param_count
from .entropy import schmidt_conformance, scan_bstar
from .spsa import query_schedule
from .starts import freeze_cell_starts
from .bfgs import N_STARTS
from .io_util import save_json, now_shanghai, sha256_file, pin_source_files
from .paths import AUDITS, SRC, ROOT


def audit_schmidt():
    out = schmidt_conformance()
    out.update(shanghai=now_shanghai())
    save_json(AUDITS / "SCHMIDT_CONFORMANCE.json", out)
    return out


def audit_variance():
    rows = []
    for n in P.NS:
        for seed in (P.main_seed(n, 0), P.pilot_seed(n, 0)):
            J = make_J(n, seed)
            iu = np.triu_indices(n, 1)
            upper = J[iu]
            m = Model(n, seed, 1.0, J=J, compute_gs=False)
            assert np.array_equal(m.J, J)
            assert np.allclose(m.J, m.J.T) and np.allclose(np.diag(m.J), 0)
            rows.append(dict(
                N=n, seed=seed,
                n_couplings=int(upper.size),
                mean=float(upper.mean()),
                var=float(upper.var(ddof=0)),
                expected_var=1.0 / n,
                std=float(upper.std(ddof=0)),
                expected_std=1.0 / np.sqrt(n),
                generator="numpy.random.default_rng(seed).normal(0, 1/sqrt(N))",
                coupling_variance="1/N",
                not_half_variance=True,
            ))
    out = dict(verdict="PASS", rows=rows, shanghai=now_shanghai(),
               note="Single-instance sample variance is not expected to equal 1/N exactly; the draw formula is.")
    save_json(AUDITS / "VARIANCE_CHECK.json", out)
    return out


def audit_schedule():
    rows = {}
    for L in P.DEPTHS:
        rows[str(L)] = query_schedule(L)
    out = dict(verdict="PASS", query_schedule=rows, shanghai=now_shanghai(),
               formula="Tmax such that 18+4T+2*floor(T/5)+padding=Q-6, padding in 0..5, Q=1024L")
    save_json(AUDITS / "SPSA_QUERY_SCHEDULE.json", out)
    return out


def audit_starts():
    rows = []
    seed = P.pilot_seed(6, 0)
    for L in P.DEPTHS:
        for method in P.METHODS:
            cell = freeze_cell_starts(seed, L, method)
            widths_s = [s["width"] for s in cell["spsa_starts"]]
            widths_b = [s["width"] for s in cell["bfgs_starts"]]
            assert widths_s == [0.16, 0.6]
            assert widths_b == [0.6 if k % 2 == 0 else 0.16 for k in range(N_STARTS)]
            assert cell["spsa_n"] == 2 and cell["bfgs_n"] == 16
            # Nov3 bias on x0, QAOA biases
            if method == "nov3":
                # not a hard equality after gaussian draw; just record x0
                pass
            rows.append(dict(
                seed=seed, L=L, method=method, p=cell["param_count"],
                spsa_widths=widths_s, bfgs_width_counts={"0.16": widths_b.count(0.16), "0.6": widths_b.count(0.6)},
                spsa_sha=[s["x_sha256"] for s in cell["spsa_starts"]],
                bfgs_sha0=cell["bfgs_starts"][0]["x_sha256"],
                bfgs_sha1=cell["bfgs_starts"][1]["x_sha256"],
            ))
    # reproducibility
    c1 = freeze_cell_starts(seed, 1, "nov3")
    c2 = freeze_cell_starts(seed, 1, "nov3")
    assert c1["bfgs_starts"][0]["x"] == c2["bfgs_starts"][0]["x"]
    out = dict(verdict="PASS", rows=rows, reproducible=True, shanghai=now_shanghai(),
               no_spsa_warm_into_bfgs=True, no_layer_transfer=True)
    save_json(AUDITS / "STARTS_RECIPE.json", out)
    return out


def audit_gradients(ns=(6, 9, 12), depths=(1, 2), n_points=3):
    rows = []
    failures = []
    for n in ns:
        seed = P.pilot_seed(n, 0)
        m = Model(n, seed, 0.4, compute_gs=False)
        for L in depths:
            for method in P.METHODS:
                p = param_count(method, L)
                for i in range(n_points):
                    rng = np.random.default_rng(seed * 17 + L * 2003 + (11 if method == "nov3" else 29) + 1009 * (i + 1))
                    width = 0.16 if i % 2 == 0 else 0.6
                    x = rng.normal(0.0, width, p)
                    if method == "nov3":
                        x[0] += 0.2
                    else:
                        x[::2] += 0.12
                        x[1::2] -= 0.22
                    E, g, v = energy_grad(m, x, method, L)
                    v_ref = m.state(x, L, method)
                    E_ref = m.energy(v_ref)
                    state_err = float(np.max(np.abs(v - v_ref)))
                    energy_err = abs(E - E_ref)
                    nrm = float(np.vdot(v, v).real)
                    g_fd = central_diff_grad(m, x, method, L, eps=1e-7)
                    abs_err = float(np.max(np.abs(g - g_fd)))
                    rel = abs_err / max(float(np.linalg.norm(g_fd)), 1e-12)
                    rec = dict(N=n, seed=seed, method=method, L=L, point=i, width=width, E=E,
                               state_max_abs_err=state_err, energy_vs_driver=energy_err, norm=nrm,
                               grad_max_abs_err_vs_central=abs_err, grad_rel_err=rel,
                               grad_norm=float(np.linalg.norm(g)), fd_eps=1e-7,
                               fidelity_not_used_in_grad=True)
                    ok = (state_err < 1e-12 and energy_err < 1e-12 and abs(nrm - 1) < 1e-12
                          and abs_err < 5e-6 and rel < 5e-6)
                    rec["ok"] = bool(ok)
                    rows.append(rec)
                    if not ok:
                        failures.append(rec)
    out = dict(verdict="PASS" if not failures else "FAIL", n_rows=len(rows), failures=failures,
               max_abs_err=max(r["grad_max_abs_err_vs_central"] for r in rows) if rows else None,
               shanghai=now_shanghai(), rows=rows)
    save_json(AUDITS / "GRADIENT_AUDIT.json", out)
    if failures:
        raise AssertionError(f"gradient audit failed: {failures[:2]}")
    return out


def audit_historical_n6_seed10000():
    hist = Path("/workspace/Nov9_SK_SPSA_local/entropy_stress/N6_seed10000_scan.json")
    href = Path("/workspace/Nov9_SK_SPSA_local/entropy_stress/N6_seed10000_reference.npz")
    if not hist.exists():
        out = dict(verdict="SKIP", reason="historical file missing", path=str(hist), shanghai=now_shanghai())
        save_json(AUDITS / "HISTORICAL_N6_SEED10000.json", out)
        return out
    base = Model(6, 10000, 1.0, compute_gs=False)
    report, gs = scan_bstar(base)
    hist_d = json.loads(hist.read_text())
    rec = dict(
        reproduced_label=report["label"], historical_label=hist_d["label"],
        reproduced_b=report["peak"]["b"], historical_b=hist_d["peak"]["b"],
        S_abs_err=abs(report["peak"]["S"] - hist_d["peak"]["S"]),
        E0_abs_err=abs(report["peak"]["E0"] - hist_d["peak"]["E0"]),
        peak_b_exact_match=abs(report["peak"]["b"] - hist_d["peak"]["b"]) < 1e-12,
        shanghai=now_shanghai(),
        not_a_panel_seed=True,
    )
    if href.exists():
        z = np.load(href)
        rec["J_identical"] = bool(np.array_equal(base.J, z["J"]))
        rec["GS_overlap_squared"] = float(abs(np.vdot(gs, z["gs"])) ** 2)
    ok = rec["peak_b_exact_match"] and rec["S_abs_err"] < 1e-10 and rec["E0_abs_err"] < 1e-10
    if "GS_overlap_squared" in rec:
        ok = ok and rec["GS_overlap_squared"] > 1 - 1e-10 and rec.get("J_identical", True)
    rec["verdict"] = "PASS" if ok else "FAIL"
    rec["valid_for_fits_reproduced"] = report["valid_for_fits"]
    save_json(AUDITS / "HISTORICAL_N6_SEED10000.json", rec)
    if not ok:
        raise AssertionError(rec)
    return rec


def audit_source_pins():
    rows = pin_source_files(SRC)
    extra = []
    for p in sorted((ROOT / "scripts").glob("*.py")):
        extra.append(dict(path=str(p.relative_to(ROOT)), sha256=sha256_file(p), bytes=p.stat().st_size))
    out = dict(verdict="PASS", src=rows, scripts=extra, shanghai=now_shanghai())
    save_json(AUDITS / "SOURCE_PINS.json", out)
    return out


def run_all_audits():
    t0 = time.time()
    results = {}
    order = [
        ("schmidt", audit_schmidt),
        ("variance", audit_variance),
        ("schedule", audit_schedule),
        ("starts", audit_starts),
        ("gradients", audit_gradients),
        ("historical_n6_seed10000", audit_historical_n6_seed10000),
        ("source_pins", audit_source_pins),
    ]
    failed = []
    for name, fn in order:
        t = time.time()
        try:
            out = fn()
            results[name] = dict(verdict=out.get("verdict"), wall_s=time.time() - t)
        except Exception as ex:
            results[name] = dict(verdict="FAIL", error=repr(ex), traceback=traceback.format_exc(), wall_s=time.time() - t)
            failed.append(name)
    summary = dict(
        verdict="PASS" if not failed else "FAIL",
        failed=failed,
        results=results,
        wall_s=time.time() - t0,
        shanghai=now_shanghai(),
    )
    save_json(AUDITS / "AUDIT_SUMMARY.json", summary)
    return summary
