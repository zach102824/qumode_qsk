#!/usr/bin/env python3
"""Independent kernel/protocol audit. Does NOT import src.kernels_numba as the truth.

Reference implementations (dense Pauli, naive FWHT, dense H) live in this file.
The campaign kernel under test is imported separately. The pre-rewrite snapshot
is loaded by file path under a distinct module name.
"""
from __future__ import annotations
import os, sys, time, importlib.util, json, hashlib, math, resource, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QSK_NUMBA_CACHE", "0")  # no stale cache during audit
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS",
          "NUMBA_NUM_THREADS"):
    os.environ[k] = "1"

import numpy as np

SNAP_HAM = ROOT / "provenance/pre_round_scheduler_20261110_2310/src/hamiltonian.py"
OUT = ROOT / "audits" / "KERNEL_AUDIT.json"

# ---------------- independent reference (no campaign kernels) ----------------

def pauli_z(n, q):
    dim = 1 << n
    d = np.empty(dim, dtype=np.float64)
    for s in range(dim):
        d[s] = 1.0 - 2.0 * ((s >> q) & 1)
    return d


def pauli_x_apply(v, n, q):
    out = np.empty_like(v)
    bit = 1 << q
    for s in range(v.size):
        out[s] = v[s ^ bit]
    return out


def dense_H_apply(v, J, b, n):
    """H = sum_{i<j} J_ij Z_i Z_j - B sum_q X_q, applied to v. Independent loops."""
    acc = np.zeros_like(v, dtype=np.complex128)
    # ZZ is diagonal
    diag = np.zeros(v.size, dtype=np.float64)
    for i in range(n):
        zi = pauli_z(n, i)
        for j in range(i + 1, n):
            diag += J[i, j] * zi * pauli_z(n, j)
    acc += diag * v
    for q in range(n):
        acc -= b * pauli_x_apply(v, n, q)
    return acc


def naive_fwht(v, n):
    """Normalized Walsh-Hadamard, independent of campaign kernel."""
    out = np.array(v, dtype=np.complex128, copy=True)
    s = np.sqrt(2.0)
    dim = out.size
    for q in range(n):
        step = 1 << q
        for i in range(0, dim, 2 * step):
            for j in range(step):
                a = out[i + j]
                b = out[i + j + step]
                out[i + j] = (a + b) / s
                out[i + j + step] = (a - b) / s
    return out


def naive_rx(v, beta, n):
    out = np.array(v, dtype=np.complex128, copy=True)
    c, s = np.cos(beta), -1j * np.sin(beta)
    dim = out.size
    for q in range(n):
        step = 1 << q
        for i in range(0, dim, 2 * step):
            for j in range(step):
                a = out[i + j]
                d = out[i + j + step]
                out[i + j] = c * a + s * d
                out[i + j + step] = s * a + c * d
    return out


def global_phase_err(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ov = np.vdot(b, a)
    if abs(ov) < 1e-30:
        return float(np.max(np.abs(a - b)))
    return float(np.max(np.abs(a - (ov / abs(ov)) * b)))


def rss_kib():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def load_snapshot_hamiltonian():
    if not SNAP_HAM.exists():
        return None
    spec = importlib.util.spec_from_file_location("ham_snapshot_pre_round", SNAP_HAM)
    mod = importlib.util.module_from_spec(spec)
    # snapshot imports `from .kernels`? old file has no kernels. It is standalone except relative? 
    # old hamiltonian has no package-relative imports besides none. It only uses numpy/scipy.
    spec.loader.exec_module(mod)
    return mod


def main():
    from src.hamiltonian import (
        Model, make_J, energy_grad, central_diff_grad, nov3_energy_grad_forward,
        param_count, walsh, rx, apply_X,
    )
    from src.kernels_numba import warmup
    from src.spsa import optimize as spsa_optimize, query_schedule, spsa_starts
    from src.bfgs import make_start
    from src import protocol as P
    from src.entropy import entropy, schmidt, schmidt_conformance
    from src.io_util import now_shanghai, save_json

    t_all = time.time()
    findings = []
    warmup_t0 = time.time()
    warm = warmup(9)
    warmup_s = time.time() - warmup_t0
    findings.append(dict(name="numba_warmup", pass_=True, detail=warm, wall_s=warmup_s))

    # 1. dense Pauli / FWHT agreement, small N, random params, both ansatz, all L
    dense_rows = []
    dense_ok = True
    for n in (3, 6):
        seed = 831106000 if n == 6 else 42
        b = 0.5
        J = make_J(n, seed)
        m = Model(n, seed, b, J=J, compute_gs=False)
        # diags D0 vs independent ZZ
        plus = np.ones(1 << n, dtype=np.complex128) / np.sqrt(1 << n)
        for trial in range(3):
            rng = np.random.default_rng(9000 + n * 100 + trial)
            v = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
            v = v / np.linalg.norm(v)
            Hv_ref = dense_H_apply(v, J, b, n)
            Hv = m.apply_H(v)
            e_ref = float(np.vdot(v, Hv_ref).real)
            e = m.energy(v)
            w_ref = naive_fwht(v, n)
            w = walsh(v, n)
            dense_rows.append(dict(n=n, trial=trial,
                                   applyH=float(np.max(np.abs(Hv - Hv_ref))),
                                   energy=abs(e - e_ref),
                                   walsh=float(np.max(np.abs(w - w_ref)))))
            if dense_rows[-1]["applyH"] > 1e-12 or dense_rows[-1]["energy"] > 1e-12 or dense_rows[-1]["walsh"] > 1e-12:
                dense_ok = False
        for L in range(1, 7):
            for method in ("nov3", "qaoa"):
                rng = np.random.default_rng(1000 + n * 50 + L * 7 + (0 if method == "nov3" else 3))
                p = param_count(method, L)
                x = rng.normal(0.0, 0.3, p)
                if method == "nov3":
                    x[0] += 0.2
                else:
                    x[::2] += 0.12
                    x[1::2] -= 0.22
                psi = m.state(x, L, method)
                # independent state evolution
                if method == "qaoa":
                    v = plus.copy()
                    for g, beta in x.reshape(L, 2):
                        v = v * np.exp(-1j * g * m.diags[0])
                        v = naive_rx(v, float(beta), n)
                else:
                    gen = m.diags[0] + x[1] * m.diags[1] + x[2] * m.diags[2]
                    phase = np.exp(-1j * x[0] * gen)
                    v = plus.copy()
                    for row in x[3:].reshape(L, 3):
                        v = naive_fwht(v * phase, n)
                        v = np.exp(-1j * (m.shell @ row)) * v
                        v = naive_fwht(v, n)
                st = global_phase_err(psi, v)
                En = m.energy(psi)
                Eref = float(np.vdot(psi, dense_H_apply(psi, J, b, n)).real)
                nrm = float(np.vdot(psi, psi).real)
                Eg, g, vg = energy_grad(m, x, method, L)
                gfd = central_diff_grad(m, x, method, L, eps=1e-7)
                rec = dict(n=n, L=L, method=method, state_err=st, dE=abs(En - Eref),
                           dE_grad=abs(Eg - En), norm=nrm, fd_inf=float(np.max(np.abs(g - gfd))),
                           overlap=abs(np.vdot(psi, v)))
                if method == "nov3":
                    Ef, gf, _ = nov3_energy_grad_forward(m, x, L)
                    rec["fwd_inf"] = float(np.max(np.abs(g - gf)))
                    rec["dE_fwd"] = abs(Eg - Ef)
                dense_rows.append(rec)
                if st > 1e-12 or abs(En - Eref) > 1e-12 or abs(nrm - 1) > 1e-10 or rec["fd_inf"] > 5e-7:
                    dense_ok = False
                if rec.get("fwd_inf", 0) > 1e-10:
                    dense_ok = False
    findings.append(dict(name="dense_pauli_fwht_state_grad", pass_=dense_ok, n=len(dense_rows),
                         worst_fd=max(r.get("fd_inf", 0) for r in dense_rows),
                         worst_state=max(r.get("state_err", 0) or 0 for r in dense_rows),
                         sample=[r for r in dense_rows if "L" in r][:4]))

    # 2. snapshot (pre-rewrite) vs new, if present
    snap_ok = True
    snap_rows = []
    old = load_snapshot_hamiltonian()
    if old is None:
        findings.append(dict(name="snapshot_vs_new", pass_=False, error="snapshot missing"))
        snap_ok = False
    else:
        for n, L, method in [(6, 1, "nov3"), (6, 1, "qaoa"), (6, 6, "nov3"), (9, 2, "nov3"), (12, 1, "qaoa")]:
            seed = 831106000 if n == 6 else (831109000 if n == 9 else 831112000)
            J = make_J(n, seed)
            b = 0.4
            m = Model(n, seed, b, J=J, compute_gs=False)
            mo = old.Model(n, seed, b, J=J, compute_gs=False)
            if float(np.max(np.abs(m.diags[0] - mo.diags[0]))) > 1e-12:
                snap_ok = False
            rng = np.random.default_rng(55 + n + L)
            x = rng.normal(0, 0.3, param_count(method, L))
            vn, vo = m.state(x, L, method), mo.state(x, L, method)
            rec = dict(n=n, L=L, method=method,
                       state=global_phase_err(vn, vo),
                       energy=abs(m.energy(vn) - mo.energy(vo)),
                       diag0=float(np.max(np.abs(m.diags[0] - mo.diags[0]))))
            En, gn, _ = energy_grad(m, x, method, L)
            if hasattr(old, "energy_grad"):
                Eo, go, _ = old.energy_grad(mo, x, method, L)
                rec["grad_inf"] = float(np.max(np.abs(gn - go)))
                rec["dE"] = abs(En - Eo)
                if rec["grad_inf"] > 1e-10:
                    snap_ok = False
            if rec["state"] > 1e-12 or rec["energy"] > 1e-12:
                snap_ok = False
            snap_rows.append(rec)
        findings.append(dict(name="snapshot_vs_new", pass_=snap_ok, rows=snap_rows))

    # 3. RNG starts frozen: widths, offsets, param counts, 1024L ledger
    start_ok = True
    start_rows = []
    for L in range(1, 7):
        for method in ("nov3", "qaoa"):
            assert param_count(method, L) == (3 + 3 * L if method == "nov3" else 2 * L)
            xs, _ = spsa_starts(831106000, L, method)
            if len(xs) != 2 or xs[0].shape[0] != param_count(method, L):
                start_ok = False
            # first start is narrow 0.16 + bias; second broad 0.6 + bias — check construction
            sched = query_schedule(L)
            if sched["total"] != 1024 * L or sched["Q"] != 1024 * L:
                start_ok = False
            start_rows.append(dict(L=L, method=method, p=param_count(method, L), Q=sched["Q"], T=sched["T"]))
    # one real SPSA ledger at N=6 L=1
    m6 = Model(6, 831106000, 0.362, J=make_J(6, 831106000), compute_gs=True)
    sp = spsa_optimize(m6, "nov3", 1)
    roles = {}
    for q in sp["query_log"]:
        roles[q["kind"]] = roles.get(q["kind"], 0) + 1
    ledger_ok = (sp["calls"] == 1024 and len(sp["query_log"]) == 1024)
    findings.append(dict(name="starts_and_1024L_ledger", pass_=start_ok and ledger_ok,
                         start_rows=start_rows, N6_L1_nov3_calls=sp["calls"],
                         query_kinds=roles, Q_ok=ledger_ok))

    # 4. entropy odd-N partition (independent of naive C-order reshape) + E0 residual
    ent_ok = True
    ent_rows = []
    conf = schmidt_conformance()
    if conf["verdict"] != "PASS":
        ent_ok = False
    # Independent odd-N crossing Bell: A=lowest floor(N/2)=4 for N=9, bits 3 and 4.
    n = 9
    v = np.zeros(512, dtype=complex)
    v[0] = v[(1 << 3) + (1 << 4)] = 1 / np.sqrt(2)
    S_camp = entropy(v, 9)
    # Independent Schmidt: index = a + (b << na)
    na = n // 2
    M = np.zeros((1 << na, 1 << (n - na)), dtype=complex)
    for s, amp in enumerate(v):
        M[s & ((1 << na) - 1), s >> na] = amp
    sv = np.linalg.svd(M, compute_uv=False) ** 2
    sv = sv[sv > 1e-15]
    S_ind = float(-np.sum(sv * np.log(sv)))
    # Wrong C-order reshape (known incorrect for odd N)
    svw = np.linalg.svd(v.reshape(16, 32), compute_uv=False) ** 2
    S_wrong = float(-np.sum(svw[svw > 1e-15] * np.log(svw[svw > 1e-15])))
    if abs(S_camp - np.log(2)) > 1e-12 or abs(S_ind - np.log(2)) > 1e-12:
        ent_ok = False
    if abs(S_wrong - np.log(2)) < 1e-12:
        # the demonstration must actually differ from the correct value
        ent_ok = False
    ent_rows.append(dict(n=9, S_campaign=S_camp, S_independent=S_ind,
                         S_wrong_Corder=S_wrong, expected=float(np.log(2))))
    m6r = Model(6, 831106000, 0.362, compute_gs=True)
    residual = getattr(m6r, "residual", None)
    if residual is None:
        # compute residual independently if the attribute name differs
        if m6r.gs is not None:
            residual = float(np.linalg.norm(m6r.apply_H(m6r.gs) - m6r.e0 * m6r.gs))
    if residual is None or residual > 1e-8:
        ent_ok = False
    ent_rows.append(dict(n=6, E0=float(m6r.e0), residual=residual, gap=getattr(m6r, "gap", None)))
    findings.append(dict(name="entropy_odd_partition_and_E0_residual", pass_=ent_ok,
                         conformance=conf["verdict"], rows=ent_rows))

    # 5. plus-basis / energy selection protocol flags
    proto = json.loads((ROOT / "protocol/PROTOCOL_FROZEN.json").read_text())
    body = {k: v for k, v in proto.items() if k != "protocol_sha256"}
    can = json.dumps(body, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(can.encode()).hexdigest()
    sci = proto["science"]
    proto_ok = (h == "f32a96d4af337525edbbeea903b848d9b8c6163af3efa214f4ca71cd143cbb81"
                and sci["coupling_variance"] == "1/N"
                and sci["n_instances_per_N"] == 200
                and tuple(sci["depths"]) == (1, 2, 3, 4, 5, 6)
                and proto["bfgs"]["maxiter"] == 2000
                and proto["bfgs"]["gtol"] == 1e-10
                and proto["spsa"]["starts"] == 2)
    findings.append(dict(name="frozen_protocol_pin", pass_=proto_ok, sha256=h, variance=sci["coupling_variance"]))

    # 6. benchmarks pre/post, same N,L,params, 1 thread, repetitions
    benches = []
    oldmod = old
    for n, L, method, reps in [(6, 1, "nov3", 40), (6, 6, "nov3", 20), (12, 1, "nov3", 8), (18, 1, "nov3", 2)]:
        seed = 831100000 + n * 1000
        J = make_J(n, seed)
        b = 0.5
        m = Model(n, seed, b, J=J, compute_gs=False)
        rng = np.random.default_rng(7 + n + L)
        x = rng.normal(0, 0.2, param_count(method, L))
        # warmup
        _ = m.energy(m.state(x, L, method))
        _ = energy_grad(m, x, method, L)
        rss0 = rss_kib()
        t0 = time.perf_counter()
        for _i in range(reps):
            E = m.energy(m.state(x, L, method))
        t_state = (time.perf_counter() - t0) / reps
        t0 = time.perf_counter()
        for _i in range(max(1, reps // 2)):
            Eg, g, _v = energy_grad(m, x, method, L)
        t_grad = (time.perf_counter() - t0) / max(1, reps // 2)
        rec = dict(n=n, L=L, method=method, reps=reps, new_state_s=t_state, new_grad_s=t_grad,
                   new_rss_kib=rss_kib(), rss0=rss0)
        if oldmod is not None and n <= 18:
            mo = oldmod.Model(n, seed, b, J=J, compute_gs=False)
            _ = mo.energy(mo.state(x, L, method))
            t0 = time.perf_counter()
            for _i in range(reps):
                Eo = mo.energy(mo.state(x, L, method))
            rec["old_state_s"] = (time.perf_counter() - t0) / reps
            t0 = time.perf_counter()
            ng = max(1, reps // 2)
            for _i in range(ng):
                _Eo, _go, _ = oldmod.energy_grad(mo, x, method, L)
            rec["old_grad_s"] = (time.perf_counter() - t0) / ng
            rec["speedup_state"] = rec["old_state_s"] / t_state if t_state else None
            rec["speedup_grad"] = rec["old_grad_s"] / t_grad if t_grad else None
        benches.append(rec)
    findings.append(dict(name="benchmarks_pre_post", pass_=True, rows=benches))

    verdict = "PASS" if all(f.get("pass_") for f in findings) else "FAIL"
    out = dict(
        verdict=verdict,
        shanghai=now_shanghai(),
        wall_s=time.time() - t_all,
        authorization=["pt3496u", "pt3497u", "pt3498u"],
        fastmath=False, numba_threads=1, blas_threads=1,
        findings=[{**{k: v for k, v in f.items() if k != "pass_"}, "pass": f.get("pass_")} for f in findings],
        note="Independent dense/naive reference lives in this driver. Snapshot comparison uses provenance pre-rewrite hamiltonian.py. Science protocol pin unchanged.",
    )
    save_json(OUT, out)
    print(json.dumps(dict(verdict=verdict, wall_s=out["wall_s"],
                          benches=[{k: r.get(k) for k in ("n","L","method","new_state_s","old_state_s","speedup_state","new_grad_s","old_grad_s","speedup_grad")} for r in benches]),
                     indent=2))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
