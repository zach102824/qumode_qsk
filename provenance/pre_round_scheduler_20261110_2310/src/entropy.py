"""Per-instance finite-N ground-state entropy peak. Spin bipartition, not cavity.

A = lowest floor(N/2) logical spin labels. Natural-log von Neumann entropy.
Parity-even branch; B=0 is not on the primary grid.
Never materialize dense 2^{N-1} matrices for N>=12 (eigsh / LinearOperator).
"""
from __future__ import annotations
import os, time, json, resource
from pathlib import Path
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[k] = "1"

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh, LinearOperator

from .hamiltonian import Model, apply_X


def schmidt(gs, n):
    na = n // 2
    aa = np.arange(2 ** na)[:, None]
    bb = np.arange(2 ** (n - na))[None, :]
    return gs[aa + (bb << na)]


def entropy(gs, n):
    sv = np.linalg.svd(schmidt(gs, n), compute_uv=False) ** 2
    sv = sv[sv > 1e-15]
    return float(-np.sum(sv * np.log(sv)))


def schmidt_conformance(ns=(6, 9, 12, 15, 18, 21)):
    rows = []
    for n in ns:
        for name, positions, expected in (
            ("product", None, 0.0),
            ("crossing_bell", (n // 2 - 1, n // 2), np.log(2)),
            ("within_A_bell", (0, 1), 0.0),
        ):
            v = np.zeros(2 ** n, dtype=complex)
            v[0] = 1
            if positions is not None:
                v[0] = 1 / np.sqrt(2)
                v[(1 << positions[0]) + (1 << positions[1])] = 1 / np.sqrt(2)
            M = schmidt(v, n)
            rho = M @ M.conj().T
            ev = np.linalg.eigvalsh(rho)
            ev = ev[ev > 1e-15]
            S = float(-np.sum(ev * np.log(ev)))
            S2 = entropy(v, n)
            if abs(S - expected) >= 1e-12 or abs(S2 - expected) >= 1e-12:
                raise AssertionError((n, name, S, S2, expected))
            rows.append(dict(n=n, test=name, entropy=S, expected=float(expected)))
    # Old odd-N C-order reshape is wrong; keep the demonstration.
    n = 9
    v = np.zeros(512, dtype=complex)
    v[0] = v[(1 << 3) + (1 << 4)] = 1 / np.sqrt(2)
    sv = np.linalg.svd(v.reshape(16, 32), compute_uv=False) ** 2
    old = float(-np.sum(sv[sv > 1e-15] * np.log(sv[sv > 1e-15])))
    return dict(
        verdict="PASS",
        cut="A=lowest floor(N/2) logical spin labels, natural-log entropy",
        not_cavity_partition=True,
        tests=rows,
        old_oddN_wrong_reshape_demonstration_entropy=old,
        correct_oddN_crossing_Bell_entropy=entropy(v, 9),
    )


def _parity_ops(n, hz_half=None):
    dim = 1 << n
    half = dim // 2
    reps = np.arange(half)
    mask = dim - 1
    # Build X in even/odd sectors as LinearOperators or CSR.
    # flipped = reps ^ (1<<q); if still in half, even-sector hop; else map by global flip.
    data_p, row_p, col_p = [], [], []
    data_m, row_m, col_m = [], [], []
    for q in range(n):
        flipped = reps ^ (1 << q)
        same = flipped < half
        idx = np.where(same, flipped, flipped ^ mask)
        data_p.append(np.ones(half))
        row_p.append(reps)
        col_p.append(idx)
        data_m.append(np.where(same, 1.0, -1.0))
        row_m.append(reps)
        col_m.append(idx)
    Xp = csr_matrix((np.concatenate(data_p), (np.concatenate(row_p), np.concatenate(col_p))), shape=(half, half))
    Xm = csr_matrix((np.concatenate(data_m), (np.concatenate(row_m), np.concatenate(col_m))), shape=(half, half))
    return reps, mask, half, Xp, Xm


def _lowest(H, k, n, seed, v0=None, dense_ok=False):
    half = H.shape[0]
    if dense_ok and n <= 9:
        if k == 1:
            ev = eigh(H.toarray(), eigvals_only=True, subset_by_index=(0, 0))
            return ev, None
        ev, vec = eigh(H.toarray(), subset_by_index=(0, k - 1))
        return ev, vec
    # Sparse path: never toarray().
    if k == 1:
        ev = eigsh(H, k=1, which="SA", tol=2e-12, return_eigenvectors=False,
                   v0=(v0 if v0 is not None else np.random.default_rng(seed + 100).normal(size=half)))
        return ev, None
    ev, vec = eigsh(H, k=k, which="SA", tol=2e-12,
                    v0=(v0 if v0 is not None else np.ones(half)))
    order = np.argsort(ev)
    return ev[order], vec[:, order]


def evaluate_field(base: Model, b, cache_ops=None):
    n = base.n
    b = round(float(b), 9)
    if cache_ops is None:
        reps, mask, half, Xp, Xm = _parity_ops(n)
        hz = base.diags[0][:half]
        cache_ops = dict(reps=reps, mask=mask, half=half, Xp=Xp, Xm=Xm, hz=hz)
    reps, mask, half, Xp, Xm, hz = (cache_ops[k] for k in ("reps", "mask", "half", "Xp", "Xm", "hz"))
    dim = 1 << n
    Hp = diags(hz) - b * Xp
    Hm = diags(hz) - b * Xm
    dense_ok = n <= 9
    v0 = None
    if cache_ops is not None:
        v0 = cache_ops.get("v0")
    ep, vp = _lowest(Hp, 2, n, base.seed, v0=(v0 if v0 is not None else np.ones(half)), dense_ok=dense_ok)
    em, _ = _lowest(Hm, 1, n, base.seed, dense_ok=dense_ok)
    v = np.asarray(vp[:, 0], dtype=np.float64).copy()
    if v.sum() < 0:
        v = -v
    gs = np.zeros(dim, dtype=complex)
    gs[reps] = v / np.sqrt(2)
    gs[reps ^ mask] = v / np.sqrt(2)
    gap = float(em[0] - ep[0])
    res = float(np.linalg.norm(Hp @ v - ep[0] * v))
    S = entropy(gs, n)
    if np.min(v) <= -1e-9:
        # Gauge/positivity failed; still return with a flag, do not crash the panel.
        pos_ok = False
    else:
        pos_ok = True
    row = dict(
        b=b, S=S, E0=float(ep[0]), gap=gap, even_gap=float(ep[1] - ep[0]),
        residual=res, parity=float(np.vdot(gs, gs[::-1]).real),
        near_degenerate=gap < 1e-9, branch="positive even GS",
        norm=float(np.vdot(gs, gs).real), positivity_ok=bool(pos_ok),
        min_even_representative=float(np.min(v)),
    )
    return row, gs, cache_ops


def scan_bstar(base: Model, progress_path=None):
    """Same field grid and peak rule as the frozen protocol. Only the peak GS is retained.

    Intermediate full-space ground states are discarded after their entropy is recorded.
    This is required at N=21: caching every field GS is ~32 MiB each and was the 4 GiB leak.
    """
    n = base.n
    cache = {}
    peak_gs = None
    ops = None
    last_v0 = None
    stats = dict(eigensolves=0, matrices_built=0, wall_s=0.0, CPU_s=0.0, peak_fields_stored=1)
    t0 = time.time(); c0 = time.process_time()

    def at(b):
        nonlocal ops, peak_gs, last_v0
        b = round(float(b), 9)
        if b in cache:
            return cache[b]
        t1 = time.time()
        # Pass the previous even-sector vector as eigsh v0 when present (same tolerances).
        if ops is not None and last_v0 is not None:
            ops = dict(ops)
            ops["v0"] = last_v0
        row, gs, ops = evaluate_field(base, b, cache_ops=ops)
        # Recover even representative for the next v0 without keeping the full GS.
        half = 1 << (n - 1)
        last_v0 = np.real(gs[:half] * np.sqrt(2.0)).astype(np.float64)
        cache[b] = row
        if peak_gs is None or row["S"] >= max(r["S"] for r in cache.values()):
            peak_gs = gs
        stats["eigensolves"] += 2
        stats["matrices_built"] += 2
        if progress_path:
            rec = dict(N=n, seed=getattr(base, "seed", None), b=b, S=row["S"],
                       n_fields=len(cache), wall_s=time.time()-t0, field_s=time.time()-t1,
                       RSS_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                       E0=row["E0"], gap=row["gap"])
            Path(progress_path).parent.mkdir(parents=True, exist_ok=True)
            with open(progress_path, "a") as f:
                f.write(json.dumps(rec)+"\n")
        return row

    for b in np.r_[np.arange(0.01, 3.0, 0.05), 3.0]:
        at(b)
    peak = max(cache.values(), key=lambda r: r["S"])
    if peak["b"] == min(cache):
        for b in np.arange(0.001, 0.01, 0.001):
            at(b)
    if peak["b"] == max(cache):
        for b in np.arange(3.05, 6.00001, 0.05):
            at(b)
    peak = max(cache.values(), key=lambda r: r["S"])
    lo, hi = min(cache), max(cache)
    center = peak["b"]
    for b in np.arange(max(lo, center - 0.05), min(hi, center + 0.05) + 1e-8, 0.005):
        at(b)
    peak = max(cache.values(), key=lambda r: r["S"])
    center2 = peak["b"]
    for b in np.arange(max(lo, center2 - 0.005), min(hi, center2 + 0.005) + 1e-8, 0.001):
        at(b)
    peak = max(cache.values(), key=lambda r: r["S"])
    ordered = sorted(cache)
    k = ordered.index(peak["b"])
    interior = 0 < k < len(ordered) - 1
    left = cache[ordered[k - 1]] if k > 0 else None
    right = cache[ordered[k + 1]] if k < len(ordered) - 1 else None
    localmax = interior and peak["S"] > left["S"] + 1e-12 and peak["S"] > right["S"] + 1e-12
    if localmax and not peak["near_degenerate"]:
        label = "interior_unique_GS_entropy_peak"
    elif localmax:
        label = "interior_even_branch_peak_near_degenerate"
    else:
        label = "boundary_or_plateau_UNRESOLVED"
    valid = (
        localmax and label == "interior_unique_GS_entropy_peak"
        and min(peak["gap"], peak["even_gap"]) > 1e-9
        and peak["residual"] < 1e-8
        and peak["parity"] > 1 - 1e-10
        and not peak["near_degenerate"]
        and peak.get("positivity_ok", True)
    )
    stats.update(wall_s=time.time() - t0, CPU_s=time.process_time() - c0)
    report = dict(
        N=n, seed=base.seed,
        cut_A=list(range(n // 2)),
        entropy_log="natural",
        selection="per-instance finite-N GS entropy peak; not ensemble-common or thermodynamic critical field",
        peak=peak, label=label, interior_validated=bool(localmax),
        valid_for_fits=bool(valid),
        neighbors=[left, right],
        scan=[cache[b] for b in ordered],
        scan_resource=stats,
        field_selection_is_classical_benchmark_not_optimizer=True,
        coupling_variance="1/N",
        no_seed_replacement="keep all 200; flag invalid rather than replace",
    )
    # Ensure returned GS matches the finally selected peak field (recompute once if needed).
    if peak_gs is None or abs(entropy(peak_gs, n) - peak["S"]) > 1e-10:
        _, peak_gs, _ = evaluate_field(base, peak["b"], cache_ops=ops)
    return report, peak_gs
