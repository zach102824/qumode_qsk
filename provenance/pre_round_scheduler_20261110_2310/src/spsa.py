"""Frozen R9 IID SPSA: Q=1024*L inclusive, 2 cold starts (narrow 0.16, broad 0.6)."""
from __future__ import annotations
import os, hashlib, collections
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[k] = "1"

import numpy as np
from .hamiltonian import param_count


def T_of_Q(Q):
    Q = int(Q)
    T = 0
    while 18 + 4 * (T + 1) + 2 * ((T + 1) // 5) <= Q - 6:
        T += 1
    prefix = 18 + 4 * T + 2 * (T // 5)
    padding = Q - 6 - prefix
    return dict(Q=Q, T=T, prefix_queries=prefix, padding=padding, readouts=6)


def query_schedule(L):
    L = int(L)
    Q = 1024 * L
    d = T_of_Q(Q)
    T = d["T"]
    roles = dict(
        initial=2,
        calibration_plus=8,
        calibration_minus=8,
        spsa_plus=2 * T,
        spsa_minus=2 * T,
        candidate_monitor=2 * (T // 5),
        charged_prefix_padding_remeasurement=d["padding"],
        readout_late_iterate=6,
    )
    total = sum(roles.values())
    if total != Q or not (0 <= d["padding"] <= 5):
        raise AssertionError((L, Q, roles, total, d))
    d.update(L=L, roles=roles, total=total)
    return d


def spsa_starts(seed, L, method):
    p = param_count(method, L)
    rng = np.random.default_rng(int(seed) * 107 + int(L) * 2003 + {"nov3": 11, "qaoa": 29}[method])
    starts = []
    for sd in (0.16, 0.6):
        x = rng.normal(0.0, sd, p)
        if method == "nov3":
            x[0] += 0.2
        else:
            x[::2] += 0.12
            x[1::2] -= 0.22
        starts.append(x)
    return starts, rng


def optimize(m, method, L, Q=None):
    L = int(L)
    Q = 1024 * L if Q is None else int(Q)
    assert method in ("nov3", "qaoa") and 1 <= L <= 6 and Q == 1024 * L
    sched = query_schedule(L)
    T = sched["T"]
    p = param_count(method, L)
    seed = m.seed
    starts, rng = spsa_starts(seed, L, method)
    # Continue the same RNG after drawing starts (matches depth_core: one rng for starts+directions).
    calls = 0
    bestE = np.inf
    best = None
    log = []
    snaps = []
    histories = []
    direction_hashes = []
    marks = {128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144, Q - 6}
    marks.discard(Q)

    def ev(x, kind, meta=None):
        nonlocal calls, bestE, best
        E = m.energy(m.state(x, L, method))
        assert np.isfinite(E)
        calls += 1
        z = dict(query=calls, kind=kind, E=E, x_sha256=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest())
        if meta:
            z.update(meta)
        log.append(z)
        if E < bestE:
            bestE = E
            best = x.copy()
        if calls in marks:
            snaps.append((calls, bestE, best.copy()))
        return E

    norms = []
    for x in starts:
        ev(x, "initial")
        for _ in range(4):
            delta = rng.choice([-1.0, 1.0], p)
            g = (ev(x + 0.1 * delta, "calibration_plus") - ev(x - 0.1 * delta, "calibration_minus")) / 0.2 * delta
            norms.append(min(float(np.linalg.norm(g)), 5.0))
    gain = float(np.clip(0.15 / ((0.12 / 10 ** 0.602) * max(float(np.median(norms)), 1e-12)), 1.0, 4.0))
    for x0 in starts:
        x = x0.copy()
        hist = []
        dirs = []
        for k in range(T):
            delta = rng.choice([-1.0, 1.0], p)
            dirs.append(delta.astype(np.int8))
            c = 0.1 / (k + 1) ** 0.101
            a = gain * 0.12 / (k + 10) ** 0.602
            g = (ev(x + c * delta, "spsa_plus") - ev(x - c * delta, "spsa_minus")) / (2 * c) * delta
            g *= min(1.0, 5.0 / max(float(np.linalg.norm(g)), 1e-15))
            x = x - a * g
            hist.append(x.copy())
            if (k + 1) % 5 == 0:
                ev(x, "candidate_monitor")
        histories.append(np.array(hist))
        direction_hashes.append(hashlib.sha256(np.ascontiguousarray(dirs, dtype=np.int8).tobytes()).hexdigest())
    padding = Q - 6 - calls
    assert 0 <= padding <= 5
    for _ in range(padding):
        ev(best, "charged_prefix_padding_remeasurement")
    assert calls == Q - 6
    prefix_E = bestE
    readouts = []
    for restart, h in enumerate(histories):
        for k in [k for k in range(T, 0, -1) if k % 5 != 0][:3]:
            x = h[k - 1]
            E = ev(x, "readout_late_iterate", dict(restart=restart, iterate_index=k))
            readouts.append(dict(E=E, x=x.tolist(), restart=restart, iterate_index=k))
    assert calls == Q
    v = m.state(best, L, method)
    F = m.fidelity(v) if m.gs is not None else None
    trace = []
    for q, E, x in snaps:
        rec = dict(query=q, E=E)
        if m.gs is not None:
            rec["F"] = m.fidelity(m.state(x, L, method))
        trace.append(rec)
    recQ = dict(query=Q, E=bestE)
    if F is not None:
        recQ["F"] = F
    trace.append(recQ)
    return dict(
        calls=Q, L=L, method=method, x=best.tolist(), best_E=bestE, F=F,
        norm=float(np.vdot(v, v).real),
        query_log=log, trace=trace,
        updates_per_trajectory=T, prefix_padding_queries=padding,
        prefix_best_E=prefix_E, readout_candidates=[{k: r[k] for k in r if k != "x"} | {"x_sha256": hashlib.sha256(np.ascontiguousarray(r["x"]).tobytes()).hexdigest()} for r in readouts],
        gain_multiplier=gain,
        query_roles=dict(collections.Counter(z["kind"] for z in log)),
        initial_points=[x.tolist() for x in starts],
        direction_hashes=direction_hashes,
        parameter_count=p,
        schedule=sched,
        fidelity_only_after_all_energy_selection=True,
        init_scheme="TWO_COLD_NARROW_0.16_BROAD_0.6_NO_WARM_NO_LAYER_TRANSFER",
    )
