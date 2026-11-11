#!/usr/bin/env python3
"""End-to-end cell timings on disjoint PILOT references. Does not write main results."""
from __future__ import annotations
import os, sys, time, json, resource
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QSK_NUMBA_CACHE", "1")
from src.resources import set_worker_thread_env
set_worker_thread_env()
import numpy as np
from src.hamiltonian import Model
from src.kernels_numba import warmup
from src.spsa import optimize as spsa_optimize
from src.bfgs import make_start, run_start
from src.io_util import save_json, now_shanghai
from src.paths import DATA, AUDITS

def rss():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def load_pilot(n):
    seed = 831100000 + n * 1000 + 800
    z = np.load(DATA / "references" / "pilot" / f"N{n}_seed{seed}_reference.npz")
    m = Model(n, seed, float(z["b"]), J=z["J"], compute_gs=False)
    m.load_gs(z["gs"], float(z["E0"]), gap=float(z["gap"]))
    return m, seed

def main():
    t_all = time.time()
    print("warmup", warmup(9), flush=True)
    rows = []
    # (n, L, method, what)
    plan = [
        (6, 1, "nov3", "spsa"),
        (6, 6, "nov3", "spsa"),
        (6, 1, "nov3", "bfgs1"),
        (6, 6, "nov3", "bfgs1"),
        (12, 1, "nov3", "spsa"),
        (12, 1, "qaoa", "spsa"),
        (12, 6, "nov3", "spsa"),
        (12, 1, "nov3", "bfgs1"),
        (12, 6, "nov3", "bfgs1"),
        (18, 1, "nov3", "spsa"),
        (18, 1, "nov3", "bfgs1"),
    ]
    cache = {}
    for n, L, method, what in plan:
        if n not in cache:
            cache[n] = load_pilot(n)
        m, seed = cache[n]
        rss0 = rss()
        t0 = time.time(); c0 = time.process_time()
        if what == "spsa":
            out = spsa_optimize(m, method, L)
            rec = dict(what=what, N=n, L=L, method=method, seed=seed,
                       calls=out["calls"], best_E=out["best_E"], F=out["F"],
                       wall_s=time.time()-t0, CPU_s=time.process_time()-c0,
                       peak_RSS_kib=rss(), rss0=rss0)
        else:
            st = make_start(seed, L, method, 0)
            out = run_start(m, method, L, st)
            rec = dict(what=what, N=n, L=L, method=method, seed=seed,
                       best_E=out["best_E"], F=out["F"], nfev=out["nfev"], nit=out["nit"],
                       success=out["success"], tightly_converged=out["tightly_converged"],
                       wall_s=time.time()-t0, CPU_s=time.process_time()-c0,
                       peak_RSS_kib=rss(), rss0=rss0)
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    out = dict(shanghai=now_shanghai(), wall_s=time.time()-t_all, rows=rows,
               note="pilot seeds, disjoint from main; not written to runs/main")
    AUDITS.mkdir(parents=True, exist_ok=True)
    save_json(AUDITS / "CELL_BENCH.json", out)
    print("wrote", AUDITS / "CELL_BENCH.json", "wall", out["wall_s"], flush=True)

if __name__ == "__main__":
    main()
