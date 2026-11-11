"""Main-campaign task dispatch. Same frozen science as the pilot; writes under data/*/main and runs/main.

Does not import or resume historical jobs. Variance is 1/N. No extra arms, no SPSA-warm, no half-variance.
"""
from __future__ import annotations
import os, time, resource, traceback, hashlib
import numpy as np

from . import protocol as P
from .hamiltonian import Model, make_J
from .entropy import scan_bstar
from .spsa import optimize as spsa_optimize
from .bfgs import make_start, run_start
from .io_util import save_json, now_shanghai, sha256_file, x_sha256
from .paths import DATA, LOGS, RUNS
from .resources import set_worker_thread_env, nice_self

COHORT = "MAIN"
MAIN_RUNS = RUNS / "main"
INST_DIR = DATA / "instances" / "main"
ENT_DIR = DATA / "entropy" / "main"
REF_DIR = DATA / "references" / "main"


def _rss_kib():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def result_path(spec):
    kind = spec["task"]
    n, seed = int(spec["N"]), int(spec["seed"])
    if kind == "entropy":
        return MAIN_RUNS / f"entropy_N{n}_seed{seed}.json"
    if kind == "spsa":
        return MAIN_RUNS / f"spsa_N{n}_seed{seed}_L{spec['L']}_{spec['method']}.json"
    if kind == "bfgs":
        return MAIN_RUNS / f"bfgs_N{n}_seed{seed}_L{spec['L']}_{spec['method']}_start{spec['start_index']}.json"
    raise ValueError(kind)


def load_existing(spec):
    path = result_path(spec)
    if not path.exists():
        return None
    try:
        rec = __import__("json").loads(path.read_text())
    except Exception:
        return None
    if rec.get("error") or rec.get("verdict") == "FAIL" or not rec.get("wall_s"):
        return None
    rec["result_file"] = str(path)
    rec["reused"] = True
    return rec


def _save_instance_and_scan(n, seed, report, J, gs):
    INST_DIR.mkdir(parents=True, exist_ok=True)
    ENT_DIR.mkdir(parents=True, exist_ok=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    iu = np.triu_indices(n, 1)
    inst = dict(
        N=n, seed=int(seed), cohort=COHORT,
        coupling_variance="1/N",
        hamiltonian=P.HAMILTONIAN,
        J_upper=J[iu].tolist(),
        b=report["peak"]["b"],
        E0=report["peak"]["E0"],
        gap=report["peak"]["gap"],
        even_gap=report["peak"]["even_gap"],
        S=report["peak"]["S"],
        label=report["label"],
        valid_for_fits=report["valid_for_fits"],
        gs_sha256=hashlib.sha256(np.ascontiguousarray(gs).tobytes()).hexdigest(),
    )
    inst_path = INST_DIR / f"N{n}_seed{seed}.json"
    scan_path = ENT_DIR / f"N{n}_seed{seed}_scan.json"
    ref_path = REF_DIR / f"N{n}_seed{seed}_reference.npz"
    save_json(inst_path, inst)
    save_json(scan_path, report)
    np.savez_compressed(ref_path, J=J, gs=gs, E0=report["peak"]["E0"], b=report["peak"]["b"],
                        gap=report["peak"]["gap"], even_gap=report["peak"]["even_gap"],
                        S=report["peak"]["S"], seed=seed, N=n)
    return dict(instance=str(inst_path), scan=str(scan_path), reference=str(ref_path),
                reference_sha256=sha256_file(ref_path), reference_bytes=ref_path.stat().st_size)


def load_main_model(n, seed):
    ref = REF_DIR / f"N{n}_seed{seed}_reference.npz"
    z = np.load(ref)
    J = z["J"]
    m = Model(n, seed, float(z["b"]), J=J, compute_gs=False)
    assert np.array_equal(m.J, J)
    m.load_gs(z["gs"], float(z["E0"]), gap=float(z["gap"]))
    return m, z


def task_entropy(spec):
    set_worker_thread_env(); nice_self(10)
    n, seed = int(spec["N"]), int(spec["seed"])
    t0 = time.time(); c0 = time.process_time()
    rss0 = _rss_kib()
    J = make_J(n, seed)
    m = Model(n, seed, 1.0, J=J, compute_gs=False)
    progress_path = LOGS / f"entropy_scan_main_N{n}_seed{seed}.jsonl"
    report, gs = scan_bstar(m, progress_path=progress_path)
    paths = _save_instance_and_scan(n, seed, report, J, gs)
    m2 = Model(n, seed, report["peak"]["b"], J=J, compute_gs=False)
    Echk = m2.energy(gs)
    out = dict(
        task="entropy", N=n, seed=seed, cohort=COHORT, index=spec.get("index"),
        b=report["peak"]["b"], S=report["peak"]["S"], E0=report["peak"]["E0"],
        gap=report["peak"]["gap"], even_gap=report["peak"]["even_gap"],
        label=report["label"], valid_for_fits=report["valid_for_fits"],
        residual=report["peak"]["residual"], parity=report["peak"]["parity"],
        E_apply_H_abs_err=abs(Echk - report["peak"]["E0"]),
        scan_resource=report["scan_resource"],
        wall_s=time.time() - t0, CPU_s=time.process_time() - c0,
        peak_RSS_kib=_rss_kib(), rss_before_kib=rss0,
        pid=os.getpid(), paths=paths, threads_env=1,
        coupling_variance="1/N",
    )
    dest = result_path(spec)
    MAIN_RUNS.mkdir(parents=True, exist_ok=True)
    save_json(dest, out)
    out["result_file"] = str(dest)
    return out


def task_spsa(spec):
    set_worker_thread_env(); nice_self(10)
    n, seed, L, method = int(spec["N"]), int(spec["seed"]), int(spec["L"]), spec["method"]
    t0 = time.time(); c0 = time.process_time()
    m, z = load_main_model(n, seed)
    plus_E = m.energy(m.plus)
    plus_F = m.fidelity(m.plus)
    out = spsa_optimize(m, method, L)
    compact_log = [(q["query"], q["kind"], q["E"]) for q in out["query_log"]]
    out.update(
        task="spsa", N=n, seed=seed, cohort=COHORT, index=spec.get("index"),
        b=float(z["b"]), E0=float(z["E0"]), plus_E=plus_E, plus_F=plus_F,
        energy_error=out["best_E"] - float(z["E0"]),
        energy_error_over_N=(out["best_E"] - float(z["E0"])) / n,
        wall_s=time.time() - t0, CPU_s=time.process_time() - c0,
        peak_RSS_kib=_rss_kib(), pid=os.getpid(),
        query_log_sha256=hashlib.sha256(repr(compact_log).encode()).hexdigest(),
        coupling_variance="1/N",
        valid_for_fits=bool(spec.get("valid_for_fits", True)),
    )
    dest = result_path(spec)
    MAIN_RUNS.mkdir(parents=True, exist_ok=True)
    # Drop the full query log from the on-disk main record (hash retained).
    stored = {k: out[k] for k in out if k not in ("query_log", "readout_candidates", "trace")}
    save_json(dest, stored)
    summary = {k: out[k] for k in out if k not in ("query_log", "readout_candidates", "trace", "initial_points", "x")}
    summary.update(result_file=str(dest), best_E=out["best_E"], F=out["F"], calls=out["calls"],
                   x_sha256=x_sha256(out["x"]))
    return summary


def task_bfgs(spec):
    set_worker_thread_env(); nice_self(10)
    n, seed, L, method, k = int(spec["N"]), int(spec["seed"]), int(spec["L"]), spec["method"], int(spec["start_index"])
    t0 = time.time(); c0 = time.process_time()
    m, z = load_main_model(n, seed)
    st = make_start(seed, L, method, k)
    out = run_start(m, method, L, st)
    out.update(
        task="bfgs", N=n, seed=seed, cohort=COHORT, index=spec.get("index"),
        L=L, method=method, b=float(z["b"]),
        wall_s=time.time() - t0, CPU_s=time.process_time() - c0,
        peak_RSS_kib=_rss_kib(), pid=os.getpid(),
        coupling_variance="1/N",
        valid_for_fits=bool(spec.get("valid_for_fits", True)),
    )
    dest = result_path(spec)
    MAIN_RUNS.mkdir(parents=True, exist_ok=True)
    stored = {k2: out[k2] for k2 in out if k2 not in ("final_grad",)}
    # keep x (small) for later energy-selection across 16 starts
    save_json(dest, stored)
    summary = {k2: out[k2] for k2 in out if k2 not in ("x", "final_grad")}
    summary.update(result_file=str(dest), x_sha256=out["x_sha256"])
    return summary


def dispatch(spec):
    kind = spec["task"]
    try:
        if kind == "entropy":
            return task_entropy(spec)
        if kind == "spsa":
            return task_spsa(spec)
        if kind == "bfgs":
            return task_bfgs(spec)
        raise ValueError(kind)
    except Exception as ex:
        rec = dict(task=kind, spec=spec, error=repr(ex), traceback=traceback.format_exc(),
                   peak_RSS_kib=_rss_kib(), pid=os.getpid(), wall_s=None, verdict="FAIL")
        try:
            dest = result_path(spec)
            MAIN_RUNS.mkdir(parents=True, exist_ok=True)
            save_json(dest.with_name(dest.stem + "_FAIL.json"), rec)
        except Exception:
            pass
        return rec


def planned_main_tasks():
    """All approved main cells: 200 instances x N x L x methods, frozen seeds, no extra arms."""
    entropy, spsa, bfgs = [], [], []
    for n in P.NS:
        for i in range(P.N_INSTANCES):
            seed = P.main_seed(n, i)
            entropy.append(dict(task="entropy", N=n, seed=seed, index=i))
            for L in P.DEPTHS:
                for method in P.METHODS:
                    spsa.append(dict(task="spsa", N=n, seed=seed, index=i, L=L, method=method))
                    for k in range(P.BFGS_STARTS):
                        bfgs.append(dict(task="bfgs", N=n, seed=seed, index=i, L=L, method=method, start_index=k))
    return dict(entropy=entropy, spsa=spsa, bfgs=bfgs)
