"""Disjoint resource pilot. No headline/main seeds. Measures RSS and wall times for ETA."""
from __future__ import annotations
import os, time, resource, traceback, hashlib
from pathlib import Path
import numpy as np

from . import protocol as P
from .hamiltonian import Model, make_J
from .entropy import scan_bstar
from .spsa import optimize as spsa_optimize
from .bfgs import make_start, run_start
from .io_util import save_json, now_shanghai, sha256_file, append_jsonl, x_sha256
from .paths import PILOT_RUNS, DATA, LOGS
from .resources import set_worker_thread_env, nice_self, memory_headroom_kib

PROGRESS = LOGS / "pilot_progress.jsonl"


def _rss_kib():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _save_instance_and_scan(n, seed, report, J, gs):
    inst_dir = DATA / "instances" / "pilot"
    ent_dir = DATA / "entropy" / "pilot"
    ref_dir = DATA / "references" / "pilot"
    inst_dir.mkdir(parents=True, exist_ok=True)
    ent_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    iu = np.triu_indices(n, 1)
    inst = dict(
        N=n, seed=int(seed), cohort="PILOT_DISJOINT",
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
    inst_path = inst_dir / f"N{n}_seed{seed}.json"
    scan_path = ent_dir / f"N{n}_seed{seed}_scan.json"
    ref_path = ref_dir / f"N{n}_seed{seed}_reference.npz"
    save_json(inst_path, inst)
    save_json(scan_path, report)
    np.savez_compressed(ref_path, J=J, gs=gs, E0=report["peak"]["E0"], b=report["peak"]["b"],
                        gap=report["peak"]["gap"], even_gap=report["peak"]["even_gap"],
                        S=report["peak"]["S"], seed=seed, N=n)
    return dict(instance=str(inst_path), scan=str(scan_path), reference=str(ref_path),
                reference_sha256=sha256_file(ref_path), reference_bytes=ref_path.stat().st_size)


def load_pilot_model(n, seed):
    ref = DATA / "references" / "pilot" / f"N{n}_seed{seed}_reference.npz"
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
    progress_path = LOGS / f"entropy_scan_N{n}_seed{seed}.jsonl"
    report, gs = scan_bstar(m, progress_path=progress_path)
    paths = _save_instance_and_scan(n, seed, report, J, gs)
    # Cross-check energy of stored GS under apply_H at B*.
    m2 = Model(n, seed, report["peak"]["b"], J=J, compute_gs=False)
    Echk = m2.energy(gs)
    out = dict(
        task="entropy", N=n, seed=seed, cohort="PILOT_DISJOINT",
        b=report["peak"]["b"], S=report["peak"]["S"], E0=report["peak"]["E0"],
        gap=report["peak"]["gap"], even_gap=report["peak"]["even_gap"],
        label=report["label"], valid_for_fits=report["valid_for_fits"],
        residual=report["peak"]["residual"], parity=report["peak"]["parity"],
        E_apply_H_abs_err=abs(Echk - report["peak"]["E0"]),
        scan_resource=report["scan_resource"],
        wall_s=time.time() - t0, CPU_s=time.process_time() - c0,
        peak_RSS_kib=_rss_kib(), rss_before_kib=rss0,
        pid=os.getpid(), paths=paths, threads_env=1,
    )
    dest = PILOT_RUNS / f"entropy_N{n}_seed{seed}.json"
    save_json(dest, out)
    out["result_file"] = str(dest)
    return out


def task_spsa(spec):
    set_worker_thread_env(); nice_self(10)
    n, seed, L, method = int(spec["N"]), int(spec["seed"]), int(spec["L"]), spec["method"]
    t0 = time.time(); c0 = time.process_time()
    m, z = load_pilot_model(n, seed)
    plus_E = m.energy(m.plus)
    plus_F = m.fidelity(m.plus)
    out = spsa_optimize(m, method, L)
    out.update(
        task="spsa", N=n, seed=seed, cohort="PILOT_DISJOINT",
        b=float(z["b"]), E0=float(z["E0"]), plus_E=plus_E, plus_F=plus_F,
        energy_error=out["best_E"] - float(z["E0"]),
        energy_error_over_N=(out["best_E"] - float(z["E0"])) / n,
        wall_s=time.time() - t0, CPU_s=time.process_time() - c0,
        peak_RSS_kib=_rss_kib(), pid=os.getpid(),
        query_log_sha256=hashlib.sha256(
            # do not drop the log from disk; hash a compact role+E stream
            repr([(z["query"], z["kind"], z["E"]) for z in out["query_log"]]).encode()
        ).hexdigest(),
    )
    # Store the full query log in the run file (pilot is small).
    dest = PILOT_RUNS / f"spsa_N{n}_seed{seed}_L{L}_{method}.json"
    save_json(dest, out)
    # Return a compact summary for the parent process (avoid huge IPC).
    summary = {k: out[k] for k in out if k not in ("query_log", "readout_candidates", "trace", "initial_points", "x")}
    summary.update(result_file=str(dest), best_E=out["best_E"], F=out["F"], calls=out["calls"],
                   x_sha256=x_sha256(out["x"]))
    return summary


def task_bfgs(spec):
    set_worker_thread_env(); nice_self(10)
    n, seed, L, method, k = int(spec["N"]), int(spec["seed"]), int(spec["L"]), spec["method"], int(spec["start_index"])
    t0 = time.time(); c0 = time.process_time()
    m, z = load_pilot_model(n, seed)
    st = make_start(seed, L, method, k)
    out = run_start(m, method, L, st)
    out.update(
        task="bfgs", N=n, seed=seed, cohort="PILOT_DISJOINT",
        L=L, method=method, b=float(z["b"]),
        wall_s=time.time() - t0, CPU_s=time.process_time() - c0,
        peak_RSS_kib=_rss_kib(), pid=os.getpid(),
    )
    dest = PILOT_RUNS / f"bfgs_N{n}_seed{seed}_L{L}_{method}_start{k}.json"
    save_json(dest, out)
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
        return dict(task=kind, spec=spec, error=repr(ex), traceback=traceback.format_exc(),
                    peak_RSS_kib=_rss_kib(), pid=os.getpid(), wall_s=None, verdict="FAIL")


def planned_tasks():
    """Fixed small disjoint pilot. Entropy for every N; SPSA/BFGS only after those refs exist.

    SPSA: N=6 L=1 nov3 and qaoa; N=6 L=6 nov3; N=12,18 L=1 nov3; N=21 L=1 nov3 if scheduled.
    BFGS: N=6 L=1 nov3 starts 0,1 and (optionally) 0..15 later if cheap; N=12/18/21 L=1 nov3 start 0.
    """
    ent = [dict(task="entropy", N=n, seed=P.pilot_seed(n, 0)) for n in P.NS]
    spsa = [
        dict(task="spsa", N=6, seed=P.pilot_seed(6, 0), L=1, method="nov3"),
        dict(task="spsa", N=6, seed=P.pilot_seed(6, 0), L=1, method="qaoa"),
        dict(task="spsa", N=6, seed=P.pilot_seed(6, 0), L=6, method="nov3"),
        dict(task="spsa", N=12, seed=P.pilot_seed(12, 0), L=1, method="nov3"),
        dict(task="spsa", N=18, seed=P.pilot_seed(18, 0), L=1, method="nov3"),
        dict(task="spsa", N=21, seed=P.pilot_seed(21, 0), L=1, method="nov3"),
    ]
    bfgs = [
        dict(task="bfgs", N=6, seed=P.pilot_seed(6, 0), L=1, method="nov3", start_index=0),
        dict(task="bfgs", N=6, seed=P.pilot_seed(6, 0), L=1, method="nov3", start_index=1),
        dict(task="bfgs", N=12, seed=P.pilot_seed(12, 0), L=1, method="nov3", start_index=0),
        dict(task="bfgs", N=18, seed=P.pilot_seed(18, 0), L=1, method="nov3", start_index=0),
        dict(task="bfgs", N=21, seed=P.pilot_seed(21, 0), L=1, method="nov3", start_index=0),
    ]
    return dict(entropy=ent, spsa=spsa, bfgs=bfgs)
