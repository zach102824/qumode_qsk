#!/usr/bin/env python3
"""Approved main 200-instance campaign. Frozen protocol, resume-safe, conservative admission.

Does not launch if a live phase-1 lock is held. Never signals historical jobs.
N=21 is single-worker. Variance 1/N. No extra arms, no half-variance, no SPSA-warm.
"""
from __future__ import annotations
import os, sys, time, traceback, fcntl, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.resources import set_worker_thread_env, allocated_cpus, memory_headroom_kib
set_worker_thread_env()

from src import protocol as P
from src.io_util import save_json, now_shanghai, append_jsonl, sha256_file, read_json
from src.paths import LOGS, PROTOCOL, AUDITS, STATUS, HEARTBEAT
from src.scheduler import AdmissionPool
from src.main_jobs import (
    planned_main_tasks, dispatch, load_existing, result_path,
    MAIN_RUNS, INST_DIR, COHORT,
)

LOCK = ROOT / "main.lock"
MAIN_STATUS = ROOT / "MAIN_STATUS.json"
PROGRESS = LOGS / "main_progress.jsonl"
SCHED_LOG = LOGS / "main_scheduler.jsonl"
PROTOCOL_SHA_CANONICAL = "f32a96d4af337525edbbeea903b848d9b8c6163af3efa214f4ca71cd143cbb81"

# Measured pilot peak RSS (KiB) by (task, N). Optimizer classes are NOT inherited from entropy.
PILOT_RSS = {
    ("entropy", 6): 66816, ("entropy", 9): 67424, ("entropy", 12): 72080,
    ("entropy", 15): 133200, ("entropy", 18): 617416, ("entropy", 21): 1933780,
    ("spsa", 6): 62184, ("spsa", 9): 62184, ("spsa", 12): 60548,
    ("spsa", 15): 114096, ("spsa", 18): 114096, ("spsa", 21): 470940,
    ("bfgs", 6): 59208, ("bfgs", 9): 61364, ("bfgs", 12): 61364,
    ("bfgs", 15): 189484, ("bfgs", 18): 189484, ("bfgs", 21): 977972,
}

# Conservative caps: stability over max-CPU. N=21 always 1.
CONFIG_MAX = {
    ("entropy", 6): 4, ("entropy", 9): 4, ("entropy", 12): 3,
    ("entropy", 15): 2, ("entropy", 18): 1, ("entropy", 21): 1,
    ("spsa", 6): 4, ("spsa", 9): 4, ("spsa", 12): 3,
    ("spsa", 15): 2, ("spsa", 18): 2, ("spsa", 21): 1,
    ("bfgs", 6): 4, ("bfgs", 9): 4, ("bfgs", 12): 3,
    ("bfgs", 15): 2, ("bfgs", 18): 2, ("bfgs", 21): 1,
}


def class_rss_kib(task, n):
    measured = PILOT_RSS[(task, int(n))]
    return int(max(measured * 1.5, 80_000))


def config_max(task, n):
    return int(CONFIG_MAX[(task, int(n))])


def reserve_kib(task, n):
    if int(n) >= 21 or (task == "entropy" and int(n) >= 18):
        return 1024 * 1024
    return 512 * 1024


def update_status(**kw):
    rec = dict(
        workstreamID="qumode_qsk_main",
        authorization=["pt3487u", "main_approved_after_pilot_eta"],
        cohort=COHORT,
        coupling_variance="1/N",
        shanghai=now_shanghai(),
        pid=os.getpid(),
    )
    if MAIN_STATUS.exists():
        try:
            rec.update(json.loads(MAIN_STATUS.read_text()))
        except Exception:
            pass
    rec.update(kw)
    rec["shanghai"] = now_shanghai()
    rec["pid"] = os.getpid()
    save_json(MAIN_STATUS, rec)
    save_json(STATUS, rec)
    save_json(HEARTBEAT, dict(shanghai=now_shanghai(), pid=os.getpid(),
                              stage=rec.get("stage"), state=rec.get("state"),
                              campaign="main"))
    return rec


def verify_frozen_protocol():
    proto = PROTOCOL / "PROTOCOL_FROZEN.json"
    audits = AUDITS / "AUDIT_SUMMARY.json"
    if not proto.exists():
        raise SystemExit("missing PROTOCOL_FROZEN.json")
    d = read_json(proto)
    body = {k: v for k, v in d.items() if k != "protocol_sha256"}
    can = json.dumps(body, sort_keys=True, separators=(",", ":"))
    import hashlib
    h = hashlib.sha256(can.encode()).hexdigest()
    if h != PROTOCOL_SHA_CANONICAL or d.get("protocol_sha256") != PROTOCOL_SHA_CANONICAL:
        raise SystemExit(f"protocol hash mismatch: {h}")
    sci = d.get("science") or {}
    if sci.get("coupling_variance") != "1/N":
        raise SystemExit("refusing: coupling_variance is not 1/N")
    if sci.get("n_instances_per_N") != 200 or tuple(sci.get("depths", [])) != (1, 2, 3, 4, 5, 6):
        raise SystemExit("refusing: frozen N/L/instance counts changed")
    if not audits.exists() or read_json(audits).get("verdict") != "PASS":
        raise SystemExit("refusing: audits are not PASS")
    return d, h


def phase1_still_running():
    lock = ROOT / "phase1.lock"
    if not lock.exists():
        return False
    try:
        pid = int(lock.read_text().strip())
    except Exception:
        return False
    return Path(f"/proc/{pid}").exists()


def attach_valid_flag(spec):
    inst = INST_DIR / f"N{spec['N']}_seed{spec['seed']}.json"
    if inst.exists():
        try:
            rec = read_json(inst)
            spec = dict(spec)
            spec["valid_for_fits"] = bool(rec.get("valid_for_fits", False))
            spec["b"] = rec.get("b")
            spec["label"] = rec.get("label")
        except Exception:
            pass
    return spec


def run_wave(name, tasks, task_kind, n, reused_note=None):
    if not tasks:
        return [], dict(wave=name, n=0, skipped_empty=True, reused_note=reused_note)
    rss = class_rss_kib(task_kind, n)
    mx = config_max(task_kind, n)
    rsv = reserve_kib(task_kind, n)
    update_status(state="running", stage=name, n_tasks=len(tasks),
                  class_rss_kib=rss, config_max=mx, reserve_kib=rsv, N=n, task_kind=task_kind)
    pool = AdmissionPool(class_rss_kib=rss, config_max=mx, reserve_kib=rsv,
                         log_path=SCHED_LOG)
    t0 = time.time()
    rows = pool.map(dispatch, tasks, max_pool=mx)
    n_err = 0
    for r in rows:
        if r is None:
            n_err += 1
            continue
        if r.get("error") or r.get("verdict") == "FAIL":
            n_err += 1
        append_jsonl(PROGRESS, dict(event="task_done", wave=name, rec={k: r.get(k) for k in
                     ("task", "N", "seed", "index", "L", "method", "start_index", "wall_s", "CPU_s",
                      "peak_RSS_kib", "best_E", "F", "label", "valid_for_fits", "error", "result_file",
                      "b", "S")},
                     shanghai=now_shanghai()))
    return rows, dict(wave=name, wall_s=time.time() - t0, n=len(tasks), n_error=n_err,
                      class_rss_kib=rss, config_max=mx, reserve_kib=rsv,
                      reused_note=reused_note)


def split_resume(tasks):
    reused, todo = [], []
    for spec in tasks:
        ex = load_existing(spec)
        if ex is not None:
            reused.append(ex)
        else:
            todo.append(spec)
    return reused, todo


def main():
    MAIN_RUNS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    if phase1_still_running():
        update_status(state="blocked", stage="phase1_still_running")
        raise SystemExit("phase-1 process still holds work; not starting main")

    lockf = LOCK.open("w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        update_status(state="blocked", stage="lock_held", error="main.lock is held")
        raise SystemExit("main.lock is held")
    lockf.write(str(os.getpid())); lockf.flush()

    t_all = time.time()
    proto, proto_sha = verify_frozen_protocol()
    started = now_shanghai()
    if MAIN_STATUS.exists():
        try:
            started = json.loads(MAIN_STATUS.read_text()).get("started_shanghai") or started
        except Exception:
            pass
    update_status(
        state="running", stage="start", started_shanghai=started, resume=True,
        protocol=str(PROTOCOL / "PROTOCOL_FROZEN.json"), protocol_sha256=proto_sha,
        note="frozen protocol reused; conservative admission; N=21 single-worker; 1/N; no extra arms",
        allocated_cpus=allocated_cpus(),
        memory=memory_headroom_kib(),
    )

    plan = planned_main_tasks()
    wave_meta = []
    counts = dict(entropy_done=0, spsa_done=0, bfgs_done=0,
                  entropy_todo=0, spsa_todo=0, bfgs_todo=0, errors=0)

    # Per-N waves: entropy first (needed for B* and GS), then SPSA, then BFGS.
    # Smaller N first so complete panels arrive before the long N=21 campaign.
    for n in P.NS:
        for kind, all_tasks in (("entropy", plan["entropy"]), ("spsa", plan["spsa"]), ("bfgs", plan["bfgs"])):
            tasks = [t for t in all_tasks if t["N"] == n]
            if kind != "entropy":
                tasks = [attach_valid_flag(t) for t in tasks]
            reused, todo = split_resume(tasks)
            counts[f"{kind}_done"] += len(reused)
            counts[f"{kind}_todo"] += len(todo)
            name = f"{kind}_N{n}"
            if reused and not todo:
                wave_meta.append(dict(wave=name, reused=len(reused), todo=0, skipped_complete=True))
                update_status(stage=f"{name}_reused", **counts)
                continue
            if reused:
                wave_meta.append(dict(wave=name, reused=len(reused), todo=len(todo)))
            if todo:
                rows, meta = run_wave(name, todo, kind, n)
                wave_meta.append(meta)
                n_ok = sum(1 for r in rows if r and not r.get("error") and r.get("verdict") != "FAIL")
                n_err = len(rows) - n_ok
                counts[f"{kind}_done"] += n_ok
                counts[f"{kind}_todo"] -= n_ok
                counts["errors"] += n_err
                update_status(stage=f"{name}_done", **counts, last_wave=meta)

    wall = time.time() - t_all
    verdict = "PASS" if counts["errors"] == 0 else "PARTIAL"
    update_status(state="main_complete", stage="complete", verdict=verdict,
                  wall_s=wall, finished_shanghai=now_shanghai(), waves=wave_meta, **counts)
    print(f"MAIN {verdict} wall_s={wall:.1f} counts={counts}", flush=True)
    return 0 if verdict in ("PASS", "PARTIAL") else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as ex:
        try:
            update_status(state="failed", stage="exception", error=repr(ex),
                          traceback=traceback.format_exc())
        except Exception:
            pass
        raise
