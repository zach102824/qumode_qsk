#!/usr/bin/env python3
"""Cumulative 10-per-N round campaign. Frozen science, resume-safe, mixed admission.

Round r = indices 10*(r-1)..10*r-1, ALL N, six depths, four arms. The next round
starts only after every N has finished the current official round. Up to 8 cores
total, N=21 cap 1, RAM-gated. Surplus completed N=6 data is retained off official
summaries. Reuses existing protocol-matching results (including all N=6 0-9).
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
from src.scheduler import MixedAdmissionPool
from src.rounds import (
    plan_round, interleave_entropy, interleave_optimize, attach_valid_flag,
    split_resume, summarize_round, completeness, class_rss_kib, config_max,
    admit_need_kib, is_heavy, interleave_entropy, interleave_optimize, ROUND_SIZE, N_ROUNDS,
)
from src.main_jobs import dispatch, MAIN_RUNS, COHORT

LOCK = ROOT / "rounds.lock"
MAIN_LOCK = ROOT / "main.lock"
ROUND_STATUS = ROOT / "ROUND_STATUS.json"
PROGRESS = LOGS / "rounds_progress.jsonl"
SCHED_LOG = LOGS / "rounds_scheduler.jsonl"
PROTOCOL_SHA_CANONICAL = "f32a96d4af337525edbbeea903b848d9b8c6163af3efa214f4ca71cd143cbb81"
KERNEL_AUDIT = AUDITS / "KERNEL_AUDIT.json"
OLD_MAIN_PID = 2726767
AUTHORIZATION = ["pt3496u", "pt3497u", "pt3498u"]
HARD_CPU = 8
RESERVE_KIB = 512 * 1024  # responsive headroom; N=21/N=18-entropy use admit_need_kib


def update_status(**kw):
    rec = dict(
        workstreamID="qumode_qsk_rounds",
        authorization=AUTHORIZATION,
        cohort=COHORT,
        coupling_variance="1/N",
        scheduler="cumulative_rounds_10_per_N",
        shanghai=now_shanghai(),
        pid=os.getpid(),
    )
    if ROUND_STATUS.exists():
        try:
            rec.update(json.loads(ROUND_STATUS.read_text()))
        except Exception:
            pass
    rec.update(kw)
    rec["shanghai"] = now_shanghai()
    rec["pid"] = os.getpid()
    rec["authorization"] = AUTHORIZATION
    save_json(ROUND_STATUS, rec)
    save_json(STATUS, rec)
    save_json(HEARTBEAT, dict(shanghai=now_shanghai(), pid=os.getpid(),
                              stage=rec.get("stage"), state=rec.get("state"),
                              campaign="rounds", current_round=rec.get("current_round")))
    return rec


def verify_frozen_protocol_and_audit():
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
        raise SystemExit("refusing: phase-1 audits are not PASS")
    if not KERNEL_AUDIT.exists() or read_json(KERNEL_AUDIT).get("verdict") != "PASS":
        raise SystemExit("refusing: independent kernel audit is not PASS")
    return d, h, read_json(KERNEL_AUDIT)


def pid_alive(pid):
    try:
        return Path(f"/proc/{int(pid)}").exists()
    except Exception:
        return False


def lock_held_by_live_pid(path):
    if not path.exists():
        return False
    try:
        pid = int(path.read_text().strip().split()[0])
    except Exception:
        return False
    return pid_alive(pid)


def historical_untouched_note():
    return "never signal historical/SIGSTOP jobs (af_tfim and others); own campaign only"


def run_stage(name, tasks, round_id):
    if not tasks:
        return [], dict(stage=name, n=0, skipped_empty=True, round=round_id)
    update_status(state="running", stage=name, current_round=round_id, n_tasks=len(tasks),
                  hard_cpu_cap=HARD_CPU, reserve_kib=RESERVE_KIB, n21_cap=1)
    pool = MixedAdmissionPool(
        rss_fn=class_rss_kib, cap_fn=config_max,
        need_fn=admit_need_kib, heavy_fn=is_heavy,
        reserve_kib=RESERVE_KIB, hard_cpu_cap=HARD_CPU, n21_cap=1,
        log_path=SCHED_LOG,
    )
    t0 = time.time()
    rows = pool.map(dispatch, tasks, max_pool=HARD_CPU)
    n_err = 0
    for r in rows:
        if r is None or r.get("error") or r.get("verdict") == "FAIL":
            n_err += 1
        if r is not None:
            append_jsonl(PROGRESS, dict(event="task_done", stage=name, round=round_id,
                                        rec={k: r.get(k) for k in
                                             ("task", "N", "seed", "index", "L", "method",
                                              "wall_s", "CPU_s", "peak_RSS_kib", "best_E", "F",
                                              "label", "valid_for_fits", "error", "result_file",
                                              "b", "S", "n_reused", "n_ran", "selected")},
                                        shanghai=now_shanghai()))
    return rows, dict(stage=name, wall_s=time.time() - t0, n=len(tasks), n_error=n_err,
                      round=round_id)


def entropy_ready(spec):
    if spec.get("task") == "entropy":
        return True
    from src.main_jobs import load_existing
    return load_existing(dict(task="entropy", N=spec["N"], seed=spec["seed"],
                              index=spec.get("index"))) is not None


def mixed_priority(tasks):
    """Large-N entropy first; optimizer cells follow so first-fit can fill idle cores."""
    ent = interleave_entropy([t for t in tasks if t["task"] == "entropy"])
    opt = interleave_optimize([t for t in tasks if t["task"] != "entropy"])
    return ent + opt


def run_one_round(round_id, started_all):
    plan = plan_round(round_id)
    e_tasks = [attach_valid_flag(t) for t in plan["entropy"]]
    e_reused, e_todo = split_resume(e_tasks)
    s_tasks = [attach_valid_flag(t) for t in plan["spsa"]]
    s_reused, s_todo = split_resume(s_tasks)
    b_tasks = [attach_valid_flag(t) for t in plan["bfgs_pack"]]
    b_reused, b_todo = split_resume(b_tasks)
    counts = dict(
        entropy_reused=len(e_reused), entropy_todo=len(e_todo), entropy_done=len(e_reused),
        spsa_reused=len(s_reused), spsa_todo=len(s_todo), spsa_done=len(s_reused),
        bfgs_reused=len(b_reused), bfgs_todo=len(b_todo), bfgs_done=len(b_reused),
        errors=0,
    )
    todo = mixed_priority(e_todo + s_todo + b_todo)
    update_status(stage=f"round{round_id}_mixed", current_round=round_id,
                  n_tasks=len(todo), **counts)
    if todo:
        update_status(state="running", stage=f"round{round_id}_mixed", current_round=round_id,
                      n_tasks=len(todo), hard_cpu_cap=HARD_CPU, reserve_kib=RESERVE_KIB, n21_cap=1, **counts)
        pool = MixedAdmissionPool(
            rss_fn=class_rss_kib, cap_fn=config_max,
            need_fn=admit_need_kib, heavy_fn=is_heavy, ready_fn=entropy_ready,
            reserve_kib=RESERVE_KIB, hard_cpu_cap=HARD_CPU, n21_cap=1,
            log_path=SCHED_LOG,
        )
        t0 = time.time()
        rows = pool.map(dispatch, todo, max_pool=HARD_CPU)
        n_err = 0
        n_e = n_s = n_b = 0
        for r in rows:
            if r is None or r.get("error") or r.get("verdict") == "FAIL":
                n_err += 1
            elif r.get("task") == "entropy":
                n_e += 1
            elif r.get("task") == "spsa":
                n_s += 1
            elif r.get("task") == "bfgs_pack":
                n_b += 1
            if r is not None:
                append_jsonl(PROGRESS, dict(event="task_done", stage=f"round{round_id}_mixed",
                                            round=round_id,
                                            rec={k: r.get(k) for k in
                                                 ("task", "N", "seed", "index", "L", "method",
                                                  "wall_s", "CPU_s", "peak_RSS_kib", "best_E", "F",
                                                  "label", "valid_for_fits", "error", "result_file",
                                                  "b", "S", "n_reused", "n_ran", "selected")},
                                            shanghai=now_shanghai()))
        counts["entropy_done"] += n_e
        counts["entropy_todo"] -= n_e
        counts["spsa_done"] += n_s
        counts["spsa_todo"] -= n_s
        counts["bfgs_done"] += n_b
        counts["bfgs_todo"] -= n_b
        counts["errors"] += n_err
        update_status(stage=f"round{round_id}_mixed_done", current_round=round_id,
                      last_stage=dict(stage=f"round{round_id}_mixed", wall_s=time.time()-t0,
                                      n=len(todo), n_error=n_err), **counts)

    comp = completeness(round_id)
    summary = None
    if comp["all_N_complete"]:
        summary = summarize_round(round_id)
        update_status(stage=f"round{round_id}_summarized", current_round=round_id,
                      summary_file=summary.get("summary_file"), completeness=comp, **counts)
    else:
        update_status(stage=f"round{round_id}_incomplete", current_round=round_id,
                      completeness=comp, **counts)
    return counts, comp, summary


def main():
    MAIN_RUNS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    if pid_alive(OLD_MAIN_PID):
        update_status(state="blocked", stage="old_main_still_alive", old_main_pid=OLD_MAIN_PID)
        raise SystemExit(f"refusing: drained main pid {OLD_MAIN_PID} is still alive")
    if lock_held_by_live_pid(MAIN_LOCK):
        update_status(state="blocked", stage="main_lock_held")
        raise SystemExit("refusing: live main.lock is held")

    lockf = LOCK.open("w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        update_status(state="blocked", stage="lock_held", error="rounds.lock is held")
        raise SystemExit("rounds.lock is held")
    lockf.write(str(os.getpid())); lockf.flush()

    t_all = time.time()
    proto, proto_sha, kaudit = verify_frozen_protocol_and_audit()
    started = now_shanghai()
    if ROUND_STATUS.exists():
        try:
            started = json.loads(ROUND_STATUS.read_text()).get("started_shanghai") or started
        except Exception:
            pass
    eta = None
    eta_path = ROOT / "ROUND1_ETA.json"
    if eta_path.exists():
        try:
            eta = read_json(eta_path)
        except Exception:
            eta = None
    update_status(
        state="running", stage="start", started_shanghai=started, resume=True,
        protocol=str(PROTOCOL / "PROTOCOL_FROZEN.json"), protocol_sha256=proto_sha,
        kernel_audit_verdict=kaudit.get("verdict"),
        note="cumulative 10-per-N rounds; reuse matching results; 8 cores total; N=21 cap 1; 1/N; no extra arms",
        historical=historical_untouched_note(),
        allocated_cpus=allocated_cpus(),
        memory=memory_headroom_kib(RESERVE_KIB),
        eta_file=str(eta_path) if eta else None,
        start_round=1,
    )

    # Run rounds in order. Next round is admitted only after all N finish the current official round.
    last_comp = None
    for r in range(1, N_ROUNDS + 1):
        comp0 = completeness(r)
        if comp0["all_N_complete"]:
            if not (ROOT / "summaries" / f"round_{r:02d}_summary.json").exists():
                summarize_round(r)
            update_status(stage=f"round{r}_already_complete", current_round=r, completeness=comp0)
            last_comp = comp0
            continue
        counts, comp, summary = run_one_round(r, started)
        last_comp = comp
        if not comp["all_N_complete"] or counts.get("errors", 0):
            update_status(state="partial", stage=f"round{r}_stopped_incomplete_or_errors",
                          current_round=r, completeness=comp, counts=counts,
                          wall_s=time.time() - t_all, finished_shanghai=now_shanghai())
            print(f"ROUNDS PARTIAL at round {r} errors={counts.get('errors')} missing={comp['n_missing']}", flush=True)
            return 1
        update_status(stage=f"round{r}_complete", current_round=r, completeness=comp,
                      summary_file=(summary or {}).get("summary_file"))

    wall = time.time() - t_all
    update_status(state="rounds_complete", stage="all_20_rounds_complete", verdict="PASS",
                  wall_s=wall, finished_shanghai=now_shanghai(), completeness=last_comp)
    print(f"ROUNDS PASS wall_s={wall:.1f}", flush=True)
    return 0


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
