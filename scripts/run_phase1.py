#!/usr/bin/env python3
"""Phase 1: freeze protocol/seeds/starts, run audits, run disjoint resource pilot.

Does not launch the 200-instance main campaign. Writes ETA for the parent to relay.
"""
from __future__ import annotations
import os, sys, time, traceback, resource, fcntl
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.resources import set_worker_thread_env, allocated_cpus, memory_headroom_kib
set_worker_thread_env()

from src import protocol as P
from src.io_util import save_json, now_shanghai, append_jsonl
from src.paths import (
    ROOT as CAMP, STATUS, HEARTBEAT, PHASE1_STATUS, PILOT_RESULTS, ETA_FILE,
    LOGS, AUDITS, PROTOCOL, MANIFESTS, PILOT_RUNS,
)
from src.freeze_all import freeze
from src.audits import run_all_audits
from src.pilot import planned_tasks, dispatch, PROGRESS
from src.scheduler import AdmissionPool
from src.spsa import query_schedule

LOCK = CAMP / "phase1.lock"


def update_status(**kw):
    rec = dict(workstreamID="qumode_qsk_phase1", authorization=["pt3487u", "pt3488u"],
               main_fits_not_started=True, shanghai=now_shanghai(), pid=os.getpid())
    if STATUS.exists():
        try:
            import json
            rec.update(json.loads(STATUS.read_text()))
        except Exception:
            pass
    rec.update(kw)
    rec["shanghai"] = now_shanghai()
    rec["pid"] = os.getpid()
    save_json(STATUS, rec)
    save_json(HEARTBEAT, dict(shanghai=now_shanghai(), pid=os.getpid(),
                              stage=rec.get("stage"), state=rec.get("state")))
    save_json(PHASE1_STATUS, rec)
    return rec


def eta_from_pilot(ent_rows, spsa_rows, bfgs_rows):
    """Post-pilot ETA only. Missing N interpolated in log2(dim). Uncertainties stated."""
    import math
    byN_ent = {r["N"]: r for r in ent_rows if r.get("wall_s") and "error" not in r}
    by_spsa = {}
    for r in spsa_rows:
        if r.get("wall_s") and "error" not in r:
            by_spsa[(r["N"], r["L"], r["method"])] = r
    by_bfgs = {}
    for r in bfgs_rows:
        if r.get("wall_s") and "error" not in r:
            by_bfgs.setdefault((r["N"], r["L"], r["method"]), []).append(r)

    def interp_time(table, n):
        if n in table:
            return table[n], "measured"
        xs = sorted(table)
        if not xs:
            return None, "missing"
        # log-linear in 2^N between nearest measured sizes
        if n < xs[0]:
            return table[xs[0]] * (2 ** (n - xs[0])), f"scaled_from_N{xs[0]}"
        if n > xs[-1]:
            return table[xs[-1]] * (2 ** (n - xs[-1])), f"scaled_from_N{xs[-1]}"
        lo = max(x for x in xs if x <= n)
        hi = min(x for x in xs if x >= n)
        if lo == hi:
            return table[lo], "measured"
        t = math.log(table[lo]) + (math.log(table[hi]) - math.log(table[lo])) * (n - lo) / (hi - lo)
        return math.exp(t), f"loglin_N{lo}_N{hi}"

    ent_t = {n: byN_ent[n]["wall_s"] for n in byN_ent}
    ent_cpu = {n: byN_ent[n]["CPU_s"] for n in byN_ent}
    entropy_eta = {}
    ent_wall_sum = 0.0
    for n in P.NS:
        tw, how = interp_time(ent_t, n)
        entropy_eta[str(n)] = dict(seconds_per_instance=tw, source=how, n_instances=200)
        if tw:
            ent_wall_sum += 200 * tw
    entropy_eta["serial_wall_s"] = ent_wall_sum

    # SPSA L=1 nov3 times
    spsa_L1 = {k[0]: v["wall_s"] for k, v in by_spsa.items() if k[1] == 1 and k[2] == "nov3"}
    ratio_L6 = None
    if (6, 1, "nov3") in by_spsa and (6, 6, "nov3") in by_spsa:
        t1, t6 = by_spsa[(6, 1, "nov3")]["wall_s"], by_spsa[(6, 6, "nov3")]["wall_s"]
        if t1 > 0:
            ratio_L6 = t6 / t1
    # If L=6 / L=1 ~ 6, scale by L; if larger, use measured ratio's implied extra.
    l_scale = {}
    for L in P.DEPTHS:
        if L == 1:
            l_scale[L] = 1.0
        elif L == 6 and ratio_L6:
            l_scale[L] = ratio_L6
        elif ratio_L6:
            # interpolate between 1 and 6 in L
            l_scale[L] = 1.0 + (ratio_L6 - 1.0) * (L - 1) / 5.0
        else:
            l_scale[L] = float(L)  # fallback: linear in budget Q=1024L

    spsa_cells = {}
    spsa_serial = 0.0
    for n in P.NS:
        t1, how = interp_time(spsa_L1, n)
        per_L = {}
        for L in P.DEPTHS:
            t = None if t1 is None else t1 * l_scale[L]
            # two methods; if only nov3 measured, assume qaoa similar unless N=6 qaoa exists
            per_L[str(L)] = t
        qaoa_scale = 1.0
        if (6, 1, "qaoa") in by_spsa and (6, 1, "nov3") in by_spsa and by_spsa[(6, 1, "nov3")]["wall_s"] > 0:
            qaoa_scale = by_spsa[(6, 1, "qaoa")]["wall_s"] / by_spsa[(6, 1, "nov3")]["wall_s"]
        inst = 200
        # main will skip invalid peaks; ETA assumes 200 (upper bound)
        nov3_sum = 0.0 if t1 is None else inst * sum(l_scale[L] * t1 for L in P.DEPTHS)
        qaoa_sum = nov3_sum * qaoa_scale
        spsa_cells[str(n)] = dict(
            L1_nov3_s=t1, L1_source=how, l_scale=l_scale, qaoa_over_nov3=qaoa_scale,
            serial_nov3_s=nov3_sum, serial_qaoa_s=qaoa_sum, serial_both_s=nov3_sum + qaoa_sum,
        )
        if t1:
            spsa_serial += nov3_sum + qaoa_sum

    bfgs_L1 = {}
    bfgs_nfev = {}
    for (n, L, method), rows in by_bfgs.items():
        if L == 1 and method == "nov3":
            bfgs_L1[n] = sum(r["wall_s"] for r in rows) / len(rows)
            bfgs_nfev[n] = sum(r.get("nfev", 0) for r in rows) / len(rows)
    bfgs_serial = 0.0
    bfgs_cells = {}
    for n in P.NS:
        t1, how = interp_time(bfgs_L1, n)
        # Without other L, scale by param count ratio as a weak prior only, and flag it.
        # 16 starts * 6 L * 2 methods * 200 inst
        # Use L=1 time * 16 already per start; for L>1 unknown. Use param-count ratio  (3+3L)/6 or (2L)/2 vs L=1 as a weak stand-in.
        if t1 is None:
            bfgs_cells[str(n)] = dict(per_start_L1_s=None, source=how)
            continue
        total = 0.0
        per_L = {}
        for L in P.DEPTHS:
            # weak scale: time ~ nfev * eval_cost; eval_cost ~ L * 2^N, nfev unknown. Use *L as a lower-ish scale.
            tL = t1 * L
            per_L[str(L)] = tL
            total += 200 * 16 * 2 * tL  # inst * starts * methods
        bfgs_cells[str(n)] = dict(
            per_start_L1_nov3_s=t1, source=how, mean_nfev_L1=bfgs_nfev.get(n),
            L_scale_assumption="time_per_start ~= L * L1_time (weak; nfev may grow)",
            serial_all_starts_s=total, per_L_start_s=per_L,
        )
        bfgs_serial += total

    # Parallel wall: if we can keep W workers, wall ~ serial / W, but memory will cut W at large N.
    cap = allocated_cpus()
    mem = memory_headroom_kib()
    rss = {}
    for r in ent_rows + spsa_rows + bfgs_rows:
        if r.get("peak_RSS_kib"):
            rss.setdefault(r.get("N"), []).append(r["peak_RSS_kib"])
    rss_max = {n: max(v) for n, v in rss.items()}
    # Conservative workers by N
    workers = {}
    head = mem["headroom_kib"]
    for n in P.NS:
        r = rss_max.get(n) or rss_max.get(max([x for x in rss_max if x <= n], default=n), 2_000_000)
        wmem = max(1, int(head / (1.5 * max(r, 1)))) if r else 1
        workers[str(n)] = int(max(1, min(cap["allocated_cpus"], wmem)))

    # Very rough campaign wall: entropy serial/W + SPSA serial weighted + BFGS
    # Use per-N workers.
    def par_wall(serial_by_n, key_serial):
        w = 0.0
        for n in P.NS:
            s = serial_by_n.get(str(n), {}).get(key_serial)
            if s:
                w += s / max(workers[str(n)], 1)
        return w

    note = (
        "Post-pilot estimate only, not a guarantee. BFGS times for L>1 and unmeasured N are extrapolated. "
        "Invalid entropy peaks would reduce fit counts. Memory pressure and other tasks will change realized wall. "
        "Scheduler admits between jobs only; this ETA is not a reservation."
    )
    return dict(
        shanghai=now_shanghai(),
        allocated_cpus=cap,
        memory_at_eta=mem,
        rss_peak_kib_by_N=rss_max,
        suggested_workers_by_N=workers,
        entropy=entropy_eta,
        spsa=dict(cells=spsa_cells, serial_s=spsa_serial,
                  approx_parallel_wall_s=par_wall(spsa_cells, "serial_both_s"),
                  L6_over_L1_N6_nov3=ratio_L6),
        bfgs=dict(cells=bfgs_cells, serial_s=bfgs_serial,
                  approx_parallel_wall_s=par_wall(bfgs_cells, "serial_all_starts_s"),
                  caveat="L>1 scaled as L * L1 start time; weak"),
        totals=dict(
            entropy_serial_s=ent_wall_sum,
            spsa_serial_s=spsa_serial,
            bfgs_serial_s=bfgs_serial,
            all_serial_s=ent_wall_sum + spsa_serial + bfgs_serial,
            approx_parallel_wall_s=(
                ent_wall_sum / max(min(workers.values()), 1)
                + par_wall(spsa_cells, "serial_both_s")
                + par_wall(bfgs_cells, "serial_all_starts_s")
            ),
        ),
        note=note,
    )


def result_path(spec):
    kind=spec["task"]
    if kind=="entropy":
        return PILOT_RUNS / f"entropy_N{spec['N']}_seed{spec['seed']}.json"
    if kind=="spsa":
        return PILOT_RUNS / f"spsa_N{spec['N']}_seed{spec['seed']}_L{spec['L']}_{spec['method']}.json"
    if kind=="bfgs":
        return PILOT_RUNS / f"bfgs_N{spec['N']}_seed{spec['seed']}_L{spec['L']}_{spec['method']}_start{spec['start_index']}.json"
    raise ValueError(kind)


def load_existing(spec):
    path=result_path(spec)
    if not path.exists():
        return None
    try:
        rec=__import__("json").loads(path.read_text())
    except Exception:
        return None
    if rec.get("error"):
        return None
    rec["result_file"]=str(path)
    rec["reused"]=True
    return rec


def run_wave(name, tasks, class_rss_kib, config_max=None):
    if not tasks:
        return [], dict(wave=name, n=0, skipped_empty=True)
    update_status(state="running", stage=f"pilot_{name}", n_tasks=len(tasks),
                  class_rss_kib=class_rss_kib, config_max=config_max)
    pool = AdmissionPool(class_rss_kib=class_rss_kib, config_max=config_max,
                         reserve_kib=512 * 1024 if max((t.get("N") or 6) for t in tasks) < 21 else 1024 * 1024)
    t0 = time.time()
    rows = pool.map(dispatch, tasks, max_pool=config_max)
    for r in rows:
        append_jsonl(PROGRESS, dict(event="task_done", wave=name, rec={k: r.get(k) for k in
                     ("task", "N", "seed", "L", "method", "start_index", "wall_s", "CPU_s",
                      "peak_RSS_kib", "best_E", "F", "label", "valid_for_fits", "error", "result_file") if k in r or True},
                     shanghai=now_shanghai()))
    return rows, dict(wave=name, wall_s=time.time() - t0, n=len(tasks),
                      decisions_tail=pool.decisions[-5:], class_rss_kib=class_rss_kib)


def main():
    PILOT_RUNS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    AUDITS.mkdir(parents=True, exist_ok=True)
    lockf = LOCK.open("w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        update_status(state="blocked", stage="lock_held", error="phase1.lock is held")
        raise SystemExit("phase1.lock is held")
    lockf.write(str(os.getpid())); lockf.flush()
    t_all = time.time()
    orig = None
    if STATUS.exists():
        try:
            orig = __import__("json").loads(STATUS.read_text())
        except Exception:
            orig = None
    started = (orig or {}).get("started_shanghai") or now_shanghai()
    update_status(state="running", stage="resume_or_freeze", started_shanghai=started,
                  resume=True, note="reuse completed audits and entropy N<=18; rerun unfinished only")
    try:
        proto = PROTOCOL / "PROTOCOL_FROZEN.json"
        auditf = AUDITS / "AUDIT_SUMMARY.json"
        if proto.exists():
            fr = dict(protocol=str(proto), reused=True, shanghai=now_shanghai())
            update_status(stage="freeze_reused", freeze=fr)
        else:
            fr = freeze()
            update_status(stage="audits", freeze=fr)
        if auditf.exists() and __import__("json").loads(auditf.read_text()).get("verdict")=="PASS":
            audits = __import__("json").loads(auditf.read_text())
            update_status(stage="audits_reused", audits_verdict="PASS", audit_wall_s=audits.get("wall_s"))
        else:
            audits = run_all_audits()
            update_status(stage="audits_done", audits_verdict=audits["verdict"], audit_wall_s=audits["wall_s"])
        if audits["verdict"] != "PASS":
            update_status(state="failed", stage="audits_failed", audits=audits)
            save_json(PILOT_RESULTS, dict(verdict="AUDITS_FAILED", audits=audits, shanghai=now_shanghai()))
            raise SystemExit("audits failed; not starting pilot")

        plan = planned_tasks()
        # Entropy waves: small N together, then 15, 18, 21 alone.
        # N=21 admission class reduced after fixing the all-field GS cache leak.
        ent_rows = []
        wave_meta = []
        groups = [
            ("entropy_N6_9_12", [t for t in plan["entropy"] if t["N"] <= 12], 300_000, None),
            ("entropy_N15", [t for t in plan["entropy"] if t["N"] == 15], 800_000, 2),
            ("entropy_N18", [t for t in plan["entropy"] if t["N"] == 18], 1_500_000, 1),
            ("entropy_N21", [t for t in plan["entropy"] if t["N"] == 21], 2_000_000, 1),
        ]
        for name, tasks, rss, mx in groups:
            reused, todo = [], []
            for spec in tasks:
                ex = load_existing(spec)
                if ex is not None:
                    reused.append(ex)
                else:
                    todo.append(spec)
            if reused:
                ent_rows.extend(reused)
                wave_meta.append(dict(wave=name, reused=len(reused), files=[r.get("result_file") for r in reused]))
            if todo:
                rows, meta = run_wave(name, todo, rss, mx)
                ent_rows.extend(rows)
                wave_meta.append(meta)
                if any(r.get("error") for r in rows):
                    update_status(stage=f"{name}_had_errors", last_errors=[r.get("error") for r in rows if r.get("error")])
            elif not reused:
                wave_meta.append(dict(wave=name, n=0, skipped_empty=True))

        failed_ent = {r["N"] for r in ent_rows if r.get("error")}
        ok_ent = {r["N"] for r in ent_rows if not r.get("error")}

        def allowed(t):
            return t["N"] in ok_ent and t["N"] not in failed_ent

        # Refresh RSS class from measured entropy peaks when possible.
        def rss_for(n, default):
            ms = [r["peak_RSS_kib"] for r in ent_rows if r.get("N") == n and r.get("peak_RSS_kib") and not r.get("error")]
            return int(max(ms) * 1.2) if ms else default

        spsa_rows = []
        spsa_groups = [
            ("spsa_N6", [t for t in plan["spsa"] if t["N"] == 6 and allowed(t)], rss_for(6, 300_000), 4),
            ("spsa_N12", [t for t in plan["spsa"] if t["N"] == 12 and allowed(t)], rss_for(12, 500_000), 2),
            ("spsa_N18", [t for t in plan["spsa"] if t["N"] == 18 and allowed(t)], rss_for(18, 1_500_000), 1),
            ("spsa_N21", [t for t in plan["spsa"] if t["N"] == 21 and allowed(t)], rss_for(21, 3_500_000), 1),
        ]
        for name, tasks, rss, mx in spsa_groups:
            if not tasks:
                continue
            reused, todo = [], []
            for spec in tasks:
                ex = load_existing(spec)
                if ex is not None:
                    reused.append(ex)
                else:
                    todo.append(spec)
            if reused:
                spsa_rows.extend(reused)
                wave_meta.append(dict(wave=name, reused=len(reused)))
            if todo:
                rows, meta = run_wave(name, todo, rss, mx)
                spsa_rows.extend(rows)
                wave_meta.append(meta)

        # After SPSA, RSS may be higher than entropy-only; use max of entropy and completed SPSA.
        def rss_opt(n, default):
            ms = [r["peak_RSS_kib"] for r in (ent_rows + spsa_rows)
                  if r.get("N") == n and r.get("peak_RSS_kib") and not r.get("error")]
            return int(max(ms) * 1.3) if ms else default

        bfgs_rows = []
        bfgs_groups = [
            ("bfgs_N6", [t for t in plan["bfgs"] if t["N"] == 6 and allowed(t)], rss_opt(6, 400_000), 4),
            ("bfgs_N12", [t for t in plan["bfgs"] if t["N"] == 12 and allowed(t)], rss_opt(12, 700_000), 2),
            ("bfgs_N18", [t for t in plan["bfgs"] if t["N"] == 18 and allowed(t)], rss_opt(18, 2_000_000), 1),
            ("bfgs_N21", [t for t in plan["bfgs"] if t["N"] == 21 and allowed(t)], rss_opt(21, 4_000_000), 1),
        ]
        for name, tasks, rss, mx in bfgs_groups:
            if not tasks:
                continue
            reused, todo = [], []
            for spec in tasks:
                ex = load_existing(spec)
                if ex is not None:
                    reused.append(ex)
                else:
                    todo.append(spec)
            if reused:
                bfgs_rows.extend(reused)
                wave_meta.append(dict(wave=name, reused=len(reused)))
            if todo:
                rows, meta = run_wave(name, todo, rss, mx)
                bfgs_rows.extend(rows)
                wave_meta.append(meta)

        eta = eta_from_pilot(ent_rows, spsa_rows, bfgs_rows)
        save_json(ETA_FILE, eta)

        def compact(r):
            keys = ("task", "N", "seed", "L", "method", "start_index", "b", "S", "E0", "gap",
                    "label", "valid_for_fits", "best_E", "F", "calls", "nit", "nfev",
                    "success", "tightly_converged", "wall_s", "CPU_s", "peak_RSS_kib",
                    "result_file", "error", "energy_error_over_N")
            return {k: r[k] for k in keys if k in r}

        pilot = dict(
            verdict="PASS" if not any(r.get("error") for r in ent_rows + spsa_rows + bfgs_rows) else "PARTIAL",
            shanghai=now_shanghai(),
            wall_s=time.time() - t_all,
            audits=str(AUDITS / "AUDIT_SUMMARY.json"),
            protocol=str(PROTOCOL / "PROTOCOL_FROZEN.json"),
            entropy=[compact(r) for r in ent_rows],
            spsa=[compact(r) for r in spsa_rows],
            bfgs=[compact(r) for r in bfgs_rows],
            waves=wave_meta,
            eta_file=str(ETA_FILE),
            main_not_started=True,
            note="Disjoint pilot only. Main 200-instance campaign is approved but waits for this checkpoint to be relayed.",
        )
        save_json(PILOT_RESULTS, pilot)
        update_status(state="phase1_complete", stage="pilot_complete_main_not_started",
                      verdict=pilot["verdict"], wall_s=pilot["wall_s"],
                      eta_totals=eta.get("totals"),
                      finished_shanghai=now_shanghai())
        print(f"PHASE1 {pilot['verdict']} wall_s={pilot['wall_s']:.1f}", flush=True)
        return 0 if pilot["verdict"] in ("PASS", "PARTIAL") and audits["verdict"] == "PASS" else 1
    except Exception as ex:
        update_status(state="failed", stage="exception", error=repr(ex), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
