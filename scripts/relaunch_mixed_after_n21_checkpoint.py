#!/usr/bin/env python3
"""Wait for the in-flight owned N=21 entropy cell to land, then drain the
stopped staged orchestrator and relaunch the mixed-queue round runner.

Signals only the owned campaign tree (lock pid 2751854 and its children).
Never touches historical jobs. Does not delete results.
"""
from __future__ import annotations
import os, sys, time, json, signal
from pathlib import Path

ROOT = Path("/workspace/qumode_qsk")
sys.path.insert(0, str(ROOT))
RESULT = ROOT / "runs/main/entropy_N21_seed831121000.json"
LOCK = ROOT / "rounds.lock"
OLD_PARENT = 2751854
HISTORICAL = {1615751, 1744891, 1822112, 1832860, 1906430, 1906431}
NOTE = ROOT / "logs/RELAUNCH_MIXED.json"
SCAN = ROOT / "logs/entropy_scan_main_N21_seed831121000.jsonl"


def pid_alive(pid):
    return Path(f"/proc/{int(pid)}").exists()


def proc_meta(pid):
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        cwd = os.readlink(f"/proc/{pid}/cwd")
        ppid = int(open(f"/proc/{pid}/stat").read().split()[3])
        state = open(f"/proc/{pid}/stat").read().split()[2]
        return dict(cmd=cmd, cwd=cwd, ppid=ppid, state=state)
    except Exception as e:
        return dict(error=str(e))


def children_of(pid):
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        p = int(name)
        try:
            ppid = int(open(f"/proc/{p}/stat").read().split()[3])
        except Exception:
            continue
        if ppid == pid:
            out.append(p)
    return out


def owned_tree(parent):
    pids = [parent]
    for c in children_of(parent):
        pids.append(c)
    return pids


def result_ok():
    if not RESULT.exists():
        return False
    try:
        d = json.loads(RESULT.read_text())
    except Exception:
        return False
    return bool(d.get("wall_s")) and not d.get("error") and d.get("N") == 21


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} CST {msg}"
    print(line, flush=True)
    with (ROOT / "logs/relaunch_watchdog.log").open("a") as f:
        f.write(line + "\n")


def main():
    log(f"watchdog start; waiting for {RESULT.name} then drain/relaunch")
    # safety: parent must still be the owned stopped runner, or already gone
    while not result_ok():
        if not pid_alive(OLD_PARENT) and not any(pid_alive(c) for c in range(2751859, 2751867)):
            log("old tree already gone before result; will still wait for result file")
        time.sleep(30)
        # heartbeat of scan
        if SCAN.exists():
            try:
                last = SCAN.read_text().strip().splitlines()[-1]
                log(f"waiting; last scan {last[:200]}")
            except Exception:
                pass
    log("N=21 result is on disk; waiting for worker to go idle/exit")
    t0 = time.time()
    while time.time() - t0 < 180:
        busy = []
        for pid in owned_tree(OLD_PARENT) if pid_alive(OLD_PARENT) else []:
            if pid == OLD_PARENT:
                continue
            m = proc_meta(pid)
            if m.get("state") in ("R", "D"):
                busy.append((pid, m.get("state")))
        if not busy:
            break
        log(f"workers still busy {busy}")
        time.sleep(5)
    # identity check before any signal
    if pid_alive(OLD_PARENT):
        meta = proc_meta(OLD_PARENT)
        if meta.get("cwd") != str(ROOT) or "scripts/run_rounds.py" not in (meta.get("cmd") or ""):
            log(f"REFUSING to signal parent, identity mismatch {meta}")
            return 2
        try:
            lockpid = int(LOCK.read_text().strip().split()[0]) if LOCK.exists() else None
        except Exception:
            lockpid = None
        if lockpid not in (None, OLD_PARENT):
            # if lock already belongs to someone else, do not signal
            if pid_alive(lockpid):
                log(f"REFUSING: lock held by live pid {lockpid}")
                return 3
        pids = owned_tree(OLD_PARENT)
        for p in pids:
            if p in HISTORICAL:
                log(f"REFUSING: historical pid in tree {p}")
                return 4
            m = proc_meta(p)
            if "run_rounds.py" not in (m.get("cmd") or "") or m.get("cwd") != str(ROOT):
                log(f"REFUSING: child identity mismatch {p} {m}")
                return 5
        # SIGTERM workers and parent (parent is STOPPED; CONT so it can exit)
        for p in pids:
            if p == OLD_PARENT:
                continue
            log(f"SIGTERM worker {p}")
            os.kill(p, signal.SIGTERM)
        log(f"SIGTERM parent {OLD_PARENT}")
        os.kill(OLD_PARENT, signal.SIGTERM)
        log(f"SIGCONT parent {OLD_PARENT}")
        os.kill(OLD_PARENT, signal.SIGCONT)
        for i in range(30):
            if not pid_alive(OLD_PARENT) and not any(pid_alive(p) for p in pids if p != OLD_PARENT):
                break
            time.sleep(1)
        if pid_alive(OLD_PARENT):
            log("parent still alive after 30s; not escalating to SIGKILL")
            return 6
        log("old tree exited")
    else:
        log("parent already gone")
    if LOCK.exists():
        try:
            lp = int(LOCK.read_text().strip().split()[0])
        except Exception:
            lp = None
        if lp is None or not pid_alive(lp):
            LOCK.unlink()
            log("removed stale rounds.lock")
        elif lp != OLD_PARENT:
            log(f"lock now belongs to {lp}; not removing, not launching")
            return 7

    env = os.environ.copy()
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
              "NUMBA_NUM_THREADS"):
        env[k] = "1"
    env["QSK_NUMBA_CACHE"] = "1"
    env["QSK_FD_CHECK"] = "0"
    logdir = ROOT / "logs"
    stdout = open(logdir / "rounds_stdout.log", "a")
    stderr = open(logdir / "rounds_stderr.log", "a")
    stdout.write(f"\n# relaunch mixed {time.strftime('%Y-%m-%d %H:%M:%S')} CST\n")
    stdout.flush()
    proc = __import__("subprocess").Popen(
        ["python3", "-u", "scripts/run_rounds.py"],
        cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr,
        start_new_session=True,
    )
    time.sleep(3)
    newpid = proc.pid
    # the python may be the pid; lock should have the interpreter pid (same if no shell)
    live = None
    if LOCK.exists():
        try:
            live = int(LOCK.read_text().strip().split()[0])
        except Exception:
            live = None
    rec = dict(
        shanghai=time.strftime("%Y-%m-%d %H:%M:%S CST"),
        old_parent=OLD_PARENT,
        new_popen_pid=newpid,
        new_lock_pid=live,
        n21_result=str(RESULT),
        note="drained staged runner after first N=21 entropy cell landed; relaunched mixed entropy+optimize queue",
        historical_untouched=sorted(HISTORICAL),
    )
    NOTE.write_text(json.dumps(rec, indent=2) + "\n")
    log(f"relaunched lock={live} popen={newpid}")
    return 0 if live and pid_alive(live) else 8


if __name__ == "__main__":
    raise SystemExit(main())
