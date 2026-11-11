"""Campaign-local between-job admission. Nice own workers. Never touch other processes.

Does not claim mid-job preemption: in-flight cells run to their next atomic checkpoint.
Backoff = launch fewer new workers. SIGSTOP is not used in this initial scheduler.
Own vs external CPU is distinguished so self-saturation does not throttle the pool.
"""
from __future__ import annotations
import os, time, resource
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

from .resources import allocated_cpus, memory_headroom_kib, set_worker_thread_env, nice_self
from .io_util import append_jsonl, now_shanghai
from .paths import LOGS, HEARTBEAT
from .io_util import save_json

SCHED_LOG = LOGS / "scheduler.jsonl"
DEFAULT_NICE = 10
DEFAULT_RESERVE_KIB = 512 * 1024
SCALE_UP_CLEAR_SAMPLES = 3


def _own_cpu_times(pids):
    total = 0.0
    for pid in list(pids):
        try:
            # /proc/pid/stat fields 14 and 15 (utime, stime) in ticks
            parts = open(f"/proc/{pid}/stat").read().split()
            total += int(parts[13]) + int(parts[14])
        except Exception:
            continue
    return total


def _host_cpu_times():
    # /proc/stat first line: cpu user nice system idle iowait irq softirq steal ...
    parts = open("/proc/stat").readline().split()
    nums = [int(x) for x in parts[1:]]
    idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
    total = sum(nums)
    return total, idle


class ResourceMonitor:
    def __init__(self, own_pids=None, reserve_kib=DEFAULT_RESERVE_KIB):
        self.own_pids = set(own_pids or [])
        self.reserve_kib = reserve_kib
        self._prev = None  # (host_total, host_idle, own_ticks, t)

    def add_pid(self, pid):
        self.own_pids.add(int(pid))

    def discard_pid(self, pid):
        self.own_pids.discard(int(pid))

    def sample(self):
        cap = allocated_cpus()
        mem = memory_headroom_kib(self.reserve_kib)
        host_tot, host_idle = _host_cpu_times()
        own = _own_cpu_times(self.own_pids)
        now = time.time()
        ext_frac = None
        own_frac = None
        host_busy = None
        if self._prev is not None:
            dt_tot = host_tot - self._prev[0]
            dt_idle = host_idle - self._prev[1]
            dt_own = own - self._prev[2]
            if dt_tot > 0:
                host_busy = max(0.0, 1.0 - dt_idle / dt_tot)
                # Own ticks are process CPU time (user+sys) in the same tick units as /proc/stat.
                ncpu = max(cap["allocated_cpus_float"], 1.0)
                own_frac = max(0.0, min(1.0, (dt_own / dt_tot)))  # fraction of all host ticks
                # External busy among allocated CPUs: host busy minus own, floored at 0.
                ext_frac = max(0.0, host_busy - own_frac)
        self._prev = (host_tot, host_idle, own, now)
        return dict(
            shanghai=now_shanghai(),
            allocated_cpus=cap["allocated_cpus"],
            allocated_cpus_float=cap["allocated_cpus_float"],
            cap_source=cap["cap_source"],
            cap=cap,
            memory=mem,
            host_busy_frac=host_busy,
            own_cpu_frac=own_frac,
            external_cpu_frac=ext_frac,
            own_pids=sorted(self.own_pids),
            n_own=len(self.own_pids),
        )


def target_workers(sample, class_rss_kib, config_max=None, external_cpu_backoff=0.35, hysteresis_up=None):
    """Compute admission target. Memory uses remaining headroom (MemAvailable already excludes our RSS)."""
    alloc = int(sample["allocated_cpus"])
    cap = alloc if config_max is None else min(alloc, int(config_max))
    cap = max(1, cap)
    rss = max(float(class_rss_kib), 1.0)
    head = int(sample["memory"]["headroom_kib"])
    # headroom already subtracted reserve. Remaining RAM for *additional* workers.
    # Running workers already reduced MemAvailable, so this caps new+we still need room
    # for processes that are running: we only admit additional if headroom fits one more.
    # Target total running is what's already running plus how many more fit, capped by CPU.
    more_by_mem = int(head / (1.5 * rss)) if rss > 0 else 0
    running = int(sample.get("n_own", 0))
    # Total we can support now: running + more_by_mem, but if headroom is computed after
    # reserve, running already consumed RAM. So max total ~ running + more_by_mem.
    mem_total = max(1, running + max(0, more_by_mem)) if running else max(0, more_by_mem)
    if mem_total < 1 and head >= rss:
        mem_total = 1
    target = min(cap, mem_total if mem_total > 0 else 0)
    # If even one worker may not fit, target 0 unless nothing is running and we must make progress?
    # Keep at least 0. Orchestrator may wait rather than oversubscribe RAM.
    ext = sample.get("external_cpu_frac")
    reason = ["cpu_cap=" + str(cap), "mem_total=" + str(mem_total), "running=" + str(running)]
    if ext is not None and ext >= external_cpu_backoff:
        # Leave room for external: reduce target by roughly ext * allocated.
        leave = int(np_ceil(ext * sample["allocated_cpus_float"]))
        target = min(target, max(0, alloc - leave))
        reason.append(f"external_cpu_frac={ext:.3f}_backoff_leave={leave}")
    target = int(max(0, target))
    return target, reason


def np_ceil(x):
    import math
    return int(math.ceil(x))


def worker_initializer(nice_value=DEFAULT_NICE):
    import sys
    root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    set_worker_thread_env()
    nice_self(nice_value)
    try:
        from src.kernels_numba import warmup
        warmup(8)
    except Exception:
        pass


class AdmissionPool:
    """Submit at most `target` in-flight jobs; on completion, recompute target. No SIGSTOP."""

    def __init__(self, class_rss_kib, log_path=SCHED_LOG, config_max=None,
                 reserve_kib=DEFAULT_RESERVE_KIB, nice_value=DEFAULT_NICE,
                 external_cpu_backoff=0.35, scale_up_clear=SCALE_UP_CLEAR_SAMPLES):
        self.class_rss_kib = class_rss_kib
        self.log_path = log_path
        self.config_max = config_max
        self.reserve_kib = reserve_kib
        self.nice_value = nice_value
        self.external_cpu_backoff = external_cpu_backoff
        self.scale_up_clear = scale_up_clear
        self.monitor = ResourceMonitor(reserve_kib=reserve_kib)
        self._clear = 0
        self._hold_target = None
        self.decisions = []

    def log(self, rec):
        rec = dict(rec)
        rec.setdefault("shanghai", now_shanghai())
        self.decisions.append(rec)
        append_jsonl(self.log_path, rec)

    def decide(self, sample, current_target):
        raw, reason = target_workers(
            sample, self.class_rss_kib, config_max=self.config_max,
            external_cpu_backoff=self.external_cpu_backoff,
        )
        # Hysteresis on scale-up only.
        if self._hold_target is None:
            self._hold_target = raw
            self._clear = 0
            return raw, reason + ["init"]
        if raw < self._hold_target:
            self._hold_target = raw
            self._clear = 0
            return raw, reason + ["scale_down"]
        if raw > self._hold_target:
            self._clear += 1
            if self._clear >= self.scale_up_clear:
                self._hold_target = raw
                self._clear = 0
                return raw, reason + ["scale_up_after_hysteresis"]
            return self._hold_target, reason + [f"hold_up_{self._clear}/{self.scale_up_clear}"]
        self._clear = 0
        return self._hold_target, reason + ["hold"]

    def map(self, fn, tasks, max_pool=None):
        tasks = list(tasks)
        pending = deque(enumerate(tasks))
        results = [None] * len(tasks)
        # Pool sized to allocated CPUs (or config_max); admission further limits in-flight submits.
        cap = allocated_cpus()["allocated_cpus"]
        pool_n = cap if max_pool is None else min(cap, int(max_pool))
        if self.config_max is not None:
            pool_n = min(pool_n, int(self.config_max))
        pool_n = max(1, pool_n)
        self.log(dict(event="pool_start", pool_n=pool_n, n_tasks=len(tasks),
                      class_rss_kib=self.class_rss_kib, nice=self.nice_value,
                      note="between_job_admission_only_no_midjob_SIGSTOP"))
        set_worker_thread_env()
        in_flight = {}
        with ProcessPoolExecutor(max_workers=pool_n, initializer=worker_initializer,
                                 initargs=(self.nice_value,)) as pool:
            # Record child pids when they appear.
            def refresh_pids():
                try:
                    for p in pool._processes.values():
                        self.monitor.add_pid(p.pid)
                except Exception:
                    pass

            while pending or in_flight:
                refresh_pids()
                sample = self.monitor.sample()
                sample["n_own"] = len(in_flight)
                # For memory target, n_own should be in-flight jobs (workers may be persistent in the pool).
                target, reason = self.decide(sample, len(in_flight))
                action = "hold"
                launched = 0
                while pending and len(in_flight) < max(target, 0):
                    idx, task = pending.popleft()
                    fut = pool.submit(fn, task)
                    in_flight[fut] = idx
                    launched += 1
                    action = "admit"
                if launched:
                    action = "admit"
                elif target < len(in_flight):
                    action = "backoff_wait_for_completion"
                elif not pending and in_flight:
                    action = "drain"
                elif pending and target == 0:
                    action = "wait_memory_or_cpu"
                self.log(dict(
                    event="decision", action=action, target=target, in_flight=len(in_flight),
                    pending=len(pending), launched=launched, reason=reason,
                    allocated_cpus=sample["allocated_cpus"], cap_source=sample["cap_source"],
                    MemAvailable_kib=sample["memory"]["MemAvailable_kib"],
                    headroom_kib=sample["memory"]["headroom_kib"],
                    external_cpu_frac=sample.get("external_cpu_frac"),
                    own_cpu_frac=sample.get("own_cpu_frac"),
                    host_busy_frac=sample.get("host_busy_frac"),
                ))
                # Keep a live heartbeat so a long in-flight job is not mistaken for a stale/dead run.
                try:
                    save_json(HEARTBEAT, dict(
                        shanghai=now_shanghai(), event="scheduler_heartbeat",
                        action=action, target=target, in_flight=len(in_flight),
                        pending=len(pending),
                        MemAvailable_kib=sample["memory"]["MemAvailable_kib"],
                        headroom_kib=sample["memory"]["headroom_kib"],
                        own_pids=sample.get("own_pids"),
                    ))
                except Exception:
                    pass
                if not in_flight:
                    time.sleep(1.0)
                    continue
                done, _ = wait(in_flight.keys(), timeout=2.0, return_when=FIRST_COMPLETED)
                for fut in done:
                    idx = in_flight.pop(fut)
                    results[idx] = fut.result()
        self.log(dict(event="pool_done", n_tasks=len(tasks)))
        return results

class MixedAdmissionPool:
    """Admit mixed (task, N) work under a hard CPU cap. Never 8x8 oversubscribe.

    Between-job admission only. First-fit over a priority-ordered pending list so
    a blocked heavy N=21 cell does not stall cheaper work behind it.
    """

    def __init__(self, rss_fn, cap_fn, reserve_kib=512 * 1024, hard_cpu_cap=8,
                 n21_cap=1, log_path=SCHED_LOG, nice_value=DEFAULT_NICE,
                 external_cpu_backoff=0.35, scale_up_clear=SCALE_UP_CLEAR_SAMPLES,
                 need_fn=None, heavy_fn=None, heavy_mutex_below_kib=6 * 1024 * 1024,
                 ready_fn=None):
        self.rss_fn = rss_fn
        self.cap_fn = cap_fn
        self.need_fn = need_fn
        self.heavy_fn = heavy_fn
        self.ready_fn = ready_fn
        self.heavy_mutex_below_kib = heavy_mutex_below_kib
        self.reserve_kib = reserve_kib
        self.hard_cpu_cap = int(hard_cpu_cap)
        self.n21_cap = int(n21_cap)
        self.log_path = log_path
        self.nice_value = nice_value
        self.external_cpu_backoff = external_cpu_backoff
        self.monitor = ResourceMonitor(reserve_kib=reserve_kib)
        self.decisions = []

    def log(self, rec):
        rec = dict(rec)
        rec.setdefault("shanghai", now_shanghai())
        self.decisions.append(rec)
        append_jsonl(self.log_path, rec)

    @staticmethod
    def _kind(spec):
        k = spec.get("task")
        return "bfgs" if k == "bfgs_pack" else k

    def _can_admit(self, spec, in_specs, sample, ext_leave):
        n = int(spec["N"])
        kind = self._kind(spec)
        if self.ready_fn is not None and not self.ready_fn(spec):
            return False, "dep_not_ready"
        running = len(in_specs)
        if running >= self.hard_cpu_cap:
            return False, "hard_cpu_cap"
        if running >= max(0, self.hard_cpu_cap - ext_leave):
            return False, "external_cpu_backoff"
        if n >= 21:
            n21 = sum(1 for s in in_specs if int(s["N"]) >= 21)
            if n21 >= self.n21_cap:
                return False, "n21_cap"
        same = sum(1 for s in in_specs if self._kind(s) == kind and int(s["N"]) == n)
        cap = int(self.cap_fn(spec))
        if same >= cap:
            return False, f"class_cap_{kind}_N{n}={cap}"
        if self.heavy_fn is not None:
            avail = int(sample["memory"]["MemAvailable_kib"])
            if (self.heavy_fn(spec)
                and any(self.heavy_fn(s) for s in in_specs)
                and avail < int(self.heavy_mutex_below_kib)):
                return False, "heavy_mutex"
        # class_rss already includes a 1.5x measured safety factor. Do not
        # multiply again (that previously demanded ~5 GiB free for N=21 and
        # stalled admission under leftover historical RSS).
        avail = int(sample["memory"]["MemAvailable_kib"])
        need = float(self.need_fn(spec) if self.need_fn is not None else self.rss_fn(spec))
        if avail < need:
            return False, f"mem_need={need:.0f}_avail={avail}"
        return True, "ok"

    def map(self, fn, tasks, max_pool=None):
        tasks = list(tasks)
        pending = list(enumerate(tasks))
        results = [None] * len(tasks)
        cap = allocated_cpus()["allocated_cpus"]
        pool_n = min(int(self.hard_cpu_cap), int(cap))
        if max_pool is not None:
            pool_n = min(pool_n, int(max_pool))
        pool_n = max(1, pool_n)
        self.log(dict(event="mixed_pool_start", pool_n=pool_n, n_tasks=len(tasks),
                      hard_cpu_cap=self.hard_cpu_cap, n21_cap=self.n21_cap,
                      reserve_kib=self.reserve_kib, nice=self.nice_value,
                      note="mixed_between_job_admission_no_midjob_SIGSTOP_no_8x8"))
        set_worker_thread_env()
        in_flight = {}
        with ProcessPoolExecutor(max_workers=pool_n, initializer=worker_initializer,
                                 initargs=(self.nice_value,)) as pool:
            def refresh_pids():
                try:
                    for p in pool._processes.values():
                        self.monitor.add_pid(p.pid)
                except Exception:
                    pass

            while pending or in_flight:
                refresh_pids()
                sample = self.monitor.sample()
                sample["n_own"] = len(in_flight)
                ext = sample.get("external_cpu_frac")
                ext_leave = 0
                if ext is not None and ext >= self.external_cpu_backoff:
                    ext_leave = int(__import__("math").ceil(ext * sample["allocated_cpus_float"]))
                in_specs = [tasks[idx] for idx in in_flight.values()]
                launched = 0
                reasons_blocked = []
                progressed = True
                while progressed and pending:
                    progressed = False
                    for pos, (idx, task) in enumerate(list(pending)):
                        ok, why = self._can_admit(task, in_specs, sample, ext_leave)
                        if ok:
                            pending.pop(pos)
                            fut = pool.submit(fn, task)
                            in_flight[fut] = idx
                            in_specs.append(task)
                            launched += 1
                            progressed = True
                            break
                        else:
                            reasons_blocked.append(why)
                    if launched >= pool_n:
                        break
                if launched:
                    action = "admit"
                elif pending and not in_flight:
                    action = "wait_memory_or_cpu"
                elif pending:
                    action = "backoff_wait_for_completion"
                else:
                    action = "drain"
                self.log(dict(
                    event="decision", action=action, in_flight=len(in_flight),
                    pending=len(pending), launched=launched,
                    blocked_reasons=reasons_blocked[:8],
                    allocated_cpus=sample["allocated_cpus"], cap_source=sample["cap_source"],
                    MemAvailable_kib=sample["memory"]["MemAvailable_kib"],
                    headroom_kib=sample["memory"]["headroom_kib"],
                    external_cpu_frac=sample.get("external_cpu_frac"),
                    own_cpu_frac=sample.get("own_cpu_frac"),
                    host_busy_frac=sample.get("host_busy_frac"),
                    in_flight_N=sorted({int(tasks[i]["N"]) for i in in_flight.values()}),
                ))
                try:
                    save_json(HEARTBEAT, dict(
                        shanghai=now_shanghai(), event="scheduler_heartbeat",
                        campaign="rounds", action=action, in_flight=len(in_flight),
                        pending=len(pending),
                        MemAvailable_kib=sample["memory"]["MemAvailable_kib"],
                        headroom_kib=sample["memory"]["headroom_kib"],
                        own_pids=sample.get("own_pids"),
                    ))
                except Exception:
                    pass
                if not in_flight:
                    time.sleep(1.0)
                    continue
                done, _ = wait(in_flight.keys(), timeout=2.0, return_when=FIRST_COMPLETED)
                for fut in done:
                    idx = in_flight.pop(fut)
                    results[idx] = fut.result()
        self.log(dict(event="mixed_pool_done", n_tasks=len(tasks)))
        return results
