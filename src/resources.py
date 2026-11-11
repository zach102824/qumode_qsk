"""Allocated CPU/memory detection. Use cgroup quota and affinity, not host os.cpu_count()."""
from __future__ import annotations
import os
from pathlib import Path

def _cgroup_dir() -> Path:
    text = Path("/proc/self/cgroup").read_text() if Path("/proc/self/cgroup").exists() else ""
    rel = ""
    for line in text.splitlines():
        if line.startswith("0::"):
            rel = line.split(":", 2)[2]
            break
    base = Path("/sys/fs/cgroup") / rel.lstrip("/")
    return base if base.exists() else Path("/sys/fs/cgroup")

def _walk_cg_file(name: str):
    p = _cgroup_dir()
    seen = set()
    while p not in seen and str(p).startswith("/sys/fs/cgroup"):
        seen.add(p)
        f = p / name
        if f.exists():
            try:
                return f, f.read_text().strip()
            except Exception:
                return f, None
        if p == Path("/sys/fs/cgroup") or p.parent == p:
            break
        p = p.parent
    return None, None

def parse_cpuset(spec: str) -> list[int]:
    cpus = []
    spec = (spec or "").strip()
    if not spec:
        return cpus
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))

def allocated_cpus() -> dict:
    """Return allocated CPU capacity from quota or effective cpuset/affinity, never raw host count alone."""
    source = []
    quota_cpus = None
    path, text = _walk_cg_file("cpu.max")
    if text:
        parts = text.split()
        source.append(dict(file=str(path), value=text))
        if parts and parts[0] != "max":
            quota, period = float(parts[0]), float(parts[1] if len(parts) > 1 else 100000)
            if period > 0:
                quota_cpus = quota / period
    cpuset = []
    path, text = _walk_cg_file("cpuset.cpus.effective")
    if text:
        cpuset = parse_cpuset(text)
        source.append(dict(file=str(path), value=text, n=len(cpuset)))
    if not cpuset:
        path, text = _walk_cg_file("cpuset.cpus")
        if text:
            cpuset = parse_cpuset(text)
            source.append(dict(file=str(path), value=text, n=len(cpuset)))
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except Exception:
        affinity = []
    host_count = os.cpu_count() or 0
    if quota_cpus is not None:
        n = max(1, int(quota_cpus) if abs(quota_cpus - round(quota_cpus)) < 1e-9 else max(1, int(quota_cpus // 1)))
        # Keep fractional quota visible; integer worker cap is floor, at least 1.
        n_float = quota_cpus
        cap_source = "cgroup_cpu.max_quota"
    elif cpuset:
        n = len(cpuset)
        n_float = float(n)
        cap_source = "cgroup_cpuset.cpus.effective"
    elif affinity:
        n = len(affinity)
        n_float = float(n)
        cap_source = "sched_getaffinity"
    else:
        n = max(1, host_count)
        n_float = float(n)
        cap_source = "os.cpu_count_fallback"
    return dict(
        allocated_cpus=int(n),
        allocated_cpus_float=n_float,
        cap_source=cap_source,
        quota_cpus=quota_cpus,
        cpuset=cpuset,
        affinity=affinity,
        host_os_cpu_count=host_count,
        host_count_not_used_as_cap=(cap_source != "os.cpu_count_fallback"),
        sources=source,
        cgroup=str(_cgroup_dir()),
    )

def meminfo() -> dict:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        parts = v.split()
        out[k] = int(parts[0])  # kB
    return out

def cgroup_memory() -> dict:
    path, text = _walk_cg_file("memory.max")
    mem_max = None if (not text or text == "max") else int(text)
    path_c, text_c = _walk_cg_file("memory.current")
    current = int(text_c) if text_c and text_c.isdigit() else None
    return dict(memory_max_bytes=mem_max, memory_current_bytes=current,
                memory_max_file=str(path) if path else None,
                memory_current_file=str(path_c) if path_c else None,
                memory_max_unlimited=(text == "max" if text else None))

def memory_headroom_kib(reserve_kib=512 * 1024) -> dict:
    info = meminfo()
    avail = int(info.get("MemAvailable", 0))
    cg = cgroup_memory()
    cg_head = None
    if cg["memory_max_bytes"] is not None and cg["memory_current_bytes"] is not None:
        cg_head = max(0, (cg["memory_max_bytes"] - cg["memory_current_bytes"]) // 1024)
    usable = avail if cg_head is None else min(avail, cg_head)
    after_reserve = max(0, usable - int(reserve_kib))
    return dict(
        MemTotal_kib=int(info.get("MemTotal", 0)),
        MemAvailable_kib=avail,
        SwapTotal_kib=int(info.get("SwapTotal", 0)),
        cgroup=cg,
        usable_kib=usable,
        reserve_kib=int(reserve_kib),
        headroom_kib=after_reserve,
    )

def set_worker_thread_env():
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
              "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"

def nice_self(n=10):
    try:
        os.nice(int(n))
        return int(n)
    except Exception:
        return None
