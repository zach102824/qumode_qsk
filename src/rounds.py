"""Cumulative 10-per-N round planning and official-round summaries.

200 instances remain the cap. Round r uses indices 10*(r-1) .. 10*r-1
for ALL N; the next round starts only after every N has finished the current
round. Surplus completed N=6 data (indices 10-199) is retained off the official
round summary. Selection is by energy, not F. No early stop at a good F.
"""
from __future__ import annotations
import math, json
from pathlib import Path
from collections import defaultdict

from . import protocol as P
from .io_util import save_json, read_json, now_shanghai
from .paths import ROOT, SUMMARIES, RUNS, DATA
from .main_jobs import load_existing, result_path, INST_DIR
from .bfgs import select_by_energy

ROUND_SIZE = 10
N_ROUNDS = P.N_INSTANCES // ROUND_SIZE  # 20
MAIN_RUNS = RUNS / "main"
PILOT_RSS = {
    ("entropy", 6): 66816, ("entropy", 9): 67424, ("entropy", 12): 72080,
    ("entropy", 15): 133200, ("entropy", 18): 617416, ("entropy", 21): 1933780,
    ("spsa", 6): 62184, ("spsa", 9): 62184, ("spsa", 12): 60548,
    ("spsa", 15): 114096, ("spsa", 18): 114096, ("spsa", 21): 470940,
    ("bfgs", 6): 59208, ("bfgs", 9): 61364, ("bfgs", 12): 61364,
    ("bfgs", 15): 189484, ("bfgs", 18): 189484, ("bfgs", 21): 977972,
}
# Conservative caps: stability over max-CPU. N=21 always 1. Total hard cap is 8.
CONFIG_MAX = {
    ("entropy", 6): 4, ("entropy", 9): 4, ("entropy", 12): 3,
    ("entropy", 15): 2, ("entropy", 18): 1, ("entropy", 21): 1,
    ("spsa", 6): 4, ("spsa", 9): 4, ("spsa", 12): 3,
    ("spsa", 15): 2, ("spsa", 18): 2, ("spsa", 21): 1,
    ("bfgs", 6): 4, ("bfgs", 9): 4, ("bfgs", 12): 3,
    ("bfgs", 15): 2, ("bfgs", 18): 2, ("bfgs", 21): 1,
}


def kind_of(spec):
    k = spec.get("task")
    return "bfgs" if k in ("bfgs", "bfgs_pack") else k


def class_rss_kib(spec):
    measured = PILOT_RSS[(kind_of(spec), int(spec["N"]))]
    return int(max(measured * 1.5, 80_000))


def config_max(spec):
    return int(CONFIG_MAX[(kind_of(spec), int(spec["N"]))])


def is_heavy(spec):
    """Jobs whose resident set is large enough to collide on this 16 GiB box."""
    n = int(spec["N"])
    return n >= 21 or (kind_of(spec) == "entropy" and n >= 18)


def admit_need_kib(spec):
    """MemAvailable must exceed measured_peak * 1.25 + 512 MiB responsive reserve.

    class_rss_kib keeps the 1.5x factor for logging; admission uses this tighter
    but still safe gate so N=21 can start when ~3.3 GiB is free (historical
    SIGSTOP jobs still hold ~2.5 GiB and must not be touched).
    """
    measured = PILOT_RSS[(kind_of(spec), int(spec["N"]))]
    return int(measured * 1.25 + 512 * 1024)


def round_indices(round_id):
    r = int(round_id)
    if not 1 <= r <= N_ROUNDS:
        raise ValueError(round_id)
    lo = (r - 1) * ROUND_SIZE
    return list(range(lo, lo + ROUND_SIZE))


def plan_round(round_id):
    """All official cells for one cumulative round: every N, depths, four arms."""
    idxs = round_indices(round_id)
    entropy, spsa, bfgs_pack = [], [], []
    for n in P.NS:
        for i in idxs:
            seed = P.main_seed(n, i)
            entropy.append(dict(task="entropy", N=n, seed=seed, index=i, round=round_id))
            for L in P.DEPTHS:
                for method in P.METHODS:
                    spsa.append(dict(task="spsa", N=n, seed=seed, index=i, L=L, method=method, round=round_id))
                    bfgs_pack.append(dict(task="bfgs_pack", N=n, seed=seed, index=i, L=L, method=method, round=round_id))
    return dict(round=round_id, indices=idxs, entropy=entropy, spsa=spsa, bfgs_pack=bfgs_pack)


def interleave_entropy(tasks):
    """Start large-N entropy first; fill remaining cores with smaller N."""
    return sorted(tasks, key=lambda t: (-int(t["N"]), int(t["index"])))


def interleave_optimize(tasks):
    """Start large-N low-L work early; zipper methods; first-fit will fill with cheap N."""
    meth_rank = {"nov3": 0, "qaoa": 1}
    kind_rank = {"spsa": 0, "bfgs_pack": 1, "bfgs": 1}

    def key(t):
        return (int(t["L"]), kind_rank.get(t["task"], 9), -int(t["N"]),
                meth_rank.get(t["method"], 9), int(t["index"]))
    return sorted(tasks, key=key)


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


def split_resume(tasks):
    reused, todo = [], []
    for spec in tasks:
        if spec["task"] == "bfgs_pack":
            missing = []
            present = []
            for k in range(P.BFGS_STARTS):
                sub = dict(task="bfgs", N=spec["N"], seed=spec["seed"], L=spec["L"],
                           method=spec["method"], start_index=k, index=spec.get("index"))
                ex = load_existing(sub)
                if ex is None:
                    missing.append(k)
                else:
                    present.append(ex)
            if not missing:
                pack = dict(spec)
                pack.update(reused=True, n_reused=len(present), n_ran=0, start_brief=present)
                try:
                    pack["selected"] = select_by_energy(present)
                except Exception:
                    pass
                reused.append(pack)
            else:
                todo.append(spec)
            continue
        ex = load_existing(spec)
        if ex is not None:
            reused.append(ex)
        else:
            todo.append(spec)
    return reused, todo


def _mean_sem(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    n = len(xs)
    if n == 0:
        return dict(n=0, mean=None, sem=None)
    mu = sum(xs) / n
    if n == 1:
        return dict(n=1, mean=mu, sem=None)
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return dict(n=n, mean=mu, sem=math.sqrt(var / n))


def _load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def summarize_round(round_id, out_path=None):
    """Official per-round summary: indices of this round only. Surplus N=6 stays off-summary."""
    idxs = set(round_indices(round_id))
    entropy = defaultdict(list)
    unresolved = defaultdict(list)
    for n in P.NS:
        for i in sorted(idxs):
            seed = P.main_seed(n, i)
            path = MAIN_RUNS / f"entropy_N{n}_seed{seed}.json"
            rec = _load_json(path)
            if rec is None:
                continue
            rec["index"] = i
            entropy[n].append(rec)
            if not rec.get("valid_for_fits", False) or rec.get("label") != "interior_unique_GS_entropy_peak":
                unresolved[n].append(dict(index=i, seed=seed, label=rec.get("label"),
                                          valid_for_fits=rec.get("valid_for_fits")))

    panels = {}
    raw_starts = {}
    for n in P.NS:
        panels[n] = {}
        for L in P.DEPTHS:
            panels[n][L] = {}
            for method in P.METHODS:
                spsa_rows, bfgs_sel, all_starts = [], [], []
                for i in sorted(idxs):
                    seed = P.main_seed(n, i)
                    sp = _load_json(MAIN_RUNS / f"spsa_N{n}_seed{seed}_L{L}_{method}.json")
                    if sp:
                        spsa_rows.append(sp)
                    starts = []
                    for k in range(P.BFGS_STARTS):
                        st = _load_json(MAIN_RUNS / f"bfgs_N{n}_seed{seed}_L{L}_{method}_start{k}.json")
                        if st:
                            starts.append(st)
                    if len(starts) == P.BFGS_STARTS:
                        sel = select_by_energy(starts)
                        picked = next(s for s in starts if s.get("start_index") == sel["selected_start"])
                        sel.update(index=i, seed=seed, N=n, L=L, method=method)
                        sel["energy_error"] = picked.get("energy_error")
                        sel["energy_error_over_N"] = picked.get("energy_error_over_N")
                        sel["best_E"] = picked.get("best_E")
                        sel["F"] = picked.get("F")
                        sel["nfev"] = picked.get("nfev")
                        sel["valid_for_fits"] = picked.get("valid_for_fits")
                        bfgs_sel.append(sel)
                    all_starts.append(dict(index=i, seed=seed, n_starts=len(starts),
                                           starts=[{k: s.get(k) for k in
                                                    ("start_index", "best_E", "F", "nfev", "nit",
                                                     "success", "tightly_converged", "hit_iteration_cap",
                                                     "energy_error")} for s in starts]))
                # paired SPSA vs BFGS on shared finished instances
                paired = []
                bfgs_by = {int(r["index"]): r for r in bfgs_sel}
                for sp in spsa_rows:
                    i = int(sp.get("index", sp["seed"] - (831100000 + n * 1000)))
                    if i in bfgs_by:
                        paired.append(dict(
                            index=i, seed=sp["seed"],
                            spsa_E=sp.get("best_E"), bfgs_E=bfgs_by[i].get("best_E"),
                            spsa_err=sp.get("energy_error"), bfgs_err=bfgs_by[i].get("energy_error"),
                            spsa_F=sp.get("F"), bfgs_F=bfgs_by[i].get("F"),
                            delta_E=(bfgs_by[i].get("best_E") - sp.get("best_E"))
                            if (bfgs_by[i].get("best_E") is not None and sp.get("best_E") is not None) else None,
                        ))
                panels[n][L][method] = dict(
                    spsa_energy_error=_mean_sem([r.get("energy_error") for r in spsa_rows]),
                    spsa_energy_error_over_N=_mean_sem([r.get("energy_error_over_N") for r in spsa_rows]),
                    spsa_F=_mean_sem([r.get("F") for r in spsa_rows]),
                    spsa_best_E=_mean_sem([r.get("best_E") for r in spsa_rows]),
                    spsa_n=len(spsa_rows),
                    bfgs_energy_selected_error=_mean_sem([r.get("energy_error") for r in bfgs_sel]),
                    bfgs_energy_selected_error_over_N=_mean_sem([r.get("energy_error_over_N") for r in bfgs_sel]),
                    bfgs_energy_selected_F=_mean_sem([r.get("F") for r in bfgs_sel]),
                    bfgs_energy_selected_E=_mean_sem([r.get("best_E") for r in bfgs_sel]),
                    bfgs_n=len(bfgs_sel),
                    paired_bfgs_minus_spsa_E=_mean_sem([p["delta_E"] for p in paired]),
                    paired_n=len(paired),
                    raw_spsa=[{k: r.get(k) for k in
                               ("index", "seed", "best_E", "F", "calls", "energy_error",
                                "energy_error_over_N", "valid_for_fits")} for r in spsa_rows],
                    raw_bfgs_selected=bfgs_sel,
                    paired=paired,
                )
                raw_starts[f"N{n}_L{L}_{method}"] = all_starts

    # surplus N=6 retained off official summary
    surplus = dict(N=6, retained=True, official_round_indices=sorted(idxs),
                   note="completed N=6 indices outside this round are retained on disk and excluded from official round means")
    extra = 0
    for p in MAIN_RUNS.glob("entropy_N6_seed*.json"):
        try:
            seed = int(p.name.split("seed")[1].split(".")[0])
            idx = seed - 831106000
            if idx not in idxs and 0 <= idx < P.N_INSTANCES:
                extra += 1
        except Exception:
            pass
    surplus["entropy_files_outside_round"] = extra

    ent_summary = {}
    for n, rows in entropy.items():
        ent_summary[n] = dict(
            n=len(rows),
            S=_mean_sem([r.get("S") for r in rows]),
            b=_mean_sem([r.get("b") for r in rows]),
            n_valid=sum(1 for r in rows if r.get("valid_for_fits")),
            n_unresolved=len(unresolved.get(n, [])),
            unresolved=unresolved.get(n, []),
            raw=[{k: r.get(k) for k in ("index", "seed", "b", "S", "E0", "gap", "even_gap",
                                        "label", "valid_for_fits")} for r in rows],
        )

    out = dict(
        round=int(round_id),
        indices=sorted(idxs),
        shanghai=now_shanghai(),
        official_only=True,
        selection="energy_not_F",
        no_early_stop_at_good_F=True,
        no_selective_drop_of_unresolved_peaks=True,
        frozen_instance_count_cap=P.N_INSTANCES,
        entropy=ent_summary,
        costs=panels,
        raw_16_starts=raw_starts,
        surplus_N6_retained_off_summary=surplus,
        coupling_variance="1/N",
    )
    SUMMARIES.mkdir(parents=True, exist_ok=True)
    dest = Path(out_path) if out_path else (SUMMARIES / f"round_{int(round_id):02d}_summary.json")
    save_json(dest, out)
    out["summary_file"] = str(dest)
    return out


def completeness(round_id):
    idxs = round_indices(round_id)
    counts = {}
    missing = []
    for n in P.NS:
        e_ok = s_ok = b_ok = 0
        for i in idxs:
            seed = P.main_seed(n, i)
            if load_existing(dict(task="entropy", N=n, seed=seed, index=i)):
                e_ok += 1
            else:
                missing.append(("entropy", n, i))
            for L in P.DEPTHS:
                for method in P.METHODS:
                    if load_existing(dict(task="spsa", N=n, seed=seed, L=L, method=method, index=i)):
                        s_ok += 1
                    else:
                        missing.append(("spsa", n, i, L, method))
                    starts = sum(1 for k in range(P.BFGS_STARTS)
                                 if load_existing(dict(task="bfgs", N=n, seed=seed, L=L, method=method, start_index=k)))
                    if starts == P.BFGS_STARTS:
                        b_ok += 1
                    else:
                        missing.append(("bfgs", n, i, L, method, starts))
        counts[n] = dict(entropy=e_ok, entropy_target=len(idxs),
                         spsa=s_ok, spsa_target=len(idxs) * len(P.DEPTHS) * len(P.METHODS),
                         bfgs_cells=b_ok, bfgs_target=len(idxs) * len(P.DEPTHS) * len(P.METHODS))
    all_n_done = all(v["entropy"] == v["entropy_target"] and v["spsa"] == v["spsa_target"]
                     and v["bfgs_cells"] == v["bfgs_target"] for v in counts.values())
    return dict(round=round_id, all_N_complete=all_n_done, counts=counts,
                n_missing=len(missing), missing_head=missing[:20])
