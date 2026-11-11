"""Write frozen protocol, seed map, and compact start archives. No optimizer fits."""
from __future__ import annotations
import hashlib
from . import protocol as P
from .starts import pack_panel_starts
from .spsa import query_schedule
from .io_util import save_json, sha256_file, sha256_json, now_shanghai, pin_source_files
from .paths import PROTOCOL, MANIFESTS, DATA, SRC, ROOT
from .resources import allocated_cpus, memory_headroom_kib


def seed_map():
    main = P.all_main_seeds()
    pilot = P.all_pilot_seeds()
    P.assert_disjoint()
    return dict(
        seed_base=P.SEED_BASE,
        formula_main="SEED_BASE + N*1000 + index, index=0..199",
        formula_pilot="SEED_BASE + N*1000 + 800 + index, index=0..0 (one disjoint seed per N)",
        main={str(n): main[n] for n in P.NS},
        pilot={str(n): pilot[n] for n in P.NS},
        n_main_per_N=P.N_INSTANCES,
        n_pilot_per_N=P.N_PILOT_PER_N,
        historical_seeds_excluded=["10000-10031", "712000-712029"],
        no_train_test_split=True,
        no_overflow_replacement=True,
        keep_all_200_and_flag_invalid=True,
    )


def build_protocol_document(seedmap, start_files, source_pins):
    sched = {str(L): query_schedule(L) for L in P.DEPTHS}
    cap = allocated_cpus()
    mem = memory_headroom_kib()
    doc = dict(
        title="qumode_qsk phase-1 frozen protocol",
        frozen_before_main_fits=True,
        frozen_shanghai=now_shanghai(),
        authorization=["pt3487u", "pt3488u"],
        location="/workspace/qumode_qsk",
        no_repo_or_cloud_mutations=True,
        science=dict(
            Ns=list(P.NS),
            n_instances_per_N=P.N_INSTANCES,
            depths=list(P.DEPTHS),
            methods=list(P.METHODS),
            coupling_variance=P.COUPLING_VARIANCE,
            hamiltonian=P.HAMILTONIAN,
            field=P.FIELD,
            init_state=P.INIT_STATE,
            exact_energy_no_shots=True,
            entropy_cut="A=lowest floor(N/2) logical spins; natural-log von Neumann; not cavity",
            parity="positive even branch; B=0 not on primary grid; invalid peaks flagged not replaced",
            store_GS_vectors_if_disk_permits=P.STORE_GS_VECTORS,
        ),
        spsa=dict(
            starts=2, widths=[0.16, 0.6],
            budget="Q=1024*L inclusive",
            cold=True, no_warm=True, no_layer_transfer=True,
            biases="nov3 x0+=0.2; qaoa gamma+=0.12 beta-=0.22",
            rng="default_rng(seed*107+L*2003+{nov3:11,qaoa:29})",
            recipe="R9 depth_core IID: 2 starts + 16 calibration + 4T + 2*floor(T/5) monitors + padding + 6 readouts = Q",
            query_schedule=sched,
            selection="lowest energy among all charged queries; F after the fact",
        ),
        bfgs=dict(
            starts=16, widths="even start_index width 0.6 broad, odd 0.16 narrow",
            n_narrow=8, n_broad=8,
            cold=True, no_spsa_warm=True, no_layer_transfer=True,
            biases="nov3 x0+=0.2; qaoa gamma+=0.12 beta-=0.22",
            rng="default_rng(seed*1000003+L*10007+{nov3:11,qaoa:29}+start_index*97)",
            maxiter=P.BFGS_MAXITER, gtol=P.BFGS_GTOL, grad_norm="inf",
            tight_convergence="success and recomputed ||g||_inf < 1e-8 and ||g||_2 < 1e-8",
            no_1024L_cap=True,
            selection="lowest-energy start among 16; also report max F if different start",
            jacobian="analytic forward-mode, pre-fit FD audit",
        ),
        scheduler=dict(
            kind="campaign_local_between_job_admission",
            blas_threads=1,
            nice=10,
            cap_from="cgroup cpu.max quota or cpuset.effective/affinity, not host os.cpu_count",
            memory_caps=True,
            midjob_SIGSTOP="not used in this initial scheduler; do not claim instant preemption",
            never_touch_other_agents_or_historical_jobs=True,
            not_a_routine_or_unbounded_autorestart=True,
        ),
        workloads=dict(
            instances=6 * 200,
            spsa_cells=6 * 200 * 6 * 2,
            spsa_queries=6 * 200 * 2 * 1024 * 21,
            bfgs_cells=6 * 200 * 6 * 2,
            bfgs_starts=6 * 200 * 6 * 2 * 16,
            note="BFGS nfev unknown until after pilot; no wall-time claim in this file",
        ),
        seeds=seedmap,
        starts=start_files,
        source_pins=source_pins,
        resource_snapshot_at_freeze=dict(cpu=cap, memory=mem),
        main_fits_not_authorized_until_pilot_checkpoint_relayed=True,
    )
    # sha over a copy without the sha field
    doc["protocol_sha256"] = sha256_json({k: v for k, v in doc.items() if k != "protocol_sha256"})
    return doc


def freeze():
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    DATA.joinpath("starts").mkdir(parents=True, exist_ok=True)
    sm = seed_map()
    save_json(PROTOCOL / "SEED_MAP.json", sm)
    save_json(MANIFESTS / "SEED_MAP.json", sm)
    # Compact starts for main and pilot (generation only; no fits).
    start_files = []
    for n in P.NS:
        start_files.extend(pack_panel_starts(n, P.main_seeds(n), DATA / "starts" / "main"))
        start_files.extend(pack_panel_starts(n, P.pilot_seeds(n), DATA / "starts" / "pilot"))
    save_json(MANIFESTS / "FROZEN_STARTS_INDEX.json", dict(
        frozen_shanghai=now_shanghai(), files=start_files,
        note="Starts frozen from the RNG recipes before any ansatz fits.",
    ))
    pins = pin_source_files(SRC)
    doc = build_protocol_document(sm, start_files, pins)
    path = PROTOCOL / "PROTOCOL_FROZEN.json"
    save_json(path, doc)
    return dict(protocol=str(path), sha256=doc["protocol_sha256"], n_start_files=len(start_files), shanghai=doc["frozen_shanghai"])
