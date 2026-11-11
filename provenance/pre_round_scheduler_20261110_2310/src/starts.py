"""Deterministic frozen starts for SPSA (2) and BFGS (16)."""
from __future__ import annotations
import numpy as np
from .hamiltonian import param_count, NS
from .spsa import spsa_starts
from .bfgs import all_starts, N_STARTS
from .io_util import x_sha256, save_json, sha256_file
from .paths import DATA, MANIFESTS

DEPTHS = (1, 2, 3, 4, 5, 6)
METHODS = ("nov3", "qaoa")


def freeze_cell_starts(seed, L, method):
    seed = int(seed); L = int(L)
    spsa = []
    xs, _ = spsa_starts(seed, L, method)
    for i, x in enumerate(xs):
        spsa.append(dict(
            start_index=i,
            width=0.16 if i == 0 else 0.6,
            width_class="narrow" if i == 0 else "broad",
            x=np.asarray(x, float).tolist(),
            x_sha256=x_sha256(x),
        ))
    bfgs = []
    for st in all_starts(seed, L, method):
        rec = dict(st)
        rec["x"] = np.asarray(st["x"], float).tolist()
        bfgs.append(rec)
    return dict(
        seed=seed, L=L, method=method,
        param_count=param_count(method, L),
        spsa_starts=spsa, bfgs_starts=bfgs,
        spsa_n=2, bfgs_n=N_STARTS,
    )


def pack_panel_starts(n, seeds, out_dir=None):
    """Compact npz per (N,L,method): SPSA (n_inst, 2, p) and BFGS (n_inst, 16, p)."""
    out_dir = out_dir or (DATA / "starts")
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in seeds]
    files = []
    for L in DEPTHS:
        for method in METHODS:
            p = param_count(method, L)
            S = np.zeros((len(seeds), 2, p), dtype=np.float64)
            B = np.zeros((len(seeds), N_STARTS, p), dtype=np.float64)
            spsa_sha = []
            bfgs_sha = []
            for i, seed in enumerate(seeds):
                cell = freeze_cell_starts(seed, L, method)
                for j, st in enumerate(cell["spsa_starts"]):
                    S[i, j] = st["x"]
                    spsa_sha.append(st["x_sha256"])
                for j, st in enumerate(cell["bfgs_starts"]):
                    B[i, j] = st["x"]
                    bfgs_sha.append(st["x_sha256"])
            path_s = out_dir / f"N{n}_L{L}_{method}_spsa_starts.npz"
            path_b = out_dir / f"N{n}_L{L}_{method}_bfgs_starts.npz"
            np.savez_compressed(path_s, x=S, seeds=np.asarray(seeds, dtype=np.int64), widths=np.array([0.16, 0.6]))
            np.savez_compressed(path_b, x=B, seeds=np.asarray(seeds, dtype=np.int64),
                                widths=np.array([0.6 if k % 2 == 0 else 0.16 for k in range(N_STARTS)]))
            files.append(dict(
                N=n, L=L, method=method,
                spsa_file=str(path_s), spsa_sha256=sha256_file(path_s),
                bfgs_file=str(path_b), bfgs_sha256=sha256_file(path_b),
                n_instances=len(seeds), spsa_shape=list(S.shape), bfgs_shape=list(B.shape),
            ))
    return files
