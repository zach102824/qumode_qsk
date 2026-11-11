"""Frozen science and seed protocol. Do not change without a new freeze file."""
NS = (6, 9, 12, 15, 18, 21)
N_INSTANCES = 200
DEPTHS = (1, 2, 3, 4, 5, 6)
METHODS = ("nov3", "qaoa")
COUPLING_VARIANCE = "1/N"
HAMILTONIAN = "H = sum_{i<j} J_ij Z_i Z_j - B sum_i X_i,  J_ij ~ N(0, 1/N) for i<j, symmetrized"
FIELD = "per-instance finite-N ground-state von Neumann entropy peak on A=lowest floor(N/2) spins; not thermodynamic critical; not Gamma=1"
INIT_STATE = "|+>^{otimes N}"
SPSA_STARTS = 2
SPSA_BUDGET = "Q=1024*L inclusive"
BFGS_STARTS = 16
BFGS_MAXITER = 2000
BFGS_GTOL = 1e-10
BFGS_TIGHT = 1e-8
SEED_BASE = 831100000
PILOT_OFFSET = 800
N_PILOT_PER_N = 1
NO_SEED_REPLACEMENT = True
NO_TRAIN_TEST_SPLIT = True
NO_SPSA_WARM_ARM = True
STORE_GS_VECTORS = True

def main_seed(n, index):
    """index in 0..199"""
    if not (0 <= int(index) < N_INSTANCES):
        raise ValueError(index)
    return int(SEED_BASE + int(n) * 1000 + int(index))

def main_seeds(n):
    return [main_seed(n, i) for i in range(N_INSTANCES)]

def pilot_seed(n, index=0):
    """Disjoint from main: offset 800+."""
    if not (0 <= int(index) < 200):  # reserved 800-999 unused by main 0-199
        raise ValueError(index)
    return int(SEED_BASE + int(n) * 1000 + PILOT_OFFSET + int(index))

def pilot_seeds(n, k=N_PILOT_PER_N):
    return [pilot_seed(n, i) for i in range(k)]

def all_main_seeds():
    return {int(n): main_seeds(n) for n in NS}

def all_pilot_seeds():
    return {int(n): pilot_seeds(n) for n in NS}

def assert_disjoint():
    for n in NS:
        m, p = set(main_seeds(n)), set(pilot_seeds(n, 8))
        if m & p:
            raise AssertionError((n, m & p))
        banned = set(range(10000, 10032)) | set(range(712000, 712030))
        if (m | p) & banned:
            raise AssertionError("collision with historical seeds")
