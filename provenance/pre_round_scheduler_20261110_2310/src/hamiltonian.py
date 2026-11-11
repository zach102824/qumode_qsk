"""Quantum SK Hamiltonian and Nov3/QAOA statevectors.

Coupling convention (frozen): J_ij ~ N(0, 1/N) for i<j, then symmetrized.
H = sum_{i<j} J_ij Z_i Z_j - B sum_i X_i
(driver sign; equivalent in law to a global minus on J). Variance is 1/N, not 1/(2N).
Initial state is |+>^{otimes N}. Exact statevector energy, no shots.
"""
from __future__ import annotations
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[k] = "1"

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

NS = (6, 9, 12, 15, 18, 21)
VARIANCE = "1/N"


def make_J(n, seed):
    n = int(n); seed = int(seed)
    rng = np.random.default_rng(seed)
    J = np.zeros((n, n), dtype=np.float64)
    iu = np.triu_indices(n, 1)
    J[iu] = rng.normal(0.0, 1.0 / np.sqrt(n), size=iu[0].size)
    J = J + J.T
    return J


def param_count(method, L):
    L = int(L)
    if method == "nov3":
        return 3 + 3 * L
    if method == "qaoa":
        return 2 * L
    raise ValueError(method)


def walsh(v, n):
    v = v.copy()
    for q in range(n):
        t = v.reshape(-1, 2, 1 << q)
        a = t[:, 0].copy()
        b = t[:, 1].copy()
        t[:, 0] = (a + b) / np.sqrt(2)
        t[:, 1] = (a - b) / np.sqrt(2)
    return v


def rx(v, b, n):
    c, s = np.cos(b), -1j * np.sin(b)
    v = v.copy()
    for q in range(n):
        t = v.reshape(-1, 2, 1 << q)
        a = t[:, 0].copy()
        d = t[:, 1].copy()
        t[:, 0] = c * a + s * d
        t[:, 1] = s * a + c * d
    return v


def _zz_diag(mat, n):
    """Build sum_{i<j} mat_ij Z_i Z_j without an N x 2^N stack."""
    dim = 1 << n
    ids = np.arange(dim, dtype=np.int64)
    out = np.zeros(dim, dtype=np.float64)
    iu, ju = np.triu_indices(n, 1)
    for i, j in zip(iu, ju):
        coeff = float(mat[i, j])
        if coeff == 0.0:
            continue
        zi = 1 - 2 * ((ids >> i) & 1)
        zj = 1 - 2 * ((ids >> j) & 1)
        out += coeff * zi * zj
    return out


def _shell_ops(n, groups):
    dim = 1 << n
    ids = np.arange(dim, dtype=np.int64)
    shell = np.zeros((dim, 3), dtype=np.float64)
    for C in groups:
        k = np.zeros(dim, dtype=np.float64)
        for q in C:
            k += (ids >> q) & 1
        shell[:, 0] += k
        shell[:, 1] += k * (k - 1) / 2.0
        shell[:, 2] += k * (k - 1) * (k - 2) / 6.0
    return shell


def apply_X(v, n):
    """Apply sum_q X_q without storing flip index tables."""
    acc = np.zeros_like(v)
    for q in range(n):
        t = v.reshape(-1, 2, 1 << q)
        block = np.empty_like(t)
        block[:, 0] = t[:, 1]
        block[:, 1] = t[:, 0]
        acc += block.reshape(v.shape)
    return acc


class Model:
    def __init__(self, n, seed, b, J=None, compute_gs=False):
        n = int(n); seed = int(seed)
        if n not in NS and n % 3 != 0:
            raise ValueError(f"N={n} not a supported multiple of 3")
        self.n = n
        self.seed = seed
        self.b = float(b)
        self.plus = np.ones(1 << n, dtype=complex) / np.sqrt(1 << n)
        self.J = np.array(J, dtype=np.float64) if J is not None else make_J(n, seed)
        if self.J.shape != (n, n):
            raise ValueError("J shape")
        iu = np.triu_indices(n, 1)
        # Features from this same J (driver convention).
        self.groups = [[i, i + n // 3, i + 2 * n // 3] for i in range(n // 3)]
        K = self.J @ self.J
        np.fill_diagonal(K, 0.0)
        M = np.zeros_like(K)
        for a, C in enumerate(self.groups):
            for T in self.groups[a + 1:]:
                mmean = self.J[np.ix_(C, T)].mean()
                M[np.ix_(C, T)] = mmean
                M[np.ix_(T, C)] = mmean
        jrms = float(np.sqrt(np.mean(self.J[iu] ** 2)))
        self.diags = []
        for mat in (self.J, K, M):
            mat = mat.copy()
            norm = float(np.sqrt(np.mean(mat[iu] ** 2)))
            if norm:
                mat = mat * (jrms / norm)
            self.diags.append(_zz_diag(mat, n))
        self.shell = _shell_ops(n, self.groups)
        self.gs = None
        self.e0 = None
        self.e1 = None
        self.gap = None
        self.residual = None
        if compute_gs:
            self.compute_gs_fullspace()

    @property
    def J_upper(self):
        iu = np.triu_indices(self.n, 1)
        return self.J[iu].copy()

    def apply_H(self, v):
        return self.diags[0] * v - self.b * apply_X(v, self.n)

    def energy(self, v):
        return float(np.vdot(v, self.apply_H(v)).real)

    def fidelity(self, v):
        if self.gs is None:
            raise RuntimeError("GS not loaded")
        return float(abs(np.vdot(self.gs, v)) ** 2)

    def state(self, x, L, method="nov3"):
        x = np.asarray(x, float)
        n = self.n
        if method == "nov3":
            assert x.shape == (param_count("nov3", L),)
            phase = np.exp(-1j * x[0] * (self.diags[0] + x[1] * self.diags[1] + x[2] * self.diags[2]))
            v = self.plus.copy()
            for row in x[3:].reshape(L, 3):
                v = walsh(np.exp(-1j * self.shell @ row) * walsh(v * phase, n), n)
            return v
        if method == "qaoa":
            assert x.shape == (param_count("qaoa", L),)
            v = self.plus.copy()
            for g, beta in x.reshape(L, 2):
                v = rx(v * np.exp(-1j * g * self.diags[0]), beta, n)
            return v
        raise ValueError(method)

    def parity_px(self, v):
        # Global bit flip is reversal of the computational-basis vector.
        return float(np.vdot(v, v[::-1]).real)

    def compute_gs_fullspace(self, k=2, tol=1e-12):
        """Sparse full-space eigsh via LinearOperator; never a dense matrix."""
        n = self.n
        dim = 1 << n
        D0 = self.diags[0]
        b = self.b

        def mv(v):
            v = np.asarray(v, dtype=np.float64)
            return D0 * v - b * apply_X(v, n)

        op = LinearOperator((dim, dim), matvec=mv, dtype=np.float64)
        v0 = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
        w, u = eigsh(op, k=k, which="SA", v0=v0, tol=tol, maxiter=200000)
        order = np.argsort(w)
        e0 = float(w[order[0]])
        e1 = float(w[order[1]]) if k > 1 else None
        psi = np.asarray(u[:, order[0]], dtype=np.float64).copy()
        if float(np.dot(v0, psi)) < 0:
            psi = -psi
        residual = float(np.linalg.norm(mv(psi) - e0 * psi))
        self.gs = psi.astype(complex)
        self.e0 = e0
        self.e1 = e1
        self.gap = None if e1 is None else (e1 - e0)
        self.residual = residual
        return self

    def load_gs(self, gs, E0, gap=None, residual=None):
        self.gs = np.asarray(gs, dtype=complex)
        self.e0 = float(E0)
        self.gap = None if gap is None else float(gap)
        if residual is None:
            self.residual = float(np.linalg.norm(self.apply_H(self.gs) - self.e0 * self.gs))
        else:
            self.residual = float(residual)
        return self


def nov3_energy_grad(m, x, L):
    x = np.asarray(x, float)
    n = m.n
    p = param_count("nov3", L)
    assert x.shape == (p,)
    g0, a1, a2 = float(x[0]), float(x[1]), float(x[2])
    gen = m.diags[0] + a1 * m.diags[1] + a2 * m.diags[2]
    phase = np.exp(-1j * g0 * gen)
    dphase_dg0 = (-1j * gen) * phase
    dphase_da1 = (-1j * g0 * m.diags[1]) * phase
    dphase_da2 = (-1j * g0 * m.diags[2]) * phase
    v = m.plus.copy()
    dv = np.zeros((p, v.size), dtype=complex)
    for ell in range(L):
        v_new = v * phase
        dv_new = dv * phase
        dv_new[0] += v * dphase_dg0
        dv_new[1] += v * dphase_da1
        dv_new[2] += v * dphase_da2
        v, dv = v_new, dv_new
        v = walsh(v, n)
        for k in range(p):
            dv[k] = walsh(dv[k], n)
        row = x[3 + 3 * ell:6 + 3 * ell]
        ang = m.shell @ row
        sph = np.exp(-1j * ang)
        v_new = v * sph
        dv_new = dv * sph
        for j in range(3):
            dv_new[3 + 3 * ell + j] += v * ((-1j * m.shell[:, j]) * sph)
        v, dv = v_new, dv_new
        v = walsh(v, n)
        for k in range(p):
            dv[k] = walsh(dv[k], n)
    nrm = float(np.vdot(v, v).real)
    if abs(nrm - 1.0) >= 1e-12:
        raise AssertionError(f"state norm {nrm}")
    Hv = m.apply_H(v)
    E = float(np.vdot(v, Hv).real)
    g = np.array([2 * float(np.vdot(dv[k], Hv).real) for k in range(p)], dtype=float)
    return E, g, v


def qaoa_energy_grad(m, x, L):
    x = np.asarray(x, float)
    n = m.n
    p = param_count("qaoa", L)
    assert x.shape == (p,)
    v = m.plus.copy()
    dv = np.zeros((p, v.size), dtype=complex)
    D = m.diags[0]
    for ell in range(L):
        ig, ib = 2 * ell, 2 * ell + 1
        gamma, beta = float(x[ig]), float(x[ib])
        phase = np.exp(-1j * gamma * D)
        v = v * phase
        dv = dv * phase
        dv[ig] += (-1j * D) * v
        v = rx(v, beta, n)
        for k in range(p):
            dv[k] = rx(dv[k], beta, n)
        Xv = apply_X(v, n)
        dv[ib] += (-1j) * Xv
    nrm = float(np.vdot(v, v).real)
    if abs(nrm - 1.0) >= 1e-12:
        raise AssertionError(f"qaoa state norm {nrm}")
    Hv = m.apply_H(v)
    E = float(np.vdot(v, Hv).real)
    g = np.array([2 * float(np.vdot(dv[k], Hv).real) for k in range(p)], dtype=float)
    return E, g, v


def energy_grad(m, x, method, L):
    if method == "nov3":
        return nov3_energy_grad(m, x, L)
    if method == "qaoa":
        return qaoa_energy_grad(m, x, L)
    raise ValueError(method)


def central_diff_grad(m, x, method, L, eps=1e-7):
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    for k in range(len(x)):
        xp = x.copy(); xm = x.copy()
        xp[k] += eps; xm[k] -= eps
        Ep = m.energy(m.state(xp, L, method))
        Em = m.energy(m.state(xm, L, method))
        g[k] = (Ep - Em) / (2 * eps)
    return g
