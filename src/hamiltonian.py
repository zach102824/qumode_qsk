"""Quantum SK Hamiltonian and Nov3/QAOA statevectors.

Coupling convention (frozen): J_ij ~ N(0, 1/N) for i<j, then symmetrized.
H = sum_{i<j} J_ij Z_i Z_j - B sum_i X_i
(driver sign; equivalent in law to a global minus on J). Variance is 1/N, not 1/(2N).
Initial state is |+>^{otimes N}. Exact statevector energy, no shots.

Hot kernels are Numba-compiled (fastmath OFF). Gradients are reverse-mode
analytic and must match finite-difference / prior forward-mode in the audit.
"""
from __future__ import annotations
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS",
          "NUMBA_NUM_THREADS","NUMBA_DEFAULT_NUM_THREADS"):
    os.environ[k] = "1"

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from .kernels_numba import (
    walsh_c, walsh_inplace_c, rx_c, rx_inplace_c,
    apply_X_c, apply_X_r, apply_H_c, apply_H_r, energy_c,
    zz_diag, shell_ops, warmup as warmup_kernels,
)

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
    v = np.ascontiguousarray(v)
    if np.iscomplexobj(v):
        return walsh_c(v.astype(np.complex128, copy=False), int(n))
    # real path used only by tests; promote to complex kernel via real view
    out = walsh_c(v.astype(np.complex128), int(n))
    return np.real(out).astype(v.dtype, copy=False)


def rx(v, b, n):
    v = np.ascontiguousarray(v, dtype=np.complex128)
    return rx_c(v, float(b), int(n))


def _zz_diag(mat, n):
    return zz_diag(np.ascontiguousarray(mat, dtype=np.float64), int(n))


def _shell_ops(n, groups):
    g = np.ascontiguousarray(groups, dtype=np.int64)
    if g.ndim != 2:
        g = np.array(groups, dtype=np.int64)
    return shell_ops(g, int(n))


def apply_X(v, n):
    v = np.ascontiguousarray(v)
    n = int(n)
    if np.iscomplexobj(v):
        return apply_X_c(v.astype(np.complex128, copy=False), n)
    return apply_X_r(v.astype(np.float64, copy=False), n)


class Model:
    def __init__(self, n, seed, b, J=None, compute_gs=False):
        n = int(n); seed = int(seed)
        if n not in NS and n % 3 != 0:
            raise ValueError(f"N={n} not a supported multiple of 3")
        self.n = n
        self.seed = seed
        self.b = float(b)
        self.plus = np.ones(1 << n, dtype=np.complex128) / np.sqrt(1 << n)
        self.J = np.array(J, dtype=np.float64) if J is not None else make_J(n, seed)
        if self.J.shape != (n, n):
            raise ValueError("J shape")
        iu = np.triu_indices(n, 1)
        self.groups = [[i, i + n // 3, i + 2 * n // 3] for i in range(n // 3)]
        groups_arr = np.asarray(self.groups, dtype=np.int64)
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
        self.shell = _shell_ops(n, groups_arr)
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
        v = np.ascontiguousarray(v)
        D0 = self.diags[0]
        if np.iscomplexobj(v):
            return apply_H_c(v.astype(np.complex128, copy=False), D0, self.b, self.n)
        return apply_H_r(v.astype(np.float64, copy=False), D0, self.b, self.n)

    def energy(self, v):
        v = np.ascontiguousarray(v, dtype=np.complex128)
        E, _Hv = energy_c(v, self.diags[0], self.b, self.n)
        return float(E)

    def fidelity(self, v):
        if self.gs is None:
            raise RuntimeError("GS not loaded")
        return float(abs(np.vdot(self.gs, v)) ** 2)

    def state(self, x, L, method="nov3"):
        x = np.asarray(x, float)
        n = self.n
        if method == "nov3":
            assert x.shape == (param_count("nov3", L),)
            gen = self.diags[0] + x[1] * self.diags[1] + x[2] * self.diags[2]
            phase = np.exp(-1j * x[0] * gen)
            v = self.plus.copy()
            rows = x[3:].reshape(int(L), 3)
            for row in rows:
                v = v * phase
                walsh_inplace_c(v, n)
                ang = self.shell @ row
                v *= np.exp(-1j * ang)
                walsh_inplace_c(v, n)
            return v
        if method == "qaoa":
            assert x.shape == (param_count("qaoa", L),)
            v = self.plus.copy()
            D = self.diags[0]
            for g, beta in x.reshape(int(L), 2):
                v *= np.exp(-1j * g * D)
                rx_inplace_c(v, float(beta), n)
            return v
        raise ValueError(method)

    def parity_px(self, v):
        return float(np.vdot(v, v[::-1]).real)

    def compute_gs_fullspace(self, k=2, tol=1e-12):
        n = self.n
        dim = 1 << n
        D0 = self.diags[0]
        b = self.b

        def mv(v):
            v = np.ascontiguousarray(v, dtype=np.float64)
            return apply_H_r(v, D0, b, n)

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
        self.gs = psi.astype(np.complex128)
        self.e0 = e0
        self.e1 = e1
        self.gap = None if e1 is None else (e1 - e0)
        self.residual = residual
        return self

    def load_gs(self, gs, E0, gap=None, residual=None):
        self.gs = np.asarray(gs, dtype=np.complex128)
        self.e0 = float(E0)
        self.gap = None if gap is None else float(gap)
        if residual is None:
            self.residual = float(np.linalg.norm(self.apply_H(self.gs) - self.e0 * self.gs))
        else:
            self.residual = float(residual)
        return self


def nov3_energy_grad(m, x, L):
    """Reverse-mode analytic gradient of E = <psi|H|psi> for Nov3."""
    x = np.asarray(x, float)
    n = m.n
    L = int(L)
    p = param_count("nov3", L)
    assert x.shape == (p,)
    g0, a1, a2 = float(x[0]), float(x[1]), float(x[2])
    D0, D1, D2 = m.diags
    gen = D0 + a1 * D1 + a2 * D2
    phase = np.exp(-1j * g0 * gen)
    v = m.plus.copy()
    layer_in = []
    for ell in range(L):
        layer_in.append(v.copy())
        v = v * phase
        walsh_inplace_c(v, n)
        row = x[3 + 3 * ell:6 + 3 * ell]
        sph = np.exp(-1j * (m.shell @ row))
        v = v * sph
        walsh_inplace_c(v, n)
    nrm = float(np.vdot(v, v).real)
    if abs(nrm - 1.0) >= 1e-12:
        raise AssertionError(f"state norm {nrm}")
    E, Hv = energy_c(v, D0, m.b, n)
    E = float(E)
    lam = Hv.copy()
    g = np.zeros(p, dtype=np.float64)
    for ell in range(L - 1, -1, -1):
        vin = layer_in[ell]
        row = x[3 + 3 * ell:6 + 3 * ell]
        vA = vin * phase
        sph = np.exp(-1j * (m.shell @ row))
        vB = walsh_c(vA, n)
        vC = vB * sph
        walsh_inplace_c(lam, n)
        for j in range(3):
            g[3 + 3 * ell + j] += 2.0 * float(np.vdot(lam, (-1j) * m.shell[:, j] * vC).real)
        lam = lam * np.conj(sph)
        walsh_inplace_c(lam, n)
        g[0] += 2.0 * float(np.vdot(lam, (-1j) * gen * vA).real)
        g[1] += 2.0 * float(np.vdot(lam, (-1j) * g0 * D1 * vA).real)
        g[2] += 2.0 * float(np.vdot(lam, (-1j) * g0 * D2 * vA).real)
        lam = lam * np.conj(phase)
    return E, g, v


def qaoa_energy_grad(m, x, L):
    """Reverse-mode analytic gradient of E = <psi|H|psi> for QAOA."""
    x = np.asarray(x, float)
    n = m.n
    L = int(L)
    p = param_count("qaoa", L)
    assert x.shape == (p,)
    D = m.diags[0]
    v = m.plus.copy()
    layer_in = []
    for ell in range(L):
        layer_in.append(v.copy())
        gamma, beta = float(x[2 * ell]), float(x[2 * ell + 1])
        v *= np.exp(-1j * gamma * D)
        rx_inplace_c(v, beta, n)
    nrm = float(np.vdot(v, v).real)
    if abs(nrm - 1.0) >= 1e-12:
        raise AssertionError(f"qaoa state norm {nrm}")
    E, Hv = energy_c(v, D, m.b, n)
    E = float(E)
    lam = Hv.copy()
    g = np.zeros(p, dtype=np.float64)
    for ell in range(L - 1, -1, -1):
        vin = layer_in[ell]
        ig, ib = 2 * ell, 2 * ell + 1
        gamma, beta = float(x[ig]), float(x[ib])
        phase = np.exp(-1j * gamma * D)
        vA = vin * phase
        vB = rx_c(vA, beta, n)
        # backward through RX(beta): adjoint is RX(-beta); dE/dβ = 2 Re(λ† (-i X vB))
        Xv = apply_X_c(vB, n)
        g[ib] += 2.0 * float(np.vdot(lam, (-1j) * Xv).real)
        # apply RX(-beta) to adjoint
        rx_inplace_c(lam, -beta, n)
        # backward through cost phase
        g[ig] += 2.0 * float(np.vdot(lam, (-1j) * D * vA).real)
        lam = lam * np.conj(phase)
    return E, g, v


def nov3_energy_grad_forward(m, x, L):
    """Prior forward-mode analytic gradient (audit / reference only)."""
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
    dv = np.zeros((p, v.size), dtype=np.complex128)
    for ell in range(int(L)):
        v_new = v * phase
        dv_new = dv * phase
        dv_new[0] += v * dphase_dg0
        dv_new[1] += v * dphase_da1
        dv_new[2] += v * dphase_da2
        v, dv = v_new, dv_new
        v = walsh_c(v, n)
        for k in range(p):
            dv[k] = walsh_c(dv[k], n)
        row = x[3 + 3 * ell:6 + 3 * ell]
        sph = np.exp(-1j * (m.shell @ row))
        v_new = v * sph
        dv_new = dv * sph
        for j in range(3):
            dv_new[3 + 3 * ell + j] += v * ((-1j * m.shell[:, j]) * sph)
        v, dv = v_new, dv_new
        v = walsh_c(v, n)
        for k in range(p):
            dv[k] = walsh_c(dv[k], n)
    nrm = float(np.vdot(v, v).real)
    if abs(nrm - 1.0) >= 1e-12:
        raise AssertionError(f"state norm {nrm}")
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
