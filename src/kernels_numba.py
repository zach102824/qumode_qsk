"""Numba hot kernels for Walsh/RX/apply_X/ZZ/shell/energy.

Scientifically equivalent to the previous NumPy butterfly / bit-feature
implementation. fastmath is OFF until an independent audit has passed.
All kernels are single-threaded; process-level admission owns the 8 cores.
"""
from __future__ import annotations
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_DEFAULT_NUM_THREADS", "1")

import numpy as np
from numba import njit

# nopython, no fastmath, explicit types. cache=True after first successful audit
# is enabled via QSK_NUMBA_CACHE (default True for production, False in audit).
_CACHE = os.environ.get("QSK_NUMBA_CACHE", "1") not in ("0", "false", "False")
_JIT = dict(fastmath=False, cache=_CACHE, nogil=True, parallel=False)


@njit(**_JIT)
def walsh_inplace_c(v, n):
    dim = v.size
    s = np.float64(np.sqrt(2.0))
    for q in range(n):
        step = 1 << q
        span = step << 1
        i = 0
        while i < dim:
            for j in range(step):
                a = v[i + j]
                b = v[i + j + step]
                v[i + j] = (a + b) / s
                v[i + j + step] = (a - b) / s
            i += span
    return v


@njit(**_JIT)
def walsh_c(v, n):
    out = v.copy()
    walsh_inplace_c(out, n)
    return out


@njit(**_JIT)
def rx_inplace_c(v, beta, n):
    dim = v.size
    c = np.cos(beta)
    s = -1j * np.sin(beta)
    for q in range(n):
        step = 1 << q
        span = step << 1
        i = 0
        while i < dim:
            for j in range(step):
                a = v[i + j]
                d = v[i + j + step]
                v[i + j] = c * a + s * d
                v[i + j + step] = s * a + c * d
            i += span
    return v


@njit(**_JIT)
def rx_c(v, beta, n):
    out = v.copy()
    rx_inplace_c(out, beta, n)
    return out


@njit(**_JIT)
def apply_X_c(v, n):
    dim = v.size
    acc = np.zeros(dim, dtype=np.complex128)
    for q in range(n):
        step = 1 << q
        span = step << 1
        i = 0
        while i < dim:
            for j in range(step):
                a = v[i + j]
                b = v[i + j + step]
                acc[i + j] += b
                acc[i + j + step] += a
            i += span
    return acc


@njit(**_JIT)
def apply_X_r(v, n):
    dim = v.size
    acc = np.zeros(dim, dtype=np.float64)
    for q in range(n):
        step = 1 << q
        span = step << 1
        i = 0
        while i < dim:
            for j in range(step):
                a = v[i + j]
                b = v[i + j + step]
                acc[i + j] += b
                acc[i + j + step] += a
            i += span
    return acc


@njit(**_JIT)
def apply_H_c(v, D0, b, n):
    return D0 * v - b * apply_X_c(v, n)


@njit(**_JIT)
def apply_H_r(v, D0, b, n):
    return D0 * v - b * apply_X_r(v, n)


@njit(**_JIT)
def energy_c(v, D0, b, n):
    Hv = apply_H_c(v, D0, b, n)
    # vdot = sum conj(v)*Hv
    acc = np.complex128(0.0)
    for i in range(v.size):
        acc += np.conj(v[i]) * Hv[i]
    return acc.real, Hv


@njit(**_JIT)
def zz_diag(J, n):
    dim = 1 << n
    out = np.zeros(dim, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            coeff = J[i, j]
            if coeff == 0.0:
                continue
            for s in range(dim):
                zi = 1 - 2 * ((s >> i) & 1)
                zj = 1 - 2 * ((s >> j) & 1)
                out[s] += coeff * zi * zj
    return out


@njit(**_JIT)
def shell_ops(groups, n):
    # groups: (n/3, 3) int64
    dim = 1 << n
    shell = np.zeros((dim, 3), dtype=np.float64)
    ngrp = groups.shape[0]
    for g in range(ngrp):
        for s in range(dim):
            k = 0.0
            for t in range(3):
                q = groups[g, t]
                k += (s >> q) & 1
            shell[s, 0] += k
            shell[s, 1] += k * (k - 1) * 0.5
            shell[s, 2] += k * (k - 1) * (k - 2) / 6.0
    return shell


@njit(**_JIT)
def mul_diag_c(v, diag_phase):
    out = np.empty_like(v)
    for i in range(v.size):
        out[i] = v[i] * diag_phase[i]
    return out


def warmup(nmax=8):
    """Touch every kernel once so worker processes do not compile mid-cell."""
    n = int(nmax)
    dim = 1 << n
    vc = (np.ones(dim, dtype=np.complex128) / np.sqrt(dim))
    vr = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    J = np.eye(n, dtype=np.float64)
    J = J + J.T
    groups = np.array([[i, i + n // 3, i + 2 * n // 3] for i in range(n // 3)], dtype=np.int64) if n % 3 == 0 else np.zeros((0, 3), dtype=np.int64)
    D0 = zz_diag(J, n)
    _ = walsh_c(vc, n)
    _ = rx_c(vc, 0.1, n)
    _ = apply_X_c(vc, n)
    _ = apply_X_r(vr, n)
    _ = apply_H_c(vc, D0, 0.5, n)
    _ = apply_H_r(vr, D0, 0.5, n)
    _ = energy_c(vc, D0, 0.5, n)
    if groups.size:
        _ = shell_ops(groups, n)
    return dict(n=n, compiled=True, cache=_CACHE)
