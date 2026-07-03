# -*- coding: utf-8 -*-
"""
holography_operator.py
======================
Self-contained A / A^H operator for holographic imaging.

Extracts the Green's function physics from holographic_imaging_gpu.py
into a reusable class that:
  - works standalone with CuPy  (inference, same as original script)
  - exposes gradient-aware torch wrappers  (LISTA training)

No global variables — all state lives in HolographyOperator instances.

Public API
----------
op = HolographyOperator(r_rx, r_vox, k, omega, mu0, dV, batch_rx=100)

CuPy interface (inference):
    b_cp = op.A(z_cp)       # (N_vox,) -> (N_rx,)  complex128
    z_cp = op.AH(b_cp)      # (N_rx,)  -> (N_vox,) complex128

PyTorch interface (training, gradient-aware):
    b_t  = op.A_torch(z_t)  # torch.cfloat, same CUDA device
    z_t  = op.AH_torch(b_t) # torch.cfloat, same CUDA device
"""

import cupy  as cp
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Green's function kernel  (verbatim from holographic_imaging_gpu.py)
# ---------------------------------------------------------------------------

def _green_yy(r_obs_batch: cp.ndarray, r_src: cp.ndarray,
              k: float, omega: float, mu0: float) -> cp.ndarray:
    """
    G_yy^E  — yy-component of the electric dyadic Green's function.

    Parameters
    ----------
    r_obs_batch : (B, 3)   observer positions  [CuPy]
    r_src       : (N_vox, 3) source positions  [CuPy]
    k           : wavenumber [rad/m]
    omega       : angular frequency [rad/s]
    mu0         : permeability of free space [H/m]

    Returns
    -------
    G : (B, N_vox) complex128 [CuPy]
    """
    dr    = r_obs_batch[:, cp.newaxis, :] - r_src[cp.newaxis, :, :]   # (B, N_vox, 3)
    R     = cp.linalg.norm(dr, axis=-1)                                # (B, N_vox)
    dy    = dr[:, :, 1]                                                # (B, N_vox)

    G_sc  = cp.exp(-1j * k * R) / (4.0 * cp.pi * R)

    alpha = -1j * k - 1.0 / R
    d2G   = G_sc * (  alpha**2 * (dy / R)**2
                    + (dy / R)**2 / R**2
                    + alpha * (1.0 - (dy / R)**2) / R )

    return -1j * omega * mu0 * (G_sc + d2G / k**2)


# ---------------------------------------------------------------------------
# HolographyOperator
# ---------------------------------------------------------------------------

class HolographyOperator:
    """
    Forward operator A and its adjoint A^H for holographic reconstruction.

    The forward model is:  b_scatter = A z
    where A[i, m] = G_yy^E(r_rx_i, r_vox_m) * dV

    Parameters
    ----------
    r_rx     : (N_rx, 3)   receiver positions   [CuPy float64]
    r_vox    : (N_vox, 3)  voxel positions       [CuPy float64]
    k        : wavenumber [rad/m]
    omega    : angular frequency [rad/s]
    mu0      : permeability of free space
    dV       : voxel volume [m^3]
    batch_rx : number of receivers per GPU batch (tune to fit VRAM)
    """

    def __init__(self, r_rx: cp.ndarray, r_vox: cp.ndarray,
                 k: float, omega: float, mu0: float,
                 dV: float, batch_rx: int = 100):
        self.r_rx     = r_rx
        self.r_vox    = r_vox
        self.k        = float(k)
        self.omega    = float(omega)
        self.mu0      = float(mu0)
        self.dV       = float(dV)
        self.batch_rx = int(batch_rx)
        self.N_rx     = int(r_rx.shape[0])
        self.N_vox    = int(r_vox.shape[0])

    # ------------------------------------------------------------------
    # CuPy interface  (drop-in replacement for the existing script)
    # ------------------------------------------------------------------

    def A(self, z_cp: cp.ndarray) -> cp.ndarray:
        """
        b = A z     (N_vox,) -> (N_rx,)  complex128

        Identical logic to A_matvec() in holographic_imaging_gpu.py.
        """
        result = cp.zeros(self.N_rx, dtype=cp.complex128)
        for i0 in range(0, self.N_rx, self.batch_rx):
            i1 = min(i0 + self.batch_rx, self.N_rx)
            G  = _green_yy(self.r_rx[i0:i1], self.r_vox,
                           self.k, self.omega, self.mu0)
            result[i0:i1] = (G * self.dV) @ z_cp
        return result

    def AH(self, b_cp: cp.ndarray) -> cp.ndarray:
        """
        z = A^H b   (N_rx,) -> (N_vox,)  complex128

        Identical logic to AH_matvec() in holographic_imaging_gpu.py.
        """
        result = cp.zeros(self.N_vox, dtype=cp.complex128)
        for i0 in range(0, self.N_rx, self.batch_rx):
            i1 = min(i0 + self.batch_rx, self.N_rx)
            G  = _green_yy(self.r_rx[i0:i1], self.r_vox,
                           self.k, self.omega, self.mu0)
            result += (cp.conj(G * self.dV) * b_cp[i0:i1, cp.newaxis]).sum(axis=0)
        return result

    def lipschitz(self, n_iter: int = 5, seed: int = 0) -> float:
        """
        Estimate the Lipschitz constant of A^H A via power iteration.
        Same algorithm as in holographic_imaging_gpu.py.
        Returns L_est (float).
        """
        rng  = np.random.default_rng(seed)
        v_np = rng.standard_normal(self.N_vox) + 1j * rng.standard_normal(self.N_vox)
        v    = cp.asarray(v_np)
        v   /= cp.linalg.norm(v)
        L    = 1.0
        for _ in range(n_iter):
            v  = self.AH(self.A(v))
            L  = float(cp.linalg.norm(v).get())
            v /= cp.linalg.norm(v)
        return L

    # ------------------------------------------------------------------
    # PyTorch interface  (gradient-aware, used in LISTA training)
    # ------------------------------------------------------------------

    def A_torch(self, z: torch.Tensor) -> torch.Tensor:
        """
        b = A z — torch.cfloat in/out (same CUDA device).
        Gradient of this op w.r.t. z flows back through A^H.
        """
        return _ApplyA.apply(z, self)

    def AH_torch(self, b: torch.Tensor) -> torch.Tensor:
        """
        z = A^H b — torch.cfloat in/out (same CUDA device).
        Gradient of this op w.r.t. b flows back through A.
        """
        return _ApplyAH.apply(b, self)


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

def _torch_to_cp(x: torch.Tensor) -> cp.ndarray:
    """torch.cfloat/cdouble (CUDA) -> cp.complex128, via numpy (safe for all dtypes)."""
    return cp.asarray(x.detach().cpu().numpy()).astype(cp.complex128)


def _cp_to_torch(x: cp.ndarray, device: torch.device) -> torch.Tensor:
    """cp.complex128 -> torch.cfloat on `device`."""
    arr = cp.asnumpy(x).astype(np.complex64)
    return torch.as_tensor(arr, dtype=torch.cfloat, device=device)


# ---------------------------------------------------------------------------
# torch.autograd.Function wrappers
#
# Rule for linear operators:
#   forward  A   -> backward A^H
#   forward  A^H -> backward A
# ---------------------------------------------------------------------------

class _ApplyA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: torch.Tensor, op: HolographyOperator) -> torch.Tensor:
        ctx.op     = op
        ctx.device = z.device
        z_cp = _torch_to_cp(z)
        b_cp = op.A(z_cp)
        return _cp_to_torch(b_cp, z.device)

    @staticmethod
    def backward(ctx, grad_b: torch.Tensor):
        # d/dz (Az) = A^H  applied to incoming gradient
        gb_cp  = _torch_to_cp(grad_b)
        gz_cp  = ctx.op.AH(gb_cp)
        grad_z = _cp_to_torch(gz_cp, ctx.device)
        return grad_z, None   # None: no gradient for `op`


class _ApplyAH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b: torch.Tensor, op: HolographyOperator) -> torch.Tensor:
        ctx.op     = op
        ctx.device = b.device
        b_cp = _torch_to_cp(b)
        z_cp = op.AH(b_cp)
        return _cp_to_torch(z_cp, b.device)

    @staticmethod
    def backward(ctx, grad_z: torch.Tensor):
        # d/db (A^H b) = A  applied to incoming gradient
        gz_cp  = _torch_to_cp(grad_z)
        gb_cp  = ctx.op.A(gz_cp)
        grad_b = _cp_to_torch(gb_cp, ctx.device)
        return grad_b, None
