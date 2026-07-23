# -*- coding: utf-8 -*-
"""
lista_holography_weighted.py
============================
W-LISTA (Weighted LISTA) for holographic reconstruction.

Strategy (A) - factorized separable per-voxel weights.

Each layer k applies a weighted L1 prox:
    |diag(w_k) z|_1 ,  with w_k(x,y,z) = w_k^(x)(x) * w_k^(y)(y) * w_k^(z)(z)

i.e. the spatial weights are the OUTER PRODUCT of three 1D vectors, one per axis:
    w_k^(x) in R^Nx ,  w_k^(y) in R^Ny ,  w_k^(z) in R^Nz

Learnable parameters (all log-parametrized to stay positive):
    log_mu     : (K,)        ISTA step size
    log_lambda : (K,)        multiplicative base threshold
    log_wx     : (K, Nx)     weights along x
    log_wy     : (K, Ny)     weights along y
    log_wz     : (K, Nz)     weights along z

Total for K=10, Nx=161, Ny=81, Nz=65:
    K*(2 + Nx + Ny + Nz) = 10 * (2 + 307) = 3090 parameters.

Compared with the uniform LISTA: 2K = 20 parameters.

Forward layer k:
    residual = A z - b
    grad     = A^H residual
    z        = z - mu_k * grad
    thr_k    = lambda_k * W_k       (W_k is the factorized 3D field, flattened)
    z        = soft_thresh_modulus(z, thr_k)    # per-voxel threshold

Init:
    log_w* = 0  =>  w* = 1  =>  uniform threshold  =>  initial behaviour
    identical to LISTAHolography (this init makes training start well).
"""

import numpy as np
import torch
import torch.nn as nn

try:
    from holography_operator import HolographyOperator
except (ImportError, ModuleNotFoundError):
    HolographyOperator = None


# ===========================================================================
# Weighted complex soft-threshold  (tensor-threshold broadcast)
# ===========================================================================

def _complex_soft_thresh(z: torch.Tensor, threshold) -> torch.Tensor:
    """
    Shrink the modulus of z by `threshold`, preserve phase.
    `threshold` can be a scalar or a tensor broadcastable to z.shape.
    """
    mag   = z.abs()
    scale = torch.clamp(mag - threshold, min=0.0) / (mag + 1e-30)
    return z * scale


# ===========================================================================
# W-LISTA network
# ===========================================================================

class WLISTAHolography(nn.Module):
    """
    K-layer unrolled Weighted ISTA with factorized separable per-voxel weights.

    The voxel grid shape (Nx, Ny, Nz) must be passed at construction time, and
    must match the order used in HolographyOperator (np.meshgrid(indexing='ij'),
    then ravel()). If z is flattened as z[i*Ny*Nz + j*Nz + k], then the outer
    product here produces the same layout.
    """

    def __init__(self,
                 K: int,
                 L_est: float,
                 Nx: int,
                 Ny: int,
                 Nz: int,
                 lambda_init: float = 1e-3):
        """
        Parameters
        ----------
        K           : number of unrolled layers
        L_est       : Lipschitz constant of A^H A (for mu init = 1/L)
        Nx, Ny, Nz  : voxel grid dimensions (must match operator)
        lambda_init : initial base-threshold
        """
        super().__init__()
        self.K  = K
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.Nz = int(Nz)

        mu0 = 1.0 / L_est
        self.log_mu     = nn.Parameter(torch.full((K,), float(np.log(mu0))))
        self.log_lambda = nn.Parameter(torch.full((K,), float(np.log(lambda_init))))

        # init weights to 1 => log=0 => uniform W_k => equivalent to standard LISTA
        self.log_wx = nn.Parameter(torch.zeros(K, self.Nx))
        self.log_wy = nn.Parameter(torch.zeros(K, self.Ny))
        self.log_wz = nn.Parameter(torch.zeros(K, self.Nz))

    # ---------------------------------------------------------------
    # Weight field (factorized outer product) for layer k
    # ---------------------------------------------------------------
    def weight_field(self, k: int) -> torch.Tensor:
        """
        Returns the 3D weight field W_k (flat, length N_vox).

        W_k[i,j,l] = w_k^(x)[i] * w_k^(y)[j] * w_k^(z)[l]

        The flatten order matches (Nx, Ny, Nz) with .ravel() in C order,
        consistent with HolographyOperator's r_vox construction.
        """
        wx = torch.exp(self.log_wx[k])                  # (Nx,)
        wy = torch.exp(self.log_wy[k])                  # (Ny,)
        wz = torch.exp(self.log_wz[k])                  # (Nz,)
        W  = (wx[:, None, None]
              * wy[None, :, None]
              * wz[None, None, :])                      # (Nx, Ny, Nz)
        return W.reshape(-1)                            # (N_vox,)

    # ---------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------
    def forward(self,
                b: torch.Tensor,
                op: HolographyOperator,
                warm_start: bool = False) -> torch.Tensor:
        """
        b          : (N_rx,)  torch.cfloat (CPU or CUDA)
        warm_start : if True, z_0 = A^H b  (matched filter init)
        Returns    : (N_vox,) torch.cfloat
        """
        device = b.device
        N_vox  = self.Nx * self.Ny * self.Nz
        assert op.N_vox == N_vox, (
            f"Operator voxel count {op.N_vox} does not match "
            f"WLISTA grid {self.Nx}x{self.Ny}x{self.Nz}={N_vox}"
        )

        import sys, time as _time
        _t_start = _time.time()
        _cpu = (str(device) == "cpu")

        if warm_start:
            if _cpu: print("  [warm-start] A^H b ...", flush=True)
            with torch.no_grad():
                z = op.AH_torch(b).to(device)
            if _cpu: print(f"  [warm-start] done  {_time.time()-_t_start:.1f}s", flush=True)
        else:
            z = torch.zeros(N_vox, dtype=torch.cfloat, device=device)

        for k in range(self.K):
            _t_layer = _time.time()
            if _cpu:
                elapsed = _t_layer - _t_start
                est_tot = (elapsed / k * self.K) if k > 0 else float("nan")
                rem     = est_tot - elapsed if k > 0 else float("nan")
                print(f"  [layer {k+1:2d}/{self.K}]  elapsed {elapsed/60:5.1f} min"
                      + (f"  ~{rem/60:.0f} min remaining" if k > 0 else ""),
                      flush=True)

            mu_k     = torch.exp(self.log_mu[k])
            lambda_k = torch.exp(self.log_lambda[k])

            # gradient step
            residual = op.A_torch(z) - b
            grad     = op.AH_torch(residual)
            z        = z - mu_k * grad

            # weighted soft-threshold (per-voxel)
            W_flat = self.weight_field(k).to(device)          # (N_vox,) real
            thr    = lambda_k * W_flat                        # (N_vox,) real
            z      = _complex_soft_thresh(z, thr)

        if _cpu:
            print(f"  [done]  totale {(_time.time()-_t_start)/60:.1f} min", flush=True)

        return z

    # ---------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def print_params(self):
        mu_v  = torch.exp(self.log_mu).detach().cpu().numpy()
        la_v  = torch.exp(self.log_lambda).detach().cpu().numpy()
        wx    = torch.exp(self.log_wx).detach().cpu().numpy()
        wy    = torch.exp(self.log_wy).detach().cpu().numpy()
        wz    = torch.exp(self.log_wz).detach().cpu().numpy()
        print("  mu      : " + "  ".join("%.3e" % v for v in mu_v))
        print("  lambda  : " + "  ".join("%.3e" % v for v in la_v))
        print(f"  w_x     : range [{wx.min():.3e}, {wx.max():.3e}]  "
              f"mean={wx.mean():.3e}  std={wx.std():.3e}")
        print(f"  w_y     : range [{wy.min():.3e}, {wy.max():.3e}]  "
              f"mean={wy.mean():.3e}  std={wy.std():.3e}")
        print(f"  w_z     : range [{wz.min():.3e}, {wz.max():.3e}]  "
              f"mean={wz.mean():.3e}  std={wz.std():.3e}")

    def weight_stats(self):
        """Return (wx, wy, wz) numpy arrays of shape (K, N_axis) for plotting."""
        wx = torch.exp(self.log_wx).detach().cpu().numpy()
        wy = torch.exp(self.log_wy).detach().cpu().numpy()
        wz = torch.exp(self.log_wz).detach().cpu().numpy()
        return wx, wy, wz
