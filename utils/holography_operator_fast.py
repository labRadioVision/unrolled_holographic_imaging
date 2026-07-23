# -*- coding: utf-8 -*-
"""
holography_operator_fast.py
===========================
Accelerated version of HolographyOperator.

The only functional difference w.r.t. holography_operator.py is that the torch
interfaces (A_torch / AH_torch) no longer round-trip through the CPU.

In the original code every matvec did:
    torch (CUDA) -> .cpu().numpy() -> cupy -> compute -> cp.asnumpy() -> torch
i.e. TWO GPU<->host copies per application of A or A^H, repeated 2K times per
layer, per scene, per epoch -- the confirmed bottleneck (~3 h/epoch).

Here we use DLPack to share GPU memory at zero cost (zero-copy) between torch
and cupy. The Green kernel and the computation are IDENTICAL to the baseline
(complex128 internally), so the numerical results are the same: only the speed
changes.

Dual path:
  - torch input on CUDA -> DLPack zero-copy path (fast training)
  - torch input on CPU  -> numpy fallback (identical to the original), useful
                           for inference/plotting when the model is on CPU.

STREAM CAVEAT (read if the results look like "noise"):
  torch and cupy use different CUDA streams. To avoid race conditions in the
  zero-copy path we run the cupy ops on the current torch stream
  (cp.cuda.ExternalStream). If this causes problems on a particular version
  combination, uncomment the `torch.cuda.synchronize()` fallback marked below,
  or set USE_TORCH_STREAM = False.

Drop-in: same constructor signature and same methods.
"""

import cupy  as cp
import numpy as np
import torch

# If True, cupy calls run on the current torch stream (correct and fast).
# If races are suspected, set False to use explicit synchronization
# (slower but rock-solid).
USE_TORCH_STREAM = False   # explicit sync: slower but deadlock-proof


# ---------------------------------------------------------------------------
# Green's function kernel  (verbatim da holographic_imaging_gpu.py)
# Mantenuto in complex128 per parita' numerica col baseline validato.
# ---------------------------------------------------------------------------

def _green_yy(r_obs_batch: cp.ndarray, r_src: cp.ndarray,
              k: float, omega: float, mu0: float) -> cp.ndarray:
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
# DLPack helpers (zero-copy GPU<->GPU) + CPU fallback
# ---------------------------------------------------------------------------

def _torch_cuda_to_cp(x: torch.Tensor) -> cp.ndarray:
    """torch CUDA tensor -> cupy array, ZERO-COPY via DLPack."""
    return cp.from_dlpack(x.detach().contiguous())


def _cp_to_torch_cuda(x: cp.ndarray) -> torch.Tensor:
    """cupy array -> torch CUDA tensor (DLPack), poi cast a cfloat."""
    return torch.from_dlpack(x).to(torch.complex64)


def _torch_cpu_to_cp(x: torch.Tensor) -> cp.ndarray:
    """Fallback: torch CPU -> cupy via numpy (come l'originale)."""
    return cp.asarray(x.detach().cpu().numpy()).astype(cp.complex128)


def _cp_to_torch_cpu(x: cp.ndarray, device) -> torch.Tensor:
    arr = cp.asnumpy(x).astype(np.complex64)
    return torch.as_tensor(arr, dtype=torch.cfloat, device=device)


class _torch_stream_ctx:
    """Esegue il blocco cupy sullo stream CUDA corrente di torch."""
    def __enter__(self):
        if USE_TORCH_STREAM and torch.cuda.is_available():
            self._s = cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
            self._s.__enter__()
        else:
            self._s = None
        return self

    def __exit__(self, *a):
        if self._s is not None:
            self._s.__exit__(*a)
        # Sync esplicito: garantisce che cupy abbia finito prima che torch
        # legga il risultato (elimina race condition cupy->torch).
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return False


# ---------------------------------------------------------------------------
# HolographyOperatorFast
# ---------------------------------------------------------------------------

class HolographyOperatorFast:
    """
    Identico a HolographyOperator ma con interfacce torch zero-copy (DLPack).

    Costruttore e API CuPy invariati. r_rx, r_vox come cupy float64.
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
    # CuPy interface (invariata, complex128)
    # ------------------------------------------------------------------
    def A(self, z_cp: cp.ndarray) -> cp.ndarray:
        z_cp   = z_cp.astype(cp.complex128)
        result = cp.zeros(self.N_rx, dtype=cp.complex128)
        for i0 in range(0, self.N_rx, self.batch_rx):
            i1 = min(i0 + self.batch_rx, self.N_rx)
            G  = _green_yy(self.r_rx[i0:i1], self.r_vox,
                           self.k, self.omega, self.mu0)
            result[i0:i1] = (G * self.dV) @ z_cp
        return result

    def AH(self, b_cp: cp.ndarray) -> cp.ndarray:
        b_cp   = b_cp.astype(cp.complex128)
        result = cp.zeros(self.N_vox, dtype=cp.complex128)
        for i0 in range(0, self.N_rx, self.batch_rx):
            i1 = min(i0 + self.batch_rx, self.N_rx)
            G  = _green_yy(self.r_rx[i0:i1], self.r_vox,
                           self.k, self.omega, self.mu0)
            result += (cp.conj(G * self.dV) * b_cp[i0:i1, cp.newaxis]).sum(axis=0)
        return result

    def A_np(self, z_np):
        """A with numpy in/out (for baselines unified with the numpy backend)."""
        return cp.asnumpy(self.A(cp.asarray(z_np)))

    def AH_np(self, b_np):
        return cp.asnumpy(self.AH(cp.asarray(b_np)))

    def lipschitz(self, n_iter: int = 5, seed: int = 0) -> float:
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
    # PyTorch interface (gradient-aware, zero-copy on CUDA)
    # ------------------------------------------------------------------
    def A_torch(self, z: torch.Tensor) -> torch.Tensor:
        return _ApplyA.apply(z, self)

    def AH_torch(self, b: torch.Tensor) -> torch.Tensor:
        return _ApplyAH.apply(b, self)


# ---------------------------------------------------------------------------
# autograd Functions
#   forward A   -> backward A^H
#   forward A^H -> backward A
# ---------------------------------------------------------------------------

class _ApplyA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, op):
        ctx.op = op
        ctx.device = z.device
        if z.is_cuda:
            with _torch_stream_ctx():
                z_cp = _torch_cuda_to_cp(z)
                b_cp = op.A(z_cp)
                out  = _cp_to_torch_cuda(b_cp)
            return out
        else:
            b_cp = op.A(_torch_cpu_to_cp(z))
            return _cp_to_torch_cpu(b_cp, z.device)

    @staticmethod
    def backward(ctx, grad_b):
        op = ctx.op
        if grad_b.is_cuda:
            with _torch_stream_ctx():
                gb_cp  = _torch_cuda_to_cp(grad_b)
                gz_cp  = op.AH(gb_cp)
                grad_z = _cp_to_torch_cuda(gz_cp)
            return grad_z, None
        else:
            gz_cp  = op.AH(_torch_cpu_to_cp(grad_b))
            return _cp_to_torch_cpu(gz_cp, ctx.device), None


class _ApplyAH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b, op):
        ctx.op = op
        ctx.device = b.device
        if b.is_cuda:
            with _torch_stream_ctx():
                b_cp = _torch_cuda_to_cp(b)
                z_cp = op.AH(b_cp)
                out  = _cp_to_torch_cuda(z_cp)
            return out
        else:
            z_cp = op.AH(_torch_cpu_to_cp(b))
            return _cp_to_torch_cpu(z_cp, b.device)

    @staticmethod
    def backward(ctx, grad_z):
        op = ctx.op
        if grad_z.is_cuda:
            with _torch_stream_ctx():
                gz_cp  = _torch_cuda_to_cp(grad_z)
                gb_cp  = op.A(gz_cp)
                grad_b = _cp_to_torch_cuda(gb_cp)
            return grad_b, None
        else:
            gb_cp  = op.A(_torch_cpu_to_cp(grad_z))
            return _cp_to_torch_cpu(gb_cp, ctx.device), None
