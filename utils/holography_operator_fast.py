# -*- coding: utf-8 -*-
"""
holography_operator_fast.py
===========================
Versione accelerata di HolographyOperator.

Unica differenza funzionale rispetto a holography_operator.py:
    le interfacce torch (A_torch / AH_torch) NON passano piu' dalla CPU.

Nel codice originale ogni matvec faceva:
    torch (CUDA) -> .cpu().numpy() -> cupy -> calcolo -> cp.asnumpy() -> torch
cioe' DUE copie GPU<->host per ogni applicazione di A o A^H, ripetute
K*2 volte per layer, per ogni scena, per ogni epoca. E' il collo di bottiglia
confermato (~3 h/epoca).

Qui usiamo DLPack per condividere la memoria GPU a costo zero (zero-copy)
tra torch e cupy. Il kernel di Green e il calcolo restano IDENTICI al
baseline (complex128 internamente), quindi i risultati numerici sono gli
stessi: cambia solo la velocita'.

Dual-path:
  - input torch su CUDA  -> percorso DLPack zero-copy (training veloce)
  - input torch su CPU   -> fallback numpy (identico all'originale), utile
                            per inferenza/plot quando il modello e' su CPU.

CAVEAT STREAM (leggere se i risultati sembrano "rumore"):
  torch e cupy usano stream CUDA diversi. Per evitare race condition nello
  zero-copy facciamo girare le operazioni cupy sullo stream corrente di torch
  (cp.cuda.ExternalStream). Se su una particolare combinazione di versioni
  questo desse problemi, decommentare il fallback `torch.cuda.synchronize()`
  marcato sotto, oppure usare USE_TORCH_STREAM = False.

Drop-in: stessa firma del costruttore e degli stessi metodi.
"""

import cupy  as cp
import numpy as np
import torch

# Se True, le chiamate cupy girano sullo stream corrente di torch (corretto e
# veloce). Se si sospettano race, mettere False per usare la sincronizzazione
# esplicita (piu' lento ma a prova di bomba).
USE_TORCH_STREAM = False   # sync esplicito: piu' lento ma a prova di deadlock (vedi hang sweep)


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
        """A su numpy-in/out (per baseline unificati con il backend numpy)."""
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
    # PyTorch interface (gradient-aware, zero-copy su CUDA)
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
