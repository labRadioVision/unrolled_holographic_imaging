# -*- coding: utf-8 -*-
"""
lista_holography_lowrank.py
===========================
LR-W-LISTA = W-LISTA + correzione low-rank supervisionata dell'operatore T.

Motivazione
-----------
Con i PEC l'approssimazione di Born non vale: il modello lineare y = T z e'
sistematicamente BIASED (lo scattering multiplo/ombreggiamento sposta soprattutto
la FASE). W-LISTA agisce solo nel prox (regolarizzazione pesata): non puo'
correggere un termine di data-fidelity polarizzato, perche' il gradiente resta
mu * T^H (T z - b) con il T sbagliato in entrambe le occorrenze.

Qui impariamo una PERTURBAZIONE A BASSO RANGO di T, lato-misura:

        T_eff = (I_M + U V^H) T ,     U, V in C^{M x r},   r << M

Razionale dimensionale: T e' M x N con M (ricevitori) << N (voxel), quindi
rank(T) <= M: tutto cio' che e' osservabile vive nel sottospazio M-dimensionale.
Una correzione lato-misura (M x M, vincolata a rango r) e' percio' la piu'
espressiva con il minimo dei parametri, e impara a rimappare le misure
predette da Born verso quelle full-wave. Costo: due prodotti M x r per matvec,
trascurabile rispetto alla fisica T (valutata on-the-fly).

T resta FISICO e NON appreso: T_eff e' una perturbazione low-rank di un
operatore fisico -> physics-consistency preservata.

Parametri appresi (oltre a quelli W-LISTA: log_mu, log_lambda, log_wx/wy/wz):
    U_re, U_im : (M, r)
    V_re, V_im : (M, r)
condivisi tra i layer (shared) per default -> pochi dati (8 scene), poca
capacita' extra. Init U = 0  ->  T_eff = T  ->  a inizio training la rete
coincide ESATTAMENTE con W-LISTA (regressione di sicurezza).

Algebra adjoint:
    T_eff^H = T^H (I + U V^H)^H = T^H (I + V U^H)
    T_eff^H r = T^H ( r + V (U^H r) )

API
---
model = LRWLISTAHolography(K, L_est, Nx, Ny, Nz, M, rank=16)
z_hat = model(b, op, warm_start=True)        # ricostruzione
y_hat = model.measure(z_hat, op)             # T_eff z  (per data-consistency loss)

`op` deve essere un HolographyOperator(Fast) con A_torch / AH_torch.
"""

import numpy as np
import torch
import torch.nn as nn

from lista_holography_weighted import WLISTAHolography, _complex_soft_thresh


class LRWLISTAHolography(WLISTAHolography):
    """W-LISTA con correzione low-rank lato-misura T_eff = (I + U V^H) T."""

    def __init__(self,
                 K: int, L_est: float,
                 Nx: int, Ny: int, Nz: int,
                 M: int,
                 rank: int = 16,
                 lambda_init: float = 1e-4,
                 corrected_warm_start: bool = True,
                 v_init_std: float = 1e-2):
        super().__init__(K=K, L_est=L_est, Nx=Nx, Ny=Ny, Nz=Nz,
                         lambda_init=lambda_init)
        self.M    = int(M)
        self.rank = int(rank)
        self.corrected_warm_start = bool(corrected_warm_start)

        # U init a 0  =>  Delta T = 0  =>  identico a W-LISTA all'inizio.
        # V init piccolo random: appena U si muove, V riceve gradiente.
        self.U_re = nn.Parameter(torch.zeros(self.M, self.rank))
        self.U_im = nn.Parameter(torch.zeros(self.M, self.rank))
        self.V_re = nn.Parameter(v_init_std * torch.randn(self.M, self.rank))
        self.V_im = nn.Parameter(v_init_std * torch.randn(self.M, self.rank))

    # ------------------------------------------------------------------
    def _UV(self):
        U = torch.complex(self.U_re, self.U_im)   # (M, r)
        V = torch.complex(self.V_re, self.V_im)   # (M, r)
        return U, V

    def measure(self, z: torch.Tensor, op) -> torch.Tensor:
        """y = T_eff z = T z + U (V^H (T z))."""
        Tz = op.A_torch(z)                         # (M,)
        U, V = self._UV()
        corr = U @ (V.conj().t() @ Tz)             # (M,)
        return Tz + corr

    def _TeffH(self, r: torch.Tensor, op) -> torch.Tensor:
        """T_eff^H r = T^H ( r + V (U^H r) )."""
        U, V = self._UV()
        rc = r + V @ (U.conj().t() @ r)            # (M,)
        return op.AH_torch(rc)                      # (N,)

    # ------------------------------------------------------------------
    def forward(self, b: torch.Tensor, op, warm_start: bool = True) -> torch.Tensor:
        device = b.device
        N_vox  = self.Nx * self.Ny * self.Nz
        assert op.N_vox == N_vox, (
            f"Operator voxel count {op.N_vox} != grid "
            f"{self.Nx}x{self.Ny}x{self.Nz}={N_vox}")
        assert op.N_rx == self.M, (
            f"Operator N_rx {op.N_rx} != model M {self.M}")

        if warm_start:
            if self.corrected_warm_start:
                z = self._TeffH(b, op)             # all'init (U=0) == A^H b
            else:
                z = op.AH_torch(b)
        else:
            z = torch.zeros(N_vox, dtype=torch.cfloat, device=device)

        for k in range(self.K):
            mu_k     = torch.exp(self.log_mu[k])
            lambda_k = torch.exp(self.log_lambda[k])

            residual = self.measure(z, op) - b      # T_eff z - b
            grad     = self._TeffH(residual, op)    # T_eff^H (.)
            z        = z - mu_k * grad

            W_flat = self.weight_field(k).to(device)
            thr    = lambda_k * W_flat
            z      = _complex_soft_thresh(z, thr)

        return z

    # ------------------------------------------------------------------
    def lowrank_stats(self):
        """Norme/diagnostica della correzione low-rank."""
        U, V = self._UV()
        with torch.no_grad():
            sv = torch.linalg.svdvals((U @ V.conj().t())).cpu().numpy() \
                 if self.M <= 4096 else None   # SVD piena solo se M piccolo
            return dict(
                U_fro=float(U.abs().pow(2).sum().sqrt()),
                V_fro=float(V.abs().pow(2).sum().sqrt()),
                UVt_top_sv=(float(sv[0]) if sv is not None else float("nan")),
                rank=self.rank,
            )

    def lowrank_frob_sq(self):
        """||U V^H||_F^2 (differenziabile) per regolarizzazione.
        ||U V^H||_F^2 = tr(V U^H U V^H) = || (U^H U)^{1/2} ... ||  ->  usiamo
        l'identita': ||U V^H||_F^2 = sum_{ij} |.|^2 = tr( (U^H U)(V^H V) ).
        """
        U, V = self._UV()
        GU = U.conj().t() @ U          # (r, r)
        GV = V.conj().t() @ V          # (r, r)
        return torch.real(torch.sum(GU * GV.t()))
