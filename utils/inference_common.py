# -*- coding: utf-8 -*-
"""
inference_common.py
===================
Helper CONDIVISO per gli script di inferenza (sintetico e reale).

Contiene:
  - auto-detect + costruzione del modello dal checkpoint
    (LISTA / W-LISTA / LR-W-LISTA), con possibilita' di forzare il tipo;
  - costruzione dell'operatore veloce (HolographyOperatorFast, DLPack);
  - inferenza del modello + metrica di data-consistency con l'operatore
    EFFETTIVO del modello (T per LISTA/W-LISTA, T_eff=(I+UV^H)T per LR-W-LISTA);
  - baseline matched filter / ISTA;
  - metriche (Signal/Clutter, MSE) e plot MIP-xy.

Il tipo di modello viene riconosciuto ispezionando le chiavi di model_state:
    'U_re'   presente -> LR-W-LISTA   (correzione low-rank)
    'log_wx' presente -> W-LISTA      (pesi spaziali)
    altrimenti        -> LISTA
M e rank della correzione low-rank sono dedotti DIRETTAMENTE dalla forma di
U_re (M, rank), quindi funziona anche con checkpoint che non salvano quei campi.
"""

import os
import numpy as np
import scipy.io as sio
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lista_holography          import LISTAHolography
from lista_holography_weighted import WLISTAHolography
from lista_holography_lowrank  import LRWLISTAHolography

MODEL_CHOICES = ["auto", "lista", "wlista", "lrwlista"]
_PRETTY = {"lista": "LISTA", "wlista": "W-LISTA", "lrwlista": "LR-W-LISTA"}


# ---------------------------------------------------------------------------
# Auto-detect + costruzione modello
# ---------------------------------------------------------------------------
def detect_kind(state_dict) -> str:
    """Riconosce il tipo di modello dalle chiavi del state_dict."""
    if "U_re" in state_dict:
        return "lrwlista"
    if "log_wx" in state_dict:
        return "wlista"
    return "lista"


def build_model(ckpt, L_est, lambda_init, model_sel="auto"):
    """
    Costruisce il modello dal checkpoint.

    model_sel : 'auto' (rileva dal checkpoint) oppure
                'lista' / 'wlista' / 'lrwlista' (forza il tipo).
    Ritorna (model, kind) con kind in {'lista','wlista','lrwlista'}.
    """
    sd = ckpt["model_state"]
    K  = int(ckpt["K"])
    detected = detect_kind(sd)

    if model_sel not in MODEL_CHOICES:
        raise ValueError(f"--model deve essere uno di {MODEL_CHOICES}")
    kind = detected if model_sel == "auto" else model_sel

    if model_sel != "auto" and model_sel != detected:
        raise ValueError(
            f"--model '{model_sel}' richiesto, ma il checkpoint contiene "
            f"parametri da '{detected}'. I pesi non combaciano: rilancia con "
            f"--model auto oppure --model {detected}.")

    if kind == "lrwlista":
        Nx, Ny, Nz = int(ckpt["Nx"]), int(ckpt["Ny"]), int(ckpt["Nz"])
        M, rank = sd["U_re"].shape          # dedotti dalla forma (robusto)
        model = LRWLISTAHolography(K=K, L_est=L_est, Nx=Nx, Ny=Ny, Nz=Nz,
                                   M=int(M), rank=int(rank),
                                   lambda_init=lambda_init)
    elif kind == "wlista":
        Nx, Ny, Nz = int(ckpt["Nx"]), int(ckpt["Ny"]), int(ckpt["Nz"])
        model = WLISTAHolography(K=K, L_est=L_est, Nx=Nx, Ny=Ny, Nz=Nz,
                                 lambda_init=lambda_init)
    else:  # lista
        model = LISTAHolography(K=K, L_est=L_est, lambda_init=lambda_init)

    model.load_state_dict(sd)
    return model, kind


def pretty(kind: str) -> str:
    return _PRETTY.get(kind, kind)


# ---------------------------------------------------------------------------
# Operatore veloce
# ---------------------------------------------------------------------------
def build_operator(ref, k, omega, x_img, y_img, z_img, mu0, batch_rx, device="cuda"):
    """Operatore A/A^H. device 'cuda' -> HolographyOperatorFast (cupy);
    'cpu' -> HolographyOperatorNumpy (NumPy puro, nessun CUDA richiesto)."""
    nx, ny = ref["S21"].shape[:2]
    x_rx = ref["X"][:, 0]; y_rx = ref["Y"][0, :]
    xx, yy = np.meshgrid(x_rx, y_rx, indexing="ij")
    r_rx = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny)])
    xxv, yyv, zzv = np.meshgrid(x_img, y_img, z_img, indexing="ij")
    r_vox = np.column_stack([xxv.ravel(), yyv.ravel(), zzv.ravel()])
    dV = (x_img[1]-x_img[0]) * (y_img[1]-y_img[0]) * (z_img[1]-z_img[0])
    if "cuda" in str(device):
        import cupy as cp
        from holography_operator_fast import HolographyOperatorFast
        return HolographyOperatorFast(cp.asarray(r_rx), cp.asarray(r_vox),
                                      k=k, omega=omega, mu0=mu0, dV=dV, batch_rx=batch_rx)
    from holography_operator_numpy import HolographyOperatorNumpy
    op = HolographyOperatorNumpy(r_rx, r_vox, k=k, omega=omega, mu0=mu0,
                                 dV=dV, batch_rx=batch_rx)
    op.verbose = True   # mostra progress batch su CPU
    return op


# ---------------------------------------------------------------------------
# Inferenza modello (+ data-consistency con l'operatore EFFETTIVO del modello)
# ---------------------------------------------------------------------------
def _model_measure(model, z_t, op):
    """y = T_eff z (LR-W-LISTA) oppure T z (LISTA / W-LISTA)."""
    if hasattr(model, "measure"):
        return model.measure(z_t, op)
    return op.A_torch(z_t)


def run_model(model, op, b_np, device, warm_start=True):
    model.eval()
    with torch.no_grad():
        b_t = torch.as_tensor(b_np.astype(np.complex64), device=device)
        z_t = model(b_t, op, warm_start=warm_start)
        y_t = _model_measure(model, z_t, op)
        dc  = float((torch.linalg.norm(y_t - b_t) /
                     (torch.linalg.norm(b_t) + 1e-30)).item())
    return z_t.detach().cpu().numpy().astype(np.complex64), dc


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
def run_matched_filter(op, b_np):
    return op.AH_np(b_np.astype(np.complex128)).astype(np.complex64)


def run_ista(op, b_np, K_iter, alpha, L_est):
    step = 1.0 / L_est
    thr  = step * alpha
    b = b_np.astype(np.complex128)
    z = np.zeros(op.N_vox, dtype=np.complex128)
    for _ in range(int(K_iter)):
        grad  = op.AH_np(op.A_np(z) - b)
        z_upd = z - step * grad
        mag   = np.abs(z_upd)
        z     = z_upd * np.maximum(0.0, 1.0 - thr / np.maximum(mag, 1e-30))
    return z.astype(np.complex64)


# ---------------------------------------------------------------------------
# Metriche
# ---------------------------------------------------------------------------
def signal_clutter(z_pred, z_true):
    occ = np.abs(z_true) > 0
    mag = np.abs(z_pred)
    return float((mag[occ].mean() + 1e-30) / (mag[~occ].mean() + 1e-30))


def metrics(z_pred, z_true):
    mp, mt = np.abs(z_pred), np.abs(z_true)
    occ = mt > 0
    return dict(
        SC=signal_clutter(z_pred, z_true),
        mse_full=float(np.mean((mp - mt) ** 2)),
        mse_occ=float(np.mean((mp[occ] - mt[occ]) ** 2)) if occ.any() else float("nan"),
        mse_bg=float(np.mean(mp[~occ] ** 2)) if (~occ).any() else float("nan"),
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _mip_xy(z, shape):
    return np.abs(z).reshape(shape).max(axis=2)


def _to_db(a):
    return 20.0 * np.log10(a / (a.max() + 1e-30) + 1e-30)


def save_outputs(out_npz, out_mat, x_img, y_img, z_img, payload):
    """Salva i risultati sia in .npz sia in .mat (volumi 3D + immagini MIP-xy).

    Nel .mat ogni volume viene salvato sia 'flat' sia rimodellato a (Nx,Ny,Nz),
    piu' la proiezione MIP-xy (l'immagine effettivamente visualizzata).
    """
    np.savez(out_npz, **payload)

    shape = (len(x_img), len(y_img), len(z_img))
    mat = dict(X_IMG=x_img, Y_IMG=y_img, Z_IMG=z_img)
    for key, val in payload.items():
        arr = np.asarray(val)
        if np.iscomplexobj(arr) and arr.size == shape[0] * shape[1] * shape[2]:
            vol = arr.reshape(shape)
            mat[key] = arr                       # flat
            mat[key + "_vol"] = vol              # 3D (Nx,Ny,Nz)
            mat[key + "_mip_xy"] = np.abs(vol).max(axis=2)   # immagine MIP
        else:
            mat[key] = arr
    sio.savemat(out_mat, mat, do_compression=True)


def save_epoch_snapshot(out_dir, prefix, epoch, x_img, y_img, z_img,
                        z_model, z_true=None, z_mf=None, z_ista=None):
    """Snapshot di riferimento per una singola epoca.

    PNG = pannelli MIP-xy (GT, MF, ISTA, modello).
    .mat / .npz = SOLO le immagini MIP-xy (compatte) di GT/MF/ISTA/modello
    (no volumi 3D): sufficiente per il confronto epoca-per-epoca.
    """
    shape = (len(x_img), len(y_img), len(z_img))
    tag = f"recon_{prefix}_ep{epoch:03d}"
    png = os.path.join(out_dir, tag + ".png")
    # ground truth escluso volutamente: non interessa nel monitoraggio per-epoca
    plot_panels(png, x_img, y_img, z_img, f"{prefix}  epoch {epoch}",
                z_model, f"recon ep{epoch}", z_true=None, z_mf=z_mf, z_ista=z_ista)

    mips = dict(epoch=int(epoch),
                X_IMG=x_img, Y_IMG=y_img,
                mip_model=_mip_xy(z_model, shape))
    if z_mf   is not None: mips["mip_mf"]   = _mip_xy(z_mf,   shape)
    if z_ista is not None: mips["mip_ista"] = _mip_xy(z_ista, shape)

    np.savez(os.path.join(out_dir, tag + ".npz"), **mips)
    sio.savemat(os.path.join(out_dir, tag + ".mat"), mips, do_compression=True)
    return png

def plot_panels(out_png, x_img, y_img, z_img, suptitle,
                z_model, model_label, z_true=None, z_mf=None, z_ista=None):
    shape = (len(x_img), len(y_img), len(z_img))
    panels = []
    if z_true is not None: panels.append((_mip_xy(z_true, shape), "Ground truth |z*|", False))
    if z_mf   is not None: panels.append((_mip_xy(z_mf,   shape), "Matched filter",     True))
    if z_ista is not None: panels.append((_mip_xy(z_ista, shape), "ISTA",               True))
    panels.append((_mip_xy(z_model, shape), model_label, True))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 5.8))
    if n == 1: axes = [axes]
    for ax, (proj, title, use_db) in zip(axes, panels):
        data = _to_db(proj) if use_db else proj
        vmin, vmax = (-30.0, 0.0) if use_db else (0.0, float(proj.max() + 1e-30))
        im = ax.pcolormesh(x_img, y_img, data.T, cmap="jet",
                           vmin=vmin, vmax=vmax, shading="nearest")
        plt.colorbar(im, ax=ax, label="dB" if use_db else "|z|", fraction=0.046)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
