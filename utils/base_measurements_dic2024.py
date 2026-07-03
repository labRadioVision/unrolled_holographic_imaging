# -*- coding: utf-8 -*-
"""
run_wlista_measurements_dic2024.py
================================
W-LISTA training su N=8 misure (Nov-Dic 2024) con modello corpo realistico.

Dataset di training:
  - Spostamento laterale (x):
      21_11_2024  x_c~2.50m  (posizione di riferimento)
      10_12_2024  x_c~2.11m
      11_12_2024  x_c~1.70m
      12_12_2024  x_c~1.30m
      13_12_2024  x_c~0.89m
  - Spostamento in profondita' (z):
      05_12_2024  z_c~1.43m
      07_12_2024  z_c~1.15m
      09_12_2024  z_c~0.85m

z_true: modello corpo parametrico (generate_z_true.py --mode body)
        Delta_eps = 1.5297  (tessuto soft/muscolo a 2.45 GHz)
        ~6200-6350 voxel attivi per misura (~0.74% della griglia)

Rispetto a v1/v2 (parallelepipedi, Delta_eps=8e-4):
  - Contrasto 1900x piu' alto -> gradiente su w molto piu' forte
  - 8 misure (vs 5) con diversita' laterale E in profondita'
  - Stessi iperparametri v2: LR_W=5e-1 (10x), LAMBDA_INIT=1e-4

Checkpoint: wlista_multimeas_dec2024_*  (non sovrascrive v1/v2)

Run
---
  .conda\\python.exe run_wlista_measurements_dic2024.py > wlista_dec2024.log 2>&1

Resume
------
  .conda\\python.exe run_wlista_measurements_dic2024.py --resume checkpoints_lista\\wlista_multimeas_dec2024_ep010.pt

Infer-only
----------
  .conda\\python.exe run_wlista_measurements_dic2024.py \\
        --infer-only checkpoints_lista\\wlista_multimeas_dec2024_best.pt \\
        --infer-meas empty_30_11_2024.mat
"""

import os, sys, time
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Importazioni GPU — opzionali: falliscono silenziosamente su macchine
# senza CUDA. Le funzioni di training che le usano (build_operator,
# train_wlista) daranno errore a runtime se chiamate senza GPU, ma le
# costanti e load_b_scatter restano disponibili per l'inferenza CPU.
try:
    import cupy as cp
    from holography_operator       import HolographyOperator
    from lista_holography_weighted import WLISTAHolography
    _GPU_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    cp = None
    HolographyOperator = None
    WLISTAHolography = None
    _GPU_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ===========================================================================
# Configuration
# ===========================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "real_data"))
OUT_DIR    = os.path.join(SCRIPT_DIR, "results_wlista_multimeas_dec2024")
CKPT_DIR   = os.path.join(SCRIPT_DIR, "checkpoints_lista")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

EMPTY_FILE = os.path.join(DATA_DIR, "empty_20_11_2024.mat")

# 8 misure: 5 spostamenti laterali (x) + 3 spostamenti in profondita' (z)
MEAS_FILES = [
    os.path.join(DATA_DIR, "empty_21_11_2024.mat"),   # x_c~2.50m  z_c~1.71m (ref laterale)
    os.path.join(DATA_DIR, "empty_10_12_2024.mat"),   # x_c~2.11m  z_c~1.71m
    os.path.join(DATA_DIR, "empty_11_12_2024.mat"),   # x_c~1.70m  z_c~1.71m
    os.path.join(DATA_DIR, "empty_12_12_2024.mat"),   # x_c~1.30m  z_c~1.71m
    os.path.join(DATA_DIR, "empty_13_12_2024.mat"),   # x_c~0.89m  z_c~1.71m
    os.path.join(DATA_DIR, "empty_05_12_2024.mat"),   # x_c~2.50m  z_c~1.43m
    os.path.join(DATA_DIR, "empty_07_12_2024.mat"),   # x_c~2.50m  z_c~1.15m
    os.path.join(DATA_DIR, "empty_09_12_2024.mat"),   # x_c~2.50m  z_c~0.85m
]

ZTRUE_FILES = [
    os.path.join(SCRIPT_DIR, "results_z_true", "z_true_21_11_2024.npz"),
    os.path.join(SCRIPT_DIR, "results_z_true", "z_true_10_12_2024.npz"),
    os.path.join(SCRIPT_DIR, "results_z_true", "z_true_11_12_2024.npz"),
    os.path.join(SCRIPT_DIR, "results_z_true", "z_true_12_12_2024.npz"),
    os.path.join(SCRIPT_DIR, "results_z_true", "z_true_13_12_2024.npz"),
    os.path.join(SCRIPT_DIR, "results_z_true", "z_true_05_12_2024.npz"),
    os.path.join(SCRIPT_DIR, "results_z_true", "z_true_07_12_2024.npz"),
    os.path.join(SCRIPT_DIR, "results_z_true", "z_true_09_12_2024.npz"),
]

DATES = ["21_11_2024", "10_12_2024", "11_12_2024", "12_12_2024", "13_12_2024",
         "05_12_2024", "07_12_2024", "09_12_2024"]
N_MEAS = len(MEAS_FILES)

# Imaging grid  (must match holographic_imaging_gpu.py)
X_IMG = np.linspace(0.0, 5.0, 161)
Y_IMG = np.linspace(0.0, 2.5,  81)
Z_IMG = np.linspace(0.3, 2.3,  65)
NX, NY, NZ = len(X_IMG), len(Y_IMG), len(Z_IMG)

FREQ_IDX = 16
C        = 3.0e8
MU0      = 4.0e-7 * np.pi
BATCH_RX = 100

# W-LISTA hyperparameters
K           = 10
N_EPOCHS    = 30
LR          = 5e-2    # LR per mu_k, lambda_k
LR_W        = 5e-1    # LR per w_x, w_y, w_z (10x piu' alto — fix gradiente debole)
LAMBDA_INIT = 1e-4    # soglia iniziale (< MF output scale ~8e-4)
L_EST       = 1.141e4
W_LOG_CLAMP = 5.0     # clamp log_w in [-W_LOG_CLAMP, +W_LOG_CLAMP]
              #   => w in [exp(-5), exp(+5)] = [0.0067, 148.4]
              #   Previene divergenza esponenziale dei pesi spaziali.

# Scaling z_true: il modello corpo usa Delta_eps=1.5297 (tessuto soft),
# ma il MF output e' fisicamente limitato a ~8e-4. Il mismatch di scala
# (~1900x) rende il loss non convergente al valore esatto ma non altera
# il segno dei gradienti (target -> peso scende, background -> peso sale).
# Impostare ZTRUE_SCALE=8e-4 per normalizzare z_true alla scala MF,
# oppure None per usare il contrasto fisico originale.
ZTRUE_SCALE = None    # es: 8e-4  oppure  None (no scaling)

CKPT_NAME   = "wlista_multimeas_dec2024"


# ===========================================================================
# Data loading
# ===========================================================================

def load_b_scatter(meas_file, ref, date):
    freqs  = ref["freqs"].flatten()
    f0     = freqs[FREQ_IDX]
    k      = 2.0 * np.pi * f0 / C
    omega  = 2.0 * np.pi * f0

    meas    = sio.loadmat(meas_file)
    s_meas  = meas["S21"][:, :, FREQ_IDX].astype(np.complex128)
    s_empty = ref["S21"][:, :, FREQ_IDX].astype(np.complex128)
    c_corr  = np.sum(s_empty * np.conj(s_meas)) / np.sum(np.abs(s_meas) ** 2)
    b_np    = (c_corr * s_meas - s_empty).ravel().astype(np.complex64)

    print(f"  {date}: |c|={20*np.log10(np.abs(c_corr)):.3f} dB  "
          f"|b|_max={np.abs(b_np).max():.3e}")
    return b_np, k, omega


def load_dataset():
    print("Loading measurements ...")
    ref = sio.loadmat(EMPTY_FILE)
    b_list, z_list = [], []
    k_val = omega_val = None

    for meas_file, ztrue_file, date in zip(MEAS_FILES, ZTRUE_FILES, DATES):
        b_np, k, omega = load_b_scatter(meas_file, ref, date)
        z_data = np.load(ztrue_file)
        z_true = z_data["z_true"].astype(np.complex64)
        if ZTRUE_SCALE is not None:
            peak = np.abs(z_true).max()
            if peak > 0:
                z_true = z_true * (ZTRUE_SCALE / peak)
        b_list.append(b_np)
        z_list.append(z_true)
        k_val, omega_val = k, omega

    print(f"  Loaded {N_MEAS} measurement/z_true pairs")
    return b_list, z_list, k_val, omega_val, ref


# ===========================================================================
# Operator
# ===========================================================================

def build_operator(ref, k, omega):
    print("\nBuilding operator ...")
    Nx_rx, Ny_rx = ref["S21"].shape[:2]
    x_rx   = ref["X"][:, 0]
    y_rx   = ref["Y"][0, :]
    XX, YY = np.meshgrid(x_rx, y_rx, indexing="ij")
    r_rx   = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(Nx_rx * Ny_rx)])

    XXv, YYv, ZZv = np.meshgrid(X_IMG, Y_IMG, Z_IMG, indexing="ij")
    r_vox = np.column_stack([XXv.ravel(), YYv.ravel(), ZZv.ravel()])
    dV    = (X_IMG[1]-X_IMG[0]) * (Y_IMG[1]-Y_IMG[0]) * (Z_IMG[1]-Z_IMG[0])

    print(f"  Receivers : {r_rx.shape[0]}  ({Nx_rx}x{Ny_rx})")
    print(f"  Voxels    : {r_vox.shape[0]}  ({NX}x{NY}x{NZ})")

    op = HolographyOperator(
        r_rx=cp.asarray(r_rx), r_vox=cp.asarray(r_vox),
        k=k, omega=omega, mu0=MU0, dV=dV, batch_rx=BATCH_RX,
    )
    return op


# ===========================================================================
# Training
# ===========================================================================

def train_wlista(op, b_list, z_list, resume_ckpt=None):
    model = WLISTAHolography(K=K, L_est=L_EST,
                             Nx=NX, Ny=NY, Nz=NZ,
                             lambda_init=LAMBDA_INIT)
    # param groups: LR_W (10x) per i pesi spaziali, LR per mu/lambda
    optim = torch.optim.Adam([
        {"params": [model.log_mu, model.log_lambda], "lr": LR},
        {"params": [model.log_wx, model.log_wy, model.log_wz], "lr": LR_W},
    ])

    start_epoch  = 1
    loss_history = []
    best_loss    = float("inf")

    # --- resume ---
    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optim_state" in ckpt:
            optim.load_state_dict(ckpt["optim_state"])
            print(f"  Optimizer state restored from checkpoint")
        else:
            print(f"  WARNING: no optimizer state in checkpoint — Adam restarted")
        start_epoch  = ckpt["epoch"] + 1
        loss_history = list(ckpt.get("loss_history", []))
        best_loss    = ckpt.get("loss", float("inf"))
        print(f"  Resuming from epoch {start_epoch}  (best loss so far: {best_loss:.4e})\n")

    remaining = N_EPOCHS - start_epoch + 1
    print(f"\n[W-LISTA v2] Training  K={K}  epochs={start_epoch}->{N_EPOCHS}  "
          f"lr_mu_lam={LR:.1e}  lr_w={LR_W:.1e}  N_meas={N_MEAS}")
    print(f"  lambda_init={LAMBDA_INIT:.2e}  L_est={L_EST:.3e}")
    print(f"  # params   = {model.num_params()}  "
          f"(= K*(2 + Nx+Ny+Nz) = {K}*(2+{NX+NY+NZ}))")
    print(f"  Estimated remaining: ~{remaining * N_MEAS * K * 2 * 33 / 60:.0f} min\n")

    # pre-convert to tensors
    b_tensors = [torch.as_tensor(b) for b in b_list]
    z_tensors = [torch.as_tensor(z) for z in z_list]

    for epoch in range(start_epoch, N_EPOCHS + 1):
        t0 = time.time()
        model.train()
        optim.zero_grad()

        indices    = np.random.permutation(N_MEAS)
        epoch_loss = torch.tensor(0.0)

        for idx in indices:
            z_pred  = model(b_tensors[idx], op, warm_start=True)
            loss_i  = torch.mean((z_pred.abs() - z_tensors[idx].abs()) ** 2)
            epoch_loss = epoch_loss + loss_i

        avg_loss = epoch_loss / N_MEAS
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        # clamp log_w per evitare divergenza esponenziale dei pesi
        with torch.no_grad():
            model.log_wx.clamp_(-W_LOG_CLAMP, W_LOG_CLAMP)
            model.log_wy.clamp_(-W_LOG_CLAMP, W_LOG_CLAMP)
            model.log_wz.clamp_(-W_LOG_CLAMP, W_LOG_CLAMP)

        elapsed  = time.time() - t0
        loss_val = avg_loss.item()
        loss_history.append(loss_val)

        mu_v  = torch.exp(model.log_mu).detach().numpy()
        lam_v = torch.exp(model.log_lambda).detach().numpy()
        wx, wy, wz = model.weight_stats()
        lw_all = torch.cat([model.log_wx.flatten(),
                            model.log_wy.flatten(),
                            model.log_wz.flatten()])
        lw_min  = lw_all.min().item()
        lw_max  = lw_all.max().item()
        lw_mean = lw_all.mean().item()
        lw_std  = lw_all.std().item()
        # frazione di pesi clamped (ai bordi +-W_LOG_CLAMP)
        n_clamped = ((lw_all <= -W_LOG_CLAMP + 1e-4) | (lw_all >= W_LOG_CLAMP - 1e-4)).sum().item()
        pct_clamped = 100.0 * n_clamped / lw_all.numel()

        print(f"  Epoch {epoch:3d}/{N_EPOCHS}  loss={loss_val:.4e}  "
              f"time={elapsed:.0f}s  "
              f"mu=[{mu_v.min():.2e}..{mu_v.max():.2e}]  "
              f"lam=[{lam_v.min():.2e}..{lam_v.max():.2e}]  "
              f"w=[{min(wx.min(),wy.min(),wz.min()):.2e}.."
              f"{max(wx.max(),wy.max(),wz.max()):.2e}]  "
              f"logw=[{lw_min:.2f}..{lw_max:.2f}]  "
              f"logw_mu={lw_mean:.2f}  logw_std={lw_std:.2f}  "
              f"clamped={pct_clamped:.1f}%"
              + ("  [CLAMPED]" if (lw_min <= -W_LOG_CLAMP + 0.01 or lw_max >= W_LOG_CLAMP - 0.01) else ""))

        # best checkpoint
        if loss_val < best_loss:
            best_loss = loss_val
            torch.save({
                "epoch": epoch, "K": K,
                "Nx": NX, "Ny": NY, "Nz": NZ,
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "loss": best_loss, "loss_history": loss_history,
                "train_dates": DATES,
            }, os.path.join(CKPT_DIR, f"{CKPT_NAME}_best.pt"))

        # checkpoint ad ogni epoca
        torch.save({
            "epoch": epoch, "K": K,
            "Nx": NX, "Ny": NY, "Nz": NZ,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "loss": loss_val, "loss_history": loss_history,
            "train_dates": DATES,
        }, os.path.join(CKPT_DIR, f"{CKPT_NAME}_ep{epoch:03d}.pt"))
        print(f"  [ckpt] Saved epoch {epoch} checkpoint")

    # final checkpoint
    ckpt_path = os.path.join(CKPT_DIR, f"{CKPT_NAME}.pt")
    torch.save({
        "epoch": N_EPOCHS, "K": K,
        "Nx": NX, "Ny": NY, "Nz": NZ,
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict(),
        "loss": loss_history[-1], "loss_history": loss_history,
        "train_dates": DATES,
    }, ckpt_path)
    print(f"\n  Checkpoint saved : {ckpt_path}")
    print(f"  Best checkpoint  : {CKPT_DIR}/{CKPT_NAME}_best.pt  (loss={best_loss:.4e})")

    return model, loss_history


# ===========================================================================
# Inference measurement resolver
# ===========================================================================

def _resolve_infer_meas(meas_paths, ztrue_paths, ref):
    """Load b_scatter da una lista di .mat (basename o path assoluto).
    Returns (b_list, z_list, dates) — z_list entries = None se nessun ztrue."""
    b_list, z_list, dates = [], [], []
    for i, mp in enumerate(meas_paths):
        if not os.path.isabs(mp):
            candidate = os.path.join(DATA_DIR, mp)
            mp = candidate if os.path.exists(candidate) else mp
        date = os.path.splitext(os.path.basename(mp))[0].replace("empty_", "")
        b_np, _, _ = load_b_scatter(mp, ref, date)
        b_list.append(b_np)
        dates.append(date)
        if ztrue_paths and i < len(ztrue_paths):
            zp = ztrue_paths[i]
            if not os.path.isabs(zp):
                zp = os.path.join(SCRIPT_DIR, "results_z_true", zp) \
                     if os.path.exists(os.path.join(SCRIPT_DIR, "results_z_true", zp)) \
                     else zp
            z_list.append(np.load(zp)["z_true"].astype(np.complex64))
        else:
            z_list.append(None)
    return b_list, z_list, dates


# ===========================================================================
# Baselines
# ===========================================================================

def run_ista_baseline(op, b_np, K_iter, alpha, L_est):
    step = 1.0 / L_est
    thr  = step * alpha
    b_cp = cp.asarray(b_np.astype(np.complex128))
    z_cp = cp.zeros(op.N_vox, dtype=cp.complex128)
    for _ in range(K_iter):
        residual = op.A(z_cp) - b_cp
        grad     = op.AH(residual)
        z_upd    = z_cp - step * grad
        mag      = cp.abs(z_upd)
        z_cp     = z_upd * cp.maximum(0.0, 1.0 - thr / cp.maximum(mag, 1e-30))
    return cp.asnumpy(z_cp).astype(np.complex64)


# ===========================================================================
# Inference & metrics
# ===========================================================================

def run_wlista_inference(model, op, b_np):
    """Returns (z_mf, z_wlista).
    z_mf     = A^H b  (matched filter = warm start z_0, free).
    z_wlista = output W-LISTA completo (warm-start usa lo stesso A^H b).
    """
    model.eval()
    with torch.no_grad():
        b_cp     = cp.asarray(b_np.astype(np.complex128))
        z_mf_np  = cp.asnumpy(op.AH(b_cp)).astype(np.complex64)
        z_wlista = model(torch.as_tensor(b_np), op, warm_start=True)
    return z_mf_np, z_wlista.numpy().astype(np.complex64)


def mip_xy(z_np):
    return np.abs(z_np).reshape(NX, NY, NZ).max(axis=2)


def to_db(arr):
    return 20.0 * np.log10(arr / (arr.max() + 1e-30) + 1e-30)


def signal_clutter(z_pred, z_true):
    occ = np.abs(z_true) > 0
    mag = np.abs(z_pred)
    return (mag[occ].mean() + 1e-30) / (mag[~occ].mean() + 1e-30)


# ===========================================================================
# Plots
# ===========================================================================

def plot_results(b_list, z_list, model, op, loss_history, dates_override=None, ckpt_tag=None):
    plot_dates = dates_override if dates_override is not None else DATES
    N_plot = len(b_list)
    print(f"\nGenerating plots ({N_plot} measurements) ...")

    # inference per tutti i metodi su tutte le misure
    z_mf_list, z_ista_list, z_wlista_list = [], [], []
    for b_np, date in zip(b_list, plot_dates):
        print(f"  MF+ISTA+W-LISTA  {date} ...")
        z_mf, z_wl = run_wlista_inference(model, op, b_np)
        z_mf_list.append(z_mf)
        z_wlista_list.append(z_wl)
        z_ista_list.append(run_ista_baseline(op, b_np, K_iter=K,
                                              alpha=LAMBDA_INIT, L_est=L_EST))

    has_ztrue  = any(z is not None for z in z_list)
    n_rows     = 4 if has_ztrue else 3
    row_labels = (["Ground truth |z*|", "Matched filter", f"ISTA K={K}", f"W-LISTA K={K}"]
                  if has_ztrue else
                  ["Matched filter", f"ISTA K={K}", f"W-LISTA K={K}"])

    fig, axes = plt.subplots(n_rows, N_plot,
                             figsize=(4.0 * N_plot, 3.5 * n_rows + 3))
    if N_plot == 1:
        axes = axes[:, np.newaxis]

    sc_mf, sc_ista, sc_wl = [], [], []
    for col, (date, z_true, z_mf, z_ista, z_wl) in enumerate(
            zip(plot_dates, z_list, z_mf_list, z_ista_list, z_wlista_list)):

        z_rows = ([z_true, z_mf, z_ista, z_wl] if has_ztrue
                  else [z_mf, z_ista, z_wl])

        if z_true is not None:
            sc_mf.append(signal_clutter(z_mf, z_true))
            sc_ista.append(signal_clutter(z_ista, z_true))
            sc_wl.append(signal_clutter(z_wl, z_true))

        for row, (z, label) in enumerate(zip(z_rows, row_labels)):
            ax = axes[row, col]
            if z is None:
                ax.axis("off"); continue
            proj = mip_xy(z)
            use_db = (label != "Ground truth |z*|")
            data   = to_db(proj) if use_db else proj
            vmin, vmax = (-30, 0) if use_db else (0, proj.max())
            ax.pcolormesh(X_IMG, Y_IMG, data.T,
                          cmap="jet", vmin=vmin, vmax=vmax, shading="nearest")
            if row == 0:
                ax.set_title(date, fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{label}\ny (m)", fontsize=7)
            ax.set_xlabel("x (m)", fontsize=7)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)

    # --- stats: loss / mu / lambda / S-C ---
    fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4))
    layers = np.arange(1, K + 1)
    mu_v   = torch.exp(model.log_mu).detach().numpy()
    lam_v  = torch.exp(model.log_lambda).detach().numpy()

    axes2[0].semilogy(range(1, len(loss_history)+1), loss_history, 'b-o', ms=4)
    axes2[0].set_xlabel("Epoch"); axes2[0].set_ylabel("Avg loss")
    axes2[0].set_title("Training loss"); axes2[0].grid(True, which='both', alpha=0.3)

    axes2[1].bar(layers, mu_v, color='steelblue')
    axes2[1].set_xlabel("Layer k"); axes2[1].set_ylabel(r"$\mu_k$")
    axes2[1].set_title("Learned step sizes")

    axes2[2].bar(layers, lam_v, color='coral')
    axes2[2].set_xlabel("Layer k"); axes2[2].set_ylabel(r"$\lambda_k$")
    axes2[2].set_title("Learned base thresholds")

    if sc_wl:
        x = np.arange(len(sc_wl)); w = 0.25
        axes2[3].bar(x - w, sc_mf,    w, label='MF',      color='steelblue')
        axes2[3].bar(x,     sc_ista,  w, label='ISTA',    color='orange')
        axes2[3].bar(x + w, sc_wl,    w, label='W-LISTA', color='seagreen')
        axes2[3].set_xticks(x)
        axes2[3].set_xticklabels(plot_dates[:len(sc_wl)], rotation=30, fontsize=7)
        axes2[3].set_ylabel("S/C ratio")
        axes2[3].set_title("Signal/clutter per misura")
        axes2[3].legend(fontsize=8)
    else:
        axes2[3].text(0.5, 0.5, "No z_true\n(metrics N/A)",
                      ha='center', va='center', transform=axes2[3].transAxes)
        axes2[3].set_title("Signal/clutter per misura")

    # --- pesi learned (heatmap layer-axis × spatial-axis) ---
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 4))
    wx, wy, wz = model.weight_stats()
    for ax, w, coord, label in zip(
            axes3,
            [wx, wy, wz],
            [X_IMG, Y_IMG, Z_IMG],
            [r"$w_k^{(x)}(x)$", r"$w_k^{(y)}(y)$", r"$w_k^{(z)}(z)$"]):
        im = ax.pcolormesh(coord, layers, w, cmap="viridis", shading="nearest")
        plt.colorbar(im, ax=ax, label="weight")
        ax.set_xlabel("position (m)")
        ax.set_ylabel("layer k")
        ax.set_title(label)

    fig.suptitle(f"W-LISTA dec2024 — Nov/Dic 2024  N={N_MEAS}  K={K}  epochs={N_EPOCHS}", fontsize=10)
    fig2.suptitle("Training stats & Signal/Clutter", fontsize=10)
    fig3.suptitle("Learned factorized weights (exp(log_w))", fontsize=10)
    fig.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    date_tag = ("_".join(plot_dates) if len(plot_dates) <= 2
                else f"{plot_dates[0]}_{plot_dates[-1]}")
    prefix   = f"{CKPT_NAME}_{ckpt_tag}" if ckpt_tag else CKPT_NAME
    out_png   = os.path.join(OUT_DIR, f"{prefix}_infer_{date_tag}.png")
    out_png2  = os.path.join(OUT_DIR, f"{prefix}_infer_{date_tag}_stats.png")
    out_png3  = os.path.join(OUT_DIR, f"{prefix}_infer_{date_tag}_weights.png")
    fig.savefig(out_png,   dpi=130, bbox_inches="tight")
    fig2.savefig(out_png2, dpi=130, bbox_inches="tight")
    fig3.savefig(out_png3, dpi=130, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved {out_png}")
    print(f"  Saved {out_png2}")
    print(f"  Saved {out_png3}")
    return out_png, z_mf_list, z_wlista_list


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint .pt to resume training from")
    parser.add_argument("--infer-only", default=None,
                        help="Skip training, run only inference+plot from this checkpoint")
    parser.add_argument("--infer-meas", nargs="+", default=None,
                        help="Measurement .mat files for inference (basename or full path). "
                             "Default: the 5 training measurements.")
    parser.add_argument("--infer-ztrue", nargs="+", default=None,
                        help="z_true .npz files for inference metrics (optional, "
                             "same order as --infer-meas).")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 65)
    print("W-LISTA dec2024 multi-measurement training  (Nov-Dic 2024, N=8)")
    print("  Strategy A: factorized separable per-voxel weights")
    print("  Body model: Delta_eps=1.53 | LR_W=5e-1 | LAMBDA_INIT=1e-4")
    print("=" * 65)

    # infer-only con misure esterne: bastala reference per la geometria
    if args.infer_only and args.infer_meas:
        print("Loading reference file only (infer-only mode) ...")
        ref   = sio.loadmat(EMPTY_FILE)
        freqs = ref["freqs"].flatten()
        f0    = freqs[FREQ_IDX]
        k     = 2.0 * np.pi * f0 / C
        omega = 2.0 * np.pi * f0
        b_list, z_list = [], []
    else:
        b_list, z_list, k, omega, ref = load_dataset()

    op = build_operator(ref, k, omega)

    if args.infer_only:
        # estrae tag dall'epoch dal nome del file (es. "ep019" o "best")
        import re as _re
        _m = _re.search(r'(ep\d+|best)', os.path.basename(args.infer_only))
        ckpt_tag = _m.group(1) if _m else os.path.splitext(os.path.basename(args.infer_only))[0]
        print(f"\nInference-only mode — loading {args.infer_only}  [tag={ckpt_tag}]")
        ckpt = torch.load(args.infer_only, map_location="cpu", weights_only=False)
        # grid dimensions: prefer from ckpt, fallback to current config
        ckpt_Nx = int(ckpt.get("Nx", NX))
        ckpt_Ny = int(ckpt.get("Ny", NY))
        ckpt_Nz = int(ckpt.get("Nz", NZ))
        model = WLISTAHolography(K=int(ckpt["K"]), L_est=L_EST,
                                 Nx=ckpt_Nx, Ny=ckpt_Ny, Nz=ckpt_Nz,
                                 lambda_init=LAMBDA_INIT)
        model.load_state_dict(ckpt["model_state"])
        loss_history = list(ckpt.get("loss_history", [ckpt["loss"]]))
        print(f"  Loaded epoch={ckpt['epoch']}  loss={ckpt['loss']:.4e}  "
              f"grid=({ckpt_Nx},{ckpt_Ny},{ckpt_Nz})")

        if args.infer_meas:
            infer_b_list, infer_z_list, infer_dates = _resolve_infer_meas(
                args.infer_meas, args.infer_ztrue, ref)
        else:
            infer_b_list, infer_z_list, infer_dates = b_list, z_list, DATES

        prefix = f"{CKPT_NAME}_{ckpt_tag}" if ckpt_tag else CKPT_NAME
        out_png, z_mf_list, z_wlista_list = plot_results(
                               infer_b_list, infer_z_list, model, op,
                               loss_history, dates_override=infer_dates,
                               ckpt_tag=ckpt_tag)

        # salva z_wlista e z_mf in .mat (uno per misura)
        print("\nSaving .mat files ...")
        for date, z_mf, z_wl in zip(infer_dates, z_mf_list, z_wlista_list):
            mat_path = os.path.join(OUT_DIR, f"{prefix}_{date}_z.mat")
            sio.savemat(mat_path, {
                "z_wlista": z_wl.reshape(ckpt_Nx, ckpt_Ny, ckpt_Nz),
                "z_mf":     z_mf.reshape(ckpt_Nx, ckpt_Ny, ckpt_Nz),
                "x_img":    X_IMG,
                "y_img":    Y_IMG,
                "z_img":    Z_IMG,
            })
            print(f"  Saved {mat_path}")
    else:
        ckpt_tag = None
        model, loss_history = train_wlista(op, b_list, z_list,
                                           resume_ckpt=args.resume)
        out_png, _, _ = plot_results(b_list, z_list, model, op, loss_history)

    # save results npz
    wx_np, wy_np, wz_np = model.weight_stats()
    npz_prefix = f"{CKPT_NAME}_{ckpt_tag}" if ckpt_tag else CKPT_NAME
    out_npz = os.path.join(OUT_DIR, f"{npz_prefix}_results.npz")
    np.savez(out_npz,
             loss_history=np.array(loss_history),
             mu_learned=torch.exp(model.log_mu).detach().numpy(),
             lam_learned=torch.exp(model.log_lambda).detach().numpy(),
             wx_learned=wx_np,
             wy_learned=wy_np,
             wz_learned=wz_np,
             X_IMG=X_IMG, Y_IMG=Y_IMG, Z_IMG=Z_IMG,
             dates=np.array(DATES))
    print(f"  Saved {out_npz}")

    total = time.time() - t_start
    print(f"\nTotal elapsed: {total/60:.1f} min")
    print("Done.")
