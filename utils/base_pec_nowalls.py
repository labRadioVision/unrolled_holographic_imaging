# -*- coding: utf-8 -*-
"""
run_wlista_synthetic_nowalls.py
===============================
LISTA / W-LISTA su dati sintetici "nowalls" (PEC, free-space, no pareti).

Variante AUTOCONTENUTA di run_wlista_synthetic.py: non richiede il file di
riferimento reale (empty_20_11_2024.mat). La geometria RX e' generata
sinteticamente (griglia 162x80 sul piano z=0) e la frequenza e' fissata
(FREQ_GHZ = 2.45), esattamente come run_mf_nowalls_cpu.py.

Dataset (Dataset TUM/sinthetic_data/):
  E_total_freespace_nowalls.mat  -- campo incidente / free-space (162,80)
        chiavi: E_total_freespace, E_phase_freespace
  E_total_Ken_PEC_nowalls.mat    -- campo totale (162,80,11), 11 posizioni
        chiavi: E_mag_total_mann, E_phase_total_mann, positions_mann (11,2)

Coordinate FEKO -> olografiche (come run_mf_nowalls_cpu.py):
  holo_x = TX_X + FEKO_y          (TX_X = 2.510)
  holo_z = FEKO_Z_WALL - FEKO_x   (FEKO_Z_WALL = 3.317)

NB: la cartella dataset si risolve automaticamente (../Dataset TUM/...),
con override tramite variabile d'ambiente HOLO_SYNTH_DIR.

Questo modulo NON va lanciato direttamente: e' il modulo base (loader + plot)
importato da run_wlista_synthetic_nowalls_gpu.py.
"""

import os, sys, time, argparse
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Importazioni GPU — opzionali: falliscono silenziosamente su macchine
# senza CUDA. Le funzioni di training che le usano daranno errore a
# runtime se chiamate senza GPU, ma le costanti e i loader restano
# disponibili per l'inferenza CPU.
try:
    import cupy as cp
    from holography_operator       import HolographyOperator
    from lista_holography          import LISTAHolography
    from lista_holography_weighted import WLISTAHolography
    _GPU_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    cp = None
    HolographyOperator = None
    LISTAHolography = None
    WLISTAHolography = None
    _GPU_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_z_true           import make_z_true_body_model

# ===========================================================================
# Paths
# ===========================================================================

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))


def _find_synth_dir():
    """Risolve la cartella dataset (override con HOLO_SYNTH_DIR)."""
    env = os.environ.get("HOLO_SYNTH_DIR")
    # Priorita': HOLO_SYNTH_DIR > cartella LOCALE (bundled nell'export) > parent.
    candidates = ([env] if env else []) + [
        os.path.join(SCRIPT_DIR, "..", "synthetic_sets"),
        os.path.join(SCRIPT_DIR, "synthetic_sets"),
    ]
    for c in candidates:
        if c and os.path.exists(os.path.join(c, "E_total_freespace_nowalls.mat")):
            return os.path.abspath(c)
    return os.path.abspath(candidates[0])


SYNTH_DIR   = _find_synth_dir()
OUT_DIR     = os.path.join(SCRIPT_DIR, "results_synthetic_nowalls")
CKPT_DIR    = os.path.join(SCRIPT_DIR, "checkpoints_lista")
ZTRUE_DIR   = os.path.join(SCRIPT_DIR, "results_z_true_nowalls")
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(CKPT_DIR,  exist_ok=True)
os.makedirs(ZTRUE_DIR, exist_ok=True)

# File sintetici nowalls
ETOTAL_FILE = os.path.join(SYNTH_DIR, "E_total_Ken_PEC_nowalls.mat")
EINC_FILE   = os.path.join(SYNTH_DIR, "E_total_freespace_nowalls.mat")

# ===========================================================================
# Imaging grid  (identica a holographic_imaging_gpu.py)
# ===========================================================================

X_IMG = np.linspace(0.0, 5.0, 161)
Y_IMG = np.linspace(0.0, 2.5,  81)
Z_IMG = np.linspace(0.3, 2.3,  65)
NX, NY, NZ = len(X_IMG), len(Y_IMG), len(Z_IMG)

# ===========================================================================
# Configurazione
# ===========================================================================

# Frequenza fissata (nessun file reference reale) — come run_mf_nowalls_cpu.py
FREQ_GHZ = 2.45
C        = 3.0e8
MU0      = 4.0e-7 * np.pi
BATCH_RX = 100

# Griglia RX sintetica (piano z=0), come run_mf_nowalls_cpu.py
NX_RX, NY_RX = 162, 80
RX_X0, RX_X1 = 0.02, 5.00
RX_Y0, RX_Y1 = 0.02, 2.48

# Train / validation split
TRAIN_IDX = [0, 1, 2, 3, 4, 5, 6, 7]   # 8 posizioni laterali
VAL_IDX   = [8, 9, 10]                  # 3 posizioni in profondita'

# Iperparametri (condivisi LISTA / W-LISTA)
K           = 10
N_EPOCHS    = 30
LR          = 5e-2
LAMBDA_INIT = 1e-4
L_EST       = 1.141e4   # riusa stima dal dataset reale (stessa geometria)

# W-LISTA only
LR_W        = 2.5e-2
W_LOG_CLAMP = 4.0

# Contrasto dielettrico (tessuto soft/muscolo a 2.45 GHz)
DELTA_Z = complex(1.5297, 0.0)

# z_true scaling: synthetic MF output is ~1.8e+03, z_true physical peak is 1.53
# (~1200x mismatch). Rescale z_true peak to MF amplitude scale to stabilise training.
# Set to None to use raw physical contrast.
ZTRUE_SCALE = None   # not needed after b normalisation by |E_inc|_mean (~2.7x residual ratio)

# Coordinate mapping FEKO -> olografiche (come run_mf_nowalls_cpu.py)
FEKO_Z_WALL = 3.317   # holo_z = FEKO_Z_WALL - FEKO_x
FEKO_X_OFF  = 2.510   # holo_x = FEKO_y + FEKO_X_OFF  (TX_X)

# Label posizioni (per plot / nomi file). Le 11 posizioni nowalls sono
# generiche: i nomi servono solo per i file di output / cache z_true.
POS_LABELS = [f"nowalls_pos{i+1:02d}" for i in range(11)]

# ===========================================================================
# Caricamento dati sintetici
# ===========================================================================

def feko_to_holo(feko_x, feko_y):
    """Converte posizione manichino FEKO -> coordinate olografiche (holo_x, holo_z)."""
    return feko_y + FEKO_X_OFF, FEKO_Z_WALL - feko_x


def make_rx_ref():
    """
    Costruisce un 'rx_ref' sintetico (niente file reale) con le chiavi attese
    dai costruttori dell'operatore: S21 (Nx,Ny,1 dummy), X (Nx,1), Y (1,Ny),
    freqs (1,). Griglia RX sul piano z=0, come run_mf_nowalls_cpu.py.
    """
    x_rx = np.linspace(RX_X0, RX_X1, NX_RX)
    y_rx = np.linspace(RX_Y0, RX_Y1, NY_RX)
    f0   = FREQ_GHZ * 1e9
    return {
        "S21":   np.zeros((NX_RX, NY_RX, 1), dtype=np.complex64),
        "X":     x_rx.reshape(-1, 1),
        "Y":     y_rx.reshape(1, -1),
        "freqs": np.array([f0]),
    }


def _freq_consts():
    f0    = FREQ_GHZ * 1e9
    k     = 2.0 * np.pi * f0 / C
    omega = 2.0 * np.pi * f0
    return f0, k, omega


def _load_complex(mat_dict, mag_key, phase_key):
    """
    Ricostruisce un campo complesso da modulo e fase (in radianti):
        E_complex = mag * exp(j * phase)
    Funziona sia se le chiavi esistono entrambe (nuovo formato mag+phase)
    sia se esiste solo il campo già complesso (formato legacy).
    """
    if mag_key in mat_dict and phase_key in mat_dict:
        mag   = mat_dict[mag_key].astype(np.float64)
        phase = mat_dict[phase_key].astype(np.float64)
        return (mag * np.exp(1j * phase)).astype(np.complex128)
    # fallback: cerca la prima chiave disponibile (formato legacy)
    keys = [k for k in mat_dict if not k.startswith("_")]
    raw  = mat_dict[keys[0]]
    print(f"  [NOTE] mag/phase keys not found — using '{keys[0]}' as-is")
    return raw.squeeze().astype(np.complex128)


def load_synthetic_dataset(model_type="wlista"):
    """
    Carica i .mat nowalls (E_total_freespace + E_total_Ken_PEC, modulo+fase).
    Ritorna (b_list, z_list, k, omega, rx_ref) per gli indici TRAIN_IDX.
    """
    print("Loading synthetic nowalls dataset ...")
    if not os.path.exists(EINC_FILE):
        raise FileNotFoundError(
            f"E_total_freespace_nowalls.mat not found in {SYNTH_DIR}")
    if not os.path.exists(ETOTAL_FILE):
        raise FileNotFoundError(
            f"E_total_Ken_PEC_nowalls.mat not found in {SYNTH_DIR}")

    einc_mat  = sio.loadmat(EINC_FILE)
    etotal_mat = sio.loadmat(ETOTAL_FILE)

    # --- campo incidente: mag * exp(j*phase) ---
    E_inc     = _load_complex(einc_mat,
                              mag_key="E_total_freespace",
                              phase_key="E_phase_freespace")  # (162, 80)
    # --- campo totale: mag * exp(j*phase) ---
    E_mag_tot  = etotal_mat["E_mag_total_mann"]               # (162, 80, 11)
    E_pha_tot  = etotal_mat["E_phase_total_mann"]             # (162, 80, 11)
    E_total    = (E_mag_tot * np.exp(1j * E_pha_tot)).astype(np.complex128)
    positions  = etotal_mat["positions_mann"]                 # (11, 2)

    if E_inc.shape != E_total.shape[:2]:
        raise ValueError(f"Shape mismatch: E_inc {E_inc.shape} vs E_total {E_total.shape[:2]}")

    # Frequenza fissata (nessun file reference reale)
    f0, k, omega = _freq_consts()
    print(f"  Frequency : {f0/1e9:.4f} GHz  (FREQ_GHZ={FREQ_GHZ})")
    print(f"  E_total   : {E_total.shape}  complex128")
    print(f"  E_inc     : {E_inc.shape}  complex128")

    b_list, z_list = [], []

    for idx in TRAIN_IDX:
        feko_x, feko_y = positions[idx]
        holo_x, holo_z = feko_to_holo(feko_x, feko_y)
        label = POS_LABELS[idx]

        # Campo diffuso normalizzato per |E_inc|_mean
        E_inc_mean = float(np.abs(E_inc).mean())
        b_np = ((E_total[:, :, idx] - E_inc) / E_inc_mean).ravel().astype(np.complex64)

        print(f"  [{idx:2d}] {label}  holo_x={holo_x:.3f}m  holo_z={holo_z:.3f}m  "
              f"|b|_max={np.abs(b_np).max():.3e}")

        # z_true con modello corpo parametrico
        z_true_path = os.path.join(ZTRUE_DIR, f"z_true_{label}.npz")
        if os.path.exists(z_true_path):
            z_true = np.load(z_true_path)["z_true"].astype(np.complex64)
            print(f"       z_true loaded from cache")
        else:
            print(f"       generating z_true ...")
            z_true = make_z_true_body_model(
                center_x = holo_x,
                center_z = holo_z,
                delta_z  = DELTA_Z,
                X=X_IMG, Y=Y_IMG, Z=Z_IMG,
            ).astype(np.complex64)
            np.savez(z_true_path, z_true=z_true,
                     X_IMG=X_IMG, Y_IMG=Y_IMG, Z_IMG=Z_IMG,
                     delta_z_re=np.float32(DELTA_Z.real),
                     delta_z_im=np.float32(DELTA_Z.imag),
                     label=label)
            print(f"       saved {z_true_path}")
        if ZTRUE_SCALE is not None:
            peak = np.abs(z_true).max()
            if peak > 0:
                z_true = z_true * (ZTRUE_SCALE / peak)
            print(f"       z_true scaled: peak {peak:.4f} -> {ZTRUE_SCALE:.3e}")

        b_list.append(b_np)
        z_list.append(z_true)

    # Geometria RX sintetica (niente file reale)
    rx_ref = make_rx_ref()
    print(f"\n  Loaded {len(TRAIN_IDX)} training samples")
    return b_list, z_list, k, omega, rx_ref


def load_validation_data(rx_ref, k):
    """Carica le N_val posizioni di validazione."""
    einc_mat   = sio.loadmat(EINC_FILE)
    etotal_mat = sio.loadmat(ETOTAL_FILE)
    E_inc      = _load_complex(einc_mat,
                               mag_key="E_total_freespace",
                               phase_key="E_phase_freespace")
    E_mag_tot  = etotal_mat["E_mag_total_mann"]
    E_pha_tot  = etotal_mat["E_phase_total_mann"]
    E_total    = (E_mag_tot * np.exp(1j * E_pha_tot)).astype(np.complex128)
    positions  = etotal_mat["positions_mann"]
    omega      = 2.0 * np.pi * (k * C / (2.0 * np.pi))

    b_list, z_list = [], []
    for idx in VAL_IDX:
        feko_x, feko_y = positions[idx]
        holo_x, holo_z = feko_to_holo(feko_x, feko_y)
        label = POS_LABELS[idx]

        E_inc_mean = float(np.abs(E_inc).mean())
        b_np = ((E_total[:, :, idx] - E_inc) / E_inc_mean).ravel().astype(np.complex64)

        z_true_path = os.path.join(ZTRUE_DIR, f"z_true_{label}.npz")
        if os.path.exists(z_true_path):
            z_true = np.load(z_true_path)["z_true"].astype(np.complex64)
        else:
            z_true = make_z_true_body_model(
                center_x=holo_x, center_z=holo_z, delta_z=DELTA_Z,
                X=X_IMG, Y=Y_IMG, Z=Z_IMG).astype(np.complex64)
            np.savez(z_true_path, z_true=z_true, X_IMG=X_IMG, Y_IMG=Y_IMG, Z_IMG=Z_IMG,
                     delta_z_re=np.float32(DELTA_Z.real), delta_z_im=np.float32(DELTA_Z.imag), label=label)
        if ZTRUE_SCALE is not None:
            peak = np.abs(z_true).max()
            if peak > 0:
                z_true = z_true * (ZTRUE_SCALE / peak)

        b_list.append(b_np)
        z_list.append(z_true)
        print(f"  VAL [{idx:2d}] {label}  holo_x={holo_x:.3f}m  holo_z={holo_z:.3f}m  "
              f"|b|_max={np.abs(b_np).max():.3e}")
    return b_list, z_list


# ===========================================================================
# Operatore
# ===========================================================================

def build_operator(rx_ref, k, omega):
    print("\nBuilding operator ...")
    Nx_rx, Ny_rx = rx_ref["S21"].shape[:2]
    x_rx   = rx_ref["X"][:, 0]
    y_rx   = rx_ref["Y"][0, :]
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
# Training  (LISTA o W-LISTA)
# ===========================================================================

def build_model(model_type, ckpt=None):
    if model_type == "lista":
        model = LISTAHolography(K=K, L_est=L_EST, lambda_init=LAMBDA_INIT)
    else:
        model = WLISTAHolography(K=K, L_est=L_EST,
                                 Nx=NX, Ny=NY, Nz=NZ,
                                 lambda_init=LAMBDA_INIT)
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state"])
    return model


def build_optimizer(model, model_type):
    if model_type == "lista":
        return torch.optim.Adam(model.parameters(), lr=LR)
    else:
        return torch.optim.Adam([
            {"params": [model.log_mu, model.log_lambda], "lr": LR},
            {"params": [model.log_wx, model.log_wy, model.log_wz], "lr": LR_W},
        ])


def train(op, b_list, z_list, model_type, ckpt_name, resume_ckpt=None):
    model  = build_model(model_type)
    optim  = build_optimizer(model, model_type)

    start_epoch  = 1
    loss_history = []
    best_loss    = float("inf")

    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optim_state" in ckpt:
            optim.load_state_dict(ckpt["optim_state"])
            print("  Optimizer state restored")
        else:
            print("  WARNING: no optimizer state — Adam restarted")
        start_epoch  = ckpt["epoch"] + 1
        loss_history = list(ckpt.get("loss_history", []))
        best_loss    = ckpt.get("loss", float("inf"))
        print(f"  Resuming from epoch {start_epoch}  (best={best_loss:.4e})\n")

    N_train = len(b_list)
    print(f"\n[{model_type.upper()}] Training  K={K}  epochs={start_epoch}->{N_EPOCHS}  "
          f"lr={LR:.1e}  N_train={N_train}")
    if model_type == "wlista":
        print(f"  lr_w={LR_W:.1e}  #params={model.num_params()}")
    print(f"  lambda_init={LAMBDA_INIT:.2e}  L_est={L_EST:.3e}")

    b_tensors = [torch.as_tensor(b) for b in b_list]
    z_tensors = [torch.as_tensor(z) for z in z_list]

    for epoch in range(start_epoch, N_EPOCHS + 1):
        t0 = time.time()
        model.train()
        optim.zero_grad()

        indices    = np.random.permutation(N_train)
        epoch_loss = torch.tensor(0.0)

        for idx in indices:
            z_pred     = model(b_tensors[idx], op, warm_start=True)
            loss_i     = torch.mean((z_pred.abs() - z_tensors[idx].abs()) ** 2)
            epoch_loss = epoch_loss + loss_i

        avg_loss = epoch_loss / N_train
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        # W-LISTA: clamp pesi
        if model_type == "wlista":
            with torch.no_grad():
                model.log_wx.clamp_(-W_LOG_CLAMP, W_LOG_CLAMP)
                model.log_wy.clamp_(-W_LOG_CLAMP, W_LOG_CLAMP)
                model.log_wz.clamp_(-W_LOG_CLAMP, W_LOG_CLAMP)

        elapsed  = time.time() - t0
        loss_val = avg_loss.item()
        loss_history.append(loss_val)

        mu_v  = torch.exp(model.log_mu).detach().numpy()
        lam_v = torch.exp(model.log_lambda).detach().numpy()

        if model_type == "wlista":
            wx, wy, wz = model.weight_stats()
            w_min = min(wx.min(), wy.min(), wz.min())
            w_max = max(wx.max(), wy.max(), wz.max())
            print(f"  Epoch {epoch:3d}/{N_EPOCHS}  loss={loss_val:.4e}  time={elapsed:.0f}s  "
                  f"mu=[{mu_v.min():.2e}..{mu_v.max():.2e}]  "
                  f"lam=[{lam_v.min():.2e}..{lam_v.max():.2e}]  "
                  f"w=[{w_min:.2e}..{w_max:.2e}]")
        else:
            print(f"  Epoch {epoch:3d}/{N_EPOCHS}  loss={loss_val:.4e}  time={elapsed:.0f}s  "
                  f"mu=[{mu_v.min():.2e}..{mu_v.max():.2e}]  "
                  f"lam=[{lam_v.min():.2e}..{lam_v.max():.2e}]")

        ckpt_data = dict(
            epoch=epoch, K=K, model_type=model_type,
            Nx=NX, Ny=NY, Nz=NZ,
            model_state=model.state_dict(),
            optim_state=optim.state_dict(),
            loss=loss_val, loss_history=loss_history,
            train_idx=TRAIN_IDX, val_idx=VAL_IDX,
        )

        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(ckpt_data, os.path.join(CKPT_DIR, f"{ckpt_name}_best.pt"))

        torch.save(ckpt_data, os.path.join(CKPT_DIR, f"{ckpt_name}_ep{epoch:03d}.pt"))
        print(f"  [ckpt] Saved epoch {epoch}")

    torch.save(ckpt_data, os.path.join(CKPT_DIR, f"{ckpt_name}.pt"))
    print(f"\n  Final ckpt  : {CKPT_DIR}/{ckpt_name}.pt")
    print(f"  Best ckpt   : {CKPT_DIR}/{ckpt_name}_best.pt  (loss={best_loss:.4e})")
    return model, loss_history


# ===========================================================================
# Inference & metriche
# ===========================================================================

def run_inference(model, op, b_np, model_type):
    model.eval()
    with torch.no_grad():
        b_cp    = cp.asarray(b_np.astype(np.complex128))
        z_mf_np = cp.asnumpy(op.AH(b_cp)).astype(np.complex64)
        z_net   = model(torch.as_tensor(b_np), op, warm_start=True)
    return z_mf_np, z_net.numpy().astype(np.complex64)


def mip_xy(z_np):
    return np.abs(z_np).reshape(NX, NY, NZ).max(axis=2)


def to_db(arr):
    return 20.0 * np.log10(arr / (arr.max() + 1e-30) + 1e-30)


def signal_clutter(z_pred, z_true):
    occ = np.abs(z_true) > 0
    mag = np.abs(z_pred)
    return (mag[occ].mean() + 1e-30) / (mag[~occ].mean() + 1e-30)


# ===========================================================================
# Plot
# ===========================================================================

def plot_results(b_list, z_list, labels, model, op, loss_history,
                 model_type, ckpt_name, split_tag="train"):
    net_name = model_type.upper()
    N_plot   = len(b_list)
    print(f"\nGenerating plots  [{split_tag}]  ({N_plot} samples) ...")

    z_mf_list, z_ista_list, z_net_list = [], [], []
    sc_mf, sc_ista, sc_net = [], [], []

    for b_np, z_true, lbl in zip(b_list, z_list, labels):
        print(f"  MF+ISTA+{net_name}  {lbl} ...")
        z_mf, z_net = run_inference(model, op, b_np, model_type)
        z_ista      = run_ista_baseline(op, b_np, K_iter=K, alpha=LAMBDA_INIT, L_est=L_EST)
        z_mf_list.append(z_mf)
        z_net_list.append(z_net)
        z_ista_list.append(z_ista)
        if z_true is not None:
            sc_mf.append(signal_clutter(z_mf, z_true))
            sc_ista.append(signal_clutter(z_ista, z_true))
            sc_net.append(signal_clutter(z_net, z_true))

    has_ztrue  = any(z is not None for z in z_list)
    n_rows     = 4 if has_ztrue else 3
    row_labels = (["Ground truth |z*|", "Matched filter", f"ISTA K={K}", f"{net_name} K={K}"]
                  if has_ztrue else
                  ["Matched filter", f"ISTA K={K}", f"{net_name} K={K}"])

    fig, axes = plt.subplots(n_rows, N_plot,
                             figsize=(max(4.0 * N_plot, 8), 3.5 * n_rows + 3))
    if N_plot == 1:
        axes = axes[:, np.newaxis]

    for col, (lbl, z_true, z_mf, z_ista, z_net) in enumerate(
            zip(labels, z_list, z_mf_list, z_ista_list, z_net_list)):
        z_rows = ([z_true, z_mf, z_ista, z_net] if has_ztrue
                  else [z_mf, z_ista, z_net])
        for row, (z, rl) in enumerate(zip(z_rows, row_labels)):
            ax = axes[row, col]
            if z is None:
                ax.axis("off"); continue
            proj     = mip_xy(z)
            use_db   = (rl != "Ground truth |z*|")
            data     = to_db(proj) if use_db else proj
            vmin, vmax = (-30, 0) if use_db else (0, proj.max())
            ax.pcolormesh(X_IMG, Y_IMG, data.T,
                          cmap="jet", vmin=vmin, vmax=vmax, shading="nearest")
            if row == 0:
                ax.set_title(lbl.replace("synth_", ""), fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{rl}\ny (m)", fontsize=7)
            ax.set_xlabel("x (m)", fontsize=7)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)

    # stats figure
    fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4))
    layers = np.arange(1, K + 1)
    mu_v   = torch.exp(model.log_mu).detach().numpy()
    lam_v  = torch.exp(model.log_lambda).detach().numpy()

    axes2[0].semilogy(range(1, len(loss_history)+1), loss_history, 'b-o', ms=4)
    axes2[0].set_xlabel("Epoch"); axes2[0].set_ylabel("Avg train loss")
    axes2[0].set_title("Training loss"); axes2[0].grid(True, which='both', alpha=0.3)

    axes2[1].bar(layers, mu_v, color='steelblue')
    axes2[1].set_xlabel("Layer k"); axes2[1].set_ylabel("mu_k")
    axes2[1].set_title("Learned step sizes")

    axes2[2].bar(layers, lam_v, color='coral')
    axes2[2].set_xlabel("Layer k"); axes2[2].set_ylabel("lambda_k")
    axes2[2].set_title("Learned thresholds")

    if sc_net:
        x = np.arange(len(sc_net)); w = 0.25
        short = [l.replace("synth_", "")[:12] for l in labels[:len(sc_net)]]
        axes2[3].bar(x - w, sc_mf,   w, label='MF',            color='steelblue')
        axes2[3].bar(x,     sc_ista, w, label=f'ISTA K={K}',   color='orange')
        axes2[3].bar(x + w, sc_net,  w, label=f'{net_name} K={K}', color='seagreen')
        axes2[3].set_xticks(x); axes2[3].set_xticklabels(short, rotation=30, fontsize=6)
        axes2[3].set_ylabel("S/C ratio"); axes2[3].set_title("Signal/clutter")
        axes2[3].legend(fontsize=8)
        # print S/C table
        print(f"\n  Signal/Clutter [{split_tag}]:")
        print(f"  {'label':30s}  {'MF':>8}  {'ISTA':>8}  {net_name:>10}")
        for lbl, sm, si, sn in zip(labels[:len(sc_net)], sc_mf, sc_ista, sc_net):
            print(f"  {lbl:30s}  {sm:8.3f}  {si:8.3f}  {sn:10.3f}")
    else:
        axes2[3].text(0.5, 0.5, "No z_true\n(S/C N/A)",
                      ha='center', va='center', transform=axes2[3].transAxes)

    fig.suptitle(f"{net_name} synthetic — K={K}  epochs={N_EPOCHS}  [{split_tag}]", fontsize=10)
    fig2.suptitle("Training stats & Signal/Clutter", fontsize=10)
    fig.tight_layout(); fig2.tight_layout()

    tag = f"{ckpt_name}_{split_tag}"
    fig.savefig( os.path.join(OUT_DIR, f"{tag}_imgs.png"),  dpi=130, bbox_inches="tight")
    fig2.savefig(os.path.join(OUT_DIR, f"{tag}_stats.png"), dpi=130, bbox_inches="tight")

    # weights heatmap (W-LISTA only)
    if model_type == "wlista":
        fig3, axes3 = plt.subplots(1, 3, figsize=(18, 4))
        wx, wy, wz = model.weight_stats()
        for ax, ww, coord, lbl3 in zip(
                axes3, [wx, wy, wz], [X_IMG, Y_IMG, Z_IMG],
                ["w_k(x)", "w_k(y)", "w_k(z)"]):
            im = ax.pcolormesh(coord, layers, ww, cmap="viridis", shading="nearest")
            plt.colorbar(im, ax=ax, label="weight")
            ax.set_xlabel("position (m)"); ax.set_ylabel("layer k")
            ax.set_title(lbl3)
        fig3.suptitle("Learned factorized weights", fontsize=10)
        fig3.tight_layout()
        fig3.savefig(os.path.join(OUT_DIR, f"{tag}_weights.png"), dpi=130, bbox_inches="tight")
        plt.close(fig3)

    plt.close("all")
    print(f"  Saved plots in {OUT_DIR}/  (prefix: {tag})")

    # salva .mat per ogni campione
    for lbl, z_mf, z_net in zip(labels, z_mf_list, z_net_list):
        mat_path = os.path.join(OUT_DIR, f"{ckpt_name}_{lbl}_z.mat")
        sio.savemat(mat_path, {
            "z_net":  z_net.reshape(NX, NY, NZ),
            "z_mf":   z_mf.reshape(NX, NY, NZ),
            "x_img":  X_IMG, "y_img": Y_IMG, "z_img": Z_IMG,
            "label":  lbl, "model": model_type,
        })
    print(f"  Saved {len(labels)} .mat files in {OUT_DIR}/")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LISTA/W-LISTA training + validation su dati sintetici")
    parser.add_argument("--model", choices=["lista", "wlista"], default="wlista",
                        help="Tipo di modello (default: wlista)")
    parser.add_argument("--resume", default=None,
                        help="Checkpoint .pt da cui riprendere il training")
    parser.add_argument("--infer-only", default=None,
                        help="Salta training: carica questo checkpoint e fai solo inference")
    parser.add_argument("--infer-pos", nargs="+", type=int, default=None,
                        help="Indici posizioni per inference (0-10). Default: train+val")
    args = parser.parse_args()

    model_type = args.model
    ckpt_name  = f"{model_type}_synthetic"

    t_start = time.time()
    print("=" * 70)
    print(f"{model_type.upper()} training su dati sintetici  —  N_train={len(TRAIN_IDX)}  N_val={len(VAL_IDX)}")
    print(f"  Train idx : {TRAIN_IDX}")
    print(f"  Val   idx : {VAL_IDX}")
    print("=" * 70)

    # --- carica dati di training ---
    if args.infer_only and args.infer_pos is not None:
        # solo geometria RX sintetica + frequenza fissa
        f0, k, omega = _freq_consts()
        rx_ref = make_rx_ref()
        b_train, z_train = [], []
    else:
        b_train, z_train, k, omega, rx_ref = load_synthetic_dataset(model_type)

    op = build_operator(rx_ref, k, omega)

    if args.infer_only:
        print(f"\nInference-only — loading {args.infer_only}")
        ckpt = torch.load(args.infer_only, map_location="cpu", weights_only=False)
        model = build_model(ckpt.get("model_type", model_type), ckpt)
        loss_history = list(ckpt.get("loss_history", [ckpt["loss"]]))
        print(f"  Loaded epoch={ckpt['epoch']}  loss={ckpt['loss']:.4e}")

        # quali posizioni inferire?
        if args.infer_pos is not None:
            all_idx = args.infer_pos
        else:
            all_idx = list(TRAIN_IDX) + list(VAL_IDX)

        einc_mat   = sio.loadmat(EINC_FILE)
        etotal_mat = sio.loadmat(ETOTAL_FILE)
        E_inc      = _load_complex(einc_mat,
                                   mag_key="E_total_freespace",
                                   phase_key="E_phase_freespace")
        E_mag_tot  = etotal_mat["E_mag_total_mann"]
        E_pha_tot  = etotal_mat["E_phase_total_mann"]
        E_total    = (E_mag_tot * np.exp(1j * E_pha_tot)).astype(np.complex128)
        positions  = etotal_mat["positions_mann"]

        # split in train / val per plot separati
        for split_idx, split_tag in [(TRAIN_IDX, "train"), (VAL_IDX, "val")]:
            sel = [i for i in all_idx if i in split_idx]
            if not sel:
                continue
            b_sel, z_sel, lbl_sel = [], [], []
            for idx in sel:
                fx, fy  = positions[idx]
                hx, hz  = feko_to_holo(fx, fy)
                lbl     = POS_LABELS[idx]
                b_np    = (E_total[:,:,idx].astype(np.complex128) - E_inc).ravel().astype(np.complex64)
                ztpath  = os.path.join(ZTRUE_DIR, f"z_true_{lbl}.npz")
                z_true  = np.load(ztpath)["z_true"].astype(np.complex64) if os.path.exists(ztpath) else None
                b_sel.append(b_np); z_sel.append(z_true); lbl_sel.append(lbl)

            plot_results(b_sel, z_sel, lbl_sel, model, op, loss_history,
                         model_type, ckpt_name, split_tag=split_tag)

    else:
        # --- training ---
        model, loss_history = train(op, b_train, z_train, model_type,
                                    ckpt_name, resume_ckpt=args.resume)

        # plot training set
        train_labels = [POS_LABELS[i] for i in TRAIN_IDX]
        plot_results(b_train, z_train, train_labels, model, op, loss_history,
                     model_type, ckpt_name, split_tag="train")

        # plot validation set
        print("\nLoading validation data ...")
        b_val, z_val = load_validation_data(rx_ref, k)
        val_labels   = [POS_LABELS[i] for i in VAL_IDX]
        plot_results(b_val, z_val, val_labels, model, op, loss_history,
                     model_type, ckpt_name, split_tag="val")

        # salva risultati globali
        out_npz = os.path.join(OUT_DIR, f"{ckpt_name}_results.npz")
        npz_data = dict(
            loss_history=np.array(loss_history),
            mu_learned=torch.exp(model.log_mu).detach().numpy(),
            lam_learned=torch.exp(model.log_lambda).detach().numpy(),
            train_idx=np.array(TRAIN_IDX),
            val_idx=np.array(VAL_IDX),
        )
        if model_type == "wlista":
            wx, wy, wz = model.weight_stats()
            npz_data.update(wx_learned=wx, wy_learned=wy, wz_learned=wz)
        np.savez(out_npz, **npz_data)
        print(f"  Saved {out_npz}")

    total = time.time() - t_start
    print(f"\nTotal elapsed: {total/60:.1f} min")
    print("Done.")
