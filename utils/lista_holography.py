# -*- coding: utf-8 -*-
"""
lista_holography.py
===================
LISTA (Learned ISTA) for holographic reconstruction.

Unrolls K ISTA iterations into a trainable PyTorch network.
The physical operator A (Green's function) is kept fixed;
only the algorithmic parameters are learned:
  - mu_k     : per-layer step size          (K scalars)
  - lambda_k : per-layer soft-threshold     (K scalars)

Requires:
  - holography_operator.py   (this repo)
  - PyTorch 2.x + CUDA
  - CuPy  (already installed in .conda)
  - A labeled dataset of (b_scatter, z_true) pairs

Usage
-----
1. Training:
   python lista_holography.py --mode train --epochs 100 --K 10

2. Inference on a single measurement:
   python lista_holography.py --mode infer --checkpoint checkpoints/lista_best.pt
"""

import argparse
import os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

try:
    import cupy as cp
    from holography_operator import HolographyOperator
except (ImportError, ModuleNotFoundError):
    cp = None
    HolographyOperator = None

# ===========================================================================
# Paths  (mirror holographic_imaging_gpu.py)
# ===========================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "Dataset TUM", "collected_data"))
OUT_DIR    = os.path.join(SCRIPT_DIR, "results_holographic")
CKPT_DIR   = os.path.join(SCRIPT_DIR, "checkpoints_lista")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

EMPTY_FILE = os.path.join(DATA_DIR, "empty_20_11_2024.mat")

# ===========================================================================
# Physical / imaging parameters  (keep in sync with holographic_imaging_gpu.py)
# ===========================================================================

FREQ_IDX   = 16
R_TX       = np.array([-0.002, 1.348, 2.667])

X_IMG = np.linspace(0.0, 5.0, 161)
Y_IMG = np.linspace(0.0, 2.5,  81)
Z_IMG = np.linspace(0.3, 2.3,  65)

C   = 3.0e8
MU0 = 4.0e-7 * np.pi

BATCH_RX = 100

# ===========================================================================
# LISTA network
# ===========================================================================

class LISTAHolography(nn.Module):
    """
    K-layer unrolled ISTA.

    Each layer k computes:
        residual = A z - b
        grad     = A^H residual
        z        = soft_thresh( z - mu_k * grad,  lambda_k )

    Learnable (all log-parameterized → always positive):
        log_mu     : (K,)  per-layer step sizes
        log_lambda : (K,)  per-layer soft-threshold values
    """

    def __init__(self, K: int, L_est: float, lambda_init: float = 1e-3):
        """
        K           : number of unrolled layers (start with 10)
        L_est       : Lipschitz constant (from op.lipschitz() or saved log)
        lambda_init : initial threshold value (tune around ISTA alpha that worked)
        """
        super().__init__()
        self.K = K

        # step size init: 1/L  (same as ISTA default, STEP_SCALE=1.0)
        mu0 = 1.0 / L_est
        self.log_mu     = nn.Parameter(torch.full((K,), float(np.log(mu0))))
        self.log_lambda = nn.Parameter(torch.full((K,), float(np.log(lambda_init))))

    def forward(self, b: torch.Tensor, op: HolographyOperator,
                warm_start: bool = False) -> torch.Tensor:
        """
        Parameters
        ----------
        b          : (N_rx,) torch.cfloat  scattered field (on CPU or CUDA)
        op         : HolographyOperator
        warm_start : if True, z_0 = A^H b  (matched filter init)
                     if False, z_0 = 0

        Returns
        -------
        z : (N_vox,) torch.cfloat  reconstructed permittivity contrast
        """
        device = b.device

        if warm_start:
            with torch.no_grad():
                z = op.AH_torch(b).to(device)   # detached init, no grad history
        else:
            z = torch.zeros(op.N_vox, dtype=torch.cfloat, device=device)

        for k in range(self.K):
            mu_k     = torch.exp(self.log_mu[k])
            lambda_k = torch.exp(self.log_lambda[k])

            # gradient step: z <- z - mu_k * A^H(A z - b)
            residual = op.A_torch(z) - b
            grad     = op.AH_torch(residual)
            z        = z - mu_k * grad

            # proximal step: complex soft-threshold on magnitude
            z = _complex_soft_thresh(z, lambda_k)

        return z

    def print_params(self):
        mu_v = torch.exp(self.log_mu).detach().cpu().numpy()
        la_v = torch.exp(self.log_lambda).detach().cpu().numpy()
        print("  mu     : " + "  ".join("%.3e" % v for v in mu_v))
        print("  lambda : " + "  ".join("%.3e" % v for v in la_v))


def _complex_soft_thresh(z: torch.Tensor, threshold) -> torch.Tensor:
    """
    Shrink the modulus of z by threshold, preserve phase.
    Works for scalar or (N_vox,) tensor threshold.
    """
    mag   = z.abs()
    scale = torch.clamp(mag - threshold, min=0.0) / (mag + 1e-30)
    return z * scale


# ===========================================================================
# Dataset
# ===========================================================================

class HolographyDataset(torch.utils.data.Dataset):
    """
    Each item: (b_scatter, z_true)

    b_scatter : (N_rx,)   complex64  — pre-computed scattered field vector
    z_true    : (N_vox,)  complex64  — ground truth permittivity contrast
                Flatten z_true from (Nx, Ny, Nz) with .ravel() before passing.

    How to build this dataset
    -------------------------
    Option A — simulation (synthetic, unlimited data):
        z_true = known sparse contrast volume (your simulation output)
        b_true = op.A(cp.asarray(z_true))  then add noise
        b_scatter = cp.asnumpy(b_true).astype(np.complex64)

    Option B — from .mat files (real measurements with known ground truth):
        Load the .mat file produced by holographic_imaging_gpu.py.
        Use z_vol as z_true (pseudo-label from a high-quality run).
        Extract b_scatter from the measurement file as done in the main script.
    """

    def __init__(self, b_list: list, z_list: list):
        assert len(b_list) == len(z_list), "b and z lists must have the same length"
        self.b = b_list   # list of np.complex64 arrays, each (N_rx,)
        self.z = z_list   # list of np.complex64 arrays, each (N_vox,)

    def __len__(self):
        return len(self.b)

    def __getitem__(self, idx):
        b = torch.as_tensor(self.b[idx].astype(np.complex64))
        z = torch.as_tensor(self.z[idx].astype(np.complex64))
        return b, z

    @staticmethod
    def from_mat_files(b_mat_paths: list, z_mat_paths: list):
        """
        Convenience loader from .mat files.

        b_mat_paths : list of measurement .mat files (contain S21, X, Y, freqs)
        z_mat_paths : list of reconstruction .mat files (contain z_vol_flat)

        The reconstruction .mat is produced by holographic_imaging_gpu.py and
        stored in results_holographic/. Use it as pseudo-label.
        """
        ref = sio.loadmat(EMPTY_FILE)
        freqs = ref["freqs"].flatten()
        f0    = freqs[FREQ_IDX]
        k     = 2.0 * np.pi * f0 / C

        b_list, z_list = [], []

        for b_path, z_path in zip(b_mat_paths, z_mat_paths):
            # --- scattered field ---
            meas   = sio.loadmat(b_path)
            s_meas = meas["S21"][:, :, FREQ_IDX].astype(np.complex128)
            s_emp  = ref["S21"][:, :, FREQ_IDX].astype(np.complex128)
            c_corr = (np.sum(s_emp * np.conj(s_meas))
                      / np.sum(np.abs(s_meas)**2))
            b_scatter = (c_corr * s_meas - s_emp).ravel().astype(np.complex64)

            # --- ground truth z ---
            z_mat  = sio.loadmat(z_path)
            z_true = z_mat["z_vol_flat"].ravel().astype(np.complex64)

            b_list.append(b_scatter)
            z_list.append(z_true)

        return HolographyDataset(b_list, z_list)


# ===========================================================================
# Loss functions
# ===========================================================================

def magnitude_mse(z_pred: torch.Tensor, z_true: torch.Tensor) -> torch.Tensor:
    """MSE on |z| — insensitive to global phase, good default."""
    return torch.mean((z_pred.abs() - z_true.abs()) ** 2)


def complex_mse(z_pred: torch.Tensor, z_true: torch.Tensor) -> torch.Tensor:
    """Full complex MSE — use only if phase is calibrated."""
    return torch.mean(torch.abs(z_pred - z_true) ** 2)


# ===========================================================================
# Training loop
# ===========================================================================

def train(model: LISTAHolography,
          dataset: HolographyDataset,
          op: HolographyOperator,
          n_epochs: int = 100,
          lr: float     = 1e-3,
          loss_fn       = magnitude_mse,
          device: str   = "cpu"):
    """
    Train LISTA on the labeled dataset.

    Note on device
    --------------
    The A / A^H operators always run on CUDA (CuPy).
    `device` here controls where the torch tensors live between operator calls.
    Use device="cpu" if VRAM is tight; use "cuda" for full GPU pipeline.
    """

    loader    = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    best_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        model.train()

        for b, z_true in loader:
            b      = b.squeeze(0).to(device)       # (N_rx,)  cfloat
            z_true = z_true.squeeze(0).to(device)  # (N_vox,) cfloat

            optimizer.zero_grad()

            z_pred = model(b, op, warm_start=True)
            loss   = loss_fn(z_pred, z_true)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(dataset)

        print(f"Epoch {epoch:4d}/{n_epochs}  loss={avg_loss:.4e}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")
        if epoch % 10 == 0 or epoch == 1:
            model.print_params()

        # save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch":       epoch,
                "K":           model.K,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "loss":        best_loss,
            }, os.path.join(CKPT_DIR, "lista_best.pt"))

        # periodic checkpoint
        if epoch % 25 == 0:
            torch.save({
                "epoch":       epoch,
                "K":           model.K,
                "model_state": model.state_dict(),
                "loss":        avg_loss,
            }, os.path.join(CKPT_DIR, f"lista_epoch{epoch:04d}.pt"))

    print(f"\nTraining done. Best loss: {best_loss:.4e}")
    print(f"Checkpoints saved to: {CKPT_DIR}")


# ===========================================================================
# Inference + save  (mirrors holographic_imaging_gpu.py output format)
# ===========================================================================

def infer_and_save(model: LISTAHolography,
                   op: HolographyOperator,
                   b_scatter_np: np.ndarray,
                   label: str,
                   f0: float,
                   device: str = "cpu"):
    """
    Run LISTA inference on one scattered field vector, save PNG + MAT.

    b_scatter_np : (N_rx,) complex numpy array (drift-corrected)
    label        : measurement label (e.g. '30_11_2024')
    f0           : frequency [Hz]
    """
    model.eval()
    with torch.no_grad():
        b  = torch.as_tensor(b_scatter_np.astype(np.complex64)).to(device)
        z  = model(b, op, warm_start=True)
        z_cpu = z.cpu().numpy()

    # reshape and project
    z_3d   = np.abs(z_cpu).reshape(len(X_IMG), len(Y_IMG), len(Z_IMG))
    xy_proj = z_3d.max(axis=2)
    xz_proj = z_3d.max(axis=1)
    yz_proj = z_3d.max(axis=0)

    def to_db(arr):
        return 20.0 * np.log10(arr / arr.max() + 1e-30)

    xy_db = to_db(xy_proj)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.pcolormesh(X_IMG, Y_IMG, xy_db.T,
                       cmap="jet", vmin=-30.0, vmax=0.0, shading="auto")
    plt.colorbar(im, ax=ax, label="dB")
    ax.set_xlabel("x  (m)"); ax.set_ylabel("y  (m)")
    ax.set_aspect("equal"); ax.set_title("MIP xy  (LISTA)")
    fig.suptitle(f"LISTA  K={model.K}  —  {label}  ({f0/1e9:.2f} GHz)")
    fig.tight_layout()

    sfx = f"_lista_K{model.K}_lam4_z{len(Z_IMG)}_{Z_IMG[0]:.1f}_{Z_IMG[-1]:.1f}"
    out_png = os.path.join(OUT_DIR, f"holographic_{label}{sfx}.png")
    out_mat = os.path.join(OUT_DIR, f"holographic_{label}{sfx}.mat")

    fig.savefig(out_png, dpi=150)
    plt.close("all")
    print(f"Saved {out_png}")

    sio.savemat(out_mat, {
        "z_vol":     z_cpu.reshape(len(X_IMG), len(Y_IMG), len(Z_IMG)),
        "z_vol_flat": z_cpu,
        "z_3d":      z_3d,
        "xy_proj":   xy_proj, "xy_db":  xy_db,
        "xz_proj":   xz_proj,
        "yz_proj":   yz_proj,
        "X_IMG": X_IMG, "Y_IMG": Y_IMG, "Z_IMG": Z_IMG,
        "f0_Hz": f0, "label": label, "K": model.K,
    })
    print(f"Saved {out_mat}")

    # top-5 voxels
    flat_sorted = np.argsort(z_3d.ravel())[::-1][:5]
    print("\n[LISTA] Top-5 voxels:")
    for rank, flat_idx in enumerate(flat_sorted):
        ii, jj, kk = np.unravel_index(flat_idx, z_3d.shape)
        print("  #%d  x=%.3f m  y=%.3f m  z=%.3f m  |z|=%.4e"
              % (rank+1, X_IMG[ii], Y_IMG[jj], Z_IMG[kk], z_3d[ii, jj, kk]))


# ===========================================================================
# Operator factory  (builds op from .mat files, same as main script)
# ===========================================================================

def build_operator(meas_file: str = None) -> tuple:
    """
    Load reference data, build HolographyOperator.
    Returns (op, b_scatter_np, label, f0).

    If meas_file is None, uses MEAS_FILE defined at top of holographic_imaging_gpu.py.
    """
    if meas_file is None:
        meas_file = os.path.join(DATA_DIR, "empty_30_11_2024.mat")

    ref  = sio.loadmat(EMPTY_FILE)
    meas = sio.loadmat(meas_file)

    freqs = ref["freqs"].flatten()
    f0    = freqs[FREQ_IDX]
    k     = 2.0 * np.pi * f0 / C
    omega = 2.0 * np.pi * f0
    lam   = C / f0

    label = os.path.splitext(os.path.basename(meas_file))[0].replace("empty_", "")
    print(f"  Measurement : {label}")
    print(f"  Frequency   : {f0/1e9:.3f} GHz  (index {FREQ_IDX})")
    print(f"  lambda      : {lam:.4f} m    k = {k:.2f} rad/m")

    # scattered field
    s_meas  = meas["S21"][:, :, FREQ_IDX].astype(np.complex128)
    s_empty = ref["S21"][:, :, FREQ_IDX].astype(np.complex128)
    c_corr  = np.sum(s_empty * np.conj(s_meas)) / np.sum(np.abs(s_meas)**2)
    b_scatter_np = (c_corr * s_meas - s_empty).ravel()
    print(f"  Drift correction: |c|={20*np.log10(np.abs(c_corr)):.4f} dB  "
          f"angle={np.degrees(np.angle(c_corr)):.2f} deg")

    # geometry
    Nx, Ny = ref["S21"].shape[:2]
    x_rx   = ref["X"][:, 0]
    y_rx   = ref["Y"][0, :]
    XX, YY = np.meshgrid(x_rx, y_rx, indexing="ij")
    r_rx_cpu  = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(Nx * Ny)])

    XXv, YYv, ZZv = np.meshgrid(X_IMG, Y_IMG, Z_IMG, indexing="ij")
    r_vox_cpu = np.column_stack([XXv.ravel(), YYv.ravel(), ZZv.ravel()])
    dV        = (X_IMG[1]-X_IMG[0]) * (Y_IMG[1]-Y_IMG[0]) * (Z_IMG[1]-Z_IMG[0])

    print(f"  Receivers : {Nx*Ny}  ({Nx}x{Ny})")
    print(f"  Voxels    : {len(r_vox_cpu)}  "
          f"({len(X_IMG)}x{len(Y_IMG)}x{len(Z_IMG)})")

    op = HolographyOperator(
        r_rx     = cp.asarray(r_rx_cpu),
        r_vox    = cp.asarray(r_vox_cpu),
        k        = k,
        omega    = omega,
        mu0      = MU0,
        dV       = dV,
        batch_rx = BATCH_RX,
    )

    return op, b_scatter_np, label, f0


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="LISTA holographic reconstruction")
    p.add_argument("--mode",       choices=["train", "infer"], default="infer")
    p.add_argument("--K",          type=int,   default=10,    help="LISTA layers")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--lambda_init",type=float, default=0.1,
                   help="Initial threshold (match ISTA alpha that gave best results)")
    p.add_argument("--checkpoint", type=str,   default=None,
                   help="Path to .pt checkpoint (infer mode)")
    p.add_argument("--meas_file",  type=str,   default=None,
                   help="Measurement .mat file (default: empty_30_11_2024.mat)")
    p.add_argument("--device",     type=str,   default="cpu",
                   help="Torch device for tensors between operator calls")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("LISTA Holographic Reconstruction")
    print("=" * 60)

    # --- build operator ---
    print("\nLoading data and building operator ...")
    op, b_scatter_np, label, f0 = build_operator(args.meas_file)

    # --- Lipschitz constant ---
    # Use value from existing log if available (saves ~5 min recomputation)
    L_SAVED = 1.141e4   # from ista_lam4_alpha0.1_run.log — update if grid changes
    print(f"  L_est (saved) = {L_SAVED:.3e}  "
          f"(recompute with op.lipschitz() if grid changed)")

    if args.mode == "train":
        # ----------------------------------------------------------------
        # Build dataset — FILL THIS SECTION with your labeled pairs
        # ----------------------------------------------------------------
        # Example A: load pseudo-labels from existing ISTA .mat results
        #
        # b_mat_paths = [args.meas_file]   # add more measurement files here
        # z_mat_paths = [os.path.join(OUT_DIR,
        #     "holographic_30_11_2024_ista_lam4_z65_0.3_2.3_alpha0.1.mat")]
        # dataset = HolographyDataset.from_mat_files(b_mat_paths, z_mat_paths)
        #
        # Example B: synthetic data (uncomment and adapt)
        #
        # from your_simulation_module import generate_synthetic_pair
        # b_list, z_list = zip(*[generate_synthetic_pair(op) for _ in range(50)])
        # dataset = HolographyDataset(list(b_list), list(z_list))
        #
        # ----------------------------------------------------------------
        raise NotImplementedError(
            "Fill in the dataset section above with your labeled pairs.\n"
            "See HolographyDataset.from_mat_files() for the pseudo-label approach."
        )

        model = LISTAHolography(K=args.K, L_est=L_SAVED,
                                lambda_init=args.lambda_init)
        print(f"\nModel: LISTA  K={args.K}  "
              f"params={sum(p.numel() for p in model.parameters())}")
        model.print_params()

        train(model, dataset, op,
              n_epochs=args.epochs, lr=args.lr,
              loss_fn=magnitude_mse, device=args.device)

    elif args.mode == "infer":
        ckpt_path = args.checkpoint or os.path.join(CKPT_DIR, "lista_best.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"No checkpoint found at {ckpt_path}.\n"
                "Run with --mode train first."
            )
        ckpt  = torch.load(ckpt_path, map_location="cpu")
        model = LISTAHolography(K=ckpt["K"], L_est=L_SAVED,
                                lambda_init=args.lambda_init)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint: epoch={ckpt['epoch']}  loss={ckpt['loss']:.4e}")
        model.print_params()

        infer_and_save(model, op, b_scatter_np, label, f0,
                       device=args.device)
