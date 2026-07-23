# -*- coding: utf-8 -*-
"""
run_wlista_synthetic_adipose.py
===============================
W-LISTA (baseline, no low-rank correction) on the adipose-tissue synthetic
dataset (E_total_Ken_grasso_nowalls.mat).

Run:
    python run_wlista_synthetic_adipose.py > wlista_adipose.log 2>&1
    python run_wlista_synthetic_adipose.py --resume checkpoints_lista_ken_grasso/wlista_ken_grasso_ep005.pt

The dataset E_total_Ken_grasso_nowalls.mat must be in synthetic_sets/.
"""

import os, sys, time, argparse
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import scipy.io as sio
import torch
import cupy as cp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "utils"))

import base_pec_nowalls as base
from holography_operator_fast  import HolographyOperatorFast
from lista_holography_weighted import WLISTAHolography
import inference_common as ic

# ---------------------------------------------------------------------------
# Override: data file (bundled in synthetic_sets/)
# ---------------------------------------------------------------------------
SYNTH_DIR   = os.path.join(SCRIPT_DIR, "synthetic_sets")
ETOTAL_FILE = os.path.join(SYNTH_DIR, "E_total_Ken_grasso_nowalls.mat")
base.ETOTAL_FILE = ETOTAL_FILE
base.SYNTH_DIR   = SYNTH_DIR

if not os.path.exists(ETOTAL_FILE):
    raise FileNotFoundError(
        f"File not found: {ETOTAL_FILE}\n"
        f"Make sure it is in {SYNTH_DIR}")

# --- auto-detect number of positions ---
_mat  = sio.loadmat(ETOTAL_FILE)
_keys = [k for k in _mat if not k.startswith("_")]
_n_pos, _k = None, _keys[0]
for _k in _keys:
    if _mat[_k].ndim == 3 and _mat[_k].shape[:2] == (162, 80):
        _n_pos = _mat[_k].shape[2]
        break
if _n_pos is None:
    _n_pos = _mat[_keys[0]].shape[-1]
print(f"[adipose] N_pos detected = {_n_pos}  (key: '{_k}')")

base.TRAIN_IDX  = list(range(_n_pos))
base.VAL_IDX    = [_n_pos - 1]
base.POS_LABELS = [f"ken_grasso_pos{i:02d}" for i in range(_n_pos)]

# ---------------------------------------------------------------------------
# Dedicated output folders
# ---------------------------------------------------------------------------
OUT_DIR  = os.path.join(SCRIPT_DIR, "results_synthetic_ken_grasso")
CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints_lista_ken_grasso")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
base.OUT_DIR  = OUT_DIR
base.CKPT_DIR = CKPT_DIR

# ---------------------------------------------------------------------------
# Hyper-parameters (identical to the synthetic W-LISTA baseline)
# ---------------------------------------------------------------------------
NX, NY, NZ  = base.NX, base.NY, base.NZ
K           = base.K
N_EPOCHS    = base.N_EPOCHS
LR          = 1e-2     # reduced from base.LR=5e-2 (higher LR looked too large)
LR_W        = 2.5e-1   # reduced proportionally from base.LR_W=5e-1
LAMBDA_INIT = base.LAMBDA_INIT
L_EST       = base.L_EST
W_LOG_CLAMP = base.W_LOG_CLAMP
REF_EPOCH_START = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Operator + training
# ---------------------------------------------------------------------------
def build_operator_fast(rx_ref, k, omega):
    print("\nBuilding FAST operator (DLPack) ...")
    Nx_rx, Ny_rx = rx_ref["S21"].shape[:2]
    x_rx = rx_ref["X"][:, 0]; y_rx = rx_ref["Y"][0, :]
    XX, YY = np.meshgrid(x_rx, y_rx, indexing="ij")
    r_rx  = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(Nx_rx * Ny_rx)])
    XXv, YYv, ZZv = np.meshgrid(base.X_IMG, base.Y_IMG, base.Z_IMG, indexing="ij")
    r_vox = np.column_stack([XXv.ravel(), YYv.ravel(), ZZv.ravel()])
    dV = (base.X_IMG[1]-base.X_IMG[0])*(base.Y_IMG[1]-base.Y_IMG[0])*(base.Z_IMG[1]-base.Z_IMG[0])
    print(f"  Receivers : {r_rx.shape[0]}  ({Nx_rx}x{Ny_rx})")
    print(f"  Voxels    : {r_vox.shape[0]}  ({NX}x{NY}x{NZ})")
    return HolographyOperatorFast(cp.asarray(r_rx), cp.asarray(r_vox),
                                  k=k, omega=omega, mu0=base.MU0, dV=dV,
                                  batch_rx=base.BATCH_RX)


def mag_mse(model, op, b, z_true):
    z_pred = model(b, op, warm_start=True)
    return torch.mean((z_pred.abs() - z_true.abs()) ** 2)


def train(op, b_tr, z_tr, b_va, z_va, ckpt_name, resume=None):
    model = WLISTAHolography(K=K, L_est=L_EST, Nx=NX, Ny=NY, Nz=NZ,
                             lambda_init=LAMBDA_INIT).to(DEVICE)
    optim = torch.optim.Adam([
        {"params": [model.log_mu, model.log_lambda], "lr": LR},
        {"params": [model.log_wx, model.log_wy, model.log_wz], "lr": LR_W},
    ])
    start_epoch, loss_history, val_history, best = 1, [], [], float("inf")
    if resume:
        ck = torch.load(resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state"])
        if "optim_state" in ck: optim.load_state_dict(ck["optim_state"])
        start_epoch  = ck["epoch"] + 1
        loss_history = list(ck.get("loss_history", []))
        val_history  = list(ck.get("val_history", []))
        best         = ck.get("best_val", float("inf"))
        print(f"  Resumed from epoch {start_epoch} (best={best:.4e})")

    b_tr = [x.to(DEVICE) for x in b_tr]; z_tr = [x.to(DEVICE) for x in z_tr]
    b_va = [x.to(DEVICE) for x in b_va]; z_va = [x.to(DEVICE) for x in z_va]
    N = len(b_tr)

    REF_DIR  = os.path.join(OUT_DIR, "epoch_recon"); os.makedirs(REF_DIR, exist_ok=True)
    b_ref    = b_va[0].clone() if b_va else b_tr[0].clone()
    z_ref    = (z_va[0] if b_va else z_tr[0]).detach().cpu().numpy()
    b_ref_np = b_ref.detach().cpu().numpy()
    z_mf_ref   = ic.run_matched_filter(op, b_ref_np)
    z_ista_ref = ic.run_ista(op, b_ref_np, K, LAMBDA_INIT, L_EST)

    print(f"\n[W-LISTA Ken_grasso] K={K} epochs={start_epoch}->{N_EPOCHS} "
          f"N_train={N} N_val={len(b_va)} device={DEVICE}")
    print(f"  #params={model.num_params()}  lr={LR:.1e}  lr_w={LR_W:.1e}")

    for epoch in range(start_epoch, N_EPOCHS + 1):
        t0 = time.time(); model.train(); optim.zero_grad()
        agg = torch.zeros((), device=DEVICE)
        for idx in np.random.permutation(N):
            agg = agg + mag_mse(model, op, b_tr[idx], z_tr[idx])
        (agg / N).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        with torch.no_grad():
            model.log_wx.clamp_(-W_LOG_CLAMP, W_LOG_CLAMP)
            model.log_wy.clamp_(-W_LOG_CLAMP, W_LOG_CLAMP)
            model.log_wz.clamp_(-W_LOG_CLAMP, W_LOG_CLAMP)

        model.eval(); vloss = 0.0
        with torch.no_grad():
            for j in range(len(b_va)):
                vloss += float(mag_mse(model, op, b_va[j], z_va[j]))
        vloss /= max(len(b_va), 1)

        tr_loss = float(agg) / N
        loss_history.append(tr_loss); val_history.append(vloss)
        print(f"  Ep {epoch:3d}/{N_EPOCHS}  train={tr_loss:.4e}  val={vloss:.4e}  "
              f"t={time.time()-t0:.0f}s")

        if epoch >= REF_EPOCH_START:
            with torch.no_grad():
                z_snap = model(b_ref, op, warm_start=True).detach().cpu().numpy()
            ic.save_epoch_snapshot(REF_DIR, f"{ckpt_name}_val0", epoch,
                                   base.X_IMG, base.Y_IMG, base.Z_IMG,
                                   z_snap, z_true=z_ref,
                                   z_mf=z_mf_ref, z_ista=z_ista_ref)

        ckpt = dict(epoch=epoch, K=K, model_type="wlista", Nx=NX, Ny=NY, Nz=NZ,
                    model_state=model.state_dict(), optim_state=optim.state_dict(),
                    loss=tr_loss, val=vloss, best_val=best,
                    loss_history=loss_history, val_history=val_history,
                    train_idx=base.TRAIN_IDX, val_idx=base.VAL_IDX)
        torch.save(ckpt, os.path.join(CKPT_DIR, f"{ckpt_name}_ep{epoch:03d}.pt"))
        if vloss < best:
            best = vloss
            torch.save(ckpt, os.path.join(CKPT_DIR, f"{ckpt_name}_best.pt"))
            print(f"    [best] val={best:.4e}")

    torch.save(ckpt, os.path.join(CKPT_DIR, f"{ckpt_name}.pt"))
    print(f"\n  Best ckpt: {CKPT_DIR}/{ckpt_name}_best.pt  best={best:.4e}")
    return model, loss_history


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume",     default=None)
    ap.add_argument("--infer-only", default=None)
    args = ap.parse_args()
    ckpt_name = "wlista_ken_grasso"

    t_start = time.time()
    print("=" * 68)
    print(f"W-LISTA Ken_grasso_nowalls  device={DEVICE}")
    print(f"  Train idx {base.TRAIN_IDX}   Val idx {base.VAL_IDX}")
    print("=" * 68)

    b_tr_np, z_tr_np, k, omega, rx_ref = base.load_synthetic_dataset("wlista")
    op = build_operator_fast(rx_ref, k, omega)
    b_va_np, z_va_np = base.load_validation_data(rx_ref, k)

    if args.infer_only:
        ck    = torch.load(args.infer_only, map_location="cpu", weights_only=False)
        model = WLISTAHolography(K=int(ck["K"]), L_est=L_EST,
                                 Nx=int(ck["Nx"]), Ny=int(ck["Ny"]), Nz=int(ck["Nz"]),
                                 lambda_init=LAMBDA_INIT)
        model.load_state_dict(ck["model_state"]); model = model.cpu()
        loss_history = list(ck.get("loss_history", [ck["loss"]]))
        print(f"  Loaded epoch={ck['epoch']} val={ck.get('val', float('nan')):.4e}")
    else:
        b_tr = [torch.as_tensor(x) for x in b_tr_np]
        z_tr = [torch.as_tensor(x) for x in z_tr_np]
        b_va = [torch.as_tensor(x) for x in b_va_np]
        z_va = [torch.as_tensor(x) for x in z_va_np]
        model, loss_history = train(op, b_tr, z_tr, b_va, z_va, ckpt_name, resume=args.resume)
        model = model.cpu()

    train_labels = [base.POS_LABELS[i] for i in base.TRAIN_IDX]
    base.plot_results(b_tr_np, z_tr_np, train_labels, model, op, loss_history,
                      "wlista", ckpt_name, split_tag="train")
    val_labels = [base.POS_LABELS[i] for i in base.VAL_IDX]
    base.plot_results(b_va_np, z_va_np, val_labels, model, op, loss_history,
                      "wlista", ckpt_name, split_tag="val")

    print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f} min\nDone.")
