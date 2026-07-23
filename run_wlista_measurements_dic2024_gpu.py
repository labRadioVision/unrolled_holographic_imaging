# -*- coding: utf-8 -*-
"""
run_wlista_measurements_dic2024_gpu.py
======================================
Full-GPU training of W-LISTA / LISTA on the 8 real measurements (Nov-Dec 2024).

  - Fast holographic operator (HolographyOperatorFast, DLPack zero-copy), with
    model and data on CUDA.
  - Magnitude-MSE loss; all measurements used for training, "best" on the train
    loss, with an optional hold-out via VAL_HOLDOUT.
  - From REF_EPOCH_START, saves per epoch the reference reconstruction
    (measurement 21_11_2024, centred) and the MIP of MF/ISTA/model (PNG/npz/mat).
  - Reuses the loaders/plotting of the base module (import base_measurements_dic2024 as base2).

Run
---
  python run_wlista_measurements_dic2024_gpu.py > wlista_dec2024_gpu.log 2>&1
  python run_wlista_measurements_dic2024_gpu.py --model lista
"""

import os, sys, time, argparse
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import torch
import cupy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))
import base_measurements_dic2024 as base2
from holography_operator_fast import HolographyOperatorFast
from lista_holography          import LISTAHolography
from lista_holography_weighted import WLISTAHolography
import inference_common as ic

OUT_DIR  = os.path.join(base2.SCRIPT_DIR, "results_wlista_dec2024_gpu")
CKPT_DIR = os.path.join(base2.SCRIPT_DIR, "checkpoints_lista")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
base2.OUT_DIR  = OUT_DIR
base2.CKPT_DIR = CKPT_DIR

NX, NY, NZ = base2.NX, base2.NY, base2.NZ
K, N_EPOCHS = base2.K, base2.N_EPOCHS
LR, LR_W    = base2.LR, base2.LR_W
LAMBDA_INIT, L_EST = base2.LAMBDA_INIT, base2.L_EST
W_LOG_CLAMP = base2.W_LOG_CLAMP

VAL_HOLDOUT = None        # e.g. 7 -> leave-one-out
REF_EPOCH_START = 5
REF_DATE = "21_11_2024"   # reference measurement (centred target)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_operator_fast(ref, k, omega):
    print("\nBuilding FAST operator (DLPack) ...")
    Nx_rx, Ny_rx = ref["S21"].shape[:2]
    x_rx = ref["X"][:, 0]; y_rx = ref["Y"][0, :]
    XX, YY = np.meshgrid(x_rx, y_rx, indexing="ij")
    r_rx = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(Nx_rx * Ny_rx)])
    XXv, YYv, ZZv = np.meshgrid(base2.X_IMG, base2.Y_IMG, base2.Z_IMG, indexing="ij")
    r_vox = np.column_stack([XXv.ravel(), YYv.ravel(), ZZv.ravel()])
    dV = (base2.X_IMG[1]-base2.X_IMG[0])*(base2.Y_IMG[1]-base2.Y_IMG[0])*(base2.Z_IMG[1]-base2.Z_IMG[0])
    print(f"  Receivers : {r_rx.shape[0]}  ({Nx_rx}x{Ny_rx})")
    print(f"  Voxels    : {r_vox.shape[0]}")
    return HolographyOperatorFast(cp.asarray(r_rx), cp.asarray(r_vox),
                                  k=k, omega=omega, mu0=base2.MU0, dV=dV,
                                  batch_rx=base2.BATCH_RX)


def build_model(model_type):
    if model_type == "lista":
        return LISTAHolography(K=K, L_est=L_EST, lambda_init=LAMBDA_INIT)
    return WLISTAHolography(K=K, L_est=L_EST, Nx=NX, Ny=NY, Nz=NZ,
                            lambda_init=LAMBDA_INIT)


def build_optimizer(model, model_type):
    if model_type == "lista":
        return torch.optim.Adam(model.parameters(), lr=LR)
    return torch.optim.Adam([
        {"params": [model.log_mu, model.log_lambda], "lr": LR},
        {"params": [model.log_wx, model.log_wy, model.log_wz], "lr": LR_W},
    ])


def mag_mse(model, op, b, z_true):
    z_pred = model(b, op, warm_start=True)
    return torch.mean((z_pred.abs() - z_true.abs()) ** 2)


def train(op, b_list, z_list, model_type, ckpt_name, resume=None):
    model = build_model(model_type).to(DEVICE)
    optim = build_optimizer(model, model_type)

    start_epoch, loss_history, val_history, best = 1, [], [], float("inf")
    if resume is not None:
        ck = torch.load(resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state"])
        if "optim_state" in ck: optim.load_state_dict(ck["optim_state"])
        start_epoch  = ck["epoch"] + 1
        loss_history = list(ck.get("loss_history", []))
        val_history  = list(ck.get("val_history", []))
        best         = ck.get("best", float("inf"))
        print(f"  Resumed from epoch {start_epoch} (best={best:.4e})")

    all_idx = list(range(len(b_list)))
    tr_idx  = [i for i in all_idx if i != VAL_HOLDOUT]
    va_idx  = [VAL_HOLDOUT] if VAL_HOLDOUT is not None else []

    b_t = [b_list[i].to(DEVICE) for i in range(len(b_list))]
    z_t = [z_list[i].to(DEVICE) for i in range(len(z_list))]

    # --- reference case for per-epoch monitoring (centred measurement) ---
    REF_DIR = os.path.join(OUT_DIR, "epoch_recon"); os.makedirs(REF_DIR, exist_ok=True)
    ridx   = base2.DATES.index(REF_DATE) if REF_DATE in base2.DATES else 0
    b_ref  = b_t[ridx]
    z_ref  = z_t[ridx].detach().cpu().numpy()
    ref_pf = f"{ckpt_name}_{REF_DATE}"
    b_ref_np   = b_ref.detach().cpu().numpy()
    z_mf_ref   = ic.run_matched_filter(op, b_ref_np)
    z_ista_ref = ic.run_ista(op, b_ref_np, K, LAMBDA_INIT, L_EST)

    print(f"\n[{model_type.upper()}-GPU reale] K={K} epochs={start_epoch}->{N_EPOCHS} "
          f"N_train={len(tr_idx)} val={va_idx} device={DEVICE}  ref={REF_DATE}")
    if model_type == "wlista":
        print(f"  #params={model.num_params()}  lr={LR:.1e} lr_w={LR_W:.1e}")

    for epoch in range(start_epoch, N_EPOCHS + 1):
        t0 = time.time(); model.train(); optim.zero_grad()
        agg = torch.zeros((), device=DEVICE)
        for idx in np.random.permutation(tr_idx):
            agg = agg + mag_mse(model, op, b_t[idx], z_t[idx])
        (agg / len(tr_idx)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        if model_type == "wlista":
            with torch.no_grad():
                model.log_wx.clamp_(-W_LOG_CLAMP, W_LOG_CLAMP)
                model.log_wy.clamp_(-W_LOG_CLAMP, W_LOG_CLAMP)
                model.log_wz.clamp_(-W_LOG_CLAMP, W_LOG_CLAMP)

        tr_loss = float(agg) / len(tr_idx)
        vloss = float("nan")
        if va_idx:
            model.eval()
            with torch.no_grad():
                vloss = float(mag_mse(model, op, b_t[va_idx[0]], z_t[va_idx[0]]))
        sel = vloss if va_idx else tr_loss
        loss_history.append(tr_loss); val_history.append(vloss)
        print(f"  Ep {epoch:3d}/{N_EPOCHS}  train={tr_loss:.4e}  val={vloss:.4e}  "
              f"t={time.time()-t0:.0f}s")

        if epoch >= REF_EPOCH_START:
            model.eval()
            with torch.no_grad():
                z_snap = model(b_ref, op, warm_start=True).detach().cpu().numpy()
            ic.save_epoch_snapshot(REF_DIR, ref_pf, epoch,
                                   base2.X_IMG, base2.Y_IMG, base2.Z_IMG,
                                   z_snap, z_true=z_ref,
                                   z_mf=z_mf_ref, z_ista=z_ista_ref)

        ckpt = dict(epoch=epoch, K=K, model_type=model_type, Nx=NX, Ny=NY, Nz=NZ,
                    model_state=model.state_dict(), optim_state=optim.state_dict(),
                    loss=tr_loss, val=vloss, best=best,
                    loss_history=loss_history, val_history=val_history,
                    train_dates=base2.DATES)
        if sel < best:
            best = sel; ckpt["best"] = best
            torch.save(ckpt, os.path.join(CKPT_DIR, f"{ckpt_name}_best.pt"))
        torch.save(ckpt, os.path.join(CKPT_DIR, f"{ckpt_name}_ep{epoch:03d}.pt"))

    torch.save(ckpt, os.path.join(CKPT_DIR, f"{ckpt_name}.pt"))
    print(f"\n  Best ckpt: {CKPT_DIR}/{ckpt_name}_best.pt  best={best:.4e}")
    return model, loss_history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["wlista", "lista"], default="wlista")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--infer-only", default=None)
    args = ap.parse_args()

    CKPT_NAME = f"{args.model}_dec2024_gpu"
    base2.CKPT_NAME = CKPT_NAME
    t_start = time.time()
    print("=" * 66)
    print(f"{args.model.upper()} dec2024 (REALE, GPU)  device={DEVICE}")
    print("=" * 66)

    b_np, z_np, k, omega, ref = base2.load_dataset()
    op = build_operator_fast(ref, k, omega)

    if args.infer_only:
        ck = torch.load(args.infer_only, map_location="cpu", weights_only=False)
        model = build_model(ck.get("model_type", args.model))
        model.load_state_dict(ck["model_state"]); model = model.cpu()
        loss_history = list(ck.get("loss_history", [ck["loss"]]))
        print(f"  Loaded epoch={ck['epoch']} loss={ck['loss']:.4e}")
    else:
        b_t = [torch.as_tensor(x) for x in b_np]
        z_t = [torch.as_tensor(x) for x in z_np]
        model, loss_history = train(op, b_t, z_t, args.model, CKPT_NAME, resume=args.resume)
        model = model.cpu()

    base2.plot_results(b_np, z_np, model, op, loss_history, ckpt_tag=CKPT_NAME)

    print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f} min\nDone.")
