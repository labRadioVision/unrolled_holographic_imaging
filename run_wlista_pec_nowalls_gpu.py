# -*- coding: utf-8 -*-
"""
run_wlista_pec_nowalls_gpu.py
=============================
Full-GPU training of W-LISTA / LISTA on the synthetic "nowalls" data (PEC,
free space): E_total_freespace_nowalls.mat + E_total_Ken_PEC_nowalls.mat.

  - Fast holographic operator (HolographyOperatorFast, DLPack zero-copy), with
    model and data on CUDA.
  - Split: all 11 positions used for training; position 7 for validation/output.
  - "best" checkpoint selected on the validation loss (position 7).
  - From REF_EPOCH_START, saves per epoch the reference reconstruction (pos 7)
    and the MIP of MF/ISTA/model (PNG/npz/mat in results_synthetic_nowalls_gpu/epoch_recon).
  - Reuses the loaders/plotting of the base module (import base_pec_nowalls as base).
    No real-measurement file is required: RX geometry and frequency are synthetic.

Run
---
  python run_wlista_pec_nowalls_gpu.py > wlista_synthetic_nowalls_gpu.log 2>&1
  python run_wlista_pec_nowalls_gpu.py --model lista
"""

import os, sys, time, argparse, glob, re
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import torch
import cupy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))
import base_pec_nowalls as base
from holography_operator_fast import HolographyOperatorFast
from lista_holography          import LISTAHolography
from lista_holography_weighted import WLISTAHolography
import inference_common as ic

OUT_DIR  = os.path.join(base.SCRIPT_DIR, "results_synthetic_nowalls_gpu")
CKPT_DIR = os.path.join(base.SCRIPT_DIR, "checkpoints_lista")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
base.OUT_DIR  = OUT_DIR
base.CKPT_DIR = CKPT_DIR

base.TRAIN_IDX = list(range(11))            # all 11 positions used for training
base.VAL_IDX   = [7]                         # position 7 for validation/plot

NX, NY, NZ = base.NX, base.NY, base.NZ
K           = base.K
N_EPOCHS    = base.N_EPOCHS
LR          = 1e-2     # reduced from base.LR=5e-2 (a higher LR was observed to collapse z)
LR_W        = 2.5e-1   # reduced proportionally from base.LR_W=5e-1
LAMBDA_INIT = base.LAMBDA_INIT
L_EST       = base.L_EST
W_LOG_CLAMP = base.W_LOG_CLAMP
REF_EPOCH_START = 2     # from this epoch on, save the per-epoch reference reconstruction (pos 7)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_operator_fast(rx_ref, k, omega):
    print("\nBuilding FAST operator (DLPack) ...")
    Nx_rx, Ny_rx = rx_ref["S21"].shape[:2]
    x_rx = rx_ref["X"][:, 0]; y_rx = rx_ref["Y"][0, :]
    XX, YY = np.meshgrid(x_rx, y_rx, indexing="ij")
    r_rx = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(Nx_rx * Ny_rx)])
    XXv, YYv, ZZv = np.meshgrid(base.X_IMG, base.Y_IMG, base.Z_IMG, indexing="ij")
    r_vox = np.column_stack([XXv.ravel(), YYv.ravel(), ZZv.ravel()])
    dV = (base.X_IMG[1]-base.X_IMG[0])*(base.Y_IMG[1]-base.Y_IMG[0])*(base.Z_IMG[1]-base.Z_IMG[0])
    print(f"  Receivers : {r_rx.shape[0]}  ({Nx_rx}x{Ny_rx})")
    print(f"  Voxels    : {r_vox.shape[0]}  ({NX}x{NY}x{NZ})")
    return HolographyOperatorFast(cp.asarray(r_rx), cp.asarray(r_vox),
                                  k=k, omega=omega, mu0=base.MU0, dV=dV,
                                  batch_rx=base.BATCH_RX)


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


def _signal_clutter(z_pred, z_true):
    occ = np.abs(z_true) > 0; mag = np.abs(z_pred)
    return float((mag[occ].mean() + 1e-30) / (mag[~occ].mean() + 1e-30))


def train(op, b_tr, z_tr, b_va, z_va, model_type, ckpt_name, resume=None):
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
        best         = ck.get("best_val", float("inf"))
        print(f"  Resumed from epoch {start_epoch} (best={best:.4e})")

    b_tr = [x.to(DEVICE) for x in b_tr]; z_tr = [x.to(DEVICE) for x in z_tr]
    b_va = [x.to(DEVICE) for x in b_va]; z_va = [x.to(DEVICE) for x in z_va]
    N = len(b_tr)

    # --- reference case for per-epoch monitoring (pos 7) ---
    REF_DIR = os.path.join(OUT_DIR, "epoch_recon"); os.makedirs(REF_DIR, exist_ok=True)
    ref_i  = base.VAL_IDX.index(7) if 7 in base.VAL_IDX else 0
    b_ref  = b_va[ref_i] if b_va else b_tr[0]
    z_ref  = (z_va[ref_i] if b_va else z_tr[0]).detach().cpu().numpy()
    ref_pf = f"{ckpt_name}_pos7"
    b_ref_np   = b_ref.detach().cpu().numpy()
    z_mf_ref   = ic.run_matched_filter(op, b_ref_np)
    z_ista_ref = ic.run_ista(op, b_ref_np, K, LAMBDA_INIT, L_EST)

    sc_mf   = _signal_clutter(z_mf_ref,   z_ref)
    sc_ista = _signal_clutter(z_ista_ref, z_ref)
    print(f"  Baseline S/C — MF={sc_mf:.3f}  ISTA={sc_ista:.3f}")

    import csv as _csv
    csv_path = os.path.join(REF_DIR, "metrics_pos7.csv")
    _write_csv_header = not os.path.exists(csv_path)

    print(f"\n[{model_type.upper()}-GPU] K={K} epochs={start_epoch}->{N_EPOCHS} "
          f"N_train={N} N_val={len(b_va)} device={DEVICE}")
    if model_type == "wlista":
        print(f"  #params={model.num_params()}  lr={LR:.1e} lr_w={LR_W:.1e}")

    for epoch in range(start_epoch, N_EPOCHS + 1):
        t0 = time.time(); model.train(); optim.zero_grad()
        agg = torch.zeros((), device=DEVICE)
        for idx in np.random.permutation(N):
            agg = agg + mag_mse(model, op, b_tr[idx], z_tr[idx])
        (agg / N).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        if model_type == "wlista":
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

        # snapshot + S/C for every epoch >= REF_EPOCH_START
        sc_model = float("nan")
        if epoch >= REF_EPOCH_START:
            model.eval()
            with torch.no_grad():
                z_snap = model(b_ref, op, warm_start=True).detach().cpu().numpy()
            sc_model = _signal_clutter(z_snap, z_ref)
            ic.save_epoch_snapshot(REF_DIR, ref_pf, epoch,
                                   base.X_IMG, base.Y_IMG, base.Z_IMG,
                                   z_snap, z_true=z_ref,
                                   z_mf=z_mf_ref, z_ista=z_ista_ref)
            with open(csv_path, "a", newline="") as _f:
                _w = _csv.writer(_f)
                if _write_csv_header:
                    _w.writerow(["epoch", "train_loss", "val_loss",
                                 "sc_mf", "sc_ista", "sc_model"])
                    _write_csv_header = False
                _w.writerow([epoch, f"{tr_loss:.6e}", f"{vloss:.6e}",
                             f"{sc_mf:.4f}", f"{sc_ista:.4f}", f"{sc_model:.4f}"])

        print(f"  Ep {epoch:3d}/{N_EPOCHS}  train={tr_loss:.4e}  val={vloss:.4e}  "
              f"S/C={sc_model:.3f}  t={time.time()-t0:.0f}s")

        ckpt = dict(epoch=epoch, K=K, model_type=model_type, Nx=NX, Ny=NY, Nz=NZ,
                    model_state=model.state_dict(), optim_state=optim.state_dict(),
                    loss=tr_loss, val=vloss, best=best,
                    loss_history=loss_history, val_history=val_history,
                    train_idx=base.TRAIN_IDX, val_idx=base.VAL_IDX)
        if vloss < best:
            best = vloss; ckpt["best_val"] = best
            torch.save(ckpt, os.path.join(CKPT_DIR, f"{ckpt_name}_best.pt"))
        torch.save(ckpt, os.path.join(CKPT_DIR, f"{ckpt_name}_ep{epoch:03d}.pt"))

    torch.save(ckpt, os.path.join(CKPT_DIR, f"{ckpt_name}.pt"))
    print(f"\n  Best (val) ckpt: {CKPT_DIR}/{ckpt_name}_best.pt  best={best:.4e}")
    return model, loss_history


def find_latest_ckpt(ckpt_dir, ckpt_name):
    """Return the highest-epoch per-epoch checkpoint (<ckpt_name>_epNNN.pt),
    excluding _best.pt. Returns the path, or None if none exist."""
    pat = os.path.join(ckpt_dir, f"{ckpt_name}_ep*.pt")
    best_ep, best_path = -1, None
    for p in glob.glob(pat):
        m = re.search(r"_ep(\d+)\.pt$", os.path.basename(p))
        if m and int(m.group(1)) > best_ep:
            best_ep, best_path = int(m.group(1)), p
    return best_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["wlista", "lista"], default="wlista")
    ap.add_argument("--resume", default=None,
                    help="explicit checkpoint path (overrides auto-resume)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing checkpoints and start from scratch")
    ap.add_argument("--infer-only", default=None)
    args = ap.parse_args()

    CKPT_NAME = f"{args.model}_synthetic_nowalls_gpu"

    # Auto-resume by default: on start, resume from the last saved epoch.
    # --resume <path> forces a specific checkpoint; --fresh forces training
    # from scratch.
    if args.fresh:
        print("[fresh] training from scratch (checkpoints ignored)")
        args.resume = None
    elif args.resume and args.resume != "auto":
        print(f"[resume] explicit checkpoint: {args.resume}")
    else:
        latest = find_latest_ckpt(CKPT_DIR, CKPT_NAME)
        if latest:
            print(f"[auto-resume] resuming from: {latest}")
            args.resume = latest
        else:
            print("[auto-resume] no checkpoint found - starting from scratch")
            args.resume = None
    t_start = time.time()
    print("=" * 68)
    print(f"{args.model.upper()} synthetic NOWALLS (GPU)  device={DEVICE}")
    print(f"  Train idx {base.TRAIN_IDX}   Val idx {base.VAL_IDX}")
    print("=" * 68)

    b_tr_np, z_tr_np, k, omega, rx_ref = base.load_synthetic_dataset(args.model)
    op = build_operator_fast(rx_ref, k, omega)
    b_va_np, z_va_np = base.load_validation_data(rx_ref, k)

    if args.infer_only:
        ck = torch.load(args.infer_only, map_location="cpu", weights_only=False)
        model = build_model(ck.get("model_type", args.model))
        model.load_state_dict(ck["model_state"]); model = model.cpu()
        loss_history = list(ck.get("loss_history", [ck["loss"]]))
        print(f"  Loaded epoch={ck['epoch']} val={ck.get('val', float('nan')):.4e}")
    else:
        b_tr = [torch.as_tensor(x) for x in b_tr_np]
        z_tr = [torch.as_tensor(x) for x in z_tr_np]
        b_va = [torch.as_tensor(x) for x in b_va_np]
        z_va = [torch.as_tensor(x) for x in z_va_np]
        model, loss_history = train(op, b_tr, z_tr, b_va, z_va,
                                    args.model, CKPT_NAME, resume=args.resume)
        model = model.cpu()

    train_labels = [base.POS_LABELS[i] for i in base.TRAIN_IDX]
    base.plot_results(b_tr_np, z_tr_np, train_labels, model, op, loss_history,
                      args.model, CKPT_NAME, split_tag="train")
    val_labels = [base.POS_LABELS[i] for i in base.VAL_IDX]
    base.plot_results(b_va_np, z_va_np, val_labels, model, op, loss_history,
                      args.model, CKPT_NAME, split_tag="val")

    print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f} min\nDone.")
