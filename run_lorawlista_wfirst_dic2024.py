# -*- coding: utf-8 -*-
"""
run_lorawlista_wfirst_dic2024.py
=====================================
LR-W-LISTA con strategia W-FIRST warmup.

Warmup (epoche 1-WARMUP_EPOCHS):
  - solo W + mu + lambda vengono addestrati  (= W-LISTA puro)
  - UV congelato (U=0, V=0 => T_eff = T, nessuna correzione low-rank)
  => il modello apprende prima i pesi W stabili (senza BETA_DATA che
     causa il collapse), poi introduce la correzione low-rank

Dopo warmup (epoca WARMUP_EPOCHS+1 in poi):
  - si aggiunge il gruppo UV all'ottimizzatore
  - tutti i parametri vengono addestrati insieme

Motivazione: W-LISTA puro (senza UV) non collassa perche' non ha BETA_DATA.
Stabilizzando prima W, evitiamo che W interagisca con UV in modo incontrollato
quando il data term e' gia' attivo.

Run
---
  .conda\\python.exe run_lorawlista_wfirst_dic2024.py --rank 8 > wfirst.log 2>&1
"""

import os, sys, time, argparse
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import scipy.io as sio
import torch
import cupy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))
import base_measurements_dic2024 as base2
from holography_operator_fast import HolographyOperatorFast
from lista_holography_lowrank import LRWLISTAHolography
import inference_common as ic

OUT_DIR  = os.path.join(base2.SCRIPT_DIR, "results_wlista_wfirst_dec2024")
CKPT_DIR = os.path.join(base2.SCRIPT_DIR, "checkpoints_lista_lowrank")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
base2.OUT_DIR  = OUT_DIR
base2.CKPT_DIR = CKPT_DIR

NX, NY, NZ = base2.NX, base2.NY, base2.NZ
K, N_EPOCHS = base2.K, base2.N_EPOCHS
LR, LR_W    = base2.LR, base2.LR_W
LAMBDA_INIT, L_EST = base2.LAMBDA_INIT, base2.L_EST

RANK          = 8
LR_LR         = 1e-5
WARMUP_EPOCHS = 6      # epoche con solo W+mu+lambda (= W-LISTA puro)
ALPHA_Z       = 1.0
BETA_DATA     = 1e-3
GAMMA_REG     = 1e-1
CKPT_NAME     = "wlista_lowrank_wfirst_dec2024"
VAL_HOLDOUT   = None
REF_DATES     = ["21_11_2024", "10_12_2024"]
REF_EPOCH_START = 2

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


def _signal_clutter(z_pred, z_true):
    occ = np.abs(z_true) > 0; mag = np.abs(z_pred)
    return float((mag[occ].mean() + 1e-30) / (mag[~occ].mean() + 1e-30))


def _mip_xy(z, shape):
    return np.abs(z).reshape(shape).max(axis=2)


def _to_db(arr):
    return 20.0 * np.log10(arr / (arr.max() + 1e-30) + 1e-30)


def _save_ref_snapshot(ref_dir, r, epoch, z_snap, x_img, y_img, z_img):
    import matplotlib.pyplot as plt, scipy.io as _sio
    shape  = (len(x_img), len(y_img), len(z_img))
    tag    = f"{r['prefix']}_ep{epoch:03d}"
    z_true = r["z_true"]; z_mf = r["z_mf"]; z_ista = r["z_ista"]

    sc_mf    = _signal_clutter(z_mf,   z_true)
    sc_ista  = _signal_clutter(z_ista, z_true)
    sc_model = _signal_clutter(z_snap, z_true)

    mip_gt    = _mip_xy(z_true, shape)
    mip_mf    = _mip_xy(z_mf,   shape)
    mip_ista  = _mip_xy(z_ista, shape)
    mip_model = _mip_xy(z_snap, shape)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, data, title, use_db in zip(
            axes,
            [mip_gt, mip_mf, mip_ista, mip_model],
            ["Ground truth", "MF", "ISTA", f"W-first ep{epoch}"],
            [False, True, True, True]):
        d = _to_db(data) if use_db else data
        vmin, vmax = (-30, 0) if use_db else (0, data.max()+1e-30)
        im = ax.pcolormesh(x_img, y_img, d.T, cmap="jet",
                           vmin=vmin, vmax=vmax, shading="nearest")
        plt.colorbar(im, ax=ax, label="dB" if use_db else "|z|", fraction=0.046)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    sc_str = f"S/C: MF={sc_mf:.2f}  ISTA={sc_ista:.2f}  model={sc_model:.2f}"
    fig.suptitle(f"{r['date']}  epoch {epoch}  —  {sc_str}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(ref_dir, tag + ".png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    np.savez(os.path.join(ref_dir, tag + ".npz"),
             z_model=z_snap.reshape(shape), z_mf=z_mf.reshape(shape),
             z_ista=z_ista.reshape(shape),  z_true=z_true.reshape(shape),
             mip_model=mip_model, mip_mf=mip_mf, mip_ista=mip_ista, mip_gt=mip_gt,
             x_img=x_img, y_img=y_img, z_img=z_img,
             epoch=np.int32(epoch), date=np.array(r["date"]),
             sc_mf=np.float32(sc_mf), sc_ista=np.float32(sc_ista),
             sc_model=np.float32(sc_model))

    _sio.savemat(os.path.join(ref_dir, tag + ".mat"), dict(
        z_model=z_snap.reshape(shape), z_mf=z_mf.reshape(shape),
        z_ista=z_ista.reshape(shape),  z_true=z_true.reshape(shape),
        mip_model=mip_model, mip_mf=mip_mf, mip_ista=mip_ista, mip_gt=mip_gt,
        x_img=x_img, y_img=y_img, z_img=z_img,
        epoch=float(epoch), date=r["date"],
        sc_mf=sc_mf, sc_ista=sc_ista, sc_model=sc_model),
        do_compression=True)

    csv_path = os.path.join(ref_dir, f"metrics_{r['date']}.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        import csv
        w = csv.writer(f)
        if write_header:
            w.writerow(["epoch", "sc_mf", "sc_ista", "sc_model"])
        w.writerow([epoch, f"{sc_mf:.4f}", f"{sc_ista:.4f}", f"{sc_model:.4f}"])


def loss_terms(model, op, b, z_true):
    z_pred = model(b, op, warm_start=True)
    loss_z = torch.mean((z_pred.abs() - z_true.abs()) ** 2)
    y_hat  = model.measure(z_pred, op)
    normb2 = torch.mean(b.abs() ** 2) + 1e-12
    loss_d = torch.mean((y_hat - b).abs() ** 2) / normb2
    reg    = model.lowrank_frob_sq()
    return ALPHA_Z*loss_z + BETA_DATA*loss_d + GAMMA_REG*reg, loss_z, loss_d, reg


def _build_optim_warmup(model):
    """Ottimizzatore fase warmup: solo W + mu + lambda (= W-LISTA puro). UV congelato."""
    return torch.optim.Adam([
        {"params": [model.log_mu, model.log_lambda], "lr": LR},
        {"params": [model.log_wx, model.log_wy, model.log_wz], "lr": LR_W},
    ])


def _build_optim_full(model):
    """Ottimizzatore fase full: aggiunge UV al gruppo gia' esistente."""
    return torch.optim.Adam([
        {"params": [model.log_mu, model.log_lambda], "lr": LR},
        {"params": [model.log_wx, model.log_wy, model.log_wz], "lr": LR_W},
        {"params": [model.U_re, model.U_im, model.V_re, model.V_im], "lr": LR_LR},
    ])


def train(op, b_list, z_list, resume=None):
    M = op.N_rx
    model = LRWLISTAHolography(K=K, L_est=L_EST, Nx=NX, Ny=NY, Nz=NZ,
                              M=M, rank=RANK, lambda_init=LAMBDA_INIT).to(DEVICE)

    in_warmup = True
    optim = _build_optim_warmup(model)

    start_epoch, loss_history, val_history, best = 1, [], [], float("inf")
    if resume is not None:
        ck = torch.load(resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state"])
        start_epoch  = ck["epoch"] + 1
        loss_history = list(ck.get("loss_history", []))
        val_history  = list(ck.get("val_history", []))
        best         = ck.get("best", float("inf"))
        if start_epoch > WARMUP_EPOCHS:
            in_warmup = False
            optim = _build_optim_full(model)
        if "optim_state" in ck:
            try:
                optim.load_state_dict(ck["optim_state"])
            except Exception:
                pass
        print(f"  Resumed from epoch {start_epoch} (best={best:.4e})")

    all_idx = list(range(len(b_list)))
    tr_idx  = [i for i in all_idx if i != VAL_HOLDOUT]
    va_idx  = [VAL_HOLDOUT] if VAL_HOLDOUT is not None else []

    b_t = [b_list[i].to(DEVICE) for i in range(len(b_list))]
    z_t = [z_list[i].to(DEVICE) for i in range(len(z_list))]

    REF_DIR = os.path.join(OUT_DIR, "epoch_recon"); os.makedirs(REF_DIR, exist_ok=True)
    refs = []
    for rd in REF_DATES:
        if rd not in base2.DATES:
            print(f"  [WARN] REF_DATE {rd} non trovato in DATES, skip")
            continue
        ri      = base2.DATES.index(rd)
        b_r     = b_t[ri]
        b_r_np  = b_r.detach().cpu().numpy()
        refs.append(dict(
            date   = rd,
            b      = b_r,
            b_np   = b_r_np,
            z_true = z_t[ri].detach().cpu().numpy(),
            prefix = f"{CKPT_NAME}_{rd}",
            z_mf   = ic.run_matched_filter(op, b_r_np),
            z_ista = ic.run_ista(op, b_r_np, K, LAMBDA_INIT, L_EST),
        ))
        print(f"  Reference snapshot: {rd}  (idx={ri})")

    print(f"\n[W-FIRST LR-W-LISTA] K={K} rank={RANK} epochs={start_epoch}->{N_EPOCHS} "
          f"warmup={WARMUP_EPOCHS} device={DEVICE}")
    print(f"  #params={model.num_params()}  ALPHA_Z={ALPHA_Z} BETA_DATA={BETA_DATA} "
          f"GAMMA_REG={GAMMA_REG}  refs={REF_DATES}\n")

    for epoch in range(start_epoch, N_EPOCHS + 1):
        # Transizione warmup -> full a WARMUP_EPOCHS+1
        if in_warmup and epoch > WARMUP_EPOCHS:
            print(f"\n  [W-first] Warmup completato a ep{epoch-1}. "
                  f"Aggiungo UV all'ottimizzatore (lr_LR={LR_LR:.1e}).\n")
            optim = _build_optim_full(model)
            in_warmup = False

        phase = "WARMUP(W)" if in_warmup else "FULL"

        t0 = time.time(); model.train(); optim.zero_grad()
        agg = torch.zeros((), device=DEVICE); lz = ld = lr_ = 0.0
        for idx in np.random.permutation(tr_idx):
            tot, a, bb, c = loss_terms(model, op, b_t[idx], z_t[idx])
            agg = agg + tot; lz += float(a); ld += float(bb); lr_ += float(c)
        (agg / len(tr_idx)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        with torch.no_grad():
            model.log_wx.clamp_(-base2.W_LOG_CLAMP, base2.W_LOG_CLAMP)
            model.log_wy.clamp_(-base2.W_LOG_CLAMP, base2.W_LOG_CLAMP)
            model.log_wz.clamp_(-base2.W_LOG_CLAMP, base2.W_LOG_CLAMP)

        tr_loss = float(agg) / len(tr_idx)
        vloss = float("nan"); vloss_z = float("nan")
        if va_idx:
            model.eval()
            with torch.no_grad():
                _vt, _vz, _, _ = loss_terms(model, op, b_t[va_idx[0]], z_t[va_idx[0]])
                vloss = float(_vt); vloss_z = float(_vz)
        sel = vloss if va_idx else tr_loss
        loss_history.append(tr_loss); val_history.append(vloss)
        st = model.lowrank_stats()
        print(f"  Ep {epoch:3d}/{N_EPOCHS} [{phase}]  train={tr_loss:.4e} "
              f"(z={lz/len(tr_idx):.3e} data={ld/len(tr_idx):.3e})  "
              f"val={vloss:.4e}  t={time.time()-t0:.0f}s  "
              f"|U|={st['U_fro']:.2e} |V|={st['V_fro']:.2e}")

        if epoch >= REF_EPOCH_START:
            model.eval()
            with torch.no_grad():
                for r in refs:
                    z_snap = model(r["b"], op, warm_start=True).detach().cpu().numpy()
                    _save_ref_snapshot(REF_DIR, r, epoch, z_snap,
                                       base2.X_IMG, base2.Y_IMG, base2.Z_IMG)

        ckpt = dict(epoch=epoch, K=K, Nx=NX, Ny=NY, Nz=NZ, M=M, rank=RANK,
                    model_state=model.state_dict(), optim_state=optim.state_dict(),
                    loss=tr_loss, val=vloss, best=best,
                    loss_history=loss_history, val_history=val_history,
                    train_dates=base2.DATES, warmup_epochs=WARMUP_EPOCHS)
        if sel < best:
            best = sel; ckpt["best"] = best
            torch.save(ckpt, os.path.join(CKPT_DIR, f"{CKPT_NAME}_best.pt"))
        torch.save(ckpt, os.path.join(CKPT_DIR, f"{CKPT_NAME}_ep{epoch:03d}.pt"))

    torch.save(ckpt, os.path.join(CKPT_DIR, f"{CKPT_NAME}.pt"))
    print(f"\n  Best ckpt: {CKPT_DIR}/{CKPT_NAME}_best.pt  best={best:.4e}")
    return model, loss_history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank",   type=int, default=RANK)
    ap.add_argument("--warmup", type=int, default=WARMUP_EPOCHS)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    RANK          = args.rank
    WARMUP_EPOCHS = args.warmup
    CKPT_NAME = f"wlista_lowrank_wfirst_dec2024_r{RANK}"
    base2.CKPT_NAME = CKPT_NAME

    t_start = time.time()
    print("=" * 70)
    print(f"LR-W-LISTA W-FIRST dec2024  rank={RANK}  warmup={WARMUP_EPOCHS}  device={DEVICE}")
    print("=" * 70)

    b_np, z_np, k, omega, ref = base2.load_dataset()
    op = build_operator_fast(ref, k, omega)

    b_list = [torch.tensor(b_np[i], dtype=torch.complex64) for i in range(len(b_np))]
    z_list = [torch.tensor(z_np[i], dtype=torch.complex64) for i in range(len(z_np))]

    train(op, b_list, z_list, resume=args.resume)
    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
