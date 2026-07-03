# W-LISTA / LoRaW-LISTA — RF Holographic Imaging

Reference code for the paper *"EM Informed Holographic Imaging via Unrolled Deep Networks,"* submitted to *IEEE Internet of Things Journal*. It implements
the unrolled reconstruction networks evaluated in the paper:

- **W-LISTA** — Weighted LISTA with a learnable, spatially-varying *ℓ₁*
  regularization (per-voxel sparsity prior).
- **LoRaW-LISTA** — W-LISTA plus a low-rank adaptation of the holographic
  operator `T`, trained with the two-stage *W-first* schedule
  (W-LISTA first, low-rank correction second).

The code covers the four cases reported in the paper: three synthetic full-wave
(FEKO) body models and one real indoor measurement campaign at 2.48 GHz.

> Code only. The measurement / simulation datasets are not included (see *Data*).
> The training entry points are the **GPU** scripts (CuPy + DLPack); shared code
> lives in `utils/`.

## Installation

```bash
pip install -r requirements.txt
```

`numpy`, `scipy`, `matplotlib` and `torch` are the core dependencies; `cupy` is
required by the GPU training scripts (install the CuPy wheel matching your CUDA
version, e.g. `cupy-cuda12x`).

## Data

The **synthetic** datasets are bundled in `synthetic_sets/`, so the three
synthetic cases run out of the box. The **real** indoor measurements are **not
yet public**: they will be released into `real_data/` at a later date, and until
then the real-measurement scripts stop with a clear "file not found" message.

| Case | Entry script | Required data files | Folder |
|------|--------------|---------------------|--------|
| PEC (nowalls) | `run_wlista_pec_nowalls_gpu.py` | `E_total_Ken_PEC_nowalls.mat`, `E_total_freespace_nowalls.mat` | `synthetic_sets/` |
| Adipose tissue | `run_wlista_adipose.py` | `E_total_Ken_grasso_nowalls.mat`, `E_total_freespace_nowalls.mat` | `synthetic_sets/` |
| Muscle tissue | `run_wlista_muscle.py` | `E_total_Ken_muscoloso.mat`, `E_inc.mat` | `synthetic_sets/` |
| Real measurements | `run_wlista_measurements_dic2024_gpu.py` | `empty_*.mat` (8 positions, Nov–Dec 2024) | `real_data/` *(released later)* |

For the real-measurement case, the ground-truth contrast volumes `z_true_*.npz`
are generated once with `utils/generate_z_true.py` (one per measurement position)
and cached in `results_z_true/`.

## Cases and how to run

The PEC and real cases expose `--model {lista,wlista}`; the adipose and muscle
cases run W-LISTA. Defaults: `K = 10` unrolled layers, `30` epochs. All scripts
require a CUDA GPU (CuPy).

```bash
# 1 — PEC (nowalls): Born strongly violated
python3 run_wlista_pec_nowalls_gpu.py --model wlista

# 2 — Adipose tissue (lossy dielectric body)
python3 run_wlista_adipose.py

# 3 — Muscle tissue (high-permittivity body)
python3 run_wlista_muscle.py

# 4 — Real indoor measurements (2.48 GHz), W-LISTA
python3 run_wlista_measurements_dic2024_gpu.py --model wlista

# 4b — LoRaW-LISTA: two-stage W-first (low-rank adaptation of T)
python3 run_lorawlista_wfirst_dic2024.py --rank 8
```

Training saves per-epoch checkpoints (`checkpoints_*/`) and reconstructions
(`results_*/`). Resume with `--resume <checkpoint.pt>`; inference only with
`--infer-only <checkpoint.pt>`.

### Inference from a trained checkpoint

Each training script can reconstruct from a saved checkpoint via `--infer-only`
(the model type — LISTA / W-LISTA / LoRaW-LISTA — is auto-detected from the file):

```bash
# PEC case, best W-LISTA checkpoint
python3 run_wlista_pec_nowalls_gpu.py \
        --infer-only checkpoints_lista/wlista_synthetic_nowalls_gpu_best.pt

# Adipose case
python3 run_wlista_adipose.py \
        --infer-only checkpoints_lista_ken_grasso/wlista_ken_grasso_best.pt
```

Each run writes the reconstruction (MIP-xy `.png` plus `.mat`/`.npz` volumes,
with matched-filter and ISTA baselines) to the case's `results_*/` folder.

## Repository layout

```
.
├── run_wlista_pec_nowalls_gpu.py         # PEC (nowalls) — LISTA / W-LISTA
├── run_wlista_adipose.py                 # adipose-tissue body — W-LISTA
├── run_wlista_muscle.py                  # muscle-tissue body — W-LISTA
├── run_wlista_measurements_dic2024_gpu.py# real measurements — LISTA / W-LISTA
├── run_lorawlista_wfirst_dic2024.py      # real measurements — LoRaW-LISTA (W-first)
├── requirements.txt
├── synthetic_sets/                       # bundled FEKO .mat datasets (3 synthetic cases)
├── real_data/                            # real measurements go here (released later)
└── utils/                                # shared library code (not run directly)
    ├── holography_operator.py            # forward operator T / Tᴴ (Green's function)
    ├── holography_operator_fast.py       # fast DLPack zero-copy operator
    ├── lista_holography.py               # LISTA network (K unrolled ISTA layers)
    ├── lista_holography_weighted.py      # W-LISTA (factorized per-voxel weights)
    ├── lista_holography_lowrank.py       # LoRaW-LISTA (T_eff = (I + U Vᴴ) T)
    ├── generate_z_true.py                # ground-truth permittivity-contrast volumes
    ├── inference_common.py               # shared inference / metrics / baselines (MF, ISTA)
    ├── base_pec_nowalls.py               # data loaders/constants for synthetic cases
    └── base_measurements_dic2024.py      # data loaders/constants for the real case
```
