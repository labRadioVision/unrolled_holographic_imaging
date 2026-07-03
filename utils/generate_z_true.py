# -*- coding: utf-8 -*-
"""
generate_z_true.py
==================
Generate synthetic ground-truth permittivity contrast vectors z_true
for the holographic imaging voxel grid.

Two shape modes are supported:

  box   -- axis-aligned rectangular box (original, fast)
  body  -- parametric human-body model (torso + legs + head, ellipsoidal
            cross-sections); parameters optionally derived from a CADFEKO
            .cfx file.  Falls back to typical mannequin anthropometry if
            the .cfx cannot be read.
  stl   -- load a triangular mesh from a .stl/.obj file (requires trimesh).

Coordinate system (holographic_imaging.py):
    x : horizontal along the scan wall      (X_IMG)
    y : vertical, y=0 at floor             (Y_IMG)
    z : depth into the room, z=0 at array  (Z_IMG)

Default voxel grid (kept in sync with holographic_imaging.py z26 run):
    X_IMG = linspace(0.0, 5.0, 161)   dx = 31.25 mm
    Y_IMG = linspace(0.0, 2.5,  81)   dy = 31.25 mm
    Z_IMG = linspace(0.3, 2.3,  65)   dz = 31.25 mm

Usage
-----
# Box (original)
python generate_z_true.py --preset single --plot

# Parametric body model at CFX-derived position
python generate_z_true.py --mode body --cfx path/to/model.cfx --delta_z_re 1.5 --label body_cfx --plot

# Parametric body at explicit position
python generate_z_true.py --mode body --center_x 2.5 --center_z 1.72 --label body_manual --plot

# Body from STL mesh
python generate_z_true.py --mode stl --stl path/to/mannequin.stl --center_x 2.5 --center_z 1.72 --label stl_body --plot
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===========================================================================
# Voxel grid  (keep in sync with holographic_imaging*.py)
# ===========================================================================

X_IMG = np.linspace(0.0, 5.0, 161)
Y_IMG = np.linspace(0.0, 2.5,  81)
Z_IMG = np.linspace(0.3, 2.3,  65)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "results_z_true")
os.makedirs(OUT_DIR, exist_ok=True)


# ===========================================================================
# CFX parameter extraction
# ===========================================================================

def extract_cfx_params(cfx_path: str) -> dict:
    """
    Read a CADFEKO .cfx file and return the mannequin geometric parameters
    in holographic coordinates.

    The .cfx file is an HDF5 container with a zlib-compressed Parasolid
    B-REP blob.  The blob does NOT contain a tessellated mesh (triangles),
    only analytical NURBS surface data.  However, B-REP control points in
    the depth range of the mannequin allow us to recover the centre position
    and approximate shoulder width / body depth.

    Coordinate mapping (user-confirmed):
        FEKO_x  <->  pptx_z  =>  holo_z = 3.317 - FEKO_x
        FEKO_y  <->  pptx_x  =>  holo_x = FEKO_y + 2.5
        FEKO_z  <->  pptx_y  =>  holo_y = FEKO_z + 1.435
        origin  :  centre of TX wall (holo = [2.5, 1.435, 3.317])

    Returns
    -------
    dict with keys:
        center_x, center_z   : mannequin horizontal centre [m]
        shoulder_half_width  : half-width in x  [m]
        body_half_depth      : half-depth in z  [m]
        foot_height          : floor level [m] (always 0)
        body_height          : total body height [m]
    """
    try:
        import h5py, zlib
    except ImportError:
        print("[CFX] h5py not available — using default parameters")
        return _default_body_params()

    try:
        with h5py.File(cfx_path, "r") as h:
            raw = bytes(h["Model"][:])
        dec = zlib.decompress(raw[16:])
    except Exception as e:
        print(f"[CFX] Cannot read {cfx_path}: {e}")
        return _default_body_params()

    n = len(dec) // 8 * 8
    v  = np.frombuffer(dec[:n], dtype="<f8")
    ok = np.isfinite(v)

    z_wall, x0, y0 = 3.317, 2.5, 1.435

    # --- Extract B-REP control points in the mannequin depth range ---
    # Mannequin nominal depth from filename: FEKO_x=1.6 => holo_z=1.717 m
    # Front face approx FEKO_x~1.3, back face FEKO_x~1.9 (body 60cm depth)
    pts = []
    for i in range(len(v) - 2):
        if not (ok[i] and ok[i+1] and ok[i+2]):
            continue
        fx, fy, fz = float(v[i]), float(v[i+1]), float(v[i+2])
        if (1.2 <= fx <= 2.0 and -0.5 <= fy <= 0.5 and -1.5 <= fz <= 0.8):
            pts.append([fx, fy, fz])

    # Also scan for symmetric shoulder feature points specifically
    # (confirmed: FEKO_x=2.5, FEKO_y=±0.259, FEKO_z=0.4 in the original analysis)
    shoulder_pts = []
    for i in range(len(v) - 2):
        if not (ok[i] and ok[i+1] and ok[i+2]):
            continue
        fx, fy, fz = float(v[i]), float(v[i+1]), float(v[i+2])
        if (0.5 <= fx <= 3.2 and 0.20 <= abs(fy) <= 0.35 and -0.5 <= fz <= 0.7):
            shoulder_pts.append([fx, fy, fz])

    params = _default_body_params()

    if shoulder_pts:
        sp = np.array(shoulder_pts)
        # shoulder half-width = max |FEKO_y|
        params["shoulder_half_width"] = float(np.abs(sp[:, 1]).max())
        # body half-depth from front/back spread in FEKO_x
        depth_span = sp[:, 0].max() - sp[:, 0].min()
        if depth_span > 0.05:
            params["body_half_depth"] = float(depth_span / 2.0)
        # body center from the FEKO_x centroid of shoulder points
        fx_center = sp[:, 0].mean()
        params["center_z"] = float(z_wall - fx_center)
        params["center_x"] = float(x0)   # FEKO_y=0 at array centre
        print(f"[CFX] shoulder_half_width={params['shoulder_half_width']:.3f} m  "
              f"body_half_depth={params['body_half_depth']:.3f} m")

    if pts:
        p = np.array(pts)
        # Refine center from B-REP points in mannequin range
        params["center_z"] = float(z_wall - p[:, 0].mean())
        print(f"[CFX] Mannequin centre: holo_x={params['center_x']:.3f}  "
              f"holo_z={params['center_z']:.3f} m")

    return params


def _default_body_params() -> dict:
    """Typical adult mannequin standing facing the array, from CFX analysis + PPT specs.
    Mannequin height = 1.89 m, width = 0.51 m (slide 5, geometry_info.pptx).
    Starting position: FEKO_x=1.6 → holo_z=1.717 m, centre of array holo_x=2.5 m.
    """
    return dict(
        center_x           = 2.500,   # m  horizontal centre of array
        center_z           = 1.717,   # m  depth from array (CFX: FEKO_x=1.6)
        shoulder_half_width= 0.259,   # m  (from CFX ±0.259; PPT width=0.51m → hw=0.255m)
        body_half_depth    = 0.145,   # m  front-to-back half (CFX vertex range 39cm)
        foot_height        = 0.000,   # m  floor at y=0
        body_height        = 1.890,   # m  total height (geometry_info.pptx slide 5)
    )


# ===========================================================================
# Parametric body model
# ===========================================================================

def make_z_true_body_model(
        center_x: float,
        center_z: float,
        delta_z: complex,
        shoulder_half_width: float = 0.259,
        body_half_depth:    float = 0.145,
        foot_height:        float = 0.0,
        body_height:        float = 1.78,
        X: np.ndarray = None,
        Y: np.ndarray = None,
        Z: np.ndarray = None,
) -> np.ndarray:
    """
    Build z_true using a parametric human-body shape model.

    The body is modelled as the UNION of:
      - Two leg cylinders (elliptic cross-section, below 55% height)
      - Torso+shoulders (elliptic cross-section varying with height, 30%-88%)
      - Neck (narrow elliptic cylinder, 88%-97%)
      - Head (sphere, centred at 94% height)

    All cross-section dimensions are derived from the CFX-extracted parameters
    (shoulder_half_width, body_half_depth) scaled by standard anthropometric
    ratios of a standing adult.

    Parameters
    ----------
    center_x, center_z : float
        Horizontal and depth centre of the body [m].
    delta_z : complex
        Permittivity contrast for occupied voxels.
    shoulder_half_width : float
        Half-width at the shoulders in x [m].  Default from CFX analysis.
    body_half_depth : float
        Half-depth (front-to-back) of the chest [m].  Default from CFX.
    foot_height : float
        y-coordinate of the floor (feet level) [m].
    body_height : float
        Total body height [m].
    X, Y, Z : np.ndarray, optional
        Override the module-level X_IMG / Y_IMG / Z_IMG grids.

    Returns
    -------
    z_true : np.ndarray, shape (Nx*Ny*Nz,), dtype complex64
        Flattened contrast volume, C-order from (Nx, Ny, Nz) with indexing='ij'.
    """
    if X is None: X = X_IMG
    if Y is None: Y = Y_IMG
    if Z is None: Z = Z_IMG

    Nx, Ny, Nz = len(X), len(Y), len(Z)
    H  = body_height
    yf = foot_height

    # Broadcast-friendly 3-D grids (indexing='ij')
    gx = X[:, None, None]   # (Nx, 1,  1)
    gy = Y[None, :, None]   # (1,  Ny, 1)
    gz = Z[None, None, :]   # (1,  1,  Nz)

    dx = gx - center_x        # horizontal offset from body centre
    dz = gz - center_z        # depth offset from body centre
    h  = gy - yf              # height above floor  (Ny dim, broadcast ok)

    # Normalised height (0=feet, 1=head top)
    t_norm = np.clip(h / H, 0.0, 1.0)   # (1, Ny, 1)

    mask = np.zeros((Nx, Ny, Nz), dtype=bool)

    # ------------------------------------------------------------------ #
    # 1.  LEGS  (0 – 55% height) — two elliptic cylinders                #
    #     offset ±leg_dx from the body centre.                            #
    #     leg_dx=50% shw, rx=28% shw → gap ≈ 3 voxels in MIP xy         #
    # ------------------------------------------------------------------ #
    leg_dx = 0.50 * shoulder_half_width   # ~13 cm offset per leg
    rx_leg = 0.28 * shoulder_half_width   # ~7 cm per leg (thinner → visible gap)
    rz_leg = 0.90 * body_half_depth       # front-to-back

    in_height_legs = (h >= 0) & (h <= 0.55 * H)
    for sign in (-1.0, +1.0):
        dx_leg = dx - sign * leg_dx
        in_ellipse = (dx_leg**2 / rx_leg**2 + dz**2 / rz_leg**2) <= 1.0
        mask |= in_ellipse & in_height_legs

    # ------------------------------------------------------------------ #
    # 2.  TORSO + SHOULDERS  (30% – 88% height)                          #
    #     Elliptic cross-section widening from waist to shoulders         #
    # ------------------------------------------------------------------ #
    in_height_torso = (h >= 0.30 * H) & (h <= 0.88 * H)
    # t_torso = 0 at waist (30%), 1 at shoulders (88%)
    t_torso = np.clip((t_norm - 0.30) / 0.58, 0.0, 1.0)

    rx_waist    = 0.67 * shoulder_half_width    # waist narrower
    rx_shoulder = 1.00 * shoulder_half_width    # full shoulder width
    rx_torso    = rx_waist + (rx_shoulder - rx_waist) * t_torso   # (1,Ny,1)

    rz_waist    = 0.75 * body_half_depth
    rz_shoulder = 1.00 * body_half_depth
    rz_torso    = rz_waist + (rz_shoulder - rz_waist) * t_torso   # (1,Ny,1)

    in_ellipse_torso = (dx**2 / rx_torso**2 + dz**2 / rz_torso**2) <= 1.0
    mask |= in_ellipse_torso & in_height_torso

    # ------------------------------------------------------------------ #
    # 3.  ARMS  (35% – 85% height) — two elliptic cylinders just         #
    #     outside the shoulder edge; arm centre at 120% shoulder_hw.     #
    #     Visible in MIP xy as lateral extensions beside the torso.      #
    # ------------------------------------------------------------------ #
    arm_cx = 1.20 * shoulder_half_width   # arm centre: 20% beyond shoulder edge
    rx_arm = 0.25 * shoulder_half_width   # arm half-width in x  (~6.5 cm)
    rz_arm = 0.70 * body_half_depth       # arm half-depth (< torso)
    in_height_arm = (h >= 0.35 * H) & (h <= 0.85 * H)
    for sign in (-1.0, +1.0):
        dx_arm = dx - sign * arm_cx
        in_ellipse_arm = (dx_arm**2 / rx_arm**2 + dz**2 / rz_arm**2) <= 1.0
        mask |= in_ellipse_arm & in_height_arm

    # ------------------------------------------------------------------ #
    # 4.  NECK  (88% – 97% height) — narrow elliptic cylinder            #
    # ------------------------------------------------------------------ #
    rx_neck = 0.25 * shoulder_half_width
    rz_neck = 0.30 * body_half_depth
    in_height_neck = (h >= 0.88 * H) & (h <= 0.97 * H)
    in_neck = (dx**2 / rx_neck**2 + dz**2 / rz_neck**2) <= 1.0
    mask |= in_neck & in_height_neck

    # ------------------------------------------------------------------ #
    # 5.  HEAD  (sphere centred at 94% height)                           #
    # ------------------------------------------------------------------ #
    head_cy = yf + 0.94 * H
    head_r  = 0.55 * shoulder_half_width   # ~14 cm radius
    in_head = (dx**2 + (gy - head_cy)**2 + dz**2) <= head_r**2
    mask |= in_head

    n_occupied = int(mask.sum())
    print(f"  Body model: center=({center_x:.3f}, {center_z:.3f}) m  "
          f"shoulder_hw={shoulder_half_width:.3f} m  "
          f"depth_hw={body_half_depth:.3f} m  "
          f"height={body_height:.3f} m")
    print(f"  Occupied voxels: {n_occupied} / {Nx*Ny*Nz}  "
          f"({100*n_occupied/(Nx*Ny*Nz):.2f}%)")

    z_vol = np.zeros((Nx, Ny, Nz), dtype=np.complex64)
    z_vol[mask] = complex(delta_z)
    return z_vol.ravel()


# ===========================================================================
# STL / OBJ mesh loading (optional, requires trimesh)
# ===========================================================================

def make_z_true_from_stl(
        stl_path: str,
        center_x: float,
        center_z: float,
        delta_z: complex,
        foot_height: float = 0.0,
        body_height_override: float = None,
        X: np.ndarray = None,
        Y: np.ndarray = None,
        Z: np.ndarray = None,
) -> np.ndarray:
    """
    Load a triangular surface mesh (STL/OBJ) and test each voxel barycentre
    for containment using a ray-casting inside/outside test (trimesh).

    The mesh is assumed to be in FEKO local model coordinates; it is
    automatically translated and (optionally) rescaled to place the body at
    (center_x, foot_height, center_z) in holographic coordinates.

    Requires: ``pip install trimesh``

    Parameters
    ----------
    stl_path : str
        Path to the STL or OBJ file.
    center_x, center_z : float
        Target horizontal and depth position [m].
    delta_z : complex
        Permittivity contrast for occupied voxels.
    foot_height : float
        y-coordinate of the floor [m].
    body_height_override : float, optional
        If provided, the mesh is scaled uniformly so its total y-extent
        equals this value.
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError(
            "trimesh is required for STL loading.\n"
            "Install with:  pip install trimesh\n"
            "Alternatively use --mode body for the parametric model.")

    if X is None: X = X_IMG
    if Y is None: Y = Y_IMG
    if Z is None: Z = Z_IMG

    mesh = trimesh.load(stl_path, force="mesh")
    if not mesh.is_watertight:
        print(f"[STL] Warning: mesh is NOT watertight — inside-test may be unreliable.")

    # --- Align mesh to holographic coordinate system ---
    # Assume the STL uses FEKO coords (x=depth, y=horizontal, z=height)
    # Swap axes to (holo_x=y, holo_y=z, holo_z=x) then apply offset
    # holo_x = FEKO_y + 2.5
    # holo_y = FEKO_z + 1.435
    # holo_z = 3.317 - FEKO_x

    # Reorder: (FEKO_x, FEKO_y, FEKO_z) -> (FEKO_y, FEKO_z, FEKO_x)
    verts = mesh.vertices.copy()
    verts_holo = np.column_stack([
        verts[:, 1] + 2.5,          # holo_x
        verts[:, 2] + 1.435,        # holo_y
        3.317 - verts[:, 0]         # holo_z
    ])

    # Compute current foot level and top
    y_min = verts_holo[:, 1].min()
    y_max = verts_holo[:, 1].max()
    h_mesh = y_max - y_min

    # Translate so feet are at foot_height
    verts_holo[:, 1] += foot_height - y_min

    # Optionally rescale to body_height_override
    if body_height_override is not None and h_mesh > 0:
        scale = body_height_override / h_mesh
        cy = foot_height   # scale around foot level
        verts_holo[:, 1] = cy + (verts_holo[:, 1] - cy) * scale

    # Translate in x and z to desired centre
    cx_mesh = (verts_holo[:, 0].min() + verts_holo[:, 0].max()) / 2.0
    cz_mesh = (verts_holo[:, 2].min() + verts_holo[:, 2].max()) / 2.0
    verts_holo[:, 0] += center_x - cx_mesh
    verts_holo[:, 2] += center_z - cz_mesh

    # Rebuild mesh with transformed vertices
    mesh_holo = trimesh.Trimesh(vertices=verts_holo, faces=mesh.faces,
                                process=False)

    print(f"[STL] Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
          f"watertight={mesh.is_watertight}")
    print(f"[STL] Holo bounding box:")
    bb = mesh_holo.bounds
    print(f"  x: {bb[0,0]:.3f}..{bb[1,0]:.3f}  "
          f"y: {bb[0,1]:.3f}..{bb[1,1]:.3f}  "
          f"z: {bb[0,2]:.3f}..{bb[1,2]:.3f}")

    Nx, Ny, Nz = len(X), len(Y), len(Z)
    gx, gy, gz = np.meshgrid(X, Y, Z, indexing="ij")
    vox_pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])   # (N,3)

    # Ray-casting containment test
    inside = mesh_holo.contains(vox_pts)   # (N,) bool

    n_occupied = int(inside.sum())
    print(f"[STL] Occupied voxels: {n_occupied} / {Nx*Ny*Nz}  "
          f"({100*n_occupied/(Nx*Ny*Nz):.2f}%)")

    z_vol = np.zeros(Nx * Ny * Nz, dtype=np.complex64)
    z_vol[inside] = complex(delta_z)
    return z_vol


# ===========================================================================
# Original box-based function (kept for backward compatibility)
# ===========================================================================

def make_z_true(mannequins: list, delta_z: complex,
                X: np.ndarray = None, Y: np.ndarray = None,
                Z: np.ndarray = None) -> np.ndarray:
    """
    Build z_true using axis-aligned rectangular boxes (original method).

    Parameters
    ----------
    mannequins : list of dicts with keys:
        'barycenter' : (x_c, y_c, z_c)  [m]  centre of the box
        'size'       : (sx, sy, sz)      [m]  full extents (width, height, depth)
    delta_z : complex
        Permittivity contrast value for occupied voxels.
    """
    if X is None: X = X_IMG
    if Y is None: Y = Y_IMG
    if Z is None: Z = Z_IMG

    Nx, Ny, Nz = len(X), len(Y), len(Z)
    mask = np.zeros((Nx, Ny, Nz), dtype=bool)

    for m in mannequins:
        x_c, y_c, z_c = m['barycenter']
        sx,  sy,  sz  = m['size']
        hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0

        ix = (X >= x_c - hx) & (X <= x_c + hx)
        iy = (Y >= y_c - hy) & (Y <= y_c + hy)
        iz = (Z >= z_c - hz) & (Z <= z_c + hz)

        box = ix[:, None, None] & iy[None, :, None] & iz[None, None, :]
        mask |= box
        print(f"  Box at ({x_c:.2f},{y_c:.2f},{z_c:.2f}) m  "
              f"size ({sx:.2f}x{sy:.2f}x{sz:.2f}) m  -> {int(box.sum())} voxels")

    n_total = int(mask.sum())
    print(f"  Total: {n_total} / {Nx*Ny*Nz}  ({100*n_total/(Nx*Ny*Nz):.2f}%)")
    z_vol = np.zeros((Nx, Ny, Nz), dtype=np.complex64)
    z_vol[mask] = complex(delta_z)
    return z_vol.ravel()


# ===========================================================================
# Visualisation
# ===========================================================================

def plot_z_true(z_true: np.ndarray, label: str,
                mannequins: list = None,
                X: np.ndarray = None,
                Y: np.ndarray = None,
                Z: np.ndarray = None):
    """Save a 3-panel MIP figure (xy, xz, yz projections) of |z_true|."""
    if X is None: X = X_IMG
    if Y is None: Y = Y_IMG
    if Z is None: Z = Z_IMG

    Nx, Ny, Nz = len(X), len(Y), len(Z)
    z_3d = np.abs(z_true).reshape(Nx, Ny, Nz)

    xy = z_3d.max(axis=2)   # (Nx, Ny)
    xz = z_3d.max(axis=1)   # (Nx, Nz)
    yz = z_3d.max(axis=0)   # (Ny, Nz)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].pcolormesh(X, Y, xy.T,  cmap='Reds', shading='nearest')
    axes[0].set_xlabel('x  (m)'); axes[0].set_ylabel('y  (m)')
    axes[0].set_title('MIP xy  (front view — project over depth z)')
    axes[0].set_aspect('equal')

    axes[1].pcolormesh(X, Z, xz.T,  cmap='Reds', shading='nearest')
    axes[1].set_xlabel('x  (m)'); axes[1].set_ylabel('z  (m)')
    axes[1].set_title('MIP xz  (floor plan — project over height y)')
    axes[1].set_aspect('equal')

    axes[2].pcolormesh(Z, Y, yz,    cmap='Reds', shading='nearest')
    axes[2].set_xlabel('z  (m)'); axes[2].set_ylabel('y  (m)')
    axes[2].set_title('MIP yz  (side view — project over horizontal x)')
    axes[2].set_aspect('equal')

    if mannequins:
        for m in mannequins:
            x_c, y_c, z_c = m['barycenter']
            sx,  sy,  sz  = m['size']
            axes[0].add_patch(plt.Rectangle(
                (x_c - sx/2, y_c - sy/2), sx, sy,
                lw=1.5, edgecolor='blue', facecolor='none', ls='--'))

    fig.suptitle(f'Ground-truth |z_true|  —  {label}', fontsize=11)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, f'z_true_{label}.png')
    fig.savefig(out_png, dpi=150)
    plt.close('all')
    print(f"  Saved {out_png}")


# ===========================================================================
# Save / load helpers
# ===========================================================================

def save_z_true(z_true: np.ndarray, label: str,
                delta_z: complex, **meta):
    out_npz = os.path.join(OUT_DIR, f'z_true_{label}.npz')
    np.savez(out_npz,
             z_true     = z_true,
             X_IMG      = X_IMG,
             Y_IMG      = Y_IMG,
             Z_IMG      = Z_IMG,
             delta_z_re = np.float32(delta_z.real),
             delta_z_im = np.float32(delta_z.imag),
             label      = label,
             **{k: str(v) for k, v in meta.items()})
    print(f"  Saved {out_npz}")
    return out_npz


def load_z_true(npz_path: str) -> np.ndarray:
    """Load z_true from a previously saved .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    return data['z_true']


# ===========================================================================
# Predefined scene presets
# ===========================================================================

PRESETS = {
    'single': [
        dict(barycenter=(2.5,  0.60, 1.20),
             size      =(0.45, 1.20, 0.25))
    ],
    'double': [
        dict(barycenter=(1.60, 0.60, 1.00),
             size      =(0.45, 1.20, 0.25)),
        dict(barycenter=(3.40, 0.60, 1.50),
             size      =(0.45, 1.20, 0.25)),
    ],
}


# ===========================================================================
# Command-line interface
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate ground-truth permittivity contrast z_true",
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('--mode', choices=['box', 'body', 'stl'], default='box',
                   help='Shape model:\n'
                        '  box  — axis-aligned box (original, default)\n'
                        '  body — parametric body (torso+legs+head ellipsoids)\n'
                        '  stl  — load surface from STL/OBJ file (needs trimesh)')

    # --- box mode ---
    p.add_argument('--preset', choices=list(PRESETS.keys()), default=None,
                   help='Predefined box scene (single / double mannequin)')
    p.add_argument('--barycenter', type=float, nargs='+', metavar='X Y Z',
                   help='Box barycentre(s) in metres')
    p.add_argument('--size', type=float, nargs='+', metavar='SX SY SZ',
                   help='Box full extents in metres (must match --barycenter count)')

    # --- body / stl modes ---
    p.add_argument('--cfx', type=str, default=None,
                   help='Path to .cfx file (body mode: extract centre + dimensions)')
    p.add_argument('--center_x', type=float, default=None,
                   help='Body horizontal centre x [m]  (overrides CFX extraction)')
    p.add_argument('--center_z', type=float, default=None,
                   help='Body depth centre z [m]  (overrides CFX extraction)')
    p.add_argument('--shoulder_hw', type=float, default=None,
                   help='Shoulder half-width [m]  (default from CFX: 0.259)')
    p.add_argument('--body_hd', type=float, default=None,
                   help='Body half-depth [m]  (default from CFX: 0.145)')
    p.add_argument('--body_height', type=float, default=1.78,
                   help='Total body height [m]  (default 1.78)')
    p.add_argument('--foot_height', type=float, default=0.0,
                   help='Floor y-coordinate [m]  (default 0.0)')

    # --- stl mode only ---
    p.add_argument('--stl', type=str, default=None,
                   help='Path to STL/OBJ mesh file  (stl mode)')

    # --- shared ---
    p.add_argument('--delta_z_re', type=float, default=1.5,
                   help='Real part of permittivity contrast  (default 1.5)')
    p.add_argument('--delta_z_im', type=float, default=0.3,
                   help='Imaginary part of permittivity contrast  (default 0.3)')
    p.add_argument('--label', type=str, default='scene',
                   help='Output file label')
    p.add_argument('--plot',  action='store_true',
                   help='Save MIP figure for visual verification')
    return p.parse_args()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == '__main__':
    args  = parse_args()
    dz    = complex(args.delta_z_re, args.delta_z_im)

    print(f"\nGenerating z_true  [mode={args.mode}]  label={args.label}")
    print(f"  delta_z = {dz.real:.3f} + {dz.imag:.3f}j")
    print(f"  Grid    : {len(X_IMG)} x {len(Y_IMG)} x {len(Z_IMG)}"
          f" = {len(X_IMG)*len(Y_IMG)*len(Z_IMG)} voxels\n")

    if args.mode == 'box':
        # ---- box mode ----
        if args.preset is not None:
            mannequins = PRESETS[args.preset]
        else:
            if args.barycenter is None or args.size is None:
                raise ValueError("box mode: provide --preset or --barycenter + --size")
            bary, size = args.barycenter, args.size
            if len(bary) % 3 or len(size) % 3 or len(bary) != len(size):
                raise ValueError("--barycenter and --size must be equal multiples of 3")
            n = len(bary) // 3
            mannequins = [dict(barycenter=(bary[3*i], bary[3*i+1], bary[3*i+2]),
                               size      =(size[3*i], size[3*i+1], size[3*i+2]))
                          for i in range(n)]
        z_true = make_z_true(mannequins, dz)
        plot_mannequins = mannequins

    elif args.mode == 'body':
        # ---- parametric body mode ----
        params = _default_body_params()

        if args.cfx is not None:
            cfx_params = extract_cfx_params(args.cfx)
            params.update(cfx_params)

        # CLI overrides
        if args.center_x    is not None: params['center_x']            = args.center_x
        if args.center_z    is not None: params['center_z']            = args.center_z
        if args.shoulder_hw is not None: params['shoulder_half_width'] = args.shoulder_hw
        if args.body_hd     is not None: params['body_half_depth']     = args.body_hd
        if args.foot_height is not None: params['foot_height']         = args.foot_height
        if args.body_height is not None: params['body_height']         = args.body_height

        z_true = make_z_true_body_model(delta_z=dz, **params)
        plot_mannequins = None   # no box outline to draw

    elif args.mode == 'stl':
        # ---- STL mesh mode ----
        if args.stl is None:
            raise ValueError("stl mode requires --stl <path>")
        cx = args.center_x if args.center_x is not None else 2.5
        cz = args.center_z if args.center_z is not None else 1.72
        z_true = make_z_true_from_stl(
            stl_path            = args.stl,
            center_x            = cx,
            center_z            = cz,
            delta_z             = dz,
            foot_height         = args.foot_height,
            body_height_override= args.body_height if args.body_height != 1.78 else None,
        )
        plot_mannequins = None

    save_z_true(z_true, args.label, dz, mode=args.mode)

    if args.plot:
        plot_z_true(z_true, args.label, plot_mannequins)

    print(f"\nDone.  z_true shape={z_true.shape}  dtype={z_true.dtype}")
    print(f"  Non-zero voxels : {np.count_nonzero(z_true)}")
    print(f"  |delta_z|       : {abs(dz):.4f}")
