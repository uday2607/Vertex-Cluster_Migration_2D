"""Synthetic CellGPU netCDF file generators for demos and tests.

SPV filename convention (from param script):
  Data_t_<T>_N_<N>_dt_<dt>_Ka_<Ka>_Kp_<Kp>_A0_1-1_P0_<sa>-<sb>_s0_<sa>-<sb>
  _c_Size_<m>_a_Cells_<a>_va_<va>_vp_<vp>_vb_<vb>
  _T1_<g>_T2_<g>_T3_0_Dr_b_<Dr_b>_Dr_tp_<Dr_tp>_vD_tp_<vD_tp>_IC_<ic>_ensem_<i>.nc

AVM filename convention (same structure, different scenario folder):
  Same token format — produced by avm_pos_gamma_hetero_adhesion.cpp.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import netcdf_file


# ---------------------------------------------------------------------------
# Low-level writer
# ---------------------------------------------------------------------------

def _write_nc(
    path: Path,
    n_cells: int,
    t_len: int,
    dt: float,
    lx: float,
    ly: float,
    pos: np.ndarray,        # (t_len, 2*n_cells)
    cell_type: np.ndarray,  # (t_len, n_cells)
    directors: np.ndarray,  # (n_cells, t_len)
) -> None:
    """Write an SPV-style nc file: pos = cell centroid positions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with netcdf_file(str(path), "w") as nc:
        nc.createDimension("time", t_len)
        nc.createDimension("coord", 2 * n_cells)
        nc.createDimension("cell", n_cells)
        nc.createDimension("boxcoord", 4)

        tv = nc.createVariable("time",      "f8", ("time",))
        pv = nc.createVariable("pos",       "f8", ("time", "coord"))
        yv = nc.createVariable("type",      "i4", ("time", "cell"))
        bv = nc.createVariable("BoxMatrix", "f8", ("boxcoord", "time"))
        dv = nc.createVariable("director",  "f8", ("cell", "time"))

        tv[:] = np.arange(t_len, dtype=float) * dt
        pv[:] = pos
        yv[:] = cell_type

        box = np.zeros((4, t_len), dtype=float)
        box[0, :] = lx
        box[3, :] = ly
        bv[:] = box

        dv[:] = directors


def _build_hex_voronoi_vertices(
    xs: np.ndarray, ys: np.ndarray, lx: float, ly: float, r: float
) -> tuple:
    """Build synthetic vertex positions and connectivity for a Voronoi-like cell layout.

    Each cell gets ~6 vertices placed on a regular hexagon of radius r around its centroid.
    Returns (vx, vy, vert_cell_neigh) as 1-D arrays ready for nc writing.

    vert_cell_neigh shape: (Nv * 3,) — for each vertex, up to 3 cell indices (padded with -1).
    This is a simplified synthetic version: each vertex is shared by one cell only.
    """
    n = len(xs)
    angles = np.linspace(0, 2 * math.pi, 7)[:-1]  # 6 vertices per cell
    verts_per_cell = 6

    all_vx = []
    all_vy = []
    for i in range(n):
        for a in angles:
            all_vx.append(xs[i] + r * math.cos(a))
            all_vy.append(ys[i] + r * math.sin(a))

    nv = n * verts_per_cell
    vcn = np.full((nv, 3), -1, dtype=np.int32)
    for i in range(n):
        for j in range(verts_per_cell):
            vcn[i * verts_per_cell + j, 0] = i

    return (np.array(all_vx, dtype=float),
            np.array(all_vy, dtype=float),
            vcn.reshape(-1))


def _write_avm_nc(
    path: Path,
    n_cells: int,
    t_len: int,
    dt: float,
    lx: float,
    ly: float,
    cell_pos: np.ndarray,   # (t_len, 2*n_cells) — cell centroid positions
    cell_type: np.ndarray,  # (t_len, n_cells)
    directors: np.ndarray,  # (n_cells, t_len)
    vert_r: float = 0.55,   # approximate cell radius for synthetic vertices
) -> None:
    """Write an AVM-style nc file with vertex positions and cellPositions.

    Matches the AVMDatabaseNetCDF variable layout:
      pos          = vertex positions (rec, Nv*2)
      cellPositions = cell centroid positions (rec, Nc*2)
      cellType     = cell types (rec, Nc)
      director     = directors (rec, Nc)
      VertexCellNeighbors = vertex-cell mapping (rec, Nv*3)
      BoxMatrix    = box dimensions (rec, 4)
      time         = scalar per record
    """
    verts_per_cell = 6
    nv = n_cells * verts_per_cell

    path.parent.mkdir(parents=True, exist_ok=True)
    with netcdf_file(str(path), "w") as nc:
        nc.createDimension("rec",    t_len)
        nc.createDimension("Nv",     nv)
        nc.createDimension("Nc",     n_cells)
        nc.createDimension("Nc2",    2 * n_cells)
        nc.createDimension("dof",    nv * 2)
        nc.createDimension("Nvn",    nv * 3)
        nc.createDimension("boxdim", 4)
        nc.createDimension("unit",   1)

        tv  = nc.createVariable("time",                "f8", ("rec", "unit"))
        pv  = nc.createVariable("pos",                 "f8", ("rec", "dof"))
        cpv = nc.createVariable("cellPositions",       "f8", ("rec", "Nc2"))
        ctv = nc.createVariable("cellType",            "f8", ("rec", "Nc"))
        dv  = nc.createVariable("director",            "f8", ("rec", "Nc"))
        vcn = nc.createVariable("VertexCellNeighbors", "i4", ("rec", "Nvn"))
        bv  = nc.createVariable("BoxMatrix",           "f8", ("rec", "boxdim"))

        for ti in range(t_len):
            tv[ti, 0] = ti * dt
            cpv[ti, :] = cell_pos[ti, :]
            ctv[ti, :] = cell_type[ti, :].astype(float)
            dv[ti, :]  = directors[:, ti]

            box = np.zeros(4)
            box[0] = lx
            box[3] = ly
            bv[ti, :] = box

            # Build synthetic vertex positions for this frame
            xs = cell_pos[ti, 0::2]
            ys = cell_pos[ti, 1::2]
            vx_t, vy_t, vcn_t = _build_hex_voronoi_vertices(xs, ys, lx, ly, vert_r)

            vert_pos = np.zeros(nv * 2)
            vert_pos[0::2] = vx_t
            vert_pos[1::2] = vy_t
            pv[ti, :]  = vert_pos
            vcn[ti, :] = vcn_t


# ---------------------------------------------------------------------------
# Generic example (used by tests — small, parametric)
# ---------------------------------------------------------------------------

def create_example_nc(path: Path, n_cells: int = 6, t_len: int = 12, dt: float = 0.1) -> None:
    """Create a small synthetic CellGPU-like netCDF file for analysis demos/tests."""
    lx, ly = 10.0, 10.0
    pos = np.zeros((t_len, 2 * n_cells), dtype=float)
    base_x = np.linspace(2.0, 7.0, n_cells)
    base_y = np.linspace(2.0, 7.0, n_cells)

    for t in range(t_len):
        xv = base_x.copy()
        yv = base_y.copy()
        # Migratory cells (types 1, 2) drift in +x
        xv[0] += 0.12 * t
        xv[1] += 0.10 * t
        xv[2] += 0.08 * t
        yv[0] += 0.02 * np.sin(0.4 * t)
        yv[1] += 0.02 * np.cos(0.3 * t)
        pos[t, 0::2] = xv
        pos[t, 1::2] = yv

    ctype = np.zeros((t_len, n_cells), dtype=np.int32)
    ctype[:, 0] = 1
    ctype[:, 1] = 2
    ctype[:, 2] = 1
    if n_cells > 3:
        ctype[:, 3] = 0
    if n_cells > 4:
        ctype[:, 4] = 0
    if n_cells > 5:
        ctype[:, 5] = 5

    dirs = np.zeros((n_cells, t_len), dtype=float)
    dirs[0, :] = 0.0
    dirs[1, :] = 0.1
    dirs[2, :] = -0.1

    _write_nc(path, n_cells, t_len, dt, lx, ly, pos, ctype, dirs)


# ---------------------------------------------------------------------------
# SPV example — realistic filename, realistic layout
# ---------------------------------------------------------------------------

def create_spv_example(
    output_dir: Path,
    N: int = 400,
    m_val: int = 8,
    a_val: int = 4,
    sa: float = 3.7,
    sb: float = 3.7,
    va: float = 2.0,
    gamma: float = 0.0,
    Ka: float = 10.0,
    Kp: float = 1.0,
    Dr_tp: float = 0.005,
    vD_tp: float = 0.02,
    IC: int = 3,
    ensem: int = 1,
    t_steps: int = 200,
    dt: float = 0.0001,
    overwrite: bool = False,
) -> Path:
    """Create a synthetic SPV example .nc file with the real filename convention.

    Filename matches the pattern used by spv_pos_gamma_hetero_adhesion.cpp:
      Data_t_{T}_N_{N}_dt_{dt}_Ka_{Ka}_Kp_{Kp}_A0_1-1_P0_{sa}-{sb}_s0_{sa}-{sb}
      _c_Size_{m}_a_Cells_{a}_va_{va}_vp_0_vb_0.01_T1_{g}_T2_{g}_T3_0
      _Dr_b_0.05_Dr_tp_{Dr_tp}_vD_tp_{vD_tp}_IC_{IC}_ensem_{i}.nc
    """
    t_total = t_steps * int(round(1.0 / dt / 1000)) if dt < 0.001 else t_steps
    t_len   = min(t_steps, 50)  # keep synthetic files small
    lx, ly  = 20.0, 20.0

    g_str  = int(gamma) if gamma == int(gamma) else gamma
    sa_str = sa
    sb_str = sb
    va_str = int(va) if va == int(va) else va
    Ka_str = int(Ka) if Ka == int(Ka) else Ka
    Kp_str = int(Kp) if Kp == int(Kp) else Kp

    fname = (
        f"Data_t_{t_steps}_N_{N}_dt_{dt}"
        f"_Ka_{Ka_str}_Kp_{Kp_str}_A0_1-1"
        f"_P0_{sa_str}-{sb_str}_s0_{sa_str}-{sb_str}"
        f"_c_Size_{m_val}_a_Cells_{a_val}"
        f"_va_{va_str}_vp_0_vb_0.01"
        f"_T1_{g_str}_T2_{g_str}_T3_0"
        f"_Dr_b_0.05_Dr_tp_{Dr_tp}_vD_tp_{vD_tp}"
        f"_IC_{IC}_ensem_{ensem}.nc"
    )
    path = output_dir / fname
    if path.exists() and not overwrite:
        return path

    rng = np.random.default_rng(seed=ensem)

    # Place N cells on a ~square lattice inside the box
    n_side = int(math.ceil(math.sqrt(N)))
    xs_all = np.tile(np.linspace(1.0, lx - 1.0, n_side), n_side)[:N]
    ys_all = np.repeat(np.linspace(1.0, ly - 1.0, n_side), n_side)[:N]
    xs_all += rng.normal(0, 0.05, N)
    ys_all += rng.normal(0, 0.05, N)
    xs_all = np.clip(xs_all, 0.5, lx - 0.5)
    ys_all = np.clip(ys_all, 0.5, ly - 0.5)

    # Cluster starts as a compact block of m_val cells near the right side
    cluster_center_x = lx - 3.0
    cluster_center_y = ly / 2.0
    cluster_cols = max(1, int(math.ceil(math.sqrt(m_val))))
    spacing = 1.0
    ci = 0
    for row in range(cluster_cols):
        for col in range(cluster_cols):
            if ci >= m_val:
                break
            xs_all[ci] = cluster_center_x + (col - cluster_cols // 2) * spacing
            ys_all[ci] = cluster_center_y + (row - cluster_cols // 2) * spacing
            ci += 1

    ctype = np.zeros((t_len, N), dtype=np.int32)
    # First a_val cells are active (type 1), next (m_val - a_val) are passive (type 2)
    ctype[:, :a_val] = 1
    ctype[:, a_val:m_val] = 2

    directors = np.zeros((N, t_len), dtype=float)
    # Active cells point roughly leftward (-x direction ≈ π) with noise
    for ci in range(a_val):
        directors[ci, :] = math.pi + rng.normal(0, 0.3, t_len)

    pos = np.zeros((t_len, 2 * N), dtype=float)
    pos[0, 0::2] = xs_all
    pos[0, 1::2] = ys_all

    # Simple Euler-like motion: cluster drifts left, background stays put
    speed_active  = 0.05 * va
    speed_passive = 0.02 * va

    for t in range(1, t_len):
        pos[t, :] = pos[t - 1, :]
        for ci in range(a_val):
            pos[t, 2 * ci]     = pos[t - 1, 2 * ci]     - speed_active  + rng.normal(0, 0.01)
            pos[t, 2 * ci + 1] = pos[t - 1, 2 * ci + 1] + rng.normal(0, 0.01)
        for ci in range(a_val, m_val):
            pos[t, 2 * ci]     = pos[t - 1, 2 * ci]     - speed_passive + rng.normal(0, 0.005)
            pos[t, 2 * ci + 1] = pos[t - 1, 2 * ci + 1] + rng.normal(0, 0.005)
        # Wrap to box
        pos[t, 0::2] = np.mod(pos[t, 0::2], lx)
        pos[t, 1::2] = np.mod(pos[t, 1::2], ly)

    _write_nc(path, N, t_len, dt, lx, ly, pos, ctype, directors)
    return path


# ---------------------------------------------------------------------------
# AVM example — vertex model, same filename convention, different scenario dir
# ---------------------------------------------------------------------------

def create_avm_example(
    output_dir: Path,
    N: int = 400,
    m_val: int = 8,
    a_val: int = 4,
    sa: float = 3.7,
    sb: float = 3.7,
    va: float = 2.0,
    gamma: float = 0.0,
    Ka: float = 10.0,
    Kp: float = 1.0,
    Dr_tp: float = 0.005,
    vD_tp: float = 0.02,
    IC: int = 3,
    ensem: int = 1,
    t_steps: int = 200,
    dt: float = 0.0001,
    overwrite: bool = False,
) -> Path:
    """Create a synthetic AVM (vertex model) example .nc file with real filename convention.

    Filename matches the pattern produced by avm_pos_gamma_hetero_adhesion.cpp.
    Cell positions represent vertex-model centroids — same layout as SPV but
    with slightly different cell ordering to reflect AVM initial conditions.
    """
    t_len  = min(t_steps, 50)
    lx, ly = 20.0, 20.0

    g_str  = int(gamma) if gamma == int(gamma) else gamma
    va_str = int(va) if va == int(va) else va
    Ka_str = int(Ka) if Ka == int(Ka) else Ka
    Kp_str = int(Kp) if Kp == int(Kp) else Kp

    fname = (
        f"Data_t_{t_steps}_N_{N}_dt_{dt}"
        f"_Ka_{Ka_str}_Kp_{Kp_str}_A0_1-1"
        f"_P0_{sa}-{sb}_s0_{sa}-{sb}"
        f"_c_Size_{m_val}_a_Cells_{a_val}"
        f"_va_{va_str}_vp_0_vb_0.01"
        f"_T1_{g_str}_T2_{g_str}_T3_0"
        f"_Dr_b_0.05_Dr_tp_{Dr_tp}_vD_tp_{vD_tp}"
        f"_IC_{IC}_ensem_{ensem}.nc"
    )
    path = output_dir / fname
    if path.exists() and not overwrite:
        return path

    rng = np.random.default_rng(seed=ensem + 1000)

    n_side = int(math.ceil(math.sqrt(N)))
    xs_all = np.tile(np.linspace(1.0, lx - 1.0, n_side), n_side)[:N]
    ys_all = np.repeat(np.linspace(1.0, ly - 1.0, n_side), n_side)[:N]
    xs_all += rng.normal(0, 0.05, N)
    ys_all += rng.normal(0, 0.05, N)

    # AVM: cluster placed slightly closer to the right edge
    cluster_center_x = lx - 2.5
    cluster_center_y = ly / 2.0
    cluster_cols = max(1, int(math.ceil(math.sqrt(m_val))))
    ci = 0
    for row in range(cluster_cols):
        for col in range(cluster_cols):
            if ci >= m_val:
                break
            xs_all[ci] = cluster_center_x + (col - cluster_cols // 2) * 1.0
            ys_all[ci] = cluster_center_y + (row - cluster_cols // 2) * 1.0
            ci += 1

    ctype = np.zeros((t_len, N), dtype=np.int32)
    ctype[:, :a_val] = 1
    ctype[:, a_val:m_val] = 2

    directors = np.zeros((N, t_len), dtype=float)
    for ci in range(a_val):
        directors[ci, :] = math.pi + rng.normal(0, 0.25, t_len)

    cell_pos = np.zeros((t_len, 2 * N), dtype=float)
    cell_pos[0, 0::2] = xs_all
    cell_pos[0, 1::2] = ys_all

    # AVM dynamics: slightly stronger coherent migration due to tension line mechanism
    speed_active  = 0.07 * va
    speed_passive = 0.03 * va

    for t in range(1, t_len):
        cell_pos[t, :] = cell_pos[t - 1, :]
        for ci in range(a_val):
            dir_noise = rng.normal(0, 0.15)
            cell_pos[t, 2 * ci]     = cell_pos[t - 1, 2 * ci]     - speed_active * (1 + 0.2 * dir_noise)
            cell_pos[t, 2 * ci + 1] = cell_pos[t - 1, 2 * ci + 1] + speed_active * 0.1 * dir_noise
        for ci in range(a_val, m_val):
            cell_pos[t, 2 * ci]     = cell_pos[t - 1, 2 * ci]     - speed_passive + rng.normal(0, 0.005)
            cell_pos[t, 2 * ci + 1] = cell_pos[t - 1, 2 * ci + 1] + rng.normal(0, 0.005)
        cell_pos[t, 0::2] = np.mod(cell_pos[t, 0::2], lx)
        cell_pos[t, 1::2] = np.mod(cell_pos[t, 1::2], ly)

    # Estimate cell radius from packing density (area per cell ~ 1 for normalized coords)
    vert_r = math.sqrt(lx * ly / N / math.pi) * 0.52

    _write_avm_nc(path, N, t_len, dt, lx, ly, cell_pos, ctype, directors, vert_r=vert_r)
    return path


