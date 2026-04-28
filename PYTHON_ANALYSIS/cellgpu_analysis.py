#!/usr/bin/env python3
"""Unified post-processing utilities for CellGPU cluster migration simulations.

Workflow:
  1) compute metrics from one or many .nc files
  2) plot key metrics from generated CSVs
  3) make a movie from a .nc trajectory

Usage:
  python cellgpu_analysis.py metrics --input "DATA/POS_GAMMA_HETERO/*.nc"
  python cellgpu_analysis.py plot    --timeseries RESULTS/PROCESSED_DATA/...csv
  python cellgpu_analysis.py movie   --input DATA/POS_GAMMA_HETERO/file.nc
  python cellgpu_analysis.py run-all --input DATA/POS_GAMMA_HETERO/file.nc

Optional heavy metrics (Voronoi shape index, MSD, T1 transitions) require:
  freud, networkx — install with:  pip install freud-analysis networkx
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import netcdf_file
from scipy.stats import sem as scipy_sem

# Optional heavy deps — imported lazily so the script works without them.
try:
    import freud
    import networkx as nx
    _HAVE_FREUD = True
except ImportError:
    _HAVE_FREUD = False

try:
    import numba as nb
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SimulationData:
    source: Path
    x: np.ndarray          # (N, T)  — cell centroid x, periodic-wrapped
    y: np.ndarray          # (N, T)  — cell centroid y, periodic-wrapped
    cell_type: np.ndarray  # (T, N)
    time: np.ndarray       # (T,)
    lx: np.ndarray         # (T,)
    ly: np.ndarray         # (T,)
    directors: Optional[np.ndarray] = None   # (T, N)
    params: Dict[str, object] = field(default_factory=dict)
    model_type: str = "SPV"                  # "SPV" or "AVM"
    # AVM-only: vertex positions and connectivity (None for SPV)
    vx: Optional[np.ndarray] = None          # (Nv, T) vertex x
    vy: Optional[np.ndarray] = None          # (Nv, T) vertex y
    vert_cell_neigh: Optional[np.ndarray] = None   # (Nv*3, T) VertexCellNeighbors
    vert_neigh: Optional[np.ndarray] = None        # (Nv*3, T) Vneighs


# ---------------------------------------------------------------------------
# Filename parameter parsing
# ---------------------------------------------------------------------------

def parse_params_from_filename(path: Path) -> Dict[str, object]:
    """Extract simulation parameters encoded in the CellGPU filename convention.

    Expected tokens (underscored key-value pairs in the filename stem):
      N, dt, Ka, Kp, P0 (sa-sb), s0 (sa-sb), va, vp, vb, T1, T2, T3,
      Dr_b, Dr_tp, vD_tp, IC, a_Cells (active cell count), c_Size (cluster size),
      ensem (ensemble index).
    """
    parts = path.stem.split("_")
    p: Dict[str, object] = {}

    def _get(key: str) -> Optional[str]:
        try:
            i = parts.index(key)
            return parts[i + 1] if i + 1 < len(parts) else None
        except ValueError:
            return None

    def _f(s: Optional[str]) -> Optional[float]:
        try:
            return float(s) if s is not None else None
        except ValueError:
            return None

    def _i(s: Optional[str]) -> Optional[int]:
        try:
            return int(s) if s is not None else None
        except ValueError:
            return None

    for k, fn in [("N", _i), ("dt", _f), ("Ka", _f), ("Kp", _f),
                  ("va", _f), ("vp", _f), ("vb", _f), ("IC", _i), ("ensem", _i),
                  ("T1", _f), ("T2", _f), ("T3", _f)]:
        v = fn(_get(k))
        if v is not None:
            p[k] = v

    p0_raw = _get("P0")
    if p0_raw and "-" in p0_raw:
        sa, sb = p0_raw.split("-", 1)
        if _f(sa) is not None:
            p["sa"] = _f(sa)
        if _f(sb) is not None:
            p["sb"] = _f(sb)

    # Dr_b and Dr_tp
    for j, token in enumerate(parts):
        if token == "Dr" and j + 1 < len(parts):
            subkey = parts[j + 1]
            val = _f(parts[j + 2] if j + 2 < len(parts) else None)
            if subkey == "b" and val is not None:
                p["Dr_b"] = val
            elif subkey == "tp" and val is not None:
                p["Dr_tp"] = val
        if token == "vD" and j + 1 < len(parts) and parts[j + 1] == "tp":
            val = _f(parts[j + 2] if j + 2 < len(parts) else None)
            if val is not None:
                p["vD_tp"] = val

    for j, token in enumerate(parts):
        if token == "Size":
            v = _i(parts[j + 1] if j + 1 < len(parts) else None)
            if v is not None:
                p["m_val"] = v
        if token == "Cells":
            v = _i(parts[j + 1] if j + 1 < len(parts) else None)
            if v is not None:
                p["a_val"] = v

    return p


# ---------------------------------------------------------------------------
# NetCDF loading helpers
# ---------------------------------------------------------------------------

def _to_float(var) -> np.ndarray:
    return np.asarray(var[:], dtype=float).copy()


def _to_int(var) -> np.ndarray:
    return np.asarray(var[:], dtype=int).copy()


def _infer_pos(raw: np.ndarray, n_hint: Optional[int]) -> np.ndarray:
    """Return pos in shape (2N, T)."""
    if raw.ndim != 2:
        raise ValueError(f"Expected 2D pos, got {raw.shape}")
    candidates = []
    if raw.shape[0] % 2 == 0:
        candidates.append(raw)
    if raw.shape[1] % 2 == 0:
        candidates.append(raw.T)
    if not candidates:
        raise ValueError(f"Cannot orient pos from shape {raw.shape}")
    if n_hint is None:
        return candidates[0]
    for c in candidates:
        if c.shape[0] // 2 == n_hint:
            return c
    return candidates[0]


def _infer_type(raw: np.ndarray, n: int) -> np.ndarray:
    """Return type in shape (T, N)."""
    if raw.ndim != 2:
        raise ValueError(f"Expected 2D type, got {raw.shape}")
    if raw.shape[1] == n:
        return raw
    if raw.shape[0] == n:
        return raw.T
    return raw if abs(raw.shape[1] - n) <= abs(raw.shape[0] - n) else raw.T


def _infer_dir(raw: np.ndarray, n: int, t: int) -> np.ndarray:
    """Return directors in shape (T, N)."""
    if raw.ndim != 2:
        raise ValueError(f"Expected 2D director, got {raw.shape}")
    if raw.shape == (t, n):
        return raw
    if raw.shape == (n, t):
        return raw.T
    return raw if raw.shape[1] == n else raw.T


def _infer_box(raw: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lx, ly) each shape (T,)."""
    if raw.ndim == 1:
        if raw.size < 4:
            raise ValueError(f"BoxMatrix too small: {raw.shape}")
        return np.full(t, float(raw[0])), np.full(t, float(raw[3]))
    if raw.ndim != 2:
        raise ValueError(f"Expected 1D/2D BoxMatrix, got {raw.shape}")
    box4t = raw if raw.shape[0] == 4 else raw.T
    if box4t.shape[1] == 1:
        return np.full(t, float(box4t[0, 0])), np.full(t, float(box4t[3, 0]))
    if box4t.shape[1] == t:
        return box4t[0, :].astype(float), box4t[3, :].astype(float)
    lx_src, ly_src = box4t[0, :].astype(float), box4t[3, :].astype(float)
    reps = int(math.ceil(t / len(lx_src)))
    return np.tile(lx_src, reps)[:t], np.tile(ly_src, reps)[:t]


def load_nc_file(path: Path) -> SimulationData:
    """Load a CellGPU netCDF file into normalized arrays.

    Detects model type from nc variable names:
    - SPV: 'pos' stores cell centroid positions (Nc*2 coords).
    - AVM: 'pos' stores vertex positions (Nv*2); 'cellPositions' stores cell centroids.
      Also reads VertexCellNeighbors and Vneighs for polygon rendering.
    """
    with netcdf_file(str(path), "r") as nc:
        for req in ("pos", "time"):
            if req not in nc.variables:
                raise ValueError(f"Missing '{req}' in {path}")

        is_avm = "cellPositions" in nc.variables

        time = _to_float(nc.variables["time"]).reshape(-1)
        t_len = len(time)

        if is_avm:
            # AVM: cell centroids in cellPositions, cell type in cellType, vertex pos in pos
            type_key = "cellType" if "cellType" in nc.variables else "type"
            if type_key not in nc.variables:
                raise ValueError(f"AVM file missing cellType/type in {path}")

            raw_cell_pos = _to_float(nc.variables["cellPositions"])
            # cellPositions shape: (rec, Nc2) or (Nc2, rec)
            if raw_cell_pos.ndim == 2:
                if raw_cell_pos.shape[0] == t_len:
                    cell_pos = raw_cell_pos.T   # → (Nc2, T)
                else:
                    cell_pos = raw_cell_pos     # already (Nc2, T)
            else:
                raise ValueError(f"Unexpected cellPositions shape {raw_cell_pos.shape}")
            n = cell_pos.shape[0] // 2

            raw_type = _to_int(nc.variables[type_key])
            cell_type = _infer_type(raw_type, n)
            if cell_type.shape[0] != t_len:
                t_len = min(t_len, cell_type.shape[0], cell_pos.shape[1])
                time = time[:t_len]
                cell_type = cell_type[:t_len, :]
                cell_pos = cell_pos[:, :t_len]

            if "BoxMatrix" in nc.variables:
                lx, ly = _infer_box(_to_float(nc.variables["BoxMatrix"]), t_len)
            else:
                lx = np.full(t_len, np.nan)
                ly = np.full(t_len, np.nan)

            directors: Optional[np.ndarray] = None
            if "director" in nc.variables:
                raw_dir = _to_float(nc.variables["director"])
                directors = _infer_dir(raw_dir, n, t_len)[:t_len, :]

            x = cell_pos[0::2, :]
            y = cell_pos[1::2, :]
            if np.all(np.isfinite(lx)) and np.all(lx > 0):
                x = np.mod(x, lx[None, :])
            if np.all(np.isfinite(ly)) and np.all(ly > 0):
                y = np.mod(y, ly[None, :])

            # Vertex arrays for polygon rendering
            raw_vpos = _to_float(nc.variables["pos"])
            nv2 = raw_vpos.shape[1] if raw_vpos.ndim == 2 else raw_vpos.shape[0]
            nv = nv2 // 2
            if raw_vpos.ndim == 2:
                vpos = raw_vpos[:t_len, :].T if raw_vpos.shape[0] == t_len else raw_vpos[:, :t_len]
            else:
                vpos = raw_vpos

            vx = vpos[0::2, :]
            vy = vpos[1::2, :]

            vcn: Optional[np.ndarray] = None
            vn: Optional[np.ndarray] = None
            if "VertexCellNeighbors" in nc.variables:
                raw_vcn = nc.variables["VertexCellNeighbors"][:].copy()
                vcn = raw_vcn[:t_len, :].T if raw_vcn.shape[0] == t_len else raw_vcn[:, :t_len]
            if "Vneighs" in nc.variables:
                raw_vn = nc.variables["Vneighs"][:].copy()
                vn = raw_vn[:t_len, :].T if raw_vn.shape[0] == t_len else raw_vn[:, :t_len]

            params = parse_params_from_filename(path)
            return SimulationData(
                source=path, x=x, y=y, cell_type=cell_type,
                time=time, lx=lx, ly=ly, directors=directors, params=params,
                model_type="AVM", vx=vx, vy=vy,
                vert_cell_neigh=vcn, vert_neigh=vn,
            )

        # ---- SPV path ----
        type_key = "type" if "type" in nc.variables else "cellType"
        if type_key not in nc.variables:
            raise ValueError(f"SPV file missing type/cellType in {path}")

        raw_type = _to_int(nc.variables[type_key])
        n_hint = raw_type.shape[1] if raw_type.ndim == 2 else None

        raw_pos = _to_float(nc.variables["pos"])
        pos = _infer_pos(raw_pos, n_hint)
        n = pos.shape[0] // 2

        cell_type = _infer_type(raw_type, n)
        if cell_type.shape[0] != t_len:
            t_len = min(t_len, cell_type.shape[0], pos.shape[1])
            time = time[:t_len]
            cell_type = cell_type[:t_len, :]
            pos = pos[:, :t_len]

        if "BoxMatrix" in nc.variables:
            lx, ly = _infer_box(_to_float(nc.variables["BoxMatrix"]), t_len)
        else:
            lx = np.full(t_len, np.nan)
            ly = np.full(t_len, np.nan)

        directors = None
        if "director" in nc.variables:
            raw_dir = _to_float(nc.variables["director"])
            directors = _infer_dir(raw_dir, n, t_len)[:t_len, :]

    x = pos[0::2, :]
    y = pos[1::2, :]

    if np.all(np.isfinite(lx)) and np.all(lx > 0):
        x = np.mod(x, lx[None, :])
    if np.all(np.isfinite(ly)) and np.all(ly > 0):
        y = np.mod(y, ly[None, :])

    params = parse_params_from_filename(path)

    return SimulationData(
        source=path, x=x, y=y, cell_type=cell_type,
        time=time, lx=lx, ly=ly, directors=directors, params=params,
        model_type="SPV",
    )


# ---------------------------------------------------------------------------
# MSD — numba JIT when available, pure numpy fallback
# ---------------------------------------------------------------------------

def _compute_msd_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_cells, t_len = x.shape
    msd = np.zeros((n_cells, t_len))
    for delta in range(1, t_len):
        dx = x[:, delta:] - x[:, :t_len - delta]
        dy = y[:, delta:] - y[:, :t_len - delta]
        msd[:, delta] = np.mean(dx**2 + dy**2, axis=1)
    return msd


if _HAVE_NUMBA:
    @nb.njit
    def _msd_nb(x, y):
        n_cells, t_len = x.shape
        msd = np.zeros((n_cells, t_len))
        for delta in range(1, t_len):
            count = t_len - delta
            for i in range(delta, t_len):
                dx = x[:, i] - x[:, i - delta]
                dy = y[:, i] - y[:, i - delta]
                msd[:, delta] += dx**2 + dy**2
            msd[:, delta] /= count
        return msd

    def compute_msd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return _msd_nb(x, y)
else:
    def compute_msd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return _compute_msd_numpy(x, y)


# ---------------------------------------------------------------------------
# Ellipse fitting (Halir & Flusser numerically stable method)
# ---------------------------------------------------------------------------

def _fit_ellipse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    D1 = np.vstack([x**2, x * y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    C = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]], dtype=float)
    M = np.linalg.inv(C) @ (S1 + S2 @ T)
    _, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0] * eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()


def _ellipse_params(coeffs: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """Return (xc, yc, semi_major, semi_minor, eccentricity, angle_rad)."""
    a, b, c, d, f, g = coeffs[0], coeffs[1] / 2, coeffs[2], coeffs[3] / 2, coeffs[4] / 2, coeffs[5]
    den = b**2 - a * c
    if den >= 0:
        raise ValueError("Coefficients do not represent an ellipse")
    x0 = (c * d - b * f) / den
    y0 = (a * f - b * d) / den
    num = 2 * (a * f**2 + c * d**2 + g * b**2 - 2 * b * d * f - a * c * g)
    fac = math.sqrt((a - c)**2 + 4 * b**2)
    ap = math.sqrt(num / den / (fac - a - c))
    bp = math.sqrt(num / den / (-fac - a - c))
    width_gt_height = ap >= bp
    if not width_gt_height:
        ap, bp = bp, ap
    r = (bp / ap)**2
    e = math.sqrt(1 - min(r, 1.0))
    if b == 0:
        phi = 0.0 if a < c else math.pi / 2
    else:
        phi = math.atan(2 * b / (a - c)) / 2
        if a > c:
            phi += math.pi / 2
    if not width_gt_height:
        phi += math.pi / 2
    phi = phi % math.pi
    return float(x0), float(y0), float(ap), float(bp), float(e), float(phi)


# ---------------------------------------------------------------------------
# Voronoi perimeters and cluster boundary helpers (requires freud)
# ---------------------------------------------------------------------------

def _voronoi_perimeters(voro, indices: np.ndarray) -> np.ndarray:
    peris = []
    for ind in indices:
        ver = voro.polytopes[ind][:, :2]
        rolled = np.roll(ver, -1, axis=0)
        peris.append(float(np.sum(np.sqrt(np.sum((rolled - ver)**2, axis=1)))))
    return np.array(peris)


def _cluster_boundary_verts(voro, indices: np.ndarray, b_indices: np.ndarray,
                            ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Shared Voronoi vertices on the cluster-background interface."""
    idx_set = set(indices.tolist())
    points: set = set()
    peri = 0.0
    area = 0.0
    nlist = voro.nlist[:]
    for ind in indices:
        area += float(voro.volumes[ind])
        neighs = np.array([j for i, j in nlist if i == ind or j == ind])
        ind_verts = set(tuple(v) for v in np.round(voro.polytopes[ind][:, :2], 7))
        for cn in np.intersect1d(b_indices, neighs):
            cn_verts = set(tuple(v) for v in np.round(voro.polytopes[cn][:, :2], 7))
            shared = np.array(list(ind_verts & cn_verts))
            for pt in shared:
                points.add(tuple(pt))
            if len(shared) == 2:
                peri += float(np.linalg.norm(shared[0] - shared[1]))
    pts = np.array(list(points)) if points else np.empty((0, 2))
    return pts[:, 0] if len(pts) else np.array([]), pts[:, 1] if len(pts) else np.array([]), area, peri


def _cluster_is_intact(voro, indices: np.ndarray) -> bool:
    if len(indices) < 2:
        return False
    idx_set = set(indices.tolist())
    nlist = voro.nlist[:]
    edges = [(i, j) for (i, j) in nlist if i in idx_set and j in idx_set]
    if not edges:
        return False
    G = nx.DiGraph()
    G.add_edges_from(edges)
    try:
        reachable = set(nx.shortest_path(G, edges[0][0]).keys())
        return idx_set == reachable
    except Exception:
        return False


def compute_voronoi_shape_metrics(
    x: np.ndarray, y: np.ndarray, cell_type: np.ndarray,
    lx: float, ly: float, t: int, a_val: int, m_val: int,
) -> Dict[str, float]:
    """Voronoi-based cluster shape metrics at a single timestep t.

    Returns dict with: c_index, c_area, c_peri, ab_ratio,
    p_active, p_passive, p_background. NaN when freud missing or cluster not intact.
    """
    nan_result = {k: float("nan") for k in
                  ("c_index", "c_area", "c_peri", "ab_ratio",
                   "p_active", "p_passive", "p_background")}
    if not _HAVE_FREUD:
        return nan_result

    n = x.shape[0]
    mask = ((x[:, t] > 0) & (x[:, t] < lx) &
            (y[:, t] > 0) & (y[:, t] < ly) &
            ((cell_type[t, :] == 1) | (cell_type[t, :] == 2)))
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return nan_result

    b_indices = np.setxor1d(np.arange(n), indices)
    XY = np.column_stack([x[:, t], y[:, t], np.zeros(n)])
    box_sys = freud.box.Box(Lx=lx, Ly=ly, is2D=True)
    voro = freud.locality.Voronoi()
    voro.compute((box_sys, XY))

    result = dict(nan_result)

    # Per-cell shape indices (perimeter / sqrt(area))
    if a_val > 0 and len(indices) >= a_val:
        pa = _voronoi_perimeters(voro, indices[:a_val])
        va = voro.volumes[indices[:a_val]]
        result["p_active"] = float(np.mean(pa / np.sqrt(va)))
    else:
        result["p_active"] = 0.0

    if a_val < m_val and len(indices) > a_val:
        pp = _voronoi_perimeters(voro, indices[a_val:])
        vp = voro.volumes[indices[a_val:]]
        result["p_passive"] = float(np.mean(pp / np.sqrt(vp)))
    else:
        result["p_passive"] = 0.0

    if len(b_indices) > 0:
        pb = _voronoi_perimeters(voro, b_indices)
        vb = voro.volumes[b_indices]
        result["p_background"] = float(np.mean(pb / np.sqrt(vb)))

    if _cluster_is_intact(voro, indices):
        x_pts, y_pts, area, peri = _cluster_boundary_verts(voro, indices, b_indices)
        if peri > 0:
            result["c_index"] = float(4 * math.pi * area / peri**2)
            result["c_area"] = float(area)
            result["c_peri"] = float(peri)
        if len(x_pts) >= 5:
            try:
                coeffs = _fit_ellipse(x_pts, y_pts)
                _, _, e_a, e_b, _, _ = _ellipse_params(coeffs)
                result["ab_ratio"] = float(e_a / e_b) if e_b > 0 else float("nan")
            except Exception:
                pass

    return result


# ---------------------------------------------------------------------------
# T1 transition detection at cluster-background and cluster-cluster interfaces
# ---------------------------------------------------------------------------

def _detect_T1(
    voro_prev, voro_curr, indices: np.ndarray,
    x: np.ndarray, y: np.ndarray, t_prev: int, t_curr: int,
) -> Tuple[int, int, int, int, int, int]:
    """Count T1 events.

    Returns: (front_creations, rear_heals, front_breaks, rear_gaps,
               homotypic_gains, homotypic_losses)
    """
    if not _HAVE_FREUD:
        return 0, 0, 0, 0, 0, 0
    idx_set = set(indices.tolist())

    def het_edges(voro):
        s = set()
        for i, j in voro.nlist[:]:
            if (i in idx_set) != (j in idx_set):
                s.add(tuple(sorted([i, j])))
        return s

    def hom_edges(voro):
        s = set()
        for i, j in voro.nlist[:]:
            if i in idx_set and j in idx_set:
                s.add(tuple(sorted([i, j])))
        return s

    het_prev, het_curr = het_edges(voro_prev), het_edges(voro_curr)
    hom_prev, hom_curr = hom_edges(voro_prev), hom_edges(voro_curr)

    appeared = het_curr - het_prev
    disappeared = het_prev - het_curr

    cx_curr = float(np.mean(x[indices, t_curr]))
    cy_curr = float(np.mean(y[indices, t_curr]))
    dx = cx_curr - float(np.mean(x[indices, t_prev]))
    dy = cy_curr - float(np.mean(y[indices, t_prev]))
    norm = math.sqrt(dx**2 + dy**2)
    u = np.array([dx, dy]) / norm if norm > 1e-9 else np.array([-1.0, 0.0])

    def classify(edge, t):
        ci = edge[0] if edge[0] in idx_set else edge[1]
        rel = np.array([x[ci, t] - cx_curr, y[ci, t] - cy_curr])
        rn = np.linalg.norm(rel)
        if rn < 1e-9:
            return 0
        return 1 if np.dot(rel / rn, u) > 0 else -1

    n_fc = sum(1 for e in appeared    if classify(e, t_curr) == 1)
    n_rh = sum(1 for e in appeared    if classify(e, t_curr) == -1)
    n_fb = sum(1 for e in disappeared if classify(e, t_prev) == 1)
    n_rg = sum(1 for e in disappeared if classify(e, t_prev) == -1)
    n_hg = len(hom_curr - hom_prev)
    n_hl = len(hom_prev - hom_curr)
    return n_fc, n_rh, n_fb, n_rg, n_hg, n_hl


# ---------------------------------------------------------------------------
# Polarity metric (PCA + director alignment)
# ---------------------------------------------------------------------------

def _compute_polarity(
    x: np.ndarray, y: np.ndarray, cell_type: np.ndarray,
    directors: Optional[np.ndarray], lx: float, ly: float, t: int,
) -> float:
    """Fraction of pairwise (active, passive) comparisons where active leads passive.

    Uses cluster principal axis aligned with mean active-cell director.
    Returns 0.5 if indeterminate.
    """
    mask = ((x[:, t] > 0) & (x[:, t] < lx) & (y[:, t] > 0) & (y[:, t] < ly) &
            ((cell_type[t, :] == 1) | (cell_type[t, :] == 2)))
    all_idx = np.where(mask)[0]
    if len(all_idx) < 2:
        return 0.5

    active_idx = all_idx[cell_type[t, all_idx] == 1]
    passive_idx = all_idx[cell_type[t, all_idx] == 2]
    if len(active_idx) == 0 or len(passive_idx) == 0:
        return 0.5

    pts = np.column_stack([x[all_idx, t], y[all_idx, t]])
    com = pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((pts - com).T))
    axis = vecs[:, -1]

    if directors is not None and t < directors.shape[0]:
        theta = directors[t, active_idx]
        mean_vec = np.array([np.mean(np.cos(theta)), np.mean(np.sin(theta))])
        if np.dot(axis, mean_vec) < 0:
            axis = -axis
    else:
        if np.mean(x[active_idx, t]) - com[0] < 0:
            axis = -axis

    ap = np.dot(np.column_stack([x[active_idx, t], y[active_idx, t]]), axis)
    pp = np.dot(np.column_stack([x[passive_idx, t], y[passive_idx, t]]), axis)
    return float(np.clip(np.mean(ap[:, None] > pp[None, :]), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Core metrics computation
# ---------------------------------------------------------------------------

def parse_type_list(text: str) -> List[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def _cluster_mask(data: SimulationData, t: int, types: Sequence[int]) -> np.ndarray:
    lx_t = float(data.lx[t]) if np.isfinite(data.lx[t]) else np.inf
    ly_t = float(data.ly[t]) if np.isfinite(data.ly[t]) else np.inf
    mask = np.isin(data.cell_type[t, :], types)
    if np.isfinite(lx_t):
        mask &= (data.x[:, t] > 0) & (data.x[:, t] < lx_t)
    if np.isfinite(ly_t):
        mask &= (data.y[:, t] > 0) & (data.y[:, t] < ly_t)
    return mask


def compute_metrics(
    data: SimulationData,
    migratory_types: Sequence[int],
    compute_voronoi: bool = False,
    compute_t1: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Compute per-timestep timeseries and scalar summary metrics.

    Args:
        data: loaded simulation data
        migratory_types: cell type IDs to track (e.g. [1, 2])
        compute_voronoi: compute Voronoi shape metrics (requires freud)
        compute_t1: detect T1 transitions (requires freud, slow)
    """
    n, t_len = data.x.shape
    time = data.time

    com_x        = np.full(t_len, np.nan)
    com_y        = np.full(t_len, np.nan)
    spread       = np.full(t_len, np.nan)
    cluster_size = np.zeros(t_len, dtype=float)
    alignment    = np.full(t_len, np.nan)
    polarity_ts  = np.full(t_len, np.nan)

    c_index_ts      = np.full(t_len, np.nan)
    c_area_ts       = np.full(t_len, np.nan)
    c_peri_ts       = np.full(t_len, np.nan)
    ab_ratio_ts     = np.full(t_len, np.nan)
    p_active_ts     = np.full(t_len, np.nan)
    p_passive_ts    = np.full(t_len, np.nan)
    p_background_ts = np.full(t_len, np.nan)

    t1_fc_ts = np.zeros(t_len)
    t1_rh_ts = np.zeros(t_len)
    t1_fb_ts = np.zeros(t_len)
    t1_rg_ts = np.zeros(t_len)
    t1_hg_ts = np.zeros(t_len)
    t1_hl_ts = np.zeros(t_len)

    lx_scalar = float(np.nanmax(data.lx)) if np.any(np.isfinite(data.lx)) else np.nan
    ly_scalar = float(np.nanmax(data.ly)) if np.any(np.isfinite(data.ly)) else np.nan

    a_val = int(data.params.get("a_val", 0))
    m_val = int(data.params.get("m_val", len(migratory_types)))

    voro_prev = None

    for t in range(t_len):
        mask = _cluster_mask(data, t, migratory_types)
        idx = np.where(mask)[0]
        cluster_size[t] = float(len(idx))

        if len(idx) == 0:
            continue

        xt, yt = data.x[idx, t], data.y[idx, t]
        com_x[t] = float(np.mean(xt))
        com_y[t] = float(np.mean(yt))
        spread[t] = float(np.mean(np.sqrt((xt - com_x[t])**2 + (yt - com_y[t])**2)))

        if data.directors is not None and t < data.directors.shape[0]:
            theta = data.directors[t, idx]
            alignment[t] = float(math.sqrt(np.mean(np.cos(theta))**2 + np.mean(np.sin(theta))**2))

        if np.isfinite(lx_scalar) and np.isfinite(ly_scalar):
            polarity_ts[t] = _compute_polarity(
                data.x, data.y, data.cell_type, data.directors, lx_scalar, ly_scalar, t)

        if compute_voronoi and _HAVE_FREUD and np.isfinite(lx_scalar) and np.isfinite(ly_scalar):
            vm = compute_voronoi_shape_metrics(
                data.x, data.y, data.cell_type, lx_scalar, ly_scalar, t, a_val, m_val)
            c_index_ts[t]      = vm["c_index"]
            c_area_ts[t]       = vm["c_area"]
            c_peri_ts[t]       = vm["c_peri"]
            ab_ratio_ts[t]     = vm["ab_ratio"]
            p_active_ts[t]     = vm["p_active"]
            p_passive_ts[t]    = vm["p_passive"]
            p_background_ts[t] = vm["p_background"]

            if compute_t1 and _HAVE_FREUD:
                XY = np.column_stack([data.x[:, t], data.y[:, t], np.zeros(n)])
                box_sys = freud.box.Box(Lx=lx_scalar, Ly=ly_scalar, is2D=True)
                voro_curr = freud.locality.Voronoi()
                voro_curr.compute((box_sys, XY))
                if voro_prev is not None:
                    fc, rh, fb, rg, hg, hl = _detect_T1(
                        voro_prev, voro_curr, idx, data.x, data.y, t - 1, t)
                    t1_fc_ts[t] = fc; t1_rh_ts[t] = rh
                    t1_fb_ts[t] = fb; t1_rg_ts[t] = rg
                    t1_hg_ts[t] = hg; t1_hl_ts[t] = hl
                voro_prev = voro_curr

    # Speed from COM
    speed   = np.full(t_len, np.nan)
    speed_x = np.full(t_len, np.nan)
    for t in range(1, t_len):
        if not (np.isfinite(com_x[t]) and np.isfinite(com_x[t - 1])):
            continue
        dt = float(time[t] - time[t - 1])
        if dt <= 0:
            continue
        dx = float(com_x[t] - com_x[t - 1])
        dy = float(com_y[t] - com_y[t - 1])
        speed[t]   = math.sqrt(dx**2 + dy**2) / dt
        speed_x[t] = dx / dt

    # MSD for migratory cells at t=0
    mig_idx_t0 = np.where(_cluster_mask(data, 0, migratory_types))[0]
    msd_mean = np.full(t_len, np.nan)
    if len(mig_idx_t0) > 0:
        msd_mean = np.mean(compute_msd(data.x[mig_idx_t0, :], data.y[mig_idx_t0, :]), axis=0)

    # Scalar summaries
    valid = np.where(np.isfinite(com_x))[0]
    if len(valid) >= 2:
        first, last = int(valid[0]), int(valid[-1])
        disp_x = float(com_x[last] - com_x[first])
        disp_y = float(com_y[last] - com_y[first])
    else:
        disp_x = disp_y = float("nan")

    t_ref = int(valid[-1]) if len(valid) >= 2 else t_len - 1

    a_idx = np.where(_cluster_mask(data, 0, [1]) if 1 in migratory_types else np.zeros(n, dtype=bool))[0]
    p_idx = np.where(_cluster_mask(data, 0, [2]) if 2 in migratory_types else np.zeros(n, dtype=bool))[0]

    mean_disp_x_active  = float(np.mean(data.x[a_idx, t_ref] - data.x[a_idx, 0])) if len(a_idx) else float("nan")
    mean_disp_x_passive = float(np.mean(data.x[p_idx, t_ref] - data.x[p_idx, 0])) if len(p_idx) else float("nan")

    final_idx = np.where(_cluster_mask(data, t_ref, migratory_types))[0]
    if len(final_idx) > 1:
        fx, fy = data.x[final_idx, t_ref], data.y[final_idx, t_ref]
        closeness = float(np.mean(np.sqrt((fx - np.mean(fx))**2 + (fy - np.mean(fy))**2)))
    else:
        closeness = float("nan")

    # Boundary success: cells with x < 6.5 at final time (convention: migrating leftward toward x=0)
    a_bound_num = int(np.count_nonzero(data.x[a_idx, t_ref] < 6.5)) if len(a_idx) else 0
    p_bound_num = int(np.count_nonzero(data.x[p_idx, t_ref] < 6.5)) if len(p_idx) else 0

    unique_types, counts = np.unique(data.cell_type[-1, :], return_counts=True)
    type_counts = {str(int(k)): int(v) for k, v in zip(unique_types, counts)}

    summary: Dict[str, object] = {
        "n_cells": float(n),
        "n_frames": float(t_len),
        "time_start": float(time[0]) if t_len else float("nan"),
        "time_end": float(time[-1]) if t_len else float("nan"),
        "cluster_size_mean": float(np.nanmean(cluster_size)),
        "cluster_size_final": float(cluster_size[t_ref]),
        "cluster_disp_x": disp_x,
        "cluster_disp_y": disp_y,
        "mean_disp_x_active": mean_disp_x_active,
        "mean_disp_x_passive": mean_disp_x_passive,
        "cluster_spread_mean": float(np.nanmean(spread)),
        "cluster_spread_final": float(spread[t_ref]),
        "closeness": closeness,
        "cluster_speed_mean": float(np.nanmean(speed[np.isfinite(speed)])) if np.any(np.isfinite(speed)) else float("nan"),
        "cluster_speed_max": float(np.nanmax(speed[np.isfinite(speed)])) if np.any(np.isfinite(speed)) else float("nan"),
        "cluster_alignment_mean": float(np.nanmean(alignment[np.isfinite(alignment)])) if np.any(np.isfinite(alignment)) else float("nan"),
        "polarity_mean": float(np.nanmean(polarity_ts[np.isfinite(polarity_ts)])) if np.any(np.isfinite(polarity_ts)) else float("nan"),
        "polarity_final": float(polarity_ts[t_ref]),
        "a_bound_num": a_bound_num,
        "p_bound_num": p_bound_num,
        "c_index_mean": float(np.nanmean(c_index_ts[np.isfinite(c_index_ts)])) if np.any(np.isfinite(c_index_ts)) else float("nan"),
        "c_index_final": float(c_index_ts[t_ref]),
        "ab_ratio_mean": float(np.nanmean(ab_ratio_ts[np.isfinite(ab_ratio_ts)])) if np.any(np.isfinite(ab_ratio_ts)) else float("nan"),
        "p_active_mean": float(np.nanmean(p_active_ts[np.isfinite(p_active_ts)])) if np.any(np.isfinite(p_active_ts)) else float("nan"),
        "p_passive_mean": float(np.nanmean(p_passive_ts[np.isfinite(p_passive_ts)])) if np.any(np.isfinite(p_passive_ts)) else float("nan"),
        "p_background_mean": float(np.nanmean(p_background_ts[np.isfinite(p_background_ts)])) if np.any(np.isfinite(p_background_ts)) else float("nan"),
        "t1_front_creations_mean": float(np.mean(t1_fc_ts)),
        "t1_rear_heals_mean": float(np.mean(t1_rh_ts)),
        "t1_front_breaks_mean": float(np.mean(t1_fb_ts)),
        "t1_rear_gaps_mean": float(np.mean(t1_rg_ts)),
        "t1_homotypic_gains_mean": float(np.mean(t1_hg_ts)),
        "t1_homotypic_losses_mean": float(np.mean(t1_hl_ts)),
        "final_type_counts": type_counts,
    }
    summary.update({f"param_{k}": v for k, v in data.params.items()})

    timeseries: Dict[str, np.ndarray] = {
        "time": time,
        "cluster_size": cluster_size,
        "com_x": com_x,
        "com_y": com_y,
        "speed": speed,
        "speed_x": speed_x,
        "spread": spread,
        "alignment": alignment,
        "polarity": polarity_ts,
        "msd_mean": msd_mean,
        "c_index": c_index_ts,
        "c_area": c_area_ts,
        "c_peri": c_peri_ts,
        "ab_ratio": ab_ratio_ts,
        "p_active": p_active_ts,
        "p_passive": p_passive_ts,
        "p_background": p_background_ts,
        "t1_front_creations": t1_fc_ts,
        "t1_rear_heals": t1_rh_ts,
        "t1_front_breaks": t1_fb_ts,
        "t1_rear_gaps": t1_rg_ts,
        "t1_homotypic_gains": t1_hg_ts,
        "t1_homotypic_losses": t1_hl_ts,
    }

    return timeseries, summary


# ---------------------------------------------------------------------------
# Ensemble txt output writers (matching old-script file conventions)
# ---------------------------------------------------------------------------

def _write_ensemble_txt(values: List[float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(" ".join(f"{v:.6f}" for v in values) + "\n")


def _write_ensemble_series_txt(rows: List[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(" ".join(f"{v:.6f}" for v in row) + "\n")


def write_ensemble_txts(
    base_path: Path,
    ensemble_summaries: List[Dict[str, object]],
    ensemble_timeseries: List[Dict[str, np.ndarray]],
) -> None:
    """Write per-ensemble .txt files matching old compute_metrics_ensemble output convention."""
    def _collect(key: str) -> List[float]:
        return [float(s[key]) for s in ensemble_summaries
                if key in s and np.isfinite(float(s[key]))]

    def _collect_ts(key: str) -> List[np.ndarray]:
        return [ts[key] for ts in ensemble_timeseries if key in ts]

    scalar_fields = [
        ("cluster_disp_x",           "_ensem_cluster_dist_x.txt"),
        ("mean_disp_x_active",       "_ensem_cluster_dist_x_active.txt"),
        ("mean_disp_x_passive",      "_ensem_cluster_dist_x_passive.txt"),
        ("closeness",                "_ensem_closeness.txt"),
        ("polarity_final",           "_ensem_polarity.txt"),
        ("c_index_final",            "_ensem_c_index.txt"),
        ("ab_ratio_mean",            "_ensem_ab.txt"),
        ("p_active_mean",            "_ensem_p_active.txt"),
        ("p_passive_mean",           "_ensem_p_passive.txt"),
        ("p_background_mean",        "_ensem_p_background.txt"),
        ("a_bound_num",              "_ensem_a_success_num.txt"),
        ("p_bound_num",              "_ensem_p_success_num.txt"),
        ("t1_front_creations_mean",  "_ensem_t1_front_creations.txt"),
        ("t1_rear_heals_mean",       "_ensem_t1_rear_heals.txt"),
        ("t1_front_breaks_mean",     "_ensem_t1_front_breaks.txt"),
        ("t1_rear_gaps_mean",        "_ensem_t1_rear_gaps.txt"),
        ("t1_homotypic_gains_mean",  "_ensem_t1_hom_gains.txt"),
        ("t1_homotypic_losses_mean", "_ensem_t1_hom_losses.txt"),
    ]
    for key, suffix in scalar_fields:
        vals = _collect(key)
        if vals:
            _write_ensemble_txt(vals, Path(str(base_path) + suffix))

    speed_rows = _collect_ts("speed")
    if speed_rows:
        _write_ensemble_series_txt(speed_rows, Path(str(base_path) + "_ensem_speed_time.txt"))

    msd_rows = _collect_ts("msd_mean")
    if msd_rows:
        _write_ensemble_series_txt(msd_rows, Path(str(base_path) + "_ensem_msd.txt"))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

_TIMESERIES_KEYS = [
    "time", "cluster_size", "com_x", "com_y", "speed", "speed_x",
    "spread", "alignment", "polarity", "msd_mean",
    "c_index", "c_area", "c_peri", "ab_ratio",
    "p_active", "p_passive", "p_background",
    "t1_front_creations", "t1_rear_heals", "t1_front_breaks",
    "t1_rear_gaps", "t1_homotypic_gains", "t1_homotypic_losses",
]


def write_timeseries_csv(path: Path, series: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [k for k in _TIMESERIES_KEYS if k in series]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(series["time"])):
            w.writerow([float(series[k][i]) for k in keys])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def resolve_inputs(pattern: str, ensemble_count: int) -> List[Path]:
    if "{i}" in pattern:
        return sorted(p for i in range(1, ensemble_count + 1)
                      if (p := Path(pattern.format(i=i))).exists())
    if any(c in pattern for c in "*?[]"):
        return sorted(Path(p) for p in glob.glob(pattern))
    p = Path(pattern)
    return [p] if p.exists() else []


def load_timeseries_csv(path: Path) -> Dict[str, np.ndarray]:
    data = np.genfromtxt(str(path), delimiter=",", names=True)
    if data.ndim == 0:
        return {nm: np.asarray([float(data[nm])]) for nm in (data.dtype.names or [])}
    return {nm: np.asarray(data[nm], dtype=float) for nm in (data.dtype.names or [])}


def _safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


# ---------------------------------------------------------------------------
# run_metrics
# ---------------------------------------------------------------------------

def run_metrics(args: argparse.Namespace) -> int:
    types = parse_type_list(args.migratory_types)
    inputs = resolve_inputs(args.input, args.ensemble_count)
    if not inputs:
        print(f"No input files found: {args.input}")
        return 1

    do_voro = getattr(args, "voronoi", False)
    do_t1   = getattr(args, "t1", False)
    if (do_voro or do_t1) and not _HAVE_FREUD:
        print("[WARN] --voronoi/--t1 requested but freud/networkx not installed; skipping.")
        do_voro = do_t1 = False

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    ensemble_rows:       List[dict]                  = []
    ensemble_summaries:  List[Dict[str, object]]     = []
    ensemble_timeseries: List[Dict[str, np.ndarray]] = []
    base_out_path: Optional[Path] = None

    for nc_path in inputs:
        try:
            data   = load_nc_file(nc_path)
            series, summary = compute_metrics(
                data, migratory_types=types,
                compute_voronoi=do_voro, compute_t1=do_t1)
        except Exception as exc:
            print(f"[WARN] Failed {nc_path}: {exc}")
            continue

        scenario = nc_path.parent.name or "default"
        out_dir  = output_root / scenario
        stem     = _safe_stem(nc_path)

        write_timeseries_csv(out_dir / f"{stem}_timeseries.csv", series)
        write_json(out_dir / f"{stem}_summary.json", {"source_file": str(nc_path), **summary})

        row = {"source_file": str(nc_path), "scenario": scenario}
        for k, v in summary.items():
            if isinstance(v, (int, float, np.floating)):
                row[k] = float(v)
        ensemble_rows.append(row)
        ensemble_summaries.append(summary)
        ensemble_timeseries.append(series)

        if base_out_path is None:
            clean = stem
            for suf in ("_ensem_1", "_ensem_2", "_ensem_3"):
                clean = clean.replace(suf, "")
            base_out_path = out_dir / clean

        print(f"[OK] {nc_path.name}")

    if not ensemble_rows:
        print("No files processed.")
        return 1

    # Ensemble-level CSV
    csv_path   = output_root / "ensemble_summary.csv"
    fieldnames = sorted({k for row in ensemble_rows for k in row})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(ensemble_rows)

    # Aggregate stats (mean, std, sem, min, max, n)
    aggs: dict = {}
    for key in fieldnames:
        if key in {"source_file", "scenario"}:
            continue
        vals = np.array([r.get(key, np.nan) for r in ensemble_rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        aggs[key] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "sem":  float(scipy_sem(vals)) if len(vals) > 1 else 0.0,
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
            "n":    int(len(vals)),
        }
    write_json(output_root / "ensemble_aggregates.json", aggs)

    # Old-style per-metric txt files
    if base_out_path is not None:
        write_ensemble_txts(base_out_path, ensemble_summaries, ensemble_timeseries)

    print(f"[DONE] {csv_path}")
    print(f"[DONE] {output_root / 'ensemble_aggregates.json'}")
    return 0


# ---------------------------------------------------------------------------
# run_plot — single timeseries CSV → publication-style figure
# ---------------------------------------------------------------------------

def _default_plot_path(ts_path: Path) -> Path:
    parts = list(ts_path.parts)
    scenario = (parts[parts.index("PROCESSED_DATA") + 1]
                if "PROCESSED_DATA" in parts else ts_path.parent.name or "default")
    return Path("RESULTS") / "PLOTS" / scenario / f"{ts_path.stem}_plot.png"


def _style_ax(ax: "plt.Axes", xlabel: str = "time  $\\tilde{t}$", ylabel: str = "") -> None:
    """Apply clean publication style to an axis."""
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def run_plot(args: argparse.Namespace) -> int:
    """Plot physics timeseries from a single *_timeseries.csv file.

    Panels (2×3 grid):
      [0,0] Cluster displacement (x and y) vs time
      [0,1] Cluster speed |ΔCOM|/Δt vs time
      [1,0] Polarity (active cells lead passive, 0–1 scale) vs time
      [1,1] Director alignment (mean cos of director vs migration axis) vs time
      [2,0] MSD of migratory cells vs time (log-log)
      [2,1] Voronoi shape metrics (circularity, a/b ratio) OR cluster spread
    """
    ts_path = Path(args.timeseries)
    if not ts_path.exists():
        print(f"Not found: {ts_path}")
        return 1

    series = load_timeseries_csv(ts_path)
    if "time" not in series:
        print(f"Missing 'time' column in {ts_path}")
        return 1

    t = series["time"]
    # Normalise time to simulation units (t_tilde = t * dt stored as is)
    C_ACTIVE  = "#06b780"
    C_PASSIVE = "#ffcc7b"
    C_DARK    = "#4a4a4a"
    C_MSD     = "#7b4fa6"
    C_SHAPE   = "#c45b7a"

    plt.rcParams.update({
        "font.family": "sans-serif", "axes.linewidth": 0.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "figure.dpi": 120,
    })

    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    fig.suptitle(ts_path.stem.replace("_timeseries", "").replace("_", "  "),
                 fontsize=8, color="#555555", y=1.01)

    # --- [0,0] Cluster displacement vs time ---
    ax = axes[0, 0]
    has_com = "com_x" in series and "com_y" in series
    if has_com:
        x0, y0 = series["com_x"][0], series["com_y"][0]
        disp_x = series["com_x"] - x0
        disp_y = series["com_y"] - y0
        ax.plot(t, disp_x, color=C_DARK,   lw=1.5, label="x displacement")
        ax.plot(t, disp_y, color=C_PASSIVE, lw=1.5, ls="--", label="y displacement")
        ax.axhline(0, color="grey", lw=0.5, ls=":")
        ax.legend(frameon=False, fontsize=8)
    ax.set_title("Cluster Displacement", fontsize=10, fontweight="bold")
    _style_ax(ax, ylabel="displacement (simulation units)")

    # --- [0,1] Speed vs time ---
    ax = axes[0, 1]
    if "speed" in series:
        spd = series["speed"]
        ax.plot(t, spd, color=C_ACTIVE, lw=1.5)
        ax.fill_between(t, 0, spd, alpha=0.15, color=C_ACTIVE)
    ax.set_title("Cluster Speed", fontsize=10, fontweight="bold")
    _style_ax(ax, ylabel="|ΔCOM| / Δt")
    ax.set_ylim(bottom=0)

    # --- [1,0] Polarity vs time ---
    ax = axes[1, 0]
    if "polarity" in series:
        pol = series["polarity"]
        ax.plot(t, pol, color=C_DARK, lw=1.5)
        ax.fill_between(t, 0.5, pol, where=(pol >= 0.5), alpha=0.2, color=C_ACTIVE,
                        label="active leads")
        ax.fill_between(t, pol, 0.5, where=(pol < 0.5), alpha=0.2, color=C_PASSIVE,
                        label="passive leads")
        ax.axhline(0.5, color="grey", lw=0.8, ls="--", label="random (0.5)")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8)
    ax.set_title("Cluster Polarity  (active leads)", fontsize=10, fontweight="bold")
    _style_ax(ax, ylabel="polarity  [0 – 1]")

    # --- [1,1] Director alignment vs time ---
    ax = axes[1, 1]
    if "alignment" in series and np.any(np.isfinite(series["alignment"])):
        ax.plot(t, series["alignment"], color=C_PASSIVE, lw=1.5)
        ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.set_title("Director Alignment", fontsize=10, fontweight="bold")
    _style_ax(ax, ylabel="⟨cos θ⟩  (migration axis)")

    # --- [2,0] MSD vs time (log-log) ---
    ax = axes[2, 0]
    if "msd_mean" in series:
        msd = series["msd_mean"]
        valid = np.isfinite(msd) & (msd > 0) & (t > 0)
        if np.sum(valid) > 2:
            ax.loglog(t[valid], msd[valid], color=C_MSD, lw=1.5)
            # Slope-1 reference (diffusive)
            t_ref = t[valid]
            ax.loglog(t_ref, msd[valid][0] * (t_ref / t_ref[0]),
                      "k--", lw=0.8, alpha=0.5, label="slope 1 (diffusive)")
            ax.legend(frameon=False, fontsize=8)
    ax.set_title("MSD  (migratory cells)", fontsize=10, fontweight="bold")
    _style_ax(ax, ylabel="MSD  (sim units²)")

    # --- [2,1] Shape / spread ---
    ax = axes[2, 1]
    has_voro = "c_index" in series and np.any(np.isfinite(series["c_index"]))
    if has_voro:
        ax.plot(t, series["c_index"], color=C_SHAPE, lw=1.5,
                label="circularity  4πA/P²")
        if "ab_ratio" in series and np.any(np.isfinite(series["ab_ratio"])):
            ax.plot(t, series["ab_ratio"], color=C_MSD, lw=1.5, ls="--",
                    label="ellipse  a/b")
        ax.axhline(1.0, color="grey", lw=0.5, ls=":")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("Cluster Shape  (Voronoi)", fontsize=10, fontweight="bold")
        _style_ax(ax, ylabel="shape index")
    elif "spread" in series:
        ax.plot(t, series["spread"], color=C_SHAPE, lw=1.5)
        ax.set_title("Cluster Spread", fontsize=10, fontweight="bold")
        _style_ax(ax, ylabel="mean radial spread")

    out = Path(args.output) if args.output else _default_plot_path(ts_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Plot: {out}")
    return 0


# ---------------------------------------------------------------------------
# run_plot_ensemble — mean±SEM bar/line plots from ensemble_aggregates.json
# ---------------------------------------------------------------------------

def run_plot_ensemble(args: argparse.Namespace) -> int:
    """Plot ensemble mean±SEM from aggregates.json.

    Produces a multi-panel figure showing mean ± SEM for each scalar metric.
    Matches the style of plot_2D_metrics_basic_voronoi_vary_gamma_pub.py.
    """
    agg_path = Path(args.aggregates)
    if not agg_path.exists():
        print(f"Not found: {agg_path}")
        return 1

    with agg_path.open() as f:
        aggs = json.load(f)

    panels = [
        ("cluster_disp_x",          "Mean displacement x",          "displacement"),
        ("closeness",               "Closeness (spread at end)",    "closeness"),
        ("polarity_final",          "Polarity (active leads)",      "polarity [0–1]"),
        ("c_index_final",           "Circularity  4πA/P²",         "circularity"),
        ("ab_ratio_mean",           "Ellipse a/b ratio",            "a/b"),
        ("p_active_mean",           "Active shape index",           "p̄ₐ"),
        ("p_passive_mean",          "Passive shape index",          "p̄ₚ"),
        ("p_background_mean",       "Background shape index",       "p̄_bg"),
        ("a_bound_num",             "Active cells at boundary",     "# cells"),
        ("p_bound_num",             "Passive cells at boundary",    "# cells"),
        ("t1_front_creations_mean", "T1 front creations / step",    "count"),
        ("t1_rear_gaps_mean",       "T1 rear gaps / step",          "count"),
        ("t1_homotypic_gains_mean", "T1 homotypic gains / step",    "count"),
    ]
    available = [(k, ti, yl) for k, ti, yl in panels if k in aggs]
    if not available:
        print("No recognised metrics in aggregates file.")
        return 1

    ncols = 2
    nrows = math.ceil(len(available) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), constrained_layout=True)
    axes_flat = np.array(axes).flatten()
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51",
              "#6d6875", "#b5838d", "#c77dff", "#0077b6", "#00b4d8",
              "#90e0ef", "#caf0f8", "#023e8a"]

    for ai, (key, title, ylabel) in enumerate(available):
        ax = axes_flat[ai]
        e = aggs[key]
        mean_v = e["mean"]
        sem_v  = e.get("sem", e.get("std", 0))
        n      = e.get("n", "?")
        ax.bar([0], [mean_v], yerr=[sem_v], color=colors[ai % len(colors)],
               capsize=5, width=0.5, error_kw={"elinewidth": 1.5, "capthick": 1.5})
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks([])
        ax.text(0, mean_v + sem_v * 1.1, f"{mean_v:.3f}±{sem_v:.3f}\n(n={n})",
                ha="center", va="bottom", fontsize=7.5)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    for ax in axes_flat[len(available):]:
        ax.set_visible(False)

    out = (Path(args.output) if getattr(args, "output", None)
           else Path("RESULTS") / "PLOTS" / "ensemble_summary_plot.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[DONE] Ensemble plot: {out}")
    return 0


# ---------------------------------------------------------------------------
# Movie rendering — Voronoi-patch style matching make_basic_voronoi_movie.py
# ---------------------------------------------------------------------------

# Cell face colours (exact hex from user spec):
#   type 0 = background/surrounding tissue  #e9e9ff (lavender)
#   type 1 = active leader cells            #06b780 (teal-green)
#   type 2 = passive follower cells         #ffc7b  -> #ffcc7b (amber/gold)
#   type 5 = edge/boundary cells            #ccccff (light blue-lavender)
_FACE_COLOR = {
    0: "#e9e9ff",    # surrounding tissue — lavender
    1: "#06b780",    # active leader — teal green
    2: "#ffcc7b",    # passive follower — amber
    3: "#aad4f5",
    4: "#7ec8e3",
    5: "#ccccff",    # edge/boundary cells — light lavender-blue
}
# Cell edge colours: non-edge cells use low-opacity dark grey; edge cells use slightly darker lavender
_EDGE_COLOR = {
    0: (0.29, 0.29, 0.29, 0.25),   # background — dark grey, low opacity
    1: (0.02, 0.55, 0.38, 0.9),    # active — darker teal
    2: (0.75, 0.55, 0.15, 0.9),    # passive — darker amber
    3: (0.3,  0.6,  0.8,  0.7),
    4: (0.2,  0.55, 0.7,  0.7),
    5: (0.55, 0.55, 0.8,  0.9),    # edge cells — same lavender but more opaque
}


def _face_color(tv: int) -> str:
    return _FACE_COLOR.get(int(tv), "white")


def _edge_color(tv: int) -> tuple:
    return _EDGE_COLOR.get(int(tv), (0.3, 0.3, 0.3, 0.8))


def _avm_polygons(data: "SimulationData", t: int, lx: float, ly: float):
    """Build AVM cell polygons from vertex-cell connectivity at timestep t.

    Returns list of vertex arrays [(k,2), ...] one per cell, or None if data missing.
    Uses VertexCellNeighbors stored in data.vert_cell_neigh (shape Nv*3, T).
    Vertex positions in data.vx/vy (Nv, T).
    """
    if (data.vx is None or data.vy is None or data.vert_cell_neigh is None):
        return None

    nv = data.vx.shape[0]
    nc = data.x.shape[0]
    nv3 = data.vert_cell_neigh.shape[0]  # Nv * 3

    vx_t = data.vx[:, t]
    vy_t = data.vy[:, t]
    vcn_t = data.vert_cell_neigh[:, t].reshape(nv, 3)  # (Nv, 3) cell indices per vertex

    polys = []
    for cell_idx in range(nc):
        # Find vertices that belong to this cell
        cell_verts = np.where(np.any(vcn_t == cell_idx, axis=1))[0]
        if len(cell_verts) < 3:
            polys.append(None)
            continue
        cx = vx_t[cell_verts].copy()
        cy = vy_t[cell_verts].copy()
        # Unwrap periodic images relative to first vertex
        for k in range(1, len(cx)):
            if cx[k] - cx[0] > lx / 2:
                cx[k] -= lx
            elif cx[k] - cx[0] < -lx / 2:
                cx[k] += lx
            if cy[k] - cy[0] > ly / 2:
                cy[k] -= ly
            elif cy[k] - cy[0] < -ly / 2:
                cy[k] += ly
        # Sort vertices by angle around centroid
        centroid = np.array([np.mean(cx), np.mean(cy)])
        angles = np.arctan2(cy - centroid[1], cx - centroid[0])
        order = np.argsort(angles)
        polys.append(np.column_stack([cx[order], cy[order]]))

    return polys


def _render_frame(
    t: int,
    data: "SimulationData",
    lx: float,
    ly: float,
    migratory_types: Sequence[int],
    snap_path: Path,
    dpi: int = 200,
) -> None:
    """Render one frame: polygon patches (dark bg) with director arrows and overlay text.

    SPV: freud Voronoi tessellation from cell centroid positions.
    AVM: polygons reconstructed from VertexCellNeighbors connectivity.
    Colors: active (type 1) = orange, passive (type 2) = green, background (type 0) = white.
    Falls back to circle-patches when Voronoi/vertex data unavailable.
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPolygon, Circle

    n = data.x.shape[0]
    types_t = data.cell_type[t, :]

    # Estimate per-cell radius for circle fallback
    cell_area = (lx * ly) / max(n, 1)
    cell_r = math.sqrt(cell_area / math.pi) * 0.95

    with plt.style.context("default"):
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="#e9e9ff")
        ax.set_facecolor("#e9e9ff")
        ax.set_xlim(-lx * 0.01, lx * 1.01)
        ax.set_ylim(-ly * 0.01, ly * 1.01)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        drawn = False

        if data.model_type == "AVM":
            # Try AVM vertex polygons first
            avm_polys = _avm_polygons(data, t, lx, ly)
            if avm_polys is not None:
                patches = []
                fcolors_list = []
                ecolors_list = []
                for i, poly in enumerate(avm_polys):
                    if poly is not None and len(poly) >= 3:
                        patches.append(MplPolygon(poly))
                        fcolors_list.append(_face_color(int(types_t[i])))
                        ecolors_list.append(_edge_color(int(types_t[i])))
                if patches:
                    pc = PatchCollection(patches, facecolors=fcolors_list,
                                         edgecolors=ecolors_list, linewidth=0.75)
                    ax.add_collection(pc)
                    ax.plot(data.x[:, t], data.y[:, t], ".", markersize=0.8,
                            color="black", zorder=3)
                    drawn = True

        if not drawn and _HAVE_FREUD:
            # SPV or AVM fallback: Voronoi from cell centroids
            pos3d = np.column_stack([data.x[:, t], data.y[:, t], np.zeros(n)]).astype(np.float32)
            box_sys = freud.box.Box(Lx=lx, Ly=ly, is2D=True)
            voro = freud.locality.Voronoi()
            voro.compute((box_sys, pos3d))

            patches = [MplPolygon(poly[:, :2]) for poly in voro.polytopes]
            fcolors = [_face_color(int(types_t[i])) for i in range(n)]
            ecolors = [_edge_color(int(types_t[i])) for i in range(n)]
            pc = PatchCollection(patches, facecolors=fcolors, edgecolors=ecolors, linewidth=0.75)
            ax.add_collection(pc)
            ax.plot(data.x[:, t], data.y[:, t], ".", markersize=0.8, color="black", zorder=3)
            drawn = True

        if not drawn:
            # Circle-patch fallback — no freud and no vertex data
            patches_fb = [Circle((data.x[i, t], data.y[i, t]), cell_r) for i in range(n)]
            fcolors_fb = [_face_color(int(types_t[i])) for i in range(n)]
            ecolors_fb = [_edge_color(int(types_t[i])) for i in range(n)]
            pc_fb = PatchCollection(patches_fb, facecolors=fcolors_fb,
                                    edgecolors=ecolors_fb, linewidth=0.5)
            ax.add_collection(pc_fb)

        # Red director arrows for active cells (type 1)
        if data.directors is not None and t < data.directors.shape[0]:
            a_idx = np.where(types_t == 1)[0]
            if len(a_idx):
                theta = data.directors[t, a_idx]
                ax.quiver(
                    data.x[a_idx, t], data.y[a_idx, t],
                    np.cos(theta), np.sin(theta),
                    color="red", angles="xy", scale_units="xy", scale=1.0,
                    headwidth=4, headlength=4, width=0.004, zorder=4,
                )

        # Per-frame text overlay: time, polarity, speed
        tval = float(data.time[t])
        pol_str = spd_str = "nan"

        mig_mask = np.isin(types_t, migratory_types)
        if np.any(mig_mask):
            mig_idx = np.where(mig_mask)[0]
            try:
                pol = _compute_polarity(
                    data.x, data.y, data.cell_type, data.directors, lx, ly, t)
                pol_str = f"{pol:.3f}"
            except Exception:
                pass
            if t > 0:
                dt_val = float(data.time[t] - data.time[t - 1])
                if dt_val > 1e-9:
                    dx = float(np.mean(data.x[mig_idx, t]) - np.mean(data.x[mig_idx, t - 1]))
                    dy = float(np.mean(data.y[mig_idx, t]) - np.mean(data.y[mig_idx, t - 1]))
                    spd_str = f"{math.sqrt(dx**2 + dy**2) / dt_val:.4f}"

        ax.text(0.02, 0.97, f"t={tval:.3f}  polarity={pol_str}  speed={spd_str}",
                transform=ax.transAxes, fontsize=8, verticalalignment="top", color="black",
                bbox=dict(facecolor="white", alpha=0.7, pad=3))

        plt.tight_layout(pad=0)
        fig.savefig(str(snap_path), dpi=dpi, format="jpeg")
        plt.close(fig)


def run_movie(args: argparse.Namespace) -> int:
    """Generate movie: per-frame JPEG snapshots assembled via ffmpeg (GIF fallback).

    Renders Voronoi polygon patches matching paper figure style:
    surrounding tissue (type 0) = lavender/blue, active cells (type 1) = green,
    passive cells (type 2) = orange, edge cells (type 5) = cornflower blue.
    Red director arrows for active cells. Falls back to circle-patch style without freud.
    """
    nc_path = Path(args.input)
    if not nc_path.exists():
        print(f"Not found: {nc_path}")
        return 1

    types  = parse_type_list(args.migratory_types)
    data   = load_nc_file(nc_path)

    t0     = max(0, args.start)
    t1     = min(data.time.shape[0], args.end if args.end is not None else data.time.shape[0])
    stride = max(1, args.stride)
    frames = list(range(t0, t1, stride))
    if not frames:
        print("No frames selected.")
        return 1

    out = (Path(args.output) if args.output
           else Path("RESULTS") / "MOVIES" / f"{nc_path.stem}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    lx = float(np.nanmax(data.lx)) if np.any(np.isfinite(data.lx)) else float(np.nanmax(data.x))
    ly = float(np.nanmax(data.ly)) if np.any(np.isfinite(data.ly)) else float(np.nanmax(data.y))
    frame_dpi = getattr(args, "dpi", 200)

    # Temp JPEG frames — include PID to avoid collisions when running in parallel
    snap_dir = out.parent / f"_tmp_{out.stem}_{os.getpid()}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    for fi, t in enumerate(frames):
        _render_frame(t, data, lx, ly, types, snap_dir / f"{fi:06d}.jpeg", dpi=frame_dpi)
        if (fi + 1) % 10 == 0 or fi == 0:
            print(f"  frame {fi + 1}/{len(frames)}", end="\r", flush=True)
    print()

    # Assemble with ffmpeg → GIF fallback
    import shutil as _sh
    if out.suffix.lower() != ".gif" and _sh.which("ffmpeg"):
        pattern = str(snap_dir / "%06d.jpeg")
        cmd = (
            f'ffmpeg -y -framerate {args.fps} -i "{pattern}" '
            f'-c:v libx264 -pix_fmt yuv420p '
            f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" '
            f'"{out}" -loglevel error'
        )
        if os.system(cmd) != 0:
            print("[WARN] ffmpeg failed — falling back to GIF.")
            out = out.with_suffix(".gif")
    else:
        if out.suffix.lower() != ".gif":
            print("[WARN] ffmpeg not found — saving GIF.")
        out = out.with_suffix(".gif")

    if out.suffix.lower() == ".gif":
        try:
            from PIL import Image as _PilImg
            imgs = [_PilImg.open(str(snap_dir / f"{fi:06d}.jpeg")).convert("RGB")
                    for fi in range(len(frames))
                    if (snap_dir / f"{fi:06d}.jpeg").exists()]
            if imgs:
                imgs[0].save(str(out), save_all=True, append_images=imgs[1:],
                             duration=int(1000 / max(1, args.fps)), loop=0)
        except ImportError:
            print("[WARN] Pillow not installed — cannot save GIF.")
            _sh.rmtree(str(snap_dir), ignore_errors=True)
            return 1

    _sh.rmtree(str(snap_dir), ignore_errors=True)
    print(f"[DONE] Movie: {out}")
    return 0


# ---------------------------------------------------------------------------
# run_all
# ---------------------------------------------------------------------------

def run_all(args: argparse.Namespace) -> int:
    code = run_metrics(argparse.Namespace(
        input=args.input, output_dir=args.output_dir,
        ensemble_count=args.ensemble_count, migratory_types=args.migratory_types,
        voronoi=getattr(args, "voronoi", False), t1=getattr(args, "t1", False),
    ))
    if code != 0:
        return code

    src      = Path(args.input)
    scenario = src.parent.name or "default"
    ts_path  = Path(args.output_dir) / scenario / f"{src.stem}_timeseries.csv"
    if ts_path.exists():
        plot_out = getattr(args, "plot_output", None) or str(_default_plot_path(ts_path))
        run_plot(argparse.Namespace(timeseries=str(ts_path), output=plot_out))

    # Prefix movie/plot output with scenario folder name so SPV and AVM don't collide
    movie_stem = f"{scenario}_{src.stem}"
    return run_movie(argparse.Namespace(
        input=args.input,
        output=getattr(args, "movie_output", None) or str(
            Path("RESULTS") / "MOVIES" / f"{movie_stem}.mp4"),
        fps=args.fps, start=args.start, end=args.end,
        stride=args.stride, migratory_types=args.migratory_types,
        show_directors=getattr(args, "show_directors", False),
        dpi=getattr(args, "dpi", 200),
    ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CellGPU post-processing: metrics, plots, and movies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # metrics
    pm = sub.add_parser("metrics", help="Compute metrics from .nc file(s)")
    pm.add_argument("--input", required=True,
                    help="Path, glob pattern, or {i} ensemble pattern")
    pm.add_argument("--output-dir", default="RESULTS/PROCESSED_DATA")
    pm.add_argument("--ensemble-count", type=int, default=100)
    pm.add_argument("--migratory-types", default="1,2")
    pm.add_argument("--voronoi", action="store_true",
                    help="Compute Voronoi shape metrics (requires freud)")
    pm.add_argument("--t1", action="store_true",
                    help="Detect T1 transitions (requires freud, slow)")
    pm.set_defaults(func=run_metrics)

    # plot (single file)
    pp = sub.add_parser("plot", help="Plot a single timeseries CSV")
    pp.add_argument("--timeseries", required=True)
    pp.add_argument("--output", default=None)
    pp.set_defaults(func=run_plot)

    # plot-ensemble
    pe = sub.add_parser("plot-ensemble", help="Ensemble mean±SEM from aggregates.json")
    pe.add_argument("--aggregates", required=True, help="Path to ensemble_aggregates.json")
    pe.add_argument("--output", default=None)
    pe.set_defaults(func=run_plot_ensemble)

    # movie
    pmov = sub.add_parser("movie", help="Create movie from .nc file")
    pmov.add_argument("--input", required=True)
    pmov.add_argument("--output", default=None)
    pmov.add_argument("--fps", type=int, default=10)
    pmov.add_argument("--start", type=int, default=0)
    pmov.add_argument("--end", type=int, default=None)
    pmov.add_argument("--stride", type=int, default=1)
    pmov.add_argument("--migratory-types", default="1,2")
    pmov.add_argument("--show-directors", action="store_true")
    pmov.add_argument("--dpi", type=int, default=200)
    pmov.set_defaults(func=run_movie)

    # run-all
    pra = sub.add_parser("run-all", help="Metrics + plot + movie for one file")
    pra.add_argument("--input", required=True)
    pra.add_argument("--output-dir", default="RESULTS/PROCESSED_DATA")
    pra.add_argument("--ensemble-count", type=int, default=100)
    pra.add_argument("--migratory-types", default="1,2")
    pra.add_argument("--voronoi", action="store_true")
    pra.add_argument("--t1", action="store_true")
    pra.add_argument("--plot-output", default=None)
    pra.add_argument("--movie-output", default=None)
    pra.add_argument("--fps", type=int, default=10)
    pra.add_argument("--start", type=int, default=0)
    pra.add_argument("--end", type=int, default=None)
    pra.add_argument("--stride", type=int, default=1)
    pra.add_argument("--show-directors", action="store_true")
    pra.add_argument("--dpi", type=int, default=200)
    pra.set_defaults(func=run_all)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args   = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
