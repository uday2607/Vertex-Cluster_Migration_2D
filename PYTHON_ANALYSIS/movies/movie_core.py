"""
Movie renderer for CellGPU trajectories (SPV and AVM) -- single panel, plus the
helpers shared with make_grid_movie.py (loaders, palette, equilibration detection).

Handles both models, auto-detected from the .nc variables:
  SPV (Voronoi)  -- variables pos/type/director, polygons from a freud
                    Voronoi tessellation of the cell centers
  AVM (vertex)   -- variables cellPositions/cellType/Vneighs/
                    VertexCellNeighbors, polygons from explicit vertices

Palette (user-specified 2026-10-28):
  background cell (type 0)  face #E9E9FF, edge #4A4A4A @ 50% opacity
  wall cell       (type 5)  face #CCCCFF, edge #4A4A4A @ 100% opacity
  active cell     (type 1)  face #06B780, edge #4A4A4A @ 50% opacity
  passive cell    (type 2)  face #FFCB7B, edge #4A4A4A @ 50% opacity
  motility arrows           red, ONLY after equilibration has ended

Equilibration is detected from the saved time axis: the last equilibration
frame and the first production frame share the same time stamp (e.g. frames
5 and 6 both t=1.0); walls (type 5) also appear at that frame. Equilibration
frames play fast (EQUIL_FRAME_SEC each), production frames slow
(PROD_FRAME_SEC each).

Run from the project root:
    python PYTHON_ANALYSIS/movies/movie_core.py <input.nc> <output.mp4>
"""
import os
import sys
import shutil
import subprocess
import tempfile
import multiprocessing
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.colors import to_rgba
from scipy.io import netcdf_file
import freud
import numba as nb

FFMPEG = shutil.which("ffmpeg")
if FFMPEG is None:
    raise SystemExit("ffmpeg not found on PATH (e.g. `conda install ffmpeg` or `apt install ffmpeg`)")

BG_FACE = "#E9E9FF"
WALL_FACE = "#CCCCFF"
ACTIVE_FACE = "#06B780"
PASSIVE_FACE = "#FFCB7B"
EDGE = "#4A4A4A"
ARROW_COLOR = "red"
FIG_BG = "white"
EQUIL_FRAME_SEC = 0.04    # equilibration frames: fast
PROD_FRAME_SEC = 0.5      # production frames: slow

FACE = {0: BG_FACE, 5: WALL_FACE, 1: ACTIVE_FACE, 2: PASSIVE_FACE}


def production_start(time):
    """Index of the first production frame: the first frame whose time
    stamp (nearly) repeats the previous one (end-of-equilibration
    duplicate; the gap is 0 or ~2e-4 depending on the file, versus 0.2
    between equilibration frames)."""
    t = np.asarray(time).ravel()
    dup = np.where(np.diff(t) < 0.01)[0]
    return int(dup[0] + 1) if len(dup) else 0


# ---- AVM polygon construction (from make_vertex_movie.py) ----
@nb.jit(nopython=True, nogil=True)
def adjust_coordinates(cell_x, cell_y, Lx, Ly):
    for i in range(len(cell_x)):
        for j in range(i + 1, len(cell_x)):
            dist_x = np.abs(cell_x[i] - cell_x[j])
            dist_y = np.abs(cell_y[i] - cell_y[j])
            if dist_x > Lx / 2:
                if cell_x[i] > cell_x[j]:
                    cell_x[j] += Lx
                else:
                    cell_x[j] -= Lx
            if dist_y > Ly / 2:
                if cell_y[i] > cell_y[j]:
                    cell_y[j] += Ly
                else:
                    cell_y[j] -= Ly


def vertex_polygons(vx, vy, ver_cell, total, Lx, Ly):
    """Polygon per cell from the cell's vertices, ordered by angle about the
    centroid. Returns (polygons, cell indices)."""
    polys, cells = [], []
    for cell in range(total):
        cv = np.where(ver_cell == cell)[0]
        if len(cv) == 0:
            continue
        cx = vx[cv].copy()
        cy = vy[cv].copy()
        adjust_coordinates(cx, cy, Lx, Ly)
        pts = np.stack([cx, cy], axis=1)
        c = pts.mean(axis=0)
        pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
        polys.append(Polygon(pts))
        cells.append(cell)
    return polys, cells


def render_frame(args):
    (k, frame_dir, model, centers, ctype, director, Lx, Ly, v0, draw_arrows, vtx) = args
    ncell = centers.shape[0]
    xy = np.zeros((ncell, 3))
    xy[:, 0] = centers[:, 0] - np.floor(centers[:, 0] / Lx) * Lx
    xy[:, 1] = centers[:, 1] - np.floor(centers[:, 1] / Ly) * Ly

    if model == "spv":
        box = freud.box.Box(Lx=Lx, Ly=Ly, is2D=True)
        voro = freud.locality.Voronoi()
        voro.compute((box, xy))
        patches = [Polygon(p[:, :2]) for p in voro.polytopes]
        cell_of_patch = np.arange(ncell)
    else:
        vx, vy, ver_cell = vtx
        patches, cell_of_patch = vertex_polygons(vx, vy, ver_cell, ncell, Lx, Ly)
        cell_of_patch = np.array(cell_of_patch)

    pt = ctype[cell_of_patch]
    faces = [to_rgba(FACE.get(int(c), BG_FACE)) for c in pt]
    edges = [to_rgba(EDGE, 1.0 if int(c) == 5 else 0.5) for c in pt]

    fig = plt.figure(figsize=(6, 6 * Ly / Lx), facecolor=FIG_BG)
    ax = fig.add_subplot()
    ax.add_collection(PatchCollection(patches, facecolors=faces, edgecolors=edges, linewidth=0.75))

    if draw_arrows:
        sel = np.where((ctype == 1) | (ctype == 2))[0]
        ax.quiver(xy[sel, 0], xy[sel, 1],
                  v0 * np.cos(director[sel]), v0 * np.sin(director[sel]),
                  color=ARROW_COLOR, angles="xy", scale_units="xy", scale=1, width=0.004)

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    plt.subplots_adjust(0, 0, 1, 1)
    fig.savefig(os.path.join(frame_dir, f"{k:05d}.png"), dpi=300, facecolor=FIG_BG)
    plt.close(fig)


def load(nc_path):
    d = netcdf_file(nc_path, "r", mmap=False)
    v = {k: d.variables[k][:] * 1 for k in d.variables}
    d.close()
    time = v["time"]
    box = v["BoxMatrix"]
    Lx, Ly = float(box[0, 0]), float(box[0, 3])
    if "cellPositions" in v:
        model = "avm"
        centers = v["cellPositions"].reshape(len(time), -1, 2)
        ctype = v["cellType"]
        T = ctype.shape[0]
        a = float(box.max())
        verts = v["pos"].reshape(T, -1, 2)
        vertex_data = [(verts[k][:, 0] - np.floor(verts[k][:, 0] / a) * a,
                        verts[k][:, 1] - np.floor(verts[k][:, 1] / a) * a,
                        v["VertexCellNeighbors"][k].reshape(-1, 3).astype(int)) for k in range(T)]
    else:
        model = "spv"
        centers = v["pos"].reshape(len(time), -1, 2)
        ctype = v["type"]
        vertex_data = [None] * len(time)
    return model, centers, ctype, v["director"], time, Lx, Ly, vertex_data


def make_movie(nc_path, out_path):
    model, centers, ctype, director, time, Lx, Ly, vertex_data = load(nc_path)
    parts = os.path.basename(nc_path).split("_")
    v0 = float(parts[parts.index("va") + 1])
    T = len(time)
    p0 = production_start(time)
    print(f"{model.upper()}: {T} frames, production starts at frame {p0} "
          f"(t={float(np.ravel(time)[p0]):.2f}); {T - p0} frames with arrows")

    frame_dir = tempfile.mkdtemp(prefix="movie_frames_")
    try:
        jobs = [(k, frame_dir, model, centers[k], ctype[k], director[k], Lx, Ly, v0, k >= p0, vertex_data[k])
                for k in range(T)]
        with multiprocessing.Pool(processes=min(8, T)) as pool:
            pool.map(render_frame, jobs)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        lst = os.path.join(frame_dir, "files.txt")
        with open(lst, "w") as f:
            for k in range(T):
                f.write(f"file '{k:05d}.png'\nduration {EQUIL_FRAME_SEC if k < p0 else PROD_FRAME_SEC}\n")
            f.write(f"file '{T - 1:05d}.png'\n")
        subprocess.check_call([
            FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
            "-r", "30", "-c:v", "libx264",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", out_path])
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    make_movie(sys.argv[1], sys.argv[2])
