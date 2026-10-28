"""
Polished grid movies (any rows x cols of SPV / AVM / Nodal simulations) for the paper movies. Reuses the loaders, palette and
equilibration detection of movie_core.py. One script for every figure: each movie is a small JSON file in specs/.

Polish:
  * cells are drawn with periodic images and the view is a clean RECTANGLE
    cut out of the box, so no cell is ever half-drawn at the box edge
  * rounded panels, column headers (regime), row labels (model, v0),
    per-panel gamma tag, title + subtitle
  * time readout ("Equilibration", then t = 0 ... end of the run, counted
    from the start of production) with a progress bar, and a legend
  * motility arrows only after equilibration; equilibration plays fast,
    production slow (same constants as movie_core.py)

Run from the project root (the example trajectories are in DATA/EXAMPLES):
    python PYTHON_ANALYSIS/movies/make_grid_movie.py fig3              # -> RESULTS/movies/fig3.mp4
    python PYTHON_ANALYSIS/movies/make_grid_movie.py fig5b --out my_movie.mp4
    python PYTHON_ANALYSIS/movies/make_grid_movie.py --spec my_spec.json --out my_movie.mp4
"""
import os
import sys
import glob
import shutil
import subprocess
import tempfile
import multiprocessing
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from matplotlib.colors import to_rgba

sys.path.insert(0, os.path.dirname(__file__))
import movie_core as mm

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # quiet font-fallback chatter


def _setup_fonts():
    """Prefer Lato (used for the paper movies); fall back to DejaVu Sans if it is not installed."""
    from matplotlib import font_manager as fm
    try:
        fm.findfont("Lato", fallback_to_default=False)
    except ValueError:
        return
    mpl.rcParams["font.family"] = ["Lato", "DejaVu Sans"]
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = "Lato"
    mpl.rcParams["mathtext.it"] = "Lato:italic"
    mpl.rcParams["mathtext.bf"] = "Lato:bold"


_setup_fonts()

INK = "#23232F"
MUTED = "#6E6E80"
RULE = "#D9D9E6"
FIG_BG = "#FFFFFF"

# layout, inches
PANEL_W = 4.6
CROP_H_UNITS = 12.0           # rectangle cut out of the 20 x 20 box (full width)
GAP = 0.28
LEFT = 1.05
RIGHT = 0.45
TOP_TITLE = 1.05
COL_HDR = 0.5
FOOT = 1.15
DPI = 150
WALL_X_MAX = 1.9              # any non-active cell drawn with its centre at x < this is a wall cell
WALL_OVERHANG = 1.5           # ... down to this far left of the box edge (the wall straddles the periodic seam)
WALL_MEMBER_MAX = 3.0         # wall-type cells keep the wall colour up to here (their centres reach x = 2.0)
DEEP = "#0A6B4D"              # cluster-tissue interface line
EDGE_CMAP = mpl.colormaps["plasma"]   # interface edge colour by edge length (short = dark, long = bright)
EDGE_LMAX = 1.2                       # edge length (cell diameter ~ 1.1) at the bright end

_STATE = {}
PANEL_FACE2 = {}   # file path -> colour of type-2 cells in that panel (override)


class Speeds(float):
    """Leader speed (a float, as before) carrying the follower speed in .u0 so arrows can follow each cell type's own speed."""
    u0 = None
    u3 = None
    face2 = None



def load_panel(f, last_frame=None):
    """mm.load(...) plus v0 (from the file name) and the wall identity set:
    cells that are type 5 or sit at x < WALL_X_MAX (non-active) at the start of
    production, applied to every frame so the wall never flickers or goes
    uncolored during equilibration."""
    parts = os.path.basename(f).split("_")
    L = mm.load(f)
    if last_frame is not None:                       # keep frames 0..last_frame only
        sl = slice(0, last_frame + 1)
        L = (L[0], L[1][sl], L[2][sl], L[3][sl], L[4][sl], L[5], L[6], L[7][sl])
    model, centers, ctype, director, time, Lx, Ly, vd = L
    p0 = mm.production_start(time)
    x0 = centers[p0][:, 0] % Lx
    ident = (ctype[p0] == 5) | ((x0 < WALL_X_MAX) & (ctype[p0] != 1) & (ctype[p0] != 2) & (ctype[p0] != 3))
    if "v1" in parts:                                  # three-cell files: one speed per cell type
        sp = Speeds(float(parts[parts.index("v1") + 1]))
        sp.u0 = float(parts[parts.index("v2") + 1])
        sp.u3 = float(parts[parts.index("v3") + 1])
    else:
        sp = Speeds(float(parts[parts.index("va") + 1]))
        sp.u0 = float(parts[parts.index("vp") + 1]) if "vp" in parts else None
    sp.face2 = PANEL_FACE2.get(f)
    return L + (sp, ident)


def _init(panel_files, last_frame=None, face=None, face2=None):
    # colour overrides are passed explicitly so they also reach worker processes that do not fork (macOS, Windows)
    mm.FACE.update(face or {})
    PANEL_FACE2.update(face2 or {})
    _STATE["panels"] = [load_panel(f, last_frame) for f in panel_files]


def _cell_polygons(model, centers_k, ctype_k, vtx_k, Lx, Ly):
    """List of (xy array, cell index) polygons for one frame."""
    ncell = centers_k.shape[0]
    if model == "spv":
        import freud
        xy = np.zeros((ncell, 3))
        xy[:, 0] = centers_k[:, 0] - np.floor(centers_k[:, 0] / Lx) * Lx
        xy[:, 1] = centers_k[:, 1] - np.floor(centers_k[:, 1] / Ly) * Ly
        voro = freud.locality.Voronoi()
        voro.compute((freud.box.Box(Lx=Lx, Ly=Ly, is2D=True), xy))
        return [(p[:, :2], i) for i, p in enumerate(voro.polytopes)]
    vx, vy, ver_cell = vtx_k
    patches, cells = mm.vertex_polygons(vx, vy, ver_cell, ncell, Lx, Ly)
    return [(p.get_xy(), c) for p, c in zip(patches, cells)]


def _rounded(ax, radius=0.035, **kw):
    bb = ax.get_position()
    w, h = bb.width * _STATE["figsize"][0], bb.height * _STATE["figsize"][1]
    return FancyBboxPatch((0, 0), 1, 1, boxstyle=f"round,pad=0,rounding_size={radius}",
                          mutation_aspect=w / h, transform=ax.transAxes, **kw)


def _interface_segments(polys, ctype_k, Lx, Ly):
    """Edges of the active cluster that border a non-cluster cell: an edge
    used by exactly one type-1 polygon (edges shared by two cluster cells are
    internal). Keys are taken modulo the box so the seam cannot split a pair."""
    cnt, seg = {}, {}
    for xy, c in polys:
        if int(ctype_k[c]) != 1:
            continue
        n = len(xy)
        for i in range(n):
            a, b = xy[i], xy[(i + 1) % n]
            ka = (round(a[0] % Lx, 3), round(a[1] % Ly, 3))
            kb = (round(b[0] % Lx, 3), round(b[1] % Ly, 3))
            key = (ka, kb) if ka <= kb else (kb, ka)
            cnt[key] = cnt.get(key, 0) + 1
            seg[key] = (a, b)
    return [seg[k] for k, v in cnt.items() if v == 1]


def _wall_copy(ct, is_member, xd):
    """Is a non-active cell DRAWN with its centre at x = xd a wall cell? (Position of the drawn copy decides, so
    cells shifted across the periodic seam get the colour of the place where they are shown.)"""
    if ct in (1, 2, 3):
        return False
    return (-WALL_OVERHANG <= xd < WALL_X_MAX) or (bool(is_member) and -WALL_OVERHANG <= xd < WALL_MEMBER_MAX)


def auto_x_window(panels, margin=0.03):
    """x-range of the rectangular cut that never slices through a cell of the 'wrong' colour, found from the data.
    The wall straddles the periodic seam, so (i) at the LEFT border only wall-coloured cells may be cut: the border
    must lie right of every non-wall copy that sits left of the wall; (ii) at the RIGHT border no wall-coloured cell
    may appear: the border must lie left of every wall-coloured polygon shifted by +Lx. All frames and panels are
    scanned, so one window fits the whole movie."""
    left_max, right_min = -np.inf, np.inf
    for P in panels:
        model, centers, ctype, director, time, Lx, Ly, vdata, v0, wall_ident = P
        for k in range(len(time)):
            polys = _cell_polygons(model, centers[k], ctype[k], vdata[k], Lx, Ly)
            xw = centers[k][:, 0] % Lx
            member = (ctype[k] == 5) | wall_ident
            for xy, c in polys:
                ct = int(ctype[k][c])
                if ct in (1, 2, 3):
                    continue
                pmin, pmax = xy[:, 0].min(), xy[:, 0].max()
                cxp = float(xy[:, 0].mean())
                for sx in (-Lx, 0.0, Lx):
                    xd = cxp + sx
                    if _wall_copy(ct, member[c], xd):
                        right_min = min(right_min, pmin + sx + Lx)
                    elif xd < -WALL_OVERHANG:
                        left_max = max(left_max, pmax + sx)
    return left_max + margin, right_min - margin


def draw_panel(ax, P, k, crop_x, crop_y, p0, interface=False, extras=None, periodic_x=True, axis_ticks=None, wall=True):
    model, centers, ctype, director, time, Lx, Ly, vdata, v0, wall_ident = P
    polys = _cell_polygons(model, centers[k], ctype[k], vdata[k], Lx, Ly)
    y0, y1 = crop_y
    xw = centers[k][:, 0] % Lx
    member = (ctype[k] == 5) | wall_ident
    sxs = (-Lx, 0.0, Lx) if periodic_x else (0.0,)
    verts, faces, edges = [], [], []
    for xy, c in polys:
        cxp = float(xy[:, 0].mean())   # where THIS polygon is (AVM polygons need not sit at the cell centre)
        for sx in sxs:
            for sy in (-Ly, 0.0, Ly):
                lo = xy.min(axis=0) + (sx, sy)
                hi = xy.max(axis=0) + (sx, sy)
                if hi[0] < crop_x[0] or lo[0] > crop_x[1] or hi[1] < y0 or lo[1] > y1:
                    continue
                # colour each DRAWN COPY by where it sits: a copy shifted across the periodic seam is a wall cell
                # only if it lies in the wall (x < 1.9), so border-clipped cells get the right colour on both edges
                ct = int(ctype[k][c])
                if ct in (1, 2, 3):
                    t = ct
                else:
                    t = 5 if (wall and _wall_copy(ct, member[c], cxp + sx)) else 0
                verts.append(xy + (sx, sy))
                faces.append(to_rgba(getattr(v0, 'face2', None) if (t == 2 and getattr(v0, 'face2', None)) else mm.FACE.get(t, mm.BG_FACE)))
                edges.append(to_rgba(mm.EDGE, 1.0 if t == 5 else 0.5))
    col = PolyCollection(verts, facecolors=faces, edgecolors=edges, linewidths=0.8, antialiaseds=True)
    clip = _rounded(ax)
    ax.add_patch(clip)
    clip.set_visible(False)
    col.set_clip_path(clip)
    ax.add_collection(col)

    if interface:
        segs = []
        for a, b in _interface_segments(polys, ctype[k], Lx, Ly):
            for sx in sxs:
                for sy in (-Ly, 0.0, Ly):
                    if max(a[0], b[0]) + sx < crop_x[0] or min(a[0], b[0]) + sx > crop_x[1] \
                            or max(a[1], b[1]) + sy < y0 or min(a[1], b[1]) + sy > y1:
                        continue
                    segs.append([(a[0] + sx, a[1] + sy), (b[0] + sx, b[1] + sy)])
        if interface == "length":
            ln = np.array([np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in segs])
            lc = LineCollection(segs, colors=EDGE_CMAP(np.clip(ln / EDGE_LMAX, 0, 1)), linewidths=3.4, capstyle="round", zorder=3)
        else:
            lc = LineCollection(segs, colors=DEEP, linewidths=2.6, capstyle="round", zorder=3)
        lc.set_clip_path(clip)
        ax.add_collection(lc)

    if k >= p0:
        sel = np.where((ctype[k] == 1) | (ctype[k] == 2) | (ctype[k] == 3))[0]
        x = centers[k][sel, 0] - np.floor(centers[k][sel, 0] / Lx) * Lx
        y = centers[k][sel, 1] - np.floor(centers[k][sel, 1] / Ly) * Ly
        speed = np.full(len(sel), float(v0))
        if getattr(v0, "u0", None) is not None:
            speed[ctype[k][sel] == 2] = v0.u0
        if getattr(v0, "u3", None) is not None:
            speed[ctype[k][sel] == 3] = v0.u3
        keep = speed > 0
        x, y, speed, sel = x[keep], y[keep], speed[keep], sel[keep]
        q = ax.quiver(x, y, speed * np.cos(director[k][sel]), speed * np.sin(director[k][sel]),
                      color=mm.ARROW_COLOR, angles="xy", scale_units="xy", scale=1,
                      width=0.0032, headwidth=4.2, headlength=4.8, zorder=5)
        q.set_clip_path(clip)

    ax.set_xlim(*crop_x)
    ax.set_ylim(y0, y1)
    if extras is None and axis_ticks is None:
        ax.axis("off")
    elif extras is None:
        # plain position axis (shared x window): ticks every 2 or 4 box units, label on the bottom row only
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_yticks([])
        step = 4 if _STATE["figsize"][0] * ax.get_position().width < 5.5 else 2
        ax.set_xticks([t for t in range(int(np.ceil(crop_x[0])), int(crop_x[1]) + 1) if t % step == 0])
        ax.tick_params(axis="x", length=3.5, width=0.8, colors=MUTED, labelsize=9, pad=3)
        if axis_ticks.get("label", True):
            ax.set_xlabel("x position (simulation length units)", fontsize=9.5, color=MUTED, labelpad=4)
    else:
        sxp, trail, cur = extras["start_x"], extras["trail"], extras["cur"]
        ln = ax.axvline(sxp, color=INK, ls=(0, (4, 3)), lw=1.1, alpha=0.55, zorder=4)
        ln.set_clip_path(clip)
        tl, = ax.plot(trail[:, 0], trail[:, 1], color=INK, lw=1.6, ls=":", zorder=6)
        tl.set_clip_path(clip)
        mk, = ax.plot([cur[0]], [cur[1]], "o", ms=5.5, mfc="white", mec=INK, mew=1.4, zorder=7)
        mk.set_clip_path(clip)
        ax.text(sxp, y1 - 0.18, "start", ha="center", va="top", fontsize=9, color=INK, zorder=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        d = round(float(extras["dx"]), 1)
        dx = "0.0" if d == 0 else f"{d:+.1f}".replace("-", "\u2212")
        ax.text(0.975, 0.93, f"\u0394x = {dx}", transform=ax.transAxes, ha="right", va="top", fontsize=11,
                color=INK, zorder=20, bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=RULE, lw=0.6, alpha=0.92))
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([t for t in range(int(np.ceil(crop_x[0])), int(crop_x[1]) + 1) if t % 2 == 0])
        ax.tick_params(axis="x", length=3.5, width=0.8, colors=MUTED, labelsize=9, pad=3)
        ax.set_xlabel("x position (simulation length units)", fontsize=9.5, color=MUTED, labelpad=4)
    ax.add_patch(_rounded(ax, fill=False, ec=RULE, lw=1.0, zorder=10, clip_on=False))


def _cluster_center(P, k):
    """Periodic-safe centroid of the active (type 1) cells at frame k."""
    centers, ctype, Lx, Ly = P[1], P[2], P[5], P[6]
    m = ctype[k] == 1
    ax_ = 2 * np.pi * centers[k][m, 0] / Lx
    ay_ = 2 * np.pi * centers[k][m, 1] / Ly
    cx = (np.arctan2(np.sin(ax_).mean(), np.cos(ax_).mean()) % (2 * np.pi)) * Lx / (2 * np.pi)
    cy = (np.arctan2(np.sin(ay_).mean(), np.cos(ay_).mean()) % (2 * np.pi)) * Ly / (2 * np.pi)
    return cx, cy


def _text_w(fig, t):
    r = fig.canvas.get_renderer()
    return t.get_window_extent(r).width / fig.dpi


def render_frame(args):
    k, outdir, spec, geom = args
    W, H = geom["fig_w"], geom["fig_h"]
    _STATE["figsize"] = (W, H)
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=FIG_BG)
    axf = fig.add_axes([0, 0, 1, 1])
    axf.set_xlim(0, W)
    axf.set_ylim(0, H)
    axf.axis("off")

    panels = _STATE["panels"]
    time = panels[0][4]
    p0 = geom["p0"]
    nrow, ncol = spec["nrow"], spec["ncol"]
    pw, ph = geom["pw"], geom["ph"]
    LEFT = geom["left"]

    # title block
    tsize = min(21, 21 * (geom["fig_w"] - LEFT - 0.4) / (0.165 * len(spec["title"])))   # long titles shrink to fit the frame
    axf.text(LEFT, H - 0.42, spec["title"], fontsize=tsize, fontweight="bold", color=INK, va="center", ha="left")
    axf.text(LEFT, H - 0.82, spec["subtitle"], fontsize=11.5, color=MUTED, va="center", ha="left")

    # column headers
    for c, hdr in enumerate(spec["col_headers"]):
        cx = LEFT + c * (pw + GAP) + pw / 2
        axf.text(cx, H - TOP_TITLE - COL_HDR / 2, hdr, fontsize=14, fontweight="bold", color=INK,
                 va="center", ha="center")

    for r in range(nrow):
        ytop = H - TOP_TITLE - COL_HDR - r * (ph + GAP)
        if spec.get("row_labels"):
            axf.text(LEFT - 0.2, ytop - ph / 2, spec["row_labels"][r], fontsize=13, fontweight="bold",
                     color=INK, va="center", ha="right", rotation=90, linespacing=1.5)
        for c in range(ncol):
            i = r * ncol + c
            x0 = LEFT + c * (pw + GAP)
            ax = fig.add_axes([x0 / W, (ytop - ph) / H, pw / W, ph / H])
            extras = None
            if spec.get("axes"):
                cs = np.array([_cluster_center(panels[i], j) for j in range(p0, max(k, p0) + 1)])
                extras = {"start_x": cs[0][0], "trail": cs, "cur": cs[-1], "dx": cs[-1][0] - cs[0][0]}
            if geom.get("windows"):
                cxr, cyr = geom["windows"][i]
            elif spec.get("follow"):
                cx, cy = _cluster_center(panels[i], k)
                ww, wh = spec["follow"]
                cxr, cyr = (cx - ww / 2, cx + ww / 2), (cy - wh / 2, cy + wh / 2)
            else:
                cxr, cyr = geom["crop_x"], geom["crop_y"]
            draw_panel(ax, panels[i], k, cxr, cyr, p0, interface=spec.get("interface", False),
                       extras=extras, periodic_x=spec.get("periodic_x", True),
                       axis_ticks=({"label": True} if (spec.get("axis") and r == nrow - 1) else None),
                       wall=spec.get("wall", True))
            ax.text(0.025, 0.93, spec["tags"][i], transform=ax.transAxes, fontsize=11, color=INK,
                    va="top", ha="left", zorder=20,
                    bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=RULE, lw=0.6, alpha=0.92))

    # progress bar + time
    grid_bottom = H - TOP_TITLE - COL_HDR - nrow * ph - (nrow - 1) * GAP
    bar_y = grid_bottom - geom["axis_pad"] - 0.32
    bar_x0, bar_x1 = LEFT, LEFT + ncol * pw + (ncol - 1) * GAP
    axf.add_patch(FancyBboxPatch((bar_x0, bar_y - 0.03), bar_x1 - bar_x0, 0.04,
                                 boxstyle="round,pad=0,rounding_size=0.02", fc=RULE, ec="none"))
    t_end = float(time[-1, 0]) - float(time[p0, 0])
    if k >= p0:
        tcur = float(time[k, 0]) - float(time[p0, 0])
        frac = max(0.0, min(1.0, tcur / t_end))
        if frac > 0:
            axf.add_patch(FancyBboxPatch((bar_x0, bar_y - 0.02), (bar_x1 - bar_x0) * frac, 0.04,
                                         boxstyle="round,pad=0,rounding_size=0.02", fc=INK, ec="none"))
        label = f"t = {tcur + (float(time[p0, 0]) if spec.get('absolute_time') else 0.0):.0f}"   # absolute simulation time if requested (t = 1 is the end of equilibration)
    else:
        label = "Equilibration"
    axf.text(bar_x1, bar_y - 0.2, label, fontsize=15, fontweight="bold", color=INK, va="center", ha="right")

    # legend
    lx, ly = bar_x0, bar_y - 0.2
    items = spec.get("legend_items") or [(mm.ACTIVE_FACE, "Cluster of active cells"), (mm.BG_FACE, "Surrounding tissue"),
                                          (mm.WALL_FACE, "Wall")]
    if spec.get("legend_wall") is False:
        items = [it for it in items if it[1] != "Wall"]
    for col, txt in items:
        axf.add_patch(Circle((lx + 0.09, ly), 0.085, fc=col, ec=mm.EDGE, lw=0.7))
        tt = axf.text(lx + 0.27, ly, txt, fontsize=10.5, color=INK, va="center", ha="left")
        lx += 0.27 + _text_w(fig, tt) + 0.4
    axf.add_patch(FancyArrowPatch((lx, ly), (lx + 0.3, ly), arrowstyle="-|>", mutation_scale=11,
                                  color=mm.ARROW_COLOR, lw=1.8))
    tt = axf.text(lx + 0.42, ly, "Motility direction", fontsize=10.5, color=INK, va="center", ha="left")
    if spec.get("interface"):
        lx += 0.42 + _text_w(fig, tt) + 0.4
        if spec.get("interface") == "length":
            for j in range(12):    # little colour bar: edge length 0 ... EDGE_LMAX
                axf.plot([lx + 0.03 * j, lx + 0.03 * (j + 1) + 0.005], [ly, ly], color=EDGE_CMAP(j / 11), lw=3.4, solid_capstyle="butt")
            axf.text(lx + 0.42, ly, f"Cluster\u2013tissue edge length (0 to {EDGE_LMAX:g})", fontsize=10.5, color=INK, va="center", ha="left")
        else:
            axf.plot([lx, lx + 0.3], [ly, ly], color=DEEP, lw=2.6, solid_capstyle="round")
            axf.text(lx + 0.42, ly, "Cluster\u2013tissue interface", fontsize=10.5, color=INK, va="center", ha="left")

    fig.savefig(os.path.join(outdir, f"{k:05d}.png"), dpi=DPI, facecolor=FIG_BG)
    plt.close(fig)


def make_grid_movie(spec, out_path):
    files = spec["files"]
    last_frame = spec.get("last_frame")
    if spec.get("stop_before_x") is not None:
        # end the movie just before any cluster cell comes closer than x = stop_before_x to the wall
        ends = []
        thr_list = spec["stop_before_x"] if isinstance(spec["stop_before_x"], (list, tuple)) else [spec["stop_before_x"]] * len(files)
        for f, thr in zip(files, thr_list):
            if thr is None:           # this panel is allowed to reach the wall
                continue
            Pn = load_panel(f)
            ctype, centers, Lx = Pn[2], Pn[1], Pn[5]
            front = np.array([(centers[k][(ctype[k] >= 1) & (ctype[k] <= 3), 0] % Lx).min() for k in range(len(Pn[4]))])
            hit = np.where(front < thr)[0]
            ends.append((int(hit[0]) - 1) if len(hit) else len(front) - 1)
        last_frame = min(ends) if last_frame is None else min(last_frame, min(ends))
        print(f"stop before x < {spec['stop_before_x']}: per-panel last frames {ends} -> using frame {last_frame}")
    panels = [load_panel(f, last_frame) for f in files]
    T = len(panels[0][4])
    assert all(len(p[4]) == T for p in panels), "panels must have the same number of frames"
    p0 = mm.production_start(panels[0][4])
    assert all(mm.production_start(p[4]) == p0 for p in panels)

    # crop rectangle: full width, CROP_H_UNITS tall, centered on where the clusters live
    follow = spec.get("follow")
    ys = []
    for model, centers, ctype, *_r in panels:
        m = ctype[0] == 1
        Ly = _r[3]
        ys.append((centers[:, m, 1] % Ly))
    ys = np.concatenate([y.ravel() for y in ys])
    ycen = 0.5 * (ys.min() + ys.max())
    crop_h = spec.get("crop_h", CROP_H_UNITS)
    crop_y = (ycen - crop_h / 2, ycen + crop_h / 2)
    if not follow:
        print(f"cluster y-range [{ys.min():.1f}, {ys.max():.1f}] -> crop y {crop_y[0]:.1f}..{crop_y[1]:.1f}")
        assert crop_y[0] <= ys.min() - 1.0 and crop_y[1] >= ys.max() + 1.0, "cluster too close to crop edge"

    # x window of the rectangular cut: chosen automatically so no border slices a wrongly coloured cell
    # (a spec can still force it with "crop_x": [x0, x1])
    crop_x = tuple(spec["crop_x"]) if spec.get("crop_x") else auto_x_window(panels)
    print(f"x window {crop_x[0]:.2f}..{crop_x[1]:.2f}  (width {crop_x[1] - crop_x[0]:.2f} of the {panels[0][5]:.0f}-wide box)")
    assert crop_x[1] - crop_x[0] > 14, "automatic x window too narrow"


    fixed = spec.get("fixed_window")
    windows = None
    if fixed:
        ww, wh = fixed
        windows = []
        for P in panels:
            cs_all = np.array([_cluster_center(P, j) for j in range(p0, len(P[4]))])
            cx = cs_all[0, 0]
            cy = 0.5 * (cs_all[:, 1].min() + cs_all[:, 1].max())      # centre on the whole path
            hi = min(P[5] - 0.8, cx + 3.2)                              # stay clear of the box seam
            windows.append(((hi - ww, hi), (cy - wh / 2, cy + wh / 2)))
        print("fixed windows:", [tuple(round(v, 1) for v in w[0]) for w in windows])
    win = follow or fixed
    pw = spec.get("panel_w", PANEL_W)
    ph = pw * (win[1] / win[0]) if win else pw * crop_h / 20.0
    axis_pad = 0.6 if (spec.get("axes") or spec.get("axis")) else 0.0
    left = spec.get("left", LEFT)
    nrow, ncol = spec["nrow"], spec["ncol"]
    geom = {
        "pw": pw, "ph": ph, "left": left, "windows": windows, "axis_pad": axis_pad,
        "fig_w": left + ncol * pw + (ncol - 1) * GAP + RIGHT,
        "fig_h": TOP_TITLE + COL_HDR + nrow * ph + (nrow - 1) * GAP + FOOT + axis_pad,
        "crop_x": crop_x, "crop_y": crop_y, "p0": p0,
    }
    print(f"{nrow}x{ncol} grid, {T} frames, production from frame {p0}; figure {geom['fig_w']:.2f} x {geom['fig_h']:.2f} in")

    outdir = tempfile.mkdtemp(prefix="grid_frames_")
    try:
        k0 = 0 if spec.get("keep_equilibration") else p0     # the movie starts when the packing is equilibrated
        jobs = [(k, outdir, spec, geom) for k in range(k0, T)]
        with multiprocessing.Pool(processes=min(8, T), initializer=_init, initargs=(files, last_frame, dict(mm.FACE), dict(PANEL_FACE2))) as pool:
            pool.map(render_frame, jobs)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        lst = os.path.join(outdir, "files.txt")
        with open(lst, "w") as f:
            for k in range(k0, T):
                f.write(f"file '{k:05d}.png'\nduration {mm.EQUIL_FRAME_SEC if k < p0 else mm.PROD_FRAME_SEC}\n")
            f.write(f"file '{T - 1:05d}.png'\n")
        subprocess.check_call([
            mm.FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
            "-r", "30", "-c:v", "libx264", "-crf", "14", "-preset", "slow",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", out_path])
        if spec.get("keep_frames"):
            shutil.copytree(outdir, spec["keep_frames"], dirs_exist_ok=True)
    finally:
        shutil.rmtree(outdir, ignore_errors=True)
    print(f"Saved: {out_path}")


SPEC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specs")
EXAMPLES = "DATA/EXAMPLES"

def resolve_file(name):
    """A spec lists either paths to .nc files or example ids ('Fig3-SPV-polarity-formation'), which are looked up as
    DATA/EXAMPLES/<id>__*.nc."""
    if name.endswith(".nc") or os.path.exists(name):
        return name
    hit = sorted(glob.glob(os.path.join(EXAMPLES, name + "__*.nc")))
    if len(hit) != 1:
        raise SystemExit(f"example '{name}' not found in {EXAMPLES}/ (expected exactly one {name}__*.nc, found {len(hit)})")
    return hit[0]


def load_spec(path):
    """Read a JSON movie spec (see specs/). Keys: nrow, ncol, files (paths or example ids), tags, title, subtitle, col_headers,
    row_labels, axis, panel_w, crop_h, crop_x, left, wall, interface ('length' colours the cluster-tissue edges by length),
    legend_items [[colour, label], ...], type_colors {"1": hex, ...}, face2 [colour of type-2 cells per panel], stop_before_x,
    absolute_time, last_frame, fixed_window, axes, periodic_x, legend_wall, frame_sec (seconds per production frame)."""
    import json
    if not os.path.exists(path):
        path = os.path.join(SPEC_DIR, path if path.endswith(".json") else path + ".json")
    with open(path) as f:
        spec = json.load(f)
    for k in ("nrow", "ncol", "files", "tags", "title", "subtitle"):
        if k not in spec:
            raise SystemExit(f"spec {path} is missing '{k}'")
    spec["files"] = [resolve_file(x) for x in spec["files"]]
    for k in ("fixed_window", "follow"):
        if spec.get(k):
            spec[k] = tuple(spec[k])
    spec.setdefault("col_headers", [""] * spec["ncol"])
    for k, c in spec.get("type_colors", {}).items():
        mm.FACE[int(k)] = c
    for fpath, c in zip(spec["files"], spec.get("face2", [])):
        if c:
            PANEL_FACE2[fpath] = c
    if "legend_items" in spec:
        spec["legend_items"] = [tuple(x) for x in spec["legend_items"]]
    if "frame_sec" in spec:
        mm.PROD_FRAME_SEC = spec["frame_sec"]
    return spec


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Render a polished multi-panel SPV / AVM / Nodal movie from a JSON spec.")
    ap.add_argument("figure", nargs="?", help="name of a spec in PYTHON_ANALYSIS/movies/specs/ (fig1, fig2, fig2_pinning, fig3, fig4, fig5a, fig5b)")
    ap.add_argument("--spec", help="path to your own JSON spec instead of a figure name")
    ap.add_argument("--out", help="output .mp4 [RESULTS/movies/<figure>.mp4]")
    a = ap.parse_args()
    name = a.spec or a.figure
    if not name:
        ap.error("give a figure name (e.g. fig3) or --spec")
    out = a.out or os.path.join("RESULTS", "movies", os.path.splitext(os.path.basename(name))[0] + ".mp4")
    make_grid_movie(load_spec(name), out)
