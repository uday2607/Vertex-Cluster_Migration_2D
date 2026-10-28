"""
Make a compact example trajectory from a full run, for the movie scripts.

Keeps every frame (or every STRIDE-th production frame with --stride) and only the variables the movie scripts read (the AVM forces and
vertex-neighbour table are dropped), stored in single precision (positions to ~1e-6) and 16-bit integers. A full T = 100 run saved every 0.2
time units is 5-30 MB; the compact example is 2-9 MB and gives the same movie.

    python PYTHON_ANALYSIS/movies/thin_example.py full_run.nc DATA/EXAMPLES/Fig3-SPV-polarity-formation__<name without ensemble index>.nc
"""
import argparse
import numpy as np
from scipy.io import netcdf_file

KEEP = ("time", "pos", "type", "cellType", "cellPositions", "VertexCellNeighbors", "director", "BoxMatrix", "means0", "meanQ")


def thin(src, dst, stride=1):
    d = netcdf_file(src, "r", mmap=False)
    t = d.variables["time"][:].ravel() * 1
    dup = np.where(np.diff(t) < 0.01)[0]
    p0 = int(dup[0] + 1) if len(dup) else 0          # first production frame (end-of-equilibration duplicate time stamp)
    idx = sorted(set(list(range(p0)) + list(range(p0, len(t), stride)) + [len(t) - 1]))
    out = netcdf_file(dst, "w")
    dims = {k: v for k, v in d.dimensions.items()}
    for k, v in dims.items():
        out.createDimension(k, None if k == "rec" else v)
    for name, var in d.variables.items():
        if name not in KEEP:
            continue
        data = var.data
        char = np.dtype(data.dtype.newbyteorder("=")).char
        if name in ("pos", "director", "cellPositions", "means0", "meanQ") and char == "d":
            char = "f"                      # single precision is plenty for positions and angles
        elif name in ("type", "cellType", "VertexCellNeighbors"):
            char = "h"                      # small integers (cell types, vertex/cell indices < 32768)
        out_var = out.createVariable(name, char, var.dimensions)
        arr = np.asarray(data)[idx] if var.dimensions and var.dimensions[0] == "rec" else np.asarray(data)
        out_var[:] = np.rint(arr) if char == "h" else arr
    out.close()
    d.close()
    return len(t), len(idx)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--stride", type=int, default=1, help="keep every STRIDE-th production frame [1 = all]")
    a = ap.parse_args()
    n, m = thin(a.src, a.dst, a.stride)
    print(f"{a.src}: {n} frames -> {m} frames ({a.dst})")
