#!/usr/bin/env python3
"""
Build the command line for the NODAL programs from the model's two parameters, alpha and beta, and the Nodal
levels N_i of the cell types in the cluster.

Rules used in the paper (v_i = motility, gamma = line tension):

    v_i = beta * N_i

    POS rule   gamma(background, i) =  alpha * N_i
               gamma(i, j)          = |alpha * N_i - alpha * N_j|

    NEG rule   gamma(i, i)          = -alpha * N_i
               gamma(i, j)          = -alpha * sqrt(N_i * N_j)

Two program families (both are shipped, see NODAL/README.md):

  single-type  nodal_pos_gamma / nodal_neg_gamma / nodal_pos_gamma_livesplit   ONE or TWO cell types
                 -v -u  speeds of type 1, 2     -m cluster size   -l number of type-1 cells
                 POS: -G bg-1   -H bg-2   -I 1-2          NEG: -G 1-1   -H 2-2   -I 1-2
  three-cell   nodal_pos_gamma_three_cell / nodal_neg_gamma_three_cell / nodal_pos_gamma_three_cell_livesplit
                 up to THREE cell types:  -l -o -z counts, -v -u -w speeds
                 POS flags: -G bg-1  -H bg-2  -I bg-3  -J 1-2  -K 1-3  -L 2-3
                 NEG flags: -G 1-1   -H 2-2   -I 3-3   -J 1-2  -K 1-3  -L 2-3

With one or two Nodal levels the single-type program is used, with three levels the three-cell program
(or force it with --three-cell). The surface-tension matrix is printed at the start of every run so the
meaning of the flags can be checked in the log.

Examples
    python NODAL/nodal_commands.py --model pos --alpha 1.0 --beta 6.0 --preset N++        # homogeneous, N = 0.44
    python NODAL/nodal_commands.py --model pos --alpha 1.0 --beta 6.0 --preset HM         # 50% + Shield, 4 + 4 cells
    python NODAL/nodal_commands.py --model pos --alpha 1.0 --beta 6.0 --preset triple --livesplit
    python NODAL/nodal_commands.py --model neg --alpha 2.0 --beta 7.0 --levels 0.44 0.16 --counts 5 3
"""
import argparse
import math

# preset name -> (levels, counts)
PRESETS = {
    "N++": ([0.44], [8]), "N+": ([0.20], [8]), "N-": ([0.08], [8]),
    "induced-N++": ([1.00], [8]), "induced-N+": ([0.23], [8]), "induced-N-": ([0.16], [8]),
    "HM": ([0.44, 0.20], [4, 4]),            # 50% + Shield
    "HL": ([0.44, 0.08], [4, 4]),            # 50% + 75%
    "induced-HM": ([1.00, 0.20], [4, 4]),    # induced 50% + Shield
    "induced-HL": ([0.44, 0.16], [4, 4]),    # 50% + induced 75%
    "triple": ([0.44, 0.20, 0.08], [3, 3, 3]),
}


def fmt(x):
    x = round(float(x), 2)
    return str(int(x)) if x == int(x) else str(x)


def build(model, alpha, beta, levels, counts, ensemble, steps, out, save, livesplit, three_cell=None,
          s0=3.7, ka=10, kp=1, ic=1):
    n = len(levels)
    if not 1 <= n <= 3 or len(counts) != n:
        raise SystemExit("give 1-3 Nodal levels and one cell count per level")
    use_three = (n == 3) if three_cell is None else three_cell
    if n == 3 and not use_three:
        raise SystemExit("three cell types need the three-cell program")
    if livesplit and model == "neg":
        raise SystemExit("the live split detector is only provided for the POS rule")
    N = list(levels) + [0.0] * (3 - n)
    cnt = list(counts) + [0] * (3 - n)
    v = [beta * x for x in N]

    if model == "pos":
        bg = [alpha * x for x in N]                          # gamma(background, i)
        cross = lambda i, j: abs(bg[i] - bg[j])
    else:
        selft = [-alpha * x for x in N]                      # gamma(i, i)
        cross = lambda i, j: -alpha * math.sqrt(N[i] * N[j])

    if not use_three:
        # single-type program: one or two types
        if model == "pos":
            flags = {"-G": bg[0], "-H": bg[1], "-I": cross(0, 1) if n > 1 else 0}
            binary = "nodal_pos_gamma_livesplit" if livesplit else "nodal_pos_gamma"
        else:
            flags = {"-G": selft[0], "-H": selft[1], "-I": cross(0, 1) if n > 1 else 0}
            binary = "nodal_neg_gamma"
        parts = [f"bin/{binary}.out", "-t", str(steps), "-e", "0.0002", "-n", "400", "-m", str(sum(cnt)),
                 "-l", str(cnt[0]), "-v", fmt(v[0]), "-u", fmt(v[1]),
                 "-s", str(s0), "-q", str(s0), "-k", str(ka), "-p", str(kp)]
    else:
        if model == "pos":
            flags = {"-G": bg[0], "-H": bg[1], "-I": bg[2],
                     "-J": cross(0, 1) if n > 1 else 0, "-K": cross(0, 2) if n > 2 else 0, "-L": cross(1, 2) if n > 2 else 0}
            binary = "nodal_pos_gamma_three_cell_livesplit" if livesplit else "nodal_pos_gamma_three_cell"
        else:
            flags = {"-G": selft[0], "-H": selft[1], "-I": selft[2],
                     "-J": cross(0, 1) if n > 1 else 0, "-K": cross(0, 2) if n > 2 else 0,
                     "-L": cross(1, 2) if n > 2 else 0}
            binary = "nodal_neg_gamma_three_cell"
        parts = [f"bin/{binary}.out", "-t", str(steps), "-e", "0.0002", "-n", "400", "-m", str(sum(cnt)),
                 "-l", str(cnt[0]), "-o", str(cnt[1]), "-z", str(cnt[2]),
                 "-v", fmt(v[0]), "-u", fmt(v[1]), "-w", fmt(v[2]),
                 "-s", str(s0), "-q", str(s0), "-r", str(s0), "-k", str(ka), "-p", str(kp)]
    for k, val in flags.items():
        parts += [k, fmt(val)]
    parts += ["-b", "0.005", "-j", "0.02", "-y", "0.01", "-i", str(ic), "-x", str(ensemble),
              "-S", str(save), "-O", out]
    return " ".join(parts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["pos", "neg"], required=True, help="gamma rule")
    ap.add_argument("--alpha", type=float, required=True, help="interaction scale")
    ap.add_argument("--beta", type=float, required=True, help="motility scale")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="named cluster composition")
    ap.add_argument("--levels", type=float, nargs="+", help="Nodal level N of each cell type (1-3 values)")
    ap.add_argument("--counts", type=int, nargs="+", help="number of cells of each type")
    ap.add_argument("--ensemble", type=int, default=1, help="ensemble index (-x), sets the random seed")
    ap.add_argument("--ic", type=int, default=1, help="initial condition (-i): 1 disc random order, 2 active in front, 3 active at the back")
    ap.add_argument("--steps", type=int, default=500000, help="production steps (dt=0.0002, 500000 = T 100)")
    ap.add_argument("--save", type=float, default=5.0, help="time between saved frames")
    ap.add_argument("--out", default="DATA", help="output root directory")
    ap.add_argument("--livesplit", action="store_true", help="use the program with the live split detector")
    ap.add_argument("--three-cell", action="store_true", help="use the three-cell program even for one or two cell types")
    a = ap.parse_args()
    if a.preset:
        levels, counts = PRESETS[a.preset]
    elif a.levels and a.counts:
        levels, counts = a.levels, a.counts
    else:
        ap.error("give --preset, or both --levels and --counts")
    print(build(a.model, a.alpha, a.beta, levels, counts, a.ensemble, a.steps, a.out, a.save, a.livesplit,
                three_cell=True if a.three_cell else None, ic=a.ic))
