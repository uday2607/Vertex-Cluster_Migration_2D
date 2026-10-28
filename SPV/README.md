# SPV drivers (self-propelled Voronoi)

| program | tension rule | flags -G / -H / -I |
|---|---|---|
| `spv_pos_gamma` | positive γ: tension at tissue/cell-type boundaries | γ(background,1) / γ(background,2) / γ(1,2) |
| `spv_neg_gamma` | negative γ: tension between cells of the same type | γ(1,1) / γ(2,2) / γ(1,2) |

Build: `bash load_mod_and_make.sh spv`  →  `bin/spv_pos_gamma.out`, `bin/spv_neg_gamma.out`.

A cluster of `-m` cells starts as a disc at the right of the box (`-i 1`); `-l` of them are active
(type 1, self-propulsion `-v`), the rest are passive followers (type 2, self-propulsion `-u`).
**A homogeneous cluster is `-l` equal to `-m` (and `-u 0`)**: there is no separate homogeneous program.
Cells at x < 2 become a pinned wall (type 5) once equilibration ends.

## Flags (defaults in brackets)

| flag | meaning |
|---|---|
| `-t` | production steps [500000]; with `-e` 0.0002 this is T = 100 |
| `-e` | time step dt [0.0002] |
| `-n` | total number of cells [400] |
| `-m` | cells in the migrating cluster [8] |
| `-l` | active cells in the cluster [m/2] |
| `-i` | initial condition: 0 two halves, 1 disc (random order), 2 disc (active cells in front), 3 disc (active cells at the back) [1] |
| `-v`, `-u` | self-propulsion speed of active cells [2] / passive followers [0.5] |
| `-G -H -I` | line tensions, see the table above |
| `-s`, `-q` | target shape index of cluster cells / background cells [3.7, 3.7] |
| `-k`, `-p` | area and perimeter moduli Ka [10], Kp [1] |
| `-b`, `-j` | rotational diffusion [0.005] and drift [0.02] of the cluster cells' polarity (Ornstein–Uhlenbeck) |
| `-d`, `-y` | rotational diffusion [0.05] and speed [0.01] of the background cells |
| `-x` | ensemble index (also the default random seed) [999] |
| `-c` | explicit random seed; −1 = clock + ensemble [−1] |
| `-S` | time between saved frames during production [5.0] |
| `-O` | output root directory [DATA] |

Output: `<O>/SPV_POS_GAMMA/Data_t_<steps>_N_<n>_dt_<dt>_Ka_<Ka>_Kp_<Kp>_A0_..._P0_<s>-<q>_s0_..._c_Size_<m>_a_Cells_<l>_va_<v>_vp_<u>_vb_..._T1_<G>_T2_<H>_T3_<I>_Dr_b_..._IC_<i>_ensem_<x>.nc`
(`SPV_NEG_GAMMA` for the negative rule). Variables: `pos`, `type`, `director`, `time`, `BoxMatrix`.

## Example

```bash
# the three regimes of the paper, homogeneous cluster of 8 cells, v0 = 1.5
for g in 0 0.2 5; do
  bin/spv_pos_gamma.out -t 500000 -e 0.0002 -n 400 -m 8 -l 8 -v 1.5 -u 0 -i 1 -x 1 -G $g -H 0 -I 0 &
done; wait
```
