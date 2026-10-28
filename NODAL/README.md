# NODAL programs (Nodal-signalling version of the SPV model)

The Nodal level N of a cell type sets its motility, v = β·N, and its interactions through α. Two families of
programs are shipped; use the single-type one for one- or two-type clusters and the three-cell one for three types.

| program | cell types | rule |
|---|---|---|
| `nodal_pos_gamma` | 1 or 2 (leaders = type 1, followers = type 2) | POS: γ(bg,i) = α N_i, γ(1,2) = \|α N_1 − α N_2\| |
| `nodal_neg_gamma` | 1 or 2 | NEG: γ(i,i) = −α N_i, γ(1,2) = −α √(N_1 N_2) |
| `nodal_pos_gamma_livesplit` | 1 or 2 | as `nodal_pos_gamma` + live cluster-split detector |
| `nodal_pos_gamma_three_cell` | 1, 2 or 3 | POS rule, three types |
| `nodal_neg_gamma_three_cell` | 1, 2 or 3 | NEG rule, three types |
| `nodal_pos_gamma_three_cell_livesplit` | 1, 2 or 3 | as `nodal_pos_gamma_three_cell` + live split detector |

Build: `bash load_mod_and_make.sh nodal`.

For the Triple case (three cell types) use the three-cell programs; `--preset triple` of the helper builds the command.

`nodal_commands.py` turns (α, β, Nodal levels, composition) into the exact command line:

```bash
python NODAL/nodal_commands.py --model pos --alpha 1.0 --beta 6.0 --preset N++      # homogeneous, N = 0.44
python NODAL/nodal_commands.py --model pos --alpha 1.0 --beta 6.0 --preset HM       # 50% + Shield, 4 + 4 cells
python NODAL/nodal_commands.py --model pos --alpha 1.0 --beta 6.0 --preset triple --livesplit
python NODAL/nodal_commands.py --model neg --alpha 2.0 --beta 7.0 --levels 0.44 0.08 --counts 5 3
```

Presets use the Nodal levels of the paper: N++ = 0.44, N+ = 0.2, N− = 0.08; induced 1.0, 0.23, 0.16. With one or
two levels the helper picks the single-type program, with three the three-cell one (`--three-cell` forces it).

## Flags

Shared with the SPV programs: `-t -e -n -m -i -k -p -b -j -d -y -x -c -S -O` (see `SPV/README.md`).

**Single-type programs** (`nodal_pos_gamma`, `nodal_neg_gamma`, `nodal_pos_gamma_livesplit`)

| flag | meaning |
|---|---|
| `-l` | number of type-1 cells (the rest of the `-m` cluster cells are type 2) |
| `-v -u` | self-propulsion speed of type 1, type 2 |
| `-s -q` | target shape index of the cluster cells, of the background |
| `-G -H -I` | POS: γ(bg,1), γ(bg,2), γ(1,2) — NEG: γ(1,1), γ(2,2), γ(1,2) |

**Three-cell programs** (`*_three_cell*`)

| flag | meaning |
|---|---|
| `-l -o -z` | number of cells of type 1, 2, 3 (they must add up to `-m`; types with count 0 are absent) |
| `-v -u -w` | self-propulsion speed of type 1, 2, 3 |
| `-s -q -r` | target shape index of type 1, 2, 3 [3.7] |

| flag | POS programs | NEG program |
|---|---|---|
| `-G` | background–type 1 | type 1–type 1 |
| `-H` | background–type 2 | type 2–type 2 |
| `-I` | type 1–type 2 | type 3–type 3 |
| `-J` | background–type 3 | type 1–type 2 |
| `-K` | type 1–type 3 | type 1–type 3 |
| `-L` | type 2–type 3 | type 2–type 3 |

Every program prints its surface-tension matrix at the start of the run, so the meaning of the flags can be
checked in the log.

Output: `<O>/<FOLDER>/Data_t_..._ensem_<x>.nc`, with `NODAL_POS_GAMMA`, `NODAL_NEG_GAMMA`,
`NODAL_POS_GAMMA_LIVESPLIT` for the single-type programs and `NODAL_POS_GAMMA_THREE_CELL`,
`NODAL_NEG_GAMMA_THREE_CELL`, `NODAL_POS_GAMMA_THREE_CELL_LIVESPLIT` for the three-cell ones. The single-type
file names contain `c_Size_<m>_a_Cells_<l>_va_<v>_vp_<u>_..._T1_<G>_T2_<H>_T3_<I>`; the three-cell ones
`c_Size_<m>_t1_<l>_t2_<o>_t3_<z>_v1_..._v3_..._G1_<G>_..._G6_<L>`.

## Live split detector (`*_livesplit`)

Checks cluster connectivity at every integration step (debounced) and writes a text file
`<name>_split_time_live.txt` next to the `.nc` with the split times and, for each split, the number of lost
contacts of each type pair that were cut by the split.
