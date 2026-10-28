# AVM drivers (active vertex model)

| program | model |
|---|---|
| `avm_pos_gamma` | active vertex model with positive-γ line tension at the cluster/tissue boundary |
| `avm_viscous_adhesion` | wet vertex model: adds rate-dependent dissipation at cell–cell junctions (flags `-X -E -M`), static line tension kept |
| `avm_viscous_dissipation_type` | wet vertex model with junction dissipation that depends on whether the two cells bordering an edge are of the same type (`-X`) or different types (`-Y`); run with `-G 0 -H 0 -I 0` to switch the line tension off |

Build: `bash load_mod_and_make.sh avm`  →  `bin/avm_*.out`.

Everything shared with the SPV drivers has the same flags and meaning, see `SPV/README.md`
(`-t -e -n -m -l -i -v -u -G -H -I -s -q -k -p -b -j -d -y -x -c -S -O`). In `avm_pos_gamma`,
`-G -H -I` are γ(background,1), γ(background,2), γ(1,2). A homogeneous cluster is again `-l` = `-m`, `-u 0`.

## Extra flags of the wet-vertex drivers

Written as named in the source (`xi0`, `eta0`, `mu`, `chiSame`, `chiDiff`); see the header comment of
each `.cpp` and `inc/updaters/selfPropelledCellVertexDynamicsWith*.h` for the equations of motion.

| flag | driver | parameter in the code |
|---|---|---|
| `-X` | `avm_viscous_adhesion` | `xi0`, dimensionless viscous adhesion coefficient [0] |
| `-E` | `avm_viscous_adhesion` | `eta0`, isotropic viscous coupling per bond [0] |
| `-M` | both | `mu`, coefficient of the vertex velocity term [1] |
| `-X` | `avm_viscous_dissipation_type` | `chiSame`, same-type edges [0] |
| `-Y` | `avm_viscous_dissipation_type` | `chiDiff`, different-type edges [0] |

*(Parameter descriptions: to be completed by the authors.)* The wet-vertex drivers solve a linear system
every step and run on the CPU only.

Output: `<O>/AVM_POS_GAMMA/Data_t_..._ensem_<x>.nc`. The wet-vertex drivers write to `AVM_VISCOUS_ADHESION/` and
`AVM_VISCOUS_DISSIPATION_TYPE/` with a slightly shorter name (`cSz_<m>_aC_<l>`, `ens_<x>`, and the extra parameters
`xi0/eta0/mu` or `chiSame/chiDiff/mu` before `Dr_b`). Variables: `pos` (vertices), `cellPositions`, `cellType`,
`Vneighs`, `VertexCellNeighbors`, `director`, `force`, `time`, `BoxMatrix`.

## Example

```bash
# pinned regime, homogeneous cluster of 8 cells, v0 = 0.75
bin/avm_pos_gamma.out -t 500000 -e 0.0002 -n 400 -m 8 -l 8 -v 0.75 -u 0 -i 1 -x 1 -G 5 -H 0 -I 0
```
