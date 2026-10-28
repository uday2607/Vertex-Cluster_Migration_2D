# Example trajectories

Small, ready-made simulation outputs, one set per movie, so every movie of the paper can be made without running a simulation.
`MANIFEST.csv` lists every file with its figure and parameters. Each file is a run from the paper's simulations with every saved frame (one per 0.2 time units, T = 100), stored in a compact format
(`PYTHON_ANALYSIS/movies/thin_example.py`: only the variables the movie scripts read, single precision), so the movies are the ones of the paper.

Names are `<id>__<parameter-encoded name>.nc` (the ensemble index is deliberately left out). The part after `__` is what the tools
read the parameters from, so keep it intact. The ids:

| movie | files |
|---|---|
| `fig1` | `Fig1-SPV-caged`, `Fig1-SPV-internalized` |
| `fig2` | `Regimes-{SPV,AVM}-{fragmentation,optimal,pinning}` |
| `fig2_pinning` | `Regimes-SPV-pinning`, `Regimes-AVM-pinning` |
| `fig3` | `Fig3-SPV-polarity-formation` |
| `fig4` | `Fig4-Nodal-control-4cell`, `Fig4-Nodal-Ecad-knockdown-4cell` |
| `fig5a` | `Fig5a-Nodal-HM-4+4`, `Fig5a-Nodal-HL-4+4` |
| `fig5b` | `Fig5b-Nodal-Triple-POS`, `Fig5b-Nodal-Triple-NEG` |

```bash
python PYTHON_ANALYSIS/movies/make_grid_movie.py fig3        # writes RESULTS/movies/fig3.mp4
python PYTHON_ANALYSIS/cellgpu_analysis.py movie --input DATA/EXAMPLES/<file>.nc
```

To make the same kind of movie from your own runs, simulate with a small `-S` (e.g. 0.2) and point a spec at your `.nc` files
(see `PYTHON_ANALYSIS/movies/README.md`). The Nodal commands are built with `NODAL/nodal_commands.py`.

The older `example_*.nc` and `POS_GAMMA_*` files are tiny synthetic/short runs used by the unit tests.
