# Cluster migration in a passive tissue — SPV and active vertex models (2D)

Simulation code for a cluster of active cells (the *migrating cluster*) pushing through a passive
background tissue, in two complementary mechanical models, built on
[CellGPU](https://gitlab.com/dmsussman/cellGPU) 0.8.0:

- **SPV** — self-propelled Voronoi: cells are points, the Voronoi tessellation defines their shape.
- **AVM** — active vertex model: cell boundaries are explicit vertices, so T1 transitions are resolved.

Cells at the cluster–tissue interface carry a line tension γ. The cluster starts at the **right** of
the periodic box and migrates **left** (−x) towards a wall of pinned cells (x < 2). Depending on γ the
cluster fragments, invades efficiently, or is pinned. A Nodal-signalling version of the model
(`NODAL/`) maps the Nodal level of each cell type to its motility (β) and interactions (α).

## Layout

```
SPV/      spv_pos_gamma, spv_neg_gamma                      -> SPV/README.md
AVM/      avm_pos_gamma, avm_viscous_adhesion,
          avm_viscous_dissipation_type                      -> AVM/README.md
NODAL/    nodal_{pos,neg}_gamma (1- or 2-type clusters), nodal_{pos,neg}_gamma_three_cell (up to 3 types),
          *_livesplit (+ live cluster-split detector),
          nodal_commands.py (builds commands from alpha, beta, Nodal levels)
                                                            -> NODAL/README.md
src/ inc/ CellGPU engine (with the additions listed in ChangeLog.md)
PYTHON_ANALYSIS/        metrics, plots, movies, tests       -> PYTHON_ANALYSIS/README.md
  movies/               publication-style movies            -> PYTHON_ANALYSIS/movies/README.md
DATA/EXAMPLES/          small ready-made trajectories       -> DATA/EXAMPLES/README.md
load_modules.sh         environment modules (source it)
load_mod_and_make.sh    compile everything
run_all.sh              end-to-end demo (compile, simulate, movie)
```

Each model folder has its own README with every command-line flag. One program per model and sign is
enough: a *homogeneous* cluster is just a run in which every cluster cell is active
(`-m 8 -l 8 -u 0`), a cluster with passive followers has `-l` smaller than `-m`.

## Quick start

```bash
source load_modules.sh               # cluster modules / environment (edit for your site)
bash load_mod_and_make.sh            # builds bin/*.out  (see "Requirements")
bash run_all.sh quick                # 6 short runs + a movie in RESULTS/demo/   (a few minutes)
```

Look at ready-made trajectories without simulating anything:

```bash
python PYTHON_ANALYSIS/movies/make_grid_movie.py fig2        # -> RESULTS/movies/fig2.mp4 (also fig1, fig2_pinning, fig3, fig4, fig5a, fig5b)
```

Run one simulation yourself (SPV, γ = 0.2, v₀ = 1.5, 8 active cells, full length T = 100):

```bash
bin/spv_pos_gamma.out -t 500000 -e 0.0002 -n 400 -m 8 -l 8 -v 1.5 -u 0 -i 1 -x 1 -G 0.2 -H 0 -I 0
```

Output goes to `DATA/<MODEL>/Data_t_..._ensem_<k>.nc` (change the root with `-O`, the time between
saved frames with `-S`). The file name encodes all parameters; the analysis and movie scripts read them
back from it, so do not rename files (you may add a prefix ending in `__`, as in `DATA/EXAMPLES`).

## Requirements

- Build: CUDA toolkit (`nvcc`), CGAL 4.9 (with Boost headers), Eigen, NetCDF with its C++ interface.
  `load_modules.sh` loads these as environment modules when the `module` command exists (source it before
  running the binaries by hand, they need the CUDA runtime library); otherwise set `CUDA_HOME` and `CGAL_DIR`
  yourself. The simulations themselves run on the CPU.
- Python (analysis, movies): numpy, scipy, matplotlib, networkx, freud-analysis, numba, and `ffmpeg`
  on the PATH.

## What a run looks like

Frames are saved every 0.2 time units during a short equilibration (cluster cells not yet moving, no
wall), then the wall cells (x < 2, type 5) are pinned and the run starts: one saved frame per `-S` time
units (default 5; use e.g. `-S 0.5` for smooth movies). The equilibration/production boundary is the
pair of frames with (nearly) the same time stamp, which the movie scripts detect automatically.

Cell types in the files: 0 background tissue, 1 active cells, 2 passive followers (or second Nodal
type), 3 third Nodal type, 5 wall.

## License and citation

MIT license (`LICENSE.md`); CellGPU citation information is in `CITATIONS.md`. The manuscript
reference will be added here once the work is published.
