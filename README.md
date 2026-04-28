# CellGPU — Heterotypic Cluster Migration (2D)

A focused simulation workspace for studying 2D heterotypic cell-cluster migration using
CellGPU version 0.8.0. The project models a cluster of active (leader) cells migrating
through a passive background tissue, using two complementary mechanical models:

- **SPV** — Soft Potts Voronoi: cells are represented by centroids; Voronoi tessellation
  defines cell geometry implicitly.
- **AVM** — Active Vertex Model: cell boundaries are tracked as explicit vertex positions,
  giving a detailed view of topology-changing events (T1 transitions).

In both models the cluster starts at the **right** side of the simulation box and migrates
**leftward (−x direction)** under self-propulsion of the active cells.

---

## Project layout

```
.
├── load_mod_and_make.sh          # load cluster modules and compile all binaries
├── run_all.sh                    # end-to-end: compile → simulate → analyse
├── makefile                      # builds all top-level .cpp → .out binaries
│
├── spv_pos_gamma_hetero_adhesion.cpp   # SPV driver, positive adhesion contrast (γ > 0)
├── spv_neg_gamma_hetero_adhesion.cpp   # SPV driver, negative adhesion contrast (γ < 0)
├── spv_nodal_pos_gamma_three_cell_types.cpp  # SPV with three cell types (positive γ)
├── spv_nodal_neg_gamma_three_cell_types.cpp  # SPV with three cell types (negative γ)
├── avm_pos_gamma_hetero_adhesion.cpp   # AVM driver, positive adhesion contrast (γ > 0)
├── avm_neg_gamma_hetero_adhesion.cpp   # AVM driver, negative adhesion contrast (γ < 0)
│
├── src/                          # CellGPU 0.8.0 simulation engine (C++/CUDA)
├── inc/                          # CellGPU header files
├── obj/                          # compiled object files (generated, not committed)
│
├── DATA/
│   ├── POS_GAMMA_HETERO/         # SPV output .nc trajectory files
│   └── POS_GAMMA_VERTEX_HETERO/  # AVM output .nc trajectory files
│
├── RESULTS/
│   ├── MOVIES/                   # animated trajectories (.gif or .mp4)
│   ├── PLOTS/
│   │   ├── POS_GAMMA_HETERO/     # SPV physics time-series plots (SPV_*.png)
│   │   └── POS_GAMMA_VERTEX_HETERO/  # AVM physics time-series plots (AVM_*.png)
│   └── PROCESSED_DATA/           # per-file CSVs, JSON summaries, ensemble stats
│
├── PYTHON_ANALYSIS/
│   ├── cellgpu_analysis.py       # unified analysis CLI (the only script you need)
│   ├── example_data.py           # synthetic nc-file generators (used by tests)
│   ├── generate_example_nc_files.py  # write example nc files to disk
│   ├── tests/
│   │   └── test_cellgpu_analysis.py  # physics-grounded unit tests (25 tests)
│   └── README.md                 # detailed analysis-workflow documentation
│
├── logs/                         # simulation and processing log files
├── INSTALLATION.md               # upstream CellGPU build notes
├── CITATIONS.md                  # how to cite CellGPU and this workspace
├── ChangeLog.md                  # CellGPU version history (upstream)
└── LICENSE.md                    # MIT license
```

---

## Prerequisites

### Cluster modules

Load once per session before compiling or running:

```bash
module load cuda/12.0.0 cgal/4.9.1 eigen/3.4.0 netcdf/4.1.3
```

Or use the helper script which loads modules and compiles in one step:

```bash
bash load_mod_and_make.sh
```

### Python environment

Python 3.8+ is required. Core dependencies:

```
numpy  scipy  matplotlib  netCDF4  Pillow
```

Optional (enables Voronoi shape index, MSD, and T1 detection):

```
freud-analysis  networkx
```

The analysis code works without the optional packages — those metrics are simply skipped.

Activate the project Python environment on this cluster:

```bash
source /nfs/scistore18/hannegrp/ugubbala/cell_env/bin/activate
```

---

## Step 1 — Build

Compile all simulation binaries:

```bash
bash load_mod_and_make.sh
```

This runs `make -Bj4`. Each `.cpp` file at the root level becomes a `.out` binary.
To compile only the main two scenarios:

```bash
make spv_pos_gamma_hetero_adhesion.out avm_pos_gamma_hetero_adhesion.out
```

---

## Step 2 — Run simulations

### Recommended: all-in-one

```bash
nohup bash run_all.sh > logs/master.log 2>&1 &
```

`run_all.sh` compiles, launches all 6 parallel simulations (SPV and AVM at γ = 0, 0.8, 1.5),
waits for them to finish, then runs the full analysis pipeline on every output file.
Follow progress with:

```bash
tail -f logs/master.log
```

### Key simulation flags

| Flag | Meaning |
|------|---------|
| `-t 100000` | number of time steps |
| `-e 0.0002` | time step dt |
| `-n 400` | total number of cells |
| `-v 3` | active-cell self-propulsion speed (v₀) |
| `-G γ -H γ` | adhesion contrast between cell types |
| `-i 3` | initial condition: cluster placed on the right, migrates left |
| `-x 1` | ensemble index (increment for multiple replicates) |

### Output file naming

Both models write descriptive filenames encoding all parameters, e.g.:

```
Data_t_100000_N_400_dt_0.0002_Ka_10_Kp_1_A0_1-1_P0_3.7-3.7_s0_3.7-3.7_
c_Size_8_a_Cells_4_va_3_vp_0_vb_0.01_T1_1.5_T2_1.5_T3_0_
Dr_b_0.05_Dr_tp_0.005_vD_tp_0.02_IC_3_ensem_1.nc
```

SPV files go to `DATA/POS_GAMMA_HETERO/`, AVM files go to `DATA/POS_GAMMA_VERTEX_HETERO/`.

---

## Step 3 — Analyse results

All analysis goes through one CLI: `PYTHON_ANALYSIS/cellgpu_analysis.py`.
Run it as a module from the project root so Python resolves imports correctly:

```bash
python -m PYTHON_ANALYSIS.cellgpu_analysis <subcommand> [options]
```

### Subcommands

#### `metrics` — extract per-frame and summary metrics

```bash
python -m PYTHON_ANALYSIS.cellgpu_analysis metrics \
    --input "DATA/POS_GAMMA_HETERO/Data_..._ensem_1.nc"
```

Outputs (created automatically):
- `RESULTS/PROCESSED_DATA/POS_GAMMA_HETERO/<file>_timeseries.csv`
- `RESULTS/PROCESSED_DATA/POS_GAMMA_HETERO/<file>_summary.json`
- `RESULTS/PROCESSED_DATA/ensemble_summary.csv` (updated in-place)

#### `plot` — physics time-series plot

```bash
python -m PYTHON_ANALYSIS.cellgpu_analysis plot \
    --timeseries "RESULTS/PROCESSED_DATA/POS_GAMMA_HETERO/<file>_timeseries.csv"
```

Output: `RESULTS/PLOTS/POS_GAMMA_HETERO/SPV_<file>_timeseries_plot.png`

#### `movie` — animated trajectory

```bash
python -m PYTHON_ANALYSIS.cellgpu_analysis movie \
    --input "DATA/POS_GAMMA_HETERO/<file>.nc" \
    --fps 12 --stride 10 --dpi 150
```

Output: `RESULTS/MOVIES/POS_GAMMA_HETERO_<file>.gif`
(GIF is used because ffmpeg is unavailable on this cluster; if ffmpeg is installed it saves `.mp4`)

#### `run-all` — metrics + plot + movie in one shot

```bash
python -m PYTHON_ANALYSIS.cellgpu_analysis run-all \
    --input "DATA/POS_GAMMA_HETERO/<file>.nc" \
    --voronoi --stride 1 --fps 5 --dpi 200
```

`--voronoi` enables Voronoi-based shape index calculation (requires `freud-analysis`).

---

## Physics — what the simulations measure

### Cell types and colours

| Type | Role | Colour |
|------|------|--------|
| 1 | Active / leader cells (self-propelled cluster) | Green `#06b780` |
| 2 | Passive / follower cells (dragged along) | Yellow `#ffcc7b` |
| 0 | Background tissue (stationary) | Light blue `#e9e9ff` |

### Migration direction

Cells start on the **right** of the periodic box and migrate **leftward** (−x).
Active cells drive collective motion; passive cells trail behind.
When the cluster crosses x = 0 going left it wraps to x = Lx (periodic boundary).

### Adhesion parameter γ

The simulations sweep γ ∈ {0, 0.8, 1.5}:

| γ | Meaning |
|---|---------|
| 0 | No adhesion contrast — active/passive interfaces identical to bulk |
| 0.8 | Moderate heterotypic adhesion |
| 1.5 | Strong heterotypic adhesion — increased interfacial tension |

Higher γ changes cluster speed, cohesion, and active–passive interface shape.

### Metrics computed per frame

| Metric | Description |
|--------|-------------|
| `cluster_disp_x` | Cluster COM x displacement (negative = moved left = physically correct) |
| `cluster_disp_y` | Cluster COM y displacement |
| `cluster_speed` | Magnitude of COM velocity |
| `polarity` | Fraction of active cells that are to the left of the passive COM |
| `director_alignment` | Mean cos(θ) between cell self-propulsion director and −x axis |
| `msd` | Mean squared displacement of cluster COM |
| `shape_index` | Perimeter/√area per cell (requires `--voronoi` flag for SPV) |

---

## Tests

Run from the project root:

```bash
python -m unittest discover -s PYTHON_ANALYSIS/tests -p "test_*.py" -v
```

Expected: **25 tests, OK** (~2 seconds).

The tests include physics-grounded checks (not just "does it run"):
- `test_cluster_moves_in_negative_x` — cluster COM displacement in x must be negative
- `test_active_cell_leads_passive` — active COM must be further left than passive COM
- `test_com_x_decreases_monotonically` — no frame where the cluster reverses direction
- `test_polarity_above_half_when_active_leads` — polarity > 0.5 when active clearly leads
- `test_msd_grows_with_time_for_drifting_cells` — MSD must grow as cells drift

---

## Reproduce all results from scratch

```bash
# 1. Build
bash load_mod_and_make.sh

# 2. Run everything (simulate + analyse), background job
nohup bash run_all.sh > logs/master.log 2>&1 &

# 3. Watch
tail -f logs/master.log
```

When complete you will have:
- 6 `.nc` trajectory files in `DATA/`
- 6 time-series CSVs + JSON summaries in `RESULTS/PROCESSED_DATA/`
- 6 physics plots in `RESULTS/PLOTS/`
- 6 animated GIFs in `RESULTS/MOVIES/`

---

## Upstream references

This workspace is built on **CellGPU version 0.8.0** by Daniel M. Sussman et al.

- Documentation: https://dmsussman.gitlab.io/cellGPUdocumentation
- arXiv: https://arxiv.org/abs/1702.02939
- Journal: http://www.sciencedirect.com/science/article/pii/S0010465517301832

See `CITATIONS.md` for citation policy and `LICENSE.md` for the MIT license.

---

## Notes

- **ffmpeg not available on this cluster** → movies are saved as `.gif` via Pillow.
  Install ffmpeg and re-run `run_all.sh` if `.mp4` output is needed.
- Large `.nc` data files and RESULTS are excluded from git (see `.gitignore`).
  Only source, scripts, and tests are committed.
- To add more ensemble replicates, change `-x <index>` and re-run.
- Nodal and neg-gamma variants (`spv_nodal_*`, `*_neg_gamma_*`) are compiled but not
  included in the default `run_all.sh` sweep; add them manually if needed.
