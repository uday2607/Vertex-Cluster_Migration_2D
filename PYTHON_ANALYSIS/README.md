# Python Analysis Workflow

This folder provides post-processing tools for CellGPU `.nc` simulation outputs.

## Main entry point

Use the unified CLI:

```bash
python cellgpu_analysis.py -h
```

## Typical usage

### 1) Compute metrics for one file

```bash
python cellgpu_analysis.py metrics --input "../DATA/SPV_POS_GAMMA/Data_..._ensem_1.nc"
```

### 2) Compute metrics for many files

```bash
python cellgpu_analysis.py metrics --input "../DATA/SPV_POS_GAMMA/*.nc"
```

### 3) Compute metrics for an ensemble pattern

```bash
python cellgpu_analysis.py metrics --input "../DATA/SPV_POS_GAMMA/Data_..._ensem_{i}.nc" --ensemble-count 100
```

### 4) Plot a generated metrics CSV

```bash
python cellgpu_analysis.py plot --timeseries "../PROCESSED_DATA/SPV_POS_GAMMA/<file>_timeseries.csv"
```

### 5) Make a movie from one simulation

```bash
python cellgpu_analysis.py movie --input "../DATA/SPV_POS_GAMMA/<file>.nc" --fps 12 --stride 2
```

### 6) Run everything for one simulation

```bash
python cellgpu_analysis.py run-all --input "../DATA/SPV_POS_GAMMA/<file>.nc"
```

## Outputs

- Per-file metrics time series: `PROCESSED_DATA/<scenario>/<file>_timeseries.csv`
- Per-file summary: `PROCESSED_DATA/<scenario>/<file>_summary.json`
- Ensemble summary table: `PROCESSED_DATA/ensemble_summary.csv`
- Ensemble aggregate stats: `PROCESSED_DATA/ensemble_aggregates.json`
- Per-file metrics time series (default): `RESULTS/PROCESSED_DATA/<scenario>/<file>_timeseries.csv`
- Per-file summary (default): `RESULTS/PROCESSED_DATA/<scenario>/<file>_summary.json`
- Ensemble summary table (default): `RESULTS/PROCESSED_DATA/ensemble_summary.csv`
- Ensemble aggregate stats (default): `RESULTS/PROCESSED_DATA/ensemble_aggregates.json`
- Plots (default): `RESULTS/PLOTS/<scenario>/<file>_timeseries_plot.png`
- Movies (default): `RESULTS/MOVIES/<file>.mp4` or fallback `RESULTS/MOVIES/<file>.gif`

## Legacy script compatibility

The old script names are still available and now call the unified CLI:

- `compute_metrics_raw_data.py` -> `metrics`
- `compute_speed_and_force_raw_data.py` -> `metrics`
- `make_basic_voronoi_movie.py` -> `movie`
- `plot_2D_metrics_basic_voronoi_vary_gamma_pub.py` -> `plot`

## Dependencies

- Python 3.8+
- numpy
- scipy
- matplotlib
- ffmpeg (optional, for `.mp4`; otherwise `.gif` is used)

## Unit tests

From repository root:

```bash
python -m unittest discover -s PYTHON_ANALYSIS/tests -p "test_*.py"
```

## Generate reusable example .nc files

```bash
python PYTHON_ANALYSIS/generate_example_nc_files.py --output-dir DATA/EXAMPLES --overwrite
```

## Movies

Publication-style movies (single or multi-panel, SPV and AVM) live in `movies/`, see `movies/README.md`.
