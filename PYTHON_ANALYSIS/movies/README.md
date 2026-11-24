# Movies

One script makes every movie of the paper: `make_grid_movie.py` reads a small JSON spec from `specs/` and the example trajectories in
`DATA/EXAMPLES/` (see the README there for the file names).

```bash
python PYTHON_ANALYSIS/movies/make_grid_movie.py fig3                 # -> RESULTS/movies/fig3.mp4
python PYTHON_ANALYSIS/movies/make_grid_movie.py fig5b --out movie.mp4
python PYTHON_ANALYSIS/movies/make_grid_movie.py --spec my_spec.json --out movie.mp4
```

| spec | Supplementary Movie | movie |
|---|---|---|
| `fig1` | 1 | SPV cluster caged in a jammed tissue next to one internalized in a fluid tissue |
| `fig2` | 2 | 2x3: SPV and AVM x Fragmentation / Collective Internalization / Kinetic Pinning |
| `fig2_pinning` | 3 | SPV and AVM pinning, fixed zoomed view: position axis, start line, centre trail, displacement, cluster-tissue edges coloured by length |
| `fig3` | 4 | Polarity Formation: 4 leaders + 4 passive followers, leaders end up in front |
| `fig4` | 5 | 4-cell cluster, Control MO above E-Cad MO (splits) |
| `fig5a` | 6 | H+M against H+L, 4 + 4 cells |
| `fig5b` | 7 | H+M+L, 3 + 3 + 3 cells, Heterotypic Interactions (positive rule) and Differential Adhesion (negative rule) |

**Movies and figures are not identical.** Each movie shows one representative run, so its parameters or panels can differ from the corresponding
figure. Movie 2 (`fig2`) uses SPV gamma = 0.2 / v0 = 1.5 and AVM gamma = 2 / v0 = 0.75 for the collective-internalization column, which are not the
values written in the Fig. 2 legend. The parameters actually used are in `DATA/EXAMPLES/MANIFEST.csv` and in the file names.

`DATA/EXAMPLES/MANIFEST.csv` links the two sides: every example file lists the Supplementary Movie and spec it is used in, its panel label
(the heading or row label shown in the movie) and its parameters.

- `movie_core.py` — loaders (SPV and AVM are detected from the variables in the `.nc`), palette, equilibration detection and a plain
  single-panel renderer: `python PYTHON_ANALYSIS/movies/movie_core.py in.nc out.mp4`.
- `thin_example.py` — turns a full run into a small example file (every equilibration frame, then one production frame per `--stride`).

Conventions: cells of the migrating cluster green (H, M, L use their own colours in fig5), wall cells pinned and drawn lilac, red arrows =
motility direction, **drawn only after equilibration**; equilibration frames play fast, production frames slowly (`frame_sec` in the spec).
Cells are drawn with periodic images and the view is a rectangular cut of the box, so no cell is cut at the box edge.

## Spec fields

Required: `nrow`, `ncol`, `files` (row-major; example ids such as `Fig3-SPV-polarity-formation`, or paths to `.nc` files), `tags`
(text in each panel's corner). Optional: `col_headers`, `row_labels`, `axis` (x axis under the bottom row), `panel_w`, `left`, `crop_h` (height of the
cut-out rectangle in box units, default 12), `crop_x` `[x0, x1]`, `fixed_window` `[width, height]`, `axes` (start line, trail, displacement),
`interface` (`true`, or `"length"` to colour the cluster-tissue edges by length), `periodic_x`, `legend_wall`, `wall` (`false`: no wall colouring),
`legend_items` `[[colour, label], ...]`, `type_colors` `{"1": "#hex", ...}`, `face2` (colour of the type-2 cells per panel), `stop_before_x` (end the
movie before a cluster cell gets closer to the wall than this; one value or one per panel, `null` = no limit), `legend_two_rows` (legend on its own row below the time label), `absolute_time` (time label counted from
the start of the run instead of from the end of equilibration), `last_frame`, `frame_sec` (seconds per production frame).

Needs numpy, scipy, matplotlib, freud-analysis, numba and `ffmpeg` on the PATH. The Lato font is used if installed (otherwise DejaVu Sans).
