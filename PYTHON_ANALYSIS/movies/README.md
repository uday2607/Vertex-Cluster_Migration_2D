# Movies

One script makes every movie of the paper: `make_grid_movie.py` reads a small JSON spec from `specs/` and the example trajectories in
`DATA/EXAMPLES/` (see the README there for the file names).

```bash
python PYTHON_ANALYSIS/movies/make_grid_movie.py fig3                 # -> RESULTS/movies/fig3.mp4
python PYTHON_ANALYSIS/movies/make_grid_movie.py fig5b --out movie.mp4
python PYTHON_ANALYSIS/movies/make_grid_movie.py --spec my_spec.json --out movie.mp4
```

| spec | movie |
|---|---|
| `fig1` | SPV cluster caged in a jammed tissue next to one internalized in a fluid tissue |
| `fig2` | 2x3: SPV and AVM x fragmentation / optimal interfacial tension / pinning |
| `fig2_pinning` | SPV and AVM pinning, fixed zoomed view: position axis, start line, centre trail, displacement, cluster-tissue edges coloured by length |
| `fig3` | polarity formation: 4 leaders + 4 passive followers, leaders end up in front |
| `fig4` | 4-cell cluster, control above E-cadherin knockdown (splits) |
| `fig5a` | H+M against H+L, 4 + 4 cells |
| `fig5b` | H+M+L, 3 + 3 + 3 cells, positive and negative interaction rule |

- `movie_core.py` — loaders (SPV and AVM are detected from the variables in the `.nc`), palette, equilibration detection and a plain
  single-panel renderer: `python PYTHON_ANALYSIS/movies/movie_core.py in.nc out.mp4`.
- `thin_example.py` — turns a full run into a small example file (every equilibration frame, then one production frame per `--stride`).

Conventions: cells of the migrating cluster green (H, M, L use their own colours in fig5), wall cells pinned and drawn lilac, red arrows =
motility direction, **drawn only after equilibration**; equilibration frames play fast, production frames slowly (`frame_sec` in the spec).
Cells are drawn with periodic images and the view is a rectangular cut of the box, so no cell is cut at the box edge.

## Spec fields

Required: `title`, `subtitle`, `nrow`, `ncol`, `files` (row-major; example ids such as `Fig3-SPV-polarity-formation`, or paths to `.nc` files), `tags`
(text in each panel's corner). Optional: `col_headers`, `row_labels`, `axis` (x axis under the bottom row), `panel_w`, `left`, `crop_h` (height of the
cut-out rectangle in box units, default 12), `crop_x` `[x0, x1]`, `fixed_window` `[width, height]`, `axes` (start line, trail, displacement),
`interface` (`true`, or `"length"` to colour the cluster-tissue edges by length), `periodic_x`, `legend_wall`, `wall` (`false`: no wall colouring),
`legend_items` `[[colour, label], ...]`, `type_colors` `{"1": "#hex", ...}`, `face2` (colour of the type-2 cells per panel), `stop_before_x` (end the
movie before a cluster cell gets closer to the wall than this; one value or one per panel, `null` = no limit), `absolute_time` (time label counted from
the start of the run instead of from the end of equilibration), `last_frame`, `frame_sec` (seconds per production frame).

Needs numpy, scipy, matplotlib, freud-analysis, numba and `ffmpeg` on the PATH. The Lato font is used if installed (otherwise DejaVu Sans).
