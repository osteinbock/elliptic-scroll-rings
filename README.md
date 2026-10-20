# Elliptic scroll rings

Simulation and analysis code for the manuscript *Axis-switching oscillations of elliptic
scroll rings in excitable media* (O. Steinbock, in preparation).

Initially planar, elliptic scroll rings in three-dimensional excitable media perform the
axis-switching oscillation known from elliptic vortex rings: while the ring drifts along its
axis, it buckles into a saddle, becomes planar again with interchanged axes, and repeats this
cycle. The code simulates such rings in three reaction–diffusion models, measures the
coefficients of the kinematic filament law on circular rings, and compares the observed
oscillation with the filament equation.

## Models

| Module | Model | Notes |
|---|---|---|
| `barkley3d.py`, `ellipse_ring.py` | Barkley model | main model of the paper; periodic 3D domains, filament tracking |
| `panfilov.py` | piecewise-linear kinetics of Panfilov & Rudenko (1987) | second model |
| `atri.py` | Atri et al. (1993) model of Ca²⁺ waves in *Xenopus* oocytes | units µm, s |
| `oregonator.py` | two-variable Oregonator | regimes examined; see SI |
| `filament_flow.py` | filament equation ∂X/∂t = αKN + βKB (+ optional rigidity) | pseudo-spectral solver |

All models use explicit Euler time stepping and a seven-point Laplacian (Numba-parallelized).
`ellipse_ring.run(..., model=...)` accepts the Panfilov, Atri, and Oregonator models as plug-ins.

## Main scripts

- `make_figs.py`: figures of the main text (`FIG_OUT` sets the output directory)
- `fig1_render.py`: rendered wave field of Fig. 1(b)
- `pan_ring.py`, `pan_alpha.py`, `pan_figure.py`, `pan_modes.py`: Panfilov–Rudenko model
- `atri_ring.py`, `atri_alpha.py`, `atri_figure.py`: Atri model
- `halfcycle.py`, `halfcycle_tables.py`: half-cycle analysis of both supplementary models (Tables S1–S4)
- `oreg2d.py`, `oreg_ring.py`: Oregonator
- `circle_run.py`, `alpha_stationarity.py`: tension and drift of circular rings (Fig. 4)
- `period_analysis.py`, `period_analysis_r2.py`: observed vs predicted half-periods (Fig. 5)
- `mode_decompose.py`, `new_tests_analysis.py`, `straight_filament.py`, `sf2_linear.py`, `buckle_save.py`: shape modes (Fig. 6)
- `twist_run.py`, `diffrot_check.py`: filament twist

Scripts whose docstring begins with "EXPLORATORY" (and `pan_compare.py`, `atri_compare.py`,
`beta_ratio_plot.py`) are earlier or side analyses that feed no figure, table, or number of the paper.

## Data

`data/` holds the processed data behind every figure, table, and quoted number:

- `data/runs/`: per-frame tables of all 94 production simulations and of the 5 grid-refinement
  runs (ring measures, validity flags, pre-mask radius, extraction diagnostics, run metadata);
  filament snapshots and fields are omitted;
- `data/analysis/`: the analysis products (half-cycle tables, coefficient tables, mode fits and
  their sensitivity sweep, straight-filament and buckling fits, `paper_numbers.json` with every
  number quoted in the manuscript and SI);
- `data/MANIFEST.json`: for each file, its SHA-256 and the command that produced it, plus the
  commit whose simulation modules produced the runs (hashes refer to the stored bytes; git keeps
  `data/` unconverted, see `.gitattributes`).

`python code/rebuild_from_archive.py data` recomputes every quoted number from these tables alone
(no simulations or snapshots; the buckling and tip-phase checks use their saved series) and
reports any difference from `data/analysis/barkley/paper_numbers.json`.

## Reproducing the results

The full simulations write their raw output (`.npz`, with snapshots) to the working directory;
these files are not part of the repository.

1. Run the simulations listed in `runs_barkley_v2.sh` (Barkley model), `runs_v2.sh`
   (Panfilov–Rudenko and Atri models), and `runs_refine.sh` (grid refinement), one command per
   line with all parameters, from the code directory (the lists write to `v2b/`, `v2/`, and `v5r/`;
   create the subfolders named in their headers).
2. `sh build_all.sh v2 v2b v5r [archive_dir]` then marks invalid frames (`mask_frames.py`), runs
   every analysis in order, writes Figs. 1–6, Tables S1–S4 and Figs. S1–S2, compares the grids
   (`refine_compare.py`), recomputes all quoted numbers (`paper_numbers.py`, which exits with an
   error and keeps the previous result if any part fails), and optionally writes the data archive
   (`export_data.py`, which only reads its input folders).

`python tests/test_regression.py` runs the regression tests (filament extraction, recentering,
smoothing and gap handling, mode fits, stale-output guards); no data files are needed.
`zfix_repair.py` corrects the ring height in tables written before a bookkeeping fix of
`ellipse_ring.run`; `stitch_validate.py` checks the joining of fragmented filaments in saved
snapshots.

## Requirements

Python 3.12, NumPy, SciPy, Numba, Matplotlib (Plotly only for interactive renderings). The run
lists and `build_all.sh` assume a POSIX shell with `sh`, `seq`, `grep`, and `xargs` (on Windows,
Git Bash). The results in `data/` were produced with Python 3.12, NumPy 2.3.3, SciPy 1.17.1, Numba
0.65.0, and Matplotlib 3.10.8.

## Figures

`figures/` contains the figures of the main text (`fig1`–`fig6`), of the supplementary material
(`figS1`, `figS2`), and the multimedia file of Fig. 1.

## License

MIT (see `LICENSE`).
