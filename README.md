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

- `make_figs.py`: figures of the main text and of the supplementary material (`FIG_OUT` sets the output directory)
- `si_tables.py`: tables of the supplementary material
- `folders.py`: locations of the four data folders, read from the environment variables `SUPP`, `BARKLEY`,
  `REFINE`, and `REVISION` (set by `build_all.sh` and `rebuild_from_archive.py`)
- `fig1_render.py`: rendered wave field of Fig. 1(b)
- `pan_ring.py`, `pan_alpha.py`, `pan_modes.py`, `pan2d.py`: Panfilov–Rudenko model
- `atri_ring.py`, `atri_alpha.py`: Atri model
- `halfcycle.py`, `halfcycle_tables.py`: half-cycles and circular-ring coefficients of both supplementary models
- `halfcycle_uniform.py`: one half-cycle analysis for all three models, with the circular-ring baseline, the
  curvature-only local law (Eq. 5), the ring's own drift, and the literal ratio of durations (Fig. 5,
  Tables S1–S2, S12, S14)
- `period_baselines.py`, `local_law_halfcycles.py`: both baselines for the Barkley half-cycles; the
  curvature-only law at finite amplitude
- `oreg2d.py`, `oreg_ring.py`, `oreg_ring_weak.py`: Oregonator
- `circle_run.py`, `alpha_stationarity.py`: tension and drift of circular rings (Fig. 4, Fig. S1, Table S3)
- `wavelength2d.py`: period, core radius, and wavelength of two-dimensional spirals (Table S4)
- `period_analysis.py`, `period_analysis_r2.py`: half-cycles between axis switches (Fig. 5)
- `mode_decompose.py`, `mode_refit.py`, `mode_definitions.py`, `new_tests_analysis.py`, `straight_filament.py`,
  `sf2_linear.py`, `buckle_save.py`, `rigidity_test.py`: shape modes, their couplings and damping, the
  filament definition, and the rigidity-extended local law (Fig. 6, Fig. S2, Tables S5–S10);
  `mode_summary.py` collects the couplings and diagonal rates with their ranges, relative to the drift
  coefficient of circular rings of the same radius
- `mode_wavenumber.py`: dependence of the out-of-plane coupling on the wavenumber, on large rings at
  b = 0.17 and on shrinking rings at b = 0.15 and 0.16 (Fig. 6b, Table S11)
- `twist_run.py`, `diffrot_check.py`: filament twist
- `controls_check.py`: controls of the revision runs (domain size, time step, seed orientation, straight
  filaments at b = 0.15 and 0.16)

Scripts whose docstring begins with "EXPLORATORY" (and `pan_compare.py`, `atri_compare.py`,
`beta_ratio_plot.py`, and the earlier supplementary figures `pan_figure.py` and `atri_figure.py`) are
earlier or side analyses that feed no figure, table, or number of the paper.

## Data

`data/` holds the processed data behind the figures, tables, and quoted numbers:

- `data/runs/`: per-frame tables of all 94 production simulations, of the 5 grid-refinement runs,
  and of the 39 runs added in the revision (ring measures, validity flags, pre-mask radius,
  extraction diagnostics, run metadata); filament snapshots and fields are omitted;
- `data/analysis/`: the analysis products (half-cycle tables with both baselines, coefficient tables,
  mode fits and their sensitivity sweeps, filament definitions and twist, the rigidity test, controls,
  two-dimensional spirals, straight-filament and buckling fits, `paper_numbers.json` with every
  number quoted in the manuscript and SI that the archived runs determine);
- `data/MANIFEST.json`: for each file, its SHA-256 and the command that produced it, plus the
  commit whose simulation modules produced the runs (the revision runs record the digest of their
  simulation modules themselves; hashes refer to the stored bytes; git keeps `data/` unconverted,
  see `.gitattributes`).

Not covered by the archive: the rotation periods of the two-dimensional spirals, the
Panfilov–Rudenko core radii, and the Atri spiral data, which come from earlier two-dimensional
runs and enter the analysis as constants (`period_analysis.py`, `pan_alpha.py`,
`halfcycle_tables.py`, `si_tables.py`), and the exploration of the Oregonator in the supplementary
material.

`python code/rebuild_from_archive.py data` recomputes every quoted number from these tables alone
(no simulations or snapshots; the buckling and tip-phase checks use their saved series, the
two-dimensional spirals their archived analysis) and reports any difference from
`data/analysis/barkley/paper_numbers.json`.

## Reproducing the results

The full simulations write their raw output (`.npz`, with snapshots) to the working directory;
these files are not part of the repository.

1. Run the simulations listed in `runs_barkley_v2.sh` (Barkley model), `runs_v2.sh`
   (Panfilov–Rudenko and Atri models), `runs_refine.sh` (grid refinement), and `runs_revision.sh`
   (circular-ring calibrations at b = 0.15–0.18, mode runs with other filament definitions and the
   rotation phase, controls, two Panfilov–Rudenko circular rings, two-dimensional spirals), one command per line with all parameters,
   from the code directory (the lists write to `v2b/`, `v2/`, `v5r/`, and `v6/`; create the
   subfolders named in their headers).
2. `sh build_all.sh v2 v2b v5r v6 [archive_dir]` then marks invalid frames (`mask_frames.py`), runs
   every analysis in order, compares the grids (`refine_compare.py`), writes Figs. 1–6 and
   Figs. S1–S4 (`make_figs.py`) and the tables of the supplementary material (`si_tables.py`),
   recomputes all quoted numbers (`paper_numbers.py`, which exits with an error and keeps the
   previous result if any part fails), and optionally writes the data archive (`export_data.py`,
   which only reads its input folders).

`python tests/test_regression.py` runs the regression tests (filament extraction, recentering,
smoothing and gap handling, mode fits, the Nyquist mode of the filament solver, the half-cycle
phase integral and calibration, the data-folder settings, stale-output guards); no data files are
needed.
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
(`figS1`–`figS4`), and the multimedia file of Fig. 1.

## License

MIT (see `LICENSE`).
