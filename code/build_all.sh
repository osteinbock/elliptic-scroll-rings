#!/bin/sh
# Rebuild every table, figure, and the data archive from the production runs, in order.
#   sh build_all.sh SUPP BARKLEY REFINE REVISION [ARCHIVE]
# SUPP: folder of runs_v2.sh outputs (Panfilov-Rudenko, Atri), BARKLEY: folder of runs_barkley_v2.sh
# outputs, REFINE: folder of runs_refine.sh outputs, REVISION: folder of runs_revision.sh outputs,
# ARCHIVE: optional output folder for export_data.py. The analyses find the four folders through the
# environment variables of the same names (folders.py).
# Requires a POSIX shell (sh, grep, seq, xargs) and Python with NumPy, SciPy, Numba, Matplotlib.
set -e
S=${1:?supplementary-model folder}
B=${2:?Barkley folder}
RF=${3:?refinement folder (runs_refine.sh outputs)}
RV=${4:?revision folder (runs_revision.sh outputs)}
HERE=$(cd "$(dirname "$0")" && pwd)
SUPP=$(cd "$S" && pwd); BARKLEY=$(cd "$B" && pwd); REFINE=$(cd "$RF" && pwd); REVISION=$(cd "$RV" && pwd)
export SUPP BARKLEY REFINE REVISION
export PYTHONPATH="$HERE"

python "$HERE/mask_frames.py" "$S"/*.npz "$B"/*.npz "$B"/*/*.npz "$RF"/*.npz "$RV"/*.npz "$RV"/*/*.npz

# Barkley model: coefficients, half-cycles, shape modes, straight filaments, checks
(
  cd "$BARKLEY"
  python "$HERE/alpha_stationarity.py"          # alpha_late.npy
  rm -f amp_factor.npy                          # cache of the filament equation: recompute with the current code
  python "$HERE/period_analysis_r2.py"          # period_table_r2.csv, amp_factor.npy
  python "$HERE/mode_decompose.py"              # mode_decompose.npy, mode_sweep.npy
  python "$HERE/new_tests_analysis.py"          # new_tests.npz (tilt, larger ring)
  python "$HERE/sf2_linear.py"                  # sf2/sf2_linear.npy (Fig. 6b)
  python "$HERE/buckle_save.py"                 # sf2/buckle.npz (Fig. 6b)
  python "$HERE/conv_compare.py"                # resolution and box-size checks
  python "$HERE/diffrot_check.py"               # tip-phase twist check
  python "$HERE/fig1_render.py"                 # fig1_fields/render_t12.npy (Fig. 1b)
)

# supplementary models: half-cycles and circular-ring coefficients
python "$HERE/halfcycle_tables.py" "$SUPP"

# grid refinement of the mode couplings and straight-filament dispersion
python "$HERE/refine_compare.py" "$BARKLEY" "$REFINE"

# revision analyses (one half-cycle pipeline for all models, shape modes, controls)
(cd "$REVISION" && python "$HERE/mode_definitions.py")              # mode_definitions.json
python "$HERE/mode_wavenumber.py"                                   # REVISION/mode_wavenumber.json
python "$HERE/controls_check.py"                                    # REVISION/controls.json
python "$HERE/halfcycle_uniform.py" "$REVISION/uniform_halfcycles.csv"
(
  cd "$BARKLEY"
  python "$HERE/period_baselines.py"                                # period_table_baselines.csv
  python "$HERE/mode_refit.py"                                      # mode_refit.json
  python "$HERE/rigidity_test.py"                                   # rigidity_test.json
  python "$HERE/local_law_halfcycles.py" local_law_halfcycles.csv  # local_law_halfcycles.csv
)

# figures and tables of the paper
(cd "$BARKLEY" && mkdir -p figs && FIG_OUT=figs python "$HERE/make_figs.py")   # figs/fig1.pdf ... figS4.pdf
python "$HERE/si_tables.py" "$BARKLEY/tables"                                  # tables/tab_*.tex (SI)

# numbers quoted in the manuscript and SI
python "$HERE/paper_numbers.py" "$SUPP" "$BARKLEY" "$REFINE" "$REVISION"

if [ -n "$5" ]; then
  python "$HERE/export_data.py" "$5" "$SUPP" "$BARKLEY" "$REFINE" "$REVISION"
fi
