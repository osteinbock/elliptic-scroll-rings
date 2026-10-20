#!/bin/sh
# Rebuild every table, figure, and the data archive from the production runs, in order.
#   sh build_all.sh SUPP BARKLEY REFINE [ARCHIVE]
# SUPP: folder of runs_v2.sh outputs (Panfilov-Rudenko, Atri), BARKLEY: folder of
# runs_barkley_v2.sh outputs, ARCHIVE: optional output folder for export_data.py.
# Requires a POSIX shell (sh, grep, seq, xargs) and Python with NumPy, SciPy, Numba, Matplotlib.
set -e
S=${1:?supplementary-model folder}
B=${2:?Barkley folder}
RF=${3:?refinement folder (runs_refine.sh outputs)}
HERE=$(cd "$(dirname "$0")" && pwd)

python "$HERE/mask_frames.py" "$S"/*.npz "$B"/*.npz "$B"/*/*.npz "$RF"/*.npz

# Barkley model: Figs. 1-6 and the numbers of Secs. II-III
(
  cd "$B"
  export PYTHONPATH="$HERE"
  python "$HERE/alpha_stationarity.py"          # alpha_late.npy, windows for Fig. 4
  python "$HERE/period_analysis_r2.py"          # period_table_r2.csv (Fig. 5)
  python "$HERE/mode_decompose.py"              # mode_decompose.npy, mode_sweep.npy (Fig. 6a)
  python "$HERE/new_tests_analysis.py"          # new_tests.npz (tilt, larger ring)
  python "$HERE/sf2_linear.py"                  # sf2/sf2_linear.npy (Fig. 6b)
  python "$HERE/buckle_save.py"                 # sf2/buckle.npz (Fig. 6b)
  python "$HERE/conv_compare.py"                # resolution and box-size checks (Sec. II)
  python "$HERE/diffrot_check.py"               # tip-phase twist check (Sec. III D)
  python "$HERE/fig1_render.py"                 # fig1_fields/render_t12.npy (Fig. 1b)
  mkdir -p figs
  FIG_OUT=figs python "$HERE/make_figs.py"      # figs/fig1.pdf ... fig6.pdf
)

# Supplementary models: Tables S1-S4, Figs. S1-S2
python "$HERE/halfcycle_tables.py" "$S"
python "$HERE/pan_figure.py" "$S" "$S/figS1.png" "$B/period_table_r2.csv"
python "$HERE/atri_figure.py" "$S" "$S/figS2.png" "$B/period_table_r2.csv"

# grid refinement of the mode couplings and straight-filament dispersion
python "$HERE/refine_compare.py" "$B" "$RF"

# numbers quoted in the manuscript and SI
python "$HERE/paper_numbers.py" "$S" "$B" "$RF"

if [ -n "$4" ]; then
  python "$HERE/export_data.py" "$4" "$S" "$B" "$RF"
fi
