# Barkley-model runs (eps = 0.02, dx = 0.2 unless noted) behind Figs. 1-6 and the quantitative
# statements of chaos_paper/main.tex, rerun after the audit fixes (z bookkeeping, longest-curve
# measure, closed flag, saved metadata). Same parameters and seeds (all runs are deterministic) as
# the original runs; output goes to v2b/ with the original relative names. Commands were traced from
# the original run logs.
# One command per line, longest groups first. Run from the code directory:
#   mkdir -p v2b/rerun_r2 v2b/sf2 v2b/conv v2b/fig1_fields v2b/movie
#   grep -v '^#' runs_barkley_v2.sh | grep . | NUMBA_NUM_THREADS=4 xargs -P 5 -I{} sh -c '{}'
# Lines must stay free of quote characters (xargs strips them). The 2D runs at the end and the
# unused runs are commented out.
#
# Buckling ring R0 = 38 (Fig. 6b triangles via buckle_save.py -> sf2/buckle.npz; Fig. 7) and large rings (Fig. 4b,c)
cd v2b/sf2 && python -u ../../ellipse_ring.py --a 1.1 --b 0.17 --eps 0.02 --Ax 38 --Ay 38 --Lxy 92 --Lz 24 --T 200 --every 0.25 --snaps $(seq 40 2.5 200) --out circXXL_R0_38_snaps.npz > circXXL_R0_38_snaps.log 2>&1
# (circXXL_R0_38.npz is the same simulation as circXXL_R0_38_snaps.npz; consumers read the _snaps file)
cd v2b/sf2 && python -u ../../circle_run.py 1.1 0.17 0.02 20 60 24 200 circXL_R0_20.npz > circXL_R0_20.log 2>&1
cd v2b/sf2 && python -u ../../circle_run.py 1.1 0.17 0.02 16 52 24 200 circXL_R0_16.npz > circXL_R0_16.log 2>&1
# Convergence checks (Sec. II: dx = 0.1 and 60 x 60 x 30 box; conv_compare.py)
cd v2b/conv && python -u ../../ellipse_ring.py --a 1.1 --b 0.17 --eps 0.02 --Ax 12 --Ay 7.5 --Lxy 60 --Lz 30 --T 240 --every 0.25 --out ell_12x7.5_bigbox.npz > ell_12x7.5_bigbox.log 2>&1
cd v2b/conv && python -u ../../ellipse_ring.py --a 1.1 --b 0.17 --eps 0.02 --Ax 8 --Ay 5 --Lxy 28 --Lz 16 --dx 0.1 --T 120 --every 0.25 --out ell_8x5_dx0.1.npz > ell_8x5_dx0.1.log 2>&1
# Straight filaments, A = 0.5 (Fig. 6b blue via sf2_linear.py; side-wall size 32 vs 64 and k = 0 controls via sf2_analysis.py)
cd v2b/sf2 && python -u ../../straight_filament.py 126 0.5 120 L64_k025.npz 64 > L64_k025.log 2>&1
cd v2b/sf2 && python -u ../../straight_filament.py 90 0.5 120 L64_k035.npz 64 > L64_k035.log 2>&1
cd v2b/sf2 && python -u ../../straight_filament.py 70 0.5 100 L64_k045.npz 64 > L64_k045.log 2>&1
cd v2b/sf2 && python -u ../../straight_filament.py 126 0.5 120 L32_k025.npz 32 > L32_k025.log 2>&1
cd v2b/sf2 && python -u ../../straight_filament.py 52 0.5 80 L64_k060.npz 64 > L64_k060.log 2>&1
cd v2b/sf2 && python -u ../../straight_filament.py 90 0.5 120 L32_k035.npz 32 > L32_k035.log 2>&1
cd v2b/sf2 && python -u ../../straight_filament.py 40 0.5 70 L64_k079.npz 64 > L64_k079.log 2>&1
cd v2b/sf2 && python -u ../../straight_filament.py 10 0 150 L64_k0.npz 64 > L64_k0.log 2>&1
cd v2b/sf2 && python -u ../../straight_filament.py 70 0.5 100 L32_k045.npz 32 > L32_k045.log 2>&1
cd v2b/sf2 && python -u ../../straight_filament.py 52 0.5 80 L32_k060.npz 32 > L32_k060.log 2>&1
cd v2b/sf2 && python -u ../../straight_filament.py 40 0.5 70 L32_k079.npz 32 > L32_k079.log 2>&1
cd v2b/sf2 && python -u ../../straight_filament.py 10 0 150 L32_k0.npz 32 > L32_k0.log 2>&1
# Long circular rings, T = 300 (Fig. 4a,b,c; alpha_stationarity.py)
cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 13 40 16 300 circL_R0_13.npz > circL_R0_13.log 2>&1
cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 9 32 16 300 circL_R0_9.npz > circL_R0_9.log 2>&1
cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 6 26 16 300 circL_R0_6.npz > circL_R0_6.log 2>&1
# Resolution and box-size controls (Sec. II and Sec. III.B: dx = 0.15, 40 x 40 x 32 box, Lz = 32)
cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 6.5 40 32 150 ctrl_circ_R0_6.5_bigbox.npz 0.2 > ctrl_circ_R0_6.5_bigbox.log 2>&1
cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 6.5 28 18 150 ctrl_circ_R0_6.5_dx015.npz 0.15 > ctrl_circ_R0_6.5_dx015.log 2>&1
cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 9 32 16 120 ctrl_circ_R0_9_dx015.npz 0.15 > ctrl_circ_R0_9_dx015.log 2>&1
cd v2b && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --Ax 8 --Ay 5 --Lxy 28 --Lz 16 --dx 0.15 --T 50 --every 0.25 --out ctrl_ell_8x5_dx015.npz > ctrl_ell_8x5_dx015.log 2>&1
cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 11 36 32 90 circ_R0_11_Lz32.npz > circ_R0_11_Lz32.log 2>&1
# Elliptic rings with r_2 modes: period_table_r2.csv (Fig. 5, 27 half-cycles; period_analysis_r2.py r2 method)
cd v2b/rerun_r2 && python -u ../../ellipse_ring.py --a 1.1 --b 0.17 --eps 0.02 --Ax 12 --Ay 7.5 --Lxy 40 --Lz 20 --T 240 --every 0.25 --snaps $(seq 4 4 240) --out ell_a1.1_b0.17_12x7.5.npz > ell_a1.1_b0.17_12x7.5.log 2>&1
cd v2b && python -u ../ellipse_ring.py --a 1.1 --b 0.18 --Ax 12 --Ay 7.5 --Lxy 40 --Lz 20 --T 220 --every 0.25 --snaps $(seq 4 4 220) --out ellB_a1.1_b0.18_12x7.5.npz > ellB_b0.18.log 2>&1
cd v2b && python -u ../ellipse_ring.py --a 1.1 --b 0.16 --Ax 12 --Ay 7.5 --Lxy 40 --Lz 20 --T 220 --every 0.25 --snaps $(seq 4 4 220) --out ellB_a1.1_b0.16_12x7.5.npz > ellB_b0.16.log 2>&1
cd v2b/rerun_r2 && python -u ../../ellipse_ring.py --a 1.1 --b 0.15 --eps 0.02 --Ax 12 --Ay 7.5 --Lxy 40 --Lz 20 --T 92 --every 0.25 --snaps $(seq 4 4 92) --out ellP_a1.1_b0.15_12x7.5.npz > ellP_a1.1_b0.15_12x7.5.log 2>&1
cd v2b/rerun_r2 && python -u ../../ellipse_ring.py --a 0.8 --b 0.09 --eps 0.02 --Ax 8 --Ay 5 --Lxy 28 --Lz 16 --T 120 --every 0.25 --snaps $(seq 4 4 120) --out ell_b0.09_8x5.npz > ell_b0.09_8x5.log 2>&1
cd v2b/rerun_r2 && python -u ../../ellipse_ring.py --a 1.1 --b 0.17 --eps 0.02 --Ax 8 --Ay 5 --Lxy 28 --Lz 16 --T 120 --every 0.25 --snaps $(seq 4 4 120) --out ell_a1.1_b0.17_8x5.npz > ell_a1.1_b0.17_8x5.log 2>&1
cd v2b/rerun_r2 && python -u ../../ellipse_ring.py --a 0.8 --b 0.05 --eps 0.02 --Ax 12 --Ay 7.5 --Lxy 40 --Lz 20 --T 46 --every 0.25 --snaps $(seq 4 4 46) --out ellP_a0.8_b0.05_12x7.5.npz > ellP_a0.8_b0.05_12x7.5.log 2>&1
# Elliptic rings, original names (Fig. 2: 8x5; Fig. 3: ellP_*; hx method of period_analysis_r2.py, which loads all seven)
cd v2b && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --Ax 12 --Ay 7.5 --Lxy 40 --Lz 20 --T 240 --every 0.25 --snaps $(seq 4 4 240) --out ell_a1.1_b0.17_12x7.5.npz > ell_a1.1_b0.17_big.log 2>&1
cd v2b && python -u ../ellipse_ring.py --a 1.1 --b 0.15 --Ax 12 --Ay 7.5 --Lxy 40 --Lz 20 --T 220 --every 0.25 --snaps $(seq 4 4 220) --out ellP_a1.1_b0.15_12x7.5.npz > ellP_a1.1_b0.15.log 2>&1
cd v2b && python -u ../ellipse_ring.py --a 0.8 --b 0.09 --Ax 8 --Ay 5 --Lxy 28 --Lz 16 --T 120 --every 0.25 --snaps $(seq 4 4 120) --out ell_b0.09_8x5.npz > ell_b0.09.log 2>&1
cd v2b && python -u ../ellipse_ring.py --a 0.8 --b 0.05 --Ax 12 --Ay 7.5 --Lxy 40 --Lz 20 --T 80 --every 0.25 --snaps $(seq 4 2 80) --out ellP_a0.8_b0.05_12x7.5.npz > ellP_a0.8_b0.05.log 2>&1
cd v2b && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --Ax 8 --Ay 5 --Lxy 28 --Lz 16 --T 120 --every 0.25 --snaps $(seq 4 2 120) --out ell_a1.1_b0.17_8x5.npz > ell_a1.1_b0.17.log 2>&1
# Ring shape modes (Fig. 6a,b squares/diamonds and tilted ring: mode_decompose.py -> mode_decompose.npy, new_tests_analysis.py -> new_tests.npz; mode_freq.py; decay rates 0.011, 0.057)
cd v2b && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --m 2 --amp 0.08 --Ax 9.5 --Ay 9.5 --Lxy 34 --Lz 18 --T 150 --every 0.2 --out modeL_m2.npz > modeL_m2.log 2>&1
cd v2b && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --m 3 --amp 0.10 --Ax 9.5 --Ay 9.5 --Lxy 34 --Lz 18 --T 100 --every 0.2 --out modeL_m3.npz > modeL_m3.log 2>&1
cd v2b && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --m 2 --amp 0.08 --Ax 6.5 --Ay 6.5 --Lxy 26 --Lz 18 --T 130 --every 0.2 --out modeR_m2_a0.08.npz > modeR_m2_a0.08.log 2>&1
cd v2b && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --m 2 --amp 0.15 --Ax 6.5 --Ay 6.5 --Lxy 26 --Lz 18 --T 130 --every 0.2 --out modeR_m2_a0.15.npz > modeR_m2_a0.15.log 2>&1
cd v2b && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --m 3 --amp 0.08 --Ax 6.5 --Ay 6.5 --Lxy 26 --Lz 18 --T 130 --every 0.2 --out modeR_m3_a0.08.npz > modeR_m3_a0.08.log 2>&1
cd v2b && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --m 3 --amp 0.15 --Ax 6.5 --Ay 6.5 --Lxy 26 --Lz 18 --T 130 --every 0.2 --out modeR_m3_a0.15.npz > modeR_m3_a0.15.log 2>&1
cd v2b && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --Ax 6.7 --Ay 6.7 --tilt 12 --Lxy 28 --Lz 20 --T 60 --every 0.2 --out tilt_m1.npz > tilt_m1.log 2>&1
# Circular rings, T = 90-160 (Fig. 4a radius traces; Fig. 4b,c and Fig. 6 beta curve via VALID_CIRCLES)
cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 7.5 30 18 160 circS_R0_7.5.npz > circS_R0_7.5.log 2>&1
cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 6.5 28 18 160 circS_R0_6.5.npz > circS_R0_6.5.log 2>&1
cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 5.5 26 18 160 circS_R0_5.5.npz > circS_R0_5.5.log 2>&1
cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 4.5 26 18 160 circS_R0_4.5.npz > circS_R0_4.5.log 2>&1
cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 5.0 30 24 150 circHB_R0_5.0.npz > circHB_R0_5.0.log 2>&1
cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 11 36 16 90 circ_R0_11.npz > circ_R0_11.log 2>&1
cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 9.5 34 16 100 circ_a1.1_b0.17_R9.npz > circ_R9.log 2>&1
cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 7 26 16 120 circ_a1.1_b0.17.npz > confirm3d_a1.1_b0.17.log 2>&1
cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 4.0 30 24 150 circHB_R0_4.0.npz > circHB_R0_4.0.log 2>&1
cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 4 24 16 90 circ_R0_4.npz > circ_R0_4.log 2>&1
# Straight filaments, A = 1.0-1.5, side walls 32 (new_tests.npz straight block; original runs compared in sf2_analysis.py)
cd v2b && python -u ../straight_filament.py 126 1.0 80 sf_k025.npz 32 > sf_k025.log 2>&1
cd v2b && python -u ../straight_filament.py 90 1.0 70 sf_k035.npz 32 > sf_k035.log 2>&1
cd v2b && python -u ../straight_filament.py 70 1.5 60 sf_k045.npz 32 > sf_k045.log 2>&1
cd v2b && python -u ../straight_filament.py 52 1.5 40 sf_k060.npz 32 > sf_k060.log 2>&1
cd v2b && python -u ../straight_filament.py 40 1.5 30 sf_k079.npz 32 > sf_k079.log 2>&1
# Ring of Fig. 1 (8 x 5): movie (Multimedia; diffrot_check.py -> diffrot.npz, twist from tip phase), twist (twist_run.py), handedness check, Fig. 1(b-f) fields
cd v2b/movie && python -u ../../ellipse_ring.py --a 1.1 --b 0.17 --eps 0.02 --Ax 8 --Ay 5 --Lxy 28 --Lz 16 --T 40.1 --every 0.2 --snaps $(seq -f %.1f 3 0.2 40) --fields --uonly --out ell8x5_movie.npz > ell8x5_movie.log 2>&1
cd v2b && python -u ../twist_run.py 1.1 0.17 8 5 75 twist_8x5.npz > twist_8x5.log 2>&1
cd v2b/fig1_fields && python -u ../../ellipse_ring.py --a 1.1 --b 0.17 --eps 0.02 --Ax 8 --Ay 5 --Lxy 28 --Lz 16 --T 13.1 --every 0.1 --snaps 12.0 12.1 12.2 12.3 12.4 12.5 12.6 12.7 12.8 12.9 13.0 --fields --out handed.npz > handed.log 2>&1
cd v2b/fig1_fields && python -u ../../ellipse_ring.py --a 1.1 --b 0.17 --eps 0.02 --Ax 8 --Ay 5 --Lxy 28 --Lz 16 --T 40 --every 0.5 --snaps 4 6 8 10 12 14 16 18 20 22 24 28 32 36 --fields --out ell8x5_fields.npz > ell8x5_fields.log 2>&1
# 2D spirals: meandering at a = 0.8, b = 0.09 (Sec. II; writes tip2d_b0.09_N350.npy) and the rotation periods
# hard-coded as smoothing periods in period_analysis.RUNS / make_figs (4.13 for b = 0.15, 3.39 for a = 0.8 b = 0.05)
cd v2b && python -u ../meander_check.py 0.09 350 200 > meander_b0.09.log 2>&1
cd v2b && python -u ../param_probe.py 0.8 0.09 0.02 > probe_0.8_0.09_0.02.log 2>&1
cd v2b && python -u ../param_probe.py 1.1 0.15 0.02 > probe_1.1_0.15_0.02.log 2>&1
cd v2b && python -u ../param_probe.py 0.8 0.05 0.02 > probe_0.8_0.05_0.02.log 2>&1
# rotation periods used as smoothing windows for b = 0.16 and 0.18 (previously not measured)
cd v2b && python -u ../param_probe.py 1.1 0.16 0.02 > probe_1.1_0.16_0.02.log 2>&1
cd v2b && python -u ../param_probe.py 1.1 0.18 0.02 > probe_1.1_0.18_0.02.log 2>&1
#
# ---- 2D spiral characterization at a = 1.1, b = 0.17 (Sec. II: period 4.44, core radius 0.49, wavelength
# 12.0; dx = 0.15 check). The original commands were inline python -c programs with quotes and loops, so
# they cannot go through the xargs runner above; run them by hand from v2b/ (full text in the notes):
#   confirm2d_a1.1_b0.17.log  : param_probe.spiral2d(1.1, 0.17, 0.02, 300, N=300)
#   ctrl_2d_dx015.log         : param_probe.spiral2d(1.1, 0.17, 0.02, 300, N=267, dx=0.15)
#   (wavelength, no log)      : 300 x 300 x 3 grid, t = 70, ray sampling from the tip
#
# ---- Runs that exist but feed no figure or manuscript number (left out; enable if wanted):
# circ_R0_{6,9,13}: identical to the first 90 time units of circL_R0_{6,9,13}; only coeff_curves.py (not in the paper)
# cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 13 40 16 90 circ_R0_13.npz > circ_R0_13.log 2>&1
# cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 9 32 16 90 circ_R0_9.npz > circ_R0_9.log 2>&1
# cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 6 26 16 90 circ_R0_6.npz > circ_R0_6.log 2>&1
# mode_m{2,3}_{s,L}: mode_damping.py only (superseded by modeR_* / mode_decompose.py)
# cd v2b && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --m 2 --amp 0.15 --Ax 9.5 --Ay 9.5 --Lxy 34 --Lz 16 --T 200 --every 0.25 --out mode_m2_L.npz > mode_m2_L.log 2>&1
# cd v2b && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --m 3 --amp 0.15 --Ax 9.5 --Ay 9.5 --Lxy 34 --Lz 16 --T 200 --every 0.25 --out mode_m3_L.npz > mode_m3_L.log 2>&1
# cd v2b && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --m 2 --amp 0.15 --Ax 6.7 --Ay 6.7 --Lxy 26 --Lz 16 --T 150 --every 0.25 --out mode_m2_s.npz > mode_m2_s.log 2>&1
# cd v2b && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --m 3 --amp 0.15 --Ax 6.7 --Ay 6.7 --Lxy 26 --Lz 16 --T 150 --every 0.25 --out mode_m3_s.npz > mode_m3_s.log 2>&1
# circH_R0_*: Lz = 14 box, excluded from Fig. 4 (spurious ring growth)
# cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 4.0 20 14 150 circH_R0_4.0.npz > circH_R0_4.0.log 2>&1
# cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 4.5 21 14 150 circH_R0_4.5.npz > circH_R0_4.5.log 2>&1
# cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 5.0 22 14 150 circH_R0_5.0.npz > circH_R0_5.0.log 2>&1
# cd v2b && python -u ../circle_run.py 1.1 0.17 0.02 5.5 23 14 150 circH_R0_5.5.npz > circH_R0_5.5.log 2>&1
# ecc/: eccentric rings (not cited; the eccentricity remark in Sec. V is speculative)
# cd v2b/ecc && python -u ../../ellipse_ring.py --a 1.1 --b 0.17 --eps 0.02 --Ax 12 --Ay 4 --Lxy 44 --Lz 20 --T 80 --every 0.25 --snaps $(seq 2 2 80) --out ell_12x4.npz > ell_12x4.log 2>&1
# cd v2b/ecc && python -u ../../ellipse_ring.py --a 1.1 --b 0.17 --eps 0.02 --Ax 10 --Ay 4 --Lxy 40 --Lz 20 --T 80 --every 0.25 --snaps $(seq 2 2 80) --out ell_10x4.npz > ell_10x4.log 2>&1
# strobe/: stroboscopic movie (not in the paper); pass2 needs strobe/triggers.txt from strobe_triggers.py on pass1
# cd v2b/strobe && python -u ../../ellipse_ring.py --a 1.1 --b 0.17 --eps 0.02 --Ax 8 --Ay 5 --Lxy 28 --Lz 16 --T 255 --every 0.25 --probe --out pass1.npz > pass1.log 2>&1
