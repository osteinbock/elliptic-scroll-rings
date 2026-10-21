# Runs for the revision (referee reports of 20 October 2026): Barkley model (eps = 0.02, dx = 0.2) and,
# at the end, two Panfilov-Rudenko circular rings and the two-dimensional spirals.
# Output goes to v6/. One command per line, longest first. Run from the code directory:
#   mkdir -p v6/sf2 v6/spiral2d
#   grep -v '^#' runs_revision.sh | grep . | NUMBA_NUM_THREADS=4 xargs -P 5 -I{} sh -c '{}'
# Lines must stay free of quote characters (xargs strips them).
#
# Extra isosurface pairs for the mode runs (b = 0.17): (u*, a u* - b) at u* = 0.3, 0.7, and u* = 0.5 with
# v* = a/2 - b -/+ 0.1; the standard pair is u = 0.5, v = a/2 - b = 0.38.
#
# Circular ring at b = 0.17 with R0 = 26 (fills the gap in K between the R0 = 20 and R0 = 38 rings)
cd v6 && python -u ../circle_run.py 1.1 0.17 0.02 26 66 16 150 circ17_R0_26.npz > circ17_R0_26.log 2>&1
# Ring shape modes with four extra filament definitions and the rotation phase (same seeds as runs_barkley_v2.sh)
cd v6 && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --m 2 --amp 0.08 --Ax 9.5 --Ay 9.5 --Lxy 34 --Lz 18 --T 150 --every 0.2 --levels 0.3,0.16 0.7,0.60 0.5,0.28 0.5,0.48 --twist --out modeL_m2_lv.npz > modeL_m2_lv.log 2>&1
cd v6 && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --m 3 --amp 0.10 --Ax 9.5 --Ay 9.5 --Lxy 34 --Lz 18 --T 100 --every 0.2 --levels 0.3,0.16 0.7,0.60 0.5,0.28 0.5,0.48 --twist --out modeL_m3_lv.npz > modeL_m3_lv.log 2>&1
# Circular-ring calibrations at b = 0.18 (rings expand; boxes leave room for growth)
cd v6 && python -u ../circle_run.py 1.1 0.18 0.02 12 46 16 100 circ18_R0_12.npz > circ18_R0_12.log 2>&1
cd v6 && python -u ../circle_run.py 1.1 0.18 0.02 10 42 16 100 circ18_R0_10.npz > circ18_R0_10.log 2>&1
cd v6 && python -u ../circle_run.py 1.1 0.18 0.02 8 38 16 100 circ18_R0_8.npz > circ18_R0_8.log 2>&1
# Circular-ring calibrations at b = 0.16 and 0.15 (rings shrink)
cd v6 && python -u ../circle_run.py 1.1 0.16 0.02 10 34 16 200 circ16_R0_10.npz > circ16_R0_10.log 2>&1
cd v6 && python -u ../circle_run.py 1.1 0.16 0.02 8 30 16 200 circ16_R0_8.npz > circ16_R0_8.log 2>&1
cd v6 && python -u ../circle_run.py 1.1 0.16 0.02 6.5 28 16 200 circ16_R0_6.5.npz > circ16_R0_6.5.log 2>&1
cd v6 && python -u ../circle_run.py 1.1 0.16 0.02 5 24 16 200 circ16_R0_5.npz > circ16_R0_5.log 2>&1
cd v6 && python -u ../circle_run.py 1.1 0.16 0.02 4 24 16 200 circ16_R0_4.npz > circ16_R0_4.log 2>&1
cd v6 && python -u ../circle_run.py 1.1 0.15 0.02 10 34 16 120 circ15_R0_10.npz > circ15_R0_10.log 2>&1
cd v6 && python -u ../circle_run.py 1.1 0.15 0.02 8 30 16 120 circ15_R0_8.npz > circ15_R0_8.log 2>&1
cd v6 && python -u ../circle_run.py 1.1 0.15 0.02 6.5 28 16 120 circ15_R0_6.5.npz > circ15_R0_6.5.log 2>&1
cd v6 && python -u ../circle_run.py 1.1 0.15 0.02 5 24 16 120 circ15_R0_5.npz > circ15_R0_5.log 2>&1
cd v6 && python -u ../circle_run.py 1.1 0.15 0.02 4 24 16 120 circ15_R0_4.npz > circ15_R0_4.log 2>&1
# Ring shape modes at R* (two seeds each), with the extra filament definitions and the rotation phase
cd v6 && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --m 2 --amp 0.08 --Ax 6.5 --Ay 6.5 --Lxy 26 --Lz 18 --T 130 --every 0.2 --levels 0.3,0.16 0.7,0.60 0.5,0.28 0.5,0.48 --twist --out modeR_m2_a0.08_lv.npz > modeR_m2_a0.08_lv.log 2>&1
cd v6 && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --m 2 --amp 0.15 --Ax 6.5 --Ay 6.5 --Lxy 26 --Lz 18 --T 130 --every 0.2 --levels 0.3,0.16 0.7,0.60 0.5,0.28 0.5,0.48 --twist --out modeR_m2_a0.15_lv.npz > modeR_m2_a0.15_lv.log 2>&1
cd v6 && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --m 3 --amp 0.08 --Ax 6.5 --Ay 6.5 --Lxy 26 --Lz 18 --T 130 --every 0.2 --levels 0.3,0.16 0.7,0.60 0.5,0.28 0.5,0.48 --twist --out modeR_m3_a0.08_lv.npz > modeR_m3_a0.08_lv.log 2>&1
cd v6 && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --m 3 --amp 0.15 --Ax 6.5 --Ay 6.5 --Lxy 26 --Lz 18 --T 130 --every 0.2 --levels 0.3,0.16 0.7,0.60 0.5,0.28 0.5,0.48 --twist --out modeR_m3_a0.15_lv.npz > modeR_m3_a0.15_lv.log 2>&1
# m = 4 at R* (k about 0.82)
cd v6 && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --m 4 --amp 0.10 --Ax 6.5 --Ay 6.5 --Lxy 26 --Lz 18 --T 70 --every 0.1 --levels 0.3,0.16 0.7,0.60 0.5,0.28 0.5,0.48 --twist --out modeR_m4_a0.10_lv.npz > modeR_m4_a0.10_lv.log 2>&1
# Controls: larger domain for a mode run; halved time step and a seed turned by 45 degrees for the 8 x 5 ellipse
cd v6 && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --m 2 --amp 0.08 --Ax 6.5 --Ay 6.5 --Lxy 36 --Lz 24 --T 130 --every 0.2 --out modeR_m2_a0.08_bigbox.npz > modeR_m2_a0.08_bigbox.log 2>&1
cd v6 && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --Ax 8 --Ay 5 --Lxy 28 --Lz 16 --T 80 --every 0.25 --dtfac 0.4 --out ell_8x5_dt04.npz > ell_8x5_dt04.log 2>&1
cd v6 && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --Ax 8 --Ay 5 --Lxy 28 --Lz 16 --T 80 --every 0.25 --rot 45 --out ell_8x5_rot45.npz > ell_8x5_rot45.log 2>&1
# Straight filaments at b = 0.15 and 0.16: is the translational branch stable outside the wrinkled-scroll band of b = 0.17?
cd v6/sf2 && python -u ../../straight_filament.py 90 0.5 120 b015_k035.npz 32 1.1 0.15 0.02 0.2 > b015_k035.log 2>&1
cd v6/sf2 && python -u ../../straight_filament.py 70 0.5 100 b015_k045.npz 32 1.1 0.15 0.02 0.2 > b015_k045.log 2>&1
cd v6/sf2 && python -u ../../straight_filament.py 52 0.5 80 b015_k060.npz 32 1.1 0.15 0.02 0.2 > b015_k060.log 2>&1
cd v6/sf2 && python -u ../../straight_filament.py 70 0.5 100 b016_k045.npz 32 1.1 0.16 0.02 0.2 > b016_k045.log 2>&1
# Panfilov-Rudenko circular rings seeded with radius 18 (third ring of the calibrations at g_f = 0.775 and 0.8,
# between the seeds 14 and 22 of runs_v2.sh; same parameters, dx = 0.6)
cd v6 && python -u ../pan_ring.py 0.775 0.330 18 18 76 50 800 pcirc_gf0.775_R18.npz > pcirc_gf0.775_R18.log 2>&1
cd v6 && python -u ../pan_ring.py 0.8 0.372 18 18 76 50 600 pcirc_gf0.8_R18.npz > pcirc_gf0.8_R18.log 2>&1
# Two-dimensional spirals: rotation period, core radius and wavelength (SI Table S4); the Panfilov-Rudenko
# spirals are computed by pan2d.py and analyzed from its saved fields
cd v6/spiral2d && python -u ../../wavelength2d.py barkley 1.1 0.15 0.02 90 450 0.2 barkley_b0.15.json > barkley_b0.15.log 2>&1
cd v6/spiral2d && python -u ../../wavelength2d.py barkley 1.1 0.16 0.02 90 450 0.2 barkley_b0.16.json > barkley_b0.16.log 2>&1
cd v6/spiral2d && python -u ../../wavelength2d.py barkley 1.1 0.17 0.02 90 450 0.2 barkley_b0.17.json > barkley_b0.17.log 2>&1
cd v6/spiral2d && python -u ../../wavelength2d.py barkley 1.1 0.18 0.02 90 450 0.2 barkley_b0.18.json > barkley_b0.18.log 2>&1
cd v6/spiral2d && python -u ../../pan2d.py 0.775 0.06 0 210 400 0.6 0.04 > pan2d_gf0.775.log 2>&1 && python -u ../../wavelength2d.py panfilov pan2d_fields_gf0.775_e0.06.npz pan2d_gf0.775_e0.06.npy panfilov_gf0.775.json >> pan2d_gf0.775.log 2>&1
cd v6/spiral2d && python -u ../../pan2d.py 0.8 0.06 0 210 400 0.6 0.04 > pan2d_gf0.8.log 2>&1 && python -u ../../wavelength2d.py panfilov pan2d_fields_gf0.8_e0.06.npz pan2d_gf0.8_e0.06.npy panfilov_gf0.8.json >> pan2d_gf0.8.log 2>&1
