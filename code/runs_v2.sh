# Production runs of the supplementary models, rerun after the audit fixes
# (z bookkeeping, stitched filaments, saved metadata). Same parameters and seeds as the
# original runs; output goes to v2/. One command per line, longest first.
# Run from the code directory:
#   mkdir -p v2 && grep -v '^#' runs_v2.sh | grep . | NUMBA_NUM_THREADS=4 xargs -P 5 -I{} sh -c '{}'
#
# Atri Ca2+ model, P = 0.1 uM, c* = 0.52 (Table S3, S4, Fig. S2), and the dx = 2 um check
cd v2 && ATRI_DX=2.0 ATRI_CSTAR=0.52 python -u ../atri_ring.py 0.1 0.7535 100 60 600 280 1800 aell_dx2_P0.1_100x60.npz $(seq 50 50 1800) > aell_dx2.log 2>&1
cd v2 && ATRI_DX=2.0 ATRI_CSTAR=0.52 python -u ../atri_ring.py 0.1 0.7535 100 100 520 280 900 acirc_dx2_P0.1.npz > acirc_dx2.log 2>&1
cd v2 && ATRI_DX=3.0 ATRI_CSTAR=0.52 python -u ../atri_ring.py 0.1 0.7535 100 60 600 280 1800 aell2_P0.1_100x60.npz $(seq 50 50 1800) > aell2_P0.1_small.log 2>&1
cd v2 && ATRI_DX=3.0 ATRI_CSTAR=0.52 python -u ../atri_ring.py 0.1 0.7535 130 80 660 280 1800 aell2_P0.1_130x80.npz $(seq 50 50 1800) > aell2_P0.1.log 2>&1
cd v2 && ATRI_DX=3.0 ATRI_CSTAR=0.52 python -u ../atri_ring.py 0.1 0.7535 100 100 520 280 900 acirc4_P0.1_c052.npz > acirc4.log 2>&1
# Panfilov-Rudenko kinetics B: elliptic rings (Table S2, Fig. S1)
cd v2 && PAN_DX=0.3 python -u ../pan_ring.py 0.8 0.372 24 15 76 40 1000 pellXR_gf0.8_24x15.npz > pellXR.log 2>&1
cd v2 && PAN_DX=0.4 python -u ../pan_ring.py 0.8 0.372 24 15 88 50 1100 pellHR_gf0.8_24x15.npz $(seq 20 20 1100) > pellHR.log 2>&1
cd v2 && python -u ../pan_ring.py 0.8 0.372 24 15 120 80 1100 pellBIG_gf0.8_24x15.npz > pellBIG.log 2>&1
cd v2 && python -u ../pan_ring.py 0.775 0.330 20 12 80 50 1600 pell_gf0.775_20x12.npz $(seq 20 20 1600) > pell_gf0.775.log 2>&1
cd v2 && python -u ../pan_ring.py 0.8 0.372 24 15 88 50 1600 pell_gf0.8_24x15.npz $(seq 20 20 1600) > pell_gf0.8.log 2>&1
# Panfilov m = 2 and m = 3 shape modes at g_f = 0.775
cd v2 && PAN_MODE=2 PAN_AMP=0.15 python -u ../pan_ring.py 0.775 0.330 17 17 80 50 1600 pmode2_gf0.775.npz > pmode2.log 2>&1
cd v2 && PAN_MODE=3 PAN_AMP=0.15 python -u ../pan_ring.py 0.775 0.330 17 17 80 50 1600 pmode3_gf0.775.npz > pmode3.log 2>&1
# Panfilov circular rings (Table S1; R = 14 rings calibrate Table S2)
cd v2 && PAN_DX=0.3 python -u ../pan_ring.py 0.8 0.372 14 14 68 40 500 pcircXR_gf0.8_R14.npz > pcircXR.log 2>&1
cd v2 && PAN_DX=0.4 python -u ../pan_ring.py 0.8 0.372 14 14 68 50 600 pcircHR_gf0.8_R14.npz > pcircHR.log 2>&1
cd v2 && python -u ../pan_ring.py 0.775 0.330 14 14 68 50 800 pcirc_gf0.775_R14.npz > pcirc_gf0.775_R14.log 2>&1
cd v2 && python -u ../pan_ring.py 0.775 0.330 22 22 84 50 800 pcirc_gf0.775_R22.npz > pcirc_gf0.775_R22.log 2>&1
cd v2 && python -u ../pan_ring.py 0.8 0.372 14 14 68 50 600 pcirc_gf0.8_R14.npz > pcirc_gf0.8_R14.log 2>&1
cd v2 && python -u ../pan_ring.py 0.8 0.372 22 22 84 50 600 pcirc_gf0.8_R22.npz > pcirc_gf0.8_R22.log 2>&1
cd v2 && python -u ../pan_ring.py 0.825 0.383 14 14 68 50 600 pcirc_gf0.825_R14.npz > pcirc_gf0.825_R14.log 2>&1
cd v2 && python -u ../pan_ring.py 0.825 0.383 22 22 84 50 600 pcirc_gf0.825_R22.npz > pcirc_gf0.825_R22.log 2>&1
cd v2 && python -u ../pan_ring.py 0.85 0.390 14 14 68 50 600 pcirc_gf0.85_R14.npz > pcirc_gf0.85_R14.log 2>&1
cd v2 && python -u ../pan_ring.py 0.85 0.390 22 22 84 50 600 pcirc_gf0.85_R22.npz > pcirc_gf0.85_R22.log 2>&1
cd v2 && python -u ../pan_ring.py 0.9 0.390 14 14 68 50 600 pcirc_gf0.9_R14.npz > pcirc_gf0.9_R14.log 2>&1
cd v2 && python -u ../pan_ring.py 0.9 0.390 22 22 84 50 600 pcirc_gf0.9_R22.npz > pcirc_gf0.9_R22.log 2>&1
# Atri weakly elliptic rings (aspect ~1.14): linear regime, radius conventions agree to < 1%
cd v2 && ATRI_DX=3.0 ATRI_CSTAR=0.52 python -u ../atri_ring.py 0.1 0.7535 100 88 600 280 2400 aellW_P0.1_100x88.npz $(seq 100 100 2400) > aellW_100x88.log 2>&1
cd v2 && ATRI_DX=3.0 ATRI_CSTAR=0.52 python -u ../atri_ring.py 0.1 0.7535 120 105 660 280 2400 aellW_P0.1_120x105.npz $(seq 100 100 2400) > aellW_120x105.log 2>&1
# Atri, Table S3 rows P = 0.085 and 0.095 (c* = 0.566, the default) and the P = 0.095 ellipse (Table S4 caption)
cd v2 && ATRI_DX=3.0 ATRI_CSTAR=0.578 python -u ../atri_ring.py 0.095 0.7466 130 80 660 280 1800 aell2_P0.095_130x80.npz $(seq 50 50 1800) > aell2_P0.095.log 2>&1
cd v2 && ATRI_DX=3.0 ATRI_CSTAR=0.566 python -u ../atri_ring.py 0.095 0.747 130 130 520 240 1200 acirc2_R130.npz > acirc2_R130.log 2>&1
cd v2 && ATRI_DX=3.0 ATRI_CSTAR=0.566 python -u ../atri_ring.py 0.095 0.747 80 80 420 240 1200 acirc2_R80.npz > acirc2_R80.log 2>&1
cd v2 && ATRI_DX=3.0 ATRI_CSTAR=0.566 python -u ../atri_ring.py 0.085 0.759 100 100 520 280 900 acirc3_P0.085.npz > acirc3_P0.085.log 2>&1
cd v2 && ATRI_DX=3.0 ATRI_CSTAR=0.566 python -u ../atri_ring.py 0.1 0.754 100 100 520 280 900 acirc3_P0.1.npz > acirc3_P0.1.log 2>&1
