# Grid refinement (dx = 0.15) of the mode couplings and of the straight-filament dispersion (Sec. III D);
# compared with the dx = 0.2 runs by refine_compare.py. Output goes to v5r/.
cd v5r && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --m 2 --amp 0.08 --Ax 6.5 --Ay 6.5 --Lxy 26 --Lz 18 --dx 0.15 --T 130 --every 0.2 --out modeR_m2_a0.08.npz > modeR_m2_a0.08.log 2>&1
cd v5r && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --m 2 --amp 0.15 --Ax 6.5 --Ay 6.5 --Lxy 26 --Lz 18 --dx 0.15 --T 130 --every 0.2 --out modeR_m2_a0.15.npz > modeR_m2_a0.15.log 2>&1
cd v5r && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --m 3 --amp 0.08 --Ax 6.5 --Ay 6.5 --Lxy 26 --Lz 18 --dx 0.15 --T 130 --every 0.2 --out modeR_m3_a0.08.npz > modeR_m3_a0.08.log 2>&1
cd v5r && python -u ../ellipse_ring.py --a 1.1 --b 0.17 --m 3 --amp 0.15 --Ax 6.5 --Ay 6.5 --Lxy 26 --Lz 18 --dx 0.15 --T 130 --every 0.2 --out modeR_m3_a0.15.npz > modeR_m3_a0.15.log 2>&1
cd v5r/sf2 && python -u ../../straight_filament.py 120 0.5 120 L32_k035_dx015.npz 32 1.1 0.17 0.02 0.15 > L32_k035_dx015.log 2>&1
