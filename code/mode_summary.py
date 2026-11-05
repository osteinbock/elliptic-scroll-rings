"""Couplings and diagonal rates of the ring shape modes with their ranges, relative to the drift
coefficient beta_c of circular rings of the same radius (Fig. 6a, Tables S5-S7 of the supplementary
material). Ranges cover seed amplitude, fit window, smoothing, dx = 0.15 and filament definition at R*, the
fit windows and filament definitions for m = 4, and fits in tau over the whole run and its halves,
constant-frequency windows and filament definitions on the larger ring. Ratios that were stored
relative to the ring's own drift coefficient are converted with the drift of the same fit or run.

    import mode_summary; rows = mode_summary.modes()      # BARKLEY, REVISION as in folders.py
"""
import os
import json
import numpy as np

import folders as FD
import halfcycle_uniform as HU
from mode_refit import local_law


def _span(values):
    v = [float(x) for x in values if np.isfinite(x)]
    return [min(v), max(v)]


def modes():
    cal = HU.barkley_calibration(0.17)
    MR = json.load(open(os.path.join(FD.BARKLEY, "mode_refit.json")))
    MD = json.load(open(os.path.join(FD.REVISION, "mode_definitions.json")))
    nt = np.load(os.path.join(FD.BARKLEY, "new_tests.npz"))["large"]
    out = []
    for m in (2, 3):
        main = [r for r in MR["rstar"] if r["m"] == m]
        rng = MR["rstar_ranges"][str(m)]
        to_c = np.mean([r["bdrift"] for r in main]) / np.mean([r["beta_c"] for r in main])   # drift-based -> beta_c-based
        defs = [r for r in MD["rstar"] if r["m"] == m]
        R = np.mean([r["R"] for r in main])
        law = local_law(cal, R, m)
        bz = [r["bz"] / r["beta_c"] for r in main]
        br = [r["br"] / r["beta_c"] for r in main]
        out.append(dict(m=m, ring="R*", R=float(R), k=float(np.mean([r["k"] for r in main])),
                        bz=float(np.mean(bz)), br=float(np.mean(br)),
                        bz_rng=_span(bz + [x * to_c for x in rng["bz_range"]] + [r["bz"] / float(cal.beta(r["R"])) for r in defs]),
                        br_rng=_span(br + [x * to_c for x in rng["br_range"]] + [r["br"] / float(cal.beta(r["R"])) for r in defs]),
                        az=float(np.mean([r["alpha_zeta"] for r in main])), ar=float(np.mean([r["alpha_r"] for r in main])),
                        az_rng=_span([r["alpha_zeta"] for r in main + defs]), ar_rng=_span([r["alpha_r"] for r in main + defs]),
                        br_loc=law["beta_r_loc"] / law["beta_c"], ar_loc=law["alpha_r_loc"], ac=law["alpha_c"],
                        phi=[float((r["dphase_deg"] % 360) - 180) for r in main],
                        gamma=float(np.mean([r["gamma"] for r in main])),
                        W=float(np.mean([np.sqrt(r["bz"] * r["br"]) for r in main]))))
    defs4 = [r for r in MD["rstar"] if r["m"] == 4]
    std4 = [r for r in defs4 if "standard" in r["definition"]][0]
    law4 = local_law(cal, std4["R"], 4)
    to_c4 = std4["bdrift"] / law4["beta_c"]
    win4 = MD["m4_windows"]
    out.append(dict(m=4, ring="R*", R=float(std4["R"]), k=4 / std4["R"],
                    bz=std4["bz"] / law4["beta_c"], br=std4["br"] / law4["beta_c"],
                    bz_rng=_span([r["bz"] / float(cal.beta(r["R"])) for r in defs4] + [w["bz_over_drift"] * to_c4 for w in win4]),
                    br_rng=_span([r["br"] / float(cal.beta(r["R"])) for r in defs4] + [w["br_over_drift"] * to_c4 for w in win4]),
                    az=std4["alpha_zeta"], ar=std4["alpha_r"],
                    az_rng=_span([r["alpha_zeta"] for r in defs4] + [w["alpha_zeta"] for w in win4]),
                    ar_rng=_span([r["alpha_r"] for r in defs4] + [w["alpha_r"] for w in win4]),
                    br_loc=law4["beta_r_loc"] / law4["beta_c"], ar_loc=law4["alpha_r_loc"], ac=law4["alpha_c"],
                    phi=[float((std4["dphase_deg"] % 360) - 180)], gamma=std4["gamma"],
                    W=float(np.sqrt(std4["bz"] * std4["br"]))))
    for m in (2, 3):
        tf = [r for r in MR["tau_fits"] if r["run"] == "larger ring" and r["m"] == m]
        full = [r for r in tf if r["t0"] == 15.0 and r["t1"] in (150.0, 100.0)][0]
        defs = [r for r in MD["large"] if r["m"] == m]
        L = nt[(nt[:, 0] == m) & np.isfinite(nt[:, 8])]
        cw = [float(cal.beta(m / k)) for k in L[:, 3]]
        bz = [r["bz"] / r["beta_c"] for r in tf] + [r["bz"] / float(cal.beta(r["R"])) for r in defs] + list(L[:, 6] / np.array(cw))
        br = [r["br"] / r["beta_c"] for r in tf] + [r["br"] / float(cal.beta(r["R"])) for r in defs] + list(L[:, 7] / np.array(cw))
        law = local_law(cal, full["R"], m)
        out.append(dict(m=m, ring="larger", R=full["R"], R_start=full["R_start"], R_end=full["R_end"], k=full["k"],
                        bz=full["bz"] / full["beta_c"], br=full["br"] / full["beta_c"], bz_rng=_span(bz), br_rng=_span(br),
                        az=full["alpha_zeta"], ar=full["alpha_r"],
                        az_rng=_span([r["alpha_zeta"] for r in tf + defs]), ar_rng=_span([r["alpha_r"] for r in tf + defs]),
                        br_loc=law["beta_r_loc"] / law["beta_c"], ar_loc=law["alpha_r_loc"], ac=law["alpha_c"],
                        phi=[float((full["dphase_deg"] % 360) - 180)], gamma=full["gamma_tau"] / full["R"] ** 2,
                        W=full["W"]))
    return out


def large_modes():
    """The rings of mode_wavenumber.py at b = 0.17 (R about 11 to 15), in the form of modes(). Ranges cover
    the fits in tau over the whole interval and its halves."""
    MW = json.load(open(os.path.join(FD.REVISION, "mode_wavenumber.json")))["large"]
    out = []
    for m in (2, 3, 4):
        tf = [r for r in MW if r["m"] == m]
        full = [r for r in tf if r["whole"]][0]
        out.append(dict(m=m, ring="large", R=full["R"], R_start=full["R_start"], R_end=full["R_end"], k=full["k"],
                        bz=full["bz"] / full["beta_c"], br=full["br"] / full["beta_c"],
                        bz_rng=_span([r["bz"] / r["beta_c"] for r in tf]), br_rng=_span([r["br"] / r["beta_c"] for r in tf]),
                        az=full["alpha_zeta"], ar=full["alpha_r"],
                        az_rng=_span([r["alpha_zeta"] for r in tf]), ar_rng=_span([r["alpha_r"] for r in tf]),
                        br_loc=full["beta_r_loc"] / full["beta_c"], ar_loc=full["alpha_r_loc"], ac=full["alpha_c"],
                        phi=[float((full["dphase_deg"] % 360) - 180)], gamma=full["gamma_tau"] / full["R"] ** 2,
                        W=full["W"], beta_c=full["beta_c"]))
    return out
