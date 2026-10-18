"""Straight-filament box and amplitude control (sf2/): Lxy = 32 vs 64, A = 0.5.

For each run: period-smoothed helical components |u_{+k}|, |u_{-k}|, local decay rates in
10-unit windows, and the rotation rate. Compares with the original A = 1.0-1.5 runs (sf_k*.npz,
Lxy = 32). k = 0 controls give the drift of an unperturbed filament (wall interaction).
Writes sf2/sf2_timecourse.png and prints the tables.
"""
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ellipse_compare as C

PERIOD = 4.44


def csmooth(t, z):
    return C.smooth(t, z.real, PERIOD) + 1j * C.smooth(t, z.imag, PERIOD)


def local_rates(t, s, width=10.0, step=5.0, floor=0.06):
    out = []
    for t0 in np.arange(5.0, t[-1] - width + 0.1, step):
        m = (t >= t0) & (t < t0 + width) & np.isfinite(s) & (np.abs(s) > floor)
        if m.sum() < 20:
            continue
        g = -np.polyfit(t[m], np.log(np.abs(s[m])), 1)[0]
        w = np.polyfit(t[m], np.unwrap(np.angle(s[m])), 1)[0]
        out.append((t0 + width / 2, g, w, np.abs(s[m]).mean()))
    return np.array(out)


def main():
    files = {"L32 A=0.5": sorted(glob.glob("sf2/L32_k0[1-9]*.npz")),
             "L64 A=0.5": sorted(glob.glob("sf2/L64_k0[1-9]*.npz")),
             "L32 A=1-1.5 (orig)": sorted(glob.glob("sf_k0*.npz"))}
    fig, ax = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for a_, (lab, fl) in zip(ax, files.items()):
        print(f"\n=== {lab}")
        for f in fl:
            d = np.load(f)
            t, k = d["t"], float(d["k"])
            sp = csmooth(t, d["uk"])
            line = a_.semilogy(t, np.abs(sp), label=f"k={k:.2f}")[0]
            if "ukm" in d:
                sm = csmooth(t, d["ukm"])
                a_.semilogy(t, np.abs(sm), "--", color=line.get_color(), lw=0.8)
            r = local_rates(t, sp)
            if len(r) == 0:
                print(f"k={k:.3f}: below floor")
                continue
            txt = "  ".join(f"t{tc:.0f}:{g / k**2:+.2f}|{amp:.2f}" for tc, g, w, amp in r[::2])
            print(f"k={k:.3f}  gamma/k^2 | amp by window: {txt}")
            print(f"         |beta| by window: " + " ".join(f"{abs(w) / k**2:.2f}" for _, _, w, _ in r[::2]))
        a_.set(title=lab, xlabel="t")
        a_.legend(fontsize=7)
    ax[0].set_ylabel("|u_k| (solid +k, dashed -k)")
    fig.tight_layout()
    fig.savefig("sf2/sf2_timecourse.png", dpi=110)
    print("\n=== k = 0 controls: mean filament position minus box centre")
    for f in sorted(glob.glob("sf2/L*_k0.npz")):
        d = np.load(f)
        t, u = d["t"], d["u0"] - (1 + 1j) * float(d["Lxy"]) / 2
        print(f, "  ".join(f"t{tt}:{np.round(u[np.abs(t - tt) < 2.3].mean(), 2)}"
                           for tt in (5, 10, 20, 40, 80, 120, 148) if (np.abs(t - tt) < 2.3).any()))


if __name__ == "__main__":
    main()
