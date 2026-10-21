"""Regression tests for the filament extraction and the analysis pipeline.

Runs without data files; each test builds its own synthetic input.
    python tests/test_regression.py          (or: pytest tests/)
"""
import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import barkley3d as B            # noqa: E402
import ellipse_compare as C      # noqa: E402
import ellipse_ring as E         # noqa: E402
import filament_flow as F        # noqa: E402


def grid(n):
    return np.meshgrid(*(np.arange(float(n)),) * 3, indexing="ij")


# ------------------------------------------------------------------ extraction

def test_false_root_face_rejected():
    """A face whose bilinear equations have no real common zero yields no point (re-audit 2a)."""
    f = np.array([[-0.06677194407135212, -0.9976269811688985], [-0.5564733857658062, 0.05499573549732584]])
    g = np.array([[-0.6721105113530887, 0.227611572997084], [-0.05956824294706542, 0.4901609019996176]])
    Fa, Ga = (np.repeat(a[:, :, None], 2, axis=2) for a in (f, g))
    low, pos, _ = B.face_points(Fa, Ga, 2)
    assert len(low) == 0


def test_accepted_roots_satisfy_both_equations():
    rng = np.random.default_rng(0)
    for _ in range(500):
        f, g = rng.normal(size=(2, 2)), rng.normal(size=(2, 2))
        Fa, Ga = (np.repeat(a[:, :, None], 2, axis=2) for a in (f, g))
        _, pos, _ = B.face_points(Fa, Ga, 2)
        for s, t in pos[:, :2]:
            bil = lambda a: a[0, 0] * (1 - s) * (1 - t) + a[1, 0] * s * (1 - t) + a[0, 1] * (1 - s) * t + a[1, 1] * s * t
            assert max(abs(bil(f)), abs(bil(g))) < 1e-6


def test_line_on_grid_plane_found():
    X, Y, Z = grid(5)
    for c in (2.0, 2.01):
        lines = B.find_curves(X - c, Y - 2.3, 1.0)
        assert len(lines) == 1


def test_circle_on_grid_plane_closed():
    X, Y, Z = grid(12)
    for c in (4.0, 4.01):
        lines = B.find_curves(np.hypot(X - 5.5, Y - 5.5) - 3.3, Z - c, 1.0, stitch_tol=2.0)
        assert len(lines) == 1 and B.is_closed(lines[0])


def test_tilted_circle_radius_and_closure():
    X, Y, Z = grid(24)
    th = np.radians(15)
    Yr = (Y - 11.7) * np.cos(th) + (Z - 11.3) * np.sin(th)
    Zr = -(Y - 11.7) * np.sin(th) + (Z - 11.3) * np.cos(th)
    rho = np.hypot(X - 11.6, Yr)
    lines = B.find_curves(rho - 6.0, Zr, 1.0, stitch_tol=2.0)
    p = max(lines, key=len)
    assert len(lines) == 1 and B.is_closed(p)
    L = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
    assert abs(L / (2 * np.pi) - 6.0) < 0.05


def test_stitch_refuses_bridge_off_surface():
    """Two open pieces whose ends are close but whose bridge leaves the surfaces are not joined."""
    X, Y, Z = grid(10)
    f, g = X - 4.5, Y - 4.5                       # the true filament is the line x = y = 4.5
    a = np.array([[5.0, 5.0, 1.0], [5.0, 5.0, 3.0]])           # on the line (cell-centred units)
    b = np.array([[6.2, 5.0, 3.4], [6.2, 5.0, 6.0]])           # 1.25 cells away, off the line
    B.LAST.clear()
    out = B.stitch([a, b], 2.0, fields=(f, g, 1.0))
    assert len(out) == 2 and B.LAST.get("refused", 0) >= 1


# ------------------------------------------------------------------ ring bookkeeping

class Translation:
    """Plug-in model whose 'filament' is a circle moving up at speed 0.317."""
    def __init__(self):
        self.t, self.offset = 0.0, 0.0

    def dt(self, dx):
        return 0.1

    def ic(self, shape, dx, Ax, Ay, z0, **kw):
        return np.zeros(shape), np.zeros(shape)

    def step(self, u, v, un, vn, dt, dx):
        self.t += dt
        un[:] = u
        vn[:] = v

    def find(self, u, v, dx):
        z = 5 + 0.317 * self.t + self.offset
        th = np.linspace(0, 2 * np.pi, 101)
        p = np.c_[5 + 2 * np.cos(th), 5 + 2 * np.sin(th), np.full_like(th, z)]
        self.offset += int(round((5 - z) / dx)) * dx          # mirrors the roll in run()
        return [p]


def test_recentering_keeps_physical_height():
    rows, frames = E.run(0, 0, 0, 2, 2, 10, 10, 0.2, 5, every_t=0.1, verbose=False, model=Translation(), snaps=[1, 3])
    t = np.array([r["t"] for r in rows])
    z = np.array([r["z"] for r in rows])
    assert np.max(np.abs(z - (5 + 0.317 * t))) < 1e-9
    for s, ls in frames.items():
        assert abs(ls[0][:, 2].mean() - (5 + 0.317 * s)) < 1e-9


# ------------------------------------------------------------------ smoothing and crossings

def test_smooth_linear_with_gaps_and_irregular_sample():
    t = np.r_[np.arange(0, 10, 0.5), 10.1, np.arange(10.5, 30, 0.5)]
    keep = (t < 14) | (t > 16)
    ys = C.smooth(t[keep], t[keep], 2.0)
    g = np.isfinite(ys)
    assert g.sum() > 20 and np.max(np.abs(ys[g] - t[keep][g])) < 1e-9


def test_smooth_leaves_long_gap_undefined():
    t = np.arange(0, 100, 0.5)
    keep = (t < 40) | (t > 60)
    ys = C.smooth(t[keep], np.sin(t[keep]), 4.0)
    assert not np.isfinite(ys[(t[keep] > 38) & (t[keep] < 62)]).any()


def test_crossings_exact_zero():
    assert C.crossings(np.arange(3.0), np.array([-1.0, 0.0, 1.0])) == [1.0]
    assert C.crossings(np.arange(3.0), np.array([1.0, 0.0, 1.0])) == []
    assert C.crossings(np.arange(5.0), np.array([-1.0, 0, 0, 0, 2.0])) == [2.0]


def test_gap_spanning_interval_rejected():
    """Re-audit item 3: a half-cycle may not span an outage that can hide crossings."""
    t = np.arange(0, 200, 0.25)
    good = ~((t > 70) & (t < 110))
    zs = C.smooth(t, np.where(good, np.sin(np.pi * t / 20), np.nan), 4.44)
    x = C.crossings(t, zs)
    kept = [(a, b) for a, b in zip(x[:-1], x[1:]) if C.max_gap(t[good], a, b) <= 4.44 / 2]
    assert all(abs((b - a) - 20) < 0.05 for a, b in kept) and len(kept) >= 5


# ------------------------------------------------------------------ filament equation and fits

def test_linear_period_weak_ellipse():
    r = F.run(5.01, 4.99, 0.0, 1.8, 30, n=48, dt=0.01, every=0.05)
    x = C.crossings(r["t"], r["hx"] - r["hy"])
    half = np.mean(np.diff(x))
    assert abs(half / (np.pi * 25 / (2 * np.sqrt(3) * 1.8)) - 1) < 1e-4


def test_mode_fit_canonical():
    from mode_theory2 import fit_mode
    t = np.linspace(0, 100, 800)
    for A, Bm, d in ((1.0, 0.8, 0.2), (0.5, 1.2, -0.3)):
        r = A * np.exp(-0.01 * t) * np.cos(0.3 * t + 2.9)
        z = Bm * np.exp(-0.01 * t) * np.sin(0.3 * t + 2.9 + d)
        f = fit_mode(t, r, z, 0.28)
        assert f["success"] and not f["at_bound"]
        assert abs(f["ratio"] - Bm / A) < 1e-3 and abs(f["dphase"] - d) < 1e-3 and abs(f["omega"] - 0.3) < 1e-4


def test_nyquist_mode_has_no_derivative():
    k = F.wavenumbers(8)
    assert k[4] == 0 and np.allclose(np.delete(k, 4), np.delete(1j * np.fft.fftfreq(8, 1 / 8), 4))
    assert np.allclose(F.wavenumbers(7), 1j * np.fft.fftfreq(7, 1 / 7))


def test_phase_integral_of_a_steady_ring():
    import halfcycle_uniform as HU
    t = np.linspace(0, 20, 401)
    w = np.sqrt(48 * 1.8 ** 2 - 0.1 ** 2) / (2 * 5.0 ** 2)
    assert abs(HU.phase(t, np.full_like(t, 5.0), 1.8, 0.1) / (w * 20) - 1) < 1e-12


def test_calibration_local_ratio():
    import halfcycle_uniform as HU
    K = np.repeat(np.linspace(0.05, 0.3, 26), 3)
    cal = HU.Calibration(np.c_[K, 0.2 - K, 2.0 - 3.0 * K], deg=3, dK=0.05, nmin=8)
    for R in (5.0, 8.0):
        exact = 1 - 3.0 / R / (2.0 - 3.0 / R)            # 1 + K beta'(K) / beta(K)
        assert abs(cal._local_ratio(1 / R) - exact) < 1e-9 and abs(cal.beta_r_over_beta(R) - exact) < 1e-9
        assert abs(cal.ratio_r(R) - exact) < 1e-4          # interpolated on a grid of 120 curvatures


def test_folders_require_environment():
    import folders
    old = os.environ.pop("REVISION", None)
    try:
        try:
            folders.REVISION
        except RuntimeError:
            pass
        else:
            raise AssertionError("unset folder accepted")
        with tempfile.TemporaryDirectory() as d:
            os.environ["REVISION"] = d
            assert folders.REVISION == os.path.abspath(d)
    finally:
        os.environ.pop("REVISION", None)
        if old is not None:
            os.environ["REVISION"] = old


def test_helical_parts_with_missing_layers():
    from straight_filament import helical_parts
    k = 2 * np.pi / 8.0
    z = (np.arange(40) + 0.5) * 0.2
    p = 32 + 32j + 0.3 * np.exp(0.4j) * np.exp(1j * k * z) + 0.1 * np.exp(-1j) * np.exp(-1j * k * z)
    A, Bm = helical_parts(p[8:], k, z[8:])
    assert abs(A - 0.3 * np.exp(0.4j)) < 1e-12 and abs(Bm - 0.1 * np.exp(-1j)) < 1e-12


# ------------------------------------------------------------------ stale outputs

def test_halfcycle_tables_fails_on_missing_input():
    import halfcycle_tables as HT
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "halfcycles_panfilov.csv"), "w").write("old\n")
        try:
            HT.main(d)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing inputs were not fatal")
        assert open(os.path.join(d, "halfcycles_panfilov.csv")).read() == "old\n"


def test_export_never_touches_sources():
    """Re-audit 3, item A: absolute input folders must not make the exporter overwrite them."""
    import hashlib
    import json
    import export_data as ED
    with tempfile.TemporaryDirectory() as d:
        raw, out = os.path.join(d, "raw"), os.path.join(d, "archive")
        os.makedirs(os.path.join(raw, "sub"))
        src = os.path.join(raw, "sub", "run.npz")
        np.savez(src, t=np.arange(3.0), Rm=np.ones(3), raw_Rm=np.ones(3), snap_1_0=np.zeros((4, 3)))
        before = hashlib.sha256(open(src, "rb").read()).hexdigest()
        cmds, prods = ED.commands, ED.PRODUCTS
        ED.commands = lambda: [("supp", os.path.join("sub", "run.npz"), "cd v2/sub && run")]
        ED.PRODUCTS = {}
        try:
            ED.main(out, raw, raw, raw, raw)
        finally:
            ED.commands, ED.PRODUCTS = cmds, prods
        assert hashlib.sha256(open(src, "rb").read()).hexdigest() == before
        dst = os.path.join(out, "runs", "supp", "sub", "run.npz")
        k = np.load(dst).files
        assert "snap_1_0" not in k and "raw_Rm" in k and "t" in k
        m = json.load(open(os.path.join(out, "MANIFEST.json")))
        assert m["runs"][0]["source_sha256"] == before and m["runs"][0]["file"] == "runs/supp/sub/run.npz"
        try:
            ED.main(os.path.join(raw, "inner"), raw, raw, raw, raw)  # archive inside an input folder
        except ValueError:
            pass
        else:
            raise AssertionError("archive inside an input folder was accepted")


def test_export_rejects_escaping_paths():
    import export_data as ED
    with tempfile.TemporaryDirectory() as d:
        try:
            ED.safe_dest(d, "runs", "..", "..", "x.npz")
        except ValueError:
            return
        raise AssertionError("escaping destination accepted")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"ok    {n}")
        except Exception as e:          # report and continue
            failed += 1
            print(f"FAIL  {n}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
