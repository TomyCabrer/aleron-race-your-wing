"""The section polar: cl, cd, cm against alpha at one Reynolds number, with
the derived numbers a wing solver reads (linear lift slope, zero-lift
angle, the monotone pre-stall branch, cl_max / cl_min, the drag bucket).

Two sources, always labelled:
    'xfoil'     converged rows from the XFOIL binary (xfoil.py)
    'estimate'  panel2d's inviscid slope + a viscous drag/stall correlation.
                Honest to about +-15 % on drag and +-0.2 on cl_max; the
                designer prints ESTIMATE next to every number that came
                from one so nobody mistakes it for a measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

RHO = 1.2
NU = 1.5e-5              # m^2/s, air at ~20 C (rho 1.2 is the study's own)


@dataclass
class Polar:
    name: str
    re: float
    source: str
    alpha: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray
    # derived on construction
    a_lin: float = 0.0            # /rad
    alpha_L0_deg: float = 0.0
    alpha_valid: tuple = (0.0, 0.0)   # monotone pre-stall branch [deg]
    cl_max: float = 0.0
    alpha_clmax_deg: float = 0.0
    cl_min: float = 0.0
    alpha_clmin_deg: float = 0.0
    cd_min: float = 0.0
    ld_max: float = 0.0
    cl_ldmax: float = 0.0
    cm0: float = 0.0
    n_rows: int = 0

    def __post_init__(self):
        a = np.asarray(self.alpha, dtype=float)
        order = np.argsort(a)
        self.alpha = a[order]
        self.cl = np.asarray(self.cl, dtype=float)[order]
        self.cd = np.asarray(self.cd, dtype=float)[order]
        self.cm = np.asarray(self.cm, dtype=float)[order]
        self.n_rows = int(self.alpha.size)
        if self.n_rows >= 3:
            self._derive()

    # ------------------------------------------------------------------
    def _derive(self) -> None:
        al, cl, cd = self.alpha, self.cl, self.cd
        i_max = int(np.argmax(cl))
        i_min = int(np.argmin(cl[: i_max + 1])) if i_max > 0 else 0
        # the monotone branch runs from the cl minimum (negative stall or
        # the sweep's first converged row) up to cl_max
        self.cl_max, self.alpha_clmax_deg = float(cl[i_max]), float(al[i_max])
        self.cl_min, self.alpha_clmin_deg = float(cl[i_min]), float(al[i_min])
        self.alpha_valid = (float(al[i_min]), float(al[i_max]))
        b = slice(i_min, i_max + 1)
        alb, clb = al[b], cl[b]
        span = self.cl_max - self.cl_min
        lo, hi = self.cl_min + 0.25 * span, self.cl_min + 0.75 * span
        m = (clb >= lo) & (clb <= hi)
        if m.sum() < 2:
            m = np.ones_like(clb, dtype=bool)
        A = np.column_stack([np.radians(alb[m]), np.ones(m.sum())])
        slope, icpt = np.linalg.lstsq(A, clb[m], rcond=None)[0]
        self.a_lin = float(slope) if slope > 0.1 else 2.0 * math.pi
        self.alpha_L0_deg = float(math.degrees(-icpt / self.a_lin))
        self.cd_min = float(np.min(cd[b]))
        ld = np.where(cd[b] > 1e-6, clb / np.maximum(cd[b], 1e-6), 0.0)
        j = int(np.argmax(ld))
        self.ld_max, self.cl_ldmax = float(ld[j]), float(clb[j])
        self.cm0 = float(np.interp(self.alpha_L0_deg, alb, self.cm[b]))

    # ------------------------------------------------------------------
    def cl_at(self, alpha_deg):
        return np.interp(alpha_deg, self.alpha, self.cl)

    def cd_at(self, alpha_deg):
        return np.interp(alpha_deg, self.alpha, self.cd)

    def cm_at(self, alpha_deg):
        return np.interp(alpha_deg, self.alpha, self.cm)

    def cd_of_cl(self, cl):
        """Profile drag on the pre-stall branch (clamped at both stalls)."""
        i0 = int(np.argmin(np.abs(self.alpha - self.alpha_valid[0])))
        i1 = int(np.argmin(np.abs(self.alpha - self.alpha_valid[1])))
        b = slice(i0, i1 + 1)
        return np.interp(cl, self.cl[b], self.cd[b])

    def alpha_of_cl(self, cl):
        i0 = int(np.argmin(np.abs(self.alpha - self.alpha_valid[0])))
        i1 = int(np.argmin(np.abs(self.alpha - self.alpha_valid[1])))
        b = slice(i0, i1 + 1)
        return np.interp(cl, self.cl[b], self.alpha[b])

    @property
    def is_estimate(self) -> bool:
        return self.source != "xfoil"

    def to_json(self) -> dict:
        return dict(name=self.name, re=float(self.re), source=self.source,
                    alpha=[round(float(v), 4) for v in self.alpha],
                    cl=[round(float(v), 5) for v in self.cl],
                    cd=[round(float(v), 6) for v in self.cd],
                    cm=[round(float(v), 5) for v in self.cm])

    @classmethod
    def from_json(cls, d: dict) -> "Polar":
        return cls(name=str(d["name"]), re=float(d["re"]), source=str(d["source"]),
                   alpha=np.asarray(d["alpha"], float), cl=np.asarray(d["cl"], float),
                   cd=np.asarray(d["cd"], float), cm=np.asarray(d["cm"], float))


# --------------------------------------------------------------------------- #
#  the ESTIMATE                                                                #
# --------------------------------------------------------------------------- #
def reynolds(V: float, chord: float, nu: float = NU) -> float:
    return max(V, 0.1) * max(chord, 1e-3) / nu


def friction_coefficient(re: float) -> float:
    """Flat-plate Cf per side, a laminar/turbulent blend that reproduces
    XFOIL's Ncrit-9 minimum drag for a 12 % section to ~10 % at 3e5-3e6."""
    re = max(float(re), 1e4)
    cf_t = 0.455 / (math.log10(re)) ** 2.58
    cf_l = 1.328 / math.sqrt(re)
    # laminar run shrinks with Re: mostly laminar at 1e5, mostly turbulent at 1e7
    w = min(max((math.log10(re) - 5.0) / 2.0, 0.0), 1.0)      # 0 at 1e5, 1 at 1e7
    wt = 0.25 + 0.6 * w
    return wt * cf_t + (1.0 - wt) * cf_l


def estimate_polar(coords: np.ndarray, re: float, name: str = "section",
                   alphas=None) -> Polar:
    """A labelled ESTIMATE polar: inviscid slope and zero-lift angle from the
    panel method, drag from a friction + form-factor + lift-dependent
    correlation, stall from a camber/thickness correlation."""
    from . import airfoil as af
    from . import panel2d as p2
    if alphas is None:
        alphas = np.arange(-14.0, 22.01, 1.0)
    alphas = np.asarray(alphas, dtype=float)
    g = af.geometry(coords)
    lin = p2.Section2D(coords).linear()
    tc, cam = g["tc"], g["camber"]
    re = max(float(re), 5e4)
    a_inv = lin["a_lin"]
    a_vis = a_inv * (0.92 - 0.04 * max(math.log10(1e6 / re), 0.0))   # decambering
    al0 = lin["alpha_L0_deg"]
    re_f = min(max((re / 1e6) ** 0.08, 0.8), 1.15)
    cl_max = min(max(1.40 + 8.0 * cam + 1.5 * (tc - 0.12), 0.7), 2.4) * re_f
    cl_min = -min(max(1.40 - 6.0 * cam + 1.5 * (tc - 0.12), 0.45), 2.0) * re_f
    cl_lin = a_vis * np.radians(alphas - al0)
    # soft stall rounding over the last 0.15 of cl, then a dropped table
    def _round(x, lim, sgn):
        d = 0.15
        y = x.copy()
        over = sgn * (x - lim) > -d
        z = sgn * (x[over] - lim)                 # -d .. +inf
        y[over] = lim - sgn * (np.where(z < d, (d - z) ** 2 / (4 * d), 0.0))
        return y
    cl = _round(cl_lin, cl_max, +1.0)
    cl = _round(cl, cl_min, -1.0)
    # keep 3 deg past each stall so the table has a visible peak
    a_s_hi = al0 + math.degrees(cl_max / a_vis) + 1.0
    a_s_lo = al0 + math.degrees(cl_min / a_vis) - 1.0
    keep = (alphas >= a_s_lo - 3.0) & (alphas <= a_s_hi + 3.0)
    post_hi = alphas > a_s_hi
    post_lo = alphas < a_s_lo
    cl = np.where(post_hi, cl_max - 0.04 * (alphas - a_s_hi), cl)
    cl = np.where(post_lo, cl_min + 0.04 * (a_s_lo - alphas), cl)
    cf = friction_coefficient(re)
    ff = 1.0 + 2.0 * tc + 60.0 * tc ** 4
    cd0 = 2.0 * cf * ff
    cl_i = a_vis * math.radians(-al0) * 1.1              # bucket centre
    cd = cd0 + 0.0095 * (cl - cl_i) ** 2
    near_hi = np.clip((cl - (cl_max - 0.35)) / 0.35, 0.0, None)
    near_lo = np.clip(((cl_min + 0.35) - cl) / 0.35, 0.0, None)
    cd = cd + 0.02 * (near_hi ** 2 + near_lo ** 2)
    cd = np.where(post_hi | post_lo, cd + 0.03 * np.abs(alphas - np.clip(alphas, a_s_lo, a_s_hi)), cd)
    cm = np.full_like(alphas, lin["cm0"])
    return Polar(name=name, re=re, source="estimate", alpha=alphas[keep], cl=cl[keep],
                 cd=cd[keep], cm=cm[keep])


# --------------------------------------------------------------------------- #
def self_check(verbose: bool = True) -> bool:
    from . import airfoil as af
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and passed
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}: {msg}")

    p = estimate_polar(af.naca4_coords("2412"), 1e6, "naca2412")
    rep("estimate: slope 5.5-6.6 /rad", 5.5 < p.a_lin < 6.6, f"a {p.a_lin:.3f}")
    rep("estimate: alpha_L0 -1.5..-2.6", -2.6 < p.alpha_L0_deg < -1.5, f"{p.alpha_L0_deg:.2f} deg")
    rep("estimate: cl_max 1.4-1.8", 1.4 < p.cl_max < 1.8, f"{p.cl_max:.3f} at {p.alpha_clmax_deg:.0f} deg")
    rep("estimate: cd_min 0.005-0.009 at Re 1e6", 0.005 < p.cd_min < 0.009, f"{p.cd_min:.4f}")
    rep("estimate: L/D max 60-130", 60 < p.ld_max < 130, f"{p.ld_max:.0f} at cl {p.cl_ldmax:.2f}")
    rep("monotone branch covers both stalls", p.alpha_valid[0] < -8 and p.alpha_valid[1] >= 11,
        f"{p.alpha_valid}")
    q = Polar.from_json(p.to_json())
    rep("json round-trip", abs(q.a_lin - p.a_lin) < 1e-3 and q.n_rows == p.n_rows, "")
    p0 = estimate_polar(af.naca4_coords("0012"), 3e5, "naca0012")
    rep("symmetric estimate is odd", abs(p0.alpha_L0_deg) < 0.05 and abs(p0.cl_max + p0.cl_min) < 0.05,
        f"aL0 {p0.alpha_L0_deg:.3f}, cl_max {p0.cl_max:.2f}, cl_min {p0.cl_min:.2f}")
    rep("Re 3e5 draggier than 1e6", p0.cd_min > p.cd_min, f"{p0.cd_min:.4f} vs {p.cd_min:.4f}")
    hi = estimate_polar(af.load_dat(af.DATA_DIR + "/s1223.dat")[1], 3e5, "s1223")
    rep("high-lift section cl_max >= 1.85", hi.cl_max >= 1.85, f"{hi.cl_max:.2f}")
    if verbose:
        print("  ALL PASS" if ok else "  FAILURES ABOVE")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
