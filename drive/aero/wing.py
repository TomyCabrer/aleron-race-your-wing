"""WingSpec (what the designer edits, what the library stores) ->
WingAero (what the vehicle reads at 1 kHz).

The analysis is the AeroBO car-wing recipe on a single wing: lattice with
tip plates and (for the top wing) the track's image, section polar read at
each strip's EFFECTIVE angle for profile drag, critical-section stall, a
wetted-area charge for the struts or pylons. Its output is deliberately
small -- an affine lift law, a quadratic drag law and two stall clamps:

    CL(alpha)  = clamp(CL0 + CLa * alpha, CL_min, CL_max)
    CD(CL)     = cd0 + cd1 * CL + cd2 * CL^2

so the physics step never calls a solver. `analyse` is incidence-free
(everything is affine in the mount angle); `design_point` evaluates one
mount angle for the read-outs.

Roles
    'flank'  a vertical panel standing off the car's side: its "lift" is
             the SIDE force the study is about (span = vertical extent)
    'top'    a rear/roof wing making downforce: solved in AeroBO's mirrored
             frame (lift = downforce, wall above, plates towards it)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

from .polar import Polar, RHO, NU, reynolds, friction_coefficient
from .vlm import Lattice, TWO_PI

ROLES = ("flank", "top")

# packaging bands the designer / optimiser stay inside (m, deg)
BOUNDS = {
    "flank": dict(span=(0.35, 1.05), chord=(0.20, 0.70), taper=(0.35, 1.0),
                  twist_deg=(-6.0, 6.0), plate_h=(0.0, 0.16), inc_deg=(-6.0, 14.0)),
    "top": dict(span=(0.70, 1.64), chord=(0.12, 0.50), taper=(0.35, 1.0),
                twist_deg=(-6.0, 6.0), plate_h=(0.0, 0.30), inc_deg=(-2.0, 16.0)),
}
V_REF = {"flank": 29.0875, "top": 40.0}      # m/s: R = 100 m limit speed / a fast straight
RE_BANK = (1e5, 1.5e5, 2e5, 3e5, 5e5, 7e5, 1e6, 1.5e6, 2e6, 3e6)


def re_bank_snap(re: float) -> float:
    lr = math.log(max(re, 1.0))
    return min(RE_BANK, key=lambda r: abs(math.log(r) - lr))


@dataclass
class WingSpec:
    name: str = "new wing"
    role: str = "flank"
    airfoil: str = "naca2412"
    span: float = 0.78
    chord: float = 0.45
    taper: float = 1.0
    twist_deg: float = 0.0
    plate_h: float = 0.0
    n_strips: int = 24
    notes: str = ""
    builtin: bool = False
    #: the study's closed-form device (CONTRACT section 4): CL0 and L/D fixed,
    #: S = 0.35 m^2. Present only on the two published panels 'fin' / 'plate';
    #: the vehicle then runs its legacy branch bit-for-bit.
    legacy: dict | None = None
    aero: dict = field(default_factory=dict)

    # ---- geometry --------------------------------------------------------
    @property
    def S(self) -> float:
        return self.span * self.chord * 0.5 * (1.0 + self.taper)

    @property
    def AR(self) -> float:
        return self.span ** 2 / max(self.S, 1e-9)

    @property
    def mac(self) -> float:
        lam = self.taper
        return (2.0 / 3.0) * self.chord * (1.0 + lam + lam * lam) / (1.0 + lam)

    def chord_at(self, eta):
        """Chord at |y|/(b/2) = eta in [0, 1] (linear taper)."""
        return self.chord * (1.0 - (1.0 - self.taper) * np.asarray(eta, float))

    def clamp(self) -> "WingSpec":
        if self.role not in ROLES:
            self.role = "flank"
        b = BOUNDS[self.role]
        for k in ("span", "chord", "taper", "twist_deg", "plate_h"):
            lo, hi = b[k]
            setattr(self, k, float(min(max(getattr(self, k), lo), hi)))
        self.n_strips = int(min(max(self.n_strips, 8), 48))
        return self

    def copy(self, **changes) -> "WingSpec":
        d = asdict(self)
        d.update(changes)
        d["aero"] = dict(self.aero) if "aero" not in changes else changes["aero"]
        return WingSpec(**d)

    def to_json(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "WingSpec":
        return cls(name=str(d.get("name", "wing")), role=str(d.get("role", "flank")),
                   airfoil=str(d.get("airfoil", "naca2412")), span=float(d.get("span", 0.78)),
                   chord=float(d.get("chord", 0.45)), taper=float(d.get("taper", 1.0)),
                   twist_deg=float(d.get("twist_deg", 0.0)), plate_h=float(d.get("plate_h", 0.0)),
                   n_strips=int(d.get("n_strips", 24)), notes=str(d.get("notes", "")),
                   builtin=bool(d.get("builtin", False)), legacy=d.get("legacy"),
                   aero=dict(d.get("aero", {}))).clamp()

    def reynolds(self, V: float | None = None) -> float:
        return reynolds(V_REF[self.role] if V is None else V, self.mac)


# --------------------------------------------------------------------------- #
#  analysis                                                                    #
# --------------------------------------------------------------------------- #
def build_lattice(spec: WingSpec, polar: Polar, ride_h: float | None = None) -> Lattice:
    b = spec.span
    lam, c0 = spec.taper, spec.chord
    tw = math.radians(spec.twist_deg)

    def chord(y):
        return c0 * (1.0 - (1.0 - lam) * np.abs(np.asarray(y, float)) / (0.5 * b))

    def twist(y):
        return tw * np.abs(np.asarray(y, float)) / (0.5 * b)

    image = None
    plate = spec.plate_h
    if spec.role == "top" and ride_h is not None:
        image = float(ride_h)                  # the track, above, in the mirrored frame
        plate = min(plate, max(ride_h - 0.03 * b - 0.01, 0.0))
    return Lattice(b, chord, twist, polar.a_lin, math.radians(polar.alpha_L0_deg),
                   N=spec.n_strips, plate_h=plate, n_plate=6, plate_a=TWO_PI, plate_L0=0.0,
                   image_z=image, image_sign=-1.0, V=1.0)


def strut_cd(spec: WingSpec, standoff: float, s_ref: float, V: float) -> float:
    """Wetted-area friction of the mounts on S_ref: two struts (flank) or two
    pylons (top) of `standoff` length; chords 0.04 m / 0.10 m."""
    c = 0.04 if spec.role == "flank" else 0.10
    cf = friction_coefficient(reynolds(V, c)) * 1.3      # + form factor
    return 2.0 * 2.0 * max(standoff, 0.0) * c * cf / max(s_ref, 1e-6)


def analyse(spec: WingSpec, polar: Polar, V: float | None = None, ride_h: float | None = None,
            standoff: float = 0.45, rho: float = RHO) -> dict:
    """The affine/quadratic laws the vehicle reads, plus a table for plots.

    Raises ValueError for a geometry the lattice refuses (e.g. plates into
    the track); the caller shows the reason and keeps the last good aero."""
    V = V_REF[spec.role] if V is None else float(V)
    lat = build_lattice(spec, polar, ride_h)
    S, AR = lat.S, lat.AR
    a_pos, a_neg = lat.stall_alpha(polar.alpha_valid[1], polar.alpha_valid[0])
    a_pos = min(a_pos, 40.0)
    a_neg = max(a_neg, -40.0)
    CL0, CLa = lat.CL0, lat.CLa
    CL_max = CL0 + CLa * math.radians(a_pos)
    CL_min = CL0 + CLa * math.radians(a_neg)
    n_pts = 15
    alphas = np.linspace(a_neg, a_pos, n_pts)
    CLs, CDs, CDis, CDps, es = [], [], [], [], []
    main = ~lat.is_plate
    plate = lat.is_plate
    c_w = lat.c * lat.width
    cf_plate = friction_coefficient(reynolds(V, spec.chord * max(spec.taper, 0.3))) * 2.0 * 1.05
    cd_strut = strut_cd(spec, standoff, S, V)
    for a in alphas:
        r = lat.solve(float(a))
        cdp = float(np.sum(polar.cd_at(np.clip(r.alpha_eff_deg[main], polar.alpha_valid[0],
                                                polar.alpha_valid[1])) * c_w[main]) / S)
        cd_pl = float(np.sum((cf_plate + 0.012 * r.cl[plate] ** 2) * c_w[plate]) / S) if plate.any() else 0.0
        CLs.append(r.CL)
        CDis.append(r.CDi)
        CDps.append(cdp)
        CDs.append(r.CDi + cdp + cd_pl + cd_strut)
        es.append(r.e)
    CLs, CDs = np.asarray(CLs), np.asarray(CDs)
    # the law is fitted on the inner 80 % of the pre-stall range: the vehicle
    # clamps CL at the stalls, and the steep rise there would bend a
    # quadratic everywhere else
    inner = slice(1, n_pts - 1)
    A = np.column_stack([np.ones(n_pts), CLs, CLs * CLs])
    cd0, cd1, cd2 = np.linalg.lstsq(A[inner], CDs[inner], rcond=None)[0]
    fit_err = float(np.max(np.abs(A[inner] @ np.array([cd0, cd1, cd2]) - CDs[inner])))
    r_mid = lat.solve(0.5 * (a_neg + a_pos))
    return dict(
        V_ref=V, rho=rho, Re=float(polar.re), polar_source=polar.source, ride_h=ride_h,
        S=float(S), AR=float(AR), MAC=float(lat.mac), plate_h_flown=float(lat.plate_h),
        CL0=float(CL0), CLa=float(CLa), CL_max=float(CL_max), CL_min=float(CL_min),
        alpha_stall_deg=float(a_pos), alpha_stall_neg_deg=float(a_neg),
        e=float(np.nanmedian(np.asarray(es))) if np.isfinite(np.nanmedian(np.asarray(es))) else 0.0,
        cd0=float(cd0), cd1=float(cd1), cd2=float(cd2), cd_fit_err=fit_err,
        cd_strut=float(cd_strut), CDp_min=float(np.min(CDps)),
        y_cp=float(r_mid.y_cp),
        table=dict(alpha=[round(float(v), 3) for v in alphas], CL=[round(float(v), 4) for v in CLs],
                   CD=[round(float(v), 5) for v in CDs], CDi=[round(float(v), 5) for v in CDis],
                   CDp=[round(float(v), 5) for v in CDps]),
    )


def spanwise(spec: WingSpec, polar: Polar, inc_deg: float, ride_h: float | None = None) -> dict:
    """Strip loading at one mount angle, for the designer's plot."""
    lat = build_lattice(spec, polar, ride_h)
    r = lat.solve(inc_deg)
    m = ~r.is_plate
    return dict(y=r.y[m].tolist(), cl=r.cl[m].tolist(), aeff=r.alpha_eff_deg[m].tolist(),
                c=r.c[m].tolist(), CL=r.CL, CDi=r.CDi, e=r.e,
                stalled=bool(np.any(r.alpha_eff_deg[m] > polar.alpha_valid[1])
                             or np.any(r.alpha_eff_deg[m] < polar.alpha_valid[0])),
                y_plate=r.y[~m].tolist(), z_plate=r.z[~m].tolist(), cl_plate=r.cl[~m].tolist())


# --------------------------------------------------------------------------- #
#  the laws the vehicle runs                                                   #
# --------------------------------------------------------------------------- #
def wing_cl(aero: dict, alpha_deg: float) -> float:
    cl = aero["CL0"] + aero["CLa"] * math.radians(alpha_deg)
    return min(max(cl, aero["CL_min"]), aero["CL_max"])


def wing_cd(aero: dict, cl: float) -> float:
    return max(aero["cd0"] + aero["cd1"] * cl + aero["cd2"] * cl * cl, 0.0)


def design_point(spec: WingSpec, inc_deg: float, V: float | None = None, x_w: float = 0.97,
                 R: float = 100.0) -> dict:
    """Read-outs at one mount angle: force, drag, L/D, stall margin and, for
    a flank panel, the study's corner-speed gain (crossover.gain)."""
    a = spec.aero
    if not a:
        return {}
    V = a["V_ref"] if V is None else float(V)
    q = 0.5 * a["rho"] * V * V
    if spec.legacy:
        cl = min(max(spec.legacy["CL0"] + 2.47 * math.radians(inc_deg), 0.0), 1.6)
        S = spec.legacy.get("S", 0.35)
        F = q * S * cl
        D = F / spec.legacy["LD"]
        cl_max = 1.6
        margin = (cl_max - cl) / 2.47
    else:
        cl = wing_cl(a, inc_deg)
        S = a["S"]
        F = q * S * cl
        D = q * S * wing_cd(a, cl)
        cl_max = a["CL_max"]
        margin = math.radians(a["alpha_stall_deg"] - inc_deg) if a["CLa"] > 0 else 0.0
    out = dict(V=V, q=q, CL=cl, S=S, F=F, D=D, LD=(F / D if D > 1e-9 else 0.0),
               stall_margin_deg=math.degrees(margin) if not spec.legacy else (cl_max - cl) / 2.47 * 57.3,
               stalled=cl >= cl_max - 1e-9)
    if spec.role == "flank":
        try:
            import crossover
            k = 0.5 * a["rho"] * S * cl
            g = crossover.gain(k, R, x_w)
            out["gain_pct"] = None if g is None else 100.0 * g
            out["k"] = k
        except Exception:
            out["gain_pct"] = None
    return out


# --------------------------------------------------------------------------- #
def self_check(verbose: bool = True) -> bool:
    from . import airfoil as af
    from .polar import estimate_polar
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and passed
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}: {msg}")

    e423 = af.load_dat(af.DATA_DIR + "/e423.dat")[1]
    spec = WingSpec("flank-e423", "flank", "e423", span=0.78, chord=0.45, taper=1.0, plate_h=0.0)
    pol = estimate_polar(e423, re_bank_snap(spec.reynolds()), "e423")
    a = analyse(spec, pol)
    spec.aero = a
    rep("flank panel: S = 0.35 m^2 (the study's one panel)", abs(a["S"] - 0.351) < 0.002, f"S {a['S']:.4f}")
    rep("flank panel: CL0 0.3-1.0 from a high-lift section", 0.3 < a["CL0"] < 1.0, f"CL0 {a['CL0']:.3f}")
    rep("flank panel: CLa 2.0-3.4 /rad (AR 1.7 lattice; the study used 2.47 at AR 1.3)", 2.0 < a["CLa"] < 3.4,
        f"CLa {a['CLa']:.3f}")
    rep("flank panel: CL_max 1.2-1.9", 1.2 < a["CL_max"] < 1.9, f"{a['CL_max']:.3f} at {a['alpha_stall_deg']:.1f} deg")
    dp = design_point(spec, 0.0)
    inc12 = math.degrees((1.2 - a["CL0"]) / a["CLa"])
    dp12 = design_point(spec, inc12)
    rep("design point at CL 1.2: L/D 2.5-5.5 (the study's 3.2 at CL 1.25, AR 1.2-1.4)",
        2.5 < dp12["LD"] < 5.5 and abs(dp12["CL"] - 1.2) < 1e-6,
        f"L/D {dp12['LD']:.2f} at {inc12:.1f} deg; at 0 deg CL {dp['CL']:.2f} L/D {dp['LD']:.1f} "
        f"F {dp['F']:.0f} N D {dp['D']:.0f} N at {dp['V']:.1f} m/s")
    rep("design point: corner-speed gain reported", dp.get("gain_pct") is not None and dp["gain_pct"] > 0,
        f"{dp.get('gain_pct', 0):+.2f} % at R = 100")
    rep("drag law fit within 30 counts on the inner range", a["cd_fit_err"] < 0.003, f"{a['cd_fit_err'] * 1e4:.1f} counts")
    sp2 = spec.copy(plate_h=0.10)
    a2 = analyse(sp2, pol)
    rep("plates raise CL0 and CLa", a2["CL0"] > a["CL0"] and a2["CLa"] > a["CLa"],
        f"CLa {a['CLa']:.3f} -> {a2['CLa']:.3f}")
    top = WingSpec("rear", "top", "s1223", span=1.40, chord=0.30, taper=0.85, plate_h=0.12)
    s1223 = af.load_dat(af.DATA_DIR + "/s1223.dat")[1]
    pt = estimate_polar(s1223, re_bank_snap(top.reynolds()), "s1223")
    free = analyse(top, pt, ride_h=None)
    ge = analyse(top, pt, ride_h=0.45)
    rep("top wing in ground effect (h 0.45 m): more downforce slope than free air",
        ge["CLa"] > free["CLa"] * 1.02, f"CLa {free['CLa']:.3f} -> {ge['CLa']:.3f}")
    top.aero = ge
    dpt = design_point(top, 6.0)
    rep("top wing: 200-1200 N at 40 m/s and 6 deg", 200 < dpt["F"] < 1200, f"F {dpt['F']:.0f} N, D {dpt['D']:.0f} N, L/D {dpt['LD']:.1f}")
    try:
        analyse(top.copy(plate_h=0.30), pt, ride_h=0.20)
        rep("plates are shortened to clear the track, never through it", True, "")
    except ValueError as exc:
        rep("plates are shortened to clear the track, never through it", False, str(exc))
    back = WingSpec.from_json(spec.copy(aero=a).to_json())
    rep("spec json round-trip", back.aero["CLa"] == a["CLa"] and back.name == spec.name, "")
    import time
    t0 = time.perf_counter()
    for _ in range(5):
        analyse(spec, pol)
    ms = (time.perf_counter() - t0) * 1e3 / 5
    rep("analyse cost", ms < 40.0, f"{ms:.1f} ms")
    if verbose:
        print("  ALL PASS" if ok else "  FAILURES ABOVE")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
