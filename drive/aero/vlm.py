"""Horseshoe vortex lattice for one wing with tip plates and an optional
rigid-wall image -- a compact port of urop-bo-aero's vlm.py (the solver
the AeroBO car-wing family flies), keeping its three load-bearing choices:

* one chordwise panel per strip, the control point a c/(4 pi) aft of the
  bound vortex so the SECTION's own lift slope `a` (from the polar) is
  reproduced exactly, and the normal rotated by (twist - alpha_L0) so the
  section's camber enters as its zero-lift angle;
* panel EDGES at cosine angles and STATIONS at the interlaced cosine
  midpoints -- the Trefftz-plane wash is evaluated at stations, the wake
  traces sit at edges, and pairing them any other way biases the induced
  drag (measured there: the interlaced pairing beats Munk's bound by 2 %
  on the winglet case, i.e. it is the one that is right);
* the image system carries the SAME unknowns with the boundary-condition
  sign (-1 for a rigid wall: w = 0 on the plane), so ground effect costs no
  new degrees of freedom.

Frames. The lattice is solved in ITS OWN frame: x downstream, y along the
span, z the lift direction. The caller maps that onto the car: for the top
wing the AeroBO trick is used verbatim -- the car is flipped, "lift" is
downforce, the track is a wall at z = +h ABOVE the wing and the plates hang
towards it -- so nothing is inverted by hand. For a flank panel the frame
is simply rotated: y is vertical, z is the inboard side-force direction.

Everything is linear in alpha, so a build gives two basis circulations
and every solve is an axpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import blend as bl

FOUR_PI = 4.0 * math.pi
TWO_PI = 2.0 * math.pi
_XHAT = np.array([1.0, 0.0, 0.0])
_EPS = 1e-12


def _norm(v):
    return np.sqrt((v * v).sum(-1))


def _vseg(P, A, B):
    """(n, m, 3) velocity at P (n,3) from unit-strength segments A->B (m,3)."""
    r1 = P[:, None, :] - A[None, :, :]
    r2 = P[:, None, :] - B[None, :, :]
    r0 = (B - A)[None, :, :]
    cr = np.cross(r1, r2)
    cr2 = (cr * cr).sum(-1)
    n1 = np.maximum(_norm(r1), _EPS)
    n2 = np.maximum(_norm(r2), _EPS)
    dot = (r0 * (r1 / n1[..., None] - r2 / n2[..., None])).sum(-1)
    fac = np.where(cr2 > _EPS, dot / (FOUR_PI * np.maximum(cr2, _EPS)), 0.0)
    return cr * fac[..., None]


def _vinf(P, X, u=_XHAT):
    """Velocity at P from a unit semi-infinite line from X to +inf along u."""
    r = P[:, None, :] - X[None, :, :]
    cr = np.cross(np.broadcast_to(u, r.shape), r)
    cr2 = (cr * cr).sum(-1)
    nr = np.maximum(_norm(r), _EPS)
    fac = np.where(cr2 > _EPS, (1.0 + (r @ u) / nr) / (FOUR_PI * np.maximum(cr2, _EPS)), 0.0)
    return cr * fac[..., None]


def _hshoe(P, A, B):
    return _vseg(P, A, B) + _vinf(P, B) - _vinf(P, A)


def _trefftz(P2, Q2):
    """(n, m, 2) velocity in the (y, z) plane at P2 from unit vortices at Q2
    running along +x (the 2-D kernel: x_hat x r / (2 pi r^2))."""
    d = P2[:, None, :] - Q2[None, :, :]
    r2 = (d * d).sum(-1)
    fac = np.where(r2 > _EPS, 1.0 / (TWO_PI * np.maximum(r2, _EPS)), 0.0)
    return np.stack([-d[..., 1], d[..., 0]], axis=-1) * fac[..., None]


@dataclass
class LatticeResult:
    alpha_deg: float
    CL: float
    CDi: float
    e: float
    AR: float
    S: float
    Gamma: np.ndarray
    cl: np.ndarray
    alpha_eff_deg: np.ndarray
    y: np.ndarray
    z: np.ndarray
    c: np.ndarray
    width: np.ndarray
    is_plate: np.ndarray
    y_cp: float = 0.0          # spanwise centre of pressure (one side), m
    x_ac: float = 0.0


class Lattice:
    """Build once per geometry; `solve(alpha)` for any incidence.

    span         b [m]; the wing runs y in [-b/2, b/2] at z = 0
    chord(y)     callable -> chord [m] (vectorised over y)
    twist(y)     callable -> geometric twist [rad], nose-up positive
    a, alpha_L0  section lift slope [/rad] and zero-lift angle [rad]
    plate_h      tip-plate DEVELOPED ARC [m] at both tips (0 = none). With no
                 blend the plate is a straight ray along +z and this is its
                 height, which is what it has always meant
    plate_a/L0   the plate's own section (a flat plate by default)
    plate_blend  fraction of that arc spent TURNING out of the wing plane, in
                 [0, 1] -- see `blend`. 0 is the sharp right-angle corner the
                 lattice has always built, bit-for-bit
    plate_shape  which turn law draws it (`blend.BLEND_SHAPES`)
    plate_cant_deg
                 the angle the plate finishes at, out of the wing plane. 90
                 (normal to the wing) is the only one carsim builds; the turn
                 law is normalised by it, so it has to be named
    image_z      z of a rigid wall (None = free air); the wing must sit
                 wholly on one side of it, plates included
    """

    def __init__(self, span: float, chord, twist, a: float, alpha_L0: float,
                 N: int = 24, plate_h: float = 0.0, n_plate: int = 6,
                 plate_a: float = TWO_PI, plate_L0: float = 0.0,
                 plate_blend: float = 0.0, plate_shape: str = "arc",
                 plate_cant_deg: float = 90.0,
                 image_z: float | None = None, image_sign: float = -1.0,
                 V: float = 1.0, sweep_deg: float = 0.0):
        b = float(span)
        if b <= 0.0:
            raise ValueError("span must be positive")
        self.b, self.V = b, float(V)
        th = np.pi * np.arange(N + 1) / N
        ye = -(b / 2.0) * np.cos(th)
        ys = -(b / 2.0) * np.cos(np.pi * (np.arange(N) + 0.5) / N)
        ze = np.zeros(N + 1)
        zs = np.zeros(N)
        c_main = np.asarray(chord(ys), dtype=float)
        tw_main = np.asarray(twist(ys), dtype=float)
        if np.any(c_main <= 1e-4):
            raise ValueError("chord collapses to zero somewhere on the span")
        is_plate = np.zeros(N, dtype=bool)
        A_y, B_y, A_z, B_z = ye[:-1], ye[1:], ze[:-1], ze[1:]
        c_all, tw_all = c_main, tw_main
        a_all = np.full(N, float(a))
        L0_all = np.full(N, float(alpha_L0))
        st_y, st_z = ys, zs
        h = float(plate_h)
        self.plate_h = 0.0
        self.plate_blend, self.plate_shape = 0.0, bl.check_shape(plate_shape)
        self.plate_cant_deg = float(plate_cant_deg)
        self.plate_projection = self.plate_tip_height = 0.0
        ramp_all = np.zeros(N)
        if h > 0.01 * b and n_plate > 0:
            self.plate_h = h
            self.plate_blend = float(plate_blend)
            #  THE STATIONS ARE ARC LENGTH NOW, not height. They were the same
            #  number while the plate was a straight vertical ray, and the
            #  cosine clustering (dense at the junction and at the tip) is
            #  unchanged -- so a sharp plate is panelled exactly as before.
            j = np.arange(n_plate + 1)
            se = 0.5 * h * (1.0 - np.cos(np.pi * j / n_plate))
            ss = 0.5 * h * (1.0 - np.cos(np.pi * (np.arange(n_plate) + 0.5) / n_plate))
            dy_e, dz_e = bl.path(se, h, plate_cant_deg, plate_blend, plate_shape)
            dy_s, dz_s = bl.path(ss, h, plate_cant_deg, plate_blend, plate_shape)
            self.plate_projection = float(dy_e[-1])
            self.plate_tip_height = float(dz_e[-1])
            #  ...and how far each station has got through the wing -> plate
            #  TRANSITION. 1 everywhere on a sharp corner, which is what makes
            #  the old one-step junction the step it was.
            w = bl.ramp(ss, np.ones(n_plate, bool), h, plate_cant_deg,
                        plate_blend, plate_shape)
            ye_s, ye_p = b / 2.0 + dy_e, -(b / 2.0 + dy_e)
            ys_s, ys_p = b / 2.0 + dy_s, -(b / 2.0 + dy_s)
            c_tip = float(np.asarray(chord(np.array([b / 2.0])))[0])
            tw_tip = float(np.asarray(twist(np.array([b / 2.0])))[0])
            # port plate runs from its top down to the wing tip; starboard
            # from the tip up -- both keep the sense "A -> B = +y then +z", so
            # the whole lattice is one continuous bound line
            A_y = np.concatenate([ye_p[::-1][:-1], A_y, ye_s[:-1]])
            B_y = np.concatenate([ye_p[::-1][1:], B_y, ye_s[1:]])
            A_z = np.concatenate([dz_e[::-1][:-1], A_z, dz_e[:-1]])
            B_z = np.concatenate([dz_e[::-1][1:], B_z, dz_e[1:]])
            st_y = np.concatenate([ys_p[::-1], st_y, ys_s])
            st_z = np.concatenate([dz_s[::-1], st_z, dz_s])
            #  carsim's plate carries the WING'S TIP CHORD, so there is no
            #  chord step here to ramp over (AeroBO's plate has its own chord,
            #  up to 3x the tip's, and ramps it on exactly this `w`).
            c_all = np.concatenate([np.full(n_plate, c_tip), c_all, np.full(n_plate, c_tip)])
            #  THE SECTION AND THE TWIST MEET. They used to arrive in ONE STEP
            #  at the junction panel: the wing's cambered alpha_L0 and its tip
            #  twist on one side of an edge, the plate's on the other. Now they
            #  ramp with the turn, so the surface finishes becoming the plate
            #  exactly where it finishes turning into it. At blend 0 `w` is 1
            #  and every one of these is the constant it used to be.
            tw_p = (1.0 - w) * tw_tip                 # the plate's own toe is 0
            a_p = (1.0 - w) * float(a) + w * float(plate_a)
            L0_p = (1.0 - w) * float(alpha_L0) + w * float(plate_L0)
            tw_all = np.concatenate([tw_p[::-1], tw_all, tw_p])
            a_all = np.concatenate([a_p[::-1], a_all, a_p])
            L0_all = np.concatenate([L0_p[::-1], L0_all, L0_p])
            is_plate = np.concatenate([np.ones(n_plate, bool), is_plate, np.ones(n_plate, bool)])
            ramp_all = np.concatenate([w[::-1], ramp_all, w])
        Np = c_all.size
        tanL = math.tan(math.radians(sweep_deg))
        A3 = np.column_stack([np.abs(A_y) * tanL, A_y, A_z])
        B3 = np.column_stack([np.abs(B_y) * tanL, B_y, B_z])
        st3 = np.column_stack([np.abs(st_y) * tanL, st_y, st_z])
        lvec = B3 - A3
        width = _norm(lvec)
        that = lvec / width[:, None]
        n0 = np.cross(np.broadcast_to(_XHAT, that.shape), that)
        n0 /= _norm(n0)[:, None]
        theta = tw_all - L0_all
        nrm = n0 * np.cos(theta)[:, None] + np.cross(that, n0) * np.sin(theta)[:, None]
        d = a_all * c_all / FOUR_PI
        cp = st3 + d[:, None] * _XHAT

        self.image_z = None if image_z is None else float(image_z)
        self.image_sign = float(image_sign)
        vel = _hshoe(cp, A3, B3)
        if self.image_z is not None:
            zz = np.concatenate([A3[:, 2], B3[:, 2], st3[:, 2]])
            if not (np.all(zz < self.image_z) or np.all(zz > self.image_z)):
                raise ValueError("the wing (or its plates) crosses the wall")
            gap = float(np.abs(zz - self.image_z).min())
            if gap < 0.02 * b:
                raise ValueError(f"wing within {gap:.3f} m of the wall (limit {0.02 * b:.3f})")
            mA, mB = A3.copy(), B3.copy()
            mA[:, 2] = 2.0 * self.image_z - mA[:, 2]
            mB[:, 2] = 2.0 * self.image_z - mB[:, 2]
            vel = vel + self.image_sign * _hshoe(cp, mA, mB)
        AIC = np.einsum("ijk,ik->ij", vel, nrm)
        from scipy.linalg import lu_factor, lu_solve
        lu = lu_factor(AIC)
        self._Gam0 = lu_solve(lu, -self.V * nrm[:, 0])
        self._Gam1 = lu_solve(lu, -self.V * nrm[:, 2])

        a2, b2, p2 = A3[:, 1:].copy(), B3[:, 1:].copy(), st3[:, 1:].copy()
        n2d = np.column_stack([-that[:, 2], that[:, 1]])
        n2d /= np.maximum(_norm(n2d), _EPS)[:, None]
        W = _trefftz(p2, b2) - _trefftz(p2, a2)
        if self.image_z is not None:
            a2i, b2i = a2.copy(), b2.copy()
            a2i[:, 1] = 2.0 * self.image_z - a2i[:, 1]
            b2i[:, 1] = 2.0 * self.image_z - b2i[:, 1]
            W = W + self.image_sign * (_trefftz(p2, b2i) - _trefftz(p2, a2i))
        self._WN = np.einsum("ijk,ik->ij", W, n2d)

        main = ~is_plate
        self.S = float(np.sum(c_all[main] * width[main]))
        self.AR = b * b / self.S
        self.mac = float(np.sum(c_all[main] ** 2 * width[main]) / self.S)
        self.ly, self.width, self.c = lvec[:, 1], width, c_all
        self.y, self.z, self.x = st3[:, 1], st3[:, 2], st3[:, 0]
        self.is_plate = is_plate
        self.blend_ramp = ramp_all
        self.a_panel, self.L0_panel = a_all, L0_all
        self.twist = tw_all
        self.N = N
        self.CL0 = 2.0 * float(self._Gam0 @ self.ly) / (self.V * self.S)
        self.CLa = 2.0 * float(self._Gam1 @ self.ly) / (self.V * self.S)

    # ------------------------------------------------------------------
    def gamma(self, alpha_rad: float) -> np.ndarray:
        return self._Gam0 + float(alpha_rad) * self._Gam1

    def solve(self, alpha_deg: float) -> LatticeResult:
        al = math.radians(alpha_deg)
        Gam = self.gamma(al)
        V, S = self.V, self.S
        CL = 2.0 * float(Gam @ self.ly) / (V * S)
        wn = self._WN @ Gam
        CDi = -float((self.width * Gam) @ wn) / (V * V * S)
        e = (CL * CL / (math.pi * self.AR * CDi)) if CDi > 1e-14 else float("nan")
        cl = 2.0 * Gam / (V * self.c)
        aeff = self.L0_panel + cl / self.a_panel
        main = ~self.is_plate
        lift_i = Gam[main] * self.ly[main]
        half = (self.y[main] > 0.0)
        tot = float(np.sum(lift_i[half]))
        y_cp = float(np.sum(lift_i[half] * self.y[main][half]) / tot) if abs(tot) > 1e-12 else 0.0
        return LatticeResult(alpha_deg=float(alpha_deg), CL=CL, CDi=CDi, e=float(e),
                             AR=float(self.AR), S=S, Gamma=Gam, cl=cl,
                             alpha_eff_deg=np.degrees(aeff), y=self.y, z=self.z,
                             c=self.c, width=self.width, is_plate=self.is_plate,
                             y_cp=y_cp, x_ac=float(np.sum(self.x[main] * np.abs(lift_i)) / max(np.sum(np.abs(lift_i)), 1e-12)))

    def alpha_eff_affine(self):
        """Per-panel (e0, e1): alpha_eff [deg] = e0 + e1 * alpha [deg]."""
        k = 2.0 / (self.V * self.c) / self.a_panel
        e0 = np.degrees(self.L0_panel + k * self._Gam0)
        e1 = np.degrees(k * self._Gam1) * (math.pi / 180.0)
        return e0, e1

    def stall_alpha(self, alpha_stall_deg: float, alpha_stall_neg_deg: float,
                    plate_stall_deg: float = 12.0) -> tuple[float, float]:
        """The wing incidence [deg] at which the FIRST strip reaches the
        section's positive / negative stall (the critical-section rule)."""
        e0, e1 = self.alpha_eff_affine()
        hi = np.where(self.is_plate, plate_stall_deg, alpha_stall_deg)
        lo = np.where(self.is_plate, -plate_stall_deg, alpha_stall_neg_deg)
        a_pos = np.inf
        a_neg = -np.inf
        for i in range(e0.size):
            if abs(e1[i]) < 1e-9:
                continue
            up = (hi[i] - e0[i]) / e1[i]
            dn = (lo[i] - e0[i]) / e1[i]
            if e1[i] < 0.0:
                up, dn = dn, up
            a_pos = min(a_pos, up)
            a_neg = max(a_neg, dn)
        return float(a_pos), float(a_neg)


# --------------------------------------------------------------------------- #
def rect(span, chord_m, **kw):
    return Lattice(span, lambda y: np.full_like(np.asarray(y, float), chord_m),
                   lambda y: np.zeros_like(np.asarray(y, float)), TWO_PI, 0.0, **kw)


def self_check(verbose: bool = True) -> bool:
    import time
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and passed
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}: {msg}")

    w = rect(8.0, 1.0, N=40)
    r = w.solve(5.0)
    # the AeroBO lattice's own number for this wing (aerobo.vlm.VLM, N = 40):
    # 4.55942959749662 /rad, e 0.9742625474419787 -- reproduced here to
    # round-off, which is the port's correctness gate
    rep("AR 8 rectangle: CL_alpha == AeroBO's lattice", abs(w.CLa - 4.55942959749662) < 1e-9,
        f"{w.CLa:.12f} /rad (Helmbold's lifting-line estimate would be 5.03)")
    rep("AR 8 rectangle: e == AeroBO's", abs(r.e - 0.9742625474419787) < 1e-9, f"e {r.e:.12f}")
    rep("symmetric loading", np.allclose(r.Gamma, r.Gamma[::-1], atol=1e-10), "")
    r2 = w.solve(10.0)
    rep("CDi quadratic in alpha", abs(r2.CDi / r.CDi - 4.0) < 1e-6, f"ratio {r2.CDi / r.CDi:.6f}")
    mid = r.alpha_eff_deg[r.alpha_eff_deg.size // 2]
    rep("alpha_eff < alpha everywhere (downwash), ~4.2 at mid-span, -> 0 at the tip",
        np.all(r.alpha_eff_deg < 5.0) and 4.0 < mid < 4.5 and r.alpha_eff_deg.min() < 0.5,
        f"{r.alpha_eff_deg.min():.2f}..{r.alpha_eff_deg.max():.2f} deg, mid {mid:.2f}")
    # ground effect: image at h = 0.25 b raises CL_alpha and cuts CDi at fixed CL
    g = rect(8.0, 1.0, N=40, image_z=2.0)
    rg = g.solve(5.0)
    cd_free_at_cl = r.CDi * (rg.CL / r.CL) ** 2
    rep("ground effect (h = 0.25 b): CL_alpha up 5-40 %", 1.05 < g.CLa / w.CLa < 1.40,
        f"x{g.CLa / w.CLa:.3f}")
    rep("ground effect: CDi down at fixed CL", rg.CDi < 0.9 * cd_free_at_cl,
        f"{rg.CDi:.5f} vs {cd_free_at_cl:.5f} free")
    # plates: raise e
    p = rect(8.0, 1.0, N=40, plate_h=0.8, n_plate=6)
    rp = p.solve(5.0)
    rep("tip plates (0.1 b): e above the planar wing", rp.e > r.e + 0.03, f"e {rp.e:.4f} vs {r.e:.4f}")
    rep("plates carry no lift of their own", abs(np.sum(rp.Gamma[rp.is_plate] * p.ly[rp.is_plate])) < 1e-12, "")
    # -- the wing -> plate TRANSITION --------------------------------------
    def _flank(**kw):
        return Lattice(0.78, lambda y: 0.45 * (1 - 0.3 * np.abs(y) / 0.39),
                       lambda y: np.radians(-3.0 * np.abs(y) / 0.39),
                       6.2, math.radians(-2.3), N=24, plate_h=0.12,
                       plate_a=6.0, plate_L0=math.radians(-1.0), **kw)
    sharp, bld = _flank(), _flank(plate_blend=0.6, plate_shape="spiral")
    sp, bp = sharp.is_plate, bld.is_plate
    rep("a sharp plate is FULLY the plate from its first panel -- the step",
        np.allclose(sharp.blend_ramp[sp], 1.0)
        and np.allclose(sharp.L0_panel[sp], math.radians(-1.0))
        and np.allclose(sharp.twist[sp], 0.0),
        "section and toe both arrive in one step at the junction")
    dev = np.where(bp & (bld.y > 0.0))[0]           # starboard, junction -> tip
    L0_w, tw_w = math.radians(-2.3), math.radians(-3.0)
    rep("a blended plate LEAVES THE WING at the wing's own section and toe",
        abs(bld.L0_panel[dev[0]] - L0_w) < 0.1 * abs(L0_w)
        and abs(bld.twist[dev[0]] - tw_w) < 0.1 * abs(tw_w)
        and abs(bld.L0_panel[dev[-1]] - math.radians(-1.0)) < 1e-9
        and abs(bld.twist[dev[-1]]) < 1e-9,
        f"alpha_L0 {np.degrees(bld.L0_panel[dev[0]]):+.2f} -> "
        f"{np.degrees(bld.L0_panel[dev[-1]]):+.2f} deg, toe "
        f"{np.degrees(bld.twist[dev[0]]):+.2f} -> {np.degrees(bld.twist[dev[-1]]):+.2f} deg")
    rep("...monotonically, on the turn's own ramp",
        np.all(np.diff(bld.blend_ramp[dev]) >= -1e-12)
        and np.all(np.diff(np.abs(bld.L0_panel[dev])) <= 1e-12),
        f"ramp {bld.blend_ramp[dev][0]:.3f} -> {bld.blend_ramp[dev][-1]:.3f}")
    rep("the WING is untouched by the ramp -- only the plate turns",
        np.allclose(sharp.c[~sp], bld.c[~bp]) and np.allclose(sharp.twist[~sp], bld.twist[~bp])
        and np.allclose(sharp.L0_panel[~sp], bld.L0_panel[~bp])
        and np.allclose(sharp.y[~sp], bld.y[~bp]),
        "carsim's transition is device-side only (blend.py's module note)")
    #  arc length is the family's invariant, so a blend is the same SIZE of
    #  plate -- it trades height for outboard reach and a smooth corner. The
    #  PANELS are straight chords of that arc, so a panelled curve is always a
    #  little short; what has to hold is that the deficit is the discretisation
    #  and converges away, not a plate that quietly shrank.
    dev6 = np.sum(bld.width[bp]) / (2.0 * 0.12)
    d24 = _flank(plate_blend=0.6, plate_shape="spiral", n_plate=24)
    dev24 = np.sum(d24.width[d24.is_plate]) / (2.0 * 0.12)
    rep("a blend keeps the plate's ARC, so it is the same size",
        abs(np.sum(sharp.width[sp]) - 2.0 * 0.12) < 1e-12
        and 0.98 < dev6 < 1.0 and (1.0 - dev24) < 0.1 * (1.0 - dev6),
        f"panelled/arc {dev6:.5f} at 6 panels, {dev24:.5f} at 24 -- chords of a curve; "
        f"it trades {sharp.plate_tip_height - bld.plate_tip_height:.4f} m of height for "
        f"{bld.plate_projection:.4f} m of reach, a side")
    rep("blend 0 IS the published lattice, bit-for-bit",
        np.array_equal(sharp.y, _flank(plate_blend=0.0, plate_shape="spiral").y)
        and sharp.solve(4.0).CDi == _flank(plate_blend=0.0).solve(4.0).CDi,
        "no wing in the library moves because this module exists")

    # cambered, twisted, tapered flank-sized panel
    lat = Lattice(0.78, lambda y: 0.45 * (1 - 0.3 * np.abs(y) / 0.39), lambda y: np.radians(-2.0 * np.abs(y) / 0.39),
                  6.2, math.radians(-2.3), N=24, plate_h=0.08)
    rs = lat.solve(0.0)
    rep("cambered section lifts at zero incidence", rs.CL > 0.05, f"CL0 {rs.CL:.3f}, CLa {lat.CLa:.3f}")
    ap, an = lat.stall_alpha(14.0, -10.0)
    rep("critical-section stall angles bracket zero", an < 0.0 < ap, f"{an:.1f} .. {ap:.1f} deg")
    t0 = time.perf_counter()
    for _ in range(10):
        Lattice(1.4, lambda y: np.full_like(y, 0.3), lambda y: np.zeros_like(y), 6.0, -0.04,
                N=24, plate_h=0.12, image_z=0.9).solve(6.0)
    ms = (time.perf_counter() - t0) * 1e3 / 10
    rep("cost: build+solve (N 24 + plates + image)", ms < 25.0, f"{ms:.2f} ms")
    if verbose:
        print("  ALL PASS" if ok else "  FAILURES ABOVE")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
