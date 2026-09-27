"""The wing -> end plate TRANSITION: how the two surfaces meet, and what the
corner costs.

A port of urop-bo-aero's transition geometry (`geometry.winglet_turn_angle`
/ `winglet_path`, `vlm.transition_ramp`) and of its junction charge
(`junction.py`), in carsim's units and carsim's frame.

THE PROBLEM IT EXISTS FOR
-------------------------
`vlm.Lattice` used to bolt the tip plates on at a right angle and change
everything about the surface in ONE STEP at the junction panel: the section
(the wing's cambered `alpha_L0` -> the plate's), the twist (the wing's tip
twist -> the plate's zero) and, in AeroBO's family, the chord. That is not a
wing with end plates, it is two surfaces sharing an edge, and AeroBO's own
note on the same defect is blunt about it -- a "blended" corner joined a
0.21 m chord to a 0.62 m one across a corner the junction model was
simultaneously giving a fillet credit for.

WHAT A BLEND IS HERE
--------------------
The plate's quarter-chord line leaves the wing plane TANGENTIALLY, turns
through the cant over `blend * h` of arc, then runs straight to the tip.
`h` is the plate's DEVELOPED ARC LENGTH, and it is the invariant: every
(blend, shape) plate of the same `h` has the same quarter-chord length and,
at equal chord, the same wetted area -- so comparing two of them compares
SHAPE, not size. `blend = 0` is a sharp corner with no turning arc, which is
the published geometry bit-for-bit and the reason every wing in the library
is unmoved by this module's existence.

What the blend trades is TIP HEIGHT for PROJECTED SPAN plus a smooth
junction. The first two the lattice sees directly (a shorter, wider device
is a different wake); the third it cannot see at all, which is what
`junction_report` is for.

WHAT IS NOT PORTED, AND WHY
---------------------------
* THE WING-SIDE ARC. AeroBO can start the turn INBOARD of the tip
  (`wing_arc` > 0), bending the wing's own outer panels into the transition.
  The law below takes the argument -- `blend_radius` is not honest without
  it -- but `vlm.Lattice` passes 0.0, so carsim's transition is device-side
  only. AeroBO's stated reason to want a wing-side arc is that a blend
  confined to the device has R <= h/|cant|, "a fraction of a tip chord", so
  the fillet credit saturates immediately and the blend "reads as a corner".
  That argument is about ITS scale, not carsim's: a flank plate at h 0.12 m
  and full blend has R = 0.076 m against a 0.45 m chord, R/c = 0.17, which
  is already PAST `FILLET_FULL_R_OVER_C`. The credit saturates here because
  the plate is deep relative to the chord, not because the blend is too
  small to matter, so there is nothing for a wing-side arc to buy.
* A SIGNED / VARIABLE CANT. carsim's plates stand normal to the wing at
  cant 90, which is what `Lattice` has always built; the functions take a
  cant because the turn law is normalised by it, and nothing offers to
  change it.
* THE CHORD RAMP. AeroBO's plate has its own chord, up to 3x the wing's tip
  chord, so its chord steps at the junction too. carsim's plate chord IS the
  wing's tip chord (`vlm.Lattice`), so that step does not exist here and the
  ramp over it would be the identity. `ramp` is what a chord scale would be
  interpolated on if one were ever added -- and the DRAWING has one now: an
  AeroBO wing carries its plate's chord ratio and chord law to the garage and
  the chase view, and `plate_stations` ramps them in on exactly that.
"""

from __future__ import annotations

import math

import numpy as np

#: How the turn is DISTRIBUTED along the transition. All three turn through
#: the same cant over the same arc, so they are the same size and differ only
#: in shape.
#:
#:   'arc'     constant curvature -- the circular fillet. Its curvature JUMPS
#:             at both ends, so the surface has a crease where it meets the
#:             wing and another where it stops turning.
#:   'smooth'  the smoothstep 3u^2 - 2u^3: curvature vanishes at both ends
#:             (G2, crease-free) at the price of a peak curvature 3/2 of the
#:             mean.
#:   'spiral'  the clothoid pair -- quadratic curvature ramp in, constant
#:             middle, ramp out. Also G2, and its peak is only 1/(1 - r) =
#:             4/3 of the mean, i.e. the same crease-free junction with an
#:             11 % gentler elbow. This is the road/rail transition spiral
#:             and the shape a blended device is actually drawn with.
BLEND_SHAPES = ("arc", "smooth", "spiral")

#: Fraction of the clothoid's turn spent on EACH curvature ramp. r -> 0
#: recovers the circular fillet, r = 1/3 matches the smoothstep's 3/2 peak,
#: r = 1/2 is the ramp-only triangle. 1/4 is AeroBO's MODEL CHOICE, kept:
#: crease-free, and the gentlest elbow of the two G2 laws. Not a measurement.
SPIRAL_RAMP_FRAC = 0.25

#: Gauss-Legendre nodes the turn is integrated on. The integrand is analytic,
#: so this is machine-accurate, and it is FIXED rather than derived from the
#: caller's sampling: `path(t)` must depend on `t` alone, or the plate's
#: geometry would move when `n_plate` did.
_QUAD_N = 32
#: ...and its nodes and weights, computed ONCE: `leggauss` is an eigen-solve
#: (~0.25 ms a call), and the garage's car page draws a plate every frame
_QUAD = np.polynomial.legendre.leggauss(_QUAD_N)

#: Internal breakpoints (in u) of each law -- the stations where its
#: curvature stops being one analytic piece. Gauss-Legendre is spectrally
#: accurate only WITHIN a smooth piece, and the path has to be unit-speed to
#: machine precision or arc length (the invariant of the whole family) leaks.
_BREAKS = {"spiral": (SPIRAL_RAMP_FRAC, 1.0 - SPIRAL_RAMP_FRAC)}


def check_shape(shape: str) -> str:
    if shape not in BLEND_SHAPES:
        raise ValueError(f"unknown blend shape {shape!r}; choose from {list(BLEND_SHAPES)}")
    return shape


def _turn_law(u, shape: str):
    """Normalised turn law L(u) on [0, 1]: L(0) = 0, L(1) = 1.

    The turn angle is `phi L(u)` and the curvature `(phi/s_tot) L'(u)`, so L
    alone decides the SHAPE of a transition -- its length, its total turn and
    therefore its arc length and wetted area are fixed outside."""
    u = np.asarray(u, dtype=float)
    if shape == "arc":
        return u
    if shape == "smooth":
        return u * u * (3.0 - 2.0 * u)
    r = SPIRAL_RAMP_FRAC
    K = 1.0 / (1.0 - r)                       # plateau slope, fixed by L(1) = 1
    return np.where(u < r, 0.5 * K * u * u / r,
                    np.where(u <= 1.0 - r, K * (u - 0.5 * r),
                             1.0 - 0.5 * K * (1.0 - u) ** 2 / r))


def _turn_law_deriv(u, shape: str):
    """L'(u) -- the normalised CURVATURE profile. Peak/mean IS the tightness
    a crease-free junction pays for; see `blend_radius_min`."""
    u = np.asarray(u, dtype=float)
    if shape == "arc":
        return np.ones_like(u)
    if shape == "smooth":
        return 6.0 * u * (1.0 - u)
    r = SPIRAL_RAMP_FRAC
    K = 1.0 / (1.0 - r)
    return np.where(u < r, K * u / r,
                    np.where(u <= 1.0 - r, np.full_like(u, K), K * (1.0 - u) / r))


def _arcs(h: float, blend_frac: float, wing_arc: float) -> tuple:
    """(s_in, s_out, s_tot): the turn's arc INBOARD of the junction on the
    wing, OUTBOARD on the plate, and their sum."""
    s_in = float(wing_arc)
    if s_in < 0.0:
        raise ValueError(f"wing_arc must be >= 0, got {s_in}")
    bf = float(blend_frac)
    if not (0.0 <= bf <= 1.0):
        raise ValueError(f"blend must be in [0, 1], got {bf}")
    s_out = bf * float(h)
    return s_in, s_out, s_in + s_out


# --------------------------------------------------------------------------- #
#  the turn                                                                    #
# --------------------------------------------------------------------------- #
def turn_angle(t, h: float, cant_deg: float, blend_frac: float = 0.0,
               shape: str = "arc", wing_arc: float = 0.0):
    """Angle out of the wing plane [rad] at arc length `t` from the junction
    (+ outboard along the plate, - inboard along the wing).

        psi(t) = phi L(u),   u = (t + s_in) / (s_in + s_out)

    Every law reaches exactly `phi` at the outer end and holds it after, so
    every plate has the same cant, the same arc length and the same wetted
    area. A SHARP corner has no turning region (s_tot = 0) and the full cant
    is reported everywhere -- which is what makes `blend = 0` the step it is.
    """
    t = np.asarray(t, dtype=float)
    phi = math.radians(float(cant_deg))
    _, _, s_tot = _arcs(h, blend_frac, wing_arc)
    if s_tot <= 0.0:
        return np.full_like(t, phi)
    u = np.clip((t + float(wing_arc)) / s_tot, 0.0, 1.0)
    return phi * _turn_law(u, check_shape(shape))


def curvature(t, h: float, cant_deg: float, blend_frac: float = 0.0,
              shape: str = "arc", wing_arc: float = 0.0):
    """Curvature [1/m] of the transition line at arc length `t` (signed).

    Reported so the SHAPE of a transition can be checked directly -- the
    circular fillet is a step, the other two are bumps that start and end at
    zero -- and so the junction charge can price the tightest radius drawn."""
    t = np.asarray(t, dtype=float)
    phi = math.radians(float(cant_deg))
    s_in, s_out, s_tot = _arcs(h, blend_frac, wing_arc)
    if s_tot <= 0.0:
        return np.zeros_like(t)
    inside = (t >= -s_in) & (t <= s_out)
    u = np.clip((t + s_in) / s_tot, 0.0, 1.0)
    k = (phi / s_tot) * _turn_law_deriv(u, check_shape(shape))
    return np.where(inside, k, 0.0)


def ramp(t, is_device, h: float, cant_deg: float, blend_frac: float = 0.0,
         shape: str = "arc", wing_arc: float = 0.0):
    """Fraction of the wing -> plate TRANSITION each station has completed:

        w(t) = psi(t) / phi   in [0, 1]

    -- 0 before the transition starts, 1 once the plate is at its cant, and
    whichever turn law is drawing the corner in between. It is what the
    plate's SECTION and TWIST (and, in a family that has one, its chord) are
    interpolated on, so the surface finishes becoming the plate exactly where
    it finishes turning into it.

    A sharp corner reports the full cant everywhere, and the second line is
    what makes that the step it is: nothing turns before the transition
    starts, so every wing station stays the wing and every plate station is
    fully the plate. A zero cant is a coplanar extension with no turn to ramp
    on, and the plate is the plate from the junction out."""
    t = np.asarray(t, dtype=float)
    phi = math.radians(float(cant_deg))
    dev = np.asarray(is_device, dtype=float)
    if phi == 0.0:
        return dev
    w = np.clip(turn_angle(t, h, cant_deg, blend_frac, shape, wing_arc) / phi, 0.0, 1.0)
    return np.where(t > -float(wing_arc), w, 0.0)


def _turn_xy(s, s_tot: float, phi: float, shape: str) -> tuple:
    """(y, z) accumulated from the START of the turn, for `s` in [0, s_tot].

    The circular fillet integrates in closed form; the others use the fixed
    Gauss-Legendre rule, one per smooth piece of the law so a station inside
    a curvature ramp is as exact as one inside the plateau."""
    s = np.asarray(s, dtype=float)
    if shape == "arc":
        R = s_tot / phi                                   # signed radius
        psi = np.clip(s / R, min(0.0, phi), max(0.0, phi))
        return R * np.sin(psi), R * (1.0 - np.cos(psi))
    xi, w = _QUAD
    breaks = _BREAKS.get(shape, ())
    if not breaks:
        tau = s[:, None] * 0.5 * (xi + 1.0)[None, :]
        psi = phi * _turn_law(np.clip(tau / s_tot, 0.0, 1.0), shape)
        half = 0.5 * s
        return half * (np.cos(psi) @ w), half * (np.sin(psi) @ w)
    edge = np.array([0.0, *breaks, 1.0], dtype=float) * s_tot
    lo = np.clip(edge[:-1][None, :], 0.0, s[:, None])
    hi = np.clip(edge[1:][None, :], 0.0, s[:, None])
    half = 0.5 * (hi - lo)
    tau = (0.5 * (lo + hi))[:, :, None] + half[:, :, None] * xi
    psi = phi * _turn_law(np.clip(tau / s_tot, 0.0, 1.0), shape)
    return (np.einsum("mkn,mk,n->m", np.cos(psi), half, w),
            np.einsum("mkn,mk,n->m", np.sin(psi), half, w))


def path(t, h: float, cant_deg: float, blend_frac: float = 0.0,
         shape: str = "arc") -> tuple:
    """(dy, dz) of the plate's quarter-chord line at developed arc `t` from
    the junction, in the lattice's own frame -- `dy` outboard along the span,
    `dz` along the lift direction the plate stands in.

    The path is unit-speed by construction: (dy, dz) is the integral of
    (cos psi, sin psi), so arc length is conserved exactly whatever the turn
    law does, and `path(h)` is the plate's tip however it is blended.

    Device side only (see the module note): `t` >= 0."""
    t = np.asarray(t, dtype=float)
    phi = math.radians(float(cant_deg))
    _, s_out, _ = _arcs(h, blend_frac, 0.0)
    check_shape(shape)
    if s_out <= 0.0:                       # sharp corner: a straight ray
        return t * math.cos(phi), t * math.sin(phi)
    inside = np.minimum(t, s_out)
    y, z = _turn_xy(inside, s_out, phi, shape)
    tail = np.maximum(t - s_out, 0.0)      # ...then straight, at the cant
    return y + tail * math.cos(phi), z + tail * math.sin(phi)


def projection(h: float, cant_deg: float, blend_frac: float = 0.0,
               shape: str = "arc") -> float:
    """Spanwise reach of the plate's TIP beyond the junction, m.

    `h cos(cant)` for the sharp ray -- zero for carsim's vertical plate --
    but a blended plate leaves the wing plane tangentially and so reaches
    OUTBOARD for the same arc length. Anything that measures the car's width
    has to use this number and not the cosine, or a blend would smuggle in
    free projected span."""
    return float(path(np.array([float(h)]), h, cant_deg, blend_frac, shape)[0][0])


def tip_height(h: float, cant_deg: float, blend_frac: float = 0.0,
               shape: str = "arc") -> float:
    """Reach of the plate's tip along the lift direction, m -- what the
    plate has to CLEAR the wall by, as opposed to the arc it spends getting
    there. The two are the same number only for a straight plate."""
    return float(path(np.array([float(h)]), h, cant_deg, blend_frac, shape)[1][0])


def height_fraction(cant_deg: float, blend_frac: float = 0.0,
                    shape: str = "arc") -> float:
    """`tip_height / h` -- and it is a function of the blend and the shape
    ALONE, because the path is homogeneous of degree one in `h`. That is what
    lets a clearance clip be inverted in one line: a plate allowed `z_max` of
    reach may be `z_max / height_fraction(...)` of arc."""
    return tip_height(1.0, cant_deg, blend_frac, shape)


# --------------------------------------------------------------------------- #
#  the DRAWING: the plate's line and chord where a picture needs them          #
# --------------------------------------------------------------------------- #
#: Least chord a DRAWN plate is given, over the wing's tip chord. AeroBO's own
#: `vlm.DEVICE_CHORD_FLOOR`, the same number and the same job: below it AeroBO
#: refuses a continued chord as a knife edge, so no wing it designed draws
#: thinner, and a hand-made wing whose continued taper would run out is drawn
#: at the floor instead of vanishing (or turning inside out) down its plate.
PLATE_CHORD_FLOOR = 0.05


def plate_stations(h: float, cant_deg: float, blend_frac: float = 0.0,
                   shape: str = "arc", c_tip: float = 1.0, scale: float = 1.0,
                   slope: float = 0.0, back: float = 0.0, n_turn: int = 4,
                   t_end: float | None = None) -> dict:
    """The plate as a DRAWING needs it: its line and its chord at the few arc
    stations that describe both exactly (the turn sampled, the straight run
    and a linear chord need only their ends), so the garage and the chase
    view sweep one shape and the game can afford it every frame.

    `h` is the DRAWN arc -- the plate's own height, or a carrying plate's
    reach to the body -- and the blend turns over `blend_frac * h` of it, as
    `path` does. `back` is a straight stub continuing a SHARP plate's line
    past the tip to the wing's far side: carsim has always drawn its plates
    through the tip, and a blended plate grows OUT of the wing and has none.

    The chord is AeroBO's lattice law (`vlm.VLM`, its winglet panels):
    `c_tip` held, or -- `slope` < 0, `endplate_chord_follows` -- the wing's
    own chord law continued past the tip at `slope` metres of chord per metre
    of arc, floored at PLATE_CHORD_FLOOR of the tip chord; times `scale`, the
    plate's chord ratio, which a blend RAMPS in over the turn (on `ramp`'s
    law) so the surface becomes the plate where it finishes turning into it,
    and a sharp corner steps at the junction.

    `t_end` (< h) draws the SAME plate only as far as that arc: a carrying
    plate longer than its reach, cut where it meets the body
    (`carried_stations`) -- its turn still spans `blend_frac * h`, so the
    drawn line is the flown one, not a shorter plate's tighter turn.

    Returns arrays over the stations: t (arc from the junction, negative on
    the stub), dy / dz (`path`'s frame: outboard, towards the wall), psi (the
    line's angle out of the wing plane: the plate's thickness is across it)
    and c (its chord)."""
    h = max(float(h), 0.0)
    phi = math.radians(float(cant_deg))
    s_out = float(blend_frac) * h
    ts = [0.0]
    if back > 0.0 and s_out <= 0.0:
        #  a sharp plate through the tip is ONE straight member (one box, as
        #  carsim drew it) unless its continued chord kinks at the junction
        ts = [-float(back)] + ([0.0] if slope < 0.0 else [])
    if s_out > 0.0:
        ts += list(np.linspace(0.0, s_out, int(n_turn) + 1)[1:])
    floor = PLATE_CHORD_FLOOR * float(c_tip)
    if slope < 0.0:                      # where the continued chord meets the floor
        t_fl = (float(c_tip) - floor) / -float(slope)
        if 0.0 < t_fl < h:
            ts.append(t_fl)
    if h > ts[-1] + 1e-9:
        ts.append(h)
    t = np.unique(np.array(ts, dtype=float))
    if t_end is not None and float(t_end) < h:
        t = np.append(t[t < float(t_end) - 1e-12], float(t_end))
    tp = np.maximum(t, 0.0)
    dy, dz = path(tp, h, cant_deg, blend_frac, shape)
    stub = t < 0.0
    dy = np.where(stub, t * math.cos(phi), dy)
    dz = np.where(stub, t * math.sin(phi), dz)
    psi = np.where(stub, phi, turn_angle(tp, h, cant_deg, blend_frac, shape))
    if s_out > 0.0 and phi != 0.0:
        w = np.clip(psi / phi, 0.0, 1.0)
    else:
        w = np.ones_like(t)              # sharp: the plate is the plate from its root
    base = np.maximum(float(c_tip) + float(slope) * tp, floor)
    return dict(t=t, dy=np.asarray(dy, float), dz=np.asarray(dz, float), psi=psi,
                c=base * (1.0 + (float(scale) - 1.0) * w))


def reach_arc(drop_at, cant_deg: float, blend_frac: float = 0.0,
              shape: str = "arc", cap: float = 2.0) -> float:
    """The arc a CARRYING plate needs to meet the body: the least h at which
    its tip has come down `drop_at(reach)` -- the body's distance from the
    junction, along the plate's standing direction, at the plate's own
    outboard reach -- because a leaning plate lands further out than it
    starts, where a car's shoulder is lower. Both the tip's height and its
    reach are h times a shape constant (`path` is degree one in h), so the
    root is bracketed on sampled arcs and bisected. `cap` when the plate runs
    alongside the body and never meets it inside that much arc."""
    hf = height_fraction(cant_deg, blend_frac, shape)
    p1 = projection(1.0, cant_deg, blend_frac, shape)

    def f(h):
        return h * hf - float(drop_at(h * p1))

    hs = np.linspace(0.0, float(cap), 33)
    for i in range(1, len(hs)):
        if f(hs[i]) >= 0.0:
            lo, hi = hs[i - 1], hs[i]
            for _ in range(32):                  # 2 m / 32 / 2^32: far below a pixel
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if f(mid) >= 0.0 else (mid, hi)
            return float(hi)
    return float(cap)


#: a BRACKET's chord: the dark strut carsim draws from a carrying plate's own
#: end to its body, as a share of the plate's end chord, never below 4 cm
BRACKET_CHORD_FRAC = 0.3
BRACKET_CHORD_MIN = 0.04


def carried_stations(h_own: float, drop_at, cant_deg: float, blend_frac: float = 0.0,
                     shape: str = "arc", c_tip: float = 1.0, scale: float = 1.0,
                     slope: float = 0.0, back: float = 0.0, cap: float = 2.0):
    """A CARRYING plate as drawn: `(plate, bracket, reached)`.

    The PLATE is AeroBO's: its own arc `h_own` (the designed `endplate_h_m`,
    whose reach margin AeroBO checked against the deck it was told), its lean,
    its turn over `blend_frac * h_own` and its chord. AeroBO's deck is a
    plane; the car's body is not -- a wing as wide as the roof has its tips
    over the shoulder, and a lean carries the plate's foot further out still,
    where the body falls away. So where the plate stops short of the drawn
    body, carsim draws the rest as a BRACKET: a slim dark strut
    (`BRACKET_CHORD_FRAC` of the plate's end chord) continuing the plate's
    final line down to the body -- the fixing, not more plate. The owner
    (2026-09-25) saw the plate itself drawn half a metre down the car's side
    to its waist, which was carsim's reach passed off as AeroBO's part.

    Where AeroBO's plate is already long enough it is cut ON the body, on ITS
    OWN line: the arc where the plate of `h_own`, turning over `blend_frac *
    h_own`, meets the body (`geometry.winglet_arc_to_height`'s question on
    carsim's body), and there is no bracket. Not the shorter plate that just
    reaches (`reach_arc`): that plate turns over `blend_frac` of ITS arc -- a
    tighter turn the engine never flew, landing up to 0.12 m further inboard
    at a full blend (AeroBO's own warning: a shorter device "would
    redistribute the blend"). A sharp plate's line is straight, so the two
    agree there. A carsim plate (`h_own` >= `cap`: its arc IS its reach) is
    still the plate that reaches. `reached` is False when not even the
    bracket gets there inside `cap` of arc (a lean too shallow for this body):
    it is drawn falling short, not bent onto it."""
    h_own = max(float(h_own), 0.0)
    if 0.0 < h_own < float(cap):
        def g(t):
            y, z = path(np.array([t]), h_own, cant_deg, blend_frac, shape)
            return float(z[0]) - float(drop_at(float(y[0])))
        if g(h_own) >= 0.0:
            ts = np.linspace(0.0, h_own, 33)
            for i in range(1, len(ts)):
                if g(ts[i]) >= 0.0:
                    lo, hi = ts[i - 1], ts[i]
                    for _ in range(32):
                        mid = 0.5 * (lo + hi)
                        lo, hi = (lo, mid) if g(mid) >= 0.0 else (mid, hi)
                    return (plate_stations(h_own, cant_deg, blend_frac, shape, c_tip, scale,
                                           slope, back, t_end=float(hi)), None, True)
    need = reach_arc(drop_at, cant_deg, blend_frac, shape, cap)
    if h_own <= 0.0 or need <= h_own + 1e-9:
        return (plate_stations(need, cant_deg, blend_frac, shape, c_tip, scale, slope, back),
                None, need < cap - 1e-9)
    st = plate_stations(h_own, cant_deg, blend_frac, shape, c_tip, scale, slope, back)
    y0, z0, psi = float(st["dy"][-1]), float(st["dz"][-1]), float(st["psi"][-1])
    cy, cz = math.cos(psi), math.sin(psi)

    def f(e):
        return (z0 + e * cz) - float(drop_at(y0 + e * cy))

    room = max(float(cap) - h_own, 0.0)
    e, reached = room, False
    es = np.linspace(0.0, room, 33)
    for i in range(1, len(es)):
        if f(es[i]) >= 0.0:
            lo, hi = es[i - 1], es[i]
            for _ in range(32):
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if f(mid) >= 0.0 else (mid, hi)
            e, reached = float(hi), True
            break
    c = max(BRACKET_CHORD_FRAC * float(st["c"][-1]), BRACKET_CHORD_MIN)
    br = dict(t=np.array([h_own, h_own + e]), dy=np.array([y0, y0 + e * cy]),
              dz=np.array([z0, z0 + e * cz]), psi=np.array([psi, psi]), c=np.array([c, c]))
    return st, br, reached


#: a flank plate's foot this far past the body's top or bottom gets a stay
STAY_MIN_M = 0.01


def plate_foot(root, span_dir, stand_dir, st: dict) -> np.ndarray:
    """Where a plate's (or a bracket's) line ENDS, in world metres: its last
    station's centre, placed as `plate_rings` places it."""
    return (np.asarray(root, float) + float(st["dy"][-1]) * np.asarray(span_dir, float)
            + float(st["dz"][-1]) * np.asarray(stand_dir, float))


def side_stay(z_foot: float, z_top: float, z_bottom: float) -> tuple | None:
    """`(z_lo, z_hi)` of the upright STAY a carrying FLANK plate needs, or
    None. AeroBO's flank plate reaches a car side that is a wall of any
    height; the drawn body's side only runs from its bottom `z_bottom` up to
    its top `z_top` there (the belt, a bonnet's edge at the front wheels).
    A foot above it is held down onto the top by the stay, one below it up
    to the bottom -- the fixing, drawn like a bracket (the owner, 2026-09-27:
    "Wing tips not well connected")."""
    z = float(z_foot)
    if z > float(z_top) + STAY_MIN_M:
        return float(z_top), z
    if z < float(z_bottom) - STAY_MIN_M:
        return z, float(z_bottom)
    return None


#: a flank wing its endplates carry, STOWED, slides in until its section
#: clears the car's side by this much; its rigid plates run on into the body
#: (the owner, 2026-09-27: switched off, the plates never went away)
STOW_GAP_M = 0.02
#: the section's face towards the car, per chord: a bound on the upper
#: surface of the sections carsim flies (NACA 4412's is 0.098)
STOW_SECTION_UP = 0.15


def stowed_standoff(chord: float, inc_deg: float) -> float:
    """How far off the car's side a flank wing its endplates carry stands
    STOWED [m]: its suction face (up to `STOW_SECTION_UP` of the chord) and
    the half chord the incidence swings in, clear of the side by
    `STOW_GAP_M`. The plates are the structure, so the wing cannot fold
    away; it retracts against the side and its plates into the body."""
    th = math.radians(abs(float(inc_deg)))
    return float(chord) * (STOW_SECTION_UP * math.cos(th) + 0.5 * math.sin(th)) + STOW_GAP_M


def plate_rings(root, span_dir, stand_dir, st: dict, thick: float = 0.012,
                wall=None) -> list:
    """`plate_stations` swept into closed rectangle rings (5 corners, the
    first repeated) in world metres, for a loft: each station a flat plate
    of its chord along x and `thick` across the line.

    `root` is the junction (the tip's mid-chord), `span_dir` the unit vector
    outboard along the span, `stand_dir` the one the plate stands towards
    (the wall). `wall` = (axis, fn) sets a CARRYING plate's last ring down on
    the body -- fn(corner) is the body's z there (axis 2: the deck, its slope
    and its shoulder) or the car side's y (axis 1) -- so its foot sits on the
    body under each corner instead of floating off one end of it."""
    root = np.asarray(root, float)
    a_, b_ = np.asarray(span_dir, float), np.asarray(stand_dir, float)
    ex = np.array([1.0, 0.0, 0.0])
    rings = []
    for dy, dz, psi, c in zip(st["dy"], st["dz"], st["psi"], st["c"]):
        ctr = root + dy * a_ + dz * b_
        n = -math.sin(psi) * a_ + math.cos(psi) * b_          # across the line
        hx, ht = 0.5 * c * ex, 0.5 * thick * n
        rings.append(np.array([ctr + hx + ht, ctr - hx + ht, ctr - hx - ht,
                               ctr + hx - ht, ctr + hx + ht]))
    if wall is not None and rings:
        axis, fn = wall
        last = rings[-1]
        for k in range(len(last)):
            last[k, axis] = float(fn(last[k]))
    return rings


# --------------------------------------------------------------------------- #
#  what the lattice cannot see: the corner's own drag                          #
# --------------------------------------------------------------------------- #
#  The Trefftz plane sees the wake trace and nothing else, so a lifting-
#  surface method values the wing/plate junction ONLY through the (y, z) line
#  it draws. By that measure a sharp corner and a smooth blend of the same
#  arc length are two different lines, and the reason a real car wing blends
#  the corner -- the interference drag of two surfaces meeting at an angle,
#  and the separation risk in the corner -- is invisible.
#
#  Hoerner (Fluid-Dynamic Drag, 1965, ch. 8) correlates the interference drag
#  of an unfilleted right-angle junction between two streamlined surfaces as
#  a DRAG AREA on the junction member's thickness,
#
#      D/q = t^2 (17 (t/c)^2 - 0.05)
#
#  quoted for roughly 0.05 <= t/c <= 0.20. Below t/c ~ 0.054 it turns
#  negative, which is where the correlation stops meaning anything, so the
#  floor is applied explicitly rather than letting a negative drag through.
#
#  The FILLET CREDIT is not literature. Hoerner reports that a fillet removes
#  much of the junction drag and the design rule of thumb is a radius of
#  ~10 % of the local chord; there is no closed-form radius law in the
#  source, so the credit is a straight-line decay from 1 (sharp) to a floor.
#  It is a CALIBRATED SHAPE -- monotone, bounded, and reducing to the
#  unfilleted correlation at R = 0 -- and not a prediction of a particular
#  fillet's drag. Which is why this charge is OFF unless a blend is being
#  asked about, and why it is reported as its own line item beside the
#  UNCREDITED number, so it can never be mistaken for solver output.

#: validity band of the Hoerner correlation (thickness ratio at the corner)
TC_VALID = (0.05, 0.20)
#: R/c at which the fillet credit reaches its floor. MODEL CHOICE.
FILLET_FULL_R_OVER_C = 0.10
#: residual fraction of the unfilleted drag a large fillet leaves. Not zero:
#: a blended corner still carries some interference. MODEL CHOICE.
FILLET_CREDIT_FLOOR = 0.15


def hoerner_drag_area(tc: float, chord: float) -> float:
    """Unfilleted right-angle junction drag AREA D/q [m^2]. Returns 0 (never
    a negative drag) where the correlation goes negative."""
    tc = float(tc)
    if not (0.0 < tc < 1.0):
        raise ValueError(f"thickness ratio must be in (0, 1), got {tc}")
    if not (float(chord) > 0.0):
        raise ValueError(f"chord must be > 0, got {chord}")
    t = tc * float(chord)
    return float(max(0.0, t * t * (17.0 * tc * tc - 0.05)))


def fillet_credit(radius: float, chord: float) -> float:
    """Multiplier in [FILLET_CREDIT_FLOOR, 1] for a fillet of `radius`."""
    if float(chord) <= 0.0:
        raise ValueError(f"chord must be > 0, got {chord}")
    r_c = max(0.0, float(radius)) / float(chord)
    frac = min(1.0, r_c / FILLET_FULL_R_OVER_C)
    return float(1.0 - (1.0 - FILLET_CREDIT_FLOOR) * frac)


def junction_cd(tc: float, chord: float, s_ref: float, radius: float = 0.0,
                n_junctions: int = 2) -> float:
    """Interference-drag coefficient on `s_ref` for `n_junctions` corners.
    A wing with a plate at each tip has two."""
    if int(n_junctions) < 0:
        raise ValueError("n_junctions must be >= 0")
    if not (float(s_ref) > 0.0):
        raise ValueError(f"reference area must be > 0, got {s_ref}")
    return float(int(n_junctions) * hoerner_drag_area(tc, chord)
                 * fillet_credit(radius, chord) / float(s_ref))


def blend_radius(h: float, cant_deg: float, blend_frac: float = 0.0,
                 wing_arc: float = 0.0) -> float:
    """MEAN radius of curvature [m] of the corner `path` draws: the turning
    arc over the angle turned. A zero blend (or a zero cant) is a sharp
    corner, reported as radius 0.

    Shape-INDEPENDENT on purpose. Total turn / total turning length is the
    same for every law that reaches the same cant over the same arc, and it
    is the only radius this reduced-order correlation can honestly be a
    function of: the credit above is a calibrated shape, so letting it
    separate a circular fillet from a curvature-continuous one would invent a
    distinction the model cannot see. The two DO differ in the wake they draw
    (which the lattice sees) and in their tightest radius, reported beside
    this one by `blend_radius_min`."""
    phi = abs(math.radians(float(cant_deg)))
    _, _, s_tot = _arcs(h, blend_frac, wing_arc)
    if s_tot <= 0.0 or phi == 0.0:
        return 0.0
    return float(s_tot / phi)


def blend_radius_min(h: float, cant_deg: float, blend_frac: float = 0.0,
                     shape: str = "arc", wing_arc: float = 0.0) -> float:
    """Tightest radius anywhere in the transition [m]. Equal to the mean for
    the constant-curvature 'arc'; a crease-free law spends its ends turning
    less so it must turn harder in the middle -- the smoothstep peaks at 3/2
    of the mean curvature (minimum radius 2/3 of the fillet's) and the
    clothoid at 4/3 (3/4). Reported, never charged, so a study can see what a
    curvature-continuous junction costs in local tightness."""
    r_mean = blend_radius(h, cant_deg, blend_frac, wing_arc)
    if r_mean <= 0.0:
        return 0.0
    k = np.abs(curvature(np.linspace(-float(wing_arc), blend_frac * float(h), 129),
                         h, cant_deg, blend_frac, shape, wing_arc))
    k_max = float(np.max(k))
    return float(1.0 / k_max) if k_max > 0.0 else r_mean


def junction_report(tc: float, chord: float, s_ref: float, h: float,
                    cant_deg: float, blend_frac: float = 0.0,
                    n_junctions: int = 2, shape: str = "arc",
                    wing_arc: float = 0.0) -> dict:
    """The full junction line item: radius, credit and CD, beside the
    UNCREDITED sharp number so the trade is readable."""
    r = blend_radius(h, cant_deg, blend_frac, wing_arc)
    return {
        "CD_junction": junction_cd(tc, chord, s_ref, r, n_junctions),
        "CD_junction_sharp": junction_cd(tc, chord, s_ref, 0.0, n_junctions),
        "blend_radius_m": r,
        "blend_radius_min_m": blend_radius_min(h, cant_deg, blend_frac, shape, wing_arc),
        "blend_shape": str(shape),
        "blend_r_over_c": float(r / chord) if chord > 0 else 0.0,
        "fillet_credit": fillet_credit(r, chord),
        "n_junctions": int(n_junctions),
        "tc_in_correlation_band": bool(TC_VALID[0] <= tc <= TC_VALID[1]),
    }


# --------------------------------------------------------------------------- #
def self_check(verbose: bool = True) -> bool:
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and passed
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}: {msg}")

    # -- the ramp ----------------------------------------------------------
    s = np.array([-0.3, -0.05, 0.01, 0.1, 0.2, 0.4])
    dev = s > 0.0
    sharp = ramp(s, dev, h=0.45, cant_deg=90.0, blend_frac=0.0, shape="spiral")
    rep("a sharp corner is EXACTLY the step", np.array_equal(sharp, dev.astype(float)),
        "every plate station fully the plate, every wing station the wing")
    r8 = ramp(s, dev, h=0.45, cant_deg=90.0, blend_frac=0.8, shape="spiral")
    rep("a blend is a monotone ramp that completes with the turn",
        np.all(r8[~dev] == 0.0) and np.all(np.diff(r8) >= 0.0)
        and 0.0 < r8[2] < r8[3] < r8[4] < 1.0 and abs(r8[-1] - 1.0) < 1e-12,
        f"{r8[2]:.3f} -> {r8[4]:.3f} -> {r8[-1]:.3f}")
    good = True
    for shape in BLEND_SHAPES:
        t = np.linspace(-0.2, 0.45, 41)
        w = ramp(t, t > 0, 0.45, 90.0, 0.8, shape)
        psi = turn_angle(t, 0.45, 90.0, 0.8, shape)
        good = good and np.allclose(w, np.where(t > 0.0, psi / math.radians(90.0), 0.0))
    rep("the ramp IS the turn angle over the cant, for every shape", good,
        "not a shape of its own: the surface stops being the wing where it stops pointing like it")
    co = ramp(np.array([-0.1, 0.1, 0.3]), np.array([False, True, True]), 0.45, 0.0, 0.7, "arc")
    rep("a coplanar device has no turn to ramp on",
        np.array_equal(co, np.array([0.0, 1.0, 1.0])), "")

    # -- the path ----------------------------------------------------------
    t = np.linspace(0.0, 0.2, 9)
    y0, z0 = path(t, 0.2, 90.0, 0.0)
    rep("blend 0 is the straight vertical plate, bit-for-bit",
        np.allclose(y0, 0.0, atol=1e-15) and np.allclose(z0, t),
        "which is why no wing in the library moves")
    lens = {}
    for shape in BLEND_SHAPES:
        tt = np.linspace(0.0, 0.2, 4001)
        yy, zz = path(tt, 0.2, 90.0, 0.7, shape)
        lens[shape] = float(np.sum(np.hypot(np.diff(yy), np.diff(zz))))
    rep("arc length is the invariant: every shape is the same SIZE",
        all(abs(v - 0.2) < 2e-7 for v in lens.values()),
        "  ".join(f"{k} {v:.7f}" for k, v in lens.items()) + " vs h 0.200")
    rep("a blend trades tip height for outboard reach",
        abs(projection(0.2, 90.0, 0.0)) < 1e-15 and projection(0.2, 90.0, 1.0, "spiral") > 0.05
        and tip_height(0.2, 90.0, 1.0, "spiral") < 0.2,
        f"full blend: reach {projection(0.2, 90.0, 1.0, 'spiral'):.4f} m, "
        f"height {tip_height(0.2, 90.0, 1.0, 'spiral'):.4f} m of 0.200 m of arc")
    rep("...and WingLab's own number for it: a quarter round reaches 61 % as high",
        abs(height_fraction(90.0, 1.0, "spiral") - 0.61) < 0.02 * 0.61,
        f"{height_fraction(90.0, 1.0, 'spiral'):.4f}")
    rep("the height fraction is scale-free (the path is degree one in h)",
        abs(tip_height(0.37, 90.0, 0.6, "smooth") / 0.37
            - height_fraction(90.0, 0.6, "smooth")) < 1e-12,
        "so a clearance clip inverts in one line")
    #  the port's correctness gate, the way `vlm.self_check` pins the lattice:
    #  urop-bo-aero's own `geometry.winglet_path(0.2, h 0.2, cant 90)` for
    #  every shape and four blends, reproduced here to the bit
    AEROBO = {
        ("arc", 0.0): (1.2246467991473533e-17, 0.2),
        ("arc", 0.3): (0.038197186342054885, 0.1781971863420549),
        ("arc", 0.55): (0.07002817496043395, 0.16002817496043392),
        ("arc", 1.0): (0.12732395447351627, 0.12732395447351624),
        ("smooth", 0.0): (1.2246467991473533e-17, 0.2),
        ("smooth", 0.3): (0.03630862132968567, 0.17630862132968567),
        ("smooth", 0.55): (0.06656580577109039, 0.1565658057710904),
        ("smooth", 1.0): (0.12102873776561887, 0.12102873776561887),
        ("spiral", 0.0): (1.2246467991473533e-17, 0.2),
        ("spiral", 0.3): (0.03645723696352015, 0.17645723696352017),
        ("spiral", 0.55): (0.06683826776645363, 0.1568382677664536),
        ("spiral", 1.0): (0.12152412321173385, 0.1215241232117338),
    }
    worst = 0.0
    for (shape, bf), (Y, Z) in AEROBO.items():
        y, z = path(np.array([0.2]), 0.2, 90.0, bf, shape)
        worst = max(worst, abs(float(y[0]) - Y), abs(float(z[0]) - Z))
    rep("the tip lands where WingLab's own geometry puts it, 12 cases",
        worst == 0.0, f"worst |delta| {worst:.3g} m over 3 shapes x 4 blends")
    rep("...and so do the reach, the height and the tightest radius",
        projection(0.2, 90.0, 0.7, "spiral") == 0.08506688624821367
        and tip_height(0.2, 90.0, 0.7, "spiral") == 0.14506688624821368
        and blend_radius_min(0.2, 90.0, 0.8, "spiral") == 0.07639437268410978,
        "urop-bo-aero geometry.winglet_projection / winglet_tip_height / "
        "junction.blend_radius_min")
    alone = float(path(np.array([0.2]), 0.2, 90.0, 0.55, "spiral")[0][0])
    n_indep = True
    for n in (4, 7, 40, 400):
        tt = np.linspace(0.0, 0.2, n + 1)
        n_indep = n_indep and float(path(tt, 0.2, 90.0, 0.55, "spiral")[0][-1]) == alone
    rep("a station's geometry depends on its own arc alone, not on n_plate",
        n_indep, "the same grid-independence rule the chord area factor exists for")

    # -- the corner's own drag --------------------------------------------
    rep("a blend cannot make the junction charge grow",
        junction_cd(0.12, 0.45, 0.35, blend_radius(0.12, 90.0, 0.5))
        < junction_cd(0.12, 0.45, 0.35, 0.0),
        f"{junction_cd(0.12, 0.45, 0.35, blend_radius(0.12, 90.0, 0.5)):.5f} "
        f"vs sharp {junction_cd(0.12, 0.45, 0.35, 0.0):.5f}")
    rep("...and cannot make it negative, or free",
        hoerner_drag_area(0.04, 0.45) == 0.0
        and abs(fillet_credit(10.0, 0.45) - FILLET_CREDIT_FLOOR) < 1e-12,
        f"t/c 0.04 is below the correlation's own root; a huge fillet still leaves "
        f"{FILLET_CREDIT_FLOOR:.0%}")
    rep("the tightest radius is the mean for the fillet and less for the G2 laws",
        abs(blend_radius_min(0.2, 90.0, 0.8, "arc") / blend_radius(0.2, 90.0, 0.8) - 1.0) < 1e-9
        and abs(blend_radius_min(0.2, 90.0, 0.8, "smooth") / blend_radius(0.2, 90.0, 0.8) - 2.0 / 3.0) < 1e-3
        and abs(blend_radius_min(0.2, 90.0, 0.8, "spiral") / blend_radius(0.2, 90.0, 0.8) - 0.75) < 1e-3,
        "arc 1.000  smooth 0.667  spiral 0.750 of the mean radius")
    j = junction_report(0.12, 0.45, 0.35, 0.12, 90.0, 0.6, shape="spiral")
    rep("a flank plate's blend saturates the credit (the module note, measured)",
        j["blend_r_over_c"] > FILLET_FULL_R_OVER_C and j["tc_in_correlation_band"],
        f"R/c {j['blend_r_over_c']:.3f} at blend 0.6, floor reached at "
        f"{FILLET_FULL_R_OVER_C:.2f} -- no wing-side arc would buy anything")
    bad = False
    try:
        turn_angle(0.0, 0.2, 90.0, 1.4)
    except ValueError:
        bad = True
    rep("a blend outside [0, 1] is refused, not clamped", bad, "")

    # -- the drawing ---------------------------------------------------------
    st = plate_stations(0.3, 70.0, 0.4, "spiral", c_tip=0.2, scale=1.8)
    yz = path(np.array([0.3]), 0.3, 70.0, 0.4, "spiral")
    st0 = plate_stations(0.3, 90.0, 0.0, "arc", c_tip=0.2, scale=1.3, back=0.15)
    rep("the drawn plate is `path` at its stations, its chord ramped in on the turn",
        abs(st["dy"][-1] - yz[0][0]) < 1e-15 and abs(st["dz"][-1] - yz[1][0]) < 1e-15
        and abs(st["c"][0] - 0.2) < 1e-15 and abs(st["c"][-1] - 0.36) < 1e-12
        and len(st0["t"]) == 2 and np.allclose(st0["c"], 0.26),
        f"blended: {st['c'][0]:.3f} -> {st['c'][-1]:.3f} m over {len(st['t'])} stations; a "
        f"sharp plate through the tip is one member ({len(st0['t'])} stations)")
    stf = plate_stations(0.5, 90.0, c_tip=0.2, slope=-0.8)
    rep("a continued chord is floored, never a knife edge or inside out",
        abs(float(stf["c"].min()) - PLATE_CHORD_FLOOR * 0.2) < 1e-15
        and float(stf["t"][1]) == (0.2 - PLATE_CHORD_FLOOR * 0.2) / 0.8,
        f"{float(stf['c'][0]):.3f} -> {float(stf['c'].min()):.3f} m, the floor met at "
        f"t {float(stf['t'][1]):.4f} m")
    h_flat = reach_arc(lambda _dy: 0.4, 70.0, 0.4, "spiral")
    shoulder = (lambda dy: 0.4 + 0.5 * max(dy - 0.05, 0.0))
    h_slope = reach_arc(shoulder, 70.0, 0.4, "spiral")
    tip = path(np.array([h_slope]), h_slope, 70.0, 0.4, "spiral")
    rep("a carrying plate's reach lands its tip on the body: a flat deck inverts the "
        "height fraction, a falling shoulder takes more arc and is met where it is",
        abs(h_flat * height_fraction(70.0, 0.4, "spiral") - 0.4) < 1e-9
        and h_flat < h_slope < 2.0 and abs(float(tip[1][0]) - shoulder(float(tip[0][0]))) < 1e-9,
        f"flat {h_flat:.4f} m, shoulder {h_slope:.4f} m of arc")
    if verbose:
        print("  ALL PASS" if ok else "  FAILURES ABOVE")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
