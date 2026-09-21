"""The SECTION as a design problem, scored on the mission's LAP.

Where this sits
---------------
Second of the three steps the garage now walks, and AeroBO's own order:

    mission  (mission.py)   what the wing is for -- a lap, in seconds
    SECTION  (this file)    designed in 2-D, where a candidate costs 3 ms
    wing     (wing.py)      the planform, asked afterwards whether the
                            winner helped it

`carsection.py` in urop-bo-aero makes the argument for the split: putting
sixteen aerofoil weights into the wing's own design vector would make a
27-row search out of a family that repository's optimiser study already
records as defeating every method it has at 7 and 10 rows. So the section is
designed here, cheaply, and the wing is asked afterwards -- and the weak point
of that split is stated rather than hidden: **a better section is not a better
wing.** The wing page re-flies the winner on the real lattice and can disagree.

The design vector
-----------------
ONE element, `N_CST = 4` weights per surface, a thickness row and the wing's
rigging incidence::

    x = [w_upper(4), w_lower(4), tc, alpha_deg]                    10 rows

* **Why thickness is its own row** rather than whatever the weights happen to
  give: the same reason `carsection.py` gives. Left to the raw weights a large
  part of the box lands at thicknesses the `cl_max` correlation was never
  written for, and the optimiser spends its budget discovering a correlation's
  limits. So thickness is stated directly, bounded, and the weights carry the
  SHAPE. The rescaling is exact and linear in the weights
  (:func:`scaled_weights`): with camber `(w_u + w_l)/2` and half-thickness
  `(w_u - w_l)/2`, multiplying the half-thickness by `k` gives

      w_u' = (1+k)/2 w_u + (1-k)/2 w_l,   w_l' = (1-k)/2 w_u + (1+k)/2 w_l

  and since `y_u' - y_l' = k (y_u - y_l)`, a non-crossing section stays
  non-crossing for any `k > 0`.
  It costs ONE redundant direction (the overall scale of the half-thickness
  part of the weights is undone by the rescaling). Stated, because a
  degenerate direction is a real thing to know about a search space, and
  accepted, because the alternative is a refusal class that eats the box.

  THERE IS A SECOND DEGENERACY, and it was found by a self-check that expected
  a refusal and did not get one: **swapping `w_upper` and `w_lower` is a
  no-op.** `airfoil.normalise_loop` orients whatever loop it is handed, so the
  swapped weights rebuild the SAME section -- MEASURED on a NACA 2412 fit,
  t/c 0.11987 and camber +0.02009 either way, identical. So the box is
  symmetric under the swap and every design has a twin. Not repaired: the
  repair would be to refuse half the box for a shape that is perfectly
  flyable, and the GP simply sees a function with a symmetry.
* **Why the incidence is here as well as on the wing page**, over the SAME
  band. `carsection.py` keeps `alpha_deg` over `carwing_multi`'s own band
  verbatim, because the same question asked over two different bands makes the
  two stages incomparable. The section's winner SEEDS the wing page's
  incidence row; the wing page is then free to move it.

The box
-------
Not taste: the per-coefficient hull of the **34 sections carsim ships**, fitted
at `N_CST = 4` and padded by `BOX_PAD`. Every bundled section is therefore
inside the box by construction, so the search can always reproduce a shape
known to fly, and `self_check` is that statement.

It is a BOX and not the hull, so it admits corners no real section occupies --
a lower surface above the upper one, most obviously. Those are REFUSED
(:func:`check_section`) and not clamped: clamping would hand the optimiser a
different design from the one it proposed and quietly break the map it is
building.

What the score is, and what it is not
-------------------------------------
`laptime`, through a FIXED reference planform. A candidate's 2-D polar is
flown on the role's reference wing -- `wing.analyse`, the real lattice, 1.4 ms
-- and the resulting `CZ*S` / `CD*S` (or the flank panel's `k` and L/D) go to
`mission.lap`. The planform is FIXED for every candidate and is the one thing
this stage may not move.

THIS IS NOT AeroBO's BRIDGE, and the difference is in carsim's favour.
`carsection.SectionBridge` is a calibrated affine reduction with a measured
fit error, because its three-dimensional solver is too expensive to call per
candidate. Here the lattice IS 1.4 ms, so the reference wing is flown exactly
and there is no fit error to report. What survives unchanged is the
experimental split -- the section does not get to move the planform -- which
is the part that was ever load-bearing.

`weighted` is kept as the pre-registered CONTROL: an equal-weight sum of
normalised `cl_max` and `(l/d)_max`, 2-D only, never touching the lap.
`carsection.py` ships the same control for the same reason -- so that "the lap
is the better objective" can be a measurement here and not an opinion in a
docstring. Equal weights are the only choice that is not itself a claim about
the exchange rate, which is the thing being tested.

Every number on this page is an ESTIMATE unless XFOIL produced it, and
`polar.py` insists on saying so: the lift slope and zero-lift angle come from
the panel method (inviscid, exact for what it models), but `cl_max`, the drag
bucket and the stall are CORRELATIONS. The designer sees ESTIMATE next to them.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import airfoil as af
from . import mission as ms
from . import screen as scr
from .polar import Polar, estimate_polar, reynolds
from .wing import WingSpec, analyse as analyse_wing

#: CST weights per surface. 4 is `carsection.py`'s own `n_cst`, and it is a
#: DIMENSION trade, not a claim that more weights buy nothing -- MEASURED over
#: all 34 bundled sections, the least-squares fit error away from the leading
#: edge is median 2.1e-3 chord at 4 and median 7.9e-4 at 8. More weights do fit
#: better. What they cost is rows: `d = 2 n_cst + 2`, so 4 is a 10-row search
#: and 8 is an 18-row one, and AeroBO's own measured budget law
#: (`evals = 10.5 + 3.0 d` for 95 %) puts those at 40 and 65 evaluations. At a
#: section whose drag comes from a +-15 % correlation in the first place, 2e-3
#: of chord is not the error that matters.
#:
#: The three worst fits (AG35 2.9e-2, BE50 1.3e-2, NACA 23012 6.7e-3) are all
#: at the single x/c = 0 node and do NOT improve at `n_cst = 8`, so they are a
#: leading-edge resampling artefact of `split_surfaces`, not a shortage of
#: weights. Named here so the next person does not go looking for shape
#: freedom that would not fix it.
N_CST = 4

#: Thickness band, t/c. NOT `carsection.TC_BOUNDS` (0.09-0.18), which is the
#: range AeroBO's measured wide-alpha NACA bank resolves a stall over. carsim
#: has no such bank: `polar.estimate_polar` gets `cl_max` from a camber and
#: thickness CORRELATION, which has no measured anchors to run out of. The
#: band here is instead the range the LIBRARY occupies (0.043 to 0.161) pulled
#: in to where the correlation was written -- thin enough for a flank panel,
#: thick enough to bolt through.
TC_BOUNDS = (0.070, 0.180)

#: Fractional padding on the library's per-coefficient hull.
BOX_PAD = 0.15

#: The weighted CONTROL's weights, fixed before any run. Equal, for the reason
#: the module docstring gives.
WEIGHTED_REFERENCE = (0.5, 0.5)

#: Normalising constants for the weighted control, so the two terms are the
#: same size. `cl_max` of the best cambered section carsim ships, and an L/D
#: a good low-Reynolds section reaches. Frozen here; a band that moves during
#: a study is a lever, and `the-band-is-not-the-lever` measured that it is not
#: even a useful one.
WEIGHTED_BAND = dict(cl_max=2.0, ld_max=80.0)

#: What a section run may maximise. The three COMPOSITE entries are
#: `screen.OBJECTIVES` -- the screen's own weighted score and its two
#: seed-referenced variants -- and they are listed FIRST because they are the
#: ones AeroBO's procedure reaches: the library is screened on the user's
#: weights, and the search then maximises the same map the shortlist was
#: chosen on. The two physical objectives below them are carsim's own and are
#: the reason the garage exists, but they are a DIFFERENT map from the screen
#: and the page says so.
SECTION_OBJECTIVES = {
    **scr.OBJECTIVES,
    "lap time": "the mission's lap, in seconds, with this section on the "
                "role's reference planform (minimised)",
    "cl_max + L/D (2-D control)": "the pre-registered control: an equal-weight "
                                  "sum of normalised cl_max and (l/d)_max, "
                                  "2-D only -- the lap is never consulted",
}

#: Score handed back for a refused candidate. Not -inf: the GP is fitted on
#: the scores it has seen, and a -inf poisons its mean and its length scale.
#: `optimize.maximise` already replaces -inf with a penalty below the worst
#: feasible value, so -inf is what this module returns and the optimiser is
#: what turns it into a number. Stated so the two are not both "fixed".
REFUSAL_SCORE = -math.inf

#: Ceiling on a candidate's zero-lift CD before it is refused unflown. See
#: `evaluate_section` for how it is sized.
CD0_SANE = 1.0


# --------------------------------------------------------------------------- #
#  the box                                                                     #
# --------------------------------------------------------------------------- #
_BOX_CACHE: tuple | None = None


def library_weights() -> tuple:
    """`(names, W)` for every section carsim ships, `W` an `(n, 2*N_CST)`
    array of `[w_upper, w_lower]` fitted at `N_CST`."""
    names, rows = [], []
    for f in af.bundled_dat_files():
        try:
            nm, c = af.load_dat(os.path.join(af.DATA_DIR, f))
        except Exception:
            continue
        wu, wl = af.fit_cst(c, N_CST)
        names.append(nm)
        rows.append(np.concatenate([wu, wl]))
    return names, np.asarray(rows, dtype=float)


def weight_box() -> tuple:
    """`(lo, hi)`, each `(2*N_CST,)`: the library's per-coefficient hull,
    padded by `BOX_PAD` of its own width. Cached -- it is a property of the
    shipped data files and cannot change while the program runs."""
    global _BOX_CACHE
    if _BOX_CACHE is None:
        _, W = library_weights()
        lo, hi = W.min(axis=0), W.max(axis=0)
        pad = BOX_PAD * (hi - lo)
        _BOX_CACHE = (lo - pad, hi + pad)
    return _BOX_CACHE


# --------------------------------------------------------------------------- #
#  the shape                                                                   #
# --------------------------------------------------------------------------- #
def scaled_weights(w_upper, w_lower, k: float) -> tuple:
    """Multiply the half-thickness part of the weights by `k`, exactly.

    Camber is untouched. Linear in the weights, so a non-crossing section
    stays non-crossing for any `k > 0` -- see the module docstring."""
    wu = np.asarray(w_upper, dtype=float)
    wl = np.asarray(w_lower, dtype=float)
    cam = 0.5 * (wu + wl)
    half = 0.5 * (wu - wl)
    return cam + k * half, cam - k * half


def section_coords(w_upper, w_lower, tc: float, n: int = 81,
                   n_iter: int = 4) -> np.ndarray:
    """The CST loop at weights `(w_upper, w_lower)`, rescaled to thickness
    `tc`.

    The rescale is exact in the WEIGHTS but the station of maximum thickness
    can move, so a single pass lands near `tc` rather than on it. Two or three
    Newton-free passes (`k *= tc / tc_measured`) converge it to below 1e-9;
    `n_iter = 4` is well past that and costs ~0.3 ms."""
    wu = np.asarray(w_upper, dtype=float)
    wl = np.asarray(w_lower, dtype=float)
    tc = float(tc)
    k = 1.0
    c = af.cst_coords(wu, wl, n)
    for _ in range(n_iter):
        t = af.geometry(c)["tc"]
        if t <= 1e-9:
            break
        k *= tc / t
        a, b = scaled_weights(wu, wl, k)
        c = af.cst_coords(a, b, n)
    return c


def check_section(coords: np.ndarray) -> str | None:
    """Why this section may not be flown, or None.

    REFUSED, not clamped: a clamped candidate is a different design from the
    one the optimiser proposed, and the map it builds would be of a function
    it never evaluated."""
    try:
        xg, yu, yl = af.split_surfaces(coords, 201)
    except Exception as e:                                       # noqa: BLE001
        return f"the loop could not be split into surfaces ({e})"
    t = yu - yl
    if np.any(t[1:-1] <= 0.0):
        return "the lower surface crosses the upper one"
    g = af.geometry(coords)
    if not (TC_BOUNDS[0] - 1e-6 <= g["tc"] <= TC_BOUNDS[1] + 1e-6):
        return f"t/c {g['tc']:.4f} is outside {TC_BOUNDS[0]:.3f}-{TC_BOUNDS[1]:.3f}"
    if g["te_angle_deg"] > 45.0:
        return f"the trailing-edge wedge is {g['te_angle_deg']:.0f} deg"
    if not np.all(np.isfinite(coords)):
        return "the loop is not finite"
    return None


# --------------------------------------------------------------------------- #
#  the problem                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class SectionProblem:
    """What a section is judged against: a mission, a role, and ONE reference
    planform the candidate may not move."""

    role: str = "top"
    #: WHICH section is being designed. 'main' is the wing's own, and the
    #: candidate's polar is what the lattice flies. 'plate' is the END PLATE's,
    #: and the candidate reaches the lattice as `plate_polar` instead -- the
    #: wing keeps its own section and the plate stops being the flat one
    #: `vlm.Lattice` assumes by default.
    #:
    #: A PLATE HAS NO INCIDENCE ROW. The lattice's plates are vertical panels
    #: at the tips carrying only `plate_a` and `plate_L0`; there is no angle to
    #: bolt them on at. So the plate's vector is 9 rows, not 10, and the
    #: incidence it is JUDGED at is the wing's own (`inc_fixed`), which belongs
    #: to the wing page and which a plate search may not move.
    target: str = "main"
    inc_fixed: float = 0.0                     # the wing's incidence, 'plate' only
    ref: WingSpec = None                       # the FIXED reference planform
    profile: "ms.TrackProfile" = None
    base: "ms.MissionAero" = field(default_factory=lambda: ms.MissionAero())
    inc_bounds: tuple = (-2.0, 16.0)
    x: float = -1.60                           # the slot's station
    h: float = 1.30                            # the slot's height / standoff
    ride_h: float | None = None                # the image-plane gap
    V_ref: float = 40.0                        # for the section's Reynolds number
    top_mode: str = "fixed"
    #: the WING's own section polar, held while a PLATE is designed against it.
    #: Without it a plate candidate would be flown on a wing whose section was
    #: also the candidate, which is two designs at once and neither measured.
    main_polar: Polar | None = None
    objective: str = "lap time"
    mu_scale: float = 1.0
    #: THE LIFT COEFFICIENT THIS SURFACE FLIES. Every screening criterion that
    #: says "at the design cl" is read here, so it is the PROBLEM's number and
    #: never the candidate's: two sections compared at two different lifts are
    #: not compared. For a 'main' section it is the reference planform's own CL
    #: at the wing's incidence (`design_cl`); for a 'plate' it is 0, because
    #: the lattice's tip panels are vertical and carry no design load -- which
    #: is exactly why `screen.DEAD["plate"]` retires three of the criteria.
    cl_design: float = 0.0
    #: the seven criterion weights the LIBRARY SCREEN ranks on, and the three
    #: composite objectives maximise. `screen.PRESETS` by target.
    weights: dict = field(default_factory=dict)
    #: t/c minimum and |cm| ceiling. `screen.GATE_OFF` switches one off.
    gates: dict = field(default_factory=lambda: dict(scr.GATES_OFF_DEFAULT))
    #: extra minimums, all off by default (`screen.FLOORS_OFF`).
    floors: dict = field(default_factory=lambda: dict(scr.FLOORS_OFF))
    #: THE FROZEN NORMALISATION BAND, measured over the screened library by
    #: `screen.bands`. None until the screen has run, and a composite
    #: objective is REFUSED until it exists: a live min-max would move with
    #: the population, so the composite would not be a fixed function of the
    #: shape and a search could not maximise it. This is what makes "the
    #: shortlist is chosen on the map it is judged on" true rather than said.
    band: dict | None = None
    #: the SEED's own sub-scores on that band -- the reference point the goal
    #: and Tchebycheff objectives are measured against.
    seed_sub: dict | None = None
    _base_lap: float = field(default=0.0, repr=False)

    def __post_init__(self):
        #  a problem built without weights opens on its target's recommended
        #  set, so `SectionProblem(target="plate")` is already the plate's
        #  question and not the wing's asked about a plate.
        if not self.weights:
            self.weights = dict(scr.PRESETS[
                scr.recommended(self.role, self.target)])

    def baseline_lap(self) -> float:
        """The lap this car does WITHOUT the wing being designed, seconds.

        Every score on the page is quoted against it, because the absolute lap
        time of a quasi-steady point mass is not the interesting number and
        inviting anyone to compare it with a driven lap would be a mistake
        (`mission.py` is explicit that it is an optimum, not a prediction).
        The DELTA is what a design decision turns on."""
        if self._base_lap <= 0.0 and self.profile is not None:
            self._base_lap = ms.lap(self.profile, self.base, mu_scale=self.mu_scale).time
        return self._base_lap

    @property
    def has_inc(self) -> bool:
        return self.target != "plate"

    @property
    def prices_inc(self) -> bool:
        """Does THIS objective read the incidence row at all?

        A COMPOSITE RUN DOES NOT. J scores a 2-D section -- the seven
        criteria are read off the candidate's own polar and the lattice is
        never built -- so the incidence coordinate is inert: every value of
        it scores the same, the optimiser's choice of it is noise, and
        AeroBO's `objective_kwargs` sends `wing: None` on a composite run for
        exactly this reason.

        It matters because the winner's incidence is WRITTEN BACK onto the
        wing's mount angle when the section is fitted. Measured, before this
        existed: a composite search on the flank left the row at -5.933 deg,
        the bottom of its own band, and fitting bolted the wing on at that
        angle -- a design decision taken by a coordinate nothing scored.
        """
        return not scr.is_composite(self.objective)

    def labels(self, unit: bool = False) -> list:
        out = [f"w_up[{i}]" for i in range(N_CST)] + [f"w_lo[{i}]" for i in range(N_CST)]
        out.append("t/c")
        if self.has_inc:
            out.append("incidence" + (" [deg]" if unit else ""))
        return out

    def bounds(self) -> np.ndarray:
        lo, hi = weight_box()
        rows = list(zip(lo, hi)) + [TC_BOUNDS]
        if self.has_inc:
            rows.append((float(self.inc_bounds[0]), float(self.inc_bounds[1])))
        return np.array(rows, dtype=float)

    def x0(self, coords: np.ndarray, inc_deg: float) -> np.ndarray:
        """The start vector: an existing section, fitted. The optimiser is
        then seeded with a shape that already flies, which is what makes
        `f(x0)` a meaningful thing to beat."""
        wu, wl = af.fit_cst(coords, N_CST)
        tc = af.geometry(coords)["tc"]
        b = self.bounds()
        x = np.concatenate([wu, wl, [tc]] + ([[float(inc_deg)]] if self.has_inc else []))
        return np.clip(x, b[:, 0], b[:, 1])


def element_from_x(x, inc_fixed: float = 0.0) -> tuple:
    """`(w_upper, w_lower, tc, inc_deg)` out of a design vector.

    A 9-row vector is an END PLATE's -- it carries no incidence -- and
    `inc_fixed` is the wing's own angle it is judged at."""
    x = np.asarray(x, dtype=float)
    inc = float(x[2 * N_CST + 1]) if x.size > 2 * N_CST + 1 else float(inc_fixed)
    return x[:N_CST], x[N_CST:2 * N_CST], float(x[2 * N_CST]), inc


def evaluate_section(x, prob: SectionProblem) -> dict:
    """One candidate, all the way to seconds.

    Returns the shape, the 2-D polar's numbers, the reference wing's lattice
    solution, the lap, and the score. `refused` carries the reason when there
    is one -- the page prints it, because 'no feasible design found' with no
    reason is the read-out that wastes an afternoon."""
    wu, wl, tc, inc = element_from_x(x, prob.inc_fixed)
    out: dict = dict(tc=tc, inc_deg=inc, refused="", score=REFUSAL_SCORE, target=prob.target)
    try:
        coords = section_coords(wu, wl, tc)
    except Exception as e:                                       # noqa: BLE001
        out["refused"] = f"the shape could not be built ({e})"
        return out
    why = check_section(coords)
    if why:
        out["refused"] = why
        return out
    out["coords"] = coords
    out["geometry"] = af.geometry(coords)

    chord = prob.ref.chord if prob.ref is not None else 0.30
    re = reynolds(prob.V_ref, chord)
    try:
        pol = estimate_polar(coords, re, name="designed")
    except Exception as e:                                       # noqa: BLE001
        out["refused"] = f"the panel method did not solve ({e})"
        return out
    out["polar"] = pol
    out["re"] = re
    ld = np.where(pol.cd > 1e-9, pol.cl / np.maximum(pol.cd, 1e-9), 0.0)
    out["ld_max"] = float(np.max(ld)) if ld.size else 0.0
    out["cl_max"] = float(pol.cl_max)
    out["a_lin"] = float(pol.a_lin)
    out["alpha_L0_deg"] = float(pol.alpha_L0_deg)

    #  THE SEVEN SCREENING CRITERIA, always. They cost a handful of
    #  interpolations off a polar that has already been built, so they are
    #  computed whatever the objective is: that is what lets the RANKING table
    #  show where a score came from even on a run scored in seconds.
    m = scr.metrics(out, prob.cl_design)
    out["metrics"] = m
    out["gate"] = scr.gate_reason(m, prob.gates, prob.floors)
    if m is not None and prob.band:
        out["sub"] = scr.sub_scores(m, prob.band)
        out["composite"] = scr.composite(out["sub"], prob.weights)
        out["points"] = scr.points(out["sub"], prob.weights)

    #  A COMPOSITE RUN NEVER FLIES A WING. J scores a 2-D section, so the
    #  lattice and the lap would be cost with no vote -- AeroBO's
    #  `objective_kwargs` sends `wing: None` on one for exactly this reason.
    #  It is also what makes a composite search ~10x cheaper per candidate
    #  than a lap search.
    if scr.is_composite(prob.objective):
        if not prob.band:
            out["refused"] = ("this objective is normalised on a band measured "
                              "over the library -- screen the library first (L)")
            return out
        if out["gate"]:
            out["refused"] = f"the screen's gates refuse it: {out['gate']}"
            return out
        out["score"] = float(scr.score_objective(
            prob.objective, out["sub"], prob.seed_sub or {}, prob.weights))
        return out

    if prob.objective.startswith("cl_max"):
        w_cl, w_ld = WEIGHTED_REFERENCE
        out["score"] = float(w_cl * out["cl_max"] / WEIGHTED_BAND["cl_max"]
                             + w_ld * out["ld_max"] / WEIGHTED_BAND["ld_max"])
        return out

    #  the lap, through the FIXED reference planform
    if prob.ref is None or prob.profile is None:
        out["refused"] = "no reference planform or no circuit"
        return out
    #  WHERE THE CANDIDATE ENTERS THE LATTICE. As the wing's own section for
    #  'main'; as the END PLATE's for 'plate', in which case the reference
    #  wing keeps the section it already has and only the tip panels change.
    try:
        if prob.target == "plate":
            aero = analyse_wing(prob.ref, prob.main_polar or pol, V=prob.V_ref,
                                ride_h=prob.ride_h, plate_polar=pol)
        else:
            aero = analyse_wing(prob.ref, pol, V=prob.V_ref, ride_h=prob.ride_h)
    except ValueError as e:
        out["refused"] = f"the lattice refused the reference wing ({e})"
        return out
    out["aero"] = aero
    if aero.get("CLa", 0.0) <= 0.0:
        out["refused"] = "the reference wing has no lift slope"
        return out
    #  A DRAG THE MISSION CANNOT INTEGRATE. The estimate polar's drag build-up
    #  can return an enormous cd0 for a pathological shape the box admits --
    #  one candidate in a 6-case x 3-seed sweep produced cd0 = 6.6e7, which
    #  the lap then handed to `_resistance` as a drag area of 2.6e7 m^2 and
    #  overflowed float64 in the straight integrator. The lap still REFUSED
    #  it ("the lap did not close"), so the answer was never wrong -- but it
    #  was reached by overflowing, which sprays numpy warnings across the
    #  terminal of anyone running a search, and it burns a full lap
    #  integration on a design that cannot fly.
    #
    #  The threshold is sized off the library rather than picked: the 39
    #  sections carsim ships span cd0 0.0097 (mh61) to 0.0321 (s1223), and
    #  300 uniform draws over the design box reached 0.0617 at worst. 1.0 is
    #  16x above anything a shape that is still a shape can reach, and it is
    #  about what a flat plate broadside would cost on its own area.
    cd0 = float(aero.get("cd0", 0.0))
    if not math.isfinite(cd0) or cd0 > CD0_SANE:
        out["refused"] = (f"the drag build-up returned {cd0:.3g} for the zero-lift "
                          f"CD, which is not a wing")
        return out
    if inc >= aero.get("alpha_stall_deg", 90.0) - 2.0:
        out["refused"] = (f"stalled: incidence {inc:+.1f} deg is within 2 deg of "
                          f"{aero['alpha_stall_deg']:.1f}")
        return out

    m_aero = ms.merge_wing(prob.base, aero, prob.role, inc, prob.x, prob.h, prob.top_mode)
    out["mission_aero"] = m_aero
    lap = ms.lap(prob.profile, m_aero, mu_scale=prob.mu_scale)
    out["lap"] = lap
    if not lap.ok:
        out["refused"] = "; ".join(lap.notes) or "the lap did not close"
        return out
    out["score"] = -float(lap.time)          # maximise minus the lap
    out["d_lap"] = float(lap.time - prob.baseline_lap())
    return out


def f_section(x, prob: SectionProblem) -> float:
    return float(evaluate_section(x, prob)["score"])


# --------------------------------------------------------------------------- #
def reference_problem(role: str = "top", track: str = "arena",
                      objective: str = "lap time",
                      target: str = "main") -> SectionProblem:
    """The problem `self_check` runs and the garage seeds a fresh page with:
    the role's own library wing as the fixed reference planform, on `track`,
    with nothing else on the car. Importing `drive.track` is deferred to here
    so the module itself stays free of the simulator."""
    from .. import track as _tr
    from .wing import BOUNDS, V_REF, RIDE_H0
    if role == "flank":
        ref = WingSpec(role="flank", airfoil="designed", span=0.80, chord=0.45,
                       taper=0.85, plate_h=0.06, ride_h=RIDE_H0["flank"])
        x, h, ride = 0.97, 0.90, RIDE_H0["flank"]
    else:
        ref = WingSpec(role="top", airfoil="designed", span=1.40, chord=0.30,
                       taper=0.85, plate_h=0.06, ride_h=RIDE_H0["top"])
        x, h, ride = -1.60, RIDE_H0["top"], RIDE_H0["top"]
    main_polar = None
    if target == "plate":
        #  a plate is designed AGAINST a wing, so the wing needs a section of
        #  its own that is held while the plate moves.
        from .polar import estimate_polar as _est
        ref.plate_h = max(ref.plate_h, 0.08)
        main_polar = _est(af.naca4_coords("2412"), reynolds(V_REF[role], ref.chord))
    return SectionProblem(role=role, ref=ref, target=target, main_polar=main_polar,
                          profile=ms.TrackProfile.from_track(_tr.make_track(track)),
                          inc_bounds=BOUNDS[role]["inc_deg"], x=x, h=h, ride_h=ride,
                          V_ref=V_REF[role], objective=objective)


def self_check(verbose: bool = True) -> bool:
    import time as _t
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and passed
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}{('  ' + msg) if msg else ''}")

    #  1. the box contains the library it was built from
    names, Wm = library_weights()
    lo, hi = weight_box()
    inside = ((Wm >= lo - 1e-12) & (Wm <= hi + 1e-12)).all(axis=1)
    rep("every bundled section is inside the box", bool(inside.all()),
        f"{int(inside.sum())}/{len(names)}")
    rep("the box is not degenerate", bool(np.all(hi > lo)))

    #  2. the CST fit, at the order this module ships
    errs = []
    for f in af.bundled_dat_files():
        try:
            _, c = af.load_dat(os.path.join(af.DATA_DIR, f))
        except Exception:
            continue
        wu, wl = af.fit_cst(c, N_CST)
        c2 = af.cst_coords(wu, wl, 81)
        xg, yu, yl = af.split_surfaces(c, 201)
        _, yu2, yl2 = af.split_surfaces(c2, 201)
        m = xg > 0.005
        errs.append(max(np.abs(yu - yu2)[m].max(), np.abs(yl - yl2)[m].max()))
    med = float(np.median(errs))
    rep(f"CST({N_CST}) reproduces the library away from the LE", med < 4e-3,
        f"median {med:.1e} chord, p90 {np.percentile(errs, 90):.1e}, worst {max(errs):.1e}")

    #  3. the thickness rescale is exact, and preserves camber
    c = af.naca4_coords("2412")
    wu, wl = af.fit_cst(c, N_CST)
    cam0 = af.geometry(c)["camber"]
    worst_t, worst_c = 0.0, 0.0
    for tc in (0.07, 0.10, 0.12, 0.15, 0.18):
        cc = section_coords(wu, wl, tc)
        g = af.geometry(cc)
        worst_t = max(worst_t, abs(g["tc"] - tc))
        worst_c = max(worst_c, abs(g["camber"] - cam0))
    rep("the thickness row lands exactly", worst_t < 1e-9, f"worst |d t/c| {worst_t:.1e}")
    rep("and does not move the camber", worst_c < 5e-4, f"worst |d camber| {worst_c:.1e}")

    #  4. `scaled_weights` is the identity at k = 1 and linear in k
    a1, b1 = scaled_weights(wu, wl, 1.0)
    rep("scaled_weights is the identity at k = 1",
        float(np.max(np.abs(a1 - wu))) < 1e-15 and float(np.max(np.abs(b1 - wl))) < 1e-15)
    a2, b2 = scaled_weights(wu, wl, 2.0)
    rep("and doubles the half-thickness exactly",
        float(np.max(np.abs((a2 - b2) - 2.0 * (wu - wl)))) < 1e-14)

    #  5. refusals refuse. NOT a global surface swap -- `normalise_loop`
    #     orients the loop, so that is a no-op and the docstring says so. The
    #     case that can actually happen is a PARTIAL crossing: a lower surface
    #     that rises above the upper one somewhere short of the trailing edge
    #     while the maximum thickness is still positive.
    swapped = evaluate_section(np.concatenate([wl, wu, [0.12], [4.0]]), reference_problem("top"))
    straight = evaluate_section(np.concatenate([wu, wl, [0.12], [4.0]]), reference_problem("top"))
    rep("swapping the two surfaces is a no-op, as the docstring claims",
        not swapped["refused"] and abs(swapped["score"] - straight["score"]) < 1e-9)
    wl_x = np.array(wl, dtype=float).copy()
    wl_x[-1] = 0.30
    r = evaluate_section(np.concatenate([wu, wl_x, [0.12], [4.0]]), reference_problem("top"))
    rep("a section whose surfaces cross is refused", bool(r["refused"]), r["refused"][:60])
    thin = np.concatenate([wu, wl, [TC_BOUNDS[0] - 0.02], [4.0]])
    rep("a section outside the thickness band is refused",
        bool(evaluate_section(thin, reference_problem("top"))["refused"]))

    #  6. a real candidate, both roles, both objectives, and what it costs
    for role in ("top", "flank"):
        prob = reference_problem(role)
        x0 = prob.x0(c, 4.0)
        _ = evaluate_section(x0, prob)                      # warm
        t0 = _t.perf_counter()
        r = evaluate_section(x0, prob)
        dt = _t.perf_counter() - t0
        rep(f"{role}: a NACA 2412 seed flies", not r["refused"] and "lap" in r,
            f"lap {r.get('lap').time:.4f} s ({r.get('d_lap', 0.0):+.4f} vs bare), "
            f"cl_max {r['cl_max']:.3f}, L/D {r['ld_max']:.1f}, {dt * 1e3:.0f} ms")
        rep(f"{role}: a candidate costs a search under 60 ms", dt < 0.060,
            f"{dt * 1e3:.0f} ms")
        pc = reference_problem(role, objective="cl_max + L/D (2-D control)")
        rc = evaluate_section(pc.x0(c, 4.0), pc)
        rep(f"{role}: the 2-D control scores without a lap",
            not rc["refused"] and "lap" not in rc and math.isfinite(rc["score"]),
            f"score {rc['score']:.4f}")

    #  6b. the END PLATE target: 9 rows, no incidence, and it reaches the
    #      lattice as `plate_polar` while the wing keeps its own section.
    pp = reference_problem("flank", target="plate")
    pp.inc_fixed = 6.0
    rep("a plate's vector is 9 rows, not 10", pp.bounds().shape[0] == 2 * N_CST + 1
        and len(pp.labels()) == 2 * N_CST + 1, f"{pp.bounds().shape[0]} rows")
    flat = evaluate_section(pp.x0(af.naca4_coords("0012"), 0.0), pp)
    camb = evaluate_section(pp.x0(af.naca4_coords("6412"), 0.0), pp)
    rep("a plate section flies and a CAMBERED plate is not the same wing",
        not flat["refused"] and not camb["refused"]
        and abs(camb["aero"]["CL0"] - flat["aero"]["CL0"]) > 1e-4,
        f"symmetric plate CL0 {flat['aero']['CL0']:.4f} -> cambered {camb['aero']['CL0']:.4f} "
        f"({100 * (camb['aero']['CL0'] / flat['aero']['CL0'] - 1):+.1f} %)")
    #  the candidate entered as the PLATE and not as the wing: the same two
    #  shapes flown as the wing's own section move CL0 somewhere else entirely.
    mp = reference_problem("flank", target="main")
    mp.ref = pp.ref
    m_flat = evaluate_section(mp.x0(af.naca4_coords("0012"), 6.0), mp)
    m_camb = evaluate_section(mp.x0(af.naca4_coords("6412"), 6.0), mp)
    rep("the wing's own section is HELD while the plate moves",
        pp.main_polar is not None
        and abs((camb["aero"]["CL0"] - flat["aero"]["CL0"])
                - (m_camb["aero"]["CL0"] - m_flat["aero"]["CL0"])) > 1e-3,
        f"as a PLATE dCL0 {camb['aero']['CL0'] - flat['aero']['CL0']:+.4f}, "
        f"as the WING's section {m_camb['aero']['CL0'] - m_flat['aero']['CL0']:+.4f}")

    #  7. determinism -- the optimiser reads this hundreds of times
    prob = reference_problem("top")
    x0 = prob.x0(c, 4.0)
    rep("the score is deterministic",
        evaluate_section(x0, prob)["score"] == evaluate_section(x0, prob)["score"])

    #  8. the start vector round-trips: x0 of a section reproduces that section
    r0 = evaluate_section(x0, prob)
    g0, g1 = af.geometry(c), r0["geometry"]
    rep("x0 round-trips the section it was fitted to",
        abs(g0["tc"] - g1["tc"]) < 1e-6 and abs(g0["camber"] - g1["camber"]) < 1e-3,
        f"t/c {g0['tc']:.4f} -> {g1['tc']:.4f}, camber {g0['camber']:+.4f} -> {g1['camber']:+.4f}")

    #  8b. a candidate whose drag build-up blew up is refused UNFLOWN, so the
    #      lap integrator is never handed a drag area it cannot integrate
    r_ok = evaluate_section(np.asarray(x0, dtype=float), prob)
    rep("a sane section is not caught by the drag guard",
        not r_ok.get("refused") and r_ok["aero"]["cd0"] < CD0_SANE,
        f"cd0 {r_ok['aero']['cd0']:.4f} against a ceiling of {CD0_SANE}")

    #  9. THE SCREEN. The weights decide the shortlist, the shortlist's own
    #     spread is the band, and the band is what makes the composite a fixed
    #     function of the shape.
    prob.cl_design = 0.9
    prob.objective = "composite"
    rep("a composite run is refused until the library has been screened",
        bool(evaluate_section(x0, prob)["refused"]),
        evaluate_section(x0, prob)["refused"][:60])

    names, rows, metr = [], [], []
    b = prob.bounds()
    for f in af.bundled_dat_files()[:18]:
        try:
            nm, coords = af.load_dat(os.path.join(af.DATA_DIR, f))
        except Exception:                                        # noqa: BLE001
            continue
        wu, wl = af.fit_cst(coords, N_CST)
        xr = np.clip(np.concatenate([wu, wl, [af.geometry(coords)["tc"], 4.0]]),
                     b[:, 0], b[:, 1])
        r = evaluate_section(xr, prob)
        if r.get("metrics") is None or r.get("gate"):
            continue
        names.append(nm); rows.append((xr, r)); metr.append(r["metrics"])
    rep("the screen keeps a population to measure a band over", len(metr) >= 6,
        f"{len(metr)} eligible of 18 read")
    prob.band = scr.bands(metr)
    rep("the band covers every criterion the library separates",
        set(prob.band) >= {"thick", "clmax", "ldmax", "ldcr", "cdcr"},
        ", ".join(sorted(prob.band)))

    scored = []
    for nm, (xr, _) in zip(names, rows):
        r = evaluate_section(xr, prob)
        scored.append((nm, r["score"], r))
    scored.sort(key=lambda t: -t[1])
    rep("the weights rank the library", math.isfinite(scored[0][1]),
        f"'{scored[0][0]}' {scored[0][1]:.2f} points, "
        f"'{scored[-1][0]}' {scored[-1][1]:.2f}")
    #  the points a criterion contributed sum to the score it was ranked on
    top = scored[0][2]
    rep("the ranking table's points add up to the score",
        abs(sum(top["points"].values()) - top["composite"]) < 1e-9)

    #  THE WEIGHTS ACTUALLY DECIDE, and the strong form of that: a screen run
    #  on ONE criterion must return that criterion's own argmax over the
    #  population. Two presets can agree by accident -- E423 wins both shipped
    #  sets, because it is both thick and high-lift -- so the check is made on
    #  single-criterion weight sets, where there is nothing to agree about.
    picks = {}
    for crit in ("thick", "clmax", "cdcr"):
        prob.weights = {crit: 1.0}
        picks[crit] = max(((nm, evaluate_section(xr, prob)["score"])
                           for nm, (xr, _) in zip(names, rows)),
                          key=lambda t: t[1])[0]
    argmax = {c: (max if c not in scr.LOWER_BETTER else min)(
        zip(names, (m[c] for m in metr)), key=lambda t: t[1])[0]
        for c in ("thick", "clmax", "cdcr")}
    rep("a screen on one criterion returns that criterion's own best section",
        picks == argmax,
        "  ".join(f"{c}: '{picks[c]}'" for c in picks))
    rep("and three criteria do not agree on one section",
        len(set(picks.values())) >= 2)
    prob.weights = dict(scr.PRESETS[scr.recommended(prob.role, "main")])

    #  a composite candidate never flies the lattice, so it is far cheaper
    prob.seed_sub = top["sub"]
    t0 = time.perf_counter()
    for _ in range(5):
        evaluate_section(x0, prob)
    t_comp = (time.perf_counter() - t0) / 5.0 * 1e3
    prob.objective = "lap time"
    t0 = time.perf_counter()
    for _ in range(5):
        evaluate_section(x0, prob)
    t_lap = (time.perf_counter() - t0) / 5.0 * 1e3
    rep("a composite candidate is cheaper than a lap candidate", t_comp < t_lap,
        f"{t_comp:.1f} ms vs {t_lap:.1f} ms")

    #  the goal objective refuses a trade the plain composite calls even
    prob.objective = "composite"
    j_seed = evaluate_section(x0, prob)["score"]
    prob.seed_sub = evaluate_section(x0, prob)["sub"]
    prob.objective = "composite, none below the seed"
    rep("at the seed the goal composite IS the composite",
        abs(evaluate_section(x0, prob)["score"] - j_seed) < 1e-9)
    prob.objective = "lift the weakest criterion"
    rep("and the Tchebycheff function is zero there",
        abs(evaluate_section(x0, prob)["score"]) < 1e-9)

    return ok


if __name__ == "__main__":
    import sys
    print("drive.aero.section self-check")
    sys.exit(0 if self_check() else 1)
