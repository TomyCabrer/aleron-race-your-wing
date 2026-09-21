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

`DESIGN_VARS` is the one ordered list of what a designer (human or the GP)
may move, in the order the garage's DESIGNER page shows the rows. BOUNDS,
`design_bounds`, `design_x0`, `design_labels` and `apply_design` all read it,
so the optimiser's design vector is the page's row order by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

from . import blend as bl
from .polar import Polar, RHO, NU, reynolds, friction_coefficient
from .vlm import Lattice, TWO_PI

ROLES = ("flank", "top")

#: The design variables, in the ONE order the garage's DESIGNER page lists
#: them (CONTRACT section 7: "section, span, chord, taper, twist, end plates,
#: the slot's mount"). Everything ordered downstream is read off this table --
#: the packaging bands, the optimiser's bounds, its start vector, its labels
#: and its decode -- so the five cannot drift apart the way five hand-written
#: lists can, and a reordered page row moves the vector with it.
#:
#: The page's first row, the section, is a discrete library choice and so is
#: not a vector coordinate: the optimiser designs the planform for whatever
#: section the page is showing. `owner` says where the number lives: 'spec' is
#: a WingSpec field, 'slot' is the mount angle, which belongs to the CarBuild
#: slot and not to the wing (one library wing can be bolted on at any angle).
#: Bands are (lo, hi) in m / deg, per role.
DESIGN_VARS = (
    #  attr             owner   label          unit   flank band      top band
    ("taper",          "spec", "taper",        "",    (0.35, 1.0),   (0.35, 1.0)),
    ("twist_root_deg", "spec", "root twist",   "deg", (-6.0, 6.0),   (-6.0, 6.0)),
    ("twist_deg",      "spec", "tip twist",    "deg", (-6.0, 6.0),   (-6.0, 6.0)),
    ("inc_deg",        "slot", "incidence",    "deg", (-6.0, 14.0),  (-2.0, 16.0)),
    ("plate_h",        "spec", "end plates",   "m",   (0.0, 0.16),   (0.0, 0.30)),
    #  the top's band is the garage's own TOP_H_MAX; the flank's spans a panel
    #  hugging the sill (DEV_OUT0) to one well clear of it
    ("ride_h",         "spec", "standoff",     "m",   (0.25, 0.70),  (0.90, 1.85)),
    #  the flank's ceiling is what the sill-to-rail height can take (see
    #  `span_fit`, which caps the band at the slot height); the top's is
    #  AeroBO's `SPAN_BOUNDS_M` ceiling, past the 1.646 m body width -- the
    #  packaging band is the widest a designer may ASK for, and a regulation
    #  or a transporter narrows it on the DESIGN BOX
    ("span",           "spec", "span",         "m",   (0.35, 1.30),  (0.70, 2.00)),
)

#: The AREA row, present only when the caller declares an area band -- the
#: `(S_m2,)` of AeroBO's `CarWingProblem` vector, and in its place: ONE ROW
#: AHEAD OF THE SPAN. With it absent the reference area is FIXED and the span
#: row IS the aspect ratio, which is the convention every coefficient in this
#: module is quoted against; with it present the score has to be read in
#: FORCES, because two candidates no longer share a reference.
AREA_VAR = ("area", "spec", "reference area", "m2", (0.15, 0.80), (0.10, 1.00))


def design_table(role: str = "flank", area: bool = False) -> tuple:
    """The rows a designer may move, in AeroBO's `evaluate_car_wing` order:

        taper, root twist, tip twist, incidence, end plates, standoff,
        (reference area,) span

    ROOT CHORD IS NOT A ROW, and that is the procedural change, not an
    omission. AeroBO sizes a wing by its AREA and its SPAN and lets the chord
    fall out of the two (`chord = 2S / (b (1 + taper))`); carrying a free root
    chord AND a free span alongside a reference area would let a candidate be
    scored against an area it does not have.  `apply_design` derives it.
    """
    rows = DESIGN_VARS
    if area:
        i = next(k for k, v in enumerate(rows) if v[0] == "span")
        rows = rows[:i] + (AREA_VAR,) + rows[i:]
    return rows

# packaging bands the designer / optimiser stay inside (m, deg). Keyed role ->
# variable; the inner dict's order IS the page order, by construction.
BOUNDS = {role: {v[0]: v[4 + i] for v in DESIGN_VARS + (AREA_VAR,)}
          for i, role in enumerate(ROLES)}
#  How the wing is carried into the body. This is a REAL aerodynamic choice,
#  not a label: each option changes the lattice the wing is solved on and/or
#  which terms of the drag build-up are charged.
#
#  'pylon'     two pylons standing the wing off the body by `standoff`.
#              Charges `strut_cd` -- twice their wetted area on S_ref at the
#              local skin-friction coefficient, times a 1.3 form factor which
#              is the allowance for the pylon/wing junction interference
#              (Hoerner, Fluid-Dynamic Drag, ch.8: a faired strut junction is
#              20-40% above flat-plate friction). Tip plates, if any, stay an
#              independent choice. THIS IS THE DEFAULT: every wing in the
#              library was analysed with it, and the whole acceptance suite is
#              measured on it.
#  'endplate'  the wing is carried by structural plates at its tips instead:
#              no pylons in the flow, so `strut_cd` is NOT charged, but the
#              plates are forced to at least MOUNT_PLATE_H so there is metal
#              to bolt through. The lift consequence is not a correlation --
#              the plates are real panels in the lattice, so the reduced tip
#              loss comes straight out of the VLM (self_check measures
#              CLa 2.179 -> 2.646 on the flank panel). Their wetted area is
#              charged by the existing `cd_pl` term, and they are heavier
#              than two pylons.
#  'none'      nothing is charged for the mount. Not a buildable car: it is
#              the idealisation to compare against, and the parity setting for
#              a legacy panel whose published L/D already includes its mounts.
MOUNTS = ("pylon", "endplate", "none")

#: ...and the two a DESIGNER may choose between. AeroBO's `carwing.MOUNTS`
#: offers two layouts and both are real ways to hold a wing up; 'none' is not
#: one. Its own docstring above says so -- "Not a buildable car: it is the
#: idealisation to compare against, and the parity setting for a legacy panel
#: whose published L/D already includes its mounts". It stays in `MOUNTS`,
#: because a stored legacy panel is decoded with it and `clamp` must still
#: accept it; it is simply not a thing the design form offers.
MOUNTS_BUILDABLE = ("pylon", "endplate")

#: Structural minimum tip-plate height for an endplate MOUNT, m. est: the
#: plate has to carry the whole wing load in bending into two body hardpoints,
#: so it needs a flange deep enough for two fasteners plus edge distance --
#: ~3x a 6 mm bolt's edge distance each side on the flank panel, more on the
#: top wing because its load is 4-5x larger (594 N at 40 m/s, self_check).
MOUNT_PLATE_H = {"flank": 0.06, "top": 0.12}

#: Areal density of the skin the mass estimate charges, kg/m^2. 1.5 mm 2024-T3
#: sheet: 2780 kg/m^3 x 0.0015 m = 4.17. Both the wing's two surfaces and the
#: plates are charged at it; the pylons are charged as solid 6 mm x chord bar.
SKIN_KG_M2 = 4.17
PYLON_KG_M = 1.6             # est: 6 mm x 100 mm 2024-T3 bar, 2780 kg/m^3

V_REF = {"flank": 29.0875, "top": 40.0}      # m/s: R = 100 m limit speed / a fast straight
#: Default distance to the imaged wall, m -- what `ride_h = 0.0` resolves to.
#: The flank's is the DEPLOYED standoff the car actually reaches, which is the
#: garage's and the renderer's DEV_OUT0 + DEV_OUT1 = 0.25 + 0.35; the top's is
#: the garage's default top slot height.
#:
#: These two disagreed before the ride row existed. `FLANK_STANDOFF` was 0.45
#: and was this module's default, while `garage._analyse` passed 0.60 for every
#: wing it designed -- so a panel's published lattice numbers depended on which
#: door it came through. A design ROW cannot carry two defaults, so they are
#: one number now, and it is the one the car deploys to.
#: The top's is the height `library.analyse_wing` fell back to before the row
#: existed, so a top wing with nothing stored is flown exactly where it was.
RIDE_H0 = {"flank": 0.60, "top": 1.30}
RE_BANK = (1e5, 1.5e5, 2e5, 3e5, 5e5, 7e5, 1e6, 1.5e6, 2e6, 3e6)


def re_bank_snap(re: float) -> float:
    lr = math.log(max(re, 1.0))
    return min(RE_BANK, key=lambda r: abs(math.log(r) - lr))


# --------------------------------------------------------------------------- #
#  the design vector: ONE order, the page's                                    #
# --------------------------------------------------------------------------- #
#  Four things used to be written out by hand and had to agree coordinate for
#  coordinate: the bounds list, the start vector, the decode inside the
#  objective and the write-back after the run. They did agree -- measured, see
#  .handoff/task2-optimiser-order.md -- but nothing held them to it. These
#  five functions are that hold: all of them walk DESIGN_VARS, so the only way
#  to reorder the optimiser is to reorder the page's table.
def design_vars(role: str = "flank", owner: str | None = None,
                area: bool = False) -> tuple[str, ...]:
    """The design vector's coordinate names, in the page's row order.
    `owner='spec'` gives just the WingSpec fields, `'slot'` the mount angle."""
    return tuple(v[0] for v in design_table(role, area)
                 if owner is None or v[1] == owner)


def design_labels(role: str = "flank", unit: bool = False,
                  area: bool = False) -> list[str]:
    """The page's own row labels, in the page's order. The flank panel stands
    on its side, so the page calls its span 'span (vertical)' -- the label
    lives here so a read-out and a row cannot end up naming a coordinate
    differently."""
    out = []
    for attr, _owner, label, u, *_bands in design_table(role, area):
        if attr == "span" and role != "top":
            label = "span (vertical)"
        if attr == "ride_h" and role == "top":
            label = "ride height"          # the track is the wall up there
        out.append(f"{label} [{u}]" if unit and u else label)
    return out


def design_bounds(role: str = "flank", area: bool = False, bands=None,
                  **override) -> list[tuple[float, float]]:
    """`[(lo, hi), ...]` for `optimize.maximise`, in the page's order.

    Keyword overrides narrow one band without touching the order -- the garage
    passes `span=` to keep a flank panel between sill and roof. An unknown
    keyword raises rather than being silently dropped: a typo there would
    quietly hand the optimiser the full packaging band.

    `bands` is the same thing as a dict, and it exists because ONE design
    variable cannot be spelled as a keyword: the AREA row is called `area` and
    so is the switch that decides whether it is in the vector at all. A caller
    that overrides every row -- which is what an editable design box is --
    therefore passes a dict. `override` is merged over it, so the garage's
    `span=` idiom still reads the same.
    """
    names = design_vars(role, area=area)
    over = dict(bands or {})
    over.update(override)
    bad = set(over) - set(names)
    if bad:
        raise ValueError(f"not design variables: {sorted(bad)}")
    override = over
    b = BOUNDS[role if role in BOUNDS else "flank"]
    out = []
    for attr in names:
        lo, hi = override.get(attr, b[attr])
        out.append((float(lo), float(hi)))
    return out


def design_x0(spec: "WingSpec", inc_deg: float, area: bool = False) -> list[float]:
    """The design vector of a wing as it stands, in the page's order: the
    optimiser's starting point and the value its result is judged against."""
    #  the RESOLVED values, not the raw fields: `ride_h` and `area` carry 0.0
    #  for "take the role's default", and a start vector holding the sentinel
    #  would not survive its own decode (apply_design resolves it), which is
    #  exactly the round-trip the self-check pins.
    def val(attr):
        if attr == "ride_h":
            return spec.ride_h_flown
        if attr == "area":
            return spec.area if spec.area > 0.0 else spec.S
        return getattr(spec, attr)
    return [float(inc_deg) if v[1] == "slot" else float(val(v[0]))
            for v in design_table(spec.role, area)]


def apply_design(spec: "WingSpec", x, *, clamp: bool = True,
                 area: bool = False) -> float:
    """Decode a design vector onto `spec` in the page's order and return the
    mount angle for the slot to take. The exact inverse of `design_x0`, which
    is the point: the objective's decode and the write-back after the run are
    now the same three lines of code, not two hand-kept lists."""
    xa = np.asarray(x, float).ravel()
    #  RESOLVE THE AREA SENTINEL FIRST, and before anything overwrites the
    #  planform it is derived from. `spec.area` carries 0.0 for "take the
    #  planform's own" (`clamp` resolves it, `design_x0` resolves it) -- and
    #  this function, which is documented as `design_x0`'s exact inverse, did
    #  not. With no area row the chord is then `2 x 0 / ...` = 0 and the
    #  lattice refuses every candidate with "chord collapses to zero
    #  somewhere on the span", which is a true sentence about a wing nobody
    #  proposed.
    #
    #  It only bites on a spec that has never been clamped since it was
    #  loaded -- a library wing opened and searched without a single row
    #  being edited first -- which is why it survived: every path that
    #  touched a design row called `clamp` on the way past.
    if not area and spec.area <= 0.0:
        spec.area = spec.span * spec.chord * 0.5 * (1.0 + spec.taper)
    rows = design_table(spec.role, area)
    if xa.size != len(rows):
        raise ValueError(f"design vector has {xa.size} coordinates, "
                         f"expected {len(rows)} "
                         f"({', '.join(design_vars(spec.role, area=area))})")
    inc = 0.0
    for v, xi in zip(rows, xa):
        if v[1] == "spec":
            setattr(spec, v[0], float(xi))
        else:
            inc = float(xi)
    #  the chord FOLLOWS the area and the span, it is not carried alongside
    #  them: S = b c (1 + taper) / 2  ->  c = 2S / (b (1 + taper)). With no
    #  area row `spec.area` is the area the wing already had, so a decode that
    #  moves nothing writes the chord back unchanged.
    spec.chord = 2.0 * spec.area / max(spec.span * (1.0 + spec.taper), 1e-9)
    if clamp:
        spec.clamp()
    return inc


#: The flank panel is a VERTICAL extent centred on its mount, so how tall it
#: may be depends on where it is bolted: it has to clear the sill below and
#: stay under the roof rail above. 0.28 m is the sill face the garage's body
#: mesh puts the rocker at, 1.34 m the rail just under the 1.440 m published
#: roof. They lived as two bare literals inside the optimiser's bounds and
#: nowhere else, which is why the DESIGNER's span row did not know about them.
SILL_Z, ROOF_Z = 0.28, 1.34


def span_fit(role: str, h: float) -> float:
    """The tallest span `role` can actually be packaged at, mounted at height
    `h` -- the upper bound BOTH the page's span row and the optimiser's span
    band should use, so the page cannot offer a panel the optimiser is
    forbidden to propose. A top wing is limited by the car's width, not by
    its ride height, so its band is unconditional."""
    lo, hi = BOUNDS[role if role in BOUNDS else "flank"]["span"]
    if role != "flank":
        return hi
    #  the floor keeps the band non-degenerate: a mount right under the rail
    #  would otherwise collapse it to nothing and leave the GP no room at all
    return max(min(hi, 2.0 * min(h - SILL_Z, ROOF_Z - h)), lo + 0.05)


def format_design(x, role: str = "flank", area: bool = False) -> str:
    """One line naming every coordinate, in the page's order, for a read-out
    or a log. Degrees carry a sign because the sign is the design decision
    (washout vs wash-in, nose-down vs nose-up)."""
    xa = np.asarray(x, float).ravel()
    bits = []
    for v, lab, xi in zip(design_table(role, area),
                          design_labels(role, area=area), xa):
        u = v[3]
        bits.append(f"{lab} {xi:+.1f}{' ' + u}" if u == "deg" else
                    f"{lab} {xi:.3f}" + (f" {u}" if u else ""))
    return "   ".join(bits)


@dataclass
class WingSpec:
    name: str = "new wing"
    role: str = "flank"
    airfoil: str = "naca2412"
    span: float = 0.78
    chord: float = 0.45
    taper: float = 1.0
    #: AeroBO carries TWO twist rows. The lattice's twist law is linear
    #: between them, so the old single-row wing is exactly twist_root_deg = 0.
    twist_root_deg: float = 0.0
    twist_deg: float = 0.0            # the TIP twist
    plate_h: float = 0.0
    #: The END PLATE's own section, by library name. "" = a FLAT PLATE, which
    #: is what `vlm.Lattice` has always defaulted to ("the plate's own section
    #: (a flat plate by default)") and what every wing in the library was
    #: analysed with -- so an empty string reproduces the shipped numbers to
    #: the bit, and `self_check` pins that.
    #:
    #: It is a real aerodynamic row, not a label. The plate is a lifting panel
    #: in the lattice, and giving it CAMBER points its own side force the same
    #: way the wing's is pointed. MEASURED on `flank-e423` at plate_h 0.12 m,
    #: through `analyse`: a plate whose zero-lift angle is -8 deg takes
    #: CL0 0.5688 -> 0.6199 (+9.0 %) and CL_max 1.958 -> 2.058 (+5.1 %).
    #: The plate's SLOPE barely enters (a_lin 6.28 -> 6.10 moves CL0 by 0.004);
    #: its ZERO-LIFT ANGLE is the whole effect, which is why the endplate's
    #: design problem is a camber problem.
    #:
    #: NOT the plate's cant: that is a problem setting in AeroBO too
    #: (`14-urop-parity.md` section 2), and it stays one here -- carsim's
    #: plates stand normal to the wing. The BLEND is a setting as well, and
    #: it is the next two rows.
    plate_airfoil: str = ""
    #: HOW THE WING AND THE PLATE MEET: the fraction of the plate's arc spent
    #: turning out of the wing plane, in [0, 1]. 0 is the right-angle corner
    #: every wing in the library was analysed with, where the surface changes
    #: from wing to plate -- its section, its twist and its direction -- in
    #: ONE STEP at the junction panel. Raise it and the plate leaves the wing
    #: tangentially and becomes itself over `plate_blend * plate_h` of arc.
    #: See `blend`; the geometry is AeroBO's, to the bit.
    #:
    #: What it trades: TIP HEIGHT for OUTBOARD REACH (a blended plate is the
    #: same arc, so it does not climb as far and it leans out -- and the wing
    #: pays for that reach out of its own span, `span_flown`) plus a corner
    #: the lattice is no longer asked to draw as a crease.
    plate_blend: float = 0.0
    #: which turn law draws the corner -- see `blend.BLEND_SHAPES`. 'arc' is
    #: the circular fillet (and the legacy value); the two G2 laws meet the
    #: wing with continuous curvature instead.
    plate_shape: str = "arc"
    #: Charge Hoerner's junction interference at the two wing/plate corners?
    #: OFF by default, and deliberately: the lattice cannot see a corner's
    #: own drag, so this is a reduced-order ADD-ON with a calibrated fillet
    #: credit in it (`blend`), and switching it on for a wing analysed without
    #: it would move a published number. It follows the blend row in the
    #: garage -- raising the blend off zero turns it on -- because the credit
    #: for blending IS the reason to blend, and a blend priced without it is
    #: all cost and no benefit.
    plate_junction: bool = False
    #: Distance to the wall the wing is imaged in: the car's flank for a flank
    #: panel (its standoff), the track for a top wing (its ride height). It is
    #: AeroBO's `ride_height_m` row -- the SAME quantity in the same role, the
    #: gap ground effect is a function of -- and not the slot's mount height,
    #: which is a packaging number the image plane never sees. 0.0 = take the
    #: role's default, which is what every wing saved before this row did.
    ride_h: float = 0.0
    #: The reference area coefficients are quoted against. 0.0 = take the
    #: planform's own b c (1 + taper) / 2, so a wing saved before this row
    #: decodes to exactly the area it was analysed at.
    area: float = 0.0
    #: how the wing is attached to the body -- see MOUNTS. 'pylon' is the
    #: default because it is what every wing in the library was analysed with.
    mount: str = "pylon"
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

    @property
    def ride_h_flown(self) -> float:
        """The imaged wall's distance, with 0.0 resolved to the role default."""
        return self.ride_h if self.ride_h > 0.0 else RIDE_H0[self.role]

    @property
    def plate_h_flown(self) -> float:
        """The tip-plate height the LATTICE sees: an endplate mount forces a
        structural minimum, because that is what the wing hangs from."""
        if self.mount == "endplate":
            return max(self.plate_h, MOUNT_PLATE_H.get(self.role, 0.06))
        return self.plate_h

    def clamp(self) -> "WingSpec":
        if self.role not in ROLES:
            self.role = "flank"
        if self.mount not in MOUNTS:
            self.mount = "pylon"
        if self.plate_shape not in bl.BLEND_SHAPES:
            self.plate_shape = "arc"
        self.plate_blend = float(min(max(self.plate_blend, 0.0), 1.0))
        if self.area <= 0.0:
            self.area = self.span * self.chord * 0.5 * (1.0 + self.taper)
        if self.ride_h <= 0.0:
            self.ride_h = RIDE_H0[self.role]
        b = BOUNDS[self.role]
        for k in design_vars(self.role, owner="spec", area=True):  # page order
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
                   twist_root_deg=float(d.get("twist_root_deg", 0.0)),
                   twist_deg=float(d.get("twist_deg", 0.0)), plate_h=float(d.get("plate_h", 0.0)),
                   #  absent on every wing saved before the row: "" is the flat
                   #  plate the lattice has always assumed, so they decode to
                   #  exactly the numbers they were analysed at.
                   plate_airfoil=str(d.get("plate_airfoil", "")),
                   #  absent on every wing saved before the transition existed,
                   #  and 0.0 / 'arc' / False IS the right-angle corner they
                   #  were analysed with -- see `blend`
                   plate_blend=float(d.get("plate_blend", 0.0)),
                   plate_shape=str(d.get("plate_shape", "arc")),
                   plate_junction=bool(d.get("plate_junction", False)),
                   #  A wing saved before the ride row existed recorded the
                   #  height it was flown at under `aero['ride_h']`. Seed the
                   #  row from it, or `clamp` fills the role default instead
                   #  and the wing is silently re-flown somewhere else: the
                   #  library's own `rear-new` sits at 1.85 m, and the default
                   #  would have moved it to 1.30 and handed it 1.05 % of CLa
                   #  it had not earned.
                   ride_h=float(d.get("ride_h") or (d.get("aero") or {}).get("ride_h") or 0.0),
                   area=float(d.get("area", 0.0)),
                   mount=str(d.get("mount", "pylon")),
                   n_strips=int(d.get("n_strips", 24)), notes=str(d.get("notes", "")),
                   builtin=bool(d.get("builtin", False)), legacy=d.get("legacy"),
                   aero=dict(d.get("aero", {}))).clamp()

    def reynolds(self, V: float | None = None) -> float:
        return reynolds(V_REF[self.role] if V is None else V, self.mac)


# --------------------------------------------------------------------------- #
#  analysis                                                                    #
# --------------------------------------------------------------------------- #
#: Standoff from the car's flank to the deployed panel, m -- the gap the body
#: image is placed across. `analyse`'s own `standoff` default, kept as a name
#: so the lattice and `strut_cd` cannot disagree about the same gap.
FLANK_STANDOFF = RIDE_H0["flank"]

#: A TOP wing's pylon length, m. It is NOT the flank standoff, and the two only
#: ever shared a number because `analyse`'s `standoff` argument had one default
#: for both roles: `strut_cd` and `wing_mass` charge a pylon of this length,
#: while the lattice's wall distance for a top wing is its RIDE HEIGHT and
#: never reads this at all. Held at the 0.45 the whole library was analysed
#: with, so making the flank's standoff a design row moves no top wing.
#:
#: UNMODELLED, and stated as such: a real pylon reaches from the deck to the
#: wing, so its length is `ride_h - deck_z(x)` and it shortens as the wing
#: comes down. AeroBO carries that in `carwing.MountSpec`; this module has no
#: deck to measure against -- it may not import the car mesh -- and charges a
#: constant instead. Fixing it is a separate change with its own numbers.
TOP_PYLON_L = 0.45


#: Least fraction of its own span a wing must keep between its plates. A
#: blended plate reaches OUTBOARD, and the span row is what the car is allowed
#: to be wide, so the wing gives that reach up out of its own span. Past this
#: the "wing" is two plates with a strip between them, and the honest answer
#: is a stated refusal rather than a lattice that solves something nobody
#: drew. It does not bind anywhere inside the packaging bands (the worst case
#: in band is a top wing at plate 0.30 m, full blend: 0.365 m of a 0.70 m
#: span, leaving 48 %), so it is a guard and not a gate.
SPAN_LEFT_MIN = 0.25


def plate_flown(spec: WingSpec, ride_h: float | None = None,
                standoff: float | None = None, body_image: bool = True,
                wall_side: float = +1.0) -> dict:
    """The plate, and the wall, AS THE LATTICE BUILDS THEM.

    One place, because three numbers depend on each other and used to be
    worked out inline: the wall's distance, the plate's arc after it has been
    shortened to clear that wall, and the span left for the wing once the
    plate's blend has reached outboard. `build_lattice` and `analyse` both
    read it, so a read-out and the geometry cannot disagree.

    Raises ValueError if the plates' reach eats the wing (see SPAN_LEFT_MIN).
    """
    if ride_h is None and spec.role == "top":
        ride_h = spec.ride_h_flown
    if standoff is None:
        standoff = spec.ride_h_flown if spec.role == "flank" else TOP_PYLON_L
    b = spec.span
    image = None
    arc = spec.plate_h_flown            # an endplate mount forces its minimum
    #  ...and the CLEARANCE is a height, while the plate's row is an ARC. They
    #  are the same number only for a straight plate; a blended one of the
    #  same arc does not climb as far (`blend.height_fraction`), so it may be
    #  LONGER before it touches the wall. Dividing by that fraction is the
    #  inverse, exactly -- the path is homogeneous of degree one in the arc.
    hf = bl.height_fraction(90.0, spec.plate_blend, spec.plate_shape)
    if spec.role == "top" and ride_h is not None:
        image = float(ride_h)                  # the track, above, in the mirrored frame
        arc = min(arc, max(ride_h - 0.03 * b - 0.01, 0.0) / max(hf, 1e-9))
    elif spec.role == "flank" and body_image and standoff > 0.0:
        #  THE CAR'S FLANK, as a rigid wall. Exactly the top wing's ground
        #  plane, one frame over: the panel's lift is the lattice's +z and on
        #  the OUTER flank the body is on the lift side (the device's useful
        #  force is INBOARD, toward the body), so the wall sits at +standoff
        #  and the tip plates point towards it and are shortened to clear it --
        #  the same three lines the track gets.
        #
        #  It was FREE AIR before, which is why the sim could not answer the
        #  owner's question about orientation: with no wall, mirroring the
        #  section is exactly equivalent to negating CL, so the two
        #  orientations were indistinguishable by construction.
        #
        #  `wall_side` = -1 puts the body on the PRESSURE side instead. That is
        #  the SAME PANEL turned over -- what it takes to deploy on the INNER
        #  flank and still point the side force at the turn centre (CONTRACT
        #  section 4, `cfg.dev_flank`). Rotating the panel 180 deg about its
        #  span carries the tip plates with it, and this lattice builds them on
        #  the lift (+z) side, so they still point +z and now face AWAY from
        #  the body: nothing to clear, hence no clip. At the standoff the car
        #  deploys to the clip does not bind on any library flank wing anyway
        #  (0.566 m at RIDE_H0['flank'] = 0.60, against plates of 0.06-0.16 m),
        #  so the two orientations differ ONLY in which side of the panel the
        #  wall is on, which is what makes them comparable.
        image = float(wall_side) * float(standoff)
        if wall_side > 0.0:
            arc = min(arc, max(standoff - 0.03 * b - 0.01, 0.0) / max(hf, 1e-9))
    blend = spec.plate_blend if arc > 0.0 else 0.0
    proj = bl.projection(arc, 90.0, blend, spec.plate_shape) if arc > 0.0 else 0.0
    span = b - 2.0 * proj
    if span < SPAN_LEFT_MIN * b:
        raise ValueError(
            f"the plates' blend reaches {proj:.3f} m outboard a side, which leaves "
            f"{span:.3f} m of a {b:.3f} m span between them -- the span row is what "
            f"the car is allowed to be wide, so the wing pays for that reach")
    return dict(image=image, arc=float(arc), blend=float(blend),
                shape=spec.plate_shape, projection=float(proj),
                tip_height=float(hf * arc), span=float(span),
                ride_h=ride_h, standoff=float(standoff))


def build_lattice(spec: WingSpec, polar: Polar, ride_h: float | None = None,
                  standoff: float | None = None,
                  body_image: bool = True, wall_side: float = +1.0,
                  plate_polar: Polar | None = None) -> Lattice:
    #  The distance to the imaged wall is the spec's OWN design row now
    #  (AeroBO's `ride_height_m`), so an explicit argument is an override --
    #  the garage still passes the slot's height for a top wing -- and None
    #  means "fly the wing as designed".
    g = plate_flown(spec, ride_h, standoff, body_image, wall_side)
    #  THE SPAN THE WING ACTUALLY FLIES. Its row is the car's width: a blended
    #  plate leans outboard, and it is the wing that gives that up, not the
    #  regulation. With no blend the reach is zero and this is the span row,
    #  which is why no wing in the library moves.
    b = g["span"]
    lam, c0 = spec.taper, spec.chord
    tw_r = math.radians(spec.twist_root_deg)
    tw_t = math.radians(spec.twist_deg)

    def chord(y):
        return c0 * (1.0 - (1.0 - lam) * np.abs(np.asarray(y, float)) / (0.5 * b))

    def twist(y):
        #  linear root -> tip, AeroBO's two-row law. twist_root_deg = 0 gives
        #  back the single-row law this module carried before, exactly.
        eta = np.abs(np.asarray(y, float)) / (0.5 * b)
        return tw_r + (tw_t - tw_r) * eta

    #  the plate's own section. `plate_polar` is None for the flat plate the
    #  lattice has always assumed, and every wing analysed before the row
    #  existed takes that branch, so the shipped numbers do not move.
    p_a, p_L0 = TWO_PI, 0.0
    if plate_polar is not None:
        p_a, p_L0 = plate_polar.a_lin, math.radians(plate_polar.alpha_L0_deg)
    return Lattice(b, chord, twist, polar.a_lin, math.radians(polar.alpha_L0_deg),
                   N=spec.n_strips, plate_h=g["arc"], n_plate=6, plate_a=p_a, plate_L0=p_L0,
                   plate_blend=g["blend"], plate_shape=g["shape"],
                   image_z=g["image"], image_sign=-1.0, V=1.0)


def strut_cd(spec: WingSpec, standoff: float, s_ref: float, V: float) -> float:
    """Wetted-area friction of the mounts on S_ref: two struts (flank) or two
    pylons (top) of `standoff` length; chords 0.04 m / 0.10 m.

    Zero unless the wing is actually pylon-mounted: an endplate mount puts no
    strut in the flow (it pays for its plates through `cd_pl` instead), and
    'none' is the mountless idealisation. See MOUNTS."""
    if spec.mount != "pylon":
        return 0.0
    c = 0.04 if spec.role == "flank" else 0.10
    cf = friction_coefficient(reynolds(V, c)) * 1.3      # + form factor
    return 2.0 * 2.0 * max(standoff, 0.0) * c * cf / max(s_ref, 1e-6)


def wing_mass(spec: WingSpec, standoff: float = 0.45) -> float:
    """Bottom-up mass of the wing AND its mount, kg -- an estimate, and the
    honest reason an endplate is not free.

    Two skins over the planform, two tip plates of `plate_h_flown` x MAC, and
    either two pylons of `standoff` (pylon mount) or nothing (the plates are
    already counted, and they ARE the mount). No ribs, no fasteners, no
    hardpoint reinforcement in the body: this is a floor, not a weight sheet.
    """
    skin = 2.0 * spec.S * SKIN_KG_M2
    plates = 2.0 * spec.plate_h_flown * spec.mac * SKIN_KG_M2
    pylons = 2.0 * max(standoff, 0.0) * PYLON_KG_M if spec.mount == "pylon" else 0.0
    return float(skin + plates + pylons)


def analyse(spec: WingSpec, polar: Polar, V: float | None = None, ride_h: float | None = None,
            standoff: float | None = None, rho: float = RHO,
            body_image: bool = True, wall_side: float = +1.0,
            plate_polar: Polar | None = None) -> dict:
    """The affine/quadratic laws the vehicle reads, plus a table for plots.

    Raises ValueError for a geometry the lattice refuses (e.g. plates into
    the track); the caller shows the reason and keeps the last good aero."""
    V = V_REF[spec.role] if V is None else float(V)
    if standoff is None:
        standoff = spec.ride_h_flown if spec.role == "flank" else TOP_PYLON_L
    lat = build_lattice(spec, polar, ride_h, standoff=standoff,
                        body_image=body_image, wall_side=wall_side,
                        plate_polar=plate_polar)
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
    #  THE CORNER'S OWN DRAG, which the lattice cannot see. A lifting-surface
    #  method values the wing/plate junction only through the (y, z) line it
    #  draws, so a sharp corner and a blend of the same arc are two lines and
    #  nothing more -- the interference drag that is the whole reason to blend
    #  is invisible to it. `blend` prices it as a reduced-order ADD-ON, with a
    #  fillet credit that is a calibrated shape and not a measurement, which
    #  is why it is charged only when the wing has asked for it and is
    #  reported as its own line item beside the UNCREDITED number.
    #
    #  The junction member is the PLATE: its chord at the corner is the wing's
    #  tip chord, and its thickness is its own section's. A plate with no
    #  section is bare sheet -- t/c below the correlation's own root -- and the
    #  correlation returns zero there rather than extrapolating.
    c_corner = float(spec.chord * spec.taper)
    tc_plate = float(getattr(plate_polar, "tc", 0.0) or 0.0)
    junc, cd_junction = None, 0.0
    if lat.plate_h > 0.0 and spec.plate_junction:
        junc = bl.junction_report(max(tc_plate, 1e-6), c_corner, S, lat.plate_h,
                                  90.0, lat.plate_blend, 2, lat.plate_shape)
        junc["tc"] = tc_plate
        junc["chord_corner_m"] = c_corner
        cd_junction = float(junc["CD_junction"]) if tc_plate > 0.0 else 0.0
        junc["CD_junction"] = cd_junction
    for a in alphas:
        r = lat.solve(float(a))
        cdp = float(np.sum(polar.cd_at(np.clip(r.alpha_eff_deg[main], polar.alpha_valid[0],
                                                polar.alpha_valid[1])) * c_w[main]) / S)
        cd_pl = float(np.sum((cf_plate + 0.012 * r.cl[plate] ** 2) * c_w[plate]) / S) if plate.any() else 0.0
        CLs.append(r.CL)
        CDis.append(r.CDi)
        CDps.append(cdp)
        CDs.append(r.CDi + cdp + cd_pl + cd_strut + cd_junction)
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
        #  the wing -> plate TRANSITION as it was flown: the arc spent turning,
        #  how far the plate leaned outboard, and the span that left the wing
        plate_blend=float(lat.plate_blend), plate_shape=str(lat.plate_shape),
        plate_projection=float(lat.plate_projection),
        plate_tip_height=float(lat.plate_tip_height), span_flown=float(lat.b),
        cd_junction=float(cd_junction), junction=junc,
        CL0=float(CL0), CLa=float(CLa), CL_max=float(CL_max), CL_min=float(CL_min),
        alpha_stall_deg=float(a_pos), alpha_stall_neg_deg=float(a_neg),
        e=float(np.nanmedian(np.asarray(es))) if np.isfinite(np.nanmedian(np.asarray(es))) else 0.0,
        cd0=float(cd0), cd1=float(cd1), cd2=float(cd2), cd_fit_err=fit_err,
        cd_strut=float(cd_strut), CDp_min=float(np.min(CDps)),
        mount=str(spec.mount), mass=float(wing_mass(spec, standoff)),
        wall_side=float(wall_side),
        y_cp=float(r_mid.y_cp),
        table=dict(alpha=[round(float(v), 3) for v in alphas], CL=[round(float(v), 4) for v in CLs],
                   CD=[round(float(v), 5) for v in CDs], CDi=[round(float(v), 5) for v in CDis],
                   CDp=[round(float(v), 5) for v in CDps]),
    )


def spanwise(spec: WingSpec, polar: Polar, inc_deg: float, ride_h: float | None = None,
             standoff: float | None = None, body_image: bool = True,
             wall_side: float = +1.0, plate_polar: Polar | None = None) -> dict:
    """Strip loading at one mount angle, for the designer's plot.

    Takes `standoff` / `body_image` so it solves the SAME lattice `analyse`
    did. It used to build its own with the defaults, which meant a caller
    asking for free air got the imaged loading back and the plot silently
    disagreed with the laws beside it.
    """
    lat = build_lattice(spec, polar, ride_h, standoff=standoff,
                        body_image=body_image, wall_side=wall_side,
                        plate_polar=plate_polar)
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

    # -- the design vector is the DESIGNER page's row order ----------------- #
    #  The order is AeroBO's `evaluate_car_wing` vector, which the DESIGNER
    #  page now lists in the same sequence (CONTRACT section 7):
    #      x = [taper, twist_root, twist_tip, alpha, endplate_h, ride, (S,) b]
    #  Root chord is NOT a row -- it follows from the area and the span, see
    #  `design_table` -- and the area row appears only when one is declared.
    page_order = ("taper", "twist_root_deg", "twist_deg", "inc_deg",
                  "plate_h", "ride_h", "span")
    rep("design vector is AeroBO's evaluate_car_wing order (CONTRACT section 7)",
        design_vars() == page_order, " -> ".join(design_vars()))
    rep("the AREA row sits one ahead of the span, and only when declared",
        design_vars(area=True) == page_order[:-1] + ("area", "span")
        and "area" not in design_vars(),
        " -> ".join(design_vars(area=True)))
    for role in ROLES:
        bl = design_bounds(role)
        labs = design_labels(role)
        rep(f"{role}: bounds, labels and BOUNDS agree coordinate for coordinate",
            len(bl) == len(labs) == len(DESIGN_VARS)
            and all(bl[i] == BOUNDS[role][a] for i, a in enumerate(design_vars(role))),
            f"{len(bl)} vars: {', '.join(labs)}")
    #  clamped, so `ride_h` and `area` carry their resolved values rather than
    #  the 0.0 sentinel: a decode fills them, and an unresolved start vector
    #  would read as a coordinate that moved on its own.
    sp0 = WingSpec("order", "flank", "naca2412", span=0.60, chord=0.40, taper=0.70,
                   twist_root_deg=1.0, twist_deg=-2.0, plate_h=0.04).clamp()
    x_ref = design_x0(sp0, 3.0)
    sp1 = sp0.copy()
    inc1 = apply_design(sp1, x_ref)
    rep("design_x0 -> apply_design is the identity", design_x0(sp1, inc1) == x_ref,
        format_design(x_ref))
    # The failure a hand-written decode has: coordinate i writing variable j.
    # Push one coordinate to its upper bound and check that exactly the one
    # quantity it names moved.
    crossed = []
    for i, attr in enumerate(design_vars("flank")):
        x = list(x_ref)
        x[i] = BOUNDS["flank"][attr][1]
        sp = sp0.copy()
        inc = apply_design(sp, x)
        moved = [a for a in design_vars("flank", owner="spec")
                 if getattr(sp, a) != getattr(sp0, a)]
        want_moved = [] if attr == "inc_deg" else [attr]
        want_inc = x[i] if attr == "inc_deg" else 3.0
        if moved != want_moved or inc != want_inc:
            crossed.append(f"{attr}: moved {moved}, inc {inc:+.1f}")
    rep("every coordinate decodes onto its own variable and no other",
        not crossed, f"{len(x_ref)} coordinates, none crossed" if not crossed else str(crossed))
    try:
        design_bounds("flank", spann=(0.4, 0.9))
        rep("an unknown bound override raises rather than being dropped", False, "no exception")
    except ValueError as exc:
        rep("an unknown bound override raises rather than being dropped", True, str(exc))
    capped = design_bounds("flank", span=(0.35, span_fit("flank", 0.90)))
    i_b = design_vars("flank").index("span")
    plain = design_bounds("flank")
    rep("an override narrows one band and leaves the order alone",
        capped[i_b][0] == 0.35 and abs(capped[i_b][1] - 0.88) < 1e-12
        and capped[:i_b] == plain[:i_b] and capped[i_b + 1:] == plain[i_b + 1:],
        f"span capped to {capped[i_b][1]:.3f} m by the sill/roof fit, "
        f"the other {len(plain) - 1} untouched")
    #  the garage's optimiser has always used this fit; reproduce its two
    #  literals exactly so adopting span_fit cannot move any bound. The band's
    #  ceiling is the packaging band's (1.30 m since the bands were opened
    #  up); the sill-to-rail fit is what actually caps a flank, at 1.06 m
    rep("span_fit reproduces the optimiser's sill/roof fit",
        all(abs(span_fit("flank", h)
                - max(min(BOUNDS["flank"]["span"][1], 2.0 * min(h - 0.28, 1.34 - h)), 0.40)) < 1e-12
            for h in (0.40, 0.60, 0.81, 0.90, 1.00, 1.15, 1.20))
        and span_fit("top", 1.6) == BOUNDS["top"]["span"][1],
        f"h 0.90 -> {span_fit('flank', 0.90):.2f} m, h 1.15 -> {span_fit('flank', 1.15):.2f} m, "
        f"top unconditional {span_fit('top', 1.6):.2f} m")

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
    # -- the rear wing's mount: pylon vs endplate vs none ------------------- #
    #  'pylon' is the shipped default and must be the number every library
    #  wing was analysed with, so it is checked against the value computed
    #  before `mount` existed rather than against its own re-run.
    m_py = analyse(top.copy(mount="pylon"), pt, ride_h=0.45)
    m_ep = analyse(top.copy(mount="endplate", plate_h=0.0), pt, ride_h=0.45)
    m_no = analyse(top.copy(mount="none"), pt, ride_h=0.45)
    rep("mount: 'pylon' is the default and reproduces the pre-mount analysis",
        WingSpec().mount == "pylon" and m_py["cd0"] == ge["cd0"] and m_py["CLa"] == ge["CLa"]
        and m_py["cd_strut"] == ge["cd_strut"],
        f"cd_strut {m_py['cd_strut'] * 1e4:.1f} counts, identical to the default path")
    rep("mount: 'none' charges no mount drag (the floor at a given plate height)",
        m_no["cd_strut"] == 0.0 and m_no["cd0"] < m_py["cd0"],
        f"cd0 {m_py['cd0'] * 1e4:.0f} ct (pylon) -> {m_no['cd0'] * 1e4:.0f} ct (none), "
        f"the {(m_py['cd0'] - m_no['cd0']) * 1e4:.0f} ct the two pylons cost")
    #  an endplate mount is the trade the owner asked for: it buys lift slope
    #  by cutting tip loss (straight out of the lattice, no correlation) and
    #  pays in plate wetted area and mass, but not in pylon drag
    rep("mount: 'endplate' forces its structural plate height",
        m_ep["plate_h_flown"] == MOUNT_PLATE_H["top"] and m_no["plate_h_flown"] == 0.12,
        f"plate_h 0.0 -> {m_ep['plate_h_flown']:.2f} m flown (MOUNT_PLATE_H top)")
    ep_vs_none = analyse(top.copy(mount="none", plate_h=0.0), pt, ride_h=0.45)
    rep("mount: 'endplate' raises CLa (less tip loss) and e over the bare wing",
        m_ep["CLa"] > ep_vs_none["CLa"] * 1.02 and m_ep["e"] > ep_vs_none["e"],
        f"CLa {ep_vs_none['CLa']:.3f} -> {m_ep['CLa']:.3f}, e {ep_vs_none['e']:.3f} -> {m_ep['e']:.3f}")
    rep("mount: 'endplate' pays for it in profile drag and mass",
        m_ep["cd0"] > ep_vs_none["cd0"] and m_ep["mass"] > ep_vs_none["mass"],
        f"cd0 {ep_vs_none['cd0'] * 1e4:.0f} -> {m_ep['cd0'] * 1e4:.0f} ct, "
        f"mass {ep_vs_none['mass']:.2f} -> {m_ep['mass']:.2f} kg "
        f"(pylon {m_py['mass']:.2f} kg)")
    rep("mount: an unknown mount clamps to 'pylon' rather than being flown",
        WingSpec(mount="swan-neck").clamp().mount == "pylon", "")

    # -- the wing -> plate TRANSITION ---------------------------------------
    naca0012 = af.naca4_coords("0012")
    p_pol = estimate_polar(naca0012, re_bank_snap(spec.reynolds() * 0.85), "naca0012")
    bl_base = spec.copy(plate_h=0.12, mount="endplate", plate_airfoil="naca0012")
    a_sharp = analyse(bl_base, pol, plate_polar=p_pol)
    a_zero = analyse(bl_base.copy(plate_shape="spiral", plate_blend=0.0), pol, plate_polar=p_pol)
    rep("transition: blend 0 is the published right-angle corner, bit-for-bit",
        all(a_sharp[k] == a_zero[k] for k in ("CL0", "CLa", "cd0", "cd1", "cd2", "S", "e")),
        "the shape row cannot move a wing that is not blended")
    bl_60 = bl_base.copy(plate_shape="spiral", plate_blend=0.6, plate_junction=True)
    a_60 = analyse(bl_60, pol, plate_polar=p_pol)
    rep("transition: a blended plate leans outboard and the WING pays for it",
        a_60["plate_projection"] > 0.0
        and abs(a_60["span_flown"] + 2.0 * a_60["plate_projection"] - bl_60.span) < 1e-12
        and a_sharp["span_flown"] == bl_base.span,
        f"reach {a_60['plate_projection']:.4f} m a side, so the span flies "
        f"{a_60['span_flown']:.4f} m of a {bl_60.span:.3f} m row")
    rep("transition: the plate keeps its ARC and gives up tip height for it",
        abs(a_60["plate_h_flown"] - a_sharp["plate_h_flown"]) < 1e-12
        and a_60["plate_tip_height"] < a_sharp["plate_tip_height"],
        f"arc {a_60['plate_h_flown']:.3f} m both; reaches "
        f"{a_60['plate_tip_height']:.4f} m up, against {a_sharp['plate_tip_height']:.4f} m")
    rep("transition: a smoother corner is a better one in the Trefftz plane",
        a_60["e"] > a_sharp["e"] and a_60["CLa"] > a_sharp["CLa"],
        f"e {a_sharp['e']:.4f} -> {a_60['e']:.4f}, CLa {a_sharp['CLa']:.3f} -> {a_60['CLa']:.3f}")
    j_sharp = analyse(bl_base.copy(plate_junction=True), pol, plate_polar=p_pol)
    rep("transition: the junction charge is real, and blending CREDITS it",
        j_sharp["cd_junction"] > a_60["cd_junction"] > 0.0
        and a_sharp["cd_junction"] == 0.0,
        f"{j_sharp['cd_junction'] * 1e4:.1f} ct sharp -> {a_60['cd_junction'] * 1e4:.1f} ct at "
        f"blend 0.6 (credit {a_60['junction']['fillet_credit']:.2f}); OFF unless asked for")
    flat = analyse(bl_base.copy(plate_airfoil="", plate_junction=True), pol)
    rep("transition: a plate with no section is bare sheet and charges nothing",
        flat["cd_junction"] == 0.0 and not flat["junction"]["tc_in_correlation_band"],
        "Hoerner's correlation scales on the junction member's THICKNESS and "
        "goes negative below t/c ~ 0.054; it returns zero rather than extrapolating")
    got = ""
    try:
        analyse(spec.copy(span=BOUNDS["flank"]["span"][0], plate_h=0.60,
                          plate_blend=1.0, plate_shape="spiral"), pol)
    except ValueError as exc:
        got = str(exc)
    rep("transition: plates that eat the wing are a stated refusal, not a solve",
        "pays for that reach" in got, got[:72] + "...")

    back = WingSpec.from_json(spec.copy(aero=a).to_json())
    rep("spec json round-trip", back.aero["CLa"] == a["CLa"] and back.name == spec.name
        and back.mount == spec.mount, f"mount '{back.mount}' survives")
    import time
    t0 = time.perf_counter()
    for _ in range(5):
        analyse(spec, pol)
    ms = (time.perf_counter() - t0) * 1e3 / 5
    #  the area SENTINEL survives the round trip. A spec straight out of the
    #  library carries area = 0.0 ("take the planform's own") until something
    #  clamps it, and `apply_design` is documented as `design_x0`'s exact
    #  inverse -- so a decode that read the sentinel literally put the chord
    #  at zero and made the lattice refuse every candidate it was handed.
    sp0 = WingSpec(role="flank", span=0.80, chord=0.45, taper=0.85, area=0.0)
    S_true = sp0.S
    xs = design_x0(sp0, 3.0)
    sp1 = WingSpec(role="flank", span=0.80, chord=0.45, taper=0.85, area=0.0)
    inc1 = apply_design(sp1, xs, clamp=False)
    rep("a spec carrying the area sentinel round-trips",
        abs(sp1.chord - 0.45) < 1e-12 and abs(sp1.area - S_true) < 1e-12
        and abs(inc1 - 3.0) < 1e-12,
        f"area 0.0 -> {sp1.area:.5f} m2, chord {sp1.chord:.5f} m")
    sp2 = WingSpec(role="flank", span=0.80, chord=0.45, taper=0.85, area=0.0)
    xm = list(xs)
    xm[design_vars("flank").index("span")] = 0.60
    apply_design(sp2, xm, clamp=False)
    rep("...and moving the span off it keeps the AREA, not the chord",
        abs(sp2.area - S_true) < 1e-12 and sp2.chord > 0.45,
        f"span 0.80 -> 0.60 m: chord 0.4500 -> {sp2.chord:.4f} m at S {sp2.area:.5f}")

    rep("analyse cost", ms < 40.0, f"{ms:.1f} ms")
    if verbose:
        print("  ALL PASS" if ok else "  FAILURES ABOVE")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
