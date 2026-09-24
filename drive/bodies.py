"""Per-car body geometry, and the wing packaging limits that follow from it.

Task 41. The owner's rule, verbatim: "Let the max span be different for each
car, the actual physical maximum the span can be without touching the ground
(for the side) and 1.2 of the width for the top wing." Before this module the
span band was the Corsa's everywhere: `aero.wing.span_fit` held a Corsa sill
(0.28 m) and roof rail (1.34 m), and the garage clamped every car's slots to
the Corsa's body.

What lives here, and nowhere else:

  * the body SHELLS (the cross-sections `render.py` draws and the garage
    fits wings to), one per body style, moved out of `render.py` so that code
    with no pygame -- the wing module, the challenges, the records, AeroBO's
    bridge -- can read a car's size;
  * `Body(car)`: a car's shell mapped onto ITS axles, with the numbers the
    limits need (`x_front`, `x_rear`, `width`, `height`, `ground`, `deck_z`);
  * the SLOT bands a car's three wing slots may move in, and its default slots;
  * the SPAN LIMITS:

        flank   span <= 2 (h - ground)   a vertical panel centred at mount
                                         height h reaches h - span/2 at its
                                         lower tip; that tip may come down to
                                         the car's own ground clearance (the
                                         lowest underbody station of its
                                         shell) and no further
        top     span <= 1.2 x width      the body's published width

    and the "Unlimited" ceiling: `UNLIMITED_FACTOR` times that, which the
    garage offers only when Settings' `wing_limits` is 'unlimited';
  * `over_limits(build, lib, car)`: which fitted wings are past their car's
    physical limit. It is COMPUTED, never stored: a build names its wings, a
    wing can be re-saved at another span, and a build can be driven on
    another car -- so legality belongs to (build, library, car) at the moment
    of the run. A run whose build has any entry here is an Unlimited run:
    it is filed apart from the official records and never goes to a public
    board.

The flank rule is STATIC: body roll dips the outer panel (about 0.12 m at
1 g on the Corsa with a panel deployed 1.42 m out), and the physics never
tests ground contact, so the rule is the one a garage inspector would apply
with the car standing still, which is what "physical maximum" is taken to
mean. The ground clearance margin -- the car's own underbody height rather
than zero -- is what keeps a legal panel from striking the road the first
time the car rolls onto it.

No pygame here, and `cars` is imported lazily: `drive.aero.wing` reads this
module from inside a function, and it must stay importable anywhere.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
#  the shells (moved verbatim from render.py; render imports them from here)   #
# --------------------------------------------------------------------------- #
#: the Corsa C's published exterior -- the same numbers render.py and
#: garage.py carry under the same names
CAR_HALF_W = 0.823       # m  published width 1.646
CAR_H = 1.440            # m  published height
CAR_X_REAR = -2.0675     # m  behind CG (0.548 overhang + b = 1.5195)

#: each style's shell, drawn round a real car's PUBLISHED exterior (length,
#: width, height; the overhangs of the MX-5 and the E39 are est). `a` is that
#: car's own CG-to-front-axle, the frame its stations are written in, so a
#: fitted car with a different a / b / L (ballast moves a and b) gets the
#: same shell placed on ITS axles. `t` is the mean track the widths scale by.
CAR_STYLE_REF = {
    'hatch': dict(length=3.817, width=1.646, height=1.440, ovh_f=0.780,
                  L=2.491, t=1.4245, a=0.97149),       # Opel Corsa C 5-door
    'roadster': dict(length=3.945, width=1.680, height=1.235, ovh_f=0.845,
                     L=2.265, t=1.4275, a=1.0872),     # Mazda MX-5 NB (ovh est)
    'saloon': dict(length=4.775, width=1.800, height=1.435, ovh_f=0.840,
                   L=2.830, t=1.519, a=1.3867),        # BMW E39 (ovh est)
    #  Renault Express (the Renault 5-based van, 1985-2000): length 4.056,
    #  width 1.566, height 1.776, wheelbase 2.580 m published (Wikipedia,
    #  "Renault Express"). t and a are cars.EXPRESS_14's own (tracks
    #  1.326 / 1.288 published by French Wikipedia; a from its est 60 %
    #  front), so the shell sits on that car's axles by the identity map.
    'van': dict(length=4.056, width=1.566, height=1.776, ovh_f=0.720,
                L=2.580, t=0.5 * (1.326 + 1.288), a=(1.0 - 0.60) * 2.580),
    #  Mercedes-Benz Citaro O530, 12 m city bus: length 11.950, width 2.550,
    #  height 3.120 (roof air conditioning), wheelbase 5.845, front overhang
    #  2.705 m published (Mercedes-Benz Citaro data sheet). t and a are
    #  cars.CITARO_O530's own (both est there), written as the same
    #  expressions so they are the same floats and the map is the identity.
    'bus': dict(length=11.950, width=2.550, height=3.120, ovh_f=2.705,
                L=5.845, t=0.5 * (2.100 + 1.840), a=(1.0 - 0.36) * 5.845),
}

#: Cross-sections down each shell: (x, z_floor, z_belt, z_top, half_w,
#: half_w_roof) in the style car's own CG frame, nose first. The hatch is
#: garage.py's STATIONS with the tail re-cut (an upright tailgate and a
#: bumper ledge instead of one 48 deg slope, so lamps and a plate have a
#: panel to sit on); the other two are drawn to their cars' published
#: length / width / height. BAND kinds name what lies between consecutive
#: stations and decide the paint (`render._band_paint3`).
STATIONS3 = (
    (1.7515, 0.24, 0.58, 0.64, 0.700, 0.52),    # bumper face
    (1.60, 0.17, 0.68, 0.74, 0.790, 0.60),
    (1.10, 0.15, 0.78, 0.83, CAR_HALF_W, 0.68),
    (0.60, 0.15, 0.85, 0.90, CAR_HALF_W, 0.70),   # scuttle
    (-0.05, 0.15, 0.88, 1.38, CAR_HALF_W, 0.64),  # A-pillar top
    (-0.80, 0.15, 0.90, CAR_H, CAR_HALF_W, 0.64),  # roof
    (-1.48, 0.16, 0.92, 1.41, 0.815, 0.62),       # roof end (spoiler lip)
    (-1.82, 0.22, 0.94, 1.04, 0.795, 0.64),       # tailgate glass base
    (-1.98, 0.26, 0.68, 0.74, 0.745, 0.66),       # tailgate foot / bumper top
    (CAR_X_REAR, 0.27, 0.58, 0.64, 0.700, 0.62),  # rear bumper face
)
_STYLE_SHELL3 = {
    'hatch': (STATIONS3, ('nose', 'bonnet', 'bonnet', 'screen', 'roof', 'roof',
                          'rglass', 'tail', 'bumper')),
    'roadster': ((
        (1.932, 0.22, 0.50, 0.55, 0.68, 0.50),     # bumper face
        (1.80, 0.16, 0.60, 0.66, 0.79, 0.62),
        (1.15, 0.14, 0.70, 0.76, 0.84, 0.72),      # over the front wheels
        (0.45, 0.14, 0.76, 0.81, 0.84, 0.74),      # scuttle: the screen's foot
        (0.20, 0.14, 0.78, 0.82, 0.84, 0.74),      # cockpit front
        (-0.75, 0.14, 0.80, 0.84, 0.84, 0.74),     # cockpit rear
        (-0.95, 0.15, 0.82, 0.87, 0.84, 0.74),     # tonneau / deck front
        (-1.70, 0.18, 0.86, 0.92, 0.83, 0.70),     # boot lid rear edge
        (-1.93, 0.24, 0.70, 0.76, 0.77, 0.66),     # tail panel foot
        (-2.0128, 0.26, 0.60, 0.66, 0.72, 0.62),   # bumper face
    ), ('nose', 'bonnet', 'bonnet', 'dash', 'cockpit', 'deck', 'deck', 'tail',
        'bumper')),
    'saloon': ((
        (2.2267, 0.24, 0.60, 0.66, 0.76, 0.56),    # bumper face
        (2.08, 0.17, 0.70, 0.76, 0.86, 0.66),
        (1.45, 0.15, 0.78, 0.84, 0.90, 0.74),      # over the front wheels
        (0.62, 0.15, 0.86, 0.92, 0.90, 0.76),      # scuttle
        (-0.10, 0.15, 0.90, 1.38, 0.90, 0.68),     # A-pillar top
        (-1.00, 0.15, 0.91, 1.435, 0.90, 0.69),    # roof
        (-1.55, 0.15, 0.92, 1.40, 0.895, 0.67),    # C-pillar top
        (-2.02, 0.17, 0.93, 1.02, 0.88, 0.72),     # rear glass base / boot lid
        (-2.42, 0.20, 0.95, 1.01, 0.85, 0.74),     # boot lid rear edge
        (-2.50, 0.25, 0.64, 0.70, 0.82, 0.72),     # boot face foot
        (-2.5483, 0.26, 0.56, 0.62, 0.78, 0.68),   # bumper face
    ), ('nose', 'bonnet', 'bonnet', 'screen', 'roof', 'roof', 'rglass', 'deck',
        'tail', 'bumper')),
    #  the Express: a Renault 5 front end and cab, then the tall load box
    #  that steps up behind the cab roof to the published 1.776 m. First
    #  draft (task 41); the drawing detail is render.py's to refine, and any
    #  change here moves the span limits with it -- they read these numbers.
    'van': ((
        (1.752, 0.24, 0.52, 0.58, 0.690, 0.52),    # bumper face
        (1.62, 0.18, 0.64, 0.70, 0.760, 0.60),
        (1.10, 0.16, 0.74, 0.80, 0.783, 0.64),     # over the front wheels
        (0.55, 0.16, 0.82, 0.88, 0.783, 0.66),     # scuttle: the screen's foot
        (-0.15, 0.16, 0.86, 1.38, 0.783, 0.64),    # cab roof front
        (-0.55, 0.16, 0.88, 1.40, 0.783, 0.66),    # cab roof rear
        (-0.62, 0.16, 0.90, 1.776, 0.783, 0.74),   # load box front top
        (-2.20, 0.20, 0.92, 1.776, 0.783, 0.74),   # load box rear top
        (-2.27, 0.26, 0.60, 1.74, 0.770, 0.72),    # rear doors
        (-2.304, 0.27, 0.54, 0.60, 0.720, 0.62),   # bumper face
    ), ('nose', 'bonnet', 'bonnet', 'screen', 'roof', 'roof', 'roof', 'tail',
        'bumper')),
    #  the Citaro: a flat front face that is nearly all windscreen, a flat
    #  roof with the air-conditioning pod amidships (the published 3.120 m),
    #  the engine bay behind the rear axle. First draft, as the van's.
    'bus': ((
        (6.446, 0.34, 0.95, 2.90, 1.200, 1.12),    # front face
        (6.36, 0.30, 1.00, 3.00, 1.260, 1.20),
        (6.10, 0.28, 1.02, 3.02, 1.275, 1.23),     # front roof edge
        (2.50, 0.28, 1.02, 3.02, 1.275, 1.23),
        (2.00, 0.28, 1.02, 3.12, 1.275, 1.10),     # air-con pod front
        (-0.50, 0.28, 1.02, 3.12, 1.275, 1.10),    # air-con pod rear
        (-1.00, 0.28, 1.02, 3.02, 1.275, 1.23),
        (-5.30, 0.30, 1.02, 3.00, 1.275, 1.22),    # rear roof edge
        (-5.46, 0.34, 0.80, 2.95, 1.260, 1.18),    # engine cover
        (-5.504, 0.38, 0.62, 0.70, 1.200, 1.10),   # rear bumper face
    ), ('nose', 'screen', 'roof', 'roof', 'roof', 'roof', 'roof', 'tail',
        'bumper')),
}

#: which body each car in `cars.CARS` wears, by KEY. A key not listed here
#: (a custom CarSpec, a test stand-in) is matched on its NAME, and anything
#: unknown draws as the hatch, which is the study's own car.
STYLE_OF = {"corsa": "hatch", "mx5": "roadster", "540i": "saloon",
            "express": "van", "bus": "bus"}


def _spec(car):
    """`car` as a CarSpec-like object, or None when it is a key with no
    registered car (a style can be looked at before `cars.py` carries it)."""
    if car is None:
        car = "corsa"
    if isinstance(car, str):
        import cars as _cars             # lazy: see the module docstring
        return _cars.CARS.get(car)
    return car


def style_of(car=None) -> str:
    """The body style of a car -- a `cars.py` KEY or a CarSpec, the two
    interchangeable. None is the Corsa. `render.car_style` is this function
    (it only resolves its own None to the fitted car first)."""
    if car is None:
        car = "corsa"
    if isinstance(car, str):
        if car in STYLE_OF:
            return STYLE_OF[car]
        spec = _spec(car)
        name = spec.name if spec is not None else car
    else:
        name = getattr(car, 'name', '')
    name = str(name or '').lower()
    if 'mx-5' in name or 'mx5' in name or 'roadster' in name:
        return 'roadster'
    if 'bmw' in name or '540' in name or 'saloon' in name or 'sedan' in name:
        return 'saloon'
    if 'express' in name or 'van' in name.split():
        return 'van'
    if 'citaro' in name or 'bus' in name.split():
        return 'bus'
    return 'hatch'


def map_stations(style: str, a: float, L: float, k_w: float) -> tuple:
    """`style`'s stations placed on a car with CG-to-front-axle `a` and
    wheelbase `L`, widths scaled by `k_w` (the car's mean track over the style
    car's). The x map keeps the overhangs in metres and stretches the stretch
    between the axles by L / L_ref, so the shell always sits on the wheels.
    For every stock car both maps are the identity. `render.CarGeom` uses
    this, so the drawn body and the limits cannot disagree."""
    ref = CAR_STYLE_REF[style]
    a_ref, L_ref = ref['a'], ref['L']

    def mx(x):
        xf = x - a_ref                        # from the style car's front axle
        if xf >= 0.0:
            return a + xf
        if xf <= -L_ref:
            return a - L + (xf + L_ref)
        return a + xf * L / L_ref
    stations, _bands = _STYLE_SHELL3[style]
    return tuple((mx(s[0]), s[1], s[2], s[3], s[4] * k_w, s[5] * k_w)
                 for s in stations)


def axles(car, style: str) -> tuple[float, float, float, float]:
    """(a, L, t_f, t_r) of `car`, falling back to the style car's own."""
    ref = CAR_STYLE_REF[style]
    L = float(getattr(car, 'L', ref['L']) or ref['L'])
    wd = getattr(car, 'wdist_f', None)
    a = float(getattr(car, 'a', (1.0 - wd) * L if wd is not None else ref['a']))
    t_f = float(getattr(car, 't_f', ref['t']) or ref['t'])
    t_r = float(getattr(car, 't_r', ref['t']) or ref['t'])
    return a, L, t_f, t_r


class Body:
    """A car's shell on its own axles, and what the limits read off it.

    `car` is a `cars.py` key, a CarSpec, or None (the Corsa). Everything is in
    the car's BODY frame: x forward of the STOCK CG, z up from the road.
    """

    def __init__(self, car=None):
        spec = _spec(car)
        self.style = style_of(car)
        ref = CAR_STYLE_REF[self.style]
        a, L, t_f, t_r = axles(spec if spec is not None else object(), self.style)
        k_w = 0.5 * (t_f + t_r) / ref['t']
        self.stations = map_stations(self.style, a, L, k_w)
        self.a, self.L = a, L
        self.x_front = self.stations[0][0]
        self.x_rear = self.stations[-1][0]
        self.half_w = max(s[4] for s in self.stations)
        self.width = 2.0 * self.half_w
        self.height = max(s[3] for s in self.stations)
        #: the lowest underbody station: the car's own ground clearance, and
        #: the lowest a flank panel's tip may come
        self.ground = min(s[1] for s in self.stations)
        self._xs = np.array([s[0] for s in self.stations][::-1])
        self._zt = np.array([s[3] for s in self.stations][::-1])
        self._hw = np.array([s[4] for s in self.stations][::-1])

    def deck_z(self, x: float) -> float:
        """The body's top surface height at station x."""
        return float(np.interp(x, self._xs, self._zt))

    def half_w_at(self, x: float) -> float:
        """The body side's half-width at station x."""
        return float(np.interp(x, self._xs, self._hw))


_BODY_CACHE: dict = {}


def body(car=None) -> Body:
    """`Body(car)`, cached on the key or the spec's shaping fields."""
    spec = _spec(car)
    if isinstance(car, str) or car is None:
        key = ("k", car or "corsa", id(spec))
    else:
        key = ("s", str(getattr(car, 'name', '')), getattr(car, 'L', None),
               getattr(car, 'wdist_f', None), getattr(car, 't_f', None),
               getattr(car, 't_r', None))
    b = _BODY_CACHE.get(key)
    if b is None:
        b = _BODY_CACHE[key] = Body(car)
    return b


# --------------------------------------------------------------------------- #
#  the slots: where each car's three wings may be bolted                       #
# --------------------------------------------------------------------------- #
#: The three stock cars keep EXACTLY the numbers the garage always used for
#: every car (flank h 0.40-1.20, top x to 0.55 m, top h to 1.85 m, the
#: default slots), so no saved build moves. A taller car raises the ceilings
#: with its body: flank centre to 0.24 m under the roof (the Corsa's 1.20 is
#: 0.24 under its 1.44), top wing to 0.41 m over it (the Corsa's 1.85).
FLANK_H_MIN = 0.40
FLANK_H_UNDER_ROOF = 0.24
TOP_H_OVER_ROOF = 0.41
TOP_REAR_MARGIN = 0.15          # m  the top wing's mount stays on the body
TOP_DECK_CLEAR = 0.14           # m  = garage TOP_STOW_GAP 0.06 + 0.08
#: per style: the top wing's forward limit, and the default slots
#: (flank x, h) and (top x, h, incidence). x is metres forward of the CG.
SLOT_TABLE = {
    'hatch':    dict(top_x_max=0.55, flank=(0.97, 0.90), top=(-0.90, 1.55, 6.0)),
    'roadster': dict(top_x_max=0.55, flank=(0.97, 0.90), top=(-0.90, 1.55, 6.0)),
    'saloon':   dict(top_x_max=0.55, flank=(0.97, 0.90), top=(-0.90, 1.55, 6.0)),
    'van':      dict(top_x_max=-0.70, flank=(0.97, 1.05), top=(-1.70, 1.95, 6.0)),
    'bus':      dict(top_x_max=5.60, flank=(3.00, 1.60), top=(-4.50, 3.30, 6.0)),
}


def flank_x_band(car=None, chord: float = 0.45) -> tuple[float, float]:
    """A flank panel's station band: the whole chord stays alongside the body."""
    b = body(car)
    return b.x_rear + 0.5 * chord, b.x_front - 0.5 * chord


def flank_h_band(car=None) -> tuple[float, float]:
    """A flank panel's mount-height band. The floor also keeps the smallest
    panel the optimiser may propose (span lo + 0.05) off the ground."""
    b = body(car)
    from .aero.wing import BOUNDS
    lo = max(FLANK_H_MIN, b.ground + 0.5 * (BOUNDS["flank"]["span"][0] + 0.05))
    hi = max(1.20, b.height - FLANK_H_UNDER_ROOF)
    return lo, hi


def top_x_band(car=None) -> tuple[float, float]:
    b = body(car)
    hi = min(SLOT_TABLE[b.style]["top_x_max"], b.x_front - TOP_REAR_MARGIN)
    return b.x_rear + TOP_REAR_MARGIN, hi


def top_h_band(car=None, x: float = -0.90) -> tuple[float, float]:
    """The top wing's height band at station x: clear of the deck below it."""
    b = body(car)
    return b.deck_z(x) + TOP_DECK_CLEAR, max(1.85, b.height + TOP_H_OVER_ROOF)


def slot_defaults(car=None) -> dict:
    """The empty car's slots for `car`: {'flank': (x, h), 'top': (x, h, inc)}."""
    t = SLOT_TABLE[body(car).style]
    return {"flank": t["flank"], "top": t["top"]}


# --------------------------------------------------------------------------- #
#  the span limits                                                             #
# --------------------------------------------------------------------------- #
#: the top wing's physical limit, as a multiple of the body's width
TOP_SPAN_OVER_WIDTH = 1.2
#: 'unlimited' lets a span go this many times past the physical limit
UNLIMITED_FACTOR = 3.0
WING_LIMITS = ("real", "unlimited")
#: floating-point slack on "past the limit": a span typed AT the limit is legal
LIMIT_TOL = 1e-6


def span_limit(role: str, car=None, h: float | None = None) -> float:
    """The PHYSICAL maximum span of a `role` wing on `car`: the owner's rule.

    flank: 2 (h - ground), h the mount height (the slot default when None);
    top:   1.2 x the body's width, wherever it is mounted.
    Never floored or capped: `span_ceiling` is what an editor offers."""
    b = body(car)
    if role == "top":
        return TOP_SPAN_OVER_WIDTH * b.width
    if h is None:
        h = slot_defaults(car)["flank"][1]
    return 2.0 * (float(h) - b.ground)


def span_ceiling(role: str, car=None, h: float | None = None,
                 unlimited: bool = False) -> float:
    """The largest span an editor (the garage's span row, the optimiser's
    band) may offer: the physical limit, times `UNLIMITED_FACTOR` when the
    player has chosen Unlimited. Floored at the packaging band's lower end +
    0.05 so the band is never empty -- `flank_h_band` keeps every legal mount
    above that floor, so on a real slot the floor never binds."""
    from .aero.wing import BOUNDS
    lim = span_limit(role, car, h) * (UNLIMITED_FACTOR if unlimited else 1.0)
    lo = BOUNDS[role if role in BOUNDS else "flank"]["span"][0]
    return max(lim, lo + 0.05)


def area_ceiling(role: str, car=None, h: float | None = None,
                 unlimited: bool = False) -> float:
    """The reference-area band's top for that span ceiling: the packaging
    band's area, grown in proportion when the span ceiling is past the
    packaging band's span, so a bus-sized span is not forced into a Corsa's
    area (a 3 m wing at 1 m^2 would be a 0.33 m chord ribbon)."""
    from .aero.wing import BOUNDS
    b = BOUNDS[role if role in BOUNDS else "flank"]
    s_hi = span_ceiling(role, car, h, unlimited)
    return b["area"][1] * max(1.0, s_hi / b["span"][1])


def _slots_of(build) -> dict:
    """{'left'|'right'|'top': (wing name, h)} from a CarBuild or its JSON."""
    out = {}
    if isinstance(build, dict):
        sl = build.get("slots", {}) if isinstance(build.get("slots"), dict) else {}
        for k in ("left", "right", "top"):
            v = sl.get(k) if isinstance(sl.get(k), dict) else {}
            try:
                out[k] = (str(v.get("wing", "") or ""), float(v.get("h", 0.0)))
            except (TypeError, ValueError):
                out[k] = ("", 0.0)
        return out
    for k in ("left", "right", "top"):
        s = getattr(build, k, None)
        out[k] = ((getattr(s, "wing", "") or "", float(getattr(s, "h", 0.0)))
                  if s is not None else ("", 0.0))
    return out


def over_limits(build, lib, car=None) -> list[dict]:
    """The wings of `build` (a CarBuild or its JSON) that are past `car`'s
    physical limit, as [{'slot', 'wing', 'span', 'limit'}], slot order.
    Empty = an official build. `lib` is the wing library (`.wings` by name);
    a wing it does not hold is skipped (nothing is fitted there)."""
    wings = getattr(lib, "wings", None) or {}
    out = []
    for key, (name, h) in _slots_of(build).items():
        if not name:
            continue
        spec = wings.get(name)
        if spec is None:
            continue
        role = "top" if key == "top" else "flank"
        lim = span_limit(role, car, h)
        span = float(getattr(spec, "span", 0.0))
        if span > lim + LIMIT_TOL:
            out.append(dict(slot=key, wing=name, span=span, limit=lim))
    return out


def is_unlimited(build, lib, car=None) -> bool:
    """True when any fitted wing is past `car`'s physical limit."""
    return bool(over_limits(build, lib, car))


def limits_text(over: list[dict]) -> str:
    """One line naming what is past the limit, for a tag or a hint."""
    return "; ".join(f"{o['slot']} {o['wing']} span {o['span']:.2f} m > {o['limit']:.2f} m"
                     for o in over)


# --------------------------------------------------------------------------- #
def self_check(verbose: bool = True) -> bool:
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}: {msg}")

    import cars as _cars
    from .aero.wing import BOUNDS, CLAMP_HI

    #  1. the stock cars' shells are the identity map: the drawn Corsa is its
    #  published 3.817 x 1.646 x 1.440 m, and the slot bands are the garage's
    #  old Corsa constants exactly
    bc = body("corsa")
    rep("the Corsa's body is its published exterior",
        (bc.x_front, bc.x_rear, bc.width, bc.height, bc.ground)
        == (1.7515, CAR_X_REAR, 2 * CAR_HALF_W, CAR_H, 0.15),
        f"x {bc.x_front:+.4f}..{bc.x_rear:+.4f}  w {bc.width:.3f}  h {bc.height:.3f}  "
        f"ground {bc.ground:.2f}")
    same = all(flank_h_band(k) == (0.40, 1.20) and top_x_band(k)[1] == 0.55
               and top_h_band(k)[1] == 1.85
               and slot_defaults(k) == {"flank": (0.97, 0.90), "top": (-0.90, 1.55, 6.0)}
               for k in ("corsa", "mx5", "540i"))
    rep("the three stock cars keep the garage's slot bands and defaults", same,
        "flank h 0.40-1.20, top x <= 0.55, top h <= 1.85, slots (0.97, 0.90) / (-0.90, 1.55, 6)")

    #  2. the owner's two rules, on every body
    rows = []
    bad = []
    for key in ("corsa", "mx5", "540i", "express", "bus"):
        b = body(key)
        h_lo, h_hi = flank_h_band(key)
        f_max = span_limit("flank", key, h_hi)
        t_max = span_limit("top", key)
        tip = h_hi - 0.5 * f_max
        if abs(tip - b.ground) > 1e-12 or abs(t_max - 1.2 * b.width) > 1e-12:
            bad.append(key)
        if span_ceiling("flank", key, h_lo) > 2.0 * (h_lo - b.ground) + 1e-12:
            bad.append(key + " (floor binds inside the band)")
        rows.append(f"{key} flank <= {f_max:.2f} m @ h {h_hi:.2f}, top <= {t_max:.3f} m")
    rep("flank tip at the car's ground clearance, top = 1.2 x width, per car",
        not bad, ("; ".join(rows)) if not bad else str(bad))
    rep("the Corsa's top limit is 1.2 x 1.646 m", abs(span_limit("top", "corsa") - 1.9752) < 1e-9,
        f"{span_limit('top', 'corsa'):.4f} m")
    rep("the bus's top limit is 1.2 x 2.550 m", abs(span_limit("top", "bus") - 3.06) < 1e-9,
        f"{span_limit('top', 'bus'):.4f} m")

    #  3. Unlimited fits inside the wing module's sanity clamp on every car,
    #  so an unlimited wing saved and reloaded keeps its span
    worst = {}
    for key in ("corsa", "mx5", "540i", "express", "bus"):
        for role in ("flank", "top"):
            h = flank_h_band(key)[1] if role == "flank" else top_h_band(key)[1]
            s = span_ceiling(role, key, h, unlimited=True)
            a = area_ceiling(role, key, h, unlimited=True)
            worst[role] = max(worst.get(role, (0, 0)), (s, a))
    fits = all(worst[r][0] <= CLAMP_HI[r]["span"] + 1e-9 and worst[r][1] <= CLAMP_HI[r]["area"] + 1e-9
               for r in ("flank", "top"))
    rep("every car's Unlimited ceiling fits inside WingSpec.clamp's",
        fits, f"flank span {worst['flank'][0]:.2f} / area {worst['flank'][1]:.2f}, "
              f"top span {worst['top'][0]:.2f} / area {worst['top'][1]:.2f} "
              f"(clamp {CLAMP_HI['flank']['span']}/{CLAMP_HI['flank']['area']}, "
              f"{CLAMP_HI['top']['span']}/{CLAMP_HI['top']['area']})")
    top_h = max(top_h_band(k)[1] for k in ("corsa", "mx5", "540i", "express", "bus"))
    rep("every car's top-wing height fits inside the ride clamp",
        top_h <= CLAMP_HI["top"]["ride_h"] + 1e-9, f"{top_h:.2f} <= {CLAMP_HI['top']['ride_h']}")

    #  4. over_limits: a build reads the same as its JSON, a wing at the
    #  limit is legal, a hair past it is not, and the car decides
    class _W:
        def __init__(self, span):
            self.span = span

    class _L:
        wings = {"f": _W(span_limit("flank", "corsa", 0.90)), "t": _W(2.50), "x": _W(1.0)}
    js = {"slots": {"left": {"wing": "f", "h": 0.90}, "right": {"wing": "f", "h": 0.90},
                    "top": {"wing": "", "h": 1.55}}}
    rep("a flank at exactly its limit is official", over_limits(js, _L, "corsa") == [],
        f"span {span_limit('flank', 'corsa', 0.90):.3f} at h 0.90")
    js["slots"]["left"]["h"] = 0.899
    o = over_limits(js, _L, "corsa")
    rep("1 mm lower, the same panel is past the limit",
        [d["slot"] for d in o] == ["left"], limits_text(o))
    js["slots"]["left"]["h"] = 0.90
    js["slots"]["top"]["wing"] = "t"
    o_c = [d["slot"] for d in over_limits(js, _L, "corsa")]
    o_b = [d["slot"] for d in over_limits(js, _L, "bus")]
    #  (on the bus the same 1.50 m flanks at h 0.90 ARE past its limit: its
    #  underbody is 0.28 m up, so 2 (0.90 - 0.28) = 1.24 m)
    rep("the car decides: a 2.50 m top wing is Unlimited on a Corsa, not on a bus",
        o_c == ["top"] and o_b == ["left", "right"], f"corsa {o_c}, bus {o_b}")
    js["slots"]["top"]["wing"] = "gone"
    rep("a wing the library does not hold is skipped", over_limits(js, _L, "corsa") == [], "")

    #  5. the style lookup still reads names, and keys agree with names
    agree = all(style_of(k) == style_of(_cars.CARS[k]) for k in _cars.CARS)
    rep("style by key == style by name for every registered car", agree,
        str({k: style_of(k) for k in _cars.CARS}))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
