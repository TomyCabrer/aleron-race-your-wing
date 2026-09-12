"""The car library: vehicle parameter sets the sim can be built on.

Convention lock (inherited from corsa_c.py, unchanged)
-----------------------------------------------------
Mass is EU kerb  = DIN kerb + 75 kg driver + 90% fuel.
Air density      rho = 1.2 kg/m^3.
Angles in degrees at the interface, radians internally.

"est" in a comment means NOT PUBLISHED. As in `corsa_c.py`: manufacturers
do not release axle weights, CG heights or inertias, so those are bottom-up
estimates with the stated band and are parameters to be calibrated, not data.
Every value that IS published carries its source on the same line.

The Corsa C is the default and is byte-identical to `corsa_c.CorsaC()` --
`CORSA_C` is built by copying that dataclass field by field rather than by
retyping its numbers, so the two cannot drift, and `self_check` asserts the
identity with `==` (not `isclose`) on every shared field. The entire
acceptance suite is measured on the stock Corsa C; nothing here may move it.

ONE TYRE'S WORTH OF DATA
------------------------
`tyre_data/` holds twelve `.tir` files but only FIVE load into `drive.tyre`
(`TNO_/TASS_/Siemens_car205_60R15`, `car145_70R13`, `car205_60R19`); the rest
are FITTYP=61 or divide by zero in `evaluate`. **All five carry the IDENTICAL
Magic Formula force and moment coefficient set** -- diffing them against
`TNO_car205_60R15.tir` leaves only `UNLOADED_RADIUS / WIDTH / ASPECT_RATIO /
RIM_RADIUS / VERTICAL_STIFFNESS` and file metadata. There is exactly one
tyre's worth of grip in this repository.

So a car cannot be given genuinely different tyre coefficients without
inventing Pacejka data, which this project does not do. Instead, and following
CONTRACT section 2's own rule ("Do NOT rescale FNOMIN or LFZO. Rescale
geometry only"), every car reads the validated `TNO_car205_60R15.tir` and
overrides the GEOMETRY to its own tyre size. The grip difference between a
2003 touring tyre and a modern performance tyre is carried by `mu_scale`,
which is an existing first-class `VehicleConfig` field -- a labelled
calibration, not fabricated data. See `tyre_note` on each car.
"""

from dataclasses import dataclass, field, fields, asdict

from corsa_c import CorsaC, RHO, G

# --- what the powertrain can actually drive ------------------------------
#: `powertrain.step` returns `T_drive` with RL = RR = 0 -- the driveline is
#: FRONT-WHEEL DRIVE and nothing else is implemented (CONTRACT section 3).
#: `drive_layout` is therefore DATA, not behaviour: a 'rwd' car drives its
#: front wheels until somebody implements the rear split. Flagged in the
#: handoff note; the field exists so that work has somewhere to land.
LAYOUTS = ("fwd", "rwd", "awd")

TYRE_DIR = "tyre_data"
#: the one coefficient set (CONTRACT section 2: this file IS the provenance of
#: `qss.TYRE`). Every car uses it and overrides the geometry.
TYRE_REF = f"{TYRE_DIR}/TNO_car205_60R15.tir"


@dataclass
class CarSpec:
    """One car. A field-for-field superset of `corsa_c.CorsaC`.

    Anything `CorsaC` has, this has under the same name and with the same
    meaning, so a `CarSpec` is a drop-in wherever the code passes `car`.
    The additions are at the bottom: the tyre geometry, the drive layout,
    the grip calibration and the provenance strings.
    """

    name: str = "Opel Corsa C 1.2 16V (2003)"

    # --- mass and geometry ---------------------------------------------
    m: float = 1010.0
    wdist_f: float = 0.61
    L: float = 2.491
    t_f: float = 1.429
    t_r: float = 1.420
    h_cg: float = 0.55

    # --- inertias -------------------------------------------------------
    Izz: float = 1200.0
    Ixx: float = 267.0
    Iyy: float = 1200.0
    m_s: float = 892.0
    m_us_f: float = 56.0
    m_us_r: float = 62.0

    # --- tyres ----------------------------------------------------------
    tyre: str = "175/65R14 82T on 5.5Jx14 ET49"
    r_roll: float = 0.283

    # --- aerodynamics ----------------------------------------------------
    Cd: float = 0.32
    A: float = 2.01
    CdA: float = 0.66
    Crr: float = 0.012

    # --- powertrain -------------------------------------------------------
    gear: tuple = (3.545, 2.143, 1.429, 1.121, 0.892)
    gear_rev: float = 3.308
    finaldrive: float = 3.94
    eta_drive: float = 0.86
    P_max: float = 55e3
    T_max: float = 110.0
    Vmax: float = 47.2
    steer_ratio: float = 16.0

    # --- suspension: EVERY VALUE HERE IS AN ESTIMATE -----------------------
    k_wheel_f: float = 17.5e3
    k_wheel_r: float = 11.06e3
    k_tyre: float = 200e3
    h_rc_f: float = 0.075
    h_rc_r: float = 0.300
    Kphi_f: float = 311.0
    Kphi_r: float = 195.0
    Kphi_tot: float = 640.0
    rollsteer_r: float = 1.0
    rollcamber_r: float = 1.0

    brakes: str = "MISSING - no disc/drum dia, pad mu or F/R torque split"
    dampers: str = "MISSING - no rates, no damping ratios"

    # --- ADDITIONS beyond CorsaC ------------------------------------------
    #: `.tir` file to read. One coefficient set exists (see module docstring);
    #: the geometry below is what actually differs between cars.
    tyre_file: str = TYRE_REF
    #: `TyreModel` geometry overrides, CONTRACT section 2 ("rescale geometry
    #: only"). R0 = rim radius + section width * aspect.
    tyre_R0: float = 0.2915
    tyre_width: float = 0.175
    tyre_aspect: float = 0.65
    tyre_rim_r: float = 0.1778
    #: Multiplier on the Corsa's WOT torque curve, fed through the EXISTING
    #: `PowertrainParams.from_car(car, power_scale)` (which scales `nm_bp` as a
    #: whole and uprates the clutch with it). 1.0 IS the Corsa.
    #:
    #: This is an APPROXIMATION and the honest limit of the feature: the
    #: 19-breakpoint torque curve is a module constant built once at import
    #: from the Corsa's engine, so a different car gets the Corsa's curve
    #: SHAPE -- peak torque at 4000 rpm, the same rev limiter -- scaled to its
    #: own peak. The MX-5 really peaks at 5000 rpm and the 540i at 3600, and
    #: neither revs like a 1.2 Corsa. Giving each car its own curve means
    #: touching `powertrain.py`, which is out of this task's scope; see the
    #: handoff note. `engine_scale` MULTIPLIES the drive's Engine setting, so
    #: the stock Corsa at Engine=stock is still exactly 1.0.
    engine_scale: float = 1.0
    #: grip calibration against the Corsa's 175/65R14 touring tyre. 1.0 IS
    #: the Corsa. This is the ONLY way this repo can express a grippier tyre,
    #: and it is a labelled scale factor, never presented as measured data.
    mu_scale: float = 1.0
    drive_layout: str = "fwd"
    tyre_note: str = "the study's own tyre; mu_scale 1.0 by definition"
    #: one-line provenance for the whole entry
    source: str = "corsa_c.py (unchanged)"

    # --- derived (identical to CorsaC) --------------------------------------
    @property
    def a(self) -> float:
        """CG to front axle, m."""
        return (1.0 - self.wdist_f) * self.L

    @property
    def b(self) -> float:
        """CG to rear axle, m."""
        return self.wdist_f * self.L

    @property
    def t(self) -> float:
        """Mean track, m."""
        return 0.5 * (self.t_f + self.t_r)

    @property
    def P_wheel(self) -> float:
        """Power at the wheels, W."""
        return self.eta_drive * self.P_max

    def copy(self, **changes) -> "CarSpec":
        d = asdict(self)
        d.update(changes)
        return CarSpec(**d)


def _from_corsa() -> CarSpec:
    """The Corsa C, copied off `corsa_c.CorsaC()` rather than retyped.

    Retyping 30 numbers into a second file is exactly how a default drifts.
    Every shared field is read from the dataclass that owns it, so the only
    way `CORSA_C` can differ from `CorsaC()` is if a field is added to one
    and not the other -- which `self_check` catches.
    """
    c = CorsaC()
    shared = {f.name for f in fields(CarSpec)} & {f.name for f in fields(CorsaC)}
    return CarSpec(**{k: getattr(c, k) for k in shared})


CORSA_C = _from_corsa()


# ======================================================================= #
#  Mazda MX-5 1.8 (NB2, 2001-2005) -- the light RWD sports car             #
# ======================================================================= #
#  Contemporary with the Corsa and its opposite in almost every respect:
#  120 kg heavier but 54 kW stronger, 226 mm shorter in the wheelbase,
#  near-neutral weight distribution instead of 61 % front, a much lower CG
#  and a far worse Cd (an open-able roadster body).
MX5_NB = CarSpec(
    name="Mazda MX-5 1.8 (NB2, 2001)",

    # DIN kerb 1065 kg (Wikipedia, "minimum options"); + 75 kg driver to reach
    # this project's EU convention. The Corsa's own pair is DIN ~935 -> EU 1010,
    # the same +75.
    m=1140.0,
    wdist_f=0.52,           # est +/-0.02  Mazda marketed "near 50:50"; the
                            # 1.8 with a driver measures ~52/48 front
    L=2.265,                # m  published (encyCARpedia; Wikipedia gives 2.270)
    t_f=1.415,              # m  published
    t_r=1.440,              # m  published
    h_cg=0.46,              # est +/-0.04  h/H = 0.37 of the 1.235 m roof

    # DI = Izz/(m*a*b) = 0.80, the same dynamic index corsa_c.py's estimate uses
    Izz=1167.0,             # est  band 1000-1350
    Ixx=310.0,              # est  Corsa's 267 scaled by sprung mass
    Iyy=1167.0,             # est
    m_s=1035.0,             # est  kg sprung
    m_us_f=50.0,            # est
    m_us_r=55.0,            # est

    tyre="195/50R15 on 6Jx15 (Sport)",      # published
    r_roll=0.2794,          # m  0.288 OD radius x 0.97

    Cd=0.36,                # published (Wikipedia, NB: "a drag coefficient of Cd=0.36")
    A=1.70,                 # est +/-0.05  0.81 x 1.680 width x 1.235 height
    CdA=0.612,              # = Cd * A; NOT independently validated against Vmax
    Crr=0.012,              # est  same rolling class as the Corsa

    # Mazda 5-speed (the mainstream NB 1.8 box; a 6-speed existed on some
    # markets) + 4.30 final drive. Ratios published.
    gear=(3.136, 1.888, 1.330, 1.000, 0.814),
    gear_rev=3.758,
    finaldrive=4.30,
    eta_drive=0.88,         # est  RWD manual: propshaft + hypoid, no CV pair
    P_max=109e3,            # W  @ 6500 rpm  published (BP-Z3, 2001-2005)
    T_max=168.0,            # Nm @ 5000 rpm  published
    Vmax=54.7,              # m/s  197 km/h published
    steer_ratio=15.0,       # est +/-1.5

    # --- suspension: EVERY VALUE HERE IS AN ESTIMATE ---------------------
    k_wheel_f=22.0e3, k_wheel_r=18.0e3, k_tyre=210e3,
    h_rc_f=0.050,           # est  double wishbone, low roll centre
    h_rc_r=0.080,           # est  double wishbone (NOT a twist beam)
    Kphi_f=340.0, Kphi_r=260.0, Kphi_tot=700.0,
    rollsteer_r=0.0,        # a double-wishbone rear does not roll-steer like
                            # the Corsa's twist beam; 0.0 is the honest default
    rollcamber_r=0.6,       # est

    tyre_file=TYRE_REF,
    tyre_R0=0.2880,         # 0.1905 rim radius + 0.195 * 0.50 section
    tyre_width=0.195,
    tyre_aspect=0.50,
    tyre_rim_r=0.1905,
    engine_scale=1.527,     # = 168/110, the Corsa's curve scaled to this car's
                            # torque peak. Shape is the Corsa's -- approximation.
    mu_scale=1.05,          # est  a 195/50R15 summer performance tyre against
                            # the Corsa's 175/65R14 touring tyre. Tyre-test
                            # literature puts that class gap at 5-12 % in dry
                            # grip; 5 % is the conservative end. CALIBRATION.
    drive_layout="rwd",
    tyre_note="195/50R15 geometry on the study's coefficients; +5 % mu_scale",
    source="Wikipedia (NB), encyCARpedia 01-mx-5-1-8-roadster; inertias/CG est",
)


# ======================================================================= #
#  BMW 540i (E39, M62TU, 1998-2003) -- the heavy fast saloon               #
# ======================================================================= #
#  The other extreme: 640 kg heavier than the Corsa and nearly four times
#  its power, on a wheelbase 340 mm longer. Interesting for this study
#  because it is the only car here whose top speed is set by a LIMITER
#  rather than by its own drag -- see self_check.
E39_540I = CarSpec(
    name="BMW 540i (E39, 1998)",

    # Claimed kerb 1705 kg (carfolio, 2002 540i); + 75 kg driver for the EU
    # convention. Wikipedia gives 1500-1845 kg across the whole E39 range.
    m=1780.0,
    wdist_f=0.51,           # est +/-0.02  BMW marketed 50:50; the V8 540i
                            # measures slightly nose-heavy
    L=2.830,                # m  published
    t_f=1.512,              # m  published
    t_r=1.526,              # m  published
    h_cg=0.55,              # est +/-0.04  h/H = 0.38 of the 1.435 m roof --
                            # the same RATIO as the Corsa, and by coincidence
                            # almost the same absolute height

    Izz=2851.0,             # est  band 2500-3200, DI = 0.80 as above
    Ixx=558.0,              # est  Corsa's scaled by sprung mass and track^2
    Iyy=2851.0,             # est
    m_s=1640.0,             # est  kg sprung
    m_us_f=65.0,            # est
    m_us_r=75.0,            # est

    tyre="235/45R17 on 8Jx17",              # published (Sport)
    r_roll=0.3120,          # m  0.3217 OD radius x 0.97

    Cd=0.30,                # est/published: BMW quoted 0.27 for the slipperiest
                            # E39; 0.30 is the figure used for the 540i
    A=2.20,                 # est +/-0.08  0.81 x 1.800 width x 1.435 height
    CdA=0.66,               # = Cd * A. Identical to the Corsa's 0.66 -- a real
                            # coincidence, not a copy: a much bigger car that
                            # is much slicker lands on the same CdA.
    Crr=0.011,              # est  slightly better than the Corsa's 0.012

    # Getrag 420G 6-speed (shared with the E39 M5), ratios published;
    # final drive 2.81 for the 540i manual (Wikipedia lists it explicitly).
    gear=(4.23, 2.53, 1.67, 1.23, 1.00, 0.83),
    gear_rev=3.75,
    finaldrive=2.81,
    eta_drive=0.88,         # est  RWD manual
    P_max=210e3,            # W  @ 5400 rpm  published (M62TUB44, 286 PS)
    T_max=440.0,            # Nm @ 3600 rpm  published
    Vmax=69.4,              # m/s  250 km/h -- an ELECTRONIC LIMITER, not a
                            # power balance. self_check reports the surplus.
    steer_ratio=17.0,       # est +/-1.5

    # --- suspension: EVERY VALUE HERE IS AN ESTIMATE ---------------------
    k_wheel_f=30.0e3, k_wheel_r=26.0e3, k_tyre=230e3,
    h_rc_f=0.060,           # est  double wishbone / strut
    h_rc_r=0.110,           # est  multilink
    Kphi_f=620.0, Kphi_r=430.0, Kphi_tot=1250.0,
    rollsteer_r=0.0,        # multilink, deliberately toe-stable in roll
    rollcamber_r=0.5,       # est

    tyre_file=TYRE_REF,
    tyre_R0=0.3217,         # 0.2159 rim radius + 0.235 * 0.45 section
    tyre_width=0.235,
    tyre_aspect=0.45,
    tyre_rim_r=0.2159,
    engine_scale=4.000,     # = 440/110. Same approximation as the MX-5's, and
                            # cruder here: a 4.4 V8's curve is nothing like a
                            # 1.2 four's, it just has the right peak.
    mu_scale=1.08,          # est  a 235/45R17 performance tyre. Same
                            # reasoning and the same caveat as the MX-5's.
    drive_layout="rwd",
    tyre_note="235/45R17 geometry on the study's coefficients; +8 % mu_scale",
    source="Wikipedia (E39), carfolio bmw-540i-96305, auto-data.net; inertias/CG est",
)


# ======================================================================= #
#  the registry -- the house pattern, as track.py does it for TRACKS       #
# ======================================================================= #
CARS = {
    "corsa": CORSA_C,
    "mx5": MX5_NB,
    "540i": E39_540I,
}
#: cycle order for the CLI and the Settings page; the Corsa is first and default
CAR_ORDER = ("corsa", "mx5", "540i")
CAR_DEFAULT = "corsa"
CAR_TITLES = {
    "corsa": "Opel Corsa C 1.2",
    "mx5": "Mazda MX-5 1.8",
    "540i": "BMW 540i",
}


def get(name: str | None = None) -> CarSpec:
    """The named car, or the Corsa. An unknown name falls back to the default
    rather than raising: this is reached from a persisted settings file, and a
    stale name there must not stop the sim from starting."""
    return CARS.get(str(name or CAR_DEFAULT), CORSA_C)


def car_name(name: str | None = None) -> str:
    return CAR_TITLES.get(str(name or CAR_DEFAULT), CAR_TITLES[CAR_DEFAULT])


# ======================================================================= #
def self_check(verbose: bool = True) -> bool:
    """The Corsa identity, then corsa_c.py's own two cross-checks per car."""
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}: {msg}")

    # --- 1. the default is the Corsa, field for field ---------------------
    c = CorsaC()
    shared = sorted({f.name for f in fields(CarSpec)} & {f.name for f in fields(CorsaC)})
    bad = [k for k in shared if getattr(CORSA_C, k) != getattr(c, k)]
    rep("CORSA_C is corsa_c.CorsaC() on every shared field",
        not bad, f"{len(shared)} fields compared with ==, {len(bad)} differ"
                 + (f": {bad}" if bad else ""))
    missing = sorted({f.name for f in fields(CorsaC)} - {f.name for f in fields(CarSpec)})
    rep("CarSpec is a superset of CorsaC", not missing,
        f"CorsaC fields absent from CarSpec: {missing or 'none'}")
    rep("derived properties agree",
        (CORSA_C.a, CORSA_C.b, CORSA_C.t, CORSA_C.P_wheel) == (c.a, c.b, c.t, c.P_wheel),
        f"a {CORSA_C.a:.4f}  b {CORSA_C.b:.4f}  t {CORSA_C.t:.5f}  "
        f"P_wheel {CORSA_C.P_wheel / 1e3:.1f} kW")
    rep("the default name resolves to the Corsa",
        get() is CORSA_C and get(None) is CORSA_C and get("nonsense") is CORSA_C
        and CAR_ORDER[0] == CAR_DEFAULT, f"CAR_ORDER {CAR_ORDER}")

    # --- 2. corsa_c.self_check's two cross-checks, on every car -----------
    #  Any parameter set that fails these is wrong -- corsa_c.py's own words.
    if verbose:
        print("\n  gearing and top-speed power balance (corsa_c.self_check, per car)")
    for key in CAR_ORDER:
        car = CARS[key]
        ratio = car.gear[-1] * car.finaldrive
        kmh_per_1000 = (1000.0 / ratio) * 2 * 3.141592653589793 * car.r_roll * 60 / 1000
        rpm_at_vmax = car.Vmax * 3.6 / kmh_per_1000 * 1000
        F = 0.5 * RHO * car.CdA * car.Vmax ** 2 + car.Crr * car.m * G
        kW_need = car.Vmax * F / 1e3
        kW_have = car.P_wheel / 1e3
        # a top gear that cannot reach Vmax below the limiter, or a Vmax the
        # engine cannot push, is a broken parameter set. The 540i is allowed
        # a surplus because its 250 km/h is an electronic limit.
        limited = key == "540i"
        gear_ok = 1500.0 < rpm_at_vmax < 7200.0
        pwr_ok = (kW_have >= kW_need * 0.98) if not limited else (kW_have > kW_need)
        rep(f"{key:6s} gearing + power balance", gear_ok and pwr_ok,
            f"{kmh_per_1000:5.1f} km/h/1000rpm in top, {rpm_at_vmax:5.0f} rpm at Vmax; "
            f"{kW_need:5.1f} kW needed vs {kW_have:5.1f} kW available"
            + ("  (limiter-set Vmax, surplus expected)" if limited else ""))

    # --- 3. the tyre each car names really loads and evaluates sanely -----
    if verbose:
        print("\n  tyre files")
    try:
        from drive.tyre import TyreModel
    except Exception as exc:                       # pragma: no cover
        rep("drive.tyre importable", False, str(exc))
        TyreModel = None
    if TyreModel is not None:
        for key in CAR_ORDER:
            car = CARS[key]
            try:
                t = TyreModel(car.tyre_file, R0=car.tyre_R0, width=car.tyre_width)
                fz = 4000.0
                mu = t.mu_y(fz)
                peak = t.peak_fy(fz)
                fx, fy, mz = t.evaluate(fz, 0.0, 0.15)
                zero = t.evaluate(fz, 0.0, 0.0)
                #  PURE-slip oddness only. The model is deliberately NOT odd
                #  under COMBINED slip: `RHX1 RHY1 RHY2` (the SHyk / SHxa
                #  shifts) are kept on purpose per CONTRACT section 2, and
                #  `drive/tyre.py` already reports that this contradicts the
                #  same section's "exactly odd to 1e-9" claim -- it proves in
                #  self_check step 4 that no shift was MISSED. Measured here:
                #  pure long 2.4e-5 relative, pure lateral 0.70 %, combined
                #  (k 0.05, a 0.10) 8.5 %. Not this module's finding to fix.
                lat_o, lat_p = t.evaluate(fz, 0.0, -0.10), t.evaluate(fz, 0.0, 0.10)
                lon_o, lon_p = t.evaluate(fz, -0.05, 0.0), t.evaluate(fz, 0.05, 0.0)
                lat_res = abs(lat_o[1] + lat_p[1]) / abs(lat_p[1])
                lon_res = abs(lon_o[0] + lon_p[0]) / abs(lon_p[0])
                sane = (0.6 < mu < 1.4 and abs(peak - mu * fz) < 0.01 * mu * fz
                        and fy < 0.0 and zero == (0.0, 0.0, 0.0)
                        and lat_res < 0.01 and lon_res < 1e-3)
                rep(f"{key:6s} {car.tyre_file.split('/')[-1]}", sane,
                    f"R0 {car.tyre_R0:.4f} w {car.tyre_width:.3f}  mu_y(4000) {mu:.4f}  "
                    f"peak_fy {peak:7.1f} N == mu*Fz  zero exact  pure-slip odd to "
                    f"{lat_res * 100:.2f} % lat / {lon_res * 1e2:.4f} % long")
            except Exception as exc:
                rep(f"{key:6s} {car.tyre_file.split('/')[-1]}", False,
                    f"{type(exc).__name__}: {exc}")
        # the finding that shaped this module, asserted so it cannot rot
        base = TyreModel(TYRE_REF)
        same = TyreModel(f"{TYRE_DIR}/car145_70R13.tir")
        rep("the .tir files differ in GEOMETRY only (one coefficient set)",
            base.mu_y(4000.0) == same.mu_y(4000.0)
            and base.evaluate(4000.0, 0.05, 0.10) == same.evaluate(4000.0, 0.05, 0.10),
            "car145_70R13 gives identical Fx/Fy/Mz to TNO_car205_60R15 -- "
            "which is why mu_scale, not a different file, carries grip")

    # --- 4. the library table --------------------------------------------
    if verbose:
        print(f"\n  {'car':24s} {'m':>6s} {'L':>6s} {'%f':>5s} {'kW':>5s} "
              f"{'Nm':>5s} {'CdA':>5s} {'mu':>5s} {'eng':>5s} {'drv':>4s} {'gears':>6s}")
        for key in CAR_ORDER:
            c2 = CARS[key]
            print(f"  {c2.name[:24]:24s} {c2.m:6.0f} {c2.L:6.3f} {100 * c2.wdist_f:5.0f} "
                  f"{c2.P_max / 1e3:5.0f} {c2.T_max:5.0f} {c2.CdA:5.2f} {c2.mu_scale:5.2f} {c2.engine_scale:5.2f} "
                  f"{c2.drive_layout:>4s} {len(c2.gear):6d}")
        print("\n  NOTE: powertrain.py is FWD-only (T_drive has RL = RR = 0), so the two"
              "\n  RWD cars drive their front wheels. drive_layout records the intent.")
        print("  ALL PASS" if ok else "  FAILURES ABOVE")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
