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
#: `drive_layout` is BEHAVIOUR: `PowertrainParams.from_car` reads it and
#: `powertrain.step` sends the torque to that pair, so a 'rwd' car really
#: drives its rear wheels and really gains traction under acceleration where a
#: 'fwd' car loses it. **'awd' is REFUSED with a ValueError**, not quietly
#: treated as one of the other two: this driveline has one clutch, one gearbox
#: and one open diff, and a centre differential with a torque split is physics
#: it does not have.
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

    # --- engine curve anchors --------------------------------------------
    #: The five numbers `powertrain.engine_curve` needs to place this engine's
    #: WOT torque curve. `T_max` @ `n_peak_torque` and `P_max` @ `n_peak_power`
    #: are the PUBLISHED pair and the built curve passes through both exactly;
    #: everything between them is shape, and the shape family is the Corsa's
    #: own validated NA curve re-anchored (see `powertrain.engine_curve`).
    #: `displacement` sets the motoring/overrun torque, which goes as
    #: FMEP*Vd/(4*pi), so it is a real per-engine quantity and not a guess.
    n_peak_torque: float = 4000.0   # rpm, published
    n_peak_power: float = 5600.0    # rpm, published
    n_idle: float = 850.0           # est band 780-900, warm, no load
    n_cut: float = 6200.0           # est band 6100-6400 (quoted for the Z12XE)
    displacement: float = 1.199e-3  # m^3  Z12XE 1199 cc, published

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

    # --- brakes: GEOMETRY, from which powertrain derives the torque ---------
    #: `corsa_c.brakes` says MISSING and it still is -- no manufacturer
    #: publishes pad mu or a torque split. What IS published, or at least
    #: widely documented, is the HARDWARE: disc and drum diameters and whether
    #: the rear is a disc or a drum. `powertrain.brake_coeffs(car)` turns that
    #: into N.m/Pa with the same formulas the BRAKE BLOCK uses for the Corsa
    #: (pad mu 0.38, two pad faces, drum C* 1.9), so what is estimated stays
    #: estimated in exactly one place.
    #:
    #: `brk_r_eff` is the effective radius, taken as
    #: (disc OD - pad height)/2 with a 40 mm pad, which reproduces the Corsa's
    #: 0.093 m from its 236 mm disc.
    brk_front_d: float = 0.236      # m  236 x 20 mm vented, documented
    brk_rear_d: float = 0.200       # m  200 mm drum, documented
    brk_rear_disc: bool = False     # False = drum (leading-trailing, C* 1.9)
    brk_piston_d: float = 0.0540    # m  est, front caliper piston
    brk_wc_d: float = 0.01905       # m  est, rear wheel cylinder (3/4 in)
    #: hand-wheel lock-to-lock, turns. With `steer_ratio` this gives the
    #: ROAD-WHEEL lock, which is what the physics and the steer aid use:
    #: lock_deg = turns*360/2/steer_ratio. The Corsa's 2.9 turns at 16.0:1
    #: reproduces the 32.625 deg the code has always used.
    steer_turns: float = 2.900      # est +/-0.1 (derived with steer_ratio 16.0)

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
    #: RETIRED, and 1.0 on every car. It used to be `T_max/110`, a bodily
    #: multiplier on the Corsa's torque curve fed through `power_scale`, and
    #: it was the honest limit of the car library: it got the magnitude right
    #: and the SHAPE wrong. `powertrain.engine_curve(car)` now builds each
    #: engine's own curve from the anchors above, so the multiplier is not
    #: only unnecessary but would DOUBLE-COUNT. The field survives because
    #: removing it silently would let an old caller pass `power_scale` twice;
    #: anything reading it now gets exactly 1.0, which restores
    #: CONTRACT reconciliation 9's original meaning -- `power_scale` is once
    #: again purely the driver's Engine setting and nothing else.
    engine_scale: float = 1.0
    #: grip calibration against the Corsa's 175/65R14 touring tyre. 1.0 IS
    #: the Corsa. This is the ONLY way this repo can express a grippier tyre,
    #: and it is a labelled scale factor, never presented as measured data.
    mu_scale: float = 1.0
    drive_layout: str = "fwd"
    tyre_note: str = "the study's own tyre; mu_scale 1.0 by definition"
    #: one-line provenance for the whole entry
    source: str = "corsa_c.py (unchanged)"

    # --- PER-CAR PHYSICS THE THREE STOCK CARS DO NOT USE (task 41) ----------
    #  A 12 m city bus broke six things that had quietly been the Corsa's on
    #  every car (the probe notes are in .handoff; each field names what it
    #  fixes). Every field below DEFAULTS TO TODAY'S BEHAVIOUR -- None means
    #  "the module constant", 1.0 / False / "" mean "as before" -- and the
    #  Corsa, the MX-5 and the 540i set none of them except `vmax_by`, which
    #  is bookkeeping for `self_check` and never reaches the physics. So the
    #  three stock cars are bit-for-bit what they were (vehicle.py asserts
    #  the Corsa at import; `self_check` below asserts all three).
    #:
    #: TYRE LOAD SCALE per axle, the Magic Formula's own LFZO (drive/tyre.py,
    #: "Load scaling"). 1.0 IS the file tyre, and CONTRACT section 2's "do
    #: not rescale LFZO" holds for every car at 1.0. A truck tyre is rated
    #: 6-12x a car tyre and cannot be the file's tyre at the file's load, so
    #: it is the same coefficient set on a tyre rated lambda times higher:
    #:     lambda = (published rated load, N) x 0.86 / FNOMIN 4000 N
    #: -- the 0.86 is this repo's own nominal-to-rated ratio (tyre.py: "a
    #: 175/65R14 82T is 4660 N max, so 4000 N is 86 %"). A TWIN pair of rear
    #: tyres is modelled as ONE tyre at twice a single's dual-rated lambda.
    tyre_lfzo_f: float = 1.0
    tyre_lfzo_r: float = 1.0
    #: front share of the lateral load transfer. None = `VehicleConfig.
    #: roll_dist_f`, the study's 0.74 CALIBRATION, which every car used to
    #: get. On a rear-engined bus (36 % front) 0.74 lifts the inner front at
    #: wdist_f*t/(2*0.74*h) = 0.44 g, so a car whose layout is that far from
    #: the Corsa's declares its own (est, and labelled so).
    roll_dist_f: float | None = None
    #: front compliance steer, rad per N of front-axle side force. None =
    #: `VehicleConfig.eps_f` (the Corsa's 5.40e-6). A per-N number cannot
    #: carry to an axle ten times as heavy: 11 deg of compliance steer at
    #: 0.6 g on the bus. Scaled by front-axle load it is the same degrees
    #: per g as the Corsa's.
    eps_f: float | None = None
    #: engine, front-wheel and rear-wheel rotational inertias, kg m^2. None
    #: = powertrain's Corsa estimates (0.16 / 0.76 / 0.73). A truck wheel on
    #: a Corsa's 0.73 kg m^2 is a 1 kHz spin mode at h*wn 1.17: the brakes
    #: chatter and the car cannot stop (measured, the probe). The clutch's
    #: numerical compliance K_c/C_c scales with `I_eng` so the locked mode
    #: keeps its 11.3 Hz and its damping ratio.
    I_eng: float | None = None
    I_wf: float | None = None
    I_wr: float | None = None
    #: scale the rest of the rev-range constants with the engine: the soft
    #: limiter band (120 rpm), the brake-downshift line (2200), stall /
    #: crank / fire speeds and the starter torque (with displacement). Off,
    #: they are the Corsa's absolute numbers on every car, and a diesel that
    #: cuts at 2500 rpm sticks in 2nd for ever 2 rpm under its own upshift
    #: point (measured). The three stock cars keep them off.
    rev_scaled: bool = False
    #: road-speed governor, m/s; 0.0 = none. A bus's is a legal fitment (EU
    #: speed limitation devices, 92/6/EEC) and its operator's setting.
    v_governor: float = 0.0
    #: AIR brakes. `brk_piston_d` / `brk_wc_d` are then the front / rear
    #: brake CHAMBER effective diameters, `brk_lever` the caliper's lever
    #: ratio, and "line pressure" is chamber pressure: the air-brake
    #: EQUIVALENT of the hydraulic formula, labelled as such. False / 1.0 is
    #: the hydraulic caliper every car had.
    brk_air: bool = False
    brk_lever: float = 1.0
    #: the rear pressure law: 'fixed' is the Corsa's fixed reducing valve
    #: (30 bar knee, 0.30 slope, absolute) on every car, as before; 'scaled'
    #: the same valve with its knee scaled by this car's full-pedal pressure;
    #: 'none' no valve -- the axle split is the actuator sizes alone, which
    #: is what an EBS's load-dependent distribution settles to on one load.
    brk_valve: str = "fixed"
    #: WHAT LIMITS THE PUBLISHED TOP SPEED, for `self_check`'s power
    #: balance: '' = judged there from the gearing (drag or gearing),
    #: 'limiter' = an electronic limiter or a governor. Bookkeeping only --
    #: nothing in the physics reads it. It replaces a `key == "540i"` test.
    vmax_by: str = ""
    #: the driving aids and the scripted drivers read THIS car's wheelbase,
    #: grip, understeer and steering lock instead of the Corsa's calibration
    #: (input.py's steer aid, drive.py's PathFollower). False on the three
    #: stock cars, whose aided and scripted laps are frozen numbers.
    own_aids: bool = False

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
    A=1.68,                 # est +/-0.05  = 0.81 x 1.680 width x 1.235 height
    CdA=0.605,              # = Cd * A. NOT validated against Vmax, and it
                            # CANNOT be -- see the note below on why 208 km/h
                            # is a gearing limit, not an aerodynamic one.
    Crr=0.012,              # est  same rolling class as the Corsa

    # Mazda 5-speed (the mainstream NB 1.8 box; a 6-speed existed on some
    # markets) + 4.30 final drive. Ratios published.
    gear=(3.136, 1.888, 1.330, 1.000, 0.814),
    gear_rev=3.758,
    finaldrive=4.30,
    eta_drive=0.88,         # est  RWD manual: propshaft + hypoid, no CV pair
    P_max=109e3,            # W  @ 6500 rpm  published (BP-Z3, 2001-2005)
    T_max=168.0,            # Nm @ 5000 rpm  published
    #  The 2001-2005 BP-Z3 with VVT (Wikipedia's NB table: 109 kW @ 6500,
    #  168 N.m @ 5000). NOTE the earlier non-VVT NB1 BP-Z3 is quoted
    #  elsewhere as 162 N.m @ 4500 / 104 kW @ 6500 -- a different engine
    #  state, not a contradiction; this car is the NB2.
    n_peak_torque=5000.0,   # rpm, published
    n_peak_power=6500.0,    # rpm, published
    n_idle=800.0,           # est band 750-850
    n_cut=7000.0,           # est band 6900-7200; the published redline is 7000
    displacement=1.839e-3,  # m^3  1839 cc, published
    Vmax=57.8,              # m/s  208 km/h published (auto-data.net, NB2 1.8
                            # 146 hp, 5-speed; the 6-speed car is quoted 214).
                            # This is a GEARING limit, not a drag limit: 5th x
                            # 4.30 gives 30.1 km/h per 1000 rpm, so 208 km/h is
                            # 6910 rpm against a 7000 rpm cut. See self_check.
    steer_ratio=15.0,       # est +/-1.5

    # --- suspension: DERIVED from published spring rates and ARB diameters
    #  Springs (NB2 Sport, OEM): 168 lb/in front, 130 lb/in rear = 29.4 /
    #  22.8 N/mm. Widely reproduced; the base NB is 118/162 lb/in, a different
    #  and softer-front car -- this entry is the Sport, as its 195/50R15 on
    #  6Jx15 says. MOTION RATIO 0.80 both ends is `est` (+/-0.05): the
    #  double-wishbone spring sits fairly outboard on the lower arm. Wheel
    #  rate = k_spring * MR^2.
    k_wheel_f=18.8e3,       # = 29.4 N/mm * 0.80^2
    k_wheel_r=14.6e3,       # = 22.8 N/mm * 0.80^2
    k_tyre=210e3,
    h_rc_f=0.050,           # est  double wishbone, low roll centre
    h_rc_r=0.080,           # est  double wishbone (NOT a twist beam)
    #  Kphi_f/Kphi_r are springs only, 0.5*k_wheel*t^2 (the same relation that
    #  reproduces the Corsa's 311/195 from its own wheel rates and tracks).
    Kphi_f=329.0, Kphi_r=263.7,
    #  Kphi_tot adds the OEM anti-roll bars: 22 mm front, 12 mm rear
    #  (documented for the NB; the 24/16 pair sold as an upgrade is
    #  aftermarket). The bar's torsional contribution goes as d^4*t^2 and the
    #  remaining geometry (arm length, bar length) is NOT published, so the
    #  constant is CALIBRATED ON THE CORSA -- the one car here with a stated
    #  ARB-inclusive Kphi_tot (640 against 506.5 springs-only, i.e. 133.5 from
    #  a 20 mm front bar) -- and applied to these diameters. est, and the only
    #  unsourced step in the chain.
    Kphi_tot=801.9,         # 329.0 + 263.7 springs + 191.7 + 17.6 bars
    rollsteer_r=0.0,        # a double-wishbone rear does not roll-steer like
                            # the Corsa's twist beam; 0.0 is the honest default
    rollcamber_r=0.6,       # est

    #  NB2 Sport: 255 mm vented front discs, 251 mm solid rear DISCS (the
    #  MX-5 has had four-wheel discs since the NA). Documented, widely.
    brk_front_d=0.255,
    brk_rear_d=0.251,
    brk_rear_disc=True,
    brk_piston_d=0.0540,    # est, same class of single-piston sliding caliper
    brk_wc_d=0.0349,        # m  est: 34.9 mm rear caliper piston, not a
                            #    wheel cylinder -- brake_coeffs reads it as the
                            #    rear piston when brk_rear_disc is True
    steer_turns=2.600,      # est +/-0.1; with steer_ratio 15.0 -> 31.2 deg
    tyre_file=TYRE_REF,
    tyre_R0=0.2880,         # 0.1905 rim radius + 0.195 * 0.50 section
    tyre_width=0.195,
    tyre_aspect=0.50,
    tyre_rim_r=0.1905,
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
    A=2.09,                 # est +/-0.08  = 0.81 x 1.800 width x 1.435 height
    CdA=0.627,              # = Cd * A. NOT validated against Vmax and CANNOT
                            # be: the 540i's 250 km/h is an electronic
                            # limiter. Its 6th gear reaches 322.8 km/h at the
                            # rev cut, so nothing about its top speed is
                            # aerodynamic.
    Crr=0.011,              # est  slightly better than the Corsa's 0.012

    # Getrag 420G 6-speed (shared with the E39 M5), ratios published;
    # final drive 2.81 for the 540i manual (Wikipedia lists it explicitly).
    gear=(4.23, 2.53, 1.67, 1.23, 1.00, 0.83),
    gear_rev=3.75,
    finaldrive=2.81,
    eta_drive=0.88,         # est  RWD manual
    P_max=210e3,            # W  @ 5400 rpm  published (M62TUB44)
    T_max=440.0,            # Nm @ 3600 rpm  published
    #  M62TUB44: 4398 cc, 210 kW @ 5400, 440 N.m @ 3600 (Wikipedia, BMW M62).
    #  The TU added inlet VANOS, which is why the real curve is flatter than
    #  the M60's -- and the re-anchored Corsa shape happens to fit it almost
    #  exactly (the power-anchor correction below is 0.990, i.e. 1 %).
    n_peak_torque=3600.0,   # rpm, published
    n_peak_power=5400.0,    # rpm, published
    n_idle=650.0,           # est band 600-700, a big V8 idles low
    n_cut=6400.0,           # est band 6300-6500; not published for the TU
    displacement=4.398e-3,  # m^3  4398 cc, published
    Vmax=69.4,              # m/s  250 km/h -- an ELECTRONIC LIMITER, not a
                            # power balance. self_check reports the surplus.
    steer_ratio=17.0,       # est +/-1.5

    # --- suspension: DERIVED from published spring rates and ARB diameters
    #  Springs (540i M-Sport): 167 lb/in front, 197 lb/in rear = 29.2 /
    #  34.5 N/mm (the rear is progressive, so its rate is the working-range
    #  value). MOTION RATIO `est`: 0.98 front (strut, spring on the damper
    #  axis) and 0.65 rear (+/-0.07; the E39's multilink spring is well
    #  inboard). The rear MR is the weakest number in this entry and it is
    #  what makes the roll gradient below look soft -- see self_check.
    k_wheel_f=28.1e3,       # = 29.2 N/mm * 0.98^2
    k_wheel_r=14.6e3,       # = 34.5 N/mm * 0.65^2
    k_tyre=230e3,
    h_rc_f=0.060,           # est  strut
    h_rc_r=0.110,           # est  multilink
    Kphi_f=560.4, Kphi_r=296.2,          # springs only, 0.5*k_wheel*t^2
    #  + anti-roll bars: 26 mm front, 14 mm rear (documented for the pre-2003
    #  540i Sport; the base car's rear bar is 13 mm and the 2003 Sport's 15).
    #  Same Corsa-calibrated d^4*t^2 constant as the MX-5 above.
    Kphi_tot=1320.1,        # 560.4 + 296.2 springs + 427.0 + 36.6 bars
    rollsteer_r=0.0,        # multilink, deliberately toe-stable in roll
    rollcamber_r=0.5,       # est

    #  E39 540i: 325 x 28 mm vented front, 320 x 20 mm vented rear discs --
    #  the V8 car's own bigger brakes, documented for the 540i/M5 chassis.
    brk_front_d=0.325,
    brk_rear_d=0.320,
    brk_rear_disc=True,
    brk_piston_d=0.0600,    # est, band 0.057-0.064 for a 325 mm disc
    brk_wc_d=0.0420,        # m  est, rear caliper piston
    steer_turns=3.000,      # est +/-0.1; with steer_ratio 17.0 -> 31.8 deg
    tyre_file=TYRE_REF,
    tyre_R0=0.3217,         # 0.2159 rim radius + 0.235 * 0.45 section
    tyre_width=0.235,
    tyre_aspect=0.45,
    tyre_rim_r=0.2159,
    mu_scale=1.08,          # est  a 235/45R17 performance tyre. Same
                            # reasoning and the same caveat as the MX-5's.
    drive_layout="rwd",
    vmax_by="limiter",      # the 250 km/h electronic limiter (see Vmax)
    tyre_note="235/45R17 geometry on the study's coefficients; +8 % mu_scale",
    source="Wikipedia (E39), carfolio bmw-540i-96305, auto-data.net; inertias/CG est",
)


# ======================================================================= #
#  Renault Express 1.4 (E7J, 1994-1997) -- the tall van                    #
# ======================================================================= #
#  Task 41: a car to put bigger wings on. The OLDER, Renault 5-based
#  Express (1985-2000; Extra in the UK, Rapid in German-speaking markets),
#  the owner's choice over the Kangoo that replaced it. The engine is the
#  E7J "Energy" 1.4 of the 1991-1997 cars, and it is chosen over the older
#  C3J/C2J pushrod 1.4s for a reason that suits this study: its published
#  pair -- 55 kW at 5600 rpm, 109 N.m at 4000 -- lands on the Corsa's own
#  anchor speeds, so `engine_curve` re-anchors the Corsa's validated shape
#  with no rpm warp below the cut at all. The van is, to within a newton-
#  metre, the Corsa's engine in a 1.78 m tall box: the difference between
#  the two cars is the BODY, which is what the wings are about.
#
#  SOURCES DISAGREE on this engine's output and the conflict is stated,
#  not smoothed: Wikipedia's Express table gives the E7J 55 kW (74 hp) @
#  5600 / 109 N.m @ 4000 (1991-1997); L'argus lists the 1995 1.4e as
#  "80 ch / 55 kW" (the two halves disagree with each other); autotitre.com
#  and French Wikipedia give the 1994-1997 1.4 80 ch @ 6000 / 107 N.m @
#  4000; car.info's Swedish register gives the 1997 Express Van 1.4 M5 as
#  55 kW / 75 hp. Three of four say 55 kW, so 55 kW, with Wikipedia's rpm.
EXPRESS_14 = CarSpec(
    name="Renault Express 1.4 (E7J, 1995)",

    # "Poids a vide" 840 kg for the 1994-1997 1.4 RT (autotitre.com), taken
    # as the DIN kerb (fluids and fuel in); + 75 kg driver for the EU
    # convention. French Wikipedia's range for the whole family is 775-1245
    # kg, so 840 is a light but plausible panel van.
    m=915.0,
    wdist_f=0.60,           # est +/-0.03  an EMPTY front-drive van: the
                            # engine and the driver sit ahead of a light box
    L=2.580,                # m  published (Wikipedia; French Wikipedia agrees)
    t_f=1.326,              # m  published (French Wikipedia, "voies avant /
    t_r=1.288,              # m  arriere 1 326 / 1 288 mm")
    h_cg=0.62,              # est +/-0.05  h/H = 0.35 of the 1.776 m roof: a
                            # tall body, but an empty box on a low floor

    Izz=1170.0,             # est  DI = Izz/(m*a*b) = 0.80, as the other cars
    Ixx=245.0,              # est  the Corsa's 267 scaled by sprung mass and
                            # track^2 (203), +20 % for the taller body
    Iyy=1170.0,             # est
    m_s=805.0,              # est  kg sprung
    m_us_f=50.0,            # est  MacPherson front, 13 in wheels
    m_us_r=60.0,            # est  trailing arms on transverse torsion bars
                            # (the Renault 5's), drums

    tyre="155/80R13",       # published (autotitre.com, 1.4 RT 1994-1997)
    r_roll=0.2804,          # m  0.2891 OD radius x 0.97

    Cd=0.42,                # = CdA / A, derived, not published
    A=2.36,                 # est +/-0.10  = 0.85 x 1.566 width x 1.776 height:
                            # a box van fills more of its width x height
                            # rectangle than a car's 0.81
    CdA=0.986,              # m^2  BACK-SOLVED from the 150 km/h top speed
                            # (below) at 0.86 x 55 kW, so the self-check's
                            # drag balance holds by construction -- it does
                            # NOT validate this number, it defines it. The
                            # other published top speed, car.info's 141 km/h
                            # for the 1997 panel van, would need 1.195 (Cd
                            # 0.51), which is not a plausible van.
    Crr=0.012,              # est  the Corsa's rolling class

    # Renault JB1 5-speed family (the E7J's transaxle in the Clio and the
    # Express): ratios as quoted for the JB1 (cliosport.net). The Express's
    # own final drive is NOT in any source reached; 4.500 is the JB1's
    # quoted pair (est, band 4.07-4.50). 5th is an overdrive: 150 km/h is
    # 5076 rpm, below the 5600 power peak, as a 1990s van's was.
    gear=(3.727, 2.048, 1.321, 0.967, 0.795),
    gear_rev=3.545,         # est  JB1
    finaldrive=4.500,       # est  see above
    eta_drive=0.86,         # est  the Corsa's FWD manual class
    P_max=55e3,             # W  @ 5600 rpm  published (Wikipedia, E7J)
    T_max=109.0,            # Nm @ 4000 rpm  published (Wikipedia; autotitre
                            # 10.9 mkg = 107 N.m @ 4000)
    n_peak_torque=4000.0,   # rpm, published
    n_peak_power=5600.0,    # rpm, published
    n_idle=800.0,           # est band 750-850
    n_cut=6000.0,           # est band 5800-6300; not published for the E7J
    displacement=1.390e-3,  # m^3  1390 cc, published (75.8 x 77 mm)
    Vmax=41.67,             # m/s  150 km/h published (autotitre.com, 1.4 RT
                            # 1994-1997). A DRAG limit: 5th reaches 177 km/h
                            # at the cut.
    steer_ratio=20.0,       # est +/-2  no power steering on the 1.4
                            # (L'argus: "Direction assistee NON"), so a slow rack
    steer_turns=3.8,        # est: -> 34.2 deg, which with L 2.580 reproduces
                            # the published 10.4 m turning circle (French
                            # Wikipedia, "rayon de braquage") at the outer front
                            # wheel within the rack's own Ackermann spread

    # --- suspension: EVERY VALUE HERE IS AN ESTIMATE. Pseudo-MacPherson
    #  front, trailing arms on torsion bars behind (French Wikipedia).
    k_wheel_f=16.0e3,       # est
    k_wheel_r=18.0e3,       # est  a van's rear is sprung for its payload
    k_tyre=190e3,           # est
    h_rc_f=0.080,           # est  strut
    h_rc_r=0.050,           # est  trailing arms: the roll centre is near the
                            # ground
    Kphi_f=245.5, Kphi_r=260.6,          # springs only, 0.5*k_wheel*t^2
    Kphi_tot=700.0,         # est  with the bars: ~6 deg/g, a van rolls more
                            # than the Corsa's 4.7-5.1
    rollsteer_r=0.0,        # trailing arms do not roll-steer
    rollcamber_r=1.0,       # est  the wheel leans with the body

    #  Renault 5-family brakes: solid front discs, rear drums (French
    #  Wikipedia: "AV : Disques / AR : Tambours"). Sizes est.
    brk_front_d=0.238,      # m  est
    brk_rear_d=0.180,       # m  est
    brk_rear_disc=False,
    brk_piston_d=0.0480,    # m  est
    brk_wc_d=0.01746,       # m  est: an 11/16 in wheel cylinder. With the
                            # Corsa's fixed valve and a 3/4 in or larger one
                            # the EMPTY van's light rear (40 %) locks FIRST
                            # (measured: 70 bar rear vs 81 front); the real
                            # vans had a load-sensing limiter for that
    tyre_file=TYRE_REF,
    tyre_R0=0.2891,         # 0.1651 rim radius + 0.155 * 0.80 section
    tyre_width=0.155,
    tyre_aspect=0.80,
    tyre_rim_r=0.1651,
    mu_scale=0.95,          # est  a narrow 1990s commercial ("C") tyre
                            # against the Corsa's 175/65R14. CALIBRATION.
    drive_layout="fwd",
    own_aids=True,          # a new car: no frozen aided lap to protect
    tyre_note="155/80R13 geometry on the study's coefficients; -5 % mu_scale",
    source="Wikipedia + fr.wikipedia (Express), autotitre.com 1.4 RT, L'argus; "
           "gearing/brakes/inertias/CG est",
)


# ======================================================================= #
#  Mercedes-Benz Citaro O530, 12 m (OM 906 hLA, 2005) -- the city bus      #
# ======================================================================= #
#  Task 41: the car for the largest wings. A 12 m low-floor city bus, the
#  owner's choice, empty but for its driver. It is modelled as a CAR with
#  four wheel stations -- the rear "wheel" on each side is the TWIN pair --
#  and it needs every new field above. Two published sources carry it:
#
#   * DaimlerChrysler press release, 10 Aug 2006 ("Erste Linienbusse
#     Mercedes-Benz Citaro mit Euro 5-Motoren ausgeliefert"): OM 906 hLA,
#     205 kW (279 PS), 1120 N.m at 1300/min, lying six in line, a 6-speed
#     automatic, disc brakes all round with ABS and ASR, independent front
#     suspension on lower wishbones with an anti-roll bar;
#   * traditionsbus.de, BVG Berlin's own Citaro O530 fleet data: 11 950 x
#     2 550 x 3 076 mm, and for the 2005 OM 906 hLA 279 PS cars (fleet nos.
#     1451-1480) Leergewicht 11 384 kg and a governed 80 km/h; dual-circuit
#     air brakes with ABS/ASR/EBS; 330 l tank.
#  and the gearbox is the ZF Ecomat 2 HP 502 C (ZF data sheet: 1100 N.m
#  city-bus rating, the 6-speed ratios below). BVG's own cars had a Voith
#  DIWA 4-speed; the press release's 6-speed is the one that fits this
#  sim's stepped box, which has no torque converter (see known gaps).
CITARO_O530 = CarSpec(
    name="Mercedes-Benz Citaro O530 12 m bus (2005)",

    # Leergewicht 11 384 kg (traditionsbus.de, BVG 1451-1480), taken as the
    # kerb with fuel in; + 75 kg driver. No passengers: the convention. If
    # that figure is DRY, 90 % of the 330 l tank would add 250 kg (+2 %).
    m=11459.0,
    wdist_f=0.36,           # est +/-0.03  rear-engined: the six sits behind
                            # the rear axle
    L=5.845,                # m  published (Mercedes-Benz Citaro data sheet)
    t_f=2.100,              # m  est  independent front axle
    t_r=1.840,              # m  est  to the centre of the twin pair
    h_cg=1.10,              # est +/-0.10  h/H = 0.36 of the 3.076 m roof:
                            # floor, axles, engine low; body and A/C high

    Izz=150000.0,           # est  band 130 000-170 000: a uniform 12 x 2.55 m
                            # slab is 142 600, the engine and axles at the ends
                            # add ~5 %
    Ixx=14000.0,            # est  band 12 000-17 000
    Iyy=150000.0,           # est
    m_s=9259.0,             # est  kg sprung
    m_us_f=700.0,           # est  two 22.5 in wheels, 430 mm discs, knuckles,
                            # half the wishbones
    m_us_r=1500.0,          # est  a portal drive axle (~900 kg), four wheels,
                            # two discs

    tyre="275/70R22.5, twin rear",       # published (Citaro standard fit)
    r_roll=0.464,           # m  0.4783 OD radius x 0.97

    Cd=0.65,                # est  band 0.55-0.80 for a flat-fronted city bus
    A=6.80,                 # est  0.95 x 2.550 width x (3.076 - 0.28) m
    CdA=4.42,               # = Cd * A. NOT validated by Vmax and it cannot
                            # be: the top speed is a governor (below).
    Crr=0.007,              # est  band 0.005-0.008, truck tyres

    # ZF Ecomat 2 6 HP 502 C: 3.43 / 2.01 / 1.42 / 1.00 / 0.83 / 0.59 (ZF
    # data sheet "HP 502 C HP 592 C HP 602 C", standard ratios). Reverse:
    # the sheet's R 11.76 includes the converter's stall ratio (its 1st is
    # quoted 8.33 = 3.43 x 2.43 on the same basis), so 11.76 / 2.43 = 4.84.
    gear=(3.43, 2.01, 1.42, 1.00, 0.83, 0.59),
    gear_rev=4.84,
    finaldrive=6.21,        # est  band 5.74-6.50, city-bus portal axles; at
                            # 6.21 the governed 80 km/h is 1676 rpm in 6th
    eta_drive=0.85,         # est  converter locked, angle drive, the portal
                            # hubs' extra gear stage
    P_max=205e3,            # W  published (press release 2006; traditionsbus)
    T_max=1120.0,           # Nm @ 1300 rpm  published (press release 2006)
    n_peak_torque=1300.0,   # rpm, published
    n_peak_power=2200.0,    # rpm  est: the OM 906 family's rated speed; the
                            # press release gives the power without it
    n_idle=600.0,           # est band 550-650
    n_cut=2500.0,           # est band 2400-2600, the governor's high idle
    displacement=6.374e-3,  # m^3  6374 cc, published (traditionsbus.de)
    Vmax=22.22,             # m/s  80 km/h, BVG's governor setting on these
                            # cars (traditionsbus.de; other batches 85/95).
    steer_ratio=20.0,       # est +/-2  power-assisted bus box
    steer_turns=4.2,        # est -> 37.8 deg at the road wheel: a 12 m bus
                            # turns a ~21 m circle, which on a 5.845 m wheel-
                            # base is a car-like 34-38 deg bicycle angle

    # --- suspension: EVERY VALUE HERE IS AN ESTIMATE. Air springs all
    #  round; wishbones and a bar in front (press release), a rigid portal
    #  axle on four links behind.
    k_wheel_f=250.0e3,      # est
    k_wheel_r=400.0e3,      # est
    k_tyre=900e3,           # est  truck tyre, ~900 N/mm
    h_rc_f=0.35,            # est  wishbones
    h_rc_r=0.75,            # est  a rigid axle's links: a high roll centre
    Kphi_f=3500.0, Kphi_r=5500.0,        # est, springs + bars, N.m/deg
    Kphi_tot=9000.0,        # est  -> 5.0 deg/g, a bus rolls
    rollsteer_r=0.0,        # rigid axle
    rollcamber_r=0.0,       # a rigid axle's wheels stay square to the road

    #  AIR DISC brakes all round (press release; traditionsbus: dual-circuit
    #  air, ABS/ASR/EBS). 430 mm discs on 22.5 in wheels est. The formula
    #  is the air-brake EQUIVALENT (brk_air): chamber area x caliper lever x
    #  pad mu x effective radius, and "line pressure" is chamber pressure.
    brk_front_d=0.430,      # m  est
    brk_rear_d=0.430,       # m  est
    brk_rear_disc=True,
    brk_air=True,
    brk_piston_d=0.1404,    # m  est: a type 24 chamber, 24 in^2 effective
    brk_wc_d=0.1282,        # m  est: type 20 EQUIVALENT -- the rear share the
                            # EBS gives an EMPTY bus, 0.83 of the front's, so
                            # the front axle locks first as ECE R13 wants
    brk_lever=15.6,         # est  air-disc caliper lever ratio, class value
    brk_valve="none",       # an EBS, not a reducing valve (see brk_valve)
    tyre_file=TYRE_REF,
    tyre_R0=0.4783,         # 0.28575 rim radius + 0.275 * 0.70 section
    tyre_width=0.275,
    tyre_aspect=0.70,
    tyre_rim_r=0.28575,
    #  275/70R22.5 148/145 (the city-bus load index pair, e.g. Michelin X
    #  InCity): 3150 kg single, 2900 kg per tyre in twin. With the 0.86
    #  nominal-to-rated ratio (see `tyre_lfzo_f`):
    tyre_lfzo_f=6.644,      # = 3150 kg x 9.81 x 0.86 / 4000 N
    tyre_lfzo_r=12.233,     # = 2 x 2900 kg x 9.81 x 0.86 / 4000 N, the pair
    #  -> the static loads, 20.2 kN front and 36.0 kN per rear pair, are 3046
    #  and 2941 N at the equivalent car load: the Corsa's own range.
    mu_scale=0.80,          # est band 0.75-0.90: a truck tyre's dry peak
                            # friction runs 20-25 % under a car tyre's (heavy-
                            # vehicle handling literature). CALIBRATION.
    drive_layout="rwd",
    roll_dist_f=0.45,       # est: wishbones + bar in front, a stiff air-sprung
                            # rigid axle behind; the inner front then lifts at
                            # 0.72 g instead of 0.44
    eps_f=8.06e-7,          # = 5.40e-6 x (Corsa front axle 616 kg / 4125 kg):
                            # the Corsa's compliance steer per g
    I_eng=2.0,              # est band 1.5-2.5: a 6.4 l six's crank, flywheel
                            # and converter impeller
    I_wf=13.0,              # est  275/70R22.5 on steel (tyre ~60 kg at r_g 0.40)
                            # + disc + hub; keeps R^2*CFX/I at the Corsa's
    I_wr=26.0,              # est  the twin pair + disc + portal gears
    rev_scaled=True,
    v_governor=22.22,       # m/s  the 80 km/h above
    vmax_by="limiter",
    own_aids=True,
    tyre_note="275/70R22.5 geometry on the study's coefficients, LOAD-SCALED "
              "x6.644 front / x12.233 rear twin; -20 % mu_scale",
    source="DaimlerChrysler press release 2006-08-10, traditionsbus.de (BVG), "
           "ZF HP 502 C data sheet; axle/inertia/CG/brake sizes est",
)


# ======================================================================= #
#  ADDED MASS -- ballast, passengers, and the wings the garage fitted      #
# ======================================================================= #
#  A mass slider that scales `m` and leaves the rest of the parameter set
#  alone is not a weight feature, it is a bug with a UI. 200 kg of lead in
#  the boot of a Corsa C moves the front weight fraction from 61 % to 49 %,
#  raises the CG by 17 mm and adds 44 % to `Izz`; a mass slider that only
#  touched `m` would report a car that corners harder because every tyre is
#  better loaded and nothing else changed.
#
#  So added mass is modelled as POINT MASSES at a station and a height, and
#  `with_masses` returns a new `CarSpec` in which everything a point mass
#  really moves has moved: `m`, `m_s`, `wdist_f` (hence `a` and `b`), `h_cg`,
#  and `Izz / Ixx / Iyy` through the parallel-axis theorem. Nothing else is
#  touched: the springs (`Kphi_*`, `k_wheel_*`), the roll centres, the
#  wheelbase, the track, the tyres, the gearing and `CdA` are all properties
#  of the car and not of what is in it -- which is exactly why a ballasted
#  car rolls MORE (same springs, more sprung mass, longer roll arm) and
#  accelerates less (same engine, more mass). `Vmax` and `P_max` are left as
#  the STOCK car's data: they are published figures for the car, not
#  predictions, and `drive/vehicle.py` never reads `Vmax`.
#
#  The parallel-axis term `m_b*d^2` is the honest model and it is cheap: a
#  point mass is a floor on a real object's inertia, never an overestimate.

#: kg/m^2 and kg/m are in `drive/aero/wing.py`; a fitted wing's mass comes
#: from `wing_mass(spec)` and is charged at its own slot station -- see
#: `drive/garage.py:CarBuild.mass_points`.


@dataclass(frozen=True)
class PointMass:
    """One lump of added mass, in the car's own frame.

    `x` is a STATION measured from the car's STOCK CG, positive FORWARD (the
    same sign convention as `VehicleConfig.x_w` and `wheel_positions`), so
    the front axle is at `+car.a` and the rear axle at `-car.b`. `h` is
    height above the ROAD, the same datum as `h_cg` and `h_w`. Lateral
    offset is not modelled: ballast goes on the centreline, and a wing on
    each flank is symmetric, so the only y-offset that could exist cancels.
    """

    m: float = 0.0
    x: float = 0.0
    h: float = 0.0
    label: str = ""


# --- where a driver can actually put ballast ---------------------------- #
#  Four stations that between them span both axes of the trade: `nose` vs
#  `boot` is the longitudinal one, `floor` vs `boot` is the height one at
#  nearly the same station, and `seat` is the control case that changes the
#  mass and nothing else. Heights are estimates and are the honest ones: a
#  hatchback's BOOT FLOOR is ~0.65 m above the road, well ABOVE the 0.55 m
#  CG, so a sandbag in the boot RAISES the CG. Only ballast bolted to the
#  floorpan (~0.30 m, which is what a race car does) lowers it.
BALLAST_STATIONS = ("nose", "seat", "floor", "boot")     # fore -> aft
BALLAST_DEFAULT = "floor"
BALLAST_LABELS = {
    "nose": "Nose (front subframe, low)",
    "seat": "Passenger seat (at the CG)",
    "floor": "Floorpan over the rear axle (low)",
    "boot": "Boot floor, behind the rear axle (high)",
}
BALLAST_SHORT = {"nose": "NOSE", "seat": "SEAT", "floor": "FLOOR", "boot": "BOOT"}
#: the Settings page cycles these; the CLI takes any value in [0, BALLAST_MAX]
BALLAST_KG = (0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0)
BALLAST_MAX = 300.0
#: heights, m above the road. est. `seat` is special-cased to h_cg so that
#: station is EXACTLY neutral in CG height -- the owner's own test case.
_BALLAST_H = {"nose": 0.32, "floor": 0.30, "boot": 0.65}
_BALLAST_OVERHANG = {"nose": 0.30, "boot": 0.25}   # m beyond the axle


def ballast_point(car: CarSpec, kg: float, where: str = BALLAST_DEFAULT) -> PointMass:
    """`kg` of ballast at one of `BALLAST_STATIONS`, as a `PointMass`.

    The stations are quoted relative to the AXLES, not as absolute numbers,
    so they mean the same thing on a 2.265 m MX-5 and a 2.830 m 540i.
    """
    kg = max(float(kg), 0.0)
    where = where if where in BALLAST_LABELS else BALLAST_DEFAULT
    if where == "nose":
        x, h = car.a + _BALLAST_OVERHANG["nose"], _BALLAST_H["nose"]
    elif where == "boot":
        x, h = -(car.b + _BALLAST_OVERHANG["boot"]), _BALLAST_H["boot"]
    elif where == "floor":
        x, h = -car.b, _BALLAST_H["floor"]
    else:                                   # 'seat': at the CG, at CG height
        x, h = 0.0, car.h_cg
    return PointMass(kg, x, h, f"ballast {kg:.0f} kg {BALLAST_SHORT[where]}")


#: `wdist_f` band the result is held inside. 200 kg over the nose of a Corsa
#: is 0.694 and 200 kg in its boot is 0.493, both inside; the clamp only
#: exists so that a silly CLI number cannot hand the EOM a car whose CG is
#: outside its own wheelbase.
WDIST_BAND = (0.20, 0.80)


def with_masses(car: CarSpec, masses=()) -> CarSpec:
    """`car` carrying `masses`, as a new `CarSpec`. Bit-for-bit identity when
    there is nothing to carry.

    Returns the SAME OBJECT when the added mass is zero -- not an equal copy,
    the same object. That is what keeps the stock car bit-for-bit: every
    derived quantity in `drive/vehicle.py` is scaled by a ratio against
    `CORSA_C`, and a ratio is exactly 1.0 only if the two floats are
    identical. `self_check` asserts the identity.
    """
    pts = [p for p in masses if p is not None and p.m > 0.0]
    mb = sum(p.m for p in pts)
    if not pts or mb <= 0.0:
        return car

    m0 = car.m
    m1 = m0 + mb
    # the CG moves by the first moment of the added mass about the stock CG
    dx = sum(p.m * p.x for p in pts) / m1                    # + = FORWARD
    dh = sum(p.m * (p.h - car.h_cg) for p in pts) / m1       # + = UP
    h1 = car.h_cg + dh

    # `b` is the CG-to-REAR-axle distance and `wdist_f == b/L`, so a CG that
    # moves forward by dx lengthens b and loads the front axle. L (the
    # wheelbase) is a property of the car and does not move.
    wd1 = (car.b + dx) / car.L
    lo, hi = WDIST_BAND
    wd1 = lo if wd1 < lo else (hi if wd1 > hi else wd1)

    # parallel axis about the NEW CG: the body's own inertia moves with it,
    # and each lump contributes m*d^2 about the axis in question. Yaw sees
    # the longitudinal arm, roll the vertical one, pitch both.
    Izz1 = car.Izz + m0 * dx * dx + sum(p.m * (p.x - dx) ** 2 for p in pts)
    Ixx1 = car.Ixx + m0 * dh * dh + sum(p.m * (p.h - h1) ** 2 for p in pts)
    Iyy1 = (car.Iyy + m0 * (dx * dx + dh * dh)
            + sum(p.m * ((p.x - dx) ** 2 + (p.h - h1) ** 2) for p in pts))

    return car.copy(m=m1, wdist_f=wd1, h_cg=h1,
                    Izz=Izz1, Ixx=Ixx1, Iyy=Iyy1,
                    # ballast and wings ride on the springs: all of it is
                    # SPRUNG. The unsprung masses are the hubs, and nothing
                    # a driver adds bolts to those.
                    m_s=car.m_s + mb)

# ======================================================================= #
#  the registry -- the house pattern, as track.py does it for TRACKS       #
# ======================================================================= #
CARS = {
    "corsa": CORSA_C,
    "mx5": MX5_NB,
    "540i": E39_540I,
    "express": EXPRESS_14,
    "bus": CITARO_O530,
}
#: cycle order for the CLI and the Settings page; the Corsa is first and default
CAR_ORDER = ("corsa", "mx5", "540i", "express", "bus")
CAR_DEFAULT = "corsa"
CAR_TITLES = {
    "corsa": "Opel Corsa C 1.2",
    "mx5": "Mazda MX-5 1.8",
    "540i": "BMW 540i",
    "express": "Renault Express 1.4",
    "bus": "Mercedes Citaro bus",
}
#: the three cars whose physics is frozen (every acceptance number, medal
#: and checkpoint was measured on them): `self_check` asserts they set none
#: of the task-41 physics fields
STOCK_CARS = ("corsa", "mx5", "540i")
#: the task-41 fields and the value each takes when a car does not use it
PHYSICS_DEFAULTS = dict(tyre_lfzo_f=1.0, tyre_lfzo_r=1.0, roll_dist_f=None,
                        eps_f=None, I_eng=None, I_wf=None, I_wr=None,
                        rev_scaled=False, v_governor=0.0, brk_air=False,
                        brk_lever=1.0, brk_valve="fixed", own_aids=False)


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
        # WHAT LIMITS THIS CAR'S TOP SPEED, which decides whether the
        # top-speed power balance can validate `CdA` at all:
        #   'drag'    P_available == P_required at Vmax, and Vmax is below the
        #             rev cut in top. Only then is CdA VALIDATED by Vmax.
        #   'gearing' Vmax is at (or within 2 % of) the rev cut in top gear:
        #             the engine runs out of revs before the air stops it.
        #   'limiter' an electronic speed limiter or a governor, below both.
        #             DECLARED by the car (`vmax_by`), because nothing in the
        #             numbers can tell a limiter from a power shortfall.
        n_cut = float(getattr(car, "n_cut", 6200.0))
        v_at_cut = kmh_per_1000 * n_cut / 1000.0
        v_pub = car.Vmax * 3.6
        surplus = kW_have / max(kW_need, 1e-9)
        if getattr(car, "vmax_by", "") == "limiter":
            mech = "limiter"
        elif v_pub > 0.98 * v_at_cut:
            mech = "gearing"
        else:
            mech = "drag"
        #  top gear at Vmax must be a speed the ENGINE can run at: above
        #  1.5 x idle and not past its own cut (3 % slack for the published
        #  redline vs the cut). Relative, so a 2500 rpm diesel is judged on
        #  its own range: the old absolute 1500-7200 rpm band passed the
        #  three petrol cars and would have passed a bus geared to lug.
        n_idle = float(getattr(car, "n_idle", 850.0))
        gear_ok = 1.5 * n_idle < rpm_at_vmax < 1.03 * n_cut
        # drag-limited cars must balance; the other two must have a SURPLUS
        # (if they did not, the quoted Vmax would be unreachable)
        pwr_ok = (0.98 <= surplus <= 1.06) if mech == "drag" else (surplus > 1.0)
        rep(f"{key:6s} gearing + power balance", gear_ok and pwr_ok,
            f"{kmh_per_1000:5.1f} km/h/1000rpm in top, {rpm_at_vmax:5.0f} rpm at Vmax; "
            f"{kW_need:5.1f} kW needed vs {kW_have:5.1f} kW available "
            f"(x{surplus:.2f}); top speed is {mech.upper()}-limited "
            f"({v_pub:.0f} vs {v_at_cut:.0f} km/h at the cut)"
            + ("  -> CdA IS validated" if mech == "drag"
               else "  -> CdA CANNOT be validated by Vmax"))

    # --- 3. the tyre each car names really loads and evaluates sanely -----
    if verbose:
        print("\n  tyre files")
    try:
        from drive.tyre import TyreModel
    except Exception as exc:                       # pragma: no cover
        rep("drive.tyre importable", False, str(exc))
        TyreModel = None
    if TyreModel is not None:
        from drive.tyre import tyre_for
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
            #  the tyre the car ACTUALLY runs (per axle, at its declared load
            #  scale), at its OWN static load and at the loaded outer wheel
            #  of 1 g with this car's transfer split: grip in the car-tyre
            #  band at the static load, and the loaded wheel inside that
            #  tyre's FZMAX -- the clamp that pinned an unscaled bus at 10 kN
            rd = car.roll_dist_f if car.roll_dist_f is not None else 0.74
            dfz_1g = car.m * G * car.h_cg / car.t
            rows = []
            fine = True
            for ax, lam, share, split in (("f", car.tyre_lfzo_f, car.wdist_f, rd),
                                          ("r", car.tyre_lfzo_r, 1.0 - car.wdist_f, 1.0 - rd)):
                ta = tyre_for(car.tyre_file, car.tyre_R0, car.tyre_width, lam)
                fz_s = 0.5 * car.m * G * share
                fz_hi = fz_s + split * dfz_1g
                mu_s = ta.mu_y(fz_s)
                fine = fine and (0.80 < mu_s < 0.95 and fz_hi < ta._fzmax
                                 and ta.LFZO == lam)
                rows.append(f"{ax} x{lam:g}: Fz {fz_s / 1e3:5.1f} kN mu {mu_s:.3f}, "
                            f"1 g outer {fz_hi / 1e3:5.1f} < FZMAX {ta._fzmax / 1e3:5.1f} kN")
            rep(f"{key:6s} loads inside its tyre's range", fine, "; ".join(rows))
        # the finding that shaped this module, asserted so it cannot rot
        base = TyreModel(TYRE_REF)
        same = TyreModel(f"{TYRE_DIR}/car145_70R13.tir")
        rep("the .tir files differ in GEOMETRY only (one coefficient set)",
            base.mu_y(4000.0) == same.mu_y(4000.0)
            and base.evaluate(4000.0, 0.05, 0.10) == same.evaluate(4000.0, 0.05, 0.10),
            "car145_70R13 gives identical Fx/Fy/Mz to TNO_car205_60R15 -- "
            "which is why mu_scale, not a different file, carries grip")

    # --- 3b. the stock three do not touch the task-41 physics fields ----
    #  CONTRACT section 2 ("do not rescale LFZO") and every frozen number
    #  hold on the Corsa, the MX-5 and the 540i because each of them leaves
    #  every one of these at its "as before" default, and their Vehicle
    #  gets the unscaled file tyre -- the Corsa's is the singleton itself.
    set_ = {k: [f for f, v in PHYSICS_DEFAULTS.items() if getattr(CARS[k], f) != v]
            for k in STOCK_CARS}
    rep("the three stock cars leave every task-41 physics field at its default",
        not any(set_.values()), str(set_) if any(set_.values()) else
        f"{len(PHYSICS_DEFAULTS)} fields x {STOCK_CARS}; the two new cars set "
        + ", ".join(f"{k}: {sum(getattr(CARS[k], f) != v for f, v in PHYSICS_DEFAULTS.items())}"
                    for k in CAR_ORDER if k not in STOCK_CARS))
    try:
        from drive.vehicle import Vehicle
        from drive.tyre import CORSA_TYRE as _CT
        unscaled = {}
        for k in STOCK_CARS:
            v = Vehicle(CARS[k])
            unscaled[k] = (all(t.LFZO == 1.0 for t in v.der.tyres)
                           and all(r is __import__("qss").TYRE for r in v.der.tyre_refs)
                           and ((v.der.tyres[0] is _CT) == (k == "corsa")))
        rep("the stock cars run the UNSCALED file tyre and qss.TYRE itself",
            all(unscaled.values()), str(unscaled))
    except Exception as exc:                       # pragma: no cover
        rep("the stock cars run the UNSCALED file tyre", False,
            f"{type(exc).__name__}: {exc}")

    # --- 4. added mass: the identity, then that everything really moved ---
    if verbose:
        print("\n  added mass (ballast / fitted wings)")
    z = with_masses(CORSA_C, ())
    rep("zero added mass is the SAME OBJECT", z is CORSA_C,
        "with_masses(car, ()) is car -- what keeps the default bit-for-bit")
    z2 = with_masses(CORSA_C, [PointMass(0.0, 1.0, 1.0, "nothing")])
    rep("a zero-kg point mass is the same object", z2 is CORSA_C,
        "a wing that weighs nothing cannot perturb the default")

    #  the SEAT station is the control case: at the CG, at CG height, so it
    #  may move the mass and NOTHING else. If this ever fails, the first
    #  moments are wrong.
    seat = with_masses(CORSA_C, [ballast_point(CORSA_C, 200.0, "seat")])
    rep("200 kg at the SEAT station moves mass only",
        seat.m == CORSA_C.m + 200.0 and seat.m_s == CORSA_C.m_s + 200.0
        and seat.wdist_f == CORSA_C.wdist_f and seat.h_cg == CORSA_C.h_cg
        and seat.Izz == CORSA_C.Izz and seat.Ixx == CORSA_C.Ixx,
        f"m {seat.m:.0f} kg  wdist_f {seat.wdist_f:.4f}  h_cg {seat.h_cg:.4f}  "
        f"Izz {seat.Izz:.1f} -- all but mass exactly unchanged")

    #  and the four stations must move the three things they are there to
    #  move, in the right DIRECTION. This is the whole point of the feature.
    if verbose:
        print(f"    {'station':7s} {'m':>6s} {'wdist_f':>8s} {'h_cg':>7s} "
              f"{'Izz':>7s} {'Ixx':>6s} {'a':>6s} {'b':>6s}")
    signs = {"nose": (+1, -1), "seat": (0, 0), "floor": (-1, -1), "boot": (-1, +1)}
    for w in BALLAST_STATIONS:
        b = with_masses(CORSA_C, [ballast_point(CORSA_C, 200.0, w)])
        if verbose:
            print(f"    {BALLAST_SHORT[w]:7s} {b.m:6.0f} {b.wdist_f:8.4f} "
                  f"{b.h_cg:7.4f} {b.Izz:7.1f} {b.Ixx:6.1f} {b.a:6.3f} {b.b:6.3f}")
        s_wd, s_h = signs[w]
        d_wd = b.wdist_f - CORSA_C.wdist_f
        d_h = b.h_cg - CORSA_C.h_cg
        ok_wd = (d_wd == 0.0) if s_wd == 0 else (d_wd * s_wd > 1e-3)
        ok_h = (d_h == 0.0) if s_h == 0 else (d_h * s_h > 1e-4)
        rep(f"{BALLAST_SHORT[w]:5s} 200 kg: balance, CG height, Izz, wheelbase",
            ok_wd and ok_h and b.Izz >= CORSA_C.Izz
            and abs(b.a + b.b - CORSA_C.L) < 1e-12 and b.m_s == CORSA_C.m_s + 200.0,
            f"d wdist_f {d_wd:+.4f}  d h_cg {d_h * 1e3:+.1f} mm  "
            f"Izz {100 * (b.Izz / CORSA_C.Izz - 1):+.1f} %  a+b == L")

    # --- 4. the library table --------------------------------------------
    if verbose:
        print(f"\n  {'car':24s} {'m':>6s} {'L':>6s} {'%f':>5s} {'kW':>5s} "
              f"{'Nm':>5s} {'CdA':>5s} {'mu':>5s} {'eng':>5s} {'drv':>4s} {'gears':>6s}")
        for key in CAR_ORDER:
            c2 = CARS[key]
            print(f"  {c2.name[:24]:24s} {c2.m:6.0f} {c2.L:6.3f} {100 * c2.wdist_f:5.0f} "
                  f"{c2.P_max / 1e3:5.0f} {c2.T_max:5.0f} {c2.CdA:5.2f} {c2.mu_scale:5.2f} {c2.engine_scale:5.2f} "
                  f"{c2.drive_layout:>4s} {len(c2.gear):6d}")
        print("\n  NOTE: drive_layout is now BEHAVIOUR -- the rwd cars drive their"
              "\n  rear wheels (powertrain.driven). 'awd' is refused, not guessed.")
        print("  ALL PASS" if ok else "  FAILURES ABOVE")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
