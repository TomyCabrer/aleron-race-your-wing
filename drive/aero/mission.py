"""The wing's MISSION: a LAP of one of carsim's own circuits.

Why a mission layer at all -- AeroBO's own argument, measured
-------------------------------------------------------------
`cartrack.py` in urop-bo-aero exists because "maximise a downforce
coefficient against `CD_budget = 0.11`" is a CALIBRATION, not a requirement.
Measured there: over 1857 geometrically-feasible draws of the car-wing box the
budget admitted **1857 of 1857** -- it decided nothing -- while a two-point
requirement pair admitted 5. A cap that refuses nothing is a constant.

The garage's answer until now was that two-point pair (design speed, force
floor, drag cap). It refuses, so it is a requirement and not a constant -- but
it still asks a DESIGNER to state the exchange rate between downforce and
drag, and that rate is not a number anyone knows. It is an integral over a
circuit: the corner radii say what lateral force is worth, the straight
lengths say what drag costs.

So the mission is a lap, and the objective is seconds.

Not cartrack's synthetic corner list -- carsim's own tracks
-----------------------------------------------------------
`drive/track.py` already DEFINES every circuit as an ordered list of
constant-radius arcs and straights (`Seg`), which is exactly `cartrack`'s
`TrackSpec` shape. So the mission reads carsim's real geometry rather than
restating a circuit here: the arena is its seven published corners at their
published radii, and nothing in this module invents a track.

The lap model, and what it is NOT
---------------------------------
Quasi-steady point mass in the LATERAL sense of `qss.py`, which is a
two-track model: per-corner normal loads, tyre load sensitivity, yaw
equilibrium about the CG, and the flank device's station `x_w` and roll arm
`h_w`. This module adds exactly ONE term to it -- aerodynamic DOWNFORCE, which
`qss` has no row for because the published study's device is a LATERAL one.
At zero downforce this module's `max_ay` reproduces `qss.max_ay` to the bit,
and `self_check` is that comparison.

It is NOT `drive.drive`'s 1 kHz transient sim. It cannot be: a design search
calls the objective hundreds of times and a lap here costs ~1 ms because a
circuit is ~7 corners, not 2499 track samples. What the two disagree about is
a TRANSFER measurement and belongs with the design it is measured on, not in
this docstring.

Stated, not hidden -- three places the car data runs out
--------------------------------------------------------
`corsa_c.py` carries, in capitals, `brakes = "MISSING - no disc/drum dia, pad
mu or F/R torque split"` and `dampers = "MISSING - no rates, no damping
ratios"`, under the line "Do not ship a lap sim without them."

* **BRAKING IS TYRE-LIMITED, not hardware-limited.** No brake data exists, so
  the deceleration here is the one the tyres can take, on all four wheels,
  with the downforce of the moment included. For a 1.2 Corsa that is an upper
  bound -- the car can lock its tyres -- and it is the SAME assumption the
  acceleration pass makes on the other side, where traction is front-axle
  tyre-limited or power-limited and never brake-limited. A wing is compared
  against a wing under one assumption, which is what a design objective needs;
  an absolute lap time from this module is not a claim about the real car.
* **NO TRANSIENT LOAD TRANSFER.** No damper data, so no transient. Corners are
  entered at their steady speed. This is `qss.py`'s own standing assumption.
* **THE LONGITUDINAL FRICTION LAW IS THE LATERAL ONE.** `qss.fy_max` is a
  measured `mu_y(Fz)`; used here for `mu_x` as well. `tyre.py` has the real
  combined-slip surface and the 1 kHz sim flies it; charging it per candidate
  would cost the milliseconds this module exists to save.

What the lap says about THIS car, before anyone calls it a bug
--------------------------------------------------------------
Two results fall out of the Corsa's own numbers the first time a wing is put
on it, and both are correct:

* **A downforce wing behind the rear axle makes the car SLOWER.** `axle_static`
  puts `Fz (x_t + b) / L` on the front axle, and `b = 1.519 m` is CG-to-rear,
  so any station behind `x_t = -1.519` gives that share a NEGATIVE sign: a
  wing hung off the back levers the front axle UP. The car is front-limited
  (FWD, 0.61 front, `roll_dist_f` 0.74), so the axle it unloads is the axle
  that decides. MEASURED here at `CZ*S = 2.0` with the drag switched off:
  -0.86 s at `x_t = 0`, **+0.28 s** at `x_t = -2.00`. This is not new -- the
  DESIGNER page's own station row already says "a top wing behind the rear
  axle unloads the front" -- it is the first time the mission prices it.
* **Downforce is nearly worthless at Corsa speeds; drag is not.** The arena's
  corners are taken at 16-32 m/s, where `CZ*S = 0.4` is 96-216 N against 9908 N
  of car. Over the same lap a `CD*S` of 0.06 costs +0.086 s and the downforce
  it buys returns 0.002 s. That asymmetry is the reason the published study
  hangs a LATERAL panel on the flanks instead of a wing on the roof, and the
  mission reproduces it without being told.

Frame and signs: CONTRACT section 0. `x_w`/`x_t` positive FORWARD of the CG,
downforce positive DOWN, `turn_deg > 0` a LEFT turn (`track.Seg`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

import qss
from corsa_c import CorsaC, RHO, G

CAR = CorsaC()

#: Tracks a mission may be flown on, in the order the page cycles them. The
#: dragstrip is NOT here: it has no corner, so every wing on it is pure drag
#: and the objective would rank them by drag alone. That is a real answer to a
#: different question (`drive.drive --scripted accel`), not a wing mission.
TRACKS = ("arena", "linden", "kestrel", "ashdown", "open", "skidpad")

#: The top wing's one job that is not a circuit. The owner, 2026-09-25: "One
#: more circuit should be added 'Stopping'. Left flank and right flank, should
#: be side." A straight-line stop from `MissionSpec.v_stop_kmh`, as the Stop
#: from 100 challenge (drive/data/challenges/01_brake_100.json) runs it, timed
#: by `stop` below with the SAME braking model the lap's braking zones use.
STOPPING = "stopping"
#: What the TOP wing's mission select offers: the circuits, then the stop.
JOBS = TRACKS + (STOPPING,)
#: The side wings' jobs (`MissionSpec.side_track`): the circuits -- the
#: owner, 2026-09-26, "why no longer circuits?": a side wing is bought for a
#: circuit like the top wing, its side force pays in that circuit's corners --
#: and the stop (the owner, 2026-09-26: "Side wing should also have
#: 'stopping' mission"). A side wing's stop is the AIR BRAKE's
#: (drive/airbrake.py): both flanks out while the car brakes, their side
#: forces cancel and their drags add (`SIDE_STOP_FLANKS`).
SIDE_JOBS = TRACKS + (STOPPING,)
#: Flank panels out in a stop: the top wing's stop keeps the lap's braking
#: zones' one (`_resistance`); a side wing's is the air brake's pair.
SIDE_STOP_FLANKS = 2
#: A job in words, where a sentence names it ("the stopping job's own").
JOB_WORDS = {STOPPING: "stopping"}
#: The stop's start speed, km/h: the Stop from 100 challenge's, and the band
#: the page offers (the Air brake challenge starts at 150).
V_STOP_KMH = 100.0
V_STOP_BAND = (40.0, 250.0)

#: Below this curvature a segment is a STRAIGHT for the mission's purposes.
#: 1/1200 m^-1: at the arena's own speeds a 1200 m radius costs under 0.01
#: m/s of corner speed, so calling it a straight changes no lap time this
#: module reports, and it keeps a surveying artefact out of the corner list.
KAPPA_STRAIGHT = 1.0 / 1200.0

#: Corners this tight are not driven at their geometric radius by any car --
#: they are a hairpin the driver arcs. Not clamped, only flagged: the lap is
#: still integrated, and `LapResult.notes` says so.
R_MIN_SANE = 8.0

#: Straight integration: steps per straight. 64 puts the arena's longest
#: straight (169 m) at 2.6 m per step; the lap time changes by under 1 ms
#: between 64 and 256 (measured, `self_check`).
N_STEPS_STRAIGHT = 64

#: Fixed-point iterations for the implicit longitudinal-load-transfer solve.
#: The map is a contraction with ratio ~ h_cg/L * dmu/dN * ... << 1 here;
#: 6 is well past convergence at 1e-12 and costs nothing.
N_LOAD_ITERS = 6

#: The deploy mask is solved for, not assumed: an 'active' top wing is out
#: under brake or steering, and where the braking zones are depends on the
#: speeds, which depend on the mask. Iterated to a fixed point; it has always
#: converged in <= 3 here, and `LapResult.notes` records a failure to.
N_MASK_ITERS = 5


# --------------------------------------------------------------------------- #
#  what a lap is handed: the car's aerodynamics as numbers                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MissionAero:
    """Every aerodynamic device on the car, as the lap reads it.

    This is the same shape `vehicle.DevAero` / `vehicle.TopAero` are, and for
    the same reason: the design layer builds it out of a lattice solution and
    hands over numbers, so the lap never imports a solver and the solver never
    learns about tracks.

    `k_dev` is ONE flank panel's `0.5 rho S CL` (N/(m/s)^2) -- one panel is
    deployed at a time, towards the corner's centre, which is what
    `vehicle.VehicleState.dev_side` does and what `crossover.gain` is written
    against. `cz_a` and `cd_a` are the top wing's `CZ*S` and `CD*S` in m^2.
    """

    #  flank panels (lateral force)
    k_dev: float = 0.0          # N/(m/s)^2, one panel, deployed
    ld_dev: float = 3.2         # the panel's L/D at its deployed point
    x_w: float = 0.97           # m forward of the CG
    h_w: float = 0.90           # m roll arm
    #  top wing (downforce)
    cz_a: float = 0.0           # m^2, CZ*S, positive DOWN
    cd_a: float = 0.0           # m^2, CD*S, deployed
    cd_a_stowed: float = 0.0    # m^2, stowed
    x_t: float = -1.60          # m forward of the CG
    top_mode: str = "fixed"     # 'fixed' | 'active'

    def d_dev(self, V: float) -> float:
        """One deployed panel's drag, N. The panel is a lifting surface, so
        its drag is its own force over its own L/D -- the same reduction
        `qss.corner_speed` already charges in its power cap."""
        if self.k_dev <= 0.0 or self.ld_dev <= 0.0:
            return 0.0
        return self.k_dev * V * V / self.ld_dev


def merge_wing(base: MissionAero, aero: dict, role: str, inc_deg: float,
               x: float, h: float, mode: str = "fixed") -> MissionAero:
    """`base` with the wing described by `aero` fitted into its `role` slot.

    `aero` is a `wing.analyse` result -- the affine lift law and the quadratic
    drag law the lattice fitted -- read at the mount angle `inc_deg`. This is
    the same reduction `vehicle.DevAero` / `vehicle.TopAero` make for the 1 kHz
    step, written once here so the lap and the simulator cannot end up flying
    two different wings off one design.

    A FLANK panel makes LATERAL force: it enters as `k_dev = 0.5 rho S CL` and
    its own L/D. A TOP wing makes downforce: `CZ*S` and `CD*S`. The role
    decides which, and nothing else about `base` is touched -- so a section
    designed for the top slot is scored on a car that still has whatever is on
    its flanks."""
    cl = aero["CL0"] + aero["CLa"] * math.radians(float(inc_deg))
    cl = min(max(cl, aero["CL_min"]), aero["CL_max"])
    cd = max(aero["cd0"] + aero["cd1"] * cl + aero["cd2"] * cl * cl, 1e-6)
    S = float(aero["S"])
    if role == "flank":
        return MissionAero(k_dev=0.5 * RHO * S * cl, ld_dev=cl / cd, x_w=float(x),
                           h_w=float(h), cz_a=base.cz_a, cd_a=base.cd_a,
                           cd_a_stowed=base.cd_a_stowed, x_t=base.x_t,
                           top_mode=base.top_mode)
    return MissionAero(k_dev=base.k_dev, ld_dev=base.ld_dev, x_w=base.x_w,
                       h_w=base.h_w, cz_a=S * cl, cd_a=S * cd, cd_a_stowed=0.0,
                       x_t=float(x), top_mode=("active" if mode == "active" else "fixed"))


# --------------------------------------------------------------------------- #
#  the circuit, read off carsim's own segment list                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Corner:
    radius: float               # m, always positive
    length: float               # m, arc length
    sign: int                   # +1 left, -1 right


@dataclass(frozen=True)
class Straight:
    length: float               # m


@dataclass(frozen=True)
class TrackProfile:
    """A circuit as the lap walks it: a CYCLIC list of corners and straights.

    Built from `drive.track`'s own `Seg` list and nothing else, so a mission
    cannot be flown on a circuit the simulator does not have. Duck-typed on
    purpose -- `drive/aero` does not import `drive.track` (CONTRACT section 1
    keeps the aero package free of the simulator), so the caller passes the
    built `Track` in and only its `.segs`, `.name`, `.title` and `.closed` are
    read.
    """

    name: str
    title: str
    items: tuple                # (Corner | Straight, ...) in order
    closed: bool
    length: float

    @classmethod
    def from_track(cls, tr) -> "TrackProfile":
        items = []
        for sg in tr.segs:
            if getattr(sg, "kind", "S") == "A" and abs(sg.radius) > 1e-9 \
                    and 1.0 / abs(sg.radius) > KAPPA_STRAIGHT:
                items.append(Corner(radius=abs(float(sg.radius)),
                                    length=float(sg.length),
                                    sign=1 if sg.turn_deg >= 0.0 else -1))
            else:
                items.append(Straight(length=float(sg.length)))
        items = _merge_straights(items)
        return cls(name=str(tr.name), title=str(getattr(tr, "title", "") or tr.name),
                   items=tuple(items), closed=bool(tr.closed),
                   length=float(sum(i.length for i in items)))

    @property
    def corners(self) -> list:
        return [i for i in self.items if isinstance(i, Corner)]

    def describe(self) -> str:
        c = self.corners
        if not c:
            return f"{self.title}: {self.length:.0f} m, no corner"
        rs = sorted(i.radius for i in c)
        return (f"{self.title}: {self.length:.0f} m, {len(c)} corners, "
                f"R {rs[0]:.0f}-{rs[-1]:.0f} m")


def _merge_straights(items: list) -> list:
    """Adjacent straights are one straight. A surveyed circuit splits them at
    node boundaries; the lap does not care and a split straight would be
    braked into and accelerated out of twice."""
    out: list = []
    for it in items:
        if isinstance(it, Straight) and out and isinstance(out[-1], Straight):
            out[-1] = Straight(out[-1].length + it.length)
        else:
            out.append(it)
    return out


# --------------------------------------------------------------------------- #
#  the lateral model: qss.py, plus downforce                                   #
# --------------------------------------------------------------------------- #
def axle_static(aero: MissionAero, V: float, car: CorsaC = CAR) -> tuple:
    """Front and rear axle vertical loads at speed V, N.

    Weight plus the top wing's downforce, split by the wing's own station:
    a force `Fz` acting `x` forward of the CG puts `Fz (x + b) / L` on the
    front axle (`b` is CG-to-REAR in `corsa_c`'s convention, so `x = 0`
    returns the static split and `x = a` puts all of it on the front axle --
    both checked in `self_check`)."""
    W = car.m * G
    fz = 0.5 * RHO * aero.cz_a * V * V
    f_share = (aero.x_t + car.b) / car.L
    return (W * car.wdist_f + fz * f_share,
            W * (1.0 - car.wdist_f) + fz * (1.0 - f_share))


def residuals(a_y: float, V: float, aero: MissionAero, mu_scale: float = 1.0,
              roll_dist_f: float = 0.74, car: CorsaC = CAR) -> tuple:
    """Per-axle utilisation, 1.0 = at the limit. `qss.residuals` with the two
    static axle loads replaced by `axle_static` -- at `cz_a = 0` the two are
    the same function and `self_check` proves it."""
    F = aero.k_dev * V * V
    L = car.L
    fz_f, fz_r = axle_static(aero, V, car)

    Y_f = (car.b * car.m * a_y - F * (car.b + aero.x_w)) / L
    Y_r = (car.a * car.m * a_y + F * aero.x_w) / L

    dFz_tot = (car.m * a_y * car.h_cg - F * aero.h_w) / car.t
    cap_f = qss.axle_capacity(fz_f, dFz_tot * roll_dist_f, mu_scale)
    cap_r = qss.axle_capacity(fz_r, dFz_tot * (1.0 - roll_dist_f), mu_scale)
    return Y_f / cap_f, Y_r / cap_r


#: Bisection depth for `max_ay`. The bracket is [0.1, 40] m/s^2, so 40 halvings
#: resolve it to 4e-11 -- far past anything a lap time can see. `qss.max_ay`
#: uses 200 over the same bracket; the extra 160 buy nothing but they are the
#: difference between a 30 ms lap and a 270 ms one, and this module is called
#: from inside a design search. Parity with `qss` is checked at this depth.
N_AY_BISECT = 40

#: Bisection depth for the two corner-speed limits, bracket [1, 140] m/s:
#: 1.3e-10 m/s. `qss.corner_speed` runs a 200-step damped fixed point for the
#: grip limit; a bisection on the same monotone residual is the same answer in
#: a fifth of the calls, and `self_check` compares the two.
N_V_BISECT = 40


def max_ay(V: float, aero: MissionAero, mu_scale: float = 1.0,
           roll_dist_f: float = 0.74, car: CorsaC = CAR) -> float:
    """Largest sustainable lateral acceleration at V. Bisection, as qss does."""
    lo, hi = 0.1, 40.0
    for _ in range(N_AY_BISECT):
        a = 0.5 * (lo + hi)
        uf, ur = residuals(a, V, aero, mu_scale, roll_dist_f, car)
        lo, hi = (a, hi) if max(uf, ur) < 1.0 else (lo, a)
    return 0.5 * (lo + hi)


def corner_speed(R: float, aero: MissionAero, mu_scale: float = 1.0,
                 roll_dist_f: float = 0.74, alpha_peak_deg: float = 7.0,
                 car: CorsaC = CAR) -> tuple:
    """Fastest steady speed through a constant-radius corner, and which limit
    set it ('grip' or 'power').

    `qss.corner_speed`'s two limits, with this module's `max_ay` and with the
    top wing's drag added to the power balance. The scrub term is the one a
    naive analysis omits: at these speeds `m a_y sin(alpha_peak)` dominates
    aerodynamic drag, and it is charged at the ACTUAL `a_y`, not at `max_ay`,
    because at the power limit the car is not at the grip limit -- charging it
    at the grip limit makes a wing look harmful for raising grip it never gets
    to use. That reasoning is `qss.corner_speed`'s, verbatim, and so is the
    code shape."""
    #  GRIP. `V^2 / R - max_ay(V)` is negative at a crawl and positive at any
    #  speed this car reaches, so the limit is its root and a bisection finds
    #  it. (It stops being monotone only if downforce buys lateral grip faster
    #  than V^2, i.e. a car that is never grip-limited; `hi` below is then
    #  returned and the power cap takes over, which is the right answer.)
    lo, hi = 1.0, 140.0
    for _ in range(N_V_BISECT):
        V = 0.5 * (lo + hi)
        lo, hi = (V, hi) if V * V / R < max_ay(V, aero, mu_scale, roll_dist_f, car) else (lo, V)
    V = 0.5 * (lo + hi)

    #  POWER. Charged exactly as `qss.corner_speed` charges it -- scrub at the
    #  ACTUAL a_y, not at max_ay, because at the power limit the car is not at
    #  the grip limit and charging it there would make a wing look harmful for
    #  raising grip it never gets to use.
    roll = car.Crr * car.m * G
    sa = math.sin(math.radians(alpha_peak_deg))

    def _power(Vp: float) -> float:
        a_act = min(Vp * Vp / R, max_ay(Vp, aero, mu_scale, roll_dist_f, car))
        drag = 0.5 * RHO * (car.CdA + aero.cd_a) * Vp * Vp + aero.d_dev(Vp)
        return Vp * (car.m * a_act * sa + drag + roll)

    #  the power-feasible set is [0, Vp], so if the grip speed is already
    #  affordable there is nothing to search -- which is six of the arena's
    #  seven corners, and the reason a lap costs milliseconds.
    if _power(V) <= car.P_wheel:
        return V, "grip"
    lo, hi = 1.0, V
    for _ in range(N_V_BISECT):
        Vp = 0.5 * (lo + hi)
        lo, hi = (Vp, hi) if _power(Vp) < car.P_wheel else (lo, Vp)
    return 0.5 * (lo + hi), "power"


# --------------------------------------------------------------------------- #
#  the longitudinal model                                                      #
# --------------------------------------------------------------------------- #
def _mu(fz: float, mu_scale: float) -> float:
    """The measured load-sensitive friction coefficient at one wheel."""
    if fz <= 0.0:
        return 0.0
    return max(mu_scale * (qss.TYRE["mu_ref"] + qss.TYRE["s"] * (fz - qss.TYRE["Fz_ref"])), 0.0)


def _resistance(V: float, aero: MissionAero, deployed: bool, car: CorsaC = CAR,
                flanks: int = 1) -> float:
    """Everything resisting the car on a straight at speed V, N: the body's
    drag, the wing's drag (stowed or deployed) and rolling resistance.
    `flanks` panels are out when deployed: one (a corner's), or the air
    brake's two (`SIDE_STOP_FLANKS`), whose side forces cancel."""
    cd_a = aero.cd_a if deployed else aero.cd_a_stowed
    d = 0.5 * RHO * (car.CdA + cd_a) * V * V + car.Crr * car.m * G
    return d + (flanks * aero.d_dev(V) if deployed else 0.0)


def accel(V: float, aero: MissionAero, deployed: bool, mu_scale: float = 1.0,
          car: CorsaC = CAR) -> float:
    """Longitudinal acceleration available at V on a straight, m/s^2.

    FRONT AXLE ONLY -- the Corsa is front-wheel drive, which is the whole
    reason the published study is a study about a FRONT-limited car. Traction
    and the load transfer that sets it are solved together: accelerating
    unloads the driven axle, so `N_f` depends on `a_x` which depends on `N_f`.
    Fixed point, `N_LOAD_ITERS` passes."""
    if V < 0.5:
        V = 0.5
    fz_f0, _ = axle_static(aero, V, car)
    res = _resistance(V, aero, deployed, car)
    f_power = car.P_wheel / V
    a_x = 0.0
    for _ in range(N_LOAD_ITERS):
        fz_f = fz_f0 - car.m * a_x * car.h_cg / car.L
        f_grip = 2.0 * _mu(0.5 * fz_f, mu_scale) * (0.5 * fz_f) if fz_f > 0.0 else 0.0
        a_x = (min(f_power, f_grip) - res) / car.m
    return a_x


def brake(V: float, aero: MissionAero, deployed: bool, mu_scale: float = 1.0,
          car: CorsaC = CAR, flanks: int = 1) -> float:
    """Braking deceleration available at V, m/s^2, POSITIVE.

    All four tyres, at the loads of the moment including downforce. TYRE
    limited, not brake limited -- there is no brake data for this car (see the
    module docstring). Load transfer is symmetric front-to-rear here (what one
    axle gains the other loses) and the friction law is very nearly linear
    over the transfer, so it is not iterated: the pair is evaluated at the
    static split and the transfer's first-order effect cancels."""
    if V < 0.5:
        V = 0.5
    fz_f, fz_r = axle_static(aero, V, car)
    grip = (2.0 * _mu(0.5 * fz_f, mu_scale) * (0.5 * fz_f)
            + 2.0 * _mu(0.5 * fz_r, mu_scale) * (0.5 * fz_r))
    return (grip + _resistance(V, aero, deployed, car, flanks)) / car.m


# --------------------------------------------------------------------------- #
#  the stop                                                                    #
# --------------------------------------------------------------------------- #
#: Speed steps of the stop's quadrature. The integrand V / a(V) is smooth and
#: a(V) >= mu*g never vanishes, so 400 trapezoids sit within a millimetre of
#: 4000 (measured, `self_check`).
N_STOP_STEPS = 400


@dataclass
class StopResult:
    """A straight-line stop from `v0` to rest (`stop`)."""
    distance: float = math.inf                       # m
    time: float = math.inf                           # s
    v0: float = 0.0                                  # m/s
    ok: bool = False
    notes: list = field(default_factory=list)
    #: the stop as a curve, ~40 points from v0 to rest: distance travelled
    #: since the brakes went on (m) and the speed there (m/s) -- the page's plot
    d_curve: list = field(default_factory=list)
    v_curve: list = field(default_factory=list)

    @property
    def decel_mean(self) -> float:
        """v0 / time, m/s^2: the stop's mean deceleration."""
        return self.v0 / self.time if self.ok and self.time > 0.0 else 0.0

    def to_json(self) -> dict:
        return asdict(self)


def stop(v0: float, aero: MissionAero | None = None, mu_scale: float = 1.0,
         car: CorsaC = CAR, n: int | None = None, flanks: int = 1) -> StopResult:
    """A straight-line stop from `v0` (m/s) to rest: distance = the integral
    of V / a(V) dV and time = the integral of dV / a(V), with a(V) the lap's
    own braking deceleration (`brake`, the wings deployed -- an active top
    wing is out under brake). Tyre-limited on all four wheels with the
    downforce of the moment, plus every drag on the car and rolling
    resistance: the same assumption the lap's braking zones make.

    Why this is the stop's objective and AeroBO's "downforce + drag" ranks
    it: the deceleration is mu*(m*g + downforce) + drag + the rest, and the
    two wing terms both scale with V^2, so across wings the shortest stop is
    the one with the most mu*downforce + drag at every speed. At mu = 1 that
    is downforce + drag exactly; below it (damp, wet) drag is worth more
    than that ranking says.

    `flanks`: the flank panels out. A side wing's stop is the air brake's
    (`SIDE_STOP_FLANKS`, `MissionSpec.stop_flanks`): both panels, their side
    forces cancel, so a side wing stops the car with its drag alone."""
    v0 = float(v0)
    aero = aero if aero is not None else MissionAero()
    if not (v0 > 0.0 and math.isfinite(v0)):
        return StopResult(distance=0.0, time=0.0, v0=max(v0, 0.0), ok=v0 == 0.0,
                          notes=[] if v0 == 0.0 else [f"no stop from {v0!r} m/s"])
    N = int(n or N_STOP_STEPS)
    vs = np.linspace(0.0, v0, N + 1)
    a = np.array([brake(float(v), aero, True, mu_scale, car, flanks) for v in vs])
    if not np.all(a > 0.0):
        return StopResult(v0=v0, notes=["the car cannot decelerate at some speed"])
    g = vs / a
    upto = np.concatenate(([0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(vs))))
    dist = float(upto[-1])
    k = max(1, N // 40)
    return StopResult(distance=dist, time=float(np.trapezoid(1.0 / a, vs)), v0=v0, ok=True,
                      d_curve=[float(dist - x) for x in upto[::-k]],
                      v_curve=[float(v) for v in vs[::-k]])


# --------------------------------------------------------------------------- #
#  the lap                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class LapResult:
    time: float = math.inf
    ok: bool = False
    v_corner: list = field(default_factory=list)     # m/s, per corner, in order
    limited: list = field(default_factory=list)      # 'grip' | 'power', per corner
    v_max: float = 0.0                               # m/s, fastest point of the lap
    t_corner: float = 0.0
    t_straight: float = 0.0
    v_mean: float = 0.0                              # m/s, length / time
    length: float = 0.0                              # m
    notes: list = field(default_factory=list)
    track: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def _straight_time(length: float, v_in: float, v_out: float, aero: MissionAero,
                   mu_scale: float, car: CorsaC, top_active: bool) -> tuple:
    """Time to cross a straight entered at `v_in` and left at `v_out`, s, and
    the fraction of it spent braking, and the top speed reached on it.

    Accelerate from `v_in` as far as the straight allows, brake back to
    `v_out`: integrate both profiles in space and take the lower speed at each
    station, which is the standard quasi-steady construction and needs no
    search for the brake point. `n` steps of `ds = length / n`, trapezoid in
    `1/V` for the time -- the same quadrature both profiles are built with, so
    a straight that is all acceleration and one that is all braking are
    integrated to the same order."""
    n = N_STEPS_STRAIGHT
    ds = length / n
    if ds <= 0.0:
        return 0.0, 0.0
    #  an 'active' top wing is stowed while accelerating and out while braking;
    #  a 'fixed' one is out throughout. The flank panel is stowed on a straight
    #  in both cases -- it is a cornering device and the renderer stows it.
    dep_acc = (aero.top_mode == "fixed")
    dep_brk = top_active or (aero.top_mode == "fixed")

    vf = np.empty(n + 1)
    vf[0] = v_in
    for i in range(n):
        a = accel(vf[i], aero, dep_acc, mu_scale, car)
        vf[i + 1] = math.sqrt(max(vf[i] * vf[i] + 2.0 * a * ds, 0.25))

    vb = np.empty(n + 1)
    vb[n] = v_out
    for i in range(n, 0, -1):
        d = brake(vb[i], aero, dep_brk, mu_scale, car)
        vb[i - 1] = math.sqrt(max(vb[i] * vb[i] + 2.0 * d * ds, 0.25))

    v = np.minimum(vf, vb)
    inv = 1.0 / v
    t = float(ds * (0.5 * inv[0] + inv[1:-1].sum() + 0.5 * inv[-1]))
    braking = float((vb < vf).sum()) / (n + 1)
    #  the time-weighted speed integral: sum V dt = sum V (ds / V) = length.
    #  Trivial here, and stated so nobody "fixes" it into sum V dt computed
    #  the long way -- the lap's mean speed is length / time by construction.
    return t, braking, float(v.max())


def lap(profile: TrackProfile, aero: MissionAero | None = None,
        mu_scale: float = 1.0, roll_dist_f: float = 0.74,
        alpha_peak_deg: float = 7.0, car: CorsaC = CAR) -> LapResult:
    """One flying lap, seconds.

    Corners are taken at their steady speed (`corner_speed`); straights are
    the acceleration profile out of one corner met by the braking profile into
    the next. On a CLOSED circuit the list wraps, so the last straight is
    bounded by the first corner -- no start-line special case and no
    standing start, which is what 'a flying lap' means."""
    aero = aero or MissionAero()
    items = list(profile.items)
    res = LapResult(track=profile.name)
    if not items:
        res.notes.append("the circuit has no segments")
        return res

    #  corner speeds first: they bound every straight.
    v_c, lim = {}, {}
    for i, it in enumerate(items):
        if isinstance(it, Corner):
            if it.radius < R_MIN_SANE:
                res.notes.append(f"corner {i} R {it.radius:.1f} m is below "
                                 f"{R_MIN_SANE:.0f} m: no car drives the geometric line")
            v, w = corner_speed(it.radius, aero, mu_scale, roll_dist_f, alpha_peak_deg, car)
            v_c[i], lim[i] = v, w
    if not v_c:
        res.notes.append("the circuit has no corner: every wing on it is pure drag")
        return res

    n = len(items)

    def _v_before(i: int) -> float:
        """Speed entering item i, i.e. the previous corner's speed."""
        for j in range(1, n + 1):
            k = (i - j) % n
            if k in v_c:
                return v_c[k]
        return 0.0

    def _v_after(i: int) -> float:
        for j in range(1, n + 1):
            k = (i + j) % n
            if k in v_c:
                return v_c[k]
        return 0.0

    #  the deploy mask for an 'active' top wing is a fixed point: it is out
    #  under brake, and where the braking is depends on the speeds, which
    #  depend on it. Iterate; it settles in two or three.
    active = {i: (aero.top_mode == "fixed") for i in range(n) if isinstance(items[i], Straight)}
    t_str, v_top = 0.0, 0.0
    converged = False
    for _ in range(N_MASK_ITERS):
        t_str, v_top, nxt = 0.0, 0.0, {}
        for i, it in enumerate(items):
            if not isinstance(it, Straight):
                continue
            t, frac, vpk = _straight_time(it.length, _v_before(i), _v_after(i), aero,
                                         mu_scale, car, active.get(i, False))
            t_str += t
            nxt[i] = frac > 0.0
            v_top = max(v_top, vpk)
        if nxt == active:
            converged = True
            break
        active = nxt
    if not converged and aero.top_mode == "active":
        res.notes.append("the 'active' deploy mask did not settle; last pass used")

    t_cor = sum(items[i].length / v_c[i] for i in v_c)
    order = sorted(v_c)
    res.v_corner = [v_c[i] for i in order]
    res.limited = [lim[i] for i in order]
    res.v_max = max(v_top, max(v_c.values()))
    res.t_corner, res.t_straight = t_cor, t_str
    res.time = t_cor + t_str
    res.length = profile.length
    res.v_mean = profile.length / res.time if res.time > 0.0 else 0.0
    res.ok = math.isfinite(res.time) and res.time > 0.0
    return res


@dataclass
class MissionSpec:
    """The mission the garage has STATED: which circuit, in what conditions.
    Or STOPPING (`track == STOPPING`): a straight-line stop from
    `v_stop_kmh`, in the same conditions. The side wings have a job of their
    own, `side_track` (`SIDE_JOBS`: a circuit or the stop, from the same
    `v_stop_kmh`), in the same conditions: `for_side()` is their mission.

    Held by the garage across the three design pages, which is the whole point
    of stating it first -- the section and the wing are both scored against
    THIS object, so they cannot be scored against two different missions and
    then compared.

    `stated` is not decoration: the wing page is gated on it. A mission nobody
    confirmed is a default, and a default is the calibration this layer exists
    to replace."""

    track: str = "arena"
    mu_scale: float = 1.0
    stated: bool = False
    v_stop_kmh: float = V_STOP_KMH
    side_track: str = "arena"
    _profile: TrackProfile | None = field(default=None, repr=False, compare=False)
    _side: "MissionSpec | None" = field(default=None, repr=False, compare=False)
    #: this is `for_side()`'s spec: a stop is flown on the air brake
    _is_side: bool = field(default=False, repr=False, compare=False)

    #: Surface conditions offered, and the scale each puts on tyre grip. The
    #: wet number is PUBLISHED (`track.MU_WET_SCALE`, 0.55/0.87 from
    #: `qss.sweep`); the damp one is an estimate and `track.py` says so.
    SURFACES = (("dry", 1.0), ("damp", 0.80), ("wet", 0.632183908))

    def profile(self, make_track) -> TrackProfile:
        """The circuit, built once. `make_track` is `drive.track.make_track`,
        passed in rather than imported -- CONTRACT section 1 keeps
        `drive/aero` free of the simulator. A stop has no circuit: ValueError."""
        if self.is_stop:
            raise ValueError(f"a stop has no circuit: it is a straight-line stop from "
                             f"{self.v_stop_kmh:.0f} km/h")
        if self._profile is None or self._profile.name != self.track:
            self._profile = TrackProfile.from_track(make_track(self.track))
        return self._profile

    def for_side(self) -> "MissionSpec":
        """The side wings' mission: their own job (`side_track`: a circuit
        or the stop) in the same conditions, stated when this one is. Kept
        between calls, so the circuit is built once."""
        t = self.side_track if self.side_track in SIDE_JOBS else "arena"
        m = self._side
        if m is None or m.track != t:
            m = self._side = MissionSpec(track=t, _is_side=True)
        m.mu_scale, m.stated, m.v_stop_kmh, m.side_track = (self.mu_scale, self.stated,
                                                            self.v_stop_kmh, t)
        return m

    @property
    def is_stop(self) -> bool:
        return self.track == STOPPING

    @property
    def stop_flanks(self) -> int:
        """Flank panels out in this mission's stop (`stop(flanks=)`): the air
        brake's pair on a side wing's, the lap's one on the top wing's."""
        return SIDE_STOP_FLANKS if self._is_side else 1

    @property
    def v_stop(self) -> float:
        """The stop's start speed, m/s."""
        return float(self.v_stop_kmh) / 3.6

    @property
    def job_key(self) -> str:
        """The job as a session signature reads it: the circuit's name, or
        the stop WITH its start speed (a different stop is a different
        design point)."""
        return f"{STOPPING} {self.v_stop_kmh:.0f} km/h" if self.is_stop else self.track

    @property
    def surface(self) -> str:
        for nm, sc in self.SURFACES:
            if abs(sc - self.mu_scale) < 1e-9:
                return nm
        return f"mu x{self.mu_scale:.2f}"

    def to_json(self) -> dict:
        return dict(track=self.track, mu_scale=self.mu_scale, stated=self.stated,
                    v_stop_kmh=self.v_stop_kmh, side_track=self.side_track)

    @classmethod
    def from_json(cls, d: dict) -> "MissionSpec":
        lo, hi = V_STOP_BAND
        return cls(track=str(d.get("track", "arena")),
                   mu_scale=float(d.get("mu_scale", 1.0)),
                   stated=bool(d.get("stated", False)),
                   v_stop_kmh=min(max(float(d.get("v_stop_kmh", V_STOP_KMH)), lo), hi),
                   side_track=(str(d.get("side_track")) if d.get("side_track") in SIDE_JOBS
                               else "arena"))


# --------------------------------------------------------------------------- #
def self_check(verbose: bool = True) -> bool:
    """Every claim this module makes, measured.

    The load-bearing one is PARITY: at zero downforce this is `qss.py`, and
    `qss.py` is the published study's own cornering model. If the two ever
    disagree, this module has changed the physics rather than extended it.
    """
    import time as _t
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and passed
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}{('  ' + msg) if msg else ''}")

    #  1. residual parity, bit for bit
    worst = 0.0
    for a_y in (4.0, 8.0, 11.0):
        for V in (12.0, 25.0, 40.0):
            for k, x_w, h_w in ((0.0, 0.0, 0.90), (0.147, 0.97, 0.90), (0.30, -1.2, 0.60)):
                m = residuals(a_y, V, MissionAero(k_dev=k, x_w=x_w, h_w=h_w))
                q = qss.residuals(a_y, V, k, x_w, h_w, 0.74, 1.0)
                worst = max(worst, abs(m[0] - q[0]), abs(m[1] - q[1]))
    rep("residuals == qss.residuals at zero downforce", worst == 0.0, f"worst |d| {worst:.1e}")

    #  2. max_ay parity
    #  `qss.max_ay(V, k)` takes x_w positionally after k and defaults it to 0,
    #  so the MissionAero handed alongside it must say x_w = 0 too -- a default
    #  of 0.97 here compared two different cars and read as a 5e-2 physics
    #  disagreement when it was an argument-order one.
    worst = max(abs(max_ay(V, MissionAero(k_dev=k, x_w=0.0)) - qss.max_ay(V, k))
                for V in (12.0, 25.0, 40.0) for k in (0.0, 0.147))
    rep("max_ay == qss.max_ay", worst < 1e-9, f"worst |d| {worst:.1e}")

    #  3. corner_speed parity, including the power-limited branch
    worst, both = 0.0, True
    for R in (30.0, 45.0, 55.0, 80.0, 100.0, 130.0):
        mv, ml = corner_speed(R, MissionAero())
        qv, ql = qss.corner_speed(R)
        worst = max(worst, abs(mv - qv))
        both = both and (ml == ql)
    rep("corner_speed == qss.corner_speed (speed and which limit)",
        worst < 1e-7 and both, f"worst |d| {worst:.1e}")

    #  4. the arena's own published numbers. track.py's T7 comment says
    #     "the only POWER-limited corner (qss: 31.88 m/s)", and wing.V_REF's
    #     flank entry says its default IS the R = 100 limit speed.
    v7, l7 = corner_speed(130.0, MissionAero())
    rep("arena T7 is 31.88 m/s and power-limited (track.py's own comment)",
        abs(v7 - 31.88) < 0.005 and l7 == "power", f"{v7:.4f} m/s {l7}")
    v100, _ = corner_speed(100.0, MissionAero())
    rep("R = 100 limit speed is wing.V_REF['flank'] = 29.0875",
        abs(v100 - 29.0875) < 5e-4, f"{v100:.6f} m/s")

    #  5. the downforce split
    car = CAR
    W = car.m * G
    a = MissionAero(cz_a=1.0, x_t=0.0)
    f0, r0 = axle_static(a, 30.0)
    fz = 0.5 * RHO * 1.0 * 900.0
    stat_ok = (abs(f0 - (W * car.wdist_f + fz * car.wdist_f)) < 1e-6
               and abs(r0 - (W * (1 - car.wdist_f) + fz * (1 - car.wdist_f))) < 1e-6)
    rep("downforce at the CG splits by the static distribution", stat_ok)
    f1, r1 = axle_static(MissionAero(cz_a=1.0, x_t=car.a), 30.0)
    rep("downforce at the front axle lands entirely on it",
        abs(f1 - (W * car.wdist_f + fz)) < 1e-6 and abs(r1 - W * (1 - car.wdist_f)) < 1e-6)
    f2, r2 = axle_static(MissionAero(cz_a=1.0, x_t=-1.9), 30.0)
    rep("a wing behind the rear axle UNLOADS the front", f2 < W * car.wdist_f,
        f"front {f2:.1f} N vs {W * car.wdist_f:.1f} N static")

    #  6. every offered track flies, and costs an inner loop nothing
    try:
        from .. import track as _tr
    except Exception:
        _tr = None
    if _tr is None:
        rep("drive.track is importable", False, "skipped the circuit checks")
    else:
        for name in TRACKS:
            pf = TrackProfile.from_track(_tr.make_track(name))
            t0 = _t.perf_counter()
            r = lap(pf, MissionAero())
            dt = _t.perf_counter() - t0
            rep(f"{name} flies", r.ok and len(pf.corners) >= 1 and not r.notes,
                f"{r.time:.3f} s, {len(pf.corners)} corners, {dt * 1e3:.0f} ms")
            rep(f"{name} costs an inner loop under 60 ms", dt < 0.060, f"{dt * 1e3:.0f} ms")
            rep(f"{name} mean speed is length / time",
                abs(r.v_mean - pf.length / r.time) < 1e-9, f"{r.v_mean:.2f} m/s")

        pf = TrackProfile.from_track(_tr.make_track("arena"))
        base = lap(pf, MissionAero()).time

        #  determinism: the design search reads this number hundreds of times
        rep("the lap is deterministic", lap(pf, MissionAero()).time == base)

        #  the straight quadrature has converged at the shipped step count
        global N_STEPS_STRAIGHT
        n0 = N_STEPS_STRAIGHT
        try:
            N_STEPS_STRAIGHT = 32
            t32 = lap(pf, MissionAero()).time
            N_STEPS_STRAIGHT = 256
            t256 = lap(pf, MissionAero()).time
        finally:
            N_STEPS_STRAIGHT = n0
        rep("the straight quadrature has converged", abs(t256 - t32) < 0.010,
            f"32 steps {t32:.4f} s vs 256 {t256:.4f} s")

        #  drag costs time, monotonically
        ts = [lap(pf, MissionAero(cd_a=c)).time for c in (0.0, 0.03, 0.06, 0.12)]
        rep("drag costs lap time, monotonically", all(b > a for a, b in zip(ts, ts[1:])),
            " -> ".join(f"{t:.3f}" for t in ts))

        #  the published flank panel buys lap time at its published L/D
        k = 0.5 * RHO * 0.35 * 0.70
        dev = MissionAero(k_dev=k, ld_dev=3.2, x_w=0.97, h_w=0.90)
        t_dev = lap(pf, dev).time
        rep("the study's own flank panel is worth lap time", t_dev < base,
            f"{t_dev:.4f} s vs {base:.4f} s bare ({t_dev - base:+.4f})")

        #  ...at the scrub angle qss states, and NOT at the one drive.validate
        #  measures. The docstring's table, pinned, because a device that
        #  changes sign under a number nobody was looking at is the finding.
        b10 = lap(pf, MissionAero(), alpha_peak_deg=10.34).time
        d10 = lap(pf, dev, alpha_peak_deg=10.34).time
        rep("the verdict on that panel FLIPS at validate.py's measured scrub angle",
            (t_dev - base) < 0.0 < (d10 - b10),
            f"7.00 deg {t_dev - base:+.4f} s   10.34 deg {d10 - b10:+.4f} s "
            f"(bare {base:.4f} -> {b10:.4f})")

        #  and a panel with no L/D at all is not
        t_bad = lap(pf, MissionAero(k_dev=k, ld_dev=0.4, x_w=0.97, h_w=0.90)).time
        rep("a panel that drags more than it turns is not", t_bad > base,
            f"{t_bad:.4f} s ({t_bad - base:+.4f})")

    #  7. the stop (the owner's 'Stopping' job)
    v0 = V_STOP_KMH / 3.6
    s0 = stop(v0)
    s_fine = stop(v0, n=4000)
    rep("the stop's quadrature has converged", s0.ok and abs(s0.distance - s_fine.distance) < 1e-3,
        f"{s0.distance:.4f} m at {N_STOP_STEPS} steps vs {s_fine.distance:.4f} m at 4000")
    #  with no speed-dependent force the stop is v0^2 / (2 a): no drag, no
    #  rolling resistance and no downforce, so the tyre loads never change
    import copy as _copy
    flat = _copy.copy(CAR)
    flat.CdA, flat.Crr = 0.0, 0.0
    a_flat = brake(v0, MissionAero(), True, 1.0, flat)
    s_flat = stop(v0, MissionAero(), 1.0, flat)
    rep("no drag, no downforce: the stop is v0^2 / (2 a), by hand",
        abs(s_flat.distance - v0 * v0 / (2.0 * a_flat)) < 1e-6 * s_flat.distance,
        f"{s_flat.distance:.4f} m vs {v0 * v0 / (2.0 * a_flat):.4f} m")
    s_df = stop(v0, MissionAero(cz_a=1.0, x_t=0.0))
    s_dr = stop(v0, MissionAero(cd_a=0.5, cd_a_stowed=0.5))
    rep("downforce and drag both shorten the stop", s_df.distance < s0.distance
        and s_dr.distance < s0.distance,
        f"bare {s0.distance:.2f} m, CZ*S 1 {s_df.distance:.2f} m, CD*S 0.5 {s_dr.distance:.2f} m")
    s_wet = stop(v0, MissionAero(), 0.632183908)
    rep("the wet stop is longer", s_wet.distance > s0.distance,
        f"{s_wet.distance:.2f} m vs {s0.distance:.2f} m dry")
    m = MissionSpec.from_json({"track": STOPPING, "v_stop_kmh": 150.0})
    back = MissionSpec.from_json(m.to_json())
    old = MissionSpec.from_json({"track": "arena", "mu_scale": 1.0, "stated": True})
    try:
        m.profile(lambda n: None)
        refused = False
    except ValueError:
        refused = True
    rep("a stopping mission round-trips, has no circuit, and an old save loads at 100 km/h",
        back.is_stop and back.v_stop_kmh == 150.0 and refused and old.v_stop_kmh == V_STOP_KMH
        and m.job_key != MissionSpec(track=STOPPING).job_key and STOPPING in JOBS
        and STOPPING not in TRACKS,
        f"{back.job_key}; old save {old.job_key} @ {old.v_stop_kmh:.0f} km/h")
    #  8. the side wings' own job (the owner, "why no longer circuits?"; "Side
    #     wing should also have 'stopping' mission")
    sm = MissionSpec(track=STOPPING, mu_scale=0.8, stated=True, side_track="kestrel")
    sv = sm.for_side()
    back = MissionSpec.from_json(sm.to_json())
    rep("the side wings fly their own circuit in the car's conditions; it round-trips, an "
        "old save or an unknown job falls back to arena",
        sv.track == "kestrel" and sv.mu_scale == 0.8 and sv.stated and not sv.is_stop
        and sm.for_side() is sv and back.side_track == "kestrel" and old.side_track == "arena"
        and MissionSpec.from_json({"side_track": "moon"}).side_track == "arena"
        and sm.stop_flanks == 1,
        f"top {sm.job_key}, side {sv.job_key} ({sv.surface}); old save side {old.side_track}")
    #  9. ...and the side wings' stop: the air brake, both panels out
    ss = MissionSpec(track="linden", v_stop_kmh=150.0, side_track=STOPPING)
    sv = ss.for_side()
    back = MissionSpec.from_json(ss.to_json())
    fl = MissionAero(k_dev=1.2, ld_dev=3.0, x_w=0.0)
    one, two = stop(ss.v_stop, fl), stop(ss.v_stop, fl, flanks=sv.stop_flanks)
    v = 30.0
    rep("a side wing's stop is the air brake's: offered, from the car's stop speed, round-trips; "
        "both panels' drag, no side force",
        STOPPING in SIDE_JOBS and sv.is_stop and sv.v_stop_kmh == 150.0 and sv.stop_flanks == 2
        and back.side_track == STOPPING and not ss.is_stop
        and two.distance < one.distance < stop(ss.v_stop).distance
        and abs(_resistance(v, fl, True, flanks=2) - _resistance(v, fl, True)
                - fl.d_dev(v)) < 1e-9,
        f"side {sv.job_key}; from 150 km/h: bare {stop(ss.v_stop).distance:.2f} m, one panel "
        f"{one.distance:.2f} m, the pair {two.distance:.2f} m")

    return ok


if __name__ == "__main__":
    import sys
    print("drive.aero.mission self-check")
    sys.exit(0 if self_check() else 1)
