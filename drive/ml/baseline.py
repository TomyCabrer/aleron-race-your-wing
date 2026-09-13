"""A hand-written driver, for the BEFORE number and for the ES to stand on.

Two jobs:

1. **The control.** Nothing here is learned. It is the obvious thing: pure
   pursuit on the centreline, a curvature-limited target speed, and a wing
   rule that arms the device when the car is about to need it. It is the
   honest baseline a learned policy has to beat, and quoting a learned lap
   time without one would be meaningless.

2. **The origin for the residual policy.** `policy.Policy` is a RESIDUAL on
   this: `action = baseline(obs) + RESID_GAIN * net(obs)`, so `theta = 0` is
   exactly this driver and the ES starts on the track at racing speed instead
   of spending its first thousand rollouts discovering that grass is slow.
   The ES then learns the LINE and the AERO USE, which is the interesting
   part and the part the owner asked for.

It reads the same `env.observe` vector the network does, so the comparison is
fair: the learned policy is given no information the baseline lacks.
"""

from __future__ import annotations

import math

import numpy as np

import qss
from corsa_c import CorsaC, G

_CAR = CorsaC()

#: The CORSA's wheelbase, m, and the DEFAULT only -- exactly the story
#: `policy.LOCK_RAD` already tells. The pure-pursuit feedforward below is the
#: bicycle angle `L * kappa`, so `L` is a property of the CAR, and wave 4
#: pointed this driver at three of them: 2.491 m (corsa), 2.265 (mx5), 2.830
#: (540i). Left hard-wired, the 540i would have been given 14 % too much
#: feedforward and the MX-5 10 % too little -- a systematic steering bias
#: dressed up as a policy result. `env.rollout` passes `veh.car.L`; for the
#: Corsa it IS this number, so every committed checkpoint is unmoved.
WHEELBASE = _CAR.L

# --- pure pursuit -----------------------------------------------------------
#  Gains from a 36-point grid sweep over (K_PSI, K_N, AY_PLAN, K_V) on the
#  arena, scored on distance covered in 120 s with the plate fitted. The
#  shipped set is the only one that both laps and keeps the car inside the
#  ribbon for the whole run; the pre-sweep set (0.85 / 0.55 / 0.80 g / 0.45)
#  went off at 31.9 s having covered 562 m, so it could not post a lap at all.
K_PSI = 1.80         # rad of road wheel per rad of heading error
K_N = 1.60           # ... per unit of normalised lateral offset
K_FF = 1.00          # feedforward fraction of the bicycle angle L*kappa

# --- speed ------------------------------------------------------------------
#: The cornering limit this driver plans to: 0.75 g, comfortably inside the
#: car's measured 0.855 g peak (CONTRACT section 4, group W) so that the
#: baseline is a competent driver rather than one permanently at the limit.
AY_PLAN = 0.75 * G
V_MAX_PLAN = 42.0    # m/s, above the Corsa's 47.2 Vmax after drag anyway
V_MIN_PLAN = 7.0     # m/s, so a hairpin does not command a stop
K_V = 0.45           # pedal per m/s of speed error
#: How much of the 70 m lookahead's curvature enters the speed plan.
#:
#: Wave 3 shipped a plan that read `obs[3]/obs[4]/obs[5]` -- the 0 / 15 / 35 m
#: lookaheads -- and never `obs[6]`, the 70 m one. That is a driver who cannot
#: see past the next 35 m: on `open` it held full throttle to s = 385 m, braked
#: for exactly 35 m, and arrived at an R = 45 m corner at 22.21 m/s against the
#: 18.20 m/s its own 0.75 g plan asked for. 32.3 -> 18.2 m/s at the measured
#: 0.82 g needs 44.5 m. It was off the road at s = 451.5 m, every run.
#:
#: A far station is DISCOUNTED rather than obeyed, because obeying it would
#: mean cornering speed on the straight that leads to the corner: with weight
#: `c` the plan at that station is `v_corner / sqrt(c)`, so the car arrives at
#: a speed the nearer stations can still brake off. 0.8 at 35 m (1.12x) was
#: already here.
#:
#: **0.6 -- the value queued in the handoff note -- is WRONG, and the sweep is
#: why this constant exists instead of a literal.** Baseline (theta = 0) at
#: dt = 1 ms over 280 s, wing plate:
#:
#:      c      arena            open
#:      0.00   64.565 s   4L    off at  451.5 m     <- wave 3, the 35 m driver
#:      0.30   64.565 s   4L    off at 1277.1 m
#:      0.35   64.255 s   4L    71.702 s   3L
#:      0.40   64.491 s   4L    71.900 s   3L
#:      0.45   64.874 s   4L    72.182 s   3L       <- shipped
#:      0.50   65.164 s   4L    72.600 s   3L
#:      0.55   65.381 s   4L    72.958 s   3L
#:      0.60   off at 1023.7 m  73.223 s   3L       <- the queued value
#:      0.70   off at 1014.9 m  73.586 s   3L
#:      1.00   off at 1014.2 m  74.122 s   3L
#:
#: So the usable band is 0.35 .. 0.55, with a cliff at each end, and 0.45 is
#: the middle of it rather than a tuned number: 0.10 of margin from a value
#: that cannot drive `open` and 0.10 from one that cannot drive the arena. It
#: costs 0.31 s a lap on the arena against the fastest swept value and 0.48 s
#: on `open`, and an ANCHOR is worth more as a driver that does not fall off
#: cliffs than as the last half per cent of lap time -- especially now that it
#: is pointed at three cars whose brakes differ by 75 %.
#:
#: The 0.60 arena failure is not "too slow", it is COMBINED BRAKING ON A LOW-MU
#: PATCH, instrumented: the arena's mu = 0.55 patch sits at s ~ 1010..1030 m,
#: 70 m after a corner, so at c >= 0.6 the driver is on the brakes (0.28 then
#: 0.66) as it turns in on mu 0.80 -> 0.55, spends its friction circle
#: longitudinally and runs wide with the steering saturated at full lock. At
#: c = 0.45 it is 2.3 m/s FASTER through the same patch and holds n = -1.0 m
#: where c = 0.60 is at n = -6.1 m and gone. A slower driver ran wide; that
#: was worth instrumenting rather than assuming.
K70_PLAN = 0.45
#: The same discount for the 120 m station, swept the same way -- see
#: `.handoff/09-ml.md`. It exists because the 540i could not brake for
#: `open`'s R = 45 m corner from 41.25 m/s with 70 m of warning at any planned
#: speed, in either the anchor or any policy: off at ~440 m in all six wave-4
#: `open` cells. At weight `c` the plan at 120 m out is `v_corner / sqrt(c)`.
K120_PLAN = 0.30
#: Floor on the `mu_ahead` the speed plan will believe. Purely defensive: the
#: lowest real value on any centreline in the library is `WET_T3`'s 0.632 and
#: the lowest reachable anywhere is `track.MU_OFF_TRACK`.
MU_FLOOR = 0.05
#: How many of the five speed-plan stations (here / 15 / 35 / 70 / 120 m) are
#: planned on `mu_here` rather than on `mu_ahead`. Swept -- see the block at
#: `baseline_action`'s speed plan and `.handoff/09-ml.md`.
MU_NEAR_STATIONS = 1

# --- driving a car these gains were never swept on ---------------------------
#  Everything above is a 1010 kg front-driven hatch's driver: pure pursuit, a
#  curvature speed plan, and a wing rule. Point it at a car that can break its
#  driven axle and it spins -- measured, arena, plate, dt 1 ms: the MX-5 at
#  485.0 m and the 540i at 487.9 m, both on every run.
#
#  **The cause is not the driven axle. It is WET_T3.** Instrumented, both cars
#  lose it between s = 470 and 515 m with beta climbing past 1 rad, and
#  `make_arena`'s surface list puts `WET_T3` at **s = 455..585 m, full width,
#  mu_scale = 0.632**. They are entering a 130 m wet patch at ~22 m/s on a
#  speed plan plotted for dry tarmac. The Corsa survives it because 55 kW
#  cannot get there fast enough to care.
#
#  That is the difference from the scripted `LapDriver`, which solved the same
#  symptom in wave 2 (.handoff/12-lapdriver.md, "RESOLVED"): its
#  `speed_profile` is **surface-aware per centreline sample** -- its own
#  docstring says making the envelope surface-aware "is not optional on
#  CIRCUIT_ARENA: WET_T3 puts mu_scale 0.632 through the fastest grip-limited"
#  corner -- so it plans around the patch. THIS driver cannot: `env.observe`'s
#  14 entries carry no surface term at all, and adding one would change N_OBS
#  and invalidate every committed checkpoint. So it has to pay for the
#  blindness with a blanket margin instead, and that is what `MARGIN_FADE`
#  below is: not a traction-control constant, a **grip-ignorance** constant.
#
#  TRIED AND DELETED, in the order they were tried:
#
#    1. throttle capped by the driven axle's MEASURED `util_r`. Not even
#       attempted here -- wave 2 measured it as a step too late every time
#       (`util_r` only rises once the rear is already sliding) and left the
#       540i 1371 m off the track.
#    2. the same cap FEEDFORWARD on the corner the driver can see
#       (`q = V^2|k0|/ay_plan`, `room = sqrt(1 - q^2)`, floored at 0.15). This
#       is what LapDriver ships, it was implemented here, and on this driver
#       it **earns nothing and costs a little**: at the shipped margin, arena
#       mx5 66.602 s with the cap against **66.339 s without**, 540i 74.073
#       against **74.022**, skidpad identical to the millisecond, and it
#       rescued no cell either way. DELETED. The reason it is redundant is
#       structural: LapDriver FLOORS the throttle and needs something to take
#       it away, while `pedal = K_V * (v_tgt - u)` is already a proportional
#       controller on speed -- it is a soft throttle by construction, and
#       capping a soft throttle is capping something that is already short.
#    3. a counter-steer term `delta += k*beta`. Also not attempted: wave 2
#       swept k over 0..4 on the 540i, never completed a lap, and made
#       `max |n|` WORSE (2.74 m at k = 0, 3.25 at k = 1, 4.46 at k = 2).
#
#  What survives is the entry speed, and only the entry speed.

#: Power per unit grip, (P_wheel/m) / peak a_y, as a multiple of the Corsa's.
#: Measured 1.0000 / 1.7936 / 2.3170. **Wave 5 demoted this from a knob to a
#: diagnostic.** Through wave 4 it gated a blanket `MARGIN_FADE` that took
#: 45 % off the 540i's planned grip EVERYWHERE, because the driver could not
#: see the arena's wet patch and had to be slow all the time to survive it.
#: The observation now carries the grip, so the driver is slow where the water
#: is and nowhere else, and **the fade is retired: `MARGIN_FADE` is gone, not
#: set to zero.** It is reported in `driver_trim` because it is a real
#: property of the car and the note cites it; nothing reads it.
#:
#: What retiring it is worth, anchor, dt 1 ms, 280 s, plate (wave 4 -> wave 5):
#:
#:      corsa   arena  64.874 -> 65.943   open  72.182 -> 72.243   skid 17.055 -> 17.055
#:      mx5     arena  66.339 -> 62.862   open  70.892 -> 65.889   skid 19.321 -> 16.567
#:      540i    arena  74.022 -> 61.654   open  OFF 444 -> 63.464  skid 22.082 -> 16.712
#:
#: The 540i is **12.4 s a lap faster on the arena and 5.4 s on the skidpad**,
#: and it drives `open` at all for the first time. The Corsa pays 1.07 s on
#: the arena, which is the honest price of a driver that now slows for water
#: it used to blast through and get away with.


def _ay_peak(car, mu_scale: float = 1.0, roll_dist_f: float = 0.74) -> float:
    """Peak sustainable a_y for this car, m/s^2, from `qss.axle_capacity`.

    The same closed form as `drive.drive.car_ay_peak` and deliberately a copy
    of it rather than an import: `drive.drive` is the pygame CLI and
    `baseline.py` is imported by `policy.py`, which the package docstring
    promises is importable with the physics absent. Ten lines of bisection is
    a cheaper price than that promise.

    Known bias, from the wave-2 measurement: good to 0.3 % on the Corsa and
    the MX-5 and **4.1 % HIGH on the 540i** (8.8534 against 8.5046 from an
    open-loop ramp steer), because it omits the scrub-drag and yaw-balance
    terms `qss.max_ay` carries. That matters where the number is used as an
    absolute limit; here it is only ever a RATIO between two cars, multiplied
    by a margin that takes 45 % off the 540i, so 4 % the optimistic way is
    absorbed many times over. `drive.drive` needs the ramp steer because it
    plans to this number directly; this does not.
    """
    W = car.m * G
    Fz_f, Fz_r = W * car.wdist_f, W * (1.0 - car.wdist_f)
    lo, hi = 0.1, 30.0
    for _ in range(200):
        a = 0.5 * (lo + hi)
        dFz = car.m * a * car.h_cg / car.t
        cap = (qss.axle_capacity(Fz_f, roll_dist_f * dFz, mu_scale)
               + qss.axle_capacity(Fz_r, (1.0 - roll_dist_f) * dFz, mu_scale))
        lo, hi = (a, hi) if car.m * a < cap else (lo, a)
    return 0.5 * (lo + hi)


_AY_REF = _ay_peak(_CAR)
_PG_REF = (_CAR.P_wheel / _CAR.m) / _AY_REF


def power_grip_ratio(car, mu_scale: float = 1.0) -> float:
    """(P_wheel/m) / peak a_y, as a multiple of the Corsa's. Exactly 1.0 for
    the Corsa -- the same arithmetic on bit-identical fields."""
    return ((car.P_wheel / car.m) / _ay_peak(car, mu_scale)) / _PG_REF


def driver_trim(car, mu_scale: float = 1.0) -> dict:
    """The per-car calibration of this one hand-written driver.

    Three numbers, and they are the ONLY things the anchor takes from the car:
    its steering lock, its wheelbase, and the grip it plans to. `K_PSI`,
    `K_N`, `K_V`, `KAPPA_ARM`, `K70_PLAN`, `K120_PLAN` and
    `MU_NEAR_STATIONS` are the same on every car on purpose -- what the
    residual sits on has to be ONE driver evaluated on three machines, or a
    cross-car comparison is comparing three different drivers.

    `ay_plan` is now the car's own peak lateral capability and nothing else:
    `AY_PLAN * ay_ratio`, a ratio of two closed-form grip estimates. It is
    exactly `AY_PLAN` for the Corsa (a number divided by itself), as are the
    lock and the wheelbase, so `theta = 0` on the Corsa is what it always was.
    The per-car SPEED conservatism that used to live here is gone; the
    observation carries the grip now.
    """
    ay_ratio = _ay_peak(car, mu_scale) / _AY_REF
    from ..vehicle import car_lock_rad          # lazy: keeps import-time pure
    return dict(lock_rad=float(car_lock_rad(car)), wheelbase=float(car.L),
                ay_plan=AY_PLAN * ay_ratio, ay_ratio=ay_ratio,
                power_grip=power_grip_ratio(car, mu_scale))


# --- the wing ---------------------------------------------------------------
#: Arm the device when the corner 35 m ahead is tight enough to be worth it.
#: `crossover.py` puts the device's useful band at small radii; 1/220 m is
#: where the panel starts paying for its drag on this car.
KAPPA_ARM = 1.0 / 220.0


def baseline_action(obs, lock_rad: float, wheelbase: float = WHEELBASE,
                   ay_plan: float = AY_PLAN) -> np.ndarray:
    """(N_OBS,) -> (steer, pedal, wing) each in [-1, 1], the policy's own units.

    `steer` is a FRACTION of lock, not radians, so it composes with the
    network's tanh output directly. `lock_rad` and `wheelbase` are passed in
    rather than imported from `policy`, which imports THIS module for the
    residual: two arguments instead of an import cycle or a duplicated
    constant. `ay_plan` comes from `driver_trim` and is the grip this car's
    driver plans to -- see the block comment above it. All three are exactly
    the Corsa's values for the Corsa, so `theta = 0` there is bit-for-bit
    what it was. `K_PSI`, `K_N`, `K_V` and `KAPPA_ARM` are
    deliberately the same numbers on every car, so that the anchor the
    residual sits on is ONE hand-written driver evaluated on three machines
    and not three different drivers.
    """
    u = obs[0] * 40.0
    n_norm = obs[1]
    psi_err = obs[2]
    k0, k1, k2, k3, k4 = (obs[3] / 50.0, obs[4] / 50.0, obs[5] / 50.0,
                          obs[6] / 50.0, obs[7] / 50.0)
    mu_here, mu_ahead = obs[15], obs[16]

    # -- steering: feedforward on the curvature just ahead, feedback on where
    #    we are. delta > 0 steers LEFT and kappa > 0 IS a left turn, so the
    #    feedforward takes kappa's sign directly; n > 0 is LEFT of the
    #    centreline and psi_err > 0 points left of the tangent, so both
    #    feedback terms are negative.
    k_path = 0.6 * k0 + 0.4 * k1
    delta = K_FF * wheelbase * k_path - K_PSI * psi_err - K_N * n_norm
    steer = delta / lock_rad

    # -- speed: plan for the tightest curvature in the lookahead window, so
    #    the braking starts before the corner rather than in it -- AND for the
    #    grip it will have when it gets there.
    #
    #    Two bands, not one, and that detail is worth 1.7 s a lap on the
    #    Corsa. The obvious version multiplies ONE planned grip by
    #    `mu_ahead` (the worst surface in the next 90 m) and it is far too
    #    pessimistic: approaching `WET_T3` on dry tarmac it plans wet-grip
    #    cornering speed for the DRY corner it is still in. Pairing each
    #    station's curvature with the grip that station will actually have
    #    costs nothing and is what a driver does -- the wet only limits you
    #    where the wet is.
    #
    #      near   here and 15 m   -> `mu_here`, the surface under the car
    #      far    35 / 70 / 120 m -> `mu_ahead`, the worst in the next 90 m
    #
    #    With `mu_here == mu_ahead` this is EXACTLY the old single-band
    #    `max()` form, so on a dry circuit nothing changed. The seam is the
    #    15 m station, which is planned on the grip under the car rather than
    #    the grip 15 m away: 0.6 s of error at 25 m/s, and the alternative
    #    (a `mu` observation per station) is five more observations.
    stn = ((abs(k0), abs(k1), 0.8 * abs(k2),
            K70_PLAN * abs(k3), K120_PLAN * abs(k4)))
    k_near = max(stn[:MU_NEAR_STATIONS]) if MU_NEAR_STATIONS else 0.0
    k_far = max(stn[MU_NEAR_STATIONS:]) if MU_NEAR_STATIONS < 5 else 0.0
    #  clamped, because these are the first observation entries whose value
    #  can make the expression undefined rather than merely wrong: a negative
    #  `mu` gives sqrt() of a negative number. `observe` can only ever produce
    #  (0, 1], but the self-check feeds random normals through this function
    #  on purpose, and a driver that raises on a nonsense input is worse than
    #  one that saturates on it.
    ay_near = ay_plan * min(max(mu_here, MU_FLOOR), 1.0)
    ay_far = ay_plan * min(max(mu_ahead, MU_FLOOR), 1.0)
    v_tgt = min(math.sqrt(ay_near / k_near) if k_near > 1e-6 else V_MAX_PLAN,
                math.sqrt(ay_far / k_far) if k_far > 1e-6 else V_MAX_PLAN)
    v_tgt = min(max(v_tgt, V_MIN_PLAN), V_MAX_PLAN)
    pedal = K_V * (v_tgt - u)

    # -- the wing: arm it for the corner ahead, and keep it armed while the
    #    front axle is actually working (the device buys front grip)
    wing = 1.0 if (max(abs(k1), abs(k2)) > KAPPA_ARM or obs[11] > 0.75) else -1.0

    return np.array([_clip1(steer), _clip1(pedal), wing])


def _clip1(v: float) -> float:
    return -1.0 if v < -1.0 else (1.0 if v > 1.0 else float(v))
