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

#: Power per unit grip, normalised to the Corsa, above which this driver plans
#: to less grip than it would on the car its gains were swept on. Measured
#: 1.0000 / 1.7936 / 2.3170, so the Corsa is inert BY PHYSICS and not by a
#: flag: 1.0 is not greater than 1.25 and its margin is exactly 1.0.
POWER_GRIP_MODULATE = 1.25
#: How fast the planned grip fades with `power_grip_ratio`:
#: `margin = clip(1 - MARGIN_FADE*(pg - 1), MARGIN_MIN, 1)`.
#:
#: **0.40, not the 0.10 wave 2 shipped**, and the difference is the surface
#: blindness above -- LapDriver plans around WET_T3 per sample and this driver
#: cannot see it, so it has to be slower everywhere instead. Swept on the
#: arena, baseline theta = 0, dt 1 ms, 280 s, both RWD cars:
#:
#:      fade   mx5 (pg 1.79)                540i (pg 2.32)
#:      0.10   margin 0.921   SPIN  490 m   margin 0.868  off  515 m
#:      0.20   margin 0.841   off   528 m   margin 0.737  off  546 m
#:      0.30   margin 0.762   off   549 m   margin 0.605  LAPS 68.636 s
#:      0.40   margin 0.683   LAPS 66.602   margin 0.473  LAPS 74.837 s
#:      0.55   margin 0.564   LAPS 71.306   margin 0.276  LAPS 88.753 s
#:
#: 0.40 is the first value at which BOTH cars lap, and past it both get
#: monotonically slower, so it is the edge of the usable band rather than the
#: middle of one -- there is no margin on the low side and that is stated as a
#: limit, not hidden. `MARGIN_MIN` then floors the 540i at 0.55 instead of the
#: 0.473 the law would give, which is inside the range the sweep shows lapping
#: (0.473 and 0.605 both lap) and costs it 0.8 s a lap against 0.473.
MARGIN_FADE = 0.40
MARGIN_MIN = 0.55


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
    `K_N`, `K_V` and `KAPPA_ARM` are the same on every car on purpose -- what
    the residual sits on has to be ONE driver evaluated on three machines, or
    a cross-car comparison is comparing three different drivers.

    Every field is exactly its Corsa value for the Corsa, so `theta = 0` there
    is unchanged and every wave-4 item-1 number stands without a retrain:
    `car_lock_rad` is 32.625 deg, `L` is 2.491 m, both ratios are a number
    divided by itself, and `pg = 1.0` is not greater than 1.25.
    """
    pg = power_grip_ratio(car, mu_scale)
    margin = (min(max(1.0 - MARGIN_FADE * (pg - 1.0), MARGIN_MIN), 1.0)
              if pg > POWER_GRIP_MODULATE else 1.0)
    ay_ratio = _ay_peak(car, mu_scale) / _AY_REF
    from ..vehicle import car_lock_rad          # lazy: keeps import-time pure
    return dict(lock_rad=float(car_lock_rad(car)), wheelbase=float(car.L),
                ay_plan=AY_PLAN * ay_ratio * margin,
                power_grip=pg, margin_scale=margin, ay_ratio=ay_ratio)


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
    k0, k1, k2, k3 = (obs[3] / 50.0, obs[4] / 50.0,
                      obs[5] / 50.0, obs[6] / 50.0)

    # -- steering: feedforward on the curvature just ahead, feedback on where
    #    we are. delta > 0 steers LEFT and kappa > 0 IS a left turn, so the
    #    feedforward takes kappa's sign directly; n > 0 is LEFT of the
    #    centreline and psi_err > 0 points left of the tangent, so both
    #    feedback terms are negative.
    k_path = 0.6 * k0 + 0.4 * k1
    delta = K_FF * wheelbase * k_path - K_PSI * psi_err - K_N * n_norm
    steer = delta / lock_rad

    # -- speed: plan for the tightest curvature in the lookahead window, so
    #    the braking starts before the corner rather than in it
    k_plan = max(abs(k0), abs(k1), 0.8 * abs(k2),
                 K70_PLAN * abs(k3))
    v_tgt = (math.sqrt(ay_plan / k_plan) if k_plan > 1e-6 else V_MAX_PLAN)
    v_tgt = min(max(v_tgt, V_MIN_PLAN), V_MAX_PLAN)
    pedal = K_V * (v_tgt - u)

    # -- the wing: arm it for the corner ahead, and keep it armed while the
    #    front axle is actually working (the device buys front grip)
    wing = 1.0 if (max(abs(k1), abs(k2)) > KAPPA_ARM or obs[10] > 0.75) else -1.0

    return np.array([_clip1(steer), _clip1(pedal), wing])


def _clip1(v: float) -> float:
    return -1.0 if v < -1.0 else (1.0 if v > 1.0 else float(v))
