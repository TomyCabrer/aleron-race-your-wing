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

# --- the wing ---------------------------------------------------------------
#: Arm the device when the corner 35 m ahead is tight enough to be worth it.
#: `crossover.py` puts the device's useful band at small radii; 1/220 m is
#: where the panel starts paying for its drag on this car.
KAPPA_ARM = 1.0 / 220.0


def baseline_action(obs, lock_rad: float,
                   wheelbase: float = WHEELBASE) -> np.ndarray:
    """(N_OBS,) -> (steer, pedal, wing) each in [-1, 1], the policy's own units.

    `steer` is a FRACTION of lock, not radians, so it composes with the
    network's tanh output directly. `lock_rad` and `wheelbase` are passed in
    rather than imported from `policy`, which imports THIS module for the
    residual: two arguments instead of an import cycle or a duplicated
    constant. They are the ONLY two things this driver takes from the car --
    `AY_PLAN`, `K_PSI`, `K_N` and `K_V` are deliberately the same numbers on
    every car, so that the anchor the residual sits on is ONE hand-written
    driver evaluated on three machines and not three different drivers. It is
    a Corsa-swept driver; wave 4 measures exactly how badly that travels.
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
    v_tgt = (math.sqrt(AY_PLAN / k_plan) if k_plan > 1e-6 else V_MAX_PLAN)
    v_tgt = min(max(v_tgt, V_MIN_PLAN), V_MAX_PLAN)
    pedal = K_V * (v_tgt - u)

    # -- the wing: arm it for the corner ahead, and keep it armed while the
    #    front axle is actually working (the device buys front grip)
    wing = 1.0 if (max(abs(k1), abs(k2)) > KAPPA_ARM or obs[10] > 0.75) else -1.0

    return np.array([_clip1(steer), _clip1(pedal), wing])


def _clip1(v: float) -> float:
    return -1.0 if v < -1.0 else (1.0 if v > 1.0 else float(v))
