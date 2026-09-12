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

# --- the wing ---------------------------------------------------------------
#: Arm the device when the corner 35 m ahead is tight enough to be worth it.
#: `crossover.py` puts the device's useful band at small radii; 1/220 m is
#: where the panel starts paying for its drag on this car.
KAPPA_ARM = 1.0 / 220.0


def baseline_action(obs, lock_rad: float) -> np.ndarray:
    """(N_OBS,) -> (steer, pedal, wing) each in [-1, 1], the policy's own units.

    `steer` is a FRACTION of lock, not radians, so it composes with the
    network's tanh output directly. `lock_rad` is passed in rather than
    imported from `policy`, which imports THIS module for the residual: one
    argument instead of an import cycle or a duplicated constant.
    """
    u = obs[0] * 40.0
    n_norm = obs[1]
    psi_err = obs[2]
    k0, k1, k2 = obs[3] / 50.0, obs[4] / 50.0, obs[5] / 50.0

    # -- steering: feedforward on the curvature just ahead, feedback on where
    #    we are. delta > 0 steers LEFT and kappa > 0 IS a left turn, so the
    #    feedforward takes kappa's sign directly; n > 0 is LEFT of the
    #    centreline and psi_err > 0 points left of the tangent, so both
    #    feedback terms are negative.
    k_path = 0.6 * k0 + 0.4 * k1
    delta = K_FF * _CAR.L * k_path - K_PSI * psi_err - K_N * n_norm
    steer = delta / lock_rad

    # -- speed: plan for the tightest curvature in the lookahead window, so
    #    the braking starts before the corner rather than in it
    k_plan = max(abs(k0), abs(k1), 0.8 * abs(k2))
    v_tgt = (math.sqrt(AY_PLAN / k_plan) if k_plan > 1e-6 else V_MAX_PLAN)
    v_tgt = min(max(v_tgt, V_MIN_PLAN), V_MAX_PLAN)
    pedal = K_V * (v_tgt - u)

    # -- the wing: arm it for the corner ahead, and keep it armed while the
    #    front axle is actually working (the device buys front grip)
    wing = 1.0 if (max(abs(k1), abs(k2)) > KAPPA_ARM or obs[10] > 0.75) else -1.0

    return np.array([_clip1(steer), _clip1(pedal), wing])


def _clip1(v: float) -> float:
    return -1.0 if v < -1.0 else (1.0 if v > 1.0 else float(v))
