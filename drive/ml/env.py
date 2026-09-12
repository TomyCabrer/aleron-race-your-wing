"""The rollout: observation, reward, termination. No pygame, no wall clock.

A rollout is a pure function of (policy parameters, track name, seed) — the
same arguments give the same reward to the last bit, which is what makes an ES
reproducible and what CONTRACT section 0 demands of anything that drives the
physics path. Nothing here is imported by `drive/`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from corsa_c import CorsaC, G
from .. import track as trk
from ..vehicle import Vehicle, VehicleConfig, Controls
from .policy import Policy, N_OBS  # noqa: F401  (re-exported for drive.ml.__main__)

#: Training and evaluation timesteps. See the package docstring for the
#: measured drift that chose them: 2 ms costs 0.128 m over 20 s and buys 2x,
#: 4 ms costs 2.618 m and is not worth it.
DT_TRAIN = 0.002
DT_EVAL = 0.001

#: Lookahead stations for the curvature the policy sees, m. 15 / 35 / 70 m is
#: roughly 0.5 / 1.2 / 2.5 s at 28 m/s -- a corner entry, a corner, and
#: whether there is another one after it.
LOOKAHEAD = (15.0, 35.0, 70.0)

#: Off-track is not a soft cost. At `MARGIN_OFF` metres beyond the ribbon edge
#: the episode ENDS: a policy that learns to cut a corner across the grass is
#: not driving, and a soft penalty is exactly what it learns to pay.
MARGIN_OFF = 1.2

#: Episode ends early if the car is going nowhere -- a policy that stops is a
#: local optimum the ES finds immediately otherwise (reward 0 beats crashing).
V_STALL = 1.5
T_STALL = 3.0

#: Reward weights. Progress along the centreline dominates; everything else is
#: a shaping term small enough that it cannot outvote lap time.
W_PROGRESS = 1.0          # per metre of centreline advanced
W_OFFTRACK = 60.0         # one-off, on termination off the track
W_SPIN = 40.0             # one-off, on termination by spin
#  There is deliberately NO term charging the wing's drag or rewarding its
#  deployment: the drag is already in the physics, and paying for it twice
#  would teach the policy to stow a device that is actually earning its keep.
#  Whether to deploy has to fall out of lap time or it means nothing.


@dataclass
class Episode:
    """What one rollout produced. `reward` is what the ES maximises."""

    reward: float = 0.0
    s_progress: float = 0.0      # m advanced along the centreline
    t: float = 0.0               # s of simulated time
    laps: int = 0
    lap_times: list = field(default_factory=list)
    v_mean: float = 0.0
    v_max: float = 0.0
    wing_frac: float = 0.0       # fraction of steps with the panel deployed
    wing_outer_frac: float = 0.0 # ... of THOSE, the fraction on the outer flank
    ended: str = "time"          # 'time' | 'offtrack' | 'spin' | 'stall'
    util_f_max: float = 0.0
    util_r_max: float = 0.0


def observe(veh, tr, out=None) -> np.ndarray:
    """The observation vector. Read-only in `veh`; allocates once if reused."""
    o = np.empty(N_OBS) if out is None else out
    s, n, kappa, psi_c, _i = trk.project(tr, veh.x, veh.y)
    half = 0.5 * tr.width
    o[0] = veh.u / 40.0
    o[1] = n / max(half, 1e-6)
    o[2] = _wrap_pi(veh.psi - psi_c)
    o[3] = kappa * 50.0
    for j, d in enumerate(LOOKAHEAD):
        sj = (s + d) % tr.length if tr.closed else min(s + d, tr.length)
        o[4 + j] = _kappa_at(tr, sj) * 50.0
    o[7] = veh.beta
    o[8] = veh.r * 2.0
    o[9] = veh.ay / G
    o[10] = veh.util_f
    o[11] = veh.util_r
    o[12] = veh.wing_deploy
    o[13] = 1.0 if abs(n) <= half else 0.0
    return o


def _kappa_at(tr, s: float) -> float:
    i = int(round(s / tr.ds))
    n = (len(tr.s) - 1) if tr.closed else len(tr.s)
    i = i % n if tr.closed else max(0, min(len(tr.s) - 1, i))
    return float(tr.kappa[i])


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rollout(policy: Policy, track: str = "arena", *, dt: float = DT_TRAIN,
            T: float = 60.0, wing: str = "plate", car=None, cfg_kwargs=None,
            V0: float = 12.0, tr=None, collect=None) -> Episode:
    """Drive `policy` for `T` seconds of simulated time and score it.

    `wing` is the car's aero: 'off' | 'fin' | 'plate'. The agent is judged with
    the device FITTED and armed by its own `wing_on` output, so learning when
    to deploy it is part of the problem rather than a separate experiment.

    `collect`, if given, is a list that receives `(t, x, y, psi, u, wing_deploy,
    wing_side)` once per 20 steps -- for plotting a learned line.
    """
    tr = trk.make_track(track) if tr is None else tr
    car = CorsaC() if car is None else car
    kw = dict(wing=wing)
    kw.update(cfg_kwargs or {})
    veh = Vehicle(car, VehicleConfig(**kw))
    x0, y0, psi0 = trk.start_pose(tr)
    veh.reset(x0, y0, psi0, V0, gear=2)

    ep = Episode()
    obs = np.empty(N_OBS)
    mu = [1.0, 1.0, 1.0, 1.0]
    crr = [1.0, 1.0, 1.0, 1.0]
    n_steps = int(round(T / dt))
    half = 0.5 * tr.width

    s_prev, _n, _k, _p, _i = trk.project(tr, veh.x, veh.y)
    s_sum = 0.0
    v_sum, wing_sum, outer_sum, t_slow = 0.0, 0.0, 0.0, 0.0

    for k in range(n_steps):
        t = k * dt                       # contract section 0: never accumulated
        observe(veh, tr, obs)
        ctl = policy.controls(obs, Controls)
        m, c, _on = trk.surface_at(tr, veh.x, veh.y)
        mu[0] = mu[1] = mu[2] = mu[3] = m
        crr[0] = crr[1] = crr[2] = crr[3] = c
        veh.step(ctl, mu, crr, dt)

        s, n, _k2, _p2, _i2 = trk.project(tr, veh.x, veh.y)
        ds = s - s_prev
        if tr.closed:                    # unwrap the seam
            if ds < -0.5 * tr.length:
                ds += tr.length
                ep.laps += 1
                ep.lap_times.append(t)
            elif ds > 0.5 * tr.length:
                ds -= tr.length
        s_sum += ds
        s_prev = s

        v_sum += veh.u
        if veh.wing_deploy > 0.05:
            wing_sum += 1.0
            # the OUTER flank is the one the contract wants: wing_side is the
            # TURN sign, so the panel is outer whenever the sign of the yaw
            # rate agrees with it
            if veh.wing_side != 0 and veh.r * veh.wing_side > 0.0:
                outer_sum += 1.0
        ep.v_max = max(ep.v_max, veh.u)
        ep.util_f_max = max(ep.util_f_max, veh.util_f)
        ep.util_r_max = max(ep.util_r_max, veh.util_r)
        if collect is not None and k % 20 == 0:
            collect.append((t, veh.x, veh.y, veh.psi, veh.u,
                            veh.wing_deploy, veh.wing_side))

        # --- termination ---------------------------------------------------
        if abs(n) > half + MARGIN_OFF:
            ep.ended = "offtrack"
            ep.t = t
            break
        if abs(veh.beta) > 1.05 or abs(veh.r) > 2.5:      # 60 deg of sideslip
            ep.ended = "spin"
            ep.t = t
            break
        t_slow = t_slow + dt if veh.u < V_STALL else 0.0
        if t_slow > T_STALL:
            ep.ended = "stall"
            ep.t = t
            break
    else:
        ep.t = n_steps * dt

    n_done = max(k + 1, 1)
    ep.s_progress = s_sum
    ep.v_mean = v_sum / n_done
    ep.wing_frac = wing_sum / n_done
    ep.wing_outer_frac = (outer_sum / wing_sum) if wing_sum > 0 else 0.0
    ep.reward = W_PROGRESS * s_sum
    if ep.ended == "offtrack":
        ep.reward -= W_OFFTRACK
    elif ep.ended == "spin":
        ep.reward -= W_SPIN
    return ep


def lap_time(policy: Policy, track: str = "arena", *, wing: str = "plate",
             dt: float = DT_EVAL, T: float = 240.0, laps: int = 3) -> dict:
    """Best flying lap at the EVALUATION timestep, for a reported number.

    Everything the ES sees is measured at `DT_TRAIN`; everything quoted to a
    human is measured here, at the contract's `DT_PHYS`.
    """
    ep = rollout(policy, track, dt=dt, T=T, wing=wing)
    times = ep.lap_times
    flying = [b - a for a, b in zip(times, times[1:])][:laps]
    return dict(best=min(flying) if flying else None, laps=ep.laps,
                flying=flying, ended=ep.ended, t=ep.t, s=ep.s_progress,
                v_mean=ep.v_mean, v_max=ep.v_max, wing_frac=ep.wing_frac,
                wing_outer_frac=ep.wing_outer_frac, reward=ep.reward)
