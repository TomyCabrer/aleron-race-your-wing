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
from .baseline import driver_trim
from .policy import Policy, N_OBS   # noqa: F401  (re-exported for __main__)

#: Training and evaluation timesteps. See the package docstring for the
#: measured drift that chose them: 2 ms costs 0.128 m over 20 s and buys 2x,
#: 4 ms costs 2.618 m and is not worth it.
DT_TRAIN = 0.002
DT_EVAL = 0.001

#: Lookahead stations for the curvature the policy sees, m. 15 / 35 / 70 m was
#: roughly 0.5 / 1.2 / 2.5 s at 28 m/s -- a corner entry, a corner, and
#: whether there is another one after it.
#:
#: **120 m is new in wave 5 and it exists for one car.** The 540i reaches
#: 41.25 m/s on `open`'s 420 m straight and the corner starts at s = 420.1 m;
#: braking 41.2 -> 17 m/s at its measured 0.9 g needs ~80 m and a 70 m station
#: cannot buy that at ANY planned speed, so anchor and policy alike went off
#: at ~440 m in all six `open` cells of wave 4. 120 m is ~2.9 s at 41 m/s and
#: is the first station that lets the plan start before the car is committed.
LOOKAHEAD = (15.0, 35.0, 70.0, 120.0)

#: How far ahead the speed plan looks for GRIP, m, and the other half of wave
#: 5. Through wave 4 the observation carried no surface term at all, so
#: neither the hand-written driver nor the network could see `WET_T3` (the
#: arena's 130 m full-width mu = 0.632 patch at s = 455..585) coming; both RWD
#: cars drove into it at ~22 m/s on a dry plan and the whole `MARGIN_FADE`
#: apparatus existed to pay for that blindness EVERYWHERE, including on a dry
#: skidpad. 90 m is the braking horizon of the fastest car in the library at
#: the speeds these circuits reach, and it is swept in `.handoff/09-ml.md`.
MU_HORIZON = 90.0

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
    wing_frac: float = 0.0       # fraction of steps with a flank panel deployed
    wing_outer_frac: float = 0.0 # ... of THOSE, the fraction on the outer flank
    #  per wing, for the free-wings head (`policy.WING_PRIOR`): what the car
    #  actually did with its three wings. `wing_both_frac` is the air brake.
    wing_l_frac: float = 0.0     # fraction of steps with the LEFT flank panel out
    wing_r_frac: float = 0.0     # ... the RIGHT one
    wing_top_frac: float = 0.0   # ... the top wing (0.0 when the car has none)
    wing_both_frac: float = 0.0  # ... BOTH flanks at once
    ended: str = "time"          # 'time' | 'offtrack' | 'spin' | 'stall'
    util_f_max: float = 0.0
    util_r_max: float = 0.0


def observe(veh, tr, out=None, mu_here=None) -> np.ndarray:
    """The observation vector. Read-only in `veh`; allocates once if reused.

    `mu_here` is the surface scale under the car. `rollout` already calls
    `trk.surface_at` every step to drive the physics and passes the answer in,
    so the observation costs nothing there; any other caller may omit it and
    pay for one extra lookup.
    """
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
    o[8] = veh.beta
    o[9] = veh.r * 2.0
    o[10] = veh.ay / G
    o[11] = veh.util_f
    o[12] = veh.util_r
    o[13] = veh.wing_deploy
    o[14] = 1.0 if abs(n) <= half else 0.0
    o[15] = (trk.surface_at(tr, veh.x, veh.y)[0] if mu_here is None
             else float(mu_here))
    o[16] = mu_ahead(tr, s)
    return o


#: `{track fingerprint: (mu at each centreline sample, min over MU_HORIZON)}`.
#: Keyed by geometry AND the surface list rather than by `id(tr)`: ids get
#: recycled after a GC, and `make_arena(surfaces=False)` has the same name,
#: length and sample count as the default arena with none of its patches --
#: exactly the collision that would silently hand a policy the wrong grip.
_MU_CACHE: dict = {}


def _mu_key(tr):
    return (tr.name, round(float(tr.length), 6), len(tr.s),
            tuple(sorted((p.s0, p.s1, p.mu_scale) for p in tr.surfaces)),
            len(tr.areas or ()))


def _mu_profile(tr):
    """(mu at each centreline sample, running min over the next MU_HORIZON).

    Built ONCE per track per process, ~2500 `surface_at` calls and a 180-step
    vectorised sliding minimum: about 15 ms, against ~70 000 steps in a
    training rollout. Doing it per step instead would put six `track.project`
    calls in the 500 Hz observation path for a number that cannot change.

    Sampled ON THE CENTRELINE (n = 0). That is exact for `WET_T3`, which spans
    the full 12 m width, and conservative-in-the-right-direction for
    `DAMP_T5_EXIT` and `WET_T2_ENTRY`, which span n = -6..0 and so include the
    centreline. A patch that touched only one edge of the road would be
    invisible here; none of the four tracks has one.
    """
    key = _mu_key(tr)
    hit = _MU_CACHE.get(key)
    if hit is not None:
        return hit
    n = len(tr.s)
    prof = np.array([trk.surface_at(tr, float(tr.xy[i][0]),
                                    float(tr.xy[i][1]))[0] for i in range(n)])
    w = max(1, int(round(MU_HORIZON / tr.ds)))
    if tr.closed:
        ext = np.concatenate([prof, prof[:w + 1]])
    else:
        ext = np.concatenate([prof, np.full(w + 1, prof[-1])])
    run = ext[:n].copy()
    for j in range(1, w + 1):
        np.minimum(run, ext[j:n + j], out=run)
    _MU_CACHE[key] = (prof, run)
    return _MU_CACHE[key]


def mu_ahead(tr, s: float) -> float:
    """The WORST surface scale on the centreline in the next `MU_HORIZON` m.

    A minimum and not a discounted average, because grip is not something you
    can be a bit short of: you either arrive on the wet at a speed it will
    hold or you do not. Curvature IS discounted with distance (`K70_PLAN`,
    `K120_PLAN`) because a distant corner can still be braked for; a distant
    wet patch cannot be braked for once you are on it.
    """
    _prof, run = _mu_profile(tr)
    i = int(round(s / tr.ds))
    n = (len(tr.s) - 1) if tr.closed else len(tr.s)
    i = i % n if tr.closed else max(0, min(len(tr.s) - 1, i))
    return float(run[i])


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
    wing_side, top_deploy)` once per 20 steps -- for plotting a learned line.
    """
    tr = trk.make_track(track) if tr is None else tr
    car = CorsaC() if car is None else car
    #  THE CAR'S OWN GRIP CALIBRATION. `cars.CarSpec.mu_scale` is how the
    #  library carries the difference between a 2003 touring tyre and a modern
    #  performance one (there is exactly one Pacejka coefficient set in this
    #  repo -- `cars.py`'s header explains why), and `Vehicle` reads it from
    #  the CONFIG, not from the car: `vehicle.py` line 1526 is
    #  `mu[i] * cfg.mu_scale`. `drive.drive` passes it (drive.py:3630) and
    #  this did not, so until wave 4 every MX-5 and 540i rollout in this
    #  package was driven on the Corsa's tyres -- mu 1.00 where the library
    #  says 1.05 and 1.08. An explicit `cfg_kwargs['mu_scale']` still wins, so
    #  the wet sweeps keep working. The Corsa's is 1.0, so nothing measured on
    #  it moves by a bit.
    kw = dict(wing=wing, mu_scale=float(getattr(car, "mu_scale", 1.0)))
    kw.update(cfg_kwargs or {})
    veh = Vehicle(car, VehicleConfig(**kw))
    x0, y0, psi0 = trk.start_pose(tr)
    veh.reset(x0, y0, psi0, V0, gear=2)

    ep = Episode()
    #  the hand-written anchor's per-car calibration, computed ONCE: its
    #  lock, its wheelbase and the grip it plans to. Every field is exactly
    #  the Corsa's value on the Corsa (`baseline.driver_trim`), so no
    #  committed Corsa number moves.
    trim = driver_trim(veh.car, float(veh.cfg.mu_scale))
    lock_rad = float(getattr(veh, "lock_rad", trim["lock_rad"]))
    wheelbase = trim["wheelbase"]
    ay_plan = trim["ay_plan"]
    k_us = trim["k_us"]
    obs = np.empty(N_OBS)
    mu = [1.0, 1.0, 1.0, 1.0]
    crr = [1.0, 1.0, 1.0, 1.0]
    n_steps = int(round(T / dt))
    half = 0.5 * tr.width

    s_prev, _n, _k, _p, _i = trk.project(tr, veh.x, veh.y)
    s_sum = 0.0
    v_sum, wing_sum, outer_sum, t_slow = 0.0, 0.0, 0.0, 0.0
    l_sum = r_sum = top_sum = both_sum = 0.0

    for k in range(n_steps):
        t = k * dt                       # contract section 0: never accumulated
        m, c, _on = trk.surface_at(tr, veh.x, veh.y)
        observe(veh, tr, obs, mu_here=m)
        #  this car's own lock and wheelbase, not the Corsa's (the constants
        #  in `policy`/`baseline` are only the defaults). Both are exactly the
        #  Corsa's values on the Corsa, so every committed checkpoint's
        #  measured numbers are unmoved by either change.
        ctl = policy.controls(obs, Controls, lock_rad=lock_rad,
                              wheelbase=wheelbase, ay_plan=ay_plan,
                              k_us=k_us)
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
        dl, dr = veh.wing_deploy_l, veh.wing_deploy_r
        if dl > 0.05:
            l_sum += 1.0
            if dr > 0.05:
                both_sum += 1.0
        if dr > 0.05:
            r_sum += 1.0
        if veh.top_deploy > 0.05:
            top_sum += 1.0
        ep.v_max = max(ep.v_max, veh.u)
        ep.util_f_max = max(ep.util_f_max, veh.util_f)
        ep.util_r_max = max(ep.util_r_max, veh.util_r)
        if collect is not None and k % 20 == 0:
            collect.append((t, veh.x, veh.y, veh.psi, veh.u,
                            veh.wing_deploy, veh.wing_side, veh.top_deploy))

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
    ep.wing_l_frac = l_sum / n_done
    ep.wing_r_frac = r_sum / n_done
    ep.wing_top_frac = top_sum / n_done
    ep.wing_both_frac = both_sum / n_done
    ep.reward = W_PROGRESS * s_sum
    if ep.ended == "offtrack":
        ep.reward -= W_OFFTRACK
    elif ep.ended == "spin":
        ep.reward -= W_SPIN
    return ep


def lap_time(policy: Policy, track: str = "arena", *, wing: str = "plate",
             dt: float = DT_EVAL, T: float = 240.0, laps: int = 3,
             car=None, collect=None, cfg_kwargs=None, tr=None) -> dict:
    """Best flying lap at the EVALUATION timestep, for a reported number.

    Everything the ES sees is measured at `DT_TRAIN`; everything quoted to a
    human is measured here, at the contract's `DT_PHYS`.

    `car` is a `cars.CarSpec` (or `None` for the stock Corsa). It was added
    in wave 4 for the cross-car matrix: `rollout` had taken a car since the
    package was written, but the only function that reports a LAP TIME could
    not, so every quoted number in the study was a Corsa number by
    construction rather than by choice.

    `cfg_kwargs` / `tr` go to `rollout` unchanged: the car's whole config
    (a garage build's three wings, the session's assists and grip) and a
    track built with the session's own options. Both default to what this
    function always measured, so every quoted number is unmoved.
    """
    ep = rollout(policy, track, dt=dt, T=T, wing=wing, car=car,
                 collect=collect, cfg_kwargs=cfg_kwargs, tr=tr)
    times = ep.lap_times
    flying = [b - a for a, b in zip(times, times[1:])][:laps]
    return dict(best=min(flying) if flying else None, laps=ep.laps,
                flying=flying, ended=ep.ended, t=ep.t, s=ep.s_progress,
                v_mean=ep.v_mean, v_max=ep.v_max, wing_frac=ep.wing_frac,
                wing_outer_frac=ep.wing_outer_frac, reward=ep.reward,
                wing_l_frac=ep.wing_l_frac, wing_r_frac=ep.wing_r_frac,
                wing_top_frac=ep.wing_top_frac, wing_both_frac=ep.wing_both_frac,
                util_f_max=ep.util_f_max, util_r_max=ep.util_r_max)
