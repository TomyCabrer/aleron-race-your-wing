"""Behavioural cloning: a lap the USER drove -> a starting genome for the swarm.

The sim records a "seed lap" (drive.drive: `K` arms it, the next complete and
valid lap from the start line is written to `runs/swarm/seed_*.json`). It is a
plain 100 Hz table of the car's published state plus the controls the driver
gave it -- nothing is taken out of the integrator, and the sim writes it with
no import of this package (CONTRACT: `drive/` never imports `drive.ml`).

This module turns that table into (observation, action) pairs in the policy's
own units and fits the residual network to them by plain gradient descent in
numpy. The result is a `Policy.theta` that DRIVES LIKE THE USER as closely as
356 parameters on top of the hand-written anchor can, and the swarm's
generation 0 is that genome plus mutations of it. Compounding error means a
clone rarely completes a lap on its own; that is what the GA is for.

The action target per row is

    steer = delta / lock_rad            (a fraction of THIS car's lock)
    pedal = throttle - brake            (one right foot, as the policy has)
    wing  = +0.8 if wing_on else -0.8   (tanh targets, not the rails)

and with the FREE-WINGS head (`n_act = 5`, the swarm's default) the one wing
target becomes three: the panel that WAS out (the seed lap records the
deploy and the yaw rate, so a deployed panel in a right turn is the left
one), and the top wing whenever the user braked with the wing armed.

and because the policy is `baseline + RESID_GAIN * net`, the NET's target is
`clip((target - baseline) / RESID_GAIN, -1, 1)` -- the clone learns only the
part of the user's driving the anchor does not already do.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np

from .baseline import baseline_action, driver_trim
from .env import observe
from .policy import (Policy, N_OBS, N_ACT, N_ACT_FREE, N_HID, RESID_GAIN,
                     expand_base)

SEED_LAP_KIND = "carsim-seed-lap-1"

#: the row schema the sim writes; kept here so the reader and the writer
#: agree by NAME (the sim looks the list up through a plain JSON header, not
#: through this module)
SEED_COLS = ("t", "x", "y", "psi", "u", "v", "r", "beta", "ay", "util_f",
             "util_r", "wing_deploy", "delta", "throttle", "brake", "wing_on")

WING_TARGET = 0.8


def load_seed_lap(path: str) -> dict:
    with open(path) as fh:
        d = json.load(fh)
    if d.get("kind") != SEED_LAP_KIND:
        raise ValueError(f"{path}: not a seed lap (kind={d.get('kind')!r})")
    cols = list(d["cols"])
    rows = np.asarray(d["rows"], dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != len(cols):
        raise ValueError(f"{path}: rows {rows.shape} do not match {len(cols)} columns")
    d["cols"], d["rows"] = cols, rows
    return d


def pairs(seed: dict, tr, car=None, mu_scale: float = 1.0,
          n_act: int = N_ACT_FREE):
    """(obs (N, N_OBS), target net output (N, n_out), baseline (N, n_out))
    from a loaded seed lap on track `tr`. `car` is the `cars.CarSpec` the lap
    was driven in (None = stock Corsa); `n_act` picks the head (4 = the one
    wing output, 5 = free wings)."""
    n_out = N_ACT_FREE if n_act >= N_ACT_FREE else 3
    cols, rows = seed["cols"], seed["rows"]
    ix = {c: i for i, c in enumerate(cols)}
    lock_rad = float(seed.get("lock_rad") or 0.0)
    if car is None:
        from corsa_c import CorsaC
        car = CorsaC()
    trim = driver_trim(car, float(mu_scale))
    if lock_rad <= 0.0:
        lock_rad = trim["lock_rad"]
    n = len(rows)
    obs = np.empty((n, N_OBS))
    base = np.empty((n, n_out))
    tgt = np.empty((n, n_out))
    fake = SimpleNamespace()
    for k in range(n):
        r = rows[k]
        fake.x, fake.y, fake.psi = r[ix["x"]], r[ix["y"]], r[ix["psi"]]
        fake.u, fake.beta, fake.r, fake.ay = (r[ix["u"]], r[ix["beta"]],
                                              r[ix["r"]], r[ix["ay"]])
        fake.util_f, fake.util_r = r[ix["util_f"]], r[ix["util_r"]]
        fake.wing_deploy = r[ix["wing_deploy"]]
        observe(fake, tr, obs[k])
        b3 = baseline_action(obs[k], lock_rad, trim["wheelbase"], trim["ay_plan"])
        base[k] = expand_base(b3, obs[k]) if n_out == N_ACT_FREE else b3
        tgt[k, 0] = r[ix["delta"]] / lock_rad
        tgt[k, 1] = r[ix["throttle"]] - r[ix["brake"]]
        on = r[ix["wing_on"]] > 0.5
        if n_out == N_ACT_FREE:
            out = r[ix["wing_deploy"]] > 0.05
            tgt[k, 2] = WING_TARGET if (out and r[ix["r"]] < 0.0) else -WING_TARGET
            tgt[k, 3] = WING_TARGET if (out and r[ix["r"]] > 0.0) else -WING_TARGET
            tgt[k, 4] = WING_TARGET if (on and r[ix["brake"]] > 0.05) else -WING_TARGET
        else:
            tgt[k, 2] = WING_TARGET if on else -WING_TARGET
    net_tgt = np.clip((tgt - base) / RESID_GAIN, -1.0, 1.0)
    return obs, net_tgt, base


def fit(obs, net_tgt, *, steps: int = 3000, lr: float = 3e-3, l2: float = 1e-4,
        seed: int = 0, max_rows: int = 4000, verbose: bool = False) -> tuple:
    """Adam on the residual net's MSE. Returns (theta, report). The head
    follows the target's width: 3 columns -> the 4-output policy, 5 -> free
    wings."""
    rng = np.random.default_rng(seed)
    n = len(obs)
    n_out = int(np.asarray(net_tgt).shape[1])
    n_act = N_ACT_FREE if n_out == N_ACT_FREE else N_ACT
    if n > max_rows:
        sel = np.sort(rng.choice(n, max_rows, replace=False))
        obs, net_tgt = obs[sel], net_tgt[sel]
        n = max_rows
    X = np.asarray(obs, float)
    Y = np.asarray(net_tgt, float)
    theta = rng.normal(0.0, 0.05, Policy.n_param(n_act))
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    n1 = N_OBS * N_HID
    loss = float("nan")
    for it in range(1, steps + 1):
        W1 = theta[:n1].reshape(N_HID, N_OBS)
        bb1 = theta[n1:n1 + N_HID]
        W2 = theta[n1 + N_HID:n1 + N_HID + N_HID * n_act].reshape(n_act, N_HID)
        bb2 = theta[n1 + N_HID + N_HID * n_act:]
        H = np.tanh(X @ W1.T + bb1)            # (n, hid)
        Z = H @ W2.T + bb2                      # (n, act)
        A = np.tanh(Z)
        E = A[:, :n_out] - Y                    # only the used outputs
        loss = float(np.mean(E ** 2))
        dA = np.zeros_like(A)
        dA[:, :n_out] = 2.0 * E / (n_out * n)
        dZ = dA * (1.0 - A ** 2)
        gW2 = dZ.T @ H
        gb2 = dZ.sum(0)
        dH = dZ @ W2
        dP = dH * (1.0 - H ** 2)
        gW1 = dP.T @ X
        gb1 = dP.sum(0)
        g = np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2]) + l2 * theta
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** it)
        vh = v / (1 - b2 ** it)
        theta = theta - lr * mh / (np.sqrt(vh) + eps)
        if verbose and (it % 500 == 0 or it == 1):
            print(f"    clone it {it:5d}  mse {loss:.5f}")
    pol = Policy(theta)
    A = np.array([pol.act(x) for x in X])[:, :n_out]
    rmse = np.sqrt(np.mean((A - Y) ** 2, axis=0))
    return theta, dict(rows=int(n), mse=loss, n_act=n_act,
                       rmse_steer=float(rmse[0]), rmse_pedal=float(rmse[1]),
                       rmse_wing=float(np.sqrt(np.mean(rmse[2:] ** 2))))


def clone_from_lap(path: str, tr, car=None, mu_scale: float = 1.0,
                   verbose: bool = False, n_act: int = N_ACT_FREE, **kw) -> tuple:
    """seed-lap file -> (theta, report). `tr` must be the lap's track."""
    seed = load_seed_lap(path)
    if seed.get("track") and getattr(tr, "name", None) and seed["track"] != tr.name:
        raise ValueError(f"{path} was driven on '{seed['track']}', not '{tr.name}'")
    obs, ytgt, _ = pairs(seed, tr, car=car, mu_scale=mu_scale, n_act=n_act)
    theta, rep = fit(obs, ytgt, verbose=verbose, **kw)
    rep.update(path=os.path.basename(path), lap_time=seed.get("lap_time"),
               track=seed.get("track"), car=seed.get("car"))
    return theta, rep


def seed_lap_trace(seed: dict) -> list:
    """The user's lap as the (t, x, y, psi, u, wing_deploy, wing_side)
    tuples `env.rollout(collect=)` produces, t re-zeroed at the lap start,
    so a viewer draws the user's ghost with the same code as a swarm car's."""
    cols, rows = seed["cols"], seed["rows"]
    ix = {c: i for i, c in enumerate(cols)}
    t0 = rows[0, ix["t"]] if len(rows) else 0.0
    out = []
    for r in rows:
        side = 0
        if r[ix["wing_deploy"]] > 0.05:
            side = 1 if r[ix["r"]] > 0 else -1
        out.append((r[ix["t"]] - t0, r[ix["x"]], r[ix["y"]], r[ix["psi"]],
                    r[ix["u"]], r[ix["wing_deploy"]], side))
    return out


def self_check(verbose: bool = True) -> bool:
    """A synthetic 'user' = the baseline driver with a known constant trim.
    The clone must recover that trim: rmse well under the trim's size."""
    from .. import track as trk
    from ..vehicle import Vehicle, VehicleConfig, Controls
    from corsa_c import CorsaC
    tr = trk.make_track("arena")
    car = CorsaC()
    trim = driver_trim(car, 1.0)
    veh = Vehicle(car, VehicleConfig(wing="plate"))
    x0, y0, psi0 = trk.start_pose(tr)
    veh.reset(x0, y0, psi0, 15.0, gear=2)
    rows = []
    dt = 0.002
    obs = np.empty(N_OBS)
    bias = np.array([0.15, -0.20, 0.0])    # the "user": anchor + a fixed trim
    for k in range(int(40.0 / dt)):
        observe(veh, tr, obs)
        a = np.clip(baseline_action(obs, trim["lock_rad"], trim["wheelbase"],
                                    trim["ay_plan"]) + RESID_GAIN * bias, -1, 1)
        ctl = Controls(delta=float(a[0]) * trim["lock_rad"],
                       throttle=max(float(a[1]), 0.0), brake=max(-float(a[1]), 0.0),
                       wing_on=bool(a[2] > 0.0), auto_gearbox=True, auto_clutch=True)
        if k % 5 == 0:
            rows.append([k * dt, veh.x, veh.y, veh.psi, veh.u, veh.v, veh.r,
                         veh.beta, veh.ay, veh.util_f, veh.util_r, veh.wing_deploy,
                         ctl.delta, ctl.throttle, ctl.brake, 1.0 if ctl.wing_on else 0.0])
        veh.step(ctl, (1.0,) * 4, (1.0,) * 4, dt)
    seed = dict(kind=SEED_LAP_KIND, cols=list(SEED_COLS), rows=np.array(rows),
                lock_rad=trim["lock_rad"], track="arena")
    ok = True
    for n_act in (N_ACT, N_ACT_FREE):
        o, y, _ = pairs(seed, tr, car, n_act=n_act)
        theta, rep = fit(o, y, steps=1500)
        good = (rep["rmse_steer"] < 0.06 and rep["rmse_pedal"] < 0.10
                and Policy(theta).n_act == n_act)
        ok = ok and good
        if verbose:
            print(f"  clone ({n_act} outputs): {rep['rows']} rows, rmse steer "
                  f"{rep['rmse_steer']:.3f} pedal {rep['rmse_pedal']:.3f} "
                  f"wing {rep['rmse_wing']:.3f}  [{'ok' if good else 'FAIL'}]")
    tr2 = seed_lap_trace(seed)
    ok = ok and len(tr2) == len(rows) and abs(tr2[0][0]) < 1e-12
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if self_check() else 1)
