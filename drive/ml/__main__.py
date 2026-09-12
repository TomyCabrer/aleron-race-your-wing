"""`python3 -m drive.ml` — the package self-check. Fast and deterministic.

What is under test is not that the policy is fast — that is what
`drive.ml.evaluate` measures on a checkpoint — but that the package is honest:
that `theta = 0` is exactly the hand-written baseline, that a rollout is a pure
function of its arguments, that a checkpoint round-trips, and above all that
importing and running any of this changes nothing about the sim.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from .baseline import K70_PLAN, baseline_action
from .env import DT_EVAL, DT_TRAIN, N_OBS, observe, rollout
from .policy import LOCK_RAD, RESID_GAIN, Policy

_res = []


def _rep(tag, passed, msg=""):
    _res.append((tag, bool(passed), msg))
    print(f"  [{'ok' if passed else 'FAIL'}] {tag}: {msg}")


def self_check(verbose: bool = True) -> bool:
    import drive.track as trk

    # --- the residual is anchored on the baseline ------------------------
    rng = np.random.default_rng(7)
    obs = rng.normal(0.0, 0.4, (64, N_OBS))
    p0 = Policy()                                # theta = 0
    worst = max(float(np.max(np.abs(p0.action(o) - baseline_action(o, LOCK_RAD))))
                for o in obs)
    _rep("theta = 0 IS the hand-written baseline", worst == 0.0,
         f"max |policy - baseline| over 64 random observations = {worst:.1e}")
    pr = Policy.random(1, 0.6)
    moved = max(float(np.max(np.abs(pr.action(o) - baseline_action(o, LOCK_RAD))))
                for o in obs)
    _rep("a trained policy actually moves the action", moved > 0.05,
         f"max |policy - baseline| = {moved:.3f} (residual gain {RESID_GAIN})")

    # --- the action can never exceed the actuators -----------------------
    big = Policy(np.full(Policy.N_PARAM, 8.0))
    acts = np.array([big.action(o) for o in obs])
    _rep("the composed action is clipped to the actuators",
         float(np.max(np.abs(acts))) <= 1.0,
         f"max |action| = {float(np.max(np.abs(acts))):.3f} over a saturated net")
    #  one right foot: throttle and brake can never both be commanded
    from ..vehicle import Controls
    both = 0
    for o in obs:
        c = big.controls(o, Controls)
        both += int(c.throttle > 0.0 and c.brake > 0.0)
    _rep("throttle and brake are never both commanded", both == 0,
         "one pedal axis, so it is structural, not learned")

    # --- a rollout is a pure function of its arguments -------------------
    tr = trk.make_track("arena")
    a = rollout(p0, "arena", T=8.0, tr=tr)
    b = rollout(p0, "arena", T=8.0, tr=tr)
    _rep("a rollout is bit-reproducible", a.reward == b.reward and a.t == b.t,
         f"reward {a.reward:.12f} twice")
    #  and the observation reads only what the contract publishes
    veh_obs = observe(_mk_vehicle(tr), tr)
    _rep("the observation is finite and the right shape",
         veh_obs.shape == (N_OBS,) and bool(np.all(np.isfinite(veh_obs))),
         f"{N_OBS} entries, all finite at the start pose")

    # --- the baseline is a competent driver ------------------------------
    ep = rollout(p0, "arena", T=70.0, wing="plate", tr=tr)
    _rep("the baseline laps the arena without leaving it",
         ep.ended == "time" and ep.laps >= 1,
         f"{ep.s_progress:.0f} m in {ep.t:.0f} s, {ep.laps} lap(s), "
         f"v_mean {ep.v_mean:.2f} m/s, ended '{ep.ended}'")
    _rep("the baseline deploys the panel on the OUTER flank",
         ep.wing_frac > 0.2 and ep.wing_outer_frac > 0.90,
         f"deployed {100 * ep.wing_frac:.0f} % of steps, "
         f"{100 * ep.wing_outer_frac:.0f} % of those on the outer flank")

    # --- the corrected anchor can drive the OPEN map ----------------------
    #  This is the whole of wave 4 item 1 in one assertion. Through wave 3
    #  `baseline_action` planned its speed off the 0 / 15 / 35 m curvature
    #  lookaheads and never `obs[6]`, the 70 m one, so on `open` it held full
    #  throttle to s = 385 m, braked for 35 m, needed 44.5, and was off the
    #  road at s = 451.5 m on every single run. `K70_PLAN` closed that, and
    #  the value is a swept one with a cliff 0.10 either side of it (see
    #  `baseline.py`), so it is exactly the kind of constant that wants a
    #  check rather than a comment. 45 s is enough to be 600 m in: the corner
    #  that used to end the episode is at s ~ 420 m.
    op = rollout(p0, "open", T=45.0, wing="plate", dt=DT_TRAIN)
    _rep("the corrected anchor brakes for open's R = 45 m corner",
         op.ended == "time" and op.s_progress > 600.0,
         f"{op.s_progress:.0f} m in {op.t:.0f} s, ended '{op.ended}' "
         f"(K70_PLAN = {K70_PLAN}; at 0.0 it was off the road at 451.5 m)")

    # --- three cars, and the anchor takes each one's own numbers ----------
    #  Wave 4 pointed this package at the whole library. Two things have to
    #  hold or a cross-car number is meaningless: the anchor's pure-pursuit
    #  feedforward must use the CAR's wheelbase (a fixed 2.491 m would give
    #  the 540i 14 % too much and the MX-5 10 % too little), and the Corsa
    #  must be bit-for-bit unmoved by that change, because every committed
    #  checkpoint's quoted lap time is a Corsa number.
    import cars
    from .baseline import WHEELBASE
    o_turn = obs[0].copy()
    o_turn[3] = o_turn[4] = o_turn[5] = 50.0 / 40.0      # a real R = 40 m bend
    a_corsa = baseline_action(o_turn, LOCK_RAD, cars.get("corsa").L)
    a_dflt = baseline_action(o_turn, LOCK_RAD)
    a_mx5 = baseline_action(o_turn, LOCK_RAD, cars.get("mx5").L)
    _rep("the anchor's feedforward follows the car's wheelbase",
         float(np.max(np.abs(a_corsa - a_dflt))) == 0.0
         and abs(a_mx5[0] - a_corsa[0]) > 1e-4,
         f"corsa == default to 0.0e+00 (L = {WHEELBASE:.3f} m); "
         f"mx5 steer differs by {a_mx5[0] - a_corsa[0]:+.5f} of lock "
         f"(L = {cars.get('mx5').L:.3f} m)")
    #  the skidpad is the ONE cell the Corsa-tuned anchor drives on all three
    #  cars: it spins the MX-5 and the 540i on the arena at 485 / 488 m and
    #  cannot brake the 540i for open's R = 45 m corner. That is a measured
    #  wave-4 finding, not a bug, so the check asserts only what holds.
    per_car = {k: rollout(p0, "skidpad", T=25.0, wing="plate",
                          car=cars.get(k)) for k in ("corsa", "mx5", "540i")}
    vs = [e.v_mean for e in per_car.values()]
    _rep("a rollout drives every car in the library",
         all(e.ended == "time" for e in per_car.values())
         and len(set(round(v, 6) for v in vs)) == 3,
         "skidpad, 25 s: " + ", ".join(
             f"{k} {e.v_mean:.3f} m/s" for k, e in per_car.items()))

    # --- the training timestep is the one the docstring claims -----------
    e1 = rollout(p0, "arena", T=20.0, dt=DT_EVAL, tr=tr)
    e2 = rollout(p0, "arena", T=20.0, dt=DT_TRAIN, tr=tr)
    _rep("dt 2 ms tracks dt 1 ms closely enough to train on",
         abs(e1.s_progress - e2.s_progress) < 0.02 * abs(e1.s_progress),
         f"{e1.s_progress:.2f} m at 1 ms vs {e2.s_progress:.2f} m at 2 ms "
         f"({100 * abs(e1.s_progress - e2.s_progress) / abs(e1.s_progress):.2f} %)")

    # --- checkpoints round-trip ------------------------------------------
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "p.json")
        pr.meta["note"] = "round-trip"
        pr.save(f)
        back = Policy.load(f)
        _rep("checkpoint round-trips exactly",
             np.array_equal(back.theta, pr.theta) and back.meta["note"] == "round-trip"
             and back.residual == pr.residual, f"{Policy.N_PARAM} parameters")

    # --- the multi-track fitness ------------------------------------------
    #  `--track arena,open` is the only thing in the trainer that can change
    #  what a SINGLE-track run scores, and a single-track run is what every
    #  committed checkpoint was made with. So assert both halves: that one
    #  track still returns `ep.reward` to the last bit, and that several
    #  return exactly the mean of reward/normaliser.
    from . import train as _tr
    _rep("the track list parses", _tr._tracks("arena") == ["arena"]
         and _tr._tracks("arena,open") == ["arena", "open"]
         and _tr._tracks("arena+skidpad") == ["arena", "skidpad"],
         "'arena' -> 1, 'arena,open' -> 2, 'arena+skidpad' -> 2")
    th = pr.theta
    _tr._init_worker(dict(track="arena", tracks=["arena"], norm=[1.0],
                          wing="plate", dt=DT_TRAIN, T=6.0))
    one = _tr._score(th)[0]
    ref = rollout(pr, "arena", dt=DT_TRAIN, T=6.0, wing="plate").reward
    _tr._init_worker(dict(track="arena", tracks=["arena", "skidpad"],
                          norm=[100.0, 200.0], wing="plate", dt=DT_TRAIN, T=6.0))
    two = _tr._score(th)[0]
    r_sk = rollout(pr, "skidpad", dt=DT_TRAIN, T=6.0, wing="plate").reward
    want = 0.5 * (ref / 100.0 + r_sk / 200.0)
    _rep("one track scores exactly ep.reward, several the normalised mean",
         one == ref and two == want,
         f"1 track {one:.9f} == {ref:.9f}; 2 tracks {two:.9f} == {want:.9f}")

    # --- additive: nothing in drive/ imports drive.ml --------------------
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    r = subprocess.run(
        [sys.executable, "-c",
         "import drive.drive, drive.vehicle, drive.render, drive.input, sys;"
         "print(any(m.startswith('drive.ml') for m in sys.modules))"],
        capture_output=True, text=True, cwd=root,
        env=dict(os.environ, CARSIM_HEADLESS="1", SDL_VIDEODRIVER="dummy",
                 SDL_AUDIODRIVER="dummy"))
    _rep("no module in drive/ imports drive.ml",
         r.returncode == 0 and r.stdout.strip().endswith("False"),
         (r.stdout.strip().splitlines() or ["<no output>"])[-1]
         + (f"  stderr: {r.stderr.strip()[-200:]}" if r.returncode else ""))

    ok = all(p for _t, p, _m in _res)
    if verbose:
        print(f"  {'ALL PASS' if ok else 'FAILURES ABOVE'}: "
              f"{sum(1 for _t, p, _m in _res if p)}/{len(_res)} checks")
    return ok


def _mk_vehicle(tr):
    from corsa_c import CorsaC
    from ..vehicle import Vehicle, VehicleConfig
    import drive.track as trk
    v = Vehicle(CorsaC(), VehicleConfig(wing="plate"))
    x, y, psi = trk.start_pose(tr)
    v.reset(x, y, psi, 12.0, gear=2)
    return v


if __name__ == "__main__":
    raise SystemExit(0 if self_check() else 1)
