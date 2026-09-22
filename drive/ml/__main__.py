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
    #  and it does NOT hold the centreline: a residual that asks for a line
    #  off the middle gets it, instead of being sprung back every tick. A
    #  net that outputs a constant steer trim of +0.15 must move the MEAN
    #  offset well left and still lap (the barrier fences the edge).
    import drive.ml.env as _env
    from .baseline import N_FREE
    ns: list = []
    _orig = _env.observe

    def _spy(veh, tr_, out=None, mu_here=None):
        o = _orig(veh, tr_, out, mu_here)
        ns.append(float(o[1]))
        return o

    class _Biased(Policy):
        def act(self, obs):
            return np.array([0.15, 0.0, 0.0, 0.0])

    _env.observe = _spy
    try:
        epb = rollout(_Biased(), "arena", T=60.0, wing="plate", tr=tr)
    finally:
        _env.observe = _orig
    n_mean = float(np.mean(ns))
    _rep("the anchor lets a residual hold a line off the centreline",
         n_mean > 0.5 * N_FREE and epb.ended == "time",
         f"steer trim +0.15 -> mean n_norm {n_mean:+.2f} (free band +-{N_FREE}), "
         f"ended '{epb.ended}'")

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
    #  The arena, not the skidpad. Through wave 4a the Corsa-tuned anchor
    #  SPUN the MX-5 at 485.0 m and the 540i at 487.9 m -- both inside
    #  WET_T3 (s 455..585, mu 0.632), which this driver's speed plan cannot
    #  see -- and the skidpad was the only cell all three could be checked on.
    #  `driver_trim`'s margin fade closed that, so the check can now assert
    #  the thing that actually matters: one hand-written driver, three very
    #  different cars, a lap each, nobody off the road.
    per_car = {k: rollout(p0, "arena", T=80.0, wing="plate",
                          car=cars.get(k)) for k in ("corsa", "mx5", "540i")}
    _rep("the anchor laps the arena in every car in the library",
         all(e.ended == "time" and e.laps >= 1 for e in per_car.values()),
         ", ".join(f"{k} {e.s_progress:.0f} m/{e.laps}L '{e.ended}'"
                   for k, e in per_car.items()))
    #  and it is genuinely a different car underneath, not the Corsa wearing
    #  three names: the library's grip calibration has to reach the physics
    #  (`env.rollout` passes `car.mu_scale`, which it did not until wave 4b)
    mus = {k: cars.get(k).mu_scale for k in ("corsa", "mx5", "540i")}
    vs = [e.v_mean for e in per_car.values()]
    _rep("each car drives as itself, with its own grip",
         len(set(round(v, 6) for v in vs)) == 3 and mus["mx5"] != 1.0,
         "v_mean " + ", ".join(f"{k} {e.v_mean:.3f}" for k, e in per_car.items())
         + "; mu_scale " + ", ".join(f"{k} {v:.2f}" for k, v in mus.items()))

    # --- the observation carries the surface, and the anchor uses it ------
    #  Wave 5's whole point. Through wave 4 the driver could not see the
    #  arena's WET_T3 (s 455..585, full width, mu 0.632) and a blanket
    #  MARGIN_FADE paid for that everywhere; now it plans around it. Two
    #  things have to hold: the profile has to FIND the patch, and the speed
    #  plan has to respond to it.
    from .env import MU_HORIZON, mu_ahead
    dry, wet = mu_ahead(tr, 100.0), mu_ahead(tr, 500.0)
    _rep("the observation finds the arena's wet patch",
         dry == 1.0 and wet < 0.7,
         f"mu_ahead({MU_HORIZON:.0f} m) = {dry:.4f} at s = 100 m (dry) and "
         f"{wet:.4f} at s = 500 m (inside WET_T3)")
    o_dry = obs[0].copy()
    #  R = 120 m at 28 m/s: fast enough that the DRY plan still wants
    #  throttle (v_corner 29.7 m/s) and the wet one does not (23.6), so the
    #  two are distinguishable. A tighter corner saturates both pedals at -1
    #  and the check would pass or fail for the wrong reason.
    o_dry[3] = o_dry[4] = 50.0 / 120.0
    o_dry[5] = o_dry[6] = o_dry[7] = 0.0
    o_dry[0] = 28.0 / 40.0
    o_dry[15] = o_dry[16] = 1.0
    o_wet = o_dry.copy()
    o_wet[15] = o_wet[16] = 0.632
    a_dry = baseline_action(o_dry, LOCK_RAD)
    a_wet = baseline_action(o_wet, LOCK_RAD)
    _rep("the speed plan slows for a wet corner",
         a_wet[1] < a_dry[1] - 0.05,
         f"pedal {a_dry[1]:+.3f} on mu 1.000 against {a_wet[1]:+.3f} on "
         f"mu 0.632, same R = 120 m corner at 28 m/s")
    #  and the 120 m station is what lets the 540i brake for open's corner
    o_far = obs[0].copy()
    o_far[0], o_far[3] = 41.0 / 40.0, 0.0
    o_far[4] = o_far[5] = o_far[6] = 0.0
    o_far[15] = o_far[16] = 1.0
    o_near = o_far.copy()
    o_far[7] = 50.0 / 45.0                     # R = 45 m, 120 m ahead
    _rep("the 120 m station reaches a corner the 70 m one cannot",
         baseline_action(o_far, LOCK_RAD)[1] < -0.5
         and baseline_action(o_near, LOCK_RAD)[1] > 0.0,
         f"at 41 m/s with an R = 45 m corner 120 m out the plan brakes "
         f"({baseline_action(o_far, LOCK_RAD)[1]:+.3f}); with the same road "
         f"empty it does not ({baseline_action(o_near, LOCK_RAD)[1]:+.3f})")

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

    # --- the free-wings head: 5 outputs, one per wing ---------------------
    from .policy import N_ACT, N_ACT_FREE, expand_base
    pf = Policy(np.zeros(Policy.n_param(N_ACT_FREE)))
    _rep("a 5-output genome is a free-wings policy", pf.free_wings and pf.n_act == N_ACT_FREE
         and not p0.free_wings, f"{pf.theta.size} parameters, {Policy.N_PARAM} for the 4-output head")
    cf = pf.controls(veh_obs, Controls)
    _rep("free wings command each panel", isinstance(cf.wing_cmd, tuple) and len(cf.wing_cmd) == 3
         and cf.wing_on == any(cf.wing_cmd) and p0.controls(veh_obs, Controls).wing_cmd is None,
         f"wing_cmd {cf.wing_cmd}; the 4-output head leaves it None")
    eb = expand_base(baseline_action(veh_obs, LOCK_RAD), veh_obs)
    _rep("the anchor spreads over three wings", eb.shape == (5,) and eb[0] == baseline_action(veh_obs, LOCK_RAD)[0])
    epf = rollout(pf, "arena", T=70.0, wing="plate", tr=tr)
    _rep("theta = 0 with free wings still laps the arena", epf.ended == "time" and epf.laps >= 1,
         f"reward {epf.reward:.0f}, laps {epf.laps}, ended {epf.ended} (one-wing head: {ep.reward:.0f})")
    wide = Policy(Policy.widen(pr.theta))
    acts_old = np.array([pr.act(o) for o in obs])
    acts_new = np.array([wide.act(o) for o in obs])
    #  the wing rows of the widened genome are ZERO (the 4-output head's wing
    #  row never had authority, see policy.WING_PRIOR), so the composed wing
    #  verdict is the anchor's prior -- exactly what the 4-output genome did
    same_wing = all(pr.controls(o, Controls).wing_on == wide.controls(o, Controls).wing_on
                    for o in obs)
    _rep("widening a 4-output genome keeps its steer, pedal and wing verdicts",
         wide.free_wings and np.allclose(acts_new[:, :2], acts_old[:, :2])
         and not np.any(acts_new[:, 2:]) and same_wing,
         f"wing_on agrees on {sum(1 for o in obs)} observations, wing rows zero")

    # --- the free head has REAL authority over all three wings -----------
    #  Through task 17 it did not: the anchor's +-1 verdicts at RESID_GAIN
    #  0.55 could never be crossed, so 'free wings' was a name. Now the
    #  verdict is a +-WING_PRIOR prior at WING_GAIN 1.0. Forced +1 on every
    #  wing output must put ALL THREE out for the whole run (both flanks =
    #  the air brake, and the top wing on a car that has one); forced -1
    #  must keep every one in; and the 4-output head must be UNMOVED by
    #  either, to the bit.
    from .policy import WING_PRIOR, WING_GAIN
    from ..vehicle import TopAero

    class _Forced(Policy):
        def __init__(self, n_act, val):
            super().__init__(np.zeros(Policy.n_param(n_act)))
            self.val = val

        def act(self, o):
            a = np.zeros(self.n_act)
            a[2:] = self.val
            return a

    top_kw = dict(top=TopAero(mode="active"))
    ep_all = rollout(_Forced(N_ACT_FREE, +1.0), "arena", T=40.0, wing="plate", tr=tr,
                     cfg_kwargs=top_kw)
    ep_none = rollout(_Forced(N_ACT_FREE, -1.0), "arena", T=40.0, wing="plate", tr=tr,
                      cfg_kwargs=top_kw)
    _rep("the free head can put all three wings out where the anchor would not",
         ep_all.wing_l_frac > 0.97 and ep_all.wing_r_frac > 0.97
         and ep_all.wing_both_frac > 0.97 and ep_all.wing_top_frac > 0.97,
         f"forced +1: L {100 * ep_all.wing_l_frac:.1f} % R {100 * ep_all.wing_r_frac:.1f} % "
         f"top {100 * ep_all.wing_top_frac:.1f} % both {100 * ep_all.wing_both_frac:.1f} % "
         f"(prior +-{WING_PRIOR} at gain {WING_GAIN})")
    _rep("... and keep every wing in where the anchor would run one",
         ep_none.wing_frac == 0.0 and ep_none.wing_top_frac == 0.0
         and ep_none.wing_l_frac == 0.0 and ep_none.wing_r_frac == 0.0,
         f"forced -1: flank {100 * ep_none.wing_frac:.1f} % top {100 * ep_none.wing_top_frac:.1f} %")
    ep4_0 = rollout(Policy(), "arena", T=40.0, wing="plate", tr=tr, cfg_kwargs=top_kw)
    ep4_p = rollout(_Forced(N_ACT, +1.0), "arena", T=40.0, wing="plate", tr=tr, cfg_kwargs=top_kw)
    ep4_m = rollout(_Forced(N_ACT, -1.0), "arena", T=40.0, wing="plate", tr=tr, cfg_kwargs=top_kw)
    _rep("the 4-output head is unmoved: its wing is still the anchor's, to the bit",
         ep4_0.reward == ep4_p.reward == ep4_m.reward
         and ep4_0.wing_frac == ep4_p.wing_frac == ep4_m.wing_frac and ep4_0.wing_frac > 0.3,
         f"reward {ep4_0.reward:.6f} for net wing -1 / 0 / +1, wing {100 * ep4_0.wing_frac:.1f} %")

    # --- the swarm: the seed lap's schema, the clone, the GA -------------
    from . import clone as _cl, swarm as _sw
    import drive.drive as _dd
    _rep("the seed lap the sim writes is the one the clone reads",
         tuple(_dd.SEED_LAP_COLS) == tuple(_cl.SEED_COLS)
         and _dd.SEED_LAP_KIND == _cl.SEED_LAP_KIND,
         f"{len(_cl.SEED_COLS)} columns, kind {_cl.SEED_LAP_KIND}")
    _rep("the clone recovers a known trim from a lap", _cl.self_check(verbose=False),
         "anchor + fixed trim, 40 s at 100 Hz -> rmse steer < 0.06, pedal < 0.10")
    _rep("the swarm: elites, lineage, determinism, state, checkpoint",
         _sw.self_check(verbose=False), "pop 6, 6 s rollouts, one process")

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
