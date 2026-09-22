"""Train the driving policy with an evolution strategy. numpy only.

    python3 -m drive.ml.train --iters 60 --pop 24 --track arena

The algorithm is the plain mirrored-sampling ES (Salimans et al. 2017,
"Evolution Strategies as a Scalable Alternative to Reinforcement Learning"),
which is four lines of numpy:

    eps ~ N(0, I)                        half a population, mirrored
    F   = rank_centre(reward(theta +- sigma*eps))
    theta <- theta + lr/(pop*sigma) * sum_i F_i * eps_i

Mirrored sampling because it removes the first-order term of the gradient
estimate's variance for free, and centred RANKS rather than raw rewards
because our reward has a -60 cliff on going off the track: with raw rewards
one crashed candidate dominates the update and the population collapses onto
"drive slowly".

The fitness is a DETERMINISTIC rollout (`env.rollout` is pure in its
arguments), so there is no evaluation noise to average over and one rollout
per candidate is the right budget. That is unusual for an ES and it is a
property of this problem, not an oversight.

`--car mx5` trains against that car's engine, brakes, mass, driven axle, lock
and wheelbase (the Corsa is the default and `--car corsa` is bit-for-bit the
no-car path). `--track arena,open` trains ONE policy on SEVERAL circuits: the
fitness is then the mean over the tracks of that track's reward divided by the
hand-written baseline's reward on the SAME track. Normalising matters. Raw
metres would let the faster circuit own the objective -- open advances more
metres in 70 s than arena does -- and the question being asked is "is it a
better driver on both", not "where can it cover the most ground". With a
single track the normaliser is exactly 1.0 and this path is bit-for-bit the
old one; nothing about the single-track runs already committed moved.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from .env import rollout, lap_time, DT_TRAIN, DT_EVAL
from .policy import Policy, N_ACT, N_ACT_FREE

#: built once per worker process; `make_track` is ~15 ms and a rollout is ~1 s,
#: so rebuilding it per candidate would be 1.5 % of the whole run for nothing
_TRACK_CACHE: dict = {}
_CFG: dict = {}


def _track(name):
    if name not in _TRACK_CACHE:
        from .. import track as trk
        _TRACK_CACHE[name] = trk.make_track(name)
    return _TRACK_CACHE[name]


def _car(name):
    """The named `cars.CarSpec`, or None for the stock Corsa.

    A NAME crosses the pickle boundary, not a `CarSpec`: the dataclass carries
    provenance strings and a `PointMass` list, and `cars.get` is a dict lookup,
    so there is nothing to gain by shipping the object to 12 workers 2880
    times. None (not `cars.get('corsa')`) is the Corsa so that a run with no
    `--car` is bit-for-bit the old code path -- `rollout` builds `CorsaC()`,
    which `cars.self_check` asserts is field-for-field `CORSA_C` anyway.
    """
    if name in (None, "", "corsa"):
        return None
    import cars
    return cars.get(name)


def _tracks(spec) -> list:
    """'arena' -> ['arena'];  'arena,open' or 'arena+open' -> both, in order."""
    if isinstance(spec, (list, tuple)):
        return [str(s) for s in spec]
    return [s for s in str(spec).replace("+", ",").split(",") if s.strip()]


def _init_worker(cfg):
    _CFG.clear()
    _CFG.update(cfg)


def _score(theta) -> tuple:
    """One candidate's fitness. Runs in a pool worker; returns plain floats
    only, so nothing large crosses the pickle boundary.

    With one track this is one rollout and `norm` is 1.0, so the returned
    fitness IS `ep.reward` to the last bit. With several it is the mean of
    reward/baseline_reward per track -- see the module docstring on why the
    normaliser is there -- and the reported diagnostics are the mean over
    tracks, except `laps` (summed) and `ended` (every track's, in order, so a
    policy that only falls off ONE of them is visible in the log).
    """
    pol = Policy(theta)
    car = _car(_CFG.get("car"))
    eps = [rollout(pol, tk, dt=_CFG["dt"], T=_CFG["T"], wing=_CFG["wing"],
                   tr=_track(tk), car=car) for tk in _CFG["tracks"]]
    nrm = _CFG["norm"]
    k = float(len(eps))
    fit = sum(e.reward / w for e, w in zip(eps, nrm)) / k
    return (fit, sum(e.s_progress for e in eps) / k, min(e.t for e in eps),
            sum(e.v_mean for e in eps) / k, sum(e.wing_frac for e in eps) / k,
            sum(e.wing_outer_frac for e in eps) / k,
            sum(e.laps for e in eps), "/".join(e.ended for e in eps))


def _save_atomic(theta, path: str, meta: dict) -> None:
    """Write to a sibling temp file and rename: a kill mid-write must not be
    able to leave a half-written checkpoint where a good one used to be."""
    tmp = f"{path}.part"
    Policy(np.asarray(theta, float).copy(), meta).save(tmp)
    os.replace(tmp, path)


def _rank_centre(x) -> np.ndarray:
    """Rewards -> centred ranks in [-0.5, +0.5]. Scale-free and outlier-proof,
    which is the point: see the module docstring on the -60 off-track cliff."""
    n = len(x)
    order = np.argsort(np.argsort(np.asarray(x, float)))
    return order / max(n - 1, 1) - 0.5


def train(track: str = "arena", wing: str = "plate", iters: int = 60,
          pop: int = 24, sigma: float = 0.10, lr: float = 0.06,
          seed: int = 0, T: float = 60.0, workers: int | None = None,
          dt: float = DT_TRAIN, out: str | None = None,
          init: str | None = None, verbose: bool = True,
          save_every: int = 1, car: str | None = None,
          free_wings: bool = False) -> dict:
    """`free_wings` breeds the 5-output head (one output per wing, see
    `policy.WING_PRIOR`) instead of the published 4-output one; a 4-output
    `init` is widened. Default False, so every committed ES run is
    reproducible as it was."""
    if pop % 2:
        pop += 1                      # mirrored sampling needs pairs
    rng = np.random.default_rng(seed)
    n_act = N_ACT_FREE if free_wings else N_ACT
    base = Policy.load(init) if init else Policy(np.zeros(Policy.n_param(n_act)))
    if free_wings and not base.free_wings:
        base = Policy(Policy.widen(base.theta), base.meta)
    theta = base.theta.copy()
    tracks = _tracks(track)
    #  One track -> normaliser exactly 1.0, so `_score` returns `ep.reward`
    #  unchanged and the arena runs already committed are reproducible bit for
    #  bit. Several -> the baseline's own reward on each, measured here once
    #  (2 rollouts, ~4 s) rather than 2880 times inside the fitness.
    if len(tracks) == 1:
        norm = [1.0]
    else:
        norm = [float(rollout(Policy(), tk, dt=dt, T=T, wing=wing,
                              car=_car(car)).reward)
                for tk in tracks]
        if verbose:
            print("  baseline reward per track (the fitness normaliser): "
                  + ", ".join(f"{tk} {w:.1f} m" for tk, w in zip(tracks, norm)))
    cfg = dict(track=tracks[0], tracks=tracks, norm=norm, wing=wing, dt=dt,
               T=T, car=car)
    workers = min(os.cpu_count() or 1, pop) if workers is None else workers

    #  the multi-track fitness is a RATIO near 1, the single-track one is
    #  metres near 1400: one print format cannot read well for both
    ffmt = "8.4f" if len(tracks) > 1 else "8.1f"
    curve, t0, best = [], time.perf_counter(), (-1e18, theta.copy())
    pool = (mp.Pool(workers, initializer=_init_worker, initargs=(cfg,))
            if workers > 1 else None)
    if pool is None:
        _init_worker(cfg)
    try:
        for it in range(iters):
            eps = rng.normal(0.0, 1.0, (pop // 2, theta.size))
            batch = np.concatenate([theta + sigma * eps, theta - sigma * eps])
            res = (pool.map(_score, list(batch), chunksize=1) if pool
                   else [_score(t) for t in batch])
            R = np.array([r[0] for r in res])
            F = _rank_centre(R)
            grad = (F[:pop // 2] - F[pop // 2:]) @ eps
            theta = theta + (lr / (pop * sigma)) * grad

            # the mean policy, scored on the same deterministic rollout: this
            # is the learning curve, not the population's best sample
            m = _score(theta) if pool is None else pool.apply(_score, (theta,))
            if m[0] > best[0]:
                best = (m[0], theta.copy())
            curve.append(dict(it=it, mean=float(m[0]), pop_best=float(R.max()),
                              pop_med=float(np.median(R)), s=float(m[1]),
                              t=float(m[2]), v_mean=float(m[3]),
                              wing_frac=float(m[4]), wing_outer=float(m[5]),
                              laps=int(m[6]), ended=m[7]))
            #  Checkpoint EVERY iteration, atomically. The first long run of
            #  this trainer was killed at iteration 111 of 150 and lost 27
            #  minutes of compute because the save was at the end; a run that
            #  cannot be interrupted is a run that has to be repeated.
            if out and save_every and (it % save_every == 0):
                _save_atomic(theta if m[0] >= best[0] else best[1], out, dict(
                    track=",".join(tracks), tracks=tracks, norm=norm,
                    car=car or "corsa",
                    wing=wing, iters=iters, pop=pop, sigma=sigma,
                    lr=lr, seed=seed, T=T, dt_train=dt, done=it + 1,
                    secs=round(time.perf_counter() - t0, 1),
                    reward=round(best[0], 2), curve=curve))
            if verbose:
                print(f"  it {it:3d}  mean {m[0]:{ffmt}}  pop best {R.max():{ffmt}}  "
                      f"med {np.median(R):{ffmt}}  s {m[1]:7.1f} m  v {m[3]:5.2f}  "
                      f"wing {m[4]:4.2f} (outer {m[5]:4.2f})  laps {m[6]}  {m[7]}"
                      f"  [{time.perf_counter() - t0:5.1f} s]", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    secs = time.perf_counter() - t0
    pol = Policy(best[1], meta=dict(
        track=",".join(tracks), tracks=tracks, norm=norm,
        car=car or "corsa",
        wing=wing, iters=iters, pop=pop, sigma=sigma, lr=lr,
        seed=seed, T=T, dt_train=dt, secs=round(secs, 1),
        reward=round(best[0], 2), curve=curve))
    if out:
        pol.save(out)
        if verbose:
            print(f"  saved {out}  ({secs:.1f} s, {iters * pop} rollouts)")
    return dict(policy=pol, curve=curve, secs=secs, reward=best[0], path=out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="train a drive.ml policy (ES, numpy only)")
    ap.add_argument("--track", default="arena",
                    help="one track, or several comma-separated ('arena,open') "
                         "to train ONE policy on all of them")
    ap.add_argument("--wing", default="plate", choices=("off", "fin", "plate"))
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--sigma", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=60.0, help="s of sim per rollout")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--dt", type=float, default=DT_TRAIN)
    ap.add_argument("--init", default=None, help="warm-start from a checkpoint")
    ap.add_argument("--out", default="drive/ml/checkpoints/policy.json")
    ap.add_argument("--save-every", type=int, default=1,
                    help="iterations between checkpoints (0 = only at the end)")
    ap.add_argument("--car", default=None,
                    help="cars.py key: corsa (default) | mx5 | 540i. The "
                         "policy trims THAT car's own lock and wheelbase.")
    ap.add_argument("--eval", action="store_true",
                    help="after training, re-measure at DT_EVAL and print lap times")
    ap.add_argument("--free-wings", action="store_true",
                    help="breed the 5-output head: one output per wing (left flank, "
                         "right flank, top), each free to deploy on its own")
    a = ap.parse_args(argv)
    tks = _tracks(a.track)
    n_par = Policy.n_param(N_ACT_FREE if a.free_wings else N_ACT)
    print(f"ES  {n_par} params{' (free wings)' if a.free_wings else ''}  "
          f"pop {a.pop}  sigma {a.sigma}  lr {a.lr}  "
          f"{a.iters} iters  dt {a.dt * 1e3:.0f} ms  T {a.duration:.0f} s  "
          f"workers {a.workers or os.cpu_count()}  "
          f"car {a.car or 'corsa'}  "
          f"track{'s' if len(tks) > 1 else ''} {'+'.join(tks)}"
          f"{f'  ({len(tks)} rollouts per candidate)' if len(tks) > 1 else ''}")
    r = train(a.track, a.wing, a.iters, a.pop, a.sigma, a.lr, a.seed,
              a.duration, a.workers, a.dt, a.out, a.init,
              save_every=a.save_every, car=a.car, free_wings=a.free_wings)
    if a.eval:
        print("\n  re-measured at DT_EVAL = 1 ms:")
        for tk in tks:
            print(f"   {tk}: ", json.dumps(lap_time(r["policy"], tk, wing=a.wing,
                                                    dt=DT_EVAL), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
