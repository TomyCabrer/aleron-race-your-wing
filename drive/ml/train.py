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
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from .env import rollout, lap_time, DT_TRAIN, DT_EVAL
from .policy import Policy

#: built once per worker process; `make_track` is ~15 ms and a rollout is ~1 s,
#: so rebuilding it per candidate would be 1.5 % of the whole run for nothing
_TRACK_CACHE: dict = {}
_CFG: dict = {}


def _track(name):
    if name not in _TRACK_CACHE:
        from .. import track as trk
        _TRACK_CACHE[name] = trk.make_track(name)
    return _TRACK_CACHE[name]


def _init_worker(cfg):
    _CFG.clear()
    _CFG.update(cfg)


def _score(theta) -> tuple:
    """One candidate's fitness. Runs in a pool worker; returns plain floats
    only, so nothing large crosses the pickle boundary."""
    ep = rollout(Policy(theta), _CFG["track"], dt=_CFG["dt"], T=_CFG["T"],
                 wing=_CFG["wing"], tr=_track(_CFG["track"]))
    return (ep.reward, ep.s_progress, ep.t, ep.v_mean, ep.wing_frac,
            ep.wing_outer_frac, ep.laps, ep.ended)


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
          save_every: int = 1) -> dict:
    if pop % 2:
        pop += 1                      # mirrored sampling needs pairs
    rng = np.random.default_rng(seed)
    base = Policy.load(init) if init else Policy(np.zeros(Policy.N_PARAM))
    theta = base.theta.copy()
    cfg = dict(track=track, wing=wing, dt=dt, T=T)
    workers = min(os.cpu_count() or 1, pop) if workers is None else workers

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
                    track=track, wing=wing, iters=iters, pop=pop, sigma=sigma,
                    lr=lr, seed=seed, T=T, dt_train=dt, done=it + 1,
                    secs=round(time.perf_counter() - t0, 1),
                    reward=round(best[0], 2), curve=curve))
            if verbose:
                print(f"  it {it:3d}  mean {m[0]:8.1f}  pop best {R.max():8.1f}  "
                      f"med {np.median(R):8.1f}  s {m[1]:7.1f} m  v {m[3]:5.2f}  "
                      f"wing {m[4]:4.2f} (outer {m[5]:4.2f})  laps {m[6]}  {m[7]}"
                      f"  [{time.perf_counter() - t0:5.1f} s]", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    secs = time.perf_counter() - t0
    pol = Policy(best[1], meta=dict(
        track=track, wing=wing, iters=iters, pop=pop, sigma=sigma, lr=lr,
        seed=seed, T=T, dt_train=dt, secs=round(secs, 1),
        reward=round(best[0], 2), curve=curve))
    if out:
        pol.save(out)
        if verbose:
            print(f"  saved {out}  ({secs:.1f} s, {iters * pop} rollouts)")
    return dict(policy=pol, curve=curve, secs=secs, reward=best[0], path=out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="train a drive.ml policy (ES, numpy only)")
    ap.add_argument("--track", default="arena")
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
    ap.add_argument("--eval", action="store_true",
                    help="after training, re-measure at DT_EVAL and print lap times")
    a = ap.parse_args(argv)
    print(f"ES  {Policy.N_PARAM} params  pop {a.pop}  sigma {a.sigma}  lr {a.lr}  "
          f"{a.iters} iters  dt {a.dt * 1e3:.0f} ms  T {a.duration:.0f} s  "
          f"workers {a.workers or os.cpu_count()}")
    r = train(a.track, a.wing, a.iters, a.pop, a.sigma, a.lr, a.seed,
              a.duration, a.workers, a.dt, a.out, a.init,
              save_every=a.save_every)
    if a.eval:
        print("\n  re-measured at DT_EVAL = 1 ms:")
        print("  ", json.dumps(lap_time(r["policy"], a.track, wing=a.wing,
                                        dt=DT_EVAL), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
