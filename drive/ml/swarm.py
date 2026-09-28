"""The swarm: a GENETIC ALGORITHM over `Policy` genomes, the best reproduce.

    python3 -m drive.ml.swarm --pop 32 --gens 20 --track arena
    python3 -m drive.ml.swarm --seed runs/swarm/seed_arena_*.json   # from a lap the user drove
    python3 -m drive.ml.swarm --seed drive/ml/checkpoints/arena_open_plate.json
    python3 -m drive.ml.swarm --resume runs/swarm/<name>_state.json
    python3 -m drive.drive --swarm 32 [--swarm-seed ...]           # the same, with a window

`train.py` is an evolution STRATEGY: one mean genome, a cloud of mirrored
perturbations, a rank-weighted step. This is the other classic and it is what
the brief asks for: a POPULATION of N distinct cars, every generation the
best ones are kept and the rest are bred from them by crossover and mutation.
It is less sample-efficient than the ES on a smooth objective and much more
robust on this one, whose fitness has a -60 cliff (off track) and a flat
plateau (stopped car): a population keeps several basins alive at once and
an elite is never lost to one bad step.

Fitness is `env.rollout`'s reward (metres of centreline, minus the cliffs),
and ties break on the best flying lap -- "the existing reward, then lap
time". Every individual is a deterministic rollout, so no re-evaluation.

The genome is `Policy.theta` (373 floats with the free-wings head, 356
with `--legacy-wings`); the residual anchor means
`theta = 0` is already a competent driver, so a population seeded from
nothing starts from that, not from noise. Seeds:

  * `none`           -- generation 0 is the anchor plus mutants of it
  * a seed-lap file  -- `clone.clone_from_lap` fits the user's lap first
  * a checkpoint     -- a `Policy` saved by `train.py` or by a previous swarm

and in all three cases individual 0 of generation 0 is the SEED ITSELF,
unmutated, so the swarm can never do worse than what it was given.

State: `runs/swarm/<name>_state.json` (the whole population, every
generation, resumable). Deliverable: `Swarm.save_best()` writes a plain
`Policy` checkpoint that `--ml-drive` drives and that the next swarm can
`--seed` from, with the lap RE-MEASURED at DT_EVAL like every quoted number
in this package.

No pygame here; the viewer lives in `drive.drive` behind `--swarm`.
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

STATE_KIND = "carsim-swarm-state-1"
STATE_DIR = os.path.join("runs", "swarm")
CKPT_DIR = os.path.join("drive", "ml", "checkpoints")

#: GA defaults. `ELITE_FRAC` of the population survives untouched; parents
#: are drawn by tournament of `TOURNAMENT` from the whole ranked population;
#: BLX-alpha crossover (`BLX`), then gaussian mutation with a sigma that is
#: adapted by the 1/5 rule (`SIGMA_UP` / `SIGMA_DOWN`). `SIGMA0` is the
#: initial spread of generation 0 around the seed and the ES's own sigma.
ELITE_FRAC = 0.15
TOURNAMENT = 3
BLX = 0.25
SIGMA0 = 0.08
SIGMA_MIN, SIGMA_MAX = 0.02, 0.40
SIGMA_UP, SIGMA_DOWN = 1.15, 0.85
P_CROSS = 0.8
#: DIVERSITY. Every genome carries its OWN mutation step (`ind["sigma"]`,
#: classic self-adaptive ES): a child inherits its parents' mean step
#: times a log-normal of `SIGMA_TAU`, then mutates by it -- so a lineage
#: that profits from big moves keeps making them while another fine-tunes,
#: instead of one population-wide sigma collapsing on the anchor's line
#: (which is what the first swarms did: 24 near-identical cars on the
#: centreline). Generation 0 is spread wide on purpose: its steps are
#: log-uniform in `SIGMA0 * [GEN0_LO, GEN0_HI]`. `Swarm.sigma` is the
#: population's MEDIAN step, kept for the history and the 1/5 rule's nudge.
SIGMA_TAU = 0.25
GEN0_LO, GEN0_HI = 0.5, 4.0

_CFG: dict = {}
_TRACK: dict = {}


def _init_worker(cfg):
    _CFG.clear()
    _CFG.update(cfg)


def _track():
    name = _CFG["track"]
    if name not in _TRACK:
        from .. import track as trk
        _TRACK[name] = trk.make_track(name, **_CFG.get("track_kw", {}))
    return _TRACK[name]


def _score(job) -> dict:
    """One individual, in a pool worker: `(theta, collect)`. Plain floats
    back, and the replay trace only when asked for (`Swarm.collect_traces`):
    a window breeding flat out has no use for it, and it is most of what
    crosses the pipe from the pool."""
    theta, collect = job
    trace = [] if collect else None
    ep = rollout(Policy(np.asarray(theta, float)), _CFG["track"], dt=_CFG["dt"],
                 T=_CFG["T"], wing=_CFG["wing"], car=_CFG.get("car"),
                 cfg_kwargs=_CFG.get("cfg_kwargs"), tr=_track(), collect=trace)
    #  the FLYING lap where there is one; otherwise the standing-start lap,
    #  so a car that completes ONE lap inside T counts as lapped (the training
    #  T is one lap and a bit -- a flying lap only ever shows on a short map)
    fl = [b - a for a, b in zip(ep.lap_times, ep.lap_times[1:])]
    first = ep.lap_times[0] if ep.lap_times else None
    return dict(reward=float(ep.reward), s=float(ep.s_progress), t=float(ep.t),
                laps=int(ep.laps), lap_best=(min(fl) if fl else first),
                lap_first=first, lap_flying=bool(fl),
                v_mean=float(ep.v_mean), wing_frac=float(ep.wing_frac),
                wings=dict(l=round(ep.wing_l_frac, 3), r=round(ep.wing_r_frac, 3),
                           top=round(ep.wing_top_frac, 3), both=round(ep.wing_both_frac, 3)),
                ended=ep.ended, trace=(trace or []))


def _measure(theta) -> dict:
    """The deliverable's number: best flying lap at DT_EVAL, 3 laps -- on
    the car the swarm BRED on (its config: a garage build's wings, the
    session's engine, assists and grip) and the track it bred on. Without
    `cfg_kwargs` a window swarm's lap was measured on the stock config."""
    r = lap_time(Policy(np.asarray(theta, float)), _CFG["track"], wing=_CFG["wing"],
                 dt=DT_EVAL, T=_CFG.get("T_eval", 240.0), car=_CFG.get("car"),
                 cfg_kwargs=_CFG.get("cfg_kwargs"), tr=_track())
    r.pop("flying", None)
    return r


def fitness_key(ind: dict) -> tuple:
    """Sort key: reward first, then the lap (a missing lap is the worst lap)."""
    lb = ind.get("lap_best")
    return (ind.get("reward", -1e18), -(lb if lb is not None else 1e9))


class Swarm:
    def __init__(self, pop: int = 32, track: str = "arena", wing: str = "plate",
                 car=None, cfg_kwargs: dict | None = None, T: float = 70.0,
                 dt: float = DT_TRAIN, seed: int = 0, name: str | None = None,
                 sigma: float = SIGMA0, workers: int | None = None,
                 track_kw: dict | None = None, car_name: str = "corsa",
                 n_act: int = N_ACT_FREE):
        self.pop = max(4, int(pop))
        #: the policy head the genomes carry: 5 = free wings (default), 4 =
        #: the published one-wing policy (`--legacy-wings`)
        self.n_act = int(n_act)
        self.track, self.wing, self.T, self.dt = track, wing, float(T), float(dt)
        self.car, self.car_name = car, car_name
        self.cfg_kwargs = dict(cfg_kwargs or {})
        self.track_kw = dict(track_kw or {})
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.name = name or f"{track}_{car_word(car_name)}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.sigma = float(sigma)
        self.gen = 0
        self.next_id = 0
        self.seed_source = "none"
        self.seed_report: dict = {}
        self.population: list = []       # dicts: id, theta, parents, born, + scores
        self.history: list = []          # one dict per evaluated generation
        self.best: dict | None = None    # best individual ever (with theta)
        self.workers = (min(os.cpu_count() or 1, self.pop) if workers is None
                        else int(workers))
        self._pool = None
        self._pending = None
        #: whether the workers send the replay trace back with each score
        #: (`_score`); the window turns it off to breed flat out
        self.collect_traces = True
        #: the car it breeds in, for the saved bot's meta (drive/race_grid.py
        #: `bred_meta`: car, stock or yours, the garage build, the engine);
        #: set by the window, None headless
        self.bred = None

    # ---- population ----------------------------------------------------
    def _new(self, theta, parents=(), sigma: float | None = None) -> dict:
        d = dict(id=self.next_id, theta=np.asarray(theta, float).copy(),
                 parents=list(parents), born=self.gen,
                 sigma=float(self.sigma if sigma is None else sigma))
        self.next_id += 1
        return d

    def _fit_head(self, theta) -> np.ndarray:
        """A genome in THIS swarm's head: a 4-output seed is widened."""
        theta = np.asarray(theta, float).ravel()
        if self.n_act >= N_ACT_FREE:
            return Policy.widen(theta)
        if Policy.head_of(theta.size) != self.n_act:
            raise ValueError("a free-wings genome cannot seed a --legacy-wings swarm")
        return theta

    def init_population(self, theta0=None, source: str = "none",
                        report: dict | None = None) -> None:
        theta0 = (np.zeros(Policy.n_param(self.n_act)) if theta0 is None
                  else self._fit_head(theta0))
        self.seed_source = source
        self.seed_report = dict(report or {})
        self.population = [self._new(theta0)]            # the seed, unmutated
        while len(self.population) < self.pop:
            #  a WIDE generation 0: each car's own step, log-uniform
            sg = self.sigma * float(np.exp(self.rng.uniform(np.log(GEN0_LO), np.log(GEN0_HI))))
            sg = float(np.clip(sg, SIGMA_MIN, SIGMA_MAX))
            self.population.append(self._new(
                theta0 + self.rng.normal(0.0, sg, theta0.size), (0,), sg))

    def seed_from(self, spec: str | None, tr=None, verbose: bool = True) -> None:
        """'none' | seed-lap json | Policy checkpoint json."""
        if not spec or spec == "none":
            self.init_population(None, "none")
            return
        with open(spec) as fh:
            kind = json.load(fh).get("kind")
        if kind == "drive.ml.Policy":
            pol = Policy.load(spec)
            #  the lineage record: the seed's meta minus what would nest
            #  (its history, ITS seed report, the ES's learning curve)
            rep = {k: v for k, v in pol.meta.items()
                   if k not in ("curve", "history", "seed_report")}
            rep["seed_source"] = pol.meta.get("seed_source", "none")
            self.init_population(pol.theta, f"ckpt:{os.path.basename(spec)}", rep)
        else:
            from . import clone
            if tr is None:
                from .. import track as trk
                tr = trk.make_track(self.track, **self.track_kw)
            theta, rep = clone.clone_from_lap(
                spec, tr, car=self.car,
                mu_scale=float(self.cfg_kwargs.get("mu_scale", 1.0)), verbose=verbose,
                n_act=self.n_act)
            if verbose:
                print(f"  cloned {rep['path']}: lap {rep.get('lap_time')}, "
                      f"rmse steer {rep['rmse_steer']:.3f} pedal {rep['rmse_pedal']:.3f}")
            self.init_population(theta, f"lap:{os.path.basename(spec)}", rep)

    # ---- evaluation ----------------------------------------------------
    def _cfg(self) -> dict:
        return dict(track=self.track, wing=self.wing, T=self.T, dt=self.dt,
                    car=self.car, cfg_kwargs=self.cfg_kwargs, track_kw=self.track_kw)

    def pool(self):
        if self._pool is None:
            if self.workers > 1:
                self._pool = mp.Pool(self.workers, initializer=_init_worker,
                                     initargs=(self._cfg(),))
            else:
                _init_worker(self._cfg())
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def stop(self) -> None:
        """Abandon an evaluation in flight and free the pool: the population
        stays as it is (unscored children are unscored), `save_state` and
        `save_best` still work."""
        self.close()
        self._pending = None

    def start_evaluation(self) -> None:
        """Kick the generation's rollouts off in the pool; `poll()` collects."""
        if self._pending is not None:
            return
        jobs = [(ind["theta"], bool(self.collect_traces)) for ind in self.population]
        p = self.pool()
        self._t0 = time.perf_counter()
        if p is None:
            self._pending = [_score(j) for j in jobs]
        else:
            self._pending = p.map_async(_score, jobs, chunksize=1)

    def ready(self) -> bool:
        if self._pending is None:
            return False
        return isinstance(self._pending, list) or self._pending.ready()

    def poll(self) -> bool:
        """Finish a started evaluation if it is done. True when scored."""
        if not self.ready():
            return False
        res = self._pending if isinstance(self._pending, list) else self._pending.get()
        self._pending = None
        for ind, r in zip(self.population, res):
            ind.update(r)
        self.population.sort(key=fitness_key, reverse=True)
        top = self.population[0]
        improved = self.best is None or fitness_key(top) > fitness_key(self.best)
        if improved:
            self.best = {k: v for k, v in top.items() if k != "trace"}
            self.best["theta"] = top["theta"].copy()
            self.best["gen"] = self.gen
        # 1/5 rule as a NUDGE on every genome's own step: widen when stuck,
        # tighten when moving; the self-adaptation in `reproduce` does the rest
        if self.gen > 0:
            k = SIGMA_DOWN if improved else SIGMA_UP
            for ind in self.population:
                ind["sigma"] = float(np.clip(ind.get("sigma", self.sigma) * k,
                                             SIGMA_MIN, SIGMA_MAX))
        self.sigma = float(np.median([ind.get("sigma", self.sigma) for ind in self.population]))
        R = np.array([ind["reward"] for ind in self.population])
        laps = [ind["lap_best"] for ind in self.population if ind.get("lap_best")]
        lap_ind = min((i for i in self.population if i.get("lap_best")),
                      key=lambda i: i["lap_best"], default=None)
        self.history.append(dict(
            gen=self.gen, best=float(R.max()), mean=float(R.mean()),
            median=float(np.median(R)), worst=float(R.min()),
            lap_best=(min(laps) if laps else None), n_lapped=len(laps),
            #  a FLYING lap, or the first lap from the rollout's rolling start
            #  (T too short for two): only a flying one compares with the medals
            lap_flying=bool(lap_ind.get("lap_flying", True)) if lap_ind else False,
            sigma=self.sigma, best_id=top["id"],
            ended={e: sum(1 for i in self.population if i["ended"] == e)
                   for e in sorted({i["ended"] for i in self.population})},
            secs=round(time.perf_counter() - self._t0, 1)))
        return True

    def evaluate(self) -> dict:
        """Blocking: score the current generation. Returns the history row."""
        self.start_evaluation()
        if not isinstance(self._pending, list):
            self._pending.wait()
        self.poll()
        return self.history[-1]

    # ---- reproduction --------------------------------------------------
    def _tournament(self) -> dict:
        idx = self.rng.integers(0, len(self.population), TOURNAMENT)
        return self.population[int(idx.min())]        # population is sorted

    def reproduce(self) -> None:
        """Ranked population (scored) -> the next, unscored, generation."""
        assert all("reward" in ind for ind in self.population), "score first"
        n_elite = max(1, int(round(ELITE_FRAC * self.pop)))
        self.gen += 1
        nxt = []
        for ind in self.population[:n_elite]:
            e = dict(ind)
            e.pop("trace", None)
            e["theta"] = ind["theta"].copy()
            e["elite"] = True
            nxt.append(e)
        while len(nxt) < self.pop:
            a = self._tournament()
            if self.rng.random() < P_CROSS:
                b = self._tournament()
                lo = np.minimum(a["theta"], b["theta"])
                hi = np.maximum(a["theta"], b["theta"])
                span = hi - lo
                child = self.rng.uniform(lo - BLX * span, hi + BLX * span)
                parents = (a["id"], b["id"])
                sg = 0.5 * (a.get("sigma", self.sigma) + b.get("sigma", self.sigma))
            else:
                child = a["theta"].copy()
                parents = (a["id"],)
                sg = a.get("sigma", self.sigma)
            #  the child's own step: the parents', log-normally perturbed
            sg = float(np.clip(sg * np.exp(SIGMA_TAU * self.rng.normal()),
                               SIGMA_MIN, SIGMA_MAX))
            child += self.rng.normal(0.0, sg, child.size)
            c = self._new(child, parents, sg)
            nxt.append(c)
        for ind in nxt:
            for k in ("reward", "s", "t", "laps", "lap_best", "lap_first", "lap_flying",
                      "v_mean", "wing_frac", "wings", "ended"):
                if not ind.get("elite"):
                    ind.pop(k, None)
        self.population = nxt

    # ---- persistence ---------------------------------------------------
    def state_path(self) -> str:
        return os.path.join(STATE_DIR, f"{self.name}_state.json")

    def save_state(self, path: str | None = None) -> str:
        path = path or self.state_path()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        pop = []
        for ind in self.population:
            d = {k: v for k, v in ind.items() if k != "trace"}
            d["theta"] = [float(v) for v in ind["theta"]]
            pop.append(d)
        best = None
        if self.best is not None:
            best = dict(self.best)
            best["theta"] = [float(v) for v in self.best["theta"]]
        d = dict(kind=STATE_KIND, name=self.name, gen=self.gen, pop=self.pop,
                 track=self.track, track_kw=self.track_kw, wing=self.wing,
                 car_name=self.car_name, T=self.T, dt=self.dt, seed=self.seed,
                 sigma=self.sigma, next_id=self.next_id,
                 seed_source=self.seed_source, seed_report=self.seed_report,
                 rng=self.rng.bit_generator.state,
                 history=self.history, best=best, population=pop,
                 n_act=self.n_act, n_param=Policy.n_param(self.n_act), bred=self.bred)
        tmp = path + ".part"
        with open(tmp, "w") as fh:
            json.dump(d, fh, default=_json_default)
        os.replace(tmp, path)
        return path

    @classmethod
    def load_state(cls, path: str, car=None, cfg_kwargs=None,
                   workers=None) -> "Swarm":
        with open(path) as fh:
            d = json.load(fh)
        if d.get("kind") != STATE_KIND:
            raise ValueError(f"{path}: not a swarm state")
        n_act = int(d.get("n_act", N_ACT))
        if int(d.get("n_param", Policy.n_param(n_act))) != Policy.n_param(n_act):
            raise ValueError(f"{path}: genomes have {d['n_param']} parameters, "
                             f"a {n_act}-output Policy has {Policy.n_param(n_act)}")
        sw = cls(pop=d["pop"], track=d["track"], wing=d["wing"], car=car,
                 cfg_kwargs=cfg_kwargs, T=d["T"], dt=d["dt"], seed=d["seed"],
                 name=d["name"], sigma=d["sigma"], workers=workers,
                 track_kw=d.get("track_kw"), car_name=d.get("car_name", "corsa"),
                 n_act=n_act)
        sw.gen, sw.next_id = int(d["gen"]), int(d["next_id"])
        sw.seed_source, sw.seed_report = d.get("seed_source", "none"), d.get("seed_report", {})
        sw.history = list(d.get("history", []))
        sw.bred = d.get("bred") if isinstance(d.get("bred"), dict) else None
        sw.rng.bit_generator.state = d["rng"]
        sw.population = []
        for p in d["population"]:
            p = dict(p)
            p["theta"] = np.asarray(p["theta"], float)
            sw.population.append(p)
        if d.get("best"):
            sw.best = dict(d["best"])
            sw.best["theta"] = np.asarray(sw.best["theta"], float)
        return sw

    def scored(self) -> bool:
        return bool(self.population) and all("reward" in i for i in self.population)

    def measure_best(self) -> dict:
        """Re-measure the best genome at DT_EVAL (the number a human is told)."""
        if self.best is None:
            raise RuntimeError("nothing evaluated yet")
        p = self.pool()
        cfg = self._cfg()
        cfg["T_eval"] = max(240.0, 3.5 * self.T)
        if p is None:
            _init_worker(cfg)
            return _measure(self.best["theta"])
        return p.apply(_measure_with, (cfg, self.best["theta"]))

    def measure_best_async(self) -> tuple:
        """The same measurement on a one-worker pool of its own, so a window
        can keep drawing (and the generation pool keep breeding) meanwhile.
        Returns (AsyncResult, pool); the caller terminates the pool."""
        if self.best is None:
            raise RuntimeError("nothing evaluated yet")
        cfg = self._cfg()
        cfg["T_eval"] = max(240.0, 3.5 * self.T)
        p = mp.Pool(1)
        return p.apply_async(_measure_with, (cfg, self.best["theta"])), p

    def best_policy(self, measured: dict | None = None) -> Policy:
        b = self.best
        meta = dict(track=self.track, tracks=[self.track], car=self.car_name,
                    wing=self.wing, T=self.T, dt_train=self.dt,
                    swarm=self.name, gen=int(b.get("gen", self.gen)),
                    pop=self.pop, generations=len(self.history),
                    seed_source=self.seed_source, seed_report=self.seed_report,
                    individual=int(b["id"]), parents=list(b.get("parents", [])),
                    reward=round(float(b["reward"]), 2),
                    lap_best_train=b.get("lap_best"), ended=b.get("ended"),
                    measured_1ms=measured, history=self.history)
        if self.bred is not None:
            meta["bred"] = dict(self.bred)     # racing it puts it back in this car
        return Policy(b["theta"].copy(), meta)

    def save_best(self, path: str | None = None, measure: bool = True,
                  measured: dict | None = None, name: str | None = None) -> tuple:
        """Write the best genome as a `Policy` checkpoint. (path, measured).
        `measured` is a result of `measure_best[_async]` already in hand.
        `name` is the bot's name (the user's, from the save prompt): the file
        is `swarm_<name>.json` and `meta['name']` carries it; default the
        swarm's own name."""
        name = safe_name(name) or self.name
        path = path or os.path.join(CKPT_DIR, f"swarm_{name}.json")
        if measured is None and measure:
            measured = self.measure_best()
        pol = self.best_policy(measured)
        pol.meta["name"] = name
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        tmp = path + ".part"
        pol.save(tmp)
        os.replace(tmp, path)
        return path, measured


def car_word(car_name) -> str:
    """A car choice as a swarm's default name shows it (the SWARM overlay,
    the save prompt, the RACE page's bot list): a stock car by its short
    shown name (`drive.prerace.car_label`, e.g. 'n540' for the key '540i'),
    plain ASCII, lower case -- never the key; any other choice (a saved
    build's) as it is."""
    try:
        import unicodedata
        import cars
        if car_name in cars.CARS:
            from ..prerace import car_label
            w = unicodedata.normalize("NFKD", car_label(car_name))
            w = "".join(ch for ch in w.encode("ascii", "ignore").decode().lower() if ch.isalnum())
            if w:
                return w
    except Exception:                      # noqa: BLE001 -- a name is not a crash
        pass
    return str(car_name)


def safe_name(name) -> str:
    """A bot name as a file stem: letters, digits, '-' and '_' only, 40 max."""
    if not name:
        return ""
    out = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(name).strip())
    return out.strip("_")[:40]


def _measure_with(cfg, theta):
    _init_worker(cfg)
    return _measure(theta)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if hasattr(o, "__dataclass_fields__"):
        return {k: getattr(o, k) for k in o.__dataclass_fields__}
    return str(o)


# ---- CLI -------------------------------------------------------------------

def wings_str(ind) -> str:
    """'L 42% R 38% top 61% both 12%' for a scored individual, or '-'."""
    w = (ind or {}).get("wings")
    if not w:
        return "-"
    return (f"L {100 * w['l']:.0f}% R {100 * w['r']:.0f}% "
            f"top {100 * w['top']:.0f}% both {100 * w['both']:.0f}%")

def _fmt_lap(v) -> str:
    return "  --  " if v is None else f"{v:6.2f}"


def run(a) -> Swarm:
    if a.resume:
        sw = Swarm.load_state(a.resume, workers=a.workers)
        print(f"resumed {sw.name}: gen {sw.gen}, pop {sw.pop}, {len(sw.history)} scored")
    else:
        import cars
        car = None if a.car in (None, "", "corsa") else cars.get(a.car)
        kw = dict(mu_scale=float(getattr(car, "mu_scale", 1.0))) if car else None
        sw = Swarm(pop=a.pop, track=a.track, wing=a.wing, car=car, cfg_kwargs=kw,
                   T=a.duration, dt=a.dt, seed=a.rng, name=a.name, sigma=a.sigma,
                   workers=a.workers, car_name=a.car or "corsa",
                   n_act=(N_ACT if a.legacy_wings else N_ACT_FREE))
        sw.seed_from(a.seed)
        print(f"swarm {sw.name}: pop {sw.pop}, seed {sw.seed_source}, "
              f"{Policy.n_param(sw.n_act)} genes ({sw.n_act} outputs), T {sw.T:.0f} s, "
              f"dt {sw.dt * 1e3:.0f} ms, workers {sw.workers}")
    try:
        for _ in range(a.gens):
            if sw.scored():
                sw.reproduce()
            h = sw.evaluate()
            sw.save_state()
            print(f"  gen {h['gen']:3d}  best {h['best']:8.1f}  mean {h['mean']:8.1f}  "
                  f"lap {_fmt_lap(h['lap_best'])}  lapped {h['n_lapped']:2d}/{sw.pop}  "
                  f"sigma {h['sigma']:.3f}  {h['ended']}  "
                  f"wings {wings_str(sw.best)}  [{h['secs']} s]", flush=True)
        if a.out or a.save:
            path, m = sw.save_best(a.out)
            print(f"  saved {path}\n  at 1 ms: {json.dumps(m, default=str)}")
    finally:
        sw.close()
    return sw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="drive.ml swarm: a genetic algorithm over driving genomes")
    ap.add_argument("--pop", type=int, default=32, help="cars per generation")
    ap.add_argument("--gens", type=int, default=10, help="generations to run")
    ap.add_argument("--track", default="arena")
    ap.add_argument("--wing", default="plate", choices=("off", "fin", "plate"))
    ap.add_argument("--car", default=None,
                    help="corsa (default) | rally | 540i | express")
    ap.add_argument("--seed", default="none",
                    help="'none', a seed-lap json the user drove (runs/swarm/seed_*.json), "
                         "or a Policy checkpoint to breed from")
    ap.add_argument("--resume", default=None, help="a runs/swarm/*_state.json to continue")
    ap.add_argument("--duration", type=float, default=70.0, help="s of sim per car")
    ap.add_argument("--dt", type=float, default=DT_TRAIN)
    ap.add_argument("--sigma", type=float, default=SIGMA0)
    ap.add_argument("--rng", type=int, default=0, help="the GA's random seed")
    ap.add_argument("--name", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None, help="checkpoint path for the best (implies --save)")
    ap.add_argument("--save", action="store_true", help="save the best to drive/ml/checkpoints/")
    ap.add_argument("--legacy-wings", action="store_true",
                    help="breed the published 4-output policy (one wing toggle, the car "
                         "picks the flank) instead of the free-wings head")
    a = ap.parse_args(argv)
    run(a)
    return 0


# ---- self-check ------------------------------------------------------------
def self_check(verbose: bool = True) -> bool:
    """Fast and deterministic: tiny population, short rollouts, one process."""
    import tempfile
    ok = True

    def rep(tag, cond, msg=""):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'ok' if cond else 'FAIL'}] {tag}  {msg}")

    import cars
    words = [car_word(k) for k in cars.CAR_ORDER]
    rep("a default name shows each stock car by its shown name, never its key",
        all(w and w != k and w.isascii() and w.isalnum() for w, k in zip(words, cars.CAR_ORDER))
        and car_word("build:my van") == "build:my van", ", ".join(words))
    sw = Swarm(pop=6, track="arena", T=6.0, seed=1, workers=1, name="selfcheck")
    sw.seed_from("none")
    rep("seed 0 is the anchor", float(np.abs(sw.population[0]["theta"]).max()) == 0.0)
    rep("free-wings head", sw.population[0]["theta"].size == Policy.n_param(N_ACT_FREE)
        and Policy(sw.population[0]["theta"]).free_wings)
    sgs = [i["sigma"] for i in sw.population[1:]]
    rep("gen 0 spread", min(sgs) < SIGMA0 < max(sgs) and len(set(sgs)) == len(sgs),
        f"steps {min(sgs):.3f}..{max(sgs):.3f}")
    h1 = sw.evaluate()
    rep("scored", sw.scored(), f"best {h1['best']:.1f} mean {h1['mean']:.1f}")
    rep("sorted", all(fitness_key(a) >= fitness_key(b) for a, b in
                      zip(sw.population, sw.population[1:])))
    ids = [i["id"] for i in sw.population]
    sw.reproduce()
    n_el = max(1, int(round(ELITE_FRAC * 6)))
    rep("elites kept", sum(1 for i in sw.population if i.get("elite")) == n_el
        and sw.population[0]["id"] == ids[0])
    rep("children unscored", all("reward" not in i for i in sw.population[n_el:]))
    rep("parents recorded", all(i["parents"] for i in sw.population[n_el:]))
    h2 = sw.evaluate()
    rep("best never regresses", sw.best["reward"] >= h1["best"],
        f"{h1['best']:.1f} -> {h2['best']:.1f}")
    rep("children carry their own step",
        all(SIGMA_MIN <= i["sigma"] <= SIGMA_MAX for i in sw.population)
        and len({round(i["sigma"], 6) for i in sw.population}) > 1)
    # determinism: same seed, same two generations, bit for bit
    sw2 = Swarm(pop=6, track="arena", T=6.0, seed=1, workers=1, name="selfcheck2")
    sw2.seed_from("none")
    sw2.evaluate(); sw2.reproduce(); sw2.evaluate()
    rep("deterministic", sw2.history[-1]["best"] == h2["best"]
        and np.array_equal(sw2.population[-1]["theta"], sw.population[-1]["theta"]))
    # state round-trip
    with tempfile.TemporaryDirectory() as td:
        p = sw.save_state(os.path.join(td, "s.json"))
        sw3 = Swarm.load_state(p, workers=1)
        rep("state round-trip", sw3.gen == sw.gen and len(sw3.population) == 6
            and np.array_equal(sw3.population[2]["theta"], sw.population[2]["theta"])
            and sw3.best["id"] == sw.best["id"])
        sw3.reproduce()
        sw.reproduce()
        rep("resume continues the rng", np.array_equal(sw3.population[-1]["theta"],
                                                        sw.population[-1]["theta"]))
        # the deliverable: a Policy the rest of the package loads
        cp, m = sw3.save_best(os.path.join(td, "best.json"), measure=False)
        pol = Policy.load(cp)
        rep("best is a Policy", pol.meta.get("swarm") == "selfcheck"
            and np.array_equal(pol.theta, sw.best["theta"]))
        # a seed from that checkpoint puts it at index 0, unmutated
        sw4 = Swarm(pop=4, track="arena", T=6.0, seed=2, workers=1, name="sc4")
        sw4.seed_from(cp)
        rep("checkpoint seeds index 0", np.array_equal(sw4.population[0]["theta"], pol.theta)
            and sw4.seed_source.startswith("ckpt:"))
        # a 4-output (published ES) checkpoint seeds a free-wings swarm, widened:
        # its steer / pedal rows carry over, its (authority-less) wing row does
        # not -- the wing rows are zero and the anchor's priors decide, which is
        # what that checkpoint's wings were doing all along
        old = Policy(np.zeros(Policy.N_PARAM))
        old.theta[-2] = 0.7                       # its wing bias
        old.theta[-4] = 0.3                       # its steer bias
        op = old.save(os.path.join(td, "old.json"))
        sw5 = Swarm(pop=4, track="arena", T=6.0, seed=3, workers=1, name="sc5")
        sw5.seed_from(op)
        w = Policy(sw5.population[0]["theta"])
        rep("4-output checkpoint widened", w.free_wings and w.b2[0] == 0.3
            and w.b2[2] == w.b2[3] == w.b2[4] == 0.0 and not np.any(w.W2[2:]))
        # the legacy head still breeds
        sw6 = Swarm(pop=4, track="arena", T=3.0, seed=4, workers=1, name="sc6", n_act=N_ACT)
        sw6.seed_from("none")
        sw6.evaluate()
        rep("legacy head breeds", sw6.population[0]["theta"].size == Policy.N_PARAM
            and sw6.scored())
        # the replay trace is optional: a swarm breeding flat out asks for none
        sw7 = Swarm(pop=4, track="arena", T=3.0, seed=5, workers=1, name="sc7")
        sw7.seed_from("none")
        sw7.collect_traces = False
        sw7.evaluate()
        rep("no trace unless asked", sw7.scored()
            and all(not i.get("trace") for i in sw7.population)
            and bool(sw6.population[0].get("trace")),
            f"{len(sw6.population[0].get('trace') or [])} rows with, 0 without")
    return ok


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1 or sys.argv[1:] == ["--self-check"]:
        raise SystemExit(0 if self_check() else 1)
    raise SystemExit(main())
