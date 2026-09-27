"""Bayesian optimisation the way AeroBO does it, in 150 lines of numpy: a
Gaussian process on the unit cube (Matern 5/2, one length-scale picked by
marginal likelihood each round), expected improvement, a Sobol start.

Sized for this garage: a design evaluation is a lattice build (~1-3 ms),
so a 32-evaluation run finishes in well under a second and the read-outs
can show the trace live. A random-search control with the same budget can
be run beside it so the panel can say what BO bought -- AeroBO's
crossover-map question at garage scale.

`maximise(f, bounds, ...)`: f maps a design vector (real units) to a
scalar; return -inf (or raise ValueError) for an infeasible design and it
is scored at a penalty below the worst feasible value seen.

`BOStepper` / `RandomStepper` are the same two searches in ask/tell form.
The garage steps them from its frame (`drive/design_jobs.py`), one
evaluation or one GP fit at a time, so a run is watched rather than waited
for; `maximise` and `random_search` are plain loops over them and return
what they always returned, to the last bit -- the self-check proves it
against verbatim copies of the old loops. `converged_report` is AeroBO's
"has it stopped improving?" rule, which the garage's "stop early" uses.

Coordinate order is the CALLER's throughout: `bounds`, `x0`, the `labels`
the caller may hand in, `x_best` and `X` are all in one order and it is the
caller's. The garage's caller takes that order from
`wing.DESIGN_VARS` = the DESIGNER page's row order, so what the optimiser
reports reads top-to-bottom like the page. `labels` is checked against the
dimension of `bounds` -- a label list that has drifted out of step with the
bounds is the one mistake here that produces plausible, wrong read-outs
rather than an exception, so it is made into an exception.
"""

from __future__ import annotations

import math

import numpy as np

_SQRT5 = math.sqrt(5.0)


# --------------------------------------------------------------------------- #
#  AeroBO's MEASURED search defaults                                           #
# --------------------------------------------------------------------------- #
#  Everything in this block is read off `data/search_budget.json` in
#  urop-bo-aero -- the frozen payload its `experiments/budget_analysis.py`
#  emits -- and nothing in it is a number chosen here. The carsim searches
#  used to size themselves by hand (48 evaluations for a section, 32 for a
#  wing, and a Sobol seed of half the budget), which was a guess in the one
#  place that repository has a measurement.

#: effort -> the convergence fraction the budget is sized for, on the study's
#: own normalised progress scale (0 = one design drawn blind out of the box,
#: 1 = the best design any run of that case found). They stop at 99 %
#: because that is where the evidence stops.
EFFORT_TARGETS = {"quick": 0.90, "balanced": 0.95, "thorough": 0.99}
EFFORTS = tuple(EFFORT_TARGETS)

#: `evals = intercept + per_dim * d`, fitted over the study's cases, per
#: target. The residual RMS and the case count are the study's own, and they
#: are carried because a law with an 8-11 evaluation RMS is a SIZING RULE and
#: not a prediction -- quoting it without its spread would be the third way
#: this number has been overstated.
#:
#:     target  intercept  per_dim   rms   cases   max measured
#:      0.90      10.03     2.22     8.0    13         47
#:      0.95       9.61     3.08    10.7    13         64
#:      0.99       2.62     5.97    11.9    13         91
BUDGET_LAW = {"quick": (10.03, 2.22, 8.0), "balanced": (9.61, 3.08, 10.7),
              "thorough": (2.62, 5.97, 11.9)}

#: Sobol initial design: `clamp(round(per_dim * d), min, max)`.
#:
#: 0.5 x d IS MEASURED, and it is measured against exactly the rule carsim
#: was using. Over 15 of the study's cases the seed sizes rank
#: 0.5 -> 1.00, 1.0 -> 2.25, 2.0 -> 2.88, 4.0 -> 4.25 (mean rank, lower
#: better), and the airfoil class re-measures it on its own cases and agrees:
#: "4 initial points (0.5x the design vector) leads on mean progress over the
#: deciding budgets". carsim was spending 16 of 48 on the initial design of a
#: 10-row section -- 1.6 x d, between the two worst arms.
N_INIT = {"per_dim": 0.5, "min": 4, "max": 16}

#: ...and the rule for a CONSTRAINED box, which is a different question: the
#: initial design there decides whether the run ever sees a FEASIBLE point,
#: and a run that does not see one rides a box corner for the rest of its
#: budget. On the study's reported 20-D box only 4.7 % of draws fly, and
#: 0.5 x d returned NOTHING on 2 of 3 seeds while d + 1 returned a design on
#: 3 of 3.
#:
#: carsim's boxes are NOT that box, and the reason is measured here: a
#: uniform draw over the section box is refused 42 % of the time on the top
#: role and 36 % on the flank, so about 60 % of draws fly and a 5-point seed
#: is all-refused with probability ~1 %. The unconstrained rule applies.
N_INIT_CONSTRAINED = {"per_dim": 1.0, "min": 8, "max": 48}


def budget_for(d: int, effort: str = "balanced") -> int:
    """How many evaluations AeroBO's measured law asks for at `d` rows."""
    if effort not in BUDGET_LAW:
        raise ValueError(f"effort must be one of {EFFORTS}, got {effort!r}")
    a, b, _rms = BUDGET_LAW[effort]
    return max(8, int(round(a + b * int(d))))


def n_init_for(d: int, budget: int, constrained: bool = False) -> int:
    """The Sobol seed size for a `d`-row search on `budget` evaluations."""
    spec = N_INIT_CONSTRAINED if constrained else N_INIT
    n = int(round(spec["per_dim"] * int(d)))
    n = max(spec["min"], min(spec["max"], n))
    return max(1, min(n, int(budget) - 1))


def split_for(d: int, budget: int, constrained: bool = False) -> tuple:
    """`(n_init, n_iter)` -- what `maximise` should be handed."""
    n_init = n_init_for(d, budget, constrained)
    return n_init, max(1, int(budget) - n_init)


def _matern52(X, Y, ell):
    d = np.sqrt(np.maximum(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1), 0.0)) / ell
    return (1.0 + _SQRT5 * d + 5.0 / 3.0 * d * d) * np.exp(-_SQRT5 * d)


class GP:
    def __init__(self, X, y, ell=0.3, noise=1e-6):
        self.X = np.asarray(X, float)
        self.mu_y = float(np.mean(y))
        self.sd_y = float(np.std(y)) or 1.0
        self.y = (np.asarray(y, float) - self.mu_y) / self.sd_y
        self.ell, self.noise = float(ell), float(noise)
        K = _matern52(self.X, self.X, self.ell) + self.noise * np.eye(len(self.X))
        self.L = np.linalg.cholesky(K + 1e-9 * np.eye(len(self.X)))
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))

    def log_marginal(self) -> float:
        return float(-0.5 * self.y @ self.alpha - np.log(np.diag(self.L)).sum()
                     - 0.5 * len(self.y) * math.log(2 * math.pi))

    def predict(self, Xs):
        Ks = _matern52(np.asarray(Xs, float), self.X, self.ell)
        mu = Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        var = np.maximum(1.0 + self.noise - (v * v).sum(0), 1e-12)
        return mu * self.sd_y + self.mu_y, np.sqrt(var) * self.sd_y


def _fit(X, y):
    best, best_ll = None, -np.inf
    for ell in (0.12, 0.2, 0.3, 0.45, 0.7, 1.0, 1.6):
        try:
            g = GP(X, y, ell)
        except np.linalg.LinAlgError:
            continue
        ll = g.log_marginal()
        if ll > best_ll:
            best, best_ll = g, ll
    return best


def _ei(mu, sd, best):
    try:
        from scipy.special import erf
    except Exception:                                    # pragma: no cover
        erf = np.vectorize(math.erf)
    z = (mu - best) / np.maximum(sd, 1e-12)
    cdf = 0.5 * (1.0 + erf(z / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return (mu - best) * cdf + sd * pdf


def sobol(n: int, d: int, seed: int = 0) -> np.ndarray:
    try:
        from scipy.stats import qmc
        m = int(math.ceil(math.log2(max(n, 2))))
        pts = qmc.Sobol(d, scramble=True, seed=seed).random_base2(m)
        return pts[:n]
    except Exception:                                   # pragma: no cover
        return np.random.default_rng(seed).random((n, d))


def _check_labels(labels, d: int):
    if labels is None:
        return None
    labels = list(labels)
    if len(labels) != d:
        raise ValueError(f"{len(labels)} labels for a {d}-variable design vector: "
                         f"{labels}")
    return labels


#: What a REFUSED candidate raises. `safe_eval` scores these -inf, as
#: `maximise` always has; anything else propagates, because a bug in an
#: objective must not read as a design the box happened to refuse.
REFUSALS = (ValueError, ZeroDivisionError, FloatingPointError)


def safe_eval(f, x) -> float:
    """`f(x)` under `maximise`'s own refusal rule: an exception in `REFUSALS`
    and any non-finite result score -inf. Other exceptions propagate."""
    try:
        val = float(f(x))
    except REFUSALS:
        return -math.inf
    return val if math.isfinite(val) else -math.inf


# --------------------------------------------------------------------------- #
#  ASK / TELL. The garage steps a search from its frame -- one evaluation, or #
#  one GP fit, per unit of work -- so a run is watched rather than waited     #
#  for. `maximise` and `random_search` are loops over these same two objects  #
#  and return what they always returned, to the last bit: the self-check      #
#  holds verbatim copies of the old loops and compares.                       #
# --------------------------------------------------------------------------- #
class BOStepper:
    """The ask/tell form of `maximise`. Driving it with `safe_eval(f, x)`
    reproduces `maximise(f, ...)` bit for bit.

    `ask()` hands out the next candidate in REAL units and is idempotent
    until `tell()`; a GP step's `ask()` is where the GP fit and the EI search
    happen, so it is the expensive half of an evaluation. `tell(val)` records
    the score and returns the progress record. `stop()` ends the run with
    what is on record; `result()` is `maximise`'s dict over it.

    What the unit cube remembers is the `u` that PRODUCED each candidate, not
    `(x - lo) / (hi - lo)` recomputed afterwards: the two can differ in the
    last bit, and the GP trained on the other one would propose a different
    next design.
    """

    def __init__(self, bounds, n_init: int = 8, n_iter: int = 24, seed: int = 0,
                 x0=None, labels=None, resume=None):
        bounds = np.asarray(bounds, float)
        lo, hi = bounds[:, 0], bounds[:, 1]
        d = len(lo)
        self.lo, self.hi, self.d = lo, hi, d
        self.labels = _check_labels(labels, d)
        self._rng = np.random.default_rng(seed)
        self._X, self._Y, self._feas = [], [], []
        self._resumed = resume is not None
        if resume is not None:
            Xr = np.asarray(resume["X"], float).reshape(-1, d)
            yr = np.asarray(resume["y"], float).reshape(-1)
            if len(Xr) != len(yr):
                raise ValueError(f"resume carries {len(Xr)} designs and {len(yr)} scores")
            for xr, v in zip(Xr, yr):
                self._X.append(np.clip((xr - lo) / np.maximum(hi - lo, 1e-12), 0.0, 1.0))
                v = float(v) if math.isfinite(v) else -math.inf
                self._Y.append(v)
                self._feas.append(math.isfinite(v))
            #  the continuation's own draws must not repeat the run it extends
            self._rng = np.random.default_rng(seed + 1000 + len(self._Y))
            U = np.zeros((0, d))
        else:
            U = sobol(n_init, d, seed)
            if x0 is not None:
                u0 = (np.asarray(x0, float) - lo) / np.maximum(hi - lo, 1e-12)
                U = np.vstack([np.clip(u0, 0.0, 1.0)[None, :], U[:-1]])
        self._U = U
        self.n_prior = len(self._Y)
        #: this run's Sobol block (x0 counted in it); 0 on a continuation
        self.n_sobol = len(U)
        self.n_planned = self.n_prior + len(U) + max(0, int(n_iter))
        #: the running best INCLUDING what was inherited, per evaluation -- the
        #: same numbers as `result()["best_trace"]`, kept as it grows
        self.trace = np.maximum.accumulate(
            np.where(np.isfinite(self._Y), self._Y, -np.inf)).tolist() if self._Y else []
        self.best = self.trace[-1] if self.trace else -math.inf
        self.pending = None
        self._pu = None
        self._pmu = self._psd = None
        self._pphase = ""
        self._stopped = False

    @property
    def k(self) -> int:
        """Evaluations on record, inherited ones included."""
        return len(self._Y)

    @property
    def done(self) -> bool:
        return self._stopped or self.k >= self.n_planned

    def phase(self) -> str:
        """Of the pending candidate, else of the next one: "sobol", "ei", or
        "resume" (a GP step of a continuation)."""
        if self.pending is not None:
            return self._pphase
        if self.k - self.n_prior < self.n_sobol:
            return "sobol"
        return "resume" if self._resumed else "ei"

    def ask(self):
        """The next candidate in real units, or None when the run is done."""
        if self.pending is not None:
            return self.pending
        if self.done:
            return None
        j = self.k - self.n_prior
        if j < self.n_sobol:
            u, mu, sd, phase = self._U[j], None, None, "sobol"
        else:
            u, mu, sd = self._gp_step()
            phase = "resume" if self._resumed else "ei"
        self._pu, self._pmu, self._psd, self._pphase = u, mu, sd, phase
        self.pending = self.lo + u * (self.hi - self.lo)
        return self.pending

    def _gp_step(self):
        """One GP-BO proposal: `maximise`'s loop body, rng calls in its order."""
        d, rng, X = self.d, self._rng, self._X
        y = np.asarray(self._Y)
        fe = np.asarray(self._feas)
        if fe.any():
            worst = y[fe].min()
            span = max(y[fe].max() - worst, 1.0)
            ys = np.where(fe, y, worst - 0.5 * span)
        else:
            ys = np.zeros_like(y)
        gp = _fit(np.asarray(X), ys)
        if gp is None:
            return rng.random(d), None, None
        best = ys.max()
        cand = rng.random((1500, d))
        top = np.argsort(ys)[-3:]
        for t in top:
            cand = np.vstack([cand, np.clip(np.asarray(X)[t] + 0.08 * rng.standard_normal((150, d)), 0, 1)])
        mu, sd = gp.predict(cand)
        ei = _ei(mu, sd, best)
        i = int(np.argmax(ei))
        #  the GP's own prediction at the design it chose, kept so a read-out
        #  can set what the surrogate expected beside what the design scored
        return cand[i], float(mu[i]), float(sd[i])

    def tell(self, val: float) -> dict:
        """Record the pending candidate's score (non-finite -> -inf) and
        return the progress record."""
        if self.pending is None:
            raise RuntimeError("tell() with no candidate pending: ask() first")
        v = float(val)
        if not math.isfinite(v):
            v = -math.inf
        x, u = self.pending, self._pu
        self._X.append(u)
        self._Y.append(v)
        self._feas.append(math.isfinite(v))
        improved = v > self.best
        if improved:
            self.best = v
        self.trace.append(self.best)
        rec = {"k": self.k, "n": self.n_planned, "i": self.k - self.n_prior,
               "phase": self._pphase, "x": np.array(x, float), "val": v, "best": self.best,
               "feasible": math.isfinite(v), "improved": bool(improved),
               "mu": self._pmu, "sd": self._psd, "secs": 0.0, "info": None}
        self.pending = self._pu = self._pmu = self._psd = None
        return rec

    def stop(self) -> None:
        """No further asks; a pending candidate is discarded unflown."""
        self._stopped = True
        self.pending = self._pu = self._pmu = self._psd = None

    def result(self) -> dict | None:
        """`maximise`'s dict over what is on record; None if nothing is."""
        if not self._Y:
            return None
        lo, hi = self.lo, self.hi
        y = np.asarray(self._Y)
        i = int(np.argmax(y))
        return dict(x_best=lo + np.asarray(self._X)[i] * (hi - lo), f_best=float(y[i]),
                    X=lo + np.asarray(self._X) * (hi - lo), y=y, n_eval=len(self._Y),
                    labels=self.labels, n_prior=self.n_prior,
                    best_trace=np.maximum.accumulate(np.where(np.isfinite(y), y, -np.inf)).tolist())


class RandomStepper:
    """The ask/tell form of `random_search`: the same rng seeding, the same
    draws, the same record. Stores REAL-unit designs, as it always has."""

    def __init__(self, bounds, n: int = 32, seed: int = 1, labels=None, resume=None):
        bounds = np.asarray(bounds, float)
        lo, hi = bounds[:, 0], bounds[:, 1]
        self.lo, self.hi, self.d = lo, hi, len(lo)
        self.labels = _check_labels(labels, len(lo))
        self._X, self._Y = [], []
        if resume is not None:
            self._X = [np.asarray(x, float) for x in resume["X"]]
            self._Y = [float(v) if math.isfinite(v) else -math.inf for v in resume["y"]]
        #  a resume re-seeds even when the record it extends is EMPTY -- that
        #  is what `random_search` did, so the stepper does it too
        self._rng = np.random.default_rng(seed + (1000 + len(self._Y) if resume is not None else 0))
        self.n_prior = len(self._Y)
        self.n_sobol = 0
        self.n_planned = self.n_prior + max(0, int(n))
        self.trace = np.maximum.accumulate(self._Y).tolist() if self._Y else []
        self.best = self.trace[-1] if self.trace else -math.inf
        self.pending = None
        self._stopped = False

    @property
    def k(self) -> int:
        return len(self._Y)

    @property
    def done(self) -> bool:
        return self._stopped or self.k >= self.n_planned

    def phase(self) -> str:
        return "random"

    def ask(self):
        if self.pending is not None:
            return self.pending
        if self.done:
            return None
        lo, hi = self.lo, self.hi
        self.pending = lo + self._rng.random(len(lo)) * (hi - lo)
        return self.pending

    def tell(self, val: float) -> dict:
        if self.pending is None:
            raise RuntimeError("tell() with no candidate pending: ask() first")
        v = float(val)
        v = v if math.isfinite(v) else -math.inf
        x = self.pending
        self._X.append(x)
        self._Y.append(v)
        improved = v > self.best
        if improved:
            self.best = v
        self.trace.append(self.best)
        self.pending = None
        return {"k": self.k, "n": self.n_planned, "i": self.k - self.n_prior,
                "phase": "random", "x": np.array(x, float), "val": v, "best": self.best,
                "feasible": math.isfinite(v), "improved": bool(improved),
                "mu": None, "sd": None, "secs": 0.0, "info": None}

    def stop(self) -> None:
        self._stopped = True
        self.pending = None

    def result(self) -> dict | None:
        """`random_search`'s dict over what is on record; None if nothing is."""
        if not self._Y:
            return None
        y = np.asarray(self._Y)
        i = int(np.argmax(y))
        return dict(x_best=self._X[i], f_best=float(y[i]), n_eval=len(self._Y), labels=self.labels,
                    X=np.asarray(self._X, float), y=y,
                    best_trace=np.maximum.accumulate(y).tolist())


def maximise(f, bounds, n_init: int = 8, n_iter: int = 24, seed: int = 0,
             progress=None, x0=None, labels=None, resume=None) -> dict:
    """BO on `f` over `bounds` = [(lo, hi), ...]. Returns x_best (real units),
    f_best, and the whole history. `x0` (real units) is evaluated first.
    `labels` names the coordinates, in the same order as `bounds`, and is
    carried into the result so a read-out cannot mis-name one.

    `resume` CONTINUES a run instead of starting one -- AeroBO's "keep going".
    A BO loop is a function of its observations, so a previous result (or any
    dict carrying `X` in real units and `y`) is handed in as the TRAINING
    SET: no Sobol block is drawn, nothing is re-flown, the counter opens at
    `len(y)` and `n_iter` more evaluations are bought. The record that comes
    back holds all of them; `n_prior` says where the old run stopped.

    A loop over `BOStepper`, which is what the garage steps live."""
    st = BOStepper(bounds, n_init, n_iter, seed, x0, labels, resume)
    while (x := st.ask()) is not None:
        rec = st.tell(safe_eval(f, x))
        if progress is not None:
            progress(rec["k"], x, rec["val"])
    return st.result()


def random_search(f, bounds, n: int = 32, seed: int = 1, labels=None,
                  resume=None) -> dict:
    """The control BO is read against. `resume` extends a previous record by
    `n` more draws, the way `maximise(resume=...)` does, so the two traces
    stay the same length after a continuation. A loop over `RandomStepper`."""
    st = RandomStepper(bounds, n, seed, labels, resume)
    while (x := st.ask()) is not None:
        st.tell(safe_eval(f, x))
    return st.result()


# --------------------------------------------------------------------------- #
#  "STOP EARLY IF IT STOPS IMPROVING"                                          #
# --------------------------------------------------------------------------- #
def patience_for(n: int) -> int:
    """How many evaluations without a worthwhile gain end a run of `n`: a
    quarter of it, at least 4. CARSIM'S OWN HEURISTIC, not a measured number
    -- AeroBO's study measured a patience for its own kinds of search, not for
    these -- and every read-out that shows it says so."""
    return max(4, int(round(0.25 * n)))


def converged_report(best_trace, patience: int, tol: float = 0.002) -> dict:
    """Did a best-so-far trace (of a MAXIMISED objective) flatten out?

    AeroBO's `budget.converged_report` rule, the same arithmetic its stop
    rule uses, so the verdict a finished run is given and the rule that would
    have stopped it cannot drift apart. The span is measured from the run
    itself (last finite best minus the first), so the rule needs no idea of
    the objective's scale; a run that has covered no span has not converged.

    verdict: "empty" (fewer than two finite values) | "too_short" (not longer
    than `patience`) | "no_incumbent" (nothing finite `patience` evaluations
    ago) | "converged" (the last `patience` bought <= tol of the span) |
    "climbing". Returns dict(verdict, gain, span, frac, patience, tol, n,
    text), `text` being AeroBO's sentence with this run's numbers."""
    hist = []
    for v in (best_trace if best_trace is not None else []):
        v = None if v is None else float(v)
        hist.append(v if v is not None and math.isfinite(v) else None)
    patience, tol = int(patience), float(tol)
    out = dict(verdict="", gain=None, span=None, frac=None, patience=patience, tol=tol,
               n=len(hist), text="")
    finite = [v for v in hist if v is not None]
    if len(finite) < 2:
        out.update(verdict="empty", text="fewer than two evaluations produced a finite score")
        return out
    span = abs(finite[-1] - finite[0])
    out["span"] = span
    if len(hist) <= patience:
        out.update(verdict="too_short",
                   text=(f"the rule needs more than {patience} evaluations before it can "
                         f"fire and this run made {len(hist)}, so it says nothing either way"))
        return out
    then = hist[len(hist) - 1 - patience]
    if then is None:
        out.update(verdict="no_incumbent",
                   text=(f"there was no finite incumbent {patience} evaluations ago, so "
                         f"the run was still finding its first feasible designs"))
        return out
    gain = finite[-1] - then
    frac = None if span <= 0.0 else gain / span
    flat = span > 0.0 and gain <= tol * span
    out.update(verdict="converged" if flat else "climbing", gain=gain, frac=frac,
               text=(f"the last {patience} evaluations bought "
                     f"{'nothing' if frac is None else f'{frac:.1%}'} of the span this run "
                     f"covered, {'under' if flat else 'over'} the {tol:.1%} the rule calls flat"))
    return out


def describe(res: dict, labels=None, fmt: str = "{:+.4g}") -> str:
    """`x_best` written out coordinate by coordinate, in the caller's order --
    the order `bounds` was given in, so for the garage the DESIGNER page's row
    order. Falls back to `x[i]` only if nobody named the variables."""
    x = np.asarray(res["x_best"], float).ravel()
    names = labels or res.get("labels") or [f"x[{i}]" for i in range(x.size)]
    names = _check_labels(names, x.size)
    return "   ".join(f"{n} {fmt.format(float(v))}" for n, v in zip(names, x))


# --------------------------------------------------------------------------- #
#  THE PARITY REFERENCES. `maximise` and `random_search` exactly as they were
#  before the ask/tell steppers replaced their loops (optimize.py as read
#  2026-09-24), copied VERBATIM and only renamed. The self-check drives the
#  steppers against them and asks for equality to the last bit: a reordered
#  rng call or a recomputed unit-cube coordinate would move a number, and the
#  garage's searches are only reproducible if that never happens.
# --------------------------------------------------------------------------- #
def _maximise_reference(f, bounds, n_init: int = 8, n_iter: int = 24, seed: int = 0,
             progress=None, x0=None, labels=None, resume=None) -> dict:
    """BO on `f` over `bounds` = [(lo, hi), ...]. Returns x_best (real units),
    f_best, and the whole history. `x0` (real units) is evaluated first.
    `labels` names the coordinates, in the same order as `bounds`, and is
    carried into the result so a read-out cannot mis-name one.

    `resume` CONTINUES a run instead of starting one -- AeroBO's "keep going".
    A BO loop is a function of its observations, so a previous result (or any
    dict carrying `X` in real units and `y`) is handed in as the TRAINING
    SET: no Sobol block is drawn, nothing is re-flown, the counter opens at
    `len(y)` and `n_iter` more evaluations are bought. The record that comes
    back holds all of them; `n_prior` says where the old run stopped."""
    bounds = np.asarray(bounds, float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    d = len(lo)
    labels = _check_labels(labels, d)
    rng = np.random.default_rng(seed)
    X, Y, feas = [], [], []
    if resume is not None:
        Xr = np.asarray(resume["X"], float).reshape(-1, d)
        yr = np.asarray(resume["y"], float).reshape(-1)
        if len(Xr) != len(yr):
            raise ValueError(f"resume carries {len(Xr)} designs and {len(yr)} scores")
        for xr, v in zip(Xr, yr):
            X.append(np.clip((xr - lo) / np.maximum(hi - lo, 1e-12), 0.0, 1.0))
            v = float(v) if math.isfinite(v) else -math.inf
            Y.append(v)
            feas.append(math.isfinite(v))
        #  the continuation's own draws must not repeat the run it extends
        rng = np.random.default_rng(seed + 1000 + len(Y))
        U = np.zeros((0, d))
    else:
        U = sobol(n_init, d, seed)
        if x0 is not None:
            u0 = (np.asarray(x0, float) - lo) / np.maximum(hi - lo, 1e-12)
            U = np.vstack([np.clip(u0, 0.0, 1.0)[None, :], U[:-1]])
    n_prior = len(Y)

    def evaluate(u):
        x = lo + u * (hi - lo)
        try:
            val = float(f(x))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            val = -math.inf
        if not math.isfinite(val):
            val = -math.inf
        X.append(u)
        Y.append(val)
        feas.append(math.isfinite(val))
        if progress is not None:
            progress(len(Y), x, val)
        return val

    for u in U:
        evaluate(u)
    for it in range(n_iter):
        y = np.asarray(Y)
        fe = np.asarray(feas)
        if fe.any():
            worst = y[fe].min()
            span = max(y[fe].max() - worst, 1.0)
            ys = np.where(fe, y, worst - 0.5 * span)
        else:
            ys = np.zeros_like(y)
        gp = _fit(np.asarray(X), ys)
        if gp is None:
            u = rng.random(d)
        else:
            best = ys.max()
            cand = rng.random((1500, d))
            top = np.argsort(ys)[-3:]
            for t in top:
                cand = np.vstack([cand, np.clip(np.asarray(X)[t] + 0.08 * rng.standard_normal((150, d)), 0, 1)])
            mu, sd = gp.predict(cand)
            ei = _ei(mu, sd, best)
            u = cand[int(np.argmax(ei))]
        evaluate(u)
    y = np.asarray(Y)
    i = int(np.argmax(y))
    return dict(x_best=lo + np.asarray(X)[i] * (hi - lo), f_best=float(y[i]),
                X=lo + np.asarray(X) * (hi - lo), y=y, n_eval=len(Y), labels=labels,
                n_prior=n_prior,
                best_trace=np.maximum.accumulate(np.where(np.isfinite(y), y, -np.inf)).tolist())


def _random_reference(f, bounds, n: int = 32, seed: int = 1, labels=None,
                  resume=None) -> dict:
    """The control BO is read against. `resume` extends a previous record by
    `n` more draws, the way `maximise(resume=...)` does, so the two traces
    stay the same length after a continuation."""
    bounds = np.asarray(bounds, float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    labels = _check_labels(labels, len(lo))
    X, Y = [], []
    if resume is not None:
        X = [np.asarray(x, float) for x in resume["X"]]
        Y = [float(v) if math.isfinite(v) else -math.inf for v in resume["y"]]
    rng = np.random.default_rng(seed + (1000 + len(Y) if resume is not None else 0))
    for _ in range(n):
        x = lo + rng.random(len(lo)) * (hi - lo)
        try:
            v = float(f(x))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            v = -math.inf
        X.append(x)
        Y.append(v if math.isfinite(v) else -math.inf)
    y = np.asarray(Y)
    i = int(np.argmax(y))
    return dict(x_best=X[i], f_best=float(y[i]), n_eval=len(Y), labels=labels,
                X=np.asarray(X, float), y=y,
                best_trace=np.maximum.accumulate(y).tolist())


# --------------------------------------------------------------------------- #
def self_check(verbose: bool = True) -> bool:
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and passed
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}: {msg}")

    def branin(x):                        # maximise -branin; optimum 0.397887
        a, b, c, r, s, t = 1, 5.1 / (4 * math.pi ** 2), 5 / math.pi, 6, 10, 1 / (8 * math.pi)
        return -(a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * math.cos(x[0]) + s)

    bnds = [(-5.0, 10.0), (0.0, 15.0)]
    import time
    t0 = time.perf_counter()
    bo = maximise(branin, bnds, n_init=8, n_iter=24, seed=3)
    dt = time.perf_counter() - t0
    rs = random_search(branin, bnds, n=32, seed=3)
    rep("BO on Branin within 0.5 of the optimum in 32 evals", -bo["f_best"] - 0.397887 < 0.5,
        f"best {-bo['f_best']:.4f} (opt 0.3979) in {dt * 1e3:.0f} ms")
    rep("BO beats random search at equal budget", bo["f_best"] >= rs["f_best"],
        f"BO {-bo['f_best']:.3f} vs random {-rs['f_best']:.3f}")

    def penalised(x):
        if x[0] > 5.0:
            return -math.inf
        return branin(x)

    bo2 = maximise(penalised, bnds, n_init=8, n_iter=16, seed=5)
    rep("infeasible designs are tolerated", math.isfinite(bo2["f_best"]) and bo2["x_best"][0] <= 5.0,
        f"best {-bo2['f_best']:.3f} at x0 {bo2['x_best'][0]:.2f}")
    rep("history complete", bo["n_eval"] == 32 and len(bo["best_trace"]) == 32, "")

    # -- a continuation continues (AeroBO's "keep going") ------------------- #
    #  The first 32 evaluations are the training set; nothing is re-flown,
    #  the counter opens at 32 and the record comes back holding all 40.
    calls = []
    more = maximise(lambda x: calls.append(1) or branin(x), bnds, n_iter=8, resume=bo)
    rep("keep going: 8 more evaluations bought, 32 inherited, none re-flown",
        len(calls) == 8 and more["n_eval"] == 40 and more["n_prior"] == 32
        and len(more["best_trace"]) == 40
        and np.allclose(more["X"][:32], bo["X"]) and np.allclose(more["y"][:32], bo["y"]),
        f"{len(calls)} flown, {more['n_eval']} on record")
    rep("...and the best can only improve", more["f_best"] >= bo["f_best"]
        and more["best_trace"][:32] == bo["best_trace"],
        f"{-bo['f_best']:.4f} -> {-more['f_best']:.4f}")
    rs_more = random_search(branin, bnds, n=8, seed=3, resume=rs)
    rep("the random control extends the same way", rs_more["n_eval"] == 40
        and len(rs_more["best_trace"]) == 40 and rs_more["f_best"] >= rs["f_best"], "")

    # -- the coordinate order is the caller's, end to end ------------------- #
    named = maximise(branin, bnds, n_init=8, n_iter=8, seed=3, labels=("x1", "x2"))
    txt = describe(named)
    rep("labels ride along and name x_best in the caller's order",
        named["labels"] == ["x1", "x2"] and txt.startswith("x1 ") and "x2 " in txt[3:], txt)
    try:
        maximise(branin, bnds, n_init=4, n_iter=1, labels=("x1",))
        rep("a label list out of step with bounds raises", False, "no exception")
    except ValueError as exc:
        rep("a label list out of step with bounds raises", True, str(exc)[:56])

    # Reordering a design vector reassigns which Sobol coordinate drives which
    # variable, so the TRAJECTORY moves; the problem does not. Branin with its
    # two variables swapped is the same surface relabelled, and the same 32
    # evaluations land just as close to the same optimum.
    swapped = maximise(lambda x: branin((x[1], x[0])), [bnds[1], bnds[0]],
                       n_init=8, n_iter=24, seed=3)
    rep("permuting the design vector does not change the problem",
        -swapped["f_best"] - 0.397887 < 0.5,
        f"as given {-bo['f_best']:.4f}, swapped {-swapped['f_best']:.4f} (opt 0.3979)")
    #  AeroBO's measured defaults, and the arithmetic that reads them
    rep("the budget law is the payload's, to the digit",
        BUDGET_LAW["balanced"][:2] == (9.61, 3.08)
        and BUDGET_LAW["quick"][:2] == (10.03, 2.22)
        and BUDGET_LAW["thorough"][:2] == (2.62, 5.97))
    b10 = budget_for(10)
    rep("a 10-row section asks for the measured budget", b10 == 40,
        f"{b10} evaluations at 'balanced' (9.61 + 3.08 x 10)")
    rep("effort moves it the way the study says",
        budget_for(10, "quick") < b10 < budget_for(10, "thorough"),
        f"quick {budget_for(10, 'quick')}, balanced {b10}, "
        f"thorough {budget_for(10, 'thorough')}")
    rep("the Sobol seed is 0.5 x d, clamped to [4, 16]",
        n_init_for(10, 40) == 5 and n_init_for(4, 40) == 4
        and n_init_for(40, 80) == 16,
        f"d=10 -> {n_init_for(10, 40)}, d=4 -> {n_init_for(4, 40)}, "
        f"d=40 -> {n_init_for(40, 80)}")
    rep("...and it never eats the whole budget", n_init_for(10, 5) == 4,
        f"budget 5 -> n_init {n_init_for(10, 5)}")
    ni, nit = split_for(10, 40)
    rep("the split adds up", ni + nit == 40, f"{ni} Sobol + {nit} BO")
    rep("a constrained box seeds harder, as the study measured",
        n_init_for(10, 40, constrained=True) == 10
        and n_init_for(10, 40, constrained=True) > n_init_for(10, 40))
    try:
        budget_for(10, "enthusiastic")
        rep("an unknown effort is refused", False)
    except ValueError:
        rep("an unknown effort is refused rather than defaulted", True)

    # -- the steppers ARE the old loops, to the last bit -------------------- #
    #  `maximise` / `random_search` (now loops over the steppers) and hand-
    #  written ask/tell loops, against the verbatim references above. Bytes,
    #  not allclose: a reordered rng call or a recomputed unit-cube coordinate
    #  moves the last bit first, and the garage's runs are only reproducible
    #  if that never happens.
    parity = ("x_best", "f_best", "X", "y", "n_eval", "n_prior", "best_trace")

    def bits(v):
        return np.asarray(v, float).tobytes()

    def same(a, b):
        return all((k in a) == (k in b) and (k not in a or bits(a[k]) == bits(b[k]))
                   for k in parity)

    def by_hand(st, fn):
        while (x := st.ask()) is not None:
            st.tell(safe_eval(fn, x))
        return st.result()

    ref = _maximise_reference(branin, bnds, n_init=8, n_iter=24, seed=3)
    rep("P1 maximise and a BOStepper loop == the old maximise, bit for bit (Branin)",
        same(bo, ref) and same(by_hand(BOStepper(bnds, 8, 24, 3), branin), ref),
        f"{ref['n_eval']} evaluations, best {-ref['f_best']:.6f}")

    x0 = np.array([2.0, 7.5])
    ref2 = _maximise_reference(branin, bnds, n_init=6, n_iter=10, seed=7, x0=x0)
    st2 = BOStepper(bnds, 6, 10, 7, x0=x0)
    first = np.array(st2.ask())
    hand2 = by_hand(st2, branin)
    lo_, hi_ = np.asarray(bnds)[:, 0], np.asarray(bnds)[:, 1]
    rep("P2 ...with x0: prepended as evaluation 1, the Sobol block's last point dropped",
        same(maximise(branin, bnds, n_init=6, n_iter=10, seed=7, x0=x0), ref2)
        and same(hand2, ref2) and np.allclose(first, x0, rtol=0.0, atol=1e-12)
        and bits(ref2["X"][1:6]) == bits(lo_ + sobol(6, 2, 7)[:5] * (hi_ - lo_)),
        f"first design {first[0]:.4f}, {first[1]:.4f}; {ref2['n_eval']} evaluations")

    def refuses(x):                       # the three ways a design is refused
        if x[0] > 5.0:
            return -math.inf
        if x[1] > 12.0:
            raise ValueError("outside the box")
        if x[0] < -3.0:
            return float("nan")
        return branin(x)

    ref3 = _maximise_reference(refuses, bnds, n_init=8, n_iter=12, seed=5)
    n_ref = int(np.sum(~np.isfinite(ref3["y"])))
    try:
        safe_eval(lambda x: None + 1, np.zeros(2))
        leaks = False
    except TypeError:
        leaks = True
    rep("P3 ...with refusals (-inf, ValueError, NaN), and safe_eval refuses nothing else",
        n_ref > 0 and same(maximise(refuses, bnds, n_init=8, n_iter=12, seed=5), ref3)
        and same(by_hand(BOStepper(bnds, 8, 12, 5), refuses), ref3)
        and safe_eval(lambda x: 1.0 / 0.0, np.zeros(2)) == -math.inf and leaks,
        f"{n_ref} of {ref3['n_eval']} refused; a TypeError propagates")

    ref4 = _maximise_reference(branin, bnds, n_iter=8, resume=bo)
    st4 = BOStepper(bnds, n_iter=8, resume=bo)
    phases4 = []
    while (x := st4.ask()) is not None:
        phases4.append(st4.phase())
        st4.tell(safe_eval(branin, x))
    rep("P4 a resume of 8 more == the old one: 32 inherited, prefix kept, all GP steps",
        same(more, ref4) and same(st4.result(), ref4) and more["n_prior"] == 32
        and bits(more["y"][:32]) == bits(bo["y"]) and phases4 == ["resume"] * 8,
        f"n_prior {st4.n_prior}, {st4.k} on record")

    r_ref = _random_reference(branin, bnds, n=32, seed=3)
    rm_ref = _random_reference(branin, bnds, n=8, seed=3, resume=rs)
    rep("P5 RandomStepper == the old random_search, fresh and resumed",
        same(rs, r_ref) and same(by_hand(RandomStepper(bnds, 32, 3), branin), r_ref)
        and same(rs_more, rm_ref)
        and same(by_hand(RandomStepper(bnds, 8, 3, resume=rs), branin), rm_ref),
        f"{r_ref['n_eval']} draws, then {rm_ref['n_eval']} on record")

    empty = dict(X=[], y=[])
    e_new = random_search(branin, bnds, n=6, seed=4, resume=empty)
    rep("P6 a resume of an EMPTY record still re-seeds to seed + 1000",
        same(e_new, _random_reference(branin, bnds, n=6, seed=4, resume=empty))
        and bits(e_new["X"]) == bits(random_search(branin, bnds, n=6, seed=1004)["X"])
        and bits(e_new["X"]) != bits(random_search(branin, bnds, n=6, seed=4)["X"]))

    st7 = BOStepper(bnds, 4, 6, seed=2)
    a, b = st7.ask(), st7.ask()
    idem = a is b and st7.k == 0
    try:
        BOStepper(bnds, 4, 2).tell(1.0)
        told = False
    except RuntimeError:
        told = True
    for _ in range(6):                    # the 4 Sobol points and 2 GP steps
        st7.tell(safe_eval(branin, st7.ask()))
    st7.ask()                             # a third GP candidate handed out ...
    st7.stop()                            # ... and never flown
    rsst = RandomStepper(bnds, 3, 1)
    rep("P7 ask() is idempotent until tell(); stop() leaves what is on record",
        idem and told and rsst.ask() is rsst.ask() and st7.done and st7.ask() is None
        and st7.pending is None and same(st7.result(),
                                         _maximise_reference(branin, bnds, n_init=4, n_iter=2, seed=2))
        and BOStepper(bnds, 4, 2).result() is None,
        f"stopped at {st7.k} of {st7.n_planned}")

    inf = -math.inf
    verdicts = {v: converged_report(tr, 4)["verdict"] for v, tr in (
        ("empty", [inf, inf, 1.0]),
        ("too_short", [1.0, 2.0, 3.0, 3.0]),
        ("no_incumbent", [inf, inf, 1.0, 2.0, 3.0, 3.0]),
        ("converged", [1.0, 5.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
        ("climbing", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))}
    txt8 = converged_report([0.0, 0.543] + [0.6] * 7 + [1.0], 8)["text"]
    rep("P8 converged_report: WingLab's five verdicts, its sentence, and no span is no verdict",
        all(k == v for k, v in verdicts.items())
        and txt8 == ("the last 8 evaluations bought 45.7% of the span this run covered, "
                     "over the 0.2% the rule calls flat")
        and converged_report([2.0] * 10, 4)["verdict"] == "climbing"
        and (patience_for(10), patience_for(40), patience_for(160)) == (4, 10, 40),
        txt8)

    st9 = BOStepper(bnds, 4, 4, seed=1)
    recs = []
    while (x := st9.ask()) is not None:
        recs.append(st9.tell(safe_eval(branin, x)))
    gp9 = [r for r in recs if r["phase"] == "ei"]
    rep("P9 GP steps carry the surrogate's finite mu / sd; Sobol points carry none",
        len(gp9) == 4 and all(math.isfinite(r["mu"]) and r["sd"] > 0.0 for r in gp9)
        and all(r["mu"] is None and r["sd"] is None for r in recs if r["phase"] == "sobol")
        and [r["k"] for r in recs] == list(range(1, 9)) and recs[-1]["best"] == max(r["val"] for r in recs),
        f"sd at the GP steps {min(r['sd'] for r in gp9):.3g} to {max(r['sd'] for r in gp9):.3g}")

    if verbose:
        print("  ALL PASS" if ok else "  FAILURES ABOVE")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
