"""Bayesian optimisation the way AeroBO does it, in 150 lines of numpy: a
Gaussian process on the unit cube (Matern 5/2, one length-scale picked by
marginal likelihood each round), expected improvement, a Sobol start.

Sized for this garage: a design evaluation is a lattice build (~1-3 ms),
so a 32-evaluation run finishes in well under a second and the read-outs
can show the trace live. A random-search twin with the same budget runs
alongside so the panel can say what BO bought -- AeroBO's crossover-map
question at garage scale.

`maximise(f, bounds, ...)`: f maps a design vector (real units) to a
scalar; return -inf (or raise ValueError) for an infeasible design and it
is scored at a penalty below the worst feasible value seen.

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


def maximise(f, bounds, n_init: int = 8, n_iter: int = 24, seed: int = 0,
             progress=None, x0=None, labels=None) -> dict:
    """BO on `f` over `bounds` = [(lo, hi), ...]. Returns x_best (real units),
    f_best, and the whole history. `x0` (real units) is evaluated first.
    `labels` names the coordinates, in the same order as `bounds`, and is
    carried into the result so a read-out cannot mis-name one."""
    bounds = np.asarray(bounds, float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    d = len(lo)
    labels = _check_labels(labels, d)
    rng = np.random.default_rng(seed)
    U = sobol(n_init, d, seed)
    if x0 is not None:
        u0 = (np.asarray(x0, float) - lo) / np.maximum(hi - lo, 1e-12)
        U = np.vstack([np.clip(u0, 0.0, 1.0)[None, :], U[:-1]])
    X, Y, feas = [], [], []

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
                best_trace=np.maximum.accumulate(np.where(np.isfinite(y), y, -np.inf)).tolist())


def random_search(f, bounds, n: int = 32, seed: int = 1, labels=None) -> dict:
    bounds = np.asarray(bounds, float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    labels = _check_labels(labels, len(lo))
    rng = np.random.default_rng(seed)
    X, Y = [], []
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
    return dict(x_best=X[i], f_best=float(y[i]), n_eval=n, labels=labels,
                best_trace=np.maximum.accumulate(y).tolist())


def describe(res: dict, labels=None, fmt: str = "{:+.4g}") -> str:
    """`x_best` written out coordinate by coordinate, in the caller's order --
    the order `bounds` was given in, so for the garage the DESIGNER page's row
    order. Falls back to `x[i]` only if nobody named the variables."""
    x = np.asarray(res["x_best"], float).ravel()
    names = labels or res.get("labels") or [f"x[{i}]" for i in range(x.size)]
    names = _check_labels(names, x.size)
    return "   ".join(f"{n} {fmt.format(float(v))}" for n, v in zip(names, x))


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
    if verbose:
        print("  ALL PASS" if ok else "  FAILURES ABOVE")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
