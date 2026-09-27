"""drive/views_common.py -- the builders two or more stage views share
(PLAN2 §8): the "designing" row every section view opens with, the section
outline and its three-panel polar, THE LIVE EVALUATION GRAPH (§8.5), the
run banner and the "now evaluating" block, and the tags a run's state is
drawn with.

Since the AeroBO pivot every number these builders draw is AeroBO's own --
an evaluation payload of a live or replayed EngineJob, a stored run record,
a section report -- or carsim's (the mission). They read the models as the
shell hands them in a `ViewCtx` (`aerobo_models` objects), never `garage`,
never the engine: a view calls a builder rather than copying it, so the
airfoil, the endplate and the wing stages cannot drift apart.

RUNNING AND FINISHED MUST NOT BE CONFUSED (the owner). Everything here
draws a run in one of two looks:
  * live -- a spinner, `RUNNING · k/N` in ACCENT, a determinate bar, the
    status line with the elapsed time and what is left, the newest point
    of the graph ringed;
  * finished -- the terminal tag from the model's STORED outcome (never a
    finished job read later): `DONE · N/N` GOOD, `CONVERGED · k/N` GOOD,
    `STOPPED · k/N` WARN ("best kept"), `FAILED` BAD, with the wall time.

Every builder is safe on a model that has nothing yet: no run, no
coordinates, no records. What it cannot show it says it cannot show, and it
never invents a number.

The graph's API (Agent E's w.conv and r.evals call it with
`WingModel.graph()`; the section views with `SurfaceModel.graph()`):

    convergence_fig(records, *, N, n_prior=0, n_init=None, handoff=None,
                    units="", live=False) -> cae.plot.Figure
    convergence_card(ui, graph, *, title="Evaluations", label=None,
                     outcome=None, height=300)
    run_banner(ui, job, outcome, *, idle="")
"""

from __future__ import annotations

import math

import numpy as np

from . import design_jobs as dj
from .cae import theme as T
from .cae.plot import Figure

#: AeroBO's objective contract (`aerobo.api.PENALTY`, objective.py): the
#: score a REFUSED evaluation reports -- the evaluator could not fly it (a
#: solver failure, a geometry out of its validity). Restated here because a
#: view never imports the engine; it is AeroBO's number, not carsim's.
PENALTY = -100.0

#: how many live screened sections the sweep list shows, newest first
SWEEP_ROWS = 8

#: AeroBO's screen statuses (a swept section's `status`), in words
SWEEP_STATUS = {"gate_tc": "dropped by the t/c gate", "gate_cm": "dropped by the |cm| gate",
                "no_cl_bracket": "never reaches the design cl", "unconverged": "XFOIL did not converge",
                "no_polar": "no polar"}


# --------------------------------------------------------------------------- #
#  small readers                                                               #
# --------------------------------------------------------------------------- #
def finite(v) -> bool:
    """Is `v` a finite number (None, NaN, ±inf and non-numbers are not)?"""
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def num(v, spec="{:.4g}", none="—") -> str:
    """`v` through `spec`, or `none` when it is not a finite number."""
    return spec.format(float(v)) if finite(v) else none


def refused(rec) -> bool:
    """An evaluation the engine REFUSED (no value, or AeroBO's PENALTY):
    drawn as a cross on the floor, never as a score."""
    f = (rec or {}).get("f")
    return not finite(f) or float(f) == PENALTY


def duration(s) -> str:
    """`~t` for a wall time in seconds: tenths under 10 s, then s, min, h."""
    s = float(s)
    if s < 10.0:
        return f"~{s:.1f} s"
    if s < 90.0:
        return f"~{s:.0f} s"
    if s < 5400.0:
        return f"~{s / 60.0:.0f} min"
    return f"~{s / 3600.0:.1f} h"


def colour(name):
    """A theme colour by the name a job's chip or a stored tag carries
    ("ACCENT", "GOOD", "WARN", "BAD"); a colour tuple passes through."""
    if isinstance(name, (tuple, list)):
        return tuple(name)
    return getattr(T, str(name), T.INK_MUTED)


def tag_of(job=None, outcome=None):
    """(text, colour tuple, note) of a run: the LIVE chip while `job` runs
    (`RUNNING · k/N`, `STOPPING · k/N`), else the terminal tag of the model's
    stored `outcome` (`design_jobs.terminal_tag`); None before any run."""
    if job is not None:
        chip = job.chip()
        if chip:
            return chip[0], colour(chip[1]), ""
    t = dj.terminal_tag(outcome)
    if t is None:
        return None
    return t[0], colour(t[1]), t[2] if len(t) > 2 else ""


def draw_tag(ui, tag, *, wall=None) -> None:
    """A run's tag as a chip (`tag_of` / a graph dict's `tag`: (text, colour
    name or tuple[, note])), the note and the wall time after it, muted."""
    if not tag:
        return
    ui.tag(str(tag[0]), colour(tag[1]))
    #  a short note rides beside the chip ("best kept", "stopped improving");
    #  a FAILED run's note is the engine's error, which the view states in
    #  full on a line of its own
    note = str(tag[2]) if len(tag) > 2 and tag[2] else ""
    bits = [note] if note and len(note) <= 40 and str(tag[0]) != "FAILED" else []
    if finite(wall) and not str(tag[0]).startswith(("RUNNING", "STOPPING")):
        bits.append(f"in {dj.clock(wall)}")
    if bits:
        ui.label(" · ".join(bits), css=T.HINT_CSS, colour=T.INK_MUTED)


def progress_bar(ui, frac, height=6, *, fill=None) -> None:
    """A determinate progress bar: PROG_TRACK under an ACCENT fill."""
    frac = max(0.0, min(1.0, float(frac))) if finite(frac) else 0.0
    fill = fill or T.ACCENT

    def draw(surf, rect):
        surf.fill(T.PROG_TRACK, rect)
        if frac > 0.0:
            surf.fill(fill, (rect.x, rect.y, max(1, int(rect.w * frac)), rect.h))
    ui.custom(height, draw)


def mono_lines(ui, lines, css=11, pitch=16) -> None:
    """Lines of MONO text at a log's pitch, (text, colour) each, cut to the
    column: one block, not a stack of rows."""
    if not lines:
        return
    lh = int(math.floor(pitch * T.S + 0.5))

    def draw(surf, rect):
        fh = T.font("mono", css).get_height()
        for j, (text, col) in enumerate(lines):
            T.text(surf, text, rect.x, rect.y + j * lh + (lh - fh + 1) // 2, "mono", css, col,
                   clip_w=rect.w)
    ui.custom(len(lines) * pitch, draw)


def x_text(x, labels=None, n=None) -> str:
    """A design vector in words: `taper 0.45 · alpha_deg 8.67 · ...` (or the
    bare numbers without labels), the first `n` rows."""
    if x is None:
        return "—"
    xs = list(x)[: n or None]
    if labels:
        return " · ".join(f"{lab} {num(v, '{:.4g}')}" for lab, v in zip(labels, xs))
    return "[" + ", ".join(num(v, "{:.4g}") for v in xs) + "]"


# --------------------------------------------------------------------------- #
#  the run banner (live vs finished, unmistakably)                             #
# --------------------------------------------------------------------------- #
def run_banner(ui, job, outcome, *, idle="") -> None:
    """The run's state, in one of the two looks and nowhere ambiguous:

    live (`job` given) -- a spinner, the `RUNNING · k/N` chip, the status
    line (`section (XFOIL) evaluation 7/164 · BO · 2:41 · ≈ 6:10 left`) and a
    determinate bar k/N;
    finished -- the terminal tag from the STORED `outcome` (DONE / CONVERGED
    / STOPPED / FAILED) with its note and wall time, and the reason a run
    ended early when it did;
    neither -- `idle` (a hint), or nothing."""
    if job is not None:
        text, kind, frac = job.status()
        with ui.row():
            ui.spinner("")
            draw_tag(ui, tag_of(job))
            ui.label(text, css=11.5, family="mono", colour=T.WARN if kind == "warn" else T.INK)
        progress_bar(ui, frac if frac is not None else (job.k / job.n if job.n else 0.0))
        return
    tag = tag_of(None, outcome)
    if tag is None:
        if idle:
            ui.hint(idle)
        return
    with ui.row():
        draw_tag(ui, tag, wall=(outcome or {}).get("wall"))
    st = (outcome or {}).get("state")
    why = (outcome or {}).get("stop_reason")
    if st == "error":
        ui.hint(f"the engine said: {(outcome or {}).get('error') or 'no reason given'}", "bad")
    elif st in ("stopped", "converged") and why:
        ui.hint(f"ended at {outcome.get('k')} of {outcome.get('n')}: {why} — the best so far is kept",
                "warn" if st == "stopped" else "ok")


# --------------------------------------------------------------------------- #
#  the live evaluation graph (PLAN2 §8.5)                                      #
# --------------------------------------------------------------------------- #
def _y_window(vals, best):
    """(lo, hi) of the graph's y axis: every scored value and the best line,
    except far outliers BELOW -- under Q1 − 3·IQR of the DISTINCT scored
    values -- which would squash the search into a line (AeroBO's
    convergence_yrange keeps the axis to the scored values the same way).
    Distinct, because an SLSQP leg re-flies the incumbent many times and a
    quartile of repeats would call an ordinary Sobol point wild. The best
    line is always inside."""
    v = vals[np.isfinite(vals)]
    b = best[np.isfinite(best)]
    if not v.size and not b.size:
        return None
    lo = float(np.min(v)) if v.size else float(np.min(b))
    hi = float(max(np.max(v) if v.size else -math.inf, np.max(b) if b.size else -math.inf))
    u = np.unique(v)
    if u.size >= 4:
        q1, q3 = np.percentile(u, [25.0, 75.0])
        lo = max(lo, float(q1 - 3.0 * (q3 - q1)))
    if b.size:
        lo = min(lo, float(np.min(b)))
    span = hi - lo
    if span <= 1e-12 * max(1.0, abs(hi)):
        span = max(1.0, abs(hi) * 0.1)
        lo, hi = lo - 0.5 * span, hi + 0.5 * span
        return lo, hi
    return lo - 0.08 * span, hi + 0.08 * span


def convergence_fig(records, *, N, n_prior=0, n_init=None, handoff=None, units="",
                    live=False) -> Figure:
    """EVERY evaluation of a run, and the best so far (PLAN2 §8.5).

    `records` are the eval payloads `{n, best, f, feasible, g, x}` -- a live
    job's, the model's stored ones, a replayed fixture's -- inherited ones of
    a Keep going included (their `n` is 1..n_prior). The x axis runs 1..N,
    N the budget, even while k < N: how far the run has to go is visible.

      * a feasible evaluation: a filled ACCENT dot at its score;
      * an infeasible one (a constraint missed, a value still flown): a
        hollow WARN ring at its score;
      * a refused one (no value, or AeroBO's PENALTY −100): a BAD cross on
        the floor, "refused (k)" -- and a value far below the rest (under
        Q1 − 3·IQR) the same, "off scale (k)", so one wild point cannot
        squash the search;
      * the best feasible so far: an ACCENT step line from the first
        feasible evaluation;
      * inherited evaluations (1..n_prior): a grey band, "inherited";
      * the Sobol → BO split (`n_init`, a fresh run's): a dashed line;
      * the BO → SLSQP handoff (`handoff` evaluations after n_prior, a
        bo_slsqp run's): a dotted line;
      * `live`: the newest evaluation ringed (it moves every frame).

    `units` is the y axis: AeroBO's `score_units` of the run ("CZ/CD (=
    F_z/D)", "composite J - goal penalty", "minus the lap time [-s]") --
    what the engine MAXIMISES, so the best line only ever rises."""
    N = max(1, int(N or 0))
    fig = Figure(xlabel="evaluation", ylabel=str(units or "objective"), legend=True,
                 empty_text="no evaluations yet — the first point appears when the first one lands",
                 xlim=(0.5, N + 0.5))
    recs = [r for r in (records or ()) if r is not None]
    n_prior = max(0, int(n_prior or 0))
    if n_prior:
        fig.band(0.5, n_prior + 0.5, colour=T.INK_FAINT, alpha=0.14, label=f"inherited ({n_prior})")
    sobol = bool(n_init and n_prior == 0 and 0 < int(n_init) < N)
    slsqp = bool(handoff and 0 < int(handoff) and n_prior + int(handoff) < N)
    #  AeroBO's bo_slsqp often hands over ONE BO step after its Sobol block
    #  (12 | 1 | 40 at 53): two labels a few pixels apart overprint, so the
    #  first line then names both and the second draws unlabelled
    close = sobol and slsqp and (n_prior + int(handoff) - int(n_init)) < 0.07 * N
    if sobol:
        fig.vline(int(n_init) + 0.5, colour=T.INK_FAINT, dash="dash",
                  label=(f"Sobol | BO · SLSQP after {n_prior + int(handoff)}" if close
                         else "Sobol | BO"))
    if slsqp:
        fig.vline(n_prior + int(handoff) + 0.5, colour=T.INK_FAINT, dash="dot",
                  label=None if close else "BO → SLSQP")
    if not recs:
        return fig
    ks = np.array([float(r.get("n", i + 1)) for i, r in enumerate(recs)])
    vals = np.array([float(r["f"]) if not refused(r) else math.nan for r in recs])
    feas = np.array([bool(r.get("feasible")) for r in recs])
    best = np.array([float(r["best"]) if finite(r.get("best")) else math.nan for r in recs])
    win = _y_window(vals, best)
    if win is None:                                         # every evaluation refused
        fig.ylim = (0.0, 1.0)
        fig.off_scale(ks, name="refused")
        return fig
    lo, hi = win
    fig.ylim = (lo, hi)
    scored = np.isfinite(vals)
    low = scored & (vals < lo)
    on = scored & ~low
    if (on & feas).any():
        fig.scatter(ks[on & feas], vals[on & feas], colour=T.ACCENT, size=6, opacity=0.8,
                    name="feasible")
    if (on & ~feas).any():
        fig.scatter(ks[on & ~feas], vals[on & ~feas], colour=T.WARN, symbol="hollow", size=7,
                    name="infeasible")
    if (~scored).any():
        fig.off_scale(ks[~scored], name="refused")
    if low.any():
        fig.off_scale(ks[low], name="off scale")
    fb = np.isfinite(best)
    if fb.any():
        i0 = int(np.flatnonzero(fb)[0])
        fig.line(ks[i0:][fb[i0:]], best[i0:][fb[i0:]], colour=T.ACCENT, width=2.5, step=True,
                 name="best so far")
    if live:
        last = recs[-1]
        y = float(last["f"]) if not refused(last) and float(last["f"]) >= lo else lo + 0.02 * (hi - lo)
        fig.scatter([ks[-1]], [y], colour=T.INK, symbol="hollow", size=13, name="newest")
    return fig


def convergence_card(ui, graph, *, title="Evaluations", label=None, outcome=None,
                     height=300) -> None:
    """The live evaluation graph in its card: a head row -- `<label> — k/N
    evaluations` and the run's tag (RUNNING while live, the stored terminal
    tag after, with the wall time) -- the figure, and one line saying how to
    read it. `graph` is a model's `graph()` dict: {records, N, n_prior,
    n_init, handoff, units, objective, live, tag}; `label` the objective in
    words (default: the graph's objective key); `outcome` the model's stored
    outcome (for the wall time of a finished run)."""
    g = dict(graph or {})
    recs = list(g.get("records") or [])
    N = int(g.get("N") or 0)
    k = max([int(r.get("n", 0)) for r in recs] or [0])
    live = bool(g.get("live"))
    with ui.card(title, pad=False):
        with ui.row():
            ui.gap(9)
            ui.label(f"{label or g.get('objective') or 'objective'} — {k}/{N} evaluations",
                     css=12, bold=True)
            ui.spacer()
            draw_tag(ui, g.get("tag"), wall=None if live else (outcome or {}).get("wall"))
            ui.gap(9)
        if live and N:
            progress_bar(ui, k / N, height=3)
        ui.plot(convergence_fig(recs, N=N, n_prior=g.get("n_prior") or 0, n_init=g.get("n_init"),
                                handoff=g.get("handoff"), units=g.get("units") or "",
                                live=live), height)
    n_ref = sum(1 for r in recs if refused(r))
    n_inf = sum(1 for r in recs if not refused(r) and not r.get("feasible"))
    bits = [f"{len(recs) - n_ref - n_inf} feasible", f"{n_inf} infeasible", f"{n_ref} refused"]
    ui.hint("One point per evaluation WingLab flew (" + ", ".join(bits) + "); the step line is the "
            "best feasible score so far, what the engine maximises.",
            help="A filled dot is a feasible design and its score. A hollow ring flew but missed a "
                 "constraint. A cross on the floor was refused by the evaluator (WingLab scores it "
                 "−100, and shows the surrogate the worst design that flew instead) or lies far "
                 "below the rest. The x axis runs to the budget, so the gap to its right is what "
                 "is left to fly. A grey band is what a Keep going inherited: nothing there was "
                 "flown again.")


# --------------------------------------------------------------------------- #
#  the live block                                                              #
# --------------------------------------------------------------------------- #
def now_evaluating(ui, ctx, job, *, labels=None, units="") -> None:
    """What the run is doing, drawn ONLY while `job` is live (after the run
    it is gone and the view shows the stored result): the evaluation and its
    phase, the last candidate -- its score, feasible or not, refused --, the
    best so far and where it was found, the incumbent's design vector
    (`labels` its row names, AeroBO's param labels), and the pace."""
    if job is None or not getattr(ctx, "live", True) or job.chip() is None:
        return
    ui.hairline()
    phase = dj.PHASE_LONG.get(job.phase, job.phase) or "starting"
    ui.kv("evaluation", f"{job.k}/{job.n} · {phase}")
    rec = job.current if getattr(job, "records", None) else None
    if rec:
        if refused(rec):
            ui.kv("last candidate", "refused by the evaluator (WingLab scores it −100)", colour=T.WARN)
        elif not rec.get("feasible"):
            ui.kv("last candidate", f"{num(rec.get('f'))} — flew, but misses a constraint",
                  colour=T.WARN)
        else:
            ui.kv("last candidate", f"{num(rec.get('f'))}  feasible")
    if job.incumbents:
        k, best, x = job.incumbents[-1]
        #  k None: the best was carried in from the evaluations a Keep going
        #  inherits (design_jobs._take_eval), not found by this run
        where = f"(evaluation {k})" if k is not None else "(inherited: found before this run resumed)"
        ui.kv("best so far", f"{num(best)} {units}".rstrip() + f"  {where}")
        if x is not None:
            ui.kv("its design", x_text(x, labels))
    else:
        ui.kv("best so far", "— (no feasible design yet)")
    new = job.k - job.n_prior
    el = job.elapsed()
    if new > 0 and el > 0.0:
        ui.kv("pace", f"{el / new:.2f} s an evaluation · {dj.clock(el)} so far")


def sweep_block(ui, job) -> None:
    """A live SCREEN under its action row: where it is (the library pass, or
    section i/n of the shortlist sweep), a bar, and the last sections swept,
    newest first -- each with its L/D at the design cl and t/c, or why it
    was dropped."""
    ui.hairline()
    i, n = int(job.k), max(1, int(job.n))
    phase = dj.PHASE_LONG.get(job.phase, job.phase) or "starting"
    with ui.row():
        ui.spinner("")
        ui.label(f"screening · {phase}", css=12, family="mono")
        ui.spacer()
        if job.sweep:
            ui.label(f"section {i}/{n}", css=12, family="mono")
    progress_bar(ui, i / n if job.sweep else 0.0)
    lines = []
    for r in reversed(job.sweep[-SWEEP_ROWS:]):
        name = str(r.get("name", "?"))
        if r.get("status") == "ok":
            mark = "eligible" if r.get("eligible") else "gated out"
            lines.append((f"{int(r.get('i', 0)):>3}  {name:<18} L/D@cl {num(r.get('ldcr'), '{:6.1f}')}  "
                          f"t/c {num(r.get('tc'), '{:.3f}')}  {mark}",
                          T.INK_MUTED if r.get("eligible") else T.INK_FAINT))
        else:
            st = str(r.get("status") or "no_polar")
            lines.append((f"{int(r.get('i', 0)):>3}  {name:<18} {SWEEP_STATUS.get(st, st)}",
                          T.INK_FAINT if st.startswith("gate_") else T.WARN))
    if i > SWEEP_ROWS:
        lines.append((f"…and {i - SWEEP_ROWS} earlier — the ranking arrives when the screen ends",
                      T.INK_FAINT))
    mono_lines(ui, lines)


# --------------------------------------------------------------------------- #
#  the "designing" row (every section view opens with it)                      #
# --------------------------------------------------------------------------- #
def own_section(m) -> str:
    """The family's own section of surface `m`, the one it flies until one
    is chosen: AeroBO's NACA 24tt bank at the built problem's t/c on the wing
    (carwing's published section), NACA 00tt at the searched t/c on the
    plates -- the same words as the shell's tree chip."""
    if m.target == "plate":
        return "NACA 00tt at the searched t/c"
    try:
        tc = float(m.session.wing.built().problem.tc)
    except Exception:                                   # noqa: BLE001 -- the words without it
        return "the family's published section"
    return f"NACA 24{int(round(100.0 * tc)):02d}" if finite(tc) else "the family's published section"


def flown_words(m) -> tuple:
    """(name, origin) of what the wing flies on surface `m` now: the section
    taken here, or -- undecided, or kept -- the family's own."""
    c = m.chosen or {}
    if m.decision in ("library", "optimised") and c:
        return str(c.get("name") or "—"), str(c.get("origin") or "")
    return own_section(m), ("the family's own, kept" if m.decision == "default"
                            else "the family's own, undecided")


def designing_row(ui, ctx, subject: str = "") -> None:
    """The row every section view opens with: `designing`, the surface's
    tag (WING SECTION / ENDPLATE), what this view has on screen (`subject`),
    what the wing flies on this surface now and where it came from, and the
    link to the other surface. On the endplate a line states the owner's
    rule: symmetric sections only."""
    m, dp = ctx.model, ctx.dp
    plate = m.target == "plate"
    name, origin = flown_words(m)
    flies = f"the {'plates fly' if plate else 'wing flies'}: {name}" + (f" ({origin})" if origin else "")
    tag = "ENDPLATE" if plate else "WING SECTION"
    if plate:
        link = ("designing.to_af", "the wing's section", "arrow_back",
                lambda: ctx.select("af", "af.screen"))
    else:
        link = ("designing.to_ep", "the endplate's section", "arrow_forward",
                lambda: ctx.select("ep", "ep.screen"))
    link_ok = not (not plate and dp.plate_locked())
    with ui.row():
        ui.label("designing", css=11.5, colour=T.INK_MUTED, min_w=80)
        ui.tag(tag, T.ACCENT)
        if subject:
            ui.label(T.ellipsize(subject, "mono", 12, max(60, ui.w // 3)), css=12, family="mono")
            ui.gap(12)
        ui.label(T.ellipsize(flies, "mono", 12, max(60, ui.w // 2)), css=12, family="mono",
                 colour=T.INK_MUTED)
        ui.spacer()
        if link_ok:
            ui.link(link[0], link[1], link[2], link[3], small=True)
    if plate:
        ui.hint("SYMMETRIC SECTIONS ONLY. The screen offers the library's symmetric sections "
                "(camber ≤ 0.5 % c) and the shape search is WingLab's symmetric CST, the lower "
                "surface the upper mirrored.", "ok", split=False,
                help="The owner's rule, and WingLab's own: an endplate is a vertical panel at zero "
                     "design lift, and a cambered plate is a permanent side load at zero toe. The "
                     "endplate problem refuses a cambered plate polar outright "
                     "(CarWingEndplateProblem._check_symmetric).")


# --------------------------------------------------------------------------- #
#  the section: its outline and its polar                                      #
# --------------------------------------------------------------------------- #
def section_outline_fig(coords, seed_coords=None, *, label="", seed_label="", tc=None,
                        x_tc=None) -> Figure:
    """A section's outline: its closed coordinate loop (AeroBO's, blunt
    trailing edge kept) filled BAND_A with a 2 px ACCENT line, named `{label}
    · t/c {tc:.4f}`; the seed's loop dotted INK_MUTED when it is another
    shape; a dashed WARN line at the maximum thickness (`x_tc`); equal
    aspect. No coordinates: the figure says so."""
    fig = Figure(xlabel="x/c", ylabel="y/c", equal=True,
                 empty_text="no coordinates for this section")
    if coords is None:
        return fig
    c = np.asarray(coords, dtype=float)
    if c.ndim != 2 or c.shape[0] < 3:
        return fig
    name = (label or "section") + (f" · t/c {float(tc):.4f}" if finite(tc) else "")
    if seed_coords is not None:
        s = np.asarray(seed_coords, dtype=float)
        if s.ndim == 2 and s.shape[0] >= 3 and (s.shape != c.shape or float(np.max(np.abs(s - c))) > 1e-9):
            fig.line(s[:, 0], s[:, 1], colour=T.INK_MUTED, width=1.4, dash="dot", closed=True,
                     name=seed_label or "seed")
    fig.line(c[:, 0], c[:, 1], colour=T.ACCENT, width=2, closed=True, fill=T.BAND_A, name=name)
    if finite(x_tc):
        fig.vline(float(x_tc), colour=T.WARN, dash="dash")
        fig.text(float(x_tc), float(np.min(c[:, 1])), f"max t/c @ {float(x_tc) * 100:.0f}%",
                 colour=T.INK_MUTED, size=9, anchor="centre")
    return fig


def polar_arrays(p):
    """(alpha [deg], cl, cd, cm) arrays of a section polar in either of
    AeroBO's two shapes -- a report's dict (`alpha_deg`, `cl`, `cd`, `cm`) or
    a `TablePolar` (`alpha_deg`, `CL`, `CD`, `CM`) -- else None."""
    if p is None:
        return None
    #  a TablePolar's lower-case names are its METHODS (cl(alpha), ...): the
    #  object is read by its upper-case tables first
    keys = ((("alpha_deg", "cl", "cd", "cm"), ("alpha", "cl", "cd", "cm")) if isinstance(p, dict)
            else (("alpha_deg", "CL", "CD", "CM"),))
    for ks in keys:
        try:
            vals = [p[k] for k in ks] if isinstance(p, dict) else [getattr(p, k) for k in ks]
            arrs = [np.asarray(v, dtype=float) for v in vals]
        except (KeyError, AttributeError, TypeError, ValueError):
            continue
        if arrs[0].ndim == 1 and arrs[0].size >= 2 and all(a.shape == arrs[0].shape for a in arrs):
            return tuple(arrs)
    return None


def polar_figs(polar, seed_polar=None, design_cl=None, cm_gate=None, *, names=("section", "seed")) -> list:
    """The three polar panels (AeroBO fig_section_polars): lift curve (α
    [deg] against c_l), drag polar (c_d against c_l), pitching moment (c_l
    against c_m). The section 2 px ACCENT; the seed dotted INK_MUTED; a
    dashed WARN line at the design lift; dotted WARN lines at ±`cm_gate`
    when that gate is on. Titles for `WorkUI.plots`: POLAR_TITLES."""
    figs = [Figure(xlabel="α [deg]", ylabel="c_l", legend=False),
            Figure(xlabel="c_d", ylabel="c_l", legend=False),
            Figure(xlabel="c_l", ylabel="c_m", legend=False)]
    own, seed = polar_arrays(polar), polar_arrays(seed_polar)
    for arr, is_seed in ((seed, True), (own, False)):
        if arr is None:
            continue
        a, cl, cd, cm = arr
        kw = (dict(colour=T.INK_MUTED, width=1.4, dash="dot", name=names[1]) if is_seed
              else dict(colour=T.ACCENT, width=2, name=names[0] if seed is not None else None))
        figs[0].line(a, cl, **kw)
        figs[1].line(cd, cl, **kw)
        figs[2].line(cl, cm, **kw)
    if seed is not None and own is not None:
        for f in figs:
            f.legend = True
    if finite(design_cl) and own is not None:
        figs[0].hline(float(design_cl), colour=T.WARN, dash="dash")
        figs[1].hline(float(design_cl), colour=T.WARN, dash="dash")
        figs[2].vline(float(design_cl), colour=T.WARN, dash="dash")
    if finite(cm_gate) and 0.0 < float(cm_gate) < 1.0:
        figs[2].hline(float(cm_gate), colour=T.WARN, dash="dot")
        figs[2].hline(-float(cm_gate), colour=T.WARN, dash="dot")
    return figs


#: the three panels' titles, in `polar_figs` order
POLAR_TITLES = ("lift curve", "drag polar", "pitching moment")
