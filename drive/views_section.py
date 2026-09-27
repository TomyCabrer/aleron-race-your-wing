"""drive/views_section.py -- the eight views of stages 2 and 2.8, Airfoil and
Endplate (PLAN2 §8.2), on AeroBO's own engine.

A view is `fn(ui, ctx)`: it draws itself top to bottom into the shell's work
area with `ui` (a `cae.widgets.WorkUI`) from what `ctx` (a
`design_shell.ViewCtx`) hands it, and never imports `garage`.

The same four functions serve both surfaces -- `ctx.model` is the session's
`aerobo_models.SurfaceModel`, `dp.af` (target "main", the wing's section) or
`dp.ep` (target "plate", the endplates'), and its `target` carries every
difference:

    screen     Library screening   the design point (its own Re -- a live
                                   XFOIL sweep of the shortlist -- or the
                                   cached library point), the criterion
                                   weights, the hard gates and floors, the
                                   live sweep  (api.screen_at_point /
                                   api.screen_airfoils)
    ranking    Ranking             AeroBO's top 14: composite J, each
                                   criterion and the points it scored; a
                                   click flies the row on the wing
    section    Section             the section on the page: its outline, its
                                   polar at the surface's point, its numbers,
                                   and what the wing flies
    optimise   Shape optimisation  AeroBO's CST + live-XFOIL search
                                   (api.optimize_airfoil, 164 evaluations at
                                   balanced), THE LIVE EVALUATION GRAPH, and
                                   the seed against what it found

Two rules of the owner's run through all of it. THE WING has no "cd at the
design cl" (it is "L/D at the design cl" at one stated lift): no weight, no
column, no comparison row -- `SurfaceModel.redundant()`. THE ENDPLATE is
symmetric: it screens only the library's symmetric sections and its shape
search is AeroBO's symmetric CST, and both are stated on the page.

Every section view opens with the "designing" row. A run's state is drawn
in one of two looks (views_common): live -- spinner, RUNNING · k/N, the bar,
the graph growing -- or finished, from the model's STORED outcome.
"""

from __future__ import annotations

import math

from . import aerobo_models as am
from . import views_common as vc
from .cae import theme as T

#: AeroBO's order of the criteria (its Criterion weights card)
WEIGHT_ORDER = ("ldcr", "clmax", "cm", "ldmax", "thick", "astall", "cdcr")

#: the criteria a user misreads, whose slider carries a "?"
WEIGHT_HELP = {"ldmax": "the best L/D anywhere on the polar — not where the wing flies",
               "ldcr": "L/D at THIS surface's design cl, the lift it will fly at",
               "cdcr": "the drag at the design cl — on a plate at cl 0 its drag is what is left",
               "astall": "the stall angle of the pre-stall branch"}

#: the ranking shows AeroBO's `top_n` rows
RANK_TOP_N = 14

#: `ranked` row metrics a criterion is read from (aerobo_models.CRITERION_METRIC)
METRIC = am.CRITERION_METRIC

#: a section's numbers, as the Section view and the comparison read them:
#: (criterion key, label, format, higher is better)
NUMBERS = (("thick", "t/c", "{:.4f}", True),
           ("clmax", "cl max", "{:.3f}", True),
           ("astall", "stall angle [deg]", "{:.1f}", True),
           ("ldmax", "(L/D) max", "{:.1f}", True),
           ("ldcr", "L/D at the design cl", "{:.1f}", True),
           ("cm", "|cm| at the design cl", "{:.4f}", False),
           ("cdcr", "cd at the design cl", "{:.5f}", False))

_SYM: list = []                          # the symmetric names' count, read once


# --------------------------------------------------------------------------- #
#  small readers                                                               #
# --------------------------------------------------------------------------- #
def _plate(ctx) -> bool:
    return getattr(ctx.model, "target", "main") == "plate"


def _kind(job) -> str:
    return getattr(getattr(job, "info", None), "kind", "") if job is not None else ""


def _live(ctx, *kinds):
    """This surface's live job when it is one of `kinds`, else None."""
    job = ctx.job if ctx.live else None
    return job if job is not None and _kind(job) in kinds else None


def _n_symmetric() -> int | None:
    """How many of AeroBO's library sections are symmetric (camber <= 0.5 %
    c, `api.symmetric_section_names`), read once."""
    if not _SYM:
        try:
            _SYM.append(len(am.bridge.api.symmetric_section_names()))
        except Exception:                               # noqa: BLE001
            _SYM.append(None)
    return _SYM[0]


def _camber(coords) -> float | None:
    """The max |mean line| of a section loop, fraction of chord (AeroBO's
    own measure, `api.section_max_camber`)."""
    if coords is None:
        return None
    try:
        v = float(am.bridge.api.section_max_camber(coords))
    except Exception:                                   # noqa: BLE001
        return None
    return v if math.isfinite(v) else None


def _crit_value(metrics: dict, k: str):
    """A criterion's raw value in a ranked row's metrics (|cm| as a size)."""
    v = (metrics or {}).get(METRIC.get(k, k))
    if not vc.finite(v):
        return None
    return abs(float(v)) if k == "cm" else float(v)


def _breakdown_value(bd: dict, k: str):
    """A criterion's raw value in an optimised section's breakdown (AeroBO's
    `evaluate` keys: tc, clmax, astall, ldmax, ldcr, cm, cd_at)."""
    key = {"thick": "tc", "cm": "cm", "cdcr": "cd_at"}.get(k, k)
    v = (bd or {}).get(key)
    if not vc.finite(v):
        return None
    return abs(float(v)) if k == "cm" else float(v)


def _shown(m) -> list:
    """The NUMBERS this surface shows (the wing has no cd@cl: the owner's D9)."""
    return [n for n in NUMBERS if n[0] not in m.redundant()]


# --------------------------------------------------------------------------- #
#  Library screening                                                           #
# --------------------------------------------------------------------------- #
def _design_point_card(ui, ctx) -> None:
    """Where this surface is screened: its own Reynolds number (a live
    XFOIL sweep of the shortlist) or the cached library point (instant),
    and the lift; on the plate, the symmetric-only library."""
    m = ctx.model
    c = m.conditions()
    with ui.card(f"Design point — the {'endplate' if _plate(ctx) else 'wing section'}"):
        if not c:
            ui.hint("this wing has no such surface to design: carried by pylons, its plates are a tip "
                    "device carrying the wing's own section", "warn")
            return
        ui.field("resrc", control="toggle", label="screened at",
                 labels={"mission": "its own Re", "library": "the library point"})
        src = c.get("re_source", "mission")
        ui.kpis([dict(label="screen at Re", value=f"{float(c['re']):.3g}",
                      unit="own" if src == "mission" else "library"),
                 dict(label="design cl", value=f"{float(c['cl_design']):.2f}"),
                 dict(label="chord", value=vc.num(c.get("chord"), "{:.3f}"), unit="m"),
                 dict(label="its own Re", value=vc.num(c.get("re_own"), "{:.3g}")),
                 dict(label="V", value=f"{float(m.session.op.V):.1f}", unit="m/s")])
        if src == "mission":
            ui.hint(f"Its own Re: WingLab ranks the whole library at the cached point, then sweeps "
                    f"the {m.shortlist} leaders LIVE in XFOIL at Re {float(c['re']):.3g} — about 20 s "
                    f"cold, instant once swept.", split=False)
        elif m.re_source == "mission":
            ui.hint("XFOIL is not available, so only the cached library point can be screened "
                    "(instant, exact at any cl).", "warn")
        else:
            ui.hint("The cached library point: every section already has a polar there, so the "
                    "screen is instant and exact at any cl — at a Reynolds number this surface "
                    "does not fly.")
        if _plate(ctx):
            n = _n_symmetric()
            ui.hint(f"Symmetric sections only — {n if n is not None else 'the'} of WingLab's 2174 "
                    f"(camber ≤ 0.5 % c). The plate is designed at cl 0: its panels are vertical "
                    f"and carry no design lift.", "ok")


def _weights_card(ui, ctx) -> None:
    m = ctx.model
    dead = m.dead()
    with ui.card("Criterion weights"):
        with ui.row():
            ui.label("set", css=T.HINT_CSS, colour=T.INK_MUTED)
            ui.tag("WingLab's recommended" if m.weights_source == "recommended" else "yours",
                   T.GOOD if m.weights_source == "recommended" else T.WARN)
        for k in WEIGHT_ORDER:
            #  THE OWNER'S CALL (2026-09-24, handoff D9): on the wing "cd at the
            #  design cl" is "L/D at the design cl" at one stated lift, so it is
            #  not offered -- the form has no row for it. Do not restore it.
            #  The plate keeps it (its drag at cl 0 is its criterion).
            if k in m.redundant() or ui.form.param(f"w.{k}") is None:
                continue
            ui.slider(f"w.{k}", help=dead.get(k) or WEIGHT_HELP.get(k, ""), dim=k in dead)
        ui.hint("Weights are normalised before scoring, so only their ratios matter.")
        if m.weights_source != "recommended":
            ui.button("rec", "Use WingLab's recommended weights", kind="flat", icon="tune")
        if dead:
            ui.hint(f"Dim: {', '.join(sorted(dead))} cannot rank a plate — at cl 0 they are read at "
                    f"a lift it never carries (held at 0).")
        if _plate(ctx):
            ui.hint(f"The plate's t/c weight stays WingLab's own "
                    f"({float(m.weights.get('thick', 0.0)):.2f}): not an option.")
        if not _plate(ctx):
            ui.hint("No “cd at the design cl” here: on the wing it is L/D at the design cl again "
                    "(one stated lift), so it would count one question twice.")


def _gate_row(ui, switch, number, label) -> None:
    """One gate: the switch, its label, the number. Nothing where the form
    has no such row (the plate's t/c: the owner, 2026-09-25, "endplate t/c
    shouldn't be given as an option")."""
    if ui.form.param(switch) is None:
        return
    with ui.row():
        ui.switch(switch, label="")
        ui.label(label, min_w=100)
        ui.number(number)


def _gates_cards(ui, ctx) -> None:
    with ui.card("Hard gates"):
        _gate_row(ui, "gtc", "tcmin", "min t/c")
        _gate_row(ui, "gcm", "cmmax", "max |cm|")
        ui.hint("A gate DROPS a section; the weights only order the ones that pass. WingLab's screen "
                "gates: t/c ≥ 0.15, |cm| ≤ 0.08.")
        if _plate(ctx):
            ui.hint("The plate's t/c is not an option: WingLab's own gate stays on.")
    with ui.card("Floors (optional)"):
        for key in ("fclmax", "fldcr", "fastall"):
            if ui.form.param(key) is not None:
                ui.field(key, control="number")
        ui.hint("At the bottom of its range a floor is off. A floor is a minimum on a "
                "higher-is-better metric.")


def _screen_summary(ui, ctx) -> None:
    """What the screen that stands produced, in AeroBO's own counts, and
    what failed or was stopped."""
    m = ctx.model
    out = m.screen.get("outcome") or {}
    if out.get("state") == "error":
        ui.hint(f"The screen failed — {out.get('error') or 'no reason given'}. "
                f"{'The previous ranking stands.' if m.ranked else ''}", "bad")
    elif out.get("state") == "stopped" and not m.ranked:
        ui.hint("The screen was stopped before it ranked anything.", "warn")
    rep = m.screen.get("report") or {}
    if not m.ranked:
        if m.msg and not out:
            ui.hint(m.msg)
        return
    st = rep.get("status_counts") or {}
    bits = [f"{len(m.ranked)} ranked", f"{rep.get('n_eligible', '?')} eligible",
            f"{rep.get('n_screened', '?')} screened"]
    if st:
        bits.append(" · ".join(f"{v} {k.replace('_', ' ')}" for k, v in st.items()))
    ui.hint(" · ".join(bits) + f" — best {m.ranked[0]['name']}", "ok")
    note = (rep.get("shortlist") or {}).get("note")
    if note:
        ui.hint(str(note))


def screen(ui, ctx) -> None:
    """Library screening: the design point, the action row (Screen, or Stop
    while it runs, and the run's tag) with -- while it runs -- the live
    sweep right under it, where a 1280x800 window still shows it, then the
    criterion weights beside the gates and floors."""
    vc.designing_row(ui, ctx)
    _design_point_card(ui, ctx)
    _screen_action(ui, ctx)
    left, right = ui.columns((3, 2))
    with left:
        _weights_card(ui, ctx)
    with right:
        _gates_cards(ui, ctx)


def _screen_action(ui, ctx) -> None:
    """Screen / Stop, the plate's "keep the family's own", the tag; then the
    live sweep, or what the screen that stands produced."""
    m = ctx.model
    job = _live(ctx, "screen")
    with ui.row():
        if job is not None:
            ui.button("stop", "Stop the screen", kind="primary", icon="stop")
        else:
            ui.button("go", "Screen the library", kind="primary", icon="search", help="L does the same")
        if _plate(ctx) and ui.form.param("decline") is not None:
            ui.button("decline", "Keep the family's own plate", kind="outline", icon="link_off")
        ui.spacer()
        vc.draw_tag(ui, _screen_tag(m, job),
                    wall=None if job else (m.screen.get("outcome") or {}).get("wall"))
    if job is not None:
        vc.sweep_block(ui, job)
        return
    _screen_summary(ui, ctx)


def _screen_tag(m, job):
    """The screen's tag: RUNNING · i/n while it sweeps; after, DONE with
    AeroBO's own count of what was screened -- a screen at the library point
    is ONE read of the warm checkpoint, and "1/1" would say nothing -- or the
    stored STOPPED / FAILED tag."""
    if job is not None:
        return vc.tag_of(job)
    out = m.screen.get("outcome")
    rep = m.screen.get("report") or {}
    if (out or {}).get("state") == "done" and rep.get("n_screened") is not None:
        return (f"DONE · {rep['n_screened']} screened", T.GOOD, "")
    return vc.tag_of(None, out)


# --------------------------------------------------------------------------- #
#  Ranking                                                                     #
# --------------------------------------------------------------------------- #
RANK_TIPS = {
    "eligible": "How many sections cleared every GATE and floor, out of how many were screened. A "
                "gate DROPS a section; the weights only order what is left.",
    "re": "The Reynolds number the ranked polars were read at: this surface's own (a live XFOIL "
          "sweep of the shortlist) or the cached library point.",
    "cl": "The lift coefficient every lift-dependent criterion was read at: this surface's "
          "design cl (the wing's reference CZ, 0 on a plate).",
    "wall": "What the screen cost on the worker thread (WingLab's own wall time).",
}

RANK_HELP = ("J is WingLab's composite: each criterion scored 0-100 on the frozen band of the "
             "library, weighted, summed. The faint number after a value is the points it put into "
             "J; a row's points add up to its J. \"lib #\" is the row's rank at the cached library "
             "point, before the live sweep at this surface's own Re re-ranked it.")


def _rank_rows(m, rep) -> tuple:
    """(columns, rows, dim) of the ranking table: rank, name, J, each
    criterion this surface carries with its points, t/c, and the library
    rank when the screen was at the surface's own Re."""
    w = {k: float(v) for k, v in (rep.get("weights") or m.weights).items()}
    tot = sum(v for v in w.values() if v > 0.0) or 1.0
    lib = any(r.get("rank_library") for r in m.ranked)
    cols = [dict(key="rank", head="#", fmt=lambda v: str(v)),
            dict(key="name", head="section", align="left"),
            dict(key="J", head="J", fmt=lambda v: vc.num(v, "{:.2f}"))]
    dim = []
    for k, head, fmt in m.columns():
        wk = w.get(k, 0.0)
        cols.append(dict(key=k, head=f"{head} · w {wk / tot:.2f}",
                         fmt=(lambda v, f=fmt: f.format(v) if vc.finite(v) else "—")))
        if wk <= 0.0:
            dim.append(k)
    if lib:
        cols.append(dict(key="lib", head="lib #", fmt=lambda v: "—" if v is None else f"{int(v)}"))
    rows = []
    for r in m.ranked:
        row = dict(rank=r["rank"], name=r["name"], J=r.get("composite"), _sub={})
        sc = r.get("scores") or {}
        for k, _h, _f in m.columns():
            row[k] = _crit_value(r.get("metrics"), k)
            if w.get(k, 0.0) > 0.0 and vc.finite(sc.get(k)):
                row["_sub"][k] = f"{w[k] / tot * float(sc[k]):+.1f}"
        if lib:
            row["lib"] = r.get("rank_library")
        rows.append(row)
    return cols, rows, dim


def _stale_hints(ui, ctx, rep) -> None:
    """The ranking was produced by a screen whose question has since moved:
    say so, beside a re-screen."""
    m = ctx.model
    was = rep.get("weights") or {}
    tot_now = sum(max(0.0, float(v)) for v in m.weights.values()) or 1.0
    tot_was = sum(max(0.0, float(v)) for v in was.values()) or 1.0
    moved = bool(was) and any(abs(float(m.weights.get(k, 0.0)) / tot_now - float(was.get(k, 0.0)) / tot_was)
                              > 1e-9 for k in m.weights)
    cond, now = m.screen.get("cond") or {}, m.conditions()
    point = bool(cond and now) and (abs(float(cond.get("re", 0.0)) - float(now.get("re", 0.0)))
                                    > 1e-6 * max(1.0, float(now.get("re", 1.0)))
                                    or abs(float(cond.get("cl_design", 0.0)) - float(now.get("cl_design", 0.0)))
                                    > 1e-9)
    if moved:
        ui.hint("These weights were edited after the screen: the order below is the one the screen "
                "produced — screen again to re-rank.", "warn")
    if point:
        ui.hint(f"Screened at Re {float(cond.get('re', 0.0)):.3g}, cl {float(cond.get('cl_design', 0.0)):.2f}; "
                f"this surface's design point is now Re {float(now.get('re', 0.0)):.3g}, cl "
                f"{float(now.get('cl_design', 0.0)):.2f} — screen again.", "warn")


def ranking(ui, ctx) -> None:
    """Ranking: the screen's counts, the stale warnings, AeroBO's top 14,
    and what the wing flies from it."""
    m, dp = ctx.model, ctx.dp
    vc.designing_row(ui, ctx)
    if not m.ranked:
        ui.hint("Nothing screened yet — run the screening on the previous tab (L).")
        return
    rep = m.screen.get("report") or {}
    #  what AeroBO actually screened at is the REPORT's point
    cond = rep.get("conditions") or m.screen.get("cond") or {}
    src = m.screen.get("re_source") or "library"
    ui.kpis([dict(label="eligible", value=f"{rep.get('n_eligible', '?')}",
                  unit=f"of {rep.get('n_screened', '?')}", tip=RANK_TIPS["eligible"]),
             dict(label="Re", value=vc.num(cond.get("re"), "{:.3g}"),
                  unit="own" if src == "mission" else "library", tip=RANK_TIPS["re"]),
             dict(label="design cl", value=vc.num(cond.get("cl_design"), "{:.2f}"), tip=RANK_TIPS["cl"]),
             dict(label="wall", value=vc.num(rep.get("wall_time_s"), "{:.2f}"), unit="s",
                  tip=RANK_TIPS["wall"])])
    live = ctx.runs.live
    if live is not None and live.owner is m and _kind(live) == "screen":
        ui.hint("A new screen is running: the table below is the last one's.", "warn")
    elif live is not None and (live.owner is m or live.owner is dp.wing):
        ui.hint(f"A run that flies this section is live ({live.info.evaluator}): a row cannot be "
                f"taken until it ends.", "warn")
    _stale_hints(ui, ctx, rep)
    cols, rows, dim = _rank_rows(m, rep)
    taken = None
    c = m.chosen or {}
    if m.decision == "library":
        taken = next((i for i, r in enumerate(m.ranked) if r["name"] == c.get("name")), None)
    take = getattr(ctx.shell, "take_rank", None)
    ui.table("rank", cols, rows, max_rows=RANK_TOP_N, selected=taken, cursor=dp.rank_list.idx,
             on_row=take, dim_cols=dim)
    ui.hint(f"Click a row (or ENTER / F) to fly it on the {'plates' if _plate(ctx) else 'wing'}; the "
            f"Section tab shows its outline and polar.", help=RANK_HELP)
    name, origin = vc.flown_words(m)
    ui.kv("the wing flies" if not _plate(ctx) else "the plates fly", f"{name}" + (f" — {origin}" if origin else ""))
    ui.kv("the polars", _point_words(rep))
    note = (rep.get("shortlist") or {}).get("note")
    if note:
        ui.hint(str(note))


def _point_words(rep) -> str:
    """Where the ranked polars came from (the report's `point` block) and
    what the screen could not read (`status_counts`), in AeroBO's counts."""
    pt = rep.get("point") or {}
    if pt.get("matched"):
        where = (f"the cached library point — {pt.get('from_sidecar', 0)} from the branch sidecar, "
                 f"{pt.get('from_record', 0)} from the checkpoint")
    else:
        where = f"re-derived at the requested point for {pt.get('rederived', '?')} sections"
    if pt and not pt.get("honoured", True):
        where += " (the requested point NOT honoured)"
    st = {k: v for k, v in (rep.get("status_counts") or {}).items() if k != "ok"}
    if st:
        where += " · " + ", ".join(f"{v} {vc.SWEEP_STATUS.get(k, k.replace('_', ' '))}" for k, v in st.items())
    return where


# --------------------------------------------------------------------------- #
#  Section                                                                     #
# --------------------------------------------------------------------------- #
def _subject(m) -> dict | None:
    """The section the Section view shows: the one taken (library or
    optimised), else the highlighted ranked row not yet taken, else None.
    {name, coords, tc, metrics(fn), source, rank, conditions, polar_key,
    taken, origin, optimised}."""
    c = m.chosen or {}
    if m.decision == "optimised" and c.get("coords"):
        rep = m.opt.get("report") or {}
        bd = (rep.get("design") or {}).get("breakdown") or {}
        return dict(name=c.get("name"), coords=c.get("coords"), tc=c.get("tc"),
                    value=lambda k: _breakdown_value(bd, k), source="optimised", rank=None,
                    conditions=c.get("conditions") or {}, taken=True, origin=c.get("origin"),
                    polar=((rep.get("section") or {}).get("design") or {}).get("polar"))
    if m.decision == "library" and c.get("name"):
        r = next((r for r in m.ranked if r["name"] == c.get("name")), None)
        return dict(name=c["name"], coords=c.get("coords"), tc=c.get("tc"),
                    value=(lambda k, rr=r: _crit_value(rr.get("metrics"), k)) if r else (lambda k: None),
                    source="library", rank=r["rank"] if r else None,
                    conditions=c.get("conditions") or {}, taken=True, origin=c.get("origin"),
                    library_point=bool(c.get("library_point")), polar=None)
    if m.ranked:
        i = int(min(max(int(m.highlight), 0), len(m.ranked) - 1))
        r = m.ranked[i]
        cond = m.screen.get("cond") or {}
        return dict(name=r["name"], coords=r.get("coords"), tc=r.get("tc"),
                    value=lambda k, rr=r: _crit_value(rr.get("metrics"), k), source="library",
                    rank=r["rank"], conditions={"re": cond.get("re"), "mach": cond.get("mach", 0.0)},
                    taken=False, origin=None,
                    library_point=m.screen.get("re_source") == "library", polar=None)
    return None


def _section_kpis(ui, ctx, sub) -> None:
    m = ctx.model
    tiles = [dict(label="section", value=str(sub["name"])),
             dict(label="source", value=("optimised" if sub["source"] == "optimised"
                                         else f"library · rank {sub['rank']}" if sub.get("rank")
                                         else "library"))]
    for k, label, fmt, _hib in _shown(m):
        v = sub["value"](k)
        if k in m.dead() or v is None:
            continue
        tiles.append(dict(label=label, value=fmt.format(v)))
    cam = _camber(sub.get("coords"))
    if cam is not None:
        sym = cam <= 0.005
        tiles.append(dict(label="max camber", value=f"{cam * 100:.2f}", unit="% c",
                          colour=(T.GOOD if sym else (T.BAD if _plate(ctx) else None))))
    ui.kpis(tiles)
    if _plate(ctx) and cam is not None:
        if cam <= 0.005:
            ui.hint(f"Symmetric: max camber {cam * 100:.2f} % c ≤ 0.5 % c — a plate at zero toe makes "
                    f"no side force.", "ok")
        else:
            ui.hint(f"Cambered ({cam * 100:.2f} % c): the endplate flies symmetric sections only.", "bad")


def _polar_block(ui, ctx, sub) -> None:
    """The section's polar at the point the wing will fly it, from what is
    at hand (`SurfaceModel.polar`: the optimised section's own sweep, the
    screening branch at the library point, or AeroBO's XFOIL cache at the
    surface's own Re) -- or a button that sweeps it as a job
    (`start_polar`: seconds of XFOIL, never a frame)."""
    m = ctx.model
    polar = m.polar()
    job = _live(ctx, "polar")
    re = (polar or {}).get("re") or (sub.get("conditions") or {}).get("re")
    design_cl = None if _plate(ctx) else (m.conditions() or {}).get("cl_design")
    title = "Polar — " + (f"Re {float(re):.3g}" if vc.finite(re) else "the library point")
    with ui.card(title, pad=False):
        if vc.polar_arrays(polar) is not None:
            ui.plots(vc.polar_figs(polar, design_cl=design_cl, cm_gate=m.gates("screen").get("cm_max")),
                     280, titles=vc.POLAR_TITLES)
        else:
            ui.empty_plot("sweeping its polar in XFOIL…" if job is not None else
                          f"its polar at Re {vc.num(re, '{:.3g}')} is not in WingLab's XFOIL cache yet", 200)
    if polar is None and job is None:
        ui.button("ui.sweep_polar", f"Sweep its polar at Re {vc.num(re, '{:.3g}')} (XFOIL, seconds)",
                  kind="outline", icon="refresh", on_click=lambda: m.start_polar())
        if str(m.msg or "").startswith(("this section's polar", "no polar")):
            ui.hint(m.msg, "warn")
    src = (polar or {}).get("source")
    ui.hint(f"XFOIL, WingLab's: {src}." if src else
            "XFOIL polars, WingLab's: the library point's come from the screening branch sidecar, "
            "a surface's own Re from a live sweep (cached once paid).")


def section(ui, ctx) -> None:
    """Section: what the wing flies on this surface, or the ranked row on
    the page -- its outline, its numbers, its polar at the surface's point,
    and the three answers: fly it, keep the family's own, refine it."""
    m = ctx.model
    sub = _subject(m)
    vc.designing_row(ui, ctx, subject=f"on the page: {sub['name']}" if sub else "")
    if sub is None:
        if m.decision == "default":
            name, _o = vc.flown_words(m)
            ui.hint(f"Kept: {name}. Screen the library to give the "
                    f"{'plates' if _plate(ctx) else 'wing'} a section of its own.", "ok")
        elif _plate(ctx):
            ui.hint("The plates fly NACA 00tt at the t/c the wing search picks (the family's own) "
                    "until a section is chosen here. Screen the symmetric library, or keep it.")
        else:
            ui.hint(f"The wing flies the family's own section ({vc.own_section(m)}) until one is "
                    f"chosen here — screen the library, then take a row of the ranking.")
        with ui.row():
            ui.button("decline", kind="outline", icon="check")
        return
    with ui.row():
        ui.tag("ON THE WING" if sub["taken"] else "NOT TAKEN YET", T.GOOD if sub["taken"] else T.WARN)
        if sub.get("origin"):
            ui.label(str(sub["origin"]), css=T.HINT_CSS, colour=T.INK_MUTED)
    _section_kpis(ui, ctx, sub)
    with ui.card("Outline", pad=False):
        ui.plot(vc.section_outline_fig(sub.get("coords"), label=str(sub["name"]), tc=sub.get("tc")), 240)
    _polar_block(ui, ctx, sub)
    with ui.card("What the wing flies"):
        if sub["taken"]:
            ui.hint(f"The {'plates fly' if _plate(ctx) else 'wing flies'} {sub['name']} from now on: "
                    f"its coordinates travel to WingLab's wing as the section flag, at the point it was "
                    f"chosen at.", "ok")
        else:
            ui.hint(f"Taking it makes {sub['name']} the {'plates’' if _plate(ctx) else 'wing’s'} section: "
                    f"WingLab's wing flies its polar at Re {vc.num((sub.get('conditions') or {}).get('re'), '{:.3g}')}.")
        with ui.row():
            if not sub["taken"]:
                ui.button("use", "Use this section", kind="primary", icon="check", help="F does the same")
            ui.button("refine", "Refine this shape", kind="outline", icon="auto_graph")
            ui.button("decline", kind="flat", icon="undo")


# --------------------------------------------------------------------------- #
#  Shape optimisation                                                          #
# --------------------------------------------------------------------------- #
#: the section objectives, short, for the graph's head (the select carries
#: AeroBO's full words, `aerobo_models.SECTION_OBJECTIVE_WORDS`)
OBJECTIVE_SHORT = {"composite_goal": "composite J, held to the seed",
                   "composite": "composite J", "composite_asf": "composite, worst-first",
                   "cd": "2-D L/D at the design cl"}


def _cst_labels(n) -> list:
    """The design vector's rows: AeroBO's CST weights -- four upper and four
    lower on the wing; four upper on the symmetric plate (lower = −upper)."""
    if n == 4:
        return [f"w_upper_{i}" for i in range(4)]
    half = n // 2
    return [f"w_upper_{i}" for i in range(half)] + [f"w_lower_{i}" for i in range(n - half)]


def _what_card(ui, ctx) -> None:
    """What the search is: AeroBO's CST section, the evaluator, the point,
    the seed."""
    m = ctx.model
    c = m.conditions()
    name, anchor = m.anchor()
    with ui.card("What this does"):
        if _plate(ctx):
            ui.hint("WingLab's SYMMETRIC CST section: four upper-surface weights, the lower surface "
                    "the upper mirrored (w_lower = −w_upper) — the plate cannot come out cambered.",
                    "ok")
        else:
            ui.hint("WingLab's CST section: four upper-surface and four lower-surface weights (8 "
                    "variables), trailing edge closed.")
        if c:
            ui.kv("each candidate", f"a live XFOIL sweep at Re {float(c['re_own']):.3g}, scored at cl "
                                    f"{float(c['cl_design']):.2f}")
        ui.kv("seed", f"{name} (its CST refit)" if name else "the family's NACA anchor (nothing chosen)")
        ui.kv("its own gates", "|cm| below; t/c held at WingLab's ≥ 0.10 — not an option on a plate"
              if _plate(ctx) else
              "t/c and |cm| below — WingLab's optimiser gates (t/c ≥ 0.10, |cm| ≤ 0.08)")


def _objective_card(ui, ctx) -> None:
    m = ctx.model
    with ui.card("Objective"):
        ui.field("obj", control="select", label="maximise")
        ui.hint(am.SECTION_OBJECTIVE_WORDS.get(m.opt["objective"], ""))
        _gate_row(ui, "gotc", "otc", "min t/c")
        _gate_row(ui, "gocm", "ocm", "max |cm|")
        if _plate(ctx):
            ui.hint("The plate's t/c is not an option: WingLab's own gate stays on.")
        if m.opt["objective"].startswith("composite"):
            ui.hint("Scored on this surface's criterion weights (2 ▸ Library screening), on the "
                    "frozen band the screen measured.")


def _search_card(ui, ctx) -> None:
    """The budget AeroBO's plan gives (164 at balanced), the optimiser, its
    Sobol start and refusal rule, the expected time."""
    m, s = ctx.model, ctx.model.session
    try:
        eff = s.policy.section_search(m.opt["objective"], m.target, int(m.opt["seed"]))
    except Exception as exc:                            # noqa: BLE001
        eff = {"error": str(exc)}
    plan = eff.get("plan")
    rec = s.policy.recommended()
    with ui.card("Search · WingLab's recommendation" if rec else "Search · your own budget"):
        if eff.get("error"):
            ui.hint(f"no plan: {eff['error']}", "bad")
            return
        n = int(eff.get("budget") or 0)
        per = getattr(plan, "per_eval_s", None)
        t = vc.duration(n * float(per)) if vc.finite(per) else "—"
        ui.kpis([dict(label="budget", value=f"{n}", unit="evaluations", colour=T.ACCENT),
                 dict(label="optimiser", value=str(eff.get("optimiser", "—"))),
                 dict(label="Sobol start", value=vc.num(eff.get("n_init"), "{:.0f}")),
                 dict(label="expected", value=t,
                      tip="WingLab's measured cost per candidate × the budget (the study's machine)")])
        if rec:
            ui.hint(f"{n} evaluations of a live XFOIL sweep — WingLab's measured {s.policy.effort} "
                    f"budget for this 8-D search ({t} at {vc.num(per, '{:.1f}')} s each).",
                    split=False, help=getattr(plan, "why", None) or None)
        ui.field("budget", control="number", unit="evaluations")
        ui.field("seed", control="number", label="random seed")
        ui.kv("refusal", f"{eff.get('refusal') or '—'} — a refused shape is shown to the surrogate as "
                         f"the worst one that flew")
        note = m.bo_note()
        if note:
            ui.hint(note, "warn")


def _action_row(ui, ctx, job) -> None:
    """Run, or Stop while it runs (the button turns into Stop); Keep going
    and Use this section once a search has finished; the run's tag."""
    m = ctx.model
    why = m.refusal()
    with ui.row():
        if job is not None:
            ui.button("stop", "Stop — keeps the best so far", kind="primary", icon="stop")
        else:
            ui.button("run", "Optimise the section", kind="primary", icon="auto_graph",
                      help="O does the same")
            if m.can_continue():
                ui.button("go", kind="outline", icon="play_arrow",
                          help="WingLab's resume: the evaluations already paid for are the new "
                               "search's training set, nothing is re-flown (K)")
            if m.has_optimised() and m.decision != "optimised":
                ui.button("use", "Use this section", kind="outline", icon="check", help="F does the same")
        ui.spacer()
        if m.decision == "optimised" and job is None:
            ui.tag("ON THE WING", T.GOOD)
    if job is None and m.can_continue():
        ui.field("more", control="number", unit="evaluations")
    if why and job is None:
        ui.hint(why, "warn")
    msg = str(m.msg or "")
    if msg.startswith("the shape search failed"):
        ui.hint(msg, "bad")


def _compare_rows(ctx) -> list:
    """Seed vs optimised on this surface's numbers, as AeroBO's run scored
    them: the report's `baseline` (the seed, evaluated at the run's point)
    against its `design` (what the search returned) -- or, when the post-run
    score job has landed, `score_optimised_section`'s rows. The wing has no
    cd@cl row (the owner's D9); a plate's dead criteria are left out."""
    m = ctx.model
    rep = m.opt.get("report") or {}
    seed_bd = (rep.get("baseline") or {}).get("breakdown") or {}
    opt_bd = (rep.get("design") or {}).get("breakdown") or {}
    score = m.opt.get("score") or {}
    s_row, o_row = score.get("seed") or {}, score.get("optimised") or {}
    rows = []
    for k, label, fmt, hib in _shown(m):
        if k in m.dead():
            continue
        if s_row and o_row:
            a, b = _crit_value(s_row.get("metrics"), k), _crit_value(o_row.get("metrics"), k)
        else:
            a, b = _breakdown_value(seed_bd, k), _breakdown_value(opt_bd, k)
        rows.append((label, a, b, hib, fmt))
    ja = s_row.get("composite") if s_row else seed_bd.get("composite")
    jb = o_row.get("composite") if o_row else opt_bd.get("composite")
    rows.append(("composite J (this surface's weights)", ja if vc.finite(ja) else None,
                 jb if vc.finite(jb) else None, True, "{:.2f}"))
    return rows


def _compare_table(ui, ctx) -> None:
    rows = []
    for label, a, b, hib, f in _compare_rows(ctx):
        if a is None and b is None:
            continue
        cell = {"metric": label, "seed": f.format(a) if a is not None else "—",
                "optimised": f.format(b) if b is not None else "—", "change": "—", "_colours": {}}
        if a is not None and b is not None:
            d = float(b) - float(a)
            pct = f"  ({100.0 * d / abs(a):+.1f} %)" if abs(a) > 1e-12 else ""
            cell["change"] = f.replace("{:", "{:+", 1).format(d) + pct
            if abs(d) > 1e-12:
                good = (d > 0.0) == bool(hib)
                colour = T.GOOD if good else T.BAD
                cell["_colours"] = {"optimised": colour, "change": colour}
        rows.append(cell)
    ui.table("opt.compare", [dict(key="metric", head="metric", align="left"),
                             dict(key="seed", head="seed"), dict(key="optimised", head="optimised"),
                             dict(key="change", head="change")], rows, sortable=False)


def _seed_coords(m):
    """The seed's own outline: the section the search was anchored on
    (the chosen one, else the ranked row by name), or None."""
    name = m.opt.get("anchor_name")
    if not name:
        return None, ""
    c = m.chosen or {}
    if c.get("name") == name and c.get("coords") is not None:
        return c["coords"], name
    for r in m.ranked:
        if r["name"] == name and r.get("coords") is not None:
            return r["coords"], name
    return None, ""


def _result_cards(ui, ctx) -> None:
    """After a search: AeroBO's numbers for it (best score and its units,
    evaluations, feasible ones, wall, how it ended), seed vs optimised, the
    optimised outline over the seed's and its polar -- and the answer."""
    m = ctx.model
    rep = m.opt.get("report") or {}
    res = rep.get("result") or {}
    design = (rep.get("section") or {}).get("design") or {}
    out = m.opt.get("outcome") or {}
    with ui.card("Optimised section"):
        ui.kpis([dict(label="best score", value=vc.num(res.get("best_score"), "{:.3f}"),
                      unit=str(res.get("score_units") or "")),
                 dict(label="evaluations", value=f"{res.get('n_evals', '—')}",
                      unit=f"of {res.get('budget', '—')}"),
                 dict(label="feasible", value=f"{res.get('n_feasible', '—')}"),
                 dict(label="t/c", value=vc.num(design.get("tc"), "{:.4f}")),
                 dict(label="wall", value=vc.num(rep.get("wall_time_s") or out.get("wall"), "{:.1f}"),
                      unit="s")])
        if res.get("partial"):
            ui.hint(f"PARTIAL — {res.get('stop_reason') or 'ended early'} after {res.get('n_evals')} of "
                    f"{res.get('budget')} evaluations; the best so far is kept.", "warn")
        if res.get("resumed"):
            ui.hint(f"A continuation: it inherited {res['resumed']} evaluations as its training set "
                    f"and flew the rest — nothing was re-flown.")
        _compare_table(ui, ctx)
        ui.hint("SEED is the section the search was anchored on, evaluated by WingLab at the same "
                "Reynolds number and design lift.")
        if (m.opt.get("score_outcome") or {}).get("state") == "error":
            ui.hint(f"the post-run score failed: {(m.opt['score_outcome'] or {}).get('error')}", "warn")
    seed, seed_name = _seed_coords(m)
    cam = _camber(design.get("coords"))
    with ui.card("Optimised outline" + (f" over the seed ({seed_name})" if seed is not None else ""), pad=False):
        ui.plot(vc.section_outline_fig(design.get("coords"), seed, label="optimised", seed_label=seed_name,
                                       tc=design.get("tc"), x_tc=design.get("tc_max_xc")), 240)
    if cam is not None:
        ui.hint(f"max camber {cam * 100:.3f} % c" + (" — symmetric, as the plate must be" if _plate(ctx) else ""),
                "ok" if _plate(ctx) and cam <= 0.005 else None)
    design_cl = None if _plate(ctx) else (rep.get("conditions") or {}).get("cl_design")
    with ui.card(f"Polar — Re {vc.num((rep.get('conditions') or {}).get('re'), '{:.3g}')}", pad=False):
        if vc.polar_arrays(design.get("polar")) is not None:
            ui.plots(vc.polar_figs(design.get("polar"), design_cl=design_cl,
                                   cm_gate=(rep.get("conditions") or {}).get("cm_max")),
                     260, titles=vc.POLAR_TITLES)
        else:
            ui.empty_plot("the report carries no polar for the optimised section", 160)


def optimise(ui, ctx) -> None:
    """Shape optimisation. Before any search: what it is, its objective and
    AeroBO's budget, then the action row. Once a search is live or has run,
    THE RUN COMES FIRST -- the action row (Run turns into Stop), the banner,
    the live evaluation graph and the "now evaluating" block, where a
    1280x800 window shows them without a scroll -- then what it found
    against its seed, and the configuration last."""
    m = ctx.model
    name, _a = m.anchor()
    vc.designing_row(ui, ctx, subject=f"seed: {name}" if name else "seed: the family's anchor")
    job = _live(ctx, "section")
    ran = job is not None or bool(m.opt.get("outcome")) or m.has_optimised()
    if not ran:
        _config_cards(ui, ctx)
    _action_row(ui, ctx, job)
    vc.run_banner(ui, job, None if job else m.opt.get("outcome"),
                  idle="No search yet: Run flies WingLab's shape optimiser live, one XFOIL sweep "
                       "per evaluation.")
    if ran:
        g = m.graph()
        vc.convergence_card(ui, g, label=OBJECTIVE_SHORT.get(g.get("objective"), g.get("objective")),
                            outcome=m.opt.get("outcome"), height=280)
    if job is not None:
        n = len((job.current or {}).get("x") or []) if job.records else 0
        vc.now_evaluating(ui, ctx, job, labels=_cst_labels(n) if n else None)
    if job is None and m.has_optimised():
        _result_cards(ui, ctx)
    if ran:
        ui.sect_head("THE SEARCH")
        _config_cards(ui, ctx)


def _config_cards(ui, ctx) -> None:
    """What the search is, its objective and gates, AeroBO's budget."""
    _what_card(ui, ctx)
    left, right = ui.columns((1, 1))
    with left:
        _objective_card(ui, ctx)
    with right:
        _search_card(ui, ctx)
