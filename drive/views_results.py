"""drive/views_results.py -- stage 4's four views: Summary, Geometry, Loading,
Evaluations (PLAN2 §8.4; AeroBO V3's gui/v3/stages/results.py, shots 22-25).

A view is `fn(ui, ctx)`: it draws itself top to bottom into the shell's work
area with `ui` (a `cae.widgets.WorkUI`) from what `ctx` (a
`design_shell.ViewCtx`) hands it, and never imports `garage` or the engine.
`ctx.model` is the slot's `aerobo_models.ResultsModel`; the wing it reads is
`ctx.model.session.wing`. The four views share one form (`ResultsModel.
params`: Keep going, Put it on the car, and the ways back to stage 3), drawn
at the foot of each.

What they show is the RUN THAT LANDED, read off its stored record (AeroBO's
`RunResult.to_dict()`), AeroBO's `design_report` of the winner (the geometry
and spanwise breakdown, a short job after every run) and the law the car
flies it with -- sampled from AeroBO's evaluator at the winner, so the
game's force at the design speed IS AeroBO's. carsim's own quasi-steady lap
with and without the wing is shown too, labelled as the second model it is.
While a Keep going flies, each view says so at its top: the numbers below
are the run it continues until the new one lands.
"""

from __future__ import annotations

import math

import numpy as np

from . import aerobo_models as am
from . import views_common as vc
from . import views_wing as vw
from .cae import theme as T
from .cae.plot import Figure

WING_NONE = "State the mission first: the wing is created when the design stages open."
PRE_HINT = ("No completed run yet: 4 Results reads the run that landed. Run the wing on "
            "3 Wing ▸ Solver.")
LIVE_HINT = ("A run is flying: the numbers below are the run it continues, and they update when "
             "it lands.")
MARGIN_HELP = ("WingLab's signed margins, each normalised by its own limit — ≥ 0 is met. Feasible "
               "means every one of them is. Their names are the family's own "
               "(api.constraint_labels_of), in the order the engine reports them.")
LAW_HELP = ("carsim's car does not run WingLab's lattice at 1 kHz: it flies an affine law SAMPLED "
            "from WingLab's own evaluator at the winning design — an incidence sweep, the lift "
            "slope from ±0.5°, the stall clamps at WingLab's own refusal edges, the drag a "
            "quadratic fitted through the design point. At the design incidence the game's "
            "½ρV²S·CZ is WingLab's force to rounding.")
LAP_HELP = ("carsim's quasi-steady model of the job with this wing in its slot, against the same "
            "car with the slot empty: the circuit's lap (a side wing's own circuit) or the "
            "stop's distance. A DIFFERENT model from WingLab's own "
            "(cartrack's point mass, which only the lap-time objective uses): the two will not "
            "agree to the tenth.")
GEOMETRY_HELP = ("WingLab's design report of the winner (api.design_report): the lattice's own "
                 "stations. The chord is the free chord law's, exactly as the engine flew it.")
MESH_HELP = ("The wing as the CAR draws it: carsim lofts the straight-taper equivalent of the "
             "chord law (root chord 2S / b(1 + taper)) — the free law's k1..k3 shape WingLab's "
             "physics, not this drawing (PLAN2 H6). The plates hang at their searched height.\n\n"
             "Drag it to turn it and see every side; the links under it jump to a view.")

#: the drawing's camera, (azimuth, elevation) in degrees about the wing's
#: centre -- carsim's frame: x forward, y left, z up. It opens where the
#: fixed camera stood (behind and above); the mouse turns it as the car
#: page's orbit does (`garage.Orbit`: 0.46 / 0.34 deg a pixel)
MESH_VIEW0 = (-135.0, 28.0)
MESH_VIEWS = (("3/4 rear", -135.0, 28.0), ("front", 0.0, 8.0), ("rear", 180.0, 8.0),
              ("outboard", 90.0, 4.0), ("top", 180.0, 85.0), ("under", -135.0, -55.0))
MESH_EL_LIMIT = 85.0


# --------------------------------------------------------------------------- #
#  what every view reads                                                       #
# --------------------------------------------------------------------------- #
def _open(ui, ctx):
    """The ResultsModel, after the line every results view opens with: the
    WING_NONE hint (None) before a session; before any run, the pre hint and
    the ways back (None); while a run flies, its banner and the line saying
    the numbers are the previous run's."""
    res = ctx.model
    if res is None:
        ui.hint(WING_NONE)
        return None
    wing = res.session.wing
    job = vw.run_job(ctx)
    if job is not None:
        vc.run_banner(ui, job, None)
        if wing.record:
            ui.hint(LIVE_HINT, "warn")
    if not wing.record:
        if job is None:
            vc.run_banner(ui, None, wing.outcome, idle=PRE_HINT)
        _actions(ui, ctx)
        return None
    return res


def _actions(ui, ctx) -> None:
    """The results form, at the foot of every view: Keep going, Put it on
    the car, and the ways back to stage 3."""
    ui.hairline()
    with ui.row(gap=8):
        ui.button("keep_r", kind="outline", icon="play_arrow")
        ui.button("save", kind="primary", icon="check")
        ui.spacer()
        ui.button("to3", "3 Wing ▸ Solver", kind="flat", icon="arrow_back")
        ui.button("tobox", "3 Wing ▸ Design box", kind="flat", icon="crop_free")


def _tag_row(ui, wing) -> None:
    """The stored terminal tag of the run these views read."""
    out = wing.outcome
    tag = vc.tag_of(None, out)
    if tag is None:
        return
    with ui.row():
        vc.draw_tag(ui, tag, wall=(out or {}).get("wall"))


def run_key(wing) -> tuple:
    """What a cached read of the run depends on, by VALUE (an object's id
    can be reused once the model has let the object go): the record's
    size and score, and the law's coefficients and design speed."""
    rec, law = wing.record or {}, wing.law or {}
    return (rec.get("n_evals"), rec.get("best_score"), rec.get("timestamp"),
            law.get("CL0"), law.get("CLa"), law.get("V_ref"), law.get("S"))


def _breakdown(res) -> dict:
    """AeroBO's breakdown at the winner: the design report's when it has
    landed, else the record's own (the same evaluation, kept by the run)."""
    rep = res.report or {}
    return dict(rep.get("breakdown") or (res.session.wing.record or {}).get("breakdown") or {})


def _report_state(ui, ctx, res) -> bool:
    """True when AeroBO's design report is in; else says why not (being
    computed, failed, or not asked for yet -- with the way to ask)."""
    if res.report:
        return True
    job = ctx.runs.job_for(res)
    if job is not None:
        with ui.row():
            ui.spinner("WingLab's design report of the winner is being computed…")
        return False
    if (res.outcome or {}).get("state") == "error":
        ui.hint(f"no design report: {res.msg or res.outcome.get('error')}", "bad")
    else:
        ui.hint("WingLab's design report of the winner has not been computed yet.")
    with ui.row():
        ui.link("res.report", "Compute it now", "play_arrow", res.start_report,
                enabled=not ctx.runs.busy)
    return False


# --------------------------------------------------------------------------- #
#  r.summary (AeroBO 22)                                                       #
# --------------------------------------------------------------------------- #
def summary(ui, ctx) -> None:
    """AeroBO 22, in its order: the KPI row, the constraint margins, the
    sections flown; then carsim's half -- the car's law and the game's
    force against AeroBO's, carsim's own lap as a cross-check -- and the
    run itself."""
    res = _open(ui, ctx)
    if res is None:
        return
    wing = res.session.wing
    sm = res.summary()
    _tag_row(ui, wing)
    _kpis(ui, wing, sm)
    _margins(ui, sm)
    _sections(ui, ctx, wing)
    _law_card(ui, wing, sm)
    _lap_card(ui, ctx, res)
    _run_card(ui, ctx, wing)
    _actions(ui, ctx)


def _kpis(ui, wing, sm) -> None:
    units = sm.get("score_units") or ""
    tiles = [dict(label="best score", value=vc.num(sm.get("best_score")),
                  unit=vw.short_units(units) or None, colour=T.ACCENT,
                  tip=f"what WingLab maximised: {units}" if units else None)]
    fw = sm.get("force_word") or "force"
    if vc.finite(sm.get("force_N")):
        tiles.append(dict(label=fw, value=f"{float(sm['force_N']):.1f}", unit="N",
                          tip="at the design speed, WingLab's breakdown"))
    if vc.finite(sm.get("drag_N")):
        tiles.append(dict(label="drag", value=f"{float(sm['drag_N']):.2f}", unit="N"))
    if vc.finite(sm.get("CZ")):
        tiles.append(dict(label="CZ", value=f"{float(sm['CZ']):.4f}"))
    if vc.finite(sm.get("CD")):
        tiles.append(dict(label="CD", value=f"{float(sm['CD']):.4f}"))
    if vc.finite(sm.get("efficiency")):
        tiles.append(dict(label="CZ / CD", value=f"{float(sm['efficiency']):.2f}"))
    if vc.finite(sm.get("lap_time_s")):
        tiles.append(dict(label="lap (WingLab)", value=f"{float(sm['lap_time_s']):.3f}", unit="s",
                          tip="cartrack's point-mass lap of carsim's circuit (the objective)"))
    n, budget = sm.get("n_evals"), sm.get("budget")
    tiles.append(dict(label="evaluations", value=f"{n}/{budget}" if budget else str(n)))
    if vc.finite(sm.get("wall_s")):
        tiles.append(dict(label="wall", value=f"{float(sm['wall_s']):.1f}", unit="s"))
    ui.kpis(tiles)
    if not sm.get("feasible"):
        ui.hint(f"No feasible design: the best of {n} evaluations misses a constraint — see the "
                f"margins below.", "bad")


def _margins(ui, sm) -> None:
    rows = sm.get("margins") or []
    with ui.card("Constraint margins", help=MARGIN_HELP):
        if not rows:
            ui.hint("this run reports no margins")
            return
        for label, g in rows:
            ok = g is not None and g >= 0.0
            with ui.row():
                ui.tag("OK" if ok else ("VIOLATED" if g is not None else "—"),
                       T.GOOD if ok else (T.BAD if g is not None else T.INK_FAINT))
                ui.label(str(label))
                ui.spacer()
                ui.label(vc.num(g, "{:+.4f}"), css=12, family="mono")
        ui.hint("Feasible ⟺ every signed margin ≥ 0.")


def _sections(ui, ctx, wing) -> None:
    s = wing.session
    with ui.card("Sections flown"):
        ui.kv("wing section", vw.flown(ctx, wing, "af"),
              link=("change in stage 2", "edit", lambda: ctx.select("af", "af.rank" if s.af.ranked
                                                                       else "af.screen")))
        if wing.choices["plates"]:
            ui.kv("plate section", vw.flown(ctx, wing, "ep"),
                  link=("change in stage 2.8", "edit",
                        lambda: ctx.select("ep", "ep.rank" if s.ep.ranked else "ep.screen")))
        else:
            ui.kv("plate section", vw.flown(ctx, wing, "ep"))
        bd = (wing.record or {}).get("breakdown") or {}
        if bd.get("polar"):
            ui.kv("polar flown", str(bd["polar"]), tip="the wing section's polar, as WingLab's "
                                                      "evaluator names it")
        if bd.get("endplate_section_label"):
            ui.kv("plate polar", str(bd["endplate_section_label"]))


def _law_card(ui, wing, sm) -> None:
    law = sm.get("law")
    with ui.card("The car's law", help=LAW_HELP):
        if not law:
            if wing.law_outcome and wing.law_outcome.get("state") == "error":
                ui.hint(f"no law: {wing.msg}", "bad")
            else:
                ui.hint("The car's law is being sampled from WingLab's evaluator at the winner.")
            return
        ui.kv("CZ(α)", f"{law['CL0']:+.4f} + {law['CLa']:.4f}·α  (α in rad)")
        ui.kv("clamped to", f"[{law['CL_min']:+.3f}, {law['CL_max']:+.3f}] — stall at "
                            f"{law['alpha_stall_neg_deg']:+.1f}° / {law['alpha_stall_deg']:+.1f}°")
        ui.kv("CD(CZ)", f"{law['cd0']:.5f} {law['cd1']:+.5f}·CZ {law['cd2']:+.5f}·CZ²",
              tip=f"fitted over the sweep; max error {100.0 * float(law.get('cd_fit_err') or 0.0):.1f} % "
                  f"off the design point, exact at it")
        ui.kv("S · AR · e", f"{float(law['S']):.4f} m² · {float(law['AR']):.3f} · {float(law['e']):.3f}")
        fw = sm.get("force_word") or "force"
        g, a = sm.get("game_force_N"), sm.get("aerobo_force_N")
        if vc.finite(g):
            same = vc.finite(a) and abs(float(g) - float(a)) <= 1e-6 * max(1.0, abs(float(a)))
            ui.kv(f"{fw} at the design speed", f"game {float(g):.3f} N · WingLab {vc.num(a, '{:.3f}')} N",
                  colour=T.GOOD if same else T.WARN,
                  tip="the game's ½ρV²S·CZ(α) with carsim's ρ against WingLab's breakdown")
        writes = [f"incidence {float(sm.get('inc_deg') or 0.0):+.2f}°"]
        if wing.role == "top":
            h = (wing.slot_updates or {}).get("h")
            if h is None:
                x = dict(zip((wing.record or {}).get("param_labels") or [], (wing.record or {}).get("best_x") or []))
                h = x.get("ride_height_m")
            if vc.finite(h):
                writes.append(f"height {float(h):.3f} m")
        ui.kv("the slot takes", ", ".join(writes) + " when the wing is put on the car")
        spec = wing.spec
        if spec is not None:
            ui.kv("wing", spec.name + (" — not on the car yet (S)" if wing.dirty else " — on the car"),
                  colour=T.WARN if wing.dirty else T.GOOD)


def _lap_card(ui, ctx, res) -> None:
    """carsim's own cross-check with and without the wing, in the job's
    terms (`ResultsModel.car_lap`: a circuit's lap, a stop's distance),
    cached per law and job: two laps are ~60 ms, too
    much for every frame."""
    wing = res.session.wing
    if not wing.law:
        return
    m = ctx.g.mission
    key = ("lap", run_key(wing), res.session.job(), m.job_key, m.surface, float(m.mu_scale),
           tuple(sorted((wing.slot_updates or {}).items())))
    ok, lap = vw.cached(ctx, key, res.car_lap)
    title = str((lap or {}).get("title") or "the lap") if ok else "the lap"
    with ui.card(f"carsim's own check — {title}", help=LAP_HELP):
        if not ok or not lap or lap.get("error"):
            ui.hint(f"the check does not close: {lap if not ok else lap.get('error', '')}", "warn")
            return
        unit, fmt, dfmt = lap.get("unit", "s"), lap.get("fmt", "{:.3f}"), lap.get("dfmt", "{:+.3f}")
        ui.kv("with this wing", vc.num(lap.get("with"), fmt + " " + unit))
        ui.kv("this slot empty", vc.num(lap.get("without"), fmt + " " + unit))
        d = lap.get("delta")
        better = vc.finite(d) and ((float(d) < 0) if lap.get("lower_better", True) else (float(d) > 0))
        ui.kv("difference", vc.num(d, dfmt + " " + unit), colour=(T.GOOD if better else None))
        ui.hint(str(lap.get("model") or ""), split=False)


def _run_card(ui, ctx, wing) -> None:
    rec = wing.record
    cfg = rec.get("config") or {}
    rep = wing.verdict() or {}
    word, kind = vw.VERDICTS.get(rep.get("verdict"), ("cannot be said", "warn"))
    with ui.card("Run"):
        ui.kv("problem", vw.base_family(wing) if rec.get("problem_name") == wing.family_name
              else str(rec.get("problem_name")).split(" · ", 1)[-1].rsplit(" #", 1)[0],
              tip=str(rec.get("problem_name")))
        ui.kv("optimiser", f"{rec.get('optimiser')} · budget {cfg.get('budget')} · seed {cfg.get('seed')}")
        ui.kv("spent as", vw.handoff_words(rec))
        n = int(rec.get("n_evals") or 0)
        if rec.get("resumed"):
            ui.kv("evaluations", f"{n} ({int(rec['resumed'])} inherited, none flown again)")
        else:
            ui.kv("evaluations", str(n))
        if rec.get("partial"):
            ui.kv("ended early", str(rec.get("stop_reason") or "partial"), colour=T.WARN)
        ui.kv("converged?", word, colour=None if kind == "ok" else T.WARN, tip=rep.get("text") or None)
        if rec.get("record_path"):
            ui.kv("record", str(rec["record_path"]))


# --------------------------------------------------------------------------- #
#  r.geometry (AeroBO 23)                                                      #
# --------------------------------------------------------------------------- #
def _stations(geo) -> tuple:
    """(wing, plate) stations of AeroBO's geometry: arrays y, chord, z of the
    lattice's own stations, the wing's sorted by y."""
    y = np.asarray(geo.get("y") or [], float)
    c = np.asarray(geo.get("chord") or [], float)
    z = np.asarray(geo.get("z") or [], float)
    n = min(y.size, c.size, z.size)
    y, c, z = y[:n], c[:n], z[:n]
    wl = np.asarray(geo.get("is_winglet") or [False] * n, bool)[:n]
    if wl.size != n:
        wl = np.zeros(n, bool)
    main = ~wl
    o = np.argsort(y[main])
    return (y[main][o], c[main][o], z[main][o]), (y[wl], c[wl], z[wl])


def geometry(ui, ctx) -> None:
    """AeroBO 23: the planform and the front view from AeroBO's design
    report, its numbers, then (carsim) the wing as the car draws it."""
    res = _open(ui, ctx)
    if res is None:
        return
    wing = res.session.wing
    if _report_state(ui, ctx, res):
        geo = res.report.get("geometry") or {}
        (yw, cw, zw), (yp, cp, zp) = _stations(geo)
        if geo.get("frame"):
            ui.hint(f"WingLab's frame: {geo['frame']}.")
        with ui.card("Planform", help=GEOMETRY_HELP, pad=False):
            ui.plot(_planform_fig(yw, cw), 320)
        with ui.card("Front view — looking upstream", pad=False,
                     help="The wing's stations and the plates' at their true height: the plates are "
                          "the lattice's vertical panels at the tips."):
            ui.plot(_front_fig(yw, zw, yp, zp, wing), 300)
        _numbers_card(ui, geo, _breakdown(res), wing)
    _mesh_card(ui, ctx, wing)
    _actions(ui, ctx)


def _planform_fig(y, c) -> Figure:
    """The wing seen from above: leading edge at −c/4, trailing edge at
    +3c/4 of each station (AeroBO's report has no sweep for this family)."""
    fig = Figure(xlabel="span y [m]", ylabel="x [m] · downstream +", equal=True, yrev=True,
                 empty_text="the report carries no stations")
    if y.size < 2:
        return fig
    fig.line(np.concatenate([y, y[::-1]]), np.concatenate([-0.25 * c, 0.75 * c[::-1]]),
             colour=T.ACCENT, width=2, closed=True, fill=T.BAND_A, name="wing (WingLab's chord law)")
    fig.hline(0.0, colour=T.INK_MUTED, width=0.8, dash="dash", label="quarter chord")
    for s in (float(np.min(y)), float(np.max(y))):
        fig.vline(s, colour=T.WARN, width=2, dash="dot")
    return fig


def _front_fig(yw, zw, yp, zp, wing) -> Figure:
    fig = Figure(xlabel="span y [m]", ylabel="z [m] (WingLab's frame)", equal=True,
                 empty_text="the report carries no stations")
    if yw.size:
        fig.line(yw, zw, colour=T.ACCENT, width=4, name="wing")
    if yp.size:
        for i, s in enumerate((-1.0, 1.0)):
            k = np.sign(yp) == s
            if k.any():
                o = np.argsort(zp[k])
                fig.line(yp[k][o], zp[k][o], colour=T.WARN, width=4, markers="circle", marker_size=3,
                         name="endplates" if i == 0 else None)
    return fig


def _numbers_card(ui, geo, bd, wing) -> None:
    rows = [("span b", geo.get("b"), "{:.4f} m"), ("area S", geo.get("S"), "{:.4f} m²"),
            ("aspect ratio", bd.get("AR"), "{:.3f}"), ("taper", geo.get("taper"), "{:.4f}"),
            ("twist root / tip", None, None), ("section t/c", geo.get("tc"), "{:.4f}"),
            ("incidence α*", bd.get("alpha_deg"), "{:+.3f}°"), ("Reynolds (MAC)", bd.get("Re_mac"), "{:.4g}")]
    with ui.card("Numbers"):
        for k, v, f in rows:
            if k == "twist root / tip":
                ui.kv(k, f"{vc.num(geo.get('twist_root_deg'), '{:+.3f}')}° / "
                         f"{vc.num(geo.get('twist_tip_deg'), '{:+.3f}')}°")
                continue
            if vc.finite(v):
                ui.kv(k, f.format(float(v)))
        if wing.choices["plates"] or vc.finite(bd.get("endplate_h_m")):
            ui.kv("endplate height", vc.num(bd.get("endplate_h_m"), "{:.4f} m"))
            #  the pylons' tip device (the fence family) has no chord, t/c or
            #  toe row of its own: no line of dashes for it
            if vc.finite(bd.get("endplate_chord_m")):
                ui.kv("endplate chord", vc.num(bd.get("endplate_chord_m"), "{:.4f} m"))
            if vc.finite(bd.get("endplate_tc")) or vc.finite(bd.get("endplate_toe_deg")):
                ui.kv("endplate t/c · toe", f"{vc.num(bd.get('endplate_tc'), '{:.4f}')} · "
                                            f"{vc.num(bd.get('endplate_toe_deg'), '{:+.2f}')}°")
        if wing.role == "top":
            ui.kv("ride height", vc.num(bd.get("ride_height_m"), "{:.4f} m"),
                  tip=f"over the deck at {vc.num(bd.get('deck_height_m'), '{:.3f}')} m")
        elif vc.finite(bd.get("reach_m")):
            ui.kv("reach to the car", f"{float(bd['reach_m']):.4f} m", tip="ground effect off")
        if bd.get("mount_label"):
            ui.kv("carried by", str(bd["mount_label"]))


def _mesh_card(ui, ctx, wing) -> None:
    """The mapped WingSpec lofted as the car page lofts it (garage.wing_polys
    through the shell's constants), turned by the mouse: drag to see every
    side, the links under it for the set views."""
    polys_fn = ctx.consts.get("wing_polys")
    spec = wing.spec if wing.spec is not None else _preview_spec(ctx, wing)
    with ui.card("On the car (carsim's drawing)", help=MESH_HELP, pad=False):
        if polys_fn is None or spec is None:
            ui.hint("the car's drawing appears once the law is derived")
            return
        cam = _mesh_cam(ctx)

        def grab(pos):
            cam["grab"] = (tuple(pos), cam["az"], cam["el"])
            return "ui"

        def turn(pos):
            p0, az0, el0 = cam.get("grab") or (tuple(pos), cam["az"], cam["el"])
            cam["az"] = (az0 - 0.46 * (pos[0] - p0[0]) + 180.0) % 360.0 - 180.0
            cam["el"] = min(max(el0 + 0.34 * (pos[1] - p0[1]), -MESH_EL_LIMIT), MESH_EL_LIMIT)
        ui.custom(340, lambda surf, rect: _draw_mesh(surf, rect, ctx, wing, spec, polys_fn),
                  on_click=grab, on_drag=turn)
        with ui.row():
            for name, az, el in MESH_VIEWS:
                ui.link(f"meshview.{name}", name, None,
                        lambda a=az, e=el: cam.update(az=a, el=e), small=True)
            ui.label(f"drag to turn · {cam['az']:+.0f}° / {cam['el']:+.0f}°", css=T.HINT_CSS,
                     colour=T.INK_MUTED)
    ui.kv("drawn as", f"span {spec.span:.3f} m · root chord {spec.chord:.3f} m · taper {spec.taper:.3f} · "
                      f"plates {spec.plate_h:.3f} m")


def _preview_spec(ctx, wing):
    """The WingSpec `fit` would make, without fitting (no dirty flag, no
    name taken): AeroBO's winner mapped by the bridge, cached per record."""
    if not wing.record or not wing.law:
        return None
    s = wing.session

    def make():
        sections = {"main": s.af.chosen if s.af.decision in ("library", "optimised") else None,
                    "plate": (s.ep.chosen if (wing.choices["plates"]
                                              and s.ep.decision in ("library", "optimised")) else None)}
        spec, _upd = am.bridge.to_wingspec(wing.record, wing.law, sections, wing.role, "preview")
        return spec
    ok, spec = vw.cached(ctx, ("preview", run_key(wing)), make)
    return spec if ok else None


class _SlotView:
    """The slot as the drawing wants it: the car's, with the run's writes."""

    def __init__(self, slot, updates):
        self.x = float(slot.x)
        self.h = float((updates or {}).get("h", slot.h))
        self.inc_deg = float((updates or {}).get("inc_deg", slot.inc_deg))
        self.wing = getattr(slot, "wing", "")
        self.mode = getattr(slot, "mode", "active")


def _draw_mesh(surf, rect, ctx, wing, spec, polys_fn) -> None:
    import pygame
    surf.fill(T.WELL, rect)
    #  ...and how it is carried and its plates drawn: a new mount, lean or
    #  blend on the same run must not show the last one's picture
    key = ("mesh", spec.name, float(spec.span), float(spec.chord), float(spec.plate_h), run_key(wing),
           rect.size, spec.mount, float(spec.pylon_frac), float(spec.plate_cant_deg),
           float(spec.plate_blend), spec.plate_shape, float(spec.plate_chord_ratio),
           bool(spec.plate_chord_follows), float(spec.taper))
    hit = ctx.state.get("_mesh")
    cam = _mesh_cam(ctx)
    view = (float(cam["az"]), float(cam["el"]))
    if hit is None or hit[0] != key:
        slot = _SlotView(wing.session.slot, wing.slot_updates or
                         {"inc_deg": (wing.law or {}).get("alpha_design_deg", 0.0)})
        try:
            #  on the car being fitted (task 41: its body side and deck)
            geo = getattr(getattr(ctx.g, "view", None), "geo", None)
            polys = polys_fn(spec, wing.key if wing.key != "right" else "left", slot, 1.0, ctx.g.lib,
                             **({"geo": geo} if geo is not None else {}))
        except Exception:                                   # noqa: BLE001 -- a drawing, not a number
            polys = []
        ctx.state["_mesh"] = hit = (key, _prepare(polys, ctx, rect.size), {})
    faces = hit[2].get(view)
    if faces is None:
        if len(hit[2]) > 64:                                # a drag passes through many views
            hit[2].clear()
        faces = hit[2][view] = _project(hit[1], view, rect.size)
    if not faces:
        T.text(surf, "nothing to draw", rect.centerx, rect.centery, "sans", 12, T.INK_FAINT,
               anchor="centre")
        return
    for pts, col in faces:
        pygame.draw.polygon(surf, col, [(rect.x + x, rect.y + y) for x, y in pts])


def _mesh_cam(ctx) -> dict:
    """The drawing's camera, kept with the view's state: {"az", "el"} in
    degrees (and the drag's grab point)."""
    cam = ctx.state.get("_mesh_cam")
    if cam is None:
        cam = ctx.state["_mesh_cam"] = {"az": MESH_VIEW0[0], "el": MESH_VIEW0[1]}
    return cam


def _basis(view) -> tuple:
    """(eye, right, up) unit vectors of a camera at (azimuth, elevation) deg."""
    az, el = math.radians(view[0]), math.radians(view[1])
    eye = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    right = np.cross(-eye, [0.0, 0.0, 1.0])
    right /= max(np.linalg.norm(right), 1e-9)
    up = np.cross(right, -eye)
    return eye, right, up


def _prepare(polys, ctx, size) -> dict | None:
    """The polygons about their centre, each with its flat shade from the
    garage's light and its colour, and ONE scale for every view: the fit of
    the set views (the old fixed camera's among them), so the wing keeps its
    size while it turns and no set view cuts it."""
    verts = [np.asarray(p[0], float) for p in polys or () if len(p[0]) >= 3]
    if not verts:
        return None
    allv = np.concatenate(verts)
    centre = 0.5 * (allv.min(axis=0) + allv.max(axis=0))
    light = np.asarray(ctx.consts.get("LIGHT_DIR", (0.45, 0.55, 0.70)), float)
    light /= max(np.linalg.norm(light), 1e-9)
    items = []
    for p in polys:
        v = np.asarray(p[0], float) - centre
        if len(v) < 3:
            continue
        kind = p[2] if len(p) > 2 else "wing"
        n = np.cross(v[1] - v[0], v[2] - v[0])
        nn = np.linalg.norm(n)
        shade = 0.45 + 0.55 * (abs(float(n @ light)) / nn if nn > 0 else 0.5)
        base = T.WARN if kind == "plate" else (T.ACCENT if str(kind).startswith("wing") else
                                               tuple(p[1][:3]))
        items.append((v, tuple(int(min(255, c * shade)) for c in base)))
    w, h = size
    pad = int(round(20 * T.S))
    scale = np.inf
    for _name, az, el in MESH_VIEWS:
        _eye, right, up = _basis((az, el))
        sx, sy = np.abs((allv - centre) @ right), np.abs((allv - centre) @ up)
        scale = min(scale, (w - 2 * pad) / max(2.0 * sx.max(), 1e-9),
                    (h - 2 * pad) / max(2.0 * sy.max(), 1e-9))
    return {"items": items, "scale": float(scale)}


def _project(prep, view, size) -> list:
    """The prepared polygons seen from `view` (azimuth, elevation deg),
    painter's order, the wing's centre at the slot's centre."""
    if not prep or not prep["items"]:
        return []
    eye, right, up = _basis(view)
    w, h = size
    scale = prep["scale"]
    faces = sorted(((float(np.mean(v @ eye)), v @ right, v @ up, col) for v, col in prep["items"]),
                   key=lambda f: f[0])
    return [([(w / 2.0 + scale * a, h / 2.0 - scale * b) for a, b in zip(fx, fy)], col)
            for _d, fx, fy, col in faces]


# --------------------------------------------------------------------------- #
#  r.loading (AeroBO 24)                                                       #
# --------------------------------------------------------------------------- #
def loading(ui, ctx) -> None:
    """AeroBO 24: the spanwise load and the bending moment from AeroBO's
    breakdown, the local lift coefficient and effective angle from its
    report, and the structural margins they set."""
    res = _open(ui, ctx)
    if res is None:
        return
    bd = _breakdown(res)
    y = np.asarray(bd.get("y") or [], float)
    load = np.asarray(bd.get("load_Npm") or [], float)
    mom = np.asarray(bd.get("moment_Nm") or [], float)
    if y.size and load.size == y.size:
        with ui.card("Spanwise load", pad=False,
                     help="The force per metre of span at each lattice station (WingLab's breakdown "
                          "at the winner, at the design speed)."):
            fig = Figure(xlabel="span y [m]", ylabel="load [N/m]", legend=False)
            fig.line(y, load, colour=T.ACCENT, width=2, markers="circle", marker_size=3)
            ui.plot(fig, 260)
    else:
        ui.hint("This run's breakdown carries no spanwise load.")
    if y.size and mom.size == y.size:
        with ui.card("Bending moment", pad=False,
                     help="The bending moment the load puts into the wing's structure, as WingLab's "
                          "structural check reads it (its supports: " + str(bd.get("supports") or "—")
                          + ")."):
            fig = Figure(xlabel="span y [m]", ylabel="moment [N·m]", legend=False)
            fig.line(y, mom, colour=T.WARN, width=2)
            ui.plot(fig, 240)
    if _report_state(ui, ctx, res):
        geo = res.report.get("geometry") or {}
        _cl_card(ui, geo)
    with ui.card("Structure"):
        ui.kv("peak moment", vc.num(bd.get("M_max_Nm"), "{:.3f} N·m"))
        ui.kv("tip deflection", vc.num(bd.get("deflection_m"), "{:.3e} m"),
              tip=f"margin {vc.num(bd.get('g_deflection'), '{:+.4f}')}")
        if vc.finite(bd.get("endplate_deflection_m")):
            ui.kv("endplate deflection", f"{float(bd['endplate_deflection_m']):.3e} m",
                  tip=f"margin {vc.num(bd.get('g_endplate'), '{:+.4f}')}")
        if vc.finite(bd.get("endplate_stress_Pa")):
            ui.kv("endplate stress", f"{float(bd['endplate_stress_Pa']) / 1e6:.2f} MPa")
        if vc.finite(bd.get("endplate_side_load_N")):
            ui.kv("endplate side load", f"{float(bd['endplate_side_load_N']):.2f} N")
        if bd.get("mount_label"):
            ui.kv("carried by", str(bd["mount_label"]))
    _actions(ui, ctx)


def _cl_card(ui, geo) -> None:
    y = np.asarray(geo.get("y") or [], float)
    cl = np.asarray(geo.get("Cl_y") or [], float)
    ae = np.asarray(geo.get("alpha_eff_deg") or [], float)
    wl = np.asarray(geo.get("is_winglet") or [False] * y.size, bool)
    n = min(y.size, cl.size, ae.size, wl.size)
    if n < 2:
        return
    y, cl, ae, wl = y[:n], cl[:n], ae[:n], wl[:n]
    main = ~wl
    o = np.argsort(y[main])
    fig = Figure(xlabel="span y [m]", ylabel="local cl", y2label="α_eff [deg]")
    fig.line(y[main][o], cl[main][o], colour=T.ACCENT, width=2, markers="circle", marker_size=3,
             name="wing — local cl")
    fig.line(y[main][o], ae[main][o], colour=T.WARN, width=1.6, dash="dot", axis="y2",
             name="α_eff [deg]")
    if wl.any():
        fig.scatter(y[wl], cl[wl], colour=T.WARN, symbol="diamond", size=5, name="endplate panels")
    with ui.card("Local lift and effective angle", pad=False,
                 help="WingLab's report: the section lift coefficient and the effective angle of attack "
                      "each lattice strip flies. A peak at the tip stalls there first."):
        ui.plot(fig, 280)


# --------------------------------------------------------------------------- #
#  r.evals (AeroBO 25)                                                         #
# --------------------------------------------------------------------------- #
def evaluations(ui, ctx) -> None:
    """AeroBO 25: every evaluation of the run -- the graph, then the table
    (its score, feasible or not, its margins by name, its design vector),
    paged by 10."""
    res = _open(ui, ctx)
    if res is None:
        return
    wing = res.session.wing
    rec = wing.record
    vc.convergence_card(ui, wing.graph(), title="Every evaluation",
                        label=wing.objectives().get(wing.choices["objective"], wing.choices["objective"]),
                        outcome=wing.outcome, height=280)
    cols, rows = _eval_table(ctx, wing, rec)
    with ui.card("Every evaluation, one row each", pad=not rows):
        if rows:
            #  ten a page: a page and its pager fit the work area at 1280 x 800, so the
            #  keyboard cursor never walks a row below the fold
            ui.table("evals", cols, rows, page_size=10, sortable=True, h_scroll=True)
        else:
            ui.hint("this run recorded no evaluations")
    ui.hint(f"Units: {rec.get('score_units') or 'the objective'} (what WingLab maximises). A refused "
            f"evaluation scores WingLab's −100.")
    _actions(ui, ctx)


def _eval_rows(wing, rec) -> list:
    """The evaluation payloads {n, f, feasible, g, x}: the model's stored
    ones when they cover the record, else rebuilt from the record's own
    arrays (feasible = every margin ≥ 0 and a score)."""
    recs = list(wing.records or [])
    n = int(rec.get("n_evals") or 0)
    if recs and len(recs) >= n:
        return recs
    X, Y, G = rec.get("eval_x") or [], rec.get("eval_y") or [], rec.get("eval_g") or []
    out = []
    for i, (x, f) in enumerate(zip(X, Y)):
        g = list(G[i]) if i < len(G) and G[i] is not None else []
        feas = vc.finite(f) and float(f) != vc.PENALTY and all(vc.finite(v) and float(v) >= 0 for v in g)
        out.append({"n": i + 1, "f": f, "feasible": feas, "g": g, "x": list(x)})
    return out


def _eval_table(ctx, wing, rec) -> tuple:
    recs = _eval_rows(wing, rec)
    labels = list(rec.get("param_labels") or [])
    names = list((rec.get("breakdown") or {}).get("constraint_labels") or []) or vw.constraint_labels(ctx, wing)
    m = max([len(r.get("g") or []) for r in recs] or [0])
    cols = [dict(key="#", head="#", align="right"), dict(key="f", head="score f", align="right"),
            dict(key="ok", head="feasible", align="left")]
    cols += [dict(key=f"g{j}", head=(names[j] if j < len(names) else f"g[{j}]"), align="right")
             for j in range(m)]
    cols += [dict(key=f"x{i}", head=lab, align="right") for i, lab in enumerate(labels)]
    rows = []
    for i, r in enumerate(recs):
        row = {"#": int(r.get("n", i + 1))}
        if vc.refused(r):
            row["f"], row["ok"] = "refused", "—"
            row["_colours"] = {"f": T.BAD}
        else:
            row["f"] = f"{float(r['f']):.6g}"
            row["ok"] = "yes" if r.get("feasible") else "no"
            if not r.get("feasible"):
                row["_colours"] = {"ok": T.WARN}
        g = list(r.get("g") or [])
        for j in range(m):
            v = g[j] if j < len(g) else None
            row[f"g{j}"] = vc.num(v, "{:+.4f}", "")
        x = list(r.get("x") or [])
        if len(x) == len(labels):
            for k, v in enumerate(x):
                row[f"x{k}"] = vc.num(v, "{:.5g}", "")
        rows.append(row)
    return cols, rows
