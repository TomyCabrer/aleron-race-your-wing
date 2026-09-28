"""drive/views_wing.py -- stage 3's four views on AeroBO's car rear wing:
Wing type, Design box, Solver, Convergence (PLAN2 §8.3; AeroBO V3's
gui/v3/stages/wing.py, shots 14-21).

A view is `fn(ui, ctx)`: it draws itself top to bottom into the shell's work
area with `ui` (a `cae.widgets.WorkUI`) from what `ctx` (a
`design_shell.ViewCtx`) hands it, and never imports `garage` or the engine.
`ctx.model` is the slot's `aerobo_models.WingModel` -- None before the
design page has opened a session -- and its forms (`type_params`,
`box_params()`, `solver_params`, `conv_params`) are what the controls bind to.

Every number here is AeroBO's -- the built problem's box, the measured
plan, the RunConfig a Run sends, the run record, every evaluation -- or
carsim's mission (the design speed, the slot's deck and standoff). A run is
drawn in one of the two looks `views_common` owns: RUNNING · k/N with the
status line, a bar and the "now evaluating" block while it flies; the
STORED DONE / CONVERGED / STOPPED / FAILED tag after. Running and finished
cannot be mistaken for each other (the owner).

The wing's screen asks no "cd at the design cl" (the owner's D9): that is a
section-stage matter, and the wing stage shows no such criterion either.
"""

from __future__ import annotations

import json
import math

import numpy as np

from . import aerobo_models as am
from . import views_common as vc
from .cae import theme as T
from .cae.plot import Figure

WING_NONE = "State the mission first: the wing is created when the design stages open."
NO_RUN = "No run yet. Configure the search on the Solver tab and press Run (F5 there, or O)."

#: what each row of AeroBO's car-wing box IS, in the Design box's second
#: line (AeroBO's param labels are the first); the ride row is worded per
#: slot below (`row_words`; 4 Results' box card words its rows the same)
ROW_WORDS = {
    "taper": "tip chord / root chord",
    "twist_root_deg": "twist at the root [deg]",
    "twist_tip_deg": "twist at the tip [deg]",
    "alpha_deg": "incidence [deg] — the mount angle the slot takes",
    "endplate_h_m": "endplate height [m]",
    "endplate_chord_ratio": "endplate chord / tip chord",
    "endplate_tc": "endplate thickness t/c",
    "endplate_toe_deg": "endplate toe [deg]",
    "endplate_cant_deg": "endplate lean [deg] — 90 is upright",
    "endplate_blend_frac": "endplate root blend [share of its height]",
    "S_m2": "planform area [m²]",
    "b_m": "span [m]",
    "chord_k1": "free chord law, 1st shape term",
    "chord_k2": "free chord law, 2nd shape term",
    "chord_k3": "free chord law, 3rd shape term",
}

#: who set a box row's band (`WingModel.band_source`) -> its chip colour
SOURCE_COLOUR = {"WingLab default": "INK_FAINT", "car packaging": "ACCENT", "slot": "ACCENT",
                 "user": "WARN", "released": "INK_FAINT", "fixed": "WARN", "fixed from 2.8": "GOOD",
                 am.TIP_PIN_SOURCE: "ACCENT"}

BOX_HEAD_HELP = {
    "constrain": "ON: the row is searched inside the band typed here (WingLab's bounds_overrides).\n\n"
                 "OFF: the row is still DESIGNED, over the family's own published band (the car's "
                 "span limit still holds). Your numbers are kept and come back with the switch.",
    "fix": "Fix a row you have already decided (WingLab's `pinned`): the optimiser never sees it, "
           "the search is one dimension smaller, and the run still reports the whole wing. It "
           "opens at the middle of its band; type the value it is held at in its low field.\n\n"
           "A fixed row is NOT a band of width zero — WingLab refuses that.",
    "source": "Who set this band. “WingLab default” is the family's own; “car packaging” the "
              "slot's size rows (the side wing's cut to WingLab's AR ≥ 3); “slot” the ride row the "
              "slot can carry; “user” your number; “fixed from 2.8” the plate's t/c pinned to the "
              "section chosen in 2.8; “tip device: none” the plate's height held at 0 by the "
              "pylons' tip device (pick another device to search it again).",
}

STOP_HELP = ("Ends the search after the evaluation in flight and KEEPS the best design found so "
             "far — WingLab rebuilds its result from its own log; Keep going resumes it.")

#: `converged_report`'s verdict -> the word "Did it converge?" shows, and its hint kind
VERDICTS = {"converged": ("flattened out", "ok"), "climbing": ("still climbing", "warn"),
            "too_short": ("cannot be said yet", "warn"), "no_incumbent": ("cannot be said", "warn"),
            "empty": ("nothing feasible was scored", "warn")}


# --------------------------------------------------------------------------- #
#  small helpers                                                               #
# --------------------------------------------------------------------------- #
def _px(n) -> int:
    """A css px at the shell's scale (the kit's own half-up rounding)."""
    return int(math.floor(n * T.S + 0.5))


def _none(ui, ctx) -> bool:
    if ctx.model is None:
        ui.hint(WING_NONE)
        return True
    return False


def run_job(ctx):
    """The wing's LIVE search (a wing or lap run), else None: the law and
    report jobs that follow a run are short tasks, not the run."""
    job = ctx.job
    if job is None or not ctx.live or getattr(job.info, "kind", "") not in ("wing", "lap"):
        return None
    return job


def short_units(units) -> str:
    """AeroBO's score units without their gloss ("CZ/CD (= F_z/D)" -> "CZ/CD")."""
    return str(units or "").split(" (")[0].strip()


def cached(ctx, key, fn):
    """`fn()` once per `key` in the view's state (a read of the engine that
    costs more than a frame should pay every frame: a margin list, a
    validity check). A failure is cached too, as its exception."""
    store = ctx.state.setdefault("_cache", {})
    if key not in store:
        if len(store) > 32:
            store.clear()
        try:
            store[key] = (True, fn())
        except Exception as exc:                            # noqa: BLE001 -- shown, never raised
            store[key] = (False, exc)
    return store[key]


def _flags_key(wing) -> str:
    try:
        return json.dumps(wing.physics_flags(), sort_keys=True, default=str)
    except Exception:                                       # noqa: BLE001
        return ""


def constraint_labels(ctx, wing) -> list:
    """The margins this family will report, in order, as AeroBO names them
    (`api.constraint_labels_of` on the built problem)."""
    ok, v = cached(ctx, ("cons", wing.family_name, _flags_key(wing)),
                   lambda: list(am.bridge.api.constraint_labels_of(wing.family_name,
                                                                    wing.physics_flags())))
    return v if ok else []


def base_family(wing) -> str:
    """AeroBO's family inside the slot family's registered name."""
    n = str(wing.family_name)
    if " · " in n:
        n = n.split(" · ", 1)[1]
    return n.rsplit(" #", 1)[0]


def slot_words(ctx, wing) -> str:
    if wing.role == "top":
        return "the top wing"
    mirrored = bool(getattr(ctx.g.build, "mirror", False))
    return "the side wing" + (" (both sides fly this design, mirrored)" if mirrored else
                              f" (the {wing.key} one: M split the pair)")


def ground_words(wing) -> str:
    """Where the slot family puts the ground (PLAN2 D2)."""
    op = wing.session.op
    lo, hi = (float(v) for v in op.ride_band)
    if wing.role == "top":
        return (f"ground effect ON — over the car's deck at {float(op.deck):.2f} m, ride height "
                f"{lo:.2f}–{hi:.2f} m (the band this slot can carry)")
    d = float(op.deck or 0.0)
    return (f"ground effect OFF — the image plane is {2.0 * d:.0f} m away; the plates reach "
            f"{lo - d:.2f}–{hi - d:.2f} m to the car's side")


def designing_row(ui, ctx, wing) -> None:
    """The row every wing view opens with: what is being designed and which
    AeroBO family flies it."""
    with ui.row():
        ui.label("designing", css=11.5, colour=T.INK_MUTED, min_w=80)
        ui.tag("TOP WING" if wing.role == "top" else "SIDE WING", T.ACCENT)
        ui.label(T.ellipsize(slot_words(ctx, wing), "mono", 12, max(60, ui.w // 3)), css=12,
                 family="mono")
        ui.gap(12)
        ui.label(T.ellipsize(f"WingLab · {base_family(wing)}", "mono", 12, max(60, ui.w // 2)),
                 css=12, family="mono", colour=T.INK_MUTED)


def own_section(wing, surface) -> str:
    """The family's own section of a surface, flown until one is chosen:
    AeroBO's NACA 24tt at the problem's t/c on the wing, NACA 00tt at the
    searched t/c on the plates."""
    if surface == "ep":
        return "NACA 00tt at the t/c the wing search picks"
    try:
        tc = float(getattr(wing.built().problem, "tc"))
    except Exception:                                       # noqa: BLE001 -- words, not a number
        return "the family's published section"
    return f"NACA 24{int(round(100.0 * tc)):02d}, the family's published section"


def flown(ctx, wing, surface) -> str:
    """What the wing flies on `surface` ("af" | "ep") in words: the section
    taken in that stage, with where it came from, or the family's own."""
    s = wing.session
    if surface == "ep" and not wing.choices["plates"]:
        return "tip devices on the pylons — they carry the wing's own section"
    m = s.af if surface == "af" else s.ep
    c = m.chosen or {}
    if m.decision in ("library", "optimised") and c:
        return str(c.get("name") or "—") + (f" ({c['origin']})" if c.get("origin") else "")
    stage = "2.8" if surface == "ep" else "2"
    return own_section(wing, surface) + (" — kept in stage " + stage if m.decision == "default"
                                         else " — until one is chosen in stage " + stage)


def objective_note(ctx, wing) -> str:
    """AeroBO's note under the objective -- or, when the job is a stop and
    the objective its own, why downforce + drag IS the stop (`STOP_NOTE`),
    and on a side wing why side force + drag stands in for it (`SIDE_STOP_NOTE`)."""
    obj = wing.choices["objective"]
    if obj == am.JOB_OBJECTIVE.get(am.ms.STOPPING) and wing.session.job() == am.ms.STOPPING:
        return am.STOP_NOTE if wing.role == "top" else am.SIDE_STOP_NOTE
    return am.OBJECTIVE_NOTE.get(obj, "").format(circuit=ctx.g.mission.track)


def force_word(wing) -> str:
    return "downforce" if wing.role == "top" else "side force"


def action_row(ui, ctx, wing, *, run_key=None) -> None:
    """Run (when this view has one) and Stop. The run's state is the banner
    under it (`views_common.run_banner`): RUNNING / STOPPING while it flies,
    the stored terminal tag after -- one tag, never two."""
    with ui.row():
        if run_key:
            ui.button(run_key, "Run", kind="primary", icon="play_arrow")
        ui.button("stop", "Stop", kind="outline", icon="stop", help=STOP_HELP)


def _eval_labels(wing, x) -> list | None:
    """The row names of a design vector `x`, when they are known to match."""
    labs = list((wing.record or {}).get("param_labels") or []) or list(wing.family_box())
    return labs if x is not None and len(labs) == len(list(x)) else None


def live_block(ui, ctx, wing, job) -> None:
    """While the search flies: the status line and bar, and what it is
    evaluating now (views_common's two blocks)."""
    vc.run_banner(ui, job, None)
    inc = job.incumbents[-1][2] if job.incumbents else None
    vc.now_evaluating(ui, ctx, job, labels=_eval_labels(wing, inc),
                      units=short_units((wing.record or {}).get("score_units")))


# --------------------------------------------------------------------------- #
#  w.type -- Wing type (AeroBO 14)                                             #
# --------------------------------------------------------------------------- #
LAW_NOTE = {True: "Free: WingLab's three chord-law rows (k1..k3) shape the chord on top of the taper.",
            False: "Straight: a plain taper from the root to the tips."}
#: "Carried by" -- AeroBO's one mount question (`bridge.MOUNT_FLAG`, its car
#: card's `_car_mount_controls`): what holds the wing to the car
MOUNT_LABELS = {"endplates": "the endplates, at the tips", "pylons": "two pylons, inboard"}
#: ...and what each answer means, in a line (`{to}`: the car's deck under a
#: top wing, the car's side for a flank)
MOUNT_NOTE = {
    "endplates": "The plates hold the wing up: they run from its tips to {to}. They are a "
                 "designed part with their own section (stage 2.8).",
    "pylons": "Two swan-neck pylons rise from the car's deck and hold the wing at {frac:g} of "
              "each half-span, over its top. The plates are only a tip device.",
    #  a side wing's pair is two straight struts from the body (garage.wing_polys)
    "pylons_side": "Two struts from the car's side hold the panel at {frac:g} of each "
                   "half-span. The plates are only a tip device.",
}
#: the stated / optimised toggle of the plate's two shape numbers (AeroBO's
#: `_plate_answer_row`: optimised, the field goes and the box row takes it)
ANSWER_LABELS = {"stated": "I state it", "free": "optimise it"}
BLEND_NOTE = ("0 is a sharp corner; more rounds the plate out of the wing over that share of its "
              "height. The plate then reaches less far down and further out, which the span row "
              "pays for.")
CANT_NOTE = "90° is upright; less leans the plate outboard, which the span row pays for."
#: the pylons' tip device, one plain line each (AeroBO's four keys and
#: `CAR_TIP_SHAPE_LABELS`; the vertical plate points at the TRACK only on a
#: top wing)
TIP_NOTE = {
    "none": "No plates: open wing tips. The plate height is fixed at 0 — one row fewer to search.",
    "vertical": "Upright plates at the tips. Their height is searched (Design box: endplate_h_m).",
    "canted": "The plates lean outboard, at the angle below (75° when first picked: 90° is "
              "upright, the vertical plate).",
    "blended": "The plates curve smoothly out of the wing (WingLab's standard blend, half their "
               "height). The curve reaches outboard, so the wing is that much shorter inside "
               "the same width, and the plate ends less far down.",
}
#: ...and its chord, the second question about a fitted device (AeroBO's v3
#: card, "its chord": `bridge.tip_chord_follows`)
CHORD_LABELS = {"follows": "follows the wing's chord", "tip": "holds the tip chord"}
CHORD_NOTE = {
    True: "The plates continue the wing's chord distribution past the tip: a tapered wing gets "
          "tapered plates, with no step at the junction.",
    False: "The plates hold the wing's tip chord the whole way: a rectangle on the tip, "
           "WingLab's published plate."}
#: the Configuration card's label column: wide enough for "Root blend (0–1)"
#: on one line, so every control of the card starts at one x
CARD_LABEL_W = 116


def tip_labels(wing) -> dict:
    """AeroBO's tip-device words for this slot: a flank's plates point at
    the car's side, not the track."""
    out = dict(am.bridge.TIP_DEVICES)
    if wing.role != "top":
        out["vertical"] = out["vertical"].replace("towards the track", "towards the car")
    return out


def _mount_card(ui, ctx, wing) -> None:
    """"Carried by" first (it picks the family), then what that answer
    brings: under the endplates the plate's root blend and lean, each stated
    or optimised; under the pylons the tip device (and its lean when canted)
    and the chord it follows. A row that does not apply is not drawn -- and
    its Param is disabled too, so the keyboard or a pad can never land on it
    (`WingModel.shows_blend_field` / `shows_cant_field` / `shows_tip_cant_field`)."""
    plates = bool(wing.choices["plates"])
    to = "the car's deck" if wing.role == "top" else "the car's side"
    lw = _px(CARD_LABEL_W)
    ui.v1_row("plates", "Carried by", control="toggle", labels=MOUNT_LABELS, label_w=lw,
              note=MOUNT_NOTE["endplates" if plates else
                              ("pylons" if wing.role == "top" else "pylons_side")].format(
                  to=to, frac=float(am.bridge.carmount.INBOARD_STATION_FRAC)))
    if plates:
        ui.v1_row("blend", "Root blend", control="toggle", labels=ANSWER_LABELS, label_w=lw)
        if wing.shows_blend_field():
            ui.v1_row("blend_frac", "Root blend (0–1)", control="number", unit="of plate height",
                      note=BLEND_NOTE, label_w=lw)
        else:
            ui.kv("root blend", "searched by WingLab — its Design box row is endplate_blend_frac",
                  link=("Design box", "north_east", lambda: ctx.select("w", "w.box")))
        ui.v1_row("cant", "Leaning at", control="toggle", labels=ANSWER_LABELS, label_w=lw)
        if wing.shows_cant_field():
            ui.v1_row("cant_deg", "Leaning at", control="number", unit="° from the wing plane",
                      note=CANT_NOTE, label_w=lw)
        else:
            ui.kv("leaning at", "searched by WingLab — its Design box row is endplate_cant_deg",
                  link=("Design box", "north_east", lambda: ctx.select("w", "w.box")))
        return
    tip = am.bridge.tip_device(wing.choices)
    ui.v1_row("tip", "Tip device", control="select", labels=tip_labels(wing), note=TIP_NOTE[tip],
              label_w=lw)
    if wing.shows_tip_cant_field():
        ui.v1_row("tip_cant_deg", "Leaning at", control="number", unit="° from the wing plane",
                  note=CANT_NOTE, label_w=lw)
    if wing.shows_tip_chord():
        ui.v1_row("tip_chord", "Its chord", control="toggle", labels=CHORD_LABELS,
                  note=CHORD_NOTE[am.bridge.tip_chord_follows(wing.choices)], label_w=lw)


def _sync_objective_menu(ui, wing) -> None:
    """The Maximise select offers what THIS family can score: AeroBO's menu
    moves with "Carried by" (lap time only on the pylons, top slot), so the
    choices are re-read from the model on every draw."""
    p = ui.form.param("obj")
    if p is not None:
        want = list(wing.objectives())
        if list(p.choices) != want:
            p.choices = want


def wing_type(ui, ctx) -> None:
    if _none(ui, ctx):
        return
    wing = ctx.model
    _sync_objective_menu(ui, wing)
    designing_row(ui, ctx, wing)
    left, right = ui.columns((3, 2), gap=12)
    with left:
        with ui.card("Configuration"):
            _mount_card(ui, ctx, wing)
            ui.hairline()
            ui.v1_row("law", "Chord law", control="toggle", note=LAW_NOTE[bool(wing.choices["chord_law"])],
                      labels={"free": "free", "straight": "straight"}, label_w=_px(CARD_LABEL_W))
            ui.hairline()
            op = wing.session.op
            ui.kv("mission", " / ".join(w for w in ctx.mp.job_words() if w) + f" · V {float(op.V):.1f} m/s",
                  link=("change in stage 1", "edit", "to1"))
            ui.kv("wing section", flown(ctx, wing, "af"), link=("change in stage 2", "edit", "to2"))
            if wing.choices["plates"]:
                #  the pylons' plates are a tip device with the wing's own
                #  section: there is no plate section to name, or to change
                ui.kv("plate section", flown(ctx, wing, "ep"), link=("change in stage 2.8", "edit", "to28"))
        _objective_card(ui, ctx, wing)
    with right:
        _derived_card(ui, ctx, wing)
        _operating_card(ui, ctx, wing)


def _objective_card(ui, ctx, wing) -> None:
    """AeroBO's car objective menu and its two limits (PLAN2 §7.1)."""
    obj = wing.choices["objective"]
    wants = am.WANTS_LIMIT.get(obj)
    with ui.card("Objective"):
        ui.v1_row("obj", "Maximise", control="select", note=objective_note(ctx, wing),
                  labels=wing.objectives())
        ui.hint("What the ANSWER must satisfy — the design box on the next tab is only where the "
                "search may look.",
                help="WingLab's car limits, both in newtons at the design speed: drag_budget_n is the "
                     "most drag a design may make, downforce_min_n the least force. 0 = none. A "
                     "design that misses a stated limit flies but is infeasible — a hollow ring on "
                     "the evaluation graph.")
        for key, label, want, lead in (
                ("cap", "Drag, no more than", wants == "cap", "drag_budget_n"),
                ("floor", f"{force_word(wing).capitalize()}, at least", wants == "floor",
                 "downforce_min_n")):
            p = ui.form.param(key)
            if p is not None and not getattr(p, "off_text", None):
                p.off_text = "none"
            with ui.row(gap=6):
                ui.label(label, css=11.5, colour=T.INK_MUTED, min_w=150, tip=f"WingLab's {lead}")
                ui.number(key, width=_px(112))
                ui.label("N", css=10.5, family="mono", colour=T.INK_FAINT_TEXT)
                if want:
                    ui.label("← this objective wants it", css=T.HINT_CSS, colour=T.AMBER_LABEL)
        if obj == "laptime":
            ui.hint("A lap prices the drag on the straights and the force in the corners itself: "
                    "neither limit is needed.")
        ok, why = cached(ctx, ("objcheck", _flags_key(wing)),
                         lambda: am.bridge.api.check_wing_objective(wing.physics_flags()))
        if not ok:
            ui.hint(f"WingLab refuses this objective as set: {why}", "bad")


def _derived_card(ui, ctx, wing) -> None:
    """AeroBO's Derived solver: the family this slot flies and what its
    evaluator reports."""
    dim, n = wing_dim(wing)
    with ui.card("Derived solver"):
        ui.sect_head(base_family(wing))
        with ui.row(gap=8):
            ui.tag(str(ctx.mp.job_words()[0]).upper(), T.INK_MUTED)
            ui.tag(f"{dim}-D", T.ACCENT)
            ui.tag("CONSTRAINED", T.WARN)
            ui.tag("GROUND EFFECT" if wing.role == "top" else "NO GROUND EFFECT", T.INK_MUTED)
        ui.hint(f"WingLab's own car-wing evaluator: {n} rows in its box, "
                f"{dim} searched.",
                help="The slot family is WingLab's family with only its ride band moved (and the game's "
                     "air, ρ 1.2): at the family's own band it is bit-for-bit WingLab's problem.")
        cons = constraint_labels(ctx, wing)
        ui.kv("margins", ", ".join(cons) if cons else "—",
              tip="the constraints a run reports, in WingLab's order; feasible = every margin ≥ 0")
        ui.kv("objective", wing.objectives().get(wing.choices["objective"], wing.choices["objective"]))


def _operating_card(ui, ctx, wing) -> None:
    """Where the mission puts this wing (carsim's half of the problem)."""
    op = wing.session.op
    with ui.card("Operating point"):
        ui.kv("design speed", f"{float(op.V):.1f} m/s",
              tip=str(op.V_source or "the stated lap's mean speed"))
        ui.kv("air", f"ρ {float(op.rho):.3g} kg/m³ · ν {float(op.nu):.3g} m²/s (the game's)")
        ui.kv("design CZ", f"{float(op.cz_design):.2f}")
        ui.hint(ground_words(wing), split=False)


def wing_dim(wing) -> tuple:
    """(rows searched, rows in the box)."""
    box = wing.family_box()
    return len(box) - len([k for k in wing.pinned() if k in box]), len(box)


# --------------------------------------------------------------------------- #
#  w.box -- Design box (AeroBO 15)                                             #
# --------------------------------------------------------------------------- #
def row_words(wing, lab) -> str:
    if lab == "ride_height_m":
        if wing.role == "top":
            return "height over the road [m] — the slot's band over the deck"
        return "the plates' reach to the car's side + the image offset [m]"
    return ROW_WORDS.get(lab, "")


def _outside(ctx, wing) -> dict:
    """{label: (searched, validated)} of the rows whose band reaches outside
    what AeroBO validates this family over (`api.rows_outside_validity`)."""
    key = ("outside", wing.family_name, _flags_key(wing),
           json.dumps(wing.bounds_overrides(), sort_keys=True, default=str))
    ok, rows = cached(ctx, key, lambda: am.bridge.api.rows_outside_validity(wing.built()))
    out = {} if not ok else {str(r.get("label")): (r.get("searched"), r.get("validated"))
                             for r in rows or ()}
    #  the TOP slot family moves AeroBO's ride band up to where the car can
    #  carry the wing (PLAN2 D2, H4): AeroBO does not flag its own moved
    #  constant, so the view compares the slot's band with the class's own
    own = aerobo_ride_band(wing)
    band = wing.family_box().get("ride_height_m")
    if wing.role == "top" and own and band and (band[0] < own[0] - 1e-9 or band[1] > own[1] + 1e-9):
        out.setdefault("ride_height_m", (tuple(band), tuple(own)))
    return out


def aerobo_ride_band(wing) -> tuple | None:
    """AeroBO's OWN ride band for this family: the class constant the slot
    family's subclass overrides (read off the built problem's class)."""
    try:
        for cls in type(wing.built().problem).__mro__[1:]:
            if "RIDE_HEIGHT_BOUNDS_M" in vars(cls):
                lo, hi = vars(cls)["RIDE_HEIGHT_BOUNDS_M"]
                return float(lo), float(hi)
    except Exception:                                       # noqa: BLE001 -- a read-out
        return None
    return None


def design_box(ui, ctx) -> None:
    if _none(ui, ctx):
        return
    wing = ctx.model
    box = wing.family_box()
    pins = wing.pinned()
    outside = _outside(ctx, wing)
    designing_row(ui, ctx, wing)
    with ui.card("Design box"):
        ui.hint("The box the search proposes designs from — every row of WingLab's built problem.",
                title="What the columns mean",
                help="Each row is one design variable of WingLab's car rear wing. “low” and “high” "
                     "are the band the search draws from; a row may only NARROW the family's own "
                     "band. The two switches are explained on their column headings.\n\n“Reset to "
                     "the family's box” gives back WingLab's own bands and the slot's.")
        if pins:
            k = len(pins)
            ui.hint(f"Fixed: {', '.join(f'{lab} = {float(v):.4g}' for lab, v in pins.items())}. The "
                    f"search is {k} dimension{'s' if k != 1 else ''} smaller than the full box.",
                    "warn", split=False)
        if outside:
            bits = []
            for lab, (_srch, val) in outside.items():
                bits.append(f"{lab} (WingLab validates {float(val[0]):.3g}–{float(val[1]):.3g})"
                            if val and len(val) == 2 else lab)
            ui.hint(f"Outside WingLab's validated band: {', '.join(bits)} — the evaluator flies it, "
                    f"but WingLab's studies did not cover it.", "warn",
                    help="api.rows_outside_validity: the family's published band is what WingLab's "
                         "studies covered. The top wing rides higher than WingLab's own car wing "
                         "(PLAN2 H4): the lattice and its ground image are valid there, the ground "
                         "effect is simply weaker.")
        _limit_hint(ui, wing, box)
        _box_grid(ui, wing, box, pins, outside)
    with ui.row():
        ui.button("boxreset", "Reset to the family's box", kind="outline", icon="restart_alt")
    _derived_geometry(ui, wing, box)
    _ride_card(ui, wing)


def limit_line(wing) -> str:
    """The slot's span limit on the car being fitted (task 41), in words."""
    s = wing.session
    return am.bridge.limit_words(wing.role, s.car, s.slot.h, s.unlimited)


def _limit_hint(ui, wing, box) -> None:
    """Task 41: the car's span limit this slot is held to, and where the
    span row's ceiling comes from (Real: the limit; Unlimited: 3x it)."""
    s = wing.session
    caps = s.op.size_caps or {}
    b = box.get("b_m")
    tone = "warn" if s.unlimited else None
    ui.hint(f"Span limit — {limit_line(wing)}", tone, split=False,
            help=("The owner's rule (drive/bodies.py): a side wing's lower tip may come down to "
                  "the car's own ground clearance, span ≤ 2 (h − ground); a top wing may be 1.2 × "
                  "the body's width. The span row opens at that limit and cannot be opened past "
                  f"{float(caps.get('b_m', 0.0)):.3g} m here"
                  + (" (Unlimited: 3 × the limit; a run past the limit is filed apart and never "
                     "official)" if s.unlimited else " (Settings ▸ Wing limits: Unlimited allows "
                                                    "3 ×; Real mode puts no wing past the limit "
                                                    "on the car)")
                  + (f". Searched now: b {float(b[0]):.3g}–{float(b[1]):.3g} m." if b else ".")))


def _box_grid(ui, wing, box, pins, outside) -> None:
    """AeroBO's `_box_grid`: one row per design variable -- its name, the
    constrain and fix switches, the band, who set it, and whether it runs
    outside what AeroBO validated."""
    rows = []
    hand = wing.hand_pins()
    for lab, (lo, hi) in box.items():
        src = wing.band_source(lab)
        if lab in am.NO_BOX_CONTROLS:
            #  the owner, 2026-09-25: "endplate t/c shouldn't be given as an
            #  option" -- shown, read-only: no switches, no band to type
            if lab in pins:
                what = (f"{float(pins[lab]):.4f}", "from 2.8")
            else:
                what = ("searched", "by WingLab")
            rows.append([("text2", lab, row_words(wing, lab), None), ("empty",), ("empty",),
                         ("text", what[0]), ("text", what[1], T.INK_FAINT_TEXT, "sans", T.NOTE_CSS),
                         ("tag", src if lab in pins else "WingLab", getattr(
                             T, SOURCE_COLOUR.get(src, "INK_FAINT"))),
                         ("empty",), ("empty",)])
            continue
        cells = [("text2", lab, row_words(wing, lab), None),
                 ("switch", f"bx.{lab}.con", T.ACCENT,
                  "fixed — no band to constrain" if lab in pins
                  else "off: the family's own band is searched"),
                 ("switch", f"bx.{lab}.fix", T.WARN,
                  "give it back to the optimiser" if lab in pins else "hold it at the value you type")]
        if lab in hand:
            #  AeroBO's: ONE field, in the low column -- the value it is held
            #  at, typed -- and the high column says what happened
            cells += [("number", f"bx.{lab}.val"),
                      ("text", "not searched", T.INK_FAINT_TEXT, "sans", T.NOTE_CSS)]
        elif lab in pins:
            cells += [("text", f"fixed at {float(pins[lab]):.5g}"),
                      ("text", "not searched", T.INK_FAINT_TEXT, "sans", T.NOTE_CSS)]
        else:
            cells += [("number", f"bx.{lab}.min"), ("number", f"bx.{lab}.max")]
        cells.append(("tag", src, getattr(T, SOURCE_COLOUR.get(src, "INK_FAINT"))))
        cells.append(("tag", "outside validated", T.WARN) if lab in outside else ("empty",))
        cells.append(("empty",))                    # the filler column: the fields stay narrow
        rows.append(cells)
    header = ["parameter", ("constrain", BOX_HEAD_HELP["constrain"]), ("fix", BOX_HEAD_HELP["fix"]),
              "low", "high", ("source", BOX_HEAD_HELP["source"]), "", ""]
    ui.grid("box", [2.4, "auto", "auto", 1, 1, "auto", "auto", 1.3], header, rows, pitch=37,
            gap=(4, 10))


def _derived_geometry(ui, wing, box) -> None:
    """AeroBO's Derived geometry: the span, area and aspect ratio the box
    allows, and the root chord of the straight-taper equivalent at its
    middle (what the car page draws, PLAN2 H6)."""
    b = box.get("b_m")
    S = box.get("S_m2")
    lam = box.get("taper")
    with ui.card("Derived geometry"):
        if b is None or S is None:
            ui.hint("this family sizes the wing without a span or area row")
            return
        b_lo, b_hi = (float(v) for v in b)
        s_lo, s_hi = (float(v) for v in S)
        ui.kv("span b", f"{b_lo:.3g} – {b_hi:.3g} m" if b_hi - b_lo > 1e-9 else f"{b_lo:.3g} m · fixed")
        ui.kv("area S", f"{s_lo:.3g} – {s_hi:.3g} m²" if s_hi - s_lo > 1e-9 else f"{s_lo:.3g} m² · fixed")
        ar_lo, ar_hi = b_lo * b_lo / max(s_hi, 1e-9), b_hi * b_hi / max(s_lo, 1e-9)
        ui.kv("aspect ratio", f"{ar_lo:.3g} – {ar_hi:.3g}",
              tip=f"AR = b²/S; WingLab flies a car wing only at AR ≥ {am.bridge.AR_MIN:g} (a single "
                  f"chordwise panel is honest only there)")
        if lam is not None:
            bm, sm, lm = 0.5 * (b_lo + b_hi), 0.5 * (s_lo + s_hi), 0.5 * sum(float(v) for v in lam)
            ui.kv("root chord", f"{2.0 * sm / (bm * (1.0 + lm)):.3f} m at the box middle",
                  tip="2S / b(1 + taper): the straight-taper equivalent the car page lofts; the free "
                      "chord law's k1..k3 shape WingLab's physics, not the drawing")
        if wing.role == "flank":
            ui.hint("The side wing's size rows are the car's span limit at this slot height (task 41) "
                    "cut to WingLab's AR ≥ 3: a taller, narrower panel than the car's published one.")


def _ride_card(ui, wing) -> None:
    """What the ride row means in this slot (PLAN2 D2)."""
    with ui.card("The ride row"):
        ui.hint(ground_words(wing), split=False)
        if wing.role == "top":
            ui.hint("Searched and written back to the slot's height when the wing is put on the "
                    "car, like the incidence.")
        else:
            ui.hint("The side wing's plates reach to the car's side: its ride row is that reach plus "
                    "the offset that switches the ground image off (the side wing is not over a floor).")


# --------------------------------------------------------------------------- #
#  w.solver -- Solver (AeroBO 16)                                              #
# --------------------------------------------------------------------------- #
REFUSAL_WORDS = {"worst": "a refused design is shown to the surrogate as the worst one that flew",
                 "sentinel": "a refused design is shown to the surrogate at WingLab's −100"}


def _search(wing) -> tuple:
    """(effective search dict, error text)."""
    try:
        return wing.search(), ""
    except Exception as exc:                                # noqa: BLE001 -- shown, never raised
        return {}, f"{type(exc).__name__}: {exc}"


def solver(ui, ctx) -> None:
    if _none(ui, ctx):
        return
    wing = ctx.model
    pol = wing.session.policy
    rec = pol.recommended()
    eff, err = _search(wing)
    plan = eff.get("plan")
    job = run_job(ctx)
    designing_row(ui, ctx, wing)
    action_row(ui, ctx, wing, run_key="run")
    if job is not None:
        live_block(ui, ctx, wing, job)
    else:
        vc.run_banner(ui, None, wing.outcome, idle=NO_RUN if not wing.record else "")
    if err:
        ui.hint(f"WingLab cannot plan this search as set: {err}", "bad")
    left, right = ui.columns((3, 2), gap=12)
    with left:
        with ui.card("Search strategy · " + ("recommended" if rec else "your own values")):
            ui.field("opt", control="readout", label="optimiser")
            opt = str(eff.get("optimiser") or "—")
            if opt == "bo_slsqp":
                ui.hint("WingLab's GP-BO over the box, handed to SLSQP for the last stretch — what "
                        "WingLab's own wing stage flies when the family and this machine take it.")
            if opt == "bo_slsqp":
                ui.kv("Sobol start", "sized inside the BO phase",
                      tip="the handoff gives BO a quarter of the budget and SLSQP the rest; WingLab "
                          "sizes the Sobol start inside the BO quarter (the run reports its split)")
            else:
                n_init = eff.get("n_init") or getattr(plan, "n_init", None)
                ui.kv("Sobol start", f"{int(n_init)} evaluations" if n_init else "—",
                      tip="the space-filling design the surrogate starts from")
            ui.kv("refusals", REFUSAL_WORDS.get(str(eff.get("refusal")), str(eff.get("refusal") or "—")))
            ui.kv("feasibility", "guide — the surrogate is steered towards the feasible region")
            if plan is not None and getattr(plan, "why", ""):
                ui.hint(str(plan.why))
        _budget_card(ui, ctx, wing, eff, plan, rec)
    with right:
        _what_will_run(ui, ctx, wing, eff)
        note = wing.bo_note()
        if note:
            with ui.card("This machine"):
                ui.hint(note, "warn", split=False)
    with ui.row():
        ui.spacer()
        ui.button("snip", "Copy the run configuration to the output", kind="flat", icon="content_copy")


def _budget_card(ui, ctx, wing, eff, plan, rec) -> None:
    budget = int(eff.get("budget") or 0)
    dim, _n = wing_dim(wing)
    est = getattr(plan, "est_seconds", None) if plan is not None else None
    per = getattr(plan, "per_eval_s", None) if plan is not None else None
    if rec and est:
        t, how = vc.duration(float(est)), "WingLab's measured cost per evaluation of this problem"
    elif per:
        t, how = vc.duration(float(per) * budget), "WingLab's measured cost per evaluation × the budget"
    else:
        t, how = "—", "no measured cost for this problem"
    with ui.card("Budget · " + ("recommended" if rec else "your own")):
        ui.kpis([dict(label="budget", value=str(budget), unit="evals"),
                 dict(label="design variables", value=str(dim)),
                 dict(label="expected", value=t, tip=how)])
        ui.field("budget", label="evaluations", unit="evals",
                 help="recommended: WingLab's measured budget for this problem at the effort on "
                      "1 Mission ▸ Search & budget (53 for the 14-D wing at balanced). Your own "
                      "values: this number")
        ui.field("seed", label="random seed")
        ui.field("stopconv", control="switch", label="stop when it stops improving")
        reach = am.bridge.convergence_rule_reach(eff) if eff else None
        if not wing.session.policy.stop_when_converged or not rec:
            ui.hint("Every run spends its whole budget." if rec else
                    "Your own values: the adaptive stop is WingLab's recommendation, so it is off.")
        elif reach is None:
            ui.hint("This plan has no adaptive stop: the budget is the run.")
        elif reach["can_fire"]:
            ui.hint(f"It stops early once the best has not improved by {100.0 * float(plan.tol):.1f} % "
                    f"in {reach['patience']} evaluations — the budget stays the backstop.")
        else:
            ui.hint(f"At this budget the stop cannot fire: it needs {reach['needs']} evaluations, the "
                    f"budget is {reach['budget']}.", "warn")
        with ui.row(gap=12):
            ui.button("mine", "use my own values", kind="flat", icon="edit")
            ui.button("s1", "stage 1 · Search & budget", kind="flat", icon="north_east")
        st = am.bridge.study_stamp() or {}
        if st:
            ui.hint(f"WingLab's budget study: measured {st.get('measured', '?')}, {st.get('n_runs', '?')} "
                    f"runs over {st.get('cases', '?')} cases.")


def _what_will_run(ui, ctx, wing, eff) -> None:
    """AeroBO's reproduce snippet: the RunConfig a Run sends, as
    `aerobo.api.run` takes it."""
    with ui.card("What will run"):
        try:
            cfg = wing.cfg()
        except Exception as exc:                            # noqa: BLE001 -- shown, never raised
            ui.hint(f"no configuration: {type(exc).__name__}: {exc}", "bad")
            return
        ui.kv("problem", base_family(wing), tip=am.bridge.shown_family(cfg.problem_name))
        ui.kv("optimiser", f"{cfg.optimiser}, {int(cfg.budget)} evaluations, seed {int(cfg.seed)}")
        ui.kv("wing section", flown(ctx, wing, "af"))
        if wing.choices["plates"]:
            ui.kv("plate section", flown(ctx, wing, "ep"))
        lines = ["api.run(api.RunConfig(", f"    problem_name={cfg.problem_name!r},",
                 f"    optimiser={cfg.optimiser!r}, budget={int(cfg.budget)}, seed={int(cfg.seed)},",
                 "    flags={"]
        for k, v in (cfg.flags or {}).items():
            lines.append(f"        {k!r}: {_short(v)},")
        lines.append("    },")
        if cfg.bounds_overrides:
            lines.append("    bounds_overrides={")
            for k, v in cfg.bounds_overrides.items():
                lines.append(f"        {k!r}: [{float(v[0]):.6g}, {float(v[1]):.6g}],")
            lines.append("    },")
        if cfg.pinned:
            lines.append("    pinned={" + ", ".join(f"{k!r}: {float(v):.6g}" for k, v in cfg.pinned.items())
                         + "},")
        lines.append("))")
        ui.code(lines)


def _short(v) -> str:
    """A flag value for the snippet: a section's coordinates are not
    printed whole."""
    if isinstance(v, dict):
        if "w_upper" in v:
            return "{" + f"'name': {v.get('name')!r}, 'w_upper': [...], 'w_lower': [...], " \
                         f"'re': {float(v.get('re') or 0):.4g}" + "}"
        return repr({k: v[k] for k in list(v)[:4]})
    if isinstance(v, float):
        return f"{v:.6g}"
    return repr(v)


# --------------------------------------------------------------------------- #
#  w.conv -- Convergence (AeroBO 17-21)                                        #
# --------------------------------------------------------------------------- #
def convergence(ui, ctx) -> None:
    if _none(ui, ctx):
        return
    wing = ctx.model
    job = run_job(ctx)
    rec = wing.record
    designing_row(ui, ctx, wing)
    action_row(ui, ctx, wing)
    if job is not None:
        live_block(ui, ctx, wing, job)
    else:
        vc.run_banner(ui, None, wing.outcome, idle=NO_RUN if not rec else "")
    if job is None and not rec:
        with ui.row():
            ui.link("conv.to_solver", "Go to the Solver", "arrow_back",
                    lambda: ctx.select("w", "w.solver"))
        if (wing.outcome or {}).get("state") == "error":
            ui.hint("Nothing of the failed run was applied.", "warn")
        return
    g = wing.graph()
    label = wing.objectives().get(wing.choices["objective"], wing.choices["objective"])
    if job is None and rec is not None:
        label = _record_objective(wing, rec, label)
    vc.convergence_card(ui, g, title="Every evaluation", label=label, outcome=wing.outcome,
                        height=300)
    if job is None and rec is not None:
        _found_card(ui, ctx, wing, rec)
        _converge_card(ui, wing, rec)
        _margins_card(ui, ctx, wing, rec)
        _onto_car_card(ui, ctx, wing)
    elif rec is not None:
        ui.hint("The cards below the graph return when this run lands: what it found, whether it "
                "converged, and the way onto the car.")


def _record_objective(wing, rec, default) -> str:
    """The objective the RECORD was flown on (the form may have moved on)."""
    obj = ((rec.get("config") or {}).get("flags") or {}).get("car_objective") or "efficiency"
    return wing.objectives().get(obj, default) if obj != wing.choices["objective"] else default


def handoff_words(rec) -> str:
    """How a bo_slsqp run spent its budget: Sobol -> BO -> SLSQP."""
    h = rec.get("handoff") or {}
    if not isinstance(h, dict) or not h:
        split = rec.get("bo_split")
        return f"{split[0]} Sobol + {split[1]} BO" if split else str(rec.get("optimiser") or "—")
    sp = h.get("bo_split") or [None, None]
    bits = []
    if sp[0]:
        bits.append(f"{sp[0]} Sobol")
    if sp[1]:
        bits.append(f"{sp[1]} BO")
    if h.get("n_b"):
        bits.append(f"{h['n_b']} SLSQP")
    words = " → ".join(bits) or "—"
    return words + ("" if h.get("handed_over", True) else " (not handed over)")


def _found_card(ui, ctx, wing, rec) -> None:
    """AeroBO's What the search bought, from the stored record: the best
    score in AeroBO's units and the winner's forces at the design speed."""
    bd = rec.get("breakdown") or {}
    best = rec.get("best_score")
    tiles = [dict(label="best", value=vc.num(best), unit=short_units(rec.get("score_units")) or None,
                  colour=T.ACCENT, tip=str(rec.get("score_units") or ""))]
    if vc.finite(bd.get("downforce_N")):
        tiles.append(dict(label=force_word(wing), value=f"{float(bd['downforce_N']):.1f}", unit="N"))
    if vc.finite(bd.get("drag_N")):
        tiles.append(dict(label="drag", value=f"{float(bd['drag_N']):.2f}", unit="N"))
    if vc.finite(bd.get("CZ")):
        tiles.append(dict(label="CZ", value=f"{float(bd['CZ']):.4f}"))
    if vc.finite(bd.get("CD")):
        tiles.append(dict(label="CD", value=f"{float(bd['CD']):.4f}"))
    if vc.finite(bd.get("lap_time_s")):
        tiles.append(dict(label="lap (WingLab)", value=f"{float(bd['lap_time_s']):.3f}", unit="s"))
    with ui.card("What the search found"):
        ui.kpis(tiles)
        n_f, n = rec.get("n_feasible"), rec.get("n_evals")
        ui.kv("feasible", f"{n_f} of {n} evaluations" + ("" if rec.get("feasible") else
                                                          " — the best is NOT feasible"),
              colour=None if rec.get("feasible") else T.BAD)
        ui.kv("spent as", handoff_words(rec))
        if rec.get("resumed"):
            ui.kv("resumed", f"from {int(rec['resumed'])} evaluations — none of them flown again")
        if rec.get("partial"):
            ui.kv("ended early", str(rec.get("stop_reason") or "partial"), colour=T.WARN)
        if vc.finite(rec.get("wall_time_s")):
            ui.kv("wall time", f"{float(rec['wall_time_s']):.1f} s")


def _converge_card(ui, wing, rec) -> None:
    """AeroBO's Did it converge? (`converged_report` at the plan's patience
    and tolerance) and Keep going (AeroBO's resume)."""
    rep = wing.verdict() or {}
    word, kind = VERDICTS.get(rep.get("verdict"), ("cannot be said", "warn"))
    n = int(rec.get("n_evals") or 0)
    with ui.card("Did it converge?"):
        ui.kpis([dict(label="verdict", value=word,
                      tip="WingLab's converged_report on this run's best-so-far history, at the "
                          "plan's patience and tolerance")])
        if rep.get("text"):
            ui.hint(str(rep["text"]), kind)
        with ui.row(gap=6):
            ui.label("more evaluations", css=11.5, min_w=150,
                     tip="added to the evaluations this run already paid for")
            ui.number("more")
            ui.spacer()
            ui.button("go", kind="primary", icon="play_arrow")
        if wing.can_continue():
            ui.hint(f"A resume, not a re-run: the {n} evaluations already paid for are the new "
                    f"search's training set — it starts at {n + 1}.", split=False)
        else:
            ui.hint("This record cannot be resumed (WingLab's can_resume): Run starts a new search.",
                    "warn")


def _margins_card(ui, ctx, wing, rec) -> None:
    """AeroBO's Constraint margins: every evaluation's signed margins, in
    AeroBO's order and names; feasible means all of them ≥ 0."""
    recs = [r for r in wing.records if isinstance(r.get("g"), (list, tuple)) and r.get("g")]
    names = list((rec.get("breakdown") or {}).get("constraint_labels") or []) or constraint_labels(ctx, wing)
    if not recs:
        return
    m = max(len(r["g"]) for r in recs)
    fig = Figure(xlabel="evaluation", ylabel="signed margin g", legend=True)
    ks = np.array([float(r.get("n", i + 1)) for i, r in enumerate(recs)])
    for j in range(m):
        ys = np.array([float(r["g"][j]) if j < len(r["g"]) and vc.finite(r["g"][j]) else math.nan
                       for r in recs])
        if not np.isfinite(ys).any():
            continue
        fig.line(ks, ys, width=1.2, markers="circle", marker_size=4,
                 name=names[j] if j < len(names) else f"g[{j}]")
    fig.hline(0.0, colour=T.BAD, width=1.5, dash="dash")
    with ui.card("Constraint margins", pad=False,
                 help="WingLab's signed margins, each normalised by its own limit: ≥ 0 is met. An "
                      "evaluation the engine refused reports none."):
        ui.plot(fig, 260)


def _onto_car_card(ui, ctx, wing) -> None:
    """Put it on the car (P6): the law sampled from AeroBO's evaluator at
    the winner, the game's force at the design speed against AeroBO's."""
    law = wing.law
    with ui.card("Onto the car"):
        if law:
            inc = float((wing.slot_updates or {}).get("inc_deg", law.get("alpha_design_deg", 0.0)))
            game = am.ResultsModel.game_force(law, inc)
            aero = (law.get("aerobo") or {}).get("F_N")
            ui.kv("car's law", f"CZ = {law['CL0']:+.4f} + {law['CLa']:.4f}·α, clamped "
                               f"[{law['CL_min']:+.3f}, {law['CL_max']:+.3f}]")
            ui.kv(f"{force_word(wing)} at {float(law['V_ref']):.1f} m/s",
                  f"game {game:.2f} N · WingLab {float(aero):.2f} N" if vc.finite(aero) else f"game {game:.2f} N",
                  tip="the game computes ½ρV²S·CZ from the law with its own ρ; at the design point "
                      "it is WingLab's force by construction")
        elif wing.law_outcome and wing.law_outcome.get("state") == "error":
            ui.hint(f"no law: {wing.msg}", "bad")
        else:
            ui.hint("The car's law is being sampled from WingLab's evaluator at the winning design "
                    "(an incidence sweep, well under a second).")
        spec = wing.spec
        if spec is not None:
            ui.kv("wing", spec.name + (" — not on the car yet" if wing.dirty else " — on the car"),
                  colour=T.WARN if wing.dirty else T.GOOD)
        with ui.row():
            ui.button("fit", kind="primary", icon="check")
        ui.hint("Saves the sections and the wing into the library, puts it in the slot (mirrored on "
                "the side wings) and writes WingLab's incidence" + (" and ride height" if wing.role == "top"
                                                               else "") + " to the slot.")
