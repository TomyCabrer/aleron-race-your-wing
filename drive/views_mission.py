"""drive/views_mission.py -- stage 1's three views: Operating point, Design
point, Search & budget (PLAN2 §8.1).

The mission is carsim's own -- a lap of one of carsim's circuits on a
surface, for a slot -- and it is the one part of the design page that is
not AeroBO's (the owner: "the only thing that could be different is the
mission and the looks"). What it SUPPLIES is AeroBO's: the slot's
operating point (`DesignSession.op`: the design speed, the reference CZ,
carsim's air, the deck and the ride band of the slot family), the Reynolds
numbers the two sections are screened and designed at, and the search
policy whose budgets are AeroBO's measured ones (164 evaluations for a
section, 53 for the wing, at balanced).

A view is `fn(ui, ctx)`: it draws itself top to bottom into the shell's
work area with `ui` (a `cae.widgets.WorkUI`) from what `ctx` (a
`design_shell.ViewCtx`) hands it -- `ctx.mp` is the MissionPage, whose
`params` form every control here binds to; `ctx.dp.session_for(slot)` the
slot's `aerobo_models.DesignSession`. It never imports `garage`, and every
number it draws is the lap's (carsim), the session's (AeroBO's arithmetic)
or AeroBO's plan (`SearchPolicy.plans`).
"""

from __future__ import annotations

import math

import numpy as np

from . import aerobo_models as am
from . import views_common as vc
from .aero import mission as ms
from .cae import theme as T
from .cae.plot import Figure
from .cae.widgets import SEG_PAD
from .track import make_track

# --------------------------------------------------------------------------- #
#  the words                                                                   #
# --------------------------------------------------------------------------- #
SLOT_HINT = ("The slot is the car page's selection; the side pair is mirrored unless M split "
             "them. Stating the mission opens WingLab's design stages for this slot.")
#: the job cards (the owner, 2026-09-25: "One more circuit should be added
#: 'Stopping'. Left flank and right flank, should be side."; 2026-09-26: "why
#: no longer circuits?" -- a side wing picks a circuit of its own)
JOB_HINT = ("The job is what the wing is for. It sets the design speed and the objective 3 Wing "
            "opens on; you can still pick any other there.")
STOP_ABOUT = ("a straight-line stop from {v:.0f} km/h to a standstill, as the Stop from 100 "
              "challenge runs it")
STOP_HINT = ("A stop decelerates at μ·(m·g + downforce) + drag, and both wing forces scale with V², "
             "so the wing with the most μ·downforce + drag stops shortest: 3 Wing opens on "
             "WingLab's “downforce + drag”, which is that ranking exactly at μ = 1 (dry). On damp "
             "or wet it rates downforce a little too high.")
AIR_STOP_HINT = ("The air brake: both side panels out while the car brakes. Their side forces "
                 "cancel; their drags add to the tyres' grip and the top wing's downforce, as "
                 "fitted. The tyres still do nearly all of the stopping.")
SIDE_HINT = ("A side wing pushes the car into the turn in this circuit's corners, so 3 Wing "
             "opens on side force. It has a job of its own: the top wing's job does not move it.")
SIDE_STOP_HINT = ("A side wing stops the car as the air brake: both panels out, their side "
                  "forces cancel and their drags add, so the stop pays for drag alone. 3 Wing "
                  "opens on WingLab's “side force + drag”, the one objective that grows with "
                  "drag. It has a job of its own: the top wing's job does not move it.")
CAR_HINT = ("quasi-steady: corners at their steady speed, straights the acceleration profile "
            "met by the braking profile — an OPTIMUM, not a prediction. This is carsim's lap; "
            "its mean speed is the design speed WingLab flies the wing at.")
BRAKING = ("Braking uses tyre grip only: brakes are not modelled separately. A wing is compared "
           "against a wing under one assumption; the absolute time is not a claim about the real "
           "car.")
STATE_FIRST = "Stating it opens 2 Airfoil, 2.8 Endplate and 3 Wing for this slot."
STATE_AGAIN = "Stating it again only re-checks it: the design stages keep their progress."
STATE_RESETS = ("Stating the mission again re-opens the design stages from the start: sections "
                "already put on a car stay in the library, the screens and runs on these pages "
                "do not.")
LOCKED_LIVE = "a run is in progress — stop it to change the mission"
NO_LAP = "this mission does not fly: {err}"

AIR_HINT = ("carsim's air — ρ 1.2 kg/m³ and ν 1.5e-5 m²/s (aero/polar.py) — not WingLab's "
            "sea-level 1.225, so the force WingLab reports IS the force the game applies: "
            "½·ρ·V²·S·CZ with the game's ρ.")
V_TIP = ("where every coefficient of the wing is read, and the Reynolds number both sections "
         "are designed at. The job's own by default: a circuit lap's mean speed (length / time; "
         "a side wing's own circuit) or a stop's start speed; or one you type.")
CZ_TIP = ("WingLab's reference lift coefficient of a car wing (1.0, `REFERENCE_CL`): the wing's "
          "section is screened and optimised at this cl. The plate's is 0 — its panels are "
          "vertical and carry no design lift.")
SECTION_POINT_HINT = ("WingLab's arithmetic, at the MIDDLE of the box the wing will search: the "
                      "wing section's chord is S/b; the plate's is its chord ratio times the "
                      "wing's tip chord, 2·S·λ/(b·(1+λ)). Re = V·c/ν.")
LIB_POINT_HINT = ("The cached library point: every one of WingLab's 2174 sections has a polar "
                  "there already (the warm checkpoint and the branch sidecar), so a screen at it "
                  "is instant and exact at any cl. The surface's OWN Re is a live XFOIL sweep of "
                  "the shortlist — about 20 s cold, instant after.")

SMODE_ROWS = (("recommended", "WingLab's measured recommendation",
               "every budget is WingLab's own plan for the problem (api.recommended_search over "
               "search_budget.json)"),
              ("own", "My own values",
               "the three budgets below are yours; the optimiser, its Sobol start and its "
               "refusal rule stay WingLab's"))
CONVERGED_HINT = ("The effort is how far WingLab's measured runs had got toward the best design "
                  "any run of the case found: quick, balanced, thorough. It moves every budget "
                  "on this page.")
STOP_HELP = ("WingLab's ConvergenceStop at the plan's patience and tolerance: a run ends once its "
             "best has not improved by the tolerance over the patience. It never ends a run AT "
             "its budget — the budget stays the ceiling — and every evaluation paid for is kept. "
             "The section plans carry no adaptive rule (patience None): a section search spends "
             "its budget unless you stop it.")
OWN_HINT = ("Your own values are live: these are the budgets stages 2, 2.8 and 3 fly, and the "
            "measured plan does not move them.")

#: the stage names in AeroBO's order, as the plans name them
STAGE_OWN_KEY = {"2 Airfoil": "own_af", "2.8 Endplate": "own_ep", "3 Wing": "own_w"}


# --------------------------------------------------------------------------- #
#  small readers                                                               #
# --------------------------------------------------------------------------- #
def _slot_label(ctx, key) -> str:
    return str(ctx.consts.get("SLOT_LABEL", {}).get(key, key)).lower()


def _sync_slot_choices(form, g) -> None:
    """The slots the toggle offers: the side wing ONCE while the pair is
    mirrored (one design, one name -- the owner, "just side wing"), both
    sides only once M has split them."""
    p = form.param("slot")
    want = ["left", "top"] if g.build.mirror else ["left", "right", "top"]
    if p is not None and list(p.choices) != want:
        p.choices = want


def _wing_name(ctx, key) -> str:
    """"side wing", "top wing": the slot's label, read as a wing."""
    label = _slot_label(ctx, key)
    return label if label.endswith("wing") else f"{label} wing"


def _session(ctx):
    """The selected slot's DesignSession (made on first use: the Mission
    page's own rows already edit its operating point), or None."""
    try:
        return ctx.dp.session_for(ctx.g.sel)
    except Exception:                                   # noqa: BLE001 -- a read-out
        return None


def _lap_ok(mp) -> bool:
    """A circuit's lap flew (not a stop, not a side wing's job)."""
    return isinstance(mp.result, ms.LapResult) and mp.result.ok


def _stop_ok(mp) -> bool:
    return isinstance(mp.result, ms.StopResult) and mp.result.ok


def _no_lap(ui, mp) -> None:
    ui.hint(NO_LAP.format(err=mp.err or "the lap did not close"), "bad")


def _delta(mp):
    """The car as it stands against the same car with no wings, s (None
    when either lap did not close)."""
    b = mp.bare
    if not (_lap_ok(mp) or _stop_ok(mp)) or b is None or not b.ok:
        return None
    return mp._delta()


def _library_point() -> dict:
    """AeroBO's cached library point (`api.screen_library_point`, read
    through the models' bridge; {} when the sidecar is missing)."""
    try:
        return dict(am.bridge.api.screen_library_point() or {})
    except Exception:                                   # noqa: BLE001
        return {}


def _toggle_or_select(ui, key) -> None:
    """A bare segmented toggle when every choice fits across the column,
    else a select: the circuit list is the track registry's, and grows."""
    p = ui.form.param(key)
    pad = 2 * max(1, math.floor(SEG_PAD * T.S + 0.5))       # a segment's padding, as WorkUI scales it
    need = sum(T.text_w(str(c), "sans", 11.5) + pad for c in p.choices)
    if need <= ui.w:
        ui.toggle(key)
    else:
        ui.select(key)


# --------------------------------------------------------------------------- #
#  the circuit polyline, once per track                                        #
# --------------------------------------------------------------------------- #
_CIRCUITS: dict = {}
#: most points the map's centreline is drawn with
MAP_POINTS = 600
#: the map's margin round the circuit, a fraction of its larger extent
MAP_PAD = 0.06


def _circuit(name):
    """The centreline of circuit `name` and a `T{i}` label per corner, in
    the order `aero.mission.TrackProfile` counts them. `make_track` costs
    7-8 ms, so this is built once per name, never per frame. A label sits a
    little inside its corner's apex, toward the arc's centre, so it stays
    inside the plot's range. None when the circuit does not build."""
    if name in _CIRCUITS:
        return _CIRCUITS[name]
    try:
        tr = make_track(name)
        xy = np.asarray(tr.xy, float)
        labels = []
        #  the map draws every frame: a point every ~2 m is the same line at
        #  300 px and a quarter of the cost (the last point stays: the seam)
        step = max(1, math.ceil(len(xy) / MAP_POINTS))
        line = np.vstack([xy[::step], xy[-1:]]) if (len(xy) - 1) % step else xy[::step]
        for k, sg in enumerate(tr.segs):
            #  a corner is what TrackProfile.from_track calls one: an arc
            #  tighter than the straight threshold
            if getattr(sg, "kind", "S") != "A" or abs(sg.radius) <= 1e-9 \
                    or 1.0 / abs(sg.radius) <= ms.KAPPA_STRAIGHT:
                continue
            s_mid = float(tr.node_s[k]) + 0.5 * float(sg.length)
            i = int(min(max(np.searchsorted(tr.s, s_mid), 0), len(xy) - 1))
            psi = float(tr.psi[i])
            sign = 1.0 if sg.turn_deg >= 0.0 else -1.0
            inset = min(0.6 * abs(float(sg.radius)), 24.0)
            nx, ny = -math.sin(psi) * sign, math.cos(psi) * sign       # toward the arc's centre
            labels.append((xy[i, 0] + inset * nx, xy[i, 1] + inset * ny, f"T{len(labels) + 1}"))
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        pad = MAP_PAD * float(max(hi - lo))
        out = dict(x=line[:, 0], y=line[:, 1], closed=bool(tr.closed), labels=labels,
                   xlim=(lo[0] - pad, hi[0] + pad), ylim=(lo[1] - pad, hi[1] + pad))
    except Exception:                                   # noqa: BLE001 -- a bad circuit is an empty map
        out = None
    _CIRCUITS[name] = out
    return out


def _map_fig(poly, n_corners) -> Figure:
    fig = Figure(equal=True, xlabel="x [m]", ylabel="y [m]", legend=False,
                 xlim=poly["xlim"], ylim=poly["ylim"])
    fig.line(poly["x"], poly["y"], colour=T.INK_MUTED, width=2, closed=poly["closed"])
    #  the corner numbers only when they are the lap's corners, one for one
    if len(poly["labels"]) == n_corners:
        for x, y, s in poly["labels"]:
            fig.text(x, y, s, colour=T.ACCENT, size=10, anchor="centre")
    return fig


# --------------------------------------------------------------------------- #
#  1 Mission ▸ Operating point                                                 #
# --------------------------------------------------------------------------- #
def operating(ui, ctx) -> None:
    """What the session designs (the slot), on what (circuit, surface), the
    car as it stands there, WHICH AeroBO problem the slot will fly, and the
    state action."""
    g, mp = ctx.g, ctx.mp
    form = ui.form
    with ui.card("What this session designs"):
        _sync_slot_choices(form, g)
        ui.toggle("slot", labels={c: _slot_label(ctx, c) + ("" if g.build.mirror or c == "top"
                                                           else f", {c}")
                                  for c in form.param("slot").choices})
        ui.kv("wing in the slot", g.build.slot(g.sel).wing or "empty — WingLab designs a new one")
        ui.hint(SLOT_HINT)
    job = mp.job()
    p = form.param("track")
    if list(p.choices) != mp.job_choices():            # the slot's own jobs
        p.choices = mp.job_choices()
    left, right = ui.columns((1, 1))
    with left:
        with ui.card("Job"):
            _toggle_or_select(ui, "track")
            if job == ms.STOPPING:
                ui.field("vstop", control="number")
                about = STOP_ABOUT.format(v=g.mission.v_stop_kmh)
            else:
                try:
                    about = mp.profile().describe()
                except Exception as exc:                # noqa: BLE001 -- the lap card says so too
                    about = f"the circuit does not build: {exc}"
            ui.hint(about, split=False, help=p.help)
            side = ctx.consts["SLOT_ROLE"][g.sel] == "flank"
            ui.hint((SIDE_STOP_HINT if job == ms.STOPPING else SIDE_HINT) if side else JOB_HINT,
                    split=False)
    with right:
        with ui.card("Surface"):
            ui.toggle("surf")
            ui.kv("grip scale", f"× {g.mission.mu_scale:.3f}")
            ui.hint(form.param("surf").help)
    _car_card(ui, ctx)
    _family_card(ui, ctx)
    _state_row(ui, ctx)


def _car_card(ui, ctx) -> None:
    """The car as it stands: what carsim's lap says about the car on this
    circuit before anything is designed."""
    mp, form = ctx.mp, ui.form

    def tip(key):
        p = form.param(key)
        return (p.help or None) if p is not None else None
    job = mp.job()
    if job == ms.STOPPING:
        with ui.card("The car as it stands  (carsim's stop)"):
            if not _stop_ok(mp):
                _no_lap(ui, mp)
                return
            r, d = mp.result, _delta(mp)
            ui.kpis([dict(label="stopping distance", value=f"{r.distance:.2f}", unit="m"),
                     dict(label="vs the car with no wings", value="—" if d is None else f"{d:+.2f}",
                          unit="m", colour=T.GOOD if d is not None and d < 0.0 else None),
                     dict(label="time to stop", value=f"{r.time:.2f}", unit="s"),
                     dict(label="mean deceleration", value=f"{r.decel_mean / 9.81:.2f}", unit="g")])
            ui.hint(AIR_STOP_HINT if mp.mission().stop_flanks > 1 else STOP_HINT, help=BRAKING)
        return
    with ui.card("The car as it stands  (carsim's lap)"):
        if not _lap_ok(mp):
            _no_lap(ui, mp)
            return
        r, d = mp.result, _delta(mp)
        ui.kpis([dict(label="lap", value=f"{r.time:.3f}", unit="s", tip=tip("lap")),
                 dict(label="vs the car with no wings", value="—" if d is None else f"{d:+.3f}",
                      unit="s", colour=T.GOOD if d is not None and d < 0.0 else None, tip=tip("dlap")),
                 dict(label="mean speed", value=f"{r.v_mean:.1f}", unit="m/s", tip=tip("vm")),
                 dict(label="fastest point", value=f"{r.v_max:.1f}", unit="m/s", tip=tip("vx"))])
        ui.hairline()
        ui.kpis([dict(label="in corners", value=f"{r.t_corner:.2f}", unit="s", tip=tip("tc")),
                 dict(label="on straights", value=f"{r.t_straight:.2f}", unit="s", tip=tip("ts"))])
        lead, rest = T.split_hint(CAR_HINT)
        ui.hint(lead, split=False, help="\n\n".join(s for s in (rest, BRAKING) if s))


def _ground_words(op) -> tuple:
    """(ground effect, ride band) of a slot family, in words: the top wing
    over the car's deck (ground effect ON), a flank with its image plane
    pushed far away (OFF) and its plate reaching the car's side."""
    lo, hi = (float(v) for v in op.ride_band)
    if op.role == "top":
        return (f"ON — WingLab's image plane is the car's deck under the wing, {float(op.deck):.2f} m "
                f"above the ground at x {op.x:+.2f} m",
                f"{lo:.2f}–{hi:.2f} m above the ground: the band the top slot can carry it in")
    deck = float(op.deck)
    return (f"OFF — the image plane is pushed {deck:.0f} m away ({2.0 * deck:.0f} m from its "
            f"image), so no ground term reaches the panel",
            f"the plate reaches {lo - deck:.2f}–{hi - deck:.2f} m to the car's side (carsim's "
            f"side-wing standoff)")


def _family_card(ui, ctx) -> None:
    """Which AeroBO problem this slot flies: the family, how the slot moves
    only its ride band (ground effect on / off), carsim's air, the objective
    the wing opens on."""
    s = _session(ctx)
    with ui.card("What WingLab flies for this slot"):
        if s is None:
            ui.hint("the slot's session could not be made — see the Output log", "bad")
            return
        op, w = s.op, s.wing
        try:
            fam = w.family
            ui.kv("WingLab family", f"“{fam.base}”")
            ui.kv("registered as", w.family_name)
        except Exception as exc:                        # noqa: BLE001
            ui.hint(f"the family could not be built: {exc}", "bad")
            return
        ge, band = _ground_words(op)
        ui.kv("ground effect", ge)
        ui.kv("ride height", band)
        other = ("the side wing flies the same family with ground effect OFF, on both sides "
                 "(mirrored)" if op.role == "top" else
                 "the top wing flies the same family with ground effect ON over the deck; the "
                 "other side flies this one mirrored")
        ui.kv("the other slots", other)
        objs = w.objectives()
        ui.kv("objective", objs.get(w.choices["objective"], w.choices["objective"]))
        ui.hint("The slot family is WingLab's own problem with ONLY the ride band moved (and "
                "carsim's air): at WingLab's own band it is bit-for-bit WingLab's family. 3 Wing ▸ "
                "Wing type picks what carries the wing (the endplates or two pylons), free chord "
                "law or straight, and the objective.")


def _state_row(ui, ctx) -> None:
    """The state action (AeroBO's Accept mission row): State the mission,
    Published defaults, the STATED tag and the sentence saying what stating
    it now would do."""
    g, dp = ctx.g, ctx.dp
    stated = bool(g.mission.stated)
    keeps = g.mission_restate_keeps()
    resets = not keeps and dp.has_progress()
    if ctx.runs.busy:
        note, kind = LOCKED_LIVE, "warn"
    elif keeps:
        note, kind = STATE_AGAIN, None
    elif resets:
        note, kind = None, None                          # said in full under the row
    else:
        note, kind = STATE_FIRST, None
    with ui.row():
        ui.button("go", "State the mission", icon="check")
        ui.button("dflt", "Published defaults", kind="outline", icon="restart_alt")
        ui.tag("STATED" if stated else "NOT STATED", T.GOOD if stated else T.WARN)
        ui.spacer()
        if note:
            ui.hint(note, kind, split=False)
    if resets and not ctx.runs.busy:
        ui.hint(STATE_RESETS, "warn")


# --------------------------------------------------------------------------- #
#  1 Mission ▸ Design point                                                    #
# --------------------------------------------------------------------------- #
def design(ui, ctx) -> None:
    """The circuit the lap is flown on, corner by corner, and the operating
    point it gives AeroBO: the design speed and CZ, carsim's air, each
    section's Reynolds number and lift, and where the wing flies."""
    g, mp = ctx.g, ctx.mp
    job = mp.job()
    if job == ms.STOPPING:
        _stop_card(ui, ctx)
        s = _session(ctx)
        if s is None:
            ui.hint("the slot's session could not be made — see the Output log", "bad")
            return
        _op_card(ui, ctx, s)
        _section_points_card(ui, ctx, s)
        _where_card(ui, ctx, s)
        return
    try:
        corners = list(mp.profile().corners)
    except Exception:                                   # noqa: BLE001 -- the map says so
        corners = []
    left, right = ui.columns((1, 1))
    with left:
        with ui.card("Circuit map", pad=False):
            poly = _circuit(job)
            if poly is None:
                ui.empty_plot(f"{job}: the circuit does not build", 300)
            else:
                ui.plot(_map_fig(poly, len(corners)), 300)
    with right:
        with ui.card("Corner by corner"):
            _corner_table(ui, mp, corners)
    s = _session(ctx)
    if s is None:
        ui.hint("the slot's session could not be made — see the Output log", "bad")
        return
    _op_card(ui, ctx, s)
    _section_points_card(ui, ctx, s)
    _where_card(ui, ctx, s)


def _stop_card(ui, ctx) -> None:
    """The stop, speed against distance: the car as it stands and with no
    wings (`aero.mission.stop`'s curves) -- the stop's own 'map'."""
    mp = ctx.mp
    v0 = ctx.g.mission.v_stop_kmh
    with ui.card(f"The stop from {v0:.0f} km/h", pad=False):
        if not _stop_ok(mp):
            _no_lap(ui, mp)
            return
        fig = Figure(xlabel="distance since the brakes went on [m]", ylabel="speed [km/h]",
                     legend=True)
        b = mp.bare
        if b is not None and b.ok:
            fig.line(b.d_curve, [3.6 * v for v in b.v_curve], colour=T.INK_MUTED, width=2,
                     dash="dash", name=f"no wings  {b.distance:.2f} m")
        r = mp.result
        fig.line(r.d_curve, [3.6 * v for v in r.v_curve], colour=T.ACCENT, width=2,
                 name=f"as it stands  {r.distance:.2f} m")
        ui.plot(fig, 260)


def _corner_table(ui, mp, corners) -> None:
    if not _lap_ok(mp):
        _no_lap(ui, mp)
        return
    r = mp.result
    rows = [dict(corner=f"T{i + 1}", R=c.radius, V=v, lim=lim,
                 _colours={"lim": T.WARN} if lim == "power" else {})
            for i, (c, v, lim) in enumerate(zip(corners, r.v_corner, r.limited))]
    cols = [dict(key="corner", head="corner", align="left"),
            dict(key="R", head="R [m]", fmt=lambda v: f"{v:.0f}"),
            dict(key="V", head="V [m/s]", fmt=lambda v: f"{v:.2f}"),
            dict(key="lim", head="limited by", align="left")]
    ui.table("corners", cols, rows, empty="the circuit has no corner")
    ui.kv("lap split", f"{r.time:.3f} s = {r.t_corner:.2f} s in corners + "
                       f"{r.t_straight:.2f} s on straights")


def _op_card(ui, ctx, s) -> None:
    """AeroBO's operating point for this slot: V (the lap's mean speed, or
    typed), the design CZ, carsim's air and the dynamic pressure."""
    op = s.op
    with ui.card(f"Operating point — the {_wing_name(ctx, ctx.g.sel)}"):
        ui.field("vauto", control="switch", help=V_TIP)
        ui.field("vdes", control="number", note=op.V_source)
        ui.field("cz", control="number", help=CZ_TIP)
        ui.hairline()
        V = float(op.V)
        ui.kpis([dict(label="design speed", value=f"{V:.2f}", unit="m/s", tip=V_TIP),
                 dict(label="q = ½ρV²", value=f"{0.5 * op.rho * V * V:.0f}", unit="Pa"),
                 dict(label="ρ", value=f"{op.rho:.3f}", unit="kg/m³", tip=AIR_HINT),
                 dict(label="ν", value=f"{op.nu:.2e}", unit="m²/s", tip=AIR_HINT),
                 dict(label="design CZ", value=f"{op.cz_design:.2f}", tip=CZ_TIP)])
        ui.hint(AIR_HINT)


def _section_points_card(ui, ctx, s) -> None:
    """Each section surface's design point, AeroBO's arithmetic at the box
    middle (`SurfaceModel.conditions`): chord, its own Re, the cl -- and the
    Re a screen actually uses (own, or the cached library point)."""
    lib = _library_point()
    rows = []
    for m in (s.af, s.ep):
        if m.target == "plate" and not s.wing.choices["plates"]:
            rows.append(dict(surface="2.8 endplate", chord=None, re_own=None, cl=None,
                             used="— carried by pylons: the plates are a tip device, no section", _colours={"used": T.INK_FAINT}))
            continue
        c = m.conditions()
        if not c:
            rows.append(dict(surface=f"{'2.8 endplate' if m.target == 'plate' else '2 wing section'}",
                             chord=None, re_own=None, cl=None, used="— no such surface"))
            continue
        src = c.get("re_source", "mission")
        used = (f"{c['re']:.3g} (its own Re)" if src == "mission"
                else f"{c['re']:.3g} (the library point)")
        rows.append(dict(surface="2.8 endplate" if m.target == "plate" else "2 wing section",
                         chord=c.get("chord"), re_own=c.get("re_own"), cl=c.get("cl_design"),
                         used=used))
    cols = [dict(key="surface", head="surface", align="left"),
            dict(key="chord", head="chord [m]", fmt=lambda v: vc.num(v, "{:.3f}")),
            dict(key="re_own", head="its own Re", fmt=lambda v: vc.num(v, "{:.3g}")),
            dict(key="cl", head="design cl", fmt=lambda v: vc.num(v, "{:.2f}")),
            dict(key="used", head="screened at Re", align="left")]
    with ui.card("Section design points"):
        ui.table("mission.sections", cols, rows, sortable=False)
        ui.hint(SECTION_POINT_HINT)
        if lib:
            ui.kv("cached library point", f"Re {float(lib.get('re', 0.0)):.3g} · Mach "
                                          f"{float(lib.get('mach', 0.0)):g} · α "
                                          f"{min(lib.get('alphas') or [0]):g}…{max(lib.get('alphas') or [0]):g}°")
        ui.hint(LIB_POINT_HINT)


def _where_card(ui, ctx, s) -> None:
    """Where the slot's wing flies (the slot family's rows): the deck and
    ride band, or the standoff and the pushed-away image; the flank's size
    rows cut to AeroBO's AR >= 3."""
    op = s.op
    ge, band = _ground_words(op)
    with ui.card("Where the wing flies"):
        ui.kv("slot", f"x {op.x:+.2f} m · h {op.h:.2f} m ({_slot_label(ctx, op.slot)})")
        ui.kv("ground effect", ge)
        ui.kv("ride height row", band)
        ui.kv("span limit", am.bridge.limit_words(op.role, op.car, op.h, op.unlimited),
              tip="task 41, drive/bodies.py: a side wing's lower tip at the car's ground clearance, a "
                  "top wing 1.2 × the body's width; Settings ▸ Wing limits: Unlimited allows 3 ×")
        for lab, (lo, hi) in (op.size_rows or {}).items():
            unit = "m²" if lab.startswith("S") else "m"
            ui.kv(f"{lab} row", f"{float(lo):.3f}–{float(hi):.3f} {unit}  (the {op.car}'s span "
                                f"limit, cut to WingLab's AR ≥ 3)")
        if op.role == "top":
            ui.hint("The top wing searches its ride height inside the band and writes the winner "
                    "back to the slot, as it does the incidence.")
        else:
            ui.hint("A side wing stands off the car's side: its endplate is what reaches the "
                    "body. With ground effect off, carsim's old body-image (the wall) is not "
                    "modelled — the owner's choice. One chordwise panel is honest only above "
                    "AR 3 (WingLab's rule), so the side wing designs taller, narrower panels.")


# --------------------------------------------------------------------------- #
#  1 Mission ▸ Search & budget                                                 #
# --------------------------------------------------------------------------- #
def search(ui, ctx) -> None:
    """Where the budgets come from (AeroBO's measured plan, or yours), how
    converged a run aims to be, what that means at each stage -- optimiser,
    budget, Sobol start, the stop rule, the expected time and the plan's own
    reason -- and where the study behind it comes from."""
    s = _session(ctx)
    form = ui.form
    pol = s.policy if s is not None else None
    own = pol is not None and not pol.recommended()
    with ui.card("Where the search settings come from"):
        ui.radio_rows("smode", SMODE_ROWS)
    with ui.card("How converged"):
        ui.field("effort", control="select", label="aim for")
        ui.kv("budgets by effort", _effort_budgets(s, form))
        ui.hint(CONVERGED_HINT)
        ui.switch("stopconv")
        ui.hint(_reach_text(s), split=False, help=STOP_HELP)
    rows = pol.plans(s) if s is not None else []
    with ui.card("What that means, stage by stage"):
        if own:
            ui.hint(OWN_HINT)
        _plans_table(ui, rows, _wing_split(s, rows))
        if own:
            for r in rows:
                key = STAGE_OWN_KEY.get(r.get("stage"))
                if key and form.param(key) is not None:
                    ui.field(key, control="number", unit="evaluations")
        else:
            smode = form.param("smode")
            ui.link("mine", "use these as my own values instead", "edit",
                    lambda: smode.set("own"), enabled=form.is_enabled(smode))
        if rows and ui.disclosure("mission.why", "why these budgets — WingLab's own reasons"):
            with ui.indent():
                for r in rows:
                    if r.get("why"):
                        ui.kv(r["stage"], "")
                        ui.hint(r["why"], split=False)
        note = s.wing.bo_note() if s is not None else ""
        if note:
            ui.hint(note, "warn")
    _provenance_card(ui, rows)


def _effort_budgets(s, form) -> str:
    """"quick 109 / 42 · balanced 164 / 53 · thorough 240 / 87 (a section /
    the wing)": the budgets each effort's plans give (AeroBO's
    `recommended_search`, cached by the bridge), the chosen one first."""
    p = form.param("effort")
    efforts = list(p.choices) if p is not None else []
    bits = []
    for e in efforts:
        try:
            sec = int(am.bridge.section_plan(s.af.opt["objective"], e).budget)
            w = s.wing
            wing = int(am.bridge.wing_plan(w.family_name, w.physics_flags(), w.bounds_overrides(),
                                           w.pinned() or None, e).budget)
            bits.append(f"{e} {sec} / {wing}")
        except Exception:                               # noqa: BLE001 -- the bare name
            bits.append(f"{e} —")
    return " · ".join(bits) + "  (a section / the wing)" if bits else "—"


def _wing_split(s, rows):
    """(n_init, n_a) the wing's bo_slsqp run will fly (AeroBO's own
    arithmetic, `bridge.handoff_split`), or None."""
    w = next((r for r in rows if r.get("stage") == "3 Wing" and r.get("dim")), None)
    if s is None or w is None:
        return None
    try:
        return am.bridge.handoff_split(s.wing.cfg(), int(w["dim"]))
    except Exception:                                   # noqa: BLE001 -- the plan's own column
        return None


def _reach_text(s) -> str:
    """Can the wing's stop rule fire at its budget (`convergence_rule_reach`),
    and the sections' lack of one."""
    if s is None:
        return "—"
    try:
        eff = s.wing.search()
        r = am.bridge.convergence_rule_reach(eff)
    except Exception:                                   # noqa: BLE001
        r = None
    if not s.policy.stop_when_converged:
        return "Off: every run spends its whole budget unless you stop it."
    if not s.policy.recommended():
        return ("Your own values: WingLab's stop rule belongs to its measured plan, so every run "
                "spends the budget you set unless you stop it.")
    if r is None:
        return "The wing's plan has no adaptive rule here; the sections never have one."
    if r["can_fire"]:
        return (f"The wing can stop early at {r['budget']}: its rule needs {r['needs']} evaluations "
                f"(patience {r['patience']}). The sections have no rule — they fly their budget.")
    return (f"At a budget of {r['budget']} the wing's rule cannot fire: it needs {r['needs']} "
            f"evaluations (patience {r['patience']}).")


def _plans_table(ui, rows, split=None) -> None:
    """One row per stage: AeroBO's plan -- variables, optimiser, budget,
    Sobol start, stop rule, the expected wall time. `split` is the wing's
    (n_init, n_a) under bo_slsqp (`bridge.handoff_split`)."""
    out = []
    for r in rows:
        if r.get("error"):
            out.append(dict(stage=r["stage"], dim=None, optimiser="—", budget=None, n_init=None,
                            rule=f"no plan: {r['error']}", est=None,
                            _colours={"rule": T.BAD}))
            continue
        pat, tol = r.get("patience"), r.get("tol")
        rule = (f"patience {pat}, tol {100.0 * float(tol):.1f} %" if pat is not None and tol is not None
                else "none — the budget")
        opt, n_init = r.get("optimiser") or "—", r.get("n_init")
        if opt == "bo_slsqp" and split:
            #  what the run flies, not the recommender's plain-BO n_init: the
            #  handoff's BO leg owns a share of the budget (AeroBO's split)
            n_init, opt = split[0], f"bo_slsqp (SLSQP after {split[1]})"
        out.append(dict(stage=r["stage"], dim=r.get("dim"), optimiser=opt, budget=r.get("budget"),
                        n_init=n_init, rule=rule, est=r.get("est_s"), _colours={"budget": T.ACCENT}))
    cols = [dict(key="stage", head="stage", align="left"),
            dict(key="dim", head="variables", fmt=lambda v: "—" if v is None else f"{int(v)}"),
            dict(key="optimiser", head="optimiser", align="left"),
            dict(key="budget", head="budget", fmt=lambda v: "—" if v is None else f"{int(v)}"),
            dict(key="n_init", head="Sobol start", fmt=lambda v: "—" if v is None else f"{int(v)}"),
            dict(key="rule", head="stop rule", align="left"),
            dict(key="est", head="expected", fmt=lambda v: vc.duration(v) if vc.finite(v) else "—")]
    ui.table("mission.plans", cols, out, sortable=False, empty="no plan — state the mission first")
    ui.hint("expected = WingLab's measured cost per candidate × the budget (the study's machine, not "
            "a measurement of this one). A section candidate is a live XFOIL sweep; a wing "
            "candidate a lattice solve.")


def _provenance_card(ui, rows) -> None:
    """Where the budgets come from: AeroBO's measured study."""
    stamp = next((r.get("stamp") for r in rows if r.get("stamp")), None) or {}
    with ui.card("Provenance"):
        if not stamp:
            ui.hint("WingLab's study stamp is not available")
            return
        ui.kv("measured", str(stamp.get("measured", "—")))
        ui.kv("runs · cases", f"{stamp.get('n_runs', '—')} runs over {stamp.get('cases', '—')} cases")
        ui.kv("on", str(stamp.get("machine", "—")))
        ui.kv("source", "WingLab v1.0.0's own study of its budgets")
        if stamp.get("note"):
            ui.hint(str(stamp["note"]))
