"""drive/aerobo_models.py -- the garage DESIGN page's models, on AeroBO's own
engine (PLAN2 §7).

The owner asked for AeroBO -- the car half of it -- inside the game, with
only the mission and the looks carsim's. So nothing on the design page
designs anything any more: every screen, section search, wing search, law
and report here is a call into the VENDORED engine (`aerobo/src/aerobo`,
unmodified) through `drive/aerobo_bridge.py`, made with the arguments
AeroBO's own V3 GUI sends, on a worker thread (`design_jobs.EngineJob`).
What stays carsim's is the mission (the circuit, the surface and the slot
supply AeroBO's operating point) and what the car does with the result (the
affine law `vehicle.DevAero` / `TopAero` read, sampled from AeroBO's
evaluator at the winning design).

The objects, one set per SLOT (`DesignSession`; the right flank is the left
one mirrored), and the API the views and the shell code against:

    SearchPolicy     Mission > Search & budget (AeroBO `_search_state`):
                     mode recommended | own, effort, stop_when_converged,
                     own budgets {"airfoil", "plate", "wing"}; plans() rows
    DesignSession    key, role, op (bridge.OperatingPoint), policy, af, ep,
                     wing, results, runs; state(stage) -> (state, reason);
                     invalidate(why), open_for(signature), set_V(v), set_cz(v)
    SurfaceModel     one section surface (AeroBO `_airfoil_workspace`):
                     target "main" | "plate"; weights (+source), screen
                     gates, floors, re_source, top_n, shortlist; screen,
                     ranked, chosen, decision, opt; conditions(),
                     start_screen(), use_ranked(i), start_optimise(extend),
                     use_optimised(), decline(), finished(), flag_value(),
                     redundant(), dead(), columns(), graph(), state/reason
    WingModel        the wing (stage 3): choices, family, family_name, box,
                     released, fixed, record, records, outcome, law, spec,
                     dirty; objectives(), physics_flags(), bounds_overrides(),
                     pinned(), plan(), search(), cfg(), start_run(extend),
                     verdict(), fit(), commit(name), graph()
    ResultsModel     report (api.design_report), summary(), car_lap()

Forms (`cae.form.Form`, what a view binds its controls to), by view:

    *.screen   SurfaceModel.screen_params: resrc (mission | library), re, cl,
               w.<crit> (main: ldcr clmax cm ldmax thick astall -- NO cdcr,
               the owner's call: "cd at the design cl" is "L/D at the design
               cl" on a wing; plate: + cdcr, with ldcr/ldmax/cm dead at
               cl 0), rec, [main] gtc, tcmin, gcm, cmmax, fclmax, fldcr,
               fastall, go (screen), stop, [plate] decline
    *.section  SurfaceModel.params: use, decline, refine
    *.opt      SurfaceModel.opt_params: obj, [main] gotc, otc, gocm, ocm, budget,
               seed, run, stop, more, go (keep going), use
    w.type     WingModel.type_params: plates ("carried by": endplates |
               pylons), law, blend, blend_frac, cant, cant_deg, tip, obj,
               cap, floor, to1, to2, to28
    w.box      WingModel.box_params(): bx.<label>.min/.max/.con/.fix per row
               of the built problem but endplate_tc (NO_BOX_CONTROLS), boxreset
    w.solver   WingModel.solver_params: opt, budget, seed, stopconv, run,
               stop, snip, mine, s1
    w.conv     WingModel.conv_params: more, go (keep going), stop, fit
    r.*        ResultsModel.params: keep_r, save, to3, tobox

The tutorial pins (drive/wing_tutorial.py) are compat names here:
`SurfaceModel.ranked` (truthy after a screen), `finished()`, `use_ranked(i)`,
`decline()`; `WingModel.spec.name`, `dirty`, `key`.

THE JOBS. Every engine call longer than a frame is an `EngineJob` on the
garage's one `RunManager`: its runner is a bridge factory that freezes every
argument at launch (the job flies a snapshot; the form stays editable), or --
in the self-checks, the design shell's check and the screenshot harness -- a
`design_jobs.ReplayRunner` over a captured fixture (`use_fixtures`). The
outcome a view draws its DONE / CONVERGED / STOPPED / FAILED tag from is the
one the model STORES when the job finishes (`job.outcome()`), never a
finished job read later.

Imports (PLAN2 §3): the bridge, design_jobs, garage_ui, cae.form, drive.aero,
drive.track -- never `garage` (the self-check reaches it lazily for the three
rows that are about the garage's own physics readers, as garage's own check
reaches `render`).

    python3 -m drive.aerobo_models          # the self-check, M1-M37
"""

from __future__ import annotations

import contextlib
import json
import math
import os

import numpy as np

from . import aerobo_bridge as bridge
from . import design_jobs as dj
from . import garage_ui as ui
from .aero import mission as ms
from .cae.form import Form
from .track import make_track

# --------------------------------------------------------------------------- #
#  the tables                                                                  #
# --------------------------------------------------------------------------- #
#: the stages a session gates, AeroBO's order (design_shell.STAGES keys)
STAGES = ("m", "af", "ep", "w", "r")
#: the captured runs the deterministic checks replay (A's `--capture`)
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "aerobo_fixtures")

#: the screening criteria, AeroBO's order (`api.SCREEN_METRICS`), with the
#: form's label and the table's column heading and format
CRITERIA = (
    ("ldcr", "L/D at the design cl", "L/D@cl", "{:5.1f}"),
    ("clmax", "cl max", "clmax", "{:4.2f}"),
    ("cm", "|cm| at the design cl", "|cm|", "{:5.3f}"),
    ("ldmax", "(L/D) max", "L/Dmax", "{:5.1f}"),
    ("thick", "thickness t/c", "t/c", "{:5.3f}"),
    ("astall", "stall angle", "astall", "{:+5.1f}"),
    ("cdcr", "cd at the design cl", "cd@cl", "{:6.4f}"),
)
#: the metric of a ranked row each criterion is READ from (api._screen_row)
CRITERION_METRIC = {"ldcr": "ldcr", "clmax": "clmax", "cm": "cm_at", "ldmax": "ldmax",
                    "thick": "tc", "astall": "astall", "cdcr": "cd_at"}
#: ON THE WING, "cd at the design cl" IS "L/D at the design cl": at a stated
#: lift one is the other's reciprocal times that lift, so weighting both
#: counts one question twice. The owner's call (2026-09-24, handoff D9):
#: the wing's screen has no cdcr weight and no cdcr column. AeroBO's own wing
#: weights already give it 0. The PLATE keeps it -- at cl 0 the two ratios
#: are read at a lift a plate never carries, and the drag is what is left.
REDUNDANT = {"main": {"cdcr": "the same question as L/D at the design cl on a wing: "
                              "at a stated lift one is the other's reciprocal"},
             "plate": {}}
#: ...and on the PLATE three criteria are dead: its design lift is ZERO, so
#: both L/D ratios and the moment are read at a lift it never carries
DEAD = {"main": {},
        "plate": {"ldcr": "L/D at cl 0 is 0 for every section: a plate carries no design lift",
                  "ldmax": "a plate is never flown at its best L/D: its design lift is zero",
                  "cm": "|cm| at cl 0 prices camber a symmetric plate does not have"}}

#: the section objectives offered (bridge.SECTION_OBJECTIVES), in words
SECTION_OBJECTIVE_WORDS = {
    "composite_goal": "the screen's weighted score, held to the seed on every criterion",
    "composite": "the screen's weighted score (plain sum)",
    "composite_asf": "the screen's criteria, worst-first (achievement scalarising)",
    "cd": "2-D L/D at the design cl",
}

#: AeroBO's car objective menu (`gui/nice_app.CAR_OBJECTIVE_LABELS`), same
#: keys, same order. NOT offered: `cz` and `cd` -- the area is a design
#: variable, and a coefficient referenced to the very area being searched is
#: refused by the engine itself (`carwing`). The flank relabels downforce as
#: SIDE force: it is the same scalar, pointed into the corner.
CAR_OBJECTIVES = (
    ("efficiency", "efficiency CZ/CD (= force per unit drag)"),
    ("downforce", "downforce, in newtons"),
    ("drag", "drag, in newtons (minimised — state a force floor below)"),
    ("laptime", "lap time round {circuit} (minimised)"),
    ("downforce_plus_drag", "downforce + drag, in newtons (the total load)"),
)
#: The objective 3 Wing opens on for each JOB (`DesignSession.job`), set when
#: the job is stated or changes; the player can pick any other. The owner,
#: 2026-09-25, asked to "maximise different things (drag, side force)": a
#: stop's is downforce + drag, which ranks wings exactly as the stop does at
#: mu = 1 (`aero.mission.stop`). A circuit keeps AeroBO's own default,
#: efficiency -- for the top wing; a side wing opens on SIDE_OBJECTIVE. A
#: side wing's stop opens on the same key, side force + drag (`SIDE_STOP_NOTE`).
JOB_OBJECTIVE = {ms.STOPPING: "downforce_plus_drag"}
#: ...and a side wing's on a circuit, whatever the circuit: side force (the
#: flank's "downforce", `bridge.car_objectives` relabels it)
SIDE_OBJECTIVE = "downforce"

#: which LIMIT each objective is the missing half of (AeroBO's
#: CAR_OBJECTIVE_WANTS_LIMIT): a ratio or a drag wants a force floor, a force
#: wants a drag ceiling; the lap prices both itself
WANTS_LIMIT = {"efficiency": "floor", "drag": "floor", "downforce": "cap",
               "downforce_plus_drag": "cap"}
#: AeroBO's notes under the select, one per objective
OBJECTIVE_NOTE = {
    "efficiency": "well-posed with the area free: the ratio does not run to the area's edge",
    "downforce": "without a drag ceiling this runs to the edges of the box (area, incidence) "
                 "and to stall",
    "drag": "wants a force floor — the least drag is no wing at all",
    "laptime": ("WingLab's point-mass lap (cartrack) on carsim's {circuit} geometry with the "
                "Corsa's mass, power and CdA; carsim's own two-track lap on Results is a "
                "different model and will not match to the tenth"),
    "downforce_plus_drag": "a ratchet: without a ceiling it takes the whole area row",
}
#: the note under the objective when the job is STOPPING and the objective its own
STOP_NOTE = ("the stop's own objective: a stop decelerates at μ·(m·g + downforce) + drag, and "
             "both wing forces scale with V², so the wing with the most μ·downforce + drag "
             "stops shortest — this ranking exactly at μ = 1 (dry); on damp or wet it rates "
             "downforce a little too high")
#: ...and on a side wing's stop, which is the air brake (`aero.mission.
#: SIDE_STOP_FLANKS`): WingLab has no "most drag" objective (its drag is
#: minimised), and side force + drag is the one that grows with drag
SIDE_STOP_NOTE = ("the side wing's stop is the air brake: both panels out, their side forces "
                  "cancel and their drags add, so the stop pays for drag alone. WingLab has no "
                  "“most drag” objective (its drag is minimised); side force + drag is the one "
                  "that grows with it — both run to the biggest panel at the highest lift, "
                  "where the drag is")

#: the default action labels of the three tool-bar verbs a run takes
STOP_TEXT = "Stop — keeps the best so far"

#: box rows the player gets NO controls for (no band, constrain or fix): the
#: owner, 2026-09-25, "endplate t/c shouldn't be given as an option". The
#: plate's t/c follows the section taken in 2.8 (`WingModel.auto_pins`); with
#: the family's own plate kept, AeroBO searches it itself (AeroBO parity)
NO_BOX_CONTROLS = frozenset({"endplate_tc"})
#: the plate's criterion weights the player gets no slider for, for the same
#: reason: its thickness weight stays AeroBO's own (`bridge.PLATE_WEIGHTS`)
PLATE_FIXED_WEIGHTS = frozenset({"thick"})
#: who fixed the plate's height when the pylons' tip device is "none": the
#: card, not the player -- releasing it IS picking another device
TIP_PIN_SOURCE = "tip device: none"


# --------------------------------------------------------------------------- #
#  small helpers                                                               #
# --------------------------------------------------------------------------- #
class _LiveLabel(ui.Param):
    """A row whose LABEL is a sentence about the model ("Keep going — 82
    more (246 in total)"), worked out whenever it is read: a label written
    once would go on quoting the run before last. garage.py has the same
    class; this module may not import garage."""

    def __init__(self, key, label_fn, *args, **kwargs):
        self._label_fn = label_fn
        super().__init__(key, "", *args, **kwargs)

    @property
    def label(self) -> str:
        return self._label_fn()

    @label.setter
    def label(self, _value) -> None:
        pass


def _finite(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _loop_tc(coords) -> float | None:
    """The thickness of a closed section loop, as a fraction of chord: the
    largest upper-minus-lower gap at one station (both surfaces resampled
    onto a common x grid)."""
    c = np.asarray(coords, dtype=float)
    if c.ndim != 2 or c.shape[0] < 8:
        return None
    i = int(np.argmin(c[:, 0]))
    up, lo = c[: i + 1][::-1], c[i:]
    x = np.linspace(0.0, 1.0, 101)
    try:
        yu = np.interp(x, up[:, 0], up[:, 1])
        yl = np.interp(x, lo[:, 0], lo[:, 1])
    except ValueError:
        return None
    return float(np.max(np.abs(yu - yl)))


_LIB_COUNT: dict = {}


def _library_count(target: str) -> int:
    """How many sections a library-point screen of `target` reads: AeroBO's
    symmetric sections for the plate, every UIUC section for the wing (the
    count its report's `n_screened` will give). Counted once."""
    if target not in _LIB_COUNT:
        try:
            if target == "plate":
                _LIB_COUNT[target] = len(bridge.api.symmetric_section_names())
            else:
                _LIB_COUNT[target] = len(list((bridge.AEROBO_ROOT / "data" / "airfoils" / "uiuc")
                                              .glob("*.dat")))
        except Exception:                                   # noqa: BLE001 -- a count for a chip
            _LIB_COUNT[target] = 0
    return max(1, int(_LIB_COUNT[target]))


def _clean(o):
    """`json.dump`'s fallback for what a record holds (numpy arrays and
    scalars); anything else is written as its text."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def _same_section(lib, a) -> str | None:
    """The name of a library section with exactly `a`'s shape (same source,
    same coordinates or NACA code), or None: a wing committed twice from one
    run -- renamed, or saved on the way out -- must not add a copy of its
    section to the library each time."""
    for name, b in getattr(lib, "airfoils", {}).items():
        if b.source != a.source:
            continue
        if a.source == "naca" and b.code == a.code:
            return name
        if a.source == "coords" and b.points and b.points == a.points:
            return name
    return None


def outcome_tag(outcome: dict | None) -> tuple | None:
    """The terminal tag a view draws from a STORED outcome (PLAN2 §6.2):
    `(text, theme colour, note)` -- "DONE · 164/164" GOOD, "CONVERGED ·
    71/164" GOOD ("stopped improving"), "STOPPED · 12/164" WARN ("best
    kept"), "FAILED" BAD (the error's first line) -- or None before any run.
    design_jobs' own table, so a job's chip and a model's tag never drift."""
    return dj.terminal_tag(outcome)


# --------------------------------------------------------------------------- #
#  replay: where the jobs come from in the deterministic checks                #
# --------------------------------------------------------------------------- #
class Replay:
    """The captured runs (`drive/data/aerobo_fixtures/*.json`, PLAN2 §6.4)
    replayed through the SAME EngineJob path by `design_jobs.ReplayRunner`.

    `runner(kind, **want)` picks the fixture of that kind whose `meta.args`
    match the most of `want` (surface / role / variant / re_source /
    objective) and hands back a ReplayRunner over it; `pause_at` freezes a
    run at k evaluations (the screenshot harness's "running" state, by
    construction rather than by timing). A request no fixture answers is an
    error the job reports as FAILED -- a check must never quietly fall back
    to a live XFOIL run.

    The law and the design report are not replayed: both are pure functions
    of a record, a few hundred milliseconds of AeroBO's own evaluator with
    no XFOIL, so the checks run them for real on the replayed record."""

    def __init__(self, root: str = FIXTURES, pause_at: int | None = None):
        self.root = root
        self.pause_at = pause_at
        self._index: list | None = None
        self.last = None                     # the last ReplayRunner handed out

    def index(self) -> list:
        """`[(path, kind, args)]` of every fixture under `root`, read once."""
        if self._index is None:
            out = []
            try:
                names = sorted(f for f in os.listdir(self.root) if f.endswith(".json"))
            except OSError:
                names = []
            for f in names:
                p = os.path.join(self.root, f)
                try:
                    with open(p) as fh:
                        d = json.load(fh)
                except (OSError, ValueError):
                    continue
                meta = d.get("meta") or {}
                args = dict(meta.get("args") or {})
                args.setdefault("name", f[:-5])
                out.append((p, str(d.get("kind", "")), args))
            self._index = out
        return self._index

    def find(self, kind: str, **want) -> str | None:
        best, score = None, -1
        for p, k, args in self.index():
            if k != kind:
                continue
            s = 0
            for key, v in want.items():
                if v is None:
                    continue
                if key in args:
                    if args[key] != v:
                        s = -10 ** 6          # a stated mismatch rules it out
                        break
                    s += 1
            if s > score:
                best, score = p, s
        return best if score >= 0 else None

    def runner(self, kind: str, **want):
        path = self.find(kind, **want)
        if path is None:
            wanted = ", ".join(f"{k}={v}" for k, v in want.items() if v is not None)

            def _missing(emit, stop):
                raise RuntimeError(f"no captured {kind} fixture for {wanted or 'this run'} "
                                   f"under {self.root} (python3 -m drive.aerobo_bridge --capture)")
            return _missing
        self.last = dj.ReplayRunner(path, pause_at=self.pause_at)
        return self.last


_REPLAY: Replay | None = None


def use_fixtures(root: str = FIXTURES, pause_at: int | None = None) -> Replay:
    """Every model job from now on replays the captured fixtures under
    `root` (the self-checks, design_shell, design_shots). Returns the Replay,
    whose `pause_at` may be moved between runs."""
    global _REPLAY
    _REPLAY = Replay(root, pause_at)
    return _REPLAY


def use_engine() -> None:
    """Back to AeroBO's engine for every job (the default)."""
    global _REPLAY
    _REPLAY = None


def replay() -> Replay | None:
    return _REPLAY


@contextlib.contextmanager
def replaying(root: str = FIXTURES, pause_at: int | None = None):
    """`with replaying(): ...` -- fixtures inside, whatever was there after."""
    global _REPLAY
    was = _REPLAY
    rp = use_fixtures(root, pause_at)
    try:
        yield rp
    finally:
        _REPLAY = was


# --------------------------------------------------------------------------- #
#  the search policy (Mission > Search & budget)                               #
# --------------------------------------------------------------------------- #
class SearchPolicy:
    """AeroBO's `_search_state`, one per garage: where the budgets come from.

    RECOMMENDED is AeroBO's measured plan and nothing else --
    `api.recommended_search` over `aerobo/data/search_budget.json` (1260
    runs, 19 cases, measured 2026-08-05): an 8-D section 109 / 164 / 240 at
    quick / balanced / thorough, the 14-D wing 42 / 53 / 87. OWN hands the
    three budgets to the player (they open on the balanced plan's). carsim's
    own budget law (40 / 37 / 31) is gone from this page: those numbers
    sized carsim's search, and that search is not what runs here."""

    MODES = ("recommended", "own")
    EFFORTS = ("quick", "balanced", "thorough")

    def __init__(self):
        self.mode = "recommended"
        self.effort = "balanced"
        #: stop as soon as the search stops improving (AeroBO's
        #: ConvergenceStop at the plan's patience and tolerance); the budget
        #: stays the backstop. Sections have no adaptive rule in the plan
        #: (patience None), so this reaches the WING only.
        self.stop_when_converged = True
        self.own = {"airfoil": 164, "plate": 164, "wing": 53}
        self.cache: dict = {}

    def set_mode(self, mode: str) -> bool:
        if mode not in self.MODES or mode == self.mode:
            return False
        self.mode = mode
        return True

    def set_effort(self, effort: str) -> bool:
        if effort not in self.EFFORTS or effort == self.effort:
            return False
        self.effort = effort
        self.cache.clear()                    # every plan is a function of it
        return True

    def recommended(self) -> bool:
        return self.mode == "recommended"

    def section_search(self, objective: str, surface: str = "main", seed: int = 0) -> dict:
        """AeroBO's `effective_airfoil_search` for one surface: the plan's
        optimiser (a non-torch one without the BO stack), budget, n_init and
        refusal -- or the player's budget with the plan's decisions."""
        return dict(bridge.section_search(self, objective, surface, seed))

    def wing_search(self, plan, family: str, seed: int = 0) -> dict:
        """AeroBO's `effective_wing_search`: bo_slsqp at the plan's budget
        when the family and this machine take it -- or the player's budget."""
        return dict(bridge.wing_search(self, plan, family, seed))

    def plans(self, session: "DesignSession") -> list:
        """The Search & budget table: one row per stage -- dimension,
        optimiser, budget, n_init, the adaptive rule, the estimated wall time
        and the plan's own `why` -- plus the study stamp on every row."""
        rows = []
        stamp = bridge.study_stamp()
        for stage, m in (("2 Airfoil", session.af), ("2.8 Endplate", session.ep)):
            if m.target == "plate" and not session.wing.choices["plates"]:
                continue
            try:
                eff = self.section_search(m.opt["objective"], m.target)
                plan = eff.get("plan")
            except Exception as exc:                  # noqa: BLE001 -- a read-out
                rows.append(dict(stage=stage, error=f"{type(exc).__name__}: {exc}", stamp=stamp))
                continue
            rows.append(dict(stage=stage, dim=getattr(plan, "dim", None),
                             optimiser=eff.get("optimiser"), budget=int(eff.get("budget") or 0),
                             n_init=eff.get("n_init"), patience=getattr(plan, "patience", None),
                             tol=getattr(plan, "tol", None),
                             est_s=getattr(plan, "est_seconds", None),
                             why=getattr(plan, "why", ""), source=eff.get("source"), stamp=stamp))
        w = session.wing
        try:
            plan = w.plan()
            eff = w.search()
            rows.append(dict(stage="3 Wing", dim=getattr(plan, "dim", None),
                             optimiser=eff.get("optimiser"), budget=int(eff.get("budget") or 0),
                             n_init=eff.get("n_init") or getattr(plan, "n_init", None),
                             patience=getattr(plan, "patience", None),
                             tol=getattr(plan, "tol", None),
                             est_s=getattr(plan, "est_seconds", None),
                             why=getattr(plan, "why", ""), source=eff.get("source"), stamp=stamp))
        except Exception as exc:                      # noqa: BLE001
            rows.append(dict(stage="3 Wing", error=f"{type(exc).__name__}: {exc}", stamp=stamp))
        return rows


# --------------------------------------------------------------------------- #
#  one section surface (stage 2 main / stage 2.8 plate)                        #
# --------------------------------------------------------------------------- #
class SurfaceModel:
    """AeroBO's `_airfoil_workspace` for one surface: SCREEN the library at
    the surface's own design point, read the RANKING, take a SECTION, or
    OPTIMISE its shape -- AeroBO's four views, on AeroBO's engine.

    `target` "main" is the wing's section (stage 2), "plate" the endplates'
    (stage 2.8). The plate is SYMMETRIC by the owner's rule: it screens only
    `api.symmetric_section_names()` (229 of 2174, camber <= 0.5 % c) and its
    shape search is AeroBO's symmetric CST (`w_lower == -w_upper`) at cl 0.

    `decision` is what finishes the stage (AeroBO's): "library" (a ranked
    section taken), "optimised" (a search's winner taken) or "default" (the
    family's own section kept -- the published one for the wing, NACA 00tt
    at the searched t/c for the plate). Undecided, the wing flies the
    family's own, which is a complete design: stage 2 is not a gate."""

    def __init__(self, session: "DesignSession", target: str):
        self.session, self.target = session, target
        self.weights = dict(bridge.PLATE_WEIGHTS if target == "plate" else bridge.WING_WEIGHTS)
        for k in REDUNDANT[target]:
            self.weights.pop(k, None)
        self.weights_source = "recommended"
        self.screen_gates = dict(bridge.SCREEN_GATES)
        self.screen_gates_on = {"tc_min": True, "cm_max": True}
        self.floors = {"clmax": None, "ldcr": None, "astall": None}
        self.re_source = "mission"
        self.top_n = int(bridge.TOP_N)
        self.shortlist = int(bridge.SHORTLIST)
        self.screen = {"report": None, "candidates": None, "outcome": None,
                       "surface": target, "cond": None, "re_source": None}
        self.ranked: list = []
        self.chosen: dict | None = None
        self.decision: str | None = None
        self.highlight = 0                   # the rank row the section view shows
        self.opt = {"objective": bridge.SECTION_OBJECTIVES[0], "gates": dict(bridge.OPT_GATES),
                    "gates_on": {"tc_min": True, "cm_max": True}, "seed": 0,
                    "report": None, "launch": None, "records": [], "outcome": None,
                    "score": None, "score_outcome": None, "prior": 0, "more": None,
                    "anchor_name": None}
        self.polars: dict = {}               # value-key -> section polar (the section view)
        self._live_n_init = None             # the Sobol block of a live fresh search
        self.msg = ""
        self.screen_params = Form(self._screen_rows(), title="")
        self.params = Form(self._section_rows(), title="")
        self.opt_params = Form(self._opt_rows(), title="")

    # -- names ------------------------------------------------------------------
    @property
    def stage(self) -> str:
        return "ep" if self.target == "plate" else "af"

    @property
    def label(self) -> str:
        return "endplate" if self.target == "plate" else "wing"

    def redundant(self) -> set:
        """Criteria this surface does not ask (their row and column are not
        drawn): the wing's cd@cl (the owner's D9)."""
        return set(REDUNDANT[self.target])

    def dead(self) -> dict:
        """Criteria that cannot rank this surface, and why: the plate's
        ldcr, ldmax and |cm| at cl 0 (drawn dim, weight held at 0)."""
        return dict(DEAD[self.target])

    def criteria(self) -> list:
        """The criteria this surface's form and table carry, AeroBO's order."""
        return [c for c in CRITERIA if c[0] not in self.redundant()]

    def columns(self) -> list:
        """The ranking table's criterion columns `(key, heading, fmt)` --
        the wing's has no cd@cl (M7)."""
        return [(k, head, fmt) for k, _lab, head, fmt in self.criteria()]

    # -- the design point ----------------------------------------------------------
    def effective_re_source(self) -> str:
        """"mission" (a live XFOIL sweep of the shortlist at this surface's own
        Re) or "library" (the cached library point, instant). Without XFOIL
        the library point is the only answer (PLAN2 D11)."""
        return self.re_source if bridge.xfoil_ok() else "library"

    def conditions(self) -> dict:
        """`{re, mach, cl_design, chord, re_own, re_source}` this surface is
        screened and designed at: AeroBO's arithmetic
        (`bridge.section_conditions`) at the middle of the box the wing will
        search -- the plate's chord is a ratio of the wing's tip chord -- or
        the library point's Re. {} where the family has no such surface
        (plain fences carry no plate section)."""
        s = self.session
        try:
            box = s.wing.family_box()
            c = bridge.section_conditions(s.op, box, self.target,
                                          re_source=self.effective_re_source())
        except Exception:                                   # noqa: BLE001 -- a read-out
            c = None
        return dict(c) if c else {}

    # -- the forms -----------------------------------------------------------------
    def _set_w(self, crit: str):
        def f(v):
            if crit in self.dead():
                return
            self.weights[crit] = max(0.0, float(v))
            self.weights_source = "user"
        return f

    def recommend(self) -> None:
        """The weights back to AeroBO's recommended set for this surface."""
        self.weights = dict(bridge.PLATE_WEIGHTS if self.target == "plate" else bridge.WING_WEIGHTS)
        for k in self.redundant():
            self.weights.pop(k, None)
        self.weights_source = "recommended"

    def _gate_get(self, which: dict, on: dict, key: str):
        return lambda: float(which.get(key, bridge.GATE_OFF[key]))

    def _gate_set(self, which: dict, key: str):
        def f(v):
            which[key] = float(v)
        return f

    def _gate_switch(self, on: dict, key: str):
        def f(v):
            on[key] = bool(v)
        return f

    def gates(self, kind: str = "screen") -> dict:
        """The gates a run is sent: the value where switched on, AeroBO's
        `GATE_OFF` where switched off."""
        vals, on = ((self.screen_gates, self.screen_gates_on) if kind == "screen"
                    else (self.opt["gates"], self.opt["gates_on"]))
        return {k: (float(vals[k]) if on.get(k, True) else float(bridge.GATE_OFF[k]))
                for k in ("tc_min", "cm_max")}

    def _floor_set(self, key: str, off: float):
        def f(v):
            self.floors[key] = None if float(v) <= off + 1e-12 else float(v)
        return f

    def _set_re_source(self, v) -> None:
        self.re_source = "library" if str(v) == "library" else "mission"

    def _screen_rows(self) -> list:
        P = ui.Param
        dead = self.dead()
        rows = [
            P("dp", "DESIGN POINT", None, kind="label"),
            P("resrc", "screened at", lambda: self.effective_re_source(), self._set_re_source,
              kind="choice", choices=["mission", "library"], enabled=bridge.xfoil_ok,
              help="mission: this surface's own Reynolds number -- a live XFOIL sweep of the "
                   f"{self.shortlist} leaders of the cached ranking (~20 s cold, instant "
                   "after). library: the cached library point, Re 1e6 -- instant. Without "
                   "XFOIL only the library point exists"),
            P("re", "Reynolds number", lambda: float(self.conditions().get("re", 0.0)), None,
              lo=None, hi=None, fmt="{:.3g}", enabled=False),
            P("cl", "design cl", lambda: float(self.conditions().get("cl_design", 0.0)), None,
              lo=None, hi=None, fmt="{:.2f}", enabled=False,
              help="the wing: WingLab's reference CZ of a car wing (1.0); the plate: 0 -- its "
                   "panels are vertical and carry no design load"),
            P("w", "CRITERION WEIGHTS", None, kind="label"),
        ]
        for k, label, _h, _f in self.criteria():
            if self.target == "plate" and k in PLATE_FIXED_WEIGHTS:
                continue                    # the plate's t/c is not an option (the owner)
            rows.append(P(f"w.{k}", label + ("   (dead at cl 0)" if k in dead else ""),
                          (lambda kk=k: float(self.weights.get(kk, 0.0))), self._set_w(k),
                          step=0.05, fine=0.01, lo=0.0, hi=1.0, fmt="{:.2f}",
                          enabled=(k not in dead), help=dead.get(k, "")))
        rows += [
            P("rec", "Use the recommended weights", None, lambda _: self.recommend(),
              kind="action", enabled=lambda: self.weights_source != "recommended",
              help="WingLab's own set for this surface (V3's session weights)"),
            P("g", "HARD GATES  (a section is dropped, not ranked low)", None, kind="label"),
        ]
        #  THE OWNER'S CALL (2026-09-25): "endplate t/c shouldn't be given as
        #  an option." The plate's screen keeps AeroBO's own t/c gate
        #  (SCREEN_GATES, switched on) underneath -- sent, never editable; the
        #  wing's section (stage 2) keeps its t/c rows. Do not restore them here.
        if self.target != "plate":
            rows += [
                P("gtc", "minimum t/c", lambda: self.screen_gates_on["tc_min"],
                  self._gate_switch(self.screen_gates_on, "tc_min"), kind="bool"),
                P("tcmin", "t/c at least",
                  self._gate_get(self.screen_gates, self.screen_gates_on, "tc_min"),
                  self._gate_set(self.screen_gates, "tc_min"), step=0.005, fine=0.001, lo=0.0,
                  hi=0.30, fmt="{:.3f}", enabled=lambda: self.screen_gates_on["tc_min"]),
            ]
        rows += [
            P("gcm", "maximum |cm|", lambda: self.screen_gates_on["cm_max"],
              self._gate_switch(self.screen_gates_on, "cm_max"), kind="bool"),
            P("cmmax", "|cm| at most", self._gate_get(self.screen_gates, self.screen_gates_on, "cm_max"),
              self._gate_set(self.screen_gates, "cm_max"), step=0.01, fine=0.002, lo=0.0,
              hi=0.5, fmt="{:.3f}", enabled=lambda: self.screen_gates_on["cm_max"]),
            P("f", "FLOORS  (off at the bottom of their range)", None, kind="label"),
            P("fclmax", "cl max at least", lambda: float(self.floors["clmax"] or 0.0),
              self._floor_set("clmax", 0.0), step=0.05, fine=0.01, lo=0.0, hi=3.0, fmt="{:.2f}"),
            P("fldcr", "L/D at the design cl at least", lambda: float(self.floors["ldcr"] or 0.0),
              self._floor_set("ldcr", 0.0), step=1.0, fine=0.25, lo=0.0, hi=250.0, fmt="{:.1f}",
              enabled=("ldcr" not in dead)),
            P("fastall", "stall angle at least", lambda: float(self.floors["astall"] or 0.0),
              self._floor_set("astall", 0.0), step=0.5, fine=0.1, lo=0.0, hi=25.0,
              unit="deg", fmt="{:.1f}"),
            P("a", "SCREEN", None, kind="label"),
            P("go", "Screen the library  (L)", None, lambda _: self.start_screen(),
              kind="action", help="ranks the section library on the weights above, at the "
                                  "design point above -- live, on a worker thread"),
            P("stop", STOP_TEXT, None, lambda _: self.session.stop(self), kind="action",
              enabled=lambda: self.session.runs.job_for(self) is not None),
        ]
        for p in rows:
            if p.key in ("fclmax", "fldcr", "fastall"):
                p.off_text = "no floor"
        if self.target == "plate":
            rows.append(P("decline", "Keep the family's own plate  (NACA 00tt)", None,
                          lambda _: self.decline(), kind="action",
                          help="the endplate family's own section: a NACA 00tt at the t/c the "
                               "wing search chooses. An explicit answer, like taking one"))
        return rows

    def _section_rows(self) -> list:
        P = ui.Param
        return [
            P("s", "SECTION", None, kind="label"),
            P("use", "Use this section  (F)", None, lambda _: self.use_highlighted(),
              kind="action", enabled=lambda: bool(self.ranked) or self.has_optimised(),
              help="the wing flies it from now on -- its coordinates, at this surface's "
                   "design point"),
            P("decline", ("Keep the family's own plate  (NACA 00tt)"
                          if self.target == "plate" else "Fly the family's published section"),
              None, lambda _: self.decline(), kind="action"),
            P("refine", "Refine this shape", None,
              lambda _: self.session.goto(f"{self.stage}.opt"), kind="action"),
        ]

    def _set_obj(self, v) -> None:
        if str(v) in bridge.SECTION_OBJECTIVES:
            self.opt["objective"] = str(v)

    def _budget_get(self) -> int:
        try:
            return int(self.session.policy.section_search(self.opt["objective"], self.target)["budget"])
        except Exception:                               # noqa: BLE001 -- a read-out
            return int(self.session.policy.own["plate" if self.target == "plate" else "airfoil"])

    def _budget_set(self, v) -> None:
        self.session.policy.own["plate" if self.target == "plate" else "airfoil"] = int(
            min(max(int(v), 4), 1000))

    def _more_get(self) -> int:
        return int(self.opt["more"] or self.continue_extra())

    def _opt_rows(self) -> list:
        P = ui.Param
        own = lambda: not self.session.policy.recommended()           # noqa: E731
        rows = [
            P("o", "SHAPE OPTIMISATION", None, kind="label"),
            P("obj", "objective", lambda: self.opt["objective"], self._set_obj, kind="choice",
              choices=list(bridge.SECTION_OBJECTIVES),
              help="; ".join(f"{k}: {SECTION_OBJECTIVE_WORDS.get(k, k)}"
                             for k in bridge.SECTION_OBJECTIVES)),
        ]
        #  the owner's call (2026-09-25, see _screen_rows): the plate's shape
        #  search keeps AeroBO's own t/c gate (OPT_GATES) underneath, not a row
        if self.target != "plate":
            rows += [
                P("gotc", "minimum t/c", lambda: self.opt["gates_on"]["tc_min"],
                  self._gate_switch(self.opt["gates_on"], "tc_min"), kind="bool"),
                P("otc", "t/c at least",
                  self._gate_get(self.opt["gates"], self.opt["gates_on"], "tc_min"),
                  self._gate_set(self.opt["gates"], "tc_min"), step=0.005, fine=0.001, lo=0.0,
                  hi=0.30, fmt="{:.3f}", enabled=lambda: self.opt["gates_on"]["tc_min"]),
            ]
        return rows + [
            P("gocm", "maximum |cm|", lambda: self.opt["gates_on"]["cm_max"],
              self._gate_switch(self.opt["gates_on"], "cm_max"), kind="bool"),
            P("ocm", "|cm| at most", self._gate_get(self.opt["gates"], self.opt["gates_on"], "cm_max"),
              self._gate_set(self.opt["gates"], "cm_max"), step=0.01, fine=0.002, lo=0.0,
              hi=0.5, fmt="{:.3f}", enabled=lambda: self.opt["gates_on"]["cm_max"]),
            P("budget", "evaluations", self._budget_get, self._budget_set, kind="int",
              lo=4, hi=1000, enabled=own,
              help="recommended: WingLab's measured budget for this 8-D search at the effort "
                   "on 1 Mission > Search & budget (164 at balanced). Own values: yours"),
            P("seed", "random seed", lambda: int(self.opt["seed"]),
              lambda v: self.opt.__setitem__("seed", int(min(max(int(v), 0), 9999))),
              kind="int", lo=0, hi=9999),
            P("run", "Run  (O)", None, lambda _: self.start_optimise(), kind="action"),
            P("stop", STOP_TEXT, None, lambda _: self.session.stop(self), kind="action",
              enabled=lambda: self.session.runs.job_for(self) is not None),
            P("more", "more evaluations", self._more_get,
              lambda v: self.opt.__setitem__("more", int(min(max(int(v), 1), 1000))),
              kind="int", lo=1, hi=1000,
              help="what Keep going buys: the evaluations already paid for are the new "
                   "search's training set, nothing is re-flown (WingLab's resume)"),
            _LiveLabel("go", self._keep_label, None, lambda _: self.start_optimise(extend=True),
                       kind="action", enabled=lambda: self.can_continue()),
            P("use", "Use this section  (F)", None, lambda _: self.use_optimised(),
              kind="action", enabled=lambda: self.has_optimised()),
        ]

    def _keep_label(self) -> str:
        if not self.can_continue():
            return "Keep going  (K)"
        spent = self.spent()
        more = self._more_get()
        return f"Keep going — {more} more ({spent + more} in total)  (K)"

    # -- screening -----------------------------------------------------------------
    def start_screen(self) -> bool:
        """Screen the library LIVE (L, the Screen button): AeroBO's
        `screen_at_point` (the surface's own Re: a library pass then a live
        XFOIL sweep of the shortlist) or `screen_airfoils` (the cached
        library point), restricted to the symmetric sections on the plate."""
        s = self.session
        if not s.ready_to_run(self):
            return False
        cond = self.conditions()
        src = self.effective_re_source()
        weights = {k: float(v) for k, v in self.weights.items()}
        gates, floors = self.gates("screen"), dict(self.floors)

        def engine():
            return bridge.screen_runner(self.target, weights, cond, gates, floors, src)
        runner = s.runner("screen", engine, surface=self.target, re_source=src)
        #  the N the chip opens on: the shortlist of a live sweep; the library
        #  pass reports its own count ("total" / "sweep" events set it)
        n = self.shortlist if src == "mission" else _library_count(self.target)
        #  the design point and what is screened, for the job's Output lines
        #  (design_jobs.line_screen_start words them)
        info = s.job_info("screen", self, objective="screen",
                          phrase=("symmetric sections only (the owner's rule)"
                                  if self.target == "plate" else ""),
                          point={"re": float(cond["re"]), "cl": float(cond["cl_design"]),
                                 "source": "own" if src == "mission" else "library"})
        job = dj.EngineJob(info, self, runner=runner, n=n, unit="section",
                           on_finish=lambda j: self._screen_done(j, cond, src),
                           notices=s.notices, replay=s.replay_of(runner))
        return s.launch(job)

    def _screen_done(self, job, cond: dict, src: str) -> None:
        out = job.outcome()
        rep_ = (job.result or {}).get("report") if isinstance(job.result, dict) else None
        if out.get("state") == "done" and not out.get("k") and (rep_ or {}).get("n_screened"):
            #  the cached library pass is ONE step with no per-section event:
            #  its count is the sections AeroBO's report says it screened
            n_scr = int(rep_["n_screened"])
            out = dict(out, k=n_scr, n=n_scr)
        self.screen["outcome"] = out
        if job.state == "error":
            self.msg = f"the screen failed: {job.error}"
            return
        res = job.result or {}
        rep = res.get("report") or {}
        if not rep.get("ranked"):
            self.msg = ("the screen was stopped before it ranked anything" if out["state"] == "stopped"
                        else "no section passed the gates and floors")
            return
        self.screen.update(report=rep, candidates=res.get("candidates") or [], cond=dict(cond),
                           re_source=src)
        self.ranked = self._merge(rep, res.get("candidates") or [])
        self.highlight = 0
        self.msg = (f"{len(self.ranked)} ranked of {rep.get('n_eligible', '?')} eligible "
                    f"({rep.get('n_screened', '?')} screened); best {self.ranked[0]['name']}")

    def _merge(self, rep: dict, cands: list) -> list:
        """The ranked rows the table and the section view read: AeroBO's
        `report["ranked"]` (name, composite, per-criterion scores and the raw
        metrics) with each row's coordinates and CST refit from
        `screen_seed_candidates`, best first."""
        by = {c.get("name"): c for c in cands}
        rows = []
        for i, r in enumerate(rep.get("ranked") or [], start=1):
            c = by.get(r.get("name")) or {}
            metrics = {k: r.get(k) for k in ("tc", "clmax", "astall", "ldmax", "ldcr",
                                             "cd_at", "cm_at", "clmax_censored")}
            rows.append(dict(rank=i, name=r.get("name"), composite=_finite(r.get("composite")),
                             scores=dict(r.get("scores") or {}), metrics=metrics,
                             tc=_finite(r.get("tc")), coords=c.get("coords"),
                             w_upper=c.get("w_upper"), w_lower=c.get("w_lower"),
                             te_gap=c.get("te_gap"), rank_library=r.get("rank_library")))
        return rows[: max(1, int(self.top_n))]

    # -- the decision --------------------------------------------------------------
    def use_ranked(self, i: int) -> None:
        """Take ranked row `i` as this surface's section (F / ENTER on the
        ranking): the library section, at the point it was screened."""
        if not self.ranked:
            self.msg = "screen the library first (L)"
            return
        i = int(min(max(int(i), 0), len(self.ranked) - 1))
        r = self.ranked[i]
        cond = self.screen.get("cond") or {}
        at_lib = self.screen.get("re_source") == "library"
        self.chosen = dict(source="library", name=r["name"], tc=r.get("tc"), coords=r.get("coords"),
                           w_upper=r.get("w_upper"), w_lower=r.get("w_lower"),
                           conditions={"re": float(cond.get("re", 0.0)),
                                       "mach": float(cond.get("mach", 0.0))},
                           library_point=at_lib,
                           origin=(f"WingLab library: {r['name']}, rank {i + 1} at "
                                   f"Re {float(cond.get('re', 0.0)):.3g}"),
                           n_evals=None)
        self.decision = "library"
        self.highlight = i
        self.session.on_section(self)
        self.msg = f"{self.label} section: {r['name']} (library, rank {i + 1})"
        self.session.log(f"{self.label} section chosen: {r['name']} — rank {i + 1} of the "
                         f"screen at Re {float(cond.get('re', 0.0)):.3g}", "ok")

    def use_highlighted(self) -> None:
        """What the section view's button and F do: the highlighted row."""
        self.use_ranked(self.highlight)

    def use_library(self, name: str) -> bool:
        """Take AeroBO library section `name` by NAME, at the cached library
        point (the garage's airfoil page: ENTER on a section AeroBO's library
        also has). False, with the reason in `msg`, when AeroBO does not know
        it -- or when the plate is offered a cambered one (the plate is
        symmetric, the owner's rule)."""
        api = bridge.api
        try:
            known = bool(api.library_section_available(name))
        except Exception:                                   # noqa: BLE001
            known = False
        if not known:
            self.msg = (f"{name} is not in WingLab's section library: the design page flies "
                        f"WingLab's sections (2 Airfoil ▸ Ranking)")
            return False
        coords = np.asarray(api.chosen_section_coords(name), dtype=float)
        if self.target == "plate" and float(api.section_max_camber(coords)) > api.SYMMETRIC_CAMBER_TOL:
            self.msg = f"{name} is cambered: the endplate flies symmetric sections only"
            return False
        pt = api.screen_library_point() or {}
        self.chosen = dict(source="library", name=str(name), tc=_loop_tc(coords),
                           coords=coords.tolist(), w_upper=None, w_lower=None,
                           conditions={"re": float(pt.get("re", 1e6)),
                                       "mach": float(pt.get("mach", 0.0))},
                           library_point=True, origin=f"WingLab library: {name} (by name)",
                           n_evals=None)
        self.decision = "library"
        self.session.on_section(self)
        self.msg = f"{self.label} section: {name} (WingLab's library, by name)"
        self.session.log(self.msg, "ok")
        return True

    def has_optimised(self) -> bool:
        rep = self.opt.get("report") or {}
        return bool(((rep.get("section") or {}).get("design") or {}).get("coords"))

    def use_optimised(self) -> None:
        """Take the shape search's winner as this surface's section."""
        if not self.has_optimised():
            self.msg = "no optimised section yet: run the shape optimisation (O)"
            return
        rep = self.opt["report"]
        d = rep["section"]["design"]
        cond = rep.get("conditions") or {}
        n = len(self.opt.get("records") or []) or int(((rep.get("result") or {}).get("n_evals")) or 0)
        seed = self.opt.get("anchor_name") or "the family's anchor"
        self.chosen = dict(source="optimised", name=f"{seed}-opt", tc=_finite(d.get("tc")),
                           coords=d.get("coords"), w_upper=list(d.get("w_upper") or []),
                           w_lower=list(d.get("w_lower") or []),
                           conditions={"re": float(cond.get("re", 0.0)),
                                       "mach": float(cond.get("mach", 0.0))},
                           library_point=False,
                           origin=f"WingLab: CST optimised from {seed}, {n} evaluations",
                           n_evals=n)
        self.decision = "optimised"
        self.session.on_section(self)
        self.msg = f"{self.label} section: the optimised shape ({n} evaluations)"
        self.session.log(f"{self.label} section chosen: CST optimised from {seed} "
                         f"({n} evaluations)", "ok")

    def decline(self) -> None:
        """Keep the family's own section -- the published one on the wing,
        NACA 00tt at the searched t/c on the plate. An ANSWER, so the stage
        reads done; no section flag is sent (`flag_value` is None)."""
        self.chosen = dict(source="default",
                           name=("NACA 00tt (the family's own)" if self.target == "plate"
                                 else "the family's published section"),
                           tc=None, coords=None, w_upper=None, w_lower=None, conditions=None,
                           library_point=False, origin="WingLab family default", n_evals=None)
        self.decision = "default"
        self.session.on_section(self)
        self.msg = ("the plates keep the family's own NACA 00tt" if self.target == "plate"
                    else "the wing flies the family's published section")
        self.session.log(self.msg)

    def finished(self) -> bool:
        """Has this stage been ANSWERED (a section taken, or the default
        kept)? The tutorial's predicate."""
        return self.decision is not None

    def flag_value(self):
        """What the wing's `section_name` / `section_name_plate` flag carries
        for this surface (`bridge.section_flag_value`), None when undecided or
        on the family's default (no flag is sent then)."""
        if self.decision not in ("library", "optimised") or not self.chosen:
            return None
        return bridge.section_flag_value(self.chosen)

    # -- shape optimisation -----------------------------------------------------
    def anchor(self) -> tuple:
        """`(name, [w_upper, w_lower] | None)` the search starts from: the
        chosen section's CST refit, or the family's NACA anchor."""
        c = self.chosen or {}
        if c.get("w_upper") and c.get("w_lower"):
            return c.get("name"), [list(c["w_upper"]), list(c["w_lower"])]
        if self.ranked and self.ranked[self.highlight].get("w_upper"):
            r = self.ranked[self.highlight]
            return r["name"], [list(r["w_upper"]), list(r["w_lower"])]
        return None, None

    def reference(self):
        """The band the composite is scored on: this screen's own measured
        reference when it carried one, else AeroBO's shipped band (None)."""
        rep = self.screen.get("report") or {}
        return rep.get("reference")

    def spent(self) -> int:
        rep = self.opt.get("report") or {}
        return int(((rep.get("result") or {}).get("n_evals")) or len(self.opt.get("records") or []))

    def can_continue(self) -> bool:
        rep = self.opt.get("report")
        return bool(rep and self.opt.get("launch") and (rep.get("result") or {}).get("eval_x"))

    def continue_extra(self) -> int:
        rep = self.opt.get("report") or {}
        res = rep.get("result") or {}
        return int(bridge.continue_extra_default({"config": res.get("config") or rep.get("config") or {},
                                                  "n_evals": res.get("n_evals"),
                                                  "partial": res.get("partial")}))

    def refusal(self) -> str | None:
        """Why a shape search cannot run here, or None: XFOIL is missing
        (every evaluation is a live XFOIL polar; PLAN2 D11). A machine without
        the BO stack is NOT a refusal -- AeroBO's non-torch optimisers fly
        instead, and `bo_note()` says so (PLAN2 §11 Q10)."""
        if not bridge.xfoil_ok():
            return bridge.XFOIL_REFUSAL
        return None

    def bo_note(self) -> str:
        """AeroBO's own sentence for a machine without torch ("" with it)."""
        return bridge.torch_note()

    def start_optimise(self, extend: bool = False) -> bool:
        """Optimise the shape LIVE (O / Run): AeroBO's `optimize_airfoil` at
        this surface's point, V3's shape arguments, AeroBO's budget (164 at
        balanced). `extend` is Keep going (K): AeroBO's resume -- the
        evaluations already paid for are the training set, nothing re-flies."""
        s = self.session
        why = self.refusal()
        if why and s.replay_active() is None:
            self.msg = why
            s.say(why, "warning")
            return False
        if not s.ready_to_run(self):
            return False
        if extend and not self.can_continue():
            self.msg = "no finished shape search to continue: Run it first (O)"
            s.say(self.msg, "info")
            return False
        n_prior = 0
        if extend:
            extra = self._more_get()
            cont = bridge.continue_section(self.opt["report"], self.opt["launch"], extra)
            shape, seed, search = cont["shape"], int(cont["seed"]), dict(cont["search"])
            resume = search.pop("resume", None)
            n_prior = int((cont.get("note") or {}).get("resumed") or self.spent())
            variant = "continued"
        else:
            cond = self.conditions()
            if not cond:
                self.msg = "this family has no such surface to design"
                s.say(self.msg, "info")
                return False
            name, anchor = self.anchor()
            self.opt["anchor_name"] = name
            g = self.gates("opt")
            shape = bridge.shape_kwargs(cond, self.target,
                                        {"objective": self.opt["objective"],
                                         "tc_min": g["tc_min"], "cm_max": g["cm_max"]},
                                        dict(self.weights), self.reference(),
                                        anchor=(tuple(anchor) if anchor else None))
            seed, resume = int(self.opt["seed"]), None
            search = self.session.policy.section_search(self.opt["objective"], self.target, seed)
            variant = "full"
        n = int(search["budget"])
        #  the Sobol block the live graph marks (a resumed run draws none)
        self._live_n_init = None if extend else search.get("n_init")
        results_dir = s.results_dir("section")

        def engine():
            return bridge.section_runner(shape, search, seed, resume, results_dir)
        runner = s.runner("section", engine, surface=self.target, variant=variant)
        pt = (self.opt["report"] or {}).get("conditions") if extend else cond
        pt = pt or {}
        info = s.job_info("section", self, objective=self.opt["objective"],
                          phrase=SECTION_OBJECTIVE_WORDS.get(self.opt["objective"], ""),
                          continued=extend, optimiser=str(search.get("optimiser") or ""),
                          n_init=self._live_n_init,
                          point=({"re": float(pt["re"]),
                                  "cl": float(pt.get("cl_design", pt.get("cl", 0.0))),
                                  "source": pt.get("re_source") or "own"} if pt.get("re") else None),
                          seed_label=str(self.opt.get("anchor_name") or ""))
        prior = list(self.opt.get("records") or []) if extend else []
        job = dj.EngineJob(info, self, runner=runner, n=n, n_prior=n_prior,
                           on_finish=lambda j: self._opt_done(j, prior, n_prior),
                           notices=s.notices, replay=s.replay_of(runner))
        return s.launch(job)

    def _opt_done(self, job, prior: list, n_prior: int) -> None:
        out = job.outcome()
        self.opt["outcome"] = out
        if job.state == "error":
            self.msg = f"the shape search failed: {job.error}"
            return
        res = job.result or {}
        rep = res.get("report")
        if not rep:
            self.msg = "the shape search returned no report"
            return
        self.opt.update(report=rep, launch=res.get("launch"),
                        records=prior + list(job.records), prior=int(n_prior), more=None,
                        score=None, score_outcome=None)
        design = (rep.get("section") or {}).get("design") or {}
        self.msg = (f"shape search {out['state']}: {len(self.opt['records'])} evaluations, "
                    f"t/c {float(design.get('tc') or 0.0):.3f}")
        self.start_score()

    def start_score(self) -> bool:
        """The post-run job: seed vs optimised on every criterion under this
        surface's own weights (`api.score_optimised_section`) -- a wide stall
        sweep per shape, so it is a job, never a frame."""
        s = self.session
        rep = self.opt.get("report")
        if not rep or s.runs.busy:
            return False
        rp = s.replay_active()
        if rp is not None and rp.find("score", surface=self.target) is None:
            #  a replayed run has no captured score, and the real one sweeps
            #  XFOIL -- which a deterministic check never starts
            return False
        weights, reference = dict(self.weights), self.reference()

        def engine():
            return bridge.score_runner(rep, weights, reference)
        runner = s.runner("score", engine, surface=self.target)
        info = s.job_info("score", self, objective="score", phrase="seed vs optimised")
        job = dj.EngineJob(info, self, runner=runner, n=1, unit="evaluation",
                           on_finish=self._score_done, notices=s.notices,
                           replay=s.replay_of(runner))
        return s.launch(job, quiet=True)

    def _score_done(self, job) -> None:
        self.opt["score_outcome"] = job.outcome()
        if job.state == "error":
            return
        self.opt["score"] = job.result

    # -- the section's polar (the Section view) ----------------------------------
    def shown(self) -> dict | None:
        """The section the Section view shows, in `chosen`'s shape: the one
        taken, else the highlighted ranked row at the point it was screened
        at. None before a screen with nothing taken, and for the family's own
        section (AeroBO flies it from its family polar, not a library one)."""
        if self.chosen and self.decision in ("library", "optimised"):
            return self.chosen
        if not self.ranked:
            return None
        r = self.ranked[min(max(int(self.highlight), 0), len(self.ranked) - 1)]
        cond = self.screen.get("cond") or {}
        return dict(source="library", name=r["name"], tc=r.get("tc"), coords=r.get("coords"),
                    w_upper=r.get("w_upper"), w_lower=r.get("w_lower"),
                    conditions={"re": float(cond.get("re", 0.0)),
                                "mach": float(cond.get("mach", 0.0))},
                    library_point=self.screen.get("re_source") == "library")

    @staticmethod
    def _polar_key(value) -> str:
        return json.dumps(value, sort_keys=True, default=str)

    def polar(self, which: dict | None = None) -> dict | None:
        """The polar of `which` (default `shown()`) from what is AT HAND --
        never an XFOIL run, so a view may call it every frame: an optimised
        section's own sweep (its report's `design.polar`), AeroBO's screening
        branch at the library point, or a cached XFOIL sweep of a library
        section at its own Re (`cache_only`). `{alpha_deg, cl, cd, cm, re,
        mach, source}`, or None: not at hand -- `start_polar` fetches it."""
        c = self.shown() if which is None else which
        if not c:
            return None
        if c.get("source") == "optimised":
            d = ((self.opt.get("report") or {}).get("section") or {}).get("design") or {}
            p = d.get("polar")
            if not p:
                return None
            return dict(alpha_deg=list(p.get("alpha_deg") or []), cl=list(p.get("cl") or []),
                        cd=list(p.get("cd") or []), cm=list(p.get("cm") or []),
                        re=_finite(p.get("re")), mach=_finite(p.get("mach")),
                        source="the optimised section's own XFOIL sweep")
        value = bridge.section_flag_value(c)
        if value is None:
            return None
        key = self._polar_key(value)
        if key in self.polars:
            return self.polars[key]
        try:
            if isinstance(value, str):
                pol = bridge.api.library_section_polar(value, cache_only=True)
            else:
                pol = bridge.api.section_polar_for(value, cache_only=True)
        except Exception:                                   # noqa: BLE001 -- a read-out
            pol = None
        out = self._polar_payload(pol, value)
        self.polars[key] = out                              # a miss is memoised too
        return out

    @staticmethod
    def _polar_payload(pol, value) -> dict | None:
        """AeroBO's TablePolar as the plain dict the views draw."""
        if pol is None:
            return None
        pt = bridge.api.section_polar_point(value)
        return dict(alpha_deg=[float(v) for v in pol.alpha_deg], cl=[float(v) for v in pol.CL],
                    cd=[float(v) for v in pol.CD], cm=[float(v) for v in pol.CM],
                    re=float(getattr(pol, "Re", None) or pt.get("re") or 0.0),
                    mach=float(pt.get("mach") or 0.0), source=str(getattr(pol, "name", "")))

    def start_polar(self, which: dict | None = None) -> bool:
        """Fetch a polar `polar()` did not have at hand, as a JOB (a live
        XFOIL sweep of the section at its own Re: seconds). Not under a
        replay (no polar is captured, and a check never starts XFOIL)."""
        s = self.session
        c = self.shown() if which is None else which
        value = bridge.section_flag_value(c) if c and c.get("source") != "optimised" else None
        if value is None or s.runs.busy:
            return False
        if s.replay_active() is not None:
            return False
        key = self._polar_key(value)
        if self.polars.get(key) is not None:
            return False
        if not bridge.xfoil_ok():
            self.msg = "this section's polar at its own Re needs XFOIL"
            return False
        runner = bridge.polar_runner(value)
        name = c.get("name") or "the section"
        info = s.job_info("polar", self, objective="polar", phrase=f"{name}'s polar")
        job = dj.EngineJob(info, self, runner=runner, n=1, unit="evaluation",
                           on_finish=lambda j: self._polar_done(j, key, value),
                           notices=s.notices)
        return s.launch(job, quiet=True)

    def _polar_done(self, job, key: str, value) -> None:
        if job.state == "error":
            self.msg = f"no polar: {job.error}"
            return
        self.polars[key] = self._polar_payload(job.result, value)

    # -- the live figure --------------------------------------------------------
    def graph(self) -> dict:
        """The inputs of the live evaluation graph (views_common, PLAN2
        §8.5): every evaluation (live while the job runs, stored after), N,
        the inherited prefix, the Sobol/BO split, the tag."""
        s = self.session
        job = s.runs.job_for(self)
        live = job is not None and getattr(job.info, "kind", "") == "section"
        recs = list(self.opt.get("records") or [])
        if live:
            recs = (recs if job.n_prior else []) + list(job.records)
        rep = self.opt.get("report") or {}
        res = rep.get("result") or {}
        if live:
            n_init = getattr(self, "_live_n_init", None) if not job.n_prior else None
        else:
            split = res.get("bo_split") or [None, None]
            n_init = split[0] if split and split[0] is not None else (
                (self.opt.get("launch") or {}).get("search", {}).get("n_init"))
        N = int(job.n) if live else int((self.opt.get("outcome") or {}).get("n") or
                                        self._budget_get())
        return dict(records=recs, N=N, n_prior=int(job.n_prior if live else self.opt.get("prior") or 0),
                    n_init=int(n_init) if n_init else None, handoff=None,
                    units=str(res.get("score_units") or self.opt["objective"]),
                    objective=self.opt["objective"], live=live,
                    tag=(job.chip() if live else outcome_tag(self.opt.get("outcome"))))

    # -- what the tree says ---------------------------------------------------------
    def state(self, view: str = "") -> str:
        """A view's state inside this (unlocked) stage: running while one of
        its jobs is live, done once the view has something, else ready."""
        job = self.session.runs.job_for(self)
        if job is not None:
            kind = getattr(job.info, "kind", "")
            if (view in ("", "screen") and kind == "screen") or (view in ("", "opt") and kind in ("section", "score")):
                return "running"
        if view == "screen":
            return "done" if self.ranked else "ready"
        if view == "rank":
            return "done" if self.ranked else "ready"
        if view == "section":
            return "done" if self.finished() else "ready"
        if view == "opt":
            return "done" if self.has_optimised() else "ready"
        return "done" if self.finished() else "ready"

    def reason(self, view: str = "") -> str:
        if view == "rank" and not self.ranked:
            return "nothing screened yet — run the screening on the previous tab"
        if view == "opt":
            why = self.refusal()
            if why:
                return why
        if not self.finished():
            return ("the plate flies the family's own NACA 00tt until a section is chosen here"
                    if self.target == "plate" else
                    "the wing flies the family's own published section until one is chosen here")
        return ""

    def reset(self) -> None:
        """Forget every screen, choice and search (a restated mission)."""
        self.screen = {"report": None, "candidates": None, "outcome": None,
                       "surface": self.target, "cond": None, "re_source": None}
        self.ranked, self.chosen, self.decision, self.highlight = [], None, None, 0
        self.opt.update(report=None, launch=None, records=[], outcome=None, score=None,
                        score_outcome=None, prior=0, more=None, anchor_name=None)
        self.polars.clear()
        self.msg = ""


# --------------------------------------------------------------------------- #
#  the wing (stage 3)                                                          #
# --------------------------------------------------------------------------- #
class WingModel:
    """Stage 3: AeroBO's car rear wing, flown for this slot.

    The FAMILY is AeroBO's own car session's (`make_session("track")`):
    "car rear wing + endplates + free chord law", 14-D, bo_slsqp, 53
    evaluations at balanced -- registered as a carsim SLOT family
    (`bridge.ensure_family`) that moves only the ride band: over the boot at
    the slot's height for the top wing (ground effect on), 100 m from its
    image for a flank (ground effect off; the plate still reaches the car's
    side), in carsim's air (rho 1.2). Wing type asks what AeroBO's own car
    card asks: first what CARRIES the wing -- the endplates (a designed part,
    stage 2.8, with its root blend and lean each stated or optimised) or a
    swan-neck pylon pair inboard (the plates become a tip device: none,
    vertical, canted, blended, following the wing's chord) -- then free chord
    law or straight, and AeroBO's car objective menu.

    The box is AeroBO's: every row of the built problem, with its band,
    and AeroBO's constrain / release / fix switches (`bounds_overrides`,
    `pinned`). The plate's t/c row is FIXED to the section chosen in 2.8
    (PLAN2 §11 Q4) and is NOT one of the player's rows: the owner, 2026-09-25,
    "endplate t/c shouldn't be given as an option" (`NO_BOX_CONTROLS`). With
    the family's own plate kept, AeroBO searches it itself, as in AeroBO."""

    def __init__(self, session: "DesignSession"):
        self.session = session
        self.key, self.role = session.key, session.role
        #: the card's answers. `plates` IS "Carried by" (True: the endplates,
        #: False: the pylons -- one key, as AeroBO's `car_endplates`); `blend` /
        #: `cant` "stated" | "free" and the stated numbers (a crease, upright:
        #: what every published run flew); `tip` the pylons' tip device and
        #: `tip_cant_deg` its lean when canted (its own key: leaning the
        #: device never leans the endplates behind the player's back). The
        #: typed numbers are never cleared by another answer: switching back
        #: gives them back (AeroBO's rule)
        self.choices = {"plates": True, "chord_law": True,
                        "blend": "stated", "blend_frac": 0.0,
                        "cant": "stated", "cant_deg": bridge.CANT_UPRIGHT,
                        "tip": bridge.TIP_DEFAULT, "tip_cant_deg": bridge.CANT_UPRIGHT,
                        "tip_chord": bridge.TIP_CHORD_DEFAULT,
                        "objective": "efficiency", "drag_budget_n": None, "downforce_min_n": None}
        self.box: dict = {}
        self.released: set = set()
        self.fixed: dict = {}
        self.seed = 0
        self.record: dict | None = None
        self.records: list = []
        self.outcome: dict | None = None
        self.prior = 0
        self.more: int | None = None
        self.law: dict | None = None
        self.law_outcome: dict | None = None
        self.spec = None
        self.slot_updates: dict = {}
        self.dirty = False
        self.msg = ""
        self._built = None
        self._built_key = None
        self.type_params = Form(self._type_rows(), title="")
        self.solver_params = Form(self._solver_rows(), title="")
        self.conv_params = Form(self._conv_rows(), title="")
        self._box_form = None
        self._box_form_key = None
        self._own, self._own_key = {}, None     # `own_band`'s cache
        self._live_split = None              # (cfg, dim, memo) of a live fresh bo_slsqp run
        self._section_specs: dict = {}       # the sections `fit` put on `spec`
        #: the sections the RECORD flew (`sections()` at its launch): what the
        #: fitted wing is built from, whatever 2 / 2.8 have chosen since
        self._record_sections: dict = {}

    # -- the family -----------------------------------------------------------------
    @property
    def family(self) -> "bridge.FamilyParams":
        """This slot's family (`bridge.family_params`), from the Wing type
        switches and the operating point: the ride band and the deck, the
        circuit and car when lap time is the objective."""
        return bridge.family_params(self.session.op, dict(self.choices), lap=self._lap())

    @property
    def family_name(self) -> str:
        return bridge.ensure_family(self.family)

    def _lap(self):
        """The circuit and the car the lap objective times (`bridge.lap_params`),
        for the TOP wing on the pylons only -- the fence family (AeroBO's
        menu: the endplate family has no track_spec; cartrack's lap cannot
        value a lateral device). None for every other objective."""
        if self.choices["objective"] != "laptime" or not self.laptime_offered():
            return None
        s = self.session
        return bridge.lap_params(s.mission, s.host.build, s.host.lib, self.key)

    def laptime_offered(self) -> bool:
        return self.role == "top" and not self.choices["plates"]

    def objectives(self) -> dict:
        """AeroBO's car objective menu for this slot, key -> label (M15):
        the flank's downforce is SIDE force; lap time only on the top wing
        carried by the pylons (the fence family), round the stated circuit."""
        return dict(bridge.car_objectives(self.role, bool(self.choices["plates"]),
                                          self.session.job()))

    def default_objective(self, quiet: bool = False) -> None:
        """Open on the job's own objective (`JOB_OBJECTIVE`; a circuit:
        efficiency; a side wing on a circuit: side force, `SIDE_OBJECTIVE`).
        Called when the job is stated or changes, never on a restatement that
        keeps the session: the player's pick stays. `quiet`: no Output line
        (a session not opened yet)."""
        job = self.session.job()
        obj = JOB_OBJECTIVE.get(job, SIDE_OBJECTIVE if self.role != "top" else "efficiency")
        objs = self.objectives()
        if obj in objs and obj != self.choices["objective"]:
            self.choices["objective"] = obj
            self._box_form = None
            if quiet:
                return
            whose = (f"the {ms.JOB_WORDS[job]} job's" if job in ms.JOB_WORDS else
                     "a side wing's" if self.role != "top" else f"the {job} job's")
            self.session.log(f"objective: {objs[obj]} — {whose} own; 3 Wing ▸ Wing type picks "
                             f"any other")

    def built(self):
        """AeroBO's built problem for the family as it stands (labels,
        bounds), cached on everything that changes it."""
        key = (self.family_name, json.dumps(self.physics_flags(), sort_keys=True, default=str),
               json.dumps(self.bounds_overrides(), sort_keys=True, default=str))
        if key != self._built_key:
            self._built = bridge.build_family(self.family_name, self.physics_flags(),
                                              self.bounds_overrides())
            self._built_key = key
        return self._built

    def family_box(self) -> dict:
        """`{label: (lo, hi)}` the run will search -- the box the section
        conditions are read at the middle of, and the Design box's rows."""
        b = self.built()
        return {str(lab): (float(lo), float(hi)) for lab, (lo, hi) in zip(b.param_labels, b.bounds)}

    # -- what the run is sent -------------------------------------------------------
    def sections(self) -> dict:
        """The sections stages 2 and 2.8 chose (their `chosen` dicts), None
        where the family's own flies."""
        s = self.session
        return {"main": s.af.chosen if s.af.decision in ("library", "optimised") else None,
                "plate": (s.ep.chosen if (self.choices["plates"]
                                          and s.ep.decision in ("library", "optimised")) else None)}

    def physics_flags(self) -> dict:
        """V3's `config.flags` for the car (`bridge.wing_physics_flags`): the
        family's switches, carsim's V (always sent), the objective and its
        limits, the sections chosen -- then AeroBO's own sanitiser."""
        return dict(bridge.wing_physics_flags(dict(self.choices), self.session.op, self.sections(),
                                              family=self.family_name))

    def auto_pins(self) -> dict:
        """The rows the card and the stages before this one FIX (PLAN2 §11
        Q4): the plate's t/c at the t/c of the section chosen in 2.8 -- it can
        no longer be released (the owner, 2026-09-25: "endplate t/c shouldn't
        be given as an option"), the plate flies the section 2.8 took, at its
        t/c -- and, under the pylons with the tip device "none", the plate's
        height at 0 (`bridge.tip_pins`: AeroBO's own pin, a smaller search)."""
        out = dict(bridge.tip_pins(self.choices))
        ep = self.session.ep
        if not self.choices["plates"] or ep.decision not in ("library", "optimised"):
            return out
        try:
            box = self._raw_box()
        except Exception:                                   # noqa: BLE001
            box = None
        out.update(bridge.plate_tc_pin(ep.chosen, box))
        return out

    def _raw_box(self) -> dict:
        """The family's box before any pin (the rows a pin is clipped into)."""
        return bridge.family_box(self.family_name, self.physics_flags(), self.bounds_overrides())

    def pinned(self) -> dict:
        """AeroBO's `pinned`: the rows fixed here (and from 2.8, and by the
        tip device). A row the player has no controls for (`NO_BOX_CONTROLS`)
        is never fixed here, and a card's pin wins over a hand-fixed value
        (the tip device "none" IS a height of 0, whatever was fixed before)."""
        out = self.hand_pins()
        out.update(self.auto_pins())
        return out

    def hand_pins(self) -> dict:
        """The rows the PLAYER fixed (`fixed`), at the values typed -- each
        held under the car's ceiling (task 41: the slot may have moved down
        since it was typed) -- without the rows a card pins (the tip
        device's height, the plate's t/c). No build: `bounds_overrides`
        reads it."""
        caps = self.session.op.size_caps or {}
        card = bridge.tip_pins(self.choices)
        out = {}
        for k, v in self.fixed.items():
            if k in NO_BOX_CONTROLS or k in card:
                continue
            out[k] = min(float(v), float(caps[k])) if k in caps else float(v)
        return out

    def own_band(self, lab: str) -> list | None:
        """The family's OWN band for a row -- no packaging, no typed band --
        what a released row is searched over (before the car's ceiling)."""
        key = (self.family_name, json.dumps(self.physics_flags(), sort_keys=True, default=str))
        if self._own_key != key:
            self._own = bridge.family_box(self.family_name, self.physics_flags(), None)
            self._own_key = key
        fb = self._own.get(lab)
        return [float(fb[0]), float(fb[1])] if fb else None

    def bounds_overrides(self) -> dict | None:
        """AeroBO's `bounds_overrides`: carsim's packaging rows for the slot
        (the flank's size cut to AeroBO's AR >= 3 validity) under the
        player's own constrained bands. A RELEASED row drops both -- AeroBO's
        "the solver's own box is searched" -- and keeps the numbers typed
        for when it is constrained again; the car's ceiling still holds it.
        A FIXED row's band is widened to hold its value (AeroBO's
        `config.bounds_overrides`): the api refuses a pin outside the box
        it is handed, and a fixed row is not searched, so the band costs no
        freedom."""
        out = {k: [float(v[0]), float(v[1])] for k, v in (self.session.op.size_rows or {}).items()
               if k not in self.released}
        for lab, (lo, hi) in self.box.items():
            if lab in self.released or lab in NO_BOX_CONTROLS:
                continue
            out[lab] = [float(lo), float(hi)]
        #  task 41: a span / area row is never searched past the car's ceiling
        #  (its physical limit in Real mode, 3x in Unlimited) -- the slot may
        #  have moved down since the row was typed, and a released row's own
        #  band may reach past it
        for lab, cap in (self.session.op.size_caps or {}).items():
            row = out.get(lab)
            if row is None and lab in self.released:
                row = self.own_band(lab)
            if row is not None and row[1] > float(cap):
                out[lab] = [min(row[0], float(cap) - 1e-4), float(cap)]
        #  a stated root blend or lean takes the plates' outboard reach out of
        #  the span row, so the packaging area row comes down to what the wing
        #  LEFT carries (`bridge.plate_area_floor`) -- at the widest span the
        #  box allows and the lowest ride it searches, a fixed row at its
        #  value; a band the player typed is theirs (`plate_fit_problem`
        #  refuses one no wing left can carry)
        s_row = out.get("S_m2")
        if s_row is not None and "S_m2" not in self.box:
            b_hi, ride_lo = self._plate_span_ride(out)
            floor = (None if b_hi is None else
                     bridge.plate_area_floor(self.session.op, self.choices, b_hi, ride_lo))
            if floor is not None and floor < float(s_row[0]):
                out["S_m2"] = [floor, float(s_row[1])]
        for lab, v in self.hand_pins().items():
            row = out.get(lab) or self.own_band(lab)
            if row is not None and not float(row[0]) <= v <= float(row[1]):
                out[lab] = [min(float(row[0]), v), max(float(row[1]), v)]
        return out or None

    def band_source(self, label: str) -> str:
        """Who set a box row's band: "tip device: none" | "fixed from 2.8" |
        "fixed" | "user" | "released" | "carsim packaging" | "slot" |
        "AeroBO default"."""
        if label in bridge.tip_pins(self.choices):
            return TIP_PIN_SOURCE
        if label in self.auto_pins():
            return "fixed from 2.8"
        if label in self.fixed:
            return "fixed"
        if label in self.released:
            return "released"
        if label in self.box:
            return "user"
        if label in (self.session.op.size_rows or {}):
            return "carsim packaging"
        if label == "ride_height_m":
            return "slot"
        return "WingLab default"

    def plan(self):
        """AeroBO's measured plan for this problem (`recommended_search`,
        cached by the bridge on everything it reads)."""
        return bridge.wing_plan(self.family_name, self.physics_flags(), self.bounds_overrides(),
                                self.pinned() or None, self.session.policy.effort)

    def search(self) -> dict:
        return self.session.policy.wing_search(self.plan(), self.family_name, int(self.seed))

    def cfg(self):
        """The RunConfig this run is (AeroBO's reproduce snippet). With the
        plate's lean or root blend handed to the optimiser -- or stated off the
        upright crease (`bridge.plate_seeded`) -- it carries AeroBO's
        `x_seed`: a plate that reaches the deck, where one flies
        (`bridge.plate_seed`) -- the one guarantee that such a search, whose
        feasible part of the box is small, returns a plate that reaches the
        car on any seed. Absent otherwise, so the upright crease and the
        pylons are the configuration they always were."""
        eff = self.search()
        x_seed = None
        if bridge.plate_seeded(self.choices) and eff["optimiser"] in bridge.api.X_SEED_OPTIMISERS:
            x_seed = bridge.plate_seed(self.family_name, self.physics_flags(),
                                       self.bounds_overrides(), self.pinned() or None)
        return bridge.wing_cfg(self.family_name, self.physics_flags(), eff,
                               int(self.seed), self.bounds_overrides(), self.pinned() or None,
                               x_seed=x_seed)

    def _plate_span_ride(self, out: dict) -> tuple:
        """`(b_hi, ride_lo)` of the box `bounds_overrides` is building: the
        widest span (a fixed span: its value; a released one: the family's
        own band) and the lowest ride (a typed or fixed ride row; None: the
        slot's band). No build -- `bounds_overrides` reads it."""
        pins = self.hand_pins()
        b_row = out.get("b_m") or (self.own_band("b_m") if "b_m" in self.released else None)
        b_hi = pins.get("b_m", None if b_row is None else float(b_row[1]))
        ride = out.get("ride_height_m")
        ride_lo = pins.get("ride_height_m", None if ride is None else float(ride[0]))
        return b_hi, ride_lo

    def plate_fit_problem(self) -> str | None:
        """Why the card's plate leaves no wing on this slot
        (`bridge.plate_fit_problem`, on the box the run would search) -- or
        None. Run refuses with it rather than fly a box with no design in it;
        never while `bridge.plate_seed` still finds one that flies."""
        try:
            box = self.family_box()
            pins = self.pinned()
        except Exception:                                   # noqa: BLE001 -- the run says why
            return None

        def at(lab, end):
            #  a fixed row is its value; else the searched band's `end`
            if lab in pins:
                return float(pins[lab])
            row = box.get(lab)
            return None if row is None else float(row[end])
        b = at("b_m", 1)
        if b is None:
            return None
        why = bridge.plate_fit_problem(self.session.op, self.choices, b, at("endplate_h_m", 1),
                                       at("ride_height_m", 0), at("S_m2", 0))
        if why and bridge.plate_seeded(self.choices):
            try:
                if bridge.plate_seed(self.family_name, self.physics_flags(),
                                     self.bounds_overrides(), self.pinned() or None) is not None:
                    return None
            except Exception:                               # noqa: BLE001 -- keep the refusal
                pass
        return why

    def bo_note(self) -> str:
        """AeroBO's own sentence for a machine without torch ("" with it):
        the non-torch optimisers fly instead (PLAN2 §11 Q10)."""
        return bridge.torch_note()

    # -- the forms ------------------------------------------------------------------
    #: the Wing type answers that pick a different FAMILY (a different box):
    #: the mount, the chord law, and whether AeroBO searches the plate's blend
    #: and lean (a searched one is a row, `api.PLATE_FREEDOMS`)
    FAMILY_KEYS = ("plates", "law", "blend", "cant")

    def shows_blend_field(self) -> bool:
        """Is "Root blend (0-1)" the player's to type? Under the endplates,
        stated. (Optimised, it is the box's endplate_blend_frac row.)"""
        return bool(self.choices["plates"]) and self.choices.get("blend") != "free"

    def shows_cant_field(self) -> bool:
        """Is the plate's "Leaning at" the player's to type? Under the
        endplates, stated (optimised: the box's endplate_cant_deg row)."""
        return bool(self.choices["plates"]) and self.choices.get("cant") != "free"

    def shows_tip_cant_field(self) -> bool:
        """...and the tip device's: under the pylons, canted only (the
        fence family has no searched lean)."""
        return (not self.choices["plates"]) and bridge.tip_device(self.choices) == "canted"

    def _set_choice(self, key: str):
        def f(v):
            if key == "plates":
                #  "Carried by": the view's endplates | pylons; the older
                #  designed | fences still read (the garage's checks set them)
                on = v if isinstance(v, bool) else str(v) in ("endplates", "designed")
                if bool(on) != bool(self.choices["plates"]):
                    self.session.log(
                        "carried by: the endplates — a designed part, section in stage 2.8"
                        if on else
                        f"carried by: a pylon pair at {bridge.carmount.INBOARD_STATION_FRAC:g} of "
                        f"each half-span — the plates are a tip device "
                        f"({bridge.tip_device(self.choices)})")
                self.choices["plates"] = bool(on)
            elif key == "law":
                self.choices["chord_law"] = (str(v) == "free")
            elif key in ("blend", "cant"):
                self.choices[key] = "free" if str(v) == "free" else "stated"
            if key in self.FAMILY_KEYS:
                self._family_moved()
            self._box_form = None
            self.session.on_family()
        return f

    def _family_moved(self) -> None:
        """A different family is a different box: the rows the player
        narrowed, released or fixed may not exist in it -- and a different
        objective menu (lap time comes and goes with the pylons on the top
        wing)."""
        labels = set(self._labels_safe())
        self.box = {k: v for k, v in self.box.items() if k in labels}
        self.fixed = {k: v for k, v in self.fixed.items() if k in labels}
        self.released = {k for k in self.released if k in labels}
        if not self.laptime_offered() and self.choices["objective"] == "laptime":
            self.choices["objective"] = "efficiency"
        self.type_params.param("obj").choices = list(self.objectives())

    def _set_number(self, key: str):
        """The typed plate numbers, held inside their bands: the blend 0-1
        of the plate's height, a lean (the plate's or the canted device's)
        `bridge.CANT_LIMITS` deg -- the nonplanar lattice's own validity."""
        lo, hi = (0.0, 1.0) if key == "blend_frac" else bridge.CANT_LIMITS

        def f(v):
            self.choices[key] = float(min(max(float(v), lo), hi))
            self.session.on_family()
        return f

    def _set_tip(self, v) -> None:
        """The pylons' tip device. "none" pins the plate's height at 0
        (`auto_pins`); leaving it searches the height again. "canted" with
        the lean still upright takes 75° -- a canted plate at 90° would be
        the vertical one under another name -- and says so."""
        tip = str(v) if str(v) in bridge.TIP_DEVICES else bridge.TIP_DEFAULT
        was = bridge.tip_device(self.choices)
        self.choices["tip"] = tip
        if tip == "canted" and float(self.choices.get("tip_cant_deg") or 90.0) >= bridge.CANT_UPRIGHT:
            self.choices["tip_cant_deg"] = bridge.CANT_SHOWN
            self.session.log(f"tip device: canted — leaning at {bridge.CANT_SHOWN:g}° (90° is the "
                             f"vertical plate); change it under Leaning at")
        elif tip == "none":
            self.session.log("tip device: none — no plates; endplate_h_m pinned at 0, one row "
                             "fewer searched")
        elif tip != was:
            self.session.log(f"tip device: {tip}" + (" — the plate's height is searched again"
                                                      if was == "none" else ""))
        self._box_form = None
        self.session.on_family()

    def shows_tip_chord(self) -> bool:
        """...and its chord: under the pylons, any device but "none"."""
        return (not self.choices["plates"]) and bridge.tip_device(self.choices) != "none"

    def _set_tip_chord(self, v) -> None:
        """Does the tip device's chord follow the wing's chord distribution
        ("follows") or hold the wing's tip chord ("tip")? A flag, not a
        design row: the box keeps its rows, the run is a new one."""
        on = v if isinstance(v, bool) else str(v) == "follows"
        if bool(on) != bridge.tip_chord_follows(self.choices):
            self.session.log("tip device: its chord follows the wing's chord distribution"
                             if on else "tip device: its chord holds the wing's tip chord "
                             "(a rectangle, WingLab's published plate)")
        self.choices["tip_chord"] = bool(on)
        self._box_form = None
        self.session.on_family()

    def reopen(self, record: dict) -> None:
        """Re-open the card on the answers a stored record was flown with
        (`bridge.choices_of_record`: its mount, freedoms, stated numbers,
        tip device, objective and limits). A legacy fence record re-opens on
        the pylons with vertical plates, the nearest answer still offered."""
        self.choices.update(bridge.choices_of_record(record, self.choices))
        self._family_moved()
        self._box_form = None
        self.session.on_family()

    def _labels_safe(self) -> list:
        try:
            return list(self.built().param_labels)
        except Exception:                             # noqa: BLE001
            return []

    def _set_objective(self, v) -> None:
        if str(v) in self.objectives():
            self.choices["objective"] = str(v)
            self._box_form = None

    def _set_limit(self, key: str):
        def f(v):
            v = float(v)
            self.choices[key] = None if v <= 0.0 else v
        return f

    def _type_rows(self) -> list:
        P = ui.Param
        return [
            P("t", "WING TYPE  (what is being built)", None, kind="label"),
            #  ONE question, asked first, as AeroBO's own card asks it: it
            #  picks the family and whether the plate is a part or a device.
            #  The key stays "plates" (the family, 2.8's lock, the objective
            #  menu and the records all read it)
            P("plates", "carried by", lambda: "endplates" if self.choices["plates"] else "pylons",
              self._set_choice("plates"), kind="choice", choices=["endplates", "pylons"],
              help="endplates: the plates hold the wing up at its tips -- a designed part "
                   "with its own section (stage 2.8). pylons: a swan-neck pair holds it "
                   "inboard, and the plates are only a tip device -- the only family "
                   "WingLab times a lap on"),
            P("blend", "root blend", lambda: self.choices.get("blend", "stated"),
              self._set_choice("blend"), kind="choice", choices=["stated", "free"],
              enabled=lambda: bool(self.choices["plates"]),
              help="stated: you type it. free: WingLab searches it (the Design box's "
                   "endplate_blend_frac row)"),
            P("blend_frac", "root blend (0–1)", lambda: float(self.choices.get("blend_frac") or 0.0),
              self._set_number("blend_frac"), step=0.05, fine=0.01, lo=0.0, hi=1.0,
              fmt="{:.2f}", enabled=self.shows_blend_field,
              help="how far up the plate the wing/plate corner is rounded off: 0 is a sharp "
                   "corner"),
            P("cant", "leaning at", lambda: self.choices.get("cant", "stated"),
              self._set_choice("cant"), kind="choice", choices=["stated", "free"],
              enabled=lambda: bool(self.choices["plates"]),
              help="stated: you type it. free: WingLab searches it (the Design box's "
                   "endplate_cant_deg row)"),
            P("cant_deg", "leaning at", lambda: float(self.choices.get("cant_deg") or 90.0),
              self._set_number("cant_deg"), step=5.0, fine=1.0, lo=bridge.CANT_LIMITS[0],
              hi=bridge.CANT_LIMITS[1], unit="°", fmt="{:.0f}", enabled=self.shows_cant_field,
              help="degrees from the wing plane: 90 is upright, less leans the plate outboard"),
            P("tip_cant_deg", "leaning at", lambda: float(self.choices.get("tip_cant_deg") or 90.0),
              self._set_number("tip_cant_deg"), step=5.0, fine=1.0, lo=bridge.CANT_LIMITS[0],
              hi=bridge.CANT_LIMITS[1], unit="°", fmt="{:.0f}", enabled=self.shows_tip_cant_field,
              help="the canted tip device's lean, degrees from the wing plane: 90 would be the "
                   "vertical plate"),
            P("tip", "tip device", lambda: bridge.tip_device(self.choices), self._set_tip,
              kind="choice", choices=list(bridge.TIP_DEVICES),
              enabled=lambda: not self.choices["plates"],
              help="; ".join(f"{k}: {v}" for k, v in bridge.TIP_DEVICES.items())),
            P("tip_chord", "its chord",
              lambda: "follows" if bridge.tip_chord_follows(self.choices) else "tip",
              self._set_tip_chord, kind="choice", choices=["follows", "tip"],
              enabled=self.shows_tip_chord,
              help="follows: the tip device's chord continues the wing's chord distribution "
                   "past the tip (a tapered wing gets tapered plates). tip: it holds the "
                   "wing's tip chord the whole way (a rectangle, WingLab's published plate)"),
            P("law", "chord law", lambda: "free" if self.choices["chord_law"] else "straight",
              self._set_choice("law"), kind="choice", choices=["free", "straight"],
              help="free: WingLab's three chord-law rows (k1..k3) on top of the taper. "
                   "straight: a plain taper"),
            P("o", "OBJECTIVE", None, kind="label"),
            P("obj", "maximise", lambda: self.choices["objective"], self._set_objective,
              kind="choice", choices=list(self.objectives()),
              help="; ".join(f"{k}: {OBJECTIVE_NOTE.get(k, '')}" for k, _l in CAR_OBJECTIVES)),
            P("cap", "drag ceiling", lambda: float(self.choices["drag_budget_n"] or 0.0),
              self._set_limit("drag_budget_n"), step=5.0, fine=1.0, lo=0.0, hi=2000.0,
              unit="N", fmt="{:.0f}",
              help="drag_budget_n: the most drag a design may make. 0 = no ceiling"),
            P("floor", "force floor", lambda: float(self.choices["downforce_min_n"] or 0.0),
              self._set_limit("downforce_min_n"), step=10.0, fine=2.0, lo=0.0, hi=5000.0,
              unit="N", fmt="{:.0f}",
              help="downforce_min_n: the least force a design may make. 0 = no floor"),
            P("n", "SECTIONS", None, kind="label"),
            P("to1", "change in stage 1", None, lambda _: self.session.goto("m.operating"),
              kind="action"),
            P("to2", "change in stage 2", None, lambda _: self.session.goto(
                "af.rank" if self.session.af.ranked else "af.screen"), kind="action"),
            P("to28", "change in stage 2.8", None, lambda _: self.session.goto("ep.screen"),
              kind="action", enabled=lambda: self.choices["plates"]),
        ]

    def box_params(self) -> Form:
        """The Design box form, one block per row of the built problem:
        min, max, constrain (the player's band on), fix (pinned at the
        current mid value). Rebuilt when the family's labels change."""
        key = tuple(self._labels_safe())
        if self._box_form is None or self._box_form_key != key:
            self._box_form = Form(self._box_rows(key), title="")
            self._box_form_key = key
        return self._box_form

    def _box_get(self, lab: str, end: int):
        def f():
            if lab in self.pinned():
                return float(self.pinned()[lab])
            if lab in self.box and lab not in self.released:
                return float(self.box[lab][end])
            fb = self.family_box().get(lab)
            return float(fb[end]) if fb else 0.0
        return f

    def _box_set(self, lab: str, end: int):
        def f(v):
            if lab in NO_BOX_CONTROLS:
                return                                  # not the player's row
            fb = self.default_band(lab)
            row = list(self.box.get(lab) or fb)
            v = self._under_cap(lab, float(v))
            pad = 1e-4
            if end == 0:
                if v > row[1] - pad:
                    self.session.say(f"{lab}: low held under high ({row[1]:.4g}) -- raise high first",
                                     "info")
                row[0] = min(v, row[1] - pad)
            else:
                if v < row[0] + pad:
                    self.session.say(f"{lab}: high held over low ({row[0]:.4g}) -- lower low first",
                                     "info")
                row[1] = max(v, row[0] + pad)
            self.box[lab] = row
            self.released.discard(lab)
        return f

    def _under_cap(self, lab: str, v: float) -> float:
        """`v`, held at the car's ceiling for a span / area row (task 41: its
        span limit in Real mode), saying so."""
        cap = (self.session.op.size_caps or {}).get(lab)
        if cap is not None and v > float(cap):
            self.session.say(f"{lab} held at {float(cap):.3g}: "
                             + bridge.limit_words(self.role, self.session.car,
                                                  self.session.slot.h,
                                                  self.session.unlimited), "info")
            return float(cap)
        return v

    def default_band(self, lab: str) -> list:
        """The band a row opens on before the player touches it: the
        family's own (AeroBO default, carsim packaging, the slot's band)."""
        saved_box, saved_rel = self.box, self.released
        try:
            self.box, self.released = {}, set()
            fb = self.family_box().get(lab)
        finally:
            self.box, self.released = saved_box, saved_rel
        return [float(fb[0]), float(fb[1])] if fb else [0.0, 1.0]

    def _set_constrain(self, lab: str):
        """AeroBO's constrain switch: ON (every row opens on it) the row is
        searched inside the band on the form; OFF it is RELEASED -- still
        designed, over the family's own published band -- and the numbers
        typed are kept for when it is switched back on. A fixed row has no
        band to constrain."""
        def f(v):
            if lab in NO_BOX_CONTROLS or lab in self.pinned():
                return                                  # not the player's row / fixed
            if bool(v) and lab in self.released:
                self.released.discard(lab)
                self.session.log(f"{lab}: constrained again — searched inside the band on the form")
            elif not bool(v) and lab not in self.released:
                self.released.add(lab)
                self.session.log(f"{lab}: released — searched over WingLab's own band; "
                                 f"your numbers come back with the switch")
        return f

    def _set_fix(self, lab: str):
        def f(v):
            if lab in NO_BOX_CONTROLS:
                return                                  # not the player's row: no fix, no release
            if lab in bridge.tip_pins(self.choices):
                #  the card's pin (tip device "none"): releasing the plate's
                #  height IS fitting a plate again -- AeroBO's own reading
                if not bool(v):
                    self._set_tip(bridge.TIP_DEFAULT)
                return
            if bool(v):
                if lab in self.fixed:
                    return
                #  AeroBO's: it opens at the MIDDLE of the band the row is
                #  searched in, and fixing takes the release off
                lo, hi = self.family_box().get(lab, (0.0, 0.0))
                self.fixed[lab] = self._under_cap(lab, 0.5 * (float(lo) + float(hi)))
                self.released.discard(lab)
                self.session.log(f"{lab}: fixed at {self.fixed[lab]:.6g} — one row fewer searched; "
                                 f"type the value it is held at in its low field")
            elif lab in self.fixed:
                self.fixed.pop(lab, None)
                self.session.log(f"{lab}: searched again")
        return f

    def _set_fixed_value(self, lab: str):
        """The value a fixed row is held at, as typed: not clamped into its
        band (AeroBO's `fixed_value`: a fixed row is not searched, so its
        band says nothing; `bounds_overrides` widens it to hold the value)
        -- only under the car's ceiling."""
        def f(v):
            if lab not in self.fixed or lab in NO_BOX_CONTROLS:
                return
            self.fixed[lab] = self._under_cap(lab, float(v))
        return f

    def reset_box(self) -> None:
        self.box, self.released, self.fixed = {}, set(), {}
        self.session.log("design box reset to the family's own bands")

    def _box_rows(self, labels) -> list:
        P = ui.Param
        rows = [P("bx", "THE BOX  (what the search may propose)", None, kind="label")]
        for lab in labels:
            if lab in NO_BOX_CONTROLS:
                continue                    # shown read-only by the view, never a control
            for end, tag in ((0, "min"), (1, "max")):
                rows.append(P(f"bx.{lab}.{tag}", f"{lab}  {tag}", self._box_get(lab, end),
                              self._box_set(lab, end), step=0.01, fine=0.002, fmt="{:.4g}",
                              enabled=(lambda la=lab: la not in self.pinned()
                                       and la not in self.released)))
            rows.append(P(f"bx.{lab}.con", f"{lab}  constrain",
                          (lambda la=lab: la not in self.released and la not in self.pinned()),
                          self._set_constrain(lab), kind="bool",
                          enabled=(lambda la=lab: la not in self.pinned())))
            rows.append(P(f"bx.{lab}.fix", f"{lab}  fix", (lambda la=lab: la in self.pinned()),
                          self._set_fix(lab), kind="bool"))
            rows.append(P(f"bx.{lab}.val", f"{lab}  fixed at",
                          (lambda la=lab: float(self.pinned().get(la, 0.0))),
                          self._set_fixed_value(lab), step=0.01, fine=0.002, fmt="{:.4g}",
                          enabled=(lambda la=lab: la in self.hand_pins())))
        rows.append(P("boxreset", "Reset to the family's box", None, lambda _: self.reset_box(),
                      kind="action"))
        return rows

    def _solver_rows(self) -> list:
        P = ui.Param
        pol = self.session.policy
        own = lambda: not pol.recommended()                              # noqa: E731

        def budget():
            try:
                return int(self.search()["budget"])
            except Exception:                                       # noqa: BLE001
                return int(pol.own["wing"])
        return [
            P("s", "SOLVER", None, kind="label"),
            P("opt", "optimiser", lambda: str(self._safe_search().get("optimiser", "—")), None,
              kind="choice", choices=[], enabled=False,
              help="bo_slsqp: WingLab's GP-BO over the box, handed to SLSQP for the last "
                   "stretch -- what WingLab's own wing stage runs when the family takes it"),
            P("budget", "evaluations", budget, lambda v: pol.own.__setitem__("wing", int(v)),
              kind="int", lo=4, hi=1000, enabled=own,
              help="recommended: WingLab's measured budget for this problem at the effort on 1 "
                   "Mission > Search & budget (53 at balanced, 14-D). Own values: yours"),
            P("seed", "random seed", lambda: int(self.seed),
              lambda v: setattr(self, "seed", int(min(max(int(v), 0), 9999))), kind="int",
              lo=0, hi=9999),
            P("stopconv", "stop when it stops improving", lambda: bool(pol.stop_when_converged),
              lambda v: setattr(pol, "stop_when_converged", bool(v)), kind="bool",
              help="WingLab's ConvergenceStop at the plan's patience and tolerance; the "
                   "budget stays the backstop"),
            P("run", "Run  (O)", None, lambda _: self.start_run(), kind="action"),
            P("stop", STOP_TEXT, None, lambda _: self.session.stop(self), kind="action",
              enabled=lambda: self.session.runs.job_for(self) is not None),
            P("snip", "copy the run configuration to the output", None,
              lambda _: self.log_config(), kind="action"),
            P("mine", "use my own values", None, lambda _: pol.set_mode("own"), kind="action",
              enabled=lambda: pol.recommended()),
            P("s1", "stage 1 · Search & budget", None, lambda _: self.session.goto("m.search"),
              kind="action"),
        ]

    def _safe_search(self) -> dict:
        try:
            return self.search()
        except Exception:                                   # noqa: BLE001 -- a read-out
            return {}

    def _conv_rows(self) -> list:
        P = ui.Param
        return [
            P("c", "CONVERGENCE", None, kind="label"),
            P("more", "more evaluations", lambda: int(self.more or self.continue_extra()),
              lambda v: setattr(self, "more", int(min(max(int(v), 1), 1000))), kind="int",
              lo=1, hi=1000),
            _LiveLabel("go", self._keep_label, None, lambda _: self.start_run(extend=True),
                       kind="action", enabled=lambda: self.can_continue()),
            P("stop", STOP_TEXT, None, lambda _: self.session.stop(self), kind="action",
              enabled=lambda: self.session.runs.job_for(self) is not None),
            P("fit", "Put it on the car  (S)", None, lambda _: self.commit(), kind="action",
              enabled=lambda: self.record is not None),
        ]

    def _keep_label(self) -> str:
        if not self.can_continue():
            return "Keep going  (K)"
        more = int(self.more or self.continue_extra())
        return f"Keep going — {more} more ({int(self.record.get('n_evals') or 0) + more} in total)  (K)"

    def log_config(self) -> None:
        """AeroBO's reproduce snippet into the Output log: the RunConfig as
        `aerobo.api.run` takes it."""
        try:
            cfg = self.cfg()
        except Exception as exc:                            # noqa: BLE001
            self.session.log(f"no configuration: {exc}", "warn")
            return
        d = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(vars(cfg))
        self.session.log("api.run(api.RunConfig(" + ", ".join(
            f"{k}={d.get(k)!r}" for k in ("problem_name", "optimiser", "budget", "seed", "flags",
                                          "bounds_overrides", "pinned")) + "))")

    # -- the run ------------------------------------------------------------------
    def can_continue(self) -> bool:
        """Can Keep going RESUME this record (AeroBO's `can_resume`: a BO
        record carries its own evaluations as the next run's training set)?"""
        return bool(self.record and bridge.api.can_resume(self.record))

    def continue_extra(self) -> int:
        return int(bridge.continue_extra_default(self.record)) if self.record else 1

    def stale_plate_record(self) -> bool:
        """Is the record a run with NO feasible design, of a stated leaning
        or blended plate, flown without a seed -- a box from before its area
        floor followed the plates' reach (`bridge.plate_area_floor`)? Keep
        going continues that box, so it could only find nothing again."""
        rec = self.record or {}
        if rec.get("feasible"):
            return False
        cfgd = rec.get("config") or {}
        flags = cfgd.get("flags") or {}
        projecting = (float(flags.get("blend_frac") or 0.0) > 0.0
                      or float(flags.get("endplate_cant_deg", bridge.CANT_UPRIGHT))
                      != bridge.CANT_UPRIGHT)
        fam = rec.get("carsim_family") or {}
        carried = bool(fam.get("plates", True)) and not fam.get("pylons")
        return projecting and carried and not cfgd.get("x_seed")

    def start_run(self, extend: bool = False) -> bool:
        """Launch the wing search LIVE (O / Run): `api.run` on this slot's
        family with V3's configuration. `extend` is Keep going (K):
        `api.continue_run_config(record, extra)` then `api.run(resume=...)` --
        nothing is re-flown, the counter continues at k + 1. An objective the
        family cannot score, or a flag the engine refuses, ends the job
        FAILED with AeroBO's own sentence (`api.check_wing_objective`)."""
        s = self.session
        if not s.ready_to_run(self):
            return False
        results_dir = s.results_dir("wing")
        pol = s.policy
        if extend:
            if not self.can_continue():
                self.msg = "no finished wing run to continue: Run it first (O)"
                s.say(self.msg, "info")
                return False
            if self.stale_plate_record():
                self.msg = ("that run found no feasible design in a box too tight for its "
                            "plate, and Keep going searches the same box: Run it again (O) -- "
                            "the box now leaves the plates' reach out of the area")
                s.say(self.msg, "warning")
                return False
            extra = int(self.more or self.continue_extra())
            fp = bridge.FamilyParams.from_json(self.record["carsim_family"])
            cfg, note = bridge.continue_wing(self.record, extra)
            eff = {"budget": int(cfg.budget), "plan": None}
            try:
                eff["plan"] = self.plan()
            except Exception:                                   # noqa: BLE001
                pass
            resume, n_prior = note.get("resume"), int(note.get("resumed") or 0)
            variant = "continued"
        else:
            try:
                fp = self.family
                eff = self.search()
                cfg = self.cfg()
            except (ValueError, KeyError, TypeError) as exc:
                self.msg = f"this wing cannot be run as set: {exc}"
                s.say(self.msg, "warning")
                return False
            why = self.plate_fit_problem()
            if why:
                self.msg = f"this wing cannot be run as set: {why}"
                s.say(self.msg, "warning")
                return False
            resume, n_prior, variant = None, 0, "full"
        #  the Sobol block and the SLSQP handoff of a fresh bo_slsqp run, by
        #  AeroBO's own arithmetic, so the live graph marks both while it runs
        #  (a resumed run has no Sobol block: AeroBO's own handoff report)
        self._live_split = None
        if not extend:
            try:
                dim = getattr(eff.get("plan"), "dim", None) or (
                    len(self.family_box()) - len(self.pinned()))
                self._live_split = (cfg, int(dim), {})
            except Exception:                                   # noqa: BLE001 -- a read-out
                self._live_split = None
        conv = bridge.stop_rule_factory(eff, pol)
        n = int(cfg.budget)

        def engine():
            return bridge.wing_runner(cfg, fp, conv, resume, results_dir)
        runner = s.runner("wing", engine, role=self.role, variant=variant,
                          objective=self.choices["objective"])
        prior = list(self.records) if extend else []
        #  a continuation flies the record's own flags, so its sections too
        flown = dict(self._record_sections) if extend else self.sections()
        split = None
        if self._live_split:
            try:
                split = bridge.handoff_split(cfg, self._live_split[1], budget=n)
            except Exception:                                   # noqa: BLE001 -- a read-out
                split = None
        dim = (self._live_split[1] if self._live_split
               else (self.record or {}).get("searched_dim") or (self.record or {}).get("dim"))
        info = s.job_info("wing", self, objective=self.choices["objective"],
                          phrase=self.objectives().get(self.choices["objective"], ""),
                          continued=extend, optimiser=str(getattr(cfg, "optimiser", "") or ""),
                          n_init=(split[0] if split else None),
                          handoff=(split[1] if split else None),
                          problem=f"{fp.base}" + (f" ({int(dim)}-D)" if dim else ""),
                          airfoil=str((flown.get("main") or {}).get("name") or ""),
                          plate=str((flown.get("plate") or {}).get("name") or ""),
                          rseed=int(getattr(cfg, "seed", 0) or 0))
        job = dj.EngineJob(info, self, runner=runner, n=n, n_prior=n_prior,
                           on_finish=lambda j: self._run_done(j, prior, n_prior, flown),
                           notices=s.notices, replay=s.replay_of(runner))
        return s.launch(job)

    def _run_done(self, job, prior: list, n_prior: int, flown: dict | None = None) -> None:
        out = job.outcome()
        self.outcome = out
        if job.state == "error":
            self.msg = f"the wing run failed: {job.error}"
            return
        rec = (job.result or {}).get("record")
        if not rec:
            self.msg = "the wing run returned no record"
            return
        self.record, self.records, self.prior = rec, prior + list(job.records), int(n_prior)
        self._record_sections = dict(flown or {})
        self.more = None
        self.law = self.spec = None
        self.law_outcome = None
        self.slot_updates = {}
        self.session.results.report = None
        feas = "feasible" if rec.get("feasible") else "no feasible design"
        self.msg = (f"wing run {out['state']}: best {rec.get('best_score')} "
                    f"{rec.get('score_units') or ''} ({feas}, {rec.get('n_evals')} evaluations)")
        self.save_record()
        self.start_law()

    def save_record(self) -> str | None:
        """The run record, kept under runs/aerobo/<slot>/ (never in a
        check: the session's `results_dir` is None there)."""
        d = self.session.results_dir("records")
        if not d or not self.record:
            return None
        try:
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"wing_{self.record.get('timestamp') or int(1e3 * os.times()[4])}.json")
            with open(path, "w") as fh:
                json.dump(self.record, fh, default=_clean)
            self.record["record_path"] = path
            return path
        except OSError:
            return None

    def verdict(self) -> dict:
        """Did the run flatten out? AeroBO's `converged_report` on its own
        history at the plan's patience and tolerance."""
        if not self.record:
            return {"verdict": None, "text": "no run yet"}
        try:
            plan = self.plan()
            patience, tol = getattr(plan, "patience", None), getattr(plan, "tol", None)
        except Exception:                                       # noqa: BLE001
            patience, tol = None, None
        return bridge._budget.converged_report(self.record.get("history") or [], patience, tol)

    def graph(self) -> dict:
        """The live evaluation graph's inputs (PLAN2 §8.5) for w.conv."""
        s = self.session
        job = s.runs.job_for(self)
        live = job is not None and getattr(job.info, "kind", "") == "wing"
        recs = list(self.records)
        if live:
            recs = (recs if job.n_prior else []) + list(job.records)
        rec = self.record or {}
        handoff = rec.get("handoff") or {}
        if live:
            #  the record is the LAST run's: a live fresh run's split is
            #  AeroBO's own arithmetic on the N the job reports
            #  (`bridge.handoff_split`, memoised per N)
            n_init = n_a = None
            ls = getattr(self, "_live_split", None) if not job.n_prior else None
            if ls:
                cfg, dim, memo = ls
                if job.n not in memo:
                    try:
                        memo[job.n] = bridge.handoff_split(cfg, dim, budget=int(job.n))
                    except Exception:                       # noqa: BLE001 -- a read-out
                        memo[job.n] = None
                n_init, n_a = memo[job.n] or (None, None)
        else:
            #  a bo_slsqp record keeps its Sobol / BO split inside the handoff
            #  (its top-level bo_split is None): the BO leg is phase A
            split = (rec.get("bo_split")
                     or (handoff.get("bo_split") if isinstance(handoff, dict) else None)
                     or [None, None])
            try:
                n_init = (split[0] if split and split[0] is not None
                          else self._safe_search().get("n_init"))
            except Exception:                               # noqa: BLE001
                n_init = None
            n_a = handoff.get("n_a") if isinstance(handoff, dict) else None
        N = int(job.n) if live else int((self.outcome or {}).get("n") or rec.get("n_evals") or 0)
        return dict(records=recs, N=N, n_prior=int(job.n_prior if live else self.prior),
                    n_init=int(n_init) if n_init else None,
                    handoff=(int(n_a) if n_a else None),
                    units=str(rec.get("score_units") or self.choices["objective"]),
                    objective=self.choices["objective"], live=live,
                    tag=(job.chip() if live else outcome_tag(self.outcome)))

    # -- onto the car --------------------------------------------------------------
    def slot_geom(self) -> dict:
        s = self.session.slot
        return {"x": float(s.x), "h": float(s.h), "V": float(self.session.op.V),
                "role": self.role, "key": self.key}

    def has_winner(self) -> bool:
        """Did the run find a design that FLIES: feasible, with a finite score?
        A run whose every evaluation missed a constraint or was refused has
        nothing to derive a law from or to put on the car."""
        rec = self.record or {}
        best = rec.get("best_score")
        try:
            finite = best is not None and math.isfinite(float(best))
        except (TypeError, ValueError):
            finite = False
        return bool(rec.get("feasible")) and finite

    def start_law(self) -> bool:
        """Derive the car's law from the finished run, as a JOB: an incidence
        sweep of AeroBO's own evaluator at the winning design (~100
        evaluations, well under a second, no XFOIL)."""
        s = self.session
        if not self.record or s.runs.busy:
            return False
        if not self.has_winner():
            self.msg = "no feasible design to put on the car: every evaluation missed a constraint or was refused -- Keep going (K) for more evaluations, or widen the Design box"
            return False
        #  not replayed (Replay's docstring): the law is a pure function of
        #  the record, so the checks derive it for real on a replayed one
        runner = bridge.law_runner(self.record, self.slot_geom())
        info = s.job_info("law", self, objective="law", phrase="the car's law from WingLab")
        job = dj.EngineJob(info, self, runner=runner, n=1, unit="evaluation",
                           on_finish=self._law_done, notices=s.notices)
        return s.launch(job, quiet=True)

    def _law_done(self, job) -> None:
        self.law_outcome = job.outcome()
        if job.state == "error":
            self.msg = f"no law: {job.error}"
            return
        self.law = job.result
        self.session.results.start_report()

    def derive_now(self) -> dict | None:
        """The law, derived SYNCHRONOUSLY (a commit, a save, a drive: PLAN2
        §5.7 -- a few hundred milliseconds of AeroBO's evaluator)."""
        if not self.record or not self.has_winner():
            return None
        self.law = bridge.derive_law(self.record, self.slot_geom())
        return self.law

    def fit(self, name: str | None = None) -> None:
        """The run's winner as a carsim `WingSpec`: AeroBO's geometry (the
        straight-taper equivalent of its chord law -- PLAN2 H6), the imported
        sections, the law in `aero`, AeroBO's provenance in `design`. The
        slot updates (incidence, the top wing's ride height) wait for
        `commit`. Marks the wing dirty (leaving the page saves it)."""
        if not self.record:
            self.msg = "no wing run yet: Run it first (O)"
            return
        if not self.has_winner():
            self.msg = "no feasible design to put on the car: every evaluation missed a constraint or was refused -- Keep going (K) for more evaluations, or widen the Design box"
            return
        if self.law is None or self._law_stale():
            self.derive_now()
        s = self.session
        nm = name or (self.spec.name if self.spec is not None else
                      s.host.lib.unique_name("wings", f"{'flank' if self.role == 'flank' else 'rear'}-winglab"))
        #  the sections the RECORD flew -- not what 2 / 2.8 hold now: the law
        #  is the record's, and a wing named after another section would lie
        sections = dict(self._record_sections or {})
        self.spec, self.slot_updates = bridge.to_wingspec(self.record, self.law, sections,
                                                          self.role, nm)
        self._section_specs = sections
        self.dirty = True
        self.msg = f"'{nm}' fitted from WingLab's winner — S saves it into the slot"

    def _law_stale(self) -> bool:
        at = (self.law or {}).get("derived_at") or {}
        g = self.slot_geom()
        return any(abs(float(at.get(k, g[k])) - g[k]) > 1e-9 for k in ("V",))

    def commit(self, name: str | None = None) -> str:
        """Save the fitted wing -- and the sections it flies, as `coords`
        AirfoilSpecs with their AeroBO origin -- into the library, put it in
        the slot (mirrored), write the slot updates. dirty -> False."""
        s = self.session
        if self.spec is None or name:
            self.fit(name)
        if self.spec is None:
            s.say(self.msg or "nothing to put on the car yet", "warning")
            return ""
        lib = s.host.lib
        spec = self.spec
        if spec.name in lib.wings and lib.wings[spec.name].builtin:
            spec.name = lib.unique_name("wings", spec.name)
        build = s.host.build
        #  task 41, Real mode: no wing past its car's physical span limit goes
        #  on the car -- in this slot, or in any other slot of its role that
        #  already carries a wing of this NAME (re-saving it re-spans them
        #  all) unless the mirror is about to copy this one over it (the
        #  garage's W, library page and old designer hold the same rule)
        keys = [self.key] + [k for k in ("left", "right", "top")
                             if k != self.key and (k == "top") == (self.role == "top")
                             and build.slot(k).wing == spec.name
                             and not (getattr(build, "mirror", False) and self.role == "flank")]
        past = s.past_limit(float(spec.span), keys)
        if past is not None:
            k, lim = past
            self.msg = (f"'{spec.name}' ({float(spec.span):.2f} m) is past the {k} slot's "
                        f"{lim:.2f} m span limit on the {s.car}: not saved (narrow the span row "
                        f"of the Design box, or raise the slot; Settings > Wing limits: "
                        f"Unlimited allows it)")
            s.say(self.msg, "warning")
            s.log(self.msg, "warn")
            return ""
        try:
            #  the sections first: a wing refers to them by name
            for part, attr in (("main", "airfoil"), ("plate", "plate_airfoil")):
                chosen = (getattr(self, "_section_specs", {}) or {}).get(part)
                a = bridge.section_to_airfoilspec(chosen, lib) if chosen else None
                if a is not None:
                    same = _same_section(lib, a)
                    if same is None:
                        lib.save_airfoil(a)
                    setattr(spec, attr, same or a.name)
            #  ...and the family's OWN sections where none was chosen: AeroBO
            #  flies NACA 24tt / 00tt at the searched thickness, which carsim's
            #  library may not hold (the renderer lofts the wing from its name)
            for attr in ("airfoil", "plate_airfoil"):
                nm = getattr(spec, attr, "")
                if nm and nm not in lib.airfoils:
                    a = bridge.section_to_airfoilspec({"source": "default", "name": nm}, lib)
                    if a is not None:
                        lib.save_airfoil(a)
                        setattr(spec, attr, a.name)
            spec.builtin, spec.legacy = False, None
            lib.save_wing(spec.copy())
        except (OSError, ValueError) as exc:
            #  a full disk, a read-only runs/, a name another record's file
            #  holds (task 39's save guard): the garage stays up and says so
            self.msg = f"could not save: {getattr(exc, 'strerror', None) or exc}"
            s.say(self.msg, "warning")
            s.log(self.msg, "error")
            return ""
        slot = build.slot(self.key)
        slot.wing = spec.name
        for k, v in (self.slot_updates or {}).items():
            setattr(slot, k, float(v))
        build.sync_mirror(self.key)
        if hasattr(build, "clamp"):
            try:
                build.clamp(lib, s.car)             # fitted to the car (task 41)
            except TypeError:
                build.clamp(lib)                    # a stand-in build without a car
        #  `clamp` can move the slot (the top's height floor); the law was
        #  derived at the slot as it was written, so re-derive it if it moved
        saved = lib.wings.get(spec.name)
        if saved is not None and law_stale(saved, slot):
            try:
                if rederive(saved, slot, s._deck_z, car=s.car):
                    lib.save_wing(saved)
                    spec.aero = dict(saved.aero)
            except Exception as exc:                            # noqa: BLE001 -- the law stands
                s.log(f"the law could not be re-derived at the clamped slot ({exc})", "warn")
        self.dirty = False
        self.msg = f"saved '{spec.name}' to the library and put it in the {self.key} slot"
        s.log(self.msg, "ok")
        #  (Unlimited mode) a wing past the limit is saved, and says what that means
        if s.unlimited:
            over = bridge.over_limits(build, lib, s.car)
            if over:
                s.log("UNLIMITED: runs with this build are filed apart, never official ("
                      + bridge.limits_text(over) + ")", "warn")
        s.on_commit()
        return spec.name

    # -- housekeeping -----------------------------------------------------------------
    def reset(self) -> None:
        self.record, self.records, self.outcome, self.prior, self.more = None, [], None, 0, None
        self._record_sections, self._section_specs, self._live_split = {}, {}, None
        self.law = self.spec = None
        self.law_outcome = None
        self.slot_updates = {}
        self.dirty = False
        self.msg = ""


# --------------------------------------------------------------------------- #
#  results (stage 4)                                                           #
# --------------------------------------------------------------------------- #
class ResultsModel:
    """Stage 4: what the run found, read off AeroBO's record and report,
    and what the CAR makes of it -- the law, the game's force at V against
    AeroBO's (equal by construction), and carsim's own lap with and without
    the wing as a cross-check, labelled as the second model it is."""

    def __init__(self, session: "DesignSession"):
        self.session = session
        self.report: dict | None = None
        self.outcome: dict | None = None
        self.msg = ""
        P = ui.Param
        w = session.wing
        self.params = Form([
            P("res", "RESULTS", None, kind="label"),
            _LiveLabel("keep_r", w._keep_label, None, lambda _: w.start_run(extend=True),
                       kind="action", enabled=lambda: w.can_continue() and not session.runs.busy,
                       help="a resume, not a re-run: the evaluations already paid for are the "
                            "training set, only the new designs are flown"),
            P("save", "Put it on the car  (S)", None, lambda _: w.commit(), kind="action",
              enabled=lambda: w.record is not None),
            P("to3", "Go to the wing stage", None, lambda _: session.goto("w.solver"),
              kind="action"),
            P("tobox", "Open the design box", None, lambda _: session.goto("w.box"),
              kind="action"),
        ], title="")

    def start_report(self) -> bool:
        """AeroBO's `design_report` for the winner (the geometry and the
        spanwise breakdown the Geometry and Loading views draw), as a job."""
        s = self.session
        rec = s.wing.record
        if not rec or s.runs.busy:
            return False
        runner = bridge.design_report_runner(rec)          # never replayed, like the law
        info = s.job_info("report", self, objective="report", phrase="the design report")
        job = dj.EngineJob(info, self, runner=runner, n=1, unit="evaluation",
                           on_finish=self._report_done, notices=s.notices)
        return s.launch(job, quiet=True)

    def _report_done(self, job) -> None:
        self.outcome = job.outcome()
        if job.state == "error":
            self.msg = f"no design report: {job.error}"
            return
        self.report = job.result

    def summary(self) -> dict:
        """The r.summary fields (PLAN2 §8.4): the score and its units, the
        breakdown's forces and coefficients, the constraint margins, the run's
        bookkeeping, the sections flown, and the car mapping."""
        w = self.session.wing
        rec = w.record
        if not rec:
            return {}
        bd = rec.get("breakdown") or {}
        law = w.law or {}
        labels = list(rec.get("param_labels") or [])
        x = list(rec.get("best_x") or [])
        #  the margins one by one are the breakdown's `g`, in its
        #  `constraint_labels` order; AeroBO's `best_g` is ONE number, the
        #  winner's tightest margin (the least of them)
        cons = list(bd.get("constraint_labels") or [])
        if not cons:
            try:
                cfgd = rec.get("config") or {}
                cons = list(bridge.api.constraint_labels_of(bridge.family_of(rec),
                                                            cfgd.get("flags")))
            except Exception:                                       # noqa: BLE001 -- a read-out
                pass
        g = bd.get("g")
        g = list(g) if isinstance(g, (list, tuple)) else []
        force_word = "side force" if w.role == "flank" else "downforce"
        flown = dict(w._record_sections or {})
        plates = bool((rec.get("carsim_family") or {}).get("plates", w.choices["plates"]))
        out = dict(
            best_score=_finite(rec.get("best_score")), score_units=rec.get("score_units") or "",
            objective=w.choices["objective"], feasible=bool(rec.get("feasible")),
            n_feasible=rec.get("n_feasible"), n_evals=rec.get("n_evals"),
            budget=(rec.get("config") or {}).get("budget"), wall_s=_finite(rec.get("wall_time_s")),
            partial=bool(rec.get("partial")), stop_reason=rec.get("stop_reason"),
            resumed=rec.get("resumed"), bo_split=rec.get("bo_split"), handoff=rec.get("handoff"),
            CZ=_finite(bd.get("CZ")), CD=_finite(bd.get("CD")),
            force_N=_finite(bd.get("downforce_N")), drag_N=_finite(bd.get("drag_N")),
            force_word=force_word,
            efficiency=(_finite(bd.get("CZ")) / _finite(bd.get("CD"))
                        if _finite(bd.get("CZ")) is not None and (_finite(bd.get("CD")) or 0) > 0
                        else None),
            lap_time_s=_finite(bd.get("lap_time_s")),
            margins=[(c, _finite(v)) for c, v in zip(cons, g)],
            best_g=_finite(rec.get("best_g")),
            design=dict(zip(labels, x)),
            #  the sections the RECORD flew, in the family it flew
            sections={"main": (flown.get("main") or {}).get("name")
                      or "the family's published section",
                      "plate": (((flown.get("plate") or {}).get("name")
                                 or "NACA 00tt (the family's own)") if plates else "fences")},
            outcome=w.outcome, tag=outcome_tag(w.outcome),
            law={k: law.get(k) for k in ("CL0", "CLa", "CL_min", "CL_max", "cd0", "cd1", "cd2",
                                         "cd_fit_err", "alpha_stall_deg", "alpha_stall_neg_deg",
                                         "S", "AR", "e")} if law else None,
        )
        if law:
            inc = float((w.slot_updates or {}).get("inc_deg", law.get("alpha_design_deg", 0.0))
                        if w.slot_updates else law.get("alpha_design_deg", 0.0))
            out["game_force_N"] = self.game_force(law, inc)
            out["aerobo_force_N"] = _finite((law.get("aerobo") or {}).get("F_N"))
            out["inc_deg"] = inc
        return out

    @staticmethod
    def game_force(law: dict, inc_deg: float) -> float:
        """The force the GAME computes from the law at its design speed:
        `0.5 rho V^2 S CZ(inc)` with carsim's rho -- what `TopAero` /
        `DevAero` hand the 1 kHz step."""
        cz = law["CL0"] + law["CLa"] * math.radians(float(inc_deg))
        cz = min(max(cz, law["CL_min"]), law["CL_max"])
        return 0.5 * float(law.get("rho", 1.2)) * float(law["V_ref"]) ** 2 * float(law["S"]) * cz

    def car_lap(self) -> dict:
        """carsim's own cross-check of this wing on the car against the same
        car with the slot empty, in the JOB's terms -- carsim's QSS, a
        DIFFERENT model from AeroBO's, shown as a cross-check:

        * a circuit (the side wings' own, `DesignSession.mission`): the lap
          time (`with_s` / `without_s` / `delta_s`);
        * STOPPING: the stop's distance from its start speed (`aero.mission.stop`;
          a side wing's is the air brake's, both panels out).

        Every kind also fills `kind`, `title`, `with`, `without`, `delta`,
        `unit`, `fmt` and `lower_better`, which is what the view reads."""
        s, w = self.session, self.session.wing
        if not w.law:
            return {}
        host = s.host
        job = s.job()
        try:
            base = host.build.mission_aero(host.lib, exclude=s.key)
            slot = s.slot
            inc = float((w.slot_updates or {}).get("inc_deg", slot.inc_deg))
            h = float((w.slot_updates or {}).get("h", slot.h))
            wing = ms.merge_wing(base, w.law, w.role, inc, slot.x, h, getattr(slot, "mode", "active"))
            m = s.mission
            mu = float(m.mu_scale)
            if job == ms.STOPPING:
                v0, n = float(m.v_stop), m.stop_flanks
                a, b = ms.stop(v0, wing, mu, flanks=n), ms.stop(v0, base, mu, flanks=n)
                da = a.distance if a.ok else None
                db = b.distance if b.ok else None
                return dict(kind="stop", title=f"the stop from {m.v_stop_kmh:.0f} km/h"
                                                + (" on the air brake" if n > 1 else ""),
                            **{"with": da, "without": db,
                               "delta": (da - db) if (a.ok and b.ok) else None},
                            unit="m", fmt="{:.2f}", dfmt="{:+.2f}", lower_better=True,
                            model="carsim's quasi-steady stop (the lap's own braking: tyre-"
                                  "limited on four wheels with the downforce of the moment, "
                                  "plus every drag) -- a second model, not WingLab's")
            prof = m.profile(make_track)
            a, b = ms.lap(prof, wing, mu_scale=mu), ms.lap(prof, base, mu_scale=mu)
        except Exception as exc:                                    # noqa: BLE001 -- a read-out
            return {"error": f"{type(exc).__name__}: {exc}"}
        ta, tb = (a.time if a.ok else None), (b.time if b.ok else None)
        d = a.time - b.time if (a.ok and b.ok) else None
        return dict(kind="lap", title=f"the {m.track} lap",
                    with_s=ta, without_s=tb, delta_s=d, **{"with": ta, "without": tb, "delta": d},
                    unit="s", fmt="{:.3f}", dfmt="{:+.3f}", lower_better=True,
                    model="carsim's quasi-steady lap (a second model: WingLab's own lap, where "
                          "it is the objective, is cartrack's point mass)")


# --------------------------------------------------------------------------- #
#  one slot's session                                                          #
# --------------------------------------------------------------------------- #
class DesignSession:
    """Everything the DESIGN page holds for ONE slot ("left" | "right" |
    "top"; a mirrored right flank is the left one's): the operating point,
    the two section surfaces, the wing and its results, on the garage's one
    RunManager and its one SearchPolicy.

    `host` is the garage (duck-typed: `lib`, `build`, `mission`, `runs`,
    `notices`, `log`, `toast`; optional `say`, `goto_view`, `aerobo_dir`,
    and task 41's `car` -- the `cars.py` key the build is fitted to, the
    Corsa when absent -- and `unlimited`, Settings' Wing limits). `deck_z(x)`,
    when given, overrides the car's top surface height at station x (the
    self-checks' reference deck); by default it is the car's own
    (`bridge.car_deck`, from drive/bodies.py -- this module may not import
    garage)."""

    def __init__(self, host, key: str, policy: SearchPolicy, deck_z=None):
        self.host, self.key = host, key
        self.role = "top" if key == "top" else "flank"
        self.policy = policy
        self._deck_z = deck_z
        #: None = follow the module's replay switch (`use_fixtures`)
        self.replay: Replay | None = None
        self.V_typed: float | None = None
        self.cz_typed: float | None = None
        self.signature = None
        self.op = self._operating_point()
        self.af = SurfaceModel(self, "main")
        self.ep = SurfaceModel(self, "plate")
        self.wing = WingModel(self)
        self.results = ResultsModel(self)
        #  the objective the job opens on shows before the job is stated
        self.wing.default_objective(quiet=True)

    # -- the host ------------------------------------------------------------------
    @property
    def runs(self):
        return self.host.runs

    @property
    def notices(self):
        return self.host.notices

    @property
    def slot(self):
        return self.host.build.slot(self.key)

    # -- the car (task 41) -------------------------------------------------------------
    @property
    def car(self) -> str:
        """The car the build is fitted to (the host's `car`; the Corsa)."""
        return bridge.car_key(getattr(self.host, "car", None))

    @property
    def unlimited(self) -> bool:
        """Settings' Wing limits is Unlimited (the host's `unlimited`)."""
        return bool(getattr(self.host, "unlimited", False))

    @property
    def deck_z(self):
        """`deck(x)` under the top wing: the explicit one, else the car's."""
        return self._deck_z if self._deck_z is not None else bridge.car_deck(self.car)

    def past_limit(self, span: float, keys=None) -> tuple | None:
        """Real mode (task 41): `(slot, limit)` of the first slot in `keys`
        (default this one) where a wing of `span` would be past its car's
        physical limit (`bridge.span_limit`, at the slot's height as the car
        fits it); None when it fits everywhere -- or in Unlimited mode."""
        if self.unlimited:
            return None
        b = self.host.build
        for k in (keys or (self.key,)):
            lim = bridge.span_limit(self.role, self.car, float(b.slot(k).h))
            if float(span) > lim + bridge.LIMIT_TOL:
                return k, lim
        return None

    def log(self, text: str, level: str = "info") -> None:
        self.host.log(text, level)

    def say(self, text: str, kind: str = "info") -> None:
        f = getattr(self.host, "say", None) or getattr(self.host, "toast", None)
        if f is not None:
            f(text, kind)

    def goto(self, view: str) -> bool:
        f = getattr(self.host, "goto_view", None)
        return bool(f(view)) if f is not None else False

    def results_dir(self, what: str) -> str | None:
        """Where engine runs write (AeroBO's `results_dir`) and carsim keeps
        its records: runs/aerobo/<slot>/, or None when the host says so (the
        checks: nothing under runs/)."""
        root = getattr(self.host, "aerobo_dir", os.path.join("runs", "aerobo"))
        if root is None:
            return None
        return os.path.join(root, self.key, what)

    # -- the operating point -----------------------------------------------------------
    def _operating_point(self):
        """The slot's operating point (`bridge.operating_point`): the stated
        lap's mean speed with the car as it stands -- the number the Mission
        page quotes -- or the typed one; AeroBO's reference CZ or the typed
        one; the top slot's deck and ride band, the size rows, the span limit
        and the ceilings from the car (`bridge.operating_point`, task 41)."""
        h = self.host
        return bridge.operating_point(self.key, self.role, h.build, self.mission, V=self.V_typed,
                                      cz=self.cz_typed, lib=h.lib, deck_z=self._deck_z,
                                      car=self.car, unlimited=self.unlimited)

    def refresh_op(self) -> None:
        """Re-read the operating point (the mission stated again, V typed)."""
        self.op = self._operating_point()

    def set_V(self, v) -> None:
        self.V_typed = None if v is None else float(min(max(float(v), 5.0), 90.0))
        self.refresh_op()

    def set_cz(self, v) -> None:
        self.cz_typed = None if v is None else float(min(max(float(v), 0.1), 3.0))
        self.refresh_op()

    # -- the replay switch -----------------------------------------------------------
    def replay_active(self) -> Replay | None:
        return self.replay if self.replay is not None else _REPLAY

    def runner(self, kind: str, engine, **want):
        """The runner a job flies: the captured fixture under a replay, else
        the bridge factory `engine()` builds (every argument frozen now)."""
        rp = self.replay_active()
        if rp is not None:
            return rp.runner(kind, **want)
        return engine()

    @staticmethod
    def replay_of(runner):
        rr = getattr(dj, "ReplayRunner", None)
        return runner if (rr is not None and isinstance(runner, rr)) else None

    # -- jobs ------------------------------------------------------------------------
    def job_info(self, kind: str, owner, *, objective: str = "", phrase: str = "",
                 continued: bool = False, optimiser: str = "", n_init=None,
                 point: dict | None = None, problem: str = "", seed_label: str = "",
                 airfoil: str = "", plate: str = "", rseed: int = 0,
                 handoff=None) -> "dj.JobInfo":
        """The texts a job talks about itself with (design_jobs.JobInfo). A
        wing run on the lap objective is a "lap" job (`dj.wing_kind`)."""
        if kind == "wing":
            kind = dj.wing_kind(objective)
        surface = (owner.label if isinstance(owner, SurfaceModel) else
                   ("top wing" if self.key == "top" else "side wing"))
        stage = (owner.stage if isinstance(owner, SurfaceModel) else
                 ("r" if isinstance(owner, ResultsModel) else "w"))
        return dj.JobInfo(kind=kind, surface=surface, stage=stage, objective=objective,
                          phrase=phrase, continued=continued, track=self.mission.track,
                          recommended=self.policy.recommended(), optimiser=str(optimiser or ""),
                          n_init=(int(n_init) if n_init else None),
                          handoff=(int(handoff) if handoff else None), point=dict(point or {}),
                          problem=problem, seed_label=seed_label, airfoil=airfoil, plate=plate,
                          rseed=int(rseed))

    def ready_to_run(self, owner) -> bool:
        if self.runs.busy:
            self.say("a run is in progress — stop it first", "warning")
            return False
        return True

    def launch(self, job, quiet: bool = False) -> bool:
        """Start `job` on the garage's RunManager (one engine thread at a
        time: refused while one is live or still finishing)."""
        if not self.runs.start(job):
            if not quiet:
                self.say("the previous run is still finishing its last evaluation"
                         if getattr(self.runs, "draining", None) else
                         "a run is in progress — stop it first", "warning")
            return False
        return True

    def stop(self, owner) -> bool:
        if self.runs.job_for(owner) is None:
            return False
        return bool(self.runs.stop())

    # -- events from the models ---------------------------------------------------------
    def on_section(self, m: SurfaceModel) -> None:
        """A section was chosen: the wing's flags change, so a law fitted to
        the last run no longer describes what Run would fly (the record
        stays: it IS what was flown)."""
        self.wing._built_key = None

    def on_family(self) -> None:
        """Wing type moved the family: the section design points move with
        its box, and the plate stage may lock or unlock."""
        self.wing._built_key = None

    def on_commit(self) -> None:
        f = getattr(self.host, "on_wing_committed", None)
        if f is not None:
            f(self.key)

    # -- the tree ------------------------------------------------------------------------
    def state(self, stage: str) -> tuple:
        """`(state, reason)` for a stage, AeroBO's `stage_states` (PLAN2
        D10): once the mission is stated 2 Airfoil, 2.8 Endplate and 3 Wing
        are all READY -- the wing flies the family's own sections until one
        is chosen -- and 4 Results waits for a completed run."""
        stated = bool(getattr(self.host.mission, "stated", False))
        not_stated = ("state the mission first — the section is designed at the Reynolds "
                      "number and lift coefficient the mission implies")
        if stage == "m":
            return ("done" if stated else "ready"), ""
        if stage in ("af", "ep"):
            m = self.af if stage == "af" else self.ep
            if stage == "ep" and not self.wing.choices["plates"]:
                return "locked", ("pylons carry this wing, so its plates are only a tip device — "
                                  "a FENCE with the wing's own chord and section, nothing here "
                                  "to give an aerofoil to. Choose \"Carried by: the endplates\" "
                                  "on 3 Wing ▸ Wing type and the plate becomes a designed part "
                                  "with a section of its own")
            if not stated:
                return "locked", not_stated
            if self.runs.job_for(m) is not None:
                return "running", ""
            if m.finished():
                return "done", ""
            return "ready", m.reason()
        if stage == "w":
            if not stated:
                return "locked", ("state the mission first — the wing is sized and flown at "
                                  "the point it states")
            if self.runs.job_for(self.wing) is not None:
                return "running", ""
            return ("done" if self.wing.record else "ready"), ""
        if stage == "r":
            if not stated:
                return "locked", "state the mission first"
            o = self.wing.outcome or {}
            if o.get("state") == "error" and not self.wing.record:
                return "error", str(o.get("error") or "the run failed")
            if not self.wing.record:
                return "locked", "no completed run yet"
            if self.runs.job_for(self.results) is not None or (
                    self.runs.job_for(self.wing) is not None):
                return "running", ""
            return "done", ""
        return "locked", f"no stage {stage!r}"

    @property
    def mission(self):
        """The mission THIS slot flies: the top wing's is the garage's; a side
        wing's is its own circuit in the same conditions (`MissionSpec.
        for_side`; the owner, 2026-09-26: "why no longer circuits?")."""
        m = self.host.mission
        return m.for_side() if self.role != "top" and hasattr(m, "for_side") else m

    def job(self) -> str:
        """The slot's JOB (`aero.mission`): STOPPING or the circuit's name --
        a side wing's own job (`MissionSpec.side_track`)."""
        m = self.mission
        return ms.STOPPING if m.is_stop else str(m.track)

    def open_for(self, signature: tuple) -> bool:
        """The mission was stated for this slot with `signature` (the job,
        surface, the rest of the car). The session is KEPT when it was made
        for the same one (True) and cleared when not (False); either way its
        operating point is re-read. A first opening or a changed job opens
        3 Wing on the job's own objective (`WingModel.default_objective`)."""
        first = self.signature is None
        kept = first or self.signature == signature
        if not kept:
            was = self.signature
            self.invalidate("the job or the surface changed" if was[:2] != signature[:2]
                            else "the rest of the car changed")
        self.signature = signature
        self.refresh_op()
        if first or not kept:
            self.wing.default_objective()
        return kept

    def invalidate(self, why: str) -> None:
        """The mission was restated with a change (circuit, surface, car):
        every section, run and result was made for another operating point."""
        for m in (self.af, self.ep):
            m.reset()
        self.wing.reset()
        self.results.report = self.results.outcome = None
        self.refresh_op()
        if why:
            self.log(f"design stages cleared: {why}", "warn")


# --------------------------------------------------------------------------- #
#  the warm-up (PLAN2 H2)                                                      #
# --------------------------------------------------------------------------- #
_WARMED = False


def warm_up() -> bool:
    """Import AeroBO's BO stack (torch, botorch) NOW, once per process.

    The garage calls it while it loads, before its first frame. Every view
    that states a search reads `api.compatible_optimisers`, whose first call
    imports torch (~0.5-0.9 s measured here), and AeroBO's first measured
    plan imports scipy.stats (~0.4 s) -- inside whichever frame first drew
    a Solver or a Search & budget table (PLAN2 F15: one 420 ms hitch). Paid
    at load instead, no frame pays them; a machine without the BO stack
    pays only the plan. True when this call did the work."""
    global _WARMED
    if _WARMED:
        return False
    _WARMED = True
    bridge.warm()
    #  ...and the first plan AeroBO measures imports scipy.stats and runs the
    #  lattice for the first time (~0.4 s): one plan of the car family now,
    #  so the first Solver or Search & budget table drawn is a cached read
    try:
        bridge.section_plan("composite_goal", "balanced")
        bridge.wing_plan(bridge.BASES[("plates", "law")], {}, None, None, "balanced")
    except Exception:                                   # noqa: BLE001 -- a warm-up only
        pass
    return True


# --------------------------------------------------------------------------- #
#  re-deriving a committed wing's law (the garage's hook)                      #
# --------------------------------------------------------------------------- #
def law_stale(spec, slot) -> bool:
    """Has the slot moved (x or h) since `spec`'s AeroBO law was derived?"""
    if getattr(spec, "engine", "carsim") != "aerobo":
        return False
    at = (spec.aero or {}).get("derived_at") or {}
    return any(abs(float(at.get(k, getattr(slot, k))) - float(getattr(slot, k))) > 1e-9
               for k in ("x", "h"))


def rederive(spec, slot, deck_z=None, car=None) -> bool:
    """Re-derive an AeroBO wing's law at the slot as it stands now (PLAN2
    §5.7: the slot moved on the car page). Synchronous, a few hundred ms of
    AeroBO's evaluator; the top wing's family follows the slot's deck and
    its ride row is the slot's height. `car` (task 41): the car's deck and
    ride band at the slot's station (`bridge.car_deck` / `top_ride_band`);
    `deck_z`, when given, overrides the deck. True when `spec.aero` was
    replaced."""
    if getattr(spec, "engine", "carsim") != "aerobo" or not spec.design:
        return False
    d = spec.design
    V = float(((spec.aero or {}).get("derived_at") or {}).get("V") or spec.aero.get("V_ref"))
    geom = {"x": float(slot.x), "h": float(slot.h), "V": V, "role": spec.role}
    band = None
    if deck_z is not None:
        deck = float(deck_z(slot.x))
    elif car is not None:
        deck = float(bridge.car_deck(car)(slot.x))
        band = bridge.top_ride_band(car, slot.x)
    else:
        deck = None
    if deck is not None and band is None and car is not None:
        band = bridge.top_ride_band(car, slot.x, deck=deck)
    record = bridge.record_of_design(d, geom, deck=deck, band=band)
    law = bridge.derive_law(record, geom)
    spec.aero = dict(law)
    return True


# =========================================================================== #
#  SELF-CHECK                                                                  #
# =========================================================================== #
class _Slot:
    """A slot as the models read it (garage.Slot's fields)."""

    def __init__(self, wing="", x=0.97, h=0.90, inc_deg=0.0, mode="active"):
        self.wing, self.x, self.h, self.inc_deg, self.mode = wing, x, h, inc_deg, mode


class _Build:
    """The three slots a session reads, with CarBuild's mirror rule -- the
    self-check's car, so M1-M24 need neither pygame nor the garage."""

    def __init__(self):
        self.left, self.right = _Slot(), _Slot()
        self.top = _Slot("", -0.90, 1.55, 6.0, "active")
        self.mirror = True

    def slot(self, key):
        return getattr(self, key)

    def sync_mirror(self, edited):
        if self.mirror and edited != "top":
            src = self.slot(edited)
            dst = "right" if edited == "left" else "left"
            setattr(self, dst, _Slot(src.wing, src.x, src.h, src.inc_deg, "active"))

    def clamp(self, lib=None, car=None):
        return self

    def mission_aero(self, lib, exclude=None):
        return ms.MissionAero()

    def to_json(self):
        return {k: vars(self.slot(k)) for k in ("left", "right", "top")}


class _Host:
    """The garage as the models see it (duck-typed), headless."""

    def __init__(self, lib):
        self.lib, self.build = lib, _Build()
        self.mission = ms.MissionSpec()
        self.mission.stated = True
        self.notices = dj.Notices()
        self.runs = dj.RunManager(self.notices)
        self.aerobo_dir = None                  # a check writes nothing under runs/
        self.said: list = []

    def log(self, text, level="info"):
        self.notices.log(text, level)

    def toast(self, text, kind="info"):
        self.notices.toast(text, kind)

    def say(self, text, kind="info"):
        self.said.append(str(text))


def _deck(x: float) -> float:
    """The self-check's car deck: garage.deck_z at the default top station."""
    return 0.9985


def self_check(verbose: bool = True) -> bool:
    """M1-M37 (PLAN2 §9.3; M33 task 41's per-car wiring; M34 the jobs; M35-M37 "Carried by",
    M37 on AeroBO's live engine): the models on the captured fixtures, replayed
    through the same EngineJob path; the law, the design report and the
    plans on AeroBO's engine for real (fast, no XFOIL). Nothing is written
    under runs/."""
    import shutil
    import tempfile
    from .aero.library import Library

    ok, n_rows = True, [0, 0]

    def rep(tag, passed, msg=""):
        nonlocal ok
        passed = bool(passed)
        ok = ok and passed
        n_rows[0] += 1
        n_rows[1] += int(passed)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}: {msg}")

    def guard(tag, fn):
        """A row whose check raised is a FAILED row, not a dead check."""
        try:
            fn()
        except Exception as exc:                        # noqa: BLE001 -- the row reports it
            import traceback
            rep(tag, False, f"{type(exc).__name__}: {exc} @ "
                            + traceback.format_exc().strip().splitlines()[-3].strip()[:90])

    tmp = tempfile.mkdtemp(prefix="carsim_aerobo_models_")
    was_env = {k: os.environ.get(k) for k in ("CARSIM_NO_XFOIL", "CARSIM_NO_TORCH")}
    lib = Library(os.path.join(tmp, "library"), use_xfoil=False)
    rp = use_fixtures(FIXTURES)
    try:
        host = _Host(lib)
        pol = SearchPolicy()
        s = DesignSession(host, "left", pol, deck_z=_deck)
        t = DesignSession(host, "top", pol, deck_z=_deck)

        def run(job_started=True):
            host.runs.run_all(120.0)
            return job_started

        # M1 ------------------------------------------------------------------
        def m1():
            st = {k: s.state(k) for k in STAGES}
            rep("M1 a fresh session: 2, 2.8 and 3 ready after the mission, results locked",
                st["m"][0] == "done" and st["af"][0] == "ready" and st["ep"][0] == "ready"
                and st["w"][0] == "ready" and st["r"] == ("locked", "no completed run yet"),
                "; ".join(f"{k} {v[0]}" for k, v in st.items()))
        guard("M1", m1)

        # M2 ------------------------------------------------------------------
        def m2():
            rows = pol.plans(s)
            budgets = [r.get("budget") for r in rows]
            stamp = rows[0].get("stamp") or {}
            rep("M2 SearchPolicy plans: WingLab's 164 / 164 / 53 at balanced, with the study stamp",
                budgets == [164, 164, 53] and bool(stamp),
                f"{[(r['stage'], r.get('budget'), r.get('optimiser')) for r in rows]}; "
                f"stamp {str(stamp)[:60]}")
        guard("M2", m2)

        # M3 ------------------------------------------------------------------
        def m3():
            lap = ms.lap(host.mission.profile(make_track), ms.MissionAero(),
                         mu_scale=host.mission.mu_scale)
            ot, of = t.op, s.op
            top_ok = (abs(ot.deck - _deck(host.build.top.x)) < 1e-12
                      and abs(ot.ride_band[0] - (ot.deck + 0.14)) < 1e-12
                      and abs(ot.ride_band[1] - 1.85) < 1e-12)
            fl_ok = (abs(of.deck - bridge.OFFSET_M) < 1e-12
                     and abs(of.ride_band[0] - (bridge.OFFSET_M + 0.25)) < 1e-9
                     and abs(of.ride_band[1] - (bridge.OFFSET_M + 0.70)) < 1e-9)
            lap_s = ms.lap(host.mission.for_side().profile(make_track), ms.MissionAero(),
                           mu_scale=host.mission.mu_scale)
            rep("M3 operating point: the top's V is its lap's mean speed, a side wing's its own "
                "circuit's; the top rides over the deck, the side wing 100 m from its image",
                abs(ot.V - lap.v_mean) < 1e-9 and abs(of.V - lap_s.v_mean) < 1e-9
                and top_ok and fl_ok,
                f"top V {ot.V:.3f} m/s ({ot.V_source}); side V {of.V:.3f} m/s ({of.V_source}); "
                f"top deck {ot.deck:.3f} band "
                f"{ot.ride_band[0]:.3f}-{ot.ride_band[1]:.2f}; flank band {of.ride_band}")
        guard("M3", m3)

        # M4 ------------------------------------------------------------------
        def m4():
            box = s.wing.family_box()
            mid = {k: 0.5 * (v[0] + v[1]) for k, v in box.items()}
            cm, cp = s.af.conditions(), s.ep.conditions()
            c_main = mid["S_m2"] / mid["b_m"]
            lam, ratio = mid["taper"], mid["endplate_chord_ratio"]
            c_pl = ratio * 2.0 * mid["S_m2"] * lam / (mid["b_m"] * (1.0 + lam))
            re_m, re_p = s.op.V * c_main / s.op.nu, s.op.V * c_pl / s.op.nu
            rep("M4 section conditions: WingLab's arithmetic at the box's middle (main cl = CZ, "
                "plate cl 0)",
                abs(cm["re_own"] - re_m) < 1e-6 * re_m and abs(cp["re_own"] - re_p) < 1e-6 * re_p
                and cm["cl_design"] == s.op.cz_design and cp["cl_design"] == 0.0,
                f"main Re {cm['re_own']:.4g} (c {c_main:.3f} m), plate Re {cp['re_own']:.4g} "
                f"(c {c_pl:.3f} m); screened at {cm['re_source']}")
        guard("M4", m4)

        # M5 ------------------------------------------------------------------
        def m5():
            s.af.re_source = "library"
            started = s.af.start_screen()
            run()
            out = s.af.screen.get("outcome") or {}
            rep("M5 the main screen (replayed) applies: 14 ranked rows, a DONE outcome",
                started and len(s.af.ranked) == 14 and bool(s.af.ranked)
                and out.get("state") == "done" and s.af.ranked[0].get("coords"),
                f"{len(s.af.ranked)} ranked, best {s.af.ranked[0]['name'] if s.af.ranked else '-'}"
                f"; {outcome_tag(out)}")
        guard("M5", m5)

        # M6 ------------------------------------------------------------------
        def m6():
            s.ep.re_source = "library"
            started = s.ep.start_screen()
            run()
            sym = set(bridge.api.symmetric_section_names())
            names = [r["name"] for r in s.ep.ranked]
            rep("M6 the plate screen ranks symmetric sections only",
                started and names and all(nm in sym for nm in names),
                f"{len(names)} ranked, all in the {len(sym)} symmetric; best "
                f"{names[0] if names else '-'}")
        guard("M6", m6)

        # M7 ------------------------------------------------------------------
        def m7():
            kw = [p.key for p in s.af.screen_params.params]
            kp = [p.key for p in s.ep.screen_params.params]
            cw = [c[0] for c in s.af.columns()]
            cp = [c[0] for c in s.ep.columns()]
            rep("M7 the wing's screen has no cd@cl weight or column (owner, D9); the plate's has",
                "w.cdcr" not in kw and "cdcr" not in cw and "w.cdcr" in kp and "cdcr" in cp
                and "cdcr" not in s.af.weights,
                f"wing weights {[k for k in kw if k.startswith('w.')]}; plate columns {cp}")
            #  the owner, 2026-09-25: "endplate t/c shouldn't be given as an
            #  option" -- the plate's forms have no t/c gate; the wing's keep it
            ow = [p.key for p in s.af.opt_params.params]
            op_ = [p.key for p in s.ep.opt_params.params]
            tc_keys = {"gtc", "tcmin", "gotc", "otc"}
            gp = s.ep.gates("screen"), s.ep.gates("opt")
            rep("M7b the plate's screen and shape search have no t/c gate rows (owner, 2026-09-25), "
                "WingLab's own gates still sent; the wing's section keeps them",
                not (tc_keys & set(kp)) and not (tc_keys & set(op_))
                and {"gtc", "tcmin"} <= set(kw) and {"gotc", "otc"} <= set(ow)
                and "gcm" in kp and "gocm" in op_
                and gp[0]["tc_min"] == float(bridge.SCREEN_GATES["tc_min"])
                and gp[1]["tc_min"] == float(bridge.OPT_GATES["tc_min"]),
                f"plate gates sent {gp[0]} / {gp[1]}")
        guard("M7", m7)

        # M8 ------------------------------------------------------------------
        def m8():
            s.af.use_ranked(0)
            fv = s.af.flag_value()
            name = s.af.ranked[0]["name"]
            #  ...and its polar is AT HAND (the screening branch at the
            #  library point: no XFOIL, no job)
            pol = s.af.polar() or {}
            rep("M8 use_ranked(0): the library section is chosen, its flag is its name, its "
                "polar at hand",
                s.af.decision == "library" and s.af.chosen["name"] == name and s.af.finished()
                and (fv == name or (isinstance(fv, dict) and fv.get("name") == name))
                and len(pol.get("cl") or []) >= 5 and not s.af.start_polar(),
                f"chosen {name}; flag {fv!r}; polar {len(pol.get('cl') or [])} points at "
                f"Re {pol.get('re') or 0:.3g}")
        guard("M8", m8)

        # M9 ------------------------------------------------------------------
        def m9():
            s.ep.use_ranked(0)
            started = s.ep.start_optimise()
            run()
            s.ep.use_optimised()
            c = s.ep.chosen or {}
            wu, wl = np.asarray(c.get("w_upper") or [1.0]), np.asarray(c.get("w_lower") or [0.0])
            fv = s.ep.flag_value() or {}
            rep("M9 the plate's optimised section is symmetric (w_lower == -w_upper) and flies "
                "with its Re",
                started and s.ep.decision == "optimised" and wu.shape == wl.shape
                and np.allclose(wl, -wu, atol=1e-12) and isinstance(fv, dict) and fv.get("re"),
                f"{len(wu)} CST weights a side, max |w_l + w_u| {float(np.max(np.abs(wl + wu))):.2e}; "
                f"flag Re {fv.get('re')}")
        guard("M9", m9)

        # M10 -----------------------------------------------------------------
        def m10():
            s.ep.decline()
            fl = s.wing.physics_flags()
            rep("M10 decline() on the plate: the family's own plate, no section_name_plate flag",
                s.ep.decision == "default" and s.ep.flag_value() is None
                and bridge.api.SECTION_PLATE_KEY not in fl and s.ep.finished(),
                f"plate flag absent; main flag {fl.get(bridge.api.SECTION_KEY)!r}")
        guard("M10", m10)

        # M11 -----------------------------------------------------------------
        def m11():
            rp.pause_at = 5
            started = s.af.start_optimise()
            job = host.runs.live
            job.settle(10.0) if job is not None else None
            k_at = job.k if job is not None else -1
            host.runs.stop()
            run()
            rp.pause_at = None
            out = s.af.opt.get("outcome") or {}
            has = s.af.has_optimised()
            rep("M11 section Stop at 5: STOPPED · 5/N, the best kept and usable",
                started and k_at == 5 and out.get("state") == "stopped" and out.get("k") == 5
                and has, f"{outcome_tag(out)}; optimised section kept: {has}")
        guard("M11", m11)

        # M12 -----------------------------------------------------------------
        def m12():
            started = s.af.start_optimise()
            run()
            g = s.af.graph()
            res = (s.af.opt.get("report") or {}).get("result") or {}
            rep("M12 section records -> the graph's inputs: N = the budget, n_init 4, one dot "
                "per evaluation",
                started and g["N"] == int(res.get("n_evals") or -1) and g["n_init"] == 4
                and len(g["records"]) == g["N"] and not g["live"],
                f"N {g['N']}, n_init {g['n_init']}, {len(g['records'])} records, tag {g['tag']}")
        guard("M12", m12)

        # M13 -----------------------------------------------------------------
        def m13():
            spent = s.af.spent()
            cont = bridge.continue_section(s.af.opt["report"], s.af.opt["launch"], 6)
            note = cont.get("note") or {}
            srch = cont.get("search") or {}
            rep("M13 section Keep going: WingLab's resume -- budget spent + extra, a payload to "
                "inherit, nothing re-flown",
                int(srch.get("budget", -1)) == spent + 6 and bool(srch.get("resume"))
                and int(note.get("resumed") or 0) == spent,
                f"{spent} -> {srch.get('budget')} ({note.get('resumed')} inherited); "
                f"can_continue {s.af.can_continue()}")
        guard("M13", m13)

        # M14 -----------------------------------------------------------------
        def m14():
            s.af.use_ranked(0)
            cfg = s.wing.cfg()
            f = cfg.flags or {}
            want = {bridge.api.CHORD_TREND_KEY: "root_largest", "mount": "tips", "section": "shaped",
                    bridge.api.BO_REFUSAL_FLAG: "worst", bridge.api.BO_FEASIBILITY_FLAG: "guide"}
            rep("M14 the wing's RunConfig is V3's for the slot family, with the chosen section",
                cfg.problem_name == s.wing.family_name and cfg.optimiser == "bo_slsqp"
                and int(cfg.budget) == 53 and all(f.get(k) == v for k, v in want.items())
                and abs(float(f.get("V", 0.0)) - s.op.V) < 1e-9 and f.get(bridge.api.SECTION_KEY),
                f"{cfg.problem_name}, {cfg.optimiser}, budget {cfg.budget}, section "
                f"{f.get(bridge.api.SECTION_KEY)!r}")
        guard("M14", m14)

        # M15 -----------------------------------------------------------------
        def m15():
            top_pl = list(t.wing.objectives())
            t.wing.choices["plates"] = False
            top_fe = list(t.wing.objectives())
            t.wing.choices["plates"] = True
            fl = s.wing.objectives()
            rep("M15 WingLab's objective menus: lap time only on the top wing with fences; the "
                "flank's downforce is side force",
                top_pl == ["efficiency", "downforce", "drag", "downforce_plus_drag"]
                and top_fe == ["efficiency", "downforce", "drag", "laptime", "downforce_plus_drag"]
                and "laptime" not in fl and "side force" in fl["downforce"]
                and "cz" not in top_fe and "cd" not in top_fe,
                f"top plates {top_pl}; top fences {top_fe}; flank '{fl['downforce']}'")
        guard("M15", m15)

        # M16 -----------------------------------------------------------------
        def m16():
            w = s.wing
            w.choices.update(objective="downforce", drag_budget_n=60.0)
            f1 = w.physics_flags()
            w.choices.update(objective="drag", drag_budget_n=None, downforce_min_n=250.0)
            f2 = w.physics_flags()
            w.choices.update(objective="efficiency", drag_budget_n=None, downforce_min_n=None)
            f3 = w.physics_flags()
            rep("M16 drag ceiling and force floor reach WingLab as drag_budget_n / downforce_min_n",
                f1.get("drag_budget_n") == 60.0 and f1.get("car_objective") == "downforce"
                and f2.get("downforce_min_n") == 250.0 and "drag_budget_n" not in f2
                and "car_objective" not in f3 and "drag_budget_n" not in f3,
                f"downforce {f1.get('drag_budget_n')} N ceiling; drag {f2.get('downforce_min_n')} "
                f"N floor; efficiency sends neither")
        guard("M16", m16)

        # M17 -----------------------------------------------------------------
        def m17():
            w = t.wing
            w.choices.update(plates=False, objective="laptime")
            fam = w.family
            name = w.family_name
            w.choices.update(plates=True, objective="efficiency")
            fam2 = w.family
            rep("M17 lap time: the top wing on fences only, timed round carsim's stated circuit",
                fam.lap is not None and str(fam.lap[0]) == host.mission.track
                and not fam.plates and fam2.lap is None and "laptime" not in s.wing.objectives(),
                f"{name}: lap {fam.lap}")
        guard("M17", m17)

        # M18 -----------------------------------------------------------------
        def m18():
            w = s.wing
            s.ep.use_ranked(0)
            pin = w.pinned()
            tc_ok = abs(float(pin.get("endplate_tc", -1)) - float(s.ep.chosen["tc"])) < 1e-9 or \
                "endplate_tc" in pin
            w.box["taper"] = [0.5, 0.8]
            bo = w.bounds_overrides() or {}
            w.released.add("taper")
            bo_rel = w.bounds_overrides() or {}
            w.released.discard("taper")
            w.fixed["twist_root_deg"] = 1.0
            pin2 = w.pinned()
            src = (w.band_source("endplate_tc"), w.band_source("taper"), w.band_source("b_m"))
            w.fixed.clear()
            w.box.clear()
            rep("M18 constrain / release / fix -> bounds_overrides / pinned; the plate's t/c is "
                "fixed from 2.8",
                tc_ok and bo.get("taper") == [0.5, 0.8] and "taper" not in bo_rel
                and pin2.get("twist_root_deg") == 1.0 and "b_m" in bo
                and src == ("fixed from 2.8", "user", "carsim packaging"),
                f"pin {pin}; sources {src}; flank size rows {bo.get('b_m')}, {bo.get('S_m2')}")
            #  ...and it is not the player's (owner, 2026-09-25): no box
            #  controls, and none of the switches' paths moves it
            w._box_form = None
            keys_ = [p.key for p in w.box_params().params]
            w._set_fix("endplate_tc")(False)
            w._set_constrain("endplate_tc")(False)
            w._box_set("endplate_tc", 0)(0.05)
            w._set_fix("endplate_tc")(True)
            w.fixed["endplate_tc"] = 0.2            # a stray value never reaches AeroBO
            pin3 = w.pinned()
            w.fixed.clear()
            s.ep.decline()
            pin4 = w.auto_pins()
            dec_bo = w.bounds_overrides() or {}
            s.ep.use_ranked(0)
            rep("M18b endplate_tc has no box controls and cannot be released: pinned to 2.8's "
                "t/c; with the family's own plate WingLab searches it (not the player's row)",
                not any(k.startswith("bx.endplate_tc.") for k in keys_)
                and any(k.startswith("bx.taper.") for k in keys_)
                and "endplate_tc" not in w.released and "endplate_tc" not in w.box
                and abs(float(pin3.get("endplate_tc", -1)) - float(pin.get("endplate_tc", -2))) < 1e-12
                and pin4 == {} and "endplate_tc" not in dec_bo
                and "endplate_tc" in w.family_box(),
                f"pin after the switches {pin3.get('endplate_tc')}; declined -> {pin4}")
            #  the form's three states, as AeroBO has them (the owner, 2026-09-26:
            #  "constrain and fix buttons don't work properly"): every row opens
            #  constrained; released, it searches the family's own band -- the
            #  packaging row dropped, the car's ceiling kept -- and the numbers
            #  typed come back with the switch; fixed, the value typed is taken as
            #  typed and its band widened to hold it, so the run is accepted
            w._box_form = None
            f_ = w.box_params()
            con0 = bool(f_.param("bx.taper.con").get()) and bool(f_.param("bx.b_m.con").get())
            f_.param("bx.taper.min").set(0.5)
            f_.param("bx.taper.con").set(False)
            rel_bo = w.bounds_overrides() or {}
            rel_en = f_.param("bx.taper.min").enabled
            f_.param("bx.b_m.con").set(False)
            b_rel = (w.bounds_overrides() or {}).get("b_m")
            cap = float((s.op.size_caps or {}).get("b_m", 0.0))
            f_.param("bx.taper.con").set(True)
            f_.param("bx.b_m.con").set(True)
            back = (w.bounds_overrides() or {}).get("taper")
            f_.param("bx.twist_tip_deg.fix").set(True)
            f_.param("bx.twist_tip_deg.val").set(9.0)           # past the family's band
            bo9 = (w.bounds_overrides() or {}).get("twist_tip_deg")
            fixed_row = (bool(f_.param("bx.twist_tip_deg.con").get()),
                         f_.param("bx.twist_tip_deg.con").enabled, f_.param("bx.twist_tip_deg.min").enabled)
            try:
                w.cfg()
                cfg_ok = True
            except Exception as exc:                        # noqa: BLE001 -- the row reports it
                cfg_ok = f"{type(exc).__name__}: {exc}"
            pin9 = w.pinned().get("twist_tip_deg")
            w.fixed.clear()
            w.box.clear()
            w.released.clear()
            rep("M18c the Design box's switches: every row opens constrained; released it "
                "searches the family's own band (under the car's ceiling) and the typed band "
                "comes back; a fixed row takes the value typed, its band widened to hold it",
                con0 and "taper" not in rel_bo and not rel_en and back is not None
                and abs(back[0] - 0.5) < 1e-12 and b_rel is not None and b_rel[1] <= cap + 1e-9
                and pin9 == 9.0 and bo9 is not None and bo9[0] <= 9.0 <= bo9[1]
                and fixed_row == (False, False, False) and cfg_ok is True,
                f"released taper -> {rel_bo.get('taper')}, back {back}; released b_m {b_rel} "
                f"(cap {cap:.3g}); fixed twist_tip 9 -> band {bo9}, cfg {cfg_ok}")
        guard("M18", m18)

        # M21 (before M19: the continuation is of the STOPPED top run) -------
        def m21():
            rp.pause_at = 8
            started = t.wing.start_run()
            job = host.runs.live
            job.settle(10.0) if job is not None else None
            host.runs.stop()
            run()
            rp.pause_at = None
            r0 = t.wing.record or {}
            o0 = t.wing.outcome or {}
            t.wing.more = 6
            started2 = t.wing.start_run(extend=True)
            run()
            r1 = t.wing.record or {}
            rep("M21 wing Stop at 8 then Keep going +6: nothing re-flown, the counter goes on",
                started and started2 and o0.get("state") == "stopped" and r0.get("partial")
                and int(r1.get("resumed") or 0) == 8 and int(r1.get("n_evals") or 0) == 14
                and len(t.wing.records) == 14,
                f"{outcome_tag(o0)} -> {outcome_tag(t.wing.outcome)}; resumed "
                f"{r1.get('resumed')}, n_evals {r1.get('n_evals')}")
        guard("M21", m21)

        # M22 -----------------------------------------------------------------
        def m22():
            v = t.wing.verdict()
            rep("M22 the verdict is WingLab's converged_report on the run's own history",
                isinstance(v, dict) and bool(v.get("text")) and "verdict" in v,
                f"{v.get('verdict')}: {str(v.get('text'))[:70]}")
        guard("M22", m22)

        # M19 -----------------------------------------------------------------
        def m19():
            rp.pause_at = 3
            started = t.wing.start_run()
            job = host.runs.live
            job.settle(10.0) if job is not None else None
            g_live = t.wing.graph()
            rp.pause_at = None
            if job is not None and job.replay is not None:
                job.replay.release()
            run()                                   # the run, then the law and report jobs
            run()
            run()
            w = t.wing
            #  the live graph marked the Sobol block and the handoff the
            #  finished run then reported (AeroBO's arithmetic both times)
            ho = (w.record or {}).get("handoff") or {}
            g_end = w.graph()
            split_ok = (g_live["live"] and g_live["n_init"] is not None
                        and (g_live["n_init"], g_live["handoff"])
                        == (g_end["n_init"], g_end["handoff"]) == ((ho.get("bo_split") or [None])[0],
                                                                  ho.get("n_a")))
            w.fit("check-top")
            law = w.law or {}
            inc = float(w.slot_updates.get("inc_deg"))
            game = ResultsModel.game_force(law, inc)
            ab = float((law.get("aerobo") or {}).get("F_N") or 0.0)
            rep("M19 fit: WingLab's law and a WingSpec from the record; the game's force at V is "
                "WingLab's (the live graph marked the run's Sobol block and handoff)",
                started and w.spec is not None and w.spec.engine == "aerobo" and ab > 0.0
                and abs(game - ab) <= 1e-9 * abs(ab) and w.dirty and split_ok,
                f"{w.spec.name if w.spec else '-'}: F game {game:.6f} N vs WingLab {ab:.6f} N at "
                f"inc {inc:+.3f} deg; law CL0 {law.get('CL0', 0):.4f} CLa {law.get('CLa', 0):.4f}; "
                f"split live {g_live['n_init']} | {g_live['handoff']}, run {ho.get('bo_split')} "
                f"n_a {ho.get('n_a')}")
        guard("M19", m19)

        # M20 -----------------------------------------------------------------
        def m20():
            n_af = len(lib.airfoils)
            t.af.use_ranked(0) if t.af.ranked else None
            name = t.wing.commit()
            top = host.build.top
            spec = lib.wings.get(name)
            imported = [a for a in lib.airfoils.values() if a.origin.startswith("WingLab")]
            upd = t.wing.slot_updates
            rep("M20 commit: the wing (and any section it flies, as coords with its WingLab origin) "
                "saved; the slot takes WingLab's incidence and ride height",
                spec is not None and top.wing == name and not t.wing.dirty
                and abs(top.inc_deg - float(upd["inc_deg"])) < 1e-12
                and abs(top.h - float(upd["h"])) < 1e-12
                and all(a.source == "coords" for a in imported) and len(lib.airfoils) >= n_af
                and spec.airfoil in lib.airfoils
                and (not spec.plate_airfoil or spec.plate_airfoil in lib.airfoils),
                f"'{name}' in the top slot at inc {top.inc_deg:+.3f} deg, h {top.h:.3f} m; "
                f"{len(imported)} imported section(s); flies {spec.airfoil} / "
                f"{spec.plate_airfoil or 'fences'} (both in the library)")
        guard("M20", m20)

        # M24 -----------------------------------------------------------------
        def m24():
            spec = lib.wings[host.build.top.wing]
            top = host.build.top
            h0 = top.h
            top.h = min(h0 + 0.10, 1.80)
            stale = law_stale(spec, top)
            done = rederive(spec, top, _deck)
            law = spec.aero
            game = ResultsModel.game_force(law, top.inc_deg)
            ab = float((law.get("aerobo") or {}).get("F_N") or 0.0)
            rep("M24 the slot moved: the law is stale, re-derived at the new h, the forces equal "
                "again",
                stale and done and not law_stale(spec, top) and abs(game - ab) <= 1e-9 * abs(ab)
                and abs(float(law["derived_at"]["h"]) - top.h) < 1e-12,
                f"h {h0:.3f} -> {top.h:.3f} m: F {ab:.3f} N (game {game:.3f} N)")
        guard("M24", m24)

        # M25 / M26: the garage's own readers (lazily: garage imports pygame) --
        def m25_26():
            from . import garage as grg
            spec = lib.wings[host.build.top.wing]
            calls = []
            real = lib.analyse_wing
            lib.analyse_wing = lambda *a, **k: (calls.append(a), real(*a, **k))[1]
            try:
                slot = grg.Slot(spec.name, host.build.top.x, host.build.top.h,
                                host.build.top.inc_deg, "active")
                ta = grg._top_aero(spec, slot, lib)
            finally:
                del lib.analyse_wing
            cz_law = spec.aero["CL0"] + spec.aero["CLa"] * math.radians(slot.inc_deg)
            rep("M25 _top_aero reads an WingLab wing's stored law: no carsim lattice call",
                ta is not None and not calls and abs(ta.CZ - cz_law) < 1e-12,
                f"CZ {ta.CZ:.6f} (law {cz_law:.6f}); analyse_wing called {len(calls)} times")
            b = grg.CarBuild()
            b.top = slot
            kw = b.cfg_kwargs(lib)
            rep("M26 CarBuild.cfg_kwargs with an WingLab top wing: TopAero's CZ and CD are the "
                "law's at the slot's incidence",
                kw.get("top") is not None and abs(kw["top"].CZ - cz_law) < 1e-12
                and abs(kw["top"].S - float(spec.aero["S"])) < 1e-12,
                f"CZ {kw['top'].CZ:.6f}, CD {kw['top'].CD:.6f}, S {kw['top'].S:.4f} m2")
        try:
            m25_26()
        except Exception as exc:                        # noqa: BLE001
            rep("M25 _top_aero reads an WingLab wing's stored law", False, f"{type(exc).__name__}: {exc}")
            rep("M26 CarBuild.cfg_kwargs with an WingLab top wing", False, f"{type(exc).__name__}: {exc}")

        # M23 -----------------------------------------------------------------
        def m23():
            sig = (host.mission.track, host.mission.surface, "rest")
            s.signature = sig
            had = (bool(s.af.ranked), s.af.decision)
            kept = s.open_for(sig)
            same = (bool(s.af.ranked), s.af.decision) == had
            host.mission.track = "open"
            cleared = not s.open_for((host.mission.track, host.mission.surface, "rest"))
            gone = not s.af.ranked and s.af.decision is None and s.wing.record is None
            host.mission.track = "arena"
            s.open_for((host.mission.track, host.mission.surface, "rest"))
            rep("M23 restating the same mission keeps the session; a changed circuit clears it",
                kept and same and had[0] and cleared and gone,
                f"kept {kept} ({had}); open -> cleared {cleared}")
        guard("M23", m23)

        # M27 -----------------------------------------------------------------
        def m27():
            os.environ["CARSIM_NO_XFOIL"] = "1"
            use_engine()
            try:
                src = s.af.effective_re_source()
                refused = not s.af.start_optimise()
                why = s.af.msg
            finally:
                use_fixtures(FIXTURES)
                rp_ = replay()
                rp_.pause_at = None
                if was_env["CARSIM_NO_XFOIL"] is None:
                    os.environ.pop("CARSIM_NO_XFOIL", None)
                else:
                    os.environ["CARSIM_NO_XFOIL"] = was_env["CARSIM_NO_XFOIL"]
            rep("M27 no XFOIL: the screen falls back to the library point, shape optimisation is "
                "refused with the reason",
                src == "library" and refused and "XFOIL" in why, why[:80])
        guard("M27", m27)

        # M28 -----------------------------------------------------------------
        def m28():
            os.environ["CARSIM_NO_TORCH"] = "1"
            try:
                note = s.wing.bo_note()
                w_opt = s.wing.search()["optimiser"]
                s_opt = pol.section_search("composite_goal", "main")["optimiser"]
            finally:
                if was_env["CARSIM_NO_TORCH"] is None:
                    os.environ.pop("CARSIM_NO_TORCH", None)
                else:
                    os.environ["CARSIM_NO_TORCH"] = was_env["CARSIM_NO_TORCH"]
            rep("M28 no torch (simulated): WingLab's note, and its non-torch optimisers fly instead",
                bool(note) and not str(w_opt).startswith("bo") and not str(s_opt).startswith("bo"),
                f"wing {w_opt}, section {s_opt}; '{note[:60]}'")
        guard("M28", m28)

        # M29 -----------------------------------------------------------------
        def m29():
            s.wing.choices["plates"] = False
            locked = s.state("ep")[0] == "locked"
            s.wing.choices["plates"] = True
            s.af.re_source = "library"
            s.af.start_screen()
            run()
            names = (bool(s.af.ranked), callable(s.af.finished), callable(s.af.use_ranked),
                     callable(s.ep.decline), hasattr(t.wing.spec, "name"),
                     isinstance(t.wing.dirty, bool), t.wing.key == "top")
            rep("M29 the tutorial's names: ranked, finished, use_ranked, decline, a locked plate "
                "stage on fences, spec.name, dirty, key",
                all(names) and locked, str(names))
        guard("M29", m29)

        # M30 -----------------------------------------------------------------
        def m30():
            sm = t.results.summary()
            bd = (t.wing.record or {}).get("breakdown") or {}
            keys = ("best_score", "score_units", "CZ", "CD", "force_N", "drag_N", "n_evals",
                    "budget", "partial", "stop_reason", "sections", "law", "game_force_N",
                    "aerobo_force_N", "tag")
            rep("M30 the results summary carries the record's numbers and the car mapping",
                all(k in sm for k in keys) and sm["CZ"] == bd.get("CZ")
                and sm["force_word"] == "downforce" and sm["law"] is not None,
                f"best {sm.get('best_score')} {sm.get('score_units')}; CZ {sm.get('CZ')}, "
                f"F {sm.get('force_N')} N; game {sm.get('game_force_N')} N")
        guard("M30", m30)

        # M31 -----------------------------------------------------------------
        def m31():
            if t.results.report is None:
                t.results.start_report()
                run()
            r = t.results.report or {}
            rep("M31 the design report job applies WingLab's geometry and breakdown",
                bool(r.get("geometry")) and bool(r.get("breakdown"))
                and (t.results.outcome or {}).get("state") == "done",
                f"keys {sorted(r)[:6]}")
        guard("M31", m31)

        # M32 -----------------------------------------------------------------
        def m32():
            rp.pause_at = 3
            first = s.wing.start_run()
            job = host.runs.live
            job.settle(10.0) if job is not None else None
            second = s.af.start_screen()
            busy = host.runs.busy
            host.runs.stop()
            run()
            rp.pause_at = None
            rep("M32 one engine thread at a time: a second start is refused while one is live",
                first and not second and busy and host.runs.live is None,
                f"first {first}, second {second}, busy {busy}")
        guard("M32", m32)

        # M33 (task 41) ------------------------------------------------------------
        def m33():
            #  the session reads the host's CAR and Wing limits: the bus's flank
            #  limit at h 0.90 is 2 (0.90 - 0.28) = 1.24 m (its own underbody),
            #  its top wing flies over its own deck in its own band, its top span
            #  row is 1.2 x its 2.55 m width; Unlimited opens the ceilings 3x but
            #  not the default rows; the Corsa's default is task 41's 1.50 m
            from . import bodies as _b
            hb = _Host(lib)
            hb.car, hb.unlimited = "bus", False
            fb, tb = DesignSession(hb, "left", pol), DesignSession(hb, "top", pol)
            hc = _Host(lib)
            fc = DesignSession(hc, "left", pol)
            hu = _Host(lib)
            hu.car, hu.unlimited = "bus", True
            fu = DesignSession(hu, "left", pol)
            body = _b.body("bus")
            band = _b.top_h_band("bus", hb.build.top.x)
            rep("M33 task 41: the session is the host car's -- the bus's flank limit, top deck, "
                "ride band and span row; Unlimited widens the ceilings only; the Corsa's 1.50 m",
                abs(fb.op.limit - 2.0 * (0.90 - body.ground)) < 1e-12
                and abs(fb.op.size_rows["b_m"][1] - fb.op.limit) < 1e-6
                and abs(tb.op.deck - body.deck_z(hb.build.top.x)) < 1e-12
                and tuple(tb.op.ride_band) == tuple(band)
                and abs(tb.op.size_rows["b_m"][1] - 1.2 * body.width) < 1e-6
                and fu.op.size_rows == fb.op.size_rows
                and abs(fu.op.size_caps["b_m"] - 3.0 * fb.op.limit) < 1e-9
                and abs(fc.op.limit - 1.50) < 1e-12 and fc.op.size_rows["b_m"][1] == 1.5
                and fc.car == "corsa" and fb.car == "bus",
                f"bus flank <= {fb.op.limit:.2f} m (rows {fb.op.size_rows['b_m']}), top deck "
                f"{tb.op.deck:.2f} band {tb.op.ride_band[0]:.2f}-{tb.op.ride_band[1]:.2f} b <= "
                f"{tb.op.size_rows['b_m'][1]:.3f}; Unlimited cap {fu.op.size_caps['b_m']:.2f} m; "
                f"Corsa {fc.op.limit:.2f} m")
        guard("M33", m33)

        # M34 (the jobs) --------------------------------------------------------------
        def m34():
            """The owner, 2026-09-25: "One more circuit should be added
            'Stopping'. Left flank and right flank, should be side." -- and
            2026-09-26, "why no longer circuits?": a side wing flies a
            job of its own (`MissionSpec.side_track`) -- and "Side wing should
            also have 'stopping' mission": the stop, on the air brake."""
            m = host.mission
            was = (m.track, m.v_stop_kmh, t.V_typed, s.V_typed, m.side_track)
            try:
                t.V_typed = s.V_typed = None
                m.track = ms.STOPPING
                t.refresh_op()
                s.refresh_op()
                stop_v = (abs(t.op.V - ms.V_STOP_KMH / 3.6) < 1e-12
                          and "start speed" in t.op.V_source and t.job() == ms.STOPPING)
                t2 = DesignSession(host, "top", pol, deck_z=_deck)
                t2.open_for((m.job_key, m.surface, "rest"))
                stop_obj = (t2.wing.choices["objective"] == "downforce_plus_drag"
                            and "laptime" not in t2.wing.objectives())
                m.side_track = "arena"
                s.refresh_op()
                v_arena = s.op.V
                side_v = s.job() == "arena" and "arena lap's mean speed" in s.op.V_source
                m.side_track = "kestrel"
                s.refresh_op()
                side_v = side_v and s.job() == "kestrel" and abs(s.op.V - v_arena) > 1e-6
                s2 = DesignSession(host, "left", pol, deck_z=_deck)
                s2.open_for((m.side_track, m.surface, "rest"))
                side_obj = (s2.wing.choices["objective"] == "downforce"
                            and "side force" in s2.wing.objectives()["downforce"])
                s2.wing.choices["objective"] = "efficiency"         # the player's pick...
                s2.open_for((m.side_track, m.surface, "rest"))     # ...survives a restatement
                keeps_pick = s2.wing.choices["objective"] == "efficiency"
                m.side_track, m.v_stop_kmh = ms.STOPPING, 150.0
                s.refresh_op()
                s4 = DesignSession(host, "left", pol, deck_z=_deck)
                s4.open_for((m.for_side().job_key, m.surface, "rest"))
                side_stop = (s.job() == ms.STOPPING and abs(s.op.V - 150.0 / 3.6) < 1e-12
                             and "start speed" in s.op.V_source and m.for_side().stop_flanks == 2
                             and s4.wing.choices["objective"] == "downforce_plus_drag"
                             and "side force + drag" in s4.wing.objectives()["downforce_plus_drag"])
                m.side_track, m.v_stop_kmh = "kestrel", was[1]
                m.track = "arena"
                t3 = DesignSession(host, "top", pol, deck_z=_deck)
                t3.open_for((m.job_key, m.surface, "rest"))
                lap_obj = t3.wing.choices["objective"] == "efficiency" and t3.job() == "arena"
                offered = ("laptime" in bridge.car_objectives("top", False, "arena")
                           and "laptime" not in bridge.car_objectives("top", False, ms.STOPPING)
                           and "laptime" not in bridge.car_objectives("flank", False, "arena"))
            finally:
                m.track, m.v_stop_kmh, m.side_track = was[0], was[1], was[4]
                t.V_typed, s.V_typed = was[2], was[3]
                t.refresh_op()
                s.refresh_op()
            rep("M34 the jobs: Stopping flies at its start speed and opens on downforce + drag "
                "(no lap time); a side wing flies its own circuit at that lap's mean speed and "
                "opens on side force, or the stop (the air brake) at its start speed on side "
                "force + drag; a restatement keeps the player's pick; a circuit opens on "
                "efficiency",
                stop_v and stop_obj and side_v and side_obj and side_stop and keeps_pick
                and lap_obj and offered,
                f"stop V {stop_v} obj {stop_obj}; side V {side_v} obj {side_obj}; side stop "
                f"{side_stop}; keeps "
                f"{keeps_pick}; circuit {lap_obj}; lap time offered right {offered}")
        guard("M34", m34)

        # M35-M37: "Carried by" (the owner, 2026-09-25) ------------------------------
        #  a fresh TOP session (lap time comes and goes with the mount there),
        #  driven through the card's own Params the way the view and a pad do
        u = DesignSession(host, "top", pol, deck_z=_deck)
        CARD0 = (("plates", "endplates"), ("law", "free"), ("blend", "stated"), ("cant", "stated"),
                 ("blend_frac", 0.0), ("cant_deg", 90.0), ("tip", "vertical"), ("tip_cant_deg", 90.0),
                 ("tip_chord", "follows"))

        def card(w, ch):
            """Back to the card's defaults, then `ch` (a bridge.MOUNT_CONFIGS
            answer set), each through its Param's setter."""
            tp = w.type_params
            for k, v in CARD0:
                tp.param(k).set(v)
            words = {"plates": lambda v: "endplates" if v else "pylons",
                     "chord_law": lambda v: "free" if v else "straight"}
            for k, v in ch.items():
                tp.param("law" if k == "chord_law" else k).set(words.get(k, lambda x: x)(v))

        def shown(w) -> dict:
            #  which of the card's conditional rows take the keyboard / pad
            return {k: w.type_params.is_enabled(w.type_params.param(k))
                    for k in ("blend", "blend_frac", "cant", "cant_deg", "tip", "tip_cant_deg",
                              "tip_chord", "to28")}

        def m35():
            w = u.wing
            bad = []
            for label, ch, want in bridge.MOUNT_CONFIGS:
                card(w, ch)
                pyl = not ch.get("plates", True)
                fl = w.physics_flags()
                labels = set(w.family_box())
                rows = {r for r in ("endplate_cant_deg", "endplate_blend_frac") if r in labels}
                bx = {p.key for p in w.box_params().params}
                on = shown(w)
                tip = ch.get("tip")
                want_on = {"blend": not pyl, "cant": not pyl, "to28": not pyl,
                           "blend_frac": not pyl and ch.get("blend") != "free",
                           "cant_deg": not pyl and ch.get("cant") != "free",
                           "tip": pyl, "tip_cant_deg": pyl and tip == "canted",
                           "tip_chord": pyl and tip != "none"}
                why = []
                if w.family.base != want["base"] or w.family.pylons != pyl:
                    why.append(f"family {w.family.base}")
                if any(fl.get(k) != v for k, v in want["sent"].items()) or any(
                        k in fl for k in want["absent"]):
                    why.append(f"flags {sorted(fl)}")
                if rows != set(want["rows"]) or any(f"bx.{r}.min" not in bx for r in rows):
                    why.append(f"rows {sorted(rows)}")
                pins = {k: v for k, v in w.pinned().items() if k == "endplate_h_m"}
                if pins != want["pins"] or ((w.band_source("endplate_h_m") == TIP_PIN_SOURCE)
                                            != bool(want["pins"])):
                    why.append(f"pins {w.pinned()} ({w.band_source('endplate_h_m')})")
                if on != want_on:
                    why.append(f"rows the keys reach {on}")
                if ("laptime" in w.objectives()) != pyl or (u.state("ep")[0] == "locked") != pyl:
                    why.append(f"lap {list(w.objectives())}, 2.8 {u.state('ep')[0]}")
                if why:
                    bad.append(f"{label}: {'; '.join(why)}")
            rep("M35 Carried by, through the card's own rows: each configuration's family, flags, "
                "box rows, pins, lap time and 2.8 lock -- and the keys reach only the rows it shows",
                not bad, "; ".join(bad) if bad else f"{len(bridge.MOUNT_CONFIGS)} configurations")

            #  the card's own rules
            tp = w.type_params
            card(w, {"plates": False})
            tp.param("tip").set("canted")
            lean0 = w.choices["tip_cant_deg"]
            tp.param("tip_cant_deg").set(60.0)
            tp.param("tip").set("vertical")
            tp.param("tip").set("canted")
            lean1 = w.choices["tip_cant_deg"]
            tp.param("tip").set("vertical")
            w.fixed["endplate_h_m"] = 0.10                  # the player's own fix ...
            tp.param("tip").set("none")
            pin_none = w.pinned().get("endplate_h_m")       # ... the device's pin wins
            w._box_form = None
            w.box_params().param("bx.endplate_h_m.fix").set(False)   # release = a plate again
            released = (w.choices["tip"], w.pinned().get("endplate_h_m"))
            w.fixed.clear()
            card(w, {"plates": True, "blend_frac": 0.3, "cant_deg": 70.0})
            tp.param("plates").set("fences")                # the old words still read
            fences = (w.choices["plates"], w.family.pylons)
            tp.param("plates").set("designed")
            kept = (w.choices["blend_frac"], w.choices["cant_deg"],
                    w.physics_flags().get("endplate_cant_deg"))
            tp.param("cant").set("free")
            w.fixed["endplate_cant_deg"] = 30.0
            cfl = w.physics_flags().get("endplate_cant_deg")
            tp.param("cant").set("stated")
            stale = "endplate_cant_deg" in w.pinned() or "endplate_cant_deg" in w.fixed
            card(w, {"plates": True})
            rep("M35b the card's rules: canted opens at 75 (a typed lean kept); 'none' pins the "
                "plate's height over a hand fix and releasing it fits a plate; typed numbers "
                "survive the mount and the optimiser; an optimised lean is never sent; the old "
                "designed / fences words still read",
                lean0 == bridge.CANT_SHOWN and lean1 == 60.0 and pin_none == 0.0
                and released == ("vertical", 0.10) and fences == (False, True)
                and kept == (0.3, 70.0, 70.0) and cfl is None and not stale,
                f"canted {lean0:g} then {lean1:g}; none pin {pin_none} over 0.10; released -> "
                f"{released}; kept {kept}; optimised lean sent {cfl}")
        guard("M35", m35)

        def m36():
            w = u.wing
            bad = []
            for label, ch, _want in bridge.MOUNT_CONFIGS:
                rec, _x, _labels = bridge._mount_record(u.op, ch)
                card(w, {"plates": not ch.get("plates", True)})       # the other mount
                w.reopen(rec)
                got = {k: w.choices.get(k) for k in ("plates", "chord_law", "blend", "cant", "tip",
                                                     "tip_cant_deg") if k in ch}
                #  ...and the card flies again exactly what the record flew
                if got != {k: ch[k] for k in got} or w.family_name != rec["config"]["problem_name"] \
                        or w.physics_flags() != rec["config"]["flags"] \
                        or {k: v for k, v in w.pinned().items() if k == "endplate_h_m"} \
                        != (rec["config"]["pinned"] or {}):
                    bad.append(f"{label}: {got}, flags {sorted(w.physics_flags())}")
            old = {"config": {"flags": {"chord_trend": "root_largest", "mount": "tips"}},
                   "carsim_family": {"role": "top", "plates": False, "chord_law": True,
                                     "ride_band": list(u.op.ride_band), "deck": None, "lap": None}}
            card(w, {"plates": True})
            w.reopen(old)
            legacy = (w.choices["plates"], w.choices["tip"])
            card(w, {"plates": True})
            rep("M36 a record re-opens the card on the answers it flew (mount, freedoms, stated "
                "numbers, tip device); a legacy fence record on the pylons, vertical plates",
                not bad and legacy == (False, "vertical"),
                "; ".join(bad) if bad else f"{len(bridge.MOUNT_CONFIGS)} records; legacy -> {legacy}")
        guard("M36", m36)

        def m37():
            #  THE REAL JOB PATH: AeroBO's engine live (no replay), a short
            #  own budget, through start_run -> EngineJob -> the law job -> fit
            w = u.wing
            own0 = dict(pol.own)
            use_engine()
            pol.set_mode("own")
            pol.own["wing"] = 10
            out = {}
            try:
                for label, ch, seed in (("pylons, canted 70",
                                         {"plates": False, "tip": "canted", "tip_cant_deg": 70.0}, 0),
                                        #  seed 0: its Sobol start alone holds no
                                        #  plate that reaches the deck (0 of 10 --
                                        #  and 0 of 59 at the recommended budget
                                        #  -- before the warm start); the upright,
                                        #  creased x_seed (`WingModel.cfg`) is what
                                        #  lands it
                                        ("endplates, both optimised",
                                         {"plates": True, "blend": "free", "cant": "free"}, 0)):
                    card(w, ch)
                    w.reset()
                    w.seed = seed
                    started = w.start_run()
                    run()                               # the run, then the law and report jobs
                    run()
                    run()
                    rec = w.record or {}
                    bd = rec.get("breakdown") or {}
                    w.fit(f"check-{label.split(',')[0]}")
                    out[label] = (started, rec, bd, w.spec, w.law)
            finally:
                pol.set_mode("recommended")
                pol.own.update(own0)
                use_fixtures(FIXTURES)
                w.seed = 0
                card(w, {"plates": True})
            ok_, bits = True, []
            for label, (started, rec, bd, spec, law) in out.items():
                best = rec.get("best_score")
                labels = list(rec.get("param_labels") or [])
                x = dict(zip(labels, rec.get("best_x") or []))
                fine = (started and rec.get("feasible") and best is not None
                        and math.isfinite(float(best)) and law is not None and spec is not None)
                if label.startswith("pylons"):
                    fine = fine and (bd.get("mount_kind") == "pylon"
                                     and abs(float(bd["pylon_length_m"])
                                             - (float(x["ride_height_m"]) - float(u.op.deck))) < 1e-9
                                     and spec.mount == "pylon" and spec.pylon_frac == 0.35
                                     and spec.plate_cant_deg == 70.0 and spec.plate_chord_follows
                                     and law.get("mount") == "pylon")
                else:
                    drawn = (spec.plate_cant_deg, spec.plate_blend, spec.plate_chord_ratio)
                    flown = tuple(float(x[k]) for k in ("endplate_cant_deg", "endplate_blend_frac",
                                                        "endplate_chord_ratio"))
                    seed_x = dict(zip(labels, ((rec.get("config") or {}).get("x_seed") or [])))
                    fine = fine and (bd.get("mount_kind") is None and spec.mount == "endplate"
                                     and seed_x.get("endplate_cant_deg") == 90.0
                                     and seed_x.get("endplate_blend_frac") == 0.0
                                     and bd.get("endplate_cant_searched")
                                     and bd.get("endplate_blend_searched")
                                     and all(abs(a - b) < 1e-12 for a, b in zip(drawn, flown))
                                     and spec.plate_shape == "spiral")
                ok_ = ok_ and bool(fine)
                bits.append(f"{label}: {rec.get('n_feasible')}/{rec.get('n_evals')} feasible, best "
                            f"{float(best) if best is not None else float('nan'):.3f}, drawn "
                            f"{getattr(spec, 'mount', '-')} lean {getattr(spec, 'plate_cant_deg', 0):.1f}"
                            f" blend {getattr(spec, 'plate_blend', 0):.2f}")
            rep("M37 the real job path (WingLab live, own budget 10, seed 0): a pylon run and a "
                "both-optimised run (warm-started on the upright, creased plate) land feasible, "
                "their law is derived and the wing is fitted as flown",
                ok_ and len(out) == 2, "; ".join(bits))
        guard("M37", m37)
    finally:
        use_engine()
        for k, v in was_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tmp, ignore_errors=True)
    if verbose:
        print(f"  {n_rows[1]}/{n_rows[0]} " + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
