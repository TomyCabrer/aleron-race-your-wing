"""drive/design_shell.py -- the AeroBO shell around the garage's MISSION and
DESIGN pages (PLAN §2.9, SPECS/aerobo_shell.md).

The two pages were a dark list and a dark navigator. They are now one frame,
AeroBO v3's: a menu bar and a tool bar across the top, the Simulation tree
and the Properties grid down the left, the selected stage's views as tabs
over a white work area, the Output log under it and the status bar along
the bottom. Everything the frame SAYS -- which stage is locked and why, what
each stage holds (the tree's chips), what is being designed (Properties),
what evaluation is running (status bar, Output) -- is read off the design
page's models every frame (`drive/aerobo_models.py`: one `DesignSession` per
slot, on AeroBO's own engine); nothing is cached that a model could move
under it.

The frame is `DesignShell` (one per `Garage`); the garage hands it the frame
on its two pages (`draw`) and every event (`handle`, `pad`). The stage views
live in `drive/views_*.py` and are imported lazily, one module per stage
group, inside the per-view try/except: a view that fails -- at import or
while drawing -- draws its own render-error card and nothing else breaks.

Three rules the rest of the garage relies on:

  * THE NAVIGATOR IS STILL A `ui.Nav`. `CaeTree` subclasses it: the gate
    (`select` without force), `refused`, `_rect` / `_hits` for the anchors
    and the mouse -- all kept. AeroBO's expand-on-select is added on top: a
    stage's views are drawn (and hit-recorded) once that stage has been
    selected, and every select scrolls its block into view.
  * STAGES ARE GATED, NOT VIEWS. A locked stage refuses with a toast naming
    the reason; inside an unlocked one every view is selectable (a forced
    select) and one with nothing to show draws its empty state.
  * A LIVE RUN LOCKS ONLY WHAT WOULD CHANGE IT. The forms of the run's own
    model and what feeds its objective draw disabled and refuse with the
    lock's sentence; a short list of global hazards (the design keys, the
    mission, starting over, the pause menu) is refused whatever is on
    screen, and so is every button that would start a second engine run
    (AeroBO runs one at a time, on one worker thread). Everything else
    stays live -- every run flies a frozen snapshot.

`dp.focus` stays two-valued ("nav" / "rows", pinned): the shell's three
keyboard regions (tree, tabs, work) are mirrored into it.

`python3 -m drive.design_shell [--only KEYS]` runs the self-check on the
captured AeroBO fixtures (KEYS: view keys or stage prefixes, `--only w.,r.`).
"""

from __future__ import annotations

import importlib
import math
import sys
import time
import traceback
from dataclasses import dataclass, field

import pygame

from . import garage_ui as ui
from . import design_jobs as dj
from .cae import chrome as C
from .cae import theme as T
from .cae import widgets as W
from .aero import mission as _ms

#: the wing objective in a chip's few words (AeroBO's keys; a side wing's
#: "downforce" is side force)
OBJECTIVE_WORDS = {"efficiency": "efficiency", "downforce": "downforce", "drag": "drag",
                   "laptime": "lap time", "downforce_plus_drag": "downforce + drag"}


def _objective_word(wing) -> str:
    w = OBJECTIVE_WORDS.get(wing.choices["objective"], str(wing.choices["objective"]))
    return w.replace("downforce", "side force") if wing.role != "top" else w
from .cae.form import Form

#: the garage pages this shell frames (the car, airfoil and library pages
#: keep their dark look)
SHELL_PAGES = ("mission", "section")

# --------------------------------------------------------------------------- #
#  the tables (PLAN §2.9.1)                                                    #
# --------------------------------------------------------------------------- #
#: key, tree label (two spaces, as AeroBO; drawn with one), crumb, views
STAGES = (
    ("m", "1  Mission", "Mission", ("m.operating", "m.design", "m.search")),
    ("af", "2  Airfoil", "Airfoil", ("af.screen", "af.rank", "af.section", "af.opt")),
    ("ep", "2.8  Endplate", "Endplate", ("ep.screen", "ep.rank", "ep.section", "ep.opt")),
    ("w", "3  Wing", "Wing", ("w.type", "w.box", "w.solver", "w.conv")),
    ("r", "4  Results", "Results", ("r.summary", "r.geometry", "r.loading", "r.evals")),
)
STAGE_KEYS = tuple(s[0] for s in STAGES)
STAGE_VIEWS = {s[0]: s[3] for s in STAGES}
STAGE_OF = {v: s[0] for s in STAGES for v in s[3]}
CRUMBS = tuple((s[0], s[2]) for s in STAGES)

_META = {"m.operating": ("Operating point", "tune"), "m.design": ("Design point", "speed"),
         "m.search": ("Search & budget", "bolt"),
         "*.screen": ("Library screening", "tune"), "*.rank": ("Ranking", "format_list_numbered"),
         "*.section": ("Section", "gesture"), "*.opt": ("Shape optimisation", "auto_graph"),
         "w.type": ("Wing type", "category"), "w.box": ("Design box", "crop_free"),
         "w.solver": ("Solver", "settings"), "w.conv": ("Convergence", "show_chart"),
         "r.summary": ("Summary", "summarize"), "r.geometry": ("Geometry", "view_in_ar"),
         "r.loading": ("Loading", "ssid_chart"), "r.evals": ("Evaluations", "table_rows")}
#: view key -> (label, icon), every one of the 19 spelled out
VIEW_META = {v: _META.get(v, _META.get("*." + v.split(".", 1)[1])) for v in STAGE_OF}

#: view key -> "module:function" in drive/ (W2 stubs; W3 builds them)
VIEWS = {
    "m.operating": "views_mission:operating", "m.design": "views_mission:design",
    "m.search": "views_mission:search",
    "af.screen": "views_section:screen", "af.rank": "views_section:ranking",
    "af.section": "views_section:section", "af.opt": "views_section:optimise",
    "ep.screen": "views_section:screen", "ep.rank": "views_section:ranking",
    "ep.section": "views_section:section", "ep.opt": "views_section:optimise",
    "w.type": "views_wing:wing_type", "w.box": "views_wing:design_box",
    "w.solver": "views_wing:solver", "w.conv": "views_wing:convergence",
    "r.summary": "views_results:summary", "r.geometry": "views_results:geometry",
    "r.loading": "views_results:loading", "r.evals": "views_results:evaluations",
}

#: the tree's state glyphs (SPECS/aerobo_shell.md §3.3)
GLYPH = {"done": ("check_circle", "GOOD"), "ready": ("radio_button_unchecked", "INK_MUTED"),
         "active": ("play_circle", "ACCENT"), "running": ("pending", "ACCENT"),
         "locked": ("lock", "INK_FAINT"), "warn": ("error_outline", "WARN"),
         "error": ("cancel", "BAD")}

# --------------------------------------------------------------------------- #
#  the texts                                                                   #
# --------------------------------------------------------------------------- #
#: what a live run leaves editable inside its own scope (PLAN §2.9.8): pure
#: navigation and file / log output
RUN_SAFE = frozenset({"stop", "snip", "to1", "to2", "to28", "to3", "tobox", "s1", "refine",
                      "x_dat", "x_csv", "x_json", "x_all", "ex_md", "ex_csv", "ex_json"})
#: the buttons that START an engine run (or put a wing on the car): AeroBO
#: flies one run at a time on one worker thread, so while any run is live
#: each of these refuses with that run's lock sentence, wherever it is
LAUNCH_KEYS = frozenset({"go", "run", "keep_r", "fit", "save"})
#: what a wing run reads from the section stages: taking or declining a
#: section there would change what the running wing is said to fly
WING_READS = frozenset({"use", "decline"})
LOCK_TEXT = "locked while the {evaluator} run is going — ■ Stop to change it"
HAZARD = "a run is in progress — stop it first"
NOT_LIVE = ("state the mission first — every section and the wing are designed at the "
            "operating point it states")
SESSION_LINE = ("Garage — WingLab's engine, on the car's mission: state the mission, then "
                "the sections, then the wing. Budgets are WingLab's measured ones (164 a section, "
                "53 the wing at balanced; 1 Mission ▸ Search & budget). Keys: TAB tree/tabs/work · "
                "ENTER do it · F5 run · ESC stop/back · F1 all keys")
STUB_HINT = "view not built yet — generic layout"
RENDER_ERROR_HINT = "This block failed to render; the rest of the page is unaffected."
WING_NONE_HINT = "State the mission first: the wing is created when the design stages open."
RANK_EMPTY_HINT = "Nothing screened yet — run the screening on the previous tab."

#: the tool bar (PLAN §2.9.5)
TOOLS = (("icon", "restart", "restart_alt",
          "Start the design over — the slot's screens, sections and runs forgotten; the mission stays"),
         ("sep",),
         ("icon", "save", "save", "Save the wing to the library and put it in the slot (S)"),
         ("sep",),
         ("primary", "run", "play_arrow", "Run", "Run the current stage (F5 · SQUARE)"),
         ("flat", "stop", "stop", "Stop", "Stop what is running (ESC · CIRCLE)"),
         ("sep",),
         ("icon", "prev", "chevron_left", "Previous stage"),
         ("icon", "next", "chevron_right", "Next stage"),
         ("sep",), ("crumbs",), ("space",), ("chips",))

#: the menu bar: (title, [(label, icon, action_id, key hint)])
MENUS = (("File", [("Start the design over", "restart_alt", "restart", None),
                   ("Save the wing", "save", "save", "S"),
                   ("Rename the wing…", "edit", "rename", "N"),
                   ("Airfoil library…", "menu_book", "airfoils", "A"),
                   ("Back", "arrow_back", "back", "ESC")]),
         ("Edit", [("Reset the design box", "crop_free", "boxreset", None),
                   ("Keep the family's own plate", "layers_clear", "flat", None),
                   ("Use the recommended weights", "tune", "rec", None)]),
         ("Solution", [("Run current stage", "play_arrow", "run", "F5"),
                       ("Stop", "stop", "stop", "ESC"),
                       ("Continue / keep going", "play_arrow", "keep", "K")]),
         ("Tools", [("Copy the run configuration to the output", "content_copy", "snip", None),
                    ("Clear output log", "clear_all", "clearlog", None)]),
         ("Help", [("Keys and controller", "keyboard", "keys", "F1"),
                   ("About this pipeline", "info", "about", None)]))

#: the design keys (pinned), each a global hazard while a run is live
DESIGN_KEYS = {pygame.K_l: "L", pygame.K_o: "O", pygame.K_k: "K", pygame.K_f: "F",
               pygame.K_s: "S", pygame.K_x: "X", pygame.K_n: "N", pygame.K_a: "A"}
#: menu actions that are a global hazard while a run is live
_MENU_HAZARDS = ("restart", "save", "rename", "airfoils", "keep")
#: the pad (PS names, PLAN §2.9.9) as the keys they stand for
_PAD_KEYS = (("up", pygame.K_UP), ("down", pygame.K_DOWN), ("left", pygame.K_LEFT),
             ("right", pygame.K_RIGHT), ("cross", pygame.K_RETURN), ("circle", pygame.K_ESCAPE),
             ("square", pygame.K_F5), ("triangle", pygame.K_TAB), ("r1", pygame.K_RIGHTBRACKET))
#: the constants a view may want from the garage (views never import it):
#: `wing_polys` / `deck_z` are how Results ▸ Geometry lofts the mapped wing
#: the way the car page draws it
CONST_NAMES = ("SLOT_LABEL", "SLOT_ROLE", "CAR_X_FRONT", "CAR_X_REAR", "LIGHT_DIR", "OBJECTIVES",
               "wing_polys", "deck_z")
#: px a wheel notch scrolls a pane (at S = 1)
WHEEL_PX = 60
#: the keys dialog (PLAN §2.9.10): (keyboard, pad, what)
KEY_ROWS = (("TAB", "TRIANGLE", "tree → tabs → work area"),
            ("UP / DOWN", "d-pad", "move in the focused region"),
            ("LEFT / RIGHT", "d-pad", "tree: stage · tabs: view · work: change the value (SHIFT / L1 fine)"),
            ("ENTER", "CROSS", "do it (the step, the button, the row)"),
            ("F5", "SQUARE", "Run the current stage"),
            ("ESC", "CIRCLE", "stop a run; otherwise back"),
            ("[ ]", "R1", "previous / next tab"),
            ("L", None, "screen the library"),
            ("O / K", None, "optimise / keep going"),
            ("F", None, "take the section on screen"),
            ("S", None, "put the wing on the car"),
            ("A", None, "the airfoil library"),
            ("N", None, "rename"),
            ("H", None, "hide the tutorial box"),
            ("F1", None, "this list"))
ABOUT_ROWS = (("1", "State the mission: a lap of one of the game's circuits on a surface, for a "
                    "slot. It sets WingLab's operating point — the design speed, the air, where "
                    "the wing sits."),
              ("2", "Choose the wing's section and the endplates' (symmetric): screen WingLab's "
                    "library, take one, or shape-optimise it with WingLab's CST search."),
              ("3", "Design the wing: WingLab's BO → SLSQP searches its car rear wing's box "
                    "against the objective you choose — efficiency, downforce, drag, lap time."),
              ("4", "Read the results: what the search found, WingLab's design report, every "
                    "evaluation — and the law the car flies it with."))
ABOUT_CLOSE = ("Every number on these pages is WingLab's own engine: its XFOIL screens and "
               "section search, its car-wing lattice, its "
               "optimisers and budgets. carsim supplies the mission and flies the winner with a "
               "law sampled from WingLab's evaluator at the winning design.")


def views_of(stage: str) -> tuple:
    return STAGE_VIEWS.get(stage, ())


#: every (family, css, bold) the chrome and the work-area kit draw with
#: (`theme.warm` shapes them now; `_warm_text` is a no-op after it)
_FACES = T.WARM_FACES


def _warm_text() -> None:
    """Measure one string in every face the shell draws with. Opening a font
    (`theme.warm`) is not the whole cost: the FIRST shaping call on each
    face instance pays the face's lazy load -- measured 71 ms for SF 12 px
    on this Mac, ~300 ms over a first shell frame -- so it is paid here,
    outside every timed frame, once per process (the faces are cached)."""
    for family, css, bold in _FACES:
        T.text_w("Ag 0.5 · ▸", family, css, bold)


def _px(v: float) -> int:
    return int(math.floor(v * T.S + 0.5))


def _finite(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def short_units(units) -> str:
    """AeroBO's score units without their gloss: "CZ/CD (= F_z/D)" ->
    "CZ/CD" (a chip has room for the name, not the definition)."""
    return str(units or "").split(" (")[0].strip()


def score_text(value, units="") -> str:
    """A run's best score as the chips and Properties print it."""
    v = _finite(value)
    if v is None:
        return "—"
    u = short_units(units)
    return f"{v:.4g}" + (f" {u}" if u else "")


# --------------------------------------------------------------------------- #
#  geometry (PLAN §4.1)                                                        #
# --------------------------------------------------------------------------- #
def scale_for(W: int, H: int) -> float:
    """The shell's scale: 1.0 for every window up to 1999 px wide (so the
    parity sizes 1280x800 and 1600x1000 are px for px), then quarter steps."""
    return max(1.0, math.floor(min(W / 1600.0, H / 1000.0) * 4.0) / 4.0)


@dataclass(frozen=True)
class Layout:
    """The shell's rectangles for one window size, at the current scale."""
    menu: pygame.Rect
    tool: pygame.Rect
    status: pygame.Rect
    tree: pygame.Rect
    props: pygame.Rect
    main: pygame.Rect
    tabs: pygame.Rect
    work: pygame.Rect
    output: pygame.Rect
    tutor_box: pygame.Rect


def layout(W: int, H: int) -> Layout:
    """SPECS/aerobo_shell.md §1.4: fixed chrome, a 3 : 2 tree / Properties
    split down the left, the work area absorbing whatever the window adds.
    `tutor_box` is where the wing tutorial's box sits on these pages: the
    work area's bottom-right, above the Output log (PLAN §4.2)."""
    top = T.MENU_H + T.TOOL_H
    col_y = top + T.PAD
    col_h = H - top - T.STATUS_H - 2 * T.PAD
    tree_h = int((col_h - T.GAP) * T.TREE_SPLIT)
    main_x = T.PAD + T.LEFT_W + T.GAP
    main_w = W - main_x - T.PAD
    main_h = col_h - T.GAP - T.OUTPUT_H
    R = pygame.Rect
    return Layout(menu=R(0, 0, W, T.MENU_H), tool=R(0, T.MENU_H, W, T.TOOL_H),
                  status=R(0, H - T.STATUS_H, W, T.STATUS_H),
                  tree=R(T.PAD, col_y, T.LEFT_W, tree_h),
                  props=R(T.PAD, col_y + tree_h + T.GAP, T.LEFT_W, col_h - T.GAP - tree_h),
                  main=R(main_x, col_y, main_w, main_h),
                  tabs=R(main_x + 1, col_y + 1, main_w - 2, T.TAB_STRIP_H),
                  work=R(main_x + 1, col_y + 1 + T.TAB_STRIP_H, main_w - 2, main_h - 2 - T.TAB_STRIP_H),
                  output=R(main_x, col_y + main_h + T.GAP, main_w, T.OUTPUT_H),
                  tutor_box=R(W - T.PAD - _px(8) - _px(520), col_y + main_h - _px(212) - _px(8),
                              _px(520), _px(212)))


# --------------------------------------------------------------------------- #
#  what a view is handed (PLAN §2.9.6)                                         #
# --------------------------------------------------------------------------- #
@dataclass
class ViewCtx:
    """Everything a view function reads. Views never import `garage`: the
    models arrive here, and the garage's constants in `consts`.

    `model` is the view's own model -- the MissionPage for m.*, the
    `aerobo_models.SurfaceModel` (dp.af / dp.ep) for af.* / ep.*, the
    `WingModel` (dp.wing) for w.* and the `ResultsModel` (dp.results) for
    r.* -- None before the design page has opened a session. `job` is that
    model's LIVE job (`runs.job_for`), None otherwise; on r.* it is the
    wing's live run when there is one (a Keep going watched from Results),
    else the results' own report job. Post-run content is read from the
    model's STORED record and outcome, never from a finished job. `state` is
    this view's persistent UI dict (tables, disclosures, check boxes)."""
    g: object
    dp: object
    mp: object
    key: str
    stage: str
    view: str
    model: object
    job: object
    live: bool
    runs: object
    notices: object
    state: dict
    select: object
    width: int
    size: tuple
    consts: dict = field(default_factory=dict)
    shell: object = None


# --------------------------------------------------------------------------- #
#  the navigator                                                               #
# --------------------------------------------------------------------------- #
class CaeTree(ui.Nav):
    """The design page's navigator, still a `garage_ui.Nav` (every pinned
    name and the gate are the Nav's own), with AeroBO's tree on top.

    `expanded` is which stages show their views: only the mission on a
    fresh garage; a successful `select` expands the selected step's stage
    and asks the shell to scroll its block into view (`reveal_stage`);
    nothing collapses a stage but its twisty. The shell draws the rows and
    records, per frame, `_rect` (the Simulation pane's body), `_hits`
    (`[(key, y, h)]`, the step rows drawn and fully visible, in tree order)
    and `_node_hits` (`[(id, Rect, kind)]`, the stage rows, the mission's
    view rows and the twisties). A collapsed stage's steps are not drawn,
    so they are not hit-recorded either."""

    def __init__(self, tree, title: str = "", state=None, note=None, reason=None):
        super().__init__(tree, title, state, note, reason)
        self.expanded = {k: k == "m" for k in STAGE_KEYS}
        self.reveal_stage = None
        self._rect = None
        self._hits = []
        self._node_hits = []

    def select(self, key: str, force: bool = False) -> bool:
        ok = super().select(key, force)
        if ok:
            stage = key.split(".", 1)[0]
            if stage in self.expanded:
                self.expanded[stage] = True
            self.reveal_stage = stage
        return ok


# --------------------------------------------------------------------------- #
#  the fallback view                                                           #
# --------------------------------------------------------------------------- #
def _control(ui_, p) -> None:
    """One Param as the control its kind asks for (PLAN §2.9.6)."""
    if p.set is None and p.kind != "action":
        ui_.readout(p.key)
    elif p.kind == "bool":
        ui_.field(p.key, control="switch")
    elif p.kind == "choice":
        n = len(p.choices)
        if n == 0:
            ui_.readout(p.key)
        else:
            ui_.field(p.key, control="toggle" if n <= 3 else "select")
    elif p.kind == "float" and p.lo == 0 and p.hi == 1 and not getattr(p, "costly", False):
        ui_.slider(p.key)
    else:
        ui_.field(p.key, control="number")


def _generic_card(ui_, title, params) -> None:
    if not params:
        return
    with ui_.card(" ".join(str(title).split()) if title else ""):
        i = 0
        while i < len(params):
            p = params[i]
            if p.kind == "action":
                acts = []
                while i < len(params) and params[i].kind == "action":
                    acts.append(params[i])
                    i += 1
                with ui_.row():
                    for a in acts:
                        ui_.button(a.key, kind="primary" if a.key in ("go", "run", "fit") else "outline")
                continue
            _control(ui_, p)
            i += 1


def _generic_rank(ui_, ctx) -> None:
    """A `.rank` view before W3: the ranking list as a two-column table;
    a click on a row takes it (the same path as ENTER on the step)."""
    items = ctx.dp.rank_list.items
    if not items:
        ui_.hint(RANK_EMPTY_HINT)
        return
    rows = [dict(name=it[0], score=it[2] if len(it) > 2 else "") for it in items]
    cols = [dict(key="name", head="section", align="left"), dict(key="score", head="score")]
    take = ctx.shell.take_rank if ctx.shell is not None else (lambda i: None)
    ui_.table("rank", cols, rows, cursor=ctx.dp.rank_list.idx, on_row=take, sortable=False)


def generic_view(ui_, ctx) -> None:
    """The layout every W2 stub draws, and any view may fall back on: one
    card per run of Params between two `kind="label"` Params (the label is
    the card's title); each Param as its kind's control -- a float on
    [0, 1] that is not `costly` a slider, any other number a number field,
    a choice of at most three a toggle and a longer one a select, a bool a
    switch, an action a button (primary for `go`, `run`, `fit`), a read-only
    Param a read-out. On `.rank` (no form) the ranking list is a table.
    Every pinned Param is therefore drawn, hit-recorded and reachable by
    the keyboard before a stage's own view exists."""
    if ctx.stage in ("w", "r") and ctx.dp.wing is None:
        ui_.hint(WING_NONE_HINT)
        return
    if ctx.view.endswith(".rank"):
        _generic_rank(ui_, ctx)
        return
    title, run = None, []
    for p in list(ui_.form.params):
        if p.kind == "label":
            _generic_card(ui_, title, run)
            title, run = p.label, []
            continue
        run.append(p)
    _generic_card(ui_, title, run)


# --------------------------------------------------------------------------- #
#  the shell                                                                   #
# --------------------------------------------------------------------------- #
class _Lock:
    """`form.locked` for one form: the shell's scoped live-run lock."""
    __slots__ = ("shell", "form")

    def __init__(self, shell, form):
        self.shell, self.form = shell, form

    def __call__(self, p):
        return self.shell.lock_reason(self.form, p)


class DesignShell:
    """The AeroBO frame of the garage's "mission" and "section" pages.

    The garage calls `draw(screen)` on those pages, `handle(ev)` with every
    event and `pad(edges, lx, ly, fine)` with the pad's edges. Everything it
    shows is derived from the garage's models on each draw; what it keeps
    is UI state only: the keyboard region, the tree's scroll and cursor, each
    view's scroll and state dict, the Output log's lines, the toasts."""

    def __init__(self, g):
        self.g = g
        w, h = g.screen.get_size()
        T.set_scale(scale_for(w, h))
        T.warm()
        _warm_text()
        self.overlay = W.Overlay()
        self.menubar = C.MenuBar("carsim", MENUS)
        self.toolbar = C.ToolBar(TOOLS, overlay=self.overlay)
        self.treedraw = C.TreeDraw(self.overlay)
        self.propgrid = C.PropertyGrid(self.overlay)
        self.tabstrip = C.TabStrip()
        self.log = C.OutputLog()
        self.statusbar = C.StatusBar(self.overlay)
        self.toasts = C.Toasts()
        self.dialog = None
        #: the tracked pointer: MOUSEMOTION and button events only, never
        #: `pygame.mouse.get_pos()` (the screenshot harness parks it off screen)
        self.mouse = (-1, -1)
        self.kbd = False                # focus rings show after keyboard / pad input only
        self._region = "tree"
        self.tree_scroll = 0
        self.props_scroll = 0
        self.tree_cursor = None         # the tree row the keyboard is on
        self.scroll: dict = {}          # view -> work-area scroll
        self.content_h: dict = {}       # view -> content height of its last draw
        self.vstate: dict = {}          # view -> its persistent UI state
        self.remembered: dict = {}      # stage -> its last view (AeroBO's ui.tab)
        self.view_errors: dict = {}     # view -> the traceback of its last failed draw
        self.safe_errors: dict = {}     # where -> the last exception `safe` swallowed
        self._logged: set = set()
        self._started = False           # the session line is logged on the first draw
        self._tc: dict = {}             # (airfoil name, spec id) -> t/c
        self._rows: list = []           # the tree rows of the last build
        self._tree_hits: list = []
        self._cells: list = []
        self._bars: dict = {}           # pane -> (track, thumb) of its last scrollbar
        self._drag = None               # (pane, ScrollDrag) while a thumb is held
        self._err_state: dict = {}
        self._reveal = None             # a stage whose block the next draw scrolls into view
        self._L = layout(w, h)
        mod = sys.modules.get(type(g).__module__)
        self.consts = {k: getattr(mod, k) for k in CONST_NAMES if hasattr(mod, k)}

    # -- model shortcuts ------------------------------------------------------------------------------
    @property
    def dp(self):
        return self.g.design_page

    @property
    def mp(self):
        return self.g.mission_page

    @property
    def stage(self) -> str:
        """The selected stage: the mission on the mission page, else the
        stage of the navigator's current step."""
        if self.g.page == "mission":
            return "m"
        return STAGE_OF.get(self.dp.nav.current(), "af")

    @property
    def view(self) -> str:
        if self.g.page == "mission":
            t = self.g.mission_tab
            return t if t in STAGE_OF else "m.operating"
        return self.dp.nav.current()

    @property
    def region(self) -> str:
        """The keyboard region, mirrored into the pinned two-valued
        `dp.focus`: "work" is "rows", "tree" and "tabs" are "nav". A caller
        that writes `dp.focus` moves the region with it."""
        if self.dp.focus == "rows":
            return "work"
        return self._region if self._region in ("tree", "tabs") else "tree"

    @region.setter
    def region(self, r: str) -> None:
        self._region = r
        self.dp.focus = "rows" if r == "work" else "nav"

    def geom(self, W: int, H: int) -> Layout:
        return layout(W, H)

    def tool_rect(self, tool_id):
        """Where the last draw put tool button `tool_id` (the tutorial's
        `('tool', id)` anchor), or None."""
        r = self.toolbar.rects.get(tool_id)
        return pygame.Rect(r) if r is not None else None

    def design_live(self) -> bool:
        g, dp = self.g, self.dp
        return bool(g.mission.stated and dp.wing is not None and dp.key == g.sel)

    def _expanded(self) -> dict:
        ex = getattr(self.dp.nav, "expanded", None)
        if ex is None:
            ex = self.__dict__.setdefault("_exp", {k: k == "m" for k in STAGE_KEYS})
        return ex

    def editing(self) -> bool:
        """Is a number field being typed into (every key belongs to it)?"""
        f = self._current_form()
        return f is not None and f.editing is not None

    # -- None-safety ------------------------------------------------------------------------------------
    def safe(self, fn, default="—", where: str = ""):
        """`fn()`, or `default` when it raises. Every read the chrome makes
        goes through here: a half-built model must never take the frame down.
        The exception is logged once (error) and kept in `safe_errors`."""
        try:
            return fn()
        except Exception as exc:                        # noqa: BLE001 -- the chrome never raises
            key = where or getattr(getattr(fn, "__code__", None), "co_name", "chrome")
            self.safe_errors[key] = f"{type(exc).__name__}: {exc}"
            tag = (key, type(exc).__name__, str(exc))
            if tag not in self._logged:
                self._logged.add(tag)
                self.g.log(f"{key}: {type(exc).__name__}: {exc}", "error")
            return default

    # -- stages -------------------------------------------------------------------------------------------
    def _base(self, stage: str) -> tuple:
        """(state, reason) of a stage before the running / active overrides
        (PLAN §2.9.3): locked | done | ready | error. The design stages are
        the slot's `DesignSession.state` -- AeroBO's gating (PLAN2 D10): once
        the mission is stated 2 Airfoil, 2.8 Endplate and 3 Wing are all
        ready, 2.8 is locked on a wing the pylons carry, 4 Results waits for
        a completed run. The session's own "running" is folded back to what
        the stage holds: the tree's running glyph is `stage_state`'s, from
        the live job."""
        g, dp, mp = self.g, self.dp, self.mp
        if stage == "m":
            if g.mission.stated:
                return "done", ""
            if not mp.flies():
                return "error", mp.err or "the lap did not close"
            return "ready", ""
        if not self.design_live():
            return "locked", NOT_LIVE
        s = dp.session
        st, why = s.state(stage)
        if st == "running":
            if stage in ("af", "ep"):
                st = "done" if (s.af if stage == "af" else s.ep).finished() else "ready"
            elif stage == "w":
                st = "done" if s.wing.record else "ready"
            else:
                st = "done"
            why = ""
        return st, why

    def stage_state(self, stage: str) -> tuple:
        """(state, reason) as the tree shows it: the base state, then a live
        engine job of the stage -> running (over done), then the selected
        unfinished stage -> active."""
        st, why = self.safe(lambda: self._base(stage), ("ready", ""), f"stage {stage}")
        job = self.g.runs.live
        if isinstance(job, dj.EngineJob) and getattr(job.info, "stage", None) == stage:
            st = "running"
        if st != "locked" and stage == self.stage and st not in ("done", "running"):
            st = "active"
        return st, why

    def select(self, stage: str, view: str | None = None) -> bool:
        """AeroBO's `ctx.select`, what every UI entry point goes through: the
        mission is shown on `view` (no commit, no reset); a locked design
        stage refuses with an info toast naming its reason; an unlocked one
        shows `view` -- or the stage's remembered view -- by a FORCED select,
        so every view of an unlocked stage is reachable (a view with nothing
        to show draws its empty state)."""
        g, dp = self.g, self.dp
        self._commit_edit()
        if stage == "m":
            v = view if view in STAGE_VIEWS["m"] else g.mission_tab
            g.goto_mission(v if v in STAGE_VIEWS["m"] else "m.operating")
            self._expanded()["m"] = True
            self._reveal = "m"
            self.tree_cursor = None
            return True
        if stage not in STAGE_VIEWS:
            return False
        st, why = self.safe(lambda: self._base(stage), ("locked", ""), f"stage {stage}")
        if st == "locked":
            g.toast(why or f"{STAGES[STAGE_KEYS.index(stage)][2]} is not available yet", "info")
            return False
        views = STAGE_VIEWS[stage]
        v = view if view in views else (self.remembered.get(stage) or views[0])
        dp.nav.select(v, force=True)
        self._expanded()[stage] = True
        self._reveal = stage
        self.remembered[stage] = v
        g.page = "section"
        self.tree_cursor = None
        return True

    def settle(self) -> None:
        """A row changed: the stage on screen may have LOCKED under it (the
        plates dropped to zero height). Only then does the page move -- a
        shut view inside an unlocked stage stays, AeroBO shows its empty
        state."""
        if self.g.page == "section" and self._base(self.stage)[0] == "locked":
            self.dp.settle()

    # -- the live-run lock (PLAN §2.9.8) ------------------------------------------------------------
    def _wing_forms(self) -> list:
        """The wing's forms and the results' (Keep going, Put it on the car):
        what a wing run's scope locks. [] before a session."""
        dp = self.dp
        w = dp.wing
        if w is None:
            return []
        out = [w.type_params, w.box_params(), w.solver_params, w.conv_params]
        if dp.results is not None:
            out.append(dp.results.params)
        return [f for f in out if f is not None]

    @staticmethod
    def _surface_forms(m) -> list:
        return [] if m is None else [m.screen_params, m.params, m.opt_params]

    def lock_reason(self, form, p):
        """Why `p` of `form` refuses while a run is live, or None.

        Inside the run's scope every Param -- setter or action -- is locked
        unless it is pure navigation or output (RUN_SAFE): a section run
        scopes its surface's three forms; a wing run (and the law / report
        jobs that follow it) the wing's and the results' forms, plus taking
        or declining a section, which is what the wing flies. The mission
        form is a global hazard. Anywhere else, a button that would START a
        second engine run is refused with the live run's sentence -- AeroBO
        flies one at a time. Everything else stays live."""
        job = self.g.runs.live
        if job is None or p is None or p.key in RUN_SAFE:
            return None
        if form is self.mp.params:
            return HAZARD
        dp, owner = self.dp, job.owner
        why = LOCK_TEXT.format(evaluator=job.info.evaluator)
        if owner is not None and (owner is dp.wing or owner is dp.results):
            if any(form is f for f in self._wing_forms()) or p.key in WING_READS:
                return why
        elif owner is not None and (owner is dp.af or owner is dp.ep):
            if any(form is f for f in self._surface_forms(owner)):
                return why
        if p.kind == "action" and p.key in LAUNCH_KEYS:
            return why
        return None

    def _forms(self) -> list:
        dp, out = self.dp, [self.mp.params]
        for m in (dp.af, dp.ep):
            out += self._surface_forms(m)
        return out + self._wing_forms()

    def _bind_locks(self) -> None:
        for f in self._forms():
            if isinstance(f, Form) and not (isinstance(f.locked, _Lock) and f.locked.shell is self):
                f.locked = _Lock(self, f)

    def hazard(self) -> bool:
        """A global hazard is refused while a run is live: True (and the
        warning toast) when it is."""
        if self.g.runs.busy:
            self.g.toast(HAZARD, "warning")
            return True
        return False

    def take_rank(self, i=None) -> None:
        """Take ranked row `i` (a click, or ENTER on the ranking): refused
        while a run of that surface -- or of the wing, which flies it -- is
        live."""
        g, dp = self.g, self.dp
        if i is not None:
            dp.rank_list.idx = int(i)
        m = dp.ep if dp.nav.current().startswith("ep.") else dp.af
        job = g.runs.live
        if job is not None and (job.owner is m or job.owner is dp.wing):
            g.toast(LOCK_TEXT.format(evaluator=job.info.evaluator), "warning")
            return
        dp.act()
        self._changed()

    # -- views ------------------------------------------------------------------------------------------
    def _view_form(self, key):
        """The Form view `key` binds its controls to (None on `.rank`, whose
        table is the ranking list, and on a wing view before there is a
        wing: WorkUI then keeps a private one in the view's state)."""
        dp = self.dp
        if key.startswith("m."):
            f = self.mp.params
        elif key.endswith(".rank") or dp.session is None:
            f = None
        elif key.startswith("r."):
            f = dp.results.params
        else:
            f = dp.rows() if dp.nav.current() == key else None
        return f if isinstance(f, Form) else None

    def _current_form(self):
        key = self.view
        f = self.safe(lambda: self._view_form(key), None, "form")
        if f is None:
            f = self.vstate.get(key, {}).get("_form")
        return f

    def _view_fn(self, key):
        mod_name, fn_name = VIEWS[key].split(":")
        mod = importlib.import_module(f".{mod_name}", __package__ or "drive")
        return getattr(mod, fn_name)

    def _ctx(self, key, state, rect) -> ViewCtx:
        g, dp, mp = self.g, self.dp, self.mp
        stage = STAGE_OF[key]
        model = {"m": mp, "af": dp.af, "ep": dp.ep, "w": dp.wing, "r": dp.results}[stage]
        job = g.runs.job_for(model) if stage != "m" and model is not None else None
        if stage == "r" and job is None and dp.wing is not None:
            job = g.runs.job_for(dp.wing)
        return ViewCtx(g=g, dp=dp, mp=mp, key=key, stage=stage, view=key, model=model, job=job,
                       live=job is not None, runs=g.runs, notices=g.notices, state=state,
                       select=self.select, width=rect.w - 2 * T.WORK_PAD_X,
                       size=g.screen.get_size(), consts=self.consts, shell=self)

    def _layout_scroll(self, key, form, target) -> None:
        """Lay view `key` out once without drawing and scroll so `target`'s
        row lies inside the work area with an 8 px margin."""
        L = self._L
        state = self.vstate.setdefault(key, {})
        ui_ = W.WorkUI(self.g.screen, L.work, form, state, scroll=self.scroll.get(key, 0),
                       kbd_focus=False, mouse=(-1, -1), overlay=self.overlay, now=time.monotonic(),
                       layout_only=True)
        try:
            self._view_fn(key)(ui_, self._ctx(key, state, L.work))
        except Exception:                                   # noqa: BLE001 -- the draw reports it
            pass
        h = ui_.end()
        s = ui_.scroll_for(target)
        if s is not None:
            self.scroll[key] = C.clamp_scroll(s, h, L.work.h)
        self.content_h[key] = h
        form.reveal = None

    def reveal_now(self, key: str) -> None:
        """Scroll the view on screen so control `key` is visible, now (what
        a keyboard move does on the next frame; tests call this)."""
        view = self.view
        form = self._current_form()
        if form is not None:
            self._layout_scroll(view, form, key)

    def _draw_view(self, surf, rect, now) -> None:
        g, dp = self.g, self.dp
        key = self.view
        state = self.vstate.setdefault(key, {})
        if key.endswith(".rank"):
            self.safe(dp.refresh_rank, None, "ranking")        # the pinned items, every draw
        form = self.safe(lambda: self._view_form(key), None, "form")
        if form is not None and form.reveal:
            self._layout_scroll(key, form, form.reveal)
        scroll = C.clamp_scroll(self.scroll.get(key, 0), self.content_h.get(key, rect.h), rect.h)
        ui_ = W.WorkUI(surf, rect, form, state, scroll=scroll,
                       kbd_focus=(self.region == "work" and self.kbd), mouse=self.mouse,
                       overlay=self.overlay, now=now)
        try:
            self._view_fn(key)(ui_, self._ctx(key, state, rect))
            h = ui_.end()
            self.view_errors.pop(key, None)
        except Exception as exc:                           # noqa: BLE001 -- isolation, PLAN §2.9.6
            ui_.end()
            tb = traceback.format_exc()
            self.view_errors[key] = tb
            tag = (key, type(exc).__name__, str(exc))
            label = VIEW_META[key][0]
            if tag not in self._logged:
                self._logged.add(tag)
                g.log(f"{label} — render error: {type(exc).__name__}: {exc}", "error")
            surf.fill(T.WELL, rect)
            err = W.WorkUI(surf, rect, None, self._err_state.setdefault(key, {}), scroll=0,
                           kbd_focus=False, mouse=self.mouse, overlay=self.overlay, now=now)
            with err.card(f"{label} — render error"):
                err.hint(RENDER_ERROR_HINT, "bad", split=False)
                err.code(tb.rstrip().splitlines()[-14:])
            h = err.end()
            scroll = 0
        self.content_h[key] = h
        self.scroll[key] = C.clamp_scroll(scroll, h, rect.h)
        self._bars["work"] = C.scrollbar(surf, rect, h, self.scroll[key], mouse=self.mouse)

    # -- the tree -----------------------------------------------------------------------------------------
    def _tc_of(self, name):
        spec = self.g.lib.airfoils.get(name)
        if spec is None:
            return None
        k = (name, id(spec))
        if k not in self._tc:
            if len(self._tc) > 256:
                self._tc.clear()
            self._tc[k] = float(spec.geometry()["tc"])
        return self._tc[k]

    def _slot_wing(self):
        """The library wing in the selected slot (None for an empty slot)."""
        g = self.g
        return g.lib.wings.get(g.build.slot(g.sel).wing)

    def _slot_session(self):
        """The session of the slot the car page has selected -- the one the
        mission would open -- made on first use (the Mission page's Design
        point rows made it already)."""
        return self.dp.session_for(self.g.sel)

    @staticmethod
    def own_section(session, stage: str) -> str:
        """The family's own section of a surface, the one it flies until a
        section is chosen: AeroBO's NACA 24tt at the problem's t/c on the
        wing, NACA 00tt at the searched t/c on the plates."""
        if stage == "ep":
            return "NACA 00tt"
        tc = _finite(getattr(session.wing.built().problem, "tc", None))
        return f"NACA 24{int(round(100.0 * tc)):02d}" if tc is not None else "the family's section"

    @staticmethod
    def searched_dim(wing) -> tuple:
        """(rows searched, rows in the box): AeroBO's box minus what is fixed."""
        n = len(wing.family_box())
        return n - len([k for k in wing.pinned() if k in wing.family_box()]), n

    def _chip(self, stage: str) -> str:
        """What a stage HOLDS (PLAN §2.9.3), drawn on locked stages too: the
        section each surface flies -- the one chosen there, or the family's
        own -- the wing's searched dimension and objective, the run's best.
        Before the design stages open, the library wing in the slot."""
        g, dp, mp = self.g, self.dp, self.mp
        if stage == "m":
            if not mp.flies():
                return "invalid"
            return " · ".join(w for w in mp.job_words() if w)
        if not self.design_live():
            w = self._slot_wing()
            if stage == "af":
                if w is None:
                    return "—"
                tc = self._tc_of(w.airfoil)
                return w.airfoil + (f" · t/c {tc:.3f}" if tc is not None else "")
            if stage == "ep":
                return f"{(w.plate_airfoil if w is not None else '') or 'flat'} (its own default)"
            if stage == "w":
                return f"{self.searched_dim(self._slot_session().wing)[0]}-D"
            return ""
        s = dp.session
        if stage in ("af", "ep"):
            if stage == "ep" and not s.wing.choices["plates"]:
                #  pylons carry the wing: its plates are a tip device (a
                #  fence with the wing's own section), no 2.8 to design
                return "pylons: a tip device"
            m = s.af if stage == "af" else s.ep
            c = m.chosen if m.decision in ("library", "optimised") else None
            if c:
                tc = _finite(c.get("tc"))
                return str(c.get("name")) + (f" · t/c {tc:.3f}" if tc is not None else "")
            own = self.own_section(s, stage)
            if m.decision == "default":
                return f"{own} (the family's own, kept)"
            if m.ranked:
                return f"{len(m.ranked)} ranked · flies {own}"
            return f"{own} (the family's own)"
        if stage == "w":
            return f"{self.searched_dim(s.wing)[0]}-D · {_objective_word(s.wing)}"
        if stage == "r":
            rec = s.wing.record
            if not rec:
                return ""
            lap = _finite((rec.get("breakdown") or {}).get("lap_time_s"))
            if s.wing.choices["objective"] == "laptime" and lap is not None:
                return f"lap {lap:.3f} s"
            return score_text(rec.get("best_score"), rec.get("score_units"))
        return ""

    def _build_rows(self, body_w=None) -> list:
        """The tree as flat rows (chrome.TreeDraw's TreeRow dicts): each
        stage, and its views when it is expanded."""
        exp = self._expanded()
        cur_stage, cur_view = self.stage, self.view
        rows = []
        text_x = T.TREE_BASE + T.TWISTY_W + T.TREE_GAP + T.ICON_PX + T.TREE_GAP
        for key, label, _crumb, views in STAGES:
            st, why = self.stage_state(key)
            icon, colour = GLYPH[st]
            opened = bool(exp.get(key))
            lab = " ".join(label.split())
            chip = self.safe(lambda k=key: self._chip(k), "—", f"chip {key}")
            if body_w is not None and chip and " · t/c " in chip:
                room = body_w - text_x - T.text_w(lab, "sans", 12) - _px(8) - _px(6)
                if T.text_w(chip, "mono", 9.5) > room:          # t/c goes before the name does
                    chip = chip.split(" · t/c ")[0]
            rows.append(dict(id=key, level=0, label=lab, icon=icon, icon_colour=getattr(T, colour),
                             chip=chip, selected=(key == cur_stage and not opened),
                             faint=(st == "locked"), twisty="open" if opened else "closed", tip=why))
            if opened:
                for v in views:
                    vl, vi = VIEW_META[v]
                    rows.append(dict(id=v, level=1, label=vl, icon=vi,
                                     icon_colour=T.INK_FAINT if st == "locked" else T.INK_MUTED,
                                     chip=None, selected=(key == cur_stage and v == cur_view),
                                     faint=False, twisty=None, tip=""))
        return rows

    def _selected_row(self, rows) -> str:
        for r in rows:
            if r["selected"]:
                return r["id"]
        return self.stage

    def _cursor(self, rows) -> str:
        ids = [r["id"] for r in rows]
        return self.tree_cursor if self.tree_cursor in ids else self._selected_row(rows)

    def _draw_tree(self, surf, body) -> None:
        rows = self._build_rows(body.w)
        self._rows = rows
        nav = self.dp.nav
        stage = getattr(nav, "reveal_stage", None) or self._reveal
        if getattr(nav, "reveal_stage", None) is not None:
            nav.reveal_stage = None
        self._reveal = None
        if stage in STAGE_VIEWS:
            views = STAGE_VIEWS[stage] if self._expanded().get(stage) else ()
            last = views[-1] if views else stage
            self.tree_scroll = self.treedraw.reveal(rows, stage, last, body.h, self.tree_scroll)
        self.tree_scroll = C.clamp_scroll(self.tree_scroll, C.TreeDraw.content_h(rows), body.h)
        if self.region == "tree" and self.kbd:
            cur = self._cursor(rows)
            for r in rows:
                r["focus"] = r["id"] == cur
        hits = self.treedraw.draw(surf, body, rows, self.tree_scroll, self.mouse)
        self._tree_hits = hits
        nav._rect = pygame.Rect(body)
        steps = set(nav.keys)
        nav._hits = [(rid, r.y, r.h) for rid, r, kind in hits
                     if kind == "row" and rid in steps and r.h >= T.TREE_ROW_H]
        if hasattr(nav, "_node_hits"):
            nav._node_hits = [(rid, pygame.Rect(r), kind) for rid, r, kind in hits
                              if kind == "twisty" or rid not in steps]

    def _tree_click(self, rid) -> None:
        if rid in STAGE_VIEWS:
            self.select(rid)
        elif rid in STAGE_OF:
            self.select(STAGE_OF[rid], rid)
        self.tree_cursor = rid

    def _tree_walk(self, d: int) -> None:
        """UP / DOWN in the tree: the next VISIBLE row that can be selected
        (a locked stage and its views are stepped over) is selected -- a view
        row shows that view, a stage row its remembered one."""
        rows = self._build_rows()
        ids = [r["id"] for r in rows]
        cur = self._cursor(rows)
        j = ids.index(cur) if cur in ids else 0
        while True:
            j += d
            if not 0 <= j < len(ids):
                return
            rid = ids[j]
            stage = rid if rid in STAGE_VIEWS else STAGE_OF.get(rid)
            if stage != "m" and self.safe(lambda s=stage: self._base(s), ("locked", ""))[0] == "locked":
                continue
            if rid in STAGE_VIEWS:
                self.select(rid)
            else:
                self.select(stage, rid)
            self.tree_cursor = rid
            return

    def _tree_fold(self, d: int) -> None:
        """LEFT / RIGHT in the tree on the mission page: fold or unfold the
        stage the cursor is on."""
        rows = self._build_rows()
        cur = self._cursor(rows)
        stage = cur if cur in STAGE_VIEWS else STAGE_OF.get(cur)
        if stage is not None:
            self._expanded()[stage] = d > 0
            self.tree_cursor = stage

    # -- properties (PLAN §3.1.3) --------------------------------------------------------------------------
    def _properties(self) -> list:
        stage = self.stage
        live = self.design_live()
        rows = self.safe(self._props_mission, [("group", "Mission")], "properties: mission")
        if stage in ("af", "w", "r"):
            rows = rows + self.safe(self._props_section, [], "properties: section")
        if stage in ("ep", "w", "r"):
            rows = rows + self.safe(self._props_endplate, [], "properties: endplate")
        if stage in ("w", "r") and live:
            rows = rows + self.safe(self._props_wing, [], "properties: wing")
        if stage == "r" and live and self.dp.wing.record:
            rows = rows + self.safe(self._props_result, [], "properties: result")
        return rows

    def _props_mission(self) -> list:
        g, mp, m = self.g, self.mp, self.g.mission
        r, job = mp.result, mp.job()
        rows = [("group", "Mission")]
        if not mp.flies():
            return rows + [("medium", " ".join(w for w in mp.job_words() if w)),
                           ("state", mp.err or "the lap did not close", T.BAD)]
        slot = self.consts["SLOT_LABEL"][g.sel].lower()
        if g.build.mirror and g.sel in ("left", "right"):
            slot += " (mirrored)"
        op = self._slot_session().op
        if job == _ms.STOPPING:
            rows += [("job", f"stopping from {m.v_stop_kmh:.0f} km/h"),
                     ("surface", f"{m.surface} (grip × {m.mu_scale:.2f})"),
                     ("slot", slot),
                     ("stop as it stands", f"{r.distance:.2f} m"),
                     ("vs no wings", f"{mp._delta():+.2f} m")]
        else:
            pf = mp.profile()
            rows += [("circuit", f"{job} ({pf.length:.0f} m, {len(pf.corners)} corners)"),
                     ("surface", f"{m.surface} (grip × {m.mu_scale:.2f})"),
                     ("slot", slot),
                     ("lap as it stands", f"{r.time:.3f} s"),
                     ("vs no wings", f"{mp._delta():+.3f} s")]
        rows += [
                 ("design speed", f"{float(op.V):.1f} m/s" + (" (typed)" if op.V_source == "typed" else "")),
                 ("state", "stated", T.GOOD) if m.stated else ("state", "not stated", T.WARN)]
        return rows

    def _props_surface(self, stage: str, title: str) -> list:
        """One section surface as Properties shows it: what it flies (the
        section chosen there, or the family's own), where that came from,
        the design point it was screened and designed at, the screen and
        the shape search so far, and what the wing run is sent."""
        g, dp = self.g, self.dp
        rows = [("group", title)]
        if not self.design_live():
            w = self._slot_wing()
            if stage == "ep":
                return rows + [("plates fly", (w.plate_airfoil if w is not None else "") or "flat")]
            if w is None:
                return rows + [("on the wing", "—")]
            spec = g.lib.airfoils.get(w.airfoil)
            tc = self._tc_of(w.airfoil)
            return rows + [("on the wing", w.airfoil),
                           ("from", (spec.origin if spec is not None else "") or "library"),
                           ("t/c", f"{tc:.4f}" if tc is not None else "—")]
        s = dp.session
        if stage == "ep" and not s.wing.choices["plates"]:
            return rows + [("plates", "a tip device on the pylons — the wing's own section",
                            T.INK_FAINT)]
        m = s.af if stage == "af" else s.ep
        c = m.chosen if m.decision in ("library", "optimised") else None
        own = self.own_section(s, stage)
        if c:
            rows += [("flies", str(c.get("name")), T.GOOD),
                     ("from", str(c.get("origin") or c.get("source")))]
            tc = _finite(c.get("tc"))
            if tc is not None:
                rows.append(("t/c", f"{tc:.4f}"))
        elif m.decision == "default":
            rows += [("flies", f"{own} (the family's own)", T.GOOD), ("from", "kept — WingLab's default")]
        else:
            rows += [("flies", f"{own} (the family's own)"), ("decision", "not decided", T.WARN)]
        cond = m.conditions()
        if cond:
            rows.append(("design point", f"Re {float(cond['re']):.3g} · cl {float(cond['cl_design']):.2f}"))
            rows.append(("Re from", "the cached library point" if m.effective_re_source() == "library"
                         else "this surface's own chord and speed"))
        rep = m.screen.get("report") or {}
        if m.ranked:
            rows.append(("screen", f"{len(m.ranked)} ranked of {rep.get('n_eligible', '?')} eligible"))
        tag = self._tag_of(m.opt.get("outcome"))
        if tag is not None:
            rows.append(("shape search", tag[0], getattr(T, tag[1])))
        if stage == "ep":
            rows.append(("symmetric", "yes — the plate screens and designs symmetric sections only"))
            pin = s.wing.auto_pins().get("endplate_tc")
            if pin is not None:
                rows.append(("t/c row", f"fixed from 2.8 at {float(pin):.4f}"))
        flag = m.flag_value()
        rows.append(("the wing is sent", "nothing — it flies its own" if flag is None
                     else ("section_name_plate" if stage == "ep" else "section_name")))
        return rows

    def _props_section(self) -> list:
        return self._props_surface("af", "Section")

    def _props_endplate(self) -> list:
        return self._props_surface("ep", "Endplate")

    @staticmethod
    def _tag_of(outcome):
        """A stored outcome's terminal tag (text, theme colour name, note)."""
        return dj.terminal_tag(outcome) if outcome else None

    @staticmethod
    def base_family(name: str) -> str:
        """AeroBO's family inside a carsim slot family's registered name:
        "carsim top · car rear wing + endplates #9e6d" -> "car rear wing + endplates"."""
        n = str(name or "")
        if " · " in n:
            n = n.split(" · ", 1)[1]
        return n.rsplit(" #", 1)[0]

    def _props_wing(self) -> list:
        s = self.dp.session
        w, op = s.wing, s.op
        dim, n = self.searched_dim(w)
        eff = w.search()
        ground = ("ON, over the deck at {:.2f} m".format(float(op.deck)) if w.role == "top"
                  and op.deck is not None else "OFF (the image plane 200 m away)")
        rows = [("group", "Wing"),
                ("problem", self.base_family(w.family_name)),
                ("ground effect", ground),
                ("objective", w.objectives().get(w.choices["objective"], w.choices["objective"]))]
        cap, floor = w.choices.get("drag_budget_n"), w.choices.get("downforce_min_n")
        rows.append(("drag ceiling", f"{float(cap):.0f} N" if cap else "none"))
        rows.append(("force floor", f"{float(floor):.0f} N" if floor else "none"))
        rows += [("searched", f"{dim} of {n} rows"),
                 ("optimiser", str(eff.get("optimiser"))),
                 ("budget", f"{int(eff.get('budget') or 0)} evaluations"
                  + (" (WingLab's plan)" if eff.get("source") == "recommended" else " (yours)")),
                 ("seed", str(int(w.seed))),
                 ("design speed", f"{float(op.V):.1f} m/s")]
        if w.spec is not None:
            rows.append(("wing", w.spec.name + (" *" if w.dirty else ""),
                         T.WARN if w.dirty else None))
        return rows

    def _props_result(self) -> list:
        s = self.dp.session
        sm = s.results.summary()
        rows = [("group", "Result")]
        tag = sm.get("tag")
        if tag:
            rows.append(("state", tag[0], getattr(T, tag[1])))
        rows.append(("best", score_text(sm.get("best_score"), sm.get("score_units"))))
        rows.append(("feasible", f"{sm.get('n_feasible', '—')} of {sm.get('n_evals', '—')}",
                     None if sm.get("feasible") else T.BAD))
        if _finite(sm.get("force_N")) is not None:
            rows.append((sm.get("force_word", "force"), f"{float(sm['force_N']):.1f} N"))
        if _finite(sm.get("drag_N")) is not None:
            rows.append(("drag", f"{float(sm['drag_N']):.2f} N"))
        if _finite(sm.get("lap_time_s")) is not None:
            rows.append(("lap (WingLab)", f"{float(sm['lap_time_s']):.3f} s"))
        if _finite(sm.get("wall_s")) is not None:
            rows.append(("wall time", f"{float(sm['wall_s']):.1f} s"))
        law = sm.get("law")
        rows.append(("car's law", "derived from WingLab's evaluator" if law else "not derived yet",
                     None if law else T.WARN))
        return rows

    # -- the tool bar, the status bar -------------------------------------------------------------------
    def _chips(self) -> list:
        g = self.g
        out = [(self.consts["SLOT_LABEL"][g.sel], T.ACCENT)]
        out += [(w.upper(), T.INK_MUTED) for w in self.mp.job_words() if w]
        role = self.consts["SLOT_ROLE"][g.sel]
        out.append(("GROUND EFFECT" if role == "top" else "NO GROUND EFFECT", T.INK_MUTED))
        out.append(("CONSTRAINED", T.WARN))
        busy = getattr(g.lib, "xfoil_busy", "")
        if busy:
            out.append((f"XFOIL · {busy}", T.WARN))
        return out

    def _cells_text(self) -> list:
        """AeroBO's three status cells: the problem, its searched dimension
        and the budget the wing run would fly -- the slot's session's, open
        or not."""
        s = self.dp.session if self.design_live() else self._slot_session()
        w = s.wing
        eff = w.search()
        head = [f"problem {self.base_family(w.family_name)}", f"dim {self.searched_dim(w)[0]}",
                f"budget {int(eff.get('budget') or 0)}"]
        return head + [("F1 keys", T.INK_FAINT)]

    def _tool_enabled(self, tid) -> bool:
        busy = self.g.runs.busy
        if tid == "stop":
            return busy
        if tid == "restart":
            return not busy
        if tid == "save":
            return not busy and self.dp.wing is not None
        return True

    def _menu_enabled(self, action) -> bool:
        """Is menu item `action` live? (Read by the menu bar while it draws,
        so it goes through `safe`: a menu never takes the frame down.)"""
        return bool(self.safe(lambda: self._menu_on(action), False, "menu"))

    def _menu_on(self, action) -> bool:
        g, dp = self.g, self.dp
        busy = g.runs.busy
        if busy and action in _MENU_HAZARDS:
            return False
        if action in ("save", "rename", "snip"):
            return dp.wing is not None and self.design_live()
        if action == "stop":
            return busy
        if action == "boxreset":
            f = dp.wing.box_params() if dp.wing is not None else None
            p = self._param(f, "boxreset")
            return p is not None and f.is_enabled(p)
        if action == "flat":
            f = dp.ep.screen_params if dp.ep is not None else None
            p = self._param(f, "decline")
            return (self.design_live() and p is not None and f.is_enabled(p)
                    and bool(dp.wing.choices["plates"]))
        if action == "rec":
            m = self._section_model()
            p = self._param(m.screen_params, "rec") if m is not None else None
            return p is not None and m.screen_params.is_enabled(p)
        return True

    @staticmethod
    def _param(form, key):
        return form.param(key) if isinstance(form, Form) else None

    def _section_model(self):
        s = self.stage
        return self.dp.af if s == "af" else (self.dp.ep if s == "ep" else None)

    # -- drawing ----------------------------------------------------------------------------------------
    def _drain(self) -> None:
        g = self.g
        if not self._started:
            self._started = True
            self.log.append(time.strftime("%H:%M:%S"), SESSION_LINE, "info")
        for stamp, text, level in g.notices.drain_log():
            self.log.append(stamp, text, level)
        for text, kind in g.notices.drain_toasts():
            self.toasts.push(text, kind)

    def draw(self, surf) -> None:
        """One frame of the shell, drawn from the models as they stand."""
        g = self.g
        now = time.monotonic()
        W_, H_ = surf.get_size()
        L = self._L = layout(W_, H_)
        self._bind_locks()
        if g.page == "section":
            self.remembered[self.stage] = self.view
        self._drain()
        surf.fill(T.CANVAS)
        self.menubar.draw(surf, L.menu, self.mouse, self._menu_enabled)
        self.toolbar.draw(surf, L.tool, enabled=self._tool_enabled, crumbs=CRUMBS, crumb_cur=self.stage,
                          chips=self.safe(self._chips, [], "chips"), mouse=self.mouse,
                          hidden=("save",) if g.page == "mission" else ())
        self._draw_tree(surf, C.pane(surf, L.tree, "Simulation"))
        pbody = C.pane(surf, L.props, "Properties")
        prow = self._properties()
        self.props_scroll = C.clamp_scroll(self.props_scroll, C.PropertyGrid.content_h(prow), pbody.h)
        self.propgrid.draw(surf, pbody, prow, self.props_scroll, self.mouse)
        C.pane(surf, L.main, None, white=True)
        views = views_of(self.stage)
        self.tabstrip.draw(surf, L.tabs, [(v, VIEW_META[v][0]) for v in views], self.view,
                           focus=(self.region == "tabs" and self.kbd), mouse=self.mouse)
        self._draw_view(surf, L.work, now)
        obody = C.pane(surf, L.output, "Output", white=True, tool="clear_all", mouse=self.mouse)
        self.log.draw(surf, obody)
        text, kind, prog = g.runs.status
        self._cells = self.statusbar.draw(surf, L.status, text, kind, prog,
                                          self.safe(self._cells_text, [], "status cells"), self.mouse)
        self.overlay.draw(surf, now)
        self.menubar.draw_menu(surf, self.mouse)
        if self.dialog is not None and self.dialog.open:
            if self.dialog._content_h is None:
                #  a dialog sizes itself to its body, which it only knows once
                #  laid out: measure it offscreen, so it opens at its size
                self.dialog.draw(pygame.Surface((W_, H_)), W_, H_, mouse=(-1, -1), now=now)
            self.dialog.draw(surf, W_, H_, mouse=self.mouse, now=now)
        self.toasts.draw(surf, W_, H_, now)

    # -- input --------------------------------------------------------------------------------------------
    def handle(self, ev) -> None:
        """Every event on the shell pages (after the garage's prompt, pause
        menu and tutor box). Stating the mission -- by Run, ENTER, the
        button or the pad -- is announced with AeroBO's toast here, once,
        whatever path stated it."""
        g, dp = self.g, self.dp
        before = (g.mission.stated, dp.opened_for)
        self._bind_locks()
        self._event(ev)
        if g.mission.stated and g.page == "section" and (not before[0] or dp.opened_for != before[1]):
            g.toast("Mission stated", "positive")
        return None

    def _event(self, ev) -> None:
        t = ev.type
        if t in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            self.mouse = tuple(ev.pos)
        if self.dialog is not None:
            if self.dialog.open:
                self.dialog.handle(ev)
                if not self.dialog.open:
                    self.dialog = None
                return
            self.dialog = None
        if self.menubar.open is not None:
            if t == pygame.KEYDOWN:
                act = self.menubar.key(ev)
                if act:
                    self._menu(act)
                return
            if t == pygame.MOUSEBUTTONDOWN:
                if getattr(ev, "button", 0) == 1:
                    act = self.menubar.click(ev.pos)
                    if act:
                        self._menu(act)
                return
        if self.overlay.handle(ev):
            pick = self.overlay.last_pick
            if pick is not None:
                self.overlay.last_pick = None
                self._changed()
            return
        if t == pygame.KEYDOWN:
            self._keydown(ev)
        elif t == pygame.MOUSEBUTTONDOWN:
            self._mouse_down(ev)
        elif t == pygame.MOUSEBUTTONUP:
            self._mouse_up(ev)
        elif t == pygame.MOUSEMOTION:
            self._mouse_motion(ev)
        elif t == pygame.MOUSEWHEEL:
            self._wheel(ev)

    def pad(self, edges, lx=0.0, ly=0.0, fine=False) -> None:
        """The pad on the shell pages, as the keys it stands for: d-pad (and
        the left stick, folded into the edges by the garage) = arrows, CROSS
        = ENTER, CIRCLE = ESC (stop / back), SQUARE = Run, TRIANGLE = the
        next region, R1 = the next tab; L1 held = fine. OPTIONS is the
        garage's pause menu."""
        mod = pygame.KMOD_SHIFT if fine else 0
        for name, key in _PAD_KEYS:
            if edges.get(name):
                self.handle(pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod, unicode="",
                                               scancode=0))

    # -- keys --------------------------------------------------------------------------------------------
    def _keydown(self, ev) -> None:
        g, dp = self.g, self.dp
        k = ev.key
        mods = getattr(ev, "mod", 0)
        shift = bool(mods & pygame.KMOD_SHIFT)
        ch = getattr(ev, "unicode", "") or ""
        self.kbd = True
        form = self._current_form()
        if form is not None and form.editing is not None:
            if form.key_input(ev) and form.editing is None and k in (pygame.K_RETURN, pygame.K_KP_ENTER,
                                                                         pygame.K_TAB):
                self._changed()
            return
        if k == pygame.K_ESCAPE:
            self._escape()
            return
        if k == pygame.K_F1 or ch == "?":
            if self.region == "work" and form is not None and form.open_help():
                return
            self.open_keys()
            return
        if k == pygame.K_F5:
            self.run()
            return
        if k == pygame.K_TAB:
            order = ("tree", "tabs", "work")
            self.region = order[(order.index(self.region) + (-1 if shift else 1)) % 3]
            return
        if k in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
            self._tab(-1 if k == pygame.K_LEFTBRACKET else 1)
            return
        if k in (pygame.K_PAGEUP, pygame.K_PAGEDOWN, pygame.K_HOME, pygame.K_END):
            self._page_key(k, form)
            return
        if k in DESIGN_KEYS and not mods & (pygame.KMOD_CTRL | pygame.KMOD_META | pygame.KMOD_ALT):
            if g.page == "section":
                if not self.hazard():
                    g._design_shortcut(k)
            return
        region = self.region
        if region == "work":
            self._work_key(ev, k, shift, form)
        elif region == "tabs":
            if k in (pygame.K_LEFT, pygame.K_RIGHT):
                self._tab(-1 if k == pygame.K_LEFT else 1)
            elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._enter_step()
        else:
            if k in (pygame.K_UP, pygame.K_DOWN):
                self._tree_walk(-1 if k == pygame.K_UP else 1)
            elif k in (pygame.K_LEFT, pygame.K_RIGHT):
                d = -1 if k == pygame.K_LEFT else 1
                if g.page == "section":
                    dp.nav.nav_group(d)
                    self.tree_cursor = None
                else:
                    self._tree_fold(d)
            elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._enter_step()

    def _enter_step(self) -> None:
        """ENTER with the tree or the tabs focused: the step itself -- state
        the mission, or the design page's ENTER (screen, take, optimise,
        launch, keep going)."""
        if self.g.page == "mission":
            self.g.state_mission()
            return
        self.dp.act()
        self._changed()

    def _work_key(self, ev, k, shift, form) -> None:
        g, dp = self.g, self.dp
        rank = self.view.endswith(".rank")
        if form is not None and not rank and form.key_input(ev):     # a digit starts typing
            return
        if k in (pygame.K_UP, pygame.K_DOWN):
            d = -1 if k == pygame.K_UP else 1
            if rank:
                dp.rank_list.nav(d)
            elif form is not None:
                form.nav(d)
            return
        if k in (pygame.K_LEFT, pygame.K_RIGHT):
            if form is None or rank:
                return
            if form.adjust(-1 if k == pygame.K_LEFT else 1, shift):
                self._changed()
            else:
                self._refused(form, form.current())
            return
        if k in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if rank:
                self.take_rank()
                return
            if form is None:
                return
            p = form.current()
            if g.page == "mission" and (p is None or p.kind != "action"):
                g.state_mission()                 # pinned: ENTER states the mission
                return
            if form.activate():
                self._changed()
            else:
                self._refused(form, p)

    def _refused(self, form, p) -> None:
        why = form.lock_reason(p) if p is not None else None
        if why:
            self.g.toast(why, "warning")

    def _page_key(self, k, form) -> None:
        if k in (pygame.K_PAGEUP, pygame.K_PAGEDOWN) and self.region == "work" and form is not None:
            if form.page(-1 if k == pygame.K_PAGEUP else 1):
                return
        key, L = self.view, self._L
        h = self.content_h.get(key, 0)
        s = self.scroll.get(key, 0)
        step = L.work.h - _px(40)
        s = {pygame.K_PAGEUP: s - step, pygame.K_PAGEDOWN: s + step,
             pygame.K_HOME: 0, pygame.K_END: h}[k]
        self.scroll[key] = C.clamp_scroll(s, h, L.work.h)

    def _tab(self, d: int) -> None:
        views = views_of(self.stage)
        if self.view not in views:
            return
        i = max(0, min(len(views) - 1, views.index(self.view) + d))
        if views[i] != self.view:
            self.select(self.stage, views[i])

    def _escape(self) -> None:
        """ESC / CIRCLE: a live run is stopped (a second press while it
        stops only says so); otherwise back, as it always was."""
        g = self.g
        job = g.runs.live
        if job is not None:
            if job.state == "running":
                g.runs.stop()
                return
            if job.state == "stopping":
                g.toast("stopping after the current evaluation", "info")
                return
        g.close_page()

    # -- the tool bar's verbs ----------------------------------------------------------------------------
    def run(self) -> None:
        """Run the current stage (AeroBO RUN_ACTIONS): state the mission;
        screen the surface's library, or optimise its shape on the Shape
        optimisation view; launch the wing (then its Convergence); on
        Results, AeroBO's design report of the winner if it is missing.
        Every run starts LIVE on the engine's worker thread; a refusal (a
        run already going, no XFOIL) is the model's own toast."""
        g, dp = self.g, self.dp
        stage = self.stage
        if stage == "m":
            g.state_mission()
            return
        if dp.session is None:
            g.toast(NOT_LIVE, "info")
            return
        if stage in ("af", "ep"):
            m = dp.af if stage == "af" else dp.ep
            if self.view.endswith(".opt"):
                m.start_optimise()
            else:
                m.start_screen()
            dp.msg = m.msg
        elif stage == "w":
            dp.run_wing()
        elif stage == "r":
            res = dp.results
            if not dp.wing.record:
                g.toast("no completed run yet — run the wing on 3 Wing ▸ Solver", "info")
            elif res.report is None and not g.runs.busy:
                res.start_report()
            elif g.runs.busy:
                g.toast(HAZARD, "warning")
            else:
                g.toast("the results are this run's — Keep going (K) buys more evaluations", "info")

    def stop(self) -> None:
        if not self.g.runs.stop():
            self.g.toast("nothing is running", "info")

    def _step(self, d: int) -> None:
        i = STAGE_KEYS.index(self.stage)
        j = max(0, min(len(STAGE_KEYS) - 1, i + d))
        if j != i:
            self.select(STAGE_KEYS[j])

    def restart(self) -> None:
        """Start the design over: the slot's screens, sections, runs and
        results forgotten (`DesignSession.invalidate`), the mission kept."""
        if self.hazard():
            return
        s = self.dp.session
        if s is not None and self.design_live():
            s.invalidate("started over from the tool bar")
        self.g.open_section()
        self._expanded()["af"] = True

    def _tool(self, tid) -> None:
        g = self.g
        if tid.startswith("crumb:"):
            self.select(tid.split(":", 1)[1])
        elif tid == "run":
            self.run()
        elif tid == "stop":
            self.stop()
        elif tid == "prev":
            self._step(-1)
        elif tid == "next":
            self._step(1)
        elif tid == "restart":
            self.restart()
        elif tid == "save" and not self.hazard():
            g._design_shortcut(pygame.K_s)

    def _menu(self, action) -> None:
        g, dp = self.g, self.dp
        if action in _MENU_HAZARDS and self.hazard():
            return
        if action == "restart":
            self.restart()
        elif action in ("save", "rename", "airfoils", "keep"):
            key = {"save": pygame.K_s, "rename": pygame.K_n, "airfoils": pygame.K_a,
                   "keep": pygame.K_k}[action]
            if g.page == "section":
                g._design_shortcut(key)
            elif action == "airfoils":
                g.open_airfoils()
            else:
                g.toast(NOT_LIVE, "info")
        elif action == "back":
            self._escape()
        elif action == "run":
            self.run()
        elif action == "stop":
            self.stop()
        elif action == "boxreset":
            self._activate_param(dp.wing.box_params() if dp.wing is not None else None, "boxreset")
        elif action == "flat":
            self._activate_param(dp.ep.screen_params if dp.ep is not None else None, "decline")
        elif action == "rec":
            m = self._section_model()
            if m is None:
                g.toast("the weights belong to 2 Airfoil and 2.8 Endplate — select one first", "info")
            else:
                self._activate_param(m.screen_params, "rec")
        elif action == "snip" and dp.wing is not None:
            dp.wing.log_config()
        elif action == "clearlog":
            self.log.clear()
        elif action == "keys":
            self.open_keys()
        elif action == "about":
            self.open_about()

    def _activate_param(self, form, key) -> None:
        p = self._param(form, key)
        if p is None or not self.design_live():
            self.g.toast(NOT_LIVE, "info")
            return
        if form.is_enabled(p):
            p.activate()
            self._changed()
        else:
            self._refused(form, p)

    # -- the mouse ---------------------------------------------------------------------------------------
    def _mouse_down(self, ev) -> None:
        if getattr(ev, "button", 0) != 1:        # 2 / 3 do nothing; 4 / 5 are the wheel's echo
            return
        L, pos = self._L, ev.pos
        self.kbd = False
        if L.menu.collidepoint(pos):
            act = self.menubar.click(pos)
            if act:
                self._menu(act)
            return
        if L.tool.collidepoint(pos):
            tid = self.toolbar.click(pos)
            if tid:
                self._tool(tid)
            return
        if L.tree.collidepoint(pos):
            self.region = "tree"
            for rid, r, kind in self._tree_hits:
                if r.collidepoint(pos):
                    if kind == "twisty":
                        stage = rid if rid in STAGE_VIEWS else STAGE_OF.get(rid)
                        ex = self._expanded()
                        ex[stage] = not ex.get(stage)
                    else:
                        self._tree_click(rid)
                    return
            return
        if L.tabs.collidepoint(pos):
            key = self.tabstrip.click(pos)
            if key:
                self.region = "tabs"
                self.select(self.stage, key)
            return
        if L.work.collidepoint(pos):
            self.region = "work"
            bar = self._bars.get("work")
            if bar is not None and bar[0].collidepoint(pos):
                drag = C.ScrollDrag()
                key = self.view
                if drag.begin(pos, bar):
                    self._drag = ("work", drag)
                else:
                    self.scroll[key] = C.ScrollDrag.page(pos, bar, self.scroll.get(key, 0),
                                                         self.content_h.get(key, 0))
                return
            self._work_click(ev)
            return
        if L.output.collidepoint(pos):
            if C.pane_tool_rect(L.output).collidepoint(pos):
                self.log.clear()
                return
            drag = C.ScrollDrag()
            if drag.begin(pos, self.log.bar):
                self._drag = ("log", drag)
            return
        if L.status.collidepoint(pos):
            if len(self._cells) >= 4 and self._cells[3] is not None and self._cells[3].collidepoint(pos):
                self.open_keys()

    def _work_click(self, ev) -> None:
        form = self._current_form()
        if form is None:
            return
        shift = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT) if pygame.get_init() else False
        r = W.route_mouse(form, None, ev, mouse=self.mouse, fine=shift, shift=shift)
        if r in ("action", "adjust"):
            self._changed()
        elif r == "select":
            key = form.hit_key(ev.pos)
            if key is not None:
                self._refused(form, form.param(key))

    def _mouse_up(self, ev) -> None:
        if getattr(ev, "button", 0) != 1:
            return
        if self._drag is not None:
            self._drag[1].end()
            self._drag = None
            return
        form = self._current_form()
        if form is not None and form.dragging:
            form.release()
            self._changed()

    def _mouse_motion(self, ev) -> None:
        held = bool(getattr(ev, "buttons", (0,))[0])
        if self._drag is not None and held:
            pane, drag = self._drag
            if pane == "work" and self._bars.get("work") is not None:
                key = self.view
                self.scroll[key] = drag.drag(ev.pos, self._bars["work"], self.content_h.get(key, 0))
            elif pane == "log" and self.log.bar is not None:
                self.log.scroll = drag.drag(ev.pos, self.log.bar, self.log.content_h)
                self.log.follow = False
            return
        form = self._current_form()
        if form is not None and form.dragging and held:
            form.drag(ev.pos)

    def _wheel(self, ev) -> None:
        L, pos = self._L, self.mouse
        dy = int(getattr(ev, "y", 0))
        step = _px(WHEEL_PX)
        shift = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT) if pygame.get_init() else False
        if L.work.collidepoint(pos):
            form = self._current_form()
            if shift and form is not None:
                form.hscroll_at(pos, -dy if dy else int(getattr(ev, "x", 0)))
                return
            key = self.view
            self.scroll[key] = C.clamp_scroll(self.scroll.get(key, 0) - dy * step,
                                              self.content_h.get(key, 0), L.work.h)
        elif L.tree.collidepoint(pos):
            self.tree_scroll = max(0, self.tree_scroll - dy * step)
        elif L.props.collidepoint(pos):
            self.props_scroll = max(0, self.props_scroll - dy * step)
        elif L.output.collidepoint(pos):
            self.log.wheel(dy)

    # -- after a change ------------------------------------------------------------------------------------
    def _changed(self) -> None:
        """A control fired or moved: the gates may have moved with it (the
        garage's own after-rows hook, as the key handlers always called)."""
        self.g._after_rows_changed()

    def _commit_edit(self) -> None:
        f = self._current_form()
        if f is not None and f.editing is not None:
            f.commit()

    # -- dialogs (PLAN §2.9.10) ----------------------------------------------------------------------------
    def open_keys(self) -> None:
        pad_first = getattr(self.g, "pad", None) is not None

        def body(ui_):
            ui_.sect_head("Keyboard, pad and mouse on the design pages")
            for kb, pad, what in KEY_ROWS:
                keys = kb if pad is None else (f"{pad} / {kb}" if pad_first else f"{kb} / {pad}")
                ui_.kv(keys, what)
            ui_.kv("mouse", "click anything; the wheel scrolls; drag a slider")
        self.dialog = C.Dialog("Keys and controller", 620, body)

    def open_about(self) -> None:
        def body(ui_):
            ui_.sect_head("Four stages, in order, each one feeding the next.")
            for n, text in ABOUT_ROWS:
                c1, c2 = ui_.columns((1, 30), gap=6)
                with c1:
                    ui_.label(n, css=12, family="mono", colour=T.ACCENT)
                with c2:
                    ui_.hint(text, split=False)
            ui_.hairline()
            ui_.hint(ABOUT_CLOSE, split=False)
        self.dialog = C.Dialog("About this pipeline", 620, body)


# ==================================================================== #
#  SELF-CHECK                                                          #
# ==================================================================== #
def _only_keys(only) -> set | None:
    """`--only` tokens as view keys: a full key ("af.screen") or a stage
    prefix ending in a dot ("w.", "af.") for every view of that stage."""
    if not only:
        return None
    toks = [t.strip() for t in only if t and t.strip()]
    return {k for k in STAGE_OF if any(k == t or (t.endswith(".") and k.startswith(t)) for t in toks)}


def self_check(verbose: bool = True, only=None) -> bool:
    """The shell against PLAN §2.9 / §4.1 and PLAN2 §6-§8, on a headless
    garage and a temporary library, every engine run REPLAYED from the
    captured fixtures (`aerobo_models.use_fixtures`: the same EngineJob path,
    no XFOIL, deterministic -- a run is frozen "running" by the replay's
    `pause_at`, never by timing): geometry, the tree (expansion, reveal,
    glyphs, chips), AeroBO's gating, the page and selection model, Run /
    Stop, RUNNING -> DONE and Stop -> STOPPED, the live evaluation graph, the
    focus regions, keyboard reach below the fold, ESC during a run, the
    live-run lock, the hint routing, None-safety, every view in pre /
    running / post (the N11 loop), view isolation, the dialogs. The car's
    law and AeroBO's design report run for real (fast, no XFOIL).

    `only` (view keys or stage prefixes, `--only w.,r.`) limits the view
    rows to those views -- and every other view is drawn by the generic
    layout instead of its module, so a sibling's half-written views module
    is never imported (PLAN §1.4, the W3 gate)."""
    import os
    import shutil
    import tempfile
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    real = importlib.import_module(f"{__package__ or 'drive'}.design_shell")
    if real is not sys.modules.get(__name__):
        #  run as `python3 -m drive.design_shell` this file is `__main__`, a
        #  second copy of the module: the garage's shell reads the real one's
        #  VIEWS, so the check (which swaps entries of it) runs there
        return real.self_check(verbose, only)
    from . import garage as grg                          # the self-check only (PLAN §1.1)
    from . import aerobo_models as am
    from .aero.library import Library

    ok = True
    n_rows = 0

    def rep(tag, passed, msg=""):
        nonlocal ok, n_rows
        ok = ok and bool(passed)
        n_rows += 1
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}" + (f": {msg}" if msg else ""))

    only = _only_keys(only)
    keys = [k for k in STAGE_OF if only is None or k in only]
    saved_views = dict(VIEWS)
    if only is not None:
        for k in VIEWS:
            if k not in only:
                VIEWS[k] = "design_shell:generic_view"

    def mine(errs) -> dict:
        """The render errors of the views under test."""
        return {k: v for k, v in errs.items() if only is None or k in only}

    def runs_state() -> dict:
        out = {}
        for root, _dirs, files in os.walk("runs"):
            for f in files:
                p = os.path.join(root, f)
                try:
                    out[p] = os.path.getmtime(p)
                except OSError:
                    pass
        return out

    runs0 = runs_state()
    tmp = tempfile.mkdtemp(prefix="carsim_shell_")
    lib = Library(os.path.join(tmp, "library"), use_xfoil=False)
    outcomes: list = []                 # every stored outcome the check saw, for the last row
    try:
        pygame.init()
        with am.replaying() as rp:

            def new_garage(size=(1280, 800), wing="flank-e423"):
                b = grg.CarBuild()
                b.left.wing = wing
                b.sync_mirror("left")
                g_ = grg.Garage(size, b, headless=True, lib=lib)
                g_.aerobo_dir = None                      # nothing under runs/ (PLAN2 §9)
                g_.export_dir = os.path.join(tmp, "export")
                return g_

            # -- geometry (PLAN §4.1) --------------------------------------------------------------
            R = pygame.Rect
            want = {(1280, 800): dict(menu=R(0, 0, 1280, 26), tool=R(0, 26, 1280, 32),
                                      status=R(0, 777, 1280, 23), tree=R(2, 60, 276, 427),
                                      props=R(2, 489, 276, 286), main=R(280, 60, 998, 563),
                                      tabs=R(281, 61, 996, 25), work=R(281, 86, 996, 536),
                                      output=R(280, 625, 998, 150), tutor_box=R(750, 403, 520, 212)),
                    (1600, 1000): dict(menu=R(0, 0, 1600, 26), tool=R(0, 26, 1600, 32),
                                       status=R(0, 977, 1600, 23), tree=R(2, 60, 276, 547),
                                       props=R(2, 609, 276, 366), main=R(280, 60, 1318, 763),
                                       tabs=R(281, 61, 1316, 25), work=R(281, 86, 1316, 736),
                                       output=R(280, 825, 1318, 150), tutor_box=R(1070, 603, 520, 212))}
            T.set_scale(1.0)
            bad = [f"{size} {k}" for size, rects in want.items() for k, r in rects.items()
                   if getattr(layout(*size), k) != r]
            chrome_copy = C.shell_rects(1600, 1000)
            bad += [f"chrome {k}" for k, r in chrome_copy.items() if getattr(layout(1600, 1000), k) != r]
            rep("geometry at 1280x800 and 1600x1000 is PLAN §4.1 (WingLab's measured rects at 1600x1000)",
                not bad, "; ".join(bad) or "menu, tool, tree, props, main, tabs, work, output, status, tutor box")
            rep("the scale is 1.0 up to 1999 px wide, then quarter steps",
                [scale_for(*s) for s in ((1280, 800), (1600, 1000), (1999, 1249), (2560, 1600), (3200, 2000))]
                == [1.0, 1.0, 1.0, 1.5, 2.0], "")
            T.set_scale(1.5)
            big = layout(2560, 1600)
            T.set_scale(1.0)
            rep("set_scale reaches the layout: 2560x1600 at S 1.5",
                big.menu.h == 39 and big.tree.w == 414 and big.status.h == 35 and layout(1280, 800).menu.h == 26,
                f"menu {big.menu.h}, left column {big.tree.w}, status {big.status.h}")

            # -- a fresh garage ----------------------------------------------------------------------
            g = new_garage()
            sh, dp, mp = g.shell, g.design_page, g.mission_page
            g.open_mission("left")
            g._draw_page()
            ids = [r["id"] for r in sh._rows]
            rep("a fresh garage expands only 1 Mission",
                ids == ["m", "m.operating", "m.design", "m.search", "af", "ep", "w", "r"], " ".join(ids))
            chips = {r["id"]: r["chip"] for r in sh._rows if r["level"] == 0}
            tc = lib.airfoils["e423"].geometry()["tc"]
            rep("chips on a fresh garage (not live) name the slot wing's sections and the family's size",
                chips["af"] == f"e423 · t/c {tc:.3f}" and chips["ep"] == "flat (its own default)"
                and chips["w"] == "14-D" and chips["m"] == "arena · dry" and chips["r"] == "",
                str(chips))
            rep("before the mission: the design stages lock with its reason; Mission is the active stage",
                [sh.stage_state(k)[0] for k in STAGE_KEYS] == ["active", "locked", "locked", "locked", "locked"]
                and sh.stage_state("af")[1] == NOT_LIVE, str([sh.stage_state(k)[0] for k in STAGE_KEYS]))
            refused = not sh.select("af", "af.screen")
            toasts = g.notices.drain_toasts()
            rep("a locked stage refuses with its reason as an info toast; nothing moves",
                refused and g.page == "mission" and (NOT_LIVE, "info") in toasts, str(toasts[-1:]))
            cells = [c if isinstance(c, str) else c[0] for c in sh._cells_text()]
            rep("the status bar: Ready, and WingLab's three cells (its family, 14-D, its budget 53) + F1 keys",
                g.runs.status[0] == "Ready" and len(sh._cells) == 4
                and cells == ["problem car rear wing + endplates + free chord law", "dim 14", "budget 53",
                              "F1 keys"], str(cells))
            rep("the tool bar hides Save on the mission page", sh.tool_rect("save") is None and
                sh.tool_rect("run") is not None, str(sh.tool_rect("run")))
            e = g.shell.log.lines
            rep("the Output log opens on the session line", e and e[0][1] == SESSION_LINE,
                e[0][1][:60] if e else "")

            # -- stating the mission -------------------------------------------------------------------
            def key(k, mod=0, ch=""):
                return g._handle(pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod, unicode=ch, scancode=0))

            #  the N11 loop's book-keeping: which view drew in which state, the worst frame
            drawn: dict = {}
            worst = [0.0, ""]

            def draw_views(state, stage_keys, force=False):
                """Draw each view of `stage_keys` in `state`. A view of a
                locked stage (4 Results before a run) is shown by a FORCED
                navigator select when `force` -- its empty state is part of
                the contract too."""
                for k in stage_keys:
                    if k not in keys:
                        continue
                    if not sh.select(STAGE_OF[k], k):
                        g.notices.drain_toasts()
                        if not force:
                            drawn.setdefault(k, set()).add(state + " (refused)")
                            continue
                        dp.nav.select(k, force=True)
                        g.page = "section"
                    t0 = time.perf_counter()
                    g._draw_page()
                    ms = (time.perf_counter() - t0) * 1e3
                    if ms > worst[0]:
                        worst[:] = [ms, f"{k} {state}"]
                    if k not in sh.view_errors:
                        drawn.setdefault(k, set()).add(state)

            def park(pause):
                """Freeze the next run at `pause` evaluations (sections swept)."""
                rp.pause_at = pause

            def settle():
                """The live run parked at its pause (or ended); its events drained."""
                job = g.runs.live
                if job is not None and hasattr(job, "settle"):
                    job.settle(10.0)
                    g.runs.pump(0.0)
                return g.runs.live

            def finish(job):
                """Let a parked replay play on, and run everything to its end."""
                rp.pause_at = None
                if job is not None and getattr(job, "replay", None) is not None:
                    job.replay.release()
                g.runs.run_all()

            draw_views("pre", STAGE_VIEWS["m"])
            sh.select("m", "m.operating")
            key(pygame.K_F5)
            g._draw_page()
            toasts = [it["text"] for it in sh.toasts.items]
            rep("Run on the mission states it: the design stages open on 2 Airfoil, with WingLab's toast",
                g.page == "section" and dp.nav.current() == "af.screen" and g.mission.stated
                and "Mission stated" in toasts and sh.stage_state("m")[0] == "done",
                f"{g.page} {dp.nav.current()}")
            rep("select('af.screen', force) expands 2 Airfoil: _hits[2] is af.section (C5)",
                dp.nav.expanded["af"] and len(dp.nav._hits) > 2 and dp.nav._hits[2][0] == "af.section"
                and dp.nav._rect == pygame.Rect(3, 83, 274, 403), f"{dp.nav._hits[:3]} in {dp.nav._rect}")
            s = dp.session
            states = {k: sh._base(k) for k in STAGE_KEYS}
            rep("WingLab's gating (PLAN2 D10): after the mission 2, 2.8 and 3 are open, 4 Results waits for "
                "a completed run", [states[k][0] for k in ("af", "ep", "w")] == ["ready"] * 3
                and states["r"] == ("locked", "no completed run yet"), str(states))
            props = dict((r[0], r[1]) for r in sh._properties() if r[0] != "group")
            rep("Properties' Section group: the wing flies the family's own NACA 24tt, and is sent no flag",
                props.get("flies") == "NACA 2412 (the family's own)"
                and props.get("the wing is sent") == "nothing — it flies its own", str(props.get("flies")))
            chips = {r["id"]: r["chip"] for r in sh._build_rows() if r["level"] == 0}
            rep("live chips name what each surface flies before a choice, and the wing's size and objective",
                chips["af"] == "NACA 2412 (the family's own)" and chips["ep"] == "NACA 00tt (the family's own)"
                and chips["w"] == "14-D · side force" and chips["r"] == "", str(chips))

            # -- views of an unlocked stage, the empty state --------------------------------------------
            ok_sel = sh.select("af", "af.rank")
            g._draw_page()
            rep("a forced tab select inside an unlocked stage draws the empty state (af.rank before a screen)",
                ok_sel and dp.nav.current() == "af.rank" and not s.af.ranked
                and "af.rank" not in mine(sh.view_errors), dp.nav.current())

            # -- the N11 loop: every view in pre / running / post ----------------------------------------
            draw_views("post", STAGE_VIEWS["m"])
            draw_views("pre", STAGE_VIEWS["af"])
            sh.select("af", "af.screen")
            park(3)
            sh.run()
            live = settle()
            draw_views("running", STAGE_VIEWS["af"] + STAGE_VIEWS["m"])
            rep("Run on 2 Airfoil ▸ Library screening screens WingLab's library LIVE (an EngineJob)",
                isinstance(live, dj.EngineJob) and live.info.kind == "screen" and live.owner is s.af
                and sh.stage_state("af")[0] == "running", f"{type(live).__name__} "
                f"{getattr(getattr(live, 'info', None), 'kind', '')}")
            finish(live)
            outcomes.append(s.af.screen.get("outcome"))
            draw_views("post", STAGE_VIEWS["af"])
            chips = {r["id"]: r["chip"] for r in sh._build_rows() if r["level"] == 0}
            rep("after the screen the chip counts the ranking; the wing still flies its own",
                bool(s.af.ranked) and chips["af"] == f"{len(s.af.ranked)} ranked · flies NACA 2412",
                chips["af"])
            sh.select("af", "af.rank")
            dp.rank_list.idx = 0
            sh.take_rank()
            chips = {r["id"]: r["chip"] for r in sh._build_rows() if r["level"] == 0}
            c = s.af.chosen or {}
            rep("taking ranked row 0 (ENTER on the ranking) names it on the chip with its t/c, and it "
                "becomes the wing's section flag", s.af.decision == "library"
                and chips["af"].startswith(f"{c.get('name')} · t/c") and s.af.flag_value() is not None,
                chips["af"])
            sh.select("af", "af.opt")
            draw_views("pre", ["af.opt"])
            sh.select("af", "af.opt")
            park(5)                                      # the captured run's own Stop point
            sh.run()
            opt_job = settle()
            draw_views("running", STAGE_VIEWS["af"])
            chip = opt_job.chip() if opt_job is not None else None
            rep("Run on Shape optimisation optimises the section LIVE: RUNNING · k/N, stage running",
                isinstance(opt_job, dj.EngineJob) and opt_job.info.kind == "section"
                and chip == ("RUNNING · 5/12", "ACCENT") and sh.stage_state("af")[0] == "running",
                f"{chip}")
            # views_common's graph on a LIVE section run, and the now-evaluating block
            from . import views_common as vc
            scratch = pygame.Surface(g.screen.get_size())

            def builders(stage_view, fn):
                st = {}
                wu = W.WorkUI(scratch, sh._L.work, None, st, scroll=0, kbd_focus=False, mouse=(-1, -1),
                              overlay=W.Overlay(), now=0.0)
                try:
                    fn(wu, sh._ctx(stage_view, st, sh._L.work))
                    return wu.end(), ""
                except Exception as exc:                        # noqa: BLE001 -- the row reports it
                    wu.end()
                    return 0, f"{type(exc).__name__}: {exc}"

            def graph_blocks(wu, ctx):
                vc.run_banner(wu, ctx.job, None)
                vc.now_evaluating(wu, ctx, ctx.job)
                vc.convergence_card(wu, ctx.model.graph(), height=260)
            gr = s.af.graph()
            h_sec, err_sec = builders("af.opt", graph_blocks)
            rep("the live evaluation graph on a live section run: every evaluation so far, x to the budget, "
                "the Sobol | BO split (views_common draws it)",
                h_sec > 300 and not err_sec and len(gr["records"]) == 5 and gr["N"] == 12
                and gr["n_init"] == 4 and gr["live"], err_sec or f"{h_sec} px; {len(gr['records'])}/{gr['N']}")
            # the lock, scoped: the section run locks its surface's forms and the mission
            f_scr, f_sec, f_opt = s.af.screen_params, s.af.params, s.af.opt_params
            why = LOCK_TEXT.format(evaluator=opt_job.info.evaluator)
            locks = {k: sh.lock_reason(f, f.param(k)) for f, k in ((f_scr, "rec"), (f_sec, "use"),
                                                                     (f_opt, "more"), (f_opt, "stop"),
                                                                     (f_sec, "refine"))}
            rep("the live run locks its own forms (rec, use, more) and leaves Stop and the way on",
                locks["rec"] == why and locks["use"] == why and locks["more"] == why
                and locks["stop"] is None and locks["refine"] is None, str(locks))
            sel_before = g.build.left.wing
            g._menu_action("defaults")
            n_wings = len(lib.wings)
            key(pygame.K_s, ch="s")
            key(pygame.K_a, ch="a")
            page_after_a = g.page
            sh.select("m")
            key(pygame.K_RETURN, ch="\r")
            rep("global hazards are refused while it runs: the pause menu, S, A, and stating the mission",
                g.runs.live is opt_job and g.build.left.wing == sel_before and len(lib.wings) == n_wings
                and page_after_a == "section" and dp.opened_for == g.mission_page.signature()
                and g.page == "mission", f"live {g.runs.live is opt_job}, page after A {page_after_a}")
            rep("the mission form is locked with the hazard's sentence",
                sh.lock_reason(mp.params, mp.params.param("track")) == HAZARD, HAZARD)
            rep("menu items the lock refuses draw disabled while it runs; Stop and the log do not",
                not sh._menu_enabled("restart") and not sh._menu_enabled("save")
                and sh._menu_enabled("stop") and sh._menu_enabled("clearlog"), "restart, save off; stop, clear on")
            box = s.wing.box_params()
            rep("an out-of-scope edit is NOT refused (the wing's taper band while the airfoil optimises)",
                sh.lock_reason(box, box.param("bx.taper.min")) is None
                and box.is_enabled(box.param("bx.taper.min")), "bx.taper.min")
            rep("...but a second engine run is: the wing's Run refuses with the live run's sentence",
                sh.lock_reason(s.wing.solver_params, s.wing.solver_params.param("run")) == why, why)
            sh.select("af", "af.opt")
            g._draw_page()
            key(pygame.K_ESCAPE)
            st1 = opt_job.state
            key(pygame.K_ESCAPE)
            toasts = g.notices.drain_toasts()
            rep("ESC with a live run stops it; a second ESC while it stops only says so",
                st1 == "stopping" and g.page == "section"
                and ("stopping after the current evaluation", "info") in toasts, f"{st1}, {toasts[-1:]}")
            finish(None)
            tag = am.outcome_tag(s.af.opt.get("outcome"))
            outcomes.append(s.af.opt.get("outcome"))
            rep("Stop -> STOPPED: the stored outcome's tag, the best kept and usable",
                tag is not None and tag[0].startswith("STOPPED · ") and tag[1] == "WARN"
                and s.af.has_optimised(), str(tag))
            key(pygame.K_ESCAPE)
            back = g.page
            sh.select("af", "af.opt")
            rep("...and once the run has ended ESC goes back, as it always did",
                back == "mission" and g.runs.live is None, back)
            draw_views("stopped", ["af.opt"])
            draw_views("post", STAGE_VIEWS["af"])
            g._draw_page()
            n0 = len(sh.log.lines)
            key(pygame.K_f, ch="f")
            g._draw_page()
            fit_lines = sh.log.lines[n0:]
            rep("F on Shape optimisation takes the optimised section: one Output line, and 2.8 opens",
                len(fit_lines) == 1 and "CST optimised" in fit_lines[0][1] and s.af.decision == "optimised"
                and dp.nav.current().startswith("ep."), fit_lines[0][1][:70] if fit_lines else "none")
            chips = {r["id"]: r["chip"] for r in sh._build_rows() if r["level"] == 0}
            rep("after F the chip is the optimised section with its t/c",
                chips["af"].startswith(f"{s.af.chosen['name']} · t/c"), chips["af"])
            # 2.8: the plate screens AeroBO's SYMMETRIC sections (the owner's rule)
            draw_views("pre", STAGE_VIEWS["ep"])
            s.ep.re_source = "library"                   # the captured plate screen is the library point's
            sh.select("ep", "ep.screen")
            sh.run()
            g.runs.run_all()
            outcomes.append(s.ep.screen.get("outcome"))
            sym = set(am.bridge.api.symmetric_section_names())
            rep("Run on 2.8 Endplate screens the symmetric sections only",
                bool(s.ep.ranked) and all(r["name"] in sym for r in s.ep.ranked),
                f"{len(s.ep.ranked)} ranked, all symmetric")
            sh.select("ep", "ep.opt")
            park(3)
            sh.run()
            ep_job = settle()
            draw_views("running", STAGE_VIEWS["ep"])
            finish(ep_job)
            outcomes.append(s.ep.opt.get("outcome"))
            draw_views("post", STAGE_VIEWS["ep"])
            s.ep.decline()
            chips = {r["id"]: r["chip"] for r in sh._build_rows() if r["level"] == 0}
            chip_kept = chips["ep"]
            s.wing.type_params.param("plates").set("pylons")
            chip_fence, ep_state = sh._chip("ep"), sh._base("ep")
            s.wing.type_params.param("plates").set("endplates")
            rep("the plate kept reads the family's own; the pylons (plates a tip device) LOCK 2.8 "
                "with WingLab's reason",
                chip_kept == "NACA 00tt (the family's own, kept)"
                and chip_fence == "pylons: a tip device"
                and ep_state[0] == "locked" and "FENCE" in ep_state[1], f"{chip_kept} / {chip_fence}")
            # "Carried by" (the owner, 2026-09-25): 3 Wing ▸ Wing type draws only the
            # rows each mount brings, a row it does not draw takes no key, and the
            # keyboard / a pad walks only what is drawn
            tp = s.wing.type_params
            cond = {"blend", "blend_frac", "cant", "cant_deg", "tip", "tip_cant_deg"}
            want_rows = {("endplates", None): {"blend", "blend_frac", "cant", "cant_deg"},
                         ("pylons", "canted"): {"tip", "tip_cant_deg"}, ("pylons", "none"): {"tip"}}
            walks = []
            for (mount, tip), want_ in want_rows.items():
                tp.param("plates").set(mount)
                if tip:
                    tp.param("tip").set(tip)
                sh.select("w", "w.type")
                g._draw_page()
                drawn_ = {k_ for k_ in tp.order if tp._index(k_) is not None} & cond
                tp.focus_key = None
                seen = set()
                for _ in range(2 * len(tp.order) + 2):
                    tp.nav(+1)
                    seen.add(tp.focus_key)
                live_hidden = {k_ for k_ in cond - drawn_ if tp.is_enabled(tp.param(k_))}
                walks.append((mount, tip, drawn_, want_, (seen & cond) - drawn_, live_hidden))
            tp.focus_key = tp.reveal = None
            tp.param("tip").set("vertical")
            tp.param("tip_cant_deg").set(90.0)
            tp.param("plates").set("endplates")
            rep("Carried by: Wing type draws what each mount brings (endplates: root blend and "
                "leaning at; pylons: the tip device, its lean when canted) and no key lands on a "
                "row it does not draw",
                all(d_ == w_ and not off and not lh for _m, _t, d_, w_, off, lh in walks)
                and "w.type" not in sh.view_errors,
                "; ".join(f"{m_}{'/' + t_ if t_ else ''}: {sorted(d_)}"
                          + (f" (keys on {sorted(off | lh)})" if off or lh else "")
                          for m_, t_, d_, _w, off, lh in walks))
            # 3 Wing: RUNNING -> DONE, the graph, the law and the report
            draw_views("pre", STAGE_VIEWS["w"])
            draw_views("pre", STAGE_VIEWS["r"], force=True)
            #  a side wing opens on side force; the captured wing fixture was
            #  flown at AeroBO's default, efficiency (a player may pick it too)
            s.wing._set_objective("efficiency")
            sh.select("w", "w.solver")
            park(6)
            sh.run()
            view_after = sh.view
            wjob = settle()
            g._draw_page()
            rep("Run on 3 Wing launches WingLab's wing search LIVE and shows its Convergence",
                isinstance(wjob, dj.EngineJob) and wjob.owner is s.wing and wjob.info.kind == "wing"
                and view_after == "w.conv", f"view {view_after}")
            text = g.runs.status[0]
            rep("RUNNING: the chip, the stage glyph and the status bar all say k/N while it flies",
                wjob.chip() == ("RUNNING · 6/20", "ACCENT") and sh.stage_state("w")[0] == "running"
                and text.startswith("wing (lattice) evaluation 6/20"), f"{wjob.chip()} · {text}")
            draw_views("running", STAGE_VIEWS["w"])
            wbox = s.wing.box_params()
            rep("the wing run locks the design box and the section it flies, not the ways back",
                sh.lock_reason(wbox, wbox.param("boxreset")) == LOCK_TEXT.format(evaluator=wjob.info.evaluator)
                and sh.lock_reason(s.af.params, s.af.params.param("use")) is not None
                and sh.lock_reason(s.results.params, s.results.params.param("to3")) is None,
                "boxreset, use locked; to3 free")
            finish(wjob)
            outcomes.append(s.wing.outcome)
            tag = am.outcome_tag(s.wing.outcome)
            gw = s.wing.graph()
            rep("DONE: the stored outcome's tag, and the graph reads the stored run (N 20, Sobol | BO at 4, "
                "BO -> SLSQP at 5)", tag is not None and tag[:2] == ("DONE · 20/20", "GOOD")
                and g.runs.live is None and gw["N"] == 20 and len(gw["records"]) == 20
                and gw["n_init"] == 4 and gw["handoff"] == 5 and not gw["live"], f"{tag} · {gw['N']}")
            sm = s.results.summary()
            gf, af_ = sm.get("game_force_N"), sm.get("aerobo_force_N")
            rep("the car's law is derived and WingLab's report landed: the game's force IS WingLab's",
                s.wing.law is not None and s.results.report is not None and gf is not None and af_
                and abs(gf - af_) <= 1e-6 * abs(af_), f"game {gf} N, WingLab {af_} N")
            chips = {r["id"]: r["chip"] for r in sh._build_rows() if r["level"] == 0}
            rep("after the wing run 4 Results is done, its chip the best score in WingLab's units",
                sh._base("r")[0] == "done"
                and chips["r"] == score_text(s.wing.record.get("best_score"), s.wing.record.get("score_units")),
                chips["r"])
            draw_views("post", STAGE_VIEWS["w"] + STAGE_VIEWS["r"])
            if "r.summary" in keys:
                # -- 4 Results ▸ Summary's "Best design in its box" (views_results.box_rows) --------------
                from . import views_results as vr
                syn = {"param_labels": ["a", "b", "c", "d", "e"], "best_x": [0.01, 1.97, 2.5, 0.5, 4.0],
                       "bounds": [[0.0, 1.0], [0.0, 2.0], [0.0, 2.0], [0.5, 0.5], [0.0, 10.0]],
                       "config": {"bounds_overrides": {"b": [0.0, 2.0], "d": [0.4, 0.6]}}, "pinned": {"d": 0.5}}
                br = {r["label"]: r for r in vr.box_rows(syn)}
                rep("the box card's rows are WingLab's design_box: riding low / high inside 2 %, outside, "
                    "narrowed = in the run's overrides; a fixed row ([v, v]) is fixed, never ridden or narrowed",
                    br["a"]["riding"] == "low" and br["b"]["riding"] == "high" and br["b"]["narrowed"]
                    and not br["a"]["narrowed"] and br["c"]["outside"] and br["d"]["fixed"] == 0.5
                    and br["d"]["frac"] is None and not br["d"]["riding"] and not br["d"]["narrowed"]
                    and not br["d"]["outside"] and br["e"]["riding"] == "" and abs(br["e"]["frac"] - 0.4) < 1e-12,
                    "; ".join(f"{k} {r['riding'] or '·'}" + " out" * r["outside"] + " narrowed" * r["narrowed"]
                              + (" fixed" if r["fixed"] is not None else "") for k, r in br.items()))
                bare = vr.box_rows({"param_labels": ["a", "b"], "best_x": [0.3, None]})
                junk = [vr.box_rows(x) for x in (None, {}, {"best_x": 3.0},
                                                 {"best_x": [1, "x", float("nan"), True], "config": "?",
                                                  "bounds": [[0, 1], "no", [2, 1, 0]], "pinned": [1]})]
                rep("a record with no box keeps its values and invents no bounds; junk never raises",
                    [r["value"] for r in bare] == [0.3, None]
                    and all(r["lo"] is None and r["frac"] is None and not r["riding"] and r["fixed"] is None
                            for r in bare)
                    and junk[:3] == [[], [], []] and [r["value"] for r in junk[3]] == [1.0, None, None, None]
                    and junk[3][0]["label"] == "x0" and junk[3][0]["riding"] == "high", "")
                rec0 = s.wing.record
                rows0 = vr.box_rows(rec0)
                src0 = vr.box_sources(s.wing, rows0) or {}
                ov0 = set((rec0.get("config") or {}).get("bounds_overrides") or {})
                #  the captured run fixed nothing: a row the box fixes now was fixed since
                since = sorted(lab for lab in s.wing.pinned() if lab in (rec0.get("param_labels") or []))
                s.wing.box["taper"] = [0.5, 0.8]                    # a band typed since the run
                try:
                    typed = vr.box_sources(s.wing, rows0) or {}
                finally:
                    s.wing.box.pop("taper", None)
                #  a run that fixed a row by hand: named while the box holds it at that value
                i_t = list(rec0.get("param_labels") or []).index("twist_root_deg")
                x_held = [0.0 if i == i_t else v for i, v in enumerate(rec0["best_x"])]
                held_rows = vr.box_rows(dict(rec0, pinned={"twist_root_deg": 0.0}, best_x=x_held))
                held = {}
                for v in (0.0, 1.0):
                    s.wing.fixed["twist_root_deg"] = v
                    try:
                        held[v] = (vr.box_sources(s.wing, held_rows) or {}).get("twist_root_deg")
                    finally:
                        s.wing.fixed.pop("twist_root_deg", None)
                rep("...on the finished run: a row per design variable, inside the box it searched; the car's "
                    "size rows narrowed and named car packaging, the ride row the slot's; a row fixed or "
                    "retyped since the run names nobody (the card says the box changed)",
                    [r["label"] for r in rows0] == list(rec0.get("param_labels") or [])
                    and all(r["lo"] is not None and not r["outside"] and r["fixed"] is None for r in rows0)
                    and {r["label"] for r in rows0 if r["narrowed"]} == ov0 == {"b_m", "S_m2"}
                    and src0.get("b_m") == src0.get("S_m2") == "car packaging"
                    and src0.get("ride_height_m") == "slot" and src0.get("taper") == "WingLab default"
                    and sorted(r["label"] for r in rows0 if r["label"] not in src0) == since
                    and "taper" not in typed and typed.get("b_m") == "car packaging"
                    and held_rows[i_t]["fixed"] == 0.0 and not held_rows[i_t]["outside"]
                    and held == {0.0: "fixed", 1.0: None},
                    ", ".join(f"{k} {v}" for k, v in src0.items() if v != "WingLab default")
                    + f"; unnamed {[r['label'] for r in rows0 if r['label'] not in src0]}; fixed by hand {held}")

                def summary_of(rec):
                    """4 Results ▸ Summary drawn on `rec`: (the hints it said, its form, error)."""
                    said, got = [], {}

                    def fn(wu, ctx):
                        real = wu.hint

                        def hint(text, *a, **kw):
                            said.append(str(text))
                            return real(text, *a, **kw)
                        wu.hint = hint
                        got["form"] = wu.form
                        vr.summary(wu, ctx)
                    s.wing.record = rec
                    try:
                        _h, err_ = builders("r.summary", fn)
                    finally:
                        s.wing.record = rec0
                    return said, got.get("form"), err_
                i_r = next(i for i, r in enumerate(rows0) if r["frac"] is not None and not r["riding"])
                said, form_, err_r = summary_of(dict(rec0, best_x=[r["hi"] if i == i_r else r["value"]
                                                                   for i, r in enumerate(rows0)]))
                link = (form_.extras.get("ui.res.box") if form_ is not None else None)
                warned = [t for t in said if "ran into a bound" in t]
                if link is not None:
                    link.set(None)
                rep("a winner riding its box edge: the card names the row and whose band it is, and its link "
                    "opens 3 Wing ▸ Design box",
                    not err_r and len(warned) == 1 and rows0[i_r]["label"] in warned[0]
                    and link is not None and sh.stage == "w" and dp.nav.current() == "w.box",
                    err_r or (warned[0][:90] if warned else "no warning"))
                sh.select("r", "r.summary")
                said, form_, err_n = summary_of(dict(rec0, bounds=None, pinned=None,
                                                     config=dict(rec0.get("config") or {}, pinned=None)))
                rep("a record with no box: its values only, and WingLab's sentence that there is no box to "
                    "place them in", not err_n and vr.BOX_NONE in said
                    and not any("ran into a bound" in t for t in said), err_n or "")
                #  a winner at the car's span limit: the CAR bound it, whoever set the band -- the
                #  slot's packaging row, the row released (its own band cut to the limit) or typed up
                #  to it, Unlimited's ceiling; a typed band short of the limit is the player's own
                w_, op_ = s.wing, s.wing.session.op
                i_b = list(rec0.get("param_labels") or []).index("b_m")
                lo_b, cap_b = rows0[i_b]["lo"], float((op_.size_caps or {})["b_m"])

                def span_at_top(how):
                    """(the record, b_m's source) with the winner at the top of b_m's band, set `how`."""
                    w_.released.discard("b_m")
                    w_.box.pop("b_m", None)
                    if how == "released":
                        w_.released.add("b_m")
                    elif how in ("typed", "short"):
                        w_.box["b_m"] = [lo_b, cap_b - (0.1 if how == "short" else 0.0)]
                    band = [float(v) for v in w_.family_box()["b_m"]]
                    rec = dict(rec0, bounds=[band if i == i_b else b for i, b in enumerate(rec0["bounds"])],
                               best_x=[band[1] if i == i_b else v for i, v in enumerate(rec0["best_x"])])
                    return rec, (vr.box_sources(w_, vr.box_rows(rec)) or {}).get("b_m")
                was = (set(w_.released), {k: list(v) for k, v in w_.box.items()}, op_.unlimited)
                capped, said_c, err_c = {}, [], "not drawn"
                try:
                    for how in ("packaging", "released", "typed", "short"):
                        capped[how] = span_at_top(how)[1]
                    op_.unlimited = True
                    capped["unlimited"] = span_at_top("packaging")[1]
                    op_.unlimited = was[2]
                    rec_rel, _src = span_at_top("released")
                    said_c, _form, err_c = summary_of(rec_rel)
                finally:
                    op_.unlimited = was[2]
                    w_.released.clear()
                    w_.released.update(was[0])
                    w_.box.clear()
                    w_.box.update(was[1])
                warned_c = [t for t in said_c if "ran into a bound" in t]
                rep("a winner at the car's span limit names the limit, not whoever set the band (packaging, "
                    "released, typed up to it; Unlimited's ceiling) -- a band short of it stays the player's; "
                    "the warning says the Design box cannot open it and gives the limit",
                    capped == {"packaging": vr.CAP_WORDS["b_m"], "released": vr.CAP_WORDS["b_m"],
                               "typed": vr.CAP_WORDS["b_m"], "short": "user", "unlimited": vr.UNLIMITED_CAP}
                    and not err_c and len(warned_c) == 1 and f"b_m ({vr.CAP_WORDS['b_m']})" in warned_c[0]
                    and "this car's ceiling" in warned_c[0] and "Widen" not in warned_c[0]
                    and any(t.startswith("Span limit — ") for t in said_c),
                    err_c or f"{capped}; cap {cap_b:.3g} m; " + (warned_c[0][:80] if warned_c else "no warning"))

            def wing_blocks(wu, ctx):
                vc.convergence_card(wu, ctx.model.graph(), height=260)
            h_w, err_w = builders("w.conv", wing_blocks)
            rep("views_common's graph card draws the finished wing run", h_w > 250 and not err_w,
                err_w or f"{h_w} px")
            # a stage that is done and runs again shows running while another is selected
            sh.select("w", "w.type")
            park(2)
            s.af.start_optimise()
            again = settle()
            rep("glyph precedence: 2 Airfoil (done) optimising again shows `running` from 3 Wing (done)",
                sh._base("af")[0] == "done" and sh.stage_state("af")[0] == "running"
                and sh.stage_state("w")[0] == "done", f"af {sh.stage_state('af')[0]}, "
                f"w {sh.stage_state('w')[0]}")
            g.runs.stop()
            finish(again)
            sh.select("r", "r.summary")
            g._draw_page()
            g.notices.drain_toasts()
            sh.run()
            rep("Run on 4 Results with the report in starts nothing and says so",
                g.runs.live is None and any("Keep going" in t for t, _k in g.notices.drain_toasts()), "")
            # a Keep going watched from Results, then stopped: STOPPED
            sh.select("r", "r.summary")
            s.wing.more = 6
            park(3)
            dp.run_wing(extend=True)
            kjob = settle()
            draw_views("running", STAGE_VIEWS["r"])
            rep("Keep going resumes WingLab's record: a continued run, its inherited evaluations counted",
                isinstance(kjob, dj.EngineJob) and kjob.info.continued and kjob.n_prior == 20,
                f"n_prior {getattr(kjob, 'n_prior', None)}")
            g.runs.stop()
            finish(kjob)
            outcomes.append(s.wing.outcome)
            tag = am.outcome_tag(s.wing.outcome)
            rep("Stop on the wing -> STOPPED on its stored outcome (best kept)",
                tag is not None and tag[0].startswith("STOPPED") and s.wing.record is not None, str(tag))
            draw_views("post", STAGE_VIEWS["r"] + ("w.conv",))
            need = {"pre", "running", "post"}
            missing = [k for k in keys if not need <= drawn.get(k, set())]
            errs = mine(sh.view_errors)
            rep("N11: every view draws in pre / running / post at 1280x800",
                not missing and not errs,
                f"{len(keys)} views, worst {worst[0]:.1f} ms at {worst[1]}"
                + (f"; missing {[(k, sorted(drawn.get(k, ()))) for k in missing]}" if missing else "")
                + (f"; errors {sorted(errs)}" if errs else ""))

            # -- C10: after a forced select of each step, that step is in the tree ----------------------
            unseen = []
            for k in dp.nav.keys:
                dp.nav.select(k, force=True)
                g._draw_page()
                if k not in [h[0] for h in dp.nav._hits]:
                    unseen.append(k)
            rep("C10: after a forced select of each of the 16 steps, it is drawn and hit-recorded",
                not unseen, f"{len(dp.nav.keys) - len(unseen)}/16" + (f", unseen {unseen}" if unseen else ""))
            ex = dp.nav.expanded
            rep("with every stage expanded at 1280x800 the tree scrolls, and a select reveals its block",
                all(ex.values()) and sh.tree_scroll > 0 and "r.evals" in [h[0] for h in dp.nav._hits],
                f"scroll {sh.tree_scroll}, {len(sh._rows)} rows")

            # -- the focus regions ---------------------------------------------------------------------
            sh.select("af", "af.screen")
            sh.region = "tree"
            seen = []
            for _ in range(4):
                seen.append((sh.region, dp.focus))
                key(pygame.K_TAB)
            dp.focus = "rows"
            rep("TAB cycles tree -> tabs -> work; dp.focus stays two-valued and moves the region",
                seen == [("tree", "nav"), ("tabs", "nav"), ("work", "rows"), ("tree", "nav")]
                and sh.region == "work", str(seen))
            key(pygame.K_F1)
            g._draw_page()
            dlg = sh.dialog is not None and sh.dialog.open and sh.dialog.title == "Keys and controller"
            key(pygame.K_ESCAPE)
            rep("F1 opens the keys dialog; ESC closes it and nothing else",
                dlg and sh.dialog is None and g.page == "section", "")
            sh.stop()
            rep("Stop with nothing running toasts it",
                ("nothing is running", "info") in g.notices.drain_toasts(), "")

            # -- keyboard reach: every control of every view, in its post state, by DOWN alone ----------
            reach_bad = []
            for k in keys:
                if k.endswith(".rank"):
                    continue
                sh.select(STAGE_OF[k], k)
                sh.region = "work"
                sh.scroll[k] = 0
                g._draw_page()
                form = sh._current_form()
                if form is None:
                    continue
                form.focus_key = None
                focusable = [q for q in form.order if form.is_enabled(form.param(q))]
                seq, invisible = [], []
                for _ in range(8 * len(form.order) + 200):          # a table's rows are inner steps
                    key(pygame.K_DOWN)
                    g._draw_page()
                    fk = form.focus_key
                    gm = form.geom(fk) if fk is not None else None
                    if gm is not None and not gm.visible:
                        invisible.append(fk)
                    if not seq or seq[-1] != fk:
                        seq.append(fk)
                    if len(seq) > 1 and seq[-1] == seq[0]:
                        break
                visited = seq[:-1] if len(seq) > 1 and seq[-1] == seq[0] else seq
                buttons = [q for q in focusable if form.geom(q) is not None and form.geom(q).kind == "button"]
                if set(visited) != set(focusable) or len(visited) != len(set(visited)) or invisible \
                        or not set(buttons) <= set(visited):
                    reach_bad.append(f"{k}: {len(set(visited))}/{len(focusable)}"
                                     + (f", off screen {invisible[:3]}" if invisible else ""))
            rep("keyboard reach: DOWN visits every control of every view once, each on screen when focused",
                not reach_bad, "; ".join(reach_bad) or f"{len([k for k in keys if not k.endswith('.rank')])} views")
            sh.region = "tree"

            # -- the mission page's tree walk -------------------------------------------------------------
            sh.select("m", "m.search")
            sh.region = "tree"
            sh.tree_cursor = None
            key(pygame.K_UP)
            up = g.mission_tab
            key(pygame.K_DOWN)
            key(pygame.K_DOWN)
            rep("on the mission page UP / DOWN walk the visible rows; a design stage row selects its stage",
                up == "m.design" and g.page == "section" and sh.stage == "af", f"{up} -> {g.page} {sh.stage}")

            # -- the hint routing -------------------------------------------------------------------------
            sh.select("af", "af.screen")
            sh.region = "work"
            f_scr = s.af.screen_params
            sh.reveal_now("w.astall")          # 0 on AeroBO's wing weights
            g._draw_page()
            gm = f_scr.geom("w.astall")
            n0 = len(sh.log.lines) + len(g.notices._log)
            moved = False
            if gm is not None and "track" in (gm.parts or {}):
                x0 = gm.x_for(0.0)
                y0 = gm.parts["track"].centery
                g._handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(x0, y0)))
                for i in range(20):
                    g._handle(pygame.event.Event(pygame.MOUSEMOTION, pos=(x0 + 3 * i, y0), rel=(3, 0),
                                                 buttons=(1, 0, 0)))
                g._draw_page()
                g._handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(x0 + 60, y0)))
                g._draw_page()
                moved = s.af.weights.get("astall", 0.0) > 0.0
            n1 = len(sh.log.lines) + len(g.notices._log)
            s.af.recommend()
            rep("a 20-event slider drag moves the weight and adds no Output line", moved and n1 == n0,
                f"{n1 - n0} lines" + ("" if gm is not None else "; no w.astall slider drawn"))
            g._draw_page()
            dashed = [t for _s, t, _l in sh.log.lines if " -- " in t]
            rep("no Output line carries the old hint bar's ' -- '", not dashed, dashed[0][:60] if dashed else "")

            # -- typing into a field swallows the design keys ----------------------------------------------
            sh.reveal_now("tcmin")
            g._draw_page()
            gm = f_scr.geom("tcmin")
            editing = False
            if gm is not None and "text" in (gm.parts or {}):
                g._handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=gm.parts["text"].center))
                g._handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=gm.parts["text"].center))
                editing = f_scr.editing == "tcmin"
                for ch_ in "l0.1":
                    key(ord(ch_), ch=ch_)
                key(pygame.K_RETURN, ch="\r")
            rep("typing into a number field: L is swallowed, 0.1 ENTER commits",
                editing and g.runs.live is None and abs(s.af.screen_gates["tc_min"] - 0.1) < 1e-12,
                f"t/c gate {s.af.screen_gates['tc_min']}")
            s.af.screen_gates["tc_min"] = am.bridge.SCREEN_GATES["tc_min"]

            # -- None-safety: a fresh garage forced onto the shell pages, and holes in the models ----------
            g2 = new_garage(wing="")
            crashed = []
            for page in ("section", "mission"):
                g2.page = page
                try:
                    g2._draw_page()
                except Exception as exc:                        # noqa: BLE001 -- the row reports it
                    crashed.append(f"{page}: {exc}")
            g2.open_mission("left")
            g2.state_mission()
            s2 = g2.design_page.session
            s2.af.chosen, s2.af.decision = {"source": "library"}, "library"     # a half-made choice
            s2.wing.outcome = {"state": "error", "k": 0, "n": 53, "error": "an engine that said nothing"}
            g2.mission_page.result = None
            for stage, view in (("w", "w.conv"), ("w", "w.solver"), ("af", "af.section"),
                                ("m", "m.operating")):
                try:
                    g2.shell.select(stage, view)
                    g2._draw_page()
                except Exception as exc:                        # noqa: BLE001
                    crashed.append(f"{view}: {exc}")
            rep("the chrome is None-safe: a fresh garage forced onto both pages, a half-made choice, a "
                "failed run, no lap", not crashed,
                "; ".join(crashed) or f"{len(g2.shell.safe_errors)} reads fell back to '—'")

            # -- isolation: a view module that fails to import breaks its own views only -------------------
            victim = "af.rank"
            VIEWS[victim] = "views_no_such_module:ranking"
            try:
                sel_ok = sh.select("af", victim)
                g._draw_page()
                broken = victim in sh.view_errors
                sh.select("af", "af.screen")
                g._draw_page()
                fine = "af.screen" not in sh.view_errors
            finally:
                VIEWS[victim] = saved_views[victim] if only is None or victim in only else "design_shell:generic_view"
            logged = any("render error" in t for _s, t, _l in sh.log.lines)
            rep("a view module that raises at import draws its render-error card; the others draw",
                sel_ok and broken and fine and logged, "ModuleNotFoundError caught, logged once")
            sh.view_errors.pop(victim, None)

            # -- the crumbs and prev / next go through the same select ----------------------------------------
            sh.select("af", "af.screen")
            sh._tool("prev")
            prev_page = g.page
            sh._tool("crumb:w")
            at_w = sh.stage
            sh._tool("next")
            rep("prev / next and the crumbs select through the stage gate",
                prev_page == "mission" and at_w == "w" and sh.stage == "r", f"{prev_page}, {at_w}, {sh.stage}")

            # -- restating an unchanged mission keeps the progress -------------------------------------------
            ranked = list(s.af.ranked)
            sh.select("m")
            key(pygame.K_F5)
            rep("Run on an unchanged, stated mission only re-checks it (the stages keep their progress)",
                g.page == "section" and dp.session is s and s.af.ranked == ranked and s.wing.record is not None,
                f"{len(ranked)} still ranked")
            failed = [o for o in outcomes if (o or {}).get("state") == "error"]
            rep("every run the check launched replayed a captured fixture: none FAILED, nothing reached XFOIL",
                all(outcomes) and not failed, "; ".join(str(o.get("error")) for o in failed)
                or f"{len(outcomes)} runs")
            rep("no view under test failed to render", not mine(sh.view_errors), str(sorted(mine(sh.view_errors))))
            rep("runs/ is untouched", runs_state() == runs0, "")
    finally:
        VIEWS.clear()
        VIEWS.update(saved_views)
        T.set_scale(1.0)
        shutil.rmtree(tmp, ignore_errors=True)
    if verbose:
        print(f"  design_shell self-check: {n_rows} rows, " + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return ok


if __name__ == "__main__":
    args = sys.argv[1:]
    only_keys = None
    if "--only" in args:
        i = args.index("--only")
        only_keys = [k for a in args[i + 1:] if not a.startswith("--") for k in a.split(",") if k]
    sys.exit(0 if self_check(only=only_keys) else 1)
