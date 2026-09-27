"""drive/wing_tutorial.py -- the wing-design tutorial (task 24).

A guided pass through the garage's DESIGN navigator -- AeroBO's own stages,
on AeroBO's own engine since the pivot (drive/aerobo_models.py) -- in the
order the tutorial walks it: the mission -> the airfoil (the screening
weights, the ranking, taking a section) -> the end plates -> the wing's
planform (the design box) -> the run and its results -> the wing fitted to
the car -> the car saved as a build -> drive it (the pre-race screen's
Build > PICK offers it). AeroBO opens its section, endplate and wing stages
together once the mission is stated (the wing flies the family's own
sections until one is chosen); the tutorial keeps its order by guidance.

Ten data-driven steps `WStep(id, title, text, do, check, anchor)`. `check(g,
mem, action)` is a predicate on the live `Garage` (its page, the mission,
the design page's two section models and its wing, the build, the library)
and the action the garage's frame returned; `anchor` names the widget the
hint points at -- a step of the Simulation tree (`('nav', 'af.screen')`), a
control of the page's form (`('row', 'go')`), a tool button (`('tool',
'save')`), a library list or the bar across the page (the tool bar on the
AeroBO pages, the key bar on the dark ones) -- and `anchor_rect` finds it
where the page's last draw put it (every garage widget keeps its own
`_rect` / `_hits` for the mouse; the shell records its tool buttons and
every control's geometry). A tree step whose stage is folded points at its
stage row, and a control laid out below the fold is scrolled in once, when
its step starts. The box says, in plain words, what the step is and what
each number means, and what to press. On the car and library pages it is
the dark box at `BOX_CAR`; on the mission and design pages (the AeroBO
shell) it is a light one at the shell's `tutor_box`, over the work area's
lower right.
Task 45: it counts ONE way ('Step 2 of 10: The mission'), says 'airfoil' as the
pages do and, on the dark pages, docks where the page under it has the least
on it (`DOCKS`, `_ink`) and points with a short arrow beside the widget, not
a line across the page.

garage.py gets hooks only: `Garage.tutor` (this object or None), queried by
`Garage.frame` (`update` after the events, `draw` after the page) and by
the garage's pause menu (`menu_rows` / `menu_action`: skip a step, end,
start over). H hides the box. Progress is the `wing_tutorial` section of
`runs/progress.json` (drive/progress.py); a player session starts it from
the garage's menu, or from the driving tutorial's last page's pointer.

Nothing here reaches the physics, and nothing is designed FOR the player:
every step is passed by the player's own press, through the garage's own
gates.
"""
from __future__ import annotations

from dataclasses import dataclass

SECTION = "wing_tutorial"
#: where the box may dock, at 1280x800, in order of preference: (left edge,
#: 'foot' | 'mid' | 'top', width). A negative left edge counts from the
#: right, the box's right side on the page's (1268). The box takes the dock
#: with the least ink under it on the page it is over (`WingTutor._place`)
#: and never the widget it points at -- the car page's empty sky, the
#: mission page's empty column. Task 45: it had ONE place, over the lower
#: right of every page, the design page's notes and SPAN LIMITS included.
DOCKS = tuple((x, v, w) for v in ("foot", "mid", "top") for x in (-1, 12, 324)
              for w in (520, 440))
DOCK_TOP, DOCK_FOOT = 52, 680       # px: under the car page's BUILD line; over the key bar
BOX_H_MAX = 320                     # px: a box's rows past this are cut
BOX_SIZE = 13                       # the box's font size
INK_LEVEL = 82                      # a pixel with every channel under this is background
ARROW_PX = 44                       # the pointer's length at ui 1 (never past 60)
BOX_CAR = (748, 468, 520, 212)      # px at 1280x800 (x the page's scale): over the page's lower right
BOX = BOX_CAR                       # the name the older checks read
BOX_GROW = 100                      # the box grows UP by at most this much (x scale): 312 px tall
WRAP_CH = 66                        # characters a line in the box
C_BOX_ACCENT = (217, 206, 85)       # the key colour: the hint and the anchor outline
RESUME_FLASH_N = 600                # garage frames (~10 s) the box says why a resume went back
#: the garage pages the AeroBO shell frames (design_shell.SHELL_PAGES)
SHELL_PAGES = ("mission", "section")


@dataclass(frozen=True)
class WStep:
    id: str
    title: str
    text: str                        # what this is, in plain words
    do: object                       # what to press: a str, or (g) -> str on this garage
    check: object                    # (g, mem, action) -> bool
    anchor: tuple = ()               # ('nav', key) | ('row', key) | ('tool', id) | ('list', name) | ('bar',)


def _dp(g):
    return g.design_page


def _cur(g) -> str:
    return _dp(g).nav.current() if g.page == "section" else ""


def _chk_open(g, m, a):
    return g.page in ("mission", "section")


def _chk_mission(g, m, a):
    return bool(g.mission.stated) and g.page == "section"


def _chk_screen(g, m, a):
    return g.page == "section" and bool(_dp(g).af.ranked)


def _chk_section(g, m, a):
    return g.page == "section" and _dp(g).af.finished()


def _chk_plates(g, m, a):
    dp = _dp(g)
    return g.page == "section" and dp.af.finished() and (dp.plate_locked() or dp.ep.finished())


def _chk_planform(g, m, a):
    """Seen the design box, then moved on (to the solver, the results...)."""
    k = _cur(g)
    if k == "w.box":
        m["seen"] = True
    return bool(m.get("seen")) and k != "w.box" and k.startswith(("w.", "r."))


def _chk_results(g, m, a):
    return _cur(g).startswith("r.")


def _chk_fit(g, m, a):
    """The designed wing is saved and it is the one in the slot. Task 45: a
    new garage (the next launch, a trip to the drive and back) has no wing
    on its design page -- then it is the wing step 8 fitted, as the tutor
    remembers it (`WingTutor.wing`), still in its slot. Step 9 could never
    pass in a new garage before."""
    w = _dp(g).wing
    if w is None:                            # a new garage: step 8's wing, kept
        kept = getattr(getattr(g, "tutor", None), "wing", None)
        return bool(kept) and kept[1] in g.lib.wings and g.build.slot(kept[0]).wing == kept[1]
    if w.spec is None:                       # no run fitted yet: nothing to save
        return False
    return (not w.dirty and w.spec.name in g.lib.wings
            and g.build.slot(w.key).wing == w.spec.name)


def _chk_build(g, m, a):
    """The car as it stands -- the designed wing still in its slot -- is a
    saved build (by content, not by name). Loading some other build from the
    list is not saving this one."""
    if not _chk_fit(g, m, a):
        return False
    try:
        from .prerace import _same_build
        js = g.build.to_json()
        return any(_same_build(b, js) for b in g.lib.builds.values())
    except Exception:                      # noqa: BLE001
        return False


def _chk_drive(g, m, a):
    return a == "drive"


def _do_open(g=None) -> str:
    """Step 1's press. That the design replaces the wing already in the slot is
    said only when the slot HAS one (task 45: over an empty car it read as a
    riddle): the selected slot, or the left one while the top is selected --
    the flank the guided start takes."""
    s = "Select a flank slot (1 = left) and press D (L3 on the pad)."
    k = getattr(g, "sel", "left")
    k = k if k in ("left", "right") else "left"
    if g is not None and g.build.slot(k).wing:
        s += (" The design replaces the wing already in that slot when you save it: to "
              "keep that one, rename the new wing at step 8.")
    return s


def do_text(st, g=None) -> str:
    """What step `st` says to press, on garage `g` (None: its plain words)."""
    return st.do(g) if callable(st.do) else st.do


STEPS = (
    WStep("open", "Design a wing",
          "A wing turns moving air into force. DOWNFORCE presses the tyres into the "
          "road; DRAG holds the car back on the straights. A FLANK wing stands on "
          "the side of the car and pushes SIDEWAYS, into the corner -- grip the tyres "
          "do not have to find -- and folds away on the straights. The garage designs "
          "one with WingLab, in its order: mission, section, endplate, wing, results.",
          _do_open, _chk_open, ("bar",)),
    WStep("mission", "The mission",
          "What is the wing FOR? A circuit and a surface. 'lap' is an estimate used to "
          "compare wings, not your lap time; its mean speed is the DESIGN SPEED WingLab flies "
          "the wing at (1 Mission ▸ Design point). Search & budget shows how many "
          "designs each stage will try -- WingLab's own measured budgets.",
          "Pick them on 1 Mission ▸ Operating point, then the 'State the mission' "
          "button (ENTER / CROSS; F5 / SQUARE runs the stage too).",
          _chk_mission, ("row", "go")),
    WStep("screen", "The airfoil: screening",
          "The AIRFOIL (section) is the shape of the wing cut front to back. The "
          "screening WEIGHTS say what matters: L/D at the design lift (lift per unit "
          "of drag) and its best L/D, cl max (the most lift it can make), |cm| (its "
          "twisting moment), thickness (room for a spar), stall angle. Screening "
          "scores WingLab's whole section library against them.",
          "The 'Screen the library' button on 2 Airfoil ▸ Library screening (or L, or "
          "F5 / SQUARE). It runs live: the status bar counts, the tag says RUNNING, "
          "then DONE.",
          _chk_screen, ("nav", "af.screen")),
    WStep("section", "The airfoil: take one",
          "The RANKING lists the sections best first, with the numbers the weights "
          "priced. Taking one puts it on the wing and moves on. (Shape optimisation "
          "can refine it further: WingLab's own search, 164 XFOIL evaluations at its "
          "balanced budget -- a long live run, and optional.)",
          "On 2 Airfoil ▸ Ranking highlight a row (click, or UP / DOWN) and press F "
          "(or ENTER / CROSS on it).",
          _chk_section, ("nav", "af.rank")),
    WStep("plates", "The end plates",
          "END PLATES are the panels at the wing's tips: they stop the air spilling "
          "round the end, which keeps the lift -- and here they also carry the wing. "
          "Their section is chosen the same way, from SYMMETRIC sections only -- or "
          "keep the family's own plate.",
          "2.8 Endplate ▸ Library screening: screen, take one -- or its 'Keep the "
          "family's own plate' button.",
          _chk_plates, ("nav", "ep.screen")),
    WStep("planform", "The planform",
          "WING TYPE is what is being built and what it is for (the objective: "
          "efficiency, force, drag). The DESIGN BOX is every number WingLab will "
          "search: span, area, taper, twist, incidence (the angle it meets the air), "
          "the plates' height and thickness, the height above the car (a flank's: "
          "its reach to the car's side). More area = more force AND more drag.",
          "Look at 3 Wing ▸ Design box (TAB / TRIANGLE to the work area), then move on "
          "to another tab.",
          _chk_planform, ("nav", "w.box")),
    WStep("results", "The run and its results",
          "Run on 3 Wing ▸ Solver (O, or F5 / SQUARE) lets WingLab search the box: "
          "every dot on the graph is one wing it tried. When the tag says DONE (or "
          "STOPPED -- the best is kept), 4 Results holds the winner: its force and "
          "drag in newtons, and what the car will feel.",
          "Run it, then open 4 Results ▸ Summary (click it in the Simulation tree, or "
          "UP / DOWN there).",
          _chk_results, ("nav", "r.summary")),
    WStep("fit", "Fit it to the car",
          "Saving puts the winner in the library and in the slot (mirrored left / "
          "right unless M split them), at WingLab's incidence, under the name the page "
          "shows. The car drives with WingLab's own forces for it.",
          "N renames it first if you like; the Save tool button (or S) puts it on the "
          "car (pad: CIRCLE -- leaving the page saves it).",
          _chk_fit, ("tool", "save")),
    WStep("build", "Save the car as a build",
          "A BUILD is the whole car: all three slots, their stations and angles, "
          "kept for the car it was made for. Saved builds are what the TIME TRIAL "
          "page offers (Build > PICK) and B steps through, on any map, each with its "
          "best lap; F makes one the car's own default.",
          "ESC / CIRCLE back to the car and S saves it (pad: OPTIONS > Save build); "
          "or L opens the library, where S / SQUARE saves it too.",
          _chk_build, ("list", "builds")),
    WStep("drive", "Drive it",
          "The TIME TRIAL page opens with this build; its laps are filed with it, "
          "so the table shows which of your builds is fastest.",
          "ESC / CIRCLE back to the car and ENTER / CROSS: drive it.",
          _chk_drive, ("bar",)),
)


#: the steps whose state is the design chain's (the mission, the pages)
CHAIN = ("mission", "screen", "section", "plates", "planform", "results", "fit")
#: why a step could not go on where it stopped (`WingTutor.resume`): the
#: chain's steps, their design page's work; step 9, its wing
RESUME_WHY = {"build": "the tutorial's wing is not in its slot now"}
RESUME_WHY_CHAIN = "the design page needed a fresh start"


def _live(g) -> bool:
    """Has garage `g` design-page work to go back to: the design page opened
    here, on a mission still stated? A new garage has neither."""
    return bool(g.mission.stated) and _dp(g).wing is not None


def can_resume(g, sid: str) -> bool:
    """Can step `sid` go on in garage `g` as it stands (task 45)? Each step
    of the chain needs the work of the ones before it on THIS garage's
    design page -- a new garage has no stated mission, and the page's gates
    re-lock on every open; step 9 needs step 8's wing in its slot. Steps 1,
    2 and 10 need nothing kept."""
    dp = _dp(g)
    if sid == "screen":
        return bool(g.mission.stated)
    if sid == "section":
        return _live(g) and bool(dp.af.ranked)
    if sid == "plates":
        return _live(g) and dp.af.finished()
    if sid in ("planform", "results"):     # 3 Wing opens with the stated mission
        return _live(g)
    if sid == "fit":                       # a completed run's wing to save
        return _live(g) and dp.wing.spec is not None
    if sid == "build":
        return _chk_fit(g, {}, None)
    return True


def _wrap(s: str, n: int = WRAP_CH) -> list:
    out, line = [], ""
    for word in str(s).split():
        if line and len(line) + 1 + len(word) > n:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def _shell(g):
    """The garage's AeroBO shell while it frames the page on screen, else
    None (the car, airfoil and library pages keep their dark look)."""
    sh = getattr(g, "shell", None)
    return sh if sh is not None and g.page in SHELL_PAGES else None


def _form_of(g):
    """The form the page's work area is bound to (a `cae.form.Form` on the
    shell pages; the rank view has none)."""
    if g.page == "mission":
        return g.mission_page.params
    if g.page == "section":
        return _dp(g).rows()
    return None


def _nav_rect(g, key):
    """A step of the Simulation tree: its row while drawn; else -- its stage
    folded, or scrolled out -- the stage's row when that is in view; else
    the tree pane's body. Never None once the tree has been drawn."""
    import pygame
    nav = _dp(g).nav
    body = getattr(nav, "_rect", None)
    if body is None:
        return None
    for k, y, h in getattr(nav, "_hits", []):
        if k == key:
            return pygame.Rect(body.x, int(y), body.w, int(h))
    stage, full = key.split(".", 1)[0], None
    try:
        from .cae import theme as T
        full = T.TREE_ROW_H
    except Exception:                      # noqa: BLE001
        pass
    for rid, r, kind in getattr(nav, "_node_hits", []):
        if rid == stage and kind == "row" and (full is None or r.h >= full):
            return pygame.Rect(body.x, r.y, body.w, r.h)
    return pygame.Rect(body)


def _plain(s: str) -> str:
    """The dark pages' font may lack the shell's ▸: those pages print '>'."""
    return str(s).replace("\u25b8", ">")


def _wrap_by(s: str, px: int, width) -> list:
    """`s` broken into lines no wider than `px` by `width(text)`."""
    out, line = [], ""
    for word in str(s).split():
        trial = f"{line} {word}".strip()
        if line and width(trial) > px:
            out.append(line)
            line = word
        else:
            line = trial
    if line:
        out.append(line)
    return out


def anchor_rect(g, anchor):
    """Where the page's last draw put the anchored widget (a pygame.Rect),
    or None when it is not on this page.

    ('bar',) is the tool bar on the shell pages and the key bar on the
    airfoil / library pages (none on the car); ('tool', id) a tool button;
    ('nav', key) a tree step with its fallbacks (`_nav_rect`); ('row', key)
    a control of the page's form, only while it is VISIBLE (`Form.rect_of`;
    a plain ParamList: its hit rows)."""
    import pygame
    if not anchor:
        return None
    kind = anchor[0]
    sh = _shell(g)
    if kind == "bar":
        if g.page == "car":
            return None
        if sh is not None:
            W, H = g.screen.get_size()
            return pygame.Rect(sh.geom(W, H).tool)
        u = g.view.ui
        return pygame.Rect(int(12 * u), int(690 * u), int(1256 * u), int(98 * u))
    if kind == "tool":
        return sh.tool_rect(anchor[1]) if sh is not None else None
    if kind == "nav":
        return _nav_rect(g, anchor[1]) if g.page == "section" else None
    if kind == "row":
        pl = _form_of(g)
        if pl is None or pl._rect is None:
            return None
        if hasattr(pl, "rect_of"):
            return pl.rect_of(anchor[1])
        for i, y, h in getattr(pl, "_hits", []):
            if 0 <= i < len(pl.params) and pl.params[i].key == anchor[1]:
                return pygame.Rect(pl._rect.x, int(y), pl._rect.w, int(h))
        return None
    if kind == "list" and g.page == "library":
        lst = getattr(g.lib_page, anchor[1], None)
        return getattr(lst, "_rect", None)
    return None


def _ink(screen, rect) -> int:
    """How much the page has drawn in `rect`: its pixels brighter than any
    panel, grid or border (every channel under INK_LEVEL is background) --
    text, plots, bars, the car. What is off the screen counts as ink."""
    import pygame
    r = pygame.Rect(rect)
    c = r.clip(screen.get_rect())
    if c.w <= 0 or c.h <= 0:
        return r.w * r.h
    m = pygame.mask.from_threshold(screen.subsurface(c), (0, 0, 0, 255),
                                   (INK_LEVEL, INK_LEVEL, INK_LEVEL, 255))
    return r.w * r.h - m.count()


def _arrow_spots(o, box, u) -> list:
    """Where a short arrow may point at the outline `o` from, as (tip,
    (dx, dy) it points along): under it and over it, near either end or
    beside the box, then beside it at its middle."""
    gap, end = int(4 * u), int(24 * u)
    xs = [o.right - end, o.x + end, box.x - end, box.right + end]
    xs = [x for x in xs if o.x + end // 2 <= x <= o.right - end // 2]
    return ([((x, o.bottom + gap), (0, -1)) for x in xs]
            + [((x, o.y - gap), (0, 1)) for x in xs]
            + [((o.right + gap, o.centery), (-1, 0)), ((o.x - gap, o.centery), (1, 0))])


def _arrow_rect(tip, d, u):
    """The px an arrow at `tip` pointing along `d` takes."""
    import pygame
    n, hw = int(ARROW_PX * u), int(7 * u)
    tail = (tip[0] - d[0] * n, tip[1] - d[1] * n)
    x0, x1 = sorted((tip[0], tail[0]))
    y0, y1 = sorted((tip[1], tail[1]))
    return pygame.Rect(x0, y0, x1 - x0, y1 - y0).inflate(2 * hw * abs(d[1]) + 2,
                                                         2 * hw * abs(d[0]) + 2)


class WingTutor:
    """The state machine; `Garage.tutor`. It survives the garage <-> drive
    round trips on `opts.wing_tutor`."""

    def __init__(self, progress=None, start: str | None = None, steps=STEPS,
                 resume: bool = True):
        """`start` the step to begin at; past step 1 it is a CONTINUE, and the
        first garage the tutor meets goes on there (`resume`: its page, the
        hint, the nearest step that can be done). `resume=False`: a tutor
        that is already in a garage (a self-check's)."""
        self.steps = tuple(steps)
        self.progress = progress
        ids = [s.id for s in self.steps]
        self.i = ids.index(start) if start in ids else 0
        self._pending = bool(resume) and self.i > 0
        #  task 45: (slot, wing name) of the wing step 8 fitted, kept in the
        #  progress file, so step 9 can pass in a new garage (`_chk_fit`); a
        #  fresh start forgets an earlier run's
        self.wing = None
        if progress is not None and self.i > 0:
            kw = progress.section(SECTION).get("wing")
            if (isinstance(kw, (list, tuple)) and len(kw) == 2 and kw[0] in ("left", "right", "top")
                    and isinstance(kw[1], str) and kw[1]):
                self.wing = (kw[0], kw[1])
        self.mem: dict = {}
        self.active = True
        self.done = False
        self.hidden = False
        self.flash = ""
        self._flash_n = 0
        self.skipped: list = []
        self._rect = None                  # where draw() last put the box
        #  the dock and the arrow, chosen once per page / navigator step /
        #  tutorial step (`_place`): (layout key, dock, (tip offset, dir) | None)
        self._at = (None, DOCKS[0], None)
        self.arrow_drawn = None            # (tail, tip) of the arrow last drawn
        self._save()

    @property
    def step(self) -> WStep:
        return self.steps[min(self.i, len(self.steps) - 1)]

    def label(self) -> str:
        """'Step 2 of 10: The mission' -- the ONE counter the box, the menu
        and the garage's hint say (task 45: the box read '2/10   1  The
        mission', the step's own number beside the tutorial's)."""
        return f"Step {self.i + 1} of {len(self.steps)}: {self.step.title}"

    # ---------------------------------------------------------------- #
    def _rewind(self, g) -> None:
        """The chain's state lives on ONE garage's pages: a new garage (the
        next launch, a trip to the drive and back) starts with no stated
        mission, and every open of the design page re-locks its section gates.
        A step inside the chain then goes back to where that work starts, so
        the box never points at a shut widget."""
        ids = [s.id for s in self.steps]
        sid = self.step.id
        to = None
        if sid in CHAIN and not g.mission.stated:
            to = "mission" if g.page == "mission" else "open"
        elif g.page == "section" and sid in CHAIN[2:]:
            dp = _dp(g)
            to = ("screen" if not (dp.af.finished() or dp.af.ranked) else
                  "section" if not dp.af.finished() else
                  "plates" if not (dp.plate_locked() or dp.ep.finished()) else None)
        if to in ids and ids.index(to) < self.i and to not in self.skipped:
            self.i, self.mem = ids.index(to), {}
            self.flash, self._flash_n = "the design page started again: from here", 150
            self._save()

    # -- continuing (task 45) -----------------------------------------------------
    def resume(self, g) -> None:
        """Go on at the step this tutor was started at, in garage `g` (the
        garage menu's 'continue at step N', the driving tutorial's pointer):
        on that step's page, the box and the garage's hint saying that step.
        A step whose work is gone with the garage it was done in
        (`can_resume`) goes on from the nearest earlier step that can be
        done, and the box and the hint say so. Task 45: 'continue at step 8'
        showed the box at step 1 ('the design page started again') under a
        hint that still said 'Step 8 of 10'."""
        self._pending = False
        want = self.i
        at = next((j for j in range(want, -1, -1) if can_resume(g, self.steps[j].id)), 0)
        why = ""
        if at != want:
            why = RESUME_WHY.get(self.steps[want].id, RESUME_WHY_CHAIN)
            self.i, self.mem = at, {}
            self.flash, self._flash_n = f"continuing from step {at + 1}: {why}", RESUME_FLASH_N
            self._save()
        self._to_page(g)
        g.hint = (f"wing tutorial: continuing from {self.label()} - {why}" if why
                  else f"wing tutorial: {self.label()}")

    def _to_page(self, g) -> None:
        """Open the page the step is done on, from the car page the garage
        menu is on: the mission page for step 2 (a FLANK slot's: the top slot
        gives way to the left one, as the guided start's), the design page
        for steps 3 to 8 -- as this garage left it when it has the work, else
        opened afresh. Steps 1, 9 and 10 are done from the car page."""
        sid = self.step.id
        flank = "left" if g.sel not in ("left", "right") else None
        if sid == "mission":
            if g.page != "mission":
                g.open_mission(flank)
        elif sid in CHAIN and g.page != "section":
            if _live(g):
                g.page = "section"
            else:
                g.open_section(flank)

    def _tell(self, g) -> None:
        """The garage's hint follows the box when it is the tutorial's own
        and still showing: a step passed, or the tutor went back, and the
        two never name different steps."""
        h = getattr(g, "hint", "")
        alpha = getattr(g, "hint_alpha", lambda: 1.0)()
        if self.active and isinstance(h, str) and h.startswith("wing tutorial:") and alpha > 0.0:
            g.hint = f"wing tutorial: {self.label()}"

    def update(self, g, action=None) -> bool:
        """Once per garage frame, after its events: True when a step passed.
        Several can pass in one frame (a build that is already saved). A
        tutor started to continue goes on at its step first (`resume`)."""
        if self.active and self._pending:
            self.resume(g)
        i0 = self.i
        if self.active:
            self._rewind(g)
        passed = False
        for _ in range(len(self.steps)):
            if not self.active:
                break
            st = self.step
            if not st.check(g, self.mem, action):
                break
            if st.id == "fit" and _dp(g).wing is not None:
                self.wing = (_dp(g).wing.key, _dp(g).wing.spec.name)
            self.flash, self._flash_n = f"done: {st.title}", 150
            self.advance()
            passed = True
        if self.i != i0:
            self._tell(g)
        if self._flash_n > 0:
            self._flash_n -= 1
        self._reveal(g)
        return passed

    def _reveal(self, g) -> None:
        """A step anchored on a control the view laid out below the fold asks
        the shell to scroll it in -- ONCE, when the step first finds it laid
        out, so the player can scroll away again (PLAN §4.2)."""
        st = self.step
        if not self.active or st.anchor[:1] != ("row",) or self.mem.get("_shown"):
            return
        form = _form_of(g) if _shell(g) is not None else None
        gm = form.geom(st.anchor[1]) if hasattr(form, "geom") else None
        if gm is None:
            return                         # not laid out on the view showing
        self.mem["_shown"] = True
        if not gm.visible:
            form.reveal = st.anchor[1]

    def advance(self, skipped: bool = False) -> None:
        if skipped and self.step.id not in self.skipped:
            self.skipped.append(self.step.id)
        self.i += 1
        self.mem = {}
        if self.i >= len(self.steps):
            self.i = len(self.steps) - 1
            self.active, self.done = False, True
        self._save()

    def skip(self) -> None:
        self.flash, self._flash_n = f"skipped: {self.step.title}", 150
        self.advance(skipped=True)

    def end(self) -> None:
        self.active = False
        self._save()

    def _save(self) -> None:
        if self.progress is None:
            return
        sec = self.progress.section(SECTION)
        sec["done"] = bool(sec.get("done")) or self.done
        sec["step"] = None if self.done else self.step.id
        sec["skipped"] = list(self.skipped)
        sec["wing"] = list(self.wing) if self.wing else None
        self.progress.save(SECTION)

    # -- the garage's pause menu ------------------------------------------------
    def menu_rows(self) -> list:
        #  task 45: the skip row names the step by its number only -- with the
        #  title ('... (The airfoil: take one, fit it)', 64 characters) it was
        #  the menu's widest row, and the garage's help beside it wrapped off
        #  the bottom of the screen. The box, under the menu, has the title.
        if not self.active:
            return []
        return [(f"Wing tutorial: skip step {self.i + 1} of {len(self.steps)}", "wt_skip"),
                ("Wing tutorial: " + ("show" if self.hidden else "hide") + " the box", "wt_hide"),
                ("Wing tutorial: end it", "wt_end")]

    def hit(self, pos) -> bool:
        """Is `pos` on the box? The garage gives it the click, not the page
        under it."""
        return self._rect is not None and pos is not None and self._rect.collidepoint(pos)

    def menu_action(self, action: str) -> bool:
        if action == "wt_skip":
            self.skip()
        elif action == "wt_hide":
            self.hidden = not self.hidden
        elif action == "wt_end":
            self.end()
        else:
            return False
        return True

    # -- drawing ----------------------------------------------------------------
    def lines(self, wrap=_wrap, g=None) -> list:
        """(text, colour-name) rows of the box: 'head', 'text', 'do', 'flash',
        'dim'. `g` the garage a step's press may depend on (step 1's)."""
        rows = []
        if self.flash and self._flash_n > 0:
            #  wrapped (task 45): a resume's note is longer than a 'done: ...'
            rows += [(ln, "flash") for ln in wrap(self.flash)]
        if self.active:
            st = self.step
            rows.append((f"WING TUTORIAL   {self.label()}", "head"))
            rows += [(ln, "text") for ln in wrap(st.text)]
            rows += [(ln, "do") for ln in wrap("> " + do_text(st, g))]
            rows += [(ln, "dim") for ln in wrap("H hides this box. ESC / OPTIONS on the "
                                                 "car: skip or end")]
        return rows

    def _rows(self, g, w: int):
        """The rows at a box `w` px wide on screen, wrapped with the font that
        draws them, and the box's height for them."""
        from . import garage_ui as ui
        u = g.view.ui
        rows = self.lines(lambda s: ui._wrap_px(g.text, _plain(s), w - 2 * int(10 * u), BOX_SIZE), g)
        return rows, min(int(BOX_H_MAX * u), len(rows) * int(17 * u) + int(16 * u))

    @staticmethod
    def _dock_rect(dock, h: int, u: float):
        import pygame
        x, v, w = dock
        top, foot = int(DOCK_TOP * u), int(DOCK_FOOT * u)
        y = {"top": top, "foot": foot - h, "mid": (top + foot - h) // 2}[v]
        return pygame.Rect(int((1268 - w if x < 0 else x) * u), y, int(w * u), h)

    def _place(self, g, a):
        """Dock the box and aim the arrow, on the page as drawn under them:
        the dock with the least ink under it that keeps off the anchored
        widget (DOCKS' order breaks a tie), then the arrow's spot beside the
        widget with the least ink under it that keeps off the box."""
        u = g.view.ui
        best, hs = None, {}
        for d in DOCKS:
            if d[2] not in hs:
                hs[d[2]] = self._rows(g, int(d[2] * u))[1]
            r = self._dock_rect(d, hs[d[2]], u)
            if a is not None and r.colliderect(a.inflate(int(12 * u), int(12 * u))):
                continue
            n = _ink(g.screen, r)
            if best is None or n < best[0]:
                best = (n, d, r)
        dock, r = (best[1], best[2]) if best else (DOCKS[0], None)
        aim = None
        if a is not None and r is not None:
            o, scr, cands = a.inflate(4, 4), g.screen.get_rect(), []
            for tip, d in _arrow_spots(o, r, u):
                ar = _arrow_rect(tip, d, u)
                if scr.contains(ar) and not ar.colliderect(r.inflate(8, 8)):
                    cands.append((_ink(g.screen, ar), len(cands), (tip[0] - o.x, tip[1] - o.y), d))
            if cands:
                _, _, off, d = min(cands)
                aim = (off, d)
        return dock, aim

    def layout(self, g) -> dict:
        """How the box sits on this page: its base rect (x, y, w, h; it
        grows UP by at most `BOX_GROW` x scale), padding, line pitch, the
        wrapped rows and the text measure the wrap used. The shell pages
        (light, the shell's `tutor_box`, drawn with the CAE theme's text) and
        the dark pages (`BOX_CAR` x the page's scale, the garage's text)."""
        import pygame
        if _shell(g) is not None:
            from .cae import theme as T
            W, H = g.screen.get_size()
            base = pygame.Rect(g.shell.geom(W, H).tutor_box)
            sc, size = T.S, 12

            def measure(s, c):
                return T.text_w(s, "sans", size, bold=(c == "head"))
        else:
            sc, size = g.view.ui, 13
            base = pygame.Rect(*(int(v * sc) for v in BOX_CAR))

            def measure(s, c):
                return g.text.width(_plain(s), size, bold=(c == "head"))
        pad = int(round(10 * sc))
        inner = base.w - 2 * pad
        rows = self.lines(lambda s: _wrap_by(s, inner, lambda t: measure(t, "text")), g)
        return dict(base=base, pad=pad, lh=int(round(17 * sc)), rows=rows, size=size,
                    measure=measure, inner=inner, grow=int(round(BOX_GROW * sc)),
                    top=int(round(8 * sc)), shell=_shell(g) is not None)

    def draw(self, g) -> None:
        """The box and an outline round the anchored widget. On the shell
        pages it is light, over the work area's lower right (the shell's
        `tutor_box`), with a line to the widget; on the dark pages it docks
        where the page under it has the least on it, with a short arrow
        beside the widget (task 45: a line ran to it from the box, across the
        page). Nothing when hidden."""
        import pygame
        self._rect, self.arrow_drawn = None, None   # a hidden or empty box takes nothing
        if self.hidden or not self.lines():
            return
        if _shell(g) is None:
            self._draw_dark(g)
            return
        from .cae import theme as T
        from .cae import widgets as CW
        lay = self.layout(g)
        rows, base, lh = lay["rows"], lay["base"], lay["lh"]
        h = min(base.h + lay["grow"], len(rows) * lh + 2 * lay["top"])
        r = pygame.Rect(base.x, base.bottom - h, base.w, h)
        CW.shadow(g.screen, r)
        g.screen.fill(T.WELL, r)
        pygame.draw.rect(g.screen, T.RULE, r, 1)
        g.screen.fill(T.ACCENT, (r.x, r.y, r.w, max(2, int(round(3 * T.S)))))
        cols = {"head": T.ACCENT, "text": T.INK, "do": T.GOOD, "flash": T.GOOD,
                "dim": T.INK_FAINT}
        outline = T.AMBER_LABEL
        self._rect = r
        y = r.y + lay["top"]
        for s, c in rows:
            if y + lh > r.bottom:
                break
            T.text(g.screen, s, r.x + lay["pad"], y, "sans", lay["size"], cols[c], bold=(c == "head"))
            y += lh
        a = anchor_rect(g, self.step.anchor) if self.active else None
        if a is not None:
            pygame.draw.rect(g.screen, outline, a.inflate(4, 4), 2)
            start = (r.x, r.y + int(lay["top"] + 2))
            if a.right < r.x:
                end = (a.right + 2, a.centery)
            else:                          # above the box (the tool bar, a control)
                end = (min(max(start[0], a.left), a.right),
                       a.bottom + 2 if a.bottom <= r.y else a.top - 2)
            pygame.draw.line(g.screen, outline, start, end, 1)
        self._reveal(g)

    def _draw_dark(self, g) -> None:
        """The box on the dark pages (car, airfoil, library), task 45's."""
        import pygame
        from . import garage_ui as ui
        u = g.view.ui
        a = anchor_rect(g, self.step.anchor) if self.active else None
        key = (g.page, _cur(g), self.step.id, self.active, a is None, g.screen.get_size())
        if key != self._at[0]:
            self._at = (key, *self._place(g, a))
        _, dock, aim = self._at
        rows, h = self._rows(g, int(dock[2] * u))
        r = ui.panel(g.screen, self._dock_rect(dock, h, u), alpha=235, accent=True)
        self._rect = r
        cols = {"head": C_BOX_ACCENT, "text": ui.C_TEXT, "do": ui.C_OK,
                "flash": ui.C_OK, "dim": ui.C_DIM}
        lh, y = int(17 * u), r.y + int(8 * u)
        for s, c in rows:
            if y + lh > r.bottom:
                break
            g.text.blit(g.screen, s, r.x + int(10 * u), y, BOX_SIZE, cols[c], bold=(c == "head"))
            y += lh
        if a is None:
            return
        o = a.inflate(4, 4)
        pygame.draw.rect(g.screen, C_BOX_ACCENT, o, 2)
        if aim is not None:
            (ox, oy), (dx, dy) = aim
            n, hl, hw = int(ARROW_PX * u), int(12 * u), int(7 * u)
            tip = (o.x + ox, o.y + oy)
            tail, base = (tip[0] - dx * n, tip[1] - dy * n), (tip[0] - dx * hl, tip[1] - dy * hl)
            pygame.draw.line(g.screen, C_BOX_ACCENT, tail, base, max(2, int(3 * u)))
            pygame.draw.polygon(g.screen, C_BOX_ACCENT,
                                [tip, (base[0] - dy * hw, base[1] + dx * hw),
                                 (base[0] + dy * hw, base[1] - dx * hw)])
            self.arrow_drawn = (tail, tip)


def saved_state(progress) -> dict:
    sec = progress.section(SECTION) if progress is not None else {}
    ids = [s.id for s in STEPS]
    step = sec.get("step")
    return dict(done=bool(sec.get("done")), step=step if step in ids else None)


def menu_row(progress, tutor=None):
    """The garage menu's row to start / continue it, or None while it runs."""
    if tutor is not None and tutor.active:
        return None
    sv = saved_state(progress)
    if sv["step"] and sv["step"] != STEPS[0].id:
        i = [s.id for s in STEPS].index(sv["step"])
        return (f"Wing-design tutorial: continue at step {i + 1}", "wt_resume")
    return ("Wing-design tutorial: a guided first wing" + (" (done)" if sv["done"] else ""),
            "wt_start")


# ==================================================================== #
#  SELF-CHECK: every predicate against a real (headless) garage        #
# ==================================================================== #
def self_check(verbose: bool = True) -> bool:
    import math
    import os
    import tempfile
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame
    from .aero.library import Library
    from . import garage as grg
    from . import aerobo_models as am
    from .garage_ui import _wrap_px as ui_wrap
    from .progress import Progress
    ok = True
    #  every run below is a captured AeroBO run replayed through the same job
    #  path (PLAN2 D12): no engine, no XFOIL, no timing
    am.use_fixtures()

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}" + (f": {msg}" if msg else ""))

    tmp = tempfile.mkdtemp(prefix="carsim_wingtut_")
    lib = Library(os.path.join(tmp, "library"), use_xfoil=False)
    g = grg.Garage((1280, 800), grg.CarBuild(), headless=True, lib=lib)
    g.aerobo_dir = None                    # a check writes nothing under runs/
    g.export_dir = os.path.join(tmp, "export")
    prog = Progress(os.path.join(tmp, "progress.json"))
    t = WingTutor(prog)
    g.tutor = t

    def key(k, mod=0):
        return g._handle(pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod))

    def frame(a=None):
        """One garage frame's worth: draw (the anchors live in the draw), then
        the tutor, as Garage.frame orders them."""
        g._draw_page()
        return t.update(g, a)

    OLD = pygame.Rect(748, 468, 520, 212)     # the one place the box had before task 45
    placed = {}

    def place(tag):
        """Draw the box over the page as it stands, then the page again, and
        keep (box, anchor outline, arrow, ink under the box, ink under the
        old place) -- measured on the page with no box on it."""
        g._draw_page()
        t.draw(g)
        a = anchor_rect(g, t.step.anchor)
        box, arw = t._rect, t.arrow_drawn
        g._draw_page()
        placed[tag] = (box, a.inflate(4, 4) if a is not None else None, arw,
                       _ink(g.screen, box), _ink(g.screen, OLD))

    import re
    #  the page each step is done on (task 45: a continued step opens it)
    PAGE = dict(open="car", mission="mission", build="car", drive="car",
                **{k: "section" for k in CHAIN[1:]})

    def nums(g_, t_):
        """(the box's step, the garage hint's step): 'Step n of 10' in each."""
        head = " ".join(s_ for s_, c in t_.lines() if c == "head")
        a, b = re.search(r"Step (\d+) of", head), re.search(r"Step (\d+) of", g_.hint or "")
        return (int(a.group(1)) if a else None, int(b.group(1)) if b else None)

    rejoined = []

    def rejoin():
        """End the tutorial and continue it from the garage menu (the car
        page) in the garage that did the work, before the step's own press:
        it goes on at the same step, on that step's page (the design page as
        it was left), the box and the hint saying that step. A car-page step
        puts the walk back on the page it was on."""
        nonlocal t
        want, page0 = t.step.id, g.page
        t.end()
        g.page, g.progress = "car", prog
        g._menu_action("wt_resume")
        g.progress = None
        t = g.tutor
        g._draw_page()
        t.update(g)
        rejoined.append((want, t.step.id, g.page, nums(g, t)))
        if g.page == "car":
            g.page = page0

    ids = [s.id for s in STEPS]
    rep("ten steps, the garage's own order, each with a predicate and an anchor",
        ids == ["open", "mission", "screen", "section", "plates", "planform", "results",
                "fit", "build", "drive"] and all(callable(s.check) for s in STEPS)
        and all(s.anchor for s in STEPS), " > ".join(ids))
    seen = {}
    # 1 the car page: nothing yet; D opens the mission
    seen["open"] = (not frame(), )
    place("car")
    key(pygame.K_d)
    seen["open"] += (frame() and t.step.id == "mission",)
    # 2 the mission: stated by ENTER
    rejoin()
    seen["mission"] = (not frame(), anchor_rect(g, t.step.anchor) is not None)
    place("mission")
    key(pygame.K_RETURN)
    seen["mission"] += (frame() and t.step.id == "screen" and g.page == "section",)
    dp = g.design_page
    # 3 screening
    rejoin()
    seen["screen"] = (not frame(), anchor_rect(g, t.step.anchor) is not None)
    place("screening")
    key(pygame.K_l)
    #  L starts a LIVE screen (PLAN T1): the garage's frame pumps it a unit
    #  or two at a time; the check runs it out in one go
    g.runs.run_all()
    seen["screen"] += (frame() and t.step.id == "section",)
    # 4 highlight the winner on the ranking, take it with F
    rejoin()
    seen["section"] = (not frame(),)
    dp.nav.select("af.rank")
    dp.rank_list.idx = 0
    seen["section"] += (not frame(),)                 # highlighted, not yet taken
    key(pygame.K_f)
    seen["section"] += (frame() and t.step.id == "plates",)
    # 5 the plates: fly them flat
    rejoin()
    seen["plates"] = (not frame(),)
    place("end plates")
    dp.ep.decline()
    seen["plates"] += (frame() and t.step.id == "planform",)
    # 6 the planform: seen, then moved on
    rejoin()
    dp.nav.select("w.type")
    seen["planform"] = (not frame(),)
    dp.nav.select("w.box")
    seen["planform"] += (not frame(), anchor_rect(g, t.step.anchor) is not None)
    dp.nav.select("w.solver")
    seen["planform"] += (frame() and t.step.id == "results",)
    # 7 the results: shut until a run completes (AeroBO's gate), open after
    rejoin()
    seen["results"] = (not frame(), not dp.nav.select("r.summary"))
    #  the side job opens on side force; the captured side-wing fixture was
    #  flown at AeroBO's default, efficiency -- picked here, as a player may
    dp.wing.type_params.param("obj").set("efficiency")
    key(pygame.K_o)                                   # Run, on 3 Wing > Solver
    g.runs.run_all()
    sel = dp.nav.select("r.summary")
    seen["results"] += (sel and frame() and t.step.id == "fit",)
    # 8 fit: S saves the wing into the slot
    rejoin()
    seen["fit"] = (not frame(),)
    key(pygame.K_s)
    seen["fit"] += (frame() and t.step.id == "build",
                    g.build.left.wing == dp.wing.spec.name)
    # 9 the build: ESC to the car, L, S (the prompt), a name
    rejoin()
    seen["build"] = (not frame(),)
    key(pygame.K_ESCAPE)
    key(pygame.K_ESCAPE)
    key(pygame.K_l)
    seen["build"] += (not frame(), anchor_rect(g, t.step.anchor) is not None)
    g._save_build_quick()
    seen["build"] += (frame() and t.step.id == "drive",)
    # 10 drive
    rejoin()
    seen["drive"] = (not frame(None),)
    key(pygame.K_ESCAPE)
    seen["drive"] += (frame("drive") and t.done and not t.active,)
    for sid in ids:
        v = seen.get(sid, ())
        rep(f"step {sid}: false before the garage state, true after", v and all(v), str(v))
    sv = saved_state(Progress(prog.path))
    rep("done and saved in the progress file", sv["done"] and sv["step"] is None, str(sv))
    # task 45: 'continue at step N' goes on AT step N -- its page, the box and
    # the hint -- in the garage that did the work (steps 2-10, before each press)
    bad = [f"{w}: {got} on {pg}, box/hint {n}" for w, got, pg, n in rejoined
           if got != w or pg != PAGE[w] or n != (ids.index(w) + 1,) * 2]
    rep("continued in the garage that did the work, every step goes on there: its page "
        "(the design page as it was left), the box's step = the hint's step",
        len(rejoined) == 9 and not bad, "; ".join(bad) or f"{len(rejoined)} steps")
    # ... and in a NEW garage (the next launch): a step whose work went with
    # the old garage goes on from the nearest one that can be done, and says so
    bad, fell = [], 0
    for j, sid in enumerate(ids):
        pj = Progress(os.path.join(tmp, f"p_res_{sid}.json"))
        pj.section(SECTION).update(step=sid, offered=True)
        pj.save(SECTION)
        gj = grg.Garage((1280, 800), grg.CarBuild(), headless=True, lib=lib)
        gj.progress = pj
        gj._menu_action("wt_resume")
        gj._draw_page()
        tj = gj.tutor
        tj.update(gj)
        gj._draw_page()
        tj.draw(gj)
        inner = tj._rect.w - 2 * int(10 * gj.view.ui) if tj._rect is not None else 0
        widest = max((gj.text.width(s_, BOX_SIZE, bold=(c == "head")) for s_, c in tj.lines(
            lambda s_: ui_wrap(gj.text, s_, inner, BOX_SIZE), gj)), default=0)
        at = j if sid in ("open", "mission", "drive") else 1
        flash = " ".join(s_ for s_, c in tj.lines() if c == "flash")
        said = (at == j or (f"continuing from step {at + 1}: " in flash
                            and gj.hint.startswith(f"wing tutorial: continuing from Step {at + 1} of")))
        if (tj.i != at or gj.page != PAGE[ids[at]] or nums(gj, tj) != (at + 1,) * 2 or not said
                or saved_state(Progress(pj.path))["step"] != ids[at] or widest > inner):
            bad.append(f"{sid}: step {tj.i + 1} on {gj.page}, box/hint {nums(gj, tj)}, '{flash}', "
                       f"widest line {widest} of {inner} px")
        fell += at != j
    rep("continued in a new garage: steps 1, 2 and 10 go on there; 3-8 (their design page's "
        "work is gone) and 9 (no wing in its slot) from step 2 on the mission page, saying why; "
        "the box's step = the hint's step = the saved step; the note fits the box",
        not bad and fell == 7, "; ".join(bad) or f"{fell} of 10 went back to step 2")
    # step 9 in a new garage with step 8's wing still in its slot: it stays at
    # 9 and saving the build passes it (before, no design page wing: never)
    fit_w = g.build.left.wing
    p9 = Progress(os.path.join(tmp, "p_res_build_kept.json"))
    p9.section(SECTION).update(step="build", offered=True, wing=["left", fit_w])
    p9.save(SECTION)
    b9 = grg.CarBuild()                     # the wing, on a car not saved as a build yet
    b9.left.wing = b9.right.wing = fit_w
    b9.left.inc_deg = b9.right.inc_deg = 3.0
    g9 = grg.Garage((1280, 800), b9, headless=True, lib=lib)
    g9.progress = p9
    g9._menu_action("wt_resume")
    g9._draw_page()
    t9_ = g9.tutor
    stayed = not t9_.update(g9) and t9_.step.id == "build" and g9.page == "car" \
        and nums(g9, t9_) == (9, 9)
    g9._save_build_quick()
    g9._draw_page()
    rep("step 9 continued in a new garage with step 8's wing in its slot stays at 9, and "
        "saving the build passes it",
        stayed and t9_.update(g9) and t9_.step.id == "drive", f"wing {fit_w}, now {t9_.step.id}")
    # the driving tutorial's pointer: a tutor made with the saved step, met by
    # the garage's first frame, goes on as the menu's row does
    pd = Progress(os.path.join(tmp, "p_res_pointer.json"))
    pd.section(SECTION).update(step="fit", offered=True)
    pd.save(SECTION)
    gd = grg.Garage((1280, 800), grg.CarBuild(), headless=True, lib=lib)
    gd.progress = pd
    gd.tutor = WingTutor(pd, start=saved_state(pd)["step"])
    was = gd.tutor.step.id
    gd._draw_page()
    gd.tutor.update(gd)
    rep("the driving tutorial's pointer continues as the menu does (step 8 -> 2, mission page)",
        was == "fit" and gd.tutor.step.id == "mission" and gd.page == "mission"
        and nums(gd, gd.tutor) == (2, 2), f"{was} -> {gd.tutor.step.id} on {gd.page}")
    # task 45: one counter, the pages' words, step 1's warning only for a wing
    t5 = WingTutor(None, start="mission")
    head = [s_ for s_, c in t5.lines() if c == "head"]
    rep("one counter: the head says 'Step n of N: ...', no title has a number of its own",
        t5.label() == "Step 2 of 10: The mission"
        and head == ["WING TUTORIAL   Step 2 of 10: The mission"]
        and t5.menu_rows()[0][0] == "Wing tutorial: skip step 2 of 10"
        and not any(ch.isdigit() for s_ in STEPS for ch in s_.title), f"{head}")
    g0 = grg.Garage((1280, 800), grg.CarBuild(), headless=True, lib=lib)
    words = " ".join(f"{s_.title} {s_.text} {do_text(s_)} {do_text(s_, g)} {do_text(s_, g0)}"
                     for s_ in STEPS)
    rep("the steps say 'airfoil' as the pages do, never 'aerofoil'; 'lap' is an estimate "
        "for comparing wings",
        "aerofoil" not in words.lower() and "airfoil" in words.lower()
        and "'lap' is an estimate used to compare wings, not your lap time" in STEPS[1].text)
    rep("step 1 says the design replaces the slot's wing only when the slot has one",
        "replaces" not in do_text(STEPS[0], g0) and "replaces" not in do_text(STEPS[0])
        and bool(g.build.left.wing) and "replaces" in do_text(STEPS[0], g),
        f"empty: '{do_text(STEPS[0], g0)}'")
    # the box off the page's text and off what it points at; a short arrow beside it
    u = g.view.ui
    bad = []
    #  the dark pages only: on the shell pages the box sits at the shell's
    #  tutor_box with a line to the widget (WingLab's look)
    placed = {k_: v_ for k_, v_ in placed.items() if k_ == "car"}
    for tag, (box, o, arw, n_new, n_old) in placed.items():
        if box is None or n_new > n_old or (o is not None and box.colliderect(o)):
            bad.append(f"{tag}: box {tuple(box) if box else None} ink {n_new} (old {n_old})")
        if o is not None:
            (x0, y0), (x1, y1) = arw or ((0, 0), (999, 999))
            off = max(o.x - x1, x1 - o.right, 0, o.y - y1, y1 - o.bottom)
            span = pygame.Rect(min(x0, x1), min(y0, y1), abs(x1 - x0) + 1, abs(y1 - y0) + 1)
            if (arw is None or math.hypot(x1 - x0, y1 - y0) > 60 * u or off > 8 * u
                    or span.colliderect(box)):
                bad.append(f"{tag}: arrow {arw} by {tuple(o)}")
    rep("on the car page the box docks where the page has no more under it than the old place, off the "
        "widget it points at; the arrow is short (<= 60 px), beside the widget, off the box",
        len(placed) == 1 and not bad,
        "; ".join(bad) or ", ".join(f"{k} {tuple(v[0])} ink {v[3]} (old {v[4]})"
                                    for k, v in placed.items()))
    # the first-wing choice's guided start: D was the press step 1 asks for
    g3 = grg.Garage((1280, 800), grg.CarBuild(), headless=True, lib=lib)
    g3.progress = Progress(os.path.join(tmp, "p_first.json"))
    g3.sel = "top"
    g3._handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d, mod=0))
    asked = g3.menu.open and g3.page == "car"
    g3._handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0))
    g3._draw_page()
    t3_ = g3.tutor
    ran = t3_ is not None and t3_.update(g3)
    rep("'Guided first wing' on the first D opens the mission (a flank's), and step 1 "
        "passes by itself",
        asked and ran and g3.page == "mission" and g3.sel == "left" and t3_.step.id == "mission"
        and t3_.label() == "Step 2 of 10: The mission",
        f"asked {asked}, page {g3.page}, slot {g3.sel}, step {t3_.step.id if t3_ else None}")
    # the box takes its own clicks, and every line fits it
    # the box takes its own clicks, and every line of every step fits it, on
    # the shell pages (the CAE theme's text) and on the dark ones
    widest, inner, where = 0, 0, ""
    for pg in ("section", "car"):
        g.page = pg
        for st in STEPS:
            lay = WingTutor(None, start=st.id).layout(g)
            wmax = max((lay["measure"](s, c) for s, c in lay["rows"]), default=0)
            if wmax - lay["inner"] >= widest - inner:
                widest, inner, where = wmax, lay["inner"], f"{pg} page, step {st.id}"
    t7 = WingTutor(None, start="section")
    g.page = "section"
    g._draw_page()
    t7.draw(g)
    bx = t7._rect
    rep("the box takes its own clicks; every line fits inside it",
        bx is not None and t7.hit(bx.center) and not t7.hit((0, 0)) and widest <= inner,
        f"widest {widest} px in {inner} ({where})")
    # T3: the car page's box is BOX_CAR; the shell pages' is the shell's own
    # tutor box, the same 520 px wide, growing up from its bottom edge
    tb = g.shell.geom(*g.screen.get_size()).tutor_box
    on_shell = bx is not None and (bx.x, bx.w, bx.bottom) == (tb.x, tb.w, tb.bottom)
    g.page = "car"
    t7.draw(g)
    u = g.view.ui
    cb = t7._rect
    on_car = cb is not None and (cb.x, cb.w, cb.bottom) == (
        int(BOX_CAR[0] * u), int(BOX_CAR[2] * u), int(BOX_CAR[1] * u) + int(BOX_CAR[3] * u))
    rep("the box sits at BOX_CAR on the car and at the shell's tutor box on the design pages",
        BOX == BOX_CAR and on_shell and on_car and tb.w == BOX_CAR[2],
        f"shell {tuple(bx) if bx else None} in {tuple(tb)}, car {tuple(cb) if cb else None}")
    g.page = "section"
    g._draw_page()
    t7.draw(g)
    g.tutor = t7
    before_ = (dp.nav.current(), dp.focus, dp.rank_list.idx)
    g._handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=bx.center))
    rep("a click on the box does not reach the page under it",
        (dp.nav.current(), dp.focus, dp.rank_list.idx) == before_, str(before_))
    g.tutor = t
    # the design page re-opened for another operating point / a new garage:
    # the tutor walks back to where the work starts. A side wing's job is
    # SIDE whatever the circuit, so what moves it is the rest of the car (the
    # top wing it is designed beside)
    t8 = WingTutor(None, start="planform", resume=False)
    inc_was = g.build.top.inc_deg
    g.build.top.inc_deg = inc_was + 1.0
    g.open_section("left")                 # the rest of the car moved: the session is cleared
    g._draw_page()
    t8.update(g)
    back1 = t8.step.id
    g.build.top.inc_deg = inc_was
    g2 = grg.Garage((1280, 800), grg.CarBuild(), headless=True, lib=lib)
    t9 = WingTutor(None, start="results", resume=False)
    t9.update(g2)
    rep("a design page re-opened for another operating point, or a new garage, rewinds the tutor",
        back1 == "screen" and t9.step.id == "open", f"{back1} / {t9.step.id}")
    # an edit after fitting un-fits it (the predicate reads the live state)
    w = dp.wing
    w.dirty = True
    rep("an unsaved edit is not 'fitted'", not _chk_fit(g, {}, None))
    w.dirty = False
    # the box draws, and hides
    t2 = WingTutor(None)
    g.page = "car"
    t2.draw(g)
    t2.hidden = True
    before = pygame.image.tostring(g.screen, "RGB")
    t2.draw(g)
    rep("H hides the box (nothing drawn)", pygame.image.tostring(g.screen, "RGB") == before)
    rep("the menu offers skip / hide / end while it runs, start / continue when not",
        [a for _, a in t2.menu_rows()] == ["wt_skip", "wt_hide", "wt_end"]
        and menu_row(None, t2) is None and menu_row(None)[1] == "wt_start")
    t2.menu_action("wt_skip")
    rep("skip moves on and is remembered", t2.step.id == "mission" and t2.skipped == ["open"])
    t2.menu_action("wt_end")
    rep("end stops it", not t2.active and t2.lines() and not any(c == "head" for _, c in t2.lines()))
    p3 = Progress(os.path.join(tmp, "p3.json"))
    t3 = WingTutor(p3, start="results")
    t3.end()
    rep("an ended tutorial offers to continue where it stopped",
        menu_row(Progress(p3.path)) == ("Wing-design tutorial: continue at step 7", "wt_resume"))

    # T4 / T5: the chain walked again at both parity sizes with the box held
    # at its FULL height; at each step, on the page it is passed on, the
    # anchor resolves and the box never covers it
    def anchor_walk(size):
        libw = Library(os.path.join(tmp, f"library_{size[0]}"), use_xfoil=False)
        gw = grg.Garage(size, grg.CarBuild(), headless=True, lib=libw)
        gw.aerobo_dir = None
        gw.export_dir = os.path.join(tmp, "export")
        tw = WingTutor(None)
        tw.lines = lambda wrap=_wrap, g=None: [("x", "text")] * 40      # the box as tall as it gets
        gw.tutor = tw
        seen_ = {}

        def kw(k):
            gw._handle(pygame.event.Event(pygame.KEYDOWN, key=k, mod=0))

        def fr():
            """Garage.frame's order (update, page, box), twice: a control
            the step asks to scroll in is on screen by the second."""
            for _ in range(2):
                tw.update(gw)
                gw._draw_page()
                tw.draw(gw)
            sid = tw.step.id
            seen_.setdefault(sid, (gw.page, anchor_rect(gw, tw.step.anchor), tw._rect))

        fr()
        kw(pygame.K_d)
        fr()
        kw(pygame.K_RETURN)
        fr()
        kw(pygame.K_l)
        gw.runs.run_all()
        fr()
        gw.design_page.af.use_ranked(0)
        kw(pygame.K_f)
        fr()
        gw.design_page.ep.decline()
        fr()
        gw.design_page.nav.select("w.box")
        fr()
        gw.design_page.nav.select("w.solver")
        fr()
        gw.design_page.wing.type_params.param("obj").set("efficiency")   # the fixture's
        kw(pygame.K_o)
        gw.runs.run_all()
        gw.design_page.nav.select("r.summary")
        fr()
        kw(pygame.K_s)
        fr()
        kw(pygame.K_ESCAPE)
        kw(pygame.K_ESCAPE)
        kw(pygame.K_l)
        fr()
        gw._save_build_quick()
        fr()
        return gw, seen_

    shell_steps = ("mission", "screen", "section", "plates", "planform", "results", "fit")
    hits, missing, bad = [], [], []
    for size in ((1280, 800), (1600, 1000)):
        gw, seen_ = anchor_walk(size)
        S_ = gw.shell.geom(*size).tutor_box.w / BOX_CAR[2]
        for sid in [s_.id for s_ in STEPS]:
            if sid not in seen_:
                missing.append(f"{size[0]}: {sid} never reached")
                continue
            pg, a, box = seen_[sid]
            if a is None:
                if sid in shell_steps:
                    missing.append(f"{size[0]}: {sid} on {pg}")
                continue
            full = box is not None and box.h == (         # the dark pages' docked box: BOX_H_MAX
                int(round((BOX_CAR[3] + BOX_GROW) * S_)) if pg in SHELL_PAGES
                else int(BOX_H_MAX * gw.view.ui))
            if box is None or not full or a.colliderect(box):
                bad.append(f"{size[0]}: {sid} {tuple(a)} vs box {tuple(box) if box else None}")
            hits.append(sid)
        if size == (1600, 1000):
            t5 = [sid for sid in shell_steps
                  if sid in seen_ and seen_[sid][0] in SHELL_PAGES and seen_[sid][1] is not None]
    rep("T4: at 1280x800 and 1600x1000 no anchor sits under the box at its full height",
        not bad and len(hits) >= 2 * len(shell_steps), "; ".join(bad) or
        f"{len(hits)} anchors clear of a {BOX_CAR[3] + BOX_GROW} px box")
    rep("T5: every shell step's anchor resolves on the 1600x1000 shell",
        not missing and t5 == list(shell_steps), "; ".join(missing) or " ".join(t5))
    am.use_engine()
    if verbose:
        print(f"wing_tutorial self-check: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
