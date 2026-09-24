"""drive/wing_tutorial.py -- the wing-design tutorial (task 24).

A guided pass through the garage's own gated DESIGN navigator, in the order
the garage enforces it: the mission -> the aerofoil (the screening weights,
the ranking, fitting the section) -> the end plates -> the wing's planform
(the design box) -> the results -> the wing fitted to the car -> the car
saved as a build -> drive it (the pre-race screen's Build > PICK offers it).

Ten data-driven steps `WStep(id, title, text, do, check, anchor)`. `check(g,
mem, action)` is a predicate on the live `Garage` (its page, the mission,
the design page's two section models and its wing, the build, the library)
and the action the garage's frame returned; `anchor` names the widget the
hint points at -- a navigator step (`('nav', 'af.screen')`), a row of the
page's form (`('row', 'go')`), a library list or the key bar -- and
`anchor_rect` finds it where the page's last draw put it (every garage
widget keeps its own `_rect` / `_hits` for the mouse). The box says, in
plain words, what the step is and what each number means, and what to press.

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
BOX = (748, 468, 520, 212)          # px at 1280x800: over the page's lower right
WRAP_CH = 66                        # characters a line in the box
C_BOX_ACCENT = (217, 206, 85)       # the key colour: the hint and the anchor outline


@dataclass(frozen=True)
class WStep:
    id: str
    title: str
    text: str                        # what this is, in plain words
    do: str                          # what to press
    check: object                    # (g, mem, action) -> bool
    anchor: tuple = ()               # ('nav', key) | ('row', key) | ('list', name) | ('bar',)


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
    """The designed wing is saved and it is the one in the slot."""
    w = _dp(g).wing
    if w is None:
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


STEPS = (
    WStep("open", "Design a wing",
          "A wing turns moving air into force. DOWNFORCE presses the tyres into the "
          "road; DRAG holds the car back on the straights. A FLANK wing stands on "
          "the side of the car and pushes SIDEWAYS, into the corner -- grip the tyres "
          "do not have to find -- and folds away on the straights. The garage designs "
          "one in order: mission, section, wing, results.",
          "Select a flank slot (1 = left) and press D (L3 on the pad). D redesigns the "
          "wing already in that slot: to keep one of yours, clear the slot first "
          "(library N: new wing), or rename it at step 7.",
          _chk_open, ("bar",)),
    WStep("mission", "1  The mission",
          "What is the wing FOR? A circuit and a surface. 'lap' is the best this car "
          "can do there as it stands, 'vs the car with no wings' what its wings are "
          "worth now (negative = faster). Time in corners is what a wing's force buys; "
          "time on straights is where its drag is paid.",
          "LEFT / RIGHT (d-pad) change a row; ENTER / CROSS states the mission.",
          _chk_mission, ("row", "go")),
    WStep("screen", "2  The aerofoil: screening",
          "The AEROFOIL (section) is the shape of the wing cut front to back. The "
          "screening WEIGHTS say what matters on this mission: clmax (the most lift "
          "it can make), L/D (lift per unit of drag), thickness (room for a spar), "
          "stall angle, pitching moment. Screening scores every section in the "
          "library against them.",
          "ENTER / CROSS (or L) on AIRFOIL > library screening.",
          _chk_screen, ("nav", "af.screen")),
    WStep("section", "3  The aerofoil: take one, fit it",
          "The RANKING lists the sections best first, with the numbers the weights "
          "priced. Taking one puts its shape on the page; F fits it to the wing "
          "and opens the next group. (Shaping it further, and the shape optimiser, "
          "are optional.)",
          "ENTER / CROSS on a ranked section, then F (pad: TRIANGLE to the rows of "
          "AIRFOIL > section, 'fit', CROSS).",
          _chk_section, ("nav", "af.rank")),
    WStep("plates", "4  The end plates",
          "END PLATES are the small panels at the wing's tips: they stop the air "
          "spilling round the end, which keeps the lift. Their section is designed "
          "the same way -- or fly them FLAT, the library's own default.",
          "ENDPLATE > library screening: screen, take one, F -- or the 'fly FLAT' row "
          "(TRIANGLE to the rows, CROSS).",
          _chk_plates, ("nav", "ep.screen")),
    WStep("planform", "5  The planform",
          "WING TYPE is how it is mounted and blended into the plates. The DESIGN "
          "BOX is the PLANFORM, the wing seen from above: span (how long), chord "
          "(how deep, front to back), taper (tip chord / root chord), twist, "
          "incidence (the angle it meets the air) and the area they make. More "
          "area = more force AND more drag.",
          "Look at WING > design box (TAB / TRIANGLE: its rows, LEFT / RIGHT: change "
          "one), then move on down the steps.",
          _chk_planform, ("nav", "w.box")),
    WStep("results", "6  The results",
          "F is the side force and D the drag, in newtons, at the design speed; "
          "L/D is force per unit of drag (higher = cheaper force). The number that "
          "decides is the LAP on your mission against the car with no wing. O "
          "(SQUARE) optimises the design box against it.",
          "Open RESULTS > summary (DOWN on the steps, ENTER / CROSS).",
          _chk_results, ("nav", "r.summary")),
    WStep("fit", "7  Fit it to the car",
          "Saving puts the wing in the library and in the slot (mirrored left / "
          "right unless M split them), under the name the page shows: a wing of "
          "yours with that name is replaced. The car view shows it; W cycles the "
          "library's wings in a slot.",
          "N renames it first if you like; S saves the wing and fits it (pad: CIRCLE "
          "-- leaving the page saves it).",
          _chk_fit, ("bar",)),
    WStep("build", "8  Save the car as a build",
          "A BUILD is the whole car: all three slots, their stations and angles, "
          "kept for the car it was made for. Saved builds are what the TIME TRIAL "
          "page offers (Build > PICK) and B steps through, on any map, each with its "
          "best lap; F makes one the car's own default.",
          "ESC / CIRCLE back to the car and S saves it (pad: OPTIONS > Save build); "
          "or L opens the library, where S / SQUARE saves it too.",
          _chk_build, ("list", "builds")),
    WStep("drive", "9  Drive it",
          "The TIME TRIAL page opens with this build; its laps are filed with it, "
          "so the table shows which of your builds is fastest.",
          "ESC / CIRCLE back to the car and ENTER / CROSS: drive it.",
          _chk_drive, ("bar",)),
)


#: the steps whose state is the design chain's (the mission, the pages)
CHAIN = ("mission", "screen", "section", "plates", "planform", "results", "fit")


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


def anchor_rect(g, anchor):
    """Where the page's last draw put the anchored widget (a pygame.Rect),
    or None when it is not on this page."""
    import pygame
    if not anchor:
        return None
    kind = anchor[0]
    u = g.view.ui
    if kind == "bar":
        return pygame.Rect(int(12 * u), int(690 * u), int(1256 * u), int(98 * u)) \
            if g.page != "car" else None
    if kind == "nav" and g.page == "section":
        nav = _dp(g).nav
        for k, y, h in getattr(nav, "_hits", []):
            if k == anchor[1] and nav._rect is not None:
                return pygame.Rect(nav._rect.x, int(y), nav._rect.w, int(h))
        return None
    if kind == "row":
        pl = (g.mission_page.params if g.page == "mission"
              else (_dp(g).rows() if g.page == "section" else None))
        if pl is None or pl._rect is None:
            return None
        for i, y, h in getattr(pl, "_hits", []):
            if 0 <= i < len(pl.params) and pl.params[i].key == anchor[1]:
                return pygame.Rect(pl._rect.x, int(y), pl._rect.w, int(h))
        return None
    if kind == "list" and g.page == "library":
        lst = getattr(g.lib_page, anchor[1], None)
        return getattr(lst, "_rect", None)
    return None


class WingTutor:
    """The state machine; `Garage.tutor`. It survives the garage <-> drive
    round trips on `opts.wing_tutor`."""

    def __init__(self, progress=None, start: str | None = None, steps=STEPS):
        self.steps = tuple(steps)
        self.progress = progress
        ids = [s.id for s in self.steps]
        self.i = ids.index(start) if start in ids else 0
        self.mem: dict = {}
        self.active = True
        self.done = False
        self.hidden = False
        self.flash = ""
        self._flash_n = 0
        self.skipped: list = []
        self._rect = None                  # where draw() last put the box
        self._save()

    @property
    def step(self) -> WStep:
        return self.steps[min(self.i, len(self.steps) - 1)]

    def label(self) -> str:
        return f"{self.i + 1}/{len(self.steps)}"

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

    def update(self, g, action=None) -> bool:
        """Once per garage frame, after its events: True when a step passed.
        Several can pass in one frame (a build that is already saved)."""
        if self.active:
            self._rewind(g)
        passed = False
        for _ in range(len(self.steps)):
            if not self.active:
                break
            st = self.step
            if not st.check(g, self.mem, action):
                break
            self.flash, self._flash_n = f"done: {st.title}", 150
            self.advance()
            passed = True
        if self._flash_n > 0:
            self._flash_n -= 1
        return passed

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
        self.progress.save(SECTION)

    # -- the garage's pause menu ------------------------------------------------
    def menu_rows(self) -> list:
        if not self.active:
            return []
        return [(f"Wing tutorial {self.label()}: skip this step ({self.step.title})", "wt_skip"),
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
    def lines(self, wrap=_wrap) -> list:
        """(text, colour-name) rows of the box: 'head', 'text', 'do', 'flash', 'dim'."""
        rows = []
        if self.flash and self._flash_n > 0:
            rows.append((self.flash, "flash"))
        if self.active:
            st = self.step
            rows.append((f"WING TUTORIAL {self.label()}   {st.title}", "head"))
            rows += [(ln, "text") for ln in wrap(st.text)]
            rows += [(ln, "do") for ln in wrap("> " + st.do)]
            rows += [(ln, "dim") for ln in wrap("H (or the menu) hides this box   ESC / "
                                                 "OPTIONS on the car: skip / end")]
        return rows

    def draw(self, g) -> None:
        """The box, over the page's lower right, and an outline round the
        anchored widget with a line to it. Nothing when hidden."""
        import pygame
        from . import garage_ui as ui
        self._rect = None                  # a hidden or empty box takes nothing
        u = g.view.ui
        size, lh = 13, int(17 * u)
        x0, y0, w0, h0 = (int(v * u) for v in BOX)
        pad = int(10 * u)
        rows = self.lines(lambda s: ui._wrap_px(g.text, s, w0 - 2 * pad, size))
        if self.hidden or not rows:
            return
        h = min(h0 + int(100 * u), len(rows) * lh + int(16 * u))
        y0 = y0 + h0 - h if h <= h0 else y0 - (h - h0)
        r = ui.panel(g.screen, (x0, y0, w0, h), alpha=235, accent=True)
        self._rect = r
        cols = {"head": C_BOX_ACCENT, "text": ui.C_TEXT, "do": ui.C_OK,
                "flash": ui.C_OK, "dim": ui.C_DIM}
        y = r.y + int(8 * u)
        for s, c in rows:
            if y + lh > r.bottom:
                break
            g.text.blit(g.screen, s, r.x + pad, y, size, cols[c], bold=(c == "head"))
            y += lh
        a = anchor_rect(g, self.step.anchor) if self.active else None
        if a is not None:
            pygame.draw.rect(g.screen, C_BOX_ACCENT, a.inflate(4, 4), 2)
            pygame.draw.line(g.screen, C_BOX_ACCENT, (r.x, r.y + int(10 * u)),
                             (a.right + 2, a.centery), 1)


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
    import os
    import tempfile
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame
    from .aero.library import Library
    from . import garage as grg
    from .garage_ui import _wrap_px as ui_wrap
    from .progress import Progress
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}" + (f": {msg}" if msg else ""))

    tmp = tempfile.mkdtemp(prefix="carsim_wingtut_")
    lib = Library(os.path.join(tmp, "library"), use_xfoil=False)
    g = grg.Garage((1280, 800), grg.CarBuild(), headless=True, lib=lib)
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

    ids = [s.id for s in STEPS]
    rep("ten steps, the garage's own order, each with a predicate and an anchor",
        ids == ["open", "mission", "screen", "section", "plates", "planform", "results",
                "fit", "build", "drive"] and all(callable(s.check) for s in STEPS)
        and all(s.anchor for s in STEPS), " > ".join(ids))
    seen = {}
    # 1 the car page: nothing yet; D opens the mission
    seen["open"] = (not frame(), )
    key(pygame.K_d)
    seen["open"] += (frame() and t.step.id == "mission",)
    # 2 the mission: stated by ENTER
    seen["mission"] = (not frame(), anchor_rect(g, t.step.anchor) is not None)
    key(pygame.K_RETURN)
    seen["mission"] += (frame() and t.step.id == "screen" and g.page == "section",)
    dp = g.design_page
    # 3 screening
    seen["screen"] = (not frame(), anchor_rect(g, t.step.anchor) is not None)
    key(pygame.K_l)
    seen["screen"] += (frame() and t.step.id == "section",)
    # 4 take the winner, fit it
    seen["section"] = (not frame(),)
    dp.af.use_ranked(0)
    seen["section"] += (not frame(),)                 # taken, not yet fitted
    key(pygame.K_f)
    seen["section"] += (frame() and t.step.id == "plates",)
    # 5 the plates: fly them flat
    seen["plates"] = (not frame(),)
    dp.ep.decline()
    seen["plates"] += (frame() and t.step.id == "planform",)
    # 6 the planform: seen, then moved on
    dp.nav.select("w.type")
    seen["planform"] = (not frame(),)
    dp.nav.select("w.box")
    seen["planform"] += (not frame(), anchor_rect(g, t.step.anchor) is not None)
    dp.nav.select("w.solver")
    seen["planform"] += (frame() and t.step.id == "results",)
    # 7 the results
    seen["results"] = (not frame(),)
    sel = dp.nav.select("r.summary")
    seen["results"] += (sel and frame() and t.step.id == "fit",)
    # 8 fit: S saves the wing into the slot
    seen["fit"] = (not frame(),)
    key(pygame.K_s)
    seen["fit"] += (frame() and t.step.id == "build",
                    g.build.left.wing == dp.wing.spec.name)
    # 9 the build: ESC to the car, L, S (the prompt), a name
    seen["build"] = (not frame(),)
    key(pygame.K_ESCAPE)
    key(pygame.K_ESCAPE)
    key(pygame.K_l)
    seen["build"] += (not frame(), anchor_rect(g, t.step.anchor) is not None)
    g._save_build_quick()
    seen["build"] += (frame() and t.step.id == "drive",)
    # 10 drive
    seen["drive"] = (not frame(None),)
    key(pygame.K_ESCAPE)
    seen["drive"] += (frame("drive") and t.done and not t.active,)
    for sid in ids:
        v = seen.get(sid, ())
        rep(f"step {sid}: false before the garage state, true after", v and all(v), str(v))
    sv = saved_state(Progress(prog.path))
    rep("done and saved in the progress file", sv["done"] and sv["step"] is None, str(sv))
    # the box takes its own clicks, and every line fits it
    t7 = WingTutor(None, start="section")
    g.page = "section"
    g._draw_page()
    t7.draw(g)
    bx = t7._rect
    inner = bx.w - 2 * int(10 * g.view.ui) if bx is not None else 0
    widest = max((g.text.width(s, 13, bold=(c == "head")) for s, c in t7.lines(
        lambda s: ui_wrap(g.text, s, inner, 13))), default=0)
    rep("the box takes its own clicks; every line fits inside it",
        bx is not None and t7.hit(bx.center) and not t7.hit((0, 0)) and widest <= inner,
        f"widest {widest} px in {inner}")
    g.tutor = t7
    before_ = (dp.nav.current(), dp.focus, dp.rank_list.idx)
    g._handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=bx.center))
    rep("a click on the box does not reach the page under it",
        (dp.nav.current(), dp.focus, dp.rank_list.idx) == before_, str(before_))
    g.tutor = t
    # a re-opened design page / a new garage: the tutor walks back to where the work starts
    t8 = WingTutor(None, start="planform")
    g.open_section("left")                 # DesignPage.open re-locks the gates
    g._draw_page()
    t8.update(g)
    back1 = t8.step.id
    g2 = grg.Garage((1280, 800), grg.CarBuild(), headless=True, lib=lib)
    t9 = WingTutor(None, start="results")
    t9.update(g2)
    rep("a re-opened design page or a new garage rewinds the tutor to the open gate",
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
    if verbose:
        print(f"wing_tutorial self-check: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
