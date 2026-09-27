"""drive/garage_ui.py -- the small widget kit the garage pages are built
from: a theme (the same dark palette as menu.py / render.py), cached text,
translucent panels, a parameter list you edit with the arrow keys or a
pad, a scrolling list box, a 2-D plot and a one-line text prompt.

Pure pygame + numpy; no physics, no file IO. Every widget draws into a
rect it is handed and keeps its own cursor state, so a page is a handful
of widgets and a key map.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

FONT_NAMES = ("Menlo", "Monaco", "DejaVu Sans Mono", "Courier New")

# palette (menu.py / garage.py / render.py agree on these)
C_BG = (27, 29, 33)
C_PANEL = (16, 17, 20)
C_PANEL_A = 216
C_BORDER = (60, 64, 70)
C_TEXT = (232, 234, 238)
C_DIM = (139, 144, 153)
C_ACCENT = (255, 140, 43)
C_SEL_BG = (40, 44, 52)
C_KEY = (217, 206, 85)
C_SECTION = (79, 163, 255)
C_OK = (78, 194, 106)
C_WARN = (226, 82, 63)
C_GRID = (44, 47, 53)
C_LINE2 = (111, 208, 140)
C_LINE3 = (200, 120, 220)
C_LINE4 = (120, 200, 230)


class Fonts:
    """SysFont cache keyed on (size, bold), scaled by the UI factor."""

    def __init__(self, ui: float = 1.0):
        self.ui = ui
        self._f = {}

    def get(self, size: int, bold: bool = False):
        key = (int(round(size * self.ui)), bold)
        f = self._f.get(key)
        if f is None:
            if not pygame.font.get_init():
                pygame.font.init()
            for name in FONT_NAMES:
                try:
                    f = pygame.font.SysFont(name, key[0], bold=bold)
                    if f is not None:
                        break
                except Exception:
                    continue
            if f is None:
                f = pygame.font.Font(None, key[0] + 4)
            self._f[key] = f
        return f


class Text:
    def __init__(self, fonts: Fonts):
        self.fonts = fonts
        self._cache = {}

    def surf(self, s: str, size: int = 14, col=C_TEXT, bold: bool = False):
        key = (s, size, col, bold)
        surf = self._cache.get(key)
        if surf is None:
            if len(self._cache) > 2048:
                self._cache.clear()
            surf = self.fonts.get(size, bold).render(s, True, col)
            self._cache[key] = surf
        return surf

    def blit(self, screen, s: str, x, y, size: int = 14, col=C_TEXT, bold: bool = False,
             right: bool = False, centre: bool = False) -> int:
        surf = self.surf(s, size, col, bold)
        w = surf.get_width()
        if right:
            x = x - w
        elif centre:
            x = x - 0.5 * w
        screen.blit(surf, (int(x), int(y)))
        return w

    def width(self, s: str, size: int = 14, bold: bool = False) -> int:
        return self.fonts.get(size, bold).size(s)[0]


def panel(screen, rect, alpha: int = C_PANEL_A, border: bool = True, accent: bool = False):
    r = pygame.Rect(rect)
    s = pygame.Surface(r.size, pygame.SRCALPHA)
    s.fill((*C_PANEL, alpha))
    screen.blit(s, r.topleft)
    if border:
        pygame.draw.rect(screen, C_BORDER, r, 1)
    if accent:
        pygame.draw.rect(screen, C_ACCENT, (r.x, r.y, r.w, 3))
    return r


# --------------------------------------------------------------------------- #
def _wrap_px(text: "Text", msg: str, px: int, size: int) -> list:
    """`msg` broken to lines that FIT `px`, measured with the font actually
    drawing them. The flat character caps this file used to carry were the
    reason a refusal read "the shortlist is chosen on the ma"."""
    out, line = [], ""
    for word in str(msg).split():
        trial = f"{line} {word}".strip()
        if line and text.width(trial, size) > px:
            out.append(line)
            line = word
        else:
            line = trial
    if line:
        out.append(line)
    return out


class Param:
    """One editable row. `get()`/`set(v)` talk to the model; `kind`:
    'float' (LEFT/RIGHT step, SHIFT fine), 'int', 'choice' (cycles
    `choices`), 'bool', 'action' (ENTER calls `set(None)`), 'label'.

    `lo` / `hi` may be CALLABLES, read at every step (task 41): a band that
    moves while the page is open -- a flank panel's span limit follows its
    mount height and the car it is on -- could not be a number captured when
    the list was built, and a stale `hi` let LEFT / RIGHT step a row past
    the cap its setter then had to undo."""

    def __init__(self, key, label, get, set=None, *, kind="float", step=0.01, fine=None,
                 lo=None, hi=None, unit="", fmt="{:.2f}", choices=(), help="", enabled=True):
        self.key, self.label = key, label
        self.get, self.set = get, set
        self.kind = kind
        self.step, self.fine = step, (fine if fine is not None else step / 5.0)
        self.lo, self.hi = lo, hi
        self.unit, self.fmt = unit, fmt
        self.choices = list(choices)
        self.help = help
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """`enabled` may be a CALLABLE, read every frame. Some rows are only
        answerable while something else is true -- the plate's blend needs a
        plate to blend -- and that something can change on a row one panel
        away, while this row is on screen. A bool captured when the list was
        built would go stale there, and the row would take an edit the model
        has nothing to do with."""
        e = self._enabled
        return bool(e() if callable(e) else e)

    def band(self) -> tuple:
        """(lo, hi) as they stand now: a callable end is read, a number is
        itself, None is no bound."""
        lo = self.lo() if callable(self.lo) else self.lo
        hi = self.hi() if callable(self.hi) else self.hi
        return lo, hi

    def value_text(self) -> str:
        v = self.get() if self.get is not None else ""
        if self.kind == "bool":
            return "on" if v else "off"
        if self.kind == "action":
            return ""
        if self.kind == "int":
            #  an int row showed "48.00", because `fmt` defaults to two
            #  decimals and nothing overrode it for the one kind that cannot
            #  have them. The kind decides the format, not the default.
            try:
                return f"{int(round(float(v)))}" + (f" {self.unit}" if self.unit else "")
            except (ValueError, TypeError):
                return str(v)
        if self.kind == "float":
            try:
                return self.fmt.format(v) + (f" {self.unit}" if self.unit else "")
            except (ValueError, TypeError):
                return str(v)
        return str(v)

    def adjust(self, direction: int, fine: bool = False) -> bool:
        if not self.enabled or self.set is None:
            return False
        lo, hi = self.band()
        if self.kind == "float":
            v = float(self.get()) + direction * (self.fine if fine else self.step)
            if lo is not None:
                v = max(v, lo)
            if hi is not None:
                v = min(v, hi)
            self.set(v)
            return True
        if self.kind == "int":
            v = int(self.get()) + direction
            if lo is not None:
                v = max(v, int(lo))
            if hi is not None:
                v = min(v, int(hi))
            self.set(v)
            return True
        if self.kind == "choice" and self.choices:
            cur = self.get()
            i = self.choices.index(cur) if cur in self.choices else 0
            self.set(self.choices[(i + direction) % len(self.choices)])
            return True
        if self.kind == "bool":
            self.set(not bool(self.get()))
            return True
        return False

    def activate(self) -> bool:
        if not self.enabled or self.set is None:
            return False
        if self.kind == "action":
            self.set(None)
            return True
        if self.kind in ("bool", "choice"):
            return self.adjust(+1)
        return False


class ParamList:
    """Rows of `Param`, driven by the keyboard OR the mouse.

    THE MOUSE IS NOT A SECOND WAY IN, it is the SAME way in. A click lands on
    exactly the row `draw` put under the cursor, and what it does there is
    what the keyboard would do on that row: an action row fires, a value row
    steps the way LEFT/RIGHT steps it. The value's two halves are the two
    arrows -- click the left of the `< value >` and it is a LEFT, the right
    and it is a RIGHT -- which is why the value is drawn with those arrows on
    the selected row in the first place.
    """

    #: how wide the value's click target is, in pixels, measured left from
    #: the right edge of the row. Wide enough for "< -0.0125 deg >" at 14 px.
    VALUE_W = 150

    def __init__(self, params, title: str = ""):
        self.params = list(params)
        self.title = title
        self.idx = 0
        self.scroll = 0
        self._rect = None
        self._hits: list = []          # (index, y, h), filled by draw()
        self._vals: dict = {}          # index -> (x0, x1) of the value as drawn
        self._hover = -1

    def current(self) -> Param | None:
        rows = [p for p in self.params if p.kind != "label"]
        if not self.params:
            return None
        self.idx = max(0, min(self.idx, len(self.params) - 1))
        return self.params[self.idx]

    def nav(self, d: int) -> None:
        if not self.params:
            return
        n = len(self.params)
        for _ in range(n):
            self.idx = (self.idx + d) % n
            if self.params[self.idx].kind != "label":
                break

    def adjust(self, d: int, fine: bool = False) -> bool:
        p = self.current()
        return bool(p and p.adjust(d, fine))

    def activate(self) -> bool:
        p = self.current()
        return bool(p and p.activate())

    def select_key(self, key: str) -> None:
        for i, p in enumerate(self.params):
            if p.key == key:
                self.idx = i
                return

    # -- the mouse ------------------------------------------------------------
    def hit(self, pos) -> int:
        """The row index under `pos`, or -1. Label rows are never hit: they
        are not selectable by the keyboard either."""
        x, y = pos
        if self._rect is None or not self._rect.collidepoint(x, y):
            return -1
        for i, ry, rh in self._hits:
            if ry <= y < ry + rh:
                return i
        return -1

    def click(self, pos, fine: bool = False) -> str:
        """A left click. Returns what it did: 'select', 'adjust', 'action'
        or '' -- the page prints the row's help afterwards either way."""
        i = self.hit(pos)
        if i < 0:
            return ""
        moved = (i != self.idx)
        self.idx = i
        p = self.params[i]
        if p.kind == "action":
            #  a first click on an action row selects it, a second fires it.
            #  Firing on the click that also moved the cursor would make a
            #  mis-aimed click run an optimiser.
            if moved:
                return "select"
            p.activate()
            return "action"
        if moved:
            return "select"
        #  already selected: the click is on one of the two arrows. The two
        #  halves split at the middle of the value AS DRAWN, not of the fixed
        #  VALUE_W zone: "< 0.35 >" is right-aligned and far narrower than
        #  VALUE_W, so halving the zone put its '<' in the right half and a
        #  click on it stepped UP
        x = pos[0]
        edge = self._rect.right - 12
        x0, x1 = self._vals.get(i, (edge - self.VALUE_W, edge))
        if x >= min(x0, edge - self.VALUE_W):
            p.adjust(1 if x >= 0.5 * (x0 + x1) else -1, fine)
            return "adjust"
        return "select"

    def wheel(self, dy: int) -> None:
        self.nav(-1 if dy > 0 else 1)

    def draw(self, screen, text: Text, rect, row_h: int = 22, size: int = 14, focus: bool = True):
        r = pygame.Rect(rect)
        self._rect, self._hits, self._vals = r, [], {}
        self._hover = self.hit(pygame.mouse.get_pos())
        y = r.y + 6
        if self.title:
            text.blit(screen, self.title, r.x + 10, y, size, C_SECTION, bold=True)
            y += row_h + 2
        n_vis = max(1, int((r.bottom - y - 4) // row_h))
        if self.idx < self.scroll:
            self.scroll = self.idx
        elif self.idx >= self.scroll + n_vis:
            self.scroll = self.idx - n_vis + 1
        val_x = r.right - 12
        for i in range(self.scroll, min(len(self.params), self.scroll + n_vis)):
            p = self.params[i]
            sel = (i == self.idx) and focus
            if p.kind == "label":
                text.blit(screen, p.label, r.x + 10, y + 3, size - 1, C_SECTION, bold=True)
                y += row_h
                continue
            self._hits.append((i, y - 2, row_h))
            if sel:
                pygame.draw.rect(screen, C_SEL_BG, (r.x + 4, y - 2, r.w - 8, row_h))
                pygame.draw.rect(screen, C_ACCENT, (r.x + 4, y - 2, 3, row_h))
            elif i == self._hover:
                pygame.draw.rect(screen, C_GRID, (r.x + 4, y - 2, r.w - 8, row_h))
            col = C_TEXT if p.enabled else C_DIM
            if p.kind == "action":
                text.blit(screen, ("> " if sel else "  ") + p.label, r.x + 12, y + 2, size,
                          C_ACCENT if p.enabled else C_DIM)
            else:
                text.blit(screen, p.label, r.x + 12, y + 2, size, col if not sel else C_TEXT)
                vt = p.value_text()
                if sel and p.kind in ("float", "int", "choice", "bool"):
                    vt = "< " + vt + " >"
                w = text.blit(screen, vt, val_x, y + 2, size, C_KEY if sel else col, right=True)
                self._vals[i] = (val_x - w, val_x)
            y += row_h
        if self.params and self.scroll + n_vis < len(self.params):
            text.blit(screen, "...", r.right - 30, r.bottom - 16, size - 2, C_DIM)
        p = self.current()
        return p.help if (p and focus) else ""


# --------------------------------------------------------------------------- #
class Nav:
    """A two-level navigator: GROUPS with ordered STEPS under them.

    The left column of the design pages. It is the procedure written down --
    the same four groups, in the same order, that the UROP app's own navigator
    carries -- so what the garage is doing is legible before anything is
    pressed rather than only afterwards.

    `tree` is `[(group_label, [(key, label), ...]), ...]`. Only STEPS are
    selectable; a group header is drawn but never lands under the cursor.
    `state(key)` is asked, per step, for one of 'done' / 'ready' / 'blocked' /
    'locked', which is what the mark in the left margin says.

    A BLOCKED OR LOCKED STEP CANNOT BE SELECTED. That is the UROP app's own
    rule and not a decoration on it: `gui/v3/app.py`'s `select()` reads
    `session.stage_states(S)`, returns without moving if the stage is locked,
    and notifies the stored REASON -- "a greyed-out node with no explanation
    is the thing this shell exists to avoid". `reason(key)` is that sentence
    here, and `refused` carries the last one so the page can print it.

    The two unavailable states say different things and are drawn
    differently. BLOCKED is "not yet": the step before it has not been
    finished, and finishing it opens this one. LOCKED is "not here": there is
    nothing on this vehicle for the step to decide -- the plates are at zero
    height, say -- and no amount of work upstream will open it.
    """

    MARK = {"done": ("+", C_OK), "ready": (">", C_ACCENT),
            "blocked": ("-", C_DIM), "locked": ("x", C_DIM)}
    #: the states a step may not be selected in
    SHUT = ("blocked", "locked")

    def __init__(self, tree, title: str = "", state=None, note=None, reason=None):
        self.tree = [(g, list(steps)) for g, steps in tree]
        self.title = title
        self.state = state or (lambda k: "ready")
        self.note = note or (lambda k: "")
        self.reason = reason or (lambda k: "")
        self.keys = [k for _, steps in self.tree for k, _ in steps]
        self.idx = 0
        self.refused = ""
        self._hits: list = []          # (key, y, h), filled by draw()

    def open(self, key: str) -> bool:
        """May this step be selected at all?"""
        return self.state(key) not in self.SHUT

    def first_open(self) -> str:
        """The first step that may be selected -- where a page opens, and
        where it falls back to when the step it was on closes under it."""
        for k in self.keys:
            if self.open(k):
                return k
        return self.keys[0] if self.keys else ""

    # -- selection ----------------------------------------------------------
    def current(self) -> str:
        if not self.keys:
            return ""
        self.idx = max(0, min(self.idx, len(self.keys) - 1))
        return self.keys[self.idx]

    def label(self, key: str) -> str:
        for _, steps in self.tree:
            for k, lab in steps:
                if k == key:
                    return lab
        return key

    def group_of(self, key: str) -> str:
        for g, steps in self.tree:
            if any(k == key for k, _ in steps):
                return g
        return ""

    def nav(self, d: int) -> None:
        """Move to the next SELECTABLE step. A shut one is stepped over
        rather than landed on and refused, because an arrow key that appears
        to do nothing is worse than one that skips: the mark in the margin
        already says the step is shut, and `note()` says why."""
        if not self.keys:
            return
        self.refused = ""
        n = len(self.keys)
        for step in range(1, n + 1):
            j = (self.idx + d * step) % n
            if self.open(self.keys[j]):
                self.idx = j
                return
        #  nothing at all is selectable: stay put rather than spin

    def nav_group(self, d: int) -> None:
        """Jump to the first OPEN step of the next / previous GROUP."""
        cur = self.group_of(self.current())
        groups = [g for g, _ in self.tree]
        if cur not in groups:
            return
        by = dict(self.tree)
        for step in range(1, len(groups) + 1):
            g = groups[(groups.index(cur) + d * step) % len(groups)]
            for k, _ in by[g]:
                if self.open(k):
                    self.select(k)
                    return

    def select(self, key: str, force: bool = False) -> bool:
        """Select `key`, or refuse it and keep the reason.

        `force` is for the page's own book-keeping -- re-selecting the step
        that was already current after its state changed -- and never for a
        user action.
        """
        if key not in self.keys:
            return False
        if not force and not self.open(key):
            self.refused = self.reason(key) or f"'{self.label(key)}' is not open yet"
            return False
        self.refused = ""
        self.idx = self.keys.index(key)
        return True

    # -- the mouse ------------------------------------------------------------
    def hit(self, pos) -> str:
        """The step key under `pos`, or "" -- from the geometry the last
        `draw` recorded, so what is clickable is exactly what was drawn."""
        x, y = pos
        if self._rect is None or not self._rect.collidepoint(x, y):
            return ""
        for key, ry, rh in self._hits:
            if ry <= y < ry + rh:
                return key
        return ""

    def click(self, pos) -> str:
        """A left click: select the step under the cursor. Returns the key
        that was taken, or "" -- a refusal leaves `refused` set."""
        k = self.hit(pos)
        if k and self.select(k):
            return k
        return ""

    _rect = None

    # -- drawing -------------------------------------------------------------
    def draw(self, screen, text: Text, rect, row_h: int = 22, size: int = 13,
             focus: bool = True):
        r = pygame.Rect(rect)
        self._rect, self._hits = r, []
        self._hover = self.hit(pygame.mouse.get_pos()) if self._rect else ""
        x, y = r.x + 10, r.y + 8
        if self.title:
            text.blit(screen, self.title, x, y, size + 1, C_SECTION, bold=True)
            y += row_h
        cur = self.current()
        for g, steps in self.tree:
            text.blit(screen, g, x, y, size, C_TEXT, bold=True)
            y += row_h
            for k, lab in steps:
                st = self.state(k)
                mark, mcol = self.MARK.get(st, self.MARK["ready"])
                sel = (k == cur)
                hot = (self._hover == k and st not in self.SHUT and not sel)
                if sel:
                    pygame.draw.rect(screen, C_SEL_BG if focus else C_PANEL,
                                     (r.x + 4, y - 3, r.width - 8, row_h - 2))
                    pygame.draw.rect(screen, C_ACCENT, (r.x + 4, y - 3, 3, row_h - 2))
                elif hot:
                    pygame.draw.rect(screen, C_GRID,
                                     (r.x + 4, y - 3, r.width - 8, row_h - 2))
                text.blit(screen, mark, x + 12, y, size, mcol)
                col = C_TEXT if (sel or st not in self.SHUT) else C_DIM
                text.blit(screen, lab, x + 28, y, size, col, bold=sel)
                self._hits.append((k, y - 3, row_h))
                y += row_h
            y += 4
        #  the sentence under the tree is the REFUSAL when there is one --
        #  a click that did nothing has to say why it did nothing -- and the
        #  current step's own note otherwise.
        #  the sentence under the tree: the REFUSAL when there is one -- a
        #  click that did nothing has to say why -- and the current step's own
        #  note otherwise. Drawn BELOW the tree rather than pinned to the
        #  bottom of the panel, and in full: the reason is the whole point of
        #  refusing, so clipping it to two lines threw away the half that
        #  said what to do about it.
        n = self.refused or self.note(cur)
        if n:
            col = C_WARN if self.refused else C_DIM
            y += 6
            if self.refused:
                text.blit(screen, "NOT YET", x, y, 11, C_WARN, bold=True)
                y += 16
            for ln in _wrap_px(text, n, r.width - 20, 11)[:6]:
                if y > r.bottom - 16:
                    break
                text.blit(screen, ln, x, y, 11, col)
                y += 15
        return r


class ListBox:
    """Rows of (label, sub, tag) with a cursor and scrolling. A row may carry
    a 4th field, the name DRAWN for it (task 45: a built-in wing's player
    name); the label stays the row's key, what `current()` and `keep` go by."""

    def __init__(self, items=(), title: str = ""):
        self.items = list(items)
        self.title = title
        self.idx = 0
        self.scroll = 0
        self._rect = None
        self._hits: list = []
        self._hover = -1

    def set_items(self, items, keep=None) -> None:
        self.items = list(items)
        if keep is not None:
            for i, it in enumerate(self.items):
                if it[0] == keep:
                    self.idx = i
                    break
        self.idx = max(0, min(self.idx, max(len(self.items) - 1, 0)))

    def current(self):
        if not self.items:
            return None
        self.idx = max(0, min(self.idx, len(self.items) - 1))
        return self.items[self.idx]

    def nav(self, d: int) -> None:
        if self.items:
            self.idx = (self.idx + d) % len(self.items)

    # -- the mouse ------------------------------------------------------------
    def hit(self, pos) -> int:
        x, y = pos
        if self._rect is None or not self._rect.collidepoint(x, y):
            return -1
        for i, ry, rh in self._hits:
            if ry <= y < ry + rh:
                return i
        return -1

    def click(self, pos) -> str:
        """Select the row under the cursor; a second click on the row that is
        already selected is the ENTER on it."""
        i = self.hit(pos)
        if i < 0:
            return ""
        if i == self.idx:
            return "activate"
        self.idx = i
        return "select"

    def wheel(self, dy: int) -> None:
        self.nav(-1 if dy > 0 else 1)

    def draw(self, screen, text: Text, rect, row_h: int = 34, size: int = 14, focus: bool = True,
             empty: str = "(empty)"):
        r = pygame.Rect(rect)
        self._rect, self._hits = r, []
        self._hover = self.hit(pygame.mouse.get_pos())
        y = r.y + 6
        if self.title:
            text.blit(screen, self.title, r.x + 10, y, size, C_SECTION, bold=True)
            y += 24
        n_vis = max(1, int((r.bottom - y - 4) // row_h))
        if self.idx < self.scroll:
            self.scroll = self.idx
        elif self.idx >= self.scroll + n_vis:
            self.scroll = self.idx - n_vis + 1
        if not self.items:
            text.blit(screen, empty, r.x + 12, y + 4, size, C_DIM)
            return
        for i in range(self.scroll, min(len(self.items), self.scroll + n_vis)):
            it = self.items[i]
            sel = (i == self.idx) and focus
            self._hits.append((i, y - 2, row_h))
            if sel:
                pygame.draw.rect(screen, C_SEL_BG, (r.x + 4, y - 2, r.w - 8, row_h - 2))
                pygame.draw.rect(screen, C_ACCENT, (r.x + 4, y - 2, 3, row_h - 2))
            elif i == self._hover:
                pygame.draw.rect(screen, C_GRID, (r.x + 4, y - 2, r.w - 8, row_h - 2))
            tag = it[2] if len(it) > 2 else ""
            text.blit(screen, ("> " if sel else "  ") + str(it[3] if len(it) > 3 else it[0]), r.x + 12, y + 1, size,
                      C_TEXT if sel else C_DIM)
            if tag:
                text.blit(screen, tag, r.right - 12, y + 1, size - 2, C_KEY if sel else C_DIM, right=True)
            if len(it) > 1 and it[1]:
                #  clipped to the PANEL, not to a fixed 70 characters. The
                #  ranking table's sub-line is a row of criterion columns and
                #  the flat cap was cutting it mid-number, in a panel with
                #  200 px of unused width beside it.
                sub, avail = str(it[1]), r.right - (r.x + 26) - 12
                if text.width(sub, size - 3) > avail:
                    n = max(1, int(len(sub) * avail / max(text.width(sub, size - 3), 1)))
                    while n > 1 and text.width(sub[:n], size - 3) > avail:
                        n -= 1
                    sub = sub[:n]
                text.blit(screen, sub, r.x + 26, y + 16, size - 3, C_DIM)
            y += row_h
        if self.scroll + n_vis < len(self.items):
            text.blit(screen, f"+{len(self.items) - self.scroll - n_vis} more", r.right - 12, r.bottom - 16,
                      size - 3, C_DIM, right=True)


# --------------------------------------------------------------------------- #
class Plot:
    """A 2-D plot in a rect: call `begin(rect, xlim, ylim)` then `line`,
    `points`, `hline`, `vline`, `label`."""

    def __init__(self, text: Text):
        self.text = text
        self.rect = pygame.Rect(0, 0, 10, 10)
        self.xlim = (0.0, 1.0)
        self.ylim = (0.0, 1.0)
        self.screen = None

    def begin(self, screen, rect, xlim, ylim, title: str = "", xlabel: str = "", ylabel: str = "",
              equal: bool = False, grid: bool = True):
        self.screen = screen
        r = pygame.Rect(rect)
        pad_l, pad_b, pad_t, pad_r = 44, 22, 20 if title else 8, 10
        self.rect = pygame.Rect(r.x + pad_l, r.y + pad_t, max(r.w - pad_l - pad_r, 10),
                                max(r.h - pad_t - pad_b, 10))
        x0, x1 = float(xlim[0]), float(xlim[1])
        y0, y1 = float(ylim[0]), float(ylim[1])
        if x1 - x0 < 1e-9:
            x1 = x0 + 1.0
        if y1 - y0 < 1e-9:
            y1 = y0 + 1.0
        if equal:
            sx = self.rect.w / (x1 - x0)
            sy = self.rect.h / (y1 - y0)
            if sx < sy:
                cy = 0.5 * (y0 + y1)
                h = self.rect.h / sx
                y0, y1 = cy - 0.5 * h, cy + 0.5 * h
            else:
                cx = 0.5 * (x0 + x1)
                w = self.rect.w / sy
                x0, x1 = cx - 0.5 * w, cx + 0.5 * w
        self.xlim, self.ylim = (x0, x1), (y0, y1)
        pygame.draw.rect(screen, (20, 21, 24), self.rect)
        if grid:
            for t in self._ticks(x0, x1):
                px = self._px(t, y0)[0]
                pygame.draw.line(screen, C_GRID, (px, self.rect.y), (px, self.rect.bottom), 1)
                self.text.blit(screen, self._fmt(t), px, self.rect.bottom + 3, 10, C_DIM, centre=True)
            for t in self._ticks(y0, y1):
                py = self._px(x0, t)[1]
                pygame.draw.line(screen, C_GRID, (self.rect.x, py), (self.rect.right, py), 1)
                self.text.blit(screen, self._fmt(t), self.rect.x - 4, py - 6, 10, C_DIM, right=True)
        pygame.draw.rect(screen, C_BORDER, self.rect, 1)
        if title:
            self.text.blit(screen, title, r.x + pad_l, r.y + 3, 12, C_SECTION, bold=True)
        if xlabel:
            self.text.blit(screen, xlabel, self.rect.right, self.rect.bottom + 3, 10, C_DIM, right=True)
        if ylabel:
            self.text.blit(screen, ylabel, self.rect.x + 3, self.rect.y + 2, 10, C_DIM)

    @staticmethod
    def _fmt(t: float) -> str:
        if abs(t) >= 100 or t == int(t):
            return f"{t:.0f}"
        if abs(t) >= 10:
            return f"{t:.1f}"
        return f"{t:.2f}".rstrip("0").rstrip(".")

    @staticmethod
    def _ticks(a: float, b: float, n: int = 5):
        span = b - a
        if span <= 0:
            return []
        raw = span / n
        mag = 10 ** math.floor(math.log10(raw))
        for m in (1, 2, 2.5, 5, 10):
            step = m * mag
            if span / step <= n + 1:
                break
        t0 = math.ceil(a / step) * step
        out = []
        t = t0
        while t <= b + 1e-9:
            out.append(round(t, 10))
            t += step
        return out

    def _px(self, x, y):
        x0, x1 = self.xlim
        y0, y1 = self.ylim
        px = self.rect.x + (x - x0) / (x1 - x0) * self.rect.w
        py = self.rect.bottom - (y - y0) / (y1 - y0) * self.rect.h
        return int(round(px)), int(round(py))

    @staticmethod
    def _finite_runs(xs, ys, closed: bool = False):
        """The polyline split at every non-finite point (NaN, +-inf):
        `(runs, whole)`, `runs` the (xs, ys) stretches of consecutive finite
        points at least 2 long, `whole` True when no point was dropped. A
        refused evaluation is scored -inf, so an optimiser trace can start
        (or dip) there; a gap in the line is what that looks like, not a
        crash. A closed outline with a gap is opened AT the gap (rotated to
        start after it), so the segment across the seam survives."""
        xs, ys = np.asarray(xs, float).ravel(), np.asarray(ys, float).ravel()
        n = min(xs.size, ys.size)
        xs, ys = xs[:n], ys[:n]
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.all():
            return ([(xs, ys)] if n >= 2 else []), True
        if closed and ok.any():
            k = int(np.flatnonzero(~ok)[0]) + 1
            xs, ys, ok = np.roll(xs, -k), np.roll(ys, -k), np.roll(ok, -k)
        edges = np.flatnonzero(np.diff(np.concatenate(([0], ok.astype(np.int8), [0]))))
        runs = [(xs[a:b], ys[a:b]) for a, b in zip(edges[::2], edges[1::2]) if b - a >= 2]
        return runs, False

    def line(self, xs, ys, col=C_ACCENT, width: int = 2, closed: bool = False):
        runs, whole = self._finite_runs(xs, ys, closed)
        if not runs:
            return
        clip = self.screen.get_clip()
        self.screen.set_clip(self.rect)
        for rx, ry in runs:
            pts = [self._px(x, y) for x, y in zip(rx, ry)]
            pygame.draw.lines(self.screen, col, closed and whole, pts, width)
        self.screen.set_clip(clip)

    def fill(self, xs, ys, col):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if xs.size < 3:
            return
        pts = [self._px(x, y) for x, y in zip(xs, ys)]
        clip = self.screen.get_clip()
        self.screen.set_clip(self.rect)
        pygame.draw.polygon(self.screen, col, pts)
        self.screen.set_clip(clip)

    def points(self, xs, ys, col=C_KEY, r: int = 3):
        for x, y in zip(xs, ys):
            if not (math.isfinite(x) and math.isfinite(y)):
                continue                # NaN / +-inf has no place on the plot (see _finite_runs)
            p = self._px(x, y)
            if self.rect.collidepoint(p):
                pygame.draw.circle(self.screen, col, p, r)

    def hline(self, y, col=C_DIM, width: int = 1):
        p0, p1 = self._px(self.xlim[0], y), self._px(self.xlim[1], y)
        if self.rect.y <= p0[1] <= self.rect.bottom:
            pygame.draw.line(self.screen, col, p0, p1, width)

    def vline(self, x, col=C_DIM, width: int = 1):
        p0, p1 = self._px(x, self.ylim[0]), self._px(x, self.ylim[1])
        if self.rect.x <= p0[0] <= self.rect.right:
            pygame.draw.line(self.screen, col, p0, p1, width)

    def label(self, x, y, s, col=C_TEXT, size: int = 11, dx: int = 4, dy: int = -14):
        p = self._px(x, y)
        self.text.blit(self.screen, s, p[0] + dx, p[1] + dy, size, col)


# --------------------------------------------------------------------------- #
class TextPrompt:
    """A one-line modal text entry: `open(title, initial)`, feed KEYDOWN
    events to `handle(ev)` -> 'ok' | 'cancel' | None, draw last.

    Task 45: the name it opens on (`initial`) is SELECTED, as a text box's
    would be, and drawn highlighted: the first printable key replaces it and
    the first BACKSPACE clears it (typing used to append to it: 'my corsa'
    and 'Fast' read 'my corsaFast'); RIGHT / END keeps it to type on after
    it; ENTER untouched takes it. CTRL+BACKSPACE (CMD on a Mac) clears the
    line at any time."""

    #: the longest name it takes
    MAX_LEN = 32
    #: what a key does to the selected name: on the key line while it is
    SEL_NOTE = "typing replaces the name   RIGHT keeps it"

    def __init__(self):
        self.open_ = False
        self.title = ""
        self.value = ""
        self.hint = ""
        self.selected = False             # the preset is selected: a key replaces it
        self._preset = False              # it opened on a name (the box is sized for its note)

    @property
    def open(self) -> bool:
        return self.open_

    def show(self, title: str, initial: str = "", hint: str = "ENTER ok   ESC cancel") -> None:
        self.open_, self.title, self.value, self.hint = True, title, initial, hint
        self.selected = self._preset = bool(initial)

    def handle(self, ev) -> str | None:
        if not self.open_ or ev.type != pygame.KEYDOWN:
            return None
        if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.open_ = False
            return "ok"
        if ev.key == pygame.K_ESCAPE:
            self.open_ = False
            return "cancel"
        if ev.key == pygame.K_BACKSPACE:
            clear = self.selected or getattr(ev, "mod", 0) & (pygame.KMOD_CTRL | pygame.KMOD_META)
            self.value = "" if clear else self.value[:-1]
            self.selected = False
            return None
        if ev.key in (pygame.K_RIGHT, pygame.K_END):
            self.selected = False             # keep the preset, type on after it
            return None
        ch = getattr(ev, "unicode", "")
        if ch and ch.isprintable():
            if self.selected:
                self.value, self.selected = "", False
            if len(self.value) < self.MAX_LEN:
                self.value += ch
        return None

    def hint_text(self, selected: bool | None = None) -> str:
        """The key line: the caller's hint and, while the preset is
        selected (`selected`: as if it were, or not), what a key does to it."""
        sel = self.selected if selected is None else selected
        return f"{self.hint}   {self.SEL_NOTE}" if sel else self.hint

    @staticmethod
    def _wrap_keys(text: Text, s: str, px: int, size: int) -> list:
        """`s` as the lines that fit `px`, broken between its '   '-spaced
        key groups (so 'ENTER ok' stays whole), a group too long for a line
        on its own between its words."""
        out, line = [], ""
        for grp in (g_.strip() for g_ in s.split("   ") if g_.strip()):
            trial = f"{line}   {grp}" if line else grp
            if text.width(trial, size) <= px:
                line = trial
                continue
            if line:
                out.append(line)
            parts = _wrap_px(text, grp, px, size) or [""]
            out += parts[:-1]
            line = parts[-1]
        if line:
            out.append(line)
        return out

    def layout(self, screen, text: Text):
        """(box, key lines): the box 520 px wide at ui 1, widened up to 720
        for a key line or a title that does not fit -- never past the
        screen's edge -- and the key line wrapped to the box when even that
        is too narrow, so no line runs past it (task 45). Sized for the line
        WITH the selected name's note when it opened on a name, so the box
        stays put when the first key drops the note."""
        W, H = screen.get_size()
        u = text.fonts.ui
        pad = int(16 * u)
        widest = self.hint_text(self._preset)
        need = max(text.width(widest, 12), text.width(self.title, 14, bold=True))
        w = int(min(max(520 * u, need + 2 * pad), 720 * u, W - 16))
        n = len(self._wrap_keys(text, widest, w - 2 * pad, 12))
        lines = self._wrap_keys(text, self.hint_text(), w - 2 * pad, 12)
        h = int(94 * u) + max(1, n, len(lines)) * int(16 * u)
        return pygame.Rect(W // 2 - w // 2, H // 2 - h // 2, w, h), lines

    def draw(self, screen, text: Text) -> None:
        if not self.open_:
            return
        u = text.fonts.ui
        box, lines = self.layout(screen, text)
        r = panel(screen, box, alpha=240, accent=True)
        pad = int(16 * u)
        text.blit(screen, self.title, r.x + pad, r.y + int(14 * u), 14, C_SECTION, bold=True)
        pygame.draw.rect(screen, C_SEL_BG, (r.x + pad, r.y + int(40 * u), r.w - 2 * pad, int(30 * u)))
        tx, ty = r.x + int(24 * u), r.y + int(46 * u)
        if self.selected:
            #  the preset, selected: dark on a highlight, and no cursor
            pygame.draw.rect(screen, C_SECTION, (tx - int(4 * u), r.y + int(43 * u),
                                                 text.width(self.value, 16) + int(8 * u),
                                                 int(24 * u)))
            text.blit(screen, self.value, tx, ty, 16, C_PANEL)
        else:
            text.blit(screen, self.value + "_", tx, ty, 16, C_TEXT)
        for i, s in enumerate(lines):
            text.blit(screen, s, r.x + pad, r.y + int(84 * u) + i * int(16 * u), 12, C_DIM)


def key_hint_bar(screen, text: Text, rect, hints, pad_hints=None, title: str = ""):
    """The bottom bar: 'KEY what | KEY what ...' in the menu's key colour."""
    r = panel(screen, rect)
    x, y = r.x + 10, r.y + 6
    if title:
        #  clipped to the width actually available: the caller also writes a
        #  right-aligned status on this line, and a long title ran underneath
        #  it. 240 px is that reservation -- the longest status the garage
        #  writes is "computing wing data... (12 left)", ~224 px at size 12.
        room = max(r.width - 20 - 240, 60)
        while title and text.width(title, 14) > room:
            title = title[:-2]
        text.blit(screen, title, x, y, 14, C_TEXT)
        y += 18
    for row in (hints, pad_hints or []):
        if not row:
            continue
        x = r.x + 10
        for key, what in row:
            x += text.blit(screen, key, x, y, 12, C_KEY) + 6
            x += text.blit(screen, what, x, y, 12, C_DIM) + 16
            if x > r.right - 160:
                break
        y += 17
    return r
