"""drive/cae/form.py -- `Form`, the ParamList a shell view binds its controls to.

A garage model keeps its rows as Params in a list, and every pinned caller
talks to that list the old way: `select_key`, `current`, `nav`, `adjust`,
`activate`, `idx`, `_rect`, `_hits`. `Form` IS that list (it subclasses
`garage_ui.ParamList`), so every one of those keeps working; what it adds is
the geometry an AeroBO work area needs, because the rows are no longer one
column the list draws itself but controls a VIEW places wherever its cards
put them (drive/cae/widgets.py, WorkUI):

  * every control the view lays out is RECORDED here each frame, with its
    rects (`Geom`), whether or not it is on screen. Keyboard and pad order is
    that draw order (`order`), so DOWN walks the page the way it reads and
    reaches a control below the fold -- a pad user has no wheel. Landing on
    one that is off screen asks the shell to scroll it in (`reveal`);
  * a click lands on exactly the control drawn under the cursor and does
    what AeroBO's does: a button fires on the FIRST click, a slider jumps to
    the snapped value under the pointer, a number field's text starts typing
    and its arrow cells step it;
  * a slider drag is COALESCED: motion events only store the value, and
    `begin` applies it once per frame. A `costly` Param (a Designer row: its
    setter re-flies the wing, 31 ms) is not applied during the drag at all,
    only on release;
  * typing into a number field captures every key until ENTER or ESC, so a
    letter typed there is never a shortcut;
  * `locked(p)` is the shell's live-run lock: a locked control refuses
    exactly like a disabled one and says why.

UI-only focusables -- links, disclosures, check boxes, tables -- are
`extras`: action Params with keys starting "ui.", in `order` like any
control but never in `params` or `_hits`, and never locked.

Pure pygame; no model knowledge. PLAN §2.2.
"""

from __future__ import annotations

import math

import pygame

from .. import garage_ui as ui

#: decimals a number field shows (AeroBO widgets.FIELD_DIGITS). What is
#: shown, never what is stored: the value behind it is not rounded.
FIELD_DIGITS = 4
#: how long a refused entry flashes BAD, seconds
FLASH_S = 0.6
#: the characters a number field accepts
_TYPED = set("0123456789.-+eE")
#: the longest buffer a number field holds
_BUF_MAX = 24


def _never(_p):
    return None


def shown(v, nd: int = FIELD_DIGITS) -> str:
    """`v` as a number field shows it: rounded to `nd` decimals, trailing
    zeros dropped ("1.2", "24", "-0.9987"), "—" for no value."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—" if v is None else str(v)
    if not math.isfinite(f):
        return "—"
    s = f"{round(f, nd):.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def number_text(p, nd: int = FIELD_DIGITS) -> str:
    """The text a number field bound to `p` shows: an int row as an int, a
    float row rounded to `nd` decimals (AeroBO shows round(v, 4), not the
    Param's list format)."""
    v = p.get() if p.get is not None else None
    if p.kind == "int":
        try:
            return str(int(round(float(v))))
        except (TypeError, ValueError):
            return "—"
    return shown(v, nd)


def off_value(p):
    """The value that means "off" for a Param with an `off_text` ("no
    limit", "no floor"), or None: its `lo` end by default, its `hi` end when
    `p.off_at == "hi"`, or the number `p.off_at` itself."""
    if not getattr(p, "off_text", ""):
        return None
    at = getattr(p, "off_at", "lo")
    if at == "lo":
        return p.lo
    if at == "hi":
        return p.hi
    return at


def is_off(p) -> bool:
    """Does `p` sit at its "off" end (so its field shows `p.off_text`)?"""
    off = off_value(p)
    if off is None:
        return False
    try:
        v = float(p.get())
    except (TypeError, ValueError):
        return False
    at = getattr(p, "off_at", "lo")
    if at == "lo":
        return v <= float(off) + 1e-12
    if at == "hi":
        return v >= float(off) - 1e-12
    return abs(v - float(off)) <= 1e-12


class Geom:
    """What one control recorded this frame, in screen coordinates.

    `kind` is one of slider number select toggle switch checkbox button
    readout link table disclosure radio. `row` is the whole row (label,
    control, unit, note), `rect` the control itself, `parts` its kind's
    sub-rects: slider {"track"}; number {"text", "up", "down"}; toggle and
    radio {choice: Rect}; select and button {"box"}; switch {"hit"}.
    `visible` is False unless the control lies inside the work-area
    viewport (it is in the keyboard order either way; a control taller than
    the viewport counts as visible while any of it shows)."""

    __slots__ = ("key", "kind", "row", "rect", "parts", "lo", "hi", "step", "visible", "choices", "labels",
                 "digits")

    def __init__(self, key, kind, row, rect, parts=None, *, lo=None, hi=None, step=None, visible=True,
                 choices=None, labels=None, digits=FIELD_DIGITS):
        self.key, self.kind = key, kind
        self.row, self.rect = pygame.Rect(row), pygame.Rect(rect)
        self.parts = parts or {}
        self.lo, self.hi, self.step = lo, hi, step
        self.visible = visible
        self.choices = choices          # select / toggle / radio: the choices drawn ...
        self.labels = labels            # ... and {choice: label}
        self.digits = digits            # number: the decimals it shows

    def x_for(self, value) -> int:
        """Slider only: the pixel column of `value` on the track (the thumb's
        centre). `value_at` is its inverse to within half a pixel, so a click
        at `x_for(v)` snaps back to exactly `v` whenever a step is wider than
        a pixel."""
        tr = self.parts["track"]
        lo, hi = float(self.lo), float(self.hi)
        f = 0.0 if hi <= lo else (float(value) - lo) / (hi - lo)
        return tr.x + int(round(max(0.0, min(1.0, f)) * (tr.w - 1)))

    def value_at(self, x) -> float:
        """Slider only: the (unsnapped) value under pixel column `x`."""
        tr = self.parts["track"]
        f = (float(x) - tr.x) / max(1, tr.w - 1)
        return float(self.lo) + max(0.0, min(1.0, f)) * (float(self.hi) - float(self.lo))

    def snap(self, v, step=None) -> float:
        """`v` on the step grid from `lo` (PLAN §2.2), clamped to the range."""
        step = float(step or self.step or 0.0)
        lo, hi = float(self.lo), float(self.hi)
        if step > 0:
            v = round(lo + round((float(v) - lo) / step) * step, 10)
        return max(lo, min(hi, v))


class _Inner:
    """A control that moves INSIDE itself before the keyboard leaves it (a
    radio group's rows, a table's cursor). Each hook is optional: `step(d)`
    -> True when it moved inside (False at an end: the focus leaves);
    `enter(d)` when the focus arrives from above (d > 0) or below; `activate()`;
    `adjust(d)` (LEFT / RIGHT); `page(d)` (PAGE UP / DOWN); `click(pos)` ->
    the click's result string."""

    __slots__ = ("step", "enter", "activate", "adjust", "page", "click")

    def __init__(self, step=None, enter=None, activate=None, adjust=None, page=None, click=None):
        self.step, self.enter, self.activate = step, enter, activate
        self.adjust, self.page, self.click = adjust, page, click


class Form(ui.ParamList):
    """A ParamList with the geometry of the controls a view drew from it
    (PLAN §2.2). Same constructor; every pinned attribute and method kept."""

    def __init__(self, params, title: str = ""):
        super().__init__(params, title)
        self.extras: dict = {}          # key -> ui.Param for UI-only focusables ("ui." keys)
        self.order: list = []           # every control laid out this frame, draw order
        self.focus_key = None           # the focused key (a param or an extra)
        self.locked = _never            # locked(p) -> the lock reason or None (the shell sets it)
        self.editing = None             # key of the number field being typed into ...
        self.buf = ""                   # ... and its text
        self.reveal = None              # key the shell should scroll into view next frame
        self.pending: dict = {}         # key -> a slider drag value not yet applied
        self.overlay = None             # the Overlay the last WorkUI drew with (drop-downs, popups)
        self.flash = None               # (key, until or None): a refused entry, drawn BAD
        self._geom: dict = {}           # key -> Geom, this frame
        self._ui: list = []             # [(rect, on_click, on_drag)]: "?" dots, sort heads, pagers ...
        self._wheel: list = []          # [(rect, fn(d))]: horizontal-scroll targets (SHIFT + wheel)
        self._inner: dict = {}          # key -> _Inner
        self._help: dict = {}           # key -> (anchor rect, title, text): F1 / "?" on the focused control
        self._cursor: dict = {}         # radio key -> the keyboard cursor inside the group
        self._drag = None               # slider key being dragged, or a callable (a scroll-bar thumb)
        self._fresh = False             # typing: the first key replaces the whole buffer
        self._focus_idx = self.idx      # idx when the focus last moved (a pinned write to idx wins)

    # -- lookups ---------------------------------------------------------------
    def _index(self, key):
        for i, p in enumerate(self.params):
            if p.key == key:
                return i
        return None

    def param(self, key):
        """The Param `key` names: params first, then extras. None if neither."""
        i = self._index(key)
        if i is not None:
            return self.params[i]
        return self.extras.get(key)

    def geom(self, key):
        """What `key` recorded this frame (also when laid out off screen)."""
        return self._geom.get(key)

    def kind_of(self, key):
        g = self._geom.get(key)
        return g.kind if g is not None else None

    def rect_of(self, key):
        """The control's rect while it is VISIBLE, else None -- what a
        tutorial anchor may point at."""
        g = self._geom.get(key)
        return pygame.Rect(g.rect) if g is not None and g.visible else None

    def is_enabled(self, p) -> bool:
        """p.enabled, not locked, and something to do: a setter or an
        action. Extras are never locked (PLAN §2.9.8)."""
        if p is None or p.kind == "label" or not p.enabled:
            return False
        if p.key not in self.extras and self.locked(p):
            return False
        return p.set is not None or p.kind == "action"

    def lock_reason(self, p):
        """Why `p` refuses right now: the live-run lock's sentence, or None
        (a plain disabled control has no sentence of its own)."""
        if p is None or p.key in self.extras:
            return None
        r = self.locked(p)
        return (r if isinstance(r, str) else "locked") if r else None

    # -- a frame ---------------------------------------------------------------
    def begin(self, rect) -> None:
        """Start a frame: apply the coalesced drag values (one set per key,
        the last position; a `costly` one only on release), then forget last
        frame's geometry. `rect` is the visible work area."""
        for key in list(self.pending):
            p = self.param(key)
            if p is not None and getattr(p, "costly", False) and self._drag == key:
                continue
            v = self.pending.pop(key)
            if p is not None:
                self._apply(p, v)
        self._rect = pygame.Rect(rect)
        self._hits = []
        self.order = []
        self._geom = {}
        self._ui = []
        self._wheel = []
        self._inner = {}
        self._help = {}

    def record(self, key, geom: Geom) -> None:
        """A control was laid out. Keyboard order is first-drawn; `_hits`
        takes Params only, and only when visible."""
        if key not in self._geom:
            self.order.append(key)
        self._geom[key] = geom
        if geom.visible:
            i = self._index(key)
            if i is not None:
                self._hits.append((i, geom.row.y, geom.row.h))

    def extra(self, key, label, action, enabled=True, help="") -> "ui.Param":
        """The UI-only action Param `key` (created once, refreshed every
        frame: the action closure is this frame's)."""
        p = self.extras.get(key)
        if p is None:
            p = ui.Param(key, label, None, None, kind="action", help=help)
            self.extras[key] = p
        p.label, p.help, p._enabled = label, help, enabled
        p.set = (lambda _v, fn=action: fn()) if action is not None else None
        return p

    def ui_hit(self, rect, on_click, on_drag=None) -> None:
        """A click target that is not a control ("?" dots, sort heads, pager
        buttons, scroll-bar thumbs). `on_click(pos)` may return the click's
        result string; `on_drag(pos)` follows the mouse until release."""
        self._ui.append((pygame.Rect(rect), on_click, on_drag))

    def ui_wheel(self, rect, fn) -> None:
        """A horizontal-scroll target: `fn(d)` for a SHIFT+wheel notch over it."""
        self._wheel.append((pygame.Rect(rect), fn))

    def inner(self, key, **hooks) -> None:
        self._inner[key] = _Inner(**hooks)

    def help_for(self, key, anchor, title, text) -> None:
        self._help[key] = (pygame.Rect(anchor), title, text)

    # -- focus -----------------------------------------------------------------
    def _focus(self, key, reveal: bool = False) -> None:
        self.focus_key = key
        i = self._index(key)
        if i is not None:
            self.idx = i
        self._focus_idx = self.idx
        g = self._geom.get(key)
        if reveal and g is not None and not g.visible:
            self.reveal = key

    def _live_focus(self):
        """The focused key, or None before anything was focused. A pinned
        caller that wrote `idx` directly has moved the focus to that row."""
        k = self.focus_key
        if k is None or self.idx == self._focus_idx:
            if k in self.extras:
                return k
            i = self._index(k) if k is not None else None
            if k is None or (i is not None and i == self.idx):
                return k
        if not self.params:
            return None
        self.idx = max(0, min(self.idx, len(self.params) - 1))
        return self.params[self.idx].key

    def current(self):
        """The focused Param (a param or an extra). Before anything was
        focused it is the list's own current row (ParamList's rule), without
        taking the focus -- so the first DOWN still lands on the first control
        drawn."""
        k = self._live_focus()
        if k is None:
            return super().current()
        if k != self.focus_key:
            self._focus(k)
        return self.param(k)

    def select_key(self, key: str) -> None:
        if self._index(key) is None and key not in self.extras:
            return
        self._focus(key, reveal=True)

    def _focusable(self, key) -> bool:
        return self.is_enabled(self.param(key))

    def nav(self, d: int) -> None:
        """Move the focus through `order` (draw order, off-screen controls
        included), skipping disabled, locked and read-only entries. A radio
        group or a table moves inside itself first. Before the first draw
        it is the list's own order."""
        if not self.order:
            super().nav(d)
            self.focus_key = self.params[self.idx].key if self.params else None
            self._focus_idx = self.idx
            return
        cur = self._live_focus()
        inner = self._inner.get(cur)
        if inner is not None and inner.step is not None and cur in self.order and inner.step(d):
            return
        n = len(self.order)
        pos = self.order.index(cur) if cur in self.order else (-1 if d > 0 else n)
        for i in range(1, n + 1):
            key = self.order[(pos + d * i) % n]
            if self._focusable(key):
                self._focus(key, reveal=True)
                inner = self._inner.get(key)
                if inner is not None and inner.enter is not None:
                    inner.enter(d)
                return

    # -- the keyboard ------------------------------------------------------------
    def adjust(self, d: int, fine: bool = False) -> bool:
        """LEFT / RIGHT: step a slider or number, cycle a select or toggle,
        flip a switch; a table scrolls sideways. Refused (False) when the
        control is disabled or locked."""
        p = self.current()
        if not self.is_enabled(p):
            return False
        inner = self._inner.get(p.key)
        if inner is not None and inner.adjust is not None:
            return bool(inner.adjust(d))
        return bool(p.adjust(d, fine))

    def activate(self) -> bool:
        """ENTER / CROSS on the focused control: a button fires, a select
        opens its drop-down, a number starts typing, a switch or toggle
        cycles, a radio group picks its cursor row, a table takes its row."""
        p = self.current()
        if not self.is_enabled(p):
            return False
        key = p.key
        inner = self._inner.get(key)
        if inner is not None and inner.activate is not None:
            return bool(inner.activate())
        kind = self.kind_of(key)
        if kind == "select":
            return self.open_select(key)
        if kind == "number":
            self.start_edit(key)
            return True
        return bool(p.activate())

    def page(self, d: int) -> bool:
        """PAGE UP / DOWN on a focused table pages it (True); anything else
        leaves the key to the shell (False: it scrolls the work area)."""
        inner = self._inner.get(self.focus_key)
        if inner is not None and inner.page is not None:
            return bool(inner.page(d))
        return False

    def open_help(self) -> bool:
        """F1 / "?" on the focused control: open its help popup."""
        h = self._help.get(self.focus_key)
        if h is None or self.overlay is None:
            return False
        self.overlay.popup(*h)
        return True

    # -- typing ----------------------------------------------------------------
    def start_edit(self, key) -> None:
        p = self.param(key)
        g = self._geom.get(key)
        self.editing = key
        nd = g.digits if g is not None else FIELD_DIGITS
        self.buf = number_text(p, nd) if p is not None and not is_off(p) else ""
        self._fresh = True

    def cancel_edit(self) -> None:
        self.editing, self.buf, self._fresh = None, "", False

    def _refuse(self, key) -> None:
        self.flash = (key, None)        # WorkUI stamps the deadline with its own clock

    def commit(self) -> bool:
        """Apply the typed buffer: `p.set(clamp(float(buf)))` (int rows
        rounded). An unparsable or non-finite entry is refused: the field
        flashes BAD and keeps its old value. An empty one sets the Param's
        "off" value when it has one ("no limit"), else it is refused too."""
        key, buf = self.editing, self.buf.strip()
        self.cancel_edit()
        p = self.param(key)
        if not self.is_enabled(p) or p.set is None:
            return False
        if not buf:
            v = off_value(p)
            if v is None:
                self._refuse(key)
                return False
        else:
            try:
                v = float(buf)
            except ValueError:
                self._refuse(key)
                return False
            if not math.isfinite(v):
                self._refuse(key)
                return False
        #  task 41's `Param.band()`: an end may be a callable (a band that
        #  moves while the page is open), read now
        lo, hi = p.band() if hasattr(p, "band") else (p.lo, p.hi)
        if lo is not None:
            v = max(v, float(lo))
        if hi is not None:
            v = min(v, float(hi))
        if p.kind == "int":
            v = int(round(v))
        p.set(v)
        return True

    def key_input(self, ev) -> bool:
        """Typing into a number field. While `editing` is set EVERY key is
        taken here (ENTER / TAB commit, ESC cancels, BACKSPACE deletes, the
        number characters append, anything else is swallowed), so a letter
        typed into a field is never a shortcut. With no edit open, a digit,
        "." or "-" on a focused number field starts one. Returns True when
        the key was taken."""
        if ev.type != pygame.KEYDOWN:
            return False
        ch = getattr(ev, "unicode", "") or ""
        if self.editing is None:
            key = self.focus_key
            if self.kind_of(key) == "number" and ch and ch in "0123456789.-" \
                    and self.is_enabled(self.param(key)):
                self.editing, self.buf, self._fresh = key, ch, False
                return True
            return False
        k = ev.key
        if k in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_TAB):
            self.commit()
        elif k == pygame.K_ESCAPE:
            self.cancel_edit()
        elif k == pygame.K_BACKSPACE:
            self.buf = "" if self._fresh else self.buf[:-1]
            self._fresh = False
        elif k == pygame.K_DELETE:
            self.buf, self._fresh = "", False
        elif ch and ch in _TYPED:
            if self._fresh:
                self.buf, self._fresh = ch, False
            elif len(self.buf) < _BUF_MAX:
                self.buf += ch
        return True

    # -- the mouse ---------------------------------------------------------------
    def _in_view(self, pos) -> bool:
        return self._rect is not None and self._rect.collidepoint(pos)

    def hit_key(self, pos):
        """The key of the control under `pos`, or None: the control whose
        own rect (or one of its parts) holds it first, then the one whose
        row does (the topmost: the last drawn). Only inside the visible work
        area. A grid row carries several controls (the Design box: constrain,
        fix, low, high); by the row alone, every click in it went to the last
        one drawn -- the high field -- so the switches and the low field
        never took a click."""
        if not self._in_view(pos):
            return None
        keys = list(reversed(list(self._geom)))
        for key in keys:
            g = self._geom[key]
            if g.rect.collidepoint(pos) or any(isinstance(r, pygame.Rect) and r.collidepoint(pos)
                                               for r in g.parts.values()):
                return key
        for key in keys:
            if self._geom[key].row.collidepoint(pos):
                return key
        return None

    def hit(self, pos) -> int:
        """The PARAM index under `pos`, or -1: by the recorded rects, and
        failing those by the list's own row rule over `_hits`."""
        if self._in_view(pos):
            key = self.hit_key(pos)
            if key is not None:
                i = self._index(key)
                return -1 if i is None else i
        return super().hit(pos)

    @property
    def dragging(self) -> bool:
        return self._drag is not None

    def click(self, pos, fine: bool = False) -> str:
        """A left click (PLAN §2.2). Returns what it did: "action" (a button,
        link or table row fired), "adjust" (a value changed), "edit" (typing
        started), "open" (a drop-down or help popup opened), "ui" (a
        UI-only control: sort, page, disclosure, check box, scroll bar),
        "select" (focus only -- also a refused click on a disabled or
        locked control), or "" (nothing under the pointer)."""
        if not self._in_view(pos):
            return ""
        for rect, on_click, on_drag in reversed(self._ui):
            if rect.collidepoint(pos):
                if self.editing is not None:
                    self.commit()
                r = on_click(pos) if on_click is not None else None
                if on_drag is not None:
                    self._drag = on_drag
                return r or "ui"
        key = self.hit_key(pos)
        if key is None:
            if self.editing is not None:
                self.commit()
            return ""
        if self.editing is not None and self.editing != key:
            self.commit()
        self._focus(key)
        p, g = self.param(key), self._geom[key]
        if not self.is_enabled(p):
            return "select"
        kind = g.kind
        if kind in ("button", "link"):
            if g.rect.collidepoint(pos):
                p.activate()
                return "action"
            return "select"
        if kind in ("checkbox", "disclosure"):
            if g.rect.collidepoint(pos) or g.parts.get("hit", g.rect).collidepoint(pos):
                p.activate()
                return "adjust" if kind == "checkbox" else "ui"
            return "select"
        if kind == "slider":
            zone = g.rect.inflate(0, max(0, g.row.h - g.rect.h))
            if zone.collidepoint(pos):
                v = g.snap(g.value_at(pos[0]), p.fine if fine else None)
                self._apply(p, v)
                self._drag = key
                return "adjust"
            return "select"
        if kind == "number":
            if g.parts.get("up") is not None and g.parts["up"].collidepoint(pos):
                return "adjust" if p.adjust(+1, fine) else "select"
            if g.parts.get("down") is not None and g.parts["down"].collidepoint(pos):
                return "adjust" if p.adjust(-1, fine) else "select"
            if g.rect.collidepoint(pos):
                if self.editing != key:
                    self.start_edit(key)
                return "edit"
            return "select"
        if kind == "select":
            if g.rect.collidepoint(pos):
                return "open" if self.open_select(key) else "select"
            return "select"
        if kind in ("toggle", "radio"):
            for choice, r in g.parts.items():
                if r.collidepoint(pos):
                    if kind == "radio":
                        self._cursor[key] = list(g.parts).index(choice)
                    p.set(choice)
                    return "adjust"
            return "select"
        if kind == "switch":
            if g.parts.get("hit", g.rect).collidepoint(pos):
                p.set(not bool(p.get()))
                return "adjust"
            return "select"
        inner = self._inner.get(key)
        if inner is not None and inner.click is not None:
            return inner.click(pos) or "select"
        return "select"

    def drag(self, pos) -> None:
        """The mouse moved with the button held: a scroll-bar thumb follows
        it; a slider only STORES the snapped value (`begin` applies it)."""
        d = self._drag
        if d is None:
            return
        if callable(d):
            d(pos)
            return
        g, p = self._geom.get(d), self.param(d)
        if g is None or g.kind != "slider" or not self.is_enabled(p):
            return
        self.pending[d] = g.snap(g.value_at(pos[0]))

    def release(self) -> None:
        """Mouse-up: a drag's last value is applied now (a `costly` Param's
        only set of the whole drag)."""
        d, self._drag = self._drag, None
        if isinstance(d, str) and d in self.pending:
            v = self.pending.pop(d)
            p = self.param(d)
            if p is not None:
                self._apply(p, v)

    def wheel_adjust(self, *_args, **_kw) -> bool:
        """Not used: on the shell pages the wheel scrolls, it never changes a
        value (PLAN §2.9.7). Kept so the name answers, and says so."""
        return False

    def hscroll_at(self, pos, d: int) -> bool:
        """A SHIFT+wheel notch at `pos`: the table or code block under it
        scrolls sideways (True), else False."""
        if not self._in_view(pos):
            return False
        for rect, fn in reversed(self._wheel):
            if rect.collidepoint(pos):
                fn(d)
                return True
        return False

    # -- drop-downs ----------------------------------------------------------------
    def open_select(self, key) -> bool:
        g, p = self._geom.get(key), self.param(key)
        if g is None or p is None or self.overlay is None or not self.is_enabled(p):
            return False
        choices = list(g.choices if g.choices is not None else p.choices)
        labels = g.labels or {}
        self.overlay.dropdown(g.parts.get("box", g.rect), choices, [str(labels.get(c, c)) for c in choices],
                              lambda c, pp=p: self._pick(pp, c), current=p.get())
        return True

    def _pick(self, p, choice) -> None:
        if self.is_enabled(p):
            p.set(choice)

    def _apply(self, p, v) -> None:
        if not self.is_enabled(p) or p.set is None:
            return
        if p.kind == "int":
            v = int(round(v))
        p.set(v)
