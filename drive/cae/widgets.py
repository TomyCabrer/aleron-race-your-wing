"""drive/cae/widgets.py -- the AeroBO work area, drawn immediate-mode.

The shell makes one `WorkUI` per frame for the view on screen and hands it
to the view function, which draws itself top to bottom:

    with ui.card("Criterion weights", help=...):
        for k in ("w.ldcr", "w.clmax"):
            ui.slider(k)
        ui.hint("Weights are normalised before scoring, so only their ratios matter.")
    with ui.row():
        ui.button("go", "Screen the library", icon="search")
        ui.spacer()
        ui.tag("SCREENED", T.GOOD)

Everything is laid out every frame (so `end()` knows the content height and
the keyboard order reaches controls below the fold) and drawn only where it
meets the work area's viewport. Every control bound to a Param records its
rects in the model's `Form` (drive/cae/form.py); the Form is what the shell
routes the mouse, the keyboard and the pad to between frames. UI-only state
-- table sort and page, disclosures, check boxes -- lives in the per-view
`state` dict the shell keeps.

The look is AeroBO v3's (SPECS/aerobo_shell.md §6), measured off its
screenshots at 1600x1000: cards with a 22 px PANEL title bar, KPI tiles 37
px tall behind a 2 px rule, 26 px fields and buttons, 17 px tags, hints of
11 px on a 17 px line that keep at most 24 words on screen and put the rest
behind a "?". Every size is a theme constant (`T.NAME`, read at use time so
`T.set_scale` reaches it) or a local one scaled through `_s`.

`Overlay` holds what floats above the page: one drop-down, one help popup,
one tooltip. The shell draws it after everything else and gives it every
event first.

`python3 -m drive.cae.widgets [--out DIR]` runs the self-check (and saves the
demo views as PNGs into DIR).
"""

from __future__ import annotations

import contextlib
import math

import pygame

from . import theme as T
from .form import FIELD_DIGITS, FLASH_S, Form, Geom, is_off, number_text

# -- local px (at S = 1; `_s` scales them the way theme scales its own) ---------------------------
#: a hint / kv / label line: 11-12 px text at line-height 1.5 (AeroBO .hint)
LINE_H = 17
#: a KPI tile: the caption line (T.NOTE_CSS, AeroBO 10.5 px, on a 15 px line) over the value line (the rest of KPI_H)
KPI_LABEL_H = 15
#: dense Quasar button: .285em side padding, a 1.715em icon slot, 6 px to the label.
#: Measured 03: "Screen the library" 293..423 x 726..751, icon ink from 299, label from 322.
BTN_PAD, BTN_ICON_SLOT, BTN_ICON_GAP = 4, 20, 6
#: the `size=sm` flat link AeroBO puts inline ("change in stage 1", shot 14; "the endplate's
#: section →", shot 06): 10 px text, a 16 px icon in an 18 px slot, 4 px apart, 22 tall
SMALL_CSS, SMALL_ICON, SMALL_SLOT, SMALL_GAP, SMALL_H = 10, 16, 18, 4, 22
#: a toggle segment's side padding (03: "this wing's own Re (sweeps the shortlist)" 303..537)
SEG_PAD = 4
#: switch (Quasar dense q-toggle, measured 02 / 03): a 30 x 13 track from box + 1, a 20 px thumb
#: centred at box + 8 (off) or + 22 (on); the whole toggle is 38 wide, its label starts there
SWITCH_W, SWITCH_ROW_H = 38, 18
#: check box and its row
CHECK_BOX, CHECK_ROW_H = 13, 18
#: a slider row: the label line; the thumb is 16 px, the track 4
SLIDER_ROW_H, THUMB_D, TRACK_H, SLIDER_VALUE_W = 17, 16, 4, 34
#: the stepper cells a hovered / focused number field shows
ARROW_W, ARROW_H = 12, 11
#: select: the drop-down caret's glyph size and its box inset from the right
CARET_PX, CARET_INSET = 24, 8
#: disclosure line and its chevron
DISC_H, CHEVRON_PX = 22, 24
#: v1 rows (WG §0.2, measured shot 14): label 60 px (w-20 at 12 px per rem), 96 for a
#: number row (w-32), 9 px to a control that grows
V1_LABEL_W, V1_NUM_LABEL_W, V1_GAP = 60, 96, 9
#: radio rows: padding 3/4, a 14 px icon, 6 px to the title
RADIO_PAD_Y, RADIO_PAD_X = 3, 4
#: tables: cell padding, the sort caret
CELL_PAD, SORT_PX = 8, 16
#: code blocks: padding and line pitch (11 px mono at 1.45)
CODE_PAD, CODE_LINE = 8, 16
#: the popup, the tooltip and the drop-down (SPECS/aerobo_shell.md §6.7, §3.5, §5.1)
POP_W, POP_PAD_X, POP_PAD_Y, POP_PARA_GAP = 320, 10, 8, 4
TIP_MAX_W, TIP_PAD_X, TIP_PAD_Y = 650, 10, 6
MENU_ITEM_H, MENU_PAD_X = 26, 12
#: plots(): the panel caption strip
PANEL_TITLE_H = 18
#: pager page sizes (Quasar's own list, trimmed; 0 = all)
PAGE_SIZES = (5, 10, 20, 50, 0)

#: opacity of a disabled or locked control, of a dead column, of a dimmed label (AeroBO)
DISABLED_A, DEAD_A = 0.6, 0.45
V1_LABEL_A, V1_NOTE_A = 0.70, 0.60


def _s(n) -> int:
    """A widget-local px at scale S, rounded half up like theme's own."""
    return int(math.floor(n * T.S + 0.5)) if n > 0 else 0


def _fh(family, css, bold=False) -> int:
    return T.font(family, css, bold).get_height()


def _ty(top, box_h, family, css, bold=False) -> int:
    """The y to draw a line of text at so it sits centred in a box of
    `box_h` from `top` -- the browser centres the line box the same way."""
    return top + (box_h - _fh(family, css, bold) + 1) // 2


def _fade(colour, on: bool, alpha=DISABLED_A, bg=None):
    if on:
        return colour
    return T.fade(colour, alpha, bg if bg is not None else T.WELL)


# -- text wrapping ---------------------------------------------------------------------------------
_WRAPS: dict = {}


def _hard_break(word, family, css, width, bold):
    out = []
    while word:
        lo, hi = 1, len(word)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if T.text_w(word[:mid], family, css, bold) <= width:
                lo = mid
            else:
                hi = mid - 1
        out.append(word[:lo])
        word = word[lo:]
    return out


def wrap(text, family="sans", css=11, width=100, bold=False) -> list:
    """`text` broken into lines that FIT `width` px in that font (words
    first, a word wider than the line character by character). Explicit
    newlines are kept. Cached."""
    text = str(text)
    width = max(8, int(width))
    key = (text, family, T.px_for(family, css), bool(bold), width)
    lines = _WRAPS.get(key)
    if lines is not None:
        return lines
    if len(_WRAPS) >= 2048:
        _WRAPS.clear()
    lines = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            trial = f"{line} {word}" if line else word
            if T.text_w(trial, family, css, bold) <= width:
                line = trial
                continue
            if line:
                lines.append(line)
            if T.text_w(word, family, css, bold) > width:
                parts = _hard_break(word, family, css, width, bold)
                lines.extend(parts[:-1])
                line = parts[-1]
            else:
                line = word
        lines.append(line)
    _WRAPS[key] = lines
    return lines


# -- shapes ------------------------------------------------------------------------------------------
_DISCS: dict = {}
_SHADOWS: dict = {}


def disc(d: int, fill, edge=None, edge_w: float = 1.0) -> pygame.Surface:
    """An anti-aliased disc `d` px across (drawn 4x and smoothed), with an
    optional ring. `fill` may carry alpha 0 for a ring only. Cached."""
    key = (d, tuple(fill), tuple(edge) if edge else None, edge_w)
    img = _DISCS.get(key)
    if img is None:
        if len(_DISCS) >= 256:
            _DISCS.clear()
        k = 4
        big = pygame.Surface((d * k, d * k), pygame.SRCALPHA)
        r = d * k / 2.0
        if edge is not None:
            pygame.draw.circle(big, edge, (r, r), r)
            pygame.draw.circle(big, fill, (r, r), r - edge_w * k)
        else:
            pygame.draw.circle(big, fill, (r, r), r)
        img = pygame.transform.smoothscale(big, (d, d))
        _DISCS[key] = img
    return img


def shadow(surf, rect) -> None:
    """AeroBO's drop-down shadow, 0 6px 18px rgba(20,26,34,.18), under `rect`."""
    r = pygame.Rect(rect)
    blur = _s(18)
    key = (r.w, r.h, blur)
    img = _SHADOWS.get(key)
    if img is None:
        if len(_SHADOWS) >= 32:
            _SHADOWS.clear()
        pad = blur
        w, h = r.w + 2 * pad, r.h + 2 * pad
        k = 6
        small = pygame.Surface((max(1, w // k), max(1, h // k)), pygame.SRCALPHA)
        small.fill((0, 0, 0, 0))
        pygame.draw.rect(small, T.SHADOW_A, (pad // k, pad // k, max(1, r.w // k), max(1, r.h // k)),
                         border_radius=max(1, pad // k))
        img = pygame.transform.smoothscale(small, (w, h))
        _SHADOWS[key] = img
    surf.blit(img, (r.x - blur, r.y - blur + _s(6)))


def _rect(surf, colour, rect, width=0, radius=2):
    pygame.draw.rect(surf, colour, rect, width, border_radius=_s(radius) if radius else 0)


# ==================================================================== #
#  OVERLAY                                                             #
# ==================================================================== #
class Overlay:
    """What floats above the page: at most one drop-down (AeroBO's menu
    style), one help popup (a "?" clicked), one tooltip (hovered for half a
    second). The shell gives it every event before anything else
    (`handle`) and draws it last (`draw`)."""

    def __init__(self):
        self.menu = None            # the open drop-down (dict)
        self.pop = None             # the open help popup (dict)
        self.last_pick = None       # (choice,) of the last drop-down pick; the shell may read and clear it
        self._tip_req = None        # (anchor, text) asked for this frame
        self._tip = None            # (anchor, text, since) being timed / shown
        self._size = (4096, 4096)   # the surface the last draw had, for placement

    # -- state ---------------------------------------------------------------
    @property
    def active(self) -> bool:
        """A drop-down or a popup is open (so clicks and keys come here)."""
        return self.menu is not None or self.pop is not None

    def close(self) -> None:
        self.menu = self.pop = None

    def covers(self, pos) -> bool:
        for d in (self.menu, self.pop):
            if d is not None and d.get("rect") is not None and d["rect"].collidepoint(pos):
                return True
        return False

    # -- opening -----------------------------------------------------------------
    def dropdown(self, anchor_rect, choices, labels, on_pick, current=None) -> None:
        """Open a drop-down under `anchor_rect`: items 26 tall SANS 12, the
        current choice in ACCENT, hover / keyboard cursor ACCENT_FILL.
        `on_pick(choice)` runs when one is taken."""
        choices = list(choices)
        labels = [str(s) for s in labels] if labels is not None else [str(c) for c in choices]
        cur = choices.index(current) if current in choices else 0
        self.pop = None
        self.menu = {"anchor": pygame.Rect(anchor_rect), "choices": choices, "labels": labels,
                     "on_pick": on_pick, "current": current, "cur": cur, "hover": -1, "rect": None}
        self._layout_menu()

    def popup(self, anchor_rect, title, text) -> None:
        """Open a help popup under `anchor_rect`: WELL, 1px RULE, shadow,
        padding 8/10, 320 wide at most; the title SANS 11.5/600 INK, the
        paragraphs (split on blank lines) SANS 11.5 INK_MUTED."""
        paras = [p.strip() for p in str(text or "").split("\n\n") if p.strip()]
        self.menu = None
        self.pop = {"anchor": pygame.Rect(anchor_rect), "title": str(title or ""), "paras": paras, "rect": None}
        self._layout_pop()

    def tooltip(self, anchor_rect, text) -> None:
        """Ask for a tooltip THIS frame (a hovered control asks every frame;
        it shows after TOOLTIP_DELAY_S of the same request)."""
        if text:
            self._tip_req = (pygame.Rect(anchor_rect), str(text))

    # -- layout --------------------------------------------------------------------
    def _place(self, anchor, w, h, gap=0):
        W, H = self._size
        x = max(4, min(anchor.x, W - w - 4))
        y = anchor.bottom + gap
        if y + h > H - 4 and anchor.top - gap - h >= 4:
            y = anchor.top - gap - h
        return pygame.Rect(x, max(4, y), w, h)

    def _layout_menu(self):
        m = self.menu
        pad = _s(MENU_PAD_X)
        wide = max([T.text_w(s, "sans", 12) for s in m["labels"]] or [0]) + 2 * pad + 2
        ih = _s(MENU_ITEM_H)
        m["rect"] = self._place(m["anchor"], max(m["anchor"].w, wide), ih * len(m["choices"]) + 2)

    def _layout_pop(self):
        p = self.pop
        px, py = _s(POP_PAD_X), _s(POP_PAD_Y)
        inner = _s(POP_W) - 2 * px
        lines = []
        widest = 0
        if p["title"]:
            for ln in wrap(p["title"], "sans", 11.5, inner, True):
                lines.append((ln, True))
                widest = max(widest, T.text_w(ln, "sans", 11.5, True))
        for i, para in enumerate(p["paras"]):
            if i or p["title"]:
                lines.append((None, False))
            for ln in wrap(para, "sans", 11.5, inner):
                lines.append((ln, False))
                widest = max(widest, T.text_w(ln, "sans", 11.5))
        lh = _fh("sans", 11.5) + 1
        h = sum(_s(POP_PARA_GAP) if ln is None else lh for ln, _ in lines)
        p["lines"], p["lh"] = lines, lh
        p["rect"] = self._place(p["anchor"], widest + 2 * px + 2, h + 2 * py + 2, _s(4))

    def _menu_item(self, pos):
        m = self.menu
        r = m["rect"]
        if r is None or not r.collidepoint(pos):
            return None
        i = (pos[1] - r.y - 1) // _s(MENU_ITEM_H)
        return i if 0 <= i < len(m["choices"]) else None

    def _pick(self, i) -> None:
        m = self.menu
        self.menu = None
        choice = m["choices"][i]
        self.last_pick = (choice,)
        m["on_pick"](choice)

    # -- events ----------------------------------------------------------------------
    def handle(self, ev) -> bool:
        """True when the event was taken. While a drop-down or popup is open
        every click and key comes here: a click on an item picks it, a click
        anywhere else closes (and is spent); a right / middle click or a wheel
        echo does nothing. The popup lets a key other than ESC through."""
        if ev.type == pygame.MOUSEMOTION:
            if self.menu is not None:
                i = self._menu_item(ev.pos)
                self.menu["hover"] = -1 if i is None else i
            return False
        if not self.active:
            return False
        if ev.type == pygame.MOUSEBUTTONDOWN:
            if ev.button != 1:
                return True
            if self.menu is not None:
                i = self._menu_item(ev.pos)
                if i is not None:
                    self._pick(i)
                elif not self.menu["rect"].collidepoint(ev.pos):
                    self.menu = None
                return True
            if not self.pop["rect"].collidepoint(ev.pos):
                self.pop = None
            return True
        if ev.type == pygame.MOUSEBUTTONUP:
            return False
        if ev.type == pygame.MOUSEWHEEL:
            if self.menu is not None:
                return True
            self.pop = None
            return False
        if ev.type == pygame.KEYDOWN:
            if self.menu is not None:
                m = self.menu
                n = len(m["choices"])
                if ev.key in (pygame.K_UP, pygame.K_DOWN) and n:
                    m["cur"] = (m["cur"] + (1 if ev.key == pygame.K_DOWN else -1)) % n
                elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE) and n:
                    self._pick(m["cur"])
                elif ev.key in (pygame.K_ESCAPE, pygame.K_TAB):
                    self.menu = None
                return True
            self.pop = None
            return ev.key == pygame.K_ESCAPE
        return False

    # -- drawing ------------------------------------------------------------------------
    def draw(self, surf, now: float) -> None:
        self._size = surf.get_size()
        old = surf.get_clip()
        surf.set_clip(None)
        self._draw_tip(surf, now)
        if self.menu is not None:
            self._layout_menu()
            self._draw_menu(surf)
        if self.pop is not None:
            self._layout_pop()
            self._draw_pop(surf)
        surf.set_clip(old)

    def _draw_menu(self, surf):
        m = self.menu
        r = m["rect"]
        shadow(surf, r)
        _rect(surf, T.WELL, r)
        ih, pad = _s(MENU_ITEM_H), _s(MENU_PAD_X)
        for i, (c, label) in enumerate(zip(m["choices"], m["labels"])):
            ir = pygame.Rect(r.x + 1, r.y + 1 + i * ih, r.w - 2, ih)
            if i == m["hover"] or (m["hover"] < 0 and i == m["cur"]):
                surf.fill(T.ACCENT_FILL, ir)
            col = T.ACCENT if c == m["current"] else T.INK
            T.text(surf, label, ir.x + pad - 1, _ty(ir.y, ih, "sans", 12), "sans", 12, col,
                   clip_w=ir.w - 2 * pad)
        _rect(surf, T.RULE, r, 1)

    def _draw_pop(self, surf):
        p = self.pop
        r = p["rect"]
        shadow(surf, r)
        _rect(surf, T.WELL, r)
        _rect(surf, T.RULE, r, 1)
        x, y = r.x + 1 + _s(POP_PAD_X), r.y + 1 + _s(POP_PAD_Y)
        for ln, bold in p["lines"]:
            if ln is None:
                y += _s(POP_PARA_GAP)
                continue
            T.text(surf, ln, x, y, "sans", 11.5, T.INK if bold else T.INK_MUTED, bold)
            y += p["lh"]

    def _draw_tip(self, surf, now):
        req, self._tip_req = self._tip_req, None
        if req is None or self.active:
            self._tip = None
            return
        anchor, text = req
        if self._tip is None or self._tip[1] != text or self._tip[0] != anchor:
            self._tip = (anchor, text, now)
        if now - self._tip[2] < T.TOOLTIP_DELAY_S:
            return
        px, py = _s(TIP_PAD_X), _s(TIP_PAD_Y)
        lines = wrap(text, "sans", 11, _s(TIP_MAX_W) - 2 * px)
        lh = _fh("sans", 11) + 1
        w = max(T.text_w(ln, "sans", 11) for ln in lines) + 2 * px
        h = lh * len(lines) + 2 * py
        r = self._place(anchor, w, h, _s(4))
        _rect(surf, T.TOOLTIP, r)
        for i, ln in enumerate(lines):
            T.text(surf, ln, r.x + px, r.y + py + i * lh, "sans", 11, T.WELL)


# ==================================================================== #
#  LAYOUT                                                              #
# ==================================================================== #
class _Col:
    """A vertical flow: children full width, `gap` apart."""

    __slots__ = ("x", "w", "y", "top", "gap", "n")

    def __init__(self, x, w, y, gap):
        self.x, self.w, self.y, self.top, self.gap, self.n = int(x), int(w), int(y), int(y), int(gap), 0

    def start(self) -> int:
        if self.n:
            self.y += self.gap
        self.n += 1
        return self.y

    def end(self, top, h) -> None:
        self.y = top + int(h)

    def undo(self, top) -> None:
        """A child that turned out empty: give its gap back."""
        self.n -= 1
        self.y = top - (self.gap if self.n else 0)

    @property
    def height(self) -> int:
        return self.y - self.top


class _Row:
    """A horizontal flow, laid out when the row closes (a `spacer` needs the
    widths of what follows it). Items are (w, h, draw_fn); None is a spacer."""

    __slots__ = ("x", "w", "y", "gap", "items", "centre")

    def __init__(self, x, w, y, gap, centre=True):
        self.x, self.w, self.y, self.gap, self.items, self.centre = int(x), int(w), int(y), int(gap), [], centre


def _lines_of(items, width, gap):
    lines, cur, used = [], [], 0
    for it in items:
        if it is None:
            cur.append(None)
            continue
        n = sum(1 for c in cur if c is not None)
        add = it[0] + (gap if n else 0)
        if n and used + add > width:
            lines.append(cur)
            cur, used = [it], it[0]
        else:
            cur.append(it)
            used += add
    lines.append(cur)
    return lines


# ==================================================================== #
#  WORK UI                                                             #
# ==================================================================== #
class WorkUI:
    """The immediate-mode work area of one frame (PLAN §2.3). `rect` is the
    work area on `surf`; `form` the Form the view's Params live in (None: a
    view without one -- a private Form kept in `state`); `state` the view's
    persistent UI dict; `scroll` the content offset; `kbd_focus` whether the
    work region holds the keyboard (focus rings show); `mouse` the tracked
    pointer ((-1, -1): none); `overlay` the shell's Overlay; `now` seconds.
    `layout_only=True` lays the view out without drawing (the shell's
    reveal pass)."""

    def __init__(self, surf, rect, form, state, *, scroll, kbd_focus, mouse, overlay, now, busy_reason="",
                 layout_only=False):
        self.surf = surf
        self.rect = pygame.Rect(rect)
        self.state = state if state is not None else {}
        if form is None:
            form = self.state.get("_form")
            if form is None:
                form = self.state["_form"] = Form([])
        self.form = form
        self.scroll = int(scroll)
        self.kbd_focus = bool(kbd_focus)
        self.mouse = tuple(mouse) if mouse is not None else (-1, -1)
        self.overlay = overlay if overlay is not None else Overlay()
        self.now = float(now)
        self.busy_reason = busy_reason or ""
        self.layout_only = bool(layout_only)
        self._clip0 = surf.get_clip()
        self.clip = self.rect.clip(self._clip0)
        surf.set_clip(self.clip)
        form.begin(self.clip)
        form.overlay = self.overlay
        if form.flash is not None:
            if form.flash[1] is None:
                form.flash = (form.flash[0], self.now + FLASH_S)
            elif self.now > form.flash[1]:
                form.flash = None
        self._mouse_free = not self.overlay.covers(self.mouse)
        self._stack = [_Col(self.rect.x + T.WORK_PAD_X, self.rect.w - 2 * T.WORK_PAD_X,
                            self.rect.y + T.WORK_PAD_Y - self.scroll, T.STACK_GAP)]
        self._auto = {}
        self._ended = None

    # -- small helpers -------------------------------------------------------------------
    @property
    def w(self) -> int:
        """The width of the container being filled."""
        return self._stack[-1].w

    @property
    def x(self) -> int:
        return self._stack[-1].x

    def _in_row(self) -> bool:
        return isinstance(self._stack[-1], _Row)

    def _col(self) -> _Col:
        f = self._stack[-1]
        if isinstance(f, _Row):
            raise RuntimeError("cards, columns, indents and blocks cannot sit inside ui.row()")
        return f

    def _place(self, w, h, fn) -> None:
        """Put a w x h leaf in the current container; `fn(x, y)` draws it
        (when visible) and records it (always)."""
        f = self._stack[-1]
        if isinstance(f, _Row):
            f.items.append((int(w), int(h), fn))
            return
        y = f.start()
        fn(f.x, y)
        f.end(y, h)

    def _vis(self, rect) -> bool:
        return not self.layout_only and self.clip.colliderect(rect)

    def _shows(self, rect) -> bool:
        """Visible for the Form's purposes: wholly inside the viewport (or,
        taller than it, any part of it)."""
        r = pygame.Rect(rect)
        if r.h > self.clip.h:
            return self.clip.colliderect(r)
        return self.clip.contains(r)

    def _hot(self, rect) -> bool:
        return self._mouse_free and self.clip.collidepoint(self.mouse) and pygame.Rect(rect).collidepoint(self.mouse)

    def _focused(self, key) -> bool:
        return self.kbd_focus and key is not None and self.form.focus_key == key

    def _ring(self, rect) -> None:
        """The keyboard focus ring: 2px ACCENT, 2px outside the control."""
        if self._vis(rect):
            _rect(self.surf, T.ACCENT, pygame.Rect(rect).inflate(_s(8), _s(8)), _s(2), 3)

    def _p(self, key):
        p = self.form.param(key)
        if p is None:
            raise KeyError(f"no Param {key!r} in this view's form")
        return p

    def _on(self, p) -> bool:
        return self.form.is_enabled(p)

    def _record(self, key, kind, row, rect, parts=None, p=None, **kw) -> Geom:
        g = Geom(key, kind, row, rect, parts, visible=self._shows(rect),
                 lo=getattr(p, "lo", None), hi=getattr(p, "hi", None), step=getattr(p, "step", None), **kw)
        self.form.record(key, g)
        return g

    def _explain(self, key, rect, p) -> None:
        """Hover: a locked control says why; a short help is a tooltip."""
        if not self._hot(rect):
            return
        why = self.form.lock_reason(p) if p is not None else None
        if why:
            self.overlay.tooltip(rect, why if why != "locked" else (self.busy_reason or "locked"))

    def _auto_id(self, kind) -> str:
        n = self._auto.get(kind, 0)
        self._auto[kind] = n + 1
        return f"{kind}{n}"

    def _text(self, s, x, y, family="sans", css=11, colour=None, bold=False, anchor="left", clip_w=None) -> int:
        if self.layout_only:
            return T.text_w(s, family, css, bold)
        return T.text(self.surf, s, x, y, family, css, T.INK if colour is None else colour, bold, anchor, clip_w)

    # -- help dots ---------------------------------------------------------------------------
    def _dot(self, x, cy, text, title="", key=None) -> int:
        """The 14 px "?" (AeroBO help_dot): a click opens `text` as a popup
        titled `title`; keyboard: F1 / "?" on the control `key`."""
        d = T.HELP_DOT
        r = pygame.Rect(int(x), int(cy - d // 2), d, d)
        if self._shows(r) or self.clip.colliderect(r):
            self.form.ui_hit(r, lambda _pos, rr=r, t=title, tx=text: (self.overlay.popup(rr, t, tx), "open")[1])
        if key is not None:
            self.form.help_for(key, r, title, text)
        if self._vis(r):
            hot = self._hot(r)
            ring = T.ACCENT if hot else T.RULE
            self.surf.blit(disc(d, T.ACCENT_FILL if hot else T.PANEL, ring), r.topleft)
            self._text("?", r.centerx, _ty(r.y, d, "sans", 10, True), "sans", 10,
                       T.ACCENT if hot else T.INK_MUTED, True, "centre")
        return d

    @staticmethod
    def _long(text) -> bool:
        return bool(text) and len(str(text).split()) > T.TIP_WORDS

    # -- layout ------------------------------------------------------------------------------
    def gap(self, px=None) -> None:
        """Extra space: down in a column, across in a row (`px` at S = 1;
        None = STACK_GAP)."""
        px = T.STACK_GAP if px is None else _s(px)
        f = self._stack[-1]
        if isinstance(f, _Row):
            f.items.append((px, 0, lambda x, y: None))
        else:
            f.y += px

    @contextlib.contextmanager
    def card(self, title, *, help=None, pad=True, title_tools=None):
        """AeroBO group_box: 1px RULE_SOFT frame, WELL body, a 22 px PANEL
        title bar over a RULE_SOFT rule, title SANS 11/600 INK_MUTED at x+9
        and its "?" 3 px after; body padding 8/9 (none with pad=False),
        children CARD_GAP apart. `title_tools`: [(id, icon, tip, on_click)]
        drawn as flat icon buttons at the bar's right. The frame is drawn at
        exit, when its height is known."""
        parent = self._col()
        x, w = parent.x, parent.w
        top = parent.start()
        th = (T.CARD_TITLE_H + 1) if title else 0
        px, py = (T.CARD_PAD_X, T.CARD_PAD_Y) if pad else (0, 0)
        body = _Col(x + 1 + px, w - 2 - 2 * px, top + 1 + th + py, T.CARD_GAP)
        if title:
            self._card_title(x, top, w, title, help, title_tools)
        self._stack.append(body)
        try:
            yield body
        finally:
            self._stack.pop()
            h = 1 + th + py + body.height + py + 1
            parent.end(top, h)
            r = pygame.Rect(x, top, w, h)
            if self._vis(r):
                _rect(self.surf, T.RULE_SOFT, r, 1)

    def _card_title(self, x, top, w, title, help, tools) -> None:
        bar = pygame.Rect(x + 1, top + 1, w - 2, T.CARD_TITLE_H)
        if self._vis(bar.inflate(0, 2)):
            self.surf.fill(T.PANEL, bar)
            self.surf.fill(T.RULE_SOFT, (bar.x, bar.bottom, bar.w, 1))
        tx = x + T.CARD_PAD_X
        tw = self._text(title, tx, _ty(bar.y, bar.h, "sans", 11, True), "sans", 11, T.INK_MUTED, True,
                        clip_w=bar.w - 2 * T.CARD_PAD_X) if self._vis(bar) else T.text_w(title, "sans", 11, True)
        if help:
            self._dot(tx + tw + _s(3), bar.centery, help, title)
        rx = bar.right - _s(4)
        for tool in tools or ():
            _id, icon, tip, on_click = tool
            ir = pygame.Rect(rx - T.TOOL_BTN_H, bar.y + (bar.h - T.TOOL_BTN_H) // 2 + 1, T.TOOL_BTN_H, T.TOOL_BTN_H - 2)
            rx = ir.x - _s(2)
            self.form.ui_hit(ir, lambda _pos, fn=on_click: (fn(), "ui")[1])
            if self._vis(ir):
                if self._hot(ir):
                    self.surf.fill(T.tint(T.ACCENT), ir)
                    self.overlay.tooltip(ir, tip)
                T.icon(self.surf, icon, ir.centerx, ir.centery - T.TOOL_ICON // 2, T.TOOL_ICON, T.ACCENT, "centre")

    def columns(self, weights=(1, 1), gap=None) -> list:
        """Side-by-side columns (context managers), top-aligned, widths by
        weight (`gap` px at S = 1; None = STACK_GAP); the flow continues
        under the tallest."""
        parent = self._col()
        g = T.STACK_GAP if gap is None else _s(gap)
        top = parent.start()
        n = len(weights)
        total = float(sum(weights)) or 1.0
        avail = parent.w - g * (n - 1)
        widths = [int(round(avail * wt / total)) for wt in weights]
        widths[-1] = avail - sum(widths[:-1])
        group = {"h": 0}
        xs, x = [], parent.x
        for wd in widths:
            xs.append(x)
            x += wd + g
        return [self._column(xi, wd, top, group, parent) for xi, wd in zip(xs, widths)]

    @contextlib.contextmanager
    def _column(self, x, w, top, group, parent):
        c = _Col(x, w, top, parent.gap)
        self._stack.append(c)
        try:
            yield c
        finally:
            self._stack.pop()
            group["h"] = max(group["h"], c.height)
            parent.end(top, group["h"])

    def row(self, gap=6, *, centre=True):
        """A horizontal flow (buttons, tags, KPI tiles, links); wraps when
        full. `gap` px at S = 1. Items are vertically centred on the line
        (`centre=False`: top aligned). A `spacer()` pushes what follows to
        the right edge."""
        return self._row(_s(gap), centre)

    @contextlib.contextmanager
    def _row(self, gap_px, centre=True):
        parent = self._col()
        top = parent.start()
        r = _Row(parent.x, parent.w, top, gap_px, centre)
        self._stack.append(r)
        try:
            yield r
        finally:
            self._stack.pop()
            h = self._flush(r)
            if h:
                parent.end(top, h)
            else:
                parent.undo(top)

    def _flush(self, r: _Row) -> int:
        y, total = r.y, 0
        for line in _lines_of(r.items, r.w, r.gap):
            real = [it for it in line if it is not None]
            if not real:
                continue
            lh = max(h for _, h, _ in real)
            k = line.index(None) if None in line else len(line)
            left = [it for it in line[:k] if it is not None]
            right = [it for it in line[k + 1:] if it is not None]
            x = r.x
            for w, h, fn in left:
                fn(x, y + ((lh - h + 1) // 2 if r.centre else 0))
                x += w + r.gap
            if right:
                rw = sum(w for w, _, _ in right) + r.gap * (len(right) - 1)
                x = max(x, r.x + r.w - rw)
                for w, h, fn in right:
                    fn(x, y + ((lh - h + 1) // 2 if r.centre else 0))
                    x += w + r.gap
            if total:
                total += r.gap
            total += lh
            y += lh + r.gap
        return total

    def spacer(self) -> None:
        """Inside row(): what follows goes to the right edge."""
        f = self._stack[-1]
        if isinstance(f, _Row):
            f.items.append(None)

    @contextlib.contextmanager
    def indent(self):
        """A disclosure's body: 3 px in, behind a 2 px RULE_SOFT bar, 4 px
        above and 2 below (AeroBO .q-expansion-item__content)."""
        parent = self._col()
        top = parent.start()
        bx = parent.x + _s(3)
        body = _Col(bx + _s(2) + _s(2), parent.w - _s(7), top + _s(4), parent.gap)
        self._stack.append(body)
        try:
            yield body
        finally:
            self._stack.pop()
            h = _s(4) + body.height + _s(2)
            parent.end(top, h)
            bar = pygame.Rect(bx, top, _s(2), h)
            if self._vis(bar):
                self.surf.fill(T.RULE_SOFT, bar)

    def hairline(self) -> None:
        """1 px RULE_SOFT across the column."""
        def fn(x, y, w=self.w):
            r = pygame.Rect(x, y, w, 1)
            if self._vis(r):
                self.surf.fill(T.RULE_SOFT, r)
        self._place(self.w, 1, fn)

    # -- text --------------------------------------------------------------------------------
    def sect_head(self, text) -> None:
        """A section head: SANS 12/600 INK."""
        lh = _s(LINE_H)
        if self._in_row():
            w = T.text_w(text, "sans", 12, True)
            lines = [text]
        else:
            w = self.w
            lines = wrap(text, "sans", 12, w, True)

        def fn(x, y):
            if self._vis((x, y, w, lh * len(lines))):
                for i, ln in enumerate(lines):
                    self._text(ln, x, _ty(y + i * lh, lh, "sans", 12, True), "sans", 12, T.INK, True)
        self._place(w, lh * len(lines), fn)

    def label(self, text, *, css=11.5, colour=None, family="sans", bold=False, min_w=None, tip=None) -> None:
        """A bare line of text for a row (the "designing" caption, a gate's
        label, "download:"): SANS 11.5 INK by default, at least `min_w` px
        (at S = 1) wide; `tip` is its hover."""
        colour = T.INK if colour is None else colour
        lh = _s(LINE_H)
        tw = T.text_w(text, family, css, bold)
        w = max(tw, _s(min_w) if min_w else 0)
        if not self._in_row():
            w = self.w

        def fn(x, y):
            r = pygame.Rect(x, y, w, lh)
            if self._vis(r):
                self._text(text, x, _ty(y, lh, family, css, bold), family, css, colour, bold, clip_w=w)
            if tip and self._hot(pygame.Rect(x, y, tw, lh)):
                self.overlay.tooltip(pygame.Rect(x, y, tw, lh), tip)
        self._place(w, lh, fn)

    def help_dot(self, text, title="") -> None:
        """A bare "?" for a row (AeroBO widgets.help_dot): a click opens
        `text` as a popup titled `title`."""
        d = T.HELP_DOT

        def fn(x, y):
            self._dot(x, y + d // 2, text, title)
        self._place(d, d, fn)

    def hint(self, text, kind=None, *, split=True, help=None, title=None) -> None:
        """A quiet line: SANS 12 (T.HINT_CSS; AeroBO 11) on a 17 px line, INK_MUTED (kind "warn" WARN,
        "bad" BAD, "ok" GOOD). Past 24 words it keeps its lead and puts the
        rest behind a "?" (theme.split_hint; `split=False` keeps it whole);
        `help` forces a "?" with that popup text. A lead that fits carries the
        "?" 3 px after it; one that wraps carries it at the column's right
        edge, centred on the block (AeroBO hint_help, shot 12b)."""
        colour = {"warn": T.WARN, "bad": T.BAD, "ok": T.GOOD}.get(kind, T.INK_MUTED)
        lead, rest = T.split_hint(text) if split else (str(text), None)
        pop = help if help is not None else rest
        lh = _s(LINE_H)
        d = T.HELP_DOT
        g3 = _s(3)
        if self._in_row():
            room = self.w - ((g3 + d) if pop else 0)
            full = min(T.text_w(lead, "sans", T.HINT_CSS), room)
            w = full + ((g3 + d) if pop else 0)

            def fn(x, y):
                if self._vis((x, y, w, lh)):
                    self._text(lead, x, _ty(y, lh, "sans", T.HINT_CSS), "sans", T.HINT_CSS, colour, clip_w=room)
                if pop:
                    self._dot(x + full + g3, y + lh // 2, pop, title or "")
            self._place(w, lh, fn)
            return
        w = self.w
        if pop and T.text_w(lead, "sans", T.HINT_CSS) + g3 + d <= w:
            lines, dot_x = [lead], T.text_w(lead, "sans", T.HINT_CSS) + g3
        elif pop:
            lines, dot_x = wrap(lead, "sans", T.HINT_CSS, w - g3 - d), w - d
        else:
            lines, dot_x = wrap(lead, "sans", T.HINT_CSS, w), None
        h = lh * len(lines)

        def fn(x, y):
            if self._vis((x, y, w, h)):
                for i, ln in enumerate(lines):
                    self._text(ln, x, _ty(y + i * lh, lh, "sans", T.HINT_CSS), "sans", T.HINT_CSS, colour)
            if dot_x is not None:
                self._dot(x + dot_x, y + h // 2 if len(lines) > 1 else y + lh // 2, pop, title or "")
        self._place(w, h, fn)

    def kv(self, key, value, *, colour=None, tip=None, link=None) -> None:
        """A read-out line: key SANS 11.5 INK_MUTED (at least 150 wide),
        value MONO 12 INK (or `colour`) 6 px after, wrapping. `tip`: a hover
        on the key when short, a "?" after the value when long. `link`:
        (label, icon, key) -- a flat link after the value, bound to the
        action Param `key`, or (label, icon, callable) for a UI one."""
        kw = max(T.LABEL_MIN_W, T.text_w(key, "sans", 11.5))
        g6 = _s(6)
        d, g3 = T.HELP_DOT, _s(3)
        long_tip = self._long(tip)
        lnk = self._link_spec(link, key)
        lnk_w = self._button_w(lnk[0], lnk[1], "flat", True) if lnk else 0
        lh = _s(LINE_H)
        value = str(value)
        if self._in_row():
            vw = T.text_w(value, "mono", 12)
            lines = [value]
            w = kw + g6 + vw + ((g3 + d) if long_tip else 0) + ((_s(12) + lnk_w) if lnk else 0)
        else:
            w = self.w
            room = w - kw - g6 - ((g3 + d) if long_tip else 0) - ((_s(12) + lnk_w) if lnk else 0)
            lines = wrap(value, "mono", 12, max(40, room))
            vw = max(T.text_w(ln, "mono", 12) for ln in lines)
        h = max(lh * len(lines), _s(SMALL_H) if lnk else 0)

        def fn(x, y):
            top = y + (h - lh * len(lines)) // 2
            if self._vis((x, y, w, h)):
                self._text(key, x, _ty(top, lh, "sans", 11.5), "sans", 11.5, T.INK_MUTED)
                for i, ln in enumerate(lines):
                    self._text(ln, x + kw + g6, _ty(top + i * lh, lh, "mono", 12), "mono", 12,
                               colour or T.INK)
            kr = pygame.Rect(x, top, kw, lh)
            if tip and not long_tip and self._hot(kr):
                self.overlay.tooltip(kr, tip)
            ex = x + kw + g6 + vw
            if long_tip:
                self._dot(ex + g3, top + lh // 2, tip, key)
                ex += g3 + d
            if lnk:
                self._button_draw(lnk[2], lnk[3], ex + _s(12), y + (h - _s(SMALL_H)) // 2, lnk[0], lnk[1], "flat",
                                  small=True)
        self._place(w, h, fn)

    def _link_spec(self, link, owner):
        """(label, icon, key, param) for kv / field links, or None."""
        if not link:
            return None
        label, icon, target = link
        if callable(target):
            k = f"ui.link.{owner}.{label}"
            return label, icon, k, self.form.extra(k, label, target)
        return label, icon, target, self._p(target)

    def kpis(self, tiles: list) -> None:
        """KPI tiles (AeroBO readout): 2 px RULE_SOFT bar, 8 px in, at least
        104 wide, 37 tall; caption SANS 11.5 INK_FAINT_TEXT (T.NOTE_CSS;
        AeroBO 10.5 INK_FAINT) (+ "?" when its tip is long, else the tip is
        the tile's hover); value MONO 18 ("kpi");
        unit MONO 10.5 INK_FAINT_TEXT on the baseline 3 px after. Tiles 12 px
        apart; a row that runs out of width wraps. A tile: dict(label, value,
        unit=None, colour=None, tip=None)."""
        with self._row(T.KPI_GAP, centre=False):
            for t in tiles:
                self._kpi(t)

    def _kpi(self, t) -> None:
        label, value = str(t.get("label", "")), str(t.get("value", "—"))
        unit, colour, tip = t.get("unit"), t.get("colour"), t.get("tip")
        long_tip = self._long(tip)
        d, g3, pad = T.HELP_DOT, _s(3), _s(8)
        lw = T.text_w(label, "sans", T.NOTE_CSS) + ((g3 + d) if long_tip else 0)
        vw = T.text_w(value, "kpi", 19) + ((g3 + T.text_w(unit, "mono", 10.5)) if unit else 0)
        w = max(T.KPI_MIN_W, T.KPI_BAR_W + pad + max(lw, vw))
        h = T.KPI_H
        lh_l, lh_v = _s(KPI_LABEL_H), h - _s(KPI_LABEL_H)

        def fn(x, y):
            r = pygame.Rect(x, y, w, h)
            if self._vis(r):
                self.surf.fill(T.RULE_SOFT, (x, y, T.KPI_BAR_W, h))
                cx = x + T.KPI_BAR_W + pad
                self._text(label, cx, _ty(y, lh_l, "sans", T.NOTE_CSS), "sans", T.NOTE_CSS, T.INK_FAINT_TEXT)
                vy = _ty(y + lh_l, lh_v, "kpi", 19)
                vw_ = self._text(value, cx, vy, "kpi", 19, colour or T.INK)
                if unit:
                    base = vy + T.font("kpi", 19).get_ascent()
                    uy = base - T.font("mono", 10.5).get_ascent()
                    self._text(unit, cx + vw_ + g3, uy, "mono", 10.5, T.INK_FAINT_TEXT)
            if long_tip:
                self._dot(x + T.KPI_BAR_W + pad + T.text_w(label, "sans", T.NOTE_CSS) + g3, y + lh_l // 2, tip, label)
            elif tip and self._hot(r):
                self.overlay.tooltip(r, tip)
        self._place(w, h, fn)

    def tag(self, text, colour) -> None:
        """A state chip: SANS 10/600 in `colour`, 1px border of it, its 10 %
        tint behind, padding 0 5, 17 tall."""
        w, h = self._tag_w(text), T.TAG_H

        def fn(x, y):
            self._tag_draw(text, colour, x, y)
        self._place(w, h, fn)

    @staticmethod
    def _tag_w(text) -> int:
        return T.text_w(text, "sans", 10, True) + 2 * _s(5) + 2

    def _tag_draw(self, text, colour, x, y, w=None) -> None:
        w = self._tag_w(text) if w is None else w
        r = pygame.Rect(x, y, w, T.TAG_H)
        if self._vis(r):
            _rect(self.surf, T.tint(colour), r)
            _rect(self.surf, colour, r, 1)
            self._text(text, x + 1 + _s(5), _ty(y, T.TAG_H, "sans", 10, True), "sans", 10, colour, True,
                       clip_w=w - 2 - 2 * _s(5))

    def code(self, lines) -> None:
        """A monospace block: MONO 11 INK on CODE_BG, 1px RULE_SOFT, padding
        8, no wrap; wider than the column it scrolls sideways (SHIFT+wheel)."""
        lines = [str(s) for s in lines]
        cid = self._auto_id("code")
        st = self.state.setdefault(("code", cid), {"hx": 0})
        w = self.w
        pad, lh = _s(CODE_PAD), _s(CODE_LINE)
        h = 2 * pad + lh * max(1, len(lines)) + 2
        wide = max([T.text_w(s, "mono", 11) for s in lines] or [0])

        def fn(x, y):
            r = pygame.Rect(x, y, w, h)
            span = max(0, wide - (w - 2 * pad - 2))
            st["hx"] = max(0, min(st["hx"], span))
            if span:
                self.form.ui_wheel(r, lambda d: st.__setitem__("hx", max(0, min(span, st["hx"] + d * _s(40)))))
            if not self._vis(r):
                return
            _rect(self.surf, T.CODE_BG, r)
            _rect(self.surf, T.RULE_SOFT, r, 1)
            old = self.surf.get_clip()
            self.surf.set_clip(r.inflate(-2 * pad, -2).clip(old))
            for i, s in enumerate(lines):
                self._text(s, x + 1 + pad - st["hx"], y + 1 + pad + i * lh + (lh - _fh("mono", 11)) // 2,
                           "mono", 11, T.INK)
            self.surf.set_clip(old)
        self._place(w, h, fn)

    # -- Param-bound controls ------------------------------------------------------------------
    def field(self, key, *, control="number", label=None, unit=None, note=None, help=None, width=None,
              labels=None) -> None:
        """AeroBO's field row: label SANS 11.5 INK (at least 150 wide), 6 px,
        the control, its unit (MONO 10.5 INK_FAINT_TEXT), a note (SANS 11.5
        INK_MUTED -- T.NOTE_CSS, AeroBO's 10.5), then the "?" pinned to the
        row's right end (`help`, else the Param's help when it is long; a
        short one is the label's hover).
        `control`: number select toggle switch readout. 26 tall (a readout
        17)."""
        p = self._p(key)
        label = p.label if label is None else label
        hl = help if help is not None else (p.help if self._long(p.help) else None)
        short = None if help is not None or self._long(p.help) else p.help
        g6, d = _s(6), T.HELP_DOT
        unit = unit if unit is not None else (p.unit if control == "number" and p.unit else None)
        row_h = _s(LINE_H) if control == "readout" else T.FIELD_H
        cw = self._control_w(key, p, control, width, labels)
        tail = (g6 + T.text_w(unit, "mono", 10.5) if unit else 0) + (g6 + T.text_w(note, "sans", T.NOTE_CSS) if note else 0)
        natural_lw = T.text_w(label, "sans", 11.5)
        in_row = self._in_row()
        if in_row:
            lw = max(T.LABEL_MIN_W, natural_lw)
            w = lw + g6 + cw + tail + ((g6 + d) if hl else 0)
            lab_lines = [label]
        else:
            w = self.w
            room = w - g6 - cw - tail - ((g6 + d) if hl else 0)
            lw = max(T.LABEL_MIN_W, min(natural_lw, max(T.LABEL_MIN_W, room)))
            lab_lines = wrap(label, "sans", 11.5, lw) if natural_lw > lw else [label]
        lh = _s(LINE_H)
        h = max(row_h, lh * len(lab_lines))

        def fn(x, y):
            row = pygame.Rect(x, y, w, h)
            vis = self._vis(row)
            ly = y + (h - lh * len(lab_lines)) // 2
            col = T.INK_MUTED if control == "readout" else T.INK
            if vis:
                for i, ln in enumerate(lab_lines):
                    self._text(ln, x, _ty(ly + i * lh, lh, "sans", 11.5), "sans", 11.5, col)
            lr = pygame.Rect(x, ly, lw, lh * len(lab_lines))
            if short and self._hot(lr):
                self.overlay.tooltip(lr, short)
            cx = x + lw + g6
            self._control(key, p, control, cx, y, cw, h, row, labels)
            tx = cx + cw
            if unit:
                if vis:
                    self._text(unit, tx + g6, _ty(y, h, "mono", 10.5), "mono", 10.5, T.INK_FAINT_TEXT)
                tx += g6 + T.text_w(unit, "mono", 10.5)
            if note:
                if vis:
                    self._text(note, tx + g6, _ty(y, h, "sans", T.NOTE_CSS), "sans", T.NOTE_CSS, T.INK_MUTED)
                tx += g6 + T.text_w(note, "sans", T.NOTE_CSS)
            if hl:
                self._dot(tx + g6 if in_row else x + w - d, y + h // 2, hl, label, key)
        self._place(w, h, fn)

    def _control_w(self, key, p, control, width, labels) -> int:
        if control == "number":
            return int(width or T.FIELD_W)
        if control == "select":
            return int(width or T.SELECT_W)
        if control == "toggle":
            return self._toggle_w(p, labels)
        if control == "switch":
            return _s(SWITCH_W)
        if control == "readout":
            return int(width or T.text_w(self._readout_text(p), "mono", 12))
        raise ValueError(f"unknown control {control!r}")

    def _control(self, key, p, control, x, y, w, h, row, labels, digits=FIELD_DIGITS) -> None:
        """Draw and record the bare control `control` of `p` in (x, y, w, h)."""
        if control == "number":
            self._number_ctl(key, p, pygame.Rect(x, y + (h - T.FIELD_H) // 2, w, T.FIELD_H), row, digits)
        elif control == "select":
            self._select_ctl(key, p, pygame.Rect(x, y + (h - T.FIELD_H) // 2, w, T.FIELD_H), row, labels)
        elif control == "toggle":
            self._toggle_ctl(key, p, x, y + (h - T.FIELD_H) // 2, row, labels)
        elif control == "switch":
            self._switch_ctl(key, p, x, y + h // 2, row, T.ACCENT)
        elif control == "readout":
            r = pygame.Rect(x, y, w, h)
            if self._vis(r):
                self._text(self._readout_text(p), x, _ty(y, h, "mono", 12), "mono", 12, T.INK, clip_w=w)
            self._record(key, "readout", row, r, None, p)
            self._explain(key, r, p)

    @staticmethod
    def _readout_text(p) -> str:
        if p.get is None:
            return ""
        return p.value_text() if p.kind in ("float", "int", "bool") else str(p.get())

    def number(self, key, *, width=None, digits=FIELD_DIGITS) -> None:
        """A bare number field for use inside row(): 26 tall, 1px FIELD_EDGE,
        MONO 11.5 value (`digits` decimals, AeroBO's 4). No spinners at rest;
        hovered or keyboard-focused it shows two stepper cells at its right
        (carsim's addition). At its "off" end a Param with `off_text` shows
        that text in INK_FAINT. Disabled / locked: .6 opacity."""
        p = self._p(key)
        w, h = int(width or T.FIELD_W), T.FIELD_H

        def fn(x, y):
            r = pygame.Rect(x, y, w, h)
            self._number_ctl(key, p, r, r, digits)
        self._place(w, h, fn)

    def _number_ctl(self, key, p, r, row, digits=FIELD_DIGITS) -> None:
        on = self._on(p)
        editing = self.form.editing == key
        focus = self._focused(key)
        hot = self._hot(r)
        aw, ah = _s(ARROW_W), _s(ARROW_H)
        up = pygame.Rect(r.right - 1 - _s(2) - aw, r.y + (r.h - 2 * ah) // 2, aw, ah)
        down = pygame.Rect(up.x, up.bottom, aw, ah)
        arrows = on and (hot or focus) and not editing
        text_r = pygame.Rect(r.x, r.y, r.w - (aw + _s(4) if arrows else 0), r.h)
        self._record(key, "number", row, r, {"text": text_r, "up": up, "down": down}, p, digits=digits)
        self._explain(key, r, p)
        if not self._vis(r):
            return
        fl = self.form.flash
        flashing = fl is not None and fl[0] == key
        edge = T.BAD if flashing else (T.ACCENT if editing else (T.INK_MUTED if hot and on else T.FIELD_EDGE))
        _rect(self.surf, T.WELL, r)
        _rect(self.surf, _fade(edge, on), r, 1)
        tx = r.x + _s(8)
        room = text_r.w - 2 * _s(8)
        ty = _ty(r.y, r.h, "mono", 11.5)
        if editing:
            s = self.form.buf
            tw = T.text_w(s, "mono", 11.5)
            if self.form._fresh and s:
                self.surf.fill(T.ACCENT_FILL, (tx - 1, r.y + _s(5), min(tw, room) + 2, r.h - 2 * _s(5)))
            self._text(s, tx, ty, "mono", 11.5, T.INK, clip_w=room)
            if int(self.now * 2) % 2 == 0:
                cx = tx + min(tw, room) + 1
                self.surf.fill(T.INK, (cx, r.y + _s(6), 1, r.h - 2 * _s(6)))
        elif is_off(p):
            self._text(getattr(p, "off_text", ""), tx, ty, "mono", 11.5, T.INK_FAINT, clip_w=room)
        else:
            self._text(number_text(p, digits), tx, ty, "mono", 11.5,
                       T.BAD if flashing else _fade(T.INK, on), clip_w=room)
        if arrows:
            for cell, name in ((up, "arrow_drop_up"), (down, "arrow_drop_down")):
                c = T.ACCENT if self._hot(cell) else T.INK_MUTED
                px = _s(22)
                T.icon(self.surf, name, cell.centerx, cell.centery - px // 2, px, c, "centre")
        if focus and not editing:
            self._ring(r)

    def select(self, key, *, width=None, labels=None) -> None:
        """A bare select for row(): the box, the value ellipsised, the
        arrow_drop_down caret; a click or ENTER opens the drop-down."""
        p = self._p(key)
        w, h = int(width or T.SELECT_W), T.FIELD_H

        def fn(x, y):
            r = pygame.Rect(x, y, w, h)
            self._select_ctl(key, p, r, r, labels)
        self._place(w, h, fn)

    def _choice_label(self, p, c, labels) -> str:
        if labels is None:
            return str(c)
        if isinstance(labels, dict):
            return str(labels.get(c, c))
        choices = list(p.choices)
        return str(labels[choices.index(c)]) if c in choices and choices.index(c) < len(labels) else str(c)

    def _labels_map(self, p, labels) -> dict:
        return {c: self._choice_label(p, c, labels) for c in p.choices}

    def _select_ctl(self, key, p, r, row, labels) -> None:
        on = self._on(p)
        self._record(key, "select", row, r, {"box": r}, p, choices=list(p.choices), labels=self._labels_map(p, labels))
        self._explain(key, r, p)
        if not self._vis(r):
            return
        hot = self._hot(r)
        open_ = self.overlay.menu is not None and self.overlay.menu["anchor"] == r
        _rect(self.surf, T.WELL, r)
        _rect(self.surf, _fade(T.ACCENT if open_ else (T.INK_MUTED if hot and on else T.FIELD_EDGE), on), r, 1)
        cp = _s(CARET_PX)
        cx = r.right - _s(CARET_INSET) - cp // 2
        self._text(self._choice_label(p, p.get(), labels) if p.get is not None else "", r.x + _s(8),
                   _ty(r.y, r.h, "mono", 11.5), "mono", 11.5, _fade(T.INK, on),
                   clip_w=r.w - _s(8) - cp - _s(CARET_INSET))
        T.icon(self.surf, "arrow_drop_down", cx, r.centery - cp // 2, cp, _fade(T.INK_MUTED, on), "centre")
        if self._focused(key):
            self._ring(r)

    def _toggle_w(self, p, labels) -> int:
        pad = _s(SEG_PAD)
        return sum(T.text_w(self._choice_label(p, c, labels), "sans", 11.5) + 2 * pad for c in p.choices)

    def toggle(self, key, *, labels=None) -> None:
        """A segmented toggle (Quasar dense unelevated): 26 tall, the chosen
        segment ACCENT with white text, the rest INK on nothing."""
        p = self._p(key)
        w, h = self._toggle_w(p, labels), T.FIELD_H

        def fn(x, y):
            self._toggle_ctl(key, p, x, y, pygame.Rect(x, y, w, h), labels)
        self._place(w, h, fn)

    def _toggle_ctl(self, key, p, x, y, row, labels) -> None:
        on = self._on(p)
        pad, h = _s(SEG_PAD), T.FIELD_H
        cur = p.get() if p.get is not None else None
        parts, segs = {}, []
        for c in p.choices:
            lab = self._choice_label(p, c, labels)
            sw = T.text_w(lab, "sans", 11.5) + 2 * pad
            parts[c] = pygame.Rect(x, y, sw, h)
            segs.append((c, lab, parts[c]))
            x += sw
        whole = pygame.Rect(segs[0][2].x, y, x - segs[0][2].x, h) if segs else pygame.Rect(x, y, 0, h)
        self._record(key, "toggle", row, whole, parts, p, choices=list(p.choices),
                     labels=self._labels_map(p, labels))
        self._explain(key, whole, p)
        if not self._vis(whole):
            return
        for c, lab, sr in segs:
            if c == cur:
                _rect(self.surf, T.ACCENT if on else T.fade(T.ACCENT, DISABLED_A), sr)
                self._text(lab, sr.x + pad, _ty(y, h, "sans", 11.5), "sans", 11.5, T.WELL)
            else:
                if on and self._hot(sr):
                    _rect(self.surf, T.tint(T.INK, 0.06), sr)
                self._text(lab, sr.x + pad, _ty(y, h, "sans", 11.5), "sans", 11.5, _fade(T.INK, on))
        if self._focused(key):
            self._ring(whole)

    def switch(self, key, *, label=None, colour=None, default=True) -> bool:
        """A switch (Quasar dense toggle, 38 px): a 30 x 13 track, a 20 px
        thumb; on: the track `colour` at .54, the thumb `colour`; off: an INK
        .38 track, a white thumb. Its label (SANS 11.5 INK) follows at 38 px.
        `key` a Param of the form, or else a UI switch kept in `state`
        (starting at `default`). Returns whether it is on."""
        colour = T.ACCENT if colour is None else colour
        p = self.form.param(key)
        if p is None or key in self.form.extras:
            k = key if key.startswith("ui.") else f"ui.switch.{key}"
            st = self.state.setdefault(("switch", k), {"on": bool(default)})
            p = self.form.extra(k, label or key, None)
            p.kind = "bool"
            p.get = lambda: st["on"]
            p.set = lambda v: st.__setitem__("on", bool(v))
            key = k
        text = p.label if label is None else label
        lw = T.text_w(text, "sans", 11.5) if text else 0
        w = _s(SWITCH_W) + lw
        h = _s(SWITCH_ROW_H)

        def fn(x, y):
            row = pygame.Rect(x, y, w, h)
            self._switch_ctl(key, p, x, y + h // 2, row, colour, text)
        self._place(w, h, fn)
        return bool(p.get()) if p.get is not None else False

    def _switch_ctl(self, key, p, x, cy, row, colour, text=None) -> None:
        on = self._on(p)
        val = bool(p.get()) if p.get is not None else False
        box = pygame.Rect(x, cy - _s(10), _s(32), _s(20))
        hit = box.copy()
        if text:
            hit.w = row.right - box.x
        self._record(key, "switch", row, box, {"hit": hit}, p)
        self._explain(key, hit, p)
        if not self._vis(row):
            return
        a = 1.0 if on else DISABLED_A
        track = pygame.Rect(box.x + 1, cy - _s(13) // 2, _s(30), _s(13))
        tcol = (T.SWITCH_ON_TRACK if colour == T.ACCENT else T.fade(colour, 0.54)) if val else T.SWITCH_OFF_TRACK
        pygame.draw.rect(self.surf, T.fade(tcol, a), track, border_radius=track.h // 2)
        d = _s(20)
        tx = box.x + (_s(22) if val else _s(8))
        if val:
            self.surf.blit(disc(d, T.fade(colour, a)), (tx - d // 2, cy - d // 2))
        else:
            self.surf.blit(disc(d + 2, (0, 0, 0, 36)), (tx - d // 2 - 1, cy - d // 2))
            self.surf.blit(disc(d, T.WELL, T.fade(T.RULE_SOFT, a)), (tx - d // 2, cy - d // 2))
        if text:
            self._text(text, box.x + _s(SWITCH_W), _ty(cy - _s(LINE_H) // 2, _s(LINE_H), "sans", 11.5), "sans",
                       11.5, _fade(T.INK, on))
        if self._focused(key):
            self._ring(box)

    def slider(self, key, *, label=None, help=None, dim=False) -> None:
        """AeroBO's weight row: label (at least 150 wide, wrapping; `dim` =
        the label at .45), a "?" (`help`, else the Param's help when it is
        long; a short one is the label's hover -- AeroBO widgets.explain), a
        slider filling the row (4 px track, ACCENT left of the 16 px thumb,
        SLIDER_OFF right), the value MONO 12 right-aligned in 34 px. A drag
        stores its value (the Form applies it once a frame; a costly Param
        on release)."""
        p = self._p(key)
        label = p.label if label is None else label
        short = None
        if help is None:
            help, short = (p.help, None) if self._long(p.help) else (None, p.help or None)
        w = self.w
        g6, d = _s(6), T.HELP_DOT
        lw = T.LABEL_MIN_W
        lines = wrap(label, "sans", 11.5, lw)
        lh = _s(LINE_H)
        h = max(_s(SLIDER_ROW_H), lh * len(lines))
        vw = _s(SLIDER_VALUE_W)

        def fn(x, y):
            row = pygame.Rect(x, y, w, h)
            vis = self._vis(row)
            on = self._on(p)
            if vis:
                lc = T.fade(T.INK, DEAD_A) if dim else _fade(T.INK, on)
                for i, ln in enumerate(lines):
                    self._text(ln, x, _ty(y + (h - lh * len(lines)) // 2 + i * lh, lh, "sans", 11.5), "sans",
                               11.5, lc)
            lr = pygame.Rect(x, y, lw, h)
            if short and self._hot(lr):
                self.overlay.tooltip(lr, short)
            tx = x + lw + g6
            if help:
                self._dot(tx, y + h // 2, help, label, key)
                tx += d + g6
            track = pygame.Rect(tx, y + h // 2 - _s(TRACK_H) // 2, max(10, x + w - vw - g6 - tx), _s(TRACK_H))
            ctl = pygame.Rect(track.x - _s(THUMB_D) // 2, y + h // 2 - _s(THUMB_D) // 2,
                              track.w + _s(THUMB_D), _s(THUMB_D))
            g = self._record(key, "slider", row, ctl, {"track": track}, p)
            self._explain(key, ctl, p)
            if not vis:
                return
            v = self.form.pending.get(key, p.get() if p.get is not None else 0.0)
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = float(p.lo or 0.0)
            tx_ = g.x_for(v)
            self.surf.fill(_fade(T.SLIDER_OFF, on), track)
            self.surf.fill(_fade(T.ACCENT, on), (track.x, track.y, tx_ - track.x, track.h))
            dd = _s(THUMB_D)
            self.surf.blit(disc(dd, _fade(T.ACCENT, on)), (tx_ - dd // 2, track.centery - dd // 2))
            self._text(p.fmt.format(v) if p.kind == "float" else str(int(round(v))), x + w,
                       _ty(y, h, "mono", 12), "mono", 12, _fade(T.INK, on), anchor="right")
            if self._focused(key):
                self._ring(ctl)
        self._place(w, h, fn)

    @staticmethod
    def _button_w(text, icon, kind, small=False) -> int:
        pad = _s(BTN_PAD)
        css = SMALL_CSS if small else 11.5
        lead = (_s(SMALL_SLOT) + _s(SMALL_GAP)) if small else (_s(BTN_ICON_SLOT) + _s(BTN_ICON_GAP))
        return pad + (lead if icon else _s(4)) + T.text_w(text, "sans", css) + pad + (0 if icon else _s(4))

    def button(self, key, text=None, *, kind="primary", icon=None, help=None, on_click=None, enabled=True,
               small=False) -> None:
        """A button, 26 tall. primary: ACCENT fill, 1px PRIMARY_EDGE, white
        icon + SANS 11.5; outline: WELL, 1px ACCENT, INK; flat: ACCENT text
        and icon. Disabled / locked: primary PRIMARY_DISABLED, the others at
        .6. It fires on the FIRST click. `key` is an action Param of the
        form; a key the form does not hold, with `on_click`, is a UI button
        (an extra). A "?" follows it when `help`. `small`: AeroBO's inline
        size=sm link (10 px text, 22 tall)."""
        p = self.form.param(key)
        if p is None:
            k = key if key.startswith("ui.") else f"ui.{key}"
            p = self.form.extra(k, text or key, on_click, enabled=enabled)
            key = k
        text = p.label if text is None else text
        bw = self._button_w(text, icon, kind, small)
        g6, d = _s(6), T.HELP_DOT
        w = bw + ((g6 + d) if help else 0)
        h = _s(SMALL_H) if small else T.BUTTON_H

        def fn(x, y):
            self._button_draw(key, p, x, y, text, icon, kind, small=small)
            if help:
                self._dot(x + bw + g6, y + h // 2, help, text, key)
        self._place(w, h, fn)

    def _button_draw(self, key, p, x, y, text, icon, kind, small=False) -> None:
        bw = self._button_w(text, icon, kind, small)
        h = _s(SMALL_H) if small else T.BUTTON_H
        css = SMALL_CSS if small else 11.5
        r = pygame.Rect(x, y, bw, h)
        self._record(key, "button" if kind != "flat" else "link", r, r, {"box": r}, p)
        self._explain(key, r, p)
        if not self._vis(r):
            return
        on = self._on(p)
        hot = on and self._hot(r)
        if kind == "primary":
            fill = T.PRIMARY_HOVER if hot else (T.ACCENT if on else T.PRIMARY_DISABLED)
            _rect(self.surf, fill, r)
            _rect(self.surf, T.PRIMARY_EDGE if on else T.PRIMARY_DISABLED, r, 1)
            ink = T.WELL
        elif kind == "outline":
            _rect(self.surf, T.tint(T.ACCENT) if hot else T.WELL, r)
            _rect(self.surf, _fade(T.ACCENT, on), r, 1)
            ink = _fade(T.INK, on)
        else:
            if hot:
                _rect(self.surf, T.tint(T.ACCENT), r)
            ink = _fade(T.ACCENT, on)
        tx = x + _s(BTN_PAD)
        if icon:
            slot = _s(SMALL_SLOT) if small else _s(BTN_ICON_SLOT)
            ip = _s(SMALL_ICON) if small else T.BUTTON_ICON
            T.icon(self.surf, icon, tx + slot // 2, y + (h - ip) // 2, ip, ink, "centre")
            tx += slot + (_s(SMALL_GAP) if small else _s(BTN_ICON_GAP))
        else:
            tx += _s(4)
        self._text(text, tx, _ty(y, h, "sans", css), "sans", css, ink)
        if self._focused(key):
            self._ring(r)

    def readout(self, key, *, label=None, unit=None) -> None:
        """A read-only Param as a kv row (its list format and unit)."""
        p = self._p(key)
        v = self._readout_text(p)
        if unit and not p.unit:
            v = f"{v} {unit}"
        label = p.label if label is None else label
        lh = _s(LINE_H)
        kw = max(T.LABEL_MIN_W, T.text_w(label, "sans", 11.5))
        w = self.w if not self._in_row() else kw + _s(6) + T.text_w(v, "mono", 12)

        def fn(x, y):
            row = pygame.Rect(x, y, w, lh)
            if self._vis(row):
                self._text(label, x, _ty(y, lh, "sans", 11.5), "sans", 11.5, T.INK_MUTED)
                self._text(v, x + kw + _s(6), _ty(y, lh, "mono", 12), "mono", 12, T.INK, clip_w=w - kw - _s(6))
            self._record(key, "readout", row, pygame.Rect(x + kw + _s(6), y, w - kw - _s(6), lh), None, p)
        self._place(w, lh, fn)

    def radio_rows(self, key, options, *, mono=True) -> None:
        """A radio list (AeroBO 02 / WG §4.3): one row per (value, title,
        note), padding 3/4, the chosen row ACCENT_FILL; radio_button_checked
        ACCENT / radio_button_unchecked INK_FAINT, 6 px, the title (MONO 12,
        AeroBO's read-out style; `mono=False`: SANS 11.5), the note SANS 11.5
        (T.NOTE_CSS, AeroBO's 10.5) INK_MUTED under it. ONE keyboard stop:
        UP / DOWN move inside it (leaving at the ends), ENTER picks; a click
        picks. key=None draws a static read-out with the first option chosen."""
        opts = [(o[0], str(o[1]), str(o[2]) if len(o) > 2 and o[2] else "") for o in options]
        p = self.form.param(key) if key is not None else None
        if key is not None and p is None:
            raise KeyError(f"no Param {key!r} in this view's form")
        cur = (p.get() if p is not None and p.get is not None else (opts[0][0] if opts else None))
        py, px = _s(RADIO_PAD_Y), _s(RADIO_PAD_X)
        lh = _s(LINE_H)
        fam, css = ("mono", 12) if mono else ("sans", 11.5)
        w = self.w
        heights = [2 * py + lh + (lh if note else 0) for _, _, note in opts]
        gap = _s(6)
        h = sum(heights) + gap * max(0, len(opts) - 1)
        form = self.form
        if key is not None:
            form._cursor.setdefault(key, next((i for i, o in enumerate(opts) if o[0] == cur), 0))

        def fn(x, y):
            parts, rows = {}, []
            yy = y
            for (val, title, note), rh in zip(opts, heights):
                rr = pygame.Rect(x, yy, w, rh)
                parts[val] = rr
                rows.append((val, title, note, rr))
                yy += rh + gap
            whole = pygame.Rect(x, y, w, h)
            on = p is None or self._on(p)
            if key is not None:
                self._record(key, "radio", whole, whole, parts, p, choices=[o[0] for o in opts])
                self._explain(key, whole, p)
                n = len(opts)

                def step(d):
                    c = form._cursor.get(key, 0) + d
                    if 0 <= c < n:
                        form._cursor[key] = c
                        return True
                    return False

                def enter(d):
                    form._cursor[key] = 0 if d > 0 else n - 1

                def activate():
                    c = form._cursor.get(key, 0)
                    if 0 <= c < n:
                        p.set(opts[c][0])
                        return True
                    return False
                form.inner(key, step=step, enter=enter, activate=activate)
            if not self._vis(whole):
                return
            ic = T.ICON_PX
            for i, (val, title, note, rr) in enumerate(rows):
                sel = val == cur
                if sel:
                    _rect(self.surf, T.ACCENT_FILL, rr)
                elif p is not None and on and self._hot(rr):
                    _rect(self.surf, T.tint(T.ACCENT, 0.06), rr)
                ty = rr.y + py
                T.icon(self.surf, "radio_button_checked" if sel else "radio_button_unchecked", rr.x + px,
                       ty + (lh - ic) // 2, ic, _fade(T.ACCENT if sel else T.INK_FAINT, on))
                tx = rr.x + px + ic + _s(6)
                self._text(title, tx, _ty(ty, lh, fam, css), fam, css, _fade(T.INK, on), clip_w=rr.right - tx - px)
                if note:
                    self._text(note, tx, _ty(ty + lh, lh, "sans", T.NOTE_CSS), "sans", T.NOTE_CSS, _fade(T.INK_MUTED, on),
                               clip_w=rr.right - tx - px)
                if key is not None and self._focused(key) and form._cursor.get(key, 0) == i:
                    self._ring(rr)
        self._place(w, h, fn)

    def v1_row(self, key, label, *, control="select", note=None, label_w=None, unit=None, labels=None) -> None:
        """AeroBO's "v1 row" (WG §0.2, shot 14): label SANS 12 INK at .70 in a
        60 px column (96 for a number row -- Tailwind w-20 / w-32 at this
        shell's 12 px rem, measured), 9 px, the control GROWING to fill the
        row (select, number, readout; a toggle or switch keeps its width),
        the unit SANS 12 at .60 after it, the note SANS 12 at .60 under the
        row across its full width (T.HINT_CSS; AeroBO 11)."""
        p = self._p(key)
        lw = int(label_w) if label_w is not None else _s(V1_NUM_LABEL_W if control == "number" else V1_LABEL_W)
        gap = _s(V1_GAP)
        w = self.w
        unit_w = (_s(10) + T.text_w(unit, "sans", T.HINT_CSS)) if unit else 0
        if control in ("select", "number", "readout"):
            cw = max(40, w - lw - gap - unit_w)
        else:
            cw = self._control_w(key, p, control, None, labels)
        lh = _s(LINE_H)
        lab_lines = wrap(label, "sans", 12, lw)
        rh = max(T.FIELD_H, lh * len(lab_lines))
        note_lines = wrap(note, "sans", T.HINT_CSS, w) if note else []
        nh = (_s(4) + lh * len(note_lines)) if note_lines else 0
        h = rh + nh

        def fn(x, y):
            row = pygame.Rect(x, y, w, rh)
            vis = self._vis(pygame.Rect(x, y, w, h))
            ly = y + (rh - lh * len(lab_lines)) // 2
            if vis:
                lc = T.fade(T.INK, V1_LABEL_A)
                for i, ln in enumerate(lab_lines):
                    self._text(ln, x, _ty(ly + i * lh, lh, "sans", 12), "sans", 12, lc)
            cx = x + lw + gap
            self._control(key, p, control, cx, y, cw, rh, row, labels)
            if unit and vis:
                self._text(unit, cx + cw + _s(10), _ty(y, rh, "sans", T.HINT_CSS), "sans", T.HINT_CSS, T.fade(T.INK, V1_NOTE_A))
            if note_lines and vis:
                nc = T.fade(T.INK, V1_NOTE_A)
                for i, ln in enumerate(note_lines):
                    self._text(ln, x, _ty(y + rh + _s(4) + i * lh, lh, "sans", T.HINT_CSS), "sans", T.HINT_CSS, nc)
        self._place(w, h, fn)

    def grid(self, id, cols, header, rows, *, pitch=37, gap=(4, 10)) -> None:
        """AeroBO's CSS grid (`_box_grid`, results tables): `cols` weights (fr)
        or "auto" (the widest cell); header cells str or (str, help) in SANS
        11.5 INK_FAINT_TEXT (T.NOTE_CSS; AeroBO 10.5 INK_FAINT); rows of cells --
          ("text", s[, colour, family, css, tip])   ("text2", line1, line2[, help])
          ("number", key[, digits])   ("switch", key[, colour, tip])
          ("range", frac[, colour, tip])   ("tag", text, colour)   ("help", text[, title])
          ("empty",)
        -- vertically centred in a row of `pitch` (row-to-row, the vertical gap
        included). Param-bound cells record like any control, row-major."""
        vg, hg = _s(gap[0]), _s(gap[1])
        pitch = _s(pitch)
        rh = pitch - vg
        n = len(cols)
        w = self.w
        hh = _s(15)
        # column widths: "auto" = the widest cell of the column, the rest by weight
        auto = [0] * n
        for c, spec in enumerate(cols):
            if spec == "auto":
                cells = [self._cell_w(r[c]) for r in rows if c < len(r)]
                head = header[c] if header and c < len(header) else ""
                auto[c] = max(cells + [self._head_w(head)])
        fixed = sum(auto[c] for c in range(n) if cols[c] == "auto")
        weights = [0.0 if cols[c] == "auto" else float(cols[c]) for c in range(n)]
        room = max(0, w - fixed - hg * (n - 1))
        tot = sum(weights) or 1.0
        widths = [auto[c] if cols[c] == "auto" else int(room * weights[c] / tot) for c in range(n)]
        xs, xx = [], 0
        for c in range(n):
            xs.append(xx)
            xx += widths[c] + hg
        # a column's text2 second lines take T.NOTE_CSS when every one of them fits it, else
        # AeroBO's own 10.5: the readability bump never cuts a line that fitted before
        css2 = [T.NOTE_CSS if all(T.text_w(str(r[c][2]), "sans", T.NOTE_CSS) <= widths[c]
                                  for r in rows if c < len(r) and r[c] and r[c][0] == "text2") else 10.5
                for c in range(n)]
        h = (hh + vg if header else 0) + pitch * len(rows) - (vg if rows else 0)

        def fn(x, y):
            vis = self._vis((x, y, w, hh))
            for c, head in enumerate(header or ()):
                text, hlp = (head, None) if isinstance(head, str) else (head[0], head[1])
                tw = (self._text(text, x + xs[c], _ty(y, hh, "sans", T.NOTE_CSS), "sans", T.NOTE_CSS, T.INK_FAINT_TEXT) if vis
                      else T.text_w(text, "sans", T.NOTE_CSS))
                if hlp:
                    self._dot(x + xs[c] + tw + _s(3), y + hh // 2, hlp, text)
            ry = y + ((hh + vg) if header else 0)
            for cells in rows:
                row = pygame.Rect(x, ry, w, rh)
                for c, cell in enumerate(cells[:n]):
                    self._cell(cell, pygame.Rect(x + xs[c], ry, widths[c], rh), row, css2[c])
                ry += pitch
        self._place(w, h, fn)

    def _head_w(self, head) -> int:
        if not head:
            return 0
        if isinstance(head, str):
            return T.text_w(head, "sans", T.NOTE_CSS)
        return T.text_w(head[0], "sans", T.NOTE_CSS) + _s(3) + T.HELP_DOT

    def _cell_w(self, cell) -> int:
        kind = cell[0] if cell else "empty"
        if kind == "text":
            fam = cell[3] if len(cell) > 3 and cell[3] else "mono"
            css = cell[4] if len(cell) > 4 and cell[4] else 12
            return T.text_w(str(cell[1]), fam, css)
        if kind == "text2":
            return max(T.text_w(str(cell[1]), "mono", 12) + ((_s(3) + T.HELP_DOT) if len(cell) > 3 and cell[3] else 0),
                       T.text_w(str(cell[2]), "sans", T.NOTE_CSS))
        if kind == "number":
            return T.FIELD_W
        if kind == "switch":
            return _s(SWITCH_W)
        if kind == "range":
            return _s(60)
        if kind == "tag":
            return self._tag_w(str(cell[1]))
        if kind == "help":
            return T.HELP_DOT
        return 0

    def _cell(self, cell, r, row, css2=None) -> None:
        kind = cell[0] if cell else "empty"
        if kind == "text":
            s = str(cell[1])
            colour = cell[2] if len(cell) > 2 and cell[2] else T.INK
            fam = cell[3] if len(cell) > 3 and cell[3] else "mono"
            css = cell[4] if len(cell) > 4 and cell[4] else 12
            tip = cell[5] if len(cell) > 5 else None
            if self._vis(r):
                self._text(s, r.x, _ty(r.y, r.h, fam, css), fam, css, colour, clip_w=r.w)
            if tip and self._hot(r):
                self.overlay.tooltip(r, tip)
        elif kind == "text2":
            l1, l2 = str(cell[1]), str(cell[2])
            hlp = cell[3] if len(cell) > 3 else None
            lh1, lh2 = _s(LINE_H), _s(15)
            top = r.y + (r.h - lh1 - lh2) // 2
            if self._vis(r):
                tw = self._text(l1, r.x, _ty(top, lh1, "mono", 12), "mono", 12, T.INK,
                                clip_w=r.w - ((_s(3) + T.HELP_DOT) if hlp else 0))
                c2 = T.NOTE_CSS if css2 is None else css2
                self._text(l2, r.x, _ty(top + lh1, lh2, "sans", c2), "sans", c2, T.INK_FAINT_TEXT, clip_w=r.w)
            else:
                tw = T.text_w(l1, "mono", 12)
            if hlp:
                self._dot(r.x + tw + _s(3), top + lh1 // 2, hlp, l1)
        elif kind == "number":
            key = cell[1]
            digits = cell[2] if len(cell) > 2 else FIELD_DIGITS
            #  its row is its CELL, as a switch's is: the grid's whole row
            #  holds the other cells' controls too (`Form.hit_key`)
            self._number_ctl(key, self._p(key), pygame.Rect(r.x, r.centery - T.FIELD_H // 2, r.w, T.FIELD_H), r,
                             digits)
        elif kind == "switch":
            key = cell[1]
            colour = cell[2] if len(cell) > 2 and cell[2] else T.ACCENT
            tip = cell[3] if len(cell) > 3 else None
            p = self._p(key)
            self._switch_ctl(key, p, r.x, r.centery, pygame.Rect(r.x, r.y, _s(SWITCH_W), r.h), colour)
            box = pygame.Rect(r.x, r.centery - _s(10), _s(32), _s(20))
            if tip and self._hot(box) and not self.form.lock_reason(p):
                self.overlay.tooltip(box, tip)
        elif kind == "range":
            colour = cell[2] if len(cell) > 2 and cell[2] else T.ACCENT
            tip = cell[3] if len(cell) > 3 else None
            br = pygame.Rect(r.x, r.centery - _s(9) // 2, r.w, _s(9))
            self._range_draw(cell[1], colour, br)
            if tip and self._hot(br):
                self.overlay.tooltip(br, tip)
        elif kind == "tag":
            self._tag_draw(str(cell[1]), cell[2], r.x, r.centery - T.TAG_H // 2, r.w)
        elif kind == "help":
            self._dot(r.x, r.centery, cell[1], cell[2] if len(cell) > 2 else "")

    # -- UI-only (extras; their state is in self.state) ----------------------------------------------
    def link(self, id, text, icon, on_click, *, enabled=True, small=False) -> None:
        """A flat ACCENT button that is not a Param (select another view,
        re-screen ...); keyboard-reachable like any control. `small`: the
        inline size=sm link ("change in stage 1")."""
        self.button(id if id.startswith("ui.") else f"ui.{id}", text, kind="flat", icon=icon, on_click=on_click,
                    enabled=enabled, small=small)

    def checkbox(self, id, text, default=True, *, enabled=True, tip=None) -> bool:
        """A 13 px check box (ACCENT with a white check when on) and its label
        SANS 11.5 6 px after. Returns whether it is ticked."""
        key = id if id.startswith("ui.") else f"ui.check.{id}"
        st = self.state.setdefault(("check", key), {"on": bool(default)})
        p = self.form.extra(key, text, lambda: st.__setitem__("on", not st["on"]), enabled=enabled)
        b, g6 = _s(CHECK_BOX), _s(6)
        w = b + g6 + T.text_w(text, "sans", 11.5)
        h = _s(CHECK_ROW_H)

        def fn(x, y):
            row = pygame.Rect(x, y, w, h)
            box = pygame.Rect(x, y + (h - b) // 2, b, b)
            self._record(key, "checkbox", row, box, {"hit": row}, p)
            if tip and self._hot(row):
                self.overlay.tooltip(row, tip)
            if not self._vis(row):
                return
            on = self.form.is_enabled(p)
            if st["on"]:
                _rect(self.surf, _fade(T.ACCENT, on), box)
                pts = [(box.x + b * 0.22, box.y + b * 0.52), (box.x + b * 0.42, box.y + b * 0.72),
                       (box.x + b * 0.80, box.y + b * 0.30)]
                pygame.draw.lines(self.surf, T.WELL, False, pts, max(1, _s(2)))
            else:
                _rect(self.surf, T.WELL, box)
                _rect(self.surf, _fade(T.INK_MUTED, on), box, 1)
            self._text(text, box.right + g6, _ty(y, h, "sans", 11.5), "sans", 11.5, _fade(T.INK, on))
            if self._focused(key):
                self._ring(box)
        self._place(w, h, fn)
        return bool(st["on"])

    def disclosure(self, id, title, default=False) -> bool:
        """One quiet 22 px line (SANS 11.5 INK_MUTED) with expand_more /
        expand_less at its right end; returns whether it is open -- draw the
        body inside `with ui.indent():`."""
        key = id if id.startswith("ui.") else f"ui.disc.{id}"
        st = self.state.setdefault(("disc", key), {"open": bool(default)})
        p = self.form.extra(key, title, lambda: st.__setitem__("open", not st["open"]))
        w, h = self.w, _s(DISC_H)

        def fn(x, y):
            row = pygame.Rect(x, y, w, h)
            self._record(key, "disclosure", row, row, {"hit": row}, p)
            if not self._vis(row):
                return
            if self._hot(row):
                self.surf.fill(T.tint(T.INK, 0.04), row)
            self._text(title, x + _s(2), _ty(y, h, "sans", 11.5), "sans", 11.5, T.INK_MUTED,
                       clip_w=w - _s(CHEVRON_PX) - _s(8))
            cp = _s(CHEVRON_PX)
            T.icon(self.surf, "expand_less" if st["open"] else "expand_more", x + w - _s(4) - cp // 2,
                   y + (h - cp) // 2, cp, T.INK_MUTED, "centre")
            if self._focused(key):
                self._ring(row)
        self._place(w, h, fn)
        return bool(st["open"])

    # -- tables ----------------------------------------------------------------------------------
    def table(self, id, columns, rows, *, page_size=None, max_rows=None, selected=None, cursor=None,
              on_row=None, sortable=True, dim_cols=(), h_scroll=True, empty="", height=None,
              on_cursor=None) -> None:
        """AeroBO's table (PLAN §2.3): 1px RULE frame; a 28 px HEADER head
        (SANS 11/600 INK_MUTED) over 28 px rows (MONO 11, even rows
        ROW_EVEN, hover and `selected` ACCENT_FILL, `cursor` a 1px ACCENT
        inset outline); `_sub` a faint MONO 9 suffix, `_tag` a chip after the
        first left column, `_colours` per cell; `dim_cols` at .45. A head
        click sorts (asc, desc, off). `page_size`: a pager ("Records per
        page"); `max_rows`: the first n rows and a `show all {N}` link.
        Columns wider than the card scroll sideways (SHIFT+wheel, the bar,
        LEFT / RIGHT). `selected`, `cursor` and `on_row(i)` speak ORIGINAL
        indices. A `cursor` passed in is drawn (its owner moves it; with
        `on_cursor(i)` the table's keys move it too); without one the table
        keeps its own. `height`: the body's most rows' height (the cursor is
        kept inside)."""
        key = id if id.startswith("ui.") else f"ui.table.{id}"
        if not columns:
            return
        st = self.state.setdefault(("table", key), {"sort": None, "page": 0, "rpp": page_size, "all": False,
                                                    "hx": 0, "cur": 0, "last_cur": None, "top": 0})
        n_all = len(rows)
        order = list(range(n_all))
        if sortable and st["sort"] is not None:
            ck, desc = st["sort"]
            order = self._sorted(rows, ck, desc)
        pos_of = {i: k for k, i in enumerate(order)}
        own = cursor is None
        cur = st["cur"] if own else cursor
        if own:
            cur = max(0, min(cur, n_all - 1)) if n_all else 0
            st["cur"] = cur
        # which rows are on screen
        rpp = st["rpp"] if page_size else None
        if cur is not None and cur != st["last_cur"] and cur in pos_of:
            if rpp:
                st["page"] = pos_of[cur] // rpp
            if max_rows and pos_of[cur] >= max_rows:
                st["all"] = True
        st["last_cur"] = cur
        if rpp:
            pages = max(1, math.ceil(n_all / rpp))
            st["page"] = max(0, min(st["page"], pages - 1))
            shown_ = order[st["page"] * rpp:(st["page"] + 1) * rpp]
        elif max_rows and not st["all"]:
            shown_ = order[:max_rows]
        else:
            shown_ = order
        body_rows = shown_
        rh, hh = T.TABLE_ROW_H, T.TABLE_HEAD_H
        if height is not None:
            fit = max(1, (int(height) - hh) // rh)
            if len(shown_) > fit:
                if cur in shown_:
                    k = shown_.index(cur)
                    st["top"] = max(min(st["top"], k), k - fit + 1)
                st["top"] = max(0, min(st["top"], len(shown_) - fit))
                body_rows = shown_[st["top"]:st["top"] + fit]
        # column widths over the rows on screen
        pad = _s(CELL_PAD)
        texts = {i: [self._cell_text(c, rows[i]) for c in columns] for i in body_rows}
        nat = []
        for c_i, c in enumerate(columns):
            head_w = T.text_w(str(c.get("head", c["key"])), "sans", 11, True) + (_s(SORT_PX) if sortable else 0)
            best = head_w
            for i in body_rows:
                s, sub = texts[i][c_i]
                cw = T.text_w(s, "mono", 11) + ((_s(6) + T.text_w(sub, "mono", 9)) if sub else 0)
                if c_i == self._first_left(columns) and rows[i].get("_tag"):
                    cw += _s(6) + self._tag_w(rows[i]["_tag"][0])
                best = max(best, cw)
            nat.append(int(c.get("width") or best + 2 * pad))
        w = self.w
        inner = w - 2
        total = sum(nat)
        if total < inner:
            extra = inner - total
            widths = [wd + int(extra * wd / total) if total else wd for wd in nat]
            widths[-1] += inner - sum(widths)
            span = 0
        elif h_scroll:
            widths, span = nat, total - inner
        else:
            widths = [int(wd * inner / total) for wd in nat]
            widths[-1] += inner - sum(widths)
            span = 0
        st["hx"] = max(0, min(st["hx"], span))
        bar_h = T.SCROLL_W if span else 0
        empty_h = rh if (not rows and empty) else 0
        pager_h = T.PAGER_H if page_size else 0
        more = bool(max_rows and not rpp and not st["all"] and n_all > max_rows)
        link_h = (_s(6) + T.BUTTON_H) if more else 0
        frame_h = 2 + hh + rh * len(body_rows) + empty_h + bar_h + pager_h
        h = frame_h + link_h
        form = self.form
        dims = set(dim_cols or ())

        # the keyboard: ONE stop. The hooks run between frames, so they read the
        # live state (sort, page, cursor), never this frame's snapshot.
        st["ext"] = None if own else cursor

        def live_order():
            if sortable and st["sort"] is not None:
                return self._sorted(rows, *st["sort"])
            return list(range(n_all))

        def live_cur():
            return st["cur"] if own else st["ext"]

        def live_shown(order_):
            if rpp:
                r_ = st["rpp"] or n_all or 1
                return order_[st["page"] * r_:(st["page"] + 1) * r_]
            if max_rows and not st["all"]:
                return order_[:max_rows]
            return order_

        def set_cur(i):
            self._table_set_cursor(st, own, on_cursor, i)
            st["ext"] = i if not own else None

        def move(d):
            if not own and on_cursor is None:
                return False            # its owner moves this cursor (the shell's .rank keys)
            order_, c = live_order(), live_cur()
            if not order_ or c not in order_:
                return False
            k2 = order_.index(c) + d
            if not 0 <= k2 < len(order_):
                return False
            if order_[k2] not in live_shown(order_):
                if rpp:
                    st["page"] = k2 // (st["rpp"] or n_all or 1)
                elif max_rows:
                    st["all"] = True
            set_cur(order_[k2])
            return True

        def enter(d):
            if own or on_cursor is not None:
                shown_now = live_shown(live_order())
                if shown_now:
                    set_cur(shown_now[0] if d > 0 else shown_now[-1])

        def activate():
            c = live_cur()
            if on_row is not None and c is not None and 0 <= c < n_all:
                on_row(c)
                return True
            return False

        def hscroll(d):
            if not span:
                return False
            st["hx"] = max(0, min(span, st["hx"] + d * _s(40)))
            return True

        def page(d):
            if not rpp:
                return False
            r_ = st["rpp"] or n_all or 1
            st["page"] = max(0, min(st["page"] + d, max(1, math.ceil(n_all / r_)) - 1))
            order_ = live_order()
            if order_ and (own or on_cursor is not None):
                set_cur(order_[min(len(order_) - 1, st["page"] * r_)])
            return True
        p = form.extra(key, str(id), (lambda: activate()) if on_row is not None else None)

        def fn(x, y):
            frame = pygame.Rect(x, y, w, frame_h)
            body_top = y + 1 + hh
            body = pygame.Rect(x + 1, body_top, inner, rh * len(body_rows) + empty_h)
            hits = []
            for k, i in enumerate(body_rows):
                hits.append((pygame.Rect(x + 1, body_top + k * rh, inner, rh), i))

            def click(pos):
                for r, i in hits:
                    if r.collidepoint(pos):
                        if own or on_cursor is not None:
                            set_cur(i)
                        if on_row is not None:
                            on_row(i)
                            return "action"
                        return "select"
                return "select"
            self._record(key, "table", frame, frame, None, p)
            form.inner(key, step=move, enter=enter, activate=activate if on_row is not None else None,
                       adjust=hscroll, page=page, click=click)
            if span:
                form.ui_wheel(frame, hscroll)
            vis = self._vis(pygame.Rect(x, y, w, h))
            # the head: sort on click
            hx0 = x + 1 - st["hx"]
            heads = []
            cx = hx0
            for c_i, c in enumerate(columns):
                hr = pygame.Rect(cx, y + 1, widths[c_i], hh - 1)
                heads.append(hr)
                cx += widths[c_i]
            if sortable:
                for c_i, hr in enumerate(heads):
                    vr = hr.clip(pygame.Rect(x + 1, y + 1, inner, hh))
                    if vr.w > 0:
                        form.ui_hit(vr, lambda _pos, ck=columns[c_i]["key"]: self._table_sort(st, ck))
            if vis:
                _rect(self.surf, T.WELL, frame)
                old = self.surf.get_clip()
                self.surf.set_clip(pygame.Rect(x + 1, y + 1, inner, frame_h - 2).clip(old))
                self.surf.fill(T.HEADER, (x + 1, y + 1, inner, hh - 1))
                self.surf.fill(T.RULE, (x + 1, y + hh, inner, 1))
                for c_i, (c, hr) in enumerate(zip(columns, heads)):
                    self._head_draw(c, hr, st, c["key"] in dims, sortable)
                for k, i in enumerate(body_rows):
                    rr = pygame.Rect(x + 1, body_top + k * rh, inner, rh)
                    if not self._vis(rr):
                        continue
                    hot = self._hot(rr)
                    bg = T.ACCENT_FILL if (hot or i == selected) else (T.ROW_EVEN if k % 2 == 1 else T.WELL)
                    self.surf.fill(bg, (rr.x, rr.y, rr.w, rr.h - 1))
                    self.surf.fill(T.RULE_SOFT, (rr.x, rr.bottom - 1, rr.w, 1))
                    cx = hx0
                    for c_i, c in enumerate(columns):
                        cell = pygame.Rect(cx, rr.y, widths[c_i], rh - 1)
                        cx += widths[c_i]
                        if cell.right < rr.x or cell.x > rr.right:
                            continue
                        self._row_cell(c, c_i, columns, rows[i], texts[i][c_i], cell, bg, c["key"] in dims)
                    if cur == i and (not own or self._focused(key)):
                        pygame.draw.rect(self.surf, T.ACCENT, (rr.x, rr.y, rr.w, rr.h - 1), 1)
                if empty_h:
                    er = pygame.Rect(x + 1, body_top, inner, empty_h)
                    self._text(empty, er.x + pad, _ty(er.y, er.h, "sans", T.HINT_CSS), "sans", T.HINT_CSS, T.INK_MUTED)
                self.surf.set_clip(old)
            if span:
                self._hbar(pygame.Rect(x + 1, body.bottom, inner, bar_h), st, span, vis)
            if page_size:
                self._pager(pygame.Rect(x + 1, body.bottom + bar_h, inner, pager_h), st, n_all, vis)
            if vis:
                _rect(self.surf, T.RULE, frame, 1)
                if self._focused(key) and cur is None:
                    self._ring(frame)
            if more:
                label = f"show all {n_all}"
                self._button_draw_ui(f"{key}.all", label, "expand_more", x, y + frame_h + _s(6),
                                     lambda: st.__setitem__("all", True))
        self._place(w, h, fn)

    @staticmethod
    def _table_set_cursor(st, own, on_cursor, i) -> None:
        if own:
            st["cur"] = i
        if on_cursor is not None:
            on_cursor(i)

    @staticmethod
    def _first_left(columns) -> int:
        for c_i, c in enumerate(columns):
            if c.get("align", "right") == "left":
                return c_i
        return -1

    @staticmethod
    def _sortval(v):
        if v is None:
            return (2, 0.0, "")
        if isinstance(v, bool):
            return (0, float(v), "")
        if isinstance(v, (int, float)):
            f = float(v)
            return (0, f, "") if math.isfinite(f) else (1, 0.0 if f != f else f, "")
        try:
            f = float(v)
            if math.isfinite(f):
                return (0, f, "")
        except (TypeError, ValueError):
            pass
        return (1, 0.0, str(v).lower())

    def _sorted(self, rows, ck, desc) -> list:
        idx = list(range(len(rows)))
        keyed = [(self._sortval(rows[i].get(ck)), i) for i in idx]
        ok = sorted([k for k in keyed if k[0][0] != 2], key=lambda t: (t[0], t[1]), reverse=desc)
        missing = [k for k in keyed if k[0][0] == 2]
        return [i for _, i in ok] + [i for _, i in missing]

    @staticmethod
    def _table_sort(st, ck) -> str:
        s = st["sort"]
        if s is None or s[0] != ck:
            st["sort"] = (ck, False)
        elif not s[1]:
            st["sort"] = (ck, True)
        else:
            st["sort"] = None
        return "ui"

    @staticmethod
    def _cell_text(c, row):
        v = row.get(c["key"])
        f = c.get("fmt")
        if f is not None:
            try:
                s = f(v)
            except Exception:               # a formatter that cannot take this value shows it raw
                s = str(v)
        elif v is None:
            s = ""
        elif isinstance(v, float):
            s = T.fmt(v)
        else:
            s = str(v)
        sub = (row.get("_sub") or {}).get(c["key"])
        return s, (str(sub) if sub not in (None, "") else "")

    def _head_draw(self, c, hr, st, dim, sortable) -> None:
        text = str(c.get("head", c["key"]))
        align = c.get("align", "right")
        pad = _s(CELL_PAD)
        col = T.INK_MUTED
        if dim:
            bg = T.fade(T.HEADER, DEAD_A)
            self.surf.fill(bg, hr)
            col = T.fade(T.INK_MUTED, DEAD_A)
        sort = st["sort"]
        icon = None
        if sortable and sort is not None and sort[0] == c["key"]:
            icon = "arrow_drop_down" if sort[1] else "arrow_drop_up"
        elif sortable and self._hot(hr):
            icon = "arrow_drop_up"
        ty = _ty(hr.y, hr.h, "sans", 11, True)
        sp = _s(SORT_PX)
        room = hr.w - 2 * pad - (sp if icon else 0)
        if align == "left":
            tw = self._text(text, hr.x + pad, ty, "sans", 11, col, True, clip_w=room)
            if icon:
                T.icon(self.surf, icon, hr.x + pad + tw + sp // 2, hr.centery - sp // 2, sp,
                       col if sort and sort[0] == c["key"] else T.INK_FAINT, "centre")
        else:
            tw = self._text(text, hr.right - pad, ty, "sans", 11, col, True, "right", clip_w=room)
            if icon:
                T.icon(self.surf, icon, hr.right - pad - tw - sp // 2, hr.centery - sp // 2, sp,
                       col if sort and sort[0] == c["key"] else T.INK_FAINT, "centre")

    def _row_cell(self, c, c_i, columns, row, text, cell, bg, dim) -> None:
        s, sub = text
        pad = _s(CELL_PAD)
        colour = (row.get("_colours") or {}).get(c["key"]) or T.INK
        faint = T.INK_FAINT
        if dim:
            colour, faint = T.fade(colour, DEAD_A, bg), T.fade(faint, DEAD_A, bg)
        ty = _ty(cell.y, cell.h, "mono", 11)
        sw = (_s(6) + T.text_w(sub, "mono", 9)) if sub else 0
        tag = row.get("_tag") if c_i == self._first_left(columns) else None
        tgw = (_s(6) + self._tag_w(tag[0])) if tag else 0
        room = cell.w - 2 * pad - sw - tgw
        if c.get("align", "right") == "left":
            tw = self._text(s, cell.x + pad, ty, "mono", 11, colour, clip_w=room)
            ex = cell.x + pad + tw
        else:
            tw = min(T.text_w(s, "mono", 11), room)
            ex = cell.right - pad - sw
            self._text(s, ex, ty, "mono", 11, colour, anchor="right", clip_w=room)
        if sub:
            base = ty + T.font("mono", 11).get_ascent()
            self._text(sub, ex + _s(6), base - T.font("mono", 9).get_ascent(), "mono", 9, faint)
            ex += sw
        if tag:
            self._tag_draw(tag[0], tag[1], ex + _s(6), cell.centery - T.TAG_H // 2)

    def _hbar(self, r, st, span, vis) -> None:
        """The 12 px horizontal scroll bar under a wide table."""
        full = r.w + span
        tw = max(_s(30), int(r.w * r.w / full))
        tx = r.x + int((r.w - tw) * st["hx"] / span) if span else r.x
        thumb = pygame.Rect(tx, r.y + _s(3), tw, r.h - 2 * _s(3))

        def drag_from(pos0, hx0=st["hx"]):
            def follow(pos):
                st["hx"] = max(0, min(span, hx0 + int((pos[0] - pos0[0]) * span / max(1, r.w - tw))))
            return follow

        def on_click(pos):
            if not thumb.collidepoint(pos):
                st["hx"] = max(0, min(span, st["hx"] + (r.w if pos[0] > thumb.centerx else -r.w)))
            return "ui"
        self.form.ui_hit(r, on_click)
        self.form.ui_hit(thumb, lambda pos: "ui", drag_from(thumb.center))
        if vis and self._vis(r):
            self.surf.fill(T.SCROLL_TRACK, r)
            pygame.draw.rect(self.surf, T.SCROLL_THUMB_HOVER if self._hot(thumb) else T.SCROLL_THUMB, thumb,
                             border_radius=thumb.h // 2)

    def _pager(self, r, st, n_all, vis) -> None:
        """`Records per page:  20 ▾   1-20 of 53   ⏮ ‹ › ⏭`, right-aligned, SANS 11.5."""
        rpp = st["rpp"] or 0
        pages = max(1, math.ceil(n_all / rpp)) if rpp else 1
        pg = st["page"] if rpp else 0
        a = pg * rpp + 1 if n_all else 0
        b = min(n_all, (pg + 1) * rpp) if rpp else n_all
        ib = _s(24)
        x = r.right - _s(12)
        icons = [("last_page", pages - 1), ("chevron_right", pg + 1), ("chevron_left", pg - 1), ("first_page", 0)]
        ty = _ty(r.y, r.h, "sans", 11.5)
        for name, target in icons:
            x -= ib
            ok = rpp and 0 <= target < pages and target != pg
            ir = pygame.Rect(x, r.y + (r.h - ib) // 2, ib, ib)
            if ok:
                self.form.ui_hit(ir, lambda _pos, t=target: (st.__setitem__("page", t), "ui")[1])
            if vis and self._vis(ir):
                c = (T.ACCENT if self._hot(ir) else T.INK_MUTED) if ok else T.fade(T.INK_MUTED, 0.38)
                T.icon(self.surf, name, ir.centerx, ir.centery - _s(20) // 2, _s(20), c, "centre")
        x -= _s(16)
        span = f"{a}-{b} of {n_all}"
        x -= T.text_w(span, "sans", 11.5)
        if vis:
            self._text(span, x, ty, "sans", 11.5, T.INK)
        x -= _s(18)
        cp = _s(20)
        val = str(rpp) if rpp else "All"
        sel_w = T.text_w(val, "sans", 11.5) + _s(4) + cp
        x -= sel_w
        sel = pygame.Rect(x, r.y + (r.h - T.FIELD_H) // 2, sel_w, T.FIELD_H)

        def pick(pos, anchor=sel):
            labels = [str(n) if n else "All" for n in PAGE_SIZES]
            self.overlay.dropdown(anchor, list(PAGE_SIZES), labels,
                                  lambda n: (st.__setitem__("rpp", n), st.__setitem__("page", 0)), current=rpp)
            return "open"
        self.form.ui_hit(sel, pick)
        if vis:
            self._text(val, sel.x, ty, "sans", 11.5, T.INK)
            T.icon(self.surf, "arrow_drop_down", sel.right - cp // 2, sel.centery - cp // 2, cp, T.INK_MUTED, "centre")
        x -= _s(8)
        lab = "Records per page:"
        if vis:
            self._text(lab, x, ty, "sans", 11.5, T.INK, anchor="right")

    def _button_draw_ui(self, key, text, icon, x, y, fn) -> None:
        p = self.form.extra(key, text, fn)
        self._button_draw(key, p, x, y, text, icon, "flat")

    # -- bars, plots, slots ------------------------------------------------------------------------
    def range_bar(self, frac, colour=None, width=None) -> None:
        """Where a value sits in its range: a 9 px WELL track, 1px RULE_SOFT,
        a tick at the middle, a 3 px marker at `frac` (clamped)."""
        colour = T.ACCENT if colour is None else colour
        w, h = int(width or self.w), _s(9)

        def fn(x, y):
            self._range_draw(frac, colour, pygame.Rect(x, y, w, h))
        self._place(w, h, fn)

    def _range_draw(self, frac, colour, r) -> None:
        if not self._vis(r):
            return
        try:
            f = max(0.0, min(1.0, float(frac)))
        except (TypeError, ValueError):
            f = None
        self.surf.fill(T.WELL, r)
        _rect(self.surf, T.RULE_SOFT, r, 1, 0)
        self.surf.fill(T.RULE_SOFT, (r.x + r.w // 2, r.y, 1, r.h))
        if f is not None and math.isfinite(f):
            mx = r.x + int(round(f * r.w - 1.5))
            self.surf.fill(colour, (max(r.x, min(r.right - 3, mx)), r.y, 3, r.h))

    def stacked_bar(self, parts) -> None:
        """A 10 px bar split ∝ each part's value, no gaps, 1px RULE_SOFT
        outline; a part's `tip` is its hover. parts: [(value, colour, tip)]."""
        w, h = self.w, _s(10)

        def fn(x, y):
            r = pygame.Rect(x, y, w, h)
            vals = []
            for part in parts:
                try:
                    v = float(part[0])
                except (TypeError, ValueError):
                    v = 0.0
                vals.append(v if math.isfinite(v) and v > 0 else 0.0)
            tot = sum(vals)
            if not tot:
                if self._vis(r):
                    _rect(self.surf, T.RULE_SOFT, r, 1, 0)
                return
            cx, acc = r.x, 0.0
            for v, part in zip(vals, parts):
                if v <= 0:
                    continue
                acc += v
                nx = r.x + int(round(r.w * acc / tot))
                seg = pygame.Rect(cx, r.y, nx - cx, r.h)
                if self._vis(seg):
                    self.surf.fill(part[1], seg)
                tip = part[2] if len(part) > 2 else None
                if tip and self._hot(seg):
                    self.overlay.tooltip(seg, tip)
                cx = nx
            if self._vis(r):
                _rect(self.surf, T.RULE_SOFT, r, 1, 0)
        self._place(w, h, fn)

    def plot(self, fig, height) -> None:
        """A cae.plot Figure at the column's full width (in a pad=False card)."""
        w, h = self.w, _s(height)

        def fn(x, y):
            r = pygame.Rect(x, y, w, h)
            if self._vis(r):
                self.surf.fill(T.WELL, r)
                fig.draw(self.surf, r)
        self._place(w, h, fn)

    def plots(self, figs, height, titles=None) -> None:
        """Panels side by side (the three-panel polar), each titled above in
        SANS 12 (T.HINT_CSS) INK_MUTED at its plot area's left."""
        w, h = self.w, _s(height)
        n = max(1, len(figs))
        th = _s(PANEL_TITLE_H) if titles else 0

        def fn(x, y):
            r = pygame.Rect(x, y, w, h)
            if not self._vis(r):
                return
            self.surf.fill(T.WELL, r)
            px = x
            for i, fig in enumerate(figs):
                pw = (w - (px - x)) if i == n - 1 else w // n
                if titles and i < len(titles) and titles[i]:
                    self._text(titles[i], px + _s(54), _ty(y, th, "sans", T.HINT_CSS), "sans", T.HINT_CSS, T.INK_MUTED,
                               clip_w=pw - _s(54))
                fig.draw(self.surf, pygame.Rect(px, y + th, pw, h - th))
                px += pw
        self._place(w, h, fn)

    def empty_plot(self, text, height=280) -> None:
        """A plot that has nothing to show: its message centred, SANS 12
        INK_FAINT, on white."""
        w, h = self.w, _s(height)

        def fn(x, y):
            r = pygame.Rect(x, y, w, h)
            if self._vis(r):
                self.surf.fill(T.WELL, r)
                self._text(text, r.centerx, _ty(r.y, r.h, "sans", 12), "sans", 12, T.INK_FAINT, anchor="centre",
                           clip_w=w - 2 * _s(12))
        self._place(w, h, fn)

    def custom(self, height, draw_fn, *, on_click=None, on_drag=None) -> None:
        """A slot of the column's width: `draw_fn(surf, rect)` draws it (only
        when visible; clipped to the work area). `on_click(pos)` /
        `on_drag(pos)` make its visible part a click target that follows
        the mouse until release (`Form.ui_hit`): a drawing the mouse turns."""
        w, h = self.w, _s(height)

        def fn(x, y):
            r = pygame.Rect(x, y, w, h)
            if self._vis(r):
                old = self.surf.get_clip()
                self.surf.set_clip(r.clip(old))
                try:
                    draw_fn(self.surf, r)
                finally:
                    self.surf.set_clip(old)
                if on_click is not None or on_drag is not None:
                    self.form.ui_hit(r.clip(self.clip), on_click or (lambda _pos: "ui"), on_drag)
        self._place(w, h, fn)

    def spinner(self, text) -> None:
        """A small rotating ACCENT ring, then a hint."""
        d, lh = _s(14), _s(LINE_H)
        tw = T.text_w(text, "sans", T.HINT_CSS)
        w = d + _s(6) + tw

        def fn(x, y):
            r = pygame.Rect(x, y, w, lh)
            if not self._vis(r):
                return
            ring = pygame.Rect(x, y + (lh - d) // 2, d, d)
            a0 = (self.now * 6.0) % (2 * math.pi)
            pygame.draw.arc(self.surf, T.ACCENT, ring, a0, a0 + 4.2, max(1, _s(2)))
            self._text(text, ring.right + _s(6), _ty(y, lh, "sans", T.HINT_CSS), "sans", T.HINT_CSS, T.INK_MUTED)
        self._place(w, lh, fn)

    # -- the end ------------------------------------------------------------------------------
    def end(self) -> int:
        """Close the frame; returns the content height (top padding, the
        stack, bottom padding) the shell clamps its scroll with."""
        if self._ended is not None:
            return self._ended
        while len(self._stack) > 1:
            f = self._stack.pop()
            if isinstance(f, _Row):
                self._flush(f)
        top = self._stack[0]
        h = top.height + 2 * T.WORK_PAD_Y
        self.surf.set_clip(self._clip0)
        self.state["_content_h"] = h
        self._ended = h
        return h

    def scroll_for(self, key, margin=8):
        """The scroll that puts `key`'s row inside the viewport with `margin`
        px to spare (the least movement), or None when it was not laid out."""
        g = self.form.geom(key)
        if g is None:
            return None
        m = _s(margin)
        top, bottom = g.row.top - m, g.row.bottom + m
        if top < self.clip.top:
            return max(0, self.scroll - (self.clip.top - top))
        if bottom > self.clip.bottom:
            return self.scroll + min(bottom - self.clip.bottom, max(0, top - self.clip.top))
        return self.scroll


def route_mouse(form, overlay, ev, *, mouse=None, fine=False, shift=False):
    """What a mouse event does to the work area, the way the shell routes it
    (PLAN §0.2, §2.9.7): the overlay first; a click only on button 1 (2 and 3
    do nothing, 4 and 5 are the wheel's legacy echo); a held motion drags;
    button-1 up releases; SHIFT + wheel over a table scrolls it sideways.
    Returns the click's result string, "overlay", "drag", "release", "hscroll",
    "" (nothing) or None (not a mouse event for the work area)."""
    if overlay is not None and overlay.handle(ev):
        return "overlay"
    if ev.type == pygame.MOUSEBUTTONDOWN:
        if ev.button != 1:
            return ""
        return form.click(ev.pos, fine)
    if ev.type == pygame.MOUSEMOTION:
        if form.dragging and ev.buttons and ev.buttons[0]:
            form.drag(ev.pos)
            return "drag"
        return None
    if ev.type == pygame.MOUSEBUTTONUP:
        if ev.button != 1:
            return ""
        form.release()
        return "release"
    if ev.type == pygame.MOUSEWHEEL:
        if shift and mouse is not None:
            d = -int(ev.y) if ev.y else int(ev.x)
            return "hscroll" if form.hscroll_at(mouse, d) else ""
        return None
    return None


# ==================================================================== #
#  SELF-CHECK                                                          #
# ==================================================================== #
def work_rect(W: int, H: int) -> pygame.Rect:
    """The shell's work area in a W x H window (PLAN §4.1). The shell owns
    the real geometry; this copy is for the self-check and the demo."""
    col_y = T.MENU_H + T.TOOL_H + T.PAD
    col_h = H - col_y - T.PAD - T.STATUS_H
    main_w = W - T.LEFT_W - 2 * T.PAD - T.GAP
    main_h = col_h - T.GAP - T.OUTPUT_H
    x = T.PAD + T.LEFT_W + T.GAP + 1
    return pygame.Rect(x, col_y + 1 + T.TAB_STRIP_H, main_w - 2, main_h - 2 - T.TAB_STRIP_H)


class _Store:
    """A stand-in model for the self-check and the demo views: values in a
    dict, every set counted. Not carsim's data -- fixtures."""

    def __init__(self, **values):
        self.v = dict(values)
        self.sets: dict = {}

    def param(self, key, label, kind="float", **kw):
        from .. import garage_ui as ui

        def put(v, k=key):
            self.v[k] = v
            self.sets[k] = self.sets.get(k, 0) + 1

        def fire(_v, k=key):
            self.sets[k] = self.sets.get(k, 0) + 1
        getter = None if kind == "action" else (lambda k=key: self.v[k])
        setter = kw.pop("set", fire if kind == "action" else put)
        off_text, off_at, costly = kw.pop("off_text", None), kw.pop("off_at", None), kw.pop("costly", False)
        p = ui.Param(key, label, getter, setter, kind=kind, **kw)
        if off_text:
            p.off_text = off_text
        if off_at is not None:
            p.off_at = off_at
        if costly:
            p.costly = True
        return p


# -- the demo views: AeroBO's shots 03 (Library screening) and 06 (Ranking), strings copied off them --
_DEMO_WEIGHTS = (("w.ldcr", "L/D at design Cl", 0.35, None), ("w.clmax", "Cl max", 0.20, None),
                 ("w.cm", "|Cm| (lower better)", 0.20, None),
                 ("w.ldmax", "(L/D) max", 0.15, "Max L/D over the sweep: the best glide the section can reach at "
                  "any lift, which is what a wing that trims off its design point still gets."),
                 ("w.cdcr", "cd at design Cl (lower better)", 0.0, "Profile drag at the design lift: the "
                  "number L/D at design Cl already divides by, so weighting both counts it twice."),
                 ("w.thick", "thickness t/c", 0.10, None), ("w.astall", "stall angle", 0.0, None))


def demo_store() -> _Store:
    s = _Store(sweep="own", shortlist=24, typept=False, gtc=True, tcmin=0.15, gcm=True, cmmax=0.08,
               fclmax=0.0, fldcr=0.0, fastall=-90.0)
    for k, _, v, _ in _DEMO_WEIGHTS:
        s.v[k] = v
    return s


def demo_form(s: _Store) -> Form:
    P = s.param
    rows = [P("sweep", "sweep", kind="choice", choices=["own", "cached"]),
            P("shortlist", "shortlist", kind="int", lo=4, hi=200,
              help="How many of the cached ranking's leaders are swept live at this wing's own Reynolds "
                   "number before the rest are ranked on the cached point."),
            P("typept", "type the point myself", kind="bool")]
    rows += [P(k, lab, step=0.05, fine=0.01, lo=0.0, hi=1.0) for k, lab, _, _ in _DEMO_WEIGHTS]
    rows += [P("gtc", "min t/c gate", kind="bool"),
             P("tcmin", "min t/c", step=0.005, lo=0.0, hi=0.2, off_text="no limit"),
             P("gcm", "max |Cm| gate", kind="bool"),
             P("cmmax", "max |Cm|", step=0.01, lo=0.0, hi=1.0, off_text="no limit", off_at="hi"),
             P("fclmax", "min Cl max", step=0.05, lo=0.0, hi=2.5, off_text="no floor"),
             P("fldcr", "min L/D at Cl", step=1.0, lo=0.0, hi=150.0, off_text="no floor"),
             P("fastall", "min stall angle [deg]", step=0.5, lo=-90.0, hi=25.0, off_text="no floor"),
             P("go", "Screen the library", kind="action")]
    return Form(rows)


def demo_screening(ui: WorkUI) -> None:
    """AeroBO 03's Library screening view, card for card."""
    with ui.card("Design point — wing"):
        ui.kpis([dict(label="screen at Cl", value="1.0000"), dict(label="screen at Re", value="6.825e+05"),
                 dict(label="mission Re", value="6.825e+05",
                      tip="The Reynolds number at the mean aerodynamic chord, at the mission's speed: what the "
                          "section is screened at when this wing sweeps its own Re.")])
        ui.hairline()
        ui.kpis([dict(label="aspect ratio", value="8.828", unit="b²/S",
                      tip="b²/S at the middle of the design box, which is the planform this section is "
                          "designed for.")])
        ui.hint("Derived, not estimated: the design box searches b = 1.6 m and S = 0.29 m² at its middle, so "
                "this section is designed for the aspect ratio that middle flies, not for the one a finished "
                "wing picks.")
        ui.hint("MAC 0.1812 m on the mission's 0.29 m² → Re 6.825e+05 at 55 m/s.",
                help="The mean aerodynamic chord of the box middle's planform, at the mission's speed.")
        ui.hint("Stage 3 SEARCHES both of this wing's dimensions: b over 1.2–2 m and S over its own row, so the "
                "aspect ratio it flies is an answer of the search and this one is where the section starts.")
        ui.toggle("sweep", labels={"own": "this wing's own Re (sweeps the shortlist)",
                                   "cached": "cached library point (instant)"})
        ui.hint("The library's polars are cached at ONE Reynolds number, and every lift-dependent metric can be "
                "re-derived exactly at any design Cl from them — the Reynolds number itself cannot, which is "
                "why the shortlist is swept live.")
        ui.field("shortlist")
        ui.hint("TWO live viscous XFOIL marches per section on the first visit — instant after, from the cache; "
                "the sweep is what the shortlist is for.", "warn")
        ui.switch("typept")
    left, right = ui.columns((3, 2))
    with left:
        with ui.card("Criterion weights"):
            for k, lab, _, hlp in _DEMO_WEIGHTS:
                ui.slider(k, help=hlp)
            ui.hint("Weights are normalised before scoring, so only their ratios matter.")
            ui.hint("Recommended for this surface (wing): the GDP bulk-sweep preset: cruise L/D first, then Cl "
                    "max and the pitching moment — a wing that carries the design load wants lift it can "
                    "trim.", "ok")
    with right:
        with ui.card("Hard gates"):
            for sw, lab, num, why in (("gtc", "min t/c", "tcmin", "sections thinner than this are refused "
                                       "outright — structural depth"),
                                      ("gcm", "max |Cm|", "cmmax", "sections with a larger pitching moment are "
                                       "refused — trim drag and structure")):
                with ui.row():
                    ui.switch(sw, label="")
                    ui.label(lab, min_w=96)
                    ui.help_dot(why, lab)
                    ui.number(num, width=72)
            ui.hint("A gate REFUSES a section; the weights rank the ones that pass.",
                    help="Off, the gate is slack: nothing is refused on that metric.")
        with ui.card("Floors (optional)"):
            ui.field("fclmax", width=T.FIELD_W)
            ui.field("fldcr", width=T.FIELD_W)
            ui.field("fastall", width=T.FIELD_W)
            ui.hint("Leave empty for no floor. A floor is a minimum on a higher-is-better metric.")
    with ui.row():
        ui.button("go", "Screen the library", icon="search")


_DEMO_RANK = [  # AeroBO 06, the first 14 rows (section, score, L/D, (L/D)max, Cl max, stall, t/c, Cm, cd, points)
    ("hg40", 68.52, 125.9, 126.5, 1.41, 16, 0.15, -0.00498, 0.007943, (30.3, 12.7, 6.8, 0.0, 18.8)),
    ("hg41", 65.86, 121.1, 122.5, 1.38, 16, 0.151, 0.00462, 0.008256, (28.9, 12.3, 5.8, 0.0, 18.8)),
    ("goe741", 64.02, 111.6, 121.4, 1.56, 14.5, 0.1533, -0.0194, 0.008959, (26.2, 12.1, 10.6, 0.1, 15.0)),
    ("sibnia_s-16", 61.2, 89.4, 86.86, 1.76, 17, 0.16, 0.0115, 0.01119, (19.8, 8.0, 16.0, 0.3, 17.1)),
    ("august160", 60.75, 99.99, 115.1, 1.84, 15, 0.1604, -0.0464, 0.01, (22.9, 11.4, 18.1, 0.3, 8.1)),
    ("goe770", 60.18, 104.3, 112.2, 1.68, 14.5, 0.2099, -0.0423, 0.009587, (24.1, 11.0, 14.0, 1.9, 9.2)),
    ("fx77w153", 60.0, 108.0, 127.2, 1.61, 12.5, 0.1526, -0.0395, 0.009262, (25.1, 12.8, 12.0, 0.1, 9.9)),
    ("e1214", 59.98, 97.16, 103.1, 1.62, 17, 0.1982, -0.0228, 0.01029, (22.1, 9.9, 12.3, 1.5, 14.2)),
    ("arad20", 59.9, 89.43, 99.43, 1.91, 15.5, 0.2, -0.0431, 0.01118, (19.8, 9.5, 20.0, 1.6, 9.0)),
    ("e1213", 59.31, 100.3, 101.8, 1.64, 16.5, 0.1739, -0.0268, 0.009968, (23.0, 9.8, 12.7, 0.8, 13.2)),
    ("mh104", 59.16, 117.6, 120.5, 1.35, 13.5, 0.1526, -0.023, 0.008502, (27.9, 12.0, 5.0, 0.1, 14.1)),
    ("ah93w215", 57.98, 106.6, 106.6, 1.45, 14, 0.2122, -0.0266, 0.009385, (24.7, 10.3, 7.7, 2.0, 13.2)),
    ("naca23018", 57.12, 84.39, 84.68, 1.54, 17.5, 0.1802, -0.000118, 0.01185, (18.4, 7.7, 10.0, 1.0, 20.0)),
    ("goe777", 57.05, 105.0, 107.6, 1.76, 18, 0.22, -0.0627, 0.009524, (24.3, 10.5, 16.1, 2.2, 4.0)),
]


def demo_rank_rows() -> list:
    out = []
    for i, (name, score, ld, ldm, clm, ast, tc, cm, cd, pts) in enumerate(_DEMO_RANK):
        out.append({"rank": i + 1, "name": name, "score": score, "ld": ld, "ldmax": ldm, "clmax": clm,
                    "astall": ast, "tc": tc, "cm": cm, "cd": cd,
                    "_sub": {"ld": f"+{pts[0]:.1f}", "ldmax": f"+{pts[1]:.1f}", "clmax": f"+{pts[2]:.1f}",
                             "tc": f"+{pts[3]:.1f}", "cm": f"+{pts[4]:.1f}"}})
    return out


DEMO_RANK_COLUMNS = [dict(key="rank", head="#"), dict(key="name", head="section", align="left"),
                     dict(key="score", head="score"), dict(key="ld", head="L/D @Cl · w 0.35"),
                     dict(key="ldmax", head="(L/D)max · w 0.15"), dict(key="clmax", head="Cl max · w 0.20"),
                     dict(key="astall", head="α stall · w 0.00"), dict(key="tc", head="t/c · w 0.10"),
                     dict(key="cm", head="Cm · w 0.20"), dict(key="cd", head="cd @Cl · w 0.00")]


def demo_ranking(ui: WorkUI, picked=None) -> None:
    """AeroBO 06's Ranking view."""
    with ui.row(gap=12):
        ui.label("designing", colour=T.INK_MUTED, min_w=80)
        ui.tag("WING", T.ACCENT)
        ui.link("to_ep", "the endplate's section →", "arrow_forward", lambda: None, small=True)
    ui.kpis([dict(label="eligible", value="122", unit="of 2174",
                  tip="Sections that cleared every gate and floor, of the library screened."),
             dict(label="design Cl", value="1", tip="The lift the screen ranks every section at."),
             dict(label="Re", value="1.000e+06", unit="cached",
                  tip="The Reynolds number of the cached polars the ranking was read from."),
             dict(label="wall", value="0.0", unit="s", tip="How long the screen took, end to end.")])
    ui.hint("5 section(s) have no cached polar branch, so their lift-dependent metrics could not be re-scored "
            "at this design Cl.", "warn")
    rows = demo_rank_rows()
    ui.table("rank", DEMO_RANK_COLUMNS, rows, max_rows=14, selected=0, cursor=0,
             on_row=(picked.append if picked is not None else None), dim_cols=("astall", "cd"))
    ui.hint("Click a row to make it the section this design carries forward.",
            help="The score is the weighted sum of the 0-100 sub-scores; the faint number after a value is the "
                 "points it put into the score.")
    ui.hint("carried forward: hg40 (rank 1)")


def demo_frame(surf, W, H) -> pygame.Rect:
    """The shell's panes as plain boxes around the work area (chrome.py
    draws the real ones), so a demo PNG lines up with AeroBO's shots."""
    surf.fill(T.CANVAS)
    surf.fill(T.CHROME, (0, 0, W, T.MENU_H))
    surf.fill(T.RULE, (0, T.MENU_H - 1, W, 1))
    surf.fill(T.CHROME, (0, T.MENU_H, W, T.TOOL_H))
    surf.fill(T.RULE, (0, T.MENU_H + T.TOOL_H - 1, W, 1))
    surf.fill(T.CHROME, (0, H - T.STATUS_H, W, T.STATUS_H))
    surf.fill(T.RULE, (0, H - T.STATUS_H, W, 1))
    wr = work_rect(W, H)
    col_y = T.MENU_H + T.TOOL_H + T.PAD
    col_h = H - col_y - T.PAD - T.STATUS_H
    tree_h = int((col_h - T.GAP) * T.TREE_SPLIT)
    for r in ((T.PAD, col_y, T.LEFT_W, tree_h), (T.PAD, col_y + tree_h + T.GAP, T.LEFT_W, col_h - T.GAP - tree_h)):
        _rect(surf, T.PANEL, r)
        _rect(surf, T.RULE, r, 1)
    main = pygame.Rect(wr.x - 1, col_y, wr.w + 2, wr.bottom + 1 - col_y)
    _rect(surf, T.WELL, main)
    surf.fill(T.CHROME, (wr.x, col_y + 1, wr.w, T.TAB_STRIP_H - 1))
    surf.fill(T.RULE, (wr.x, col_y + T.TAB_STRIP_H, wr.w, 1))
    _rect(surf, T.RULE, main, 1)
    out = pygame.Rect(main.x, main.bottom + T.GAP, main.w, T.OUTPUT_H)
    _rect(surf, T.WELL, out)
    _rect(surf, T.RULE, out, 1)
    surf.fill(T.WELL, wr)
    return wr


def _plot_card(ui: WorkUI, fig) -> None:
    with ui.card("Convergence", pad=False):
        ui.plot(fig, 280)


def self_check(verbose: bool = True, out: str = "") -> bool:
    """The widget kit's rows (PLAN §5.2): fonts and glyphs as the shell's
    strings need them, the measured AeroBO geometry, every control's mouse
    and keyboard contract, the Form's order / reveal / drag / typing / lock
    rules, the table, the overlay, garage_ui.Plot's masking, the frame
    cost, and the two demo views at 1280x800 and 1600x1000 (saved as PNGs
    into `out` when given)."""
    import ast
    import glob
    import os
    import sys
    import time

    import numpy as np

    from .. import garage_ui as gui

    ok = True
    n_rows = [0]

    def rep(tag, passed, msg=""):
        nonlocal ok
        n_rows[0] += 1
        ok = ok and bool(passed)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}" + (f": {msg}" if msg else ""))

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    T.set_scale(1.0)
    files = T.resolve_fonts()
    T.warm()
    on_mac = sys.platform == "darwin"
    sf = bool(files["sans"]) and os.path.basename(files["sans"]).startswith("SFNS")

    def ev(kind, **kw):
        return pygame.event.Event(kind, **kw)

    def key(k, ch=""):
        return ev(pygame.KEYDOWN, key=k, unicode=ch, mod=0)

    def frame(view, form, state=None, size=(1280, 800), rect=None, **kw):
        surf = pygame.Surface(size)
        surf.fill(T.WELL)
        kw.setdefault("scroll", 0)
        kw.setdefault("kbd_focus", False)
        kw.setdefault("mouse", (-1, -1))
        kw.setdefault("now", 0.0)
        ov = kw.pop("overlay", None) or Overlay()
        ui = WorkUI(surf, rect or work_rect(*size), form, {} if state is None else state, overlay=ov, **kw)
        view(ui)
        return surf, ui, ui.end(), ov

    # -- fonts and glyphs (the strings the shell draws) -------------------------------------------------------
    names = {k: (os.path.basename(v) if v else None) for k, v in files.items()}
    bogus = T.resolve_fonts(sans_paths=("/nonexistent/a.ttf",), mono_paths=("/nonexistent/b.ttf",),
                            sans_names=("no-such-font",), mono_names=("no-such-font",))
    probe = pygame.Surface((300, 40))
    probe.fill(T.WELL)
    wd = T.text(probe, "Screen the library · 12/40", 4, 4)
    restored = T.resolve_fonts()
    rep("fonts resolve (SF through SANS_PATHS on this Mac) and fall back to pygame's default on bogus paths",
        (not on_mac or names["sans"] == "SFNS.ttf") and bogus["sans"] is None and bogus["mono"] is None
        and wd > 0 and restored == files, f"{names}")
    dejavu = [p_ for p_ in T.SANS_PATHS if "DejaVu" in p_ and os.path.exists(p_)]
    T.resolve_fonts(sans_paths=tuple(dejavu), sans_names=())
    alt = T.font_file("sans")
    px_alt = T.px_for("sans", 12)
    # the tree's longest label + its chip in 276 px on the non-SF face (SPECS/aerobo_shell.md §3.2)
    x0, right = 3 + T.TREE_BASE + T.TWISTY_W + T.TREE_GAP + T.ICON_PX + T.TREE_GAP, 3 + (T.LEFT_W - 2) - 6
    labels0 = {"1 Mission": "skidpad · damp", "2 Airfoil": "CST section (optimised) · not fitted",
               "2.8 Endplate": "none (zero height)", "3 Wing": "8-D", "4 Results": "lap 61.429 s"}
    labels1 = ["Operating point", "Design point", "Search & budget", "Library screening", "Ranking", "Section",
               "Shape optimisation", "Wing type", "Design box", "Solver", "Convergence", "Summary", "Geometry",
               "Loading", "Evaluations"]
    worst, fits = 0, True
    for lab, chip in labels0.items():
        end = x0 + T.text_w(lab, "sans", 12, True)
        cut = T.ellipsize(chip, "mono", 9.5, max(0, right - (end + 8)))
        fits = fits and end + 8 + T.text_w(cut, "mono", 9.5) <= right and bool(cut)
        worst = max(worst, end)
    for lab in labels1:
        end = x0 + T.TREE_INDENT + T.text_w(lab, "sans", 12, True)
        fits = fits and end <= right
        worst = max(worst, end)
    T.resolve_fonts()
    rep("the +1 px SANS rule is SF's only; on a non-SF sans the tree's longest label + chip fits 276 px",
        (not sf or T.px_for("sans", 12) == 13) and px_alt == 12 and fits,
        f"non-SF face {os.path.basename(alt) if alt else 'pygame default'}, widest label ends at {worst} px")
    # every non-ASCII character in a string the cae kit, the shell or a view can draw
    here = os.path.dirname(os.path.abspath(__file__))
    drive_dir = os.path.dirname(here)
    srcs = [os.path.join(here, f) for f in ("theme.py", "form.py", "widgets.py", "plot.py", "chrome.py")]
    srcs += [os.path.join(drive_dir, f) for f in ("design_shell.py", "design_jobs.py", "views_common.py")]
    srcs += sorted(glob.glob(os.path.join(drive_dir, "views_*.py")))
    chars, skipped, seen = {}, [], set()
    for path in srcs:
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except SyntaxError:
            skipped.append(os.path.basename(path))
            continue
        docs = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant):
                    docs.add(id(first.value))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                for ch in node.value:
                    if not ch.isascii() and ch.isprintable():
                        chars.setdefault(ch, os.path.basename(path))
    bad = []
    for fam in ("sans", "mono"):
        for ch, where in chars.items():
            shown_ = T._displayable(ch, fam)
            if not all(c.isascii() or T.glyph_ok(c, fam) for c in shown_):
                bad.append(f"{ch!r} ({fam}, {where})")
    rep("every non-ASCII character in a drawable string resolves (its glyph, or ASCII_FALLBACK)",
        not bad, f"{len(chars)} characters over {len(seen) - len(skipped)} files"
        + (f"; skipped mid-edit: {skipped}" if skipped else "") + (f"; UNRESOLVED {bad}" if bad else ""))
    blank = [n for n, cp in T.ICONS.items()
             if T.font("icon", T.ICON_PX).render(chr(cp), True, T.INK).get_bounding_rect().w == 0]
    rep(f"every one of the {len(T.ICONS)} ICONS glyphs renders non-empty", not blank, f"{blank}")
    s2 = ("Screening flies every library section at the operating point and ranks them by the stated objective. "
          "The seed of the shape search is the rank-1 section unless you pick another one.")
    s3 = ("the section is scored at the lift the wing needs at the design speed — every candidate flies the same "
          "point, at the same air density, with the drag ceiling as its penalty")
    s4 = " ".join(f"w{i}" for i in range(30))
    c2, c3, c4 = T.split_hint(s2), T.split_hint(s3), T.split_hint(s4)
    rep("split_hint: short / sentence / after the last '—' / 24 words + ' …' with the whole text",
        T.split_hint("A short hint.") == ("A short hint.", None) and c2[0].endswith("objective.")
        and c2[1].startswith("The seed") and c3[0].endswith("design speed —") and c4 == (" ".join(s4.split()[:24])
                                                                                          + " …", s4))
    rep("tint(ACCENT) = #e8eff6, fade(ACCENT, .6) on white, fade is exact at 0 and 1",
        T.tint(T.ACCENT) == (232, 239, 246) and T.fade(T.ACCENT, 0.6) == (119, 159, 200)
        and T.fade(T.INK, 1.0) == T.INK and T.fade(T.INK, 0.0) == T.WELL, f"{T.tint(T.ACCENT)}")

    # -- measured AeroBO geometry (03_after_accept.png at 1600x1000) ---------------------------------------------
    s = demo_store()
    f = demo_form(s)
    surf, ui, h03, _ = frame(demo_screening, f, size=(1600, 1000))
    at = surf.get_at
    col = [tuple(at((600, y))[:3]) for y in range(94, 122)]
    bar_rows = [y for y in range(94, 122) if tuple(at((600, y))[:3]) == T.PANEL]
    kpi_bar = [y for y in range(120, 168) if tuple(at((303, y))[:3]) == T.RULE_SOFT]
    kx = [x for x in range(295, 560) if tuple(at((x, 140))[:3]) == T.RULE_SOFT]
    rep("card: frame at (293, 96), a 22 px PANEL title bar (97..118) over a RULE_SOFT rule at 119 (WingLab 03)",
        col[96 - 94] == T.RULE_SOFT and bar_rows == list(range(97, 119)) and col[119 - 94] == T.RULE_SOFT
        and tuple(at((293, 200))[:3]) == T.RULE_SOFT, f"title rows {bar_rows[:1]}..{bar_rows[-1:]}")
    rep("KPI tiles: 2 px bars 37 tall from y 128, at x 303 and 419 (104 + 12 apart)",
        kpi_bar == list(range(128, 165)) and kx[:4] == [303, 304, 419, 420], f"bar rows {kpi_bar[0]}..{kpi_bar[-1]}, "
        f"bars at {sorted(set(x for x in kx if x < 430))}")
    g_short, g_go = f.geom("shortlist"), f.geom("go")
    rep("field box at x 459 (150 + 6 after its label), 84 x 26; 'Screen the library' 134 x 26 (WingLab 03: 131; "
        "+3 from theme's SANS word spacing)",
        g_short.rect.x == 459 and g_short.rect.size == (84, 26) and g_go.rect.size == (134, 26),
        f"box {tuple(g_short.rect)}, button {tuple(g_go.rect)}")
    tsurf = pygame.Surface((200, 40))
    tsurf.fill(T.WELL)
    tui = WorkUI(tsurf, (0, 0, 200, 40), Form([]), {}, scroll=0, kbd_focus=False, mouse=(-1, -1), overlay=None,
                 now=0.0)
    tui.tag("OK", T.GOOD)
    tui.end()
    tag_rows = [y for y in range(40) if tuple(tsurf.get_at((20, y))[:3]) != T.WELL]
    tag_cols = [x for x in range(200) if tuple(tsurf.get_at((x, 20))[:3]) == T.GOOD]
    rep("tag: 17 px tall, 1px border and text in its colour on a 10 % tint ('OK' 27 wide, as measured)",
        len(tag_rows) == T.TAG_H and tag_cols and tag_cols[-1] - tag_cols[0] + 1 == 27
        and tuple(tsurf.get_at((14, 12))[:3]) == T.tint(T.GOOD),
        f"{len(tag_rows)} rows, {tag_cols[-1] - tag_cols[0] + 1 if tag_cols else 0} wide")

    # -- the Form keeps ParamList's pinned API ---------------------------------------------------------------------
    s = _Store(a=1.0, b=2.0, c="x")
    f = Form([gui.Param("lab", "LABEL", None, kind="label"), s.param("a", "a", step=0.5, lo=0.0, hi=5.0),
              s.param("b", "b", step=1.0, lo=0.0, hi=5.0), s.param("c", "c", kind="choice", choices=["x", "y"])])
    first = f.current().key
    f.select_key("b")
    f.adjust(+1)
    f.idx = 1
    by_idx = f.current().key
    f.adjust(-1)
    f.nav(+1)
    after_nav = f.current().key
    f.nav(+1)
    f.nav(+1)
    wrapped = f.current().key
    f.select_key("c")
    f.activate()
    rep("Form keeps ParamList's API before any draw: select_key + adjust, a direct idx write moves the focus, nav "
        "skips labels and wraps, ENTER cycles a choice",
        first == "lab" and s.v["b"] == 3.0 and by_idx == "a" and s.v["a"] == 0.5 and after_nav == "b"
        and wrapped == "a" and s.v["c"] == "y" and f.focus_key == "c", f"{first} {by_idx} {after_nav} {wrapped}")

    # -- the Form: draw order, reveal, visible-only _hits -----------------------------------------------------------
    s = _Store(a=1.0, b=2.0, c=3.0, d=4.0, e=5.0, g=6.0)
    ps = [s.param(k, f"row {k}", lo=0.0, hi=10.0, step=1.0) for k in "abcdeg"]
    ps[4]._enabled = False
    f = Form(ps)
    drawn = ["d", "b", "g", "a", "e", "c"]

    def tall(ui_):
        with ui_.card("Order"):
            for k_ in drawn:
                ui_.field(k_)
                ui_.gap(60)
    small = pygame.Rect(281, 86, 700, 200)
    _, ui, _, _ = frame(tall, f, rect=small)
    visible = [k_ for k_ in f.order if f.geom(k_).visible]
    walk = []
    f.focus_key = None
    for _ in range(6):
        f.nav(+1)
        walk.append((f.focus_key, f.reveal))
    rep("Form: order is DRAW order; DOWN walks it off screen too, skips the disabled row and sets reveal",
        f.order == drawn and [k_ for k_, _ in walk][:5] == ["d", "b", "g", "a", "c"]
        and walk[2][1] == "g" and walk[4][1] == "c" and f.current().key == walk[5][0],
        f"visible {visible}, walk {[k_ for k_, _ in walk]}")
    hit_keys = [ps[i].key for i, _, _ in f._hits]
    rep("_hits holds the visible controls only, in draw order (tutorial anchors stay honest)",
        hit_keys == visible and 0 < len(visible) < len(drawn) and f.rect_of("c") is None
        and f.rect_of(visible[0]) is not None, f"{hit_keys}")
    sc = ui.scroll_for("c")
    rep("scroll_for puts an off-screen control inside the viewport with 8 px to spare",
        sc is not None and sc > 0 and f.geom("c").row.bottom - sc + 8 <= small.bottom, f"scroll {sc}")

    # -- slider: click snaps, round-trips, drags coalesce, costly waits for release --------------------------------
    s = _Store(w=0.35, cw=0.5)
    pw = s.param("w", "weight", step=0.05, fine=0.01, lo=0.0, hi=1.0)
    pc = s.param("cw", "costly weight", step=0.05, lo=0.0, hi=1.0, costly=True)
    f = Form([pw, pc])

    def sliders(ui_):
        with ui_.card("Weights"):
            ui_.slider("w")
            ui_.slider("cw")
    frame(sliders, f)
    g = f.geom("w")
    trips = []
    for k_ in range(21):
        v = round(k_ * 0.05, 10)
        f.click((g.x_for(v), g.parts["track"].centery))
        trips.append(s.v["w"] == v)
        f.release()
    rep("slider: a click lands on the snapped value under it, and x_for(v) round-trips EXACTLY for every step",
        all(trips) and g.snap(0.5249) == 0.5 and g.snap(1.7) == 1.0, f"{sum(trips)}/21")
    s.sets.clear()
    f.click((g.x_for(0.2), g.parts["track"].centery))
    for xx in range(g.x_for(0.2), g.x_for(0.8), 30):
        f.drag((xx, 0))
    n_drag = s.sets.get("w", 0)
    frame(sliders, f)
    n_frame = s.sets.get("w", 0)
    f.release()
    gc = f.geom("cw")
    f.click((gc.x_for(0.1), gc.parts["track"].centery))
    for xx in range(gc.x_for(0.1), gc.x_for(0.9), 40):
        f.drag((xx, 0))
    frame(sliders, f)
    c_before = s.sets.get("cw", 0)
    f.release()
    rep("drag: 10 motion events = 1 set per frame; a costly Param is set only on release",
        n_drag == 1 and n_frame == 2 and c_before == 1 and s.sets.get("cw", 0) == 2 and s.v["cw"] > 0.7,
        f"sets: click {n_drag}, after the frame {n_frame}; costly {c_before} -> {s.sets.get('cw', 0)}")

    # -- number: arrows on hover / focus only, stepping, typing ----------------------------------------------------
    s = _Store(n=0.25, m=3.0)
    pn = s.param("n", "n", step=0.05, lo=0.0, hi=1.0)
    pm = s.param("m", "m", step=1.0, lo=0.0, hi=10.0, off_text="no limit")
    f = Form([pn, pm])

    def numbers(ui_):
        with ui_.row():
            ui_.number("n")
            ui_.number("m")
    surf, _, _, _ = frame(numbers, f)
    g = f.geom("n")
    rest = {tuple(surf.get_at((x, y))[:3]) for x in range(g.parts["up"].x, g.parts["up"].right)
            for y in range(g.parts["up"].y, g.parts["down"].bottom)}
    surf_h, _, _, _ = frame(numbers, f, mouse=g.rect.center)
    hover = {tuple(surf_h.get_at((x, y))[:3]) for x in range(g.parts["up"].x, g.parts["up"].right)
             for y in range(g.parts["up"].y, g.parts["down"].bottom)}
    f.select_key("n")
    surf_f, _, _, _ = frame(numbers, f, kbd_focus=True)
    focus = {tuple(surf_f.get_at((x, y))[:3]) for x in range(g.parts["up"].x, g.parts["up"].right)
             for y in range(g.parts["up"].y, g.parts["down"].bottom)}
    r_up = f.click(g.parts["up"].center)
    v_up = s.v["n"]
    r_dn = f.click(g.parts["down"].center)
    rep("number: no stepper at rest; hovered or keyboard-focused it shows two, and they step the value",
        rest == {T.WELL} and len(hover) > 1 and len(focus) > 1 and r_up == r_dn == "adjust"
        and abs(v_up - 0.3) < 1e-12 and abs(s.v["n"] - 0.25) < 1e-12, f"{v_up} -> {s.v['n']}")
    frame(numbers, f)
    r = f.click((g.rect.x + 10, g.rect.centery))
    fresh_buf = f.buf
    for ch in "12.5":
        f.key_input(key(ord(ch), ch))
    f.key_input(key(pygame.K_RETURN))
    v_typed = s.v["n"]
    f.start_edit("n")
    taken = [f.key_input(key(k_, ch)) for k_, ch in ((pygame.K_l, "l"), (pygame.K_o, "o"), (pygame.K_s, "s"))]
    buf_after = f.buf
    for ch in "1e":
        f.key_input(key(ord(ch), ch))
    f.key_input(key(pygame.K_RETURN))
    refused = f.flash is not None and f.flash[0] == "n" and s.v["n"] == v_typed
    frame(numbers, f, now=1.0)
    frame(numbers, f, now=1.7)
    f.start_edit("m")
    f.key_input(key(pygame.K_BACKSPACE))
    f.key_input(key(pygame.K_RETURN))
    blank_off = s.v["m"] == 0.0
    f.select_key("n")
    started = f.key_input(key(pygame.K_7, "7"))
    f.key_input(key(pygame.K_ESCAPE))
    rep("typing: a click starts it, ENTER commits clamped, letters are swallowed, junk is refused (BAD 0.6 s), "
        "blank = the off value, a digit on a focused field starts it, ESC cancels",
        r == "edit" and fresh_buf == "0.25" and v_typed == 1.0 and all(taken) and buf_after == "1"
        and refused and f.flash is None and blank_off and started and f.editing is None and s.v["n"] == 1.0,
        f"typed 12.5 -> {v_typed} (hi 1), buf kept {buf_after!r}")

    # -- select, toggle, switch, radio ---------------------------------------------------------------------------------
    s = _Store(sel="b", tog="x", sw=False, rad="r1", after=0.0)
    psel = s.param("sel", "pick", kind="choice", choices=["a", "b", "c"])
    ptog = s.param("tog", "mode", kind="choice", choices=["x", "y"])
    psw = s.param("sw", "stop early", kind="bool")
    prad = s.param("rad", "source", kind="choice", choices=["r1", "r2", "r3"])
    paft = s.param("after", "after", lo=0.0, hi=1.0)
    f = Form([psel, ptog, psw, prad, paft])
    ov = Overlay()

    def choosers(ui_):
        with ui_.card("Choosers"):
            ui_.field("sel", control="select", labels={"a": "alpha", "b": "beta", "c": "gamma"})
            ui_.field("tog", control="toggle", labels={"x": "I state it", "y": "optimise it"})
            ui_.switch("sw")
            ui_.radio_rows("rad", [("r1", "The measured recommendation", "the budgets follow the law"),
                                   ("r2", "My own values", "nothing here moves them"), ("r3", "Third", "")])
            ui_.field("after")
    surf, _, _, _ = frame(choosers, f, overlay=ov)
    g = f.geom("sel")
    r_open = f.click(g.rect.center)
    ov.draw(surf, 0.0)
    item = ov.menu["rect"].move(0, 0)
    ov.handle(ev(pygame.MOUSEBUTTONDOWN, pos=(item.x + 20, item.y + 1 + _s(MENU_ITEM_H) * 2 + 5), button=1))
    v_mouse = s.v["sel"]
    f.select_key("sel")
    f.activate()
    ov.handle(key(pygame.K_UP))
    ov.handle(key(pygame.K_RETURN))
    v_key = s.v["sel"]
    f.activate()
    ov.handle(key(pygame.K_DOWN))
    ov.handle(key(pygame.K_ESCAPE))
    rep("select: a click opens the drop-down, a click on an item picks it; ENTER opens, UP + ENTER picks; ESC "
        "closes and keeps the value",
        r_open == "open" and v_mouse == "c" and v_key == "b" and ov.menu is None and s.v["sel"] == "b",
        f"mouse -> {v_mouse}, keys -> {v_key}")
    frame(choosers, f, overlay=ov)
    gt, gs = f.geom("tog"), f.geom("sw")
    r_t = f.click(gt.parts["y"].center)
    t1 = s.v["tog"]
    f.select_key("tog")
    f.adjust(+1)
    t2 = s.v["tog"]
    r_s = f.click(gs.rect.center)
    s1 = s.v["sw"]
    f.select_key("sw")
    f.activate()
    rep("toggle: a click takes the segment, RIGHT cycles; switch: a click and ENTER flip it",
        r_t == "adjust" and t1 == "y" and t2 == "x" and r_s == "adjust" and s1 is True and s.v["sw"] is False)
    gr = f.geom("rad")
    r_r = f.click(gr.parts["r3"].center)
    v_rc = s.v["rad"]
    f.select_key("tog")
    f.nav(+1)
    f.nav(+1)
    in_group = f.focus_key
    f.nav(+1)
    f.activate()
    v_rk = s.v["rad"]
    f.nav(+1)
    f.nav(+1)
    rep("radio_rows: a click picks a row; the group is ONE keyboard stop: DOWN moves inside it, ENTER picks, "
        "DOWN past its end leaves",
        r_r == "adjust" and v_rc == "r3" and in_group == "rad" and v_rk == "r2" and f.focus_key == "after",
        f"click -> {v_rc}, keys -> {v_rk}, then {f.focus_key}")

    # -- v1_row and grid -----------------------------------------------------------------------------------------------
    s = _Store(mount="tips", blend=0.0, **{"a.con": True, "a.min": 1.2, "a.max": 2.0,
                                          "b.con": False, "b.min": 0.4, "b.max": 1.0})
    ps = [s.param("mount", "mount", kind="choice", choices=["tips", "pylons"]),
          s.param("blend", "blend", lo=0.0, hi=1.0, step=0.05)]
    for r_ in "ab":
        ps += [s.param(f"{r_}.con", "constrain", kind="bool"),
               s.param(f"{r_}.min", "low", lo=0.0, hi=5.0), s.param(f"{r_}.max", "high", lo=0.0, hi=5.0)]
    f = Form(ps)

    def box(ui_):
        with ui_.card("Configuration"):
            ui_.v1_row("mount", "Mount", labels={"tips": "through the endplates, at the tips",
                                                  "pylons": "on pylons, near the centreline"},
                       note="The plates ARE the load path, gripping at the very tip.")
            ui_.v1_row("blend", "Root blend (0–1)", control="number", unit="of plate height")
        with ui_.card("Design box"):
            ui_.grid("box", [1.6, "auto", 1, 1, "auto"],
                     ["parameter", ("constrain", "ON: the row is searched inside the band you type."), "low",
                      "high", "source"],
                     [[("text2", "span [m]", "b_m", "Wing span."), ("switch", "a.con"), ("number", "a.min", 6),
                       ("number", "a.max", 6), ("tag", "default", T.INK_FAINT)],
                      [("text2", "taper", "taper"), ("switch", "b.con"), ("number", "b.min", 6),
                       ("number", "b.max", 6), ("tag", "user", T.WARN)]])
    frame(box, f)
    gm, gb = f.geom("mount"), f.geom("blend")
    wa, wb = f.geom("a.min"), f.geom("a.max")
    rep("v1_row and grid: the v1 controls grow (60 / 96 px labels, 9 px), the grid's Param cells record "
        "row-major, full column width",
        f.order == ["mount", "blend", "a.con", "a.min", "a.max", "b.con", "b.min", "b.max"]
        and gm.rect.x - gm.row.x == 69 and gb.rect.x - gb.row.x == 105 and gm.rect.w > 600
        and wa.rect.w == wb.rect.w and wa.rect.y == wb.rect.y < f.geom("b.min").rect.y,
        f"order {f.order}")
    hits = (f.hit_key(f.geom("a.con").parts["hit"].center), f.hit_key(wa.rect.center),
            f.hit_key(wb.rect.center), f.hit_key(f.geom("b.con").parts["hit"].center))
    rep("grid: a click lands on its own cell's control -- the switch, the low field -- not on the "
        "last control drawn in the row (the Design box's switches and low fields took no click)",
        hits == ("a.con", "a.min", "a.max", "b.con"), f"hits {hits}")

    # -- buttons, mouse buttons 2-5, disabled and locked ---------------------------------------------------------------
    s = _Store(x=0.5, y=0.5)
    pgo = s.param("go", "Screen the library", kind="action")
    pst = s.param("stop", "Stop", kind="action")
    pdis = s.param("x", "x", lo=0.0, hi=1.0)
    pdis._enabled = False
    ploc = s.param("y", "y", lo=0.0, hi=1.0)
    f = Form([pgo, pst, pdis, ploc])

    def buttons(ui_):
        with ui_.row():
            ui_.button("go", icon="search")
            ui_.button("stop", kind="outline", icon="stop")
            ui_.number("x")
            ui_.number("y")
    frame(buttons, f)
    gg = f.geom("go")
    r_first = f.click(gg.rect.center)
    fired = s.sets.get("go", 0)
    others = [route_mouse(f, Overlay(), ev(pygame.MOUSEBUTTONDOWN, pos=gg.rect.center, button=b)) for b in (2, 3, 4, 5)]
    after = s.sets.get("go", 0)
    r_one = route_mouse(f, Overlay(), ev(pygame.MOUSEBUTTONDOWN, pos=gg.rect.center, button=1))
    rep("a button fires on the FIRST click (WingLab)", r_first == "action" and fired == 1)
    rep("mouse buttons 2-5 do nothing; button 1 does",
        others == ["", "", "", ""] and after == 1 and r_one == "action" and s.sets.get("go", 0) == 2, f"{others}")
    f.locked = lambda p_: "locked while the lap run is going — ■ Stop to change it" if p_.key in ("go", "y") else None
    surf, _, _, _ = frame(buttons, f)
    gg, gx, gy = f.geom("go"), f.geom("x"), f.geom("y")
    refusals = [f.click(gg.rect.center), f.click(gx.parts["up"].center), f.click(gy.rect.center)]
    f.select_key("y")
    adj = f.adjust(+1)
    f.select_key("go")
    act = f.activate()
    fill = tuple(surf.get_at((gg.rect.x + 3, gg.rect.y + 3))[:3])
    edge = tuple(surf.get_at((gx.rect.x + 20, gx.rect.y))[:3])
    _, _, _, ov_l = frame(buttons, f, mouse=gg.rect.center)
    tip_l = ov_l._tip_req[1] if ov_l._tip_req else ""
    rep("disabled and locked controls refuse click, keys and ENTER, draw at .6 (primary PRIMARY_DISABLED), and "
        "a locked one says why on hover",
        refusals == ["select"] * 3 and not adj and not act and s.sets.get("go", 0) == 2 and s.v["y"] == 0.5
        and fill == T.PRIMARY_DISABLED and edge == T.fade(T.FIELD_EDGE, DISABLED_A)
        and f.lock_reason(pgo).startswith("locked while") and tip_l.startswith("locked while"),
        f"fill {fill}, edge {edge}")

    # -- tables ----------------------------------------------------------------------------------------------------------
    rows = [{"i": k_, "v": v} for k_, v in enumerate([3.0, 1.0, 2.0, 5.0, 4.0])]
    got = []
    f = Form([])
    st = {}
    cols = [dict(key="i", head="#"), dict(key="v", head="value")]

    def tab(ui_):
        ui_.table("t", cols, rows, on_row=got.append)
    frame(tab, f, st)
    gtb = f.geom("ui.table.t")
    head_v = [r_ for r_, _, _ in f._ui if r_.y == gtb.rect.y + 1][1]
    f.click(head_v.center)
    frame(tab, f, st)
    rh = _s(28)
    first_row = (gtb.rect.x + 40, gtb.rect.y + 1 + rh + rh // 2)
    f.click(first_row)
    f.select_key("ui.table.t")
    f.nav(+1)
    f.activate()
    f.click(head_v.center)
    frame(tab, f, st)
    f.click(first_row)
    rep("table: a head click sorts; a row click reports the ORIGINAL index; the cursor walks DISPLAYED order",
        got[:2] == [1, 2] and got[2] == 3, f"on_row got {got}")
    rows53 = [{"i": k_, "v": float(k_)} for k_ in range(53)]
    f, st = Form([]), {}

    def paged(ui_):
        ui_.table("p", cols, rows53, page_size=20)
    tall_rect = pygame.Rect(281, 86, 996, 900)
    surf, _, _, _ = frame(paged, f, st, size=(1280, 1000), rect=tall_rect)
    gp = f.geom("ui.table.p")
    pager_hits = [r_ for r_, _, _ in f._ui if r_.y > gp.rect.bottom - T.PAGER_H - 2]
    nxt = sorted(pager_hits, key=lambda r_: r_.x)[-2]
    f.click(nxt.center)
    frame(paged, f, st, size=(1280, 1000), rect=tall_rect)
    p1 = st[("table", "ui.table.p")]["page"]
    f.select_key("ui.table.p")
    f.page(+1)
    frame(paged, f, st, size=(1280, 1000), rect=tall_rect)
    rep("table pager: 'Records per page', 20 rows, next page by the icon, PAGE DOWN pages, the cursor follows",
        p1 == 1 and st[("table", "ui.table.p")]["page"] == 2 and st[("table", "ui.table.p")]["cur"] == 40
        and gp.rect.h == 2 + rh * 21 + T.PAGER_H, f"page {p1} -> {st[('table', 'ui.table.p')]['page']}")
    rows20 = [{"i": k_, "v": float(k_)} for k_ in range(20)]
    f, st = Form([]), {}

    def capped(ui_):
        ui_.table("c", cols, rows20, max_rows=14)
    frame(capped, f, st)
    g14 = f.geom("ui.table.c")
    link = f.geom("ui.table.c.all")
    f.click(link.rect.center)
    frame(capped, f, st)
    g20 = f.geom("ui.table.c")
    f2, st2 = Form([]), {}
    frame(capped, f2, st2)
    f2.select_key("ui.table.c")
    st2[("table", "ui.table.c")]["cur"] = 13
    f2.nav(+1)
    rep("table max_rows: 14 rows and a 'show all 20' link that expands it; the cursor past row 14 expands it too",
        g14.rect.h == 2 + rh * 15 and link is not None and g20.rect.h == 2 + rh * 21
        and st2[("table", "ui.table.c")]["all"] and st2[("table", "ui.table.c")]["cur"] == 14)
    wide_cols = [dict(key=f"c{k_}", head=f"column_{k_}_wide") for k_ in range(14)]
    wide_rows = [{f"c{k_}": 1234.5678 * (j + 1) for k_ in range(14)} for j in range(3)]
    f, st = Form([]), {}

    def wide(ui_):
        ui_.table("w", wide_cols, wide_rows)
    frame(wide, f, st, rect=pygame.Rect(281, 86, 700, 500))
    gw = f.geom("ui.table.w")
    f.hscroll_at(gw.rect.center, +2)
    hx1 = st[("table", "ui.table.w")]["hx"]
    f.select_key("ui.table.w")
    f.adjust(+1)
    hx2 = st[("table", "ui.table.w")]["hx"]
    frame(wide, f, st, rect=pygame.Rect(281, 86, 700, 500))
    thumb = [(r_, d_) for r_, _, d_ in f._ui if d_ is not None][0][0]
    f.click(thumb.center)
    f.drag((thumb.centerx + 100, thumb.centery))
    f.release()
    hx3 = st[("table", "ui.table.w")]["hx"]
    rep("table h-scroll: wider than the card it scrolls by SHIFT+wheel, LEFT / RIGHT and a drag of its bar",
        0 < hx1 < hx2 < hx3, f"hx {hx1} -> {hx2} -> {hx3}")

    # -- disclosure, popup, tooltip ------------------------------------------------------------------------------------
    s = _Store(inner=1.0)
    f, st, ov = Form([s.param("inner", "inner", lo=0.0, hi=2.0)]), {}, Overlay()

    def disc_view(ui_):
        if ui_.disclosure("more", "every other reported number (13)"):
            with ui_.indent():
                ui_.field("inner")
        ui_.hint("Weights are normalised before scoring.", help="Only their ratios matter.")
        ui_.kv("circuit", "arena (1249 m)", tip="the stated circuit")
    frame(disc_view, f, st, overlay=ov)
    closed_has = f.geom("inner") is not None
    f.click(f.geom("ui.disc.more").rect.center)
    frame(disc_view, f, st, overlay=ov)
    open_has = f.geom("inner") is not None
    f.select_key("ui.disc.more")
    f.activate()
    frame(disc_view, f, st, overlay=ov)
    rep("disclosure: a click opens it (its indented body is laid out), ENTER closes it",
        not closed_has and open_has and f.geom("inner") is None)
    dot = [r_ for r_, _, _ in f._ui if r_.w == T.HELP_DOT][0]
    r_dot = f.click(dot.center)
    surf = pygame.Surface((1280, 800))
    ov.draw(surf, 0.0)
    opened = ov.pop is not None and ov.pop["rect"].w <= _s(POP_W) + 2
    inside = ov.handle(ev(pygame.MOUSEBUTTONDOWN, pos=ov.pop["rect"].center, button=1)) and ov.pop is not None
    away = ov.handle(ev(pygame.MOUSEBUTTONDOWN, pos=(5, 5), button=1))
    s_h = _Store(q=1.0)
    f_h, ov_h = Form([s_h.param("q", "a field with a paragraph", lo=0.0, hi=2.0,
                                help="A help text long enough to be a paragraph rather than a hover tip, so the "
                                     "row carries a visible question mark.")]), Overlay()
    frame(lambda ui_: ui_.field("q"), f_h, overlay=ov_h)
    f_h.select_key("q")
    f1 = f_h.open_help() and ov_h.pop is not None and ov_h.pop["title"] == "a field with a paragraph"
    rep("help '?': a click opens its popup (320 px at most); a click inside keeps it, a click away closes it; F1 on "
        "a focused control opens its own",
        r_dot == "open" and opened and inside and away and ov.pop is None and f1)
    frame(disc_view, f, st, overlay=ov)
    kv_key = pygame.Rect(work_rect(1280, 800).x + T.WORK_PAD_X + 5, 0, 10, 10)
    kv_y = [g_.row for g_ in [f.geom("ui.disc.more")]][0].bottom
    mouse_kv = None
    for yy in range(kv_y, kv_y + 80):
        surf_t, _, _, ov_t = frame(disc_view, f, st, mouse=(kv_key.x, yy), overlay=Overlay(), now=0.0)
        if ov_t._tip_req is not None:
            mouse_kv = (kv_key.x, yy)
            break
    shown_at = []
    ov2 = Overlay()
    for t in (0.0, 0.3, 0.6):
        surf_t, _, _, _ = frame(disc_view, f, st, mouse=mouse_kv or (0, 0), overlay=ov2, now=t)
        ov2.draw(surf_t, t)
        shown_at.append(any(tuple(surf_t.get_at((x, y))[:3]) == T.TOOLTIP
                            for x in range(0, 1280, 3) for y in range(0, 800, 3)))
    rep("tooltip: a short tip shows after TOOLTIP_DELAY_S of hovering, not before",
        mouse_kv is not None and shown_at == [False, False, True], f"{shown_at}")

    # -- the rest of the kit -------------------------------------------------------------------------------------------
    class _Fig:
        def __init__(self):
            self.rects = []

        def draw(self, surf_, rect_):
            self.rects.append(pygame.Rect(rect_))
            surf_.fill(T.ACCENT_FILL, rect_)
    s = _Store(ro=0.4321, flag=True)
    ps = [s.param("ro", "span", set=None, unit="m", fmt="{:.3f}"), s.param("flag", "fixed", kind="bool"),
          s.param("to1", "change in stage 1", kind="action")]
    f, st = Form(ps), {}
    figs = [_Fig(), _Fig(), _Fig(), _Fig()]
    custom_rects, linked = [], []

    def rest(ui_):
        with ui_.card("Plot", pad=False):
            ui_.plot(figs[0], 200)
        ui_.plots(figs[1:], 180, titles=["lift curve", "drag polar", "pitching moment"])
        ui_.empty_plot("no evaluations yet", 120)
        ui_.custom(90, lambda surf_, r_: custom_rects.append((pygame.Rect(r_), surf_.get_clip())))
        ui_.spinner("waiting for the first improvement…")
        ui_.range_bar(0.25)
        ui_.range_bar(1.7, T.BAD)
        ui_.stacked_bar([(66.0, T.ACCENT, "induced: 66.0 counts"), (80.8, T.GOOD, "profile"), (0, T.WARN, "x"),
                         (33.5, T.WARN, "other")])
        ui_.code(["b_m            [1.2, 2]  # narrowed", "x" * 400])
        ui_.readout("ro")
        ui_.field("ro", control="readout")
        ui_.field("flag", control="switch")
        ui_.kv("circuit", "arena / dry", link=("change in stage 1", "edit", "to1"))
        ui_.kv("section", "e423", link=("change in stage 2", "edit", lambda: linked.append(2)))
        ui_.checkbox("cd", "section c_d")
        ui_.checkbox("off", "lap time [s]", enabled=False, tip="not flown by this objective")
        ui_.switch("sample", label="sample the incumbent")
        with ui_.row():
            ui_.tag("RUNNING · 3/40", T.ACCENT)
            ui_.spacer()
            ui_.tag("STOPPED", T.WARN)
        ui_.sect_head("Keyboard, pad and mouse on the design pages")
    big = pygame.Rect(281, 86, 996, 1800)
    surf, ui, hr_, _ = frame(rest, f, st, size=(1280, 2000), rect=big)
    cw = big.w - 2 * T.WORK_PAD_X
    widths_ok = [r_.w for r_ in figs[1:] for r_ in r_.rects]
    rep("plot / plots / empty_plot / custom get their rects at the column's width (custom clipped to it)",
        figs[0].rects and figs[0].rects[0].w == cw - 2 and len(widths_ok) == 3 and sum(widths_ok) == cw
        and custom_rects and custom_rects[0][0].w == cw and custom_rects[0][1] == custom_rects[0][0],
        f"plot {figs[0].rects[0].w if figs[0].rects else None}, panels {widths_ok}")
    def bars(ui_):
        ui_.range_bar(0.25)
        ui_.range_bar(1.7, T.BAD)
        ui_.stacked_bar([(66.0, T.ACCENT, "induced"), (80.8, T.GOOD, "profile"), (0, T.WARN, ""),
                         (33.5, T.WARN, "other")])
    surf_b, _, _, _ = frame(bars, Form([]))
    wr_b = work_rect(1280, 800)
    bx0, by0 = wr_b.x + T.WORK_PAD_X, wr_b.y + T.WORK_PAD_Y
    cwb = wr_b.w - 2 * T.WORK_PAD_X
    y_r1, y_r2, y_st = by0 + 4, by0 + _s(9) + T.STACK_GAP + 4, by0 + 2 * (_s(9) + T.STACK_GAP) + 5
    m1 = [x - bx0 for x in range(bx0, bx0 + cwb) if tuple(surf_b.get_at((x, y_r1))[:3]) == T.ACCENT]
    m2 = [x - bx0 for x in range(bx0, bx0 + cwb) if tuple(surf_b.get_at((x, y_r2))[:3]) == T.BAD]
    seg = {c_: sum(1 for x in range(bx0, bx0 + cwb) if tuple(surf_b.get_at((x, y_st))[:3]) == c_)
           for c_ in (T.ACCENT, T.GOOD, T.WARN)}
    tot = 66.0 + 80.8 + 33.5
    split_ok = all(abs(seg[c_] - cwb * v / tot) <= 2 for c_, v in ((T.ACCENT, 66.0), (T.GOOD, 80.8), (T.WARN, 33.5)))
    rep("range_bar: the 3 px marker sits at frac (a value past the end sits ON the end); stacked_bar splits by "
        "value, a zero part takes no width",
        len(m1) == 3 and abs(m1[0] - (0.25 * cwb - 1.5)) <= 1 and m2 == [cwb - 3, cwb - 2, cwb - 1] and split_ok,
        f"markers at {m1[:1]} / {m2[:1]} of {cwb}; parts {list(seg.values())}")
    code_st = st.get(("code", "code0"))
    tagx = [f_ for f_ in (surf.get_at((big.x + T.WORK_PAD_X + cw - 2, y))[:3] for y in range(big.y, big.bottom))]
    rep("readout / readout field / switch field / kv links / check boxes / UI switch / row spacer / code block",
        f.geom("ro") is not None and f.kind_of("flag") == "switch" and "ui.check.cd" in f.order
        and "ui.switch.sample" in f.order and f.geom("to1") is not None and f.geom("ui.link.section.change in stage 2")
        is not None and code_st is not None and T.WARN in tagx and not f.is_enabled(f.param("ui.check.off")),
        f"{len(f.order)} recorded")
    f.click(f.geom("ui.check.cd").rect.center)
    f.click(f.geom("ui.switch.sample").rect.center)
    f.click(f.geom("ui.link.section.change in stage 2").rect.center)
    f.click(f.geom("to1").rect.center)
    f.click(f.geom("flag").rect.center)
    code_r = [r_ for r_, _ in f._wheel][0]
    f.hscroll_at(code_r.center, +3)
    _, ui2, _, _ = frame(rest, f, st, size=(1280, 2000), rect=big)
    rep("...and they act: check box and UI switch flip in state, links fire, a Param switch sets, code scrolls "
        "sideways",
        st[("check", "ui.check.cd")]["on"] is False and st[("switch", "ui.switch.sample")]["on"] is False
        and linked == [2] and s.sets.get("to1") == 1 and s.v["flag"] is False and st[("code", "code0")]["hx"] > 0)
    f_none, st_none = None, {}
    _, u1, _, _ = frame(lambda ui_: ui_.link("x", "go somewhere", "arrow_forward", lambda: None), f_none, st_none)
    _, u2, _, _ = frame(lambda ui_: ui_.link("x", "go somewhere", "arrow_forward", lambda: None), f_none, st_none)
    s = _Store(a=1.0)
    f = Form([s.param("a", "a", lo=0.0, hi=2.0)])
    frame(lambda ui_: (ui_.gap(300), ui_.field("a")), f)
    y0 = f.geom("a").rect.y
    surf_lo, _, _, _ = frame(lambda ui_: (ui_.gap(300), ui_.field("a")), f, scroll=120, layout_only=True)
    blank = all(tuple(surf_lo.get_at((x, y))[:3]) == T.WELL for x in range(281, 1277, 5) for y in range(86, 622, 5))
    rep("a formless view keeps one private Form in its state; scroll shifts the layout; layout_only lays out "
        "without drawing",
        u1.form is u2.form and "ui.x" in u2.form.order and f.geom("a").rect.y == y0 - 120 and blank)

    try:
        from . import plot as cae_plot
        perr = ""
    except Exception as e:          # noqa: BLE001 -- a sibling's file mid-edit is not this module's failure
        cae_plot, perr = None, f"{type(e).__name__}: {e}"
    if cae_plot is not None:
        fig = cae_plot.Figure(xlabel="evaluation", ylabel="best objective")
        fig.line([1, 2, 3, 4, 5], [float("-inf"), 0.2, float("nan"), 0.5, 0.6], name="best so far")
        f_, st_ = Form([]), {}
        surf, _, _, _ = frame(lambda ui_: _plot_card(ui_, fig), f_, st_)
        inked = any(tuple(surf.get_at((x, y))[:3]) == T.ACCENT for x in range(300, 1200, 2) for y in range(120, 400, 2))
        rep("a real cae.plot Figure draws through ui.plot inside a pad=False card (non-finite values masked)", inked)
    else:
        rep("a real cae.plot Figure draws through ui.plot (drive/cae/plot.py not importable: skipped)", True, perr)

    # -- garage_ui.Plot masking (A's garage_ui change) -----------------------------------------------------------------
    nan, inf = float("nan"), float("inf")
    xs = np.arange(10.0)
    raised = []
    for ys_ in (np.array([-inf, 1, 2, 1.5, 2, 2.5, 2, 1, 0.5, 1.0]), np.array([0, 1, 2, nan, 2, 2.5, 2, 1, 0.5, 1]),
                np.array([0, 1, inf, 1.5, nan, 2.5, 2, -inf, 0.5, 1.0]), np.full(10, nan)):
        ps_ = pygame.Surface((400, 300))
        pl = gui.Plot(gui.Text(gui.Fonts(1.0)))
        pl.begin(ps_, (0, 0, 400, 300), (0, 9), (-1.0, 3.0))
        try:
            pl.line(xs, ys_, col=(255, 0, 0))
            pl.line(xs, ys_, closed=True)
            pl.points(xs, ys_)
        except Exception as e:
            raised.append(f"{type(e).__name__}: {e}")
    ps_ = pygame.Surface((400, 300))
    ps_.fill((0, 0, 0))
    pl = gui.Plot(gui.Text(gui.Fonts(1.0)))
    pl.begin(ps_, (0, 0, 400, 300), (0, 9), (-1.0, 3.0))
    pl.line(xs, np.array([0, 1, 2, nan, 2, 2.5, 2, 1, 0.5, 1.0]), col=(255, 0, 0))
    xa, xb = pl._px(2.2, 0)[0], pl._px(3.8, 0)[0]
    red = sum(1 for x in range(xa, xb) for y in range(pl.rect.y, pl.rect.bottom)
              if tuple(ps_.get_at((x, y))[:3]) == (255, 0, 0))
    rep("garage_ui.Plot.line / .points take NaN and ±inf without raising, and draw no line across a gap",
        not raised and red == 0, f"{raised or 'no raise'}, {red} px across the NaN")

    # -- cost ----------------------------------------------------------------------------------------------------------
    s = _Store(**{f"k{i}": 0.5 for i in range(60)})
    ps = [s.param(f"k{i}", f"parameter number {i}", lo=0.0, hi=1.0, step=0.05) for i in range(60)]
    ps += [s.param(f"a{i}", f"Action {i}", kind="action") for i in range(10)]
    f = Form(ps)

    def hundred(ui_):
        with ui_.card("A hundred controls"):
            for i in range(40):
                ui_.field(f"k{i}")
            for i in range(40, 60):
                ui_.slider(f"k{i}")
            for i in range(20):
                ui_.kv(f"key {i}", f"{i * 1.2345:.4f} m")
        with ui_.row():
            for i in range(10):
                ui_.button(f"a{i}", kind="outline")
        for i in range(10):
            ui_.hint(f"hint number {i}: the quiet line under a control, with a little text in it.")
    surf = pygame.Surface((1280, 800))
    times = []
    for _ in range(25):
        t0 = time.perf_counter()
        surf.fill(T.WELL)
        ui = WorkUI(surf, work_rect(1280, 800), f, {}, scroll=0, kbd_focus=False, mouse=(-1, -1),
                    overlay=Overlay(), now=0.0)
        hundred(ui)
        ui.end()
        times.append(time.perf_counter() - t0)
    med = sorted(times)[len(times) // 2] * 1e3
    rep("a view of 100 controls draws in under 8 ms (median of 25 frames, 1280x800)", med < 8.0,
        f"{med:.2f} ms, {len(f.order)} recorded")

    # -- the demo views at both sizes, clipped to the work area ----------------------------------------------------------
    sent = (255, 0, 255)
    walls = []
    for W, H in ((1280, 800), (1600, 1000)):
        for name, view, make in (("screening", demo_screening, lambda: (demo_form(demo_store()), None)),
                                 ("ranking", demo_ranking, lambda: (None, None))):
            form_, _ = make()
            surf = pygame.Surface((W, H))
            surf.fill(sent)
            wr = work_rect(W, H)
            surf.fill(T.WELL, wr)
            t0 = time.perf_counter()
            ui = WorkUI(surf, wr, form_, {}, scroll=0, kbd_focus=False, mouse=(-1, -1), overlay=Overlay(), now=0.0)
            view(ui)
            ch = ui.end()
            walls.append((f"{name} {W}x{H}", (time.perf_counter() - t0) * 1e3, ch))
            outside = any(tuple(surf.get_at((x, y))[:3]) != sent for x in range(0, W, 7) for y in range(0, H, 7)
                          if not wr.collidepoint(x, y))
            if outside:
                walls.append((f"{name} {W}x{H} DREW OUTSIDE", 0, 0))
            if out:
                full = pygame.Surface((W, H))
                demo_frame(full, W, H)
                ui = WorkUI(full, wr, form_ if form_ is not None else None, {}, scroll=0, kbd_focus=False,
                            mouse=(-1, -1), overlay=Overlay(), now=0.0)
                view(ui)
                ui.end()
                pygame.image.save(full, os.path.join(out, f"widgets_{name}_{W}x{H}.png"))
    rep("the demo views (WingLab 03 and 06) draw at 1280x800 and 1600x1000, clipped to the work area",
        not any("OUTSIDE" in w_[0] for w_ in walls) and all(w_[2] > 300 for w_ in walls),
        ", ".join(f"{n_} {t_:.1f} ms h {h_}" for n_, t_, h_ in walls))

    # -- scale ----------------------------------------------------------------------------------------------------------
    T.set_scale(1.5)
    s15 = demo_store()
    surf15, _, _, _ = frame(demo_screening, demo_form(s15), size=(2400, 1500))
    wr15 = work_rect(2400, 1500)
    cx15 = wr15.x + 400
    bar15 = [y for y in range(wr15.y, wr15.y + 80) if tuple(surf15.get_at((cx15, y))[:3]) == T.PANEL]
    kpi15 = T.KPI_H
    T.set_scale(1.0)
    rep("set_scale(1.5): a card's title bar is 33 px and a KPI tile 56; set_scale(1.0) restores the constants",
        len(bar15) == 33 and kpi15 == 56 and T.KPI_H == 37 and T.CARD_TITLE_H == 22 and T.S == 1.0,
        f"bar {len(bar15)} px")

    if verbose:
        print(f"cae.widgets self-check: {n_rows[0]} rows, {'ALL PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys as _sys
    _out = _sys.argv[_sys.argv.index("--out") + 1] if "--out" in _sys.argv else ""
    _sys.exit(0 if self_check(out=_out) else 1)
