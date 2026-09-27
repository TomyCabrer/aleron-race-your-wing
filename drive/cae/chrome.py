"""drive/cae/chrome.py -- the AeroBO shell's chrome, the frame every design
view sits in: pane frames, the menu bar and its drop-downs, the tool bar
(buttons, crumbs, chips), the Simulation tree, the Properties grid, the tab
strip, the Output log, the status bar, toasts, dialogs and scrollbars.

Every widget is DRAWN from state it is handed each frame and keeps only its
own small UI state (which menu is open, the log's lines and scroll, the
toasts' clocks); design_shell.py owns what the rows, tabs and cells say.
The geometry is AeroBO v3's, measured on its screenshots at 1600x1000
(SPECS/aerobo_shell.md §1-§8): a 21 px tree pitch from the body's top + 3,
indents of 4 + 13 per level, a 44 / 56 Properties split, 22 px tabs sitting
on the strip's rule, the log's 17 px lines inset 20 / 16, status cells at
most 34 characters wide. Text is placed where the browser puts it, not
centred by formula: each widget's text offsets were read off the same
shots (the cap tops of "Simulation", "1 Mission", "Ready", ...).

Hover tooltips (a tree row's reason, an ellipsised property, a tool button)
go to the shell's `widgets.Overlay` through its `tooltip(anchor_rect,
text)`: a widget built with `overlay=` calls it for the hovered item, and
every widget also leaves that request in `.tip` = (Rect, text) or None, for
a caller without an overlay (and for the self-check). Drop-down menus and
dialogs draw themselves (they carry icons, key hints and disabled items the
generic drop-down does not); call their draw last, over everything else.

Sizes are read as `T.NAME` when drawing, so `T.set_scale` reaches them; the
few offsets that are not theme constants go through `_px`. Mouse: every
click method is for button 1 only -- the shell filters (PLAN §0.2).
Pure pygame + numpy.
"""

from __future__ import annotations

import math

import pygame

from . import theme as T

#: Quasar's modal backdrop, rgba(0, 0, 0, .4)
_BACKDROP = (0, 0, 0, 102)
#: QScrollArea's thumb (the Output log's): black at .2, 10 px, no track
_THIN_THUMB = (0, 0, 0, 51)
_THIN_W = 10
#: Quasar's disabled look for a menu item: opacity .6
_MENU_OFF = 0.6
#: a disabled tool button's opacity. AeroBO's stylesheet asks for .38, but
#: Quasar's own `.q-btn.disabled{opacity:.7!important}` wins: the greyed Stop
#: in shot 03 is (115, 116, 117), grey-9 at .7 over the bar
_TOOL_OFF = 0.7
#: Quasar's hover on a flat button: its own colour at .15 over the bar
_FLAT_HOVER = 0.15
#: menu drop-down icon px (Quasar's q-item icon; not measurable in any shot)
_MENU_ICON = 18
#: the app name's letter-spacing, em (AeroBO theme.py .cae-appname)
_APP_TRACK = 0.02
#: a status cell holds at most this many characters (CSS max-width 34ch)
_CELL_CH = 34
#: the toast types: fill, text colour, icon (Quasar Notify, SH §6.14)
_TOAST = {"positive": (T.GOOD, T.WELL, "check_circle"), "negative": (T.BAD, T.WELL, "warning"),
          "warning": (T.WARN, T.PANEL, "warning"), "info": (T.TOAST_INFO, T.WELL, "info")}
_TOAST_ALIAS = {"ok": "positive", "good": "positive", "success": "positive", "error": "negative",
                "bad": "negative", "warn": "warning"}
#: log level -> message colour name in theme (SH §7.1)
_LEVEL = {"info": "INK", "ok": "GOOD", "warn": "WARN", "error": "BAD"}
#: status square colour name by kind (SH §8.1)
_SQUARE = {"idle": "INK_FAINT", "busy": "ACCENT", "ok": "GOOD", "warn": "WARN", "error": "BAD"}

_shadows: dict = {}             # (w, h, S) -> the drop-down shadow surface


def _px(v: float) -> int:
    """An offset that is not a theme constant, at the current scale."""
    return int(math.floor(v * T.S + 0.5))


def _sans_y(box_y: int, box_h: int, css: float, bold: bool = False) -> int:
    """Top of a SANS line centred in a box, as the browser places it."""
    return box_y + (box_h - T.font("sans", css, bold).get_height()) // 2


def _mono_y(box_y: int, box_h: int, css: float) -> int:
    """Top of a MONO line centred in a box: SF Mono sits 1 px higher in
    the browser's line box than a plain centre (prop values, log, cells)."""
    return box_y + (box_h - T.font("mono", css).get_height()) // 2 - 1


def _hit(rect, pos) -> bool:
    return rect is not None and pos is not None and rect.collidepoint(pos)


def _tip(widget, rect, text) -> None:
    """Record a hover tooltip request, and hand it to the overlay if any."""
    if not text:
        return
    widget.tip = (pygame.Rect(rect), str(text))
    ov = getattr(widget, "overlay", None)
    if ov is not None:
        ov.tooltip(pygame.Rect(rect), str(text))


def _shadow(surf, rect) -> None:
    """AeroBO's drop-down / popup shadow, `0 6px 18px rgba(20,26,34,.18)`,
    as a blurred copy of the box (cached per size)."""
    r = pygame.Rect(rect)
    blur = _px(18)
    key = (r.w, r.h, T.S)
    img = _shadows.get(key)
    if img is None:
        if len(_shadows) > 32:
            _shadows.clear()
        w, h = r.w + 2 * blur, r.h + 2 * blur
        base = pygame.Surface((w, h), pygame.SRCALPHA)
        base.fill(T.SHADOW_A, pygame.Rect(blur, blur, r.w, r.h))
        k = max(1, blur // 3)
        small = pygame.transform.smoothscale(base, (max(1, w // k), max(1, h // k)))
        img = pygame.transform.smoothscale(small, (w, h))
        _shadows[key] = img
    surf.blit(img, (r.x - blur, r.y - blur + _px(6)))


def tag(surf, x: int, y: int, text: str, colour, anchor: str = "left") -> pygame.Rect:
    """An AeroBO tag / chip (SH §6.6): SANS 10/600 in `colour`, a 1 px
    border in the same colour over a 10 % tint, 17 px tall, padding 0 5.
    `anchor` "right" puts its right edge at x. Returns its rect."""
    w = T.text_w(text, "sans", 10, True) + 2 * _px(5) + 2
    r = pygame.Rect(x - w if anchor == "right" else x, y, w, T.TAG_H)
    pygame.draw.rect(surf, T.tint(colour), r, border_radius=2)
    pygame.draw.rect(surf, colour, r, 1, border_radius=2)
    T.text(surf, text, r.x + 1 + _px(5), r.y + (r.h - T.font("sans", 10, True).get_height()) // 2,
           "sans", 10, colour, True)
    return r


# -- scrolling ---------------------------------------------------------------------------------------
def clamp_scroll(scroll, content_h, view_h) -> int:
    """`scroll` kept inside [0, content_h - view_h]."""
    return int(max(0, min(int(scroll), int(content_h) - int(view_h))))


def _thumb(rect: pygame.Rect, content_h: int, scroll: int, min_h: int):
    if content_h <= rect.h or rect.h <= 0:
        return None
    th = max(min_h, int(rect.h * rect.h / content_h))
    span = content_h - rect.h
    y = rect.y + int(round((rect.h - th) * max(0, min(scroll, span)) / span))
    return pygame.Rect(rect.x, y, rect.w, th)


def scrollbar(surf, body, content_h, scroll, *, mouse=None, thin=False):
    """The vertical scrollbar at the right edge of a scrolling pane body,
    drawn only when the content overflows. Returns (track, thumb) or None.

    Default: the stylesheet's 12 px bar (track SCROLL_TRACK, thumb
    SCROLL_THUMB -- SCROLL_THUMB_HOVER under the mouse -- inset 3 px,
    radius 6). `thin`: QScrollArea's thumb the Output log shows in shot 03,
    10 px of black at .2 with no track."""
    b = pygame.Rect(body)
    w = _px(_THIN_W) if thin else T.SCROLL_W
    track = pygame.Rect(b.right - w, b.y, w, b.h)
    thumb = _thumb(track, int(content_h), int(scroll), _px(10) if thin else _px(24))
    if thumb is None:
        return None
    if thin:
        T.blend_rgba(surf, thumb, _THIN_THUMB)
        return track, thumb
    surf.fill(T.SCROLL_TRACK, track)
    inner = thumb.inflate(-2 * _px(3), -2 * _px(3))
    col = T.SCROLL_THUMB_HOVER if _hit(thumb, mouse) else T.SCROLL_THUMB
    pygame.draw.rect(surf, col, inner, border_radius=max(1, inner.w // 2))
    return track, thumb


class ScrollDrag:
    """A scrollbar thumb being dragged: `begin` on mouse-down (True when
    the press is on the thumb), `drag` on motion returns the new scroll,
    `end` on mouse-up. A press on the track pages instead (`page`)."""

    def __init__(self):
        self.grab = None                # px from the thumb's top to the press

    def begin(self, pos, bar) -> bool:
        if bar is None or not bar[1].collidepoint(pos):
            return False
        self.grab = pos[1] - bar[1].y
        return True

    def drag(self, pos, bar, content_h) -> int:
        track, thumb = bar
        free = max(1, track.h - thumb.h)
        frac = (pos[1] - self.grab - track.y) / free
        return clamp_scroll(frac * (content_h - track.h), content_h, track.h)

    @staticmethod
    def page(pos, bar, scroll, content_h) -> int:
        track, thumb = bar
        step = track.h - _px(40)
        return clamp_scroll(scroll - step if pos[1] < thumb.y else scroll + step, content_h, track.h)

    def end(self) -> None:
        self.grab = None

    @property
    def active(self) -> bool:
        return self.grab is not None


# -- pane ----------------------------------------------------------------------------------------------
def pane(surf, rect, title=None, *, white=False, tool=None, mouse=None) -> pygame.Rect:
    """A framed pane (SH §6.1): 1 px RULE frame, radius 2, a 22 px title bar
    (HEADER, its own RULE bottom included) with the title SANS 11/600
    INK_MUTED at x + 8, an optional flat ACCENT icon tool at its right
    (`tool` = an icon name, e.g. "clear_all"; hit rect: `pane_tool_rect`),
    and a PANEL body (WELL when `white`). Returns the body rect."""
    r = pygame.Rect(rect)
    surf.fill(T.WELL if white else T.PANEL, r.inflate(-2, -2))
    body = r.inflate(-2, -2)
    if title is not None or tool:
        bar = pygame.Rect(r.x + 1, r.y + 1, r.w - 2, T.PANE_TITLE_H - 1)
        surf.fill(T.HEADER, bar)
        pygame.draw.line(surf, T.RULE, (r.x + 1, r.y + T.PANE_TITLE_H), (r.right - 2, r.y + T.PANE_TITLE_H))
        if title:
            T.text(surf, title, r.x + _px(8), _sans_y(bar.y, bar.h, 11, True), "sans", 11, T.INK_MUTED, True,
                   clip_w=r.w - _px(16) - (_px(31) if tool else 0))
        if tool:
            tr = pane_tool_rect(r)
            if _hit(tr, mouse):
                surf.fill(T.fade(T.ACCENT, _FLAT_HOVER, T.HEADER), tr)
            T.icon(surf, tool, tr.centerx, bar.y + (bar.h - T.TOOL_ICON) // 2, T.TOOL_ICON, T.ACCENT, "centre")
        body = pygame.Rect(r.x + 1, r.y + T.PANE_TITLE_H + 1, r.w - 2, r.h - T.PANE_TITLE_H - 2)
    pygame.draw.rect(surf, T.RULE, r, 1, border_radius=2)
    return body


def pane_tool_rect(rect) -> pygame.Rect:
    """The hit rect of a pane's title-bar tool (a 31 px tool button 7 px in
    from the right edge, clipped to the title bar)."""
    r = pygame.Rect(rect)
    w = 2 + 2 * _px(6) + T.TOOL_ICON
    return pygame.Rect(r.right - 1 - _px(7) - w, r.y + 1, w, T.PANE_TITLE_H - 1)


# -- menu bar ------------------------------------------------------------------------------------------
class MenuBar:
    """The menu bar (SH §5.1): the app name, then the menu titles; a click
    opens a title's drop-down (WELL, 1 px RULE, radius 2, the drop-down
    shadow; 26 px items: icon INK_MUTED, 6 px, label SANS 12, and the key
    hint right-aligned MONO 11 INK_FAINT -- a carsim addition).

    `menus` = [(title, [(label, icon, action_id, key_hint) | None, ...])],
    None a separator. `open` is the open menu's title (None = closed);
    `cursor` its highlighted item for the keyboard. `draw` paints the bar,
    `draw_menu` the open drop-down (call it after everything else); while
    `open` is set, route every click and key here first."""

    def __init__(self, app_name, menus):
        self.app_name = str(app_name)
        self.menus = [(str(t), list(items)) for t, items in menus]
        self.open = None
        self.cursor = -1
        self._titles: list = []         # [(title, Rect)] from the last draw
        self._enabled = lambda action_id: True
        self._bar = pygame.Rect(0, 0, 0, 0)

    # -- layout
    def _items(self, title):
        for t, items in self.menus:
            if t == title:
                return items
        return []

    def _menu_layout(self):
        """(box, [(item or None, Rect)]) of the open drop-down."""
        if self.open is None:
            return None, []
        anchor = next((r for t, r in self._titles if t == self.open), None)
        if anchor is None:
            return None, []
        items = self._items(self.open)
        pad, icon_w, gap = _px(12), _px(_MENU_ICON), _px(6)
        lw = max((T.text_w(it[0], "sans", 12) for it in items if it), default=0)
        hw = max((T.text_w(it[3], "mono", 11) for it in items if it and len(it) > 3 and it[3]), default=0)
        w = max(_px(160), pad + icon_w + gap + lw + (_px(24) + hw if hw else 0) + pad) + 2
        rows, y = [], self._bar.bottom + 1
        for it in items:
            h = _px(26) if it else _px(9)
            rows.append((it, pygame.Rect(anchor.x + 1, y, w - 2, h)))
            y += h
        box = pygame.Rect(anchor.x, self._bar.bottom, w, y - self._bar.bottom + 1)
        return box, rows

    def _selectable(self):
        """Indexes of the open menu's items that the keyboard may land on."""
        return [i for i, it in enumerate(self._items(self.open)) if it and self._enabled(it[2])]

    # -- drawing
    def draw(self, surf, rect, mouse, enabled) -> None:
        r = pygame.Rect(rect)
        self._bar = r
        self._enabled = enabled or (lambda action_id: True)
        surf.fill(T.CHROME, (r.x, r.y, r.w, r.h - 1))
        pygame.draw.line(surf, T.RULE, (r.x, r.bottom - 1), (r.right - 1, r.bottom - 1))
        body_h = r.h - 1
        x = r.x + _px(4) + _px(6)
        w_app = T.text(surf, self.app_name, x, _sans_y(r.y, body_h, 11, True), "sans", 11, T.INK_MUTED, True)
        # the name is tracked .02em (AeroBO's .cae-appname): its box is that much wider than its glyphs
        x += w_app + _px(_APP_TRACK * 11 * len(self.app_name)) + _px(10) + 1
        self._titles = []
        th = T.font("sans", 12).get_height() + 2 * _px(2) + 1
        for title, _items in self.menus:
            w = 2 * _px(9) + T.text_w(title, "sans", 12)
            box = pygame.Rect(x, r.y + (body_h - th + 1) // 2, w, th)
            self._titles.append((title, box))
            if self.open is not None and self.open != title and _hit(box, mouse):
                self.open, self.cursor = title, -1          # sliding along an open menu bar
            if self.open == title or _hit(box, mouse):
                pygame.draw.rect(surf, T.ACCENT_FILL, box, border_radius=2)
            T.text(surf, title, box.x + _px(9), _sans_y(r.y, body_h, 12), "sans", 12, T.INK)
            x += w + 1

    def draw_menu(self, surf, mouse) -> None:
        """The open drop-down, if any (draw it over everything else)."""
        box, rows = self._menu_layout()
        if box is None:
            return
        _shadow(surf, box)
        surf.fill(T.WELL, box)
        pygame.draw.rect(surf, T.RULE, box, 1, border_radius=2)
        pad = _px(12)
        for i, (it, rr) in enumerate(rows):
            if it is None:
                pygame.draw.line(surf, T.RULE_SOFT, (rr.x + pad, rr.centery), (rr.right - pad, rr.centery))
                continue
            label, icon, action = it[0], it[1], it[2]
            hint = it[3] if len(it) > 3 else None
            on = self._enabled(action)
            if on and (_hit(rr, mouse) or i == self.cursor):
                surf.fill(T.ACCENT_FILL, rr)
            ink = T.INK if on else T.fade(T.INK, _MENU_OFF)
            muted = T.INK_MUTED if on else T.fade(T.INK_MUTED, _MENU_OFF)
            if icon:
                T.icon(surf, icon, rr.x + pad, rr.y + (rr.h - _px(_MENU_ICON)) // 2, _px(_MENU_ICON), muted)
            T.text(surf, label, rr.x + pad + _px(_MENU_ICON) + _px(6), _sans_y(rr.y, rr.h, 12), "sans", 12, ink)
            if hint:
                T.text(surf, hint, rr.right - pad, _mono_y(rr.y, rr.h, 11), "mono", 11,
                       T.INK_FAINT if on else T.fade(T.INK_FAINT, _MENU_OFF), anchor="right")

    # -- input
    def close(self) -> None:
        self.open, self.cursor = None, -1

    def click(self, pos) -> str | None:
        """A left click: on a title it opens / closes that menu (None); on an
        enabled item it closes the menu and returns the item's action_id;
        anywhere else it closes an open menu (None, consumed)."""
        for title, box in self._titles:
            if box.collidepoint(pos):
                if self.open == title:
                    self.close()
                else:
                    self.open, self.cursor = title, -1
                return None
        if self.open is None:
            return None
        box, rows = self._menu_layout()
        for it, rr in rows:
            if it and rr.collidepoint(pos):
                if not self._enabled(it[2]):
                    return None                     # a disabled item: nothing, the menu stays open
                self.close()
                return it[2]
        self.close()
        return None

    def key(self, ev) -> str | None:
        """While a menu is open: UP / DOWN move the highlight over enabled
        items, LEFT / RIGHT switch menus, ENTER picks (returns its
        action_id), ESC closes. Returns None for every key but a pick."""
        if self.open is None or ev.type != pygame.KEYDOWN:
            return None
        k = ev.key
        if k == pygame.K_ESCAPE:
            self.close()
            return None
        if k in (pygame.K_LEFT, pygame.K_RIGHT):
            names = [t for t, _ in self.menus]
            i = names.index(self.open) + (1 if k == pygame.K_RIGHT else -1)
            self.open, self.cursor = names[i % len(names)], -1
            return None
        sel = self._selectable()
        if k in (pygame.K_UP, pygame.K_DOWN) and sel:
            if self.cursor not in sel:
                self.cursor = sel[0] if k == pygame.K_DOWN else sel[-1]
            else:
                j = sel.index(self.cursor) + (1 if k == pygame.K_DOWN else -1)
                self.cursor = sel[j % len(sel)]
            return None
        if k in (pygame.K_RETURN, pygame.K_KP_ENTER) and self.cursor in sel:
            action = self._items(self.open)[self.cursor][2]
            self.close()
            return action
        return None


# -- tool bar ------------------------------------------------------------------------------------------
class ToolBar:
    """The tool bar (SH §5.2): 32 px of CHROME with its RULE bottom, padding
    0 6, items 3 px apart and vertically centred.

    `items`, in order:
        ("icon", id, icon, tip)                 a flat icon button, 31 x 24
        ("primary", id, icon, label, tip)       ACCENT, PRIMARY_EDGE border, white (Run)
        ("flat", id, icon, label, tip)          icon + label, grey-9 (Stop)
        ("sep",)                                1 x 18 RULE, margin 0 4
        ("crumbs",)                             the breadcrumb
        ("space",)                              flexible space
        ("chips",)                              the right-aligned tags
    Buttons are 24 tall, padding 0 6, a 17 px icon and a 6 px gap to the
    label (SANS 11.5, weight 500 = regular); a disabled one is drawn at
    opacity .7 (measured, see _TOOL_OFF); hover: a RULE border and the
    button's colour at .15 (the primary: PRIMARY_HOVER). After `draw`,
    `rects` maps every drawn button id (and "crumb:<stage>") to its rect --
    the tutorial's `('tool', id)` anchors."""

    def __init__(self, items, overlay=None):
        self.items = list(items)
        self.overlay = overlay
        self.rects: dict = {}
        self.tip = None
        self._hits: list = []           # [(Rect, id, enabled)]

    def _button_w(self, item) -> int:
        w = 2 + 2 * _px(6) + T.TOOL_ICON
        if item[0] in ("primary", "flat") and item[3]:
            w += _px(6) + T.text_w(item[3], "sans", 11.5)
        return w

    def _crumbs_w(self, crumbs) -> int:
        gap = _px(3)
        w = 0
        for i, (stage, label) in enumerate(crumbs):
            if i:
                w += gap + T.text_w("▸", "sans", 11) + gap
            w += T.text_w(label, "sans", 11)
        return w

    def draw(self, surf, rect, *, enabled, crumbs=(), crumb_cur=None, chips=(), mouse=None, hidden=()) -> None:
        """Paint the bar. `enabled(id) -> bool` per button; `crumbs` =
        [(stage, label)] with `crumb_cur` the current stage (INK / 600);
        `chips` = [(text, colour)]; ids in `hidden` are not drawn (a
        separator left next to another, or at an end, is dropped too)."""
        r = pygame.Rect(rect)
        surf.fill(T.CHROME, (r.x, r.y, r.w, r.h - 1))
        pygame.draw.line(surf, T.RULE, (r.x, r.bottom - 1), (r.right - 1, r.bottom - 1))
        body = pygame.Rect(r.x, r.y, r.w, r.h - 1)
        enabled = enabled or (lambda _id: True)
        self.rects, self._hits, self.tip = {}, [], None
        items = [it for it in self.items if not (len(it) > 1 and it[0] != "sep" and it[1] in hidden)]
        cleaned = []
        for it in items:                                 # no doubled / leading separators
            if it[0] == "sep" and (not cleaned or cleaned[-1][0] == "sep"):
                continue
            cleaned.append(it)
        while cleaned and cleaned[-1][0] == "sep":
            cleaned.pop()
        # the width everything but the space takes, so the space can absorb the rest
        gap = _px(3)
        chips_w = sum(T.text_w(t, "sans", 10, True) + 2 * _px(5) + 2 for t, _c in chips) + \
            _px(6) * max(0, len(chips) - 1)
        fixed = 0
        for it in cleaned:
            kind = it[0]
            if kind == "sep":
                fixed += 2 * _px(4) + 1
            elif kind == "crumbs":
                fixed += self._crumbs_w(crumbs)
            elif kind == "chips":
                fixed += chips_w
            elif kind != "space":
                fixed += self._button_w(it)
        spare = max(0, r.w - 2 * _px(6) - fixed - gap * max(0, len(cleaned) - 1))
        x = r.x + _px(6)
        by = body.y + (body.h - T.TOOL_BTN_H + 1) // 2
        for n, it in enumerate(cleaned):
            if n:
                x += gap
            kind = it[0]
            if kind == "sep":
                sh = _px(18)
                pygame.draw.line(surf, T.RULE, (x + _px(4), body.y + (body.h - sh + 1) // 2),
                                 (x + _px(4), body.y + (body.h - sh + 1) // 2 + sh - 1))
                x += 2 * _px(4) + 1
            elif kind == "space":
                x += spare
            elif kind == "crumbs":
                x = self._draw_crumbs(surf, x, body, crumbs, crumb_cur)
            elif kind == "chips":
                cx = r.right - _px(6) - chips_w
                cy = body.y + (body.h - T.TAG_H + 1) // 2
                for text, colour in chips:
                    cx = tag(surf, cx, cy, text, colour).right + _px(6)
                x += chips_w
            else:
                w = self._button_w(it)
                self._draw_button(surf, pygame.Rect(x, by, w, T.TOOL_BTN_H), it, bool(enabled(it[1])), mouse)
                x += w

    def _draw_button(self, surf, b: pygame.Rect, it, on: bool, mouse) -> None:
        kind, bid, icon = it[0], it[1], it[2]
        label = it[3] if kind in ("primary", "flat") else ""
        tip = it[-1] if len(it) > (4 if kind != "icon" else 3) else ""
        self.rects[bid] = b
        self._hits.append((b, bid, on))
        hot = on and _hit(b, mouse)
        if kind == "primary":
            fill = T.PRIMARY_HOVER if hot else T.ACCENT
            edge, ink = T.PRIMARY_EDGE, T.WELL
            if not on:
                fill, edge, ink = (T.fade(fill, _TOOL_OFF, T.CHROME), T.fade(edge, _TOOL_OFF, T.CHROME),
                                   T.fade(T.WELL, _TOOL_OFF, T.CHROME))
            pygame.draw.rect(surf, fill, b, border_radius=2)
            pygame.draw.rect(surf, edge, b, 1, border_radius=2)
        else:
            ink = T.STOP_GREY if on else T.fade(T.STOP_GREY, _TOOL_OFF, T.CHROME)
            if hot:
                pygame.draw.rect(surf, T.fade(T.STOP_GREY, _FLAT_HOVER, T.CHROME), b, border_radius=2)
                pygame.draw.rect(surf, T.RULE, b, 1, border_radius=2)
        ix = b.x + 1 + _px(6)
        T.icon(surf, icon, ix, b.y + (b.h - T.TOOL_ICON) // 2, T.TOOL_ICON, ink)
        if label:
            T.text(surf, label, ix + T.TOOL_ICON + _px(6), _sans_y(b.y, b.h, 11.5), "sans", 11.5, ink)
        if tip and _hit(b, mouse):
            _tip(self, b, tip)

    def _draw_crumbs(self, surf, x, body, crumbs, cur) -> int:
        gap = _px(3)
        y = _sans_y(body.y, body.h, 11)
        for i, (stage, label) in enumerate(crumbs):
            if i:
                x += gap
                x += T.text(surf, "▸", x, y, "sans", 11, T.INK_MUTED) + gap
            here = stage == cur
            w = T.text(surf, label, x, y, "sans", 11, T.INK if here else T.INK_MUTED, here)
            r = pygame.Rect(x, body.y + _px(4), w, body.h - 2 * _px(4))
            self.rects["crumb:" + str(stage)] = r
            self._hits.append((r, "crumb:" + str(stage), True))
            x += w
        return x

    def click(self, pos) -> str | None:
        """The id of the enabled button (or "crumb:<stage>") under `pos`."""
        for r, bid, on in self._hits:
            if r.collidepoint(pos):
                return bid if on else None
        return None


# -- the Simulation tree ---------------------------------------------------------------------------------
class TreeDraw:
    """The Simulation tree (SH §3.2), drawn from a flat list of rows:

        TreeRow = dict(id, level, label, icon, icon_colour, chip, selected,
                       faint, twisty (None | "open" | "closed"), tip,
                       focus (optional: the keyboard ring))

    Rows are 21 px from the body's top + 3; the content starts at
    4 + 13 x level: a 12 px twisty box ("▾" / "▸", SANS 9 INK_MUTED), 5 px,
    the 14 px icon, 5 px, the label (SANS 12; 600 when selected; INK_FAINT
    when faint), and the chip (MONO 9.5 INK_FAINT) right-aligned 6 px inside
    the right edge. AeroBO's row is a nowrap flex box (gap 5, the badge's
    margin-left auto + padding-left 8), so when label and chip do not fit,
    the chip sits 13 px after the label and the twisty box is the one item
    that gives way, down to its glyph: the icon and label slide left (shot
    03's "2.8 Endplate" row slides 8 px, not SH §3.2's "up to 3"). What
    still does not fit is cut at the edge ("CST section (optimised) · t/c
    0." in shot 12). Fills: selected ACCENT_FILL, hover TREE_HOVER (hover
    wins, as AeroBO's CSS has it)."""

    def __init__(self, overlay=None):
        self.overlay = overlay
        self.tip = None

    @staticmethod
    def content_h(rows) -> int:
        """The tree's scrollable height: the rows plus 3 px above and below."""
        return 2 * _px(3) + T.TREE_ROW_H * len(rows)

    @staticmethod
    def _twisty_w(glyph: str, x: int, right: int, label_w: int, chip_w: int) -> int:
        """The twisty box's width in a row whose content starts at `x` and
        must end by `right`: TWISTY_W, less whatever the row overflows by,
        but never narrower than its glyph (0 for a row without one). Icon,
        label and chip keep their width (CSS: a Quasar icon does not shrink,
        nowrap text not below its own width)."""
        need = T.TWISTY_W + T.TREE_GAP + T.ICON_PX + T.TREE_GAP + label_w
        if chip_w:
            need += T.TREE_GAP + _px(8) + chip_w
        floor = min(T.TWISTY_W, T.text_w(glyph, "sans", 9)) if glyph else 0
        return max(floor, T.TWISTY_W - max(0, x + need - right))

    def draw(self, surf, body_rect, rows, scroll, mouse) -> list:
        """Draw the rows scrolled by `scroll` px; returns [(row_id, Rect,
        kind)] for every visible row, kind "twisty" (a stage row's twisty
        box, listed first) or "row" -- hit-test in list order."""
        body = pygame.Rect(body_rect)
        self.tip = None
        hits = []
        clip0 = surf.get_clip()
        surf.set_clip(body.clip(clip0) if clip0 else body)
        try:
            y0 = body.y + _px(3) - int(scroll)
            rh = T.TREE_ROW_H
            for i, row in enumerate(rows):
                y = y0 + rh * i
                if y + rh <= body.y or y >= body.bottom:
                    continue
                rr = pygame.Rect(body.x, y, body.w, rh)
                vis = rr.clip(body)
                hot = _hit(vis, mouse)
                if hot:
                    surf.fill(T.TREE_HOVER, rr)
                elif row.get("selected"):
                    surf.fill(T.ACCENT_FILL, rr)
                x = body.x + T.TREE_BASE + T.TREE_INDENT * int(row.get("level", 0))
                right = body.right - _px(6)
                sel = bool(row.get("selected"))
                label = str(row.get("label", ""))
                chip = row.get("chip")
                cw = T.text_w(chip, "mono", 9.5) if chip else 0
                tw = row.get("twisty")
                glyph = ("▾" if tw == "open" else "▸") if tw else ""
                tw_w = self._twisty_w(glyph, x, right, T.text_w(label, "sans", 12, sel), cw)
                if tw:
                    T.text(surf, glyph, x + tw_w // 2, _sans_y(y, rh, 9), "sans", 9, T.INK_MUTED, anchor="centre")
                    hits.append((row.get("id"), pygame.Rect(x, y, tw_w, rh).clip(body), "twisty"))
                x += tw_w + T.TREE_GAP
                if row.get("icon"):
                    T.icon(surf, row["icon"], x, y + (rh - T.ICON_PX + 1) // 2, T.ICON_PX,
                           row.get("icon_colour") or T.INK_MUTED)
                x += T.ICON_PX + T.TREE_GAP
                lw = T.text(surf, label, x, _sans_y(y, rh, 12), "sans", 12, T.INK_FAINT if row.get("faint") else T.INK,
                            sel, clip_w=max(0, right - x))
                if chip:
                    cx = max(x + lw + T.TREE_GAP + _px(8), right - cw)
                    T.text(surf, chip, cx, y + (rh - T.font("mono", 9.5).get_height()) // 2, "mono", 9.5, T.INK_FAINT)
                if row.get("focus"):
                    pygame.draw.rect(surf, T.ACCENT, rr, 1)
                hits.append((row.get("id"), vis, "row"))
                if hot and row.get("tip"):
                    _tip(self, vis, row["tip"])
        finally:
            surf.set_clip(clip0)
        return hits

    def reveal(self, rows, first_id, last_id, body_h, scroll) -> int:
        """The smallest change of `scroll` that shows rows first_id..last_id
        (a stage's block: its row through its last view row) inside a body
        `body_h` tall; when the block is taller than the body, the stage row
        and the selected row are what must show. Unknown ids leave the
        scroll as it is (clamped)."""
        ids = [r.get("id") for r in rows]
        total = self.content_h(rows)
        if first_id not in ids:
            return clamp_scroll(scroll, total, body_h)
        a = ids.index(first_id)
        b = ids.index(last_id) if last_id in ids else a
        a, b = min(a, b), max(a, b)
        rh, top_pad = T.TREE_ROW_H, _px(3)
        if (b - a + 1) * rh > body_h:
            sel = next((i for i in range(a, b + 1) if rows[i].get("selected")), a)
            b = sel
        top = top_pad + rh * a
        bottom = top_pad + rh * (b + 1)
        s = int(scroll)
        if top < s:
            s = top if a else 0
        elif bottom > s + body_h:
            s = bottom - body_h
            if top < s:                                  # the stage row wins over the selected one
                s = top
        return clamp_scroll(s, total, body_h)


# -- Properties ------------------------------------------------------------------------------------------
class PropertyGrid:
    """The read-only Properties grid (SH §4.1). Rows are ("group", title)
    or (key, value) / (key, value, colour):

      * group: HEADER fill 22 px + a RULE bottom (23 pitch), SANS 10.5/600
        INK_MUTED at x + 6;
      * fact: key 44 % (SANS 12 INK_MUTED; T.HINT_CSS, AeroBO 11), a RULE_SOFT vertical rule, value
        56 % (MONO 11, INK or the row's colour), both padded 6 px, cut with
        "…" and carrying their full text as a hover tooltip; 20 px + a
        RULE_SOFT bottom (21 pitch).

    No rows: the hint "nothing selected" at 6 / 8."""

    def __init__(self, overlay=None):
        self.overlay = overlay
        self.tip = None

    def draw(self, surf, body_rect, rows, scroll, mouse) -> int:
        """Draw the rows scrolled by `scroll`; returns the content height."""
        body = pygame.Rect(body_rect)
        self.tip = None
        clip0 = surf.get_clip()
        surf.set_clip(body.clip(clip0) if clip0 else body)
        try:
            if not rows:
                T.text(surf, "nothing selected", body.x + _px(8), body.y + _px(6), "sans", T.HINT_CSS, T.INK_MUTED)
                return 0
            kw = int(math.floor(T.PROP_KEY_FRAC * body.w + 0.5))
            pad = _px(6)
            y = body.y - int(scroll)
            gh, rh = T.PROP_GROUP_H, T.PROP_ROW_H
            for row in rows:
                if row and row[0] == "group":
                    if y + gh > body.y and y < body.bottom:
                        surf.fill(T.HEADER, (body.x, y, body.w, gh - 1))
                        pygame.draw.line(surf, T.RULE, (body.x, y + gh - 1), (body.right - 1, y + gh - 1))
                        T.text(surf, row[1], body.x + pad, y + _px(2), "sans", 10.5, T.INK_MUTED, True,
                               clip_w=body.w - 2 * pad)
                    y += gh
                    continue
                if y + rh > body.y and y < body.bottom:
                    key, value = str(row[0]), str(row[1])
                    colour = row[2] if len(row) > 2 and row[2] else T.INK
                    pygame.draw.line(surf, T.RULE_SOFT, (body.x, y + rh - 1), (body.right - 1, y + rh - 1))
                    pygame.draw.line(surf, T.RULE_SOFT, (body.x + kw - 1, y), (body.x + kw - 1, y + rh - 2))
                    kr = pygame.Rect(body.x, y, kw, rh - 1)
                    vr = pygame.Rect(body.x + kw, y, body.w - kw, rh - 1)
                    T.text(surf, key, kr.x + pad, _sans_y(y, rh - 1, T.HINT_CSS), "sans", T.HINT_CSS, T.INK_MUTED,
                           clip_w=kw - 1 - 2 * pad)
                    T.text(surf, value, vr.x + pad, _mono_y(y, rh - 1, 11), "mono", 11, colour,
                           clip_w=vr.w - 2 * pad)
                    if _hit(kr.clip(body), mouse):
                        _tip(self, kr.clip(body), key)
                    elif _hit(vr.clip(body), mouse):
                        _tip(self, vr.clip(body), value)
                y += rh
            return y + int(scroll) - body.y
        finally:
            surf.set_clip(clip0)

    @staticmethod
    def content_h(rows) -> int:
        return sum(T.PROP_GROUP_H if r and r[0] == "group" else T.PROP_ROW_H for r in rows)


# -- tab strip -------------------------------------------------------------------------------------------
class TabStrip:
    """The work area's tab strip (SH §6.2): CHROME, 25 px with its RULE
    bottom, padding 3 5 0 5, tabs 2 px apart sitting on the rule. An
    inactive tab is 22 px, TAB_OFF (TAB_HOVER under the mouse), a RULE
    frame without a bottom, top corners radius 2, SANS 11.5 INK_MUTED,
    padding 0 11; the active one is WELL with INK / 600 and sits 1 px lower.
    `focus` draws the keyboard ring (1 px ACCENT inset) on the active tab.
    There is no disabled tab (AeroBO): every view of an unlocked stage is
    selectable, and one with nothing to show draws its empty state."""

    def __init__(self):
        self._hits: list = []

    def draw(self, surf, rect, tabs, active, *, focus=False, mouse=None) -> list:
        """Paint the strip; returns [(key, Rect)] of the drawn tabs."""
        r = pygame.Rect(rect)
        surf.fill(T.CHROME, (r.x, r.y, r.w, r.h - 1))
        rule_y = r.bottom - 1
        self._hits = []
        x = r.x + _px(5)
        th = T.TAB_H
        clip0 = surf.get_clip()
        surf.set_clip(r.clip(clip0) if clip0 else r)
        try:
            for tab in tabs:
                key, label = tab[0], str(tab[1])
                on = key == active
                w = 2 + 2 * _px(11) + T.text_w(label, "sans", 11.5, on)
                tr = pygame.Rect(x, rule_y - th + (1 if on else 0), w, th)
                fill = T.WELL if on else (T.TAB_HOVER if _hit(tr, mouse) else T.TAB_OFF)
                pygame.draw.rect(surf, fill, tr, border_top_left_radius=2, border_top_right_radius=2)
                # the frame has no bottom edge: run it down to the strip's rule, drawn last
                frame = pygame.Rect(tr.x, tr.y, tr.w, rule_y - tr.y + 1)
                pygame.draw.rect(surf, T.RULE, frame, 1, border_top_left_radius=2, border_top_right_radius=2)
                # the label sits at the same height on an active tab as on the rest
                fh = T.font("sans", 11.5, on).get_height()
                T.text(surf, label, x + 1 + _px(11), rule_y - th + 1 + (th - fh) // 2, "sans", 11.5,
                       T.INK if on else T.INK_MUTED, on)
                if on and focus:
                    pygame.draw.rect(surf, T.ACCENT, tr.inflate(-2, -2), 1)
                self._hits.append((key, tr))
                x += w + _px(2)
            pygame.draw.line(surf, T.RULE, (r.x, rule_y), (r.right - 1, rule_y))
        finally:
            surf.set_clip(clip0)
        return list(self._hits)

    def click(self, pos) -> str | None:
        for key, tr in self._hits:
            if tr.collidepoint(pos):
                return key
        return None


# -- Output log ------------------------------------------------------------------------------------------
class OutputLog:
    """The Output log (SH §7.1): one line per event, `HH:MM:SS` MONO 11
    INK_FAINT, 6 px, then the message MONO 11 in its level's colour (info
    INK, ok GOOD, warn WARN, error BAD); 17 px lines inset 20 / 16; a long
    message wraps at spaces and its continuation lines align with the
    message column. At most LOG_MAX (400) lines, oldest dropped; every new
    line scrolls to the tail. The wheel scrolls it (3 lines a notch); after
    `draw`, `bar` is its thumb for a `ScrollDrag` (QScrollArea's look: 10
    px of black at .2, which is what shot 03 shows, not the 12 px bar)."""

    def __init__(self, max_lines=None):
        self.max_lines = int(max_lines or T.LOG_MAX)
        self.lines: list = []           # [(stamp, text, level)]
        self.scroll = 0
        self.follow = True              # pinned to the tail
        self._wrap: dict = {}           # (text, width, px) -> [str]
        self._rows = None               # [(line index, row in the line, text)] for one width
        self._rows_key = None
        self.bar = None                 # (track, thumb) of the last draw, None when nothing scrolls
        self.content_h = 0

    def append(self, stamp, text, level="info") -> None:
        self.lines.append((str(stamp), str(text), level if level in _LEVEL else "info"))
        if len(self.lines) > self.max_lines:
            del self.lines[:len(self.lines) - self.max_lines]
        self._rows = None
        self.follow = True

    def clear(self) -> None:
        self.lines = []
        self._rows = None
        self.scroll = 0
        self.follow = True

    def wheel(self, dy) -> None:
        """A MOUSEWHEEL notch count (positive = up, towards older lines)."""
        self.scroll = max(0, int(self.scroll - dy * 3 * T.LOG_LINE_H))
        self.follow = False

    def _wrapped(self, text: str, width: int) -> list:
        key = (text, width, T.px_for("mono", 11))
        out = self._wrap.get(key)
        if out is not None:
            return out
        if len(self._wrap) > 4 * T.LOG_MAX:
            self._wrap.clear()
        out = []
        for para in text.split("\n"):
            cur = ""
            for word in para.split(" "):
                cand = word if not cur else cur + " " + word
                if T.text_w(cand, "mono", 11) <= width or not cur:
                    cur = cand
                else:
                    out.append(cur)
                    cur = word
                while T.text_w(cur, "mono", 11) > width and len(cur) > 1:     # one token wider than the column
                    n = max(1, width // max(1, T.text_w("0", "mono", 11)))
                    out.append(cur[:n])
                    cur = cur[n:]
            out.append(cur)
        self._wrap[key] = out
        return out

    def _layout(self, width: int) -> list:
        key = (len(self.lines), id(self.lines[-1]) if self.lines else 0, width, T.px_for("mono", 11))
        if self._rows is None or self._rows_key != key:
            rows = []
            for i, (_stamp, text, _level) in enumerate(self.lines):
                for j, piece in enumerate(self._wrapped(text, width)):
                    rows.append((i, j, piece))
            self._rows, self._rows_key = rows, key
        return self._rows

    def geometry(self, body_rect):
        """(message x, wrap width, content height) for a body rect."""
        body = pygame.Rect(body_rect)
        x0 = body.x + T.LOG_INSET_X
        msg_x = x0 + T.text_w("00:00:00", "mono", 11) + _px(6)
        width = max(_px(40), body.right - T.SCROLL_W - msg_x)
        n = len(self._layout(width))
        return msg_x, width, T.LOG_INSET_Y + n * T.LOG_LINE_H + _px(9)

    def draw(self, surf, body_rect) -> None:
        body = pygame.Rect(body_rect)
        surf.fill(T.WELL, body)
        msg_x, width, content_h = self.geometry(body)
        rows = self._layout(width)
        span = max(0, content_h - body.h)
        if self.follow:
            self.scroll = span
        self.scroll = clamp_scroll(self.scroll, content_h, body.h)
        if self.scroll >= span:
            self.follow = True
        self.content_h = content_h
        lh = T.LOG_LINE_H
        x0 = body.x + T.LOG_INSET_X
        top = body.y + T.LOG_INSET_Y - self.scroll
        first = max(0, (body.y - top) // lh)
        clip0 = surf.get_clip()
        surf.set_clip(body.clip(clip0) if clip0 else body)
        try:
            for k in range(first, len(rows)):
                y = top + k * lh
                if y >= body.bottom:
                    break
                i, j, piece = rows[k]
                stamp, _text, level = self.lines[i]
                ty = _mono_y(y, lh, 11)
                if j == 0:
                    T.text(surf, stamp, x0, ty, "mono", 11, T.INK_FAINT)
                T.text(surf, piece, msg_x, ty, "mono", 11, getattr(T, _LEVEL[level]))
            self.bar = scrollbar(surf, body, content_h, self.scroll, thin=True)
        finally:
            surf.set_clip(clip0)


# -- status bar ------------------------------------------------------------------------------------------
class StatusBar:
    """The status bar (SH §8.1): a RULE top over CHROME; a 7 x 7 state
    square at x = 8 (idle INK_FAINT, busy ACCENT, ok GOOD, warn WARN, error
    BAD); the message SANS 11 INK_MUTED at x = 23; a 120 x 8 progress bar
    (PROG_TRACK, ACCENT fill) 8 px before the first cell, only while
    `progress` is not None; then the cells, right-aligned 8 px in, 9 px
    apart: MONO 11 INK_MUTED (or the cell's colour) behind a 1 px RULE left
    border and 8 px of padding, each at most 34 characters wide with "…".

    AeroBO has three cells (problem, dim, budget); the shell passes a
    fourth, `F1 keys` in INK_FAINT (carsim addition), and clicks it through
    the rects `draw` returns."""

    def __init__(self, overlay=None):
        self.overlay = overlay
        self.tip = None

    def draw(self, surf, rect, text, kind, progress, cells, mouse) -> list:
        """Paint the bar; returns the cells' rects, in the order given."""
        r = pygame.Rect(rect)
        self.tip = None
        pygame.draw.line(surf, T.RULE, (r.x, r.y), (r.right - 1, r.y))
        body = pygame.Rect(r.x, r.y + 1, r.w, r.h - 1)
        surf.fill(T.CHROME, body)
        sq = _px(7)
        pygame.draw.rect(surf, getattr(T, _SQUARE.get(kind, "INK_FAINT")),
                         (r.x + _px(8), body.y + (body.h - sq + 1) // 2, sq, sq), border_radius=1)
        # cells from the right
        ch = T.text_w("0", "mono", 11)
        cap = _CELL_CH * ch
        pad, gap = _px(8), _px(9)
        specs = []
        for c in cells or ():
            if isinstance(c, str):
                s, col = c, T.INK_MUTED
            else:
                s, col = str(c[0]), (c[1] if len(c) > 1 and c[1] else T.INK_MUTED)
            w = min(cap, 1 + pad + T.text_w(s, "mono", 11))
            specs.append((s, col, w))
        x = r.right - pad
        rects = [None] * len(specs)
        for i in range(len(specs) - 1, -1, -1):
            s, col, w = specs[i]
            x -= w
            rects[i] = pygame.Rect(x, body.y, w, body.h)
            if i:
                x -= gap
        first_x = rects[0].x if rects else r.right - pad
        bh = _px(16)
        for (s, col, w), cr in zip(specs, rects):
            pygame.draw.line(surf, T.RULE, (cr.x, body.y + (body.h - bh) // 2),
                             (cr.x, body.y + (body.h - bh) // 2 + bh - 1))
            shown = T.ellipsize(s, "mono", 11, w - 1 - pad)
            T.text(surf, shown, cr.x + 1 + pad, _mono_y(body.y, body.h, 11), "mono", 11, col)
            if shown != s and _hit(cr, mouse):
                _tip(self, cr, s)
        msg_right = first_x - pad
        if progress is not None:
            pw, ph = _px(120), _px(8)
            pr = pygame.Rect(first_x - pad - pw, body.y + (body.h - ph) // 2, pw, ph)
            surf.fill(T.PROG_TRACK, pr)
            frac = max(0.0, min(1.0, float(progress))) if math.isfinite(float(progress)) else 0.0
            if frac > 0:
                surf.fill(T.ACCENT, (pr.x, pr.y, max(1, int(round(pw * frac))), ph))
            msg_right = pr.x - pad
        mx = r.x + _px(23)
        T.text(surf, text or "", mx, _sans_y(body.y, body.h, 11), "sans", 11, T.INK_MUTED,
               clip_w=max(0, msg_right - mx))
        return rects


# -- toasts ----------------------------------------------------------------------------------------------
class Toasts:
    """Quasar notifications (SH §6.14): bottom-centre, 10 px above the
    window's edge, 48 px tall, stacking upward 10 px apart (the newest at
    the bottom); a 24 px icon 16 px in, 16 px to the text (SANS 12), 16 px
    of padding after it; radius 2 and a soft shadow; gone after TOAST_S
    (5 s). kind: positive (GOOD, check_circle) | negative (BAD, warning) |
    warning (WARN, PANEL text) | info (TOAST_INFO, info); ok / warn / error
    are accepted as aliases."""

    def __init__(self):
        self.items: list = []           # [dict(text, kind, t0)]
        self.rects: list = []           # the drawn rects, newest last

    def push(self, text, kind="info", now=None) -> None:
        """Queue a toast; its 5 s start at `now` (None: at its first draw).
        A toast identical to one still showing restarts that one instead of
        stacking a copy."""
        kind = _TOAST_ALIAS.get(kind, kind)
        if kind not in _TOAST:
            kind = "info"
        for it in self.items:
            if it["text"] == str(text) and it["kind"] == kind:
                self.items.remove(it)
                break
        self.items.append(dict(text=str(text), kind=kind, t0=now))

    def draw(self, surf, W, H, now) -> None:
        self.items = [it for it in self.items if it["t0"] is None or now - it["t0"] < T.TOAST_S]
        for it in self.items:
            if it["t0"] is None:
                it["t0"] = now
        self.rects = []
        pad, icon_px = _px(16), _px(24)
        lh = _px(17)
        max_text = max(_px(120), int(W * 0.6) - 3 * pad - icon_px)
        bottom = H - _px(10)
        for it in reversed(self.items):
            fill, ink, icon = _TOAST[it["kind"]]
            lines = _wrap_words(it["text"], "sans", 12, max_text)
            tw = max(T.text_w(s, "sans", 12) for s in lines)
            h = max(_px(48), 2 * _px(15) + lh * len(lines))
            w = pad + icon_px + pad + tw + pad
            r = pygame.Rect(W // 2 - w // 2, bottom - h, w, h)
            _shadow(surf, r)
            pygame.draw.rect(surf, fill, r, border_radius=2)
            T.icon(surf, icon, r.x + pad, r.y + (h - icon_px) // 2, icon_px, ink)
            fh = T.font("sans", 12).get_height()
            ty = r.y + (h - lh * len(lines)) // 2 + (lh - fh + 1) // 2
            for s in lines:
                T.text(surf, s, r.x + pad + icon_px + pad, ty, "sans", 12, ink)
                ty += lh
            self.rects.insert(0, r)
            bottom = r.y - _px(10)


def _wrap_words(text: str, family: str, css: float, width: int) -> list:
    """`text` broken at spaces into lines no wider than `width` px (a word
    wider than that keeps a line of its own)."""
    out, cur = [], ""
    for word in str(text).split():
        cand = word if not cur else cur + " " + word
        if not cur or T.text_w(cand, family, css) <= width:
            cur = cand
        else:
            out.append(cur)
            cur = word
    out.append(cur)
    return out


# -- dialog ----------------------------------------------------------------------------------------------
class Dialog:
    """A modal AeroBO dialog (SH §6.15): `width` px (620), white, the pane
    title style (title SANS 11/600 INK_MUTED, a close ✕ at the right) over
    a body drawn by `body_fn(ui)` with a `widgets.WorkUI` (padding 9); on
    Quasar's backdrop (black at .4). It sizes itself to its content (up to
    the window less 40 px each side; then the body scrolls with the wheel).
    ESC, the ✕ or a click on the backdrop closes it; while open, `handle`
    consumes every event."""

    def __init__(self, title, width=620, body_fn=None):
        self.title = str(title)
        self.width = int(width)
        self.body_fn = body_fn
        self.open = True
        self.scroll = 0
        self.state: dict = {}           # the body's WorkUI state (disclosures, ...)
        self._content_h = None          # the body's height as last laid out
        self._card = None
        self._close = None
        self._body = None
        self._overlay = None            # a private Overlay when the caller passes none

    def close(self) -> None:
        self.open = False

    def _geometry(self, W, H):
        w = min(_px(self.width), W - 2 * _px(16))
        body_h = self._content_h if self._content_h is not None else _px(120)
        h = min(T.PANE_TITLE_H + body_h + 2, H - 2 * _px(40))
        card = pygame.Rect((W - w) // 2, (H - h) // 2, w, h)
        body = pygame.Rect(card.x + 1, card.y + T.PANE_TITLE_H + 1, card.w - 2, card.h - T.PANE_TITLE_H - 2)
        return card, body

    def draw(self, surf, W, H, *, mouse=None, now=0.0, overlay=None, form=None) -> pygame.Rect:
        """Draw the backdrop, the card and its body; returns the body rect.
        The body is a `widgets.WorkUI` on `form` (None: WorkUI's private
        empty Form, kept in `state`) with `overlay` for its "?" popups."""
        if not self.open:
            return pygame.Rect(0, 0, 0, 0)
        T.blend_rgba(surf, (0, 0, W, H), _BACKDROP)
        card, body = self._geometry(W, H)
        self._card, self._body = card, body
        surf.fill(T.WELL, card)
        bar = pygame.Rect(card.x + 1, card.y + 1, card.w - 2, T.PANE_TITLE_H - 1)
        surf.fill(T.HEADER, bar)
        pygame.draw.line(surf, T.RULE, (card.x + 1, card.y + T.PANE_TITLE_H), (card.right - 2, card.y + T.PANE_TITLE_H))
        T.text(surf, self.title, card.x + _px(8), _sans_y(bar.y, bar.h, 11, True), "sans", 11, T.INK_MUTED, True,
               clip_w=card.w - _px(48))
        self._close = pane_tool_rect(card)
        if _hit(self._close, mouse):
            surf.fill(T.fade(T.STOP_GREY, _FLAT_HOVER, T.HEADER), self._close)
        T.icon(surf, "close", self._close.centerx, bar.y + (bar.h - T.TOOL_ICON) // 2, T.TOOL_ICON, T.INK_MUTED,
               "centre")
        pygame.draw.rect(surf, T.RULE, card, 1)
        if self.body_fn is not None:
            self._draw_body(surf, body, mouse, now, overlay, form)
        return body

    def _draw_body(self, surf, body, mouse, now, overlay, form) -> None:
        from . import widgets                       # the work-area kit (it imports plot, form, theme)
        if overlay is None:
            if self._overlay is None:
                self._overlay = widgets.Overlay()
            overlay = self._overlay
        # WorkUI pads its rect by WORK_PAD_X / WORK_PAD_Y; a dialog's body is padded 9
        inner = body.inflate(2 * (T.WORK_PAD_X - _px(9)), 2 * (T.WORK_PAD_Y - _px(9)))
        clip0 = surf.get_clip()
        surf.set_clip(body.clip(clip0) if clip0 else body)
        try:
            ui = widgets.WorkUI(surf, inner, form, self.state, scroll=self.scroll, kbd_focus=False,
                                mouse=mouse if mouse is not None else (-1, -1), overlay=overlay, now=now)
            self.body_fn(ui)
            h = int(ui.end())
        finally:
            surf.set_clip(clip0)
        self._content_h = max(_px(40), h - 2 * (T.WORK_PAD_Y - _px(9)))
        self.scroll = clamp_scroll(self.scroll, self._content_h, body.h)

    def handle(self, ev) -> bool:
        """Route one event while open (always consumed). ESC, a click on the
        ✕ or on the backdrop close the dialog; the wheel scrolls the body."""
        if not self.open:
            return False
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
            self.close()
        elif ev.type == pygame.MOUSEBUTTONDOWN and getattr(ev, "button", 0) == 1:
            if _hit(self._close, ev.pos) or (self._card is not None and not self._card.collidepoint(ev.pos)):
                self.close()
        elif ev.type == pygame.MOUSEWHEEL and self._body is not None and self._content_h is not None:
            self.scroll = clamp_scroll(self.scroll - ev.y * 3 * _px(17), self._content_h, self._body.h)
        return True


# ==================================================================== #
#  SELF-CHECK                                                          #
# ==================================================================== #
def shell_rects(W: int, H: int) -> dict:
    """The shell's pane rects for a W x H window at the current scale (PLAN
    §4.1, SPECS/aerobo_shell.md §1.4): menu, tool, status, tree, props,
    main, tabs, work, output. design_shell owns the live layout; this copy
    is what the self-check's mock frame is drawn on."""
    top = T.MENU_H + T.TOOL_H
    col_y = top + T.PAD
    col_h = H - top - T.STATUS_H - 2 * T.PAD
    tree_h = int((col_h - T.GAP) * T.TREE_SPLIT)
    main_x = T.PAD + T.LEFT_W + T.GAP
    main_w = W - main_x - T.PAD
    main_h = col_h - T.GAP - T.OUTPUT_H
    return {"menu": pygame.Rect(0, 0, W, T.MENU_H), "tool": pygame.Rect(0, T.MENU_H, W, T.TOOL_H),
            "status": pygame.Rect(0, H - T.STATUS_H, W, T.STATUS_H),
            "tree": pygame.Rect(T.PAD, col_y, T.LEFT_W, tree_h),
            "props": pygame.Rect(T.PAD, col_y + tree_h + T.GAP, T.LEFT_W, col_h - T.GAP - tree_h),
            "main": pygame.Rect(main_x, col_y, main_w, main_h),
            "tabs": pygame.Rect(main_x + 1, col_y + 1, main_w - 2, T.TAB_STRIP_H),
            "work": pygame.Rect(main_x + 1, col_y + 1 + T.TAB_STRIP_H, main_w - 2, main_h - 2 - T.TAB_STRIP_H),
            "output": pygame.Rect(main_x, col_y + main_h + T.GAP, main_w, T.OUTPUT_H)}


def _ico(state):
    """(icon, colour) of a tree node state (SH §3.3)."""
    return {"done": ("check_circle", T.GOOD), "ready": ("radio_button_unchecked", T.INK_MUTED),
            "active": ("play_circle", T.ACCENT), "running": ("pending", T.ACCENT),
            "locked": ("lock", T.INK_FAINT), "warn": ("error_outline", T.WARN), "error": ("cancel", T.BAD)}[state]


def _stage(sid, label, state, chip, twisty, *, selected=False, tip=""):
    icon, col = _ico(state)
    return dict(id=sid, level=0, label=label, icon=icon, icon_colour=col, chip=chip, selected=selected,
                faint=state == "locked", twisty=twisty, tip=tip)


def _view(vid, label, icon, *, selected=False, faint=False):
    return dict(id=vid, level=1, label=label, icon=icon, icon_colour=T.INK_FAINT if faint else T.INK_MUTED,
                chip=None, selected=selected, faint=False, twisty=None, tip="")


#: shot 03_after_accept's content, verbatim, for the pixel comparison
_REPLICA_03 = dict(
    app="WingLab",
    menus=[("File", [("New session", "restart_alt", "new", None),
                     ("Open results folder path", "folder_open", "path", None)]),
           ("Edit", [("Reset design box", "crop_free", "box", None),
                     ("Clear chosen section", "layers_clear", "clear", None)]),
           ("Solution", [("Run current stage", "play_arrow", "run", None), ("Stop", "stop", "stop", None)]),
           ("Tools", [("Copy run snippet to output", "content_copy", "snip", None),
                      ("Clear output log", "clear_all", "clearlog", None)]),
           ("Help", [("About this pipeline", "info", "about", None)])],
    tools=[("icon", "restart", "restart_alt", "New session — reset every stage"), ("sep",),
           ("primary", "run", "play_arrow", "Run", "Run the current stage"),
           ("flat", "stop", "stop", "Stop", "Stop what is running"), ("sep",),
           ("icon", "prev", "chevron_left", "Previous stage"), ("icon", "next", "chevron_right", "Next stage"),
           ("sep",), ("crumbs",), ("space",), ("chips",)],
    crumbs=[("m", "Mission"), ("af", "Airfoil"), ("ep", "Endplate"), ("w", "Wing"), ("r", "Results")],
    crumb_cur="af", chips=[("AIR", T.INK_MUTED), ("CONSTRAINED", T.WARN)],
    tree=[_stage("m", "1 Mission", "done", "CL 1.000", "open"),
          _view("m.operating", "Operating point", "tune"), _view("m.design", "Design point", "speed"),
          _view("m.search", "Search & budget", "bolt"),
          _stage("af", "2 Airfoil", "active", "the family's own section", "open"),
          _view("af.screen", "Library screening", "tune", selected=True),
          _view("af.rank", "Ranking", "format_list_numbered"), _view("af.section", "Section", "gesture"),
          _view("af.opt", "Shape optimisation", "auto_graph"),
          _stage("ep", "2.8 Endplate", "ready", "NACA 0010 (its own default)", "closed"),
          _stage("w", "3 Wing", "ready", "14-D", "closed"),
          _stage("r", "4 Results", "locked", "", "closed", tip="no completed run yet")],
    props=[("group", "Mission"), ("medium", "Car rear wing (track)"), ("design weight", "537.3 N"),
           ("speed", "55.00 m/s"), ("density", "1.225 kg/m³"), ("dynamic pressure", "1853 Pa"),
           ("reference area", "0.290 m²"), ("CL design", "1.0000"), ("group", "Section"),
           ("source", "not chosen"), ("AR estimate", "8.828"), ("MAC (estimate)", "0.1812 m"),
           ("Re at MAC", "6.825e+05"), ("screened at Re", "6.825e+05"), ("screened at Cl", "1.0000")],
    tabs=[("af.screen", "Library screening"), ("af.rank", "Ranking"), ("af.section", "Section"),
          ("af.opt", "Shape optimisation")], tab="af.screen",
    log=[("00:11:24", "V3.5 session started — state the mission, then the section, then the wing.", "info"),
         ("00:11:24", "criterion weights for the wing now follow its job (wing) — the GDP bulk-sweep preset: cruise "
                      "L/D first, then Cl max and the pitching moment — a wing that carries the design load", "warn"),
         ("00:11:24", "criterion weights for the second surface now follow its job (wing) — the GDP bulk-sweep "
                      "preset: cruise L/D first, then Cl max and the pitching moment — a wing that carries the design "
                      "load", "warn"),
         ("00:11:24", "medium set to Car rear wing (track) — solver family: car rear wing + endplates + free chord "
                      "law", "info"),
         ("00:11:48", "mission accepted — CL 1.0000, Re at MAC 6.825e+05, MAC 0.1812 m", "ok"),
         ("00:11:50", "design box measured for this mission: twist_root_deg -0.9987–1.262, twist_tip_deg "
                      "-4.772–0.8464, alpha_deg 1.721–12, endplate_h_m 0.1333–0.5309, ride_height_m 0.3–0.6079, "
                      "endplate_toe_deg -4.891–4.211, b_m 1.261–2", "info")],
    status=("Ready", "idle", None),
    cells=["problem " + "Car rear wing, endplates designed (14-D) + chord law"[:38], "dim 14", "budget 53"],
    stop_on=False, toast=("Mission accepted", "positive"))

#: a fresh Garage's shell (Mission expanded, the design stages collapsed and
#: locked), in PLAN §2.9 / §3.1's words; the chips are PLAN §2.9.3's examples.
#: The busy status line (a §3.1.8 template with its bar) and the enabled Stop
#: are there so one frame shows every chrome widget, not a fresh Garage's state
_MOCK_FRESH = dict(
    app="carsim",
    menus=[("File", [("Start the design over", "restart_alt", "restart", None), ("Save the wing", "save", "save", "S"),
                     ("Rename the wing…", "edit", "rename", "N"), ("Airfoil library…", "menu_book", "lib", "A"),
                     ("Back", "arrow_back", "back", "ESC")]),
           ("Edit", [("Reset the design box", "crop_free", "boxreset", None),
                     ("Let the plates fly flat", "layers_clear", "flat", None),
                     ("Use the recommended weights", "tune", "rec", None)]),
           ("Solution", [("Run current stage", "play_arrow", "run", "F5"), ("Stop", "stop", "stop", "ESC"),
                         ("Continue / keep going", "play_arrow", "keep", "K")]),
           ("Tools", [("Ask XFOIL for this section", "science", "xfoil", "X"),
                      ("Copy the run summary to the output", "content_copy", "snip", None),
                      ("Clear output log", "clear_all", "clearlog", None)]),
           ("Help", [("Keys and controller", "keyboard", "keys", "F1"),
                     ("About this pipeline", "info", "about", None)])],
    tools=[("icon", "restart", "restart_alt", "Start the design over — every stage re-locked; the mission stays"),
           ("sep",), ("icon", "save", "save", "Save the wing to the library and put it in the slot (S)"), ("sep",),
           ("primary", "run", "play_arrow", "Run", "Run the current stage (F5 · SQUARE)"),
           ("flat", "stop", "stop", "Stop", "Stop what is running (ESC · CIRCLE)"), ("sep",),
           ("icon", "prev", "chevron_left", "Previous stage"), ("icon", "next", "chevron_right", "Next stage"),
           ("sep",), ("crumbs",), ("space",), ("chips",)],
    hidden=("save",),
    crumbs=[("m", "Mission"), ("af", "Airfoil"), ("ep", "Endplate"), ("w", "Wing"), ("r", "Results")],
    crumb_cur="m", chips=[("LEFT FLANK", T.ACCENT), ("ARENA", T.INK_MUTED), ("DRY", T.INK_MUTED),
                          ("CONSTRAINED", T.WARN)],
    tree=[_stage("m", "1 Mission", "active", "arena · dry", "open"),
          _view("m.operating", "Operating point", "tune", selected=True), _view("m.design", "Design point", "speed"),
          _view("m.search", "Search & budget", "bolt"),
          _stage("af", "2 Airfoil", "locked", "e423 · t/c 0.125", "closed",
                 tip="state the mission first — the section and the wing are both scored against its lap"),
          _stage("ep", "2.8 Endplate", "locked", "flat (its own default)", "closed"),
          _stage("w", "3 Wing", "locked", "7-D", "closed"),
          _stage("r", "4 Results", "locked", "", "closed")],
    props=[("group", "Mission"), ("state", "not stated", T.WARN)],
    tabs=[("m.operating", "Operating point"), ("m.design", "Design point"), ("m.search", "Search & budget")],
    tab="m.operating",
    log=[("12:00:00", "carsim garage — state the mission, then the sections, then the wing. Budgets follow WingLab's "
                      "measured law (1 Mission ▸ Search & budget). Keys: TAB tree/tabs/work · ENTER do it · F5 run · "
                      "ESC stop/back · F1 all keys", "info"),
         ("12:00:04", "circuit set to arena — the mission has to be stated again", "info"),
         ("12:00:06", "search settings: the measured recommendation is live — every stage's budget follows its own "
                      "design vector", "info"),
         ("12:00:09", "stop early: on — a run ends once the best has not improved by 0.2% of its span in a quarter "
                      "of its budget", "info"),
         ("12:00:12", "XFOIL did not converge for e423: estimate kept", "warn"),
         ("12:00:15", "run abandoned — nothing was applied (left the garage)", "warn")],
    status=("screening 12/34 sections · 2-D estimate", "busy", 12 / 34),
    cells=["problem left flank wing", "dim 7", "budget 40", ("F1 keys", T.INK_FAINT)],
    stop_on=True, toast=None)


def _mock_shell(surf, content, *, mouse=(-1, -1), now=0.0, overlay=None) -> dict:
    """A static shell frame from a content dict (the two above): every
    chrome widget in its place. Returns what was drawn (rects, hits)."""
    W, H = surf.get_size()
    R = shell_rects(W, H)
    surf.fill(T.CANVAS)
    mb = MenuBar(content["app"], content["menus"])
    tb = ToolBar(content["tools"], overlay)
    mb.draw(surf, R["menu"], mouse, lambda a: True)
    tb.draw(surf, R["tool"], enabled=lambda i: i != "stop" or content["stop_on"], crumbs=content["crumbs"],
            crumb_cur=content["crumb_cur"], chips=content["chips"], mouse=mouse, hidden=content.get("hidden", ()))
    tree_body = pane(surf, R["tree"], "Simulation")
    tree = TreeDraw(overlay)
    tree_hits = tree.draw(surf, tree_body, content["tree"], 0, mouse)
    props_body = pane(surf, R["props"], "Properties")
    PropertyGrid(overlay).draw(surf, props_body, content["props"], 0, mouse)
    main_body = pane(surf, R["main"], None, white=True)
    tabs = TabStrip()
    tab_hits = tabs.draw(surf, R["tabs"], content["tabs"], content["tab"], mouse=mouse)
    out_body = pane(surf, R["output"], "Output", white=True, tool="clear_all", mouse=mouse)
    log = OutputLog()
    for line in content["log"]:
        log.append(*line)
    log.draw(surf, out_body)
    text, kind, prog = content["status"]
    cells = StatusBar(overlay).draw(surf, R["status"], text, kind, prog, content["cells"], mouse)
    toasts = Toasts()
    if content.get("toast"):
        toasts.push(content["toast"][0], content["toast"][1], now=now)
        toasts.draw(surf, W, H, now)
    return dict(R=R, menu=mb, tool=tb, tree=tree_hits, tree_body=tree_body, props_body=props_body,
                main_body=main_body, tabs=tab_hits, output=out_body, log=log, cells=cells, toasts=toasts)


def self_check(verbose: bool = True, out: str = "") -> bool:
    """Every chrome widget against AeroBO's measured geometry (SPECS/
    aerobo_shell.md §1-§8, shot 03_after_accept), their input methods, and a
    full mock shell at 1600x1000. `out`: a directory for the mock frames."""
    import os
    import time

    import numpy as np
    n_ok = n_all = 0
    ok = True

    def rep(tag_, passed, msg=""):
        nonlocal ok, n_ok, n_all
        n_all += 1
        n_ok += bool(passed)
        ok = ok and bool(passed)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag_}" + (f": {msg}" if msg else ""))

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    T.warm()
    sf = os.path.basename(T.font_file("sans") or "").startswith("SFNS")
    surf = pygame.Surface((1600, 1000))

    def ink(rect, bg, s=None, tol=24):
        """The bounding box of the pixels of `rect` that differ from `bg`."""
        s = s or surf
        r = pygame.Rect(rect).clip(s.get_rect())
        sub = s.subsurface(r)
        m = pygame.mask.from_threshold(sub, bg, (tol, tol, tol, 255))
        m.invert()
        bs = m.get_bounding_rects()
        if not bs:
            return None
        b = bs[0].unionall(bs[1:])
        return b.move(r.x, r.y)

    def px(x, y):
        return tuple(surf.get_at((x, y)))[:3]

    class TipSink:                      # records what a widget asks the overlay for
        def __init__(self):
            self.asked = []

        def tooltip(self, rect, text):
            self.asked.append((pygame.Rect(rect), text))

    # 1 pane
    surf.fill(T.CANVAS)
    body = pane(surf, (2, 60, 276, 547), "Simulation")
    title = ink((3, 61, 200, 21), T.HEADER)
    rep("pane: 1 px RULE frame, 22 px title bar (HEADER + RULE), title at x + 8, body (3, 83, 274, 523)",
        tuple(body) == (3, 83, 274, 523) and px(2, 300) == T.RULE and px(100, 70) == T.HEADER
        and px(100, 82) == T.RULE and px(100, 83) == T.PANEL and title is not None and 10 <= title.x <= 11
        and (not sf or title.y == 68), f"body {tuple(body)}, title ink {title}")

    # 2 menu bar
    menus = [("File", [("New", "restart_alt", "new", None), None, ("Save the wing", "save", "save", "S")]),
             ("Edit", [("Reset the design box", "crop_free", "boxreset", None),
                       ("Let the plates fly flat", "layers_clear", "flat", None)])]
    mb = MenuBar("carsim", menus)
    surf.fill(T.CANVAS)
    live = {"new", "save", "boxreset"}
    mb.draw(surf, (0, 0, 1600, 26), (-1, -1), lambda a: a in live)
    file_box = mb._titles[0][1]
    mb.click(file_box.center)
    opened = mb.open == "File"
    mb.draw(surf, (0, 0, 1600, 26), (-1, -1), lambda a: a in live)
    mb.draw_menu(surf, (-1, -1))
    box, rows = mb._menu_layout()
    picked = mb.click(rows[2][1].center)
    mb.click(mb._titles[1][1].center)
    dead = mb.click(mb._menu_layout()[1][1][1].center)          # "Let the plates fly flat": disabled
    still_open = mb.open == "Edit"

    def key(k):
        return mb.key(pygame.event.Event(pygame.KEYDOWN, key=k, mod=0))
    key(pygame.K_DOWN)
    via_keys = key(pygame.K_RETURN)
    mb.click(file_box.center)
    key(pygame.K_ESCAPE)
    rep("menu bar: the first title 10 px + 1 after the tracked app name; a title opens its drop-down under the bar, "
        "an item returns its action and closes, a disabled item does nothing, DOWN + ENTER pick, ESC closes",
        opened and box.y == 26 and picked == "save" and dead is None and still_open and via_keys == "boxreset"
        and mb.open is None and file_box.x == 4 + 6 + T.text_w("carsim", "sans", 11, True) + 1 + 10 + 1,
        f"{picked} {dead} {via_keys}, File box {file_box}")

    # 3 tool bar (AeroBO's own items; SH §5.2 measured x)
    items = _REPLICA_03["tools"]
    tb = ToolBar(items)
    surf.fill(T.CANVAS)
    tb.draw(surf, (0, 26, 1600, 32), enabled=lambda i: i != "stop", crumbs=_REPLICA_03["crumbs"], crumb_cur="af",
            chips=_REPLICA_03["chips"], mouse=(-1, -1))
    run, stop, prev = tb.rects["run"], tb.rects["stop"], tb.rects["prev"]
    seps = [x for x in range(0, 300) if px(x, 41) == T.RULE and px(x, 33) == T.RULE and px(x, 32) == T.CHROME]
    crumb0 = tb.rects["crumb:m"]
    chips_ink = ink((1300, 30, 300, 24), T.CHROME)
    stop_click = tb.click(stop.center)
    at_03 = (run.right - 1 == 110 and abs(seps[1] - 184) <= 2 and abs(seps[2] - 264) <= 2
             and abs(crumb0.x - 272) <= 2)
    rep("tool bar: New 6..36, separators at 44 / 184±2 / 264±2, Run 52..110 x 30..53, crumbs from 272±2, chips "
        "ending 6 px in, 17 tall; clicks return ids (Stop disabled: None)",
        tb.rects["restart"].x == 6 and tb.rects["restart"].w == 31 and seps[0] == 44 and len(seps) == 3
        and (run.x, run.y, run.bottom - 1) == (52, 30, 53) and stop.x == run.right + 3 and (not sf or at_03)
        and chips_ink is not None and chips_ink.right == 1594 and (chips_ink.y, chips_ink.h) == (33, 17)
        and tb.click(run.center) == "run" and stop_click is None and tb.click(prev.center) == "prev"
        and tb.click(tb.rects["crumb:w"].center) == "crumb:w",
        f"Run {tuple(run)}, seps {seps}, crumbs {crumb0.x}, chips {chips_ink}")
    tb2 = ToolBar(_MOCK_FRESH["tools"])
    tb2.draw(surf, (0, 26, 1600, 32), enabled=lambda i: True, crumbs=[], chips=[], hidden=("save",))
    seps2 = [x for x in range(0, 300) if px(x, 41) == T.RULE and px(x, 33) == T.RULE and px(x, 32) == T.CHROME]
    rep("tool bar: a hidden button is not drawn and takes its doubled separator with it",
        "save" not in tb2.rects and tb2.rects["run"].x == 52 and seps2[:1] == [44], f"{seps2}")

    # 4 tree rows (SH §3.2)
    surf.fill(T.CANVAS)
    tree_body = pane(surf, (2, 60, 276, 547), "Simulation")
    tree = TreeDraw()
    rows = _REPLICA_03["tree"]
    hits = tree.draw(surf, tree_body, rows, 0, (-1, -1))
    row_rects = [(i, r) for i, r, k in hits if k == "row"]
    tw_first = [k for _i, _r, k in hits[:2]] == ["twisty", "row"]
    y_ok = all(r.y == 86 + 21 * n and r.h == 21 for n, (_i, r) in enumerate(row_rects))
    lab0 = ink((40, 86, 60, 21), T.PANEL)
    lab1 = ink((52, 107, 40, 21), T.PANEL)
    ic0 = ink((22, 86, 17, 21), T.PANEL)
    tw0 = ink((6, 86, 14, 21), T.PANEL)
    sel = px(200, 191 + 10) == T.ACCENT_FILL
    faint = ink((40, 86 + 21 * 11, 80, 21), T.PANEL, tol=4)
    faint_col = faint is not None and all(abs(a - b) <= 40 for a, b in zip(
        min((px(x, y) for x in range(faint.x, faint.right) for y in range(faint.y, faint.bottom)), key=sum),
        T.INK_FAINT))
    rep("tree: rows 21 px from body + 3; twisty 7..19, icon from 24, label from 43 (level 1: 56); the selected row "
        "ACCENT_FILL; a locked row INK_FAINT; twisty hits listed first",
        y_ok and tw_first and lab0 is not None and 43 <= lab0.x <= 45 and lab1 is not None and 56 <= lab1.x <= 58
        and ic0 is not None and 24 <= ic0.x <= 26 and tw0 is not None and 7 <= tw0.x and tw0.right <= 19
        and sel and faint_col and (not sf or lab0.y == 91),
        f"label {lab0}, level 1 {lab1}, icon {ic0}, twisty {tw0}")

    # 5 chips (SH §3.2, AeroBO theme.py:111-122): right-aligned 6 px in; a chip that does not fit sits 13 px
    #   after the label (gap 5 + padding 8), the twisty box gives way first -- icon and label slide left, at
    #   most down to the glyph -- and whatever still does not fit is cut at the pane edge
    chip0 = ink((200, 86, 77, 21), T.PANEL)

    def row_parts(chip):
        """Column spans [twisty, icon, label, chip] of a one-row tree (a chip of "0"s inks as one span)."""
        surf.fill(T.CANVAS)
        body = pane(surf, (2, 60, 276, 547), "Simulation")
        tree.draw(surf, body, [dict(id="x", level=0, label="Endplate", icon="radio_button_unchecked",
                                    icon_colour=T.INK_MUTED, chip=chip, selected=False, faint=False,
                                    twisty="closed", tip="")], 0, (-1, -1))
        a = pygame.surfarray.pixels3d(surf)[3:277, 86:107].astype(int)
        on = (np.abs(a - np.array(T.PANEL)).max(axis=2) > 24).any(axis=1)
        spans: list = []
        for x in (np.flatnonzero(on) + 3).tolist():
            if spans and x - spans[-1][1] <= 4:
                spans[-1][1] = x
            else:
                spans.append([x, x])
        return [tuple(s) for s in spans]

    lab_w = T.text_w("Endplate", "sans", 12)
    give = T.TWISTY_W - T.text_w("▸", "sans", 9)
    room = 271 - (43 + lab_w + 13)
    n_fit = max(n for n in range(1, 80) if T.text_w("0" * n, "mono", 9.5) <= room)    # "0"s that fit at rest
    over = 43 + lab_w + 13 + T.text_w("0" * (n_fit + 1), "mono", 9.5) - 271          # one more overflows
    rest, slid, cut = row_parts("00"), row_parts("0" * (n_fit + 1)), row_parts("0" * (n_fit + 12))
    outside = ink((277, 83, 1, 21), T.RULE, tol=1)
    shapes_ok = len(rest) == len(slid) == len(cut) == 4
    slide = min(over, give)
    rep("tree chips: MONO 9.5 right-aligned 6 px inside the edge; one that does not fit slides the icon and label "
        "left by what it lacks (at most the twisty's give), sits 13 px after the label, and is cut at the pane edge",
        chip0 is not None and chip0.right in (270, 271) and shapes_ok
        and rest[3][1] in (269, 270) and over > 0
        and slid[1][0] == rest[1][0] - slide and slid[2][0] == rest[2][0] - slide
        and (over > give or slid[3][1] == rest[3][1])
        and cut[2][0] == rest[2][0] - give and 13 <= cut[3][0] - cut[2][1] <= 16 and cut[3][1] >= 274
        and outside is None,
        f"chip {chip0}; label x {rest[2][0]} / {slid[2][0]} / {cut[2][0]} (overflow {over}, give {give}), "
        f"cut chip {cut[3] if shapes_ok else cut}")

    # 6 reveal
    rows24 = [dict(id=f"r{i}", level=0 if i % 5 == 0 else 1, label=f"row {i}", selected=(i == 13)) for i in range(24)]
    body_h = 403                        # 1280x800: 19 rows
    r_same = tree.reveal(rows24, "r0", "r4", body_h, 0)
    r_down = tree.reveal(rows24, "r20", "r23", body_h, 0)
    r_up = tree.reveal(rows24, "r1", "r4", body_h, 107)
    r_big = tree.reveal(rows24, "r10", "r23", 150, 0)
    rep("TreeDraw.reveal: no move when the block shows; the least scroll down / up to show it; a block taller than "
        "the body shows its stage row and the selected row",
        r_same == 0 and r_down == 3 + 21 * 24 - body_h and r_up == 3 + 21 * 1 and 3 + 21 * 10 >= r_big
        and 3 + 21 * 14 <= r_big + 150, f"{r_same} {r_down} {r_up} {r_big}")

    # 7 Properties (SH §4.1)
    surf.fill(T.CANVAS)
    pbody = pane(surf, (2, 609, 276, 366), "Properties")
    sink = TipSink()
    pg = PropertyGrid(sink)
    long_val = "CST section (optimised) · t/c 0.151, seeded from hg40"
    prow = _REPLICA_03["props"][:3] + [("name", long_val)]
    h = pg.draw(surf, pbody, prow, 0, (200, 632 + 23 + 21 * 2 + 10))
    kx = ink((3, 655, 110, 20), T.PANEL)
    vx = ink((124, 655, 150, 20), T.PANEL)
    rep("Properties: group 23 pitch (HEADER + RULE), facts 21 (RULE_SOFT), key 44 % | rule at x 123 | value from "
        "124, text at 9 / 130, cut with '…' and the full text as its tooltip; key ink from y 660 (HINT_CSS)",
        tuple(pbody) == (3, 632, 274, 342) and h == 23 + 3 * 21 and px(200, 640) == T.HEADER and px(200, 654) == T.RULE
        and px(200, 675) == T.RULE_SOFT and px(123, 665) == T.RULE_SOFT and px(124, 670) == T.PANEL
        and kx is not None and 9 <= kx.x <= 11 and vx is not None and 130 <= vx.x <= 131
        and sink.asked and sink.asked[-1][1] == long_val and (not sf or (kx.y, vx.y) == (660, 659)),
        # the key's ink starts at 660, AeroBO's at 661: the key is SANS 12 (T.HINT_CSS), 1 px over AeroBO's 11
        f"key {kx}, value {vx}, tip {sink.asked[-1][1][:24] if sink.asked else None}")

    # 8 tab strip (SH §6.2): no disabled style
    surf.fill(T.CANVAS)
    ts = TabStrip()
    tabs = _REPLICA_03["tabs"]
    hits = ts.draw(surf, (281, 61, 1316, 25), tabs, "af.screen", focus=False, mouse=(-1, -1))
    a, b = hits[0][1], hits[1][1]
    look = [(px(t.x + 3, 75), px(t.x, t.y + 5)) for _k, t in hits[1:]]
    hits_dis = ts.draw(surf, (281, 61, 1316, 25), [(k, lab, "disabled") for k, lab in tabs], "af.screen")
    look_dis = [(px(t.x + 3, 75), px(t.x, t.y + 5)) for _k, t in hits_dis[1:]]
    rep("tab strip: first tab at 286, 2 px apart, inactive 63..84 TAB_OFF, active 64..85 WELL on the RULE, label 12 "
        "px in; no disabled look (the same pixels whatever a tab carries); click returns the key",
        a.x == 286 and a.y == 64 and b.y == 63 and b.x == a.right + 2 and px(a.x + 5, 75) == T.WELL
        and px(b.x + 5, 75) == T.TAB_OFF and px(b.x, 70) == T.RULE and px(300, 85) == T.RULE
        and look == look_dis and ts.click(b.center) == "af.rank"
        and (not sf or all(abs(t.x - x) <= 4 for (_k, t), x in zip(hits, (286, 412, 481, 548)))),
        f"tabs x {[t.x for _k, t in hits]} (WingLab 286 412 481 548)")

    # 9 output log (SH §7.1)
    log = OutputLog()
    for i in range(450):
        log.append("00:12:%02d" % (i % 60), f"line {i}", "info")
    capped = len(log.lines) == T.LOG_MAX and log.lines[0][1] == "line 50"
    log.clear()
    long_msg = " ".join(f"word{i}" for i in range(80))
    log.append("00:11:24", long_msg, "warn")
    for i in range(8):
        log.append("00:11:25", f"short {i}", "ok" if i % 2 else "info")
    surf.fill(T.CANVAS)
    obody = pane(surf, (280, 825, 1318, 150), "Output", white=True, tool="clear_all")
    log.draw(surf, obody)
    msg_x, width, content_h = log.geometry(obody)
    tail = ink((obody.x, obody.bottom - 26, 400, 17), T.WELL)
    log.scroll = 0
    log.follow = False
    log.draw(surf, obody)
    head_ts = ink((obody.x + 18, obody.y + 14, 60, 17), T.WELL)
    wrap2 = ink((obody.x, obody.y + 16 + 17, msg_x - obody.x - 1, 17), T.WELL)
    wrap2_msg = ink((msg_x - 1, obody.y + 16 + 17, 200, 17), T.WELL)
    log.wheel(-10)
    rep("Output: capped at 400 (oldest dropped); timestamp at +20 / +16, message 6 px after it; wrapped rows start "
        "at the message column; new lines scroll to the tail",
        capped and head_ts is not None and 301 <= head_ts.x <= 302 and wrap2 is None and wrap2_msg is not None
        and msg_x == 301 + T.text_w("00:00:00", "mono", 11) + 6 and tail is not None
        and (not sf or head_ts.y == obody.y + 16 + 4) and len(log._layout(width)) > 9,
        f"ts {head_ts}, msg x {msg_x}, rows {len(log._layout(width))}")

    # 10 status bar (SH §8.1)
    surf.fill(T.CANVAS)
    sink = TipSink()
    sb = StatusBar(sink)
    cells = ["problem Car rear wing, endplates designed (14-D) + chord law", "dim 14", "budget 53",
             ("F1 keys", T.INK_FAINT)]
    rects = sb.draw(surf, (0, 977, 1600, 23), "Ready", "idle", None, cells, (1300, 988))
    no_bar = ink((700, 983, 300, 12), T.CHROME)
    ch = T.text_w("0", "mono", 11)
    sb.draw(surf, (0, 977, 1600, 23), "XFOIL evaluation 14/164", "busy", 14 / 164, cells, (-1, -1))
    bar_px = px(rects[0].x - 8 - 120 + 2, 988), px(rects[0].x - 8 - 2, 988)
    rep("status bar: square at 8 by kind, message at 23; four cells right-aligned 8 px in, 9 apart, <= 34 "
        "characters (the long one cut with '…', tooltip); progress 120 x 8 only while it is not None",
        len(rects) == 4 and rects[-1].right == 1592 and all(rects[i + 1].x - rects[i].right == 9 for i in range(3))
        and rects[0].w == min(34 * ch, rects[0].w) and rects[0].w == 34 * ch and no_bar is None
        and bar_px == (T.ACCENT, T.PROG_TRACK) and px(10, 988) == T.ACCENT and sink.asked
        and sink.asked[0][1] == cells[0],
        f"cells {[r.x for r in rects]} (WingLab 1231 1466 1523 + the fourth)")

    # 11 toasts (SH §6.14)
    toasts = Toasts()
    toasts.push("Mission accepted", "positive", now=0.0)
    toasts.push("already running", "warn", now=1.0)
    surf.fill(T.WELL)
    toasts.draw(surf, 1600, 1000, 1.5)
    r_old, r_new = toasts.rects
    w_want = 16 + 24 + 16 + T.text_w("Mission accepted", "sans", 12) + 16
    toasts.draw(surf, 1600, 1000, 5.2)
    left = [t["text"] for t in toasts.items]
    rep("toasts: bottom-centre 10 px above the edge, 48 tall, the newest at the bottom, 10 px apart; gone after 5 s",
        r_new.bottom == 990 and r_new.h == 48 and abs(r_new.centerx - 800) <= 1 and r_old.bottom == r_new.y - 10
        and r_old.w == w_want and px(r_new.x + 4, r_new.y + 4) == T.WARN and left == ["already running"],
        f"{r_old} {r_new}, left {left} (WingLab 'Mission accepted' 714..885 x 942..989)")

    # 12 dialog
    widgets = None
    try:
        from . import widgets
    except Exception as e:              # noqa: BLE001 -- reported by the rows that need it
        werr = f"{type(e).__name__}: {e}"
    if widgets is not None:
        def body_fn(ui):
            ui.sect_head("Four stages, in order, each one feeding the next.")
            ui.kv("1", "State the mission")
            ui.hairline()
            ui.hint("Every number here is carsim's own physics.")
        dlg = Dialog("About this pipeline", body_fn=body_fn)
        surf.fill(T.CANVAS)
        dlg.draw(surf, 1600, 1000, mouse=(-1, -1), now=0.0)
        dlg.draw(surf, 1600, 1000, mouse=(-1, -1), now=0.0)        # sized to its content
        card = dlg._card
        text_in = ink(dlg._body, T.WELL)
        consumed = dlg.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a, mod=0))
        esc = dlg.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0))
        closed_esc = not dlg.open
        dlg2 = Dialog("Keys and controller")
        dlg2.draw(surf, 1600, 1000)
        dlg2.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=dlg2._close.center))
        dlg3 = Dialog("x")
        dlg3.draw(surf, 1600, 1000)
        dlg3.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(5, 5)))
        rep("dialog: 620 px, centred, sized to its WorkUI body (padded 9); modal (consumes keys); ESC, the ✕ and "
            "the backdrop close it",
            card.w == 620 and abs(card.centerx - 800) <= 1 and card.h < 400 and text_in is not None
            and text_in.x - dlg._body.x in (9, 10, 11) and consumed and dlg.open is False and esc and closed_esc
            and not dlg2.open and not dlg3.open, f"card {card}, body ink {text_in}")
    else:
        rep("dialog: needs drive/cae/widgets.py (WorkUI) -- not importable", False, werr)

    # 13 tooltips through the real Overlay
    if widgets is not None:
        ov = widgets.Overlay()
        surf.fill(T.CANVAS)
        tb_ = pane(surf, (2, 60, 276, 547), "Simulation")
        tr = TreeDraw(ov)
        tip_row = (140, 86 + 21 * 11 + 8)
        drawn = []
        for t in (0.0, 0.3, 0.6):
            tr.draw(surf, tb_, _REPLICA_03["tree"], 0, tip_row)
            ov.draw(surf, t)
            below = pygame.Rect(3, 86 + 21 * 12, 600, 40)
            drawn.append(ink(below, T.PANEL) is not None and any(
                px(x, y) == T.TOOLTIP for x in range(below.x, below.right, 3) for y in range(below.y, below.bottom, 3)))
            pane(surf, (2, 60, 276, 547), "Simulation")
        rep("tooltips: a hovered row with a reason asks the shell's Overlay, which shows it after TOOLTIP_DELAY_S",
            tr.tip is not None and tr.tip[1] == "no completed run yet" and drawn == [False, False, True], f"{drawn}")
    else:
        rep("tooltips: need drive/cae/widgets.py (Overlay) -- not importable", False, werr)

    # 14 the mock shell at 1600x1000 against shot 03 (SH §1.2, §3.2, §6.2, §8.1)
    surf.fill(T.CANVAS)
    t0 = time.perf_counter()
    m = _mock_shell(surf, _REPLICA_03, now=0.0)
    ms = 1e3 * (time.perf_counter() - t0)
    R = m["R"]
    rect_ok = (tuple(R["tree"]) == (2, 60, 276, 547) and tuple(R["props"]) == (2, 609, 276, 366)
               and tuple(R["main"]) == (280, 60, 1318, 763) and tuple(R["tabs"]) == (281, 61, 1316, 25)
               and tuple(R["work"]) == (281, 86, 1316, 736) and tuple(R["output"]) == (280, 825, 1318, 150))
    def label_x(r):
        """Where a row's label starts: 43 / 56, less what an overflowing chip takes of the twisty box."""
        x = 43 + T.TREE_INDENT * r["level"]
        if r.get("chip"):
            over = x + T.text_w(r["label"], "sans", 12, r["selected"]) + 13 + T.text_w(r["chip"], "mono", 9.5) - 271
            x -= min(max(0, over), T.TWISTY_W - T.text_w("▾" if r["twisty"] == "open" else "▸", "sans", 9))
        return x
    want = [label_x(r) for r in rows]
    lab = [ink((want[i] - 3, 86 + 21 * i, 60, 21), T.ACCENT_FILL if rows[i]["selected"] else T.PANEL)
           for i in range(len(rows))]
    lab_ok = all(lb is not None and lb.x - want[i] in (0, 1, 2) for i, lb in enumerate(lab))
    cells_x = [r.x for r in m["cells"]]
    rep("mock shell 1600x1000: pane rects = WingLab's measured ones; tree labels at 43 / 56 on the 21 px pitch, the "
        "overflowing '2.8 Endplate' row slid left as WingLab's is; status cells right-aligned (MONO 11 is 7 px a "
        "glyph here, 6.6 in the browser)",
        rect_ok and lab_ok and (not sf or (abs(cells_x[-1] - 1523) <= 6 and abs(cells_x[-2] - 1466) <= 8
                                            and abs(lab[9].x - 36) <= 1)),
        f"labels x {sorted(set(lb.x for lb in lab if lb))} (2.8 Endplate {lab[9].x if lab[9] else None}, WingLab 36); "
        f"cells {cells_x} vs WingLab [1231, 1466, 1523]; frame {ms:.1f} ms")

    # 15 glyphs: every non-ASCII character in chrome's own strings is drawn
    pool = "▸▾…✕—·’" + "".join(str(v) for d in (_REPLICA_03, _MOCK_FRESH) for v in _strings(d))
    missing = sorted({c for c in pool if not c.isascii() and not T.glyph_ok(c, "sans")
                      and c not in "".join(T.ASCII_FALLBACK)})
    rep("every non-ASCII character in the chrome strings resolves in SANS (glyph or ASCII stand-in)", not missing,
        f"{missing}")

    # 16 cost of a full chrome frame
    t0 = time.perf_counter()
    for _ in range(10):
        _mock_shell(surf, _MOCK_FRESH, now=0.0)
    ms = 1e3 * (time.perf_counter() - t0) / 10
    rep("a full chrome frame (every widget, cached text) draws in < 8 ms", ms < 8.0, f"{ms:.2f} ms")

    # 17 set_scale reaches the chrome
    T.set_scale(1.5)
    try:
        s2 = pygame.Surface((2400, 1500))
        m2 = _mock_shell(s2, _MOCK_FRESH, now=0.0)
        rows2 = [r for _i, r, k in m2["tree"] if k == "row"]
        scaled = rows2[1].y - rows2[0].y == 32 and m2["R"]["tree"].w == 414 and m2["tabs"][0][1].h == 33
    finally:
        T.set_scale(1.0)
    rep("set_scale(1.5): the tree pitch, the left column and the tabs follow (32 / 414 / 33 px)", scaled)

    if out:
        os.makedirs(out, exist_ok=True)
        for name, content in (("chrome_03_replica_1600x1000.png", _REPLICA_03),
                              ("chrome_fresh_mock_1600x1000.png", _MOCK_FRESH)):
            s3 = pygame.Surface((1600, 1000))
            _mock_shell(s3, content, now=0.0)
            pygame.image.save(s3, os.path.join(out, name))
        s4 = pygame.Surface((1280, 800))
        _mock_shell(s4, _MOCK_FRESH, now=0.0)
        pygame.image.save(s4, os.path.join(out, "chrome_fresh_mock_1280x800.png"))
        if verbose:
            print(f"  mock frames: {out}")
    if verbose:
        print(f"  {'ALL PASS' if ok else 'FAILURES ABOVE'}: {n_ok}/{n_all} checks")
    return ok


def _strings(d):
    """Every string in a nested content dict (for the glyph row)."""
    if isinstance(d, str):
        yield d
    elif isinstance(d, dict):
        for v in d.values():
            yield from _strings(v)
    elif isinstance(d, (list, tuple)):
        for v in d:
            yield from _strings(v)


if __name__ == "__main__":
    import sys
    _out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else ""
    sys.exit(0 if self_check(out=_out) else 1)
