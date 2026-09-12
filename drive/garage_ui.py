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
class Param:
    """One editable row. `get()`/`set(v)` talk to the model; `kind`:
    'float' (LEFT/RIGHT step, SHIFT fine), 'int', 'choice' (cycles
    `choices`), 'bool', 'action' (ENTER calls `set(None)`), 'label'."""

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
        self.enabled = enabled

    def value_text(self) -> str:
        v = self.get() if self.get is not None else ""
        if self.kind == "bool":
            return "on" if v else "off"
        if self.kind == "action":
            return ""
        if self.kind in ("float", "int"):
            try:
                return self.fmt.format(v) + (f" {self.unit}" if self.unit else "")
            except (ValueError, TypeError):
                return str(v)
        return str(v)

    def adjust(self, direction: int, fine: bool = False) -> bool:
        if not self.enabled or self.set is None:
            return False
        if self.kind == "float":
            v = float(self.get()) + direction * (self.fine if fine else self.step)
            if self.lo is not None:
                v = max(v, self.lo)
            if self.hi is not None:
                v = min(v, self.hi)
            self.set(v)
            return True
        if self.kind == "int":
            v = int(self.get()) + direction
            if self.lo is not None:
                v = max(v, int(self.lo))
            if self.hi is not None:
                v = min(v, int(self.hi))
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
    def __init__(self, params, title: str = ""):
        self.params = list(params)
        self.title = title
        self.idx = 0
        self.scroll = 0

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

    def draw(self, screen, text: Text, rect, row_h: int = 22, size: int = 14, focus: bool = True):
        r = pygame.Rect(rect)
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
            if sel:
                pygame.draw.rect(screen, C_SEL_BG, (r.x + 4, y - 2, r.w - 8, row_h))
                pygame.draw.rect(screen, C_ACCENT, (r.x + 4, y - 2, 3, row_h))
            col = C_TEXT if p.enabled else C_DIM
            if p.kind == "action":
                text.blit(screen, ("> " if sel else "  ") + p.label, r.x + 12, y + 2, size,
                          C_ACCENT if p.enabled else C_DIM)
            else:
                text.blit(screen, p.label, r.x + 12, y + 2, size, col if not sel else C_TEXT)
                vt = p.value_text()
                if sel and p.kind in ("float", "int", "choice", "bool"):
                    vt = "< " + vt + " >"
                text.blit(screen, vt, val_x, y + 2, size, C_KEY if sel else col, right=True)
            y += row_h
        if self.params and self.scroll + n_vis < len(self.params):
            text.blit(screen, "...", r.right - 30, r.bottom - 16, size - 2, C_DIM)
        p = self.current()
        return p.help if (p and focus) else ""


# --------------------------------------------------------------------------- #
class ListBox:
    """Rows of (label, sub, tag) with a cursor and scrolling."""

    def __init__(self, items=(), title: str = ""):
        self.items = list(items)
        self.title = title
        self.idx = 0
        self.scroll = 0

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

    def draw(self, screen, text: Text, rect, row_h: int = 34, size: int = 14, focus: bool = True,
             empty: str = "(empty)"):
        r = pygame.Rect(rect)
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
            if sel:
                pygame.draw.rect(screen, C_SEL_BG, (r.x + 4, y - 2, r.w - 8, row_h - 2))
                pygame.draw.rect(screen, C_ACCENT, (r.x + 4, y - 2, 3, row_h - 2))
            tag = it[2] if len(it) > 2 else ""
            text.blit(screen, ("> " if sel else "  ") + str(it[0]), r.x + 12, y + 1, size,
                      C_TEXT if sel else C_DIM)
            if tag:
                text.blit(screen, tag, r.right - 12, y + 1, size - 2, C_KEY if sel else C_DIM, right=True)
            if len(it) > 1 and it[1]:
                text.blit(screen, str(it[1])[:70], r.x + 26, y + 16, size - 3, C_DIM)
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

    def line(self, xs, ys, col=C_ACCENT, width: int = 2, closed: bool = False):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if xs.size < 2:
            return
        pts = [self._px(x, y) for x, y in zip(xs, ys)]
        clip = self.screen.get_clip()
        self.screen.set_clip(self.rect)
        pygame.draw.lines(self.screen, col, closed, pts, width)
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
    events to `handle(ev)` -> 'ok' | 'cancel' | None, draw last."""

    def __init__(self):
        self.open_ = False
        self.title = ""
        self.value = ""
        self.hint = ""

    @property
    def open(self) -> bool:
        return self.open_

    def show(self, title: str, initial: str = "", hint: str = "ENTER ok   ESC cancel") -> None:
        self.open_, self.title, self.value, self.hint = True, title, initial, hint

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
            self.value = self.value[:-1]
            return None
        ch = getattr(ev, "unicode", "")
        if ch and ch.isprintable() and len(self.value) < 32:
            self.value += ch
        return None

    def draw(self, screen, text: Text) -> None:
        if not self.open_:
            return
        W, H = screen.get_size()
        w, h = 520, 110
        r = panel(screen, (W // 2 - w // 2, H // 2 - h // 2, w, h), alpha=240, accent=True)
        text.blit(screen, self.title, r.x + 16, r.y + 14, 14, C_SECTION, bold=True)
        pygame.draw.rect(screen, C_SEL_BG, (r.x + 16, r.y + 40, w - 32, 30))
        text.blit(screen, self.value + "_", r.x + 24, r.y + 46, 16, C_TEXT)
        text.blit(screen, self.hint, r.x + 16, r.y + 82, 12, C_DIM)


def key_hint_bar(screen, text: Text, rect, hints, pad_hints=None, title: str = ""):
    """The bottom bar: 'KEY what | KEY what ...' in the menu's key colour."""
    r = panel(screen, rect)
    x, y = r.x + 10, r.y + 6
    if title:
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
