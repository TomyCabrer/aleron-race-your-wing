"""drive/menu.py -- the pause / help menu, shared by the drive and the garage.

ESC on the keyboard or OPTIONS on the pad opens it. It pauses whatever is
underneath (the sim's accumulator, the garage's pad editing), lists the
actions the user can take (resume, reset, garage, quit) and prints every
control for the keyboard and for whichever pad is attached.

Pure UI. No physics, no wall clock, no imports from the rest of the package:
`drive.input` and `drive.garage` both import this module, and `render.py`
draws it duck-typed through `HudData.menu` without importing it at all.

Command vocabulary (the strings `Menu.handle` understands):

    nav_up / nav_down   move the cursor
    nav_left/nav_right  step the highlighted item's value -> returns
                        'prev:<action>' / 'next:<action>', menu stays open
    select              run the highlighted item -> returns its action
    back / menu         close                    -> returns 'resume'
    hover:X:Y           the MOUSE: the row under it becomes the cursor
    click:X:Y           a left press ARMS the row under it ...
    release:X:Y         ... and the release on that same row runs it (=
                        select). A press within CLICK_GUARD_DRAWS frames of
                        a page change is ignored, so the second click of a
                        double-click cannot run a row of the page the first
                        one opened. A right click is sent as 'menu' by the
                        input layer, the wheel as nav_up / nav_down.

A page with more rows than fit shows a scrolling window of them.

Everything else is ignored so a caller can pass its whole command stream.
"""

from __future__ import annotations

import pygame

FONT_NAMES = ("Menlo", "Monaco", "DejaVu Sans Mono", "Courier New")

C_OVERLAY = (0, 0, 0, 150)
C_PANEL = (16, 17, 20, 238)
C_BORDER = (60, 64, 70)
C_TEXT = (232, 234, 238)
C_DIM = (139, 144, 153)
C_ACCENT = (255, 140, 43)
C_SEL_BG = (40, 44, 52)
C_KEY = (217, 206, 85)
C_SECTION = (79, 163, 255)

#: frames after a show() before a mouse press arms a row (~0.25 s at 60 fps)
CLICK_GUARD_DRAWS = 15

NAV_HINT = [("UP / DOWN", "move"), ("LEFT / RIGHT", "change value"),
            ("ENTER", "select"), ("ESC", "back")]
NAV_HINT_PAD = [("UP / DOWN", "move  (d-pad, stick)"),
                ("LEFT / RIGHT", "change value  (d-pad, stick)"),
                ("ENTER", "select  (CROSS)"), ("ESC", "back  (CIRCLE, OPTIONS)")]


class StickNav:
    """A stick axis as up/down EDGES: push past 0.6, re-arm under 0.3.

    Without the hysteresis a stick held up scrolls one row per frame."""

    PUSH = 0.6
    REARM = 0.3

    def __init__(self):
        self.armed = True

    def poll(self, y: float) -> str | None:
        """SDL sign convention: +y is DOWN."""
        if self.armed:
            if y <= -self.PUSH:
                self.armed = False
                return "nav_up"
            if y >= self.PUSH:
                self.armed = False
                return "nav_down"
        elif abs(y) < self.REARM:
            self.armed = True
        return None


class Menu:
    """Items + cursor + help columns. `show()`, `handle(cmd)`, `draw(screen)`.

    `items`    list of (label, action) - action is the string handed back by
               `handle('select')`.
    `sections` list of (title, rows) with rows = [(key, what), ...]; drawn as
               columns to the right of the items.
    """

    def __init__(self, title: str = "PAUSED", items=(), sections=(),
                 footer: str = "", subtitle: str = "", note: str = ""):
        self.title = title
        self.items = list(items)
        self.sections = list(sections)
        self.footer = footer
        self.subtitle = subtitle
        self.note = note
        self.idx = 0
        self.open = False
        self.columns = 2                  # help columns: 2 (pause) or 1 (settings)
        self._ui = None
        self._fonts = {}
        self._cache = {}
        self._surfs = {}                  # overlay / panel surfaces by size
        self._rows = []                   # the item rows as drawn: (i, (x, y, w, h))
        self._top = 0                     # first row of the scrolling window
        self._armed = None                # the row a mouse press armed
        self._draws = 0                   # frames drawn since the last show()

    # -- state --------------------------------------------------------------
    def show(self, items=None, sections=None, subtitle=None, note=None,
             footer=None, title=None, idx=0, columns=None) -> None:
        """Open (or re-open) the menu. `idx` keeps the cursor where it was
        when a settings page re-shows itself after a value is cycled;
        `columns=1` stacks every help section (and the note) in one column
        to the right of the items, for pages with long item labels."""
        if items is not None:
            self.items = list(items)
        if sections is not None:
            self.sections = list(sections)
        if subtitle is not None:
            self.subtitle = subtitle
        if note is not None:
            self.note = note
        if footer is not None:
            self.footer = footer
        if title is not None:
            self.title = title
        if columns is not None:
            self.columns = 1 if int(columns) <= 1 else 2
        self.idx = (int(idx) % len(self.items)) if self.items else 0
        self.open = True
        self._armed = None
        self._draws = 0

    def hide(self) -> None:
        self.open = False

    def move(self, d: int) -> None:
        if self.items:
            self.idx = (self.idx + d) % len(self.items)

    def action(self) -> str | None:
        return self.items[self.idx][1] if self.items else None

    def handle(self, cmd: str) -> str | None:
        """Returns the action to run, or None. Closes itself on select/back."""
        if not self.open:
            return None
        if cmd == "nav_up":
            self.move(-1)
        elif cmd == "nav_down":
            self.move(+1)
        elif cmd in ("nav_left", "nav_right"):
            # step the highlighted value; the page decides what that means
            # and re-shows itself, so the menu stays open
            a = self.action()
            if a:
                return ("prev:" if cmd == "nav_left" else "next:") + a
        elif cmd == "select":
            a = self.action()
            self.open = False
            return a or "resume"
        elif cmd in ("back", "menu"):
            self.open = False
            return "resume"
        elif cmd.startswith(("hover:", "click:", "release:")):
            i = self.hit(cmd)
            if cmd.startswith("release:"):
                armed, self._armed = self._armed, None
                if i is None or i != armed:
                    return None
                self.idx = i
                a = self.action()
                self.open = False
                return a or "resume"
            if i is None:
                return None
            self.idx = i
            if cmd.startswith("click:"):
                self._armed = i if self._draws >= CLICK_GUARD_DRAWS else None
        return None

    # -- the mouse ------------------------------------------------------------
    def hit(self, cmd_or_xy):
        """The item row under a point ('hover:X:Y', 'click:X:Y' or (x, y)),
        as last drawn; None off every row (or before the first draw)."""
        try:
            if isinstance(cmd_or_xy, str):
                _, xs, ys = cmd_or_xy.split(":")
                x, y = float(xs), float(ys)
            else:
                x, y = float(cmd_or_xy[0]), float(cmd_or_xy[1])
        except (ValueError, TypeError):
            return None
        for i, (rx, ry, rw, rh) in self._rows:
            if rx <= x < rx + rw and ry <= y < ry + rh and i < len(self.items):
                return i
        return None

    def row_centre(self, i: int):
        """(x, y) in screen pixels of item row i as last drawn (it must be in
        the visible window)."""
        rx, ry, rw, rh = dict(self._rows)[i]
        return int(rx + rw // 2), int(ry + rh // 2)

    # -- drawing --------------------------------------------------------------
    def _font(self, size: int, bold: bool = False):
        key = (size, bold)
        f = self._fonts.get(key)
        if f is None:
            if not pygame.font.get_init():
                pygame.font.init()
            for name in FONT_NAMES:
                try:
                    f = pygame.font.SysFont(name, size, bold=bold)
                    if f is not None:
                        break
                except Exception:
                    continue
            if f is None:
                f = pygame.font.Font(None, size + 4)
            self._fonts[key] = f
        return f

    def _txt(self, s, font, col):
        key = (s, id(font), col)
        surf = self._cache.get(key)
        if surf is None:
            if len(self._cache) > 512:
                self._cache.clear()
            surf = font.render(s, True, col)
            self._cache[key] = surf
        return surf

    def _blit(self, screen, s, x, y, font, col=C_TEXT) -> int:
        surf = self._txt(s, font, col)
        screen.blit(surf, (int(x), int(y)))
        return surf.get_width()

    @staticmethod
    def _wrap(text: str, font, max_w: float) -> list[str]:
        """Greedy word wrap by rendered width."""
        out, line = [], ""
        for word in text.split():
            cand = (line + " " + word).strip()
            if line and font.size(cand)[0] > max_w:
                out.append(line)
                line = word
            else:
                line = cand
        if line:
            out.append(line)
        return out or [""]

    def draw(self, screen: pygame.Surface) -> None:
        """Dim the frame underneath and draw the panel. Nothing if closed."""
        if not self.open:
            return
        W, H = screen.get_size()
        u = min(W / 1280.0, H / 800.0)
        if self._ui != u:
            self._ui, self._fonts, self._cache, self._surfs = u, {}, {}, {}
        f_title = self._font(int(round(30 * u)), bold=True)
        f_item = self._font(int(round(18 * u)))
        f_lbl = self._font(int(round(14 * u)))
        f_sec = self._font(int(round(14 * u)), bold=True)

        # the panel height follows the tallest help column; the help columns
        # start to the right of the widest item (a settings page has long
        # value labels, a pause page has short verbs)
        row_h = 21 * u
        w = min(1160 * u, W - 16 * u)
        items_w = max(310 * u, max((f_item.size("> " + lbl)[0] for lbl, _ in self.items),
                                   default=0) + 24 * u)
        items_w = min(items_w, w - 56 * u)
        c0 = max(350 * u, 28 * u + items_w + 22 * u)
        col_px = (c0, c0 + 410 * u)
        col_h = [0.0, 0.0]
        placed = []                       # (col, y_offset, title, rows)
        one = (self.columns == 1)
        for title, rows in self.sections:
            c = 0 if (one or col_h[0] <= col_h[1]) else 1
            placed.append((c, col_h[c], title, rows))
            col_h[c] += (len(rows) + 1) * row_h + 14 * u
        if self.note:
            c = 0 if (one or col_h[0] <= col_h[1]) else 1
            lines = self._wrap(self.note, f_lbl, w - col_px[c] - 24 * u)
            placed.append((c, col_h[c], None, [(ln, "") for ln in lines]))
            col_h[c] += (len(lines) + 1) * row_h
        # a list longer than the window scrolls: only `vis` rows are drawn,
        # the window follows the cursor (the PICK page's library can grow)
        hint_h = (len(NAV_HINT) + 1) * row_h + 30 * u
        n = len(self.items)
        cap = max(3, int((H - 16 * u - 96 * u - hint_h - 44 * u) // (36 * u)))
        vis = min(n, cap)
        if n > vis:
            self._top = min(max(self._top, self.idx - vis + 1), self.idx, n - vis)
        else:
            self._top = 0
        items_h = 96 * u + 36 * u * vis + hint_h
        h = max(items_h, 96 * u + max(col_h) + 44 * u)
        h = min(h, H - 16 * u)
        x0, y0 = (W - w) * 0.5, (H - h) * 0.5

        # the two alpha surfaces are built once per size: a fresh 1280x800
        # SRCALPHA fill every frame was 4 ms on its own
        ov = self._surfs.get(("ov", W, H))
        if ov is None:
            ov = pygame.Surface((W, H), pygame.SRCALPHA)
            ov.fill(C_OVERLAY)
            self._surfs[("ov", W, H)] = ov
        screen.blit(ov, (0, 0))
        pw, ph = int(w), int(h)
        panel = self._surfs.get(("panel", pw, ph))
        if panel is None:
            panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
            panel.fill(C_PANEL)
            pygame.draw.rect(panel, C_BORDER, panel.get_rect(), 1)
            pygame.draw.rect(panel, C_ACCENT, (0, 0, pw, int(3 * u)))
            self._surfs[("panel", pw, ph)] = panel
        screen.blit(panel, (int(x0), int(y0)))

        # title + subtitle
        self._blit(screen, self.title, x0 + 28 * u, y0 + 18 * u, f_title, C_TEXT)
        if self.subtitle:
            self._blit(screen, self.subtitle, x0 + 28 * u, y0 + 58 * u, f_lbl, C_DIM)

        # items (the visible window of them)
        y = y0 + 96 * u
        self._rows = []
        self._draws += 1
        top = self._top
        for i in range(top, top + vis):
            label = self.items[i][0]
            self._rows.append((i, (int(x0 + 16 * u), int(y - 6 * u), int(items_w), int(34 * u))))
            sel = (i == self.idx)
            if sel:
                pygame.draw.rect(screen, C_SEL_BG,
                                 (int(x0 + 16 * u), int(y - 6 * u),
                                  int(items_w), int(32 * u)))
                pygame.draw.rect(screen, C_ACCENT,
                                 (int(x0 + 16 * u), int(y - 6 * u),
                                  int(4 * u), int(32 * u)))
            self._blit(screen, ("> " if sel else "  ") + label, x0 + 28 * u, y,
                       f_item, C_TEXT if sel else C_DIM)
            if (i == top and top > 0) or (i == top + vis - 1 and top + vis < n):
                more = f"^ {top} more" if i == top and top > 0 else f"v {n - top - vis} more"
                self._blit(screen, more, x0 + 16 * u + items_w - f_lbl.size(more)[0] - 8 * u,
                           y + 4 * u, f_lbl, C_DIM)
            y += 36 * u
        hint = NAV_HINT_PAD if any("PAD" in (t or "") or "DUALSENSE" in (t or "")
                                   for t, _ in self.sections) else NAV_HINT
        if not any(a.startswith("set:") for _, a in self.items):
            hint = [row for row in hint if not row[0].startswith("LEFT")]
        y += 6 * u
        for key, what in hint:
            self._blit(screen, key, x0 + 28 * u, y, f_lbl, C_DIM)
            self._blit(screen, what, x0 + 28 * u + 96 * u, y, f_lbl, C_DIM)
            y += row_h

        # help columns; the key column widens to the section's longest key
        for c, yoff, title, rows in placed:
            x = x0 + col_px[c]
            yy = y0 + 96 * u + yoff
            key_w = max(118 * u, max((f_lbl.size(k)[0] for k, wh in rows if wh),
                                     default=0) + 12 * u)
            if title:
                self._blit(screen, title, x, yy, f_sec, C_SECTION)
                yy += row_h
            for key, what in rows:
                if what:
                    self._blit(screen, key, x, yy, f_lbl, C_KEY)
                    self._blit(screen, what, x + key_w, yy, f_lbl, C_TEXT)
                else:
                    self._blit(screen, key, x, yy, f_lbl, C_DIM)
                yy += row_h

        if self.footer:
            self._blit(screen, self.footer, x0 + 28 * u, y0 + h - 30 * u, f_lbl, C_DIM)


# --------------------------------------------------------------------------- #
def self_check(verbose: bool = True) -> bool:
    import os
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and passed
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}: {msg}")

    m = Menu("PAUSED", [("Resume", "resume"), ("Reset", "reset"), ("Quit", "quit")],
             [("KEYBOARD", [("UP", "throttle"), ("ESC", "this menu")])])
    rep("closed menu ignores input", m.handle("select") is None)
    m.show()
    rep("show() opens at the first item", (m.open, m.idx, m.action()), "")
    m.handle("nav_down")
    m.handle("nav_down")
    rep("cursor wraps", m.action() == "quit", m.action())
    m.handle("nav_down")
    rep("wrap to top", m.idx == 0, str(m.idx))
    m.handle("nav_up")
    a = m.handle("select")
    rep("select returns the action and closes", (a, m.open) == ("quit", False), f"{a} {m.open}")
    m.show()
    rep("back -> 'resume' and closes", (m.handle("back"), m.open) == ("resume", False), "")
    m.show()
    rep("'menu' while open -> 'resume'", m.handle("menu") == "resume", "")
    rep("unknown commands ignored", m.handle("wing") is None, "")
    m.show(idx=1)
    rep("nav_left / nav_right -> prev:/next: and stay open",
        (m.handle("nav_left"), m.handle("nav_right"), m.open)
        == ("prev:reset", "next:reset", True), "")
    m.show(items=[("A", "a"), ("B", "b"), ("C", "c")], idx=2, title="SETTINGS")
    rep("show(idx=, title=) keeps the cursor and retitles",
        (m.idx, m.action(), m.title) == (2, "c", "SETTINGS"), f"{m.idx} {m.title}")
    m.show(idx=7)
    rep("show(idx) wraps", m.idx == 1, str(m.idx))

    s = StickNav()
    seq = [s.poll(y) for y in (0.0, -0.7, -0.9, -0.2, -0.8, 0.0, 0.8, 0.9, 0.0)]
    rep("stick nav edges with hysteresis",
        seq == [None, "nav_up", None, None, "nav_up", None, "nav_down", None, None],
        str(seq))

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    scr = pygame.display.set_mode((1280, 800))
    scr.fill((27, 29, 33))
    m.show()
    import time
    t0 = time.perf_counter()
    m.draw(scr)                            # cold: SysFont scans the system once
    cold = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    for _ in range(20):
        m.draw(scr)
    ms = (time.perf_counter() - t0) * 1e3 / 20.0
    rep("draw budget", ms < 4.0, f"{ms:.2f} ms/frame warm ({cold:.0f} ms cold font scan)")
    px = scr.get_at((640, 400))[:3]
    rep("panel drawn over the frame", px != (27, 29, 33), str(px))
    # the mouse: hover moves the cursor, a click runs the row, off-row
    # points do nothing
    x2, y2 = m.row_centre(2)
    m.show(idx=0)
    m.draw(scr)
    m.handle(f"hover:{x2}:{y2}")
    hov = m.idx == 2 and m.open
    x1, y1 = m.row_centre(1)
    early = (m.handle(f"click:{x1}:{y1}"), m.handle(f"release:{x1}:{y1}"))   # right after show
    for _ in range(CLICK_GUARD_DRAWS):
        m.draw(scr)
    miss = m.handle("click:3:3") is None and m.handle("release:3:3") is None and m.open
    drag = (m.handle(f"click:{x1}:{y1}"), m.handle(f"release:{x2}:{y2}"))    # off the row
    got = (m.handle(f"click:{x1}:{y1}"), m.handle(f"release:{x1}:{y1}"))
    rep("mouse: hover moves the cursor; press + release on one row runs it; a press "
        "right after a page change, a miss or a drag off the row does nothing",
        hov and early == (None, None) and miss and drag == (None, None)
        and got == (None, "b") and not m.open,
        f"hover {hov} early {early} miss {miss} drag {drag} click -> {got}")
    many = Menu("PICK", [(f"build {k:02d}", f"b{k}") for k in range(40)])
    many.show(idx=0)
    many.draw(scr)
    shown0 = [i for i, _ in many._rows]
    many.show(idx=33)
    many.draw(scr)
    shown1 = [i for i, _ in many._rows]
    bottom = max(r[1] + r[3] for _, r in many._rows)
    rep("a long list scrolls: the window follows the cursor and stays on screen",
        shown0[0] == 0 and 33 in shown1 and len(shown1) < 40 and bottom <= 800
        and many.hit(many.row_centre(33)) == 33,
        f"{len(shown1)} of 40 rows shown, {shown1[0]}..{shown1[-1]}, bottom {bottom} px")
    m.hide()
    scr.fill((27, 29, 33))
    m.draw(scr)
    rep("hidden menu draws nothing", scr.get_at((640, 400))[:3] == (27, 29, 33), "")
    if verbose:
        print("  ALL PASS" if ok else "  FAILURES ABOVE")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
