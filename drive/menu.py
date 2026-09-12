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
    select              run the highlighted item -> returns its action
    back / menu         close                    -> returns 'resume'

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

NAV_HINT = [("UP / DOWN", "move"), ("ENTER", "select"), ("ESC", "back")]
NAV_HINT_PAD = [("UP / DOWN", "move  (d-pad, stick)"), ("ENTER", "select  (CROSS)"),
                ("ESC", "back  (CIRCLE, OPTIONS)")]


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
        elif cmd == "select":
            a = self.action()
            self.open = False
            return a or "resume"
        elif cmd in ("back", "menu"):
            self.open = False
            return "resume"
        return None

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
        items_h = 96 * u + 36 * u * len(self.items) + (len(NAV_HINT) + 1) * row_h + 30 * u
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

        # items
        y = y0 + 96 * u
        for i, (label, _) in enumerate(self.items):
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
            y += 36 * u
        hint = NAV_HINT_PAD if any("PAD" in (t or "") or "DUALSENSE" in (t or "")
                                   for t, _ in self.sections) else NAV_HINT
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
