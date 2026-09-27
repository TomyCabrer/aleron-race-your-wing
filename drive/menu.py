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

A help row too long for its column wraps under its text column, and the
panel grows by the extra lines; the note wraps the same way. A second help
column narrower than HELP_MIN_W px folds into the first (the garage's long
rows push its help far right), and a note with no such room right of the
items goes under their key legend instead.

A row can carry a colour SWATCH -- the Settings page's Paint row shows the
paint it names -- through a side channel, `show(..., swatches={action: rgb})`:
a small filled square drawn right after that row's label. The rows stay
(label, action) 2-tuples, which is what every caller unpacks.

A help row whose text is a `Dim` (a plain str to everything else) is drawn
dim, its key too: the tutorial's finished steps under the ones to come. One
whose text is a `Warn` is drawn amber, its key too: a challenge page's
settings that rule out stars (task 45). A page whose help column is full
(a challenge's) puts its note under the items' key legend itself:
`show(note_under=True)`.

Everything else is ignored so a caller can pass its whole command stream.
"""

from __future__ import annotations

import re

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
C_SWATCH_EDGE = (206, 210, 216)    # a swatch's 1 px outline: a dark paint still reads
C_WARN = (255, 176, 46)            # a Warn help row: amber, apart from C_KEY's yellow

#: frames after a show() before a mouse press arms a row (~0.25 s at 60 fps)
CLICK_GUARD_DRAWS = 15

#: the narrowest a help column or the note may be, px at 1280x800: under it
#: the garage's note came out one word a line, over its key table
HELP_MIN_W = 260

NAV_HINT = [("UP / DOWN", "move"), ("LEFT / RIGHT", "change value"),
            ("ENTER", "select"), ("ESC", "back")]
NAV_HINT_PAD = [("UP / DOWN", "move  (d-pad, stick)"),
                ("LEFT / RIGHT", "change value  (d-pad, stick)"),
                ("ENTER", "select  (CROSS)"), ("ESC", "back  (CIRCLE, OPTIONS)")]


def footer_says(footer: str, key: str) -> str | None:
    """What a page's footer says `key` does, so the key legend agrees with it:
    'ENTER / CROSS select   ESC / CIRCLE not now' -> 'not now' for 'ESC' (the
    words up to the next run of 2+ spaces). None when the footer is silent."""
    m = re.search(rf"\b{key} / [A-Z]+ (.+?)(?: {{2,}}|$)", footer or "")
    return m.group(1).strip() if m else None


class Dim(str):
    """A help row's text drawn dim (C_DIM), its key too, not in C_TEXT /
    C_KEY. A str everywhere else, so the rows stay (key, what) 2-tuples."""
    __slots__ = ()


class Warn(str):
    """A help row's text drawn amber (C_WARN), its key too: what a page
    warns about (a challenge's settings that rule out stars, task 45). A str
    everywhere else, so the rows stay (key, what) 2-tuples."""
    __slots__ = ()


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
    `swatches` {action: (r, g, b)}: a colour swatch after that row's label.
    """

    def __init__(self, title: str = "PAUSED", items=(), sections=(),
                 footer: str = "", subtitle: str = "", note: str = "", swatches=None):
        self.title = title
        self.items = list(items)
        self.sections = list(sections)
        self.swatches = dict(swatches or {})
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
        self._last_text_rects = []        # (text, Rect) of every text the last draw() blitted
        self._last_panel = None           # the panel's Rect as last drawn
        self.art = None                   # a page's own drawing (show(art=))
        self.art_h = 0.0
        self.typed = frozenset()          # rows that take typed digits (show(typed=))
        self.help_for = None              # idx -> sections: help for the row (show(help_for=))
        self.note_under = False           # the note under the items (show(note_under=))

    # -- state --------------------------------------------------------------
    def show(self, items=None, sections=None, subtitle=None, note=None,
             footer=None, title=None, idx=0, columns=None, art=None,
             art_h: float = 0.0, typed=(), help_for=None, swatches=None,
             note_under: bool = False) -> None:
        """Open (or re-open) the menu. `idx` keeps the cursor where it was
        when a settings page re-shows itself after a value is cycled;
        `columns=1` stacks every help section (and the note) in one column
        to the right of the items, for pages with long item labels.
        `art(screen, rect)` is the page's own drawing (the LAP RESULTS
        page's card), given `art_h` px at the top of the first help column;
        every show() sets it, so the next page never inherits it. `typed`:
        the actions of the rows that take the keyboard's digits (a digit on
        one returns 'type:<d>:<action>'); set by every show() too.
        `swatches` {action: (r, g, b)} draws a small colour swatch right
        after the label of the row with that action. The swatches belong to
        the rows: new `items` without `swatches` clear them (the pause page
        shown after the settings page has none), a re-show that keeps the
        items keeps them. `note_under` puts the note under the items' key
        legend, not under the help (a challenge's page: its help column is
        full, the items' has room); set by every show() too."""
        self.art = art
        self.note_under = bool(note_under)
        self.art_h = float(art_h) if art is not None else 0.0
        self.typed = frozenset(typed or ())
        #: `help_for(idx)` -> the help sections for the highlighted row (the
        #: settings page: what that setting does, with its pad buttons);
        #: None = the page's fixed `sections`. Set by every show() too.
        self.help_for = help_for
        if items is not None:
            self.items = list(items)
            if swatches is None:
                self.swatches = {}
        if swatches is not None:
            self.swatches = dict(swatches)
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
        elif cmd.startswith("digit:"):
            # a typed digit: only on a row the page said takes one; the
            # menu stays open, the page re-shows itself
            a = self.action()
            if a and a in self.typed:
                return f"type:{cmd[6:7]}:{a}"
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
        self._last_text_rects.append((s, surf.get_rect(topleft=(int(x), int(y)))))
        return surf.get_width()

    @staticmethod
    def _wrap(text: str, font, max_w: float) -> list[str]:
        """Greedy word wrap by rendered width; a single word wider than the
        line (a file path) is cut where it fills one."""
        out, line = [], ""
        for word in text.split():
            cand = (line + " " + word).strip()
            if not line or font.size(cand)[0] > max_w:
                if line:
                    out.append(line)
                while len(word) > 1 and font.size(word)[0] > max_w:
                    k = len(word) - 1
                    while k > 1 and font.size(word[:k])[0] > max_w:
                        k -= 1
                    out.append(word[:k])
                    word = word[k:]
                line = word
            else:
                line = cand
        if line:
            out.append(line)
        return out or [""]

    def _lines(self, text: str, font, max_w: float) -> list[str]:
        """A help row (or the note) as the lines it takes: itself when it
        fits (its double spaces kept), else wrapped. Cached: the layout runs
        every frame."""
        key = ("wrap", text, id(font), int(max_w))
        got = self._cache.get(key)
        if got is None:
            if len(self._cache) > 512:
                self._cache.clear()
            got = [text] if font.size(text)[0] <= max_w else self._wrap(text, font, max_w)
            self._cache[key] = got
        return got

    @staticmethod
    def _key_w(rows, font, u: float) -> float:
        """A section's key column: its widest key + a gap, at least 118 px;
        but step numbers (every key a number of 3 characters or fewer: the
        tutorial's ' 1' .. '13') get just their width + 16 px, not a 110 px
        gap before their text. A '--' placeholder keeps the 118 px, in line
        with the sections around it."""
        keys = [k for k, wh in rows if wh]
        widest = max((font.size(k)[0] for k in keys), default=0)
        nums = [k.strip() for k in keys if k.strip()]
        if nums and all(len(k) <= 3 and k.isdigit() for k in nums):
            return widest + 16 * u
        return max(118 * u, widest + 12 * u)

    def draw(self, screen: pygame.Surface) -> None:
        """Dim the frame underneath and draw the panel. Nothing if closed."""
        if not self.open:
            return
        W, H = screen.get_size()
        u = min(W / 1280.0, H / 800.0)
        if self._ui != u:
            self._ui, self._fonts, self._cache, self._surfs = u, {}, {}, {}
        self._last_text_rects = []
        f_title = self._font(int(round(30 * u)), bold=True)
        f_item = self._font(int(round(18 * u)))
        f_lbl = self._font(int(round(14 * u)))
        f_sec = self._font(int(round(14 * u)), bold=True)

        # the panel height follows the tallest help column; the help columns
        # start to the right of the widest item (a settings page has long
        # value labels, a pause page has short verbs)
        row_h = 21 * u
        w = min(1160 * u, W - 16 * u)
        sw = max(6, int(round(14 * u)))           # a swatch's side, px
        sw_gap = int(round(8 * u))
        items_w = max(310 * u, max((f_item.size("> " + lbl)[0]
                                    + (sw_gap + sw if a in self.swatches else 0)
                                    for lbl, a in self.items), default=0) + 24 * u)
        n_rows = len(self.items)
        cap0 = max(3, int((H - 16 * u - 96 * u - (len(NAV_HINT) + 1) * row_h - 30 * u
                           - 44 * u) // (36 * u)))
        if n_rows > cap0:                 # a list that scrolls: room for "v N more"
            items_w += f_lbl.size(f"v {n_rows} more")[0] + 12 * u   # beside the widest label
        items_w = min(items_w, w - 56 * u)
        c0 = max(350 * u, 28 * u + items_w + 22 * u)
        col_px = (c0, c0 + 410 * u)
        col_h = [0.0, 0.0]
        if self.art is not None:          # the page's drawing heads column 0
            col_h[0] = self.art_h + 14 * u
        placed = []                       # (col, y_offset, title, key_w, [(key, what, lines)])
        one = (self.columns == 1)
        # a second column narrower than HELP_MIN_W is no column: the garage's
        # long rows push its help far right, and its note came out one word
        # a line over the key table. Everything stacks in the first.
        if not one and w - col_px[1] - 24 * u < HELP_MIN_W * u:
            one = True
        # where a column's text stops: the next column, or the panel's edge
        right = ((col_px[1] - 16 * u) if not one else (w - 24 * u), w - 24 * u)
        secs = self.sections
        if self.help_for is not None:
            try:
                secs = list(self.help_for(self.idx))
            except Exception:             # noqa: BLE001 -- the help never takes the menu down
                secs = self.sections
        for title, rows in secs:
            c = 0 if (one or col_h[0] <= col_h[1]) else 1
            key_w = self._key_w(rows, f_lbl, u)
            # a row too long for its column wraps under its text column (a
            # key-only row under itself); the column grows by its lines
            laid = [(key, what, self._lines(what or key, f_lbl,
                                            max(120 * u, right[c] - col_px[c]
                                                - (key_w if what else 0))))
                    for key, what in rows]
            placed.append((c, col_h[c], title, key_w, laid))
            col_h[c] += (sum(len(ls) for _, _, ls in laid) + 1) * row_h + 14 * u
        note_lines = []                   # the note under the items' key legend
        if self.note:
            c = 0 if (one or col_h[0] <= col_h[1]) else 1
            if right[c] - col_px[c] >= HELP_MIN_W * u and not self.note_under:
                lines = self._lines(self.note, f_lbl, right[c] - col_px[c])
                placed.append((c, col_h[c], None, 0, [("", "", lines)]))
                col_h[c] += (len(lines) + 1) * row_h
            else:                         # no room right of the items (or the page
                #                               asked, `note_under`): under them
                note_lines = self._lines(self.note, f_lbl, items_w - 12 * u)
        # a list longer than the window scrolls: only `vis` rows are drawn,
        # the window follows the cursor (the PICK page's library can grow)
        hint_h = (len(NAV_HINT) + 1) * row_h + 30 * u
        if note_lines:
            hint_h += len(note_lines) * row_h + 10 * u
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
        self._last_panel = pygame.Rect(int(x0), int(y0), pw, ph)

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
            lw = self._blit(screen, ("> " if sel else "  ") + label, x0 + 28 * u, y,
                            f_item, C_TEXT if sel else C_DIM)
            rgb = self.swatches.get(self.items[i][1])
            if rgb is not None:
                # centred on the label's line, rounded, a 1 px light outline
                sr = pygame.Rect(int(x0 + 28 * u) + lw + sw_gap,
                                 int(y + 0.5 * (f_item.get_height() - sw)), sw, sw)
                rad = max(2, int(round(3 * u)))
                pygame.draw.rect(screen, tuple(int(c) for c in rgb[:3]), sr, border_radius=rad)
                pygame.draw.rect(screen, C_SWATCH_EDGE, sr, 1, border_radius=rad)
            if (i == top and top > 0) or (i == top + vis - 1 and top + vis < n):
                more = f"^ {top} more" if i == top and top > 0 else f"v {n - top - vis} more"
                self._blit(screen, more, x0 + 16 * u + items_w - f_lbl.size(more)[0] - 8 * u,
                           y + 4 * u, f_lbl, C_DIM)
            y += 36 * u
        hint = NAV_HINT_PAD if any("PAD" in (t or "") or "DUALSENSE" in (t or "")
                                   for t, _ in secs) else NAV_HINT
        if not any(a.startswith("set:") for _, a in self.items):
            hint = [row for row in hint if not row[0].startswith("LEFT")]
        # ENTER / ESC say what this page's footer says they do ('not now' on
        # the welcome page), keeping the pad's "  (CIRCLE, ...)" tail
        said = [(k, footer_says(self.footer, k) if k in ("ENTER", "ESC") else None, w)
                for k, w in hint]
        hint = [(k, s + (w[w.index("  ("):] if "  (" in w else "") if s else w)
                for k, s, w in said]
        # the key column fits its widest key ('LEFT / RIGHT' ran into its text)
        hint_kw = max(96 * u, max((f_lbl.size(k)[0] for k, _ in hint), default=0) + 12 * u)
        y += 6 * u
        for key, what in hint:
            self._blit(screen, key, x0 + 28 * u, y, f_lbl, C_DIM)
            self._blit(screen, what, x0 + 28 * u + hint_kw, y, f_lbl, C_DIM)
            y += row_h
        if note_lines:                    # the note, when no column had room for it
            y += 10 * u
            for ln in note_lines:
                self._blit(screen, ln, x0 + 28 * u, y, f_lbl, C_DIM)
                y += row_h

        # help columns; the key column widens to the section's longest key,
        # a wrapped row's lines continue under its text
        for c, yoff, title, key_w, laid in placed:
            x = x0 + col_px[c]
            yy = y0 + 96 * u + yoff
            if title:
                self._blit(screen, title, x, yy, f_sec, C_SECTION)
                yy += row_h
            for key, what, lines in laid:
                dim = isinstance(what, Dim)
                warn = isinstance(what, Warn)
                if what:
                    self._blit(screen, key, x, yy, f_lbl,
                               C_DIM if dim else C_WARN if warn else C_KEY)
                for ln in lines:
                    if what:
                        self._blit(screen, ln, x + key_w, yy, f_lbl,
                                   C_DIM if dim else C_WARN if warn else C_TEXT)
                    else:
                        self._blit(screen, ln, x, yy, f_lbl, C_DIM)
                    yy += row_h

        if self.art is not None:
            try:
                self.art(screen, pygame.Rect(int(x0 + col_px[0]), int(y0 + 96 * u),
                                             int(w - col_px[0] - 24 * u), int(self.art_h)))
            except Exception as exc:      # noqa: BLE001 -- a page's drawing never
                self.art = None           # takes the menu down
                print(f"menu: the page's drawing failed ({type(exc).__name__}: {exc})")

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
    # the key legend: its value column clears 'LEFT / RIGHT', and ENTER / ESC
    # say what the page's footer says (the welcome page's ESC is 'not now')
    lg = Menu("WELCOME", [("Car        Opel Corsa C", "set:car"), ("Back", "back")],
              footer="ENTER / CROSS select   ESC / CIRCLE not now   or click a row")
    lg.show()
    seen = []
    blit0 = lg._blit

    def _rec(screen, s_, x, y, font, col=C_TEXT):
        w_ = blit0(screen, s_, x, y, font, col)
        seen.append((s_, int(x), int(y), w_))
        return w_
    lg._blit = _rec
    lg.draw(scr)
    at = {s_: (x, y, w_) for s_, x, y, w_ in seen}
    lr, cv, esc = at.get("LEFT / RIGHT"), at.get("change value"), at.get("ESC")
    esc_what = [s_ for s_, x, y, _ in seen if esc and y == esc[1] and s_ != "ESC"]
    rep("key legend: the value column clears 'LEFT / RIGHT'; ENTER / ESC follow the footer",
        bool(lr and cv) and cv[0] >= lr[0] + lr[2] + 8 and esc_what == ["not now"]
        and footer_says("ESC / OPTIONS resume   R reset", "ESC") == "resume"
        and footer_says("or click a row", "ENTER") is None,
        f"'change value' at x {cv[0] if cv else None}, key ends "
        f"{lr[0] + lr[2] if lr else None}, ESC row {esc_what}")
    # a page's own drawing (the LAP RESULTS page's card): handed its rect at
    # the top of the first help column, the sections below it; the next
    # show() without one clears it; one that raises is dropped, not fatal
    got = []
    m.show(items=[("Back", "b")], sections=[("LIST", [("a", "b")])], columns=1,
           art=lambda s_, r_: got.append(tuple(r_)), art_h=120)
    m.draw(scr)
    placed_ok = bool(got) and got[0][3] == 120 and got[0][2] > 300
    m.show(items=[("Back", "b")])
    m.draw(scr)
    cleared = m.art is None and len(got) == 1

    def _boom(s_, r_):
        raise ValueError("x")
    m.show(items=[("Back", "b")], art=_boom, art_h=40)
    m.draw(scr)
    rep("a page's own drawing: its rect, cleared by the next page, a failure dropped",
        placed_ok and cleared and m.art is None, str(got[:1]))
    # typed digits reach only the rows a page declares; the menu stays open
    m.show(items=[("Cars", "set:n"), ("Back", "b")], typed=("set:n",))
    on_row = m.handle("digit:7")
    m.handle("nav_down")
    off_row = m.handle("digit:7")
    m.show(items=[("Cars", "set:n")])
    undeclared = m.handle("digit:7")
    rep("a digit on a typed row -> 'type:<d>:<action>'; nothing elsewhere",
        on_row == "type:7:set:n" and off_row is None and undeclared is None and m.open)
    # the swatch side channel: a square of the row's colour right after its
    # label and inside its row (so clear of the help columns), on that row
    # only; the rows stay 2-tuples; a re-show keeps it, new items drop it
    cob = (40, 72, 186)
    sm = Menu("SETTINGS", [("Car        Opel Corsa C", "set:car"),
                           ("Paint      cobalt blue", "set:paint"), ("Back", "back")],
              [("KEYBOARD", [("ESC", "back")])])

    def swatch_px():
        scr.fill((27, 29, 33))
        sm.draw(scr)
        a = pygame.surfarray.pixels3d(scr)
        hit = (a[..., 0] == cob[0]) & (a[..., 1] == cob[1]) & (a[..., 2] == cob[2])
        del a
        xs, ys = hit.nonzero()
        return int(hit.sum()), xs, ys
    sm.show(swatches={"set:paint": cob})
    n_sw, xs, ys = swatch_px()
    rows = dict(sm._rows)
    rx, ry, rw, rh = rows[1]
    lw = sm._font(18).size("  " + sm.items[1][0])[0]
    inside = bool(n_sw) and (xs.min() >= rx + 12 + lw and xs.max() < rx + rw
                             and ys.min() >= ry and ys.max() < ry + rh)
    sm.show(idx=1)
    kept = swatch_px()[0]
    sm.show(items=[("Resume", "resume"), ("Quit", "quit")])
    gone = swatch_px()[0]
    rep("a row's colour swatch: after its label, inside its row, kept on a re-show, "
        "dropped with new items; rows stay 2-tuples",
        100 <= n_sw <= 14 * 14 and inside and kept == n_sw and gone == 0
        and all(len(it) == 2 for it in sm.items) and sm.action() == "resume",
        f"{n_sw} px at x {xs.min() if n_sw else 0}-{xs.max() if n_sw else 0} (label ends "
        f"{rx + 12 + lw}, row {rx}-{rx + rw}), re-show {kept}, new items {gone}")
    # long help rows wrap inside the panel (the tutorial's pages cut theirs at
    # the edge): a 300-character row, a key-only one and a note, one column;
    # every text ends 8 px inside the panel, above the footer
    long = ("at a sector line: purple = best ever in this class, green = better "
            "than your PB lap, red = slower. " * 4)[:300]
    wm = Menu("TUTORIAL 9/13", [("Continue", "c"), ("End the tutorial", "e")],
              [("TIMING", [("sectors", "the lap is cut in 3"), ("the flash", long)]),
               ("NEXT", [(long, "")])], footer="ENTER continue", note=long)
    wm.show(columns=1)
    wm.draw(scr)

    def _inside(mn):
        pr = mn._last_panel
        return pr is not None and all(r.right <= pr.right - 8 and r.left >= pr.left
                                      for _, r in mn._last_text_rects)

    def _overlaps(mn):
        rs = [r for _, r in mn._last_text_rects]
        return sum(rs[i].colliderect(rs[j]) for i in range(len(rs)) for j in range(i))
    pieces = [r for s_, r in wm._last_text_rects if len(s_) > 20 and s_ in long]
    foot = [r for s_, r in wm._last_text_rects if s_ == "ENTER continue"]
    rep("a 300-character help row, a key-only row and the note wrap inside the panel "
        "(8 px in), their lines above the footer; nothing overprints",
        len(long) == 300 and _inside(wm) and len(pieces) >= 6 and bool(foot)
        and max(r.bottom for r in pieces) <= foot[0].top and _overlaps(wm) == 0,
        f"{len(pieces)} wrapped lines, rightmost {max(r.right for _, r in wm._last_text_rects)}"
        f" / panel {wm._last_panel.right if wm._last_panel else None}, "
        f"{_overlaps(wm)} overlaps")
    # the garage's shape: long item rows push two help columns so far right
    # that the second is ~100 px; it folds into the first, so the no-pad note
    # sits under the key table at its full width, not one word a line over it.
    # Two columns with room stay two, the first stopping short of the second.
    gar = Menu("GARAGE", [("Resume", "resume"),
                          ("Design the left wing  (mission -> section -> wing)", "design")]
               + [(f"Load a build ...  (the library's builds) {k}", f"b{k}") for k in range(10)],
               [("KEYBOARD", [("mouse drag / wheel", "orbit / zoom"),
                              ("W / SHIFT+W", "next / previous library wing in the slot"),
                              ("B / SHIFT+B", "next / previous saved build of this car")]
                 + [(f"K{k}", "a garage key and what it does") for k in range(14)])],
               note="no controller: pair the DualSense (CREATE+PS) - it hot-plugs")
    gar.show()
    gar.draw(scr)
    at = {s_: r for s_, r in gar._last_text_rects}
    kb, nt = at.get("KEYBOARD"), [r for s_, r in gar._last_text_rects if "controller" in s_]
    folded = bool(kb and nt) and nt[0].left == kb.left and nt[0].top > kb.top
    two = Menu("PAUSED", [("Resume", "resume"), ("Quit", "quit")],
               [("KEYBOARD", [("F / G", long)]), ("PS5 DUALSENSE", [("L2 / R2", long)])])
    two.show()
    two.draw(scr)
    cols = sorted({r.left for s_, r in two._last_text_rects
                   if s_ in ("KEYBOARD", "PS5 DUALSENSE")})
    rep("a second column under HELP_MIN_W folds into the first (the garage's note under "
        "its key table); two columns with room stay two; no text overlaps or leaves the panel",
        folded and _inside(gar) and _overlaps(gar) == 0 and len(nt) <= 2
        and len(cols) == 2 and _inside(two) and _overlaps(two) == 0,
        f"note at x {nt[0].left if nt else None} in {len(nt)} line(s), KEYBOARD at x "
        f"{kb.left if kb else None}; two columns at {cols}; overlaps {_overlaps(gar)} / "
        f"{_overlaps(two)}")
    # step numbers (' 1' .. '13') get their own width + 16 px of key column,
    # not the 118 px floor; a '--' placeholder keeps the floor
    st = Menu("TUTORIAL", [("Start", "s"), ("Back", "b")],
              [("THE STEPS", [(f"{k:2d}", f"step {k}") for k in range(1, 14)]),
               ("TOP 5", [("--", "none yet")])])
    st.show(columns=1)
    st.draw(scr)
    at = {s_: r for s_, r in st._last_text_rects}
    gap = at["step 13"].left - at["13"].left
    gap_dash = at["none yet"].left - at["--"].left
    want = st._font(14).size("13")[0] + 16
    rep("step numbers: a key column of the widest number + 16 px, not 118; '--' keeps 118",
        abs(gap - want) <= 1 and gap_dash >= 118,
        f"number -> text {gap} px (want {want}), '--' -> text {gap_dash} px")
    # a Dim row (the tutorial's finished steps): its key and text in C_DIM,
    # a plain row's in C_KEY / C_TEXT; a Dim is a str to everyone else
    dm = Menu("TUTORIAL", [("Back", "b")],
              [("THE STEPS", [(" 1", Dim("Throttle and brake  (arena)  done")),
                              (" 2", "Steering  (arena)  <- now")])])
    dm.show(columns=1)
    dm.draw(scr)
    drawn = {(k[0], k[2]) for k in dm._cache if len(k) == 3}
    want = {(" 1", C_DIM), ("Throttle and brake  (arena)  done", C_DIM),
            (" 2", C_KEY), ("Steering  (arena)  <- now", C_TEXT)}
    rep("a Dim help row draws its key and text dim, a plain one bright; a Dim is a str",
        want <= drawn and Dim("a") == "a" and isinstance(Dim("a"), str),
        f"missing {sorted(want - drawn)}")
    # a Warn row (a challenge's settings that rule out stars, task 45): its
    # key and text amber, a wrapped one's every line; a Warn is a str too
    wm = Menu("STOP FROM 100", [("Back", "b")],
              [("YOUR SETUP", [("yours", "ABS off  TC on  automatic"),
                               ("!", Warn("ABS off: expect longer stops " + "x" * 90))])])
    wm.show(columns=1)
    wm.draw(scr)
    drawn = {(k[0], k[2]) for k in wm._cache if len(k) == 3}
    amber = {t for t, c in drawn if c == C_WARN}
    w_lines = [t for t, _ in wm._last_text_rects if t in amber and t != "!"]   # drawn order
    rep("a Warn help row draws its key and text amber, every wrapped line; a plain one "
        "stays C_KEY / C_TEXT; a Warn is a str",
        {("!", C_WARN), ("yours", C_KEY), ("ABS off  TC on  automatic", C_TEXT)} <= drawn
        and len(w_lines) >= 2 and "".join(w_lines).replace(" ", "")
        == ("ABS off: expect longer stops " + "x" * 90).replace(" ", "")
        and Warn("a") == "a" and isinstance(Warn("a"), str) and C_WARN not in (C_KEY, C_TEXT),
        f"amber lines {len(w_lines)}")
    # note_under (a challenge's page, task 45): the note under the items'
    # key legend, wrapped to the items' width, inside the panel; the next
    # show() without it puts the note back under the help
    nu_note = "You start at 100 km/h, and the car holds it until you brake. " * 3
    wm.show(items=[("Start", "go"), ("Back", "b")], note=nu_note, columns=1, note_under=True)
    wm.draw(scr)
    at = {t: r for t, r in wm._last_text_rects}
    sec_x = at["YOUR SETUP"].left
    n_rects = [r for t, r in wm._last_text_rects if t and t in nu_note and len(t) > 12]
    under = (len(n_rects) >= 2 and all(r.right < sec_x and r.top > at["ESC"].bottom
                                       and wm._last_panel.contains(r) for r in n_rects))
    wm.show(note=nu_note, columns=1)
    wm.draw(scr)
    back = [r for t, r in wm._last_text_rects if t and t in nu_note and len(t) > 12]
    rep("note_under: the note under the items' key legend, left of the help, in the panel; "
        "the next show() without it puts it back under the help",
        under and wm.note_under is False and back and all(r.left >= sec_x for r in back),
        f"{len(n_rects)} lines under the legend, then {len(back)} right")
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
