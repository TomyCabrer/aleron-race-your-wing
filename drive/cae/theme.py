"""drive/cae/theme.py -- the AeroBO v3 light CAE theme as Python constants,
and the text and icon drawing every cae widget goes through.

The tokens are AeroBO's own (~/dev/urop-bo-aero/gui/v3/theme.py, as
RGB tuples) and the sizes are the ones measured off its screenshots at
1600x1000. The look they make, in AeroBO's words: structure comes from
hairlines and title bars, never shadow or gradient; one accent (a desk blue)
marks selection and the primary action and every other colour is a state;
the system UI face at 12 px with a monospace face for EVERY number; nothing
centred, nothing oversized, radius 2.

Four things here are not a copy of a stylesheet, because pygame is not a
browser:

  * FONTS are files, found per OS (SF Pro / SF Mono on macOS, Segoe UI /
    Consolas on Windows, DejaVu on Linux), then by `match_font`, then
    pygame's own default. pygame draws SF's narrower Display cut, so SANS at
    12 css px and below is drawn 1 px larger -- on SF only (`px_for`).
  * WORDS are spaced by rule: that cut's space is ~2 px at these sizes, so
    SANS text is drawn a word at a time and a space advances at least
    0.28 em (`WORD_SPACE_EM`). Drawing and every measure (`text_w`,
    `ellipsize`, the wraps built on them) share the one rule.
  * A character the resolved font does not have is drawn as its ASCII
    stand-in (`ASCII_FALLBACK`), not as a box: "▸" reads ">" on a font
    without it, and a label never shows tofu.
  * SIZES scale: `set_scale(S)` rewrites every px constant below in place
    for a large window (1.0 up to 1999 px wide, so the 1280x800 and
    1600x1000 layouts are 1:1). That is why every reader does
    `from drive.cae import theme as T` and reads `T.NAME` when it draws:
    `from .theme import LEFT_W` would keep the old number.

Text and icon surfaces are cached (a frame redraws the same few hundred
strings); the caches are cleared when they fill, on `set_scale`, and when
the fonts are re-resolved. Pure pygame + stdlib; the only file read is the
bundled icon font (drive/data/fonts/, Apache-2.0, see its LICENSE file).
"""

from __future__ import annotations

import math
import os
import re

import pygame

# -- colours (RGB tuples) ---------------------------------------------------------------------
CANVAS = (200, 204, 211)        # desktop grey behind the frames
CHROME = (230, 233, 237)        # menu bar, tool bar, tab strip, status bar
PANEL = (241, 243, 245)         # tree and properties bodies, card title bars, help-dot fill
WELL = (255, 255, 255)          # work area, log, cards, tables, plots
RULE = (173, 180, 191)          # frame hairlines, tabs, table frame
RULE_SOFT = (210, 215, 222)     # hairlines inside a pane: card border, prop rows, KPI bar
HEADER = (223, 227, 232)        # pane title bar, prop group header, table header
INK = (24, 28, 34)              # main text
INK_MUTED = (77, 87, 99)        # secondary text, pane titles, crumbs, status text
INK_FAINT = (120, 130, 143)     # captions, timestamps, tree chips, locked rows
#: INK_FAINT for small caption TEXT on white (KPI captions and units, grid heads, a field's
#: unit, plot labels): halfway to INK_MUTED, 5.3:1 on white where INK_FAINT is 3.9:1 -- a
#: readability pass; INK_FAINT itself stays for icons, rules, chips, stamps and locked rows
INK_FAINT_TEXT = (99, 109, 121)
ACCENT = (29, 95, 164)          # selection, primary action, busy
ACCENT_FILL = (207, 224, 243)   # selected tree row, menu hover, table row hover
GOOD = (26, 122, 70)            # done / ok / feasible
WARN = (148, 103, 10)           # caution / stopped / cancelled
BAD = (163, 42, 37)             # error / violated
TREE_HOVER = (227, 232, 238)
TAB_OFF = (221, 225, 230)
TAB_HOVER = (233, 237, 241)
ROW_EVEN = (247, 249, 250)      # table even rows
PRIMARY_EDGE = (22, 76, 133)    # primary button border
PRIMARY_HOVER = (61, 117, 175)
PRIMARY_DISABLED = (97, 143, 191)
FIELD_EDGE = (154, 162, 174)    # input / select border
TOOLTIP = (43, 49, 56)          # tooltip and plot hover label background
PROG_TRACK = (206, 209, 212)    # status-bar progress track
SLIDER_OFF = (206, 206, 206)    # slider track right of the thumb
SWITCH_OFF_TRACK = (167, 169, 171)   # measured 03 / 02: Quasar's track is currentColor (INK) at .38 ...
SWITCH_ON_TRACK = (133, 169, 206)    # ... and ACCENT at .54 when on (the spec's #bdbdbd / #8eafd1 were estimates)
TOAST_INFO = (49, 204, 236)     # Quasar's default "info" toast
STOP_GREY = (66, 66, 66)        # the Stop tool button's label
AMBER_LABEL = (180, 83, 9)
SCROLL_TRACK = (233, 236, 239)
SCROLL_THUMB = (188, 195, 204)
SCROLL_THUMB_HOVER = (164, 172, 183)
CODE_BG = (247, 249, 250)
SERIES = [(29, 95, 164), (26, 122, 70), (148, 103, 10), (163, 42, 37), (91, 74, 158), (13, 122, 134)]
GRID_A = (24, 28, 34, 31)        # rgba(24,28,34,.12)  plot grid
ZERO_A = (24, 28, 34, 71)        # rgba(24,28,34,.28)  plot zero line
BAND_A = (29, 95, 164, 41)       # rgba(29,95,164,.16) filled outlines / seed band
SHADOW_A = (20, 26, 34, 46)      # rgba(20,26,34,.18)  drop-down shadow, 0 6px 18px
RAMP = [(22, 48, 92), (29, 95, 164), (44, 143, 181), (58, 163, 122), (143, 174, 60)]   # 3-D local-cl ramp, stops 0/.30/.58/.80/1

# -- geometry (px at scale S = 1; set_scale(S) rewrites every size below in place, §4.1) -----------
S = 1.0
MENU_H, TOOL_H, STATUS_H = 26, 32, 23
LEFT_W, GAP, PAD, OUTPUT_H = 276, 2, 2, 150
PANE_TITLE_H, TAB_STRIP_H, TAB_H = 22, 25, 22
TREE_ROW_H, TREE_INDENT, TREE_BASE, TWISTY_W, ICON_PX, TREE_GAP = 21, 13, 4, 12, 14, 5
TREE_SPLIT = 0.6                       # tree : Properties = 3 : 2 at every size (AeroBO); the tree scrolls
PROP_ROW_H, PROP_GROUP_H, PROP_KEY_FRAC = 21, 23, 0.44
LOG_LINE_H, LOG_INSET_X, LOG_INSET_Y, LOG_MAX = 17, 20, 16, 400
TAG_H, HELP_DOT, FIELD_H, FIELD_W, SELECT_W, LABEL_MIN_W = 17, 14, 26, 84, 168, 150
KPI_BAR_W, KPI_MIN_W, KPI_GAP, KPI_H = 2, 104, 12, 37
CARD_TITLE_H, CARD_PAD_X, CARD_PAD_Y, CARD_GAP = 22, 9, 8, 6     # CARD_GAP = gap between a card's children
WORK_PAD_X, WORK_PAD_Y, STACK_GAP = 12, 10, 9
TABLE_ROW_H, TABLE_HEAD_H, PAGER_H = 28, 28, 36
BUTTON_H, BUTTON_PAD_X, BUTTON_ICON = 26, 8, 18
TOOL_BTN_H, TOOL_ICON = 24, 17
SCROLL_W = 12
HEARTBEAT_S, TOAST_S, TOOLTIP_DELAY_S = 0.5, 5.0, 0.5
HINT_WORDS, TIP_WORDS = 24, 8
#: the quiet text's css sizes: a readability pass over AeroBO's (the owner: keep the AeroBO
#: look "if not difficult to read"). AeroBO sets its hints, notes, captions and Properties keys
#: at 10-11 px in grey, which is hard to read in a game window; these roles are drawn 1 css px
#: larger (= 1 pygame px) and every other size is AeroBO's own. They are css, so set_scale
#: leaves them alone -- px_for scales them like any other css size.
HINT_CSS = 12       # AeroBO 11: hints, Properties keys, v1 notes and units, panel titles, captions
NOTE_CSS = 11.5     # AeroBO 10.5: field / radio notes, KPI captions, grid heads and second lines, legends
ANNOT_CSS = 11      # AeroBO 10: a plot's shape labels ("Sobol | BO", a band's name)

#: the px constants `set_scale` rewrites. NOT in it, deliberately: S itself,
#: the two fractions (TREE_SPLIT, PROP_KEY_FRAC), the log's line cap, the
#: three durations and the two word counts -- scaling those would be wrong
#: (round(0.6 * 1.5) is a tree with no Properties pane under it).
_PX_NAMES = ("MENU_H", "TOOL_H", "STATUS_H", "LEFT_W", "GAP", "PAD", "OUTPUT_H",
             "PANE_TITLE_H", "TAB_STRIP_H", "TAB_H",
             "TREE_ROW_H", "TREE_INDENT", "TREE_BASE", "TWISTY_W", "ICON_PX", "TREE_GAP",
             "PROP_ROW_H", "PROP_GROUP_H", "LOG_LINE_H", "LOG_INSET_X", "LOG_INSET_Y",
             "TAG_H", "HELP_DOT", "FIELD_H", "FIELD_W", "SELECT_W", "LABEL_MIN_W",
             "KPI_BAR_W", "KPI_MIN_W", "KPI_GAP", "KPI_H",
             "CARD_TITLE_H", "CARD_PAD_X", "CARD_PAD_Y", "CARD_GAP",
             "WORK_PAD_X", "WORK_PAD_Y", "STACK_GAP", "TABLE_ROW_H", "TABLE_HEAD_H", "PAGER_H",
             "BUTTON_H", "BUTTON_PAD_X", "BUTTON_ICON", "TOOL_BTN_H", "TOOL_ICON", "SCROLL_W")
_BASE = {name: globals()[name] for name in _PX_NAMES}      # the S = 1 values, for set_scale

# -- fonts ----------------------------------------------------------------------------------------
FONTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fonts")
ICON_FONT = os.path.join(FONTS_DIR, "MaterialIcons-Regular.ttf")
SANS_PATHS = ("/System/Library/Fonts/SFNS.ttf",                          # macOS
              r"C:\Windows\Fonts\segoeui.ttf",                           # Windows
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",         # Debian / Ubuntu / SteamOS
              "/usr/share/fonts/TTF/DejaVuSans.ttf")                     # Arch
MONO_PATHS = ("/System/Library/Fonts/SFNSMono.ttf", r"C:\Windows\Fonts\consola.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
              "/System/Library/Fonts/Menlo.ttc")
SANS_NAMES = ("segoeui", "dejavusans", "arial", "helvetica")             # match_font only after the paths
MONO_NAMES = ("consolas", "dejavusansmono", "menlo", "couriernew")
# NOTE measured: on macOS match_font("sfns"...) returns ARIAL, so SF is found only through SANS_PATHS.
# role -> (family, css px, bold); pygame px: SANS <=12 css -> css+1 ONLY when the resolved sans file's basename
# starts with "SFNS" (the narrower Display cut); every other font: css px. MONO = css px; 'kpi' = MONO 18
# (19 css measures as 18). Fractions round half up first. Every px is multiplied by S before rounding.
#: the KPI value's size: family "kpi" is the MONO file at this css px whatever
#: `css` a caller passes (AeroBO's 19 px readout measures as 18 in pygame:
#: "6.825e+05" is 101 px in its screenshot, 99 at 18 and 108 at 19)
KPI_CSS = 18
#: family -> which resolved file draws it
_FILE_OF = {"sans": "sans", "mono": "mono", "kpi": "mono", "icon": "icon"}
#: pygame's default font is drawn at 0.6875 of the size asked for (font.c);
#: the last-resort fallback asks for px / 0.6875 so its text is not a third
#: smaller than every other font's
_DEFAULT_FONT_SCALE = 0.6875
#: a codepoint no font maps (a noncharacter): what it renders as IS the
#: font's missing-glyph box, which `glyph_ok` compares against
_NOTDEF = "\uffff"
#: how many rendered strings (and widths, and ellipsised strings) each cache
#: holds before it is emptied
TEXT_CACHE_MAX = 4096
#: SANS word spacing, in em. NOTE measured (pygame 2.5.2, SDL_ttf 2.20): SF's
#: Display cut draws its space 2 px wide at 10-12 px and 3 px at 13-16, a
#: third of a browser's ~0.25 em, so "lap as it stands" reads "lapasitstands".
#: SANS text is therefore drawn one word at a time -- kerning kept inside each
#: word -- and every ASCII space advances max(the font's own space,
#: WORD_SPACE_EM * px) (`space_w`). DejaVu's and Segoe UI's spaces are already
#: that wide, so the rule moves only SF. MONO, kpi and icon text are drawn whole.
WORD_SPACE_EM = 0.28

_files: dict = {}           # "sans" / "mono" / "icon" -> resolved path, or None = pygame's default
_fonts: dict = {}           # (family, px, bold) -> pygame.font.Font
_text: dict = {}            # (s, family, px, bold, colour) -> Surface
_widths: dict = {}          # (s, family, px, bold) -> int
_ellip: dict = {}           # (s, family, px, bold, max_w) -> str
_spaces: dict = {}          # (family, px, bold) -> the px one space advances (SANS word spacing)
_glyph: dict = {}           # (ch, family) -> bool
_notdef: dict = {}          # family -> (size, bytes) of the missing-glyph box
_icons: dict = {}           # (codepoint, px, colour) -> Surface
_fills: dict = {}           # rgba -> scratch SRCALPHA surface (blend_rgba)


def _half_up(v: float) -> int:
    """Round half up (11.5 -> 12, 10.5 -> 11): CSS's rounding, not Python's
    round-half-to-even, which would draw 10.5 px text at 10."""
    return int(math.floor(v + 0.5))


def _init() -> None:
    if not pygame.font.get_init():
        pygame.font.init()


def _opens(path) -> bool:
    try:
        pygame.font.Font(path, 12)
        return True
    except Exception:           # missing, unreadable, or a format SDL_ttf cannot open (WOFF2)
        return False


def _find(paths, names):
    """The first of `paths` that exists and opens, else what `match_font`
    finds for `names` (ONE call with the whole tuple: its first call walks
    the system's font list, which is seconds on a Linux box running
    fc-list), else None = pygame's default font."""
    for p in paths:
        if p and os.path.exists(p) and _opens(p):
            return p
    if names:
        try:
            p = pygame.font.match_font(tuple(names))
        except Exception:
            p = None
        if p and _opens(p):
            return p
    return None


def _clear() -> None:
    for cache in (_fonts, _text, _widths, _ellip, _spaces, _glyph, _notdef, _icons):
        cache.clear()


def resolve_fonts(sans_paths=None, mono_paths=None, sans_names=None, mono_names=None) -> dict:
    """Find the SANS, MONO and icon files (D2) and empty every cache built
    on the old ones. Runs once per process on first use (drive.py builds a
    new Garage on every garage entry; the files do not change under it).

    The arguments replace the module's tuples for this resolution only --
    the self-checks pass bogus ones to prove the fallbacks, then call it
    again with none to put the real fonts back. Returns the resolved paths,
    `{"sans": ..., "mono": ..., "icon": ...}`, None meaning pygame's default.
    """
    _init()
    _files["sans"] = _find(SANS_PATHS if sans_paths is None else sans_paths,
                           SANS_NAMES if sans_names is None else sans_names)
    _files["mono"] = _find(MONO_PATHS if mono_paths is None else mono_paths,
                           MONO_NAMES if mono_names is None else mono_names)
    _files["icon"] = ICON_FONT if _opens(ICON_FONT) else None
    _clear()
    return dict(_files)


def font_file(family: str):
    """The file `family` is drawn with (None = pygame's default font)."""
    if not _files:
        resolve_fonts()
    return _files[_FILE_OF[family]]


def _is_sf() -> bool:
    p = font_file("sans")
    return bool(p) and os.path.basename(p).startswith("SFNS")


def px_for(family: str, css: float) -> int:
    """The pygame px a `css` px text of `family` is drawn at (rule above).

    SF's +1 is a WIDTH calibration against AeroBO's screenshots: "Operating
    point" is 88 px there, 84 at Font(SFNS, 12) and 87 at 13. Segoe UI and
    DejaVu are drawn in the cut a browser uses, so they take css px as is.
    """
    if family == "kpi":
        css = KPI_CSS
    c = _half_up(css)
    if family == "sans" and css <= 12 and _is_sf():
        c += 1
    return max(1, _half_up(c * S))


def _font(family: str, px: int, bold: bool = False) -> pygame.font.Font:
    key = (family, px, bold)
    f = _fonts.get(key)
    if f is None:
        path = font_file(family)
        if path is None:
            f = pygame.font.Font(None, max(1, _half_up(px / _DEFAULT_FONT_SCALE)))
        else:
            f = pygame.font.Font(path, px)
        if bold:
            f.set_bold(True)          # a separate instance: set_bold is per Font object
        _fonts[key] = f
    return f


def font(family: str, css: float, bold: bool = False) -> pygame.font.Font:
    """The cached Font for `family` ("sans" | "mono" | "kpi" | "icon") at
    `css` px (scaled by S, SF +1 rule). Bold is pygame's synthetic bold,
    which is what AeroBO's weight 600 looks like; weight 500 is regular.

    NOTE measured (pygame 2.5.2, SDL_ttf 2.20, "Configuration" at 12-13 px):
    synthetic bold carries 1.7x regular's ink at +1 px of width. A real face
    is LIGHTER and wider: SF's own Semibold instance (SFNS.ttf is one
    variable font; only pygame.freetype can pick an instance) 1.36x at +11-16 %,
    its Bold 1.5-1.75x at +11-21 %; Arial Bold 1.35x at +4-16 %. So no real bold
    face is loaded -- it would be less bold and break the AeroBO widths."""
    return _font(family, px_for(family, css), bold)


# -- glyphs ----------------------------------------------------------------------------------------
ASCII_FALLBACK = {"·": "-", "▸": ">", "—": "-", "–": "-", "≤": "<=", "≥": ">=", "⟺": "<=>", "²": "2", "³": "3",
                  "⁻⁴": "e-4", "α": "a", "μ": "mu", "σ": "s", "°": " deg", "½": "1/2", "✕": "x", "▲": "^",
                  "▼": "v", "◉": "(o)", "○": "( )", "…": "...", "→": "->", "←": "<-", "×": "x", "±": "+/-",
                  "⏮": "|<", "⏭": ">|", "‹": "<", "›": ">", "▾": "v", "“": '"', "”": '"'}
#: longest key first, so "⁻⁴" is replaced whole before any one-character key
_FALLBACK_ORDER = sorted(ASCII_FALLBACK.items(), key=lambda kv: -len(kv[0]))
_tobytes = getattr(pygame.image, "tobytes", None) or pygame.image.tostring


def _pixels(f: pygame.font.Font, s: str):
    img = f.render(s, True, (0, 0, 0))
    return img.get_size(), _tobytes(img, "RGBA")


def glyph_ok(ch: str, family: str) -> bool:
    """Does the font resolved for `family` have a glyph for `ch`?

    NOTE measured (pygame 2.5.2, SDL_ttf 2.20): `Font.metrics(ch)` returns the
    missing-glyph box's metrics for a BMP character the font lacks, so a
    metrics test alone passes "⏮" in SF Pro and "▸" in pygame's default
    font. A character counts as present when its metrics exist AND it does
    not render pixel for pixel as the font's missing-glyph box.
    """
    key = (ch, family)
    ok = _glyph.get(key)
    if ok is None:
        f = _font(family, 12)
        try:
            nd = _notdef.get(family)
            if nd is None:
                nd = _notdef[family] = _pixels(f, _NOTDEF)
            ok = f.metrics(ch)[0] is not None and _pixels(f, ch) != nd
        except Exception:
            ok = False
        _glyph[key] = ok
    return ok


def _displayable(s: str, family: str) -> str:
    """`s` with every character the font lacks replaced by its ASCII
    stand-in. A character with no stand-in is left as it is."""
    if s.isascii() or family == "icon":
        return s
    for key, rep in _FALLBACK_ORDER:
        if key in s and not all(glyph_ok(c, family) for c in key):
            s = s.replace(key, rep)
    return s


# -- text ------------------------------------------------------------------------------------------
def _spaced(s: str, family: str) -> bool:
    """Is `s` drawn a word at a time (SANS with a space in it)?"""
    return family == "sans" and " " in s


def _space_px(family: str, px: int, bold: bool) -> int:
    key = (family, px, bold)
    sp = _spaces.get(key)
    if sp is None:
        own = _font(family, px, bold).size(" ")[0]
        sp = _spaces[key] = max(own, _half_up(WORD_SPACE_EM * px)) if family == "sans" else own
    return sp


def space_w(family="sans", css=12, bold=False) -> int:
    """The px one space advances in `family` at `css` px -- for SANS the
    word-spacing rule (`WORD_SPACE_EM`), for MONO the font's own space.
    `text_w(" ")` is the same number."""
    return _space_px(family, px_for(family, css), bool(bold))


def _measure(s: str, family: str, px: int, bold: bool) -> int:
    """The width `_render` draws `s` (already glyph-fallen-back) at: the
    font's own width, or for SANS with spaces the sum of its words plus
    one `_space_px` per space. THE one width rule -- text_w, ellipsize,
    every wrap and every hit rect measure through it."""
    f = _font(family, px, bold)
    if not _spaced(s, family):
        return f.size(s)[0]
    words = s.split(" ")
    return sum(f.size(w)[0] for w in words if w) + _space_px(family, px, bold) * (len(words) - 1)


def _render(s: str, family: str, px: int, bold: bool, colour) -> pygame.Surface:
    f = _font(family, px, bold)
    if not _spaced(s, family):
        return f.render(s, True, colour)
    sp = _space_px(family, px, bold)
    img = pygame.Surface((max(1, _measure(s, family, px, bold)), f.get_height()), pygame.SRCALPHA)
    img.fill((*colour[:3], 0))          # the text's colour at alpha 0: a word's antialiased edge blends true
    x = 0
    for w in s.split(" "):
        if w:
            img.blit(f.render(w, True, colour), (x, 0))
            x += f.size(w)[0]
        x += sp
    return img


def _surface(s: str, family: str, px: int, bold: bool, colour) -> pygame.Surface:
    key = (s, family, px, bold, colour)
    img = _text.get(key)
    if img is None:
        if len(_text) >= TEXT_CACHE_MAX:
            _text.clear()
        img = _render(_displayable(s, family), family, px, bold, colour)
        _text[key] = img
    return img


def text_w(s: str, family="sans", css=12, bold=False) -> int:
    """The drawn width of `s` in px (after the glyph fallback, SANS with
    the word-spacing rule)."""
    s = str(s)
    px = px_for(family, css)
    key = (s, family, px, bool(bold))
    w = _widths.get(key)
    if w is None:
        if len(_widths) >= TEXT_CACHE_MAX:
            _widths.clear()
        w = _measure(_displayable(s, family), family, px, bool(bold))
        _widths[key] = w
    return w


def ellipsize(s, family, css, max_w, bold=False) -> str:
    """`s` cut at the right with "…" so it fits `max_w` px ("" when not even
    the ellipsis fits). A string that fits comes back unchanged."""
    s = str(s)
    px = px_for(family, css)
    key = (s, family, px, bool(bold), int(max_w))
    out = _ellip.get(key)
    if out is None:
        if len(_ellip) >= TEXT_CACHE_MAX:
            _ellip.clear()
        out = _cut(s, family, css, int(max_w), bool(bold))
        _ellip[key] = out
    return out


def _cut(s: str, family: str, css: float, max_w: int, bold: bool) -> str:
    if text_w(s, family, css, bold) <= max_w:
        return s
    if text_w("…", family, css, bold) > max_w:
        return ""
    lo, hi = 0, len(s)              # s[:lo] + "…" fits; s[:hi + 1] + "…" does not
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if text_w(s[:mid].rstrip() + "…", family, css, bold) <= max_w:
            lo = mid
        else:
            hi = mid - 1
    return s[:lo].rstrip() + "…"


def text(surf, s, x, y, family="sans", css=12, colour=INK, bold=False, anchor="left", clip_w=None) -> int:
    """Draw `s` with its line box's TOP at `y`; `x` is its left edge, right
    edge or centre by `anchor` ("left" | "right" | "centre"). `clip_w` cuts
    it with "…" to that many px. Returns the drawn width."""
    s = str(s)
    if clip_w is not None:
        s = ellipsize(s, family, css, clip_w, bold)
    if not s:
        return 0
    img = _surface(s, family, px_for(family, css), bool(bold), tuple(colour))
    w = img.get_width()
    x = int(x)
    if anchor == "right":
        x -= w
    elif anchor in ("centre", "center"):
        x -= w // 2
    surf.blit(img, (x, int(y)))
    return w


def set_scale(s: float) -> None:
    """Scale every px constant of the geometry block to `s` times its S = 1
    value (rounded half up) and empty the font and text caches; the font px
    follow through `px_for`. Called by DesignShell.__init__ with
    max(1.0, floor(min(W/1600, H/1000) * 4) / 4): 1.0 for every window up to
    1999 px wide, so the parity sizes 1280x800 and 1600x1000 are untouched.
    `set_scale(1.0)` restores the constants exactly."""
    global S
    S = float(s)
    g = globals()
    for name, base in _BASE.items():
        g[name] = _half_up(base * S)
    _clear()
    _fills.clear()


# -- icons -----------------------------------------------------------------------------------------
ICONS = {  # every glyph used; all verified present in the bundled TTF
    "check_circle": 0xE86C, "radio_button_unchecked": 0xE836, "play_circle": 0xE1C4, "pending": 0xEF64,
    "lock": 0xE897, "error_outline": 0xE001, "cancel": 0xE5C9,
    "tune": 0xE429, "speed": 0xE9E4, "bolt": 0xEA0B, "format_list_numbered": 0xE242, "gesture": 0xE155,
    "auto_graph": 0xE4FB, "category": 0xE574, "crop_free": 0xE3C2, "settings": 0xE8B8, "show_chart": 0xE6E1,
    "summarize": 0xF071, "view_in_ar": 0xE9FE, "ssid_chart": 0xEB66, "table_rows": 0xF101,
    "play_arrow": 0xE037, "stop": 0xE047, "chevron_left": 0xE5CB, "chevron_right": 0xE5CC, "restart_alt": 0xF053,
    "clear_all": 0xE0B8, "folder_open": 0xE2C8, "layers_clear": 0xE53C, "content_copy": 0xE14D, "info": 0xE88E,
    "close": 0xE5CD, "search": 0xE8B6, "link_off": 0xE16F, "sync": 0xE627, "warning": 0xE002, "expand_more": 0xE5CF,
    "expand_less": 0xE5CE, "arrow_drop_down": 0xE5C5, "arrow_drop_up": 0xE5C7, "save": 0xE161, "edit": 0xE3C9,
    "arrow_back": 0xE5C4, "arrow_forward": 0xE5C8, "check": 0xE5CA, "undo": 0xE166, "refresh": 0xE5D5, "add": 0xE145,
    "remove": 0xE15B, "file_download": 0xE2C4, "description": 0xE873, "keyboard": 0xE312, "menu_book": 0xEA19,
    "science": 0xEA4B, "first_page": 0xE5DC, "last_page": 0xE5DD, "north_east": 0xF1E1, "auto_awesome": 0xE65F,
    "radio_button_checked": 0xE837, "check_box": 0xE834, "check_box_outline_blank": 0xE835, "table_view": 0xF1BE,
    "data_object": 0xEAD3, "library_add": 0xE02E, "straighten": 0xE41C,
}


def icon(surf, name_or_cp, x, y, px=None, colour=INK_MUTED, anchor="left") -> int:
    """Draw a Material icon (a name in ICONS, or a codepoint) in a px x px
    box whose top is `y`; `anchor` as in `text`. `px` is device px, not css
    -- pass the geometry constants (T.ICON_PX, T.TOOL_ICON, T.BUTTON_ICON),
    which set_scale has already scaled; None = T.ICON_PX at call time.
    Returns the drawn width. An unknown name raises KeyError: icon names are
    constants in the code, and a typo should fail the first self-check that
    draws it, not draw nothing."""
    cp = ICONS[name_or_cp] if isinstance(name_or_cp, str) else int(name_or_cp)
    px = ICON_PX if px is None else int(px)
    colour = tuple(colour)
    key = (cp, px, colour)
    img = _icons.get(key)
    if img is None:
        if len(_icons) >= 1024:
            _icons.clear()
        img = _font("icon", px).render(chr(cp), True, colour)
        _icons[key] = img
    w = img.get_width()
    x = int(x)
    if anchor == "right":
        x -= w
    elif anchor in ("centre", "center"):
        x -= w // 2
    surf.blit(img, (x, int(y)))
    return w


# -- colour ----------------------------------------------------------------------------------------
def fade(colour, alpha: float, bg=WELL) -> tuple:
    """`colour` at opacity `alpha` over `bg`, as the opaque RGB a browser
    shows: disabled tool buttons .38, disabled controls .6, dead columns .45."""
    a = max(0.0, min(1.0, float(alpha)))
    return tuple(_half_up(c * a + b * (1.0 - a)) for c, b in zip(colour[:3], bg[:3]))


def tint(colour, alpha=0.10, bg=WELL) -> tuple:
    """A tag's background: its colour at 10% over what is beneath
    (AeroBO widgets._tint). ACCENT on white is #e8eff6."""
    return fade(colour, alpha, bg)


def blend_rgba(surf, rect, rgba):
    """Fill `rect` with a translucent `rgba` (a 3-tuple or alpha 255 is a
    plain fill). One scratch surface per colour, grown to the largest rect
    asked for, so a frame of grid lines allocates nothing."""
    r = pygame.Rect(rect)
    if r.w <= 0 or r.h <= 0:
        return
    a = rgba[3] if len(rgba) > 3 else 255
    if a >= 255:
        surf.fill(tuple(rgba[:3]), r)
        return
    if a <= 0:
        return
    key = tuple(rgba)
    img = _fills.get(key)
    if img is None or img.get_width() < r.w or img.get_height() < r.h:
        if img is None and len(_fills) >= 32:
            _fills.clear()
        w = max(r.w, img.get_width() if img else 0)
        h = max(r.h, img.get_height() if img else 0)
        img = pygame.Surface((w, h), pygame.SRCALPHA)
        img.fill(key)
        _fills[key] = img
    surf.blit(img, r.topleft, (0, 0, r.w, r.h))


# -- words -----------------------------------------------------------------------------------------
#: a sentence end a hint may be split at: the lookbehind keeps the stop on
#: the lead, and the next character has to open a sentence, so a decimal
#: point ("0.488 m in front") never matches
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z“‘\"(])")
#: the fallback for a first sentence that is itself a paragraph: after the
#: last em-dash or semicolon, which STAYS on the lead ("...for the job —"
#: says out loud that there is more)
_CLAUSE = re.compile(r"(?<=[—;])\s+")


def split_hint(text: str, max_words=HINT_WORDS) -> tuple:
    """`(on_screen, behind_the_mark)` for one quiet line, AeroBO's
    widgets.split_hint: a hint of more than `max_words` words keeps its
    leading whole sentences (as many as fit `max_words`) on screen and puts
    the rest behind a "?"; a first sentence too long for that is split after
    its last "—" or ";" inside the limit; one with neither keeps its first
    `max_words` words plus " …" and the WHOLE text goes in the popup. Words
    are moved, never dropped. A hint short enough is `(text, None)`."""
    t = str(text)
    words = t.split()
    if len(words) <= max_words:
        return (t, None)
    for pattern in (_SENTENCE, _CLAUSE):
        best = None
        for m in pattern.finditer(t):
            if len(t[:m.start()].split()) > max_words:
                break
            best = m
        if best is not None:
            return (t[:best.start()].strip(), t[best.end():].strip() or None)
    return (" ".join(words[:max_words]) + " …", t)


def fmt(v, nd=4) -> str:
    """A number as the shell prints it (always in MONO): `nd` significant
    figures, 3 for the very large and the very small, "0" for a rounding
    residue, "—" for no value (None, NaN, ±inf). A non-number is str()."""
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return "—"
    a = abs(f)
    if a < 1e-9:
        return "0"
    if a >= 1e6 or a < 1e-4:
        return f"{f:.3g}"
    return f"{f:.{nd}g}"


#: every (family, css, bold) the chrome and the work-area kit draw with --
#: the faces `warm` shapes a string in
WARM_FACES = ([("sans", c, False) for c in (9, 10, 10.5, 11, 11.5, 12)]
              + [("sans", c, True) for c in (10, 10.5, 11, 11.5, 12)]
              + [("mono", c, False) for c in (9, 9.5, 10, 10.5, 11, 11.5, 12)] + [("kpi", 18, False)])


def warm() -> None:
    """Resolve the three files and open the sizes a first frame draws, so
    the first measured frame never pays font discovery -- and SHAPE a string
    in every face the kit draws with (`WARM_FACES`): opening a face is not
    the whole cost, the first shaping call on each face instance pays its
    lazy load (measured 71 ms for SF 12 px on the development Mac, ~430 ms
    over a first shell frame)."""
    font_file("sans")
    for px in range(10, 14):
        for family in ("sans", "mono"):
            _font(family, _half_up(px * S))
            _font(family, _half_up(px * S), True)
    font("kpi", KPI_CSS)
    for family, css, bold in WARM_FACES:
        text_w("Ag 0.5 · ▸", family, css, bold)
    for px in (ICON_PX, TOOL_ICON, BUTTON_ICON):
        _font("icon", px)
    for family in ("sans", "mono"):
        glyph_ok("…", family)


# ==================================================================== #
#  SELF-CHECK                                                          #
# ==================================================================== #
def self_check(verbose: bool = True, out: str = "") -> bool:
    """Fonts resolve (and fall back), icons render, text measures, draws and
    cuts, the glyph fallback works, set_scale round-trips, and the small
    rules (split_hint, tint, fmt) hold. `out`: also save a specimen sheet
    PNG there. The widgets self-check carries the rows that need the shell's
    own strings (every non-ASCII character in use, the longest tree label)."""
    import sys
    import time
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}" + (f": {msg}" if msg else ""))

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    t0 = time.perf_counter()
    files = resolve_fonts()
    warm()
    t_warm = time.perf_counter() - t0
    names = {k: (os.path.basename(v) if v else None) for k, v in files.items()}
    on_mac = sys.platform == "darwin"
    rep("the three fonts resolve, the icon font from drive/data/fonts",
        files["sans"] and files["mono"] and files["icon"] == ICON_FONT
        and (not on_mac or (names["sans"] == "SFNS.ttf" and names["mono"] == "SFNSMono.ttf")),
        f"{names}, resolve + warm {1e3 * t_warm:.0f} ms")
    sf = _is_sf()
    want = {("sans", 12): 13, ("sans", 11.5): 13, ("sans", 10.5): 12, ("sans", 9.5): 11, ("sans", 13): 13,
            ("mono", 11): 11, ("mono", 11.5): 12, ("mono", 9.5): 10, ("kpi", 19): 18, ("icon", 14): 14}
    if not sf:
        want.update({("sans", 12): 12, ("sans", 11.5): 12, ("sans", 10.5): 11, ("sans", 9.5): 10})
    got = {k: px_for(*k) for k in want}
    rep("px: SANS +1 at <= 12 css on SF only, half up; MONO as is; kpi 18",
        got == want, f"sf={sf} {sorted((f'{k[0]} {k[1]}', v) for k, v in got.items() if v != want[k])}")
    if sf:
        w = text_w("Operating point", "sans", 11.5)
        rep("SF width calibration: 'Operating point' at 11.5 css is 88 px (WingLab 88; word spacing on)",
            w == 88, f"{w} px")
    # -- word spacing: SANS a word at a time, one rule for drawing and measuring
    sp = {c: space_w("sans", c) for c in (10.5, 11, 12)}
    own = {c: font("sans", c).size(" ")[0] for c in sp}
    rule = all(sp[c] == max(own[c], _half_up(WORD_SPACE_EM * px_for("sans", c))) for c in sp)
    add = all(text_w("vs no wings", "sans", c, b) == text_w("vs", "sans", c, b) + text_w("no", "sans", c, b)
              + text_w("wings", "sans", c, b) + 2 * space_w("sans", c, b) and text_w(" ", "sans", c, b) == space_w("sans", c, b)
              for c in sp for b in (False, True))
    mono_whole = text_w("lap as it stands", "mono", 11) == font("mono", 11).size("lap as it stands")[0]
    gap_img = _surface("lap as", "sans", px_for("sans", 11), False, INK)
    cols = [any(gap_img.get_at((x, y))[3] for y in range(gap_img.get_height())) for x in range(gap_img.get_width())]
    runs, cur = [], 0
    for c in cols + [True]:
        if c:
            if cur:
                runs.append(cur)
            cur = 0
        else:
            cur += 1
    rep("SANS word spacing: a space advances max(own, 0.28 em); a spaced string measures as its words + "
        "spaces; the drawn gap is visible; MONO drawn whole",
        rule and add and mono_whole and gap_img.get_width() == text_w("lap as", "sans", 11)
        and (not sf or (sp[11] >= 3 and max(runs or [0]) >= 3)),
        f"space {sp} (font's own {own}), widest blank run {max(runs or [0])} px")

    # -- icons
    blank = []
    nd = _pixels(_font("icon", ICON_PX), _NOTDEF)
    for name, cp in ICONS.items():
        img = _font("icon", ICON_PX).render(chr(cp), True, INK)
        if img.get_bounding_rect().w == 0 or _pixels(_font("icon", ICON_PX), chr(cp)) == nd:
            blank.append(name)
    rep(f"every one of the {len(ICONS)} ICONS glyphs renders, and is not the missing-glyph box",
        not blank, f"blank {blank}")
    sf_ = pygame.Surface((64, 32), pygame.SRCALPHA)
    wi = icon(sf_, "check_circle", 4, 4, colour=GOOD)
    box = sf_.get_bounding_rect()
    rep("icon() draws in a px x px box at (x, y) and returns its width",
        wi == ICON_PX and box.w > 0 and box.x >= 4 and box.y >= 4
        and box.right <= 4 + ICON_PX and box.bottom <= 4 + ICON_PX, f"w {wi}, ink {box}")
    try:
        icon(sf_, "no_such_icon", 0, 0)
        rep("an unknown icon name raises KeyError", False)
    except KeyError:
        rep("an unknown icon name raises KeyError", True)

    # -- text
    def ink_box(surface):
        m = pygame.mask.from_threshold(surface, WELL, (1, 1, 1, 255))
        m.invert()
        bs = m.get_bounding_rects()
        return bs[0].unionall(bs[1:]) if bs else pygame.Rect(0, 0, 0, 0)

    probe = pygame.Surface((400, 60))
    probe.fill(WELL)
    s = "Operating point"
    w = text(probe, s, 10, 10, "sans", 12, INK)
    r = ink_box(probe)
    rep("text() draws from its top-left, returns the width text_w measures",
        w == text_w(s) and r.x >= 10 and r.right <= 10 + w and r.y >= 10
        and r.bottom <= 10 + font("sans", 12).get_height(), f"w {w}, ink {r}")
    probe.fill(WELL)
    wr = text(probe, s, 300, 10, anchor="right")
    br = ink_box(probe)
    probe.fill(WELL)
    wc = text(probe, s, 200, 10, anchor="centre")
    bc = ink_box(probe)
    rep("anchor right ends at x, centre straddles x",
        wr == wc == w and 300 - 3 <= br.right <= 300 and abs(bc.centerx - 200) <= 2, f"{br} {bc}")
    n0 = len(_text)
    for _ in range(3):
        text(probe, s, 10, 10)
    rep("a repeated draw is served from the cache", len(_text) == n0, f"{len(_text)} surfaces")
    long_s = "Maximise downforce at the operating point, subject to the drag ceiling"
    cut = ellipsize(long_s, "sans", 12, 150)
    wcut = text(probe, long_s, 10, 10, clip_w=150)
    rep("ellipsize cuts to fit with '…'; clip_w draws the cut; a short string is untouched",
        cut.endswith("…") and text_w(cut) <= 150 and wcut == text_w(cut) and ellipsize(s, "sans", 12, 150) == s
        and ellipsize(s, "sans", 12, 2) == "", repr(cut))
    t1 = time.perf_counter()
    for i in range(2000):
        text(probe, s, 10, 10, "sans", 12, INK)
    us = 1e6 * (time.perf_counter() - t1) / 2000
    rep("a cached text() draw is cheap (< 25 us)", us < 25, f"{us:.1f} us")
    for i in range(TEXT_CACHE_MAX + 10):
        _surface(f"k{i}", "mono", 11, False, INK)
    rep("the text cache is capped", len(_text) <= TEXT_CACHE_MAX, f"{len(_text)}")

    # -- glyphs
    unresolved = {}
    for fam in ("sans", "mono"):
        miss = [k for k in ASCII_FALLBACK if not all(glyph_ok(c, fam) for c in k)]
        unresolved[fam] = "".join(miss)
        bad = [k for k in miss if not _displayable(k, fam).isascii()]
        rep(f"{fam}: every ASCII_FALLBACK character is drawn, as itself or its stand-in",
            not bad, f"stand-ins for {unresolved[fam]!r}" + (f"; BAD {bad}" if bad else ""))
    if sf:
        rep("SF lacks '✕' and draws 'x' in its place (a missing glyph is detected, not boxed)",
            not glyph_ok("✕", "sans") and glyph_ok("▸", "sans") and text_w("✕") == text_w("x"))

    # -- forced non-SF, then no font at all
    dejavu = [p for p in SANS_PATHS if "DejaVu" in p and os.path.exists(p)]
    alt = resolve_fonts(sans_paths=tuple(dejavu), sans_names=())
    kind = os.path.basename(alt["sans"]) if alt["sans"] else "pygame default"
    probe.fill(WELL)
    wl = text(probe, "2 Airfoil ▸ Ranking", 10, 10)
    rep(f"a non-SF sans ({kind}) takes css px (no +1) and still draws",
        not _is_sf() and px_for("sans", 12) == 12 and wl > 0 and ink_box(probe).w > 0)
    bogus = resolve_fonts(sans_paths=("/nonexistent/font.ttf",), mono_paths=("/nonexistent/mono.ttf",),
                          sans_names=("no-such-font-name",), mono_names=("no-such-font-name",))
    probe.fill(WELL)
    wb = text(probe, "Run · 12/40 ▸ Stop", 10, 10)
    wm = text(probe, fmt(0.123456), 10, 30, "mono", 11)
    shown = _displayable("Run · 12/40 ▸ Stop", "sans")
    rep("bogus paths and names fall back to pygame's default font, legibly sized, missing glyphs as ASCII",
        bogus["sans"] is None and bogus["mono"] is None and wb > 0 and wm > 0 and ink_box(probe).w > 0
        and 10 <= font("sans", 12).get_height() <= 18 and "▸" not in shown and "> Stop" in shown,
        f"h {font('sans', 12).get_height()}, {shown!r}")
    back = resolve_fonts()
    rep("resolve_fonts() with no arguments puts the real fonts back", back == files)

    # -- set_scale
    set_scale(1.5)
    scaled = (LEFT_W, TREE_ROW_H, ICON_PX, KPI_BAR_W, px_for("sans", 12) if sf else None)
    kept = (TREE_SPLIT, PROP_KEY_FRAC, LOG_MAX, HINT_WORDS, TIP_WORDS, TOAST_S)
    set_scale(1.0)
    restored = all(globals()[n] == b for n, b in _BASE.items())
    rep("set_scale(1.5) scales the px constants (half up) and font px; set_scale(1.0) restores them",
        scaled == (414, 32, 21, 3, 20 if sf else None) and kept == (0.6, 0.44, 400, 24, 8, 5.0)
        and restored and S == 1.0 and (not sf or px_for("sans", 12) == 13), f"{scaled}")

    # -- words, colours, numbers
    short = "Screening flies every library section at the mission's operating point."
    s2 = ("Screening flies every library section at the operating point and ranks them by the stated "
          "objective. The seed of the shape search is the rank-1 section unless you pick another one.")
    s3 = ("the section is scored at the lift the wing needs at the design speed — every candidate flies "
          "the same point, at the same air density, with the drag ceiling as its penalty")
    s4 = " ".join(f"w{i}" for i in range(30))
    c1, c2, c3, c4 = (split_hint(t) for t in (short, s2, s3, s4))
    rep("split_hint: short -> (text, None); sentence; '—' kept on the lead; 24 words + ' …' with the whole text",
        c1 == (short, None)
        and c2 == ("Screening flies every library section at the operating point and ranks them by the "
                   "stated objective.",
                   "The seed of the shape search is the rank-1 section unless you pick another one.")
        and c3[0].endswith("design speed —")
        and c3[1] == "every candidate flies the same point, at the same air density, with the drag ceiling as its penalty"
        and c4 == (" ".join(f"w{i}" for i in range(24)) + " …", s4),
        f"{c3}")
    dec = ("The plate sits 0.488 m in front, i.e. the axle line, at every ride height. Then "
           + "more " * 15 + "words.")
    rep("split_hint: neither a decimal point nor 'i.e. the' is a sentence end",
        split_hint(dec)[0] == "The plate sits 0.488 m in front, i.e. the axle line, at every ride height.",
        repr(split_hint(dec)[0]))
    rep("tint(ACCENT) on white is #e8eff6; fade is opacity over the background",
        tint(ACCENT) == (232, 239, 246) and fade(INK, 1.0) == INK and fade(INK, 0.0, PANEL) == PANEL
        and fade(ACCENT, 0.6) == (119, 159, 200), f"{tint(ACCENT)} {fade(ACCENT, 0.6)}")
    probe.fill(WELL)
    blend_rgba(probe, (5, 5, 20, 20), GRID_A)
    blend_rgba(probe, (5, 5, 10, 10), GRID_A)
    px1, px2 = probe.get_at((20, 20))[:3], probe.get_at((7, 7))[:3]
    exp1 = tuple(round(c * 31 / 255 + 255 * (1 - 31 / 255)) for c in (24, 28, 34))
    rep("blend_rgba: GRID_A over white, and twice is darker", all(abs(a - b) <= 1 for a, b in zip(px1, exp1))
        and px2[0] < px1[0] and probe.get_at((30, 30))[:3] == WELL, f"{px1} vs {exp1}, {px2}")
    fm = [fmt(v) for v in (None, float("nan"), float("-inf"), 0.0, -3e-12, 1234567.0, 12.3456, 0.5, 5e-5,
                           -0.0012345, 48, "n/a")]
    rep("fmt: 4 g, '—' for none, '.3g' very large / small, '0' residue",
        fm == ["—", "—", "—", "0", "0", "1.23e+06", "12.35", "0.5", "5e-05", "-0.001234", "48", "n/a"], f"{fm}")

    if out:
        _specimen(out)
        if verbose:
            print(f"  specimen sheet: {out}")
    if verbose:
        print(f"cae.theme self-check: {'PASS' if ok else 'FAIL'}")
    return ok


def _specimen(path: str) -> None:
    """The type scale and the icons on a sheet, for a look against AeroBO's
    screenshots (SPECS/aerobo_shell.md §2.3)."""
    sheet = pygame.Surface((760, 420))
    sheet.fill(CANVAS)
    pygame.draw.rect(sheet, WELL, (8, 8, 744, 404))
    pygame.draw.rect(sheet, RULE, (8, 8, 744, 404), 1)
    y = 16
    rows = [("sans", 12, False, INK, "body 12 · Operating point ▸ 2 Airfoil — mission stated"),
            ("sans", 12, True, INK, "tree selected 12/600 · 2.8 Endplate"),
            ("sans", 11, True, INK_MUTED, "pane title 11/600 · SIMULATION  PROPERTIES"),
            ("sans", 11.5, False, INK, "field label 11.5 · Design speed  Drag ceiling ≤ 0.12"),
            ("sans", 11, False, INK_MUTED, "hint 11 · the section is scored at the lift the wing needs (?)"),
            ("sans", 10.5, False, INK_FAINT, "KPI label 10.5 · screen at Cl"),
            ("sans", 10, True, ACCENT, "TAG 10/600 · RUNNING · 12/40"),
            ("mono", 12, False, INK, "kv 12 mono · 1.0000  0.4882  -12.35  6.825e+05"),
            ("mono", 11, False, INK_MUTED, "log 11 mono · 00:11:24  seed NACA 4412, budget 40"),
            ("mono", 9.5, False, INK_FAINT, "chip 9.5 mono · s1223 · 40/40"),
            ("kpi", 19, False, INK, "6.825e+05"),
            ]
    for fam, css, bold, col, s in rows:
        h = font(fam, css, bold).get_height()
        text(sheet, s, 18, y, fam, css, col, bold, clip_w=720)
        text(sheet, f"{fam} {css} -> {px_for(fam, css)} px", 740, y, "mono", 9.5, INK_FAINT, anchor="right")
        y += h + 6
    tw = text_w("RUNNING · 12/40", "sans", 10, True)
    pygame.draw.rect(sheet, tint(ACCENT), (18, y, tw + 10, TAG_H))
    pygame.draw.rect(sheet, ACCENT, (18, y, tw + 10, TAG_H), 1)
    text(sheet, "RUNNING · 12/40", 18 + 5, y + 1, "sans", 10, ACCENT, True)
    y += TAG_H + 10
    x = 18
    for i, name in enumerate(ICONS):
        icon(sheet, name, x, y, TOOL_ICON, ACCENT if i % 7 == 0 else INK_MUTED)
        x += TOOL_ICON + 6
        if x > 730:
            x, y = 18, y + TOOL_ICON + 6
    pygame.image.save(sheet, path)


if __name__ == "__main__":
    import sys
    _out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else ""
    sys.exit(0 if self_check(out=_out) else 1)
