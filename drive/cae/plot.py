"""drive/cae/plot.py -- `Figure`, AeroBO's plotly look (gui/v3/figstyle.plot)
drawn with pygame: white paper, the desk palette, MONO tick labels, a faint
grid, a horizontal legend above the axes and no title (the card around a
figure is its caption).

A Figure is a list of things to draw, built fresh by a view every frame and
drawn into whatever rect the work area hands it:

    fig = Figure(xlabel="evaluation", ylabel="best objective")
    fig.line(n, best, name="best so far")
    fig.off_scale(refused_n)
    fig.draw(surf, rect)

The rules are plotly's where plotly has one, because the screenshots are
plotly's (SPECS/aerobo_shell.md §6.16, aerobo_wing.md §0.3):

  * ticks: plotly's own auto-tick -- about one per 80 px on x and 40 px on
    y, the step rounded UP to 2, 5 or 10 x 10^n -- and its labels (digits
    from the step, trailing zeros dropped, an SI suffix past 1e±3, a real
    minus sign);
  * margins l54 r14 t10 b40 that GROW to fit the tick labels and the axis
    titles (plotly's automargin: a y title plus "95.5" ticks push the axis
    to 59 px), and a legend above the axes at 0.35 of the plot width;
  * autorange: the data span, padded 6 % a side only when the axis carries
    markers (a lines-only trace runs edge to edge); a single value v is
    v ± 1. A range given as xlim / ylim wins, end by end.

**Non-finite values are masked, never drawn and never raised on** (defect
9.1-1): a line is split at every NaN / ±inf into its finite runs, a marker
at one is skipped, and a figure with no finite data at all draws its empty
state. A refused evaluation is scored -inf, so an optimiser trace can start
there; `off_scale` is how the figure shows those.

Colours are theme tokens (`from drive.cae import theme as T`); a trace given
no colour takes the next colour of the colorway, T.SERIES, as plotly does.
Every px (margins, widths, marker sizes, fonts) is multiplied by T.S when it
is drawn. Pure pygame + numpy.
"""

from __future__ import annotations

import math

import numpy as np
import pygame
import pygame.gfxdraw

from . import theme as T

#: plotly's dtick rounding set: the rough step is rounded UP (strictly) to
#: 2, 5 or 10 times its power of ten, so a step is never "1 x 10^n" by
#: accident of the division (07's shape plot: 0.2, not 0.1)
_ROUND = (2.0, 5.0, 10.0)
#: plotly's px per tick: nticks = clamp(axis length / this, 4, 9) + 1
_TICK_PX = {"x": 80, "y": 40}
#: SI suffixes of plotly's default exponent format "B" (1e9 is "B", not "G")
_SI = {-15: "f", -12: "p", -9: "n", -6: "μ", -3: "m", 3: "k", 6: "M", 9: "B", 12: "T"}
_MINUS = "−"
#: autorange padding for an axis that carries markers, each side
_PAD = 0.06

# layout, at S = 1 (measured on 12_opt_stopped / 18_wing_launch_4s: the axis
# lines sit just outside the plot area, tick digits 5 px under the x axis,
# the x title's ascender 28 px under it, y labels 2 px left of the y axis
# and the rotated y title 15 px left of the widest label, 3 px in from the
# figure's edge when it pushes the axis: "95.5" + "best objective" -> 59)
_TICK_GAP = 2                   # tick label <-> axis line
_TITLE_GAP = 15                 # widest y tick label <-> y title
_EDGE = 4                       # figure edge <-> y title (plus the axis column)
_XTITLE_DY = 24                 # axis line -> top of the x title's line box
_XTITLE_PAD = 9                 # under the x title's line box
_LEGEND_DY = 10                 # figure top -> legend box
_LEGEND_H = 31                  # one row, border included
_LEGEND_ROW = 19                # each further row of a wrapped legend
_LEGEND_SWATCH = 40             # an entry's symbol column; its text starts here
_LEGEND_GAP = 5                 # entry text -> next entry
_LEGEND_UNDER = 2               # legend box -> plot area

_rotated: dict = {}             # (s, px, face) -> the y title turned 90 degrees


def _px(v: float) -> int:
    """A layout constant at the current scale, rounded half up."""
    return int(math.floor(v * T.S + 0.5))


def _label_ink(colour):
    """A shape's label colour: its line's, except INK_FAINT, whose small
    text is drawn in INK_FAINT_TEXT (the readability pass; the line keeps
    INK_FAINT)."""
    return T.INK_FAINT_TEXT if tuple(colour[:3]) == T.INK_FAINT else colour


def _floats(v) -> np.ndarray:
    """`v` as a flat float array; None and anything that is not a number
    become NaN (and are then masked like any other non-finite value)."""
    if v is None:
        return np.zeros(0)
    try:
        return np.asarray(v, dtype=float).ravel()
    except (TypeError, ValueError):
        out = []
        for e in np.asarray(v, dtype=object).ravel():
            try:
                out.append(float(e))
            except (TypeError, ValueError):
                out.append(math.nan)
        return np.asarray(out, dtype=float)


def _pair(x, y):
    """Two coordinate arrays cut to the shorter one's length."""
    xs, ys = _floats(x), _floats(y)
    n = min(xs.size, ys.size)
    return xs[:n], ys[:n]


def _runs(ok: np.ndarray):
    """(start, stop) of every stretch of consecutive True in `ok`."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], ok.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def finite_runs(xs, ys, closed: bool = False):
    """The polyline (xs, ys) split at every non-finite point: `(runs,
    whole)`, `runs` the (xs, ys) stretches of consecutive finite points at
    least 2 long, `whole` True when no point was dropped. A closed outline
    with a gap is opened AT the gap (rotated to start after it), so the
    segment across the seam survives. (garage_ui.Plot does the same for the
    old dark pages.)"""
    xs, ys = _pair(xs, ys)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if ok.all():
        return ([(xs, ys)] if xs.size >= 2 else []), True
    if closed and ok.any():
        k = int(np.flatnonzero(~ok)[0]) + 1
        xs, ys, ok = np.roll(xs, -k), np.roll(ys, -k), np.roll(ok, -k)
    return [(xs[a:b], ys[a:b]) for a, b in _runs(ok) if b - a >= 2], False


# -- ticks -----------------------------------------------------------------------------------------
def nice_ticks(lo: float, hi: float, length: float, axis: str = "x"):
    """plotly's linear auto-ticks for the range [lo, hi] drawn `length` px
    long: `(ticks, step)`. nticks = clamp(length / 80 (x) or 40 (y), 4, 9)
    + 1; the step is span / nticks rounded up to 2, 5 or 10 x 10^n."""
    span = hi - lo
    if not (math.isfinite(span) and span > 0):
        return [], 1.0
    nt = min(max(length / (_TICK_PX.get(axis, 80) * T.S), 4.0), 9.0) + 1.0
    rough = span / nt
    base = 10.0 ** math.floor(math.log10(rough))
    step = base * next((m for m in _ROUND if m > rough / base + 1e-12), 10.0)
    k0 = math.ceil(lo / step - 1e-9)
    ticks = []
    for k in range(k0, k0 + 64):
        t = k * step
        if t > hi + 1e-9 * step:
            break
        ticks.append(t)
    return ticks, step


def _minus() -> str:
    return _MINUS if T.glyph_ok(_MINUS, "mono") else "-"


def tick_labels(ticks, step: float, lo: float, hi: float) -> list:
    """plotly's labels for linear ticks: two digits past the step's leading
    digit, trailing zeros dropped ("0.05", "95.5", "10"), an SI suffix when
    the range's magnitude passes 1e±3 ("20k", "100μ"), a real minus sign."""
    maxend = max(abs(lo), abs(hi))
    rexp = math.floor(math.log10(maxend) + 0.01) if maxend > 0 else 0
    exp = 3 * math.floor((rexp - 1) / 3 + 0.5) if abs(rexp) > 3 else 0
    if exp and exp not in _SI:
        return [f"{t:.3g}".replace("-", _minus()) for t in ticks]
    digits = max(0, 2 - math.floor(math.log10(step) + 0.01) + exp)
    out = []
    for t in ticks:
        s = f"{abs(t) / 10.0 ** exp:.{digits}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        if s == "0":
            out.append("0")
            continue
        out.append((_minus() if t < 0 else "") + s + _SI.get(exp, ""))
    return out


def _log_ticks(lo: float, hi: float, length: float, axis: str):
    """plotly's log-axis ticks, in log10 units: `(ticks, labels)`. Whole
    decades when a tick would span more than 0.7 of one; inside one decade,
    linear ticks labelled in full; otherwise 1-2-5 (or every digit) per
    decade, the decades labelled in full and the rest by their digit."""
    span = hi - lo
    if not (math.isfinite(span) and span > 0):
        return [], []
    nt = min(max(length / (_TICK_PX.get(axis, 80) * T.S), 4.0), 9.0) + 1.0
    rough = span / nt
    if rough > 0.7:
        step = math.ceil(rough)
        ticks = [float(k) for k in range(math.ceil(lo - 1e-9), math.floor(hi + 1e-9) + 1) if k % step == 0]
        return ticks, [_value_label(10.0 ** t) for t in ticks]
    if span < 1:
        vt, vstep = nice_ticks(10.0 ** lo, 10.0 ** hi, length, axis)
        vt = [v for v in vt if v > 0]
        return [math.log10(v) for v in vt], tick_labels(vt, vstep, 10.0 ** lo, 10.0 ** hi)
    digits = (1, 2, 5) if rough > 0.3 else tuple(range(1, 10))
    ticks, labels = [], []
    for dec in range(math.floor(lo) - 1, math.ceil(hi) + 1):
        for d in digits:
            t = dec + math.log10(d)
            if lo - 1e-9 <= t <= hi + 1e-9:
                ticks.append(t)
                labels.append(_value_label(10.0 ** dec) if d == 1 else str(d))
    return ticks, labels


def _value_label(v: float) -> str:
    """One value in plotly's style: plain up to 1e±3, SI-suffixed past it."""
    if v == 0:
        return "0"
    e = math.floor(math.log10(abs(v)) + 1e-9)
    exp = 3 * math.floor(e / 3) if abs(e) > 3 else 0
    if exp and exp not in _SI:
        return f"{v:.3g}"
    s = f"{abs(v) / 10.0 ** exp:.6f}".rstrip("0").rstrip(".")
    return (_minus() if v < 0 else "") + s + _SI.get(exp, "")


# -- stroking ----------------------------------------------------------------------------------------
def _dash_pattern(dash, width: float):
    """plotly's dash arrays (drawing.dashStyle): dot = w, w; dash = 3w, 3w,
    with w = max(line width, 3). None = solid."""
    if not dash:
        return None
    w = max(float(width), 3.0 * T.S)
    return (w, w) if dash == "dot" else (3.0 * w, 3.0 * w)


def _dashes(pts: np.ndarray, pattern):
    """The on-pieces of a polyline under a dash `pattern` (on, off), the
    pattern running on across the vertices as a browser's does."""
    on, off = pattern
    period = on + off
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    out, cur = [], []
    pos = 0.0                       # arc length along the pattern
    for i, L in enumerate(seg):
        p0, p1 = pts[i], pts[i + 1]
        t = 0.0
        while t < L - 1e-9:
            phase = pos % period
            if phase < on:
                run = min(on - phase, L - t)
                a = p0 + (p1 - p0) * (t / L)
                b = p0 + (p1 - p0) * ((t + run) / L)
                if not cur:
                    cur = [a]
                cur.append(b)
            else:
                run = min(period - phase, L - t)
                if cur:
                    out.append(np.array(cur))
                    cur = []
            t += run
            pos += run
    if len(cur) >= 2:
        out.append(np.array(cur))
    return out


def _stroke(surf, colour, pts: np.ndarray, width: float, dash=None) -> None:
    """An anti-aliased polyline `width` px wide (float px points). Width 1
    is pygame's aaline; a wider line is a solid core one px narrower, whose
    anti-aliased edges make up the last px, with a round joint at each
    vertex from 2.5 px up."""
    if pts.shape[0] < 2:
        return
    if dash:
        for piece in _dashes(pts, _dash_pattern(dash, width)):
            _stroke(surf, colour, piece, width)
        return
    w = max(1.0, float(width))
    if w < 1.5:
        pygame.draw.aalines(surf, colour, False, pts.tolist())
        return
    h = max(0.5, (w - 1.0) / 2.0)
    d = np.diff(pts, axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    keep = L > 1e-6
    for (x0, y0), (dx, dy), l in zip(pts[:-1][keep], d[keep], L[keep]):
        nx, ny = -dy / l * h, dx / l * h
        quad = ((x0 + nx, y0 + ny), (x0 + dx + nx, y0 + dy + ny),
                (x0 + dx - nx, y0 + dy - ny), (x0 - nx, y0 - ny))
        pygame.draw.polygon(surf, colour, quad)
        pygame.draw.aalines(surf, colour, True, quad)
    if w >= 2.5:
        r = max(1, int(round(h)))
        for x, y in pts[1:-1]:
            pygame.draw.circle(surf, colour, (int(round(x)), int(round(y))), r)


def _marker(surf, symbol: str, x: float, y: float, size: float, colour, alpha: int = 255) -> None:
    """A plotly marker `size` px across (plotly's size is the diameter)."""
    c = tuple(colour[:3]) + (alpha,)
    xi, yi = int(round(x)), int(round(y))
    r = max(1, int(round(size / 2.0)))
    if symbol == "diamond":
        pts = [(xi, yi - r - 1), (xi + r + 1, yi), (xi, yi + r + 1), (xi - r - 1, yi)]
        pygame.gfxdraw.filled_polygon(surf, pts, c)
        pygame.gfxdraw.aapolygon(surf, pts, c)
    elif symbol == "x":
        # plotly's "x" is a bold cross, its arms about a third of its size wide
        a = max(1.0, size / 3.5)
        for sx in (1, -1):
            p0, p1 = (x - r * sx, y - r), (x + r * sx, y + r)
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            ln = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / ln * a / 2, dx / ln * a / 2
            quad = [(p0[0] + nx, p0[1] + ny), (p1[0] + nx, p1[1] + ny),
                    (p1[0] - nx, p1[1] - ny), (p0[0] - nx, p0[1] - ny)]
            pygame.gfxdraw.filled_polygon(surf, [(int(round(u)), int(round(v))) for u, v in quad], c)
            pygame.gfxdraw.aapolygon(surf, [(int(round(u)), int(round(v))) for u, v in quad], c)
    elif symbol == "hollow":
        pygame.gfxdraw.aacircle(surf, xi, yi, r, c)
        if r > 2:
            pygame.gfxdraw.aacircle(surf, xi, yi, r - 1, c)
    else:                               # "circle"
        pygame.gfxdraw.filled_circle(surf, xi, yi, r, c)
        pygame.gfxdraw.aacircle(surf, xi, yi, r, c)


def _polygon(surf, pts: np.ndarray, rgba) -> None:
    """A translucent filled polygon (gfxdraw blends and honours the clip)."""
    if pts.shape[0] < 3:
        return
    pygame.gfxdraw.filled_polygon(surf, [(int(round(x)), int(round(y))) for x, y in pts], tuple(rgba))


def _rotated_title(s: str) -> pygame.Surface:
    """The y title (SANS 11 INK) turned to read bottom to top, cached."""
    px = T.px_for("sans", 11)
    key = (s, px, T.font_file("sans"))
    img = _rotated.get(key)
    if img is None:
        if len(_rotated) > 256:
            _rotated.clear()
        flat = pygame.Surface((max(1, T.text_w(s, "sans", 11)), T.font("sans", 11).get_height()), pygame.SRCALPHA)
        T.text(flat, s, 0, 0, "sans", 11, T.INK)
        img = pygame.transform.rotate(flat, 90)
        _rotated[key] = img
    return img


# -- the figure --------------------------------------------------------------------------------------
class Figure:
    """One 2-D plot, plotly's look. Build it (`line`, `scatter`, `hline`,
    `vline`, `band`, `fill_between`, `text`, `off_scale`), then `draw`.

    After `draw`, `plot_rect` is the plot area, `xrange` / `yrange` /
    `y2range` the ranges drawn (log axes: in log10 units), `legend_rect`
    the legend box (None when there is no legend) and `to_px(x, y)` maps a
    data point to the screen.
    """

    def __init__(self, *, xlabel="", ylabel="", y2label=None, equal=False, xrev=False, yrev=False,
                 xlim=None, ylim=None, y2lim=None, legend=True, margins=(54, 14, 10, 40), y2log=False,
                 ylog=False, empty_text=""):
        self.xlabel, self.ylabel, self.y2label = xlabel or "", ylabel or "", y2label
        self.equal, self.xrev, self.yrev = bool(equal), bool(xrev), bool(yrev)
        self.xlim, self.ylim, self.y2lim = xlim, ylim, y2lim
        self.legend = bool(legend)
        self.margins = tuple(margins)
        self.ylog, self.y2log = bool(ylog), bool(y2log)
        self.empty_text = empty_text or ""
        self._traces: list = []         # lines, scatters, fills, off-scale: drawn in order
        self._shapes: list = []         # bands (under the traces), hlines / vlines (over them)
        self._notes: list = []          # text annotations, on top
        self._next = 0                  # colorway position
        self.plot_rect = None
        self.xrange = self.yrange = self.y2range = None
        self.legend_rect = None

    # -- building ------------------------------------------------------------------------------------
    def _colour(self, colour):
        if colour is not None:
            return tuple(colour)
        c = T.SERIES[self._next % len(T.SERIES)]
        self._next += 1
        return c

    def line(self, x, y, *, colour=None, width=2, dash=None, name=None, axis="y", step=False,
             markers=None, marker_size=4, marker_colours=None, fill=None, closed=False):
        """A polyline. `dash` None | "dash" | "dot"; `step` draws post-steps
        (each value held until the next x); `markers` None | "circle" |
        "diamond" | "x" | "hollow" at every point (`marker_colours` one per
        point); `fill` an rgba (T.BAND_A) filling the outline as a polygon;
        `closed` joins the last point to the first. `colour` None = the next
        colorway colour."""
        xs, ys = _pair(x, y)
        self._traces.append(dict(kind="line", x=xs, y=ys, colour=self._colour(colour), width=float(width),
                                 dash=dash, name=name, axis=axis, step=bool(step), markers=markers,
                                 size=float(marker_size), mcolours=marker_colours, fill=fill,
                                 closed=bool(closed)))

    def scatter(self, x, y, *, colour=None, symbol="circle", size=6, name=None, axis="y", colours=None,
                opacity=1.0):
        """Markers only: `symbol` "circle" | "diamond" | "x" | "hollow",
        `size` px across, `colours` one per point, `opacity` 0-1."""
        xs, ys = _pair(x, y)
        self._traces.append(dict(kind="scatter", x=xs, y=ys, colour=self._colour(colour), symbol=symbol,
                                 size=float(size), name=name, axis=axis, colours=colours,
                                 alpha=int(round(255 * max(0.0, min(1.0, float(opacity)))))))

    def hline(self, y, *, colour=None, width=1, dash=None, label=None, axis="y"):
        """A horizontal line across the plot at `y` (drawn over the traces,
        not counted in the autorange); `label` above its right end."""
        self._shapes.append(dict(kind="hline", v=float(_floats([y])[0]) if y is not None else math.nan,
                                 colour=tuple(colour or T.INK_MUTED), width=float(width), dash=dash,
                                 label=label, axis=axis))

    def vline(self, x, *, colour=None, width=1, dash=None, label=None):
        """A vertical line at `x`; `label` beside its top."""
        self._shapes.append(dict(kind="vline", v=float(_floats([x])[0]) if x is not None else math.nan,
                                 colour=tuple(colour or T.INK_MUTED), width=float(width), dash=dash,
                                 label=label))

    def band(self, x0, x1, *, colour=None, alpha=0.12, label=None):
        """A full-height x band (plotly vrect) under the traces, `colour` at
        `alpha`; `label` at its top-left, SANS 11 (T.ANNOT_CSS; AeroBO 10)."""
        a, b = _floats([x0, x1])
        self._shapes.append(dict(kind="band", a=a, b=b, colour=tuple(colour or T.INK_MUTED),
                                 alpha=float(alpha), label=label))

    def fill_between(self, x, y_lo, y_hi, *, rgba=None):
        """The region between two curves (default T.BAND_A), split wherever
        any of the three is non-finite."""
        xs, lo = _pair(x, y_lo)
        _, hi = _pair(x, y_hi)
        n = min(xs.size, lo.size, hi.size)
        self._traces.append(dict(kind="between", x=xs[:n], lo=lo[:n], hi=hi[:n], rgba=tuple(rgba or T.BAND_A),
                                 axis="y", name=None))

    def text(self, x, y, s, *, colour=None, size=10, anchor="left"):
        """An annotation at a data point: SANS `size`, vertically centred on
        `y`, `anchor` left | right | centre on `x`."""
        self._notes.append(dict(x=x, y=y, s=str(s), colour=tuple(colour or T.INK_MUTED), size=size,
                                anchor=anchor))

    def off_scale(self, xs, *, name="refused / off scale"):
        """Points that have no value to plot (refused, -inf, off the scale):
        BAD "x" markers, size 7, along the floor at y = lo + 0.02 (hi - lo);
        in the legend as "{name} ({k})". Their x counts in the x range;
        they never stretch the y axis."""
        x = _floats(xs)
        x = x[np.isfinite(x)]
        self._traces.append(dict(kind="off", x=x, y=np.zeros(x.size), colour=tuple(T.BAD), name=name,
                                 axis="y", size=7.0))

    # -- ranges ----------------------------------------------------------------------------------------
    def _log(self, axis: str) -> bool:
        return (axis == "y" and self.ylog) or (axis == "y2" and self.y2log)

    def _tf(self, v: np.ndarray, log: bool) -> np.ndarray:
        """Axis units: log10 for a log axis, where a value <= 0 is masked."""
        if not log:
            return v
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.log10(np.where(v > 0, v, np.nan))
        return out

    def _data_range(self, which: str):
        """The (lo, hi, has_markers) of the finite data on axis `which`
        ("x", "y", "y2"), in axis units; lo None when there is none."""
        vals, marked = [], False
        for tr in self._traces:
            if which == "x":
                v = tr["x"]
            elif tr.get("axis", "y") != which or tr["kind"] == "off":
                continue
            elif tr["kind"] == "between":
                v = np.concatenate((tr["lo"], tr["hi"]))
            else:
                v = self._tf(tr["y"], self._log(which))
            if tr["kind"] in ("scatter", "off") or (tr["kind"] == "line" and tr["markers"]):
                marked = True
            v = v[np.isfinite(v)]
            if v.size:
                vals.append((float(v.min()), float(v.max())))
        if not vals:
            return None, None, marked
        return min(a for a, _ in vals), max(b for _, b in vals), marked

    def _range(self, which: str, lim):
        """The axis range: the data's (padded 6 % a side when it carries
        markers; v ± 1 for a single value), each end overridden by `lim`."""
        lo, hi, marked = self._data_range(which)
        log = self._log(which)
        if lo is None:
            lo, hi = (0.0, 1.0)
        elif hi - lo <= 1e-12 * max(1.0, abs(lo)):
            lo, hi = lo - 1.0, hi + 1.0
        elif marked:
            pad = _PAD * (hi - lo)
            lo, hi = lo - pad, hi + pad
        if lim is not None:
            a, b = (list(lim) + [None, None])[:2]
            if a is not None and math.isfinite(float(a)) and (not log or a > 0):
                lo = math.log10(a) if log else float(a)
            if b is not None and math.isfinite(float(b)) and (not log or b > 0):
                hi = math.log10(b) if log else float(b)
        if not hi > lo:
            lo, hi = lo - 1.0, lo + 1.0
        return lo, hi

    def _has(self, axis: str) -> bool:
        return any(tr.get("axis", "y") == axis for tr in self._traces)

    def _finite_any(self) -> bool:
        for tr in self._traces:
            if tr["kind"] == "off":
                if tr["x"].size:
                    return True
                continue
            ys = np.concatenate((tr["lo"], tr["hi"])) if tr["kind"] == "between" else \
                self._tf(tr["y"], self._log(tr.get("axis", "y")))
            xs = np.concatenate((tr["x"], tr["x"])) if tr["kind"] == "between" else tr["x"]
            if (np.isfinite(xs) & np.isfinite(ys)).any():
                return True
        return False

    # -- mapping ---------------------------------------------------------------------------------------
    def _map(self, v: np.ndarray, rng, lo_px: float, hi_px: float, rev: bool) -> np.ndarray:
        a, b = rng
        if rev:
            a, b = b, a
        return lo_px + (v - a) / (b - a) * (hi_px - lo_px)

    def _xpx(self, v):
        P = self.plot_rect
        return self._map(np.asarray(v, dtype=float), self.xrange, P.x, P.right - 1, self.xrev)

    def _ypx(self, v, axis="y"):
        P = self.plot_rect
        rng = self.y2range if axis == "y2" else self.yrange
        return self._map(np.asarray(v, dtype=float), rng, P.bottom - 1, P.y, self.yrev and axis != "y2")

    def to_px(self, x, y, axis="y"):
        """Screen position of a data point after `draw` (None before, or for
        a point with no place on the axes)."""
        if self.plot_rect is None:
            return None
        xv = float(_floats([x])[0])
        yv = float(self._tf(_floats([y]), self._log(axis))[0])
        if not (math.isfinite(xv) and math.isfinite(yv)):
            return None
        return float(self._xpx(xv)), float(self._ypx(yv, axis))

    # -- legend ----------------------------------------------------------------------------------------
    def _entries(self):
        out = []
        for tr in self._traces:
            if tr.get("name") is None:
                continue
            label = str(tr["name"])
            if tr["kind"] == "off":
                if not tr["x"].size:
                    continue
                label = f"{label} ({tr['x'].size})"
            out.append((label, tr))
        return out

    def _legend_rows(self, entries, avail: int):
        """Entries laid out left to right, wrapped to rows no wider than
        `avail`: [[(x_offset, label, trace), ...], ...] and the widest row."""
        rows, row, x, widest = [], [], 0, 0
        sw, gap = _px(_LEGEND_SWATCH), _px(_LEGEND_GAP)
        for label, tr in entries:
            w = sw + T.text_w(label, "sans", T.NOTE_CSS) + gap
            if row and x + w > avail:
                rows.append(row)
                widest = max(widest, x)
                row, x = [], 0
            row.append((x, label, tr))
            x += w
        if row:
            rows.append(row)
            widest = max(widest, x)
        return rows, widest

    def _draw_legend(self, surf, rows, box: pygame.Rect) -> None:
        T.blend_rgba(surf, box, (255, 255, 255, 217))          # WELL at 85 %
        pygame.draw.rect(surf, T.RULE_SOFT, box, 1)
        th = T.font("sans", T.NOTE_CSS).get_height()
        for i, row in enumerate(rows):
            cy = box.y + _px(15) + i * _px(_LEGEND_ROW)
            for x0, label, tr in row:
                ix = box.x + 1 + x0
                self._swatch(surf, tr, ix, cy)
                T.text(surf, label, ix + _px(_LEGEND_SWATCH), cy - th // 2, "sans", T.NOTE_CSS, T.INK_MUTED)

    def _swatch(self, surf, tr, ix: int, cy: int) -> None:
        cx = ix + _px(20)
        if tr["kind"] == "off":
            _marker(surf, "x", cx, cy, tr["size"] * T.S, tr["colour"])
        elif tr["kind"] == "scatter":
            _marker(surf, tr["symbol"], cx, cy, min(tr["size"], 12) * T.S, tr["colour"], tr["alpha"])
        else:
            if tr["fill"] is not None:
                r = pygame.Rect(ix + _px(6), cy - _px(5), _px(28), _px(10))
                pygame.gfxdraw.box(surf, r, tuple(tr["fill"]))
            x0, x1 = ix + _px(5), ix + _px(35)
            _stroke(surf, tr["colour"], np.array([[x0, cy], [x1, cy]], float), tr["width"] * T.S, tr["dash"])
            if tr["markers"]:
                _marker(surf, tr["markers"], cx, cy, min(tr["size"], 12) * T.S, tr["colour"])

    # -- drawing ---------------------------------------------------------------------------------------
    def draw(self, surf, rect) -> None:
        """Draw the figure into `rect` (paper included). Never raises on
        data: non-finite values are masked (module docstring)."""
        fig = pygame.Rect(rect)
        self.plot_rect = None
        self.legend_rect = None
        if fig.w < 16 or fig.h < 16:
            return
        surf.fill(T.WELL, fig)
        empty = not self._finite_any()
        if empty and self.empty_text:
            h = T.font("sans", 12).get_height()
            T.text(surf, self.empty_text, fig.centerx, fig.centery - h // 2, "sans", 12, T.INK_FAINT,
                   anchor="centre", clip_w=fig.w - 16)
            return
        clip0 = surf.get_clip()
        surf.set_clip(fig.clip(clip0) if clip0 else fig)
        try:
            self._draw(surf, fig, empty)
        finally:
            surf.set_clip(clip0)

    def _layout(self, fig: pygame.Rect, twin: bool, entries):
        """Margins (base, grown to fit labels and titles, plotly's
        automargin) and the plot rect; the ranges and ticks on it."""
        ml, mr, mt, mb = (_px(m) for m in self.margins)
        rows, widest = [], 0
        top = mt
        if self.legend and entries:
            rows, widest = self._legend_rows(entries, fig.w - _px(20))
            top = max(mt, _px(_LEGEND_DY) + _px(_LEGEND_H) + (len(rows) - 1) * _px(_LEGEND_ROW)
                      + _px(_LEGEND_UNDER))
        bottom = mb
        if self.xlabel:
            bottom = max(mb, _px(_XTITLE_DY) + T.font("sans", 11).get_height() + _px(_XTITLE_PAD))
        left, right = ml, mr
        self.xrange = self._range("x", self.xlim)
        self.yrange = self._range("y", self.ylim)
        self.y2range = self._range("y2", self.y2lim) if twin else None
        for _ in range(2):                      # labels set the margins, the margins the ticks
            P = pygame.Rect(fig.x + left, fig.y + top, max(8, fig.w - left - right), max(8, fig.h - top - bottom))
            if self.equal:
                self._equalise(P)
            yt = self._ticks("y", P)
            wy = max((T.text_w(s, "mono", 10) for s in yt[1]), default=0)
            need = _px(_TICK_GAP) + wy
            if self.ylabel:
                need += _px(_TITLE_GAP) + _rotated_title(self.ylabel).get_width() + _px(_EDGE)
            left = max(ml, need)
            if twin:
                y2t = self._ticks("y2", P)
                w2 = max((T.text_w(s, "mono", 10) for s in y2t[1]), default=0)
                need2 = _px(_TICK_GAP) + w2 + _px(_EDGE)
                if self.y2label:
                    need2 += _px(_TITLE_GAP) + _rotated_title(self.y2label).get_width()
                right = max(mr, need2)
        P = pygame.Rect(fig.x + left, fig.y + top, max(8, fig.w - left - right), max(8, fig.h - top - bottom))
        if self.equal:
            self._equalise(P)
        return P, rows, widest

    def _equalise(self, P: pygame.Rect) -> None:
        """One scale on both axes: the looser range grows about its centre
        (the data ranges are re-read first, so repeated passes agree)."""
        self.xrange = self._range("x", self.xlim)
        self.yrange = self._range("y", self.ylim)
        (x0, x1), (y0, y1) = self.xrange, self.yrange
        sx, sy = P.w / (x1 - x0), P.h / (y1 - y0)
        if sx > sy:
            c, half = 0.5 * (x0 + x1), 0.5 * P.w / sy
            self.xrange = (c - half, c + half)
        else:
            c, half = 0.5 * (y0 + y1), 0.5 * P.h / sx
            self.yrange = (c - half, c + half)

    def _ticks(self, axis: str, P: pygame.Rect):
        """(tick values in axis units, labels) for `axis` on plot rect P."""
        rng = {"x": self.xrange, "y": self.yrange, "y2": self.y2range}[axis]
        length = P.w if axis == "x" else P.h
        kind = "x" if axis == "x" else "y"
        if axis != "x" and self._log(axis):
            return _log_ticks(rng[0], rng[1], length, kind)
        ticks, step = nice_ticks(rng[0], rng[1], length, kind)
        return ticks, tick_labels(ticks, step, rng[0], rng[1])

    def _draw(self, surf, fig: pygame.Rect, empty: bool) -> None:
        twin = self._has("y2")
        entries = [] if empty else self._entries()
        P, rows, widest = self._layout(fig, twin, entries)
        self.plot_rect = P
        xt, xl = self._ticks("x", P)
        yt, yl = self._ticks("y", P)
        # grid, then the zero lines (under the data)
        for t in xt:
            px = int(round(float(self._xpx(t))))
            T.blend_rgba(surf, (px, P.y, 1, P.h), T.GRID_A)
        for t in yt:
            py = int(round(float(self._ypx(t))))
            T.blend_rgba(surf, (P.x, py, P.w, 1), T.GRID_A)
        if self.xrange[0] < 0 < self.xrange[1]:
            T.blend_rgba(surf, (int(round(float(self._xpx(0.0)))), P.y, 1, P.h), T.ZERO_A)
        if not self.ylog and self.yrange[0] < 0 < self.yrange[1]:
            T.blend_rgba(surf, (P.x, int(round(float(self._ypx(0.0)))), P.w, 1), T.ZERO_A)
        clip0 = surf.get_clip()
        surf.set_clip(P.clip(clip0))
        try:
            for sh in self._shapes:
                if sh["kind"] == "band":
                    self._draw_band(surf, sh)
            for tr in self._traces:
                self._draw_trace(surf, tr)
            for sh in self._shapes:
                if sh["kind"] in ("hline", "vline"):
                    self._draw_rule(surf, sh)
        finally:
            surf.set_clip(clip0)
        for n in self._notes:
            p = self.to_px(n["x"], n["y"])
            if p is not None:
                h = T.font("sans", n["size"]).get_height()
                T.text(surf, n["s"], p[0], p[1] - h // 2, "sans", n["size"], n["colour"], anchor=n["anchor"])
        # axes over the data: lines just outside the plot area, then labels
        pygame.draw.line(surf, T.INK_MUTED, (P.x - 1, P.y), (P.x - 1, P.bottom), 1)
        pygame.draw.line(surf, T.INK_MUTED, (P.x - 1, P.bottom), (P.right, P.bottom), 1)
        th = T.font("mono", 10).get_height()
        for t, s in zip(xt, xl):
            T.text(surf, s, int(round(float(self._xpx(t)))), P.bottom + _px(3), "mono", 10, T.INK_MUTED,
                   anchor="centre")
        wy = 0
        for t, s in zip(yt, yl):
            wy = max(wy, T.text(surf, s, P.x - 1 - _px(_TICK_GAP), int(round(float(self._ypx(t)))) - th // 2,
                                "mono", 10, T.INK_MUTED, anchor="right"))
        if self.xlabel:
            T.text(surf, self.xlabel, P.centerx, P.bottom + _px(_XTITLE_DY), "sans", 11, T.INK, anchor="centre")
        if self.ylabel:
            img = _rotated_title(self.ylabel)
            x = P.x - 1 - _px(_TICK_GAP) - wy - _px(_TITLE_GAP) - img.get_width()
            surf.blit(img, (max(fig.x + 1, x), P.centery - img.get_height() // 2))
        if twin:
            pygame.draw.line(surf, T.INK_MUTED, (P.right, P.y), (P.right, P.bottom), 1)
            y2t, y2l = self._ticks("y2", P)
            w2 = 0
            for t, s in zip(y2t, y2l):
                w2 = max(w2, T.text(surf, s, P.right + 1 + _px(_TICK_GAP),
                                    int(round(float(self._ypx(t, "y2")))) - th // 2, "mono", 10, T.INK_MUTED))
            if self.y2label:
                img = _rotated_title(self.y2label)
                surf.blit(img, (P.right + 1 + _px(_TICK_GAP) + w2 + _px(_TITLE_GAP),
                                P.centery - img.get_height() // 2))
        if rows:
            h = _px(_LEGEND_H) + (len(rows) - 1) * _px(_LEGEND_ROW)
            w = widest + 2
            x = P.x + int(0.35 * P.w)
            if x + w > fig.right - _px(4):
                x = max(fig.x + _px(4), fig.right - _px(4) - w)
            box = pygame.Rect(x, fig.y + _px(_LEGEND_DY), w, h)
            self._draw_legend(surf, rows, box)
            self.legend_rect = box

    def _draw_band(self, surf, sh) -> None:
        if not (math.isfinite(sh["a"]) and math.isfinite(sh["b"])):
            return
        P = self.plot_rect
        a, b = sorted(float(v) for v in self._xpx([sh["a"], sh["b"]]))
        r = pygame.Rect(int(round(a)), P.y, max(1, int(round(b - a))), P.h)
        T.blend_rgba(surf, r, tuple(sh["colour"][:3]) + (int(round(255 * sh["alpha"])),))
        if sh["label"]:
            T.text(surf, sh["label"], r.x + _px(4), P.y + _px(2), "sans", T.ANNOT_CSS, _label_ink(sh["colour"]))

    def _draw_rule(self, surf, sh) -> None:
        if not math.isfinite(sh["v"]):
            return
        P = self.plot_rect
        w = sh["width"] * T.S
        if sh["kind"] == "hline":
            axis = sh["axis"]
            v = self._tf(np.array([sh["v"]]), self._log(axis))[0]
            if not math.isfinite(v):
                return
            y = float(round(float(self._ypx(v, axis))))       # a rule is crisp: whole pixels
            if not (P.y - 1 <= y <= P.bottom):
                return
            _stroke(surf, sh["colour"], np.array([[P.x, y], [P.right - 1, y]]), w, sh["dash"])
            if sh["label"]:
                h = T.font("sans", T.ANNOT_CSS).get_height()
                T.text(surf, sh["label"], P.right - _px(4), int(y) - h - _px(1), "sans", T.ANNOT_CSS,
                       _label_ink(sh["colour"]),
                       anchor="right")
        else:
            x = float(round(float(self._xpx(sh["v"]))))
            if not (P.x - 1 <= x <= P.right):
                return
            _stroke(surf, sh["colour"], np.array([[x, P.y], [x, P.bottom - 1]]), w, sh["dash"])
            if sh["label"]:
                T.text(surf, sh["label"], int(x) + _px(4), P.y + _px(2), "sans", T.ANNOT_CSS, _label_ink(sh["colour"]))

    def _points(self, xs, ys, axis) -> np.ndarray:
        return np.column_stack((self._xpx(xs), self._ypx(ys, axis)))

    def _draw_trace(self, surf, tr) -> None:
        kind, axis = tr["kind"], tr.get("axis", "y")
        if kind == "between":
            ok = np.isfinite(tr["x"]) & np.isfinite(tr["lo"]) & np.isfinite(tr["hi"])
            for a, b in _runs(ok):
                if b - a < 2:
                    continue
                top = self._points(tr["x"][a:b], tr["hi"][a:b], "y")
                bot = self._points(tr["x"][a:b], tr["lo"][a:b], "y")[::-1]
                _polygon(surf, np.vstack((top, bot)), tr["rgba"])
            return
        if kind == "off":
            lo, hi = self.yrange
            y = lo + 0.02 * (hi - lo)
            for x in tr["x"]:
                px, py = float(self._xpx(x)), float(self._ypx(y))
                _marker(surf, "x", px, py, tr["size"] * T.S, tr["colour"])
            return
        ys = self._tf(tr["y"], self._log(axis))
        if kind == "scatter":
            ok = np.isfinite(tr["x"]) & np.isfinite(ys)
            if not ok.any():
                return
            pts = self._points(tr["x"][ok], ys[ok], axis)
            cols = _per_point(tr["colours"], ok, tr["colour"])
            for (px, py), c in zip(pts, cols):
                _marker(surf, tr["symbol"], px, py, tr["size"] * T.S, c, tr["alpha"])
            return
        runs, whole = finite_runs(tr["x"], ys, tr["closed"])
        if tr["fill"] is not None:
            ok = np.isfinite(tr["x"]) & np.isfinite(ys)
            if ok.sum() >= 3:
                _polygon(surf, self._points(tr["x"][ok], ys[ok], axis), tr["fill"])
        w = tr["width"] * T.S
        for rx, ry in runs:
            if tr["step"]:
                rx = np.repeat(rx, 2)[1:]
                ry = np.repeat(ry, 2)[:-1]
            pts = self._points(rx, ry, axis)
            if tr["closed"] and whole:
                pts = np.vstack((pts, pts[:1]))
            if w > 0:
                _stroke(surf, tr["colour"], pts, w, tr["dash"])
        if tr["markers"]:
            ok = np.isfinite(tr["x"]) & np.isfinite(ys)
            pts = self._points(tr["x"][ok], ys[ok], axis)
            cols = _per_point(tr["mcolours"], ok, tr["colour"])
            for (px, py), c in zip(pts, cols):
                _marker(surf, tr["markers"], px, py, tr["size"] * T.S, c)


def _per_point(colours, ok: np.ndarray, default) -> list:
    """The per-point colours of the kept (finite) points; `default` where a
    list is missing or too short."""
    n = int(ok.size)
    if colours is None:
        return [default] * int(ok.sum())
    cols = list(colours)[:n] + [default] * max(0, n - len(colours))
    return [tuple(c) for c, k in zip(cols, ok) if k]


# ==================================================================== #
#  SELF-CHECK                                                          #
# ==================================================================== #
def self_check(verbose: bool = True, out: str = "") -> bool:
    """Masking, the twin axis, reversed and equal axes, off-scale markers,
    the legend, steps, dashes, fills, the empty state, plotly's ticks and
    labels, automargin and the draw cost. `out`: save a specimen sheet."""
    import os
    import time
    n_ok = n_all = 0
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok, n_ok, n_all
        n_all += 1
        n_ok += bool(passed)
        ok = ok and bool(passed)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}" + (f": {msg}" if msg else ""))

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    T.warm()
    sf = os.path.basename(T.font_file("sans") or "").startswith("SFNS")
    surf = pygame.Surface((900, 360))

    def fresh():
        surf.fill(T.CANVAS)
        return pygame.Rect(10, 10, 860, 300)

    def ink(colour, r, tol=12):
        """How many pixels of `r` are within `tol` of `colour`."""
        a = pygame.surfarray.pixels3d(surf)[r.x:r.right, r.y:r.bottom].astype(int)
        return int((np.abs(a - np.array(colour[:3])).max(axis=2) <= tol).sum())

    # 1 masking (defect 9.1-1): nothing raises, a NaN is a gap
    nan, inf = math.nan, math.inf
    raised = []
    for label, build in (
            ("-inf first", lambda f: f.line([1, 2, 3, 4], [-inf, 1, 2, 3])),
            ("NaN inside, markers", lambda f: f.line([1, 2, 3, 4, 5], [1, nan, 2, 3, 2], markers="circle")),
            ("+inf and NaN, step", lambda f: f.line([0, 1, 2, 3], [inf, 1, nan, 2], step=True)),
            ("all non-finite", lambda f: f.line([nan, nan], [inf, -inf])),
            ("one finite point", lambda f: f.line([1], [2])),
            ("NaN in x, closed + fill", lambda f: f.line([0, 1, nan, 0], [0, 1, 1, 1], closed=True, fill=T.BAND_A)),
            ("None / str entries", lambda f: f.line([0, 1, None, 3], [0, "x", 1, 2])),
            ("scatter + fill_between", lambda f: (f.scatter([1, nan, 3], [1, 2, inf]),
                                                  f.fill_between([0, 1, 2], [0, nan, 0], [1, 1, 1]))),
            ("log axis with <= 0", lambda f: f.line([1, 2, 3], [0, -1, 10])),
            ("y2 all NaN", lambda f: (f.line([0, 1], [0, 1]), f.line([0, 1], [nan, nan], axis="y2")))):
        f = Figure(xlabel="x", ylabel="y", ylog=(label == "log axis with <= 0"))
        try:
            build(f)
            f.draw(surf, fresh())
        except Exception as e:           # noqa: BLE001 -- the row reports it
            raised.append(f"{label}: {type(e).__name__} {e}")
    f = Figure(margins=(10, 10, 10, 10))
    f.line([0, 1, 2, 3, 4], [0, 0, nan, 0, 0], colour=T.BAD, width=2)
    f.draw(surf, fresh())
    P = f.plot_rect
    gx = int(round(float(f._xpx(2.0))))
    gap = ink(T.BAD, pygame.Rect(gx - 3, P.y, 7, P.h))
    runs_drawn = (ink(T.BAD, pygame.Rect(P.x + 5, P.y, 40, P.h)) > 0
                  and ink(T.BAD, pygame.Rect(P.right - 45, P.y, 40, P.h)) > 0)
    rep("non-finite values are masked: nothing raises (10 cases), a NaN leaves a gap, both runs draw",
        not raised and gap == 0 and runs_drawn, f"{raised} gap px {gap}")

    # 2 twin axis
    f = Figure(xlabel="evaluation", ylabel="c_l", y2label="α_eff [deg]")
    f.line([0, 10], [0.5, 0.5], colour=T.ACCENT)
    f.line([0, 10], [300, 300], colour=T.WARN, axis="y2")
    f.line([0, 10], [0, 1], colour=T.GOOD)
    f.line([0, 10], [0, 1000], colour=T.GOOD, axis="y2")
    f.draw(surf, fresh())
    ya, yb = f.to_px(5, 0.5)[1], f.to_px(5, 500, "y2")[1]
    y2r = f.y2range
    rep("twin axis: y2 has its own range and ticks on the right, the right margin grows",
        abs(ya - yb) <= 1.0 and y2r[0] <= 0 and y2r[1] >= 1000 and f.plot_rect.right < 900 - 14 - 20,
        f"mid rows {ya:.1f} / {yb:.1f}, y2 {y2r}, plot right {f.plot_rect.right}")

    # 3 reversed + equal aspect
    f = Figure(equal=True, xrev=True, xlabel="x/c", ylabel="y/c")
    f.line([0, 1], [0, 0.1])
    f.draw(surf, fresh())
    (x0, _), (x1, _) = f.to_px(0, 0), f.to_px(1, 0)
    sx = f.plot_rect.w / (f.xrange[1] - f.xrange[0])
    sy = f.plot_rect.h / (f.yrange[1] - f.yrange[0])
    rep("xrev draws x = 0 at the right; equal gives one px per unit on both axes",
        x0 > x1 and abs(sx / sy - 1) < 0.01, f"x0 {x0:.0f} x1 {x1:.0f}, {sx:.1f} vs {sy:.1f} px/unit")

    # 4 off-scale markers at the floor
    f = Figure(xlabel="evaluation", ylabel="objective")
    f.line(range(1, 21), np.linspace(20, 50, 20), name="best feasible so far")
    f.off_scale([3, 9, 15])
    f.draw(surf, fresh())
    lo, hi = f.yrange
    yfl = f.to_px(9, lo + 0.02 * (hi - lo))
    bad = ink(T.BAD, pygame.Rect(int(yfl[0]) - 5, int(yfl[1]) - 5, 11, 11), tol=40)
    labels = [lab for lab, _ in f._entries()]
    rep("off_scale: BAD x markers at lo + 2 % of the span, legend '(k)', y range from the scored values only",
        bad >= 12 and "refused / off scale (3)" in labels and f.yrange[0] > 15,
        f"{bad} BAD px, {labels}, y {f.yrange[0]:.1f}..{f.yrange[1]:.1f}")

    # 5 legend
    lr = f.legend_rect
    P = f.plot_rect
    f2 = Figure(legend=False)
    f2.line([0, 1], [0, 1], name="a")
    f2.line([0, 1], [1, 0], name="b")
    f2.draw(surf, fresh())
    rep("legend: above the axes at 0.35 of the plot width, WELL 85 % + RULE_SOFT; off with legend=False",
        lr is not None and abs(lr.x - (P.x + int(0.35 * P.w))) <= 1 and lr.bottom <= P.y
        and lr.y == 10 + 10 and lr.h == 31 and f2.legend_rect is None, f"{lr} over plot {P}")

    # 6 step lines
    f = Figure(margins=(10, 10, 10, 10))
    f.line([0, 1, 2], [0, 1, 1], step=True, colour=T.BAD, width=2)
    f.draw(surf, fresh())
    a, b = f.to_px(0.5, 0), f.to_px(0.5, 1)
    held = ink(T.BAD, pygame.Rect(int(a[0]) - 1, int(a[1]) - 2, 3, 5)) > 0
    not_diag = ink(T.BAD, pygame.Rect(int(b[0]) - 1, int(b[1]) - 2, 3, 5)) == 0
    rep("step=True holds each value to the next x (post-steps)", held and not_diag)

    # 7 dash and dot
    def pattern(dash):
        f = Figure(margins=(10, 10, 10, 10), xlim=(0, 1), ylim=(0, 1))
        f.hline(0.5, colour=T.BAD, width=1, dash=dash)
        f.draw(surf, fresh())
        y = int(round(f.to_px(0, 0.5)[1]))
        row = pygame.surfarray.pixels3d(surf)[f.plot_rect.x:f.plot_rect.right, y].astype(int)
        on = (np.abs(row - np.array(T.BAD)).max(axis=1) <= 60).astype(np.int8)
        rs = _runs(on.astype(bool))
        lens = [b - a for a, b in rs][1:-1]
        return (int(np.median(lens)) if lens else 0), len(rs)
    d9, nd = pattern("dash")
    d3, nt = pattern("dot")
    rep("dash / dot: plotly's arrays (9 px on / 9 off, 3 / 3 at widths <= 3)",
        8 <= d9 <= 10 and 2 <= d3 <= 4 and nt > nd > 10, f"dash {d9} px x{nd}, dot {d3} px x{nt}")

    # 8 fill
    f = Figure(margins=(10, 10, 10, 10), xlim=(0, 1), ylim=(0, 1))
    f.line([0.1, 0.9, 0.9, 0.1], [0.1, 0.1, 0.9, 0.9], closed=True, fill=T.BAND_A, colour=T.ACCENT)
    f.draw(surf, fresh())
    c = surf.get_at(tuple(int(v) for v in f.to_px(0.5, 0.5)))[:3]
    want = T.fade(T.BAND_A[:3], T.BAND_A[3] / 255)
    outside = surf.get_at(tuple(int(v) for v in f.to_px(0.03, 0.5)))[:3]
    rep("fill: BAND_A blended inside the closed outline, paper outside",
        all(abs(p - q) <= 3 for p, q in zip(c, want)) and tuple(outside) == T.WELL, f"{tuple(c)} vs {want}")

    # 9 empty state
    f = Figure(empty_text="no evaluations yet", xlabel="evaluation")
    f.line([nan], [nan])
    r = fresh()
    f.draw(surf, r)
    box = pygame.mask.from_threshold(surf, T.INK_FAINT, (40, 40, 40, 255)).get_bounding_rects()
    box = box[0].unionall(box[1:]) if box else pygame.Rect(0, 0, 0, 0)
    axes_drawn = ink(T.INK_MUTED, r, tol=2)
    f2 = Figure(xlabel="evaluation")
    f2.draw(surf, fresh())
    rep("empty: the message centred (SANS 12 INK_FAINT), no axes; without a message, just the frame",
        box.w > 60 and abs(box.centerx - r.centerx) <= 2 and abs(box.centery - r.centery) <= 4
        and axes_drawn == 0 and f.plot_rect is None and f2.plot_rect is not None, f"{box}")

    # 10 ticks and labels (plotly's numbers, measured on AeroBO's shots)
    t1, s1 = nice_ticks(-2.04, 55.96, 1221, "x")
    t2, s2 = nice_ticks(9.55, 55.4, 1221, "x")
    t3, s3 = nice_ticks(93.75, 95.76, 242, "y")
    t4, s4 = nice_ticks(-0.02, 1.02, 1210, "x")
    lab3 = tick_labels(t3, s3, 93.75, 95.76)
    lab5 = tick_labels(*nice_ticks(-0.06, 0.11, 200, "y"), -0.06, 0.11)
    big = tick_labels(*nice_ticks(0, 52000, 600, "x"), 0, 52000)
    rep("ticks: plotly's auto-ticks and labels (18: 10s, 20: 5s, 12: 0.5s, 07: 0.2s; '−0.05'; '20k')",
        s1 == 10 and t1[0] == 0 and s2 == 5 and len(t2) == 10 and s3 == 0.5 and lab3 == ["94", "94.5", "95", "95.5"]
        and abs(s4 - 0.2) < 1e-12 and lab5[0] in ("−0.05", "-0.05") and "0.05" in lab5 and "20k" in big,
        f"{s1} {s2} {s3} {s4} {lab3} {lab5} {big}")

    # 11 margins: the base, and automargin
    f = Figure(xlabel="evaluation", ylabel="objective")
    f.line([1, 53], [20, 50])
    r = fresh()
    f.draw(surf, r)
    base_ok = f.plot_rect.x == r.x + 54 and f.plot_rect.right == r.right - 14 and f.plot_rect.y == r.y + 10

    def left_need(fig, title):
        """The left margin the drawn y tick labels and `title` need (automargin's rule)."""
        labels = fig._ticks("y", fig.plot_rect)[1]
        return (_px(_TICK_GAP) + max(T.text_w(s, "mono", 10) for s in labels) + _px(_TITLE_GAP)
                + _rotated_title(title).get_width() + _px(_EDGE))
    f2 = Figure(xlabel="evaluation", ylabel="best objective")
    f2.line([1, 28], [94.76, 94.76])
    f2.draw(surf, r)
    pushed = f2.plot_rect.x - r.x
    f3 = Figure(xlabel="evaluation", ylabel="lift [N]")         # labels too wide for 54 px in any face
    f3.line([0, 1], [1234567.5, 1234568.5])
    f3.draw(surf, r)
    grown = f3.plot_rect.x - r.x
    rep("margins l54 r14 t10, bottom 48 with an x title; the left margin grows to fit the y labels + title ('95.5' "
        "+ 'best objective': the axis at 59, shot 12's 353)",
        base_ok and (not sf or r.bottom - f.plot_rect.bottom == 48) and pushed == max(54, left_need(f2, f2.ylabel))
        and (not sf or pushed == 60) and grown == left_need(f3, f3.ylabel) > 54,
        f"plot {f.plot_rect}, bottom margin {r.bottom - f.plot_rect.bottom}, axis at {pushed - 1}, wide labels "
        f"{grown - 1}")

    # 12 cost
    xs = np.arange(1, 301)
    ys = np.maximum.accumulate(np.sin(xs * 0.05) * 20 + xs * 0.1)
    t0 = time.perf_counter()
    for _ in range(20):
        f = Figure(xlabel="evaluation", ylabel="objective")
        f.scatter(xs, ys - 3, colour=T.GOOD, size=6, opacity=0.75, name="per-eval")
        f.line(xs, ys, width=2.5, name="best feasible so far")
        f.off_scale([5, 50, 150])
        f.hline(0, colour=T.BAD, dash="dash")
        f.draw(surf, fresh())
    ms = 1e3 * (time.perf_counter() - t0) / 20
    rep("a 300-evaluation convergence figure (markers, line, legend) draws in < 6 ms", ms < 6.0, f"{ms:.2f} ms")

    if out:
        _specimen(out)
        if verbose:
            print(f"  specimen sheet: {out}")
    if verbose:
        print(f"  {'ALL PASS' if ok else 'FAILURES ABOVE'}: {n_ok}/{n_all} checks")
    return ok


def _specimen(path: str) -> None:
    """Four figures on a sheet, for a look against AeroBO's shots 07 / 12 /
    18 / 20."""
    sheet = pygame.Surface((1300, 700))
    sheet.fill(T.WELL)
    xs = np.arange(1, 54)
    best = np.array([29.5] * 11 + [42.2] * 17 + [48.9] * 16 + [52.65] * 9)
    f = Figure(xlabel="evaluation", ylabel="objective")
    f.scatter(xs, best - np.abs(np.sin(xs)) * 8, colour=T.GOOD, size=6, opacity=0.75, name="per-eval")
    f.line(xs, best, width=2.5, colour=T.ACCENT, name="best feasible so far")
    f.off_scale([7, 29, 45])
    f.draw(sheet, pygame.Rect(0, 0, 1290, 280))
    t = np.linspace(0, 2 * np.pi, 120)
    g = Figure(equal=True, xlabel="x/c", ylabel="y/c")
    g.line(0.5 + 0.5 * np.cos(t), 0.06 * np.sin(t) + 0.02 * np.sin(2 * t), fill=T.BAND_A, name="hg40 · t/c 0.1500")
    g.line(0.5 + 0.5 * np.cos(t), 0.055 * np.sin(t), colour=T.INK_MUTED, width=1.4, dash="dot", name="seed")
    g.vline(0.3, colour=T.WARN, dash="dash", label="max t/c @ 30%")
    g.draw(sheet, pygame.Rect(0, 290, 640, 280))
    h = Figure(xlabel="evaluation", ylabel="best objective", y2label="c_d", legend=True)
    h.line([1, 28], [94.76, 94.76])
    h.line(range(1, 29), np.linspace(0.02, 0.012, 28), colour=T.WARN, axis="y2", markers="diamond", name="c_d")
    h.draw(sheet, pygame.Rect(650, 290, 640, 280))
    pygame.image.save(sheet, path)


if __name__ == "__main__":
    import sys
    _out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else ""
    sys.exit(0 if self_check(out=_out) else 1)
