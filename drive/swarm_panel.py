"""drive/swarm_panel.py -- the swarm's progress panel, and its free values (task 26).

The swarm stays open-ended (no generation count, no "train N more": ESC
stops it, as before). What it lacked was a sense of where it is going. The
panel, drawn in the swarm window's top-left text panel, shows:

* the BEST LAP PER GENERATION, the last `ROWS` generations, each with a bar
  scaled between the slowest and the fastest lap on show, and how many cars
  lapped;
* the class's MEDAL LINES (drive/medals.py: author / gold / silver /
  bronze) and the OWNER'S PB in the class (drive/records.py), each with the
  swarm's best lap's gap to it (- = the swarm is faster).

The class is the map, the car the swarm breeds in, the session's engine and
surface (plan D1). The swarm's lap times are at its TRAINING step (2 ms), the
medals and the PB at 1 ms, so the panel says so; the saved bot's lap is
re-measured at 1 ms when it is saved (K / ESC), as before.

Free values (the Deploy-swarm page): the population is any integer in
[POP_MIN, POP_MAX] and the sim time any whole second in [T_MIN, T_MAX]
(`clamp_pop`, `clamp_T`). Reaching one is quick (task 34, the owner's "one
by one to 128 takes a lot of time"): LEFT / RIGHT jump between FIXED
NUMBERS (`POP_PRESETS`, `T_PRESETS`; `step_value`), ENTER cycles them
(`cycle_value`), and on a keyboard the digits TYPE any value (`Typed`: up
to three digits, BACKSPACE takes one back, clamped into the range as you
type; ENTER sets it, and LEFT / RIGHT, another row or leaving the page end
the number -- the next digit starts a new one).

Pure functions of numbers and dicts: no pygame, no drive.ml, no file I/O
(the PB is handed in by the caller, which is the one that may read a player
file -- a windowed player session only).
"""
from __future__ import annotations

import math

POP_MIN, POP_MAX = 4, 128
T_MIN, T_MAX = 20.0, 240.0
#: LEFT / RIGHT and ENTER go through these; the digits reach everything between
POP_PRESETS = (4, 8, 16, 24, 32, 48, 64, 96, 128)
T_PRESETS = (20.0, 30.0, 45.0, 60.0, 70.0, 90.0, 120.0, 150.0, 180.0, 240.0)
TYPE_DIGITS = 3              # a typed number has at most this many digits
ROWS = 8                     # generations the panel lists
BAR = 18                     # characters of the widest bar
MEDALS = ("author", "gold", "silver", "bronze")


def clamp_pop(v) -> int:
    try:
        return int(min(max(int(round(float(v))), POP_MIN), POP_MAX))
    except (TypeError, ValueError):
        return 24


def clamp_T(v) -> float:
    try:
        return float(min(max(round(float(v)), T_MIN), T_MAX))
    except (TypeError, ValueError):
        return 70.0


def _pre(key):
    return (POP_PRESETS, clamp_pop) if key == "pop" else (T_PRESETS, clamp_T)


def step_value(key: str, cur, d: int):
    """LEFT / RIGHT on the page's Cars (`pop`) or Sim time (`T`) row: the
    next fixed number that way -- from a typed value between two, its
    neighbour on that side; no wrap (RIGHT on 128 stays 128)."""
    pre, clamp = _pre(key)
    c = clamp(cur)
    if d > 0:
        return clamp(next((p for p in pre if p > c), pre[-1]))
    return clamp(next((p for p in reversed(pre) if p < c), pre[0]))


def cycle_value(key: str, cur):
    """ENTER / CROSS on the row: the next fixed number, round to the first."""
    pre, clamp = _pre(key)
    c = clamp(cur)
    return clamp(next((p for p in pre if p > c), pre[0]))


class Typed:
    """Digits typed on a row (the keyboard's 0-9 on Cars / Sim time): the
    number so far, shown on the row until the player acts -- ENTER sets it,
    LEFT / RIGHT, another row or leaving the page end it (`clear`, the
    page's side). Another row's digit, or a digit after TYPE_DIGITS, starts a
    new number; no clock: what the row shows is what the next key extends."""

    def __init__(self):
        self.key, self.buf = None, ""

    def digit(self, key: str, ch: str):
        """Add one digit; returns the value typed (not yet clamped)."""
        if key != self.key or len(self.buf) >= TYPE_DIGITS:
            self.buf = ""
        self.key = key
        self.buf += str(ch)[:1] if str(ch)[:1].isdigit() else ""
        return int(self.buf) if self.buf else None

    def backspace(self, key: str, value):
        """BACKSPACE on the row: one digit back -- of the number being typed,
        or of the row's value when none is. Returns the value (None: empty)."""
        if key != self.key or not self.buf:
            self.key, self.buf = key, f"{float(value):.0f}"
        self.buf = self.buf[:-1]
        return int(self.buf) if self.buf else None

    def shown(self, key: str) -> str:
        """The digits to show on `key`'s row, or ''."""
        return self.buf if key == self.key else ""

    def clear(self) -> None:
        self.key, self.buf = None, ""


def row_value(key: str, value, typed: str) -> str:
    """The row's number: the typed digits while typing (and the clamped
    value when they are out of range), else the value."""
    txt = f"{value:.0f}" if key == "T" else f"{value}"
    if not typed:
        return txt
    return f"{typed}_" if int(typed) == float(value) else f"{typed}_ = {txt}"


def _t(v) -> str:
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return "--"
    m, s = divmod(float(v), 60.0)
    return f"{int(m)}:{s:06.3f}" if m >= 1 else f"{s:.3f}"


def _gap(best, ref) -> str:
    if not (isinstance(best, (int, float)) and isinstance(ref, (int, float))
            and math.isfinite(best) and math.isfinite(ref)):
        return ""
    d = best - ref
    return f"{d:+.2f} s" + ("  (beaten)" if d <= 0 else "")


def lap_rows(history, pop: int, rows: int = ROWS) -> list:
    """The best lap of each of the last `rows` generations, with a bar: the
    longest bar is the fastest lap on show."""
    hist = list(history or [])[-rows:]
    laps = [h.get("lap_best") for h in hist if isinstance(h.get("lap_best"), (int, float))]
    out = []
    if not hist:
        return ["  (the first generation is being scored)"]
    lo, hi = (min(laps), max(laps)) if laps else (0.0, 0.0)
    for h in hist:
        lb = h.get("lap_best")
        if isinstance(lb, (int, float)) and math.isfinite(lb):
            frac = 1.0 if hi <= lo else 1.0 - 0.8 * (lb - lo) / (hi - lo)
            bar = "#" * max(1, int(round(BAR * frac)))
            out.append(f"  gen {int(h.get('gen', 0)):4d}  {_t(lb):>9s}  {bar:<{BAR}s}  "
                       f"{int(h.get('n_lapped', 0)):3d}/{pop} lapped")
        else:
            out.append(f"  gen {int(h.get('gen', 0)):4d}  {'no lap':>9s}  {'':<{BAR}s}  "
                       f"{int(h.get('n_lapped', 0)):3d}/{pop} lapped")
    return out


def best_lap(history, best=None):
    """The swarm's best lap so far (training step), or None."""
    laps = [h.get("lap_best") for h in (history or [])
            if isinstance(h.get("lap_best"), (int, float)) and math.isfinite(h["lap_best"])]
    if best and isinstance(best.get("lap_best"), (int, float)):
        laps.append(best["lap_best"])
    return min(laps) if laps else None


def _flying(history, best=None) -> bool:
    """Are the swarm's laps FLYING laps? A generation whose best lap is the
    first one from the rollout's rolling start (its T too short for two)
    is not comparable with the medals and the PB, which are flying laps."""
    rows = [h for h in (history or []) if h.get("lap_best")]
    ok = all(h.get("lap_flying", True) for h in rows)
    if best and best.get("lap_best"):
        ok = ok and bool(best.get("lap_flying", True))
    return ok


def panel_lines(history, pop: int, key: str, targets=None, pb=None, best=None,
                dt_train: float = 0.002, why_none: str = "") -> list:
    """The panel's lines (the swarm window's text panel; '!' = highlighted).
    `why_none`: why there are no medals / PB to compare with (a skidpad of
    another radius ...)."""
    lines = [f"PROGRESS  {key}   laps at the {dt_train * 1e3:.0f} ms training step; "
             f"medals and your PB at 1 ms"]
    lines += lap_rows(history, pop)
    b = best_lap(history, best)
    fly = _flying(history, best)
    lines.append(f"  the swarm's best lap: {_t(b)}" + (
        "" if fly or b is None else
        "   (its FIRST lap, from 12 m/s at the line: raise Sim time for a flying lap)"))
    if not fly:
        b = None                           # not like for like: no gaps, no 'beaten'
    if why_none:
        lines.append(f"  no medals or PB to compare: {why_none}")
        return lines
    if targets:
        lines.append("  " + "   ".join(f"{m.upper()} {_t(targets.get(m))} {_gap(b, targets.get(m))}"
                                       .rstrip() for m in MEDALS[:2]))
        lines.append("  " + "   ".join(f"{m.upper()} {_t(targets.get(m))} {_gap(b, targets.get(m))}"
                                       .rstrip() for m in MEDALS[2:]))
    else:
        lines.append("  no medals for this class (no reference lap)")
    if isinstance(pb, (int, float)) and math.isfinite(pb):
        g = _gap(b, pb)
        lines.append(("!" if g.endswith("(beaten)") else "")
                     + f"  YOUR PB {_t(pb)}" + (f"   the swarm {g}" if g else ""))
    else:
        lines.append("  YOUR PB: none in this class yet")
    return lines


# ==================================================================== #
#  SELF-CHECK                                                          #
# ==================================================================== #
def self_check(verbose: bool = True) -> bool:
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}" + (f": {msg}" if msg else ""))

    rep("population: any integer in [4, 128], clamped; LEFT / RIGHT the fixed numbers, no wrap",
        (clamp_pop(3), clamp_pop(4), clamp_pop(57), clamp_pop(128), clamp_pop(500),
         step_value("pop", 128, +1), step_value("pop", 4, -1), step_value("pop", 24, +1),
         step_value("pop", 57, +1), step_value("pop", 57, -1), clamp_pop("x"))
        == (4, 4, 57, 128, 128, 128, 4, 32, 64, 48, 24))
    rep("sim time: whole seconds in [20, 240]; LEFT / RIGHT the fixed numbers",
        (clamp_T(10), clamp_T(97.4), clamp_T(999), step_value("T", 240, +1),
         step_value("T", 20, -1), step_value("T", 70, +1), step_value("T", 97, -1))
        == (20.0, 97.0, 240.0, 240.0, 20.0, 90.0, 90.0))
    rep("ENTER cycles the fixed numbers, round to the first",
        [cycle_value("pop", v) for v in (4, 24, 100, 128)] == [8, 32, 128, 4]
        and cycle_value("T", 240) == 20.0 and cycle_value("T", 70) == 90.0)
    ty = Typed()
    seq = [ty.digit("pop", c) for c in "128"]
    four = ty.digit("pop", "6")                          # a 4th digit: a new number
    ty.digit("pop", "2")
    row = ty.digit("T", "9")                             # another row: a new number
    shown = (ty.shown("T"), ty.shown("pop"))
    ty.clear()
    after = ty.digit("pop", "3")                         # after an ENTER / a move: new
    bs = [ty.backspace("pop", 3), ty.backspace("pop", 3)]
    bv = [Typed().backspace("pop", 128), Typed().backspace("T", 90.0)]   # of the value
    rep("typing: up to three digits; another row, or a clear, starts again; BACKSPACE",
        seq == [1, 12, 128] and four == 6 and row == 9 and shown == ("9", "")
        and after == 3 and bs == [None, None] and bv == [12, 9], f"{seq} {four} {bs} {bv}")
    rep("the row shows the digits, and the clamped value when out of range",
        (row_value("pop", 12, "12"), row_value("pop", 4, "1"), row_value("T", 240.0, "300"),
         row_value("T", 90.0, "")) == ("12_", "1_ = 4", "300_ = 240", "90"))
    hist = [dict(gen=g, lap_best=(None if g < 2 else 70.0 - g), n_lapped=(0 if g < 2 else g))
            for g in range(12)]
    rows = lap_rows(hist, 24)
    fastest = [r for r in rows if "59.000" in r]
    slowest = [r for r in rows if "1:06.000" in r]
    rep("the last 8 generations, the fastest lap with the longest bar",
        len(rows) == ROWS and fastest and slowest
        and fastest[0].count("#") == BAR and slowest[0].count("#") < BAR, "\n" + "\n".join(rows))
    rep("a generation with no lap says so", "no lap" in lap_rows(hist[:3], 24)[0])
    tg = dict(author=58.237, gold=59.402, silver=61.731, bronze=65.226)
    lines = panel_lines(hist, 24, "arena|corsa|sport|patch", tg, pb=60.5)
    txt = "\n".join(lines)
    rep("medal lines with the gap of the swarm's best (59.000) to each",
        "AUTHOR 58.237 +0.76 s" in txt and "GOLD 59.402 -0.40 s  (beaten)" in txt
        and "BRONZE 1:05.226 -6.23 s" in txt, txt)
    rep("your PB and the gap to it, highlighted when the swarm is faster",
        any(ln.startswith("!") and "YOUR PB 1:00.500" in ln and "-1.50 s" in ln for ln in lines))
    none = panel_lines([], 24, "dragstrip|corsa|sport|none", None, None)
    rep("no medals, no PB, no generation yet: said, not invented",
        "no medals for this class" in "\n".join(none) and "none in this class yet" in none[-1]
        and "being scored" in none[1])
    rep("the best individual's lap counts even past the listed generations",
        best_lap(hist[-2:], dict(lap_best=57.5)) == 57.5)
    roll = [dict(h, lap_flying=False) for h in hist]
    rl = "\n".join(panel_lines(roll, 24, "arena|corsa|sport|patch", tg, pb=60.5))
    rep("a rolling-start first lap is said so and not compared (no gap, no 'beaten')",
        "FIRST lap" in rl and "(beaten)" not in rl and "+0.76 s" not in rl, rl)
    odd = panel_lines(hist, 24, "skidpad|corsa|sport|none", tg, pb=17.0,
                      why_none="a skidpad of radius 30 m")
    rep("an off-standard map compares with nothing, and says why",
        "radius 30 m" in odd[-1] and not any("AUTHOR" in ln for ln in odd))
    if verbose:
        print(f"swarm_panel self-check: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
