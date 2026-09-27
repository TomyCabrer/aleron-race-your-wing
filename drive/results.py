"""drive/results.py -- the lap's results card (task 27).

When a timed lap closes, the HUD shows a RESULTS card for `SHOW_S` seconds,
over the top centre (drawn by `render._draw_results`; it does not stop the
car -- a time trial is lap after lap): the lap time, the delta to the PB it
was driven against, each sector coloured (purple = the class's best ever,
green = better than that PB lap's, red = slower, plain = nothing to compare
with), the medal it earned, its
place in the class's top 5 -- and, for a NEW PB, an animation: the card drops
in, "NEW PB" pulses gold for `PULSE_S`. A lap that does not count gets a
short card that says why.

`card(res, book)` builds it from the recorder's lap result (`records.
LapRecorder` -> `Sim._rec_lap`) and the book the lap was just filed in;
`view(card, age_s)` is what `HudData.results` carries each frame (the card
plus its age, or None once it is over). Pure: no pygame.

The card is also kept (task 32): the session's last `LOG_N` cards, and
Settings > Last lap (`summary`) opens the LAP RESULTS page -- the last card
itself, settled (`render.Renderer.draw_card`), and this session's laps
(`page_rows`).

An UNLIMITED lap (task 41: a build with a wing past its car's physical span
limit, filed in the Unlimited book -- drive/records.py) says so on the card
(`unlimited`: the header reads UNLIMITED LAP / UNLIMITED PB) and in its
summary; its delta, place and medal are the Unlimited book's.
"""
from __future__ import annotations

import math

SHOW_S = 6.0           # s the card stays up
LOG_N = 10             # cards a session keeps for the LAP RESULTS page
PULSE_S = 2.5          # s the NEW PB header pulses
DROP_S = 0.25          # s the card takes to drop in


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def next_text(nx) -> str:
    """The card's words for `prerace.next_medal`'s (name, target, gap): the
    medal and the time still to find, 'AUTHOR in 0.421 s' (the TIME TRIAL
    page's '(0.421 s to go)'; no minus sign, which read as already under
    it). '' when there is none or no gap yet."""
    if not nx or len(nx) < 3 or not _num(nx[2]):
        return ""
    return f"{str(nx[0]).upper()} in {float(nx[2]):.3f} s"


def card(res: dict, book=None) -> dict:
    """The card for one closed lap. `book` is the RecordBook it was filed in
    (None: no sector colours from the class's history)."""
    from .records import fmt_time
    t = res.get("time")
    c = dict(time=fmt_time(t), valid=bool(res.get("valid")), why=str(res.get("why") or ""),
             delta="", delta_sign=0, sectors=[], medal=str(res.get("medal") or ""),
             medal_best=bool(res.get("medal_best")), pos="", new_pb=False,
             key=str(res.get("key") or ""), next="", unlimited=bool(res.get("unlimited")))
    if not c["valid"]:
        return c
    pb0 = res.get("pb_before")
    if _num(pb0) and _num(t):
        d = float(t) - float(pb0)
        c["delta"] = f"{d:+.3f}"
        c["delta_sign"] = -1 if d < 0 else (1 if d > 0 else 0)
    else:
        c["delta"] = ("first Unlimited lap in this class" if c["unlimited"]
                      else "first lap in this class")
    if _num(t) and c["key"]:               # the medal the PB (this lap counted) goes for next
        try:
            from .prerace import next_medal
            nx = next_medal(c["key"], min(float(t), float(pb0)) if _num(pb0) else float(t))
            c["next"] = next_text(nx)
        except Exception:                  # noqa: BLE001 -- a card never stops the car
            c["next"] = ""
    pos = res.get("pos")
    if isinstance(pos, int) and pos > 0:
        c["pos"] = f"P{pos}"
        c["new_pb"] = pos == 1 and (not _num(pb0) or float(t) < float(pb0))
    secs = res.get("sectors") or []
    best = []
    prev = []
    if book is not None and res.get("key"):
        try:
            best = list(book.best_sectors(res["key"]))
            laps = book.laps(res["key"])
            #  the PB this lap was driven against: the one it pushed down, or
            #  the one it did not beat
            ref = laps[1] if (c["new_pb"] and len(laps) > 1) else (laps[0] if laps and not c["new_pb"]
                                                                    else None)
            prev = list((ref or {}).get("sectors") or [])
        except Exception:                  # noqa: BLE001 -- a card never stops the car
            best, prev = [], []
    for i, s in enumerate(secs):
        if not _num(s):
            continue
        col = ""
        if i < len(best) and _num(best[i]) and s <= best[i] + 1e-9:
            col = "purple"
        elif i < len(prev) and _num(prev[i]) and s < prev[i]:
            col = "green"
        elif i < len(prev) and _num(prev[i]):
            col = "red"
        c["sectors"].append((f"S{i + 1} {s:6.3f}", col))
    return c


ROW_CHARS = 60         # a list row's text at most (a long reason is cut, not run off the page)


def _cut(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 3].rstrip() + "..."


def summary(c, short: bool = False) -> str:
    """One line for the page's list -- the time, the delta, the place, the
    medal (NEW PB); or why it did not count. `short`: the Settings row (the
    time, the delta, the medal), which must fit beside the other rows."""
    if not c:
        return "no timed lap yet"
    unl = "UNLIMITED" if c.get("unlimited") else ""
    if not c.get("valid"):
        return "  ".join(p for p in (f"{c.get('time', '')}", unl) if p) + "  not counted" + (
            "" if short else (f": {c['why']}" if c.get("why") else ""))
    d = str(c.get("delta", "") or "")
    d = d if d.startswith(("+", "-")) else "first in class"
    medal = str(c.get("medal", "") or "").upper()
    if short:
        return "  ".join(p for p in (str(c.get("time", "")), d, medal, unl) if p)
    parts = [str(c.get("time", "")), d, str(c.get("pos", "") or ""), medal,
             "NEW PB" if c.get("new_pb") else "", unl]
    return "  ".join(p for p in parts if p)


def page_rows(log, key: str = "") -> list:
    """The LAP RESULTS page's list: the session's kept cards, newest first,
    each with its lap number in the session (`n`, set when it was kept; the
    list's own order for a card without one). A card filed in another class
    than `key` (a live engine change mid-session) says whose engine."""
    rows = []
    for k, c in list(enumerate(list(log or []), 1))[-LOG_N:]:
        n = c.get("n", k) if isinstance(c, dict) else k
        txt = summary(c)
        ck = str((c or {}).get("key", "") or "")
        if key and ck and ck != key:
            parts = ck.split("|")
            txt += f"  ({parts[2]} engine)" if len(parts) == 4 else "  (another class)"
        rows.append((f"lap {n}", _cut(txt, ROW_CHARS)))
    return rows[::-1] or [("--", "no timed lap yet in this session")]


def view(c, age_s: float):
    """What the HUD draws this frame: the card with its age, or None."""
    if not c or not _num(age_s) or age_s < 0.0 or age_s > SHOW_S:
        return None
    return dict(c, age=float(age_s))


def drop(age_s: float) -> float:
    """0 -> 1 as the card drops in (an ease-out)."""
    x = min(max(age_s / DROP_S, 0.0), 1.0)
    return 1.0 - (1.0 - x) ** 3


def pulse(age_s: float) -> float:
    """0..1, the NEW PB header's pulse; 0 after PULSE_S."""
    if age_s > PULSE_S:
        return 0.0
    return 0.5 + 0.5 * math.sin(2.0 * math.pi * 3.0 * age_s)


# ==================================================================== #
#  SELF-CHECK                                                          #
# ==================================================================== #
def self_check(verbose: bool = True) -> bool:
    import tempfile
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}" + (f": {msg}" if msg else ""))

    from . import records as rec
    root = tempfile.mkdtemp(prefix="carsim_results_")
    book = rec.RecordBook(root)
    key = rec.class_key("arena", "corsa", "sport", "patch")
    r0 = rec._fake_rec(61.40, [20.2, 20.5, 20.7])
    book.insert(key, r0, save=False)
    # a new PB, sector 1 the best ever, sector 2 better than the old PB's, 3 slower
    res = dict(time=60.90, valid=True, pb_before=61.40, key=key, why="",
               sectors=[20.0, 20.3, 20.6], medal="gold", medal_best=True)
    res["pos"] = book.insert(key, rec._fake_rec(60.90, [20.0, 20.3, 20.6]), save=False)
    c = card(res, book)
    rep("a new PB: the time, the delta to the PB it beat, P1, the medal",
        c["time"] == "1:00.900" and c["delta"] == "-0.500" and c["delta_sign"] == -1
        and c["pos"] == "P1" and c["new_pb"] and c["medal"] == "gold", str(c))
    rep("its sectors: best ever purple (all three are, on a two-lap book)",
        [col for _, col in c["sectors"]] == ["purple", "purple", "purple"], str(c["sectors"]))
    # a slow lap with the class's best S2 ever (20.10), then a lap: P2, plus
    # delta, S1 slower than the PB lap's, S2 better than it but not the best
    # ever, S3 the best ever
    book.insert(key, rec._fake_rec(62.50, [20.9, 20.1, 21.5]), save=False)
    res2 = dict(time=61.10, valid=True, pb_before=60.90, key=key,
                sectors=[20.1, 20.25, 20.55], medal="silver")
    res2["pos"] = book.insert(key, rec._fake_rec(61.10, [20.1, 20.25, 20.55]), save=False)
    c2 = card(res2, book)
    rep("a slower lap: +delta, P2, no animation; sectors red / green / purple",
        c2["delta"] == "+0.200" and c2["pos"] == "P2" and not c2["new_pb"]
        and [col for _, col in c2["sectors"]] == ["red", "green", "purple"], str(c2["sectors"]))
    c3 = card(dict(time=40.0, valid=False, why="not a full lap", key=key))
    rep("a lap that does not count: a short card with the reason",
        not c3["valid"] and c3["why"] == "not a full lap" and not c3["sectors"])
    c4 = card(dict(time=62.0, valid=True, pb_before=float("inf"), key=key, pos=1,
                   sectors=[]), None)
    rep("the class's first lap: said so, a new PB", c4["delta"] == "first lap in this class"
        and c4["new_pb"])
    from .prerace import next_medal
    nx = next_medal(key, 60.90)            # the PB after c, and after c2 (slower)
    rep("the next medal from the PB this lap leaves: a new PB's own, a slower lap's the old one's",
        c["next"] == next_text(nx) and c2["next"] == c["next"] and c3["next"] == "",
        f"{c['next']!r}")
    #  the words: the medal and the time still to find, never a minus sign
    #  (Ashdown's 1:05.408 against its author 1:04.987)
    tg_ = dict(author=64.987, gold=66.0, silver=68.0, bronze=70.0)
    nt = next_text(next_medal(key, 65.408, tg_))
    rep("the next medal reads 'AUTHOR in 0.421 s'; nothing past the author or with no gap",
        nt == "AUTHOR in 0.421 s" and next_text(next_medal(key, 64.9, tg_)) == ""
        and next_text(next_medal(key, float("inf"), tg_)) == ""
        and next_text(None) == "", repr(nt))
    rep("the settings row and the page's list: time, delta, place, medal; or why",
        summary(c) == "1:00.900  -0.500  P1  GOLD  NEW PB"
        and summary(c3) == "40.000  not counted: not a full lap"
        and summary(c4).startswith("1:02.000  first in class  P1") and summary(None)
        == "no timed lap yet" and c["key"] == key, summary(c4))
    many = [c2] * (LOG_N + 3) + [c]
    rows = page_rows(many)
    rep("the page lists the last LOG_N laps, newest first",
        len(rows) == LOG_N and rows[0] == (f"lap {LOG_N + 4}", summary(c))
        and page_rows([])[0][0] == "--", str(rows[:2]))
    kept = [dict(c2, n=21 + i) for i in range(LOG_N)]     # the Sim keeps LOG_N, numbered
    other = dict(c2, key=key.replace("sport", "stock"), n=31)
    rows2 = page_rows(kept[1:] + [other], key)
    long_why = dict(c3, why="snapshot failed (AttributeError: " + "x" * 200 + ")")
    rep("a kept card keeps its lap number; another class says so; a long reason is cut",
        rows2[0] == ("lap 31", summary(other) + "  (stock engine)")
        and rows2[-1][0] == "lap 22" and len(page_rows([long_why])[0][1]) <= ROW_CHARS
        and summary(c, short=True) == "1:00.900  -0.500  GOLD", str(rows2[:2]))
    #  task 41: an Unlimited lap's card and summary say so (its place, its
    #  delta and its medal are the Unlimited book's)
    cu = card(dict(res2, unlimited=True), book)
    cu0 = card(dict(time=55.0, valid=True, pb_before=float("nan"), key=key, pos=1,
                    unlimited=True), None)
    cu3 = card(dict(c3, valid=False, why="off track", time=40.0, unlimited=True))
    rep("an Unlimited lap: the card is tagged, the summaries say UNLIMITED",
        cu["unlimited"] and not c2["unlimited"] and cu["pos"] == c2["pos"]
        and summary(cu) == summary(c2) + "  UNLIMITED"
        and summary(cu, short=True).endswith("UNLIMITED")
        and cu0["delta"] == "first Unlimited lap in this class"
        and summary(cu3) == "40.000  UNLIMITED  not counted: off track",
        f"{summary(cu)!r}; {summary(cu3)!r}")
    rep("the card lasts SHOW_S, drops in, pulses only for a new PB's first seconds",
        view(c, 1.0)["age"] == 1.0 and view(c, SHOW_S + 0.1) is None and view(None, 1.0) is None
        and drop(0.0) == 0.0 and drop(DROP_S) == 1.0 and pulse(PULSE_S + 0.1) == 0.0
        and 0.0 <= pulse(0.3) <= 1.0)
    if verbose:
        print(f"results self-check: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
