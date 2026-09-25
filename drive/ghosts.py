"""drive/ghosts.py -- two ghosts and the live delta (task 22).

Ghost 1 is your PB for the class: the 50 Hz pose trace task 19 files with
every record. Ghost 2 is a slot (plan decision D2): none, the REFERENCE BOT
(the lap the class's author medal time was set with, `drive.medals`), or any
of your top-5 laps -- the default is the reference bot. Both are
`drive.drive._Replay`s drawn through the existing ghost path
(`HudData.ghosts`: flat ground silhouettes, in the plan views and in the 3-D
chase view alike), labelled `PB` and `REF` / `P3`.

They are CLOCKED FROM THE LINE: a ghost appears when you cross the start
line and restarts at every crossing, so it is always the lap you are on
against the lap it recorded. Before the first crossing (the out-lap) there
is no ghost, and a ghost that finishes first waits on the line.

The LIVE DELTA is read the way the race's gap is (`drive.drive.RACE_GAP_*`):
by track position, not by time. The PB trace carries its centreline distance
from the line (`s`, task 19); at your distance `p` the PB had been driving
for `t_pb(p)`, and you have been driving for `tau`, so the delta is `tau -
t_pb(p)` -- negative (green) is ahead. It is drawn large at the top centre.

At each SECTOR line the sector time flashes: PURPLE when it is the best that
sector has ever been driven in the class (`RecordBook.best_sectors`, which
every valid lap updates, not only the top 5), GREEN when it beats the PB
lap's own sector, RED when it does not.

In an UNLIMITED session (task 41: the build has a wing past its car's span
limit) the book is the class's Unlimited book (`records.unlimited_book`), so
the PB ghost, the delta and the flashes race the Unlimited laps -- never the
official ones -- and the ghosts say so: `UNL PB`, `UNL P3`.

`J` toggles both ghosts (the delta stays). Nothing here touches physics: the
`Sim` calls `event` for each `LapTimer` event and reads `ghost_tuples`,
`delta` and `flash` when it builds the HUD.

    python3 -m drive.ghosts      the self-check
"""

from __future__ import annotations

import math

import numpy as np

from . import records as rec

#: the ghost-2 slot: 'ref' | 'none' | 'top2' .. 'top5' (your P1 IS the PB,
#: which is ghost 1 already)
SLOTS = ("ref", "none") + tuple(f"top{i}" for i in range(2, rec.TOP_N + 1))
SLOT_DEFAULT = "ref"
C_GHOST_PB = (120, 220, 160)      # your PB: green-grey
C_GHOST_2 = (205, 205, 215)       # the second ghost: light grey
FLASH_S = 2.5                     # s a sector flash stays up
#: the delta is only read inside the PB's own lap: a car more than this far
#: past the PB's last sample (it went round the back of the line) has none
DELTA_MAX_OVERRUN_M = 30.0


def slot_label(slot: str, book=None, key: str | None = None) -> str:
    if slot == "none":
        return "none"
    if slot == "ref":
        return "reference bot"
    if slot.startswith("top"):
        i = int(slot[3:])
        t = float("nan")
        if book is not None and key is not None:
            laps = book.laps(key)
            if len(laps) >= i:
                t = float(laps[i - 1]["time"])
        unl = "Unlimited " if getattr(book, "unlimited", False) else ""
        return f"your {unl}P{i}" + (f" ({rec.fmt_time(t)})" if math.isfinite(t) else " (none yet)")
    return str(slot)


class Curve:
    """Track distance -> time since the line, from a trace's `s` and `t`
    columns (a lap that backs up a metre is made monotone, not refused)."""

    __slots__ = ("s", "t", "s_end", "t_end")

    def __init__(self, arr: np.ndarray):
        s = np.maximum.accumulate(np.asarray(arr[:, 8], dtype=np.float64))
        t = np.asarray(arr[:, 0], dtype=np.float64)
        self.s, self.t = s, t
        self.s_end = float(s[-1]) if len(s) else 0.0
        self.t_end = float(t[-1]) if len(t) else 0.0

    def t_at(self, p: float) -> float:
        if not len(self.s) or p > self.s_end + DELTA_MAX_OVERRUN_M:
            return float("nan")
        return float(np.interp(p, self.s, self.t))


def _has_trace(lp) -> bool:
    return isinstance(lp, dict) and ("trace" in lp or lp.get("_trace_arr") is not None)


def _trace_of(lp) -> np.ndarray:
    """A record's trace: the filed one, or -- for a lap the recorder has only
    just filed -- the raw one it keeps in memory until the file has it."""
    if "trace" in lp:
        return rec.decode_trace(lp["trace"])
    return np.asarray(lp["_trace_arr"], dtype=np.float64)


def _replay(arr, colour, label):
    from .drive import _Replay            # the swarm's replay: one ghost path
    return _Replay(arr[:, :8], colour, True, label)


def reference_trace(key: str):
    """The reference bot's lap for the class, from drive.medals (task 21);
    None without one."""
    try:
        from . import medals
        return medals.reference_trace(key)
    except Exception:                      # noqa: BLE001 -- medals are optional
        return None


class GhostSet:
    """The two ghosts, the delta curve and the sector flash for ONE class."""

    def __init__(self, book, key: str, slot: str = SLOT_DEFAULT, enabled: bool = True,
                 ref_fn=reference_trace):
        self.book = book
        self.key = key
        self.slot = slot if slot in SLOTS else SLOT_DEFAULT
        self.enabled = bool(enabled)
        self.ref_fn = ref_fn
        self.pb = None                     # _Replay of the PB
        self.curve = None                  # Curve of the PB
        self.pb_secs: list = []
        self.g2 = None                     # _Replay of the slot
        self._pb_sig = None
        self._g2_sig = None
        self._ver = None                   # the book's version last loaded
        self._prog = None                  # centreline metres since the line
        self._s_last = None
        self.flash = None                  # (text, colour name, until sim t)
        self.refresh()

    # -- what is loaded ------------------------------------------------------
    def refresh(self) -> None:
        """(Re)load the PB and the slot when either has changed: a new PB
        lands at a line crossing, and the next lap races it."""
        self._ver = getattr(self.book, "version", None)
        unl = "UNL " if getattr(self.book, "unlimited", False) else ""   # task 41
        pb = self.book.pb(self.key)
        #  a lap the recorder has just filed is LIGHT until its filing thread
        #  adds the trace: its signature changes again when the trace lands
        sig = (pb.get("time"), pb.get("date"), _has_trace(pb)) if pb else None
        if sig != self._pb_sig:
            self._pb_sig = sig
            self.pb = self.curve = None
            self.pb_secs = []
            if pb and _has_trace(pb):
                try:
                    arr = _trace_of(pb)
                    if len(arr) >= 2:
                        self.pb = _replay(arr, C_GHOST_PB, unl + "PB")
                        self.curve = Curve(arr)
                    self.pb_secs = list(pb.get("sectors") or [])
                except Exception as exc:   # noqa: BLE001 -- a bad trace: no ghost
                    print(f"ghosts: the PB trace would not load ({type(exc).__name__}: {exc})")
        g2sig = (self.slot, sig, tuple((lp.get("time"), _has_trace(lp))
                                       for lp in self.book.laps(self.key)))
        if g2sig != self._g2_sig:
            self._g2_sig = g2sig
            self.g2 = None
            arr, label = None, ""
            if self.slot == "ref":
                arr, label = self.ref_fn(self.key) if self.ref_fn else None, "REF"
            elif self.slot.startswith("top"):
                i = int(self.slot[3:])
                laps = self.book.laps(self.key)
                if len(laps) >= i and _has_trace(laps[i - 1]):
                    try:
                        arr, label = _trace_of(laps[i - 1]), f"{unl}P{i}"
                    except Exception:      # noqa: BLE001
                        arr = None
            if arr is not None and len(arr) >= 2:
                self.g2 = _replay(np.asarray(arr), C_GHOST_2, label)

    def sync(self, key: str | None = None) -> None:
        """Follow the class (a live engine change moves it) and the book (a
        lap filed, a trace landed): reload only when something changed."""
        if key is not None and key != self.key:
            self.key = key
            self._pb_sig = self._g2_sig = None
            self.refresh()
        elif getattr(self.book, "version", None) != self._ver:
            self.refresh()

    def set_slot(self, slot: str) -> None:
        self.slot = slot if slot in SLOTS else SLOT_DEFAULT
        self._g2_sig = None
        self.refresh()

    def step_slot(self, d: int) -> str:
        i = SLOTS.index(self.slot) if self.slot in SLOTS else 0
        self.set_slot(SLOTS[(i + (1 if d >= 0 else -1)) % len(SLOTS)])
        return self.slot

    # -- the lap clock ---------------------------------------------------------
    @staticmethod
    def tau(sim):
        t0 = getattr(sim.lap, "t_lap_start", None)
        return None if t0 is None else float(sim.t) - float(t0)

    def event(self, sim, e) -> None:
        """A `LapTimer` event: a crossing reloads (a new PB races from the
        next lap) and restarts the progress count; a sector line flashes its
        colour."""
        if e[0] in ("start", "lap"):
            self.refresh()
            L, s = float(sim.track.length), float(sim.s)
            self._prog = s - L if (sim.track.closed and s > 0.5 * L) else s
            self._s_last = s
        elif e[0] == "sector":
            i, t = int(e[1]), float(e[3])
            best = self.book.best_sectors(self.key)
            b = best[i] if i < len(best) else None
            p = self.pb_secs[i] if i < len(self.pb_secs) else None
            if not self._counts(sim, i):
                col = ""                   # a lap that will not be filed: no colour
            elif b is None or t <= b:      # can claim "best ever" or "beats the PB"
                col = "purple"
            elif p is not None and t < p:
                col = "green"
            else:
                col = "red"
            ref = p if p is not None else b
            d = f"  {t - ref:+.3f}" if ref is not None else ""
            self.flash = (f"S{i + 1}  {rec.fmt_time(t)}{d}", col, float(sim.t) + FLASH_S)

    @staticmethod
    def _counts(sim, i: int) -> bool:
        """Can the lap this sector belongs to be a record? Not on the out-lap,
        not after all four wheels were off, not when the recorder has dropped
        it (a reset, the wet toggle, slow motion). The LAST sector's event
        comes after the lap event, which has already re-armed the timer's
        validity for the next lap: that one reads `lap_valid`."""
        lt = sim.lap
        if getattr(lt, "t_lap_start", None) is None:
            return False
        last = i == len(getattr(lt, "lines", [0.0])) - 1
        if not (lt.lap_valid if last else getattr(lt, "_valid_run", True)):
            return False
        r = getattr(sim, "recorder", None)
        return r is None or r.recording

    def _progress(self, sim):
        """Centreline metres since the line, counted on the RIBBON only (the
        race gap's trail approach): off it -- the open map's pad, where the
        projection flips between the two sides of the loop -- there is none."""
        if self._prog is None:
            return None
        L, s = float(sim.track.length), float(sim.s)
        on = abs(float(getattr(sim, "n_lat", 0.0))) <= 0.5 * float(sim.track.width) + 2.0
        ds = s - self._s_last
        if sim.track.closed:
            ds += L if ds < -0.5 * L else (-L if ds > 0.5 * L else 0.0)
        if on and abs(ds) <= 30.0:
            self._prog += ds
        self._s_last = s
        return self._prog if on else None

    # -- what the HUD shows ---------------------------------------------------
    def ghost_tuples(self, sim) -> list:
        if not self.enabled:
            return []
        tau = self.tau(sim)
        if tau is None:
            return []
        out = []
        for g in (self.g2, self.pb):       # the PB drawn last: on top
            if g is None:
                continue
            x, y, psi, _u, _wd, _ws, _alive, _td = g.pose(tau)
            out.append((x, y, psi, g.colour, g.label))
        return out

    def delta(self, sim) -> float:
        """Your lap time minus the PB's at the same track distance: nan before
        the first crossing, without a PB, off the ribbon, behind the line,
        past the PB's lap, or on a lap the recorder has dropped (a reset)."""
        p = self._progress(sim)            # always: it keeps the count going
        tau = self.tau(sim)
        if tau is None or self.curve is None or p is None or p < 0.0:
            return float("nan")
        r = getattr(sim, "recorder", None)
        if r is not None and not r.recording:
            return float("nan")
        t_pb = self.curve.t_at(p)
        return tau - t_pb if math.isfinite(t_pb) else float("nan")

    def flash_now(self, sim):
        f = self.flash
        if f is None or float(sim.t) > f[2]:
            return None
        return f[0], f[1]


# ==================================================================== #
#  SELF-CHECK                                                          #
# ==================================================================== #
def self_check(verbose: bool = True) -> bool:
    import tempfile
    from types import SimpleNamespace
    ok = True
    n_ok = n_all = 0

    def rep(tag, passed, msg=""):
        nonlocal ok, n_ok, n_all
        n_all += 1
        n_ok += bool(passed)
        ok = ok and bool(passed)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}: {msg}")

    if verbose:
        print("drive/ghosts.py self-check")
    # a synthetic 60 s lap on a 1200 m loop at 20 m/s, and a slower one
    L = 1200.0

    def lap(v, t_total, secs):
        rows = []
        for k in range(int(t_total * 50) + 1):
            t = k / 50.0
            s = min(v * t, L)
            rows.append((t, s, 0.0, 0.0, v, 0.0, 0.0, 0.0, s))
        r = rec._fake_rec(t_total, secs)
        r["trace"] = rec.encode_trace(rows)
        return r

    root = tempfile.mkdtemp(prefix="carsim_ghosts_")
    book = rec.RecordBook(root)
    key = rec.class_key("arena", "corsa", "sport", "patch")
    book.insert(key, lap(20.0, 60.0, [20.0, 20.0, 20.0]))
    book.insert(key, lap(19.0, L / 19.0, [21.0, 21.0, 21.2]))
    ref = np.array([(k / 50.0, 10.0 + k, 0.0, 0.0, 30.0, 0, 0, 0, 30.0 * k / 50.0)
                    for k in range(40 * 50)])
    gs = GhostSet(book, key, ref_fn=lambda k: ref)
    track = SimpleNamespace(length=L, closed=True, width=12.0)
    lt = SimpleNamespace(t_lap_start=None, lap_valid=True, _valid_run=True, lines=[0.0, 400.0, 800.0])
    sim = SimpleNamespace(t=100.0, s=L - 2.0, n_lat=0.0, track=track, lap=lt, recorder=None)
    rep("no ghost and no delta before the first crossing",
        gs.ghost_tuples(sim) == [] and math.isnan(gs.delta(sim)))

    def drive_to(s_end, t_end, n=200):
        """Move the car to s_end at t_end in n frames, reading the delta each."""
        s0, t0 = sim.s, sim.t
        span = (s_end - s0) % L if s_end < s0 and s0 - s_end > 0.5 * L else s_end - s0
        d = float("nan")
        for k in range(1, n + 1):
            sim.s = (s0 + span * k / n) % L
            sim.t = t0 + (t_end - t0) * k / n
            d = gs.delta(sim)
        return d

    drive_to(0.02, 100.001)                       # over the line ...
    lt.t_lap_start = 100.0
    gs.event(sim, ("start", 0, 100.0, float("nan")))
    d = drive_to(590.0, 130.0)                    # 30 s in, 10 m short of the PB
    rep("delta by distance: 10 m behind at 20 m/s = +0.5 s", abs(d - 0.5) < 2e-3, f"{d:+.4f} s")
    d = drive_to(622.0, 129.9 + 0.0, 20)          # AHEAD, just past half distance
    rep("ahead past half distance is a number, not a blank (review)", abs(d - (29.9 - 31.1)) < 2e-3,
        f"{d:+.4f} s")
    g = gs.ghost_tuples(sim)
    rep("two ghosts, the PB drawn last, labelled",
        [x[4] for x in g] == ["REF", "PB"] and g[1][3] == C_GHOST_PB and g[0][3] == C_GHOST_2,
        str([(x[4], round(x[0], 1)) for x in g]))
    sim.n_lat = 40.0                              # out on the open map's pad
    rep("no delta off the ribbon", math.isnan(gs.delta(sim)))
    sim.n_lat = 0.0
    gs.enabled = False
    rep("J hides the ghosts, the delta stays", gs.ghost_tuples(sim) == []
        and math.isfinite(gs.delta(sim)))
    gs.enabled = True
    gs.set_slot("top2")
    g = gs.ghost_tuples(sim)
    rep("ghost 2 can be one of your top 5 (not P1: that IS the PB)",
        [x[4] for x in g] == ["P2", "PB"] and "top1" not in SLOTS, str([x[4] for x in g]))
    gs.set_slot("none")
    rep("ghost 2 can be nothing", [x[4] for x in gs.ghost_tuples(sim)] == ["PB"])
    rep("slot labels", slot_label("ref") == "reference bot"
        and slot_label("top2", book, key).startswith("your P2 (1:03.158)")
        and slot_label("top5", book, key) == "your P5 (none yet)",
        slot_label("top2", book, key))
    rep("a finished ghost waits on the line",
        abs(GhostSet(book, key, slot="none").pb.pose(400.0)[0] - L) < 1e-6)
    #  task 41: an Unlimited session races the Unlimited book's laps, and its
    #  ghosts say so; the official book's PB is not in it
    ubook = rec.unlimited_book(root)
    ubook.insert(key, dict(lap(24.0, L / 24.0, [16.0, 16.5, 16.5]), unlimited=True))
    ubook.insert(key, dict(lap(22.0, L / 22.0, [18.0, 18.0, 18.5]), unlimited=True))
    gu = GhostSet(ubook, key, slot="top2", ref_fn=lambda k: ref)
    gu.event(sim, ("start", 0, sim.t, float("nan")))
    rep("an Unlimited session's ghosts are its own book's, labelled UNL",
        gu.pb is not None and gu.pb.label == "UNL PB" and gu.g2.label == "UNL P2"
        and abs(ubook.pb_time(key) - L / 24.0) < 1e-9 and book.pb_time(key) == 60.0
        and slot_label("top2", ubook, key).startswith("your Unlimited P2")
        and [x[4] for x in gs.ghost_tuples(sim)] != [] and GhostSet(book, key).pb.label == "PB",
        f"{gu.pb.label} / {gu.g2.label}; {slot_label('top2', ubook, key)}")
    gs.event(sim, ("lap", 1, sim.t, 30.0))        # a crossing, then backwards over it
    drive_to(L - 5.0, sim.t + 1.0, 20)
    rep("no delta for a car behind the line", math.isnan(gs.delta(sim)))
    cols = []
    for t in (19.5, 20.5, 20.2):                  # best ever, then worse than the PB
        gs.event(sim, ("sector", 0, sim.t, t))
        cols.append(gs.flash_now(sim)[1])
    gs.event(sim, ("sector", 1, sim.t, 19.99))
    cols.append(gs.flash_now(sim)[1])
    rep("sector flash: purple best ever, red slower than the PB",
        cols == ["purple", "red", "red", "purple"], str(cols))
    lt._valid_run = False                         # all four wheels were off
    gs.event(sim, ("sector", 1, sim.t, 18.0))
    inval = gs.flash_now(sim)[1]
    lt._valid_run, t0_ = True, lt.t_lap_start
    lt.t_lap_start = None                         # the out-lap
    gs.event(sim, ("sector", 1, sim.t, 18.0))
    outlap = gs.flash_now(sim)[1]
    lt.t_lap_start, lt.lap_valid = t0_, False     # the LAST sector reads lap_valid
    gs.event(sim, ("sector", 2, sim.t, 18.0))
    last = gs.flash_now(sim)[1]
    lt.lap_valid = True
    sim.recorder = SimpleNamespace(recording=False)   # a lap the recorder dropped
    gs.event(sim, ("sector", 1, sim.t, 18.0))
    dropped = gs.flash_now(sim)[1]
    rep("no colour for a sector of a lap that cannot count (review)",
        (inval, outlap, last, dropped) == ("", "", "", ""),
        f"invalid {inval!r} out-lap {outlap!r} last {last!r} dropped {dropped!r}")
    rep("... and no delta on a dropped lap", math.isnan(gs.delta(sim)))
    sim.recorder = None
    b2 = rec.RecordBook(tempfile.mkdtemp(prefix="carsim_ghosts_"))
    b2.insert(key, lap(20.0, 60.0, [20.0, 20.0, 20.0]))
    b2.insert(key, lap(21.0, 58.0, [19.0, 19.5, 19.5]))   # best sectors now 19.0 / 19.5
    b2.load(key)["laps"] = [lp for lp in b2.load(key)["laps"]   # the PB is the 60 s lap,
                            if lp["time"] == 60.0]              # the class best sectors the 58's
    g3 = GhostSet(b2, key, slot="none")
    g3.event(sim, ("sector", 0, sim.t, 19.8))
    rep("green: slower than the class best, faster than the PB lap",
        g3.flash_now(sim)[1] == "green", str(g3.flash_now(sim)))
    sim.t += FLASH_S + 0.1
    rep("the flash goes away", g3.flash_now(sim) is None)
    light = dict(time=55.0, date="x", _trace_arr=np.array([(0, 0, 0, 0, 20, 0, 0, 0, 0.0),
                                                          (60, 1200, 0, 0, 20, 0, 0, 0, 1200.0)]))
    b2.load(key)["laps"].insert(0, light)          # a lap just filed: light, trace in memory
    b2.version += 1
    g3.sync(key)
    rep("a PB just filed races at once (its trace in memory)", g3.curve is not None
        and g3.pb is not None and abs(g3.curve.t_end - 60.0) < 1e-9)
    if verbose:
        print(f"  {'ALL PASS' if ok else 'FAILURES ABOVE'}: {n_ok}/{n_all} checks")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
