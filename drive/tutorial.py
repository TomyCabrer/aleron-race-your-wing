"""drive/tutorial.py -- the driving tutorial (task 23).

Thirteen data-driven steps, `Step(id, map, kind, title, text, hint, check, ...)`:

  1 pedals      arena    throttle to 50 km/h, then brake to a stop
  2 steer       arena    steer both ways at speed
  3 turn1       arena    turn 1 from the line without leaving the road
  4 reset       arena    press R
  5 assists     (page)   what ABS, TC, the steering aid and the gearbox do
  6 wing_off    skidpad  one circle with the flank wing OFF (F / CIRCLE)
  7 wing_on     skidpad  one circle with it ON
  8 wing_result (page)   the two laps' mean lateral g, and the difference
  9 timing      (page)   sectors, the PB ghost, the delta, the medals
 10 lap         arena    one valid lap
 11 manual_intro (page)  OPTIONAL: the manual gearbox -- try it, or skip both
 12 manual      arena    OPTIONAL: up to 3rd by hand, then a downshift at speed
 13 done        (page)   what next (the wing-design tutorial is one press)

Steps 11-12 are an optional GROUP (`Step.group`): the page's "Skip it" skips
the whole group. Step 12 drives on the manual box (`Step.gearbox`): an
automatic is switched to Manual (auto clutch) for the step and back after
(drive.Sim._tutorial_gearbox; `Tutorial.gearbox_prev` remembers it, and a
box the player changed meanwhile is theirs).

A DRIVE step is an overlay box on the road (HudData.tutorial, drawn by
render._draw_tutorial) with what to do, a live status line, a hint after
`HINT_AFTER_S` without success, and why an attempt did not count. Its
`check(frame, mem)` is a predicate on a `frame` -- a plain namespace of the
HUD's and the sim's state this frame (speed, lateral g, yaw rate, `s`, on the
road, the wing toggle, the LapTimer events and the discrete commands since
the last frame; `frame_of`) -- and `mem`, the step's own memory. The
self-check drives every predicate with synthetic frames; drive.py's V34
drives the whole tutorial headless with ScriptedInput, the manual box
included.

A PAGE step is a paused menu page (keyboard, pad, mouse, like every page):
Continue, or End. ESC, in a drive step or on a page, opens the pause menu
on its Tutorial page: skip the step, start over, end the tutorial.

A step names its MAP; when the session is on another one the tutorial asks
for a restart and `drive.run_interactive_cli` moves the map (the session is
rebuilt there: TAB cannot leave a map the tutorial is on). The wing steps
need a flank wing: a car without one gets the library's published plate for
those laps (`wing_car`, in memory only, never saved).

Progress lives in `runs/progress.json`, section `tutorial`
(drive/progress.py): offered, the step reached, done, skipped steps and the
wing laps' numbers. Only a player session opens it; scripted and headless
runs never read it. The tutorial is offered once, on the first launch, and
is always in the pause menu.

Nothing here touches the physics: the tutorial reads the sim, and the only
thing it does to it is a reset to the line when a step starts from there.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

from corsa_c import G

SECTION = "tutorial"
#: the fields each saved result must carry (the pages format them as numbers)
_RESULT_FIELDS = {"wing_off": ("ay", "t"), "wing_on": ("ay", "t"), "lap": ("t",)}
HINT_AFTER_S = 20.0         # s of sim time in a drive step before its hint shows
DONE_FLASH_S = 3.0          # s the "done:" line stays up
PEDALS_KMH = 50.0           # step 1: this fast, then ...
STOP_KMH = 3.0              # ... this slow
STEER_KMH = 20.0            # step 2: a yaw rate counts above this speed
STEER_YAW_DEG = 6.0         # deg/s each way (0.1 rad/s: a lane change, not a twitch)
TURN1_PAST_M = 15.0         # step 3: this far past turn 1's exit
MAX_DS_M = 30.0             # a jump in s bigger than this between frames is a teleport
LAP_MIN_FRACTION = 0.95     # a lap is a lap ROUND the circuit (records.py's rule)
MANUAL_TOP = 3              # step 12: up to this gear by hand ...
MANUAL_DOWN_KMH = 30.0      # ... then a downshift at this speed or more (under braking)
TUTORIAL_WING = "plate"     # the library's published flank panel (wing_car)
MINUTES = 10                # what the offer says it takes

KEYS = "ESC / OPTIONS: the tutorial menu (skip a step, start over, end)"


# ==================================================================== #
#  THE STEPS                                                           #
# ==================================================================== #
@dataclass(frozen=True)
class Step:
    id: str
    map: str | None            # the track it is driven on; None = wherever
    kind: str                  # 'drive' (overlay + predicate) | 'page' (paused page)
    title: str
    text: object               # str, or (ctx, tut) -> (note, sections) for a page
    hint: str = ""
    check: object = None       # (frame, mem) -> bool, drive steps
    status: object = None      # (frame, mem) -> str, the live line
    reset: bool = False        # the step starts on the line, standing
    setup: object = None       # (track, mem) -> None, once when the step starts
    wing: bool = False         # needs a car with a flank wing
    gearbox: str | None = None  # drives on this box ('manual'): an automatic is
    #                             switched for the step and back after
    group: str = ""            # an OPTIONAL group: its page offers "Skip it",
    #                             which skips every step of the group


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _ds(f, m):
    """Metres along the centreline since the last frame, wrapped on a closed
    track; None on the first frame or across a teleport (a reset)."""
    s0 = m.get("s_prev")
    m["s_prev"] = f.s
    if s0 is None:
        return None
    d = f.s - s0
    if f.closed and f.L > 0:
        d = (d + 0.5 * f.L) % f.L - 0.5 * f.L
    return d if abs(d) <= MAX_DS_M else None


# -- 1 throttle and brake ------------------------------------------------
def _chk_pedals(f, m):
    m["vmax"] = max(m.get("vmax", 0.0), f.V_kmh)
    if f.V_kmh >= PEDALS_KMH:
        m["fast"] = True
    return bool(m.get("fast")) and f.V_kmh < STOP_KMH


def _st_pedals(f, m):
    if not m.get("fast"):
        return f"{f.V_kmh:3.0f} of {PEDALS_KMH:.0f} km/h"
    return f"now stop: {f.V_kmh:3.0f} km/h"


# -- 2 steering -----------------------------------------------------------
def _chk_steer(f, m):
    if f.V_kmh >= STEER_KMH:
        if f.yaw_deg >= STEER_YAW_DEG:
            m["left"] = True
        elif f.yaw_deg <= -STEER_YAW_DEG:
            m["right"] = True
    return bool(m.get("left") and m.get("right"))


def _st_steer(f, m):
    if f.V_kmh < STEER_KMH:
        return f"faster: {f.V_kmh:.0f} of {STEER_KMH:.0f} km/h"
    yes = lambda k: "yes" if m.get(k) else "not yet"   # noqa: E731
    return f"left {yes('left')}   right {yes('right')}"


# -- 3 turn 1 -------------------------------------------------------------
def turn1_span(tr):
    """(start, end) of the first corner along the centreline, from the
    track's own segments; None on a track with no corner."""
    s = 0.0
    for seg in getattr(tr, "segs", []):
        if seg.kind == "A":
            return s, s + seg.length
        s += seg.length
    return None


def _setup_turn1(tr, m):
    sp = turn1_span(tr)
    if sp is not None:
        m["t1"], m["goal"] = sp[0], sp[1] + TURN1_PAST_M
    m["armed"], m["clean"] = True, True


def _chk_turn1(f, m):
    if "goal" not in m:
        return False
    if "reset" in f.cmds:
        # a reset: another go, from wherever R put the car (the frame after
        # it is already there) -- before the corner it counts, past it
        # SHIFT+R is the way back to the line
        m["clean"] = True
        m["armed"] = f.s < m["t1"] or f.s > f.L - MAX_DS_M
    if not f.on_track:
        m["clean"] = False
    return bool(m["armed"] and m["clean"] and m["goal"] <= f.s < m["goal"] + 200.0)


def _st_turn1(f, m):
    if "goal" not in m:
        return ""
    if not m["clean"]:
        return "off the road: R (CREATE) puts you back, then try again"
    if not m["armed"]:
        return "past the corner: SHIFT+R starts over at the line"
    to_go = m["goal"] - f.s if f.s < m["goal"] else 0.0
    return f"turn 1: {to_go:4.0f} m to go"


# -- 4 reset ----------------------------------------------------------------
def _chk_reset(f, m):
    return "reset" in f.cmds


# -- 6, 7 the wing on the skidpad ---------------------------------------------
def _circle_restart(m, why=""):
    m.update(dist=0.0, ay_dt=0.0, tt=0.0, clean=True)
    m["why"] = why


def _circle(f, m, want_on: bool) -> bool:
    """One flying LAP of the skidpad, line to line, on the road, with the
    wing in the wanted state throughout: its mean |lateral g| (time-weighted)
    and its time go to m['result']. Line to line, so both wing laps are
    flying laps: a lap from a standstill would hand the second one the
    difference."""
    if "dist" not in m:
        _circle_restart(m)
        m["counting"] = False
    dt = f.t - m.get("t_prev", f.t)
    m["t_prev"] = f.t
    ds = _ds(f, m)
    for e in f.events:
        if e[0] not in ("start", "lap"):
            continue
        if (e[0] == "lap" and m["counting"] and m["clean"] and m["tt"] > 0.0
                and m["dist"] >= LAP_MIN_FRACTION * f.L):
            m["result"] = dict(ay=round(m["ay_dt"] / m["tt"], 4), t=round(float(e[3]), 3))
            return True
        _circle_restart(m, "" if f.wing_on == want_on else m.get("why", ""))
        m["counting"] = True                   # a lap starts at this crossing
    if "reset" in f.cmds:
        _circle_restart(m, "reset: the lap starts again at the line")
        m["counting"] = False
        return False
    if f.wing_on != want_on:
        m["clean"] = False
        m["why"] = "the wing is " + ("OFF: F (CIRCLE) switches it on" if want_on
                                     else "ON: F (CIRCLE) switches it off")
        return False
    if want_on and getattr(f, "wing_mode", 0) not in (0, 3):
        #  LEFT / RIGHT / ALL 3 (drive/airbrake.py) are not the corner's law:
        #  the ON lap measures AUTO (AIR BRAKE is AUTO off the brake)
        m["clean"] = False
        m["why"] = "the wing mode is not AUTO: G (TRIANGLE) steps it back to AUTO"
        return False
    if not f.on_track:
        m["clean"] = False
        m["why"] = "off the circle: this lap does not count, the next starts at the line"
    if m["why"].startswith("the wing"):
        m["why"] = "the next lap starts at the line" if m["counting"] else ""
    if ds is None or dt <= 0.0 or not m["counting"] or not m["clean"]:
        return False
    m["dist"] += ds
    m["ay_dt"] += abs(f.ay_g) * dt
    m["tt"] += dt
    return False


def _chk_wing_off(f, m):
    return _circle(f, m, False)


def _chk_wing_on(f, m):
    return _circle(f, m, True)


def _st_circle(f, m):
    if "dist" not in m or f.L <= 0:
        return ""
    wing = "" if f.has_flank else "   (this car has no flank wing)"
    if not m.get("counting") or not m.get("clean"):
        return f"to the line: the lap starts there   now {abs(f.ay_g):.2f} g{wing}"
    ay = m["ay_dt"] / m["tt"] if m.get("tt", 0.0) > 0.0 else 0.0
    return (f"lap {100.0 * max(m['dist'], 0.0) / f.L:3.0f} %   mean {ay:.2f} g   "
            f"now {abs(f.ay_g):.2f} g{wing}")


# -- 10 one valid lap -----------------------------------------------------------
def _chk_lap(f, m):
    for e in f.events:
        if e[0] == "start":
            m.update(counting=True, dist=0.0)
        elif e[0] == "lap":
            full = m.get("counting") and m.get("dist", 0.0) >= LAP_MIN_FRACTION * f.L
            rw = getattr(f, "rec_why", "")
            if full and f.lap_valid and not rw:
                m["result"] = dict(t=round(float(e[3]), 3))
                return True
            if m.get("counting"):
                m["why"] = ("that lap did not count: "
                            + ("all four wheels left the road" if not f.lap_valid
                               else f"it was not recorded ({rw})" if rw
                               else "it was not a full lap") + ". The next one starts now")
            m.update(counting=True, dist=0.0)
    if "reset" in f.cmds:
        m["counting"] = False
        m["why"] = "a reset: the lap starts again at the line"
    ds = _ds(f, m)
    if ds is not None and m.get("counting"):
        m["dist"] = m.get("dist", 0.0) + ds
    return False


def _st_lap(f, m):
    if not m.get("counting"):
        return "drive to the line: the clock starts there"
    return f"lap {100.0 * max(m.get('dist', 0.0), 0.0) / max(f.L, 1.0):3.0f} %   {f.lap_time:6.1f} s"


# -- 12 the manual gearbox (optional) -------------------------------------------------
def _chk_manual(f, m):
    """Up to MANUAL_TOP by hand, then a downshift at MANUAL_DOWN_KMH or more
    (under braking, before a corner): the two things the automatic did. A
    gear is read only while one is in (a shift passes through neutral), so
    3 -> N -> 2 is one downshift. Only a change UP counts toward the top
    gear. A reset (R keeps the speed and may land in any gear) starts the
    reading over; the automatic's own shifts count for nothing -- a spell
    on it clears the top gear too."""
    if f.gearbox == "auto":
        m["why"] = ("the gearbox is on Automatic: ESC > Settings > Gearbox > Manual, "
                    "or ESC > Tutorial skips this step")
        m.pop("g", None)
        m["top"] = 0
        return False
    if str(m.get("why", "")).startswith("the gearbox"):
        m["why"] = ""
    if "reset" in f.cmds:
        m.pop("g", None)
    g = f.gear
    if g >= 1:
        last, top = m.get("g"), m.get("top", 0)
        if last is None:
            top = max(top, 1) if g == 1 else top   # a reading from 1st is a start
        elif g > last:
            top = max(top, g)                      # a change up, by hand
        elif g < last and top >= MANUAL_TOP and f.V_kmh >= MANUAL_DOWN_KMH:
            m["down"] = True
        m["g"], m["top"] = g, top
    return bool(m.get("down"))


def _st_manual(f, m):
    g = str(f.gear) if f.gear >= 1 else ("R" if f.gear < 0 else "N")
    top = m.get("top", 0)
    if top < MANUAL_TOP:
        return f"gear {g}   up with E (R1) at the lights: {max(top, 1)} of {MANUAL_TOP}"
    return f"gear {g}   now brake, and down with Q (L1) above {MANUAL_DOWN_KMH:.0f} km/h"


# -- the pages ---------------------------------------------------------------------
def _onoff(v) -> str:
    return "on" if v else "off"


def _pg_assists(ctx, tut):
    s = ctx.settings
    gb = getattr(s, "gearbox", "auto")
    note = ("The assists are on the settings page (ESC > Settings). Every lap "
            "stores which ones were on, and they are shown next to your times; they "
            "never split a class, so turn them off when you are ready and your "
            "tables stay the same.")
    rows = [(f"ABS  {_onoff(getattr(s, 'abs', True))}",
             "keeps the wheels turning under hard braking, so the car still steers"),
            (f"TC  {_onoff(getattr(s, 'tc', False))}",
             "cuts the power when the driven wheels spin: exits without a slide"),
            (f"steer aid  {_onoff(getattr(s, 'steer_aid', True))}",
             "limits the lock to what the front tyres can use; off = the full lock"),
            (f"gearbox  {gb}",
             "auto shifts for you; manual: E / Q (R1 / L1); clutch: Z as well")]
    return note, [("THE ASSISTS", rows)]


def _fmt_g(r) -> str:
    return f"{r['ay']:.2f} g in {r['t']:.1f} s" if r else "not measured (skipped)"


#: a difference in mean lateral g smaller than this is "about the same"
WING_SAME_G = 0.02


def _pg_wing(ctx, tut):
    off, on = tut.results.get("wing_off"), tut.results.get("wing_on")
    rows = [("wing OFF", _fmt_g(off)), ("wing ON", _fmt_g(on))]
    verdict = ""
    if off and on:
        d = on["ay"] - off["ay"]
        rows.append(("difference", f"{d:+.2f} g   ({on['t'] - off['t']:+.2f} s a lap)"))
        if d > WING_SAME_G:
            verdict = "The wing lap held more lateral g: grip the panel added. "
        elif d < -WING_SAME_G:
            verdict = ("The wing lap held less: you pushed less on it. Near the limit "
                       "(the tyres squeal) the panel adds grip. ")
        else:
            verdict = "About the same. "
    note = (verdict + "A flank wing is a small wing on the side of the car. In a corner "
            "the panel on the OUTSIDE opens and pushes the car toward the inside of the "
            "turn: a sideways force the tyres do not have to find. It costs drag, so it "
            "opens only in corners (G picks the side). The published plate is small -- "
            "at the limit it is worth about +2 % of corner speed (README, The device) -- "
            "so two laps by hand mostly show how hard each was driven. The garage "
            "(BACKSPACE) designs bigger ones.")
    return note, [("YOUR TWO LAPS", rows)]


def _pg_timing(ctx, tut):
    rows = [("sectors", "the lap is cut in 3; the timing panel shows each one"),
            ("the flash", "at a sector line: purple = best ever in this class, "
                          "green = better than your PB lap, red = slower"),
            ("PB ghost", "your best lap in this class, a flat car on the road, from the line"),
            ("ghost 2", "the reference bot (the pre-race page changes it); J hides both"),
            ("delta", "under the timing panel: -0.23 in green = ahead of your PB here")]
    secs = [("TIMING", rows)]
    key = getattr(ctx, "key", None)
    tg = None
    if key:
        try:
            from . import medals
            tg = medals.targets(key)
        except Exception:                  # noqa: BLE001 -- medals are optional
            tg = None
    if tg:
        from .records import fmt_time
        secs.append(("MEDALS, THIS CLASS", [(m, fmt_time(tg[m]))
                                            for m in ("author", "gold", "silver", "bronze")]))
    note = ("A CLASS is the map, the car, the engine and the surface. Your 5 best "
            "valid laps in each are kept; the medals are set by a reference driver "
            "on the stock car. Next: one valid lap -- it goes in the table.")
    return note, secs


def _pg_manual(ctx, tut):
    gb = getattr(getattr(ctx, "settings", None), "gearbox", "auto")
    mine = ("Your gearbox is already manual, so nothing changes." if gb != "auto" else
            "For the next step the box is on MANUAL; it goes back to Automatic after "
            "(ESC > Settings > Gearbox keeps manual for good).")
    note = ("Optional. With the manual box you choose the gear: the right one for a "
            "corner's exit, engine braking into it, and no change up in the middle of "
            "a corner. " + mine)
    keys = [("E / R1", "shift up"), ("Q / L1", "shift down"),
            ("shift lights", "the LEDs over the rev bar: all lit = change up now"),
            ("the gear", "the big number next to the speed; N while it changes")]
    how = [("up", "foot flat, at the shift lights: the box lifts for the change"),
           ("down", "under braking, one gear at a time, before you turn in;"),
           ("", "the box blips the throttle to match the revs"),
           ("pulling away", "in 1st; the box works the clutch, it cannot stall"),
           ("clutch mode", "Settings > Gearbox > Manual + clutch pedal: Z (SQUARE)"),
           ("", "is the clutch, you launch on it, it can stall; S restarts")]
    return note, [("THE KEYS", keys), ("HOW", how)]


def _pg_done(ctx, tut):
    from .records import fmt_time
    lap = tut.results.get("lap")
    rows = [("your lap", fmt_time(lap["t"]) if lap else "skipped")]
    off, on = tut.results.get("wing_off"), tut.results.get("wing_on")
    if off and on:
        rows.append(("the wing", f"{on['ay'] - off['ay']:+.2f} g on the skidpad"))
    if tut.skipped:
        titles = {s.id: s.title for s in tut.steps}
        rows.append(("skipped", ", ".join(titles.get(k, k) for k in tut.skipped)))
    nxt = [("Time trial", "ESC > Time trial: your top 5, medals, another build")]
    if getattr(ctx, "garage", False):
        nxt.append(("Wing design", "the row below: your first wing, step by step, in the garage"))
    nxt += [("Gearbox", "ESC > Settings > Gearbox: Manual keeps it for good"),
            ("Tutorial", "ESC > Tutorial takes it again, any time")]
    return "That is the whole loop: tweak the car, drive it, beat the number.", \
        [("DONE", rows), ("NEXT", nxt)]


STEPS = (
    Step("pedals", "arena", "drive", "Throttle and brake",
         "Hold UP (R2) to reach 50 km/h, then DOWN (L2) to stop.",
         hint="UP is the throttle, DOWN the brake. The car starts in 1st; the "
              "automatic gearbox shifts for you.",
         check=_chk_pedals, status=_st_pedals, reset=True),
    Step("steer", "arena", "drive", "Steering",
         "Keep some speed and steer LEFT, then RIGHT (the left stick).",
         hint="Steering does nothing at a standstill: 20 km/h or more. Small "
              "inputs: the car answers quickly.",
         check=_chk_steer, status=_st_steer),
    Step("turn1", "arena", "drive", "Turn 1",
         "From the line, take the first corner without leaving the road.",
         hint="Brake on the straight BEFORE the corner, turn in, and add throttle "
              "only once the car points out of it.",
         check=_chk_turn1, status=_st_turn1, reset=True, setup=_setup_turn1),
    Step("reset", "arena", "drive", "Reset",
         "Press R (CREATE): back to the last sector line you crossed. "
         "SHIFT+R (Full reset) puts you on the start line.",
         hint="R, once.", check=_chk_reset),
    Step("assists", None, "page", "What the assists do", _pg_assists),
    Step("wing_off", "skidpad", "drive", "The wing, lap 1: OFF",
         "Wing OFF (F / CIRCLE toggles it). Drive a flying lap of the circle, line "
         "to line, as fast as the car will hold it.",
         hint="Stay on the painted circle. Near the limit the tyres squeal: that "
              "is as fast as it goes.",
         check=_chk_wing_off, status=_st_circle, reset=True, wing=True),
    Step("wing_on", "skidpad", "drive", "The wing, lap 2: ON",
         "Now switch the wing ON (F / CIRCLE) and drive another flying lap, "
         "just as hard.",
         hint="The panel on the outside of the turn opens by itself as you corner.",
         check=_chk_wing_on, status=_st_circle, wing=True),
    Step("wing_result", None, "page", "What the wing did", _pg_wing),
    Step("timing", "arena", "page", "Sectors, the PB ghost, medals", _pg_timing),
    Step("lap", "arena", "drive", "One valid lap",
         "Drive one full lap. The clock starts when you cross the line; all four "
         "wheels off the road and the lap does not count.",
         hint="Brake before the corners, not in them. Any valid lap will do.",
         check=_chk_lap, status=_st_lap, reset=True),
    Step("manual_intro", None, "page", "The manual gearbox (optional)", _pg_manual,
         group="manual"),
    Step("manual", "arena", "drive", "Manual gearbox (optional)",
         "The box is on MANUAL. Pull away in 1st and shift UP with E (R1) each time "
         "the shift lights fill, up to 3rd. Then brake and shift DOWN with Q (L1), "
         "one gear, before a corner.",
         hint="E / R1 up, Q / L1 down; the box works the clutch, it cannot stall. "
              "Change down while braking, not in the corner. ESC > Tutorial skips it.",
         check=_chk_manual, status=_st_manual, reset=True, gearbox="manual",
         group="manual"),
    Step("done", None, "page", "Tutorial complete", _pg_done),
)


# ==================================================================== #
#  THE FRAME: what a predicate sees                                    #
# ==================================================================== #
def frame_of(sim, tut=None):
    """This frame's state for the predicates: the HUD's numbers plus the
    sim's `s`, the LapTimer events and the discrete commands (R ...) since
    the last frame. Reads the sim; changes nothing in it."""
    v = sim.veh
    events, cmds = [], []
    if tut is not None:
        log = sim.events_log
        if tut._ev_src is not log:         # a new session: its own log
            tut._ev_src, tut._ev_i = log, len(log)
        events = list(log[tut._ev_i:])
        tut._ev_i = len(log)
        cmds, tut._cmds = tut._cmds, []
    rec, rec_why = getattr(sim, "recorder", None), ""
    last = getattr(rec, "last", None)
    if (rec is not None and isinstance(last, dict) and not last.get("valid")
            and any(e[0] == "lap" for e in events)):
        rec_why = str(last.get("why") or "not recorded")   # the lap the records refused
    cfg = v.cfg
    has_flank = (getattr(cfg, "wing", "off") != "off"
                 or getattr(cfg, "dev_left", None) is not None
                 or getattr(cfg, "dev_right", None) is not None)
    return SimpleNamespace(
        t=float(sim.t), V_kmh=math.hypot(v.u, v.v) * 3.6, ay_g=float(v.ay) / G,
        yaw_deg=math.degrees(v.r), s=float(sim.s), L=float(sim.track.length),
        closed=bool(sim.track.closed), on_track=bool(sim.on_track),
        lap_valid=bool(sim.lap.lap_valid), lap_time=float(sim.lap.lap_time),
        wing_on=bool(sim.wing_on), has_flank=bool(has_flank),
        gear=int(getattr(v, "gear", 0)), gearbox=str(getattr(sim, "gearbox", "auto")),
        wing_mode=int(getattr(sim, "wing_side_mode", 0) or 0),
        events=events, cmds=cmds, track=sim.track.name, rec_why=rec_why)


# ==================================================================== #
#  THE TUTORIAL                                                        #
# ==================================================================== #
class Tutorial:
    """The state machine. One per tutorial run; it survives session restarts
    (drive.run_interactive_cli carries it on `opts.tutorial`)."""

    def __init__(self, progress=None, start: str | None = None, steps=STEPS):
        self.steps = tuple(steps)
        self.progress = progress
        sec = progress.section(SECTION) if progress is not None else {}
        res = sec.get("results")
        self.results = {}
        for k, v in (res.items() if isinstance(res, dict) else ()):
            #  a hand-edited or foreign entry must not take a page down
            if isinstance(v, dict) and all(_num(v.get(f_)) for f_ in _RESULT_FIELDS.get(k, ())):
                self.results[k] = v
            else:
                print(f"progress: tutorial result {k!r} ignored (malformed)")
        sk = sec.get("skipped")
        self.skipped = [str(x) for x in sk] if isinstance(sk, list) else []
        ids = [s.id for s in self.steps]
        self.i = ids.index(start) if start in ids else 0
        if self.i == 0:                        # from the top: a fresh run
            self.skipped = []
            self.results = {}
        self.active = True
        self.done = False
        self.flash = ""
        self._flash_until = -1.0
        self._t_now = 0.0
        self._sim = None                       # the session last ticked (its t restarts at 0)
        self._last = None
        #: (the player's box, the one this tutorial set) while a Step.gearbox
        #: step has switched it; drive.Sim._tutorial_gearbox puts it back.
        #: Saved in the progress file as well (`set_gearbox_prev`): a run
        #: that dies mid-step leaves it there for the next launch to undo
        #: (`gearbox_left`)
        self.gearbox_prev = None
        self.gearbox_seen = None               # the step whose box was checked: once
        self._begin()
        self._save()

    # ---------------------------------------------------------------- #
    def _begin(self) -> None:
        self.mem: dict = {}
        self._t0 = None
        self._setup_due = True
        self._cmds: list = []
        self._ev_src, self._ev_i = None, 0
        self._last = None

    @property
    def step(self) -> Step:
        return self.steps[min(self.i, len(self.steps) - 1)]

    def label(self) -> str:
        return f"{self.i + 1}/{len(self.steps)}"

    def map_wanted(self):
        return self.step.map if self.active else None

    def wants_wing(self) -> bool:
        return self.active and self.step.wing

    def gearbox_wanted(self):
        """The box the current step drives on (Step.gearbox), or None."""
        st = self.step
        return st.gearbox if (self.active and st.kind == "drive") else None

    def set_gearbox_prev(self, v) -> None:
        self.gearbox_prev = tuple(v) if v else None
        self._save()

    def command(self, ev: str) -> None:
        """A discrete command the session saw (a reset). Kept to the next frame."""
        if self.active:
            self._cmds.append(ev)

    # ---------------------------------------------------------------- #
    def tick(self, sim):
        """Once per frame, after the physics. Returns what the session must
        do: 'restart' (the step is on another map), 'page' (a page step: open
        it, paused), 'done' (a drive step just passed), or None."""
        if not self.active:
            return None
        if sim is not self._sim:               # a new session: its sim.t starts at 0 again,
            d = float(sim.t) - self._t_now     # so the flash and the hint clock move with it
            self._flash_until += d
            if self._t0 is not None:
                self._t0 += d
            self._sim, self._t_now = sim, float(sim.t)
        st = self.step
        if st.map and st.map != sim.track.name:
            return "restart"
        if st.kind == "page":
            return "page"
        self._t_now = float(sim.t)
        if self._setup_due:
            self._setup_due = False
            if st.reset:
                sim.reset(to_checkpoint=False)
            if st.setup is not None:
                st.setup(sim.track, self.mem)
            self._cmds = []                # the step's own reset is not the player's
            self._ev_src, self._ev_i = sim.events_log, len(sim.events_log)
            self._t0 = float(sim.t)        # the hint's clock starts with the step
            return None
        if sim.paused:
            return None
        f = frame_of(sim, self)
        if self._t0 is None:
            self._t0 = f.t
        self._last = f
        if st.check is not None and st.check(f, self.mem):
            self._passed(st, sim)
            return "done"
        return None

    def _passed(self, st: Step, sim) -> None:
        res = self.mem.get("result")
        extra = ""
        if res is not None:
            self.results[st.id] = dict(res)
            if "ay" in res:
                extra = f": {res['ay']:.2f} g in {res['t']:.1f} s"
            elif "t" in res:
                from .records import fmt_time
                extra = f": {fmt_time(res['t'])}"
                key = getattr(getattr(sim, "recorder", None), "key", None)
                try:
                    from . import medals
                    m = medals.medal_for(key, res["t"]) if key else None
                except Exception:          # noqa: BLE001
                    m = None
                if m:
                    extra += f"  {m.upper()}"
        self.flash = f"done: {st.title}{extra}"
        self._flash_until = float(sim.t) + DONE_FLASH_S
        self.advance()

    def advance(self, skipped: bool = False) -> None:
        if skipped and self.step.id not in self.skipped:
            self.skipped.append(self.step.id)
        if skipped:                            # a skipped step has no number: never an old one
            self.results.pop(self.step.id, None)
        self.i += 1
        if self.i >= len(self.steps):
            self.i = len(self.steps) - 1
            self.active = False
            self.done = True
        else:
            self._begin()
        self._save()

    def skip(self, group: bool = False) -> None:
        """Skip this step; with `group`, every step of its optional group."""
        g = self.step.group if group else ""
        self.flash = f"skipped: {self.step.title}"
        self._flash_until = self._t_now + DONE_FLASH_S
        self.advance(skipped=True)
        while g and self.active and self.step.group == g:
            self.advance(skipped=True)

    def end(self) -> None:
        """Stop here; the pause menu's Tutorial page continues from this step."""
        self.active = False
        self._save()

    def _save(self) -> None:
        if self.progress is None:
            return
        sec = self.progress.section(SECTION)
        sec["offered"] = True
        sec["done"] = bool(sec.get("done")) or self.done
        sec["step"] = None if self.done else self.step.id
        sec["results"] = dict(self.results)
        sec["skipped"] = list(self.skipped)
        sec["gearbox_prev"] = list(self.gearbox_prev) if self.gearbox_prev else None
        self.progress.save(SECTION)

    # ---------------------------------------------------------------- #
    def overlay(self):
        """HudData.tutorial for a drive step (None otherwise): the head, what
        to do, the live status, why the last attempt did not count, the hint
        (after HINT_AFTER_S), the "done:" flash and the keys."""
        flash = self.flash if self._t_now < self._flash_until else ""
        if not self.active or self.step.kind != "drive":
            return dict(head="TUTORIAL", flash=flash) if flash else None
        st, f = self.step, self._last
        status = st.status(f, self.mem) if (st.status is not None and f is not None) else ""
        t_in = (f.t - self._t0) if (f is not None and self._t0 is not None) else 0.0
        return dict(head=f"TUTORIAL {self.label()}   {st.title}", text=st.text,
                    status=status, warn=str(self.mem.get("why", "") or ""),
                    hint=st.hint if t_in >= HINT_AFTER_S else "", flash=flash, foot=KEYS)

    def page(self, ctx):
        """(title, subtitle, note, sections, items) for a page step."""
        st = self.step
        note, secs = st.text(ctx, self) if callable(st.text) else (str(st.text), [])
        if st.id == "done":
            items = [("Drive on", "tut_next")]
            if getattr(ctx, "garage", False):
                items.append(("Next: the wing-design tutorial (in the garage)", "tut_wing"))
        elif st.group:                         # an optional group's page
            items = [("Try it", "tut_next"),
                     ("Skip it" + (" (the gearbox stays as it is)" if st.group == "manual"
                                   else ""), "tut_skip_group"),
                     ("End the tutorial", "tut_end")]
        else:
            items = [("Continue", "tut_next"), ("End the tutorial", "tut_end")]
        flash = self.flash if self._t_now < self._flash_until else ""
        sub = flash or f"step {self.label()}"
        return f"TUTORIAL {self.label()}  {st.title}", sub, note, secs, items


# ==================================================================== #
#  THE PAUSE MENU'S PAGE AND THE FIRST-LAUNCH OFFER                    #
# ==================================================================== #
def step_list(tut=None, progress=None) -> list:
    """The steps as help rows, the current one marked."""
    cur = tut.step.id if (tut is not None and tut.active) else None
    rows = []
    for i, s in enumerate(STEPS):
        mark = "  <- now" if s.id == cur else ""
        rows.append((f"{i + 1:2d}", s.title + (f"  ({s.map})" if s.map else "") + mark))
    return [("THE STEPS", rows)]


def saved_state(progress) -> dict:
    sec = progress.section(SECTION) if progress is not None else {}
    step = sec.get("step")
    ids = [s.id for s in STEPS]
    return dict(offered=bool(sec.get("offered")), done=bool(sec.get("done")),
                step=step if step in ids else None,
                index=ids.index(step) if step in ids else None)


def gearbox_left(progress):
    """(the player's box, the one a tutorial step set) that a run which did
    not end cleanly (a crash, a kill, a closed terminal) left in the progress
    file, or None. `run_interactive_cli` undoes it at launch."""
    sec = progress.section(SECTION) if progress is not None else {}
    gp = sec.get("gearbox_prev")
    modes = ("auto", "manual", "clutch")
    if isinstance(gp, list) and len(gp) == 2 and all(x in modes for x in gp):
        return tuple(gp)
    return None


def menu_items(tut=None, progress=None, garage: bool = False) -> list:
    """The pause menu's Tutorial page (with a garage: the wing-design
    tutorial's row, drive/wing_tutorial.py)."""
    wing = [("Wing-design tutorial: a guided first wing, in the garage", "wt_garage")] \
        if garage else []
    if tut is not None and tut.active:
        return [("Back to the drive", "tut_back"),
                (f"Skip this step ({tut.step.title})", "tut_skip"),
                ("Start over", "tut_start"),
                ("End the tutorial (continue it later from here)", "tut_end")] + wing
    sv = saved_state(progress)
    rows = []
    if sv["step"] is not None and sv["index"]:
        st = STEPS[sv["index"]]
        rows.append((f"Continue at step {sv['index'] + 1}: {st.title}", "tut_resume"))
        rows.append(("Start over", "tut_start"))
    else:
        rows.append(((("Take it again" if sv["done"] else "Start the driving tutorial")
                      + f" ({len(STEPS)} steps, about {MINUTES} minutes)"), "tut_start"))
    return rows + wing + [("Back", "tut_back")]


def menu_row(progress=None, tut=None) -> str:
    """The pause page's row for it."""
    if tut is not None and tut.active:
        return f"Tutorial: step {tut.label()}, skip or end"
    sv = saved_state(progress)
    if sv["step"] is not None and sv["index"]:
        return f"Tutorial: continue at step {sv['index'] + 1}"
    return "Tutorial: learn to drive" + (" (done)" if sv["done"] else "")


def offer_items(garage: bool = False) -> list:
    """The first launch's WELCOME rows; with a garage, the wing-design
    tutorial too (drive/wing_tutorial.py)."""
    rows = [(f"Start the driving tutorial ({len(STEPS)} steps, about {MINUTES} minutes)",
             "tut_start")]
    if garage:
        rows.append(("Wing-design tutorial: design a wing, in the garage", "wt_garage"))
    return rows + [("Not now (the pause menu has both: ESC > Tutorial)", "tut_later")]


OFFER_ITEMS = offer_items()
OFFER_NOTE = ("New here? The driving tutorial takes you through the pedals, the "
              "steering, a corner, the reset, the assists, what the flank wing does on "
              "the skidpad, the timing and the medals, one valid lap and, if you like, "
              "the manual gearbox. The wing-design tutorial, in the garage, walks you "
              "through designing your own wing.")


def wing_car(grg, design, lib, car=None):
    """The car the wing steps are driven in when `design` has no flank wing:
    a copy with the library's published plate on both flanks (named so; in
    memory, never saved), fitted to `car` -- the car the session drives (a
    `cars.py` key; None: the build's own tag, else the Corsa -- task 41's
    per-car slot bands). None when the car has one already, or the library
    has no plate."""
    if design is None or grg is None or lib is None:
        return None
    if getattr(design.left, "wing", "") or getattr(design.right, "wing", ""):
        return None
    if TUTORIAL_WING not in getattr(lib, "wings", {}):
        return None
    d = grg.CarBuild.from_json(design.to_json())
    d.name = f"{design.name} + tutorial plate"
    d.left.wing = TUTORIAL_WING
    d.mirror = True
    d.clamp(lib, car)
    return d


# ==================================================================== #
#  SELF-CHECK                                                          #
# ==================================================================== #
def _frame(**kw):
    f = dict(t=0.0, V_kmh=0.0, ay_g=0.0, yaw_deg=0.0, s=0.0, L=1249.2, closed=True,
             on_track=True, lap_valid=True, lap_time=0.0, wing_on=False, has_flank=True,
             gear=1, gearbox="manual", wing_mode=0, events=[], cmds=[], track="arena",
             rec_why="")
    f.update(kw)
    return SimpleNamespace(**f)


def self_check(verbose: bool = True) -> bool:
    import os
    import tempfile
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}" + (f": {msg}" if msg else ""))

    # --- 1 pedals
    m = {}
    a = _chk_pedals(_frame(V_kmh=2.0), m)
    b = _chk_pedals(_frame(V_kmh=35.0), m)
    c = _chk_pedals(_frame(V_kmh=51.0), m)
    d = _chk_pedals(_frame(V_kmh=20.0), m)
    e = _chk_pedals(_frame(V_kmh=2.5), m)
    rep("pedals: standing is not done, 50 km/h then a stop is", (a, b, c, d, e)
        == (False, False, False, False, True), _st_pedals(_frame(V_kmh=20.0), m))
    # --- 2 steer
    m = {}
    r = [_chk_steer(_frame(V_kmh=10.0, yaw_deg=20.0), m),     # too slow: ignored
         _chk_steer(_frame(V_kmh=30.0, yaw_deg=8.0), m),      # left
         _chk_steer(_frame(V_kmh=30.0, yaw_deg=-3.0), m),     # not enough
         _chk_steer(_frame(V_kmh=30.0, yaw_deg=-7.0), m)]     # right
    rep("steer: both ways above 20 km/h, a twitch or a crawl does not count",
        r == [False, False, False, True] and m.get("left") and m.get("right"), str(r))
    # --- 3 turn 1 (the arena's own segments)
    from . import track as trk
    arena = trk.make_arena()
    sp = turn1_span(arena)
    rep("turn 1 is the arena's first arc (169.3 .. 247.8 m)",
        sp is not None and abs(sp[0] - 169.267) < 0.01 and abs(sp[1] - 247.807) < 0.01, str(sp))
    m = {}
    _setup_turn1(arena, m)
    goal = m["goal"]
    clean = [_chk_turn1(_frame(s=s_), m) for s_ in (0.0, 100.0, 200.0, goal - 1.0, goal + 0.5)]
    rep("turn 1: a clean run from the line passes past its exit", clean[-1] and not any(clean[:-1]),
        _st_turn1(_frame(s=100.0), m))
    m = {}
    _setup_turn1(arena, m)
    r = [_chk_turn1(_frame(s=180.0), m), _chk_turn1(_frame(s=200.0, on_track=False), m),
         _chk_turn1(_frame(s=goal + 1.0), m)]
    off_msg = _st_turn1(_frame(s=goal + 1.0), m)
    r.append(_chk_turn1(_frame(s=0.2, cmds=["reset"]), m))           # R: back to the line
    r += [_chk_turn1(_frame(s=s_), m) for s_ in (100.0, goal + 2.0)]
    rep("turn 1: off the road does not pass; R and a clean run does",
        r == [False, False, False, False, False, True] and "R" in off_msg, f"{r} {off_msg!r}")
    m = {}
    _setup_turn1(arena, m)
    _chk_turn1(_frame(s=250.0, on_track=False), m)
    r = [_chk_turn1(_frame(s=390.7, cmds=["reset"]), m), _chk_turn1(_frame(s=420.0), m)]
    rep("turn 1: R to a sector line past the corner is not a go at it",
        r == [False, False] and "SHIFT+R" in _st_turn1(_frame(s=420.0), m), _st_turn1(_frame(s=420.0), m))
    # --- 4 reset
    rep("reset: the command passes, nothing else does",
        _chk_reset(_frame(cmds=["reset"]), {}) and not _chk_reset(_frame(V_kmh=80.0), {}))
    # --- 6 / 7 the wing lap (the 50 m skidpad: L = 314.16 m), line to line
    Ls = 2 * math.pi * 50.0
    m = {}
    _chk_wing_off(_frame(t=0.0, s=0.0, L=Ls, wing_on=True, track="skidpad"), m)
    why_on = m.get("why", "")
    got, t, s_ = False, 0.0, 0.0
    trace = []
    for k in range(1200):                  # 20 m/s, 0.8 g, 50 ms frames: 15.7 s a lap
        t += 0.05
        s_new = s_ + 1.0
        ev = []
        if s_new >= Ls:                    # the line
            s_new -= Ls
            ev = [("start", 0, t, float("nan"))] if not trace else \
                [("lap", len(trace), t, t - trace[-1])]
            trace.append(t)
        s_ = s_new
        got = _chk_wing_off(_frame(t=t, s=s_, L=Ls, ay_g=(0.8 if k % 2 else -0.8),
                                   events=ev, track="skidpad"), m)
        if got:
            break
    res = m.get("result", {})
    rep("wing lap: the wrong wing state is refused with the key to press",
        "F (CIRCLE)" in why_on and "switches it off" in why_on, why_on)
    rep("wing lap: out-lap, then a flying lap line to line -> mean |a_y| and the lap time",
        got and len(trace) == 2 and abs(res.get("ay", 0) - 0.8) < 1e-9
        and abs(res.get("t", 0) - (trace[1] - trace[0])) < 1e-9, f"{res} laps at {trace}")
    m = {}
    _chk_wing_on(_frame(t=0.0, s=0.0, L=Ls, wing_on=True, track="skidpad",
                        events=[("lap", 3, 0.0, 16.0)]), m)
    _chk_wing_on(_frame(t=0.05, s=1.0, L=Ls, wing_on=True, ay_g=0.9, track="skidpad"), m)
    d0 = m["dist"]
    _chk_wing_on(_frame(t=0.10, s=2.0, L=Ls, wing_on=True, on_track=False, track="skidpad"), m)
    spoiled = not m["clean"] and "does not count" in m["why"]
    for k in range(3, 320):
        _chk_wing_on(_frame(t=0.05 * k, s=float(k) % Ls, L=Ls, wing_on=True, ay_g=0.9,
                            track="skidpad"), m)
    no = _chk_wing_on(_frame(t=16.0, s=0.5, L=Ls, wing_on=True, track="skidpad",
                             events=[("lap", 4, 16.0, 16.0)]), m)
    _chk_wing_on(_frame(t=16.05, s=1.5, L=Ls, wing_on=False, track="skidpad"), m)
    wrong = not m["clean"] and "switches it on" in m["why"]
    rep("wing lap: off the circle spoils the lap (not passed at the line), so "
        "does the wrong wing", d0 == 1.0 and spoiled and not no and m["counting"] and wrong,
        f"{d0} {m['why']!r}")
    m = {}
    _chk_wing_on(_frame(t=0.0, s=0.0, L=Ls, wing_on=True, track="skidpad",
                        events=[("start", 0, 0.0, float("nan"))]), m)
    _chk_wing_on(_frame(t=0.05, s=200.0, L=Ls, wing_on=True, track="skidpad"), m)
    rep("wing lap: a teleport adds nothing", m["dist"] == 0.0 and m["counting"])
    m = {}
    _chk_wing_on(_frame(t=0.0, s=0.0, L=Ls, wing_on=True, track="skidpad",
                        events=[("start", 0, 0.0, float("nan"))]), m)
    _chk_wing_on(_frame(t=0.05, s=1.0, L=Ls, wing_on=True, wing_mode=2, track="skidpad"), m)
    rep("wing lap ON: the ALL 3 / LEFT / RIGHT mode spoils it, with the key back to AUTO",
        not m["clean"] and "G (TRIANGLE)" in m["why"], m["why"])
    # --- 10 the lap
    L = 1249.2
    m = {}
    r = [_chk_lap(_frame(s=5.0, events=[("start", 0, 1.0, float("nan"))]), m)]
    for k in range(1, 250):
        r.append(_chk_lap(_frame(s=5.0 + 5.0 * k, lap_time=0.1 * k), m))
    r.append(_chk_lap(_frame(s=3.0, events=[("lap", 1, 70.0, 61.2)]), m))
    rep("lap: a full valid lap passes, with its time", r[-1] and not any(r[:-1])
        and m["result"] == {"t": 61.2}, str(m.get("result")))
    m = {}
    _chk_lap(_frame(s=5.0, events=[("start", 0, 1.0, float("nan"))]), m)
    for k in range(1, 250):
        _chk_lap(_frame(s=5.0 + 5.0 * k), m)
    bad = _chk_lap(_frame(s=3.0, lap_valid=False, events=[("lap", 1, 70.0, 61.2)]), m)
    rep("lap: all four wheels off -> not passed, and why", not bad and "wheels" in m["why"], m["why"])
    m = {}
    _chk_lap(_frame(s=5.0, events=[("start", 0, 1.0, float("nan"))]), m)
    short = _chk_lap(_frame(s=4.0, events=[("lap", 1, 13.0, 12.0)]), m)   # reversed over the line
    rep("lap: a 'lap' that did not go round is not a lap", not short and "full lap" in m["why"],
        m["why"])
    m = {}
    _chk_lap(_frame(s=5.0, events=[("start", 0, 1.0, float("nan"))]), m)
    for k in range(1, 250):
        _chk_lap(_frame(s=5.0 + 5.0 * k), m)
    refused = _chk_lap(_frame(s=3.0, events=[("lap", 1, 70.0, 61.2)],
                              rec_why="the wet toggle (T) is on"), m)
    rep("lap: a lap the records refused does not pass, and says why",
        not refused and "not recorded (the wet toggle" in m["why"], m["why"])
    # --- 12 the manual gearbox
    m = {}
    seq = [(1, 0.0), (1, 30.0), (0, 40.0), (2, 45.0), (0, 70.0), (3, 72.0), (3, 90.0)]
    r = [_chk_manual(_frame(gear=g, V_kmh=v), m) for g, v in seq]
    up_st = _st_manual(_frame(gear=3, V_kmh=90.0), m)
    r += [_chk_manual(_frame(gear=0, V_kmh=60.0), m), _chk_manual(_frame(gear=2, V_kmh=55.0), m)]
    rep("manual: up to 3rd through neutral is not done, then 3 -> N -> 2 at speed is",
        not any(r[:-1]) and r[-1] and "down with Q" in up_st, up_st)
    m = {}
    for g, v in ((1, 10.0), (2, 50.0), (3, 80.0)):
        _chk_manual(_frame(gear=g, V_kmh=v), m)
    slow = _chk_manual(_frame(gear=1, V_kmh=0.0, cmds=["reset"]), m)      # R: stopped in 1st
    rep("manual: a reset (stopped, back in 1st) is not a downshift", not slow and m["top"] == 3)
    m = {}
    for g, v in ((1, 10.0), (2, 50.0), (3, 80.0)):
        _chk_manual(_frame(gear=g, V_kmh=v), m)
    r_fast = _chk_manual(_frame(gear=1, V_kmh=87.0, cmds=["reset"]), m)   # R mid-shift, at speed
    r_next = _chk_manual(_frame(gear=1, V_kmh=86.0), m)
    _chk_manual(_frame(gear=4, V_kmh=90.0, gearbox="auto"), m)             # the automatic's 4th
    back = [_chk_manual(_frame(gear=g, V_kmh=v), m) for g, v in ((4, 90.0), (3, 70.0))]
    rep("manual: R at speed in any gear, or a spell on the automatic, is no downshift",
        not r_fast and not r_next and not any(back) and m["top"] == 0, str(m))
    m = {}
    low = [_chk_manual(_frame(gear=g, V_kmh=v), m) for g, v in ((1, 20.0), (2, 45.0), (1, 35.0))]
    auto = _chk_manual(_frame(gear=3, V_kmh=90.0, gearbox="auto"), m)
    rep("manual: a downshift before 3rd is not it; on Automatic it says how to switch",
        not any(low) and not auto and "Settings > Gearbox" in m["why"], m["why"])
    # --- the steps are well formed
    ids = [s.id for s in STEPS]
    form = (len(set(ids)) == len(ids) == 13
            and all((s.kind == "drive") == (s.check is not None) for s in STEPS)
            and all(s.map in (None, "arena", "skidpad") for s in STEPS)
            and all(s.hint for s in STEPS if s.kind == "drive")
            and STEPS[-1].kind == "page"
            and [s.id for s in STEPS if s.group] == ["manual_intro", "manual"]
            and STEPS[ids.index("manual_intro")].kind == "page"
            and [s.id for s in STEPS if s.gearbox] == ["manual"])
    rep("13 steps, every drive step has a predicate and a hint; the manual pair "
        "is an optional group", form, " ".join(ids))

    # --- the state machine, on a fake sim, with a temporary progress file
    from .progress import Progress
    tmp = tempfile.mkdtemp(prefix="carsim_tutorial_")
    prog = Progress(os.path.join(tmp, "progress.json"))

    class _Lap:
        lap_valid, lap_time = True, 0.0

    class _Sim:
        def __init__(self, name, L_):
            self.track = SimpleNamespace(name=name, length=L_, closed=True, segs=arena.segs)
            self.veh = SimpleNamespace(u=0.0, v=0.0, ay=0.0, r=0.0,
                                       cfg=SimpleNamespace(wing="plate"))
            self.t, self.s, self.on_track, self.wing_on = 0.0, 0.0, True, False
            self.paused, self.events_log, self.lap, self.resets = False, [], _Lap(), 0

        def reset(self, to_checkpoint=False):
            self.resets += 1
            self.s = 0.0
            tut.command("reset")           # what drive.Sim.reset's hook does

    tut = Tutorial(prog)
    sk = _Sim("skidpad", Ls)
    rep("on the wrong map the tutorial asks for a restart", tut.tick(sk) == "restart"
        and tut.map_wanted() == "arena")
    ar = _Sim("arena", arena.length)
    first = tut.tick(ar)                   # step 1's setup: to the line
    rep("a drive step that starts on the line resets the car, once, and the "
        "reset is not the player's", first is None and ar.resets == 1 and tut._cmds == [])
    ar.veh.u = 60.0 / 3.6
    ar.t = 3.0
    tut.tick(ar)
    ar.veh.u, ar.t = 0.0, 8.0
    rep("pedals pass through the frame", tut.tick(ar) == "done" and tut.step.id == "steer",
        tut.step.id)
    rep("progress is saved at each step", Progress(prog.path).section(SECTION).get("step") == "steer")
    ov = tut.overlay()
    rep("the overlay: the step, the done flash, the keys",
        ov and "2/13" in ov["head"] and ov["flash"].startswith("done: Throttle") and "ESC" in ov["foot"],
        str(ov))
    tut.tick(ar)
    ar.t = 8.0 + HINT_AFTER_S + 1.0
    tut.tick(ar)
    rep("the hint shows after HINT_AFTER_S", tut.overlay()["hint"] == STEPS[1].hint)
    tut.skip()
    tut.skip()
    tut.skip()
    rep("skip moves on and is remembered; a page step asks for the page",
        tut.step.id == "assists" and tut.skipped == ["steer", "turn1", "reset"]
        and tut.tick(ar) == "page", f"{tut.step.id} {tut.skipped}")
    title, sub, note, secs, items = tut.page(SimpleNamespace(settings=SimpleNamespace(
        abs=True, tc=False, steer_aid=True, gearbox="auto"), key=None))
    rep("the assists page: four assists, Continue / End",
        len(secs[0][1]) == 4 and [a for _, a in items] == ["tut_next", "tut_end"]
        and "5/13" in title, title)
    tut.advance()
    rep("the wing step wants a flank wing and the skidpad", tut.wants_wing()
        and tut.tick(ar) == "restart")
    tut.end()
    sv = saved_state(Progress(prog.path))
    rep("End keeps the step for later", sv["step"] == "wing_off" and not sv["done"]
        and not tut.active and tut.tick(ar) is None, str(sv))
    rows = menu_items(None, Progress(prog.path))
    rep("the menu page offers to continue there",
        rows[0] == ("Continue at step 6: The wing, lap 1: OFF", "tut_resume")
        and "continue at step 6" in menu_row(Progress(prog.path)), str(rows[0]))
    rep("with a garage the page offers the wing-design tutorial too",
        "wt_garage" in [a for _, a in menu_items(None, Progress(prog.path), garage=True)]
        and "wt_garage" not in [a for _, a in rows])
    t2 = Tutorial(Progress(prog.path), start="wing_off")
    rep("... and a resumed tutorial starts there with the skipped steps kept",
        t2.step.id == "wing_off" and t2.skipped == ["steer", "turn1", "reset"])
    t2.results.update(wing_off=dict(ay=0.81, t=16.4), wing_on=dict(ay=0.88, t=15.9))
    t2.i = [s.id for s in STEPS].index("wing_result")
    note, secs = _pg_wing(None, t2)
    rep("the wing page shows both laps and the difference",
        "+0.07 g" in secs[0][1][2][1] and "OUTSIDE" in note and "more lateral g" in note,
        secs[0][1][2][1])
    t2.results["wing_on"] = dict(ay=0.82, t=16.3)
    rep("... and says 'about the same' inside WING_SAME_G, not a gain",
        "About the same" in _pg_wing(None, t2)[0])
    # the optional group: its page offers Skip, which skips both steps
    t2.i = [s.id for s in STEPS].index("manual_intro")
    t2._begin()
    ctx_ = SimpleNamespace(settings=SimpleNamespace(gearbox="auto"), key=None, garage=True)
    _t, _s, note_m, secs_m, items_m = t2.page(ctx_)
    rep("the manual page: optional, keys and how, Try / Skip / End",
        [a for _, a in items_m] == ["tut_next", "tut_skip_group", "tut_end"]
        and "Optional" in note_m and "Automatic after" in note_m
        and [t_ for t_, _ in secs_m] == ["THE KEYS", "HOW"], note_m[:60])
    t2.advance()
    rep("the manual drive step wants the manual box; a page wants none",
        t2.step.id == "manual" and t2.gearbox_wanted() == "manual")
    t2.set_gearbox_prev(("auto", "manual"))
    left = gearbox_left(Progress(prog.path))
    t2.set_gearbox_prev(None)
    rep("the switch is in the progress file while it lasts (a crash is undone at launch)",
        left == ("auto", "manual") and gearbox_left(Progress(prog.path)) is None, str(left))
    t2.i -= 1
    t2._begin()
    t2.skip(group=True)
    rep("Skip it on the page skips the whole group, to the done page",
        t2.step.id == "done" and t2.gearbox_wanted() is None
        and t2.skipped[-2:] == ["manual_intro", "manual"], f"{t2.step.id} {t2.skipped}")
    _t, _s, note_d, secs_d, items_d = t2.page(ctx_)
    no_g = t2.page(SimpleNamespace(settings=None, key=None, garage=False))[4]
    rep("the done page: the wing-design tutorial is a row with a garage; skipped "
        "steps by title", [a for _, a in items_d] == ["tut_next", "tut_wing"]
        and [a for _, a in no_g] == ["tut_next"]
        and "Manual gearbox (optional)" in secs_d[0][1][-1][1], str(secs_d[0][1][-1]))
    rep("the WELCOME offer: the wing-design tutorial with a garage",
        [a for _, a in offer_items(True)] == ["tut_start", "wt_garage", "tut_later"]
        and [a for _, a in OFFER_ITEMS] == ["tut_start", "tut_later"])
    t2.i = len(STEPS) - 1
    t2._begin()
    t2.advance()
    sv = saved_state(Progress(prog.path))
    rep("the last page finishes it: done, saved", t2.done and not t2.active and sv["done"]
        and menu_row(Progress(prog.path)) == "Tutorial: learn to drive (done)", str(sv))
    t3 = Tutorial(None)
    rep("a tutorial without a progress file runs (and saves nothing)",
        t3.step.id == "pedals" and t3.progress is None)
    # start over: a fresh run; a malformed saved result is dropped with a note
    pg = Progress(os.path.join(tmp, "p2.json"))
    pg.section(SECTION).update(step="wing_result", results={
        "wing_off": {"t": 16.0}, "wing_on": {"ay": "0.9", "t": 15.0}, "lap": {"t": 61.0}})
    t4 = Tutorial(pg, start="wing_result")
    rep("malformed saved results are dropped, the good one kept",
        t4.results == {"lap": {"t": 61.0}} and "not measured" in _pg_wing(None, t4)[1][0][1][0][1])
    t5 = Tutorial(pg)
    rep("start over is a fresh run: no old numbers", t5.results == {} and t5.skipped == [])
    # a restart: the new session's clock starts at 0, the flash still lasts DONE_FLASH_S
    t6 = Tutorial(None)
    s1 = _Sim("arena", arena.length)
    s1.t = 140.0
    t6.tick(s1)                            # setup at t = 140
    t6.flash, t6._flash_until = "done: Reset", 143.0
    s2 = _Sim("arena", arena.length)       # a restart: t = 0
    t6.tick(s2)
    on_now = bool(t6.overlay() and t6.overlay().get("flash"))
    s2.t = 3.5
    t6.tick(s2)
    rep("after a restart the flash lasts its 3 s, not the old session's clock",
        on_now and not (t6.overlay() or {}).get("flash"), f"{t6._flash_until}")
    # --- the wing car
    try:
        from .garage import CarBuild
        lib = SimpleNamespace(wings={"plate": SimpleNamespace(chord=0.30, role="flank")})
        g = SimpleNamespace(CarBuild=CarBuild)
        bare = CarBuild(name="my corsa")
        wc = wing_car(g, bare, lib)
        fitted = CarBuild.from_json(wc.to_json()) if wc else None
        rep("a car with no flank wing gets the plate for the wing laps",
            wc is not None and wc.left.wing == "plate" and wc.right.wing == "plate"
            and bare.left.wing == "" and "tutorial" in wc.name and fitted is not None)
        rep("a car with a flank wing keeps its own", wing_car(g, wc, lib) is None)
        #  fitted to the session's car (review of task 41, finding 7): on the
        #  bus the plate stays at the bus's 1.60 m flank slot and the top at
        #  3.30 m; without the car the Corsa's bands would pull them down
        bus = CarBuild.for_car("bus")
        wb, wn = wing_car(g, bus, lib, car="bus"), wing_car(g, bus, lib)
        rep("the wing car is fitted to the session's car",
            (wb.left.h, wb.top.h) == (1.60, 3.30) and wn.left.h == 1.20
            and (bus.left.h, bus.left.wing) == (1.60, ""),
            f"bus: flank h {wb.left.h:.2f}, top h {wb.top.h:.2f} (car-less: {wn.left.h:.2f})")
    except Exception as exc:               # noqa: BLE001
        rep("the wing car", False, f"{type(exc).__name__}: {exc}")
    if verbose:
        print(f"tutorial self-check: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
