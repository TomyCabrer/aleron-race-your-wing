"""drive/challenges.py -- challenges (task 25).

A challenge is a JSON file in `drive/data/challenges/` (kind
`carsim-challenge-1`):

    id, title, blurb,
    class        'track|car|engine|surface' (plan D1): the session it runs in
    constraints  what the BUILD may carry: max_wing_area (m^2 per slot),
                 max_wing_mass (kg, all wings fitted), max_ballast (kg),
                 slots (the slots allowed a wing), max_cda (m^2: the car's
                 own CdA plus every wing's DEPLOYED drag area)
    goal         {metric, threshold (= 1 star), + the metric's parameter}
    stars        {"2": {threshold}, "3": {threshold, efficiency: {<a
                 constraint key>: bound}}} -- 2 stars = a tighter number, 3 =
                 tighter still AND the efficiency bound
    ref          the measured run every threshold is derived from: the
                 driver, the build, the value; x = the multipliers

THRESHOLDS ARE DERIVED, NEVER HAND-TYPED (the medals' rule, D4, and the
same multipliers): the reference run -- a scripted driver, headless, on the
reference build, in the challenge's class -- measures a value V, and 1 / 2 /
3 stars are V x 1.12 / 1.06 / 1.02 for a lower-is-better metric and V /
1.12 / 1.06 / 1.02 for a higher-is-better one. So the reference run earns
three stars by 2 % by construction, and every threshold is a measured
number. `python3 -m drive.challenges --measure` runs the references and
prints them; `--write` puts them (and the thresholds) into the files.
drive.py's V35 runs every reference again and checks it earns 3 stars on a
build that meets the challenge's constraints AND its 3-star bound -- and
that the thresholds in the file are the ones that run derives.

Metrics, measured by the SAME per-step meter live and headless (`Meter`,
attached as `Sim.challenge` through three hooks: after every physics step,
every LapTimer event, every reset):

  lap_time       s, lower   a valid lap that went round (95 % of the length
                            counted from the crossing -- the records' rule)
  stop_distance  m, lower   from the moment the car, having been faster,
                            slows through v0 to under 0.1 m/s (dragstrip).
                            The car starts ROLLING at goal.start_kmh, and
                            every reset (R, SHIFT+R) puts it back there
  skid_ay        g, higher  mean |lateral g| over a flying lap of the
                            skidpad, line to line, on the road throughout
  drag_time      s, lower   from standing (the car first moves) to
                            distance_m down the dragstrip
  trap_speed     km/h, hi   the speed at distance_m, same run

The constraints are checked before a challenge starts (the CHALLENGES page
and again at the session's start); a refusal says what is wrong and what to
do about it, the garage's gating style. Progress (best value, stars) is the
`challenges` section of `runs/progress.json` (drive/progress.py). Only a
player session opens it.
"""
from __future__ import annotations

import json
import math
import os

from corsa_c import G

KIND = "carsim-challenge-1"
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "challenges")
SECTION = "challenges"
#: 1 / 2 / 3 stars: the medal multipliers (plan D4; drive.medals)
STAR_X = (1.12, 1.06, 1.02)
METRICS = {
    "lap_time": dict(unit="s", lower=True, tracks=("arena", "open", "skidpad")),
    "stop_distance": dict(unit="m", lower=True, tracks=("dragstrip",), param="v0_kmh"),
    "skid_ay": dict(unit="g", lower=False, tracks=("skidpad",)),
    "drag_time": dict(unit="s", lower=True, tracks=("dragstrip",), param="distance_m"),
    "trap_speed": dict(unit="km/h", lower=False, tracks=("dragstrip",), param="distance_m"),
}
CONSTRAINT_KEYS = ("max_wing_area", "max_wing_mass", "max_ballast", "slots", "max_cda")
SLOT_NAMES = ("left", "right", "top")
LAP_MIN_FRACTION = 0.95
STOP_V = 0.10                  # m/s: stopped (BrakeDriver's own threshold)
MOVE_V = 0.10                  # m/s: a drag run's clock starts
STILL_V = 0.05                 # m/s: standing, ready for a drag run
MAX_DS = 10.0                  # m a step: more is a teleport (LapTimer's guard)
START_S = 2.5                  # m down the open strip: a stop's rolling start (drive._build's)
T_MAX = {"lap_time": 260.0, "skid_ay": 120.0, "stop_distance": 60.0,
         "drag_time": 90.0, "trap_speed": 90.0}


# ==================================================================== #
#  FORMAT                                                              #
# ==================================================================== #
def fmt_value(metric: str, v) -> str:
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return "--"
    if metric == "lap_time":
        from .records import fmt_time
        return fmt_time(v)
    unit = METRICS[metric]["unit"]
    return {"m": f"{v:.2f} m", "g": f"{v:.3f} g", "s": f"{v:.3f} s",
            "km/h": f"{v:.1f} km/h"}[unit]


def stars_text(n: int) -> str:
    return "*" * int(n) + "-" * (3 - int(n))


# ==================================================================== #
#  THE FILES                                                           #
# ==================================================================== #
def validate(d: dict) -> list:
    """Why this challenge file is not usable ([] = fine)."""
    from .records import split_key
    bad = []
    if not isinstance(d, dict) or d.get("kind") != KIND:
        return [f"kind is not {KIND!r}"]
    for k in ("id", "title", "blurb", "class", "constraints", "goal", "stars", "ref"):
        if k not in d:
            bad.append(f"no {k!r}")
    if bad:
        return bad
    try:
        track, car, engine, surface = split_key(d["class"])
    except ValueError as exc:
        return [str(exc)]
    import cars
    from . import track as trk
    from .drive import ENGINE_MODES, SURFACE_MODES
    if track not in trk.TRACK_ORDER:
        bad.append(f"no map {track!r}")
    if car not in cars.CAR_ORDER:
        bad.append(f"no car {car!r}")
    if engine not in ENGINE_MODES:
        bad.append(f"no engine {engine!r}")
    if surface not in SURFACE_MODES:
        bad.append(f"no surface {surface!r}")
    g = d["goal"]
    met = METRICS.get(g.get("metric"))
    if met is None:
        return bad + [f"no metric {g.get('metric')!r}"]
    if track not in met["tracks"]:
        bad.append(f"{g['metric']} is not measured on the {track}")
    if met.get("param") and not _num(g.get(met["param"])):
        bad.append(f"{g['metric']} needs {met['param']}")
    elif g["metric"] == "stop_distance" and not (
            _num(g.get("start_kmh")) and g["v0_kmh"] < g["start_kmh"] <= 2.0 * g["v0_kmh"]):
        bad.append("stop_distance needs start_kmh, above v0_kmh and at most twice it: "
                   "the car starts rolling there")
    for k, v in d["constraints"].items():
        if k not in CONSTRAINT_KEYS:
            bad.append(f"unknown constraint {k!r}")
        elif k == "slots":
            if not isinstance(v, list) or any(s not in SLOT_NAMES for s in v):
                bad.append("slots must be a list of left / right / top")
        elif not _num(v) or v < 0:
            bad.append(f"{k} must be a number >= 0")
    st = d["stars"]
    t1 = g.get("threshold")
    t2 = (st.get("2") or {}).get("threshold")
    t3 = (st.get("3") or {}).get("threshold")
    eff = (st.get("3") or {}).get("efficiency")
    if not all(_num(t) for t in (t1, t2, t3)):
        bad.append("the three thresholds must be numbers (python3 -m drive.challenges --write)")
    else:
        order = (t1 > t2 > t3) if met["lower"] else (t1 < t2 < t3)
        if not order:
            bad.append("the stars must tighten: 1 < 2 < 3")
    if not isinstance(eff, dict) or not eff or any(k not in CONSTRAINT_KEYS or k == "slots"
                                                   for k in eff):
        bad.append("3 stars needs an efficiency bound (a numeric constraint key)")
    ref = d["ref"]
    if not isinstance(ref, dict) or not isinstance(ref.get("driver"), str):
        bad.append("ref needs a driver")
    elif ref.get("wing_mode", "auto") not in REF_WING_MODES:
        bad.append(f"ref.wing_mode must be one of {sorted(REF_WING_MODES)}")
    elif not (isinstance(ref.get("build", "none"), dict)
              or ref.get("build", "none") in REF_BUILDS):
        bad.append(f"ref.build must be one of {list(REF_BUILDS)} or a build JSON")
    elif _num(ref.get("value")) and all(_num(t) for t in (t1, t2, t3)):
        want = derived(g["metric"], ref["value"])
        if any(abs(a - b) > 1e-9 * max(1.0, abs(b)) for a, b in zip((t1, t2, t3), want)):
            bad.append("the thresholds are not the ones ref.value derives (--write)")
    return bad


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def derived(metric: str, value: float) -> tuple:
    """(1-star, 2-star, 3-star) thresholds from a measured reference value."""
    if METRICS[metric]["lower"]:
        return tuple(value * x for x in STAR_X)
    return tuple(value / x for x in STAR_X)


def load_all(folder: str = DIR) -> dict:
    """{id: challenge} for every valid file, in file-name order; a bad file is
    skipped with a printed note, never a crash."""
    out = {}
    try:
        names = sorted(f for f in os.listdir(folder) if f.endswith(".json"))
    except OSError:
        return out
    for f in names:
        p = os.path.join(folder, f)
        try:
            with open(p) as fh:
                d = json.load(fh)
        except (OSError, ValueError) as exc:
            print(f"challenges: {f} ignored ({exc.__class__.__name__})")
            continue
        why = validate(d)
        if why:
            print(f"challenges: {f} ignored ({'; '.join(why)})")
            continue
        out[d["id"]] = d
    return out


def thresholds(ch: dict) -> tuple:
    st = ch["stars"]
    return (float(ch["goal"]["threshold"]), float(st["2"]["threshold"]),
            float(st["3"]["threshold"]))


# ==================================================================== #
#  THE BUILD: what the constraints read                                #
# ==================================================================== #
def build_stats(build_json, lib, car, ballast_kg: float) -> dict:
    """What a build carries, as the constraints read it: the wing area in each
    slot (m^2, `WingSpec.S`), the fitted wings' mass (kg, the garage's own
    `mass_points`), the ballast (kg), the slots with a wing, and the drag
    area with everything deployed (m^2: the car's CdA + the top wing's CD*S +
    EACH fitted flank panel's own D/q: G can put both out)."""
    from .garage import CarBuild, SLOTS
    from .aero.wing import design_point, V_REF
    b = CarBuild.from_json(build_json or {}).clamp(lib)
    area = {}
    for k in SLOTS:
        w = lib.wings.get(b.slot(k).wing)
        area[k] = float(w.S) if (w is not None and b.slot(k).wing) else 0.0
    mass = float(sum(pm.m for pm in b.mass_points(lib)))
    slots = [k for k in SLOTS if b.slot(k).wing and area.get(k, 0.0) > 0.0]
    ma = b.mission_aero(lib)               # the top wing's CD*S, deployed
    flank_cda = 0.0
    for k in ("left", "right"):            # EVERY flank panel out (G = both): each its own D/q
        s_ = b.slot(k)
        w = lib.wings.get(s_.wing) if s_.wing else None
        if w is None:
            continue
        lib.analyse_wing(w)
        dp = design_point(w, s_.inc_deg, V=V_REF["flank"], x_w=s_.x)
        if dp and dp.get("q", 0.0) > 0.0 and dp.get("D", 0.0) > 0.0:
            flank_cda += dp["D"] / dp["q"]
    cda = float(car.CdA) + float(ma.cd_a) + float(flank_cda)
    return dict(area=area, mass=mass, ballast=float(ballast_kg or 0.0), slots=slots, cda=cda)


def refusals(bounds: dict, stats: dict) -> list:
    """Why a build may not start (or earn the 3-star bound): [] = it may.
    Each reason says what is wrong and what to do, the garage's style."""
    out = []
    for k, v in bounds.items():
        if k == "slots":
            extra = [s for s in stats["slots"] if s not in v]
            if extra:
                out.append(f"the {' and '.join(extra)} wing{'s are' if len(extra) > 1 else ' is'} "
                           f"not allowed here: take {'them' if len(extra) > 1 else 'it'} off "
                           f"in the garage (BACKSPACE), or pick a build without")
        elif k == "max_wing_area":
            big = {s: a for s, a in stats["area"].items() if a > v + 1e-9}
            if big:
                s, a = max(big.items(), key=lambda kv: kv[1])
                out.append(f"the {s} wing is {a:.3f} m^2, over the {v:.3f} m^2 a slot may "
                           f"carry: a smaller wing (garage > design box: span x chord)")
        elif k == "max_wing_mass" and stats["mass"] > v + 1e-9:
            out.append(f"the wings weigh {stats['mass']:.1f} kg, over {v:.1f} kg: fewer or "
                       f"smaller wings")
        elif k == "max_ballast" and stats["ballast"] > v + 1e-9:
            out.append(f"{stats['ballast']:.0f} kg of ballast, over {v:.0f} kg: ESC > "
                       f"Settings > Ballast")
        elif k == "max_cda" and stats["cda"] > v + 1e-9:
            out.append(f"the car's drag area with every wing out is {stats['cda']:.3f} m^2, "
                       f"over {v:.3f} m^2: less wing, or a lower-drag one")
    return out


def stars_for(ch: dict, value, stats: dict) -> int:
    """0..3 for a measured `value` driven on a build with `stats`."""
    if not _num(value):
        return 0
    lower = METRICS[ch["goal"]["metric"]]["lower"]
    try:
        t1, t2, t3 = thresholds(ch)
    except (TypeError, ValueError, KeyError):
        return 0                           # not measured yet (--write)
    beat = (lambda t: value <= t) if lower else (lambda t: value >= t)
    if not beat(t1):
        return 0
    if not beat(t2):
        return 1
    if beat(t3) and not refusals(ch["stars"]["3"]["efficiency"], stats):
        return 3
    return 2


def better(metric: str, a, b) -> bool:
    """Is `a` a better result than `b` (None = no result)?"""
    if not _num(a):
        return False
    if not _num(b):
        return True
    return a < b if METRICS[metric]["lower"] else a > b


# ==================================================================== #
#  THE METER: one per attempt series, the same live and headless      #
# ==================================================================== #
class Meter:
    """Watches a Sim through the three hooks (`step` after every physics
    step, `event` per LapTimer event, `reset`) and produces `result` (a
    float in the metric's unit) when an attempt completes; it re-arms by
    itself for the next one."""

    def __init__(self, goal: dict, track):
        self.metric = goal["metric"]
        self.goal = goal
        self.L = float(track.length)
        self.closed = bool(track.closed)
        self.width = float(track.width)
        self.result = None             # the last completed attempt
        self.results: list = []
        self.why = ""                  # why the last attempt did not count
        self._s = None
        self._V = 0.0
        self.reset()

    def reset(self) -> None:
        self.counting = False          # a lap / a stop / a run is in progress
        self.dist = 0.0
        self.ay_dt = 0.0
        self.tt = 0.0
        self.clean = True
        self.armed = False
        self.t0 = None
        self.s0 = None
        self._s = None

    # -- the hooks ----------------------------------------------------------
    def event(self, sim, e) -> None:
        if self.metric not in ("lap_time", "skid_ay") or e[0] not in ("start", "lap"):
            return
        if e[0] == "lap" and self.counting:
            full = self.dist >= LAP_MIN_FRACTION * self.L
            if not full:
                self.why = "not a full lap"
            elif not sim.lap.lap_valid:
                self.why = "all four wheels left the road"
            elif self.metric == "skid_ay" and not self.clean:
                self.why = "off the circle"
            else:
                val = float(e[3]) if self.metric == "lap_time" else (
                    self.ay_dt / self.tt / G if self.tt > 0.0 else float("nan"))
                self._done(val)
        self.counting, self.dist, self.ay_dt, self.tt, self.clean = True, 0.0, 0.0, 0.0, True

    def step(self, sim) -> None:
        m = self.metric
        s = float(sim.s)
        s0, self._s = self._s, s
        v = sim.veh
        V = math.hypot(v.u, v.v)
        V0, self._V = self._V, V
        dt = sim.dt
        if m in ("lap_time", "skid_ay"):
            if s0 is None or not self.counting:
                return
            ds = s - s0
            if self.closed:
                ds = (ds + 0.5 * self.L) % self.L - 0.5 * self.L
            if abs(ds) <= MAX_DS and abs(sim.n_lat) <= 0.5 * self.width + 2.0:
                self.dist += ds
            if m == "skid_ay":
                if not sim.on_track:
                    self.clean = False
                self.ay_dt += abs(float(v.ay)) * dt
                self.tt += dt
        elif m == "stop_distance":
            vt = float(self.goal["v0_kmh"]) / 3.6
            if V >= vt:
                self.armed, self.counting, self.dist = True, False, 0.0
            elif self.armed and not self.counting and V0 >= vt > V:
                self.counting = True       # the part of this step after the crossing
                f = (V0 - vt) / max(V0 - V, 1e-12)
                self.dist = 0.5 * (vt + V) * dt * (1.0 - f)
            elif self.counting:
                self.dist += 0.5 * (V0 + V) * dt
                if V < STOP_V:
                    self.armed = self.counting = False
                    self._done(self.dist)
        else:                              # drag_time / trap_speed: a standing run
            D = float(self.goal["distance_m"])
            if not self.counting:
                if V < STILL_V:
                    self.armed, self.s0 = True, s
                elif self.armed and V >= MOVE_V:
                    self.counting, self.t0 = True, float(sim.t) - dt
            elif s0 is not None and s < s0 - 1e-3:
                self.armed = self.counting = False      # backwards: not a standing run
                self.why = "went backwards: stop, then launch"
            elif s0 is not None and s >= self.s0 + D > s0:
                f = (self.s0 + D - s0) / max(s - s0, 1e-12)
                t_fin = float(sim.t) - dt + f * dt
                trap = (V0 + f * (V - V0)) * 3.6
                self.armed = self.counting = False
                self._done(t_fin - self.t0 if m == "drag_time" else trap)

    def _done(self, val: float) -> None:
        self.result = float(val)
        self.results.append(self.result)
        self.why = ""

    # -- the box ------------------------------------------------------------
    def status(self, sim) -> str:
        m = self.metric
        V = math.hypot(sim.veh.u, sim.veh.v) * 3.6
        if m in ("lap_time", "skid_ay"):
            if not self.counting:
                return "to the line: the attempt starts there"
            pct = 100.0 * max(self.dist, 0.0) / max(self.L, 1.0)
            if m == "skid_ay":
                ay = self.ay_dt / self.tt / G if self.tt > 0 else 0.0
                return f"lap {pct:3.0f} %   mean {ay:.3f} g" + ("" if self.clean else "   OFF")
            return f"lap {pct:3.0f} %   {sim.lap.lap_time:6.1f} s"
        if m == "stop_distance":
            v0 = self.goal["v0_kmh"]
            if self.counting:
                return f"stopping: {self.dist:5.1f} m   {V:3.0f} km/h"
            return (f"brake now: {V:3.0f} km/h" if self.armed
                    else f"accelerate past {v0:.0f} km/h ({V:3.0f}), then brake to a stop")
        D = self.goal["distance_m"]
        if self.counting:
            return f"{float(sim.s) - self.s0:5.0f} of {D:.0f} m   {V:3.0f} km/h"
        return ("GO: the clock starts when the car moves" if self.armed
                else "stop (SHIFT+R puts you on the line): a run starts from standing")


class ChallengeRun:
    """A challenge being driven: the meter, the build's stats, the best and
    the stars, the progress file. `Sim.challenge`."""

    def __init__(self, ch: dict, track, stats: dict, progress=None):
        self.ch = ch
        self.stats = stats
        self.meter = Meter(ch["goal"], track)
        self.progress = progress
        self.best, self.stars = _best(progress, ch["id"])
        self.refused = ""                  # the build breaks a rule: listed, endable, never counted
        from .records import split_key
        from .track import MU_WET_SCALE
        self._cls = split_key(ch["class"])
        self._gw = MU_WET_SCALE if self._cls[3] == "all" else 1.0
        self.last = None               # (value, stars) of the last attempt
        self._counted = False          # ... and whether it counted (the box's LAST STOP)
        self.note = ""
        self._seen = 0

    # the Sim's hooks
    def _void(self, sim) -> str:
        """Why the attempt in progress cannot count, checked EVERY step: out
        of the class (a live engine change, the T toggle -- even one undone
        before the finish), slow motion or single steps (the records' rules)."""
        s = sim.settings
        if (sim.track.name, s.car, s.engine, s.wet) != self._cls:
            return "the map, car, engine or surface is not the challenge's"
        if abs(float(sim.global_wet) - self._gw) > 1e-12:
            return "the wet toggle (T) is on"
        if getattr(sim, "time_scale", 1.0) != 1.0:
            return "slow motion"
        if getattr(sim, "paused", False):
            return "single-stepped"
        return ""

    def step(self, sim) -> None:
        why = self._void(sim)
        if why:
            if self.meter.counting or self.meter.armed:
                self.meter.why = why
            self.meter.reset()             # the attempt in progress is gone
            return
        self.meter.step(sim)
        if len(self.meter.results) != self._seen:
            self._collect(sim)

    def event(self, sim, e) -> None:
        if self._void(sim):
            return                         # step() drops the attempt
        self.meter.event(sim, e)
        if len(self.meter.results) != self._seen:
            self._collect(sim)

    def reset(self) -> None:
        self.meter.reset()

    def rolling(self, tr, pt_p):
        """(s0, V0, gear) every reset puts the car at in a stop challenge
        (task 40): rolling at the goal's start_kmh -- above v0, so the brake
        is full before the distance counts -- START_S down the open strip, in
        the gear the automatic holds there on full throttle. None = a
        standing start (every other metric)."""
        g = self.ch["goal"]
        if g["metric"] != "stop_distance" or not _num(g.get("start_kmh")):
            return None
        from . import powertrain as ptm
        V0 = float(g["start_kmh"]) / 3.6
        gear = next((k for k in range(1, len(pt_p.gear))
                     if ptm.rpm_at_speed(pt_p, k, V0) < ptm.n_up_schedule(pt_p, k, 1.0)),
                    len(pt_p.gear))
        return (0.0 if tr.closed else START_S), V0, gear

    def class_why(self, sim) -> str:
        """'' while the session is still in the challenge's class, else why
        not (a live engine change, the T wet toggle ...): such a result is
        not the challenge's."""
        if self.refused:
            return f"the build breaks a rule ({self.refused})"
        s = sim.settings
        if (sim.track.name, s.car, s.engine, s.wet) != self._cls:
            return "the map, car, engine or surface is not the challenge's"
        if abs(float(sim.global_wet) - self._gw) > 1e-12:
            return "the wet toggle (T) is on"
        return ""

    def _collect(self, sim) -> None:
        self._seen = len(self.meter.results)
        v = self.meter.results[-1]
        why = self.class_why(sim)
        if why:
            self.last, self._counted = (v, 0), False
            self.note = f"{self.ch['title']}: not counted -- {why}"
            return
        n = stars_for(self.ch, v, self.stats)
        metric = self.ch["goal"]["metric"]
        new_best = better(metric, v, self.best)
        self.last, self._counted = (v, n), True
        self.note = (f"{self.ch['title']}: {fmt_value(metric, v)}  [{stars_text(n)}]"
                     + ("  NEW BEST" if new_best else ""))
        if new_best or n > self.stars:
            if new_best:
                self.best = v
            self.stars = max(self.stars, n)
            if self.progress is not None:
                sec = self.progress.section(SECTION)
                sec[self.ch["id"]] = dict(best=self.best, stars=self.stars)
                self.progress.save(SECTION)

    def _status(self, sim) -> str:
        """The box's live line, in the HUD's number font. A stop keeps its
        LAST distance there until the next stop starts counting (task 40):
        the green line above it is small, and the car stands still after."""
        m, g = self.meter, self.ch["goal"]
        if g["metric"] != "stop_distance" or m.counting:
            return m.status(sim)
        last = ""
        if self.last is not None:
            v, n = self.last
            last = (f"LAST STOP {fmt_value('stop_distance', v)} "
                    + (f"[{stars_text(n)}]" if self._counted else "(not counted)") + "   ")
        why = self.class_why(sim) or (
            "slow motion" if getattr(sim, "time_scale", 1.0) != 1.0 else "")
        if why:                            # the attempt is void every step (step())
            return last + f"not counting: {why}"
        V = math.hypot(sim.veh.u, sim.veh.v) * 3.6
        if m.armed or V >= float(g["v0_kmh"]):
            return last + f"brake now: {V:3.0f} km/h"
        if _num(g.get("start_kmh")) and not getattr(sim, "rivals", None):
            return last + f"R: again from {g['start_kmh']:.0f} km/h"
        return last + m.status(sim)       # a race resets to the line, standing

    def overlay(self, sim) -> dict:
        """The box (render._draw_tutorial's dict)."""
        ch = self.ch
        metric = ch["goal"]["metric"]
        t1, t2, t3 = thresholds(ch)
        eff = "; ".join(f"{_bound_text(k, v)}" for k, v in ch["stars"]["3"]["efficiency"].items())
        text = (f"{ch['blurb']}  Stars: 1 {fmt_value(metric, t1)}, 2 {fmt_value(metric, t2)}, "
                f"3 {fmt_value(metric, t3)} with {eff}.")
        best = (f"   best {fmt_value(metric, self.best)} [{stars_text(self.stars)}]"
                if self.best is not None else "")
        return dict(head=f"CHALLENGE  {ch['title']}{best}", text=text,
                    status=self._status(sim),
                    warn=(f"did not count: {self.meter.why}" if self.meter.why else ""),
                    hint="", flash=self.note,
                    foot="ESC > Challenges: end it, or another one")


def _bound_text(k: str, v) -> str:
    return {"max_wing_area": f"at most {v:.2f} m^2 of wing a slot",
            "max_wing_mass": (f"at most {v:.1f} kg of wing" if v > 0 else "no wings"),
            "max_ballast": (f"at most {v:.0f} kg of ballast" if v > 0 else "no ballast"),
            "max_cda": f"a drag area of at most {v:.3f} m^2"}.get(k, f"{k} {v}")


def constraints_text(ch: dict) -> list:
    rows = []
    for k, v in ch["constraints"].items():
        rows.append(("allowed" if k == "slots" else "rule",
                     ("wings in " + ", ".join(v)) if k == "slots" else _bound_text(k, v)))
    return rows


# ==================================================================== #
#  THE PAGES (drive.py's Sim shows them; keyboard, pad, mouse)          #
# ==================================================================== #
LIST_HELP = [("CHALLENGES", [
    ("stars", "1 = done, 2 = a tighter number, 3 = tighter still AND"),
    ("", "the efficiency rule (less wing, less drag, no ballast...)"),
    ("rules", "what the build may carry; checked before the start"),
    ("class", "each runs in its own map, car, engine and surface;"),
    ("", "your settings come back when you end it"),
    ("R", "a reset starts the attempt again"),
])]


def _best(progress, cid):
    """(best, stars) saved for a challenge; a malformed entry (a hand edit, a
    future format) is ignored with a note, never a crash."""
    e = progress.section(SECTION).get(cid) if progress is not None else None
    if e is None:
        return None, 0
    n = e.get("stars", 0) if isinstance(e, dict) else None
    if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n <= 3:
        note = f"progress.json: challenges.{cid} ignored (malformed)"
        if note not in progress.notes:
            progress.notes.append(note)
            print(note)
        return None, 0
    b = e.get("best")
    return (b if _num(b) else None), n


def total_stars(progress, allc=None) -> tuple:
    allc = load_all() if allc is None else allc
    return sum(_best(progress, c)[1] for c in allc), 3 * len(allc)


def menu_row(progress, run=None) -> str:
    """The pause page's row."""
    if run is not None:
        return f"Challenges: {run.ch['title']} running -- end it, or another"
    got, of = total_stars(progress)
    return f"Challenges: {got} of {of} stars"


def list_items(allc, progress, run=None) -> list:
    rows = []
    for cid, ch in allc.items():
        b, n = _best(progress, cid)
        mark = "  <- now" if (run is not None and run.ch["id"] == cid) else ""
        rows.append((f"{ch['title'][:24]:<24s} [{stars_text(n)}] "
                     f"{fmt_value(ch['goal']['metric'], b) if b is not None else '':>11s}{mark}",
                     f"ch:{cid}"))
    if run is not None:
        rows.append(("End the challenge (your own map / car / engine / surface back)", "ch_end"))
    rows.append(("Back", "ch_back"))
    return rows


def class_text(ch) -> str:
    from .records import class_label
    try:
        return class_label(ch["class"])
    except Exception:                      # noqa: BLE001
        return ch["class"]


def detail(ch, stats, why, progress) -> tuple:
    """(items, sections, note) of one challenge's page."""
    metric = ch["goal"]["metric"]
    t1, t2, t3 = thresholds(ch)
    eff = ch["stars"]["3"]["efficiency"]
    goal = [("1 star", fmt_value(metric, t1)), ("2 stars", fmt_value(metric, t2)),
            ("3 stars", f"{fmt_value(metric, t3)} with "
                        + "; ".join(_bound_text(k, v) for k, v in eff.items()))]
    rules = constraints_text(ch) or [("rules", "none: any build")]
    if stats is not None:
        rules.append(("your build", f"{stats['mass']:.1f} kg of wing, drag area "
                                    f"{stats['cda']:.3f} m^2, wings in "
                                    + (", ".join(stats['slots']) or "no slot")))
    b, n = _best(progress, ch["id"])
    mine = [("best", (f"{fmt_value(metric, b)}  [{stars_text(n)}]" if b is not None
                      else "none yet"))]
    secs = [("GOAL: " + {"lap_time": "a lap", "stop_distance": "a stop",
                         "skid_ay": "lateral g", "drag_time": "the time",
                         "trap_speed": "the speed"}[metric].upper(), goal),
            ("RULES", rules), ("YOURS", mine)]
    if why:
        note = "CANNOT START: " + " -- and ".join(why)
        items = [("Cannot start: change the build first (see below)", "ch_list"),
                 ("Back", "ch_list")]
    else:
        note = ch["blurb"] + " Your map, car, engine and surface come back when you end it."
        items = [("Start", f"ch_go:{ch['id']}"), ("Back", "ch_list")]
    return items, secs, note


# ==================================================================== #
#  THE REFERENCE RUNS                                                  #
# ==================================================================== #
def ref_build(name) -> dict:
    """The reference builds: 'none' (no wings), 'plate' (the published plate
    on both flanks, as the tutorial fits it), 'tall' (the plate on both
    flanks and the rear-s1223 top wing: every wing an air brake has; task
    35), or a build JSON itself."""
    if isinstance(name, dict):
        return dict(name)
    base = dict(version=2, name=f"ref-{name}", mirror=True, builtin=False,
                slots={"left": {"wing": "", "x": 0.97, "h": 0.9, "inc_deg": 0.0},
                       "top": {"wing": ""}})
    if name in ("plate", "tall"):
        base["slots"]["left"]["wing"] = "plate"
    if name == "tall":
        base["slots"]["top"] = {"wing": "rear-s1223", "x": -0.9, "h": 1.55, "inc_deg": 6.0}
    return base


#: a reference's wing mode (`ref.wing_mode`; drive/airbrake.py): the G key's
#: mode the reference drives with, as a player would set it
REF_WING_MODES = {"auto": 0, "air_brake": 3, "all": 2}
REF_BUILDS = ("none", "plate", "tall")


def _build_name(b) -> str:
    return b if isinstance(b, str) else str((b or {}).get("name", "build"))


def ref_driver(spec: str, tr, car, gw: float):
    """'lapdriver:<margin>' | 'brake:<pedal>:<v0 km/h>' | 'straight:<throttle>'."""
    from .drive import LapDriver, BrakeDriver, StraightDriver
    kind, _, arg = spec.partition(":")
    if kind == "lapdriver":
        return LapDriver(tr, margin=float(arg), global_wet=gw, car=car)
    if kind == "brake":
        pedal, _, v0 = arg.partition(":")
        return BrakeDriver(v_trigger=float(v0) / 3.6, pedal=float(pedal))
    if kind == "straight":
        return StraightDriver(throttle=float(arg))
    raise ValueError(f"no reference driver {spec!r}")


def measure(ch: dict, lib, t_max: float | None = None, probe=None) -> dict:
    """Run the challenge's reference, headless, exactly as a session builds
    the car (`drive._session_car`), with the meter attached as a session's
    is, on the wing mode `ref.wing_mode` (a player's G; AUTO when absent).
    `probe(sim)` after every step (V38). A stop challenge starts rolling,
    through the Sim's own reset (task 40). {value, stars, stats, t_sim,
    start: (s, V, gear) at the start}."""
    from types import SimpleNamespace
    from .records import split_key
    from . import track as trk
    from . import drive as D
    from .garage import CarBuild
    from .vehicle import Vehicle, VehicleConfig
    track, car_name, engine, surface = split_key(ch["class"])
    ref = ch["ref"]
    settings = D.Settings(path="", track=track, car=car_name, engine=engine, wet=surface,
                          abs=bool(ref.get("abs", True)), tc=bool(ref.get("tc", True)))
    opts = SimpleNamespace(track=track, radius=50.0, cw=False, wet=surface, dt=D.DT_PHYS,
                           wing="off", wing_x=0.97, wing_h=0.90, wing_inc=0.0,
                           dev_flank="outer", wing_cfg=None, mass_points=())
    design = CarBuild.from_json(ref_build(ref.get("build", "none"))).clamp(lib)
    D._apply_design(opts, design, lib)
    tr, car, cfg_kwargs, gw = D._session_car(opts, settings)
    stats = build_stats(design.to_json(), lib, car, 0.0)
    cfg = VehicleConfig(**cfg_kwargs)
    veh = Vehicle(car, cfg)
    x, y, psi = trk.start_pose(tr, 0.0)
    veh.reset(x, y, psi, V=0.0, gear=1)
    drv = ref_driver(ref["driver"], tr, car, gw)
    sim = D.Sim(veh, tr, D.ScriptedInput(drv), dt=D.DT_PHYS, wing=opts.wing,
                global_wet=gw, settings=settings)
    if cfg.has_designed():
        sim.wing_on = True                 # a garage build starts armed, as a session's does
    sim.wing_side_mode = REF_WING_MODES[ref.get("wing_mode", "auto")]
    run = ChallengeRun(ch, tr, stats)
    sim.challenge = run
    if run.rolling(tr, veh.pt_p) is not None:
        sim.reset()                        # a stop rolls from start_kmh, as a session's does
    start = (float(sim.s), math.hypot(veh.u, veh.v), int(veh.gear))
    n = int(round((t_max or T_MAX[ch["goal"]["metric"]]) / sim.dt))
    for _ in range(n):
        sim.step_physics(sim.dt)
        if probe is not None:
            probe(sim)
        if run.last is not None:
            break
    v = run.last[0] if run.last else float("nan")
    return dict(value=v, stars=(run.last[1] if run.last else 0), stats=stats, t_sim=sim.t,
                start=start,
                refusals=refusals(ch["constraints"], stats)
                + refusals(ch["stars"]["3"]["efficiency"], stats))


def _write(ch: dict, path: str, value: float) -> None:
    t1, t2, t3 = derived(ch["goal"]["metric"], value)
    ch["goal"]["threshold"] = t1
    ch["stars"]["2"]["threshold"] = t2
    ch["stars"]["3"]["threshold"] = t3
    ch["ref"]["value"] = value
    ch["ref"]["x"] = list(STAR_X)
    with open(path, "w") as fh:
        json.dump(ch, fh, indent=2)
        fh.write("\n")


def main(argv=None) -> int:
    import argparse
    import tempfile
    ap = argparse.ArgumentParser(prog="python3 -m drive.challenges")
    ap.add_argument("--measure", action="store_true", help="run every reference, print it")
    ap.add_argument("--write", action="store_true", help="... and write the thresholds")
    a = ap.parse_args(argv)
    if not (a.measure or a.write):
        return 0 if self_check() else 1
    from .aero.library import Library
    lib = Library(os.path.join(tempfile.mkdtemp(prefix="carsim_chal_"), "library"),
                  use_xfoil=False)
    for f in sorted(os.listdir(DIR)):
        if not f.endswith(".json"):
            continue
        p = os.path.join(DIR, f)
        ch = json.load(open(p))
        r = measure(ch, lib)
        print(f"{ch['id']:<14s} {ch['class']:<28s} {ch['goal']['metric']:<13s} "
              f"{ch['ref']['driver']:<22s} {_build_name(ch['ref'].get('build', 'none')):<6s} -> "
              f"{fmt_value(ch['goal']['metric'], r['value']):>12s}  "
              f"({r['t_sim']:.1f} s sim)  cda {r['stats']['cda']:.3f} mass "
              f"{r['stats']['mass']:.1f}" + (f"  REFUSED: {r['refusals']}" if r["refusals"] else ""))
        if a.write and _num(r["value"]):
            _write(ch, p, r["value"])
    return 0


# ==================================================================== #
#  SELF-CHECK                                                          #
# ==================================================================== #
def self_check(verbose: bool = True) -> bool:
    import tempfile
    from types import SimpleNamespace
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}" + (f": {msg}" if msg else ""))

    # --- every file
    files = sorted(f for f in os.listdir(DIR) if f.endswith(".json"))
    allc = {}
    for f in files:
        d = json.load(open(os.path.join(DIR, f)))
        why = validate(d)
        rep(f"{f} is valid", not why, "; ".join(why) or
            f"{d['class']} {d['goal']['metric']} 1/2/3 = "
            + " / ".join(fmt_value(d['goal']['metric'], t) for t in thresholds(d)))
        if not why:
            allc[d["id"]] = d
    metrics = {c["goal"]["metric"] for c in allc.values()}
    tracks = {c["class"].split("|")[0] for c in allc.values()}
    surf = {c["class"].split("|")[3] for c in allc.values()}
    modes = {c["ref"].get("wing_mode", "auto") for c in allc.values()}
    rep("8 challenges: stops (one on the air brake), skidpad circles, laps, and wet",
        len(allc) == 8 and {"stop_distance", "skid_ay", "lap_time"} == metrics
        and "air_brake" in modes and "all" in surf
        and {"dragstrip", "skidpad", "arena", "open"} <= tracks and len(load_all()) == 8,
        f"{len(allc)} files, metrics {sorted(metrics)}, surfaces {sorted(surf)}")
    # --- a broken file is refused with its reasons
    good = next(iter(allc.values()))
    bad = json.loads(json.dumps(good))
    bad["goal"]["threshold"] = bad["stars"]["3"]["threshold"] * (0.9 if METRICS[
        bad["goal"]["metric"]]["lower"] else 1.1)
    bad["class"] = "arena|corsa|warp|patch"
    bad["constraints"]["wings"] = 2
    why = validate(bad)
    rep("a bad file: every fault named", any("engine" in w for w in why)
        and any("unknown constraint" in w for w in why)
        and any("tighten" in w for w in why), "; ".join(why))
    # --- a stop challenge starts rolling (task 40)
    from . import track as _trk
    from .powertrain import PowertrainParams
    stops = [c for c in allc.values() if c["goal"]["metric"] == "stop_distance"]
    rep("every stop challenge starts rolling, above its v0",
        len(stops) == 3 and all(c["goal"]["v0_kmh"] < c["goal"]["start_kmh"] for c in stops),
        ", ".join(f"{c['id']} {c['goal']['start_kmh']:.0f} > {c['goal']['v0_kmh']:.0f}"
                  for c in stops))
    why_s = []
    for sk in (None, stops[0]["goal"]["v0_kmh"], 3.0 * stops[0]["goal"]["v0_kmh"]):
        d_ = json.loads(json.dumps(stops[0]))
        d_["goal"].pop("start_kmh", None)
        if sk is not None:
            d_["goal"]["start_kmh"] = sk
        why_s.append(validate(d_))
    rep("a stop with no start_kmh, or one not above v0 (or over twice it), is refused",
        all(any("start_kmh" in w for w in w_) for w_ in why_s), str(why_s))
    pt = PowertrainParams()
    strip = _trk.make_track("dragstrip", surfaces=False)
    pose = ChallengeRun(stops[0], strip, {}).rolling(strip, pt)
    lap = next(c for c in allc.values() if c["goal"]["metric"] == "lap_time")
    rep("the rolling pose: start_kmh down the strip in the automatic's gear; a lap stands",
        pose is not None and pose[0] == START_S
        and abs(pose[1] * 3.6 - stops[0]["goal"]["start_kmh"]) < 1e-9
        and 1 < pose[2] <= len(pt.gear)
        and ChallengeRun(lap, strip, {}).rolling(strip, pt) is None, str(pose))
    from types import SimpleNamespace as _NS
    run_s = ChallengeRun(stops[0], strip, {})
    _c = stops[0]["class"].split("|")
    fs = _NS(track=strip, settings=_NS(car=_c[1], engine=_c[2], wet=_c[3]), global_wet=1.0,
             time_scale=1.0, paused=False, rivals=[],
             veh=_NS(u=stops[0]["goal"]["start_kmh"] / 3.6, v=0.0))
    st_go = run_s._status(fs)
    fs.global_wet = 0.5
    st_wet = run_s._status(fs)
    fs.global_wet, fs.rivals, fs.veh.u = 1.0, [object()], 0.0
    st_race = run_s._status(fs)
    rep("the stop's line: brake now; T on says why it cannot count; a race promises no roll",
        st_go.startswith("brake now") and st_wet == "not counting: the wet toggle (T) is on"
        and "again from" not in st_race, f"{st_go!r} / {st_wet!r} / {st_race!r}")
    tmp = tempfile.mkdtemp(prefix="carsim_chal_")
    with open(os.path.join(tmp, "a.json"), "w") as fh:
        fh.write("{not json")
    json.dump(bad, open(os.path.join(tmp, "b.json"), "w"))
    json.dump(good, open(os.path.join(tmp, "c.json"), "w"))
    rep("load_all skips a broken file and a bad one, keeps the good one",
        list(load_all(tmp)) == [good["id"]])
    # --- the constraint checker
    st = dict(area={"left": 0.35, "right": 0.35, "top": 0.42}, mass=14.2, ballast=50.0,
              slots=["left", "right", "top"], cda=0.81)
    r = refusals(dict(slots=["left", "right"], max_wing_area=0.40, max_wing_mass=10.0,
                      max_ballast=0.0, max_cda=0.75), st)
    rep("constraints: slot, area, mass, ballast and drag each refused, each with what to do",
        len(r) == 5 and "top wing is not allowed" in r[0] and "0.420 m^2" in r[1]
        and "14.2 kg" in r[2] and "Ballast" in r[3] and "0.810" in r[4], " | ".join(r))
    rep("constraints: a build inside every bound starts",
        refusals(dict(slots=["left", "right", "top"], max_wing_area=0.5, max_wing_mass=15.0,
                      max_ballast=50.0, max_cda=0.81), st) == [])
    ch = dict(good)
    lower = METRICS[ch["goal"]["metric"]]["lower"]
    t1, t2, t3 = thresholds(ch)
    eff = ch["stars"]["3"]["efficiency"]
    clean = dict(area={"left": 0.0, "right": 0.0, "top": 0.0}, mass=0.0, ballast=0.0,
                 slots=[], cda=0.60)
    k = sorted(eff)[0]
    dirty = dict(clean, **({"mass": eff[k] + 1.0} if k == "max_wing_mass" else
                           {"ballast": eff[k] + 1.0} if k == "max_ballast" else
                           {"cda": eff[k] + 0.01} if k == "max_cda" else
                           {"area": {"left": eff[k] + 0.01, "right": 0.0, "top": 0.0},
                            "slots": ["left"]}))
    worse = (lambda t: t * 1.001) if lower else (lambda t: t * 0.999)
    got = (stars_for(ch, worse(t1), clean), stars_for(ch, t1, clean), stars_for(ch, t2, clean),
           stars_for(ch, t3, clean), stars_for(ch, t3, dirty))
    rep("stars: none / 1 / 2 / 3, and the 3rd needs the efficiency bound", got == (0, 1, 2, 3, 2),
        f"{got} ({ch['id']}: {k})")
    # --- the meters, on a synthetic sim
    tr = SimpleNamespace(length=1500.0, closed=False, width=15.0)

    def fsim():
        return SimpleNamespace(s=0.0, n_lat=0.0, on_track=True, dt=0.001, t=0.0, n=0,
                               veh=SimpleNamespace(u=0.0, v=0.0, ay=0.0),
                               lap=SimpleNamespace(lap_valid=True, lap_time=0.0))
    m = Meter(dict(metric="stop_distance", v0_kmh=100.0), tr)
    s = fsim()
    V, x, a = 0.0, 0.0, 4.0
    for i in range(40000):                 # 4 m/s^2 up to 105 km/h, then -9 m/s^2
        if V >= 105 / 3.6:
            a = -9.0
        V = max(V + a * s.dt, 0.0)
        x += V * s.dt
        s.veh.u, s.s, s.t = V, x, (i + 1) * s.dt
        m.step(s)
        if m.result is not None:
            break
    want = (100 / 3.6) ** 2 / (2 * 9.0)
    rep("stop distance: from v0 down to a stop, sub-step", m.result is not None
        and abs(m.result - want) < 0.02, f"{m.result:.4f} m vs {want:.4f} m (v^2/2a)")
    m = Meter(dict(metric="drag_time", distance_m=402.336), tr)
    m2 = Meter(dict(metric="trap_speed", distance_m=402.336), tr)
    s = fsim()
    V, x = 0.0, 0.0
    for i in range(60000):
        s.t = (i + 1) * s.dt
        if i >= 500:                       # 0.5 s standing, then 3 m/s^2
            V += 3.0 * s.dt
            x += V * s.dt
        s.veh.u, s.s = V, x
        m.step(s)
        m2.step(s)
        if m.result is not None and m2.result is not None:
            break
    T = math.sqrt(2 * 402.336 / 3.0)
    T_clock = T - MOVE_V / 3.0             # the clock starts at MOVE_V, not at the first mm
    rep("drag: the clock from the car moving (0.1 m/s), time and trap speed at the distance",
        m.result is not None and abs(m.result - T_clock) < 0.002
        and abs(m2.result - 3.0 * T * 3.6) < 0.05,
        f"{m.result:.4f} s vs {T_clock:.4f}; {m2.result:.2f} km/h vs {3.0 * T * 3.6:.2f}")
    trl = SimpleNamespace(length=314.159, closed=True, width=20.0)
    m = Meter(dict(metric="skid_ay"), trl)
    s = fsim()
    m.event(s, ("start", 0, 0.0, float("nan")))
    for i in range(1, 15708):
        s.s = (i * 0.02) % 314.159
        s.veh.ay = 8.0 if i % 2 else -8.0
        s.on_track = not (7000 < i < 7005)
        m.step(s)
    m.event(s, ("lap", 1, 15.7, 15.7))
    first = (m.result, m.why)
    s.on_track = True
    for i in range(1, 15708):
        s.s = (i * 0.02) % 314.159
        s.veh.ay = 8.0
        m.step(s)
    m.event(s, ("lap", 2, 31.4, 15.7))
    rep("skid a_y: a lap off the circle does not count; a clean one is its mean |a_y|",
        first == (None, "off the circle") and m.result is not None
        and abs(m.result - 8.0 / G) < 1e-9,
        f"{first}, then {m.result}")
    m = Meter(dict(metric="lap_time"), trl)
    s = fsim()
    m.event(s, ("start", 0, 0.0, float("nan")))
    s.s = 5.0
    m.step(s)
    s.s = 3.0
    m.step(s)
    m.event(s, ("lap", 1, 14.0, 14.0))
    short = (m.result, m.why)
    rep("lap: a lap that did not go round does not count", short == (None, "not a full lap"),
        str(short))
    # --- the run keeps the best and the stars in the progress file
    from .progress import Progress
    prog = Progress(os.path.join(tmp, "progress.json"))
    run = ChallengeRun(ch, trl, clean, prog)
    from .records import split_key
    from .track import MU_WET_SCALE
    tk, ck, ek, sk = split_key(ch["class"])
    csim = SimpleNamespace(track=SimpleNamespace(name=tk), settings=SimpleNamespace(
        car=ck, engine=ek, wet=sk), global_wet=(MU_WET_SCALE if sk == "all" else 1.0))
    for v in (worse(t1), t2, 0.5 * (t1 + t2)):    # none, 2 stars, then a worse 1-star lap
        run.meter._done(v)
        run._collect(csim)
    sec = Progress(prog.path).section(SECTION).get(ch["id"], {})
    rep("the best and the most stars are saved; a worse attempt changes nothing",
        sec.get("stars") == 2 and abs(sec.get("best", 0) - t2) < 1e-12
        and "[*--]" in run.note and "NEW BEST" not in run.note, f"{sec} / {run.note}")
    csim.settings.engine = "stock" if ek != "stock" else "sport"
    run.meter._done(t3)
    run._collect(csim)
    rep("a result after a live engine change is not the challenge's",
        "not counted" in run.note and Progress(prog.path).section(SECTION)[ch["id"]]["stars"] == 2,
        run.note)
    # the per-step guard: an attempt with T, a live change or slow motion in it is dropped
    csim.settings.engine = ek
    run.meter.counting, run.meter.armed = True, True
    csim.time_scale, csim.paused = 0.25, False
    run.step(csim)
    slow = (not run.meter.counting and run.meter.why == "slow motion")
    csim.time_scale = 1.0
    csim.global_wet = 1.0 if sk == "all" else 0.632183908
    run.meter.counting, run.meter.armed = True, True
    run.step(csim)
    wet = not run.meter.counting and "wet toggle" in run.meter.why
    rep("slow motion or the T toggle inside an attempt drops it, even undone later", slow and wet,
        run.meter.why)
    # a standing run that reverses is void
    m = Meter(dict(metric="trap_speed", distance_m=100.0), tr)
    s = fsim()
    for x_, V_ in ((450.0, 0.0), (449.0, 1.0), (300.0, 2.0), (5.0, 0.0)):
        s.s, s.veh.u = x_, V_
        m.step(s)
    rep("a drag run that goes backwards is void; the next stand-still re-arms there",
        not m.counting and m.armed and m.s0 == 5.0, f"{m.s0} {m.why!r}")
    # every flank panel's drag, left or right
    from .aero.library import Library
    lib = Library(os.path.join(tmp, "library"), use_xfoil=False)
    car = SimpleNamespace(CdA=0.66)
    one_l = build_stats(dict(ref_build("none"), mirror=False,
                             slots={"left": {"wing": "plate"}, "right": {"wing": ""}}), lib, car, 0)
    one_r = build_stats(dict(ref_build("none"), mirror=False,
                             slots={"left": {"wing": ""}, "right": {"wing": "plate"}}), lib, car, 0)
    both = build_stats(ref_build("plate"), lib, car, 0)
    rep("the drag area counts each flank panel, left or right, both when both are fitted",
        abs(one_l["cda"] - one_r["cda"]) < 1e-9 and one_l["cda"] > 0.66
        and abs((both["cda"] - 0.66) - 2 * (one_l["cda"] - 0.66)) < 1e-9,
        f"left {one_l['cda']:.3f} right {one_r['cda']:.3f} both {both['cda']:.3f}")
    # a malformed saved entry: ignored with a note
    bad_p = Progress(os.path.join(tmp, "bad.json"))
    bad_p.section(SECTION).update({"brake_100": None, "brake_wet": 5, "skid_dry": {"stars": "x"},
                                   "skid_wet": {"stars": 9},
                                   "airbrake_150": {"best": 90.0, "stars": 1}})
    rep("malformed saved entries are ignored with a note, the good one counts",
        total_stars(bad_p)[0] == 1 and _best(bad_p, "skid_dry") == (None, 0)
        and any("malformed" in n for n in bad_p.notes))
    #  a challenge that is gone (task 36 replaced drag_400 / trap_1000): its
    #  saved entry is never read -- a time in seconds is not a stop distance
    old_p = Progress(os.path.join(tmp, "old.json"))
    old_p.section(SECTION).update({"drag_400": {"best": 16.2, "stars": 3},
                                   "trap_1000": {"best": 170.0, "stars": 3}})
    rep("a removed challenge's saved entry counts for nothing and is never read",
        total_stars(old_p)[0] == 0 and "drag_400" not in load_all()
        and all(_best(old_p, cid) == (None, 0) for cid in load_all()))
    if verbose:
        print(f"challenges self-check: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(main())
