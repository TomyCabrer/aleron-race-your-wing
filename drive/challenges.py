"""drive/challenges.py -- challenges (task 25).

A challenge is a JSON file in `drive/data/challenges/` (kind
`carsim-challenge-1`):

    id, title, blurb,
    class        'track|car|engine|surface' (plan D1): the session it runs in
                 (its car is the player's pick, below)
    constraints  what the BUILD may carry: max_wing_area (m^2 per slot),
                 max_wing_mass (kg, all wings fitted), max_ballast (kg),
                 max_cda (m^2: the car's own CdA plus every wing's DEPLOYED
                 drag area)
    goal         {metric, + the metric's parameter}
    stars        {"3": {efficiency: {<a constraint key>: bound}}} -- 2 stars
                 = a tighter number, 3 = tighter still AND the efficiency
                 bound
    ref          how the reference drives: the driver, abs / tc, and the
                 G mode it drives in (`wing_mode`)

THE CAR AND THE WINGS ARE THE PLAYER'S PICK (task 44). The owner: "the
player picks the config at the top of the challenge page, and also picks
the car". Every challenge is driven in any car of `cars.CAR_ORDER` and one
of four wing CONFIGS -- FULL WING (top + side), ONLY TOP, ONLY TOP FIXED,
TOP FIXED + SIDE -- applied to the fitted copy of the player's build
(`config_build`: their own wings where they have them, the stock ones lent
where they do not; "Top wing would represent the normal wing a car has.
Just hide and no use for side wing"). The class is the file's with the car
swapped. So one file is 5 x 4 = 20 COMBOS, each with its own reference,
thresholds, stars and best, keyed '<id>|<car>|<config>'; `resolve` makes
the combo's challenge dict, which everything below reads as it always read
a file. (`constraints.slots` is gone: the config decides the slots.)

THRESHOLDS ARE DERIVED, NEVER HAND-TYPED (the medals' rule, D4, and the
same multipliers): the reference run -- a scripted driver, headless, on the
config's STOCK wings in the combo's car and class -- measures a value V,
and 1 / 2 / 3 stars are V x 1.12 / 1.06 / 1.02 for a lower-is-better
metric and V / 1.12 / 1.06 / 1.02 for a higher-is-better one. So the
reference run earns three stars by 2 % by construction, and every
threshold is a measured number. The measured values live in ONE file,
`refs.json` beside the challenges (160 combos: each 'ok' with its value,
or 'unavailable' with why -- a car that cannot hold the circle, or is
governed under the stop's speed: the page says "not for this car" and
refuses Start); the files carry no numbers, and `resolve` derives the
thresholds at load. `python3 -m drive.challenges --measure` runs every
reference (a process pool) and prints them; `--write` writes refs.json.
drive.py's V35 re-measures a subset and checks each earns 3 stars on a
build that meets the challenge's constraints AND its 3-star bound, and
gives back the value refs.json holds.

Metrics, measured by the SAME per-step meter live and headless (`Meter`,
attached as `Sim.challenge` through three hooks: after every physics step,
every LapTimer event, every reset):

  lap_time       s, lower   a valid lap that went round (95 % of the length
                            counted from the crossing -- the records' rule)
  stop_distance  m, lower   from the moment the car, having been faster,
                            slows through v0 to under 0.1 m/s (dragstrip).
                            The car starts ROLLING at goal.start_kmh (at
                            most v0), and every reset (R, SHIFT+R) puts it
                            back there. AT v0 it is HELD there, on rails,
                            until the brake is in (`StopHold`, task 42): the
                            distance runs from the brake at v0
                            THE STOP BOARD (round 3, the owner's call): 1
                            and 2 stars are the distance lines; the 3rd is
                            the car's NOSE stopped within BOARD_TOL of a
                            painted board, short or past, however it was
                            braked. An amber brake marker is painted
                            MARKER_T s of v0 past the start, the board the
                            combo's 3-star distance past the marker
                            (`board_marks`): the player judges the brake
  skid_ay        g, higher  mean |lateral g| over a flying lap of the
                            skidpad, line to line, on the road throughout
  drag_time      s, lower   from standing (the car first moves) to
                            distance_m down the dragstrip
  trap_speed     km/h, hi   the speed at distance_m, same run

The constraints are checked before a challenge starts (the CHALLENGES page
and again at the session's start), on the copy the config makes; a refusal
says what is wrong and what to do about it, the garage's gating style.
Progress (best value, stars) is the `challenges` section of
`runs/progress.json` (drive/progress.py), keyed per combo (task 44). An
entry saved per challenge id before task 44 was driven on the old reference
car, the Corsa with every wing (FULL WING), so it is READ as that combo's
until the combo has an entry of its own (`combo_best`, task 45): its best,
with the stars that best earns against today's thresholds, never more than
it had. Nothing is written for it; the next result that counts writes the
combo's own entry, and the old one stays in the file. Only a player session
opens it.

UNLIMITED RUNS (task 41). A build with any wing past its car's PHYSICAL span
limit (drive/bodies.py: a flank's lower tip at the car's ground clearance, a
top wing 1.2 x the car's width) is an Unlimited build: `build_stats` says so
(`unlimited`, and the reasons in `over_limits`). It is still judged by the
challenge's own rules -- its refusals stand, an impossible wing may well be
over `max_wing_area` -- and earns stars by the same thresholds, but its best
and its stars go to a SEPARATE progress section, `challenges_unlimited` (not
nested in `challenges`, whose entries `_collect` replaces whole), and are
shown in their own Unlimited spot beside the official one: on the box, the
list and the detail page. `total_stars` and `menu_row` count official stars
only.
"""
from __future__ import annotations

import json
import math
import os

from corsa_c import G

KIND = "carsim-challenge-1"
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "challenges")
SECTION = "challenges"
#: an Unlimited build's best and stars (task 41): kept apart, never official
SECTION_UNLIMITED = "challenges_unlimited"
#: 1 / 2 / 3 stars: the medal multipliers (plan D4; drive.medals)
STAR_X = (1.12, 1.06, 1.02)
METRICS = {
    "lap_time": dict(unit="s", lower=True,
                     tracks=("arena", "linden", "kestrel", "ashdown", "open", "skidpad")),
    "stop_distance": dict(unit="m", lower=True, tracks=("dragstrip",), param="v0_kmh"),
    "skid_ay": dict(unit="g", lower=False, tracks=("skidpad",)),
    "drag_time": dict(unit="s", lower=True, tracks=("dragstrip",), param="distance_m"),
    "trap_speed": dict(unit="km/h", lower=False, tracks=("dragstrip",), param="distance_m"),
}
CONSTRAINT_KEYS = ("max_wing_area", "max_wing_mass", "max_ballast", "max_cda")
LAP_MIN_FRACTION = 0.95
STOP_V = 0.10                  # m/s: stopped (BrakeDriver's own threshold)
MOVE_V = 0.10                  # m/s: a drag run's clock starts
STILL_V = 0.05                 # m/s: standing, ready for a drag run
MAX_DS = 10.0                  # m a step: more is a teleport (LapTimer's guard)
START_S = 2.5                  # m down the open strip: a stop's rolling start (drive._build's)
#: a stop held at its v0 (task 42) lets go when the brake is in: the pedal at
#: full (the keyboard's, 0.2 s after DOWN), or a squeeze of HOLD_MIN or more
#: held still (inside HOLD_STILL) for HOLD_SETTLE -- a pad's trigger is read
#: once a frame, 17 ms at 60 fps. A pedal on its way up or down never lets go
HOLD_SETTLE = 0.10             # s
HOLD_MIN = 0.30                # pedal: airbrake.BRAKE_ON, where the air brake comes out
HOLD_STILL = 0.05              # pedal: a hand on a trigger is this steady
HOLD_ROOM = 300.0              # m of strip a held car keeps ahead: past it, back to START_S
#: the precision stop board (task 45, round 3; the owner: "a painted board at
#: the 3-star distance; the 3rd star needs the nose within 1 m of it. The
#: player judges the brake point"). The amber BRAKE MARKER is MARKER_T s of
#: v0 past the start; the BOARD is the combo's 3-star distance past the
#: marker, so a stop as good as the 3-star one, braked at the marker, ends on
#: it. Its checker is 2 x BOARD_TOL deep: the nose on it is the 3rd star
MARKER_T = 3.0                 # s of v0, the start to the brake marker
BOARD_TOL = 1.0                # m: the nose this near the board, short or past
#: sim seconds a reference measurement may take. lap_time's is an out-lap
#: and one flying lap on an arena-length map: a lap challenge on Kestrel
#: (1.91 km) in a slow wet class would need more -- scale it with the
#: track's length before adding one
T_MAX = {"lap_time": 260.0, "skid_ay": 120.0, "stop_distance": 60.0,
         "drag_time": 90.0, "trap_speed": 90.0}
#: the four wing configs a challenge is driven in (task 44), in the page's
#: order; the keys are stable (progress and refs.json are keyed by them)
CONFIGS = ("full", "top", "top_fixed", "top_fixed_side")
#: the Wings row: plain words (the owner: "full wing = top + side (make it
#: clear for the user)")
CONFIG_LABELS = {"full": "FULL WING: top + side", "top": "ONLY TOP",
                 "top_fixed": "ONLY TOP, FIXED", "top_fixed_side": "TOP FIXED + SIDE"}
#: what each wing does in a config: the page's WINGS section
CONFIG_TOP_WHAT = {"active": "moves: out when you brake or turn", "fixed": "always out"}
SIDE_WHAT = "the outer one out in corners"
#: the top wing's mode in each config, and whether the side wings are used
CONFIG_TOP_MODE = {"full": "active", "top": "active", "top_fixed": "fixed",
                   "top_fixed_side": "fixed"}
CONFIG_SIDE = {"full": True, "top": False, "top_fixed": False, "top_fixed_side": True}
#: the stock wings a config lends a build that has none there -- the
#: reference car's (`ref_build('tall')`): the rear-s1223 top wing, the
#: published plate on both flanks
STOCK_TOP = "rear-s1223"
STOCK_SIDE = "plate"
#: refs.json: every (challenge, car, config)'s measured reference, the ONE
#: place a threshold comes from (`resolve` derives them with STAR_X)
REFS_FILE = "refs.json"
REFS_PATH = os.path.join(DIR, REFS_FILE)
REFS_KIND = "carsim-challenge-refs-1"
#: a stop from a v0 more than this over a car's governor is not for that car
GOVERNOR_TOL_KMH = 0.5


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
            _num(g.get("start_kmh")) and 0.0 < g["start_kmh"] <= g["v0_kmh"]):
        bad.append("stop_distance needs start_kmh, above 0 and at most v0_kmh: the car "
                   "starts rolling there (at v0, held there until the brake is in)")
    for k, v in d["constraints"].items():
        if k == "slots":                   # task 44: the wing config decides the slots
            bad.append("constraints.slots is gone: the player's wing config picks the slots")
        elif k not in CONSTRAINT_KEYS:
            bad.append(f"unknown constraint {k!r}")
        elif not _num(v) or v < 0:
            bad.append(f"{k} must be a number >= 0")
    st = d["stars"]
    eff = (st.get("3") or {}).get("efficiency") if isinstance(st, dict) else None
    #  task 44: the numbers are refs.json's, one per combo, derived at load
    #  (`resolve`) -- a threshold typed into the file would be a second truth
    typed = [n for n, v in (("goal.threshold", g.get("threshold")),
                            ("stars.2", (st or {}).get("2")),
                            ("stars.3.threshold", ((st or {}).get("3") or {}).get("threshold")),
                            ("ref.value", (d["ref"] or {}).get("value")
                             if isinstance(d["ref"], dict) else None))
             if v is not None]
    if typed:
        bad.append(f"{', '.join(typed)} in the file: the thresholds come from {REFS_FILE} "
                   "(python3 -m drive.challenges --write)")
    if not isinstance(eff, dict) or not eff or any(k not in CONSTRAINT_KEYS for k in eff):
        bad.append("3 stars needs an efficiency bound (a numeric constraint key)")
    ref = d["ref"]
    if not isinstance(ref, dict) or not isinstance(ref.get("driver"), str):
        bad.append("ref needs a driver")
    elif ref.get("wing_mode", "auto") not in REF_WING_MODES:
        bad.append(f"ref.wing_mode must be one of {sorted(REF_WING_MODES)}")
    elif "build" in ref:
        bad.append("ref.build is gone: each config's stock wings are its reference (task 44)")
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
        names = sorted(f for f in os.listdir(folder) if f.endswith(".json") and f != REFS_FILE)
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
    """(1, 2, 3-star thresholds) of a RESOLVED challenge (`resolve`)."""
    st = ch["stars"]
    return (float(ch["goal"]["threshold"]), float(st["2"]["threshold"]),
            float(st["3"]["threshold"]))


# ==================================================================== #
#  THE COMBOS: a challenge x a car x a wing config (task 44)            #
# ==================================================================== #
def combo_key(cid: str, car: str, config: str) -> str:
    """'<id>|<car>|<config>': the key of refs.json and of the progress."""
    return f"{cid}|{car}|{config}"


_REFS_CACHE: dict = {}


def load_refs(path: str = REFS_PATH) -> dict:
    """refs.json's {combo key: entry}; {} (with a printed note) for a file
    that is missing or not refs.json -- every combo is then 'not measured
    yet', never a crash. Re-read only when the file changes."""
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return {}
    hit = _REFS_CACHE.get(path)
    if hit is not None and hit[0] == mt:
        return hit[1]
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"challenges: {REFS_FILE} ignored ({exc.__class__.__name__})")
        d = None
    refs = d.get("refs") if isinstance(d, dict) and d.get("kind") == REFS_KIND else None
    if not isinstance(refs, dict):
        if d is not None:
            print(f"challenges: {REFS_FILE} ignored (not {REFS_KIND})")
        refs = {}
    _REFS_CACHE[path] = (mt, refs)
    return refs


def _entry_why(ch: dict, e) -> str:
    """'' when refs.json's entry `e` is a usable reference for the resolved
    combo `ch`, else why not."""
    if not isinstance(e, dict):
        return "not measured yet (python3 -m drive.challenges --write)"
    if (e.get("class"), e.get("driver"), e.get("wing_mode")) != (
            ch["class"], ch["ref"]["driver"], ch["ref"].get("wing_mode", "auto")):
        return f"its reference in {REFS_FILE} is out of date (python3 -m drive.challenges --write)"
    if e.get("status") == "unavailable":
        return str(e.get("why") or "its reference cannot be driven")
    if (e.get("status") != "ok" or not _num(e.get("value")) or e["value"] <= 0.0
            or e.get("drove", ch["ref"]["driver"]) not in _backoffs(ch["ref"]["driver"])):
        return f"its reference in {REFS_FILE} is malformed"
    return ""


def resolve(ch: dict, car: str, config: str, refs: dict | None = None) -> dict:
    """The challenge `ch` (a file's dict) as driven in `car` with the wing
    `config` (task 44): a copy with the class's car swapped (the map, the
    engine, the surface and the aids kept; an engine a car cannot take -> the
    seat's default), `key` / `car` / `config`, the reference's G mode (its
    own on a config with side wings -- AIR BRAKE needs them -- else AUTO),
    and the thresholds derived from refs.json's value. `available` False
    (with `unavailable`, why) when the combo has no usable reference: the
    page says "not for this car" and refuses Start."""
    from .records import split_key
    from .drive import ENGINE_MODES, ENGINE_DEFAULT
    c = json.loads(json.dumps(ch))
    track, _car, engine, surface = split_key(ch["class"])
    if engine not in ENGINE_MODES:
        engine = ENGINE_DEFAULT
    c["class"] = "|".join((track, car, engine, surface))
    c["car"], c["config"], c["key"] = car, config, combo_key(ch["id"], car, config)
    c["ref"]["wing_mode"] = ch["ref"].get("wing_mode", "auto") if CONFIG_SIDE[config] else "auto"
    e = (load_refs() if refs is None else refs).get(c["key"])
    why = _entry_why(c, e)
    c["available"], c["unavailable"] = not why, why
    if not why:
        #  the driver that drove it: the file's, or one that backed off
        #  (`measure_combo`), so a re-measure drives the very same run
        c["ref"]["driver"] = str(e.get("drove") or c["ref"]["driver"])
        t1, t2, t3 = derived(c["goal"]["metric"], float(e["value"]))
        c["goal"]["threshold"] = t1
        c["stars"]["2"] = {"threshold": t2}
        c["stars"]["3"]["threshold"] = t3
        c["ref"]["value"], c["ref"]["x"] = float(e["value"]), list(STAR_X)
    return c


def combos(allc: dict, refs: dict | None = None) -> list:
    """Every combo of every challenge, resolved, in the list's order."""
    import cars
    refs = load_refs() if refs is None else refs
    return [resolve(ch, car, cfg, refs) for ch in allc.values()
            for car in cars.CAR_ORDER for cfg in CONFIGS]


def validate_refs(allc: dict, refs: dict) -> list:
    """Why refs.json does not cover `allc` ([] = it does): every combo
    present and current, each 'ok' with a value or 'unavailable' with a
    why."""
    bad, every = [], combos(allc, refs)
    for c in every:
        e = refs.get(c["key"])
        if c["available"]:
            continue
        why = e.get("why") if isinstance(e, dict) and e.get("status") == "unavailable" else None
        if isinstance(why, str) and why.strip() and c["unavailable"] == why:
            continue                       # unavailable, and it says why
        bad.append(f"{c['key']}: {c['unavailable'] or 'unavailable with no why'}")
    extra = sorted(set(refs) - {c["key"] for c in every})
    if extra:
        bad.append(f"{len(extra)} entries for no combo ({', '.join(extra[:3])} ...)")
    return bad


def not_for_car(ch: dict) -> str:
    """Why a resolved combo cannot be driven at all, before any run ('' =
    it can): a stop from a speed over the car's governor (the Citaro's 80
    km/h) -- held at a v0 it could never reach, it would be no stop of its."""
    import cars
    g = ch["goal"]
    car = cars.get(ch["car"])
    gov = float(getattr(car, "v_governor", 0.0) or 0.0) * 3.6
    if g["metric"] == "stop_distance" and gov > 0.0 and gov < float(g["v0_kmh"]) - GOVERNOR_TOL_KMH:
        return (f"the {cars.car_name(ch['car'])} is governed to {gov:.0f} km/h: it never "
                f"reaches {g['v0_kmh']:.0f}")
    return ""


def g_modes(config) -> tuple:
    """The G modes a challenge run steps through (task 44): the config IS
    the wing mode, so G may only add the air brake, and only where there
    are side wings for it -- (AUTO, AIR BRAKE); a top-only config (AUTO,):
    G says the challenge sets the wings. None (no config): every mode."""
    from . import airbrake as ab
    if config is None:
        return tuple(ab.CYCLE)
    return (ab.AUTO, ab.AIR) if CONFIG_SIDE.get(config) else (ab.AUTO,)


# ==================================================================== #
#  THE BUILD: what the constraints read                                #
# ==================================================================== #
def build_stats(build_json, lib, car, ballast_kg: float) -> dict:
    """What a build carries, as the constraints read it: the wing area in each
    slot (m^2, `WingSpec.S`), the fitted wings' mass (kg, the garage's own
    `mass_points`), the ballast (kg), the slots with a wing, and the drag
    area with everything deployed (m^2: the car's CdA + the top wing's CD*S +
    EACH fitted flank panel's own D/q: G can put both out). And (task 41)
    whether it is an UNLIMITED build on `car` -- any wing past the car's
    physical span limit (`unlimited`, with `bodies.over_limits`' reasons in
    `over_limits`): judged on the build as fitted to `car`."""
    from .garage import CarBuild, SLOTS
    from .aero.wing import design_point, V_REF
    from .bodies import over_limits
    b = CarBuild.from_json(build_json or {}).clamp(lib, car)
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
    over = over_limits(b, lib, car)
    return dict(area=area, mass=mass, ballast=float(ballast_kg or 0.0), slots=slots, cda=cda,
                unlimited=bool(over), over_limits=over)


def config_parts(build_json, lib, config: str) -> dict:
    """Which wings a config drives (task 44): {'top': (wing, whose),
    'side': (wing, whose) or None} -- whose is 'yours' (the build's own) or
    'stock' (lent: the build has none there). `build_json` is the player's
    WORKING build (None = no build: every wing stock)."""
    from .garage import CarBuild, SLOT_ROLE
    b = CarBuild.from_json(build_json) if isinstance(build_json, dict) else None

    def own(key):
        if b is None or not b.slot(key).wing:
            return ""
        w = lib.wings.get(b.slot(key).wing)
        return b.slot(key).wing if (w is not None and w.role == SLOT_ROLE[key]) else ""
    top = own("top")
    out = dict(top=(top, "yours") if top else (STOCK_TOP, "stock"), side=None)
    if CONFIG_SIDE[config]:
        side = own("left") or own("right")
        out["side"] = (side, "yours") if side else (STOCK_SIDE, "stock")
    return out


def config_build(build_json, lib, car, config: str):
    """The FITTED COPY a challenge drives in `config` on `car` (task 44; the
    owner: "Top wing would represent the normal wing a car has. Just hide
    and no use for side wing"). From the player's working build
    (`build_json`, None = none; never changed):

      top slot    the build's own top wing, else the STOCK one lent
                  (`STOCK_TOP`, the reference's, at the car's own default
                  station), in the config's mode: active (FULL, ONLY TOP)
                  or fixed (the two FIXED ones)
      side slots  FULL / TOP FIXED + SIDE: the build's own flanks, else the
                  stock plate pair lent (at the car's default flank
                  station); ONLY TOP / ONLY TOP FIXED: emptied -- nothing
                  drawn, no force, no mass

    then `CarBuild.clamp(lib, car)` fits it to the car, as every session's
    copy is (task 41). With no build this IS the config's reference car:
    `measure` drives exactly what a player with no wings of their own
    drives."""
    from .garage import CarBuild, Slot
    from .bodies import slot_defaults
    if isinstance(build_json, dict):
        b = CarBuild.from_json(build_json).clamp(lib, car)   # a wing the library lacks: gone
    else:
        b = CarBuild.for_car(car)
        b.name = f"stock {config}"
    parts = config_parts(b.to_json(), lib, config)
    d = slot_defaults(car)
    (fx, fh), (tx, th, ti) = d["flank"], d["top"]
    if parts["top"][1] == "stock":
        b.top = Slot(STOCK_TOP, tx, th, ti)
    b.top.mode = CONFIG_TOP_MODE[config]
    if parts["side"] is None:
        b.left.wing = b.right.wing = ""
    elif parts["side"][1] == "stock":
        b.left, b.mirror = Slot(STOCK_SIDE, fx, fh, 0.0), True
    return b.clamp(lib, car)


def refusals(bounds: dict, stats: dict) -> list:
    """Why a build may not start (or earn the 3-star bound): [] = it may.
    Each reason says what is wrong and what to do, the garage's style."""
    out = []
    for k, v in bounds.items():
        if k == "max_wing_area":
            big = {s: a for s, a in stats["area"].items() if a > v + 1e-9}
            if big:
                s, a = max(big.items(), key=lambda kv: kv[1])
                out.append(f"the {s} wing is {a:.3f} m², over the {v:.3f} m² a slot may "
                           f"carry: a smaller wing (garage ▸ 3 Wing ▸ Design box: its area "
                           f"row, S_m2)")
        elif k == "max_wing_mass" and stats["mass"] > v + 1e-9:
            out.append(f"the wings weigh {stats['mass']:.1f} kg, over {v:.1f} kg: fewer or "
                       f"smaller wings")
        elif k == "max_ballast" and stats["ballast"] > v + 1e-9:
            out.append(f"{stats['ballast']:.0f} kg of ballast, over {v:.0f} kg: ESC > "
                       f"Settings > Ballast")
        elif k == "max_cda" and stats["cda"] > v + 1e-9:
            out.append(f"the car's drag area with every wing out is {stats['cda']:.3f} m², "
                       f"over {v:.3f} m²: less wing, or a lower-drag one")
    return out


def stars_for(ch: dict, value, stats: dict, miss=None) -> int:
    """0..3 for a measured `value` driven on a build with `stats`. On a stop
    with a board (`board_marks`, round 3) the 3rd star is not a distance: it
    is `miss`, the metres the nose stopped past the board (short < 0),
    within BOARD_TOL; None (no pose) is no 3rd star."""
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
    third = (_num(miss) and abs(miss) <= BOARD_TOL + 1e-9) if board_marks(ch) else beat(t3)
    if third and not refusals(ch["stars"]["3"]["efficiency"], stats):
        return 3
    return 2


# ==================================================================== #
#  THE STOP BOARD (task 45, round 3)                                    #
# ==================================================================== #
def board_marks(ch: dict):
    """(marker s, board s), m down the strip, of a stop (the precision stop
    board): the amber brake marker MARKER_T s of v0 past the start, the
    board the combo's 3-star distance past the marker. None for another
    metric, or a stop with no 3-star distance (not measured)."""
    g = ch.get("goal") or {}
    if g.get("metric") != "stop_distance" or not _num(g.get("v0_kmh")):
        return None
    try:
        t3 = thresholds(ch)[2]
    except (TypeError, ValueError, KeyError):
        return None
    marker = START_S + MARKER_T * float(g["v0_kmh"]) / 3.6
    return marker, marker + t3


def board_brake_s(ch: dict):
    """Where the reference's nose brakes for its stop to end ON the board:
    the board less the reference's own distance (V35 and the self-check
    drive it there, `measure(brake_at=)`); None with no board."""
    marks = board_marks(ch)
    v = (ch.get("ref") or {}).get("value")
    return None if marks is None or not _num(v) else marks[1] - float(v)


def nose_x(car) -> float:
    """m from the car's CG to its front bumper: the drawn shell's nose
    (`bodies.Body.x_front`), for a cars.py key or the fitted CarSpec (its
    own a, so ballast moves it as it moves the drawn car); 0 if unknown."""
    try:
        from .bodies import body
        return float(body(car).x_front)
    except Exception:                      # noqa: BLE001 -- a stand-in with no car
        return 0.0


def board_text(miss, n: int) -> str:
    """The stop's line against the board: 'board: 0.6 m short [***]',
    'board: 1.4 m past - 3rd star needs within 1 m'; '' with no reading.
    A miss inside BOARD_TOL never reads over it, nor one outside under it."""
    if not _num(miss):
        return ""
    d = abs(float(miss))
    on = d <= BOARD_TOL + 1e-9
    d1 = min(round(d, 1), BOARD_TOL) if on else max(round(d, 1), BOARD_TOL + 0.1)
    where = "on it" if d1 < 0.05 else f"{d1:.1f} m {'past' if miss > 0 else 'short'}"
    if on:
        return f"board: {where} [{stars_text(n)}]"
    return f"board: {where} - 3rd star needs within {BOARD_TOL:.0f} m"


# ==================================================================== #
#  THE PLAYER'S SETUP against the one the stars were set with          #
# ==================================================================== #
#  Task 45, round 3. The owner: a challenge keeps the player's own setup --
#  their ABS, TC, gearbox and wings -- and "the page and the result say
#  plainly which settings rule out stars". Nothing a challenge drives
#  changes; only what the player is told. The stars were set by the
#  reference: ABS and TC as its file says (on when absent), the automatic,
#  no ballast, the config's stock wings (`measure`).
#: the gearbox in the page's 'yours' row (drive.GEARBOX_MODES' keys)
GEARBOX_WORDS = {"auto": "automatic", "manual": "manual", "clutch": "manual + clutch"}
#: ... and in a mark or a result: 'manual box: the stars assume the automatic'
GEARBOX_BOX = {"auto": "the automatic", "manual": "manual box", "clutch": "manual + clutch"}
#: the gearbox a reference drives: `measure`'s Settings keep the default
REF_GEARBOX = "auto"
#: the 3-star bounds the wings decide (the ballast is ESC > Settings')
WING_BOUNDS = ("max_wing_area", "max_wing_mass", "max_cda")


def ref_setup(ch: dict) -> dict:
    """The aids and gearbox the stars were set with: the reference's own, as
    `measure` drives it -- ABS and TC from `ref` (on when absent), the
    automatic."""
    ref = ch.get("ref") or {}
    return dict(abs=bool(ref.get("abs", True)), tc=bool(ref.get("tc", True)),
                gearbox=REF_GEARBOX)


def player_setup(settings) -> dict:
    """The player's aids and gearbox, from the session's Settings. One a
    stand-in lacks (the self-checks' synthetic sims) is None: never called a
    difference."""
    return {k: getattr(settings, k, None) for k in ("abs", "tc", "gearbox")}


def _and(words: list) -> str:
    """'a', 'a and b', 'a, b and c'."""
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _yours_text(k: str, stats: dict) -> str:
    """The copy's own number for a bound: '20.3 kg', '0.420 m² (top)'."""
    if k == "max_wing_area":
        s, a = _largest_wing(stats)
        return f"{a:.3f} m²" + (f" ({s})" if a > 0.0 else "")
    return {"max_wing_mass": f"{stats['mass']:.1f} kg",
            "max_ballast": f"{stats['ballast']:.0f} kg",
            "max_cda": f"{stats['cda']:.3f} m²"}.get(k, "?")


def _own_wings(parts) -> bool:
    """Does the config drive any wing of the player's own (`config_parts`)?"""
    return bool(parts) and any(p_ and p_[1] == "yours" for p_ in (parts["top"], parts["side"]))


def setup_diffs(ch: dict, setup, parts=None, stats=None) -> list:
    """What in the player's setup differs from the one the stars were set
    with (task 45, round 3), in the page's order: [{key, yours, theirs,
    mark, rules_out}] -- 'ABS off' / 'ABS on' / 'ABS off: expect longer
    stops'. `setup` is `player_setup`'s (a None there is no difference),
    `parts` `config_parts`' (whose wings drive), `stats` the copy's
    `build_stats`. TC is left out on a stop: it trims the throttle, and a
    stop has none. The ballast and the wings mark the 3rd star they rule
    out, by the bound (`rules_out`), before any run."""
    metric = ch["goal"]["metric"]
    stop = metric == "stop_distance"
    ref, you, out = ref_setup(ch), dict(setup or {}), []
    onoff = {True: "on", False: "off"}
    for k, name in (("abs", "ABS"), ("tc", "TC")):
        v = you.get(k)
        if v is None or bool(v) == ref[k] or (k == "tc" and stop):
            continue
        yours, theirs = f"{name} {onoff[bool(v)]}", f"{name} {onoff[ref[k]]}"
        if not bool(v) and k == "abs":
            mark = yours + (": expect longer stops" if stop else
                            ": the wheels lock when you brake hard")
        elif not bool(v):
            mark = yours + ": the wheels spin when you power out"
        else:
            mark = f"{yours}: the stars were set with {theirs}"
        out.append(dict(key=k, yours=yours, theirs=theirs, mark=mark, rules_out=False))
    g = you.get("gearbox")
    if g in GEARBOX_BOX and g != ref["gearbox"]:
        yours, theirs = GEARBOX_BOX[g], GEARBOX_BOX[ref["gearbox"]]
        out.append(dict(key="gearbox", yours=yours, theirs=theirs,
                        mark=f"{yours}: the stars assume {theirs}", rules_out=False))
    eff = ch["stars"]["3"]["efficiency"]
    broke = ({k: v for k, v in eff.items() if refusals({k: v}, stats)}
             if isinstance(stats, dict) else {})
    bal = float(stats.get("ballast") or 0.0) if isinstance(stats, dict) else 0.0
    if bal > 0.0 or "max_ballast" in broke:
        yours = f"{bal:.0f} kg of ballast"
        mark = ("your ballast rules out the 3rd star: "
                + _bound_text("max_ballast", broke["max_ballast"]) if "max_ballast" in broke
                else f"{yours}: the stars were set with none")
        out.append(dict(key="ballast", yours=yours, theirs="no ballast", mark=mark,
                        rules_out="max_ballast" in broke))
    wing_broke = [(k, v) for k, v in broke.items() if k in WING_BOUNDS]
    if wing_broke or _own_wings(parts):
        #  the bound alone: RULES' amber '3rd star' row right under it has
        #  the copy's own number (one line: the page is full)
        mark = ("your wings rule out the 3rd star: "
                + "; ".join(_bound_text(k, v) for k, v in wing_broke) if wing_broke
                else "your wings: the stars were set with the stock ones")
        out.append(dict(key="wings", yours="your wings", theirs="the stock wings", mark=mark,
                        rules_out=bool(wing_broke)))
    return out


def setup_note(diffs: list) -> str:
    """The result's words for `setup_diffs`: 'ABS off, manual box: the stars
    were set with ABS on and the automatic'; '' with none."""
    if not diffs:
        return ""
    return (", ".join(d["yours"] for d in diffs) + ": the stars were set with "
            + _and([d["theirs"] for d in diffs]))


def _wings_word(parts) -> str:
    """Whose wings drive, for the 'yours' row: 'stock wings', 'your wings',
    'your top, stock side'."""
    whose = [p_[1] for p_ in (parts["top"], parts["side"]) if p_]
    if all(w == "stock" for w in whose):
        return "stock wings"
    if all(w == "yours" for w in whose):
        return "your wings"
    return (("your" if parts["top"][1] == "yours" else "stock") + " top, "
            + ("your" if parts["side"][1] == "yours" else "stock") + " side")


def setup_section(ch: dict, setup, parts=None, stats=None) -> tuple:
    """The page's YOUR SETUP section, under WINGS (task 45, round 3): the
    player's aids, gearbox and wings as they drive it, what the stars were
    set with, and an amber '!' row (menu.Warn) for each difference with what
    it costs -- 'ABS off: expect longer stops', 'your wings rule out the 3rd
    star: at most 15.0 kg of wing'."""
    from .menu import Warn
    you, ref = dict(setup or {}), ref_setup(ch)
    onoff = {True: "on", False: "off"}
    mine = [f"{n} {onoff[bool(you[k])]}" for k, n in (("abs", "ABS"), ("tc", "TC"))
            if you.get(k) is not None]
    if you.get("gearbox") in GEARBOX_WORDS:
        mine.append(GEARBOX_WORDS[you["gearbox"]])
    theirs = [f"ABS {onoff[ref['abs']]}", f"TC {onoff[ref['tc']]}", GEARBOX_WORDS[ref["gearbox"]]]
    if parts:
        mine.append(_wings_word(parts))
        theirs.append("stock wings")
    bal = float(stats.get("ballast") or 0.0) if isinstance(stats, dict) else 0.0
    if bal > 0.0:
        mine.append(f"{bal:.0f} kg ballast")
        theirs.append("no ballast")
    rows = [("yours", "  ".join(mine)), ("stars set", "with " + ", ".join(theirs))]
    rows += [("!", Warn(d["mark"])) for d in setup_diffs(ch, setup, parts, stats)]
    return ("YOUR SETUP", rows)


def result_warn(ch: dict, value, n: int, stats: dict, setup=None, parts=None) -> str:
    """What a result with `n` stars lacks (task 45), the box's warn row: how
    far it is from the next star -- '1 star needs 44.17 m (5.82 m short)',
    by the metric's direction -- and, on 2 stars, the 3rd star's rule the
    build breaks: 'no 3rd star: the wings weigh 20.3 kg, over 15.0 kg: fewer
    or smaller wings'. Then (round 3) what in the player's setup (`setup`:
    `player_setup`'s, `parts`: `config_parts`') differs from the stars':
    '... - ABS off, manual box: the stars were set with ABS on and the
    automatic'. '' on 3 stars."""
    if n >= 3 or not _num(value):
        return ""
    metric = ch["goal"]["metric"]
    out = []
    try:
        t = thresholds(ch)[n]
    except (TypeError, ValueError, KeyError):
        t = None                           # not measured yet (--write)
    if n == 2 and board_marks(ch):
        t = None                           # round 3: a stop's 3rd star is the board, no
                                           # distance (the box's board row says how far off)
    short = None if t is None else (value - t) if METRICS[metric]["lower"] else (t - value)
    if short is not None and short > 0.0:  # on 2 stars it may be the rule alone
        gap = f"{short:.3f} s" if metric == "lap_time" else fmt_value(metric, short)
        out.append(("1 star needs" if n == 0 else f"{n + 1} stars need")
                   + f" {fmt_value(metric, t)} ({gap} short)")
    eff = refusals(ch["stars"]["3"]["efficiency"], stats) if n == 2 and stats else []
    if eff:
        out.append("no 3rd star: " + eff[0])
    #  a rule the build breaks is said once: 'no 3rd star' above, if there
    diff = setup_note([d for d in setup_diffs(ch, setup, parts, stats or None)
                       if not (eff and d["rules_out"])])
    return " - ".join(p_ for p_ in ("; ".join(out), diff) if p_)


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

    def placed(self, V: float) -> None:
        """A reset's rolling pose (task 42): its speed is the last one the
        meter saw, so a stop held AT v0 counts from its first braked step."""
        self._V = float(V)
        if self.metric == "stop_distance" and self._V >= float(self.goal["v0_kmh"]) / 3.6:
            self.armed = True

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
                #  the out-lap (task 45): the metres left to the line, where
                #  the attempt starts -- a car ON the line has a whole lap
                return f"OUT LAP  {(self.L - float(sim.s)) % self.L or self.L:.0f} m to the line"
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


class StopHold:
    """A stop challenge held AT its v0 (task 42): `Sim.step_physics` carries
    the car on at V0 -- on the centre line, in the pose `Sim.reset` gives,
    no physics -- until the brake is in (`holds`). The distance counts from
    the step it lets go, so the pedal's travel (the keyboard's 0.2 s ramp, a
    pad's trigger) is in nobody's metres, and the brake is on at v0 exactly.
    A tap, a pedal still coming up after DOWN is let go, or a finger resting
    on L2 never lets go of it. Past `s_end` (HOLD_ROOM before the strip's
    end) it goes back to s0."""

    def __init__(self, s0: float, V0: float, gear: int, s_end: float):
        self.s0, self.V0, self.gear = float(s0), float(V0), int(gear)
        self.s_end = max(float(s_end), self.s0)
        self.s = self.s0
        self._b = 0.0                  # the pedal where it was last still
        self._t = 0.0                  # s it has been still there

    def holds(self, brake, dt: float) -> bool:
        """True while the car stays held: no brake, a pedal on its way up or
        down, or one still under HOLD_MIN. False = let go: the pedal at full,
        or still at HOLD_MIN or more for HOLD_SETTLE (a squeeze)."""
        b = float(brake or 0.0)
        if b >= 1.0:
            return False
        if b <= 0.0 or abs(b - self._b) > HOLD_STILL:
            self._b, self._t = max(b, 0.0), 0.0    # lifted, or moving: start again
            return True
        self._t += dt
        return b < HOLD_MIN or self._t < HOLD_SETTLE - 1e-12

    def advance(self, dt: float) -> tuple:
        """(s, wrapped): V0 * dt on down the strip, back to s0 past s_end."""
        self.s += self.V0 * dt
        if self.s > self.s_end:
            self.s = self.s0
            return self.s, True
        return self.s, False


class ChallengeRun:
    """A challenge being driven: the meter, the build's stats, the best and
    the stars, the progress file. `Sim.challenge`."""

    def __init__(self, ch: dict, track, stats: dict, progress=None, parts=None):
        self.ch = ch
        self.stats = stats
        #  task 45, round 3: whose wings drive (`config_parts`), for the
        #  result's 'the stars were set with the stock wings'
        self.parts = parts
        self.meter = Meter(ch["goal"], track)
        self.progress = progress
        #  task 41: an Unlimited build's run keeps its best and stars in its
        #  own section; `best` / `stars` are the section this run counts in,
        #  `official` the official ones (shown beside them on the box)
        self.unlimited = bool(stats.get("unlimited")) if isinstance(stats, dict) else False
        self.section = SECTION_UNLIMITED if self.unlimited else SECTION
        #  task 44: a combo's progress is its own ('<id>|<car>|<config>');
        #  a bare file (the self-checks' meters) keeps its id
        self.key = ch.get("key", ch["id"])
        #  task 45: on the Corsa / FULL WING a result saved per challenge id
        #  before task 44 is the best to beat until this combo saves its own
        self.best, self.stars = combo_best(progress, ch, self.section)
        self.official = combo_best(progress, ch)
        self.refused = ""                  # the build breaks a rule: listed, endable, never counted
        from .records import split_key
        from .track import MU_WET_SCALE
        self._cls = split_key(ch["class"])
        self._gw = MU_WET_SCALE if self._cls[3] == "all" else 1.0
        self.last = None               # (value, stars) of the last attempt
        self.note = ""                 # ... the box's green line: the result alone
        self.warn = ""                 # ... its warn row: the next star, the 3rd's rule
        self._seen = 0
        #  round 3: a stop's precision board -- (marker s, board s), None off
        #  a stop; the last stop's nose against it (m past, short < 0) and
        #  that in words (the box's board row); the car's CG-to-nose, read once
        self.board = board_marks(ch)
        self.miss = None
        self.board_line = ""
        self._nose = None

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
        (task 40): rolling at the goal's start_kmh -- at most v0, and at v0
        held there until the brake is in (`hold`, task 42) -- START_S down
        the open strip, in the gear the automatic holds there on full
        throttle. None = a standing start (every other metric)."""
        g = self.ch["goal"]
        if g["metric"] != "stop_distance" or not _num(g.get("start_kmh")):
            return None
        from . import powertrain as ptm
        V0 = float(g["start_kmh"]) / 3.6
        gear = next((k for k in range(1, len(pt_p.gear))
                     if ptm.rpm_at_speed(pt_p, k, V0) < ptm.n_up_schedule(pt_p, k, 1.0)),
                    len(pt_p.gear))
        return (0.0 if tr.closed else START_S), V0, gear

    def placed(self, sim, V: float) -> None:
        """After a reset's rolling pose (task 42): the meter takes its speed,
        so a stop held at v0 counts from its first braked step. Not while
        the run is void (T, slow motion ...): nothing may arm it then, and
        the held steps do once it is not."""
        if not self._void(sim):
            self.meter.placed(V)

    def hold(self, tr, pose):
        """The StopHold for `rolling`'s pose when the goal starts AT v0 (task
        42), else None: below v0 the car rolls free and is braked through v0
        on the way down, the crossing rule."""
        g = self.ch["goal"]
        if pose is None or not (g["metric"] == "stop_distance" and _num(g.get("start_kmh"))
                                and float(g["start_kmh"]) >= float(g["v0_kmh"])):
            return None
        return StopHold(pose[0], pose[1], pose[2], float(tr.length) - HOLD_ROOM)

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
        metric = self.ch["goal"]["metric"]
        #  task 45: the green line is the result alone, short enough for the
        #  box; everything longer -- why it did not count, what the next star
        #  needs, the 3rd star's rule -- is the warn row, which the box wraps
        self.miss, self.board_line = None, ""
        if why:
            self.last = (v, 0)
            self.note = f"{self.ch['title']}: {fmt_value(metric, v)}  not counted"
            self.warn = f"not counted: {why}"
            return
        if self.board is not None:         # round 3: where the nose stopped, against the board
            nose = self.nose_s(sim)
            self.miss = None if nose is None else nose - self.board[1]
        n = stars_for(self.ch, v, self.stats, self.miss)
        if self.board is not None:
            self.board_line = board_text(self.miss, n)
        new_best = better(metric, v, self.best)
        self.last = (v, n)
        tag = (("  NEW UNLIMITED BEST" if new_best else "  UNLIMITED") if self.unlimited
               else ("  NEW BEST" if new_best else ""))
        self.note = f"{self.ch['title']}: {fmt_value(metric, v)} [{stars_text(n)}]{tag}"
        #  the aids as they are NOW: ESC > Settings can change them mid-run
        self.warn = result_warn(self.ch, v, n, self.stats, player_setup(sim.settings),
                                self.parts)
        if new_best or n > self.stars:
            if new_best:
                self.best = v
            self.stars = max(self.stars, n)
            if self.progress is not None:
                sec = self.progress.section(self.section)
                sec[self.key] = dict(best=self.best, stars=self.stars)
                self.progress.save(self.section)

    def nose_s(self, sim):
        """m down the strip of the car's nose: its CG's s plus the shell's
        nose (`nose_x`, the fitted car's), along the car. None for a
        stand-in with no pose (the self-checks' meters)."""
        s = getattr(sim, "s", None)
        if not _num(s):
            return None
        veh = getattr(sim, "veh", None)
        if self._nose is None:
            car = getattr(veh, "car", None)
            self._nose = nose_x(car if car is not None else (self.ch.get("car") or self._cls[1]))
        psi, psi_c = getattr(veh, "psi", None), getattr(sim, "psi_c", None)
        c = math.cos(psi - psi_c) if _num(psi) and _num(psi_c) else 1.0
        return float(s) + self._nose * c

    def marks(self, sim):
        """(marker s, board s) for the renderer to paint (round 3), while the
        session is on the challenge's map; else None."""
        tr = getattr(sim, "track", None)
        return self.board if getattr(tr, "name", None) == self._cls[0] else None

    def aim(self, sim) -> str:
        """The box's board row (round 3), live: braking, the metres to the
        board; after a stop, that stop against it (`board_line`, until the
        next one brakes); before any, what the 3rd star asks and where the
        brake marker is; held past the board, R. '' off a stop."""
        if self.marks(sim) is None:
            return ""
        marker, board = self.board
        nose = self.nose_s(sim)
        if self.meter.counting and nose is not None:
            d = board - nose
            return f"board: {d:.1f} m ahead" if d >= 0.0 else f"board: {-d:.1f} m past"
        if (getattr(sim, "stop_hold", None) is not None and nose is not None
                and nose > board + BOARD_TOL):
            return "past the board: R starts you again"
        if self.board_line:
            return self.board_line
        if nose is not None and nose > marker:
            return f"stop your nose on the board: 3rd star ({max(board - nose, 0.0):.0f} m ahead)"
        return "stop your nose on the board: 3rd star (brake marker ahead)"

    def _status(self, sim) -> str:
        """The box's live line, in the HUD's number font: the live state and
        nothing else (task 45). A stop's last distance is the green line
        above it (`note`), never joined to this one -- the two together
        wrapped the big font mid-phrase. Held at v0 it names the keys: DOWN
        / L2, and G first on a stop whose reference drives the AIR BRAKE
        while the car is not on it."""
        m, g = self.meter, self.ch["goal"]
        if g["metric"] != "stop_distance" or m.counting:
            return m.status(sim)
        why = self.class_why(sim) or (
            "slow motion" if getattr(sim, "time_scale", 1.0) != 1.0 else "")
        if why:                            # the attempt is void every step (step())
            return f"not counting: {why}"
        V = math.hypot(sim.veh.u, sim.veh.v) * 3.6
        if getattr(sim, "stop_hold", None) is not None:
            air = REF_WING_MODES["air_brake"]
            if (REF_WING_MODES.get(self.ch["ref"].get("wing_mode", "auto")) == air
                    and getattr(sim, "wing_side_mode", None) != air):
                return f"held at {g['v0_kmh']:.0f} km/h: G for AIR BRAKE, then DOWN / L2"
            return f"held at {g['v0_kmh']:.0f} km/h: DOWN / L2 brakes"
        if m.armed or V >= float(g["v0_kmh"]):
            return f"brake now: {V:3.0f} km/h"
        if (_num(g.get("start_kmh")) and not getattr(sim, "rivals", None)
                and (g["start_kmh"] >= g["v0_kmh"] or V < 1.0)):
            return f"R: again from {g['start_kmh']:.0f} km/h"
        return m.status(sim)               # a race resets to the line, standing

    def overlay(self, sim) -> dict:
        """The box (render._draw_tutorial's dict)."""
        ch = self.ch
        metric = ch["goal"]["metric"]
        t1, t2, t3 = thresholds(ch)
        eff = "; ".join(f"{_bound_text(k, v)}" for k, v in ch["stars"]["3"]["efficiency"].items())
        #  round 3: a stop's 3rd star is the board, not a third distance
        third = ("3 also the nose within 1 m of the board," if self.board is not None
                 else f"3 {fmt_value(metric, t3)}")
        text = (combo_text(ch) + f"{ch['blurb']}  Stars: 1 {fmt_value(metric, t1)}, "
                f"2 {fmt_value(metric, t2)}, {third} with {eff}.")
        best = (f"   best {fmt_value(metric, self.best)} [{stars_text(self.stars)}]"
                if self.best is not None else "")
        if self.unlimited:
            #  task 41: this run counts in the Unlimited spot, said plainly;
            #  the official best beside it is the one it does NOT touch
            ob, on = self.official
            best = ("   UNLIMITED" + (f" best {fmt_value(metric, self.best)} "
                                      f"[{stars_text(self.stars)}]" if self.best is not None else "")
                    + (f"   official {fmt_value(metric, ob)} [{stars_text(on)}]"
                       if ob is not None else "   official: none"))
            text += ("  UNLIMITED build (" + _over_text(self.stats.get("over_limits")) +
                     "): results kept in their own spot, never official.")
        #  the warn row: an attempt that did not count since the last result
        #  (the newer news), else what that result lacks (task 45)
        return dict(head=f"CHALLENGE  {ch['title']}{best}", text=text,
                    status=self._status(sim), aim=self.aim(sim),
                    warn=(f"did not count: {self.meter.why}" if self.meter.why else self.warn),
                    hint="", flash=self.note,
                    foot="ESC > Challenges: end it, or another one")


def _over_text(over) -> str:
    """What is past its limit, short: 'top 2.60 > 1.98 m'."""
    return "; ".join(f"{o.get('slot', '')} {float(o.get('span', 0.0)):.2f} > "
                     f"{float(o.get('limit', 0.0)):.2f} m" for o in (over or [])) or "past the limit"


def _bound_text(k: str, v) -> str:
    """A bound in words; the player-facing area unit is 'm²' (task 45: the
    menu's and the box's Menlo draw it)."""
    return {"max_wing_area": f"at most {v:.2f} m² of wing a slot",
            "max_wing_mass": (f"at most {v:.1f} kg of wing" if v > 0 else "no wings"),
            "max_ballast": (f"at most {v:.0f} kg of ballast" if v > 0 else "no ballast"),
            "max_cda": f"a drag area of at most {v:.3f} m²"}.get(k, f"{k} {v}")


def _largest_wing(stats: dict) -> tuple:
    """(slot, m²) of the build's largest wing, the one an area bound reads
    (`refusals`); ('', 0.0) with no wing."""
    return max(stats["area"].items(), key=lambda kv: kv[1], default=("", 0.0))


def bound_check(k: str, v, stats) -> str:
    """A 3-star bound against the copy's `stats` (task 45), a RULES row:
    'at most 15.0 kg of wing  -  yours 20.3 kg  NO'. The mark is
    `refusals`' own verdict, so the page and the result agree; the bound
    alone with no stats (a build that cannot be read)."""
    if stats is None:
        return _bound_text(k, v)
    return (f"{_bound_text(k, v)}  -  yours {_yours_text(k, stats)}  "
            + ("NO" if refusals({k: v}, stats) else "OK"))


def constraints_text(ch: dict) -> list:
    return [("rule", _bound_text(k, v)) for k, v in ch["constraints"].items()]


def combo_text(ch: dict) -> str:
    """'Opel Corsa C 1.2, FULL WING: top + side. ' -- the box's first words
    on a combo (task 44); '' on a bare file."""
    if not ch.get("config"):
        return ""
    import cars
    return f"{cars.car_name(ch['car'])}, {CONFIG_LABELS[ch['config']]}. "


# ==================================================================== #
#  THE PAGES (drive.py's Sim shows them; keyboard, pad, mouse)          #
# ==================================================================== #
#  task 45: a finished attempt can score 0 stars, so 1 star is 'the first
#  number', not 'done'; what ending a challenge gives back is the 'end' row
#  here, since the list's End row said it and its width pushed this column
#  off the panel
LIST_HELP = [("CHALLENGES", [
    ("stars", "1 = the first number, 2 = tighter, 3 = tighter still"),
    ("", "AND the efficiency rule: less wing, drag or ballast"),
    ("stops", "3rd star: stop the nose within 1 m of the board"),   # round 3
    ("car, wings", "picked at the top: LEFT / RIGHT; each car and"),
    ("", "wing config has its own stars (shown: the pick)"),
    ("rules", "what the build may carry; checked before the start"),
    ("class", "each runs in its own map, engine and surface"),
    ("end", "your map, car, engine and surface come back"),
    ("R", "a reset starts the attempt again"),
])]


def _best(progress, cid, section: str = SECTION):
    """(best, stars) saved for a challenge; a malformed entry (a hand edit, a
    future format) is ignored with a note, never a crash. `section` is
    SECTION (official) or SECTION_UNLIMITED (task 41)."""
    e = progress.section(section).get(cid) if progress is not None else None
    if e is None:
        return None, 0
    n = e.get("stars", 0) if isinstance(e, dict) else None
    if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n <= 3:
        note = f"progress.json: {section}.{cid} ignored (malformed)"
        if note not in progress.notes:
            progress.notes.append(note)
            print(note)
        return None, 0
    b = e.get("best")
    return (b if _num(b) else None), n


#: the combo an entry saved per challenge id (before task 44) was driven in:
#: the old reference car, the Corsa with every wing (`ref_build('tall')`)
OLD_COMBO = ("corsa", "full")


def combo_best(progress, ch: dict, section: str = SECTION):
    """(best, stars) of the combo `ch` (a resolved challenge; a bare file
    keeps its id): its own entry, else -- on the Corsa / FULL WING only --
    the entry saved per challenge id before task 44 (task 45: a returning
    player's stars were '0 of 456' and every row [---]). That best keeps
    the stars it earns against the combo's thresholds today, by the number
    alone (the old build is not known) and never more than it had. Read
    only: `_collect` writes the combo's own entry with the next result that
    counts. An entry for a challenge that is gone is never read."""
    key = ch.get("key", ch["id"])
    if (progress is None or key != combo_key(ch["id"], *OLD_COMBO)
            or key in progress.section(section)):
        return _best(progress, key, section)
    b, n = _best(progress, ch["id"], section)
    if b is None:
        return None, 0                     # no number to judge again
    try:
        t = thresholds(ch)
    except (TypeError, ValueError, KeyError):
        return b, 0                        # the combo not measured yet (--write)
    lower = METRICS[ch["goal"]["metric"]]["lower"]
    return b, min(n, sum(1 for t_ in t if (b <= t_ if lower else b >= t_)))


def total_stars(progress, allc=None, refs=None) -> tuple:
    """(stars got, stars there are): OFFICIAL stars only -- an Unlimited
    run's stars are never counted here (task 41) -- over every AVAILABLE
    combo (task 44: 3 x the combos refs.json has a reference for); an entry
    saved per challenge id before task 44 counts on the Corsa / FULL WING
    (`combo_best`)."""
    allc = load_all() if allc is None else allc
    every = [c for c in combos(allc, refs) if c["available"]]
    return sum(combo_best(progress, c)[1] for c in every), 3 * len(every)


def pick_stars(progress, car: str = "corsa", config: str = "full", allc=None,
               refs=None) -> tuple:
    """(stars got, stars there are) for ONE pick, the car and wing config
    the challenge pages show (task 44): 3 per challenge that pick can drive,
    each counted as the list's rows count it (`combo_best`, official only).
    The list's subtitle and the pause page's row both say it (task 45)."""
    allc = load_all() if allc is None else allc
    refs = load_refs() if refs is None else refs
    got = of = 0
    for ch in allc.values():
        c = resolve(ch, car, config, refs)
        if c["available"]:
            got += combo_best(progress, c)[1]
            of += 3
    return got, of


def menu_row(progress, run=None, car: str = "corsa", config: str = "full") -> str:
    """The pause page's row; with a challenge running, its title (the list
    it opens has the End row and every other challenge). Task 45: else the
    stars of the pick the list opens on (`car`, `config`), as the list's
    subtitle counts them -- 'of 456' was every car x config, a total that
    meant nothing to a player."""
    if run is not None:
        return f"Challenges: {run.ch['title']} running"
    got, of = pick_stars(progress, car, config)
    return f"Challenges: {got} of {of} stars (this car + wings)"


def pick_text(car: str, config: str) -> str:
    """'Opel Corsa C 1.2, FULL WING: top + side': the list's subtitle."""
    import cars
    return f"{cars.car_name(car)}, {CONFIG_LABELS[config]}"


def list_items(allc, progress, run=None, car: str = "corsa", config: str = "full",
               refs=None) -> list:
    """The list: each challenge's stars and best for the PICKED car and
    config (task 44), or 'not for this car'."""
    refs = load_refs() if refs is None else refs
    rows = []
    for cid, ch in allc.items():
        c = resolve(ch, car, config, refs)
        mark = "  <- now" if (run is not None and run.ch.get("key", run.ch["id"]) == c["key"]) else ""
        if not c["available"]:
            rows.append((f"{ch['title'][:24]:<24s} not for this car{mark}", f"ch:{cid}"))
            continue
        b, n = combo_best(progress, c)
        ub, un = combo_best(progress, c, SECTION_UNLIMITED)
        #  an Unlimited result (task 41) in its own spot after the official one
        unl = f"  unlimited [{stars_text(un)}]" if ub is not None else ""
        rows.append((f"{ch['title'][:24]:<24s} [{stars_text(n)}] "
                     f"{fmt_value(ch['goal']['metric'], b) if b is not None else '':>11s}"
                     f"{unl}{mark}",
                     f"ch:{cid}"))
    if run is not None:
        rows.append(("End the challenge", "ch_end"))     # LIST_HELP's 'end' says what comes back
    rows.append(("Back", "ch_back"))
    return rows


def _short(name: str, n: int = 16) -> str:
    return name if len(name) <= n else name[:n - 2] + ".."


def wings_line(parts: dict) -> str:
    """Which wings drive, and whose (task 44): 'top: rear-new (yours) -
    side: Side plate (stock)' or 'top: Rear wing (stock) - side wings
    hidden'. The WINGS section's first row since task 45; a built-in wing
    by the garage's own player name (`garage.wing_shown`, round 3), never
    its library key."""
    from .garage import wing_shown
    (tn, tw), side = parts["top"], parts["side"]
    return (f"top: {_short(wing_shown(tn)[0], 20)} ({tw}) - "
            + (f"side: {_short(wing_shown(side[0])[0], 20)} ({side[1]})" if side
               else "side wings hidden"))


def pick_rows(car: str, config: str, parts: dict) -> list:
    """The page's first two rows (task 44): the Car and the Wings, LEFT /
    RIGHT or a click to cycle. Which wings drive is the WINGS section's
    first row (task 45): as a row of its own it took the cursor, and ENTER
    there did nothing. `parts` is kept for the callers."""
    import cars
    return [(f"{'Car':<8s}< {cars.car_name(car)} >", "set:ch_car"),
            (f"{'Wings':<8s}< {CONFIG_LABELS[config]} >", "set:ch_cfg")]


def wings_section(config: str, parts: dict) -> tuple:
    """The WINGS help section: which wings drive and whose (`wings_line`),
    what each does in this config, and G."""
    rows = [("wings", wings_line(parts)), ("top", CONFIG_TOP_WHAT[CONFIG_TOP_MODE[config]])]
    if parts["side"]:
        rows += [("side", SIDE_WHAT), ("G", "AUTO, or AIR BRAKE: all out as you brake")]
    else:
        rows += [("side", "hidden, unused"), ("G", "the challenge sets the wings")]
    return ("WINGS", rows)


def class_text(ch) -> str:
    """The page's subtitle (task 45), the pre-race page's style from the
    proper names: 'Dragstrip  ·  Opel Corsa C 1.2  ·  Stock 1.2 16V (75 hp)
    ·  Dry everywhere'; the records' short label if a name is missing."""
    from .records import class_label, split_key
    try:
        import cars
        from . import track as trk
        from .drive import engine_label, SURFACE_LABELS
        t, c, e, s = split_key(ch["class"])
        return "  ·  ".join((trk.TRACK_TITLES[t], cars.car_name(c),
                             engine_label(e, cars.get(c)), SURFACE_LABELS[s]))
    except Exception:                      # noqa: BLE001
        try:
            return class_label(ch["class"])
        except Exception:                  # noqa: BLE001
            return str(ch.get("class", "")) if isinstance(ch, dict) else str(ch)


def other_configs(ch: dict, refs: dict | None = None, allc: dict | None = None) -> list:
    """The wing configs, other than the resolved combo `ch`'s own, its car
    can drive this challenge in (task 45): the not-for-this-car note offers
    another wing config only when one of them is there. From the file
    (`allc`, default `load_all()`) resolved afresh; [] for a bare file."""
    raw = (load_all() if allc is None else allc).get(ch.get("id"))
    if raw is None or not ch.get("config") or not ch.get("car"):
        return []
    refs = load_refs() if refs is None else refs
    return [cfg for cfg in CONFIGS
            if cfg != ch["config"] and resolve(raw, ch["car"], cfg, refs)["available"]]


def detail(ch, stats, why, progress, parts=None, refs=None, allc=None, setup=None) -> tuple:
    """(items, sections, note) of one challenge's page. A resolved combo
    (task 44) with `parts` (`config_parts`) opens with the Car / Wings rows
    and the WINGS section; one that is not available says "not for this
    car" and has no Start (`refs` / `allc`: `other_configs`'). The RULES
    section checks each 3-star bound against the copy's `stats` (task 45),
    a bound it breaks in amber. With the player's `setup` (`player_setup`,
    round 3), YOUR SETUP under WINGS: theirs against the stars', each
    difference marked, a 3rd star their wings rule out named before Start."""
    from .menu import Warn
    metric = ch["goal"]["metric"]
    pick = (pick_rows(ch["car"], ch["config"], parts)
            if parts is not None and ch.get("config") else [])
    wsec = ([wings_section(ch["config"], parts)]
            if parts is not None and ch.get("config") else [])
    if not ch.get("available", True):
        #  task 45: the wing config is offered only where another one of
        #  this car can drive it (no wing gets the bus to 100 km/h)
        try:
            cfg_too = bool(other_configs(ch, refs, allc))
        except Exception:                  # noqa: BLE001 -- the note never takes the page down
            cfg_too = False
        note = (f"NOT FOR THIS CAR: {ch.get('unavailable') or 'no reference'}. Pick another "
                + ("car or wing config above." if cfg_too else "car above."))
        return pick + [("Back", "ch_list")], wsec, note
    t1, t2, t3 = thresholds(ch)
    eff = ch["stars"]["3"]["efficiency"]
    bounds = "; ".join(_bound_text(k, v) for k, v in eff.items())
    goal = [("1 star", fmt_value(metric, t1)), ("2 stars", fmt_value(metric, t2))]
    marks = board_marks(ch)
    if marks is not None:                  # round 3: a stop's 3rd star is the board
        goal += [("3 stars", "stop the nose within 1 m of the board"),
                 ("", f"with {bounds}"),
                 ("board", f"the checker {marks[1] - marks[0]:.1f} m past the amber brake marker")]
    else:
        goal.append(("3 stars", f"{fmt_value(metric, t3)} with {bounds}"))
    rules = constraints_text(ch) or [("rules", "none: any build")]
    #  task 45: each 3-star bound with this copy's number and OK / NO; a NO
    #  in amber (round 3)
    rules += [("3rd star", Warn(bound_check(k, v, stats))
               if stats is not None and refusals({k: v}, stats) else bound_check(k, v, stats))
              for k, v in eff.items()]
    if stats is not None:
        rules.append(("wing mass", f"{stats['mass']:.1f} kg"))
        if "max_wing_area" in {**ch["constraints"], **eff}:   # a rule reads the area
            s_, a_ = _largest_wing(stats)
            rules.append(("wing area", f"{a_:.3f} m² ({s_}, the largest)" if a_ > 0.0
                          else "no wing"))
        rules += [("drag area", f"{stats['cda']:.3f} m²"),
                  ("wings in", ", ".join(stats["slots"]) or "no slot")]
        if stats.get("unlimited"):         # task 41: said before the start, plainly
            rules += [("UNLIMITED", _over_text(stats.get("over_limits"))),
                      ("", "past the car's span limit: this build's results go to"),
                      ("", "the Unlimited spot, never the official one")]
    b, n = combo_best(progress, ch)
    ub, un = combo_best(progress, ch, SECTION_UNLIMITED)
    mine = [("best", (f"{fmt_value(metric, b)}  [{stars_text(n)}]" if b is not None
                      else "none yet"))]
    if ub is not None or (stats is not None and stats.get("unlimited")):
        mine.append(("unlimited", (f"{fmt_value(metric, ub)}  [{stars_text(un)}]  not official"
                                   if ub is not None else "none yet")))
    #  round 3: the player's own setup against the stars', under the wings
    ssec = [setup_section(ch, setup, parts, stats)] if setup is not None else []
    secs = wsec + ssec + [("GOAL: " + {"lap_time": "a lap", "stop_distance": "a stop",
                                       "skid_ay": "lateral g", "drag_time": "the time",
                                       "trap_speed": "the speed"}[metric].upper(), goal),
                          ("RULES", rules), ("YOURS", mine)]
    if why:                                # what blocks the start comes first
        note = "CANNOT START: " + " -- and ".join(why) + "."
        items = pick + [("Fix it in the garage", "garage"), ("Back", "ch_list")]
    else:
        note = ch["blurb"] + " Your map, car, engine and surface come back when you end it."
        items = pick + [("Start", f"ch_go:{ch['id']}"), ("Back", "ch_list")]
    return items, secs, note


# ==================================================================== #
#  THE REFERENCE RUNS                                                  #
# ==================================================================== #
def ref_build(name) -> dict:
    """The bare reference builds a dict without a config drives (`measure`'s
    `ref.build`: V38, the self-checks): 'none' (no wings), 'plate' (the
    published plate on both flanks, as the tutorial fits it), 'tall' (the
    plate on both flanks and the rear-s1223 top wing: every wing an air brake
    has; task 35), or a build JSON itself. A challenge's reference is its
    config's stock wings since task 44 (`config_build`; FULL WING on the
    Corsa is 'tall', bit for bit)."""
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
#: mode the reference drives with, as a player would set it. ALL 3 ('all')
#: was removed in task 44; the three TOP modes are there for V38.
REF_WING_MODES = {"auto": 0, "air_brake": 3, "top": 4, "top_fixed": 5, "top_fixed_side": 6}
REF_BUILDS = ("none", "plate", "tall")


def ref_driver(spec: str, tr, car, gw: float):
    """'lapdriver:<margin>' | 'brake:<pedal>:<v0 km/h>[:<clutch>]' |
    'straight:<throttle>'. The brake's clutch pedal defaults to 1 (declutched,
    the brake scripts'); the stops' references give 0, a player's on the
    automatic (task 42): the engine braking is in their number too."""
    from .drive import LapDriver, BrakeDriver, StraightDriver
    kind, _, arg = spec.partition(":")
    if kind == "lapdriver":
        return LapDriver(tr, margin=float(arg), global_wet=gw, car=car)
    if kind == "brake":
        pedal, _, rest = arg.partition(":")
        v0, _, clutch = rest.partition(":")
        return BrakeDriver(v_trigger=float(v0) / 3.6, pedal=float(pedal),
                           clutch=float(clutch) if clutch else 1.0)
    if kind == "straight":
        return StraightDriver(throttle=float(arg))
    raise ValueError(f"no reference driver {spec!r}")


class _BrakeAt:
    """A driver held off the brake until the car's nose is `s_nose` m down
    the strip, then the driver it wraps (round 3): a stop's reference held
    on at v0 to the brake point `board_brake_s` gives, so its nose stops on
    the board. The same held state lets go there as at the start."""

    def __init__(self, inner, s_nose: float):
        self.inner, self.s_nose, self.go = inner, float(s_nose), False

    def __call__(self, t, veh, tr):
        c = self.inner(t, veh, tr)
        if not self.go:
            from .track import project
            _s, _n, _k, psi_c, _i = project(tr, veh.x, veh.y)
            self.go = _s + nose_x(veh.car) * math.cos(veh.psi - psi_c) >= self.s_nose
            if not self.go:
                c.brake = 0.0              # held on: the hold lets go with the brake
        return c


def measure(ch: dict, lib, t_max: float | None = None, probe=None, driver=None,
            brake_at=None) -> dict:
    """Run the challenge's reference, headless, exactly as a session builds
    the car (`drive._session_car`), with the meter attached as a session's
    is, on the wing mode `ref.wing_mode` (a player's G; AUTO when absent).
    A resolved combo (task 44, `resolve`) drives its config's STOCK wings
    (`config_build` of no build: what a player with no wings of their own
    drives); a bare dict (V38) its `ref.build`.
    `probe(sim)` after every step (V38); `driver(t, veh, tr)` in place of
    the reference's (V35). A stop challenge starts rolling, through the
    Sim's own reset (task 40), held at v0 until the brake is in (task 42);
    with `brake_at` (round 3) held on until its nose is that far down the
    strip (`_BrakeAt`; the references themselves brake at once).
    {value, stars, stats, t_sim, start: (s, V, gear) at the start,
    refusals, why: the meter's why the last attempt did not count, miss:
    the stop's nose past its board (m, short < 0; None off a stop)}."""
    from types import SimpleNamespace
    from .records import split_key
    from . import track as trk
    from . import drive as D
    from .garage import CarBuild
    from .vehicle import Vehicle, VehicleConfig
    track, car_name, engine, surface = split_key(ch["class"])
    ref = ch["ref"]
    rs = ref_setup(ch)                     # the page's 'stars set with' reads the same
    settings = D.Settings(path="", track=track, car=car_name, engine=engine, wet=surface,
                          abs=rs["abs"], tc=rs["tc"], gearbox=rs["gearbox"])
    opts = SimpleNamespace(track=track, radius=50.0, cw=False, wet=surface, dt=D.DT_PHYS,
                           wing="off", wing_x=0.97, wing_h=0.90, wing_inc=0.0,
                           dev_flank="outer", wing_cfg=None, mass_points=())
    if ch.get("config"):
        design = config_build(None, lib, car_name, ch["config"])
    else:
        design = CarBuild.from_json(ref_build(ref.get("build", "none"))).clamp(lib, car_name)
    D._apply_design(opts, design, lib)
    tr, car, cfg_kwargs, gw = D._session_car(opts, settings)
    stats = build_stats(design.to_json(), lib, car, 0.0)
    cfg = VehicleConfig(**cfg_kwargs)
    veh = Vehicle(car, cfg)
    x, y, psi = trk.start_pose(tr, 0.0)
    veh.reset(x, y, psi, V=0.0, gear=1)
    drv = driver or ref_driver(ref["driver"], tr, car, gw)
    if brake_at is not None:
        drv = _BrakeAt(drv, brake_at)
    sim = D.Sim(veh, tr, D.ScriptedInput(drv), dt=D.DT_PHYS, wing=opts.wing,
                global_wet=gw, settings=settings)
    if cfg.has_designed():
        sim.wing_on = True                 # a garage build starts armed, as a session's does
    sim.wing_side_mode = REF_WING_MODES[ref.get("wing_mode", "auto")]
    run = ChallengeRun(ch, tr, stats)
    sim.challenge = run
    if run.rolling(tr, veh.pt_p) is not None:
        sim.reset()                        # a stop rolls from start_kmh (held at v0), as a session's does
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
                start=start, why=run.meter.why, miss=run.miss,
                refusals=refusals(ch["constraints"], stats)
                + refusals(ch["stars"]["3"]["efficiency"], stats))


#: what a reference that gave no result could not do, by metric
_CANNOT = {"lap_time": "drive a clean lap here", "skid_ay": "hold the circle",
           "stop_distance": "stop here", "drag_time": "run the strip",
           "trap_speed": "run the strip"}
#: a lap reference that leaves the road backs its margin off this much a
#: try, down to BACKOFF_MIN, before its combo is called unavailable: the MX-5
#: with only a top wing slides off the arena at the file's 0.90, not at 0.85
BACKOFF_STEP = 0.05
BACKOFF_MIN = 0.70


def _backoffs(spec: str) -> list:
    """The driver specs a reference tries, in order: the file's, then (a
    'lapdriver:<margin>') each margin BACKOFF_STEP lower, to BACKOFF_MIN."""
    kind, _, arg = spec.partition(":")
    if kind != "lapdriver":
        return [spec]
    m0 = float(arg)
    n = int(round((m0 - BACKOFF_MIN) / BACKOFF_STEP))
    return [spec] + [f"lapdriver:{m0 - k * BACKOFF_STEP:.2f}" for k in range(1, n + 1)]


def measure_combo(ch: dict, car: str, config: str, lib) -> dict:
    """refs.json's entry for one combo (a challenge FILE's dict, a car, a
    config): {status 'ok', value} or {status 'unavailable', why}, with the
    class / driver / G mode it was measured in (`_entry_why` checks them,
    so a file edited since is 'out of date', never silently wrong), and
    `drove`, the driver that did it when a lap reference had to back off."""
    import cars
    c = resolve(ch, car, config, refs={})
    out = {"class": c["class"], "driver": c["ref"]["driver"],
           "wing_mode": c["ref"].get("wing_mode", "auto")}
    why = not_for_car(c)
    if why:
        return dict(out, status="unavailable", why=why)
    first = None
    for spec in _backoffs(c["ref"]["driver"]):
        c["ref"]["driver"] = spec
        r = measure(c, lib)
        first = first or r
        if _num(r["value"]):
            break
    out["t_sim"] = round(float(r["t_sim"]), 3)
    if spec != out["driver"] and _num(r["value"]):
        out["drove"] = spec
    if not _num(r["value"]):
        r = first
        return dict(out, status="unavailable",
                    why=f"the {cars.car_name(car)} cannot {_CANNOT[c['goal']['metric']]}"
                        + (f" ({r['why']})" if r["why"] else
                           f" (no result in {T_MAX[c['goal']['metric']]:.0f} s)"))
    if r["refusals"]:                      # the stock wings break its rules on this car
        return dict(out, status="unavailable",
                    why=f"its stock wings break its rules on the {cars.car_name(car)} "
                        f"({r['refusals'][0]})")
    return dict(out, status="ok", value=float(r["value"]))


_POOL_LIB = None


def _measure_job(job):
    """A pool worker's combo: (file name, car, config, temp root) -> (key,
    entry). Each worker builds its own library once, in the parent's temp
    root (removed by the parent)."""
    global _POOL_LIB
    fname, car, config, root = job
    if _POOL_LIB is None:
        from .aero.library import Library
        _POOL_LIB = Library(os.path.join(root, f"lib_{os.getpid()}"), use_xfoil=False)
    with open(os.path.join(DIR, fname)) as fh:
        ch = json.load(fh)
    return combo_key(ch["id"], car, config), measure_combo(ch, car, config, _POOL_LIB)


def write_refs(refs: dict, path: str = REFS_PATH) -> None:
    """refs.json, one combo a line (in the list's order: small diffs)."""
    import cars
    order = {k: i for i, k in enumerate(
        combo_key(cid, car, cfg) for cid in load_all() for car in cars.CAR_ORDER
        for cfg in CONFIGS)}
    keys = sorted(refs, key=lambda k: (order.get(k, len(order)), k))
    body = ",\n".join(f"    {json.dumps(k)}: {json.dumps(refs[k], sort_keys=True)}" for k in keys)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(f'{{\n  "kind": {json.dumps(REFS_KIND)},\n  "refs": {{\n{body}\n  }}\n}}\n')
    os.replace(tmp, path)
    _REFS_CACHE.pop(path, None)


def main(argv=None) -> int:
    import argparse
    import shutil
    import tempfile
    import time
    import cars
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ap = argparse.ArgumentParser(prog="python3 -m drive.challenges")
    ap.add_argument("--measure", action="store_true",
                    help="run every combo's reference (challenge x car x config), print it")
    ap.add_argument("--write", action="store_true", help=f"... and write {REFS_FILE}")
    ap.add_argument("--only", default="",
                    help="only the combos whose key contains this ('lap_open', '|bus|'); "
                         f"--write merges them into {REFS_FILE}")
    ap.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)),
                    help="worker processes")
    a = ap.parse_args(argv)
    if not (a.measure or a.write):
        return 0 if self_check() else 1
    files = {d["id"]: f for f in sorted(os.listdir(DIR))
             if f.endswith(".json") and f != REFS_FILE
             for d in [json.load(open(os.path.join(DIR, f)))]}
    allc = load_all()
    #  the long ones first (laps, then the skidpad): the pool ends together
    slow = {"lap_time": 0, "skid_ay": 1}
    jobs = sorted(((files[cid], car, cfg) for cid in allc for car in cars.CAR_ORDER
                   for cfg in CONFIGS if a.only in combo_key(cid, car, cfg)),
                  key=lambda j: slow.get(allc[next(c for c, f in files.items() if f == j[0])]
                                         ["goal"]["metric"], 2))
    root = tempfile.mkdtemp(prefix="carsim_chal_")
    out, t0 = {}, time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as pool:
            futs = [pool.submit(_measure_job, j + (root,)) for j in jobs]
            for fu in as_completed(futs):
                key, e = fu.result()
                out[key] = e
                cid = key.split("|")[0]
                val = (fmt_value(allc[cid]["goal"]["metric"], e["value"]) if e["status"] == "ok"
                       else "unavailable")
                print(f"{len(out):3d}/{len(jobs)} {key:<32s} {val:>12s}"
                      + (f"  {e['why']}" if e["status"] != "ok" else
                         f"  ({e.get('t_sim', 0.0):.1f} s sim)"
                         + (f"  drove {e['drove']}" if e.get("drove") else "")), flush=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    n_ok = sum(e["status"] == "ok" for e in out.values())
    print(f"{len(out)} combos: {n_ok} ok, {len(out) - n_ok} unavailable, "
          f"{time.perf_counter() - t0:.0f} s")
    if a.write:
        refs = dict(load_refs()) if a.only else {}
        refs.update(out)
        write_refs(refs)
        print(f"wrote {REFS_PATH}")
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

    # --- the metrics table
    from .records import LAP_TRACKS
    rep("lap_time is measured on every map with a lap",
        set(METRICS["lap_time"]["tracks"]) == set(LAP_TRACKS),
        ", ".join(METRICS["lap_time"]["tracks"]))
    # --- every file
    files = sorted(f for f in os.listdir(DIR) if f.endswith(".json") and f != REFS_FILE)
    allc = {}
    for f in files:
        d = json.load(open(os.path.join(DIR, f)))
        why = validate(d)
        rep(f"{f} is valid", not why, "; ".join(why) or
            f"{d['class']} {d['goal']['metric']}, 3 stars with "
            + "; ".join(_bound_text(k, v) for k, v in d["stars"]["3"]["efficiency"].items()))
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
    bad["goal"]["threshold"] = 40.0        # a typed number: a second truth beside refs.json
    bad["class"] = "arena|corsa|warp|patch"
    bad["constraints"]["wings"] = 2
    bad["constraints"]["slots"] = ["left", "right"]
    why = validate(bad)
    rep("a bad file: every fault named (a typed threshold, the gone slots rule too)",
        any("engine" in w for w in why) and any("unknown constraint" in w for w in why)
        and any(REFS_FILE in w for w in why) and any("slots is gone" in w for w in why),
        "; ".join(why))
    # --- task 44: the combos -- every challenge x car x wing config
    import cars as _cars_
    from .aero.library import Library
    tmp = tempfile.mkdtemp(prefix="carsim_chal_")
    lib = Library(os.path.join(tmp, "library"), use_xfoil=False)
    refs = load_refs()
    every = combos(allc, refs)
    n_ok = sum(c["available"] for c in every)
    gone = [c for c in every if not c["available"]]
    rep(f"{REFS_FILE} covers every combo (8 x 5 cars x 4 configs), each ok with its value or "
        "unavailable with why, and is current with the files",
        len(every) == 160 and validate_refs(allc, refs) == [] and n_ok >= 120,
        f"{n_ok} ok, {len(gone)} unavailable"
        + (f" ({gone[0]['key']}: {gone[0]['unavailable']})" if gone else "")
        + ("; " + "; ".join(validate_refs(allc, refs)[:3]) if validate_refs(allc, refs) else ""))
    fl = [c for c in every if c["car"] == "corsa" and c["config"] == "full"]
    rep("every challenge is available on the Corsa with FULL WING", len(fl) == 8
        and all(c["available"] for c in fl), ", ".join(c["key"] for c in fl if not c["available"]))
    ab = [c for c in every if c["id"] == "airbrake_150"]
    rep("a stop over the bus's 80 km/h governor is not for it; the 80 km/h wet stop is",
        all(not c["available"] and "governed to 80 km/h" in c["unavailable"]
            for c in every if c["car"] == "bus" and c["id"] in ("brake_100", "airbrake_150"))
        and all(c["available"] for c in every if c["car"] == "bus" and c["id"] == "brake_wet"),
        next((c["unavailable"] for c in every if c["key"] == "brake_100|bus|full"), ""))
    #  a stop held at v0 has a FIXED top wing out when the brake goes in, as
    #  a car running at v0 has it (drive.Sim._hold_top, task 44): let go
    #  stowed, each FIXED config measured its moving twin to the bit
    twin = {"top_fixed": "top", "top_fixed_side": "full"}
    held = [(c, refs.get(combo_key(c["id"], c["car"], twin[c["config"]]), {}).get("value"))
            for c in every if c["goal"]["metric"] == "stop_distance" and c["available"]
            and c["config"] in twin]
    same = [c["key"] for c, v in held if c["ref"]["value"] == v]
    corsa = [(c["ref"]["value"], v) for c, v in held
             if c["car"] == "corsa" and c["config"] == "top_fixed"]
    rep("the stops: no FIXED config measures its moving twin (ONLY TOP, FULL WING) to the "
        "bit -- the fixed top wing is out when the brake goes in -- and the Corsa's ONLY TOP, "
        "FIXED stops shorter than ONLY TOP",
        len(held) == 26 and not same and len(corsa) == 3
        and all(_num(v) and f < v for f, v in corsa),
        f"{len(held)} combos, the same: {same[:3]}; the Corsa fixed / moving "
        + ", ".join(f"{f:.2f} / {v:.2f} m" for f, v in corsa if _num(v)))
    fake_v = 42.0
    fake = {combo_key(good["id"], "mx5", "top"): {
        "status": "ok", "value": fake_v, "class": "|".join(
            [good["class"].split("|")[0], "mx5"] + good["class"].split("|")[2:]),
        "driver": good["ref"]["driver"], "wing_mode": "auto"}}
    r_ = resolve(good, "mx5", "top", fake)
    stale = resolve(good, "mx5", "top", {k: dict(v, driver="lapdriver:0.5")
                                          for k, v in fake.items()})
    rep("resolve: the class's car swapped, the combo's key, thresholds derived from refs.json "
        "at load; an entry measured with another driver is out of date",
        r_["available"] and r_["class"].split("|")[1] == "mx5"
        and r_["class"].split("|")[::2] == good["class"].split("|")[::2]
        and r_["key"] == f"{good['id']}|mx5|top" and thresholds(r_) == derived(
            good["goal"]["metric"], fake_v) and r_["ref"]["value"] == fake_v
        and not stale["available"] and "out of date" in stale["unavailable"]
        and "threshold" not in good["goal"], f"{r_['class']} {thresholds(r_)}")
    wm = {c["config"]: c["ref"]["wing_mode"] for c in ab if c["car"] == "corsa"}
    rep("the reference's G mode: the air brake on the side configs, AUTO on the top-only ones",
        wm == {"full": "air_brake", "top": "auto", "top_fixed": "auto",
               "top_fixed_side": "air_brake"}, str(wm))
    from .airbrake import AUTO as _AUTO, AIR as _AIR
    rep("G in a challenge: AUTO / AIR BRAKE with side wings; a top-only config, AUTO alone",
        g_modes("full") == (_AUTO, _AIR) and g_modes("top_fixed_side") == (_AUTO, _AIR)
        and g_modes("top") == (_AUTO,) and g_modes("top_fixed") == (_AUTO,))
    #  the configs on a build: the stock wings lent where it has none, its
    #  own where it has them, the side wings gone on a top-only config
    from .garage import CarBuild
    ref_full = config_build(None, lib, "corsa", "full")
    tall = CarBuild.from_json(ref_build("tall")).clamp(lib, "corsa")
    kw_full, kw_tall = ref_full.cfg_kwargs(lib), tall.cfg_kwargs(lib)
    rep("no build + FULL WING on the Corsa is the old air-brake reference car, bit for bit",
        kw_full == kw_tall and ref_full.mass_points(lib) == tall.mass_points(lib),
        f"top {ref_full.top}")
    own = CarBuild.from_json(dict(ref_build("none"), slots={
        "left": {"wing": "flank-e423", "x": 0.9, "h": 0.8, "inc_deg": 2.0},
        "top": {"wing": "", "x": -0.9, "h": 1.6, "inc_deg": 6.0, "mode": "fixed"}})).to_json()
    own_top = json.loads(json.dumps(ref_build("tall")))
    own_top["slots"]["top"]["wing"], own_top["slots"]["left"]["wing"] = "rear-s1223", ""
    cb = {c: config_build(own, lib, "corsa", c) for c in CONFIGS}
    bus_top = config_build(own, lib, "bus", "top")
    parts = config_parts(own, lib, "full"), config_parts(own_top, lib, "top_fixed")
    rep("a build with flanks and no top: FULL keeps its flanks and borrows the stock top "
        "(active); ONLY TOP / ONLY TOP FIXED drop the flanks; the FIXED ones fix the top",
        cb["full"].left.wing == cb["full"].right.wing == "flank-e423"
        and cb["full"].top.wing == STOCK_TOP and cb["full"].top.mode == "active"
        and cb["top"].left.wing == cb["top"].right.wing == "" and cb["top"].top.mode == "active"
        and cb["top_fixed"].left.wing == "" and cb["top_fixed"].top.mode == "fixed"
        and cb["top_fixed_side"].left.wing == "flank-e423"
        and cb["top_fixed_side"].top.mode == "fixed" and bus_top.top.h > 3.0
        and bus_top.left.wing == "" and json.loads(json.dumps(own)) == own,
        f"full top {cb['full'].top}; bus top h {bus_top.top.h:.2f}")
    no_fl = config_build(own_top, lib, "corsa", "full")
    #  round 3: the line names a built-in wing as the garage does, never by
    #  its library key; a wing the player made keeps its own name
    import re
    from .garage import BUILTIN_WING_SHOWN
    mine_p = dict(top=("my-rear", "yours"), side=(STOCK_SIDE, "stock"))
    lines_ = [wings_line(parts[0]), wings_line(parts[1]), wings_line(mine_p)]
    rep("a build with a top and no flanks: its own top, the stock plate pair lent; the page's "
        "line says whose each is, each built-in by its garage name, no library key",
        no_fl.top.wing == "rear-s1223" and no_fl.left.wing == no_fl.right.wing == STOCK_SIDE
        and parts[0] == dict(top=(STOCK_TOP, "stock"), side=("flank-e423", "yours"))
        and parts[1] == dict(top=("rear-s1223", "yours"), side=None)
        and lines_[0] == "top: Rear wing (stock) - side: Low-drag side wing (yours)"
        and lines_[1] == "top: Rear wing (yours) - side wings hidden"
        and lines_[2] == "top: my-rear (yours) - side: Side plate (stock)"
        and not any(nm_ in BUILTIN_WING_SHOWN for ln_ in lines_
                    for nm_ in re.findall(r"(?:top|side): (.+?) \((?:yours|stock)\)", ln_)),
        " / ".join(repr(ln_) for ln_ in lines_))
    #  every stock config meets every challenge's rules and 3-star bound on
    #  every car: a player with no wings of their own can always start, and
    #  the reference earns its third star
    brk = []
    for c in every:
        if not c["available"]:
            continue
        st_ = build_stats(config_build(None, lib, c["car"], c["config"]).to_json(), lib,
                          _cars_.get(c["car"]), 0.0)
        if refusals(c["constraints"], st_) + refusals(c["stars"]["3"]["efficiency"], st_) \
                or st_["unlimited"]:
            brk.append(c["key"])
    rep("the stock wings of every config meet every rule and 3-star bound, on every car, "
        "and are never Unlimited", not brk, ", ".join(brk[:4]))
    # --- a stop challenge starts AT its speed, held until the brake is in (tasks 40, 42)
    from . import track as _trk
    from .powertrain import PowertrainParams
    stops = [c for c in allc.values() if c["goal"]["metric"] == "stop_distance"]
    rep("every stop challenge starts at its v0, never above; its reference drives the "
        "automatic's clutch",
        len(stops) == 3 and all(c["goal"]["start_kmh"] == c["goal"]["v0_kmh"]
                                and c["ref"]["driver"].count(":") == 3
                                and float(c["ref"]["driver"].rsplit(":", 1)[1]) == 0.0
                                for c in stops),
        ", ".join(f"{c['id']} {c['goal']['start_kmh']:.0f} = {c['goal']['v0_kmh']:.0f} "
                  f"{c['ref']['driver']}" for c in stops))
    why_s = []
    for sk in (None, 0.0, 1.2 * stops[0]["goal"]["v0_kmh"], 0.9 * stops[0]["goal"]["v0_kmh"]):
        d_ = json.loads(json.dumps(stops[0]))
        d_["goal"].pop("start_kmh", None)
        if sk is not None:
            d_["goal"]["start_kmh"] = sk
        why_s.append(validate(d_))
    rep("a stop with no start_kmh, one at 0 or one above v0 is refused; one below v0 starts",
        all(any("start_kmh" in w for w in w_) for w_ in why_s[:3]) and why_s[3] == [],
        str(why_s))
    pt = PowertrainParams()
    strip = _trk.make_track("dragstrip", surfaces=False)
    pose = ChallengeRun(stops[0], strip, {}).rolling(strip, pt)
    lap = next(c for c in allc.values() if c["goal"]["metric"] == "lap_time")
    rep("the rolling pose: start_kmh down the strip in the automatic's gear; a lap stands",
        pose is not None and pose[0] == START_S
        and abs(pose[1] * 3.6 - stops[0]["goal"]["start_kmh"]) < 1e-9
        and 1 < pose[2] <= len(pt.gear)
        and ChallengeRun(lap, strip, {}).rolling(strip, pt) is None, str(pose))
    #  the hold: at v0 only; the keyboard's ramp lets go at full, a squeeze
    #  once it stops rising, no brake never; past its room, back to the start
    h = ChallengeRun(stops[0], strip, {}).hold(strip, pose)
    low = json.loads(json.dumps(stops[0]))
    low["goal"]["start_kmh"] = 0.9 * low["goal"]["v0_kmh"]
    run_low = ChallengeRun(low, strip, {})
    held = isinstance(h, StopHold) and h.s_end == strip.length - HOLD_ROOM and (
        run_low.hold(strip, run_low.rolling(strip, pt)) is None
        and ChallengeRun(lap, strip, {}).hold(strip, None) is None)
    dt_ = 0.001
    kb, b, n_kb = StopHold(*pose, strip.length - HOLD_ROOM), 0.0, 0
    while kb.holds(b, dt_) and n_kb < 1000:
        b, n_kb = min(1.0, b + 5.0 * dt_), n_kb + 1     # input.RATE_BRAKE's rise, 5 /s
    sq, n_sq = StopHold(*pose, strip.length - HOLD_ROOM), 0
    while sq.holds(0.6, dt_) and n_sq < 1000:
        n_sq += 1
    idle = StopHold(*pose, strip.length - HOLD_ROOM)
    never = all(idle.holds(0.0, dt_) for _ in range(5000))

    def keys(pattern):                     # DOWN pressed / released per step, the keyboard's ramp
        h_, b_ = StopHold(*pose, strip.length - HOLD_ROOM), 0.0
        for down in pattern:
            b_ = min(1.0, b_ + 5.0 * dt_) if down else max(0.0, b_ - 8.0 * dt_)
            if not h_.holds(b_, dt_):
                return False
        return True
    rest = StopHold(*pose, strip.length - HOLD_ROOM)
    never = never and all(rest.holds(0.03, dt_) for _ in range(5000)) and keys(
        [True] * 170 + [False] * 500) and keys([False] * 500)
    after = StopHold(*pose, strip.length - HOLD_ROOM)   # R just after DOWN was let go
    b_, lets = 0.9, []
    for _ in range(500):
        b_ = max(0.0, b_ - 8.0 * dt_)
        lets.append(after.holds(b_, dt_))
    never = never and all(lets)
    wr = StopHold(START_S, 40.0, 5, START_S + 1.0)
    wrap = [wr.advance(0.02) for _ in range(2)]
    held = held and n_kb in (200, 201) and n_sq == 100 and never and wrap == [
        (START_S + 0.8, False), (START_S, True)]
    rep("the hold: at v0 only; lets go at full pedal (the keyboard's 0.2 s) or a squeeze "
        "0.1 s still; never with no brake, a tap, a pedal falling after R or a finger "
        "resting on L2 (3 %); past its room, back to the start",
        held, f"{h} keyboard {n_kb} steps, squeeze {n_sq}, never {never}, wrap {wrap}")
    from types import SimpleNamespace as _NS
    run_s = ChallengeRun(stops[0], strip, {})
    _c = stops[0]["class"].split("|")
    fs = _NS(track=strip, settings=_NS(car=_c[1], engine=_c[2], wet=_c[3]), global_wet=1.0,
             time_scale=1.0, paused=False, rivals=[], stop_hold=h,
             veh=_NS(u=stops[0]["goal"]["start_kmh"] / 3.6, v=0.0))
    st_held = run_s._status(fs)
    fs.stop_hold = None
    st_go = run_s._status(fs)
    fs.global_wet = 0.5
    st_wet = run_s._status(fs)
    fs.global_wet, fs.veh.u = 1.0, 20.0
    st_after = run_s._status(fs)
    fs.rivals, fs.veh.u = [object()], 0.0
    st_race = run_s._status(fs)
    fs.rivals, fs.veh.u = [], 0.9 * stops[0]["goal"]["v0_kmh"] / 3.6
    st_low = run_low._status(fs)           # a start under v0: the crossing rule, as before
    fs.veh.u = 0.0
    st_low0 = run_low._status(fs)
    rep("the stop's line: held, it names the brake keys; brake now; T on says why it cannot "
        "count; R again once it let go; a race promises no roll; under v0, accelerate past it",
        st_held == f"held at {stops[0]['goal']['v0_kmh']:.0f} km/h: DOWN / L2 brakes"
        and st_go.startswith("brake now") and st_wet == "not counting: the wet toggle (T) is on"
        and st_after == f"R: again from {stops[0]['goal']['start_kmh']:.0f} km/h"
        and "again from" not in st_race
        and st_low.startswith(f"accelerate past {stops[0]['goal']['v0_kmh']:.0f} km/h")
        and st_low0 == f"R: again from {low['goal']['start_kmh']:.0f} km/h",
        f"{st_held!r} / {st_go!r} / {st_wet!r} / {st_after!r} / {st_race!r} / {st_low!r} / "
        f"{st_low0!r}")
    #  task 45: the air brake's stop, held, says G first while G is not on
    #  AIR BRAKE (its reference drives there); on it, the brake keys alone
    air = next(c for c in stops if c["ref"].get("wing_mode") == "air_brake")
    _ca = air["class"].split("|")
    fa = _NS(track=strip, settings=_NS(car=_ca[1], engine=_ca[2], wet=_ca[3]), global_wet=1.0,
             time_scale=1.0, paused=False, rivals=[], stop_hold=h, wing_side_mode=0,
             veh=_NS(u=air["goal"]["start_kmh"] / 3.6, v=0.0))
    run_a = ChallengeRun(air, strip, {})
    st_air = [run_a._status(fa)]
    fa.wing_side_mode = REF_WING_MODES["air_brake"]
    st_air.append(run_a._status(fa))
    v0a = f"held at {air['goal']['v0_kmh']:.0f} km/h: "
    rep("the air brake's stop, held: G for AIR BRAKE first while G is not on it; on it, "
        "the brake keys alone", st_air == [v0a + "G for AIR BRAKE, then DOWN / L2",
                                          v0a + "DOWN / L2 brakes"], str(st_air))
    #  ... and a stop's last distance is the green line, never the live one:
    #  the big font wrapped mid-phrase with the two joined (task 45)
    rs = resolve(stops[0], "corsa", "full", refs)
    run_r = ChallengeRun(rs, strip, dict(area={"left": 0.0, "right": 0.0, "top": 0.0},
                                         mass=0.0, ballast=0.0, slots=[], cda=0.6))
    run_r.meter._done(thresholds(rs)[2])
    fs.s = run_r.board[1] - nose_x("corsa") - 0.4      # its nose stopped 0.4 m short of the board
    run_r._collect(fs)
    del fs.s
    st_r = []
    for hold_, wet_, u_ in ((h, 1.0, rs["goal"]["start_kmh"] / 3.6), (None, 1.0, 0.0),
                            (None, 0.5, 0.0), (None, 1.0, rs["goal"]["v0_kmh"] / 3.6 + 1.0)):
        fs.stop_hold, fs.global_wet, fs.veh.u = hold_, wet_, u_
        st_r.append(run_r._status(fs))
    fs.stop_hold, fs.global_wet, fs.veh.u = None, 1.0, 0.0
    rep("a stop's status never has 'LAST STOP': the result is the green line above it "
        "(held, stopped, T on, rolling)",
        not any("LAST STOP" in s_ or "m [" in s_ for s_ in st_r)
        and st_r[0].startswith("held at") and st_r[1].startswith("R: again from")
        and st_r[2].startswith("not counting") and st_r[3].startswith("brake now")
        and run_r.note == f"{rs['title']}: {fmt_value('stop_distance', thresholds(rs)[2])} "
                          "[***]  NEW BEST",
        f"{run_r.note!r} | " + " / ".join(map(repr, st_r)))
    fs.global_wet = 0.5                    # R under T: the meter stays unarmed, no stale warning
    run_v = ChallengeRun(stops[0], strip, {})
    run_v.placed(fs, stops[0]["goal"]["v0_kmh"] / 3.6)
    run_v.step(fs)
    fs.global_wet = 1.0
    run_ok = ChallengeRun(stops[0], strip, {})
    run_ok.placed(fs, stops[0]["goal"]["v0_kmh"] / 3.6)
    rep("a reset's pose arms the meter, but not while the run is void (no stale 'did not count')",
        not run_v.meter.armed and run_v.meter.why == "" and run_ok.meter.armed,
        f"void: armed {run_v.meter.armed} why {run_v.meter.why!r}; not void: {run_ok.meter.armed}")
    with open(os.path.join(tmp, "a.json"), "w") as fh:
        fh.write("{not json")
    json.dump(bad, open(os.path.join(tmp, "b.json"), "w"))
    json.dump(good, open(os.path.join(tmp, "c.json"), "w"))
    rep("load_all skips a broken file and a bad one, keeps the good one",
        list(load_all(tmp)) == [good["id"]])
    # --- the constraint checker
    st = dict(area={"left": 0.35, "right": 0.35, "top": 0.42}, mass=14.2, ballast=50.0,
              slots=["left", "right", "top"], cda=0.81)
    r = refusals(dict(max_wing_area=0.40, max_wing_mass=10.0, max_ballast=0.0, max_cda=0.75), st)
    rep("constraints: area, mass, ballast and drag each refused, each with what to do",
        len(r) == 4 and "0.420 m²" in r[0] and "14.2 kg" in r[1] and "Ballast" in r[2]
        and "0.810" in r[3], " | ".join(r))
    rep("constraints: a build inside every bound starts",
        refusals(dict(max_wing_area=0.5, max_wing_mass=15.0, max_ballast=50.0, max_cda=0.81),
                 st) == [])
    ch = resolve(good, "mx5", "top", fake)
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
    #  (the nose on the board: `ch` is a stop, whose 3rd star it is -- round 3)
    got = (stars_for(ch, worse(t1), clean), stars_for(ch, t1, clean), stars_for(ch, t2, clean),
           stars_for(ch, t3, clean, 0.0), stars_for(ch, t3, dirty, 0.0))
    rep("stars: none / 1 / 2 / 3, and the 3rd needs the efficiency bound", got == (0, 1, 2, 3, 2),
        f"{got} ({ch['id']}: {k})")
    #  round 3, the owner's precision stop board: on a stop 1 and 2 stars are
    #  the distance lines and the 3rd is the nose within 1 m of the board,
    #  short or past, however it was braked -- the 2-star distance with the
    #  nose on it is 3; the 3-star distance 2 m past, or with no pose, is 2;
    #  on the board with a 1-star distance is 1; a lap keeps its 3rd number
    lap_c = resolve(allc["lap_arena"], "corsa", "full", refs)
    lt = thresholds(lap_c)
    brd = (stars_for(ch, t2, clean, 0.5), stars_for(ch, t2, clean, -BOARD_TOL),
           stars_for(ch, t3, clean, 2.0), stars_for(ch, t3, clean, -1.01),
           stars_for(ch, t3, clean), stars_for(ch, 0.5 * (t1 + t2), clean, 0.0),
           stars_for(lap_c, lt[2], clean), stars_for(lap_c, lt[1], clean, 0.0))
    rep("the stop board: the 3rd star is the nose within 1 m of the board (the 2-star distance "
        "on it: 3; 2 m past, 1.01 m short or no pose: 2; a 1-star distance on it: 1); a lap's "
        "3rd star is still its time", brd == (3, 3, 2, 2, 2, 1, 3, 2), str(brd))
    #  ... where it is: the amber marker 3 s of v0 past the start, the board
    #  the combo's 3-star distance past the marker, both on the strip before
    #  the hold's wrap, on every stop combo with a reference; none on a lap,
    #  nor on a stop that is not for the car (no 3-star distance)
    wrap_s, bad_m = strip.length - HOLD_ROOM, []
    for c in every:
        mk = board_marks(c)
        if c["goal"]["metric"] != "stop_distance" or not c["available"]:
            if mk is not None:
                bad_m.append(c["key"])
            continue
        if (mk is None or abs(mk[0] - (START_S + MARKER_T * c["goal"]["v0_kmh"] / 3.6)) > 1e-9
                or abs(mk[1] - mk[0] - thresholds(c)[2]) > 1e-9 or mk[1] + BOARD_TOL >= wrap_s):
            bad_m.append(c["key"])
    n_st = sum(1 for c in every if board_marks(c))
    rb = resolve(allc["brake_100"], "corsa", "full", refs)
    mk_, bd_ = board_marks(rb)
    rep("the board's place: the amber brake marker 3 s of v0 past the start, the board the "
        "combo's 3-star distance past it, both before the hold's wrap; on every stop combo "
        "with a reference, none on a lap or a stop not for the car",
        not bad_m and n_st >= 40 and n_st == sum(1 for c in every if c["available"] and
                                                  c["goal"]["metric"] == "stop_distance"),
        f"{n_st} stop combos; Stop from 100 in the Corsa: marker {mk_:.1f} m, board "
        f"{bd_:.1f} m; {bad_m[:3]}")
    bt = (board_text(-0.62, 3), board_text(1.4, 2), board_text(0.01, 3), board_text(1.04, 2),
          board_text(-0.97, 3), board_text(None, 2))
    rep("the stop's board line: 'board: 0.6 m short [***]' / 'board: 1.4 m past - 3rd star "
        "needs within 1 m'; a miss just over 1 m never reads 1.0, one inside never over it",
        bt == ("board: 0.6 m short [***]", "board: 1.4 m past - 3rd star needs within 1 m",
               "board: on it [***]", "board: 1.1 m past - 3rd star needs within 1 m",
               "board: 1.0 m short [***]", ""), " | ".join(bt))
    #  the box's board row, live: held before the marker, what the 3rd star
    #  asks; past it, the metres to the board; held past the board, R;
    #  braking, the metres to it; stopped, that stop against it -- kept
    #  through R until the next stop brakes. A lap has none
    nx = nose_x("corsa")
    _cb = rb["class"].split("|")
    sb = _NS(track=strip, settings=_NS(car=_cb[1], engine=_cb[2], wet=_cb[3]), global_wet=1.0,
             time_scale=1.0, paused=False, rivals=[], stop_hold=h, s=START_S, psi_c=0.0,
             veh=_NS(u=rb["goal"]["v0_kmh"] / 3.6, v=0.0, psi=0.0))
    run_b = ChallengeRun(rb, strip, clean)
    aims = [run_b.aim(sb)]
    for s_ in (mk_ + 5.0 - nx, bd_ + BOARD_TOL + 0.5 - nx):
        sb.s = s_
        aims.append(run_b.aim(sb))
    sb.stop_hold, run_b.meter.counting = None, True
    for s_ in (bd_ - 14.2 - nx, bd_ + 0.8 - nx):
        sb.s = s_
        aims.append(run_b.aim(sb))
    run_b.meter.counting = False
    sb.s, sb.veh.u = bd_ - 2.0 - nx, 0.0               # 2 m short, a 3-star distance
    run_b.meter._done(thresholds(rb)[2])
    run_b._collect(sb)
    miss2 = (run_b.last, run_b.board_line, run_b.warn)
    sb.s = bd_ - 0.62 - nx
    run_b.meter._done(thresholds(rb)[1])               # the 2-star distance, 0.6 m short
    run_b._collect(sb)
    aims.append(run_b.aim(sb))
    sb.stop_hold, sb.s = h, START_S                    # R: held at the start again
    aims.append(run_b.aim(sb))
    ov_b = run_b.overlay(sb)
    sb.veh.psi, sb.psi_c, sb.s = 0.1, 0.0, 50.0
    tilt = run_b.nose_s(sb) - (50.0 + nx * math.cos(0.1))
    run_l = ChallengeRun(lap_c, strip, clean)
    t3_s = fmt_value("stop_distance", thresholds(rb)[2])
    rep("the box's board row: held, 'stop your nose on the board' and where; held past it, R; "
        "braking, the metres to it; stopped, the stop against it (kept through R); the box's "
        "text has no 3rd distance; the nose along the car; a lap has no board row",
        aims == ["stop your nose on the board: 3rd star (brake marker ahead)",
                 f"stop your nose on the board: 3rd star ({bd_ - mk_ - 5.0:.0f} m ahead)",
                 "past the board: R starts you again", "board: 14.2 m ahead",
                 "board: 0.8 m past", "board: 0.6 m short [***]", "board: 0.6 m short [***]"]
        and miss2 == ((thresholds(rb)[2], 2), "board: 2.0 m short - 3rd star needs within 1 m",
                      "")
        and ov_b["aim"] == aims[-1]
        and "3 also the nose within 1 m of the board, with at most 15.0 kg" in ov_b["text"]
        and t3_s not in ov_b["text"] and abs(tilt) < 1e-9 and abs(nx - 1.7515) < 1e-9
        and run_l.aim(sb) == "" and run_l.board is None,
        " / ".join(aims) + f" | {miss2[1]!r}")
    g_st = dict(detail(rb, clean, [], None)[1])["GOAL: A STOP"]
    g_lap = dict(detail(lap_c, clean, [], None)[1])["GOAL: A LAP"]
    rep("the page's GOAL on a stop: '3 stars: stop the nose within 1 m of the board', and "
        "where the board is; a lap's keeps its 3rd time",
        g_st[2:] == [("3 stars", "stop the nose within 1 m of the board"),
                     ("", "with at most 15.0 kg of wing"),
                     ("board", f"the checker {bd_ - mk_:.1f} m past the amber brake marker")]
        and len(g_st) == 5 and g_lap[2][0] == "3 stars"
        and g_lap[2][1].startswith(fmt_value("lap_time", lt[2]) + " with"),
        " | ".join(f"{a} {b}" for a, b in g_st))
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
    m = Meter(dict(metric="stop_distance", v0_kmh=100.0), tr)
    m.placed(100 / 3.6)                    # a reset's pose AT v0 (task 42), braked at once
    s, V, x = fsim(), 100 / 3.6, 0.0
    for i in range(10000):
        V = max(V - 9.0 * s.dt, 0.0)
        x += V * s.dt
        s.veh.u, s.s, s.t = V, x, (i + 1) * s.dt
        m.step(s)
        if m.result is not None:
            break
    rep("a stop placed at v0 counts from its first braked step", m.result is not None
        and abs(m.result - want) < 0.02, f"{m.result} m vs {want:.4f} m")
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
    #  task 45: before the line, a lap or a circle attempt counts down the
    #  out-lap's metres; a car ON the line has a whole lap to go
    outs = []
    for met in ("lap_time", "skid_ay"):
        m = Meter(dict(metric=met), trl)
        s = fsim()
        for s_ in (trl.length - 150.0, 0.0):
            s.s = s_
            outs.append(m.status(s))
        m.event(s, ("start", 0, 0.0, float("nan")))
        outs.append(m.status(s))
    rep("a lap or skidpad attempt before the line: 'OUT LAP  <m> m to the line', then the lap",
        outs[0::3] == ["OUT LAP  150 m to the line"] * 2
        and outs[1::3] == [f"OUT LAP  {trl.length:.0f} m to the line"] * 2
        and all(o.startswith("lap ") for o in outs[2::3]), " / ".join(outs))
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
    sec = Progress(prog.path).section(SECTION).get(ch["key"], {})
    rep("the best and the most stars are saved, under the combo's key; a worse attempt "
        "changes nothing",
        sec.get("stars") == 2 and abs(sec.get("best", 0) - t2) < 1e-12 and ch["id"] not in
        Progress(prog.path).section(SECTION)
        and "[*--]" in run.note and "NEW BEST" not in run.note, f"{sec} / {run.note}")
    csim.settings.engine = "stock" if ek != "stock" else "sport"
    run.meter._done(t3)
    run._collect(csim)
    rep("a result after a live engine change is not the challenge's",
        "not counted" in run.note and Progress(prog.path).section(SECTION)[ch["key"]]["stars"] == 2,
        run.note)
    # a 3-star number on a build outside the 3-star bound: 2 stars, and why
    run3 = ChallengeRun(ch, trl, dirty)
    run3.meter._done(t3)
    on_board = run3.board[1] - nose_x(ch["car"]) + 0.3          # the nose 0.3 m past it
    run3._collect(SimpleNamespace(track=csim.track, global_wet=csim.global_wet, s=on_board,
                                  settings=SimpleNamespace(car=ck, engine=ek, wet=sk)))
    ov3 = run3.overlay(SimpleNamespace(track=csim.track, global_wet=csim.global_wet,
                                       settings=csim.settings, veh=SimpleNamespace(u=0.0, v=0.0),
                                       s=0.0))
    rep(f"a 3-star number on a build outside the bound ({k}), its nose on the board: 2 stars; "
        "the green line is the result alone, 'no 3rd star' and why in the warn row (task 45)",
        run3.last == (t3, 2) and k == "max_wing_mass"
        and run3.note == f"{ch['title']}: {fmt_value(ch['goal']['metric'], t3)} [**-]  NEW BEST"
        and ov3["flash"] == run3.note and "3rd star" not in ov3["flash"]
        and ov3["warn"] == "no 3rd star: " + refusals(eff, dirty)[0]
        and run3.board_line == "board: 0.3 m past [**-]",
        f"{ov3['flash']!r} | {ov3['warn']!r} | {run3.board_line!r}")
    #  the warn row after a result under 3 stars: the gap to the next star,
    #  by the metric's direction (task 45)
    mt = ch["goal"]["metric"]
    skd = resolve(allc["skid_dry"], "corsa", "full", refs)
    s1, s3 = thresholds(skd)[0], thresholds(skd)[2]
    gaps = (result_warn(ch, worse(t1), 0, clean), result_warn(ch, 0.5 * (t1 + t2), 1, clean),
            result_warn(ch, worse(t3), 2, dirty), result_warn(ch, t3, 3, clean),
            result_warn(skd, s1 - 0.01, 0, clean), result_warn(skd, s3 - 0.001, 2, clean),
            result_warn(ch, worse(t3), 2, clean))
    rep("under 3 stars the warn row says what the next star needs and how far off it was, "
        "lower- or higher-is-better; on 2 stars the 3rd star's rule too -- on a stop no 3rd "
        "distance (round 3: the board row says it); nothing on 3",
        gaps[0] == f"1 star needs {fmt_value(mt, t1)} ({fmt_value(mt, worse(t1) - t1)} short)"
        and gaps[1] == (f"2 stars need {fmt_value(mt, t2)} "
                        f"({fmt_value(mt, 0.5 * (t1 + t2) - t2)} short)")
        and gaps[2] == f"no 3rd star: {refusals(eff, dirty)[0]}"
        and gaps[3] == "" and gaps[4] == (f"1 star needs {fmt_value('skid_ay', s1)} "
                                          f"({fmt_value('skid_ay', 0.01)} short)")
        and gaps[5] == (f"3 stars need {fmt_value('skid_ay', s3)} "
                        f"({fmt_value('skid_ay', 0.001)} short)")
        and gaps[6] == "" and mt == "stop_distance",
        " | ".join(gaps))
    # the page (task 44): the Car and Wings rows first, then Start -- whose
    # wings drive is the WINGS section's first row (task 45: a row of its
    # own took the cursor); a refused build gets the garage in one press; a
    # combo with no reference is not for this car and has no Start
    pt = config_parts(None, lib, "top")
    it_ok, sec_ok, _n = detail(ch, clean, [], None, parts=pt)
    it_sd, sec_sd, note_sd = detail(ch, dict(clean, slots=["top"]), ["x"], None, parts=pt)
    r_sd = dict(sec_sd)["RULES"]
    un = resolve(good, "bus", "top", {})
    it_un, sec_un, note_un = detail(un, None, [], None, parts=pt, refs={})
    rep("the page: Car < MX-5 >, Wings < ONLY TOP >, then Start (no info row: nothing dead "
        "takes the cursor); the WINGS section says whose wings drive, what each does and "
        "that G cannot change them",
        [a for _, a in it_ok] == ["set:ch_car", "set:ch_cfg", f"ch_go:{good['id']}", "ch_list"]
        and it_ok[0][0] == f"Car     < {_cars_.car_name('mx5')} >"
        and it_ok[1][0] == "Wings   < ONLY TOP >"
        and sec_ok[0][0] == "WINGS"
        and sec_ok[0][1][:2] == [("wings", "top: Rear wing (stock) - side wings hidden"),
                                 ("top", CONFIG_TOP_WHAT["active"])]
        and ("side", "hidden, unused") in sec_ok[0][1]
        and ("G", "the challenge sets the wings") in sec_ok[0][1],
        f"{it_ok[:3]} | {sec_ok[0][1][0]}")
    rep("a refused build: 'Fix it in the garage' is the first row after the pick; the build "
        "in three rows",
        it_sd[2] == ("Fix it in the garage", "garage")
        and not any(a.startswith("ch_go:") for _, a in it_sd)
        and r_sd[-3:] == [("wing mass", "0.0 kg"), ("drag area", "0.600 m²"),
                          ("wings in", "top")], str(r_sd[-3:]))
    #  task 45: another wing config is offered only where this car has one
    #  that can drive it -- the bus, governed to 80 km/h, has none for a
    #  stop from 100; a Corsa missing one config's reference has three more
    bus = resolve(allc["brake_100"], "bus", "full", refs)
    _i, _s, note_bus = detail(bus, None, [], None, parts=pt, refs=refs, allc=allc)
    refs_gap = {k_: e_ for k_, e_ in refs.items() if k_ != combo_key(good["id"], "corsa", "top")}
    gap = resolve(good, "corsa", "top", refs_gap)
    _i, _s, note_gap = detail(gap, None, [], None, parts=pt, refs=refs_gap, allc=allc)
    rep("a combo with no reference: 'not for this car', no Start, the pick rows kept; "
        "'another car' alone unless another wing config of this car can drive it",
        not un["available"] and note_un.startswith("NOT FOR THIS CAR")
        and note_un.endswith(". Pick another car above.")
        and [a for _, a in it_un] == ["set:ch_car", "set:ch_cfg", "ch_list"]
        and not bus["available"] and note_bus.endswith("governed to 80 km/h: it never reaches "
                                                       "100. Pick another car above.")
        and not gap["available"] and note_gap.endswith(" Pick another car or wing config above.")
        and other_configs(gap, refs_gap, allc) == ["full", "top_fixed", "top_fixed_side"],
        f"{note_bus} | {note_gap}")
    #  task 45: each 3-star bound on the page with this copy's number and
    #  OK / NO -- `refusals`' own verdict; the largest wing's area where a
    #  rule reads the area; the bound alone when the build cannot be read
    def star_rows(c_, st_):
        return [w_ for k_, w_ in dict(detail(c_, st_, [], None, parts=pt)[1])["RULES"]
                if k_ == "3rd star"]
    skd3 = resolve(allc["skid_dry"], "corsa", "full", refs)
    big = dict(clean, area={"left": 0.0, "right": 0.0, "top": 0.42}, slots=["top"])
    r_big = dict(detail(skd3, big, [], None, parts=pt)[1])["RULES"]
    marks = (star_rows(ch, clean), star_rows(ch, dirty), star_rows(ch, None),
             star_rows(skd3, big), star_rows(skd3, clean))
    rep("the RULES section: each 3rd-star bound with yours and OK / NO (a build over the mass "
        "rule: NO); the area's rule adds the largest wing's row; no stats, the bound alone",
        k == "max_wing_mass" and marks[0] == ["at most 15.0 kg of wing  -  yours 0.0 kg  OK"]
        and marks[1] == [f"at most 15.0 kg of wing  -  yours {dirty['mass']:.1f} kg  NO"]
        and marks[2] == ["at most 15.0 kg of wing"]
        and marks[3] == ["at most 0.40 m² of wing a slot  -  yours 0.420 m² (top)  NO"]
        and marks[4] == ["at most 0.40 m² of wing a slot  -  yours 0.000 m²  OK"]
        and ("wing area", "0.420 m² (top, the largest)") in r_big
        and "wing area" not in dict(r_sd),
        " | ".join(m_[0] if m_ else "-" for m_ in marks))
    #  round 3 (the owner: a challenge keeps the player's own setup, and the
    #  page and the result say plainly which settings rule out stars): YOUR
    #  SETUP under WINGS, theirs against the stars', each difference an
    #  amber '!' (menu.Warn); TC is not marked on a stop (no throttle there)
    from .menu import Warn
    off = dict(abs=False, tc=False, gearbox="manual")
    same = ref_setup(ch)
    own_p = config_parts(own_top, lib, "top")      # the build's own top wing
    pg_off = detail(ch, clean, [], None, parts=own_p, setup=off)
    ys_off = dict(pg_off[1])["YOUR SETUP"]
    ys_same = dict(detail(ch, clean, [], None, parts=pt, setup=same)[1])["YOUR SETUP"]
    ys_skd = dict(detail(skd3, clean, [], None, parts=pt, setup=off)[1])["YOUR SETUP"]
    no_setup = [t_ for t_, _ in detail(ch, clean, [], None, parts=pt)[1]]
    rep("YOUR SETUP: the player's ABS / TC / gearbox / wings against what the stars were set "
        "with, under WINGS; ABS off and the manual box each an amber '!' with what it costs "
        "(TC too, but not on a stop); nothing marked when they match; no setup, no section",
        ch["goal"]["metric"] == "stop_distance"
        and [t_ for t_, _ in pg_off[1]][:2] == ["WINGS", "YOUR SETUP"]
        and ys_off[:2] == [("yours", "ABS off  TC off  manual  your wings"),
                           ("stars set", "with ABS on, TC on, automatic, stock wings")]
        and ys_off[2:] == [("!", "ABS off: expect longer stops"),
                           ("!", "manual box: the stars assume the automatic"),
                           ("!", "your wings: the stars were set with the stock ones")]
        and all(isinstance(w_, Warn) for _, w_ in ys_off[2:])
        and ys_same == [("yours", "ABS on  TC on  automatic  stock wings"),
                        ("stars set", "with ABS on, TC on, automatic, stock wings")]
        and ("!", "TC off: the wheels spin when you power out") in ys_skd
        and ("!", "ABS off: the wheels lock when you brake hard") in ys_skd
        and "YOUR SETUP" not in no_setup and any(a_.startswith("ch_go:") for _, a_ in pg_off[0]),
        " | ".join(w_ for _, w_ in ys_off))
    #  a 3rd star the build rules out is named before Start, amber, by the
    #  bound -- the wings' or the ballast's -- and RULES' NO row is amber too
    ys_dirty = dict(detail(ch, dirty, [], None, parts=pt, setup=same)[1])
    bw = resolve(allc["brake_wet"], "corsa", "full", refs)
    heavy = dict(clean, ballast=50.0)
    ys_bal = dict(detail(bw, heavy, [], None, parts=pt, setup=same)[1])["YOUR SETUP"]
    star_row = [w_ for k_, w_ in ys_dirty["RULES"] if k_ == "3rd star"]
    rep("a build over a 3rd-star bound: 'your wings rule out the 3rd star: <the bound>' in "
        "YOUR SETUP before Start (Start still there), amber, and RULES' NO row amber; 50 kg "
        "of ballast on a no-ballast star: 'your ballast rules out the 3rd star'",
        ys_dirty["YOUR SETUP"][2:] == [("!", "your wings rule out the 3rd star: "
                                             "at most 15.0 kg of wing")]
        and isinstance(ys_dirty["YOUR SETUP"][2][1], Warn)
        and len(star_row) == 1 and isinstance(star_row[0], Warn) and star_row[0].endswith("NO")
        and not isinstance([w_ for k_, w_ in dict(detail(ch, clean, [], None, parts=pt,
                                                          setup=same)[1])["RULES"]
                            if k_ == "3rd star"][0], Warn)
        and ys_bal[0] == ("yours", "ABS on  TC on  automatic  stock wings  50 kg ballast")
        and ys_bal[1] == ("stars set", "with ABS on, TC on, automatic, stock wings, no ballast")
        and ys_bal[2:] == [("!", "your ballast rules out the 3rd star: no ballast")],
        f"{ys_dirty['YOUR SETUP'][2:]} | {ys_bal[2:]}")
    #  the result names the difference after the gap (the Settings as they
    #  are at the result: ESC > Settings can change them mid-run); a rule
    #  'no 3rd star' already names is not said twice; 3 stars say nothing;
    #  the reference's own setup (measure's Settings) is no difference
    sim_off = SimpleNamespace(track=csim.track, global_wet=csim.global_wet, s=on_board,
                              settings=SimpleNamespace(car=ck, engine=ek, wet=sk, **off))
    run_off = ChallengeRun(ch, trl, clean, parts=own_p)
    run_off.meter._done(worse(t1))
    run_off._collect(sim_off)
    w0_ = run_off.warn
    run_off.meter._done(t3)
    run_off._collect(sim_off)
    w3_ = run_off.warn
    run_d = ChallengeRun(ch, trl, dirty, parts=own_p)
    run_d.meter._done(worse(t3))
    run_d._collect(sim_off)
    from . import drive as _D
    rs_ = player_setup(_D.Settings(path="", abs=same["abs"], tc=same["tc"]))
    rep("the result names what differs: '1 star needs .. (.. short) - ABS off, manual box, "
        "your wings: the stars were set with ABS on, the automatic and the stock wings'; "
        "'no 3rd star' and the setup, the wings once; nothing on 3 stars",
        w0_ == (f"1 star needs {fmt_value(mt, t1)} ({fmt_value(mt, worse(t1) - t1)} short) - "
                "ABS off, manual box, your wings: the stars were set with ABS on, the "
                "automatic and the stock wings")
        and "ABS off" in w0_ and w3_ == "" and run_off.last == (t3, 3)
        and run_d.warn == (f"no 3rd star: {refusals(eff, dirty)[0]} - ABS off, manual "
                           "box: the stars were set with ABS on and the automatic")
        and setup_diffs(ch, rs_) == [] and rs_["gearbox"] == REF_GEARBOX,
        f"{w0_!r} | {run_d.warn!r}")
    #  task 45: the subtitle from the proper names, the pre-race page's way;
    #  a class a name is missing for falls back to the records' label
    from .drive import engine_label
    sub_ok = class_text(ch)
    rep("the page's subtitle: the map, the car, the engine and the surface by name, not "
        "the file's keys ('dragstrip mx5 stock dry'); an unknown map: the records' label",
        ch["class"] == "dragstrip|mx5|stock|none"
        and sub_ok == ("Dragstrip  ·  " + _cars_.car_name("mx5") + "  ·  "
                       + engine_label("stock", _cars_.get("mx5")) + "  ·  Dry everywhere")
        and "Drag" in sub_ok and "mx5" not in sub_ok.split()
        and class_text({"class": "moon|mx5|stock|none"}) == "moon  mx5  stock  dry",
        sub_ok)
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
    kf = lambda cid: combo_key(cid, "corsa", "full")          # noqa: E731
    bad_p = Progress(os.path.join(tmp, "bad.json"))
    bad_p.section(SECTION).update({kf("brake_100"): None, kf("brake_wet"): 5,
                                   kf("skid_dry"): {"stars": "x"}, kf("skid_wet"): {"stars": 9},
                                   kf("airbrake_150"): {"best": 90.0, "stars": 1}})
    rep("malformed saved entries are ignored with a note, the good one counts",
        total_stars(bad_p)[0] == 1 and _best(bad_p, kf("skid_dry")) == (None, 0)
        and any("malformed" in n for n in bad_p.notes))
    #  task 45: an entry saved per challenge id before task 44 is read as the
    #  Corsa / FULL WING's (the old reference car) until that combo saves its
    #  own: its best, judged again by today's thresholds and never more stars
    #  than it had (a 2-star number saved with 3: 2; a 3-star number saved
    #  with 1: 1); every other car and config starts clean; a challenge that
    #  is gone (task 36 replaced drag_400 / trap_1000) is never read; reading
    #  writes nothing, and the old entries stay in the file
    of_ = lambda cid: resolve(allc[cid], "corsa", "full", refs)          # noqa: E731
    tb, ta, ts = (thresholds(of_(c_)) for c_ in ("brake_100", "lap_arena", "skid_dry"))
    b_old = 0.5 * (tb[1] + tb[2])
    old_p = Progress(os.path.join(tmp, "old.json"))
    old_p.section(SECTION).update({"drag_400": {"best": 16.2, "stars": 3},
                                   "trap_1000": {"best": 170.0, "stars": 3},
                                   "brake_100": {"best": b_old, "stars": 3},
                                   "lap_arena": {"best": ta[2], "stars": 1},
                                   "skid_dry": {"best": 0.5 * (ts[0] + ts[1]), "stars": 3},
                                   "brake_wet": {"best": 30.0, "stars": 3},
                                   kf("brake_wet"): {"best": 50.0, "stars": 0}})
    old_p.section(SECTION_UNLIMITED)["brake_100"] = {"best": tb[2] - 0.5, "stars": 3}
    old_p.save(SECTION)
    old_p.save(SECTION_UNLIMITED)
    on_disk = json.dumps(Progress(old_p.path).data, sort_keys=True)
    li_old = dict((a, t) for t, a in list_items(allc, old_p, car="corsa", config="full", refs=refs))
    li_top = dict((a, t) for t, a in list_items(allc, old_p, car="corsa", config="top", refs=refs))
    got_old = {c["id"]: combo_best(old_p, c)[1] for c in every if c["key"] == kf(c["id"])}
    run_old = ChallengeRun(of_("brake_100"), strip, {}, old_p)
    _i, secs_old, _n = detail(of_("brake_100"), None, [], old_p)
    rep("a returning player's stars saved per challenge id (before task 44) count on the "
        "Corsa / FULL WING, judged by today's thresholds and capped at the old stars; "
        "other cars and configs start clean; a removed challenge is never read; nothing "
        "is written",
        got_old == {"brake_100": 2, "lap_arena": 1, "skid_dry": 1, "brake_wet": 0,
                    "skid_wet": 0, "airbrake_150": 0, "lap_open": 0, "lap_wet": 0}
        and combo_best(old_p, of_("brake_wet")) == (50.0, 0)
        and total_stars(old_p, allc, refs)[0] == 4 and "drag_400" not in load_all()
        and all(combo_best(old_p, c) == (None, 0) for c in every
                if (c["car"], c["config"]) != OLD_COMBO)
        and "[**-]" in li_old["ch:brake_100"] and "unlimited [***]" in li_old["ch:brake_100"]
        and "[---]" in li_top["ch:brake_100"] and "unlimited" not in li_top["ch:brake_100"]
        and (run_old.best, run_old.stars, run_old.official) == (b_old, 2, (b_old, 2))
        and dict(secs_old)["YOURS"][0] == ("best", f"{fmt_value('stop_distance', b_old)}  [**-]")
        and json.dumps(Progress(old_p.path).data, sort_keys=True) == on_disk,
        f"{got_old} | {li_old['ch:brake_100']!r} | {li_top['ch:brake_100']!r}")
    #  ... until a result counts: the combo's own entry, the old one left
    tk_o, ck_o, ek_o, sk_o = split_key(run_old.ch["class"])
    run_old.meter._done(b_old - 0.1)       # a better stop, still 2 stars
    run_old._collect(SimpleNamespace(track=SimpleNamespace(name=tk_o), settings=SimpleNamespace(
        car=ck_o, engine=ek_o, wet=sk_o), global_wet=(MU_WET_SCALE if sk_o == "all" else 1.0)))
    d_old = Progress(old_p.path).section(SECTION)
    rep("a better result on the carried combo saves the combo's own entry; the old one stays",
        "NEW BEST" in run_old.note and d_old.get(kf("brake_100")) == dict(best=b_old - 0.1, stars=2)
        and d_old.get("brake_100") == {"best": b_old, "stars": 3}
        and combo_best(Progress(old_p.path), of_("brake_100")) == (b_old - 0.1, 2),
        f"{run_old.note} | {d_old.get(kf('brake_100'))}")
    #  round 3: a stop's 3 stars saved by its distance before the board keep
    #  their 3; a shorter stop off the board saves its best, the 3 stay
    b3_p = Progress(os.path.join(tmp, "board3.json"))
    b3_p.section(SECTION)[kf("brake_100")] = dict(best=tb[2], stars=3)
    b3_p.save(SECTION)
    run_b3 = ChallengeRun(of_("brake_100"), strip, clean, b3_p)
    before3 = (run_b3.best, run_b3.stars)
    run_b3.meter._done(tb[2] - 0.2)
    run_b3._collect(SimpleNamespace(track=SimpleNamespace(name=tk_o), global_wet=1.0,
                                    s=run_b3.board[1] + 3.0,       # the nose well past it
                                    settings=SimpleNamespace(car=ck_o, engine=ek_o, wet=sk_o)))
    rep("the board keeps saved progress: a stop's 3 stars saved by its distance before the "
        "board stay 3; a shorter stop off the board is 2 stars, saves its best, the 3 stay",
        before3 == (tb[2], 3) and run_b3.last == (tb[2] - 0.2, 2)
        and Progress(b3_p.path).section(SECTION)[kf("brake_100")] == dict(best=tb[2] - 0.2,
                                                                          stars=3),
        f"{before3} -> {run_b3.last}, saved {Progress(b3_p.path).section(SECTION)[kf('brake_100')]}")
    got_t, of_t = total_stars(Progress(os.path.join(tmp, "none.json")), allc, refs)
    rep("every car x config: the available combos, 3 stars each (the list's '(all: ...)')",
        (got_t, of_t) == (0, 3 * n_ok), f"{got_t} of {of_t}")
    #  task 45: the pause page's row counts the pick the list opens on, as
    #  the list's subtitle does (`pick_stars`), not every car x config ('1 of
    #  456 stars'): 3 on the Corsa / FULL WING's stop, 2 on its TOP's
    pick_p = Progress(os.path.join(tmp, "pick.json"))
    pick_p.section(SECTION).update({kf("brake_100"): {"best": tb[2], "stars": 3},
                                    combo_key("brake_100", "corsa", "top"): {"best": tb[1],
                                                                              "stars": 2}})
    n_of = lambda cfg: sum(c["available"] for c in every                  # noqa: E731
                           if (c["car"], c["config"]) == ("corsa", cfg))
    rows_p = (menu_row(Progress(os.path.join(tmp, "none.json"))),
              menu_row(pick_p, car="corsa", config="full"),
              menu_row(pick_p, car="corsa", config="top"))
    rep("the pause page's row: the stars of the picked car + wings, out of 3 per challenge "
        "that pick can drive (the list's subtitle's numbers), not every combo's",
        rows_p == (f"Challenges: 0 of {3 * n_of('full')} stars (this car + wings)",
                   f"Challenges: 3 of {3 * n_of('full')} stars (this car + wings)",
                   f"Challenges: 2 of {3 * n_of('top')} stars (this car + wings)")
        and pick_stars(pick_p, "corsa", "top", allc, refs) == (2, 3 * n_of("top"))
        and total_stars(pick_p, allc, refs) == (5, 3 * n_ok) and 3 * n_of("full") < 3 * n_ok,
        " | ".join(rows_p))
    #  task 45: with a challenge running, the pause row is its title alone
    #  (the list's subtitle repeated a 'end it, or another'); the list's End
    #  row is short, never the widest item (its long label pushed the help
    #  column off the panel), and the help's 'end' row says what comes back;
    #  1 star is the first number, not 'done' (a finished stop can score 0)
    li_run = list_items(allc, old_p, run_old, car="corsa", config="full", refs=refs)
    help_ = dict(LIST_HELP[0][1])
    rep("a running challenge: the pause row names it, the list marks it and ends with a short "
        "End row and Back; the help says what ending gives back and what 1 star is",
        menu_row(old_p, run_old) == f"Challenges: {run_old.ch['title']} running"
        and li_run[-2:] == [("End the challenge", "ch_end"), ("Back", "ch_back")]
        and "<- now" in dict((a, t) for t, a in li_run)["ch:brake_100"]
        and all(len(li_run[-2][0]) < len(t) for t, a in li_run if a.startswith("ch:"))
        and help_.get("end") == "your map, car, engine and surface come back"
        and help_["stars"].startswith("1 = the first number") and "done" not in help_["stars"]
        and help_.get("stops") == "3rd star: stop the nose within 1 m of the board",
        f"{menu_row(old_p, run_old)!r} | {li_run[-2][0]!r}")
    #  task 41: an UNLIMITED build (a wing past the car's span limit) is still
    #  judged by the challenge's rules, but its best and stars go to their own
    #  section, shown in their own spot; the official ones never move
    import cars as _cars
    huge = lib.wings["rear-s1223"].copy(name="huge-top", builtin=False)
    huge.span = 2.6                         # the Corsa's top limit is 1.2 x 1.646 = 1.975 m
    lib.save_wing(huge)
    corsa = _cars.get("corsa")
    st_u = build_stats(dict(ref_build("tall"), slots=dict(ref_build("tall")["slots"],
                                                          top=dict(ref_build("tall")["slots"]["top"],
                                                                   wing="huge-top"))),
                       lib, corsa, 0)
    st_o = build_stats(ref_build("tall"), lib, corsa, 0)
    rep("build_stats says whether the build is Unlimited on the car, and why",
        st_u["unlimited"] and [o["slot"] for o in st_u["over_limits"]] == ["top"]
        and not st_o["unlimited"] and st_o["over_limits"] == [],
        _over_text(st_u["over_limits"]))
    up = Progress(os.path.join(tmp, "unl.json"))
    up.section(SECTION)[ch["key"]] = dict(best=worse(t1), stars=0)
    up.save(SECTION)
    usim = SimpleNamespace(track=SimpleNamespace(name=tk), settings=SimpleNamespace(
        car=ck, engine=ek, wet=sk), global_wet=(MU_WET_SCALE if sk == "all" else 1.0),
        s=on_board)                        # the nose on the board (round 3)
    ru = ChallengeRun(ch, trl, dict(clean, unlimited=True, over_limits=st_u["over_limits"]), up)
    ru.meter._done(t3)
    ru._collect(usim)
    disk = Progress(up.path)
    ov = ru.overlay(SimpleNamespace(**dict(vars(usim), veh=SimpleNamespace(u=0.0, v=0.0), s=0.0)))
    rep("an Unlimited run: counted in its own section, the official best untouched",
        ru.last == (t3, 3) and "UNLIMITED" in ru.note
        and disk.section(SECTION_UNLIMITED)[ch["key"]] == dict(best=t3, stars=3)
        and disk.section(SECTION)[ch["key"]] == dict(best=worse(t1), stars=0)
        and total_stars(disk, allc, fake)[0] == 0,
        f"{ru.note}  |  official {disk.section(SECTION)[ch['key']]}")
    li = dict((a, t) for t, a in list_items(allc, disk, car="mx5", config="top", refs=fake))
    _i, secs_u, _n = detail(ch, dict(clean, unlimited=True, over_limits=st_u["over_limits"]),
                            [], disk)
    rep("the box, the list and the detail page show the Unlimited spot beside the official",
        "UNLIMITED" in ov["head"] and "official" in ov["head"] and "UNLIMITED" in ov["text"]
        and "unlimited [***]" in li[f"ch:{ch['id']}"]
        and ("unlimited", f"{fmt_value(ch['goal']['metric'], t3)}  [***]  not official")
        in dict(secs_u)["YOURS"] and any(k == "UNLIMITED" for k, _ in dict(secs_u)["RULES"])
        and ov["text"].startswith(f"{_cars_.car_name('mx5')}, ONLY TOP. "),
        ov["head"])
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    if verbose:
        print(f"challenges self-check: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(main())
