"""drive/medals.py -- medal times per class (task 21, plan decision D4).

A medal is a promise that a time is REACHABLE in this class, so every
threshold here is DERIVED from a lap that was actually driven, never typed
in. The AUTHOR time of a class is the best valid flying lap any of the
reference drivers sets, headless, in the class's stock car; the three
medals below it are fixed multiples of it:

    gold = author x GOLD_X (1.02)   silver = x SILVER_X (1.06)   bronze = x BRONZE_X (1.12)

A class nobody laps has NO medals ("--" on the screen) rather than an
invented time: a gold nobody can prove is reachable is worse than none.

The class is records.py's (plan D1): `track|car|engine|surface`. Seven maps
(four circuits, the open map, the skidpad, the dragstrip) x the cars of
`cars.CAR_ORDER` (four since task 46: the Corsa, the rally Escort, the 540i
and the Renault Express -- the MX-5 and the Citaro bus are retired, their
classes no longer listed) x three engines x three surfaces = 252 classes,
every one of which has an entry -- an author time, or a reason there is
none. The DRAGSTRIP has no lap (records.py excludes it for the same
reason), so its 36 classes say so. A class the table does not hold yet (a
car added since the last `--build`) simply has no medals at runtime --
`targets` / `medal_for` return None, the pages show the em dash -- and this
module's self-check names the cars the table is missing.

How a reference lap is driven
-----------------------------
Exactly the way a player's is timed, so that "beat the author" compares like
with like:

* the STOCK build of the class's car: `cars.get(car)`, no wings, no ballast,
  `power_scale = ENGINE_SCALE[engine]`, the car's own `mu_scale`;
* the class's track as a session builds it (`trk.make_track`, the STANDARD
  skidpad -- records.SKIDPAD_STD -- surfaces off only for surface 'none'),
  the global wet for surface 'all', dt = DT_PHYS;
* a standing start on the line in 1st (`trk.start_pose(tr, 0.0)`), like a
  session start, so the first 'lap' event is the FLYING lap after an out-lap
  -- the lap a player's HUD first times;
* timed by a `drive.drive.Sim` and its own `LapTimer`, the HUD's machinery: a
  lap counts only when the 'lap' event arrives with `sim.lap.lap_valid`, it
  went ROUND (every sector line in order and records.LAP_MIN_FRACTION of the
  length -- a player record's own rule) and it has NO SPIN in it (|beta|
  under SPIN_BETA_DEG throughout): a lap that stayed on the tarmac through a
  180 would make a gold a spin away.

The drivers: `LapDriver` (the scripted quasi-steady driver, through
`ScriptedInput`, at margins 0.90 / 0.80 / 0.70 / 0.60 -- the careful ones are
what lap the power-heavy wet classes at all), the `drive.ml` ANCHOR (`Policy()`, theta = 0: the
hand-written driver the policies are residuals on) and every BUNDLED
checkpoint (git-tracked: a swarm saves players' bots in the same directory,
and a bot on one machine must not set a time the repo cannot reproduce) whose
meta names this car and this track -- the two policy kinds
driven exactly as `drive.drive._ml_input` drives them (`observe` ->
`Policy.controls` with the car's `driver_trim`). Each runs twice, aids OFF
and aids ON (ABS + TC, the interactive defaults), and the entry records which
one set the time. A run stops after FLYING_LAPS valid flying laps, when the
car is LOST (off the map, or spun to a stop, for LOST_S; or under STALL_M of
progress in STALL_S) or at T_MAX.
Every run is kept in the entry's `tried` list, so a missing medal comes with
the evidence for it.

Two classes whose PHYSICS is identical are driven once and share the result:
the skidpad and the open map have no wet patches to switch off, so their
'none' and 'patch' classes build the same track at the same grip, bit for
bit. The entry says so (`ran_as`).

The author lap's POSE TRACE (the pre-race screen and task 22's ghost 2) is
recorded while that lap is driven, at REF_TRACE_HZ, in records.TRACE_COLS
order, with records.encode_trace, into REF_PATH.

Staleness
---------
`inputs_hash` is a sha256 over what a medal time is a function of: every
track definition (both surface settings), every car spec (not its display
name: cars.DISPLAY_FIELDS), the engine scales,
the surface modes, the wet scale, the off-track grip, DT_PHYS, the three
multipliers -- and the REFERENCE DRIVERS: LapDriver's margins, the lap rules,
the stop rules and every bundled checkpoint file's sha256. NOT covered: the
vehicle / powertrain physics and LapDriver's code (a physics change must be
followed by a rebuild by hand). `--only` refuses to run on a stale table: a
partial build would stamp it current with the other maps dropped. Floats
are hashed at 10 significant digits, so a last-bit difference between two
machines' track integration is not "stale" while any real edit is. The
renderer's paint (a patch's colour and label) is left out: repainting a wet
patch does not move a lap. A stale table still loads and still shows medals;
the self-check prints the regenerate command as a SOFT fail.

Import rule: numpy and records.py at module level. `drive.drive`, `track`
and `cars` are imported inside functions (drive.drive will import this
module for the HUD), `drive.ml` only inside the build's worker (an OPTIONAL
sub-package: a missing one costs the policy drivers, nothing else). Never
pygame. The runtime API reads two JSON files, caches them by mtime, and
never raises.

    python3 -m drive.medals                          the self-check
    python3 -m drive.medals --build [--workers N] [--only TRACK[,TRACK]]
    python3 -m drive.medals --show                   the table, one row a class
"""

from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

from . import records as rec

# ==================================================================== #
#  CONSTANTS                                                           #
# ==================================================================== #
#: plan D4: the medals are these multiples of the author time
GOLD_X, SILVER_X, BRONZE_X = 1.02, 1.06, 1.12
#: best first; `records.RecordBook.set_best_medal` takes this as its order
MEDALS = ("author", "gold", "silver", "bronze")
MULT = {"author": 1.0, "gold": GOLD_X, "silver": SILVER_X, "bronze": BRONZE_X}

MEDALS_KIND = "carsim-medals-1"
REF_KIND = "carsim-reference-laps-1"
_HERE = os.path.dirname(os.path.abspath(__file__))
#: package-relative, never cwd-relative: the table ships with the code
DATA_PATH = os.path.join(_HERE, "data", "medals.json")
REF_PATH = os.path.join(_HERE, "data", "reference_laps.json")
CKPT_DIR = os.path.join(_HERE, "ml", "checkpoints")

#: the author lap's trace: enough for a ghost (the renderer interpolates) at
#: two fifths of the records' 50 Hz; ~12 KB of JSON an arena-length lap
#: (the six lap maps' 162 are 1.8 MB)
REF_TRACE_HZ = 20
#: a run stops once it has this many VALID flying laps
FLYING_LAPS = 2
#: sim seconds a run may take: the out-lap plus FLYING_LAPS flying laps with
#: margin, at the slowest class (the 75 hp car on the all-wet surface). The
#: circuits after the arena take its 300 s pro rata to their length, rounded
#: up and never under 1.1x the slowest run measured there (LapDriver 0.60,
#: all-wet, the 540i with aids on): Linden 252 s, Kestrel 333 s, Ashdown 278 s.
#: The oval (task 46) takes the same rule, 300 s x 1902.5 / 1249.2 = 457 ->
#: 460 s, far over its slowest run (it laps fastest of all per metre):
#: LapDriver 0.60 all-wet, aids on, 241 / 243 / 248 s (Corsa / 540i / Express)
T_MAX = {"skidpad": 110.0, "arena": 300.0, "open": 420.0,
         "linden": 280.0, "kestrel": 460.0, "ashdown": 340.0, "fairfield": 460.0}
#: LOST: further than this outside the ribbon's edge, or spun (|beta| over
#: LOST_BETA rad below LOST_V m/s), for LOST_S seconds -- the race bot's
#: respawn test (`drive.drive.Rival`), which here ends the run instead
LOST_N_EXTRA = 4.0
LOST_BETA = 1.05
LOST_V = 4.0
LOST_S = 3.0
#: LOST also when the car makes no progress: under STALL_M of centreline in
#: STALL_S seconds (a car spinning donuts passes back under LOST_BETA every
#: rotation and would otherwise burn the whole T_MAX)
STALL_S = 12.0
STALL_M = 30.0
#: the two assist settings every driver runs with: name, ABS, TC
AIDS = (("off", False, False), ("on", True, True))
#: the scripted driver's margins: its default (the acceptance lap's) and
#: three more careful ones. One margin is not enough on the power-heavy wet
#: classes: at 0.90 the 2x MX-5 on the all-wet skidpad spins every lap and the
#: only valid one it keeps is a 43.3 s lap with a spin in it (the stock MX-5
#: laps in 20.5 s), and the 2x 540i on the all-wet arena never finishes one;
#: at 0.60 / 0.80 they lap in 25.1 / 78.3 s. A careful driver is still
#: the same LapDriver, so the author time stays "the best a reference driver
#: actually drove".
LAPDRIVER_MARGIN = 0.90
LAPDRIVER_MARGINS = (0.90, 0.80, 0.70, 0.60)
#: a reference lap must be a lap ROUND (records.LAP_MIN_FRACTION of the
#: length, the player's rule) with no spin in it: a lap that stayed on the
#: tarmac through a 180 is valid for a player, but as the author time it
#: promises a gold that is a spin away
SPIN_BETA_DEG = 35.0

REASON_NO_MAP = "no lap on this map"
REASON_NO_REF = "no reference lap"
REASON_NOT_BUILT = "not built yet: python3 -m drive.medals --build"
NONE_MARK = "\u2014"                 # an em dash: 'no medals', never a time
REGEN = "python3 -m drive.medals --build"


# ==================================================================== #
#  THE CLASSES                                                         #
# ==================================================================== #
def class_keys() -> list:
    """Every class, in menu order: TRACK_ORDER x CAR_ORDER x ENGINE_MODES x
    SURFACE_MODES (252: 7 maps x 4 cars x 3 x 3, task 46)."""
    import cars
    from . import track as trk
    from .drive import ENGINE_MODES, SURFACE_MODES
    return [rec.class_key(t, c, e, s) for t in trk.TRACK_ORDER for c in cars.CAR_ORDER
            for e in ENGINE_MODES for s in SURFACE_MODES]


def _build_track(track: str, surface: str):
    from . import track as trk
    return trk.make_track(track, rec.SKIDPAD_STD[0], rec.SKIDPAD_STD[1],
                          surfaces=(surface != "none"))


# ==================================================================== #
#  STALENESS                                                           #
# ==================================================================== #
_PAINT = ("colour", "label")           # the renderer's, not the physics'


def _canon(x):
    """A value -> plain JSON with floats at 10 significant digits."""
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        return float(f"{f:.10g}") if math.isfinite(f) else repr(f)
    if isinstance(x, np.ndarray):
        return _canon(x.tolist())
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return _canon(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    return repr(x)


def _unpainted(d):
    return {k: v for k, v in d.items() if k not in _PAINT} if isinstance(d, dict) else d


def _track_desc(tr) -> dict:
    """What a lap on this track is a function of."""
    dc = (lambda o: dataclasses.asdict(o) if dataclasses.is_dataclass(o) else o)
    return dict(name=tr.name, closed=bool(tr.closed), width=float(tr.width),
                length=float(tr.length), origin=list(tr.origin), heading0=float(tr.heading0),
                segs=[dc(s) for s in tr.segs], sector_s=[float(s) for s in tr.sector_s],
                surfaces=[_unpainted(dc(p)) for p in tr.surfaces],
                areas=[_unpainted(dc(a)) for a in tr.areas])


def _hash(obj) -> str:
    blob = json.dumps(_canon(obj), sort_keys=True, default=repr, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def hash_inputs() -> dict:
    """The payload `current_hash` digests (public so a diff can be read)."""
    import cars
    from . import track as trk
    from .drive import ENGINE_SCALE, SURFACE_MODES, MU_WET_SCALE, DT_PHYS
    tracks = {f"{name}|surfaces={s}": _track_desc(trk.make_track(name, rec.SKIDPAD_STD[0],
                                                                   rec.SKIDPAD_STD[1], surfaces=s))
              for name in trk.TRACK_ORDER for s in (True, False)}
    ck = {}
    for p in bundled_paths():
        with open(p, "rb") as fh:
            ck[os.path.basename(p)] = hashlib.sha256(fh.read()).hexdigest()
    return dict(tracks=tracks,
                #  a car's display name (cars.DISPLAY_FIELDS) is words on
                #  screen, not a medal input (task 49: the cars renamed)
                cars={k: {f: x for f, x in dataclasses.asdict(v).items()
                          if f not in cars.DISPLAY_FIELDS}
                      for k, v in sorted(cars.CARS.items())},
                engine_scale=dict(ENGINE_SCALE), surface_modes=list(SURFACE_MODES),
                mu_wet_scale=float(MU_WET_SCALE), dt_phys=float(DT_PHYS),
                off_track=dict(mu=float(trk.MU_OFF_TRACK), crr=float(trk.CRR_OFF_SCALE)),
                skidpad=list(rec.SKIDPAD_STD),
                multipliers=dict(gold=GOLD_X, silver=SILVER_X, bronze=BRONZE_X),
                #  the reference drivers ARE an input: who drives, how
                #  carefully, what a counted lap is, and the bots' own files
                drivers=dict(lapdriver_margins=list(LAPDRIVER_MARGINS),
                             spin_beta_deg=SPIN_BETA_DEG, full_lap=rec.LAP_MIN_FRACTION,
                             flying_laps=FLYING_LAPS, t_max=dict(T_MAX), aids=list(AIDS),
                             lost=[LOST_N_EXTRA, LOST_BETA, LOST_V, LOST_S, STALL_S, STALL_M],
                             checkpoints=ck))


_HASH_CACHE: list = []


def current_hash() -> str:
    """sha256 of `hash_inputs()`; computed once per process."""
    if not _HASH_CACHE:
        _HASH_CACHE.append(_hash(hash_inputs()))
    return _HASH_CACHE[0]


def is_stale(table=None) -> bool:
    """True when the table was built from other tracks / cars / engines than
    these (or is empty)."""
    table = load() if table is None else table
    return not table.get("inputs_hash") or table.get("inputs_hash") != current_hash()


# ==================================================================== #
#  THE RUNTIME API: cheap, cached, never raises                        #
# ==================================================================== #
_CACHE: dict = {}                      # abspath -> (stamp, table)
_TRACE_CACHE: dict = {}                # (abspath, stamp, key) -> ndarray
_NOTED: set = set()


def _note_once(path: str, why: str) -> None:
    k = (os.path.abspath(path), why)
    if k not in _NOTED:
        _NOTED.add(k)
        print(f"medals: {os.path.basename(path)} ignored ({why}); no medals shown")


def _stamp(path: str):
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _empty(note: str = "") -> dict:
    return dict(kind=MEDALS_KIND, inputs_hash=None, classes={}, empty=True, note=note,
                thresholds=dict(gold=GOLD_X, silver=SILVER_X, bronze=BRONZE_X))


def _read_json(path: str, kind: str):
    """(dict, None) or (None, why)."""
    try:
        with open(path) as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError) as exc:
        return None, f"unreadable: {type(exc).__name__}"
    if not isinstance(raw, dict) or raw.get("kind") != kind:
        got = raw.get("kind") if isinstance(raw, dict) else type(raw).__name__
        return None, f"kind {got!r}, want {kind!r}"
    return raw, None


def load(path: str = DATA_PATH) -> dict:
    """The medal table. Cached by mtime; a missing, corrupt or foreign file
    is an EMPTY table (a note is printed once), never an exception."""
    ap = os.path.abspath(path)
    st = _stamp(ap)
    hit = _CACHE.get(ap)
    if hit is not None and hit[0] == st:
        return hit[1]
    try:
        raw, why = _read_json(ap, MEDALS_KIND)
        if raw is not None and not isinstance(raw.get("classes"), dict):
            raw, why = None, "no 'classes' table"
    except Exception as exc:               # noqa: BLE001 -- the HUD must not die here
        raw, why = None, f"{type(exc).__name__}: {exc}"
    if raw is None:
        _note_once(ap, why)
        raw = _empty(why)
    _CACHE[ap] = (st, raw)
    return raw


def _good(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0.0


def targets(key: str, table=None):
    """{author, gold, silver, bronze} in seconds, or None when the class has
    no reference lap (or no entry)."""
    try:
        table = load() if table is None else table
        e = table.get("classes", {}).get(key)
        if not isinstance(e, dict) or not _good(e.get("author")):
            return None
        out = {m: e.get(m) for m in MEDALS}
        return {m: float(v) for m, v in out.items()} if all(_good(v) for v in out.values()) else None
    except Exception:                      # noqa: BLE001
        return None


def medal_for(key: str, t, table=None):
    """The best medal a lap of `t` s earns in the class: 'author' | 'gold' |
    'silver' | 'bronze' | None. `t <= threshold` earns it."""
    tg = targets(key, table)
    if tg is None or not _good(t):
        return None
    for m in MEDALS:
        if float(t) <= tg[m]:
            return m
    return None


def reference_driver(key: str, table=None):
    """The name of the driver that set the author time, or None."""
    try:
        table = load() if table is None else table
        e = table.get("classes", {}).get(key)
        if isinstance(e, dict) and _good(e.get("author")) and e.get("driver"):
            return str(e["driver"])
    except Exception:                      # noqa: BLE001
        pass
    return None


def _load_ref(path: str = REF_PATH) -> dict:
    ap = os.path.abspath(path)
    st = _stamp(ap)
    hit = _CACHE.get(ap)
    if hit is not None and hit[0] == st:
        return hit[1]
    try:
        raw, why = _read_json(ap, REF_KIND)
        if raw is not None and not isinstance(raw.get("laps"), dict):
            raw, why = None, "no 'laps' table"
    except Exception as exc:               # noqa: BLE001
        raw, why = None, f"{type(exc).__name__}: {exc}"
    if raw is None:
        _note_once(ap, why)
        raw = dict(kind=REF_KIND, inputs_hash=None, laps={}, empty=True, note=why)
    _CACHE[ap] = (st, raw)
    return raw


def reference_trace(key: str, path: str = REF_PATH):
    """The author lap as an (n, 9) float array in records.TRACE_COLS order
    (`[:, :8]` is a `_Replay` trace, `[:, 8]` the progress from the line),
    or None. Decoded once and cached; the array is read-only."""
    try:
        ap = os.path.abspath(path)
        ref = _load_ref(ap)
        ck = (ap, _stamp(ap), key)
        if ck in _TRACE_CACHE:
            return _TRACE_CACHE[ck]
        d = ref.get("laps", {}).get(key)
        out = None
        if isinstance(d, dict):
            a = rec.decode_trace(d)
            if a.ndim == 2 and a.shape[0] > 0 and a.shape[1] == len(rec.TRACE_COLS):
                a.setflags(write=False)
                out = a
        _TRACE_CACHE[ck] = out
        return out
    except Exception:                      # noqa: BLE001
        return None


def fmt_time(t) -> str:
    """m:ss.mmm; the em dash for no time."""
    return rec.fmt_time(t) if _good(t) else NONE_MARK


def medal_label(m) -> str:
    return m.upper() if m in MEDALS else NONE_MARK


def targets_line(key: str, table=None) -> str:
    """'AUTHOR 1:00.835  GOLD 1:02.052  SILVER 1:04.485  BRONZE 1:08.135', or
    the em dash for a class with no reference lap."""
    tg = targets(key, table)
    if tg is None:
        return f"medals {NONE_MARK}"
    return "  ".join(f"{m.upper()} {fmt_time(tg[m])}" for m in MEDALS)


# ==================================================================== #
#  THE BUILD                                                           #
# ==================================================================== #
def _lapdriver_name(margin: float) -> str:
    """'lapdriver' at the default margin, 'lapdriver-0.80' at another."""
    return "lapdriver" if abs(margin - LAPDRIVER_MARGIN) < 1e-9 else f"lapdriver-{margin:.2f}"


def _lapdriver_margin(name: str) -> float:
    return LAPDRIVER_MARGIN if name == "lapdriver" else float(name.split("-", 1)[1])


def _ckpt_matches(meta: dict, car: str, track: str) -> bool:
    if (meta.get("car") or "corsa") != car:
        return False
    tracks = [t.strip() for t in str(meta.get("track") or "").split(",") if t.strip()]
    return track in tracks


def bundled_paths() -> list:
    """The checkpoint files that SHIP: the git-tracked ones under CKPT_DIR.
    The same directory is where a swarm saves the bots a player breeds
    (`swarm_<name>.json`), and a bot on one machine must not set an author
    time the repo cannot reproduce. Without git (an installed game) every
    file there is taken, with a note -- only the build and the self-check's
    staleness test ask."""
    import subprocess
    allp = sorted(glob.glob(os.path.join(CKPT_DIR, "*.json")))
    try:
        r = subprocess.run(["git", "ls-files", "--", CKPT_DIR], cwd=os.path.dirname(_HERE),
                           capture_output=True, text=True, timeout=20)
        if r.returncode == 0:
            root = os.path.dirname(_HERE)
            tracked = {os.path.abspath(os.path.join(root, ln.strip()))
                       for ln in r.stdout.splitlines() if ln.strip()}
            return [p for p in allp if os.path.abspath(p) in tracked]
    except Exception:                      # noqa: BLE001 -- no git: every file
        pass
    _note_once(CKPT_DIR, "no git: every checkpoint file counts as bundled")
    return allp


def checkpoints() -> list:
    """[(name, path, meta)] of the bundled `drive.ml` checkpoints, by name.
    Read as plain JSON: nothing of `drive.ml` is imported to list them."""
    out = []
    for p in bundled_paths():
        try:
            with open(p) as fh:
                meta = json.load(fh).get("meta") or {}
        except (OSError, ValueError, AttributeError):
            meta = {}
        out.append((os.path.basename(p)[:-5], p, meta if isinstance(meta, dict) else {}))
    return out


def drivers_for(car: str, track: str, ckpts=None) -> list:
    """[(name, path)]: the scripted driver, the anchor, then every checkpoint
    bred or trained for this car on this track."""
    ckpts = checkpoints() if ckpts is None else ckpts
    out = [(_lapdriver_name(m), None) for m in LAPDRIVER_MARGINS] + [("anchor", None)]
    for name, path, meta in ckpts:
        if _ckpt_matches(meta, car, track):
            out.append((name if name not in ("lapdriver", "anchor") else f"ckpt:{name}", path))
    return out


def _physics_key(track: str, surface: str) -> str:
    """Two classes with the same key drive the same track at the same grip."""
    from .drive import MU_WET_SCALE
    tr = _build_track(track, surface)
    return _hash(dict(track=_track_desc(tr), gw=rec.surface_global_wet(surface, MU_WET_SCALE)))


def _driver_input(driver: str, path, tr, car, cfg, gw: float):
    """The driver in the seat, as a `ScriptedInput`."""
    from .drive import ScriptedInput, LapDriver
    if driver.startswith("lapdriver"):
        return ScriptedInput(LapDriver(tr, margin=_lapdriver_margin(driver), global_wet=gw, car=car))
    #  a drive.ml policy, driven exactly as `drive.drive._ml_input` drives
    #  one: `observe(veh, track)` -> `Policy.controls` with the car's trim
    from .ml.env import observe
    from .ml.policy import Policy
    from .ml.baseline import driver_trim
    from .vehicle import Controls
    pol = Policy() if path is None else Policy.load(path)
    tm = driver_trim(car, float(cfg.mu_scale))

    def fn(t, veh, track):
        return pol.controls(observe(veh, track), Controls, lock_rad=tm["lock_rad"],
                            wheelbase=tm["wheelbase"], ay_plan=tm["ay_plan"], k_us=tm["k_us"])

    return ScriptedInput(fn)


def _drive(job: dict) -> dict:
    import cars
    from . import track as trk
    from .drive import Sim, DT_PHYS, MU_WET_SCALE, ENGINE_SCALE
    from .vehicle import Vehicle, VehicleConfig

    track, surface = job["track"], job["surface"]
    car = cars.get(job["car"])
    tr = _build_track(track, surface)
    gw = rec.surface_global_wet(surface, MU_WET_SCALE)
    cfg = VehicleConfig(wing="off", mu_scale=float(car.mu_scale),
                        power_scale=float(ENGINE_SCALE[job["engine"]]),
                        abs_on=bool(job["abs"]), tc_on=bool(job["tc"]))
    veh = Vehicle(car, cfg)
    x, y, psi = trk.start_pose(tr, 0.0)
    veh.reset(x, y, psi, V=0.0, gear=1)      # a session's standing start
    inp = _driver_input(job["driver"], job["path"], tr, car, cfg, gw)
    dt = DT_PHYS
    sim = Sim(veh, tr, inp, dt=dt, global_wet=gw)
    sim.track_radius, sim.track_cw = rec.SKIDPAD_STD

    L = float(tr.length)
    half = 0.5 * float(tr.width)
    stride = max(1, int(round(1.0 / (REF_TRACE_HZ * dt))))
    n_max = int(round(T_MAX[track] / dt))
    laps: list = []                    # valid flying laps, s
    invalid = 0
    best_t, best_rows = math.inf, None
    lap = None                         # the lap being driven
    lost_s, lost_at, why = 0.0, None, ""

    def row(lp):
        v = sim.veh
        ds = float(sim.s) - lp["s_last"]
        if ds < -0.5 * L:
            ds += L
        elif ds > 0.5 * L:
            ds -= L
        if abs(ds) <= 30.0:
            lp["prog"] += ds
        lp["s_last"] = float(sim.s)
        lp["rows"].append((float(sim.t) - lp["t_cross"], v.x, v.y, v.psi, v.u,
                           float(v.wing_deploy), float(v.wing_side),
                           float(getattr(v, "top_deploy", 0.0)), lp["prog"]))

    spin_rad = math.radians(SPIN_BETA_DEG)
    spun = short = 0
    n_sec = len(tr.sector_s) if len(tr.sector_s) > 1 else 0
    stall_n = int(round(STALL_S / dt))
    s_hist = []                        # (n, unwrapped progress) every second
    prog_all, s_prev_all = 0.0, float(sim.s)
    for _ in range(n_max):
        sim.step_physics(dt)
        if lap is not None:
            b = abs(float(veh.beta))
            if b > lap["beta_max"]:
                lap["beta_max"] = b
        crossed = None
        for e in sim.events_log:
            if e[0] == "sector" and lap is not None and int(e[1]) != (n_sec - 1 if n_sec else -1):
                #  every sector line in order, as a player's record needs;
                #  the last sector's event comes with the lap event below
                if int(e[1]) != lap["next_sec"]:
                    lap["out_of_order"] = True
                lap["next_sec"] = int(e[1]) + 1
            if e[0] == "lap":
                if lap is not None:
                    row(lap)           # the close, at the end of the crossing step
                if not sim.lap.lap_valid:
                    invalid += 1
                elif (lap is None or lap["prog"] < rec.LAP_MIN_FRACTION * L
                      or lap.get("out_of_order") or (n_sec and lap["next_sec"] != n_sec - 1)):
                    short += 1         # not a lap ROUND (the player's rule)
                elif lap["beta_max"] > spin_rad:
                    spun += 1          # a spin in it: not an author lap
                else:
                    t_lap = float(e[3])
                    laps.append(t_lap)
                    if t_lap < best_t:
                        best_t, best_rows = t_lap, lap["rows"]
            if e[0] in ("start", "lap"):
                crossed = float(e[2])
        sim.events_log.clear()
        if crossed is not None:
            s0 = float(sim.s) - (L if float(sim.s) > 0.5 * L else 0.0)
            lap = dict(t_cross=crossed, n0=sim.n, rows=[], prog=s0, s_last=float(sim.s),
                       beta_max=0.0, next_sec=0)
            row(lap)
        elif lap is not None and (sim.n - lap["n0"]) % stride == 0:
            row(lap)
        if len(laps) >= FLYING_LAPS:
            break
        if sim.n % 1000 == 0:          # progress once a second: a stalled car is lost
            ds = float(sim.s) - s_prev_all
            ds += L if ds < -0.5 * L else (-L if ds > 0.5 * L else 0.0)
            prog_all += ds if abs(ds) < 100.0 else 0.0
            s_prev_all = float(sim.s)
            s_hist.append((sim.n, prog_all))
            old_ = [p for n_, p in s_hist if n_ <= sim.n - stall_n]
            if old_ and prog_all - old_[-1] < STALL_M:
                why = f"lost at {sim.t:.0f} s (no progress)"
                break
        lost = (abs(sim.n_lat) > half + LOST_N_EXTRA
                or (abs(veh.beta) > LOST_BETA and math.hypot(veh.u, veh.v) < LOST_V))
        if lost:
            if lost_s == 0.0:
                lost_at = sim.t
            lost_s += dt
            if lost_s > LOST_S:
                why = f"lost at {lost_at:.0f} s"
                break
        else:
            lost_s = 0.0

    if not laps and not why:
        why = (f"no valid lap in {T_MAX[track]:.0f} s"
               + (f" ({spun} spun)" if spun else "") + (f" ({short} not round)" if short else ""))
    trace = None
    if best_rows:
        trace = rec.encode_trace(best_rows)
        trace["hz"] = REF_TRACE_HZ
    return dict(best=(best_t if laps else None), laps=laps, invalid=invalid, spun=spun,
                short=short, why=why, t_end=float(sim.t), trace=trace)


def _run_job(job: dict) -> dict:
    """One (physics class, driver, aids) run, in a pool worker. Plain data
    back; a failure is a result, never an exception."""
    w0 = time.time()
    try:
        out = _drive(job)
    except Exception as exc:               # noqa: BLE001
        out = dict(best=None, laps=[], invalid=0, why=f"error: {type(exc).__name__}: {exc}",
                   t_end=0.0, trace=None)
    out.update(job=job, wall=time.time() - w0)
    return out


def _result_text(r: dict) -> str:
    laps = r.get("laps") or []
    if laps:
        txt = f"{rec.fmt_time(min(laps))} ({len(laps)} lap{'s' if len(laps) != 1 else ''})"
        if r.get("why"):
            txt += f", then {r['why']}"
        return txt
    return r.get("why") or "no valid lap"


def _atomic_json(path: str, d: dict, indent=None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(d, fh, indent=indent, separators=((",", ": ") if indent else (",", ":")))
        fh.write("\n")
    os.replace(tmp, path)


def build(workers=None, only=None, verbose: bool = True) -> dict:
    """Drive every reference run and write DATA_PATH and REF_PATH.

    `only`: a list of tracks to (re)build; every other class keeps its entry
    from the current table when that table is not stale, and otherwise says
    it is not built."""
    import multiprocessing as mp
    import importlib
    from .drive import ENGINE_MODES, SURFACE_MODES
    import cars
    from . import track as trk

    w0 = time.time()
    workers = int(workers) if workers else max(1, (os.cpu_count() or 1) - 4)
    lap_tracks = [t for t in trk.TRACK_ORDER if t in rec.LAP_TRACKS]
    todo = [t for t in lap_tracks if only is None or t in only]
    h = current_hash()
    if only is not None:
        raw, _why = _read_json(DATA_PATH, MEDALS_KIND)
        if raw is None or raw.get("inputs_hash") != h:
            #  an input changed: every map's times are stale, and a partial
            #  build would stamp the table current with the rest dropped
            raise SystemExit("medals --build --only: the table on disk is stale or missing "
                             f"({_why or 'another inputs hash'}); rebuild every map: {REGEN}")
    ckpts = checkpoints()

    # the physics of each (track, surface), so identical ones are driven once
    pkey = {(t, s): _physics_key(t, s) for t in todo for s in SURFACE_MODES}
    runs: dict = {}                    # (track, surface-leader, car, engine) -> class keys
    leader: dict = {}                  # class key -> the class it ran as
    for t in todo:
        first = {}
        for s in SURFACE_MODES:
            first.setdefault(pkey[(t, s)], s)
        for c in cars.CAR_ORDER:
            for e in ENGINE_MODES:
                for s in SURFACE_MODES:
                    lead = rec.class_key(t, c, e, first[pkey[(t, s)]])
                    leader[rec.class_key(t, c, e, s)] = lead
    jobs = []
    for lead in sorted(set(leader.values()), key=lambda k: (lap_tracks.index(rec.split_key(k)[0]), k)):
        t, c, e, s = rec.split_key(lead)
        for dname, dpath in drivers_for(c, t, ckpts):
            for aname, a_abs, a_tc in AIDS:
                jobs.append(dict(key=lead, track=t, car=c, engine=e, surface=s,
                                 driver=dname, path=dpath, aids=aname, abs=a_abs, tc=a_tc))
    # longest laps first, so the pool does not end on one long run (the
    # oval, task 46: 1.9 km, but the fastest lap, ~55 s, after Linden's)
    order = {"kestrel": 0, "open": 1, "ashdown": 2, "arena": 3, "linden": 4, "fairfield": 5,
             "skidpad": 6}
    jobs.sort(key=lambda j: (order.get(j["track"], 7), j["key"], j["driver"], j["aids"]))
    if verbose:
        n_cls = len(leader)
        print(f"medals --build: {n_cls} classes on {', '.join(todo) or 'no track'} "
              f"({len(set(leader.values()))} distinct), {len(jobs)} runs, {workers} workers")

    results: dict = {}                 # lead key -> [result]
    mod = importlib.import_module("drive.medals")   # pickled by its real name
    if jobs:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(mod._run_job, jobs, chunksize=1), 1):
                j = r["job"]
                results.setdefault(j["key"], []).append(r)
                if verbose:
                    print(f"  [{i:3d}/{len(jobs)}] {j['key']:<28s} {j['driver']:<34s} aids {j['aids']:<3s} "
                          f"{_result_text(r):<34s} {r['wall']:5.1f} s", flush=True)

    # ---- the table ------------------------------------------------------
    old = None
    if only is not None:
        raw, _why = _read_json(DATA_PATH, MEDALS_KIND)
        if raw is not None and raw.get("inputs_hash") == h:
            old = raw
    old_ref = None
    if old is not None:
        raw, _why = _read_json(REF_PATH, REF_KIND)
        if raw is not None and raw.get("inputs_hash") == h:
            old_ref = raw
    classes, laps = {}, {}
    drv_order = {d: i for i, d in enumerate([_lapdriver_name(m) for m in LAPDRIVER_MARGINS]
                                            + ["anchor"] + [n for n, _p, _m in ckpts])}
    used = set()
    for key in class_keys():
        t = rec.split_key(key)[0]
        if t not in rec.LAP_TRACKS:
            classes[key] = dict(author=None, reason=REASON_NO_MAP)
            continue
        if key not in leader:
            prev = (old or {}).get("classes", {}).get(key)
            if isinstance(prev, dict) and ("author" in prev):
                classes[key] = prev
                if prev.get("author") is not None and old_ref is not None and key in old_ref.get("laps", {}):
                    laps[key] = old_ref["laps"][key]
                if prev.get("driver"):
                    used.add(prev["driver"])
                for tr_ in prev.get("tried", []):
                    used.add(tr_.get("driver"))
            else:
                classes[key] = dict(author=None, reason=REASON_NOT_BUILT)
            continue
        lead = leader[key]
        rs = sorted(results.get(lead, []),
                    key=lambda r: (drv_order.get(r["job"]["driver"], 99), r["job"]["driver"],
                                   r["job"]["aids"] != "off"))
        tried = [dict(driver=r["job"]["driver"], aids=r["job"]["aids"],
                      best=(min(r["laps"]) if r["laps"] else None),
                      laps=[float(x) for x in r["laps"]], invalid=int(r["invalid"]),
                      result=_result_text(r)) for r in rs]
        for r in rs:
            used.add(r["job"]["driver"])
        win = None
        for r in rs:                       # first in driver order wins a tie
            if r["best"] is not None and (win is None or r["best"] < win["best"]):
                win = r
        if win is None:
            ent = dict(author=None, reason=REASON_NO_REF, tried=tried)
        else:
            a = float(win["best"])
            ent = dict(author=a, gold=a * GOLD_X, silver=a * SILVER_X, bronze=a * BRONZE_X,
                       driver=win["job"]["driver"],
                       aids=dict(abs=bool(win["job"]["abs"]), tc=bool(win["job"]["tc"])),
                       tried=tried)
            if win.get("trace"):
                laps[key] = win["trace"]
        if lead != key:
            ent["ran_as"] = lead
        classes[key] = ent

    secs = time.time() - w0
    table = dict(kind=MEDALS_KIND, generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
                 inputs_hash=h, thresholds=dict(gold=GOLD_X, silver=SILVER_X, bronze=BRONZE_X),
                 drivers=sorted((d for d in used if d), key=lambda d: (drv_order.get(d, 99), d)),
                 build_seconds=round(secs, 1), workers=workers,
                 flying_laps=FLYING_LAPS, t_max=dict(T_MAX), trace_hz=REF_TRACE_HZ,
                 built=[t for t in lap_tracks
                        if t in todo or t in ((old or {}).get("built") or [])],
                 classes=classes)
    _atomic_json(DATA_PATH, table, indent=1)
    _atomic_json(REF_PATH, dict(kind=REF_KIND, inputs_hash=h, hz=REF_TRACE_HZ, laps=laps))
    _CACHE.clear()
    _TRACE_CACHE.clear()
    if verbose:
        n_auth = sum(1 for e in classes.values() if e.get("author") is not None)
        print(f"wrote {DATA_PATH} ({os.path.getsize(DATA_PATH) / 1024:.0f} KB) and "
              f"{REF_PATH} ({os.path.getsize(REF_PATH) / 1024:.0f} KB): "
              f"{n_auth} classes with medals, build {secs:.0f} s")
    return table


def show(table=None) -> None:
    """One line a class: author, driver, aids, the medals."""
    table = load() if table is None else table
    print(f"medals: {len(table.get('classes', {}))} classes, generated {table.get('generated')}, "
          f"{'STALE' if is_stale(table) else 'current'}")
    for key in class_keys():
        e = table.get("classes", {}).get(key) or {}
        if _good(e.get("author")):
            aids = e.get("aids") or {}
            print(f"  {key:<28s} {fmt_time(e['author']):>9s}  {e.get('driver', ''):<34s} "
                  f"aids {'on ' if aids.get('abs') or aids.get('tc') else 'off'}  "
                  f"gold {fmt_time(e.get('gold'))}  silver {fmt_time(e.get('silver'))}  "
                  f"bronze {fmt_time(e.get('bronze'))}")
        else:
            print(f"  {key:<28s} {NONE_MARK:>9s}  {e.get('reason', 'no entry')}")


# ==================================================================== #
#  SELF-CHECK                                                          #
# ==================================================================== #
def self_check(verbose: bool = True) -> bool:
    import contextlib
    import io
    import subprocess
    import tempfile
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
        print("drive/medals.py self-check")
    table = load()
    cls = table.get("classes", {})
    keys = class_keys()

    # 1. every class: an author time, or a reason there is none -------------
    def _covered(k):
        e = cls.get(k)
        if not isinstance(e, dict):
            return False
        if _good(e.get("author")):
            return True
        # a class with no medals must say WHY in one of the two honest
        # ways: its map has no lap, or every reference driver was tried
        if rec.split_key(k)[0] not in rec.LAP_TRACKS:
            return e.get("reason") == REASON_NO_MAP
        return e.get("reason") == REASON_NO_REF and bool(e.get("tried"))
    bad = [k for k in keys if not _covered(k)]
    auth = [k for k in keys if isinstance(cls.get(k), dict) and _good(cls[k].get("author"))]
    reasons = {}
    for k in keys:
        e = cls.get(k)
        if isinstance(e, dict) and not _good(e.get("author")):
            reasons[e.get("reason")] = reasons.get(e.get("reason"), 0) + 1
    extra = sorted(set(cls) - set(keys))
    #  task 46: which cars the table has no class for at all (a car added
    #  since the last build -- their laps simply show no medals until then)
    unbuilt = sorted({rec.split_key(k)[1] for k in bad if k not in cls})
    rep("every class covered", bool(keys) and not bad,
        f"{len(keys)} classes: {len(auth)} with an author time, "
        + ", ".join(f"{n} '{r}'" for r, n in sorted(reasons.items(), key=lambda x: str(x[0])))
        + (f"; missing/bad {bad[:3]}" if bad else "")
        + (f"; no class at all for {', '.join(unbuilt)} (rebuild: {REGEN})" if unbuilt else "")
        + (f"; {len(extra)} unknown keys ignored" if extra else ""))

    # 1b. every lap map has a run budget: `_drive` reads T_MAX[track], and a
    #     KeyError there is caught per job and filed as 'no reference lap',
    #     which check 1 accepts -- so a map added without one would get no
    #     medals, silently
    rep("every lap map has a run budget", set(T_MAX) == set(rec.LAP_TRACKS),
        ", ".join(f"{t} {T_MAX[t]:.0f} s" for t in rec.LAP_TRACKS if t in T_MAX)
        + (f"; missing {sorted(set(rec.LAP_TRACKS) - set(T_MAX))}"
           if set(rec.LAP_TRACKS) - set(T_MAX) else ""))

    # 2. bronze > silver > gold > author > 0, each exactly author x constant --
    bad2 = []
    for k in auth:
        e = cls[k]
        a = e["author"]
        exact = all(e.get(m) == a * MULT[m] for m in ("gold", "silver", "bronze"))
        order = (_good(e.get("bronze")) and _good(e.get("silver")) and _good(e.get("gold"))
                 and e["bronze"] > e["silver"] > e["gold"] > a > 0.0)
        if not (exact and order):
            bad2.append(k)
    rep("bronze > silver > gold > author > 0, exactly author x constant",
        bool(auth) and not bad2,
        f"{len(auth) - len(bad2)}/{len(auth)} classes; x {GOLD_X} / {SILVER_X} / {BRONZE_X}"
        + (f"; bad {bad2[:3]}" if bad2 else ""))

    # 3. medal_for at the boundaries ---------------------------------------
    syn = dict(kind=MEDALS_KIND, classes={
        "a|b|c|d": dict(author=60.0, gold=60.0 * GOLD_X, silver=60.0 * SILVER_X,
                        bronze=60.0 * BRONZE_X, driver="x"),
        "a|b|c|e": dict(author=None, reason=REASON_NO_REF)})
    probes = [(syn, "a|b|c|d")] + ([(table, auth[0])] if auth else [])
    fails = []
    for tb, k in probes:
        tg = targets(k, tb)
        for i, m in enumerate(MEDALS):
            thr = tg[m]
            above = MEDALS[i + 1] if i + 1 < len(MEDALS) else None
            cases = [(thr, m), (np.nextafter(thr, 0.0), m), (thr - 1e-3, m),
                     (np.nextafter(thr, math.inf), above)]
            for t, want in cases:
                got = medal_for(k, float(t), tb)
                if got != want:
                    fails.append(f"{k} t={t!r}: {got} want {want}")
        for t in (tg["bronze"] + 1e-6, tg["bronze"] * 2.0, float("nan"), -1.0, 0.0, None):
            if medal_for(k, t, tb) is not None:
                fails.append(f"{k} t={t}: not None")
    if medal_for("a|b|c|e", 1.0, syn) is not None or medal_for("zz|zz|zz|zz", 1.0, syn) is not None:
        fails.append("a class with no author earned a medal")
    no_auth = next((k for k in keys if k not in auth), None)
    if no_auth is not None and medal_for(no_auth, 0.001, table) is not None:
        fails.append(f"{no_auth} has no author and earned a medal")
    rep("medal_for boundaries", not fails,
        (f"at / one ulp either side of each threshold, past bronze, nan, None, no author"
         f" ({len(probes)} tables)") if not fails else "; ".join(fails[:3]))

    # 4. the author laps' traces --------------------------------------------
    ref = _load_ref()
    bad4 = []
    worst_end = 0.0
    n_rows = []
    for k in auth:
        d = ref.get("laps", {}).get(k)
        tr_ = reference_trace(k)
        if not isinstance(d, dict) or tr_ is None:
            bad4.append(f"{k}: no trace")
            continue
        if int(d.get("hz", 0)) != REF_TRACE_HZ:
            bad4.append(f"{k}: hz {d.get('hz')}")
        t0, t1 = float(tr_[0, 0]), float(tr_[-1, 0])
        end = abs(t1 - cls[k]["author"])
        worst_end = max(worst_end, end)
        n_rows.append(tr_.shape[0])
        if not (0.0 <= t0 <= 0.05) or end > 0.2 or np.any(np.diff(tr_[:, 0]) <= 0.0):
            bad4.append(f"{k}: t {t0:.3f}..{t1:.3f} vs author {cls[k]['author']:.3f}")
    rep("author traces decode", bool(auth) and not bad4,
        (f"{len(auth) - len(bad4)}/{len(auth)} at {REF_TRACE_HZ} Hz, "
         f"{min(n_rows) if n_rows else 0}-{max(n_rows) if n_rows else 0} rows, start at t ~ 0, "
         f"last t within {worst_end:.3f} s of the author time")
        + (f"; {bad4[:3]}" if bad4 else ""))

    # 5. staleness: a SOFT fail --------------------------------------------
    h = current_hash()
    if is_stale(table):
        print(f"  [soft] medals are stale: {REGEN}  "
              f"(table {str(table.get('inputs_hash'))[:12]}, inputs now {h[:12]})")
        rep("staleness checked (soft)", True, "STALE -- not a failure; regenerate")
    else:
        rep("staleness checked (soft)", True, f"current, inputs {h[:12]}")

    # 6. a corrupt / missing / foreign file is an empty table, not a crash --
    tmp = tempfile.mkdtemp(prefix="carsim_medals_")
    got = []
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for name, body in (("missing.json", None), ("corrupt.json", "{ not json"),
                               ("foreign.json", json.dumps(dict(kind="carsim-medals-0", classes={}))),
                               ("list.json", "[1, 2]"), ("nocls.json", json.dumps(dict(kind=MEDALS_KIND)))):
                p = os.path.join(tmp, name)
                if body is not None:
                    with open(p, "w") as fh:
                        fh.write(body)
                tb = load(p)
                got.append(isinstance(tb, dict) and tb.get("classes") == {} and tb.get("empty")
                           and targets("arena|corsa|stock|patch", tb) is None
                           and medal_for("arena|corsa|stock|patch", 60.0, tb) is None)
            got.append(reference_trace("arena|corsa|stock|patch", os.path.join(tmp, "corrupt.json")) is None)
        rep("a missing / corrupt / foreign file is an empty table", all(got),
            f"{sum(bool(g) for g in got)}/{len(got)} cases, nothing raised")
    except Exception as exc:               # noqa: BLE001
        rep("a missing / corrupt / foreign file is an empty table", False, f"{type(exc).__name__}: {exc}")

    # 7. the import rule: no drive.ml, no pygame at import ------------------
    try:
        code = ("import sys, drive.medals; "
                "print([m for m in sys.modules if m.startswith('drive.ml') or m.startswith('pygame')])")
        r = subprocess.run([sys.executable, "-c", code], cwd=os.path.dirname(_HERE),
                           capture_output=True, text=True, timeout=60)
        rep("import drive.medals loads no drive.ml, no pygame", r.returncode == 0
            and r.stdout.strip() == "[]", r.stdout.strip() or r.stderr.strip()[-120:])
    except Exception as exc:               # noqa: BLE001
        rep("import drive.medals loads no drive.ml, no pygame", False, f"{type(exc).__name__}: {exc}")

    if verbose:
        print(f"  {'ALL PASS' if ok else 'FAILURES ABOVE'}: {n_ok}/{n_all} checks")
    return ok


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python3 -m drive.medals",
                                 description="Medal times per class (task 21). "
                                             "No argument: the self-check.")
    ap.add_argument("--build", action="store_true", help="drive the reference laps, write the table")
    ap.add_argument("--workers", type=int, default=None,
                    help="pool size (default: CPUs - 4, at least 1)")
    ap.add_argument("--only", default=None, help="TRACK[,TRACK]: rebuild these tracks only")
    ap.add_argument("--self-check", action="store_true", help="the self-check (the default)")
    ap.add_argument("--show", action="store_true", help="print the table")
    a = ap.parse_args(argv)
    if a.build:
        only = None
        if a.only:
            only = [t.strip() for t in a.only.split(",") if t.strip()]
            from . import track as trk
            unknown = [t for t in only if t not in trk.TRACK_ORDER]
            if unknown:
                ap.error(f"unknown track(s) {unknown}; tracks: {', '.join(trk.TRACK_ORDER)}")
        build(workers=a.workers, only=only)
        return 0
    if a.show:
        show()
        return 0
    return 0 if self_check() else 1


if __name__ == "__main__":
    sys.exit(main())
