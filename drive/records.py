"""drive/records.py -- lap records: your top 5 per class (task 19).

Every lap the user drives in a timed session is recorded. The ones that are
VALID (`LapTimer.lap_valid`, and nothing happened mid-lap that the physics
cannot reproduce) go into the class's top 5, persisted to
`runs/records/<class>.json`.

The CLASS (plan decision D1) is `track|car|engine|surface`, e.g.
`arena|corsa|sport|patch`. The build (wings, ballast) is NOT in the key:
designing the car is the game. Assists (ABS, TC, steer aid, gearbox) are
stored with each lap and never split a class. The file name is the key with
`|` written as `__` (`arena__corsa__sport__patch.json`): `|` is not a legal
file-name character on Windows, and the Steam build ships there.

What a lap keeps
----------------
* time, sector times, date, the build (name + the full `CarBuild` JSON), the
  assists;
* a POSE TRACE at 50 Hz in `drive.drive._Replay`'s schema (`t x y psi u
  wing_deploy wing_side top_deploy`) plus the centreline progress `s` the
  live delta is read against (task 22), fixed-point and zlib'd;
* a CONTROLS LOG: the `Controls` actually handed to `Vehicle.step`, one row
  per physics step, run-length encoded by step index (the step offsets where a
  field changes, and its values there) -- kept for future score verification,
  never shown or uploaded;
* the lap's START STATE: every dynamic attribute of the `Vehicle` (its
  integrated `VehicleState`, the `PowertrainState`, the ABS / TC memories),
  the harness's surface samples and the `LapTimer`, exactly -- JSON writes a
  float as the shortest repr that reads back to the same bits -- and the car
  and `VehicleConfig` it was driven with. `resimulate(rec)` rebuilds that
  car, restores that state and replays the log through a headless `Sim`, and
  gets the lap time back BIT FOR BIT (drive.py's V31 asserts it).

The one thing that makes the log exact AND small: the harness QUANTISES the
five continuous controls before the step while a recorder is attached (road
wheel to 2^-24 rad = 6e-8 rad, pedals to 2^-20). The quantised value is the
one the car is driven with, so the log holds integers and nothing is lost; the
resolution is a thousand times below anything a hand or a stick can hold. It
happens in `Sim.step_physics`, on the input side, never inside
`Vehicle.step`, and only with a recorder attached -- no scripted or headless
acceptance run has one, so every published number is untouched.

A lap is NOT recorded when the physics cannot replay it: a reset or a
teleport, a live setting change (engine, gearbox, ABS, TC, aids) or the `T`
wet toggle inside it. Nor when it is not a lap ROUND the circuit (a sector
line out of order, or under 95 % of the length -- `LapTimer` fires a lap
after its 3 s lockout however the car got back to the line, reversing
included), or when any of it ran in slow motion or single-stepped. The HUD
says why; the lap is simply not a record. A live change also RETARGETS the
recorder: the engine is in the class. A write that fails keeps the book in
memory and says NOT SAVED -- a record never stops the car.

Tracks: the four circuits (`arena`, `linden`, `kestrel`, `ashdown` --
track.CIRCUITS), `open` and `skidpad` (the standard 50 m pad) have laps. The
DRAGSTRIP has no finish time -- `LapTimer` never fires a lap on an open
track -- and is EXCLUDED rather than given an invented finish line: a
standing-start quarter mile is a different game mode, not a lap.

Player files are never read by a scripted or headless run: only
`drive.drive._interactive_session` attaches a recorder to `runs/records/`;
the self-checks write to a temporary directory.

UNLIMITED RUNS (task 41)
------------------------
The owner's rule: a wing may be given a span past its car's PHYSICAL limit
(`drive/bodies.py`: a flank panel's lower tip at the car's ground clearance,
a top wing 1.2 x the car's width) "just for fun", and such runs "won't go
towards the public leaderboard". Whether a run is one is COMPUTED at every
session's start from the build, the garage library and the car
(`bodies.over_limits`) -- never stored in a build, because a build names its
wings and a wing can be re-saved at another span. A session with any wing
past its limit is an UNLIMITED session and its recorder files into a
SEPARATE BOOK, `runs/records/unlimited/`, under the SAME class key
(`unlimited_book`): so an Unlimited lap never touches an official PB, top 5,
build best, best sector, best medal or ghost, and the medal targets and the
reference ghost still apply to it (it earns its medal into the Unlimited
book, shown as Unlimited). Every such lap also carries `unlimited: true` and
the reasons (`over_limits`: slot, wing, span, limit), and an official book
REFUSES a lap that says it is Unlimited. `runs/records/last_builds.json`
stays in the official folder whichever book asks. `publishable(lap)` is the
one test the planned public leaderboard (T28) must apply.

    python3 -m drive.records      the self-check
"""

from __future__ import annotations

import base64
import dataclasses
import json
import math
import os
import time
import zlib
from array import array

import numpy as np

RECORDS_KIND = "carsim-records-1"
LAST_BUILDS_KIND = "carsim-last-builds-1"
RECORDS_DIR = os.path.join("runs", "records")
LAST_BUILDS_FILE = "last_builds.json"
#: the Unlimited book's folder, under the official one (task 41)
UNLIMITED_DIR = "unlimited"
TOP_N = 5

#: the tracks a lap time exists on (the dragstrip is excluded, see above).
#: A literal, not track.CIRCUITS + ..., so this module stays importable
#: without the track; track.py's names are what these must match (the
#: self-check asserts it)
LAP_TRACKS = ("arena", "linden", "kestrel", "ashdown", "open", "skidpad")
#: a lap is filed only if the car went ROUND: every sector line in order and
#: at least this much of the track's length of centreline progress
LAP_MIN_FRACTION = 0.95
#: an open lap is dropped past this: a parked car (or an hour on the open
#: map's pad) must not hold an ever-growing log -- 4 min is 4x an arena lap
LAP_MAX_S = 240.0
#: the skidpad a record is for: the settings file cannot change it, only
#: the command line can (`--radius`, `--cw`), and a 30 m pad is not the same
#: track as a 50 m one
SKIDPAD_STD = (50.0, False)

#: the pose trace: `drive.drive._Replay`'s eight columns, then the
#: unwrapped centreline progress from the line (task 22's live delta).
TRACE_HZ = 50
TRACE_COLS = ("t", "x", "y", "psi", "u", "wing_deploy", "wing_side", "top_deploy", "s")
#: fixed point per column: ms, mm, mm, 1e-4 rad, mm/s, 1e-3, 1, 1e-3, mm
TRACE_SCALE = (1000.0, 1000.0, 1000.0, 10000.0, 1000.0, 1000.0, 1.0, 1000.0, 1000.0)

#: the controls log. The five continuous fields are QUANTISED before the step
#: (see the module docstring); the rest are ints / bools / the free-wings code.
Q_DELTA = float(2 ** 24)          # road wheel, rad
Q_PEDAL = float(2 ** 20)          # throttle, brake, clutch, handbrake
CTL_FLOAT = ("delta", "throttle", "brake", "clutch", "handbrake")
CTL_INT = ("gear_req", "auto_gearbox", "auto_clutch", "wing_on", "starter", "wing_cmd")
CTL_FIELDS = CTL_FLOAT + CTL_INT
_Q = (Q_DELTA, Q_PEDAL, Q_PEDAL, Q_PEDAL, Q_PEDAL)

#: `Vehicle` attributes that are the CAR, not its state: rebuilt from the car
#: and the config by `Vehicle.__init__`, never snapshotted
_VEH_STRUCTURAL = frozenset(("car", "cfg", "der", "pt_p", "tyre", "tel", "pos",
                             "guard_events"))
#: `LapTimer` attributes that are the track, not the timing
_TIMER_STRUCTURAL = frozenset(("L", "closed", "lines", "lockout", "wrap"))


# ==================================================================== #
#  THE CLASS                                                           #
# ==================================================================== #
def class_key(track: str, car: str, engine: str, surface: str) -> str:
    """`track|car|engine|surface` (D1)."""
    return "|".join((str(track), str(car), str(engine), str(surface)))


def split_key(key: str) -> tuple:
    parts = str(key).split("|")
    if len(parts) != 4:
        raise ValueError(f"not a class key: {key!r}")
    return tuple(parts)


def class_file(key: str) -> str:
    """The file name of a class: `|` -> `__` (Windows forbids `|`)."""
    for p in split_key(key):
        if not p or not all(c.isalnum() or c in "-_" for c in p):
            raise ValueError(f"class key component {p!r} is not a plain name")
    return key.replace("|", "__") + ".json"


def class_label(key: str) -> str:
    """A short human label: 'arena  corsa  sport  wet patches'."""
    try:
        t, c, e, s = split_key(key)
    except ValueError:
        return str(key)
    surf = {"patch": "wet patches", "none": "dry", "all": "wet"}.get(s, s)
    return f"{t}  {c}  {e}  {surf}"


def records_reason(track: str, radius: float = 50.0, cw: bool = False):
    """None when laps on this track are recorded, else the reason they are
    not (shown on the HUD)."""
    if track not in LAP_TRACKS:
        if track == "dragstrip":
            return "the dragstrip has no lap: no records there"
        return f"no records on '{track}'"
    if track == "skidpad" and (abs(float(radius) - SKIDPAD_STD[0]) > 1e-9
                               or bool(cw) != SKIDPAD_STD[1]):
        return (f"records are for the standard {SKIDPAD_STD[0]:.0f} m anticlockwise "
                f"skidpad, not R {float(radius):g}{' cw' if cw else ''}")
    return None


#: the `CarBuild` JSON keys that label a build rather than describe the car:
#: its library name, the library's built-in flag, and (task 41) the car it
#: was made for. Everything that asks "is this the same car?" strips exactly
#: these -- `build_id` here, `prerace._same_build` (the pre-race page's
#: "saved", the PICK's autosave, the wing tutorial's build step) -- so that
#: tagging a build with its car changes no build's identity: a build saved
#: before task 41 (no "car" key) and the same build tagged "bus" hash to the
#: same id, and every PB filed before keeps its build (the self-check proves
#: it against the pre-task-41 formula). The car a lap was driven in is in the
#: CLASS key already, so dropping the tag from the hash loses nothing.
BUILD_META = ("name", "builtin", "car")


def build_id(name, build_json) -> str:
    """What a build's best lap is filed under: its CONTENT -- a hash of the
    `CarBuild` JSON without its labels (`BUILD_META`: the name, the library
    flag, the car it was made for) -- so a car edited in the garage that kept
    its name is a different build, and two names for one car are one. A
    published one-panel car (no JSON) is known by its name."""
    if isinstance(build_json, dict):
        import hashlib
        body = {k: v for k, v in build_json.items() if k not in BUILD_META}
        return "b:" + hashlib.sha1(json.dumps(body, sort_keys=True, default=repr)
                                   .encode()).hexdigest()[:16]
    return "n:" + str(name or "")


def build_car(build_json) -> str:
    """The car a build was made for (a `cars.py` key), "" for a build saved
    before task 41 -- one made for ANY car -- or anything not a build."""
    c = build_json.get("car", "") if isinstance(build_json, dict) else ""
    return c if isinstance(c, str) else ""


def build_fits(build_json, car: str) -> bool:
    """May `car` open with this build without the player choosing it? Only
    its own builds and the any-car ones from before task 41: a Corsa build is
    never silently put on a bus (task 41)."""
    return build_car(build_json) in ("", str(car or ""))


def last_key(track: str, car: str) -> str:
    """The per-map memory's key for `car` on `track` (task 41)."""
    return f"{track}|{car}"


def surface_global_wet(surface: str, wet_scale: float) -> float:
    """The global grip scale a surface setting runs at ('all' -> wet)."""
    return float(wet_scale) if surface == "all" else 1.0


# ==================================================================== #
#  COMPACT, EXACT ENCODINGS                                            #
# ==================================================================== #
#: an int series is packed in the narrowest little-endian width that holds it
#: (the deltas of a lap's log are mostly 0 and small), then zlib at level 6:
#: level 9 on a whole lap cost ~0.4 s at the line, level 6 is 20x faster for
#: a file 4 % larger
_ZLEVEL = 6
_WIDTHS = (("i1", np.int8), ("i2", np.int16), ("i4", np.int32), ("i8", np.int64))


def _b64z(a) -> str:
    a = np.asarray(a, dtype=np.int64)
    lo, hi = (int(a.min()), int(a.max())) if a.size else (0, 0)
    for tag, dt in _WIDTHS:
        info = np.iinfo(dt)
        if info.min <= lo and hi <= info.max:
            break
    raw = np.ascontiguousarray(a.astype("<" + tag)).tobytes()
    return tag + ":" + base64.b64encode(zlib.compress(raw, _ZLEVEL)).decode("ascii")


def _unb64z(s: str) -> np.ndarray:
    tag, sep, body = s.partition(":")
    if not sep:                            # untagged: 8-byte ints
        tag, body = "i8", s
    if tag not in dict(_WIDTHS):
        raise ValueError(f"unknown int width {tag!r}")
    raw = zlib.decompress(base64.b64decode(body.encode("ascii")))
    return np.frombuffer(raw, dtype="<" + tag).astype(np.int64)


def rle_encode(v) -> dict:
    """Run-length encode an int64 series by index: the offsets where the
    value changes (the first sample always), and the value there -- both
    delta-coded and zlib'd. Exact."""
    v = np.asarray(v, dtype=np.int64)
    n = int(v.size)
    if n == 0:
        return dict(n=0, i="", v="")
    ch = np.empty(n, dtype=bool)
    ch[0] = True
    np.not_equal(v[1:], v[:-1], out=ch[1:])
    idx = np.flatnonzero(ch).astype(np.int64)
    val = v[idx]
    return dict(n=n, i=_b64z(np.diff(idx, prepend=0)), v=_b64z(np.diff(val, prepend=0)))


def rle_decode(d: dict) -> np.ndarray:
    n = int(d.get("n", 0))
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.cumsum(_unb64z(d["i"]))
    val = np.cumsum(_unb64z(d["v"]))
    if idx.size != val.size or idx[0] != 0 or idx[-1] >= n or np.any(np.diff(idx) <= 0):
        raise ValueError("corrupt run-length series")
    out = np.empty(n, dtype=np.int64)
    ends = np.append(idx[1:], n)
    out[:] = np.repeat(val, ends - idx)
    return out


def _wing_code(cmd) -> int:
    """`Controls.wing_cmd` -> int: None -> -1, else base-3 over (l, r, top)
    with 0 False, 1 True, 2 None."""
    if cmd is None:
        return -1
    c = 0
    for k, x in enumerate(tuple(cmd)[:3]):
        c += (2 if x is None else (1 if x else 0)) * (3 ** k)
    return c


def _wing_decode(c: int):
    c = int(c)
    if c < 0:
        return None
    out = []
    for _ in range(3):
        d = c % 3
        c //= 3
        out.append(None if d == 2 else bool(d))
    return tuple(out)


# ---- the generic, exact state serialiser -------------------------------
def _dc_registry() -> dict:
    """The dataclasses a snapshot may carry, by class name."""
    reg = {}
    try:
        from .vehicle import VehicleState, VehicleConfig, DevAero, TopAero
        reg.update(VehicleState=VehicleState, VehicleConfig=VehicleConfig,
                   DevAero=DevAero, TopAero=TopAero)
    except Exception:                      # noqa: BLE001
        pass
    try:
        from .powertrain import PowertrainState, PtInput
        reg.update(PowertrainState=PowertrainState, PtInput=PtInput)
    except Exception:                      # noqa: BLE001
        pass
    try:
        import cars
        reg["CarSpec"] = cars.CarSpec
    except Exception:                      # noqa: BLE001
        pass
    try:
        from corsa_c import CorsaC
        reg["CorsaC"] = CorsaC
    except Exception:                      # noqa: BLE001
        pass
    return reg


def ser(x):
    """A value -> plain JSON, exactly. Floats are written by `json` as the
    shortest repr that reads back to the same bits; tuples, numpy arrays and
    dataclasses are tagged so `deser` rebuilds the same types. Anything else
    raises -- a snapshot that silently drops state is worse than none."""
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    if isinstance(x, np.ndarray):
        return {"__nd__": str(x.dtype), "v": x.tolist()}
    if isinstance(x, tuple):
        return {"__t__": [ser(e) for e in x]}
    if isinstance(x, list):
        return [ser(e) for e in x]
    if isinstance(x, dict) and all(isinstance(k, str) and not k.startswith("__") for k in x):
        return {k: ser(v) for k, v in x.items()}
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {"__dc__": type(x).__name__,
                "f": {f.name: ser(getattr(x, f.name)) for f in dataclasses.fields(x)
                      if f.init}}
    raise TypeError(f"records.ser: cannot snapshot a {type(x).__name__}")


def deser(x, reg=None):
    if isinstance(x, list):
        return [deser(e, reg) for e in x]
    if isinstance(x, dict):
        if "__nd__" in x:
            return np.array(x["v"], dtype=np.dtype(x["__nd__"]))
        if "__t__" in x:
            return tuple(deser(e, reg) for e in x["__t__"])
        if "__dc__" in x:
            reg = reg if reg is not None else _dc_registry()
            cls = reg.get(x["__dc__"])
            if cls is None:
                raise TypeError(f"records.deser: unknown dataclass {x['__dc__']!r}")
            return cls(**{k: deser(v, reg) for k, v in x["f"].items()})
        return {k: deser(v, reg) for k, v in x.items()}
    return x


def vehicle_snapshot(veh) -> dict:
    """Every dynamic attribute of a `Vehicle`, exactly."""
    return {k: ser(v) for k, v in sorted(veh.__dict__.items())
            if k not in _VEH_STRUCTURAL}


def restore_vehicle(veh, snap: dict) -> None:
    reg = _dc_registry()
    for k, v in snap.items():
        setattr(veh, k, deser(v, reg))


def timer_snapshot(lt) -> dict:
    return {k: ser(v) for k, v in sorted(lt.__dict__.items())
            if k not in _TIMER_STRUCTURAL and k != "crossings"}


def restore_timer(lt, snap: dict) -> None:
    for k, v in snap.items():
        setattr(lt, k, deser(v))
    lt.crossings = []


# ---- the pose trace ----------------------------------------------------
def encode_trace(rows) -> dict:
    a = np.asarray(rows, dtype=np.float64).reshape(-1, len(TRACE_COLS))
    q = np.rint(a * np.asarray(TRACE_SCALE)).astype(np.int64)
    return dict(hz=TRACE_HZ, cols=list(TRACE_COLS), scale=list(TRACE_SCALE), n=int(len(a)),
                data={c: _b64z(np.diff(q[:, j], prepend=0)) for j, c in enumerate(TRACE_COLS)})


def decode_trace(d: dict) -> np.ndarray:
    """(n, 9) float64 in TRACE_COLS order; `[:, :8]` is a `_Replay` trace."""
    cols = list(d.get("cols", TRACE_COLS))
    scale = list(d.get("scale", TRACE_SCALE))
    n = int(d.get("n", 0))
    out = np.zeros((n, len(TRACE_COLS)))
    for j, c in enumerate(TRACE_COLS):
        if c in cols and n:
            k = cols.index(c)
            col = np.cumsum(_unb64z(d["data"][c]))
            if col.size != n:
                raise ValueError(f"trace column {c}: {col.size} rows, want {n}")
            out[:, j] = col / float(scale[k])
    return out


def decode_controls(d: dict) -> dict:
    """The log -> {field: int64 array} (floats still quantised)."""
    return {f: rle_decode(d["data"][f]) for f in CTL_FIELDS}


def controls_at(dec: dict, i: int, Controls):
    """The i-th logged step as a fresh `Controls`."""
    return Controls(delta=int(dec["delta"][i]) / Q_DELTA,
                    throttle=int(dec["throttle"][i]) / Q_PEDAL,
                    brake=int(dec["brake"][i]) / Q_PEDAL,
                    clutch=int(dec["clutch"][i]) / Q_PEDAL,
                    handbrake=int(dec["handbrake"][i]) / Q_PEDAL,
                    gear_req=int(dec["gear_req"][i]),
                    auto_gearbox=bool(dec["auto_gearbox"][i]),
                    auto_clutch=bool(dec["auto_clutch"][i]),
                    wing_on=bool(dec["wing_on"][i]),
                    starter=bool(dec["starter"][i]),
                    wing_cmd=_wing_decode(dec["wing_cmd"][i]))


def lap_size_bytes(rec: dict) -> int:
    return len(json.dumps(rec, separators=(",", ":")))


# ==================================================================== #
#  THE BOOK: top 5 per class, on disk                                  #
# ==================================================================== #
def _atomic_json(path: str, d: dict) -> None:
    """Write through a temp file of this process and thread, synced to disk,
    and keep the previous file as `<name>.bak`: a crash between the two
    renames leaves the .bak, which `RecordBook.load` falls back to."""
    import threading
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.part"
    with open(tmp, "w") as fh:
        json.dump(d, fh, separators=(",", ":"))
        fh.flush()
        os.fsync(fh.fileno())
    if os.path.exists(path):
        os.replace(path, path + ".bak")
    os.replace(tmp, path)


def _stash_bad(path: str, why: str) -> None:
    """A corrupt or foreign file is moved aside, never overwritten blind."""
    bad = f"{path}.bad-{time.strftime('%Y%m%d_%H%M%S')}"
    try:
        os.replace(path, bad)
        print(f"records: {os.path.basename(path)} ignored ({why}); kept as {bad}")
    except OSError:
        print(f"records: {os.path.basename(path)} ignored ({why})")


#: the medals, best first (plan D4; `drive.medals` owns the times)
MEDAL_ORDER = ("author", "gold", "silver", "bronze")


class RecordBook:
    """`runs/records/<class>.json`: the class's top `TOP_N` laps (sorted,
    fastest first), its best sector times ever, and its best medal.

    One book per session is shared by the recorder, the pre-race screen and
    the ghosts. It is safe to use from the recorder's filing thread (one
    lock), and `save` MERGES with what is on disk before it writes, so a
    second writer (another book, another instance of the game) loses
    nothing. A file that cannot be READ (a permission, a sharing lock) is
    not corruption: the class is marked unreadable for the session and never
    written over. A file that does not PARSE is moved aside, and the last
    good copy (`.bak`) is used when there is one."""

    def __init__(self, root: str = RECORDS_DIR, last_root: str | None = None,
                 unlimited: bool = False):
        import threading
        self.root = root
        #: task 41: an UNLIMITED book (`unlimited_book`) files the laps of a
        #: build past its car's span limit; an official one refuses them.
        #: `last_root` is where last_builds.json lives (default `root`): the
        #: Unlimited book keeps the official folder's.
        self.unlimited = bool(unlimited)
        self.last_root = last_root
        self._cache: dict = {}
        self._lock = threading.RLock()
        self.notes: list = []          # what was ignored on load, and why
        self.save_error = ""           # the last failed write, '' after a good one
        self.version = 0               # bumped on every change (the ghosts reload)

    def path(self, key: str) -> str:
        return os.path.join(self.root, class_file(key))

    @staticmethod
    def _empty(key: str) -> dict:
        return dict(kind=RECORDS_KIND, key=key, laps=[], best_sectors=[], best_medal=None,
                    build_bests={})

    def _parse(self, p: str, key: str) -> dict:
        """The file as a book. OSError = could not read; anything else =
        not a records file of this class."""
        with open(p) as fh:
            text = fh.read()
        try:
            raw = json.loads(text)
            if not isinstance(raw, dict) or raw.get("kind") != RECORDS_KIND:
                raise ValueError(f"kind {raw.get('kind') if isinstance(raw, dict) else None!r}"
                                 f", want {RECORDS_KIND!r}")
            if raw.get("key") != key:
                raise ValueError(f"key {raw.get('key')!r}")
            laps = [lp for lp in raw.get("laps", [])
                    if isinstance(lp, dict) and isinstance(lp.get("time"), (int, float))
                    and math.isfinite(lp["time"]) and lp["time"] > 0.0]
            laps.sort(key=lambda lp: lp["time"])
            d = self._empty(key)
            d["laps"] = laps[:TOP_N]
            bs = raw.get("best_sectors", [])
            d["best_sectors"] = [float(x) if isinstance(x, (int, float)) and math.isfinite(x)
                                 else None for x in (bs if isinstance(bs, list) else [])]
            bm = raw.get("best_medal")
            d["best_medal"] = bm if bm in MEDAL_ORDER else None
            bb = raw.get("build_bests", {})
            d["build_bests"] = {str(k): float(v) for k, v in (bb.items() if isinstance(bb, dict) else ())
                                if isinstance(v, (int, float)) and math.isfinite(v) and v > 0.0}
            return d
        except OSError:
            raise
        except Exception as exc:           # noqa: BLE001 -- any parse / schema failure
            raise ValueError(str(exc)) from exc

    def load(self, key: str) -> dict:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            d = self._empty(key)
            p = self.path(key)
            src = p if os.path.exists(p) else (p + ".bak" if os.path.exists(p + ".bak") else None)
            if src is not None:
                try:
                    d = self._parse(src, key)
                    if src != p:
                        self.notes.append(f"{os.path.basename(p)}: missing, read its .bak")
                except OSError as exc:
                    self.notes.append(f"{os.path.basename(p)}: unreadable ({exc}); "
                                      f"not written this session")
                    print(f"records: {self.notes[-1]}")
                    d = self._empty(key)
                    d["_unreadable"] = str(exc)
                except ValueError as exc:
                    self.notes.append(f"{os.path.basename(src)}: {exc}")
                    _stash_bad(src, str(exc))
                    d = self._empty(key)
                    if src == p and os.path.exists(p + ".bak"):
                        try:
                            d = self._parse(p + ".bak", key)
                            self.notes.append(f"{os.path.basename(p)}: restored from its .bak")
                        except Exception:  # noqa: BLE001
                            d = self._empty(key)
            self._cache[key] = d
            return d

    def _merge_disk(self, key: str, d: dict) -> None:
        """Fold what is on disk NOW into the book before it is written."""
        p = self.path(key)
        if not os.path.exists(p):
            return
        try:
            disk = self._parse(p, key)
        except ValueError:
            return                         # the next load stashes it; ours wins
        seen = {(lp.get("time"), lp.get("date")) for lp in d["laps"]}
        extra = [lp for lp in disk["laps"] if (lp.get("time"), lp.get("date")) not in seen]
        if extra:
            laps = sorted(d["laps"] + extra, key=lambda lp: lp["time"])
            d["laps"][:] = laps[:TOP_N]
        self._merge_sectors(d, disk.get("best_sectors") or [])
        bb = d.setdefault("build_bests", {})
        for name, t in (disk.get("build_bests") or {}).items():
            if name not in bb or t < bb[name]:
                bb[name] = t
        a, b = d.get("best_medal"), disk.get("best_medal")
        if b in MEDAL_ORDER and (a not in MEDAL_ORDER or MEDAL_ORDER.index(b) < MEDAL_ORDER.index(a)):
            d["best_medal"] = b

    def save(self, key: str):
        """Write the class file. A failed write (a read-only install, a full
        disk, a file held by a sync client) keeps the book in memory, notes
        why and returns None: a record that cannot be saved must never stop
        the car."""
        p = self.path(key)
        with self._lock:
            d = self.load(key)
            try:
                if d.get("_unreadable"):
                    raise OSError(f"the file was unreadable at load ({d['_unreadable']}); "
                                  f"not written over")
                self._merge_disk(key, d)
                out = {k: v for k, v in d.items() if not k.startswith("_")}
                out["laps"] = [{k: v for k, v in lp.items() if not k.startswith("_")}
                               for lp in d["laps"] if not lp.get("_pending")]
                _atomic_json(p, out)
            except OSError as exc:
                self.save_error = f"{type(exc).__name__}: {exc}"
                self.notes.append(f"{os.path.basename(p)}: not saved ({self.save_error})")
                print(f"records: {os.path.basename(p)} not saved ({self.save_error})")
                return None
            self.save_error = ""
            self.version += 1
            return p

    def laps(self, key: str) -> list:
        with self._lock:
            return list(self.load(key)["laps"])

    def pb(self, key: str):
        with self._lock:
            laps = self.load(key)["laps"]
            return laps[0] if laps else None

    def pb_time(self, key: str) -> float:
        p = self.pb(key)
        return float(p["time"]) if p else float("nan")

    def best_sectors(self, key: str) -> list:
        with self._lock:
            return list(self.load(key)["best_sectors"])

    def build_best(self, key: str, name: str, build_json=None) -> float:
        """A build's best lap in the class (nan when it has none): the
        pre-race screen's PICK rows (task 20). Every valid lap updates it,
        not only the top 5; it is filed by `build_id` (the car, not its name)."""
        with self._lock:
            t = self.load(key).get("build_bests", {}).get(build_id(name, build_json))
        return float(t) if isinstance(t, (int, float)) and math.isfinite(t) else float("nan")

    def rank_of(self, key: str, t: float):
        """The top-N position a lap of time `t` WOULD take (1-based), or None."""
        with self._lock:
            laps = self.load(key)["laps"]
            pos = 1 + sum(1 for lp in laps if lp["time"] <= t)
        return pos if pos <= TOP_N else None

    def insert(self, key: str, rec: dict, save: bool = True):
        """Put a VALID lap in the class. Returns its 1-based position in the
        top N, or None when it is not fast enough. The best sectors and the
        build's best are updated either way (a slow lap can still hold the
        class's best S2). An equal time goes BEHIND the one already there:
        the first to set a time keeps it. An OFFICIAL book refuses a lap that
        says it is Unlimited (task 41: ValueError; the recorder shows it)."""
        if rec.get("unlimited") and not self.unlimited:
            raise ValueError("an Unlimited lap is never filed in the official book")
        with self._lock:
            d = self.load(key)
            self._merge_sectors(d, rec.get("sectors") or [])
            b = rec.get("build") if isinstance(rec.get("build"), dict) else {}
            bid = build_id(b.get("name", ""), b.get("json"))
            bb = d.setdefault("build_bests", {})
            t = float(rec["time"])
            if bid != "n:" and (bid not in bb or t < bb[bid]):
                bb[bid] = t
            pos = self.rank_of(key, t)
            if pos is not None:
                d["laps"].insert(pos - 1, rec)
                del d["laps"][TOP_N:]
            self.version += 1
            if save:
                self.save(key)
            return pos

    @staticmethod
    def _merge_sectors(d: dict, secs: list) -> None:
        bs = list(d.get("best_sectors") or [])
        while len(bs) < len(secs):
            bs.append(None)
        for i, s in enumerate(secs):
            if s is None or not math.isfinite(s) or s <= 0.0:
                continue
            if bs[i] is None or s < bs[i]:
                bs[i] = float(s)
        d["best_sectors"] = bs

    def set_best_medal(self, key: str, medal: str, order: tuple = MEDAL_ORDER,
                       save: bool = True) -> bool:
        """Keep the best medal ever earned in the class (`order` best first).
        True when it improved. `save=False` leaves the write to the caller
        (the recorder's filing thread)."""
        with self._lock:
            d = self.load(key)
            cur = d.get("best_medal")
            if medal not in order:
                return False
            if cur in order and order.index(cur) <= order.index(medal):
                return False
            d["best_medal"] = medal
            self.version += 1
            if save:
                self.save(key)
            return True

    # ---- the pre-race screen's per-track default build (task 20) ------
    #  Task 41: kept per MAP AND CAR. An entry's key is `track|car` (the car
    #  it was DRIVEN in), so a bus session on the arena no longer writes over
    #  the build the Corsa last used there. A file from before task 41 holds
    #  bare `track` keys; those are still read, for any car, but only when the
    #  build they hold is one that car may open with (`build_fits`) -- which
    #  every pre-task-41 build is, as it carries no car. Nothing is rewritten:
    #  the old entry simply stops being consulted once the car has its own.
    def _last_path(self) -> str:
        return os.path.join(self.last_root or self.root, LAST_BUILDS_FILE)

    def last_builds(self) -> dict:
        p = self._last_path()
        if not os.path.exists(p):
            return {}
        try:
            with open(p) as fh:
                raw = json.load(fh)
            if not isinstance(raw, dict) or raw.get("kind") != LAST_BUILDS_KIND:
                raise ValueError(f"kind {raw.get('kind') if isinstance(raw, dict) else None!r}")
            tr = raw.get("tracks", {})
            return tr if isinstance(tr, dict) else {}
        except Exception as exc:           # noqa: BLE001 -- ANY bad file is ignored
            self.notes.append(f"{LAST_BUILDS_FILE}: {exc}")
            _stash_bad(p, str(exc))
            return {}

    def last_build(self, track: str, car: str | None = None):
        """The entry `{name, build, date}` last used on `track` -- with `car`
        (task 41): the one that car used there, else a pre-task-41 bare-track
        entry whose build `car` may open with; never a build made for another
        car. `car` None: the bare-track entry only (the old reading)."""
        tracks = self.last_builds()

        def entry(k):
            e = tracks.get(k)
            return e if isinstance(e, dict) and isinstance(e.get("build"), dict) else None
        if car is None:
            return entry(track)
        e = entry(last_key(track, car))
        if e is None:                      # this car has no entry of its own yet
            e = entry(track)
        return e if e is not None and build_fits(e["build"], car) else None

    def set_last_build(self, track: str, name: str, build: dict,
                       car: str | None = None) -> bool:
        """This build is `track`'s from now on -- for `car` (task 41), under
        `track|car`; without a car, the bare-track key (the old writing).

        A build made for ANOTHER car (`build_fits` False: a 540i build driven
        in a Corsa challenge, a bus build the player picked on the Corsa) is
        never filed under this car's key (review of task 41, finding 0):
        `last_build` would never hand it back, so writing it would only wipe
        the build this car really last used there. False, nothing written."""
        if car and not build_fits(build, car):
            return False
        tracks = self.last_builds()
        tracks[last_key(track, car) if car else track] = dict(
            name=str(name), build=build, date=time.strftime("%Y-%m-%dT%H:%M:%S"))
        try:
            _atomic_json(self._last_path(), dict(kind=LAST_BUILDS_KIND, tracks=tracks))
        except OSError as exc:
            self.notes.append(f"{LAST_BUILDS_FILE}: not saved ({exc})")
            print(f"records: {LAST_BUILDS_FILE} not saved ({exc})")
            return False
        return True

    def rename_last_build(self, old: str, new: str) -> int:
        """A library build renamed `old` -> `new` (the garage's R; review of
        task 41, finding 9): every per-map entry whose name -- or whose
        build's name -- is `old` now says `new`, so the next map change does
        not bring the build back under its deleted name. The number of
        entries moved; the file is rewritten (atomically, `_atomic_json`)
        only when one moved, and a failed write is noted, never raised."""
        tracks = self.last_builds()
        moved = 0
        for e in tracks.values():
            if not isinstance(e, dict):
                continue
            b = e.get("build")
            b_hit = isinstance(b, dict) and b.get("name") == old
            if e.get("name") == old or b_hit:
                e["name"] = str(new)
                if b_hit:
                    b["name"] = str(new)
                moved += 1
        if not moved:
            return 0
        try:
            _atomic_json(self._last_path(), dict(kind=LAST_BUILDS_KIND, tracks=tracks))
        except OSError as exc:
            self.notes.append(f"{LAST_BUILDS_FILE}: not saved ({exc})")
            print(f"records: {LAST_BUILDS_FILE} not saved ({exc})")
            return 0
        return moved


def unlimited_book(root: str = RECORDS_DIR) -> RecordBook:
    """The UNLIMITED book beside the official one at `root` (task 41):
    `<root>/unlimited/<class>.json`, the same class keys, its own PBs, top 5,
    build bests and best medal; last_builds.json stays in `root`."""
    return RecordBook(os.path.join(root, UNLIMITED_DIR), last_root=root, unlimited=True)


def publishable(lap) -> bool:
    """May this lap go to a PUBLIC leaderboard? THE ONE HOOK the planned
    public boards (plan T28: `leaderboard.submit`, the local and the Steam
    backends) must call before they submit anything -- a lap from a build
    with any wing past its car's physical span limit (`unlimited`, task 41)
    is never published, whatever book it came from. False for an Unlimited
    lap (and for anything that is not a lap record), True otherwise."""
    return isinstance(lap, dict) and not bool(lap.get("unlimited"))


# ==================================================================== #
#  THE RECORDER: one per session, fed by Sim.step_physics              #
# ==================================================================== #
class LapRecorder:
    """Records every lap of one session and files the valid ones.

    `Sim.step_physics` calls, per step: `controls(sim, ctl)` right before
    `Vehicle.step` (quantises the continuous fields in place, logs the row),
    `event(sim, e)` for every `LapTimer` event, and `after_step(sim)` once the
    step is complete. A lap OPENS at the end of the step in which the car
    crossed the line (the state snapshot is the start of the next step) and
    CLOSES at the end of the step with the next crossing, so that step's
    sector event is in. `discard(why)` drops the open lap (a reset, a live
    setting change, the wet toggle).

    `meta` is a dict the session fills: build name + JSON, assists,
    settings, the physics block (`physics_block`). `on_lap(result)` is
    called with each closed lap's outcome (the HUD note).

    FILING happens off the physics step: the step that crosses the line only
    decides the lap is valid and where it ranks (cheap); encoding the log and
    the trace and writing the class file (~15-90 ms, longer for a long lap)
    run on one worker thread, so the car never stutters at the line.
    `flush()` waits for it -- the session calls it when it ends, before the
    next session's book reads the file."""

    def __init__(self, book: RecordBook, key: str, meta: dict,
                 expect_global_wet: float = 1.0, dt: float = 0.001, on_lap=None,
                 threaded: bool = True):
        self.book = book
        self.key = key
        self.meta = meta
        self.expect_wet = float(expect_global_wet)
        self.dt = float(dt)
        self.trace_stride = max(1, int(round(1.0 / (TRACE_HZ * self.dt))))
        self.max_steps = int(round(LAP_MAX_S / self.dt))
        self.on_lap = on_lap
        self.threaded = bool(threaded)
        self._pool = None              # the filing thread, made on first use
        self._futs: list = []
        self._open = None              # the lap being driven
        self._open_at = None           # t_cross of a crossing this step
        self._close = False            # a 'lap' event this step
        self._why = ""                 # why the open lap was discarded
        self.last = None               # the last closed lap's result dict
        self.n_laps = 0                # laps closed (valid or not)
        self.n_valid = 0               # ... of which valid and sent to be filed
        self.n_saved = 0               # ... and filed in the top N (worker side)

    # -- the per-step hooks ---------------------------------------------
    def controls(self, sim, ctl) -> None:
        d = int(round(ctl.delta * Q_DELTA))
        t = int(round(ctl.throttle * Q_PEDAL))
        b = int(round(ctl.brake * Q_PEDAL))
        c = int(round(ctl.clutch * Q_PEDAL))
        h = int(round(ctl.handbrake * Q_PEDAL))
        ctl.delta = d / Q_DELTA
        ctl.throttle = t / Q_PEDAL
        ctl.brake = b / Q_PEDAL
        ctl.clutch = c / Q_PEDAL
        ctl.handbrake = h / Q_PEDAL
        lap = self._open
        if lap is not None:
            lap["ctl"].extend((d, t, b, c, h, int(ctl.gear_req), int(bool(ctl.auto_gearbox)),
                               int(bool(ctl.auto_clutch)), int(bool(ctl.wing_on)),
                               int(bool(ctl.starter)), _wing_code(ctl.wing_cmd)))

    def event(self, sim, e) -> None:
        kind = e[0]
        if kind == "sector":
            lap = self._open
            if lap is not None:
                i = int(e[1])
                if i != lap["next_sec"]:   # a line crossed out of order: the car
                    lap["short"] = True    # did not go round (it reversed, cut)
                lap["next_sec"] = i + 1
                lap["sectors"][i] = float(e[3])
        elif kind in ("start", "lap"):
            if kind == "lap":
                self._close = True
            self._open_at = float(e[2])

    def after_step(self, sim) -> None:
        if self._close:
            self._close = False
            if self._open is not None:
                self._finish(sim)
            else:
                self._skipped(sim)
        if self._open_at is not None:
            t_cross, self._open_at = self._open_at, None
            self._begin(sim, t_cross)
            return
        lap = self._open
        if lap is None:
            return
        # the physics is the same in slow motion and one step at a time, but
        # a lap driven at a quarter of the speed is not the lap the table is
        # for: neither is a record
        if getattr(sim, "time_scale", 1.0) != 1.0:
            self.discard("slow motion")
            return
        if getattr(sim, "paused", False):
            self.discard("single-stepped")
            return
        k = sim.n - lap["n0"]
        if k > self.max_steps:             # parked, or an hour on the pad: the
            self.discard(f"over {LAP_MAX_S / 60.0:.0f} min")   # buffers stay bounded
            return
        if k % self.trace_stride == 0:
            self._trace_row(sim, lap)

    def discard(self, why: str) -> None:
        if self._open is not None or self._open_at is not None:
            self._why = why
        self._open = None
        self._open_at = None
        self._close = False

    def retarget(self, settings, pending=None) -> None:
        """A live setting changed. The ENGINE is in the class and the assists
        go with every lap, so the next lap is filed under the settings it is
        driven with, not the ones the session started on. Only the engine can
        change live: map, car and surface are restart settings, and the
        settings page may be showing a PREVIEWED one (`pending` holds the
        running values), so those three are kept from the running key."""
        t, c, _e, w = split_key(self.key)
        self.key = class_key(t, c, settings.engine, w)
        self.meta["assists"] = assists_of(settings)
        snap = dict(settings.as_dict())
        snap.update(dict(pending or {}))
        self.meta["settings"] = snap

    @property
    def recording(self) -> bool:
        return self._open is not None

    # -- open / close ------------------------------------------------------
    def _begin(self, sim, t_cross: float) -> None:
        if abs(float(sim.global_wet) - self.expect_wet) > 1e-12:
            self._open = None
            self._why = "the wet toggle (T) is on"
            return
        veh = sim.veh
        # progress starts at the car's distance past the line (a step's worth),
        # not at 0: the trace's `s` is then centreline metres FROM THE LINE,
        # which is what a live delta is read against
        L = float(sim.track.length)
        s0 = float(sim.s)
        if sim.track.closed and s0 > 0.5 * L:
            s0 -= L
        try:
            self._open = dict(
                n0=int(sim.n), t_cross=float(t_cross), ctl=array("q"), trace=[], sectors={},
                next_sec=0, short=False, prog=s0, s_last=float(sim.s),
                start=dict(n=int(sim.n), t=float(sim.t), t_cross=float(t_cross),
                           s=float(sim.s), mu=[float(m) for m in sim.mu],
                           crr=[float(c) for c in sim.crr],
                           on_track4=[bool(o) for o in sim.on_track4],
                           lap=timer_snapshot(sim.lap), veh=vehicle_snapshot(veh)),
                physics=dict(physics_block(sim)),
                date=time.strftime("%Y-%m-%dT%H:%M:%S"))
        except Exception as exc:           # noqa: BLE001 -- e.g. a state type ser()
            self._open = None              # does not know: no record, never a crash
            self._why = f"snapshot failed ({type(exc).__name__}: {exc})"
            print(f"records: {self._why}")
            return
        self._why = ""
        self._trace_row(sim, self._open)

    def _trace_row(self, sim, lap) -> None:
        v = sim.veh
        L = float(sim.track.length)
        ds = float(sim.s) - lap["s_last"]
        if sim.track.closed:
            if ds < -0.5 * L:
                ds += L
            elif ds > 0.5 * L:
                ds -= L
        if abs(ds) <= 30.0:
            lap["prog"] += ds
        lap["s_last"] = float(sim.s)
        lap["trace"].append((float(sim.t) - lap["t_cross"], v.x, v.y, v.psi, v.u,
                             float(v.wing_deploy), float(v.wing_side),
                             float(getattr(v, "top_deploy", 0.0)), lap["prog"]))

    def _skipped(self, sim) -> None:
        """A lap ended that was not being recorded (discarded mid-lap)."""
        self.n_laps += 1
        res = dict(time=float(sim.lap.last_lap), valid=False, pos=None,
                   pb_before=self.book.pb_time(self.key), key=self.key,
                   why=self._why or "not recorded")
        if self.meta.get("unlimited"):
            res["unlimited"] = True
        self.last = res
        if self.on_lap is not None:
            self.on_lap(res)

    def _finish(self, sim) -> None:
        lap, self._open = self._open, None
        self.n_laps += 1
        t = float(sim.lap.last_lap)
        res = dict(time=t, valid=bool(sim.lap.lap_valid), pos=None,
                   pb_before=self.book.pb_time(self.key), key=self.key, why="")
        if self.meta.get("unlimited"):     # task 41: the card and the note say so
            res["unlimited"] = True
        self._trace_row(sim, lap)
        n_sec = max(len(getattr(sim.lap, "lines", [0.0])), 1)
        secs = ([lap["sectors"].get(i) for i in range(n_sec)] if n_sec > 1 else [])
        L = float(sim.track.length)
        if not res["valid"]:
            res["why"] = "off track"
        elif lap["short"] or any(x is None for x in secs) or lap["prog"] < LAP_MIN_FRACTION * L:
            # `LapTimer` counts a crossing after its 3 s lockout whichever way
            # the car got back to the line; a lap is a lap ROUND the circuit
            res["valid"], res["why"] = False, "not a full lap"
        if res["valid"]:
            self.n_valid += 1
            res["sectors"] = secs
            # the lap enters the book NOW, light (time, sectors, build,
            # assists), so its rank, the PB and the next lap's delta are
            # right at once; its trace, log and start state are encoded and
            # written by the filing thread, which clears `_pending`. A
            # pending lap is never written to disk half-built.
            light = self._light(lap, t, secs)
            try:
                res["pos"] = self.book.insert(self.key, light, save=False)
            except Exception as exc:       # noqa: BLE001 -- never stop the car
                res["save_error"] = f"{type(exc).__name__}: {exc}"
            else:
                job = (self.key, lap, light, res)
                if self.threaded:
                    if self._pool is None:
                        from concurrent.futures import ThreadPoolExecutor
                        self._pool = ThreadPoolExecutor(max_workers=1,
                                                        thread_name_prefix="carsim-records")
                    self._futs = [f for f in self._futs if not f.done()]
                    self._futs.append(self._pool.submit(self._file, *job))
                else:
                    self._file(*job)
        self.last = res
        if self.on_lap is not None:
            self.on_lap(res)

    def _file(self, key, lap, light, res) -> None:
        """Encode and write one valid lap (the filing thread). A failure is a
        note on the HUD, never an exception into the drive."""
        err = ""
        try:
            heavy = self._heavy(lap)
            with self.book._lock:
                light.update(heavy)
                light.pop("_pending", None)
                self.book.version += 1
            if self.book.save(key) is None:
                err = self.book.save_error or "not saved"
        except Exception as exc:           # noqa: BLE001 -- never stop the car
            err = f"{type(exc).__name__}: {exc}"
            print(f"records: lap not filed ({err})")
        if res.get("pos") is not None and not err:
            self.n_saved += 1
        if err:
            res2 = dict(res, save_error=err, refiled=True)   # the lap's SECOND call
            self.last = res2
            if self.on_lap is not None:
                self.on_lap(res2)

    def save_later(self, key: str) -> None:
        """Write the class file off the physics step (the filing thread, in
        order after any lap it is filing); at once when not threaded."""
        if self.threaded:
            if self._pool is None:
                from concurrent.futures import ThreadPoolExecutor
                self._pool = ThreadPoolExecutor(max_workers=1,
                                                thread_name_prefix="carsim-records")
            self._futs.append(self._pool.submit(self.book.save, key))
        else:
            self.book.save(key)

    def flush(self, timeout: float | None = None) -> None:
        """Wait until every lap handed to the filing thread is on disk."""
        for f in list(self._futs):
            try:
                f.result(timeout=timeout)
            except Exception:              # noqa: BLE001 -- _file never raises
                pass
        self._futs = []

    def close(self) -> None:
        self.flush()
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def _light(self, lap: dict, t: float, secs: list) -> dict:
        """The part of a record the book ranks by (cheap: the line step),
        plus the raw trace IN MEMORY (`_trace_arr`, never written): against
        a busy physics loop the filing thread can take seconds to encode
        (the GIL), and the ghost of a new PB has to race the very next lap."""
        m = self.meta
        out = dict(version=1, time=t, sectors=list(secs), date=lap["date"], key=self.key,
                   build=dict(name=m.get("build_name", ""), json=m.get("build_json")),
                   assists=dict(m.get("assists", {})), settings=dict(m.get("settings", {})),
                   _pending=True, _trace_arr=np.asarray(lap["trace"], dtype=np.float64))
        if m.get("unlimited"):             # task 41: the flag and the reasons, on the lap
            out["unlimited"] = True
            out["over_limits"] = [dict(o) for o in m.get("over_limits", [])]
        return out

    @staticmethod
    def _heavy(lap: dict) -> dict:
        """The part that costs: the trace and the controls log, encoded."""
        rows = np.frombuffer(lap["ctl"], dtype=np.int64).reshape(-1, len(CTL_FIELDS))
        return dict(trace=encode_trace(lap["trace"]),
                    controls=dict(q_delta=Q_DELTA, q_pedal=Q_PEDAL, fields=list(CTL_FIELDS),
                                  n=int(rows.shape[0]),
                                  data={f: rle_encode(rows[:, j]) for j, f in enumerate(CTL_FIELDS)}),
                    start=lap["start"], physics=lap["physics"])


def fmt_time(t) -> str:
    """m:ss.mmm (or ss.mmm under a minute); '--' for nothing."""
    if t is None or not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0.0:
        return "--"
    ms = int(round(float(t) * 1000.0))
    m, rem = divmod(ms, 60000)
    s, ms = divmod(rem, 1000)
    return f"{m:d}:{s:02d}.{ms:03d}" if m else f"{s:d}.{ms:03d}"


def lap_note(res: dict) -> tuple:
    """A closed lap's HUD line and how long it stays: (text, seconds). Short:
    the bottom bar holds ~45 characters, and the PB, the place and the medal
    are also in the timing panel."""
    t = res.get("time")
    if not res.get("valid"):
        why = res.get("why") or "invalid"
        return f"LAP {fmt_time(t)} not recorded ({why})", 5.0
    pos, pb0 = res.get("pos"), res.get("pb_before")
    if res.get("save_error"):
        return f"LAP {fmt_time(t)} NOT SAVED ({res['save_error'][:40]})", 8.0
    medal = res.get("medal")
    tag = (f" {str(medal).upper()}" + ("!" if res.get("medal_best") else "")) if medal else ""
    #  an Unlimited lap (task 41) says so first: its PB and its place are the
    #  Unlimited book's, never the official ones
    lap_w, pb_w = ("UNLIMITED LAP", "UNLIMITED PB") if res.get("unlimited") else ("LAP", "NEW PB")
    if pos == 1:
        gain = (f" ({float(t) - pb0:+.3f})" if isinstance(pb0, float) and math.isfinite(pb0)
                else " (first)")
        return f"{pb_w} {fmt_time(t)}{gain} P1/{TOP_N}{tag}", 8.0
    if pos is not None:
        return f"{lap_w} {fmt_time(t)} P{pos}/{TOP_N}{tag}", 6.0
    return f"{lap_w} {fmt_time(t)} outside the top {TOP_N}{tag}", 5.0


def assists_of(settings) -> dict:
    """The assists a lap is filed with (D1: shown, never a class)."""
    return dict(abs=bool(settings.abs), tc=bool(settings.tc),
                steer_aid=bool(settings.steer_aid), gearbox=str(settings.gearbox))


def session_recorder(opts, settings, wet_scale: float, root: str = RECORDS_DIR,
                     on_lap=None, over=None):
    """The interactive session's recorder, or (None, reason) when this
    session records nothing. The ONE place `runs/records/` is attached:
    `drive.drive._interactive_session` calls it, nothing scripted does.

    `over` (task 41): the session build's wings past the car's physical span
    limit (`bodies.over_limits`, computed at the session's start). Any, and
    the recorder files into the UNLIMITED book (`unlimited_book`) and every
    lap carries `unlimited` and those reasons."""
    if getattr(opts, "headless", False) or getattr(opts, "script", None):
        return None, "a headless run: no records"
    if getattr(opts, "ml_drive", None):
        return None, "the ML driver is at the wheel: no records"
    why = records_reason(settings.track, getattr(opts, "radius", 50.0), getattr(opts, "cw", False))
    if why:
        return None, why
    key = class_key(settings.track, settings.car, settings.engine, settings.wet)
    meta = dict(build_name=str(getattr(opts, "build_name", "") or ""),
                build_json=getattr(opts, "build_json", None),
                assists=assists_of(settings), settings=dict(settings.as_dict()))
    over = [dict(o) for o in (over or [])]
    if over:
        meta.update(unlimited=True, over_limits=over)
    book = unlimited_book(root) if over else RecordBook(root)
    rec = LapRecorder(book, key, meta,
                      expect_global_wet=surface_global_wet(settings.wet, wet_scale),
                      dt=float(getattr(opts, "dt", 0.001)), on_lap=on_lap)
    return rec, ""


def physics_block(sim) -> dict:
    """What `resimulate` needs to rebuild the car and the track exactly."""
    tr = sim.track
    return dict(dt=float(sim.dt), global_wet=float(sim.global_wet),
                track=dict(name=tr.name, radius=float(getattr(sim, "track_radius", 50.0)),
                           cw=bool(getattr(sim, "track_cw", False)),
                           surfaces=bool(tr.surfaces)),
                car=ser(sim.veh.car), cfg=ser(sim.veh.cfg))


# ==================================================================== #
#  REPLAY                                                              #
# ==================================================================== #
def resimulate(rec: dict, max_extra_s: float = 5.0, on_step=None, on_event=None) -> dict:
    """Rebuild the lap's car and track, restore its start state and drive the
    controls log through a headless `Sim`. Returns the time and sectors the
    `LapTimer` sees and whether they match the record bit for bit."""
    from . import track as trk
    from .vehicle import Controls, Vehicle
    from .drive import Sim, ScriptedInput, LapTimer

    ph, st = rec["physics"], rec["start"]
    reg = _dc_registry()
    car = deser(ph["car"], reg)
    cfg = deser(ph["cfg"], reg)
    tk = ph["track"]
    tr = trk.make_track(tk["name"], tk.get("radius", 50.0), tk.get("cw", False),
                        surfaces=bool(tk.get("surfaces", True)))
    veh = Vehicle(car, cfg)
    restore_vehicle(veh, st["veh"])
    dec = decode_controls(rec["controls"])
    n_log = int(rec["controls"]["n"])
    i0 = int(st["n"])

    def drv(t, v, T_):
        k = sim.n - i0
        if k < n_log:
            return controls_at(dec, k, Controls)
        return Controls()                  # past the log: coast (a miss shows)

    sim = Sim(veh, tr, ScriptedInput(drv), dt=float(ph["dt"]),
              global_wet=float(ph["global_wet"]))
    sim.n, sim.t = i0, float(st["t"])
    sim.s, sim._s_prev = float(st["s"]), 0.0
    sim.mu, sim.crr = list(st["mu"]), list(st["crr"])
    sim.on_track4 = list(st["on_track4"])
    sim.lap = LapTimer(tr)
    restore_timer(sim.lap, st["lap"])
    got = None
    secs = {}
    n_max = n_log + int(max_extra_s / sim.dt)
    for _ in range(n_max):
        sim.step_physics(sim.dt)
        for e in sim.events_log:
            if on_event is not None:       # e.g. a GhostSet reading the replay
                on_event(sim, e)
            if e[0] == "sector":
                secs[int(e[1])] = float(e[3])
            elif e[0] == "lap":
                got = float(e[3])
        sim.events_log.clear()
        if on_step is not None:
            on_step(sim)
        if got is not None:
            break
    n_sec = len(rec.get("sectors") or [])
    sec_list = [secs.get(i) for i in range(n_sec)]
    return dict(time=got, sectors=sec_list, steps=sim.n - i0, n_log=n_log,
                exact=(got is not None and got == float(rec["time"])),
                sectors_exact=(sec_list == list(rec.get("sectors") or [])))


# ==================================================================== #
#  SELF-CHECK                                                          #
# ==================================================================== #
def _fake_rec(t: float, secs=None) -> dict:
    return dict(version=1, time=float(t), sectors=list(secs or []), date="2026-01-01T00:00:00",
                key="", build=dict(name="b", json=None), assists={}, settings={},
                trace=encode_trace([(0.0,) * len(TRACE_COLS)]), controls={}, start={},
                physics={})


def self_check(verbose: bool = True) -> bool:
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
        print("drive/records.py self-check")

    # -- the class key and its file name ---------------------------------
    k = class_key("arena", "corsa", "sport", "patch")
    rep("class key", k == "arena|corsa|sport|patch" and split_key(k) == ("arena", "corsa", "sport", "patch")
        and class_file(k) == "arena__corsa__sport__patch.json", f"{k} -> {class_file(k)}")
    bad = False
    try:
        class_file("arena|corsa|../x|patch")
    except ValueError:
        bad = True
    rep("a path in a key component is refused", bad)
    from . import track as trk
    rep("which tracks record",
        all(records_reason(t) is None for t in trk.CIRCUITS)
        and records_reason("open") is None
        and records_reason("skidpad") is None and records_reason("dragstrip") is not None
        and records_reason("skidpad", 30.0) is not None
        and records_reason("skidpad", 50.0, True) is not None
        and set(LAP_TRACKS) == set(trk.CIRCUITS) | {"open", "skidpad"}
        and set(LAP_TRACKS) <= set(trk.TRACK_ORDER),
        f"{', '.join(trk.CIRCUITS)}, open, the 50 m skidpad yes; dragstrip, other skidpads no")

    # -- the encodings are exact -----------------------------------------
    rng = np.random.default_rng(3)
    series = [np.zeros(0, dtype=np.int64), np.array([7]), np.full(1000, -3),
              np.cumsum(rng.integers(-2, 3, 5000)), rng.integers(-2 ** 40, 2 ** 40, 3000),
              np.repeat(rng.integers(-9, 9, 40), rng.integers(1, 200, 40))]
    rt = all(np.array_equal(rle_decode(rle_encode(s)), np.asarray(s, dtype=np.int64)) for s in series)
    rep("run-length log round-trips exactly", rt, f"{len(series)} series incl. empty, constant, random")
    codes = [None, (True, False, None), (False, True, None), (True, True, None),
             (False, False, False), (True, True, True)]
    rep("free-wings code round-trips", all(_wing_decode(_wing_code(c)) == (None if c is None else tuple(c))
                                           for c in codes), str([_wing_code(c) for c in codes]))
    x = np.array([0.0, -0.0, 1e-300, -1.5, math.pi, 1e300])
    y = deser(json.loads(json.dumps(ser(dict(a=x, b=(1, 2.5), c=[math.e, None, True])))))
    rep("state serialiser is bit exact", np.array_equal(x.view(np.int64), y["a"].view(np.int64))
        and y["b"] == (1, 2.5) and y["c"] == [math.e, None, True], "floats, -0.0, tuples, arrays")
    tr_rows = [(0.02 * i + rng.uniform(0, 1e-3), 10.0 + 0.3 * i + rng.normal(),
                -4.0 + 0.1 * i + rng.normal(), 0.001 * i + rng.normal(), rng.uniform(0, 60),
                rng.uniform(0, 1), (i % 3) - 1, rng.uniform(0, 1), 0.6 * i + rng.uniform())
               for i in range(200)]
    back = decode_trace(encode_trace(tr_rows))
    err = np.max(np.abs(back - np.asarray(tr_rows)) * np.asarray(TRACE_SCALE))
    rep("trace fixed point", back.shape == (200, 9) and err <= 0.5 + 1e-9,
        f"max error {err:.3f} quanta")

    # -- insert / order / evict / persist / corrupt tolerance ------------
    root = tempfile.mkdtemp(prefix="carsim_records_")
    book = RecordBook(root)
    pos = [book.insert(k, _fake_rec(t, [20.0, 21.0, 22.0])) for t in (63.0, 61.0, 62.0)]
    rep("insert returns the position", pos == [1, 1, 2], str(pos))
    times = [lp["time"] for lp in book.laps(k)]
    rep("sorted fastest first", times == [61.0, 62.0, 63.0], str(times))
    for t in (60.5, 64.0, 65.0, 66.0):
        book.insert(k, _fake_rec(t))
    times = [lp["time"] for lp in book.laps(k)]
    rep("top 5 evicts the slowest", times == [60.5, 61.0, 62.0, 63.0, 64.0], str(times))
    p_slow = book.insert(k, _fake_rec(70.0, [19.5, 25.0, 25.0]))
    rep("a slow lap is not a record, but its best sector is kept",
        p_slow is None and book.best_sectors(k) == [19.5, 21.0, 22.0], str(book.best_sectors(k)))
    p_tie = book.insert(k, _fake_rec(61.0))
    rep("a tie goes behind the time already set", p_tie == 3, str(p_tie))
    book2 = RecordBook(root)
    rep("persisted and read back", [lp["time"] for lp in book2.laps(k)] == [60.5, 61.0, 61.0, 62.0, 63.0]
        and book2.pb_time(k) == 60.5, str([lp["time"] for lp in book2.laps(k)]))
    rep("best medal keeps the best", book2.set_best_medal(k, "silver", ("gold", "silver", "bronze"))
        and not book2.set_best_medal(k, "bronze", ("gold", "silver", "bronze"))
        and book2.set_best_medal(k, "gold", ("gold", "silver", "bronze"))
        and RecordBook(root).load(k)["best_medal"] == "gold")
    k2 = class_key("open", "mx5", "stock", "none")
    with open(book.path(k2), "w") as fh:
        fh.write("{ this is not json")
    b3 = RecordBook(root)
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        empty = b3.laps(k2)
    moved = [f for f in os.listdir(root) if f.startswith(class_file(k2) + ".bad-")]
    rep("a corrupt file is ignored with a note and kept aside", empty == [] and b3.notes and moved,
        f"{b3.notes[:1]} -> {moved[:1]}")
    k3 = class_key("skidpad", "540i", "tuned", "all")
    with open(b3.path(k3), "w") as fh:
        json.dump(dict(kind="carsim-records-0", key=k3, laps=[_fake_rec(1.0)]), fh)
    with contextlib.redirect_stdout(io.StringIO()):
        old = b3.laps(k3)
    rep("an old kind is ignored, not crashed on", old == [], b3.notes[-1] if b3.notes else "")
    b3.set_last_build("arena", "fast one", dict(version=2, name="fast one"))
    b3.set_last_build("open", "wet one", dict(version=2, name="wet one"))
    rep("last build per track", RecordBook(root).last_build("arena")["name"] == "fast one"
        and RecordBook(root).last_build("open")["build"]["name"] == "wet one"
        and RecordBook(root).last_build("skidpad") is None)
    #  task 41: per map AND car. The bare "arena" entry above is a file from
    #  before task 41 (its build has no car): every car may still open with
    #  it until that car has an entry of its own; a build tagged for another
    #  car is never handed out, whichever key holds it.
    b3.set_last_build("arena", "bus wings", dict(version=2, name="bus wings", car="bus"),
                      car="bus")
    b3.set_last_build("linden", "corsa fast", dict(version=2, name="corsa fast", car="corsa"),
                      car="corsa")
    b3.set_last_build("kestrel", "corsa own", dict(version=2, name="corsa own", car="corsa"),
                      car="corsa")
    wrote_other = b3.set_last_build("kestrel", "picked", dict(version=2, name="picked", car="bus"),
                                    car="corsa")   # a bus build the player drove on the Corsa
    b3.set_last_build("ashdown", "mine", dict(version=2, name="mine", car="corsa"),
                      car="corsa")
    b3.set_last_build("ashdown", "old any-car", dict(version=2, name="old any-car"))
    rb = RecordBook(root)
    lb = {(t, c): (rb.last_build(t, c) or {}).get("name")
          for t in ("arena", "linden", "kestrel", "ashdown") for c in ("corsa", "bus")}
    rep("last build per map AND car: each car its own; a pre-task-41 entry for any car "
        "until it has one; another car's build never -- not even written over the car's "
        "own entry (review finding 0)",
        lb == {("arena", "corsa"): "fast one", ("arena", "bus"): "bus wings",
               ("linden", "corsa"): "corsa fast", ("linden", "bus"): None,
               ("kestrel", "corsa"): "corsa own", ("kestrel", "bus"): None,
               ("ashdown", "corsa"): "mine", ("ashdown", "bus"): "old any-car"}
        and wrote_other is False
        and rb.last_build("arena")["name"] == "fast one"
        and "arena|bus" in rb.last_builds() and "arena|corsa" not in rb.last_builds(),
        str(lb))
    #  a library rename moves the per-map memory with it (review finding 9):
    #  the entry's name and its build's name; an unknown name moves nothing
    moved = b3.rename_last_build("mine", "mine pro")
    e_ = RecordBook(root).last_build("ashdown", "corsa") or {}
    rep("a renamed build is renamed in the per-map memory too",
        moved == 1 and e_.get("name") == "mine pro"
        and e_.get("build", {}).get("name") == "mine pro"
        and b3.rename_last_build("nobody", "x") == 0, f"{moved} entry moved")
    #  and the BUILD_META strip: tagging a build with its car (task 41) moves
    #  no PB -- the id is the pre-task-41 formula's, for the old JSON and the
    #  tagged one alike
    import hashlib
    old_js = dict(version=2, name="fast", mirror=True, builtin=False,
                  slots={"left": {"wing": "fin", "x": 0.97, "h": 0.9, "inc_deg": 0.0,
                                  "mode": "active"}})
    pre41 = "b:" + hashlib.sha1(json.dumps({k: v for k, v in old_js.items()
                                            if k not in ("name", "builtin")},
                                           sort_keys=True, default=repr)
                                .encode()).hexdigest()[:16]
    ids = {build_id("fast", old_js), build_id("fast", dict(old_js, car="bus")),
           build_id("other", dict(old_js, car="", builtin=True))}
    rep("a build's id ignores its name, library flag and car: the pre-task-41 id for "
        "an old build and the same build tagged with a car",
        ids == {pre41} and build_id("x", dict(old_js, mirror=False)) != pre41
        and BUILD_META == ("name", "builtin", "car"), f"{pre41} ({len(ids)} distinct)")

    # -- what is not a lap, and what must not stop the car ----------------
    try:
        pr = _lifecycle_probe(tempfile.mkdtemp(prefix="carsim_records_"))
        rep("reversing over the line and back is not a lap",
            pr["reverse"]["valid"] is False and pr["reverse"]["why"] == "not a full lap"
            and pr["reverse_filed"] == 0, f"timer said {pr['reverse']['time']:.3f} s -> "
            f"{pr['reverse']['why']!r}, {pr['reverse_filed']} filed")
        rep("a lap in slow motion or single-stepped is not a record",
            pr["slowmo"]["why"] == "slow motion" and pr["step"]["why"] == "single-stepped",
            f"{pr['slowmo']['why']!r}, {pr['step']['why']!r}")
        rep("a full kinematic lap IS one", pr["full"]["valid"] and pr["full"]["pos"] == 1,
            f"{pr['full']['time']:.3f} s P{pr['full']['pos']}")
    except Exception as exc:               # noqa: BLE001
        import traceback
        traceback.print_exc()
        rep("lifecycle probe", False, f"{type(exc).__name__}: {exc}")
    ro = tempfile.mkdtemp(prefix="carsim_records_ro_")
    os.chmod(ro, 0o500)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            bro = RecordBook(os.path.join(ro, "records"))
            p_ro = bro.insert(k, _fake_rec(59.0))
            lb_ro = bro.set_last_build("arena", "x", {})
        rep("a read-only disk keeps the lap in memory and does not raise",
            p_ro == 1 and bro.save_error and not lb_ro and bro.pb_time(k) == 59.0,
            bro.save_error[:60])
    finally:
        os.chmod(ro, 0o700)
    # an unreadable file (a lock, a permission) is NOT corruption: kept, and
    # never written over this session
    kr = class_key("arena", "mx5", "sport", "none")
    br0 = RecordBook(root)
    for t in (70.0, 71.0):
        br0.insert(kr, _fake_rec(t))
    os.chmod(br0.path(kr), 0o000)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            br1 = RecordBook(root)
            seen = br1.laps(kr)
            br1.insert(kr, _fake_rec(65.0))
    finally:
        os.chmod(br0.path(kr), 0o644)
    kept = [lp["time"] for lp in RecordBook(root).laps(kr)]
    rep("an unreadable file is kept, not stashed and not written over",
        seen == [] and br1.save_error and kept == [70.0, 71.0]
        and not [f for f in os.listdir(root) if f.startswith(class_file(kr) + ".bad")], str(kept))
    # two writers: the second save MERGES what the first wrote
    ka, kb = RecordBook(root), RecordBook(root)
    km = class_key("open", "540i", "sport", "all")
    ka.pb_time(km), kb.pb_time(km)
    kb.insert(km, _fake_rec(58.0))
    ka.insert(km, _fake_rec(61.0))
    rep("a second writer loses nothing (save merges the file)",
        [lp["time"] for lp in RecordBook(root).laps(km)] == [58.0, 61.0])
    # a torn main file falls back to the last good copy
    with open(book.path(km), "w") as fh:
        fh.write('{"kind": "carsim-rec')
    with contextlib.redirect_stdout(io.StringIO()):
        bk = RecordBook(root).laps(km)
    rep("a torn file is restored from its .bak", [lp["time"] for lp in bk] == [58.0],
        str([lp["time"] for lp in bk]))
    # a live engine change retargets; a PREVIEWED map / car / surface does not
    rr = LapRecorder(RecordBook(root), class_key("arena", "corsa", "stock", "patch"), {}, 1.0)
    from types import SimpleNamespace as _NS
    prev = _NS(track="open", car="mx5", engine="tuned", wet="all", abs=True, tc=False,
               steer_aid=True, gearbox="auto", as_dict=lambda: dict(track="open"))
    rr.retarget(prev, {"track": "arena"})
    rep("a live engine change retargets the class, a previewed map does not",
        rr.key == "arena|corsa|tuned|patch" and rr.meta["assists"]["abs"] is True
        and rr.meta["settings"]["track"] == "arena", rr.key)
    with open(book.path(k3), "w") as fh:
        fh.write('{"kind": "carsim-records-1", "key": "%s", "laps": [{"time": 1%s}]}' % (k3, "0" * 400))
    with contextlib.redirect_stdout(io.StringIO()):
        huge = RecordBook(root).laps(k3)
    rep("an out-of-range number in a file is a bad file, not a crash", huge == [])

    # -- task 41: an UNLIMITED session files apart -------------------------
    #  a build past its car's span limit gets its own book under the SAME
    #  class key; the official PB, top 5, build bests, best medal and the
    #  per-map builds never see its laps, and no Unlimited lap is publishable
    ku = class_key("arena", "corsa", "stock", "none")
    off = RecordBook(root)
    off.insert(ku, _fake_rec(62.0, [20.0, 21.0, 21.0]))
    off.set_best_medal(ku, "bronze")
    off.set_last_build("arena", "mine", {"slots": {}})
    st = _NS(track="arena", car="corsa", engine="stock", wet="none", abs=False, tc=False,
             steer_aid=True, gearbox="auto", as_dict=lambda: dict(track="arena"))
    over = [dict(slot="top", wing="huge", span=2.6, limit=1.9752)]
    who = _NS(build_name="huge", build_json={"slots": {}})
    ru, _w = session_recorder(who, st, 0.63, root=root, over=over)
    ro, _w = session_recorder(who, st, 0.63, root=root)
    lu = ru._light(dict(date="d", trace=[(0.0,) * 9]), 55.0, [18.0, 18.5, 18.5])
    lo = ro._light(dict(date="d", trace=[(0.0,) * 9]), 55.0, [18.0, 18.5, 18.5])
    rep("an Unlimited session records into its own book, the same class key",
        ru.book.unlimited and ru.book.root == os.path.join(root, UNLIMITED_DIR)
        and ru.key == ro.key == ku and not ro.book.unlimited and ro.book.root == root
        and lu.get("unlimited") is True and lu.get("over_limits") == over
        and "unlimited" not in lo and "over_limits" not in lo,
        f"{os.path.relpath(ru.book.root, root)}/{class_file(ku)}; the lap carries "
        f"unlimited + {len(lu['over_limits'])} reason")
    lap_u = dict(_fake_rec(55.0, [18.0, 18.5, 18.5]), unlimited=True, over_limits=over)
    pos_u = ru.book.insert(ku, lap_u)
    ru.book.set_best_medal(ku, "gold")
    fresh_o, fresh_u = RecordBook(root), unlimited_book(root)
    rep("...where it is P1 and gold, and the official book is untouched",
        pos_u == 1 and fresh_u.pb_time(ku) == 55.0 and fresh_u.load(ku)["best_medal"] == "gold"
        and fresh_o.pb_time(ku) == 62.0 and fresh_o.load(ku)["best_medal"] == "bronze"
        and len(fresh_o.laps(ku)) == 1 and fresh_o.best_sectors(ku) == [20.0, 21.0, 21.0]
        and len(fresh_o.load(ku)["build_bests"]) == len(off.load(ku)["build_bests"]),
        f"Unlimited PB {fresh_u.pb_time(ku)} {fresh_u.load(ku)['best_medal']}, official PB "
        f"{fresh_o.pb_time(ku)} {fresh_o.load(ku)['best_medal']}")
    try:
        off.insert(ku, lap_u)
        refused = False
    except ValueError:
        refused = True
    rep("an official book refuses a lap that says it is Unlimited",
        refused and [lp["time"] for lp in RecordBook(root).laps(ku)] == [62.0])
    ru.book.set_last_build("linden", "huge", {"slots": {}})
    rep("the per-map builds stay in the official folder whichever book writes them",
        ru.book.last_build("arena")["name"] == "mine"
        and RecordBook(root).last_build("linden")["name"] == "huge"
        and not os.path.exists(os.path.join(root, UNLIMITED_DIR, LAST_BUILDS_FILE)))
    rep("publishable(): the public boards' one hook -- never an Unlimited lap",
        publishable(fresh_o.laps(ku)[0]) is True and publishable(fresh_u.laps(ku)[0]) is False
        and publishable(lu) is False and publishable(lo) is True and publishable(None) is False)
    n_pb = lap_note(dict(time=55.0, valid=True, pos=1, pb_before=float("nan"), unlimited=True))[0]
    n_p3 = lap_note(dict(time=57.0, valid=True, pos=3, pb_before=55.0, unlimited=True))[0]
    rep("the HUD note says UNLIMITED plainly", n_pb.startswith("UNLIMITED PB 55.000")
        and n_p3.startswith("UNLIMITED LAP 57.000 P3")
        and lap_note(dict(time=57.0, valid=True, pos=3, pb_before=55.0))[0].startswith("LAP 57"),
        f"{n_pb!r}, {n_p3!r}")
    rep("a scripted / headless run still records nothing, Unlimited or not",
        session_recorder(_NS(headless=True), st, 0.63, root=root, over=over)[0] is None
        and session_recorder(_NS(script="x"), st, 0.63, root=root, over=over)[0] is None)

    # -- a real lap: recorded, filed, re-simulated bit for bit ------------
    # (the full scripted 3-lap acceptance is drive.py's V31; this one is a
    #  skidpad lap in a BALLASTED MX-5 with the ABS and TC on and a designed
    #  panel, so the snapshot carries a CarSpec, the aids' memories and a
    #  DevAero)
    try:
        res = _replay_probe(root)
        rep("a ballasted mx5 skidpad lap re-simulates bit for bit", res["exact"],
            f"lap {res['time']:.6f} s, replay {res['replay']:.6f} s, "
            f"{res['bytes'] / 1024:.0f} KB on disk, {res['n_log']} logged steps")
    except Exception as exc:               # noqa: BLE001
        import traceback
        traceback.print_exc()
        rep("a ballasted mx5 skidpad lap re-simulates bit for bit", False, f"{type(exc).__name__}: {exc}")

    if verbose:
        print(f"  {'ALL PASS' if ok else 'FAILURES ABOVE'}: {n_ok}/{n_all} checks")
    return ok


def _lifecycle_probe(root: str) -> dict:
    """The recorder against a KINEMATIC car on the arena (s commanded, no
    physics), through a real `LapTimer`: a reverse over the line and back,
    a lap in slow motion, one single-stepped, and a clean lap."""
    from types import SimpleNamespace
    from . import track as trk
    from .drive import LapTimer
    tr = trk.make_arena()
    L = float(tr.length)
    book = RecordBook(root)
    key = class_key("arena", "corsa", "stock", "patch")
    out = {}
    rec = LapRecorder(book, key, {}, 1.0, 0.001, on_lap=lambda r: out.setdefault("_q", []).append(r))
    veh = SimpleNamespace(x=0.0, y=0.0, psi=0.0, u=0.0, wing_deploy=0.0, wing_side=0,
                          top_deploy=0.0, car=None, cfg=None)
    sim = SimpleNamespace(track=tr, lap=LapTimer(tr), s=L - 5.0, t=0.0, n=0, dt=0.001, veh=veh,
                          mu=[1.0] * 4, crr=[1.0] * 4, on_track4=[True] * 4, global_wet=1.0,
                          time_scale=1.0, paused=False)

    def drive(s_of_t, T, flags=None):
        t0 = sim.t
        for _ in range(int(round(T / sim.dt))):
            s_prev = sim.s
            sim.s = s_of_t(sim.t - t0 + sim.dt) % L
            if flags:
                flags(sim.t - t0)
            for e in sim.lap.update(sim.t, s_prev, sim.s, sim.dt):
                rec.event(sim, e)
            sim.n += 1
            sim.t = sim.n * sim.dt
            rec.after_step(sim)

    s0 = L - 5.0
    # over the line (start), back behind it, and forward over it again after
    # the 3 s lockout: the timer calls that a lap
    drive(lambda t: s0 + 10.0 * min(t, 0.8) - 10.0 * max(0.0, min(t - 0.8, 0.6))
          + 10.0 * max(0.0, t - 4.4), 5.0)
    out["reverse"] = out["_q"][-1]
    out["reverse_filed"] = len(book.laps(key))
    # now whole laps at 40 m/s from just behind the line
    s1 = sim.s

    def lap_at(s_start):
        return lambda t: s_start + 40.0 * t
    drive(lap_at(s1), L / 40.0,
          flags=lambda tt: setattr(sim, "time_scale", 0.25 if 5.0 < tt < 10.0 else 1.0))
    sim.time_scale = 1.0
    out["slowmo"] = out["_q"][-1]
    s2 = sim.s
    drive(lap_at(s2), L / 40.0, flags=lambda tt: setattr(sim, "paused", 3.0 < tt < 3.002))
    sim.paused = False
    out["step"] = out["_q"][-1]
    s3 = sim.s
    drive(lap_at(s3), L / 40.0 + 0.1)
    out["full"] = out["_q"][-1]
    rec.close()
    return out


def _replay_probe(root: str) -> dict:
    """Drive a scripted skidpad lap with a recorder on, file it, re-simulate
    it from the file."""
    import cars
    from dataclasses import replace
    from . import track as trk
    from .vehicle import Controls, Vehicle, VehicleConfig, DevAero
    from .drive import Sim, ScriptedInput, PathFollower

    base = cars.get("mx5")
    car = cars.with_masses(base, [cars.ballast_point(base, 75.0, "boot")])
    cfg = VehicleConfig(mu_scale=float(base.mu_scale), abs_on=True, tc_on=True, power_scale=1.5,
                        dev_left=DevAero.legacy("plate", 0.97, 0.90, 2.0),
                        dev_right=DevAero.legacy("plate", 0.97, 0.90, 2.0))
    tr = trk.make_track("skidpad", 50.0, False, surfaces=True)
    veh = Vehicle(car, cfg)
    s0 = tr.length - 15.0
    x, y = trk.point_at(tr, s0, 0.0)
    _, _, _, psi, _ = trk.project(tr, x, y)
    veh.reset(x, y, psi, V=15.0, gear=2)
    drv = PathFollower(16.0, wing_on=True)
    sim = Sim(veh, tr, ScriptedInput(drv), dt=0.001)
    sim.s = s0
    book = RecordBook(root)
    key = class_key("skidpad", "mx5", "tuned", "patch")
    rec = LapRecorder(book, key, dict(build_name="probe", build_json=None,
                                      assists=dict(abs=True, tc=True), settings={}),
                      expect_global_wet=1.0, dt=sim.dt)
    sim.recorder = rec
    for _ in range(int(60.0 / sim.dt)):
        sim.step_physics(sim.dt)
        if rec.n_valid:
            break
    rec.close()
    if not rec.n_saved:
        raise RuntimeError(f"no lap recorded ({rec.last})")
    lap = RecordBook(root).pb(key)                 # from the FILE
    rs = resimulate(lap)
    return dict(exact=rs["exact"], time=lap["time"], replay=rs["time"] or float("nan"),
                bytes=lap_size_bytes(lap), n_log=rs["n_log"])


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
