"""drive/aerobo_bridge.py -- AeroBO's own engine, as the garage's DESIGN page
calls it.

The DESIGN page used to run carsim's port of AeroBO. It now runs AeroBO
itself: `aerobo/` at the repository root is the owner's engine vendored
UNMODIFIED (`aerobo/VENDORED.md` says which commit and how to re-sync), and
this module is the only file in carsim that imports it. Everything carsim
has to adapt is here, carsim-side, so no vendored file is ever edited:

* ENVIRONMENT (`xfoil_ok`, `bo_ok`, `warm`): `aerobo/src` first on the path,
  AeroBO pointed at the XFOIL carsim already uses, the imported package
  asserted to be the vendored copy, and the warm library-screen checkpoint
  installed into `aerobo/results/` on first import (without it the first
  screen runs XFOIL over the whole library).
* SLOT FAMILIES (`FamilyParams`, `ensure_family`): AeroBO's car rear-wing
  families, re-registered per slot. The only engine switch for ground
  proximity is the class constant `RIDE_HEIGHT_BOUNDS_M` (the evaluator
  refuses any ride height outside it -- there is no flag), so a slot family
  calls AeroBO's OWN builder and re-instantiates the built problem as a
  carsim subclass that moves only that band (plus the deck, carsim's air and,
  for the lap objective, carsim's circuit and car). At the family's own band
  the clone is AeroBO's family bit for bit (row B7). The TOP slot flies with
  ground effect over the car's deck; the FLANKS fly the same design with the
  image plane pushed `OFFSET_M` away -- ground effect off, mirrored.
* THE OPERATING POINT (`operating_point`, `section_conditions`): the stated
  mission's lap supplies AeroBO's speed; the section conditions are AeroBO's
  own arithmetic at the middle of the box the run will search.
* V3's CONFIGURATION BUILDERS (`shape_kwargs`, `section_search`,
  `wing_physics_flags`, `wing_plan`, `wing_search`, `search_flags`,
  `wing_cfg`, ...): the arguments AeroBO's V3 app sends, for the car, with
  the same semantics as `gui/v3/config.py` / `session.py` /
  `stages/airfoil.py` (which are not vendored -- the GUI is AeroBO's, not
  carsim's).
* RUNNERS (`screen_runner`, `section_runner`, `wing_runner`, ...): what each
  background job executes. A runner is `runner(emit, stop) -> result` (the
  contract `design_jobs.EngineJob` drives, PLAN2 section 6.1): it reports
  through `emit(kind, **payload)` only and stops cooperatively through AeroBO's
  own `stop_rule` / `cancel` hooks reading `stop.is_set()`.
* THE LAP (`track_spec`, `car_spec`): carsim's circuits and the Corsa as
  AeroBO's `cartrack` reads them, for the lap-time objective.
* FIT TO THE CAR (`derive_law`, `to_wingspec`, `section_to_airfoilspec`): the
  law the game flies is SAMPLED from AeroBO's evaluator at the winning design,
  so the car feels AeroBO's forces (row B19: equal to 1e-9).
* FIXTURES (`capture`): real runs recorded once, replayed by the UI's
  deterministic checks.

No pygame; importable headless. `python3 -m drive.aerobo_bridge` is the
self-check (rows B1-B32; the four XFOIL rows skip without XFOIL).
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from .aero import xfoil as _xf
from .aero.polar import RHO as CARSIM_RHO, NU as CARSIM_NU


# =========================================================================== #
#  ENVIRONMENT                                                                 #
# =========================================================================== #
REPO = Path(__file__).resolve().parents[1]
AEROBO_ROOT = REPO / "aerobo"
AEROBO_SRC = AEROBO_ROOT / "src"
#: the warm library-screen checkpoint the vendored tree carries (it is not in
#: AeroBO's git: results/ is ignored there) and where AeroBO reads it from --
#: `api.SCREEN_CHECKPOINT`, `Path(api.__file__).parents[2] / "results"`
SEED = AEROBO_ROOT / "seed" / "airfoil_screen_checkpoint.json"
RESULTS = AEROBO_ROOT / "results"
#: run records carsim keeps (runs/ is already ignored)
RUNS = REPO / "runs" / "aerobo"
#: the commit the vendored tree and every captured fixture name
AEROBO_COMMIT = "3f1b07d"


def _setup_path() -> None:
    """`aerobo/src` first on the path, and AeroBO pointed at carsim's XFOIL.

    BEFORE the import: `xfoil_run.DEFAULT_XFOIL_BIN` is resolved when that
    module loads (`AEROBO_XFOIL_BIN` -> `which xfoil` -> Homebrew), so an env
    var set afterwards would be read by nothing. `setdefault`, so a player
    who points AeroBO somewhere on purpose keeps their choice."""
    src = str(AEROBO_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    xb = _xf.binary()
    if xb:
        os.environ.setdefault("AEROBO_XFOIL_BIN", xb)


_setup_path()
from aerobo import api                                  # noqa: E402
from aerobo import carwing, endplate, cartrack          # noqa: E402
from aerobo import carmount                              # noqa: E402
from aerobo import sizing as _sizing                     # noqa: E402
from aerobo.optimize import budget as _budget           # noqa: E402

#  An installed development copy of AeroBO must never be picked up silently:
#  it would be a different engine answering under this one's name.
if AEROBO_SRC.resolve() not in Path(api.__file__).resolve().parents:
    raise ImportError(f"aerobo was imported from {api.__file__}, not from the vendored "
                      f"copy under {AEROBO_SRC} -- another aerobo is already on the path")


def install_seed() -> bool:
    """Copy the warm checkpoint into `aerobo/results/` if it is not there.

    Written to a temporary name and `os.replace`d, so a second process (or a
    kill mid-copy) can never leave AeroBO reading half a file. False when
    nothing was copied (already installed, or no seed in this tree)."""
    dst = Path(api.SCREEN_CHECKPOINT)
    if dst.is_file() or not SEED.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    shutil.copyfile(SEED, tmp)
    os.replace(tmp, dst)
    return True


install_seed()


def xfoil_ok() -> bool:
    """carsim's rule for live XFOIL: the binary is present and the player (or a
    test) has not switched it off with CARSIM_NO_XFOIL. Without it screening
    runs at the cached library point only and shape optimisation is refused
    (PLAN2 D11)."""
    return _xf.available()


#: why a section run is refused without XFOIL -- one sentence, shown beside
#: the button that would have launched it
XFOIL_REFUSAL = ("shape optimisation needs XFOIL (every candidate is a live viscous sweep); "
                 "it is not installed here or CARSIM_NO_XFOIL is set -- the library screen "
                 "still works, at the cached library point")

_BO = {}


def bo_ok() -> bool:
    """Can AeroBO's Bayesian optimisers run here? torch AND botorch importable.

    Checked by `find_spec` rather than by importing: the import costs the
    frame ~0.4 s (PLAN2 F15) and belongs on a worker thread (`warm`).
    CARSIM_NO_TORCH=1 simulates a machine without the stack."""
    if os.environ.get("CARSIM_NO_TORCH"):
        return False
    if "ok" not in _BO:
        _BO["ok"] = all(importlib.util.find_spec(m) is not None for m in ("torch", "botorch"))
    return _BO["ok"]


def _torch_threads() -> None:
    """TWO torch threads, whatever the core count. AeroBO's GP is fitted on at
    most a few hundred points, where extra intra-op threads buy nothing and,
    on a loaded machine, cost a great deal: MEASURED 2026-09-25 on the M2 Pro
    (12 cores, load average ~11 with a game running), 16 BO evaluations of
    'car rear wing + endplates' took 11.3 s at torch's default 8 threads and
    5.1 s at 2. Two also leaves the rest of the machine to the game's frame."""
    if _BO.get("threads") or not bo_ok():
        return
    import torch
    torch.set_num_threads(2)
    _BO["threads"] = True


def warm() -> None:
    """Import torch and botorch in the CALLING thread (a warm-up job runs this
    on the engine's worker when the design page first opens, so the first
    real run does not pay the import inside the frame)."""
    if not bo_ok():
        return
    import torch                                        # noqa: F401
    import botorch                                      # noqa: F401
    _torch_threads()


def torch_note() -> str:
    """AeroBO's own sentence for a machine without the BO stack ("" with it)."""
    if bo_ok():
        return ""
    if os.environ.get("CARSIM_NO_TORCH"):
        return ("Bayesian optimisation and the BO→SLSQP handoff are switched off "
                "(CARSIM_NO_TORCH) -- only the optimisers that need no torch are offered.")
    return api.torch_optimiser_note()


def study_stamp() -> dict:
    """Where the recommended budgets come from: AeroBO's measured study."""
    return _budget.study_stamp()


# =========================================================================== #
#  SLOT FAMILIES (PLAN2 D2-D4)                                                 #
# =========================================================================== #
#: the flank's image plane is pushed this far away: ground effect OFF. The
#: plate's REACH (ride - deck) is still the standoff to the car's side, and the
#: image is 2 x OFFSET_M off. Measured convergence of CZ with the offset: 20 m
#: 0.6480719, 50 m 0.6480516, 200 m 0.6480479, 1000 m 0.6480476 -- at 100 m the
#: ground term is below 5e-6 relative (row B9 holds it under 1e-5).
OFFSET_M = 100.0

#: AeroBO's car families the Wing type card can derive, keyed (plates, chord law).
#: The default is the V3 car session's own family (`make_session("track")`).
BASES = {("plates", "law"): "car rear wing + endplates + free chord law",
         ("plates", "taper"): "car rear wing + endplates",
         ("fences", "law"): "car rear wing + free chord law",
         ("fences", "taper"): "car rear wing"}
#: AeroBO's designed-plate family (`api._PLATE_BASE`): the one base the plate's
#: two freedoms (`api.PLATE_FREEDOMS`, "cant" and "blend") are generated on --
#: `api.plate_freedom_name` brackets them in, "[free cant, free blend]" -- and
#: the suffix every car family takes for the free chord law (all eight
#: designed-plate names exist in `api.PROBLEM_SPECS`, row B28)
PLATE_BASE = BASES[("plates", "taper")]
LAW_SUFFIX = " + free chord law"

#: the flags a slot family decides and a caller may NOT send: where the car's
#: deck is (the slot's geometry) and which circuit a lap is timed on (carsim's,
#: through `FamilyParams.lap`)
SLOT_OWNED_FLAGS = ("deck_height_m",) + tuple(api.CAR_LAP_KEYS)

#: bumped whenever a slot family's builder changes what f(x) means. It is part
#: of every family's NAME, and AeroBO's evaluation cache keys on the name
#: (`eval_cache.problem_fingerprint`) -- so a changed bridge can never be
#: answered out of a cache the old one filled.
BRIDGE_PHYSICS = 1


@dataclass(frozen=True)
class FamilyParams:
    """Everything that makes a slot family differ from AeroBO's.

    `lap` is (circuit name, surface, mu_scale, x_t, mass_kg) -- the circuit
    and the car the lap objective is timed on -- and only on the TOP slot of a
    plain-fence family (AeroBO's designed-endplate class has no track_spec).
    Every field is in the family's name, so two slots that differ in any of
    them are two problems to the engine and its cache.

    WHAT CARRIES THE WING (the owner, 2026-09-25: "Carried by endplate still
    produces inboard pylons"). `plates` IS the mount, as in AeroBO's own card:
    the endplates carry it (a designed plate, stage 2.8) or a swan-neck PYLON
    pair does (`pylons`, the fence family with AeroBO's continuum mount at
    `carmount.INBOARD_STATION_FRAC`, its struts reaching down to `deck` -- so a
    pylon family carries the deck too). `free` is the designed plate's
    freedoms AeroBO searches instead of being told (a sorted subset of
    `api.PLATE_FREEDOMS`: "blend", "cant"). A fence family WITHOUT `pylons` is
    the legacy one every record before this flew: plain plates at the tips.

    `free` and `pylons` are out of the repr and folded into the name's hash
    only when they are not their defaults (`family_name`), so every family
    name registered before them -- and AeroBO's evaluation cache, which keys
    on it -- is byte-identical (row B28)."""

    role: str                   # "top" | "flank"
    plates: bool                # the endplates carry it (designed, stage 2.8) vs fences
    chord_law: bool             # free chord law (k1..k3) vs straight taper
    ride_band: tuple            # (lo, hi) m: the subclass's RIDE_HEIGHT_BOUNDS_M
    deck: float | None          # deck_height_m (plates or pylons: 0 < deck < ride_band[0])
    lap: tuple | None = None
    rho: float = CARSIM_RHO     # carsim's air (PLAN2 D3), not AeroBO's 1.225
    mu: float = CARSIM_RHO * CARSIM_NU
    free: tuple = field(default=(), repr=False)     # the plate's searched freedoms
    pylons: bool = field(default=False, repr=False)  # a fence family on the pylon pair

    def __post_init__(self):
        free = tuple(sorted({str(k) for k in (self.free or ())}))
        bad = [k for k in free if k not in api.PLATE_FREEDOMS]
        if bad:
            raise ValueError(f"the plate's freedoms are {list(api.PLATE_FREEDOMS)}, got {bad}")
        if free and not self.plates:
            raise ValueError("only a DESIGNED plate has a lean or a blend to search: a fence "
                             "family states them")
        if self.pylons and self.plates:
            raise ValueError("one mount: the endplates carry the wing, or the pylons do")
        object.__setattr__(self, "free", free)
        object.__setattr__(self, "pylons", bool(self.pylons))

    @property
    def base(self) -> str:
        """AeroBO's family: the designed plate with its freedoms bracketed
        in by AeroBO's own name builder, or the fence family."""
        if self.plates:
            return (api.plate_freedom_name(PLATE_BASE, set(self.free))
                    + (LAW_SUFFIX if self.chord_law else ""))
        return BASES[("fences", "law" if self.chord_law else "taper")]

    @classmethod
    def from_json(cls, d: dict) -> "FamilyParams":
        """A record's `carsim_family` (asdict) back into the frozen key. A
        record written before `free` / `pylons` existed has neither: it is
        the family it flew (a fence record: the legacy tip-borne fences)."""
        d = dict(d)
        d["ride_band"] = tuple(float(v) for v in d["ride_band"])
        d["deck"] = None if d.get("deck") is None else float(d["deck"])
        lap = d.get("lap")
        d["lap"] = None if lap is None else tuple(lap)
        d["free"] = tuple(d.get("free") or ())
        d["pylons"] = bool(d.get("pylons", False))
        return cls(**d)


def family_name(p: FamilyParams) -> str:
    """`carsim <role> · <AeroBO family> #<8 hex>` -- readable, and unique per
    parameter set (the hash covers every field and the bridge's physics rev).
    `free` / `pylons` join the hash only when set, so a family that has
    neither keeps the name it was registered under before they existed."""
    src = repr((BRIDGE_PHYSICS, p))
    if p.free or p.pylons:
        src += repr(("free", tuple(p.free), "pylons", bool(p.pylons)))
    key = hashlib.sha1(src.encode()).hexdigest()[:8]
    return f"carsim {p.role} · {p.base} #{key}"


_SUBCLASSES: dict = {}
_FAMILIES: dict = {}


def _slot_class(base_cls, band: tuple):
    """AeroBO's problem class with ONLY the ride band moved (cached per band)."""
    key = (base_cls, tuple(float(v) for v in band))
    if key not in _SUBCLASSES:
        _SUBCLASSES[key] = type(f"{base_cls.__name__}_carsim_{band[0]:g}_{band[1]:g}", (base_cls,),
                                {"RIDE_HEIGHT_BOUNDS_M": (float(band[0]), float(band[1]))})
    return _SUBCLASSES[key]


def _init_fields(problem) -> dict:
    """Every init field of a built problem, as the kwargs that rebuild it."""
    return {f.name: getattr(problem, f.name) for f in dataclasses.fields(problem) if f.init}


def _evaluators(q):
    """AeroBO's (fg, evaluate) pair for whichever car class `q` is."""
    if isinstance(q, endplate.CarWingEndplateProblem):
        return (lambda x: endplate.fg_car_wing_endplate(x, q),
                lambda x: endplate.evaluate_car_wing_endplate(x, q))
    return (lambda x: carwing.fg_car_wing(x, q), lambda x: carwing.evaluate_car_wing(x, q))


def ensure_family(p: FamilyParams) -> str:
    """Register (idempotent) the slot family `p` and return its name.

    Its builder calls AeroBO's OWN builder for the base family, then
    re-instantiates the built problem as a subclass that moves only
    `RIDE_HEIGHT_BOUNDS_M`, with the deck, carsim's air and (lap family) the
    circuit and car set on the copied init fields. The user's box is applied
    twice, on purpose: AeroBO's builder reads the size rows into the problem
    (`_car_size_band_kwargs`), and the ride row -- which AeroBO's band would
    refuse -- is re-applied on the subclass's own bounds, so a user's ride band
    inside the slot's still narrows it."""
    if isinstance(p, dict):
        p = FamilyParams.from_json(p)
    name = family_name(p)
    if name in _FAMILIES and name in api.PROBLEM_SPECS:
        return name
    base = api.PROBLEM_SPECS[p.base]
    lap_fields = None
    if p.lap is not None:
        if p.plates or p.role != "top":
            raise ValueError("a lap is timed only on the TOP slot of a plain-fence family: WingLab's "
                             "designed-endplate class has no track_spec, and cartrack's lap cannot "
                             "value a lateral device")
        lap_fields = _lap_fields(p.lap)

    def build(mission_kwargs, flags, bounds_overrides):
        flags = {k: v for k, v in (flags or {}).items() if k not in SLOT_OWNED_FLAGS}
        if lap_fields is not None:
            # AeroBO constructs a lap-capable problem only when told a circuit;
            # the synthetic one is replaced by carsim's below
            flags["car_track"] = "synthetic"
        inner = {k: v for k, v in (bounds_overrides or {}).items() if k != "ride_height_m"}
        built = base.build(mission_kwargs, flags, inner or None)
        prob = built.problem
        kw = _init_fields(prob)
        kw["rho"], kw["mu"] = float(p.rho), float(p.mu)
        if p.deck is not None and kw.get("mount_spec") is not None:
            # the PYLONS reach down to the slot's deck, which AeroBO keeps on
            # the MountSpec (the problem refuses a second deck beside one), so
            # the pylon is ride - deck long -- what the breakdown reports (row
            # B29). Set HERE, on the copied fields, and not as the
            # `deck_height_m` flag: AeroBO's own class checks a deck against
            # its OWN ride band (0.05-0.6 m) before the subclass moves it, and
            # refuses the car's 0.9 m deck under a 1.04-1.85 m wing
            kw["mount_spec"] = dataclasses.replace(kw["mount_spec"], deck_height_m=float(p.deck))
        elif p.deck is not None and p.plates and "deck_height_m" in kw:
            kw["deck_height_m"] = float(p.deck)
        if lap_fields is not None:
            kw.update(lap_fields)
        q = _slot_class(type(prob), p.ride_band)(**kw)
        labels = tuple(built.param_labels)
        fg, ev = _evaluators(q)
        return api._BuiltProblem(problem=q, callable=fg, evaluate=ev,
                                 bounds=api._apply_overrides(q.bounds, labels, bounds_overrides),
                                 is_constrained=True, dim=q.dim, param_labels=labels,
                                 medium=built.medium)

    for attr in ("takes_section", "takes_section_plate", "gates_ar"):
        if hasattr(base.build, attr):
            setattr(build, attr, getattr(base.build, attr))
    spec = dataclasses.replace(base, name=name, display=name, build=build,
                               flags=tuple(f for f in base.flags if f not in SLOT_OWNED_FLAGS))
    spec.__dict__.pop("_default_bounds_cache", None)
    api.PROBLEM_SPECS[name] = spec
    _FAMILIES[name] = p
    return name


def family_of(record: dict) -> str:
    """Re-register the slot family a stored record was run on, and name it.

    Called before ANY engine call that looks a record's problem up
    (`continue_run_config`, `design_report`, a law re-derivation): a record
    outlives the process that registered its family."""
    fam = (record or {}).get("carsim_family")
    if fam is None:
        return str(((record or {}).get("config") or {}).get("problem_name") or "")
    return ensure_family(FamilyParams.from_json(fam))


def choices_of_record(record: dict, base: dict | None = None) -> dict:
    """The Wing type card's answers a stored record was flown with: its mount
    and chord law (`carsim_family`), the plate's freedoms and the numbers it
    STATED (its flags), the tip device (its flags, and "none" = the plate's
    height pinned at 0), the objective and the two limits. `base` (the card
    as it stands) keeps every answer the record does not state -- a lean the
    record SEARCHED keeps the number the player typed, as AeroBO's card does.

    A LEGACY fence record (plain plates at the tips, no pylon flags) re-opens
    as the nearest answer the card still offers: the pylons, vertical plates.
    The record itself keeps its family and re-fits as it flew (`record_mount`)."""
    ch = dict(base or {})
    rec = record or {}
    cfgd = rec.get("config") or {}
    flags = dict(cfgd.get("flags") or {})
    pinned = dict(cfgd.get("pinned") or {})
    free: set = set()
    if rec.get("carsim_family") is not None:
        fp = FamilyParams.from_json(rec["carsim_family"])
        ch["plates"], ch["chord_law"], free = bool(fp.plates), bool(fp.chord_law), set(fp.free)
    if bool(ch.get("plates", True)):
        ch["blend"] = "free" if "blend" in free else "stated"
        ch["cant"] = "free" if "cant" in free else "stated"
        if "blend" not in free:
            ch["blend_frac"] = float(flags.get("blend_frac") or 0.0)
        if "cant" not in free:
            ch["cant_deg"] = float(flags.get("endplate_cant_deg") or CANT_UPRIGHT)
    else:
        if pinned.get("endplate_h_m") is not None and float(pinned["endplate_h_m"]) == 0.0:
            ch["tip"] = "none"
        elif flags.get("endplate_cant_deg") is not None:
            ch["tip"], ch["tip_cant_deg"] = "canted", float(flags["endplate_cant_deg"])
        elif float(flags.get("blend_frac") or 0.0) > 0.0:
            ch["tip"] = "blended"
        else:
            ch["tip"] = TIP_DEFAULT
        if ch["tip"] != "none":
            ch["tip_chord"] = bool(flags.get("endplate_chord_follows"))
    ch["objective"] = str(flags.get("car_objective") or api.CAR_DEFAULT_OBJECTIVE)
    for k in ("drag_budget_n", "downforce_min_n"):
        ch[k] = None if flags.get(k) is None else float(flags[k])
    return ch


def build_family(name: str, flags: dict | None = None, bounds_overrides: dict | None = None,
                 pinned: dict | None = None):
    """AeroBO's built problem for a family (the Design box reads its rows)."""
    spec = api.PROBLEM_SPECS[name]
    built = spec.build({}, dict(flags or {}), bounds_overrides)
    return built


def family_box(name: str, flags: dict | None = None, bounds_overrides: dict | None = None) -> dict:
    """`{label: [lo, hi]}` of the box a run of `name` will search."""
    built = build_family(name, flags, bounds_overrides)
    return {str(lab): [float(lo), float(hi)] for lab, (lo, hi) in zip(built.param_labels, built.bounds)}


# =========================================================================== #
#  THE OPERATING POINT (PLAN2 5.3)                                             #
# =========================================================================== #
from .aero.wing import BOUNDS as _WING_BOUNDS, V_REF as _V_REF  # noqa: E402
#: task 41's per-car body geometry and span limits (pure: numpy, no pygame).
#: Every number below that depends on the CAR -- the flank's span and area
#: rows, the deck under the top wing, the top's ride band, the ceilings a
#: player may open a row to -- is read from here for the build's car and the
#: Settings' Wing limits (Real / Unlimited)
from . import bodies as _bodies  # noqa: E402

#: the car's reference downforce coefficient the main section is designed at:
#: AeroBO V3's `session.REFERENCE_CL` ("a plain reference chosen by this
#: shell, labelled as such, and not a published target"), the default of its
#: track card's design CZ
REFERENCE_CL = 1.0
#: the top wing's lowest ride height over the deck under it: carsim's own
#: packaging rule (`garage.CarBuild.clamp` -> `bodies.top_h_band`: h >=
#: deck(x) + TOP_STOW_GAP 0.06 + 0.08), so the band searched is the band the
#: slot can carry
TOP_DECK_CLEAR_M = float(_bodies.TOP_DECK_CLEAR)
#: ...and the highest on the three stock cars (`BOUNDS["top"]["ride_h"]`,
#: `bodies.TOP_H_MAX_STOCK`); a van's or a bus's is its own
#: (`bodies.top_h_band`: 0.41 m over its roof -- a Citaro's top wing rides
#: ~3.3 m up). Kept for the old callers; the band is `top_ride_band(car, x)`
TOP_RIDE_MAX_M = float(_WING_BOUNDS["top"]["ride_h"][1])
#: the flank panel's standoff from the car's side: carsim's own band. It is the
#: plate's REACH on a flank family (ride - deck, both shifted by OFFSET_M)
FLANK_STANDOFF_M = tuple(float(v) for v in _WING_BOUNDS["flank"]["ride_h"])
#: the flank's packaging span band before task 41 (`BOUNDS["flank"]["span"]`
#: then; the Corsa's sill / roof fit capped it at 0.88 m at h 0.90). Since
#: task 41 the span row's ceiling is the CAR's physical limit
#: (`bodies.span_limit`: the panel's lower tip at the car's own ground
#: clearance -- 1.50 m on the Corsa at h 0.90), read per car by
#: `flank_size_rows`; this tuple is history, no longer a cap
FLANK_SPAN_M = (0.60, 1.30)
AR_MIN = float(_sizing.AR_LIMITS[0])
AR_MAX = float(_sizing.AR_LIMITS[1])
#: AeroBO's own floor on the car's area row (`S_m2` 0.10-0.48 by default); the
#: flank's is a little higher, a panel the size of carsim's smallest
FLANK_AREA_MIN_M2 = 0.12
#: AeroBO's car family's own size rows (carwing.SPAN_BOUNDS_M / AREA_BOUNDS_M2):
#: the top wing's box before the car's limit caps its span
TOP_SPAN_M = tuple(float(v) for v in carwing.SPAN_BOUNDS_M)
TOP_AREA_M2 = tuple(float(v) for v in carwing.AREA_BOUNDS_M2)
#: "past the limit" tolerance: a span AT the limit is legal (bodies')
LIMIT_TOL = float(_bodies.LIMIT_TOL)


# -- the car (task 41): every per-car number, from drive/bodies.py ------------
def car_key(car=None) -> str:
    """A car as the key the models carry ("corsa" for None / "")."""
    if car is None or car == "":
        return "corsa"
    return car if isinstance(car, str) else str(getattr(car, "name", "") or "corsa")


def car_deck(car=None):
    """`deck(x)`: the height of the car's top surface under a top wing at
    station x -- the surface the top slot's height band is measured from
    (`bodies.top_h_band`). The three stock cars keep the garage's own Corsa
    deck (`bodies.legacy_deck_z` == `garage.deck_z`, pinned there), as their
    slot bands do; a van or a bus reads its own shell."""
    return _bodies.legacy_deck_z if _bodies.is_stock(car) else _bodies.body(car).deck_z


def top_ride_band(car=None, x: float = -0.90, deck: float | None = None) -> tuple:
    """The top slot's ride-height band at station x on `car`: `bodies.
    top_h_band` (clear of the deck by 0.14 m, up to 1.85 m on a stock car,
    0.41 m over its roof on a van or a bus). With an explicit `deck` (the
    self-checks' reference car), the floor is that deck + 0.14 m."""
    lo, hi = _bodies.top_h_band(car, float(x))
    if deck is not None:
        lo = float(deck) + TOP_DECK_CLEAR_M
    return float(lo), float(hi)


def flank_h(car, h: float) -> float:
    """A flank's mount height as the car fits it (`bodies.flank_h_band`:
    `CarBuild.clamp`'s rule, and the height `bodies.over_limits` judges)."""
    lo, hi = _bodies.flank_h_band(car)
    return float(min(max(float(h), lo), hi))


def span_limit(role: str, car=None, h: float | None = None) -> float:
    """The PHYSICAL span limit of a `role` wing on `car` (the owner's rule,
    `bodies.span_limit`): a flank's lower tip at the car's ground clearance
    at mount height h (fitted into the car's band), a top wing 1.2 x the
    body's width. What Real mode holds every designed wing to."""
    if role == "flank" and h is not None:
        h = flank_h(car, h)
    return float(_bodies.span_limit(role, car, h))


def limit_words(role: str, car=None, h: float | None = None, unlimited: bool = False) -> str:
    """The slot's span limit in one line, for the pages."""
    b = _bodies.body(car)
    lim = span_limit(role, car, h)
    if role == "flank":
        why = (f"{lim:.2f} m at h {flank_h(car, h if h is not None else 0.90):.2f} m "
               f"(the panel's lower tip at the {b.ground:.2f} m ground clearance)")
    else:
        why = f"{lim:.2f} m (1.2 x the {b.width:.3f} m body width)"
    mode = (f"Unlimited: up to {_bodies.UNLIMITED_FACTOR:g}x, a run past it is filed apart"
            if unlimited else "Real")
    return f"{car_key(car)}: span <= {why} · {mode}"


def over_limits(build, lib, car=None) -> list:
    """`bodies.over_limits`: the fitted wings of `build` past `car`'s
    physical limit ([] = an official build)."""
    return _bodies.over_limits(build, lib, car)


def limits_text(over: list) -> str:
    return _bodies.limits_text(over)


def size_rows(role: str, car=None, h: float = 0.90, unlimited: bool = False) -> dict:
    """The slot's span and area rows (`b_m`, `S_m2`) the Design box opens on:
    `flank_size_rows` for a flank; for the top, AeroBO's own car-family rows
    with the span capped at the car's limit (1.2 x its width) and the area
    kept where every corner is inside AeroBO's AR 3-40."""
    if role == "flank":
        return flank_size_rows(h, car, unlimited)
    b_lo = TOP_SPAN_M[0]
    b_hi = _bodies.span_ceiling("top", car, None, unlimited)
    s_hi = min(b_lo * b_lo / AR_MIN, _bodies.area_ceiling("top", car, None, unlimited))
    s_lo = min(max(TOP_AREA_M2[0], b_hi * b_hi / AR_MAX), 0.5 * s_hi)
    return {"b_m": [round(b_lo, 6), round(b_hi, 6)], "S_m2": [round(s_lo, 6), round(s_hi, 6)]}


def size_caps(role: str, car=None, h: float = 0.90, unlimited: bool = False) -> dict:
    """The most a player may open the span and area rows to on this slot
    (`bodies.span_ceiling` / `area_ceiling`): the car's physical limit in
    Real mode, `UNLIMITED_FACTOR` times it with Settings' Wing limits on
    Unlimited. The Design box clips a typed band to it."""
    hh = flank_h(car, h) if role == "flank" else None
    return {"b_m": float(_bodies.span_ceiling(role, car, hh, unlimited)),
            "S_m2": float(_bodies.area_ceiling(role, car, hh, unlimited))}


@dataclass
class OperatingPoint:
    """What the mission says the wing in one slot flies at (Mission › Design point).

    `V` is the stated lap's mean speed (length / time) unless typed; the air
    is carsim's (PLAN2 D3) so AeroBO's forces ARE the forces the game
    computes. `deck` / `ride_band` are the slot family's: the car's surface
    under a top wing and the band the slot can carry it in, or -- on a flank --
    the image plane pushed `OFFSET_M` away with the standoff as the band."""

    slot: str
    role: str
    V: float
    V_source: str
    x: float
    h: float
    deck: float | None
    ride_band: tuple
    size_rows: dict
    cz_design: float = REFERENCE_CL
    rho: float = CARSIM_RHO
    nu: float = CARSIM_NU
    circuit: str = ""
    surface: str = ""
    mu_scale: float = 1.0
    #: task 41: the car the slot is on (a `cars.py` key), Settings' Wing
    #: limits, the slot's physical span limit there (`span_limit`) and the
    #: ceilings a row may be opened to (`size_caps`)
    car: str = "corsa"
    unlimited: bool = False
    limit: float = 0.0
    size_caps: dict = dataclasses.field(default_factory=dict)

    @property
    def mu(self) -> float:
        return self.rho * self.nu

    def to_json(self) -> dict:
        d = asdict(self)
        d["ride_band"] = list(self.ride_band)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "OperatingPoint":
        d = dict(d)
        d["ride_band"] = tuple(d["ride_band"])
        return cls(**d)


_LAPS: dict = {}


def _lap_v(build, mission, lib):
    """carsim's own lap of the stated circuit with the car as it stands: its
    mean speed is the number the Mission page quotes as the design speed.

    Memoised on what the lap reads -- the circuit, the grip and the car's
    devices as numbers (`MissionAero` is frozen) -- because every slot's
    session re-reads its operating point (made, restated, V typed) and a
    lap is ~30-70 ms of carsim's QSS. (Memo added by the models agent.)"""
    from .aero import mission as ms
    from .track import make_track
    aero = ms.MissionAero()
    if build is not None and lib is not None and hasattr(build, "mission_aero"):
        aero = build.mission_aero(lib)
    key = (str(getattr(mission, "track", "")), float(getattr(mission, "mu_scale", 1.0)), aero)
    r = _LAPS.get(key)
    if r is None:
        r = ms.lap(mission.profile(make_track), aero, mu_scale=mission.mu_scale)
        if len(_LAPS) > 32:
            _LAPS.clear()
        _LAPS[key] = r
    return r


def operating_point(slot: str, role: str, build, mission, V: float | None = None,
                    cz: float | None = None, *, lib=None, deck_z=None, car=None,
                    unlimited: bool = False) -> OperatingPoint:
    """The slot's operating point for AeroBO.

    `build` is the garage's CarBuild (only `.slot(key)` -> x, h and
    `.mission_aero(lib)` are read), `mission` the stated `aero.mission.
    MissionSpec`. `car` is the car the build is fitted to (a `cars.py` key;
    None is the Corsa) and `unlimited` Settings' Wing limits (task 41): the
    top's deck and ride band, the flank's span and area rows, the span limit
    and the ceilings a row may be opened to are that car's
    (`drive/bodies.py`). `deck_z(x)`, when passed, overrides the car's deck
    under the top wing (the self-checks' reference car). `V` / `cz` typed
    override the lap's speed and the reference CZ.

    The design speed is the JOB's (the owner, 2026-09-25: "One more circuit
    should be added 'Stopping'. Left flank and right flank, should be side"):
    a circuit's lap mean speed, the stop's start speed when the top wing's
    job is STOPPING. A flank flies its OWN circuit (`MissionSpec.for_side`,
    the owner 2026-09-26: "why no longer circuits?"), so its speed is that
    lap's mean. `circuit` on the point names the job ("stopping") where there
    is no circuit."""
    from .aero import mission as _msn
    s = build.slot(slot)
    x, h = float(s.x), float(s.h)
    ck = car_key(car)
    if role != "top" and hasattr(mission, "for_side"):
        mission = mission.for_side()
    circuit = str(getattr(mission, "track", ""))
    surface = str(getattr(mission, "surface", ""))
    typed = V is not None and float(V) > 0.0
    if typed:
        v, src = float(V), "typed"
    elif circuit == _msn.STOPPING:
        kmh = float(getattr(mission, "v_stop_kmh", _msn.V_STOP_KMH))
        v, src = kmh / 3.6, f"the stop's start speed ({kmh:.0f} km/h)"
    else:
        r = _lap_v(build, mission, lib)
        if r.ok and r.v_mean > 0.0:
            v, src = float(r.v_mean), f"the {circuit} lap's mean speed ({surface})"
        else:
            v = float(_V_REF["top"])
            src = f"carsim's top reference speed (the {circuit} lap did not close)"
    if role == "top":
        if deck_z is not None:
            deck = float(deck_z(x))
            band = top_ride_band(ck, x, deck=deck)
        else:
            deck = float(car_deck(ck)(x))
            band = top_ride_band(ck, x)
    else:
        deck = OFFSET_M
        band = (OFFSET_M + FLANK_STANDOFF_M[0], OFFSET_M + FLANK_STANDOFF_M[1])
    #  the box opens on the car's REAL limit; Unlimited lets the player open a
    #  row to 3x it (`size_caps`) but does not move the default search
    size = size_rows(role, ck, h, unlimited=False)
    caps = size_caps(role, ck, h, unlimited=bool(unlimited))
    return OperatingPoint(slot=str(slot), role=str(role), V=v, V_source=src, x=x, h=h, deck=deck,
                          ride_band=band, size_rows=size,
                          cz_design=float(cz) if cz is not None and float(cz) > 0.0 else REFERENCE_CL,
                          circuit=circuit, surface=surface,
                          mu_scale=float(getattr(mission, "mu_scale", 1.0)),
                          car=ck, unlimited=bool(unlimited),
                          limit=span_limit("top" if role == "top" else "flank", ck, h),
                          size_caps=caps)


def flank_size_rows(h: float, car=None, unlimited: bool = False) -> dict:
    """The flank's span and area rows at mount height `h` on `car`: the car's
    span ceiling (task 41, `bodies.span_ceiling`: the panel's lower tip at the
    car's ground clearance -- 1.50 m on the Corsa at h 0.90 -- three times
    that in Unlimited) and its area ceiling (`bodies.area_ceiling`), cut to
    AeroBO's AR >= 3. `h` is taken as the car fits it (`flank_h`).

    The area row runs up to the AR-3 area of the tallest panel the slot can
    carry (the biggest honest panel -- what a side-force objective wants),
    under the car's area ceiling. The span row starts at the narrowest panel
    that still carries the MIDDLE of that area row at AR 3, so every span in
    the box can fly at least half the areas. (Before task 41 the Corsa's
    sill / roof fit capped the span at 0.88 m at h 0.90: 76 of 200 Sobol
    draws feasible against AeroBO's own family's 85.) The remaining refusals
    are AeroBO's (the root-largest chord law, AoA validity), the same ones
    its car family meets."""
    hh = flank_h(car, h)
    b_hi = float(_bodies.span_ceiling("flank", car, hh, unlimited))
    s_hi = min(b_hi * b_hi / AR_MIN, float(_bodies.area_ceiling("flank", car, hh, unlimited)))
    s_lo = min(FLANK_AREA_MIN_M2, 0.5 * s_hi)
    b_lo = min(math.sqrt(AR_MIN * 0.5 * (s_lo + s_hi)), b_hi - 0.05)
    return {"b_m": [round(b_lo, 6), round(b_hi, 6)], "S_m2": [round(s_lo, 6), round(s_hi, 6)]}


def _mid(box: dict, key: str) -> float | None:
    row = box.get(key)
    return None if row is None else 0.5 * (float(row[0]) + float(row[1]))


def section_conditions(op: OperatingPoint, box: dict, surface: str,
                       re_source: str = "mission") -> dict | None:
    """The (re, mach, cl_design) a surface's section is screened and designed at.

    AeroBO's arithmetic (`session.section_conditions` + `surface_design_point`
    + `_plate_geometry`), at the MIDDLE of the box the run will search:
    main -- the mean chord S/b (taper is a design variable, so the taper-free
    chord), cl = the reference CZ; plate -- the plate's own chord, a ratio of
    the wing's tip chord `ratio * 2 S lambda / (b (1 + lambda))`, cl = 0 (a
    plate at zero toe makes no side force). With `re_source == "library"` (or
    no XFOIL) the Reynolds number is the cached library point's. None where
    the box has no such surface (a plain-fence family has no plate section)."""
    S, b = _mid(box, "S_m2"), _mid(box, "b_m")
    if S is None or b is None or not (S > 0.0 and b > 0.0):
        return None
    if surface == "plate":
        lam, ratio = _mid(box, "taper"), _mid(box, "endplate_chord_ratio")
        if lam is None or ratio is None or not 1.0 + lam > 0.0:
            return None
        chord, cl = ratio * 2.0 * S * lam / (b * (1.0 + lam)), 0.0
    else:
        chord, cl = S / b, float(op.cz_design)
    re_own = float(op.V) * chord / float(op.nu)
    out = {"re": re_own, "mach": 0.0, "cl_design": cl, "chord": chord, "re_own": re_own,
           "re_source": "mission"}
    if re_source == "library" or not xfoil_ok():
        pt = api.screen_library_point() or {}
        out["re"], out["mach"] = float(pt.get("re", 1e6)), float(pt.get("mach", 0.0))
        out["re_source"] = "library"
    return out


# =========================================================================== #
#  V3's CONFIGURATION BUILDERS (PLAN2 5.4)                                     #
# =========================================================================== #
#: the V3 car session's recommended screening weights. The WING's carries no
#: cd@cl criterion: "cd at the design cl" is redundant with "L/D at the design
#: cl" there (the owner's decision, handoff D9), so it is neither weighted nor
#: a ranking column. The PLATE keeps it -- at cl 0 its drag IS its criterion --
#: and weights nothing that dies at zero lift (ldcr, ldmax, cm).
WING_WEIGHTS = {"ldcr": 0.35, "clmax": 0.2, "cm": 0.2, "ldmax": 0.15, "thick": 0.1, "astall": 0.0}
PLATE_WEIGHTS = {"ldcr": 0.0, "clmax": 0.25, "cm": 0.0, "ldmax": 0.0, "cdcr": 0.35, "thick": 0.2,
                 "astall": 0.2}
#: criteria a surface does not offer at all (the owner's D9) ...
REDUNDANT = {"main": frozenset({"cdcr"}), "plate": frozenset()}
#: ... and the plate's criteria that are identically dead at cl 0 (shown dim)
DEAD_AT_CL0 = frozenset({"ldcr", "ldmax", "cm"})
#: the published screening gates, and the shape optimiser's (stages/airfoil.py)
SCREEN_GATES = {"tc_min": 0.15, "cm_max": 0.08}
OPT_GATES = {"tc_min": 0.10, "cm_max": 0.08}
GATE_OFF = {"tc_min": 0.0, "cm_max": 1.0e9}
#: rows the ranking keeps, and leaders re-swept at the surface's own Re
TOP_N = 14
SHORTLIST = 24
#: the section objectives offered, default first. "pareto" is left out: a front
#: run cannot be stopped mid-search, which contradicts RUNNING -> STOPPED, and
#: it returns a set rather than a section to fly (PLAN2 Q6)
SECTION_OBJECTIVES = ("composite_goal", "composite", "composite_asf", "cd")
#: AeroBO's own labels (stages/airfoil.OBJECTIVE_CHOICES, a car has no wing mode)
SECTION_OBJECTIVE_LABELS = {
    "composite_goal": "composite score, no WEIGHTED criterion below the seed",
    "composite": "composite score of your six criteria",
    "composite_asf": "lift the WEAKEST of your six criteria (Tchebycheff)",
    "cd": "2-D L/D at the design Cl",
}
#: the shell's feasibility policy for a BO run (config.FEASIBILITY_DEFAULT)
FEASIBILITY_DEFAULT = "guide"
#: a section search's form, as V3's car session opens it
OPT_DEFAULT = {"objective": "composite_goal", "tc_min": OPT_GATES["tc_min"],
               "cm_max": OPT_GATES["cm_max"], "twist_order": 1, "twist_max_deg": 6.0,
               "alpha_max_deg": 10.0, "chord_order": 2, "chord_max_frac": 0.5, "censored": None}
#: the stated physics V3 sends every car run (config.stated_physics: the wide
#: end of the chord law is the root)
CHORD_TREND = "root_largest"
#: why a partial run stopped, on the record the player keeps
STOP_PLAYER = "stopped by the player"
STOP_CONVERGED = "the search stopped improving"

#: AeroBO's car objective menu (gui/nice_app.CAR_OBJECTIVE_LABELS), same keys,
#: same order. `cz` / `cd` are never offered: a coefficient against the very
#: area being searched is maximised by shrinking the wing (carwing refuses it).
CAR_OBJECTIVE_LABELS = {
    "efficiency": "efficiency CZ/CD (= downforce per unit drag)",
    "downforce": "downforce, in newtons",
    "drag": "drag, in newtons (minimised — state a downforce floor below)",
    "laptime": "lap time round the chosen circuit (minimised)",
    "downforce_plus_drag": "downforce + drag, in newtons (the total load)",
}


def car_objectives(role: str, plates: bool, circuit: str | None = None) -> dict:
    """The Maximise menu for one slot: AeroBO's car menu, minus what the
    family cannot run. Lap time is timed only on the TOP slot of a plain-fence
    family (AeroBO's designed-endplate class has no track_spec, and cartrack
    cannot value a lateral device); the flank relabels downforce as side force."""
    table = endplate.ENDPLATE_OBJECTIVES if plates else carwing.CAR_OBJECTIVES
    out = {}
    for k, label in CAR_OBJECTIVE_LABELS.items():
        if k not in table or k in ("cz", "cd"):
            continue
        if k == "laptime":
            if role != "top" or plates or str(circuit or "") not in _ms.TRACKS:
                continue                                # no circuit: a stop, or a side wing
            label = f"lap time round {circuit} (minimised)"
        if role == "flank":
            label = label.replace("downforce", "side force")
        out[k] = label
    return out


def shape_kwargs(cond: dict, surface: str, opt: dict | None = None, weights: dict | None = None,
                 reference: dict | None = None, anchor=None) -> dict:
    """Every argument that says WHICH SEARCH a section run is (= stages/airfoil.
    shape_kwargs for a car surface). Kept afterwards as the run's LAUNCH
    SNAPSHOT: "Keep going" re-flies exactly this, never the form.

    `anchor` is (w_upper, w_lower) of the chosen section, or None for the
    family's NACA anchor; `reference` the screen's frozen band (its report's
    "reference"), or None for the shipped one."""
    o = {**OPT_DEFAULT, **(opt or {})}

    def gate(key):
        v = o.get(key)
        return float(GATE_OFF[key] if v is None else v)

    objective = str(o.get("objective") or "composite_goal")
    out = dict(re=float(cond["re"]), mach=float(cond["mach"]), cl_design=float(cond["cl_design"]),
               # a vertical panel searches a SYMMETRIC family: stated HERE, in
               # the snapshot, so a continuation can never resume a plate's
               # search over cambered shapes
               symmetric=(surface == "plate"),
               tc_min=gate("tc_min"), cm_max=gate("cm_max"),
               anchor=([list(map(float, anchor[0])), list(map(float, anchor[1]))]
                       if anchor else None),
               twist_order=int(o["twist_order"]), twist_max_deg=float(o["twist_max_deg"]),
               alpha_max_deg=float(o["alpha_max_deg"]), chord_order=int(o["chord_order"]),
               chord_max_frac=float(o["chord_max_frac"]))
    if objective in api.AIRFOIL_COMPOSITE_OBJECTIVES:
        out.update(objective=objective, wing=None,
                   score_weights=dict(weights if weights is not None else
                                      (PLATE_WEIGHTS if surface == "plate" else WING_WEIGHTS)),
                   score_reference=reference)
        if o.get("censored"):
            out["censored"] = str(o["censored"])
    else:
        out.update(objective="cd", wing=None)
    return out


_PLANS: dict = {}


def _cached(key, fn):
    k = json.dumps(key, sort_keys=True, default=str)
    if k not in _PLANS:
        _PLANS[k] = fn()
    return _PLANS[k]


def section_plan(objective: str = "composite_goal", effort: str = "balanced"):
    """AeroBO's measured plan for a section search (session.airfoil_plan, car:
    no wing guess). Composite objectives share one 2-D problem and budget."""
    name = objective if objective in api.AIRFOIL_COMPOSITE_OBJECTIVES else "cd"

    def build():
        cfg = api.airfoil_run_config(objective=name, wing=None)
        return api.recommended_search(cfg.problem_name, flags=cfg.flags, effort=effort,
                                      objective=name)
    return _cached(("section", name, effort), build)


def _pol(policy, key, default=None):
    """A SearchPolicy field, from the object or a plain dict."""
    if policy is None:
        return default
    if isinstance(policy, dict):
        return policy.get(key, default)
    return getattr(policy, key, default)


def _recommended(policy) -> bool:
    return str(_pol(policy, "mode", "recommended")) == "recommended"


def _optimisers(problem: str) -> list:
    """AeroBO's compatible optimisers, minus the torch ones on a machine without it."""
    names = list(api.compatible_optimisers(problem))
    if not bo_ok():
        names = [n for n in names if n not in api.TORCH_ONLY_OPTIMISERS]
    return names


def _fallback_optimiser(names: list) -> str:
    for n in ("bo_slsqp", "bo", "ga", "sobol"):
        if n in names:
            return n
    return names[0]


def section_search(policy=None, objective: str = "composite_goal", surface: str = "main",
                   seed: int = 0) -> dict:
    """What a section search will ACTUALLY run with (session.effective_airfoil_search).

    Recommended: the plan's optimiser (when this machine can fly it), budget,
    refusal and Sobol seed. "own": the player's budget with the plan's two
    decisions carried -- AeroBO's "use these as my own" keeps refusal and
    n_init, which a hand-typed number has no field for."""
    plan = section_plan(objective, str(_pol(policy, "effort", "balanced")))
    names = _optimisers(api.AIRFOIL_PROBLEM)
    optimiser = plan.optimiser if plan.optimiser in names else _fallback_optimiser(names)
    out = {"optimiser": optimiser, "budget": int(plan.budget), "seed": int(seed),
           "n_restarts": max(1, int(plan.n_restarts)),
           "refusal": plan.refusal if optimiser in api.BO_ITER_OPTIMISERS else None,
           "n_init": plan.n_init if optimiser in api.BO_ITER_OPTIMISERS else None,
           "source": "recommended", "plan": plan}
    if not _recommended(policy):
        own = _pol(policy, "own", {}) or {}
        key = "plate" if surface == "plate" else "airfoil"
        out.update(budget=int(own.get(key, plan.budget)), n_restarts=1, source="own")
    return out


def section_flag_value(chosen: dict | None):
    """What `section_name` / `section_name_plate` carries for a chosen section
    (= session.section_flag_value): a library pick as its NAME, with the point
    it was screened at unless that is the cached library point; a designed
    section as its CST weights and the point it was designed at. None for an
    undecided surface or the family's own section."""
    if not chosen or chosen.get("source") in (None, "default"):
        return None
    cond = chosen.get("conditions") or {}
    if chosen.get("source") == "library" and chosen.get("name"):
        try:
            re_sec = float(cond["re"])
        except (KeyError, TypeError, ValueError):
            return str(chosen["name"])
        mach = float(cond.get("mach") or 0.0)
        lib = api.screen_library_point() or {}
        if lib and abs(re_sec - float(lib["re"])) <= 1e-9 * abs(re_sec) \
                and abs(mach - float(lib.get("mach", 0.0))) <= 1e-12:
            return str(chosen["name"])
        return {"name": str(chosen["name"]), "re": re_sec, "mach": mach}
    w_u, w_l = chosen.get("w_upper"), chosen.get("w_lower")
    if not w_u or not w_l:
        return None
    out = {"w_upper": [float(v) for v in w_u], "w_lower": [float(v) for v in w_l],
           "name": chosen.get("name") or "designed CST"}
    if cond.get("re"):
        out["re"] = float(cond["re"])
    if cond.get("mach"):
        out["mach"] = float(cond["mach"])
    return out


def plate_free(choices: dict | None) -> tuple:
    """The designed plate's freedoms the card hands to the optimiser (sorted
    `api.PLATE_FREEDOMS` keys): each of "Root blend" / "Leaning at" set to
    "optimise it". () under the pylons, whose fence has none to search."""
    ch = choices or {}
    if not bool(ch.get("plates", True)):
        return ()
    return tuple(sorted(k for k in api.PLATE_FREEDOMS if ch.get(k) == "free"))


def family_params(op: OperatingPoint, choices: dict | None = None,
                  lap: tuple | None = None) -> FamilyParams:
    """The slot family an operating point and the Wing type card derive.

    `choices`: {"plates": bool (True: the endplates carry it; False: the
    pylons do), "chord_law": bool, "blend" / "cant": "stated" | "free",
    "objective": str, ...}. A new run on fences is ALWAYS the pylon pair: the
    legacy tip-borne fence is a record's family, no longer offered (AeroBO's
    card has no such answer either). `lap` (from `lap_params`) is attached
    only when lap time is the objective -- it is what makes the family time a
    lap at all."""
    ch = choices or {}
    plates = bool(ch.get("plates", True))
    law = bool(ch.get("chord_law", True))
    if ch.get("objective") == "laptime":
        if lap is None:
            raise ValueError("lap time needs the circuit and the car: pass lap_params(...)")
    else:
        lap = None
    return FamilyParams(role=op.role, plates=plates, chord_law=law,
                        ride_band=tuple(float(v) for v in op.ride_band),
                        # the plates reach the deck, and so do the pylons
                        deck=(float(op.deck) if op.deck is not None else None),
                        lap=lap, rho=float(op.rho), mu=float(op.mu),
                        free=plate_free(ch), pylons=not plates)


#: THE MOUNT, as AeroBO's own car card asks it (`gui/nice_app.
#: _car_mount_controls`): ONE question, "Carried by". The plain `mount` flag
#: is carwing's published "tips" on every run -- the endplates carry the wing
#: there, and under the pylons `api._car_mount_kwargs` builds the MountSpec
#: from it and drops it. carwing's "inboard" (a second pair of PLATES) is not
#: a card answer: the owner, 2026-09-25, "Carried by endplate still produces
#: inboard pylons".
MOUNT_FLAG = "tips"
#: the tip devices a pylon-borne wing's plates can be: AeroBO's own four keys
#: and words (`CAR_TIP_SHAPE_LABELS`). Under the endplates there is no such
#: menu -- the plate IS the mount, and its blend and lean are asked instead
TIP_DEVICES = {"none": "none — no plate at all",
               "vertical": "endplates, vertical (towards the track)",
               "canted": "endplates, canted outboard",
               "blended": "endplates, blended into the wing"}
TIP_DEFAULT = "vertical"
#: does a fitted tip device's CHORD continue the wing's chord distribution
#: (AeroBO's `car_endplate_chord_follows`, the v3 card's "its chord" switch)
#: or hold the wing's tip chord (a rectangle, every published plate)? The
#: player's answer, `choices["tip_chord"]`; ON when unset -- the owner,
#: 2026-09-25, "follows the chord of wing", and 2026-09-26, "tip devices
#: should be given option to follow chord distribution (for wing design both)"
TIP_CHORD_DEFAULT = True
#: the blend a "blended" tip device implies (AeroBO's BLEND_FRAC_DEFAULT: the
#: shape states the value, there is no field) and the law every blend follows
BLEND_FRAC_DEFAULT = 0.5
BLEND_SHAPE = "spiral"
#: a plate's lean, deg from the wing plane: 90 upright (every published plate,
#: and sent as nothing); the band is the nonplanar lattice's own validity
#: (`geometry.WINGLET_CANT_LIMITS_DEG`), not a recommendation
CANT_UPRIGHT = 90.0
from aerobo.geometry import WINGLET_CANT_LIMITS_DEG as _CANT_LIMITS  # noqa: E402
CANT_LIMITS = (float(_CANT_LIMITS[0]), float(_CANT_LIMITS[1]))
#: the lean "canted" picks when the stored one is upright: a canted device at
#: 90 deg would be the vertical one under another name
CANT_SHOWN = 75.0


def pylon_flags() -> dict:
    """What the PYLON mount is, in AeroBO's flags (`gui/nice_app.
    car_mount_layout_flags("pylons")`): a swan-neck pair at AeroBO's own
    "near the middle" station (`carmount.INBOARD_STATION_FRAC`), landing on
    the pressure side -- four statements, every other MountSpec field its
    module's own. The deck it reaches down to is the SLOT's (the family's
    build adds it: `deck_height_m` is slot-owned)."""
    return {"car_mount_model": "continuum", "mount_kind": "pylon",
            "mount_station_frac": float(carmount.INBOARD_STATION_FRAC),
            "mount_side": "pressure"}


def tip_device(choices: dict | None) -> str:
    """The pylon-borne wing's tip device (a `TIP_DEVICES` key; vertical when
    unset). None of it applies under the endplates."""
    t = str((choices or {}).get("tip") or TIP_DEFAULT)
    return t if t in TIP_DEVICES else TIP_DEFAULT


def tip_chord_follows(choices: dict | None) -> bool:
    """Does the tip device's chord follow the wing's chord distribution
    (`TIP_CHORD_DEFAULT` when unset)?"""
    v = (choices or {}).get("tip_chord")
    return TIP_CHORD_DEFAULT if v is None else bool(v)


def _cant(choices: dict | None, key: str = "cant_deg") -> float:
    """A stated lean, inside the lattice's band: `cant_deg` the designed
    plate's, `tip_cant_deg` the canted tip device's (two answers, two keys:
    leaning the pylons' device never leans the endplates behind the
    player's back)."""
    v = (choices or {}).get(key)
    v = CANT_UPRIGHT if v is None else float(v)
    return float(min(max(v, CANT_LIMITS[0]), CANT_LIMITS[1]))


def _blend(choices: dict | None) -> float:
    return float(min(max(float((choices or {}).get("blend_frac") or 0.0), 0.0), 1.0))


def plate_flags(choices: dict | None) -> dict:
    """The designed plate's blend and lean (AeroBO's `car_flags` under the
    tips): a STATED blend is sent as `blend_frac` (+ its law) only above 0 --
    0 is the crease every published run flew; a stated lean as
    `endplate_cant_deg` only off upright. A SEARCHED one is a family's row
    and its flag is never sent (AeroBO refuses the pair); a searched blend
    still states its law. Nothing when the plates are not the mount."""
    ch = choices or {}
    if not bool(ch.get("plates", True)):
        return {}
    free = set(plate_free(ch))
    out = {}
    if "blend" in free:
        out["blend_shape"] = BLEND_SHAPE
    elif _blend(ch) > 0.0:
        out["blend_frac"] = _blend(ch)
        out["blend_shape"] = BLEND_SHAPE
    if "cant" not in free and _cant(ch) != CANT_UPRIGHT:
        out["endplate_cant_deg"] = _cant(ch)
    return out


def tip_flags(choices: dict | None) -> dict:
    """The pylon mount and its tip device, in flags: `pylon_flags`, then
    canted -> `endplate_cant_deg` (its stated lean, `tip_cant_deg`), blended -> AeroBO's
    default blend and law, and a fitted device whose chord FOLLOWS THE WING'S
    (`tip_chord_follows`, on unless the player holds the tip chord) sends
    `endplate_chord_follows` -- a tapered wing gets a tapered plate, no step
    at the tip; off, the plate holds the tip chord (AeroBO's default, the
    flag not sent). "none" sends no
    device flag: it is a PIN on the plate's height (`tip_pins`). Nothing
    when the endplates are the mount."""
    ch = choices or {}
    if bool(ch.get("plates", True)):
        return {}
    out = pylon_flags()
    tip = tip_device(ch)
    if tip == "canted":
        out["endplate_cant_deg"] = _cant(ch, "tip_cant_deg")
    elif tip == "blended":
        out["blend_frac"] = BLEND_FRAC_DEFAULT
        out["blend_shape"] = BLEND_SHAPE
    if tip != "none" and tip_chord_follows(ch):
        out["endplate_chord_follows"] = True
    return out


def tip_pins(choices: dict | None) -> dict:
    """"none" is not a flag -- there is no plate of no height, only a height
    of zero -- so it is AeroBO's PIN: `endplate_h_m` at 0, out of the design
    vector (`gui/v3/stages/wing._set_car_tip`). {} for any other device."""
    ch = choices or {}
    if bool(ch.get("plates", True)) or tip_device(ch) != "none":
        return {}
    return {"endplate_h_m": 0.0}


def wing_physics_flags(choices: dict | None, op: OperatingPoint, sections: dict | None = None,
                       family: str | None = None) -> dict:
    """The physics flags V3 sends a car run (config.flags for the track medium).

    The stated chord trend, the mount (`MOUNT_FLAG`; under the pylons AeroBO's
    continuum pylon pair and its tip device, `tip_flags`), the plate's
    construction section and its stated blend and lean when the plates carry
    the wing (`plate_flags`), the SPEED -- always, since carsim's is never the
    family's 55 m/s -- the objective when it is not AeroBO's default, the two
    limits in newtons where stated, and the sections stages 2 / 2.8 chose.
    Then AeroBO's own sanitiser, the backstop that drops what the family does
    not declare. `family` is the slot family's registered name (required for
    lap time, whose family carries the lap)."""
    ch = choices or {}
    fam = family or ensure_family(family_params(op, ch))
    plates = bool(ch.get("plates", True))
    out = {api.CHORD_TREND_KEY: CHORD_TREND, "mount": MOUNT_FLAG}
    if plates:
        out["section"] = str(ch.get("endplate_section") or "shaped")
        out.update(plate_flags(ch))
    else:
        out.update(tip_flags(ch))
    out["V"] = float(op.V)
    obj = ch.get("objective")
    if obj and obj != api.CAR_DEFAULT_OBJECTIVE:
        out["car_objective"] = str(obj)
    if ch.get("drag_budget_n") is not None:
        out["drag_budget_n"] = float(ch["drag_budget_n"])
    if ch.get("downforce_min_n") is not None:
        out["downforce_min_n"] = float(ch["downforce_min_n"])
    secs = sections or {}
    main = section_flag_value(secs.get("main"))
    if main:
        out[api.SECTION_KEY] = main
    plate = section_flag_value(secs.get("plate")) if plates else None
    if plate:
        out[api.SECTION_PLATE_KEY] = plate
    out, _dropped = api.sanitise_flags(fam, out)
    return out


def wing_plan(family: str, flags: dict, bounds_overrides: dict | None = None,
              pinned: dict | None = None, effort: str = "balanced"):
    """AeroBO's measured plan for the wing (session.wing_plan; objective "lod")."""
    return _cached(("wing", family, flags, bounds_overrides, pinned, effort),
                   lambda: api.recommended_search(family, flags=flags, mission_kwargs={},
                                                  bounds_overrides=bounds_overrides, effort=effort,
                                                  objective="lod", pinned=pinned or None))


def wing_search(policy, plan, family: str, seed: int = 0) -> dict:
    """What the wing run will ACTUALLY fly with (session.effective_wing_search).

    Recommended: the plan's budget flown as `bo_slsqp` where the family and
    this machine can (the recommender cannot name the handoff arm -- its study
    predates it -- and the later certified measurement puts BO -> SLSQP above
    pure BO on 42 of 54 runs), with no n_init (the handoff's BO phase owns a
    quarter of the budget and `_bo_split` clamps it). "own": the player's
    budget, the same decisions."""
    names = _optimisers(family)
    if plan is None:
        opt = _fallback_optimiser(names)
        return {"optimiser": opt, "budget": int(_pol(policy, "own", {}).get("wing", 53)),
                "n_init": None, "seed": int(seed), "refusal": "sentinel", "source": "own",
                "plan": None, "patience": None, "tol": None}
    optimiser = plan.optimiser if plan.optimiser in names else _fallback_optimiser(names)
    out = {"optimiser": optimiser, "budget": int(plan.budget), "n_init": plan.n_init,
           "seed": int(seed), "refusal": plan.refusal, "source": "recommended", "plan": plan,
           "patience": plan.patience, "tol": plan.tol}
    if optimiser == "bo" and "bo_slsqp" in names:
        out.update(optimiser="bo_slsqp", n_init=None)
    elif optimiser != plan.optimiser:
        # the study's winner is not offered here: keep its BUDGET (a property
        # of the problem's size) and say which optimiser is flying
        out.update(n_init=None, refusal="sentinel")
    if not _recommended(policy):
        own = _pol(policy, "own", {}) or {}
        out.update(budget=int(own.get("wing", plan.budget)), source="own")
    return out


def search_flags(base: dict, eff: dict) -> dict:
    """`base` with the SEARCH decisions applied (config.search_flags): the
    Sobol seed size, the refusal imputation and the feasibility policy reach
    the two arms that run a BO loop directly (`bo`, `bo_slsqp`)."""
    out = dict(base)
    for k in ("acqf", api.BO_N_INIT_FLAG, api.BO_REFUSAL_FLAG, api.BO_FEASIBILITY_FLAG,
              "block_optimisers"):
        out.pop(k, None)
    if eff["optimiser"] == "blocks":
        out[api.BO_FEASIBILITY_FLAG] = FEASIBILITY_DEFAULT
    if eff["optimiser"] not in api.BO_ITER_OPTIMISERS:
        return out
    if eff.get("n_init"):
        out[api.BO_N_INIT_FLAG] = int(eff["n_init"])
    if eff.get("refusal") and eff["refusal"] != "sentinel":
        out[api.BO_REFUSAL_FLAG] = str(eff["refusal"])
    out[api.BO_FEASIBILITY_FLAG] = FEASIBILITY_DEFAULT
    return out


#: the designed plate's searched rows and the value that states the plate
#: every published run flew: upright, meeting the wing in a crease
_PLATE_STATED = {"endplate_cant_deg": CANT_UPRIGHT, "endplate_blend_frac": 0.0}


def plate_seeded(choices: dict | None) -> bool:
    """Does a run of these card answers take `plate_seed`'s warm start? When
    the endplates carry the wing and the plate is anything but the upright
    crease every published run flew: a lean or a root blend handed to the
    optimiser, OR a STATED blend above 0 or lean off upright. A stated blend
    shortens the plate's reach and spends span outboard exactly as a
    searched one does, so its feasible box is as small (0 of 128 Sobol points
    on a side slot at blend 0.8). The upright crease stays AeroBO's own
    unseeded configuration."""
    ch = choices or {}
    return bool(ch.get("plates", True)) and bool(plate_free(ch) or plate_flags(ch))


#: the reach candidate's plate is this much longer than the least that
#: reaches the bottom of the ride band (the margin is h_ep's, not the ride's)
SEED_REACH_SLACK = 1.10
#: ...and, only when none of the three flies, the same candidate on a plate
#: just longer than the least: a full blend or a shallow lean on a narrow
#: slot, where the plate `SEED_REACH_SLACK` longer projects so far that no
#: wing is left between the two (`plate_area_floor`)
SEED_TIGHT_SLACK = 1.02


def plate_reach(op: "OperatingPoint", choices: dict | None,
                slack: float = SEED_REACH_SLACK, ride_lo: float | None = None) -> tuple | None:
    """`(h, width)` of the designed plates at the least that reaches the car:
    the arc `h` [m] of a plate at the card's STATED lean and blend (a searched
    one at the upright crease, as `plate_seed` seeds it) whose tip gets down
    to the bottom of the ride row (`ride_lo`: the box's own when the player
    typed or fixed it, else the slot's band; `slack` over it: the seed's
    reach candidate's `SEED_REACH_SLACK` by default, 1 for the exact least),
    and the WIDTH [m] the two of them take out of the span row there --
    AeroBO's span row is the overall width, plates included, and a leaning
    or blended plate projects outboard (`geometry.winglet_projection`). None
    when the plates do not carry the wing, or the slot has no deck to reach."""
    ch = choices or {}
    if not bool(ch.get("plates", True)) or op.deck is None:
        return None
    from aerobo import geometry as _geo
    free = set(plate_free(ch))
    cant = CANT_UPRIGHT if "cant" in free else _cant(ch)
    frac = 0.0 if "blend" in free else _blend(ch)
    hf = float(_geo.winglet_tip_height(1.0, cant, frac, BLEND_SHAPE))
    pf = float(_geo.winglet_projection(1.0, cant, frac, BLEND_SHAPE))
    if not hf > 0.0:
        return None
    h = float(slack) * max(_ride_lo(op, ride_lo) - float(op.deck), 0.0) / hf
    return h, 2.0 * max(pf, 0.0) * h


def _ride_lo(op: "OperatingPoint", ride_lo: float | None) -> float:
    return float(op.ride_band[0]) if ride_lo is None else float(ride_lo)


def plate_area_floor(op: "OperatingPoint", choices: dict | None, b_hi: float,
                     ride_lo: float | None = None) -> float | None:
    """The area row's floor a PROJECTING plate needs (the owner, 2026-09-27:
    "Root blend 1 for wing design doesn't produce a solution"), or None.

    The slot's area row (`size_rows`) is sized for a wing as wide as the span
    row. A stated root blend or lean takes the plates' outboard reach out of
    that width (`plate_reach`), and AeroBO checks AR >= 3 on the wing LEFT:
    on a flank 1.08 m wide at blend 1, the plates that reach the car's side
    take 0.55 m, the 0.53 m wing left carries at most 0.09 m2, and the row's
    0.12 m2 floor left 0 designs in the box -- 0 of 50 feasible, no seed.
    The floor is half the AR-3 area of the wing left at the widest span
    `b_hi` (a fixed span: its value) by the plate that just reaches the
    bottom of the ride row `ride_lo`, so the box holds the designs that fly.
    None where nothing needs moving (an upright crease, the pylons) or no
    wing is left at all (`plate_fit_problem` says so)."""
    r = plate_reach(op, choices, slack=1.0, ride_lo=ride_lo)
    if r is None or not r[1] > 1e-9:                      # cos 90 deg is 6e-17, not 0
        return None
    b_wing = float(b_hi) - r[1]
    if not b_wing > 0.0:
        return None
    return round(0.5 * b_wing * b_wing / AR_MIN, 6)


#: the least wing left between two carrying plates that `plate_fit_problem`
#: lets a run try. Narrower, a sweep of the five cars' flanks (leans 55-90,
#: blends 0.6-1) found some seeded and some not (a 7.7 cm wing between full
#: plates on a bus flank flew); the model refuses only where `plate_seed`
#: finds nothing, so a toy wing that flies is still flown
PLATE_WING_MIN_M = 0.12


def plate_fit_problem(op: "OperatingPoint", choices: dict | None, b_hi: float,
                      h_hi: float | None = None, ride_lo: float | None = None,
                      s_lo: float | None = None) -> str | None:
    """Why no wing can fly with the card's plate on this slot -- or None.
    The plates must reach the car: at a full blend or a shallow lean they
    project so far outboard that nothing of the span row is left for a wing,
    or the tallest plate the box allows (`h_hi`) cannot get down to the
    bottom of the ride row (`ride_lo`), or the wing left is too narrow for
    the least area the box asks (`s_lo`, at AeroBO's AR 3 -- a typed,
    released or fixed area row). `b_hi` is the widest span the box allows (a
    fixed span: its value). Said BEFORE a run that could only return "no
    feasible design": on the exact least reach (no slack), so only a box
    with no design in it is refused."""
    r = plate_reach(op, choices, slack=1.0, ride_lo=ride_lo)
    if r is None:
        return None
    h, width = r
    ch = choices or {}
    free = set(plate_free(ch))
    said = []
    if "blend" not in free and _blend(ch) > 0.0:
        said.append(f"root blend {_blend(ch):.2f}")
    if "cant" not in free and _cant(ch) != CANT_UPRIGHT:
        said.append(f"lean {_cant(ch):.0f}°")
    what = " and ".join(said) or "this plate"
    reach = _ride_lo(op, ride_lo) - float(op.deck)
    if h_hi is not None and h > float(h_hi) + 1e-9:
        return (f"the endplates at {what} cannot reach the car: getting {reach:.2f} m to it takes "
                f"a {h:.2f} m plate and the box stops at {float(h_hi):.2f} m. Blend less, lean "
                f"nearer upright, or bring the ride row closer to the car")
    b_wing = float(b_hi) - width
    if b_wing < PLATE_WING_MIN_M:
        return (f"the endplates at {what} leave no wing: reaching the car ({reach:.2f} m away) "
                f"they take {width:.2f} m of the {float(b_hi):.2f} m span, "
                f"{100.0 * max(b_wing, 0.0):.0f} cm left between them. Blend less or "
                f"lean nearer upright, or give the slot more span")
    if s_lo is not None and float(s_lo) > b_wing * b_wing / AR_MIN * (1.0 + 1e-9):
        return (f"the endplates at {what} leave a {b_wing:.2f} m wing between them, which "
                f"carries at most {b_wing * b_wing / AR_MIN:.3f} m2 (AR {AR_MIN:g}), and the area "
                f"row starts at {float(s_lo):.3f} m2. Lower the area row in the Design box, "
                f"blend less or lean nearer upright")
    return None


def plate_seed(family: str, flags: dict, bounds_overrides: dict | None = None,
               pinned: dict | None = None):
    """A design that FLIES, for AeroBO's `x_seed`, when the card hands the
    designed plate's lean or root blend to the optimiser, or STATES one off
    the upright crease (`plate_seeded`) -- or None.

    WHY. Searching them opens a big box whose feasible part is small: the
    plate still has to REACH the car, and a shallow lean or a long turn cannot
    (below about 48 deg nothing in the box reaches the top of the ride band).
    Measured on the slot families, 256 Sobol points: 17 fly with both searched
    on the top wing and 3 on a flank (55 and 21 with both stated) -- so a
    recommended run's initial design can miss them all, and seed 0 did: 0 of
    59 feasible, "no feasible design to put on the car", on both slots, while
    seed 2 found 9 of 10. AeroBO's `x_seed` is the one exact remedy it has
    (`api._seed_rows`): the seed is evaluated first and never screened out, so
    a run told a design that flies returns one that flies, at least as good.

    WHICH. Three candidates, each with the searched rows at the plate they
    generalise -- upright, a crease -- and the pins at their pins: the box
    centre; the same at the bottom of the ride band with the tallest plate
    (the easiest reach); and the REACH candidate: the bottom of the ride
    band with the shortest plate that gets down to the deck there
    (`SEED_REACH_SLACK` over it, on the plate's own tip-height fraction at
    its lean and blend -- `geometry.winglet_tip_height`), the widest span
    row, and the area moved inside AeroBO's AR band for the wing LEFT once
    the plates' outboard projection is taken out of that width. A
    blended or leaning plate needs it: the tallest plate projects so far that
    the wing left fails the AR band at the centre area (a side slot, blend
    0.6: the plate's 0.22 m a side leaves a 0.88 m wing under a 0.43 m2 area,
    AR 1.8). Of those AeroBO's own evaluator passes (every margin >= 0) the
    best-scoring is the seed -- the floor the run's answer cannot fall below;
    None when none does, and the run is AeroBO's own unseeded search. Cached
    on everything it reads: three evaluations."""
    def find():
        built = build_family(family, dict(flags or {}), bounds_overrides)
        labels = [str(v) for v in built.param_labels]
        prob = built.problem
        stated = (float(getattr(prob, "blend_frac", 0.0) or 0.0) > 0.0
                  or float(getattr(prob, "endplate_cant_deg", CANT_UPRIGHT)) != CANT_UPRIGHT)
        if not (any(k in labels for k in _PLATE_STATED) or stated):
            return None
        box = np.asarray(built.bounds, dtype=float)
        pins = {str(k): float(v) for k, v in (pinned or {}).items()}
        centre = 0.5 * (box[:, 0] + box[:, 1])
        low = centre.copy()
        for lab, end in (("ride_height_m", 0), ("endplate_h_m", 1)):
            if lab in labels and lab not in pins:
                low[labels.index(lab)] = box[labels.index(lab), end]

        def fixed(x):
            x = x.copy()
            for lab, v in _PLATE_STATED.items():
                if lab in labels:
                    i = labels.index(lab)
                    x[i] = min(max(v, box[i, 0]), box[i, 1])
            for lab, v in pins.items():
                if lab in labels:
                    x[labels.index(lab)] = v
            return x

        def reach(slack=SEED_REACH_SLACK, stiff=False):
            """The reach candidate (see WHICH), or None where the problem has
            no plate / ride / span row to set. `stiff`: the plate's chord at
            its box's widest -- the stiffest plate, for a long plate over a
            short wing whose centre-chord plate bends past its margin."""
            from aerobo import geometry as _geo
            x = fixed(low)
            if stiff and "endplate_chord_ratio" in labels and "endplate_chord_ratio" not in pins:
                c = labels.index("endplate_chord_ratio")
                x[c] = box[c, 1]
            ix = {lab: labels.index(lab) for lab in ("endplate_h_m", "ride_height_m", "b_m",
                                                     "S_m2", "endplate_cant_deg",
                                                     "endplate_blend_frac") if lab in labels}
            if not all(k in ix for k in ("endplate_h_m", "ride_height_m", "b_m")) \
                    or not hasattr(prob, "reach_m"):
                return None
            cant = float(x[ix["endplate_cant_deg"]] if "endplate_cant_deg" in ix
                         else prob.endplate_cant_deg)
            frac = float(x[ix["endplate_blend_frac"]] if "endplate_blend_frac" in ix
                         else prob.blend_frac)
            shape = str(prob.blend_shape)
            hf = _geo.winglet_tip_height(1.0, cant, frac, shape)
            pf = _geo.winglet_projection(1.0, cant, frac, shape)
            if not hf > 0.0:
                return None
            i = ix["endplate_h_m"]
            if "endplate_h_m" not in pins:
                need = slack * prob.reach_m(float(x[ix["ride_height_m"]])) / hf
                x[i] = min(max(need, box[i, 0]), box[i, 1])
            j = ix["b_m"]
            if "b_m" not in pins:
                x[j] = box[j, 1]
            if "S_m2" in ix and "S_m2" not in pins:
                k = ix["S_m2"]
                b_wing = float(x[j]) - 2.0 * pf * float(x[i])
                lo = max(box[k, 0], b_wing * b_wing / AR_MAX)
                hi = min(box[k, 1], b_wing * b_wing / AR_MIN)
                if not (b_wing > 0.0 and lo <= hi):
                    return None
                x[k] = min(max(float(x[k]), lo), lo + 0.9 * (hi - lo))
            return x

        best = None
        #  the fallbacks are flown only when nothing before them flies, so a
        #  slot the first three seed keeps the seed it always had
        for make in (lambda: fixed(centre), lambda: fixed(low), reach,
                     lambda: None if best is not None else reach(SEED_TIGHT_SLACK),
                     lambda: None if best is not None else reach(SEED_REACH_SLACK, True),
                     lambda: None if best is not None else reach(SEED_TIGHT_SLACK, True)):
            x = make()
            if x is None:
                continue
            try:
                f, g = built.callable(x)
            except Exception:                               # noqa: BLE001
                continue
            g = np.atleast_1d(np.asarray(g, dtype=float))
            if np.isfinite(f) and float(f) > float(carwing.PENALTY) and np.all(g >= 0.0) \
                    and (best is None or float(f) > best[0]):
                best = (float(f), [float(v) for v in x])
        return None if best is None else best[1]
    return _cached(("plate_seed", family, flags, bounds_overrides, pinned), find)


def wing_cfg(family: str, flags: dict, eff: dict, seed: int = 0,
             bounds_overrides: dict | None = None, pinned: dict | None = None,
             x_seed=None):
    """The RunConfig the wing's Run button launches (config.build_cfg): absent
    keys stay absent, so an untouched card is AeroBO's own configuration."""
    return api.RunConfig(problem_name=family, mission_kwargs={}, flags=search_flags(flags, eff),
                         optimiser=str(eff["optimiser"]), budget=int(eff["budget"]),
                         seed=int(seed), bounds_overrides=bounds_overrides or None,
                         pinned=pinned or None,
                         x_seed=[float(v) for v in x_seed] if x_seed else None)


def handoff_split(cfg, dim: int, budget: int | None = None) -> tuple | None:
    """`(n_init, n_a)` a FRESH `bo_slsqp` run of `cfg` will fly -- its Sobol
    block and the evaluation SLSQP takes over after -- by AeroBO's own
    arithmetic (`handoff.split_budget` at the run's phi, then `_bo_split` on
    the BO phase's budget), so the live graph can mark both while the run is
    still going; the record reports them only at the end
    (`record["handoff"]`). `budget` overrides `cfg.budget` (the N the job
    actually reported). None for any other optimiser. (Added by the models
    agent, PLAN2 §8.5.)"""
    if getattr(cfg, "optimiser", None) != "bo_slsqp":
        return None
    from aerobo.optimize.handoff import split_budget
    n = int(cfg.budget if budget is None else budget)
    n_a, _n_b = split_budget(n, api._handoff_phi(cfg))
    n_init, _n_iter = api._bo_split(int(n_a), int(dim), api._bo_n_init(cfg))
    return int(n_init), int(n_a)


def stop_rule_factory(eff: dict, policy=None):
    """A FACTORY for the adaptive stop, or None (session.stop_rule_factory).

    A factory because a stop rule holds its run's history. It fires only on
    the PLATEAU: `ConvergenceStop` also answers True at n >= max_evals, and
    passing that through made every complete run come back partial."""
    if not (_recommended(policy) and bool(_pol(policy, "stop_when_converged", True))):
        return None
    plan = eff.get("plan")
    if plan is None or plan.patience is None or plan.tol is None:
        return None
    patience, tol, budget = int(plan.patience), float(plan.tol), int(eff["budget"])

    def factory():
        rule = _budget.ConvergenceStop(patience=patience, tol=tol, max_evals=budget)

        def _stop(i, best):
            fired = rule.update(best)
            return bool(fired and rule.n < rule.max_evals)
        return _stop
    return factory


def convergence_rule_reach(eff: dict) -> dict | None:
    """Can the adaptive stop fire at THIS budget? (session.convergence_rule_reach)"""
    plan = eff.get("plan")
    if plan is None or plan.patience is None or plan.tol is None:
        return None
    patience, budget = int(plan.patience), int(eff["budget"])
    return {"can_fire": budget > patience + 1, "patience": patience, "needs": patience + 2,
            "budget": budget}


def stop_or_converged(converged, cancelled):
    """THE STOP BUTTON AND THE CONVERGENCE RULE ARE ONE STOP RULE
    (stages/airfoil.stop_or_converged): the run ends between evaluations and
    AeroBO rebuilds its result from its own log -- partial, with the incumbent
    as best_x -- instead of unwinding and losing every evaluation paid for."""
    def _rule(i, best) -> bool:
        if cancelled():
            return True
        return bool(converged(i, best)) if converged is not None else False
    return _rule


def continue_extra_default(record: dict) -> int:
    """How many evaluations "Keep going" offers (session.continue_extra_default):
    the rest of the budget for a run stopped by hand, half of it again for one
    that spent it all."""
    cfgd = (record or {}).get("config") or {}
    budget = int(cfgd.get("budget") or 0)
    spent = int((record or {}).get("n_evals") or 0)
    if (record or {}).get("partial") and budget > spent:
        return int(budget - spent)
    return max(1, budget // 2)


def plate_tc_pin(chosen_plate: dict | None, box: dict | None = None) -> dict:
    """The plate's thickness row fixed to the section chosen in 2.8 (PLAN2 Q4:
    otherwise the plate flies one section's polar at another thickness). {} for
    the family's own plate. Clipped into the row's band when one is given, so
    the pin is always a value AeroBO accepts."""
    if not chosen_plate or chosen_plate.get("source") in (None, "default"):
        return {}
    tc = chosen_plate.get("tc")
    if tc is None:
        return {}
    tc = float(tc)
    row = (box or {}).get("endplate_tc")
    if row is not None:
        tc = min(max(tc, float(row[0])), float(row[1]))
    return {"endplate_tc": tc}


# =========================================================================== #
#  THE RUNNERS (PLAN2 5.5)                                                     #
# =========================================================================== #
#  A runner is `runner(emit, stop) -> result`, run on the engine's worker
#  thread by `design_jobs.EngineJob` (PLAN2 6.1). It never touches pygame, a
#  model, the library or Notices: it reports through `emit(kind, **payload)`
#  only ("eval", "sweep", "phase", "total", "log") and stops cooperatively --
#  every engine call gets AeroBO's own `stop_rule` / `cancel` hook reading
#  `stop.is_set()`, so a Stop ends the search BETWEEN evaluations and AeroBO
#  rebuilds a partial result from its own log (the best found is kept).
#
#  The factories freeze every argument at launch (deep copies): the job flies
#  a snapshot, and a form edited while it runs cannot reach it.
def _num(v):
    """A JSON-safe float: None for a missing or non-finite value (the fixtures
    are JSON, and a refused evaluation's NaN is not)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _vec(v):
    """A JSON-safe list of floats, or None."""
    if v is None:
        return None
    return [_num(u) for u in np.asarray(v, dtype=float).ravel()]


def _eval(emit, offset: int = 0, top: list | None = None):
    """AeroBO's rich progress callback as "eval" events.

    `api.run` hands a callback that declares **kw the per-evaluation payload
    (f, g, feasible, x). `n` counts the inherited evaluations -- a resumed run
    reports k + 1 first -- and `offset` shifts a restart's count past the
    restarts before it. `top` (a one-element list) keeps the best-so-far
    monotone across restarts, as V3's section stage does: k independent
    searches are ONE answer."""
    top = top if top is not None else [None]

    def cb(i, best, **kw):
        b = _num(best)
        if b is not None and (top[0] is None or b > top[0]):
            top[0] = b
        emit("eval", n=int(offset) + int(i), best=top[0], f=_num(kw.get("f")),
             feasible=bool(kw.get("feasible")), g=_vec(kw.get("g")), x=_vec(kw.get("x")))
    return cb


def _sweep(emit, phase: str):
    """A screen's per-section callback as "sweep" events. The first one IS the
    phase change: `screen_at_point` hands its callback to the shortlist pass
    only (the library pass is a read of the warm checkpoint). `cb.swept`
    counts them: none at all means the point's own checkpoint answered."""
    def cb(i, n, rec):
        if not cb.swept:
            emit("phase", text=phase)
        cb.swept += 1
        rec = rec or {}
        emit("sweep", i=int(i), n=int(n), name=str(rec.get("name") or "?"),
             status=str(rec.get("status") or ""), eligible=bool(rec.get("eligible")),
             ldcr=_num(rec.get("ldcr")), tc=_num(rec.get("tc")))
    cb.swept = 0
    return cb


def _gate(gates: dict | None, key: str, default: dict) -> float:
    """A hard gate's value; a gate switched off (None) is AeroBO's GATE_OFF."""
    g = {**default, **(gates or {})}
    return float(GATE_OFF[key] if g.get(key) is None else g[key])


def screen_runner(surface: str, weights: dict, cond: dict, gates: dict | None = None,
                  floors: dict | None = None, re_source: str = "mission",
                  top_n: int = TOP_N, shortlist: int = SHORTLIST):
    """Screen AeroBO's library for one surface (V3's `_screen_worker`).

    At the surface's own Reynolds number (`re_source` "mission", XFOIL
    present): `api.screen_at_point` -- the whole library at the cached point
    to choose who is worth a sweep, then a live XFOIL sweep of the shortlist
    (~20 s cold, instant after). Otherwise `api.screen_airfoils` at the
    cached library point: a read of the warm checkpoint, exact at any design
    cl through the branch sidecar. The plate screens the symmetric sections
    only (`api.symmetric_section_names`: a vertical panel may not be given a
    cambered one, and filtering afterwards would spend the shortlist on
    sections it cannot use). Every ranked row is refitted as CST so any row
    of the table can be chosen."""
    weights = {str(k): float(v) for k, v in (weights or {}).items()}
    tc_min = _gate(gates, "tc_min", SCREEN_GATES)
    cm_max = _gate(gates, "cm_max", SCREEN_GATES)
    fl = {str(k): float(v) for k, v in (floors or {}).items()
          if k in api.SCREEN_FLOOR_KEYS and v is not None}
    cond = copy.deepcopy(dict(cond))
    surface, re_source = str(surface), str(re_source)

    def run(emit, stop):
        emit("phase", text="library pass")
        names = api.symmetric_section_names() if surface == "plate" else None
        lib = api.screen_library_point() or {}
        lib_re = float(lib.get("re", 1e6))
        common = dict(cl_design=float(cond["cl_design"]), tc_min=tc_min, cm_max=cm_max,
                      floors=fl, top_n=int(top_n), names=names, cancel=stop.is_set)
        own = (re_source == "mission" and xfoil_ok()
               and abs(float(cond["re"]) - lib_re) > 1e-9 * lib_re)
        if own:
            sweep = _sweep(emit, "shortlist sweep")
            rep = api.screen_at_point(weights, re=float(cond["re"]),
                                      mach=float(cond.get("mach") or 0.0),
                                      shortlist=int(shortlist), progress_cb=sweep, **common)
            if not sweep.swept:
                # a second visit to the same point: every shortlisted polar
                # is already in the point's checkpoint, so nothing is swept
                emit("phase", text="shortlist answered from this point's XFOIL cache")
        else:
            rep = api.screen_airfoils(weights, re=lib_re, mach=float(lib.get("mach", 0.0)),
                                      progress_cb=_sweep(emit, "sweeping uncached sections"),
                                      **common)
        emit("phase", text="refitting the ranked sections")
        cands = api.screen_seed_candidates(rep, n=len(rep.get("ranked") or []))
        return {"report": rep, "candidates": cands, "stopped": stop.is_set(),
                "surface": surface, "re_source": "mission" if own else "library"}
    return run


def section_runner(shape: dict, search: dict, seed: int = 0, resume: dict | None = None,
                   results_dir=None):
    """Shape-optimise one section (V3's section worker): AeroBO's CST +
    live-XFOIL search, `api.optimize_airfoil`, with the stage's arguments.

    Sections have no adaptive stop (the plan's patience is None), so a
    partial section run is always the player's Stop. AeroBO's restart loop is
    kept verbatim (k = 1 for this class: the loop runs once and the run is
    the single search, byte for byte); a continuation that RESUMES is one
    search by definition. The result carries the LAUNCH SNAPSHOT "Keep
    going" re-flies (`continue_section`), never the form."""
    shape = copy.deepcopy(dict(shape))
    search = copy.deepcopy(dict(search))
    resume = copy.deepcopy(resume)
    seed = int(seed)
    n_restarts = 1 if resume is not None else max(1, int(search.get("n_restarts") or 1))
    launch = {"shape": shape, "seed": seed,
              "search": {"optimiser": str(search["optimiser"]), "budget": int(search["budget"]),
                         "refusal": search.get("refusal"), "n_init": search.get("n_init")},
              "n_restarts": n_restarts}

    def run(emit, stop):
        _torch_threads()
        # N is this run's own budget (a continuation's is spent + extra): the
        # chip and a replay of this run both read it from here
        emit("total", n=int(search["budget"]) * n_restarts)
        emit("phase", text="section search (XFOIL)")
        best_rep, best_y, top, last = None, None, [None], [0]

        def seen(kind, **p):
            if kind == "eval":
                last[0] = int(p["n"])
            emit(kind, **p)
        for k in range(n_restarts):
            if stop.is_set():
                break
            rep = api.optimize_airfoil(
                **shape, optimiser=str(search["optimiser"]), budget=int(search["budget"]),
                refusal=search.get("refusal"), n_init=search.get("n_init"),
                feasibility=FEASIBILITY_DEFAULT, resume=resume, seed=seed + k,
                # a restart's count starts where the one before it ended
                progress_cb=_eval(seen, offset=last[0], top=top),
                results_dir=results_dir,
                stop_rule=stop_or_converged(None, stop.is_set), stop_reason=STOP_PLAYER)
            y = ((rep.get("result") or {}).get("best_score") if isinstance(rep, dict) else None)
            if best_rep is None or (y is not None and (best_y is None or y > best_y)):
                best_rep, best_y = rep, y
        if stop.is_set() and isinstance(best_rep, dict):
            (best_rep.get("result") or {})["stop_reason"] = STOP_PLAYER
        return {"report": best_rep, "launch": copy.deepcopy(launch)}
    return run


def score_runner(report: dict, weights: dict, reference: dict | None = None):
    """The post-run job: the seed and the optimised section on every criterion
    under this surface's weights (`api.score_optimised_section` -- a wide
    stall sweep per shape, seconds, so never a frame)."""
    report = copy.deepcopy(report)
    weights = {str(k): float(v) for k, v in (weights or {}).items()}
    reference = copy.deepcopy(reference)

    def run(emit, stop):
        emit("phase", text="scoring seed vs optimised")
        return api.score_optimised_section(report, weights, reference=reference)
    return run


#: what a post-run score carries (a `score_sections` report): the seed's row,
#: the optimised section's, and optimised - seed per criterion
SCORE_KEYS = frozenset({"seed", "optimised", "delta"})


def polar_runner(value):
    """A chosen section's polar when the cache does not hold it
    (`api.section_polar_for`: a live XFOIL sweep at the section's point)."""
    value = copy.deepcopy(value)

    def run(emit, stop):
        emit("phase", text="section polar (XFOIL)")
        return api.section_polar_for(value)
    return run


def _family_json(p: FamilyParams) -> dict:
    """`asdict(p)` as JSON will hold it (tuples as lists)."""
    d = asdict(p)
    d["ride_band"] = [float(v) for v in p.ride_band]
    d["lap"] = None if p.lap is None else list(p.lap)
    d["free"] = list(p.free)
    return d


def wing_runner(cfg, family_params, conv_factory=None, resume: dict | None = None,
                results_dir=None):
    """Run the wing (V3's wing worker): `api.run` on the slot family with
    V3's configuration, AeroBO's evaluation cache on.

    The family is registered again on the worker (idempotent: a record may
    outlive the process that registered it), the objective validated with
    AeroBO's own sentence before anything is flown, and the stop rule is the
    Stop button OR the plateau rule. Which of the two fired is only known
    afterwards: a partial run with the Stop pressed is the player's (AeroBO's
    airfoil stage overwrites `stop_reason` the same way)."""
    cfg = copy.deepcopy(cfg)
    fp = (family_params if isinstance(family_params, FamilyParams)
          else FamilyParams.from_json(family_params))
    resume = copy.deepcopy(resume)

    def run(emit, stop):
        ensure_family(fp)
        api.check_wing_objective(cfg.flags)
        _torch_threads()
        emit("total", n=int(cfg.budget))
        emit("phase", text=("lap (lattice + lap)" if fp.lap is not None else "wing (lattice)"))
        conv = conv_factory() if conv_factory is not None else None
        res = api.run(cfg, progress_cb=_eval(emit), results_dir=results_dir,
                      stop_rule=stop_or_converged(conv, stop.is_set),
                      stop_reason=STOP_CONVERGED, eval_cache=True, resume=resume)
        d = res.to_dict()
        d["carsim_family"] = _family_json(fp)
        if d.get("partial") and stop.is_set():
            d["stop_reason"] = STOP_PLAYER
        return {"record": d}
    return run


_CFG_KEYS = ("problem_name", "mission_kwargs", "flags", "optimiser", "budget", "seed",
             "bounds_overrides", "pinned", "x_seed")


def record_cfg(record: dict):
    """The RunConfig a stored record was flown with, on its (re-registered)
    slot family."""
    fam = family_of(record)
    cfgd = dict((record or {}).get("config") or {})
    kw = {k: cfgd.get(k) for k in _CFG_KEYS if k in cfgd}
    kw["problem_name"] = fam or kw.get("problem_name")
    kw["mission_kwargs"] = kw.get("mission_kwargs") or {}
    kw["flags"] = kw.get("flags") or {}
    return api.RunConfig(**kw)


def design_report(record: dict) -> dict:
    """AeroBO's design report for a record's winner: the geometry and the
    spanwise breakdown the Results views draw."""
    return api.design_report(record_cfg(record), record["best_x"])


def design_report_runner(record: dict):
    """`design_report` as a job (a lattice solve of the winner and its
    report: well under a second, but never a frame)."""
    record = copy.deepcopy(record)

    def run(emit, stop):
        emit("phase", text="design report")
        return design_report(record)
    return run


def continue_wing(record: dict, extra: int):
    """"Keep going" on a wing record: `api.continue_run_config(record, extra)`
    -> (cfg, note). A BO run resumes (`note["resume"]`, `note["resumed"]`):
    nothing is re-flown, the counter continues at k + 1."""
    family_of(record)
    return api.continue_run_config(record, int(extra))


def continue_section(report: dict, launch: dict, extra: int | None = None) -> dict:
    """Arm "give this finished SECTION search more evaluations" (= V3's
    `session.continue_section`, resume branch first). Never runs.

    Re-flies the LAUNCH SNAPSHOT at the new budget, never the form. A BO
    search carries its evaluations, so the continuation inherits them
    (`api.resume_payload`) and buys only the new ones; a run stopped by hand
    is finished at the budget it asked for (`continue_extra_default`).
    Returns {"shape", "seed", "search" (+ budget, resume, a pinned n_init),
    "note", "extra"}; raises ValueError with AeroBO's sentence when the run
    cannot be continued."""
    res = (report or {}).get("result") or {}
    cfgd = (report or {}).get("config") or res.get("config") or {}
    if not cfgd.get("problem_name"):
        raise ValueError("there is no finished section search to continue")
    if not (launch or {}).get("shape"):
        raise ValueError("this run was not launched here, so the search it flew cannot be "
                         "repeated exactly -- optimise again to start a run that can be continued")
    if int(launch.get("n_restarts", 1) or 1) > 1:
        raise ValueError("this run was several independent searches, and continuing one of "
                         "them is not the same search")
    old = int(cfgd.get("budget") or 0)
    spent = int(res.get("n_evals") or 0)
    if extra is None:
        extra = continue_extra_default({"config": cfgd, "n_evals": spent,
                                        "partial": bool(res.get("partial"))})
    extra = int(extra)
    if extra < 1:
        raise ValueError("a continuation must add at least one evaluation")
    total = spent + extra
    rec = {"config": cfgd, "eval_x": res.get("eval_x"), "eval_y": res.get("eval_y"),
           "eval_g": res.get("eval_g"), "is_constrained": res.get("is_constrained")}
    resume = api.resume_payload(rec) if api.can_resume(rec) else None
    pinned_init = None
    if total <= old and resume is None:
        note = {"exact": True, "budget": old, "added": old - spent, "was": spent,
                "why": (f"the same search, finished: the {spent} evaluations it flew come back "
                        f"out of the XFOIL cache and the {old - spent} it never reached follow")}
    elif resume is not None:
        note = {"exact": True, "budget": total, "added": extra, "was": spent,
                "resume": resume, "resumed": spent,
                "why": (f"a resume, not a re-run: the {spent} evaluations already paid for are "
                        f"this search's training set, so it starts at {spent} and buys {extra} "
                        f"NEW sections")}
    else:
        cont_cfg, note = api.continue_run_config(
            {"config": cfgd, "dim": res.get("dim"), "searched_dim": res.get("searched_dim"),
             "bo_split": res.get("bo_split")}, total - old)
        note = {**note, "was": spent, "added": total - spent}
        pinned_init = (cont_cfg.flags or {}).get(api.BO_N_INIT_FLAG)
    search = {**dict(launch.get("search") or {}), "budget": int(note["budget"])}
    if pinned_init is not None:
        search["n_init"] = int(pinned_init)
    search["resume"] = note.get("resume")
    return {"shape": copy.deepcopy(dict(launch["shape"])), "seed": int(launch.get("seed") or 0),
            "search": search, "note": note, "extra": extra}


# =========================================================================== #
#  THE LAP (PLAN2 5.6, D4): carsim's circuits and the Corsa, as cartrack reads  #
# =========================================================================== #
#  AeroBO times a lap only on the families WITHOUT designed endplates
#  (`CarWingEndplateProblem` has no track_spec, and its objective menu drops
#  "laptime"), and cartrack's point-mass lap reads a wing's CZ*S / CD*S as
#  downforce -- it cannot value a lateral device. So, exactly as AeroBO's own
#  menu, lap time is offered on the TOP slot with plain fences only.
#
#  The model is AeroBO's point-mass lap on carsim's circuit geometry with the
#  Corsa's mass, power and CdA. carsim's own two-track lap (drive.aero.mission)
#  is a different model; the Results page shows it beside AeroBO's number as a
#  cross-check, labelled, and the two will not match to the tenth.
from .aero import mission as _ms                        # noqa: E402

#: the car cartrack is told about (the same Corsa `aero.mission` laps)
CAR = _ms.CAR
#: what the lap objective says about its own model (shown on Wing type)
LAP_NOTE = ("WingLab's point-mass lap (cartrack) on carsim's {circuit} geometry with the "
            "Corsa's mass, power and CdA; carsim's own two-track lap on Results is a "
            "different model and will not match to the tenth")


def lap_offered(circuit: str) -> str:
    """"" when AeroBO can time a lap of `circuit`, else why not (cartrack's own
    rules: a closed circuit with at least one corner)."""
    from .track import make_track
    try:
        track_spec(_ms.TrackProfile.from_track(make_track(str(circuit))))
    except (ValueError, KeyError) as exc:
        return str(exc)
    return ""


def track_spec(profile) -> "cartrack.TrackSpec":
    """carsim's circuit (`aero.mission.TrackProfile`, built from `drive.track`'s
    own segments, straights already merged) as AeroBO's `cartrack.TrackSpec`:
    one to one -- Corner(radius, arc) / Straight(length), in order. Refuses
    what cartrack refuses (no corner) and an open strip (a lap time is only
    well posed on a closed circuit)."""
    if not getattr(profile, "closed", True):
        raise ValueError(f"{profile.name} is not a closed circuit, so it has no lap to time")
    segs = []
    for it in profile.items:
        if isinstance(it, _ms.Corner):
            segs.append(cartrack.Corner(radius_m=float(it.radius), arc_m=float(it.length)))
        else:
            segs.append(cartrack.Straight(length_m=float(it.length)))
    return cartrack.TrackSpec(segments=tuple(segs), name=f"carsim {profile.name}")


def tyre_law(mass_kg: float, mu_scale: float = 1.0) -> tuple:
    """(mu0, k_load): carsim's tyre as cartrack's power law, matched at the
    static load.

    carsim's tyre is affine per wheel, `mu = mu_ref + s (Fz - Fz_ref)` (qss,
    TNO 205/60R15, measured); cartrack's is `mu0 (N / N_ref)^-k` on the car's
    whole load, with N_ref the static weight. At the static load the two
    share the value and the slope when `mu0 = mu(W/4)` and `k = -PD2 / PD1`
    with PD1 = mu(W/4), PD2 = s W/4 -- AeroBO's own conversion
    (`cartrack._k_load_from_affine_slope`), applied to carsim's tyre."""
    import qss
    fz = float(mass_kg) * _ms.G / 4.0
    pd1 = qss.TYRE["mu_ref"] + qss.TYRE["s"] * (fz - qss.TYRE["Fz_ref"])
    pd2 = qss.TYRE["s"] * fz
    return float(mu_scale) * pd1, float(cartrack._k_load_from_affine_slope(pd1, pd2))


def _other_wing_mass(build, lib, key: str) -> float:
    """kg of the wings fitted in the OTHER slots (`CarBuild.mass_points`): the
    lap carries them; the wing being designed is the one in `key`."""
    if build is None or lib is None or not hasattr(build, "mass_points"):
        return 0.0
    return float(sum(float(p.m) for p in build.mass_points(lib)
                     if not str(getattr(p, "name", "")).startswith(f"{key} ")))


def car_spec(car=None, build=None, lib=None, slot_x: float = -0.90, mu_scale: float = 1.0,
             mass_kg: float | None = None) -> "cartrack.CarSpec":
    """The Corsa as cartrack's `CarSpec`.

    mass = the car + the OTHER fitted wings; tyre = carsim's, matched at the
    static load (`tyre_law`, x the surface's mu_scale); power = P_max through
    eta_drive; drag area = the Corsa's CdA 0.66 (everything but the wing);
    downforce area 0 (carsim models no body downforce, so the balance
    cartrack reports is the wing's alone -- reported, never constrained: no
    window is stated); the frame = the Corsa's wheelbase and CG, the wing at
    its slot station (`x` positive forward of the CG); carsim's air."""
    car = CAR if car is None else car
    m = float(mass_kg) if mass_kg is not None else float(car.m) + _other_wing_mass(build, lib, "top")
    mu0, k = tyre_law(m, mu_scale)
    a = float(car.a)                                     # CG to the FRONT axle
    return cartrack.CarSpec(mass_kg=m, mu0=mu0, k_load=k, power_w=float(car.P_max),
                            drivetrain_eta=float(car.eta_drive), cda_car_m2=float(car.CdA),
                            cza_car_m2=0.0, wheelbase_m=float(car.L), x_cg_frac=a / float(car.L),
                            x_wing_frac=(a - float(slot_x)) / float(car.L), x_cp_car_frac=0.5,
                            rho=CARSIM_RHO)


def lap_params(mission, build=None, lib=None, key: str = "top") -> tuple:
    """The lap a slot family times, as the hashable tuple `FamilyParams.lap`
    carries: (circuit, surface, mu_scale, x_t, mass_kg). Rounded, so the same
    car on the same circuit is the same family (and the same cache)."""
    circuit = str(getattr(mission, "track", "arena"))
    mu_scale = float(getattr(mission, "mu_scale", 1.0))
    surface = str(getattr(mission, "surface", "")) or "dry"
    x_t = float(build.slot(key).x) if build is not None else -0.90
    m = float(CAR.m) + _other_wing_mass(build, lib, key)
    return (circuit, surface, round(mu_scale, 9), round(x_t, 6), round(m, 3))


_LAPS: dict = {}


def _lap_fields(lap: tuple) -> dict:
    """track_spec / car_spec for a lap tuple (cached: a family is rebuilt on
    every evaluation's `spec.build`)."""
    lap = tuple(lap)
    if lap not in _LAPS:
        from .track import make_track
        circuit, _surface, mu_scale, x_t, mass = lap
        prof = _ms.TrackProfile.from_track(make_track(str(circuit)))
        _LAPS[lap] = {"track_spec": track_spec(prof),
                      "car_spec": car_spec(slot_x=float(x_t), mu_scale=float(mu_scale),
                                           mass_kg=float(mass))}
    return dict(_LAPS[lap])


# =========================================================================== #
#  FIT TO THE CAR (PLAN2 5.7, D9)                                              #
# =========================================================================== #
#  The game's laws are DERIVED BY SAMPLING AeroBO's own evaluator at the
#  winning design: an incidence sweep on a sampling twin of the run's problem.
#  Measured: the lattice's CZ is affine in incidence over the feasible range,
#  so the affine law is exact there; it passes through AeroBO's design point
#  (CZ and CD) by construction; the stall clamps are AeroBO's own refusal
#  edges; drag is a quadratic fit, pinned to the design point. The car then
#  flies AeroBO's forces: `TopAero.from_aero(law, a*)` gives CZ* and CD*, so
#  the game's 1/2 rho V^2 S CZ* IS AeroBO's downforce_N (same rho, D3).
from .aero.wing import WingSpec as _WingSpec, wing_mass as _wing_mass  # noqa: E402
from .aero.airfoil import AirfoilSpec as _AirfoilSpec                 # noqa: E402

#: the sampling twin's incidence band: wide enough that the feasible run is
#: ended by AeroBO's own refusals (polar validity, the plate's linear range),
#: not by the optimiser's box. Sampling only -- never optimised over.
SWEEP_ALPHA_DEG = (-20.0, 30.0)
SWEEP_STEP_DEG = 0.5
#: half the central difference the lift slope is read with
SLOPE_STEP_DEG = 0.5
#: rows of the law's table (the plots): the feasible run, thinned to this
TABLE_ROWS = 15

_TWINS: dict = {}


def _sampling_twin(problem):
    """The run's problem with ONLY the incidence band widened (cached per class)."""
    base = type(problem)
    if base not in _TWINS:
        _TWINS[base] = type(f"{base.__name__}_sampling", (base,),
                            {"ALPHA_BOUNDS_DEG": SWEEP_ALPHA_DEG})
    return _TWINS[base](**_init_fields(problem))


def _record_problem(record: dict):
    """(cfg, built, labels, x*) of a record's winner, on its re-registered
    family with the flags and box it was flown with (the evaluator refuses a
    vector outside its problem's size rows, so the box matters). A PINNED
    row is read at its pin (`record_values`): the tip device "none" flew a
    plate of height 0, whatever else the vector holds there."""
    cfg = record_cfg(record)
    built = api.PROBLEM_SPECS[cfg.problem_name].build(dict(cfg.mission_kwargs or {}),
                                                      dict(cfg.flags or {}), cfg.bounds_overrides)
    labels = [str(v) for v in built.param_labels]
    xs = np.asarray(record["best_x"], dtype=float).copy()
    for lab, pin in (cfg.pinned or {}).items():
        if str(lab) in labels:
            xs[labels.index(str(lab))] = float(pin)
    return cfg, built, labels, xs


def _ix(labels: list, name: str) -> int | None:
    return labels.index(name) if name in labels else None


def derive_law(record: dict, slot_geom: dict | None = None) -> dict:
    """The aero dict `WingSpec.aero` carries for AeroBO's winner (PLAN2 5.7).

    Every key `aero.wing.analyse` returns that anything reads, so
    `design_point`, `DevAero.from_aero`, `TopAero.from_aero`, the challenges
    and the HUD work unchanged; plus `engine`, `derived_at` (the slot it was
    derived at) and `aerobo` (AeroBO's own force, drag and coefficients at the
    design). Raises ValueError when the winner is not flyable."""
    geom = dict(slot_geom or {})
    cfg, built, labels, xs = _record_problem(record)
    q = _sampling_twin(built.problem)
    evaluate = _evaluators(q)[1]
    ia = labels.index("alpha_deg")
    a_star = float(xs[ia])
    ev = evaluate(xs)
    if not ev.get("feasible"):
        raise ValueError(f"the run found no flyable design: {ev.get('reason') or 'refused'}")

    def at(a):
        x = xs.copy()
        x[ia] = float(a)
        return evaluate(x)

    # the lift slope: a central difference where both sides fly, one-sided at
    # a refusal edge
    up, dn = at(a_star + SLOPE_STEP_DEG), at(a_star - SLOPE_STEP_DEG)
    if up["feasible"] and dn["feasible"]:
        cla = (up["CZ"] - dn["CZ"]) / math.radians(2.0 * SLOPE_STEP_DEG)
    elif up["feasible"] or dn["feasible"]:
        side, sgn = (up, 1.0) if up["feasible"] else (dn, -1.0)
        cla = sgn * (side["CZ"] - ev["CZ"]) / math.radians(SLOPE_STEP_DEG)
    else:
        raise ValueError("the winner flies at one incidence only: WingLab refuses it half a "
                         "degree either side, so no lift slope can be read")
    cl0 = float(ev["CZ"]) - cla * math.radians(a_star)

    # the sweep: the feasible run CONTIGUOUS with the design is the law's
    # range; its ends are AeroBO's own refusals (the stall proxy)
    grid = sorted(set(np.round(np.arange(SWEEP_ALPHA_DEG[0], SWEEP_ALPHA_DEG[1] + 1e-9,
                                         SWEEP_STEP_DEG), 9).tolist()) | {a_star})
    rows = {a: (ev if a == a_star else at(a)) for a in grid}
    k0 = grid.index(a_star)
    lo = hi = k0
    while lo - 1 >= 0 and rows[grid[lo - 1]]["feasible"]:
        lo -= 1
    while hi + 1 < len(grid) and rows[grid[hi + 1]]["feasible"]:
        hi += 1
    run = grid[lo:hi + 1]
    A = np.radians(run)
    CZ = np.array([rows[a]["CZ"] for a in run], dtype=float)
    CD = np.array([rows[a]["CD"] for a in run], dtype=float)
    residual = float(np.max(np.abs(cl0 + cla * A - CZ)))
    ends = (float(rows[run[0]]["CZ"]), float(rows[run[-1]]["CZ"]))

    # drag: least squares CD(CZ), then pinned through AeroBO's design point
    if len(run) >= 3:
        c2, c1, c0 = (float(v) for v in np.polyfit(CZ, CD, 2))
    else:
        c2, c1, c0 = 0.0, 0.0, float(ev["CD"])
    c0 += float(ev["CD"]) - (c0 + c1 * float(ev["CZ"]) + c2 * float(ev["CZ"]) ** 2)
    fit = c0 + c1 * CZ + c2 * CZ * CZ
    fit_err = float(np.max(np.abs(fit - CD) / np.maximum(np.abs(CD), 1e-12)))

    # the WING's span. Two numbers share the name b_m: the design ROW is the
    # overall width (the plates' outboard reach included, what the band
    # bounds), the breakdown's is the wing left once the plates have taken
    # that reach out of it (endplate.py / carwing.py: b_m = overall_width_m -
    # 2 x endplate_projection_m) -- the span AR and the beam are. Taking the
    # projection off the breakdown's again drew every leaning or blended wing
    # short by twice its plate's reach (row B32: span^2 / S == AR)
    if ev.get("b_m") is not None:
        b_wing = float(ev["b_m"])
    else:
        b_wing = (float(xs[labels.index("b_m")])
                  - 2.0 * float(ev.get("endplate_projection_m") or 0.0))
    S = float(ev["S_m2"])
    ir, ih = _ix(labels, "ride_height_m"), _ix(labels, "endplate_h_m")
    role = str(geom.get("role") or (FamilyParams.from_json(record["carsim_family"]).role
                                    if record.get("carsim_family") else "top"))
    V = float((cfg.flags or {}).get("V") or q.V)
    pick = np.unique(np.linspace(0, len(run) - 1, min(TABLE_ROWS, len(run))).round().astype(int))
    law = dict(
        V_ref=V, rho=float(q.rho), Re=float(ev.get("Re_mac") or 0.0), polar_source="aerobo",
        ride_h=(float(xs[ir]) if (role == "top" and ir is not None) else None),
        S=S, AR=float(ev["AR"]), MAC=S / max(b_wing, 1e-9),
        plate_h_flown=float(xs[ih]) if ih is not None else 0.0, span_flown=b_wing,
        CL0=float(cl0), CLa=float(cla), CL_min=min(ends), CL_max=max(ends),
        alpha_stall_deg=float(run[-1]), alpha_stall_neg_deg=float(run[0]),
        e=float(ev.get("e") or 0.0), cd0=c0, cd1=c1, cd2=c2, cd_fit_err=fit_err,
        cd_junction=float(ev.get("CD_junction") or 0.0), cd_strut=float(ev.get("cd0_struts") or 0.0),
        CDp_min=float(min(float(rows[a].get("CDp") or 0.0) for a in run)),
        mount=record_mount(cfg.flags),
        table=dict(alpha=[round(float(run[i]), 3) for i in pick],
                   CL=[round(float(CZ[i]), 5) for i in pick],
                   CD=[round(float(CD[i]), 6) for i in pick],
                   CDi=[round(float(rows[run[i]].get("CDi") or 0.0), 6) for i in pick],
                   CDp=[round(float(rows[run[i]].get("CDp") or 0.0), 6) for i in pick]),
        engine="aerobo", alpha_design_deg=a_star, law_residual=residual,
        derived_at={"x": _num(geom.get("x")), "h": _num(geom.get("h")), "V": V},
        aerobo={"F_N": float(ev["downforce_N"]), "D_N": float(ev["drag_N"]),
                "CZ": float(ev["CZ"]), "CD": float(ev["CD"]), "q_Pa": _num(ev.get("q_Pa")),
                "problem": cfg.problem_name})
    law["mass"] = float(_wing_mass(_geometry_spec(record, law, role, tcs=_family_tc(built, labels, xs))))
    return law


def law_runner(record: dict, slot_geom: dict | None = None):
    """`derive_law` as a job (~100 evaluations of AeroBO's lattice, no XFOIL)."""
    record, slot_geom = copy.deepcopy(record), copy.deepcopy(slot_geom)

    def run(emit, stop):
        emit("phase", text="the car's law from WingLab's evaluator")
        return derive_law(record, slot_geom)
    return run


def _family_tc(built, labels: list, xs) -> tuple:
    """(main t/c, plate t/c) of the family's OWN sections: the NACA 24xx
    family at the problem's t/c, and the plate's NACA 00tt at the searched
    `endplate_tc` (AeroBO: "the endplate's own default")."""
    ip = _ix(labels, "endplate_tc")
    return (float(getattr(built.problem, "tc", 0.12)),
            float(xs[ip]) if ip is not None else None)


def _naca(prefix: str, tc: float) -> str:
    return f"naca{prefix}{int(round(100.0 * float(tc))):02d}"


def record_mount(flags: dict | None) -> str:
    """What carried the wing a record flew: "pylon" when its flags state
    AeroBO's continuum pylon mount, else "endplate" -- the plates carried it
    (a designed plate, or a LEGACY fence record, which flew plain plates at
    the tips: it re-fits and draws as it flew)."""
    f = flags or {}
    return ("pylon" if str(f.get("car_mount_model") or "") == "continuum"
            and str(f.get("mount_kind") or "") == "pylon" else "endplate")


def record_values(record: dict, labels: list | None = None) -> dict:
    """{row: the value the winner FLEW}: its vector, and every PINNED row at
    its pin -- a row fixed by the player or by the tip device "none" (its
    height pinned at 0) is the design's too, whether or not the record's
    vector carries it."""
    labels = [str(v) for v in (labels if labels is not None else record.get("param_labels") or [])]
    xs = [float(u) for u in record.get("best_x") or []]
    v = {lab: xs[i] for i, lab in enumerate(labels) if i < len(xs)}
    for lab, pin in (((record.get("config") or {}).get("pinned")) or {}).items():
        v[str(lab)] = float(pin)
    return v


def _geometry_spec(record: dict, law: dict, role: str, name: str = "winglab wing",
                   sections: dict | None = None, tcs: tuple | None = None) -> "_WingSpec":
    """The WingSpec the renderer lofts from AeroBO's winner (no aero yet).

    The mount and the plate are DRAWN as flown (the contract fields of
    `aero.wing.WingSpec`): the pylon pair at its station, or the plates that
    carried it; the plate's lean and root blend -- the searched row, else the
    stated flag, else upright / a crease -- its turn law, its chord (the
    designed plate's row; 1 on the fence family, whose plate is the tip chord) and
    whether it follows the wing's chord law; its height as flown (0 under the
    tip device "none"). None of it is physics: the car flies `aero`, AeroBO's
    law."""
    if tcs is None or not record.get("param_labels"):
        _cfg, built, labels, xs = _record_problem(record)
        tcs = _family_tc(built, labels, xs)
    labels = [str(v) for v in record["param_labels"]] if record.get("param_labels") else labels
    v = record_values(record, labels)
    flags = dict((record.get("config") or {}).get("flags") or {})
    span = float(law["span_flown"])
    lam = float(v.get("taper", 1.0))
    S = float(law["S"])
    tc_main, tc_plate = tcs
    secs = sections or {}
    main = (secs.get("main") or {}).get("name") or _naca("24", tc_main)
    plate = ((secs.get("plate") or {}).get("name")
             or (_naca("00", tc_plate) if tc_plate is not None else ""))
    ride = v.get("ride_height_m", 0.0)
    mount = record_mount(flags)
    cant = v.get("endplate_cant_deg", flags.get("endplate_cant_deg"))
    blend = v.get("endplate_blend_frac", flags.get("blend_frac"))
    return _WingSpec(
        name=str(name), role=str(role), airfoil=str(main), span=span, chord=2.0 * S / (span * (1.0 + lam)),
        taper=lam, twist_root_deg=v.get("twist_root_deg", 0.0), twist_deg=v.get("twist_tip_deg", 0.0),
        plate_h=v.get("endplate_h_m", 0.0), plate_airfoil=str(plate),
        ride_h=(ride if role == "top" else ride - OFFSET_M), area=S, mount=mount,
        pylon_frac=(float(flags.get("mount_station_frac") or carmount.INBOARD_STATION_FRAC)
                    if mount == "pylon" else _WingSpec.pylon_frac),
        plate_cant_deg=float(CANT_UPRIGHT if cant is None else cant),
        plate_blend=float(blend or 0.0), plate_shape=str(flags.get("blend_shape") or "arc"),
        #  no chord row: WingLab's fence family (the pylons' tip device, a
        #  legacy fence), whose plate IS the wing's tip chord -- ratio 1, not
        #  carsim's customary 1.3 x an unstated ratio draws
        plate_chord_ratio=float(v.get("endplate_chord_ratio", 1.0)),
        plate_chord_follows=bool(flags.get("endplate_chord_follows")),
        notes="designed by WingLab on the garage's DESIGN page")


def to_wingspec(record: dict, law: dict, sections: dict | None, role: str, name: str):
    """(WingSpec, slot_updates) for AeroBO's winner (PLAN2 5.7).

    The geometry is AeroBO's: span = b_m - 2 x the plate's projection, the
    root chord of the straight-taper EQUIVALENT (2 S / (b (1 + taper))) --
    the free chord law's k1..k3 are not drawn (PLAN2 H6; the physics is the
    law, never the render) -- twists, plate height, the ride row (the flank's
    as the standoff it is). `sections` = {"main": chosen | None, "plate":
    chosen | None} (the models' `chosen` dicts); an undecided surface flies
    the family's own section (NACA 24tt main, NACA 00tt plate). `design`
    carries the provenance the law can be re-derived from; `aero` is the law.
    The mount and the plate are drawn as the record flew them (`_geometry_spec`:
    `mount`, `pylon_frac`, `plate_cant_deg`, `plate_blend` / `plate_shape`,
    `plate_chord_ratio`, `plate_chord_follows`, `plate_h`).
    Slot updates: the incidence is AeroBO's alpha*, and the TOP wing's height
    its searched ride height; the flanks keep x/h (mirrored by the garage)."""
    spec = _geometry_spec(record, law, role, name, sections)
    labels = [str(v) for v in record.get("param_labels") or []]
    xs = [float(v) for v in record["best_x"]]
    cfgd = dict(record.get("config") or {})
    fam = record.get("carsim_family")
    flags = dict(cfgd.get("flags") or {})
    spec.design = {
        "engine": "aerobo", "problem": family_of(record), "family": copy.deepcopy(fam),
        "labels": labels, "x": xs, "flags": flags,
        "bounds_overrides": copy.deepcopy(cfgd.get("bounds_overrides")),
        "pinned": copy.deepcopy(cfgd.get("pinned")),
        "record_path": record.get("path"), "n_evals": int(record.get("n_evals") or 0),
        "best_score": _num(record.get("best_score")), "score_units": record.get("score_units"),
        "objective": flags.get("car_objective") or api.CAR_DEFAULT_OBJECTIVE,
        "aerobo": AEROBO_COMMIT}
    spec.aero = dict(law)
    updates = {"inc_deg": float(law["alpha_design_deg"])}
    if role == "top" and law.get("ride_h") is not None:
        updates["h"] = float(law["ride_h"])
    return spec, updates


def record_of_design(design: dict, geom: dict, deck: float | None = None,
                     band: tuple | None = None) -> dict:
    """A record a committed wing's law can be RE-DERIVED from, at the slot as
    it stands now (the car page moved it). The top wing's family follows the
    deck under its new station (`deck`) and the slot's ride band there
    (`band`, `top_ride_band(car, x)`; without it, the deck + 0.14 m up to the
    family's own ceiling -- the car's it was designed on), and its ride row
    IS the slot's height; a flank flies with ground effect off, so its law
    does not depend on where it is mounted and only `derived_at` moves. `V`
    is the law's own speed."""
    d = copy.deepcopy(dict(design))
    fp = FamilyParams.from_json(d["family"])
    labels = list(d["labels"])
    x = [float(v) for v in d["x"]]
    flags = dict(d.get("flags") or {})
    if geom.get("V"):
        flags["V"] = float(geom["V"])
    if fp.role == "top" and geom.get("h") is not None:
        if deck is not None:
            band = (tuple(float(v) for v in band) if band is not None
                    else (float(deck) + TOP_DECK_CLEAR_M, float(fp.ride_band[1])))
            # the deck under the new station: the plates reach it, and so do
            # the pylons (a legacy fence family has none)
            fp = dataclasses.replace(fp, ride_band=band,
                                     deck=(float(deck) if (fp.plates or fp.pylons) else fp.deck))
        lo, hi = fp.ride_band
        if "ride_height_m" in labels:
            x[labels.index("ride_height_m")] = min(max(float(geom["h"]), lo), hi)
    return {"config": {"problem_name": ensure_family(fp), "mission_kwargs": {}, "flags": flags,
                       "bounds_overrides": d.get("bounds_overrides"), "pinned": d.get("pinned")},
            "best_x": x, "param_labels": labels, "carsim_family": _family_json(fp),
            "n_evals": d.get("n_evals"), "best_score": d.get("best_score"),
            "score_units": d.get("score_units")}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "-", str(name).strip().lower()).strip("-") or "section"


def section_to_airfoilspec(chosen: dict | None, lib=None) -> "_AirfoilSpec | None":
    """A section the design page chose, as a carsim library AirfoilSpec.

    The COORDINATES themselves (source "coords", AeroBO's closed loop), so no
    CST convention can differ between the two codes; CST weights ride along
    where AeroBO designed the shape. The family's own sections ("default")
    are carsim's NACA generator at the same code. The name is unique in `lib`
    (a library pick keeps its UIUC name when free), and `origin` says where
    the shape came from and at what Reynolds number."""
    if not chosen:
        return None
    src = chosen.get("source")
    if src == "default":
        tc = chosen.get("tc")
        name = str(chosen.get("name") or "")
        m = re.search(r"(\d{4})", name)
        code = m.group(1) if m else (f"00{int(round(100 * float(tc))):02d}" if tc else None)
        if code is None:
            return None
        return _AirfoilSpec(name=f"naca{code}", source="naca", code=code,
                            notes=f"NACA {code}: the WingLab family's own section")
    coords = chosen.get("coords")
    if coords is None or len(coords) < 10:
        return None
    pts = [[round(float(p[0]), 6), round(float(p[1]), 6)] for p in coords]
    re_ = _num((chosen.get("conditions") or {}).get("re"))
    at = f" @ Re {re_:.2g}" if re_ else ""
    if src == "optimised":
        seed = str(chosen.get("seed") or chosen.get("anchor_name") or "its seed")
        n = int(chosen.get("n_evals") or 0)
        base = f"{_slug(seed)}-winglab"
        origin = (f"WingLab: CST optimised from {seed}"
                  + (f", {n} evaluations" if n else "") + at)
    else:
        base = _slug(chosen.get("name") or "section")
        origin = f"WingLab: {chosen.get('name')}{at}"
    name = base
    if lib is not None and name in getattr(lib, "airfoils", {}):
        name = lib.unique_name("airfoils", base)
    return _AirfoilSpec(name=name, source="coords", points=pts,
                        w_upper=[float(v) for v in chosen.get("w_upper") or []],
                        w_lower=[float(v) for v in chosen.get("w_lower") or []],
                        notes=str(chosen.get("origin") or origin), origin=origin)


# =========================================================================== #
#  FIXTURES (PLAN2 5.8, 6.4, D12)                                              #
# =========================================================================== #
#  Real engine runs, captured once on a machine with XFOIL
#  (`python3 -m drive.aerobo_bridge --capture`), replayed by the design
#  page's deterministic checks through the SAME EngineJob path
#  (`design_jobs.ReplayRunner`). Every event and every result below is what
#  AeroBO emitted and returned -- nothing is synthesised; the only edit is
#  rounding DRAWING arrays to 6 significant figures: the x each "eval" event
#  carries (the plot's copy) and the shape arrays in `FIXTURE_ROUNDED` (a
#  section's 2001-point thickness distribution alone was half of 1.1 MB). The
#  run records keep AeroBO's exact numbers: a replayed record is continued
#  (Keep going) and re-derived (the law) exactly as a live one, and AeroBO
#  refuses to resume evaluations a rounding has moved off its box.
FIXTURE_DIR = REPO / "drive" / "data" / "aerobo_fixtures"
FIXTURE_FORMAT = 1
FIXTURE_KINDS = ("wing", "section", "screen", "lap")
FIXTURE_EVENT_KINDS = ("eval", "sweep", "phase", "total", "log")
#: (file stem, kind, the `meta.args` a replay is matched on). The models ask
#: for a screen by (surface, re_source), a section by (surface, variant) and a
#: wing by (role, variant, objective); a lap-time run is a WING job.
FIXTURES = (
    ("screen-main-library", "screen", {"surface": "main", "re_source": "library"}),
    ("screen-plate-library", "screen", {"surface": "plate", "re_source": "library"}),
    ("screen-main-mission", "screen", {"surface": "main", "re_source": "mission"}),
    ("section-main", "section", {"surface": "main", "variant": "full"}),
    ("section-main-continued", "section", {"surface": "main", "variant": "continued"}),
    ("section-plate", "section", {"surface": "plate", "variant": "full"}),
    ("section-plate-continued", "section", {"surface": "plate", "variant": "continued"}),
    ("wing-top", "wing", {"role": "top", "variant": "full", "objective": "efficiency"}),
    ("wing-top-continued", "wing", {"role": "top", "variant": "continued",
                                    "objective": "efficiency"}),
    ("wing-flank", "wing", {"role": "flank", "variant": "full", "objective": "efficiency"}),
    ("wing-flank-continued", "wing", {"role": "flank", "variant": "continued",
                                      "objective": "efficiency"}),
    ("wing-top-laptime", "wing", {"role": "top", "variant": "full", "objective": "laptime"}),
)
#: the budgets the fixtures fly: short, real runs (PLAN2 5.8)
CAPTURE_SECTION_BUDGET, CAPTURE_SECTION_STOP, CAPTURE_SECTION_MORE = 12, 5, 6
CAPTURE_WING_BUDGET, CAPTURE_WING_STOP, CAPTURE_WING_MORE = 20, 8, 6
CAPTURE_LAP_BUDGET = 10
#: the drawing-only arrays a fixture stores at 6 significant figures
FIXTURE_ROUNDED = frozenset({"thickness", "thickness_xc", "coords", "profile_V", "profile_x"})
#: what the files may weigh together: measured 0.72 MB (PLAN2 aimed at 0.6;
#: the sections' thickness distributions are 2001 points each), then 0.88 MB
#: at the recapture for task 41's per-car rows: those section searches beat
#: their seed, so their reports carry the seed's own drawing too
#: (`report.section.baseline`, 42 kB a report)
FIXTURE_MAX_BYTES = 900_000


def _jsonable(obj):
    """Plain JSON: numpy -> Python, tuples -> lists, non-finite -> None."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _num(obj)
    if obj is None or isinstance(obj, str):
        return obj
    if dataclasses.is_dataclass(obj):
        return _jsonable(asdict(obj))
    return str(obj)


def _sig6(v):
    return None if v is None else float(f"{float(v):.6g}")


def _round_x(fx: dict) -> dict:
    """A copy of `fx` with the drawing arrays at 6 s.f.: each eval event's x,
    and every array under a `FIXTURE_ROUNDED` key. Run records untouched."""
    def sig(v):
        if isinstance(v, list):
            return [sig(u) for u in v]
        return _sig6(v) if isinstance(v, float) else v

    def walk(o):
        if isinstance(o, dict):
            return {k: (sig(v) if k in FIXTURE_ROUNDED else walk(v)) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        return o
    fx = walk(copy.deepcopy(fx))
    for evs in (fx["events"], (fx.get("stopped") or {}).get("events") or []):
        for kind, p in evs:
            if kind == "eval" and p.get("x") is not None:
                p["x"] = [_sig6(v) for v in p["x"]]
    return fx


def _reference_car():
    """The car the self-check and the capture fly: carsim's default slots
    (flanks at x 0.97, h 0.90; top at x -0.90, h 1.55), the arena, dry, and a
    0.90 m deck under the top wing (PLAN2's measured case; the garage passes
    its own `garage.deck_z`)."""
    from types import SimpleNamespace

    class _Build:
        def __init__(self):
            self._s = {"left": SimpleNamespace(x=0.97, h=0.90, wing=""),
                       "right": SimpleNamespace(x=0.97, h=0.90, wing=""),
                       "top": SimpleNamespace(x=-0.90, h=1.55, wing="")}

        def slot(self, key):
            return self._s[key]

    return _Build(), _ms.MissionSpec(track="arena", mu_scale=1.0, stated=True), (lambda x: 0.90)


class _Tape:
    """A list-emit that records a run as the fixture's [kind, payload] pairs
    and, with `stop_at`, presses Stop when that evaluation arrives."""

    def __init__(self, name: str, stop_at: int | None = None, say=print):
        self.events, self.stop = [], threading.Event()
        self.name, self.stop_at, self.say, self.k = name, stop_at, say, 0

    def __call__(self, kind, **p):
        self.events.append([kind, _jsonable(p)])
        if kind in ("eval", "sweep"):
            self.k += 1
            n = p.get("n") if kind == "eval" else p.get("i")
            if self.say and (self.k == 1 or self.k % 5 == 0):
                self.say(f"    {self.name}: {kind} {n}")
            if self.stop_at is not None and self.k >= self.stop_at:
                self.stop.set()


def _cold_point(cond: dict, gates: dict):
    """The own-Re screen's point checkpoint, moved aside for the capture and
    put back afterwards: with it in place AeroBO answers the shortlist from
    it and emits no sweep, and the fixture is for the RUNNING screen. The
    XFOIL polar cache stays, so the sweep is real but quick."""
    import contextlib
    from aerobo import airfoil as _af

    @contextlib.contextmanager
    def ctx():
        prob = _af.AirfoilProblem(re=float(cond["re"]), mach=float(cond.get("mach") or 0.0),
                                  cl_design=float(cond["cl_design"]),
                                  tc_min=float(gates["tc_min"]), cm_max=float(gates["cm_max"]))
        path = Path(api.screen_checkpoint(prob))
        aside = path.with_name(path.name + ".capture-aside")
        moved = path.is_file() and path != Path(api.SCREEN_CHECKPOINT)
        if moved:
            os.replace(path, aside)
        try:
            yield
        finally:
            if moved:
                os.replace(aside, path)
    return ctx()


def capture(out_dir=FIXTURE_DIR, verbose: bool = True, deck: float | None = None) -> dict:
    """Run and record every fixture in FIXTURES (a few minutes, XFOIL needed).

    Returns {stem: bytes written}. The top wing's operating point is the
    reference car's (`_reference_car`; `deck` overrides its 0.90 m deck)."""
    if not xfoil_ok():
        raise RuntimeError("the capture flies live XFOIL: install it (or unset CARSIM_NO_XFOIL)")
    say = print if verbose else None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    build, mission, deck_z = _reference_car()
    if deck is not None:
        deck_z = (lambda x, d=float(deck): d)
    top = operating_point("top", "top", build, mission, deck_z=deck_z)
    flank = operating_point("left", "flank", build, mission)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    written: dict = {}
    t_all = time.perf_counter()

    def meta(args: dict, **more) -> dict:
        return {"captured": stamp, "aerobo": AEROBO_COMMIT, "bridge_physics": BRIDGE_PHYSICS,
                "args": _jsonable({**args, **more})}

    def fly(name, runner, stop_at=None):
        import warnings
        tape = _Tape(name, stop_at, say)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            # the engine's own warnings, where they fell in the run, as the
            # Output panel's lines (design_jobs routes a worker's warnings
            # into its job the same way)
            warnings.simplefilter("default")
            warnings.showwarning = (lambda m, c, f, ln, file=None, line=None:
                                    tape("log", text=f"{c.__name__}: {m}", level="warn"))
            result = runner(tape, tape.stop)
        if say:
            say(f"  {name}: {tape.k} events{' (stopped)' if stop_at else ''}, "
                f"{time.perf_counter() - t0:.1f} s")
        return tape.events, _jsonable(result)

    def write(stem, kind, args, events, result, stopped=None, **more):
        fx = {"format": FIXTURE_FORMAT, "kind": kind, "meta": meta(args, **more),
              "events": events, "result": result}
        if stopped is not None:
            fx["stopped"] = {"events": stopped[0], "result": stopped[1]}
        text = json.dumps(_round_x(fx), separators=(",", ":"), allow_nan=False)
        p = out / f"{stem}.json"
        tmp = p.with_name(f".{p.name}.tmp")
        tmp.write_text(text)
        os.replace(tmp, p)
        written[stem] = len(text)

    args = {stem: a for stem, _k, a in FIXTURES}
    top_flags = wing_physics_flags({}, top, None, ensure_family(family_params(top)))
    top_box = family_box(ensure_family(family_params(top)), top_flags, top.size_rows or None)

    # -- screens --------------------------------------------------------------
    lib = api.screen_library_point()
    lib_cond = {"re": float(lib["re"]), "mach": 0.0}
    main_cond = section_conditions(top, top_box, "main")
    plate_cond = section_conditions(top, top_box, "plate")
    ev, res = fly("screen main (library point)", screen_runner(
        "main", WING_WEIGHTS, {**lib_cond, "cl_design": main_cond["cl_design"]}, SCREEN_GATES,
        None, "library"))
    write("screen-main-library", "screen", args["screen-main-library"], ev, res,
          re=lib_cond["re"], cl=main_cond["cl_design"])
    ev, res = fly("screen plate (library point)", screen_runner(
        "plate", PLATE_WEIGHTS, {**lib_cond, "cl_design": 0.0}, SCREEN_GATES, None, "library"))
    plate_screen = res
    write("screen-plate-library", "screen", args["screen-plate-library"], ev, res,
          re=lib_cond["re"], cl=0.0)
    with _cold_point(main_cond, SCREEN_GATES):
        ev, res = fly("screen main (own Re)", screen_runner(
            "main", WING_WEIGHTS, main_cond, SCREEN_GATES, None, "mission"))
    main_screen = res
    write("screen-main-mission", "screen", args["screen-main-mission"], ev, res,
          re=main_cond["re"], cl=main_cond["cl_design"])

    # -- sections: full (+ a real Stop), and a Keep going of the full run --------
    def section(surface, cond, screen, weights):
        cand = (screen.get("candidates") or [None])[0]
        anchor = (cand["w_upper"], cand["w_lower"]) if cand else None
        shape = shape_kwargs(cond, surface, None, weights, (screen["report"] or {}).get("reference"),
                             anchor=anchor)
        search = dict(section_search(None, "composite_goal", surface))
        search.pop("plan", None)
        search["budget"] = CAPTURE_SECTION_BUDGET
        full = fly(f"section {surface}", section_runner(shape, search, 0))
        stopped = fly(f"section {surface} (Stop)", section_runner(shape, search, 0),
                      stop_at=CAPTURE_SECTION_STOP)
        more = {"budget": search["budget"], "n_init": search["n_init"], "re": cond["re"],
                "cl": cond["cl_design"], "seed_section": cand["name"] if cand else None}
        write(f"section-{surface}", "section", args[f"section-{surface}"], *full,
              stopped=stopped, stopped_at=CAPTURE_SECTION_STOP, **more)
        cont = continue_section(full[1]["report"], full[1]["launch"], CAPTURE_SECTION_MORE)
        s2 = dict(cont["search"])
        resume = s2.pop("resume")
        ev2 = fly(f"section {surface} (Keep going +{CAPTURE_SECTION_MORE})",
                  section_runner(cont["shape"], s2, cont["seed"], resume))
        write(f"section-{surface}-continued", "section", args[f"section-{surface}-continued"],
              *ev2, **{**more, "budget": s2["budget"], "n_prior": cont["note"].get("resumed")})

    section("main", main_cond, main_screen, WING_WEIGHTS)
    section("plate", plate_cond, plate_screen, PLATE_WEIGHTS)

    # -- wings: full (+ a real Stop), and a Keep going of the STOPPED run --------
    def wing(role, op, choices, stem, budget, stop_at=None, more_evals=None, lap=None):
        fp = family_params(op, choices, lap)
        fam = ensure_family(fp)
        flags = wing_physics_flags(choices, op, None, fam)
        plan = wing_plan(fam, flags, op.size_rows or None, None, "balanced")
        eff = dict(wing_search(None, plan, fam))
        eff["budget"] = int(budget)
        cfg = wing_cfg(fam, flags, eff, 0, op.size_rows or None)
        conv = stop_rule_factory(eff)
        full = fly(f"{stem}", wing_runner(cfg, fp, conv))
        stopped = (fly(f"{stem} (Stop)", wing_runner(cfg, fp, conv), stop_at=stop_at)
                   if stop_at else None)
        more = {"budget": int(budget), "optimiser": cfg.optimiser, "problem": fam,
                "V": op.V, "stopped_at": stop_at}
        write(stem, "wing", args[stem], *full, stopped=stopped, **more)
        if more_evals:
            rec = stopped[1]["record"]
            cfg2, note = continue_wing(rec, more_evals)
            cont = fly(f"{stem} (Keep going +{more_evals})",
                       wing_runner(cfg2, fp, None, note.get("resume")))
            write(f"{stem}-continued", "wing", args[f"{stem}-continued"], *cont,
                  **{**more, "budget": int(cfg2.budget), "n_prior": note.get("resumed")})

    wing("top", top, {}, "wing-top", CAPTURE_WING_BUDGET, CAPTURE_WING_STOP, CAPTURE_WING_MORE)
    wing("flank", flank, {}, "wing-flank", CAPTURE_WING_BUDGET, CAPTURE_WING_STOP,
         CAPTURE_WING_MORE)
    wing("top", top, {"plates": False, "objective": "laptime"}, "wing-top-laptime",
         CAPTURE_LAP_BUDGET, lap=lap_params(mission, build, None, "top"))

    total = sum(written.values())
    if say:
        say(f"  captured {len(written)} fixtures, {total / 1e3:.0f} kB, into "
            f"{out.relative_to(REPO) if out.is_relative_to(REPO) else out} in "
            f"{time.perf_counter() - t_all:.0f} s")
    return written


def fixture_problems(fx) -> list:
    """Everything wrong with a fixture's shape, [] when none (the format
    `design_jobs.load_fixture` reads; checked here too, since the bridge does
    not import the job layer)."""
    if not isinstance(fx, dict):
        return ["not a JSON object"]
    out = []
    if fx.get("format") != FIXTURE_FORMAT:
        out.append(f"format {fx.get('format')!r}")
    if fx.get("kind") not in FIXTURE_KINDS:
        out.append(f"kind {fx.get('kind')!r}")
    m = fx.get("meta")
    if not isinstance(m, dict) or not isinstance(m.get("args"), dict):
        out.append("meta.args missing")
    elif m.get("aerobo") != AEROBO_COMMIT:
        out.append(f"captured on aerobo {m.get('aerobo')!r}, vendored {AEROBO_COMMIT} "
                   f"(re-capture: python3 -m drive.aerobo_bridge --capture)")
    if "result" not in fx:
        out.append("result missing")

    def check(evs, where):
        if not isinstance(evs, list) or not all(
                isinstance(e, list) and len(e) == 2 and e[0] in FIXTURE_EVENT_KINDS
                and isinstance(e[1], dict) for e in evs):
            out.append(f"{where}events are not [kind, payload] pairs")
            return
        ns = [p.get("n") for k, p in evs if k == "eval"]
        if any(not isinstance(v, int) for v in ns) or any(b <= a for a, b in zip(ns, ns[1:])):
            out.append(f"{where}eval n is not an increasing integer")
    check(fx.get("events"), "")
    if fx.get("stopped") is not None:
        st = fx["stopped"]
        if not isinstance(st, dict) or "result" not in st:
            out.append("stopped variant has no result")
        else:
            check(st.get("events"), "stopped ")
    return out


#: "CARRIED BY" -- every configuration the owner's request names (2026-09-25:
#: the endplates with the root blend and the lean each stated or optimised,
#: the pylons with each tip device; both chord laws where cheap), as
#: (label, the card's answers, what must follow): AeroBO's family, the flags
#: sent with their values, the flags NOT sent, the plate rows the box
#: searches, the pins. Written out, not derived: rows B29-B32 (and the
#: models' M35) hold the code to this table, not to itself.
_PYL = {"car_mount_model": "continuum", "mount_kind": "pylon", "mount_side": "pressure",
        "mount_station_frac": float(carmount.INBOARD_STATION_FRAC)}
_NO_PYL = tuple(_PYL) + ("endplate_chord_follows",)
MOUNT_CONFIGS = (
    ("endplates, stated", {"plates": True},
     {"base": "car rear wing + endplates + free chord law", "sent": {"section": "shaped"},
      "absent": ("blend_frac", "blend_shape", "endplate_cant_deg") + _NO_PYL, "rows": (), "pins": {}}),
    ("endplates, blend 0.3 + lean 70", {"plates": True, "blend_frac": 0.3, "cant_deg": 70.0},
     {"base": "car rear wing + endplates + free chord law",
      "sent": {"section": "shaped", "blend_frac": 0.3, "blend_shape": "spiral",
               "endplate_cant_deg": 70.0},
      "absent": _NO_PYL, "rows": (), "pins": {}}),
    ("endplates, lean optimised", {"plates": True, "cant": "free", "cant_deg": 70.0},
     {"base": "car rear wing + endplates [free cant] + free chord law", "sent": {"section": "shaped"},
      "absent": ("endplate_cant_deg", "blend_frac", "blend_shape") + _NO_PYL,
      "rows": ("endplate_cant_deg",), "pins": {}}),
    ("endplates, blend optimised", {"plates": True, "blend": "free", "blend_frac": 0.3},
     {"base": "car rear wing + endplates [free blend] + free chord law",
      "sent": {"section": "shaped", "blend_shape": "spiral"},
      "absent": ("blend_frac", "endplate_cant_deg") + _NO_PYL, "rows": ("endplate_blend_frac",),
      "pins": {}}),
    ("endplates, both optimised", {"plates": True, "blend": "free", "cant": "free"},
     {"base": "car rear wing + endplates [free cant, free blend] + free chord law",
      "sent": {"section": "shaped", "blend_shape": "spiral"},
      "absent": ("blend_frac", "endplate_cant_deg") + _NO_PYL,
      "rows": ("endplate_cant_deg", "endplate_blend_frac"), "pins": {}}),
    ("endplates, both optimised, straight taper",
     {"plates": True, "blend": "free", "cant": "free", "chord_law": False},
     {"base": "car rear wing + endplates [free cant, free blend]",
      "sent": {"section": "shaped", "blend_shape": "spiral"},
      "absent": ("blend_frac", "endplate_cant_deg") + _NO_PYL,
      "rows": ("endplate_cant_deg", "endplate_blend_frac"), "pins": {}}),
    ("pylons, none", {"plates": False, "tip": "none"},
     {"base": "car rear wing + free chord law", "sent": dict(_PYL),
      "absent": ("section", "blend_frac", "blend_shape", "endplate_cant_deg",
                 "endplate_chord_follows"), "rows": (), "pins": {"endplate_h_m": 0.0}}),
    ("pylons, vertical", {"plates": False, "tip": "vertical"},
     {"base": "car rear wing + free chord law", "sent": {**_PYL, "endplate_chord_follows": True},
      "absent": ("section", "blend_frac", "blend_shape", "endplate_cant_deg"), "rows": (),
      "pins": {}}),
    ("pylons, vertical, tip chord held", {"plates": False, "tip": "vertical", "tip_chord": False},
     {"base": "car rear wing + free chord law", "sent": dict(_PYL),
      "absent": ("section", "blend_frac", "blend_shape", "endplate_cant_deg",
                 "endplate_chord_follows"), "rows": (), "pins": {}}),
    ("pylons, canted 70", {"plates": False, "tip": "canted", "tip_cant_deg": 70.0},
     {"base": "car rear wing + free chord law",
      "sent": {**_PYL, "endplate_chord_follows": True, "endplate_cant_deg": 70.0},
      "absent": ("section", "blend_frac", "blend_shape"), "rows": (), "pins": {}}),
    ("pylons, blended", {"plates": False, "tip": "blended"},
     {"base": "car rear wing + free chord law",
      "sent": {**_PYL, "endplate_chord_follows": True, "blend_frac": 0.5, "blend_shape": "spiral"},
      "absent": ("section", "endplate_cant_deg"), "rows": (), "pins": {}}),
    ("pylons, blended, straight taper", {"plates": False, "tip": "blended", "chord_law": False},
     {"base": "car rear wing",
      "sent": {**_PYL, "endplate_chord_follows": True, "blend_frac": 0.5, "blend_shape": "spiral"},
      "absent": ("section", "endplate_cant_deg"), "rows": (), "pins": {}}),
)
#: the family names every run before "Carried by" was registered under, as
#: the untouched copy computed them for the self-check's car (the reference
#: car, arena, dry; row B28): they, and AeroBO's cache under them, must not move
LEGACY_NAMES = {
    "top default": "carsim top · car rear wing + endplates + free chord law #9e6d1757",
    "flank default": "carsim flank · car rear wing + endplates + free chord law #7fea3ecb",
    "top endplates, straight": "carsim top · car rear wing + endplates #2583fdd3",
    "top fences": "carsim top · car rear wing + free chord law #d4db5855",
    "flank fences, straight": "carsim flank · car rear wing #9f9e6cb6",
    "top fences, lap": "carsim top · car rear wing + free chord law #1136519d",
}


def _mount_record(op: OperatingPoint, ch: dict, stale_pin: float | None = None) -> tuple:
    """(record, x, labels) of a hand-built record of card `ch` at the middle
    of its box -- the shape a run record has (config, best_x, labels,
    carsim_family). `stale_pin` puts that value in the vector at every pinned
    row, so a reader that ignores the pin is caught."""
    fp = family_params(op, ch)
    fam = ensure_family(fp)
    flags = wing_physics_flags(ch, op, None, fam)
    pins = tip_pins(ch)
    built = api.PROBLEM_SPECS[fam].build({}, flags, op.size_rows)
    labels = [str(v) for v in built.param_labels]
    x = 0.5 * (built.bounds[:, 0] + built.bounds[:, 1])
    for lab, v in pins.items():
        x[labels.index(lab)] = float(v if stale_pin is None else stale_pin)
    rec = {"config": {"problem_name": fam, "mission_kwargs": {}, "flags": flags,
                      "bounds_overrides": copy.deepcopy(op.size_rows), "pinned": pins or None},
           "best_x": [float(v) for v in x], "param_labels": labels,
           "carsim_family": _family_json(fp)}
    return rec, x, labels


# =========================================================================== #
#  SELF-CHECK  (python3 -m drive.aerobo_bridge; rows B1-B32, PLAN2 9.1)        #
# =========================================================================== #
#: the provenance block VENDORED.md carries (written by aerobo/sync.sh)
_PROV = re.compile(r"<!-- provenance:begin.*?```(.*?)```.*?<!-- provenance:end -->", re.S)


def vendored_provenance() -> dict:
    """VENDORED.md's provenance block as `{field: value}`."""
    m = _PROV.search((AEROBO_ROOT / "VENDORED.md").read_text())
    out = {}
    for line in (m.group(1) if m else "").strip().splitlines():
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out


def vendored_digest(root: Path = AEROBO_ROOT) -> tuple:
    """(file counts, content sha256) of the vendored tree -- the same definition
    aerobo/sync.sh writes into VENDORED.md: sha256 over sorted
    `path NUL sha256(file)` lines, bytecode caches excluded."""
    rows, counts = [], {"src": 0, "data": 0, "records": 0, "LICENSE": 0}
    for top in counts:
        base = root / top
        paths = [base] if base.is_file() else [
            Path(d) / f for d, _, fs in os.walk(base)
            if "__pycache__" not in Path(d).parts for f in fs
            if f != ".DS_Store" and not f.endswith(".pyc")]
        for p in paths:
            rel = p.relative_to(root).as_posix()
            rows.append(f"{rel}\0{hashlib.sha256(p.read_bytes()).hexdigest()}\n")
            counts[top] += 1
    rows.sort()
    return counts, hashlib.sha256("".join(rows).encode()).hexdigest()


def _tree_state(root: Path = AEROBO_ROOT) -> dict:
    """path -> (size, mtime_ns) of every vendored file outside results/ and the
    bytecode caches: what B26 compares before and after the whole check."""
    out = {}
    for d, dirs, fs in os.walk(root):
        parts = Path(d).relative_to(root).parts
        if parts[:1] == ("results",) or "__pycache__" in parts:
            continue
        for f in fs:
            p = Path(d) / f
            st = p.stat()
            out[p.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns)
    return out


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def self_check(verbose: bool = True, slow: bool = False) -> bool:
    """Rows B1-B32 (PLAN2 9.1; B27 task 41; B28-B32 "Carried by"). AeroBO's
    own warnings (the feasibility guide's notes, botorch's) are collected
    rather than printed between the rows: in the game the warnings router
    shows them in the Output panel."""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ok = _self_check(verbose, slow)
    if verbose and caught:
        seen = sorted({f"{w.category.__name__}: {str(w.message).split('. ')[0]}" for w in caught})
        print(f"  ({len(caught)} engine warnings, {len(seen)} distinct -- the Output panel's lines)")
    return ok


def _self_check(verbose: bool = True, slow: bool = False) -> bool:
    t_start = time.perf_counter()
    tree0 = _tree_state()
    ok = True
    counts = {"pass": 0, "fail": 0, "skip": 0}

    def rep(tag, passed, msg="", skip=False):
        nonlocal ok
        if skip:
            counts["skip"] += 1
            if verbose:
                print(f"  [skip] {tag}: {msg}")
            return
        ok = ok and bool(passed)
        counts["pass" if passed else "fail"] += 1
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}: {msg}")

    def guard(tag, fn):
        """A row whose own code raised is a FAILED row, never a crashed check."""
        try:
            fn()
        except Exception as exc:                        # noqa: BLE001
            import traceback
            rep(tag, False, f"{type(exc).__name__}: {exc}")
            if verbose:
                traceback.print_exc()

    # -- B1-B4: the vendored tree -------------------------------------------- #
    def b1():
        where = Path(api.__file__).resolve()
        rep("B1 aerobo imported from the vendored copy",
            AEROBO_SRC.resolve() in where.parents and str(AEROBO_SRC) == sys.path[0],
            f"{where.relative_to(REPO)}; aerobo/src first on sys.path")
    guard("B1 aerobo imported from the vendored copy", b1)

    def b2():
        prov = vendored_provenance()
        n, digest = vendored_digest()
        want = f"src {n['src']}, data {n['data']}, records {n['records']}, LICENSE {n['LICENSE']}"
        seed_sha = prov.get("seed sha256", "").split()[0] if prov.get("seed sha256") else ""
        rep("B2 VENDORED.md: commit, file counts, content digest and seed match the tree",
            prov.get("commit") == AEROBO_COMMIT and prov.get("files") == want
            and (n["src"], n["data"], n["records"], n["LICENSE"]) == (68, 2260, 1, 1)
            and prov.get("content sha256") == digest and seed_sha == _sha(SEED),
            f"commit {prov.get('commit')} ({prov.get('describe')}), {want}, "
            f"content {digest[:12]}, seed {seed_sha[:12]}")
    guard("B2 VENDORED.md: commit, file counts, content digest and seed match the tree", b2)

    def b3():
        stamp = json.loads((AEROBO_ROOT / "data" / "search_budget.json").read_text()).get("measured")
        n_dat = len(list((AEROBO_ROOT / "data" / "airfoils" / "uiuc").glob("*.dat")))
        branch = AEROBO_ROOT / "records" / "airfoil_screen_branch_v2.json"
        rep("B3 data present: the budget study, the UIUC library, the branch sidecar",
            stamp == "2026-08-05" and n_dat == 2174 and branch.is_file(),
            f"search_budget measured {stamp}, {n_dat} uiuc .dat, "
            f"{branch.name} {branch.stat().st_size / 1e6:.1f} MB")
    guard("B3 data present: the budget study, the UIUC library, the branch sidecar", b3)

    def b4():
        dst = Path(api.SCREEN_CHECKPOINT)
        again = install_seed()
        n = len(json.loads(dst.read_text())) if dst.is_file() else 0
        rep("B4 the warm screen checkpoint is installed where WingLab reads it",
            dst.is_file() and not again and n == 2174,
            f"{dst.relative_to(REPO)}: {n} sections; a second install copies nothing")
    guard("B4 the warm screen checkpoint is installed where WingLab reads it", b4)

    # -- B5-B10: the slot families ------------------------------------------- #
    def b5():
        names = set(api.problem_names())
        rep("B5 WingLab registers the four car families the slots are built on",
            set(BASES.values()) <= names, ", ".join(BASES.values()))
    guard("B5 WingLab registers the four car families the slots are built on", b5)

    def b6():
        sec = [section_plan("composite_goal", e).budget for e in _budget.EFFORTS]
        top = ensure_family(FamilyParams("top", True, True, (1.04, 1.85), 0.90))
        flags = {"V": 40.0, "chord_trend": "root_largest", "mount": "tips", "section": "shaped"}
        wing = [wing_plan(top, flags, None, None, e).budget for e in _budget.EFFORTS]
        rep("B6 WingLab's budgets: section 109/164/240, wing (top slot, 14-D) 42/53/87",
            sec == [109, 164, 240] and wing == [42, 53, 87],
            f"section {'/'.join(map(str, sec))}, wing {'/'.join(map(str, wing))} "
            f"(study measured {study_stamp().get('measured')})")
    guard("B6 WingLab's budgets: section 109/164/240, wing (top slot, 14-D) 42/53/87", b6)

    def b7():
        base = BASES[("plates", "law")]
        own = endplate.CarWingEndplateProblem
        same = ensure_family(FamilyParams("top", True, True, tuple(own.RIDE_HEIGHT_BOUNDS_M), None,
                                          rho=float(own.rho), mu=float(own.mu)))
        flags = {"chord_trend": "root_largest", "mount": "tips", "section": "shaped"}
        b0 = api.PROBLEM_SPECS[base].build({}, flags, None)
        b1_ = api.PROBLEM_SPECS[same].build({}, flags, None)
        x = 0.5 * (b0.bounds[:, 0] + b0.bounds[:, 1])
        e0, e1 = b0.evaluate(x), b1_.evaluate(x)
        same_box = np.array_equal(b0.bounds, b1_.bounds) and b0.param_labels == b1_.param_labels
        rep("B7 identity: a slot family at WingLab's own band, deck and air IS WingLab's family",
            same_box and all(e0[k] == e1[k] for k in ("CZ", "CD", "score"))
            and hasattr(api, "_BuiltProblem") and hasattr(api, "_apply_overrides"),
            f"CZ {e1['CZ']:.10f}, CD {e1['CD']:.10f}, score {e1['score']:.10f} bit for bit; "
            f"box identical")
    guard("B7 identity: a slot family at WingLab's own band, deck and air IS WingLab's family", b7)

    def b8():
        top = ensure_family(FamilyParams("top", True, True, (1.04, 1.85), 0.90))
        built = api.PROBLEM_SPECS[top].build({}, {"V": 40.0, "mount": "tips", "section": "shaped",
                                                  "chord_trend": "root_largest"}, None)
        ir = list(built.param_labels).index("ride_height_m")
        x = 0.5 * (built.bounds[:, 0] + built.bounds[:, 1])
        got = []
        for h in (1.04, 1.85):
            xx = x.copy()
            xx[ir] = h
            ev = built.evaluate(xx)
            got.append((h, bool(ev["feasible"]), float(ev.get("CZ") or 0.0)))
        rep("B8 top slot: ground effect over the deck, feasible across the slot's band",
            all(f for _, f, _ in got) and got[0][2] > got[1][2],
            "; ".join(f"h {h:.2f} m CZ {cz:.4f}" for h, _, cz in got) + " over a 0.90 m deck")
    guard("B8 top slot: ground effect over the deck, feasible across the slot's band", b8)

    def b9():
        czs = []
        for off in (OFFSET_M, 1000.0):
            fam = ensure_family(FamilyParams("flank", True, True, (off + 0.25, off + 0.70), off))
            built = api.PROBLEM_SPECS[fam].build({}, {"V": 29.0875, "mount": "tips", "section": "shaped",
                                                      "chord_trend": "root_largest"}, None)
            x = 0.5 * (built.bounds[:, 0] + built.bounds[:, 1])
            czs.append(float(built.evaluate(x)["CZ"]))
        rel = abs(czs[0] - czs[1]) / abs(czs[1])
        rep("B9 flank: ground effect OFF (the image plane 200 m away)", rel < 1e-5,
            f"CZ {czs[0]:.7f} at {OFFSET_M:g} m vs {czs[1]:.7f} at 1000 m: {rel:.1e} relative")
    guard("B9 flank: ground effect OFF (the image plane 200 m away)", b9)

    # the self-check's car: carsim's default slots, the arena dry, and a 0.90 m
    # deck under the top wing (the scuttle's height; the garage passes its own
    # garage.deck_z)
    build, mission, deck = _reference_car()
    ctx = {}

    def b10():
        op = operating_point("left", "flank", build, mission)
        fam = ensure_family(family_params(op))
        flags = wing_physics_flags({}, op, None, fam)
        built = api.PROBLEM_SPECS[fam].build({}, flags, op.size_rows)
        from scipy.stats import qmc
        u = qmc.Sobol(len(built.param_labels), scramble=True, seed=0).random_base2(8)[:200]
        lo, hi = built.bounds[:, 0], built.bounds[:, 1]
        n_ok = sum(bool(built.evaluate(lo + ui * (hi - lo))["feasible"]) for ui in u)
        ctx["flank_op"], ctx["flank_fam"] = op, fam
        rep("B10 flank default box: WingLab finds a feasible design often enough to search",
            n_ok >= 60, f"{n_ok}/200 Sobol draws feasible at h {op.h:.2f} m "
            f"(b {op.size_rows['b_m']}, S {op.size_rows['S_m2']}, V {op.V:.2f} m/s)")
    guard("B10 flank default box: WingLab finds a feasible design often enough to search", b10)

    def b11():
        op = operating_point("top", "top", build, mission, deck_z=deck)
        fam = ensure_family(family_params(op))
        flags = wing_physics_flags({}, op, None, fam)
        plan = wing_plan(fam, flags, op.size_rows or None, None, "balanced")
        eff = wing_search(None, plan, fam)
        cfg = wing_cfg(fam, flags, eff, 0, op.size_rows or None)
        want = {"chord_trend": "root_largest", "mount": "tips", "section": "shaped", "V": op.V,
                "bo_refusal": "worst", "bo_feasibility": "guide"}
        ctx["top_op"], ctx["top_fam"], ctx["top_cfg"], ctx["top_eff"] = op, fam, cfg, eff
        rep("B11 V3 parity: the default top-slot run is WingLab's car session's",
            cfg.problem_name == fam and cfg.optimiser == "bo_slsqp" and cfg.budget == 53
            and all(cfg.flags.get(k) == v for k, v in want.items())
            and "car_objective" not in cfg.flags and cfg.pinned is None,
            f"{cfg.optimiser} x {cfg.budget}, flags {sorted(cfg.flags)}, V {op.V:.2f} m/s "
            f"({op.V_source}), ride band {op.ride_band[0]:.2f}-{op.ride_band[1]:.2f} m")
    guard("B11 V3 parity: the default top-slot run is WingLab's car session's", b11)

    # -- B12-B14: the runners, called directly with a list for emit ----------- #
    def _collect(stop_at=None, stop=None):
        """(events, emit): emit appends; with `stop_at` it presses Stop when
        that evaluation (or sweep) arrives -- the player's button, on time."""
        events = []

        def emit(kind, **p):
            events.append((kind, p))
            if stop_at is not None and kind in ("eval", "sweep") \
                    and int(p.get("n") if kind == "eval" else p.get("i")) >= stop_at:
                stop.set()
        return events, emit

    def _ns(events):
        return [p["n"] for k, p in events if k == "eval"]

    def _top_wing(budget):
        op, fam = ctx["top_op"], ctx["top_fam"]
        flags = wing_physics_flags({}, op, None, fam)
        eff = {"optimiser": "bo", "budget": int(budget), "n_init": 4, "refusal": "worst"}
        return wing_cfg(fam, flags, eff, 0, op.size_rows or None), _FAMILIES[fam]

    def b12():
        cfg, fp = _top_wing(8)
        events, emit = _collect()
        t0 = time.perf_counter()
        rec = wing_runner(cfg, fp)(emit, threading.Event())["record"]
        ns = _ns(events)
        ctx["wing8"] = rec
        rep("B12 wing runner: one eval event per evaluation, n 1..8, the record kept",
            ns == list(range(1, 9)) and rec["n_evals"] == 8 and not rec.get("partial")
            and rec["carsim_family"] == _family_json(fp) and events[0] == ("total", {"n": 8})
            and all(p["x"] is not None and len(p["x"]) == len(rec["param_labels"])
                    for k, p in events if k == "eval"),
            f"bo x 8 on the top slot: n {ns[0]}..{ns[-1]}, best {rec['best_score']:.3f} "
            f"{rec.get('score_units') or ''}, {time.perf_counter() - t0:.1f} s")
    guard("B12 wing runner: one eval event per evaluation, n 1..8, the record kept", b12)

    def b13():
        cfg, fp = _top_wing(8)
        stop = threading.Event()
        ev1, emit1 = _collect(stop_at=5, stop=stop)
        rec = wing_runner(cfg, fp)(emit1, stop)["record"]
        cfg2, note = continue_wing(rec, 3)
        ev2, emit2 = _collect()
        rec2 = wing_runner(cfg2, fp, None, note.get("resume"))(emit2, threading.Event())["record"]
        rep("B13 Stop keeps the best, Keep going resumes it: nothing re-flown",
            rec.get("partial") and rec.get("stop_reason") == STOP_PLAYER and _ns(ev1) == [1, 2, 3, 4, 5]
            and rec["n_evals"] == 5 and continue_extra_default(rec) == 3
            and _ns(ev2) == [6, 7, 8] and rec2.get("resumed") == 5 and len(rec2["history"]) == 8,
            f"stopped at {rec['n_evals']} ({rec.get('stop_reason')}); +3 -> n {_ns(ev2)}, "
            f"resumed {rec2.get('resumed')}, history {len(rec2['history'])}")
    guard("B13 Stop keeps the best, Keep going resumes it: nothing re-flown", b13)

    def b14():
        lib = api.screen_library_point()
        cond = {"re": float(lib["re"]), "mach": 0.0, "cl_design": REFERENCE_CL}
        t0 = time.perf_counter()
        main = screen_runner("main", WING_WEIGHTS, cond, SCREEN_GATES, None, "library")(
            _collect()[1], threading.Event())
        plate = screen_runner("plate", PLATE_WEIGHTS, {**cond, "cl_design": 0.0}, SCREEN_GATES,
                              None, "library")(_collect()[1], threading.Event())
        #  the same screen with the t/c gate at the shape optimiser's 0.10: the
        #  gate reaches the engine (the published 0.15 keeps far fewer)
        thin = screen_runner("main", WING_WEIGHTS, cond, OPT_GATES, None, "library")(
            _collect()[1], threading.Event())["report"]
        ctx["lib_main"] = main
        mr, pr = main["report"], plate["report"]
        sym = set(api.symmetric_section_names())
        names = [r["name"] for r in pr["ranked"]]
        camber = [api.section_max_camber(np.asarray(c["coords"])) for c in plate["candidates"][:5]]
        rep("B14 library-point screens: the wing ranks the library, the plate symmetric only",
            main["re_source"] == "library" and mr["n_screened"] == 2174 and mr["n_eligible"] >= 100
            and thin["n_eligible"] >= 500 and len(mr["ranked"]) == TOP_N
            and len(main["candidates"]) == TOP_N
            and len(sym) == 229 and pr["n_screened"] == 229 and set(names) <= sym
            and max(camber) <= 0.005,
            f"main {mr['n_eligible']} of {mr['n_screened']} eligible at cl {REFERENCE_CL:g}, "
            f"t/c >= 0.15 ({thin['n_eligible']} at t/c >= 0.10; leader {mr['ranked'][0]['name']}); "
            f"plate {pr['n_eligible']} of {pr['n_screened']} symmetric, top-5 camber "
            f"<= {max(camber):.4f}; {time.perf_counter() - t0:.2f} s")
    guard("B14 library-point screens: the wing ranks the library, the plate symmetric only", b14)

    # -- B15-B17: what a run is launched with ----------------------------------- #
    def b15():
        lib = api.screen_library_point()["re"]
        at_lib = section_flag_value({"source": "library", "name": "e423",
                                     "conditions": {"re": lib, "mach": 0.0}})
        own = section_flag_value({"source": "library", "name": "e423",
                                  "conditions": {"re": 6.8e5, "mach": 0.0}})
        opt = section_flag_value({"source": "optimised", "name": "CST", "w_upper": [0.2] * 4,
                                  "w_lower": [-0.1] * 4, "conditions": {"re": 6.8e5}})
        dflt = section_flag_value({"source": "default", "name": "NACA 0010"})
        rep("B15 section flag values: a name, a name at its point, CST weights at theirs",
            at_lib == "e423" and own == {"name": "e423", "re": 6.8e5, "mach": 0.0}
            and set(opt) == {"w_upper", "w_lower", "name", "re"} and dflt is None,
            f"library point -> {at_lib!r}; own Re -> {sorted(own)}; optimised -> {sorted(opt)}; "
            f"family default -> None")
    guard("B15 section flag values: a name, a name at its point, CST weights at theirs", b15)

    def b16():
        from .track import make_track
        prof = _ms.TrackProfile.from_track(make_track("arena"))
        ts = track_spec(prof)
        op = ctx["top_op"]
        ch = {"plates": False, "chord_law": True, "objective": "laptime"}
        lap = lap_params(mission, build, None, "top")
        fam = ensure_family(family_params(op, ch, lap))
        flags = wing_physics_flags(ch, op, None, fam)
        built = api.PROBLEM_SPECS[fam].build({}, flags, None)
        ev = built.evaluate(0.5 * (built.bounds[:, 0] + built.bounds[:, 1]))
        refused = ""
        try:
            api.PROBLEM_SPECS[BASES[("plates", "law")]].build(
                {}, {"car_objective": "laptime", "chord_trend": CHORD_TREND}, None)
        except ValueError as exc:
            refused = str(exc)
        why_not = lap_offered("dragstrip")
        lt = ev.get("lap_time_s")
        rep("B16 lap time: carsim's arena as cartrack's lap, the Corsa as its car",
            len(ts.segments) == 14 and ev["feasible"] and lt is not None and math.isfinite(lt)
            and 30.0 < lt < 120.0 and "car_track" not in api.PROBLEM_SPECS[fam].flags
            and bool(refused) and bool(why_not) and not lap_offered("open"),
            f"{len(ts.segments)} segments, {ts.length_m:.0f} m; mid-box lap {lt:.2f} s "
            f"(mu0 {built.problem.car_spec.mu0:.3f}, k {built.problem.car_spec.k_load:.4f}, "
            f"{built.problem.car_spec.mass_kg:.0f} kg); the endplate family refuses it; "
            f"dragstrip: {why_not[:40]}")
    guard("B16 lap time: carsim's arena as cartrack's lap, the Corsa as its car", b16)

    def b17():
        top_f = car_objectives("top", False, "arena")
        top_p = car_objectives("top", True, "arena")
        fl = car_objectives("flank", False, "arena")
        want_f = [k for k in CAR_OBJECTIVE_LABELS if k in carwing.CAR_OBJECTIVES]
        want_p = [k for k in CAR_OBJECTIVE_LABELS if k in endplate.ENDPLATE_OBJECTIVES]
        rep("B17 the car objective menu is WingLab's (no cz/cd); lap time on top + fences only",
            list(top_f) == want_f and list(top_p) == want_p and "laptime" not in fl
            and "laptime" not in top_p and not ({"cz", "cd"} & set(top_f))
            and "side force" in fl["downforce"],
            f"top/fences {list(top_f)}; top/plates {list(top_p)}; flank {list(fl)}")
    guard("B17 the car objective menu is WingLab's (no cz/cd); lap time on top + fences only", b17)

    # -- B18-B20: the law the car flies, and the wing it is drawn as ------------- #
    def b18():
        rec = ctx["wing8"]
        t0 = time.perf_counter()
        geom = {"x": build.slot("top").x, "h": build.slot("top").h, "role": "top"}
        law = derive_law(rec, geom)
        a = law["alpha_design_deg"]
        cz, cd = law["aerobo"]["CZ"], law["aerobo"]["CD"]
        cz_law = law["CL0"] + law["CLa"] * math.radians(a)
        cd_law = law["cd0"] + law["cd1"] * cz + law["cd2"] * cz * cz
        _c, built, labels, xs = _record_problem(rec)
        x = xs.copy()
        x[labels.index("alpha_deg")] = law["alpha_stall_deg"]
        at_pos = _evaluators(_sampling_twin(built.problem))[1](x)
        ctx["law"] = law
        rep("B18 the law IS WingLab at the design: CZ and CD exact, CZ affine in incidence",
            abs(cz_law - cz) <= 1e-12 and abs(cd_law - cd) <= 1e-12 and law["law_residual"] < 1e-6
            and at_pos["feasible"] and at_pos["CZ"] == law["CL_max"] and law["engine"] == "aerobo"
            and law["CLa"] > 0.0 and law["alpha_stall_neg_deg"] < a < law["alpha_stall_deg"],
            f"a* {a:.2f} deg: CZ {cz:.6f} (law {cz_law - cz:+.0e}), CD {cd:.6f} (law "
            f"{cd_law - cd:+.0e}); CL0 {law['CL0']:.4f}, CLa {law['CLa']:.3f}/rad over "
            f"{law['alpha_stall_neg_deg']:g}..{law['alpha_stall_deg']:g} deg (residual "
            f"{law['law_residual']:.0e}), CD fit {100 * law['cd_fit_err']:.1f} %; "
            f"{time.perf_counter() - t0:.2f} s")
    guard("B18 the law IS WingLab at the design: CZ and CD exact, CZ affine in incidence", b18)

    def b19():
        from .vehicle import TopAero
        law = ctx["law"]
        a, V = law["alpha_design_deg"], law["V_ref"]
        top = TopAero.from_aero(law, a, build.slot("top").x, build.slot("top").h)
        f_game = 0.5 * law["rho"] * V * V * top.S * top.CZ
        d_game = 0.5 * law["rho"] * V * V * top.S * top.CD
        f_ab, d_ab = law["aerobo"]["F_N"], law["aerobo"]["D_N"]
        rep("B19 the car feels WingLab's forces: the game's downforce and drag at V",
            abs(f_game - f_ab) <= 1e-9 * abs(f_ab) and abs(d_game - d_ab) <= 1e-9 * abs(d_ab),
            f"V {V:.2f} m/s, rho {law['rho']:g}: game {f_game:.6f} N / {d_game:.6f} N, "
            f"WingLab {f_ab:.6f} N / {d_ab:.6f} N")
    guard("B19 the car feels WingLab's forces: the game's downforce and drag at V", b19)

    def b20():
        from .aero.wing import WingSpec
        from .aero.airfoil import AirfoilSpec
        rec, law = ctx["wing8"], ctx["law"]
        spec, upd = to_wingspec(rec, law, None, "top", "rear-aerobo")
        back = WingSpec.from_json(json.loads(json.dumps(spec.to_json())))
        same = (json.dumps(back.design, sort_keys=True) == json.dumps(spec.design, sort_keys=True)
                and json.dumps(back.aero, sort_keys=True) == json.dumps(spec.aero, sort_keys=True))
        #  the law re-derived from the wing's own provenance, at the same slot:
        #  the same numbers (the garage's hook when a slot moves)
        again = derive_law(record_of_design(back.design, {"x": build.slot("top").x,
                                                          "h": upd["h"], "V": law["V_ref"],
                                                          "role": "top"}, deck=0.90))
        cand = dict(ctx["lib_main"]["candidates"][0])
        cand.update(source="library", conditions={"re": 1e6, "mach": 0.0})
        af = section_to_airfoilspec(cand, None)
        af2 = AirfoilSpec.from_json(json.loads(json.dumps(af.to_json())))
        rep("B20 the wing round-trips through the library with its provenance and law",
            same and back.engine == "aerobo" and back.mount == "endplate"
            and abs(back.area - law["S"]) < 1e-12 and upd["inc_deg"] == law["alpha_design_deg"]
            and upd["h"] == law["ride_h"] and again["aerobo"]["CZ"] == law["aerobo"]["CZ"]
            and af2.source == "coords" and af2.coords().shape[0] > 40 and "WingLab" in af2.origin,
            f"'{back.name}': span {back.span:.3f} m, chord {back.chord:.3f} m, taper "
            f"{back.taper:.2f}, S {back.area:.3f} m2, {back.airfoil}/{back.plate_airfoil}; slot "
            f"inc {upd['inc_deg']:.2f} deg, h {upd['h']:.3f} m; section '{af2.name}' ({af2.origin})")
    guard("B20 the wing round-trips through the library with its provenance and law", b20)

    # -- B21-B24: live XFOIL (skipped without it) ------------------------------ #
    xf = xfoil_ok()

    def _xf_row(tag, fn):
        if not xf:
            rep(tag, True, "no XFOIL here (or CARSIM_NO_XFOIL)", skip=True)
            return
        guard(tag, fn)

    def _surface_cond(surface):
        op, fam = ctx["top_op"], ctx["top_fam"]
        box = family_box(fam, wing_physics_flags({}, op, None, fam), op.size_rows or None)
        return section_conditions(op, box, surface)

    tiny = {"optimiser": "bo", "budget": 3, "refusal": "worst", "n_init": 2}

    def b21():
        t0 = time.perf_counter()
        shape = shape_kwargs(_surface_cond("plate"), "plate")
        stop = threading.Event()
        ev1, emit1 = _collect(stop_at=2, stop=stop)
        out = section_runner(shape, tiny, 0)(emit1, stop)
        r1 = out["report"]["result"]
        cont = continue_section(out["report"], out["launch"], 1)
        s2 = dict(cont["search"])
        resume = s2.pop("resume")
        ev2, emit2 = _collect()
        out2 = section_runner(cont["shape"], s2, cont["seed"], resume)(emit2, threading.Event())
        d = out2["report"]["section"]["design"]
        wu, wl = np.asarray(d["w_upper"], float), np.asarray(d["w_lower"], float)
        camber = api.section_max_camber(np.asarray(d["coords"], float))
        rep("B21 plate section: symmetric CST, Stop at 2, Keep going +1 resumes at 3",
            shape["symmetric"] and _ns(ev1) == [1, 2] and r1.get("partial")
            and r1.get("stop_reason") == STOP_PLAYER and _ns(ev2) == [3]
            and out2["report"]["result"].get("resumed") == 2 and np.allclose(wl, -wu)
            and camber < 1e-4,
            f"Re {shape['re']:.3g}, cl {shape['cl_design']:g}; n {_ns(ev1)} then {_ns(ev2)}, "
            f"w_lower == -w_upper, max camber {camber:.1e}; {time.perf_counter() - t0:.1f} s")
    _xf_row("B21 plate section: symmetric CST, Stop at 2, Keep going +1 resumes at 3", b21)

    def b22():
        t0 = time.perf_counter()
        shape = shape_kwargs(_surface_cond("main"), "main")
        events, emit = _collect()
        out = section_runner(shape, tiny, 0)(emit, threading.Event())
        sec = (out["report"] or {}).get("section") or {}
        coords = (sec.get("design") or {}).get("coords") or []
        ctx["main3"] = out["report"]
        rep("B22 main section: a 3-evaluation live-XFOIL search returns a section",
            _ns(events) == [1, 2, 3] and len(coords) == 160 and not shape["symmetric"]
            and "cdcr" not in shape["score_weights"],
            f"Re {shape['re']:.3g}, cl {shape['cl_design']:g}: {len(coords)} coords, "
            f"t/c {float(sec['design'].get('tc') or 0.0):.3f}; {time.perf_counter() - t0:.1f} s")
    _xf_row("B22 main section: a 3-evaluation live-XFOIL search returns a section", b22)

    def b23():
        t0 = time.perf_counter()
        sc = score_runner(ctx["main3"], WING_WEIGHTS, None)(_collect()[1], threading.Event())
        rep("B23 the post-run score: seed vs optimised on the surface's weights",
            isinstance(sc, dict) and SCORE_KEYS <= set(sc),
            f"keys {sorted(sc)}; {time.perf_counter() - t0:.1f} s")
    _xf_row("B23 the post-run score: seed vs optimised on the surface's weights", b23)

    def b24():
        t0 = time.perf_counter()
        cond = _surface_cond("main")
        events, emit = _collect()
        out = screen_runner("main", WING_WEIGHTS, cond, SCREEN_GATES, None, "mission")(
            emit, threading.Event())
        sw = [p for k, p in events if k == "sweep"]
        rp = out["report"]
        phases = [p["text"] for k, p in events if k == "phase"]
        short = rp.get("shortlist") or {}
        #  cold: one sweep event per shortlisted section; warm (a second
        #  visit to the point): none, and the phase says the cache answered
        swept_ok = ((len(sw) >= SHORTLIST and "shortlist sweep" in phases
                     and [p["i"] for p in sw] == list(range(1, len(sw) + 1)))
                    or (not sw and any("cache" in s for s in phases)))
        rep("B24 own-Re screen: the library pass chooses, XFOIL sweeps the shortlist",
            out["re_source"] == "mission" and short.get("source") == "library"
            and int(short.get("n") or 0) >= SHORTLIST and swept_ok
            and len(rp["ranked"]) == TOP_N and all("rank_library" in r for r in rp["ranked"]),
            f"Re {cond['re']:.3g}: shortlist {short.get('n')}, "
            f"{f'{len(sw)} swept live' if sw else 'answered from the point cache'}, leader "
            f"{rp['ranked'][0]['name']} (library rank {rp['ranked'][0].get('rank_library')}); "
            f"{time.perf_counter() - t0:.1f} s")
    _xf_row("B24 own-Re screen: the library pass chooses, XFOIL sweeps the shortlist", b24)

    # -- B27: task 41, the car's limits (fast, no XFOIL) -------------------------- #
    def b27():
        #  every per-car number is drive/bodies.py's, for each of the five cars
        #  at its default slots: the flank's span row IS its limit at h (the
        #  lower tip at its own ground clearance) cut to AR >= 3, the top's
        #  span row 1.2 x its width, its deck and ride band the slot's
        #  (bodies.top_h_band), and Unlimited opens only the ceilings, 3x
        from types import SimpleNamespace as NS
        bad, rows = [], []
        for ck in ("corsa", "mx5", "540i", "express", "bus"):
            d = _bodies.slot_defaults(ck)
            (fx, fh), (tx, th, _ti) = d["flank"], d["top"]
            bld = NS(slot=lambda k, _s={"left": NS(x=fx, h=fh), "top": NS(x=tx, h=th)}: _s[k])
            of = operating_point("left", "flank", bld, mission, car=ck)
            ot = operating_point("top", "top", bld, mission, car=ck)
            ou = operating_point("left", "flank", bld, mission, car=ck, unlimited=True)
            b = _bodies.body(ck)
            lim_f, lim_t = 2.0 * (fh - b.ground), 1.2 * b.width
            b_f, s_f = of.size_rows["b_m"], of.size_rows["S_m2"]
            ok_ = (abs(of.limit - lim_f) < 1e-12 and abs(b_f[1] - lim_f) < 1e-6
                   and b_f[1] * b_f[1] / s_f[1] >= AR_MIN - 1e-6
                   and b_f[0] * b_f[0] / (0.5 * (s_f[0] + s_f[1])) >= AR_MIN - 1e-4
                   and abs(ot.limit - lim_t) < 1e-12 and abs(ot.size_rows["b_m"][1] - lim_t) < 1e-6
                   and tuple(ot.ride_band) == tuple(_bodies.top_h_band(ck, tx))
                   and abs(ot.deck - car_deck(ck)(tx)) < 1e-12
                   and ou.size_rows == of.size_rows
                   and abs(ou.size_caps["b_m"] - _bodies.UNLIMITED_FACTOR * lim_f) < 1e-9)
            if not ok_:
                bad.append(ck)
            rows.append(f"{ck} flank {lim_f:.2f}@{fh:.2f} top {lim_t:.3f} ride "
                        f"{ot.ride_band[0]:.2f}-{ot.ride_band[1]:.2f}")
        corsa = operating_point("left", "flank", build, mission)
        rep("B27 task 41: each car's span rows, deck and ride band are drive/bodies.py's; the "
            "Corsa's flank row is 1.50 m at h 0.90",
            not bad and corsa.car == "corsa" and corsa.size_rows["b_m"][1] == 1.5,
            ("; ".join(rows)) if not bad else f"wrong on {bad}")
    guard("B27 task 41: each car's span rows, deck and ride band are drive/bodies.py's; the "
          "Corsa's flank row is 1.50 m at h 0.90", b27)

    # -- B28-B32: "Carried by" (the owner, 2026-09-25: "Carried by endplate still
    #    produces inboard pylons ... root blend and leaning at ... the tip device
    #    and follows the chord of wing") -------------------------------------------- #
    top_op = operating_point("top", "top", build, mission, deck_z=deck)
    fl_op = operating_point("left", "flank", build, mission)

    def b28():
        #  every name registered before is byte-identical (AeroBO's cache keys
        #  on it): the defaults through `family_params`, the legacy fences as
        #  a record's `carsim_family` re-reads them, the lap family too
        lap = lap_params(mission, build, None, "top")
        legacy = {
            "top default": family_params(top_op),
            "flank default": family_params(fl_op),
            "top endplates, straight": family_params(top_op, {"chord_law": False}),
            "top fences": FamilyParams("top", False, True, tuple(top_op.ride_band), None),
            "flank fences, straight": FamilyParams("flank", False, False, tuple(fl_op.ride_band),
                                                   None),
            "top fences, lap": FamilyParams("top", False, True, tuple(top_op.ride_band), None,
                                            lap=lap)}
        names = {k: family_name(p) for k, p in legacy.items()}
        #  ...and a record written before the two fields existed
        old = {k: v for k, v in _family_json(legacy["top fences"]).items()
               if k not in ("free", "pylons")}
        reread = family_name(FamilyParams.from_json(old))
        #  the fixtures' records were flown under names the bridge still derives
        fx_ok = []
        for stem, kind, _a in FIXTURES:
            p = FIXTURE_DIR / f"{stem}.json"
            if kind == "wing" and p.is_file():
                rec = json.loads(p.read_text())["result"]["record"]
                fx_ok.append(family_of(rec) == rec["config"]["problem_name"])
        #  the designed plate's eight names are AeroBO's, and the pylons' is new
        eight = [api.plate_freedom_name(PLATE_BASE, set(f)) + law
                 for f in ((), ("cant",), ("blend",), ("blend", "cant")) for law in ("", LAW_SUFFIX)]
        pyl = family_name(family_params(top_op, {"plates": False}))
        refused = []
        for bad in (dict(plates=False, free=("cant",)), dict(plates=True, pylons=True),
                    dict(plates=True, free=("toe",))):
            try:
                FamilyParams("top", chord_law=True, ride_band=(1.0, 1.8), deck=0.9, **bad)
            except ValueError:
                refused.append(True)
        rep("B28 every family name registered before is byte-identical; the eight designed-plate "
            "names are WingLab's; the pylons' family is a new name",
            names == LEGACY_NAMES and reread == LEGACY_NAMES["top fences"] and fx_ok
            and all(fx_ok) and all(n in api.PROBLEM_SPECS for n in eight) and len(set(eight)) == 8
            and pyl not in LEGACY_NAMES.values() and len(refused) == 3,
            f"{len(names)} legacy names equal the untouched copy's, {sum(fx_ok)}/{len(fx_ok)} wing "
            f"fixtures re-derive their own; pylons -> {pyl.split(' #')[-1]}")
    guard("B28 every family name registered before is byte-identical", b28)

    def b29():
        #  the card's answers -> AeroBO's family, the flags (sent / not sent,
        #  nothing dropped or refused), the plate rows the box searches, the pins
        bad = []
        for label, ch, want in MOUNT_CONFIGS:
            fp = family_params(top_op, ch)
            fam = ensure_family(fp)
            flags = wing_physics_flags(ch, top_op, None, fam)
            _kept, dropped = api.sanitise_flags(fam, flags)
            try:
                api.check_flags(fam, flags)
                refused = ""
            except KeyError as exc:
                refused = str(exc)
            labels = set(family_box(fam, flags, top_op.size_rows))
            why = []
            if fp.base != want["base"] or f" · {want['base']} #" not in fam:
                why.append(f"family {fp.base!r}")
            if flags.get("mount") != MOUNT_FLAG or any(flags.get(k) != v
                                                       for k, v in want["sent"].items()):
                why.append(f"sent {({k: flags.get(k) for k in want['sent']})}")
            if any(k in flags for k in want["absent"]):
                why.append(f"sent {[k for k in want['absent'] if k in flags]}")
            if dropped or refused:
                why.append(f"dropped {dropped} refused {refused!r}")
            rows = {r for r in ("endplate_cant_deg", "endplate_blend_frac") if r in labels}
            if rows != set(want["rows"]):
                why.append(f"rows {sorted(rows)}")
            if tip_pins(ch) != want["pins"] or (fp.pylons != (not fp.plates)) \
                    or (fp.deck != top_op.deck):
                why.append(f"pins {tip_pins(ch)} pylons {fp.pylons} deck {fp.deck}")
            if why:
                bad.append(f"{label}: {'; '.join(why)}")
        rep("B29 Carried by: each configuration's family, flags (sent / not sent, none dropped or "
            "refused), searched plate rows and pins",
            not bad, "; ".join(bad) if bad else
            f"{len(MOUNT_CONFIGS)} configurations: endplates x stated / 0.3 + 70 / lean / blend / "
            f"both (x2 laws), pylons x none / vertical (chord follows or held) / canted 70 / blended (x2 laws)")
    guard("B29 Carried by: each configuration's family, flags, rows and pins", b29)

    def _centre_eval(op, ch):
        rec, x, labels = _mount_record(op, ch)
        cfgd = rec["config"]
        built = api.PROBLEM_SPECS[cfgd["problem_name"]].build({}, cfgd["flags"], op.size_rows)
        return built, built.evaluate(np.asarray(x)), x, labels

    def _asked(op, ch, want, built, ev, x, labels) -> list:
        """What the breakdown says against what the card asked (B30, B31)."""
        why = []
        q = built.problem
        v = dict(zip(labels, (float(u) for u in x)))
        if not (ev.get("feasible") and math.isfinite(float(ev.get("CZ") or math.nan))
                and math.isfinite(float(ev.get("CD") or math.nan))):
            return [f"not flyable at the box centre ({ev.get('reason') or 'a margin'})"]
        if not ch.get("plates", True):
            tip = tip_device(ch)
            ride = v["ride_height_m"]
            if not (ev.get("mount_kind") == "pylon" and ev.get("mount_side") == "pressure"
                    and abs(float(ev["y_station_frac"]) - carmount.INBOARD_STATION_FRAC) < 1e-12
                    and abs(float(ev["deck_height_m"]) - float(op.deck)) < 1e-12
                    and abs(float(ev["pylon_length_m"]) - (ride - float(op.deck))) < 1e-9):
                why.append(f"mount {ev.get('mount_kind')} at {ev.get('y_station_frac')}, pylon "
                           f"{ev.get('pylon_length_m')} m (ride {ride:.3f} - deck {op.deck})")
            cant = float(want["sent"].get("endplate_cant_deg", CANT_UPRIGHT))
            blend = float(want["sent"].get("blend_frac", 0.0))
            if not (float(q.endplate_cant_deg) == cant and float(q.blend_frac) == blend
                    and bool(q.endplate_chord_follows) == (tip != "none" and tip_chord_follows(ch))
                    and (tip != "none" or float(ev["endplate_h_m"]) == 0.0)
                    and (float(ev["endplate_projection_m"]) > 0.0) == (tip in ("canted", "blended"))):
                why.append(f"device {tip}: cant {q.endplate_cant_deg}, blend {q.blend_frac}, "
                           f"follows {q.endplate_chord_follows}, h {ev.get('endplate_h_m')}")
            return why
        if ev.get("mount_kind") is not None or ev.get("pylon_length_m") is not None \
                or ev.get("mount") != MOUNT_FLAG:
            why.append(f"a pylon under the endplates ({ev.get('mount')}, {ev.get('mount_kind')})")
        cant = v.get("endplate_cant_deg", float(want["sent"].get("endplate_cant_deg", CANT_UPRIGHT)))
        blend = v.get("endplate_blend_frac", float(want["sent"].get("blend_frac", 0.0)))
        shape = "spiral" if ("blend_shape" in want["sent"]) else "arc"
        if not (abs(float(ev["endplate_cant_deg"]) - cant) < 1e-12
                and abs(float(ev["endplate_blend_frac"]) - blend) < 1e-12
                and ev.get("endplate_blend_shape") == shape
                and bool(ev.get("endplate_cant_searched")) == ("endplate_cant_deg" in v)
                and bool(ev.get("endplate_blend_searched")) == ("endplate_blend_frac" in v)):
            why.append(f"plate cant {ev.get('endplate_cant_deg')} (asked {cant}), blend "
                       f"{ev.get('endplate_blend_frac')} (asked {blend}), "
                       f"{ev.get('endplate_blend_shape')}")
        return why

    def b30():
        #  ONE REAL AeroBO evaluation per configuration, at the middle of its
        #  box on the top slot: finite, and the breakdown says what was asked
        bad, got = [], []
        for label, ch, want in MOUNT_CONFIGS:
            built, ev, x, labels = _centre_eval(top_op, ch)
            why = _asked(top_op, ch, want, built, ev, x, labels)
            if why:
                bad.append(f"{label}: {'; '.join(why)}")
            else:
                got.append(f"{float(ev['CZ']):.3f}")
        rep("B30 one real WingLab evaluation per configuration: finite, and the breakdown says what "
            "was asked (the pylon pair at 0.35 of the semi-span, ride - deck long; no pylon under "
            "the endplates; lean, blend and chord as sent; no plate under 'none')",
            not bad, "; ".join(bad) if bad else f"top slot CZ {', '.join(got)}")
    guard("B30 one real WingLab evaluation per configuration", b30)

    def b31():
        #  the FLANK's pylons reach the car's side (the standoff is ride - deck,
        #  both 100 m out), and the lap family is the top wing on the pylons
        bad = []
        for label, ch, want in MOUNT_CONFIGS:
            if ch.get("plates", True):
                continue
            built, ev, x, labels = _centre_eval(fl_op, ch)
            why = _asked(fl_op, ch, want, built, ev, x, labels)
            if why:
                bad.append(f"flank {label}: {'; '.join(why)}")
        ch = {"plates": False, "objective": "laptime", "tip": "canted", "tip_cant_deg": 70.0}
        lap = lap_params(mission, build, None, "top")
        fp = family_params(top_op, ch, lap)
        fam = ensure_family(fp)
        flags = wing_physics_flags(ch, top_op, None, fam)
        api.check_flags(fam, flags)
        api.check_wing_objective(flags)
        built = api.PROBLEM_SPECS[fam].build({}, flags, top_op.size_rows)
        labels = list(built.param_labels)
        x = 0.5 * (built.bounds[:, 0] + built.bounds[:, 1])
        ev = built.evaluate(x)
        lt = ev.get("lap_time_s")
        want = dict(next(w_ for l_, _c, w_ in MOUNT_CONFIGS if l_ == "pylons, canted 70"))
        lap_ok = (fp.pylons and fp.lap == lap and flags.get("car_objective") == "laptime"
                  and not _asked(top_op, ch, want, built, ev, x, labels)
                  and lt is not None and 30.0 < float(lt) < 120.0
                  and "laptime" in car_objectives("top", False, "arena")
                  and "laptime" not in car_objectives("top", True, "arena"))
        if not lap_ok:
            bad.append(f"lap: {fam}, lap {lt}, {_asked(top_op, ch, want, built, ev, x, labels)}")
        rep("B31 the flank's pylons reach the car's side; the lap family (top wing on the pylons, "
            "canted 70) builds and times a lap",
            not bad, "; ".join(bad) if bad else
            f"flank pylons {fl_op.ride_band[0] - fl_op.deck:.2f}-"
            f"{fl_op.ride_band[1] - fl_op.deck:.2f} m "
            f"to the car's side; mid-box lap {float(lt):.2f} s on {fam.split(' · ')[1]}, canted 70")
    guard("B31 the flank's pylons reach the car's side; the lap family builds and times a lap", b31)

    def b32():
        #  the fitted WingSpec per mount (decision 6), and the records: the
        #  card re-opens on them, a legacy fence record re-fits as it flew,
        #  a moved slot moves the pylons' deck too
        from .aero.wing import WingSpec
        bad = []
        specs = {}
        for label, ch, want in MOUNT_CONFIGS:
            rec, x, labels = _mount_record(top_op, ch, stale_pin=0.123)
            v = dict(zip(labels, (float(u) for u in x)))
            law = {"span_flown": v["b_m"], "S": v["S_m2"], "alpha_design_deg": v["alpha_deg"],
                   "ride_h": v["ride_height_m"]}
            spec, _upd = to_wingspec(rec, law, None, "top", "check")
            back = WingSpec.from_json(json.loads(json.dumps(spec.to_json())))
            pyl = not ch.get("plates", True)
            sent = want["sent"]
            exp = dict(mount="pylon" if pyl else "endplate",
                       pylon_frac=carmount.INBOARD_STATION_FRAC if pyl else WingSpec.pylon_frac,
                       plate_cant_deg=v.get("endplate_cant_deg",
                                            float(sent.get("endplate_cant_deg", CANT_UPRIGHT))),
                       plate_blend=v.get("endplate_blend_frac", float(sent.get("blend_frac", 0.0))),
                       plate_shape=str(sent.get("blend_shape") or "arc"),
                       plate_chord_ratio=1.0 if pyl else v["endplate_chord_ratio"],
                       plate_chord_follows=bool(sent.get("endplate_chord_follows")),
                       plate_h=0.0 if want["pins"] else v["endplate_h_m"])
            got = {k: getattr(back, k) for k in exp}
            if any((abs(float(got[k]) - float(exp[k])) > 1e-12) if isinstance(exp[k], float)
                   else got[k] != exp[k] for k in exp):
                bad.append(f"{label}: {got} != {exp}")
            #  the card re-opens on the record's own answers; a number the
            #  record SEARCHED keeps the one the player had typed (55, 0.2)
            typed = {"cant_deg": 55.0, "blend_frac": 0.2}
            card = choices_of_record(rec, typed)
            want_card = {k: ch[k] for k in ("plates", "chord_law", "tip", "tip_cant_deg") if k in ch}
            if ch.get("plates", True):
                for mode, k, dflt in (("blend", "blend_frac", 0.0), ("cant", "cant_deg", CANT_UPRIGHT)):
                    free = ch.get(mode) == "free"
                    want_card[mode] = "free" if free else "stated"
                    want_card[k] = typed[k] if free else float(ch.get(k, dflt))
            if any(card.get(k) != v for k, v in want_card.items()):
                bad.append(f"{label}: re-opens {({k: card.get(k) for k in want_card})}, "
                           f"flown {want_card}")
            specs[label] = (rec, spec)
        #  a real law on the pinned record: AeroBO's evaluator reads the PIN
        rec_n, spec_n = specs["pylons, none"]
        law_n = derive_law(rec_n, {"x": -0.90, "h": 1.55, "role": "top"})
        if not (law_n["mount"] == "pylon" and law_n["plate_h_flown"] == 0.0):
            bad.append(f"law of 'none': mount {law_n['mount']}, plate {law_n['plate_h_flown']}")
        #  ...and the WING's span: the breakdown's b_m is already the wing left
        #  once a leaning / blended plate took its reach out of the overall
        #  width, so the law's span must close AR = b^2 / S (the verifier found
        #  it short by twice the plate's reach, 2026-09-25)
        rec_l, _spec_l = specs["endplates, blend 0.3 + lean 70"]
        law_l = derive_law(rec_l, {"x": -0.90, "h": 1.55, "role": "top"})
        ar_l = float(law_l["span_flown"]) ** 2 / float(law_l["S"])
        if abs(ar_l - float(law_l["AR"])) > 1e-6 * float(law_l["AR"]):
            bad.append(f"lean 70 + blend 0.3: span {law_l['span_flown']:.4f} m gives b^2/S "
                       f"{ar_l:.4f} against WingLab's AR {float(law_l['AR']):.4f}")
        #  a LEGACY fence record (no pylon flags: plain plates at the tips)
        old_fp = FamilyParams("top", False, True, tuple(top_op.ride_band), None)
        old = {"config": {"flags": {"chord_trend": CHORD_TREND, "mount": "tips", "V": 30.0}},
               "carsim_family": {k: v for k, v in _family_json(old_fp).items()
                                 if k not in ("free", "pylons")}}
        card = choices_of_record(old)
        if not (record_mount(old["config"]["flags"]) == "endplate" and card["plates"] is False
                and card["tip"] == "vertical" and family_of(old) == LEGACY_NAMES["top fences"]):
            bad.append(f"legacy fence: {record_mount(old['config']['flags'])}, card {card}")
        #  a moved slot: the pylon wing's family follows the new deck
        d_moved = record_of_design(spec_n.design, {"x": -0.95, "h": 1.60, "V": 30.0}, deck=1.00)
        fp_moved = FamilyParams.from_json(d_moved["carsim_family"])
        if not (fp_moved.pylons and fp_moved.deck == 1.00
                and abs(fp_moved.ride_band[0] - (1.00 + TOP_DECK_CLEAR_M)) < 1e-12):
            bad.append(f"moved: {fp_moved}")
        rep("B32 the fitted wing is drawn as flown -- mount, station, lean, blend, chord, height "
            "(0 under 'none', pinned rows read); the card re-opens on a record; a legacy fence "
            "record re-fits as it flew; a moved slot moves the pylons' deck",
            not bad, "; ".join(bad) if bad else
            f"{len(specs)} configurations fitted and round-tripped; 'none' law plate "
            f"{law_n['plate_h_flown']:g} m; legacy fence -> {card['tip']} on the card, "
            f"{record_mount(old['config']['flags'])} on the car")
    guard("B32 the fitted wing is drawn as flown; records re-open the card", b32)

    # -- B33: the owner, 2026-09-27: "Root blend 1 for wing design doesn't
    #    produce a solution" -- the plates' outboard reach comes out of the span,
    #    so the area row's floor comes down to what the wing left carries ------- #
    def b33():
        #  the owner's own run: a left flank 1.08 m wide, the area row from
        #  0.12 m2, blend 1 stated -> 0 of 50 feasible and no seed
        ch1 = {"plates": True, "blend": "stated", "blend_frac": 1.0, "cant": "stated",
               "cant_deg": CANT_UPRIGHT, "objective": "downforce"}
        owner = dataclasses.replace(fl_op, size_rows={"b_m": [0.873613, 1.08],
                                                      "S_m2": [0.12, 0.3888]})
        fam = ensure_family(family_params(owner, ch1))
        flags = wing_physics_flags(ch1, owner, {}, family=fam)
        rows = copy.deepcopy(owner.size_rows)
        before = plate_seed(fam, flags, rows, None)
        floor = plate_area_floor(owner, ch1, rows["b_m"][1])
        rows["S_m2"] = [floor, rows["S_m2"][1]]
        seed = plate_seed(fam, flags, rows, None)
        fit = plate_fit_problem(owner, ch1, rows["b_m"][1], 0.6)
        #  an upright crease moves nothing; a flank with no wing left is told so
        flat = plate_area_floor(owner, dict(ch1, blend_frac=0.0), 1.08)
        narrow = dataclasses.replace(fl_op, size_rows={"b_m": [0.433013, 0.5],
                                                       "S_m2": [0.041667, 0.083333]})
        refused = plate_fit_problem(narrow, ch1, 0.5, 0.6)
        #  a released / typed area row from 0.10 m2 on a 1.02 m flank: the
        #  0.52 m wing left carries 0.09 m2 at most -- said, not flown
        area = plate_fit_problem(owner, ch1, 1.02, 0.6, None, 0.10)
        good = (before is None and floor is not None and floor < 0.12 and seed is not None
                and fit is None and flat is None and bool(refused) and "no wing" in refused
                and bool(area) and "area row" in area)
        rep("B33 root blend 1 on the owner's 1.08 m flank: the area floor comes down to the wing "
            "left, a seed that flies is found; an upright crease moves nothing; a 0.50 m flank "
            "and an area row the wing left cannot carry are refused in words", good,
            f"seed before: {'none' if before is None else 'found'}; area floor 0.12 -> {floor} m2; "
            f"seed after: {'found' if seed is not None else 'NONE'}; fit {fit!r}; upright floor "
            f"{flat}; 0.50 m flank: {refused!r}; area row from 0.10 on 1.02 m: {area!r}")
    guard("B33 root blend 1 flies: the area floor follows the plates' reach", b33)

    # -- B25: the captured fixtures ----------------------------------------------- #
    def b25():
        bad, size, fx_of = {}, 0, {}
        for stem, kind, args in FIXTURES:
            p = FIXTURE_DIR / f"{stem}.json"
            if not p.is_file():
                bad[stem] = ["missing (python3 -m drive.aerobo_bridge --capture)"]
                continue
            fx = json.loads(p.read_text())
            probs = fixture_problems(fx)
            if fx.get("kind") != kind or any((fx.get("meta") or {}).get("args", {}).get(k) != v
                                             for k, v in args.items()):
                probs.append("kind / args do not match FIXTURES")
            if probs:
                bad[stem] = probs
            size += p.stat().st_size
            fx_of[stem] = fx
        if bad:
            rep("B25 the captured fixtures load, match the format and fit the checks", False,
                "; ".join(f"{k}: {', '.join(v)}" for k, v in bad.items()))
            return

        def evals(fx, part=None):
            evs = (fx[part] if part else fx)["events"]
            return [p["n"] for k, p in evs if k == "eval"]
        sec, wt = fx_of["section-main"], fx_of["wing-top"]
        sec_st = sec["stopped"]["result"]["report"]["result"]
        wt_st = wt["stopped"]["result"]["record"]
        cont = fx_of["wing-top-continued"]["result"]["record"]
        sweeps = [p for k, p in fx_of["screen-main-mission"]["events"] if k == "sweep"]
        plate = fx_of["section-plate"]["result"]["report"]["section"]["design"]
        fits = (evals(sec) == list(range(1, CAPTURE_SECTION_BUDGET + 1))
                and sec["meta"]["args"]["n_init"] == 4
                and evals(sec, "stopped") == list(range(1, CAPTURE_SECTION_STOP + 1))
                and sec_st.get("partial") and sec_st.get("stop_reason") == STOP_PLAYER
                and evals(fx_of["section-main-continued"])[0] == CAPTURE_SECTION_BUDGET + 1
                and evals(wt, "stopped") == list(range(1, CAPTURE_WING_STOP + 1))
                and wt_st.get("partial") and wt_st.get("stop_reason") == STOP_PLAYER
                and evals(fx_of["wing-top-continued"])[0] == CAPTURE_WING_STOP + 1
                and cont.get("resumed") == CAPTURE_WING_STOP
                and cont.get("n_evals") == CAPTURE_WING_STOP + CAPTURE_WING_MORE
                and len(sweeps) >= SHORTLIST
                and np.allclose(plate["w_lower"], -np.asarray(plate["w_upper"]))
                and fx_of["wing-top-laptime"]["result"]["record"]["carsim_family"]["lap"])
        stamp = fx_of["wing-top"]["meta"]["captured"]
        rep("B25 the captured fixtures load, match the format and fit the checks",
            fits and size <= FIXTURE_MAX_BYTES,
            f"{len(fx_of)} fixtures, {size / 1e3:.0f} kB (<= {FIXTURE_MAX_BYTES / 1e3:.0f}), "
            f"captured {stamp} on aerobo {AEROBO_COMMIT}; section Stop at "
            f"{CAPTURE_SECTION_STOP}, wing Stop at {CAPTURE_WING_STOP} then +{CAPTURE_WING_MORE}, "
            f"{len(sweeps)} own-Re sweeps")
    guard("B25 the captured fixtures load, match the format and fit the checks", b25)

    # -- END OF ROWS --
    def b26():
        tree1 = _tree_state()
        moved = sorted(k for k in set(tree0) | set(tree1) if tree0.get(k) != tree1.get(k))
        rep("B26 nothing under aerobo/ but results/ changed during the check", not moved,
            f"{len(tree1)} files unchanged" if not moved else f"changed: {moved[:5]}")
    guard("B26 nothing under aerobo/ but results/ changed during the check", b26)

    wall = time.perf_counter() - t_start
    n_run = counts["pass"] + counts["fail"]
    if verbose:
        print(f"  {'ALL PASS' if ok else 'FAILURES ABOVE'}: {counts['pass']}/{n_run} checks"
              + (f", {counts['skip']} skipped (no XFOIL)" if counts["skip"] else "")
              + f" in {wall:.1f} s")
    return ok


if __name__ == "__main__":
    if "--capture" in sys.argv:
        #  python3 -m drive.aerobo_bridge --capture [dir] [--deck m]
        i = sys.argv.index("--capture")
        nxt = sys.argv[i + 1] if len(sys.argv) > i + 1 else ""
        deck = float(sys.argv[sys.argv.index("--deck") + 1]) if "--deck" in sys.argv else None
        capture(nxt if nxt and not nxt.startswith("--") else FIXTURE_DIR, deck=deck)
        sys.exit(0)
    sys.exit(0 if self_check(slow="--slow" in sys.argv) else 1)
