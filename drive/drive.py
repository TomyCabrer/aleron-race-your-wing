"""The harness: fixed-timestep main loop, scripted virtual drivers and the CLI.

Why this file is shaped the way it is
-------------------------------------
There are two loops, and they are deliberately NOT the same loop with a flag.

`run_interactive()` owns a wall clock, an accumulator and pygame. It is allowed
to drop simulated time when the machine cannot keep up, because a real-time sim
that spirals is worse than one that stutters.

`run_headless()` owns none of those things. It is a `for` loop over
`int(round(T/dt))` calls to `step_physics`, with no clock read, no accumulator
and no pygame call anywhere in its call tree. That is the ONLY reason V20
(byte-identical CSVs from two runs of the same script) is worth anything: if a
wall clock could reach the physics path through a shared code path, the
determinism test would be measuring nothing. Every acceptance number in this
project is produced through the headless loop.

The accumulator is the part that goes wrong, and it goes wrong in three ways:

  * pygame's first frame on macOS is 0.5-2 s. Fed to the accumulator that is
    500-2000 physics steps before the first frame is ever drawn, and the car is
    somewhere else. Hence `_first_frame`, forced after init, after any reset and
    after unpause: dt_wall = dt, acc = 0, exactly one step (V24).
  * a long stall spirals. `MAX_SUBSTEPS = 40` bails and the remainder is
    DROPPED, never carried (V23).
  * slow-motion. `time_scale` multiplies dt_wall. It must never multiply
    DT_PHYS: changing the step size at runtime changes the integration accuracy
    and can destabilise the implicit wheel ODE mid-corner.

Surface friction is looked up PER WHEEL at 200 Hz, four `track.project()` calls
per update. A single car-centre lookup is 4x cheaper and silently deletes
split-mu, which is the most interesting thing a wet surface does to a FWD car
under braking and the entire reason DAMP_T5_EXIT exists (V27).

Lap detection uses the wrap window (`s_prev > L-20 and s_now < 20`), a 3 s
lockout and a sub-step interpolated crossing fraction. "Did s decrease" alone
double-fires on lateral noise near the line; whole-step timing cannot meet
V25's 0.002 s tolerance because one step is 0.001 s.

`input.py` and `render.py` are imported LAZILY inside the functions that need
them, so that the whole headless path -- every script, every acceptance number
-- runs even if neither exists yet. `ScriptedInput` is duplicated here as a
six-line fallback for the same reason; see DEVIATIONS.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from math import atan, atan2, cos, degrees, hypot, radians, sin, sqrt

import numpy as np

import qss
from corsa_c import CorsaC, G, RHO

from . import telemetry as tlm
from . import track as trk
from .vehicle import Controls, Vehicle, VehicleConfig, wheel_positions

# ==================================================================== #
#  HARNESS CONSTANTS -- specs/harness.txt, CONTRACT section 8          #
# ==================================================================== #
DT_PHYS = 0.001              # s   derived: resolves sigma/V = 13 ms relaxation and
                             #     the wheel lock/unlock transient. RTF measured
                             #     below; there is headroom for 0.001.
DT_PHYS_FALLBACK = 0.002     # s   500 Hz fallback selected by --dt if V21 fails on
                             #     slower hardware. Re-validate V5-V12 if used.
FPS = 60                     # Hz  published requirement
MAX_SUBSTEPS = 40            # steps/frame  derived: a 60 fps frame is 16.7 steps,
                             #     so 40 is 2.4x catch-up before time is dropped
MAX_FRAME_DT = 0.10          # s   derived: caps a window drag / GC pause / debugger
TELEM_HZ = 100               # Hz  derived: >30 samples across a 0.36 s shift
SURFACE_LOOKUP_HZ = 200      # Hz  derived: 0.17 m of travel per update at 33 m/s
SKID_HZ = 100                # Hz  spec eq.16
SLOWMO_SCALE = 0.25          # -   the '[' key; multiplies dt_wall, NEVER DT_PHYS

LAP_WRAP_WINDOW = 20.0       # m   the wrap case of eq.17
LAP_LOCKOUT_S = 3.0          # s   minimum lap; kills the double-fire
DELTA_LOCK_DEG = 32.625      # deg road wheel = 522 deg at the wheel / 16.0
MU_WET_SCALE = 0.632183908   # published, qss.sweep: 0.55/0.87
AY_MAX_DRY = 8.4608          # m/s^2  derived: qss.max_ay(V) with k = 0

# Script tuning. These are DRIVER gains, never physics: nothing here can change
# a force. They only decide how well the virtual driver uses the car.
KP_N = 0.06                  # rad per m of lateral error (spec's stated gain)
KD_PSI = 0.55                # rad per rad of heading error. est, band 0.3-0.9.
                             #     Without it the pure-P follower limit-cycles
                             #     on the skidpad at >18 m/s (measured).
KI_N = 0.020                 # rad per m.s. est. A pure-P follower has to hold a
                             #     STEADY offset to generate the understeer steer
                             #     angle: at R=50, 20 m/s that is n = -3.0 m, so
                             #     the car is really running R=53 and the
                             #     bisection converges on the controller, not the
                             #     car (measured: 20.01 m/s, ay 0.71 g). With the
                             #     integrator the offset goes to zero and ay
                             #     reaches the envelope.
I_N_LIM = 0.30               # rad  anti-windup on that integrator (17 deg)
KP_V = 0.55                  # pedal per (m/s) of speed error. est.
KI_V = 0.30                  # pedal per (m/s . s). est; anti-windup at +/-1.
N_DEPART = 3.0               # m   |n| beyond which a skidpad attempt has failed.
                             #     Half the 10 m skidpad half-width, so departure
                             #     is unambiguous well before the car runs out of
                             #     tarmac.
BETA_SPIN_DEG = 35.0         # deg body slip that counts as a spin

DEVIATIONS = (
    "1. ScriptedInput is defined HERE as well as (eventually) in input.py. The "
    "contract puts it in input.py, but every acceptance number in this file is "
    "produced by a scripted headless run, and those must not stop working "
    "because a parallel module is mid-write. drive.py prefers "
    "drive.input.ScriptedInput when it imports and falls back to the local one, "
    "which is six lines and reads no clock and no pygame.",
    "2. dropped_s counts BOTH the time discarded by the MAX_SUBSTEPS bail and "
    "the time discarded by the MAX_FRAME_DT clamp. Eq.1 clamps dt_wall to 0.10 s "
    "before the accumulator sees it, so on V23's 1.5 s stall the accumulator "
    "itself can only ever discard 0.06 s. V23 asks for ~1.46 s, which is "
    "1.5 - 40*DT_PHYS: the only accounting that produces it is 'simulated time "
    "that arrived and was not run', which is what dropped_s now means.",
    "3. The first frame forces acc = dt directly rather than acc += dt*time_scale. "
    "In slow motion the latter gives acc = 0.00025 s and ZERO physics steps on "
    "the first frame, which fails V24's 'exactly 1 step' for a reason that has "
    "nothing to do with the guard it is testing.",
    "4. The auto-clutch / shift state machine of spec eq.7 is NOT implemented "
    "here. powertrain.update_shift already owns it (declutch/gate/engage plus the "
    "launch and anti-stall assist) and vehicle.step drives it from "
    "Controls.auto_gearbox. A second copy in the harness would be a second "
    "answer to the same question.",
    "5. V25 is measured kinematically. A constant-30 m/s lap of CIRCUIT_ARENA is "
    "dynamically impossible -- T2 is R=30, whose grip limit is 15.93 m/s -- so no "
    "vehicle run can produce the number. What V25 actually constrains is the "
    "timer, so LapTimer is driven with an exact 30 m/s traversal of the "
    "centreline. A real driven lap is reported separately and honestly.",
    "6. Bare `python3 -m drive.drive` runs the self-check, not the window. Every "
    "other module in the package answers its self-check that way and the build "
    "instructions require it; --interactive (or any other CLI option) starts the "
    "sim as the contract's CLI describes.",
)


# ==================================================================== #
#  SETTINGS (the ESC menu's second page; persisted between launches)   #
# ==================================================================== #
SETTINGS_PATH = os.path.join("runs", "settings.json")
GEARBOX_MODES = ("auto", "manual", "clutch")          # drive.input.GEARBOX_MODES
GEARBOX_LABELS = {"auto": "Automatic",
                  "manual": "Manual (auto clutch)",
                  "clutch": "Manual + clutch pedal"}
GEARBOX_HUD = {"auto": "AUTO", "manual": "MAN", "clutch": "MAN+CL"}
SURFACE_MODES = ("patch", "none", "all")
SURFACE_LABELS = {"none": "Dry everywhere", "patch": "Dry, wet patches",
                  "all": "Wet everywhere"}
CAMERA_MODES = ("car_up", "chase", "world_up")
CAMERA_LABELS = {"car_up": "Car up", "chase": "Chase", "world_up": "World up"}
# The Engine setting: VehicleConfig.power_scale (WOT torque x, clutch uprated
# with it; powertrain.from_car). 'stock' is the car every script measures.
ENGINE_MODES = ("stock", "tuned", "sport")
ENGINE_SCALE = {"stock": 1.0, "tuned": 1.5, "sport": 2.0}
ENGINE_LABELS = {"stock": "Stock 1.2 16V (75 hp)", "tuned": "Tuned (~110 hp)",
                 "sport": "Sport (~150 hp)"}
ENGINE_HUD = {"stock": "75 HP", "tuned": "110 HP", "sport": "150 HP"}
ENGINE_DEFAULT = "sport"      # the seat's default: a 75 hp 1.2 is slow from it
SOUND_MODES = ("off", "low", "mid", "high")
SOUND_VOLUME = {"off": 0.0, "low": 0.3, "mid": 0.6, "high": 1.0}
SOUND_LABELS = {"off": "Off", "low": "Low", "mid": "Medium", "high": "High"}
SOUND_DEFAULT = "mid"


@dataclass
class Settings:
    """What the ESC > Settings page edits. Saved to runs/settings.json on every
    change and reloaded at launch, so the car starts the way it was left.
    Explicit command-line flags win over the file for that launch (and are
    then saved). Scripted / headless runs never read this file: the
    acceptance numbers must not depend on what the last driver clicked."""

    track: str = "arena"          # trk.TRACK_ORDER
    engine: str = ENGINE_DEFAULT  # ENGINE_MODES -> VehicleConfig.power_scale
    gearbox: str = "auto"         # GEARBOX_MODES
    abs: bool = True              # VehicleConfig.abs_on
    tc: bool = True               # VehicleConfig.tc_on
    steer_aid: bool = True        # input.KeyboardInput.steer_limit
    wet: str = "patch"            # SURFACE_MODES
    camera: str = "car_up"        # CAMERA_MODES
    sound: str = SOUND_DEFAULT    # SOUND_MODES -> audio.CarSound volume
    path: str = field(default=SETTINGS_PATH, repr=False, compare=False)

    KEYS = ("track", "engine", "gearbox", "abs", "tc", "steer_aid", "wet",
            "camera", "sound")

    def clamp(self) -> "Settings":
        if self.track not in trk.TRACKS:
            self.track = "arena"
        if self.engine not in ENGINE_MODES:
            self.engine = ENGINE_DEFAULT
        if self.gearbox not in GEARBOX_MODES:
            self.gearbox = "auto"
        if self.wet not in SURFACE_MODES:
            self.wet = "patch"
        if self.camera not in CAMERA_MODES:
            self.camera = "car_up"
        if self.sound not in SOUND_MODES:
            self.sound = SOUND_DEFAULT
        self.abs = bool(self.abs)
        self.tc = bool(self.tc)
        self.steer_aid = bool(self.steer_aid)
        return self

    @property
    def power_scale(self) -> float:
        return ENGINE_SCALE.get(self.engine, 1.0)

    @property
    def volume(self) -> float:
        return SOUND_VOLUME.get(self.sound, 0.0)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("path", None)
        return d

    def save(self, path: str | None = None) -> str:
        path = path or self.path
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump(self.as_dict(), f, indent=2)
        except OSError as exc:
            print(f"settings: could not save {path} ({exc})")
        return path

    @classmethod
    def load(cls, path: str = SETTINGS_PATH) -> "Settings":
        s = cls(path=path)
        try:
            with open(path) as f:
                d = json.load(f)
            for k in cls.KEYS:
                if k in d:
                    setattr(s, k, d[k])
        except (OSError, ValueError, TypeError):
            pass
        return s.clamp()

    def apply_cli(self, opts) -> "Settings":
        """Explicit flags override the file; then the resolved values are
        written back onto opts so one code path builds the session."""
        if getattr(opts, "track", None):
            self.track = opts.track
        if getattr(opts, "engine", None):
            self.engine = opts.engine
        if getattr(opts, "gearbox", None):
            self.gearbox = opts.gearbox
        if getattr(opts, "abs", None) is not None:
            self.abs = bool(opts.abs)
        if getattr(opts, "tc", None) is not None:
            self.tc = bool(opts.tc)
        if getattr(opts, "sound", None):
            self.sound = opts.sound
        if getattr(opts, "steer_limit", None) is not None:
            self.steer_aid = bool(opts.steer_limit)
        if getattr(opts, "wet", None):
            self.wet = opts.wet
        if getattr(opts, "camera", None):
            self.camera = opts.camera
        self.clamp()
        return self.to_opts(opts)

    def to_opts(self, opts) -> "Settings":
        """Write the current values onto opts. Called before EVERY session:
        a restart after TAB or the Map setting must build the new map, not
        the one the command line named at launch."""
        opts.track, opts.gearbox, opts.abs = self.track, self.gearbox, self.abs
        opts.steer_limit, opts.wet, opts.camera = self.steer_aid, self.wet, self.camera
        opts.engine, opts.tc, opts.sound = self.engine, self.tc, self.sound
        opts.auto_gearbox = (self.gearbox == "auto")
        return self

    def cycle(self, key: str) -> None:
        """Advance one setting to its next value (the menu's ENTER)."""
        if key == "track":
            order = trk.TRACK_ORDER
            self.track = order[(order.index(self.track) + 1) % len(order)]
        elif key == "engine":
            self.engine = ENGINE_MODES[(ENGINE_MODES.index(self.engine) + 1)
                                       % len(ENGINE_MODES)]
        elif key == "tc":
            self.tc = not self.tc
        elif key == "sound":
            self.sound = SOUND_MODES[(SOUND_MODES.index(self.sound) + 1)
                                     % len(SOUND_MODES)]
        elif key == "gearbox":
            self.gearbox = GEARBOX_MODES[(GEARBOX_MODES.index(self.gearbox) + 1)
                                         % len(GEARBOX_MODES)]
        elif key == "abs":
            self.abs = not self.abs
        elif key == "steer_aid":
            self.steer_aid = not self.steer_aid
        elif key == "wet":
            self.wet = SURFACE_MODES[(SURFACE_MODES.index(self.wet) + 1)
                                     % len(SURFACE_MODES)]
        elif key == "camera":
            self.camera = CAMERA_MODES[(CAMERA_MODES.index(self.camera) + 1)
                                       % len(CAMERA_MODES)]


SETTINGS_HELP = [
    ("SETTINGS", [
        ("ENTER / CROSS", "cycle the value"),
        ("ESC / CIRCLE", "back to the pause menu"),
        ("TAB", "next map (while driving)"),
        ("BACKSPACE", "garage (while driving; touchpad on the pad)"),
    ]),
    ("ENGINE", [
        ("Stock", "the 1.2 16V, 75 hp: every scripted number is this car"),
        ("Tuned / Sport", "1.5x / 2x the torque curve, clutch uprated to suit;"),
        ("", "TC keeps the fronts from spinning through 1st"),
    ]),
    ("GEARBOX", [
        ("Automatic", "the box shifts and works the clutch"),
        ("Manual", "E / Q or R1 / L1 shift; the box does the launch,"),
        ("", "the downshift blip and the restart - it cannot stall"),
        ("Manual + clutch", "Z / SQUARE is the only clutch: launch on it,"),
        ("", "it can stall; clutch fully in (or S) restarts it"),
    ]),
    ("MAPS", [
        ("Arena circuit", "1249 m, 7 corners R 30..130 m, wet patches"),
        ("Open proving ground", "522 x 362 m pad: skidpad circles, slalom,"),
        ("", "300 m drag lane, wet square; road round the edge"),
        ("Skidpad", "constant radius (--radius), guide circles"),
        ("Dragstrip", "1500 m straight; 1/8 mile, 1/4 mile, km gates"),
    ]),
]
SETTINGS_NOTE = ("Map and surface changes restart the session on the new map with "
                 "the same car; the rest apply at once. Saved to runs/settings.json.")


# ==================================================================== #
#  SCRIPTED INPUT (fallback; see DEVIATION 1)                          #
# ==================================================================== #
class ScriptedInput:
    """A virtual driver: `fn(t, vehicle, track) -> Controls`.

    Reads no wall clock, no pygame and no module-level state, which is the whole
    property that makes a headless run bitwise reproducible. `t = n_steps*dt` is
    counted here rather than read off the Sim, exactly as
    `drive.input.ScriptedInput` does -- the two are behaviourally identical, and
    `Sim` accepts either (see `_bind_input`). The local copy is what every entry
    in SCRIPTS uses, so that the entire acceptance path imports no pygame at all.
    """

    def __init__(self, fn, vehicle=None, track=None):
        self.fn = fn
        self.vehicle = vehicle
        self.track = track
        self.n_steps = 0
        self.dt = 0.0
        self.events: list = []

    def bind(self, vehicle, track=None):
        self.vehicle = vehicle
        self.track = track
        return self

    @property
    def t(self) -> float:
        return self.n_steps * self.dt

    def poll_events(self) -> list:
        out, self.events = self.events, []
        return out

    def update(self, dt, V=0.0, beta_deg=0.0, rpm=0.0, gear=1) -> Controls:
        self.dt = dt
        ctl = self.fn(self.n_steps * dt, self.vehicle, self.track)
        self.n_steps += 1
        return ctl

    def reset(self) -> None:
        self.n_steps = 0


def _scripted(fn):
    """Wrap a driver function. Deliberately the LOCAL ScriptedInput.

    drive.input.ScriptedInput is equivalent and is accepted by Sim, but it lives
    in a module that imports pygame at import time. Nothing on the acceptance
    path should have to. `Sim` binds whichever it is handed.
    """
    return ScriptedInput(fn)


# ==================================================================== #
#  LAP AND SECTOR TIMING  (spec eq.17)                                 #
# ==================================================================== #
class LapTimer:
    """Sub-step-accurate line crossings on a closed track.

    Two things make this harder than `if s < s_prev`. The wrap discontinuity at
    the start line looks identical to a car reversing across it, and lateral
    noise while sitting on the line looks like a crossing every step. So: the
    wrap case is required to be a wrap (`s_prev > L-20 and s_now < 20`), a 3 s
    lockout kills re-fires, and the crossing time is interpolated INSIDE the
    step -- V25's 0.002 s tolerance is tighter than the 0.001 s step itself, so
    whole-step timing cannot pass it no matter how the rest is written.
    """

    def __init__(self, track, lockout=LAP_LOCKOUT_S, wrap=LAP_WRAP_WINDOW):
        self.L = float(track.length)
        self.closed = bool(track.closed)
        self.lines = [float(s) for s in (track.sector_s or [0.0])]
        self.lockout = float(lockout)
        self.wrap = float(wrap)
        self.reset()

    def reset(self):
        self.lap = 0
        self.sector = 0
        self.t_lap_start = None
        self.t_sec_start = None
        self.last_lap = float("nan")
        self.best_lap = float("nan")
        self.lap_time = 0.0
        self.sector_times: list = [float("nan")] * len(self.lines)
        self.sector_best: list = [float("nan")] * len(self.lines)
        self.lap_valid = True
        self._valid_run = True
        self.crossings: list = []          # (t_cross, kind, index)

    # ---------------------------------------------------------------- #
    def _crossed(self, S, s_prev, s_now):
        """(hit, frac). frac is the fraction of THIS step at which s passed S."""
        L, w = self.L, self.wrap
        forward = (s_prev < S <= s_now)
        wrapped = (self.closed and s_prev > L - w and s_now < w)
        if not (forward or wrapped):
            return False, 0.0
        ds = (s_now - s_prev) % L
        if ds <= 0.0:
            return False, 0.0
        d0 = (S - s_prev) % L
        if d0 > ds:
            return False, 0.0
        return True, d0 / ds

    def update(self, t_prev, s_prev, s_now, dt, all_off_track=False,
               active=True):
        """Advance one physics step. Returns the list of events for this step.

        `active=False` (the car is off the ribbon -- in the middle of the
        open map's pad) suspends crossing detection: there the nearest
        centreline point flips between the two sides of the perimeter loop
        and `s` jumps by hundreds of metres in one step, which would read as
        a whole lap. The same jump guard (> 10 m in one step: at 1 kHz a real
        step is < 0.06 m) also covers a reset and a projection flip near the
        ribbon.
        """
        if all_off_track:
            self._valid_run = False
        events = []
        ds = (s_now - s_prev) % self.L if self.closed else (s_now - s_prev)
        if not active or ds > 10.0:
            if self.t_lap_start is not None:
                self.lap_time = (t_prev + dt) - self.t_lap_start
            return events
        for idx, S in enumerate(self.lines):
            hit, frac = self._crossed(S, s_prev, s_now)
            if not hit:
                continue
            t_cross = t_prev + frac * dt
            if idx == 0:
                if self.t_lap_start is not None:
                    if t_cross - self.t_lap_start < self.lockout:
                        continue           # lockout: a re-fire, not a lap
                    self.last_lap = t_cross - self.t_lap_start
                    self.lap_valid = self._valid_run
                    if self._valid_run and (math.isnan(self.best_lap)
                                            or self.last_lap < self.best_lap):
                        self.best_lap = self.last_lap
                    self.lap += 1
                    events.append(("lap", self.lap, t_cross, self.last_lap))
                    self.crossings.append((t_cross, "lap", self.lap))
                else:
                    events.append(("start", 0, t_cross, float("nan")))
                    self.crossings.append((t_cross, "start", 0))
                self.t_lap_start = t_cross
                self._valid_run = True
            if self.t_sec_start is not None and len(self.lines) > 1:
                prev = (idx - 1) % len(self.lines)
                self.sector_times[prev] = t_cross - self.t_sec_start
                if (math.isnan(self.sector_best[prev])
                        or self.sector_times[prev] < self.sector_best[prev]):
                    self.sector_best[prev] = self.sector_times[prev]
                events.append(("sector", prev, t_cross, self.sector_times[prev]))
            self.t_sec_start = t_cross
            self.sector = idx
        if self.t_lap_start is not None:
            self.lap_time = (t_prev + dt) - self.t_lap_start
        return events


# ==================================================================== #
#  SKID BUFFER (harness-side; render.py draws it)                      #
# ==================================================================== #
class NullSkidBuffer:
    """Headless stand-in. Same three methods, stores nothing, draws nothing.

    CONTRACT section 7 gives SkidBuffer to render.py, and this module used to
    carry a second implementation of it. The two `visible()` signatures drifted
    -- render.py calls `visible(cam, radius, max_segs)` while the copy here read
    the third positional as `t_now` -- so every mark was culled as stale, the
    call returned four empty lists instead of a list of segments, and the very
    first rendered frame died unpacking `[]`. Headless runs never saw it because
    nothing there draws.

    So there is exactly one SkidBuffer now, it lives in render.py, and `Sim`
    only builds one when a renderer exists. This class is what it uses
    otherwise: emission at 100 Hz that nothing will ever read is pure cost.
    """

    __slots__ = ()

    def emit(self, t, xy4, active):
        pass

    def clear(self):
        pass

    def visible(self, cam_xy, radius_m, max_segs=600, t_now=None):
        return []

    def n_points(self):
        return 0


# ==================================================================== #
#  THE SIM                                                             #
# ==================================================================== #
class Sim:
    """One car, one track, one driver, one loop.

    `step_physics` is the single place a physics step is taken, so the
    interactive and headless loops cannot drift apart in what they simulate --
    only in WHEN they call it.
    """

    def __init__(self, vehicle: Vehicle, track, inp,
                 renderer=None, telem=None, dt: float = DT_PHYS,
                 wing: str = "off", global_wet: float = 1.0,
                 settings: Settings | None = None):
        self.veh = vehicle
        self.track = track
        self.inp = inp
        self.renderer = renderer
        self.telem = telem
        self.dt = float(dt)
        self.wing = wing
        self.global_wet = float(global_wet)
        # the settings page (ESC > Settings). A scripted Sim gets the defaults
        # and never saves them; the interactive session hands in the loaded
        # file and every change is written back.
        self.settings = settings if settings is not None else Settings(path="")
        self.gearbox = self.settings.gearbox
        self.audio = None          # audio.CarSound; only a windowed session
        self.sound_enabled = False # set by _interactive_session: the one place
        #                            that may touch the mixer (V26's fake
        #                            renderer, or any headless Sim, never does)

        # decimation strides, derived from dt so --dt 0.002 still logs at 100 Hz
        self.surf_stride = max(1, int(round(1.0 / (SURFACE_LOOKUP_HZ * self.dt))))
        self.skid_stride = max(1, int(round(1.0 / (SKID_HZ * self.dt))))

        self.n = 0
        self.t = 0.0
        self.mu = [1.0, 1.0, 1.0, 1.0]
        self.crr = [1.0, 1.0, 1.0, 1.0]
        self.on_track4 = [True, True, True, True]
        self.ctl = Controls()

        self.s = 0.0
        self.n_lat = 0.0
        self.kappa_track = 0.0
        self.psi_c = 0.0
        self.on_track = True
        self._s_prev = None

        self.lap = LapTimer(track)
        # render.py owns SkidBuffer (CONTRACT section 7). Only build a real one
        # when something is going to draw it; otherwise emitting at 100 Hz for
        # a whole scripted run is cost with no reader.
        if renderer is not None:
            from . import render as _rnd
            self.skid = _rnd.SkidBuffer()
        else:
            self.skid = NullSkidBuffer()

        # --- accumulator state (interactive only) --------------------
        self.acc = 0.0
        self.alpha_render = 0.0
        self.time_scale = 1.0
        self.dropped_s = 0.0
        self.dropped_frames = 0
        self._first_frame = True
        self.paused = False
        self.single_step = False
        self.quit = False
        self.rtf = 0.0
        self._rtf_wall = 0.0
        self._rtf_sim = 0.0

        self.wing_on = (wing != "off")
        self.wing_side_mode = 0            # 0 auto, +1 force left, -1 force right
        self.pose_prev = (self.veh.x, self.veh.y, self.veh.psi)
        self.events_log: list = []
        self.stop_reason = ""
        self._bound = False

        # pause menu (ESC / OPTIONS): built lazily, render path only
        self.menu = None
        self.has_garage = False            # set by the interactive session
        self.hud_cfg = None                # the garage build's HudData fields
        self._menu_was_paused = False
        self._menu_page = "main"           # 'main' | 'settings'

        self._bind_input()
        self._sample_surfaces()
        self.set_gearbox(self.gearbox)

    # ---------------------------------------------------------------- #
    def _bind_input(self):
        """Give the driver its vehicle and track, once.

        Two shapes exist in the wild: `bind(vehicle, track)` (drive.input's and
        the local ScriptedInput's) and bare attributes. `t` is NOT set from here
        -- both ScriptedInputs count their own `n_steps*dt`, and writing to a
        read-only property is how this first broke.
        """
        if self._bound:
            return
        inp = self.inp
        b = getattr(inp, "bind", None)
        if callable(b):
            try:
                b(self.veh, self.track)
                self._bound = True
                return
            except TypeError:
                pass
        for name, val in (("vehicle", self.veh), ("veh", self.veh),
                          ("track", self.track)):
            if hasattr(inp, name):
                try:
                    setattr(inp, name, val)
                except AttributeError:
                    pass
        self._bound = True

    def wheel_world(self):
        """(4,) world contact-patch positions, FL FR RL RR."""
        c = self.veh.car
        cs, sn = cos(self.veh.psi), sin(self.veh.psi)
        x, y = self.veh.x, self.veh.y
        return [(x + bx * cs - by * sn, y + bx * sn + by * cs)
                for (bx, by) in wheel_positions(c)]

    def _sample_surfaces(self):
        """PER WHEEL. Four project() calls. See the module docstring."""
        pts = self.wheel_world()
        for i, (wx, wy) in enumerate(pts):
            mu, crr, on = trk.surface_at(self.track, wx, wy, self.global_wet)
            self.mu[i] = mu
            self.crr[i] = crr
            self.on_track4[i] = on

    # ---------------------------------------------------------------- #
    def step_physics(self, dt: float) -> None:
        """Exactly one physics step. No wall clock, no pygame, no allocation storm."""
        veh, tr = self.veh, self.track

        if self.n % self.surf_stride == 0:
            self._sample_surfaces()

        V = hypot(veh.u, veh.v)
        beta_deg = degrees(atan2(veh.v, max(abs(veh.u), 0.5)))
        self._bind_input()
        ctl = self.inp.update(dt, V, beta_deg, veh.rpm, veh.gear)
        if self.wing_on and not ctl.wing_on:
            ctl.wing_on = True             # the harness's F toggle, OR'd in
        self.ctl = ctl

        self.pose_prev = (veh.x, veh.y, veh.psi)
        veh.step(ctl, tuple(self.mu), tuple(self.crr), dt)

        s_prev = self.s if self._s_prev is not None else None
        s, n_lat, kt, psi_c, _i = trk.project(tr, veh.x, veh.y)
        self.s, self.n_lat, self.kappa_track, self.psi_c = s, n_lat, kt, psi_c
        on_ribbon = abs(n_lat) <= 0.5 * tr.width
        # the HUD's OFF TRACK: the ribbon on a circuit, ribbon OR pad on the
        # open map (track.on_tarmac reuses the projection just made)
        self.on_track = on_ribbon or (bool(tr.areas)
                                      and trk.on_tarmac(tr, veh.x, veh.y, n_lat))

        t_prev = self.t
        if s_prev is not None:
            evs = self.lap.update(t_prev, s_prev, s, dt,
                                  all_off_track=not any(self.on_track4),
                                  active=abs(n_lat) <= 0.5 * tr.width + 2.0)
            for e in evs:
                self.events_log.append(e)
                if self.telem is not None and e[0] in ("lap", "sector"):
                    self.telem.mark(f"{e[0]}{e[1]}")
        self._s_prev = s

        if self.n % self.skid_stride == 0:
            self._emit_skid()

        self.n += 1
        self.t = self.n * dt               # t = n*dt, NEVER accumulated
        self._log(self.n)

    def _emit_skid(self):
        veh = self.veh
        act = [False] * 4
        for i in range(4):
            cap = qss.fy_max(float(veh.Fz[i]), mu_scale=self.mu[i], **qss.TYRE)
            act[i] = (abs(float(veh.Fy[i])) / max(cap, 1.0) > 0.92
                      or abs(float(veh.kappa[i])) > 0.12)
        if any(act):
            self.skid.emit(self.t, self.wheel_world(), act)

    # ---------------------------------------------------------------- #
    def telemetry_row(self) -> dict:
        """The 65-column schema as a plain dict. Missing -> the writer's nan."""
        v, c = self.veh, self.ctl
        V = hypot(v.u, v.v)
        Fz, Fx, Fy = v.Fz, v.Fx, v.Fy
        al, ka, om = v.alpha, v.kappa, v.omega
        d_road = degrees(c.delta)
        return {
            "t": self.t, "x": v.x, "y": v.y, "psi": v.psi,
            "u": v.u, "v": v.v, "r": v.r, "V": V,
            "beta_deg": degrees(v.beta), "ax": v.ax, "ay": v.ay,
            "ax_g": v.ax / G, "ay_g": v.ay / G,
            "gear": v.gear, "rpm": v.rpm, "clutch_eng": bool(v.engaged),
            "throttle": c.throttle, "brake": c.brake, "clutch": c.clutch,
            "handbrake": c.handbrake, "delta_road_deg": d_road,
            "steer_in": d_road / DELTA_LOCK_DEG,
            "Fz_fl": Fz[0], "Fz_fr": Fz[1], "Fz_rl": Fz[2], "Fz_rr": Fz[3],
            "Fx_fl": Fx[0], "Fx_fr": Fx[1], "Fx_rl": Fx[2], "Fx_rr": Fx[3],
            "Fy_fl": Fy[0], "Fy_fr": Fy[1], "Fy_rl": Fy[2], "Fy_rr": Fy[3],
            "alpha_fl_deg": degrees(al[0]), "alpha_fr_deg": degrees(al[1]),
            "alpha_rl_deg": degrees(al[2]), "alpha_rr_deg": degrees(al[3]),
            "kappa_fl": ka[0], "kappa_fr": ka[1],
            "kappa_rl": ka[2], "kappa_rr": ka[3],
            "mu_fl": self.mu[0], "mu_fr": self.mu[1],
            "mu_rl": self.mu[2], "mu_rr": self.mu[3],
            "omega_fl": om[0], "omega_fr": om[1],
            "omega_rl": om[2], "omega_rr": om[3],
            "util_f": v.util_f, "util_r": v.util_r, "limited_by": v.limited_by,
            "wing_deploy": v.wing_deploy, "wing_side": v.wing_side,
            "F_wing": v.F_wing, "D_wing": v.D_wing,
            "s": self.s, "n": self.n_lat, "kappa_track": self.kappa_track,
            "on_track": bool(self.on_track),
            "lap": self.lap.lap, "sector": self.lap.sector,
            "lap_time": self.lap.lap_time,
        }

    def _log(self, index):
        if self.telem is not None:
            self.telem.maybe_log(index, self.telemetry_row())

    # ---------------------------------------------------------------- #
    #  THE ACCUMULATOR                                                 #
    # ---------------------------------------------------------------- #
    def pump(self, dt_wall_raw: float) -> int:
        """One frame's worth of physics. Returns the number of steps taken.

        Split out of run_interactive so V23/V24 can drive it with a synthetic
        frame time and no window. Everything eq.1 specifies happens here and
        nowhere else.
        """
        raw = float(dt_wall_raw)
        if self._first_frame:
            # macOS's 0.5-2 s first frame, and the same after reset/unpause.
            self._first_frame = False
            dt_wall = self.dt
            self.acc = self.dt          # NOT acc += dt*time_scale (DEVIATION 3)
            raw = self.dt
        else:
            dt_wall = min(raw, MAX_FRAME_DT)
            self.acc += dt_wall * self.time_scale

        n = 0
        while self.acc >= self.dt and n < MAX_SUBSTEPS:
            self.step_physics(self.dt)
            self.acc -= self.dt
            n += 1

        clipped = max(raw - dt_wall, 0.0) * self.time_scale
        over = self.acc if self.acc >= self.dt else 0.0
        if clipped > 0.0 or over > 0.0:
            self.dropped_s += clipped + over
            self.dropped_frames += 1
            if over > 0.0:
                self.acc = 0.0
        self.alpha_render = self.acc / self.dt
        return n

    # ---------------------------------------------------------------- #
    def reset(self, to_checkpoint: bool = False) -> None:
        """R = back to the last sector line; SHIFT+R = full reset.

        Both force the first-frame guard, because the wall-clock gap across a
        reset is exactly the 0.5-2 s pause the guard exists for.
        """
        tr = self.track
        s0 = 0.0
        V0 = 0.0
        gear = 1
        if to_checkpoint and tr.sector_s:
            cands = [s for s in tr.sector_s if s <= self.s]
            s0 = max(cands) if cands else 0.0
            V0 = min(hypot(self.veh.u, self.veh.v), 25.0)
            gear = max(self.veh.gear, 1)
        x, y = trk.point_at(tr, s0, 0.0)
        _, _, _, psi_c, _ = trk.project(tr, x, y)
        if self.gearbox == "clutch" and V0 < 0.5:
            gear = 0                       # a stationary H-pattern car sits in neutral
        self.veh.reset(x, y, psi_c, V=V0, gear=gear)
        self.s = s0
        self._s_prev = None
        self.acc = 0.0
        self._first_frame = True
        self._sample_surfaces()
        if not to_checkpoint:
            self.lap.reset()
            self.skid.clear()

    def unpause(self):
        self.paused = False
        self._first_frame = True

    # ---------------------------------------------------------------- #
    def run_headless(self, duration_s: float) -> None:
        """No clock. No accumulator. No pygame. Bitwise reproducible."""
        n = int(round(float(duration_s) / self.dt))
        self._log(0)                       # the t = 0 row
        for _ in range(n):
            self.step_physics(self.dt)
            if self.quit:
                break

    # ---------------------------------------------------------------- #
    def run_interactive(self) -> None:
        """Eq.1. The only loop in this package that touches a wall clock."""
        import pygame

        clock = pygame.time.Clock()
        # tick_busy_loop, not tick. Measured on this machine with the full HUD:
        # the frame's actual work is 7.9 ms, comfortably inside the 16.7 ms
        # budget, but Clock.tick pads with pygame.time.delay, which sleeps in
        # whole milliseconds and overshoots -- it produced 19.6 ms frames, i.e.
        # 49 fps, while claiming to cap at 60. tick_busy_loop spins instead and
        # holds 60. The physics is a fixed 1 kHz behind an accumulator either
        # way, so this is frame pacing and nothing else: the car is identical,
        # it just stops looking like it is stuttering. The cost is that the
        # render thread burns the ~8 ms it would otherwise have slept.
        self._first_frame = True
        self._log(0)
        t_wall0 = time.perf_counter()
        while not self.quit:
            dt_wall = clock.tick_busy_loop(FPS) / 1000.0
            for ev in self.inp.poll_events():
                self.handle_event(ev)
            if self.quit:
                break

            if self.paused:
                if self.single_step:
                    self.step_physics(self.dt)
                    self.single_step = False
                self.acc = 0.0
                self.alpha_render = 0.0
                nstep = 0
            else:
                nstep = self.pump(dt_wall)

            w = time.perf_counter() - t_wall0
            self._rtf_wall = w
            self._rtf_sim = self.t
            self.rtf = self.t / w if w > 0 else 0.0

            if self.renderer is not None:
                hud = self.hud_data()
                hud.paused = self.paused
                hud.time_scale = self.time_scale
                self.renderer.update_camera(self.veh, dt_wall)
                self.renderer.draw_frame(self.veh, self.pose_prev,
                                         self.alpha_render, self.ctl,
                                         hud, self.skid)
                self.renderer.present()
                fb = getattr(self.inp, "feedback", None)
                if fb is not None:
                    fb(hud)                      # pad rumble; render loop only
                if self.audio is not None:
                    self.audio.update(hud)       # engine / tyres; render loop only
            del nstep
        if self.audio is not None:
            self.audio.stop()

    # ---------------------------------------------------------------- #
    #  SETTINGS                                                        #
    # ---------------------------------------------------------------- #
    def set_gearbox(self, mode: str) -> None:
        """'auto' | 'manual' | 'clutch' -> the input layer's Controls flags.

        Handing a stationary car in 1st to the clutch-pedal mode would stall
        it on the spot (clutch out, wheels stopped): the box goes to neutral
        first, as a driver would leave it. Rolling, the gear is kept."""
        if mode not in GEARBOX_MODES:
            mode = "auto"
        self.gearbox = mode
        self.settings.gearbox = mode
        sg = getattr(self.inp, "set_gearbox", None)
        if sg is not None:
            sg(mode)
        v = self.veh
        if mode == "clutch" and hypot(v.u, v.v) < 0.5 and v.pt_s.gear > 0:
            self._select_neutral()

    def set_engine(self, mode: str) -> None:
        """'stock' | 'tuned' | 'sport' -> the powertrain's torque curve and
        clutch, live: the state (engine speed, gear, clutch slip) is untouched,
        only the parameter set is swapped."""
        from . import powertrain as ptm
        if mode not in ENGINE_MODES:
            mode = ENGINE_DEFAULT
        self.settings.engine = mode
        v = self.veh
        v.cfg.power_scale = ENGINE_SCALE[mode]
        v.pt_p = ptm.PowertrainParams.from_car(v.car, power_scale=v.cfg.power_scale)

    def _audio_apply(self) -> None:
        """The Sound setting: build, re-level or drop the CarSound. Never
        raises: no audio device just means a silent sim."""
        vol = self.settings.volume
        if self.renderer is None or not self.sound_enabled or vol <= 0.0:
            if self.audio is not None:
                self.audio.stop()
                self.audio = None
            return
        if self.audio is None:
            try:
                from .audio import CarSound
                snd = CarSound(volume=vol)
                self.audio = snd if snd.ok else None
                if not snd.ok:
                    print(f"sound unavailable ({snd.error})")
            except Exception as exc:
                print(f"drive.audio unavailable ({exc}); silent")
                self.audio = None
        else:
            self.audio.set_volume(vol)

    def _select_neutral(self) -> None:
        from . import powertrain as ptm
        v = self.veh
        v.pt_s.gear = 0
        v.pt_s.n_tot = 0.0
        v.pt_s.I_w_front_eff = ptm.I_w_front(v.pt_p, 0)
        v.pt_s.shift_phase, v.pt_s.clutch_auto = "none", 0.0
        v.gear = 0

    def apply_setting(self, key: str) -> bool:
        """Cycle one setting, apply it live, save. Returns True when the
        session has to be rebuilt (a new map or a new surface set)."""
        s = self.settings
        s.cycle(key)
        restart = False
        if key == "track" or key == "wet":
            restart = True
        elif key == "gearbox":
            self.set_gearbox(s.gearbox)
        elif key == "engine":
            self.set_engine(s.engine)
        elif key == "abs":
            self.veh.cfg.abs_on = bool(s.abs)
        elif key == "tc":
            self.veh.cfg.tc_on = bool(s.tc)
        elif key == "sound":
            self._audio_apply()
        elif key == "steer_aid":
            ssl = getattr(self.inp, "set_steer_limit", None)
            if ssl is not None:
                ssl(bool(s.steer_aid))
        elif key == "camera":
            if self.renderer is not None:
                self.renderer.cfg.mode = s.camera
        if s.path:
            s.save()
        return restart

    def restart(self) -> None:
        """End this session so the outer loop builds a new one from the
        (already saved) settings, in the same window, with the same car."""
        self.stop_reason = "restart"
        self.quit = True

    # ---------------------------------------------------------------- #
    def handle_event(self, ev: str) -> None:
        """Discrete commands from poll_events(). Unknown strings are ignored."""
        if self.menu is not None and self.menu.open:
            self._menu_event(ev)
            return
        if ev in ("quit", "escape"):
            self.quit = True
        elif ev == "menu":
            self._menu_open()
        elif ev == "reset":
            self.reset(to_checkpoint=True)
        elif ev in ("full_reset", "reset_full"):
            self.reset(to_checkpoint=False)
        elif ev == "track_next":
            # TAB: the next map, same car, same settings, new session
            if self.apply_setting("track"):
                self.restart()
        elif ev == "gearbox":
            self.apply_setting("gearbox")
        elif ev == "pause":
            if self.paused:
                self.unpause()
            else:
                self.paused = True
        elif ev in ("step", "single_step"):
            self.single_step = True
        elif ev == "slowmo":
            self.time_scale = SLOWMO_SCALE
        elif ev in ("realtime", "normal_speed"):
            self.time_scale = 1.0
        elif ev == "wing":
            self.wing_on = not self.wing_on
        elif ev == "wing_side":
            self.wing_side_mode = {0: +1, +1: -1, -1: 0}[self.wing_side_mode]
        elif ev == "wet":
            self.global_wet = (MU_WET_SCALE if self.global_wet == 1.0 else 1.0)
            self._sample_surfaces()
        elif ev == "marker":
            if self.telem is not None:
                self.telem.mark("marker")
        elif ev == "clear_skid":
            self.skid.clear()
        elif ev == "garage":
            # hand the session to the 3D editor; the outer loop (run_interactive_cli)
            # re-enters it and comes back here with the car it built
            self.stop_reason = "garage"
            self.quit = True
        elif self.renderer is not None:
            self._view_event(ev)

    # ---------------------------------------------------------------- #
    def _menu_subtitle(self) -> str:
        cfg = self.veh.cfg
        wing = (f"{cfg.wing} x_w {cfg.x_w:+.2f} h_w {cfg.h_w:.2f}"
                if cfg.wing != "off" else "no flank panel")
        if getattr(cfg, "has_designed", lambda: False)():
            names = [getattr(w, "name", "") for w in (cfg.dev_left, cfg.dev_right, cfg.top)
                     if w is not None]
            wing = "garage build: " + ", ".join(dict.fromkeys(names))
        return (f"{self.track.title or self.track.name}   lap {self.lap.lap}   "
                f"{wing}   {GEARBOX_HUD.get(self.gearbox, '')}   t {self.t:.1f} s")

    def _menu_show_main(self, idx: int = 0) -> None:
        """The pause page: resume / settings / resets / garage / quit."""
        items = [("Resume", "resume"),
                 ("Settings: map, gearbox, ABS, aids, camera", "settings"),
                 ("Reset to last sector line", "reset"),
                 ("Full reset (skid marks + timing)", "full_reset")]
        if self.has_garage:
            items.append(("Garage: build the flank panel (3D)", "garage"))
        items.append(("Quit", "quit"))
        layout = getattr(self.inp, "layout", None)
        try:
            from .input import menu_help, MENU_NO_PAD
            sections = menu_help(layout)
            note = "" if layout else MENU_NO_PAD
        except Exception:
            sections, note = [], ""
        foot = "ESC / OPTIONS resume   R reset   SHIFT+R full reset   TAB next map"
        if self.has_garage:
            foot += "   BACKSPACE garage"
        self.menu.show(items=items, sections=sections, subtitle=self._menu_subtitle(),
                       note=note, footer=foot, title="PAUSED", idx=idx, columns=2)
        self._menu_page = "main"

    def _settings_items(self) -> list:
        s = self.settings
        rows = [(f"{'Map':<11s}{trk.TRACK_TITLES.get(s.track, s.track)}", "set:track"),
                (f"{'Engine':<11s}{ENGINE_LABELS[s.engine]}", "set:engine"),
                (f"{'Gearbox':<11s}{GEARBOX_LABELS[s.gearbox]}", "set:gearbox"),
                (f"{'ABS':<11s}{'On' if s.abs else 'Off'}", "set:abs"),
                (f"{'TC':<11s}{'On' if s.tc else 'Off'}", "set:tc"),
                (f"{'Steer aid':<11s}{'On' if s.steer_aid else 'Off'}", "set:steer_aid"),
                (f"{'Surface':<11s}{SURFACE_LABELS[s.wet]}", "set:wet"),
                (f"{'Camera':<11s}{CAMERA_LABELS[s.camera]}", "set:camera"),
                (f"{'Sound':<11s}{SOUND_LABELS[s.sound]}", "set:sound")]
        if self.has_garage:
            rows.append(("Garage (3D panel editor)", "garage"))
        rows.append(("Back", "settings_back"))
        return rows

    def _menu_show_settings(self, idx: int = 0) -> None:
        self.menu.show(items=self._settings_items(), sections=SETTINGS_HELP,
                       subtitle=self._menu_subtitle(), note=SETTINGS_NOTE,
                       footer="ENTER / CROSS cycle   ESC / CIRCLE back   "
                              "BACKSPACE garage", title="SETTINGS", idx=idx,
                       columns=1)
        self._menu_page = "settings"

    def _menu_open(self) -> None:
        """ESC / OPTIONS. Pauses the accumulator and hands the inputs to the
        menu (nav keys only until it closes). Without a renderer there is
        nothing to draw it on, so it degrades to a plain pause toggle."""
        if self.renderer is None:
            if self.paused:
                self.unpause()
            else:
                self.paused = True
            return
        if self.menu is None:
            from .menu import Menu
            self.menu = Menu("PAUSED")
        self._menu_show_main()
        self._menu_was_paused = self.paused
        self.paused = True
        sm = getattr(self.inp, "set_menu", None)
        if sm is not None:
            sm(True)

    def _menu_close(self) -> None:
        self.menu.hide()
        self._menu_page = "main"
        sm = getattr(self.inp, "set_menu", None)
        if sm is not None:
            sm(False)
        if not self._menu_was_paused:
            self.unpause()                 # first-frame guard: no catch-up burst

    def _menu_event(self, ev: str) -> None:
        """Everything while the menu is open. Hotkeys (R, SHIFT+R, BACKSPACE,
        CREATE, touchpad) act directly; the rest goes through the cursor.
        The settings page re-shows itself after every value change with the
        cursor where it was; ESC / CIRCLE there goes back to the pause page."""
        if ev == "quit":                   # window close
            self.quit = True
            return
        if ev in ("reset", "full_reset", "garage"):
            action = ev
        else:
            action = self.menu.handle(ev)
            if action is None:
                return
        if self._menu_page == "settings":
            idx = self.menu.idx
            if action in ("resume", "settings_back"):
                self._menu_show_main(idx=1)
                return
            if action.startswith("set:"):
                if self.apply_setting(action[4:]):
                    self._menu_close()
                    self.restart()
                else:
                    self._menu_show_settings(idx=idx)
                return
        elif action == "settings":
            self._menu_show_settings()
            return
        self._menu_close()
        if action == "reset":
            self.reset(to_checkpoint=True)
        elif action == "full_reset":
            self.reset(to_checkpoint=False)
        elif action == "garage":
            self.stop_reason = "garage"
            self.quit = True
        elif action == "quit":
            self.quit = True

    def _view_event(self, ev: str) -> None:
        """Camera / HUD toggles. Renderer config only; never physics."""
        cfg = self.renderer.cfg
        if ev == "camera":
            order = ("car_up", "chase", "world_up")
            cfg.mode = order[(order.index(cfg.mode) + 1) % 3] if cfg.mode in order else "car_up"
        elif ev == "zoom_in":
            self.renderer.set_zoom(self.renderer.zoom_manual * 1.25)
        elif ev == "zoom_out":
            self.renderer.set_zoom(self.renderer.zoom_manual / 1.25)
        elif ev == "zoom_auto":
            self.renderer.set_zoom(1.0)
        elif ev == "hud":
            order = ("full", "minimal", "off")
            cfg.hud = order[(order.index(cfg.hud) + 1) % 3] if cfg.hud in order else "full"
        elif ev == "vectors":
            cfg.show_vectors = not cfg.show_vectors
        elif ev == "gg":
            cfg.show_gg = not cfg.show_gg
        elif ev == "skid":
            cfg.show_skid = not cfg.show_skid

    # ---------------------------------------------------------------- #
    def hud_data(self):
        """Everything the HUD shows, assembled once per frame.

        util_f/util_r/limited_by come straight off the vehicle, which computes
        them by calling qss.fy_max with qss.TYRE (contract section 4). The
        harness must not recompute them: two implementations of mu(Fz) is
        exactly how V28 fails for a reason nobody can find.
        """
        v = self.veh
        d = dict(
            V=hypot(v.u, v.v), V_kmh=hypot(v.u, v.v) * 3.6, rpm=v.rpm, gear=v.gear,
            ay_g=v.ay / G, ax_g=v.ax / G, yaw_rate_deg=degrees(v.r),
            beta_deg=degrees(v.beta),
            util_f=v.util_f, util_r=v.util_r, limited_by=v.limited_by,
            Fz=np.asarray(v.Fz), mu=np.asarray(self.mu),
            wing_on=self.wing_on, wing_deploy=v.wing_deploy,
            wing_side=v.wing_side, F_wing=v.F_wing, D_wing=v.D_wing,
            lap=self.lap.lap, lap_time=self.lap.lap_time,
            last_lap=self.lap.last_lap, best_lap=self.lap.best_lap,
            sector=self.lap.sector, sector_times=list(self.lap.sector_times),
            sector_best=list(self.lap.sector_best), lap_valid=self.lap.lap_valid,
            on_track=self.on_track,
            mu_scale_car=sum(self.mu) / 4.0,
            rtf=self.rtf, dropped_frames=self.dropped_frames,
            x_w=v.cfg.x_w, h_w=v.cfg.h_w, wing_type=v.cfg.wing,
            inc_deg=degrees(v.cfg.delta_dev_geom), menu=self.menu,
            Fx=np.asarray(v.Fx), Fy=np.asarray(v.Fy),
            kappa=np.asarray(v.kappa), alpha=np.asarray(v.alpha),
            delta_wheel=np.asarray(v.delta_wheel),
            wheel_lift=np.asarray(v.wheel_lift, dtype=bool),
            stalled=bool(v.stalled), on_limiter=bool(v.on_limiter),
            gearbox=GEARBOX_HUD.get(self.gearbox, ""),
            abs_active=bool(v.abs_active[0] or v.abs_active[1]
                            or v.abs_active[2] or v.abs_active[3]),
            tc_active=bool(getattr(v, "tc_active", False)),
            engine=ENGINE_HUD.get(self.settings.engine, ""),
            eng_load=float(getattr(v, "eng_load", 0.0)),
            track_name=self.track.title or self.track.name,
            F_top=float(getattr(v, "F_top", 0.0)), D_top=float(getattr(v, "D_top", 0.0)),
            top_deploy=float(getattr(v, "top_deploy", 0.0)),
        )
        if self.hud_cfg:
            d.update(self.hud_cfg)          # the garage build's wing geometry
        try:
            from .render import HudData
            return HudData(**d)
        except Exception:
            from types import SimpleNamespace
            return SimpleNamespace(**d)


# ==================================================================== #
#  VIRTUAL DRIVERS                                                     #
# ==================================================================== #
def _wrap_pi(a):
    return atan2(sin(a), cos(a))


class SpeedPI:
    """Pedal from speed error. Shared by every script so they cannot disagree.

    Integral term with hard anti-windup: without it the skidpad follower sits
    0.3-0.5 m/s below target for the whole 8 s window and the bisection converges
    onto the CONTROLLER's limit rather than the car's.
    """

    def __init__(self, kp=KP_V, ki=KI_V):
        self.kp, self.ki, self.I = kp, ki, 0.0

    def reset(self):
        self.I = 0.0

    def __call__(self, V_tgt, V, dt):
        e = V_tgt - V
        self.I = min(max(self.I + e * dt, -3.0), 3.0)
        u = self.kp * e + self.ki * self.I
        if u >= 0.0:
            return min(u, 1.0), 0.0
        return 0.0, min(-u, 1.0)


class PathFollower:
    """Centreline follower: curvature feedforward + P on n + D on heading.

    delta = atan(L*kappa) - KP_N*n - KD_PSI*psi_err
    n > 0 is LEFT of the centreline and delta > 0 steers LEFT, so both feedback
    terms are negative. Getting either sign wrong gives a car that diverges
    smoothly and looks like a physics bug.
    """

    def __init__(self, V_tgt, wing_on=False, gains=(KP_N, KD_PSI, KI_N)):
        self.V_tgt = V_tgt
        self.wing_on = wing_on
        self.kp_n, self.kd, self.ki_n = gains
        self.I_n = 0.0
        self.pi = SpeedPI()
        self.max_n = 0.0
        self.max_n_tail = 0.0
        self.max_beta = 0.0
        self.V_hist: list = []
        self.n_hist: list = []

    def target(self, t, veh, tr, s, kt):
        return self.V_tgt

    def steer(self, veh, tr, dt=0.001):
        s, n, kt, psi_c, _ = trk.project(tr, veh.x, veh.y)
        psi_err = _wrap_pi(veh.psi - psi_c)
        if abs(n) < 2.0:            # integrate only near the line: winding up
            self.I_n += n * dt      # during the capture transient is what makes
        self.I_n = min(max(self.I_n, -I_N_LIM / max(self.ki_n, 1e-9)),
                       I_N_LIM / max(self.ki_n, 1e-9))
        delta = (atan(veh.car.L * kt) - self.kp_n * n
                 - self.ki_n * self.I_n - self.kd * psi_err)
        lock = radians(DELTA_LOCK_DEG)
        return min(max(delta, -lock), lock), s, n, kt

    def __call__(self, t, veh, tr):
        delta, s, n, kt = self.steer(veh, tr)
        V = hypot(veh.u, veh.v)
        thr, brk = self.pi(self.target(t, veh, tr, s, kt), V, 0.001)

        self.max_n = max(self.max_n, abs(n))
        self.max_beta = max(self.max_beta, abs(degrees(veh.beta)))
        self.V_hist.append(V)
        self.n_hist.append(n)
        return Controls(delta=delta, throttle=thr, brake=brk,
                        auto_gearbox=True, wing_on=self.wing_on)


def speed_profile(tr, margin=0.90, car=None, global_wet=1.0):
    """Quasi-steady target speed at every centreline sample (driver aid only).

    Uses the CONSTANT dry envelope AY_MAX_DRY rather than calling qss.max_ay per
    point: with k = 0 (no device) max_ay is independent of V, so the 200-step
    bisection would return the same 8.4608 m/s 2499 times. With a device it is
    not constant, but the driver deliberately targets the no-device envelope --
    the wing's job is to be measurable, not to be driven around.

    The envelope IS made surface-aware, per sample, and that is not optional on
    CIRCUIT_ARENA: WET_T3 puts mu_scale 0.632 through the fastest grip-limited
    corner on the lap. A driver targeting the dry envelope there understeers
    straight off -- measured |n| = 13.3 m against a 6 m half-width, i.e. the
    scripted lap never completes. Scaling a_y by the local mu costs one
    surface_at() call per sample, once, and is what a driver who has walked the
    track already knows. The braking pass uses the mu at the point being braked
    FROM, so the profile slows down before the patch rather than inside it.
    """
    car = car or CorsaC()
    k = np.abs(np.asarray(tr.kappa))
    ds = tr.ds
    N = len(k)
    mu = np.array([trk.surface_at(tr, float(x), float(y), global_wet)[0]
                   for x, y in tr.xy])
    ay = margin * AY_MAX_DRY * mu
    V = np.where(k < 1e-6, car.Vmax, np.sqrt(ay / np.maximum(k, 1e-9)))
    V = np.minimum(V, car.Vmax)
    wraps = 2 if tr.closed else 1
    for _ in range(wraps):                       # forward: traction limit
        for i in range(N - 1) if not tr.closed else range(N):
            j = (i + 1) % N
            drag = 0.5 * RHO * car.CdA * V[i] ** 2 + car.Crr * car.m * G
            ax = max(car.P_wheel / (car.m * max(V[i], 3.0)) - drag / car.m, 0.1)
            V[j] = min(V[j], sqrt(V[i] ** 2 + 2 * ax * ds))
    for _ in range(wraps):                       # backward: braking limit
        for i in range(N - 1, 0, -1) if not tr.closed else range(N - 1, -1, -1):
            j = (i - 1) % N
            V[j] = min(V[j], sqrt(V[i] ** 2 + 2 * ay[i] * ds))
    return V


class StraightDriver(PathFollower):
    """Full throttle (or a fixed pedal pair) in a straight line, lane kept.

    A lane keeper is NOT optional here and it is not a driver aid in the sense
    the spec forbids: with `delta` pinned at exactly 0 this chassis yaws +6.8 deg
    over a 40 s WOT run (measured, bare Vehicle, mu = 1 everywhere), because the
    tyre model is not exactly odd in alpha at kappa != 0 -- the residual
    PEY3 asymmetry tyre.py and vehicle.py both flag in their open issues. A real
    driver holds the car in its lane; a script that does not, drives off the
    strip at 200 m and reports a meaningless top speed. The steer command is
    still commanded DIRECTLY, with no speed-dependent limiter anywhere.
    """

    def __init__(self, throttle=1.0, brake=0.0, clutch=0.0, wing_on=False):
        super().__init__(0.0, wing_on=wing_on)
        self.thr, self.brk, self.clu = throttle, brake, clutch

    def __call__(self, t, veh, tr):
        delta, _s, n, _kt = self.steer(veh, tr)
        self.max_n = max(self.max_n, abs(n))
        self.V_hist.append(hypot(veh.u, veh.v))
        return Controls(delta=delta, throttle=self.thr, brake=self.brk,
                        clutch=self.clu, auto_gearbox=True,
                        wing_on=self.wing_on)


class LapDriver(PathFollower):
    """PathFollower whose speed target comes from the quasi-steady profile."""

    def __init__(self, tr, margin=0.90, wing_on=False, global_wet=1.0):
        super().__init__(0.0, wing_on=wing_on)
        self.prof = speed_profile(tr, margin, global_wet=global_wet)
        self.ds = tr.ds
        self.N = len(self.prof)
        self.lookahead = 12.0            # m, so the driver brakes BEFORE the corner

    def target(self, t, veh, tr, s, kt):
        i = int((s + self.lookahead) / self.ds) % self.N
        return float(self.prof[i])


# ==================================================================== #
#  SCRIPTS                                                             #
# ==================================================================== #
def _build(track_name="arena", radius=50.0, cw=False, wing="off",
           x_w=0.97, h_w=0.90, wet="patch", dt=DT_PHYS, telem_path=None,
           telem_hz=TELEM_HZ, precision="6g", driver=None, mu_scale=1.0,
           cmdline=None, start_V=0.0, gear=1, tag="", start_s=None):
    """One place that assembles a headless Sim, so every script agrees."""
    tr = trk.make_track(track_name, radius, cw, surfaces=(wet != "none"))
    global_wet = MU_WET_SCALE if wet == "all" else 1.0

    cfg = VehicleConfig(wing=wing, x_w=x_w, h_w=h_w, mu_scale=mu_scale)
    veh = Vehicle(CorsaC(), cfg)
    # Stage OPEN tracks 2.5 m past the line. track.surface_at() rejects a
    # longitudinal overshoot on an open track, so a car parked exactly at s = 0
    # has both rear contact patches 1.52 m behind the strip and they read as
    # grass: crr_scale 25 is 164 N.m of rolling drag per rear wheel and the
    # launch is wrong before the clutch bites. Closed tracks wrap and need none
    # of this.
    if start_s is None:
        start_s = 0.0 if tr.closed else 2.5
    x, y = trk.point_at(tr, start_s, 0.0)
    _, _, _, psi0, _ = trk.project(tr, x, y)
    veh.reset(x, y, psi0, V=start_V, gear=gear)

    telem = None
    if telem_path:
        meta = dict(dt=dt, track=tr.name, car=veh.car,
                    tyre_file=tlm.DEFAULT_TYRE_FILE,
                    wing=dict(wing=wing, x_w=x_w, h_w=h_w),
                    harness=dict(DT_PHYS=dt, FPS=FPS, MAX_SUBSTEPS=MAX_SUBSTEPS,
                                 MAX_FRAME_DT=MAX_FRAME_DT,
                                 SURFACE_LOOKUP_HZ=SURFACE_LOOKUP_HZ,
                                 global_wet=global_wet, radius=radius, tag=tag),
                    cmdline=list(cmdline or sys.argv))
        telem = tlm.TelemetryWriter(telem_path, hz=telem_hz,
                                    precision=precision, meta=meta)

    inp = _scripted(driver if driver is not None
                    else (lambda t, v, T: Controls()))
    return Sim(veh, tr, inp, renderer=None, telem=telem, dt=dt,
               wing=wing, global_wet=global_wet)


# ---- accel ---------------------------------------------------------------
def accel_script(opts) -> dict:
    """WOT from rest. The gearbox shifts itself (powertrain.update_shift)."""
    drv = StraightDriver(throttle=1.0, wing_on=(opts.wing != "off"))
    sim = _build("dragstrip", wing=opts.wing, x_w=opts.wing_x, h_w=opts.wing_h,
                 wet=opts.wet, dt=opts.dt, telem_path=opts.telemetry,
                 telem_hz=opts.telem_hz, precision=opts.telem_precision,
                 driver=drv, tag="accel")
    trace = []
    n = int(round(opts.duration / sim.dt))
    sim._log(0)
    for _ in range(n):
        sim.step_physics(sim.dt)
        trace.append((sim.t, hypot(sim.veh.u, sim.veh.v), sim.s,
                      sim.veh.gear, sim.veh.rpm))
    if sim.telem:
        sim.telem.close()

    t100 = float("nan")
    gear100 = rpm100 = float("nan")
    for (t0, v0, _s0, _g0, _r0), (t1, v1, _s1, g1, r1) in zip(trace, trace[1:]):
        if v0 < 27.7778 <= v1:
            f = (27.7778 - v0) / (v1 - v0)
            t100 = t0 + f * (t1 - t0)
            gear100, rpm100 = g1, r1
            break
    v_end = trace[-1][1]
    v_1500 = float("nan")
    for (t0, v0, s0, *_), (t1, v1, s1, *_) in zip(trace, trace[1:]):
        if s0 < 1500.0 <= s1 and sim.track.length >= 1500.0:
            v_1500 = v0 + (v1 - v0) * (1500.0 - s0) / max(s1 - s0, 1e-9)
            break
    return dict(script="accel", t_0_100_s=t100, gear_at_100=gear100,
                rpm_at_100=rpm100, V_end=v_end, V_at_1500m=v_1500,
                s_end=trace[-1][2], csv=opts.telemetry)


# ---- brake ---------------------------------------------------------------
class BrakeDriver(StraightDriver):
    """Accelerate to 100 km/h, then full brake. Distance measured from trigger.

    The lane keeper stays live through the stop, which is what makes the run
    survive split-mu: with the two right-hand wheels on a 0.632 patch the car
    yaws into the wet side and an open-loop brake script leaves the strip.
    """

    def __init__(self, v_trigger=100 / 3.6, pedal=1.0):
        super().__init__(throttle=1.0)
        self.v_trigger = v_trigger
        self.pedal = pedal
        self.braking = False
        self.t0 = None
        self.p0 = None
        self.v0 = None
        self.t_stop = None
        self.dist = float("nan")
        self.peak_g = 0.0
        self.kappa_min = 0.0

    def __call__(self, t, veh, tr):
        V = hypot(veh.u, veh.v)
        if not self.braking and V >= self.v_trigger:
            self.braking = True
            self.t0, self.p0, self.v0 = t, (veh.x, veh.y), V
        if self.braking:
            if V > 0.10:
                self.dist = hypot(veh.x - self.p0[0], veh.y - self.p0[1])
                self.peak_g = max(self.peak_g, -veh.ax / G)
                self.kappa_min = min(self.kappa_min, min(float(k) for k in veh.kappa))
            elif self.t_stop is None:
                self.t_stop = t
            self.thr, self.brk, self.clu = 0.0, self.pedal, 1.0
        else:
            self.thr, self.brk, self.clu = 1.0, 0.0, 0.0
        return super().__call__(t, veh, tr)


def brake_script(opts) -> dict:
    drv = BrakeDriver(pedal=1.0)
    sim = _build("dragstrip", wing=opts.wing, wet=opts.wet, dt=opts.dt,
                 telem_path=opts.telemetry, telem_hz=opts.telem_hz,
                 precision=opts.telem_precision, driver=drv, tag="brake")
    sim.run_headless(opts.duration)
    if sim.telem:
        sim.telem.close()
    return dict(script="brake", distance_m=drv.dist,
                v_trigger=drv.v0, peak_decel_g=drv.peak_g,
                kappa_min=drv.kappa_min,
                t_brake=drv.t0, t_stop=drv.t_stop, csv=opts.telemetry)


# ---- skidpad limit -------------------------------------------------------
def _skidpad_attempt(V_tgt, radius, wing, x_w, h_w, wet, dt, cw=False,
                     window=8.0, telem_path=None, telem_hz=TELEM_HZ,
                     precision="6g", mu_scale=1.0):
    """One 8 s constant-speed window. Returns (held, diagnostics)."""
    drv = PathFollower(V_tgt, wing_on=(wing != "off"))
    sim = _build("skidpad", radius=radius, cw=cw, wing=wing, x_w=x_w, h_w=h_w,
                 wet=wet, dt=dt, telem_path=telem_path, telem_hz=telem_hz,
                 precision=precision, driver=drv, mu_scale=mu_scale,
                 start_V=V_tgt, gear=3, tag=f"skid{radius:g}")
    n = int(round(window / dt))
    sim._log(0)
    held = True
    # A single sample at the last step is not a measurement: the path follower
    # is still working, so util_f/util_r dither by a few percent and can even
    # come out with the wrong SIGN of difference between two configurations.
    # Average over the same last third the speed is judged on.
    tail_from = int(0.66 * n)
    acc = dict(uf=0.0, ur=0.0, ay=0.0, fw=0.0, k=0)
    for i in range(n):
        sim.step_physics(sim.dt)
        if i >= tail_from:
            acc["uf"] += sim.veh.util_f
            acc["ur"] += sim.veh.util_r
            acc["ay"] += abs(sim.veh.ay)
            acc["fw"] += abs(sim.veh.F_wing)
            acc["k"] += 1
        if abs(sim.n_lat) > N_DEPART or abs(degrees(sim.veh.beta)) > BETA_SPIN_DEG:
            held = False
            break
        if not math.isfinite(sim.veh.u):
            held = False
            break
    # judge on the last third of the window, once the transient has washed out
    tail = drv.V_hist[int(0.66 * len(drv.V_hist)):] or [0.0]
    V_mean = sum(tail) / len(tail)
    n_tail = drv.n_hist[int(0.66 * len(drv.n_hist)):] or [0.0]
    n_mean = sum(n_tail) / len(n_tail)
    if held and V_mean < V_tgt - 0.35:
        held = False               # the car could not even reach the target
    k = max(acc["k"], 1)
    diag = dict(V_cmd=V_tgt, V_mean=V_mean, n_mean_tail=n_mean,
                max_n=drv.max_n,
                max_beta=drv.max_beta, ay_g=acc["ay"] / k / G,
                util_f=acc["uf"] / k, util_r=acc["ur"] / k,
                util_f_final=sim.veh.util_f, util_r_final=sim.veh.util_r,
                limited_by=sim.veh.limited_by, F_wing=acc["fw"] / k,
                steps=i + 1)
    if sim.telem:
        sim.telem.close()
    return held, diag


def skidpad_limit_script(opts, wing=None, telem_path=None) -> dict:
    """Bisect the sustainable speed at a fixed radius over 8 s windows.

    NOTE, and it is the spec's own note: a closed-loop skidpad controller
    saturates BELOW the car (CONTRACT section 4). The reference limit comes from
    vehicle.ramp_steer / steady_state_corner; this script measures what a driver
    can hold, which is a different and smaller number. Both are reported.
    """
    wing = opts.wing if wing is None else wing
    R = opts.radius
    lo, hi = 5.0, 1.25 * sqrt(AY_MAX_DRY * R) + 2.0
    ok_lo = False
    diag_lo = {}
    for _ in range(6):
        held, d = _skidpad_attempt(lo, R, wing, opts.wing_x, opts.wing_h,
                                   opts.wet, opts.dt, opts.cw)
        if held:
            ok_lo, diag_lo = True, d
            break
        lo *= 0.6
    if not ok_lo:
        return dict(script="skidpad_limit", radius=R, wing=wing,
                    V_limit=float("nan"), note="no holding speed found")
    for _ in range(12):                       # 12 bisections -> ~5 mm/s
        mid = 0.5 * (lo + hi)
        held, d = _skidpad_attempt(mid, R, wing, opts.wing_x, opts.wing_h,
                                   opts.wet, opts.dt, opts.cw)
        if held:
            lo, diag_lo = mid, d
        else:
            hi = mid
    # final holding run, logged
    tp = telem_path if telem_path is not None else opts.telemetry
    held, diag = _skidpad_attempt(lo, R, wing, opts.wing_x, opts.wing_h,
                                  opts.wet, opts.dt, opts.cw,
                                  telem_path=tp, telem_hz=opts.telem_hz,
                                  precision=opts.telem_precision)
    out = dict(script="skidpad_limit", radius=R, wing=wing, V_limit=lo,
               csv=tp, **diag)
    try:
        from .vehicle import steady_state_corner
        ref = steady_state_corner(R, cfg=VehicleConfig(wing=wing, x_w=opts.wing_x,
                                                       h_w=opts.wing_h))
        out["V_openloop_ref"] = ref["V"]
        out["ay_g_openloop_ref"] = ref["ay_g"]
        out["limited_by_openloop"] = ref["limiting"].upper()
    except Exception as exc:                    # never let a rig break a script
        out["V_openloop_ref"] = float("nan")
        out["openloop_error"] = repr(exc)
    kf = 0.5 * RHO * 0.35 * {"off": 0.0, "fin": 0.70, "plate": 1.25}[wing]
    out["V_qss"] = qss.corner_speed(R, k=kf, x_w=opts.wing_x, h_w=opts.wing_h,
                                    power_cap=False)[0]
    return out


# ---- wing A/B ------------------------------------------------------------
def wing_ab_script(opts) -> dict:
    """Paired constant-radius runs, wing off then wing on, at the SAME radius.

    The skidpad is the right place for this: the gain is a couple of percent of
    corner speed and on a lap it is buried under whatever the virtual driver
    does differently on the two runs.

    WHICH NUMBER IS THE ANSWER, and this is the whole point of the function.
    The headline `gain_pct` is the OPEN-LOOP ramp-steer gain in qss-parity mode,
    because that is the only one comparable with crossover.gain and qss. The
    closed-loop skidpad bisection is reported too, clearly labelled, and must
    NOT be read as a measurement: the path-following controller saturates about
    10% below the car (measured 26.27 m/s against 29.30 at R = 100 m), and it
    does not saturate at the same place with the wing on as with it off, so its
    A/B difference is contaminated by the controller and came out at +6.9% --
    nearly three times the real figure and above every physical cap.

    Three open-loop numbers are reported, and the spread between them IS the
    result:
      parity     Cs_psi = 0 and dCLda = 0. Directly comparable to crossover.
      fixed CL   the car's own body side force on, device CL held at CL0.
      drive      the device's CL also grows with body slip (dCLda = 2.47/rad).

    util_r is printed beside every one of them, because a gain above the
    understeer-margin cap sqrt(mu_r/mu_f) - 1 is only ever reachable by making
    the REAR axle the limiting one, and on a FWD hatch that is a spin, not a
    lap time. crossover.q3_gain asks for exactly this assertion.
    """
    from .vehicle import steady_state_corner

    base = opts.telemetry or tlm.run_path("skidpad", tag="wingab")
    stem, ext = os.path.splitext(base)
    on_wing = opts.wing if opts.wing != "off" else "fin"
    R, xw, hw = opts.radius, opts.wing_x, opts.wing_h
    mu = 0.55 / 0.87 if opts.wet == "all" else 1.0

    def _ol(wing, parity, dclda=None):
        cfg = VehicleConfig(qss_parity=parity, wing=wing, x_w=xw, h_w=hw,
                            mu_scale=mu)
        if dclda is not None:
            cfg.dCLda = dclda
        return steady_state_corner(R, cfg=cfg)

    modes = {}
    for tag, parity, dclda in (("parity", True, None),
                               ("fixed_CL", False, 0.0),
                               ("drive", False, None)):
        off = _ol("off", parity, dclda)
        on = _ol(on_wing, parity, dclda)
        # sqrt(mu_r/mu_f) - 1, read off the BASELINE's own axle utilisations:
        # as the front is relieved the rear becomes limiting, and that bounds
        # the whole study.
        cap = sqrt(off["util_f"] / off["util_r"]) - 1.0
        g = 100.0 * (on["V"] / off["V"] - 1.0)
        modes[tag] = dict(
            V_off=off["V"], V_on=on["V"], gain_pct=g,
            util_f_on=on["util_f"], util_r_on=on["util_r"],
            limiting_off=off["limiting"], limiting_on=on["limiting"],
            beta_on_deg=on["beta_deg"], CL_on=on["CL_dev"], F_dev=on["F_dev"],
            cap_pct=100.0 * cap, above_cap=g > 100.0 * cap,
            rear_limited=(on["limiting"] == "rear"))

    # what a driver can actually hold. Reported, never used as the answer.
    r_off = skidpad_limit_script(opts, wing="off",
                                 telem_path=f"{stem}_limit_off{ext}")
    r_on = skidpad_limit_script(opts, wing=on_wing,
                                telem_path=f"{stem}_limit_on{ext}")
    V0, V1 = r_off["V_limit"], r_on["V_limit"]
    g_cl = (V1 / V0 - 1.0) * 100.0 if V0 else float("nan")

    # THE TELEMETRY PAIR, and it is deliberately MATCHED SPEED. Logging each
    # config at its own closed-loop limit would make plots.compare show the
    # controller's saturation difference, not the device's effect. Both runs
    # held at the wing-off limit instead: the speed trace is then identical by
    # construction and every remaining difference -- util_r, body slip, the
    # steer the driver has to wind on -- is the device and nothing else.
    matched = {}
    for tag, w in (("off", "off"), ("on", on_wing)):
        held, d = _skidpad_attempt(V0, R, w, xw, hw, opts.wet, opts.dt, opts.cw,
                                   telem_path=f"{stem}_{tag}{ext}",
                                   telem_hz=opts.telem_hz,
                                   precision=opts.telem_precision,
                                   mu_scale=mu)
        matched[tag] = dict(held=held, **d)

    kf = 0.5 * RHO * 0.35 * {"off": 0.0, "fin": 0.70, "plate": 1.25}[on_wing]
    return dict(
        script="wing_ab", radius=R, wing=on_wing, x_w=xw, h_w=hw,
        # THE answer
        gain_pct=modes["parity"]["gain_pct"],
        V_off=modes["parity"]["V_off"], V_on=modes["parity"]["V_on"],
        F_wing=modes["parity"]["F_dev"],
        util_r_on=modes["parity"]["util_r_on"],
        limiting_on=modes["parity"]["limiting_on"],
        cap_pct=modes["parity"]["cap_pct"],
        # the same thing with the car's own aero, and with the device's own
        # incidence bonus, which crossover does not model at all
        gain_pct_fixed_CL=modes["fixed_CL"]["gain_pct"],
        gain_pct_drive=modes["drive"]["gain_pct"],
        rear_limited_fixed_CL=modes["fixed_CL"]["rear_limited"],
        rear_limited_drive=modes["drive"]["rear_limited"],
        modes=modes,
        # the references
        V_off_qss=r_off.get("V_qss"), V_on_qss=r_on.get("V_qss"),
        gain_pct_qss=100.0 * (r_on["V_qss"] / r_off["V_qss"] - 1.0),
        gain_pct_closed_form=100.0 * (crossover_gain(kf, R, xw) or float("nan")),
        # drivability only -- NOT a measurement
        V_off_closedloop=V0, V_on_closedloop=V1,
        gain_pct_closedloop_DRIVABILITY_ONLY=g_cl,
        # matched-speed pair: this is what plots.compare should be pointed at
        matched_V=V0,
        matched_util_r_off=matched["off"]["util_r"],
        matched_util_r_on=matched["on"]["util_r"],
        matched_util_f_off=matched["off"]["util_f"],
        matched_util_f_on=matched["on"]["util_f"],
        csv_off=f"{stem}_off{ext}", csv_on=f"{stem}_on{ext}",
        csv_limit_off=r_off.get("csv"), csv_limit_on=r_on.get("csv"))


def crossover_gain(k, R, x_w):
    """crossover.gain, imported lazily so a broken analysis file cannot stop a run."""
    try:
        import crossover
        return crossover.gain(k, R, x_w)
    except Exception:
        return None


# ---- ramp probe ----------------------------------------------------------
class RampProbe:
    """Reproduces spec eq.3 and eq.4 and records what they produce.

    This is the harness-side copy of the input ramps, run at DT_PHYS with a
    scripted key pattern, so V15/V18 can be checked with no window and no
    keyboard. When input.py is importable its `steer_limit_deg` is checked too;
    the RAMPS themselves are re-derived here rather than reached into, because
    the point of the probe is to state what the numbers should be.
    """

    STEER_RATIO = 16.0
    W_DRIVE = 900.0 / STEER_RATIO           # 56.25 deg/s at the road wheel

    def __init__(self):
        self.delta = 0.0                    # deg, road wheel
        self.thr = 0.0
        self.brk = 0.0
        self.log: list = []

    @staticmethod
    def _ramp(x, tgt, rise, fall, dt):
        return min(tgt, x + rise * dt) if tgt > x else max(tgt, x - fall * dt)

    def _w_ret(self, V):
        return (720.0 + 720.0 * min(max(V / 22.0, 0.0), 1.0)) / self.STEER_RATIO

    def __call__(self, t, veh, tr):
        dt = 0.001
        V = hypot(veh.u, veh.v)
        # key pattern: RIGHT 0.0-1.0 s, released 1.0-1.6 s;
        #              throttle 1.6-2.4 s, released 2.4-3.0 s;
        #              brake 3.0-3.6 s, released 3.6-4.2 s
        right = (t < 1.0)
        up = (1.6 <= t < 2.4)
        down = (3.0 <= t < 3.6)

        d_in = -1.0 if right else 0.0       # RIGHT is -delta by the ISO sign
        if d_in == 0.0:
            wr = self._w_ret(V)
            if abs(self.delta) <= wr * dt:
                self.delta = 0.0
            else:
                self.delta -= math.copysign(wr * dt, self.delta)
        elif d_in * self.delta < 0.0:
            self.delta += d_in * (self.W_DRIVE + self._w_ret(V)) * dt
        else:
            self.delta += d_in * self.W_DRIVE * dt
        self.delta = min(max(self.delta, -DELTA_LOCK_DEG), DELTA_LOCK_DEG)

        self.thr = self._ramp(self.thr, 1.0 if up else 0.0, 3.5, 6.0, dt)
        self.brk = self._ramp(self.brk, 1.0 if down else 0.0, 5.0, 8.0, dt)
        self.log.append((t, self.delta, self.thr, self.brk))
        return Controls(delta=radians(self.delta), throttle=self.thr,
                        brake=self.brk, auto_gearbox=True)


def ramp_probe_script(opts) -> dict:
    drv = RampProbe()
    sim = _build("dragstrip", wet="none", dt=opts.dt,
                 telem_path=opts.telemetry, telem_hz=opts.telem_hz,
                 precision=opts.telem_precision, driver=drv, tag="ramp")
    sim.run_headless(min(opts.duration, 5.0))
    if sim.telem:
        sim.telem.close()

    def at(tq):
        i = min(int(round(tq / sim.dt)) - 1, len(drv.log) - 1)
        return drv.log[max(i, 0)]

    d200 = at(0.200)[1]
    t_lock = next((r[0] for r in drv.log if abs(r[1]) >= DELTA_LOCK_DEG - 1e-9),
                  float("nan"))
    t_thr_up = next((r[0] - 1.6 for r in drv.log if r[0] >= 1.6 and r[2] >= 1.0),
                    float("nan"))
    t_thr_dn = next((r[0] - 2.4 for r in drv.log if r[0] >= 2.4 and r[2] <= 0.0),
                    float("nan"))
    t_brk_up = next((r[0] - 3.0 for r in drv.log if r[0] >= 3.0 and r[3] >= 1.0),
                    float("nan"))
    out = dict(script="ramp_probe", delta_at_0p2s_deg=abs(d200),
               t_to_lock_s=t_lock, t_throttle_0_1=t_thr_up,
               t_throttle_1_0=t_thr_dn, t_brake_0_1=t_brk_up,
               csv=opts.telemetry)
    try:
        from .input import steer_limit_deg
        out["delta_lim_20"] = steer_limit_deg(20.0, 0.0)
        out["delta_lim_30"] = steer_limit_deg(30.0, 0.0)
        out["delta_lim_30_beta12"] = steer_limit_deg(30.0, 12.0)
    except Exception:
        out["input_module"] = "not importable yet - limiter table not checked"
    return out


# ---- a driven lap (extra; the arena driver the wing study needs) ---------
def lap_script(opts) -> dict:
    drv = LapDriver(trk.make_arena(surfaces=(opts.wet != "none")),
                    margin=opts.margin, wing_on=(opts.wing != "off"),
                    global_wet=(MU_WET_SCALE if opts.wet == "all" else 1.0))
    sim = _build("arena", wing=opts.wing, x_w=opts.wing_x, h_w=opts.wing_h,
                 wet=opts.wet, dt=opts.dt, telem_path=opts.telemetry,
                 telem_hz=opts.telem_hz, precision=opts.telem_precision,
                 driver=drv, start_V=25.0, gear=3, tag="lap")
    sim.run_headless(opts.duration)
    if sim.telem:
        sim.telem.close()
    laps = [e for e in sim.events_log if e[0] == "lap"]
    return dict(script="lap", laps=len(laps),
                lap_times=[round(e[3], 4) for e in laps],
                best=min([e[3] for e in laps], default=float("nan")),
                max_n=drv.max_n, csv=opts.telemetry)


# ---- driveability probe --------------------------------------------------
# THE ONE SCRIPTED PATH THAT DELIBERATELY SWITCHES THE DRIVER AIDS ON.
#
# CONTRACT section 9 item 9 says the aids never enter a measurement, and that
# still holds: nothing below is an acceptance number, validate.py does not call
# any of it, and every figure it prints is labelled as a DRIVEABILITY figure.
# But an aid that cannot be measured cannot be tuned either, and the owner's
# bug report ("no acceleration while steering, no steering while braking") is a
# statement about the aids at HIS settings -- ENGINE sport (power_scale 2.0),
# ABS on, TC on, steer aid on -- so the probe has to be able to reproduce that
# car. It therefore takes the aid state as arguments and runs BOTH input paths:
#
#   'kb'  -> input.KeyboardInput driven by a synthetic key state through
#            set_keys(), steer_limit as the settings file has it. This is the
#            human path: the ramps, the limiter and the fine modifier all run.
#   'raw' -> the local ScriptedInput with delta commanded directly. No aid in
#            the input layer at all, so a difference between the two rows is an
#            input-layer effect and a difference within one row is physics.
#
# Everything here is pure in (state, dt) like the rest of the scripts: the key
# pattern is a function of t only and no wall clock is read.

class _KeyPattern:
    """A synthetic keyboard, driven through KeyboardInput exactly as a human is.

    `pattern(t) -> dict(up=, down=, left=, right=, fine=, ...)`. The object is
    NOT a driver function: Sim would hand it (t, veh, track) and KeyboardInput
    wants set_keys() + update(dt, V, beta, rpm, gear), so the probe loops by
    hand instead of going through Sim. That is deliberate -- Sim's job is the
    accumulator, and the probe needs the input layer, not the loop.
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, t):
        return self.fn(t)


def _probe_vehicle(power_scale=1.0, tc=False, abs_on=False, wet="none",
                   start_V=0.0, gear=1, track_name="open"):
    """A bare Vehicle staged on a track, with the aid flags the probe asks for."""
    tr = trk.make_track(track_name, 50.0, False, surfaces=(wet != "none"))
    cfg = VehicleConfig(power_scale=power_scale, tc_on=tc, abs_on=abs_on)
    veh = Vehicle(CorsaC(), cfg)
    start_s = 0.0 if tr.closed else 2.5
    x, y = trk.point_at(tr, start_s, 0.0)
    _, _, _, psi0, _ = trk.project(tr, x, y)
    veh.reset(x, y, psi0, V=start_V, gear=gear)
    return veh, tr


def _probe_row(t, veh, ctl, kb):
    """One sample. The column set IS the bug report: load, gain, slip, yaw."""
    return dict(
        t=t, V=hypot(veh.u, veh.v), u=veh.u, ax=veh.ax, ay=veh.ay, r=veh.r,
        x=veh.x, y=veh.y, psi=veh.psi, beta_deg=degrees(veh.beta),
        throttle=ctl.throttle, brake=ctl.brake,
        delta_cmd_deg=degrees(ctl.delta),
        delta_wheel_deg=degrees(veh.delta_wheel[0]),
        delta_lim_deg=(kb.delta_lim_deg if kb is not None else DELTA_LOCK_DEG),
        eng_load=veh.eng_load, tc_gain=veh.tc_gain, tc_active=veh.tc_active,
        kappa_fl=float(veh.kappa[0]), kappa_fr=float(veh.kappa[1]),
        alpha_fl_deg=degrees(veh.alpha[0]), alpha_fr_deg=degrees(veh.alpha[1]),
        Fz_fl=float(veh.Fz[0]), Fz_fr=float(veh.Fz[1]),
        Fy_f=float(veh.Fy[0] + veh.Fy[1]), Fx_f=float(veh.Fx[0] + veh.Fx[1]),
        util_f=veh.util_f, util_r=veh.util_r,
        abs_n=int(veh.abs_active[0]) + int(veh.abs_active[1]),
        gear=veh.gear, rpm=veh.rpm)


def _probe_run(T, *, keys=None, delta_fn=None, power_scale=1.0, tc=False,
               abs_on=False, steer_limit=True, start_V=0.0, gear=1,
               dt=DT_PHYS, log_hz=50.0, wet="none"):
    """Run one probe case and return its sample list.

    Exactly one of `keys` (the KeyboardInput path) or `delta_fn` (the raw path,
    `delta_fn(t, veh) -> Controls`) is given.
    """
    veh, _tr = _probe_vehicle(power_scale, tc, abs_on, wet=wet,
                              start_V=start_V, gear=gear)
    kb = None
    if keys is not None:
        from .input import KeyboardInput, K_US_DEG_MEASURED
        kb = KeyboardInput(steer_limit=steer_limit, k_us_deg=K_US_DEG_MEASURED)
    n = int(round(T / dt))
    stride = max(1, int(round(1.0 / (log_hz * dt))))
    rows = []
    for i in range(n):
        t = i * dt
        if kb is not None:
            kb.set_keys(**keys(t))
            V = hypot(veh.u, veh.v)
            beta_deg = degrees(atan2(veh.v, max(abs(veh.u), 0.5)))
            ctl = kb.update(dt, V, beta_deg, veh.rpm, veh.gear)
        else:
            ctl = delta_fn(t, veh)
        veh.step(ctl, (1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0), dt)
        if i % stride == 0:
            rows.append(_probe_row(t, veh, ctl, kb))
    return rows


def _mean(rows, key, t0=None, t1=None):
    sel = [r for r in rows
           if (t0 is None or r["t"] >= t0) and (t1 is None or r["t"] < t1)]
    return sum(r[key] for r in sel) / len(sel) if sel else float("nan")


def _step_delta(dmax_deg, throttle=0.0, brake=0.0, clutch=0.0, t_step=1.0):
    """The raw path's twin of holding an arrow key: the SAME 56.25 deg/s ramp."""
    def fn(t, veh):
        d = min(dmax_deg, W_DRIVE_DEG_PROBE * max(0.0, t - t_step))
        return Controls(delta=radians(d), throttle=throttle, brake=brake,
                        clutch=clutch, auto_gearbox=True)
    return fn


W_DRIVE_DEG_PROBE = 900.0 / 16.0        # 56.25; the input layer's hand rate


def _probe_accel_while_steering(opts, tc, power_scale, V0, gear, angles):
    """Symptom (b): does the car still accelerate with the wheel turned?

    Full throttle throughout, a step steer wound on at the keyboard's own rate,
    and the answer is `mean ax` once the corner has settled. The TC columns are
    the diagnosis: on a FWD car the INSIDE front unloads and spins, so a TC
    that reads max(kx_FL, kx_FR) cuts the engine for a wheel that is light, not
    for a car that is out of traction.
    """
    out = []
    for dmax in angles:
        rows = _probe_run(5.0, delta_fn=_step_delta(dmax, throttle=1.0),
                          power_scale=power_scale, tc=tc, abs_on=False,
                          start_V=V0, gear=gear, dt=opts.dt)
        out.append(dict(
            delta_deg=dmax, tc=tc,
            mean_ax=_mean(rows, "ax", 3.0), V_end=rows[-1]["V"],
            ay=_mean(rows, "ay", 3.0),
            eng_load=_mean(rows, "eng_load", 3.0),
            tc_gain=_mean(rows, "tc_gain", 3.0),
            kappa_fl=_mean(rows, "kappa_fl", 3.0),
            kappa_fr=_mean(rows, "kappa_fr", 3.0),
            Fz_fl=_mean(rows, "Fz_fl", 3.0), Fz_fr=_mean(rows, "Fz_fr", 3.0),
            util_f=_mean(rows, "util_f", 3.0)))
    return out


def _probe_steer_while_braking(opts, abs_on, V0, gear, angles):
    """Symptom (c): does the car still turn with the brake buried?

    The number that matters is the SETTLED yaw rate against the commanded
    angle. A car whose yaw rate PEAKS partway through the available lock and
    then falls is a car whose front axle is past the tyre peak: adding lock
    subtracts turning, which is exactly what "I can't steer while braking"
    feels like from the driver's seat.
    """
    out = []
    for dmax in angles:
        rows = _probe_run(2.5, delta_fn=_step_delta(dmax, brake=1.0, clutch=1.0),
                          power_scale=1.0, tc=False, abs_on=abs_on,
                          start_V=V0, gear=gear, dt=opts.dt)
        out.append(dict(
            delta_deg=dmax, abs_on=abs_on,
            r=_mean(rows, "r", 1.5, 2.0), ay=_mean(rows, "ay", 1.5, 2.0),
            alpha_f=_mean(rows, "alpha_fl_deg", 1.5, 2.0),
            beta=_mean(rows, "beta_deg", 1.5, 2.0),
            Fy_f=_mean(rows, "Fy_f", 1.5, 2.0),
            delta_wheel=_mean(rows, "delta_wheel_deg", 1.5, 2.0),
            V_end=rows[-1]["V"],
            abs_n=max(r["abs_n"] for r in rows)))
    return out


def _probe_limit_table():
    """Symptom (a): what the aid actually allows, and how fast it gets there.

    `lim_b20` is the symmetric `steer_limit_deg`, i.e. what the OLD clamp
    allowed in EITHER direction at 20 deg of body slip. `into_b20` / `out_b20`
    are the directional pair the clamp now uses: the bonus all goes to the
    counter-steer side and lock into the slide stays at the floor.
    """
    from .input import (steer_limit_deg, steer_limit_pair_deg, _return_rate_deg,
                        K_US_DEG_MEASURED)
    rows = []
    for V in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0):
        d0 = steer_limit_deg(V, 0.0, k_us_deg=K_US_DEG_MEASURED)
        # beta < 0 is a left-turn slide: the catch is RIGHT lock, so the RIGHT
        # member of the pair is the counter-steer side and the LEFT one is lock
        # into the slide.
        into20, out20 = steer_limit_pair_deg(V, -20.0,
                                             k_us_deg=K_US_DEG_MEASURED)
        rows.append(dict(
            V=V, lim_b0=d0,
            lim_b10=steer_limit_deg(V, 10.0, k_us_deg=K_US_DEG_MEASURED),
            lim_b20=steer_limit_deg(V, 20.0, k_us_deg=K_US_DEG_MEASURED),
            into_b20=into20, out_b20=out20,
            t_to_lim=d0 / W_DRIVE_DEG_PROBE,
            w_return=_return_rate_deg(V),
            t_to_centre=d0 / _return_rate_deg(V)))
    return rows


def _probe_keyboard_brake_steer(opts, power_scale):
    """Symptoms (a) + (c) through the REAL aid: DOWN + LEFT held from 30 m/s.

    The raw-path sweep above says where yaw response peaks; this says where the
    aid actually parks the driver, which is the half he can feel. `delta_max`
    is the most lock the clamp ever handed over, and THE ANSWER IS `dpsi`, the
    total heading change over the 3 s -- not the instantaneous yaw rate. A car
    dragging 22 deg of lock at alpha_f = 32 deg has a high BODY yaw rate
    because it is pirouetting, and scoring `r` alone calls that good steering.
    Measured through this very case, aid on:

        symmetric beta bonus   delta_max 22.32   dpsi 37.98 deg   lateral 13.67 m
        directional bonus      delta_max 14.08   dpsi 41.10 deg   lateral 13.77 m

    i.e. 8.2 deg LESS lock buys 3.1 deg MORE heading change. That is the whole
    of symptom (a): the old aid let the driver spend lock on slip angle.
    """
    keys = lambda t: dict(down=(t >= 0.5), left=(t >= 0.5))
    out = []
    for aid in (True, False):
        rows = _probe_run(3.0, keys=keys, power_scale=power_scale, tc=True,
                          abs_on=True, steer_limit=aid, start_V=30.0, gear=5,
                          dt=opts.dt, log_hz=100.0)
        r_abs = [abs(r["r"]) for r in rows]
        p0, pn = rows[0], rows[-1]
        dx, dy = pn["x"] - p0["x"], pn["y"] - p0["y"]
        cs, sn = cos(p0["psi"]), sin(p0["psi"])
        out.append(dict(
            steer_aid=aid,
            delta_max=max(abs(r["delta_cmd_deg"]) for r in rows),
            delta_end=abs(rows[-1]["delta_cmd_deg"]),
            lim_end=rows[-1]["delta_lim_deg"],
            # what the car actually did with the lock it was given
            dpsi=degrees(pn["psi"] - p0["psi"]),
            lateral=abs(-dx * sn + dy * cs), longitudinal=dx * cs + dy * sn,
            r_peak=max(r_abs), r_end=abs(rows[-1]["r"]),
            alpha_f_max=max(abs(r["alpha_fl_deg"]) for r in rows),
            beta_max=max(abs(r["beta_deg"]) for r in rows),
            V_end=rows[-1]["V"]))
    return out


def _probe_keyboard_accel_steer(opts, power_scale):
    """Symptom (b) through the REAL keyboard: UP alone vs UP + RIGHT held."""
    out = []
    for label, held in (("up", dict(up=True)),
                        ("up+right", dict(up=True, right=True))):
        rows = _probe_run(3.0, keys=(lambda t, h=held: dict(h)),
                          power_scale=power_scale, tc=True, abs_on=True,
                          steer_limit=True, start_V=20.0, gear=4, dt=opts.dt,
                          log_hz=100.0)
        out.append(dict(keys=label, ax=_mean(rows, "ax", 2.0),
                        V_end=rows[-1]["V"],
                        tc_gain=_mean(rows, "tc_gain", 2.0),
                        eng_load=_mean(rows, "eng_load", 2.0),
                        delta=rows[-1]["delta_cmd_deg"], r=rows[-1]["r"]))
    return out


def _probe_keyboard_combos(opts):
    """Does the input layer lose one of two simultaneous presses?

    up+right, down+right and the LSHIFT fine modifier, each held for 1.5 s
    through the real KeyboardInput. If both axes move, the layer is innocent.
    """
    out = []
    combos = (("up+right", dict(up=True, right=True)),
              ("down+right", dict(down=True, right=True)),
              ("up+left+fine", dict(up=True, left=True, fine=True)),
              ("up only", dict(up=True)),
              ("right only", dict(right=True)))
    for label, held in combos:
        rows = _probe_run(1.5, keys=(lambda t, h=held: dict(h)),
                          power_scale=2.0, tc=True, abs_on=True,
                          steer_limit=True, start_V=20.0, gear=4, dt=opts.dt)
        last = rows[-1]
        out.append(dict(combo=label, throttle=last["throttle"],
                        brake=last["brake"],
                        delta_cmd=last["delta_cmd_deg"],
                        delta_lim=last["delta_lim_deg"],
                        ax=last["ax"], r=last["r"]))
    return out


def drive_probe_script(opts) -> dict:
    """DRIVEABILITY PROBE -- not a measurement. See the block comment above.

    Reproduces the owner's three symptoms as numbers at HIS settings
    (`runs/settings.json`: engine sport, ABS on, TC on, steer aid on) and, for
    each, the same case with the aid switched off so the raw physics is visible
    next to it. Writes the whole matrix to the run's JSON sidecar so a before /
    after diff is a file diff.
    """
    power_scale = ENGINE_SCALE[opts.engine] if opts.engine in ENGINE_SCALE else 1.0
    angles_a = (3.0, 6.0, 9.0, 14.0)
    angles_b = (2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 20.0, 32.0)

    accel = []
    for tc in (True, False):
        accel += _probe_accel_while_steering(opts, tc, power_scale, 15.0, 3,
                                             angles_a)
    brake = []
    for ab in (True, False):
        brake += _probe_steer_while_braking(opts, ab, 30.0, 5, angles_b)

    limits = _probe_limit_table()
    combos = _probe_keyboard_combos(opts)
    kb_brake = _probe_keyboard_brake_steer(opts, power_scale)
    kb_accel = _probe_keyboard_accel_steer(opts, power_scale)

    # the headline numbers: the three symptoms as one float each
    def acc(tc, d):
        return next(r for r in accel if r["tc"] is tc and r["delta_deg"] == d)
    r_peak = max(r["r"] for r in brake if r["abs_on"])
    r_lock = next(r["r"] for r in brake if r["abs_on"] and r["delta_deg"] == 32.0)
    lim30 = next(r for r in limits if r["V"] == 30.0)

    res = dict(
        script="drive_probe", engine=opts.engine, power_scale=power_scale,
        # (b) no acceleration while steering
        b_ax_tc_on_6deg=acc(True, 6.0)["mean_ax"],
        b_ax_tc_off_6deg=acc(False, 6.0)["mean_ax"],
        b_ax_tc_on_14deg=acc(True, 14.0)["mean_ax"],
        b_ax_tc_off_14deg=acc(False, 14.0)["mean_ax"],
        b_tc_gain_6deg=acc(True, 6.0)["tc_gain"],
        b_tc_gain_14deg=acc(True, 14.0)["tc_gain"],
        b_kappa_inside_6deg=acc(False, 6.0)["kappa_fl"],
        b_kappa_outside_6deg=acc(False, 6.0)["kappa_fr"],
        # (c) no steering while braking
        c_r_peak=r_peak, c_r_at_full_lock=r_lock,
        c_r_ratio=(r_lock / r_peak if r_peak else float("nan")),
        c_r_locked_wheels=max(r["r"] for r in brake if not r["abs_on"]),
        # (a) steering feel
        a_lim_30_beta0=lim30["lim_b0"], a_lim_30_beta20=lim30["lim_b20"],
        a_lim_30_into_b20=lim30["into_b20"], a_lim_30_out_b20=lim30["out_b20"],
        a_t_to_lim_30=lim30["t_to_lim"], a_t_to_centre_30=lim30["t_to_centre"],
        # the aid as the driver meets it: DOWN+LEFT from 30 m/s
        c_kb_delta_max=kb_brake[0]["delta_max"],
        c_kb_dpsi=kb_brake[0]["dpsi"], c_kb_lateral=kb_brake[0]["lateral"],
        c_kb_alpha_f_max=kb_brake[0]["alpha_f_max"],
        c_kb_delta_max_no_aid=kb_brake[1]["delta_max"],
        c_kb_dpsi_no_aid=kb_brake[1]["dpsi"],
        # ... and UP vs UP+RIGHT at 20 m/s
        b_kb_ax_straight=kb_accel[0]["ax"], b_kb_ax_steering=kb_accel[1]["ax"],
        b_kb_tc_gain_steering=kb_accel[1]["tc_gain"],
        accel_matrix=accel, brake_matrix=brake, limit_table=limits,
        key_combos=combos, kb_brake_steer=kb_brake, kb_accel_steer=kb_accel,
        csv=None)
    if opts.telemetry:
        # the matrix belongs beside the other runs, in the same JSON style
        base, _ = os.path.splitext(os.fspath(opts.telemetry))
        with open(base + ".json", "w") as fh:
            json.dump(res, fh, indent=2, default=str)
        res["csv"] = base + ".json"
    return res


SCRIPTS = {
    "skidpad_limit": skidpad_limit_script,
    "accel": accel_script,
    "brake": brake_script,
    "wing_ab": wing_ab_script,
    "ramp_probe": ramp_probe_script,
    "lap": lap_script,                 # extra: the closed-circuit driver
    "drive_probe": drive_probe_script, # DRIVEABILITY, aids ON; not a measurement
}


# ==================================================================== #
#  SELF-CHECK  (V20, V21, V23, V24, V25 + the accel run)               #
# ==================================================================== #
def _tmpdir():
    import tempfile
    return tempfile.mkdtemp(prefix="carsim_drive_")


class _Opts:
    """A stand-in for the argparse namespace, so scripts can be called directly."""

    def __init__(self, **kw):
        d = dict(track="arena", radius=50.0, cw=False, wet="patch", wing="off",
                 wing_x=0.97, wing_h=0.90, wing_inc=0.0, dt=DT_PHYS, duration=60.0,
                 telemetry=None, telem_hz=TELEM_HZ, telem_precision="6g",
                 margin=0.90, gearbox="auto", auto_gearbox=True, abs=False,
                 steer_limit=True, camera="car_up", engine="stock", tc=False,
                 sound="off")
        d.update(kw)
        self.__dict__.update(d)


def _v20_determinism(tmp, verbose=True):
    """Two identical scripted runs, full precision, must be byte-identical."""
    import hashlib
    paths = []
    for i in (0, 1):
        p = os.path.join(tmp, f"det{i}.csv")
        drv = PathFollower(18.0)
        sim = _build("skidpad", radius=50.0, wing="fin", wet="patch",
                     dt=DT_PHYS, telem_path=p, precision="full", driver=drv,
                     start_V=18.0, gear=3, cmdline=["determinism"])
        sim.run_headless(6.0)
        sim.telem.close()
        paths.append(p)
    h = [hashlib.sha256(open(p, "rb").read()).hexdigest() for p in paths]
    size = os.path.getsize(paths[0])
    ok = (h[0] == h[1])
    if verbose:
        print(f"  V20 determinism : sha256 {h[0][:16]} vs {h[1][:16]}  "
              f"({size} bytes, {'IDENTICAL' if ok else 'DIFFER'})")
    return ok, dict(sha=h[0], bytes=size)


def _v21_rtf(tmp, seconds=60.0, verbose=True):
    p = os.path.join(tmp, "rtf.csv")
    drv = LapDriver(trk.make_arena(), margin=0.88)
    sim = _build("arena", wet="patch", dt=DT_PHYS, telem_path=p,
                 driver=drv, start_V=25.0, gear=3)
    t0 = time.perf_counter()
    sim.run_headless(seconds)
    wall = time.perf_counter() - t0
    sim.telem.close()
    rtf = seconds / wall
    if verbose:
        print(f"  V21 real-time   : {seconds:.0f} s sim in {wall:.2f} s wall, "
              f"RTF {rtf:.2f} (need >= 2.0, target >= 3.0), "
              f"{sim.telem.rows_written} telemetry rows")
    return rtf >= 2.0, dict(wall_s=wall, rtf=rtf, rows=sim.telem.rows_written)


def _v23_accumulator(verbose=True):
    drv = StraightDriver(throttle=0.0)
    sim = _build("dragstrip", wet="none", dt=DT_PHYS, driver=drv)
    sim.pump(DT_PHYS)                       # consume the first-frame guard
    n0, d0 = sim.n, sim.dropped_s
    steps = sim.pump(1.5)                   # the synthetic stall
    dropped = sim.dropped_s - d0
    ran = sim.n - n0
    nxt = sim.pump(1.0 / 60.0)              # and the sim continues in real time
    ok = (steps == MAX_SUBSTEPS and ran == MAX_SUBSTEPS
          and abs(dropped - 1.46) < 0.005 and 15 <= nxt <= 18)
    if verbose:
        print(f"  V23 accumulator : 1.5 s stall -> {steps} steps "
              f"(MAX_SUBSTEPS {MAX_SUBSTEPS}), dropped_s +{dropped:.4f} s "
              f"(expect ~1.46), dropped_frames {sim.dropped_frames}, "
              f"next frame {nxt} steps")
    return ok, dict(steps=steps, dropped_s=dropped, next_frame_steps=nxt,
                    dropped_frames=sim.dropped_frames)


def _v24_first_frame(verbose=True):
    drv = StraightDriver(throttle=0.0)
    sim = _build("dragstrip", wet="none", dt=DT_PHYS, driver=drv)
    a = sim.pump(1.8)                       # macOS's first frame
    sim.reset(to_checkpoint=False)
    b = sim.pump(0.9)                       # the gap across a reset
    sim.paused = True
    sim.unpause()
    c = sim.pump(2.0)                       # the gap across an unpause
    ok = (a == 1 and b == 1 and c == 1)
    if verbose:
        print(f"  V24 first frame : init {a} step, after reset {b} step, "
              f"after unpause {c} step (need exactly 1 each)")
    return ok, dict(init=a, after_reset=b, after_unpause=c)


def _v25_lap_timing(verbose=True):
    """Constant 30 m/s along the CIRCUIT_ARENA centreline, through LapTimer.

    Driven kinematically -- see DEVIATION 5. The quantity under test is the
    sub-step interpolation, and a whole-step timer would show a quantisation
    error of up to 1 ms here, which is half the tolerance on its own.
    """
    tr = trk.make_arena()
    lt = LapTimer(tr)
    V, dt, L = 30.0, DT_PHYS, tr.length
    s0 = L - 30.0                    # stage 1 s before the line so the first
    n = int(round(3.6 * L / V / dt))  # crossing is a real wrap, not a cold start
    s_prev = s0
    for i in range(n):
        t_prev = i * dt
        s_now = (s0 + (i + 1) * V * dt) % L
        lt.update(t_prev, s_prev, s_now, dt)
        s_prev = s_now
    exact = L / V
    laps = [c for c in lt.crossings if c[1] == "lap"]
    starts = [c for c in lt.crossings if c[1] == "start"]
    times = [c[0] for c in starts + laps]
    measured = [b - a for a, b in zip(times, times[1:])]
    err = max(abs(m - exact) for m in measured) if measured else float("inf")
    ok = bool(measured) and err < 0.002
    if verbose:
        print(f"  V25 lap timing  : L/V = {exact:.6f} s, measured "
              f"{[round(m, 6) for m in measured]}, max error {err*1e6:.2f} us "
              f"(tol 2000 us)")
    return ok, dict(exact_s=exact, measured_s=measured, max_err_s=err)


def _accel_end_to_end(tmp, verbose=True):
    p = os.path.join(tmp, "accel.csv")
    o = _Opts(track="dragstrip", wet="none", duration=40.0, telemetry=p)
    res = accel_script(o)
    rows = sum(1 for _ in open(p)) - 1
    ok = (14.5 <= res["t_0_100_s"] <= 16.0) and rows > 0
    if verbose:
        print(f"  accel run       : 0-100 km/h {res['t_0_100_s']:.3f} s in gear "
              f"{res['gear_at_100']} at {res['rpm_at_100']:.0f} rpm; "
              f"V(40 s) {res['V_end']:.3f} m/s; s {res['s_end']:.1f} m; "
              f"{rows} CSV rows")
    res["csv_rows"] = rows
    return ok, res


def _v26_settings_and_menu(tmp, verbose=True):
    """Settings persistence + the ESC menu state machine, without a window.

    The menu logic is pure (drive/menu.py); the renderer is duck-typed by a
    namespace with a `cfg.mode`, which is all the settings page touches.
    """
    from types import SimpleNamespace
    import contextlib, io
    path = os.path.join(tmp, "settings.json")
    s = Settings(path=path)
    s.track, s.gearbox, s.abs, s.wet = "open", "clutch", False, "all"
    s.engine, s.tc, s.sound = "tuned", False, "low"
    s.save()
    back = Settings.load(path)
    rt_ok = (back.track, back.gearbox, back.abs, back.wet, back.camera,
             back.engine, back.tc, back.sound) == (
        "open", "clutch", False, "all", "car_up", "tuned", False, "low")
    bad = Settings(path=path)
    bad.track, bad.gearbox, bad.engine, bad.sound = "moon", "dsg", "v8", "11"
    bad.clamp()
    clamp_ok = (bad.track, bad.gearbox, bad.engine, bad.sound) == (
        "arena", "auto", ENGINE_DEFAULT, SOUND_DEFAULT)
    o = _Opts(track="skidpad", gearbox=None, abs=None, steer_limit=None,
              wet=None, camera=None, engine=None, tc=None, sound="off")
    back.apply_cli(o)
    cli_ok = (back.track == "skidpad" and back.gearbox == "clutch"
              and o.gearbox == "clutch" and o.auto_gearbox is False
              and o.abs is False and o.wet == "all" and o.steer_limit is True
              and o.engine == "tuned" and o.tc is False and o.sound == "off"
              and back.sound == "off" and back.power_scale == 1.5)
    cyc = Settings(path="")
    seq = []
    for _ in range(len(trk.TRACK_ORDER)):
        cyc.cycle("track")
        seq.append(cyc.track)
    cycle_ok = tuple(seq) == tuple(trk.TRACK_ORDER[1:]) + (trk.TRACK_ORDER[0],)

    # the menu state machine on a headless Sim with a real keyboard input
    from .input import BlendedInput, KeyboardInput
    with contextlib.redirect_stdout(io.StringIO()):
        inp = BlendedInput(KeyboardInput(), None, announce=False, hotplug=False)
    tr = trk.make_arena()
    veh = Vehicle(CorsaC(), VehicleConfig(abs_on=True))
    x, y, psi = trk.start_pose(tr)
    veh.reset(x, y, psi)
    st = Settings(path=path)
    sim = Sim(veh, tr, inp, settings=st)
    sim.renderer = SimpleNamespace(cfg=SimpleNamespace(mode="car_up"))
    sim.has_garage = True
    ev = sim.handle_event
    ev("menu")
    m_open = sim.menu is not None and sim.menu.open and sim.paused and inp.menu
    ev("nav_down")                                  # -> Settings
    ev("select")
    page_ok = sim._menu_page == "settings" and sim.menu.open
    ev("nav_down")                                  # -> Engine
    ev("select")                                    # sport -> stock (wraps)
    eng_ok = (st.engine == ENGINE_MODES[(ENGINE_MODES.index(ENGINE_DEFAULT) + 1)
                                         % len(ENGINE_MODES)]
              and veh.cfg.power_scale == ENGINE_SCALE[st.engine]
              and abs(veh.pt_p.T_clutch_cap - 200.0 * veh.cfg.power_scale) < 1e-9
              and sim.menu.idx == 1)
    ev("nav_down")                                  # -> Gearbox
    ev("select")                                    # auto -> manual
    c = inp.update(DT_PHYS, 0.0)
    gb_ok = (st.gearbox == "manual" and sim.gearbox == "manual"
             and (c.auto_gearbox, c.auto_clutch) == (False, True)
             and sim.menu.open and sim._menu_page == "settings" and sim.menu.idx == 2)
    ev("select")                                    # manual -> clutch: neutral
    neut_ok = (st.gearbox == "clutch" and veh.gear == 0 and veh.pt_s.gear == 0)
    for _ in range(300):
        veh.step(inp.update(DT_PHYS, 0.0), (1.0,) * 4, (1.0,) * 4, DT_PHYS)
    neut_ok = neut_ok and not veh.stalled and veh.rpm > 600.0
    ev("select")                                    # clutch -> auto
    ev("select")                                    # auto -> manual
    gb_ok = gb_ok and neut_ok and st.gearbox == "manual"
    ev("nav_down")                                  # -> ABS
    ev("select")                                    # on -> off
    abs_ok = (st.abs is False and veh.cfg.abs_on is False and sim.menu.idx == 3)
    ev("nav_down")                                  # -> TC
    ev("select")                                    # on -> off
    tc_ok = (st.tc is False and veh.cfg.tc_on is False and sim.menu.idx == 4)
    ev("select")                                    # back on
    tc_ok = tc_ok and st.tc is True and veh.cfg.tc_on is True
    ev("nav_down")                                  # -> Steering aid
    ev("select")
    aid_ok = (st.steer_aid is False and inp.kb.steer_limit is False)
    ev("select")                                    # back on again
    ev("nav_down"); ev("nav_down")                  # -> Camera
    ev("select")
    cam_ok = st.camera == "chase" and sim.renderer.cfg.mode == "chase"
    ev("nav_down")                                  # -> Sound
    ev("select")                                    # mid -> high (no window: no CarSound)
    snd_ok = (st.sound == SOUND_MODES[(SOUND_MODES.index(SOUND_DEFAULT) + 1)
                                       % len(SOUND_MODES)]
              and sim.audio is None and sim.menu.idx == 8)
    saved = Settings.load(path)
    saved_ok = (saved.gearbox, saved.abs, saved.steer_aid, saved.camera,
                saved.engine, saved.tc, saved.sound) == (
        "manual", False, True, "chase", st.engine, True, st.sound)
    ev("menu")                                      # ESC on settings -> main page
    back_ok = sim._menu_page == "main" and sim.menu.open
    ev("menu")                                      # ESC on main -> closed, running
    closed_ok = (not sim.menu.open and not sim.paused and not inp.menu)
    ev("menu"); ev("nav_down"); ev("select")        # settings again
    ev("select")                                    # Map: arena -> open: restart
    map_ok = (sim.quit and sim.stop_reason == "restart" and st.track == "open"
              and not sim.menu.open and Settings.load(path).track == "open")
    sim.quit, sim.stop_reason = False, ""
    ev("track_next")                                # TAB: open -> skidpad
    tab_ok = sim.stop_reason == "restart" and st.track == "skidpad"
    sim.quit, sim.stop_reason = False, ""
    ev("garage")
    gar_ok = sim.stop_reason == "garage" and sim.quit
    sim.quit, sim.stop_reason = False, ""
    ev("menu"); ev("nav_down"); ev("select")        # settings
    for _ in range(9):
        ev("nav_down")                              # -> Garage entry
    ev("select")
    gar2_ok = sim.stop_reason == "garage" and sim.quit and not sim.menu.open
    ok = all((rt_ok, clamp_ok, cli_ok, cycle_ok, m_open, page_ok, eng_ok, gb_ok,
              abs_ok, tc_ok, aid_ok, cam_ok, snd_ok, saved_ok, back_ok, closed_ok,
              map_ok, tab_ok, gar_ok, gar2_ok))
    if verbose:
        print(f"  V26 settings    : round-trip {rt_ok}, clamp {clamp_ok}, cli {cli_ok}, "
              f"cycle {cycle_ok}; menu open {m_open}, settings page {page_ok}, "
              f"engine {eng_ok}, gearbox {gb_ok}, abs {abs_ok}, tc {tc_ok}, aid {aid_ok}, "
              f"camera {cam_ok}, sound {snd_ok}, saved {saved_ok}, back {back_ok}, "
              f"closed {closed_ok}, map restart {map_ok}, TAB {tab_ok}, "
              f"garage {gar_ok}/{gar2_ok}")
    return ok, dict(round_trip=rt_ok, cli=cli_ok,
                    menu=m_open and page_ok and eng_ok and gb_ok and tc_ok and snd_ok,
                    map_restart=map_ok, garage=gar_ok and gar2_ok)


def _v27_gearbox_modes(verbose=True):
    """The three driver models on the real car: auto never stalls, manual
    (paddles + auto clutch) gets to 100 km/h on the driver's own shifts and
    never stalls, clutch mode stalls on a dumped clutch and restarts with
    the pedal fully in, then launches properly on a slipped clutch."""
    mu1 = (1.0,) * 4
    dt = DT_PHYS

    # auto: WOT from rest with the handbrake on for 2 s, then released
    veh = Vehicle(CorsaC(), VehicleConfig())
    veh.reset(gear=1)
    stalled = False
    for k in range(int(10.0 / dt)):
        hb = 1.0 if k * dt < 2.0 else 0.0
        veh.step(Controls(throttle=1.0, handbrake=hb), mu1, mu1, dt)
        stalled = stalled or veh.stalled
    auto_ok = (not stalled) and veh.u > 15.0 and veh.gear >= 2
    auto_res = (veh.u, veh.gear, stalled)

    # manual: the driver shifts at 5800 rpm, the box works the clutch
    veh = Vehicle(CorsaC(), VehicleConfig())
    veh.reset(gear=1)
    t100 = float("nan")
    last_shift = -9.0
    stalled = False
    for k in range(int(20.0 / dt)):
        t = k * dt
        req = 0
        if veh.rpm > 5800.0 and veh.gear < 5 and t - last_shift > 1.0 and veh.gear > 0:
            req, last_shift = +1, t
        veh.step(Controls(throttle=1.0, gear_req=req, auto_gearbox=False,
                          auto_clutch=True), mu1, mu1, dt)
        stalled = stalled or veh.stalled
        if math.isnan(t100) and veh.u >= 27.7778:
            t100 = t
    man_ok = (not stalled) and 13.5 <= t100 <= 18.0 and veh.gear >= 3
    man_res = (t100, veh.gear)

    # clutch: dump it at idle with no throttle -> stall; clutch in -> fires;
    # then a real launch: 40% throttle, clutch released over 1.5 s
    veh = Vehicle(CorsaC(), VehicleConfig())
    veh.reset(gear=1)
    t_stall = float("nan")
    for k in range(int(2.0 / dt)):
        veh.step(Controls(auto_gearbox=False, auto_clutch=False), mu1, mu1, dt)
        if veh.stalled and math.isnan(t_stall):
            t_stall = k * dt
    stalled_ok = not math.isnan(t_stall) and t_stall < 1.0
    t_fire = float("nan")
    for k in range(int(2.0 / dt)):
        veh.step(Controls(clutch=1.0, auto_gearbox=False, auto_clutch=False), mu1, mu1, dt)
        if not veh.stalled and math.isnan(t_fire):
            t_fire = k * dt
    fire_ok = not math.isnan(t_fire) and t_fire < 1.0 and 600.0 < veh.rpm < 1100.0
    stalled = False
    for k in range(int(6.0 / dt)):
        t = k * dt
        clu = max(0.0, 1.0 - t / 1.5)
        veh.step(Controls(throttle=0.4, clutch=clu, auto_gearbox=False,
                          auto_clutch=False), mu1, mu1, dt)
        stalled = stalled or veh.stalled
    launch_ok = (not stalled) and veh.u > 5.0 and veh.gear == 1
    clu_res = (t_stall, t_fire, veh.u)

    ok = auto_ok and man_ok and stalled_ok and fire_ok and launch_ok
    if verbose:
        print(f"  V27 gearbox     : auto {auto_res[0]:.1f} m/s in gear {auto_res[1]}, "
              f"stalled {auto_res[2]} -> {auto_ok}; manual 0-100 {man_res[0]:.2f} s "
              f"gear {man_res[1]} -> {man_ok}; clutch stalls at {clu_res[0]:.2f} s, "
              f"fires {clu_res[1]:.2f} s after clutch-in, launches to "
              f"{clu_res[2]:.1f} m/s -> {stalled_ok and fire_ok and launch_ok}")
    return ok, dict(auto=auto_res, manual=man_res, clutch=clu_res)


def _v29_engine_tc(verbose=True):
    """The Engine setting and the TC on the interactive path (keyboard
    ramps, auto box, ABS on, the open map's pad), full throttle from rest:
    the stock car lands 0-100 where the scripted run does and the TC never
    touches it; the 2x car spins its fronts to the limiter without TC and
    gets to 100 km/h in under 10 s with it, slip held near the drive peak.
    The settings page swaps the parameter set live."""
    import contextlib, io
    from . import powertrain as ptm
    from .input import BlendedInput, KeyboardInput, K_US_DEG_MEASURED

    def wot(scale, tc, T):
        tr = trk.make_track("open", 50.0, False, surfaces=True)
        veh = Vehicle(CorsaC(), VehicleConfig(abs_on=True, tc_on=tc, power_scale=scale))
        x, y, psi = trk.start_pose(tr, 0.0)
        veh.reset(x, y, psi, V=0.0, gear=1)
        with contextlib.redirect_stdout(io.StringIO()):
            kb = KeyboardInput(steer_limit=True, k_us_deg=K_US_DEG_MEASURED)
            inp = BlendedInput(kb, None, announce=False, hotplug=False)
        sim = Sim(veh, tr, inp, settings=Settings(path=""))
        sim.set_gearbox("auto")
        kb.set_keys(up=True)
        t100, kmax, tc_t = float("nan"), 0.0, 0.0
        for _ in range(int(T / DT_PHYS)):
            sim.step_physics(DT_PHYS)
            k = max(float(veh.kappa[0]), float(veh.kappa[1]))
            if k > kmax:
                kmax = k
            if veh.tc_active:
                tc_t += DT_PHYS
            if math.isnan(t100) and hypot(veh.u, veh.v) >= 27.7778:
                t100 = sim.t
        return t100, kmax, tc_t

    stock = wot(1.0, True, 16.5)
    stock_ok = 14.3 <= stock[0] <= 15.8 and stock[2] == 0.0 and stock[1] < 0.12
    raw = wot(2.0, False, 4.0)
    raw_ok = raw[1] > 0.8
    tc = wot(2.0, True, 11.0)
    tc_ok = tc[0] < 10.0 and tc[1] < 0.5 and tc[2] > 0.3

    # the settings page swaps the parameter set live
    veh = Vehicle(CorsaC(), VehicleConfig())
    sim = Sim(veh, trk.make_arena(), ScriptedInput(lambda t, v, T: Controls()),
              settings=Settings(path=""))
    sim.settings.engine, sim.settings.tc = "stock", False
    p0 = ptm.PowertrainParams.from_car(CorsaC())
    sim.apply_setting("engine")                 # stock -> tuned
    p = veh.pt_p
    live_ok = (sim.settings.engine == "tuned" and veh.cfg.power_scale == 1.5
               and abs(p.T_clutch_cap - 300.0) < 1e-9
               and abs(ptm.wot_torque(p, 4000.0) / ptm.wot_torque(p0, 4000.0) - 1.5) < 1e-9
               and ptm.overrun_torque(p, 4000.0) == ptm.overrun_torque(p0, 4000.0)
               and p.n_cut == p0.n_cut and p.gear == p0.gear)
    sim.apply_setting("tc")
    live_ok = live_ok and veh.cfg.tc_on is True and sim.settings.tc is True

    ok = stock_ok and raw_ok and tc_ok and live_ok
    if verbose:
        print(f"  V29 engine/TC   : stock 0-100 {stock[0]:.2f} s, kappa {stock[1]:.3f}, "
              f"TC {stock[2]:.1f} s -> {stock_ok}; 2x no TC kappa {raw[1]:.2f} -> {raw_ok}; "
              f"2x TC 0-100 {tc[0]:.2f} s, kappa {tc[1]:.2f}, TC active {tc[2]:.1f} s "
              f"-> {tc_ok}; live swap {live_ok}")
    return ok, dict(stock=stock, raw=raw, tc=tc, live=live_ok)


def _v28_open_map(verbose=True):
    """The open map through the harness: a scripted lap of the perimeter
    road completes and times; a drive across the middle of the pad stays
    'on track', fires no lap or sector event, and the projection never
    throws. Plus the surfaces, per wheel."""
    # 17 m/s: the R = 45 m corners limit at 19.5 m/s, and the follower must
    # not be asked to drive at the limit. Staged 10 m before the line so the
    # first crossing is the 'start' and the lap is timed inside the window.
    L = trk.make_open().length
    drv = PathFollower(17.0)
    sim = _build("open", driver=drv, start_V=17.0, gear=3, tag="open",
                 start_s=L - 10.0)
    sim.run_headless(110.0)
    laps = [e for e in sim.events_log if e[0] == "lap"]
    lap_ok = len(laps) >= 1 and drv.max_n < 4.0 and 90.0 <= laps[0][3] <= 110.0

    # across the pad: from north of the skidpad, due east, 12 s at ~15 m/s.
    # y = 200 clears the wet square (y 95..175) and the pad edge (y 317).
    sim2 = _build("open", driver=lambda t, v, T: Controls(throttle=0.30, wing_on=False),
                  start_V=15.0, gear=3)
    sim2.veh.reset(105.0, 200.0, 0.0, V=15.0, gear=3)
    sim2._sample_surfaces()
    sim2.s, sim2._s_prev = None, None
    on = True
    mus = set()
    for _ in range(int(12.0 / sim2.dt)):
        sim2.step_physics(sim2.dt)
        on = on and sim2.on_track and all(sim2.on_track4)
        mus.add(round(sim2.mu[0], 3))
    # |dy| < 8 m: with delta pinned at 0 the MF6.2 PEY3 asymmetry yaws the
    # free-steered car ~2 deg over 12 s (StraightDriver's docstring); a lane
    # keeper would hide exactly the thing this rig is not about.
    cross_ok = (on and not sim2.events_log and 250.0 < sim2.veh.x < 345.0
                and abs(sim2.veh.y - 200.0) < 8.0 and mus == {1.0})
    # the wet square is felt by the wheels that are in it
    sim3 = _build("open", driver=lambda t, v, T: Controls(), start_V=0.0)
    sim3.veh.reset(300.0, 130.0, 0.0, V=0.0, gear=1)
    sim3._sample_surfaces()
    wet_ok = all(abs(m - MU_WET_SCALE) < 1e-9 for m in sim3.mu) and all(sim3.on_track4)
    sim3.veh.reset(210.0, -80.0, 0.0, V=0.0, gear=1)
    sim3._sample_surfaces()
    grass_ok = all(abs(m - trk.MU_OFF_TRACK) < 1e-9 for m in sim3.mu) and not any(sim3.on_track4)
    ok = lap_ok and cross_ok and wet_ok and grass_ok
    if verbose:
        print(f"  V28 open map    : {len(laps)} lap(s) of the perimeter, first "
              f"{laps[0][3] if laps else float('nan'):.2f} s, max |n| {drv.max_n:.2f} m; "
              f"pad crossing on-track {cross_ok} (x {sim2.veh.x:.0f} m, "
              f"{len(sim2.events_log)} events); wet square {wet_ok}; grass {grass_ok}")
    return ok, dict(laps=len(laps), lap_s=laps[0][3] if laps else float("nan"),
                    max_n=drv.max_n, cross=cross_ok, wet=wet_ok, grass=grass_ok)


def self_check(verbose=True) -> bool:
    """python3 -m drive.drive  ->  the harness acceptance numbers."""
    tmp = _tmpdir()
    if verbose:
        print("=" * 74)
        print("drive.py self-check   (headless; no wall clock in any physics path)")
        print("=" * 74)
    results = {}
    ok = True
    for name, fn in (("V24", lambda: _v24_first_frame(verbose)),
                     ("V23", lambda: _v23_accumulator(verbose)),
                     ("V25", lambda: _v25_lap_timing(verbose)),
                     ("V26", lambda: _v26_settings_and_menu(tmp, verbose)),
                     ("V27", lambda: _v27_gearbox_modes(verbose)),
                     ("V28", lambda: _v28_open_map(verbose)),
                     ("V29", lambda: _v29_engine_tc(verbose)),
                     ("V20", lambda: _v20_determinism(tmp, verbose)),
                     ("accel", lambda: _accel_end_to_end(tmp, verbose)),
                     ("V21", lambda: _v21_rtf(tmp, 60.0, verbose))):
        good, data = fn()
        results[name] = data
        ok = ok and good
        if not good and verbose:
            print(f"    ^^ {name} FAILED")
    if verbose:
        print("-" * 74)
        print(f"  {'ALL PASS' if ok else 'FAILURES ABOVE'}   artefacts in {tmp}")
    return ok


# ==================================================================== #
#  CLI                                                                 #
# ==================================================================== #
KEYS_HELP = """\
ARROW UP throttle | ARROW DOWN brake | ARROW LEFT/RIGHT steer | LSHIFT fine
Z clutch | SPACE handbrake | S starter | E shift up | Q shift down
F flank-wing toggle | G cycle wing side (auto / left / right)
R reset to last sector line | SHIFT+R full reset
P pause | O single physics step | [ ] slow-mo 0.25x / 1.0x
C camera | - / = zoom | 0 auto zoom | H HUD | V vectors | B g-g | N skid | X clear
T toggle wet | M telemetry marker | L toggle recording | TAB next map
BACKSPACE garage (3D panel editor) | ESC menu: settings (map, engine, gearbox, ABS, TC, aids, sound), reset, quit
PS5 pad: R2 throttle | L2 brake | L-stick steer | R1/L1 shift | CROSS handbrake | SQUARE clutch
         CIRCLE wing | TRIANGLE wing side | OPTIONS menu | CREATE reset | TOUCHPAD garage
         d-pad up HUD / down vectors / left slow-mo / right normal | R3 camera | L3 auto zoom"""
_HELP_PRINTED = False


def build_parser():
    p = argparse.ArgumentParser(
        prog="python3 -m drive.drive",
        description="Corsa C flank-wing driving simulator.",
        epilog=KEYS_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # track / wet / camera / gearbox / abs / steer aid default to None: the
    # interactive session fills them from runs/settings.json (an explicit
    # flag wins and is saved); scripts fill them with the fixed defaults.
    p.add_argument("--track", default=None,
                   choices=sorted(trk.TRACKS.keys()))
    p.add_argument("--radius", type=float, default=50.0, help="skidpad radius, m")
    p.add_argument("--cw", action="store_true", help="skidpad clockwise")
    p.add_argument("--wet", default=None, choices=("none", "patch", "all"))
    p.add_argument("--wing", default="off", choices=("off", "fin", "plate"))
    p.add_argument("--wing-x", dest="wing_x", type=float, default=0.97)
    p.add_argument("--wing-h", dest="wing_h", type=float, default=0.90)
    p.add_argument("--wing-inc", dest="wing_inc", type=float, default=0.0,
                   help="panel built-in incidence, deg (VehicleConfig.delta_dev_geom)")
    p.add_argument("--build", default=None,
                   help="drive a car saved in the garage library (runs/library/builds/NAME.json)")
    p.add_argument("--garage", action="store_true",
                   help="open the 3D editor first; ENTER / cross drives the "
                        "car you built, BACKSPACE / touchpad comes back")
    p.add_argument("--tyre", default=tlm.DEFAULT_TYRE_FILE)
    p.add_argument("--dt", type=float, default=DT_PHYS)
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--size", default="1280x800")
    p.add_argument("--camera", default=None,
                   choices=("car_up", "world_up", "chase"))
    p.add_argument("--engine", default=None, choices=ENGINE_MODES,
                   help="stock 75 hp (every script) | tuned 1.5x | sport 2x "
                        "(the interactive default; runs/settings.json)")
    p.add_argument("--tc", dest="tc", action="store_true", default=None,
                   help="engine-only traction control (interactive default on)")
    p.add_argument("--no-tc", dest="tc", action="store_false")
    p.add_argument("--sound", default=None, choices=SOUND_MODES,
                   help="engine / tyre / wind sound level (interactive default mid)")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--render", default=None,
                   choices=("off", "offscreen", "window"))
    p.add_argument("--script", default=None,
                   choices=sorted(SCRIPTS.keys()))
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--telemetry", default=None,
                   help="CSV path; '-' or 'auto' picks runs/<track>_<stamp>.csv")
    p.add_argument("--telem-hz", dest="telem_hz", type=int, default=TELEM_HZ)
    p.add_argument("--telem-precision", dest="telem_precision",
                   default="6g", choices=("6g", "full"))
    p.add_argument("--no-steer-limit", dest="steer_limit",
                   action="store_false", default=None)
    p.add_argument("--gearbox", default=None, choices=GEARBOX_MODES,
                   help="auto (default) | manual (paddles, auto clutch) | "
                        "clutch (H-pattern: Z / SQUARE is the only clutch)")
    p.add_argument("--manual", dest="gearbox", action="store_const", const="manual")
    p.add_argument("--auto-gearbox", dest="gearbox", action="store_const", const="auto")
    p.add_argument("--abs", dest="abs", action="store_true", default=None)
    p.add_argument("--no-abs", dest="abs", action="store_false")
    p.add_argument("--margin", type=float, default=0.90,
                   help="virtual driver's fraction of the grip envelope")
    p.add_argument("--pad-calib", action="store_true")
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--interactive", action="store_true",
                   help="force the window even with no other option given")
    return p


def _apply_headless(opts):
    """SDL env vars BEFORE pygame is imported anywhere in the package."""
    if opts.headless or opts.script:
        os.environ["CARSIM_HEADLESS"] = "1"
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"


def _pad_calib() -> int:
    import pygame
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("no gamepad found - keyboard only")
        return 0
    j = pygame.joystick.Joystick(0)
    j.init()
    from . import input as inp_mod
    pad = inp_mod.GamepadInput(joystick=j)
    print(f"{j.get_name()}: {j.get_numaxes()} axes, {j.get_numbuttons()} buttons, "
          f"{j.get_numhats()} hats  ->  layout '{pad.layout}'")
    print(f"axes: steer {pad.map.get('steer')}  throttle {pad.map.get('throttle')}  "
          f"brake {pad.map.get('brake')}   (triggers must REST at -1.0)")
    print("buttons: " + "  ".join(f"{i}={pad.button_name(i)}:{c}"
                                  for i, c in sorted(pad.map['buttons'].items())))
    print("move every stick and press every button; 15 s. Wrong? write "
          f"{inp_mod.PAD_CONFIG_PATH} with the corrected indices.")
    clock = pygame.time.Clock()
    for _ in range(150):
        pygame.event.pump()
        ax = [round(j.get_axis(i), 2) for i in range(j.get_numaxes())]
        bt = [i for i in range(j.get_numbuttons()) if j.get_button(i)]
        hats = [j.get_hat(i) for i in range(j.get_numhats())]
        named = [pad.button_name(i) for i in bt]
        print(f"axes {ax}  pressed {bt} {named}  hats {hats}")
        clock.tick(10)
    return 0


def main(argv=None) -> int:
    # No arguments drives. Every other module in this package runs its own
    # self-check when invoked bare, and this one used to as well for symmetry --
    # but drive.py is the entry point, and an entry point called `drive` that
    # does not drive is a wart, not a convention. The self-check is one flag
    # away, and drive/validate.py invokes it explicitly.
    argv = list(sys.argv[1:] if argv is None else argv)
    opts = build_parser().parse_args(argv)
    _apply_headless(opts)
    interactive = not (opts.self_check or opts.pad_calib or opts.script
                       or opts.headless)
    if not interactive:
        # scripts and headless runs: fixed defaults, never the settings file
        opts.track = opts.track or "arena"
        opts.wet = opts.wet or "patch"
        opts.camera = opts.camera or "car_up"
        opts.gearbox = opts.gearbox or "auto"
        opts.auto_gearbox = (opts.gearbox == "auto")
        opts.abs = bool(opts.abs) if opts.abs is not None else False
        opts.tc = bool(opts.tc) if opts.tc is not None else False
        opts.engine = opts.engine or "stock"
        opts.sound = "off"
        opts.steer_limit = True if opts.steer_limit is None else bool(opts.steer_limit)

    if opts.self_check:
        return 0 if self_check() else 1
    if opts.pad_calib:
        return _pad_calib()

    if opts.telemetry in ("-", "auto") or (opts.script and not opts.telemetry):
        # A scripted run always writes a CSV: the run IS the deliverable, and a
        # script that prints a number without leaving the trace behind cannot be
        # replotted or diffed.
        tname = {"accel": "dragstrip", "brake": "dragstrip",
                 "ramp_probe": "dragstrip", "skidpad_limit": "skidpad",
                 "wing_ab": "skidpad"}.get(opts.script, opts.track)
        opts.telemetry = tlm.run_path(tname, tag=opts.script or "")

    # ---- scripted (headless) --------------------------------------
    if opts.script:
        res = SCRIPTS[opts.script](opts)
        print(f"--- script {opts.script} ---")
        for k, v in res.items():
            if isinstance(v, float):
                print(f"  {k:22s} {v:.6g}")
            else:
                print(f"  {k:22s} {v}")
        return 0

    if opts.headless and opts.render != "offscreen":
        # Headless with no script and no offscreen renderer: run the driver
        # profile for --duration so the mode is still useful, not a no-op.
        drv = (LapDriver(trk.TRACKS[opts.track](), margin=opts.margin,
                         wing_on=(opts.wing != "off"))
               if opts.track == "arena" else PathFollower(20.0,
                                                          wing_on=opts.wing != "off"))
        sim = _build(opts.track, radius=opts.radius, cw=opts.cw, wing=opts.wing,
                     x_w=opts.wing_x, h_w=opts.wing_h, wet=opts.wet, dt=opts.dt,
                     telem_path=opts.telemetry, telem_hz=opts.telem_hz,
                     precision=opts.telem_precision, driver=drv,
                     start_V=20.0 if opts.track != "dragstrip" else 0.0, gear=3)
        sim.run_headless(opts.duration)
        if sim.telem:
            sim.telem.close()
        print(f"headless {opts.duration:g} s on {opts.track}: "
              f"t={sim.t:.3f} s, laps={sim.lap.lap}, "
              f"best={sim.lap.best_lap:.3f} s, csv={opts.telemetry}")
        return 0

    # ---- interactive ----------------------------------------------
    return run_interactive_cli(opts)


def _apply_design(opts, design, lib=None) -> dict:
    """The garage's car -> VehicleConfig fields on opts.

    A `CarBuild` (three slots) puts the full kwargs on `opts.wing_cfg` and
    the renderer's geometry on `opts.hud_cfg`; the four legacy fields are
    filled too so the caption and the scripts still read them. A
    `WingDesign` (the published one-panel car) is accepted for old callers.
    """
    from . import garage as grg
    if isinstance(design, grg.WingDesign):
        kw = design.cfg_kwargs()
        opts.wing, opts.wing_x, opts.wing_h = kw["wing"], kw["x_w"], kw["h_w"]
        opts.wing_inc = design.inc_deg
        opts.wing_cfg, opts.hud_cfg = None, None
        return kw
    lib = lib or grg.library()
    kw = design.cfg_kwargs(lib)
    opts.wing, opts.wing_x, opts.wing_h = kw["wing"], kw["x_w"], kw["h_w"]
    opts.wing_inc = degrees(kw["delta_dev_geom"])
    opts.wing_cfg = kw
    opts.hud_cfg = design.hud_kwargs(lib)
    return kw


def run_garage_cli(opts) -> int:
    """Start in the garage. Kept for callers of the old name; the loop is
    run_interactive_cli's."""
    opts.garage = True
    return run_interactive_cli(opts)


def run_interactive_cli(opts) -> int:
    """The window: garage <-> drive <-> drive(restart) ... in ONE pygame
    session, until Quit (menu or window close).

    State machine. `mode` is 'garage' or 'drive':
      garage  -> the editor; ENTER / cross -> save the design, apply it to
                 opts, mode = drive. Quit / window close -> exit.
      drive   -> a session. stop_reason 'garage'  -> mode = garage (BACKSPACE,
                 touchpad, or the menu / settings entry); 'restart' -> a new
                 session (the settings page changed the map or the surface,
                 or TAB); anything else -> exit.
    The pad object survives every transition (hot-plugged or not) and its
    edge state is re-seeded at each boundary so the button that ended one
    session cannot act in the next. Settings and the garage design are both
    loaded here and saved on every change, so the next launch starts where
    this one stopped; explicit command-line flags override the files.
    """
    try:
        import pygame
    except Exception as exc:
        print(f"pygame unavailable ({exc}); running the self-check instead")
        return 0 if self_check() else 1
    pygame.init()

    settings = Settings.load().apply_cli(opts)
    settings.save()
    w, h = (int(v) for v in opts.size.lower().split("x"))

    grg = None
    design = None
    lib = None
    try:
        from . import garage as grg
        lib = grg.library()
        explicit = any(a.startswith(("--wing", "--wing-x", "--wing-h", "--wing-inc"))
                       for a in sys.argv[1:])
        want = getattr(opts, "build", None)
        if want:
            if want in lib.builds:
                design = grg.CarBuild.from_json(lib.builds[want])
            else:
                print(f"build {want!r} is not in the library "
                      f"({', '.join(sorted(lib.builds)) or 'empty'}); using the last garage car")
        if design is None and not explicit:
            design = grg.CarBuild.load()      # the last car built is the car
        if design is None:
            design = grg.CarBuild.from_json(dict(wing=opts.wing, x_w=opts.wing_x, h_w=opts.wing_h,
                                                 inc_deg=getattr(opts, "wing_inc", 0.0)))
        design.clamp(lib)
        _apply_design(opts, design, lib)
    except Exception as exc:
        print(f"garage unavailable ({exc})")
        grg = None

    mode = "garage" if (opts.garage and grg is not None) else "drive"
    pad = None
    try:
        while True:
            if mode == "garage":
                g = grg.Garage((w, h), design, pad=pad, lib=lib)
                action = g.run()
                design, pad = g.build, g.pad
                if action != "drive":
                    break
                design.save()
                _apply_design(opts, design, lib)
                print(f"garage -> drive: {design.summary(lib)}")
                mode = "drive"
            sim = _interactive_session(opts, pad=pad, settings=settings,
                                       garage=(grg is not None))
            pad = getattr(sim.inp, "pad", pad)
            if sim.stop_reason == "garage" and grg is not None:
                mode = "garage"
                continue
            if sim.stop_reason == "restart":
                continue
            break
    finally:
        pygame.quit()
    return 0


def _interactive_session(opts, pad=None, garage=False, settings=None):
    """One windowed session. Returns the Sim (stop_reason says why it ended).
    pygame is initialised by the caller and NOT quit here, so the garage and
    the settings page can run several sessions in one window."""
    global _HELP_PRINTED

    if settings is None:
        settings = Settings(path="").apply_cli(opts)
    settings.to_opts(opts)                 # the settings are the truth; opts
    tr = trk.make_track(opts.track, opts.radius, opts.cw,   # is the carrier
                        surfaces=(opts.wet != "none"))
    global_wet = MU_WET_SCALE if opts.wet == "all" else 1.0

    aero_kw = dict(wing=opts.wing, x_w=opts.wing_x, h_w=opts.wing_h,
                   delta_dev_geom=radians(getattr(opts, "wing_inc", 0.0)))
    if getattr(opts, "wing_cfg", None):
        aero_kw.update(opts.wing_cfg)      # the garage build's three wings
    cfg = VehicleConfig(**aero_kw,
                        abs_on=bool(settings.abs), tc_on=bool(settings.tc),
                        power_scale=settings.power_scale)
    veh = Vehicle(CorsaC(), cfg)
    x, y, psi = trk.start_pose(tr, 0.0)
    veh.reset(x, y, psi, V=0.0, gear=0 if settings.gearbox == "clutch" else 1)

    # input: keyboard + gamepad. The blend is ALWAYS used (pad may be None):
    # it is what hot-plugs a DualSense that is switched on after launch.
    # The limiter runs on the MEASURED understeer gradient (input.py's own
    # calibration finding): with the spec's 3.2 deg/g the aid capped the
    # driver at 67% of the angle the car needs at R = 50 and the car could
    # never reach its own grip limit from the seat.
    inp = None
    try:
        from . import input as inp_mod
        kb = inp_mod.KeyboardInput(steer_limit=settings.steer_aid,
                                   k_us_deg=inp_mod.K_US_DEG_MEASURED)
        if pad is not None:
            pad.steer_limit = kb.steer_limit
            pad.k_us_deg = kb.k_us_deg
            pad.delta_deg = 0.0
            pad.seed_edges()               # the button that ended the last
            pad.set_menu(False)            # session is not a press in this one
        else:
            try:
                if inp_mod.GamepadInput.available():
                    pad = inp_mod.GamepadInput(0, steer_limit=settings.steer_aid,
                                               k_us_deg=kb.k_us_deg)
            except Exception as exc:
                print(f"gamepad found but not usable ({exc}) - keyboard only")
                pad = None
        inp = inp_mod.BlendedInput(kb, pad, announce=not _HELP_PRINTED)
    except Exception as exc:
        print(f"drive.input unavailable ({exc}); coasting with a null driver")
        inp = ScriptedInput(lambda t, v, T: Controls(auto_gearbox=opts.auto_gearbox))

    renderer = None
    if opts.render != "off":
        try:
            from . import render as rnd
            w, h = (int(v) for v in opts.size.lower().split("x"))
            cfgv = rnd.ViewConfig(size=(w, h), fps=opts.fps, mode=settings.camera)
            renderer = rnd.Renderer(cfgv, tr,
                                    headless=(opts.render == "offscreen"
                                              or opts.headless))
        except Exception as exc:
            print(f"drive.render unavailable ({exc}); running without a window")

    telem = None
    if opts.telemetry:
        telem = tlm.TelemetryWriter(
            opts.telemetry, hz=opts.telem_hz, precision=opts.telem_precision,
            meta=dict(dt=opts.dt, track=tr.name, car=veh.car,
                      tyre_file=opts.tyre,
                      wing=dict(wing=opts.wing, x_w=opts.wing_x, h_w=opts.wing_h,
                                inc_deg=getattr(opts, "wing_inc", 0.0)),
                      settings=settings.as_dict(),
                      cmdline=list(sys.argv)))

    sim = Sim(veh, tr, inp, renderer=renderer, telem=telem, dt=opts.dt,
              wing=opts.wing, global_wet=global_wet, settings=settings)
    sim.has_garage = bool(garage)          # the menu offers the garage
    sim.hud_cfg = getattr(opts, "hud_cfg", None)   # the build's wing geometry
    if cfg.has_designed():
        sim.wing_on = True                 # a garage build starts armed, as --wing does
    if not _HELP_PRINTED:
        print(KEYS_HELP)
        _HELP_PRINTED = True
    print(f"session: {tr.title or tr.name} | engine {ENGINE_LABELS[settings.engine]} | "
          f"gearbox {GEARBOX_LABELS[settings.gearbox]} | "
          f"ABS {'on' if settings.abs else 'off'} | TC {'on' if settings.tc else 'off'} | "
          f"steering aid {'on' if settings.steer_aid else 'off'} | "
          f"surface {SURFACE_LABELS[settings.wet]} | sound {SOUND_LABELS[settings.sound]} | "
          f"wing {opts.wing} x_w {opts.wing_x:+.2f} h_w {opts.wing_h:.2f}")
    if renderer is not None and opts.render != "offscreen" and not opts.headless:
        sim.sound_enabled = True
        sim._audio_apply()
    try:
        sim.run_interactive()
    finally:
        if telem is not None:
            telem.close()
    return sim


if __name__ == "__main__":
    sys.exit(main())
